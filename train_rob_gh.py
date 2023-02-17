# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning a 🤗 Transformers model for sequence classification on GLUE."""
import sys; sys.path.append('/work/awilf/utils/'); from alex_utils import *
import wandb
import argparse
import json
import logging
import math
import os
import random
from pathlib import Path

from torch import nn
import datasets
import evaluate
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    RobertaForSequenceClassification
)
from transformers.utils import check_min_version, get_full_repo_name, send_example_telemetry
from transformers.utils.versions import require_version
# from modeling_roberta import RobertaForSequenceClassification

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.27.0.dev0")

logger = get_logger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    
    ## GH specific params
    parser.add_argument(
        "--_tags",
        type=str,
        default='debug',
    )
    parser.add_argument(
        "--N",
        type=float,
        default=1e4,
        help="Number of datapoints to generate",
    )
    parser.add_argument(
        "--hinter_dim",
        type=int,
        default=100,
        help="",
    )
    parser.add_argument(
        "--gen_data_only",
        type=int,
        default=0,
        help="if you only want to generate the data, not run training",
    )
    parser.add_argument(
        "--hinter_drop",
        type=float,
        default=.2,
        help="",
    )
    parser.add_argument(
        "--guesser_drop",
        type=float,
        default=.2,
        help="",
    )
    parser.add_argument(
        "--np",
        type=int,
        default=-1,
        help="Set > 1 if you want n and p to be the same. Overrides n and p",
    )
    parser.add_argument(
        "--freeze_enc",
        type=int,
        default=0,
        help="Freeze encoder",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="Number of elements in the negative set",
    )
    parser.add_argument(
        "--p",
        type=int,
        default=2,
        help="Number of elements in the positive set",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=100,
        help="Number of datapoints to generate",
    )
    parser.add_argument(
        "--clm_max_train_steps",
        type=int,
        default=500,
        help="Number of datapoints to generate",
    )
    parser.add_argument(
        "--clm_freeze_enc",
        type=int,
        default=1,
        help="Freeze encoder during clm",
    )
    ## end GH specific

    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )

    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )
    args = parser.parse_args()
    
    args.N = int(args.N)
    if args.np > 1:
        args.n = args.np
        args.p = args.np

    return args


def main():
    args = parse_args()
    device = 'cuda'
    
    if args.seed is not None:
        set_seed(args.seed)

    if 'debug' not in args._tags:
        wandb.init(
            project='siqa',
            entity='socialiq',
            config=vars(args),
            tags=args._tags.split(','),
        )

    ## LOAD DATASET
    ds = load_jsonl(args.dataset_name)
    ds = arlmap(lambda elt: elt['context'], ds)
    new_ds = []

    new_ds_name_base = f'{args.dataset_name}.bak.{args.N}.{args.n}.{args.p}'
    new_ds_name_train = f'{new_ds_name_base}.train'
    new_ds_name_valid = f'{new_ds_name_base}.valid'
    
    if True:
    # if not exists(new_ds_name_train) or not exists(new_ds_name_valid):
        for _ in tqdm(range(args.N)):
            # choose n positives, p negatives without replacement
            group = np.random.choice(ds, size=args.n+args.p, replace=False)
            pos = group[:args.p]
            neg = group[args.p:]

            new_ds.append({
                'pos': lmap(lambda elt: str(elt), pos),
                'neg': lmap(lambda elt: str(elt), neg),
            })
            # new_ds.append([prompt, *lmap(lambda elt: str(elt), pos), *lmap(lambda elt: str(elt), neg)])
        train_ds, valid_ds = train_test_split(new_ds, test_size=.2)
        save_jsonl(new_ds_name_train, train_ds)
        save_jsonl(new_ds_name_valid, valid_ds)
    
    if args.gen_data_only:
        exit()
    ds = load_dataset('json', data_files={'train': new_ds_name_train, 'valid': new_ds_name_valid})
    ##

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=1)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    args.hidden_size = config.hidden_size
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        config=config,
        ignore_mismatched_sizes=args.ignore_mismatched_sizes,
    )
    guesser = nn.Sequential(
        nn.Linear(config.hidden_size+args.hinter_dim, 1, bias=True),
        nn.Dropout(args.guesser_drop),
        nn.Sigmoid(),
    )
    hinter = nn.Sequential(
        nn.Linear(config.hidden_size*(args.p+args.n), args.hinter_dim, bias=True),
        nn.Dropout(args.hinter_drop)
    )

    model = model.to(device)
    guesser = guesser.to(device)
    hinter = hinter.to(device)

    # Preprocessing the datasets
    padding = "max_length"
    def get_tokens(list_of_lists, name, num_elts): # list of list of strings, each sublist is pos or neg list
        arr = [elt2 for elt in list_of_lists for elt2 in elt] # flatten
        tokens = tokenizer(arr, padding=padding, max_length=args.max_length, truncation=False)

        # reshape to original number of rows
        return {
            f'{name}_{key}': ar(tokens[key]).reshape(len(list_of_lists), num_elts, -1) 
            for key in tokens.keys()
        }

    def preprocess_function(examples):
        # Tokenize the texts
        pos = get_tokens(examples['pos'], 'pos', args.p)
        neg = get_tokens(examples['neg'], 'neg', args.n)
        result = {
            **pos,
            **neg
        }
        return result

    processed_datasets = ds.map(
        preprocess_function,
        batched=True,
        desc="Running tokenizer on dataset",
        remove_columns=['pos', 'neg'],
    )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["valid"]

    data_collator = default_data_collator

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in guesser.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in hinter.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Train!
    progress_bar = tqdm(range(args.max_train_steps))
    completed_steps = 0
    starting_epoch = 0

    def model_forward(model, hinter,  guesser, batch, loss_fn):
        ## get encodings of pos, neg, and hints

        # reformulate batch to be how model expects it
        this_bs, p, seq_len = batch['pos_input_ids'].shape
        assert args.p==p and this_bs==batch['neg_input_ids'].shape[0]

        new_batch = { 'input_ids': None, 'attention_mask': None }
        for k in new_batch.keys():
            for split in ['pos', 'neg']:
                if new_batch[k] is None: # pos
                    new_batch[k] = batch[f'{split}_{k}'].reshape(-1, seq_len) # reshape so each seq processed separately
                else:
                    to_add = batch[f'{split}_{k}'].reshape(-1, seq_len)
                    new_batch[k] = torch.cat([new_batch[k], to_add], dim=0) # pos then neg. total shape: (args.p+args.n)*bs, seq_len
        
        batch = new_batch
        batch = {k:v.type(torch.long).to(device) for k,v in batch.items()}
        batch['output_hidden_states'] = True
        outputs = model(**batch)
        cls_tokens = outputs['hidden_states'][-1][:,0] # bs, hidden_dim

        with torch.no_grad():
            labels = torch.cat([torch.ones(int(this_bs*args.p)), torch.zeros(int(this_bs*args.n))])
            labels = labels.to(device)

        # get hints
        boundary = args.p*this_bs
        pos_cls, neg_cls = cls_tokens[:boundary], cls_tokens[boundary:]
        hinter_pos_cls = pos_cls.reshape(this_bs,args.p*args.hidden_size)
        hinter_neg_cls = neg_cls.reshape(this_bs,args.n*args.hidden_size)
        hinter_input = torch.cat([hinter_pos_cls, hinter_neg_cls], -1)
        hints = hinter(hinter_input)

        # guess
        ## pos
        pos_hint_expansion = hints[:,None,:].repeat(1,args.p,1)
        pos_hint_expansion = pos_hint_expansion.reshape(this_bs*args.p, -1)
        pos_guesser_in = torch.cat([pos_hint_expansion, pos_cls], -1)
        pos_guesses = guesser(pos_guesser_in)

        ## neg
        neg_hint_expansion = hints[:,None,:].repeat(1,args.n,1)
        neg_hint_expansion = neg_hint_expansion.reshape(this_bs*args.n, -1)
        neg_guesser_in = torch.cat([neg_hint_expansion, neg_cls], -1)
        neg_guesses = guesser(neg_guesser_in)

        all_guesses = torch.cat([pos_guesses,neg_guesses], 0).reshape(-1)
        loss = loss_fn(all_guesses, labels)

        return loss

    def eval_and_log(model, hinter, guesser, loss_fn, eval_dataloader, train_losses):
        train_loss = np.mean(train_losses)
        train_losses = []

        model.eval()
        eval_losses = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                loss = model_forward(model, hinter, guesser, batch, loss_fn)
                eval_losses.append(loss.item())
        
        eval_loss = np.mean(eval_losses)

        print(f'train loss: {train_loss:.3f}')
        print(f'valid loss: {eval_loss:.3f}')

        if 'debug' not in args._tags:
            wandb.log(
                {
                    "train_loss": train_loss,
                    "valid_loss": eval_loss,
                    # "epoch": epoch,
                    # "step": completed_steps,
                },
                step=completed_steps,
            )
        model.train()

    loss_fn = nn.MSELoss()
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        
        train_losses = []
        for step, batch in enumerate(train_dataloader):
            loss = model_forward(model, hinter, guesser, batch, loss_fn)
            loss.backward()
            
            train_losses.append(loss.item())
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            completed_steps += 1

            if completed_steps >= args.max_train_steps:
                break
            
            if completed_steps % args.log_every == 0:
                eval_and_log(model, hinter, guesser, loss_fn, eval_dataloader, train_losses)

        eval_and_log(model, hinter, guesser, loss_fn, eval_dataloader, train_losses)

if __name__ == "__main__":
    main()