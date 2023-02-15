#!/usr/bin/env python
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
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import sys; sys.path.append('/work/awilf/utils/'); from alex_utils import *
import argparse
import json
import logging
import math
import os
import random
from itertools import chain
from pathlib import Path

import wandb
import datasets
import torch
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from torch import nn

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    # GPT2ForSequenceClassification,
)
from modeling_gpt2 import GPT2ForSequenceClassification
from transformers.utils import check_min_version, get_full_repo_name, send_example_telemetry
from transformers.utils.versions import require_version

from eval_gpt import eval_gpt
from train_gpt import train_model as train_model_lm

from sklearn.model_selection import train_test_split

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.27.0.dev0")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = get_logger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    
    ## GH specific params
    parser.add_argument(
        "--N",
        type=float,
        default=1e4,
        help="Number of datapoints to generate",
    )
    parser.add_argument(
        "--np",
        type=int,
        default=-1,
        help="Set > 1 if you want n and p to be the same. Overrides n and p",
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
        "--run_clm_every",
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
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--eval_every", type=float, default=100, help="Evaluate zero shot every eval_every batches of training"
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
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
        type=float,
        default=400,
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
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--no_keep_linebreaks", action="store_true", help="Do not keep line breaks when using TXT files."
    )
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
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--_tags",
        type=str,
        default='debug',
    )
    args = parser.parse_args()

    args.max_train_steps = int(args.max_train_steps)
    args.eval_every = int(args.eval_every)
    args.N = int(args.N)
    if args.np > 1:
        args.n = args.np
        args.p = args.np

    # Sanity checks
    if args.dataset_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, json or txt file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, json or txt file."

    return args

class Gpt2ClassificationCollator(object):
    def __init__(self, tokenizer, max_sequence_len=None):
        self.tokenizer = tokenizer
        self.max_sequence_len = tokenizer.model_max_length

    def __call__(self, batch):
        full_list = []
        for elt in batch:
            full_list.extend([elt['prompt'], *elt['pos'], *elt['neg']])
        inputs = self.tokenizer(text=full_list, return_tensors="pt", padding=True, truncation=True,  max_length=self.max_sequence_len)
        return inputs
        
def evaluate_model(model, testing_model, tokenizer):
    print('## Evaluating ##')
    model.eval()
    testing_model.to(device)
    testing_model.transformer = model.transformer
    testing_model.eval()
    losses = []
    with torch.no_grad():
        eval_args = {
            'model': testing_model,
            'tokenizer': tokenizer,
            'lm': 'gpt2',
            'dataset_file': "/work/awilf/siqa/tasks/socialiqa_dev.jsonl",
            'out_dir': '/work/awilf/siqa/results',
            'device': 0,
            'reader': 'socialiqa',
        }
        acc = eval_gpt(**eval_args)
    print('## End Evaluation ##')
    testing_model.to('cpu')
    model.to(device)
    model.train()
    return acc

def validate_model(model, guesser, loss_fn, valid_dataloader):
    model.eval()
    guesser.eval()
    with torch.no_grad():
        losses = []
        for step, batch in enumerate(tqdm(valid_dataloader)):
            loss = model_forward(model, guesser, batch, loss_fn)
            losses.append(loss.item())

    model.train()
    guesser.train()
    return np.mean(losses)

def model_forward(model, guesser, batch, loss_fn):
    ## get encodings of pos, neg, and hints
    batch = {k:v.type(torch.long).to(device) for k,v in batch.items()}
    input = {
        **batch,
        'return_hidden': True
    }
    try:
        encodings = model(**input) # per_device_train_batch_size*(1+args.p+args.n) b/c for each sample, flattens out prompt, pos, and neg and encodes all in the same batch
    except:
        hi=2
        print('ERROR WITH INFERENCE!')
        exit()
    this_bs = int(encodings.shape[0] / (1+args.p+args.n)) # almost always will be per_device_train_batch_size, except for edge cases
    encodings = encodings.reshape(this_bs, (1+args.p+args.n), -1)
    hints = encodings[:,0] # bs, hidden_dim
    pos = encodings[:,1:(args.p+1)] # bs,args.p,hidden_dim
    neg = encodings[:,(args.p+1):] # bs,args.n,hidden_dim
    ##

    ## run through guesser
    # pos_hint_expansion = hints[:,None,:].repeat(1,args.p,1)
    # neg_hint_expansion = hints[:,None,:].repeat(1,args.n,1)
    pos_hint_expansion = hints[:,None,:].expand(-1,args.p,-1)
    neg_hint_expansion = hints[:,None,:].expand(-1,args.n,-1)

    pos_hints = torch.cat([pos_hint_expansion, pos], -1).reshape(int(this_bs*args.p), -1) # expands hints to same dim as pos, cats along last dim, reshapes to (bs*args.p, hidden_dim)
    neg_hints = torch.cat([neg_hint_expansion, neg], -1).reshape(int(this_bs*args.n), -1) # expands hints to same dim as pos, cats along last dim, reshapes to (bs*args.p, hidden_dim)
    guesser_input = torch.cat([pos_hints, neg_hints], 0)
    guesser_output = guesser(guesser_input).reshape(-1) # bs*(args.p+args.n),

    with torch.no_grad():
        labels = torch.cat([torch.ones(int(this_bs*args.p)), torch.zeros(int(this_bs*args.n))])
        labels = labels.to(device)

    loss = loss_fn(guesser_output, labels)
    return loss

def main():
    global args
    args = parse_args()

    if 'debug' not in args._tags:
        wandb.init(
            project='siqa',
            entity='socialiq',
            config=vars(args),
            tags=args._tags.split(','),
        )

    accelerator_log_kwargs = {}
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)

    if args.seed is not None:
        set_seed(args.seed)

    assert args.dataset_name is not None

    ## construct dataset ##
    ds = load_jsonl(args.dataset_name)
    ds = arlmap(lambda elt: elt['context'], ds)
    new_ds = []
    new_ds_name = f'{args.dataset_name}.bak.{int(np.random.choice(np.arange(1e4)))}'

    def get_prompt(pos, neg):
        ## TODO
        prompt = '''The following sentences are grouped into two sets, Set A and Set B.\nSet A: {}\nSet B: {}\nThe word that best describes the sentences in Set A but not the sentences in Set B is: '''.format(
            ' '.join([f'({i+1}) {elt}' for i,elt in enumerate(pos)]),
            ' '.join([f'({i+1}) {elt}' for i,elt in enumerate(neg)]),
        )
        return prompt

    new_ds_name_base = f'{args.dataset_name}.bak.{args.N}.{args.n}.{args.p}'
    new_ds_name_train = f'{new_ds_name_base}.train'
    new_ds_name_valid = f'{new_ds_name_base}.valid'
    if not exists(new_ds_name_train) or not exists(new_ds_name_valid):
        for _ in tqdm(range(args.N)):
            # choose n positives, p negatives without replacement
            group = np.random.choice(ds, size=args.n+args.p, replace=False)
            pos = group[:args.p]
            neg = group[args.p:]
            prompt = get_prompt(pos, neg)
            new_ds.append({
                'prompt': prompt,
                'pos': lmap(lambda elt: str(elt), pos),
                'neg': lmap(lambda elt: str(elt), neg),
            })
            # new_ds.append([prompt, *lmap(lambda elt: str(elt), pos), *lmap(lambda elt: str(elt), neg)])
        train_ds, valid_ds = train_test_split(new_ds, test_size=.2)
        save_jsonl(new_ds_name_train, train_ds)
        save_jsonl(new_ds_name_valid, valid_ds)
    
    # exit()
    ds = load_dataset('json', data_files={'train': new_ds_name_train, 'valid': new_ds_name_valid})
    ##

    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.model_name_or_path, config=config)
    guesser = nn.Sequential(
        nn.Linear(config.n_embd*2, 1, bias=True),
        nn.Sigmoid(),
    )
    guesser = guesser.to(device)

    testing_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
    model.to(device)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    collator = Gpt2ClassificationCollator(tokenizer=tokenizer)
    train_dataloader = DataLoader(ds['train'], shuffle=True, collate_fn=collator, batch_size=args.per_device_train_batch_size )
    valid_dataloader = DataLoader(ds['valid'], shuffle=True, collate_fn=collator, batch_size=args.per_device_train_batch_size )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
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
            "params": [p for n, p in guesser.named_parameters()],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Train!
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0

    model.train()
    guesser.train()
    total_loss = 0
    completed_steps = 0
    eval_interval_steps = 0

    loss_fn = nn.MSELoss()
    
    for _ in range(int(1e8)): # large number of "epochs" - we're just using max_train_steps
        break_signal = False
        for step, batch in enumerate(train_dataloader):
            model.zero_grad()
            guesser.zero_grad()

            loss = model_forward(model, guesser, batch, loss_fn)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(guesser.parameters(), 1.0)

            total_loss += loss.item()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            completed_steps += 1
            eval_interval_steps += 1

            # run clm before evaluation
            # if completed_steps%args.run_clm_every==0:
            #     testing_model.transformer = model.transformer
            #     testing_model = train_model_lm(args, model_in=testing_model)
            #     model = model.to(device)

            # evaluate
            if completed_steps%args.eval_every==0:
                with torch.no_grad():
                    valid_loss = validate_model(model, guesser, loss_fn, valid_dataloader)
                    # acc = evaluate_model(model, testing_model, tokenizer)
                    total_loss /= eval_interval_steps
                    eval_interval_steps = 0
                    print('valid_loss', valid_loss)
                    print('train_loss', total_loss)

                    if 'debug' not in args._tags:
                        wandb.log({
                            # 'eval_acc': acc,
                            'valid_loss': valid_loss,
                            'train_loss': total_loss,
                        }, step=completed_steps)
                    
                    total_loss = 0

            if completed_steps >= args.max_train_steps:
                break_signal = True
                break
        
        if break_signal:
            break

    # acc = evaluate_model(model, testing_model, tokenizer)
    # if 'debug' not in args._tags:
    #     wandb.summary['acc'] = acc

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
        
    rmrf(new_ds_name)
if __name__ == "__main__":
    main()