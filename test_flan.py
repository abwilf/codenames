from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import sys; sys.path.append('/work/awilf/utils/'); from alex_utils import *
from flan_examples import examples, questions
from prompts import prompts, make_str
import openai, string

# args
few_shot_k = 5
model_name = 'flan-t5-xl'
# model_name = 'chatgpt'

if 'flan' in model_name:
    model = AutoModelForSeq2SeqLM.from_pretrained(f"google/{model_name}")
    model = model.to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(f"google/{model_name}")

nouns = read_txt('nouns.txt').split('\n')
openai.api_key = read_txt('open_ai_key.txt').strip().rstrip()


def get_chatgpt_out(prompt):
    res=openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    hint = res.choices[0].message.content.split(' ')[0].translate(str.maketrans('', '', string.punctuation))
    return hint

def get_flan_out(prompt):
    # inputs = tokenizer("A step by step recipe to make bolognese pasta:", return_tensors="pt")
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = inputs.to('cuda')
    outputs = model.generate(**inputs)
    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return outputs
    
def get_prompt(pos, neg, prompt_fn, few_k=5):
    # add the question    
    question_prompt = prompt_fn(pos, neg, '')
    
    # add few shot examples
    prompt = ''
    for pos,neg,hint in examples[:few_k]:
        prompt += prompt_fn(pos, neg, hint) + '\n\n'

    return prompt + question_prompt

df = {k: [] for k in ['pos', 'neg'] + [f'prompt_{i}' for i in range(len(prompts))]}
for pos,neg,_ in tqdm(questions):
    df['pos'].append(make_str(pos))
    df['neg'].append(make_str(neg))

    for prompt_idx in range(len(prompts)):
        prompt_fn = prompts[prompt_idx]
        prompt = get_prompt(pos, neg, prompt_fn, few_k=few_shot_k)
        
        if model_name == 'chatgpt':
            hint = get_chatgpt_out(prompt)
        else:
            hint = get_flan_out(prompt)
        
        overlap = int(hint in make_str(pos)+make_str(neg))
        df[f'prompt_{prompt_idx}'].append(f'{hint}/{overlap}')

df = pd.DataFrame(df) # make dataframe
print(df)
hi=2