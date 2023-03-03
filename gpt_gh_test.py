import openai
import sys; sys.path.append('/work/awilf/utils/'); from alex_utils import *
import wandb
import string
from sklearn.metrics import accuracy_score
from scipy.special import comb

arg_defaults = [
    ('--_tags', str, 'debug,gpt'), # NOTE: required if you use deploy_sweeps. please do not remove the _. Use 'debug' if you don't want wandb to sync.
    ('--seed', int, 42),
    ('--wdb_project', str, ''), # defaults to chdir, but will be overwritten if called as part of a sweep
    ('--wdb_entity', str, 'socialiq'),

    # main arguments
    ('--N', int, 10),
    ('--n', int, 1),
    ('--p', int, 2),
    ('--np', int, 2), # -1 to use different n and p. by default, p=np, n=np-1. Must be > 1
    ('--guesser_model', str, 'chatgpt'),
    ('--guesser_num_retries', int, 3), # how many times to tell the guesser the answer wasn't formatted correctly
    ('--guesser_few_shot', int, 0), # how many few shot examples to give
    ('--dataset', str, 'noun'), # nouns or siqa
]


hint_prompt = '''\
The following is a game. The goal is to find a word that describes the words in Set A BUT NOT the words in Set B.
Set A: {}
Set B: {}
Differentiating word: '''

examples = [
    {
        'words': ['weather', 'food', 'snowboard'],
        'hint': 'jacket',
        'seta': 'weather, snowboard',
        'setb': 'food',
    },
    {
        'words': ['music', 'meat', 'year'],
        'hint': 'spoil',
        'seta': 'meat, year',
        'setb': 'music',
    },
    {
        'words': ['understanding', 'reading', 'data'],
        'hint': 'book',
        'seta': 'understanding, reading',
        'setb': 'data'
    }
]
def set_seed(seed):
    if seed in [-1,0]:
        seed = random.randint(0,100)
    random.seed(seed)
    np.random.seed(seed)

def get_phrasing():
    a = '''please phrase your answer in the following way:
    Set A: {}
    Set B: {}
    '''.format(', '.join(['word']*args.p), ', '.join(['word']*args.n))
    return a

def get_base_prompt(hint, words):
    # hint is str, words is arr
    a = srep('''\
    The following is a game. There are {} words in total, \
    divided into two sets: Set A and Set B. There are {} words \
    in Set A and {} words in Set B. The hint describes words in Set A but not words in Set B.
    The hint is: {}
    The words are: {}

    Please choose which words are in Set A and which words are in Set B, and {}''').format(args.n+args.p, args.p, args.n, hint, ', '.join(words), get_phrasing())
    return a

def get_answer(example):
    return f'''Set A: {example['seta']}\nSet B: {example['setb']}\n'''

def get_full_example(example):
    prompt = get_base_prompt(example['hint'], example['words'])
    return [
        {'role': 'user', 'content': prompt},
        {'role': 'assistant', 'content': get_answer(example)}
    ]

def get_guess(all_words, hint):
    words, labels = zip(*all_words)
    
    message_examples = [get_full_example(example) for example in examples][:args.guesser_few_shot]
    message_examples = [elt for arr in message_examples for elt in arr]  # flatten
    base_prompt = get_base_prompt(hint, words)
    m = [
        {"role": "system", "content": "You are a helpful assistant."},
        *message_examples,
        {'role': 'user', 'content': base_prompt},
    ]
    res=openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=m,
    )
    # parse response: make sure it's accurate, if not try num_retries times to get it to fix the answer
    try_again_pfx = 'This answer was malformed. Please try again and this time'
    y_pred, y_true = None, None
    for i in range(args.guesser_num_retries):
        answer = res.choices[0].message.content
        try_again_msg = ''
        ans = answer.split('\n')
        try:
            ans = lfilter(lambda elt: elt.strip()[:3]=='Set', ans)
            ans = lmap(lambda elt: elt.split(':',1)[1].split(','), ans)
        except:
            hi=2
            assert False, f'Answer: {answer}\nHint: {hint}\nWords: {words}\n'

        if len(ans)!=2:
            try_again_msg = f'{try_again_pfx} {get_phrasing()}'
        pos, neg = ans
        pos = lmap(lambda elt: elt.strip().rstrip().lower(), pos)
        neg = lmap(lambda elt: elt.strip().rstrip().lower(), neg)
        if not len(neg)==args.n or not len(pos)==args.p:
            try_again_msg = f'{try_again_pfx} make sure you list {args.p} words for Set A and {args.n} words for Set B. And remember to {get_phrasing()}'
        if not np.all([elt in words for elt in pos]):
            try_again_msg = f'{try_again_pfx} make sure all the words in Set A are in the words list above. And remember to {get_phrasing()}'
        if not np.all([elt in words for elt in neg]):
            try_again_msg = f'{try_again_pfx} make sure all the words in Set B are in the words list above. And remember to {get_phrasing()}'
        if np.any([elt in pos for elt in neg]):
            try_again_msg = f'{try_again_pfx} make sure that there is no overlap between the words in Set A and Set B.'

        if try_again_msg != '':
            m.append({'role': 'assistant', 'content': answer})
            m.append({'role': 'user', 'content': try_again_msg})
            res=openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=m,
            )
        else:
            y_pred = []
            y_true = []
            for word,lab in zip(words,labels):
                y_pred.append(1) if word.lower() in pos else y_pred.append(0)
                y_true.append(lab)
    if y_pred is None:
        print('Failed to get a well formed response from chatgpt; here is the message history:\n','\n###\n'.join(lmap(lambda elt: elt['content'], m)))
    return y_pred, y_true, m

def main():
    global args
    
    # TODO (optional): if there are any features of the argparse parser you want to add, initialize and pass in your parser here.
    parser = None
    args = process_defaults(arg_defaults, parser_in=parser)
    assert args.guesser_model == 'chatgpt'
    set_seed(args.seed)
    if args.np!=-1:
        assert args.np > 1
        args.p = args.np
        args.n = args.np-1

    if 'debug' not in args._tags:
        wandb.init(
            project=args.wdb_project,
            entity=args.wdb_entity, 
            config=vars(args),
            tags=args._tags.split(','),
        )

    openai.api_key = read_txt('open_ai_key.txt').strip().rstrip()
    nouns = read_txt('nouns.txt').split('\n')

    results = {
        'y_pred': [],
        'y_true': [],
        'words': [],
        'positives': [],
        'negatives': [],
        'hints': [],
        'failed': []
    }
    num_badly_formed = 0
    for i in tqdm(range(args.N)):
        group = np.random.choice(nouns, size=args.n+args.p, replace=False)
        pos = group[:args.p]
        neg = group[args.p:]
        results['positives'].append(pos)
        results['negatives'].append(neg)

        # hint
        this_hint_prompt = hint_prompt.format(', '.join(pos), ', '.join(neg))
        res=openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": this_hint_prompt},
                # {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
                # {"role": "user", "content": "Where was it played?"}
            ],
        )
        hint = res.choices[0].message.content.split(' ')[0].translate(str.maketrans('', '', string.punctuation))
        results['hints'].append(hint)

        # guess
        pos = lmap(lambda elt: (elt, 1), pos)
        neg = lmap(lambda elt: (elt, 0), neg)
        all_words = pos + neg
        random.shuffle(all_words)
        results['words'].append(all_words)

        _y_pred, _y_true, m = get_guess(all_words, hint)
        if _y_pred is None:
            results['failed'].append(m)
        else:
            results['y_pred'].append(_y_pred)
            results['y_true'].append(_y_true)

    results['y_true'] = ar(results['y_true'])
    results['y_pred'] = ar(results['y_pred'])
    corr_arr = (results['y_true'] == results['y_pred']).all(-1)
    results['total_acc'] = corr_arr.sum() / corr_arr.shape[0]
    print('Accuracy:', results['total_acc'])
    random_acc = 1/comb(args.n+args.p, args.p)
    print('Random Accuracy: ', random_acc)
    print(f'Number failed: {len(results["failed"])}')
    
    if 'debug' not in args._tags:
        wandb.summary['accuracy'] = results['total_acc']
        wandb.summary['random_acc'] = random_acc

if __name__=='__main__':
    main()

    