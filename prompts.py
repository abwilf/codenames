import sys; sys.path.append('/work/awilf/utils/'); from alex_utils import *

def make_str(elt):
    if isinstance(elt, str):
        return elt
    else:
        return ', '.join(elt)
    
def prompt0(pos, neg, hint=''):
    # hint_prompt = srep('''\
    # Alex: I want to find a word that is closely related to the set of words {} but distinct from {}.
    # Andrew: That word is clearly {}.
    # Alex: What about if the sets were {} and {}?
    # Andrew: Easy. The answer is {}.
    # Alex: Ok what about this one. The sets are {} and {}.
    # Andrew: Oh that one isn't hard at all. The word is clearly''').format(pos, neg, hint)
    # return hint_prompt
    
    prompt = srep('''\
    Question: What is the word that describes the words in Set A BUT NOT the words in Set B? Note: the answer should NOT be in either Set A or Set B.
    Context: Set A: {}. Set B: {}.
    Answer: {}''').format(make_str(pos), make_str(neg), hint)
    return prompt

prompts = [
    prompt0,   
]