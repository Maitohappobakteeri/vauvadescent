import numpy as np
import os
import common
import random
from prepare_data import vocab_size, words_to_lookup, SpecialCharacters
from log import log, LogTypes, important
import torch

topics = common.list_all_files(os.path.join(common.cache_dir, "posts"))
lengths = []
for t in topics:
    data = common.load_json_file(t)
    lengths.append(len(data))

important(f"Tokens in data: {sum(lengths)}")
important(f"Available unique training segments: {sum(lengths) // 1000}")
important(f"With 8 batches this takes {(sum(lengths) // 1000) // 8} iterations")

data = np.array(common.load_json_file(random.choice(topics)))
data = torch.nn.functional.one_hot(torch.from_numpy(data[:128]).long(), num_classes=vocab_size).float()

vocab = common.load_json_file(os.path.join(common.cache_dir, "characters.json"))
words = vocab["words"]
syllables = vocab["syllables"]
word_lookup = words_to_lookup(words)
char_to_index = vocab["char_to_index"]
index_to_char = vocab["index_to_char"]

def logits_to_index(t):
    arr = t.detach().cpu().numpy()
    return np.random.choice([i for i in range(vocab_size)], p=arr)

def is_new_post(i):
    return index_to_char.get(str(i), "#") == SpecialCharacters.NEW_POST.value

def index_to_char_fun(i):
    char = index_to_char.get(str(i), "#")
    if char == SpecialCharacters.NEW_POST.value:
        return "\n\nPostaus:\n"
    elif char == "\n":
        return " "
    else:
        return char
    
s = ""
for d in data:
    s = s + index_to_char_fun(logits_to_index(d))
important("Reversed encoded posts")
log(s, type=LogTypes.DATA, multiline=True)

important("How the posts are split to tokens")
for d in data:
    i = logits_to_index(d)
    log(f"{i:>5}: {index_to_char_fun(i)}", type=LogTypes.DATA)