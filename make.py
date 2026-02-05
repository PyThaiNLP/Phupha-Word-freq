import time
from pythainlp.corpus import thai_dict, thai_words, thai_orst_words
import requests
from tqdm.auto import tqdm
import pickle
def get_wordcount(w):
    payload = {
        'index': 'v2_cc-2025-30',
        'query_type': 'count',
        'query': w,
    }
    result = requests.post('https://api.infini-gram-mini.io/', json=payload).json()
    return result["count"]

thailist=list(thai_dict())+list(thai_words())+list(thai_orst_words())
thailist=list(set(thailist))

wordc={}
def make_dataset(w):
    global wordc
    if w not in wordc.keys():
        wordc[w]=get_wordcount(w)
    elif not isinstance(wordc[w],int):
        wordc[w]=get_wordcount(w)

def try_data(w):
    global wordc
    max_error=0
    while True:
        try:
            if max_error>10:
                break
            make_dataset(w)
            break
        except:
            print(f"Oops! That was no valid for {w}. Try again...")
            time.sleep(1)
            max_error+=1

for w in tqdm(thailist):
    try_data(w)
notinlist=set(thailist)-set(wordc.keys())
for w in tqdm(notinlist):
    try_data(w)
with open("words.pickle", 'wb') as file:
    pickle.dump(wordc, file)

import pandas as pd

save={"word":[],"count":[]}
for key, value in wordc.items():
    save["word"].append(key)
    save["count"].append(value)

df = pd.DataFrame(save)
df.to_csv("wordfreq.csv",index=False)