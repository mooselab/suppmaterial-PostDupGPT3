import numpy as np
import openai
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
ps = PorterStemmer()
from tqdm import trange, tqdm
import time
import re

# preprocessing function

def convert_HTML(text):
    # remove duplicate text + extract code blocks 
    # return pure text and code blocks
    
    bs_text = BeautifulSoup(text)
    text = "Possible Duplicate:"
    dup_can = bs_text.find(lambda tag: tag.name == "strong" and text in tag.text)
    if dup_can!=None:
        wrapping = dup_can.find_previous('blockquote')
        if wrapping!= None: wrapping.clear()
#         print(dup_can.parent.parent)

    text = "This is a duplicate of"
    dup_can = bs_text.find(lambda tag: tag.name == "p" and text in tag.text)
    if dup_can!=None:
        wrapping = dup_can.clear()

    # extract code block
    
    cb_list = []
#     cbs = bs_text.find_all("code")
    cbs = bs_text.select("pre > code")
    
    bq_list = []
    bqs = bs_text.select("blockquote")
    
    
    if len(cbs)!=0:
        for cb in cbs:
            cb_list.append(cb.get_text())
            cb.clear()
            
    if len(bqs)!=0:
        for bq in bqs:
            bq_list.append(bq.get_text())
            bq.clear()
            
    text = re.sub(r"data:image\/jpeg;base64\S+", "", bs_text.get_text())
    return text, cb_list, bq_list


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

        
def check_key(bat, dataset, id_list):
    sw = True
    for i in bat:
        if 'gpt' not in dataset[id_list[i]].keys():
            sw = False
    return sw
    
def get_gpt_embedding(dataset, batch_size = 2):
    id_list = list(dataset.keys())
    len_data = len(dataset)
    pbar = tqdm(total = len_data)
    for bat in batch(range(0, len_data), batch_size):
        pbar.update(len(bat))
        text_in_batch = []
        if check_key(bat, dataset, id_list):
            # print('pass')
            continue
        
        for i in bat:
            post = dataset[id_list[i]]
            text = convert_HTML(post['Title']+'. '+post['Body'])[0]
            if len(text.split()) >= 8000:
                print(len(text.split()))
            if len(text)==0:
                text = ' '
            text_in_batch.append(text)
            
        bat_emb = openai.Embedding.create(input=text_in_batch, model="text-embedding-ada-002")["data"]
        for i in bat:
            post = dataset[id_list[i]]
            post['gpt'] = np.array(bat_emb[i%batch_size]['embedding'])
    
    return dataset



# define your access key from OPENAI account

openai.api_key = 'sk-XXXX' 

dataset = np.load('./dataset/all_question_gpt.npy', allow_pickle = True)[()]

for _ in range(1000000):
    openai.api_key = keys[_%len(keys)]
    if _%10 == 0 and _ != 0:
        print('save.')
        np.save('./dataset/all_question_gpt.npy', dataset)

    try:
        all_data_with_emb = get_gpt_embedding(dataset, batch_size = 64)
        break
    except openai.error.RateLimitError:
        print('Rate limit reached.')
        time.sleep(10)
        continue
    except openai.error.InvalidRequestError:
        print('exceed!')
        np.save('./dataset/all_question_gpt.npy', dataset)
        break
        
np.save('./dataset/all_question_gpt.npy', dataset)
