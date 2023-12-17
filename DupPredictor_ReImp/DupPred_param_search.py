import nltk
import numpy as np
# import pandas as pd # pandas lib for data handling
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
from tqdm import tqdm
from gensim.test.utils import datapath
from sklearn.metrics.pairwise import cosine_similarity
import gensim
import os
import sys
import json
import sklearn
import random
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
ps = PorterStemmer()
sw_nltk = stopwords.words('english') # bag of all common english stop words


def convert_HTML(text):
    # remove duplicate text + isolate code blocks 
    
    bs_text = BeautifulSoup(text,features="lxml")
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
    
    if len(cbs)!=0:
        for cb in cbs:
            cb_list.append(cb.get_text())
            cb.clear()
            
    text = bs_text.get_text()
    return text, cb_list

def TokStem(text,s_str=False):
    stoplist = set(sw_nltk)
    text = text.replace("'", '')
    words =  nltk.word_tokenize(text.lower())
    ltz = WordNetLemmatizer()
    final = []
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~``.'''
    contract_strings = ["'s", "\"","'ve"]
    for word in words:
        if word not in stoplist and word not in punc and word not in contract_strings and word!='"':
            final.append(ps.stem(word))
    if s_str==True:
        return " ".join(final)
    return final

def get_corpora(dataset, save_model = False):
    data_words = []
    for i in tqdm(range(0, len(dataset))): #len(dataset)
        for j in ['ori', 'dup']:
            post = dataset[i][j]
            title_text = TokStem(dataset[i][j]['Title'], False)
            body_text = TokStem(convert_HTML(dataset[i][j]['Body'])[0], False)
            post_text = title_text + body_text
            data_words.append(post_text)
    id2word = gensim.corpora.Dictionary(data_words)
    
    corpus = []
    for text in data_words:
        new = id2word.doc2bow(text)
        corpus.append(new)

    lda_model = gensim.models.LdaMulticore(workers=15, corpus=corpus, id2word=id2word, num_topics=100, minimum_probability=0.0, random_state=1)
    
    if save_model == True:
#         id2word = {k:v for k, v in lda_model.id2word.items()}
#         lda_model.id2word = None
        lda_path = './lda_model'
        if not os.path.exists(lda_path):
            os.makedirs(lda_path)
        lda_model.save(lda_path+'/lda.gensim')
#         lda_model.id2word = id2word  # restore the dictionary
#         with open(lda_path+'/lda.id2word.json', 'wb') as out:
#             json.dump(id2word, out)
    
    return lda_model

def get_lda_rep(text, lda_model):
    text_bow = lda_model.id2word.doc2bow(text)
    topic_rep = lda_model[text_bow]
    ret = np.zeros((1,100))
    topic_rep = np.array(topic_rep)[:,1]
    return topic_rep

def cos_sim(a,b): # finding cosine similarity score for given 2 strings
    # input two strings that are split into words
    us = [] # creating union set
    us += b
    for x in a:
        if x not in b:
            us.append(x)
    freqa,freqb,frequ = dict(),dict(),dict()
    wa,wb = [], []
    for word in a: # freq bag for string a
        if word in freqa:
            freqa[word] += 1
        else:
            freqa[word] = 1
    for word in b:# freq bag for string b
        if word in freqb:
            freqb[word] += 1
        else:
            freqb[word] = 1
    for word in us: # freq bag for union of a and b
        if word in frequ:
            frequ[word] += 1
        else:
            frequ[word] = 1
    for i in range(len(us)): # calc TitleVec for a and b
        x = us[i]
        if x in a:
            wa.append(freqa[x]/frequ[x])
        else:
            wa.append(0)
        if x in b:
            wb.append(freqb[x]/frequ[x])
        else:
            wb.append(0)
    wa = np.array(wa)
    wb = np.array(wb)
    
    # ret = np.dot(wa,wb)/(np.linalg.norm(wa)*np.linalg.norm(wb)) # cosine value
    ret =  cosine_similarity([wa],[wb])# cosine value
        # fix empty
    if np.isnan(ret):
        ret = 0
    
    return ret[0][0]


def cal_sims(train_set):
    
    fea_list = []
    lb_list = []
    for sample in train_set:
        fea = []
        post_1 = sample['ori']
        post_2 = sample['dup']
        title_1 = TokStem(post_1['Title'])
        title_2 = TokStem(post_2['Title'])
        title_sim = cos_sim(title_1, title_2)
        
        body_1 = TokStem(convert_HTML(post_1['Body'])[0])
        body_2 = TokStem(convert_HTML(post_2['Body'])[0])
        body_sim = cos_sim(body_1, body_2)
        
        title_body_1 = title_1+body_1
        title_body_2 = title_2+body_2
        topic_1 = get_lda_rep(title_body_1, lda_model)
        topic_2 = get_lda_rep(title_body_2, lda_model)
        topic_sim = cosine_similarity([topic_1],[topic_2])[0][0]
        # topic_sim =  np.dot(topic_1,topic_2)/(np.linalg.norm(topic_1)*np.linalg.norm(topic_2))
    #     print(topic_sim)
        # tag_1 = trans_tags(post_1['tag'])
        # tag_2 = trans_tags(post_2['tag'])

        tag_sim = cos_sim(post_1['Tags'], post_2['Tags'])
        
        fea.append(title_sim)
        fea.append(body_sim)
        fea.append(topic_sim)
        fea.append(tag_sim)
        lb_list.append(int(sample['lb']))
        fea_list.append(fea)
    return fea_list, lb_list

def composer_(sp, alpha, belta, gamma, delta):
    sim_score = alpha * sp[0] + belta * sp[1] + gamma * sp[2] + delta * sp[3]
    return sim_score

def batch_composer(fea_list, alpha = 0.8, belta = 0.51, gamma=0.37, delta=0.04):
    sim_list = []
    for fea in fea_list:
        sim_list.append(composer_(fea, alpha, belta, gamma, delta))
    return sim_list

def grid_search(dataset, max_itr = 100):
    
    # itr for grid search
    all_best_roc = 0.5
    all_best_para = [0,0,0,0]
    for n_itr in range(max_itr):
        # initialize the parameters
        best_roc = 0.5
        alpha = random.uniform(0, 1)
        belta = random.uniform(0, 1)
        gamma = random.uniform(0, 1)
        delta = random.uniform(0, 1)
        para_list = [alpha,belta,gamma,delta]
        # print('initial paras:', para_list)
        for i_para in range(len(para_list)):
            print('para:',i_para)
            para_best = para_list[i_para]
            para_list[i_para] = 0
            while para_list[i_para]<=1:
                print('cur:%3f,%3f,%3f,%3f' % (para_list[0], para_list[1], para_list[2], para_list[3]))
                sim_list = []
                fea_list, lb_list = cal_sims(dataset)
                sim_list = batch_composer(fea_list, *para_list)
                roc_auc = sklearn.metrics.roc_auc_score(lb_list,sim_list)
                # roc_auc=0
                print('cur roc: %2f'% roc_auc)
                if roc_auc > best_roc:
                    print('cur run best ROC: %2f' % roc_auc)
                    # print('cur run best para: %3f,%3f,%3f,%3f' % (para_list[0], para_list[1], para_list[2], para_list[3]))
                    para_best = para_list[i_para]
                    best_roc = roc_auc
                    if roc_auc> all_best_roc:
                        all_best_para = para_list
                        all_best_roc = roc_auc
                        print('*cur all best ROC: %2f*' % all_best_roc)
                        print('*cur all best para: %3f,%3f,%3f,%3f*' % (all_best_para[0], all_best_para[1], all_best_para[2], all_best_para[3]))
                para_list[i_para] = para_list[i_para] +0.01
            para_list[i_para] = para_best
            print('update:%3f, %3f,%3f,%3f' % (para_list[0], para_list[1], para_list[2], para_list[3]))
        print('*ITR:%d*cur all best ROC: %2f*' % (n_itr, all_best_roc))
        print('*ITR:%d*cur all best para: %3f,%3f,%3f,%3f*' % (n_itr ,all_best_para[0], all_best_para[1], all_best_para[2], all_best_para[3]))
    print('Finial best para:',all_best_para)
    return all_best_para
    

if __name__ == "__main__":
    train_set = np.load('./dataset/trainset.npy', allow_pickle = True)
    lda_model = get_corpora(train_set, save_model = True)
    best_param = grid_search(train_set)