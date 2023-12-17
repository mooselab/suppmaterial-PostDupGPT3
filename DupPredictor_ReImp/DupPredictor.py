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
import datetime

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

def composer(post_1, post_2, alpha = 0.8, belta = 0.51, gamma=0.37, delta=0.04):
    title_1 = TokStem(post_1['Title'])
    title_2 = TokStem(post_2['Title'])
    
    title_sim = cos_sim(title_1, title_2)
    
    body_1 = TokStem(convert_HTML(post_1['Body'])[0])
    body_2 = TokStem(convert_HTML(post_2['Body'])[0])
    if len(body_1)==0 or len(body_2) == 0:
        body_sim = 0
    else:
        body_sim = cos_sim(body_1, body_2)

    
    title_body_1 = title_1+body_1
    title_body_2 = title_2+body_2
    topic_1 = get_lda_rep(title_body_1, lda_model)
    topic_2 = get_lda_rep(title_body_2, lda_model)
    topic_sim =  np.dot(topic_1,topic_2)/(np.linalg.norm(topic_1)*np.linalg.norm(topic_2))
#     print(topic_sim)
    # tag_1 = trans_tags(post_1['tag'])
    # tag_2 = trans_tags(post_2['tag'])
    
    tag_sim = cos_sim(post_1['Tags'], post_2['Tags'])
    
    sim_score = alpha * title_sim + belta * body_sim + gamma * topic_sim + delta * tag_sim
    
    return sim_score

def batch_composer(fea_list, alpha = 0.52, belta = 0.70, gamma=0.01, delta=0.39):
    sim_list = []
    for fea in fea_list:
        sim_list.append(composer(fea, alpha, belta, gamma, delta))
    return sim_list


def compare_with_candidate(query, can_list, max_can = 50):
    can_score = {}
    for can in can_list:
        score = composer(query, can)
        can_score[can['Id']] = score
    # sort according to the score    
    ranking = list(int(i[0]) for i in sorted(can_score.items(), key=lambda x: x[1], reverse=True))[:max_can]
    return ranking

# generate rankings for all queries

def generate_rankings(query_list, can_list, max_list=50):
    gts_list = []
    ranking_list = []
    for query in tqdm(query_list):
        gts= query['dupof']
        ranking = compare_with_candidate(query, can_list, max_list)
        gts_list.append(gts)
        ranking_list.append(ranking)
    return gts_list, ranking_list

def cal_top_n_score(gts_list, rank_list, N = 10):
    n_samples = len(gts_list)
    score = 0
    for i in range(n_samples):
        gts = gts_list[i]
        rl = rank_list[i][:N]
        hit = 0
        for gt in gts:
            if gt in rl:
                hit = 1
        score = score+hit
    return score/n_samples
        
def itr_domains(dml):
    for dm in dml:
        test_set = np.load('./dataset/'+ dm +'.npz', allow_pickle = True)
        query_questions = test_set['query']
        candidate_questions = test_set['candidate']
        start = datetime.datetime.now()
        print(start)
        gts_list, ranking_list = generate_rankings(query_questions, candidate_questions)
        end = datetime.datetime.now()
        print(end, end-start)
        np.savez('./tmp/'+dm+'.npz', gt = gts_list, ranking = ranking_list)
        top_1 = cal_top_n_score(gts_list, ranking_list, 1)
        top_3 = cal_top_n_score(gts_list, ranking_list, 3)
        top_5 = cal_top_n_score(gts_list, ranking_list, 5)
        top_10 = cal_top_n_score(gts_list, ranking_list, 10)
        top_30 = cal_top_n_score(gts_list, ranking_list, 30)
        print(top_1,top_3,top_5,top_10,top_30)
        with open('./tmp/results.txt', 'a') as f:
            f.writelines([str(start),'\n'])
            f.writelines([str(end),'\n'])
            f.writelines([str(dm)+':'+str(top_1)+' '+str(top_3)+' '+str(top_5)+' '+str(top_10)+' '+str(top_30),'\n'])
        
if __name__ == "__main__":
    domain_list = ['android', 'gis', 'tex', 'mathematica', 'programmers', 'stats', 'unix', 'webmasters', 'wordpress']
    lda_model = gensim.models.LdaMulticore.load('./lda_model/lda.gensim')
    itr_domains(domain_list)
