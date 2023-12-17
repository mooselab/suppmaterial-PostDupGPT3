from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer


def convert_HTML(text):
    # remove duplicate text + extract code blocks 
    # return pure text and code blocks
    
    bs_text = BeautifulSoup(text, features="lxml")
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


def get_copus(data):
    data_len = len(data)
    corpus = []
    for i in range(data_len):
        for j in ['dup', 'ori']:
            text = TokStem(data[i][j]['Title'] + ' ' + convert_HTML(data[i][j]['Body'])[0])
            corpus.append(text)
    return corpus

# corpus = get_copus(train_set)