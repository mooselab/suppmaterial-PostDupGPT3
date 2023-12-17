# PostDataset.py

import random
from torch.utils.data import Dataset


class TestDataset(Dataset):
    def __init__(self, dataset, get_ori = False):
        self.dup = dataset['dup']
        self.ori = dataset['ori']
        self.rel = dataset['rel']
        self.get_ori = get_ori
        # self.prep_fn = prep_fn
        # self.vectorizor = vectorizor
        
        self.dup_key_list = list(self.dup.keys())
        self.ori_key_list = list(self.ori.keys())
        
    def __len__(self):
        if self.get_ori:
            return len(self.ori)
        else:
            return len(self.dup)
    
    def __getitem__(self, idx):
        key_list = ['Title', 'Body', 'Tags']
        
        if self.get_ori:
            cur_key = self.ori_key_list[idx]
            cur_post = self.ori[cur_key]
            # post_query = {key: cur_post[key] for key in key_list}
            # if self.prep_fn:
            #     self.prep_fn(post_query)
            # emb_ori = self.vectorizor(post_query['Title'] + post_query['Body'])
            emb_ori = cur_post['gpt']
            return cur_key, emb_ori
        else:
            cur_key = self.dup_key_list[idx]
            cur_post = self.dup[cur_key]
            label_list = list(self.rel[cur_key])
            # post_query = {key: cur_post[key] for key in key_list}
            # if self.prep_fn:
            #     self.prep_fn(post_query)
            # emb_anc = self.vectorizor(post_query['Title'] + post_query['Body'])
            emb_anc = cur_post['gpt']
            return emb_anc, label_list

class DupPairDatasetOnTheFly(Dataset):
    def __init__(self, dataset): #, transform=None, target_transform=None
        self.dup = dataset['dup']
        self.ori = dataset['ori']
        self.rel = dataset['rel']
        # self.prep_fn = prep_fn
        # self.vectorizor = vectorizor

        self.dup_key_list = list(self.dup.keys())
        self.ori_key_list = list(self.ori.keys())
        
    def __len__(self):
        return len(self.dup)
    
    def __getitem__(self, idx):
        
        # key_list = ['Title', 'Body', 'Tags']
            
        cur_key = self.dup_key_list[idx]
        cur_post = self.dup[cur_key]   # anchor
        ori_post_list = list(self.rel[cur_key])
        ori_post = self.ori[random.choice(ori_post_list)]  # pos
        while True:
            cur_neg_id = random.choice(self.ori_key_list)
            if cur_neg_id not in ori_post_list:
                break
        neg_post = self.ori[cur_neg_id]   # neg
        
        emb_anc = cur_post['gpt']
        emb_pos = ori_post['gpt']
        emb_neg = neg_post['gpt']
    
        return {'anc': emb_anc, 'pos': emb_pos, 'neg': emb_neg}