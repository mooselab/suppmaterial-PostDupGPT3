# DuplicateNet.py

import torch
from utils import TestDataset, DupPairDatasetOnTheFly
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

class DuplicateNet(torch.nn.Module):
    def __init__(self, dim_emb, loss_fn, sim_fn = None): 
        super(DuplicateNet, self).__init__()
        self.loss_fn = loss_fn
        self.dim_emb = dim_emb
        if sim_fn == None:
            self.sim_fn = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        else:
            self.sim_fn = sim_fn
        self.fc_p1 = torch.nn.Linear(self.dim_emb, 50)

    def forward(self, emb):
        return self.fc_p1(emb)
    
    def forward_eval(self, emb):
        self.eval()
        return self.fc_p1(emb)
        
    def embedding_generation(self, dataset):
        self.eval()
        device = next(self.parameters()).device
        
        test_ori_dataset = TestDataset(dataset, get_ori = True)
        ori_dataloader = DataLoader(test_ori_dataset, batch_size=32, shuffle=False)
        
        with torch.no_grad():
#             dup_emb=torch.tensor([]).to(device, dtype = torch.float64)
            ori_id=torch.tensor([]).to(device, dtype = torch.int)
            ori_emb=torch.tensor([]).to(device, dtype = torch.float64)
            
            print('Generating Embeddings for candidates.')
            for _, data in enumerate(tqdm(ori_dataloader)):
                output = self.forward_eval(data[1].to(device, dtype = torch.float64))
                ori_id = torch.cat((ori_id, data[0].to(device, dtype = torch.float64)))
                ori_emb = torch.cat((ori_emb, output))
        return ori_id, ori_emb
    
    def evaluation(self, dataset, ori_id, ori_emb):
        
        test_dataset = TestDataset(dataset, get_ori = False)
        device = next(self.parameters()).device
        label_list=[]
        ranking_list=[]

        for i, data in enumerate(tqdm(test_dataset), 0):
            output = self.forward_eval(torch.tensor(data[0]).to(device, dtype = torch.float64))
            output = torch.reshape(output, (1,-1))
            dis_list = self.sim_fn(output, ori_emb)
            _, indices_list = torch.sort(dis_list, descending=True)
            id_ranking = ori_id[indices_list]
            ranking_list.append(id_ranking[0:100].to('cpu'))
            label_list.append(data[1])
        return label_list, ranking_list

    def calculate_top_n(self, label_list, ranking_list, top_n = 10):
        
        hit_cnt = 0
        for i, lbs in enumerate(label_list):
            hit_flag = False
            for lb in lbs:
                if lb in ranking_list[i][0:top_n]:
                    hit_flag = True
            if hit_flag == True:
                hit_cnt = hit_cnt + 1

        total = len(label_list)

        print('Top-'+str(top_n)+':', hit_cnt/total)
        return hit_cnt/total
    
    def save_model(self, path='./trained_models/model.pt'):
        dir_path = os.path.dirname(path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        torch.save(self.state_dict(), path)