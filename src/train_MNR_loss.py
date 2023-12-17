import numpy as np
# import random
import torch
from DuplicateNet import *
# import utils
from utils import TestDataset, DupPairDatasetOnTheFly
from torch import cuda
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os

from transformers import *
from transformers.utils import logging

class train_MNR_loss():
    def __init__(self, model, device, writer, train_loader, test_loader= None):
        self.model = model
        self.device = device
        self.writer = writer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = torch.optim.Adam(params = self.model.parameters(), lr = 0.01)
        return
    
    def evaluation(self, ep):
        ori_id, ori_emb = self.model.embedding_generation(test_set)
        label_list, ranking_list = self.model.evaluation(test_set, ori_id, ori_emb)
        # [top_3, top_5, top_10, top_20, top_30]
        top_1 = self.model.calculate_top_n(label_list, ranking_list, top_n = 1)
        top_3 = self.model.calculate_top_n(label_list, ranking_list, top_n = 3)
        top_5 = self.model.calculate_top_n(label_list, ranking_list, top_n = 5)
        top_10 = self.model.calculate_top_n(label_list, ranking_list, top_n = 10)
        top_30 = self.model.calculate_top_n(label_list, ranking_list, top_n = 30)
        writer.add_scalars("Epoch_Top_K", {'Top_1': top_1, 'Top_3': top_3, 'Top_5': top_5, 'Top_10': top_10, 'Top_30': top_30}, ep)
        writer.flush()
        return top_30
    
    def train(self, epoch=500):
        self.evaluation(0)
        self.model.train()
        best_score = 0
        for ep in range(1, epoch+1):
            total_loss = 0
            cnt = 0
            for _, data in enumerate(tqdm(self.train_loader), 0):
                cnt = cnt + len(data['anc'])

                emb_anc = data['anc'].to(self.device, dtype = torch.float64)
                emb_pos = data['pos'].to(self.device, dtype = torch.float64)
                
                output_anc = self.model(emb_anc)
                output_pos = self.model(emb_pos)

                scores = []
                scale = 20.0
                for a_i in output_anc:
                    scores.append(self.model.sim_fn(a_i.reshape(1, a_i.shape[0]), output_pos)*scale)
                scores = torch.stack(scores)
                labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)
        
                loss = self.model.loss_fn(scores, labels)
                writer.add_scalars("Loss "+"Epoch_"+str(ep), {'train': loss.item()}, _)
                writer.flush()
                total_loss = total_loss + loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            
            writer.add_scalars("Epoch_Loss", {'train': total_loss/cnt}, ep)
            writer.flush()
            print('loss: ', total_loss/cnt)
            if ep%1 ==0:
                cur_score = self.evaluation(ep)
                if cur_score >= best_score:
                    best_score = cur_score
                    self.model.save_model()
        return best_score


if __name__ == "__main__":

    device = 'cuda' if cuda.is_available() else 'cpu'
    train_set = np.load('../dataset/tiny_train.npy', allow_pickle = True)[()]
    test_set = np.load('../dataset/tiny_test.npy', allow_pickle = True)[()]

    train_dataset = DupPairDatasetOnTheFly(train_set)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    cross_entropy_loss_fn = torch.nn.CrossEntropyLoss()
    fea_dim = train_dataset[0]['anc'].shape[0]
    dup_model = DuplicateNet(fea_dim, cross_entropy_loss_fn, cos).double()
    dup_model.to(device)


    writer = SummaryWriter()
    train_class = train_MNR_loss(dup_model, device, writer, train_loader)
    os.system("rm -rf ./runs/*")
    acc = train_class.train()
