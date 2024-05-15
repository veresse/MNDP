import torch
import torch.nn.functional as F
import sys
from torch import nn
import numpy as np
from gat import GAT
from transformer import *
from interformer import Decoder
from hyperparameter import hyperparameter
from sw_tf import SwinTransformerModel

hp = hyperparameter()

sys.path.append('..')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class sw_tf(nn.Module):
    def  __init__(self,hid_dim, dropout, device):
        super(sw_tf,self).__init__()
        self.hid_dim = hid_dim

        self.batch_size = hp.batch
        self.max_drug = hp.MAX_DRUG_LEN
        self.max_protein = hp.MAX_PROTEIN_LEN
        self.pro_emb = hp.pro_emb
        self.smi_emb = hp.smi_emb

        self.dropout = dropout
        self.device = device

        self.sw_pro_embed = nn.Embedding(self.pro_emb, 64, padding_idx=0)
        self.sw_smi_embed = nn.Embedding(self.smi_emb, 64, padding_idx=0)

        self.sw_tf = nn.Sequential(SwinTransformerModel(),
                                   nn.Dropout(dropout));

        self.gat = GAT()

        self.out = nn.Sequential(
            nn.Linear(576, 512),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(512, 2)
        )

    def forward(self, compound, adj, smi_ids, prot_ids):
        drug_gat = self.gat(compound, adj)

        sw_pro_embed = self.sw_pro_embed(prot_ids)
        sw_smi_embed = self.sw_smi_embed(smi_ids)

        sw_smi = self.sw_tf(sw_smi_embed)
        sw_pro = self.sw_tf(sw_pro_embed)

        sw_smi = sw_smi.max(dim=1)[0]
        sw_pro = sw_pro.max(dim=1)[0]
        gat_smi = drug_gat.max(dim=1)[0]

        out_fc = torch.cat([sw_smi, sw_pro,gat_smi], dim=1)

        return self.out(out_fc)

class in_tf(nn.Module):
    def __init__(self,hid_dim, dropout, device):
        super(in_tf,self).__init__()
        self.hid_dim = hid_dim

        self.batch_size = hp.batch
        self.max_drug = hp.MAX_DRUG_LEN
        self.max_protein = hp.MAX_PROTEIN_LEN
        self.pro_emb = hp.pro_emb
        self.smi_emb = hp.smi_emb

        self.dropout = dropout
        self.device = device

        self.sw_pro_embed =   nn.Embedding(self.pro_emb, 32, padding_idx=0)
        self.sw_smi_embed = nn.Embedding(self.smi_emb, 32, padding_idx=0)

        self.interformer = Decoder(32, 1, 8, 512, 0.2)

        self.gat = GAT()

        self.out = nn.Sequential(
            nn.Linear(64, 512),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(512, 2)
        )

    def forward(self, compound, adj, smi_ids, prot_ids):
        sw_pro_embed = self.sw_pro_embed(prot_ids)
        sw_smi_embed = self.sw_smi_embed(smi_ids)

        inf_pro, inf_smi = self.interformer(sw_pro_embed,sw_smi_embed)

        inf_smi = inf_smi.max(dim=1)[0]
        inf_pro = inf_pro.max(dim=1)[0]

        out_fc = torch.cat([inf_smi, inf_pro], dim=1)

        return self.out(out_fc)

class cnn(nn.Module):
    def __init__(self,hid_dim, dropout, device):
        super(cnn,self).__init__()
        self.hid_dim = hid_dim

        self.batch_size = hp.batch
        self.max_drug = hp.MAX_DRUG_LEN
        self.max_protein = hp.MAX_PROTEIN_LEN
        self.pro_emb = hp.pro_emb
        self.smi_emb = hp.smi_emb

        self.dropout = dropout
        self.device = device

        self.sw_pro_embed = nn.Embedding(self.pro_emb, 64, padding_idx=0)
        self.sw_smi_embed = nn.Embedding(self.smi_emb, 64, padding_idx=0)

        self.smi_cnn = nn.Sequential(nn.Conv1d(in_channels=64,out_channels=40,kernel_size=4),
                                     nn.BatchNorm1d(40),
                                     nn.ReLU(),
                                     nn.Dropout(dropout),
                                     nn.Conv1d(in_channels=40,out_channels=80,kernel_size=6),
                                     nn.BatchNorm1d(80),
                                     nn.ReLU(),
                                     nn.Dropout(dropout),
                                     nn.Conv1d(in_channels=80,out_channels=160,kernel_size=8))
        self.smi_maxpool = nn.MaxPool1d(241)
        self.pro_cnn = nn.Sequential(nn.Conv1d(in_channels=64,out_channels=40,kernel_size=4),
                                     nn.BatchNorm1d(40),
                                     nn.ReLU(),
                                     nn.Dropout(dropout),
                                     nn.Conv1d(in_channels=40,out_channels=80,kernel_size=8),
                                     nn.BatchNorm1d(80),
                                     nn.ReLU(),
                                     nn.Dropout(dropout),
                                     nn.Conv1d(in_channels=80,out_channels=160,kernel_size=12))
        self.pro_maxpool = nn.MaxPool1d(1003)
        self.gat = GAT()

        self.out = nn.Sequential(
            nn.Linear(384, 512),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(512, 2)
        )

    def forward(self, compound, adj, smi_ids, prot_ids):
        drug_gat = self.gat(compound, adj)

        pro_embed = self.sw_pro_embed(prot_ids)
        smi_embed = self.sw_smi_embed(smi_ids)
        prot_ids=prot_ids.float()
        pro_embed = pro_embed.permute(0, 2, 1)
        smi_embed = smi_embed.permute(0, 2, 1)
        cnn_pro = self.pro_cnn(pro_embed)
        cnn_pro = self.pro_maxpool(cnn_pro).squeeze(2)
        cnn_smi = self.smi_cnn(smi_embed)
        cnn_smi = self.smi_maxpool(cnn_smi).squeeze(2)


        gat_smi = drug_gat.max(dim=1)[0]

        out_fc = torch.cat([cnn_pro, cnn_smi,gat_smi], dim=1)

        return self.out(out_fc)

class trans(nn.Module):
    def __init__(self,hid_dim, dropout, device):
        super(trans,self).__init__()
        self.hid_dim = hid_dim

        self.batch_size = hp.batch
        self.max_drug = hp.MAX_DRUG_LEN
        self.max_protein = hp.MAX_PROTEIN_LEN
        self.pro_emb = hp.pro_emb
        self.smi_emb = hp.smi_emb

        self.dropout = dropout
        self.device = device


        self.sw_pro_embed = nn.Embedding(self.pro_emb, 32, padding_idx=0)
        self.sw_smi_embed = nn.Embedding(self.smi_emb, 32, padding_idx=0)

        self.smi_cnn = transformer(32,32)
        self.pro_cnn = transformer(32,32)

        self.gat = GAT()

        self.out = nn.Sequential(
            nn.Linear(64, 512),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(512, 2)
        )

    def forward(self, compound, adj, smi_ids, prot_ids):

        sw_pro_embed = self.sw_pro_embed(prot_ids)
        sw_smi_embed = self.sw_smi_embed(smi_ids)
        prot_ids=prot_ids.float()
        cnn_pro = self.pro_cnn(sw_pro_embed)
        cnn_smi = self.smi_cnn(sw_smi_embed)

        inf_smi = cnn_smi.max(dim=1)[0]
        inf_pro = cnn_pro.max(dim=1)[0]

        out_fc = torch.cat([inf_smi, inf_pro], dim=1)

        return self.out(out_fc)

class MNDT(nn.Module):
    def __init__(self, hid_dim, dropout, device):
        super(MNDT, self).__init__()

        self.hid_dim = hid_dim

        self.batch_size = hp.batch
        self.max_drug = hp.MAX_DRUG_LEN
        self.max_protein = hp.MAX_PROTEIN_LEN
        self.pro_emb = hp.pro_emb
        self.smi_emb = hp.smi_emb

        self.dropout = dropout
        self.device = device

        self.model1 = sw_tf(self.hid_dim,self.dropout,self.device)
        self.model2 = in_tf(self.hid_dim,self.dropout,self.device)
        self.model3 = cnn(self.hid_dim,self.dropout,self.device)
        self.model4 = trans(self.hid_dim,self.dropout,self.device)


        self. out = nn.Sequential(
            nn.Linear(6, 512),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(512, 2)
        )


    def forward(self, compound, adj, smi_ids, prot_ids):
        model1 = self.model1(compound, adj, smi_ids, prot_ids)
        model2 = self.model2(compound, adj, smi_ids, prot_ids)
        model3 = self.model3(compound, adj, smi_ids, prot_ids)


        out = torch.cat([model1,model2,model3], dim=1)


        return self.out(out)

    def __call__(self, data, train=True):
        compound, adj, correct_interaction, smi_ids, prot_ids, atom_num, protein_num = data
        weight_ce = torch.FloatTensor([1, 3]).cuda()
        Loss = nn.CrossEntropyLoss(weight=weight_ce)
        if train:
            predicted_interaction = self.forward(compound, adj, smi_ids, prot_ids)
            loss = Loss(predicted_interaction, correct_interaction)
            return loss
        else:
            predicted_interaction = self.forward(compound, adj, smi_ids, prot_ids)
            correct_labels = correct_interaction.to('cpu').data.numpy()
            ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
            predicted_labels = np.argmax(ys, axis=1)
            predicted_scores = ys[:, 1]
            return correct_labels, predicted_labels, predicted_scores
