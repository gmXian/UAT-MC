# coding: utf-8

import numpy as np
import os
import torch
import torch.nn as nn

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss, InfoNCELoss
from common.init import xavier_normal_initialization
import torch.nn.functional as F


class VBPR(GeneralRecommender):
    r"""BPR is a basic matrix factorization model that be trained in the pairwise way.
    """
    def __init__(self, config, dataloader):
        super(VBPR, self).__init__(config, dataloader)

        self.u_embedding_size = self.i_embedding_size = config['embedding_size']
        self.reg_weight = config['reg_weight']  
        self.align_weight = config['align_weight'] 

        self.u_embedding = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_users, self.u_embedding_size * 2)))
        self.i_embedding = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_items, self.i_embedding_size)))
        
        self.item_raw_v_features = self.v_feat
        self.item_raw_t_features = self.t_feat
        
        self.item_v_proj = nn.Sequential(
            nn.Linear(self.item_raw_v_features.shape[1],1024),
            nn.ReLU(),
            nn.LayerNorm(1024),
            nn.Linear(1024,256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, self.i_embedding_size)
        )
        self.item_t_proj = nn.Sequential(
            nn.Linear(self.item_raw_t_features.shape[1],256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256,self.i_embedding_size)
        )
        self.temperature = nn.Parameter(torch.tensor(0.07))

        self.item_linear = nn.Linear(self.i_embedding_size*2,self.i_embedding_size)

        self.loss = BPRLoss()
        self.reg_loss = EmbLoss()
        self.align_loss = InfoNCELoss()

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def get_user_embedding(self, user):
        return self.u_embedding[user, :]

    def get_item_embedding(self, item):
        return self.item_embedding[item, :]

    def forward(self,aligned_item_v_features=None, aligned_item_t_features=None,dropout=0.0):
        if aligned_item_v_features is None or aligned_item_t_features is None:
            aligned_item_v_features = self.item_v_proj(self.item_raw_v_features)
            aligned_item_v_features = F.normalize(aligned_item_v_features,dim=-1)
            aligned_item_t_features = self.item_t_proj(self.item_raw_t_features)
            aligned_item_t_features = F.normalize(aligned_item_t_features,dim=-1)
        
        similarity = aligned_item_v_features @ aligned_item_t_features.T
        similarity *= self.temperature.exp()

        aligned_item_features = torch.cat((aligned_item_v_features,aligned_item_t_features),dim=1)
        item_embedding = self.item_linear(aligned_item_features)
        item_embedding = torch.cat((self.i_embedding,item_embedding),dim=-1)

        user_e = F.dropout(self.u_embedding,dropout)
        item_e = F.dropout(item_embedding,dropout)
        return user_e,item_e,similarity

    def align_forward(self):
        aligned_item_v_features = self.item_v_proj(self.item_raw_v_features)
        aligned_item_v_features = F.normalize(aligned_item_v_features,dim=-1)
        aligned_item_t_features = self.item_t_proj(self.item_raw_t_features)
        aligned_item_t_features = F.normalize(aligned_item_t_features,dim=-1)
        return aligned_item_v_features, aligned_item_t_features
    


    def calculate_loss(self, interaction):
        """
        loss on one batch
        :param interaction:
            batch data format: tensor(3, batch_size)
            [0]: user list; [1]: positive items; [2]: negative items
        :return:
        """
        user = interaction[0]
        pos_item = interaction[1]
        neg_item = interaction[2]

        user_embeddings, item_embeddings,vt_similarity = self.forward()
        user_e = user_embeddings[user, :]
        pos_e = item_embeddings[pos_item, :]
        neg_e = item_embeddings[neg_item, :]
        pos_item_score, neg_item_score = torch.mul(user_e, pos_e).sum(dim=1), torch.mul(user_e, neg_e).sum(dim=1)
        mf_loss = self.loss(pos_item_score, neg_item_score)
        reg_loss = self.reg_loss(user_e, pos_e, neg_e)
        align_loss = self.align_loss(vt_similarity)
        loss = mf_loss + self.reg_weight * reg_loss + self.align_weight * align_loss
        return loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_embeddings, item_embeddings,_ = self.forward()
        user_e = user_embeddings[user, :]
        all_item_e = item_embeddings
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score
    
    def get_user_and_item_embedding(self, anchor_users, v_feat, t_feat, attack_flag=False):
        user_tensor, item_tensor,_ = self.forward(v_feat, t_feat)
        temp_user_tensor = user_tensor[anchor_users, :]
        return temp_user_tensor, item_tensor
    
    def enjoy_attack(self, anchor_users, attack_item_id, v_feat, t_feat, v_feat_attacked, t_feat_attacked):
        if v_feat_attacked is not None:
            self.v_feat_attacked = v_feat_attacked
        else:
            self.v_feat_attacked = v_feat
        if t_feat_attacked is not None:
            self.t_feat_attacked = t_feat_attacked
        else:
            self.t_feat_attacked = t_feat
        
        user_tensor_before_attack, item_tensor_before_attack,_ = self.forward()
        user_tensor_after_attack, item_tensor_after_attack,_ = self.forward(aligned_item_v_features=self.v_feat_attacked, aligned_item_t_features=self.t_feat_attacked)
        temp_user_tensor_before_attack = user_tensor_before_attack[anchor_users, :]
        temp_user_tensor_after_attack = user_tensor_after_attack[anchor_users, :]
        score_matrix_before_attack = torch.matmul(temp_user_tensor_before_attack, item_tensor_before_attack.t())
        score_matrix_after_attack = torch.matmul(temp_user_tensor_after_attack, item_tensor_after_attack.t())
        return score_matrix_before_attack,score_matrix_after_attack

