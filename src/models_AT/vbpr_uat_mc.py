# coding: utf-8

import numpy as np
import os
import torch
import torch.nn as nn

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss, InfoNCELoss
from common.init import xavier_normal_initialization
import torch.nn.functional as F


class VBPR_UAT_MC(GeneralRecommender):
    r"""BPR is a basic matrix factorization model that be trained in the pairwise way.
    """
    def __init__(self, config, dataloader,logger=None):
        super(VBPR_UAT_MC, self).__init__(config, dataloader)

        # load parameters info
        if logger is not None:
            self.logger = logger
        self.config = config
        self.u_embedding_size = self.i_embedding_size = config['embedding_size']
        self.reg_weight = config['reg_weight']  # float32 type: the weight decay for l2 normalizaton
        self.align_weight = config['align_weight'] # float32 type: the weight for semantic alignment loss
        self.adv_weight = config['adv_weight'] # float32 type: the weight for adversarial loss
        self.adv_align_weight = config['adv_align_weight'] # float32 type: the weight for adversarial alignment loss

        # define layers and loss
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
    def load_normal_trained_state(self,logger, path='./checkpoints_saves/VBPR_baby_best_model.pth'):
        path = f"./checkpoints_saves/VBPR_{self.config['dataset']}_best_model.pth"
        logger.info('Loading the normal trained state from %s' % path)
        checkpoint = torch.load(path)['state_dict'] 
        self.load_state_dict(checkpoint, strict=False)


    def get_user_embedding(self, user):
        return self.u_embedding[user, :]

    def get_item_embedding(self, item):
        return self.item_embedding[item, :]
    
    def calculate_BPR_loss(self, interaction,perturbed_v_feat=None, perturbed_t_feat=None):
        user = interaction[0]
        pos_item = interaction[1]
        neg_item = interaction[2]

        user_embeddings, item_embeddings,vt_similarity = self.forward(perturbed_v_feat, perturbed_t_feat)
        user_e = user_embeddings[user, :]
        pos_e = item_embeddings[pos_item, :]
        neg_e = item_embeddings[neg_item, :]
        pos_item_score, neg_item_score = torch.mul(user_e, pos_e).sum(dim=1), torch.mul(user_e, neg_e).sum(dim=1)
        mf_loss = self.loss(pos_item_score, neg_item_score)
        return mf_loss
    
    def generate_adversarial_noise(self, interaction):
        self.aligned_item_t_features,self.aligned_item_v_features = self.align_forward()
        epsilon_ratios_v = epsilon_ratios_t = self.config['epsilons'] 

        attack_goal = self.config['attack_goal'] if 'attack_goal' in self.config else attack_goal
        if attack_goal == "1":
            attack_method = 'fgsm_first_order_derivative' 
            delta_t = torch.zeros_like(self.aligned_item_t_features,requires_grad=True)
            delta_v = torch.zeros_like(self.aligned_item_v_features,requires_grad=True)
            epsilon_t = epsilon_ratios_t * torch.norm(self.aligned_item_t_features, p=2, dim=1, keepdim=True)
            epsilon_v = epsilon_ratios_v * torch.norm(self.aligned_item_v_features, p=2, dim=1, keepdim=True)

            perturbed_feat_t = self.aligned_item_t_features + delta_t
            perturbed_feat_v = self.aligned_item_v_features + delta_v
            perturbed_feat_t = torch.clamp(perturbed_feat_t, 
                                            self.aligned_item_t_features - epsilon_t, 
                                            self.aligned_item_t_features + epsilon_t)
            perturbed_feat_v = torch.clamp(perturbed_feat_v, 
                                            self.aligned_item_v_features - epsilon_v, 
                                            self.aligned_item_v_features + epsilon_v)
            bpr_loss = self.calculate_BPR_loss(interaction, perturbed_feat_v, perturbed_feat_t)
            grad_v = torch.autograd.grad(bpr_loss, delta_v, create_graph=True)[0]
            grad_t = torch.autograd.grad(bpr_loss, delta_t, create_graph=True)[0]
            cosine_sim = F.cosine_similarity(grad_v, grad_t, dim=1)
            cosine_sim = torch.mean(cosine_sim)
            total_loss = bpr_loss + self.adv_align_weight * cosine_sim
            total_loss.backward(retain_graph=False)
            
            with torch.no_grad():
                norm_t = torch.norm(delta_t.grad, p=2, dim=1, keepdim=True)
                delta_t.data.add_(epsilon_t * delta_t.grad / (norm_t + 1e-8)) 
                delta_t.data = torch.clamp(delta_t, -epsilon_t, epsilon_t)
                delta_t.grad.zero_()

                norm_v = torch.norm(delta_v.grad, p=2, dim=1, keepdim=True)
                delta_v.data.add_(epsilon_v * delta_v.grad / (norm_v + 1e-8)) 
                delta_v.data = torch.clamp(delta_v, -epsilon_v, epsilon_v)
                delta_v.grad.zero_()


            final_perturbation_t = delta_t.detach()
            final_perturbation_v = delta_v.detach()
        return final_perturbation_v, final_perturbation_t
        
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
    
    def AT_forward(self, interaction,aligned_item_v_features=None, aligned_item_t_features=None, dropout=0.0):
        if aligned_item_v_features is None or aligned_item_t_features is None:
            aligned_item_v_features = self.item_v_proj(self.item_raw_v_features)
            aligned_item_v_features = F.normalize(aligned_item_v_features,dim=-1)
            aligned_item_t_features = self.item_t_proj(self.item_raw_t_features)
            aligned_item_t_features = F.normalize(aligned_item_t_features,dim=-1)

        aligned_item_features = torch.cat((aligned_item_v_features,aligned_item_t_features),dim=1)
        item_embedding = self.item_linear(aligned_item_features)
        item_embedding = torch.cat((self.i_embedding,item_embedding),dim=-1)        

        v_feat_adv_noise, t_feat_adv_noise = self.generate_adversarial_noise(interaction)
        aligned_item_v_features = aligned_item_v_features + v_feat_adv_noise
        aligned_item_t_features = aligned_item_t_features + t_feat_adv_noise
        noisy_aligned_item_features = torch.cat((aligned_item_v_features,aligned_item_t_features),dim=1)
        noisy_item_e = self.item_linear(noisy_aligned_item_features)
        noisy_item_e = torch.cat((self.i_embedding,noisy_item_e),dim=-1)
        
        user_e = F.dropout(self.u_embedding,dropout)
        item_e = F.dropout(item_embedding,dropout)
        return user_e, item_e, noisy_item_e
    


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

        self.aligned_item_t_features,self.aligned_item_v_features = self.align_forward()
        user_embeddings, item_embeddings,noisy_item_embedding = self.AT_forward(interaction,self.aligned_item_v_features, self.aligned_item_t_features)
        user_e = user_embeddings[user, :]
        pos_e = item_embeddings[pos_item, :]
        neg_e = item_embeddings[neg_item, :]
        noisy_pos_e = noisy_item_embedding[pos_item, :]
        noisy_neg_e = noisy_item_embedding[neg_item, :]
        pos_item_score, neg_item_score = torch.mul(user_e, pos_e).sum(dim=1), torch.mul(user_e, neg_e).sum(dim=1)
        mf_loss = self.loss(pos_item_score, neg_item_score)
        noisy_pos_item_score, noisy_neg_item_score = torch.mul(user_e, noisy_pos_e).sum(dim=1), torch.mul(user_e, noisy_neg_e).sum(dim=1)
        noisy_mf_loss = self.adv_weight * self.loss(noisy_pos_item_score, noisy_neg_item_score)
        reg_loss = self.reg_loss(user_e, pos_e, neg_e)
        loss = mf_loss + noisy_mf_loss + self.reg_weight * reg_loss
        return loss, mf_loss, self.reg_weight * reg_loss,noisy_mf_loss

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

