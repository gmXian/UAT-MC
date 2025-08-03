# coding: utf-8


import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, degree
import torch_geometric

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss, InfoNCELoss
from common.init import xavier_uniform_initialization


class MMGCN_UAT_MC(GeneralRecommender):
    def __init__(self, config, dataset,logger=None):
        super(MMGCN_UAT_MC, self).__init__(config, dataset)
        self.num_user = self.n_users
        self.num_item = self.n_items
        num_user = self.n_users
        num_item = self.n_items
        dim_x = self.u_embedding_size = self.i_embedding_size = config['embedding_size']
        num_layer = config['n_layers']
        batch_size = config['train_batch_size']         # not used
        self.aggr_mode = 'mean'
        self.concate = 'False'
        has_id = True
        self.weight = torch.tensor([[1.0], [-1.0]]).to(self.device)
        self.reg_weight = config['reg_weight']
        self.align_weight = config['align_weight']
        self.adv_weight = config['adv_weight']
        self.adv_align_weight = config['adv_align_weight']
        # load parameters info
        if logger is not None:
            self.logger = logger
        self.config = config

        # packing interaction in training into edge_index
        train_interactions = dataset.inter_matrix(form='coo').astype(np.float32)
        edge_index = torch.tensor(self.pack_edge_index(train_interactions), dtype=torch.long)
        self.edge_index = edge_index.t().contiguous().to(self.device)
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1, 0]]), dim=1)
        self.num_modal = 0

        if self.v_feat is not None: 
            self.v_gcn = GCN(self.edge_index, batch_size, num_user, num_item, dim_x, dim_x, self.aggr_mode,
                             self.concate, num_layer=num_layer, has_id=has_id, dim_latent=256, device=self.device)
            self.num_modal += 1

        if self.t_feat is not None: 
            self.t_gcn = GCN(self.edge_index, batch_size, num_user, num_item, dim_x, dim_x,
                             self.aggr_mode, self.concate, num_layer=num_layer, has_id=has_id, device=self.device)
            self.num_modal += 1

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

        # self.id_embedding = nn.init.xavier_normal_(torch.rand((num_user+num_item, dim_x), requires_grad=True)).to(self.device)
        self.id_embedding = nn.Parameter(nn.init.xavier_normal_(torch.rand((num_user + num_item, dim_x), requires_grad=True)).to(self.device))
        self.result = nn.init.xavier_normal_(torch.rand((num_user + num_item, dim_x))).to(self.device)

        self.align_loss = InfoNCELoss()
    def load_normal_trained_state(self,logger, path='./checkpoints_saves/MMGCN_baby_best_model.pth'):
        path = f"./checkpoints_saves/MMGCN_{self.config['dataset']}_best_model.pth"
        logger.info('Loading the normal trained state from %s' % path)
        checkpoint = torch.load(path)['state_dict'] 
        self.load_state_dict(checkpoint, strict=False)


    def calculate_BPR_loss(self, interaction,perturbed_v_feat=None, perturbed_t_feat=None):
        batch_users = interaction[0]
        pos_items = interaction[1] + self.n_users
        neg_items = interaction[2] + self.n_users

        user_tensor = batch_users.repeat_interleave(2)
        stacked_items = torch.stack((pos_items, neg_items))
        item_tensor = stacked_items.t().contiguous().view(-1)

        representation = self.v_gcn(perturbed_v_feat, self.id_embedding)
        representation += self.t_gcn(perturbed_t_feat, self.id_embedding)
        representation /= self.num_modal
        out = representation

        user_score = out[user_tensor]
        item_score = out[item_tensor]
        score = torch.sum(user_score * item_score, dim=1).view(-1, 2)
        loss = -torch.mean(torch.log(torch.sigmoid(torch.matmul(score, self.weight))))
        return loss

    def pack_edge_index(self, inter_mat):
        rows = inter_mat.row
        cols = inter_mat.col + self.n_users
        # ndarray([598918, 2]) for ml-imdb
        return np.column_stack((rows, cols))
    
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
            bpr_loss = self.calculate_BPR_loss(interaction,perturbed_feat_v,perturbed_feat_t)
            grad_v = torch.autograd.grad(bpr_loss, delta_v, retain_graph=True)[0]
            grad_t = torch.autograd.grad(bpr_loss, delta_t, retain_graph=True)[0]
            cosine_sim = F.cosine_similarity(grad_v, grad_t, dim=1).mean()
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
        
    def forward(self,aligned_item_v_features=None, aligned_item_t_features=None):
        if aligned_item_v_features is None or aligned_item_t_features is None:
            aligned_item_v_features = self.item_v_proj(self.item_raw_v_features)
            aligned_item_v_features = F.normalize(aligned_item_v_features,dim=-1)
            aligned_item_t_features = self.item_t_proj(self.item_raw_t_features)
            aligned_item_t_features = F.normalize(aligned_item_t_features,dim=-1)

        similarity = aligned_item_v_features @ aligned_item_t_features.T
        similarity *= self.temperature.exp()

        representation = None
        representation = self.v_gcn(aligned_item_v_features, self.id_embedding)
        representation += self.t_gcn(aligned_item_t_features, self.id_embedding)

        representation /= self.num_modal

        self.result = representation
        return representation,similarity
    
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

        representation = None
        representation = self.v_gcn(aligned_item_v_features, self.id_embedding)
        representation += self.t_gcn(aligned_item_t_features, self.id_embedding)
        representation /= self.num_modal

        v_feat_adv_noise, t_feat_adv_noise = self.generate_adversarial_noise(interaction)
        aligned_item_v_features = aligned_item_v_features + v_feat_adv_noise
        aligned_item_t_features = aligned_item_t_features + t_feat_adv_noise
        noisy_representation = None
        noisy_representation = self.v_gcn(aligned_item_v_features, self.id_embedding)
        noisy_representation += self.t_gcn(aligned_item_t_features, self.id_embedding)
        noisy_representation /= self.num_modal

        return representation, noisy_representation

    def calculate_loss(self, interaction):
        batch_users = interaction[0]
        pos_items = interaction[1] + self.n_users
        neg_items = interaction[2] + self.n_users

        user_tensor = batch_users.repeat_interleave(2)
        stacked_items = torch.stack((pos_items, neg_items))
        item_tensor = stacked_items.t().contiguous().view(-1)

        self.aligned_item_t_features,self.aligned_item_v_features = self.align_forward()
        out,noisy_out = self.AT_forward(interaction, aligned_item_v_features=self.aligned_item_v_features, aligned_item_t_features=self.aligned_item_t_features)

        user_score = out[user_tensor]
        item_score = out[item_tensor]
        score = torch.sum(user_score * item_score, dim=1).view(-1, 2)
        loss = -torch.mean(torch.log(torch.sigmoid(torch.matmul(score, self.weight))))
        reg_embedding_loss = (self.id_embedding[user_tensor]**2 + self.id_embedding[item_tensor]**2).mean()
        if self.v_feat is not None:
            reg_embedding_loss += (self.v_gcn.preference**2).mean()
        reg_loss = self.reg_weight * reg_embedding_loss

        noisy_user_score = noisy_out[user_tensor]
        noisy_item_score = noisy_out[item_tensor]
        noisy_score = torch.sum(noisy_user_score * noisy_item_score, dim=1).view(-1, 2)
        noisy_loss = -torch.mean(torch.log(torch.sigmoid(torch.matmul(noisy_score, self.weight))))
        all_loss = loss + reg_loss + self.adv_weight * noisy_loss

        return all_loss, loss, reg_loss, noisy_loss

    def full_sort_predict(self, interaction):
        out,_ = self.forward()
        user_tensor = out[:self.n_users]
        item_tensor = out[self.n_users:]

        temp_user_tensor = user_tensor[interaction[0], :]
        score_matrix = torch.matmul(temp_user_tensor, item_tensor.t())
        return score_matrix
    
    def get_user_and_item_embedding(self, anchor_users, v_feat, t_feat, attack_flag=False):
        out,_ = self.forward(v_feat, t_feat)
        user_tensor = out[:self.n_users]
        item_tensor = out[self.n_users:]
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
        
        out,_ = self.forward()
        attack_out,_ = self.forward(aligned_item_v_features=self.v_feat_attacked, aligned_item_t_features=self.t_feat_attacked)
        user_tensor_before_attack = out[:self.n_users]
        item_tensor_before_attack = out[self.n_users:]
        user_tensor_after_attack = attack_out[:self.n_users]
        item_tensor_after_attack = attack_out[self.n_users:]
        temp_user_tensor_before_attack = user_tensor_before_attack[anchor_users, :]
        temp_user_tensor_after_attack = user_tensor_after_attack[anchor_users, :]
        score_matrix_before_attack = torch.matmul(temp_user_tensor_before_attack, item_tensor_before_attack.t())
        score_matrix_after_attack = torch.matmul(temp_user_tensor_after_attack, item_tensor_after_attack.t())
        return score_matrix_before_attack,score_matrix_after_attack
    

    




class GCN(torch.nn.Module):
    def __init__(self, edge_index, batch_size, num_user, num_item, dim_feat, dim_id, aggr_mode, concate, num_layer,
                 has_id, dim_latent=None, device='cpu'):
        super(GCN, self).__init__()
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.dim_id = dim_id
        self.dim_feat = dim_feat
        self.dim_latent = dim_latent
        self.edge_index = edge_index
        self.aggr_mode = aggr_mode
        self.concate = concate
        self.num_layer = num_layer
        self.has_id = has_id
        self.device = device

        if self.dim_latent:
            self.preference = nn.init.xavier_normal_(torch.rand((num_user, self.dim_latent), requires_grad=True)).to(self.device)
            #self.preference = nn.Parameter(nn.init.xavier_normal_(torch.rand((num_user, self.dim_latent))))

            self.MLP = nn.Linear(self.dim_feat, self.dim_latent)
            self.conv_embed_1 = BaseModel(self.dim_latent, self.dim_latent, aggr=self.aggr_mode)
            nn.init.xavier_normal_(self.conv_embed_1.weight)
            self.linear_layer1 = nn.Linear(self.dim_latent, self.dim_id)
            nn.init.xavier_normal_(self.linear_layer1.weight)
            self.g_layer1 = nn.Linear(self.dim_latent + self.dim_id, self.dim_id) if self.concate else nn.Linear(
                self.dim_latent, self.dim_id)
            nn.init.xavier_normal_(self.g_layer1.weight)

        else:
            self.preference = nn.init.xavier_normal_(torch.rand((num_user, self.dim_feat), requires_grad=True)).to(self.device)
            #self.preference = nn.Parameter(nn.init.xavier_normal_(torch.rand((num_user, self.dim_feat))))

            self.conv_embed_1 = BaseModel(self.dim_feat, self.dim_feat, aggr=self.aggr_mode)
            nn.init.xavier_normal_(self.conv_embed_1.weight)
            self.linear_layer1 = nn.Linear(self.dim_feat, self.dim_id)
            nn.init.xavier_normal_(self.linear_layer1.weight)
            self.g_layer1 = nn.Linear(self.dim_feat + self.dim_id, self.dim_id) if self.concate else nn.Linear(
                self.dim_feat, self.dim_id)
            nn.init.xavier_normal_(self.g_layer1.weight)

        self.conv_embed_2 = BaseModel(self.dim_id, self.dim_id, aggr=self.aggr_mode)
        nn.init.xavier_normal_(self.conv_embed_2.weight)
        self.linear_layer2 = nn.Linear(self.dim_id, self.dim_id)
        nn.init.xavier_normal_(self.linear_layer2.weight)
        self.g_layer2 = nn.Linear(self.dim_id + self.dim_id, self.dim_id) if self.concate else nn.Linear(self.dim_id,
                                                                                                         self.dim_id)

        self.conv_embed_3 = BaseModel(self.dim_id, self.dim_id, aggr=self.aggr_mode)
        nn.init.xavier_normal_(self.conv_embed_3.weight)
        self.linear_layer3 = nn.Linear(self.dim_id, self.dim_id)
        nn.init.xavier_normal_(self.linear_layer3.weight)
        self.g_layer3 = nn.Linear(self.dim_id + self.dim_id, self.dim_id) if self.concate else nn.Linear(self.dim_id,
                                                                                                         self.dim_id)

    def forward(self, features, id_embedding):
        temp_features = self.MLP(features) if self.dim_latent else features

        x = torch.cat((self.preference, temp_features), dim=0)
        x = F.normalize(x)

        h = F.leaky_relu(self.conv_embed_1(x, self.edge_index))  # equation 1
        x_hat = F.leaky_relu(self.linear_layer1(x)) + id_embedding if self.has_id else F.leaky_relu(
            self.linear_layer1(x))  # equation 5
        x = F.leaky_relu(self.g_layer1(torch.cat((h, x_hat), dim=1))) if self.concate else F.leaky_relu(
            self.g_layer1(h) + x_hat)

        h = F.leaky_relu(self.conv_embed_2(x, self.edge_index))  # equation 1
        x_hat = F.leaky_relu(self.linear_layer2(x)) + id_embedding if self.has_id else F.leaky_relu(
            self.linear_layer2(x))  # equation 5
        x = F.leaky_relu(self.g_layer2(torch.cat((h, x_hat), dim=1))) if self.concate else F.leaky_relu(
            self.g_layer2(h) + x_hat)

        h = F.leaky_relu(self.conv_embed_3(x, self.edge_index))  # equation 1
        x_hat = F.leaky_relu(self.linear_layer3(x)) + id_embedding if self.has_id else F.leaky_relu(
            self.linear_layer3(x))  # equation 5
        x = F.leaky_relu(self.g_layer3(torch.cat((h, x_hat), dim=1))) if self.concate else F.leaky_relu(
            self.g_layer3(h) + x_hat)

        return x


class BaseModel(MessagePassing):
    def __init__(self, in_channels, out_channels, normalize=True, bias=True, aggr='add', **kwargs):
        super(BaseModel, self).__init__(aggr=aggr, **kwargs)
        self.aggr = aggr
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.weight = nn.Parameter(torch.Tensor(self.in_channels, out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        torch_geometric.nn.inits.uniform(self.in_channels, self.weight)

    def forward(self, x, edge_index, size=None):
        x = torch.matmul(x, self.weight)
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        return x_j

    def update(self, aggr_out):
        return aggr_out

    def __repr(self):
        return '{}({},{})'.format(self.__class__.__name__, self.in_channels, self.out_channels)