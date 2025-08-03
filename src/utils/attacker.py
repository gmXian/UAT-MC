
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from utils.utils import read_json,read_and_save,read_result,cal_evalution_metrics,compute_grad_cos_sim
import numpy as np
import pandas as pd
import shutil
from torch.autograd import Variable
import torch.nn.functional as F
from utils.topk_evaluator import TopKEvaluator

class Attacker(nn.Module):
    def __init__(self,config,rs_model,attack_v_method,attack_t_method,device,logger,seed,test_data,no_logger=False):
        super(Attacker,self).__init__()
        self.config = config
        self.logger = logger
        self.no_logger = no_logger
        self.device = device
        self.seed = str(seed)
        self.rs_model = rs_model 
        self.eval_data = test_data 
        self.evaluator = TopKEvaluator(config)
        if self.rs_model is not None:
            self.rs_model.eval()

        self.attack_v_method = attack_v_method
        self.attack_t_method = attack_t_method

        self.t_feat,self.v_feat = None,None
        self.t_feat_attacked,self.v_feat_attacked = None,None

        dataset_path = os.path.abspath(self.config['data_path'] + self.config['dataset'])

        self.v_feat_file_path = os.path.join(dataset_path, self.config['vision_feature_file'])
        self.t_feat_file_path = os.path.join(dataset_path, self.config['text_feature_file'])

        self.item2id = read_json(os.path.join(dataset_path, self.config['item2id_file']))
        self.id2item = read_json(os.path.join(dataset_path, self.config['id2item_file']))


    def pgd_attack_vt(self,attack_item_id,v_feat,t_feat,anchor_users,epsilon_ratios):
        origin_v_feature = v_feat[attack_item_id].detach().clone()
        origin_t_feature = t_feat[attack_item_id].detach().clone()
        epsilon_v = epsilon_ratios
        epsilon_t = epsilon_ratios

        adv_v_feature = origin_v_feature.detach().clone()
        adv_t_feature = origin_t_feature.detach().clone()

        adv_v_feature = adv_v_feature + torch.zeros_like(adv_v_feature) 
        adv_t_feature = adv_t_feature + torch.zeros_like(adv_t_feature)

        adv_v_feature = torch.clamp(adv_v_feature, min=origin_v_feature - epsilon_v, max=origin_v_feature + epsilon_v)
        adv_t_feature = torch.clamp(adv_t_feature, min=origin_t_feature - epsilon_t, max=origin_t_feature + epsilon_t)
        
        if not self.no_logger:
            self.logger.info(f"Attack item {attack_item_id} with epsilon_v: {epsilon_v}, epsilon_t: {epsilon_t}")
        num_iters = int(self.config['num_iters'])
        step_size_v = epsilon_v / num_iters * 1.25 
        step_size_t = epsilon_t / num_iters * 1.25
        if not self.no_logger:
            self.logger.info(f"Step size_v: {step_size_v}, Step size_t: {step_size_t}")
        for i in range(num_iters):
            adv_v_feature.requires_grad = True
            adv_t_feature.requires_grad = True
            self.rs_model.zero_grad()
           
            v_feat_attacked = v_feat.detach().clone()
            v_feat_attacked[attack_item_id] = adv_v_feature
            t_feat_attacked = t_feat.detach().clone()
            t_feat_attacked[attack_item_id] = adv_t_feature
            anchor_users_embed, all_items_embed = self.rs_model.get_user_and_item_embedding(anchor_users, v_feat_attacked, t_feat_attacked, attack_flag=True)
            if self.config['attack_v_loss_fun'] == 'max_hit_50':
                all_scores = torch.mm(anchor_users_embed, all_items_embed.T) 
                target_scores = all_scores[:, attack_item_id].unsqueeze(1)
                sorted_scores, _ = torch.sort(all_scores, dim=1, descending=True)
                threshold_scores = sorted_scores[:, 49].unsqueeze(1) 
                score_diff = target_scores - threshold_scores
                margin = 1
                loss = torch.mean(F.relu(score_diff + margin))
                if not self.no_logger:
                    self.logger.info(f"[PGD-max_hit_50] Initial Loss: {loss.item()}")
            grads = torch.autograd.grad(loss, [adv_v_feature, adv_t_feature], retain_graph=False)
            gard_v, gard_t = grads
            adv_v_feature = adv_v_feature + step_size_v * gard_v.sign()
            adv_t_feature = adv_t_feature + step_size_t * gard_t.sign()

            delta_v = torch.clamp(adv_v_feature - origin_v_feature, min=-epsilon_v, max=epsilon_v)
            delta_t = torch.clamp(adv_t_feature - origin_t_feature, min=-epsilon_t, max=epsilon_t)
            adv_v_feature = torch.clamp(origin_v_feature + delta_v, min=origin_v_feature - epsilon_v, max=origin_v_feature + epsilon_v).detach()
            adv_t_feature = torch.clamp(origin_t_feature + delta_t, min=origin_t_feature - epsilon_t, max=origin_t_feature + epsilon_t).detach()
        return adv_v_feature, adv_t_feature
    


    def fgsm_attack_vt(self, attack_item_id, v_feat, t_feat, anchor_users, epsilon_ratios):
        origin_v_feature = v_feat[attack_item_id].detach().clone()
        origin_t_feature = t_feat[attack_item_id].detach().clone()

        epsilon_v = epsilon_ratios * torch.norm(origin_v_feature, p=2)
        epsilon_t = epsilon_ratios * torch.norm(origin_t_feature, p=2)
        
        if not self.no_logger:
            self.logger.info(f"[FGSM] Attack item {attack_item_id} with epsilon_v: {epsilon_v.item()}, epsilon_t: {epsilon_t.item()}")

        origin_v_feature.requires_grad = True
        origin_t_feature.requires_grad = True

        v_feat_input = v_feat.detach().clone()
        v_feat_input[attack_item_id] = origin_v_feature
        t_feat_input = t_feat.detach().clone()
        t_feat_input[attack_item_id] = origin_t_feature

        self.rs_model.zero_grad()
        anchor_users_embed, all_items_embed = self.rs_model.get_user_and_item_embedding(
            anchor_users, v_feat_input, t_feat_input, attack_flag=True)

                
        if self.config['attack_v_loss_fun'] == 'max_hit_50':
            target_item_embed = all_items_embed[attack_item_id, :]
            all_scores = torch.mm(anchor_users_embed, all_items_embed.T)  
            target_scores = all_scores[:, attack_item_id].unsqueeze(1)  
            
            sorted_scores, _ = torch.sort(all_scores, dim=1, descending=True) 
            threshold_scores = sorted_scores[:, 49].unsqueeze(1)  
            score_diff =  target_scores - threshold_scores
            margin = 10
            loss = torch.mean(F.relu(score_diff + margin))
            if not self.no_logger:
                self.logger.info(f"[FGSM-max_hit_50] Initial Loss: {loss.item()}")
        loss.backward()
        with torch.no_grad():
            adv_v_feature = origin_v_feature + epsilon_v * origin_v_feature.grad.sign()
            adv_t_feature = origin_t_feature + epsilon_t * origin_t_feature.grad.sign()
            adv_v_feature = torch.clamp(adv_v_feature, min=origin_v_feature - epsilon_v, max=origin_v_feature + epsilon_v)
            adv_t_feature = torch.clamp(adv_t_feature, min=origin_t_feature - epsilon_t, max=origin_t_feature + epsilon_t)
        return adv_v_feature, adv_t_feature



    def forward(self,anchor_users,all_users,attack_item_id,attack_modal,results_t,results_vt,k):
        self.t_feat_attacked,self.v_feat_attacked = None,None
        self.attack_modal = attack_modal
       
        if 'MLP' in self.config['model']:
            self.v_feat,self.t_feat = self.rs_model.align_forward() 
        else:
            self.v_feat = torch.from_numpy(np.load(self.v_feat_file_path, allow_pickle=True)).type(torch.FloatTensor).to(self.device)
            self.t_feat = torch.from_numpy(np.load(self.t_feat_file_path, allow_pickle=True)).type(torch.FloatTensor).to(self.device)
        
 
        if attack_modal == "vt":
            save_v_path = self.config['save_v_path']  
            save_t_path = self.config['save_t_path'] 
            save_vt_path = self.config['save_vt_path']  
            epsilon_ratios = self.config['epsilon']
            saved_item_vt = read_result(save_vt_path)
            results_vt[attack_item_id]['attacked_text_feat_path'] = save_t_path + f"{self.id2item[str(attack_item_id)]}.npy"
            results_vt[attack_item_id]['attacked_image_feat_path'] = save_v_path + f"{self.id2item[str(attack_item_id)]}.npy"
            if self.config['attack_method'] == 'pgd':
                item_image_feat_attacked, item_text_feat_attacked = self.pgd_attack_vt(attack_item_id, self.v_feat, self.t_feat, anchor_users, epsilon_ratios)
                
            elif self.config['attack_method'] == 'fgsm':
                item_image_feat_attacked, item_text_feat_attacked = self.fgsm_attack_vt(attack_item_id, self.v_feat, self.t_feat, anchor_users, epsilon_ratios)

            self.t_feat_attacked = self.t_feat.clone()
            self.t_feat_attacked[attack_item_id] = item_text_feat_attacked.clone()
            self.v_feat_attacked = self.v_feat.clone()
            self.v_feat_attacked[attack_item_id] = item_image_feat_attacked.clone()
            if self.config['test_use_all_user'] == '1':
                self.cal_vt_result(attack_item_id,all_users,results_vt,k,v_feat_attacked=self.v_feat_attacked,t_feat_attacked=self.t_feat_attacked)
            else:
                self.cal_vt_result(attack_item_id,anchor_users,results_vt,k,v_feat_attacked=self.v_feat_attacked,t_feat_attacked=self.t_feat_attacked)
            
            if not os.path.exists(results_vt[attack_item_id]['attacked_text_feat_path']):
                np.save(results_vt[attack_item_id]['attacked_text_feat_path'], item_text_feat_attacked.detach().cpu().numpy())
            if not os.path.exists(results_vt[attack_item_id]['attacked_image_feat_path']):
                np.save(results_vt[attack_item_id]['attacked_image_feat_path'], item_image_feat_attacked.detach().cpu().numpy())
            read_and_save(results_vt, save_vt_path)
            if not self.no_logger:
                self.logger.info(f"Save the result of item {attack_item_id} to {save_vt_path}")


        
    def cal_vt_result(self,attack_item_id,anchor_users,results_vt,k,v_feat_attacked=None,t_feat_attacked=None):
        if not self.no_logger:
            self.logger.info(f"Attacked ItemID: {attack_item_id}, Attack Modal: {self.config['attack_modal']}")
        with torch.no_grad():
            score_matrix_before_attack,score_matrix_after_attack = self.rs_model.enjoy_attack(anchor_users,
                                                                                            attack_item_id,
                                                                                            self.v_feat,
                                                                                            self.t_feat,
                                                                                            v_feat_attacked = v_feat_attacked,
                                                                                            t_feat_attacked = t_feat_attacked)
        results_vt[attack_item_id]['avg_rank_before'] ,results_vt[attack_item_id]['avg_rank_after'],\
            results_vt[attack_item_id]['avg_rank_change'] = cal_top_k(attack_item_id,score_matrix_before_attack,score_matrix_after_attack,anchor_users)
        all_metrics_before_dict = {}
        all_metrics_after_dict = {}
        for i in k:
            all_metrics_before_dict[f"hit_rate@{i}"] = 0
            all_metrics_after_dict[f"hit_rate@{i}"] = 0
        hit_rate_before = cal_evalution_metrics(score_matrix_before_attack, attack_item_id, all_metrics_before_dict,k=k)
        hit_rate_after = cal_evalution_metrics(score_matrix_after_attack,attack_item_id,all_metrics_after_dict,k=k) 
        all_metrics_before_list = []
        all_metrics_list = []
        for i in k:
            all_metrics_before_dict[f"hit_rate@{i}"] /= len(anchor_users)
            all_metrics_after_dict[f"hit_rate@{i}"] /= len(anchor_users)
            all_metrics_before_list.append(all_metrics_before_dict[f"hit_rate@{i}"])
            all_metrics_list.append(all_metrics_after_dict[f"hit_rate@{i}"])

        results_vt[attack_item_id]["hit_rate_before"] = all_metrics_before_dict
        results_vt[attack_item_id]["hit_rate_after"] = all_metrics_after_dict

        self.config["goal_k"] = '50'
        results_vt[attack_item_id]['init_anchor_acc'] = results_vt[attack_item_id]["hit_rate_before"][f'hit_rate@{self.config["goal_k"]}']
        results_vt[attack_item_id]['attacked_anchor_acc'] = results_vt[attack_item_id]["hit_rate_after"][f'hit_rate@{self.config["goal_k"]}']
        results_vt[attack_item_id]['improved_anchor_acc'] = results_vt[attack_item_id]["hit_rate_after"][f'hit_rate@{self.config["goal_k"]}'] - results_vt[attack_item_id]["hit_rate_before"][f'hit_rate@{self.config["goal_k"]}']
        if not self.no_logger:
            self.logger.info(f"ItemID:{attack_item_id} avg_rank_before: {results_vt[attack_item_id]['avg_rank_before']:.2f}, avg_rank_after: {results_vt[attack_item_id]['avg_rank_after']:.2f}, rank_change: {results_vt[attack_item_id]['avg_rank_change']:.2f}")
            self.logger.info(f"ItemID:{attack_item_id} hit_rate_before:{results_vt[attack_item_id]['hit_rate_before']}")
            self.logger.info(f"ItemID:{attack_item_id} hit_rate_after:{results_vt[attack_item_id]['hit_rate_after']}")


    @torch.no_grad()
    def evaluate_promotion_attack(self, attack_item_id,v_feat_attacked,t_feat_attacked, is_test=False, idx=0):
        self.rs_model.eval()
        batch_matrix_list_before_attack = []
        batch_matrix_list_after_attack = []
        for batch_idx, batched_data in enumerate(self.eval_data):
            before_attack_scores, after_attack_scores = self.rs_model.enjoy_attack(batched_data[0],
                                                                                   attack_item_id,
                                                                                    self.v_feat,
                                                                                    self.t_feat,
                                                                                    v_feat_attacked = v_feat_attacked,
                                                                                    t_feat_attacked = t_feat_attacked)
            masked_items = batched_data[1]
            before_attack_scores[masked_items[0], masked_items[1]] = -1e10
            after_attack_scores[masked_items[0], masked_items[1]] = -1e10
            _, topk_index_before_attack = torch.topk(before_attack_scores, max(self.config['topk']), dim=-1) 
            _, topk_index_after_attack = torch.topk(after_attack_scores, max(self.config['topk']), dim=-1)  

            batch_matrix_list_before_attack.append(topk_index_before_attack)
            batch_matrix_list_after_attack.append(topk_index_after_attack)
        return self.evaluator.evaluate(batch_matrix_list_before_attack, self.eval_data, is_test=is_test, idx=idx), self.evaluator.evaluate(batch_matrix_list_after_attack, self.eval_data, is_test=is_test, idx=idx)



def cal_top_k(attack_item_id,score_matrix_before_attack,score_matrix_after_attack,anchor_users):
    attack_item_scores_before = score_matrix_before_attack[:, attack_item_id].unsqueeze(1) 
    attack_item_scores_after = score_matrix_after_attack[:, attack_item_id].unsqueeze(1) 
    ranks_before = (score_matrix_before_attack > attack_item_scores_before).sum(dim=1)  
    ranks_after = (score_matrix_after_attack > attack_item_scores_after).sum(dim=1)   
    avg_rank_before = ranks_before.float().sum().item()
    avg_rank_after = ranks_after.float().sum().item()
    avg_rank_change = avg_rank_before- avg_rank_after

    return avg_rank_before, avg_rank_after, avg_rank_change