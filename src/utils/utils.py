# coding: utf-8
# @email  : enoche.chow@gmail.com

"""
Utility functions
##########################
"""

import numpy as np
import torch
import importlib
import datetime
import random
import json
import os
import torch.nn.functional as F

def compute_grad_cos_sim(gard_v, gard_t):
    """
    计算两个模态梯度的余弦相似度
    """
    grad_v_flat = gard_v.view(-1)
    grad_t_flat = gard_t.view(-1)
    cos_sim = F.cosine_similarity(grad_v_flat.unsqueeze(0), grad_t_flat.unsqueeze(0)).item()
    return cos_sim

def set_path(config,seed):
    encode_map = {
    "openai/clip-vit-base-patch16": "CLIP",
    "sentence-transformers/all-mpnet-base-v2":"Sentence",
    "MMFashion":"MMFashion"
    }
    if config['target_model'] is None: # 白盒攻击
        if not config['attack_cold_start'] and not config['attack_hot_start']:
            if 'best' not in config['save_epoch']:
                config['result_folder'] = f"./results/white_attack_100/{config['model']}.{config['dataset']}.{config['save_epoch']}.{encode_map[config['v_encoder_name']]}.{encode_map[config['t_encoder_name']]}/"
            else:
                config['result_folder'] = f"./results/white_attack_100/{config['model']}.{config['dataset']}.{encode_map[config['v_encoder_name']]}.{encode_map[config['t_encoder_name']]}/"
        elif config['attack_hot_start']:
            if 'best' not in config['save_epoch']:
                config['result_folder'] = f"./results/white_attack_hot/{config['model']}.{config['dataset']}.{config['save_epoch']}.{encode_map[config['v_encoder_name']]}.{encode_map[config['t_encoder_name']]}/"
            else:
                config['result_folder'] = f"./results/white_attack_hot/{config['model']}.{config['dataset']}.{encode_map[config['v_encoder_name']]}.{encode_map[config['t_encoder_name']]}/"
        else:
            if 'best' not in config['save_epoch']:
                config['result_folder'] = f"./results/white_attack_cold/{config['model']}.{config['dataset']}.{config['save_epoch']}.{encode_map[config['v_encoder_name']]}.{encode_map[config['t_encoder_name']]}/"
            else:
                config['result_folder'] = f"./results/white_attack_cold/{config['model']}.{config['dataset']}.{encode_map[config['v_encoder_name']]}.{encode_map[config['t_encoder_name']]}/"
        # 再为每个模态单独设置路径
        if config['attack_modal'] == 'v':
            config['save_v_path'] = f"{config['result_folder']}/v/{config['attack_v_loss_fun']}/attacked_image_feat.{config['model']}.{config['dataset']}.{config['attack_t_method']}.{config['attack_v_method']}.{config['epsilon']}.{config['num_iters']}.{seed}/"
            config['save_v_result_path'] = f"{config['result_folder']}/v/{config['attack_v_loss_fun']}/{config['model']}.{config['dataset']}.{config['attack_v_method']}.{config['epsilon']}.{config['num_iters']}.{seed}.json"
            if config["re_attack"]:
                # 删除旧的攻击结果
                if os.path.exists(config['save_v_path']):
                    os.system(f'rm -rf {config["save_v_path"]}')
                    os.system(f'rm -rf {config["save_v_result_path"]}')
                    os.system(f"rm -rf {config['result_folder']}/v/{config['attack_v_loss_fun']}/embedding/")
            if not os.path.exists(f"{config['result_folder']}/v/{config['attack_v_loss_fun']}/embedding/"):
                os.makedirs(f"{config['result_folder']}/v/{config['attack_v_loss_fun']}/embedding/")
            if not os.path.exists(config['save_v_path']):
                os.makedirs(config['save_v_path'])
            with open(config['save_v_result_path'], 'w') as f:
                json.dump({}, f)
            config['save_v_embedding_path'] = f"{config['result_folder']}/v/{config['attack_v_loss_fun']}/embedding/{config['model']}.{config['dataset']}.{config['attack_v_method']}.{config['epsilon']}.{config['num_iters']}.{seed}_feat.npy"
        else:
            if 't' == config['attack_modal']:
                config['save_t_path'] = f"{config['result_folder']}/t/{config['attack_v_loss_fun']}/attacked_text_feat.{config['model']}.{config['dataset']}.{config['attack_t_method']}.{config['attack_v_method']}.{config['epsilon']}.{config['num_iters']}.{seed}/"
                config['save_t_result_path'] = f"{config['result_folder']}/t/{config['attack_v_loss_fun']}/{config['model']}.{config['dataset']}.{config['attack_t_method']}.{config['epsilon']}.{config['num_iters']}.{seed}.json"
                config['save_t_embedding_path'] = f"{config['result_folder']}/t/embedding/{config['model']}.{config['dataset']}.{config['attack_t_method']}.{config['goal_k']}.{seed}_feat.npy"
                if not os.path.exists(f"{config['result_folder']}/t/embedding/"):
                    os.makedirs(f"{config['result_folder']}/t/embedding/")
                if not os.path.exists(config['save_t_path']):
                    os.makedirs(config['save_t_path'])
                with open(config['save_t_result_path'], 'w') as f:
                    json.dump({}, f)
            if config['attack_modal'] == 'tv':
                config['save_v_path'] = f"{config['result_folder']}/tv/{config['attack_v_loss_fun']}/attacked_image_feat.{config['model']}.{config['dataset']}.{config['attack_t_method']}.{config['attack_v_method']}.{config['epsilon']}.{config['num_iters']}.{seed}/"
                config['save_t_path'] = f"{config['result_folder']}/tv/{config['attack_v_loss_fun']}/attacked_text_feat.{config['model']}.{config['dataset']}.{config['attack_t_method']}.{config['attack_v_method']}.{config['epsilon']}.{config['num_iters']}.{seed}/"
                config['save_tv_path'] = f"{config['result_folder']}/tv/{config['attack_v_loss_fun']}/{config['model']}.{config['dataset']}.{config['attack_t_method']}.{config['attack_v_method']}.{config['epsilon']}.{config['num_iters']}.{seed}.json"
                if not os.path.exists(f"{config['result_folder']}/tv/{config['attack_v_loss_fun']}/embedding/"):
                    os.makedirs(f"{config['result_folder']}/tv/{config['attack_v_loss_fun']}/embedding/")
                if not os.path.exists(config['save_v_path']):
                    os.makedirs(config['save_v_path'])
                if not os.path.exists(config['save_t_path']):
                    os.makedirs(config['save_t_path'])
                with open(config['save_tv_path'], 'w') as f:
                    json.dump({}, f)
                config['save_tv_v_embedding_path'] = f"{config['result_folder']}/tv/{config['attack_v_loss_fun']}/embedding/{config['model']}.{config['dataset']}.{config['attack_t_method']}.{config['attack_v_method']}.{config['epsilon']}.{config['num_iters']}.{seed}_v_feat.npy"
                config['save_tv_t_embedding_path'] = f"{config['result_folder']}/tv/{config['attack_v_loss_fun']}/embedding/{config['model']}.{config['dataset']}.{config['attack_t_method']}.{config['attack_v_method']}.{config['epsilon']}.{config['num_iters']}.{seed}_t_feat.npy"
            
            if config['attack_modal'] == 't_v':
                config['save_v_path'] = f"{config['result_folder']}/t_v/{config['attack_v_loss_fun']}/attacked_image_feat.{config['model']}.{config['dataset']}.{config['attack_t_method']}.{config['attack_v_method']}.{config['epsilon']}.{config['num_iters']}.{seed}/"
                config['save_t_path'] = f"{config['result_folder']}/t_v/{config['attack_v_loss_fun']}/attacked_text_feat.{config['model']}.{config['dataset']}.{config['attack_t_method']}.{config['attack_v_method']}.{config['epsilon']}.{config['num_iters']}.{seed}/"
                config['save_t_v_path'] = f"{config['result_folder']}/t_v/{config['attack_v_loss_fun']}/{config['model']}.{config['dataset']}.{config['attack_t_method']}.{config['attack_v_method']}.{config['epsilon']}.{config['num_iters']}.{seed}.json"
                if not os.path.exists(f"{config['result_folder']}/t_v/{config['attack_v_loss_fun']}/embedding/"):
                    os.makedirs(f"{config['result_folder']}/t_v/{config['attack_v_loss_fun']}/embedding/")
                if not os.path.exists(config['save_v_path']):
                    os.makedirs(config['save_v_path'])
                if not os.path.exists(config['save_t_path']):
                    os.makedirs(config['save_t_path'])
                with open(config['save_t_v_path'], 'w') as f:
                    json.dump({}, f)
                config['save_t_v_v_embedding_path'] = f"{config['result_folder']}/t_v/{config['attack_v_loss_fun']}/embedding/{config['model']}.{config['dataset']}.{config['attack_t_method']}.{config['attack_v_method']}.{config['epsilon']}.{config['num_iters']}.{seed}_v_feat.npy"
                config['save_t_v_t_embedding_path'] = f"{config['result_folder']}/t_v/{config['attack_v_loss_fun']}/embedding/{config['model']}.{config['dataset']}.{config['attack_t_method']}.{config['attack_v_method']}.{config['epsilon']}.{config['num_iters']}.{seed}_t_feat.npy"
            if config['attack_modal'] == 'vt':
                config['save_v_path'] = f"{config['result_folder']}/vt/{config['attack_v_loss_fun']}/attacked_image_feat.{config['model']}.{config['dataset']}.{config['attack_t_method']}.{config['attack_v_method']}.{config['epsilon']}.{config['num_iters']}.{seed}/"
                config['save_t_path'] = f"{config['result_folder']}/vt/{config['attack_v_loss_fun']}/attacked_text_feat.{config['model']}.{config['dataset']}.{config['attack_t_method']}.{config['attack_v_method']}.{config['epsilon']}.{config['num_iters']}.{seed}/"
                config['save_vt_path'] = f"{config['result_folder']}/vt/{config['attack_v_loss_fun']}/{config['model']}.{config['dataset']}.{config['attack_t_method']}.{config['attack_v_method']}.{config['epsilon']}.{config['num_iters']}.{seed}.json"
                if config["re_attack"]:
                    # 删除旧的攻击结果
                    if os.path.exists(config['save_v_path']):
                        os.system(f'rm -rf {config["save_v_path"]}')
                        os.system(f'rm -rf {config["save_t_path"]}')
                        os.system(f"rm -rf {config['result_folder']}/vt/{config['attack_v_loss_fun']}/embedding/")
                if not os.path.exists(f"{config['result_folder']}/vt/{config['attack_v_loss_fun']}/embedding/"):
                    os.makedirs(f"{config['result_folder']}/vt/{config['attack_v_loss_fun']}/embedding/")
                if not os.path.exists(config['save_v_path']):
                    os.makedirs(config['save_v_path'])
                if not os.path.exists(config['save_t_path']):
                    os.makedirs(config['save_t_path'])
                with open(config['save_vt_path'], 'w') as f:
                    json.dump({}, f)
                config['save_vt_v_embedding_path'] = f"{config['result_folder']}/vt/{config['attack_v_loss_fun']}/embedding/{config['model']}.{config['dataset']}.{config['attack_t_method']}.{config['attack_v_method']}.{config['epsilon']}.{config['num_iters']}.{seed}_v_feat.npy"
                config['save_vt_t_embedding_path'] = f"{config['result_folder']}/vt/{config['attack_v_loss_fun']}/embedding/{config['model']}.{config['dataset']}.{config['attack_t_method']}.{config['attack_v_method']}.{config['epsilon']}.{config['num_iters']}.{seed}_t_feat.npy"
                


    else: # 黑盒攻击
        config['result_folder'] = f"./results/white_attack/{config['model']}.{config['dataset']}.{encode_map[config['v_encoder_name']]}.{encode_map[config['t_encoder_name']]}/"
        config['target_result_folder'] = f"./results/black_attack/{config['model']}.{config['target_model']}.{config['dataset']}.{encode_map[config['v_encoder_name']]}.{encode_map[config['t_encoder_name']]}/"
        if config['attack_modal'] == 'v':
            # 代理模型生成的对抗样本的保存地址
            config['save_v_path'] = f"{config['result_folder']}/v/{config['attack_v_loss_fun']}/{config['model']}.{config['dataset']}.{config['attack_v_method']}.{config['epsilon']}.{config['num_iters']}.{seed}/"
            config['save_v_result_path'] = f"{config['result_folder']}/v/{config['attack_v_loss_fun']}/{config['model']}.{config['dataset']}.{config['attack_v_method']}.{config['epsilon']}.{config['num_iters']}.{seed}.json"

            # 目标模型保存对抗样本结果的地址
            config['target_save_v_result_path'] = f"{config['target_result_folder']}/v/{config['attack_v_loss_fun']}/{config['model']}.{config['target_model']}.{config['dataset']}.{config['attack_v_method']}.{config['epsilon']}.{config['num_iters']}.{seed}.json"
            if not os.path.exists(f"{config['target_result_folder']}/v/{config['attack_v_loss_fun']}/embedding/"):
                os.makedirs(f"{config['target_result_folder']}/v/{config['attack_v_loss_fun']}/embedding/")
        else:
            if 't' in config['attack_modal']:
                config['save_t_path'] = f"{config['result_folder']}/t/{config['model']}.{config['dataset']}.{config['attack_t_method']}.{config['goal_k']}.{seed}.json"
                config['target_save_t_path'] = f"{config['target_result_folder']}/t/{config['model']}.{config['target_model']}.{config['dataset']}.{config['attack_t_method']}.{config['goal_k']}.{seed}.json"
                if not os.path.exists(f"{config['target_result_folder']}/t/embedding/"):
                    os.makedirs(f"{config['target_result_folder']}/t/embedding/")
        if config['attack_modal'] == 'vt':
            config['save_v_path'] = f"{config['result_folder']}/v/{config['attack_v_loss_fun']}/{config['model']}.{config['dataset']}.{config['attack_v_method']}.{config['epsilon']}.{config['num_iters']}.{seed}/"
            config['save_vt_path'] = f"{config['result_folder']}/vt/{config['attack_v_loss_fun']}/{config['model']}.{config['dataset']}.{config['attack_t_method']}.{config['goal_k']}.{config['attack_v_method']}.{config['epsilon']}.{config['num_iters']}.{seed}.json"
            config['target_save_vt_path'] = f"{config['target_result_folder']}/vt/{config['model']}.{config['target_model']}.{config['dataset']}.{config['attack_t_method']}.{config['goal_k']}.{config['attack_v_method']}.{config['epsilon']}.{config['num_iters']}.{seed}.json"
            if not os.path.exists(f"{config['target_result_folder']}/vt/{config['attack_v_loss_fun']}/"):
                os.makedirs(f"{config['target_result_folder']}/vt/{config['attack_v_loss_fun']}/")
        if config['attack_modal'] == 'tv':
            config['save_v_path'] = f"{config['result_folder']}/tv/{config['attack_v_loss_fun']}/attacked_image_feat.{config['model']}.{config['dataset']}.{config['attack_t_method']}.{config['attack_v_method']}.{config['epsilon']}.{config['num_iters']}.{seed}/"
            config['save_t_path'] = f"{config['result_folder']}/tv/{config['attack_v_loss_fun']}/attacked_text_feat.{config['model']}.{config['dataset']}.{config['attack_t_method']}.{config['attack_v_method']}.{config['epsilon']}.{config['num_iters']}.{seed}/"
            config['save_tv_path'] = f"{config['result_folder']}/tv/{config['attack_v_loss_fun']}/{config['model']}.{config['dataset']}.{config['attack_t_method']}.{config['attack_v_method']}.{config['epsilon']}.{config['num_iters']}.{seed}.json"
            config['target_save_tv_path'] = f"{config['target_result_folder']}/tv/{config['attack_v_loss_fun']}/{config['model']}.{config['target_model']}.{config['dataset']}.{config['attack_t_method']}.{config['attack_v_method']}.{config['epsilon']}.{config['num_iters']}.{seed}.json"
            if not os.path.exists(f"{config['target_result_folder']}/tv/{config['attack_v_loss_fun']}/"):
                os.makedirs(f"{config['target_result_folder']}/tv/{config['attack_v_loss_fun']}/")

    return config



def cal_evalution_metrics(score_matrix, item_id,all_metrics=None, k=None):

    sorted_indices = torch.argsort(score_matrix, dim=1, descending=True)
    rankings = (sorted_indices == item_id).nonzero(as_tuple=True)[1]
    batch_metrics = {}
    for i in k:
        hit_k = torch.sum(rankings < i).item()
        all_metrics[f"hit_rate@{i}"] += hit_k # 命中个数
        batch_metrics[f"hit_rate@{i}"] = hit_k /rankings.shape[0] # 命中率
    return batch_metrics

def read_and_save(result, path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            result.update(json.load(f))
    with open(path, 'w') as f:
        json.dump(result, f, indent=4)

# def read_path_feat_list(path):
#     # 读取路径下的所有.npy文件，并返回一个文件名列表
#     file_list = []
#     if os.path.exists(path):
#         for file_name in os.listdir(path):
#             if file_name.endswith('.npy'):
#                 file_list.append(file_name[:-4])  # 去掉文件名的.npy后缀
    
#     return file_list

                

def read_result(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            result = json.load(f)
    else:
        result = {}
    return result





def get_local_time():
    r"""Get current time

    Returns:
        str: current time
    """
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y-%H-%M-%S')

    return cur


def get_model(model_name):
    r"""Automatically select model class based on model name
    Args:
        model_name (str): model name
    Returns:
        Recommender: model class
    """
    model_file_name = model_name.lower()
    if 'at' in model_file_name:
        module_path = '.'.join(['models_AT', model_file_name])
    else:
        module_path = '.'.join(['models', model_file_name])


    if importlib.util.find_spec(module_path, __name__):
        model_module = importlib.import_module(module_path, __name__)

    model_class = getattr(model_module, model_name)
    return model_class
def get_attacked_model(model_name):
    r"""Automatically select model class based on model name,which will be attacked.
    Args:
        model_name (str): model name
    Returns:
        Recommender: model class
    """
    model_file_name = model_name.lower()
    module_path = '.'.join(['models_attack', model_file_name])
    if importlib.util.find_spec(module_path, __name__):
        model_module = importlib.import_module(module_path, __name__)
    
    model_class = getattr(model_module, model_name)
    return model_class

def get_AT_model(model_name):
    model_file_name = model_name.lower()
    moudle_path = '.'.join(['models_AT', model_file_name])
    if importlib.util.find_spec(moudle_path, __name__):
        model_module = importlib.import_module(moudle_path, __name__)
    model_class = getattr(model_module, model_name)
    return model_class


def read_json(path, as_int=False):
    with open(path, 'r') as f:
        raw = json.load(f)
        if as_int:
            data = dict((int(key), value) for (key, value) in raw.items())
        else:
            data = dict((key, value) for (key, value) in raw.items())
        del raw
        return data


def get_trainer():
    return getattr(importlib.import_module('common.trainer'), 'Trainer')


def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def early_stopping(value, best, cur_step, max_step, bigger=True):
    r""" validation-based early stopping

    Args:
        value (float): current result
        best (float): best result
        cur_step (int): the number of consecutive steps that did not exceed the best result
        max_step (int): threshold steps for stopping
        bigger (bool, optional): whether the bigger the better

    Returns:
        tuple:
        - float,
          best result after this step
        - int,
          the number of consecutive steps that did not exceed the best result after this step
        - bool,
          whether to stop
        - bool,
          whether to update
    """
    stop_flag = False
    update_flag = False
    if bigger:
        if value > best:
            cur_step = 0
            best = value
            update_flag = True
        else:
            cur_step += 1
            if cur_step > max_step:
                stop_flag = True
    else:
        if value < best:
            cur_step = 0
            best = value
            update_flag = True
        else:
            cur_step += 1
            if cur_step > max_step:
                stop_flag = True
    return best, cur_step, stop_flag, update_flag


def dict2str(result_dict):
    r""" convert result dict to str

    Args:
        result_dict (dict): result dict

    Returns:
        str: result str
    """

    result_str = ''
    for metric, value in result_dict.items():
        result_str += str(metric) + ': ' + '%.04f' % value + '    '
    return result_str


############ LATTICE Utilities #########

def build_knn_neighbourhood(adj, topk):
    knn_val, knn_ind = torch.topk(adj, topk, dim=-1)
    weighted_adjacency_matrix = (torch.zeros_like(adj)).scatter_(-1, knn_ind, knn_val)
    return weighted_adjacency_matrix


def compute_normalized_laplacian(adj):
    rowsum = torch.sum(adj, -1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt)
    L_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    return L_norm


def build_sim(context):
    context_norm = context.div(torch.norm(context, p=2, dim=-1, keepdim=True))
    sim = torch.mm(context_norm, context_norm.transpose(1, 0))
    return sim

def get_sparse_laplacian(edge_index, edge_weight, num_nodes, normalization='none'):
    from torch_scatter import scatter_add
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)

    if normalization == 'sym':
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    elif normalization == 'rw':
        deg_inv = 1.0 / deg
        deg_inv.masked_fill_(deg_inv == float('inf'), 0)
        edge_weight = deg_inv[row] * edge_weight
    return edge_index, edge_weight

def get_dense_laplacian(adj, normalization='none'):
    if normalization == 'sym':
        rowsum = torch.sum(adj, -1)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt)
        L_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    elif normalization == 'rw':
        rowsum = torch.sum(adj, -1)
        d_inv = torch.pow(rowsum, -1)
        d_inv[torch.isinf(d_inv)] = 0.
        d_mat_inv = torch.diagflat(d_inv)
        L_norm = torch.mm(d_mat_inv, adj)
    elif normalization == 'none':
        L_norm = adj
    return L_norm

def build_knn_normalized_graph(adj, topk, is_sparse, norm_type):
    device = adj.device
    knn_val, knn_ind = torch.topk(adj, topk, dim=-1)
    if is_sparse:
        tuple_list = [[row, int(col)] for row in range(len(knn_ind)) for col in knn_ind[row]]
        row = [i[0] for i in tuple_list]
        col = [i[1] for i in tuple_list]
        i = torch.LongTensor([row, col]).to(device)
        v = knn_val.flatten()
        edge_index, edge_weight = get_sparse_laplacian(i, v, normalization=norm_type, num_nodes=adj.shape[0])
        return torch.sparse_coo_tensor(edge_index, edge_weight, adj.shape)
    else:
        weighted_adjacency_matrix = (torch.zeros_like(adj)).scatter_(-1, knn_ind, knn_val)
        return get_dense_laplacian(weighted_adjacency_matrix, normalization=norm_type)