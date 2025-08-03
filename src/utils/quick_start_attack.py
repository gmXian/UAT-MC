from logging import getLogger
import numpy as np
from utils.dataset import RecDataset
from utils.dataloader import TrainDataLoader, EvalDataLoader
from utils.logger import init_logger
from utils.configurator import Config
from utils.utils import init_seed, get_attacked_model, get_trainer, dict2str,read_result,set_path
from utils.encoder import text_encoder,image_encoder
import platform
import os
import torch
import json

def quick_start_attack(model, dataset, save_epoch, config_dict,mg=False,tm = None): 
    if tm is None:
        config = Config(model, dataset, config_dict, mg, attack=True)
        config['target_model'] = tm
        config['save_epoch'] = save_epoch
    else:
        config = Config(tm, dataset, config_dict, mg, attack=True)
        config['target_model'] = tm
        config['model'] = model

    init_logger(config)
    logger = getLogger()
    # print config infor
    logger.info('██Server: \t' + platform.node())
    logger.info('██Dir: \t' + os.getcwd() + '\n')
    logger.info(config)

    # load data
    dataset = RecDataset(config)
    # print dataset statistics
    logger.info(str(dataset))

    train_dataset, valid_dataset, test_dataset = dataset.split()
    logger.info('\n====Training====\n' + str(train_dataset))
    logger.info('\n====Validation====\n' + str(valid_dataset))
    logger.info('\n====Testing====\n' + str(test_dataset))

  
    for epsilon in config['epsilons']:
        config['epsilon'] = float(epsilon)
        seeds_hit_50_before = []
        seeds_hit_50_after = []

        if config['attack_cold_start'] or config['attack_hot_start']: 
            config['seed'] = [1]

        for seed in config['seed']: 
            # seed = config['seed'] 
            # init_seed(int(seed))
            init_seed(int(seed))

            config = set_path(config,seed)

            # wrap into dataloader
            train_data = TrainDataLoader(config, train_dataset, batch_size=config['train_batch_size'], shuffle=True)
            test_data = EvalDataLoader(config, test_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size'])
            user_num = config['user_num']
            item_num = config['item_num']

            # set random state of dataloader
            train_data.pretrain_setup()

            # Load model
            logger.info(f"model:{config['model']}")



            rs_model = get_attacked_model(config['model'])(config, train_data).to(config['device'])
            if mg:
                rs_checkpoint = torch.load(f"./checkpoints_saves/{config['model']}_{config['dataset']}_best_model_mg.pth")
            else:
                if 'best' in save_epoch:
                    logger.info(f"Load model checkpoint from {config['model']}_{config['dataset']}_best_model.pth")
                    rs_checkpoint = torch.load(f"./checkpoints_saves/{config['model']}_{config['dataset']}_best_model.pth")['state_dict'] 
                else:
                    logger.info(f"Load model checkpoint from {config['model']}_{config['dataset']}_{config['save_epoch']}.pth")
                    rs_checkpoint = torch.load(f"./checkpoints_saves/{config['model']}_{config['dataset']}_{config['save_epoch']}.pth")['state_dict'] 
            rs_model.load_state_dict(rs_checkpoint)

            from utils.attacker import Attacker
            attack_t_method = config['attack_t_method']
            attack_v_method = config['attack_v_method']
            
            attacker = Attacker(config,rs_model,attack_v_method=attack_v_method,attack_t_method=attack_t_method,device=config['device'],logger=logger,seed=seed,test_data=test_data)

            if not config['attack_cold_start'] and not config['attack_hot_start']:
                attack_item_id_list = np.random.choice(item_num,size=int(100),replace=False)
                attack_item_id_list = [k for k in attack_item_id_list]
                logger.info(f"Normal sample:{attack_item_id_list}, len:, {len(attack_item_id_list)}")
            elif config['attack_hot_start']:
                dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
                with open(os.path.join(dataset_path, 'item_cnt_50.json'), 'r') as f:
                    hot_start_item_ids = json.load(f)
                attack_item_id_list = np.random.choice(hot_start_item_ids, size=int(100), replace=False)
                logger.info(f"Hot items sample:{attack_item_id_list}, len:, {len(attack_item_id_list)}")
            else:
                dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
                with open(os.path.join(dataset_path, 'item_cnt_5.json'), 'r') as f:
                    cold_start_item_ids = json.load(f)
                attack_item_id_list = np.random.choice(cold_start_item_ids, size=int(100), replace=False)
                logger.info(f"Cold items sample:{attack_item_id_list}, len:, {len(attack_item_id_list)}")
            config['attack_item_id_list'] = attack_item_id_list

            np.random.seed(seed)
            if config['dataset'] == 'microlens':
                anchor_users = np.random.choice(user_num,size=int(user_num * 0.4),replace=False)
            else:
                anchor_users = np.random.choice(user_num, size=int(user_num * 0.4), replace=False)
            anchor_users = torch.tensor(anchor_users).to(config['device'])
            all_users = torch.tensor(range(user_num)).to(config['device'])
            k = [10,20,50,100,200,500,1000,2000,5000]

            attack_modal = config['attack_modal']

            for attack_item_id in attack_item_id_list:
                attack_item_id = int(attack_item_id)
                results_t = {}
                results_t[attack_item_id] = {}
                results_vt = {}
                results_vt[attack_item_id] = {}

                attacker.forward(anchor_users=anchor_users,
                                all_users=all_users,
                                attack_item_id=attack_item_id,
                                attack_modal=attack_modal,
                                results_t=results_t,
                                results_vt=results_vt,
                                k=k)
            
            save_vt_path = config['save_vt_path']
            read_vt_result = read_result(save_vt_path)
            logger.info(f"model:{config['model']};dataset:{config['dataset']};attack_t_method:{attack_t_method};attack_v_method:{config['attack_v_method']};epsilon:{config['epsilon']};num_iters:{config['num_iters']};seed:{str(seed)}")
            seed_hit_50_before,seed_hit_50_after = analysis_result(result=read_vt_result,logger=logger)
            seeds_hit_50_before.append(seed_hit_50_before)
            seeds_hit_50_after.append(seed_hit_50_after)
            
        logger.info(f"seed_hit_50_before:{seeds_hit_50_before}")
        logger.info(f"seed_hit_50_after:{seeds_hit_50_after}")
        logger.info(f"seed_hit_50_before_mean:{np.mean(seeds_hit_50_before)}")
        logger.info(f"seed_hit_50_after_mean:{np.mean(seeds_hit_50_after)}")
        logger.info(f"{((np.mean(seeds_hit_50_after)/np.mean(seeds_hit_50_before))-1)*100}%")
        
                                                                                    


def analysis_result(result,logger=None):
    item_cnt = len(result)
    improve_cnt = 0  
    before_num = {}
    after_num = {}
    improve_num_relative = {}  
    improve_num_absolute = {} 
    k = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]

    for i in k:
        before_num[f"hit_rate@{i}"] = 0
        after_num[f"hit_rate@{i}"] = 0
        improve_num_relative[f"hit_rate@{i}"] = 0
        improve_num_absolute[f"hit_rate@{i}"] = 0

    for i in result.keys():
        hit_rate_50_improve = result[i]['hit_rate_after']["hit_rate@50"] - result[i]['hit_rate_before']["hit_rate@50"]
        if hit_rate_50_improve != 0: 
            logger.info(f"{i}, {hit_rate_50_improve}")
            improve_cnt += 1
            for j in k:
                improve_num_relative[f"hit_rate@{j}"] += result[i]['hit_rate_after'][f"hit_rate@{j}"] - result[i]['hit_rate_before'][f"hit_rate@{j}"]
        for j in k:
            before_num[f"hit_rate@{j}"] += result[i]['hit_rate_before'][f"hit_rate@{j}"]
            after_num[f"hit_rate@{j}"] += result[i]['hit_rate_after'][f"hit_rate@{j}"]
            improve_num_absolute[f"hit_rate@{j}"] += result[i]['hit_rate_after'][f"hit_rate@{j}"] - result[i]['hit_rate_before'][f"hit_rate@{j}"]

    for i in k:
        before_num[f"hit_rate@{i}"] /= item_cnt
        after_num[f"hit_rate@{i}"] /= item_cnt
        improve_num_absolute[f"hit_rate@{i}"] /= item_cnt
        if improve_cnt > 0:
            improve_num_relative[f"hit_rate@{i}"] /= improve_cnt
    if logger is not None:
        logger.info(f"{improve_cnt / item_cnt}")
        logger.info(before_num)
        logger.info(after_num)
        logger.info(improve_num_relative)
        logger.info(improve_num_absolute)
    return before_num['hit_rate@50'],after_num['hit_rate@50']






