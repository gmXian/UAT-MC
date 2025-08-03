# Generate Adversarial Perturbations under the Evasion-based Promotion Attack Setting
import os
import argparse
from utils.quick_start_attack import quick_start_attack as quick_start
os.environ['NUMEXPR_MAX_THREADS'] = '48'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='VBPR_MLP', help='name of models')
    parser.add_argument('--target_model', '-tm', type=str, default='None', help='name of target models')
    parser.add_argument('--dataset', '-d', type=str, default='baby', help='name of datasets')
    parser.add_argument('--mg', action="store_true", help='whether to use Mirror Gradient, default is False')
    parser.add_argument('--attack_modal','-am', type=str, default='None', help='name of attack modal')
    parser.add_argument('--attack_method', '-method', type=str, default='pgd', help='attack method for visual modal, default is pgd')
    parser.add_argument('--save_epoch', '-e', type=str, default='best', help='number of epochs')
    parser.add_argument('--attack_cold_start', '-ac', action='store_true',default=False, help='whether to attack cold start items, default is False')
    parser.add_argument('--attack_hot_start', '-ah', action='store_true',default=False, help='whether to attack popular items, default is False')
    parser.add_argument('--loss_function', '-lf', type=str, default='max_ui_3', help='promotion attack loss function, default is max_ui')
    parser.add_argument('--re_attack', '-re', action='store_true',default=False,  help='delete the old attack results and re-attack, default is False')
    args, _ = parser.parse_known_args()
    config_dict = {
            # 'gpu_id': 1,
            'attack_modal':args.attack_modal,
            'attack_cold_start':args.attack_cold_start,
            'attack_hot_start':args.attack_hot_start,
            'attack_v_loss_fun':args.loss_function,
            're_attack':args.re_attack,
            'attack_method':args.attack_method,
        }
    quick_start(model=args.model, dataset=args.dataset,save_epoch=args.save_epoch, config_dict=config_dict, mg=args.mg)

