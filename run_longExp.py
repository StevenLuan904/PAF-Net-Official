import argparse
import copy
import datetime
import os
import time
import random
from pprint import pprint

import torch
import numpy as np

import settings
from settings import data_settings
from data_provider import data_loader
from exp.exp_main import Exp_Main

print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


parser = argparse.ArgumentParser(description='Time Series Forecasting')

parser.add_argument('--is_training', type=int, default=1)
parser.add_argument('--train_only', action='store_true', default=False)
parser.add_argument('--wo_test', action='store_true', default=False)
parser.add_argument('--only_test', action='store_true', default=False)
parser.add_argument('--model', type=str, required=False, default='PAFNet')
parser.add_argument('--override_hyper', action='store_true', default=True)
parser.add_argument('--compile', action='store_true', default=False)
parser.add_argument('--reduce_bs', type=str_to_bool, default=False)
parser.add_argument('--normalization', type=str, default=None)
parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

parser.add_argument('--root_path', type=str, default='./dataset/')
parser.add_argument('--dataset', type=str, default='Production')
parser.add_argument('--features', type=str, default='M')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--target', type=str, default='OT')
parser.add_argument('--freq', type=str, default='h')
parser.add_argument('--wrap_data_class', type=list, default=[])

# forecasting task
parser.add_argument('--seq_len', type=int, default=360)
parser.add_argument('--label_len', type=int, default=48)
parser.add_argument('--pred_len', type=int, default=36)

# PAFNet
parser.add_argument('--decomp_method', type=str, default='dct')
parser.add_argument('--patch_leaders', type=int, default=1)
parser.add_argument('--decomposition', type=int, default=5)
parser.add_argument('--revin', type=int, default=1)
parser.add_argument('--patch_len', type=int, default=16)

# LIFT
parser.add_argument('--leader_num', type=int, default=4)
parser.add_argument('--state_num', type=int, default=8)
parser.add_argument('--prefetch_path', type=str, default='./prefetch/')
parser.add_argument('--tag', type=str, default='_max')
parser.add_argument('--prefetch_batch_size', type=int, default=16)
parser.add_argument('--variable_batch_size', type=int, default=32)
parser.add_argument('--max_leader_num', type=int, default=16)
parser.add_argument('--masked_corr', action='store_true', default=False)
parser.add_argument('--efficient', type=str_to_bool, default=True)
parser.add_argument('--pin_gpu', type=str_to_bool, default=True)
parser.add_argument('--pretrain', action='store_true', default=False)
parser.add_argument('--freeze', action='store_true', default=False)
parser.add_argument('--lift', action='store_true', default=False)
parser.add_argument('--temperature', type=float, default=1.0)

# DLinear
parser.add_argument('--individual', action='store_true', default=False)

# PatchTST
parser.add_argument('--fc_dropout', type=float, default=0.05)
parser.add_argument('--head_dropout', type=float, default=0.0)
parser.add_argument('--stride', type=int, default=8)
parser.add_argument('--padding_patch', default='end')
parser.add_argument('--affine', type=int, default=0)
parser.add_argument('--subtract_last', type=int, default=0)
parser.add_argument('--kernel_size', type=int, default=25)

# Transformer
parser.add_argument('--embed_type', type=int, default=0)
parser.add_argument('--d_model', type=int, default=128)
parser.add_argument('--n_heads', type=int, default=8)
parser.add_argument('--e_layers', type=int, default=2)
parser.add_argument('--d_layers', type=int, default=1)
parser.add_argument('--d_ff', type=int, default=2048)
parser.add_argument('--moving_avg', type=int, default=25)
parser.add_argument('--factor', type=int, default=3)
parser.add_argument('--distil', action='store_false', default=True)
parser.add_argument('--dropout', type=float, default=0.05)
parser.add_argument('--embed', type=str, default='timeF')
parser.add_argument('--activation', type=str, default='gelu')
parser.add_argument('--output_attention', action='store_true')
parser.add_argument('--output_enc', action='store_true')
parser.add_argument('--do_predict', action='store_true')

# Crossformer
parser.add_argument('--seg_len', type=int, default=24)
parser.add_argument('--win_size', type=int, default=2)
parser.add_argument('--num_routers', type=int, default=10)

# MTGNN
parser.add_argument('--subgraph_size', type=int, default=20)
parser.add_argument('--in_dim', type=int, default=1)

# optimization
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--itr', type=int, default=1)
parser.add_argument('--train_epochs', type=int, default=150)
parser.add_argument('--begin_valid_epoch', type=int, default=0)
parser.add_argument('--patience', type=int, default=80)
parser.add_argument('--optim', type=str, default='Adam')
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--des', type=str, default='test')
parser.add_argument('--loss', type=str, default='mse')
parser.add_argument('--lradj', type=str, default='type1')
parser.add_argument('--use_amp', action='store_true', default=False)
parser.add_argument('--pct_start', type=float, default=0.3)
parser.add_argument('--warmup_epochs', type=int, default=5)

# GPU
parser.add_argument('--use_gpu', type=str_to_bool, default=True)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--use_multi_gpu', action='store_true', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3')
parser.add_argument('--test_flop', action='store_true', default=False)
parser.add_argument("--local-rank", default=os.getenv('LOCAL_RANK', -1), type=int)



args = parser.parse_args()
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

import platform
if platform.system() != 'Windows':
    args.num_workers = 0
else:
    torch.cuda.set_per_process_memory_fraction(48/61, 0)

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

args.enc_in, args.c_out = data_settings[args.dataset][args.features]
args.data_path = data_settings[args.dataset]['data']
args.dec_in = args.enc_in

if args.tag and args.tag[0] != '_':
    args.tag = '_' + args.tag

args.data = 'custom'

FLAG_LIFT = args.lift
if FLAG_LIFT:
    Exp = Exp_Lead
    args.wrap_data_class.append(data_loader.Dataset_Lead_Pretrain if args.freeze else data_loader.Dataset_Lead)
else:
    Exp = Exp_Main

args.model_id = f'{args.dataset}_{args.seq_len}_{args.pred_len}_{args.model}'
if args.normalization is not None:
    args.model_id += '_' + args.normalization

if args.override_hyper and args.model in settings.hyperparams:
    if 'prefetch_batch_size' in data_settings[args.dataset]:
        args.__setattr__('prefetch_batch_size', data_settings[args.dataset]['prefetch_batch_size'])
    for k, v in settings.get_hyperparams(args.dataset, args.model, args).items():
        args.__setattr__(k, v)

if args.local_rank != -1:
    torch.cuda.set_device(args.local_rank)
    args.gpu = args.local_rank
    torch.distributed.init_process_group(backend="nccl", init_method='env://')
    args.num_gpus = torch.cuda.device_count()
    args.batch_size = args.batch_size // args.num_gpus

if FLAG_LIFT and args.pretrain and args.freeze:
    args.lradj = 'type3'

K_tag = f'_K{args.leader_num}' if args.leader_num > 8 and args.enc_in > 8 else ''
prefetch_path = os.path.join(args.prefetch_path, f'{args.dataset}_L{args.seq_len}{K_tag}{args.tag}')
if not os.path.exists(prefetch_path + '_train.npz'):
    K_tag = f'_K16' if args.leader_num > 8 and args.enc_in > 8 else ''
    prefetch_path = os.path.join(args.prefetch_path, f'{args.dataset}_L{args.seq_len}{K_tag}{args.tag}')
args.prefetch_path = prefetch_path

if args.lift and 'Linear' in args.model:
    args.patience = max(args.patience, 5)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    train_data, train_loader, vali_data, vali_loader = None, None, None, None
    test_data, test_loader = None, None

    if args.is_training:
        all_results = {'mse': [], 'mae': []}
        for ii in range(args.itr):
            fix_seed = 2021 + ii if args.model == 'PatchTST' and args.dataset in ['ECL', 'Traffic', 'Illness', 'Weather'] else 2023 + ii
            setup_seed(fix_seed)

            setting = f'{args.model_id}_{args.data}_ft{args.features}_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_lr{args.learning_rate}_dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_dl{args.d_layers}_df{args.d_ff}_fc{args.factor}_eb{args.embed}_dt{args.distil}_{args.des}_{ii}'

            if args.pretrain:
                pretrain_setting = f'{args.model_id}_{args.border_type if hasattr(args, "border_type") and args.border_type else args.data}_ft{args.features}_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_lr{settings.pretrain_lr(args.model, args.dataset, args.pred_len, args.learning_rate)}_dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_dl{args.d_layers}_df{args.d_ff}_fc{args.factor}_eb{args.embed}_dt{args.distil}_{args.des}_{ii}'
                args.pred_path = os.path.join('./results/', pretrain_setting, 'real_prediction.npy')
                args.load_path = os.path.join('./checkpoints/', pretrain_setting, 'checkpoint.pth')
                
                if FLAG_LIFT and args.freeze and not os.path.exists(args.pred_path) and args.local_rank <= 0:
                    _args = copy.deepcopy(args)
                    _args.freeze = False
                    _args.wrap_data_class = []
                    exp = Exp_Main(_args)
                    exp.predict(pretrain_setting, True)
                    torch.cuda.empty_cache()

            if args.lift:
                setting += '_lift'

            exp = Exp(args)
            checkpoint_path = os.path.join("checkpoints", setting, 'checkpoint.pth')
            
            if not args.only_test or not os.path.exists(checkpoint_path):
                _, train_data, train_loader, vali_data, vali_loader = exp.train(setting, train_data, train_loader, vali_data, vali_loader)
                torch.cuda.empty_cache()
            else:
                exp.load_checkpoint(checkpoint_path)

            if not args.wo_test and not args.train_only and args.local_rank <= 0:
                mse, mae, test_data, test_loader = exp.test(setting, test_data, test_loader)
                all_results['mse'].append(mse)
                all_results['mae'].append(mae)

            if args.do_predict:
                exp.predict(setting, True)

            torch.cuda.empty_cache()
            
        if not args.wo_test and not args.train_only and args.local_rank <= 0:
            for k in all_results.keys():
                all_results[k] = np.array(all_results[k])
                all_results[k] = [all_results[k].mean(), all_results[k].std()]
            pprint(all_results)
    else:
        setting = f'{args.model_id}_{getattr(args, "border_type", args.data)}_ft{args.features}_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_lr{args.learning_rate}_dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_dl{args.d_layers}_df{args.d_ff}_fc{args.factor}_eb{args.embed}_dt{args.distil}_{args.des}_0'
        args.load_path = os.path.join(args.checkpoints, setting, 'checkpoint.pth')
        
        if args.lift:
            setting += '_lift'

        exp = Exp(args)

        if args.do_predict:
            exp.predict(setting, True)
        else:
            exp.test(setting, test=1)
        torch.cuda.empty_cache()
