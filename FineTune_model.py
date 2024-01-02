import numpy as np
import torch
import torch.nn as nn
import pickle
import warnings
import tqdm
import os
import argparse
from omegaconf import OmegaConf

from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import KFold

from Dare.Datasets import Dataset_FT
from Dare.utils import ModelConfigs
from Dare.bert_pretraining import BERT_PT
from Dare.bert import BERT
from Dare.utils_FT import WeigthedBCELoss, rnn_liner_FT, import_model, EarlyStopping
from Dare.trainer import finetuner


# To get weights for class balances
#loader = torch.utils.data.DataLoader(ds, batch_size=len(ds)) #len(ds)
#get_weights = lambda x: compute_class_weight('balanced',classes=np.array([0,1]),y=x)
#
# WEIGHTS = {}
#for data in loader:
#    diags_all = torch.cat([data[f'd0_{y}'] for y in range(2015,2019)]).numpy()
#    WEIGHTS['diags'] = np.apply_along_axis(get_weights, 0, diags_all)
#    
#    diags_all = torch.cat([data[f'm0_{y}'] for y in range(2015,2019)]).numpy()
#    WEIGHTS['drugs'] = np.apply_along_axis(get_weights, 0, diags_all)
#    
#    targets = data[f'hba1c_targets'].numpy()
#    WEIGHTS['targets'] = np.apply_along_axis(get_weights, 0,  targets.reshape((len(targets)*4, -1)))
WEIGHTS = {
    'diags': np.array([[2.55449591, 0.61147619, 0.63088829, 0.59991361, 0.52707033],
                       [0.62168435, 2.74263146, 2.41002571, 3.00216156, 9.73520249]]),
    'drugs': np.array([[0.86573091, 4.16019525], [1.18356268, 0.56830237]]),
    'targets': np.array([[0.85010881], [1.21406371]]) }

# Seeds to reproduce results in cross validation
states = {
    '01': {'k': 10, 'states': [456], 'train_val': True},
    '09': {'k': 10, 'states': [798], 'train_val': False},
    '02': {'k': 5, 'states': [123, 456], 'train_val': True},
    '0875': {'k': 8, 'states': [123, 456], 'train_val': False},
    '05': {'k': 2, 'states': [123, 456, 789, 101, 121], 'train_val': True},
    '0666': {'k': 3, 'states': [123, 456, 789, 101], 'train_val': False},
    '08': {'k': 5, 'states': [123, 456], 'train_val': False},
}


def main(args, res_path_base):
    
    configs = {
        'attn_heads': args.attn_heads,
        'hidden_size': args.hidden_size,
        'n_layers': args.n_layers,
        'relative': args.relative}
    configs = ModelConfigs(configs, configs_path=args.config_file)
    
    paths = OmegaConf.load(args.paths_file)
    with open(paths.idps_tests, 'rb') as f:
        idps_test = pickle.load(f)
    ds = Dataset_FT(paths, configs, idps_test, args.only_rnn)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.only_rnn:
        bert = None
    else:
        bert = import_model(configs, args.model_epoch)
        bert.eval()
        for param in bert.parameters():
            param.requires_grad = False
        bert.to(device)
            
    for st_it, opt in states.items():
        res_path = res_path_base + f'CROSS_{st_it}.pkl'
        print('START:')
        print(f'Saving results in {res_path}')
        if os.path.isfile(res_path):
            with open(res_path, 'rb') as f:
                losses = pickle.load(f)
            k_start = len(losses)
        else:
            losses = {}
            k_start = 0
        k = 0
        for state in opt['states']:
            kf = KFold(n_splits=opt['k'], shuffle=True, random_state=state)
            for i, (train_indices, val_indices) in enumerate(kf.split(ds)):
            
                print(f'START ITERATION {k+1}')
                if k >= k_start and k <=10:
                    # get data loaders
                    if opt['train_val']:
                        train_sampler = SubsetRandomSampler(train_indices)
                        valid_sampler = SubsetRandomSampler(val_indices)
                    else:
                        train_sampler = SubsetRandomSampler(val_indices)
                        valid_sampler = SubsetRandomSampler(train_indices)
                    train_loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, sampler=train_sampler)
                    validation_loader = torch.utils.data.DataLoader(ds, batch_size=256, sampler=valid_sampler)
                
                    # get model
                    if bert is None:
                        rnn_inp_dim = 218
                    else:
                        rnn_inp_dim = bert.hidden
                        
                    model = rnn_liner_FT(rnn_inp_dim, 2, rnn_dims, outs_size)
                    model.to(device)
                    if args.weighted_loss:
                        weights_classes =  WEIGHTS[args.predict]
                    else:
                        weights_classes = None
                    trainer = finetuner(bert, model, train_loader, validation_loader, device=device, weights_classes=weights_classes, predict=args.predict)
            
                    # train model 
                    losses_tmp = {}
                    max_roc = 0
                    es = EarlyStopping()
                    for ep in range(args.epochs):
                        losses_tmp[f'train_{i}'] = trainer.train_iteration( ep, 30, True)
                        res, labs_i, preds_i, idps_i = trainer.test_iteration( ep, 10, True)
                        losses_tmp[f'test_{ep}'] = res
                        if res['avg_roc'] >= max_roc:
                            max_roc = res['avg_roc']
                            losses_tmp['labs'] = labs_i.numpy()
                            losses_tmp['pred'] = preds_i.numpy()
                            losses_tmp['idps'] = idps_i
                            
                        # early stopping
                        es(res['avg_roc'])
                        if es.early_stop:
                            print("Early stop at epoch:", ep)
                            break
                    
                    del trainer, model   
                    losses[f'K_{k}'] = losses_tmp
                
                k +=1
                print('------------------------------------------------------------------')
                print()
                with open(res_path, 'wb') as outp:
                    pickle.dump(losses, outp)
    
    

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--predict', choices=['diags', 'drugs', 'targets'], default='diags')
    parser.add_argument('--only_rnn',  action='store_true')
    parser.set_defaults(only_rnn=False)
    parser.add_argument('--weighted_loss', action='store_true')
    parser.add_argument('--no-weighted_loss', dest='weighted_loss', action='store_false')
    parser.set_defaults(weighted_loss=True)
    parser.add_argument('--config_file', type=str, default='Dare/Configs/configs_std.yaml')
    parser.add_argument('--paths_file', type=str, default='Dare/Configs/data_paths.yaml')
    parser.add_argument('--path_results', type=str, default='Results/FT/')
    parser.add_argument('--model_epoch', type=int, default=9)
    parser.add_argument('--hidden_size', type=int, default=360)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--attn_heads', type=int, default=18)
    parser.add_argument('--relative', action='store_true')
    parser.add_argument('--no-relative', dest='relative', action='store_false')
    parser.set_defaults(relative=True)
    
    args = parser.parse_args()
    RES_PATH = args.path_results
    #RES_PATH = os.path.join(RES_PATH, f'{"weight_BCE" if args.weighted_loss else "BCE"}/{args.predict}/')
    RES_PATH = os.path.join(RES_PATH, f'{args.predict}/res_with_labs')
    if not os.path.exists(RES_PATH): # If it doesn't exist, create it
        os.mkdir(RES_PATH)
        print(f"Directory '{RES_PATH}' not found, created new one.")
    
        
    if args.predict == 'diags':
        outs_size = 20
        rnn_dims = [100, 50, 10]
    elif args.predict == 'drugs':
        outs_size = 8
        rnn_dims = [100, 50, 10]
    else:
        outs_size = 4
        rnn_dims = [100, 50, 10]
    
    rnn_name = '2' + "_".join([str(i) for i in rnn_dims])
    if args.only_rnn:
        model_name = f'rnn_{rnn_name}_'
    else:
        if args.relative:
            model_name = f'BERT_R_{args.attn_heads}_{args.hidden_size}_{args.n_layers}_rnn_{rnn_name}_'
        else:
            model_name = f'BERT_{args.attn_heads}_{args.hidden_size}_{args.n_layers}_rnn_{rnn_name}_'
        model_name = f'BERT_rnn_{rnn_name}_'
            
    res_path = os.path.join(RES_PATH,model_name)
    main(args, res_path)
    
    