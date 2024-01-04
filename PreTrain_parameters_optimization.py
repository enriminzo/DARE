from omegaconf import OmegaConf
import numpy as np
import torch
import torch.nn as nn
import pickle
import warnings
import tqdm
import os
import argparse

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import  HyperBandForBOHB, ASHAScheduler
from ray.tune.search.bohb import  TuneBOHB
from ray.air import session, RunConfig

from Dare import Dataset_PT, pretrainer, BERT
from Dare.Model.utils import ModelConfigs



# Define hyperparameters space and initial points (optional)
config_PT = {
    "hidden_size": tune.choice([108, 144, 180, 216, 252, 288,  360, 432, 576]),
    "n_layers": tune.randint(3,10),
    "attn_heads": tune.choice([6, 12, 18]),
    "relative": tune.choice([True,False])
}

config_init = [
    {"hidden_size": 360, "n_layers": 3, "attn_heads": 18, 'relative': True}]

def pretrain_model(config_PT, args):
    
    config_PT['feed_forward_hidden'] = config_PT['hidden_size']*4
    configs = ModelConfigs(config_PT, configs_path=args.config_file)
    paths = OmegaConf.load(args.paths_file)
    losses_train_path = f"LOSSES{args.experiment_name}_{config_PT['n_layers']}_{config_PT['hidden_size']}_{config_PT['attn_heads']}{'_R' if config_PT['relative'] else ''}.pkl"
    losses_train_path = os.path.join(args.path_results,losses_train_path)
    with open(paths.idps_tests, 'rb') as f:
        idps_test = pickle.load(f)
    ds = Dataset_PT(paths, configs, idps_to_drop=idps_test)
    trainer = pretrainer(BERT(configs), ds, .20,  batch_size=args.batch_size, log_freq=None, model_path=None, lr=1e-4)
    for epoch in range(args.epochs):
        train_res = trainer.train(epoch, losses_train_path)
        test_res = trainer.test(epoch, losses_train_path)
        #with tune.checkpoint_dir(epoch) as checkpoint_dir: # uncomment this to save all models
                #path = os.path.join(checkpoint_dir, "checkpoint")
                #torch.save(trainer.model.state_dict(), path)
                #trainer.save_model(epoch, path)
        session.report(test_res) 
        
        
def main(args):
    
    train_func = tune.with_resources(lambda x: pretrain_model(x, args), {"gpu": 0.5})
    
    # BOHB scheduler
    algo = TuneBOHB(points_to_evaluate=config_init)
    #algo = tune.search.ConcurrencyLimiter(algo, max_concurrent=4)
    scheduler = HyperBandForBOHB(
        time_attr="epoch",
        max_t=args.epochs,
        reduction_factor=4,
        stop_last_trials=False)
    
    # ASHAS scheduler
    #scheduler = ASHAScheduler(
    #    metric="loss",
    #    mode="min",
    #    max_t=max_epochs,
    #    grace_period=3,
    #    reduction_factor=2)
    
    reporter = CLIReporter(
        parameter_columns=["hidden_size", "n_layers", "att_heads", 'relative'],
        metric_columns=["training_iteration", 'avg_loss', "avg_events_loss", "avg_acc_diags", "avg_acc_drugs"],
        max_report_frequency = 1200,
        sort_by_metric = 'True')
    
    tuner = tune.Tuner(
        train_func,
        tune_config=tune.TuneConfig(
                metric="avg_loss",
                mode="min",
                search_alg=algo,
                scheduler=scheduler,
                num_samples=args.samples),
        run_config=RunConfig(
                storage_path=args.path_results,
                name=args.experiment_name,
                stop={"training_iteration": args.epochs},
                progress_reporter=reporter),
        param_space=config_PT
        )
    if False: #args.restore:
        path_restore = os.path.join(args.path_results, args.experiment_name)
        tuner = tune.Tuner.restore(path=path_restore, trainable=train_func, param_space=config_PT)
    results = tuner.fit()
    
    
    
if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--samples', type=int, default=25)
    parser.add_argument('--restore',  action='store_true')
    parser.set_defaults(restore=False)
    parser.add_argument('--config_file', type=str, default='/home/enrico/DARE/Dare/Configs/configs_std.yaml')
    parser.add_argument('--paths_file', type=str, default='/home/enrico/DARE/Dare/Configs/data_paths.yaml')
    parser.add_argument('--path_results', type=str, default='/home/enrico/DARE/Results/PT')
    parser.add_argument('--experiment_name', type=str, default='experiment_0')
    
    args = parser.parse_args()
    os.makedirs(args.path_results, exist_ok=True)
    main(args)
        