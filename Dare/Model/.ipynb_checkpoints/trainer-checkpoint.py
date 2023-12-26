import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn.functional import binary_cross_entropy as BCE

from Dare.Model.bert_pretraining import BERT_PT
from Dare.Model.bert import BERT
from Dare.Model.utils_FT import EarlyStopping
from Dare.Model.optim_schedule import ScheduledOptim

import numpy as np
import tqdm
import warnings
import os
import pickle
from sklearn.metrics import roc_auc_score


#--------------------------------------------------------------------------------------
#---------- PRETRAING TRAINER + LOSS --------------------------------------------------
#--------------------------------------------------------------------------------------


class MaskedMSELoss(torch.nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, prediction, target, mask):
        diff = (torch.flatten(prediction) - torch.flatten(target)) ** 2.0 * torch.flatten(mask)
        loss = torch.sum(diff) / torch.sum(mask)
        if torch.isnan(loss):
            return torch.tensor(0)
        
        return loss


class pretrainer:
    def __init__(self, bert, dataset, val_perc,
                 n_diags=None, n_drugs=None, n_vars=None,
                 batch_size=32, lr=1e-4, betas=(0.9, 0.999), weight_decay=0.01, warmup_steps=10000, log_freq=10, 
                 model_path=None, device=None):
        
        # split data in test and validation
        indices = list(range(len(dataset)))
        split = int(np.floor(val_perc*len(dataset)))
        np.random.seed(123)
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)
        self.train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        self.validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)
        
        # define devices and model
        
        self.bert = bert
        self.model = BERT_PT(self.bert, n_diags, n_drugs, n_vars)
        self.parallel_gpu=False
        if device==None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.parallel_gpu=False
        elif device=='parallel':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = nn.DataParallel(self.model)
            self.parallel_gpu=True
        elif device=='cpu':
            self.device = torch.device("cpu")
            self.parallel_gpu=False
        else:
            self.device = torch.device("cuda")
            torch.cuda.set_device(device)
            self.parallel_gpu=False
            
            
        if model_path!=None:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device)['model_state_dict'])
        self.model.to(self.device)
        
        # define optimizer
        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(self.optim, bert.hidden, n_warmup_steps=warmup_steps)
        
        # define Losses
        self.L_events = MaskedMSELoss()
        self.log_freq = log_freq
        
    def train(self, epoch, res_path=None, return_losses=False, mininterval=None):
        
        to_ret = self.iteration(epoch, self.train_loader,True,res_path=res_path, mininterval=mininterval)
        if return_losses:
            return to_ret

    def test(self, epoch, res_path=None, return_losses=True, mininterval=None):
        
        to_ret = self.iteration(epoch, self.validation_loader, train=False, res_path=res_path, mininterval=mininterval)
        if return_losses:
            return to_ret

    def iteration(self, epoch, data_loader, train=True, res_path = None, mininterval=None):
        
        # Setting the tqdm progress bar
        str_code = "train" if train else "test"  
        if not train:
            self.model.eval()
        if mininterval is not None:
            data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}",  mininterval=600)
        else:
            data_iter = enumerate(data_loader)
        
        warnings.filterwarnings('ignore', "Max number of events reached, some events were removed!")
        avg_loss, avg_p0_loss, avg_d_loss, avg_m_loss, diags_corr, diags_tot, drugs_corr, drugs_tot, avg_events_loss  = 0, 0, 0, 0, 0, 0, 0, 0, 0
        for i, data in data_iter:
            # 0. batch_data will be sent into the device(GPU or cpu)
            data = {key: value.to(self.device) for key, value in data.items()}
            # 1. forward
            if self.bert.relative:
                ei_pred, d0_pred, m0_pred= self.model.forward(data['p0'], data['d0'], data['m0'], data['d_all'], data['m_all'], data['v_all'], data['t_all'] , data['types_all'], data['dists'])
            else:
                ei_pred, d0_pred, m0_pred= self.model.forward(data['p0'], data['d0'], data['m0'], data['d_all'], data['m_all'], data['v_all'], data['t_all'] , data['types_all'], None)
            # 2. get losses
            events_loss = self.L_events(ei_pred, data['v_all_labs'], data['v_all_labs']!=0)
            if events_loss==0:
                warnings.warn(f'Event loss=0 in epoch {epoch} batch {i}')
                print(f'Event loss=0 in batch {i}')
            #p0_loss = self.L_events(p0_pred, data['p0_labs'], data['p0_labs']!=0)
            d0_loss = BCE(d0_pred, data['d0_labs'], weight=data['d0_labs']!=-1)
            m0_loss = BCE(m0_pred, data['m0_labs'], weight=data['m0_labs']!=-1)
            loss = events_loss + 1*(d0_loss + m0_loss) 
            
            # 3. backward and optimization only in train
            if train:
                self.optim_schedule.zero_grad()
                loss.backward()
                self.optim_schedule.step_and_update_lr()
                
            # 4. get evaluation metrics
            
            rows_tmp = (data['d0_labs']!=-1)[:,0]
            diags_corr += (d0_pred[rows_tmp,:]>0.5).eq(data['d0_labs'][rows_tmp,:]).sum().item()
            diags_tot += data['d0_labs'][(data['d0_labs']!=-1)[:,0]].nelement()
            rows_tmp = (data['m0_labs']!=-1)[:,0]
            drugs_corr += (m0_pred[rows_tmp,:]>0.5).eq(data['m0_labs'][rows_tmp,:]).sum().item()
            drugs_tot += data['m0_labs'][(data['m0_labs']!=-1)[:,0]].nelement()
            avg_loss += loss.item()
            #avg_p0_loss += p0_loss.item()
            avg_d_loss += d0_loss.item()
            avg_m_loss += m0_loss.item()
            avg_events_loss += events_loss.item()
            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": round(avg_loss / (i+1), 3),
                #"avg_p0_loss": round(avg_p0_loss / (i+1), 3),
                "avg_events_loss": round(avg_events_loss / (i+1), 3),
                "avg_d0_loss": round(avg_d_loss / (i+1), 3),
                "avg_m0_loss": round(avg_m_loss / (i+1), 3),
                "avg_acc_diags": round(diags_corr / diags_tot * 100, 3),
                "avg_acc_drugs": round(drugs_corr / drugs_tot * 100, 3),
                
            }
            
            if self.log_freq!=None:
                if i % self.log_freq == 0:
                    print(i)
                    data_iter.write(str(post_fix))
        
        #end of epoch, save results
        epoch_res = {
                "epoch": epoch,
                "avg_loss": avg_loss / len(data_loader),
                #"avg_p0_loss": avg_p0_loss / len(data_loader),
                "avg_events_loss": avg_events_loss / len(data_loader),
                "avg_d0_loss": avg_d_loss / len(data_loader),
                "avg_m0_loss": avg_m_loss / len(data_loader),
                "avg_acc_diags": diags_corr / diags_tot * 100, 
                "avg_acc_drugs": drugs_corr / drugs_tot * 100,
            }
        
        if not train:
            self.model.train()
            
        if res_path is not None:
            # load old data results
            if os.path.exists(res_path):
                with open(res_path, 'rb') as f:
                    res_all = pickle.load(f)
            else:
                res_all={}
            #upload results
            if train:
                k = f'{epoch}-train'
            else:
                k = f'{epoch}-test'
            res_all[k] = epoch_res
            with open(res_path, 'wb') as outp:
                pickle.dump(res_all, outp)
        
        return epoch_res
        
        
    def save_model(self, epoch, file_path = '/home/enrico/transformer/results/bert_trained.model'):
        
        #output_path = file_path + ".ep%d" % epoch
        file_path = file_path + ".ep%d" % epoch
        if self.parallel_gpu:
            torch.save({'epoch': epoch, 'model_state_dict': self.model.module.state_dict(),
                        'optimizer_state_dict': self.optim.state_dict(), 'scheduler': self.optim_schedule},
                       file_path)
        #    torch.save(self.model.module.state_dict(), output_path)
        else:
            torch.save({'epoch': epoch, 'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optim.state_dict(), 'scheduler': self.optim_schedule},
                   file_path)
        #    torch.save(self.model.state_dict(), output_path)
        
        print("EP:%d Model Saved on:" % epoch, file_path)
        
#--------------------------------------------------------------------------------------
#---------- FINETUNIG TRAINER + LOSS --------------------------------------------------
#--------------------------------------------------------------------------------------        

class WeigthedBCELoss(nn.Module):
    '''
        Weighted Binary Cross Entropy Loss for fine tuning
    '''
    
    def __init__(self, mask_val=-1, weights=None, device=None ):
        super(WeigthedBCELoss, self).__init__()
        self.loss_fun = nn.BCELoss(reduction = 'none')
        self.weights = weights
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
    def forward(self, prediction, target, pred_length=4):
        
        if self.weights is not None:
            bs,_ = target.shape
            w1 = np.tile(self.weights[1,:], (bs,pred_length))
            w1 = torch.tensor(w1).to(self.device)*(target==1)
            w0 = np.tile(self.weights[0,:], (bs,pred_length))
            w0 = torch.tensor(w0).to(self.device)*(target==0)
            loss = self.loss_fun(prediction, target)*(w0+w1)
        else:
            loss = self.loss_fun(prediction, target)
            
        return torch.mean(loss)
    
    
        
class finetuner:
    def __init__(self,bert, model, trainloader, valloader, device=None, weights_classes=None,  predict='diags'):
        
        self.bert = bert
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.optim = torch.optim.Adam(model.parameters())
        self.es = EarlyStopping()
        self.predict = predict
        vars_to_use = ['p0', 'd0', 'm0', 'd_all','m_all', 'v_all', 't_all', 'types_all', 'dists']
        if predict == 'diags':
            vars_to_use = ['d0_2015', 'd0_2016', 'd0_2017', 'd0_2018']
        elif predict == 'drugs':
            vars_to_use = ['m0_2015', 'm0_2016', 'm0_2017', 'm0_2018']
        elif predict == 'targets':
            vars_to_use = ['hba1c_targets']
        else:
            err_msg = f'Prediction {predict} not valid, pleaseprovide one of "diags", "drugs" ot "targets"'
            raise ValueError(err_msg)
            
        if bert is None:
            vars_to_use = ['rnn_inputs'] + vars_to_use
        else:
            vars_to_use = ['p0', 'd0', 'm0', 'd_all','m_all', 'v_all', 't_all', 'types_all', 'dists'] + vars_to_use
        self.vars_to_use = vars_to_use
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.loss = WeigthedBCELoss(weights=weights_classes, device=self.device) 
            
        
    def train_iteration(self, epoch, mininterval=None, print_res=True):
        
        if mininterval is not None:
            data_iter = tqdm.tqdm(enumerate(self.trainloader),
                              desc="EP_%s:%d" % ('train', epoch),
                              total=len(self.trainloader),
                              bar_format="{l_bar}{r_bar}",  mininterval=mininterval)
        else:
            data_iter = enumerate(self.trainloader)
        
        warnings.filterwarnings('ignore', "Max number of events reached, some events were removed!")
        avg_loss = 0
        for i, data in data_iter:
            # 0. batch_data will be sent into the device(GPU or cpu)
            data = {key: value.to(self.device) for key, value in data.items() if key in self.vars_to_use}
            # 1. forward
            if self.bert is not None:
                outs = self.bert.forward(data['p0'], data['d0'], data['m0'], data['d_all'], data['m_all'], data['v_all'], data['t_all'] , data['types_all'], data['dists'])
            else:
                outs = data['rnn_inputs']
            outs = self.model.forward(outs)
            # 2. get loss
            if self.predict == 'diags':
                labs = torch.cat((data['d0_2015'], data['d0_2016'], data['d0_2017'], data['d0_2018']), dim=1).float()
            elif self.predict == 'drugs':
                labs = torch.cat((data['m0_2015'], data['m0_2016'], data['m0_2017'], data['m0_2018']), dim=1).float()
            else:
                labs = data['hba1c_targets'].float()
            batch_loss = self.loss(outs, labs)
            # 3. backpropagation
            self.optim.zero_grad()
            batch_loss.backward()
            self.optim.step()
            # 4. print losses 
            avg_loss += batch_loss.item()
            post_fix = {
                    "epoch": epoch,
                    "iter": i,
                    "avg_loss": round(avg_loss / (i+1), 3)}
        if print_res:
            print('END TRAIN EPOCH')
            print(str(post_fix))
            
        return post_fix
    
    def test_iteration(self, epoch, mininterval=None, print_res=True):
        
        if mininterval is not None:
            data_iter = tqdm.tqdm(enumerate(self.valloader),
                              desc="EP_%s:%d" % ('test', epoch),
                              total=len(self.valloader),
                              bar_format="{l_bar}{r_bar}",  mininterval=mininterval)
        else:
            data_iter = enumerate(self.valloader)
        warnings.filterwarnings('ignore', "Max number of events reached, some events were removed!")
        avg_loss, avg_roc = 0, 0
        self.model.eval()
        i_tmp = 0
        idps_all = []
        for i, data in data_iter:
            # 0. batch_data will be sent into the device(GPU or cpu)
            idps_tmp = data['idp']
            data = {key: value.to(self.device) for key, value in data.items() if key in self.vars_to_use}
            # 1. forward
            with torch.no_grad():
                if self.bert is not None:
                    outs = self.bert.forward(data['p0'], data['d0'], data['m0'], data['d_all'], data['m_all'], data['v_all'], data['t_all'] , data['types_all'], data['dists'])
                else:
                    outs = data['rnn_inputs']
                outs = self.model.forward(outs)
                # 2. get loss
                if self.predict == 'diags':
                    labs = torch.cat((data['d0_2015'], data['d0_2016'], data['d0_2017'], data['d0_2018']), dim=1).float()
                elif self.predict == 'drugs':
                    labs = torch.cat((data['m0_2015'], data['m0_2016'], data['m0_2017'], data['m0_2018']), dim=1).float()
                else:
                    labs = data['hba1c_targets'].float()
                batch_loss = self.loss(outs, labs)
            # 3. print losses 
            avg_loss += batch_loss.item()
            idps_all += idps_tmp
            try:
                outs_all = torch.cat((outs_all, outs.cpu()))
                labs_all = torch.cat((labs_all, labs.cpu()))
            except Exception as e:
                outs_all = outs.cpu()
                labs_all = labs.cpu()
        
            post_fix = {
                    "epoch": epoch,
                    "iter": i,
                    "avg_loss": avg_loss / (i+1)}
        # Get total AUC and AUC by year
        try:
            avg_roc = roc_auc_score(labs_all, outs_all)
        except Exception as e: 
            avg_roc = np.nan
            warn_msg = f'NAN for batch {i}, epoch {epoch}'
            warnings.warn(warn_msg)
        post_fix['avg_roc'] = avg_roc
        if print_res:
            print('END TEST EPOCH')
            print(str(post_fix))
        
        for ind in range(labs_all.shape[1]):
            try:
                post_fix[f'avg_roc_{2014+ind}'] = roc_auc_score(labs_all[:,ind], outs_all[:,ind])
            except Exception as e: 
                post_fix[f'avg_roc_{2014+ind}'] = np.nan
    
        self.model.train()
        return post_fix, labs_all, outs_all, idps_all