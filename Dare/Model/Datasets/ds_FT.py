import numpy as np
import pandas as pd
import pickle  
import torch
from torch.utils.data import Dataset
import random
from omegaconf import OmegaConf
import json
from Dare.Model.events import Event
from Dare.Model.Datasets.utils import get_inputs

class Dataset_FT(Dataset):
    def __init__(self, paths, config, idps=None, return_rnn_inps=False):
        
        with open(paths.d0_m0_path, 'rb') as f:
            d0_m0 = pickle.load(f)
        self.diags = d0_m0['d0']
        self.drugs = d0_m0['m0']
        with open(paths.events_path, 'rb') as f:
            self.events_list = pickle.load(f)
        self.P0_gen = pd.read_csv(paths.P0_gen_path, index_col=0)
        if idps!=None:
            self.P0_gen = self.P0_gen.loc[self.P0_gen.idp.isin(idps),]
            self.P0_gen = self.P0_gen.reset_index()
                
        self.nmax = config.Embedding.nmax
        self.padd_time = config.Embedding.padd_indx_time
        self.padd_type = config.Embedding.padd_indx_type
        self.relative = config.Model.relative
        with open(paths.cods_labs_path) as f:
            cods_labs = json.load(f)
        self.diags_labs = cods_labs['diags_labs']
        self.drugs_labs = cods_labs['drugs_labs']
        self.masks = config.masks
        with open(paths.cods_lists_path) as f:
            cods_lists = json.load(f)
        self.vars_list = cods_lists['vars']
        self.return_rnn = return_rnn_inps
        
    def __len__(self):
        return len(self.P0_gen)
    
    def __getitem__(self, idx):
        
        y = min(self.events_list.keys())
        idp = self.P0_gen.iloc[idx]['idp']
        evs_list = {idp: self.events_list[y][idp].copy()}
        p0, d0, m0, d_all, m_all, v_all, t_all, types_all = get_inputs([idp], y, self.P0_gen, self.diags, self.drugs, evs_list,
                                                                       self.nmax, self.padd_time, self.padd_type, False)
       
        
        years = [y_tmp for y_tmp in self.events_list.keys() if y_tmp!=y]
        labs_out = self.get_d_m_outputs(idp, [y+1 for y in years])
        hba1c_out = self.get_hba1c_outputs(idp, years)
        hba1c_targets = self.get_hba1c_targets(idp)
        
        to_return = {
            'idp': idp,
            'd_all': torch.tensor(d_all).float(),
            'm_all': torch.tensor(m_all).float(),
            'v_all': torch.tensor(v_all).float(),
            't_all': torch.tensor(t_all).long(),
            'types_all': torch.tensor(types_all).long(),
            'p0': torch.tensor(p0).float(),
            'd0': torch.tensor(d0).float(),
            'm0': torch.tensor(m0).float(),
            }
        if self.relative:
            dists = self.get_distance_matrix(t_all) #np.apply_along_axis(self.get_distance_matrix, 0, t_all)
            to_return['dists'] = torch.tensor(dists).long()
        else:
            to_return['dists'] = None
        if self.return_rnn:
            to_return['rnn_inputs'] = self.get_rnn_inps(to_return)
            
        return {**to_return, **labs_out, **hba1c_out, **hba1c_targets}
                
    def get_distance_matrix(self, v):
        """
            Get a distance matrix from a vector v 
        """
        dist = np.empty((len(v),len(v)),dtype=int) 
        for i in range(len(v)):
            dist[i,:] = np.abs(v[i] - v)
        dist[v==self.padd_time, :] = self.padd_time
        dist[:,v==self.padd_time] = self.padd_time
        return dist
    
    def get_d_m_outputs(self, idp, years):
        labs = {}
        for y in years:
            d0 = self.diags[y][idp]
            labs[f'd0_{y}'] = torch.tensor([int(len(set(np.where(d0==1)[0]).intersection(pos))>0) for diag, pos in self.diags_labs.items()])
            m0 = self.drugs[y][idp]
            labs[f'm0_{y}'] = torch.tensor([int(len(set(np.where(m0==1)[0]).intersection(pos))>0) for diag, pos in self.drugs_labs.items()])
        return labs
    
    def get_hba1c_outputs(self, idp, years):
        indx = self.vars_list.index('HBA1C')
        outs = []
        for y in years:
            evs = self.events_list[y][idp]
            mean_hbaic, n_hba1c = 0, 0
            for ev in evs:
                if ev.get_val()[2][indx] !=0:
                    mean_hbaic += ev.get_val()[2][indx]
                    n_hba1c += 1
            outs.append(mean_hbaic/n_hba1c)
        return {'hba1c_outs': torch.tensor(outs)}
    
    def get_hba1c_targets(self, idp):
        indx = self.vars_list.index('HBA1C')
        outs = []
        target = -0.299 # = 7%
        for y in [2014,2015,2016,2017]:# [2013,2014]:
            evs = self.events_list[y][idp]
            mean_hbaic, n_hba1c = 0, 0
            for ev in evs:
                if ev.get_val()[2][indx] !=0:
                    mean_hbaic += ev.get_val()[2][indx]
                    n_hba1c += 1
            outs.append(mean_hbaic/n_hba1c)
            
        return {'hba1c_targets': torch.tensor([tr<=target for tr in outs]).float()} #, 'hba1c_class': pos}
    
    def get_rnn_inps(self, data):
        
        #get single output
        outs0 = torch.cat((torch.zeros(1), data['p0'], torch.zeros(data['v_all'].shape[1]), data['m0'], data['d0']))
        outs1 = torch.cat((data['t_all'][3:][:,None], data['p0'].repeat(40,1), data['v_all'], data['m_all'], data['d_all']), dim=1)
        outs1 = outs1[outs1[:,0].sort()[1]]
        outs1 = torch.cat((outs0[None,:], outs1))

        return outs1