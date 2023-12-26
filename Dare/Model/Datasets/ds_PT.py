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


class Dataset_PT(Dataset):
    def __init__(self, paths, config, idps_to_drop=None, years_to_drop = None):
        
        with open(paths.d0_m0_path, 'rb') as f:
            d0_m0 = pickle.load(f)
        self.diags = d0_m0['d0']
        self.drugs = d0_m0['m0']
        with open(paths.events_path, 'rb') as f:
            self.events_list = pickle.load(f)
        self.P0_gen = pd.read_csv(paths.P0_gen_path, index_col=0)
        
        if years_to_drop!=None:
            self.events_list = {y: data for y,data in self.events_list.items() if y not in years_to_drop}
            self.diags = {y: data for y,data in self.diags.items() if y not in years_to_drop}
            self.drugs = {y: data for y,data in self.drugs.items() if y not in years_to_drop}                
        if idps_to_drop!=None:
            self.P0_gen = self.P0_gen.loc[~self.P0_gen.idp.isin(idps_to_drop),]
            for y, dict_tmp in self.events_list.items():
                for idp in idps_to_drop:
                    dict_tmp.pop(idp, None)
        
                
        self.nmax = config.Embedding.nmax
        self.padd_time = config.Embedding.padd_indx_time
        self.padd_type = config.Embedding.padd_indx_type
        self.relative = config.Model.relative
        with open(paths.cods_labs_path) as f:
            cods_labs = json.load(f)
        self.diags_labs = cods_labs['diags_labs']
        self.drugs_labs = cods_labs['drugs_labs']
        self.masks = config.masks
        
        
    def __len__(self):
        return sum([len(events) for y, events in self.events_list.items()])
    
    def __getitem__(self, idx):
        
        y, idp = self.get_random_sequence(idx)
        #print(y, idp)
        evs_list = {idp: self.events_list[y][idp].copy()}
        p0, d0, m0, d_all, m_all, v_all, t_all, types_all = get_inputs([idp], y, self.P0_gen, self.diags, self.drugs, evs_list,
                                                                       self.nmax, self.padd_time, self.padd_type, False)
            
        p0, p0_labs, m0, m0_labs, d0, d0_labs, v_all, v_all_labs = self.mask_sequence(p0, d0, m0, v_all)
        
        
        if self.relative:
            dists = self.get_distance_matrix(t_all) #np.apply_along_axis(self.get_distance_matrix, 0, t_all)
            return {
            'd_all': torch.tensor(d_all).float(),
            'm_all': torch.tensor(m_all).float(),
            'v_all': torch.tensor(v_all).float(),
            't_all': torch.tensor(t_all).long(),
            'types_all': torch.tensor(types_all).long(),
            'p0': torch.tensor(p0).float(),
            'd0': torch.tensor(d0).float(),
            'm0': torch.tensor(m0).float(),
            'dists': torch.tensor(dists).long(),
            'p0_labs': torch.tensor(p0_labs).float(),
            'm0_labs': torch.tensor(m0_labs).float(),
            'd0_labs': torch.tensor(d0_labs).float(),
            'v_all_labs': torch.tensor(v_all_labs).float()}
        else: 
            return {
            'd_all': torch.tensor(d_all).float(),
            'm_all': torch.tensor(m_all).float(),
            'v_all': torch.tensor(v_all).float(),
            't_all': torch.tensor(t_all).long(),
            'types_all': torch.tensor(types_all).long(),
            'p0': torch.tensor(p0).float(),
            'd0': torch.tensor(d0).float(),
            'm0': torch.tensor(m0).float(),
            'p0_labs': torch.tensor(p0_labs).float(),
            'm0_labs': torch.tensor(m0_labs).float(),
            'd0_labs': torch.tensor(d0_labs).float(),
            'v_all_labs': torch.tensor(v_all_labs).float()}
    
    def get_random_sequence(self, idx):
    
        for y, seqs in self.events_list.items():
            if idx < (len(seqs)):
                idp = list(seqs.keys())[idx]
                return y, idp
            else:
                idx -= len(seqs)
                
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
    
    def mask_sequence(self, p0, d0, m0, seqs):
        
        #1) MLM: mask variabels events:
        seqs, seqs_labs = self.MLM_evsprediction(seqs, 0.15)
        
        #2) NSP: mask P0 vectors
        p0, p0_labs, m0, m0_labs, d0, d0_labs = self.NSP_maskP0(p0, d0, m0)
                
        return p0, p0_labs, m0, m0_labs, d0, d0_labs, seqs, seqs_labs
    
    def NSP_maskP0(self,  p0, d0, m0):
        d0_labs = np.ones(len(self.diags_labs.keys()))*-1
        m0_labs = np.ones(len(self.drugs_labs.keys()))*-1
        p0_labs = np.zeros(p0[[1,2]].shape)
        prob = random.random()
        d_mask, m_mask, p_mask = self.masks['diags'], self.masks['drugs'], self.masks['age']
        if prob< 0.5: # change stuff in half of the patients
            
            if prob < 0.1: # mask diag and mask
                d0_labs = [int(len(set(np.where(d0==1)[0]).intersection(pos))>0) for diag, pos in self.diags_labs.items()]
                d0 = np.ones(d0.shape)*self.masks['diags']
                
                m0_labs = [int(len(set(np.where(m0==1)[0]).intersection(pos))>0) for diag, pos in self.drugs_labs.items()]
                m0 = np.ones(m0.shape)*self.masks['drugs']
                
            elif prob< 0.3: # mask drugs
                m0_labs = [int(len(set(np.where(m0==1)[0]).intersection(pos))>0) for diag, pos in self.drugs_labs.items()]
                m0 = np.ones(m0.shape)*self.masks['drugs']
                
            else: # mask diag  
                d0_labs = [int(len(set(np.where(d0==1)[0]).intersection(pos))>0) for diag, pos in self.diags_labs.items()]
                d0 = np.ones(d0.shape)*self.masks['diags']
                
                
        return p0, p0_labs, m0, m0_labs, d0, d0_labs
    
    def MLM_evsprediction(self, seqs, p=0.15):
        
        seqs_labs = np.zeros(seqs.shape)
        mask_loss = np.zeros(1)
        for i in range(seqs.shape[0]):
            ev = seqs[i,:]
            if sum(ev)>0: # event is a variable
                mask_loss = 1
                prob = random.random()
                if prob < p:
                    prob /= p
                    seqs_labs[i,:] = seqs[i,:]                    
                    if prob < 0.7: # mask the event
                        seqs[i,ev!=0] = self.masks['vars']
                    elif prob < 0.85: # add perturbation
                        seqs[i,ev!=0] += random.gauss(0,0.1)
                        
        return seqs, seqs_labs
