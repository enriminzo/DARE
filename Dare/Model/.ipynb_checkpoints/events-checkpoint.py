import pandas as pd
import numpy as np
import json
import os
from omegaconf import OmegaConf

DATA_PATH = '../DARE/Dare/Configs/data_paths.yaml'
paths = OmegaConf.load(DATA_PATH)
with open(paths.cods_lists_path, 'r')  as f:
    cods_list = json.load(f)

class Event:
    '''
        Class to represent a singe event, containing TYPE, VALUE and TIME of the event. 
        Note that when using an event you always need a cods_list vocabulary with keys 'diags', 'drugs' and 'vars' 
    '''

    def __init__(self, cod, val, t):
        
        if type(cod) is list:
            self.tp = 'd'
            try:
                self.cod = [cods_list['diags'].index(c) for c in cod]
            except:
                print(cod)
                raise ValueError(f'{cod} is strange.')
            self.val = 1
        else:
            if cod in cods_list['diags']:
                self.tp = 'd'
                self.cod = cods_list['diags'].index(cod)
                self.val = 1
            elif cod in cods_list['drugs']:
                self.tp = 'm'
                self.cod = cods_list['drugs'].index(cod)
                self.val = 1
            elif cod in cods_list['vars']:
                self.tp = 'var'
                self.cod = cods_list['vars'].index(cod)
                self.val = val
            else:
                print(cod, type(cod))
                raise ValueError(f'{cod} is not a valid code.')
        self.t = t
        
        
    def __lt__(self, other):
        
        if self.t != other.t:
            return self.t < other.t
        elif self.tp != other.tp:
            return self.tp < self.tp
        else:
            if type(self.cod) is list or type(other.cod) is list:
                return self.val < other.val
            
            return self.cod < other.cod
    
    def __str__(self):
        return f"TYPE: {self.tp}, COD: {self.get_cod()}, VALUE: {self.val}, TIME: {self.t}"
    

    def get_type(self):
        
        return self.tp
    
    def get_type_num(self):
        
        if self.tp == 'd':
            return 1
        elif self.tp == 'm':
            return 2
        else:
            return 3
    
    def get_cod(self):
        
        if self.tp=='d':
            if type(self.cod) is list:
                cod_orig = [cods_list['diags'][c] for c in self.cod]
            else:
                cod_orig = cods_list['diags'][self.cod]
        elif self.tp=='m':
            cod_orig = cods_list['drugs'][self.cod]
        else:
            cod_orig = cods_list['vars'][self.cod]
        return cod_orig
        

    def get_val(self):
         
        v_d = np.zeros(np.size(cods_list['diags']))
        if self.tp=='d': 
            if type(self.cod) is list:
                for c in self.cod:
                     v_d[c] = self.val
            else:
                v_d[self.cod] = self.val
                
        v_m = np.zeros(np.size(cods_list['drugs']))
        if self.tp=='m': v_m[self.cod] = self.val
            
        v_var = np.zeros(np.size(cods_list['vars']))
        if self.tp=='var': v_var[self.cod] = self.val    
        return v_d, v_m, v_var

    def get_time(self):
        
        return self.t
    
    def get_time_from_start(self, date_0=0, unit='days'):
        
        if date_0==0:
            date_0 = f'{self.t[0:4]}-01-01'
        diff = pd.to_datetime(self.t) - pd.to_datetime(date_0)
        return diff.days

