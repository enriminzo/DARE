import numpy as np
import pandas as pd
import json
import warnings
import torch

DATA_DIR = '/home/enrico/DARE/Data/'
with open(DATA_DIR+'cods_lists.json', 'r') as f:
    cods_list = json.load(f)

#-------------------------------------------------------------------------------------------------------

def get_inputs_from_list(evs, n_max, padd_time, padd_type, ordered=True):
    
    """
        Return numpy matrices that can be used as input from a list of events.
        INPUTS: 
            - evs: list of events of a single patient
            - n_max: max number of events for the list. If len(events_list)<n_max matrices are padded.
            - padd_time: padding value fot times vectors
            - padd_type: padding value for type vectors
        RETURNS:
            -d_idp: matrix of diagnosis of shape [n_max, n_diags]
            -m_idp: matrix of medications of shape [n_max, n_drugs]
            -v_idp: matrix of variables of shape [n_max, n_vars]
            -times_idp: matrix of times of the event (respect to 01/01 of the same year) of shape [n_max, 1]
            -types_idp: matrix of types of the events (1=diagnosis, 2=medication, 3=variable) of shape [n_max, 1]
    """
    
    #if len(evs)>n_max:
    #    warnings.warn("Max number of events reached, some events were removed!")
        
    d_idp, m_idp, v_idp, times_idp, types_idp = [],[],[],[],[]
    n_evs = len(evs)
    for i in range(n_max):
        if i<n_evs:
            ev = evs[i]
            d_tmp, m_tmp, v_tmp = ev.get_val()
            times_idp.append(ev.get_time_from_start())
            types_idp.append(ev.get_type_num())
        else:
            d_tmp, m_tmp, v_tmp = np.zeros(d_tmp.shape), np.zeros(m_tmp.shape), np.zeros(v_tmp.shape)
            times_idp.append(padd_time)
            types_idp.append(padd_type)
        d_idp.append(d_tmp)
        m_idp.append(m_tmp)
        v_idp.append(v_tmp)
    d_idp = np.concatenate(d_idp).reshape((n_max,len(d_tmp)))
    m_idp = np.concatenate(m_idp).reshape((n_max,len(m_tmp)))
    v_idp = np.concatenate(v_idp).reshape((n_max,len(v_tmp)))
    times_idp = np.array(times_idp).reshape((n_max,1))
    types_idp = np.array(types_idp).reshape((n_max,1))
    
    if ordered:
        orders = np.argsort(times_idp,0)
        times_idp = times_idp[orders[:,0]]
        types_idp = types_idp[orders[:,0]]
        d_idp = d_idp[orders[:,0]]
        v_idp = v_idp[orders[:,0]]
        m_idp = m_idp[orders[:,0]]
    return d_idp, m_idp, v_idp, times_idp, types_idp


#-------------------------------------------------------------------------------------------------------

def get_pO_matrix(idps_y, y, P0, time_unit='Y', norm=True):
    """
        Get p0 matrix for a list of patients.
        INPUTS: 
            - idps_y: idps_list
            - y: starting time (year) to be cosidereted as t0
            - P0: matirx of general info of patients, with columns idp, dnaix, sexe, diag_date
            - time_unit: time to calculate age. 
        OUTPUTS:
            - p0: dataframe with columns age, sex and time from diagnosis and indices idps.
        
    """
    
    p0 = P0.loc[P0.idp.isin(idps_y),:].copy()
    p0['sex'] = (p0.sexe=='D').astype(int)
    if norm:
        p0['age'] = p0[f'age_{y}']
        p0['tfd'] = p0[f'tfd_{y}']
    else:
        p0['age'] = p0.dnaix.apply(lambda x: (pd.to_datetime(f'{y}-01-01') - pd.to_datetime(x))/np.timedelta64(1,time_unit))
        p0['tfd'] = p0.diag_date.apply(lambda x: (pd.to_datetime(f'{y}-01-01') - pd.to_datetime(x))/np.timedelta64(1,time_unit))
    p0 = p0.set_index('idp')
    p0 = p0.loc[idps_y,['sex', 'age', 'tfd']]
    
    return p0

#-------------------------------------------------------------------------------------------------------

def get_inputs(idps, y, P0, diags, drugs, evs_list, n_max, padd_time=366, padd_type=6, tensors=True):
    """
        Get tensors to be feed in a InputEmbedding layer. 
        INPUTS: 
            - idps: list of idps 
            - y: starting time (year) to be cosidereted as t0
            - P0: matirx of general info of patients, with columns idp, dnaix, sexe, diag_date
            - diags: matrix of diangosis, with columns idp, cod, dat, dbaixa
            - drugs: matrix of drug usage, with columns idp, start, end, cod
            - evs_list: dict of events with structure idp:[e1,e2,..]
            - n_max: max number of events fot the list. If len(events_list)<n_max matrices are padded.
            - padd_time: padding value fot times vectors
            - padd_type: padding value for type vectors
            - tensors: boolean, should return vectors as tensor?
        OUTPUTS:
            - p0: represents patient at time y, tensor with shape (n_idp, 3) with columns sex, age, time from diag
            - d0: diagnosis at time y, tensor with shape (n_idp, n_diag)
            - m0: drugs at time y, tensor with shape (n_idp, n_drugs)
            - d_all: diagnosis events, tensor with shape (n_idp, n_max, n_diag)
            - m_all: drugs prescriptions events, tensor with shape (n_idp, n_max, n_drugs)
            - v_all: variable events, tensor with shape (n_idp, n_max, n_vars)
            - t_all: time between y and events (days), tensor with shape (n_idp, n_max+3)
            - type_all: type of each event, tensor with shape (n_idp, n_max+3)
     """
    
    # 1) get P0 matrix
    p0 = get_pO_matrix(idps, y, P0)
    p0 = np.squeeze(p0.values)
    
    # 2) get d0 and m0 matrices
    d0 = diags[y]
    d0 = np.vstack([d0[idp] for idp in idps])
    d0 = np.squeeze(d0)
    m0 = drugs[y]
    m0 = np.vstack([m0[idp] for idp in idps])
    m0 = np.squeeze(m0)
    
    # 3) get other matrices
    d_all, m_all, v_all, t_all, types_all = [],[],[],[],[]
    for idp in idps:
        d_idp, m_idp, v_idp, times_idp, types_idp = get_inputs_from_list(evs_list[idp], n_max, padd_time, padd_type)
        d_all.append(d_idp)
        m_all.append(m_idp)
        v_all.append(v_idp)
        t_all.append(times_idp)
        types_all.append(types_idp)
    d_all = np.squeeze(np.concatenate(d_all).reshape((len(idps), n_max, d_idp.shape[1])))
    m_all = np.squeeze(np.concatenate(m_all).reshape((len(idps), n_max, m_idp.shape[1])))
    v_all = np.squeeze(np.concatenate(v_all).reshape((len(idps), n_max, v_idp.shape[1])))
    t_all = np.concatenate(t_all).reshape((len(idps), n_max))
    types_all = np.concatenate(types_all).reshape((len(idps), n_max))
    
    # 4) add times and types for P0, d0, m0
    t_init = np.zeros((len(idps), 3))
    types_init = [[0,1,2] for i in range(len(idps))]
    types_init = np.concatenate(types_init).reshape((len(idps), 3))
    t_all = np.squeeze(np.concatenate([t_init,t_all],axis=1))
    types_all = np.squeeze(np.concatenate([types_init, types_all], axis=1))
    
    # 5) get tensors
    if tensors:
        d_all = torch.tensor(d_all).float()
        m_all = torch.tensor(m_all).float()
        v_all = torch.tensor(v_all).float()
        t_all = torch.tensor(t_all).long()
        types_all = torch.tensor(types_all).long()
        p0 = torch.tensor(p0).float()
        d0 = torch.tensor(d0).float()
        m0 = torch.tensor(m0).float()
        
    return p0, d0, m0, d_all, m_all, v_all, t_all, types_all    
    
    
    

        