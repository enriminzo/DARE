""" 
Get p0 vectors for t0 = 2013--2017
"""

import sys
sys.path.append('/home/enrico/transformer_final/')

import pandas as pd
import numpy as np
import pickle
import os
import argparse
import json
import time

def load_table(tbl_name, data_dr, sep='|', parser_dates=['dat']):
    dr = os.path.join(data_dr, tbl_name)
    tbl = pd.read_table(dr, sep=sep, parse_dates=parser_dates)
    return tbl

def get_d0_m0(idps_y, y, diags, drugs):
    '''
        Function to calculate d0 and m0 vectors at year y, 
        i.e. diagnosis and drugs used for each patient at beginnig of the year
        INPUTS:
            - idps_y: idps_list
            - y: starting time (year) to be cosidereted as t0
            - diags: matrix of diangosis, with columns idp, cod, dat, dbaixa
            - drugs: matrix of drug usage, with columns idp, start, end, cod
        OUPUTS:
            - df_diags_y: dataframe with indices the idps and columns ITC codes of diangosis
            - df_drugs_y: dataframe with indices the idps and columns ATC codes for medicines
            
    '''
    
    
    # get diagnosis before y but still active
    df_tmp = diags.loc[(diags.idp.isin(idps_y)) & (diags.dat<f'{y}-01-01') & (diags.cod.isin(cods_list['diags'])),:]
    df_tmp = df_tmp[['idp','cod']].drop(df_tmp[df_tmp.dbaixa< f'{y}-01-01'].index)
    df_tmp['val'] = 1
    df_diags_y = df_tmp.pivot_table(index='idp', columns='cod', fill_value=0)
    df_diags_y.columns = [col[1] for col in df_diags_y.columns]
        # add all diagnosis with 0
    for d in cods_list['diags']:
        if d not in df_diags_y.columns:
            df_diags_y[d] = 0  
    df_diags_y = df_diags_y[cods_list['diags']]
        # add all patients
    idps_tmp = [idp for idp in idps_y if idp not in df_diags_y.index]
    df_tmp = pd.DataFrame(0, index=idps_tmp, columns=cods_list['diags'])
    df_diags_y = df_diags_y.append(df_tmp)
    
    # gut drugs before y but still used
    df_tmp = drugs.loc[(drugs.idp.isin(idps_y)) & (drugs.start<f'{y}-01-01') & (drugs.cod.isin(cods_list['drugs'])),:]
    df_tmp = df_tmp[['idp','cod']].drop(df_tmp[df_tmp.end< f'{y}-01-01'].index)
    df_tmp['val'] = 1
    df_drugs_y = df_tmp.pivot_table(index='idp', columns='cod', fill_value=0)
    df_drugs_y.columns = [col[1] for col in df_drugs_y.columns]
        # add all diagnosis with 0
    for d in cods_list['drugs']:
        if d not in df_drugs_y.columns:
            df_drugs_y[d] = 0  
    df_drugs_y = df_drugs_y[cods_list['drugs']]
        # add all patients
    idps_tmp = [idp for idp in idps_y if idp not in df_drugs_y.index]
    df_tmp = pd.DataFrame(0, index=idps_tmp, columns=cods_list['drugs'])
    df_drugs_y = df_drugs_y.append(df_tmp)
    
    return df_diags_y, df_drugs_y

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--orig_data_dir', type=str, default='/home/enrico/Desktop/dm2_data/original_tables/')
    parser.add_argument('--table_pop', type=str, default='ALGDM2_entregable_poblacio_20190620_081435.txt')
    parser.add_argument('--data_dir', type=str, default='/home/enrico/transformer_final/Data/')
    args = parser.parse_args()
    
    # LOAD DATA
    with open(os.path.join(args.data_dir, 'cods_lists.json'), 'r') as f:
        cods_list = json.load(f)
    
    df_gen = load_table(args.table_pop, args.orig_data_dir, parser_dates=['dnaix', 'sortida'])
    diags = pd.read_csv(os.path.join(args.data_dir,'diags_events.csv'), index_col=0)
    drugs = pd.read_csv(os.path.join(args.data_dir,'drugs_events_15d.csv'), index_col=0)
    with open(os.path.join(args.data_dir,'events_list_std.pkl'), 'rb') as f:
        events_list = pickle.load(f)
        
        
    # GET P0 MATRIX
    idps_all = [list(events_list[y].keys()) for y in events_list.keys()]
    idps_all = set([idp for idps_y in idps_all for idp in idps_y])
    
    diags_first = diags.loc[(diags.cod.str.contains('E11|E14')) & (diags.idp.isin(idps_all)),:]
    diags_first = diags_first.groupby('idp')['dat'].min().reset_index()
    
    drugs_first = drugs.loc[drugs.idp.isin(idps_all),:].groupby('idp')['start'].min().reset_index()
    
    dd_first = pd.merge(diags_first, drugs_first, how='outer')
    #dd_first.loc[dd_first.start.isna,]
    dd_first['diag_date'] = dd_first.apply(lambda x: min(pd.to_datetime(x['start']), pd.to_datetime(x['dat'])), axis=1)
    dd_first.loc[dd_first.diag_date.isna(),'diag_date'] = pd.to_datetime(dd_first.loc[dd_first.diag_date.isna(),'dat'])
    
    df_P0_gen = pd.merge(df_gen[['idp', 'dnaix', 'sexe', 'situacio', 'sortida']], dd_first[['idp', 'diag_date']],how='right')
    df_P0_gen['age_diag'] = df_P0_gen.apply(lambda x: (pd.to_datetime(x.diag_date) - pd.to_datetime(x.dnaix))/np.timedelta64(1,'Y'), axis=1)
    df_P0_gen = df_P0_gen.loc[df_P0_gen.age_diag>=18,]
    
    # GET STD DATA
    ages, tfds = [],[]
    for y in range(2013,2018):
        idps_y = events_list[y].keys()
        df_P0_gen[f'age_{y}'] = df_P0_gen.dnaix.apply(lambda x: (pd.to_datetime(f'{y}-01-01') - pd.to_datetime(x))/np.timedelta64(1,'Y'))
        df_P0_gen[f'tfd_{y}'] = df_P0_gen.diag_date.apply(lambda x: (pd.to_datetime(f'{y}-01-01') - pd.to_datetime(x))/np.timedelta64(1,'Y'))
        ages += list(df_P0_gen.loc[df_P0_gen.idp.isin(idps_y), f'age_{y}'])
        tfds += list(df_P0_gen.loc[df_P0_gen.idp.isin(idps_y), f'tfd_{y}'])
        
    age_cols = [f'age_{y}' for y in range(2013,2018)]
    df_P0_gen[age_cols] = (df_P0_gen[age_cols]-np.mean(ages))/np.std(ages)
    
    tfd_cols = [f'tfd_{y}' for y in range(2013,2018)]
    df_P0_gen[tfd_cols] = (df_P0_gen[tfd_cols]-np.mean(tfds))/np.std(tfds) 
    
    # GET D0 and M0
    d0_y, m0_y = {},{}
    for y in range(2013,2019):
        idps_y = list(events_list[min(y,2017)].keys())
        d0, m0 = get_d0_m0(idps_y, y, diags,drugs)
        d0_y[y] = {idp: np.squeeze(d0.loc[idp].values) for idp in d0.index}
        m0_y[y] = {idp: np.squeeze(m0.loc[idp].values) for idp in m0.index}
        
    # SAVE (and clean useless events)
    #df_P0_gen.to_csv(os.path.join(args.data_dir,'P0_general_std.pkl'))
    #with open(os.path.join(args.data_dir,'d0_m0.pkl'), 'wb') as outp:
    #    pickle.dump({'d0':d0_y, 'm0': m0_y}, outp)
    events_list2 = {}
    idps_all = set(df_P0_gen.idp)
    for y in range(2013,2018):
        events_list2[y] = dict([ (i,events_list[y][i]) for i in events_list[y].keys() if i in idps_all])   
    #with open(os.path.join(args.data_dir,'events_list_std.pkl'), 'wb') as outp:
    #    pickle.dump(events_list2, outp)