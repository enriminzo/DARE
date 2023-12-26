"""
Preprocessing original data from SIDIAP db to get events tables. Depending on your data you may need to adapt this code. Generated_tables are:
- diags_events.csv: table with columns: idp | cod (icd-10) | dat (diagnosis date) | dbaixa (termination date) 
- drugs_events_{freq}d.csv: table with columns: idp | cod | start | end . freq depends on the frequency used to merge different prescriptions in the same interval. We used 15 days 
- vars_all_events.csv: table with columns idp | cod | dat | val. Outliers values to be filtered out can be defined in the outliers dictionary 
"""

import pandas as pd
import numpy as np
import os
import argparse
from datetime import datetime, timedelta, date
import time

def load_table(tbl_name, data_dr, sep='|', parser_dates=['dat']):
    dr = os.path.join(data_dr, tbl_name)
    tbl = pd.read_table(dr, sep=sep, parse_dates=parser_dates)
    return tbl

# Functions to clean drugs
def get_intervals(spans, ci=15):
    # Gets as input a list of spans, represented by touple (start, end) and a "confidence interval" in which two prescriptions can be merged
    # Returns the same list merging spans so that there are not overlaped spans.
    # Spans closer then ci days are considered overlapped. 
    
    to_return = []
    ci = timedelta(days=ci)
    for s in spans:
        
        # if alredy exists an interval contained in s, drop it
        check_time_middle = [i for i,t in enumerate(to_return) if s[0]<=t[0] and s[1]>=t[1]]
        if check_time_middle:
            to_return = [t for i,t in enumerate(to_return) if i not in check_time_middle]
        
        check_time_start = [i for i,t in enumerate(to_return) if (s[0]>=(t[0]-ci) and s[0]<=(t[1]+ci))]
        check_time_end = [i for i,t in enumerate(to_return) if (s[1]>=(t[0]-ci) and s[1]<=(t[1]+ci))] 
        
        # start of s contained in an interval..
        if check_time_start:
            index_start = check_time_start[0]
            new_start = min(s[0], to_return[index_start][0])
            # ..and its end is in another one
            if check_time_end:
                index_end = check_time_end[0]
                #if index_end!= index_start:
                new_end = max(s[1], to_return[index_end][1])
                to_return[index_start] = (new_start, new_end)
                if index_end!= index_start: to_return.remove(to_return[index_end])
                    
            # .. and its end is not in another one
            else:
                new_end = max(s[1], to_return[index_start][1])
                to_return[index_start] = (new_start, new_end)
                
        # end of s contained in an interval (but not its start)
        elif check_time_end:
            index_end = check_time_end[0]
            new_start = min(s[0], to_return[index_end][0])
            new_end = max(s[1], to_return[index_end][1])
            to_return[index_end] = (new_start, new_end)
            
        else:
            to_return.append(s)
            
    return to_return

def get_df_f_p(df_f_p):
    # Gets as input a df of prescriptions for a single patient p.
    # Returns a df with no overlapping prescriptions, according to the definiton of overlapping in "get_intervals"
    
    spans_orig =  [(x['dat'],x['dbaixa']) for i,x in df_f_p.iterrows()]
    spans = get_intervals(spans_orig)
    starts,ends = zip(*spans)
    to_ret = pd.DataFrame( data = {'start': starts, 'end': ends})   
   
    return to_ret

# Function to clean variables
outliers = {
    'HBA1C': [3.5, 20],
    'TT103': [12, 1000],
    'EK201': [60,240],
    'EK202': [30,130],
    'COLHDL': [20, 1000]
}

def remove_outliers(row):
    
    v = row.cod
    if v in outliers.keys():
        if row.val <= outliers[v][0] or row.val>=outliers[v][1]:
            return False
    return True

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/enrico/Desktop/dm2_data/original_tables/')
    parser.add_argument('--table_var_cl', type=str, default='ALGDM2_entregable_variables_cliniques_20190620_081435.txt')
    parser.add_argument('--table_var_anal', type=str, default='ALGDM2_entregable_variables_analitiques_20190620_081435.txt')
    parser.add_argument('--table_drugs', type=str, default='ALGDM2_entregable_farmacs_prescrits_20190620_081435.txt')
    parser.add_argument('--table_diags', type=str, default='ALGDM2_entregable_diagnostics_20190620_081435.txt')
    parser.add_argument('--save_dir', type=str, default='/home/enrico/transformer_final/Data/')
    args = parser.parse_args()
    
    # DIAGS
    print('Get diags events...')
    diags = load_table(args.table_diags, args.data_dir, parser_dates=['dat', 'dbaixa'])
    diags.drop(columns='agr', axis=1, inplace=True)
    diags = diags.reset_index()
    diags.to_csv(os.path.join(args.save_dir, 'diags_events.csv'))
    
    # DRUGS
    print('Get drugs events...')
    df_pres = load_table(args.table_drugs, args.data_dir, parser_dates=['dat', 'dbaixa'] )
    df_pres.cod = df_pres.cod.apply(lambda x: x[0:5])
    
    analyzed_drugs = []
    res_tot = []
    for f in  df_pres.cod.unique(): #drugs.keys():
        tic = time.time()
        print('   Working on drugs: {}'.format(f))
        analyzed_drugs.append(f)
        df_f = df_pres.loc[df_pres.cod==f,]
        res = df_f.groupby('idp').apply(get_df_f_p).reset_index()
        res.drop('level_1', inplace=True, axis=1)
        res['cod'] = f
        res_tot.append(res)
        toc=time.time()
        tot_time = (toc-tic)/60
        print(f'   End, {tot_time:.2f} min needed')
        
    res = pd.concat(res_tot)
    res = res.reset_index(drop=True)
    res.to_csv(os.path.join(args.save_dir,'drugs_events_15d.csv'))
    
    # VARS
    print('Get vars')
    var_anal = load_table(args.table_var_anal, args.data_dir)
    var_cl = load_table(args.table_var_cl, args.data_dir)
    var_anal.drop(columns='val_txt', axis=1, inplace=True)
    df_vars = pd.concat([var_anal, var_cl])
    df_vars.drop(columns='agr', axis=1, inplace=True)
    to_rem = df_vars.apply(remove_outliers, axis=1)
    df_vars = df_vars.loc[to_rem,]
    del_vars = df_vars.cod.value_counts()
    del_vars = del_vars.loc[del_vars<50000].index
    df_vars = df_vars.loc[~(df_vars.cod.isin(del_vars)),]
    
    df_vars_filt = []
    filt = False  # If filt: each year keep only data of patients with 10+ measures and 2+ HbA1c measures -> filterlater in the process !!
    for y in range(2013,2018):
        df_vars_tmp = df_vars.loc[df_vars.dat.between(f'{y}-01-01',f'{y}-12-31'),]
        if filt:
            count_idps = df_vars_tmp.idp.value_counts()
            df_tmp = df_vars_tmp.loc[df_vars_tmp.idp.isin(count_idps.index[count_idps>=10]),].copy()
            idps_hba1c = df_tmp.idp.loc[df_tmp.cod=='HBA1C'].value_counts()
            idps_hba1c = idps_hba1c.loc[idps_hba1c>=2].index
            df_tmp = df_tmp.loc[df_tmp.idp.isin(idps_hba1c),]
        else:
            df_tmp = df_vars_tmp.copy()
        df_vars_filt.append(df_tmp)
        
    df_vars_filt = pd.concat(df_vars_filt)
    df_vars_filt.reset_index(drop=True, inplace=True)
    df_vars_filt.to_csv(os.path.join(args.save_dir, 'vars_all_events_dm2.csv'))
    