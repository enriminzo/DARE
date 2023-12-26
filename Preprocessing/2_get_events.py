"""
Reads the tables generated in step 1 and crearranged tehm as envets objects that can be used in the dataset.
"""

import sys
sys.path.append('/home/enrico/transformer_final/')

import pandas as pd
import numpy as np
from multiprocessing import Pool
from itertools import repeat
import pickle
from Model import Event
import os
import json


def create_events(row):
    ev = Event(cod=row.cod, val=row.val, t=row.t)
    return ev

def check_events(df_d, df_m, df_v):
    check_1 = (df_v.cod == 'HBA1C').any()
    check_2 = (len(df_d) + len(df_m) + len(df_v)) >=5
    return check_1 & check_2

def get_events_list(idp, year):
    
    df_m = drugs_events.loc[(drugs_events.idp==idp) & (drugs_events.t.between(f'{y}-01-01',f'{y}-12-31')),]
    df_d = diags_events.loc[(diags_events.idp==idp) & (diags_events.t.between(f'{y}-01-01',f'{y}-12-31')),]
    df_v = vars_events.loc[(vars_events.idp==idp) & (vars_events.t.between(f'{y}-01-01',f'{y}-12-31')),]
    if check_events(df_d, df_m, df_v):
        tmp_m, tmp_d = [],[]
        if not df_m.empty:
            tmp_m = list(df_m.apply(func=create_events, axis=1))
        if not df_d.empty:
            tmp_d = list(df_d.apply(func=create_events, axis=1))
        tmp_v = list(df_v.apply(func=create_events, axis=1))
        return (tmp_v + tmp_d + tmp_m, idp)
    else:
        return []

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/enrico/transformer_final/Data/')
    parser.add_argument('--table_vars', type=str, default='vars_all_events.csv')
    parser.add_argument('--table_drugs', type=str, default='drugs_events_15d.csv')
    parser.add_argument('--table_diags', type=str, default='diags_events.csv')
    parser.add_argument('--vars_std', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    
    # load data and normalize if needed
    diags = pd.read_csv(os.path.join(args.data_dir, args.table_diags), index_col=0)
    drugs = pd.read_csv(os.path.join(args.data_dir, args.table_drugs), index_col=0)
    variables = pd.read_csv(os.path.join(args.data_dir, args.table_vars))
    drugs_events = drugs.loc[drugs.start>='2013-01-01', ['idp', 'start', 'cod']]
    drugs_events.rename(columns={'start': 't'}, inplace=True)
    drugs_events['val'] = 1
    
    diags_events = diags.loc[diags.dat>='2013-01-01', ['idp', 'dat', 'cod']]
    diags_events.rename(columns={'dat': 't'}, inplace=True)
    diags_events['val'] = 1
    
    vars_events = variables[['idp', 'cod', 'dat', 'val']]
    vars_events = vars_events.rename(columns={'dat': 't'})
    if args.vars_std:
        vars_events['val'] = vars_events[['cod', 'val']].groupby('cod').transform(lambda x: (x - x.mean()) / (x.std()))
        out_file = os.path.join(args.data_dir, 'events_list_std.pkl')
    else:
        out_file = os.path.join(args.data_dir, 'events_list.pkl')
        
    # get events
    events_lists = {}
    for y in range(2013,2018):  
        print(f'Year: {y}')
        if y not in events_lists.keys():
            idps = vars_events.loc[vars_events.t.between(f'{y}-01-01',f'{y}-12-31'),'idp'].unique()
            with Pool(processes=12) as pool:
                events_y = pool.starmap(get_events_list, zip(idps, repeat(y)))
            dict_tmp=dict()
            for events,idp in [e for e in events_y if e]:
                dict_tmp.setdefault(idp, events)
            events_lists[y] = dict_tmp.copy()
            with open(out_file, 'wb') as outp:
                pickle.dump(events_lists, outp)

    