"""
Code to extract the codes for diagnosis, treatments and variables and get the cods_labs.json and cods_lists.json
Cods_lists.json file will be usaed as tokenizer for the different types. 
Cods_labs.json divide each code in the corresponding category. Change diags_labs and drugs_labs variable to define categories.
"""

import pandas as pd
import numpy as np
import os
import json
import argparse

def load_table(tbl_name, data_dr, sep='|', parser_dates=['dat']):
    dr = os.path.join(data_dr, tbl_name)
    tbl = pd.read_table(dr, sep=sep, parse_dates=parser_dates)
    return tbl


if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/enrico/Desktop/dm2_data/original_tables/')
    parser.add_argument('--table_var_cl', type=str, default='ALGDM2_entregable_variables_cliniques_20190620_081435.txt')
    parser.add_argument('--table_var_anal', type=str, default='ALGDM2_entregable_variables_analitiques_20190620_081435.txt')
    parser.add_argument('--table_drugs', type=str, default='ALGDM2_entregable_farmacs_prescrits_20190620_081435.txt')
    parser.add_argument('--table_diags', type=str, default='ALGDM2_entregable_diagnostics_20190620_081435.txt')
    parser.add_argument('--save_dir', type=str, default='/home/enrico/transformer_final/Data/')
    args = parser.parse_args()
    
    var_anal = load_table(args.table_var_anal, args.data_dir)
    var_cl = load_table(args.table_var_cl, args.data_dir)
    var_anal.drop(columns='val_txt', axis=1, inplace=True)
    df_vars = pd.concat([var_anal, var_cl])
    df_vars.drop(columns='agr', axis=1, inplace=True)
    vars_list = list(df_vars.cod.unique())
    
    df_pres = load_table(args.table_drugs, args.data_dir, parser_dates=['dat', 'dbaixa'] )
    df_pres.cod = df_pres.cod.apply(lambda x: x[0:5])
    drugs_list = list(np.sort(df_pres.cod.unique()))
    drugs_labs = {
         'insulin': [i for i, s in enumerate(drugs_list) if 'A10A' in s],
         'other': [i for i, s in enumerate(drugs_list) if 'A10B' in s]}
    
    diags = load_table(args.table_diags, args.data_dir, parser_dates=['dat', 'dbaixa'])
    diags_list = list(np.sort(diags.cod.unique()))
    diags_labs = {
      'HTN': [i for i, s in enumerate(diags_list) if 'I10' in s] + 
             [i for i, s in enumerate(diags_list) if 'I11' in s] + 
             [i for i, s in enumerate(diags_list) if 'I12' in s] + 
             [i for i, s in enumerate(diags_list) if 'I13' in s] + 
             [i for i, s in enumerate(diags_list) if 'I15' in s],
      'CVD': [i for i, s in enumerate(diags_list) if 'I20' in s] + 
             [i for i, s in enumerate(diags_list) if 'I21' in s] + 
             [i for i, s in enumerate(diags_list) if 'I22' in s] + 
             [i for i, s in enumerate(diags_list) if 'I23' in s] + 
             [i for i, s in enumerate(diags_list) if 'I24' in s] + 
             [i for i, s in enumerate(diags_list) if 'I25' in s] + 
        [diags_list.index('T82.2'), diags_list.index('E11.5'), diags_list.index('Z95.1'), diags_list.index('Z95.5') ],
      'NEURO': [diags_list.index('E11.4'), diags_list.index('E14.4')] + [i for i, s in enumerate(diags_list) if 'G' in s],
      'OPH': [diags_list.index('E11.3'), diags_list.index('E13.3'), diags_list.index('E14.3'), diags_list.index('H36.0')],
      'CKD': [diags_list.index('E11.2'), diags_list.index('N08.3'), diags_list.index('E13.2'), diags_list.index('E14.2')] }
    
    cods_list = {
        'diags': diags_list, 
        'drugs': drugs_list, 
        'vars': vars_list}
    with open(os.path.join(args.save_dir, 'cods_lists.json'), 'w') as f:
        json.dump(cods_list, f)
        
    cods_labs = {
        'diags_labs': diags_labs,
        'drugs_labs': drugs_labs}
    with open(os.path.join(args.save_dir, 'cods_labs.json'), 'w') as f:
        json.dump(cods_labs, f)
    
    
    
    