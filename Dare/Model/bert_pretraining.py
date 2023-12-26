import torch.nn as nn
import os
import json
from omegaconf import OmegaConf

DATA_PATH = '../DARE/Dare/Configs/data_paths.yaml'
paths = OmegaConf.load(DATA_PATH)

class predict_d0(nn.Module):
    def __init__(self, hidden, nd):
        """
        :param hidden: BERT model output size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, nd)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x[:, 1]))
    
class predict_m0(nn.Module):
    def __init__(self, hidden, nm):
        """
        :param hidden: BERT model output size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, nm)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x[:, 2]))
    
class predict_ei(nn.Module):
    def __init__(self, hidden, nv):
        """
        :param hidden: BERT model output size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, nv)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear(x[:, 3:])
    
    
    
class BERT_PT(nn.Module):
    """
    BERT Language Model
    Initial Status Prediction + Masked Language Model
    """

    def __init__(self, bert, n_diags=None, n_drugs=None, n_vars=None):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        
        if n_diags is None or n_drugs is None:
            with open(paths.cods_labs_path, 'r') as f:
                cods_labs = json.load(f)
            if n_diags is None:
                n_diags = len(cods_labs['diags_labs'])
            if n_drugs is None:
                n_drugs = len(cods_labs['drugs_labs'])
        if n_vars is None:
            with open(paths.cods_lists_path, 'r') as f:
                cods_lists = json.load(f)
            n_vars = len(cods_lists['vars'])
            
        self.bert = bert
        self.predit_d0 = predict_d0(self.bert.hidden, n_diags)
        self.predit_m0 = predict_m0(self.bert.hidden, n_drugs)
        self.predit_ei = predict_ei(self.bert.hidden, n_vars)

    def forward(self, p0, d0, m0, d_all, m_all, v_all, t_all, types_all, dists):
        x = self.bert(p0, d0, m0, d_all, m_all, v_all, t_all , types_all, dists)
        return self.predit_ei(x), self.predit_d0(x), self.predit_m0(x)