import torch
import torch.nn as nn
import numpy as np
import os
from Dare.Model.bert_pretraining import BERT_PT
from Dare.Model.bert import BERT



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
    
    
# -----------------------------------------------------------------------------------------------------------------   

class EarlyStopping():
    ''' 
        Early stopping class for roc auc 
    '''
    def __init__(self, tolerance=5, min_delta=1e-5):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False
        self.max_auc = 0

    def __call__(self, test_auc):
        
        if (test_auc-self.max_auc) > self.min_delta: #model is imporving performances
            self.counter = 0
            self.max_auc = test_auc
        else:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True    

# -----------------------------------------------------------------------------------------------------------------

def import_model(config, epoch, model_path=None, model_name=None, device=None, n_diags=None, n_drugs=None, n_vars=None):
    '''
        Load model from pretrained one, return only the model without the pretraing layers
    '''
    if model_name is None:
        if config.Model.relative:
            model_name = f'BERT_PT_R_attnheads{config.Model.attn_heads}_hiddens{config.Model.hidden}_layers{config.Model.n_layers}'
        else:
            model_name = f'BERT_PT_attnheads{config.Model.attn_heads}_hiddens{config.Model.hidden}_layers{config.Model.n_layers}'
    if model_path is None:
        model_path = '/home/enrico/transformer_final/Results/Models_pretrained/'
    
    model_path = os.path.join(model_path, model_name+".ep%d" % epoch)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'file "{model_path}" not found, please provide a valid path')
        
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    model = BERT_PT(BERT(config), n_diags, n_drugs, n_vars)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f'successfully loaded model {model_path}')
    return model.bert

# -----------------------------------------------------------------------------------------------------------------

class rnn_liner_FT(torch.nn.Module):
    '''
        Recurrent layer followed by linear layer for finetunig 
    '''
    
    def __init__(self, input_size, n_grus, layers_size, out_size=20):
        super().__init__()
        self.rnn = nn.GRU(input_size, layers_size[0], num_layers=n_grus, batch_first=True, bidirectional=True) 
        
        self.linears = nn.ModuleList(
            [nn.Linear(layers_size[i-1],layers_size[i]) if i!=1 else nn.Linear(2*layers_size[i-1],layers_size[i]) for i in range(1,len(layers_size))])
        self.act = nn.ReLU()
        self.out = nn.Linear(layers_size[-1], out_size)
        self.act_out = nn.Sigmoid()
        
        
    def forward(self, x):
        
        self.rnn.flatten_parameters()
        x, _ = self.rnn(x)
        x = x[:,-1,:]
        for lay in self.linears:
            x = self.act(lay(x))
        return self.act_out(self.out(x))
    
    
