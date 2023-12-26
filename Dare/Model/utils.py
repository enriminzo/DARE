import torch
import torch.nn as nn
import math
import warnings
from omegaconf import OmegaConf


class ConfigEmbedding():
    def __init__(self, P0_size, nd, nm, nv, nmax,
                 hidden_size, hidden_dropout_prob,
                 type_size, padd_indx_type,
                 time_emb, max_time_embedding, padd_indx_time):
        self.P0_size = P0_size
        self.nd = nd
        self.nm = nm
        self.nv = nv
        self.nmax = nmax
        self.hidden_size = hidden_size
        self.type_size = type_size
        self.max_time_embedding = max_time_embedding
        self.hidden_dropout_prob = hidden_dropout_prob
        self.time_emb = time_emb
        self.padd_indx_time = padd_indx_time
        self.padd_indx_type = padd_indx_type
        
class ConfigModel():
    def __init__(self, attn_heads, hidden, n_layers, feed_forward_hidden, dropout_trans,
                 max_time_embedding, learn, padd_indx_time,
                 relative, dropout_attention):
        self.attn_heads = attn_heads
        self.hidden = hidden
        self.n_layers = n_layers
        self.feed_forward_hidden = feed_forward_hidden
        self.dropout_trans = dropout_trans
        self.max_time_embedding = max_time_embedding
        self.learn = learn
        self.padd_indx_time = padd_indx_time
        self.relative = relative
        self.dropout_attention = dropout_attention
        
class ModelConfigs:
    def __init__(self, new_configs=None, configs_path=None):
        
        if new_configs is None and configs_path is None:
            raise ValueError("Pleas provide a configs dict or a path to a config file.")
        
        if configs_path is not None:
            default_configs = OmegaConf.load(configs_path)
        else:
            default_configs = dict(n_layers=5, P0_size=3, nd=181, nm=13, nv=20, nmax=40,
                                   hidden_size=100, dropout_emb=0.1, type_size=7, padd_indx_type=6, feed_forward_hidden=1440,
                                   time_emb='sin', max_time_embedding=367, padd_indx_time=366,
                                   attn_heads=5, dropout_trans=0.1,learn=True, 
                                   relative=True, dropout_attention=0.1, masks=None)
        config = default_configs.copy()
        
        if new_configs is not None:
            unexpected_keys = set(new_configs.keys()) - set(default_configs.keys())
            if unexpected_keys:
                warnings.warn(f"Unexpected keys: {unexpected_keys}")   
            config.update(new_configs)
                    
                    
        self.Embedding = ConfigEmbedding(config.P0_size, config.nd, config.nm, config.nv, config.nmax, config.hidden_size,
                                             config.dropout_emb, config.type_size, config.padd_indx_type, 
                                             config.time_emb, config.max_time_embedding, config.padd_indx_time)
        self.Model = ConfigModel(config.attn_heads, config.hidden_size, config.n_layers, config.feed_forward_hidden, config.dropout_trans,
                                 config.max_time_embedding, config.learn, config.padd_indx_time, config.relative, config.dropout_attention)

        if config.masks==None:
            self.masks = {'diags': 0.5, 'drugs':0.5, 'age': -5, 'tfd':-1, 'vars':0}
        else:
            self.masks = config.masks

## ------------------------------------------------------------------------------------

class GetDistaceMatrices(nn.Module):
    def __init__(self, padd_indx):
        super().__init__()
        self.padd_indx = padd_indx
        
    def forward(self, times ):
        batch_size, seq_len = times.shape
        dists = torch.empty([batch_size,seq_len,seq_len], dtype=torch.long)
        for i in range(seq_len):
            dists[:,i,:] = torch.abs(times - times[:,i].unsqueeze(1).repeat(1,seq_len))
        dists = dists.masked_fill((times.unsqueeze(1).repeat(1,seq_len,1)==self.padd_indx),self.padd_indx)
        dists = dists.masked_fill((times.unsqueeze(2).repeat(1,1,seq_len)==self.padd_indx),self.padd_indx)
        return dists
    
## ----------------------------------------------------------------------------------------
class GELU(nn.Module):
    
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


## ------------------------------------------------------------------------------------
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))
    
    
## ------------------------------------------------------------------------------------   
class LayerNorm(nn.Module):
    "Construct a layernorm module"

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    
    
## ------------------------------------------------------------------------------------    
class SublayerConnection(nn.Module):
    

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))