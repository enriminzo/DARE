import torch
import torch.nn as nn
from torch import cat, tanh
import math

class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len, padd_indx):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        if padd_indx:
            pe[padd_indx,:] = 0

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, x].squeeze()
    
    
class InputEmbedding(nn.Module):
    
    """
        Construct the embeddings from P0 vector and events list. 
        Require config class with:
            - P0_size: size of P0 vector
            - nd: size of diagnosis dictionary
            - nm: size of medications dicitonary
            - nv: size of variables dictionary
            - type size: number of possible events types
            - hidden_size: dimension of the embedding space
            - max_time_embedding: max distance between an event and t0
            - time_emb: type of tyme embedding could be:
                    sin: sinusoidal as in original transformer
                    learn: learnable 
                    something else: no time embedding is used
            - hidden_dropout_prob: dropout probability 
            - padd_indx_type: padding index for type vectors
            - padd_indx_time: padding index for time vectors
    """

    def __init__(self, config):
        super(InputEmbedding, self).__init__()
        self.P0_embeddings = nn.Linear(config.P0_size, config.hidden_size)
        self.d_embeddings = nn.Linear(config.nd, config.hidden_size, bias=False)
        self.m_embeddings = nn.Linear(config.nm, config.hidden_size, bias=False)
        self.v_embeddings = nn.Linear(config.nv, config.hidden_size, bias=False)
        self.type_embeddings = nn.Embedding(config.type_size, config.hidden_size, padding_idx=config.padd_indx_type)
        if config.time_emb == 'sin':
            self.time_embeddings = PositionalEmbedding(config.hidden_size, config.max_time_embedding, config.padd_indx_time)
            self._use_time_emb = True
        elif config.time_emb == 'learn':
            self.time_embeddings = nn.Embedding(config.max_time_embedding, config.hidden_size, padding_idx=config.padd_indx_time)
            self._use_time_emb = True
        else:
            self.time_embeddings = None
            self._use_time_emb = False
        self.LayerNorm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, P0, d0, m0, d, m, var,  times, types):
        
        # event embedding at t0
        P0_embed = tanh(self.P0_embeddings(P0))
        d0_embed = self.d_embeddings(d0)
        m0_embed = self.m_embeddings(m0)
        
        # events embedding
        d_embed = self.d_embeddings(d)
        m_embed = self.m_embeddings(m)
        var_embed = tanh(self.v_embeddings(var))
        event_embed = torch.sum(torch.stack([m_embed, d_embed, var_embed]), dim=0)
        event_embed = torch.cat((P0_embed[:,None,:], d0_embed[:,None,:], m0_embed[:,None,:],event_embed), 1)
        
        # type embedding
        type_embed = self.type_embeddings(types)
        embeddings = event_embed + type_embed
        
        # time embedding 
        if self._use_time_emb:
            time_embed = self.time_embeddings(times)
            embeddings = time_embed + embeddings
        
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings
        
    