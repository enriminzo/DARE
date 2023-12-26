import torch
import torch.nn as nn
from torch import cat, tanh
import torch.nn.functional as F
import numpy as np
import math

class RelativePositionalEmbedding(nn.Module):

    def __init__(self, max_time_embedding, hidden_size, learn, padd_indx):
        super(RelativePositionalEmbedding, self).__init__()
        if learn:
            self.r_embedding = nn.Embedding(max_time_embedding, hidden_size, padding_idx=padd_indx)
        else: 
            self.r_embedding = PositionalEmbedding(hidden_size, max_time_embedding, padd_indx=padd_indx)
            
    def forward(self, dist_mat):
        return self.r_embedding(dist_mat)
    

class RelativeAttention(nn.Module):
    """
        Compute Scaled Dot Product  Relative Attention
    """

    def forward(self, query, key, value, R, mask, dropout):
        
        if R is not None:
            QK = torch.matmul(query, key.transpose(-2, -1))
            QR = torch.matmul(query.unsqueeze(-2),R.transpose(-2,-1)).squeeze()
            scores = (QK + QR) / math.sqrt(query.size(-1))
        else:
            QK = torch.matmul(query, key.transpose(-2, -1))
            scores = (QK) / math.sqrt(query.size(-1))
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        
        if mask is not None:
            p_attn = p_attn.masked_fill(mask == 0, 0)
            
        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn
    
class MultiHeadedRelativeAttention(nn.Module):
    """
        Implement MultiHead Relative Attention layer
        INPUTS:
            - h: number of heads
            - d_model: hidden size dimension (of input and output). Note that must be d_model%h == 0.
            - relative: boolean, use relative attention?
            - dropout: dropout probability (calculated in each head)
            - max_time_embedding: max distance between two events
            - learn: boolean, should time embedding be learnable?
            - padd_indx_time: padding index for type vectors
    """

    def __init__(self, h, d_model, relative, dropout, max_time_embedding, learn, padd_indx_time):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v=d_k=d_q
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = RelativeAttention()

        self.dropout = nn.Dropout(p=dropout)
        self.relative = relative
        if relative:
            self.posEmbedding = RelativePositionalEmbedding(max_time_embedding, d_model, learn=learn, padd_indx=padd_indx_time)
        
    def forward(self, query, key, value, dists, mask):   
        """
            Forward function for multihead attention. 
            Query key and values are tensors with shape= (batch_size, sequence length, d_model)
            Dist is a tensor of distance matrices with shape=(batch_size, sequence length, sequence length)
            Mask is applied to scores, is a tensor with shape=(batch_size, sequence length, sequence length)
        
        """
        batch_size = query.size(0)
        seq_len = query.size(1)
        
        # Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]
        if self.relative:
            R = self.posEmbedding(dists)
            R = R.view(batch_size, seq_len, seq_len, self.h, self.d_k).transpose(1,3).transpose(2,3)
        else:
            R = None
        # Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, R, mask=mask, dropout=self.dropout)

        # "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)