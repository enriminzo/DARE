import torch.nn as nn
from Dare.Model.attention import MultiHeadedRelativeAttention
from Dare.Model.utils import PositionwiseFeedForward, SublayerConnection


class TransformerBlock(nn.Module):
    """
        Transformer block implementation with relative distance 
        INPUT (config class):
        - attn_heads: number of heads
        - hidden: hidden size dimension (of input and output). Note that must be d_model%h == 0.
        - relative: boolean, use relative attention?
        - dropout_attention: dropout probability calculated in each head 
        - dropout_trans: dropout probability calculated by other transformer layers
        - max_time_embedding: max distance between two events
        - learn: boolean, should time embedding be learnable?
        - padd_indx_time: padding index for type vectors
    """

    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadedRelativeAttention(h=config.attn_heads, d_model=config.hidden, relative=config.relative, dropout=config.dropout_attention, max_time_embedding=config.max_time_embedding, learn=config.learn, padd_indx_time=config.padd_indx_time)
        self.feed_forward = PositionwiseFeedForward(d_model=config.hidden, d_ff=config.feed_forward_hidden, dropout=config.dropout_trans)
        self.input_sublayer = SublayerConnection(size=config.hidden, dropout=config.dropout_trans)
        self.output_sublayer = SublayerConnection(size=config.hidden, dropout=config.dropout_trans)
        self.dropout = nn.Dropout(p=config.dropout_trans)

    def forward(self, x, dists,mask):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, dists=dists, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)