import torch.nn as nn
from Dare.Model.embedding import InputEmbedding
from Dare.Model.transformer import TransformerBlock
from torch import matmul

class BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers with relative attention implemented
    """

    def __init__(self, config):
        """
        - congig: configuration for the bert model, must include
            - config.Transformer: configurations for the transformer blocks (see TransfomerBlock)
            - config.Embeddig: configurations for the embedding layer (see InputEmbedding)
            - config.n_layers: number of transformer blocks in serie
        """

        super().__init__()
        self.hidden = config.Model.hidden
        self.n_layers = config.Model.n_layers
        self.attn_heads = config.Model.attn_heads
        self.padd_indx_time = config.Model.padd_indx_time
        self.relative = config.Model.relative

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = InputEmbedding(config.Embedding)
        
        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(config.Model) for _ in range(config.Model.n_layers)])
        
    

    def forward(self, p0, d0, m0, d_all, m_all, v_all, t_all , types_all, dists=None):
        
        
        # attention mask for padded token (not for masked ones!)
        if self.relative:
            mask = (dists != self.padd_indx_time).unsqueeze(1).repeat(1,self.attn_heads, 1, 1)
        else:
            mask_tmp = (t_all!= self.padd_indx_time ).float() #long()
            mask =  matmul(mask_tmp.unsqueeze(2), mask_tmp.unsqueeze(1)).unsqueeze(1).repeat(1,self.attn_heads, 1, 1)
            
        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(p0, d0, m0, d_all, m_all, v_all, t_all , types_all)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, dists, mask)

        return x



    
