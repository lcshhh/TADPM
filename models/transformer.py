import torch
from torch import nn
import torch
from torch import nn
from torch.nn.functional import softmax
from torch.nn.init import kaiming_normal_
import numpy as np
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dim):
        super(ScaledDotProductAttention, self).__init__()

        self.scale_factor = np.power(dim, -0.5)

    def forward(self, value, key, query, mask=None):
        # (batch, q_len, seq_len)
        adjacency = query.bmm(key.transpose(1, 2)) * self.scale_factor

        if mask is not None:
            adjacency.data.masked_fill_(mask.data, -float('inf'))

        attention = softmax(adjacency, 2)
        return attention.bmm(value), attention


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, dim, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.n_heads = n_heads
        self.dim_m = dim
        self.dim_q_k = dim
        self.dim_v = dim

        self.query_projection = nn.Parameter(
            torch.FloatTensor(n_heads, dim, dim))
        self.key_projection = nn.Parameter(
            torch.FloatTensor(n_heads, dim, dim))
        self.value_projection = nn.Parameter(
            torch.FloatTensor(n_heads, dim, dim))
        self.attention = ScaledDotProductAttention(dim)
        self.output = nn.Linear(dim * n_heads, dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_normalization = nn.LayerNorm(dim, eps=1e-12)

        # Initialize projection tensors
        for parameter in [
                self.query_projection, self.key_projection,
                self.value_projection
        ]:
            kaiming_normal_(parameter.data)
    
    def stack_heads(self, tensor):
        return tensor.view(-1, self.dim_m).repeat(self.n_heads, 1, 1)

    def forward(self, value, key, query, mask=None, return_attn_weight=False):
        seq_len = key.shape[1]
        q_len = query.shape[1]
        batch_size = query.shape[0]

        residual = query
        # (batch, x, dim_m) -> (n_heads, batch * x, dim_m)
        value, key, query = map(self.stack_heads, [value, key, query])

        if mask is not None:
            mask = self.stack_mask(mask)

        # (n_heads, batch * x, dim_m) -> (n_heads, batch * x, projection) -> (n_heads * batch, x, projection)
        # where `projection` is `dim_q_k`, `dim_v` for each input respectively.
        value = value.bmm(self.value_projection).view(-1, seq_len, self.dim_v)
        key = key.bmm(self.key_projection).view(-1, seq_len, self.dim_q_k)
        query = query.bmm(self.query_projection).view(-1, q_len, self.dim_q_k)

        # (n_heads * batch, q_len, dim_v)
        context, attn_weight = self.attention(value, key, query, mask)

        # # (n_heads * batch, q_len, dim_v) -> (batch * q_len, n_heads, dim_v) -> (batch, q_len, n_heads * dim_v)
        # context = context.view(self.n_heads, -1, self.dim_v).transpose(0, 1).view(-1, q_len, self.n_heads * self.dim_v)

        # (n_heads * batch, q_len, dim_v) -> (batch, q_len, n_heads * dim_v)
        context_heads = context.split(batch_size, dim=0)
        concat_heads = torch.cat(context_heads, dim=-1)

        # (batch, q_len, n_heads * dim_v) -> (batch, q_len, dim_m)
        out = self.output(concat_heads)
        out = self.dropout(out)
        if(return_attn_weight):
            return self.layer_normalization(out + residual),attn_weight
        else:
            return self.layer_normalization(out + residual)
        
class PositionWise(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super(PositionWise, self).__init__()

        self.feedforward = nn.Sequential(
            nn.Linear(dim,dim), nn.ReLU(), nn.Linear(dim,dim),
            nn.Dropout(dropout))
        self.normalization = nn.LayerNorm(dim, eps=1e-12)

    def forward(self, input):
        # There's nothing difficult here.
        residual = input
        output = self.feedforward(input)
        output = self.normalization(output + residual)
        return output

class TransformerEncoderLayer(nn.Module):
    def __init__(self, n_heads, dim,dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(n_heads, dim,
                                            dropout)
        self.positionwise = PositionWise(dim, dropout)

    def forward(self, input, mask=None):
        enc_att = self.attention(input, input, input, mask=mask,return_attn_weight=False)
        output = self.positionwise(enc_att)
        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self, n_heads, dim, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()

        self.masked_attention = MultiHeadAttention(n_heads, dim, dropout)
        self.attention = MultiHeadAttention(n_heads, dim,dropout)
        self.positionwise = PositionWise(dim, dropout)

    def forward(self, input, encoder_output, mask=None):
        dec_att = self.masked_attention(input, input, input, mask)
        adj_att = self.attention(
            value=encoder_output, key=encoder_output, query=dec_att,mask=None)
        output = self.positionwise(adj_att)

        return output