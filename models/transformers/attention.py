import numpy as np
import torch
from torch import nn
from torch.nn.functional import softmax
from torch.nn.init import kaiming_normal_


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dim_q_k):
        """Scaled Dot-Product Attention model: :math:`softmax(QK^T/sqrt(dim))V`.

        Args:
            dim_q_k (int): dimension of `queries` and `keys`.

        Inputs: query, key, value, mask
            - **value** of shape `(batch, seq_len, dim_v)`:  a float tensor containing `value`.
            - **key** of shape `(batch, seq_len, dim_q_k)`: a float tensor containing `key`.
            - **query** of shape `(batch, q_len, dim_q_k)`: a float tensor containing `query`.
            - **mask** of shape `(batch, q_len, seq_len)`, default None: a byte tensor containing mask for
              illegal connections between query and value.

        Outputs: attention, attention_weights
            - **attention** of shape `(batch, q_len, dim_v)` a float tensor containing attention
              along `query` and `value` with the corresponding `key`.
            - **attention_weights** of shape `(batch, q_len, seq_len)`: a float tensor containing distribution of
              attention weights.
        """
        super(ScaledDotProductAttention, self).__init__()

        self.scale_factor = np.power(dim_q_k, -0.5)

    def forward(self, value, key, query, mask=None):
        # (batch, q_len, seq_len)
        adjacency = query.bmm(key.transpose(1, 2)) * self.scale_factor

        if mask is not None:
            adjacency.data.masked_fill_(mask.data, -float('inf'))

        attention = softmax(adjacency, 2)
        return attention.bmm(value), attention


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, dim_m, dim_q_k, dim_v, dropout=0.1):
        """Multi-Head Attention model.

        Args:
            n_heads (int): number of heads.
            dim_m (int): hidden size of model.
            dim_q_k (int): dimension of projection `queries` and `keys`.
            dim_v (int): dimension of projection `values`.
            dropout (float, optional): dropout probability.

        Inputs:
            - **value** of shape `(batch, seq_len, dim_m)`: a float tensor containing `value`.
            - **key** of shape `(batch, seq_len, dim_m)`: a float tensor containing `key`.
            - **query** of shape `(batch, q_len, dim_m)`: a float tensor containing `query`.
            - **mask** of shape `(batch, q_len, seq_len)`: default None: a byte tensor containing mask for
              illegal connections between query and value.

        Outputs:
            - **attention** of shape `(batch, q_len, dim_m)`: a float tensor containing attention
              along `query` and `value` with the corresponding `key` using Multi-Head Attention mechanism.
        """
        super(MultiHeadAttention, self).__init__()

        self.n_heads = n_heads
        self.dim_m = dim_m
        self.dim_q_k = dim_q_k
        self.dim_v = dim_v

        self.query_projection = nn.Parameter(
            torch.FloatTensor(n_heads, dim_m, dim_q_k))
        self.key_projection = nn.Parameter(
            torch.FloatTensor(n_heads, dim_m, dim_q_k))
        self.value_projection = nn.Parameter(
            torch.FloatTensor(n_heads, dim_m, dim_v))
        self.attention = ScaledDotProductAttention(dim_q_k)
        self.output = nn.Linear(dim_v * n_heads, dim_m)
        self.dropout = nn.Dropout(dropout)
        self.layer_normalization = nn.LayerNorm(dim_m, eps=1e-12)

        # Initialize projection tensors
        for parameter in [
                self.query_projection, self.key_projection,
                self.value_projection
        ]:
            kaiming_normal_(parameter.data)

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

    def stack_mask(self, mask):
        """Prepare mask tensor for multi-head Scaled Dot-Product Attention.

        Args:
            mask: A byte tensor of shape `(batch, q_len, seq_len)`.

        Returns:
            A byte tensor of shape `(n_heads * batch, q_len, seq_len)`.
        """
        return mask.repeat(self.n_heads, 1, 1)

    def stack_heads(self, tensor):
        """Prepare tensor for multi-head projection.

        Args:
            tensor: A float input tensor of shape `(batch, x, dim_m)`.

        Returns:
            Stacked input tensor n_head times of shape `(n_heads, batch * x, dim_m)`.
        """
        return tensor.view(-1, self.dim_m).repeat(self.n_heads, 1, 1)
