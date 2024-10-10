import torch
from torch import nn

from models.transformers import (MultiHeadAttention, PositionalEmbedding,
                             PositionWise)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        """Transformer encoder layer.

        Args:
            dim_m (int): Dimension of model.
            dim_q_k (int): Dimension of `query` & `key` attention projections.
            dim_v (int): Dimension of `value` attention projection.
            n_heads (int): Number of attention heads.
            dim_i (int): Inner dimension of feed-forward position-wise sublayer.
            dropout (float): Dropout probability.

        Inputs:
            - **input** of shape `(batch, enc_seq_len, dim_m)`, a float tensor, where `batch` is batch size,
              `enc_seq_len` is length of encoder sequence for this batch and `dim_m` is hidden size of model.
              Input embedding has `dim_m` size too.

        Outputs:
            - **output** of shape `(batch, seq_len, dim_m)`, a float tensor.
        """
        super(TransformerEncoderLayer, self).__init__()
        n_heads = nhead
        dim = d_model
        self.attention = MultiHeadAttention(n_heads, dim, dim, dim,
                                            dropout)
        self.positionwise = PositionWise(dim, dim, dropout)

    def forward(self, input, mask=None,return_attn_weight=False):
        if(return_attn_weight):
            enc_att,attn_weight=self.attention(input, input, input, mask=mask,return_attn_weight=True)
            output = self.positionwise(enc_att)
            return output,attn_weight
        else:
            enc_att = self.attention(input, input, input, mask=mask,return_attn_weight=False)
            output = self.positionwise(enc_att)
            return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        """Transformer decoder layer.

        Args:
            dim_m (int): Dimension of model.
            dim_q_k (int): Dimension of `query` & `key` attention projections.
            dim_v (int): Dimension of `value` attention projection.
            n_heads (int): Number of attention heads.
            dim_i (int): Inner dimension of feed-forward position-wise sublayer.
            dropout (float): Dropout probability.

        Inputs:
            - **input** of shape `(batch, dec_seq_len, dim_m)`, a float tensor, where `batch` is batch size,
              `dec_seq_len` is length of decoder sequence for this batch and `dim_m` is hidden size of model.
              Input embedding has `dim_m` size too.
            - **encoder_output** of shape `(batch, enc_seq_len, dim_m)`, a float tensor, where `enc_seq_len` is length
              of encoder sequence.
            - **mask** of shape `(batch, dec_seq_len, dec_sec_len)`, a byte tensor containing mask for
              illegal connections between encoder and decoder sequence tokens. It's used to preserving
              the auto-regressive property.

        Outputs:
            - **output** of shape `(batch, dec_seq_len, dim_m)`, a float tensor.
        """
        super(TransformerDecoderLayer, self).__init__()
        n_heads = nhead
        dim = d_model
        self.masked_attention = MultiHeadAttention(n_heads, dim, dim,
                                                   dim, dropout)
        self.attention = MultiHeadAttention(n_heads, dim, dim, dim,
                                            dropout)
        self.positionwise = PositionWise(dim, dim, dropout)

    def forward(self, input, encoder_output, mask=None, extra_mask=None):
        dec_att = self.masked_attention(input, input, input, mask)
        adj_att = self.attention(
            value=encoder_output, key=encoder_output, query=dec_att,mask=extra_mask)
        output = self.positionwise(adj_att)

        return output


class Transformer(nn.Module):
    def __init__(self,
                 max_seq_len,
                 vocab_size,
                 emb_size=250,
                 embeddings=None,
                 n_layers=6,
                 dim_m=512,
                 dim_q_k=64,
                 dim_v=64,
                 n_heads=8,
                 dim_i=2048,
                 dropout=0.1):
        """Transformer model from 'Attention Is All You Need' paper.

        Args:
            max_seq_len (int): Maximum sequence length.
            vocab_size (int): Vocabulary size.
            emb_size (int, optional): Embedding size. You do not need to specify a value if you are using
              embedding weights.
            embeddings (torch.Tensor, optional): Long tensor of shape `(vocab_size, emb_size)` - embedding tensor.
              Embedding size value would inherited from shape of this tensor.
            n_layers (int, optional): Number of transformer layers.
            dim_m (int, optional): Model hidden size, must be equal with embedding size.
            dim_q_k (int, optional): Dimension of `query` & `key` attention projections.
            dim_v (int, optional): Dimension of `value` attention projection.
            n_heads (int, optional): Number of attention heads.
            dim_i (int, optional): Inner dimension of feed-forward position-wise sublayer.
            dropout (float, optional): Dropout probability.

        Variables:
            - **encoder_state**: a float tensor of shape `(batch, enc_seq_len, dim_m)` containing encoder state from
              last layer.

        Inputs:
            - **enc_seq** of shape `(batch, enc_seq_len)`, a long tensor encoder input sequence.
            - **dec_seq** of shape `(batch, dec_seq_len)`, a long tensor decoder input sequence.

        Outputs:
            - **output** of of shape `(batch, dec_seq_len, vocab_size)`, a float tensor of vocabulary probability
              distribution.

        Notes:
            - For optimizing model, encoder state stores in local variable and calculate only one per batch. After
              auto-regressive process encoder state must be reset. You can do this using
              :func:`Transformer.reset_encoder_state`.
        """
        super(Transformer, self).__init__()

        self.positional_encoding = PositionalEmbedding(
            max_seq_len, dim_m, vocab_size, emb_size, embeddings)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(dim_m, dim_q_k, dim_v, n_heads, dim_i,
                                    dropout) for i in range(n_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(dim_m, dim_q_k, dim_v, n_heads, dim_i,
                                    dropout) for i in range(n_layers)
        ])
        # I think it's better to use smooth transition from dim_m to vocab_size
        self.out = nn.Sequential(
            nn.Linear(dim_m, vocab_size),
            # nn.ReLU(),
            # nn.Linear(7000, vocab_size),
        )
        self.softmax = nn.Softmax(-1)

        self.encoder_state = None

    def forward(self, enc_seq, dec_seq):
        # Calculate encoder state for batch.
        if self.encoder_state is None:
            # Sum embeddings with positional encodings.
            self.encoder_state = self.positional_encoding(enc_seq)

            for enc_layer in self.encoder_layers:
                self.encoder_state = enc_layer(self.encoder_state)

        # Decoder block.
        # Apply positional encoding.
        dec_state = self.positional_encoding(dec_seq)

        mask = self.autoregressive_mask(dec_seq)

        for dec_layer in self.decoder_layers:
            dec_state = dec_layer(dec_state, self.encoder_state, mask)

        output = self.out(dec_state)

        return output

    def reset_encoder_state(self):
        """Reset previous encoder state of batch. This method must calls before process new batch.
        """
        self.encoder_state = None

    @staticmethod
    def autoregressive_mask(tensor):
        """Generate auto-regressive mask for tensor. It's used to preserving the auto-regressive property.

        Args:
            tensor (torch.Tensor): of shape `(batch, seq_len, dim)`.

        Returns:
            torch.Tensor: a byte mask tensor of shape `(batch, seq_len, seq_len)` containing mask for
            illegal attention connections between decoder sequence tokens.

        """
        batch_size, seq_len = tensor.shape
        x = torch.ones(
            seq_len, seq_len, device=tensor.device).tril(-1).transpose(0, 1)

        return x.repeat(batch_size, 1, 1).byte()
