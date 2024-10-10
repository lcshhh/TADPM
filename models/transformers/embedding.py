import numpy as np
import torch
from torch import nn


class PositionalEmbedding(nn.Module):
    def __init__(self,
                 max_seq_len,
                 dim_m,
                 vocab_size,
                 emb_size=None,
                 embeddings=None):
        """Embeddings with positional encoding.

        Args:
            max_seq_len (int): Max length of the sequence.
            dim_m (int): Model dimension.
            vocab_size (int): Vocabulary size.
            emb_size (int, optional): Embedding size. You do not need to specify a value if you are using
              embedding weights.
            embeddings (torch.Tensor, optional): Tensor `(vocab_size, emb_size)` of embeddings weights. Embedding size
              value would inherited from shape of this tensor.

        Inputs:
            - **input**: Long tensor of shape `(batch, seq_len)` - input sequence.

        Outputs:
            - **output**: Float tensor of shape `(batch, seq_len, emb_size)` - output sequence.

        Notes:
            - Model dimension and embedding size haven't to be equal. There is an alignment layer, that project
              embedding to model size.
        """
        super(PositionalEmbedding, self).__init__()
        self.max_seq_len = max_seq_len + 1
        self.dim_m = dim_m

        self.positional = nn.Embedding(max_seq_len + 1, dim_m, padding_idx=0)

        if embeddings is None:
            self.embedding = nn.Embedding(vocab_size, emb_size)
        else:
            emb_size = embeddings.shape[1]
            self.embedding = nn.Embedding(*embeddings.shape)
            self.embedding.weight = nn.Parameter(
                embeddings, requires_grad=False)

        self.alignment = nn.Linear(emb_size, dim_m, bias=False)

        self.reset_parameters()

    def forward(self, input):
        mask = input == 0

        pos_mask = self.position_mask(input)
        pos_mask.masked_fill_(mask, 0)
        return self.alignment(
            self.embedding(input)) + self.positional(pos_mask)

    def reset_parameters(self):
        # Lookup table for position codes: (max_seq_len, dim_m)
        weights = [
            self.sin_position_scale(i, np.arange(0, self.dim_m))
            for i in range(self.max_seq_len)
        ]
        weights = np.stack(weights)
        weights[1:, ::2] = np.sin(weights[1:, ::2])
        weights[1:, 1::2] = np.cos(weights[1:, 1::2])
        self.positional.weight = nn.Parameter(
            torch.tensor(weights, dtype=torch.float), requires_grad=False)

    def sin_position_scale(self, pos, i):
        """Position scaling :math:`pos/10000^{i*dim_m}` for Sinusoidal Positional Encoding.

        Args:
            pos (int): Position index.
            i (numpy.ndarray): Dimension indexes.

        Returns:
            float: Scaled value.
        """
        return pos / np.power(1e4, i / self.dim_m)

    @staticmethod
    def position_mask(tensor):
        """Generate position mask for tensor.

        Args:
            tensor (torch.tensor): a float tensor of shape `(batch_size, seq_len, *)`.

        Returns:
            torch.tensor: an int tensor of word positions.

        """
        # Maybe it would be more productive to use a global buffer of positions `(max_batch_size, max_seq_len)`
        # and get a mask from this buffer using slicing.
        batch_size, seq_len = tensor.shape
        mask = torch.arange(
            1, seq_len + 1, dtype=torch.long, device=tensor.device).repeat(
                batch_size, 1)

        return mask
