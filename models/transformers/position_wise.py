from torch import nn


class PositionWise(nn.Module):
    def __init__(self, dim_m, dim_i, dropout=0.1):
        """Position-wise Feed-Forward Network.

        Args:
            dim_m (int): input and output dimension.
            dim_i (int): inner dimension.
            dropout (float, optional): dropout probability.

        Inputs:
            - **input** of shape `(batch, *, dim_m)`: a float tensor.

        Outputs:
            - **output** of shape `(batch, *, dim_m)`: a float tensor.
        """
        super(PositionWise, self).__init__()

        self.feedforward = nn.Sequential(
            nn.Linear(dim_m, dim_i), nn.ReLU(), nn.Linear(dim_i, dim_m),
            nn.Dropout(dropout))
        self.normalization = nn.LayerNorm(dim_m, eps=1e-12)

    def forward(self, input):
        # There's nothing difficult here.
        residual = input
        output = self.feedforward(input)
        output = self.normalization(output + residual)
        return output
