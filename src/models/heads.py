import torch
from torch import nn


class RegressionHead(nn.Module):
    """Simple MLP head ending in a regression output."""

    def __init__(self, input_dim: int, hidden_dims=(), dropout=0.0, output_dim: int = 1):
        super().__init__()
        layers = []
        dims = [input_dim, *hidden_dims, output_dim]
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(d_in, d_out))
            if d_out != output_dim:
                layers.append(nn.ReLU())
                if dropout and dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
