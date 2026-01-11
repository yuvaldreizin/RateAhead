import torch
from torch import nn


class SharedTabularEncoder(nn.Module):
    """Shared encoder: consume concatenated numeric + encoded categorical features."""

    def __init__(self, input_dim: int, hidden_dims=(128, 64), dropout=0.1):
        super().__init__()
        layers = []
        dims = [input_dim, *hidden_dims]
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(d_in, d_out))
            layers.append(nn.ReLU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SeparateTabularEncoder(nn.Module):
    """Separate blocks for numeric and categorical, then concat."""

    def __init__(
        self,
        num_dim: int,
        cat_dim: int,
        num_hidden=(64,),
        cat_hidden=(64,),
        dropout=0.1,
    ):
        super().__init__()
        self.num_block = self._mlp(num_dim, num_hidden, dropout)
        self.cat_block = self._mlp(cat_dim, cat_hidden, dropout)
        combined_dim = (num_hidden[-1] if num_hidden else num_dim) + (cat_hidden[-1] if cat_hidden else cat_dim)
        self.out_dim = combined_dim

    @staticmethod
    def _mlp(input_dim: int, hidden_dims, dropout):
        if not hidden_dims:
            return nn.Identity()
        layers = []
        dims = [input_dim, *hidden_dims]
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(d_in, d_out))
            layers.append(nn.ReLU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        num_out = self.num_block(x_num)
        cat_out = self.cat_block(x_cat)
        return torch.cat([num_out, cat_out], dim=1)
