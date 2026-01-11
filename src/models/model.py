from typing import Literal, Optional, Tuple

import torch
from torch import nn

from .encoders import SeparateTabularEncoder, SharedTabularEncoder
from .heads import RegressionHead


class TabularRegressor(nn.Module):
    """
    Wrapper that supports shared or separate encoders for numeric/categorical inputs.
    Mode:
      - "shared": expects a single concatenated input
      - "separate": expects a tuple (x_num, x_cat)
    """

    def __init__(
        self,
        mode: Literal["shared", "separate"],
        input_dim: Optional[int] = None,
        num_dim: Optional[int] = None,
        cat_dim: Optional[int] = None,
        encoder_hidden=(128, 64),
        head_hidden=(),
        dropout=0.1,
    ):
        super().__init__()
        self.mode = mode
        if mode == "shared":
            if input_dim is None:
                raise ValueError("input_dim is required for shared mode")
            self.encoder = SharedTabularEncoder(input_dim, hidden_dims=encoder_hidden, dropout=dropout)
            enc_out_dim = encoder_hidden[-1] if encoder_hidden else input_dim
        elif mode == "separate":
            if num_dim is None or cat_dim is None:
                raise ValueError("num_dim and cat_dim are required for separate mode")
            self.encoder = SeparateTabularEncoder(
                num_dim=num_dim,
                cat_dim=cat_dim,
                num_hidden=encoder_hidden,
                cat_hidden=encoder_hidden,
                dropout=dropout,
            )
            enc_out_dim = self.encoder.out_dim
        else:
            raise ValueError("mode must be 'shared' or 'separate'")

        self.head = RegressionHead(enc_out_dim, hidden_dims=head_hidden, dropout=dropout)

    def forward(self, x: torch.Tensor | Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        if self.mode == "shared":
            z = self.encoder(x)
        else:
            if not isinstance(x, (tuple, list)) or len(x) != 2:
                raise ValueError("Separate mode expects a tuple (x_num, x_cat)")
            z = self.encoder(x[0], x[1])
        return self.head(z).squeeze(-1)