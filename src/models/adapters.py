from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class AdapterConfig:
    adapter: str  # "lora" or "dora"
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    train_bias: bool = False
    train_base: bool = False


class LoRALinear(nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        rank: int,
        alpha: float,
        dropout: float,
        train_bias: bool,
        train_base: bool,
    ):
        super().__init__()
        out_features, in_features = weight.shape
        self.rank = rank
        self.scaling = alpha / max(rank, 1)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else None

        self.weight = nn.Parameter(weight, requires_grad=train_base)
        if bias is None:
            self.bias = None
        else:
            self.bias = nn.Parameter(bias, requires_grad=train_bias or train_base)

        self.lora_A = nn.Parameter(torch.randn(rank, in_features, device=weight.device, dtype=weight.dtype) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, device=weight.device, dtype=weight.dtype))

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float,
        train_bias: bool,
        train_base: bool,
    ) -> "LoRALinear":
        return cls(
            linear.weight.data.clone(),
            linear.bias.data.clone() if linear.bias is not None else None,
            rank,
            alpha,
            dropout,
            train_bias,
            train_base,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = F.linear(x, self.weight, self.bias)
        if self.rank == 0:
            return result
        x_in = self.dropout(x) if self.dropout is not None else x
        lora_update = F.linear(F.linear(x_in, self.lora_A), self.lora_B) * self.scaling
        return result + lora_update


class DoRALinear(nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        rank: int,
        alpha: float,
        dropout: float,
        train_bias: bool,
        train_base: bool,
        eps: float = 1e-8,
    ):
        super().__init__()
        out_features, in_features = weight.shape
        self.rank = rank
        self.scaling = alpha / max(rank, 1)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else None
        self.eps = eps

        self.weight = nn.Parameter(weight, requires_grad=train_base)
        if bias is None:
            self.bias = None
        else:
            self.bias = nn.Parameter(bias, requires_grad=train_bias or train_base)

        self.lora_A = nn.Parameter(torch.randn(rank, in_features, device=weight.device, dtype=weight.dtype) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, device=weight.device, dtype=weight.dtype))
        # Initialize scale to match pretrained weight norms so w_dora == weight at init.
        init_scale = torch.norm(weight, dim=1).clamp_min(self.eps)
        self.dora_scale = nn.Parameter(init_scale)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float,
        train_bias: bool,
        train_base: bool,
    ) -> "DoRALinear":
        return cls(
            linear.weight.data.clone(),
            linear.bias.data.clone() if linear.bias is not None else None,
            rank,
            alpha,
            dropout,
            train_bias,
            train_base,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.rank == 0:
            return F.linear(x, self.weight, self.bias)
        x_in = self.dropout(x) if self.dropout is not None else x
        delta_w = self.scaling * (self.lora_B @ self.lora_A)
        w_eff = self.weight + delta_w
        row_norm = torch.norm(w_eff, dim=1, keepdim=True).clamp_min(self.eps)
        w_hat = w_eff / row_norm
        w_dora = self.dora_scale[:, None] * w_hat
        return F.linear(x, w_dora, self.bias)


def _replace_linears(module: nn.Module, factory: Callable[[nn.Linear], nn.Module]) -> None:
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, factory(child))
        else:
            _replace_linears(child, factory)


def apply_adapters_to_model(model: nn.Module, cfg: AdapterConfig) -> None:
    """Replace Linear layers in encoder/head with LoRA/DoRA adapters."""
    if not cfg.train_base:
        for param in model.parameters():
            param.requires_grad = False

    adapter = cfg.adapter.lower()
    if adapter == "lora":
        factory = lambda layer: LoRALinear.from_linear(
            layer,
            cfg.rank,
            cfg.alpha,
            cfg.dropout,
            cfg.train_bias,
            cfg.train_base,
        )
    elif adapter == "dora":
        factory = lambda layer: DoRALinear.from_linear(
            layer,
            cfg.rank,
            cfg.alpha,
            cfg.dropout,
            cfg.train_bias,
            cfg.train_base,
        )
    else:
        raise ValueError(f"Unknown adapter type: {cfg.adapter}")

    _replace_linears(model.encoder, factory)
    _replace_linears(model.head, factory)
