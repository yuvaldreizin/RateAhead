from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .model import TabularRegressor


@dataclass
class TrainConfig:
    """Lightweight config for training tabular regressors."""

    mode: str  # "shared" or "separate"
    input_dim: Optional[int] = None
    num_dim: Optional[int] = None
    cat_dim: Optional[int] = None
    encoder_hidden: Tuple[int, ...] = (128, 64)
    head_hidden: Tuple[int, ...] = ()
    dropout: float = 0.1
    lr: float = 1e-3
    epochs: int = 50
    batch_size: int = 64
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    freeze_encoder_epochs: int = 0  # for finetune


def make_loaders(
    X_train,
    y_train,
    X_val,
    y_val,
    batch_size=64,
    separate: bool = False,
):
    """Create PyTorch DataLoaders; supports shared (single tensor) or separate (num, cat)."""

    def to_tensor(x):
        return torch.from_numpy(x) if isinstance(x, (list, tuple)) else torch.from_numpy(x)

    if separate:
        train_ds = TensorDataset(to_tensor(X_train[0]), to_tensor(X_train[1]), torch.from_numpy(y_train))
        val_ds = TensorDataset(to_tensor(X_val[0]), to_tensor(X_val[1]), torch.from_numpy(y_val))

        def make_collate(ds):
            return DataLoader(ds, batch_size=batch_size, shuffle=ds is train_ds)

        return make_collate(train_ds), make_collate(val_ds)
    else:
        train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
        return (
            DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        )


def train_model(cfg: TrainConfig, train_loader, val_loader, X_is_separate: bool = False):
    """Basic training loop with MSE loss and optional freeze hook for finetune."""
    model = TabularRegressor(
        mode=cfg.mode,
        input_dim=cfg.input_dim,
        num_dim=cfg.num_dim,
        cat_dim=cfg.cat_dim,
        encoder_hidden=cfg.encoder_hidden,
        head_hidden=cfg.head_hidden,
        dropout=cfg.dropout,
    ).to(cfg.device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    history = {"train_loss": [], "val_loss": [], "train_mae": [], "val_mae": []}

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_losses, train_mae = [], []
        for batch in train_loader:
            if X_is_separate:
                xb_num, xb_cat, yb = batch
                xb = (xb_num.to(cfg.device), xb_cat.to(cfg.device))
            else:
                xb, yb = batch
                xb = xb.to(cfg.device)
            yb = yb.to(cfg.device)

            pred = model(xb).unsqueeze(1)
            loss = loss_fn(pred, yb.unsqueeze(1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_losses.append(loss.item())
            train_mae.append(torch.abs(pred - yb.unsqueeze(1)).mean().item())

        model.eval()
        val_losses, val_mae = [], []
        with torch.no_grad():
            for batch in val_loader:
                if X_is_separate:
                    xb_num, xb_cat, yb = batch
                    xb = (xb_num.to(cfg.device), xb_cat.to(cfg.device))
                else:
                    xb, yb = batch
                    xb = xb.to(cfg.device)
                yb = yb.to(cfg.device)
                pred = model(xb).unsqueeze(1)
                val_losses.append(loss_fn(pred, yb.unsqueeze(1)).item())
                val_mae.append(torch.abs(pred - yb.unsqueeze(1)).mean().item())

        history["train_loss"].append(sum(train_losses) / len(train_losses))
        history["val_loss"].append(sum(val_losses) / len(val_losses))
        history["train_mae"].append(sum(train_mae) / len(train_mae))
        history["val_mae"].append(sum(val_mae) / len(val_mae))

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | train MSE {history['train_loss'][-1]:.4f} | "
                f"val MSE {history['val_loss'][-1]:.4f} | "
                f"train MAE {history['train_mae'][-1]:.4f} | "
                f"val MAE {history['val_mae'][-1]:.4f}"
            )

        # Simple freeze/unfreeze hook for finetune
        if cfg.freeze_encoder_epochs and epoch == cfg.freeze_encoder_epochs:
            for _, param in model.encoder.named_parameters():
                param.requires_grad = True

    return model, history