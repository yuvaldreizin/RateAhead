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
    loss_type: str = "mse"  # "mse" or "hl_gauss"
    num_bins: int = 101
    sigma_ratio: float = 0.75
    bin_centers: Optional[torch.Tensor] = None
    output_dim: Optional[int] = None
    monitor: str = "val_loss"
    monitor_mode: str = "min"


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
    loss_type = getattr(cfg, "loss_type", "mse")
    output_dim = getattr(cfg, "output_dim", None)
    if loss_type == "hl_gauss" and output_dim is None:
        output_dim = int(getattr(cfg, "num_bins", 101))
    if output_dim is None:
        output_dim = 1

    model = TabularRegressor(
        mode=cfg.mode,
        input_dim=cfg.input_dim,
        num_dim=cfg.num_dim,
        cat_dim=cfg.cat_dim,
        encoder_hidden=cfg.encoder_hidden,
        head_hidden=cfg.head_hidden,
        dropout=cfg.dropout,
        output_dim=output_dim,
    ).to(cfg.device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_mse": [],
        "val_mse": [],
        "train_mae": [],
        "val_mae": [],
    }

    if loss_type == "hl_gauss":
        if cfg.bin_centers is None:
            raise ValueError("cfg.bin_centers is required for hl_gauss loss")
        bin_centers = torch.as_tensor(cfg.bin_centers, device=cfg.device, dtype=torch.float32)
        if bin_centers.ndim != 1:
            raise ValueError("cfg.bin_centers must be a 1D tensor or array")
        if bin_centers.numel() < 2:
            raise ValueError("cfg.bin_centers must have at least 2 values")
        vmin = bin_centers[0]
        vmax = bin_centers[-1]
        bin_width = (vmax - vmin) / float(bin_centers.numel() - 1)
        sigma = float(cfg.sigma_ratio) * bin_width
        sigma = torch.clamp(sigma, min=1e-6)

        def _hl_gauss_loss(logits: torch.Tensor, y_true: torch.Tensor):
            y_clamped = torch.clamp(y_true, vmin, vmax).unsqueeze(-1)
            diffs = bin_centers.unsqueeze(0) - y_clamped
            weights = torch.exp(-(diffs ** 2) / (2.0 * sigma ** 2))
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-12)
            log_probs = torch.log_softmax(logits, dim=1)
            loss = -(weights * log_probs).sum(dim=1).mean()
            probs = torch.softmax(logits, dim=1)
            pred = (probs * bin_centers.unsqueeze(0)).sum(dim=1)
            return loss, pred
    else:
        loss_fn = nn.MSELoss()

    monitor_key = getattr(cfg, "monitor", "val_loss")
    monitor_mode = getattr(cfg, "monitor_mode", "min")
    best_metric = None
    best_epoch = None
    best_state = None

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_losses, train_mse, train_mae = [], [], []
        for batch in train_loader:
            if X_is_separate:
                xb_num, xb_cat, yb = batch
                xb = (xb_num.to(cfg.device), xb_cat.to(cfg.device))
            else:
                xb, yb = batch
                xb = xb.to(cfg.device)
            yb = yb.to(cfg.device)

            if loss_type == "hl_gauss":
                logits = model(xb)
                loss, pred = _hl_gauss_loss(logits, yb)
            else:
                pred = model(xb).squeeze(-1)
                loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_losses.append(loss.item())
            train_mse.append(torch.mean((pred - yb) ** 2).item())
            train_mae.append(torch.abs(pred - yb).mean().item())

        model.eval()
        val_losses, val_mse, val_mae = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                if X_is_separate:
                    xb_num, xb_cat, yb = batch
                    xb = (xb_num.to(cfg.device), xb_cat.to(cfg.device))
                else:
                    xb, yb = batch
                    xb = xb.to(cfg.device)
                yb = yb.to(cfg.device)
                if loss_type == "hl_gauss":
                    logits = model(xb)
                    loss, pred = _hl_gauss_loss(logits, yb)
                else:
                    pred = model(xb).squeeze(-1)
                    loss = loss_fn(pred, yb)
                val_losses.append(loss.item())
                val_mse.append(torch.mean((pred - yb) ** 2).item())
                val_mae.append(torch.abs(pred - yb).mean().item())

        epoch_train_loss = sum(train_losses) / len(train_losses)
        epoch_val_loss = sum(val_losses) / len(val_losses)
        epoch_train_mse = sum(train_mse) / len(train_mse)
        epoch_val_mse = sum(val_mse) / len(val_mse)
        epoch_train_mae = sum(train_mae) / len(train_mae)
        epoch_val_mae = sum(val_mae) / len(val_mae)

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_mse"].append(epoch_train_mse)
        history["val_mse"].append(epoch_val_mse)
        history["train_mae"].append(epoch_train_mae)
        history["val_mae"].append(epoch_val_mae)

        if monitor_key in history:
            current_metric = history[monitor_key][-1]
            is_best = best_metric is None
            if monitor_mode == "min":
                is_best = is_best or current_metric < best_metric
            else:
                is_best = is_best or current_metric > best_metric
            if is_best:
                best_metric = current_metric
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | train loss {epoch_train_loss:.4f} | "
                f"val loss {epoch_val_loss:.4f} | "
                f"train MSE {epoch_train_mse:.4f} | val MSE {epoch_val_mse:.4f} | "
                f"train MAE {epoch_train_mae:.4f} | val MAE {epoch_val_mae:.4f}"
            )

        # Simple freeze/unfreeze hook for finetune
        if cfg.freeze_encoder_epochs and epoch == cfg.freeze_encoder_epochs:
            for _, param in model.encoder.named_parameters():
                param.requires_grad = True

    if best_state is not None:
        model.load_state_dict(best_state)
    if best_epoch is not None:
        model.best_epoch = best_epoch
        model.best_metric = best_metric
        model.monitor_key = monitor_key

    return model, history