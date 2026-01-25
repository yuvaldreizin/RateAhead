import torch


def save_checkpoint(model, path):
    """Save a model's state_dict to disk."""
    torch.save(model.state_dict(), path)


def load_checkpoint(model, path, map_location=None):
    """Load state_dict into model and return it."""
    try:
        # PyTorch 2.6 defaults to weights_only=True; fallback for legacy checkpoints.
        state = torch.load(path, map_location=map_location)
    except Exception:
        state = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    return model
