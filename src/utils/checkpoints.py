import torch


def save_checkpoint(model, path):
    """Save a model's state_dict to disk."""
    torch.save(model.state_dict(), path)


def load_checkpoint(model, path, map_location=None):
    """Load state_dict into model and return it."""
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state)
    return model
