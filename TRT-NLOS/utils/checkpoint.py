"""Strict checkpoint loading for single-GPU and DP/DDP checkpoints."""
import torch


def load_checkpoint(model, path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    state = checkpoint.get('state_dict', checkpoint)
    state = {key.removeprefix('module.'): value for key, value in state.items()}
    target = model.module if hasattr(model, 'module') else model
    target.load_state_dict(state, strict=True)
    return checkpoint
