"""Original TRT-NLOS reconstruction loss (root mean squared error)."""
import torch
import torch.nn.functional as F


def criterion_L2(est, gt):
    return torch.sqrt(F.mse_loss(est, gt))
