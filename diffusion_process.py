"""
Module: diffusion_process.py
Defines base and Gaussian diffusion processes with support for training score-based models.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from torch import Tensor


class DiffussionProcess:
    def __init__(
        self,
        drift_coefficient: Callable[[float, float], float] = lambda x_t, t: 0.0,
        diffusion_coefficient: Callable[[float], float] = lambda t: 1.0,
    ):
        self.drift_coefficient = drift_coefficient
        self.diffusion_coefficient = diffusion_coefficient


class GaussianDiffussionProcess(DiffussionProcess):
    """Gaussian diffusion process with analytical mean and std for training."""

    kind = "Gaussian"

    def __init__(
        self,
        drift_coefficient: Callable[[float, float], float] = lambda x_t, t: 0.0,
        diffusion_coefficient: Callable[[float], float] = lambda t: 1.0,
        mu_t: Callable[[float, float], float] = lambda x_0, t: x_0,
        sigma_t: Callable[[float], float] = lambda t: np.sqrt(t),
    ):
        super().__init__(drift_coefficient, diffusion_coefficient)
        self.mu_t = mu_t
        self.sigma_t = sigma_t

    def loss_function(self, score_model, x_0, eps=1e-5, class_labels=None):
        t = torch.rand(x_0.shape[0], device=x_0.device) * (1.0 - eps) + eps
        t_expanded = t.view(-1, 1, 1, 1)
        noise = torch.randn_like(x_0)
        sigma = self.sigma_t(t)
        x_t = x_0 + sigma.view(-1, 1, 1, 1) * noise
        score = score_model(x_t, t, class_labels=class_labels)
        loss = torch.mean(torch.sum((score * sigma.view(-1, 1, 1, 1) + noise) ** 2, dim=(1, 2, 3)))
        return loss


if __name__ == "__main__":
    import doctest

    doctest.testmod()
