"""
Module: noise_schedules.py
Defines different noise schedules used in diffusion models.
"""

import torch
import numpy as np


def sigma_exponential(t, sigma_min=0.01, sigma_max=25.0):
    """Exponential noise schedule."""
    return sigma_max ** t


def sigma_linear(t, sigma_min=0.01, sigma_max=25.0):
    """Linear noise schedule."""
    return sigma_min + t * (sigma_max - sigma_min)


def sigma_cosine(t, sigma_min=0.01, sigma_max=25.0):
    """Cosine-based noise schedule."""
    return sigma_min + 0.5 * (sigma_max - sigma_min) * (1 + torch.cos(np.pi * (1 - t)))


def get_sigma_schedule(schedule_type, **kwargs):
    """Returns the sigma schedule function given a type."""
    if schedule_type == "exponential":
        return lambda t: sigma_exponential(t, **kwargs)
    elif schedule_type == "linear":
        return lambda t: sigma_linear(t, **kwargs)
    elif schedule_type == "cosine":
        return lambda t: sigma_cosine(t, **kwargs)
    else:
        raise ValueError(f"Schedule {schedule_type} not supported.")
