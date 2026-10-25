"""
Module: samplers.py
Implements sampling algorithms for diffusion models.
"""

from __future__ import annotations

from typing import Callable, Union

import numpy as np
import torch
from torch import Tensor


def get_sampler(sampler_type: str) -> Callable:
    """Get the sampler function based on the specified type."""
    if sampler_type == "euler":
        return euler_maruyama_integrator
    elif sampler_type == "pc":
        return predictor_corrector_sampler
    elif sampler_type == "pf":
        return probability_flow_sampler
    elif sampler_type == "expint":
        return exponential_integrator_sampler
    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}")


def euler_maruyama_integrator(
    x_0: Tensor,
    t_0: float,
    t_end: float,
    n_steps: int,
    drift_coefficient: Callable[[Tensor, Tensor], Tensor],
    diffusion_coefficient: Callable[[Tensor], Tensor],
    score_model,
    seed: Union[int, None] = None
) -> Tensor:
    """Euler-Maruyama integrator for stochastic differential equations."""
    device = x_0.device
    if seed is not None:
        torch.manual_seed(seed)

    times = torch.linspace(t_0, t_end, n_steps + 1, device=device)
    dt = times[1] - times[0]

    x_t = torch.tensor(
        np.empty((*np.shape(x_0), len(times))),
        dtype=torch.float32,
        device=device,
    )
    x_t[..., 0] = x_0

    z = torch.randn_like(x_t)

    for n, t in enumerate(times[:-1]):
        t = torch.ones(x_0.shape[0], device=device) * t
        x_t[..., n + 1] = (
            x_t[..., n]
            + drift_coefficient(x_t[..., n], t) * dt
            + diffusion_coefficient(t).view(-1, 1, 1, 1)
            * torch.sqrt(torch.abs(dt))
            * z[..., n]
        )

    return times, x_t


def predictor_corrector_sampler(
    x_0: Tensor,
    t_0: float,
    t_end: float,
    n_steps: int,
    drift_coefficient: Callable[[Tensor, Tensor], Tensor],
    diffusion_coefficient: Callable[[Tensor], Tensor],
    score_model,
    snr: float = 0.16,
    seed: Union[int, None] = None
) -> Tensor:
    """Predictor–Corrector sampler for diffusion models."""
    device = x_0.device
    if seed is not None:
        torch.manual_seed(seed)

    times = torch.linspace(t_0, t_end, n_steps + 1, device=device)
    dt = times[1] - times[0]

    x = x_0
    images = []

    for i in range(n_steps):
        t = times[i]
        images.append(x)
        t_batch = t * torch.ones(x.shape[0], device=device)

        g = diffusion_coefficient(t_batch).view(-1, 1, 1, 1)
        grad = score_model(x, t_batch)
        noise = torch.randn_like(x)
        step_size = (snr * g) ** 2 * 2.0
        x = x + step_size * grad + torch.sqrt(2.0 * step_size) * noise

        noise = torch.randn_like(x)
        x = (
            x
            + drift_coefficient(x, t_batch) * dt
            + g * torch.sqrt(torch.abs(dt)) * noise
        )

    images.append(x)
    x_t = torch.stack(images, dim=-1)
    return times, x_t


def probability_flow_sampler(
    x_0: Tensor,
    t_0: float,
    t_end: float,
    n_steps: int,
    drift_coefficient: Callable[[Tensor, Tensor], Tensor],
    diffusion_coefficient: Callable[[Tensor], Tensor],
    score_model,
    seed: Union[int, None] = None
) -> Tensor:
    """ODE-based probability flow sampler."""
    device = x_0.device
    if seed is not None:
        torch.manual_seed(seed)

    times = torch.linspace(t_0, t_end, n_steps + 1, device=device)
    dt = times[1] - times[0]

    x = x_0
    traj = []

    for i, t in enumerate(times[:-1]):
        traj.append(x)
        t_batch = t * torch.ones(x.shape[0], device=device)
        g = diffusion_coefficient(t_batch).view(-1, 1, 1, 1)
        f = drift_coefficient(x, t_batch)
        s = score_model(x, t_batch)
        drift_pf = f - 0.5 * g ** 2 * s
        x = x + drift_pf * dt

    traj.append(x)
    x_t = torch.stack(traj, dim=-1)
    return times, x_t


def exponential_integrator_sampler(
    x_0: Tensor,
    t_0: float,
    t_end: float,
    n_steps: int,
    drift_coefficient: Callable[[Tensor, Tensor], Tensor],
    diffusion_coefficient: Callable[[Tensor], Tensor],
    score_model,
    seed: Union[int, None] = None
) -> Tensor:
    """ODE sampler using an exponential integrator method."""
    device = x_0.device
    if seed is not None:
        torch.manual_seed(seed)

    times = torch.linspace(t_0, t_end, n_steps + 1, device=device)
    dt = times[1] - times[0]

    x = x_0
    traj = []

    for t in times[:-1]:
        traj.append(x)
        t_batch = t * torch.ones(x.shape[0], device=device)
        drift_pf = drift_coefficient(x, t_batch)
        x = x + drift_pf * dt

    traj.append(x)
    x_t = torch.stack(traj, dim=-1)
    return times, x_t
