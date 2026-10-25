"""
Module: evaluate_metrics.py
Contains functions for loading models and computing FID and Inception Score metrics.
"""

import random
import torch
import numpy as np
from score_model import ScoreNet
from torchvision import datasets, transforms
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance
from torch.utils.data import DataLoader
import torch.nn.functional as F
from diffusion_functions import get_diffusion_functions, get_sigma_schedule, get_dataset
from samplers import euler_maruyama_integrator
from functools import partial


class RandomColorize:
    def __call__(self, img):
        r = img * random.uniform(0.6, 1.0)
        g = img * random.uniform(0.2, 0.8)
        b = img * random.uniform(0.7, 1.0)
        return torch.stack([r, g, b], dim=0)


def cargar_modelo(model_path, schedule_type, diffusion_type):
    """Carga un modelo ScoreNet entrenado desde fichero."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    sigma_fn = get_sigma_schedule(schedule_type, sigma_max=25.0, sigma_min=0.01)
    _, _, _, sigma_t_fn, _ = get_diffusion_functions(diffusion_type, sigma_fn)

    model = torch.nn.DataParallel(
        ScoreNet(marginal_prob_std=partial(sigma_t_fn), num_classes=10)
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model


def compute_fid_and_iscore(model, dataset="cifar10", diffusion_type="ve", schedule_type="exponential"):
    """Compute FID and Inception Score for a diffusion model on a given dataset."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    sigma_fn = get_sigma_schedule(schedule_type, sigma_max=25.0, sigma_min=0.01)
    _, diffusion_fn, _, _, backward_drift = get_diffusion_functions(diffusion_type, sigma_fn)

    fid_metric = FrechetInceptionDistance(feature=64).to(device)
    is_metric = InceptionScore().to(device)

    real_dataset = get_dataset(dataset)
    real_loader = DataLoader(real_dataset, batch_size=32, shuffle=True, num_workers=0)
    real_images, _ = next(iter(real_loader))
    real_images = real_images.to(device)
    real_resized = F.interpolate((real_images + 1) / 2, size=(299, 299), mode='bilinear', align_corners=False)
    real_uint8 = (real_resized * 255).clamp(0, 255).to(torch.uint8)
    fid_metric.update(real_uint8, real=True)

    model.eval()
    batch_size = 32
    x_0 = torch.randn(batch_size, 3, 32, 32, device=device)
    labels = torch.randint(0, 10, (batch_size,), device=device)

    def drift_wrapper(x_t, t):
        return backward_drift(
            lambda x, t: model(x, t, class_labels=labels),
            x_t, t
        )

    with torch.no_grad():
        times, traj = euler_maruyama_integrator(
            x_0=x_0,
            t_0=1.0,
            t_end=1e-3,
            n_steps=500,
            drift_coefficient=drift_wrapper,
            diffusion_coefficient=diffusion_fn,
            score_model=lambda x, t: model(x, t, class_labels=labels),
        )
        fake_images = traj[..., -1]

    fake_resized = F.interpolate((fake_images + 1) / 2, size=(299, 299), mode='bilinear', align_corners=False)
    fake_uint8 = (fake_resized * 255).clamp(0, 255).to(torch.uint8)
    fid_metric.update(fake_uint8, real=False)
    is_metric.update(fake_uint8)

    fid_score = fid_metric.compute().item()
    is_mean, is_std = is_metric.compute()
    return fid_score, is_mean, is_std


def bpd(loss, image_shape):
    """Compute bits per dimension (BPD) for a given loss and image shape."""
    return loss / (np.prod(image_shape) * np.log(2))
