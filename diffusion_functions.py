import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import numpy as np
from functools import partial
from tqdm import trange
import random

from diffusion_process import GaussianDiffussionProcess
from samplers import get_sampler
from diffusion_utilities import plot_image_evolution
from score_model import ScoreNet
from noise_schedules import get_sigma_schedule

# --- Global parameters ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sigma = 50
batch_size = 32
learning_rate = 1e-3
T = 1.0

# --- Tipos de procesos ---
class DiffusionType:
    VE = "ve"
    VP = "vp"
    SUBVP = "subvp"

# --- Funciones genéricas para cada tipo ---
def get_diffusion_functions(diff_type, sigma):
    if diff_type == DiffusionType.VE:
        def drift_fn(x_t, t):
            return torch.zeros_like(x_t)

        def diffusion_fn(t):
            return sigma(t)

        def mu_fn(x_0, t):
            return x_0

        def sigma_t_fn(t):
            s = sigma(t)
            return torch.sqrt(0.5 * (s**2 - 1.0) / np.log(sigma(torch.tensor(1.0))))

        def backward_drift(score_model, x_t, t):
            g = diffusion_fn(t).view(-1, 1, 1, 1)
            return -g**2 * score_model(x_t, t)

    elif diff_type in [DiffusionType.VP, DiffusionType.SUBVP]:
        beta_min = 0.1
        beta_max = 20.0

        def beta(t):
            return beta_min + t * (beta_max - beta_min)

        def integral_beta(t):
            return 0.5 * t * (2 * beta_min + t * (beta_max - beta_min))

        def drift_fn(x_t, t):
            return -0.5 * beta(t)[:, None, None, None] * x_t

        def diffusion_fn(t):
            if diff_type == DiffusionType.VP:
                return torch.sqrt(beta(t))
            elif diff_type == DiffusionType.SUBVP:
                return beta(t) * (1.0 - torch.exp(-2.0 * integral_beta(t)))

        def mu_fn(x_0, t):
            return x_0 * torch.exp(-0.5 * integral_beta(t))

        def sigma_t_fn(t):
            return torch.sqrt(1.0 - torch.exp(-integral_beta(t)))

        def backward_drift(score_model, x_t, t):
            b = beta(t).view(-1, 1, 1, 1)
            g = diffusion_fn(t).view(-1, 1, 1, 1)
            if diff_type == DiffusionType.VP:
                return -0.5 * b * score_model(x_t, t)
            else:
                return -(0.5 * b + g**2) * score_model(x_t, t)

    else:
        raise ValueError(f"Unknown diffusion type: {diff_type}")

    return drift_fn, diffusion_fn, mu_fn, sigma_t_fn, backward_drift


class RandomColorize:
    def __call__(self, img):
        r = img * random.uniform(0.6, 1.0)
        g = img * random.uniform(0.2, 0.8)
        b = img * random.uniform(0.1, 0.6)
        return torch.cat([r, g, b], dim=0)


def get_dataset(dataset):
    if dataset == "mnist":
        dataset = datasets.MNIST(
            root="data",
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), RandomColorize()])
        )
    elif dataset == "cifar10":
        dataset = datasets.CIFAR10(
            root="data",
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        )
    else:
        print("El dataset introducido es desconocido. El modelo se entrenará con CIFAR-10")
        dataset = datasets.CIFAR10(
            root="data",
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        )
    return dataset


def train_model(n_epochs=50, dataset="cifar10", diffusion_type="ve", schedule_type="exponential"):
    sigma_fn = get_sigma_schedule(schedule_type, sigma_max=25.0, sigma_min=0.01)
    drift_fn, diffusion_fn, mu_fn, sigma_t_fn, _ = get_diffusion_functions(diffusion_type, sigma_fn)

    diffusion_process = GaussianDiffussionProcess(
        drift_coefficient=drift_fn,
        diffusion_coefficient=diffusion_fn,
        mu_t=mu_fn,
        sigma_t=sigma_t_fn
    )

    full_dataset = get_dataset(dataset)
    data_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    score_model = torch.nn.DataParallel(
        ScoreNet(marginal_prob_std=partial(sigma_t_fn), num_classes=10)
    ).to(device)

    optimizer = Adam(score_model.parameters(), lr=learning_rate)

    for epoch in trange(n_epochs):
        avg_loss = 0.0
        num_items = 0
        avg_bpd = 0.0

        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            loss = diffusion_process.loss_function(score_model, x, class_labels=y)

            image_dims = x.shape[1] * x.shape[2] * x.shape[3]
            bpd = loss / (np.log(2) * image_dims)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() * x.size(0)
            num_items += x.size(0)

        avg_bpd += bpd.item() * x.size(0)
        print(f"Epoch {epoch+1}/{n_epochs} - Loss: {avg_loss / num_items:.6f} - BPD: {avg_bpd / num_items:.6f}")
        torch.save(score_model.state_dict(), f"./checkpoints/checkpoint_{dataset}_{diffusion_type}_{schedule_type}.pth")

    return score_model


def generate_and_plot(
    score_model,
    dataset="cifar10",
    diffusion_type="ve",
    schedule_type="exponential",
    sampler="euler",
    n_images=10,
    label=None,
    imputation=False
):
    sigma_fn = get_sigma_schedule(schedule_type, sigma_max=25.0, sigma_min=0.01)
    _, diffusion_fn, _, _, backward_drift = get_diffusion_functions(diffusion_type, sigma_fn)

    sampler = get_sampler(sampler)

    if imputation:
        dataset = get_dataset(dataset)
        loader = DataLoader(dataset, batch_size=n_images, shuffle=True)
        x_real, y_real = next(iter(loader))
        x_real, y_real = x_real.to(device), y_real.to(device)

        if label is not None:
            all_loader = DataLoader(dataset, batch_size=1000, shuffle=True)
            for x_batch, y_batch in all_loader:
                mask_idx = (y_batch == label)
                if mask_idx.sum() >= n_images:
                    x_real = x_batch[mask_idx][:n_images].to(device)
                    y_real = y_batch[mask_idx][:n_images].to(device)
                    break
            labels = y_real
        else:
            labels = y_real

        mask = torch.ones_like(x_real)
        mask[:, :, 10:18, 10:18] = 0
        x_0 = torch.randn_like(x_real).to(device)

        score_fn = partial(score_model, class_labels=labels)

        def drift_wrapper(x_t, t):
            return backward_drift(score_fn, x_t, t)

        with torch.no_grad():
            times, synthetic_images_t = sampler(
                x_0=x_0,
                t_0=1.0,
                t_end=1e-3,
                n_steps=500,
                drift_coefficient=drift_wrapper,
                diffusion_coefficient=diffusion_fn,
                score_model=score_fn
            )

            for i in range(synthetic_images_t.shape[-1]):
                synthetic_images_t[..., i] = (
                    synthetic_images_t[..., i] * (1 - mask) + x_real * mask
                )

    else:
        image_size = 32 if dataset == "cifar10" else 28
        x_0 = torch.randn(n_images, 3, image_size, image_size).to(device)

        if label is not None:
            labels = torch.full((n_images,), label, dtype=torch.long, device=device)
            score_fn = lambda x, t: score_model(x, t, class_labels=labels)
        else:
            score_fn = lambda x, t: score_model(x, t, class_labels=torch.randint(0, 10, (x.shape[0],), device=device))

        def drift_wrapper(x_t, t):
            return backward_drift(score_fn, x_t, t)

        with torch.no_grad():
            times, synthetic_images_t = sampler(
                x_0=x_0,
                t_0=1.0,
                t_end=1e-3,
                n_steps=500,
                drift_coefficient=drift_wrapper,
                diffusion_coefficient=diffusion_fn,
                score_model=score_fn
            )

    synthetic_images_t = synthetic_images_t.permute(4, 0, 1, 2, 3)

    _ = plot_image_evolution(
        images=synthetic_images_t.cpu(),
        n_images=n_images,
        n_intermediate_steps=[0, 100, 200, 470, 480, 490, 500],
        figsize=(20, 20)
    )
