"""
Module: diffusion_utilities.py
Provides visualization functions for image grids, diffusion evolution and animations.
"""

from numpy.typing import ArrayLike

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import Colormap

import torch
from torchvision.utils import make_grid
from torchvision.transforms import functional


def plot_image_grid(
    images: torch.Tensor,
    figsize: tuple,
    n_rows: int,
    n_cols: int,
    padding: int = 2,
    pad_value: int = 1.0,
    cmap: Colormap = "gray",
    normalize: bool = False,
    axis_on_off: str = "off",
):
    """Displays a grid of images."""
    grid = make_grid(
        images,
        nrow=n_cols,
        padding=padding,
        normalize=normalize,
        pad_value=pad_value,
    )

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(functional.to_pil_image(grid), cmap=cmap)
    ax.axis("off")
    return fig, ax


def plot_image_evolution(images, n_images=10, n_intermediate_steps=None, figsize=(15, 4)):
    """Visualizes the evolution of generated images over diffusion steps."""
    if n_intermediate_steps is None:
        n_intermediate_steps = [0, 100, 200, 300, 400, 450, 470, 480, 490, 500]

    fig, axes = plt.subplots(len(n_intermediate_steps), n_images, figsize=figsize)

    time_steps_first = images.shape[0] == 501

    for i, step in enumerate(n_intermediate_steps):
        for j in range(n_images):
            img = images[step, j] if time_steps_first else images[j, step]
            img = img.clamp(0.0, 1.0).detach().cpu()

            if img.shape[0] == 3:
                img = img.permute(1, 2, 0).numpy()
                axes[i, j].imshow(img)
            elif img.shape[0] == 1:
                img = img.squeeze(0).numpy()
                axes[i, j].imshow(img, cmap='gray')
            else:
                raise ValueError(f"Unexpected image shape: {img.shape}")

            axes[i, j].axis("off")

    plt.tight_layout()
    plt.show()


def animation_images(images_t, interval, figsize):
    """Creates an animation from image frames."""
    _, _, n_frames = np.shape(images_t)

    fig, ax = plt.subplots(figsize=figsize)
    img_display = ax.imshow(images_t[:, :, 0], cmap="gray")

    def update(t):
        img_display.set_array(images_t[:, :, t])
        return [img_display]

    return (
        fig,
        ax,
        animation.FuncAnimation(
            fig,
            update,
            frames=n_frames,
            interval=interval,
            blit=False
        )
    )
