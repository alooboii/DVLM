# diffusion/ddpm.py
"""
Full ancestral DDPM sampler and image-saving utilities.
"""

import os
import torch
from torchvision.utils import save_image, make_grid

from diffusion.posterior import p_sample_step


@torch.no_grad()
def ddpm_sample(
    model,
    schedule,
    shape,
    device,
    save_trajectory=False,
    traj_timesteps=None,
    verbose=True,
):
    """
    Generate samples by running the full reverse chain from x_L ~ N(0,I) to x_0.

    Args:
        model           : eps_theta(x_t, t) network
        schedule        : DiffusionSchedule
        shape           : (B, C, H, W) -- number and shape of samples
        device          : torch device
        save_trajectory : if True, save intermediate x_t at selected steps
        traj_timesteps  : list of t values (1-indexed) to capture;
                          defaults to [L, 3L/4, L/2, L/4, 1]
        verbose         : print progress

    Returns:
        x0              : final samples, shape (B, C, H, W)
        trajectory      : dict {t: tensor} if save_trajectory, else empty dict
    """
    L = schedule.L
    B = shape[0]

    if save_trajectory and traj_timesteps is None:
        traj_timesteps = {L, 3 * L // 4, L // 2, L // 4, 1}
    traj_timesteps = set(traj_timesteps) if traj_timesteps else set()

    # ----------------------------------------------------------------------- #
    # Start from pure Gaussian noise  p(x_L) = N(0, I)
    # ----------------------------------------------------------------------- #
    x_t = torch.randn(shape, device=device)

    trajectory = {}
    if L in traj_timesteps:
        trajectory[L] = x_t.cpu().clone()

    model.eval()

    for t_val in reversed(range(1, L + 1)):
        if verbose and t_val % 200 == 0:
            print(f"  sampling step t={t_val}/{L}", flush=True)

        t = torch.full((B,), t_val, device=device, dtype=torch.long)
        eps_hat = model(x_t, t)
        x_t, _ = p_sample_step(schedule, x_t, t, eps_hat)

        if t_val in traj_timesteps:
            trajectory[t_val] = x_t.cpu().clone()

    return x_t, trajectory


def unnormalize(x, value_range=(-1.0, 1.0)):
    """Rescale from value_range to [0, 1] for display."""
    lo, hi = value_range
    return (x.clamp(lo, hi) - lo) / (hi - lo)


def save_sample_grid(samples, path, nrow=8, value_range=(-1.0, 1.0)):
    """
    Save a grid of samples to `path`.

    Args:
        samples    : tensor (B, C, H, W), values in value_range
        path       : output file path (e.g. 'output/samples.png')
        nrow       : images per row in the grid
        value_range: input range to rescale from
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    imgs = unnormalize(samples, value_range)
    save_image(imgs, path, nrow=nrow)


def save_trajectory_grid(trajectory, path, nrow=None, value_range=(-1.0, 1.0)):
    """
    Save a grid showing the denoising trajectory for a single sample.

    trajectory: dict {t_val: tensor (B, C, H, W)}
    Each column shows the image at one timestep (sorted descending: x_L ... x_0).
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    sorted_keys = sorted(trajectory.keys(), reverse=True)  # L, ..., 1
    # Take only the first sample from each
    frames = [unnormalize(trajectory[k][:1], value_range) for k in sorted_keys]
    grid = make_grid(torch.cat(frames, dim=0), nrow=len(frames))
    save_image(grid, path)
    print(f"Saved trajectory ({len(frames)} steps) → {path}")
