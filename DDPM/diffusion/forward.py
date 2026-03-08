# diffusion/forward.py
"""
Forward (noising) process utilities.

q(x_1:L | x_0) = prod_{i=1}^{L} q(x_i | x_{i-1})
q(x_i | x_{i-1}) = N( sqrt(alpha_i) * x_{i-1}, beta_i * I )

Closed-form marginal (eq. 2):
q(x_i | x_0) = N( sqrt(bar_alpha_i) * x_0, (1 - bar_alpha_i) * I )
             => x_i = sqrt(bar_alpha_i)*x_0 + sqrt(1-bar_alpha_i)*eps,  eps~N(0,I)
"""

import torch


def q_sample(schedule, x0, t, eps=None):
    """
    Sample from the forward process marginal q(x_t | x_0).

    Uses the reparameterisation:
        x_t = sqrt(bar_alpha_t) * x_0  +  sqrt(1 - bar_alpha_t) * eps

    Args:
        schedule : DiffusionSchedule
        x0       : clean data, shape (B, C, H, W)
        t        : 1-indexed timesteps, shape (B,)   -- values in {1, ..., L}
        eps      : optional pre-sampled noise; sampled from N(0,I) if None

    Returns:
        x_t      : noised data, same shape as x0
        eps      : the noise that was used (needed by the training loss)
    """
    if eps is None:
        eps = torch.randn_like(x0)

    t_idx = t - 1  # convert to 0-indexed for tensor lookup

    sqrt_ab = schedule._extract(schedule.sqrt_alphas_bar, t_idx, x0.shape)
    sqrt_1mab = schedule._extract(schedule.sqrt_one_minus_alphas_bar, t_idx, x0.shape)

    x_t = sqrt_ab * x0 + sqrt_1mab * eps
    return x_t, eps


def sample_timesteps(batch_size, L, device):
    """
    Sample a batch of timesteps uniformly from {1, 2, ..., L}.

    Args:
        batch_size: number of timesteps to sample
        L         : total diffusion steps
        device    : torch device

    Returns:
        t: integer tensor of shape (batch_size,), values in [1, L]
    """
    return torch.randint(low=1, high=L + 1, size=(batch_size,), device=device)
