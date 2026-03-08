# diffusion/schedule.py
"""
Variance schedule {beta_i}_{i=1}^L and all derived scalar tensors.

Indexing convention used throughout this codebase:
  - Diffusion timesteps t are 1-indexed: t in {1, 2, ..., L}
  - All precomputed tensors are 0-indexed: tensor[0] corresponds to t=1
  - Use t_idx = t - 1 when indexing into these tensors
"""

import torch
import numpy as np


def make_beta_schedule(L, schedule_type="linear", beta_min=1e-4, beta_max=0.02):
    """
    Build the variance schedule {beta_i}_{i=1}^L.

    Args:
        L            : number of diffusion steps
        schedule_type: "linear" (Ho et al. 2020) or "cosine" (Nichol & Dhariwal 2021)
        beta_min     : lower endpoint for linear schedule
        beta_max     : upper endpoint for linear schedule

    Returns:
        betas: float32 tensor of shape (L,)
    """
    if schedule_type == "linear":
        betas = torch.linspace(beta_min, beta_max, L, dtype=torch.float64)

    elif schedule_type == "cosine":
        # Nichol & Dhariwal (2021) cosine schedule -- clips large betas for stability
        s = 0.008
        steps = L + 1
        t = torch.linspace(0, L, steps, dtype=torch.float64) / L
        alphas_bar = torch.cos((t + s) / (1.0 + s) * np.pi / 2.0) ** 2
        alphas_bar = alphas_bar / alphas_bar[0]  # normalise so bar_alpha_0 = 1
        betas = 1.0 - alphas_bar[1:] / alphas_bar[:-1]
        betas = torch.clamp(betas, min=1e-5, max=0.9999)

    else:
        raise ValueError(f"Unknown schedule_type '{schedule_type}'")

    return betas.float()


class DiffusionSchedule:
    """
    Wraps the variance schedule and exposes every precomputed scalar needed
    by the forward process, the true posterior, and the reverse sampler.

    All tensors have shape (L,) and live on `device`.  Use _extract() to
    pull out a batch of values and broadcast them against an image tensor.
    """

    def __init__(
        self,
        L=1000,
        schedule_type="linear",
        beta_min=1e-4,
        beta_max=0.02,
        device="cpu",
    ):
        self.L = L
        self.device = device

        # ------------------------------------------------------------------ #
        #  Basic schedule quantities                                           #
        # ------------------------------------------------------------------ #

        # beta_i  (0-indexed, shape L)
        betas = make_beta_schedule(L, schedule_type, beta_min, beta_max).to(device)
        self.betas = betas

        # alpha_i = 1 - beta_i
        alphas = 1.0 - betas
        self.alphas = alphas

        # bar_alpha_i = prod_{j=1}^{i} alpha_j   (cumulative product)
        alphas_bar = torch.cumprod(alphas, dim=0)
        self.alphas_bar = alphas_bar

        # bar_alpha_{i-1}, with bar_alpha_0 := 1   (needed for posterior)
        alphas_bar_prev = torch.cat(
            [torch.ones(1, device=device, dtype=torch.float32), alphas_bar[:-1]]
        )
        self.alphas_bar_prev = alphas_bar_prev

        # ------------------------------------------------------------------ #
        #  Forward process scalars  -- eq. (2) in the appendix               #
        #  q(x_i | x_0) = N( sqrt(bar_alpha_i)*x_0, (1-bar_alpha_i)*I )     #
        # ------------------------------------------------------------------ #

        self.sqrt_alphas_bar = torch.sqrt(alphas_bar)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1.0 - alphas_bar)

        # ------------------------------------------------------------------ #
        #  Posterior variance  -- eq. (6)                                     #
        #  tilde_beta_i = (1 - bar_alpha_{i-1}) / (1 - bar_alpha_i) * beta_i #
        #  Note: tilde_beta_1 = 0 (no noise injected at the final step)      #
        # ------------------------------------------------------------------ #

        posterior_variance = (1.0 - alphas_bar_prev) / (1.0 - alphas_bar) * betas
        self.posterior_variance = posterior_variance
        # Clamped log for numerical safety
        self.log_posterior_variance = torch.log(
            torch.clamp(posterior_variance, min=1e-20)
        )

        # ------------------------------------------------------------------ #
        #  Posterior mean coefficients  -- eq. (7)                           #
        #  tilde_mu_i = coef_x0 * x_0  +  coef_xt * x_t                     #
        # ------------------------------------------------------------------ #

        self.posterior_mean_coef_x0 = (
            torch.sqrt(alphas_bar_prev) * betas / (1.0 - alphas_bar)
        )
        self.posterior_mean_coef_xt = (
            torch.sqrt(alphas) * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar)
        )

        # ------------------------------------------------------------------ #
        #  Helpers for the reverse-step mean  -- eq. (8)                     #
        # ------------------------------------------------------------------ #

        # 1 / sqrt(alpha_i)
        self.sqrt_recip_alphas = torch.rsqrt(alphas)

        # beta_i / sqrt(1 - bar_alpha_i)  (coefficient of epsilon in eq. 8)
        self.betas_over_sqrt_one_minus_alphas_bar = (
            betas / self.sqrt_one_minus_alphas_bar
        )

        # ------------------------------------------------------------------ #
        #  Signal-to-noise ratio  SNR(i) = bar_alpha_i / (1 - bar_alpha_i)   #
        # ------------------------------------------------------------------ #

        self.snr = alphas_bar / (1.0 - alphas_bar)

    # ---------------------------------------------------------------------- #

    def _extract(self, tensor, t_idx, x_shape):
        """
        Extract values at (0-indexed) positions t_idx from a 1D tensor,
        then reshape for broadcasting against a tensor of shape x_shape.

        Args:
            tensor  : 1D tensor of shape (L,)
            t_idx   : 0-indexed batch of timesteps, shape (B,)
            x_shape : shape of the image tensor (B, C, H, W)

        Returns:
            tensor of shape (B, 1, 1, ...) with len(x_shape)-1 trailing dims
        """
        vals = tensor[t_idx]  # (B,)
        trailing = [1] * (len(x_shape) - 1)
        return vals.reshape(t_idx.shape[0], *trailing)

    def to(self, device):
        """Move every tensor to `device` in-place and return self."""
        self.device = device
        attrs = [
            "betas",
            "alphas",
            "alphas_bar",
            "alphas_bar_prev",
            "sqrt_alphas_bar",
            "sqrt_one_minus_alphas_bar",
            "posterior_variance",
            "log_posterior_variance",
            "posterior_mean_coef_x0",
            "posterior_mean_coef_xt",
            "sqrt_recip_alphas",
            "betas_over_sqrt_one_minus_alphas_bar",
            "snr",
        ]
        for attr in attrs:
            setattr(self, attr, getattr(self, attr).to(device))
        return self
