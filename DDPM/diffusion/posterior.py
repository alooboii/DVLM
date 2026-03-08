# diffusion/posterior.py
"""
True forward posterior and reverse-step mechanics.

The true posterior is (eq. 5-7):
  q(x_{i-1} | x_i, x_0) = N( tilde_mu_i(x_i, x_0), tilde_beta_i * I )

where
  tilde_beta_i   = (1 - bar_alpha_{i-1}) / (1 - bar_alpha_i) * beta_i    (eq. 6)
  tilde_mu_i     = coef_x0 * x_0  +  coef_xt * x_t                       (eq. 7)

The reverse model uses epsilon-parameterisation (eq. 8):
  mu_theta(x_t, t) = (1/sqrt(alpha_t)) * ( x_t - beta_t/sqrt(1-bar_alpha_t) * eps_hat )
"""

import torch


# --------------------------------------------------------------------------- #
#  True posterior                                                              #
# --------------------------------------------------------------------------- #


def q_posterior_mean_var(schedule, x0, x_t, t):
    """
    Compute the true posterior mean and variance:
        q(x_{t-1} | x_t, x_0)  =  N( tilde_mu_t, tilde_beta_t * I )

    Args:
        schedule: DiffusionSchedule
        x0      : clean data (B, C, H, W)
        x_t     : noised data at step t (B, C, H, W)
        t       : 1-indexed timesteps (B,)

    Returns:
        posterior_mean: tilde_mu_t,  shape (B, C, H, W)
        posterior_var : tilde_beta_t, shape (B, 1, 1, 1)   (scalar per sample)
    """
    t_idx = t - 1

    coef_x0 = schedule._extract(schedule.posterior_mean_coef_x0, t_idx, x0.shape)
    coef_xt = schedule._extract(schedule.posterior_mean_coef_xt, t_idx, x_t.shape)

    posterior_mean = coef_x0 * x0 + coef_xt * x_t
    posterior_var = schedule._extract(schedule.posterior_variance, t_idx, x_t.shape)

    return posterior_mean, posterior_var


# --------------------------------------------------------------------------- #
#  Epsilon-parameterised reverse step                                          #
# --------------------------------------------------------------------------- #


def predict_x0_from_eps(schedule, x_t, t, eps_hat):
    """
    Invert the forward reparameterisation to recover x_hat_0 from predicted noise.

        x_hat_0 = ( x_t - sqrt(1 - bar_alpha_t) * eps_hat ) / sqrt(bar_alpha_t)

    Args:
        schedule: DiffusionSchedule
        x_t     : noised image (B, C, H, W)
        t       : 1-indexed timesteps (B,)
        eps_hat : predicted noise from eps_theta network (B, C, H, W)

    Returns:
        x0_hat: predicted clean image (B, C, H, W)
    """
    t_idx = t - 1

    sqrt_ab = schedule._extract(schedule.sqrt_alphas_bar, t_idx, x_t.shape)
    sqrt_1mab = schedule._extract(schedule.sqrt_one_minus_alphas_bar, t_idx, x_t.shape)

    x0_hat = (x_t - sqrt_1mab * eps_hat) / sqrt_ab
    return x0_hat


def p_mean_from_eps(schedule, x_t, t, eps_hat):
    """
    Compute the reverse process mean mu_theta(x_t, t) using the
    epsilon-parameterisation (eq. 8).

    We first recover x_hat_0 from eps_hat, clamp it to [-1, 1] for stability,
    then compute the posterior mean with that estimate.  This is equivalent to
    eq. (8) but routes through the posterior formula, making the connection to
    eq. (7) explicit.

    Args:
        schedule: DiffusionSchedule
        x_t     : noised image (B, C, H, W)
        t       : 1-indexed timesteps (B,)
        eps_hat : network noise prediction (B, C, H, W)

    Returns:
        mean    : mu_theta(x_t, t)
        var     : tilde_beta_t (same convention as posterior)
        x0_hat  : estimated clean image (for inspection / trajectory saving)
    """
    x0_hat = predict_x0_from_eps(schedule, x_t, t, eps_hat)
    x0_hat = torch.clamp(x0_hat, -1.0, 1.0)  # numerical stability

    mean, var = q_posterior_mean_var(schedule, x0_hat, x_t, t)
    return mean, var, x0_hat


def p_sample_step(schedule, x_t, t, eps_hat):
    """
    One ancestral sampling step: sample x_{t-1} ~ p_theta(x_{t-1} | x_t).

        x_{t-1} = mu_theta(x_t, t) + sqrt(tilde_beta_t) * z,   z~N(0,I),  t > 1
        x_{0}   = mu_theta(x_1, 1)                                          t = 1

    Args:
        schedule: DiffusionSchedule
        x_t     : current noised image (B, C, H, W)
        t       : 1-indexed timesteps (B,)   -- may be a mixed batch
        eps_hat : network noise prediction (B, C, H, W)

    Returns:
        x_prev  : sample from p_theta(x_{t-1} | x_t)
        x0_hat  : estimated clean image (for trajectory visualisation)
    """
    mean, var, x0_hat = p_mean_from_eps(schedule, x_t, t, eps_hat)

    noise = torch.randn_like(x_t)

    # Mask noise: do NOT add noise at the very last denoising step (t == 1)
    # Shape: (B,) -> broadcast to (B, 1, 1, 1)
    mask = (t > 1).float()
    for _ in range(len(x_t.shape) - 1):
        mask = mask.unsqueeze(-1)

    x_prev = mean + mask * torch.sqrt(var) * noise
    return x_prev, x0_hat
