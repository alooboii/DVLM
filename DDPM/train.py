# train.py
"""
DDPM training script.

Example usage (Kaggle / command line):

  python train.py \
      --dataset fashionmnist \
      --steps 100000 \
      --batch_size 128 \
      --lr 2e-4 \
      --save_dir ./output \
      --sample_every 5000

Sanity-check mode (overfit on 256 images -- should work fast):

  python train.py --dataset mnist --sanity_check --steps 5000 --save_dir ./output_sanity
"""

import os
import math
import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diffusion.schedule import DiffusionSchedule
from diffusion.forward import q_sample, sample_timesteps
from diffusion.ddpm import ddpm_sample, save_sample_grid, save_trajectory_grid
from models.unet import UNet


# --------------------------------------------------------------------------- #
#  Argument parsing                                                            #
# --------------------------------------------------------------------------- #


def get_args():
    p = argparse.ArgumentParser(description="Train DDPM")

    # Dataset
    p.add_argument(
        "--dataset",
        type=str,
        default="fashionmnist",
        choices=["mnist", "fashionmnist", "cifar10"],
        help="Dataset to train on",
    )
    p.add_argument("--data_root", type=str, default="./data")

    # Diffusion schedule
    p.add_argument("--L", type=int, default=1000, help="Diffusion steps")
    p.add_argument(
        "--schedule", type=str, default="linear", choices=["linear", "cosine"]
    )
    p.add_argument("--beta_min", type=float, default=1e-4)
    p.add_argument("--beta_max", type=float, default=0.02)

    # Model
    p.add_argument("--base_ch", type=int, default=32)
    p.add_argument(
        "--ch_mults",
        type=str,
        default="1,2,4",
        help="Channel multipliers, comma-separated, e.g. '1,2,4'",
    )
    p.add_argument("--num_res_blocks", type=int, default=1)
    p.add_argument("--t_emb_dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--no_attention", action="store_true")

    # Training
    p.add_argument("--steps", type=int, default=100_000, help="Gradient steps")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument(
        "--grad_clip", type=float, default=1.0, help="Max gradient norm (0 = disabled)"
    )

    # Saving / logging
    p.add_argument("--save_dir", type=str, default="./output")
    p.add_argument(
        "--sample_every",
        type=int,
        default=5000,
        help="Sample and save a grid every N steps",
    )
    p.add_argument(
        "--ckpt_every", type=int, default=25000, help="Save checkpoint every N steps"
    )
    p.add_argument("--num_samples", type=int, default=64)
    p.add_argument("--nrow", type=int, default=8)

    # Sanity check
    p.add_argument(
        "--sanity_check",
        action="store_true",
        help="Overfit on 256 images (tests basic correctness)",
    )

    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="")
    p.add_argument(
        "--resume", type=str, default="", help="Path to checkpoint to resume from"
    )

    return p.parse_args()


# --------------------------------------------------------------------------- #
#  Dataset loading                                                             #
# --------------------------------------------------------------------------- #


def get_dataset(name, root, train=True):
    """
    Load a torchvision dataset with normalisation to [-1, 1].
    Returns (dataset, in_ch, image_size, num_classes).
    """
    transform_gray = T.Compose(
        [
            T.ToTensor(),  # [0, 1]
            T.Normalize((0.5,), (0.5,)),  # -> [-1, 1]
        ]
    )
    transform_rgb = T.Compose(
        [
            T.ToTensor(),
            T.RandomHorizontalFlip(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    if name == "mnist":
        ds = torchvision.datasets.MNIST(
            root, train=train, download=True, transform=transform_gray
        )
        return ds, 1, 28, 10
    elif name == "fashionmnist":
        ds = torchvision.datasets.FashionMNIST(
            root, train=train, download=True, transform=transform_gray
        )
        return ds, 1, 28, 10
    elif name == "cifar10":
        ds = torchvision.datasets.CIFAR10(
            root, train=train, download=True, transform=transform_rgb
        )
        return ds, 3, 32, 10
    else:
        raise ValueError(f"Unknown dataset: {name}")


# --------------------------------------------------------------------------- #
#  EMA                                                                        #
# --------------------------------------------------------------------------- #


class EMA:
    """Exponential moving average of model parameters."""

    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {
            name: param.data.clone() for name, param in model.named_parameters()
        }

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply_shadow(self, model):
        """Copy EMA weights into model (used at eval/sample time)."""
        for name, param in model.named_parameters():
            param.data.copy_(self.shadow[name])

    def restore(self, model, backup):
        """Restore original weights from backup dict."""
        for name, param in model.named_parameters():
            param.data.copy_(backup[name])

    def state_dict(self):
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state):
        self.decay = state["decay"]
        self.shadow = state["shadow"]


# --------------------------------------------------------------------------- #
#  Verification helpers                                                        #
# --------------------------------------------------------------------------- #


def verify_timestep_distribution(L, n=10_000, device="cpu"):
    """Check that timesteps are sampled uniformly (histogram should be flat)."""
    ts = sample_timesteps(n, L, device).cpu().numpy()
    counts, _ = np.histogram(ts, bins=10, range=(1, L + 1))
    expected = n / 10
    max_dev = max(abs(c - expected) / expected for c in counts)
    print(
        f"  [Sanity] Timestep histogram max deviation from uniform: {max_dev:.3f} "
        f"(should be < 0.05 for large n)"
    )


def verify_forward_moments(schedule, x0_sample, n_trials=1000):
    """
    Empirically check  E[x_t] ≈ sqrt(bar_alpha_t) * x0
    and                Var(x_t) ≈ 1 - bar_alpha_t
    at a mid-point timestep.
    """
    i = schedule.L // 2
    t = torch.tensor([i])
    t_idx = t - 1

    sqrt_ab = schedule.sqrt_alphas_bar[t_idx[0]].item()
    var_exp = (1 - schedule.alphas_bar[t_idx[0]]).item()

    samples = []
    x0 = x0_sample[:1]  # single image for reproducibility
    for _ in range(n_trials):
        x_t, _ = q_sample(schedule, x0, t)
        samples.append(x_t)
    samples = torch.cat(samples, dim=0)  # (n_trials, C, H, W)

    mean_emp = samples.mean(dim=0)
    var_emp = samples.var(dim=0).mean().item()

    mean_err = (mean_emp - sqrt_ab * x0[0]).abs().mean().item()
    print(
        f"  [Sanity] Forward process at t={i}: "
        f"mean error={mean_err:.4f} (expect ≈0), "
        f"var_emp={var_emp:.4f}, var_theory={var_exp:.4f}"
    )


# --------------------------------------------------------------------------- #
#  Plot helpers                                                                #
# --------------------------------------------------------------------------- #


def save_schedule_plots(schedule, save_dir):
    """Save plots of bar_alpha_i and SNR(i) against timestep."""
    i = np.arange(1, schedule.L + 1)
    ab = schedule.alphas_bar.cpu().numpy()
    snr = schedule.snr.cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(i, ab)
    axes[0].set_xlabel("Timestep i")
    axes[0].set_ylabel("bar_alpha_i")
    axes[0].set_title("Cumulative product of (1 - beta_i)")

    axes[1].semilogy(i, snr)
    axes[1].set_xlabel("Timestep i")
    axes[1].set_ylabel("SNR(i) = bar_alpha_i / (1 - bar_alpha_i)  [log scale]")
    axes[1].set_title("Signal-to-Noise Ratio")

    plt.tight_layout()
    path = os.path.join(save_dir, "schedule.png")
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"Saved schedule plot -> {path}")


def save_loss_plot(losses, save_dir):
    """Save a smoothed training loss curve."""
    fig, ax = plt.subplots(figsize=(8, 4))
    # Simple moving average for readability
    window = max(1, len(losses) // 200)
    kernel = np.ones(window) / window
    smoothed = np.convolve(losses, kernel, mode="valid")
    ax.plot(smoothed, linewidth=0.8)
    ax.set_xlabel("Gradient step")
    ax.set_ylabel("L_simple (MSE)")
    ax.set_title("Training loss")
    plt.tight_layout()
    path = os.path.join(save_dir, "loss.png")
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"Saved loss plot -> {path}")


# --------------------------------------------------------------------------- #
#  Main training loop                                                          #
# --------------------------------------------------------------------------- #


def train(args):
    # ── Reproducibility ───────────────────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "samples"), exist_ok=True)

    # ── Dataset ───────────────────────────────────────────────────────────
    train_ds, in_ch, img_size, num_classes = get_dataset(args.dataset, args.data_root)
    print(f"Dataset: {args.dataset}  |  in_ch={in_ch}  img={img_size}x{img_size}")

    if args.sanity_check:
        # Overfit on 256 random images to test the pipeline
        idx = random.sample(range(len(train_ds)), 256)
        train_ds = Subset(train_ds, idx)
        print("Sanity-check mode: using only 256 training images.")

    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # Print some real images as a reference baseline
    real_iter = iter(loader)
    real_images = next(real_iter)[0][: args.num_samples]
    save_sample_grid(
        real_images, os.path.join(args.save_dir, "real_samples.png"), nrow=args.nrow
    )
    print("Saved real sample grid.")

    # ── Diffusion schedule ────────────────────────────────────────────────
    schedule = DiffusionSchedule(
        L=args.L,
        schedule_type=args.schedule,
        beta_min=args.beta_min,
        beta_max=args.beta_max,
        device=device,
    )
    save_schedule_plots(schedule, args.save_dir)

    # ── Model ─────────────────────────────────────────────────────────────
    ch_mults = tuple(int(x) for x in args.ch_mults.split(","))
    model = UNet(
        in_ch=in_ch,
        base_ch=args.base_ch,
        ch_mults=ch_mults,
        num_res_blocks=args.num_res_blocks,
        t_emb_dim=args.t_emb_dim,
        dropout=args.dropout,
        use_attention=(not args.no_attention),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    ema = EMA(model, decay=args.ema_decay)

    start_step = 0
    losses = []

    # ── Optional resume ──────────────────────────────────────────────────
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        ema.load_state_dict(ckpt["ema"])
        start_step = ckpt["step"]
        losses = ckpt.get("losses", [])
        print(f"Resumed from step {start_step}  ({args.resume})")

    # ── Pre-training sanity checks ────────────────────────────────────────
    print("\n── Pre-training sanity checks ──")
    verify_timestep_distribution(args.L, device=device)

    # Grab one batch for moment check
    x0_ref, _ = next(iter(loader))
    x0_ref = x0_ref.to(device)
    verify_forward_moments(schedule, x0_ref)
    print()

    # ── Training loop ─────────────────────────────────────────────────────
    data_iter = iter(loader)
    step = start_step

    print(f"Training for {args.steps} steps (starting at {step})...")

    model.train()
    while step < args.steps:
        # Cycle through data
        try:
            x0, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x0, _ = next(data_iter)

        x0 = x0.to(device)
        B = x0.size(0)

        # ── Core training iteration (Algorithm 1) ─────────────────────────
        # 1. Sample timesteps uniformly
        t = sample_timesteps(B, args.L, device)

        # 2. Sample noise
        eps = torch.randn_like(x0)

        # 3. Compute x_t via closed-form forward process  (eq. 2)
        x_t, eps = q_sample(schedule, x0, t, eps=eps)

        # 4. Predict noise
        eps_hat = model(x_t, t)

        # 5. L_simple = E[ || eps - eps_hat ||^2 ]   (eq. 9 / eq. 14)
        loss = F.mse_loss(eps_hat, eps)

        # 6. Optimise
        optimizer.zero_grad()
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        ema.update(model)

        losses.append(loss.item())
        step += 1

        # ── Logging ───────────────────────────────────────────────────────
        if step % 500 == 0 or step == 1:
            recent = np.mean(losses[-500:]) if len(losses) >= 500 else np.mean(losses)
            print(f"  step={step:>7d}/{args.steps}  loss={recent:.5f}")

        # ── Periodic sampling ─────────────────────────────────────────────
        if step % args.sample_every == 0 or step == args.steps:
            print(f"  Generating samples at step {step}...")

            # Swap in EMA weights, sample, restore
            backup = {n: p.data.clone() for n, p in model.named_parameters()}
            ema.apply_shadow(model)

            shape = (args.num_samples, in_ch, img_size, img_size)
            samples, traj = ddpm_sample(
                model,
                schedule,
                shape,
                device,
                save_trajectory=True,
                verbose=False,
            )

            save_sample_grid(
                samples,
                os.path.join(args.save_dir, "samples", f"step_{step:07d}.png"),
                nrow=args.nrow,
            )

            # Denoising trajectory for one sample
            save_trajectory_grid(
                traj,
                os.path.join(args.save_dir, "samples", f"trajectory_{step:07d}.png"),
            )

            ema.restore(model, backup)
            model.train()

        # ── Checkpointing ─────────────────────────────────────────────────
        if step % args.ckpt_every == 0 or step == args.steps:
            ckpt_path = os.path.join(args.save_dir, f"checkpoint_step{step}.pt")
            torch.save(
                {
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "ema": ema.state_dict(),
                    "args": vars(args),
                    "losses": losses,
                },
                ckpt_path,
            )
            print(f"  Checkpoint saved -> {ckpt_path}")

    # ── Final plots ────────────────────────────────────────────────────────
    save_loss_plot(losses, args.save_dir)
    # Save loss values as numpy for later analysis
    np.save(os.path.join(args.save_dir, "losses.npy"), np.array(losses))
    print("Training complete.")


if __name__ == "__main__":
    args = get_args()
    train(args)
