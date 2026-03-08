# eval.py
"""
Evaluation script for a trained DDPM checkpoint.

Computes:
  - Final sample grid (qualitative)
  - Denoising trajectory visualisation
  - Dataset-FID and Dataset-KID (using a LeNet feature extractor trained on real data)
  - Classifier accuracy + class-distribution entropy on generated samples
  - Nearest-neighbour memorisation check (pixel space)
  - ELBO / bits-per-dimension estimate on the test set

Example:
  python eval.py \
      --checkpoint ./output/checkpoint_step100000.pt \
      --dataset fashionmnist \
      --save_dir ./output/eval \
      --n_samples 10000
"""

import os
import argparse
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from torchvision.utils import make_grid, save_image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diffusion.schedule  import DiffusionSchedule
from diffusion.forward   import q_sample
from diffusion.posterior import q_posterior_mean_var, predict_x0_from_eps
from diffusion.ddpm      import ddpm_sample, save_sample_grid, save_trajectory_grid, unnormalize
from models.unet         import UNet
from models.classifier   import LeNet, train_classifier


# --------------------------------------------------------------------------- #
#  Arguments                                                                   #
# --------------------------------------------------------------------------- #

def get_args():
    p = argparse.ArgumentParser(description="Evaluate a DDPM checkpoint")
    p.add_argument("--checkpoint",   type=str, required=True)
    p.add_argument("--dataset",      type=str, default="fashionmnist",
                   choices=["mnist", "fashionmnist", "cifar10"])
    p.add_argument("--data_root",    type=str, default="./data")
    p.add_argument("--save_dir",     type=str, default="./eval_output")
    p.add_argument("--n_samples",    type=int, default=10000,
                   help="Generated samples for quantitative metrics")
    p.add_argument("--n_real",       type=int, default=10000,
                   help="Real test samples for comparison")
    p.add_argument("--batch_size",   type=int, default=128)
    p.add_argument("--seed",         type=int, default=0)
    p.add_argument("--device",       type=str, default="")
    p.add_argument("--skip_bpd",     action="store_true",
                   help="Skip the ELBO/bpd computation (slow)")
    p.add_argument("--n_bpd",        type=int, default=1000,
                   help="Number of test samples for bpd estimate")
    return p.parse_args()


# --------------------------------------------------------------------------- #
#  Dataset                                                                     #
# --------------------------------------------------------------------------- #

def load_test_dataset(name, root):
    tf_gray = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    tf_rgb  = T.Compose([T.ToTensor(), T.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

    if name == "mnist":
        ds = torchvision.datasets.MNIST(root, train=False, download=True, transform=tf_gray)
        return ds, 1, 28, 10
    elif name == "fashionmnist":
        ds = torchvision.datasets.FashionMNIST(root, train=False, download=True, transform=tf_gray)
        return ds, 1, 28, 10
    elif name == "cifar10":
        ds = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=tf_rgb)
        return ds, 3, 32, 10

def load_train_dataset(name, root):
    tf_gray = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    tf_rgb  = T.Compose([T.ToTensor(), T.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    if name == "mnist":
        return torchvision.datasets.MNIST(root, train=True, download=True, transform=tf_gray)
    elif name == "fashionmnist":
        return torchvision.datasets.FashionMNIST(root, train=True, download=True, transform=tf_gray)
    elif name == "cifar10":
        return torchvision.datasets.CIFAR10(root, train=True, download=True, transform=tf_rgb)


# --------------------------------------------------------------------------- #
#  FID / KID helpers                                                           #
# --------------------------------------------------------------------------- #

def compute_fid(mu_g, sigma_g, mu_r, sigma_r):
    """
    Frechet Inception Distance (in any feature space).
    FID = ||mu_g - mu_r||^2 + Tr( Sigma_g + Sigma_r - 2*(Sigma_g @ Sigma_r)^{1/2} )
    """
    diff = mu_g - mu_r
    term1 = diff @ diff

    # Matrix sqrt via eigendecomposition (stable for small d)
    M = sigma_g @ sigma_r
    vals, vecs = torch.linalg.eigh(M.double())
    vals = torch.clamp(vals, min=0)                   # numerical: remove tiny negatives
    sqrt_M = (vecs * vals.sqrt().unsqueeze(0)) @ vecs.T

    term2 = torch.trace(sigma_g + sigma_r - 2.0 * sqrt_M.float())
    return (term1 + term2).item()


def compute_kid(feats_gen, feats_real, degree=3):
    """
    Kernel Inception Distance with polynomial kernel k(x,y) = (x·y/d + 1)^degree.
    KID is an unbiased estimator -- more robust than FID for small sample counts.
    """
    n_g = feats_gen.shape[0]
    n_r = feats_real.shape[0]
    d   = feats_gen.shape[1]

    K_gg = ((feats_gen @ feats_gen.T) / d + 1) ** degree
    K_rr = ((feats_real @ feats_real.T) / d + 1) ** degree
    K_gr = ((feats_gen @ feats_real.T) / d + 1) ** degree

    # Unbiased diagonal removal
    kid = (
        (K_gg.sum() - K_gg.diag().sum()) / (n_g * (n_g - 1))
        + (K_rr.sum() - K_rr.diag().sum()) / (n_r * (n_r - 1))
        - 2 * K_gr.mean()
    )
    return kid.item()


@torch.no_grad()
def extract_features(model, loader, device, n_max=None):
    """Extract feature vectors from a trained LeNet."""
    all_feats  = []
    all_labels = []
    total      = 0
    model.eval()
    for x, y in loader:
        x = x.to(device)
        feat = model.extract_features(x).cpu()
        all_feats.append(feat)
        all_labels.append(y)
        total += x.size(0)
        if n_max and total >= n_max:
            break
    return torch.cat(all_feats), torch.cat(all_labels)


@torch.no_grad()
def extract_generated_features(model_gen, schedule, classifier, shape_per_batch,
                                n_total, device, batch_size=128):
    """Generate samples in batches and extract features."""
    all_feats  = []
    all_logits = []
    generated  = 0
    in_ch, H, W = shape_per_batch

    classifier.eval()
    while generated < n_total:
        bs = min(batch_size, n_total - generated)
        samples, _ = ddpm_sample(
            model_gen, schedule, (bs, in_ch, H, W), device, verbose=False
        )
        samples = samples.to(device)
        feat   = classifier.extract_features(samples).cpu()
        logit  = classifier(samples).cpu()
        all_feats.append(feat)
        all_logits.append(logit)
        generated += bs
        print(f"  Generated {generated}/{n_total}", end="\r")

    print()
    return torch.cat(all_feats), torch.cat(all_logits)


# --------------------------------------------------------------------------- #
#  ELBO / bits-per-dimension                                                   #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def estimate_elbo_bpd(model, schedule, dataset, device, n_samples=1000, batch_size=64):
    """
    Estimate the ELBO upper bound on NLL, reported in bits-per-dimension.

    For each x_0 in the test set:
      ELBO(x_0) = L_prior  +  L_recon  +  sum_i L_diffusion_i

    L_prior   = KL( q(x_L | x_0) || N(0,I) )   -- closed form
    L_diffusion_i = KL( q(x_{i-1}|x_i,x_0) || p_theta(x_{i-1}|x_i) )  -- Gaussian KL

    We use a single-sample Monte Carlo estimate for the diffusion terms by
    evaluating at one randomly chosen i per data point (unbiased estimator
    of the sum when multiplied by L).

    See Section 5.1.4 of the PA1 appendix and Appendix A of the DDPM paper.

    Returns:
        bpd: average bits-per-dimension on the sampled test set
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    model.eval()

    L   = schedule.L
    D   = None    # data dimensionality, set on first batch
    log2e = math.log2(math.e)

    total_elbo = 0.0
    n_processed = 0

    for x0, _ in loader:
        if n_processed >= n_samples:
            break
        x0 = x0.to(device)
        B  = x0.size(0)
        if D is None:
            D = x0[0].numel()

        # ── L_prior = KL( q(x_L|x_0) || N(0,I) ) ─────────────────────────
        # q(x_L|x_0) = N( sqrt(bar_alpha_L)*x0, (1-bar_alpha_L)*I )
        # KL( N(mu, sigma^2 I) || N(0,I) ) = 0.5 * (sigma^2 + ||mu||^2 - d - log(sigma^2))
        ab_L   = schedule.alphas_bar[-1]                         # scalar
        mu_L   = math.sqrt(ab_L.item()) * x0                    # (B, C, H, W)
        var_L  = (1.0 - ab_L).item()

        kl_prior = 0.5 * (
            var_L * D
            + (mu_L ** 2).sum(dim=(1, 2, 3))        # (B,)
            - D
            - D * math.log(var_L)
        )                                              # (B,)

        # ── L_diffusion (Monte Carlo over i) ──────────────────────────────
        # Sample one timestep i uniformly from {2, ..., L}
        i_rand  = torch.randint(2, L + 1, (B,), device=device)   # 1-indexed
        i_idx   = i_rand - 1

        # Sample x_{i_rand} via forward process
        eps     = torch.randn_like(x0)
        x_i, _ = q_sample(schedule, x0, i_rand, eps=eps)

        # True posterior: q(x_{i-1} | x_i, x_0)
        mu_q, var_q = q_posterior_mean_var(schedule, x0, x_i, i_rand)

        # Reverse model: p_theta(x_{i-1} | x_i)  -- Gaussian with same variance
        eps_hat = model(x_i, i_rand)
        x0_hat  = torch.clamp(predict_x0_from_eps(schedule, x_i, i_rand, eps_hat), -1.0, 1.0)
        mu_p, _ = q_posterior_mean_var(schedule, x0_hat, x_i, i_rand)

        # KL between two Gaussians with same variance tilde_beta_i:
        # KL = ||mu_q - mu_p||^2 / (2 * var_q)
        var_q_scalar = var_q.squeeze()            # (B,)

        kl_i = 0.5 * (
            ((mu_q - mu_p) ** 2).sum(dim=(1, 2, 3)) / var_q_scalar.clamp(min=1e-8)
        )                                          # (B,)

        # Scale to estimate sum over i=2..L   (L-1 terms, unbiased when MC'd)
        kl_diffusion = kl_i * (L - 1)

        # ── L_recon (i=1 term, simplified) ────────────────────────────────
        # Use same eps-prediction; treat as MSE in x0 space scaled by 1/(2*var_1)
        t1       = torch.ones(B, device=device, dtype=torch.long)
        x_1, _  = q_sample(schedule, x0, t1, eps=torch.randn_like(x0))
        eps_1   = model(x_1, t1)
        x0_hat1 = predict_x0_from_eps(schedule, x_1, t1, eps_1)
        var_1   = schedule.posterior_variance[0].item()    # tilde_beta_1 ≈ 0 (use raw beta_1 instead)
        var_1   = max(schedule.betas[0].item(), 1e-8)

        l_recon = 0.5 * ((x0 - x0_hat1) ** 2).sum(dim=(1, 2, 3)) / var_1  # (B,)

        # ── Total ELBO ────────────────────────────────────────────────────
        elbo_batch = kl_prior + kl_diffusion + l_recon    # (B,)
        total_elbo += elbo_batch.sum().item()
        n_processed += B

    avg_nll_nats = total_elbo / n_processed
    bpd          = avg_nll_nats / D * log2e             # convert to bits/dim
    return bpd


# --------------------------------------------------------------------------- #
#  Nearest-neighbour memorisation check                                        #
# --------------------------------------------------------------------------- #

def nearest_neighbor_check(gen_samples, train_dataset, n_check=16,
                            save_path="nn_check.png", value_range=(-1.0, 1.0)):
    """
    For each of `n_check` generated samples, find the nearest training image
    in pixel (L2) space and save side-by-side.
    """
    gen = gen_samples[:n_check]                   # (n_check, C, H, W)
    gen_flat = gen.reshape(n_check, -1).float()   # (n_check, D)

    # Build a small training cache (flat)
    cache_x = []
    loader  = DataLoader(train_dataset, batch_size=512, shuffle=False)
    for x, _ in loader:
        cache_x.append(x)
        if sum(b.shape[0] for b in cache_x) >= 5000:
            break
    cache_x = torch.cat(cache_x)[:5000]
    cache_flat = cache_x.reshape(cache_x.shape[0], -1).float()

    # L2 distances
    dists = torch.cdist(gen_flat, cache_flat)     # (n_check, 5000)
    nn_idx = dists.argmin(dim=1)                  # (n_check,)
    nn_x   = cache_x[nn_idx]                      # (n_check, C, H, W)

    # Interleave: gen, nn, gen, nn, ...
    rows = []
    for i in range(n_check):
        rows.append(unnormalize(gen[i:i+1],  value_range))
        rows.append(unnormalize(nn_x[i:i+1], value_range))
    grid = make_grid(torch.cat(rows), nrow=2)
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    save_image(grid, save_path)
    print(f"Saved nearest-neighbour check -> {save_path}")


# --------------------------------------------------------------------------- #
#  Main eval                                                                   #
# --------------------------------------------------------------------------- #

def main():
    args   = get_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    torch.manual_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    # ── Load checkpoint ───────────────────────────────────────────────────
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    train_args = argparse.Namespace(**ckpt["args"])

    # ── Rebuild schedule + model ──────────────────────────────────────────
    schedule = DiffusionSchedule(
        L=train_args.L,
        schedule_type=train_args.schedule,
        beta_min=train_args.beta_min,
        beta_max=train_args.beta_max,
        device=device,
    )

    ch_mults = tuple(int(x) for x in train_args.ch_mults.split(","))

    test_ds, in_ch, img_size, num_classes = load_test_dataset(args.dataset, args.data_root)
    train_ds = load_train_dataset(args.dataset, args.data_root)

    model = UNet(
        in_ch=in_ch,
        base_ch=train_args.base_ch,
        ch_mults=ch_mults,
        num_res_blocks=train_args.num_res_blocks,
        t_emb_dim=train_args.t_emb_dim,
        dropout=0.0,              # no dropout at eval
        use_attention=(not getattr(train_args, "no_attention", False)),
    ).to(device)

    # Load EMA weights from checkpoint (they live inside the ema dict)
    ema_state = ckpt["ema"]["shadow"]
    state_dict = {k: v for k, v in ema_state.items()}
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded (EMA weights).")

    # ── 1. Qualitative sample grid ─────────────────────────────────────────
    print("\n[1] Generating sample grid...")
    n_grid  = min(64, args.n_samples)
    samples, trajectory = ddpm_sample(
        model, schedule, (n_grid, in_ch, img_size, img_size), device,
        save_trajectory=True, verbose=True,
    )
    save_sample_grid(samples, os.path.join(args.save_dir, "samples.png"))
    save_trajectory_grid(trajectory, os.path.join(args.save_dir, "trajectory.png"))

    # ── 2. Train a LeNet feature extractor ─────────────────────────────────
    print("\n[2] Training LeNet feature extractor on real data...")
    classifier = train_classifier(
        train_ds, num_classes=num_classes, epochs=10, device=device
    )
    cls_path = os.path.join(args.save_dir, "classifier.pt")
    torch.save(classifier.state_dict(), cls_path)

    # Extract real features from test set
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    print("Extracting features from real test data...")
    real_feats, real_labels = extract_features(classifier, test_loader, device, n_max=args.n_real)
    print(f"  Real features: {real_feats.shape}")

    # ── 3. Generate samples for quantitative metrics ───────────────────────
    print(f"\n[3] Generating {args.n_samples} samples for metrics...")
    gen_feats, gen_logits = extract_generated_features(
        model, schedule, classifier,
        shape_per_batch=(in_ch, img_size, img_size),
        n_total=args.n_samples,
        device=device,
        batch_size=args.batch_size,
    )
    print(f"  Generated features: {gen_feats.shape}")

    # Save a larger grid of generated samples
    all_gen_path = os.path.join(args.save_dir, "generated_samples_large.png")
    gen_samples_display, _ = ddpm_sample(
        model, schedule, (min(args.n_samples, 256), in_ch, img_size, img_size),
        device, verbose=False
    )
    save_sample_grid(gen_samples_display, all_gen_path, nrow=16)

    # ── 4. Dataset-FID ─────────────────────────────────────────────────────
    print("\n[4] Computing Dataset-FID...")
    real_feats_f = real_feats.float()
    gen_feats_f  = gen_feats.float()

    mu_r  = real_feats_f.mean(0)
    sig_r = torch.cov(real_feats_f.T)
    mu_g  = gen_feats_f.mean(0)
    sig_g = torch.cov(gen_feats_f.T)

    fid = compute_fid(mu_g, sig_g, mu_r, sig_r)
    print(f"  Dataset-FID  = {fid:.3f}")

    # ── 5. Dataset-KID ─────────────────────────────────────────────────────
    print("\n[5] Computing Dataset-KID...")
    # Subsample for efficiency
    n_kid = min(5000, gen_feats_f.shape[0], real_feats_f.shape[0])
    kid = compute_kid(gen_feats_f[:n_kid], real_feats_f[:n_kid])
    print(f"  Dataset-KID  = {kid:.5f}")

    # ── 6. Classifier accuracy + entropy ───────────────────────────────────
    print("\n[6] Classifier-based quality and diversity...")
    probs   = torch.softmax(gen_logits, dim=-1)            # (N, K)
    pred_y  = probs.argmax(dim=1)

    # Compute test-set accuracy as a sanity reference
    real_logits_all, real_labels_all = [], []
    classifier.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            real_logits_all.append(classifier(x).cpu())
            real_labels_all.append(y)
    real_logits_cat = torch.cat(real_logits_all)
    real_labels_cat = torch.cat(real_labels_all)
    real_acc = (real_logits_cat.argmax(1) == real_labels_cat).float().mean().item()
    print(f"  Classifier accuracy on real test data: {real_acc:.4f}")

    # Accuracy on generated samples (how well do they "look" like valid digits?)
    # (Uses the predicted class histogram; there is no ground truth for generated data.)
    # Instead report class distribution entropy
    class_hist = torch.zeros(num_classes)
    for c in range(num_classes):
        class_hist[c] = (pred_y == c).float().mean()
    entropy = -(class_hist * torch.log(class_hist.clamp(min=1e-8))).sum().item()
    max_entropy = math.log(num_classes)
    print(f"  Class distribution entropy on generated samples: "
          f"{entropy:.4f}  (max = {max_entropy:.4f}, "
          f"higher => better diversity)")
    print(f"  Class histogram: {class_hist.numpy().round(3)}")

    # ── 7. Nearest-neighbour memorisation check ─────────────────────────────
    print("\n[7] Nearest-neighbour memorisation check...")
    gen_for_nn  = gen_samples_display[:16].cpu()
    nearest_neighbor_check(
        gen_for_nn, train_ds,
        n_check=16,
        save_path=os.path.join(args.save_dir, "nn_check.png"),
    )

    # ── 8. ELBO / bits-per-dimension ───────────────────────────────────────
    if not args.skip_bpd:
        print(f"\n[8] Estimating ELBO/bpd on {args.n_bpd} test samples...")
        bpd = estimate_elbo_bpd(
            model, schedule, test_ds, device,
            n_samples=args.n_bpd,
            batch_size=args.batch_size,
        )
        print(f"  Estimated bpd = {bpd:.4f} bits/dim")
    else:
        bpd = float("nan")
        print("  [Skipped bpd computation]")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n══════════════ Evaluation Summary ══════════════")
    print(f"  Dataset-FID       : {fid:.3f}")
    print(f"  Dataset-KID       : {kid:.5f}")
    print(f"  Class entropy     : {entropy:.4f} / {max_entropy:.4f}")
    print(f"  Classifier acc    : {real_acc:.4f}  (on real data, reference)")
    print(f"  BPD (ELBO bound)  : {bpd:.4f}")
    print("═══════════════════════════════════════════════")

    # Save summary as text
    with open(os.path.join(args.save_dir, "results.txt"), "w") as f:
        f.write(f"checkpoint: {args.checkpoint}\n")
        f.write(f"dataset:    {args.dataset}\n")
        f.write(f"n_samples:  {args.n_samples}\n")
        f.write(f"Dataset-FID: {fid:.4f}\n")
        f.write(f"Dataset-KID: {kid:.6f}\n")
        f.write(f"Class entropy: {entropy:.4f} / {max_entropy:.4f}\n")
        f.write(f"BPD (ELBO):  {bpd:.4f}\n")


if __name__ == "__main__":
    main() 