# DDPM from First Principles

A clean, well-documented PyTorch implementation of **Denoising Diffusion Probabilistic Models** (Ho et al., 2020), built from scratch for learning purposes. Every component maps directly to the paper's equations — no black-box diffusion libraries.

> **Reference:** [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — Ho, Jain, Abbeel (NeurIPS 2020)

---

## Features

- Full DDPM forward process, posterior, and ancestral sampler
- Linear and cosine variance schedules
- U-Net with sinusoidal timestep embeddings, GroupNorm + SiLU, skip connections, and optional bottleneck self-attention
- Exponential Moving Average (EMA) of model weights
- Pre-training sanity checks (forward moment verification, timestep uniformity)
- Periodic sample grids and denoising trajectory visualisations during training
- Evaluation: Dataset-FID, Dataset-KID, classifier entropy, nearest-neighbour memorisation check, and ELBO/bits-per-dimension estimate
- Resume from checkpoint
- tqdm progress bars with live loss and gradient norm display

---

## Project Structure

```
.
├── diffusion/
│   ├── __init__.py
│   ├── schedule.py       # Variance schedule {beta_i} and all derived scalars
│   ├── forward.py        # Forward process q(x_t | x_0), timestep sampling
│   ├── posterior.py      # True posterior q(x_{t-1} | x_t, x_0), reverse step
│   └── ddpm.py           # Full ancestral sampler, image saving utilities
│
├── models/
│   ├── __init__.py
│   ├── unet.py           # U-Net noise predictor eps_theta(x_t, t)
│   └── classifier.py     # LeNet feature extractor for evaluation metrics
│
├── train.py              # Training script
└── eval.py               # Evaluation script
```

The separation is intentional — `diffusion/` contains only math (no neural net calls), `models/` contains only architecture (no diffusion math).

---

## Installation

```bash
pip install torch torchvision tqdm matplotlib numpy einops
```

Tested with Python 3.10+, PyTorch 2.x.

---

## Usage

### Training

```bash
python train.py \
    --dataset fashionmnist \
    --steps 100000 \
    --batch_size 128 \
    --lr 2e-4 \
    --no_attention \
    --save_dir ./output
```

**Sanity check first (recommended):** overfits on 256 images to verify the pipeline is correct before committing to a full run.

```bash
python train.py \
    --dataset fashionmnist \
    --sanity_check \
    --steps 5000 \
    --save_dir ./output_sanity
```

If samples start resembling the dataset after ~5k steps, the pipeline is working.

### Evaluation

```bash
python eval.py \
    --checkpoint ./output/checkpoint_step100000.pt \
    --dataset fashionmnist \
    --save_dir ./output/eval \
    --n_samples 10000
```

---

## Training Arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `fashionmnist` | `mnist`, `fashionmnist`, or `cifar10` |
| `--data_root` | `./data` | Dataset download location |
| `--L` | `1000` | Number of diffusion steps |
| `--schedule` | `linear` | `linear` or `cosine` variance schedule |
| `--beta_min` | `1e-4` | Lower bound of linear schedule |
| `--beta_max` | `0.02` | Upper bound of linear schedule |
| `--base_ch` | `32` | Base channel count for U-Net |
| `--ch_mults` | `1,2,4` | Channel multipliers per resolution level |
| `--num_res_blocks` | `1` | ResBlocks per encoder/decoder level |
| `--t_emb_dim` | `128` | Timestep embedding dimension |
| `--dropout` | `0.0` | Dropout probability in ResBlocks |
| `--no_attention` | `False` | Disable bottleneck self-attention |
| `--steps` | `100000` | Total gradient steps |
| `--batch_size` | `128` | Training batch size |
| `--lr` | `2e-4` | Adam learning rate |
| `--ema_decay` | `0.9999` | EMA decay factor |
| `--grad_clip` | `1.0` | Max gradient norm (0 = disabled) |
| `--sample_every` | `5000` | Generate sample grid every N steps |
| `--ckpt_every` | `25000` | Save checkpoint every N steps |
| `--num_samples` | `64` | Number of samples in periodic grids |
| `--resume` | `` | Path to checkpoint to resume from |
| `--sanity_check` | `False` | Overfit on 256 images |
| `--seed` | `42` | Random seed |

---

## Output Structure

After training, `save_dir` contains:

```
output/
├── real_samples.png          # Reference grid of real training images
├── schedule.png              # bar_alpha_i and SNR(i) plots
├── training_curves.png       # Smoothed loss and gradient norm
├── losses.npy                # Raw per-step loss values
├── grad_norms.npy            # Raw per-step gradient norms
├── checkpoint_step25000.pt   # Periodic checkpoints
├── checkpoint_step50000.pt
├── checkpoint_step100000.pt
└── samples/
    ├── step_0005000.png      # Sample grids at each sample_every interval
    ├── step_0010000.png
    ├── trajectory_0005000.png  # Denoising trajectory visualisation
    └── ...
```

After evaluation, `eval save_dir` contains:

```
eval/
├── samples.png                  # Final 64-sample grid
├── trajectory.png               # Denoising trajectory
├── generated_samples_large.png  # 256-sample grid
├── nn_check.png                 # Generated vs nearest training neighbour
├── classifier.pt                # Trained LeNet feature extractor
└── results.txt                  # Numeric summary of all metrics
```

---

## Architecture

### U-Net

The noise predictor $\epsilon_\theta(x_t, t)$ follows a standard encoder-decoder U-Net with:

- **Sinusoidal timestep embedding** → MLP projection → injected into every ResBlock as a spatial bias
- **Encoder:** repeated ResBlocks + strided 3×3 downsampling at each resolution level; channel count grows as `base_ch × ch_mults`
- **Bottleneck:** ResBlock → optional single-head self-attention → ResBlock
- **Decoder:** mirrors encoder with skip connections via channel-dim concatenation, nearest-neighbour upsampling + 3×3 conv
- **GroupNorm + SiLU** throughout
- **1×1 output conv** back to image channels

Default config for FashionMNIST: `base_ch=32`, `ch_mults=(1,2,4)` → channels `[32, 64, 128]`, ~1.5M parameters.

### Variance Schedule

The linear schedule from Ho et al. 2020:

$$\beta_i = \text{linspace}(\beta_{\min}, \beta_{\max}, L), \quad \beta_{\min} = 10^{-4},\ \beta_{\max} = 0.02$$

All derived quantities ($\bar\alpha_i$, $\tilde\beta_i$, posterior mean coefficients, SNR) are precomputed once and stored as tensors.

---

## Key Equations

**Forward process (closed-form marginal):**

$$q(x_t \mid x_0) = \mathcal{N}\!\left(\sqrt{\bar\alpha_t}\, x_0,\ (1 - \bar\alpha_t) I\right)$$

**Training objective** ($L_{\text{simple}}$):

$$\mathcal{L}_{\text{simple}}(\theta) = \mathbb{E}_{t, x_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$$

**Reverse step:**

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\!\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x_t, t)\right) + \sqrt{\tilde\beta_t}\, z, \quad z \sim \mathcal{N}(0, I)$$

**Signal-to-noise ratio:**

$$\text{SNR}(t) = \frac{\bar\alpha_t}{1 - \bar\alpha_t}$$

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| **Dataset-FID** | Fréchet distance between real and generated feature distributions (LeNet features) |
| **Dataset-KID** | Polynomial kernel MMD — unbiased, more stable than FID for small sample counts |
| **Class entropy** | Entropy of predicted class histogram on generated samples; higher = better diversity |
| **Nearest-neighbour check** | Each generated image paired with its closest training image in pixel space; detects memorisation |
| **BPD (ELBO bound)** | Bits-per-dimension estimated via Monte Carlo ELBO on test set |

ImageNet-Inception FID is intentionally avoided for MNIST-scale datasets due to domain mismatch — dataset-specific features are used instead.

---

## Pre-training Sanity Checks

Before training begins, the script automatically runs:

1. **Timestep uniformity:** samples 10k timesteps and checks the histogram is flat (max deviation from uniform should be < 5%)
2. **Forward moment verification:** empirically estimates $\mathbb{E}[x_t]$ and $\text{Var}(x_t)$ at $t = L/2$ and compares against the theoretical values $\sqrt{\bar\alpha_t}\, x_0$ and $1 - \bar\alpha_t$

If either check fails, there is a bug in the schedule or forward process — do not proceed to full training.

---

## Resuming Training

```bash
python train.py \
    --dataset fashionmnist \
    --steps 200000 \
    --resume ./output/checkpoint_step100000.pt \
    --save_dir ./output
```

The checkpoint stores model weights, EMA shadow weights, optimizer state, step count, loss history, and gradient norm history.

---

## References

```bibtex
@article{ho2020ddpm,
  title   = {Denoising Diffusion Probabilistic Models},
  author  = {Jonathan Ho and Ajay Jain and Pieter Abbeel},
  journal = {NeurIPS},
  year    = {2020},
  url     = {https://arxiv.org/abs/2006.11239}
}

@inproceedings{ronneberger2015unet,
  title     = {U-Net: Convolutional Networks for Biomedical Image Segmentation},
  author    = {Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
  booktitle = {MICCAI},
  year      = {2015},
  url       = {https://arxiv.org/abs/1505.04597}
}
```