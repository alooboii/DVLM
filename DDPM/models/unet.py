# models/unet.py
"""
Small U-Net  eps_theta(x_t, t)  for DDPM.

Architecture:
  - Sinusoidal timestep embedding  -> MLP  -> t_emb  (injected into every ResBlock)
  - Encoder: repeated ResBlocks + Downsample at each resolution level
  - Bottleneck: ResBlock -> optional SelfAttention -> ResBlock
  - Decoder: mirror of encoder with skip connections via channel-dim concatenation
  - GroupNorm + SiLU throughout
  - 1x1 conv at the output to map back to image channels

Default config for MNIST/FashionMNIST:
  base_ch=32, ch_mults=(1,2,4), num_res_blocks=1
  Resolutions: 28x28 -> 14x14 -> 7x7 (bottleneck)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _get_norm_groups(ch, max_groups=32):
    """Largest divisor of ch that is <= max_groups (used for GroupNorm)."""
    for g in range(max_groups, 0, -1):
        if ch % g == 0:
            return g
    return 1


# --------------------------------------------------------------------------- #
#  Timestep embedding                                                          #
# --------------------------------------------------------------------------- #


def sinusoidal_embedding(timesteps, dim):
    """
    Sinusoidal position embedding (Vaswani et al. 2017) for diffusion timesteps.
    Maps integer timesteps in {1,...,L} to R^dim vectors.

    Args:
        timesteps: 1D integer tensor of shape (B,)
        dim      : embedding dimension

    Returns:
        emb: float tensor of shape (B, dim)
    """
    assert timesteps.ndim == 1, "timesteps must be a 1D tensor"
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000)
        * torch.arange(half, dtype=torch.float32, device=timesteps.device)
        / (half - 1)
    )  # (half,)
    args = timesteps[:, None].float() * freqs[None, :]  # (B, half)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)
    if dim % 2 == 1:  # zero-pad if odd
        emb = F.pad(emb, (0, 1))
    return emb


class TimestepEmbedding(nn.Module):
    """
    Two-layer MLP that maps sinusoidal embeddings to the channel dimension
    used by ResBlocks.
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# --------------------------------------------------------------------------- #
#  Residual block                                                              #
# --------------------------------------------------------------------------- #


class ResBlock(nn.Module):
    """
    Residual block with GroupNorm + SiLU and timestep-embedding injection.

    Architecture:
        h = conv1( SiLU( GroupNorm(x) ) )
        h = h + Linear(t_emb)[:, :, None, None]   # inject t as bias
        h = conv2( Dropout( SiLU( GroupNorm(h) ) ) )
        return h + skip(x)

    This matches the design in Ho et al. 2020 (Appendix B).
    """

    def __init__(self, in_ch, out_ch, t_emb_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(_get_norm_groups(in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        # Project t_emb to out_ch -- added as a spatial bias after conv1
        self.t_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim, out_ch),
        )

        self.norm2 = nn.GroupNorm(_get_norm_groups(out_ch), out_ch)
        self.dropout = nn.Dropout(p=dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        # Skip connection: 1x1 conv to match channels if needed
        self.skip = (
            nn.Conv2d(in_ch, out_ch, 1, bias=False)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        # t_emb: (B, t_emb_dim) -> projected to (B, out_ch) -> (B, out_ch, 1, 1)
        h = h + self.t_proj(t_emb)[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


# --------------------------------------------------------------------------- #
#  Downsampling / upsampling                                                   #
# --------------------------------------------------------------------------- #


class Downsample(nn.Module):
    """Halve spatial resolution with a strided 3x3 convolution."""

    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """Double spatial resolution: nearest-neighbour interpolation + 3x3 conv."""

    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


# --------------------------------------------------------------------------- #
#  (Optional) Self-attention at the bottleneck                                 #
# --------------------------------------------------------------------------- #


class SelfAttention(nn.Module):
    """
    Single-head self-attention block for the bottleneck.
    Applied after flattening spatial dims to a sequence.
    """

    def __init__(self, ch):
        super().__init__()
        self.norm = nn.GroupNorm(_get_norm_groups(ch), ch)
        self.qkv = nn.Conv2d(ch, ch * 3, kernel_size=1)
        self.proj = nn.Conv2d(ch, ch, kernel_size=1)
        self.scale = ch**-0.5

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, C, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # each (B, C, N)

        attn = torch.softmax(
            torch.bmm(q.transpose(1, 2), k) * self.scale, dim=-1
        )  # (B, N, N)
        h = torch.bmm(v, attn.transpose(1, 2)).reshape(B, C, H, W)
        return x + self.proj(h)


# --------------------------------------------------------------------------- #
#  U-Net                                                                       #
# --------------------------------------------------------------------------- #


class UNet(nn.Module):
    """
    U-Net noise predictor  eps_theta(x_t, t).

    Args:
        in_ch         : image channels (1 for MNIST, 3 for CIFAR)
        base_ch       : channel count at the finest resolution
        ch_mults      : tuple of channel multipliers per resolution level
                        e.g. (1, 2, 4) -> [32, 64, 128] with base_ch=32
        num_res_blocks: ResBlocks per encoder/decoder level
        t_emb_dim     : dimension of the projected timestep embedding
        dropout       : dropout probability inside ResBlocks
        use_attention : add SelfAttention at the bottleneck
    """

    def __init__(
        self,
        in_ch=1,
        base_ch=32,
        ch_mults=(1, 2, 4),
        num_res_blocks=1,
        t_emb_dim=128,
        dropout=0.0,
        use_attention=True,
    ):
        super().__init__()

        # Sinusoidal embedding dimension  (before MLP projection)
        sin_dim = base_ch * 4
        self._sin_dim = sin_dim

        # MLP: sin_dim -> t_emb_dim
        self.t_emb_mlp = TimestepEmbedding(sin_dim, t_emb_dim)

        channels = [base_ch * m for m in ch_mults]  # channel count per level
        n_levels = len(channels)

        # ------------------------------------------------------------------ #
        #  Input projection                                                   #
        # ------------------------------------------------------------------ #
        self.init_conv = nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1)

        # ------------------------------------------------------------------ #
        #  Encoder                                                            #
        # ------------------------------------------------------------------ #
        # enc_blocks[i] : ModuleList of ResBlocks for level i
        # downsamplers[i]: Downsample or None (None at the deepest level)
        # skip_chs[i]   : number of channels produced by encoder level i
        #                  (used to size the decoder's concat input)

        self.enc_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        skip_chs = []

        ch = base_ch
        for i, ch_out in enumerate(channels):
            level = nn.ModuleList(
                [
                    ResBlock(ch if j == 0 else ch_out, ch_out, t_emb_dim, dropout)
                    for j in range(num_res_blocks)
                ]
            )
            self.enc_blocks.append(level)
            ch = ch_out
            skip_chs.append(ch)

            if i < n_levels - 1:
                self.downsamplers.append(Downsample(ch))
            else:
                self.downsamplers.append(None)  # no downsampling at bottom

        # ------------------------------------------------------------------ #
        #  Bottleneck                                                         #
        # ------------------------------------------------------------------ #
        self.mid_res1 = ResBlock(ch, ch, t_emb_dim, dropout)
        self.mid_attn = SelfAttention(ch) if use_attention else nn.Identity()
        self.mid_res2 = ResBlock(ch, ch, t_emb_dim, dropout)

        # ------------------------------------------------------------------ #
        #  Decoder                                                            #
        # ------------------------------------------------------------------ #
        # Mirrors encoder in reverse.  At each level:
        #   - concatenate skip from the corresponding encoder level
        #   - apply ResBlocks
        #   - upsample (except at the finest level)

        self.dec_blocks = nn.ModuleList()
        self.upsamplers = nn.ModuleList()

        for i, ch_out in enumerate(reversed(channels)):
            skip_ch = skip_chs[n_levels - 1 - i]

            # First ResBlock takes the concatenated tensor [h || skip]
            in_ch_first = ch + skip_ch
            level = nn.ModuleList()
            for j in range(num_res_blocks):
                level.append(
                    ResBlock(
                        in_ch_first if j == 0 else ch_out,
                        ch_out,
                        t_emb_dim,
                        dropout,
                    )
                )
            self.dec_blocks.append(level)
            ch = ch_out

            if i < n_levels - 1:
                self.upsamplers.append(Upsample(ch))
            else:
                self.upsamplers.append(None)

        # ------------------------------------------------------------------ #
        #  Output head                                                        #
        # ------------------------------------------------------------------ #
        self.out_norm = nn.GroupNorm(_get_norm_groups(ch), ch)
        self.out_conv = nn.Conv2d(ch, in_ch, kernel_size=1)

    # ---------------------------------------------------------------------- #

    def forward(self, x, t):
        """
        Args:
            x: noised image tensor (B, C, H, W)
            t: 1-indexed timestep tensor (B,)

        Returns:
            eps_hat: predicted noise (B, C, H, W), same shape as x
        """
        # ── Timestep embedding ────────────────────────────────────────────
        t_emb = self.t_emb_mlp(sinusoidal_embedding(t, self._sin_dim))  # (B, t_emb_dim)

        # ── Encoder ───────────────────────────────────────────────────────
        h = self.init_conv(x)

        skips = []
        for level, ds in zip(self.enc_blocks, self.downsamplers):
            for block in level:
                h = block(h, t_emb)
            skips.append(h)  # save for skip connection
            if ds is not None:
                h = ds(h)

        # ── Bottleneck ────────────────────────────────────────────────────
        h = self.mid_res1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_res2(h, t_emb)

        # ── Decoder ───────────────────────────────────────────────────────
        for level, us in zip(self.dec_blocks, self.upsamplers):
            skip = skips.pop()  # match encoder level (LIFO)
            h = torch.cat([h, skip], dim=1)  # channel concatenation
            for block in level:
                h = block(h, t_emb)
            if us is not None:
                h = us(h)

        # ── Output ────────────────────────────────────────────────────────
        h = F.silu(self.out_norm(h))
        return self.out_conv(h)
