"""Rate-conditioned two-branch neural preprocessor (paper Figure 2 + FiLM).

The preprocessor edits the input video *before* compression so the downstream
machine-vision task survives the codec better, while keeping the output close to
the source (residual design => high perceptual similarity).

Design:

    input video x [B,C,T,H,W]  +  rate condition c [B, cond_dim]
        |
        +--> temporal branch : 3D convs over the inter-frame dimension,
        |                       FiLM-modulated by c   -> temporal features
        +--> spatial  branch : per-frame 2D convs (3D with T-kernel 1),
        |                       FiLM-modulated by c   -> spatial features
        +--> conditional attention merges the two feature sets
        +--> residual connection: out = x + res_scale * delta

``c`` carries the target compression operating point (currently a single
normalised QP; the vector is sized by ``cond_dim`` so an explicit log target-rate
can be appended for rate control). FiLM (Perez et al. 2018) injects it as a
per-channel affine ``(1+gamma) * feat + beta`` inside every residual block, so a
*single* preprocessor adapts across the whole rate range instead of learning one
blurry average (fixes the baseline's rate-blindness).

Everything is differentiable so gradients from the codec + task analyzer flow
back into these weights. Only this module is trained.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _act() -> nn.Module:
    return nn.LeakyReLU(0.1, inplace=True)


class FiLM(nn.Module):
    """Feature-wise linear modulation from the rate condition vector.

    Predicts per-channel (gamma, beta) from ``cond`` and applies
    ``(1 + gamma) * x + beta``. The last layer is zero-initialised so at start
    gamma=beta=0 -> exact identity (stable early training, same spirit as the
    zero-init residual tail).
    """

    def __init__(self, cond_dim: int, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, ch), _act(), nn.Linear(ch, 2 * ch)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.net(cond).chunk(2, dim=1)  # each [B, ch]
        gamma = gamma[:, :, None, None, None]
        beta = beta[:, :, None, None, None]
        return x * (1.0 + gamma) + beta


class _ResBlock3d(nn.Module):
    """Residual block (temporal or spatial kernels) with FiLM conditioning."""

    def __init__(self, ch: int, temporal: bool, cond_dim: int):
        super().__init__()
        # temporal branch: (3,3,3) sees neighbouring frames.
        # spatial  branch: (1,3,3) sees only within-frame context.
        kt = 3 if temporal else 1
        pt = 1 if temporal else 0
        self.conv1 = nn.Conv3d(ch, ch, (kt, 3, 3), padding=(pt, 1, 1))
        self.conv2 = nn.Conv3d(ch, ch, (kt, 3, 3), padding=(pt, 1, 1))
        self.film = FiLM(cond_dim, ch)
        self.act = _act()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        y = self.conv2(self.act(self.conv1(x)))
        y = self.film(y, cond)
        return x + y


class _Branch(nn.Module):
    """A stack of FiLM residual blocks (either temporal or spatial)."""

    def __init__(self, in_ch: int, feat_ch: int, n_blocks: int, temporal: bool, cond_dim: int):
        super().__init__()
        kt = 3 if temporal else 1
        pt = 1 if temporal else 0
        self.head = nn.Conv3d(in_ch, feat_ch, (kt, 3, 3), padding=(pt, 1, 1))
        self.act = _act()
        self.blocks = nn.ModuleList(
            [_ResBlock3d(feat_ch, temporal, cond_dim) for _ in range(n_blocks)]
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.act(self.head(x))
        for blk in self.blocks:
            h = blk(h, cond)
        return h


class _ConditionalAttention(nn.Module):
    """Conditional attention fusion of the temporal and spatial features.

    The gate is *conditioned* on both branches jointly: we concatenate them,
    predict a per-location soft weight, and use it to blend the two streams.
    """

    def __init__(self, feat_ch: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv3d(2 * feat_ch, feat_ch, 1),
            _act(),
            nn.Conv3d(feat_ch, feat_ch, 1),
            nn.Sigmoid(),
        )

    def forward(self, f_t: torch.Tensor, f_s: torch.Tensor) -> torch.Tensor:
        g = self.gate(torch.cat([f_t, f_s], dim=1))
        return g * f_t + (1.0 - g) * f_s


class VideoPreprocessor(nn.Module):
    """Rate-conditioned neural video preprocessor.

    Args:
        in_ch:    input channels (3 for RGB).
        feat_ch:  feature width of each branch.
        n_blocks: residual blocks per branch.
        res_scale: scales the learned residual before adding to the input.
        cond_dim: size of the rate condition vector (1 = normalised QP).
    """

    def __init__(
        self,
        in_ch: int = 3,
        feat_ch: int = 32,
        n_blocks: int = 3,
        res_scale: float = 1.0,
        cond_dim: int = 1,
    ):
        super().__init__()
        self.cond_dim = cond_dim
        self.temporal = _Branch(in_ch, feat_ch, n_blocks, True, cond_dim)
        self.spatial = _Branch(in_ch, feat_ch, n_blocks, False, cond_dim)
        self.fuse = _ConditionalAttention(feat_ch)
        self.tail = nn.Sequential(
            nn.Conv3d(feat_ch, feat_ch, (1, 3, 3), padding=(0, 1, 1)),
            _act(),
            nn.Conv3d(feat_ch, in_ch, (1, 3, 3), padding=(0, 1, 1)),
        )
        self.res_scale = res_scale
        # Zero-init the last conv so the preprocessor starts as (approximate)
        # identity -> stable early training.
        nn.init.zeros_(self.tail[-1].weight)
        nn.init.zeros_(self.tail[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """x: [B,C,T,H,W] in [0,1]. cond: [B, cond_dim] rate operating point
        (defaults to zeros = top-quality operating point). Returns video in [0,1]."""
        if cond is None:
            cond = x.new_zeros(x.shape[0], self.cond_dim)
        f_t = self.temporal(x, cond)
        f_s = self.spatial(x, cond)
        fused = self.fuse(f_t, f_s)
        delta = self.tail(fused)
        out = x + self.res_scale * delta
        return out.clamp(0.0, 1.0)
