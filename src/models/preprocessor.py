"""Two-branch neural preprocessor (paper Figure 2).

The preprocessor edits the input video *before* compression so that the
downstream machine-vision task survives the codec better, while keeping the
output close to the source (residual design => high perceptual similarity).

Design (matches the paper's description):

    input video  x  [B, C, T, H, W]
        |
        +--> temporal branch : 3D convs over the inter-frame dimension
        |                       (kernel spans T) -> temporal features
        |
        +--> spatial  branch : per-frame 2D convs, realised as 3D convs with
        |                       temporal kernel size 1 -> spatial features
        |
        +--> conditional attention merges the two feature sets
        |
        +--> residual connection: out = x + delta   (delta is small)

Everything is differentiable so gradients from the codec + task analyzer flow
back into these weights. Only this module is trained.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _act() -> nn.Module:
    return nn.LeakyReLU(0.1, inplace=True)


class _ResBlock3d(nn.Module):
    """Residual block whose conv kernels can be temporal or purely spatial."""

    def __init__(self, ch: int, temporal: bool):
        super().__init__()
        # temporal branch: (3,3,3) sees neighbouring frames.
        # spatial  branch: (1,3,3) sees only within-frame context.
        kt = 3 if temporal else 1
        pt = 1 if temporal else 0
        self.conv1 = nn.Conv3d(ch, ch, (kt, 3, 3), padding=(pt, 1, 1))
        self.conv2 = nn.Conv3d(ch, ch, (kt, 3, 3), padding=(pt, 1, 1))
        self.act = _act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv2(self.act(self.conv1(x)))
        return x + y


class _Branch(nn.Module):
    """A stack of residual blocks (either temporal or spatial)."""

    def __init__(self, in_ch: int, feat_ch: int, n_blocks: int, temporal: bool):
        super().__init__()
        kt = 3 if temporal else 1
        pt = 1 if temporal else 0
        self.head = nn.Conv3d(in_ch, feat_ch, (kt, 3, 3), padding=(pt, 1, 1))
        self.act = _act()
        self.blocks = nn.Sequential(
            *[_ResBlock3d(feat_ch, temporal) for _ in range(n_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.act(self.head(x)))


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
    """Neural video preprocessor.

    Args:
        in_ch:    input channels (3 for RGB).
        feat_ch:  feature width of each branch.
        n_blocks: residual blocks per branch.
        res_scale: scales the learned residual before adding to the input.
                   Small values keep the output near the source at init.
    """

    def __init__(
        self,
        in_ch: int = 3,
        feat_ch: int = 32,
        n_blocks: int = 3,
        res_scale: float = 1.0,
    ):
        super().__init__()
        self.temporal = _Branch(in_ch, feat_ch, n_blocks, temporal=True)
        self.spatial = _Branch(in_ch, feat_ch, n_blocks, temporal=False)
        self.fuse = _ConditionalAttention(feat_ch)
        self.tail = nn.Sequential(
            nn.Conv3d(feat_ch, feat_ch, (1, 3, 3), padding=(0, 1, 1)),
            _act(),
            nn.Conv3d(feat_ch, in_ch, (1, 3, 3), padding=(0, 1, 1)),
        )
        self.res_scale = res_scale
        # Initialise the last conv near zero so the preprocessor starts as an
        # (approximate) identity -> stable early training.
        nn.init.zeros_(self.tail[-1].weight)
        nn.init.zeros_(self.tail[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, T, H, W] in [0, 1]. Returns preprocessed video in [0, 1]."""
        f_t = self.temporal(x)
        f_s = self.spatial(x)
        fused = self.fuse(f_t, f_s)
        delta = self.tail(fused)
        out = x + self.res_scale * delta
        return out.clamp(0.0, 1.0)
