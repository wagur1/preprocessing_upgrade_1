"""Standalone checks for FiLM rate-conditioning (run where torch is installed)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.preprocessor import FiLM, VideoPreprocessor


def main() -> None:
    x = torch.randn(2, 8, 4, 16, 16)

    # (a) FiLM is exact identity at init (zero-init last layer), any cond.
    film = FiLM(cond_dim=1, ch=8)
    for level in (0.0, 0.5, 1.0):
        cond = torch.full((2, 1), level)
        assert torch.allclose(film(x, cond), x), "FiLM must start as identity"

    # (b) once weights are non-zero, different rate conditions -> different output.
    for p in film.parameters():
        torch.nn.init.normal_(p, std=0.1)
    y0 = film(x, torch.zeros(2, 1))
    y1 = film(x, torch.ones(2, 1))
    assert not torch.allclose(y0, y1), "FiLM output must depend on the condition"

    # (c) preprocessor: conditioned forward keeps shape; starts as identity.
    pre = VideoPreprocessor(feat_ch=8, n_blocks=2, cond_dim=1)
    vid = torch.rand(2, 3, 4, 64, 64)
    out = pre(vid, torch.full((2, 1), 0.7))
    assert out.shape == vid.shape
    assert torch.allclose(out, vid, atol=1e-6), "zero-init tail -> identity at start"
    # cond=None default must also run.
    assert pre(vid).shape == vid.shape
    print("FiLM conditioning self-check passed")


if __name__ == "__main__":
    main()
