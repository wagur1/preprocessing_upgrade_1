"""Standalone checks for the real-forward/proxy-backward STE."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.engine import _straight_through


def main() -> None:
    x_pre = torch.tensor([-0.4, 0.2, 0.8], requires_grad=True)
    x_hat_prx = 2.0 * x_pre
    x_hat_real = x_pre.detach().round()
    x_hat = _straight_through(x_hat_prx, x_hat_real)

    assert torch.allclose(x_hat, x_hat_real)
    x_hat.sum().backward()
    assert torch.allclose(x_pre.grad, torch.full_like(x_pre, 2.0))

    qp_to_quality = {22: 8, 27: 5, 32: 3, 37: 2, 42: 1}
    mapped = [qp_to_quality[qp] for qp in sorted(qp_to_quality)]
    assert all(a >= b for a, b in zip(mapped, mapped[1:]))
    print("STE self-check passed")


if __name__ == "__main__":
    main()
