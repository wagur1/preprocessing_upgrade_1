"""Combined training objective (paper Eq. 1).

    L = alpha * (L_D + lambda * L_R) + L_Acc

  * L_D   : distortion -- MSE between the codec reconstruction x_hat and the
            *source* input video x (keeps the preprocessed+coded output close
            to the original, protecting perceptual quality).
  * L_R   : rate -- estimated bits-per-pixel from the codec's entropy model.
  * L_Acc : task accuracy loss from the frozen analyzer.

Paper defaults: alpha = 10, lambda = 0.001.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F


@dataclass
class LossWeights:
    alpha: float = 10.0
    lam: float = 0.001  # lambda in the paper


def rate_distortion_accuracy_loss(
    x_source: torch.Tensor,
    x_hat: torch.Tensor,
    bpp: torch.Tensor,
    acc_loss: torch.Tensor,
    w: LossWeights,
) -> Dict[str, torch.Tensor]:
    l_d = F.mse_loss(x_hat, x_source)
    l_r = bpp
    total = w.alpha * (l_d + w.lam * l_r) + acc_loss
    return {
        "loss": total,
        "loss_distortion": l_d.detach(),
        "loss_rate": l_r.detach(),
        "loss_acc": acc_loss.detach(),
    }
