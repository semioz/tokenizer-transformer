import math
import os
from typing import BinaryIO, IO

import torch

# Cosine annealing: decay LR from alpha_max to alpha_min along a half-cosine
# (0 -> pi). Slope is 0 at both ends, so it eases in/out instead of
# cliff-dropping. T_c must be known up front since the curve is defined
# relative to its endpoints. Warmup ramps 0 -> alpha_max first (transformers
# are unstable with high LR at random init); after T_c we hold at alpha_min.

def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """Cosine learning rate schedule with linear warmup.

    Three phases:
      - Warm-up      (t <  T_w): linear ramp 0 -> alpha_max
      - Cosine decay (T_w <= t <= T_c): cosine from alpha_max -> alpha_min
      - Post-anneal  (t >  T_c): hold at alpha_min
    """
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    if it <= cosine_cycle_iters:
        decay_progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        cosine = 0.5 * (1 + math.cos(decay_progress * math.pi))
        return min_learning_rate + cosine * (max_learning_rate - min_learning_rate)
    return min_learning_rate


# Gradient clipping prevents training instability from rare batches that produce
# outsized gradients. Compute the L2 norm of the combined gradient vector
# (all params flattened together); if it exceeds max_l2_norm, scale every grad
# down by max_l2_norm / (||g||_2 + eps) so the resulting norm lands just under M.
# The eps (1e-6) guards against divide-by-zero when grads are all zero.
# We clamp the scale to <= 1.0 so we never amplify small gradients.

def gradient_clipping(parameters, max_l2_norm: float, eps: float = 1e-6) -> None:
    """Clip gradients in-place so their combined L2 norm is at most max_l2_norm."""
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return

    total_norm = torch.norm(torch.stack([g.norm(2) for g in grads]), 2)
    clip_coef = max_l2_norm / (total_norm + eps)
    if clip_coef < 1:
        for g in grads:
            g.mul_(clip_coef)

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
        },
        out,
    )

def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    obj = torch.load(src, weights_only=False)
    model.load_state_dict(obj["model"])
    optimizer.load_state_dict(obj["optimizer"])
    return obj["iteration"]
