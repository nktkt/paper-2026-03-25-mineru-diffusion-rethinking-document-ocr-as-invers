"""Uncertainty-driven curriculum learning utilities.

Implements Stage 2 from the paper: hard-case mining via inference consistency.

Algorithm:
1. Run T stochastic forward passes on each sample
2. Compute consistency score C(x) = 1 - normalized_variance(predictions)
3. Select hard cases where C(x) < tau
4. Apply weighted loss: w(x) = 1 + beta * (1 - C(x))
"""

import torch
import torch.nn as nn
from typing import List, Optional

from .config import CurriculumConfig


@torch.no_grad()
def compute_inference_consistency(
    model: nn.Module,
    x_0: torch.Tensor,
    num_passes: int = 5,
    mask_token_id: int = 0,
    fixed_t: float = 0.5,
) -> torch.Tensor:
    """Compute inference consistency across T stochastic passes.

    For each sample, runs T forward passes with different noise realizations
    and measures how consistent the predictions are.

    C(x) = 1 - mean_variance_across_passes / max_possible_variance

    Args:
        model: The diffusion model.
        x_0: Clean token IDs [batch, seq_len].
        num_passes: Number of stochastic forward passes (T).
        mask_token_id: [MASK] token ID.
        fixed_t: Fixed timestep for all passes.

    Returns:
        Consistency scores [batch] in [0, 1]. Higher = more consistent.
    """
    from .diffusion import forward_process

    model.eval()
    B, L = x_0.shape
    all_predictions = []

    for _ in range(num_passes):
        t = torch.full((B,), fixed_t, device=x_0.device)
        x_t = forward_process(x_0, t, mask_token_id)
        logits = model(x_t, t)
        preds = logits.argmax(dim=-1)  # [B, L]
        all_predictions.append(preds)

    # Stack predictions: [T, B, L]
    preds_stack = torch.stack(all_predictions, dim=0).float()

    # Compute per-position variance across passes, then mean over positions
    variance = preds_stack.var(dim=0)  # [B, L]
    mean_variance = variance.mean(dim=-1)  # [B]

    # Normalize: approximate max variance as vocab_size^2 / 12 (uniform)
    # Simplified: just use 1 / (1 + mean_variance) as consistency
    consistency = 1.0 / (1.0 + mean_variance)

    return consistency


def select_hard_cases(
    consistency_scores: torch.Tensor,
    threshold: float = 0.8,
) -> torch.Tensor:
    """Select hard cases (low consistency) for curriculum learning.

    Args:
        consistency_scores: Per-sample consistency [batch].
        threshold: Samples below this threshold are selected.

    Returns:
        Boolean mask [batch] where True = hard case.
    """
    return consistency_scores < threshold


def compute_curriculum_weights(
    consistency_scores: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    """Compute sample weights for curriculum learning.

    w(x) = 1 + beta * (1 - C(x))

    Hard cases (low C) get higher weight.

    Args:
        consistency_scores: Per-sample consistency [batch].
        beta: Weight scaling factor.

    Returns:
        Sample weights [batch].
    """
    return 1.0 + beta * (1.0 - consistency_scores)


def weighted_loss(
    per_sample_loss: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Apply curriculum weights to per-sample losses.

    Args:
        per_sample_loss: Loss per sample [batch].
        weights: Curriculum weights [batch].

    Returns:
        Weighted mean loss (scalar).
    """
    return (per_sample_loss * weights).mean()
