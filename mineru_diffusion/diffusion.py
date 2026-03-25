"""Discrete diffusion process for MinerU-Diffusion.

Implements the absorbing-state discrete diffusion from the paper:
- Forward process: q(x_t | x_0) = Prod_i Cat(x_t^i; (1-t)*delta_{x_0^i} + t*delta_{[MASK]})
- Reverse process: p_theta(x_0^i | x_t, Q) via neural network
- ELBO training objective (Eq. 5 from paper)

Based on MDLM (Masked Diffusion Language Models) formulation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def forward_process(
    x_0: torch.Tensor,
    t: torch.Tensor,
    mask_token_id: int = 0,
) -> torch.Tensor:
    """Apply discrete diffusion forward process.

    q(x_t | x_0) = Prod_i Cat(x_t^i; (1-t)*delta_{x_0^i} + t*delta_{[MASK]})

    Each token is independently masked with probability t.

    Args:
        x_0: Clean token IDs [batch, seq_len].
        t: Diffusion timestep in [0, 1], shape [batch] or [batch, 1].
        mask_token_id: Index of the [MASK] token.

    Returns:
        x_t: Noised token IDs [batch, seq_len].
    """
    if t.dim() == 1:
        t = t.unsqueeze(-1)  # [batch, 1]

    # Each token is masked with probability t
    mask_prob = t.expand_as(x_0)
    mask = torch.rand_like(mask_prob.float()) < mask_prob
    x_t = x_0.clone()
    x_t[mask] = mask_token_id
    return x_t


def get_masked_positions(x_t: torch.Tensor, mask_token_id: int = 0) -> torch.Tensor:
    """Get boolean mask of [MASK] positions.

    Args:
        x_t: Noised token IDs [batch, seq_len].
        mask_token_id: Index of the [MASK] token.

    Returns:
        Boolean mask [batch, seq_len] where True = masked position.
    """
    return x_t == mask_token_id


def compute_elbo_loss(
    logits: torch.Tensor,
    x_0: torch.Tensor,
    x_t: torch.Tensor,
    t: torch.Tensor,
    mask_token_id: int = 0,
) -> torch.Tensor:
    """Compute the discrete diffusion ELBO loss (Eq. 5 from paper).

    J(x_0, Q, theta) = integral_0^1 1/(t*|x_0|) *
        E_{q(x_t|x_0)} [sum_{i: x_t^i=[MASK]} log p_theta(x_0^i | x_t, Q)] dt

    We approximate the integral via the sampled t and compute loss only
    on masked positions.

    Args:
        logits: Model predictions [batch, seq_len, vocab_size].
        x_0: Clean token IDs [batch, seq_len].
        x_t: Noised token IDs [batch, seq_len].
        t: Diffusion timestep [batch] or [batch, 1].
        mask_token_id: Index of [MASK] token.

    Returns:
        Scalar loss tensor.
    """
    if t.dim() == 2:
        t = t.squeeze(-1)  # [batch]

    batch_size, seq_len = x_0.shape
    masked = get_masked_positions(x_t, mask_token_id)  # [batch, seq_len]

    # Log probabilities at masked positions
    log_probs = F.log_softmax(logits, dim=-1)  # [batch, seq_len, vocab_size]
    target_log_probs = log_probs.gather(
        2, x_0.unsqueeze(-1)
    ).squeeze(-1)  # [batch, seq_len]

    # Zero out non-masked positions
    masked_log_probs = target_log_probs * masked.float()

    # Sum over masked positions per sample
    sum_log_probs = masked_log_probs.sum(dim=-1)  # [batch]

    # Weight by 1 / (t * |x_0|) as per the ELBO
    # Clamp t to avoid division by zero at t=0
    t_clamped = t.clamp(min=1e-5)
    weights = 1.0 / (t_clamped * seq_len)

    # Negative ELBO (we minimize)
    loss = -(weights * sum_log_probs).mean()
    return loss


def sample_timesteps(batch_size: int, device: torch.device) -> torch.Tensor:
    """Sample random timesteps uniformly from (0, 1].

    Args:
        batch_size: Number of timesteps to sample.
        device: Device for the tensor.

    Returns:
        Timesteps [batch_size] in (0, 1].
    """
    # Uniform in (0, 1], avoid exactly 0
    return torch.rand(batch_size, device=device).clamp(min=1e-5)
