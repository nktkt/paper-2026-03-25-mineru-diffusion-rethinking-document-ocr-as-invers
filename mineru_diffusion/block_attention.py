"""Block-attention mechanism for MinerU-Diffusion.

Implements the block-attention mask from the paper:
- Within blocks: Full bidirectional attention (parallel diffusion denoising)
- Across blocks: Causal attention (autoregressive structure)

M_ij = 1  if b(i) = b(j)   (same block)
     = 1  if b(j) < b(i)   (preceding block)
     = 0  otherwise          (future block)

This reduces complexity from O(L^2) to O(B * L'^2) where L = B * L'.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def create_block_attention_mask(
    seq_len: int,
    block_size: int,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create the block-attention mask as described in the paper.

    Args:
        seq_len: Total sequence length L.
        block_size: Tokens per block L'.
        device: Device for the mask tensor.
        dtype: Data type for the mask.

    Returns:
        Attention mask [seq_len, seq_len] where 1 = attend, 0 = mask.
    """
    # Compute block index for each position
    block_ids = torch.arange(seq_len, device=device) // block_size  # [seq_len]

    # M_ij = 1 if b(i) == b(j) or b(j) < b(i)
    # Equivalently: M_ij = 1 if b(j) <= b(i)
    row_blocks = block_ids.unsqueeze(1)  # [seq_len, 1]
    col_blocks = block_ids.unsqueeze(0)  # [1, seq_len]

    mask = (col_blocks <= row_blocks).to(dtype)  # [seq_len, seq_len]
    return mask


def create_block_attention_bias(
    seq_len: int,
    block_size: int,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create block-attention bias for use with scaled_dot_product_attention.

    Returns -inf for masked positions, 0 for attended positions.

    Args:
        seq_len: Total sequence length.
        block_size: Tokens per block.
        device: Device for tensor.
        dtype: Data type.

    Returns:
        Attention bias [1, 1, seq_len, seq_len].
    """
    mask = create_block_attention_mask(seq_len, block_size, device, dtype)
    bias = torch.where(mask == 1, torch.tensor(0.0, dtype=dtype, device=device),
                       torch.tensor(float("-inf"), dtype=dtype, device=device))
    return bias.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]


class BlockAttention(nn.Module):
    """Multi-head attention with block-attention mask.

    Within each block: full bidirectional attention.
    Across blocks: causal (can only attend to preceding blocks).
    """

    def __init__(self, hidden_dim: int, num_heads: int, block_size: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.block_size = block_size

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with block-attention.

        Args:
            x: Input tensor [batch, seq_len, hidden_dim].
            attn_bias: Pre-computed attention bias [1, 1, seq_len, seq_len].
                If None, creates one from block_size.

        Returns:
            Output tensor [batch, seq_len, hidden_dim].
        """
        B, T, D = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        if attn_bias is None:
            attn_bias = create_block_attention_bias(T, self.block_size, x.device, x.dtype)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_bias,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
        )

        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.o_proj(out)
