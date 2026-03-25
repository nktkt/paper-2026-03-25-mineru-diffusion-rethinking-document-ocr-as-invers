"""MinerU-Diffusion model: Document OCR via block-attention diffusion decoding.

Architecture:
- Vision encoder (stub): projects image features to hidden_dim
- Token embedding + timestep embedding
- Block-attention Transformer decoder: diffusion denoising with block-causal mask
- Output projection: predicts token logits at masked positions

Based on "MinerU-Diffusion" (arXiv:2603.22458).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .config import ModelConfig
from .block_attention import BlockAttention, create_block_attention_bias
from .diffusion import forward_process, compute_elbo_loss, sample_timesteps, get_masked_positions


class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding for diffusion models."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed timestep t into sinusoidal features.

        Args:
            t: Timestep values [batch] in [0, 1].

        Returns:
            Embeddings [batch, dim].
        """
        half_dim = self.dim // 2
        emb = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        emb = t.unsqueeze(-1).float() * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class FeedForward(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, hidden_dim: int, intermediate_dim: int, dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class DiffusionDecoderBlock(nn.Module):
    """Single block-attention Transformer layer for diffusion decoding."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn_norm = nn.RMSNorm(config.hidden_dim, eps=config.norm_eps)
        self.attn = BlockAttention(
            config.hidden_dim, config.num_heads, config.block_size, config.dropout
        )
        self.ffn_norm = nn.RMSNorm(config.hidden_dim, eps=config.norm_eps)
        self.ffn = FeedForward(config.hidden_dim, config.intermediate_dim, config.dropout)

    def forward(self, x: torch.Tensor, attn_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), attn_bias)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class VisionEncoderStub(nn.Module):
    """Placeholder vision encoder (paper uses Qwen2-VL-7B).

    Projects random or pre-extracted visual features to the decoder's hidden dim.
    In a real implementation, this would be a full vision transformer.
    """

    def __init__(self, vision_dim: int, hidden_dim: int, max_visual_tokens: int = 64):
        super().__init__()
        self.proj = nn.Linear(vision_dim, hidden_dim)
        self.max_visual_tokens = max_visual_tokens

    def forward(self, visual_features: Optional[torch.Tensor] = None,
                batch_size: int = 1, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        """Encode visual features or produce dummy features.

        Args:
            visual_features: Pre-extracted features [batch, num_tokens, vision_dim].
                If None, produces zero features.
            batch_size: Batch size (used when visual_features is None).
            device: Device (used when visual_features is None).

        Returns:
            Projected features [batch, num_visual_tokens, hidden_dim].
        """
        if visual_features is not None:
            return self.proj(visual_features)
        # Dummy features for testing
        dummy = torch.zeros(batch_size, self.max_visual_tokens,
                            self.proj.in_features, device=device)
        return self.proj(dummy)


class MinerUDiffusion(nn.Module):
    """MinerU-Diffusion: Document OCR via block-attention diffusion decoding.

    Combines a vision encoder with a block-attention diffusion Transformer decoder.
    The decoder uses discrete diffusion (absorbing-state MDLM) with block-causal
    attention: full bidirectional attention within blocks, causal across blocks.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Vision encoder (stub)
        self.vision_encoder = VisionEncoderStub(
            config.vision_dim, config.hidden_dim
        )

        # Token and timestep embeddings
        self.tok_emb = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.time_emb = SinusoidalTimestepEmbedding(config.hidden_dim)
        self.time_proj = nn.Linear(config.hidden_dim, config.hidden_dim)

        # Block-attention Transformer decoder
        self.layers = nn.ModuleList([
            DiffusionDecoderBlock(config) for _ in range(config.num_layers)
        ])
        self.final_norm = nn.RMSNorm(config.hidden_dim, eps=config.norm_eps)

        # Output projection
        self.output_proj = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        visual_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass: predict clean tokens from noised input.

        Args:
            x_t: Noised token IDs [batch, seq_len].
            t: Diffusion timestep [batch] in [0, 1].
            visual_features: Optional visual conditioning [batch, num_vis, vision_dim].

        Returns:
            Logits [batch, seq_len, vocab_size].
        """
        B, T = x_t.shape

        # Token embeddings
        h = self.tok_emb(x_t)  # [B, T, D]

        # Add timestep embedding (broadcast to all positions)
        t_emb = self.time_proj(self.time_emb(t))  # [B, D]
        h = h + t_emb.unsqueeze(1)

        # Add visual conditioning if provided
        if visual_features is not None:
            vis = self.vision_encoder(visual_features)  # [B, V, D]
            # Prepend visual tokens (cross-attention via concatenation)
            h = torch.cat([vis, h], dim=1)  # [B, V+T, D]
            total_len = h.shape[1]
        else:
            total_len = T

        # Block-attention bias
        attn_bias = create_block_attention_bias(
            total_len, self.config.block_size, h.device, h.dtype
        )

        # Transformer layers
        for layer in self.layers:
            h = layer(h, attn_bias)

        h = self.final_norm(h)

        # Remove visual prefix if added
        if visual_features is not None:
            h = h[:, vis.shape[1]:, :]  # [B, T, D]

        logits = self.output_proj(h)  # [B, T, vocab_size]
        return logits

    def compute_loss(
        self,
        x_0: torch.Tensor,
        visual_features: Optional[torch.Tensor] = None,
    ) -> dict:
        """Compute ELBO training loss.

        Args:
            x_0: Clean token IDs [batch, seq_len].
            visual_features: Optional visual conditioning.

        Returns:
            Dictionary with 'loss' and metadata.
        """
        B = x_0.shape[0]
        mask_id = self.config.diffusion.mask_token_id

        # Sample timestep
        t = sample_timesteps(B, x_0.device)

        # Forward diffusion
        x_t = forward_process(x_0, t, mask_id)

        # Predict clean tokens
        logits = self.forward(x_t, t, visual_features)

        # ELBO loss (only on masked positions)
        loss = compute_elbo_loss(logits, x_0, x_t, t, mask_id)

        # Count masked tokens for monitoring
        num_masked = get_masked_positions(x_t, mask_id).float().sum()

        return {
            "loss": loss,
            "num_masked": num_masked.item(),
            "timestep_mean": t.mean().item(),
        }

    @torch.no_grad()
    def generate(
        self,
        seq_len: int,
        batch_size: int = 1,
        num_steps: int = 10,
        confidence_threshold: float = 0.95,
        visual_features: Optional[torch.Tensor] = None,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """Generate tokens via iterative diffusion denoising.

        Starts from all [MASK] tokens and progressively reveals tokens
        using confidence-based adaptive scheduling.

        Args:
            seq_len: Length of sequence to generate.
            batch_size: Number of sequences.
            num_steps: Maximum number of denoising steps.
            confidence_threshold: Token confirmation threshold (tau).
            visual_features: Optional visual conditioning.
            device: Device for generation.

        Returns:
            Generated token IDs [batch_size, seq_len].
        """
        self.eval()
        mask_id = self.config.diffusion.mask_token_id

        # Start from all [MASK]
        x = torch.full((batch_size, seq_len), mask_id, dtype=torch.long, device=device)
        confirmed = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)

        for step in range(num_steps):
            # Decreasing timestep: from ~1 to ~0
            t_val = 1.0 - step / num_steps
            t = torch.full((batch_size,), t_val, device=device)

            logits = self.forward(x, t, visual_features)
            probs = F.softmax(logits, dim=-1)

            # Get most likely tokens and their confidence
            max_probs, predicted = probs.max(dim=-1)

            # Confirm tokens above threshold (and not already confirmed)
            newly_confirmed = (max_probs >= confidence_threshold) & (~confirmed)
            confirmed = confirmed | newly_confirmed
            x[confirmed] = predicted[confirmed]

            # If all tokens confirmed, stop early
            if confirmed.all():
                break

        # Fill remaining masks with argmax
        remaining = ~confirmed
        if remaining.any():
            t_final = torch.full((batch_size,), 0.01, device=device)
            logits = self.forward(x, t_final, visual_features)
            final_pred = logits.argmax(dim=-1)
            x[remaining] = final_pred[remaining]

        return x

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
