"""Configuration for MinerU-Diffusion."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DiffusionConfig:
    """Discrete diffusion process configuration."""

    num_diffusion_steps: int = 10
    mask_token_id: int = 0  # [MASK] token index in vocabulary


@dataclass
class ModelConfig:
    """MinerU-Diffusion model configuration.

    Architecture from the paper:
    - Vision encoder: stub (Qwen2-VL-7B in paper)
    - Diffusion decoder: block-attention Transformer (SDAR-1.7B-Chat-b32 in paper)
    - Block size: 32 tokens
    """

    vocab_size: int = 8192
    hidden_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    intermediate_dim: Optional[int] = None
    block_size: int = 32
    max_seq_len: int = 2048
    dropout: float = 0.0
    norm_eps: float = 1e-5
    vision_dim: int = 256  # Vision encoder output dim (stub)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)

    def __post_init__(self):
        if self.intermediate_dim is None:
            self.intermediate_dim = 4 * self.hidden_dim
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        if self.block_size <= 0:
            raise ValueError(f"block_size must be positive, got {self.block_size}")

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.num_heads


@dataclass
class SchedulerConfig:
    """Confidence-based adaptive scheduler configuration.

    From paper ablations:
    - tau=0.95 -> 108.9 TPS, 93.37% accuracy (recommended)
    - tau=0.50 -> 164.8 TPS, 87.61% accuracy (fast)
    - tau=0.99 -> 21.86 TPS, 93.02% accuracy (conservative)
    """

    confidence_threshold: float = 0.95
    max_steps: int = 20
    min_steps: int = 1


@dataclass
class CurriculumConfig:
    """Uncertainty-driven curriculum learning configuration.

    Stage 2 from the paper: hard-case mining via inference consistency.
    """

    num_stochastic_passes: int = 5  # T passes for consistency check
    confidence_threshold: float = 0.8  # tau for sample selection
    beta: float = 1.0  # Weight scaling factor: w(x) = 1 + beta * (1 - C(x))


PRESET_CONFIGS = {
    "tiny": ModelConfig(
        vocab_size=512, hidden_dim=128, num_layers=2, num_heads=4,
        block_size=8, max_seq_len=128, vision_dim=64,
    ),
    "small": ModelConfig(
        vocab_size=4096, hidden_dim=256, num_layers=4, num_heads=8,
        block_size=16, max_seq_len=512, vision_dim=128,
    ),
    "medium": ModelConfig(
        vocab_size=8192, hidden_dim=512, num_layers=6, num_heads=8,
        block_size=32, max_seq_len=2048, vision_dim=256,
    ),
}
