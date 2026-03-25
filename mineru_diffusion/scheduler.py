"""Confidence-based adaptive inference scheduler.

Implements the dynamic scheduling from the paper:
- Tokens are confirmed when their predicted confidence exceeds tau
- Higher tau = more conservative (slower but more accurate)
- Lower tau = more aggressive (faster but less accurate)

From paper ablations:
    tau=0.50 -> 164.8 TPS, 87.61% accuracy
    tau=0.95 -> 108.9 TPS, 93.37% accuracy
    tau=0.99 -> 21.86 TPS, 93.02% accuracy
"""

import torch
from dataclasses import dataclass, field
from typing import List, Optional

from .config import SchedulerConfig


@dataclass
class SchedulerState:
    """Tracks the state of the adaptive scheduler during inference."""

    step: int = 0
    total_tokens: int = 0
    confirmed_tokens: int = 0
    steps_taken: int = 0
    confidence_history: List[float] = field(default_factory=list)

    @property
    def fraction_confirmed(self) -> float:
        if self.total_tokens == 0:
            return 0.0
        return self.confirmed_tokens / self.total_tokens

    @property
    def is_complete(self) -> bool:
        return self.confirmed_tokens >= self.total_tokens


class AdaptiveScheduler:
    """Confidence-based adaptive scheduler for diffusion inference.

    At each denoising step:
    1. Model predicts token probabilities at all positions
    2. Tokens with confidence >= tau are confirmed and frozen
    3. Remaining tokens continue denoising
    4. Process repeats until all confirmed or max_steps reached
    """

    def __init__(self, config: Optional[SchedulerConfig] = None):
        self.config = config or SchedulerConfig()

    def create_state(self, seq_len: int, batch_size: int = 1) -> SchedulerState:
        """Initialize scheduler state for a new sequence."""
        return SchedulerState(
            total_tokens=seq_len * batch_size,
            confirmed_tokens=0,
        )

    def step(
        self,
        probs: torch.Tensor,
        confirmed: torch.Tensor,
        state: SchedulerState,
    ) -> tuple:
        """Execute one scheduling step.

        Args:
            probs: Token probabilities [batch, seq_len, vocab_size].
            confirmed: Boolean mask of already-confirmed tokens [batch, seq_len].
            state: Current scheduler state.

        Returns:
            Tuple of (newly_confirmed_mask, predicted_tokens, updated_state).
        """
        # Get max confidence and predictions
        max_probs, predicted = probs.max(dim=-1)  # [batch, seq_len]

        # Confirm tokens above threshold that aren't already confirmed
        newly_confirmed = (max_probs >= self.config.confidence_threshold) & (~confirmed)

        # Update state
        state.step += 1
        state.steps_taken += 1
        state.confirmed_tokens = (confirmed | newly_confirmed).sum().item()
        state.confidence_history.append(max_probs.mean().item())

        return newly_confirmed, predicted, state

    def should_stop(self, state: SchedulerState) -> bool:
        """Check if inference should stop."""
        if state.is_complete:
            return True
        if state.steps_taken >= self.config.max_steps:
            return True
        return False

    def get_timestep(self, state: SchedulerState) -> float:
        """Get the diffusion timestep for the current step.

        Linearly decreases from ~1 to ~0 over max_steps.
        """
        return max(0.01, 1.0 - state.step / self.config.max_steps)


def compute_token_confidence(probs: torch.Tensor) -> torch.Tensor:
    """Compute per-token confidence from probability distribution.

    Confidence = max probability across vocabulary.

    Args:
        probs: Token probabilities [batch, seq_len, vocab_size].

    Returns:
        Confidence scores [batch, seq_len].
    """
    return probs.max(dim=-1).values
