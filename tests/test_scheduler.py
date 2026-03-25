"""Tests for confidence-based adaptive scheduler."""

import pytest
import torch

from mineru_diffusion.scheduler import (
    AdaptiveScheduler, SchedulerState, SchedulerConfig, compute_token_confidence,
)


class TestAdaptiveScheduler:
    def test_create_state(self):
        scheduler = AdaptiveScheduler()
        state = scheduler.create_state(seq_len=32, batch_size=2)
        assert state.total_tokens == 64
        assert state.confirmed_tokens == 0
        assert not state.is_complete

    def test_step_confirms_high_confidence(self):
        scheduler = AdaptiveScheduler(SchedulerConfig(confidence_threshold=0.9))
        state = scheduler.create_state(seq_len=4, batch_size=1)
        confirmed = torch.zeros(1, 4, dtype=torch.bool)

        # Probabilities where some tokens are confident
        probs = torch.zeros(1, 4, 10)
        probs[0, 0, 5] = 0.95  # High confidence
        probs[0, 1, 3] = 0.50  # Low confidence
        probs[0, 2, 7] = 0.99  # High confidence
        probs[0, 3, 1] = 0.80  # Below threshold

        newly_confirmed, predicted, state = scheduler.step(probs, confirmed, state)
        assert newly_confirmed[0, 0].item() is True
        assert newly_confirmed[0, 1].item() is False
        assert newly_confirmed[0, 2].item() is True
        assert newly_confirmed[0, 3].item() is False

    def test_already_confirmed_not_reconfirmed(self):
        scheduler = AdaptiveScheduler(SchedulerConfig(confidence_threshold=0.5))
        state = scheduler.create_state(seq_len=4, batch_size=1)
        confirmed = torch.tensor([[True, False, True, False]])

        probs = torch.ones(1, 4, 10) * 0.9
        newly_confirmed, _, state = scheduler.step(probs, confirmed, state)
        # Already confirmed tokens should not be newly confirmed
        assert newly_confirmed[0, 0].item() is False
        assert newly_confirmed[0, 2].item() is False

    def test_should_stop_complete(self):
        scheduler = AdaptiveScheduler()
        state = SchedulerState(total_tokens=10, confirmed_tokens=10)
        assert scheduler.should_stop(state)

    def test_should_stop_max_steps(self):
        scheduler = AdaptiveScheduler(SchedulerConfig(max_steps=5))
        state = SchedulerState(total_tokens=10, confirmed_tokens=3, steps_taken=5)
        assert scheduler.should_stop(state)

    def test_get_timestep_decreasing(self):
        scheduler = AdaptiveScheduler(SchedulerConfig(max_steps=10))
        state1 = SchedulerState(step=1)
        state5 = SchedulerState(step=5)
        state9 = SchedulerState(step=9)
        assert scheduler.get_timestep(state1) > scheduler.get_timestep(state5)
        assert scheduler.get_timestep(state5) > scheduler.get_timestep(state9)


class TestComputeTokenConfidence:
    def test_shape(self):
        probs = torch.randn(2, 8, 100).softmax(dim=-1)
        confidence = compute_token_confidence(probs)
        assert confidence.shape == (2, 8)

    def test_range(self):
        probs = torch.randn(2, 8, 100).softmax(dim=-1)
        confidence = compute_token_confidence(probs)
        assert (confidence >= 0).all()
        assert (confidence <= 1).all()
