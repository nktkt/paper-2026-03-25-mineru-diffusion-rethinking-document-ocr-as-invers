"""Tests for discrete diffusion process."""

import pytest
import torch

from mineru_diffusion.diffusion import (
    forward_process, get_masked_positions, compute_elbo_loss, sample_timesteps,
)

MASK_ID = 0


class TestForwardProcess:
    def test_output_shape(self):
        x_0 = torch.randint(1, 100, (4, 32))
        t = torch.tensor([0.5, 0.5, 0.5, 0.5])
        x_t = forward_process(x_0, t, MASK_ID)
        assert x_t.shape == x_0.shape

    def test_no_masking_at_t0(self):
        x_0 = torch.randint(1, 100, (4, 32))
        t = torch.zeros(4)
        x_t = forward_process(x_0, t, MASK_ID)
        assert (x_t == x_0).all()

    def test_full_masking_at_t1(self):
        x_0 = torch.randint(1, 100, (4, 32))
        t = torch.ones(4)
        x_t = forward_process(x_0, t, MASK_ID)
        assert (x_t == MASK_ID).all()

    def test_partial_masking(self):
        torch.manual_seed(42)
        x_0 = torch.randint(1, 100, (8, 64))
        t = torch.full((8,), 0.5)
        x_t = forward_process(x_0, t, MASK_ID)
        masked_frac = (x_t == MASK_ID).float().mean()
        assert 0.3 < masked_frac < 0.7

    def test_preserves_dtype(self):
        x_0 = torch.randint(1, 100, (2, 16), dtype=torch.long)
        t = torch.tensor([0.5, 0.5])
        x_t = forward_process(x_0, t, MASK_ID)
        assert x_t.dtype == torch.long

    def test_2d_timestep(self):
        x_0 = torch.randint(1, 100, (2, 16))
        t = torch.tensor([[0.5], [0.5]])
        x_t = forward_process(x_0, t, MASK_ID)
        assert x_t.shape == x_0.shape


class TestGetMaskedPositions:
    def test_all_masked(self):
        x_t = torch.zeros(2, 8, dtype=torch.long)
        mask = get_masked_positions(x_t, MASK_ID)
        assert mask.all()

    def test_none_masked(self):
        x_t = torch.ones(2, 8, dtype=torch.long)
        mask = get_masked_positions(x_t, MASK_ID)
        assert not mask.any()


class TestComputeElboLoss:
    def test_loss_is_scalar(self):
        B, L, V = 4, 16, 100
        logits = torch.randn(B, L, V)
        x_0 = torch.randint(1, V, (B, L))
        x_t = forward_process(x_0, torch.full((B,), 0.5), MASK_ID)
        t = torch.full((B,), 0.5)
        loss = compute_elbo_loss(logits, x_0, x_t, t, MASK_ID)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_loss_is_positive(self):
        B, L, V = 4, 16, 100
        logits = torch.randn(B, L, V)
        x_0 = torch.randint(1, V, (B, L))
        x_t = forward_process(x_0, torch.full((B,), 0.5), MASK_ID)
        t = torch.full((B,), 0.5)
        loss = compute_elbo_loss(logits, x_0, x_t, t, MASK_ID)
        assert loss.item() > 0

    def test_loss_gradient(self):
        B, L, V = 2, 8, 50
        logits = torch.randn(B, L, V, requires_grad=True)
        x_0 = torch.randint(1, V, (B, L))
        x_t = forward_process(x_0, torch.full((B,), 0.5), MASK_ID)
        t = torch.full((B,), 0.5)
        loss = compute_elbo_loss(logits, x_0, x_t, t, MASK_ID)
        loss.backward()
        assert logits.grad is not None


class TestSampleTimesteps:
    def test_shape(self):
        t = sample_timesteps(8, torch.device("cpu"))
        assert t.shape == (8,)

    def test_range(self):
        t = sample_timesteps(100, torch.device("cpu"))
        assert (t > 0).all()
        assert (t <= 1).all()
