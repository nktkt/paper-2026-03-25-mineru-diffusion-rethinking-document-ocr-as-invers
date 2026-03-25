"""Tests for MinerU-Diffusion model."""

import pytest
import torch

from mineru_diffusion.config import ModelConfig, DiffusionConfig, PRESET_CONFIGS
from mineru_diffusion.model import MinerUDiffusion


@pytest.fixture
def tiny_config():
    return PRESET_CONFIGS["tiny"]


class TestMinerUDiffusion:
    def test_forward_shape(self, tiny_config):
        model = MinerUDiffusion(tiny_config)
        x_t = torch.randint(0, tiny_config.vocab_size, (2, 32))
        t = torch.tensor([0.5, 0.5])
        logits = model(x_t, t)
        assert logits.shape == (2, 32, tiny_config.vocab_size)

    def test_forward_with_visual(self, tiny_config):
        model = MinerUDiffusion(tiny_config)
        x_t = torch.randint(0, tiny_config.vocab_size, (2, 16))
        t = torch.tensor([0.5, 0.5])
        vis = torch.randn(2, 8, tiny_config.vision_dim)
        logits = model(x_t, t, visual_features=vis)
        assert logits.shape == (2, 16, tiny_config.vocab_size)

    def test_compute_loss(self, tiny_config):
        model = MinerUDiffusion(tiny_config)
        x_0 = torch.randint(1, tiny_config.vocab_size, (4, 32))
        result = model.compute_loss(x_0)
        assert "loss" in result
        assert result["loss"].ndim == 0
        assert torch.isfinite(result["loss"])
        assert result["loss"].item() > 0

    def test_loss_gradient(self, tiny_config):
        model = MinerUDiffusion(tiny_config)
        x_0 = torch.randint(1, tiny_config.vocab_size, (2, 16))
        result = model.compute_loss(x_0)
        result["loss"].backward()
        # Check gradient on decoder layers (vision encoder has no grad without visual input)
        for name, param in model.named_parameters():
            if param.requires_grad and "vision_encoder" not in name:
                assert param.grad is not None, f"No gradient for {name}"

    def test_loss_decreases(self, tiny_config):
        torch.manual_seed(0)
        model = MinerUDiffusion(tiny_config)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        x_0 = torch.randint(1, tiny_config.vocab_size, (8, 16))

        losses = []
        for _ in range(50):
            result = model.compute_loss(x_0)
            losses.append(result["loss"].item())
            optimizer.zero_grad()
            result["loss"].backward()
            optimizer.step()

        # Compare average of last 10 vs first 10 (diffusion loss is noisy)
        avg_first = sum(losses[:10]) / 10
        avg_last = sum(losses[-10:]) / 10
        assert avg_last < avg_first

    def test_generate_shape(self, tiny_config):
        model = MinerUDiffusion(tiny_config)
        output = model.generate(seq_len=16, batch_size=2, num_steps=5)
        assert output.shape == (2, 16)

    def test_generate_no_masks(self, tiny_config):
        model = MinerUDiffusion(tiny_config)
        mask_id = tiny_config.diffusion.mask_token_id
        output = model.generate(seq_len=16, batch_size=1, num_steps=10)
        # After generation, there should be no [MASK] tokens
        assert (output != mask_id).all()

    def test_generate_with_visual(self, tiny_config):
        model = MinerUDiffusion(tiny_config)
        vis = torch.randn(1, 8, tiny_config.vision_dim)
        output = model.generate(seq_len=16, batch_size=1, num_steps=5,
                                visual_features=vis)
        assert output.shape == (1, 16)

    def test_parameter_count(self, tiny_config):
        model = MinerUDiffusion(tiny_config)
        assert model.count_parameters() > 0

    def test_presets_valid(self):
        for name, config in PRESET_CONFIGS.items():
            model = MinerUDiffusion(config)
            x_t = torch.randint(0, config.vocab_size, (1, 16))
            t = torch.tensor([0.5])
            logits = model(x_t, t)
            assert logits.shape == (1, 16, config.vocab_size)
