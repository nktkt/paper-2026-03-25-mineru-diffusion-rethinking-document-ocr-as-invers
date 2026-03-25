"""Tests for training pipeline."""

import pytest
import torch
from torch.utils.data import DataLoader

from mineru_diffusion.config import PRESET_CONFIGS
from mineru_diffusion.train import create_model_with_vision, train_stage, get_cosine_schedule
from mineru_diffusion.data import SyntheticOCRDataset


class TestCreateModelWithVision:
    def test_creates_model(self):
        config = PRESET_CONFIGS["tiny"]
        model = create_model_with_vision(config, image_size=64)
        assert model is not None
        # Should have CNN encoder, not stub
        from mineru_diffusion.vision_encoder import CNNVisionEncoder
        assert isinstance(model.vision_encoder, CNNVisionEncoder)

    def test_forward_with_real_images(self):
        config = PRESET_CONFIGS["tiny"]
        model = create_model_with_vision(config, image_size=64)

        images = torch.randn(2, 3, 64, 64)
        x_t = torch.randint(0, config.vocab_size, (2, 16))
        t = torch.tensor([0.5, 0.5])
        # Pass raw images; model.forward handles encoding internally
        logits = model(x_t, t, visual_features=images)
        assert logits.shape == (2, 16, config.vocab_size)


class TestTrainStage:
    def test_runs_without_error(self):
        config = PRESET_CONFIGS["tiny"]
        model = create_model_with_vision(config, image_size=64)

        dataset = SyntheticOCRDataset(
            num_samples=16, vocab_size=config.vocab_size,
            max_seq_len=32, image_size=64,
        )
        loader = DataLoader(dataset, batch_size=4, drop_last=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = get_cosine_schedule(optimizer, 1, len(loader))

        metrics = train_stage(
            model, loader, optimizer, scheduler, config,
            num_epochs=1, log_interval=100, device="cpu",
        )
        assert len(metrics) > 0
        assert all("loss" in m for m in metrics)


class TestCosineSchedule:
    def test_warmup(self):
        optimizer = torch.optim.SGD([torch.randn(1, requires_grad=True)], lr=1.0)
        scheduler = get_cosine_schedule(optimizer, num_warmup_steps=10, num_training_steps=100)

        lrs = []
        for _ in range(100):
            lrs.append(scheduler.get_last_lr()[0])
            optimizer.step()
            scheduler.step()

        # During warmup, LR should increase
        assert lrs[5] > lrs[0]
        # After warmup, LR should decrease
        assert lrs[-1] < lrs[10]
