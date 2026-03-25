"""Tests for OCR inference pipeline."""

import pytest
import torch

from mineru_diffusion.config import PRESET_CONFIGS
from mineru_diffusion.train import create_model_with_vision
from mineru_diffusion.tokenizer import OCRTokenizer
from mineru_diffusion.inference import OCRPipeline
from mineru_diffusion.scheduler import SchedulerConfig


@pytest.fixture
def pipeline():
    config = PRESET_CONFIGS["tiny"]
    # Use real CNN encoder so 4D image tensors work
    model = create_model_with_vision(config, image_size=64)
    tokenizer = OCRTokenizer(config.vocab_size)
    sched_config = SchedulerConfig(confidence_threshold=0.5, max_steps=5)
    return OCRPipeline(model, tokenizer, sched_config, device="cpu")


class TestOCRPipeline:
    def test_predict_returns_string(self, pipeline):
        image = torch.randn(3, 64, 64)
        result = pipeline.predict(image, max_seq_len=16, num_steps=3)
        assert isinstance(result, str)

    def test_predict_batch(self, pipeline):
        images = torch.randn(2, 3, 64, 64)
        results = pipeline.predict_batch(images, max_seq_len=16, num_steps=3)
        assert len(results) == 2
        assert all(isinstance(r, str) for r in results)

    def test_get_confidence_map(self, pipeline):
        image = torch.randn(3, 64, 64)
        tokens, confidences = pipeline.get_confidence_map(image, max_seq_len=16)
        assert tokens.shape == (16,)
        assert confidences.shape == (16,)
        assert (confidences >= 0).all()
        assert (confidences <= 1).all()

    def test_predict_4d_input(self, pipeline):
        image = torch.randn(1, 3, 64, 64)
        result = pipeline.predict(image, max_seq_len=16, num_steps=3)
        assert isinstance(result, str)
