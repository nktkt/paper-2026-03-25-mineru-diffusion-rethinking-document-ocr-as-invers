"""Tests for data loading utilities."""

import pytest
import torch

from mineru_diffusion.data import SyntheticOCRDataset, apply_augmentation


class TestSyntheticOCRDataset:
    def test_length(self):
        ds = SyntheticOCRDataset(num_samples=50, vocab_size=100, max_seq_len=32)
        assert len(ds) == 50

    def test_sample_shape(self):
        ds = SyntheticOCRDataset(num_samples=10, vocab_size=100,
                                  max_seq_len=32, image_size=64)
        sample = ds[0]
        assert sample["input_ids"].shape == (32,)
        assert sample["attention_mask"].shape == (32,)
        assert sample["image"].shape == (3, 64, 64)

    def test_no_mask_in_ground_truth(self):
        ds = SyntheticOCRDataset(num_samples=10, vocab_size=100, max_seq_len=32)
        sample = ds[0]
        ids = sample["input_ids"]
        mask = sample["attention_mask"]
        # Non-padded positions should not contain MASK (0) or PAD (1)
        active_ids = ids[mask.bool()]
        assert (active_ids >= 2).all()

    def test_deterministic(self):
        ds1 = SyntheticOCRDataset(num_samples=5, seed=42)
        ds2 = SyntheticOCRDataset(num_samples=5, seed=42)
        assert torch.equal(ds1[0]["input_ids"], ds2[0]["input_ids"])


class TestAugmentation:
    def test_output_shape(self):
        img = torch.randn(3, 64, 64)
        aug = apply_augmentation(img, level="medium")
        assert aug.shape == img.shape

    def test_levels(self):
        img = torch.zeros(3, 32, 32)
        for level in ["light", "medium", "heavy"]:
            aug = apply_augmentation(img, level=level)
            assert aug.shape == img.shape
