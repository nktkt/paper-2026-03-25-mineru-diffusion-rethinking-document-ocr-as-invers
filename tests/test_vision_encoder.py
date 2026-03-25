"""Tests for vision encoders."""

import pytest
import torch

from mineru_diffusion.vision_encoder import (
    PatchEmbedding, CNNVisionEncoder, ViTVisionEncoder, DynamicResolutionEncoder,
)


class TestPatchEmbedding:
    def test_output_shape(self):
        pe = PatchEmbedding(image_size=224, patch_size=16, embed_dim=256)
        x = torch.randn(2, 3, 224, 224)
        out = pe(x)
        assert out.shape == (2, 196, 256)  # 14*14 patches

    def test_num_patches(self):
        pe = PatchEmbedding(image_size=128, patch_size=16)
        assert pe.num_patches == 64  # 8*8


class TestCNNVisionEncoder:
    def test_output_shape(self):
        enc = CNNVisionEncoder(hidden_dim=128, num_visual_tokens=64)
        x = torch.randn(2, 3, 224, 224)
        out = enc(x)
        assert out.shape[0] == 2
        assert out.shape[2] == 128

    def test_variable_input_size(self):
        enc = CNNVisionEncoder(hidden_dim=128, num_visual_tokens=64)
        for size in [112, 224, 448]:
            x = torch.randn(1, 3, size, size)
            out = enc(x)
            assert out.shape[0] == 1
            assert out.shape[2] == 128

    def test_gradient_flow(self):
        enc = CNNVisionEncoder(hidden_dim=64, num_visual_tokens=16)
        x = torch.randn(1, 3, 112, 112, requires_grad=True)
        out = enc(x)
        out.sum().backward()
        assert x.grad is not None


class TestViTVisionEncoder:
    def test_output_shape(self):
        enc = ViTVisionEncoder(image_size=128, patch_size=16, hidden_dim=128, num_layers=2)
        x = torch.randn(2, 3, 128, 128)
        out = enc(x)
        assert out.shape == (2, 64, 128)  # 8*8 patches

    def test_num_visual_tokens(self):
        enc = ViTVisionEncoder(image_size=64, patch_size=8)
        assert enc.num_visual_tokens == 64


class TestDynamicResolutionEncoder:
    def test_small_image(self):
        enc = DynamicResolutionEncoder(patch_size=16, hidden_dim=128,
                                        max_visual_tokens=64, num_layers=2)
        x = torch.randn(1, 3, 64, 64)  # 4*4=16 patches < 64
        out = enc(x)
        assert out.shape == (1, 16, 128)

    def test_large_image_truncated(self):
        enc = DynamicResolutionEncoder(patch_size=16, hidden_dim=128,
                                        max_visual_tokens=64, num_layers=2)
        x = torch.randn(1, 3, 256, 256)  # 16*16=256 patches > 64
        out = enc(x)
        assert out.shape == (1, 64, 128)
