"""Tests for block-attention mechanism."""

import pytest
import torch

from mineru_diffusion.block_attention import (
    create_block_attention_mask,
    create_block_attention_bias,
    BlockAttention,
)


class TestBlockAttentionMask:
    def test_shape(self):
        mask = create_block_attention_mask(16, block_size=4)
        assert mask.shape == (16, 16)

    def test_within_block_bidirectional(self):
        mask = create_block_attention_mask(8, block_size=4)
        # First block (positions 0-3): all attend to each other
        assert mask[0, 3] == 1  # pos 0 can see pos 3
        assert mask[3, 0] == 1  # pos 3 can see pos 0
        # Second block (positions 4-7): all attend to each other
        assert mask[4, 7] == 1
        assert mask[7, 4] == 1

    def test_causal_across_blocks(self):
        mask = create_block_attention_mask(12, block_size=4)
        # Block 1 (pos 4-7) can attend to block 0 (pos 0-3)
        assert mask[4, 0] == 1
        assert mask[7, 3] == 1
        # Block 0 cannot attend to block 1
        assert mask[0, 4] == 0
        assert mask[3, 7] == 0

    def test_future_block_masked(self):
        mask = create_block_attention_mask(12, block_size=4)
        # Block 0 cannot attend to block 2
        assert mask[0, 8] == 0
        # Block 1 cannot attend to block 2
        assert mask[4, 8] == 0

    def test_non_divisible_length(self):
        # seq_len=10, block_size=4 -> blocks of [4, 4, 2]
        mask = create_block_attention_mask(10, block_size=4)
        assert mask.shape == (10, 10)
        # Last block (pos 8-9) can attend to previous blocks
        assert mask[8, 0] == 1
        assert mask[9, 7] == 1
        # Previous blocks cannot attend to last block
        assert mask[0, 8] == 0

    def test_block_size_1(self):
        # block_size=1 = fully causal
        mask = create_block_attention_mask(4, block_size=1)
        expected = torch.tensor([
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 1],
        ], dtype=torch.float32)
        assert torch.equal(mask, expected)

    def test_block_size_equals_seq_len(self):
        # block_size=seq_len = fully bidirectional
        mask = create_block_attention_mask(4, block_size=4)
        assert (mask == 1).all()

    def test_diagonal_is_one(self):
        mask = create_block_attention_mask(16, block_size=4)
        assert (mask.diag() == 1).all()


class TestBlockAttentionBias:
    def test_shape(self):
        bias = create_block_attention_bias(8, block_size=4)
        assert bias.shape == (1, 1, 8, 8)

    def test_values(self):
        bias = create_block_attention_bias(8, block_size=4)
        # Attended positions should be 0
        assert bias[0, 0, 0, 0] == 0
        # Masked positions should be -inf
        assert bias[0, 0, 0, 4] == float("-inf")


class TestBlockAttentionModule:
    def test_output_shape(self):
        attn = BlockAttention(hidden_dim=64, num_heads=4, block_size=4)
        x = torch.randn(2, 16, 64)
        out = attn(x)
        assert out.shape == x.shape

    def test_gradient_flow(self):
        attn = BlockAttention(hidden_dim=64, num_heads=4, block_size=4)
        x = torch.randn(2, 8, 64, requires_grad=True)
        out = attn(x)
        out.sum().backward()
        assert x.grad is not None

    def test_non_divisible_seq(self):
        attn = BlockAttention(hidden_dim=64, num_heads=4, block_size=4)
        x = torch.randn(2, 10, 64)  # 10 not divisible by 4
        out = attn(x)
        assert out.shape == (2, 10, 64)
