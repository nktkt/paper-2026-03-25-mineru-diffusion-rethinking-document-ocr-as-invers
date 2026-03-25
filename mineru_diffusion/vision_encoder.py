"""Vision encoder for MinerU-Diffusion.

The paper uses Qwen2-VL-7B as the vision encoder. This module provides:
1. A lightweight CNN-based encoder for local testing/training
2. A ViT-style patch encoder for medium-scale experiments
3. A HuggingFace adapter for plugging in pretrained VL models

All encoders produce [batch, num_visual_tokens, hidden_dim] features
that are prepended to the decoder input.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class PatchEmbedding(nn.Module):
    """Convert image into patch embeddings (ViT-style)."""

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 256,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert images to patch embeddings.

        Args:
            x: Images [batch, channels, height, width].

        Returns:
            Patch embeddings [batch, num_patches, embed_dim].
        """
        # x: [B, C, H, W] -> [B, D, H/P, W/P]
        x = self.proj(x)
        B, D, H, W = x.shape
        # Flatten spatial dims: [B, D, H*W] -> [B, H*W, D]
        x = x.flatten(2).transpose(1, 2)
        return x


class CNNVisionEncoder(nn.Module):
    """Lightweight CNN-based vision encoder for local experiments.

    Architecture: 4-layer CNN with progressive downsampling,
    followed by adaptive pooling to produce fixed-count visual tokens.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 256,
        num_visual_tokens: int = 64,
    ):
        super().__init__()
        self.num_visual_tokens = num_visual_tokens

        self.features = nn.Sequential(
            # Layer 1: 3 -> 64
            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            # Layer 2: 64 -> 128
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            # Layer 3: 128 -> 256
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            # Layer 4: 256 -> hidden_dim
            nn.Conv2d(256, hidden_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )

        # Adaptive pool to get fixed number of spatial tokens
        grid_size = int(math.sqrt(num_visual_tokens))
        self.pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))
        self.actual_tokens = grid_size * grid_size

        # Project to match decoder dim if needed
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images into visual token features.

        Args:
            images: Input images [batch, channels, height, width].
                Accepts any spatial size.

        Returns:
            Visual features [batch, num_visual_tokens, hidden_dim].
        """
        features = self.features(images)  # [B, D, H', W']
        features = self.pool(features)  # [B, D, grid, grid]
        B, D, H, W = features.shape
        features = features.flatten(2).transpose(1, 2)  # [B, H*W, D]
        features = self.proj(features)
        return features


class ViTVisionEncoder(nn.Module):
    """ViT-style vision encoder with self-attention over patches.

    A small Vision Transformer that processes document images into
    visual tokens for conditioning the diffusion decoder.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, hidden_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, hidden_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    @property
    def num_visual_tokens(self) -> int:
        return self.patch_embed.num_patches

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images via ViT.

        Args:
            images: Input images [batch, channels, image_size, image_size].

        Returns:
            Visual features [batch, num_patches, hidden_dim].
        """
        x = self.patch_embed(images)  # [B, N, D]
        x = x + self.pos_embed
        x = self.encoder(x)
        x = self.norm(x)
        return x


class DynamicResolutionEncoder(nn.Module):
    """Dynamic-resolution vision encoder inspired by NaViT.

    Handles variable-size document images by:
    1. Splitting into fixed-size patches
    2. Encoding patches independently
    3. Pooling to a fixed budget of visual tokens (paper uses up to 2048)
    """

    def __init__(
        self,
        patch_size: int = 16,
        in_channels: int = 3,
        hidden_dim: int = 256,
        max_visual_tokens: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.max_visual_tokens = max_visual_tokens
        self.hidden_dim = hidden_dim

        self.patch_proj = nn.Conv2d(
            in_channels, hidden_dim,
            kernel_size=patch_size, stride=patch_size,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode variable-resolution images.

        Args:
            images: Input images [batch, channels, height, width].
                Height and width must be divisible by patch_size.

        Returns:
            Visual features [batch, num_tokens, hidden_dim].
                num_tokens = min(H/P * W/P, max_visual_tokens).
        """
        patches = self.patch_proj(images)  # [B, D, H/P, W/P]
        B, D, H, W = patches.shape
        num_patches = H * W

        x = patches.flatten(2).transpose(1, 2)  # [B, N, D]

        # Truncate to max_visual_tokens if needed
        if num_patches > self.max_visual_tokens:
            # Adaptive average pool along sequence dimension
            x = x.transpose(1, 2)  # [B, D, N]
            x = F.adaptive_avg_pool1d(x, self.max_visual_tokens)
            x = x.transpose(1, 2)  # [B, max_tokens, D]

        x = self.encoder(x)
        x = self.norm(x)
        return x
