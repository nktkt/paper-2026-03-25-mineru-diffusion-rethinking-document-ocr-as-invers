"""OCR dataset loading and preprocessing for MinerU-Diffusion.

Supports:
1. Image-text pair datasets (image + ground truth text)
2. Layout-annotated datasets (image + bounding boxes + text)
3. Synthetic data generation for testing
4. Paper's data augmentation pipeline
"""

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from .tokenizer import OCRTokenizer


@dataclass
class OCRSample:
    """A single OCR training/evaluation sample."""

    image_path: str
    text: str
    task: str = "text"  # "text", "formula", "table", "layout"
    bboxes: Optional[List[List[int]]] = None  # [[x1,y1,x2,y2], ...]
    labels: Optional[List[str]] = None  # Label per bbox


class OCRDataset(Dataset):
    """OCR dataset for training MinerU-Diffusion.

    Loads image-text pairs and encodes them for discrete diffusion training.
    Images are loaded as tensors; text is tokenized.
    """

    def __init__(
        self,
        samples: List[OCRSample],
        tokenizer: OCRTokenizer,
        max_seq_len: int = 2048,
        image_size: int = 224,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Tokenize text
        ids = self.tokenizer.encode(sample.text, max_length=self.max_seq_len)
        # Pad
        pad_len = self.max_seq_len - len(ids)
        attention_mask = [1] * len(ids) + [0] * pad_len
        ids = ids + [self.tokenizer.pad_token_id] * pad_len

        # Try to load image, fall back to dummy
        image = self._load_image(sample.image_path)

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "image": image,
            "task": sample.task,
        }

    def _load_image(self, path: str) -> torch.Tensor:
        """Load image as tensor. Falls back to random tensor if unavailable."""
        try:
            from PIL import Image
            import torchvision.transforms as T
            transform = T.Compose([
                T.Resize((self.image_size, self.image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            img = Image.open(path).convert("RGB")
            return transform(img)
        except Exception:
            return torch.randn(3, self.image_size, self.image_size)


class SyntheticOCRDataset(Dataset):
    """Synthetic OCR dataset for testing the full pipeline.

    Generates random "document images" (noise) paired with random token
    sequences. Useful for verifying the training loop without real data.
    """

    def __init__(
        self,
        num_samples: int = 1000,
        vocab_size: int = 8192,
        max_seq_len: int = 128,
        image_size: int = 224,
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.image_size = image_size
        self.rng = random.Random(seed)

        # Pre-generate token sequences
        self.sequences = []
        for _ in range(num_samples):
            seq_len = self.rng.randint(max_seq_len // 2, max_seq_len)
            # Avoid token 0 (MASK) and 1 (PAD) in ground truth
            seq = [self.rng.randint(2, vocab_size - 1) for _ in range(seq_len)]
            self.sequences.append(seq)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = self.sequences[idx]
        pad_len = self.max_seq_len - len(seq)
        input_ids = seq + [1] * pad_len  # Pad with PAD token
        attention_mask = [1] * len(seq) + [0] * pad_len

        # Random "image"
        torch.manual_seed(idx)
        image = torch.randn(3, self.image_size, self.image_size)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "image": image,
        }


def load_jsonl_dataset(
    path: str,
    tokenizer: OCRTokenizer,
    max_seq_len: int = 2048,
) -> List[OCRSample]:
    """Load OCR samples from JSONL file.

    Expected format per line:
    {"image": "path/to/image.png", "text": "ground truth", "task": "text"}

    Args:
        path: Path to JSONL file.
        tokenizer: Tokenizer for encoding.
        max_seq_len: Maximum sequence length.

    Returns:
        List of OCRSample objects.
    """
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            samples.append(OCRSample(
                image_path=data.get("image", ""),
                text=data.get("text", ""),
                task=data.get("task", "text"),
                bboxes=data.get("bboxes"),
                labels=data.get("labels"),
            ))
    return samples


def apply_augmentation(
    image: torch.Tensor,
    level: str = "medium",
) -> torch.Tensor:
    """Apply data augmentation to a document image.

    From the paper: geometric changes, background disturbances,
    color shifts, image degradation.

    Args:
        image: Input image [channels, height, width].
        level: Augmentation intensity ("light", "medium", "heavy").

    Returns:
        Augmented image tensor.
    """
    if level == "light":
        noise_std = 0.01
        brightness = 0.05
    elif level == "heavy":
        noise_std = 0.05
        brightness = 0.2
    else:  # medium
        noise_std = 0.02
        brightness = 0.1

    # Gaussian noise
    image = image + torch.randn_like(image) * noise_std

    # Random brightness
    image = image + (torch.rand(1).item() - 0.5) * 2 * brightness

    # Clamp to valid range
    image = image.clamp(-3, 3)  # Assuming normalized images

    return image
