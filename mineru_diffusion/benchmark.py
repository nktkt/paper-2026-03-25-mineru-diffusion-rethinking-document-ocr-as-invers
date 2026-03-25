"""Semantic Shuffle benchmark for evaluating visual grounding.

From the paper: The Semantic Shuffle benchmark disrupts semantic coherence
by shuffling words within text regions while preserving visual layout.
AR decoders degrade sharply (rely on linguistic priors), while diffusion
decoders remain robust (visual grounding).

Distortion levels control the fraction of words that are shuffled.
"""

import random
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class ShuffledSample:
    """A sample from the Semantic Shuffle benchmark."""

    original_text: str
    shuffled_text: str
    distortion_level: float
    words_shuffled: int
    total_words: int


def shuffle_words(
    text: str,
    distortion_level: float = 0.5,
    seed: Optional[int] = None,
) -> ShuffledSample:
    """Shuffle words in text at a given distortion level.

    Randomly selects a fraction of words (controlled by distortion_level)
    and shuffles their positions. Preserves punctuation attachment.

    Args:
        text: Input text to shuffle.
        distortion_level: Fraction of words to shuffle [0, 1].
        seed: Random seed for reproducibility.

    Returns:
        ShuffledSample with original and shuffled text.
    """
    rng = random.Random(seed)
    words = text.split()

    if len(words) <= 1 or distortion_level <= 0:
        return ShuffledSample(
            original_text=text,
            shuffled_text=text,
            distortion_level=distortion_level,
            words_shuffled=0,
            total_words=len(words),
        )

    num_to_shuffle = max(2, int(len(words) * distortion_level))
    num_to_shuffle = min(num_to_shuffle, len(words))

    # Select indices to shuffle
    indices = list(range(len(words)))
    selected = sorted(rng.sample(indices, num_to_shuffle))

    # Shuffle the selected words
    selected_words = [words[i] for i in selected]
    rng.shuffle(selected_words)

    shuffled = words.copy()
    for i, idx in enumerate(selected):
        shuffled[idx] = selected_words[i]

    shuffled_text = " ".join(shuffled)
    words_changed = sum(1 for a, b in zip(words, shuffled) if a != b)

    return ShuffledSample(
        original_text=text,
        shuffled_text=shuffled_text,
        distortion_level=distortion_level,
        words_shuffled=words_changed,
        total_words=len(words),
    )


def create_benchmark_dataset(
    texts: List[str],
    distortion_levels: Optional[List[float]] = None,
    seed: int = 42,
) -> List[ShuffledSample]:
    """Create a Semantic Shuffle benchmark dataset.

    For each text and distortion level, creates a shuffled version.

    Args:
        texts: Input texts.
        distortion_levels: List of distortion levels to test.
        seed: Base random seed.

    Returns:
        List of ShuffledSample objects.
    """
    if distortion_levels is None:
        distortion_levels = [0.0, 0.25, 0.5, 0.75, 1.0]

    samples = []
    for i, text in enumerate(texts):
        for level in distortion_levels:
            sample = shuffle_words(text, level, seed=seed + i)
            samples.append(sample)

    return samples


def evaluate_robustness(
    predictions: List[str],
    references: List[str],
) -> dict:
    """Evaluate OCR robustness via character-level accuracy.

    Args:
        predictions: Predicted texts.
        references: Ground-truth texts (original, pre-shuffle).

    Returns:
        Dictionary with accuracy metrics.
    """
    if not predictions:
        return {"char_accuracy": 0.0, "exact_match": 0.0, "count": 0}

    total_chars = 0
    correct_chars = 0
    exact_matches = 0

    for pred, ref in zip(predictions, references):
        # Character-level accuracy
        max_len = max(len(pred), len(ref))
        if max_len == 0:
            exact_matches += 1
            continue

        matches = sum(1 for a, b in zip(pred, ref) if a == b)
        correct_chars += matches
        total_chars += max_len

        if pred == ref:
            exact_matches += 1

    return {
        "char_accuracy": correct_chars / max(1, total_chars),
        "exact_match": exact_matches / len(predictions),
        "count": len(predictions),
    }
