"""Tests for Semantic Shuffle benchmark."""

import pytest

from mineru_diffusion.benchmark import (
    shuffle_words, create_benchmark_dataset, evaluate_robustness, ShuffledSample,
)


class TestShuffleWords:
    def test_no_distortion(self):
        result = shuffle_words("hello world", distortion_level=0.0)
        assert result.shuffled_text == "hello world"
        assert result.words_shuffled == 0

    def test_full_distortion(self):
        result = shuffle_words("a b c d e f g h", distortion_level=1.0, seed=42)
        assert result.total_words == 8
        assert result.shuffled_text != "a b c d e f g h" or result.words_shuffled == 0

    def test_single_word(self):
        result = shuffle_words("hello", distortion_level=1.0)
        assert result.shuffled_text == "hello"
        assert result.words_shuffled == 0

    def test_deterministic(self):
        r1 = shuffle_words("the quick brown fox jumps", 0.5, seed=42)
        r2 = shuffle_words("the quick brown fox jumps", 0.5, seed=42)
        assert r1.shuffled_text == r2.shuffled_text

    def test_preserves_word_set(self):
        text = "alpha beta gamma delta"
        result = shuffle_words(text, distortion_level=1.0, seed=42)
        assert set(result.shuffled_text.split()) == set(text.split())

    def test_distortion_level_stored(self):
        result = shuffle_words("test text", distortion_level=0.75)
        assert result.distortion_level == 0.75

    def test_empty_text(self):
        result = shuffle_words("", distortion_level=0.5)
        assert result.shuffled_text == ""


class TestCreateBenchmarkDataset:
    def test_default_levels(self):
        texts = ["hello world", "test sentence"]
        samples = create_benchmark_dataset(texts)
        # 2 texts * 5 default levels = 10 samples
        assert len(samples) == 10

    def test_custom_levels(self):
        texts = ["a b c d"]
        samples = create_benchmark_dataset(texts, distortion_levels=[0.0, 0.5])
        assert len(samples) == 2

    def test_all_samples_valid(self):
        texts = ["the quick brown fox"]
        samples = create_benchmark_dataset(texts)
        for s in samples:
            assert isinstance(s, ShuffledSample)
            assert s.total_words > 0


class TestEvaluateRobustness:
    def test_perfect_match(self):
        preds = ["hello world", "test"]
        refs = ["hello world", "test"]
        metrics = evaluate_robustness(preds, refs)
        assert metrics["exact_match"] == 1.0
        assert metrics["char_accuracy"] == 1.0

    def test_no_match(self):
        preds = ["zzzzz"]
        refs = ["aaaaa"]
        metrics = evaluate_robustness(preds, refs)
        assert metrics["exact_match"] == 0.0
        assert metrics["char_accuracy"] == 0.0

    def test_partial_match(self):
        preds = ["hell"]
        refs = ["hello"]
        metrics = evaluate_robustness(preds, refs)
        assert 0 < metrics["char_accuracy"] < 1

    def test_empty(self):
        metrics = evaluate_robustness([], [])
        assert metrics["count"] == 0
