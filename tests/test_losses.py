"""Tests for loss functions and metrics."""

import pytest
from mineru_diffusion.losses import (
    edit_distance, normalized_edit_distance, character_error_rate,
    word_error_rate, compute_iou, page_iou, teds_similarity,
    compute_task_metrics,
)


class TestEditDistance:
    def test_identical(self):
        assert edit_distance("hello", "hello") == 0

    def test_empty(self):
        assert edit_distance("", "hello") == 5
        assert edit_distance("hello", "") == 5

    def test_substitution(self):
        assert edit_distance("cat", "car") == 1

    def test_insertion(self):
        assert edit_distance("cat", "cats") == 1

    def test_deletion(self):
        assert edit_distance("cats", "cat") == 1


class TestNormalizedEditDistance:
    def test_identical(self):
        assert normalized_edit_distance("hello", "hello") == 0.0

    def test_completely_different(self):
        ned = normalized_edit_distance("abc", "xyz")
        assert ned == 1.0

    def test_both_empty(self):
        assert normalized_edit_distance("", "") == 0.0


class TestCER:
    def test_perfect(self):
        assert character_error_rate("hello", "hello") == 0.0

    def test_empty_target(self):
        assert character_error_rate("something", "") == 1.0

    def test_empty_both(self):
        assert character_error_rate("", "") == 0.0


class TestWER:
    def test_perfect(self):
        assert word_error_rate("hello world", "hello world") == 0.0

    def test_one_word_wrong(self):
        wer = word_error_rate("hello earth", "hello world")
        assert wer > 0


class TestIoU:
    def test_identical_boxes(self):
        assert compute_iou([0, 0, 100, 100], [0, 0, 100, 100]) == 1.0

    def test_no_overlap(self):
        assert compute_iou([0, 0, 50, 50], [100, 100, 200, 200]) == 0.0

    def test_partial_overlap(self):
        iou = compute_iou([0, 0, 100, 100], [50, 50, 150, 150])
        assert 0 < iou < 1


class TestPageIoU:
    def test_perfect_match(self):
        boxes = [[0, 0, 100, 100], [200, 200, 300, 300]]
        score = page_iou(boxes, boxes)
        assert score == 1.0

    def test_no_match(self):
        pred = [[0, 0, 10, 10]]
        target = [[500, 500, 600, 600]]
        score = page_iou(pred, target)
        assert score == 0.0

    def test_empty(self):
        assert page_iou([], [[0, 0, 10, 10]]) == 0.0


class TestTEDS:
    def test_identical(self):
        assert teds_similarity("<fcel>A<nl>", "<fcel>A<nl>") == 1.0

    def test_different(self):
        score = teds_similarity("<fcel>A<nl>", "<fcel>B<nl>")
        assert 0 < score < 1


class TestComputeTaskMetrics:
    def test_text_metrics(self):
        metrics = compute_task_metrics(["hello", "world"], ["hello", "world"], "text")
        assert metrics["cer"] == 0.0
        assert metrics["accuracy"] == 1.0

    def test_formula_metrics(self):
        metrics = compute_task_metrics(["\\frac{1}{2}"], ["\\frac{1}{2}"], "formula")
        assert metrics["cer"] == 0.0

    def test_table_metrics(self):
        metrics = compute_task_metrics(["<fcel>A"], ["<fcel>A"], "table")
        assert metrics["teds"] == 1.0

    def test_empty(self):
        assert compute_task_metrics([], [], "text") == {}
