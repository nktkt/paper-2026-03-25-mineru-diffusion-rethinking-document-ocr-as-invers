"""Task-specific loss functions and metrics for MinerU-Diffusion.

Implements the metrics used in the paper for different OCR tasks:
- Text: Edit distance (normalized), CER, WER
- Layout: PageIoU
- Formula: CDM (Character Detection Metric)
- Table: TEDS (Tree Edit Distance based Similarity)

Also provides the uncertainty-weighted loss from curriculum learning:
    w(x) = 1 + beta * (1 - C(x))
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple


def edit_distance(pred: str, target: str) -> int:
    """Compute Levenshtein edit distance between two strings.

    Args:
        pred: Predicted string.
        target: Target string.

    Returns:
        Edit distance (int).
    """
    m, n = len(pred), len(target)
    dp = list(range(n + 1))

    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if pred[i - 1] == target[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp

    return dp[n]


def normalized_edit_distance(pred: str, target: str) -> float:
    """Compute normalized edit distance (0 = perfect, 1 = completely wrong).

    Args:
        pred: Predicted string.
        target: Target string.

    Returns:
        Normalized edit distance in [0, 1].
    """
    if not pred and not target:
        return 0.0
    max_len = max(len(pred), len(target))
    return edit_distance(pred, target) / max_len


def character_error_rate(pred: str, target: str) -> float:
    """Compute Character Error Rate (CER).

    CER = edit_distance(pred, target) / len(target)

    Args:
        pred: Predicted string.
        target: Target string.

    Returns:
        CER value. Can exceed 1.0 if pred is much longer than target.
    """
    if not target:
        return 0.0 if not pred else 1.0
    return edit_distance(pred, target) / len(target)


def word_error_rate(pred: str, target: str) -> float:
    """Compute Word Error Rate (WER).

    Args:
        pred: Predicted string.
        target: Target string.

    Returns:
        WER value.
    """
    pred_words = pred.split()
    target_words = target.split()
    if not target_words:
        return 0.0 if not pred_words else 1.0
    return edit_distance(" ".join(pred_words), " ".join(target_words)) / len(target_words)


def compute_iou(box1: List[int], box2: List[int]) -> float:
    """Compute Intersection over Union between two bounding boxes.

    Args:
        box1: [x1, y1, x2, y2].
        box2: [x1, y1, x2, y2].

    Returns:
        IoU value in [0, 1].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / max(union, 1e-8)


def page_iou(
    pred_boxes: List[List[int]],
    target_boxes: List[List[int]],
    iou_threshold: float = 0.5,
) -> float:
    """Compute PageIoU metric for layout detection.

    Matches predicted boxes to ground truth via IoU and computes
    the average IoU of matched pairs.

    Args:
        pred_boxes: Predicted bounding boxes [[x1,y1,x2,y2], ...].
        target_boxes: Ground truth bounding boxes.
        iou_threshold: Minimum IoU for a match.

    Returns:
        PageIoU score in [0, 1].
    """
    if not pred_boxes or not target_boxes:
        return 0.0

    matched_ious = []
    used_targets = set()

    for pred_box in pred_boxes:
        best_iou = 0.0
        best_idx = -1
        for j, target_box in enumerate(target_boxes):
            if j in used_targets:
                continue
            iou = compute_iou(pred_box, target_box)
            if iou > best_iou:
                best_iou = iou
                best_idx = j
        if best_iou >= iou_threshold and best_idx >= 0:
            matched_ious.append(best_iou)
            used_targets.add(best_idx)

    if not matched_ious:
        return 0.0

    # Average IoU of matched pairs, weighted by recall
    precision = len(matched_ious) / len(pred_boxes)
    recall = len(matched_ious) / len(target_boxes)
    avg_iou = sum(matched_ious) / len(matched_ious)

    return avg_iou * (2 * precision * recall / max(precision + recall, 1e-8))


def teds_similarity(pred_html: str, target_html: str) -> float:
    """Compute TEDS (Tree Edit Distance based Similarity) for tables.

    Simplified version: computes normalized edit distance on the
    HTML/OTSL string representation.

    NOTE: Full TEDS requires tree parsing. This is an approximation.

    Args:
        pred_html: Predicted table structure (HTML or OTSL).
        target_html: Ground truth table structure.

    Returns:
        TEDS score in [0, 1]. Higher = better.
    """
    return 1.0 - normalized_edit_distance(pred_html, target_html)


def compute_task_metrics(
    predictions: List[str],
    targets: List[str],
    task: str = "text",
) -> dict:
    """Compute task-specific metrics for a batch.

    Args:
        predictions: Predicted strings.
        targets: Ground truth strings.
        task: Task type ("text", "formula", "table", "layout").

    Returns:
        Dictionary with metric name -> value.
    """
    if not predictions:
        return {}

    if task == "text":
        cers = [character_error_rate(p, t) for p, t in zip(predictions, targets)]
        neds = [normalized_edit_distance(p, t) for p, t in zip(predictions, targets)]
        return {
            "cer": sum(cers) / len(cers),
            "ned": sum(neds) / len(neds),
            "accuracy": sum(1 for p, t in zip(predictions, targets) if p == t) / len(predictions),
        }
    elif task == "formula":
        cers = [character_error_rate(p, t) for p, t in zip(predictions, targets)]
        return {
            "cer": sum(cers) / len(cers),
            "cdm": 1.0 - sum(cers) / len(cers),  # CDM approximation
        }
    elif task == "table":
        teds_scores = [teds_similarity(p, t) for p, t in zip(predictions, targets)]
        return {
            "teds": sum(teds_scores) / len(teds_scores),
        }
    else:
        neds = [normalized_edit_distance(p, t) for p, t in zip(predictions, targets)]
        return {"ned": sum(neds) / len(neds)}
