"""Binary classification metrics and threshold selection for fire / no_fire
(positive class = "fire", label 1).
"""

import numpy as np
from sklearn.metrics import roc_auc_score


def fire_probs(logits: np.ndarray) -> np.ndarray:
    """Softmax over the class axis, returning P(fire) for each example."""
    exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return exp[:, 1] / exp.sum(axis=-1)


def binary_metrics(preds: np.ndarray, labels: np.ndarray, probs: np.ndarray | None = None) -> dict:
    """Compute accuracy, precision, recall, f1, and optionally roc_auc."""
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    metrics = {"accuracy": (tp + tn) / len(labels), "precision": precision, "recall": recall, "f1": f1}
    if probs is not None:
        metrics["roc_auc"] = float(roc_auc_score(labels, probs))
    return metrics


def find_best_threshold(probs: np.ndarray, labels: np.ndarray) -> tuple[float, dict]:
    """Sweep classification thresholds and return the one maximizing F1 on
    (probs, labels), together with the metrics at that threshold.
    """
    best_threshold, best_metrics = 0.5, binary_metrics((probs >= 0.5).astype(int), labels)
    for threshold in np.arange(0.01, 1.0, 0.01):
        metrics = binary_metrics((probs >= threshold).astype(int), labels)
        if metrics["f1"] > best_metrics["f1"]:
            best_threshold, best_metrics = round(float(threshold), 2), metrics
    return best_threshold, best_metrics
