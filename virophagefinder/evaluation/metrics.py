"""Evaluation metrics for binary classification."""

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


def compute_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute comprehensive classification metrics.

    Args:
        labels: Ground truth binary labels (0 or 1)
        probabilities: Predicted probabilities
        threshold: Classification threshold

    Returns:
        Dictionary of metric names to values
    """
    predictions = (probabilities >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "specificity": _specificity(labels, predictions),
        "f1": f1_score(labels, predictions, zero_division=0),
        "mcc": matthews_corrcoef(labels, predictions),
    }

    # AUC metrics (only if both classes present)
    if len(np.unique(labels)) > 1:
        metrics["auc_roc"] = roc_auc_score(labels, probabilities)
        metrics["auc_pr"] = average_precision_score(labels, probabilities)
    else:
        metrics["auc_roc"] = 0.0
        metrics["auc_pr"] = 0.0

    # Confusion matrix
    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    metrics["tn"] = int(cm[0, 0])
    metrics["fp"] = int(cm[0, 1])
    metrics["fn"] = int(cm[1, 0])
    metrics["tp"] = int(cm[1, 1])

    return metrics


def _specificity(labels: np.ndarray, predictions: np.ndarray) -> float:
    """Compute specificity (true negative rate)."""
    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp = cm[0, 0], cm[0, 1]
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def stratified_evaluation(
    labels: np.ndarray,
    probabilities: np.ndarray,
    stratify_by: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Dict[str, float]]:
    """Evaluate metrics stratified by a grouping variable.

    Args:
        labels: Ground truth labels
        probabilities: Predicted probabilities
        stratify_by: Array of group identifiers (e.g., length bins, ecosystem)
        threshold: Classification threshold

    Returns:
        Dictionary mapping group name to metrics dictionary
    """
    results = {}
    unique_groups = np.unique(stratify_by)

    for group in unique_groups:
        mask = stratify_by == group
        if mask.sum() < 5:  # Skip groups with too few samples
            continue
        group_metrics = compute_metrics(
            labels[mask], probabilities[mask], threshold
        )
        group_metrics["n_samples"] = int(mask.sum())
        results[str(group)] = group_metrics

    return results


def print_metrics(metrics: Dict[str, float], title: str = "Metrics"):
    """Pretty-print metrics dictionary."""
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key:15s}: {value:.4f}")
        else:
            print(f"  {key:15s}: {value}")
    print(f"{'='*50}\n")


def length_bin_evaluation(
    lengths: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    bins: Optional[List[int]] = None,
) -> Dict[str, Dict[str, float]]:
    """Evaluate metrics stratified by sequence length bins.

    Args:
        lengths: Sequence lengths
        labels: Ground truth labels
        probabilities: Predicted probabilities
        bins: Length bin boundaries (default: [500, 1000, 3000, 5000, 10000, inf])

    Returns:
        Dictionary mapping bin label to metrics
    """
    if bins is None:
        bins = [500, 1000, 3000, 5000, 10000, float("inf")]

    bin_labels = []
    for length in lengths:
        for i, upper in enumerate(bins):
            lower = bins[i - 1] if i > 0 else 0
            if lower <= length < upper:
                if upper == float("inf"):
                    bin_labels.append(f">{lower}bp")
                else:
                    bin_labels.append(f"{lower}-{upper}bp")
                break

    return stratified_evaluation(labels, probabilities, np.array(bin_labels))


def find_optimal_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    metric: str = "f1",
) -> Tuple[float, float]:
    """Find optimal classification threshold maximizing a given metric.

    Args:
        labels: Ground truth labels
        probabilities: Predicted probabilities
        metric: Metric to maximize ('f1', 'mcc', 'youden')

    Returns:
        (optimal_threshold, best_metric_value)
    """
    thresholds = np.arange(0.05, 0.95, 0.01)
    best_value = -1
    best_threshold = 0.5

    for t in thresholds:
        preds = (probabilities >= t).astype(int)
        if metric == "f1":
            value = f1_score(labels, preds, zero_division=0)
        elif metric == "mcc":
            value = matthews_corrcoef(labels, preds)
        elif metric == "youden":
            value = recall_score(labels, preds) + _specificity(labels, preds) - 1
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if value > best_value:
            best_value = value
            best_threshold = t

    return best_threshold, best_value
