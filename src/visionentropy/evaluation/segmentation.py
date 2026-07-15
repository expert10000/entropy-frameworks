from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def binary_metrics(
    prediction: NDArray[np.generic],
    target: NDArray[np.generic],
) -> dict[str, float]:
    predicted = np.asarray(prediction).astype(bool)
    truth = np.asarray(target) > 0
    if predicted.shape != truth.shape:
        raise ValueError("prediction and target must have the same shape")

    true_positive = float(np.logical_and(predicted, truth).sum())
    false_positive = float(np.logical_and(predicted, ~truth).sum())
    false_negative = float(np.logical_and(~predicted, truth).sum())
    true_negative = float(np.logical_and(~predicted, ~truth).sum())

    union = true_positive + false_positive + false_negative
    dice_denominator = (2.0 * true_positive) + false_positive + false_negative
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    total = true_positive + true_negative + false_positive + false_negative

    return {
        "mean_iou": _safe_divide(true_positive, union),
        "dice": _safe_divide(2.0 * true_positive, dice_denominator),
        "pixel_accuracy": _safe_divide(true_positive + true_negative, total),
        "precision": _safe_divide(true_positive, precision_denominator),
        "recall": _safe_divide(true_positive, recall_denominator),
    }


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 1.0
    return float(numerator / denominator)
