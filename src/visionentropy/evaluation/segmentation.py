from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from sklearn.metrics import roc_auc_score
from skimage.segmentation import find_boundaries


def binary_metrics(
    prediction: NDArray[np.generic],
    target: NDArray[np.generic],
    *,
    score_map: NDArray[np.generic] | None = None,
    boundary_tolerance: int = 2,
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
    specificity_denominator = true_negative + false_positive
    total = true_positive + true_negative + false_positive + false_negative

    return {
        "mean_iou": _safe_divide(true_positive, union),
        "dice": _safe_divide(2.0 * true_positive, dice_denominator),
        "boundary_f1": boundary_f1(predicted, truth, tolerance=boundary_tolerance),
        "error_detection_auroc": error_detection_auroc(predicted, truth, score_map),
        "pixel_accuracy": _safe_divide(true_positive + true_negative, total),
        "precision": _safe_divide(true_positive, precision_denominator),
        "recall": _safe_divide(true_positive, recall_denominator),
        "specificity": _safe_divide(true_negative, specificity_denominator),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
    }


def boundary_f1(
    prediction: NDArray[np.generic],
    target: NDArray[np.generic],
    *,
    tolerance: int = 2,
) -> float:
    predicted_boundary = find_boundaries(np.asarray(prediction).astype(bool), mode="outer")
    target_boundary = find_boundaries(np.asarray(target).astype(bool), mode="outer")
    if not predicted_boundary.any() and not target_boundary.any():
        return 1.0
    if not predicted_boundary.any() or not target_boundary.any():
        return 0.0

    structure = np.ones(((tolerance * 2) + 1, (tolerance * 2) + 1), dtype=bool)
    target_band = ndimage.binary_dilation(target_boundary, structure=structure)
    predicted_band = ndimage.binary_dilation(predicted_boundary, structure=structure)
    boundary_precision = _safe_divide(
        float(np.logical_and(predicted_boundary, target_band).sum()),
        float(predicted_boundary.sum()),
    )
    boundary_recall = _safe_divide(
        float(np.logical_and(target_boundary, predicted_band).sum()),
        float(target_boundary.sum()),
    )
    return _safe_divide(
        2.0 * boundary_precision * boundary_recall,
        boundary_precision + boundary_recall,
    )


def error_detection_auroc(
    prediction: NDArray[np.generic],
    target: NDArray[np.generic],
    score_map: NDArray[np.generic] | None,
) -> float:
    if score_map is None:
        return 0.5
    error_pixels = np.asarray(prediction).astype(bool) != np.asarray(target).astype(bool)
    if np.unique(error_pixels).size < 2:
        return 1.0
    scores = _zero_one(np.asarray(score_map, dtype=np.float32)).ravel()
    return float(roc_auc_score(error_pixels.ravel().astype(np.uint8), scores))


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 1.0
    return float(numerator / denominator)


def _zero_one(values: NDArray[np.generic]) -> NDArray[np.float32]:
    array = np.asarray(values, dtype=np.float32)
    min_value = float(array.min())
    max_value = float(array.max())
    if np.isclose(min_value, max_value):
        return np.zeros_like(array, dtype=np.float32)
    return ((array - min_value) / (max_value - min_value)).astype(np.float32)
