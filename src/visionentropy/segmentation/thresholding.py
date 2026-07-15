from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


def maximum_entropy_threshold(values: NDArray[np.generic], *, bins: int = 256) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("values cannot be empty")
    if np.isclose(array.min(), array.max()):
        return float(array.min())

    histogram, edges = np.histogram(array.ravel(), bins=bins)
    probabilities = histogram.astype(np.float64)
    probabilities /= probabilities.sum()
    cumulative = np.cumsum(probabilities)

    best_index = 0
    best_score = -np.inf
    for index in range(1, len(probabilities) - 1):
        p_background = probabilities[: index + 1]
        p_foreground = probabilities[index + 1 :]
        weight_background = cumulative[index]
        weight_foreground = 1.0 - weight_background
        if weight_background <= 0 or weight_foreground <= 0:
            continue

        h_background = _entropy(p_background / weight_background)
        h_foreground = _entropy(p_foreground / weight_foreground)
        score = h_background + h_foreground
        if score > best_score:
            best_score = score
            best_index = index

    return float(edges[best_index + 1])


@dataclass(frozen=True)
class MaximumEntropySegmenter:
    bins: int = 256
    foreground: str = "high"
    name: str = "maximum_entropy_threshold"

    def segment(self, values: NDArray[np.generic]) -> NDArray[np.bool_]:
        threshold = maximum_entropy_threshold(values, bins=self.bins)
        array = np.asarray(values, dtype=np.float32)
        if self.foreground == "high":
            return array >= threshold
        if self.foreground == "low":
            return array <= threshold
        raise ValueError("foreground must be 'high' or 'low'")


def _entropy(probabilities: NDArray[np.float64]) -> float:
    p = probabilities[probabilities > 0]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log(p)).sum())
