from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans

ForegroundRule = Literal["mask_overlap", "bright", "dark"]


@dataclass(frozen=True)
class KMeansSegmentationResult:
    prediction: NDArray[np.bool_]
    labels: NDArray[np.int32]
    centers: NDArray[np.float64]
    foreground_label: int
    foreground_rule: str


@dataclass(frozen=True)
class FeatureKMeansSegmenter:
    n_clusters: int = 2
    foreground: ForegroundRule = "mask_overlap"
    random_state: int = 0
    name: str = "feature_kmeans"

    def segment(
        self,
        features: NDArray[np.generic],
        *,
        target: NDArray[np.generic] | None = None,
    ) -> KMeansSegmentationResult:
        feature_array = np.asarray(features, dtype=np.float32)
        if feature_array.ndim != 3:
            raise ValueError("features must have shape (height, width, channels)")
        if feature_array.shape[-1] < 1:
            raise ValueError("features must contain at least one channel")

        height, width, channel_count = feature_array.shape
        flattened = feature_array.reshape(-1, channel_count)
        model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init="auto")
        labels = model.fit_predict(flattened).reshape(height, width).astype(np.int32)
        foreground_label, rule = self._foreground_label(
            labels=labels,
            centers=model.cluster_centers_,
            target=target,
        )
        return KMeansSegmentationResult(
            prediction=labels == foreground_label,
            labels=labels,
            centers=model.cluster_centers_,
            foreground_label=int(foreground_label),
            foreground_rule=rule,
        )

    def _foreground_label(
        self,
        *,
        labels: NDArray[np.int32],
        centers: NDArray[np.float64],
        target: NDArray[np.generic] | None,
    ) -> tuple[int, str]:
        if self.foreground == "mask_overlap" and target is not None:
            truth = np.asarray(target) > 0
            if truth.shape != labels.shape:
                raise ValueError("target must match feature height and width")
            scores = [
                float(np.logical_and(labels == label, truth).sum())
                for label in range(self.n_clusters)
            ]
            return int(np.argmax(scores)), "mask_overlap_eval"

        intensity_channel = centers[:, 0]
        if self.foreground == "dark":
            return int(np.argmin(intensity_channel)), "dark_intensity"
        return int(np.argmax(intensity_channel)), "bright_intensity"
