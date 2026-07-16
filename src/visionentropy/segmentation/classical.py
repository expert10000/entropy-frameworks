from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from skimage import filters
from skimage.segmentation import watershed
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture

ForegroundRule = Literal["mask_overlap", "bright", "dark", "high", "low"]


@dataclass(frozen=True)
class ClassicalSegmentationResult:
    prediction: NDArray[np.bool_]
    score_map: NDArray[np.float32]
    labels: NDArray[np.int32] | None = None
    centers: NDArray[np.float64] | None = None
    foreground_label: int | None = None
    foreground_rule: str | None = None
    threshold: float | None = None


@dataclass(frozen=True)
class OtsuSegmenter:
    foreground: ForegroundRule = "high"
    name: str = "otsu"

    def segment(self, values: NDArray[np.generic]) -> ClassicalSegmentationResult:
        score = _zero_one(values)
        threshold = _safe_otsu(score)
        prediction = _threshold(score, threshold, self.foreground)
        return ClassicalSegmentationResult(
            prediction=prediction,
            score_map=score,
            threshold=threshold,
            foreground_rule=_threshold_rule(self.foreground),
        )


@dataclass(frozen=True)
class AdaptiveThresholdSegmenter:
    window_radius: int = 4
    foreground: ForegroundRule = "high"
    name: str = "local_adaptive"

    def segment(self, values: NDArray[np.generic]) -> ClassicalSegmentationResult:
        score = _zero_one(values)
        block_size = adaptive_block_size(score.shape, self.window_radius)
        threshold_map = filters.threshold_local(score, block_size=block_size)
        if self.foreground in {"low", "dark"}:
            prediction = score <= threshold_map
        else:
            prediction = score >= threshold_map
        return ClassicalSegmentationResult(
            prediction=prediction,
            score_map=score,
            threshold=float(np.mean(threshold_map)),
            foreground_rule=f"local_{_threshold_rule(self.foreground)}",
        )


@dataclass(frozen=True)
class GaussianMixtureSegmenter:
    n_components: int = 2
    foreground: ForegroundRule = "mask_overlap"
    random_state: int = 0
    name: str = "gaussian_mixture"

    def segment(
        self,
        features: NDArray[np.generic],
        *,
        target: NDArray[np.generic] | None = None,
    ) -> ClassicalSegmentationResult:
        feature_array = _feature_array(features)
        height, width, channel_count = feature_array.shape
        flattened = feature_array.reshape(-1, channel_count)
        model = GaussianMixture(n_components=self.n_components, random_state=self.random_state)
        labels = model.fit_predict(flattened).reshape(height, width).astype(np.int32)
        probabilities = model.predict_proba(flattened).reshape(height, width, self.n_components)
        foreground_label, rule = _foreground_label(
            labels=labels,
            centers=model.means_,
            n_labels=self.n_components,
            foreground=self.foreground,
            target=target,
        )
        return ClassicalSegmentationResult(
            prediction=labels == foreground_label,
            score_map=probabilities[..., foreground_label].astype(np.float32),
            labels=labels,
            centers=model.means_,
            foreground_label=foreground_label,
            foreground_rule=rule,
        )


@dataclass(frozen=True)
class RandomForestSegmenter:
    n_estimators: int = 80
    max_train_pixels: int = 6000
    random_state: int = 0
    name: str = "random_forest"

    def segment(
        self,
        features: NDArray[np.generic],
        *,
        target: NDArray[np.generic] | None = None,
    ) -> ClassicalSegmentationResult:
        feature_array = _feature_array(features)
        height, width, channel_count = feature_array.shape
        flattened = feature_array.reshape(-1, channel_count)
        training_target, rule = _training_labels(feature_array[..., 0], target)
        training_flat = training_target.reshape(-1)
        sample_indices = _balanced_sample_indices(training_flat, self.max_train_pixels, self.random_state)
        model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            class_weight="balanced",
            min_samples_leaf=2,
            n_jobs=1,
        )
        model.fit(flattened[sample_indices], training_flat[sample_indices])
        foreground_index = int(np.where(model.classes_ == 1)[0][0]) if 1 in model.classes_ else 0
        probabilities = model.predict_proba(flattened)[:, foreground_index].reshape(height, width)
        prediction = probabilities >= 0.5
        return ClassicalSegmentationResult(
            prediction=prediction,
            score_map=probabilities.astype(np.float32),
            foreground_label=1,
            foreground_rule=rule,
        )


@dataclass(frozen=True)
class WatershedSegmenter:
    foreground: ForegroundRule = "high"
    name: str = "watershed"

    def segment(
        self,
        values: NDArray[np.generic],
        *,
        gradient_map: NDArray[np.generic] | None = None,
    ) -> ClassicalSegmentationResult:
        score = _zero_one(values)
        gradient = _zero_one(gradient_map) if gradient_map is not None else _zero_one(filters.sobel(score))
        threshold = _safe_otsu(score)
        markers = np.zeros(score.shape, dtype=np.int32)
        if self.foreground in {"low", "dark"}:
            markers[score >= np.quantile(score, 0.75)] = 1
            markers[score <= threshold] = 2
        else:
            markers[score <= np.quantile(score, 0.25)] = 1
            markers[score >= threshold] = 2
        if np.count_nonzero(markers == 2) == 0:
            markers[score == score.max()] = 2
        if np.count_nonzero(markers == 1) == 0:
            markers[score == score.min()] = 1
        labels = watershed(gradient, markers=markers).astype(np.int32)
        return ClassicalSegmentationResult(
            prediction=labels == 2,
            score_map=score,
            labels=labels,
            foreground_label=2,
            foreground_rule="watershed_markers",
            threshold=threshold,
        )


@dataclass(frozen=True)
class RegionGrowingSegmenter:
    tolerance: float | None = None
    foreground: ForegroundRule = "high"
    name: str = "region_growing"

    def segment(self, values: NDArray[np.generic]) -> ClassicalSegmentationResult:
        score = _zero_one(values)
        threshold = _safe_otsu(score)
        if self.foreground in {"low", "dark"}:
            seed = np.unravel_index(int(np.argmin(score)), score.shape)
            tolerance = self.tolerance if self.tolerance is not None else _default_tolerance(score)
            candidate = score <= min(float(score[seed]) + tolerance, threshold)
        else:
            seed = np.unravel_index(int(np.argmax(score)), score.shape)
            tolerance = self.tolerance if self.tolerance is not None else _default_tolerance(score)
            candidate = score >= max(float(score[seed]) - tolerance, threshold)
        labels, _ = ndimage.label(candidate)
        seed_label = int(labels[seed])
        if seed_label == 0:
            prediction = candidate
        else:
            prediction = labels == seed_label
        prediction = ndimage.binary_fill_holes(prediction).astype(bool)
        return ClassicalSegmentationResult(
            prediction=prediction,
            score_map=score,
            labels=labels.astype(np.int32),
            foreground_label=seed_label,
            foreground_rule=f"seed_tolerance_{tolerance:.3f}",
            threshold=threshold,
        )


def adaptive_block_size(shape: tuple[int, ...], window_radius: int) -> int:
    smallest_axis = max(3, min(int(shape[0]), int(shape[1])))
    requested = max(3, (int(window_radius) * 2) + 1)
    block_size = min(requested, smallest_axis if smallest_axis % 2 == 1 else smallest_axis - 1)
    return max(3, block_size if block_size % 2 == 1 else block_size - 1)


def _threshold(values: NDArray[np.float32], threshold: float, foreground: ForegroundRule) -> NDArray[np.bool_]:
    if foreground in {"low", "dark"}:
        return values <= threshold
    return values >= threshold


def _threshold_rule(foreground: ForegroundRule) -> str:
    return "low_score" if foreground in {"low", "dark"} else "high_score"


def _safe_otsu(values: NDArray[np.generic]) -> float:
    array = np.asarray(values, dtype=np.float32)
    if np.isclose(float(array.min()), float(array.max())):
        return float(array.min())
    return float(filters.threshold_otsu(array))


def _feature_array(features: NDArray[np.generic]) -> NDArray[np.float32]:
    feature_array = np.asarray(features, dtype=np.float32)
    if feature_array.ndim != 3:
        raise ValueError("features must have shape (height, width, channels)")
    if feature_array.shape[-1] < 1:
        raise ValueError("features must contain at least one channel")
    return feature_array


def _foreground_label(
    *,
    labels: NDArray[np.int32],
    centers: NDArray[np.float64],
    n_labels: int,
    foreground: ForegroundRule,
    target: NDArray[np.generic] | None,
) -> tuple[int, str]:
    if foreground == "mask_overlap" and target is not None:
        truth = np.asarray(target) > 0
        if truth.shape != labels.shape:
            raise ValueError("target must match feature height and width")
        scores = [float(np.logical_and(labels == label, truth).sum()) for label in range(n_labels)]
        return int(np.argmax(scores)), "mask_overlap_eval"

    intensity_channel = centers[:, 0]
    if foreground in {"low", "dark"}:
        return int(np.argmin(intensity_channel)), "dark_intensity"
    return int(np.argmax(intensity_channel)), "bright_intensity"


def _training_labels(
    intensity: NDArray[np.float32],
    target: NDArray[np.generic] | None,
) -> tuple[NDArray[np.bool_], str]:
    if target is not None:
        truth = np.asarray(target) > 0
        if truth.shape == intensity.shape and np.unique(truth).size == 2:
            return truth, "mask_supervised_train"
    threshold = _safe_otsu(intensity)
    return intensity >= threshold, "otsu_pseudo_labels"


def _balanced_sample_indices(labels: NDArray[np.bool_], max_pixels: int, random_state: int) -> NDArray[np.int64]:
    rng = np.random.default_rng(random_state)
    positive = np.flatnonzero(labels)
    negative = np.flatnonzero(~labels)
    if positive.size == 0 or negative.size == 0:
        all_indices = np.arange(labels.size)
        count = min(max_pixels, all_indices.size)
        return rng.choice(all_indices, size=count, replace=False).astype(np.int64)

    per_class = max(1, min(max_pixels // 2, positive.size, negative.size))
    selected = np.concatenate(
        [
            rng.choice(positive, size=per_class, replace=False),
            rng.choice(negative, size=per_class, replace=False),
        ]
    )
    rng.shuffle(selected)
    return selected.astype(np.int64)


def _default_tolerance(values: NDArray[np.float32]) -> float:
    return float(np.clip(np.std(values) * 0.5, 0.08, 0.22))


def _zero_one(values: NDArray[np.generic]) -> NDArray[np.float32]:
    array = np.asarray(values, dtype=np.float32)
    min_value = float(array.min())
    max_value = float(array.max())
    if np.isclose(min_value, max_value):
        return np.zeros_like(array, dtype=np.float32)
    return ((array - min_value) / (max_value - min_value)).astype(np.float32)
