from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from skimage import filters
from skimage.io import imsave

from visionentropy.entropy import LocalEntropyMap
from visionentropy.evaluation import binary_metrics
from visionentropy.pipeline.vertical_slice import (
    _boundary_region_features,
    _error_map,
    _heatmap,
    _load_sample,
    _preprocess_sample,
    _to_uint8_rgb,
    _to_viewable_image,
    _zero_one,
)
from visionentropy.representations import build_representation
from visionentropy.segmentation import FeatureKMeansSegmenter, MaximumEntropySegmenter, maximum_entropy_threshold


def run_baseline_entropy_comparison(config: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    experiment = config.get("experiment", {})
    output_directory = Path(experiment.get("output_directory", "outputs/runs/comparison"))
    images_directory = output_directory / "images"
    data_directory = output_directory / "data"
    images_directory.mkdir(parents=True, exist_ok=True)
    data_directory.mkdir(parents=True, exist_ok=True)

    sample = _load_sample(config.get("dataset", {}))
    sample = _preprocess_sample(sample, config.get("preprocessing", {}), config.get("dataset", {}))
    target = np.asarray(sample.mask) > 0 if sample.mask is not None else None

    gray = build_representation("grayscale").transform(sample.image).data
    gray = _zero_one(np.asarray(gray, dtype=np.float32))
    entropy_config = config.get("entropy", {}).get("parameters", {})
    bins = int(entropy_config.get("bins", 64))
    window_radius = int(entropy_config.get("window_radius", 4))
    entropy_map = LocalEntropyMap(window_radius=window_radius, bins=bins).compute(gray).map
    entropy_score = _zero_one(entropy_map)
    gradient_score = _zero_one(filters.sobel(gray))
    feature_stack = _boundary_region_features(
        grayscale=gray,
        entropy_map=entropy_map,
        gradient_map=gradient_score,
    )
    feature_kmeans = FeatureKMeansSegmenter(foreground="mask_overlap", random_state=0).segment(
        feature_stack,
        target=target,
    )

    shared_artifacts = _save_shared_artifacts(
        images_directory=images_directory,
        sample=sample,
        gray=gray,
        entropy_map=entropy_map,
        gradient_score=gradient_score,
        target=target,
    )

    variants = [
        _variant(
            variant_id="baseline_a_grayscale_otsu",
            title="Baseline A: grayscale + Otsu",
            kind="baseline",
            description="Non-entropy intensity threshold.",
            score=gray,
            prediction=_threshold_otsu(gray),
            threshold=_safe_otsu(gray),
            images_directory=images_directory,
            target=target,
        ),
        _variant(
            variant_id="baseline_b_grayscale_adaptive",
            title="Baseline B: grayscale + adaptive threshold",
            kind="baseline",
            description="Non-entropy local intensity threshold.",
            score=gray,
            prediction=gray >= filters.threshold_local(gray, _adaptive_block_size(gray.shape, window_radius)),
            threshold=None,
            images_directory=images_directory,
            target=target,
        ),
        _variant(
            variant_id="experiment_c_local_shannon",
            title="Experiment C: local Shannon entropy only",
            kind="entropy",
            description="Entropy map segmented directly.",
            score=entropy_score,
            prediction=MaximumEntropySegmenter(bins=bins).segment(entropy_map),
            threshold=maximum_entropy_threshold(entropy_map, bins=bins),
            images_directory=images_directory,
            target=target,
        ),
        _variant(
            variant_id="experiment_d_grayscale_local_shannon",
            title="Experiment D: grayscale + local Shannon",
            kind="entropy",
            description="Intensity and entropy fused before thresholding.",
            score=_mean_score(gray, entropy_score),
            prediction=_threshold_otsu(_mean_score(gray, entropy_score)),
            threshold=_safe_otsu(_mean_score(gray, entropy_score)),
            images_directory=images_directory,
            target=target,
        ),
        _variant(
            variant_id="experiment_e_grayscale_gradient_local_shannon",
            title="Experiment E: feature stack + KMeans",
            kind="entropy",
            description="Intensity, boundary entropy, and gradient clustered into regions.",
            score=np.mean(feature_stack, axis=-1),
            prediction=feature_kmeans.prediction,
            threshold=None,
            images_directory=images_directory,
            target=target,
        ),
    ]

    best_variant_id = _best_variant_id(variants)
    payload = {
        "sampleId": sample.sample_id,
        "experiment": experiment.get("name", "comparison"),
        "outputDirectory": str(output_directory),
        "runtime": {"duration_seconds": time.perf_counter() - started},
        "parameters": {"bins": bins, "windowRadius": window_radius},
        "artifacts": shared_artifacts,
        "variants": variants,
        "bestVariantId": best_variant_id,
    }
    np.save(data_directory / "grayscale.npy", gray.astype(np.float32))
    np.save(data_directory / "entropy_map.npy", entropy_map.astype(np.float32))
    (output_directory / "comparison.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _save_shared_artifacts(
    *,
    images_directory: Path,
    sample: Any,
    gray: np.ndarray,
    entropy_map: np.ndarray,
    gradient_score: np.ndarray,
    target: np.ndarray | None,
) -> dict[str, str]:
    artifacts = {
        "original_image": str(images_directory / "original.png"),
        "grayscale": str(images_directory / "grayscale.png"),
        "entropy_map": str(images_directory / "entropy_map.png"),
        "gradient_map": str(images_directory / "gradient_map.png"),
    }
    imsave(artifacts["original_image"], _to_uint8_rgb(sample.image), check_contrast=False)
    imsave(artifacts["grayscale"], _to_viewable_image(gray), check_contrast=False)
    imsave(artifacts["entropy_map"], _heatmap(entropy_map), check_contrast=False)
    imsave(artifacts["gradient_map"], _heatmap(gradient_score), check_contrast=False)
    if target is not None:
        artifacts["ground_truth"] = str(images_directory / "ground_truth.png")
        imsave(artifacts["ground_truth"], (target.astype(np.uint8) * 255), check_contrast=False)
    return artifacts


def _variant(
    *,
    variant_id: str,
    title: str,
    kind: str,
    description: str,
    score: np.ndarray,
    prediction: np.ndarray,
    threshold: float | None,
    images_directory: Path,
    target: np.ndarray | None,
) -> dict[str, Any]:
    artifacts = {
        "score_map": str(images_directory / f"{variant_id}_score.png"),
        "prediction": str(images_directory / f"{variant_id}_prediction.png"),
    }
    predicted = np.asarray(prediction).astype(bool)
    imsave(artifacts["score_map"], _heatmap(score), check_contrast=False)
    imsave(artifacts["prediction"], (predicted.astype(np.uint8) * 255), check_contrast=False)
    metrics = {}
    if target is not None:
        artifacts["error_map"] = str(images_directory / f"{variant_id}_error.png")
        imsave(artifacts["error_map"], _error_map(predicted, target), check_contrast=False)
        metrics = binary_metrics(predicted, target, score_map=score)
    if threshold is not None:
        metrics = {**metrics, "threshold": float(threshold)}
    return {
        "id": variant_id,
        "title": title,
        "kind": kind,
        "description": description,
        "threshold": float(threshold) if threshold is not None else None,
        "metrics": metrics,
        "artifacts": artifacts,
    }


def _threshold_otsu(score: np.ndarray) -> np.ndarray:
    threshold = _safe_otsu(score)
    return np.asarray(score) >= threshold


def _safe_otsu(score: np.ndarray) -> float:
    values = np.asarray(score, dtype=np.float32)
    if np.isclose(float(values.min()), float(values.max())):
        return float(values.mean())
    return float(filters.threshold_otsu(values))


def _adaptive_block_size(shape: tuple[int, ...], window_radius: int) -> int:
    desired = max(3, (window_radius * 2) + 1)
    limit = max(3, min(int(shape[0]), int(shape[1])))
    if limit % 2 == 0:
        limit -= 1
    return max(3, min(desired if desired % 2 else desired + 1, limit))


def _mean_score(*scores: np.ndarray) -> np.ndarray:
    return np.mean([_zero_one(score) for score in scores], axis=0).astype(np.float32)


def _best_variant_id(variants: list[dict[str, Any]]) -> str | None:
    scored = [variant for variant in variants if "mean_iou" in variant.get("metrics", {})]
    if not scored:
        return None
    return max(scored, key=lambda variant: variant["metrics"]["mean_iou"])["id"]
