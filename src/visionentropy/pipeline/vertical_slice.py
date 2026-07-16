from __future__ import annotations

import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import yaml
from scipy import ndimage
from skimage import filters
from skimage.io import imsave
from skimage.segmentation import slic

from visionentropy.datasets.base import ImageSample
from visionentropy.datasets.skimage_examples import SkimageExamplesDataset
from visionentropy.datasets.synthetic_shapes import (
    SyntheticShapesConfig,
    SyntheticShapesDataset,
    synthetic_config_from_preset,
)
from visionentropy.entropy import LocalEntropyMap
from visionentropy.evaluation import binary_metrics
from visionentropy.pipeline.base import PipelineResult
from visionentropy.pipeline.metadata import build_run_metadata
from visionentropy.preprocessing import ComposeTransforms, NormalizeImage, ResizeSample
from visionentropy.representations import build_representation
from visionentropy.segmentation import (
    AdaptiveThresholdSegmenter,
    FeatureKMeansSegmenter,
    GaussianMixtureSegmenter,
    MaximumEntropySegmenter,
    OtsuSegmenter,
    RandomForestSegmenter,
    RegionGrowingSegmenter,
    WatershedSegmenter,
    maximum_entropy_threshold,
)

matplotlib.use("Agg")
from matplotlib import colormaps  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402


def run_vertical_slice_from_config(config_path: str | Path) -> PipelineResult:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    return run_vertical_slice(config, config_path=path)


def run_vertical_slice(config: dict[str, Any], *, config_path: Path | None = None) -> PipelineResult:
    started = time.perf_counter()
    experiment = config.get("experiment", {})
    output_directory = Path(experiment.get("output_directory", "outputs/runs/vertical_slice"))
    images_directory = output_directory / "images"
    data_directory = output_directory / "data"
    images_directory.mkdir(parents=True, exist_ok=True)
    data_directory.mkdir(parents=True, exist_ok=True)

    sample = _load_sample(config.get("dataset", {}))
    sample = _preprocess_sample(sample, config.get("preprocessing", {}), config.get("dataset", {}))

    representation_name = config.get("representation", {}).get("name", "grayscale")
    representation = build_representation(representation_name).transform(sample.image)
    entropy_config = config.get("entropy", {}).get("parameters", {})
    entropy_name = config.get("entropy", {}).get("name", "shannon")
    entropy_scope = config.get("entropy", {}).get("scope", "local")
    bins = int(entropy_config.get("bins", 64))
    window_radius = int(entropy_config.get("window_radius", 4))
    entropy_result = LocalEntropyMap(window_radius=window_radius, bins=bins).compute(representation.data)

    segmentation_name = config.get("segmentation", {}).get("name", "maximum_entropy_threshold")
    segmenter_config = config.get("segmentation", {}).get("parameters", {})
    threshold_bins = int(segmenter_config.get("bins", bins))
    gradient_map = None
    feature_stack = None
    cluster_labels = None
    cluster_centers = None
    foreground_label = None
    foreground_rule = None
    score_map = entropy_result.map
    threshold_curve = False
    grayscale = _grayscale(sample.image)
    gradient_map = filters.sobel(grayscale)

    if segmentation_name in {"feature_kmeans", "boundary_region_kmeans", "kmeans"}:
        feature_entropy_map = LocalEntropyMap(window_radius=window_radius, bins=bins).compute(grayscale).map
        feature_stack = _boundary_region_features(
            grayscale=grayscale,
            entropy_map=feature_entropy_map,
            gradient_map=gradient_map,
        )
        segmenter = FeatureKMeansSegmenter(
            foreground=segmenter_config.get("foreground", "mask_overlap"),
            random_state=int(segmenter_config.get("random_state", 0)),
        )
        cluster_result = segmenter.segment(feature_stack, target=sample.mask)
        segmentation = cluster_result.prediction
        cluster_labels = cluster_result.labels
        cluster_centers = cluster_result.centers
        foreground_label = cluster_result.foreground_label
        foreground_rule = cluster_result.foreground_rule
        score_map = np.mean(feature_stack, axis=-1)
        threshold = None
    elif segmentation_name in {"gaussian_mixture", "gmm"}:
        feature_entropy_map = LocalEntropyMap(window_radius=window_radius, bins=bins).compute(grayscale).map
        feature_stack = _boundary_region_features(
            grayscale=grayscale,
            entropy_map=feature_entropy_map,
            gradient_map=gradient_map,
        )
        segmenter = GaussianMixtureSegmenter(
            foreground=segmenter_config.get("foreground", "mask_overlap"),
            random_state=int(segmenter_config.get("random_state", 0)),
        )
        segment_result = segmenter.segment(feature_stack, target=sample.mask)
        segmentation = segment_result.prediction
        cluster_labels = segment_result.labels
        cluster_centers = segment_result.centers
        foreground_label = segment_result.foreground_label
        foreground_rule = segment_result.foreground_rule
        score_map = segment_result.score_map
        threshold = None
    elif segmentation_name in {"random_forest", "rf"}:
        feature_entropy_map = LocalEntropyMap(window_radius=window_radius, bins=bins).compute(grayscale).map
        feature_stack = _boundary_region_features(
            grayscale=grayscale,
            entropy_map=feature_entropy_map,
            gradient_map=gradient_map,
        )
        segmenter = RandomForestSegmenter(random_state=int(segmenter_config.get("random_state", 0)))
        segment_result = segmenter.segment(feature_stack, target=sample.mask)
        segmentation = segment_result.prediction
        foreground_label = segment_result.foreground_label
        foreground_rule = segment_result.foreground_rule
        score_map = segment_result.score_map
        threshold = None
    elif segmentation_name in {"otsu", "threshold_otsu"}:
        segmenter = OtsuSegmenter(foreground=segmenter_config.get("foreground", "high"))
        segment_result = segmenter.segment(grayscale)
        segmentation = segment_result.prediction
        foreground_rule = segment_result.foreground_rule
        score_map = segment_result.score_map
        threshold = segment_result.threshold
    elif segmentation_name in {"local_adaptive", "adaptive_threshold"}:
        segmenter = AdaptiveThresholdSegmenter(
            window_radius=window_radius,
            foreground=segmenter_config.get("foreground", "high"),
        )
        segment_result = segmenter.segment(grayscale)
        segmentation = segment_result.prediction
        foreground_rule = segment_result.foreground_rule
        score_map = segment_result.score_map
        threshold = segment_result.threshold
    elif segmentation_name == "watershed":
        segmenter = WatershedSegmenter(foreground=segmenter_config.get("foreground", "high"))
        segment_result = segmenter.segment(grayscale, gradient_map=gradient_map)
        segmentation = segment_result.prediction
        cluster_labels = segment_result.labels
        foreground_label = segment_result.foreground_label
        foreground_rule = segment_result.foreground_rule
        score_map = segment_result.score_map
        threshold = segment_result.threshold
    elif segmentation_name == "region_growing":
        tolerance_value = segmenter_config.get("tolerance")
        tolerance = float(tolerance_value) if tolerance_value not in {None, ""} else None
        segmenter = RegionGrowingSegmenter(
            tolerance=tolerance,
            foreground=segmenter_config.get("foreground", "high"),
        )
        segment_result = segmenter.segment(grayscale)
        segmentation = segment_result.prediction
        cluster_labels = segment_result.labels
        foreground_label = segment_result.foreground_label
        foreground_rule = segment_result.foreground_rule
        score_map = segment_result.score_map
        threshold = segment_result.threshold
    else:
        foreground = segmenter_config.get("foreground", "high")
        segmenter = MaximumEntropySegmenter(bins=threshold_bins, foreground=foreground)
        segmentation = segmenter.segment(entropy_result.map)
        threshold = maximum_entropy_threshold(entropy_result.map, bins=threshold_bins)
        foreground_rule = f"{foreground}_entropy"
        threshold_curve = True

    metrics = {}
    if sample.mask is not None:
        metrics = binary_metrics(segmentation, sample.mask, score_map=score_map)

    artifacts = _save_artifacts(
        output_directory=output_directory,
        images_directory=images_directory,
        data_directory=data_directory,
        config=config,
        config_path=config_path,
        sample=sample,
        representation_data=representation.data,
        entropy_map=entropy_result.map,
        segmentation=segmentation,
        metrics=metrics,
        threshold=threshold,
        gradient_map=gradient_map,
        feature_stack=feature_stack,
        cluster_labels=cluster_labels,
        score_map=score_map,
        threshold_curve=threshold_curve,
        started=started,
    )

    duration = time.perf_counter() - started
    run_metadata = build_run_metadata(
        config,
        sample_id=sample.sample_id,
        runtime={"duration_seconds": duration},
    )
    return PipelineResult(
        sample_id=sample.sample_id,
        representation=representation,
        features=None,
        entropy=entropy_result,
        segmentation=segmentation,
        metrics=metrics,
        artifacts=artifacts,
        runtime={"duration_seconds": duration},
        metadata={
            "experiment": experiment.get("name", "vertical_slice"),
            "run_metadata": run_metadata,
            "entropy_measure": entropy_name,
            "entropy_scope": entropy_scope,
            "segmentation_method": segmentation_name,
            "threshold": threshold,
            "feature_channels": ["grayscale", "local_entropy", "gradient_magnitude"]
            if feature_stack is not None
            else None,
            "cluster_centers": cluster_centers.tolist() if cluster_centers is not None else None,
            "foreground_label": foreground_label,
            "foreground_rule": foreground_rule,
        },
    )


def _load_sample(dataset_config: dict[str, Any]) -> ImageSample:
    name = dataset_config.get("name", "synthetic_shapes")
    sample_index = int(dataset_config.get("sample_index", 0))

    if name == "synthetic_shapes":
        image_size = tuple(dataset_config.get("image_size", (256, 256)))
        config_kwargs = {
            "image_size": (int(image_size[0]), int(image_size[1])),
            "sample_count": int(dataset_config.get("sample_count", 16)),
        }
        preset = dataset_config.get("preset")
        if preset and preset != "custom":
            config = synthetic_config_from_preset(str(preset), **config_kwargs)
        else:
            config = SyntheticShapesConfig(
                **config_kwargs,
                shape_count=_optional_int(dataset_config.get("shape_count")),
                min_shapes=int(dataset_config.get("min_shapes", 2)),
                max_shapes=int(dataset_config.get("max_shapes", 5)),
                foreground_texture=float(dataset_config.get("foreground_texture", 0.0)),
                background_texture=float(dataset_config.get("background_texture", 0.0)),
                gaussian_noise=float(dataset_config.get("gaussian_noise", 0.0)),
                impulse_noise=float(dataset_config.get("impulse_noise", 0.0)),
                boundary_blur=float(dataset_config.get("boundary_blur", 0.7)),
                illumination_gradient=float(dataset_config.get("illumination_gradient", 0.0)),
                allow_overlap=bool(dataset_config.get("allow_overlap", False)),
                contrast=float(dataset_config.get("contrast", 1.0)),
                preset=str(preset) if preset else None,
                seed=int(dataset_config.get("seed", 42)),
            )
        dataset = SyntheticShapesDataset(config)
        return dataset[sample_index]

    if name == "skimage_examples":
        return SkimageExamplesDataset()[sample_index]

    raise ValueError(f"Vertical slice currently supports synthetic_shapes and skimage_examples, not {name}.")


def _optional_int(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    return int(value)


def _preprocess_sample(
    sample: ImageSample,
    preprocessing_config: dict[str, Any],
    dataset_config: dict[str, Any],
) -> ImageSample:
    transforms = []
    resize_config = preprocessing_config.get("resize")
    if resize_config:
        transforms.append(
            ResizeSample(height=int(resize_config["height"]), width=int(resize_config["width"]))
        )
    elif image_size := dataset_config.get("image_size"):
        transforms.append(ResizeSample(height=int(image_size[0]), width=int(image_size[1])))

    normalization = preprocessing_config.get("normalization", {})
    mode = normalization.get("mode") if isinstance(normalization, dict) else None
    if mode:
        transforms.append(NormalizeImage(mode=mode))

    if not transforms:
        return sample
    return ComposeTransforms(transforms).transform(sample)


def _save_artifacts(
    *,
    output_directory: Path,
    images_directory: Path,
    data_directory: Path,
    config: dict[str, Any],
    config_path: Path | None,
    sample: ImageSample,
    representation_data: np.ndarray,
    entropy_map: np.ndarray,
    segmentation: np.ndarray,
    metrics: dict[str, float],
    threshold: float | None,
    started: float,
    gradient_map: np.ndarray | None = None,
    feature_stack: np.ndarray | None = None,
    cluster_labels: np.ndarray | None = None,
    score_map: np.ndarray | None = None,
    threshold_curve: bool = False,
) -> dict[str, str]:
    config_target = output_directory / "config.yaml"
    if config_path is not None:
        config_target.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        config_target.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    artifacts = {
        "config": str(config_target),
        "original_image": str(images_directory / "original.png"),
        "representation": str(images_directory / "representation.png"),
        "entropy_map": str(images_directory / "entropy_map.png"),
        "local_mean": str(images_directory / "local_mean.png"),
        "local_variance": str(images_directory / "local_variance.png"),
        "histogram": str(images_directory / "histogram.png"),
        "superpixel_map": str(images_directory / "superpixel_map.png"),
        "score_map": str(images_directory / "score_map.png"),
        "prediction": str(images_directory / "prediction.png"),
        "metrics_json": str(output_directory / "metrics.json"),
        "metrics_csv": str(output_directory / "metrics.csv"),
        "runtime": str(output_directory / "runtime.json"),
        "environment": str(output_directory / "environment.json"),
        "run_metadata": str(output_directory / "run_metadata.json"),
        "summary": str(output_directory / "summary.md"),
        "prediction_array": str(data_directory / "prediction.npy"),
        "entropy_array": str(data_directory / "entropy_map.npy"),
    }

    imsave(artifacts["original_image"], _to_uint8_rgb(sample.image), check_contrast=False)
    imsave(artifacts["representation"], _to_viewable_image(representation_data), check_contrast=False)
    imsave(artifacts["entropy_map"], _heatmap(entropy_map), check_contrast=False)
    grayscale = _grayscale(sample.image)
    local_mean, local_variance = _local_statistics(grayscale, radius=_artifact_radius(config))
    imsave(artifacts["local_mean"], _to_viewable_image(local_mean), check_contrast=False)
    imsave(artifacts["local_variance"], _heatmap(local_variance), check_contrast=False)
    imsave(artifacts["superpixel_map"], _superpixel_map(sample.image), check_contrast=False)
    imsave(artifacts["score_map"], _heatmap(score_map if score_map is not None else entropy_map), check_contrast=False)
    _save_histogram(Path(artifacts["histogram"]), representation_data, threshold=threshold)
    if threshold is not None and threshold_curve:
        artifacts["threshold_curve"] = str(images_directory / "threshold_curve.png")
        _save_threshold_curve(Path(artifacts["threshold_curve"]), entropy_map, threshold=threshold)
    imsave(artifacts["prediction"], (segmentation.astype(np.uint8) * 255), check_contrast=False)
    np.save(artifacts["prediction_array"], segmentation.astype(np.uint8))
    np.save(artifacts["entropy_array"], entropy_map.astype(np.float32))

    if gradient_map is not None:
        artifacts["gradient_map"] = str(images_directory / "gradient_map.png")
        artifacts["gradient_array"] = str(data_directory / "gradient_map.npy")
        imsave(artifacts["gradient_map"], _heatmap(gradient_map), check_contrast=False)
        np.save(artifacts["gradient_array"], gradient_map.astype(np.float32))

    if feature_stack is not None:
        artifacts["feature_stack_array"] = str(data_directory / "feature_stack.npy")
        np.save(artifacts["feature_stack_array"], feature_stack.astype(np.float32))

    if cluster_labels is not None:
        artifacts["cluster_labels"] = str(images_directory / "cluster_labels.png")
        artifacts["cluster_labels_array"] = str(data_directory / "cluster_labels.npy")
        imsave(artifacts["cluster_labels"], _label_image(cluster_labels), check_contrast=False)
        np.save(artifacts["cluster_labels_array"], cluster_labels.astype(np.int32))

    if sample.mask is not None:
        artifacts["ground_truth"] = str(images_directory / "ground_truth.png")
        artifacts["error_map"] = str(images_directory / "error_map.png")
        target = np.asarray(sample.mask) > 0
        imsave(artifacts["ground_truth"], (target.astype(np.uint8) * 255), check_contrast=False)
        imsave(artifacts["error_map"], _error_map(segmentation, target), check_contrast=False)

    metrics_payload = {**metrics}
    if threshold is not None:
        metrics_payload["threshold"] = float(threshold)
    Path(artifacts["metrics_json"]).write_text(
        json.dumps(metrics_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_metrics_csv(Path(artifacts["metrics_csv"]), metrics_payload)
    runtime_payload = {"duration_seconds": time.perf_counter() - started}
    Path(artifacts["runtime"]).write_text(
        json.dumps(runtime_payload, indent=2),
        encoding="utf-8",
    )
    Path(artifacts["run_metadata"]).write_text(
        json.dumps(
            build_run_metadata(config, sample_id=sample.sample_id, runtime=runtime_payload),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    Path(artifacts["environment"]).write_text(
        json.dumps(_environment_payload(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    Path(artifacts["summary"]).write_text(
        _summary(sample.sample_id, metrics_payload, artifacts),
        encoding="utf-8",
    )
    return artifacts


def _artifact_radius(config: dict[str, Any]) -> int:
    entropy_config = config.get("entropy", {}).get("parameters", {})
    return int(entropy_config.get("window_radius", 4))


def _local_statistics(values: np.ndarray, *, radius: int) -> tuple[np.ndarray, np.ndarray]:
    size = max(3, (radius * 2) + 1)
    array = np.asarray(values, dtype=np.float32)
    local_mean = ndimage.uniform_filter(array, size=size, mode="reflect")
    local_square_mean = ndimage.uniform_filter(array * array, size=size, mode="reflect")
    local_variance = np.maximum(local_square_mean - (local_mean * local_mean), 0.0)
    return local_mean.astype(np.float32), local_variance.astype(np.float32)


def _save_histogram(path: Path, values: np.ndarray, *, threshold: float | None) -> None:
    figure, axis = plt.subplots(figsize=(5.2, 3.2), dpi=130)
    axis.hist(np.asarray(values, dtype=np.float32).ravel(), bins=64, color="#1f7a8c", alpha=0.86)
    if threshold is not None:
        axis.axvline(threshold, color="#ef476f", linewidth=2.0, label="threshold")
        axis.legend(frameon=False, fontsize=8)
    axis.set_title("Representation histogram")
    axis.set_xlabel("value")
    axis.set_ylabel("pixels")
    axis.grid(True, alpha=0.22)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _save_threshold_curve(path: Path, values: np.ndarray, *, threshold: float) -> None:
    thresholds, scores = _kapur_objective_curve(values)
    figure, axis = plt.subplots(figsize=(5.2, 3.2), dpi=130)
    axis.plot(thresholds, scores, color="#1f7a8c", linewidth=2.0)
    axis.axvline(threshold, color="#ef476f", linewidth=2.0, label="selected")
    axis.set_title("Kapur objective J(t) = H0(t) + H1(t)")
    axis.set_xlabel("threshold")
    axis.set_ylabel("objective")
    axis.grid(True, alpha=0.22)
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _kapur_objective_curve(values: np.ndarray, *, bins: int = 128) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=np.float64)
    histogram, edges = np.histogram(array.ravel(), bins=bins)
    probabilities = histogram.astype(np.float64)
    total = probabilities.sum()
    if total <= 0:
        return edges[1:-1], np.zeros(max(0, bins - 2), dtype=np.float64)
    probabilities /= total
    cumulative = np.cumsum(probabilities)
    scores = []
    thresholds = []
    for index in range(1, len(probabilities) - 1):
        weight_background = cumulative[index]
        weight_foreground = 1.0 - weight_background
        if weight_background <= 0 or weight_foreground <= 0:
            score = 0.0
        else:
            p_background = probabilities[: index + 1] / weight_background
            p_foreground = probabilities[index + 1 :] / weight_foreground
            score = _distribution_entropy(p_background) + _distribution_entropy(p_foreground)
        scores.append(score)
        thresholds.append(edges[index + 1])
    return np.asarray(thresholds), np.asarray(scores)


def _distribution_entropy(probabilities: np.ndarray) -> float:
    positive = probabilities[probabilities > 0]
    if positive.size == 0:
        return 0.0
    return float(-(positive * np.log(positive)).sum())


def _superpixel_map(image: np.ndarray) -> np.ndarray:
    labels = slic(
        _zero_one(np.asarray(image, dtype=np.float32)),
        n_segments=96,
        compactness=12.0,
        sigma=0.5,
        start_label=0,
        channel_axis=-1,
    )
    return _label_image(labels)


def _grayscale(image: np.ndarray) -> np.ndarray:
    return _zero_one(build_representation("grayscale").transform(image).data)


def _boundary_region_features(
    *,
    grayscale: np.ndarray,
    entropy_map: np.ndarray,
    gradient_map: np.ndarray,
) -> np.ndarray:
    return np.stack(
        [
            _zero_one(grayscale),
            _zero_one(entropy_map),
            _zero_one(gradient_map),
        ],
        axis=-1,
    ).astype(np.float32)


def _to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)
    return (_zero_one(array) * 255).astype(np.uint8)


def _to_viewable_image(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 2:
        return (_zero_one(array) * 255).astype(np.uint8)
    return (_zero_one(array) * 255).astype(np.uint8)


def _heatmap(values: np.ndarray) -> np.ndarray:
    normalized = _zero_one(np.asarray(values, dtype=np.float32))
    rgba = colormaps["magma"](normalized)
    return (rgba[..., :3] * 255).astype(np.uint8)


def _error_map(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    predicted = prediction.astype(bool)
    truth = target.astype(bool)
    image = np.zeros((*truth.shape, 3), dtype=np.uint8)
    image[np.logical_and(predicted, truth)] = (45, 156, 124)
    image[np.logical_and(predicted, ~truth)] = (239, 71, 111)
    image[np.logical_and(~predicted, truth)] = (255, 209, 102)
    return image


def _label_image(labels: np.ndarray) -> np.ndarray:
    label_array = np.asarray(labels, dtype=np.int32)
    palette = np.array(
        [
            [31, 122, 140],
            [255, 209, 102],
            [239, 71, 111],
            [45, 156, 124],
        ],
        dtype=np.uint8,
    )
    return palette[np.mod(label_array, len(palette))]


def _zero_one(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    min_value = float(array.min())
    max_value = float(array.max())
    if np.isclose(min_value, max_value):
        return np.zeros_like(array, dtype=np.float32)
    return ((array - min_value) / (max_value - min_value)).astype(np.float32)


def _write_metrics_csv(path: Path, metrics: dict[str, float]) -> None:
    lines = ["metric,value"]
    lines.extend(f"{key},{value}" for key, value in metrics.items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summary(sample_id: str, metrics: dict[str, float], artifacts: dict[str, str]) -> str:
    metric_lines = "\n".join(f"- {key}: {value:.4f}" for key, value in metrics.items())
    artifact_lines = "\n".join(f"- {key}: `{value}`" for key, value in artifacts.items())
    return f"""# VisionEntropy Vertical Slice

Sample: `{sample_id}`

## Metrics

{metric_lines}

## Artifacts

{artifact_lines}
"""


def _environment_payload() -> dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
    }


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()
