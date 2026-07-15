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
from skimage.io import imsave

from visionentropy.datasets.base import ImageSample
from visionentropy.datasets.skimage_examples import SkimageExamplesDataset
from visionentropy.datasets.synthetic_shapes import SyntheticShapesConfig, SyntheticShapesDataset
from visionentropy.entropy import LocalEntropyMap
from visionentropy.evaluation import binary_metrics
from visionentropy.pipeline.base import PipelineResult
from visionentropy.preprocessing import ComposeTransforms, NormalizeImage, ResizeSample
from visionentropy.representations import build_representation
from visionentropy.segmentation import MaximumEntropySegmenter, maximum_entropy_threshold

matplotlib.use("Agg")
from matplotlib import colormaps  # noqa: E402


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
    bins = int(entropy_config.get("bins", 64))
    window_radius = int(entropy_config.get("window_radius", 4))
    entropy_result = LocalEntropyMap(window_radius=window_radius, bins=bins).compute(representation.data)

    segmenter_config = config.get("segmentation", {}).get("parameters", {})
    threshold_bins = int(segmenter_config.get("bins", bins))
    foreground = segmenter_config.get("foreground", "high")
    segmenter = MaximumEntropySegmenter(bins=threshold_bins, foreground=foreground)
    segmentation = segmenter.segment(entropy_result.map)
    threshold = maximum_entropy_threshold(entropy_result.map, bins=threshold_bins)

    metrics = {}
    if sample.mask is not None:
        metrics = binary_metrics(segmentation, sample.mask)

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
        started=started,
    )

    duration = time.perf_counter() - started
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
            "threshold": threshold,
        },
    )


def _load_sample(dataset_config: dict[str, Any]) -> ImageSample:
    name = dataset_config.get("name", "synthetic_shapes")
    sample_index = int(dataset_config.get("sample_index", 0))

    if name == "synthetic_shapes":
        image_size = tuple(dataset_config.get("image_size", (256, 256)))
        dataset = SyntheticShapesDataset(
            SyntheticShapesConfig(
                image_size=(int(image_size[0]), int(image_size[1])),
                sample_count=int(dataset_config.get("sample_count", 16)),
                seed=int(dataset_config.get("seed", 42)),
            )
        )
        return dataset[sample_index]

    if name == "skimage_examples":
        return SkimageExamplesDataset()[sample_index]

    raise ValueError(f"Vertical slice currently supports synthetic_shapes and skimage_examples, not {name}.")


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
    threshold: float,
    started: float,
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
        "prediction": str(images_directory / "prediction.png"),
        "metrics_json": str(output_directory / "metrics.json"),
        "metrics_csv": str(output_directory / "metrics.csv"),
        "runtime": str(output_directory / "runtime.json"),
        "environment": str(output_directory / "environment.json"),
        "summary": str(output_directory / "summary.md"),
        "prediction_array": str(data_directory / "prediction.npy"),
        "entropy_array": str(data_directory / "entropy_map.npy"),
    }

    imsave(artifacts["original_image"], _to_uint8_rgb(sample.image), check_contrast=False)
    imsave(artifacts["representation"], _to_viewable_image(representation_data), check_contrast=False)
    imsave(artifacts["entropy_map"], _heatmap(entropy_map), check_contrast=False)
    imsave(artifacts["prediction"], (segmentation.astype(np.uint8) * 255), check_contrast=False)
    np.save(artifacts["prediction_array"], segmentation.astype(np.uint8))
    np.save(artifacts["entropy_array"], entropy_map.astype(np.float32))

    if sample.mask is not None:
        artifacts["ground_truth"] = str(images_directory / "ground_truth.png")
        artifacts["error_map"] = str(images_directory / "error_map.png")
        target = np.asarray(sample.mask) > 0
        imsave(artifacts["ground_truth"], (target.astype(np.uint8) * 255), check_contrast=False)
        imsave(artifacts["error_map"], _error_map(segmentation, target), check_contrast=False)

    metrics_payload = {**metrics, "threshold": float(threshold)}
    Path(artifacts["metrics_json"]).write_text(
        json.dumps(metrics_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_metrics_csv(Path(artifacts["metrics_csv"]), metrics_payload)
    Path(artifacts["runtime"]).write_text(
        json.dumps({"duration_seconds": time.perf_counter() - started}, indent=2),
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
