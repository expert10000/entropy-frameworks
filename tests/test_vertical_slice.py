from pathlib import Path

import numpy as np

from visionentropy.entropy import LocalEntropyMap
from visionentropy.evaluation import binary_metrics
from visionentropy.pipeline import run_vertical_slice
from visionentropy.segmentation import FeatureKMeansSegmenter, MaximumEntropySegmenter, maximum_entropy_threshold


def test_local_entropy_map_matches_image_shape() -> None:
    image = np.zeros((32, 32), dtype=np.float32)
    image[8:24, 8:24] = 1.0

    result = LocalEntropyMap(window_radius=2, bins=32).compute(image)

    assert result.map.shape == image.shape
    assert result.value is not None
    assert float(result.map.max()) >= 0.0


def test_maximum_entropy_segmenter_returns_binary_mask() -> None:
    values = np.concatenate([np.zeros(64), np.ones(64)]).reshape(16, 8)

    threshold = maximum_entropy_threshold(values, bins=8)
    mask = MaximumEntropySegmenter(bins=8).segment(values)

    assert 0.0 <= threshold <= 1.0
    assert mask.shape == values.shape
    assert mask.dtype == bool


def test_feature_kmeans_segmenter_uses_mask_overlap_for_eval_label() -> None:
    intensity = np.zeros((16, 16), dtype=np.float32)
    intensity[4:12, 4:12] = 1.0
    entropy = np.zeros_like(intensity)
    gradient = np.zeros_like(intensity)
    features = np.stack([intensity, entropy, gradient], axis=-1)
    target = intensity > 0

    result = FeatureKMeansSegmenter(foreground="mask_overlap", random_state=0).segment(
        features,
        target=target,
    )

    assert result.prediction.shape == target.shape
    assert result.foreground_rule == "mask_overlap_eval"
    assert result.prediction.dtype == bool


def test_binary_metrics_for_perfect_prediction() -> None:
    target = np.array([[0, 1], [1, 0]])
    prediction = target.astype(bool)

    metrics = binary_metrics(prediction, target)

    assert metrics["mean_iou"] == 1.0
    assert metrics["dice"] == 1.0
    assert metrics["pixel_accuracy"] == 1.0


def test_vertical_slice_writes_expected_outputs(tmp_path: Path) -> None:
    output_directory = tmp_path / "run"
    config = {
        "experiment": {"name": "test_vertical_slice", "output_directory": str(output_directory)},
        "dataset": {"name": "synthetic_shapes", "sample_index": 0, "image_size": [64, 64]},
        "preprocessing": {
            "resize": {"height": 64, "width": 64},
            "normalization": {"mode": "zero_one"},
        },
        "representation": {"name": "grayscale"},
        "entropy": {"parameters": {"bins": 32, "window_radius": 2}},
        "segmentation": {"parameters": {"bins": 32, "foreground": "high"}},
    }

    result = run_vertical_slice(config)

    assert result.sample_id == "synthetic_0000"
    assert result.metadata["run_metadata"]["dataset"] == "synthetic_shapes"
    assert result.metadata["run_metadata"]["sample"] == 0
    assert result.metadata["run_metadata"]["representation"] == "grayscale"
    assert result.segmentation.shape == (64, 64)
    assert "dice" in result.metrics
    assert (output_directory / "metrics.json").exists()
    assert (output_directory / "run_metadata.json").exists()
    assert (output_directory / "images" / "entropy_map.png").exists()
    assert (output_directory / "images" / "local_mean.png").exists()
    assert (output_directory / "images" / "local_variance.png").exists()
    assert (output_directory / "images" / "histogram.png").exists()
    assert (output_directory / "images" / "threshold_curve.png").exists()
    assert (output_directory / "images" / "superpixel_map.png").exists()
    assert (output_directory / "images" / "score_map.png").exists()
    assert (output_directory / "images" / "prediction.png").exists()


def test_vertical_slice_feature_kmeans_writes_feature_outputs(tmp_path: Path) -> None:
    output_directory = tmp_path / "feature_run"
    config = {
        "experiment": {"name": "test_feature_kmeans", "output_directory": str(output_directory)},
        "dataset": {"name": "synthetic_shapes", "sample_index": 0, "image_size": [64, 64]},
        "preprocessing": {
            "resize": {"height": 64, "width": 64},
            "normalization": {"mode": "zero_one"},
        },
        "representation": {"name": "grayscale"},
        "entropy": {"parameters": {"bins": 32, "window_radius": 2}},
        "segmentation": {
            "name": "feature_kmeans",
            "parameters": {"bins": 32, "foreground": "mask_overlap", "random_state": 0},
        },
    }

    result = run_vertical_slice(config)

    assert result.segmentation.shape == (64, 64)
    assert result.metadata["foreground_rule"] == "mask_overlap_eval"
    assert result.metadata["feature_channels"] == [
        "grayscale",
        "local_entropy",
        "gradient_magnitude",
    ]
    assert (output_directory / "images" / "gradient_map.png").exists()
    assert (output_directory / "images" / "cluster_labels.png").exists()
    assert (output_directory / "images" / "local_mean.png").exists()
    assert (output_directory / "images" / "local_variance.png").exists()
    assert (output_directory / "images" / "histogram.png").exists()
    assert (output_directory / "images" / "superpixel_map.png").exists()
    assert (output_directory / "images" / "score_map.png").exists()
    assert (output_directory / "data" / "feature_stack.npy").exists()
