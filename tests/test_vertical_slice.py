from pathlib import Path

import numpy as np

from visionentropy.entropy import LocalEntropyMap
from visionentropy.evaluation import binary_metrics
from visionentropy.pipeline import run_vertical_slice
from visionentropy.segmentation import MaximumEntropySegmenter, maximum_entropy_threshold


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
    assert result.segmentation.shape == (64, 64)
    assert "dice" in result.metrics
    assert (output_directory / "metrics.json").exists()
    assert (output_directory / "images" / "entropy_map.png").exists()
    assert (output_directory / "images" / "prediction.png").exists()
