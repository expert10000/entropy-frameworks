from pathlib import Path

import numpy as np

from visionentropy.deep import deep_learning_available
from visionentropy.deep.features import activation_entropy_map
from visionentropy.entropy import LocalEntropyMap
from visionentropy.evaluation import binary_metrics
from visionentropy.graphs import analyze_region_graph
from visionentropy.pipeline import run_vertical_slice
from visionentropy.representations import build_region_representation
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


def test_region_representation_builds_stats_and_graph() -> None:
    image = np.zeros((32, 32, 3), dtype=np.float32)
    image[:16, :16] = [1.0, 0.0, 0.0]
    image[16:, 16:] = [0.0, 1.0, 0.0]
    intensity = image.mean(axis=-1)
    entropy_map = np.zeros((32, 32), dtype=np.float32)
    entropy_map[8:24, 8:24] = 1.0

    regions = build_region_representation(
        image,
        intensity=intensity,
        entropy_map=entropy_map,
        n_segments=12,
        entropy_bins=8,
    )

    assert regions.labels.shape == intensity.shape
    assert regions.region_count > 1
    assert len(regions.edges) > 0
    assert {"mean_intensity", "region_entropy", "neighbor_count"}.issubset(regions.stats[0])


def test_graph_entropy_framework_analyzes_region_graph() -> None:
    image = np.zeros((32, 32, 3), dtype=np.float32)
    image[:16, :] = [1.0, 0.0, 0.0]
    image[16:, :] = [0.0, 1.0, 0.0]
    intensity = image.mean(axis=-1)
    entropy_map = np.zeros((32, 32), dtype=np.float32)
    entropy_map[:, 12:20] = 0.8
    regions = build_region_representation(
        image,
        intensity=intensity,
        entropy_map=entropy_map,
        n_segments=12,
        entropy_bins=8,
    )

    graph = analyze_region_graph(regions)
    payload = graph.payload(regions)

    assert graph.node_entropy.shape == (regions.region_count,)
    assert graph.edge_entropy.shape == (len(regions.edges),)
    assert graph.partition_labels.shape == (regions.region_count,)
    assert graph.normalized_spectral_entropy >= 0.0
    assert payload["partitionCount"] >= 1


def test_activation_entropy_map_matches_feature_spatial_shape() -> None:
    features = np.zeros((4, 6, 5), dtype=np.float32)
    features[0] = 1.0
    features[1, 2:4, 2:4] = 2.0

    entropy = activation_entropy_map(features)

    assert entropy.shape == (6, 5)
    assert float(entropy.max()) <= 1.0
    assert float(entropy.min()) >= 0.0


def test_classical_segmenters_return_binary_masks() -> None:
    intensity = np.zeros((24, 24), dtype=np.float32)
    intensity[6:18, 6:18] = 1.0
    entropy = np.zeros_like(intensity)
    gradient = np.zeros_like(intensity)
    gradient[5:19, 5:19] = 0.4
    features = np.stack([intensity, entropy, gradient], axis=-1)
    target = intensity > 0

    results = [
        OtsuSegmenter().segment(intensity),
        AdaptiveThresholdSegmenter(window_radius=2).segment(intensity),
        GaussianMixtureSegmenter(random_state=0).segment(features, target=target),
        RandomForestSegmenter(n_estimators=12, random_state=0).segment(features, target=target),
        WatershedSegmenter().segment(intensity, gradient_map=gradient),
        RegionGrowingSegmenter(tolerance=0.2).segment(intensity),
    ]

    for result in results:
        assert result.prediction.shape == intensity.shape
        assert result.prediction.dtype == bool
        assert result.score_map.shape == intensity.shape


def test_binary_metrics_for_perfect_prediction() -> None:
    target = np.array([[0, 1], [1, 0]])
    prediction = target.astype(bool)

    metrics = binary_metrics(prediction, target)

    assert metrics["mean_iou"] == 1.0
    assert metrics["dice"] == 1.0
    assert metrics["boundary_f1"] == 1.0
    assert metrics["error_detection_auroc"] == 0.5
    assert metrics["pixel_accuracy"] == 1.0
    assert metrics["specificity"] == 1.0
    assert metrics["true_positive"] == 2.0
    assert metrics["true_negative"] == 2.0


def test_binary_metrics_scores_error_detection_auroc() -> None:
    target = np.array([[0, 1], [1, 0]])
    prediction = np.array([[0, 0], [1, 1]]).astype(bool)
    score_map = np.array([[0.1, 0.9], [0.1, 0.9]])

    metrics = binary_metrics(prediction, target, score_map=score_map)

    assert metrics["error_detection_auroc"] == 1.0


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
    assert result.metadata["regions"]["count"] > 1
    assert result.metadata["regions"]["edge_count"] > 0
    assert result.metadata["graph"]["partition_count"] >= 1
    assert result.metadata["graph"]["normalized_spectral_entropy"] >= 0.0
    assert result.segmentation.shape == (64, 64)
    assert "dice" in result.metrics
    assert "boundary_f1" in result.metrics
    assert "error_detection_auroc" in result.metrics
    assert "specificity" in result.metrics
    assert "true_positive" in result.metrics
    assert (output_directory / "metrics.json").exists()
    assert (output_directory / "run_metadata.json").exists()
    assert (output_directory / "images" / "entropy_map.png").exists()
    assert (output_directory / "images" / "local_mean.png").exists()
    assert (output_directory / "images" / "local_variance.png").exists()
    assert (output_directory / "images" / "histogram.png").exists()
    assert (output_directory / "images" / "threshold_curve.png").exists()
    assert (output_directory / "images" / "superpixel_map.png").exists()
    assert (output_directory / "images" / "region_labels.png").exists()
    assert (output_directory / "images" / "region_mean.png").exists()
    assert (output_directory / "images" / "region_entropy.png").exists()
    assert (output_directory / "images" / "region_graph.png").exists()
    assert (output_directory / "images" / "graph_node_entropy.png").exists()
    assert (output_directory / "images" / "graph_edge_entropy.png").exists()
    assert (output_directory / "images" / "graph_spectral_entropy.png").exists()
    assert (output_directory / "images" / "graph_partition.png").exists()
    assert (output_directory / "region_stats.json").exists()
    assert (output_directory / "region_stats.csv").exists()
    assert (output_directory / "region_graph.json").exists()
    assert (output_directory / "graph_entropy.json").exists()
    assert (output_directory / "data" / "region_labels.npy").exists()
    assert (output_directory / "data" / "graph_node_entropy.npy").exists()
    assert (output_directory / "data" / "graph_partition.npy").exists()
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


def test_vertical_slice_writes_deep_entropy_outputs_when_enabled(tmp_path: Path) -> None:
    if not deep_learning_available():
        return
    output_directory = tmp_path / "deep_run"
    config = {
        "experiment": {"name": "test_deep", "output_directory": str(output_directory)},
        "dataset": {"name": "synthetic_shapes", "sample_index": 0, "image_size": [48, 48]},
        "preprocessing": {
            "resize": {"height": 48, "width": 48},
            "normalization": {"mode": "zero_one"},
        },
        "representation": {"name": "grayscale"},
        "entropy": {"parameters": {"bins": 24, "window_radius": 2}},
        "segmentation": {"name": "otsu", "parameters": {"bins": 24, "foreground": "high"}},
        "deep": {"enabled": True, "model": "resnet18", "image_size": 64, "random_state": 0},
    }

    result = run_vertical_slice(config)

    assert result.metadata["deep"]["available"] is True
    assert result.metadata["deep"]["model"] == "resnet18"
    assert result.metadata["deep"]["mean_activation_entropy"] >= 0.0
    assert (output_directory / "images" / "deep_feature_map.png").exists()
    assert (output_directory / "images" / "activation_entropy.png").exists()
    assert (output_directory / "images" / "latent_entropy.png").exists()
    assert (output_directory / "images" / "predictive_entropy.png").exists()
    assert (output_directory / "deep_entropy.json").exists()
    assert (output_directory / "data" / "deep_feature_map.npy").exists()
    assert (output_directory / "data" / "activation_entropy.npy").exists()
    assert (output_directory / "data" / "latent_vector.npy").exists()
    assert (output_directory / "data" / "predictive_logits.npy").exists()


def test_vertical_slice_runs_stage_one_classical_methods(tmp_path: Path) -> None:
    methods = [
        "otsu",
        "local_adaptive",
        "gaussian_mixture",
        "random_forest",
        "watershed",
        "region_growing",
    ]

    for method in methods:
        output_directory = tmp_path / method
        config = {
            "experiment": {"name": f"test_{method}", "output_directory": str(output_directory)},
            "dataset": {
                "name": "synthetic_shapes",
                "sample_index": 0,
                "image_size": [48, 48],
                "preset": "s01_clean_high_contrast",
            },
            "preprocessing": {
                "resize": {"height": 48, "width": 48},
                "normalization": {"mode": "zero_one"},
            },
            "representation": {"name": "grayscale"},
            "entropy": {"parameters": {"bins": 24, "window_radius": 2}},
            "segmentation": {
                "name": method,
                "parameters": {"bins": 24, "foreground": "mask_overlap", "random_state": 0},
            },
        }

        result = run_vertical_slice(config)

        assert result.segmentation.shape == (48, 48)
        assert result.metadata["segmentation_method"] == method
        assert "mean_iou" in result.metrics
        assert (output_directory / "images" / "prediction.png").exists()
        assert (output_directory / "images" / "score_map.png").exists()
