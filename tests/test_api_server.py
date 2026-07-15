from pathlib import Path

from visionentropy.api.server import (
    build_comparison_config,
    build_run_config,
    build_run_slug,
    comparison_result_payload,
    dataset_preview_payload,
    resolve_local_path,
    run_history_payload,
    run_result_payload,
)
from visionentropy.pipeline import run_baseline_entropy_comparison, run_vertical_slice


def test_dataset_preview_payload_writes_preview_images() -> None:
    payload = dataset_preview_payload(
        {
            "name": "synthetic_shapes",
            "sample_index": "0",
            "representation": "grayscale",
            "height": "64",
            "width": "64",
        }
    )

    assert payload["sampleId"] == "synthetic_0000"
    assert payload["images"]["original"].startswith("/api/files")
    assert payload["images"]["representation"].startswith("/api/files")
    assert payload["images"]["mask"].startswith("/api/files")


def test_build_run_config_uses_ui_parameters() -> None:
    config = build_run_config(
        {
            "dataset": "synthetic_shapes",
            "sampleIndex": 2,
            "representation": "lab",
            "entropyMeasure": "shannon",
            "entropyScope": "local",
            "segmentationMethod": "kapur",
            "height": 96,
            "width": 128,
            "bins": 32,
            "windowRadius": 3,
        }
    )

    assert config["dataset"]["sample_index"] == 2
    assert config["preprocessing"]["resize"] == {"height": 96, "width": 128}
    assert config["representation"] == {"name": "lab"}
    assert config["entropy"]["name"] == "shannon"
    assert config["entropy"]["scope"] == "local"
    assert config["entropy"]["parameters"] == {"bins": 32, "window_radius": 3}
    assert config["segmentation"]["name"] == "kapur"
    assert "synthetic_002_shannon_local_kapur_r3_b32" in config["experiment"]["name"]


def test_build_run_config_defaults_to_feature_kmeans() -> None:
    config = build_run_config({"dataset": "synthetic_shapes", "sampleIndex": 0})

    assert config["segmentation"]["name"] == "feature_kmeans"
    assert config["segmentation"]["parameters"]["foreground"] == "mask_overlap"
    assert "synthetic_000_shannon_local_feature_kmeans_r4_b64" in config["experiment"]["name"]


def test_build_comparison_config_uses_ui_parameters() -> None:
    config = build_comparison_config(
        {
            "dataset": "synthetic_shapes",
            "sampleIndex": 2,
            "height": 96,
            "width": 128,
            "bins": 32,
            "windowRadius": 3,
        }
    )

    assert config["dataset"]["sample_index"] == 2
    assert config["preprocessing"]["resize"] == {"height": 96, "width": 128}
    assert config["entropy"]["parameters"] == {"bins": 32, "window_radius": 3}
    assert config["comparison"]["name"] == "baseline_vs_entropy"
    assert "comparison_synthetic_002_baseline_vs_entropy_r3_b32" in config["experiment"]["name"]


def test_build_run_slug_encodes_algorithms() -> None:
    slug = build_run_slug(
        dataset_name="synthetic_shapes",
        sample_index=0,
        entropy_measure="shannon",
        entropy_scope="local",
        segmentation_method="kapur",
        window_radius=4,
        bins=64,
    )

    assert slug == "synthetic_000_shannon_local_kapur_r4_b64"


def test_run_result_payload_returns_artifact_urls(tmp_path: Path) -> None:
    result = run_vertical_slice(
        {
            "experiment": {"name": "api_test", "output_directory": str(tmp_path / "run")},
            "dataset": {"name": "synthetic_shapes", "sample_index": 0, "image_size": [64, 64]},
            "preprocessing": {
                "resize": {"height": 64, "width": 64},
                "normalization": {"mode": "zero_one"},
            },
            "representation": {"name": "grayscale"},
            "entropy": {"parameters": {"bins": 32, "window_radius": 2}},
            "segmentation": {"parameters": {"bins": 32, "foreground": "high"}},
        }
    )

    payload = run_result_payload(result)

    assert payload["sampleId"] == "synthetic_0000"
    assert payload["artifacts"]["entropy_map"].startswith("/api/files")
    assert "dice" in payload["metrics"]


def test_comparison_result_payload_returns_variant_artifact_urls(tmp_path: Path) -> None:
    result = run_baseline_entropy_comparison(
        {
            "experiment": {"name": "api_comparison_test", "output_directory": str(tmp_path / "comparison")},
            "dataset": {"name": "synthetic_shapes", "sample_index": 0, "image_size": [48, 48]},
            "preprocessing": {
                "resize": {"height": 48, "width": 48},
                "normalization": {"mode": "zero_one"},
            },
            "entropy": {"parameters": {"bins": 16, "window_radius": 2}},
        }
    )

    payload = comparison_result_payload(result)

    assert payload["sampleId"] == "synthetic_0000"
    assert payload["artifacts"]["entropy_map"].startswith("/api/files")
    assert len(payload["variants"]) == 5
    assert payload["variants"][0]["artifacts"]["prediction"].startswith("/api/files")


def test_run_history_payload_lists_existing_runs() -> None:
    payload = run_history_payload()

    assert "runs" in payload
    assert isinstance(payload["runs"], list)


def test_resolve_local_path_rejects_parent_escape() -> None:
    try:
        resolve_local_path("../outside.txt")
    except ValueError as error:
        assert "outside" in str(error)
    else:
        raise AssertionError("expected workspace escape to be rejected")
