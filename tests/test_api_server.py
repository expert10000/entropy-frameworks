from pathlib import Path
import time

from visionentropy.api import server as api_server
from visionentropy.api.server import (
    build_comparison_config,
    build_run_config,
    build_run_slug,
    cancel_job_payload,
    comparison_result_payload,
    dataset_preview_payload,
    job_status_payload,
    resolve_local_path,
    run_history_payload,
    run_result_payload,
    start_run_job,
    synthetic_parameters_from_payload,
    synthetic_presets_payload,
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


def test_synthetic_presets_payload_lists_benchmarks() -> None:
    payload = synthetic_presets_payload()

    assert payload["presets"][0]["id"] == "custom"
    assert [preset["id"] for preset in payload["presets"][1:]] == [
        "s01_clean_high_contrast",
        "s02_gaussian_noise",
        "s03_impulse_noise",
        "s04_blurred_boundaries",
        "s05_textured_foreground",
        "s06_textured_background",
        "s07_overlapping_objects",
        "s08_low_contrast",
    ]


def test_synthetic_parameters_from_payload_normalizes_ui_fields() -> None:
    parameters = synthetic_parameters_from_payload(
        {
            "synthetic": {
                "syntheticPreset": "custom",
                "shapeCount": 4,
                "foregroundTexture": 0.2,
                "backgroundTexture": 0.1,
                "gaussianNoise": 0.03,
                "impulseNoise": 0.04,
                "boundaryBlur": 1.5,
                "illuminationGradient": 0.2,
                "allowOverlap": True,
                "syntheticContrast": 0.7,
                "syntheticSeed": 123,
            }
        }
    )

    assert parameters["shape_count"] == 4
    assert parameters["foreground_texture"] == 0.2
    assert parameters["allow_overlap"] is True
    assert parameters["seed"] == 123


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
            "synthetic": {"syntheticPreset": "s02_gaussian_noise"},
            "deepModel": "resnet34",
            "deepLayer": "layer3",
            "deepRepresentationLevel": "superpixel_embedding",
            "deepUncertaintyMethod": "fuzzy_rough",
            "deepNeighborhoodK": 9,
            "deepSimilaritySigma": 0.8,
        }
    )

    assert config["dataset"]["sample_index"] == 2
    assert config["preprocessing"]["resize"] == {"height": 96, "width": 128}
    assert config["representation"] == {"name": "lab"}
    assert config["entropy"]["name"] == "shannon"
    assert config["entropy"]["scope"] == "local"
    assert config["entropy"]["parameters"] == {"bins": 32, "window_radius": 3}
    assert config["segmentation"]["name"] == "kapur"
    assert config["deep"]["enabled"] is True
    assert config["deep"]["model"] == "resnet34"
    assert config["deep"]["layer"] == "layer3"
    assert config["deep"]["representation_level"] == "superpixel_embedding"
    assert config["deep"]["uncertainty_method"] == "fuzzy_rough"
    assert config["deep"]["neighborhood_k"] == 9
    assert config["deep"]["similarity_sigma"] == 0.8
    assert config["dataset"]["preset"] == "s02_gaussian_noise"
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
    assert payload["runMetadata"]["dataset"] == "synthetic_shapes"
    assert payload["runMetadata"]["representation"] == "grayscale"
    assert payload["runMetadata"]["entropy"]["bins"] == 32
    assert payload["runMetadata"]["entropy"]["radius"] == 2
    assert payload["artifacts"]["entropy_map"].startswith("/api/files")
    assert payload["artifacts"]["region_entropy"].startswith("/api/files")
    assert payload["artifacts"]["graph_partition"].startswith("/api/files")
    assert payload["regions"]["count"] > 1
    assert payload["graph"]["partition_count"] >= 1
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


def test_job_registry_reports_progress_and_result() -> None:
    api_server.JOB_REGISTRY = api_server.ApiJobRegistry()

    job = api_server.job_registry().start(
        "slice",
        lambda progress: (progress("Working", 1, 3, 42.0), {"ok": True})[1],
    )
    payload = wait_for_job(job.id)

    assert payload["state"] == "complete"
    assert payload["stage"] == "Complete"
    assert payload["percent"] == 100.0
    assert payload["result"] == {"ok": True}
    assert [entry["stage"] for entry in payload["timeline"]][-1] == "Complete"
    assert payload["timeline"][0]["durationSeconds"] >= 0.0


def test_start_run_job_completes_with_status_result() -> None:
    api_server.JOB_REGISTRY = api_server.ApiJobRegistry()

    job = start_run_job(
        {
            "dataset": "synthetic_shapes",
            "sampleIndex": 0,
            "height": 32,
            "width": 32,
            "bins": 16,
            "windowRadius": 1,
            "deepEnabled": False,
        }
    )
    payload = wait_for_job(job["jobId"], timeout_seconds=20.0)

    assert payload["state"] == "complete"
    assert payload["result"]["sampleId"] == "synthetic_0000"
    assert payload["result"]["artifacts"]["entropy_map"].startswith("/api/files")
    assert payload["elapsedSeconds"] >= 0.0


def test_job_registry_cancels_cooperatively() -> None:
    api_server.JOB_REGISTRY = api_server.ApiJobRegistry()

    def task(progress):
        progress("First checkpoint", 1, 3, 20.0)
        time.sleep(0.2)
        progress("Second checkpoint", 2, 3, 60.0)
        return {"ok": False}

    job = api_server.job_registry().start("slice", task)
    wait_for_job_stage(job.id, "First checkpoint")
    cancelling = cancel_job_payload(job.id)
    payload = wait_for_job(job.id)

    assert cancelling["state"] == "cancelling"
    assert payload["state"] == "cancelled"
    assert payload["cancelRequested"] is True
    assert payload["result"] is None
    assert payload["error"] == "Job cancelled"


def test_resolve_local_path_rejects_parent_escape() -> None:
    try:
        resolve_local_path("../outside.txt")
    except ValueError as error:
        assert "outside" in str(error)
    else:
        raise AssertionError("expected workspace escape to be rejected")


def wait_for_job(job_id: str, *, timeout_seconds: float = 5.0) -> dict:
    deadline = time.time() + timeout_seconds
    payload = job_status_payload(job_id)
    while payload["state"] in {"queued", "running", "cancelling"} and time.time() < deadline:
        time.sleep(0.05)
        payload = job_status_payload(job_id)
    if payload["state"] in {"queued", "running", "cancelling"}:
        raise AssertionError(f"job did not finish: {payload}")
    return payload


def wait_for_job_stage(job_id: str, stage: str, *, timeout_seconds: float = 5.0) -> dict:
    deadline = time.time() + timeout_seconds
    payload = job_status_payload(job_id)
    while payload["stage"] != stage and payload["state"] in {"queued", "running"} and time.time() < deadline:
        time.sleep(0.02)
        payload = job_status_payload(job_id)
    if payload["stage"] != stage:
        raise AssertionError(f"job did not reach stage {stage}: {payload}")
    return payload
