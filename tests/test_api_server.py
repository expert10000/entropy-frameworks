from pathlib import Path

from visionentropy.api.server import (
    build_run_config,
    dataset_preview_payload,
    resolve_local_path,
    run_history_payload,
    run_result_payload,
)
from visionentropy.pipeline import run_vertical_slice


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
            "height": 96,
            "width": 128,
            "bins": 32,
            "windowRadius": 3,
        }
    )

    assert config["dataset"]["sample_index"] == 2
    assert config["preprocessing"]["resize"] == {"height": 96, "width": 128}
    assert config["representation"] == {"name": "lab"}
    assert config["entropy"]["parameters"] == {"bins": 32, "window_radius": 3}


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
