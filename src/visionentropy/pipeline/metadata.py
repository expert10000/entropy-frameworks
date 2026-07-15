from __future__ import annotations

from typing import Any


def build_run_metadata(
    config: dict[str, Any],
    *,
    sample_id: str | None = None,
    runtime: dict[str, float] | None = None,
) -> dict[str, Any]:
    experiment = config.get("experiment", {})
    dataset = config.get("dataset", {})
    representation = config.get("representation", {})
    entropy = config.get("entropy", {})
    entropy_parameters = entropy.get("parameters", {})
    segmentation = config.get("segmentation", {})
    segmentation_parameters = segmentation.get("parameters", {})

    return {
        "run": experiment.get("name"),
        "dataset": dataset.get("name"),
        "sample": dataset.get("sample_index"),
        "sampleId": sample_id,
        "syntheticPreset": dataset.get("preset"),
        "representation": representation.get("name"),
        "entropy": {
            "name": entropy.get("name", "shannon"),
            "scope": entropy.get("scope", "local"),
            "bins": entropy_parameters.get("bins"),
            "radius": entropy_parameters.get("window_radius"),
        },
        "segmentation": {
            "name": segmentation.get("name"),
            "foreground": segmentation_parameters.get("foreground"),
        },
        "seed": dataset.get("seed"),
        "runtimeSeconds": (runtime or {}).get("duration_seconds"),
    }
