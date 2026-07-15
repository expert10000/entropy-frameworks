from pathlib import Path

from visionentropy.pipeline import run_baseline_entropy_comparison


def test_baseline_entropy_comparison_writes_expected_outputs(tmp_path: Path) -> None:
    output_directory = tmp_path / "comparison"
    result = run_baseline_entropy_comparison(
        {
            "experiment": {"name": "test_comparison", "output_directory": str(output_directory)},
            "dataset": {"name": "synthetic_shapes", "sample_index": 0, "image_size": [48, 48]},
            "preprocessing": {
                "resize": {"height": 48, "width": 48},
                "normalization": {"mode": "zero_one"},
            },
            "entropy": {"parameters": {"bins": 16, "window_radius": 2}},
        }
    )

    assert result["sampleId"] == "synthetic_0000"
    assert len(result["variants"]) == 5
    assert result["bestVariantId"] is not None
    assert (output_directory / "comparison.json").exists()
    assert (output_directory / "images" / "baseline_a_grayscale_otsu_prediction.png").exists()
    assert (output_directory / "images" / "experiment_e_grayscale_gradient_local_shannon_error.png").exists()
    assert all("dice" in variant["metrics"] for variant in result["variants"])
