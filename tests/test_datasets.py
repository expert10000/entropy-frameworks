from pathlib import Path

from visionentropy.datasets.registry import DatasetSpec, dataset_status, load_dataset_specs
from visionentropy.datasets.skimage_examples import SkimageExamplesDataset
from visionentropy.datasets.synthetic_shapes import (
    SyntheticShapesConfig,
    SyntheticShapesDataset,
    synthetic_config_from_preset,
    synthetic_preset_names,
)


def test_synthetic_shapes_returns_image_and_mask() -> None:
    sample = SyntheticShapesDataset()[0]

    assert sample.sample_id == "synthetic_0000"
    assert sample.image.shape == (256, 256, 3)
    assert sample.mask is not None
    assert sample.mask.shape == (256, 256)
    assert sample.metadata["impulse_noise"] == 0.0


def test_synthetic_presets_are_fixed_benchmarks() -> None:
    presets = synthetic_preset_names()
    config = synthetic_config_from_preset("s03_impulse_noise", image_size=(64, 64))
    sample = SyntheticShapesDataset(config)[0]

    assert len(presets) == 8
    assert config.preset == "s03_impulse_noise"
    assert sample.image.shape == (64, 64, 3)
    assert sample.metadata["impulse_noise"] == 0.06


def test_synthetic_generator_exposes_texture_and_overlap_controls() -> None:
    sample = SyntheticShapesDataset(
        SyntheticShapesConfig(
            image_size=(64, 64),
            shape_count=4,
            foreground_texture=0.25,
            background_texture=0.25,
            gaussian_noise=0.02,
            boundary_blur=1.2,
            illumination_gradient=0.2,
            allow_overlap=True,
            contrast=0.6,
            seed=7,
        )
    )[0]

    assert sample.metadata["shape_count"] == 4
    assert sample.metadata["foreground_texture"] == 0.25
    assert sample.metadata["background_texture"] == 0.25
    assert sample.metadata["allow_overlap"] is True


def test_skimage_examples_returns_rgb_sample() -> None:
    sample = SkimageExamplesDataset()[0]

    assert sample.sample_id.startswith("skimage_")
    assert sample.image.ndim == 3
    assert sample.image.shape[-1] == 3


def test_generated_dataset_status_is_ready() -> None:
    spec = DatasetSpec(name="synthetic_shapes", title="Synthetic Shapes", mode="generated")

    status = dataset_status(spec)

    assert status.ready is True
    assert status.missing_paths == ()


def test_user_managed_dataset_reports_missing_paths(tmp_path: Path) -> None:
    spec = DatasetSpec(
        name="oxford_iiit_pet",
        title="Oxford-IIIT Pet",
        mode="user_managed",
        root=Path("data/raw/oxford_iiit_pet"),
        required_paths=("images", "annotations"),
    )

    status = dataset_status(spec, project_root=tmp_path)

    assert status.ready is False
    assert len(status.missing_paths) == 2


def test_dataset_specs_load_from_config() -> None:
    specs = load_dataset_specs()

    assert "synthetic_shapes" in specs
    assert "skimage_examples" in specs
    assert "oxford_iiit_pet" in specs
