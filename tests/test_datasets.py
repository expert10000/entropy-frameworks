from pathlib import Path

from visionentropy.datasets.registry import DatasetSpec, dataset_status, load_dataset_specs
from visionentropy.datasets.skimage_examples import SkimageExamplesDataset
from visionentropy.datasets.synthetic_shapes import SyntheticShapesDataset


def test_synthetic_shapes_returns_image_and_mask() -> None:
    sample = SyntheticShapesDataset()[0]

    assert sample.sample_id == "synthetic_0000"
    assert sample.image.shape == (256, 256, 3)
    assert sample.mask is not None
    assert sample.mask.shape == (256, 256)


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
