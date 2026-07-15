import numpy as np

from visionentropy.datasets.synthetic_shapes import SyntheticShapesDataset
from visionentropy.preprocessing import ComposeTransforms, NormalizeImage, ResizeSample
from visionentropy.representations import (
    GrayscaleRepresentation,
    LabRepresentation,
    RGBRepresentation,
    build_representation,
)


def test_resize_sample_resizes_image_and_mask() -> None:
    sample = SyntheticShapesDataset()[0]

    resized = ResizeSample(height=128, width=96).transform(sample)

    assert resized.image.shape == (128, 96, 3)
    assert resized.mask is not None
    assert resized.mask.shape == (128, 96)


def test_normalize_zero_one_scales_uint_like_image() -> None:
    sample = SyntheticShapesDataset()[0]
    sample = type(sample)(
        sample_id=sample.sample_id,
        image=(sample.image * 255).astype(np.uint8),
        mask=sample.mask,
        label=sample.label,
        metadata=sample.metadata,
    )

    normalized = NormalizeImage(mode="zero_one").transform(sample)

    assert normalized.image.dtype == np.float32
    assert 0.0 <= float(normalized.image.min()) <= 1.0
    assert 0.0 <= float(normalized.image.max()) <= 1.0


def test_compose_transforms_applies_in_order() -> None:
    sample = SyntheticShapesDataset()[0]

    transformed = ComposeTransforms(
        [ResizeSample(height=64, width=64), NormalizeImage(mode="zero_one")]
    ).transform(sample)

    assert transformed.image.shape == (64, 64, 3)
    assert transformed.metadata["resize"] == {"height": 64, "width": 64}
    assert transformed.metadata["normalization"] == {"mode": "zero_one"}


def test_rgb_representation_returns_three_channels() -> None:
    image = SyntheticShapesDataset()[0].image

    result = RGBRepresentation().transform(image)

    assert result.name == "rgb"
    assert result.data.shape == (256, 256, 3)
    assert result.channels == ("red", "green", "blue")


def test_grayscale_representation_returns_single_plane() -> None:
    image = SyntheticShapesDataset()[0].image

    result = GrayscaleRepresentation().transform(image)

    assert result.name == "grayscale"
    assert result.data.shape == (256, 256)
    assert result.channels == ("intensity",)


def test_lab_representation_returns_lab_channels() -> None:
    image = SyntheticShapesDataset()[0].image

    result = LabRepresentation().transform(image)

    assert result.name == "lab"
    assert result.data.shape == (256, 256, 3)
    assert result.channels == ("l", "a", "b")


def test_representation_registry_builds_channel_view() -> None:
    image = SyntheticShapesDataset()[0].image

    result = build_representation("red").transform(image)

    assert result.name == "channel_red"
    assert result.data.shape == (256, 256)
    assert result.channels == ("red",)
