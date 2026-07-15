"""Dataset interfaces and loaders."""

from visionentropy.datasets.base import ImageDataset, ImageSample
from visionentropy.datasets.registry import DatasetSpec, DatasetStatus, dataset_status, load_dataset_specs
from visionentropy.datasets.skimage_examples import SkimageExamplesDataset
from visionentropy.datasets.synthetic_shapes import SyntheticShapesDataset

__all__ = [
    "DatasetSpec",
    "DatasetStatus",
    "ImageDataset",
    "ImageSample",
    "SkimageExamplesDataset",
    "SyntheticShapesDataset",
    "dataset_status",
    "load_dataset_specs",
]
