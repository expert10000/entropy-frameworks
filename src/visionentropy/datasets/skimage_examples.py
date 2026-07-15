from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from skimage import color, data

from visionentropy.datasets.base import ImageSample


@dataclass(frozen=True)
class SkimageExample:
    name: str
    loader: Callable[[], np.ndarray]


class SkimageExamplesDataset:
    """Small built-in images for smoke tests and demos."""

    examples = (
        SkimageExample("camera", data.camera),
        SkimageExample("coins", data.coins),
        SkimageExample("moon", data.moon),
        SkimageExample("page", data.page),
        SkimageExample("chelsea", data.chelsea),
    )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> ImageSample:
        if index < 0 or index >= len(self):
            raise IndexError(index)

        example = self.examples[index]
        image = example.loader()
        if image.ndim == 2:
            image = color.gray2rgb(image)

        image = image.astype(np.float32)
        if image.max() > 1.0:
            image /= 255.0

        return ImageSample(
            sample_id=f"skimage_{example.name}",
            image=image,
            mask=None,
            label=example.name,
            metadata={"source": "skimage_examples"},
        )
