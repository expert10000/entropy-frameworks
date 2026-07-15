from __future__ import annotations

from typing import Protocol

from visionentropy.datasets.base import ImageSample


class SampleTransform(Protocol):
    name: str

    def transform(self, sample: ImageSample) -> ImageSample:
        ...


class ComposeTransforms:
    name = "compose"

    def __init__(self, transforms: list[SampleTransform]) -> None:
        self.transforms = transforms

    def transform(self, sample: ImageSample) -> ImageSample:
        current = sample
        for transform in self.transforms:
            current = transform.transform(current)
        return current
