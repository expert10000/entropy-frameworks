from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from visionentropy.datasets.base import ImageSample


@dataclass(frozen=True)
class NormalizeImage:
    mode: str = "zero_one"
    name: str = "normalize"

    def transform(self, sample: ImageSample) -> ImageSample:
        image = np.asarray(sample.image, dtype=np.float32)

        if self.mode == "zero_one":
            image = self._zero_one(image)
        elif self.mode == "standard":
            image = self._standard(image)
        else:
            raise ValueError(f"Unsupported normalization mode: {self.mode}")

        metadata = {
            **sample.metadata,
            "normalization": {"mode": self.mode},
        }
        return ImageSample(
            sample_id=sample.sample_id,
            image=image.astype(np.float32),
            mask=sample.mask,
            label=sample.label,
            metadata=metadata,
        )

    @staticmethod
    def _zero_one(image: np.ndarray) -> np.ndarray:
        if image.size == 0:
            raise ValueError("image cannot be empty")
        if image.min() >= 0.0 and image.max() <= 1.0:
            return image
        min_value = float(image.min())
        max_value = float(image.max())
        if np.isclose(min_value, max_value):
            return np.zeros_like(image, dtype=np.float32)
        return (image - min_value) / (max_value - min_value)

    @staticmethod
    def _standard(image: np.ndarray) -> np.ndarray:
        std = float(image.std())
        if np.isclose(std, 0.0):
            return np.zeros_like(image, dtype=np.float32)
        return (image - float(image.mean())) / std
