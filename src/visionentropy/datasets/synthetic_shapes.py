from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from skimage.draw import disk, ellipse, rectangle
from skimage.filters import gaussian
from skimage.util import random_noise

from visionentropy.datasets.base import ImageSample


@dataclass(frozen=True)
class SyntheticShapesConfig:
    image_size: tuple[int, int] = (256, 256)
    sample_count: int = 16
    min_shapes: int = 2
    max_shapes: int = 5
    noise_amount: float = 0.03
    seed: int = 42


class SyntheticShapesDataset:
    """Procedural dataset for deterministic segmentation experiments."""

    def __init__(self, config: SyntheticShapesConfig | None = None) -> None:
        self.config = config or SyntheticShapesConfig()

    def __len__(self) -> int:
        return self.config.sample_count

    def __getitem__(self, index: int) -> ImageSample:
        if index < 0 or index >= len(self):
            raise IndexError(index)

        rng = np.random.default_rng(self.config.seed + index)
        height, width = self.config.image_size
        image = np.zeros((height, width, 3), dtype=np.float32)
        mask = np.zeros((height, width), dtype=np.uint8)

        shape_count = int(rng.integers(self.config.min_shapes, self.config.max_shapes + 1))
        for shape_id in range(1, shape_count + 1):
            color = rng.uniform(0.25, 1.0, size=3)
            shape_kind = rng.choice(["disk", "ellipse", "rectangle"])
            rr, cc = self._draw_shape(shape_kind, rng, height, width)
            image[rr, cc] = color
            mask[rr, cc] = shape_id

        image = gaussian(image, sigma=0.7, channel_axis=-1, preserve_range=True)
        image = random_noise(image, mode="s&p", amount=self.config.noise_amount, seed=self.config.seed + index)
        image = np.clip(image, 0.0, 1.0).astype(np.float32)

        return ImageSample(
            sample_id=f"synthetic_{index:04d}",
            image=image,
            mask=mask,
            metadata={"source": "synthetic_shapes", "shape_count": shape_count},
        )

    @staticmethod
    def _draw_shape(
        shape_kind: str,
        rng: np.random.Generator,
        height: int,
        width: int,
    ) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
        center_r = int(rng.integers(height // 5, height - height // 5))
        center_c = int(rng.integers(width // 5, width - width // 5))
        radius_r = int(rng.integers(max(8, height // 14), max(12, height // 5)))
        radius_c = int(rng.integers(max(8, width // 14), max(12, width // 5)))

        if shape_kind == "disk":
            return disk((center_r, center_c), min(radius_r, radius_c), shape=(height, width))

        if shape_kind == "ellipse":
            return ellipse(center_r, center_c, radius_r, radius_c, shape=(height, width))

        start = (max(0, center_r - radius_r), max(0, center_c - radius_c))
        extent = (min(height - start[0], radius_r * 2), min(width - start[1], radius_c * 2))
        return rectangle(start=start, extent=extent, shape=(height, width))
