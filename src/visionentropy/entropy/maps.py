from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from skimage.filters.rank import entropy as rank_entropy
from skimage.morphology import disk
from skimage.util import img_as_ubyte

from visionentropy.entropy.base import EntropyResult
from visionentropy.entropy.information import ShannonEntropy


@dataclass(frozen=True)
class LocalEntropyMap:
    window_radius: int = 4
    bins: int = 256
    name: str = "local_entropy"

    def compute(self, image: NDArray[np.generic]) -> EntropyResult:
        if self.window_radius < 1:
            raise ValueError("window_radius must be at least 1")

        gray = np.asarray(image, dtype=np.float32)
        if gray.ndim == 3:
            gray = gray.mean(axis=-1)
        gray = _zero_one(gray)
        entropy_map = rank_entropy(img_as_ubyte(gray), disk(self.window_radius)).astype(np.float32)
        max_value = float(np.log2(max(self.bins, 2)))
        normalized_map = entropy_map / max_value if max_value > 0 else entropy_map

        return EntropyResult(
            value=ShannonEntropy().compute(gray, bins=self.bins).value,
            map=normalized_map.astype(np.float32),
            metadata={
                "bins": self.bins,
                "window_radius": self.window_radius,
                "normalized": True,
            },
        )


def _zero_one(values: NDArray[np.generic]) -> NDArray[np.float32]:
    array = np.asarray(values, dtype=np.float32)
    min_value = float(array.min())
    max_value = float(array.max())
    if np.isclose(min_value, max_value):
        return np.zeros_like(array, dtype=np.float32)
    if min_value >= 0.0 and max_value <= 1.0:
        return array
    return ((array - min_value) / (max_value - min_value)).astype(np.float32)
