from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from skimage import color

from visionentropy.representations.base import RepresentationResult


def ensure_float_rgb(image: NDArray[np.generic]) -> NDArray[np.float32]:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 2:
        array = color.gray2rgb(array)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError("expected an RGB image with shape H x W x 3")
    if array.max() > 1.0 or array.min() < 0.0:
        min_value = float(array.min())
        max_value = float(array.max())
        if np.isclose(min_value, max_value):
            return np.zeros_like(array, dtype=np.float32)
        array = (array - min_value) / (max_value - min_value)
    return array.astype(np.float32)


@dataclass(frozen=True)
class RGBRepresentation:
    name: str = "rgb"

    def transform(self, image: NDArray[np.generic]) -> RepresentationResult:
        data = ensure_float_rgb(image)
        return RepresentationResult(
            name=self.name,
            data=data,
            channels=("red", "green", "blue"),
            metadata={"color_space": "rgb"},
        )


@dataclass(frozen=True)
class GrayscaleRepresentation:
    name: str = "grayscale"

    def transform(self, image: NDArray[np.generic]) -> RepresentationResult:
        rgb = ensure_float_rgb(image)
        data = color.rgb2gray(rgb).astype(np.float32)
        return RepresentationResult(
            name=self.name,
            data=data,
            channels=("intensity",),
            metadata={"color_space": "grayscale"},
        )


@dataclass(frozen=True)
class LabRepresentation:
    name: str = "lab"

    def transform(self, image: NDArray[np.generic]) -> RepresentationResult:
        rgb = ensure_float_rgb(image)
        data = color.rgb2lab(rgb).astype(np.float32)
        return RepresentationResult(
            name=self.name,
            data=data,
            channels=("l", "a", "b"),
            metadata={"color_space": "lab"},
        )


@dataclass(frozen=True)
class ColorChannelRepresentation:
    channel: str
    name: str = "channel"

    def transform(self, image: NDArray[np.generic]) -> RepresentationResult:
        rgb = RGBRepresentation().transform(image)
        channel_map = {"red": 0, "green": 1, "blue": 2}
        if self.channel not in channel_map:
            raise ValueError(f"Unsupported channel: {self.channel}")
        index = channel_map[self.channel]
        return RepresentationResult(
            name=f"{self.name}_{self.channel}",
            data=rgb.data[..., index].astype(np.float32),
            channels=(self.channel,),
            metadata={"source": "rgb", "channel_index": index},
        )
