from __future__ import annotations

from visionentropy.representations.base import ImageRepresentation
from visionentropy.representations.color_spaces import (
    ColorChannelRepresentation,
    GrayscaleRepresentation,
    LabRepresentation,
    RGBRepresentation,
)


def build_representation(name: str) -> ImageRepresentation:
    registry: dict[str, ImageRepresentation] = {
        "rgb": RGBRepresentation(),
        "grayscale": GrayscaleRepresentation(),
        "lab": LabRepresentation(),
        "red": ColorChannelRepresentation("red"),
        "green": ColorChannelRepresentation("green"),
        "blue": ColorChannelRepresentation("blue"),
    }
    try:
        return registry[name]
    except KeyError as error:
        available = ", ".join(sorted(registry))
        raise ValueError(f"Unknown representation '{name}'. Available: {available}") from error
