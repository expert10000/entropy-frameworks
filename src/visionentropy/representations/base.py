from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class RepresentationResult:
    name: str
    data: NDArray[np.generic]
    channels: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


class ImageRepresentation(Protocol):
    name: str

    def transform(self, image: NDArray[np.generic]) -> RepresentationResult:
        ...
