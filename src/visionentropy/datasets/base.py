from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ImageSample:
    """A dataset-independent image sample."""

    sample_id: str
    image: NDArray[np.generic]
    mask: NDArray[np.generic] | None = None
    label: int | str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ImageDataset(Protocol):
    """Common protocol for all VisionEntropy image datasets."""

    def __len__(self) -> int:
        ...

    def __getitem__(self, index: int) -> ImageSample:
        ...
