from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class EntropyResult:
    value: float | None = None
    map: Any | None = None
    distribution: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class EntropyMeasure(Protocol):
    name: str

    def compute(self, data: Any, **kwargs: Any) -> EntropyResult:
        ...
