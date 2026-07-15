from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from visionentropy.datasets.base import ImageSample
from visionentropy.entropy.base import EntropyResult


@dataclass
class PipelineContext:
    sample: ImageSample
    preprocessed_image: object | None = None
    representation: object | None = None
    features: object | None = None
    entropy_result: EntropyResult | None = None
    segmentation: object | None = None
    metrics: dict[str, float] | None = None


@dataclass(frozen=True)
class PipelineResult:
    sample_id: str
    representation: Any
    features: Any
    entropy: EntropyResult
    segmentation: Any
    metrics: dict[str, float]
    artifacts: dict[str, str] = field(default_factory=dict)
    runtime: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
