"""Pipeline orchestration."""

from visionentropy.pipeline.base import PipelineContext, PipelineResult
from visionentropy.pipeline.comparison import run_baseline_entropy_comparison
from visionentropy.pipeline.vertical_slice import run_vertical_slice, run_vertical_slice_from_config

__all__ = [
    "PipelineContext",
    "PipelineResult",
    "run_baseline_entropy_comparison",
    "run_vertical_slice",
    "run_vertical_slice_from_config",
]
