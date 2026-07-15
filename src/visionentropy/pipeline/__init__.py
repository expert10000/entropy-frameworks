"""Pipeline orchestration."""

from visionentropy.pipeline.base import PipelineContext, PipelineResult
from visionentropy.pipeline.vertical_slice import run_vertical_slice, run_vertical_slice_from_config

__all__ = ["PipelineContext", "PipelineResult", "run_vertical_slice", "run_vertical_slice_from_config"]
