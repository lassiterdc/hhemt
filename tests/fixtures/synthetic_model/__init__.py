"""Synthetic TRITON-SWMM model generator for fast PC tests.

Public entry points:
    SyntheticModelParams — frozen dataclass of generation parameters
    SyntheticCaseArtifacts — frozen dataclass of paths to generated artifacts
    DEFAULT_PARAMS — the standard 20x30 @ 10m test model
    get_or_build_synthetic_case(params) — build-or-reuse-from-cache
"""

from tests.fixtures.synthetic_model.cache import (
    DEFAULT_PARAMS,
    SyntheticCaseArtifacts,
    SyntheticModelParams,
    get_or_build_synthetic_case,
)

__all__ = [
    "DEFAULT_PARAMS",
    "SyntheticCaseArtifacts",
    "SyntheticModelParams",
    "get_or_build_synthetic_case",
]
