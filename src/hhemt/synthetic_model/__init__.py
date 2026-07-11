"""Synthetic TRITON-SWMM model generators (lifted to src per ADR-8 / PIP-2 Phase 1).

Public surface — the deterministic ``(params) -> files`` generation building
blocks used by the synthetic-experiment framework (``hhemt.synthetic_experiment``)
and re-exported test-side by ``tests/fixtures/synthetic_model``:

    SyntheticModelParams   — frozen dataclass of generation parameters
    SyntheticCaseArtifacts — frozen dataclass of generated-artifact paths
    build_synthetic_case(params, dest_dir) — cache-root-agnostic builder
"""

from hhemt.synthetic_model._build import (
    SyntheticCaseArtifacts,
    SyntheticModelParams,
    build_synthetic_case,
)

__all__ = ["SyntheticCaseArtifacts", "SyntheticModelParams", "build_synthetic_case"]
