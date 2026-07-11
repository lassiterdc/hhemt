"""Test-tier re-export shim over the lifted hhemt.synthetic_model subpackage
plus the test-only cache wrapper. Existing ``from tests.fixtures.synthetic_model
import ...`` callsites keep working; import direction is tests -> src."""

from hhemt.synthetic_model import (
    SyntheticCaseArtifacts,
    SyntheticModelParams,
    build_synthetic_case,
)
from tests.fixtures.synthetic_model.cache import (
    DEFAULT_PARAMS,
    get_or_build_synthetic_case,
)

__all__ = [
    "DEFAULT_PARAMS",
    "SyntheticCaseArtifacts",
    "SyntheticModelParams",
    "build_synthetic_case",
    "get_or_build_synthetic_case",
]
