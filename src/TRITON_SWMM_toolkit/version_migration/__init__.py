"""TRITON-SWMM_toolkit version-migration system.

Forward-only migrations of persistent on-disk state (analysis trees and
system directories), authored as numbered Python modules under
``versions/`` and applied via the ``MigrationContext`` primitive DSL.

Public surface:
    LAYOUT_VERSION         - current canonical layout version
    MINIMUM_SUPPORTED_VERSION - floor below which migration is refused
    run_migration, status, baseline, verify - high-level functions (also CLI)
    MigrationError        - exception base
"""

from __future__ import annotations

from TRITON_SWMM_toolkit.version_migration.constants import (
    LAYOUT_VERSION,
    MINIMUM_SUPPORTED_VERSION,
)
from TRITON_SWMM_toolkit.version_migration.exceptions import (
    BaselineRequiredError,
    LayoutVersionError,
    MigrationConflictError,
    MigrationError,
    RegistryError,
)
from TRITON_SWMM_toolkit.version_migration.runner import (
    baseline,
    run_migration,
    status,
    verify,
)

__all__ = [
    "LAYOUT_VERSION",
    "MINIMUM_SUPPORTED_VERSION",
    "MigrationError",
    "LayoutVersionError",
    "BaselineRequiredError",
    "MigrationConflictError",
    "RegistryError",
    "run_migration",
    "status",
    "baseline",
    "verify",
]
