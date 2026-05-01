"""Version-migration constants.

`LAYOUT_VERSION` is the canonical current layout version. Bump when an
on-disk breaking change is introduced — the CI Check A enforces that a
bump implies a matching `versions/V{N:04d}__*.py` and matching golden
fixtures.

`MINIMUM_SUPPORTED_VERSION` is the floor below which `migrate` refuses to
run. Raised manually on major toolkit releases per the resolved decision
in the master plan.
"""

from __future__ import annotations

LAYOUT_VERSION: int = 4
MINIMUM_SUPPORTED_VERSION: int = 0

#: Default _version.json filename (used by both analysis and system stamps).
VERSION_FILE_NAME: str = "_version.json"

#: Lock timeout for filelock-guarded _version.json writes (seconds).
LOCK_TIMEOUT_SECONDS: float = 30.0
