"""Reviewer check for change 1121-uvpin: the pin is an absolute instant, never a relative span.

The defect is a moving cutoff. A relative value such as "3 days" in the project setting would
reproduce it, so the pin must parse as an RFC 3339 UTC instant. Runs no uv command.
"""

from __future__ import annotations

import tomllib
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCKFILE_LAST_CHANGE_UTC = datetime(2026, 11, 16, 16, 52, 27, tzinfo=UTC)  # commit 423c5d6d


def _pin() -> str:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["uv"]["exclude-newer"]


def test_the_pin_is_an_absolute_utc_instant_not_before_the_lockfile_last_change() -> None:
    """Property: the pinned value is a fixed UTC timestamp at or after the lock's last change (no version moved)."""
    pin = _pin()
    instant = datetime.fromisoformat(pin.replace("Z", "+00:00"))
    assert pin.endswith("Z") and instant.tzinfo is not None, pin
    assert instant >= LOCKFILE_LAST_CHANGE_UTC, pin
