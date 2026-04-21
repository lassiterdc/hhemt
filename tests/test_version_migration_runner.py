"""Unit tests for version_migration.runner / CLI - exit-code mapping."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from TRITON_SWMM_toolkit.version_migration import runner, state
from TRITON_SWMM_toolkit.version_migration.constants import LAYOUT_VERSION
from TRITON_SWMM_toolkit.version_migration.exceptions import LayoutVersionError


def test_cli_baseline_out_of_range_exits_2(tmp_path: Path) -> None:
    """__main__ baseline with version > LAYOUT_VERSION exits 2 (validation)."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "TRITON_SWMM_toolkit.version_migration",
            "baseline",
            str(tmp_path),
            "99",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2


def test_baseline_below_range_raises(tmp_path: Path) -> None:
    """runner.baseline with version below MINIMUM_SUPPORTED_VERSION raises."""
    with pytest.raises(LayoutVersionError):
        runner.baseline(tmp_path, -1)


def test_lazy_stamping_writes_version_file(tmp_path: Path) -> None:
    """stamp_new_target on a fresh dir writes _version.json at LAYOUT_VERSION."""
    state.stamp_new_target(tmp_path, LAYOUT_VERSION)
    assert (tmp_path / "_version.json").exists()
    st = state.read_version_file(tmp_path)
    assert st is not None
    assert st.layout_version == LAYOUT_VERSION
