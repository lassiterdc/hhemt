"""V0005 inline-report-config migration: happy-path + fail-fast + idempotency.

Fixtures live under tests/fixtures/legacy_layouts/v0005_unit_test/ rather than
under v4/ or v5/ to avoid colliding with the auto-discovery glob in
test_version_migration_golden.py::_discover_fixture_pairs (which requires
each v{N}/ directory to be a single full-corpus chain fixture).
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from hhemt.version_migration.context import MigrationContext
from hhemt.version_migration.exceptions import MigrationBlockedError
from hhemt.version_migration.versions import (
    V0005__inline_report_config as V0005,
)

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "legacy_layouts" / "v0005_unit_test"


def _run_upgrade(target_dir: Path) -> None:
    ctx = MigrationContext(
        target_dir=target_dir,
        dry_run=False,
        migration_id="V0005",
    )
    V0005.upgrade(ctx)
    ctx.execute()


def test_v0005_upgrades_recoverable_snakefile(tmp_path):
    """Snakefile carries `--report-config {existing-path}` -> inline the
    referenced YAML into cfg_analysis.yaml::report."""
    src = FIXTURE_ROOT / "recoverable"
    dst = tmp_path / "recoverable"
    shutil.copytree(src, dst)
    src_cfg = Path("/tmp/v0005_fixture_source_cfg.yaml")
    src_cfg.write_text(yaml.safe_dump({"interactive": {"static_backend": "matplotlib"}}))
    try:
        _run_upgrade(dst)
        result = yaml.safe_load((dst / "cfg_analysis.yaml").read_text())
        assert result["report"]["interactive"]["static_backend"] == "matplotlib"
    finally:
        src_cfg.unlink(missing_ok=True)


def test_v0005_raises_when_no_snakefile(tmp_path):
    """No Snakefile -> MigrationBlockedError naming the analysis path and the
    minimum `report:` block to add. Fail-fast invariant: no partial state
    written to cfg_analysis.yaml."""
    src = FIXTURE_ROOT / "no_snakefile"
    dst = tmp_path / "no_snakefile"
    shutil.copytree(src, dst)
    with pytest.raises(MigrationBlockedError) as excinfo:
        _run_upgrade(dst)
    msg = str(excinfo.value)
    assert str(dst) in msg
    assert "no Snakefile present" in msg
    assert "static_backend" in msg
    after = yaml.safe_load((dst / "cfg_analysis.yaml").read_text())
    assert "report" not in after


def test_v0005_raises_when_source_cfg_missing(tmp_path):
    """Snakefile references a missing path -> MigrationBlockedError."""
    src = FIXTURE_ROOT / "missing_source_cfg"
    dst = tmp_path / "missing_source_cfg"
    shutil.copytree(src, dst)
    with pytest.raises(MigrationBlockedError) as excinfo:
        _run_upgrade(dst)
    msg = str(excinfo.value).lower()
    assert "missing" in msg
    after = yaml.safe_load((dst / "cfg_analysis.yaml").read_text())
    assert "report" not in after


def test_v0005_idempotent(tmp_path):
    """Re-running upgrade on a v5 analysis is a byte-identical no-op."""
    src = FIXTURE_ROOT / "v5_recoverable"
    dst = tmp_path / "v5_recoverable"
    shutil.copytree(src, dst)
    src_cfg = Path("/tmp/v0005_fixture_source_cfg.yaml")
    src_cfg.write_text(yaml.safe_dump({"interactive": {"static_backend": "plotly"}}))
    try:
        before = (dst / "cfg_analysis.yaml").read_text()
        _run_upgrade(dst)
        after = (dst / "cfg_analysis.yaml").read_text()
        assert before == after
    finally:
        src_cfg.unlink(missing_ok=True)


def test_v0005_re_raises_on_blocked_target(tmp_path):
    """Re-running upgrade on an unrecoverable v4 target raises the same
    MigrationBlockedError; no partial state stamped between attempts."""
    src = FIXTURE_ROOT / "no_snakefile"
    dst = tmp_path / "blocked"
    shutil.copytree(src, dst)
    with pytest.raises(MigrationBlockedError):
        _run_upgrade(dst)
    with pytest.raises(MigrationBlockedError):
        _run_upgrade(dst)


def test_v0005_skips_system_directory(tmp_path):
    """system_directory passes have no cfg_analysis.yaml — V0005 must
    early-return without error and without raising MigrationBlockedError."""
    system_dir = tmp_path / "system_directory"
    system_dir.mkdir()
    _run_upgrade(system_dir)
    assert not (system_dir / "cfg_analysis.yaml").exists()
