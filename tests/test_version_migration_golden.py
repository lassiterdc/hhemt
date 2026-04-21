"""Golden round-trip tests for version_migration migrations.

Parametrized over every (from_version, to_version) pair with committed
fixtures on both sides. As migrations land in Phase 5, the parametrize list
expands.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest
import yaml

from TRITON_SWMM_toolkit.scenario import compute_event_id_slug
from TRITON_SWMM_toolkit.version_migration import runner
from TRITON_SWMM_toolkit.version_migration.context import MigrationContext
from TRITON_SWMM_toolkit.version_migration.exceptions import (
    MigrationConflictError,
)

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "legacy_layouts"


def _copy_fixture(name: str, dest: Path) -> Path:
    """Copy a committed fixture into tmp_path (preserves the fixture itself)."""
    src = FIXTURE_ROOT / name
    out = dest / name
    shutil.copytree(src, out)
    return out


def _walk_relative(root: Path) -> set[str]:
    return {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() or p.is_dir()}


def _cfg_paths_from_fixture(fixture_dir: Path) -> dict[str, Path]:
    return {
        "system": fixture_dir / "cached_configs" / "system.yaml",
        "analysis": fixture_dir / "cached_configs" / "analysis.yaml",
    }


def _fixture_expected_slugs(fixture_dir: Path) -> set[str]:
    """Read expected slugs directly from fixture's analysis.yaml without
    instantiating TRITONSWMM_analysis. Avoids PySwmm MultiSimulationError
    (CLAUDE.md Gotcha #3) across tests in one session."""
    analysis_yaml = fixture_dir / "cached_configs" / "analysis.yaml"
    data = yaml.safe_load(analysis_yaml.read_text())
    return {compute_event_id_slug(ix) for ix in data.get("expected_weather_indexers", [])}


@pytest.fixture(autouse=True)
def _patch_build_expected_slugs(monkeypatch, request):
    """Reads slugs from fixture's cached_configs/analysis.yaml rather than
    instantiating TRITONSWMM_analysis. Slow test opts out."""
    if "slow" in request.keywords:
        return

    def fake_build(self: MigrationContext) -> set[str]:
        if self.cfg_paths is None:
            return set()
        fixture_dir = self.cfg_paths["analysis"].parent.parent
        return _fixture_expected_slugs(fixture_dir)

    monkeypatch.setattr(MigrationContext, "build_expected_slugs_for_current_version", fake_build)


def test_v0_to_v1_round_trip(tmp_path: Path) -> None:
    work = _copy_fixture("v0", tmp_path)
    expected = FIXTURE_ROOT / "v1"
    result = runner.run_migration(work, target=1, apply=True, cfg_paths=_cfg_paths_from_fixture(work))
    assert result.applied
    assert result.migrations_applied == ["V0001__rename_scenario_dirs"]
    expected_files = _walk_relative(expected) - {"_version.json"}
    actual_files = _walk_relative(work) - {"_version.json"}
    assert expected_files == actual_files


def test_v0_to_v1_dry_run_no_mutation(tmp_path: Path) -> None:
    work = _copy_fixture("v0", tmp_path)
    before = _walk_relative(work)
    runner.run_migration(work, target=1, apply=False, cfg_paths=_cfg_paths_from_fixture(work))
    after = _walk_relative(work)
    assert before == after


def test_v0_to_v1_idempotent(tmp_path: Path) -> None:
    work = _copy_fixture("v0", tmp_path)
    runner.run_migration(work, target=1, apply=True, cfg_paths=_cfg_paths_from_fixture(work))
    state_after_first = json.loads((work / "_version.json").read_text())
    runner.run_migration(work, target=1, apply=True, cfg_paths=_cfg_paths_from_fixture(work))
    state_after_second = json.loads((work / "_version.json").read_text())
    assert state_after_first == state_after_second


def test_v0_to_v1_collision_detection(tmp_path: Path) -> None:
    work = _copy_fixture("v0", tmp_path)
    sims = work / "sims"
    (sims / "2-event_id.0").mkdir()
    with pytest.raises(MigrationConflictError, match="slug collision"):
        runner.run_migration(work, target=1, apply=True, cfg_paths=_cfg_paths_from_fixture(work))


def test_v0_to_v1_unknown_slug_skipped(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    import logging as _lg

    caplog.set_level(_lg.WARNING, logger="TRITON_SWMM_toolkit.version_migration.context")
    work = _copy_fixture("v0", tmp_path)
    sims = work / "sims"
    (sims / "9-orphan_slug").mkdir()
    runner.run_migration(work, target=1, apply=True, cfg_paths=_cfg_paths_from_fixture(work))
    assert (sims / "9-orphan_slug").exists()
    assert (sims / "event_id.0").exists()
    assert any(
        r.levelname == "WARNING" and "unexpected slug" in r.getMessage()
        for r in caplog.records
    )


@pytest.mark.slow
def test_v0_to_v1_against_norfolk_sensitivity_analysis_fixture(
    norfolk_sensitivity_analysis, tmp_path: Path
) -> None:
    """Integration test: backport the live norfolk_sensitivity_analysis tree
    to v0 form, run V0001, assert the original tree shape is restored."""
    analysis_dir = norfolk_sensitivity_analysis.analysis_paths.analysis_dir
    for sims in [analysis_dir / "sims"] + list(analysis_dir.glob("subanalyses/sa_*/sims")):
        for i, entry in enumerate(sorted(sims.iterdir())):
            if entry.is_dir() and not entry.name[0].isdigit():
                entry.rename(sims / f"{i}-{entry.name}")
    result = runner.run_migration(analysis_dir, target=1, apply=True)
    assert result.applied
    pattern = re.compile(r"^\d+-")
    for sims in [analysis_dir / "sims"] + list(analysis_dir.glob("subanalyses/sa_*/sims")):
        for entry in sims.iterdir():
            assert not pattern.match(entry.name), f"unexpected legacy form: {entry}"
