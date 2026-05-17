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


def _discover_fixture_pairs() -> list[tuple[int, int]]:
    """Glob FIXTURE_ROOT for v{N}/ dirs; yield every (from_v, to_v) with from_v < to_v.

    Master plan §7 commits to "detection via fixture glob, not hand-maintained
    parametrize list". When future phases add v5/, this picks it up automatically.
    """
    versions = sorted(
        int(p.name[1:])
        for p in FIXTURE_ROOT.iterdir()
        if p.is_dir() and p.name.startswith("v") and p.name[1:].isdigit()
    )
    return [(a, b) for i, a in enumerate(versions) for b in versions[i + 1 :]]


@pytest.mark.parametrize(
    "from_v, to_v",
    _discover_fixture_pairs(),
    ids=lambda p: f"v{p}",
)
def test_pair_round_trip(from_v: int, to_v: int, tmp_path: Path) -> None:
    work = _copy_fixture(f"v{from_v}", tmp_path)
    expected = FIXTURE_ROOT / f"v{to_v}"
    runner.run_migration(
        work, target=to_v, apply=True, cfg_paths=_cfg_paths_from_fixture(work)
    )
    expected_files = _walk_relative(expected) - {"_version.json"}
    actual_files = _walk_relative(work) - {"_version.json"}
    assert expected_files == actual_files, (
        f"v{from_v} -> v{to_v} mismatch: "
        f"missing={expected_files - actual_files}, extra={actual_files - expected_files}"
    )


@pytest.mark.parametrize(
    "from_v, to_v",
    _discover_fixture_pairs(),
    ids=lambda p: f"v{p}",
)
def test_pair_round_trip_idempotent(from_v: int, to_v: int, tmp_path: Path) -> None:
    """Apply (from_v, to_v) twice; second apply must be a no-op.

    Asserts: (a) migration_history length unchanged, (b) tree walk identical.
    """
    work = _copy_fixture(f"v{from_v}", tmp_path)
    cfg = _cfg_paths_from_fixture(work)
    runner.run_migration(work, target=to_v, apply=True, cfg_paths=cfg)
    state_first = json.loads((work / "_version.json").read_text())
    tree_first = _walk_relative(work)
    runner.run_migration(work, target=to_v, apply=True, cfg_paths=cfg)
    state_second = json.loads((work / "_version.json").read_text())
    tree_second = _walk_relative(work)
    assert len(state_second["migration_history"]) == len(
        state_first["migration_history"]
    )
    assert tree_first == tree_second


def test_v0001_tolerates_mixed_completion_state(tmp_path: Path) -> None:
    """V0001 must migrate complete + incomplete legacy sims dirs uniformly.

    Synthesizes a v0 tree with 5 legacy sims dirs (3 complete with all log_*.json
    files, 2 incomplete missing some logs). Asserts V0001 renames all 5 to
    sims/{slug}/ form with no errors and no skips.
    """
    work = _copy_fixture("v0", tmp_path)
    sims = work / "sims"
    # Existing fixture has sims/0-event_id.0/ already (complete). Add 4 more.
    for i in range(1, 5):
        scenario = sims / f"{i}-event_id.{i}"
        (scenario / "build").mkdir(parents=True)
        (scenario / "build" / ".gitkeep").touch()
        (scenario / "log_triton.json").write_text("{}")
        if i < 3:
            # Complete: all three log files
            (scenario / "log_tritonswmm.json").write_text("{}")
            (scenario / "log_swmm.json").write_text("{}")
        # i in {3, 4}: incomplete (missing log_tritonswmm.json + log_swmm.json)

    # Also extend analysis.yaml expected_weather_indexers so all 5 slugs are valid
    analysis_yaml = work / "cached_configs" / "analysis.yaml"
    data = yaml.safe_load(analysis_yaml.read_text())
    existing = list(data.get("expected_weather_indexers", []))
    seen_keys = {tuple(sorted(x.items())) for x in existing}
    for i in range(1, 5):
        candidate = {"event_id": f"{i}"}
        if tuple(sorted(candidate.items())) not in seen_keys:
            existing.append(candidate)
    data["expected_weather_indexers"] = existing
    analysis_yaml.write_text(yaml.safe_dump(data, sort_keys=False))

    runner.run_migration(work, target=1, apply=True, cfg_paths=_cfg_paths_from_fixture(work))

    # All 5 should be renamed
    for i in range(5):
        renamed = sims / f"event_id.{i}"
        assert renamed.is_dir(), f"V0001 did not rename sims/{i}-event_id.{i}/ to sims/event_id.{i}/"
    # No leftover legacy entries
    legacy = re.compile(r"^\d+-")
    for entry in sims.iterdir():
        assert not legacy.match(entry.name), f"unexpected legacy form: {entry}"


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


def test_v5_to_v6_preserves_fingerprint_mtime(tmp_path: Path) -> None:
    """V0006 rewrites fingerprint payloads without bumping mtime.

    Snakemake rerun-triggers include "mtime"; a v1→v3 schema upgrade that
    bumps mtime would spuriously rerun every sa_id chain. Asserts the
    rewritten file's mtime is within 1 second of the original (filesystem
    timestamp resolution).
    """
    work = _copy_fixture("v5", tmp_path)
    fp = work / "_status" / "sa-0_inputs.json"
    # Create the v1 fingerprint at a known-old mtime
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(
        '{"__schema_version__":1,"fields":{"cpus_per_sim":1,"n_omp_threads":1,"run_mode":"serial"}}\n'
    )
    import os
    old_mtime = 1_700_000_000.0
    os.utime(fp, (old_mtime, old_mtime))

    runner.run_migration(
        work, target=6, apply=True, cfg_paths=_cfg_paths_from_fixture(work)
    )

    new_mtime = fp.stat().st_mtime
    assert abs(new_mtime - old_mtime) < 1.0, (
        f"V0006 bumped mtime from {old_mtime} to {new_mtime}; "
        "fingerprint rewrite must preserve mtime per workflow.py:1609 "
        "rerun-triggers contract"
    )
    # And the content must be at schema v3
    assert json.loads(fp.read_text())["__schema_version__"] == 3
