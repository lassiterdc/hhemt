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
    assert any(r.levelname == "WARNING" and "unexpected slug" in r.getMessage() for r in caplog.records)


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
    runner.run_migration(work, target=to_v, apply=True, cfg_paths=_cfg_paths_from_fixture(work))
    expected_files = _walk_relative(expected) - {"_version.json"}
    actual_files = _walk_relative(work) - {"_version.json"}
    assert expected_files == actual_files, (
        f"v{from_v} -> v{to_v} mismatch: missing={expected_files - actual_files}, extra={actual_files - expected_files}"
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
    assert len(state_second["migration_history"]) == len(state_first["migration_history"])
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
def test_v0_to_v1_against_norfolk_sensitivity_analysis_fixture(norfolk_sensitivity_analysis, tmp_path: Path) -> None:
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


def test_v5_to_v6_pins_fingerprint_mtime_to_prepare_flag_reference(tmp_path: Path) -> None:
    """V0006 rewrites fingerprint payloads with mtime pinned to a downstream
    output's reference mtime, NOT preserved from the (possibly bumped) prior
    fingerprint mtime.

    Snakemake rerun-triggers include "mtime". The fingerprint is an input to
    the per-sa_id prepare rule; the prepare rule's output is
    `_status/b_prepare_sa-{sa_id}_*_complete.flag`. To prevent Snakemake from
    planning a rerun, the fingerprint's mtime must be ≤ the prepare-flag's
    mtime. V0006's `_resolve_reference_mtime` picks the prepare-flag's mtime
    as priority-1 reference precisely so this invariant holds.

    This test fabricates the operational failure mode encountered on Rivanna
    (2026-05-17): a fingerprint touched AFTER the original prepare-flag was
    written, then "preserved" by the prior V0006 implementation as wall-clock
    time. The new V0006 implementation pins to the prepare-flag's mtime
    regardless of the fingerprint's current mtime, so the cascade does not
    fire.
    """
    work = _copy_fixture("v5", tmp_path)
    status_dir = work / "_status"
    status_dir.mkdir(parents=True, exist_ok=True)

    # Set up the v1 fingerprint at a recently-bumped (wall-clock-ish) mtime —
    # simulating the failure mode where `analysis.run(dry_run=True)` was
    # invoked after the breaking feature landed but before V0006 ran.
    fp = status_dir / "sa-0_inputs.json"
    fp.write_text('{"__schema_version__":1,"fields":{"cpus_per_sim":1,"n_omp_threads":1,"run_mode":"serial"}}\n')
    import os

    bumped_mtime = 1_800_000_000.0  # represents "today's wall-clock"
    os.utime(fp, (bumped_mtime, bumped_mtime))

    # Pre-place a prepare flag at a known-earlier mtime — this is the reference
    # V0006 should pin to (priority 1 in _resolve_reference_mtime).
    prepare_flag = status_dir / "b_prepare_sa-0_evt-year.9_event_type.compound_event_id.1_complete.flag"
    prepare_flag.write_text("")
    prepare_flag_mtime = 1_700_000_000.0  # represents "original May 13 run"
    os.utime(prepare_flag, (prepare_flag_mtime, prepare_flag_mtime))

    runner.run_migration(work, target=6, apply=True, cfg_paths=_cfg_paths_from_fixture(work))

    new_mtime = fp.stat().st_mtime
    assert abs(new_mtime - prepare_flag_mtime) < 1.0, (
        f"V0006 left fingerprint mtime at {new_mtime}; expected to be pinned "
        f"to prepare-flag reference {prepare_flag_mtime} (within filesystem "
        "timestamp resolution). The fingerprint must end up ≤ prepare-flag's "
        "mtime to prevent Snakemake's mtime rerun trigger from firing."
    )
    # And the content must be at schema v3
    assert json.loads(fp.read_text())["__schema_version__"] == 3


def test_v6_to_v7_clears_snakemake_metadata_with_backup(tmp_path: Path) -> None:
    """V0007 clears `.snakemake/metadata/` so Snakemake's `--rerun-triggers
    input` set-comparison does not fire on the rename-induced input-set
    change, and retains a backup at `.snakemake/metadata.bak.V0007`.

    Without this clear, the per-sa_id rules' persisted metadata still
    references the old `a_setup_complete.flag` name even after V0007 renames
    the file. Snakemake reads the metadata, compares to the new Snakefile's
    declaration, detects the set-change, and plans reruns for every affected
    rule — defeating the migration's purpose.
    """
    work = _copy_fixture("v5", tmp_path)
    # V0007 requires subanalyses/ (sensitivity-only); the v5 fixture has it
    assert (work / "subanalyses").is_dir()

    # Pre-place the legacy flag V0007 will rename
    (work / "_status").mkdir(parents=True, exist_ok=True)
    legacy_flag = work / "_status" / "a_setup_complete.flag"
    legacy_flag.write_text("")

    # Pre-populate .snakemake/metadata/ with a fake rule record to verify the
    # clear-with-backup behavior. The content doesn't need to be valid
    # Snakemake metadata — only that the dir exists and gets cleared+backed up.
    metadata_dir = work / ".snakemake" / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    fake_record = metadata_dir / "fake_rule_record.json"
    fake_record.write_text('{"input_set": ["_status/a_setup_complete.flag"]}\n')

    runner.run_migration(work, target=7, apply=True, cfg_paths=_cfg_paths_from_fixture(work))

    assert not metadata_dir.exists(), (
        "V0007 must remove .snakemake/metadata/ after backing it up; the "
        "metadata still references the pre-rename input-file name and would "
        "trigger Snakemake's set-change rerun cascade"
    )
    backup_dir = work / ".snakemake" / "metadata.bak.V0007"
    assert backup_dir.is_dir(), (
        "V0007 must back up the cleared metadata to .snakemake/metadata.bak.V0007 for audit + recovery"
    )
    assert (backup_dir / "fake_rule_record.json").is_file(), "Backup must contain the original metadata files"
    # And the flag rename must have happened
    assert not legacy_flag.exists()
    assert (work / "_status" / "a_setup_target_0_complete.flag").is_file()


def test_v6_to_v7_metadata_clear_idempotent_on_rerun(tmp_path: Path) -> None:
    """Re-running V0007 against an already-migrated tree must not double-back-up
    the metadata (the existing backup would be silently overwritten otherwise).
    The primitive's existing-backup short-circuit handles this.
    """
    work = _copy_fixture("v5", tmp_path)
    (work / "_status").mkdir(parents=True, exist_ok=True)
    (work / "_status" / "a_setup_complete.flag").write_text("")
    metadata_dir = work / ".snakemake" / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / "original.json").write_text("original")

    # First run: clear + backup
    runner.run_migration(work, target=7, apply=True, cfg_paths=_cfg_paths_from_fixture(work))
    backup_dir = work / ".snakemake" / "metadata.bak.V0007"
    assert backup_dir.is_dir()
    original_backup_content = (backup_dir / "original.json").read_text()
    assert original_backup_content == "original"

    # Simulate a subsequent Snakemake run repopulating metadata with new content
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / "new_post_migration.json").write_text("regenerated")

    # Second run: must NOT overwrite the existing backup. The primitive's
    # short-circuit means metadata_dir is left alone too (re-clear would
    # require a new backup_label).
    runner.run_migration(work, target=7, apply=True, cfg_paths=_cfg_paths_from_fixture(work))
    # Backup retained with original content
    assert (backup_dir / "original.json").read_text() == "original"
    # And the second backup-conflict short-circuit means the post-migration
    # metadata is not destroyed
    assert (metadata_dir / "new_post_migration.json").read_text() == "regenerated"


def test_v9_to_v10_backfills_plot_id_and_orphans_old_stem_figures(
    tmp_path: Path,
) -> None:
    """V0010 backfills the canonical ``plot_id`` into a figure manifest (derived
    from the manifest's ``plots/`` path) and orphan-deletes the old-stem figure
    siblings, leaving the manifest as the regenerated figure's provenance anchor.

    The committed golden fixtures are intentionally plots-less, so the
    parametrized ``test_pair_round_trip`` exercises only V0010's no-op path (a
    plots-less analysis tree is unchanged by V0010, hence the v10 fixture is
    file-set-identical to v9). This test pre-places a minimal ``plots/`` subtree
    -- the established ``test_v5_to_v6`` / ``test_v6_to_v7`` pattern of seeding a
    copied fixture -- to exercise V0010's actual backfill + orphan-sweep logic,
    which the file-SET round-trip walk cannot (it ignores file content, and a
    committed post-migration fixture would have to carry the FileLock ``.lock``
    artifact ``log_add_field`` leaves on disk).
    """
    work = _copy_fixture("v9", tmp_path)
    # Pre-place a v9-form per-sim figure: renderer-kind-only stem, event_id in the
    # directory path, manifest WITHOUT plot_id, plus the old-stem figure siblings.
    event_dir = work / "plots" / "per_sim" / "event_id.0"
    event_dir.mkdir(parents=True)
    manifest_path = event_dir / "peak_flood_depth.manifest.json"
    manifest_path.write_text(json.dumps({"renderer_module": "per_sim_peak_flood_depth"}))
    figure = event_dir / "peak_flood_depth.png"
    figure.write_bytes(b"\x89PNG\r\n\x1a\n")  # non-empty
    preview = event_dir / "peak_flood_depth.preview.png"
    preview.write_bytes(b"\x89PNG\r\n\x1a\n")

    runner.run_migration(work, target=10, apply=True, cfg_paths=_cfg_paths_from_fixture(work))

    # Backfill: plot_id == canonical id derived from the manifest's plots/ path
    # (renderer-kind stem + evt.{event_id from the directory}), NOT re-minted via
    # workflow.py -- so the migration carries no drift surface with the renderer.
    migrated = json.loads(manifest_path.read_text())
    assert migrated["plot_id"] == "peak_flood_depth__evt.event_id.0"
    # Orphan-sweep: old-stem figure siblings removed; manifest retained.
    assert not figure.exists()
    assert not preview.exists()
    assert manifest_path.exists()

    # Idempotent on the mutation path: re-running V0010 leaves plot_id stable and
    # does not error on the already-removed figures (guarded_remove no-ops).
    runner.run_migration(work, target=10, apply=True, cfg_paths=_cfg_paths_from_fixture(work))
    assert json.loads(manifest_path.read_text())["plot_id"] == "peak_flood_depth__evt.event_id.0"
    assert not figure.exists()
