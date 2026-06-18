"""V0008 fix-per-rank-diff-aggregation migration: branch coverage + idempotency.

Fixtures live under ``tests/fixtures/legacy_layouts/v0008_unit_test/`` rather
than under ``v7/`` or ``v8/`` to avoid colliding with the auto-discovery glob
in ``test_version_migration_golden.py::_discover_fixture_pairs`` (which
requires each ``v{N}/`` directory to be a single full-corpus chain fixture
participating in every cross-pair (a, b) round-trip).

Two fixture variants exercise the two branches of V0008.upgrade:

- ``regenerate/`` — has both raw ``out_tritonswmm/performance/performance{N}.txt``
  files AND the legacy (broken) ``processed/TRITONSWMM_perf_tseries.zarr`` and
  ``TRITONSWMM_perf_summary.zarr``. V0008._regenerate_perf_summary should fire.
- ``stamp_stale/`` — has only the legacy ``processed/TRITONSWMM_perf_*.zarr``
  stores; the raw ``out_tritonswmm/performance/`` directory is absent.
  V0008._stamp_stale should fire and write a
  ``_V0008_legacy_perf_summary_stale.json`` sidecar marker next to the
  un-regeneratable legacy zarr.

The synth regression test at ``tests/test_synth_03_perf_tseries_diff.py``
covers the module-level helpers (``_aggregate_perf_tseries``,
``_aggregate_perf_summary``) directly with controlled synthetic inputs; this
file covers the V0008 migration-orchestration logic that delegates to those
helpers and writes the regenerated zarr stores plus the migration_history
stamp.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import xarray as xr

from hhemt.version_migration import runner

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "legacy_layouts" / "v0008_unit_test"

_SIM_SUBPATH = Path("subanalyses") / "sa_22" / "sims" / "year.9_event_type.compound_event_id.1"


def _copy_variant(name: str, tmp_path: Path) -> Path:
    src = FIXTURE_ROOT / name
    dst = tmp_path / name
    shutil.copytree(src, dst)
    return dst


def test_v0008_regenerates_when_raw_perf_present(tmp_path: Path) -> None:
    """Raw performance/ directory present -> V0008 regenerates both per-scenario
    zarr stores with corrected values, leaves migration_history clean of stale
    markers."""
    work = _copy_variant("regenerate", tmp_path)
    perf_summary_path = work / _SIM_SUBPATH / "processed" / "TRITONSWMM_perf_summary.zarr"
    perf_tseries_path = work / _SIM_SUBPATH / "processed" / "TRITONSWMM_perf_tseries.zarr"

    runner.run_migration(work, target=8, apply=True)

    # Both zarr stores still exist after V0008 regenerates them in place.
    assert perf_summary_path.is_dir(), "regenerated perf_summary.zarr should exist"
    assert perf_tseries_path.is_dir(), "regenerated perf_tseries.zarr should exist"

    # The regenerated summary opens via xarray and exposes the wallclock-semantic
    # variables. The notes attr carries V0008's regeneration marker.
    ds = xr.open_zarr(perf_summary_path, consolidated=False)
    assert "Total" in ds.data_vars, "regenerated summary must expose the Total variable"
    assert "Simulation" in ds.data_vars, "regenerated summary must expose the Simulation variable"
    assert "SWMM" in ds.data_vars, "regenerated summary must expose the SWMM variable"
    assert float(ds["Total"].item()) > 0.0, "Total wallclock must be positive for a real run"
    notes = ds.attrs.get("notes", "")
    assert "V0008-regenerated" in notes, (
        f"regenerated summary notes attr should mention V0008-regenerated; got {notes!r}"
    )

    # The regenerate branch does NOT write a stale-marker sidecar — that
    # marker is reserved for the stamp_stale branch (raw perf absent).
    stale_marker = perf_summary_path.parent / "_V0008_legacy_perf_summary_stale.json"
    assert not stale_marker.exists(), (
        f"regenerate branch must not emit a stale marker sidecar; got {stale_marker}"
    )

    # And the layout_version is now 8.
    state = json.loads((work / "_version.json").read_text())
    assert state["layout_version"] == 8


def test_v0008_stamps_stale_when_raw_perf_absent(tmp_path: Path) -> None:
    """Raw performance/ directory absent -> V0008 stamps migration_history with
    legacy_perf_summary_stale entry; legacy zarr stores are left untouched
    (no source of truth to re-aggregate from)."""
    work = _copy_variant("stamp_stale", tmp_path)
    perf_summary_path = work / _SIM_SUBPATH / "processed" / "TRITONSWMM_perf_summary.zarr"
    legacy_summary_mtime_before = perf_summary_path.stat().st_mtime_ns

    runner.run_migration(work, target=8, apply=True)

    # The legacy summary zarr should be unchanged (no rewrite).
    assert perf_summary_path.is_dir(), "legacy perf_summary.zarr should still exist"
    legacy_summary_mtime_after = perf_summary_path.stat().st_mtime_ns
    assert legacy_summary_mtime_before == legacy_summary_mtime_after, (
        "stamp_stale branch must not rewrite the legacy perf_summary.zarr"
    )

    # A sidecar stale-marker file is written next to the un-regeneratable
    # legacy zarr.
    stale_marker = perf_summary_path.parent / "_V0008_legacy_perf_summary_stale.json"
    assert stale_marker.is_file(), (
        f"stamp_stale branch must write a sidecar marker at {stale_marker}"
    )
    marker_payload = json.loads(stale_marker.read_text())
    assert marker_payload["version_from"] == 7
    assert marker_payload["version_to"] == 8
    assert marker_payload["migration_id"] == "V0008__fix_per_rank_diff_aggregation"
    assert "perf_tseries_zarr_relative_to_analysis_dir" in marker_payload

    # Layout version bumps to 8 via the runner's standard record_migration.
    state = json.loads((work / "_version.json").read_text())
    assert state["layout_version"] == 8


def test_v0008_idempotent_on_rerun(tmp_path: Path) -> None:
    """Running V0008 twice via the runner is a no-op on the second pass:
    migration_history length unchanged, _version.json byte-equivalent."""
    work = _copy_variant("regenerate", tmp_path)

    runner.run_migration(work, target=8, apply=True)
    state_first = json.loads((work / "_version.json").read_text())

    runner.run_migration(work, target=8, apply=True)
    state_second = json.loads((work / "_version.json").read_text())

    # The runner detects layout_version is already 8 on the second pass and
    # plans no migrations; migration_history therefore doesn't grow.
    assert len(state_second.get("migration_history", [])) == len(
        state_first.get("migration_history", [])
    )
    assert state_first == state_second
