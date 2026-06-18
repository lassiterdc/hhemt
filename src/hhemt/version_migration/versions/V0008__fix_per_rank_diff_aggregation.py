"""V0008 — Fix per-rank diff aggregation in performance summary.

The pre-V0008 ``_export_performance_tseries`` called ``pd.diff()`` on a
``(timestep_min, Rank)`` MultiIndex without grouping by Rank, producing
inter-rank-skew "deltas" that telescoped to incorrect per-rank cumulative
values. Downstream ``_export_performance_summary``'s ``mean(Rank)`` of these
broken sums equals ``last_rank_final_cumulative / N_ranks`` — wrong by a
factor of approximately ``N_ranks`` for wallclock-semantic columns.

This migration:

  1. If ``TRITONSWMM_perf_tseries.zarr`` (the per-scenario raw timeseries)
     is still present on disk AND the raw ``performance/`` directory still
     contains per-checkpoint ``performance{N}.txt`` files, re-run the
     corrected aggregation in-place to regenerate
     ``TRITONSWMM_perf_summary.zarr`` (and ``TRITONSWMM_perf_tseries.zarr``)
     with correct values. Idempotent: re-running is a no-op once correct
     values are present.
  2. If the raw timeseries has been cleaned (per
     ``_clear_raw_TRITON_outputs``), stamp the analysis dir's
     ``_version.json::migration_history`` with a ``legacy_perf_summary_stale``
     entry but DO NOT overwrite the legacy ``*_perf_summary.zarr`` values —
     there is no source-of-truth to re-aggregate from. Emit a clear log
     message so operators see they need to re-run the workflow to regenerate.

The corrected aggregation algorithm is the module-level
``process_simulation._aggregate_perf_tseries`` /
``_aggregate_perf_summary`` pair shipped in the same commit as this
migration. The migration delegates to those helpers rather than
reimplementing the algorithm inline — see Phase 1 Spec 8 of the
``superlinear-speedup-fixes`` master plan for the architectural rationale.

References:

  - Decision rationale + verified numerical values:
    ``library/docs/decisions/hhemt/LAYOUT_VERSION 8 fix per rank diff in performance aggregation.md``
    (created in the same commit as this migration).
  - Regression test (canonical durable reproducer):
    ``tests/test_synth_03_perf_tseries_diff.py``.
  - The bug: ``src/hhemt/process_simulation.py`` line 435 in
    pre-V0008 commits.
  - The fix: ``groupby(level='Rank').diff()`` replacement at the same site,
    plus extraction of ``_aggregate_perf_tseries`` / ``_aggregate_perf_summary``
    as module-level helpers.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from hhemt.version_migration.context import MigrationContext

version_from: int = 7
version_to: int = 8
description: str = (
    "Fix per-rank diff aggregation in _export_performance_tseries: replace "
    ".diff() that crosses rank boundaries with groupby(level='Rank').diff(). "
    "Legacy per-scenario perf_tseries.zarr and perf_summary.zarr (pre-V0008) "
    "contain values wrong by factor ~N_ranks; this migration regenerates them "
    "when raw timeseries are present and stamp-marks the analysis dir otherwise."
)

logger = logging.getLogger(__name__)


def upgrade(ctx: MigrationContext) -> None:
    """Re-aggregate or stamp-mark each per-scenario performance summary.

    Iterates every ``subanalyses/sa_*/sims/.../processed/TRITONSWMM_perf_tseries.zarr``
    (and the TRITON-only variant — both share the same raw-input layout under
    ``out_tritonswmm/performance/``). For each:

      - If the source raw ``performance/`` directory still exists with
        per-checkpoint files, re-run ``_aggregate_perf_tseries`` /
        ``_aggregate_perf_summary`` against it and overwrite the per-scenario
        zarr stores.
      - Otherwise stamp ``migration_history`` with
        ``legacy_perf_summary_stale: True``.
    """
    target_dir = ctx.target_dir
    # Search both the multisim layout (sims/...) and the sensitivity layout (subanalyses/sa_*/sims/...).
    perf_zarr_patterns = [
        "sims/*/processed/TRITONSWMM_perf_tseries.zarr",
        "subanalyses/sa_*/sims/*/processed/TRITONSWMM_perf_tseries.zarr",
    ]
    perf_zarrs: list[Path] = []
    for pattern in perf_zarr_patterns:
        perf_zarrs.extend(target_dir.glob(pattern))

    if not perf_zarrs:
        logger.info("V0008: no per-scenario perf_tseries.zarr found; nothing to migrate")
        return

    regenerated_count = 0
    stamp_only_count = 0
    for perf_tseries_zarr in perf_zarrs:
        # perf_tseries_zarr layout: <sim_dir>/processed/TRITONSWMM_perf_tseries.zarr
        sim_dir = perf_tseries_zarr.parent.parent
        raw_perf_dir = sim_dir / "out_tritonswmm" / "performance"
        if raw_perf_dir.exists() and any(raw_perf_dir.glob("performance*.txt")):
            _regenerate_perf_summary(sim_dir, raw_perf_dir)
            regenerated_count += 1
        else:
            _stamp_stale(target_dir, perf_tseries_zarr)
            stamp_only_count += 1

    logger.info(
        "V0008 complete: regenerated %d perf_summary.zarr from raw timeseries; "
        "stamp-marked %d as legacy_perf_summary_stale.",
        regenerated_count,
        stamp_only_count,
    )


def _regenerate_perf_summary(sim_dir: Path, raw_perf_dir: Path) -> None:
    """Re-run the corrected aggregation against the raw per-checkpoint files.

    Delegates to the module-level ``_aggregate_perf_tseries`` and
    ``_aggregate_perf_summary`` helpers in ``process_simulation.py`` — those
    are the canonical aggregator and ship in the same commit as this migration.
    The migration regenerates the per-scenario ``perf_tseries.zarr`` and
    ``perf_summary.zarr`` by invoking the corrected helpers and writing the
    outputs through xarray's ``to_zarr``.
    """
    # Local import to avoid a top-level cycle (process_simulation imports
    # from this version_migration package via the toolkit-level migration
    # registry; importing it at module-load time would invert the dependency).
    from hhemt.process_simulation import (
        _aggregate_perf_summary,
        _aggregate_perf_tseries,
    )

    ds_tseries = _aggregate_perf_tseries(raw_perf_dir)
    perf_tseries_path = sim_dir / "processed" / "TRITONSWMM_perf_tseries.zarr"
    ds_tseries.to_zarr(perf_tseries_path, mode="w", consolidated=False)

    ds_summary = _aggregate_perf_summary(raw_perf_dir)
    ds_summary.attrs["units"] = "seconds"
    ds_summary.attrs["notes"] = (
        "V0008-regenerated: slowest-rank cumulative cost. 'Total' / 'Simulation' "
        "/ 'Init' ≈ wallclock elapsed from triton.exe start through final "
        "checkpoint barrier. Category columns are upper bounds on per-category "
        "contribution to wallclock (slowest-rank category cost, not per-rank "
        "means)."
    )
    perf_summary_path = sim_dir / "processed" / "TRITONSWMM_perf_summary.zarr"
    ds_summary.to_zarr(perf_summary_path, mode="w", consolidated=False)
    logger.info("V0008: regenerated %s", perf_summary_path)


_STALE_MARKER_FILENAME = "_V0008_legacy_perf_summary_stale.json"


def _stamp_stale(target_dir: Path, perf_tseries_zarr: Path) -> None:
    """Write a sidecar marker file next to the un-regeneratable legacy zarr.

    The marker lives at ``<sim_dir>/processed/_V0008_legacy_perf_summary_stale.json``
    and contains a small JSON payload with the relative perf_tseries_zarr path
    and the bump from version 7→8. Downstream consumers (renderers, operators)
    can grep for the marker filename to identify scenarios whose
    ``TRITONSWMM_perf_summary.zarr`` carries pre-V0008 (numerically wrong)
    values.

    Idempotent: re-running V0008 with the marker already present overwrites it
    with the same content. The migration_history record itself is owned by the
    runner (``state.record_migration``), which produces a properly-typed
    ``HistoryEntry`` — this sidecar carries the per-zarr-path detail the
    history-entry schema does not accommodate.
    """
    marker_path = perf_tseries_zarr.parent / _STALE_MARKER_FILENAME
    relative = perf_tseries_zarr.relative_to(target_dir)
    payload = {
        "version_from": 7,
        "version_to": 8,
        "migration_id": "V0008__fix_per_rank_diff_aggregation",
        "perf_tseries_zarr_relative_to_analysis_dir": str(relative),
        "note": (
            "Pre-V0008 perf_summary.zarr at this path contains values wrong "
            "by approximately a factor of N_ranks (the cross-rank pd.diff() "
            "bug). Raw out_tritonswmm/performance/ directory was absent at "
            "migration time, so V0008 could not re-aggregate. Re-run the "
            "workflow's process_sa_* / process rule to regenerate the "
            "corrected summary."
        ),
    }
    marker_path.write_text(json.dumps(payload, indent=2))
    logger.warning(
        "V0008: cannot regenerate %s — raw performance/ directory absent. "
        "Wrote stale marker at %s. Re-run the workflow to regenerate.",
        perf_tseries_zarr.parent / "TRITONSWMM_perf_summary.zarr",
        marker_path,
    )
