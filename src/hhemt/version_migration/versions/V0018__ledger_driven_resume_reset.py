"""V0018 — Regenerate per-scenario performance zarrs under ledger-driven resume-reset detection.

Pre-V0018, ``_aggregate_perf_tseries`` INFERRED resume boundaries from the data:
``idx_resets = (deltas <= 0).all(axis=1)``, a conjunction requiring EVERY one of the nine
columns to decrease. At a real boundary the restarted process re-pays initialization, so
``Init`` INCREASES (measured 0.05694 -> 0.07816 s on sa_serial_6_r1), one positive column
defeats ``.all()``, the reset is missed, and a large negative delta is summed as elapsed
time. Measured blast radius on the synth_cc campaign: 56 of 112 sims -- every resumed sim
on both models, zero clean sims. On sa_serial_6_r1 the on-disk Total read 300.5 s against
a correct 409.30 s.

The fix replaces inference with a ledger join on
``TRITONSWMM_model_log.resume_reporting_tsteps``. This migration regenerates the on-disk
zarrs, which were written by the old code and which ``_export_performance_tseries``'s
``_already_written`` guard will never revisit on its own.

WHY FOUR GLOB FAMILIES AND NOT V0008'S TWO -- read before "simplifying" the globs.
V0008's docstring claims it covers "the TRITON-only variant -- both share the same
raw-input layout under ``out_tritonswmm/performance/``". BOTH HALVES ARE FALSE. It globs
only ``TRITONSWMM_perf_tseries.zarr`` and hardcodes ``out_tritonswmm/performance``, while
the TRITON-only outputs are ``TRITON_only_perf_tseries.zarr`` /
``TRITON_only_perf_summary.zarr`` under ``out_triton/performance`` (scenario.py). The
families share neither filename nor directory, so V0008 was a SILENT NO-OP on every
pure-TRITON sim -- it iterated zero files, logged "nothing to migrate", and exited 0. The
campaign this migration exists to repair is pure-TRITON. Coverage is pinned by a test
parametrized over both families, which fails on the TRITON-only arm if these globs ever
regress to V0008's shape.

WHY THE EMPTY-LEDGER BRANCH SKIPS RATHER THAN REWRITES.
With no recorded resume there are no reset rows to select, so the old inferred predicate
and the new ledger join agree EXACTLY -- both reduce to a plain per-rank diff with the
head-fill. Rewriting would produce byte-identical numbers while touching mtimes, firing
Snakemake's mtime rerun triggers across the consolidate/report cascade for no gain.
Skipping keeps every clean-arm tree byte-unchanged and keeps this migration's diff
auditable: whatever it rewrote, it rewrote because a resume was recorded.

WHY CONSOLIDATION IS INVALIDATED BY SIGNAL, NOT BY DELETION.
The consolidate step is guarded by a three-way gate (processing_analysis.py): it SKIPS
when the tree exists, the log says complete, and the stamped
``consolidation_inputs_fingerprint`` matches; otherwise it ``fast_rmtree``s and rebuilds.
That fingerprint covers tree SHAPE only (consolidation_version,
toggle_consolidate_timeseries, enabled_model_types) under an explicit contract to exclude
byte-only changes. This migration changes VALUES, not shape -- so the fingerprint is
invariant and the gate would SKIP, leaving corrected per-scenario numbers stranded below a
stale tree. Two signals are therefore cleared: the ``e_consolidate_*`` /
``f_consolidate_master`` status flags (so Snakemake re-plans the rule at all) and
``datatree_consolidation_complete`` in log.json (so the rule rebuilds instead of skipping).
The migration deletes NO consolidated artifact -- the gate performs its own
``fast_rmtree``, inside the module that owns consolidation.

DRY-RUN CONTRACT -- deliberately NOT copied from V0008. The runner calls ``upgrade()`` on
BOTH paths (runner.py:163) and gates only ``ctx.execute()`` (runner.py:180), so a migration
that writes directly inside ``upgrade()`` mutates on a dry run. V0008 does exactly that and
is unsafe under ``run_migration(apply=False)``. This module performs its full scan and
classification read-only, and gates every write on ``ctx.dry_run``, so a dry run reports
what it would do and touches nothing.

References:
  - Fix: ``src/hhemt/process_simulation.py::_aggregate_perf_tseries``.
  - Finding: ``library/knowledge/hhemt/perf reset detection is ledger joinable by integer
    index and the segment final row is an exact per column oracle.md``.
  - Behavioural test: ``tests/test_version_migration_V0018.py``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from hhemt.version_migration.context import MigrationContext

version_from: int = 17
version_to: int = 18
description: str = (
    "Regenerate per-scenario perf_tseries/perf_summary zarrs under ledger-driven "
    "resume-reset detection. Pre-V0018 values for RESUMED sims are wrong: the "
    "(deltas <= 0).all(axis=1) predicate missed boundaries where Init increased, summing "
    "a large negative delta as elapsed time. Regenerates where the raw performance{N}.txt "
    "set survives; stamp-marks as uncorrectable where it does not; skips never-resumed "
    "sims (old and new agree exactly). Covers BOTH the coupled and TRITON-only families."
)

logger = logging.getLogger(__name__)

_STALE_MARKER_FILENAME = "_V0018_resume_reset_uncorrectable.json"

#: (perf-zarr glob, raw perf subdir, model-log basename, summary-zarr basename).
#: The cross product V0008 got wrong. Four entries, not two -- see the module docstring.
_FAMILIES: list[tuple[str, str, str, str]] = [
    (
        "sims/*/processed/TRITONSWMM_perf_tseries.zarr",
        "out_tritonswmm",
        "log_tritonswmm.json",
        "TRITONSWMM_perf_summary.zarr",
    ),
    (
        "subanalyses/sa_*/sims/*/processed/TRITONSWMM_perf_tseries.zarr",
        "out_tritonswmm",
        "log_tritonswmm.json",
        "TRITONSWMM_perf_summary.zarr",
    ),
    (
        "sims/*/processed/TRITON_only_perf_tseries.zarr",
        "out_triton",
        "log_triton.json",
        "TRITON_only_perf_summary.zarr",
    ),
    (
        "subanalyses/sa_*/sims/*/processed/TRITON_only_perf_tseries.zarr",
        "out_triton",
        "log_triton.json",
        "TRITON_only_perf_summary.zarr",
    ),
]


def _resume_steps_for(sim_dir: Path, log_name: str) -> list[int]:
    """Realized resume boundaries from the per-sim model log.

    Absent / legacy / unparseable -> [] (no recorded resume). Coalescing to [] is correct
    rather than lenient: the ledger is the ONLY admissible source, so its absence means
    "no resume recorded", never "go infer one from the data".
    """
    p = sim_dir / log_name
    if not p.exists():
        return []
    try:
        payload = json.loads(p.read_text())
    except (OSError, ValueError):
        return []
    return [int(v) for v in (payload.get("resume_reporting_tsteps") or [])]


def upgrade(ctx: MigrationContext) -> None:
    target_dir = ctx.target_dir
    regenerated = 0
    skipped_no_resume = 0
    stamped_uncorrectable = 0
    touched_analysis_dirs: set[Path] = set()

    for pattern, out_subdir, log_name, summary_name in _FAMILIES:
        for perf_tseries_zarr in sorted(target_dir.glob(pattern)):
            sim_dir = perf_tseries_zarr.parent.parent  # <sim>/processed/<name>.zarr
            raw_perf_dir = sim_dir / out_subdir / "performance"
            resume_steps = _resume_steps_for(sim_dir, log_name)

            if not resume_steps:
                skipped_no_resume += 1
                continue

            if raw_perf_dir.exists() and any(raw_perf_dir.glob("performance*.txt")):
                if not ctx.dry_run:
                    _regenerate(sim_dir, raw_perf_dir, perf_tseries_zarr, summary_name, resume_steps)
                regenerated += 1
                touched_analysis_dirs.add(_analysis_dir_for(sim_dir, target_dir))
            else:
                # Resumed, but clear_raw removed performance/ (it is in
                # _CLEAR_RAW_DELETE_SUBDIRS). No source of truth to re-aggregate from, so
                # this sim is PERMANENTLY uncorrectable. This branch MUST NOT be folded
                # into the skip above: "never resumed" and "resumed but unrepairable" are
                # opposite states, and merging them is how a migration launders a
                # known-wrong value as a clean pass.
                if not ctx.dry_run:
                    _stamp_uncorrectable(target_dir, perf_tseries_zarr, resume_steps)
                stamped_uncorrectable += 1

    for analysis_dir in sorted(touched_analysis_dirs):
        if not ctx.dry_run:
            _invalidate_consolidation_signals(analysis_dir)

    _mode = "DRY RUN (no writes)" if ctx.dry_run else "applied"
    logger.info(
        "V0018 %s: regenerated %d resumed sim(s) across %d analysis dir(s); skipped %d "
        "never-resumed sim(s) (byte-unchanged by design); stamp-marked %d resumed sim(s) "
        "as UNCORRECTABLE.",
        _mode,
        regenerated,
        len(touched_analysis_dirs),
        skipped_no_resume,
        stamped_uncorrectable,
    )
    if stamped_uncorrectable:
        logger.warning(
            "V0018: %d resumed simulation(s) could NOT be corrected -- their raw "
            "performance{N}.txt inputs were cleared. These carry wrong values "
            "permanently and must be re-simulated if the numbers matter. Grep for %s.",
            stamped_uncorrectable,
            _STALE_MARKER_FILENAME,
        )


def _analysis_dir_for(sim_dir: Path, target_dir: Path) -> Path:
    """The analysis dir owning a sim: the sub-analysis root, else the target root.

    Path-only by construction -- it never instantiates TRITONSWMM_scenario, whose
    __init__ mkdir's directories (the summary_paths.py precedent).
    """
    for parent in sim_dir.parents:
        if parent.parent.name == "subanalyses":
            return parent
        if parent == target_dir:
            break
    return target_dir


def _regenerate(
    sim_dir: Path,
    raw_perf_dir: Path,
    perf_tseries_zarr: Path,
    summary_name: str,
    resume_steps: list[int],
) -> None:
    """Re-run the corrected aggregation against the surviving raw checkpoints.

    Delegates to the module-level helpers rather than reimplementing: they are the single
    source of truth for the algorithm. Local import avoids inverting the
    process_simulation -> version_migration dependency.
    """
    from hhemt.process_simulation import _aggregate_perf_summary, _aggregate_perf_tseries

    ds_tseries = _aggregate_perf_tseries(raw_perf_dir, resume_steps=resume_steps)
    ds_tseries.to_zarr(perf_tseries_zarr, mode="w", consolidated=False)

    ds_summary = _aggregate_perf_summary(raw_perf_dir, resume_steps=resume_steps)
    ds_summary.attrs["units"] = "seconds"
    ds_summary.attrs["notes"] = (
        "V0018-regenerated under ledger-driven resume-reset detection. Per-column "
        "slowest-rank cumulative cost, summed across the boundaries recorded in "
        f"resume_reporting_tsteps={resume_steps}. Total/Simulation/Init are CUMULATIVE "
        "across every allocation. TRITON-internal: excludes per-attempt process launch, "
        "checkpoint read, and post-last-checkpoint work that was discarded and recomputed "
        "-- the whole-process wall is wall_clock_ledger_s."
    )
    ds_summary.to_zarr(sim_dir / "processed" / summary_name, mode="w", consolidated=False)
    logger.info("V0018: regenerated %s (resume_steps=%s)", perf_tseries_zarr, resume_steps)


def _invalidate_consolidation_signals(analysis_dir: Path) -> None:
    """Make the next run REBUILD the consolidated tree, without deleting it here.

    Two independent gates, both of which must be cleared -- see the module docstring:
      1. ``_status/e_consolidate_*`` / ``f_consolidate_master*`` decide whether Snakemake
         re-plans the consolidate rule at all.
      2. ``log.json::datatree_consolidation_complete`` decides whether that rule rebuilds
         or prints "Not overwriting" and exits. The consolidate gate does its own
         ``fast_rmtree`` once this is falsy, so no artifact is deleted from here.
    """
    status_dir = analysis_dir / "_status"
    if status_dir.is_dir():
        for flag in sorted(status_dir.iterdir()):
            if flag.name.startswith(("e_consolidate_", "f_consolidate_master")):
                flag.unlink(missing_ok=True)  # EXEMPT-DU: status flag
                logger.info("V0018: cleared consolidate flag %s", flag)

    log_path = analysis_dir / "log.json"
    if not log_path.exists():
        return
    try:
        payload = json.loads(log_path.read_text())
    except (OSError, ValueError):
        logger.warning("V0018: could not read %s to invalidate consolidation", log_path)
        return
    if payload.get("datatree_consolidation_complete") is None:
        return  # already falsy / field absent on a legacy log -- nothing to clear
    payload["datatree_consolidation_complete"] = None
    log_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("V0018: cleared datatree_consolidation_complete in %s", log_path)


def _stamp_uncorrectable(target_dir: Path, perf_tseries_zarr: Path, resume_steps: list[int]) -> None:
    """Mark a resumed sim whose raw inputs are gone as permanently uncorrectable."""
    marker = perf_tseries_zarr.parent / _STALE_MARKER_FILENAME
    marker.write_text(
        json.dumps(
            {
                "layout_version_from": version_from,
                "layout_version_to": version_to,
                "perf_tseries_zarr": str(perf_tseries_zarr.relative_to(target_dir)),
                "resume_reporting_tsteps": resume_steps,
                "reason": (
                    "This simulation resumed from a hotstart, so its pre-V0018 performance "
                    "values are wrong (a missed reset boundary summed a large negative "
                    "delta as elapsed time). The raw performance{N}.txt inputs needed to "
                    "re-aggregate have been cleared, so the values CANNOT be corrected in "
                    "place. Re-simulate if these numbers are load-bearing."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.warning("V0018: UNCORRECTABLE %s (raw perf inputs cleared)", perf_tseries_zarr)
