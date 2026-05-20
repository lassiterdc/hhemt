"""Phase 2 reprocess tests — analysis-level reprocess re-fires downstream only.

Plan: ``library/docs/planning/projects/TRITON-SWMM_toolkit/features/reprocess_downstream_stages/2 analysis level reprocess cli scoped snakefile generator.md``.

R6: ``c_run_*`` simulation completion flags must be untouched by a reprocess —
the reprocess driver never re-fires simulations.

R7: ``analysis_datatree.zarr`` must be regenerated (overwrite=True) when
reprocess starts at ``consolidate``.
"""
from __future__ import annotations

import pytest


@pytest.mark.usefixtures("tritonswmm_cpu_compiled")
def test_reprocess_consolidate_refires_datatree_not_sims(synthetic_multisim_completed):
    """R6 + R7: reprocess(start_with='consolidate') leaves sim flags untouched
    and regenerates the analysis datatree zarr."""
    a = synthetic_multisim_completed
    status_dir = a.analysis_paths.analysis_dir / "_status"

    # Capture pre-reprocess state: sim flag file set and a datatree mtime
    # signal. The zarr is a directory-shaped store, so we read mtime from the
    # top-level .zgroup / .zarray sentinel file when present (more sensitive
    # to overwrite than the directory's own mtime, which only changes on
    # entry add/remove).
    before_run_flags = sorted(status_dir.glob("c_run_*"))
    assert before_run_flags, (
        "Expected synthetic_multisim_completed fixture to have produced c_run_* flags; "
        "fixture state is incomplete."
    )
    dt = a.analysis_paths.analysis_datatree_zarr
    assert dt is not None and dt.exists(), (
        "Expected analysis_datatree_zarr to exist after fixture setup; got "
        f"{dt!r}."
    )
    # Capture a mtime signal robust to zarr layout: prefer .zgroup; fall
    # back to the zarr root's own mtime if the sentinel layout differs.
    zgroup_sentinel = dt / ".zgroup"
    mtime_target = zgroup_sentinel if zgroup_sentinel.exists() else dt
    mtime0 = mtime_target.stat().st_mtime

    # Re-fire consolidate + downstream.
    result = a.reprocess(start_with="consolidate", execution_mode="local", verbose=False)
    assert result.get("success"), (
        f"reprocess(consolidate) failed: {result.get('message','(no message)')}. "
        f"Snakemake log: {result.get('snakemake_logfile')}"
    )

    # R6: sim flags untouched.
    after_run_flags = sorted(status_dir.glob("c_run_*"))
    assert before_run_flags == after_run_flags, (
        "Reprocess must not modify the c_run_* simulation flag set. "
        f"Before: {[p.name for p in before_run_flags]!r}; "
        f"after: {[p.name for p in after_run_flags]!r}."
    )

    # R7: datatree mtime advanced (overwrite re-wrote the zarr).
    mtime1 = mtime_target.stat().st_mtime
    assert mtime1 > mtime0, (
        "Reprocess(start_with='consolidate') must regenerate the analysis "
        f"datatree zarr. mtime0={mtime0!r}, mtime1={mtime1!r}, "
        f"target={mtime_target!r}."
    )


@pytest.mark.usefixtures("tritonswmm_cpu_compiled")
def test_reprocess_render_only(synthetic_multisim_completed):
    """Reprocess(start_with='render') re-renders the report against existing
    plots without re-firing consolidate or sim rules."""
    a = synthetic_multisim_completed
    analysis_dir = a.analysis_paths.analysis_dir

    result = a.reprocess(start_with="render", execution_mode="local", verbose=False)
    assert result.get("success"), (
        f"reprocess(render) failed: {result.get('message','(no message)')}. "
        f"Snakemake log: {result.get('snakemake_logfile')}"
    )

    # At least one report artifact present.
    html = analysis_dir / "analysis_report.html"
    zipfile = analysis_dir / "analysis_report.zip"
    assert html.exists() or zipfile.exists(), (
        "Expected reprocess(render) to materialize analysis_report.{html,zip}. "
        f"Neither found at {analysis_dir}."
    )
