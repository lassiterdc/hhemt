"""Phase 2 reprocess tests — analysis-level reprocess re-fires downstream only.

Plan: ``library/docs/planning/projects/TRITON-SWMM_toolkit/features/reprocess_downstream_stages/2 analysis level reprocess cli scoped snakefile generator.md``.

R6: ``c_run_*`` simulation completion flags must be untouched by a reprocess —
the reprocess driver never re-fires simulations.

R7: ``analysis_datatree.zarr`` must be regenerated (overwrite=True) when
reprocess starts at ``consolidate``.

Phase 3 (reprocess_orchestrator_liveness_gate) adds the gate-integration
scenarios (a)/(b)/(e): a reprocess PROCEEDS past ``_submitted/`` sim-worker
sentinels with no live ``_orchestrator/`` driver, REFUSES fast on a live
``_orchestrator/`` driver, and NEVER reaches the interactive ``input()``
prompt (``skip_lock_check=True`` bypass).
"""
from __future__ import annotations

import json
import subprocess

import pytest

from TRITON_SWMM_toolkit import orchestrator_sentinels as osent
from TRITON_SWMM_toolkit.workflow import _NON_INTERACTIVE_LOCK_CLEAR_ENV, WorkflowError


def _fake_ps_run(ps_alive):
    """``subprocess.run`` stub: ``ps -p {pid}`` exits 0 iff pid in ``ps_alive``.

    Mirrors the gate-unit harness in test_synth_09; the gate's local-mode arm
    probes driver liveness via ``ps -p {pid}``.
    """

    def _run(cmd, *a, **k):
        rc = 1
        if cmd[:2] == ["ps", "-p"] and int(cmd[2]) in ps_alive:
            rc = 0
        return subprocess.CompletedProcess(cmd, rc, b"", b"")

    return _run


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


@pytest.mark.usefixtures("tritonswmm_cpu_compiled")
def test_reprocess_proceeds_with_submitted_workers_no_orchestrator(synthetic_multisim_completed):
    """(a) R2: reprocess PROCEEDS when ``_submitted/`` sim-WORKER sentinels are
    present but no live ``_orchestrator/`` DRIVER sentinel exists.

    The liveness gate must distinguish queued/running sim workers (which a
    reprocess legitimately coexists with) from a live orchestration driver
    (which it must refuse). A ``_submitted/`` sentinel alone must not gate.
    """
    a = synthetic_multisim_completed
    analysis_dir = a.analysis_paths.analysis_dir
    submitted = analysis_dir / "_status" / "_submitted"
    submitted.mkdir(parents=True, exist_ok=True)
    worker = submitted / "run_tritonswmm_evt-gatecheck.json"
    worker.write_text(json.dumps({"slurm_jobid": "999999"}))
    try:
        result = a.reprocess(start_with="render", execution_mode="local", verbose=False)
        assert result.get("success"), (
            "Reprocess must proceed with _submitted/ workers present and no live "
            f"_orchestrator/ sentinel; got {result.get('message', '(no message)')!r}. "
            f"Snakemake log: {result.get('snakemake_logfile')}"
        )
    finally:
        worker.unlink(missing_ok=True)


@pytest.mark.usefixtures("tritonswmm_cpu_compiled")
def test_reprocess_refuses_fast_with_live_orchestrator(synthetic_multisim_completed, monkeypatch):
    """(b) R3: a live ``_orchestrator/`` DRIVER sentinel makes reprocess refuse
    fast with a ``WorkflowError`` — never ``input()``, never a snakemake subprocess.

    Tested at the ``submit_reprocess_workflow`` builder level: the gate fires
    before any downstream-flag invalidation, so this assertion does not mutate
    the session-scoped fixture's consolidated state.
    """
    a = synthetic_multisim_completed
    builder = a._workflow_builder
    analysis_dir = a.analysis_paths.analysis_dir
    osent.write_orchestrator_sentinel(analysis_dir, driver_id="live-driver", workflow_submission_mode="local", pid=4242)
    monkeypatch.setattr("subprocess.run", _fake_ps_run({4242}))
    try:
        with pytest.raises(WorkflowError) as excinfo:
            builder.submit_reprocess_workflow(start_with="render", execution_mode="local", dry_run=False, verbose=False)
        assert "live orchestration driver" in excinfo.value.stderr
    finally:
        osent.remove_orchestrator_sentinel(analysis_dir, "live-driver")


@pytest.mark.usefixtures("tritonswmm_cpu_compiled")
def test_reprocess_never_calls_input_even_with_stale_lock(synthetic_multisim_completed, monkeypatch):
    """(e) non-TTY no-hang: even with the non-interactive lock-clear env var
    UNSET and a stale ``.snakemake/locks/*.lock`` planted (the exact condition
    that drives the run path to ``input()``), the reprocess path's
    ``skip_lock_check=True`` returns before the prompt.

    Proven by making ``builtins.input`` raise: if the reprocess path ever
    reached the toolkit-side lock prompt, this test would fail loudly rather
    than hang.
    """
    a = synthetic_multisim_completed
    analysis_dir = a.analysis_paths.analysis_dir
    # Force the interactive branch reachable on the run path: unset the test
    # env var so _check_and_clear_snakemake_lock does NOT silently rmtree, and
    # plant a stale lock so a non-skipped check would prompt.
    monkeypatch.delenv(_NON_INTERACTIVE_LOCK_CLEAR_ENV, raising=False)
    locks_dir = analysis_dir / ".snakemake" / "locks"
    locks_dir.mkdir(parents=True, exist_ok=True)
    stale_lock = locks_dir / "gatecheck.lock"
    stale_lock.write_text("stale")

    def _boom(*a, **k):
        raise AssertionError("input() must never be reached on the reprocess path")

    monkeypatch.setattr("builtins.input", _boom)
    try:
        result = a.reprocess(start_with="render", execution_mode="local", verbose=False)
        assert result.get("success"), (
            "Reprocess must complete without prompting; got "
            f"{result.get('message', '(no message)')!r}. "
            f"Snakemake log: {result.get('snakemake_logfile')}"
        )
    finally:
        stale_lock.unlink(missing_ok=True)


def test_reprocess_dry_run_no_destructive_mutation(synthetic_multisim_completed):
    """R5/R6: analysis.reprocess(dry_run=True, start_with='consolidate') must NOT
    delete analysis_datatree.zarr nor re-stamp the analysis-scope _du.json. The
    completion flag MAY be deleted (it is the cheap mtime trigger)."""
    from TRITON_SWMM_toolkit.du_sentinels import compute_and_write_scope_sentinel

    a = synthetic_multisim_completed
    zarr = a.analysis_paths.analysis_datatree_zarr
    assert zarr is not None and zarr.exists(), "fixture precondition: consolidated zarr present"
    # Establish a known analysis-scope _du.json so the no-restamp assertion is
    # unconditional (R5): without this, an absent sentinel skips the mtime check.
    compute_and_write_scope_sentinel(a.analysis_paths.analysis_dir, scope="analysis")
    du = a.analysis_paths.analysis_dir / "_status" / "_du.json"
    assert du.exists(), "precondition: analysis-scope _du.json materialized"
    du_mtime0 = du.stat().st_mtime_ns

    result = a.reprocess(start_with="consolidate", execution_mode="local", dry_run=True, verbose=False)
    assert result.get("success"), f"dry-run reprocess failed: {result.get('message')!r}"

    assert zarr.exists(), "dry-run reprocess must NOT delete the consolidated zarr"
    assert du.stat().st_mtime_ns == du_mtime0, "dry-run reprocess must NOT re-stamp _du.json"
