"""Phase 2 reprocess tests — analysis-level reprocess re-fires downstream only.

Plan: ``library/docs/planning/projects/hhemt/features/reprocess_downstream_stages/
2 analysis level reprocess cli scoped snakefile generator.md``.

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

from hhemt import orchestrator_sentinels as osent
from hhemt.workflow import _NON_INTERACTIVE_LOCK_CLEAR_ENV, WorkflowError


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


def _zarr_mtime_target(zarr):
    """Return a mtime signal robust to zarr layout: prefer the ``.zgroup``
    sentinel (sensitive to overwrite); fall back to the zarr root."""
    zgroup_sentinel = zarr / ".zgroup"
    return zgroup_sentinel if zgroup_sentinel.exists() else zarr


@pytest.mark.usefixtures("tritonswmm_cpu_compiled")
def test_reprocess_consolidate_default_preserves_zarr(synthetic_multisim_completed_isolated):
    """Phase 2 FQ1 default (regenerate_existing=False): reprocess(consolidate)
    leaves sim flags untouched, PRESERVES the consolidated zarr (mtime
    unchanged — no rebuild, no DU restamp walk), and re-fires the report."""
    a = synthetic_multisim_completed_isolated
    status_dir = a.analysis_paths.analysis_dir / "_status"

    before_run_flags = sorted(status_dir.glob("c_run_*"))
    assert before_run_flags, (
        "Expected synthetic_multisim_completed_isolated fixture to have produced c_run_* flags; "
        "fixture state is incomplete."
    )
    dt = a.analysis_paths.analysis_datatree_zarr
    assert dt is not None and dt.exists(), f"Expected analysis_datatree_zarr to exist after fixture setup; got {dt!r}."
    mtime_target = _zarr_mtime_target(dt)
    mtime0 = mtime_target.stat().st_mtime

    # Default reprocess — must NOT delete/rebuild the zarr.
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

    # Phase 2 default: zarr PRESERVED — mtime unchanged (no rebuild).
    mtime1 = mtime_target.stat().st_mtime
    assert mtime1 == mtime0, (
        "Default reprocess(consolidate) must PRESERVE the consolidated zarr "
        f"(mtime unchanged). mtime0={mtime0!r}, mtime1={mtime1!r}, "
        f"target={mtime_target!r}."
    )
    # Report re-rendered against the preserved zarr.
    html = a.analysis_paths.analysis_dir / "analysis_report.html"
    zf = a.analysis_paths.analysis_dir / "analysis_report.zip"
    assert html.exists() or zf.exists(), (
        "Default reprocess(consolidate) must re-render the report. "
        f"Neither analysis_report.{{html,zip}} found at {a.analysis_paths.analysis_dir}."
    )


@pytest.mark.usefixtures("tritonswmm_cpu_compiled")
def test_reprocess_consolidate_regenerate_existing_rebuilds_zarr(synthetic_multisim_completed_isolated):
    """Phase 2 regenerate_existing=True: reprocess(consolidate) deletes and
    rebuilds the consolidated zarr (mtime advances); sim flags untouched."""
    a = synthetic_multisim_completed_isolated
    status_dir = a.analysis_paths.analysis_dir / "_status"

    before_run_flags = sorted(status_dir.glob("c_run_*"))
    dt = a.analysis_paths.analysis_datatree_zarr
    assert dt is not None and dt.exists(), f"fixture precondition: zarr present; got {dt!r}"
    mtime_target = _zarr_mtime_target(dt)
    mtime0 = mtime_target.stat().st_mtime

    result = a.reprocess(start_with="consolidate", execution_mode="local", regenerate_existing=True, verbose=False)
    assert result.get("success"), (
        f"reprocess(consolidate, regenerate_existing=True) failed: "
        f"{result.get('message','(no message)')}. Snakemake log: {result.get('snakemake_logfile')}"
    )

    after_run_flags = sorted(status_dir.glob("c_run_*"))
    assert before_run_flags == after_run_flags, "Reprocess must not modify the c_run_* flag set."

    mtime1 = mtime_target.stat().st_mtime
    assert mtime1 > mtime0, (
        "regenerate_existing=True reprocess(consolidate) must REBUILD the consolidated "
        f"zarr (mtime advance). mtime0={mtime0!r}, mtime1={mtime1!r}, target={mtime_target!r}."
    )


@pytest.mark.usefixtures("tritonswmm_cpu_compiled")
def test_consolidate_to_datatree_rebuilds_when_log_incomplete(synthetic_multisim_completed_isolated):
    """R4/D5: consolidate_to_datatree treats a present-but-log-incomplete zarr
    as corrupt and rebuilds it — it must NOT early-return on ``.exists()`` alone."""
    a = synthetic_multisim_completed_isolated
    zarr = a.analysis_paths.analysis_datatree_zarr
    assert zarr is not None and zarr.exists(), "fixture precondition: consolidated zarr present"

    # Force the canonical completion signal to False while the zarr stays on disk.
    a._refresh_log()
    a.log.datatree_consolidation_complete.set(False)
    mtime_target = _zarr_mtime_target(zarr)
    mtime0 = mtime_target.stat().st_mtime

    # .exists()=True but log-complete=False → the early-return must NOT fire;
    # the zarr is treated as corrupt and rebuilt.
    a.process.consolidate_to_datatree()

    assert zarr.exists(), "rebuild must leave a valid zarr on disk"
    mtime1 = mtime_target.stat().st_mtime
    assert mtime1 > mtime0, (
        "consolidate_to_datatree must REBUILD a present-but-log-incomplete zarr "
        f"(mtime advance). mtime0={mtime0!r}, mtime1={mtime1!r}."
    )
    # Rebuild re-stamps the completion signal.
    a._refresh_log()
    assert (
        a.log.datatree_consolidation_complete.get() is True
    ), "rebuild must re-set datatree_consolidation_complete=True"


@pytest.mark.usefixtures("tritonswmm_cpu_compiled")
def test_reprocess_render_only(synthetic_multisim_completed_isolated):
    """Reprocess(start_with='render') re-renders the report against existing
    plots without re-firing consolidate or sim rules."""
    a = synthetic_multisim_completed_isolated
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
        "Expected reprocess(render) to materialize analysis_report.{html,zip}. " f"Neither found at {analysis_dir}."
    )


@pytest.mark.usefixtures("tritonswmm_cpu_compiled")
def test_reprocess_proceeds_with_submitted_workers_no_orchestrator(synthetic_multisim_completed_isolated):
    """(a) R2: reprocess PROCEEDS when ``_submitted/`` sim-WORKER sentinels are
    present but no live ``_orchestrator/`` DRIVER sentinel exists.

    The liveness gate must distinguish queued/running sim workers (which a
    reprocess legitimately coexists with) from a live-or-indeterminate orchestration
    driver (which it must refuse). A ``_submitted/`` sentinel alone must not gate.
    """
    a = synthetic_multisim_completed_isolated
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
def test_reprocess_refuses_fast_with_live_orchestrator(synthetic_multisim_completed_isolated, monkeypatch):
    """(b) R3: a live ``_orchestrator/`` DRIVER sentinel makes reprocess refuse
    fast with a ``WorkflowError`` — never ``input()``, never a snakemake subprocess.

    Tested at the ``submit_reprocess_workflow`` builder level: the gate fires
    before any downstream-flag invalidation, so this assertion does not mutate
    the session-scoped fixture's consolidated state.
    """
    a = synthetic_multisim_completed_isolated
    builder = a._workflow_builder
    analysis_dir = a.analysis_paths.analysis_dir
    osent.write_orchestrator_sentinel(analysis_dir, driver_id="live-driver", workflow_submission_mode="local", pid=4242)
    monkeypatch.setattr("subprocess.run", _fake_ps_run({4242}))
    try:
        with pytest.raises(WorkflowError) as excinfo:
            builder.submit_reprocess_workflow(start_with="render", execution_mode="local", dry_run=False, verbose=False)
        # Assert the LIVENESS CLASSIFICATION, not the prose — see the twin
        # assertion in test_synth_08_sensitivity_reprocess.py. The tri-state gate
        # (Gotcha 72) emits the same generic "live-or-indeterminate" prefix for
        # both ALIVE and UNKNOWN/held, so only the `liveness=ALIVE` tag actually
        # verifies R3's "refuses fast with a LIVE driver" contract.
        assert "liveness=ALIVE" in excinfo.value.stderr
    finally:
        osent.remove_orchestrator_sentinel(analysis_dir, "live-driver")


@pytest.mark.usefixtures("tritonswmm_cpu_compiled")
def test_reprocess_refusal_does_not_destroy_the_report(synthetic_multisim_completed_isolated, monkeypatch):
    """A REFUSED reprocess must not have destroyed the report it refused to rebuild.

    The FACADE-level complement of test_reprocess_refuses_fast_with_live_orchestrator,
    which asserts the same refusal one layer down at the builder. That scoping is not
    incidental: until the claim was hoisted, `reprocess()` ran
    `_invalidate_downstream_flags` -- which unlinks analysis_report.{html,zip} on every
    arm -- BEFORE `submit_reprocess_workflow` reached the orchestrator-liveness gate, so
    the gate could only refuse after the artifact was gone. Measured 2026-08-16 on the
    UVA benchmarking_cpu_uva arm: report deleted, then rc=1, then zero analysis_report.*
    anywhere under the scratch base.

    This test therefore asserts an ORDERING, which is the one property a reachability
    claim cannot certify. It FAILS against the pre-hoist code (the refusal arrives, the
    report does not survive) and passes once the claim precedes the destruction.
    """
    a = synthetic_multisim_completed_isolated
    analysis_dir = a.analysis_paths.analysis_dir

    report_html = analysis_dir / "analysis_report.html"
    report_html.write_text("<html><body>pre-existing report</body></html>")
    plots_dir = analysis_dir / "plots"
    plots_before = sorted(p.name for p in plots_dir.rglob("*") if p.is_file()) if plots_dir.exists() else []

    osent.write_orchestrator_sentinel(
        analysis_dir, driver_id="live-driver-ordering", workflow_submission_mode="local", pid=4242
    )
    monkeypatch.setattr("subprocess.run", _fake_ps_run({4242}))
    try:
        with pytest.raises(WorkflowError) as excinfo:
            a.reprocess(start_with="render", execution_mode="local", verbose=False)
        assert "liveness=ALIVE" in excinfo.value.stderr, (
            "precondition: the refusal must come from the LIVE arm of the tri-state gate, "
            "not from an UNKNOWN/held classification -- otherwise this test would pass for "
            f"the wrong reason. Got: {excinfo.value.stderr!r}"
        )
        assert report_html.exists(), (
            "A refusing gate destroyed the artifact it refused to rebuild. "
            "analysis_report.html was present before reprocess() and is gone after a "
            "refusal, which means destructive invalidation still precedes the "
            "orchestrator-liveness claim in the reprocess facade."
        )
        plots_after = sorted(p.name for p in plots_dir.rglob("*") if p.is_file()) if plots_dir.exists() else []
        assert plots_after == plots_before, (
            "A refused reprocess must leave plots/ untouched. (The render arm never "
            "deletes plots even on success -- it is the surgical report-shell-only path -- "
            "so any change here means a non-render arm ran, or the refusal came too late.)"
        )
    finally:
        osent.remove_orchestrator_sentinel(analysis_dir, "live-driver-ordering")
        report_html.unlink(missing_ok=True)


@pytest.mark.usefixtures("tritonswmm_cpu_compiled")
def test_reprocess_never_calls_input_even_with_stale_lock(synthetic_multisim_completed_isolated, monkeypatch):
    """(e) non-TTY no-hang: even with the non-interactive lock-clear env var
    UNSET and a stale ``.snakemake/locks/*.lock`` planted (the exact condition
    that drives the run path to ``input()``), the reprocess path's
    ``skip_lock_check=True`` returns before the prompt.

    Proven by making ``builtins.input`` raise: if the reprocess path ever
    reached the toolkit-side lock prompt, this test would fail loudly rather
    than hang.
    """
    a = synthetic_multisim_completed_isolated
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


@pytest.mark.usefixtures("tritonswmm_cpu_compiled")
def test_dry_run_submit_workflow_leaves_no_orchestrator_sentinel(synthetic_multisim_completed_isolated, monkeypatch):
    """TWO-ARM DIFFERENTIAL: dry_run leaves NO orchestrator sentinel; a real
    detached run leaves exactly one.

    A dry run submits nothing, allocates nothing and writes no zarr, so it is not the
    "unarbitrated concurrent consolidate-zarr write" the sentinel exists to advertise.
    Writing one was actively harmful on a detached mode: the enrich has no job_id and no
    session_name to merge, so the record keeps null identity fields, and the tri-state
    gate reads that as UNKNOWN-held until the mtime-age fail-safe at
    hpc_total_job_duration_min + 30 -- blocking reprocess for the better part of a
    long-walltime arm on a rehearsal that never ran.

    The SECOND arm is what makes the first meaningful. Suppressing the sentinel
    unconditionally would also make arm 1 pass, while silently disarming the
    concurrency gate for every real detached driver. Arm 1 FAILS pre-fix; arm 2 passes
    both before and after, and is the guard against over-correcting.

    The builder is stubbed: the property under test is the sentinel lifecycle inside
    submit_workflow, so removing the Snakemake subprocess removes no code under test.
    """
    # The OE-A node-local preflight refuses a slurm-locus submit whose configs sit
    # under the system temp dir. This fixture is a tmp_path clone and the `batch_job`
    # set below is a PRETEND detached mode, which is exactly the false-positive shape
    # the acknowledgement exists for -- the sentinel lifecycle under test has nothing
    # to do with cross-node path visibility.
    monkeypatch.setenv("HHEMT_ALLOW_NODE_LOCAL_CONFIGS", "1")
    a = synthetic_multisim_completed_isolated
    analysis_dir = a.analysis_paths.analysis_dir
    orch_dir = osent.orchestrator_dir(analysis_dir)

    # Detached mode is required: under "local" the finally-removal fires on BOTH arms,
    # so the defect is invisible and arm 1 would pass pre-fix for the wrong reason.
    monkeypatch.setattr(a.cfg_analysis, "multi_sim_run_method", "batch_job")
    monkeypatch.setattr(
        a._workflow_builder,
        "submit_workflow",
        lambda **kwargs: {"success": True, "mode": "tmux", "job_id": None, "session_name": None},
    )

    def _sentinel_names():
        return sorted(p.name for p in orch_dir.glob("*.json")) if orch_dir.exists() else []

    for stale in _sentinel_names():
        (orch_dir / stale).unlink(missing_ok=True)

    try:
        # ARM 1 -- rehearsal. No driver exists, so no record of one may persist.
        a.submit_workflow(dry_run=True, verbose=False)
        assert _sentinel_names() == [], (
            "A dry run left an orchestrator sentinel. On a detached mode nothing removes "
            "it and its identity fields are null, so the tri-state gate holds it as "
            "UNKNOWN for hpc_total_job_duration_min + 30 minutes -- a reprocess blocked "
            f"by a driver that never existed. Found: {_sentinel_names()}"
        )

        # ARM 2 -- real detached run. The durable record MUST still be written, or the
        # fix has disarmed the concurrency gate instead of correcting it.
        a.submit_workflow(dry_run=False, verbose=False)
        assert len(_sentinel_names()) == 1, (
            "A real batch_job submit must leave exactly one durable orchestrator "
            "sentinel for the gate's liveness probes to reclaim. Suppressing it would "
            f"disarm the reprocess concurrency gate. Found: {_sentinel_names()}"
        )
    finally:
        for name in _sentinel_names():
            (orch_dir / name).unlink(missing_ok=True)


def test_reprocess_dry_run_no_destructive_mutation(synthetic_multisim_completed_isolated):
    """R5/R6: analysis.reprocess(dry_run=True, start_with='consolidate') must NOT
    delete analysis_datatree.zarr nor re-stamp the analysis-scope _du.json. The
    completion flag MAY be deleted (it is the cheap mtime trigger)."""
    from hhemt.du_sentinels import compute_and_write_scope_sentinel

    a = synthetic_multisim_completed_isolated
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
