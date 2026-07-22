"""Phase 3: sensitivity master reprocess refreshes per-sa + master datatree.

The Phase 3 reprocess contract (R12 in the master plan): a sensitivity
master analysis at the post-master-consolidate state can be reprocessed
with ``sensitivity.reprocess(start_with="consolidate", execution_mode="local")``
to regenerate ``sensitivity_datatree.zarr`` (and the master consolidate
flag) without re-running any simulation rule.

The session-scoped ``synthetic_sensitivity_completed`` fixture
(``tests/conftest.py``) runs the synth sensitivity master once per pytest
session to the ``f_consolidate_master_complete.flag`` state; these tests
consume the per-test ``synthetic_sensitivity_completed_isolated`` copy-on-read
wrapper (D1) so the reprocess mutation lands on a ``tmp_path`` clone, then
assert the master datatree zarr's mtime advances.

Phase 3 (reprocess_orchestrator_liveness_gate) adds the sensitivity
analogues of gate-integration scenarios (a)/(b)/(e) against the master
``_status/_orchestrator/`` authority dir: reprocess PROCEEDS past
``_submitted/`` sim-worker sentinels with no live driver, REFUSES fast on a
live ``_orchestrator/`` driver, and NEVER reaches the interactive
``input()`` prompt.
"""

import json
import subprocess

import pytest

from hhemt import orchestrator_sentinels as osent
from hhemt.workflow import _NON_INTERACTIVE_LOCK_CLEAR_ENV, WorkflowError

pytestmark = [pytest.mark.requires_snakemake_subprocess]


def _fake_ps_run(ps_alive):
    """``subprocess.run`` stub: ``ps -p {pid}`` exits 0 iff pid in ``ps_alive``."""

    def _run(cmd, *a, **k):
        rc = 1
        if cmd[:2] == ["ps", "-p"] and int(cmd[2]) in ps_alive:
            rc = 0
        return subprocess.CompletedProcess(cmd, rc, b"", b"")

    return _run


def _zarr_mtime_target(zarr):
    """mtime signal robust to zarr layout: prefer ``.zgroup``; fall back to root."""
    zgroup = zarr / ".zgroup"
    return zgroup if zgroup.exists() else zarr


def test_sensitivity_reprocess_consolidate_default_preserves_zarr(synthetic_sensitivity_completed_isolated):
    """Phase 2 FQ1 default (regenerate_existing=False): sensitivity master
    reprocess(consolidate) PRESERVES the master datatree zarr (mtime unchanged —
    no rebuild, no DU restamp) and re-fires the report."""
    sa = synthetic_sensitivity_completed_isolated
    mdt = sa.master_analysis.analysis_paths.sensitivity_datatree_zarr
    assert mdt.exists(), "fixture should have materialized sensitivity_datatree.zarr"
    mtime_target = _zarr_mtime_target(mdt)
    mtime0 = mtime_target.stat().st_mtime
    result = sa.reprocess(start_with="consolidate", execution_mode="local")
    assert result["success"], f"reprocess failed: {result.get('message')!r}"
    assert mtime_target.stat().st_mtime == mtime0, (
        "Default sensitivity reprocess(consolidate) must PRESERVE the master "
        f"datatree zarr (mtime unchanged). target={mtime_target!r}."
    )
    # Report re-rendered against the preserved zarr.
    master_dir = sa.master_analysis.analysis_paths.analysis_dir
    html = master_dir / "analysis_report.html"
    zf = master_dir / "analysis_report.zip"
    assert html.exists() or zf.exists(), (
        f"Default sensitivity reprocess must re-render the master report; none found at {master_dir}."
    )


def test_sensitivity_reprocess_consolidate_regenerate_existing_rebuilds_zarr(synthetic_sensitivity_completed_isolated):
    """Phase 2 regenerate_existing=True: sensitivity master reprocess(consolidate)
    deletes and rebuilds the master datatree zarr (mtime advances)."""
    sa = synthetic_sensitivity_completed_isolated
    mdt = sa.master_analysis.analysis_paths.sensitivity_datatree_zarr
    assert mdt.exists(), "fixture precondition: master zarr present"
    mtime_target = _zarr_mtime_target(mdt)
    mtime0 = mtime_target.stat().st_mtime
    result = sa.reprocess(start_with="consolidate", execution_mode="local", regenerate_existing=True)
    assert result["success"], f"reprocess(regenerate_existing=True) failed: {result.get('message')!r}"
    assert mtime_target.stat().st_mtime > mtime0, (
        "regenerate_existing=True sensitivity reprocess(consolidate) must REBUILD the "
        f"master datatree zarr (mtime advance). target={mtime_target!r}."
    )


def test_sensitivity_reprocess_consolidate_subset_sa_ids(synthetic_sensitivity_completed_isolated):
    """sensitivity.reprocess(sa_ids=[...]) restricts per-sa invalidation to the subset.

    The invalidation step only deletes the named per-sa consolidate flags; the
    master consolidate flag is invalidated regardless. After reprocess, both the
    subset's per-sa flags and the master flag must be re-created (success exit
    proves Snakemake completed the consolidate + master_consolidation chain).
    """
    sa = synthetic_sensitivity_completed_isolated
    status_dir = sa.master_analysis.analysis_paths.analysis_dir / "_status"
    all_sa_ids = [str(sid) for sid in sa.sub_analyses.keys()]
    # Subset: first sub-analysis only.
    subset = all_sa_ids[:1]
    result = sa.reprocess(
        start_with="consolidate",
        sa_ids=subset,
        execution_mode="local",
    )
    assert result["success"], f"reprocess(subset) failed: {result.get('message')!r}"
    # Master flag re-created after reprocess.
    assert (status_dir / "f_consolidate_master_complete.flag").exists()
    # Every per-sa flag exists post-reprocess (the subset ones re-created, the
    # others were never invalidated).
    for sid in all_sa_ids:
        assert (status_dir / f"e_consolidate_sa-{sid}_complete.flag").exists()


def test_sensitivity_reprocess_proceeds_with_submitted_workers_no_orchestrator(
    synthetic_sensitivity_completed_isolated,
):
    """(a) R2: sensitivity reprocess PROCEEDS when ``_submitted/`` sim-WORKER
    sentinels are present in the master dir but no live ``_orchestrator/``
    DRIVER sentinel exists."""
    sa = synthetic_sensitivity_completed_isolated
    master_dir = sa.master_analysis.analysis_paths.analysis_dir
    submitted = master_dir / "_status" / "_submitted"
    submitted.mkdir(parents=True, exist_ok=True)
    worker = submitted / "run_tritonswmm_evt-gatecheck.json"
    worker.write_text(json.dumps({"slurm_jobid": "999999"}))
    try:
        result = sa.reprocess(start_with="consolidate", execution_mode="local", verbose=False)
        assert result.get("success"), (
            "Sensitivity reprocess must proceed with _submitted/ workers present and no "
            f"live _orchestrator/ sentinel; got {result.get('message', '(no message)')!r}. "
            f"Snakemake log: {result.get('snakemake_logfile')}"
        )
    finally:
        worker.unlink(missing_ok=True)


def test_sensitivity_reprocess_refuses_fast_with_live_orchestrator(
    synthetic_sensitivity_completed_isolated, monkeypatch
):
    """(b) R3: a live master ``_orchestrator/`` DRIVER sentinel makes the
    sensitivity reprocess refuse fast with a ``WorkflowError``.

    Tested at the ``submit_reprocess_workflow`` builder level so no per-sa or
    master flags are invalidated (the facade's pre-invalidation is irrelevant —
    the gate fires first in submit).
    """
    sa = synthetic_sensitivity_completed_isolated
    builder = sa._workflow_builder
    master_dir = sa.master_analysis.analysis_paths.analysis_dir
    osent.write_orchestrator_sentinel(master_dir, driver_id="live-driver", workflow_submission_mode="local", pid=4242)
    monkeypatch.setattr("subprocess.run", _fake_ps_run({4242}))
    try:
        with pytest.raises(WorkflowError) as excinfo:
            builder.submit_reprocess_workflow(
                start_with="consolidate", execution_mode="local", dry_run=False, verbose=False
            )
        # Assert the LIVENESS CLASSIFICATION, not the prose. The tri-state gate
        # (Gotcha 72) emits "live-or-indeterminate orchestration driver" as a
        # generic prefix for BOTH liveness=ALIVE and liveness=UNKNOWN/held, so
        # asserting that prefix would pass even if a definitively-live driver were
        # misclassified as indeterminate — deleting the discrimination this test
        # exists to provide. `liveness=ALIVE` is the semantic tag and is what R3
        # actually requires here (verified 2026-07-21: the emitted entry is
        # `live-driver (mode=local, host=laptop, liveness=ALIVE)`).
        assert "liveness=ALIVE" in excinfo.value.stderr
    finally:
        osent.remove_orchestrator_sentinel(master_dir, "live-driver")


def test_sensitivity_reprocess_never_calls_input_even_with_stale_lock(
    synthetic_sensitivity_completed_isolated, monkeypatch
):
    """(e) non-TTY no-hang: even with the non-interactive lock-clear env var
    UNSET and a stale ``.snakemake/locks/*.lock`` planted on the master dir,
    the sensitivity reprocess path's ``skip_lock_check=True`` returns before
    the interactive prompt (``builtins.input`` raises if reached)."""
    sa = synthetic_sensitivity_completed_isolated
    master_dir = sa.master_analysis.analysis_paths.analysis_dir
    monkeypatch.delenv(_NON_INTERACTIVE_LOCK_CLEAR_ENV, raising=False)
    locks_dir = master_dir / ".snakemake" / "locks"
    locks_dir.mkdir(parents=True, exist_ok=True)
    stale_lock = locks_dir / "gatecheck.lock"
    stale_lock.write_text("stale")

    def _boom(*a, **k):
        raise AssertionError("input() must never be reached on the reprocess path")

    monkeypatch.setattr("builtins.input", _boom)
    try:
        result = sa.reprocess(start_with="consolidate", execution_mode="local", verbose=False)
        assert result.get("success"), (
            "Sensitivity reprocess must complete without prompting; got "
            f"{result.get('message', '(no message)')!r}. "
            f"Snakemake log: {result.get('snakemake_logfile')}"
        )
    finally:
        stale_lock.unlink(missing_ok=True)


def test_sensitivity_reprocess_dry_run_no_destructive_mutation(synthetic_sensitivity_completed_isolated):
    """R3/R4: sensitivity.reprocess(dry_run=True, start_with='consolidate') must NOT
    delete the master or any sub-analysis datatree zarr, nor re-stamp _du.json sentinels."""
    from hhemt.du_sentinels import compute_and_write_scope_sentinel

    sa = synthetic_sensitivity_completed_isolated
    master_zarr = sa.analysis_paths.sensitivity_datatree_zarr
    assert master_zarr is not None and master_zarr.exists(), "fixture precondition: master zarr present"
    sub_zarrs = [
        s.analysis_paths.analysis_datatree_zarr
        for s in sa.sub_analyses.values()
        if s.analysis_paths.analysis_datatree_zarr is not None
        and s.analysis_paths.analysis_datatree_zarr.exists()
    ]
    assert sub_zarrs, "fixture precondition: at least one sub-analysis zarr present"
    # Establish a known master-scope _du.json so the no-restamp assertion is
    # unconditional (R4): without this, an absent sentinel skips the mtime check.
    master_analysis_dir = sa.master_analysis.analysis_paths.analysis_dir
    compute_and_write_scope_sentinel(master_analysis_dir, scope="analysis")
    master_du = master_analysis_dir / "_status" / "_du.json"
    assert master_du.exists(), "precondition: master-scope _du.json materialized"
    master_du_mtime0 = master_du.stat().st_mtime_ns

    result = sa.reprocess(start_with="consolidate", execution_mode="local", dry_run=True, verbose=False)
    assert result.get("success"), f"dry-run sensitivity reprocess failed: {result.get('message')!r}"

    assert master_zarr.exists(), "dry-run must NOT delete the master sensitivity_datatree.zarr"
    for z in sub_zarrs:
        assert z.exists(), f"dry-run must NOT delete sub-analysis zarr {z}"
    assert master_du.stat().st_mtime_ns == master_du_mtime0, "dry-run must NOT re-stamp master _du.json"


def test_reprocess_rebuild_rewrites_summary(synthetic_sensitivity_completed_isolated):
    """FIX 1 end-to-end: sensitivity ``reprocess(start_with='process',
    regenerate_existing=True)`` re-fires the per-sa ``process_*`` rebuild rules
    AND clears the per-model processing log so the runner actually re-writes the
    per-scenario summaries — then the consolidate + master_consolidation chain
    rebuilds ``sensitivity_datatree.zarr`` against the freshly-written summaries
    (no ``FileNotFoundError``).

    This is the compile-gated regression for the silent-failure mode FIX 1
    closes: before the fix, ``start_with='process'`` deleted the per-scenario
    ``processed/`` dir but never cleared ``processing_log.outputs``, so the
    rebuilt process rule's runner skipped every ``_export_*`` write
    (``_already_written`` gate, Gotcha #28) and master consolidation then failed
    to find the per-sa summary it expected. Asserts the master datatree zarr's
    mtime advances and ``result["success"]`` is True.
    """
    sa = synthetic_sensitivity_completed_isolated
    mdt = sa.master_analysis.analysis_paths.sensitivity_datatree_zarr
    assert mdt.exists(), "fixture precondition: master sensitivity_datatree.zarr present"
    mtime_target = _zarr_mtime_target(mdt)
    mtime0 = mtime_target.stat().st_mtime

    result = sa.reprocess(
        start_with="process",
        regenerate_existing=True,
        execution_mode="local",
    )

    assert result["success"], (
        "reprocess(start_with='process', regenerate_existing=True) must succeed "
        f"(consolidate rebuilt against freshly-written summaries); got "
        f"{result.get('message')!r}. Snakemake log: {result.get('snakemake_logfile')}"
    )
    assert mtime_target.stat().st_mtime > mtime0, (
        "process-stage reprocess(regenerate_existing=True) must REBUILD the master "
        f"datatree zarr (mtime advance). target={mtime_target!r}."
    )


def test_unlink_dprocess_flags_for_regenerate_clears_only_matching_flags(tmp_path):
    """R2/R5 fast guard (no compile, no fixture mutation): the extracted FIX-2b
    free function unlinks every per-sa d_process_* flag for the named targets and
    leaves non-target / non-d_process flags intact. This is the seconds-scale
    tripwire for the d5d0084 re-removal class — it exercises EXACTLY the loop that
    commit dropped, with zero coupling to reprocess()'s destructive body (D1
    Option A)."""
    from hhemt.sensitivity_analysis import (
        _unlink_dprocess_flags_for_regenerate,
    )

    status_dir = tmp_path / "_status"
    status_dir.mkdir()
    # Two target sa_ids with d_process flags, plus decoys that MUST survive.
    (status_dir / "d_process_tritonswmm_sa-0_evt-a_complete.flag").touch()
    (status_dir / "d_process_tritonswmm_sa-1_evt-b_complete.flag").touch()
    (status_dir / "d_process_tritonswmm_sa-2_evt-c_complete.flag").touch()  # not a target
    (status_dir / "c_run_tritonswmm_sa-0_evt-a_complete.flag").touch()  # not d_process
    (status_dir / "e_consolidate_sa-0_complete.flag").touch()  # not d_process

    _unlink_dprocess_flags_for_regenerate(["0", "1"], status_dir)

    survivors = sorted(p.name for p in status_dir.glob("*.flag"))
    assert survivors == [
        "c_run_tritonswmm_sa-0_evt-a_complete.flag",
        "d_process_tritonswmm_sa-2_evt-c_complete.flag",
        "e_consolidate_sa-0_complete.flag",
    ], f"free function must unlink only target d_process flags; survivors={survivors}"


@pytest.fixture
def synth_partial_state_analysis(synthetic_sensitivity_completed_isolated):
    """A completed synth sensitivity analysis with ONE sub-analysis induced into
    the summary-absent partial state (its d_process/c_run flags left intact),
    for conditional-process-emit regression coverage (R5)."""
    from tests.fixtures.test_case_builder import induce_incomplete_subanalysis

    sa = synthetic_sensitivity_completed_isolated
    target_sa_id = sorted(sa.sub_analyses)[0]
    induce_incomplete_subanalysis(sa, target_sa_id, delete_master_tree=True)
    return sa, target_sa_id


@pytest.mark.requires_snakemake_subprocess
def test_reprocess_conditional_emit_over_partial_state(synth_partial_state_analysis):
    """R5 (local-Snakemake run, NOT a fast unit test — launches a real reprocess
    subprocess; the genuinely-fast R5 guard is the monkeypatch-based unlink test
    above): a process-stage reprocess over a partial-state sensitivity analysis
    rebuilds the induced-incomplete sub and succeeds (the conditional process-emit
    path fires only for the incomplete sub; complete subs are untouched)."""
    sa, target_sa_id = synth_partial_state_analysis
    result = sa.reprocess(start_with="process", regenerate_existing=True, execution_mode="local")
    assert result["success"], (
        f"conditional-emit reprocess over partial state must succeed; got {result.get('message')!r}"
    )
    mdt = sa.master_analysis.analysis_paths.sensitivity_datatree_zarr
    assert mdt.exists(), "master sensitivity_datatree.zarr must be rebuilt after partial-state reprocess"


# ---------------------------------------------------------------------------
# Edit 3 (2026-07-20) — scoped-delete failure must surface, and an explicit
# execution_mode="local" must not be overridden by multi_sim_run_method.
#
# Production failure this encodes: reprocess(regenerate_existing=True) routed the
# scoped delete to SLURM, the delete workflow failed 112 times and raised, the
# call site DISCARDED the returned dict, and reprocess fell through to consolidate
# against trees it believed were deleted. The result was a silent no-op that took
# a full debugging cycle to attribute. regenerate_existing invalidates by DELETION,
# so a failed deletion makes the flag a lie.
# ---------------------------------------------------------------------------


def _force_hpc(sa):
    """Make master AND every sub-analysis config a COHERENT HPC run.

    Forcing the MASTER alone is not enough, and the reason is a MIXED-OBJECT
    read worth recording. ``workflow.py::_build_resource_block`` (:1525) raises
    ``ValueError("hpc partition must be set when generating SLURM resources")``
    when ``partition is None and self.cfg_analysis.multi_sim_run_method !=
    "local"`` (:1572). At the per-sub reprocess emission site
    (``workflow.py:8062``) those two reads resolve against DIFFERENT objects:

    * the guard PREDICATE reads ``self.cfg_analysis`` on the base builder,
      which ``SnakemakeWorkflowBuilder.__init__`` ALIASES to
      ``master_analysis.cfg_analysis`` (:767) — so forcing the master DOES
      reach it;
    * the partition VALUE is ``sub_analysis.cfg_analysis.
      hpc_setup_and_analysis_processing_partition`` — a DISTINCT per-sub model
      built by ``model_validate({**master.model_dump(), **overlay})``, which a
      master mutation never touches (probed: all 4 subs report
      ``proc_part=None, run_method='local'`` after forcing the master alone).

    Forcing the master therefore ARMS the guard while leaving the value it
    demands unset, and every SLURM-bound per-sub rule raises before the test's
    own assertion. Both tiers must be forced.

    If a THIRD config object ever appears on this path (a per-target
    synthesized system YAML, or a persisted per-sub ``sa_{id}.yaml`` re-read
    in-process), do NOT add a third tier here — rebuild the fixture from a
    config that already carries these fields instead.
    """
    forced = {
        "multi_sim_run_method": "batch_job",
        "hpc_ensemble_partition": "standard",
        "hpc_setup_and_analysis_processing_partition": "standard",
    }
    cfgs = [sa.master_analysis.cfg_analysis]
    cfgs += [sub.cfg_analysis for sub in sa.sub_analyses.values()]
    for cfg in cfgs:
        for field, value in forced.items():
            try:
                setattr(cfg, field, value)
            except (AttributeError, TypeError, ValueError):  # pydantic-frozen model
                object.__setattr__(cfg, field, value)


def _stub_delete(sa, calls, *, success):
    """Replace the scoped-delete submission with a recording stub."""
    builder = sa._workflow_builder._base_builder

    def _fake(*args, **kwargs):
        calls.append(kwargs)
        return {
            "success": success,
            "returncode": 0 if success else 1,
            "snakemake_logfile": "/tmp/fake_reprocess_delete.log",
        }

    builder.submit_reprocess_delete_workflow = _fake


def test_failed_scoped_delete_raises_instead_of_consolidating_stale(
    synthetic_sensitivity_completed_isolated,
):
    """A scoped reprocess-delete that reports failure must RAISE, not fall through.

    Before Edit 3 the returned dict was discarded entirely, so this call completed
    'successfully' while the consolidated trees survived untouched.
    """
    sa = synthetic_sensitivity_completed_isolated
    _force_hpc(sa)
    calls = []
    _stub_delete(sa, calls, success=False)

    with pytest.raises(WorkflowError) as excinfo:
        sa.reprocess(
            start_with="consolidate",
            execution_mode="auto",
            regenerate_existing=True,
            verbose=False,
        )

    assert calls, "the scoped delete must have been routed before the failure check"
    msg = str(excinfo.value)
    # The message must point at the executor-INDEPENDENT capture Edit 4 creates.
    assert "logs/delete_reprocess/" in msg, (
        "the WorkflowError must name the toolkit-owned per-rule log directory; "
        f"got: {msg!r}"
    )
    assert "NOT invalidated" in msg, "the message must state that the trees survived"


def test_explicit_local_execution_mode_is_not_overridden_by_run_method(
    synthetic_sensitivity_completed_isolated,
):
    """execution_mode='local' must keep the scoped delete off the SLURM route.

    The config says batch_job; the caller says local. The caller wins. Before
    Edit 3 the config won silently, so a caller forcing local execution still had
    the deletion offloaded to SLURM.
    """
    sa = synthetic_sensitivity_completed_isolated
    _force_hpc(sa)
    calls = []
    # success=False so that IF the SLURM route were taken, the raise would fire
    # and fail this test loudly rather than passing for the wrong reason.
    _stub_delete(sa, calls, success=False)

    result = sa.reprocess(
        start_with="consolidate",
        execution_mode="local",
        regenerate_existing=True,
        verbose=False,
    )

    assert not calls, (
        "execution_mode='local' must NOT route the scoped delete through the "
        f"SLURM workflow; it was invoked {len(calls)} time(s)."
    )
    assert result["success"], f"local-route reprocess failed: {result.get('message')!r}"
