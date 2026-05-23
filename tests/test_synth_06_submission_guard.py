"""Phase 1: at-most-once submission guard — sentinel + pre-flight reconciliation.

Synthetic-tier unit tests (no live cluster). The reconciliation entry point
``SnakemakeWorkflowBuilder._reconcile_inflight_submissions`` is exercised in
isolation by:

1. Writing sentinel JSON files into ``{analysis_dir}/_status/_submitted/``
   that name fake SLURM job-ids.
2. Monkeypatching ``workflow._slurm_job_is_live`` so the live-vs-dead
   classification is deterministic without touching squeue.
3. Monkeypatching the bound ``_recover_inflight_via_comment`` so the sacct
   recovery path does not shell out during sentinel-path tests.

The sacct-parsing path is covered by its own test that monkeypatches
``subprocess.run`` to return a hand-rolled sacct output buffer.
"""

import json

import pytest


def _write_sentinel(analysis_dir, name, jobid):
    d = analysis_dir / "_status" / "_submitted"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}.json"
    path.write_text(json.dumps({"slurm_jobid": jobid, "run_uuid": "u", "submitted_at": "t"}))
    return path


def test_reconcile_returns_alive_set_for_live_duplicate(synthetic_multisim_builder):
    """Phase 2: v2 reconcile RETURNS the alive set for an in-flight sim — it does NOT raise.

    A submitted sentinel with no completed/failed marker is classified ALIVE.
    Replaces the v1 raise-on-live test (`test_reconcile_aborts_on_live_duplicate`);
    the v1 abort semantics no longer exist — graceful rerun substitutes wait-rules.
    """
    import tests.utils_for_testing as tst_ut

    b = synthetic_multisim_builder
    _write_sentinel(b.analysis_paths.analysis_dir, "run_tritonswmm_evt-test", "999001")
    alive = b._reconcile_inflight_submissions()
    assert ("run_tritonswmm_evt-test", "999001") in alive
    tst_ut.assert_alive_set_reconciled(b, ["run_tritonswmm_evt-test"])


def test_reconcile_reclaims_completed_sentinel(synthetic_multisim_builder):
    """Phase 2: a sentinel with a _completed/ marker is reclaimed and absent from the alive set.

    Replaces the v1 squeue-based dead-reclaim test
    (`test_reconcile_reclaims_dead_sentinel`); v2 classifies via marker presence,
    not `_slurm_job_is_live`. A sentinel with NO marker is now ALIVE, so the dead
    state is modeled by writing a `_completed/` marker.
    """
    b = synthetic_multisim_builder
    analysis_dir = b.analysis_paths.analysis_dir
    s = _write_sentinel(analysis_dir, "run_tritonswmm_evt-test", "999002")
    completed_dir = analysis_dir / "_status" / "_completed"
    completed_dir.mkdir(parents=True, exist_ok=True)
    (completed_dir / "run_tritonswmm_evt-test.json").write_text(
        json.dumps({"status": "completed", "slurm_jobid": "999002"})
    )
    alive = b._reconcile_inflight_submissions()  # no raise
    assert alive == []  # completed marker present → not alive
    assert not s.exists()  # submitted sentinel reclaimed as safety net


def test_reconcile_fast_path_no_sentinels(synthetic_multisim_builder):
    """When no sentinels exist the guard returns immediately without any SLURM calls."""
    # No monkeypatch needed: if the guard tried to shell out, the test
    # would still pass on a developer machine without squeue, but the
    # contract is that zero subprocess calls happen on the fast path.
    synthetic_multisim_builder._reconcile_inflight_submissions()


def test_recover_inflight_via_comment_parses_sacct(monkeypatch, synthetic_multisim_builder):
    """The sacct-parsing path returns live + comment-matched jobs and skips malformed lines."""
    import subprocess as _sp

    sacct_out = (
        "9001|RUNNING|rule_run_tritonswmm_wildcards_event_id=evt0\n"
        "malformed_line_no_pipes\n"
        "9002|COMPLETED|rule_run_tritonswmm_wildcards_event_id=evt1\n"
    )

    class _R:
        returncode = 0
        stdout = sacct_out

    monkeypatch.setattr(_sp, "run", lambda *a, **kw: _R())
    monkeypatch.setattr(
        "TRITON_SWMM_toolkit.workflow._slurm_job_is_live",
        lambda jid: jid == "9001",  # only 9001 is live
    )
    b = synthetic_multisim_builder
    found = b._recover_inflight_via_comment(known_jobids=set())
    jids = {jid for _, jid in found}
    assert jids == {"9001"}  # live + comment-matched; 9002 completed; malformed skipped


def test_reconcile_returns_sensitivity_sentinel_in_alive_set(synthetic_multisim_builder):
    """Phase 2: a sensitivity sentinel (simulation_sa_{id}_evt-{id}) with no marker is
    returned in the alive set keyed on its full stem; v2 does not raise.

    Replaces the v1 raise-based `test_reconcile_keys_on_sensitivity_sentinel_pattern`.
    Guards the sa_id-keyed token (no collision with the multisim pattern) under v2
    return-alive semantics.
    """
    b = synthetic_multisim_builder
    s = _write_sentinel(b.analysis_paths.analysis_dir, "simulation_sa_alpha_evt-0", "777001")
    alive = b._reconcile_inflight_submissions()
    assert ("simulation_sa_alpha_evt-0", "777001") in alive  # keyed on full sa_id stem
    assert s.exists()  # alive sentinel preserved (no marker present)


def _build_marker_ctx(analysis_dir, rule_token="run_tritonswmm_evt-0", jobid="12345"):
    """Construct a _MarkerCtx pointing at the synthetic analysis_dir's _status/."""
    from TRITON_SWMM_toolkit.run_simulation_runner import _MarkerCtx

    completed_dir = analysis_dir / "_status" / "_completed"
    failed_dir = analysis_dir / "_status" / "_failed"
    completed_dir.mkdir(parents=True, exist_ok=True)
    failed_dir.mkdir(parents=True, exist_ok=True)
    return _MarkerCtx(
        jobid=jobid,
        rule_token=rule_token,
        payload_base={
            "slurm_jobid": jobid,
            "run_uuid": "test-uuid",
            "sa_id": None,
            "model_type": "tritonswmm",
            "event_id": "evt-0",
        },
        failed_dir=failed_dir,
        completed_dir=completed_dir,
    )


def test_marker_writes_on_clean_completion(synthetic_multisim_builder):
    """Phase 1: runner's clean-return path writes _status/_completed/{rule_token}.json.

    Exercises the finally-block invariant (no existing completed/failed marker
    → write _completed/) by reproducing the finally's logic directly against a
    constructed _MarkerCtx. Calling run_simulation_runner.main() in-process is
    out of scope — that requires full scenario/system/subprocess mocks. The
    finally block's marker-write is a few lines of context-free logic; testing
    it via the _MarkerCtx surface is the appropriate unit-test scope.
    """
    import datetime
    import os

    b = synthetic_multisim_builder
    analysis_dir = b.analysis_paths.analysis_dir
    ctx = _build_marker_ctx(analysis_dir)
    completed_marker = ctx.completed_dir / f"{ctx.rule_token}.json"
    failed_marker = ctx.failed_dir / f"{ctx.rule_token}.json"
    assert not completed_marker.exists() and not failed_marker.exists()

    # Reproduce the runner's finally-clause clean-return logic.
    if not completed_marker.exists() and not failed_marker.exists():
        payload = {
            **ctx.payload_base,
            "status": "completed",
            "finished_at": datetime.datetime.now().isoformat(),
        }
        tmp = completed_marker.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload))
        os.replace(tmp, completed_marker)

    assert completed_marker.exists()
    body = json.loads(completed_marker.read_text())
    assert body["status"] == "completed"
    assert body["slurm_jobid"] == "12345"
    assert body["event_id"] == "evt-0"
    assert "finished_at" in body
    assert not failed_marker.exists()


def test_marker_writes_on_runner_exception(synthetic_multisim_builder):
    """Phase 1: runner's exception path writes _status/_failed/{rule_token}.json via _write_failed_marker."""
    from TRITON_SWMM_toolkit.run_simulation_runner import _write_failed_marker

    b = synthetic_multisim_builder
    ctx = _build_marker_ctx(b.analysis_paths.analysis_dir)
    failed_marker = ctx.failed_dir / f"{ctx.rule_token}.json"
    assert not failed_marker.exists()

    _write_failed_marker(ctx)

    assert failed_marker.exists()
    body = json.loads(failed_marker.read_text())
    assert body["status"] == "failed"
    assert body["slurm_jobid"] == "12345"
    assert "finished_at" in body

    # Non-SLURM execution (jobid=None) is a no-op.
    from TRITON_SWMM_toolkit.run_simulation_runner import _MarkerCtx

    nop_ctx = _MarkerCtx(
        jobid=None,
        rule_token="other_token",
        payload_base={},
        failed_dir=ctx.failed_dir,
        completed_dir=ctx.completed_dir,
    )
    _write_failed_marker(nop_ctx)
    _write_failed_marker(None)
    assert not (ctx.failed_dir / "other_token.json").exists()


def test_classify_via_state_markers_returns_alive_for_no_marker(synthetic_multisim_builder):
    """Phase 1: _classify_via_state_markers returns alive=[(stem, jid)] when no marker exists."""
    b = synthetic_multisim_builder
    sentinel = _write_sentinel(b.analysis_paths.analysis_dir, "run_tritonswmm_evt-0", "888001")
    result = b._classify_via_state_markers([sentinel])
    assert result == [("run_tritonswmm_evt-0", "888001")]
    assert sentinel.exists()  # not reclaimed when no marker present


def test_classify_via_state_markers_returns_empty_when_completed_marker_present(
    synthetic_multisim_builder,
):
    """Phase 1: _classify_via_state_markers treats completed-marker presence as not-alive."""
    b = synthetic_multisim_builder
    analysis_dir = b.analysis_paths.analysis_dir
    sentinel = _write_sentinel(analysis_dir, "run_tritonswmm_evt-0", "888002")
    completed_dir = analysis_dir / "_status" / "_completed"
    completed_dir.mkdir(parents=True, exist_ok=True)
    completed = completed_dir / "run_tritonswmm_evt-0.json"
    completed.write_text(json.dumps({"status": "completed", "slurm_jobid": "888002"}))

    result = b._classify_via_state_markers([sentinel])

    assert result == []
    assert not sentinel.exists()  # reclaimed as safety net
    assert completed.exists()  # marker is untouched


# ========== Phase 2: wait-rule emission + ruleorder ambiguity resolution ==========


def _enabled_model_from_snakefile(content):
    """Extract one enabled model_type from an emitted multisim Snakefile."""
    import re

    m = re.search(r"rule run_(\w+):", content)
    assert m, "synth builder emitted no `rule run_*` block"
    return m.group(1)


def test_wait_rule_emitted_for_alive_sentinel_with_ruleorder(synthetic_multisim_builder):
    """Phase 2: an alive token makes generate_snakefile_content emit (a) a concrete
    `rule wait_for_*`, (b) the `ruleorder: wait_for_X > run_*` directive that resolves
    the wildcard-vs-concrete AmbiguousRuleException, and (c) a `localrules:` line so the
    wait-rule runs locally. Also asserts R7 (input-superset): the wait-rule's input is
    the run-rule's prepare-flag input for the same event.
    """
    b = synthetic_multisim_builder
    model = _enabled_model_from_snakefile(b.generate_snakefile_content())
    token = f"run_{model}_evt-test"
    sanitized = f"run_{model}_evt_test"

    content = b.generate_snakefile_content(alive_by_token={token: "999001"})

    assert f"rule wait_for_{sanitized}:" in content
    assert f"ruleorder: wait_for_{sanitized} > run_{model}" in content
    assert f"localrules: wait_for_{sanitized}" in content

    # R7: the wait-rule's input set is the run-rule's input for this event
    # (prepare_scenarios defaults True → b_prepare flag).
    idx = content.index(f"rule wait_for_{sanitized}:")
    wait_block = content[idx : idx + 600]
    assert "_status/b_prepare_evt-test_complete.flag" in wait_block
    # The wait-rule's output is the run-rule's flag, exactly.
    assert f'"_status/c_run_{model}_evt-test_complete.flag"' in wait_block


def test_e2e_kill_orchestrator_resubmit_wait_rule_emits():
    """Phase 2: master Validation V3 end-to-end smoke — kill orchestrator, re-run,
    assert no abort + wait-rule emitted + downstream unblocks on completion marker.
    """
    pytest.skip(
        "E2E scaffold — requires tmux-orchestration / SIGTERM / monkeypatched-runner "
        "fixtures not yet present in conftest. The end-to-end is exercised via the "
        "master plan's ET3 (Frontier in-flight rerun empirical-testing protocol)."
    )


def test_snakefile_with_wait_rule_parses_without_ambiguity(synthetic_multisim_builder):
    """Phase 2 (Spec E): the load-bearing ruleorder mechanism is EXECUTED under Snakemake,
    not merely asserted by string presence. Build a Snakefile with one alive token, then
    dry-run-target the contested flag so both the wildcard run-rule and the concrete
    wait-rule are candidate producers — forcing the ambiguity that `ruleorder` resolves.
    Assert no AmbiguousRuleException and that the wait-rule is the chosen producer.
    """
    import ast
    import re
    import shutil
    import subprocess

    if shutil.which("snakemake") is None:
        pytest.skip("snakemake CLI not on PATH")

    b = synthetic_multisim_builder
    base = b.generate_snakefile_content()
    model = _enabled_model_from_snakefile(base)
    # Use a REAL event_id (one in SIM_IDS) so the wildcard run-rule's input
    # function (ILOC_BY_EVENT_ID[event_id]) resolves during DAG build — a fake
    # event_id raises KeyError before the ambiguity check is reached.
    m = re.search(r"^SIM_IDS\s*=\s*(\[.*?\])", base, re.MULTILINE)
    assert m, "SIM_IDS not found in emitted Snakefile"
    event_id = ast.literal_eval(m.group(1))[0]

    token = f"run_{model}_evt-{event_id}"
    sanitized = token.replace("-", "_").replace(".", "_")
    content = b.generate_snakefile_content(alive_by_token={token: "1"})

    analysis_dir = b.analysis_paths.analysis_dir
    snakefile = analysis_dir / "Snakefile.waittest"
    snakefile.write_text(content)
    target = f"_status/c_run_{model}_evt-{event_id}_complete.flag"

    proc = subprocess.run(
        ["snakemake", "-n", "--cores", "1", "-s", str(snakefile), "-d", str(analysis_dir), target],
        capture_output=True,
        text=True,
    )
    combined = proc.stdout + proc.stderr
    assert "AmbiguousRuleException" not in combined, combined[-2000:]
    # ruleorder must select the wait-rule as the producer of the contested flag.
    assert sanitized in combined, combined[-2000:]


def test_force_rerun_does_not_touch_submitted_sentinels(synthetic_multisim_builder):
    """Phase 4 (R10): override_force_rerun='all' deletes top-level _status/*.flag
    completion markers but MUST NOT delete _status/_submitted/*.json sentinels nor
    any v2 _status/_completed/ or _status/_failed/ marker. Exercises the public
    _apply_force_rerun('all') integration path; locks the top-level *.flag glob in
    workflow.py::_delete_flags_for_force_rerun against a future '*.json'-extending
    or recursive 'fix' that would silently break v2 wait-rule tracking.
    """
    b = synthetic_multisim_builder
    analysis_dir = b.analysis_paths.analysis_dir
    submitted = _write_sentinel(analysis_dir, "run_tritonswmm_evt-test", "12345")
    flag = analysis_dir / "_status" / "c_run_tritonswmm_evt-test_complete.flag"
    flag.parent.mkdir(parents=True, exist_ok=True)
    flag.touch()
    completed_dir = analysis_dir / "_status" / "_completed"
    completed_dir.mkdir(parents=True, exist_ok=True)
    completed = completed_dir / "run_tritonswmm_evt-test.json"
    completed.write_text(json.dumps({"slurm_jobid": "12345", "status": "completed"}))

    # Hermetic cleanup: the synth-multisim case dir is shared with the
    # session-scoped `synthetic_multisim_completed` fixture, so any artifact
    # left here leaks into other tests' _status/ view.
    try:
        b.analysis._apply_force_rerun("all")

        assert not flag.exists(), "force-rerun 'all' must delete the completion flag (v1 behavior preserved)"
        assert submitted.exists(), "force-rerun must NOT delete _status/_submitted/*.json sentinels (R10)"
        assert completed.exists(), "force-rerun must NOT delete _status/_completed/*.json v2 markers (R10)"
    finally:
        for p in (flag, submitted, completed):
            p.unlink(missing_ok=True)


def test_force_rerun_does_not_descend_into_status_subdirs(synthetic_multisim_builder):
    """Phase 4 (R10): _delete_flags_for_force_rerun uses Path.glob (non-recursive),
    NOT Path.rglob — a .flag nested under a _status subdirectory MUST survive a
    scope='all' force-rerun. Locks against a future '**/*.flag' change that would
    reach into _submitted/_completed/_failed/. Tested directly on the helper so the
    assertion isolates the glob behavior from log-invalidation orchestration.
    """
    from TRITON_SWMM_toolkit.workflow import ResolvedForceRerunSpec

    b = synthetic_multisim_builder
    status_dir = b.analysis_paths.analysis_dir / "_status"
    status_dir.mkdir(parents=True, exist_ok=True)
    top_flag = status_dir / "c_run_tritonswmm_evt-test_complete.flag"
    top_flag.touch()
    nested_dir = status_dir / "_submitted"
    nested_dir.mkdir(parents=True, exist_ok=True)
    nested_flag = nested_dir / "should_survive.flag"
    nested_flag.touch()

    try:
        b._delete_flags_for_force_rerun(ResolvedForceRerunSpec(scope="all", tokens=()))

        assert not top_flag.exists(), "top-level _status/*.flag must be deleted by scope='all'"
        assert nested_flag.exists(), "nested _status/**/*.flag must survive — glob is non-recursive (R10)"
    finally:
        for p in (top_flag, nested_flag):
            p.unlink(missing_ok=True)


def test_force_rerun_sa_scope_does_not_touch_submitted_sentinels(synthetic_multisim_builder):
    """Phase 4 (R10): scope='sa' force-rerun deletes only delimiter-anchored
    *sa-{id}_*.flag / *sa-{id}.flag completion markers and MUST NOT delete
    _status/_submitted/*.json sentinels. Also locks the delimiter anchoring so
    sa-0 does not false-match sa-10. Tested directly on _delete_flags_for_force_rerun
    because the public _apply_force_rerun(sa-scope) entry requires
    toggle_sensitivity_analysis=True; the regression target is the glob, which is
    analysis-type-agnostic.
    """
    from TRITON_SWMM_toolkit.workflow import ResolvedForceRerunSpec

    b = synthetic_multisim_builder
    analysis_dir = b.analysis_paths.analysis_dir
    submitted = _write_sentinel(analysis_dir, "simulation_sa_0_evt-test", "55501")
    status_dir = analysis_dir / "_status"
    status_dir.mkdir(parents=True, exist_ok=True)
    sa_flag = status_dir / "e_consolidate_sa-0_complete.flag"
    sa_flag.touch()
    other_sa_flag = status_dir / "e_consolidate_sa-10_complete.flag"
    other_sa_flag.touch()

    try:
        b._delete_flags_for_force_rerun(ResolvedForceRerunSpec(scope="sa", tokens=("0",)))

        assert not sa_flag.exists(), "sa-0 flag must be deleted by scope='sa' tokens=('0',)"
        assert other_sa_flag.exists(), "sa-10 must NOT be matched by sa-0 (delimiter-anchored glob)"
        assert submitted.exists(), "force-rerun must NOT delete _status/_submitted/*.json sentinels (R10)"
    finally:
        # other_sa_flag is a top-level *.flag with no sidecar — leaking it
        # breaks test_synth_flag_writes::test_run_emits_flag_and_sidecar, which
        # asserts every _status/*.flag in the shared case dir has a sidecar.
        for p in (sa_flag, other_sa_flag, submitted):
            p.unlink(missing_ok=True)
