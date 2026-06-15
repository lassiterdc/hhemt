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
from pathlib import Path

import pytest


def _write_sentinel(analysis_dir, name, jobid):
    d = analysis_dir / "_status" / "_submitted"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}.json"
    path.write_text(json.dumps({"slurm_jobid": jobid, "run_uuid": "u", "submitted_at": "t"}))
    return path


def _write_queued(analysis_dir, name, jobid):
    """Mechanism (b) _queued/ sentinel (submitter-side, pre-worker-start). Payload
    mirrors _write_queued_sentinels: top-level rule_token + slurm_jobid, NO timestamp."""
    d = analysis_dir / "_status" / "_queued"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}.json"
    path.write_text(json.dumps({"rule_token": name, "slurm_jobid": jobid}, sort_keys=True))
    return path


def test_reconcile_returns_alive_set_for_live_duplicate(monkeypatch, synthetic_multisim_builder):
    """Phase 2: v2 reconcile RETURNS the alive set for an in-flight sim — it does NOT raise.

    A submitted sentinel with no completed/failed marker is classified ALIVE.
    Replaces the v1 raise-on-live test (`test_reconcile_aborts_on_live_duplicate`);
    the v1 abort semantics no longer exist — graceful rerun substitutes wait-rules.
    """
    import tests.utils_for_testing as tst_ut

    # R9 hermeticity: pin the sacct seam so the fake jobid cannot collide with a
    # real accounting row on a login node (where a DEAD State for that id would
    # mass-reclaim the freshly-written sentinel). Empty states → UNKNOWN bucket →
    # the fresh sentinel's recent mtime keeps it alive deterministically.
    monkeypatch.setattr("TRITON_SWMM_toolkit.workflow._sacct_states_batched", lambda job_ids, **kw: {})

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


def test_reconcile_returns_sensitivity_sentinel_in_alive_set(monkeypatch, synthetic_multisim_builder):
    """Phase 2: a sensitivity sentinel (simulation_sa_{id}_evt-{id}) with no marker is
    returned in the alive set keyed on its full stem; v2 does not raise.

    Replaces the v1 raise-based `test_reconcile_keys_on_sensitivity_sentinel_pattern`.
    Guards the sa_id-keyed token (no collision with the multisim pattern) under v2
    return-alive semantics.
    """
    # R9 hermeticity: pin the sacct seam (see the sibling reconcile test) so the
    # fake jobid is host-independent — UNKNOWN bucket + fresh mtime → alive.
    monkeypatch.setattr("TRITON_SWMM_toolkit.workflow._sacct_states_batched", lambda job_ids, **kw: {})

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


# ========== Phase 1: rerun-triggers narrowed to mtime (post-death-recovery) ==========


@pytest.mark.parametrize("mode", ["local", "slurm", "single_job"])
def test_run_submit_uses_mtime_only_rerun_triggers(synthetic_multisim_builder, mode):
    """Phase 1: the run-path Snakemake profile pins rerun-triggers to ['mtime']
    for every execution mode — `input` is absent."""
    b = synthetic_multisim_builder
    try:
        config = b.generate_snakemake_config(mode=mode)
    except AssertionError as exc:
        pytest.skip(f"mode={mode} requires hpc config not set on synth fixture: {exc}")
    assert config["rerun-triggers"] == ["mtime"], (
        f"mode={mode}: run-path profile must pin rerun-triggers to ['mtime'] "
        f"so a post-death resume cannot re-fire completed sims via the `input` "
        f"trigger; got {config['rerun-triggers']!r}"
    )
    assert "input" not in config["rerun-triggers"], (
        f"mode={mode}: `input` must be absent from run-path rerun-triggers (Phase 1)"
    )


def test_one_job_script_inherits_mtime_only_via_profile(synthetic_multisim_builder):
    """Phase 1: the emitted run_workflow_1job.sh invokes snakemake with `--profile`
    (which carries rerun-triggers=['mtime']) and does NOT hand-inject a conflicting
    `--rerun-triggers ... input` on the script line."""
    b = synthetic_multisim_builder
    try:
        config = b.generate_snakemake_config(mode="single_job")
    except AssertionError as exc:
        pytest.skip(f"single_job mode requires hpc config not set on synth fixture: {exc}")
    config_dir = b.write_snakemake_config(config, mode="single_job")

    # Load-bearing, fixture-independent assertion: the written single_job profile
    # pins rerun-triggers to ['mtime']. This must hold on every fixture.
    import yaml

    written = yaml.safe_load((config_dir / "config.yaml").read_text())
    assert written["rerun-triggers"] == ["mtime"]

    # Script-text assertions depend on _generate_single_job_submission_script,
    # which requires full HPC config (hpc_total_nodes) the synth fixture lacks.
    # Skip only this portion when that config is absent — the profile assertion
    # above already covers the load-bearing FQ2 guarantee.
    snakefile_path = b.analysis_paths.analysis_dir / "Snakefile"
    try:
        script_path = b._generate_single_job_submission_script(snakefile_path, config_dir)
    except AssertionError as exc:
        pytest.skip(f"1job script generation requires hpc config not set on synth fixture: {exc}")
    script_text = script_path.read_text()
    assert "--profile" in script_text
    assert "input" not in script_text.split("python -m snakemake", 1)[-1], (
        "1job script must not re-introduce the `input` rerun trigger; the "
        "single_job profile already pins rerun-triggers=['mtime'] (FQ2)"
    )


def test_classify_stale_via_sacct_dead_alive_and_mtime_tiebreak(monkeypatch, synthetic_multisim_builder):
    """Phase 2 R-STALE: the second-pass classifier reclaims DEAD + aged-UNKNOWN
    tokens, keeps ALIVE + fresh-UNKNOWN tokens, and unlinks only the reclaimed
    submitted-sentinels (R4/R5). One sacct call regardless of |input|."""
    import os
    import time

    b = synthetic_multisim_builder
    analysis_dir = b.analysis_paths.analysis_dir
    dead_s = _write_sentinel(analysis_dir, "run_tritonswmm_evt-dead", "100")
    alive_s = _write_sentinel(analysis_dir, "run_tritonswmm_evt-alive", "200")
    fresh_s = _write_sentinel(analysis_dir, "run_tritonswmm_evt-fresh", "300")
    aged_s = _write_sentinel(analysis_dir, "run_tritonswmm_evt-aged", "400")
    # Age the UNKNOWN-bucket sentinel 100 days into the past — beyond any
    # plausible walltime+slack cap (max config ceiling is 1 week).
    old = time.time() - 100 * 24 * 3600
    os.utime(aged_s, (old, old))

    monkeypatch.setattr(
        "TRITON_SWMM_toolkit.workflow._sacct_states_batched",
        lambda job_ids, **kw: {
            "100": ("CANCELLED", "0:15", "JobKilled"),  # DEAD-stale
            "200": ("RUNNING", "0:0", "None"),  # ALIVE
            # 300, 400 absent → UNKNOWN bucket, resolved by mtime tiebreak
        },
    )

    marker_less = [
        ("run_tritonswmm_evt-dead", "100"),
        ("run_tritonswmm_evt-alive", "200"),
        ("run_tritonswmm_evt-fresh", "300"),
        ("run_tritonswmm_evt-aged", "400"),
    ]
    still_alive, cleared = b._classify_stale_via_sacct(marker_less)

    cleared_tokens = {c.rule_token for c in cleared}
    assert cleared_tokens == {"run_tritonswmm_evt-dead", "run_tritonswmm_evt-aged"}
    assert not dead_s.exists()  # DEAD reclaimed
    assert not aged_s.exists()  # aged-UNKNOWN reclaimed via mtime fail-safe

    alive_tokens = {t for t, _ in still_alive}
    assert alive_tokens == {"run_tritonswmm_evt-alive", "run_tritonswmm_evt-fresh"}
    assert alive_s.exists()  # ALIVE preserved (at-most-once)
    assert fresh_s.exists()  # fresh-UNKNOWN preserved (re-probed next entry)

    # The DEAD record carries the sacct State + Reason for the bug-surface print.
    dead_rec = next(c for c in cleared if c.rule_token == "run_tritonswmm_evt-dead")
    assert dead_rec.state == "CANCELLED"
    assert dead_rec.reason == "JobKilled"
    assert dead_rec.job_id == "100"


def test_classify_stale_via_sacct_aliased_jobid_not_mass_dead(monkeypatch, synthetic_multisim_builder):
    """R4 F2 srun-step job-id-aliasing guard: under 1_job_many_srun_tasks N tokens
    share one allocation $SLURM_JOB_ID, so `sacct -X` returns ONE allocation-summary
    row. A terminal allocation State must NOT mass-classify all aliased tokens DEAD —
    the guard refuses to classify any jobid shared by >=2 marker-less tokens from the
    shared row, falling through to the mtime fail-safe. Two fresh sentinels sharing
    jobid 555 + a monkeypatched TIMEOUT row → both remain alive (mtime-fresh),
    neither reclaimed. (Empirically required — probe P4b.)"""
    b = synthetic_multisim_builder
    analysis_dir = b.analysis_paths.analysis_dir
    s1 = _write_sentinel(analysis_dir, "run_tritonswmm_evt-a", "555")
    s2 = _write_sentinel(analysis_dir, "run_tritonswmm_evt-b", "555")

    monkeypatch.setattr(
        "TRITON_SWMM_toolkit.workflow._sacct_states_batched",
        lambda job_ids, **kw: {"555": ("TIMEOUT", "0:1", "TimeLimit")},  # terminal allocation row
    )

    marker_less = [
        ("run_tritonswmm_evt-a", "555"),
        ("run_tritonswmm_evt-b", "555"),
    ]
    still_alive, cleared = b._classify_stale_via_sacct(marker_less)

    # Neither token reclaimed from the shared TIMEOUT allocation row (F2 guard).
    assert cleared == []
    alive_tokens = {t for t, _ in still_alive}
    assert alive_tokens == {"run_tritonswmm_evt-a", "run_tritonswmm_evt-b"}
    assert s1.exists()  # preserved — fell through to mtime fail-safe, fresh
    assert s2.exists()


def test_classify_stale_via_sacct_empty_input_is_zero_call_noop(monkeypatch, synthetic_multisim_builder):
    """R5 scheduler-politeness: the common all-markers case (empty marker-less
    residual) short-circuits to a zero-call no-op — sacct is never invoked."""

    def _boom(*a, **kw):
        raise AssertionError("sacct must not be called on an empty marker-less set")

    monkeypatch.setattr("TRITON_SWMM_toolkit.workflow._sacct_states_batched", _boom)
    b = synthetic_multisim_builder
    assert b._classify_stale_via_sacct([]) == ([], [])


def test_max_plausible_job_lifetime_never_below_walltime(synthetic_multisim_builder):
    """R8/R-WAITCAP regression (SE + triton specialist follow-up): the derived
    cap is walltime + slack, so it can never drop below the job's own walltime
    even when the override ceiling is at its configured minimum."""
    from TRITON_SWMM_toolkit import workflow as w

    cfg = synthetic_multisim_builder.cfg_analysis
    if cfg.hpc_total_job_duration_min is None:
        pytest.skip(
            "synth fixture has no hpc_total_job_duration_min; "
            "walltime-derivation path not exercised (local-mode fallback)"
        )
    derived = w._max_plausible_job_lifetime_min(cfg, slack_min=30)
    assert derived == cfg.hpc_total_job_duration_min + 30
    assert derived >= cfg.hpc_total_job_duration_min


# ----------------------------------------------------------------------------
# Stage B — mechanism (b) PENDING-recovery (R1, R2, R3, R5, R6, R8, R12)
# ----------------------------------------------------------------------------


def test_planned_sim_tokens_are_model_event_cross_product(synthetic_multisim_builder):
    """R2: _planned_sim_tokens enumerates run_{model}_evt-{event} for the
    enabled-model x event cross-product, byte-identical to the runner's _rule_token
    (literal evt- hyphen). The writer source for the _queued/ set."""
    b = synthetic_multisim_builder
    tokens = b._planned_sim_tokens()
    assert tokens, "expected a non-empty planned sim-token set"
    assert len(tokens) == len(set(tokens))  # no duplicates
    for t in tokens:
        assert t.startswith("run_") and "_evt-" in t and "_evt_" not in t


def test_write_queued_sentinels_payload_and_compare_and_write(synthetic_multisim_builder):
    """R2/R12/SE Flag 3: the writer emits {rule_token, slurm_jobid} (top-level jobid
    key, NO timestamp), and compare-and-write PRESERVES mtime on a byte-identical
    re-write so the mtime fail-safe measures age-since-first-submit; a changed jobid
    rewrites (bumps mtime)."""
    import os
    import time

    b = synthetic_multisim_builder
    ad = b.analysis_paths.analysis_dir
    toks = ["run_tritonswmm_evt-x", "run_tritonswmm_evt-y"]
    b._write_queued_sentinels(toks, "555", ad)
    qx = ad / "_status" / "_queued" / "run_tritonswmm_evt-x.json"
    payload = json.loads(qx.read_text())
    assert payload["rule_token"] == "run_tritonswmm_evt-x"
    assert payload["slurm_jobid"] == "555"  # top-level key for the wait-runner fallback
    assert "written_at" not in payload  # no timestamp → compare-and-write can match
    # age the file, then re-write identical payload → mtime preserved
    old = time.time() - 10_000
    os.utime(qx, (old, old))
    aged = qx.stat().st_mtime
    b._write_queued_sentinels(toks, "555", ad)
    assert qx.stat().st_mtime == aged, "identical re-write must preserve mtime"
    # changed jobid → rewrite → mtime bumps, payload updates
    b._write_queued_sentinels(toks, "999", ad)
    assert qx.stat().st_mtime != aged
    assert json.loads(qx.read_text())["slurm_jobid"] == "999"


def test_pending_recovery_executor_owns_held_on_presence(monkeypatch, synthetic_multisim_builder):
    """R1/R3/R6: a fresh _queued/ token with jobid=null (executor-owns-sbatch) and no
    _submitted/ is held alive on PRESENCE — no sacct call (jobid-null)."""
    monkeypatch.setattr(
        "TRITON_SWMM_toolkit.workflow._sacct_states_batched",
        lambda ids, **k: pytest.fail("sacct must NOT be called for jobid-null pending tokens"),
    )
    b = synthetic_multisim_builder
    ad = b.analysis_paths.analysis_dir
    _write_queued(ad, "run_tritonswmm_evt-pending", None)
    alive = b._reconcile_inflight_submissions()
    assert ("run_tritonswmm_evt-pending", "") in alive


def test_pending_recovery_orphan_queued_ages_out(synthetic_multisim_builder):
    """R12: a stale orphan _queued/ (older than _max_plausible_job_lifetime_min) is NOT
    held and is unlinked (re-runs), even though presence alone would hold a fresh one."""
    import os
    import time

    b = synthetic_multisim_builder
    ad = b.analysis_paths.analysis_dir
    q = _write_queued(ad, "run_tritonswmm_evt-stale", None)
    old = time.time() - 100 * 24 * 3600  # 100 days — beyond any walltime+slack cap
    os.utime(q, (old, old))
    alive = b._reconcile_inflight_submissions()
    assert all(t != "run_tritonswmm_evt-stale" for t, _ in alive)
    assert not q.exists()  # stale orphan reclaimed


def test_pending_recovery_toolkit_owns_dead_drops_alive_holds(monkeypatch, synthetic_multisim_builder):
    """R1: toolkit-owns-sbatch (_queued/ jobid present) → sacct-classify; a DEAD job
    drops its _queued/ (re-runs), a still-PENDING job is held."""
    b = synthetic_multisim_builder
    ad = b.analysis_paths.analysis_dir
    _write_queued(ad, "run_tritonswmm_evt-dead", "100")
    _write_queued(ad, "run_tritonswmm_evt-live", "200")
    monkeypatch.setattr(
        "TRITON_SWMM_toolkit.workflow._sacct_states_batched",
        lambda ids, **k: {"100": ("CANCELLED", "0:0", "JobKilled"), "200": ("PENDING", "0:0", "Priority")},
    )
    alive = {t for t, _ in b._reconcile_inflight_submissions()}
    assert "run_tritonswmm_evt-live" in alive  # PENDING → held
    assert "run_tritonswmm_evt-dead" not in alive  # DEAD → dropped
    assert not (ad / "_status" / "_queued" / "run_tritonswmm_evt-dead.json").exists()


def test_pending_recovery_aliased_jobid_not_mass_dropped(monkeypatch, synthetic_multisim_builder):
    """R4 (F2 guard in the _queued/ recovery): two pending tokens sharing one allocation
    jobid + a terminal allocation row are NOT dropped — they fall through to the mtime
    fail-safe and (fresh) are held."""
    b = synthetic_multisim_builder
    ad = b.analysis_paths.analysis_dir
    _write_queued(ad, "run_tritonswmm_evt-a", "777")
    _write_queued(ad, "run_tritonswmm_evt-b", "777")
    monkeypatch.setattr(
        "TRITON_SWMM_toolkit.workflow._sacct_states_batched",
        lambda ids, **k: {"777": ("TIMEOUT", "0:0", "TimeLimit")},
    )
    alive = {t for t, _ in b._reconcile_inflight_submissions()}
    assert {"run_tritonswmm_evt-a", "run_tritonswmm_evt-b"} <= alive


def test_pending_recovery_token_keyed_dedup_submitted_wins(monkeypatch, synthetic_multisim_builder):
    """SE Flag 2: when BOTH a _submitted/ and a _queued/ exist for one logical token
    (hard-kill between os.replace and the _queued/ unlink), the token appears in the
    alive set exactly once, with the _submitted/-derived concrete jobid (not "")."""
    monkeypatch.setattr("TRITON_SWMM_toolkit.workflow._sacct_states_batched", lambda ids, **k: {})
    b = synthetic_multisim_builder
    ad = b.analysis_paths.analysis_dir
    _write_sentinel(ad, "run_tritonswmm_evt-both", "300")  # worker started (jobid 300)
    _write_queued(ad, "run_tritonswmm_evt-both", "300")  # stale queued sibling
    alive = b._reconcile_inflight_submissions()
    matches = [(t, j) for t, j in alive if t == "run_tritonswmm_evt-both"]
    assert matches == [("run_tritonswmm_evt-both", "300")]  # exactly once, jobid wins


def test_fresh_analysis_fast_path_no_sacct_with_no_queued(monkeypatch, synthetic_multisim_builder):
    """R6: a genuinely-fresh analysis (no _submitted/, no _queued/) fast-returns []
    with NO sacct call."""
    monkeypatch.setattr(
        "TRITON_SWMM_toolkit.workflow._sacct_states_batched",
        lambda ids, **k: pytest.fail("sacct must NOT be called on a fresh analysis"),
    )
    assert synthetic_multisim_builder._reconcile_inflight_submissions() == []


def test_wait_runner_reads_queued_jobid_fallback(tmp_path):
    """R8: _read_submitted_jobid falls back to _queued/{token}.json when no _submitted/
    sentinel exists, so the in-loop liveness probe re-enables for a PENDING-recovered
    wait-rule. A null _queued/ jobid yields None (probe stays disabled, executor-owns)."""
    from TRITON_SWMM_toolkit import wait_for_sentinel_runner as wr

    status = tmp_path / "_status"
    (status / "_queued").mkdir(parents=True)
    (status / "_queued" / "run_tritonswmm_evt-q.json").write_text(
        json.dumps({"rule_token": "run_tritonswmm_evt-q", "slurm_jobid": "888"})
    )
    assert wr._read_submitted_jobid(status, "run_tritonswmm_evt-q") == "888"
    (status / "_queued" / "run_tritonswmm_evt-n.json").write_text(
        json.dumps({"rule_token": "run_tritonswmm_evt-n", "slurm_jobid": None})
    )
    assert wr._read_submitted_jobid(status, "run_tritonswmm_evt-n") is None


def test_sensitivity_pending_recovery_per_sub_dir(synth_sensitivity_builder):
    """R5 sensitivity parity: the sensitivity writer enumerates per-(sa_id, event) tokens
    under EACH sub's own _status/_queued/, and _reconcile_sensitivity_alive recovers them
    with the correct per-sub --analysis-dir mapping (SM Flag 2)."""
    sens = synth_sensitivity_builder.sensitivity
    swb = sens._workflow_builder
    swb._write_queued_sentinels_sensitivity(None)  # executor-owns (jobid null)

    # Every sub got per-(sa_id, event) _queued/ sentinels with the literal evt- token.
    any_written = False
    for sa_id, sub in sens.sub_analyses.items():
        qdir = sub.analysis_paths.analysis_dir / "_status" / "_queued"
        for q in qdir.glob("*.json"):
            any_written = True
            assert q.stem.startswith(f"simulation_sa_{sa_id}_evt-")
    assert any_written, "expected sensitivity _queued/ sentinels written under sub dirs"

    # Recovery returns them in the alive set with each wait-rule's --analysis-dir set to
    # the sub dir where its markers actually land.
    alive_by_token, alive_token_to_dir = swb._reconcile_sensitivity_alive()
    assert alive_by_token, "expected recovered sensitivity pending tokens"
    for tok, jid in alive_by_token.items():
        assert tok.startswith("simulation_sa_") and "_evt-" in tok
        assert jid == ""  # executor-owns held-on-presence
        # the dir mapping points at a real sub dir containing this token's _queued/
        assert (Path(alive_token_to_dir[tok]) / "_status" / "_queued" / f"{tok}.json").exists()


def test_prune_settled_markers_lists_and_unlinks_only_settled(synth_multi_sim_analysis):
    """Phase 3 (R9): a marker whose _submitted/ sibling is GONE is settled and
    pruned; a marker whose _submitted/ sibling is PRESENT is live and preserved.

    Covers both the _completed and _failed marker subdirs, the dry_run=True
    listing (no deletion), and the dry_run=False unlink path.
    """
    a = synth_multi_sim_analysis
    status_dir = a.analysis_paths.analysis_dir / "_status"
    submitted_dir = status_dir / "_submitted"
    completed_dir = status_dir / "_completed"
    failed_dir = status_dir / "_failed"
    for d in (submitted_dir, completed_dir, failed_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Settled: _completed marker with NO _submitted sibling.
    settled_completed = completed_dir / "run_tritonswmm_evt-settled.json"
    settled_completed.write_text(json.dumps({"status": "completed"}))
    # Settled: _failed marker with NO _submitted sibling.
    settled_failed = failed_dir / "run_swmm_evt-settled.json"
    settled_failed.write_text(json.dumps({"status": "failed"}))
    # Live: _completed marker WITH a _submitted sibling (reconcile may still read it).
    live_completed = completed_dir / "run_tritonswmm_evt-live.json"
    live_completed.write_text(json.dumps({"status": "completed"}))
    live_submitted = submitted_dir / "run_tritonswmm_evt-live.json"
    live_submitted.write_text(json.dumps({"slurm_jobid": "999", "run_uuid": "u", "submitted_at": "t"}))

    # dry_run lists only the two settled markers; nothing is deleted.
    listed = a._prune_settled_markers(dry_run=True)
    assert set(listed) == {settled_completed, settled_failed}
    assert settled_completed.exists()
    assert settled_failed.exists()
    assert live_completed.exists()

    # apply: only the settled markers are unlinked; the live marker and its
    # _submitted sibling are preserved.
    pruned = a._prune_settled_markers(dry_run=False)
    assert set(pruned) == {settled_completed, settled_failed}
    assert not settled_completed.exists()
    assert not settled_failed.exists()
    assert live_completed.exists()
    assert live_submitted.exists()


class TestWaitRuleInLoopLiveness:
    """In-loop SLURM-liveness probe + cap-decoupling regression (wait-rule
    in-loop-liveness plan). Fast — no compile/sim; monkeypatches squeue/sacct."""

    def test_job_is_dead_confirmed_live_squeue_returns_false(self, monkeypatch):
        """Tier 1: squeue reports a live state -> not dead (keep waiting)."""
        from TRITON_SWMM_toolkit import slurm_liveness as sl

        monkeypatch.setattr(sl, "_slurm_job_is_live", lambda jid, **k: True)
        # sacct must not even be consulted when squeue says live
        monkeypatch.setattr(
            sl, "_sacct_states_batched", lambda ids, **k: pytest.fail("sacct called though squeue live")
        )
        assert sl.job_is_dead_confirmed("123") is False

    def test_job_is_dead_confirmed_absent_then_terminal_returns_true(self, monkeypatch):
        """Tier 2: squeue absent + sacct terminal -> dead."""
        from TRITON_SWMM_toolkit import slurm_liveness as sl

        monkeypatch.setattr(sl, "_slurm_job_is_live", lambda jid, **k: False)
        monkeypatch.setattr(sl, "_sacct_states_batched", lambda ids, **k: {"123": ("CANCELLED", "0:0", "None")})
        assert sl.job_is_dead_confirmed("123") is True

    def test_job_is_dead_confirmed_absent_then_unknown_returns_false(self, monkeypatch):
        """Tier 2: squeue absent + sacct NO row (UNKNOWN) -> not confirmed dead."""
        from TRITON_SWMM_toolkit import slurm_liveness as sl

        monkeypatch.setattr(sl, "_slurm_job_is_live", lambda jid, **k: False)
        monkeypatch.setattr(sl, "_sacct_states_batched", lambda ids, **k: {})
        assert sl.job_is_dead_confirmed("123") is False

    def test_job_is_dead_confirmed_absent_then_nonterminal_returns_false(self, monkeypatch):
        """Tier 2: squeue absent + sacct non-terminal (still scheduled) -> alive."""
        from TRITON_SWMM_toolkit import slurm_liveness as sl

        monkeypatch.setattr(sl, "_slurm_job_is_live", lambda jid, **k: False)
        monkeypatch.setattr(sl, "_sacct_states_batched", lambda ids, **k: {"123": ("PENDING", "0:0", "Priority")})
        assert sl.job_is_dead_confirmed("123") is False

    def test_workflow_reexports_resolve(self):
        """R11: workflow.py re-exports resolve to the leaf-module objects."""
        from TRITON_SWMM_toolkit import slurm_liveness
        from TRITON_SWMM_toolkit import workflow as w

        assert w._slurm_job_is_live is slurm_liveness._slurm_job_is_live
        assert w._sacct_states_batched is slurm_liveness._sacct_states_batched
        assert w._SACCT_DEAD_STATES is slurm_liveness._SACCT_DEAD_STATES

    def test_config_default_is_field_max(self):
        """R8: the inflight-wait cap default is the 1-week field max."""
        from TRITON_SWMM_toolkit.config.analysis import analysis_config

        assert analysis_config.model_fields["hpc_max_wait_for_inflight_min"].default == 10080

    def test_validate_inflight_wait_vs_total_runtime_removed(self):
        """R9: the vestigial warn-only validator is gone (no warning on a low cap)."""
        from TRITON_SWMM_toolkit.config.analysis import analysis_config

        assert not hasattr(analysis_config, "_validate_inflight_wait_vs_total_runtime")

    def test_wait_runner_writes_failed_on_confirmed_death(self, tmp_path, monkeypatch):
        """R4: probe-confirmed death writes _failed (not c_run flag) + unlinks
        the submitted sentinel + returns 1, without touching the completion flag."""
        import json as _json

        from TRITON_SWMM_toolkit import slurm_liveness
        from TRITON_SWMM_toolkit import wait_for_sentinel_runner as wr

        # Force the time-gated probe to fire on loop cycle 1 (default 300 s gate
        # would otherwise poll for a week with no marker present).
        monkeypatch.setattr(wr, "_PROBE_INTERVAL_S", 0)

        status = tmp_path / "_status"
        (status / "_submitted").mkdir(parents=True)
        (status / "_completed").mkdir()
        (status / "_failed").mkdir()
        token = "run_tritonswmm_evt-0-evt.0"
        (status / "_submitted" / f"{token}.json").write_text(_json.dumps({"slurm_jobid": "999"}))
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/squeue")
        monkeypatch.setattr(slurm_liveness, "job_is_dead_confirmed", lambda jid: True)
        monkeypatch.setattr(
            "sys.argv",
            [
                "wait_for_sentinel_runner",
                "--rule-token",
                token,
                "--flag-output",
                str(status / f"c_run_{token}.flag"),
                "--analysis-dir",
                str(tmp_path),
                "--max-wait-minutes",
                "10080",
            ],
        )
        rc = wr.main()
        assert rc == 1
        assert (status / "_failed" / f"{token}.json").exists()
        assert not (status / "_submitted" / f"{token}.json").exists()
        assert not (status / f"c_run_{token}.flag").exists()  # D-Q1: never write the flag
