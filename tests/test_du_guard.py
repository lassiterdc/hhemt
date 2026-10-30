"""The DU guard fires, spares processing, and is ABSENT when unconfigured."""

from __future__ import annotations

import json
import os
import time
import types

from hhemt.workflow import SnakemakeWorkflowBuilder


def _fake_statvfs(free_bytes: int, calls: list):
    def _f(path):
        calls.append(str(path))
        return types.SimpleNamespace(f_bavail=free_bytes, f_frsize=1, f_blocks=10**13)

    return _f


def _builder(tmp_path, *, floor: int | None):
    """Returns (builder, cancels, fp_calls). fp_calls is the POSITIVE CONTROL."""
    analysis_dir = tmp_path / "analysis"
    (analysis_dir / "_status").mkdir(parents=True)
    b = SnakemakeWorkflowBuilder.__new__(SnakemakeWorkflowBuilder)
    b.analysis_paths = types.SimpleNamespace(analysis_dir=analysis_dir)
    b.cfg_hpc_system = types.SimpleNamespace(halt_sims_below_free_bytes=floor)
    b.cfg_analysis = types.SimpleNamespace(multi_sim_run_method="batch_job")
    cancels: list[dict] = []
    fp_calls: list[int] = []

    def _stub_cancel(verbose=True, **kw):
        cancels.append(kw)
        return {"success": True, "cancelled": [("run_triton_evt-0", "111")]}

    b.analysis = types.SimpleNamespace(cancel=_stub_cancel)

    def _stub_fp():
        fp_calls.append(1)
        return "fp"

    b._status_progress_fingerprint = _stub_fp
    b._workflow_has_live_slurm_jobs = lambda **kw: set()
    b._tmux_snakemake_exit_status = lambda s: 0
    b._get_module_load_prefix = lambda: ""
    return b, cancels, fp_calls


def _drive(monkeypatch, alive_polls: int = 1):
    """tmux has-session: ALIVE for `alive_polls`, then GONE."""
    state = {"n": 0}

    def _run(argv, *a, **kw):
        state["n"] += 1
        rc = 0 if state["n"] <= alive_polls else 1
        return types.SimpleNamespace(returncode=rc, stdout="", stderr="")

    monkeypatch.setattr("hhemt.workflow.subprocess.run", _run)


def test_guard_fires_below_the_floor(tmp_path, monkeypatch):
    calls: list[str] = []
    b, cancels, fp_calls = _builder(tmp_path, floor=100 * 10**9)
    monkeypatch.setattr(os, "statvfs", _fake_statvfs(1 * 10**9, calls))
    _drive(monkeypatch)
    result = b._wait_for_tmux_session_completion("sess", verbose=False, poll_interval_s=0)
    assert result["completed"] is False
    assert "DU guard" in result["message"]
    assert calls, "the guard never called statvfs"
    assert len(cancels) == 1


def test_guard_is_absent_when_the_field_is_unset(tmp_path, monkeypatch):
    calls: list[str] = []
    b, cancels, fp_calls = _builder(tmp_path, floor=None)
    monkeypatch.setattr(os, "statvfs", _fake_statvfs(1, calls))
    _drive(monkeypatch)
    b._wait_for_tmux_session_completion("sess", verbose=False, poll_interval_s=0)
    # >= 2, NOT truthiness: _status_progress_fingerprint is ALSO called PRE-LOOP at
    # workflow.py:6078, so a non-empty list is consistent with the loop exiting
    # before the guard site. The second call is the in-loop one at :6111, one line
    # after the guard at :6110 -- only that one proves control passed through.
    assert len(fp_calls) >= 2, f"guard site never reached; fp_calls={len(fp_calls)}"
    assert calls == [], f"statvfs was called on the unset path: {calls}"
    assert cancels == []
    assert not (tmp_path / "analysis" / "_status" / "_halted_du.json").exists()


def test_guard_halts_sims_and_names_only_sim_rule_classes(tmp_path, monkeypatch):
    calls: list[str] = []
    b, cancels, fp_calls = _builder(tmp_path, floor=100 * 10**9)
    monkeypatch.setattr(os, "statvfs", _fake_statvfs(1 * 10**9, calls))
    _drive(monkeypatch)
    b._wait_for_tmux_session_completion("sess", verbose=False, poll_interval_s=0)
    assert cancels[0]["rule_classes"] == ("run_", "simulation_member_")
    assert "process_" not in cancels[0]["rule_classes"]
    assert "consolidate_" not in cancels[0]["rule_classes"]


def test_selective_cancel_spares_processing_jobs(tmp_path, monkeypatch):
    from hhemt.analysis import TRITONSWMM_analysis

    adir = tmp_path / "analysis"
    sub = adir / "_status" / "_submitted"
    sub.mkdir(parents=True)
    for tok, jid in [
        ("run_triton_evt-0", "111"),
        ("simulation_member_3_evt-7", "222"),
        ("process_tritonswmm_evt-0", "333"),
        ("consolidate_scenario_evt-0", "444"),
    ]:
        (sub / f"{tok}.json").write_text(json.dumps({"rule_token": tok, "slurm_jobid": jid}))

    a = TRITONSWMM_analysis.__new__(TRITONSWMM_analysis)
    # The selective arm (_selective_cancel, 7a34488d) also reads analysis_log_directory
    # (tmux-log widening); an EMPTY log dir pins the pre-widening cancel set below.
    (adir / "logs").mkdir()
    a.analysis_paths = types.SimpleNamespace(analysis_dir=adir, analysis_log_directory=adir / "logs")
    # cancel() reads BOTH log fields and analysis_id before reaching the selective
    # arm, so the fake must carry all three. snakemake_pid is read but unused on
    # this arm -- see the ordering finding in the deliverable.
    a.log = types.SimpleNamespace(
        tmux_session_name=types.SimpleNamespace(get=lambda: "sess"),
        snakemake_pid=types.SimpleNamespace(get=lambda: 4242),
    )
    a.cfg_analysis = types.SimpleNamespace(analysis_id="test_analysis")
    # Step 1 resolves the live PID through the builder BEFORE the selective arm is
    # reached, even though that arm never uses it. Stubbed here; the pass-through
    # dependency is recorded as an ordering finding in the deliverable. The selective
    # arm also asks the builder for the run uuids (tmux-log widening, 7a34488d); an
    # empty tuple makes _squeue_live_jobids return None without a subprocess.
    a._workflow_builder = types.SimpleNamespace(
        _get_snakemake_pid_from_tmux=lambda s: 4242,
        _tmux_slurm_run_uuids=lambda: (),
    )

    argvs: list[list[str]] = []

    def _fake_run(argv, *args, **kw):
        argvs.append(list(argv))
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    # Patch the subprocess MODULE, not an attribute of hhemt.analysis. Measured,
    # in two steps: the dotted string form makes pytest try to IMPORT
    # `hhemt.analysis.subprocess` (ImportError), and the object form fails with
    # AttributeError because analysis.py has NO module-scope `import subprocess`
    # -- cancel() imports it function-locally. That local import binds the same
    # module object, so patching subprocess.run here reaches it.
    import subprocess as _sp

    monkeypatch.setattr(_sp, "run", _fake_run)
    res = a.cancel(verbose=False, rule_classes=("run_", "simulation_member_"))

    scancels = [v for v in argvs if v and v[0] == "scancel"]
    assert len(scancels) == 1, f"expected one scancel, got {argvs}"
    cancelled = set(scancels[0][1:])
    assert cancelled == {"111", "222"}, f"wrong job set cancelled: {cancelled}"
    assert "333" not in cancelled, "cancelled a PROCESSING job -- [Q212] violated"
    assert "444" not in cancelled, "cancelled a CONSOLIDATE job -- [Q212] violated"
    assert res["success"] is True
    assert any(v[:2] == ["tmux", "kill-session"] for v in argvs)


def test_selective_cancel_reaches_the_arm_when_the_snakemake_pid_is_unresolvable(tmp_path, monkeypatch):
    """Property: a `rule_classes` cancel scancels the sim class even when NO Snakemake PID
    can be resolved for the session.

    Class: REGRESSION -- red on the pre-change tree, where the selective dispatch sat below
    `elif not snakemake_pid:` and that early return killed the tmux session and returned a
    mapping with no `cancelled` key, emitting no `scancel`.

    Why it is required: the DU guard calls `cancel(rule_classes=("run_", "simulation_member_"))`
    exactly on the arm where the PID is unresolvable -- `snakemake_pid` is written into the
    analysis log only under `if snakemake_pid:`, so a logfile carrying none proves no
    submission captured one. A cancel that takes the PID-not-found return kills the driver
    and cancels no sim, so the sims it exists to stop keep running.

    What kills it: moving the `if rule_classes is not None:` dispatch back below the
    `_get_snakemake_pid_from_tmux` resolution.

    Second correct implementation: guarding the early return with `and rule_classes is None`
    instead of moving the dispatch also greens this test -- the assertion is on the emitted
    `scancel` argv, not on which statement routed control there.

    Differently-positioned satisfying input: `test_selective_cancel_spares_processing_jobs`
    (PID resolvable, returns 4242) covers the same class from the other side.
    """
    from hhemt.analysis import TRITONSWMM_analysis

    adir = tmp_path / "analysis_nopid"
    sub = adir / "_status" / "_submitted"
    sub.mkdir(parents=True)
    for tok, jid in [
        ("run_triton_evt-0", "111"),
        ("simulation_member_3_evt-7", "222"),
        ("process_tritonswmm_evt-0", "333"),
        ("consolidate_scenario_evt-0", "444"),
    ]:
        (sub / f"{tok}.json").write_text(json.dumps({"rule_token": tok, "slurm_jobid": jid}))
    (adir / "logs").mkdir()

    a = TRITONSWMM_analysis.__new__(TRITONSWMM_analysis)
    a.analysis_paths = types.SimpleNamespace(analysis_dir=adir, analysis_log_directory=adir / "logs")
    # BOTH PID sources are None: the live re-resolution AND the value recorded at submit.
    # The pre-fix `elif not snakemake_pid:` needs both falsy to be reached, so a fixture
    # that leaves the log value at 4242 (as its sibling does) never enters the defect.
    a.log = types.SimpleNamespace(
        tmux_session_name=types.SimpleNamespace(get=lambda: "sess"),
        snakemake_pid=types.SimpleNamespace(get=lambda: None),
        workflow_canceled=types.SimpleNamespace(set=lambda _v: None),
        workflow_cancellation_time=types.SimpleNamespace(set=lambda _v: None),
    )
    a.cfg_analysis = types.SimpleNamespace(analysis_id="test_analysis_nopid")
    a._workflow_builder = types.SimpleNamespace(
        _get_snakemake_pid_from_tmux=lambda s: None,
        _tmux_slurm_run_uuids=lambda: (),
    )

    argvs: list[list[str]] = []

    def _fake_run(argv, *args, **kw):
        argvs.append(list(argv))
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    import subprocess as _sp

    monkeypatch.setattr(_sp, "run", _fake_run)
    a.cancel(verbose=False, rule_classes=("run_", "simulation_member_"))

    scancels = [v for v in argvs if v and v[0] == "scancel"]
    assert len(scancels) == 1, f"selective arm emitted no scancel; argv seen: {argvs}"
    cancelled = set(scancels[0][1:])
    assert cancelled == {"111", "222"}, f"wrong job set cancelled: {cancelled}"
    assert "333" not in cancelled, "cancelled a PROCESSING job -- [Q212] violated"
    assert "444" not in cancelled, "cancelled a CONSOLIDATE job -- [Q212] violated"


def test_guard_message_does_not_claim_deliberate_sparing_without_a_cancel_key(tmp_path, monkeypatch):
    """Property: when `cancel()` returns a mapping with NO `cancelled` key, the DU guard's
    message does not carry the deliberate-sparing claim, and it is DISTINGUISHABLE from the
    message a real cancellation produces.

    Class: REGRESSION -- red on the pre-change tree, whose message read
    "Cancelled 0 simulation job(s); PROCESSING jobs left running deliberately ([Q212])"
    for a mapping carrying no `cancelled` key.

    Why it is required: an ABSENT key means the arm that would have populated it was never
    reached, so the sims were neither cancelled nor spared on purpose. `message` is the only
    channel that reaches the operator, and the pre-fix sentence asserts a deliberation that
    did not happen.

    What kills it: reinstating `_res.get("cancelled", [])` as the whole of the message, or
    collapsing the absent state onto the truthy arm's sentence.

    Second correct implementation: any three-branch body that renders the absent state
    differently from the truthy state and does not cite [Q212] passes, whatever its wording.
    The assertions are a decision-citation absence and a comparison between two rendered
    messages -- never a phrase, so a reworded-but-correct implementation is not red here.

    Not asserted: the absent-vs-empty inequality. `is None` and an empty list are different
    facts, but an implementation may honestly render both as one "what was cancelled is not
    known" sentence, and pinning their inequality would forbid that.
    """
    calls: list[str] = []
    b, cancels, fp_calls = _builder(tmp_path, floor=100 * 10**9)

    def _msg_for(payload: dict) -> str:
        # ONE builder and ONE analysis dir across both firings, so the floor, the free-space
        # reading and the halt-record path are byte-identical and the `cancelled` state is
        # the ONLY input that differs between the two rendered messages.
        b.analysis = types.SimpleNamespace(cancel=lambda verbose=True, **kw: {"success": True, **payload})
        monkeypatch.setattr(os, "statvfs", _fake_statvfs(1 * 10**9, calls))
        _drive(monkeypatch)
        return b._wait_for_tmux_session_completion("sess", verbose=False, poll_interval_s=0)["message"]

    msg = _msg_for({"message": "session killed"})  # no `cancelled` key at all
    truthy_msg = _msg_for({"cancelled": [("run_triton_evt-0", "111")]})

    assert "DU guard" in msg, msg
    # PROPERTY, not wording: `[Q212]` is the decision citation that carrying the
    # deliberate-sparing claim requires, so its absence says the claim is absent whatever
    # sentence an implementation words it in.
    assert "[Q212]" not in msg, msg
    # ...and the absent state must be DISTINGUISHED from a real cancellation. This compares
    # two rendered messages to each other, never to a phrase.
    assert msg != truthy_msg, msg


def test_guard_message_does_not_claim_deliberate_sparing_on_an_empty_cancel(tmp_path, monkeypatch):
    """Property: when `cancel()` returns `cancelled` PRESENT and EMPTY, the DU guard's
    message does not carry the deliberate-sparing claim.

    Class: REGRESSION -- red on the pre-change tree, which rendered the empty list as
    "Cancelled 0 simulation job(s); PROCESSING jobs left running deliberately ([Q212])".

    Why it is required: an EMPTY list means the arm RAN and its selection found nothing.
    That does not establish that no sim remained queued -- the selection reads
    `_status/_submitted/`, which is worker-written at process START and therefore blind to
    a PENDING sim, and an unanswerable squeue is coerced to an empty set. Reporting
    deliberate sparing here is the same untruth as the absent-key arm, one state over.

    What kills it: collapsing the two falsy states onto the truthy arm's sentence.

    Second correct implementation: any three-branch body whose empty arm does not cite
    [Q212] passes, whatever its wording -- the assertion is the decision citation's absence,
    never a phrase.

    The empty-vs-truthy DISTINCTION is pinned by
    test_guard_message_distinguishes_a_real_cancel_from_an_empty_selection and is not
    duplicated here; this test owns the claim's absence on the empty state.
    """
    calls: list[str] = []
    b, cancels, fp_calls = _builder(tmp_path, floor=100 * 10**9)
    b.analysis = types.SimpleNamespace(cancel=lambda verbose=True, **kw: {"success": True, "cancelled": []})
    monkeypatch.setattr(os, "statvfs", _fake_statvfs(1 * 10**9, calls))
    _drive(monkeypatch)
    msg = b._wait_for_tmux_session_completion("sess", verbose=False, poll_interval_s=0)["message"]
    assert "DU guard" in msg, msg
    assert "[Q212]" not in msg, msg


def test_the_halt_record_does_not_latch(tmp_path, monkeypatch):
    calls: list[str] = []
    b, cancels, fp_calls = _builder(tmp_path, floor=100 * 10**9)
    monkeypatch.setattr(os, "statvfs", _fake_statvfs(1 * 10**9, calls))
    _drive(monkeypatch)
    b._wait_for_tmux_session_completion("sess", verbose=False, poll_interval_s=0)
    record = tmp_path / "analysis" / "_status" / "_halted_du.json"
    assert record.exists()
    payload = json.loads(record.read_text())
    assert payload["free_bytes_observed"] == 1 * 10**9
    assert payload["floor_bytes"] == 100 * 10**9
    assert payload["du_sentinel_total_bytes"] is None

    b2, cancels2, fp2 = _builder(tmp_path / "second", floor=100 * 10**9)
    monkeypatch.setattr(os, "statvfs", _fake_statvfs(500 * 10**9, calls))
    _drive(monkeypatch)
    result = b2._wait_for_tmux_session_completion("sess", verbose=False, poll_interval_s=0)
    assert len(fp2) >= 2, f"guard site never reached on the second evaluation; fp_calls={len(fp2)}"
    assert cancels2 == [], "the guard fired with 500 GB free -- it latched"
    assert result["completed"] is True
    assert record.exists(), "the record was consumed; it is disclosure, not state"


def test_guard_message_distinguishes_a_real_cancel_from_an_empty_selection(tmp_path, monkeypatch):
    """Property: the DU guard renders a DIFFERENT message when `cancel()` reports a
    NON-EMPTY `cancelled` list than when it reports an EMPTY one, on an otherwise
    identical firing.

    Class: NEW CAPABILITY -- the three-branch partition at the guard's message site. The
    two sibling tests pin the ABSENT-key arm and the EMPTY-list arm independently; neither
    pins that the TRUTHY arm is a THIRD state rather than a collapse onto either.

    Why it is required -- checkable against the code alone: an absent key, an empty list
    and a non-empty list mean three different things (the arm was never reached / the arm
    ran and matched nothing / N sims were cancelled and processing was spared). If the
    truthy arm emits the empty arm's sentence, the operator reading a REAL cancellation is
    told nothing was cancelled, which is the untruth this chunk exists to remove, one
    state over.

    What kills it: making the `elif _n_cancelled:` arm emit the empty arm's sentence. That
    single edit leaves every other test in this module green, which is what makes this
    assertion the only one that holds the third state.

    Second correct implementation, differing in the thing the assertion turns on: any
    three-branch body emitting three DISTINCT sentences passes, whatever the wording --
    the assertion compares the two rendered messages to each other and never to a phrase,
    so it cannot be satisfied by a collapse and cannot be broken by a rephrase.

    Residual, named: this pins the PARTITION and not the CONTENT of the truthy sentence. A
    truthy arm emitting a third wrong-but-distinct sentence passes here; content on the
    truthy arm is the prose finding's subject, not this assertion's.
    """
    calls: list[str] = []
    b, cancels, fp_calls = _builder(tmp_path, floor=100 * 10**9)

    def _msg_for(payload):
        # ONE builder and ONE analysis dir across both firings, so the halt-record path,
        # the floor and the free-space reading are byte-identical and the ONLY input that
        # differs between the two messages is the `cancelled` value.
        b.analysis = types.SimpleNamespace(cancel=lambda verbose=True, **kw: {"success": True, "cancelled": payload})
        monkeypatch.setattr(os, "statvfs", _fake_statvfs(1 * 10**9, calls))
        _drive(monkeypatch)
        return b._wait_for_tmux_session_completion("sess", verbose=False, poll_interval_s=0)["message"]

    with_jobs = _msg_for([("run_triton_evt-0", "111")])
    without_jobs = _msg_for([])

    assert "DU guard" in with_jobs, with_jobs
    assert "DU guard" in without_jobs, without_jobs
    assert with_jobs != without_jobs, (
        "the guard renders an IDENTICAL message for a real cancellation and for an empty "
        f"selection -- the three-branch form collapsed somewhere below the return: {with_jobs!r}"
    )


# ---------------------------------------------------------------------------
# cancel()'s driver-reachability probe: THREE states, not two
# ---------------------------------------------------------------------------
#
# `tmux has-session` answers "is the DRIVER's session still there". It is asked with a
# bare `subprocess.run`, so it has three outcomes and only one of them is a negative:
# return 0 (ALIVE), a non-zero return (GONE), and a raise (UNKNOWN). The fixtures below
# differ ONLY in which of the three the probe produces.


def _cancel_harness(adir):
    """The minimal `TRITONSWMM_analysis` one `cancel()` call needs: no compile, no SLURM.

    Three `_status/_submitted/` tokens are written -- two in the sim class `cancel()` is
    asked to cancel and one PROCESSING token that must survive ([Q212]) -- so a test can
    tell a scancel of the sim class from a scancel of everything.
    """
    from hhemt.analysis import TRITONSWMM_analysis

    sub = adir / "_status" / "_submitted"
    sub.mkdir(parents=True)
    for tok, jid in [
        ("run_triton_evt-0", "111"),
        ("simulation_member_3_evt-7", "222"),
        ("process_tritonswmm_evt-0", "333"),
    ]:
        (sub / f"{tok}.json").write_text(json.dumps({"rule_token": tok, "slurm_jobid": jid}))
    (adir / "logs").mkdir()

    a = TRITONSWMM_analysis.__new__(TRITONSWMM_analysis)
    a.analysis_paths = types.SimpleNamespace(analysis_dir=adir, analysis_log_directory=adir / "logs")
    # snakemake_pid is None so the NON-selective arm can never reach `os.kill`; no test
    # here takes that arm, and a real PID in the fixture would make an accidental fall
    # through send a live signal.
    a.log = types.SimpleNamespace(
        tmux_session_name=types.SimpleNamespace(get=lambda: "sess"),
        snakemake_pid=types.SimpleNamespace(get=lambda: None),
        workflow_canceled=types.SimpleNamespace(set=lambda _v: None),
        workflow_cancellation_time=types.SimpleNamespace(set=lambda _v: None),
    )
    a.cfg_analysis = types.SimpleNamespace(analysis_id="test_probe_states")
    # An EMPTY uuid tuple makes _squeue_live_jobids answer "not known" without a
    # subprocess, so the recorded argv list holds the probe and the cancel calls only.
    a._workflow_builder = types.SimpleNamespace(
        _get_snakemake_pid_from_tmux=lambda s: None,
        _tmux_slurm_run_uuids=lambda: (),
    )
    return a


def _probe_fixture(argvs: list, *, has_session_rc: int = 0, raise_on_probe: bool = False):
    """Records every argv; answers the reachability probe with one of the three outcomes."""
    import subprocess as _sp

    def _run(argv, *args, **kw):
        argvs.append(list(argv))
        if list(argv[:2]) == ["tmux", "has-session"]:
            if raise_on_probe:
                # The reachability artifact: an Lmod cluster with no `tmux` module loaded
                # produces exactly this from a bare argv (workflow.py's tri-state liveness
                # gate learned it the same way).
                raise FileNotFoundError(2, "No such file or directory", "tmux")
            return types.SimpleNamespace(returncode=has_session_rc, stdout="", stderr="")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    return _sp, _run


def _scancels(argvs: list) -> list[list[str]]:
    return [v for v in argvs if v and v[0] == "scancel"]


def test_an_unanswerable_reachability_probe_never_claims_no_jobs_were_running(tmp_path, monkeypatch):
    """Property: when the `tmux has-session` reachability probe RAISES, `cancel()` RETURNS a
    mapping that does not claim `jobs_were_running: False`, and that mapping reports the
    failed probe instead of substituting a negative answer for it.

    Class: REGRESSION -- red on the pre-change tree, whose bare `subprocess.run` at the
    probe site let the raise propagate out of `cancel()` (pre-fix output pasted in the
    build return).

    Why it is required -- checkable against the code alone: a probe that declines to answer
    has established nothing about the drive or its queued sims. Reporting `False` there is
    the same untruth as the "already completed" arm, moved one exception handler outward:
    a "cannot determine" becomes a positive claim that nothing was running.

    What kills it: catching the raise and taking the GONE arm (the already-completed
    mapping), or catching it and reporting `jobs_were_running: False` with an empty
    `errors`. Either greens every other test in this module.

    Second correct implementation, differing in the thing the assertion turns on: omitting
    the `jobs_were_running` key entirely, or returning `None`, both pass -- the assertion is
    `.get(...) is not False` and an error list that is non-empty, never a key's presence or
    a sentence.

    Differently-positioned satisfying input, and it is the POSITIVE CONTROL: the same
    harness with an ANSWERING probe must emit its scancel and carry no error. Without it,
    "cancel() returned" would be consistent with the probe never having been reached.

    Reachability: a raise from a bare `["tmux", ...]` argv is the measured failure mode of
    an unloaded `tmux` module on an Lmod cluster -- `FileNotFoundError`, not a non-zero
    return, so it lands here rather than on GONE.

    Invocation not isolatable from this tree -- discrimination shown instead: restoring the
    bare probe call (deleting the classify step) re-raises inside `cancel()` and reds this
    test, which is what the pre-fix run below records.
    """
    argvs: list[list[str]] = []
    a = _cancel_harness(tmp_path / "analysis_unknown")
    _sp, _run = _probe_fixture(argvs, raise_on_probe=True)
    monkeypatch.setattr(_sp, "run", _run)

    res = a.cancel(verbose=False, rule_classes=("run_", "simulation_member_"))

    assert res.get("jobs_were_running") is not False, (
        f"an unanswerable probe was reported as a determined absence of running jobs: {res}"
    )
    assert res.get("errors"), f"the failed probe was not reported at all: {res}"

    # POSITIVE CONTROL -- the same fixture, the probe answering, so the two assertions
    # above cannot hold merely because the probe was never reached.
    ctl_argvs: list[list[str]] = []
    b = _cancel_harness(tmp_path / "analysis_alive_control")
    _sp2, _run2 = _probe_fixture(ctl_argvs)
    monkeypatch.setattr(_sp2, "run", _run2)
    ctl = b.cancel(verbose=False, rule_classes=("run_", "simulation_member_"))

    assert _scancels(ctl_argvs), f"control arm never reached the selective cancel: {ctl_argvs}"
    assert not ctl.get("errors"), f"control arm reported an error on an answering probe: {ctl}"


def test_a_gone_driver_still_scancels_the_sim_class(tmp_path, monkeypatch):
    """Property: a NON-ZERO return from the reachability probe still runs the selective
    cancel, and the result reports that the driver was already gone.

    Class: REGRESSION -- red on the pre-change tree, which returned the already-completed
    mapping (no `scancel` emitted, `jobs_were_running: False`) the moment the probe returned
    non-zero.

    Why it is required -- checkable against the code alone: the probe asks about the tmux
    SESSION, while the sims it submitted outlive it -- `_status/_submitted/` is written by
    the workers, and a PENDING sim has not written one yet. So the driver's absence is not
    evidence that its sims are gone, and the arm that exists to cancel them must still run.

    What kills it: returning the already-completed mapping on the GONE arm, or dispatching
    the selective arm only when the probe returned 0.

    Second correct implementation, differing in the thing the assertion turns on: keeping
    `session_canceled` True on this arm would fail only the report assertion; moving the
    selective dispatch above the probe entirely passes every assertion here, because the
    test turns on the emitted `scancel` argv and the report, not on where the branch sits.

    Reachability: the alive-loop race -- the wait loop observes the session, the driver
    exits, and `cancel()`'s own probe then returns non-zero -- which is the one instance
    the diagnosis records for this arm.

    Invocation not isolatable from this tree -- discrimination shown instead: reinstating
    the early return on a non-zero probe re-reds this test, which the pre-fix run records.
    """
    argvs: list[list[str]] = []
    a = _cancel_harness(tmp_path / "analysis_gone")
    _sp, _run = _probe_fixture(argvs, has_session_rc=1)
    monkeypatch.setattr(_sp, "run", _run)

    res = a.cancel(verbose=False, rule_classes=("run_", "simulation_member_"))

    scancels = _scancels(argvs)
    assert len(scancels) == 1, f"a gone driver suppressed the selective cancel; argv: {argvs}"
    assert set(scancels[0][1:]) == {"111", "222"}, f"wrong job set cancelled: {scancels[0]}"
    assert "333" not in scancels[0], "cancelled a PROCESSING job -- [Q212] violated"
    # The report must say the SESSION was not what this call ended, while the jobs were.
    assert res["session_canceled"] is False, f"claimed to have cancelled a session already gone: {res}"
    assert res["jobs_were_running"] is True, f"scancelled jobs while reporting none running: {res}"
    assert "already completed" not in res["message"], (
        f"reported the workflow as already completed while cancelling its sims: {res['message']!r}"
    )


def test_the_reachability_probe_distinguishes_alive_gone_and_unknown(tmp_path, monkeypatch):
    """Property: the three probe outcomes produce three DISTINCT reports on the properties
    an operator acts on -- `jobs_were_running`, `session_canceled` and the presence of an
    error -- rather than two reports and a collapse.

    Class: NEW CAPABILITY -- the three-state partition at the probe. The sibling tests pin
    the GONE arm and the UNKNOWN arm separately; neither pins that UNKNOWN is a THIRD state
    rather than a collapse onto GONE.

    Why it is required -- checkable against the code alone: ALIVE and GONE differ only in
    whether a session was there to kill, while UNKNOWN differs from both in having
    determined nothing. A collapse of UNKNOWN onto GONE reports "nothing was running" from
    a non-answer; a collapse of GONE onto ALIVE reports a killed session that had already
    ended.

    What kills it: routing the raise into the GONE arm (signature 2 becomes 3), or keeping
    `session_canceled: True` on the GONE arm (signature 2 becomes 1). Each leaves the two
    sibling tests green, which is what makes this the assertion that holds the partition.

    Second correct implementation, differing in the thing the assertion turns on: any body
    whose three arms differ on at least one of the three properties passes, whatever the
    messages say and whatever extra keys it adds -- the assertion reads a 3-tuple per arm.

    Reachability: the fixture differs ONLY in the probe outcome, so the three arms are the
    same call with the same inputs.

    Invocation not isolatable from this tree -- discrimination shown instead: the named
    collapse edits above change what the code does on this fixture and red the signature
    assertion; the pre-fix tree reds it by raising on the third arm.
    """
    signatures: dict[str, tuple] = {}
    for name, kwargs in [
        ("alive", {"has_session_rc": 0}),
        ("gone", {"has_session_rc": 1}),
        ("unknown", {"raise_on_probe": True}),
    ]:
        argvs: list[list[str]] = []
        a = _cancel_harness(tmp_path / name)
        _sp_mod, _run = _probe_fixture(argvs, **kwargs)
        monkeypatch.setattr(_sp_mod, "run", _run)
        res = a.cancel(verbose=False, rule_classes=("run_", "simulation_member_"))
        signatures[name] = (res.get("jobs_were_running"), res["session_canceled"], bool(res.get("errors")))
        if name != "unknown":
            # A raised probe is UNKNOWN whatever this arm chooses to DO about cancellation;
            # the two answering arms must both still reach the sim class.
            assert _scancels(argvs), f"{name}: the sim class was never scancelled: {argvs}"

    assert len(set(signatures.values())) == 3, (
        "the three probe outcomes do not produce three distinct reports; signatures "
        f"(jobs_were_running, session_canceled, errors) were {signatures}"
    )


# ---------------------------------------------------------------------------
# The liveness query inside the selective arm is a THIRD probe on the same arm,
# and it has the same shape: two ANSWERS (a set, possibly empty) and NOT-KNOWN
# (`None`). The fixtures below differ ONLY in which one it produces, while the
# tmux log carries one sim token that `_status/_submitted/` does NOT -- so the
# observed-set widening is the only route that token has into the cancel set,
# and "answered: nothing live" and "could not answer" become tellable apart
# only if the arm reports the difference.
# ---------------------------------------------------------------------------

OBSERVED_ONLY_TOK = "run_tritonswmm_evt-0"
OBSERVED_ONLY_JID = "9999"
SUBMITTED_TOK = "run_tritonswmm_evt-1"
SUBMITTED_JID = "111"
RUN_UUID = "11111111-2222-3333-4444-555555555555"

OBSERVED_LOG = (
    "rule run_tritonswmm:\n"
    f"    output: _status/c_{OBSERVED_ONLY_TOK}_complete.flag\n"
    "    jobid: 501\n"
    f"Job 501 has been submitted with SLURM jobid {OBSERVED_ONLY_JID} (log: x).\n"
)


def _widening_harness(adir, *, sim_complete_flag: bool = False):
    """A selective-cancel arm whose tmux log names ONE sim that `_submitted/` does not.

    `_tmux_slurm_run_uuids` answers a REAL uuid, so `_squeue_live_jobids` reaches
    `subprocess.run` and the third state is produced by the PRODUCTION query rather than
    simulated at the call site: a fixture that stubbed the query itself would green an arm
    that is wrong about the query's own contract.
    """
    from hhemt.analysis import TRITONSWMM_analysis

    sub = adir / "_status" / "_submitted"
    sub.mkdir(parents=True)
    (sub / f"{SUBMITTED_TOK}.json").write_text(json.dumps({"rule_token": SUBMITTED_TOK, "slurm_jobid": SUBMITTED_JID}))
    if sim_complete_flag:
        (sub.parent / f"c_{OBSERVED_ONLY_TOK}_complete.flag").write_text("")
    logs = adir / "logs"
    logs.mkdir()
    (logs / "tmux_session_20260101_000000.log").write_text(OBSERVED_LOG)

    a = TRITONSWMM_analysis.__new__(TRITONSWMM_analysis)
    a.analysis_paths = types.SimpleNamespace(analysis_dir=adir, analysis_log_directory=logs)
    a.log = types.SimpleNamespace(
        tmux_session_name=types.SimpleNamespace(get=lambda: "sess"),
        snakemake_pid=types.SimpleNamespace(get=lambda: None),
        workflow_canceled=types.SimpleNamespace(set=lambda _v: None),
        workflow_cancellation_time=types.SimpleNamespace(set=lambda _v: None),
    )
    a.cfg_analysis = types.SimpleNamespace(analysis_id="test_squeue_abstention")
    a._workflow_builder = types.SimpleNamespace(
        _get_snakemake_pid_from_tmux=lambda s: None,
        _tmux_slurm_run_uuids=lambda: (RUN_UUID,),
    )
    return a


def _squeue_fixture(argvs: list, *, live: str | None = None, absent: bool = False):
    """Records every argv; answers the liveness query with one of its three outcomes.

    `absent=True` reproduces the MEASURED condition (no `squeue` on PATH): a
    `FileNotFoundError` out of `subprocess.run`, which `_squeue_live_jobids` catches into
    `None`. `absent=False, live=None` is an ANSWER of "nothing live"; `live={jid}` is an
    answer with one live jobid.
    """
    import subprocess as _sp

    def _run(argv, *args, **kw):
        argvs.append(list(argv))
        if argv and argv[0] == "squeue":
            if absent:
                raise FileNotFoundError(2, "No such file or directory", "squeue")
            return types.SimpleNamespace(returncode=0, stdout="" if live is None else f"{live}\n", stderr="")
        if list(argv[:2]) == ["tmux", "has-session"]:
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    return _sp, _run


def test_an_unanswerable_squeue_abstains_from_the_widening_and_reports_it(tmp_path, monkeypatch):
    """Property: when the selective arm's liveness query cannot answer, the observed-sim
    widening is ABSTAINED -- the `_submitted/` matches are still scancelled -- and the
    abstention is REPORTED in `errors` and in `message`, so the result is distinguishable
    from an answering query that found nothing live, which cancels the SAME set and reports
    no abstention.

    Class: REGRESSION -- red on the pre-change tree, which read
    `_squeue_live_jobids(...) or set()`. On the unanswerable path every tmux-log-observed
    sim was then filtered by `_jid not in _live`, so the whole observed set was dropped and
    the arm reported a small or zero count accurately while sims it never saw stayed queued.
    The pre-fix run is pasted in the build return.

    Why it is required -- checkable against the code alone: `_squeue_live_jobids` returns
    `None` for a THIRD state and its docstring binds callers to read it as NOT-KNOWN, never
    as "nothing live"; the other caller honours that. Coercing it here turns a non-answer
    into an answer and makes the two outcomes INDISTINGUISHABLE at the one surface an
    operator reads.

    What kills it: restoring `or set()` (the abstention disappears and the two arms collapse
    onto one report), or abstaining silently (the cancelled set is right and `errors` is
    empty, which is the same under-report one channel over).

    Second correct implementation, differing in the thing the assertion turns on: any wording
    passes -- the assertions are (i) the scancel SET, (ii) `errors` non-empty on the
    unanswerable arm and empty on the answering arm, and (iii) the two rendered `message`
    values differing. None of them reads a phrase.

    Differently-positioned satisfying input, and it is the POSITIVE CONTROL: the same fixture
    with the query answering `{OBSERVED_ONLY_JID}` must ADD that token to the cancel set.
    Without it, "the token was not cancelled" would be consistent with a fixture whose log the
    parser never read, and the abstention assertions would hold vacuously.

    Reachability: a cluster login node with no `squeue` module loaded produces exactly
    `FileNotFoundError` from a bare `["squeue", ...]` argv, which is the state driven here
    through `subprocess.run` -- the same measured condition the liveness gate's own absent-
    squeue test uses.
    """
    reports: dict[str, dict] = {}
    cancelled_sets: dict[str, set] = {}
    for name, kwargs in [
        ("unanswerable", {"absent": True}),
        ("answered_nothing_live", {"live": None}),
        ("answered_one_live", {"live": OBSERVED_ONLY_JID}),
    ]:
        argvs: list[list[str]] = []
        a = _widening_harness(tmp_path / name)
        _sp_mod, _run = _squeue_fixture(argvs, **kwargs)
        monkeypatch.setattr(_sp_mod, "run", _run)
        reports[name] = a.cancel(verbose=False, rule_classes=("run_", "simulation_member_"))
        scancels = _scancels(argvs)
        assert len(scancels) == 1, f"{name}: expected one scancel, got {argvs}"
        cancelled_sets[name] = set(scancels[0][1:])

    # The `_submitted/` match survives on EVERY arm -- the abstention withholds the widening,
    # never the worker-written set that does not depend on this query.
    for name, got in cancelled_sets.items():
        assert SUBMITTED_JID in got, f"{name}: dropped the _submitted/ match: {cancelled_sets}"

    # NOT-KNOWN: abstain from the widening, and say so in BOTH channels.
    assert OBSERVED_ONLY_JID not in cancelled_sets["unanswerable"], (
        f"an unanswerable query still widened the observed set: {cancelled_sets['unanswerable']}"
    )
    assert reports["unanswerable"]["errors"], f"the abstention was not reported at all: {reports['unanswerable']}"
    assert any("squeue" in _e for _e in reports["unanswerable"]["errors"]), (
        f"the reported error does not name the query that could not answer: {reports['unanswerable']}"
    )
    assert "squeue" in reports["unanswerable"]["message"], (
        f"the message does not carry the abstention: {reports['unanswerable']['message']!r}"
    )

    # ANSWERED, nothing live: same cancelled set, and NO abstention. This is the arm the
    # defect made indistinguishable from the one above.
    assert cancelled_sets["answered_nothing_live"] == cancelled_sets["unanswerable"], (
        "the answering arm cancelled a different set, so the discrimination below is not "
        f"about the report: {cancelled_sets}"
    )
    assert not reports["answered_nothing_live"]["errors"], (
        f"an answering query reported an error: {reports['answered_nothing_live']}"
    )
    assert reports["answered_nothing_live"]["message"] != reports["unanswerable"]["message"], (
        "an unanswerable query and an answering one that found nothing live render the SAME "
        f"message: {reports['unanswerable']['message']!r}"
    )

    # POSITIVE CONTROL -- the observed token IS reachable through the widening, so its
    # absence on the abstention arm is the abstention and not an unparsed fixture.
    assert OBSERVED_ONLY_JID in cancelled_sets["answered_one_live"], (
        "the observed-only token never entered the cancel set even with the query answering "
        f"it live, so the fixture's log is not being read: {cancelled_sets['answered_one_live']}"
    )
    assert not reports["answered_one_live"]["errors"], (
        f"a fully answering query reported an error: {reports['answered_one_live']}"
    )


def test_the_sim_complete_skip_still_applies_to_a_widened_token(tmp_path, monkeypatch):
    """Property: a widened `run_` token whose `c_{token}_complete.flag` exists is SKIPPED --
    its sim is done and the same job is now reclaiming -- both when the liveness query answers
    and when it cannot, because the flag needs no `squeue`.

    Class: REGRESSION GUARD -- the skip exists on the pre-change tree and R3's abstention must
    not remove it. The absent-flag arm is the differentially-positioned counterpart.

    Why it is required -- checkable against the code alone: scancelling a reclaiming job costs
    the processing the campaign is waiting on, so the skip is load-bearing on exactly the arm
    the widening exists for. An abstention implemented by bypassing the flag test "because we
    cannot filter anyway" would reintroduce that cost on the unanswerable path.

    What kills it: deleting the flag test from the widening loop, or hoisting the abstention
    ABOVE it so a flag-bearing token is withheld (and reported as possibly-queued) instead of
    resolved.

    Second correct implementation, differing in the thing the assertion turns on: any ordering
    of the loop's three decisions that leaves the flag's verdict the same on both query states
    passes -- the assertions are the scancel set and `errors`, never a statement order.

    Differently-positioned satisfying input: the same fixture WITHOUT the flag, on the
    answering arm, must scancel the token -- which proves the flag is what excluded it rather
    than some other filter.
    """
    # (a) answered, flag present: skipped.
    argvs_a: list[list[str]] = []
    a = _widening_harness(tmp_path / "flag_answered", sim_complete_flag=True)
    _sp_a, _run_a = _squeue_fixture(argvs_a, live=OBSERVED_ONLY_JID)
    monkeypatch.setattr(_sp_a, "run", _run_a)
    a.cancel(verbose=False, rule_classes=("run_", "simulation_member_"))
    got_a = set(_scancels(argvs_a)[0][1:])
    assert OBSERVED_ONLY_JID not in got_a, (
        f"scancelled a sim whose c_*_complete flag exists (reclaiming in flight): {got_a}"
    )

    # (b) unanswerable, flag present: the flag resolves it without squeue, so it is neither
    # scancelled nor reported as withheld.
    argvs_b: list[list[str]] = []
    b = _widening_harness(tmp_path / "flag_unanswerable", sim_complete_flag=True)
    _sp_b, _run_b = _squeue_fixture(argvs_b, absent=True)
    monkeypatch.setattr(_sp_b, "run", _run_b)
    res_b = b.cancel(verbose=False, rule_classes=("run_", "simulation_member_"))
    got_b = set(_scancels(argvs_b)[0][1:])
    assert OBSERVED_ONLY_JID not in got_b, f"scancelled a flag-complete sim while abstaining: {got_b}"
    assert not res_b["errors"], f"a flag-complete token was reported as withheld, but its flag needs no squeue: {res_b}"

    # (c) answered, NO flag: widened and scancelled -- the positive control for (a).
    argvs_c: list[list[str]] = []
    c = _widening_harness(tmp_path / "flag_absent")
    _sp_c, _run_c = _squeue_fixture(argvs_c, live=OBSERVED_ONLY_JID)
    monkeypatch.setattr(_sp_c, "run", _run_c)
    c.cancel(verbose=False, rule_classes=("run_", "simulation_member_"))
    got_c = set(_scancels(argvs_c)[0][1:])
    assert OBSERVED_ONLY_JID in got_c, (
        f"the flag-absent arm never widened, so (a) is not evidence about the flag: {got_c}"
    )


def test_the_gone_driver_transcript_does_not_claim_a_kill_or_a_completed_workflow(tmp_path, monkeypatch, capsys):
    """Property: on the GONE+selective path the CONSOLE TRANSCRIPT makes neither of the two
    claims the returned mapping no longer makes -- that a session was killed, and that the
    workflow had already completed -- while the ALIVE transcript still says the session was
    killed, because there it is true.

    Class: REGRESSION -- red on the pre-fix tree, whose GONE print read "(workflow already
    completed)" and whose `_selective_cancel` print tail read "driver session killed AFTER
    scancel" on both arms.

    Why it is required -- checkable against the code alone: the mapping and the console are
    two channels carrying the same claim, and this round exists because one of them was
    corrected while the other kept asserting the opposite. An operator reads the console;
    a consumer reads the mapping. A defect that survives in either channel is the same
    defect, one reader over.

    What kills it: reverting either print to its single-sentence form, or making the driver
    clause state-INdependent. The second is why the ALIVE control is asserted -- a body that
    simply deleted the clause would drop the kill report from the arm where a kill really
    happened, and that is the different defect this test also refuses.

    Second correct implementation, differing in the thing the assertion turns on: any
    wording pass, as long as the GONE transcript carries neither false claim and the ALIVE
    transcript carries the kill report -- the assertions read presence and absence of the
    two claims, never a preferred phrasing.

    Differently-positioned satisfying input, and it is the POSITIVE CONTROL: the ALIVE arm
    on the same fixture, which must still report the kill.

    Invocation not isolatable from this tree -- discrimination shown instead: restoring the
    unconditional driver clause re-adds "driver session killed" to the GONE transcript and
    re-reds this test.
    """
    gone = _cancel_harness(tmp_path / "gone_verbose")
    gone_argvs: list[list[str]] = []
    _sp, _run = _probe_fixture(gone_argvs, has_session_rc=1)
    monkeypatch.setattr(_sp, "run", _run)
    gone.cancel(verbose=True, rule_classes=("run_", "simulation_member_"))
    gone_out = capsys.readouterr().out

    alive = _cancel_harness(tmp_path / "alive_verbose")
    alive_argvs: list[list[str]] = []
    _sp2, _run2 = _probe_fixture(alive_argvs)
    monkeypatch.setattr(_sp2, "run", _run2)
    alive.cancel(verbose=True, rule_classes=("run_", "simulation_member_"))
    alive_out = capsys.readouterr().out

    assert "workflow already completed" not in gone_out, gone_out
    assert "driver session killed" not in gone_out, gone_out
    # POSITIVE CONTROL: the kill report belongs to the arm where a kill happened.
    assert "driver session killed" in alive_out, alive_out


def test_the_full_cancel_distinguishes_an_unanswerable_probe_from_a_gone_session(tmp_path, monkeypatch):
    """Property: on the FULL cancel (`rule_classes=None`, the arm that has no selective arm
    and therefore nothing to cancel either way) an unanswerable probe returns
    `jobs_were_running` that is not False, while a GONE probe returns the documented False --
    so the two are distinguishable on that arm too.

    Class: NEW CAPABILITY -- the fourth arm. The three sibling tests all pass
    `rule_classes`, so nothing exercised UNKNOWN or GONE on the arm where the probe result
    decides the whole return.

    Why it is required -- checkable against the code alone: UNKNOWN's contract is that no
    absence claim is made from an unanswered probe. That contract is a property of the probe,
    not of the arm the probe is reached from, so an arm where it does not hold is a hole in
    it -- and the full cancel is the arm a caller reaches with no arguments at all.

    What kills it: routing the raise into the GONE return on this arm, which is the natural
    one-line collapse because the two branches sit four lines apart.

    Second correct implementation, differing in the thing the assertion turns on: any body
    whose UNKNOWN arm omits the key or returns None passes; the assertions never name a key
    that must exist, only one that must not be False.

    Reachability: same fixture, same call, differing only in the probe outcome.

    Invocation not isolatable from this tree -- discrimination shown instead: collapsing the
    UNKNOWN arm onto the GONE return re-reds the first assertion.

    Residual, named: this pins the DISTINCTION and does not ask the full-cancel GONE arm to
    stop reporting False. That arm still answers a question about the sims with a fact about
    the session, and it is recorded as a per-package erratum rather than fixed here, because
    changing a public return on that path is a shape decision.
    """
    unknown = _cancel_harness(tmp_path / "full_unknown")
    unknown_argvs: list[list[str]] = []
    _sp, _run = _probe_fixture(unknown_argvs, raise_on_probe=True)
    monkeypatch.setattr(_sp, "run", _run)
    u_res = unknown.cancel(verbose=False)

    gone = _cancel_harness(tmp_path / "full_gone")
    gone_argvs: list[list[str]] = []
    _sp2, _run2 = _probe_fixture(gone_argvs, has_session_rc=1)
    monkeypatch.setattr(_sp2, "run", _run2)
    g_res = gone.cancel(verbose=False)

    assert u_res.get("jobs_were_running") is not False, (
        f"the full cancel reported a determined absence from an unanswerable probe: {u_res}"
    )
    assert u_res.get("errors"), f"the failed probe was not reported on the full cancel: {u_res}"
    assert g_res.get("jobs_were_running") is False, (
        f"the GONE full-cancel arm changed its documented no-op report: {g_res}"
    )
    assert not g_res.get("errors"), f"a gone session was reported as an error: {g_res}"


# `tmux has-session` answers "is the DRIVER's session still there", and R4's three-state
# contract is pinned by the tests above. The tests below cover its COMPLEMENT: every OTHER
# `subprocess.run` site in `_selective_cancel` and `cancel()` is an ACTION (or a poll), and
# a raise from any of them is a NON-ANSWER too -- no evidence the action took effect and
# none that it did not. Each site gets its own test because the reaction differs per site,
# and a single test over the set would let one site's arm cover another's absence.
#
# Two sites are deliberately absent. The reachability probe is 2a.1's and is pinned by
# `test_an_unanswerable_reachability_probe_never_claims_no_jobs_were_running`; the `sacct`
# query is the one site whose existing `try` is WIDENED rather than wrapped, which is why
# its test raises an exception the old `(TimeoutExpired, FileNotFoundError)` tuple did not
# catch -- a `FileNotFoundError` there was already handled and would not be red pre-fix.


def _raise_fixture(
    argvs: list,
    *,
    raise_on: str | None = None,
    exc: type[BaseException] = FileNotFoundError,
    has_session_rc: int = 0,
    ps_rc: int = 1,
):
    """Records every argv; each external call ANSWERS except the one named in `raise_on`.

    `raise_on` is one of "tmux has-session", "tmux kill-session", "scancel", "ps", "sacct",
    or None. The key is the CALL's identity, so a fixture varying only `raise_on` varies
    only WHICH call fails -- which is what makes the sites comparable to one another.

    `ps_rc` defaults to 1 so the Step 3 wait loop exits on its first poll. A 0 there would
    spin the loop for `wait_timeout` seconds of real time, since the loop is bounded by the
    wall clock rather than by the poll count.
    """
    import subprocess as _sp

    def _run(argv, *a, **kw):
        argvs.append(list(argv))
        first = argv[0] if argv else ""
        if first == "tmux" and len(argv) > 1:
            key = f"tmux {argv[1]}"
        elif first in ("scancel", "ps", "sacct"):
            key = first
        else:
            key = None
        if key is not None and key == raise_on:
            raise exc(2, "synthetic: the call could not run", first)
        if key == "tmux has-session":
            return types.SimpleNamespace(returncode=has_session_rc, stdout="", stderr="")
        if key == "ps":
            return types.SimpleNamespace(returncode=ps_rc, stdout="", stderr="")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    return _sp, _run


def _full_cancel_harness(adir, *, pid: int = 987654):
    """`_cancel_harness` with a RESOLVABLE Snakemake PID, so the full arm runs past Step 1.

    Without one the full cancel returns at its PID-not-found branch and never reaches the
    `ps` poll, the `sacct` query or Step 5's kill.
    """
    a = _cancel_harness(adir)
    a.log.snakemake_pid = types.SimpleNamespace(get=lambda: pid)
    a._workflow_builder = types.SimpleNamespace(
        _get_snakemake_pid_from_tmux=lambda s: pid,
        _tmux_slurm_run_uuids=lambda: ("run-uuid-abcd",),
    )
    return a


def _no_signal_no_sleep(monkeypatch):
    """Neutralise the two real-world effects the full-cancel tests would otherwise have.

    `os.kill` is REPLACED rather than avoided, because the full arm signals its resolved
    PID and a fixture that let it through would send SIGINT to a live process. `time.sleep`
    is a no-op for the 2-second poll interval ALONE; every other duration is delegated to
    the real function so nothing else in the test run is altered.
    """
    import time as _time

    monkeypatch.setattr(os, "kill", lambda _pid, _sig: None)
    _real_sleep = _time.sleep
    monkeypatch.setattr(_time, "sleep", lambda s: None if s == 2 else _real_sleep(s))


def test_a_scancel_that_cannot_run_is_reported_instead_of_escaping_cancel(tmp_path, monkeypatch):
    """Property: when `scancel` RAISES on the selective arm, `cancel()` RETURNS a mapping
    that records the failure and claims no cancellation, instead of letting the raise escape.

    Class: REGRESSION -- red on the pre-change tree, whose bare `subprocess.run` at the
    scancel site propagated `FileNotFoundError` out of `cancel()` (pre-fix output pasted in
    the build return).

    Why it is required -- checkable against the code alone: `scancel` is an ACTION, and a
    raise from it is no evidence that the jobs were cancelled and none that they survived.
    The pre-change code answered that non-answer one frame up with `workers_canceled: True`
    and `jobs_were_running: True`, so an operator was told the sim class was cancelled by a
    call that never ran.

    What kills it: catching the raise and leaving the success-path mapping intact (the arm
    would then report both fields True with an empty `errors`), or letting the raise through.

    Second correct implementation, differing in the thing the assertion turns on: reporting
    `jobs_were_running: False` instead of `None` still passes THIS test, because its
    assertion is `is not True` -- any answer that does not claim the scancel reached the
    jobs, and omitting the key, are both admissible here. That is this test's scope and
    not the field's contract: the `None` pin lives in
    `test_a_scancel_that_cannot_run_never_reports_that_no_jobs_were_running` and its GONE-
    arm sibling, which is where the mutant this docstring used to tolerate is killed.

    Differently-positioned satisfying input, and it is the POSITIVE CONTROL: the same
    harness with an ANSWERING scancel, which must carry no error, report success, and report
    `jobs_were_running: True`.

    Reachability: an Lmod cluster with no `slurm` client loaded raises `FileNotFoundError`
    from a bare `["scancel", ...]` argv rather than returning non-zero.

    Invocation not isolatable from this tree -- discrimination shown instead: restoring the
    bare call re-raises inside `cancel()` and reds the first assertion, which the pre-fix
    run records.
    """
    argvs: list[list[str]] = []
    a = _cancel_harness(tmp_path / "scancel_raises")
    _sp, _run = _raise_fixture(argvs, raise_on="scancel")
    monkeypatch.setattr(_sp, "run", _run)

    res = a.cancel(verbose=False, rule_classes=("run_", "simulation_member_"))

    joined = " | ".join(res.get("errors") or [])
    assert "scancel" in joined, f"the failed scancel was not reported: {res}"
    assert res["workers_canceled"] is False, f"claimed a cancellation from a scancel that never ran: {res}"
    assert res.get("jobs_were_running") is not True, f"claimed the sim class was reached: {res}"
    assert res["success"] is False, f"reported success with a failed external call: {res}"

    # POSITIVE CONTROL -- the same fixture, scancel answering, so the assertions above
    # cannot hold merely because the selective arm was never reached.
    ctl_argvs: list[list[str]] = []
    b = _cancel_harness(tmp_path / "scancel_answers")
    _sp2, _run2 = _raise_fixture(ctl_argvs)
    monkeypatch.setattr(_sp2, "run", _run2)
    ctl = b.cancel(verbose=False, rule_classes=("run_", "simulation_member_"))

    assert [v for v in ctl_argvs if v and v[0] == "scancel"], f"control arm emitted no scancel: {ctl_argvs}"
    assert not ctl.get("errors"), f"control arm reported an error on an answering scancel: {ctl}"
    assert ctl["jobs_were_running"] is True, ctl
    assert ctl["success"] is True, ctl


def test_a_driver_kill_that_cannot_run_is_reported_instead_of_escaping_cancel(tmp_path, monkeypatch):
    """Property: when `tmux kill-session` RAISES on the selective arm, `cancel()` RETURNS a
    mapping that records the failure, still reports the scancel it DID perform, and claims
    no session was killed.

    Class: REGRESSION -- red on the pre-change tree, whose second bare `subprocess.run` in
    `_selective_cancel` propagated the raise out of `cancel()`.

    Why it is required -- checkable against the code alone: the two calls in this function
    are independent actions, so one failing says nothing about the other. The pre-change
    mapping reported `session_canceled: True` on this path, which asserts a kill the raise
    proves did not run.

    What kills it: catching the raise and keeping `session_canceled: True`; or suppressing
    the scancel as well, which would make one failure swallow an action that did happen.

    Second correct implementation, differing in the thing the assertion turns on: any body
    that reports the kill failure and a False `session_canceled` passes -- the assertions
    read the scancel argv (through the fixture) and two mapping fields, never a sentence.

    Differently-positioned satisfying input, and it is the POSITIVE CONTROL: the same
    harness with an ANSWERING kill, which must report `session_canceled: True` and success.

    Reachability: an unloaded `tmux` module on an Lmod cluster raises `FileNotFoundError`
    from a bare `["tmux", ...]` argv -- the same measured mechanism as the probe's.

    Invocation not isolatable from this tree -- discrimination shown instead: restoring the
    bare call re-raises and reds the `session_canceled` assertion.
    """
    argvs: list[list[str]] = []
    a = _cancel_harness(tmp_path / "kill_raises")
    _sp, _run = _raise_fixture(argvs, raise_on="tmux kill-session")
    monkeypatch.setattr(_sp, "run", _run)

    res = a.cancel(verbose=False, rule_classes=("run_", "simulation_member_"))

    joined = " | ".join(res.get("errors") or [])
    assert "kill-session" in joined, f"the failed kill was not reported: {res}"
    assert res["session_canceled"] is False, f"claimed a session was killed by a kill that never ran: {res}"
    assert res["success"] is False, f"reported success with a failed external call: {res}"
    # The scancel ran and answered, so its facts must survive the other call's failure.
    assert [v for v in argvs if v and v[0] == "scancel"], f"the failure suppressed the scancel: {argvs}"
    assert res["workers_canceled"] is True, f"lost the scancel facts when the kill failed: {res}"

    # POSITIVE CONTROL -- the same fixture, the kill answering.
    ctl_argvs: list[list[str]] = []
    b = _cancel_harness(tmp_path / "kill_answers")
    _sp2, _run2 = _raise_fixture(ctl_argvs)
    monkeypatch.setattr(_sp2, "run", _run2)
    ctl = b.cancel(verbose=False, rule_classes=("run_", "simulation_member_"))

    assert ctl["session_canceled"] is True, ctl
    assert ctl["success"] is True, ctl


def test_the_pid_not_found_kill_reports_its_failure_instead_of_escaping(tmp_path, monkeypatch):
    """Property: on the FULL cancel's PID-not-found arm, a raising `tmux kill-session`
    produces a returned mapping that records both the missing PID and the failed kill, and
    that does NOT report the session as cancelled.

    Class: REGRESSION -- red on the pre-change tree, whose bare kill at that site propagated
    `FileNotFoundError` out of `cancel()`.

    Why it is required -- checkable against the code alone: this arm kills the session as
    its LAST resort, so a raise here leaves the driver running with the returned mapping
    saying `session_canceled: True`. An operator reading it would believe the workflow was
    stopped while its driver was untouched.

    What kills it: catching the raise and returning the pre-change mapping (`success` and
    `session_canceled` both True), or dropping the pre-existing PID error entry while adding
    the kill entry.

    Second correct implementation, differing in the thing the assertion turns on: any body
    reporting `session_canceled: False` with both errors present passes -- the assertions
    read the two error substrings and the field, never a message phrasing.

    Differently-positioned satisfying input, and it is the POSITIVE CONTROL: the same
    harness with an ANSWERING kill, which must keep `success` and `session_canceled` True
    with the PID entry still recorded -- that entry is documented behaviour, not an error
    this change may drop.

    Reachability: an unloaded `tmux` module raises `FileNotFoundError` from the bare argv.

    Invocation not isolatable from this tree -- discrimination shown instead: restoring the
    bare call re-raises and reds the first assertion.
    """
    argvs: list[list[str]] = []
    a = _cancel_harness(tmp_path / "nopid_kill_raises")
    _sp, _run = _raise_fixture(argvs, raise_on="tmux kill-session")
    monkeypatch.setattr(_sp, "run", _run)

    res = a.cancel(verbose=False)

    joined = " | ".join(res.get("errors") or [])
    assert "kill-session" in joined, f"the failed kill was not reported: {res}"
    assert "Could not find Snakemake PID" in joined, f"the pre-existing PID error was dropped: {res}"
    assert res["session_canceled"] is False, f"claimed a session was killed by a kill that never ran: {res}"
    assert res["success"] is False, f"reported success with a failed external call: {res}"

    # POSITIVE CONTROL -- the same arm with the kill answering: the documented no-op-PID
    # report is `success`/`session_canceled` True WITH the PID entry recorded.
    ctl_argvs: list[list[str]] = []
    b = _cancel_harness(tmp_path / "nopid_kill_answers")
    _sp2, _run2 = _raise_fixture(ctl_argvs)
    monkeypatch.setattr(_sp2, "run", _run2)
    ctl = b.cancel(verbose=False)

    assert ctl["session_canceled"] is True, ctl
    assert ctl["success"] is True, ctl
    assert "Could not find Snakemake PID" in " | ".join(ctl.get("errors") or []), ctl


def test_an_unanswerable_ps_poll_never_claims_the_snakemake_process_exited(tmp_path, monkeypatch, capsys):
    """Property: when `ps` RAISES inside the Step 3 wait loop, `cancel()` RETURNS with the
    failure recorded, `success` False, and NO claim that the process did or did not exit --
    neither the console's "Snakemake process exited" nor a "did not exit within Ns" timeout
    message for a wait the loop never reached.

    Class: REGRESSION -- red on the pre-change tree, whose loop body called `subprocess.run`
    with no enclosing try at all, so the raise escaped `cancel()`.

    Why it is required -- checkable against the code alone: `ps` is a POLL, and a poll that
    cannot run answers nothing about the process. The mapping carries no exit field, so the
    claim surfaces only in the transcript -- and the timeout message is computed from the
    loop having run to its end, so emitting it after an early stop asserts a wait that never
    happened. Both are non-answers converted into positive findings.

    What kills it: catching the raise and setting `process_exited = True` (which prints
    "Snakemake process exited"), or catching it and letting the timeout branch run (which
    claims a timeout). The console assertion is what catches the first -- a mapping-only
    assertion would pass it.

    Second correct implementation, differing in the thing the assertion turns on: breaking
    the loop on a separate flag, continuing the loop to the timeout, or returning early are
    all admissible as long as neither sentence appears and the error is present.

    Differently-positioned satisfying input, and it is the POSITIVE CONTROL: the same
    fixture with `ps` answering "gone" (`ps_rc=1`), which must reach a clean
    "Workflow canceled" with no ps error -- proving the loop ran and the ps site was
    reached, not merely skipped.

    Reachability: a minimal container image with no procps installed raises
    `FileNotFoundError` from a bare `["ps", ...]` argv.

    Invocation not isolatable from this tree -- discrimination shown instead: restoring the
    bare call re-raises and reds the first assertion.
    """
    argvs: list[list[str]] = []
    a = _full_cancel_harness(tmp_path / "ps_raises")
    _no_signal_no_sleep(monkeypatch)
    _sp, _run = _raise_fixture(argvs, raise_on="ps")
    monkeypatch.setattr(_sp, "run", _run)

    res = a.cancel(verbose=True)
    out = capsys.readouterr().out

    joined = " | ".join(res.get("errors") or [])
    assert "ps -p" in joined, f"the failed poll was not reported: {res}"
    assert "Snakemake process exited" not in out, f"claimed the process exited from a poll that never ran: {out}"
    assert "did not exit within" not in res["message"], f"claimed a timeout the loop never reached: {res['message']!r}"
    assert "did not exit within" not in out, f"claimed a timeout the loop never reached: {out}"
    assert res["success"] is False, f"reported success with a failed external call: {res}"

    # POSITIVE CONTROL -- the same fixture with the poll answering "gone".
    ctl_argvs: list[list[str]] = []
    b = _full_cancel_harness(tmp_path / "ps_answers")
    _no_signal_no_sleep(monkeypatch)
    _sp2, _run2 = _raise_fixture(ctl_argvs)
    monkeypatch.setattr(_sp2, "run", _run2)
    ctl = b.cancel(verbose=False)

    assert [v for v in ctl_argvs if v[:2] == ["ps", "-p"]], f"control arm never polled: {ctl_argvs}"
    assert "Workflow canceled" in ctl["message"], f"control arm did not reach the clean path: {ctl}"
    assert ctl["success"] is True, ctl


def test_an_unanswerable_sacct_reports_the_workers_unverified_instead_of_escaping(tmp_path, monkeypatch, capsys):
    """Property: an exception the narrow `(TimeoutExpired, FileNotFoundError)` tuple did NOT
    catch, raised by the `sacct` query, is absorbed by the widened general arm: the workers
    are reported UNVERIFIED, `cancel()` returns, and the exception's identity reaches the
    operator on the console WITHOUT rewriting the published `errors` entry.

    Class: REGRESSION -- red on the pre-change tree, whose `except (subprocess.TimeoutExpired,
    FileNotFoundError)` let a `PermissionError` escape `cancel()`.

    Why it is required -- checkable against the code alone: every raise from this block has
    the same consequence -- the worker set was not asked, so the workers are unverified --
    and the conclusion must not depend on which exception class arrived. A narrow tuple
    made the report depend on the exception's type, and let the rest escape into
    workflow.py's broad handler, which replaced the whole cancel mapping.

    What kills it: restoring the narrow tuple (the raise escapes and reds the first
    assertion), or folding the exception detail into `errors` -- which reds the
    `PermissionError not in joined` assertion and, one file over,
    `test_cancel_worker_check.py`'s exact-membership contract.

    Second correct implementation, differing in the thing the assertion turns on: printing
    the detail in any wording passes -- the console assertion reads the exception CLASS
    name, and the list assertion reads the entry's exact published text, so neither turns on
    a preferred phrasing.

    Differently-positioned satisfying input, and it is the POSITIVE CONTROL: the same
    fixture with `sacct` answering, which must report `workers_canceled: True` with no
    unverified entry -- proving the sacct site was reached and the widened arm did not fire
    on a healthy call.

    Reachability: `sacct` present but not executable (a partially-provisioned Lmod module)
    raises `PermissionError` from `subprocess.run`, which is neither `TimeoutExpired` nor
    `FileNotFoundError`.

    Invocation not isolatable from this tree -- discrimination shown instead: narrowing the
    except tuple back to the two original classes re-raises and reds the first assertion.
    """
    argvs: list[list[str]] = []
    a = _full_cancel_harness(tmp_path / "sacct_raises")
    _no_signal_no_sleep(monkeypatch)
    _sp, _run = _raise_fixture(argvs, raise_on="sacct", exc=PermissionError)
    monkeypatch.setattr(_sp, "run", _run)

    res = a.cancel(verbose=True)
    out = capsys.readouterr().out

    joined = " | ".join(res.get("errors") or [])
    assert "Could not verify worker job status" in joined, f"the unanswerable query was not reported: {res}"
    # The ENTRY text is a published contract (tests/test_cancel_worker_check.py pins this
    # string in `errors`), so the exception's identity must ride the console, not the list.
    assert "PermissionError" in out, f"the exception class never reached the operator: {out}"
    assert "PermissionError" not in joined, f"the published error entry was rewritten to carry a detail: {res}"
    assert res["workers_canceled"] is False, f"claimed verification from a query that never ran: {res}"
    assert res["success"] is False, f"reported success with a failed external call: {res}"
    assert [v for v in argvs if v and v[0] == "sacct"], f"the sacct site was never reached: {argvs}"

    # POSITIVE CONTROL -- the same fixture with the query answering.
    ctl_argvs: list[list[str]] = []
    b = _full_cancel_harness(tmp_path / "sacct_answers")
    _no_signal_no_sleep(monkeypatch)
    _sp2, _run2 = _raise_fixture(ctl_argvs)
    monkeypatch.setattr(_sp2, "run", _run2)
    ctl = b.cancel(verbose=False)

    ctl_joined = " | ".join(ctl.get("errors") or [])
    assert "Could not verify worker job status" not in ctl_joined, f"the widened arm fired on a healthy query: {ctl}"
    assert ctl["workers_canceled"] is True, ctl
    assert ctl["success"] is True, ctl


def test_a_final_kill_that_cannot_run_is_reported_instead_of_escaping(tmp_path, monkeypatch):
    """Property: when Step 5's `tmux kill-session` RAISES, `cancel()` RETURNS a mapping that
    records the failure, reports `success` False, and never claims the session was
    terminated.

    Class: REGRESSION -- red on the pre-change tree, whose Step 5 read
    `kill_result.returncode` off a call that had raised, so the raise escaped `cancel()` and
    the whole cancel report was replaced by the broad handler's generic message.

    Why it is required -- checkable against the code alone: this is the arm's LAST external
    call and its result decides whether the console may say "Tmux session terminated". A
    raise there means the kill did not run, which is the opposite of that sentence, and it
    is also not a non-zero RETURN -- the branch that reads `stderr` has nothing to read.

    What kills it: catching the raise and taking the `returncode == 0` branch, or assigning
    a fresh success-shaped result object so `success` stays True.

    Second correct implementation, differing in the thing the assertion turns on: any body
    that appends the failure to `errors` and branches on the error instead of a returncode
    passes -- `success` is computed from `len(errors) == 0` at the return, so the assertion
    reads that computed field rather than the branch shape.

    Differently-positioned satisfying input, and it is the POSITIVE CONTROL: the same
    fixture with the kill answering, which must report `success` True and must still reach
    the step (proved by the emitted kill argv).

    Reachability: an unloaded `tmux` module raises `FileNotFoundError` from the bare argv.

    Invocation not isolatable from this tree -- discrimination shown instead: restoring the
    bare call re-raises and reds the first assertion.
    """
    argvs: list[list[str]] = []
    a = _full_cancel_harness(tmp_path / "step5_kill_raises")
    _no_signal_no_sleep(monkeypatch)
    _sp, _run = _raise_fixture(argvs, raise_on="tmux kill-session")
    monkeypatch.setattr(_sp, "run", _run)

    res = a.cancel(verbose=False)

    joined = " | ".join(res.get("errors") or [])
    assert "kill-session" in joined, f"the failed final kill was not reported: {res}"
    assert res["success"] is False, f"reported success with a failed external call: {res}"
    assert "terminated" not in res["message"], f"claimed a termination that never ran: {res['message']!r}"

    # POSITIVE CONTROL -- the same fixture with the kill answering.
    ctl_argvs: list[list[str]] = []
    b = _full_cancel_harness(tmp_path / "step5_kill_answers")
    _no_signal_no_sleep(monkeypatch)
    _sp2, _run2 = _raise_fixture(ctl_argvs)
    monkeypatch.setattr(_sp2, "run", _run2)
    ctl = b.cancel(verbose=False)

    assert [v for v in ctl_argvs if v[:2] == ["tmux", "kill-session"]], f"control arm emitted no kill: {ctl_argvs}"
    assert ctl["success"] is True, ctl


# ── Reviewer finding, chunk 2a.2 ───────────────────────────────────────────────
# The six tests above stop at `cancel()`'s returned mapping. That mapping's only
# consumer is the DU guard at `src/hhemt/workflow.py:6780`, which reads
# `_res.get("cancelled")` and -- when the list is non-empty -- prints
# "Cancelled {n} simulation job(s)". `_selective_cancel` returns
# `"cancelled": _matched`, which is the set it SELECTED, not the set it
# scancelled. On the scancel-failure path nothing was scancelled, so the caller
# states a cancellation that never happened: the same untruth this chunk removes
# one level down, surviving at the one field the caller actually reads.


def test_a_scancel_that_cannot_run_does_not_hand_the_caller_a_cancelled_list(tmp_path, monkeypatch):
    """Property: when `scancel` RAISES on the selective arm, the mapping's `cancelled`
    list is EMPTY, because that field is documented as the pairs SCANCELLED and the
    one consumer reads a non-empty list as "N jobs cancelled".

    Class: NEW CAPABILITY -- the plausible wrong implementation is the tree as this
    chunk leaves it, which returns the SELECTED set `_matched` whether or not the
    scancel ran; red under it, output pasted in the round file.

    Why it is required -- checkable against this code alone: `_selective_cancel`'s
    docstring defines `cancelled` as "(the (token, jobid) pairs scancelled)", and
    `workflow.py:6780` branches `elif _n_cancelled:` into the sentence
    "Cancelled {len} simulation job(s)". Populating that list from the selection
    rather than from the scancel makes both statements untrue on the failure path.

    What kills it: returning `_matched` as `cancelled` when the scancel raised --
    which is what the tree does today.

    Second correct implementation, materially different from the obvious fix: the
    branches may be reordered so `cancelled` is assigned only inside the
    `_scancel_err is None` arm, or the key may be built before the calls and
    filled after; the assertion turns on the returned list, never on where or in
    which order it is assigned.

    Differently-positioned satisfying input, and it is the POSITIVE CONTROL: the
    same fixture with an ANSWERING scancel, whose `cancelled` list must be
    NON-empty -- without it, "empty" would be satisfied by an arm that never
    selected anything at all.

    Reachability: an Lmod cluster with no `slurm` client loaded raises
    `FileNotFoundError` from a bare `["scancel", ...]` argv; the pre-change red for
    that input class is pasted in the round file.

    Invocation not isolatable from this tree -- discrimination shown instead:
    giving `cancelled` an empty list on the scancel-failure arm greens this test,
    which is a change to what the code DOES on this input.
    """
    a = _cancel_harness(tmp_path / "cancelled_scancel_raises")
    argvs: list[list[str]] = []
    _sp, _run = _raise_fixture(argvs, raise_on="scancel")
    monkeypatch.setattr(_sp, "run", _run)

    res = a.cancel(verbose=False, rule_classes=("run_", "simulation_member_"))

    assert res.get("errors"), f"fixture never reached the scancel-failure path: {res}"
    assert res["cancelled"] == [], (
        "a scancel that never ran was handed to the caller as having cancelled "
        f"{res['cancelled']}; workflow.py:6780 reads this list as 'Cancelled N job(s)': {res}"
    )

    # POSITIVE CONTROL -- the scancel answering, so "empty" cannot be satisfied by
    # an arm that selected nothing.
    ctl_argvs: list[list[str]] = []
    b = _cancel_harness(tmp_path / "cancelled_scancel_answers")
    _sp2, _run2 = _raise_fixture(ctl_argvs)
    monkeypatch.setattr(_sp2, "run", _run2)

    ctl = b.cancel(verbose=False, rule_classes=("run_", "simulation_member_"))

    assert ctl["cancelled"], f"the answering arm selected nothing, so the control is vacuous: {ctl}"
    assert not ctl.get("errors"), ctl


# ── Coordinator dispatch, chunk 2a.2 round 2 — the surviving mutant ────────────
# Every test above pins `jobs_were_running` with `is not True`, or via the tri-state
# signature tuple. Neither distinguishes `None` from `False`, so an arm that reports
# `jobs_were_running: False` from a scancel that could not run passes all of them --
# measured: 20 passed with exactly that one-line substitution applied.
#
# `False` is not a milder version of the same untruth, it IS the untruth: `cancel()`'s
# returned-mapping contract reads False as "a determined absence of running jobs"
# (`jobs_were_running`: "False when no session is recorded or it no longer exists"),
# and a scancel that never ran determined nothing about the jobs it was handed. The
# field has NO production reader -- `grep -rn jobs_were_running --include=*.py src/
# tests/` returns only this module and `analysis.py`'s own mapping constructors and
# docstrings -- so the docstring's contract is defended by assertion or not at all.
#
# These are two tests and not one because the selective arm is dispatched from two
# call sites in `cancel()` and the shape requires both: the ALIVE path returns
# `_selective_cancel`'s mapping verbatim, while the GONE path REBUILDS it
# (`{**_selective_result, "session_canceled": False, "message": ...}`). An assertion
# taken only where the mapping is returned unchanged would not reach the caller one
# level out, where the dict is reconstructed.


def test_a_scancel_that_cannot_run_never_reports_that_no_jobs_were_running(tmp_path, monkeypatch):
    """Property: when `scancel` RAISES on the selective arm and the driver session is
    ALIVE, `cancel()`'s mapping returns `jobs_were_running` as `None` -- the contract's
    "not determined" -- and never `False`.

    Class: NEW CAPABILITY -- plausible wrong implementation is the tree as round 1 left
    it with `else None` -> `else False` at the `jobs_were_running` key, which passed the
    whole module (20 passed) before this test existed; red under it, output pasted in the
    round file. It is NOT regression-class: the pre-change tree (HEAD) has no such arm at
    all, so its behaviour on this input is an escaping raise, not the mutant.

    Both-states anchor: the assertion turns on the returned VALUE of a key that exists in
    both the current and the proposed code, never on wording either version introduces.

    Population: the claim ranges over "every raise from the scancel call on the selective
    arm"; the instrument reads one exception class (`FileNotFoundError`) on two dispatch
    arms, which is a strict subset of the claim's population. The assertion reads a value
    the arm computes from `_scancel_err`, which is exception-class-agnostic.

    Why it is required -- checkable against the code alone: the key's own contract, in
    `cancel()`'s docstring, reads `False` as a DETERMINED absence. On this input the
    scancel was handed two jobids and could not run, so nothing was determined either
    way; reporting `False` converts a non-answer into a positive finding one level out
    from where the chunk's other tests stop.

    What kills it: emitting `False` from the `_scancel_err is not None` arm (the named
    mutant), or dropping the key so that a caller's `.get(...)` default decides.
    `is None` is strict -- an omitted key fails it as well, which is deliberate: this
    key is part of the mapping's documented shape.

    Second correct implementation, differing in the thing the assertion turns on: any
    expression that yields `None` on the failure arm passes -- a conditional expression,
    an assignment inside the success arm only, or a sentinel resolved at the return --
    the assertion reads the returned value, never the branch shape.

    Differently-positioned satisfying input, and it is the POSITIVE CONTROL: the same
    harness with an ANSWERING scancel must report `jobs_were_running: True`. Without it,
    `None` would be satisfied by an arm that hard-codes `None` for this field on every
    input, or by one that selected nothing and reported `bool([])`.

    Reachability: an Lmod cluster with no `slurm` client loaded raises
    `FileNotFoundError` from a bare `["scancel", ...]` argv.

    Invocation not isolatable from this tree -- discrimination shown instead: rewriting
    the failure arm to `False` reds this test and greens every other one in the module.
    """
    argvs: list[list[str]] = []
    a = _cancel_harness(tmp_path / "jobs_running_scancel_raises")
    _sp, _run = _raise_fixture(argvs, raise_on="scancel")
    monkeypatch.setattr(_sp, "run", _run)

    res = a.cancel(verbose=False, rule_classes=("run_", "simulation_member_"))

    assert res.get("errors"), f"fixture never reached the scancel-failure path: {res}"
    assert res.get("jobs_were_running") is None, (
        "a scancel that never ran reported a DETERMINED absence of running jobs; "
        f"the contract reserves False for recorded-and-gone, so this must be None: {res}"
    )

    # POSITIVE CONTROL -- the scancel answering, so `None` cannot be satisfied by an arm
    # that never selected anything or that hard-codes `None` for this field.
    ctl_argvs: list[list[str]] = []
    b = _cancel_harness(tmp_path / "jobs_running_scancel_answers")
    _sp2, _run2 = _raise_fixture(ctl_argvs)
    monkeypatch.setattr(_sp2, "run", _run2)

    ctl = b.cancel(verbose=False, rule_classes=("run_", "simulation_member_"))

    assert _scancels(ctl_argvs), f"control arm never reached the scancel site: {ctl_argvs}"
    assert ctl.get("jobs_were_running") is True, f"control arm reported no jobs running: {ctl}"
    assert not ctl.get("errors"), ctl


def test_a_scancel_that_cannot_run_never_reports_no_jobs_were_running_on_the_gone_arm(tmp_path, monkeypatch):
    """Property: the SAME contract holds one level further out, on `cancel()`'s GONE-driver
    arm, where the mapping is REBUILT rather than returned -- a raising `scancel` there
    still yields `jobs_were_running: None`, never `False`.

    Class: NEW CAPABILITY -- red under the same one-line mutant as its sibling, and red
    under a second one the sibling cannot see: a rebuild that supplies `False` for this
    key while spreading the rest (`{**_selective_result, "jobs_were_running": False,
    ...}`); both reds are pasted in the round file. Not regression-class: HEAD has no
    such arm on this path, so its behaviour here is an escaping raise, not either mutant.

    Both-states anchor: the assertion turns on the returned VALUE of a key that exists in
    both the current and the proposed code, never on the rebuilt message wording.

    Population: the claim ranges over "every raise from the scancel call on the GONE
    dispatch arm"; the instrument reads `FileNotFoundError` there, a strict subset.

    Why it is required -- checkable against the code alone: `cancel()`'s GONE branch
    constructs a NEW dict from `_selective_result` to correct two fields the driver's
    absence falsifies. That reconstruction is a separate place the contract can be lost,
    and it is the caller one level out from `_selective_cancel` -- the level at which the
    sibling's assertion, taken where the mapping passes through unchanged, cannot speak.

    What kills it: hard-coding `False` on the GONE rebuild; emitting `False` from
    `_selective_cancel`'s failure arm; or dropping the key in the rebuild.

    Second correct implementation, differing in the thing the assertion turns on: the
    rebuild may leave the key untouched (spread), or set it explicitly from the same
    expression the success arm uses -- the assertion reads the returned value.

    Differently-positioned satisfying input, and it is the POSITIVE CONTROL: the same
    GONE arm with an ANSWERING scancel must report `jobs_were_running: True` and
    `session_canceled: False` -- proving the arm really did take the GONE path (the
    rebuild is why the message and this field differ from the ALIVE arm) and that the
    scancel site was reached.

    Reachability: identical to the sibling's -- an unloaded `slurm` client.

    Invocation not isolatable from this tree -- discrimination shown instead: supplying
    `False` in the GONE rebuild reds this test alone.
    """
    argvs: list[list[str]] = []
    a = _cancel_harness(tmp_path / "jobs_running_scancel_raises_gone")
    _sp, _run = _raise_fixture(argvs, raise_on="scancel", has_session_rc=1)
    monkeypatch.setattr(_sp, "run", _run)

    res = a.cancel(verbose=False, rule_classes=("run_", "simulation_member_"))

    assert res.get("errors"), f"fixture never reached the scancel-failure path: {res}"
    assert res["session_canceled"] is False, f"the GONE rebuild did not run: {res}"
    assert res.get("jobs_were_running") is None, (
        "the GONE rebuild reported a DETERMINED absence of running jobs from a scancel "
        f"that never ran; the contract reserves False for recorded-and-gone: {res}"
    )

    # POSITIVE CONTROL -- GONE arm, scancel answering: True AND the rebuild's own marker.
    ctl_argvs: list[list[str]] = []
    b = _cancel_harness(tmp_path / "jobs_running_scancel_answers_gone")
    _sp2, _run2 = _raise_fixture(ctl_argvs, has_session_rc=1)
    monkeypatch.setattr(_sp2, "run", _run2)

    ctl = b.cancel(verbose=False, rule_classes=("run_", "simulation_member_"))

    assert _scancels(ctl_argvs), f"control arm never reached the scancel site: {ctl_argvs}"
    assert ctl["session_canceled"] is False, f"control arm did not take the GONE path: {ctl}"
    assert ctl.get("jobs_were_running") is True, f"control arm reported no jobs running: {ctl}"


# ---------------------------------------------------------------------------
# _classify_live_sentinels — corroboration of a "not live" verdict, and the
# walltime age tier that bounds the hold it creates.
#
# These six tests are one mechanism with separable terms, and each isolates a
# different term: the corroborator's verdict (tests 1 and 3), the age tier that
# bounds an unconfirmable hold (tests 2 and 4), the precedence of a
# confirmed-live accounting row over that tier (test 5), and the zero-call no-op
# an empty sweep must preserve (test 6).
# ---------------------------------------------------------------------------


def _liveness_builder(tmp_path, *, total_job_min: int = 120, wait_min: int = 60):
    """Builder carrying only what ``_classify_live_sentinels`` reads.

    ``hpc_total_job_duration_min`` is what
    ``_max_plausible_job_lifetime_min`` bounds on, so with the module default
    slack of 30 the cap is ``(total_job_min + 30) * 60`` seconds. Setting it
    explicitly keeps every age assertion below derived from the same
    instrument the production code consults rather than from a literal.
    """
    analysis_dir = tmp_path / "analysis"
    (analysis_dir / "_status" / "_submitted").mkdir(parents=True)
    b = SnakemakeWorkflowBuilder.__new__(SnakemakeWorkflowBuilder)
    b.analysis_paths = types.SimpleNamespace(analysis_dir=analysis_dir)
    b.cfg_analysis = types.SimpleNamespace(
        hpc_total_job_duration_min=total_job_min,
        hpc_max_wait_for_inflight_min=wait_min,
    )
    return b


def _write_submitted_sentinel(builder, stem: str, jobid: str, *, age_s: float = 0.0):
    """Write a submitted-sentinel the way a worker does, optionally backdated.

    Backdating uses ``os.utime`` on the file itself because the age tier reads
    ``stat().st_mtime`` — the same instrument, not a proxy for it.
    """
    path = builder.analysis_paths.analysis_dir / "_status" / "_submitted" / f"{stem}.json"
    path.write_text(json.dumps({"slurm_jobid": jobid, "run_uuid": "u"}))
    if age_s:
        when = time.time() - age_s
        os.utime(path, (when, when))
    return path


def _stub_liveness(monkeypatch, *, squeue_live: bool = False, sacct_rows: dict):
    """Stub BOTH liveness instruments at the module the code under test reads.

    ``squeue_live`` is the probe's answer; ``sacct_rows`` is the corroborator's
    full reply, so ``{}`` is the unconfirmable case, a terminal row is a
    confirmed death and a non-terminal row is a confirmed live job.
    """
    monkeypatch.setattr("hhemt.workflow._slurm_job_is_live", lambda jid, **kw: squeue_live)
    monkeypatch.setattr("hhemt.workflow._sacct_states_batched", lambda job_ids, **kw: dict(sacct_rows))


def _cap_s(builder) -> float:
    """The age tier the production code uses, in seconds, from its own helper."""
    from hhemt.workflow import _max_plausible_job_lifetime_min

    return _max_plausible_job_lifetime_min(builder.cfg_analysis) * 60


def test_an_unconfirmable_death_verdict_holds_a_fresh_sentinel(tmp_path, monkeypatch):
    """Property: a "not live" squeue verdict that the accounting database cannot
    corroborate does NOT reclaim a fresh submitted-sentinel.

    Class: REGRESSION — red on the pre-change tree, where the same inputs
    unlink the file and return an empty alive set.

    Why it is required: ``squeue -j`` reporting nothing is equally what a job
    whose accounting record has aged out looks like. Reclaiming on that answer
    alone discards the only record naming a job that may still be running, and
    the next reconcile then submits a second copy of its work.

    What kills it: any shape where the corroboration runs but its verdict does
    not gate the unlink — e.g. calling the corroborator and continuing to the
    reclaim branch regardless.

    Second correct implementation: inline the batched ``sacct`` query at the
    call site and branch on ``states.get(jid) is None`` with no helper at all.
    Same three verdicts, every assertion here still passes.
    """
    b = _liveness_builder(tmp_path)
    s = _write_submitted_sentinel(b, "run_tritonswmm_evt-fresh", "48211")
    _stub_liveness(monkeypatch, squeue_live=False, sacct_rows={})

    alive = b._classify_live_sentinels([s], reclaim_dead=True)

    assert s.exists(), "an unconfirmable death verdict reclaimed a fresh sentinel"
    assert alive == [("run_tritonswmm_evt-fresh", "48211")], (
        f"the fresh unconfirmable sentinel was not reported alive: {alive}"
    )


def test_the_unconfirmable_hold_is_bounded_by_the_walltime_age_tier(tmp_path, monkeypatch):
    """Property: the unconfirmable hold ends once the sentinel's mtime passes
    the walltime cap — a FRESH and an AGED sentinel in ONE call diverge.

    Class: REGRESSION — red on the pre-change tree only for the fresh arm (the
    aged arm is reclaimed there too, for the wrong reason); the two arms
    together are what pin the age tier rather than the reclaim default.

    Why it is required: without a bound, a record purged from the accounting
    database reads as live forever and refuses every ``analysis.delete()`` on
    the tree until an operator passes ``override_in_flight``. The bound is the
    whole reason corroboration is affordable at the widest destructive site.

    What kills it: a tiebreak with inverted polarity (hold the aged one,
    reclaim the fresh one) reds BOTH assertions, which is why the two arms are
    in one call.

    Second correct implementation: compare ``s.stat().st_mtime`` against an
    absolute deadline computed once as ``time.time() - max_plausible_s``,
    rather than an age against a duration. Passes.
    """
    b = _liveness_builder(tmp_path)
    cap = _cap_s(b)
    fresh = _write_submitted_sentinel(b, "run_tritonswmm_evt-fresh", "48212", age_s=cap - 600)
    aged = _write_submitted_sentinel(b, "run_tritonswmm_evt-aged", "48213", age_s=cap + 60)
    _stub_liveness(monkeypatch, squeue_live=False, sacct_rows={})

    alive = b._classify_live_sentinels([fresh, aged], reclaim_dead=True)

    assert fresh.exists(), "the age tier reclaimed a sentinel younger than the walltime cap"
    assert not aged.exists(), "the age tier did not resolve a sentinel older than the walltime cap"
    assert alive == [("run_tritonswmm_evt-fresh", "48212")], f"alive set is wrong: {alive}"


def test_a_confirmed_death_is_reclaimed_while_a_live_row_is_held(tmp_path, monkeypatch):
    """Property: with BOTH sentinels fresh, the corroborator's own verdict is
    the discriminator — a terminal row reclaims, a non-terminal row holds.

    Class: REGRESSION — red on the pre-change tree, which reclaims both.

    Why it is required: the corroboration must actually decide, not merely run.
    A terminal accounting row is positive evidence the job is gone, and it is
    the only input in this whole method that can support a reclaim without an
    appeal to age.

    What kills it: a corroboration that fires but does not gate — the dead arm
    survives, so the reclaim assertion reds.

    Second correct implementation: reuse ``_SACCT_DEAD_STATES`` membership
    through a set operation (``{row[0]} & _SACCT_DEAD_STATES``) instead of an
    ``in`` test. Passes.
    """
    b = _liveness_builder(tmp_path)
    dead = _write_submitted_sentinel(b, "run_tritonswmm_evt-dead", "48214")
    live = _write_submitted_sentinel(b, "run_tritonswmm_evt-live", "48215")
    _stub_liveness(
        monkeypatch,
        squeue_live=False,
        sacct_rows={
            "48214": ("CANCELLED", "0:15", "JobKilled"),
            "48215": ("RUNNING", "0:0", "None"),
        },
    )

    alive = b._classify_live_sentinels([dead, live], reclaim_dead=True)

    assert not dead.exists(), "a sacct-confirmed death did not reclaim its sentinel"
    assert live.exists(), "a sacct-confirmed LIVE row was reclaimed"
    assert alive == [("run_tritonswmm_evt-live", "48215")], f"alive set is wrong: {alive}"


def test_the_age_tier_resolves_without_unlinking_on_the_delete_path(tmp_path, monkeypatch):
    """Property: on the delete path (``reclaim_dead=False``) the age tier still
    RESOLVES an aged unconfirmable sentinel — it leaves the alive set while the
    file stays on disk for the delete workflow to remove.

    Class: REGRESSION — red on the pre-change tree, where the aged sentinel is
    absent from the alive set for the wrong reason and the fresh one is absent
    too.

    Why it is required: this is the difference between a bounded hold and a
    refusal nobody can clear. ``reclaim_dead`` governs the UNLINK; if it also
    governed liveness, the widest destructive site would be the one place the
    bound did not apply.

    What kills it: a shape that gates the resolve on ``reclaim_dead`` as well as
    the unlink — the aged arm stays in the alive set and the first assertion
    reds.

    Second correct implementation: compute the alive set first, then unlink the
    resolved ones in a second pass when ``reclaim_dead``. Passes.
    """
    b = _liveness_builder(tmp_path)
    cap = _cap_s(b)
    fresh = _write_submitted_sentinel(b, "run_tritonswmm_evt-fresh", "48216", age_s=cap - 600)
    aged = _write_submitted_sentinel(b, "run_tritonswmm_evt-aged", "48217", age_s=cap + 60)
    _stub_liveness(monkeypatch, squeue_live=False, sacct_rows={})

    alive = b._classify_live_sentinels([fresh, aged], reclaim_dead=False)

    assert alive == [("run_tritonswmm_evt-fresh", "48216")], (
        f"the delete path did not resolve the aged sentinel: {alive}"
    )
    assert fresh.exists() and aged.exists(), "reclaim_dead=False unlinked a sentinel"


def test_a_confirmed_live_row_outranks_the_age_tier(tmp_path, monkeypatch):
    """Property: a sentinel whose accounting row is NON-terminal is held however
    old it is — the age tier applies only to the unconfirmable verdict.

    Class: REGRESSION — red on the pre-change tree, which reclaims it because
    squeue said not-live.

    Why it is required: the tier exists to bound NOT-KNOWN, not to overrule
    positive evidence. A job still in the scheduler is exactly the case a
    reclaim must survive.

    What kills it: a shape that applies the age tier to every not-live verdict
    without first checking the corroborator's answer.

    Second correct implementation: order the branches as a verdict dispatch
    table (``{"dead": reclaim, "alive": hold, "unknown": age_tier}[verdict]``)
    rather than sequential early returns. Passes.
    """
    b = _liveness_builder(tmp_path)
    cap = _cap_s(b)
    s = _write_submitted_sentinel(b, "run_tritonswmm_evt-old-live", "48218", age_s=cap + 60)
    _stub_liveness(monkeypatch, squeue_live=False, sacct_rows={"48218": ("PENDING", "0:0", "Priority")})

    alive = b._classify_live_sentinels([s], reclaim_dead=True)

    assert s.exists(), "the age tier overruled a confirmed-live accounting row"
    assert alive == [("run_tritonswmm_evt-old-live", "48218")], f"alive set is wrong: {alive}"


def test_an_empty_sentinel_sweep_makes_no_liveness_call(tmp_path, monkeypatch):
    """Property: an empty sweep returns ``[]`` and reaches NEITHER liveness
    instrument — the common nothing-to-reconcile case costs no scheduler round
    trip.

    Class: NEW CAPABILITY — the plausible wrong implementation is a batched
    corroborator call hoisted above the emptiness check, which is the shape a
    "one call for N ids" refactor naturally takes (``_classify_stale_via_sacct``
    already has that shape). Under it an empty sweep still calls ``sacct`` and
    this test reds; the pasted output is in the coder's round-2 return.

    Why it is required: this method runs on every delete preflight, and on a
    clean tree it runs against an empty sweep. A probe there buys nothing.

    What kills it: a ``_sacct_states_batched`` call reached with no sentinel in
    hand. Measured at both placements the refactor can take, and they red
    different arms — which is why both are named. Placed ABOVE the emptiness
    check, an empty sweep calls sacct and the property assertion reds
    (``assert ['sacct'] == []``). Placed BELOW the guard, the empty sweep is
    still protected and the must-fire control reds instead
    (``['sacct', 'squeue', 'sacct']`` against the expected
    ``['squeue', 'sacct']``). Output for both is pasted in the coder's round-2
    return.

    Not established: what this test does NOT defend is the ``if not
    sentinel_paths: return []`` guard. Removing that guard is behaviourally
    invisible here and the suite stays green, because an empty loop also makes
    no call; the guard's only remaining effect is that
    ``_max_plausible_job_lifetime_min(self.cfg_analysis)`` is not evaluated on
    an empty sweep. Pinning THAT would over-constrain, since computing the cap
    eagerly is a correct implementation and would fail such a test — so the
    guard carries no test by design, and this field names that rather than
    claiming a kill it does not make.

    Second correct implementation: move the cap computation below the point
    where the sweep is known non-empty, or drop the guard entirely. Both pass,
    because this test measures INSTRUMENT CALLS and never config reads.
    """
    called: list[str] = []

    def _must_fire(job_ids, **kw):
        called.append("sacct")
        return {}

    b = _liveness_builder(tmp_path)
    monkeypatch.setattr("hhemt.workflow._sacct_states_batched", _must_fire)
    monkeypatch.setattr("hhemt.workflow._slurm_job_is_live", lambda jid, **kw: called.append("squeue") or False)

    assert b._classify_live_sentinels([], reclaim_dead=True) == []
    assert called == []  # zero-call no-op

    # MUST-FIRE CONTROL: the same instrument on a non-empty sweep must fire, so
    # the zero above is an absence of work and not a broken monkeypatch.
    s = _write_submitted_sentinel(b, "run_tritonswmm_evt-ctl", "48219")
    b._classify_live_sentinels([s], reclaim_dead=True)
    assert called == ["squeue", "sacct"], f"the control arm never reached the instruments: {called}"
