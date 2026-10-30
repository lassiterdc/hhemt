"""Phase 2 unit tests for the reprocess orchestration-liveness gate.

Harness patterns mirror tests/test_synth_06_submission_guard.py (sentinel
fixtures) and tests/test_synth_delete_slurm_lift_and_sentinels.py (sacct/ps/tmux
monkeypatching).
"""

import json
from pathlib import Path

from hhemt import orchestrator_sentinels as osent
from hhemt.exceptions import WorkflowError


def _write_sentinel(analysis_dir, driver_id, mode, **kw):
    return osent.write_orchestrator_sentinel(analysis_dir, driver_id=driver_id, workflow_submission_mode=mode, **kw)


def test_a_no_orchestrator_sentinel_proceeds(synthetic_multisim_builder):
    """(a)+(f) default-safe: no _orchestrator/ sentinel ⇒ gate returns None even
    when _submitted/ sim sentinels are present."""
    b = synthetic_multisim_builder
    # simulate queued workers: write a _submitted/ sim sentinel
    sub = b.analysis_paths.analysis_dir / "_status" / "_submitted"
    sub.mkdir(parents=True, exist_ok=True)
    (sub / "run_tritonswmm_evt-x.json").write_text(json.dumps({"slurm_jobid": "1"}))
    assert b._orchestrator_liveness_gate() is None


def test_b_live_local_pid_refuses(synthetic_multisim_builder, monkeypatch):
    """(b) a live local driver (ps -p alive) ⇒ gate returns a WorkflowError."""
    b = synthetic_multisim_builder
    _write_sentinel(b.analysis_paths.analysis_dir, "d1", "local", pid=4242)
    monkeypatch.setattr("subprocess.run", _fake_run(ps_alive={4242}))
    err = b._orchestrator_liveness_gate()
    assert err is not None and "orchestration driver" in err.stderr


def test_c_stale_dead_pid_reclaimed_then_proceeds(synthetic_multisim_builder, monkeypatch):
    """(c) dead-pid sentinel reclaimed (file removed), gate returns None."""
    b = synthetic_multisim_builder
    s = _write_sentinel(b.analysis_paths.analysis_dir, "d2", "local", pid=9999)
    monkeypatch.setattr("subprocess.run", _fake_run(ps_alive=set()))
    assert b._orchestrator_liveness_gate() is None
    assert not Path(s).exists()  # reclaimed


def test_d_reprocess_vs_reprocess_mutex(synthetic_multisim_builder, monkeypatch):
    """(d) a second reprocess sees the first's live sentinel and refuses; the
    caller's own self-sentinel is excluded via exclude_driver_id."""
    b = synthetic_multisim_builder
    _write_sentinel(b.analysis_paths.analysis_dir, "reproc-A", "local", pid=4242)
    s_b = _write_sentinel(b.analysis_paths.analysis_dir, "reproc-B", "local", pid=4243)
    monkeypatch.setattr("subprocess.run", _fake_run(ps_alive={4242, 4243}))
    # reproc-B running the gate excludes itself, still sees live reproc-A
    err = b._orchestrator_liveness_gate(exclude_driver_id="reproc-B")
    assert err is not None
    assert Path(s_b).exists()  # own sentinel untouched


def test_e_single_job_dead_jobid_is_corroborated_before_reclaim(synthetic_multisim_builder, monkeypatch):
    """1_job_many_srun_tasks arm: a not-live jobid is CORROBORATED before anything is reclaimed.

    REPLACES the pre-fix ``test_e_single_job_dead_jobid_reclaimed``, which asserted
    that a bare ``squeue``-dead answer reclaims the sentinel and lets the gate
    proceed. That is the exact behaviour decision D1 was ruled to change — a
    "not running" answer is confirmed by a second, separate query before a
    destructive act — so this is a decision REVERSAL and not a regression. The
    pre-fix assertion was captured red against this test, and this test was
    captured red against the pre-fix tree; both are pasted in chunk 2b.3's return.

    The gate's arms are the two HOST-AGNOSTIC arms of ``_orchestrator_liveness_gate``,
    and this one has no ``try``/``except`` around its probe (the sibling arm does),
    so the corroboration must not depend on that arm's exception posture.
    """
    b = synthetic_multisim_builder
    s = _write_sentinel(b.analysis_paths.analysis_dir, "d3", "1_job_many_srun_tasks", slurm_jobid="55")
    monkeypatch.setattr("hhemt.workflow._slurm_job_is_live", lambda jid, **k: False)
    monkeypatch.setattr("hhemt.workflow._corroborate_not_live", lambda jid: "unknown")
    err = b._orchestrator_liveness_gate()
    assert isinstance(err, WorkflowError), "an unconfirmable second query is UNKNOWN, never a reclaim"
    # Operator-visible signal asserted, not merely a counter: the refusal's LISTING
    # entry names the verdict and the cap. `"UNKNOWN/held"` alone is NOT discriminating —
    # the refusal's static explanatory paragraph contains that phrase unconditionally —
    # so the assertion keys on the held listing's tail.
    assert "UNKNOWN/held" in err.stderr
    assert "min cap, sentinel=" in err.stderr
    assert Path(s).exists(), "UNKNOWN is held, never reclaimed"


def test_e2_single_job_live_jobid_refuses(synthetic_multisim_builder, monkeypatch):
    """1_job_many_srun_tasks arm: squeue-live jobid ⇒ gate refuses, sentinel kept."""
    b = synthetic_multisim_builder
    s = _write_sentinel(b.analysis_paths.analysis_dir, "d4", "1_job_many_srun_tasks", slurm_jobid="56")
    monkeypatch.setattr("hhemt.workflow._slurm_job_is_live", lambda jid, **k: True)
    err = b._orchestrator_liveness_gate()
    assert err is not None and "orchestration driver" in err.stderr
    assert Path(s).exists()  # live ⇒ preserved


def test_e3_single_job_corroborated_death_reclaims(synthetic_multisim_builder, monkeypatch):
    """1_job_many_srun_tasks arm: a CONFIRMED death still reclaims and proceeds.

    Without this arm the corroboration would be indistinguishable from a blanket
    refusal, which would strand every legitimately dead sentinel on this arm.
    """
    b = synthetic_multisim_builder
    s = _write_sentinel(b.analysis_paths.analysis_dir, "d3b", "1_job_many_srun_tasks", slurm_jobid="57")
    monkeypatch.setattr("hhemt.workflow._slurm_job_is_live", lambda jid, **k: False)
    monkeypatch.setattr("hhemt.workflow._corroborate_not_live", lambda jid: "dead")
    assert b._orchestrator_liveness_gate() is None
    assert not Path(s).exists()  # corroborated gone ⇒ reclaimed


def test_e4_single_job_corroborated_alive_blocks_as_alive(synthetic_multisim_builder, monkeypatch):
    """1_job_many_srun_tasks arm: a non-terminal second-query row is ALIVE, not merely held.

    The distinction is operator-visible — 'liveness=ALIVE' is a direct measurement,
    while 'UNKNOWN/held … cap' says the probe could not answer — so this asserts
    the message, not just the block.
    """
    b = synthetic_multisim_builder
    s = _write_sentinel(b.analysis_paths.analysis_dir, "d3c", "1_job_many_srun_tasks", slurm_jobid="58")
    monkeypatch.setattr("hhemt.workflow._slurm_job_is_live", lambda jid, **k: False)
    monkeypatch.setattr("hhemt.workflow._corroborate_not_live", lambda jid: "alive")
    err = b._orchestrator_liveness_gate()
    assert isinstance(err, WorkflowError)
    assert "liveness=ALIVE" in err.stderr
    # The held LISTING entry is `liveness=UNKNOWN/held {n}min < {n}min cap, sentinel={p}`.
    # The refusal's static explanatory paragraph mentions "UNKNOWN/held" unconditionally,
    # so the negative half keys on the listing tail rather than on that phrase.
    assert "min cap, sentinel=" not in err.stderr
    assert Path(s).exists()


def test_e5_single_job_unconfirmable_hold_is_bounded_by_the_age_cap(synthetic_multisim_builder, monkeypatch):
    """No SECOND tier is introduced at the arms: the gate's EXISTING mtime-age cap bounds the hold.

    Chunk 2b.3's per-site disposition for both gate arms is "unconfirmable resolves
    to ``alive = None``, bounded by the gate's existing tier"; a builder that adds a
    second bound has duplicated the fail-safe. This drives the shared tier to its
    exhausted end, so an unconditional hold reds here.
    """
    b = synthetic_multisim_builder
    s = _write_sentinel(b.analysis_paths.analysis_dir, "d3d", "1_job_many_srun_tasks", slurm_jobid="59")
    _backdate(s, _cap_seconds(b) + 120)
    monkeypatch.setattr("hhemt.workflow._slurm_job_is_live", lambda jid, **k: False)
    monkeypatch.setattr("hhemt.workflow._corroborate_not_live", lambda jid: "unknown")
    assert b._orchestrator_liveness_gate() is None
    assert not Path(s).exists(), "past the age cap an unanswerable record resolves like a confirmed death"


def test_e6_single_job_raising_second_query_is_unknown_not_a_raise(synthetic_multisim_builder, monkeypatch):
    """A second query that RAISES is NOT-KNOWN, never an exception out of the classifier.

    The gate's tri-state contract is the subject: ``_corroborate_not_live`` documents
    that it never raises, but it delegates to ``_sacct_states_batched``, which catches
    only ``TimeoutExpired`` and ``FileNotFoundError`` — so any other ``OSError``
    reaches this arm. The arm that owes the corroboration here has NO ``try``/``except``
    of its own, which makes an unguarded call the one place a raise can escape a
    function whose entire return contract is ``WorkflowError | None``.
    """
    b = synthetic_multisim_builder
    s = _write_sentinel(b.analysis_paths.analysis_dir, "d3e", "1_job_many_srun_tasks", slurm_jobid="60")
    monkeypatch.setattr("hhemt.workflow._slurm_job_is_live", lambda jid, **k: False)

    def _no_second_query(_jid):
        raise PermissionError("sacct")

    monkeypatch.setattr("hhemt.workflow._corroborate_not_live", _no_second_query)
    err = b._orchestrator_liveness_gate()
    assert isinstance(err, WorkflowError), "an unanswerable second query is UNKNOWN, never a raise"
    assert "min cap, sentinel=" in err.stderr  # the held LISTING entry, not the static prose
    assert Path(s).exists()


def test_e7_arm_one_foreign_origin_with_a_jobid_is_a_corroborated_hold(synthetic_multisim_builder, monkeypatch):
    """The job-id arm is host-agnostic and tested FIRST, so the origin host does not select the family.

    Class: NEW CAPABILITY — the cause token does not exist pre-change, so this reds
    there. Its subject is not the wording but the FAMILY ASSIGNMENT: a message that
    keys the family on host= calls this record a host-local gap, and the host-local
    remedy ("re-run from the origin host") provably cannot change the answer, because
    this arm never consults the origin host.

    Why required: a single-job sentinel written on a compute node and gated from a
    login node is the ordinary cross-node resume, not an edge case.
    """
    b = synthetic_multisim_builder
    s = _write_sentinel(b.analysis_paths.analysis_dir, "d3f", "1_job_many_srun_tasks", slurm_jobid="61")
    _set_hostname(s, "some-other-node")
    monkeypatch.setattr("hhemt.workflow._slurm_job_is_live", lambda jid, **k: False)
    monkeypatch.setattr("hhemt.workflow._corroborate_not_live", lambda jid: "unknown")
    err = b._orchestrator_liveness_gate()
    assert isinstance(err, WorkflowError)
    assert "liveness=UNKNOWN/held (corroborated-hold)" in err.stderr
    assert Path(s).exists()


def test_e8_same_host_jobid_with_tmux_unavailable_is_not_a_corroborated_hold(synthetic_multisim_builder, monkeypatch):
    """A same-host detached record never reaches a job-id arm, so a recorded job id does not make it a hold.

    Class: NEW CAPABILITY — reds pre-change. Its subject is the family assignment the
    other way round: a message keying on "a RECORDED job id is a corroborated hold"
    tells the operator that squeue answered and sacct could not, for a record where
    neither ran. The tmux remedy hint is asserted here too, because a cause whose
    remedy is missing is the defect this round exists to close.
    """
    b = synthetic_multisim_builder
    s = _write_sentinel(b.analysis_paths.analysis_dir, "d3g", "batch_job", slurm_jobid="62", tmux_session_name="sess-z")

    def _probe_must_not_run(_jid, **k):
        raise AssertionError("a same-host detached record takes the tmux arm, never a job-id arm")

    called: list[str] = []
    monkeypatch.setattr("hhemt.workflow._slurm_job_is_live", _probe_must_not_run)
    monkeypatch.setattr("hhemt.workflow._corroborate_not_live", lambda jid: called.append(jid) or "unknown")
    monkeypatch.setattr("hhemt.workflow.SnakemakeWorkflowBuilder._tmux_session_is_live", lambda self, _sess: None)

    err = b._orchestrator_liveness_gate()
    assert isinstance(err, WorkflowError)
    assert "liveness=UNKNOWN/held (tmux-unavailable)" in err.stderr
    assert "module load tmux" in err.stderr, "this cause's remedy is the module load hint"
    assert called == [], "no corroboration may be claimed for a record that never probed a job id"
    assert Path(s).exists()


def test_e9_arm_two_with_squeue_missing_is_a_distinct_cause_from_a_hold(synthetic_multisim_builder, monkeypatch):
    """The must-fire PAIR with test_e10: identical record FIELDS, different causes.

    Both records are `batch_job`, unrecorded origin, recorded job id. Nothing in the
    record distinguishes them; only the branch does. That is the measurement which
    establishes that a message inferring the family from the record cannot be correct,
    and therefore that the gate must emit it.
    """
    b = synthetic_multisim_builder
    s = _write_sentinel(b.analysis_paths.analysis_dir, "d3h", "batch_job", slurm_jobid="63")
    _strip_hostname(s)

    def _no_squeue(_jid, **k):
        raise FileNotFoundError("squeue")

    monkeypatch.setattr("hhemt.workflow._slurm_job_is_live", _no_squeue)
    monkeypatch.setattr("hhemt.workflow._corroborate_not_live", lambda jid: "unknown")
    err = b._orchestrator_liveness_gate()
    assert isinstance(err, WorkflowError)
    assert "liveness=UNKNOWN/held (squeue-unavailable)" in err.stderr
    assert Path(s).exists()


def test_e10_arm_two_with_a_working_squeue_is_a_corroborated_hold(synthetic_multisim_builder, monkeypatch):
    """The other half of the e9/e10 pair: SAME fields as test_e9, cause differs.

    Together the two tests are the control that a field-keyed message cannot pass:
    any rule matching e9's record to a family matches e10's identical record to it too.
    """
    b = synthetic_multisim_builder
    s = _write_sentinel(b.analysis_paths.analysis_dir, "d3i", "batch_job", slurm_jobid="64")
    _strip_hostname(s)
    monkeypatch.setattr("hhemt.workflow._slurm_job_is_live", lambda jid, **k: False)
    monkeypatch.setattr("hhemt.workflow._corroborate_not_live", lambda jid: "unknown")
    err = b._orchestrator_liveness_gate()
    assert isinstance(err, WorkflowError)
    assert "liveness=UNKNOWN/held (corroborated-hold)" in err.stderr
    assert Path(s).exists()


def test_e11_foreign_origin_with_no_identity_is_unenriched_not_host_local(synthetic_multisim_builder, monkeypatch):
    """A foreign-origin record with nothing to probe is NOT a host-local gap.

    Class: NEW CAPABILITY — reds pre-change (and reds against the round-3 code,
    which assigned this record ``host-local``). Its subject is the remedy's
    executability: ``host-local`` tells the operator to re-run from the origin host
    "so the probe can answer", and for this record there is no probe on ANY host —
    it carries no pid, no tmux session name and no job id. Obeying that remedy moves
    the record to ``unenriched`` and the sentinel is still held.

    Why required: a sentinel written on a compute node and gated from a login node
    before enrichment is the ordinary window the identity-enrichment seam opens.
    """
    b = synthetic_multisim_builder
    s = _write_sentinel(b.analysis_paths.analysis_dir, "d3j", "tmux")
    _set_hostname(s, "some-other-node")
    err = b._orchestrator_liveness_gate()
    assert isinstance(err, WorkflowError)
    assert "liveness=UNKNOWN/held (unenriched)" in err.stderr
    assert Path(s).exists()


def test_e12_foreign_origin_batch_job_with_a_pid_but_no_session_is_unenriched(synthetic_multisim_builder):
    """The probe an origin host could run is the one THIS MODE reads, not "any identity field".

    Class: NEW CAPABILITY — reds pre-change. A `batch_job` record's host-local probe is
    `tmux has-session` against ``tmux_session_name``; a recorded ``pid`` is not a probe
    the batch_job arm can use. So "carries an identity field" is too loose a test — on
    the origin host this record still lands in ``unenriched``, and the cause must say so
    from the first read.
    """
    b = synthetic_multisim_builder
    s = _write_sentinel(b.analysis_paths.analysis_dir, "d3k", "batch_job", pid=4242)
    _set_hostname(s, "some-other-node")
    err = b._orchestrator_liveness_gate()
    assert isinstance(err, WorkflowError)
    assert "liveness=UNKNOWN/held (unenriched)" in err.stderr
    assert Path(s).exists()


def test_e13_foreign_origin_with_a_probeable_session_is_still_host_local(synthetic_multisim_builder, monkeypatch):
    """Must-fire control for e11/e12: a foreign record whose mode CAN be probed elsewhere stays host-local.

    Without this the narrowing above would pass by assigning ``unenriched`` to every
    foreign-origin record, which is a different wrong answer — the origin-host remedy
    is genuinely correct for this shape.
    """
    b = synthetic_multisim_builder
    s = _write_sentinel(b.analysis_paths.analysis_dir, "d3l", "batch_job", tmux_session_name="sess-q")
    _set_hostname(s, "some-other-node")
    err = b._orchestrator_liveness_gate()
    assert isinstance(err, WorkflowError)
    assert "liveness=UNKNOWN/held (host-local)" in err.stderr
    assert Path(s).exists()


def _fake_run(ps_alive=frozenset(), tmux_alive=frozenset(), tmux_available=True):
    """Return a subprocess.run stub.

    ``ps -p {pid}``                      -> rc 0 iff pid in ``ps_alive``.
    ``bash -c "...tmux has-session..."`` -> rc 0 iff the session is in
    ``tmux_alive``; rc 127 + "command not found" when ``tmux_available`` is
    False (the Lmod module-not-loaded shape). The gate now routes tmux through
    ``bash -c`` + ``_get_module_load_prefix()`` like every other tmux call site
    in workflow.py, so the bare-argv form is gone.
    """
    import subprocess

    def _run(cmd, *a, **k):
        rc, err = 1, ""
        if cmd[:2] == ["ps", "-p"] and int(cmd[2]) in ps_alive:
            rc = 0
        elif cmd[:2] == ["bash", "-c"] and "tmux has-session" in cmd[2]:
            if not tmux_available:
                rc, err = 127, "bash: line 1: tmux: command not found"
            elif any(s in cmd[2] for s in tmux_alive):
                rc = 0
        return subprocess.CompletedProcess(cmd, rc, "", err)

    return _run


def _cap_seconds(builder):
    from hhemt.workflow import _max_plausible_job_lifetime_min

    return _max_plausible_job_lifetime_min(builder.cfg_analysis) * 60


def _set_hostname(sentinel_path, hostname):
    p = Path(sentinel_path)
    payload = json.loads(p.read_text())
    payload["hostname"] = hostname
    p.write_text(json.dumps(payload))


def _strip_hostname(sentinel_path):
    """Reproduce a sentinel written before the hostname field existed."""
    p = Path(sentinel_path)
    payload = json.loads(p.read_text())
    payload.pop("hostname", None)
    p.write_text(json.dumps(payload))


def _backdate(sentinel_path, seconds):
    import os

    st = Path(sentinel_path).stat()
    os.utime(sentinel_path, (st.st_atime - seconds, st.st_mtime - seconds))


def test_k_write_stamps_origin_hostname(synthetic_multisim_builder):
    """Schema: every newly written sentinel records its origin host."""
    import socket

    b = synthetic_multisim_builder
    s = _write_sentinel(b.analysis_paths.analysis_dir, "d-host", "local", pid=1)
    assert json.loads(Path(s).read_text())["hostname"] == socket.gethostname()


def test_f_foreign_host_sentinel_is_held_not_reclaimed(synthetic_multisim_builder, monkeypatch):
    """T1: a foreign-origin sentinel is UNKNOWN -> held (gate refuses) and NOT
    reclaimed, even though the host-local probe would report 'dead'."""
    b = synthetic_multisim_builder
    s = _write_sentinel(b.analysis_paths.analysis_dir, "d-foreign", "local", pid=9999)
    _set_hostname(s, "some-other-login-node")
    monkeypatch.setattr("subprocess.run", _fake_run(ps_alive=set()))
    err = b._orchestrator_liveness_gate()
    assert err is not None and "UNKNOWN/held" in err.stderr
    assert Path(s).exists()  # never reclaim on an unanswerable probe


def test_f2_foreign_host_sentinel_ages_out(synthetic_multisim_builder, monkeypatch):
    """T1 bound: past the max-plausible driver lifetime the held sentinel is
    reclaimed as UNKNOWN/age-exceeded, so conservative-hold cannot block
    forever. This is the branch that unblocks a stale cross-node sentinel."""
    b = synthetic_multisim_builder
    s = _write_sentinel(b.analysis_paths.analysis_dir, "d-foreign-old", "local", pid=9999)
    _set_hostname(s, "some-other-login-node")
    _backdate(s, _cap_seconds(b) + 120)
    monkeypatch.setattr("subprocess.run", _fake_run(ps_alive=set()))
    assert b._orchestrator_liveness_gate() is None
    assert not Path(s).exists()


def test_g_legacy_sentinel_without_hostname_is_unknown_not_dead(synthetic_multisim_builder, monkeypatch):
    """Backward compat: a sentinel predating the hostname field must NOT be
    dead-classified from a negative host-local probe."""
    b = synthetic_multisim_builder
    s = _write_sentinel(b.analysis_paths.analysis_dir, "d-legacy", "local", pid=9999)
    _strip_hostname(s)
    monkeypatch.setattr("subprocess.run", _fake_run(ps_alive=set()))
    err = b._orchestrator_liveness_gate()
    assert err is not None and "unrecorded" in err.stderr
    assert Path(s).exists()


def test_h_unenriched_identity_field_is_unknown_not_dead(synthetic_multisim_builder, monkeypatch):
    """T3 (SAME-host): between write_orchestrator_sentinel and
    enrich_orchestrator_sentinel a LIVE batch_job driver's sentinel carries
    tmux_session_name=None for the whole of submit_workflow(). That must be
    UNKNOWN, not DEAD -- the two-valued form reclaimed a live driver here."""
    b = synthetic_multisim_builder
    s = _write_sentinel(b.analysis_paths.analysis_dir, "d-unenriched", "batch_job")
    monkeypatch.setattr("subprocess.run", _fake_run())
    err = b._orchestrator_liveness_gate()
    assert err is not None and "UNKNOWN/held" in err.stderr
    assert Path(s).exists()


def test_i_missing_tmux_binary_is_unknown_not_dead(synthetic_multisim_builder, monkeypatch):
    """T2: tmux unavailable (Lmod module not loaded) must yield UNKNOWN-held --
    NOT a FileNotFoundError, and NOT a silent false-proceed."""
    b = synthetic_multisim_builder
    s = _write_sentinel(b.analysis_paths.analysis_dir, "d-notmux", "batch_job", tmux_session_name="sess-x")
    monkeypatch.setattr("subprocess.run", _fake_run(tmux_available=False))
    err = b._orchestrator_liveness_gate()
    assert err is not None and "UNKNOWN/held" in err.stderr
    assert Path(s).exists()
    assert b._tmux_session_is_live("sess-x") is None


def test_j_tmux_probe_is_tristate_on_same_host(synthetic_multisim_builder, monkeypatch):
    """tmux RAN and answered: a live session refuses; an absent session on a
    KNOWN-same-host sentinel is dead evidence and is reclaimed."""
    b = synthetic_multisim_builder
    s_live = _write_sentinel(b.analysis_paths.analysis_dir, "d-tmux-live", "batch_job", tmux_session_name="alive-sess")
    monkeypatch.setattr("subprocess.run", _fake_run(tmux_alive={"alive-sess"}))
    assert b._orchestrator_liveness_gate() is not None
    assert Path(s_live).exists()
    Path(s_live).unlink()

    s_dead = _write_sentinel(b.analysis_paths.analysis_dir, "d-tmux-dead", "batch_job", tmux_session_name="gone-sess")
    monkeypatch.setattr("subprocess.run", _fake_run(tmux_alive=set()))
    assert b._orchestrator_liveness_gate() is None
    assert not Path(s_dead).exists()
