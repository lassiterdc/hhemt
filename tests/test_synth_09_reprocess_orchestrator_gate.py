"""Phase 2 unit tests for the reprocess orchestration-liveness gate.

Harness patterns mirror tests/test_synth_06_submission_guard.py (sentinel
fixtures) and tests/test_synth_delete_slurm_lift_and_sentinels.py (sacct/ps/tmux
monkeypatching).
"""

import json
from pathlib import Path

from hhemt import orchestrator_sentinels as osent


def _write_sentinel(analysis_dir, driver_id, mode, **kw):
    return osent.write_orchestrator_sentinel(
        analysis_dir, driver_id=driver_id, workflow_submission_mode=mode, **kw
    )


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


def test_e_single_job_dead_jobid_reclaimed(synthetic_multisim_builder, monkeypatch):
    """1_job_many_srun_tasks arm: squeue-dead jobid ⇒ reclaimed, gate None.

    The gate's single-job arm calls workflow._slurm_job_is_live (NOT
    _classify_stale_via_sacct — that helper resolves its sentinel under
    _status/_submitted/{token}.json and would mtime-tiebreak against a
    non-existent _orchestrator path → false-proceed). Patch the symbol the
    gate actually imports.
    """
    b = synthetic_multisim_builder
    s = _write_sentinel(
        b.analysis_paths.analysis_dir, "d3", "1_job_many_srun_tasks", slurm_jobid="55"
    )
    monkeypatch.setattr("hhemt.workflow._slurm_job_is_live", lambda jid, **k: False)
    assert b._orchestrator_liveness_gate() is None
    assert not Path(s).exists()  # reclaimed


def test_e2_single_job_live_jobid_refuses(synthetic_multisim_builder, monkeypatch):
    """1_job_many_srun_tasks arm: squeue-live jobid ⇒ gate refuses, sentinel kept."""
    b = synthetic_multisim_builder
    s = _write_sentinel(
        b.analysis_paths.analysis_dir, "d4", "1_job_many_srun_tasks", slurm_jobid="56"
    )
    monkeypatch.setattr("hhemt.workflow._slurm_job_is_live", lambda jid, **k: True)
    err = b._orchestrator_liveness_gate()
    assert err is not None and "orchestration driver" in err.stderr
    assert Path(s).exists()  # live ⇒ preserved


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


def test_g_legacy_sentinel_without_hostname_is_unknown_not_dead(
    synthetic_multisim_builder, monkeypatch
):
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
    s = _write_sentinel(
        b.analysis_paths.analysis_dir, "d-notmux", "batch_job", tmux_session_name="sess-x"
    )
    monkeypatch.setattr("subprocess.run", _fake_run(tmux_available=False))
    err = b._orchestrator_liveness_gate()
    assert err is not None and "UNKNOWN/held" in err.stderr
    assert Path(s).exists()
    assert b._tmux_session_is_live("sess-x") is None


def test_j_tmux_probe_is_tristate_on_same_host(synthetic_multisim_builder, monkeypatch):
    """tmux RAN and answered: a live session refuses; an absent session on a
    KNOWN-same-host sentinel is dead evidence and is reclaimed."""
    b = synthetic_multisim_builder
    s_live = _write_sentinel(
        b.analysis_paths.analysis_dir, "d-tmux-live", "batch_job", tmux_session_name="alive-sess"
    )
    monkeypatch.setattr("subprocess.run", _fake_run(tmux_alive={"alive-sess"}))
    assert b._orchestrator_liveness_gate() is not None
    assert Path(s_live).exists()
    Path(s_live).unlink()

    s_dead = _write_sentinel(
        b.analysis_paths.analysis_dir, "d-tmux-dead", "batch_job", tmux_session_name="gone-sess"
    )
    monkeypatch.setattr("subprocess.run", _fake_run(tmux_alive=set()))
    assert b._orchestrator_liveness_gate() is None
    assert not Path(s_dead).exists()
