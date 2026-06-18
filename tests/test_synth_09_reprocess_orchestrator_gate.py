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
    assert err is not None and "live orchestration driver" in err.stderr


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
    assert err is not None and "live orchestration driver" in err.stderr
    assert Path(s).exists()  # live ⇒ preserved


def _fake_run(ps_alive=frozenset(), tmux_alive=frozenset()):
    """Return a subprocess.run stub: ps -p {pid} rc=0 iff pid in ps_alive;
    tmux has-session -t {s} rc=0 iff s in tmux_alive."""
    import subprocess

    def _run(cmd, *a, **k):
        rc = 1
        if cmd[:2] == ["ps", "-p"] and int(cmd[2]) in ps_alive:
            rc = 0
        elif cmd[:2] == ["tmux", "has-session"] and cmd[3] in tmux_alive:
            rc = 0
        return subprocess.CompletedProcess(cmd, rc, b"", b"")

    return _run
