import os
import signal
import types

from hhemt.run_simulation import TRITONSWMM_run


class _FakeProc:
    """Minimal Popen stand-in: poll() returns None until 'killed' (via the
    monkeypatched os.killpg, which sets _killed) or until the sim 'completes'
    after `complete_after_polls` polls. `pid` lets the watcher compute a pgid via
    os.getpgid(proc.pid) and signal the process group with os.killpg."""

    def __init__(self, complete_after_polls=None, pid=4242):
        self._killed = False
        self._polls = 0
        self._complete_after = complete_after_polls
        self.pid = pid

    def poll(self):
        if self._killed:
            return -15  # bash killed by the process-group SIGTERM
        self._polls += 1
        if self._complete_after is not None and self._polls > self._complete_after:
            return 0
        return None

    @property
    def returncode(self):
        return -15 if self._killed else 0


def _install_fake_killpg(monkeypatch, proc, recorder):
    """Patch os.getpgid/os.killpg so the watcher's process-group SIGTERM is
    captured (and marks the fake proc 'killed') without touching a real group.
    Mirrors Rivanna semantics: the group SIGTERM is what reaps the srun step."""
    monkeypatch.setattr(os, "getpgid", lambda pid: pid)

    def _fake_killpg(pgid, sig):
        recorder["pgid"] = pgid
        recorder["sig"] = sig
        proc._killed = True

    monkeypatch.setattr(os, "killpg", _fake_killpg)


def test_deterministic_kill_fires_after_n_checkpoints(tmp_path, monkeypatch):
    """>= N+1 cfg files present -> ONE process-group SIGTERM fires (Rivanna
    srun-step teardown), killed return code."""
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    for i in range(4):  # 4 >= n_checkpoints(3) + 1 -> threshold already met
        (cfg_dir / f"config_{i:04d}.cfg").write_text("x\n")
    fake_self = types.SimpleNamespace(_hotstart_cfg_dir=lambda mt: cfg_dir)
    proc = _FakeProc(pid=4242)
    rec = {}
    _install_fake_killpg(monkeypatch, proc, rec)
    rc = TRITONSWMM_run.wait_with_deterministic_checkpoint_kill(
        fake_self, proc, model_type="tritonswmm", n_checkpoints=3, poll_interval_s=0
    )
    # process-group SIGTERM delivered to the bash+srun group leader's pgid
    assert rec["pgid"] == 4242
    assert rec["sig"] == signal.SIGTERM
    assert proc._killed is True
    assert rc == -15


def test_no_kill_when_sim_completes_before_threshold(tmp_path, monkeypatch):
    """Sim finishes before N+1 cfg files exist -> no kill, clean return code."""
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()  # empty -> threshold never reached
    fake_self = types.SimpleNamespace(_hotstart_cfg_dir=lambda mt: cfg_dir)
    proc = _FakeProc(complete_after_polls=2)
    rec = {}
    _install_fake_killpg(monkeypatch, proc, rec)
    rc = TRITONSWMM_run.wait_with_deterministic_checkpoint_kill(
        fake_self, proc, model_type="tritonswmm", n_checkpoints=3, poll_interval_s=0
    )
    assert proc._killed is False
    assert rec == {}  # os.killpg never called when the sim completes on its own
    assert rc == 0
