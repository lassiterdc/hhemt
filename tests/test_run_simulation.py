import types

from hhemt.run_simulation import TRITONSWMM_run


class _FakeProc:
    """Minimal Popen stand-in: poll() returns None until killed (or until the
    sim 'completes' after `complete_after_polls` polls)."""

    def __init__(self, complete_after_polls=None):
        self._killed = False
        self._polls = 0
        self._complete_after = complete_after_polls

    def poll(self):
        if self._killed:
            return -9
        self._polls += 1
        if self._complete_after is not None and self._polls > self._complete_after:
            return 0
        return None

    def kill(self):
        self._killed = True

    @property
    def returncode(self):
        return -9 if self._killed else 0


def test_deterministic_kill_fires_after_n_checkpoints(tmp_path):
    """>= N+1 cfg files present -> SIGKILL fires once, killed return code."""
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    for i in range(4):  # 4 >= n_checkpoints(3) + 1 -> threshold already met
        (cfg_dir / f"config_{i:04d}.cfg").write_text("x\n")
    fake_self = types.SimpleNamespace(_hotstart_cfg_dir=lambda mt: cfg_dir)
    proc = _FakeProc()
    rc = TRITONSWMM_run.wait_with_deterministic_checkpoint_kill(
        fake_self, proc, model_type="tritonswmm", n_checkpoints=3, poll_interval_s=0
    )
    assert proc._killed is True
    assert rc == -9


def test_no_kill_when_sim_completes_before_threshold(tmp_path):
    """Sim finishes before N+1 cfg files exist -> no kill, clean return code."""
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()  # empty -> threshold never reached
    fake_self = types.SimpleNamespace(_hotstart_cfg_dir=lambda mt: cfg_dir)
    proc = _FakeProc(complete_after_polls=2)
    rc = TRITONSWMM_run.wait_with_deterministic_checkpoint_kill(
        fake_self, proc, model_type="tritonswmm", n_checkpoints=3, poll_interval_s=0
    )
    assert proc._killed is False
    assert rc == 0
