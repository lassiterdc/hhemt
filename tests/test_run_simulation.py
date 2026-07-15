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


# --- sticky-False completion-latch regression (no cluster, no compile) ---------
#
# Guards the Rivanna synth_cc_resume 0/28 defect: model_run_completed must NOT
# latch on a False record, because run_simulation_runner writes the field this
# method returns. See run_simulation.py::model_run_completed.


class _FakeField:
    def __init__(self, value=None):
        self.value = value

    def get(self):
        return self.value

    def set(self, v):
        self.value = v


class _FakeModelLog:
    def __init__(self, completed=None):
        self.simulation_completed = _FakeField(completed)


def _gate(tmp_path, *, recorded, marker_text, model_type="tritonswmm", rpt_ok=True):
    """Drive run_simulation.TRITONSWMM_run.model_run_completed against a fake log
    + a real on-disk marker file, with the coupled-rpt gate stubbed."""
    log_file = tmp_path / f"model_{model_type}.log"
    log_file.write_text(marker_text)

    fake_self = types.SimpleNamespace(
        _scenario=types.SimpleNamespace(get_log=lambda mt: _FakeModelLog(recorded)),
        _analysis_level_model_logfile=lambda mt: log_file,
        performance_file=lambda model_type: tmp_path / "performance.txt",
        _coupled_swmm_report_finalized=lambda mt: rpt_ok,
    )
    return TRITONSWMM_run.model_run_completed(fake_self, model_type)


def test_false_record_is_not_sticky_when_raw_markers_say_complete(tmp_path):
    """THE regression: a prior attempt recorded False; this attempt's resume ran to
    t=end and wrote 'Simulation ends'. The gate MUST re-derive True, not latch."""
    assert _gate(tmp_path, recorded=False, marker_text="... Simulation ends\n") is True


def test_false_record_stays_false_when_raw_markers_say_incomplete(tmp_path):
    """A False record with no completion marker must still be False (no false green)."""
    assert _gate(tmp_path, recorded=False, marker_text="Time: 100 dt: 0.5\n") is False


def test_true_record_is_authoritative_and_still_gated_on_coupled_rpt(tmp_path):
    """The True path is unchanged: authoritative, but still ANDed with the rpt gate."""
    # True + finalized rpt -> True (raw markers absent on purpose: never consulted)
    assert _gate(tmp_path, recorded=True, marker_text="") is True
    # True + unfinalized coupled rpt -> False
    assert _gate(tmp_path, recorded=True, marker_text="", rpt_ok=False) is False
