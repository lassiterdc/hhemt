"""Per-scenario attempt guard: a second Slurm job must never write a scenario another job holds.

Why this module exists: Snakemake's Slurm executor resubmitted a job it saw end as NODE_FAIL
while the first job's process was still alive on the failed node, and both appended to one
scenario's output directory for hours (interleaved steps in outfall_stage.bin). The guard in
run_simulation_runner reads a lock record in the simulation directory, cancels a still
non-terminal holder, and refuses (writing nothing) when it cannot establish the holder is gone.

Slurm is never touched: `RecordingSlurm` stands in for squeue, sacct and scancel, replaying
recorded output and logging every call, and a fake clock makes the bounded wait instant.
"""

from __future__ import annotations

import json

from hhemt.run_simulation_runner import (
    _ATTEMPT_LOCK_NAME,
    _acquire_scenario_attempt,
    _release_scenario_attempt,
    _slurm_job_state,
)

LABEL = "year.0/tritonswmm (event_iloc=0)"


def _width(cmd):
    """Column width `sacct -o State[%N]` prints: 10 by default."""
    spec = cmd[cmd.index("-o") + 1]
    return int(spec.split("%")[1]) if "%" in spec else 10


def _truncate(state, width):
    """What the real tool does to a name wider than the column: cut, end with `+`."""
    return state if len(state) <= width else state[: width - 1] + "+"


class RecordingSlurm:
    """Recording stub for squeue/sacct/scancel.

    `states[jobid]` is a list of states returned by successive queries of that job (the
    last entry repeats); `None` as an entry means Slurm no longer knows the job.
    """

    def __init__(self, states=None):
        self.states = {k: list(v) for k, v in (states or {}).items()}
        self.calls: list[list[str]] = []

    def __call__(self, cmd):
        self.calls.append(list(cmd))
        tool = cmd[0]
        if tool == "scancel":
            return 0, ""
        jobid = cmd[cmd.index("-j") + 1]
        seq = self.states.get(jobid, [None])
        state = seq[0]
        if len(seq) > 1:
            seq.pop(0)
        if tool == "squeue":
            return 0, (f"{state}\n" if state in ("RUNNING", "PENDING", "COMPLETING") else "")
        if tool == "sacct":
            return 0, (f"{_truncate(state, _width(cmd))}\n" if state else "")
        raise AssertionError(f"unexpected command {cmd}")

    def of(self, tool):
        return [c for c in self.calls if c[0] == tool]


class FakeTime:
    def __init__(self):
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def _write_lock(sim_folder, jobid, node="n1"):
    sim_folder.mkdir(parents=True, exist_ok=True)
    (sim_folder / _ATTEMPT_LOCK_NAME).write_text(json.dumps({"slurm_jobid": jobid, "node": node}))


def _acquire(sim_folder, jobid, slurm, ft=None, **kw):
    ft = ft or FakeTime()
    return _acquire_scenario_attempt(
        sim_folder,
        jobid,
        "node7",
        scenario_label=LABEL,
        run_cmd=slurm,
        sleep=ft.sleep,
        clock=ft.clock,
        wait_s=kw.pop("wait_s", 30.0),
        poll_s=kw.pop("poll_s", 5.0),
    )


def _holder_of(sim_folder):
    return json.loads((sim_folder / _ATTEMPT_LOCK_NAME).read_text())["slurm_jobid"]


def test_no_record_takes_the_lock_without_querying_slurm(tmp_path):
    sim = tmp_path / "sims" / "year.0"  # does not exist yet: the guard creates it
    slurm = RecordingSlurm()
    assert _acquire(sim, "200", slurm) is None
    assert slurm.calls == []
    record = json.loads((sim / _ATTEMPT_LOCK_NAME).read_text())
    assert record["slurm_jobid"] == "200" and record["node"] == "node7"
    assert not list(sim.glob("*.tmp")), "atomic write must leave no temp file"


def test_record_naming_the_current_job_proceeds_without_querying_slurm(tmp_path):
    sim = tmp_path / "sim"
    _write_lock(sim, "200")
    slurm = RecordingSlurm()
    assert _acquire(sim, "200", slurm) is None
    assert slurm.calls == []
    assert _holder_of(sim) == "200"


def test_record_naming_a_terminal_job_never_blocks(tmp_path):
    for terminal in ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY"):
        sim = tmp_path / terminal
        _write_lock(sim, "100")
        slurm = RecordingSlurm({"100": [terminal]})
        assert _acquire(sim, "200", slurm) is None, terminal
        assert slurm.of("scancel") == [], terminal
        assert _holder_of(sim) == "200", terminal


def test_record_naming_a_job_slurm_no_longer_knows_never_blocks(tmp_path):
    sim = tmp_path / "sim"
    _write_lock(sim, "100")
    slurm = RecordingSlurm({"100": [None]})
    assert _acquire(sim, "200", slurm) is None
    assert slurm.of("scancel") == []
    assert _holder_of(sim) == "200"


def test_running_holder_is_cancelled_and_the_guard_proceeds_once_it_is_gone(tmp_path):
    sim = tmp_path / "sim"
    _write_lock(sim, "100")
    # RUNNING at the first look, still COMPLETING after scancel, then terminal.
    slurm = RecordingSlurm({"100": ["RUNNING", "COMPLETING", "CANCELLED"]})
    ft = FakeTime()
    assert _acquire(sim, "200", slurm, ft) is None
    assert slurm.of("scancel") == [["scancel", "100"]]
    first_scancel = slurm.calls.index(["scancel", "100"])
    assert all(c[0] != "scancel" for c in slurm.calls[:first_scancel])
    assert ft.sleeps, "the guard must poll after scancel, not assume the cancel took"
    assert _holder_of(sim) == "200"


def test_running_holder_that_never_leaves_refuses_and_writes_nothing(tmp_path):
    sim = tmp_path / "sim"
    _write_lock(sim, "100", node="n1")
    before = {p.name: p.read_text() for p in sim.iterdir()}
    slurm = RecordingSlurm({"100": ["RUNNING"]})  # scancel has no effect
    ft = FakeTime()
    message = _acquire(sim, "200", slurm, ft, wait_s=30.0, poll_s=5.0)
    assert message is not None
    for needle in (LABEL, "100", "RUNNING", "n1", "Nothing was written", "scontrol show job 100"):
        assert needle in message, needle
    assert ft.now >= 30.0, "the wait must be bounded by wait_s, not abandoned early"
    assert {p.name: p.read_text() for p in sim.iterdir()} == before
    assert _holder_of(sim) == "100", "the previous holder's record must be left intact"


def test_refusal_adds_no_sibling_file(tmp_path):
    sim = tmp_path / "sim"
    _write_lock(sim, "100")
    slurm = RecordingSlurm({"100": ["PENDING"]})
    assert _acquire(sim, "200", slurm) is not None
    assert sorted(p.name for p in sim.iterdir()) == [_ATTEMPT_LOCK_NAME]


def test_node_fail_holder_is_scancelled_waited_for_and_refused(tmp_path):
    sim = tmp_path / "sim"
    _write_lock(sim, "100")
    slurm = RecordingSlurm({"100": ["NODE_FAIL"]})
    ft = FakeTime()
    message = _acquire(sim, "200", slurm, ft)
    assert message is not None
    for needle in (LABEL, "100", "NODE_FAIL", str(sim / _ATTEMPT_LOCK_NAME), "node must be back"):
        assert needle in message, needle
    assert slurm.of("scancel") == [["scancel", "100"]]
    assert ft.now >= 30.0
    assert _holder_of(sim) == "100"


def test_node_fail_holder_that_reports_a_final_state_later_proceeds(tmp_path):
    sim = tmp_path / "sim"
    _write_lock(sim, "100")
    slurm = RecordingSlurm({"100": ["NODE_FAIL", "NODE_FAIL", "FAILED"]})
    assert _acquire(sim, "200", slurm) is None
    assert _holder_of(sim) == "200"


def test_out_of_memory_holder_survives_sacct_column_truncation(tmp_path):
    sim = tmp_path / "sim"
    _write_lock(sim, "100")
    slurm = RecordingSlurm({"100": ["OUT_OF_MEMORY"]})
    assert _acquire(sim, "200", slurm) is None
    assert slurm.of("scancel") == []
    sacct_cmd = slurm.of("sacct")[0]
    assert _width(sacct_cmd) >= len("OUT_OF_MEMORY"), "the State column must be wide enough for the longest state name"
    assert _truncate("OUT_OF_MEMORY", 10) == "OUT_OF_ME+", (
        "the stub must truncate like the real tool at its default width"
    )
    assert _holder_of(sim) == "200"


def test_local_run_without_a_job_id_skips_the_guard_entirely(tmp_path):
    sim = tmp_path / "sim"
    _write_lock(sim, "100")
    slurm = RecordingSlurm({"100": ["RUNNING"]})
    assert _acquire(sim, None, slurm) is None
    assert _acquire(sim, "", slurm) is None
    assert slurm.calls == []
    assert _holder_of(sim) == "100", "a local run must not touch a Slurm job's record"


def test_local_run_does_not_create_the_simulation_directory(tmp_path):
    sim = tmp_path / "never_created"
    assert _acquire(sim, None, RecordingSlurm()) is None
    assert not sim.exists()


def test_unreadable_record_names_no_holder(tmp_path):
    sim = tmp_path / "sim"
    sim.mkdir()
    (sim / _ATTEMPT_LOCK_NAME).write_text("{not json")
    slurm = RecordingSlurm()
    assert _acquire(sim, "200", slurm) is None
    assert slurm.calls == []
    assert _holder_of(sim) == "200"


def test_release_removes_only_its_own_record(tmp_path):
    sim = tmp_path / "sim"
    _write_lock(sim, "200")
    _release_scenario_attempt(sim, "300")
    assert (sim / _ATTEMPT_LOCK_NAME).exists(), "another job's record must survive"
    _release_scenario_attempt(sim, "200")
    assert not (sim / _ATTEMPT_LOCK_NAME).exists()
    _release_scenario_attempt(sim, "200")  # absent record: no error


def test_state_query_prefers_squeue_then_falls_back_to_sacct():
    slurm = RecordingSlurm({"7": ["RUNNING"], "8": ["COMPLETED"], "9": [None]})
    assert _slurm_job_state("7", slurm) == "RUNNING"
    assert slurm.of("sacct") == []
    assert _slurm_job_state("8", slurm) == "COMPLETED"
    assert slurm.of("sacct") == [["sacct", "-X", "-n", "-j", "8", "-o", "State%20"]]
    assert _slurm_job_state("9", slurm) is None


def test_state_query_strips_sacct_decoration_and_reports_unknown_on_failure():
    def cancelled_by(cmd):
        return (0, "") if cmd[0] == "squeue" else (0, "CANCELLED by 1234\n")

    assert _slurm_job_state("7", cancelled_by) == "CANCELLED"

    def broken(cmd):
        return 1, ""

    assert _slurm_job_state("7", broken) == "UNKNOWN"


def test_unknown_state_is_not_treated_as_gone(tmp_path):
    sim = tmp_path / "sim"
    _write_lock(sim, "100")
    message = _acquire(sim, "200", lambda cmd: (1, ""), wait_s=10.0)
    assert message is not None and "100" in message
    assert _holder_of(sim) == "100"
