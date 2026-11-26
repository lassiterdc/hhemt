"""Reviewer tests for the per-scenario attempt guard (chunk 1g.1 repair, G39).

Real Slurm command output, not the coder's stub's idealised output:
- `sacct -o State` is 10 columns wide and ends a longer name with `+` (OUT_OF_MEMORY -> `OUT_OF_ME+`).
- A job that Slurm lists as NODE_FAIL can still have a live process on the lost node: the
  measured incident this repair exists for.
"""

from __future__ import annotations

import json

from hhemt.run_simulation_runner import _ATTEMPT_LOCK_NAME, _acquire_scenario_attempt


class _Clock:
    def __init__(self):
        self.now = 0.0

    def clock(self):
        return self.now

    def sleep(self, s):
        self.now += s


def _lock(sim, jobid):
    sim.mkdir(parents=True, exist_ok=True)
    (sim / _ATTEMPT_LOCK_NAME).write_text(json.dumps({"slurm_jobid": jobid, "node": "n1"}))


def _acquire(sim, run_cmd):
    ft = _Clock()
    return _acquire_scenario_attempt(
        sim,
        "200",
        "node7",
        scenario_label="year.0/tritonswmm (event_iloc=0)",
        run_cmd=run_cmd,
        sleep=ft.sleep,
        clock=ft.clock,
        wait_s=30.0,
        poll_s=5.0,
    )


def _holder(sim):
    return json.loads((sim / _ATTEMPT_LOCK_NAME).read_text())["slurm_jobid"]


def test_out_of_memory_holder_as_sacct_truncates_it_is_terminal(tmp_path):
    """DEFECT: `sacct -X -n -j ID -o State` prints `OUT_OF_ME+`, which is not in the terminal
    set, so a holder that died of OOM is treated as alive, scancelled, waited for 30 s and
    then refused: a false refusal on a job that is gone."""

    def real_cmd(cmd):
        if cmd[0] == "squeue":  # job aged out of the controller
            return 1, ""
        if cmd[0] == "sacct":
            return 0, "OUT_OF_ME+\n"
        return 0, ""  # scancel

    sim = tmp_path / "sim"
    _lock(sim, "100")
    assert _acquire(sim, real_cmd) is None
    assert _holder(sim) == "200"


def test_node_fail_holder_cannot_be_shown_gone_so_the_guard_refuses(tmp_path):
    """DEFECT against the measured incident: job 23487260 was NODE_FAIL in Slurm's books while
    its process lived 3 more hours on the lost node. The guard sends a best-effort scancel and
    proceeds to start a writer, i.e. it does not prevent the diagnosed double write.
    A holder that cannot be shown gone must be refused with nothing written (criterion c)."""

    def real_cmd(cmd):
        if cmd[0] == "squeue":
            return 0, "NODE_FAIL\n"
        if cmd[0] == "sacct":
            return 0, " NODE_FAIL\n"
        return 0, ""

    sim = tmp_path / "sim"
    _lock(sim, "100")
    refusal = _acquire(sim, real_cmd)
    assert refusal is not None and "100" in refusal
    assert _holder(sim) == "100", "refusal must leave the holder's record and the directory untouched"


def test_completing_holder_after_scancel_refuses_and_writes_nothing(tmp_path):
    """Down node: scancel leaves the job in COMPLETING indefinitely; must refuse."""

    def real_cmd(cmd):
        if cmd[0] == "squeue":
            return 0, "COMPLETING\n"
        return 0, ""

    sim = tmp_path / "sim"
    _lock(sim, "100")
    refusal = _acquire(sim, real_cmd)
    assert refusal is not None and "COMPLETING" in refusal
    assert sorted(p.name for p in sim.iterdir()) == [_ATTEMPT_LOCK_NAME]
    assert _holder(sim) == "100"


def test_squeue_invalid_job_id_error_falls_back_to_sacct_terminal(tmp_path):
    """Real squeue for an aged-out id exits 1 with the error on stderr and nothing on stdout."""

    def real_cmd(cmd):
        if cmd[0] == "squeue":
            return 1, ""
        if cmd[0] == "sacct":
            return 0, "CANCELLED+\n"
        return 0, ""

    sim = tmp_path / "sim"
    _lock(sim, "100")
    assert _acquire(sim, real_cmd) is None
    assert _holder(sim) == "200"
