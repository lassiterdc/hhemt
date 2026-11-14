"""Launched-step-id recovery: query SLURM's record, refuse rather than guess.

The runner's own ``SLURM_STEP_ID`` is the jobstep plugin's srun (always ``.0``), so
recording it collapses every attempt of a resumed sim onto one ledger key
(``read_attempt_index_by_jobstep`` builds ``{jobid}.{step}``) and under-reports the
report's Attempts column, which counts KEYS in that map, to 1 for every resumed sim.

These pin the REFUSAL rules as much as the happy path, because the asymmetry is the whole
design: a None degrades to today's value, whereas a wrong step id silently mislabels an
attempt.
"""

import types

import pytest

from hhemt.run_simulation_runner import _parse_step_ids, _read_launched_step_id


def _analysis(run_method="batch_job"):
    return types.SimpleNamespace(cfg_analysis=types.SimpleNamespace(multi_sim_run_method=run_method))


# --- parser -----------------------------------------------------------------------


def test_parser_keeps_the_solver_step_and_drops_slurm_bookkeeping():
    """`.batch`/`.extern` are SLURM's own steps and `python` is the runner itself."""
    out = "18396677|ab9f4df0\n18396677.batch|batch\n18396677.0|python\n18396677.1|triton.exe\n"
    assert _parse_step_ids(out) == ["1"]


def test_parser_returns_every_solver_step_when_several_exist():
    """The 1_job_many_srun_tasks shape from the 18396677 probe: four solver steps."""
    out = (
        "18396677.batch|batch\n18396677.0|python\n"
        "18396677.1|triton.exe\n18396677.2|triton.exe\n"
        "18396677.3|triton.exe\n18396677.4|triton.exe\n"
    )
    assert _parse_step_ids(out) == ["1", "2", "3", "4"]


def test_parser_ignores_the_allocation_row_which_has_no_step_suffix():
    assert _parse_step_ids("18396677|ab9f4df0\n") == []


@pytest.mark.parametrize("junk", ["", "   \n\n", "garbage-with-no-pipe\n", "18396677.x|triton.exe\n"])
def test_parser_never_raises_on_malformed_output(junk):
    assert _parse_step_ids(junk) == []


# --- refusal rules ----------------------------------------------------------------


def test_no_slurm_job_id_is_a_no_op(monkeypatch):
    """Local and serial runs must be byte-identical to before this existed."""
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    assert _read_launched_step_id(_analysis(), "triton") is None


def test_shared_allocation_mode_refuses_without_querying(monkeypatch):
    """Under 1_job_many_srun_tasks N concurrent sims share one $SLURM_JOB_ID, so a
    solver-named step may belong to a different event -- the same srun-step aliasing that
    makes workflow.py's _aliased_jids guard refuse. It must not even run sacct."""
    monkeypatch.setenv("SLURM_JOB_ID", "18396677")

    def _boom(*a, **k):  # pragma: no cover - must never be reached
        raise AssertionError("sacct was queried despite a shared allocation")

    monkeypatch.setattr("subprocess.run", _boom)
    assert _read_launched_step_id(_analysis("1_job_many_srun_tasks"), "triton") is None


def test_multiple_solver_steps_refuse_rather_than_pick(monkeypatch):
    """A multi-match means the allocation cannot attribute a step to this attempt.
    Picking the highest would be plausible, not defensible; a wrong pick is the one
    outcome worse than no pick."""
    monkeypatch.setenv("SLURM_JOB_ID", "18396677")
    out = "18396677.0|python\n18396677.1|triton.exe\n18396677.2|triton.exe\n"
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **k: types.SimpleNamespace(returncode=0, stdout=out, stderr=""),
    )
    assert _read_launched_step_id(_analysis(), "triton") is None


def test_single_solver_step_is_recovered(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "18396677")
    out = "18396677.batch|batch\n18396677.0|python\n18396677.1|triton.exe\n"
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **k: types.SimpleNamespace(returncode=0, stdout=out, stderr=""),
    )
    assert _read_launched_step_id(_analysis(), "triton") == "1"


def test_nonzero_exit_yields_none_not_an_exception(monkeypatch):
    """Status is read from the actuator, and a failed query is an absence, never a raise --
    this is diagnostic metadata and must not fail a simulation that otherwise succeeded."""
    monkeypatch.setenv("SLURM_JOB_ID", "18396677")
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **k: types.SimpleNamespace(returncode=1, stdout="", stderr="sacct: error"),
    )
    assert _read_launched_step_id(_analysis(), "triton") is None


def test_subprocess_raising_yields_none_not_an_exception(monkeypatch):
    """No sacct on PATH, or a timeout, must also degrade rather than propagate."""
    monkeypatch.setenv("SLURM_JOB_ID", "18396677")

    def _raise(*a, **k):
        raise FileNotFoundError("sacct")

    monkeypatch.setattr("subprocess.run", _raise)
    assert _read_launched_step_id(_analysis(), "triton") is None
