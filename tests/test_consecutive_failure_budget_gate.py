"""The durable consecutive-failure budget gate.

The property this file asserts: *the consecutive-failure budget refuses the (N+1)-th launch and the counter is
era-relative rather than cumulative.*

WHY THE PROPERTY IS REQUIRED, as a consequence a reader can check against the code.
Snakemake's ``retries:`` is compared against ``job.attempt``, which lives on an in-memory
job object initialised to 1 when a Snakemake process builds its DAG and is persisted
nowhere. Under ``--executor slurm`` at least two such processes exist per rule execution,
so the declared budget restarts at each nesting level and the realised ceiling is their
PRODUCT. Measured on member 39 of the a6000 benchmarking arm: a declared ``retries: 3``
produced TEN solver launches, every one re-attempting the same deterministic refusal.
A gate in the RUNNER is the only available additive bound, because the runner is the only
party that both persists state across those processes and sits upstream of the solver
launch.

THE INVARIANT, stated once: a simulation whose failures are DETERMINISTIC is launched at
most ``hpc_consecutive_failure_budget_simulate`` times per era, while a simulation that is
merely INCOMPLETE -- the walltime-killed, hotstart-resuming case the developer armed
``hpc_restart_times_simulate`` for -- is never refused however many times
it resumes.

These tests exercise ``_consecutive_failure_budget_refusal`` and
``_bump_terminal_failures`` against a REAL ``TRITONSWMM_model_log`` on a tmp path, so the
int LogField validator and serializer the budget added are exercised too. Nothing here
compiles or runs a solver, so the file is not simulation-bearing and runs locally.
"""

import pytest

from hhemt.log import TRITONSWMM_model_log
from hhemt.run_simulation_runner import (
    _bump_terminal_failures,
    _consecutive_failure_budget_refusal,
)

BUDGET = 3


class _StubScenario:
    """Minimal ``scenario`` surface the two helpers touch: ``get_log(model_type)``.

    Deliberately returns the SAME log object rather than a fresh one per call. The
    production ``scenario.get_log`` returns a fresh ``from_json`` instance every time
    precisely because ``LogField.set()`` auto-writes the whole log; holding one instance
    here keeps the test's subject the GATE ARITHMETIC rather than the persistence
    machinery, which ``test_counter_round_trips_through_the_log_file`` covers separately
    and on disk.
    """

    def __init__(self, log):
        self._log = log

    def get_log(self, model_type):
        return self._log


@pytest.fixture
def log(tmp_path):
    return TRITONSWMM_model_log(logfile=str(tmp_path / "log_tritonswmm.json"))


def test_legacy_log_coalesces_to_zero_and_does_not_refuse(log):
    """An unset counter is a legacy log, and a legacy log is not a spent budget.

    Kills: making the gate read ``consecutive_terminal_failures.get()`` without the
    ``or 0`` coalesce, which would raise or compare None against an int and refuse (or
    crash on) every pre-existing analysis tree on its first post-upgrade launch.
    """
    assert log.consecutive_terminal_failures.get() is None
    assert _consecutive_failure_budget_refusal(_StubScenario(log), "tritonswmm", BUDGET, 0) is None


def test_refuses_only_the_budget_plus_first_launch(log):
    """``budget`` terminal failures are permitted; the (budget+1)-th launch is refused.

    This is the DoD's second clause -- *it fails terminally ``budget+1`` times and IS
    refused on the last*. The input class quantified over is every
    ``(member, event, model_type)`` whose prior consecutive terminal-failure count is
    ``>= budget``; the loop walks the boundary from below rather than asserting a single
    point, so an off-by-one in either direction fails here.

    Kills: changing the gate's comparison to ``>`` (which would permit budget+1 failures
    and refuse only at budget+2) or to launch-count rather than prior-failure semantics.
    """
    scen = _StubScenario(log)
    for attempt in range(1, BUDGET + 1):
        assert _consecutive_failure_budget_refusal(scen, "tritonswmm", BUDGET, 0) is None, (
            f"attempt {attempt} of {BUDGET} must be permitted"
        )
        _bump_terminal_failures(scen, "tritonswmm")
        assert log.consecutive_terminal_failures.get() == attempt

    refusal = _consecutive_failure_budget_refusal(scen, "tritonswmm", BUDGET, 0)
    assert refusal is not None, f"launch {BUDGET + 1} must be refused after {BUDGET} terminal failures"
    assert "hpc_consecutive_failure_budget_simulate" in refusal, (
        "the refusal must name the knob an operator would change"
    )


def test_a_completing_attempt_discharges_every_failure_before_it(log):
    """The counter is CONSECUTIVE: a completion resets it, so the budget is not cumulative.

    Input class: any member that failed terminally one or more times and then completed.
    Kills: incrementing without ever resetting -- i.e. a CUMULATIVE counter, which would
    refuse a member that has failed ``budget`` times across the whole life of the
    analysis even though it has since completed and is healthy.
    """
    scen = _StubScenario(log)
    for _ in range(BUDGET):
        _bump_terminal_failures(scen, "tritonswmm")
    assert _consecutive_failure_budget_refusal(scen, "tritonswmm", BUDGET, 0) is not None

    # The production reset, as run_simulation_runner performs it on the success path.
    log.consecutive_terminal_failures.set(0)

    assert _consecutive_failure_budget_refusal(scen, "tritonswmm", BUDGET, 0) is None


def test_a_resume_sweep_is_never_refused_however_many_times_it_resumes(log):
    """The DoD's first clause: a sim resuming successfully more than ``budget`` times is NOT refused.

    This is the test that separates this counter from ``n_resumes`` and it is the one the
    shape names as the single likeliest way an implementer gets this wrong. ``n_resumes``
    counts SUCCESSFUL resumes and is cumulative across force-rerun eras; a gate keyed on it
    would refuse exactly the healthy long-running simulations the retries exist to serve.

    The input class is every walltime-killed-then-resumed attempt. A walltime kill is a
    SIGKILL, so it reaches NO Python-side failure return and therefore calls
    ``_bump_terminal_failures`` nowhere -- which is modelled here by advancing
    ``n_resumes`` well past the budget while leaving the terminal-failure counter alone.

    Kills: keying the gate on ``n_resumes``, or incrementing the terminal-failure counter
    at runner ENTRY instead of on the failure paths. Either change reds this test, and
    either change would have refused member 38's legitimate 20-resume sweep.
    """
    scen = _StubScenario(log)
    log.n_resumes.set(BUDGET * 7)

    assert _consecutive_failure_budget_refusal(scen, "tritonswmm", BUDGET, 0) is None
    assert log.consecutive_terminal_failures.get() in (None, 0)


def test_the_counter_is_era_relative_and_a_forced_rerun_opens_a_new_era(log):
    """A spent budget plus ``force_rerun_pending`` does NOT refuse, and the counter resets.

    Input class: every member a ``stage="simulate"`` force-rerun names whose terminal-failure
    count has already reached the budget. A forced re-run exists to escape a stuck state, so
    a gate that honoured the previous era's count would refuse the very attempt the force was
    issued to permit -- and the operator's one escape hatch would be inert exactly when it is
    needed.

    This also pins WHERE the reset lives. The production consume of ``force_rerun_pending``
    happens after ``Popen``; a reset sited only there is too late, because the gate runs
    before the launch. Kills: moving the era reset out of the gate.
    """
    scen = _StubScenario(log)
    for _ in range(BUDGET):
        _bump_terminal_failures(scen, "tritonswmm")
    assert _consecutive_failure_budget_refusal(scen, "tritonswmm", BUDGET, 0) is not None

    log.force_rerun_pending.set(True)

    assert _consecutive_failure_budget_refusal(scen, "tritonswmm", BUDGET, 0) is None
    assert log.consecutive_terminal_failures.get() == 0, (
        "the era reset must ZERO the counter, not merely bypass the comparison -- "
        "a bypass would refuse again on the next attempt once the marker is consumed"
    )


def test_budget_zero_disables_the_gate(log):
    """``budget <= 0`` is the documented disable, and it must not even read the log.

    Input class: every analysis whose operator set the knob to 0. This is the arm that
    preserves byte-identical pre-gate behaviour, so it is asserted rather than assumed.
    """
    scen = _StubScenario(log)
    for _ in range(BUDGET * 5):
        _bump_terminal_failures(scen, "tritonswmm")
    assert _consecutive_failure_budget_refusal(scen, "tritonswmm", 0, 0) is None


def test_counter_round_trips_through_the_log_file(tmp_path):
    """The counter is DURABLE across processes, which is the whole point of the mechanism.

    Input class: every (member, event, model_type) whose attempts span more than one
    Snakemake process -- which under ``--executor slurm`` is all of them. An in-memory
    counter would satisfy every other test in this file and close none of the defect,
    because the multiplication this gate bounds happens BETWEEN processes.

    Kills: holding the count on the runner's own stack or in a module global, and
    omitting ``consecutive_terminal_failures`` from the int validator/serializer groups
    (the value would then not survive the JSON round trip).
    """
    logfile = tmp_path / "log_tritonswmm.json"
    first = TRITONSWMM_model_log(logfile=str(logfile))
    _bump_terminal_failures(_StubScenario(first), "tritonswmm")
    _bump_terminal_failures(_StubScenario(first), "tritonswmm")
    assert first.consecutive_terminal_failures.get() == 2

    reloaded = TRITONSWMM_model_log.from_json(logfile)
    assert reloaded.consecutive_terminal_failures.get() == 2, (
        "the count must survive the log round trip, or it cannot bound a ceiling that "
        "multiplies across Snakemake processes"
    )
