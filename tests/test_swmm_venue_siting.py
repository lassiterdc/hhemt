"""The venue guard is WIRED IN, and wired in BEFORE the engine is constructed.

WHY THIS EXISTS SEPARATELY FROM test_compile_venue.py. That module proves the
PREDICATE decides correctly. It cannot prove the predicate is CALLED, or called
at the right point. Those are different failures with the same symptom -- SWMM
runs when it should not -- and only one of them is covered by a predicate test.

This is a Gate A unit test: no toolchain, no solver, no SWMM engine. It patches
the predicate rather than the environment, so it asserts WIRING and stays
independent of how the predicate itself decides.
"""

from types import SimpleNamespace

import pytest

import hhemt.swmm_runoff_modeling as srm


class _NeverBuilt:
    """A stand-in for pyswmm.Simulation that fails the test if it is constructed."""

    constructed = False

    def __init__(self, *_args, **_kwargs):
        type(self).constructed = True

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def execute(self):
        pass


def _scenario_double(tmp_path):
    """The minimum SWMMRunoffModeler.__init__ and run_swmm_hydro_model actually read.

    `__init__` reads `scenario._analysis.cfg_analysis` and `scenario._system` in
    addition to the scenario itself; `run_swmm_hydro_model` reads the log entry
    and the .inp path. Nothing else on the real scenario is touched on this path,
    which is what makes a double viable here at all.
    """
    state = {"done": False}
    log = SimpleNamespace(
        hydro_swmm_sim_completed=SimpleNamespace(
            get=lambda: state["done"],
            set=lambda v: state.__setitem__("done", v),
        )
    )
    paths = SimpleNamespace(swmm_hydro_inp=tmp_path / "model.inp")
    return SimpleNamespace(
        log=log,
        scen_paths=paths,
        _analysis=SimpleNamespace(cfg_analysis=SimpleNamespace()),
        _system=SimpleNamespace(),
    )


@pytest.fixture(autouse=True)
def _reset_sentinel():
    _NeverBuilt.constructed = False
    yield
    _NeverBuilt.constructed = False


def test_refusal_fires_before_the_engine_is_constructed(monkeypatch, tmp_path):
    """Arm 1 — the guard is called, and called BEFORE Simulation(...)."""
    monkeypatch.setattr(srm, "_assert_validated_swmm_stack", lambda: None)
    monkeypatch.setattr(srm, "swmm_execution_refused", lambda: True)
    monkeypatch.setattr(srm, "Simulation", _NeverBuilt)

    modeler = srm.SWMMRunoffModeler(_scenario_double(tmp_path))
    with pytest.raises(RuntimeError, match="undeclared machine"):
        modeler.run_swmm_hydro_model()

    assert _NeverBuilt.constructed is False, (
        "the engine was constructed despite the guard refusing -- the guard is "
        "either not called or called AFTER Simulation(...)"
    )


def test_engine_is_reached_when_the_guard_permits(monkeypatch, tmp_path):
    """Arm 2 — the discriminating half: a guard wired to refuse unconditionally
    would pass arm 1 and fail here."""
    monkeypatch.setattr(srm, "_assert_validated_swmm_stack", lambda: None)
    monkeypatch.setattr(srm, "swmm_execution_refused", lambda: False)
    monkeypatch.setattr(srm, "Simulation", _NeverBuilt)

    modeler = srm.SWMMRunoffModeler(_scenario_double(tmp_path))
    modeler.run_swmm_hydro_model()

    assert _NeverBuilt.constructed is True, (
        "the engine was never constructed even though the guard permitted -- the guard refuses unconditionally"
    )


def test_the_guard_is_not_bypassed_by_the_already_run_short_circuit(monkeypatch, tmp_path):
    """Arm 3 — the branch that skips execution must also skip the guard.

    run_swmm_hydro_model returns early when the sim is already complete. That
    path constructs nothing, so it must not raise either: a guard placed above
    the short-circuit would refuse a run that was never going to execute.
    """
    monkeypatch.setattr(srm, "_assert_validated_swmm_stack", lambda: None)
    monkeypatch.setattr(srm, "swmm_execution_refused", lambda: True)
    monkeypatch.setattr(srm, "Simulation", _NeverBuilt)

    scenario = _scenario_double(tmp_path)
    scenario.log.hydro_swmm_sim_completed.set(True)
    modeler = srm.SWMMRunoffModeler(scenario)

    modeler.run_swmm_hydro_model()  # must not raise
    assert _NeverBuilt.constructed is False
