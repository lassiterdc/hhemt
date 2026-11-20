"""A system that does not run the coupled model neither writes the coupled cfg nor judges the TRITON build for it.

Change 1118-tk1. A standalone-SWMM bundle lists open-water outfalls and has no TRITON checkout, because nothing
builds a solver for it. ``prepare_scenario`` called the coupled cfg generator unconditionally, and the generator's
``driven_stage`` branch refused the missing build, so every prepare job of such a bundle failed. These tests are
one-time tests of that change: they guard the behaviour of ``hhemt.scenario.TRITONSWMM_scenario`` and are to be
promoted into the scenario module tests or deleted at the landing. None compiles or runs a solver; every
collaborator is a stand-in.
"""

from __future__ import annotations

import ast
import inspect
import re
import textwrap
from types import SimpleNamespace

import pytest

from hhemt import scenario as scenario_module
from hhemt.exceptions import ConfigurationError
from hhemt.scenario import TRITONSWMM_scenario
from tests.test_scenario_boundary_guards import _scenario_for_coupled_cfg

# A guard is a call, inside the coupled cfg generator, of a module function that judges the TRITON build or the
# DEM rim. The population is read from the generator's own source, so a guard added later joins it unasked.
_GUARD_NAME = re.compile(r"^(require_\w*triton\w*|check_dem_rim_against_swmm_depth)$")


def _guards_called_by_the_coupled_cfg_generator() -> list[str]:
    """Names of the guards ``_generate_TRITON_SWMM_cfg`` calls, each once, in source order."""
    source = textwrap.dedent(inspect.getsource(TRITONSWMM_scenario._generate_TRITON_SWMM_cfg))
    names = [
        node.func.id
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and _GUARD_NAME.match(node.func.id)
    ]
    return list(dict.fromkeys(names))


def _every_guard_is_reachable_config(cfg_analysis) -> None:
    """Set the fields that make each guard's trigger hold, so a coupled system reaches all of them."""
    cfg_analysis.swmm_node_rim_from_dem = True
    cfg_analysis.dem_rim_resolution = "processed"
    cfg_analysis.manhole_table = "table.csv"
    cfg_analysis.swmm_open_water_outfalls = ["OF1"]
    cfg_analysis.open_water_outfall_representation = "driven_stage"


def _uncoupled(scen, *, toggle_tritonswmm_model: bool):
    """Mark the stand-in system as a standalone-SWMM system (or a coupled one) and return the scenario."""
    cfg_system = scen._system.cfg_system
    cfg_system.toggle_tritonswmm_model = toggle_tritonswmm_model
    cfg_system.toggle_triton_model = False
    cfg_system.toggle_swmm_model = True
    return scen


def _spy_on_every_guard(monkeypatch) -> list[str]:
    """Replace each guard by a recorder; the list it fills names the guards that ran, in order."""
    ran: list[str] = []
    for name in _guards_called_by_the_coupled_cfg_generator():

        def recorder(*args, _name=name, **kwargs):
            """Record one guard call; the DEM-rim check reports one failing node so the exemption guard is reached."""
            ran.append(_name)
            return ["N1"] if _name == "check_dem_rim_against_swmm_depth" else None

        monkeypatch.setattr(scenario_module, name, recorder)
    return ran


def test_the_guard_population_is_read_from_the_generator_and_is_not_empty():
    """Property: the population the absence test ranges over is derived from the generator and holds the guards the
    measured defect names.

    Class: the module functions ``_generate_TRITON_SWMM_cfg`` calls whose name matches the guard matcher
    (unit: distinct callee names; matcher: ``_GUARD_NAME``).
    Reachability: the generator's source is the one place its calls are listed; artifact: ``scenario.py``.
    Both-states anchor: reads the names of calls, present before and after the gate.
    Control (must fire): the driven-outfall guard of the measured failure is in the set, so the matcher is not
    returning an empty or unrelated set.

    Why required: an absence test over an empty population passes whatever the code does.
    What kills it: renaming the guards out of the matcher's reach, or moving them out of the generator.
    A second correct implementation: the generator calling its guards through one helper, with the matcher widened
    to name that helper; the test would then be recut by its owner.
    """
    population = _guards_called_by_the_coupled_cfg_generator()
    assert "require_driven_outfalls_capable_triton" in population, population
    assert len(population) >= 6, population


@pytest.mark.parametrize("coupled", [True, False], ids=["coupled-control", "uncoupled"])
def test_no_guard_of_the_coupled_cfg_runs_for_a_system_that_does_not_run_the_coupled_model(
    tmp_path, monkeypatch, coupled
):
    """Property: with ``toggle_tritonswmm_model`` False the coupled cfg generator calls none of the guards of the
    population, and with it True it calls every one of them.

    Class: NEW CAPABILITY. The population is every guard the generator calls (the previous test derives it);
    the plausible wrong implementation is gating one guard, for example only the driven-outfall one, which
    leaves the others to refuse the same system.
    Reachability: every guard's trigger holds in this configuration (DEM rim on, manhole table set, a driven
    outfall listed, the DEM-rim check reporting a failing node), and the coupled arm shows it: it runs all of them.
    Both-states anchor: reads which guards ran, a property of the control flow present before and after.
    Control (must fire): the coupled arm records every guard of the population, so the uncoupled arm's empty list
    is an absence and not a stand-in that never reached them.
    Invocation not isolatable from this tree; discrimination shown instead: deleting the gate reds the uncoupled
    arm with the full guard list.

    Why required: each guard refuses the missing build or rim data of a system that never reads them.
    What kills it: calling any guard before the gate, or gating only some of them.
    A second correct implementation: gating at the caller (``prepare_scenario``) instead of inside the generator,
    provided the test's call enters the generator through that caller; this test calls the generator directly,
    so it would then be recut to enter through ``prepare_scenario``, which the next test does.
    """
    scen, cfg = _scenario_for_coupled_cfg(tmp_path, monkeypatch, rim_from_dem=True, environment="native")
    _every_guard_is_reachable_config(scen._analysis.cfg_analysis)
    _uncoupled(scen, toggle_tritonswmm_model=coupled)
    ran = _spy_on_every_guard(monkeypatch)
    scen._generate_TRITON_SWMM_cfg()
    population = _guards_called_by_the_coupled_cfg_generator()
    if coupled:
        assert sorted(ran) == sorted(population), (ran, population)
    else:
        assert ran == [], ran
        assert not cfg.exists(), "an uncoupled system must not get a coupled cfg"


def test_the_coupled_system_still_refuses_a_driven_outfall_with_an_incapable_triton(tmp_path, monkeypatch):
    """Property: a coupled system that lists a driven outfall and has no capable TRITON build is refused, naming
    the build, exactly as before the gate.

    Class: REGRESSION on the coupled arm, held green: coupled systems (``toggle_tritonswmm_model`` True) whose
    software directory has no ``config_utils.h`` carrying the key read.
    Reachability: the guard's trigger is a listed driven outfall; artifact: ``require_driven_outfalls_capable_triton``.
    Both-states anchor: reads the refusal's text, which the measured failure carries.
    Held mutation that reds it: gating the generator on ``toggle_swmm_model`` instead, or on the outfall list.

    Why required: the gate must not become a way to skip the refusal for systems that do run the coupled model.
    What kills it: an early return that ignores the toggle's value.
    A second correct implementation: any gate that tests ``toggle_tritonswmm_model`` and nothing else.
    """
    scen, _cfg = _scenario_for_coupled_cfg(
        tmp_path, monkeypatch, rim_from_dem=False, environment="native", software_dir=tmp_path / "absent_triton"
    )
    scen._analysis.cfg_analysis.swmm_open_water_outfalls = ["OF1"]
    scen._analysis.cfg_analysis.open_water_outfall_representation = "driven_stage"
    _uncoupled(scen, toggle_tritonswmm_model=True)
    with pytest.raises(ConfigurationError, match="is not known to read it"):
        scen._generate_TRITON_SWMM_cfg()


def test_a_swmm_only_scenario_prepares_with_driven_outfalls_listed_and_no_triton_checkout(tmp_path, monkeypatch):
    """Property: preparing a standalone-SWMM scenario that lists open-water outfalls, with no TRITON checkout,
    completes without a refusal and writes no coupled cfg.

    Class: REGRESSION, red on the pre-change tree with ``ConfigurationError ... is not known to read it``:
    ``prepare_scenario`` runs for a system with ``toggle_swmm_model`` True and both TRITON toggles False, a
    ``driven_stage`` outfall list and a software directory that does not exist (the bundle ``phase_b_cpu``).
    Reachability: ``prepare_scenario`` calls the generator for every system; artifact: the traceback of the
    measured failure, ``scenario.py`` ``prepare_scenario`` -> ``_generate_TRITON_SWMM_cfg``.
    Both-states anchor: reads whether the call raises and whether the cfg file exists, not wording.
    Invocation not isolatable from this tree; discrimination shown instead: restoring the unconditional call
    reds this test with the refusal.

    Why required: the whole bundle cannot prepare a scenario otherwise.
    What kills it: any guard of the coupled generator running for the uncoupled system.
    A second correct implementation: gating in ``prepare_scenario`` and leaving the generator ungated.
    """
    scen, cfg = _scenario_for_coupled_cfg(
        tmp_path, monkeypatch, rim_from_dem=False, environment="native", software_dir=tmp_path / "absent_triton"
    )
    scen._analysis.cfg_analysis.swmm_open_water_outfalls = ["OF1"]
    scen._analysis.cfg_analysis.open_water_outfall_representation = "driven_stage"
    _uncoupled(scen, toggle_tritonswmm_model=False)
    scen._system.cfg_system.toggle_use_swmm_for_hydrology = False
    scen.event_iloc = 0
    scen.backend = "cpu"
    scen._verify_native_build_or_skip_in_container = lambda: None
    scen._write_sim_weather_nc = lambda: None
    scen._runoff_modeler = SimpleNamespace(
        write_swmm_rainfall_dat_files=lambda: None, write_swmm_waterlevel_dat_files=lambda: None
    )
    scen._input_generator = SimpleNamespace(
        create_hydraulics_model=lambda path: None, create_external_boundary_condition_files=lambda: None
    )
    scen._full_model_builder = SimpleNamespace(create_full_model=lambda path: None)
    scen._link_native_builds_into_sim = lambda: None
    scen.scen_paths.swmm_full_inp = tmp_path / "full.inp"
    monkeypatch.setattr(scenario_module, "refuse_non_cms_flow_units", lambda paths: None)
    monkeypatch.setattr(scenario_module, "_assert_scenario_forcing_window_agreement", lambda scenario: None)
    scen.prepare_scenario()
    assert not cfg.exists(), "a standalone-SWMM system must not get a coupled cfg"
