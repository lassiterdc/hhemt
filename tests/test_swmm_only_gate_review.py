"""Review tests of change 1118-tk1: a system that does not run the coupled model neither writes the coupled cfg nor
judges the TRITON build or the DEM rim for it, and a coupled system still does both.

The reviewing seat wrote these against commit 31a7adc2 without reading the coder's list of guards. The population of
guards is read from ``scenario.py`` itself, by a call-graph closure (not a name pattern) from the two functions that
open the TRITON build's sources, so a guard added later joins every test below unasked. None compiles or runs a
solver; every collaborator is a stand-in, so the file runs on a laptop.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from hhemt import scenario as scenario_module
from hhemt.exceptions import ConfigurationError
from tests.test_dem_rim_validation import _ok_cells
from tests.test_dem_rim_validation import _scenario as _rim_scenario
from tests.test_scenario_boundary_guards import _scenario_for_coupled_cfg

#: The two functions that open the TRITON build's ``src/config_utils.h`` (or the container's recorded commit).
_PRIMITIVES = frozenset({"require_triton_reading_key", "triton_coupling_read"})
_SRC = Path(scenario_module.__file__).parent


def _module_functions(path: Path) -> tuple[dict[str, ast.FunctionDef], ast.Module]:
    tree = ast.parse(path.read_text())
    return {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}, tree


def _called_names(node: ast.AST) -> set[str]:
    """Names called as plain functions (``f(...)``) anywhere under ``node``."""
    return {c.func.id for c in ast.walk(node) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}


def _build_readers() -> frozenset[str]:
    """Every module-level function of ``scenario.py`` that reaches a primitive, by call-graph closure."""
    functions, _ = _module_functions(Path(scenario_module.__file__))
    readers = set(_PRIMITIVES)
    while True:
        grown = {name for name, fn in functions.items() if name not in readers and _called_names(fn) & readers}
        if not grown:
            return frozenset(readers)
        readers |= grown


def _guards() -> frozenset[str]:
    """The readers that are not the primitives: the guards a caller invokes."""
    return _build_readers() - _PRIMITIVES


def _spy_on(monkeypatch, names) -> list[str]:
    """Replace each named reader of the scenario module by a recorder; the list it fills is what ran, in order."""
    ran: list[str] = []
    for name in names:

        def recorder(*args, _name=name, **kwargs):
            """Record one call; the DEM-rim check reports a failing node so the exemption guard is reached."""
            ran.append(_name)
            return ["N1"] if _name == "check_dem_rim_against_swmm_depth" else None

        monkeypatch.setattr(scenario_module, name, recorder)
    return ran


def _configure(scen, *, tritonswmm: bool, triton: bool = False, representation: str = "driven_stage") -> None:
    """Make every guard's trigger hold and set the toggles that say whether the system runs the coupled model."""
    analysis = scen._analysis.cfg_analysis
    analysis.swmm_node_rim_from_dem = True
    analysis.dem_rim_resolution = "processed"
    analysis.manhole_table = "table.csv"
    analysis.swmm_open_water_outfalls = ["OF1"]
    analysis.open_water_outfall_representation = representation
    system = scen._system.cfg_system
    system.toggle_tritonswmm_model = tritonswmm
    system.toggle_triton_model = triton
    system.toggle_swmm_model = True
    system.toggle_use_swmm_for_hydrology = False


def _make_preparable(scen, tmp_path, monkeypatch) -> None:
    """Stand-ins for every collaborator of ``prepare_scenario`` that is not under review."""
    scen.event_iloc = 0
    scen.backend = "cpu"
    scen._verify_native_build_or_skip_in_container = lambda: None
    scen._write_sim_weather_nc = lambda: None
    scen._link_native_builds_into_sim = lambda: None
    scen._runoff_modeler = SimpleNamespace(
        write_swmm_rainfall_dat_files=lambda: None, write_swmm_waterlevel_dat_files=lambda: None
    )
    scen._input_generator = SimpleNamespace(
        create_hydraulics_model=lambda path: None,
        create_external_boundary_condition_files=lambda: None,
        update_hydraulics_model_to_have_1_inflow_node_per_DEM_gridcell=lambda verbose=False: None,
    )
    scen._full_model_builder = SimpleNamespace(create_full_model=lambda path: None)
    scen.scen_paths.swmm_full_inp = tmp_path / "full.inp"
    monkeypatch.setattr(scenario_module, "refuse_non_cms_flow_units", lambda paths: None)
    monkeypatch.setattr(scenario_module, "_assert_scenario_forcing_window_agreement", lambda scenario: None)


def test_the_guard_census_is_a_call_graph_closure_and_every_call_site_is_the_gated_generator():
    """Property: every call of a function that reads the TRITON build, in the whole package, sits in the one method
    the gate covers, or in the run-entry check that already filters to coupled systems.

    Class: every ``def`` of ``scenario.py`` that reaches ``require_triton_reading_key`` or ``triton_coupling_read``
    through any chain (unit: function names; matcher: the closure in ``_build_readers``), and every ``.py`` of
    ``src/hhemt`` that names one of them as a call (unit: file and enclosing function).
    Reachability: the closure is read from the live source, so a method added tomorrow that calls a guard changes
    the census and reds this test; artifact: ``scenario.py``, ``analysis.py``.
    Both-states anchor: names of calls and of enclosing functions, present before and after the gate.
    Control (must fire): the census holds the six guards of the measured defect and both primitives.

    Why required: the gate sits at the head of one method; a second site that reads the build, outside it, would
    refuse a SWMM-only system exactly as the measured failure did and no test of the gated method would see it.
    What kills it: a guard called from ``prepare_scenario`` or from another method, ungated.
    A second correct implementation: moving the gate to ``prepare_scenario`` and leaving the generator ungated,
    which reds the next test and keeps this one green.
    """
    readers = _build_readers()
    assert {
        "require_driven_outfalls_capable_triton",
        "require_manhole_table_capable_triton",
        "require_dem_rim_capable_triton",
        "require_dem_rim_exempt_capable_triton",
        "require_triton_couples_the_listed_nodes",
        "check_dem_rim_against_swmm_depth",
    } <= readers, sorted(readers)
    assert _PRIMITIVES <= readers
    _, tree = _module_functions(Path(scenario_module.__file__))
    methods = {
        method.name
        for cls in tree.body
        if isinstance(cls, ast.ClassDef)
        for method in cls.body
        if isinstance(method, ast.FunctionDef) and _called_names(method) & readers
    }
    assert methods == {"_generate_TRITON_SWMM_cfg"}, methods
    sites: set[str] = set()
    for path in sorted(_SRC.rglob("*.py")):
        if path.name == "scenario.py":
            continue
        tree_other = ast.parse(path.read_text())
        for fn in ast.walk(tree_other):
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and _called_names(fn) & readers:
                sites.add(f"{path.name}::{fn.name}")
    assert sites == {"analysis.py::_refuse_coupling_the_triton_build_would_not_read"}, sites
    analysis_source = (_SRC / "analysis.py").read_text()
    body = analysis_source.split("def _refuse_coupling_the_triton_build_would_not_read")[1].split("\n    def ")[0]
    assert "_coupled_system_configs(self)" in body, "the run-entry check no longer filters to coupled systems"


@pytest.mark.parametrize(
    ("label", "representation"),
    [("swmm-only-driven", "driven_stage"), ("swmm-only-crown-rim", "crown_rim")],
)
def test_no_guard_reading_the_triton_build_runs_in_prepare_scenario_for_an_uncoupled_system(
    tmp_path, monkeypatch, label, representation
):
    """Property: through ``prepare_scenario``, a standalone-SWMM system calls none of the guards of the census and
    writes no coupled cfg; the coupled control calls every one it reaches. A TRITON-only system is deliberately not
    an arm here: whether it is gated is the routed question of the review, and a test of it would take a side.

    Class: NEW CAPABILITY over the census (``_guards``): two standalone-SWMM positions (a driven outfall, and a
    crown_rim outfall). The plausible wrong implementation is a gate on the driven-outfall guard alone, or on
    ``toggle_swmm_model``.
    Reachability: every guard's trigger holds in this configuration (``_configure``), and the control arm
    shows it by recording every guard the representation reaches; artifact: ``_generate_TRITON_SWMM_cfg``.
    Both-states anchor: reads which guards ran and whether a file exists, not wording.
    Control (must fire): the same stand-in with the toggle True records every guard of the census.
    Invocation not isolatable from this tree; discrimination shown instead: deleting the gate reds each uncoupled
    arm with every reachable guard in ``ran``.

    Why required: each guard refuses the missing build or rim data of a system that never reads them.
    What kills it: gating one guard, gating on the wrong toggle, or calling a guard before the gate.
    A second correct implementation: gating in ``prepare_scenario`` instead of the generator.
    """
    guards = set(_guards())
    if representation == "crown_rim":
        guards.discard("require_driven_outfalls_capable_triton")  # a crown_rim list reaches no driven-outfall guard
    ran_by_arm: dict[bool, list[str]] = {}
    for coupled in (False, True):
        arm = tmp_path / ("coupled" if coupled else "uncoupled")
        arm.mkdir()
        scen, cfg = _scenario_for_coupled_cfg(arm, monkeypatch, rim_from_dem=True, environment="native")
        _configure(scen, tritonswmm=coupled, representation=representation)
        _make_preparable(scen, arm, monkeypatch)
        ran_by_arm[coupled] = _spy_on(monkeypatch, sorted(_build_readers()))
        scen.prepare_scenario()
        if coupled:
            assert cfg.exists()
        else:
            assert not cfg.exists(), f"{label}: an uncoupled system must not get a coupled cfg"
    assert ran_by_arm[False] == [], ran_by_arm[False]
    assert set(ran_by_arm[True]) == guards, (ran_by_arm[True], sorted(guards))


@pytest.mark.parametrize("position", ["absent-directory", "clone-without-the-key", "container"])
def test_a_coupled_system_still_refuses_a_driven_outfall_it_cannot_read_and_an_uncoupled_one_prepares(
    tmp_path, monkeypatch, position
):
    """Property: with the same listed driven outfall, ``prepare_scenario`` refuses a coupled system whose TRITON
    build is not known to read the key, in three different incapable states, and completes for the uncoupled one.

    Class: REGRESSION on the coupled arm, held green, and the measured defect on the uncoupled arm, over three
    incapable positions: no clone at the configured directory (the defect's own state), a clone whose
    ``config_utils.h`` does not carry the key, and a container run (no image carries the key).
    Reachability: ``require_driven_outfalls_capable_triton`` is reached by any coupled system listing a driven
    outfall; artifact: ``scenario.py`` ``_generate_TRITON_SWMM_cfg``.
    Both-states anchor: reads whether the call raises ``ConfigurationError`` naming ``driven_stage``, and whether
    the cfg file exists.
    Held mutation that reds it: a gate that tests ``toggle_swmm_model`` or the outfall list instead of
    ``toggle_tritonswmm_model``, or an early return that ignores the toggle's value.

    Why required: the gate must not become a way to skip the refusal for systems that do run the coupled model.
    What kills it: an unconditional return; a gate keyed on anything but the coupled toggle.
    A second correct implementation: any gate that tests ``toggle_tritonswmm_model`` and nothing else.
    """
    environment = "container" if position == "container" else "native"
    software = tmp_path / "absent"
    if position == "clone-without-the-key":
        software = tmp_path / "clone"
        (software / "src").mkdir(parents=True)
        (software / "src" / "config_utils.h").write_text('argsd("swmm_coupling_nodes", argmap, "");\n')
    for coupled in (True, False):
        arm = tmp_path / ("coupled" if coupled else "uncoupled")
        arm.mkdir()
        scen, cfg = _scenario_for_coupled_cfg(
            arm, monkeypatch, rim_from_dem=False, environment=environment, software_dir=software
        )
        _configure(scen, tritonswmm=coupled)
        scen._analysis.cfg_analysis.swmm_node_rim_from_dem = False
        scen._analysis.cfg_analysis.manhole_table = None
        _make_preparable(scen, arm, monkeypatch)
        if coupled:
            with pytest.raises(ConfigurationError, match="driven_stage"):
                scen.prepare_scenario()
        else:
            scen.prepare_scenario()
            assert not cfg.exists()


def test_the_dem_rim_check_still_refuses_a_coupled_node_and_an_uncoupled_system_reads_no_dem(tmp_path, monkeypatch):
    """Property: the DEM-rim check landed in 2f520c9d still stops a coupled system whose node cell is below SWMM's
    depth, a standalone-SWMM system neither refuses nor reads the DEM, and a TRITON-only one writes its cfg unjudged.

    Class: the node A of ``test_dem_rim_validation`` lowered 0.01 m under SWMM's depth (a DEM that fails the
    check), judged for a coupled system (refused, naming A) and for an uncoupled one (no refusal, no cfg, and the
    processed DEM file deleted so that any read would raise).
    Reachability: ``swmm_node_rim_from_dem`` True with ``dem_rim_resolution`` processed writes the key, which
    makes the check run; artifact: ``check_dem_rim_against_swmm_depth``.
    Both-states anchor: reads whether the call raises and which node it names, not wording.
    Held mutation that reds the coupled arm: a gate that returns before the check for every system.

    Why required: the check's whole value is that it runs for the systems that use the DEM rim.
    What kills it: an early return that ignores the toggle's value.
    A second correct implementation: moving the check behind the toggle test in its own body (it already tests it).
    """
    cells = {**_ok_cells(), "A": 1.0 + 0.5 - 0.01}
    scen, cfg, _ = _rim_scenario(tmp_path / "coupled", monkeypatch, cells)
    scen._system.cfg_system.toggle_tritonswmm_model = True
    with pytest.raises(ConfigurationError) as caught:
        scen._generate_TRITON_SWMM_cfg()
    assert re.findall(r"^    - (\w+):", str(caught.value), flags=re.M) == ["A"]
    assert not cfg.exists()
    scen, cfg, root = _rim_scenario(tmp_path / "swmm_only", monkeypatch, cells)
    scen._system.cfg_system.toggle_tritonswmm_model = False
    scen._system.cfg_system.toggle_triton_model = False
    (root / "system" / "elevation_10.00m.dem").unlink()
    scen._generate_TRITON_SWMM_cfg()
    assert not cfg.exists()
    # A TRITON-only system keeps its coupled cfg (commit 495bf92b pins it) and is not judged for the rim.
    scen, cfg, _ = _rim_scenario(tmp_path / "triton_only", monkeypatch, cells)
    scen._system.cfg_system.toggle_tritonswmm_model = False
    scen._system.cfg_system.toggle_triton_model = True
    scen._generate_TRITON_SWMM_cfg()
    assert cfg.is_file()


def test_a_swmm_only_scenario_with_no_outfalls_and_the_key_off_prepares_without_a_coupled_cfg(tmp_path, monkeypatch):
    """Property: a SWMM-only scenario that lists nothing and sets no DEM-rim key prepares, and writes no coupled cfg.

    Class: SWMM-only systems with every optional key off, the arm of the bundle the defect did not touch.
    Reachability: ``prepare_scenario`` runs for every system; artifact: ``scenario.py`` ``prepare_scenario``.
    Both-states anchor: reads whether preparation completes and whether the cfg file exists.
    Held mutation that reds it: a gate that raises for an uncoupled system, or any change that makes the generator
    write the coupled cfg for it.

    Why required: an arm the defect never touched must keep preparing, and a standalone-SWMM run reads no coupled
    cfg (its solver command takes the ``.inp`` file, ``run_simulation.py`` model type ``swmm``).
    What kills it: a gate that refuses; an unconditional cfg write.
    A second correct implementation: writing the cfg and not judging it, which reds the second assertion and would
    need the criterion's "do not run the coupled-config generation" read as "do not judge".
    """
    scen, cfg = _scenario_for_coupled_cfg(tmp_path, monkeypatch, rim_from_dem=False, environment="native")
    scen._analysis.cfg_analysis.swmm_open_water_outfalls = []
    scen._analysis.cfg_analysis.manhole_table = None
    scen._analysis.cfg_analysis.dem_rim_resolution = None
    system = scen._system.cfg_system
    system.toggle_tritonswmm_model = False
    system.toggle_triton_model = False
    system.toggle_swmm_model = True
    system.toggle_use_swmm_for_hydrology = False
    _make_preparable(scen, tmp_path, monkeypatch)
    scen.prepare_scenario()
    assert not cfg.exists()
