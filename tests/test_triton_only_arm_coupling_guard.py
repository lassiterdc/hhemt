"""Tests that a TRITON-only scenario is not stopped by the coupling-node check it has no list for.

Problem under test. ``prepare_scenario`` writes the coupling-node list only for a coupled system
(``toggle_tritonswmm_model``), but it builds the coupled cfg for every system. That cfg's tail
judges the list against the model's ``[INFLOWS]`` section, and it opens the list file whenever the
TRITON build does not read ``swmm_coupling_nodes`` (a build the toolkit recognizes by its
``[INFLOWS]`` read). A TRITON-only system run on such a build therefore stopped scenario
preparation with ``FileNotFoundError`` on a file nobody had written, which the synthetic solver
matrix measured on two of its pins (``3a832f7d`` and ``b3820a44``). These tests drive the real
``prepare_scenario`` and the real cfg generator, and fake every collaborator that would reach a
solver, so none of them compiles or runs TRITON or SWMM.

The build identity is a container commit from the key-less set, written here as a literal, not
imported, so the module's own set is checked against an independently written value.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

#: A TRITON container commit whose coupled-node read is the key-less ``[INFLOWS]`` read (it does
#: not read ``swmm_coupling_nodes``); it is one of the two pins the matrix measured failing.
_KEYLESS_PIN = "b3820a448f304b3f732f4b6fac5564adf86ac333"

#: The hydraulics model the fake template step copies. Its ``[INFLOWS]`` section names ``J1``
#: once, so a build that reads ``[INFLOWS]`` couples exactly ``["J1"]``.
_HYDRAULICS = (
    "[OPTIONS]\nFLOW_UNITS CMS\n\n"
    "[INFLOWS]\n;;Node Constituent Time Series Type Mfactor Sfactor Baseline Pattern\n"
    ';;---- ----\nJ1 FLOW "" FLOW 1 1 0\n\n'
)


def _prepare(tmp_path, monkeypatch, *, coupled, written_list=None, one_per_cell=False):
    """Run the real ``prepare_scenario`` on a fake system; return ``(scenario, cfg path, list path)``.

    The real pieces are ``prepare_scenario``, the coupled cfg generator and the coupling check.
    Everything that would reach a solver, the weather, the SWMM runs or the native build is a
    fake. ``written_list`` is the text the fake coupled selection step writes as the coupling
    list; ``None`` writes nothing, which is what a TRITON-only system leaves behind. The
    selection step is only called for a coupled system, as in production.
    """
    from hhemt import scenario as scenario_module
    from tests.test_scenario_boundary_guards import _scenario_for_coupled_cfg

    scen, cfg = _scenario_for_coupled_cfg(
        tmp_path,
        monkeypatch,
        rim_from_dem=False,
        environment="container",
        branch_key=_KEYLESS_PIN,
        recorded_sha=_KEYLESS_PIN,
    )
    hydraulics = scen.scen_paths.swmm_hydraulics_inp
    listed = scenario_module.coupling_nodes_file_for(hydraulics)
    template = tmp_path / "hydraulics_template.inp"
    template.write_text(_HYDRAULICS)
    cfg_system = scen._system.cfg_system
    cfg_system.toggle_tritonswmm_model = coupled
    cfg_system.toggle_swmm_model = False
    cfg_system.toggle_use_swmm_for_hydrology = False
    cfg_system.SWMM_hydraulics = template
    scen._analysis.cfg_analysis.one_coupled_inflow_node_per_dem_cell = one_per_cell

    def create_hydraulics(source, dest):
        """Stand in for the template step: the model is copied unchanged."""
        Path(dest).write_text(Path(source).read_text())

    def select_coupling_nodes(verbose=False):
        """Stand in for the coupled selection step, which is what writes the list."""
        if written_list is not None:
            listed.write_text(written_list)

    scen.event_iloc = 0
    scen.backend = "cpu"
    scen._verify_native_build_or_skip_in_container = lambda: None
    scen._write_sim_weather_nc = lambda: None
    scen._runoff_modeler = SimpleNamespace(
        write_swmm_rainfall_dat_files=lambda: None,
        write_swmm_waterlevel_dat_files=lambda: None,
    )
    scen._input_generator = SimpleNamespace(
        create_hydraulics_model_from_template=create_hydraulics,
        create_external_boundary_condition_files=lambda: None,
        update_hydraulics_model_to_have_1_inflow_node_per_DEM_gridcell=select_coupling_nodes,
    )
    scen._generate_TRITON_cfg = lambda: None
    scen._link_native_builds_into_sim = lambda: None
    monkeypatch.setattr(scenario_module, "_assert_scenario_forcing_window_agreement", lambda scenario: None)
    cfg.parent.mkdir(parents=True, exist_ok=True)
    scen.prepare_scenario()
    return scen, cfg, listed


@pytest.mark.parametrize("one_per_cell", [False, True])
def test_a_triton_only_system_is_prepared_on_a_build_that_ignores_the_coupling_list(
    tmp_path, monkeypatch, one_per_cell
):
    """Property: a system with the coupled model off finishes preparation on a key-less build.

    Class: TRITON-only systems (``toggle_tritonswmm_model`` off), so no coupling list is
    written, on a TRITON build that does not read ``swmm_coupling_nodes``, with the
    one-node-per-cell option off and on. Reachability: the matrix's two failing TRITON-only
    arms (pins ``3a832f7d`` and ``b3820a44``) are exactly this input, and this test's own
    assertion that no list exists shows the input is the one that opened a missing file. Both
    states: the assertion turns on whether preparation raises for a system with no list file, a
    property of the old and the new code alike, not on any new wording. Why: the list is read
    only by the coupled run, so a system that never wrote it must not be refused for lacking it.
    What kills it: running the coupling check whatever the toggle, which raises
    ``FileNotFoundError`` for the off case and a ``ConfigurationError`` for the option-on case.
    Second correct implementation: passing the toggle from ``prepare_scenario`` into the cfg
    generator instead of reading it inside; the assertions read only whether preparation
    completed and what files exist. Invocation: the test builds its scenario and runs it in
    ``tmp_path``, so the tree the run resolves from does not decide it.
    """
    _, cfg, listed = _prepare(tmp_path, monkeypatch, coupled=False, one_per_cell=one_per_cell)

    assert not listed.exists()  # the premise: nothing wrote the list this run reads
    assert cfg.is_file()


def test_the_cfg_a_triton_only_system_writes_is_the_one_a_coupled_system_writes(tmp_path, monkeypatch):
    """Property: skipping the check changes no byte of the coupled cfg a TRITON-only system writes.

    Class: the same hydraulics model prepared once with the coupled model on (a matching
    coupling list written) and once with it off (no list), on the same key-less build. Why: the
    fix is a validation change only, so the cfg a system gets must not depend on whether the
    check ran; anything that differs would be a change to what the solver is given. What kills
    it: skipping the whole cfg generator for a TRITON-only system, or writing a different
    ``swmm_coupling_nodes`` line for it. Second correct implementation: judging the list through
    a flag passed from ``prepare_scenario``; the assertion compares the two files' lines with
    each run's own directory name replaced. Both states: the comparison reads the generator's
    output, which exists before and after the change (the off arm raised before writing the
    file only in the old code, and its absence there is what the first test reds on).
    """
    on_dir = tmp_path / "on"
    off_dir = tmp_path / "off"
    on_dir.mkdir()
    off_dir.mkdir()
    _, cfg_on, _ = _prepare(on_dir, monkeypatch, coupled=True, written_list="J1\n")
    _, cfg_off, _ = _prepare(off_dir, monkeypatch, coupled=False)

    on_lines = cfg_on.read_text().replace(str(on_dir), "ROOT").splitlines()
    off_lines = cfg_off.read_text().replace(str(off_dir), "ROOT").splitlines()
    assert on_lines == off_lines
    assert any(line.startswith("swmm_coupling_nodes=") for line in off_lines)


@pytest.mark.parametrize(
    ("written_list", "error", "match"),
    [
        (None, FileNotFoundError, "coupling_nodes.txt"),
        ("J2\n", "ConfigurationError", "does not read swmm_coupling_nodes"),
        ("J1\n", None, None),
    ],
    ids=["list-never-written", "list-disagrees-with-inflows", "list-agrees-with-inflows"],
)
def test_a_coupled_system_is_still_judged_against_its_coupling_list(tmp_path, monkeypatch, written_list, error, match):
    """Property: with the coupled model on, the coupling check still opens and judges the list.

    Class: coupled systems on a key-less build whose list is absent, disagrees with the
    model's ``[INFLOWS]`` section, or agrees with it. Why: the fix removes the check for a
    system that has no list, and must not weaken it for one that should have one; a coupled
    run whose list was never written is a defect, and the check is what reports it. What kills
    it: dropping the check for every system, or catching the missing-file error. Second
    correct implementation: moving the check into ``prepare_scenario`` under the toggle; the
    assertions read only the exception type and a phrase of its message, and that agreement
    passes. Held before and after the change: this test is green on the old code too, so it is
    a characterization of what must not move.
    """
    from hhemt.exceptions import ConfigurationError

    expected = ConfigurationError if error == "ConfigurationError" else error
    if expected is None:
        _prepare(tmp_path, monkeypatch, coupled=True, written_list=written_list)
        return
    with pytest.raises(expected, match=match):
        _prepare(tmp_path, monkeypatch, coupled=True, written_list=written_list)
