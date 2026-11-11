"""Reviewer tests for the readers of a removed scenario: the kept geometry file and the renderers that reach it.

Problem this file solves: once a scenario directory is removed, a page of the report must take an event's conduit
geometry from the copy the removal kept, and no per-event renderer may build the run object of the removed scenario.
The coder's tests sample the helper on one removed and one live event; these cover the content of what the helper
returns and the class of renderer call sites. No solver, no Snakemake; the multi-sim golden is extracted into
`tmp_path`.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import hhemt.report_renderers as _renderers
from tests.fixtures.golden_trees import construct_golden_analysis, dry_run_plan
from tests.test_df_status_removed_scenarios import _slug

_RENDERER_DIR = Path(_renderers.__file__).parent
_HYDRO_MARK = "; HYDROLOGY-ONLY MARKER"
_FORBIDDEN_CALLS = {"_retrieve_sim_run_processing_object", "_retrieve_sim_runs", "TRITONSWMM_scenario"}


def _forbidden_call_sites(source: str, allowed_functions: frozenset[str]) -> list[tuple[str, int]]:
    """`(function, line)` of every call to a run-object builder outside the allowed functions of one module."""
    tree = ast.parse(source)
    found: list[tuple[str, int]] = []
    for func in (n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))):
        if func.name in allowed_functions:
            continue
        for node in ast.walk(func):
            if isinstance(node, ast.Call):
                callee = node.func
                name = callee.attr if isinstance(callee, ast.Attribute) else getattr(callee, "id", "")
                if name in _FORBIDDEN_CALLS:
                    found.append((func.name, node.lineno))
    return found


def _removed_tree_with_marked_hydrology(golden_extractor):
    """The multi-sim golden with scenario 0 removed after its hydrology `.inp` was given a marker line, so the kept
    copy of each kind can be told from the other (the committed golden's two files are byte-equal)."""
    from hhemt.scenario_records import capture_system_overview_inps, remove_scenario_directory

    system_dir = golden_extractor("multisim")
    analysis = construct_golden_analysis(system_dir)
    dry_run_plan(analysis)
    models = list(analysis._get_enabled_model_types())
    evt = _slug(analysis, 0)
    sim = Path(analysis.analysis_paths.simulation_directory) / evt
    hydro = sim / "swmm" / "hydro.inp"
    hydro.write_text(hydro.read_text() + _HYDRO_MARK + "\n")
    capture_system_overview_inps(sim, analysis.analysis_paths.analysis_dir)
    remove_scenario_directory(
        analysis.analysis_paths.analysis_dir, evt, sim, run_completed={m: True for m in models}, captured={}
    )
    return analysis


def test_the_kept_geometry_of_a_removed_event_is_the_conduit_bearing_file_the_live_scenario_holds(golden_extractor):
    """Property: for a removed event, `event_hydraulics_inp` returns a file that holds a `[CONDUITS]` section and whose
    bytes equal the hydraulics (else combined) `.inp` a live scenario of the same analysis holds; for a live event it
    returns that scenario's own file, exactly the expression the renderers used before.

    Class: both arms of the helper over one analysis with a removed and a live scenario. Plausible wrong
    implementation: returning the hydrology `.inp` (no conduits, a blank figure) or a copy of another network.
    Must-fire control: scenario 0's hydrology `.inp` carries a marker line before the removal, so the kept hydrology
    copy and the kept hydraulics copy differ and the kept file must not carry the marker.
    Why required: the page draws the conduit network from this file; the hydrology model's network is not it.
    Killed by: `_resolve_inp_sources(analysis)[0]` in place of `[1]`. Passes under a second correct implementation
    that copies the geometry into the record.
    """
    from hhemt.report_renderers.per_sim_conduit_flow import event_hydraulics_inp

    a = _removed_tree_with_marked_hydrology(golden_extractor)
    live_proc = a._retrieve_sim_run_processing_object(1)
    live_paths = live_proc.scen_paths
    expected = Path(getattr(live_paths, "swmm_hydraulics_inp", None) or live_paths.swmm_full_inp)
    assert "[CONDUITS]" in expected.read_text(), "control: the live hydraulics file carries conduits"
    assert _HYDRO_MARK not in expected.read_text(), "control: the live hydraulics file has no marker"
    assert event_hydraulics_inp(a, 1) == expected, "live arm unchanged"
    kept = event_hydraulics_inp(a, 0)
    assert "scenario_records" in kept.as_posix() and kept.name == "hydraulics.inp"
    assert "[CONDUITS]" in kept.read_text()
    assert kept.read_bytes() == expected.read_bytes()
    assert _HYDRO_MARK not in kept.read_text(), "the kept file is the hydraulics copy, not the marked hydrology one"


@pytest.mark.parametrize("module", ["per_sim_conduit_flow", "per_sim_event_page", "per_sim_peak_flood_depth"])
def test_no_per_sim_renderer_builds_a_run_object_outside_the_geometry_helper(module):
    """Property: in each per-event renderer module, a call to `_retrieve_sim_run_processing_object`,
    `_retrieve_sim_runs` or `TRITONSWMM_scenario` occurs only inside `event_hydraulics_inp`, the one function that
    guards it with the removed-event test.

    Class: every function of the three modules the per-event page reaches (the page, its conduit panel, its flood
    panel). Plausible wrong implementation: a panel builder that reads its own scenario paths again, which raises for
    a removed scenario. Must-fire control: the same walker over a planted function that calls the builder reports it.
    Not established: other renderer modules are outside this class (see the residual on `sensitivity_benchmarking`).
    """
    planted = "def panel(analysis, i):\n    return analysis._retrieve_sim_run_processing_object(i)\n"
    assert _forbidden_call_sites(planted, frozenset()) == [("panel", 2)], "control: the walker sees a planted call"
    source = (_RENDERER_DIR / f"{module}.py").read_text()
    assert _forbidden_call_sites(source, frozenset({"event_hydraulics_inp"})) == []
