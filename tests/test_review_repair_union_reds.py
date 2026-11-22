"""Reviewer tests for change 1117-REPAIR1 (the repair of the union-suite reds of the 0.3.0 release).

Nothing here compiles or runs a solver. Every test is a laptop-tier check over a population the coder's own file
(``tests/test_repair_union_reds_unit.py``) samples by hand-built stand-ins; these draw the population from the
toolkit's own derivation and from the two instruments that compute the same thread expectation, so a stand-in that
drifts from the real object is visible here.

Held tests (green before this turn and required to stay green): all four. Each is a characterization of a property
the repair relies on, with the named change that reds it in its docstring.
"""

from __future__ import annotations

import itertools
import re
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import pandas as pd
import pytest
from plotly.subplots import make_subplots

matplotlib.use("Agg")

import swmmio  # noqa: E402

from hhemt.config.report import SystemMapConfig  # noqa: E402
from hhemt.consolidate_workflow import validate_resource_usage  # noqa: E402
from hhemt.report_renderers._provenance import ProvenanceLog  # noqa: E402
from hhemt.report_renderers._swmm_gis_layers import export_swmm_gis_layers  # noqa: E402
from hhemt.report_renderers.system_overview import (  # noqa: E402
    _build_manifest_data,
    _draw_hydraulics_panel,
    _draw_hydraulics_panel_plotly,
    _draw_hydrology_panel,
    _draw_hydrology_panel_plotly,
)
from hhemt.run_simulation import omp_thread_count  # noqa: E402
from hhemt.swmm_runnable import DUMMY_NODE_PREFIX, derive_role_model, set_threads  # noqa: E402
from tests.fixtures.swmm_runnable_inputs import runnable_source_text  # noqa: E402

# The one outfall of the runnable source model. Coupling it converts it to a junction, leaves no outfall in the
# role, and so makes the derivation add the dummy chain.
_ONLY_OUTFALL = "dummy_outfall"
_THREADS_ROW = re.compile(r"^[ \t]*THREADS[ \t]+(\S+)[ \t]*$", re.IGNORECASE | re.MULTILINE)


@pytest.fixture(scope="module")
def _derived_dummy_models(tmp_path_factory):
    """The hydrology and hydraulics role models the toolkit derives when every outfall is coupled, read by swmmio."""
    root = tmp_path_factory.mktemp("derived")
    source = runnable_source_text()
    models = {}
    for role in ("hydrology", "hydraulics"):
        path = root / f"{role}.inp"
        path.write_text(derive_role_model(source, role, [_ONLY_OUTFALL]), encoding="utf-8")
        models[role] = swmmio.Model(str(path))
    return models


@pytest.fixture(scope="module")
def _plotly_warm():
    """Pay plotly's one-time template initialisation in setup, outside the timed tests."""
    make_subplots(rows=1, cols=1)


def _dummy_names(frame) -> list[str]:
    return [str(n) for n in frame.index if str(n).startswith(DUMMY_NODE_PREFIX)]


def test_the_derivation_this_file_draws_from_does_carry_the_dummy_chain(_derived_dummy_models):
    """Control that must fire: both real derived roles hold dummy rows, so the drawing tests below are not vacuous.

    Property: coupling the model's only outfall yields a dummy junction, outfall and conduit in the hydraulics
    role. Class: the two derived roles. Why required: the other tests here pass trivially on a model with no dummy
    rows. Killed by: a derivation that stopped injecting the chain (then the repair's premise is gone, which is the
    finding). A second correct implementation: a derivation that names the chain differently but keeps the prefix.
    """
    hydraulics = _derived_dummy_models["hydraulics"].inp
    assert (
        _dummy_names(hydraulics.junctions) and _dummy_names(hydraulics.outfalls) and _dummy_names(hydraulics.conduits)
    )
    assert not _dummy_names(hydraulics.coordinates)  # the reason a per-node coordinate read raises KeyError


def test_the_matplotlib_panels_draw_a_real_derived_dummy_model(_derived_dummy_models):
    """Property: the matplotlib hydrology and hydraulics panels neither raise on nor draw the dummy chain.

    Class: the two panel drawers of ``system_overview`` over the REAL derived roles (the coder's own test uses a
    hand-built look-alike). Why required: the figure is built from exactly these frames and one unfiltered loop raises
    ``KeyError``. Killed by: removing the dummy filter at the matplotlib panel's three reads. A second correct
    implementation: filtering the model once where it is loaded instead of at each read.
    """
    import matplotlib.pyplot as plt

    bounds = (-10_000.0, -10_000.0, 10_000.0, 10_000.0)
    fig, axes = plt.subplots(1, 2)
    try:
        prov = ProvenanceLog()
        _draw_hydrology_panel(axes[0], _derived_dummy_models["hydrology"], "hydro.inp", bounds, SystemMapConfig(), prov)
        _draw_hydraulics_panel(
            axes[1], _derived_dummy_models["hydraulics"], "hydraulics.inp", bounds, SystemMapConfig(), prov
        )
        notes = [a["note"] for a in prov.serialize()]
    finally:
        plt.close(fig)
    hydraulics = _derived_dummy_models["hydraulics"].inp
    real_junctions = len(hydraulics.junctions) - len(_dummy_names(hydraulics.junctions))
    assert f"junctions ({real_junctions})" in notes


def test_the_plotly_panels_and_manifest_counts_use_a_real_derived_dummy_model(_derived_dummy_models, _plotly_warm):
    """Property: the Plotly panels draw no dummy-chain trace and the manifest counts exclude the chain.

    Class: the two Plotly drawers and ``_build_manifest_data`` over the real derived roles. Why required: the Plotly
    figure is the report's interactive map and its junction loop raised the same ``KeyError``. Killed by: removing
    the filter at the Plotly panel's reads or at the three manifest counts. A second correct implementation: counting
    from the real nodes' own table.
    """
    hydro, hydraulics = _derived_dummy_models["hydrology"], _derived_dummy_models["hydraulics"]
    cfg = SystemMapConfig()
    fig = make_subplots(rows=1, cols=2)
    prov = ProvenanceLog()
    _draw_hydrology_panel_plotly(fig, hydro, "hydro.inp", cfg, prov, col=1)
    _draw_hydraulics_panel_plotly(fig, hydraulics, "hydraulics.inp", cfg, prov, col=2)
    drawn = [str(v) for t in fig.data for v in (t.text if t.text is not None else []) if v is not None]
    assert not [v for v in drawn if DUMMY_NODE_PREFIX in v]
    manifest = _build_manifest_data("a", (-1.0, -1.0, 1.0, 1.0), hydro, hydraulics, False)
    counts = {p["name"]: p["element_counts"] for p in manifest["panels"] if "element_counts" in p}["hydraulics"]
    inp = hydraulics.inp
    assert counts["junctions"] == len(inp.junctions) - len(_dummy_names(inp.junctions))
    assert counts["outfalls"] == len(inp.outfalls) - len(_dummy_names(inp.outfalls))
    assert counts["conduits"] == len(inp.conduits) - len(_dummy_names(inp.conduits))


def test_the_geojson_layers_carry_no_dummy_chain_feature_from_a_real_derived_model(_derived_dummy_models, tmp_path):
    """Property: no exported GeoJSON layer holds a feature named for the dummy chain.

    Class: the five layers ``export_swmm_gis_layers`` writes over the real derived roles. Why required: the layers
    are durable outputs a GIS user opens, and the chain is a solver scaffold with no coordinates. Killed by: a
    writer that began emitting uncoordinated nodes. A second correct implementation: filtering by name rather than
    by the absence of a coordinate row.
    """
    layers = export_swmm_gis_layers(_derived_dummy_models["hydrology"], _derived_dummy_models["hydraulics"], tmp_path)
    assert set(layers) >= {"junctions", "outfalls", "conduits"}
    for layer, path in layers.items():
        assert DUMMY_NODE_PREFIX not in Path(path).read_text(), f"{layer} layer carries a dummy-chain feature"


@pytest.mark.parametrize("swmm_threads", [1, 2, 3, 4])
def test_the_hydrology_role_keeps_the_base_threads_row_whatever_swmm_threads_is(swmm_threads):
    """Property: the hydrology role's ``THREADS`` rows are the base model's, independent of ``swmm_threads``.

    Class: ``swmm_threads`` in 1..4 against the synthetic base model (which carries ``THREADS`` 1). Why required:
    an assertion that the hydrology model's rows equal ``min(swmm_threads, omp)`` holds only while that happens to
    equal the base's value; this pins what the role actually does, so a test of it can be written without that
    coupling. Killed by: a derivation that wrote ``swmm_threads`` into the hydrology role (the toolkit documents it
    does not). A second correct implementation: dropping the base row from the hydrology role (then the rows are
    empty and the set-inclusion holds).
    """
    source = runnable_source_text()
    base_rows = _THREADS_ROW.findall(source)
    assert base_rows == ["1"]  # the control that must fire
    hydrology = derive_role_model(source, "hydrology", [])
    assert _THREADS_ROW.findall(hydrology) == base_rows
    # the roles that DO take swmm_threads rewrite the row in place to it
    for role in ("hydraulics", "full"):
        assert _THREADS_ROW.findall(set_threads(derive_role_model(source, role, []), swmm_threads)) == [
            str(swmm_threads)
        ]


def _swmm_status_row(*, run_mode, n_omp_threads, actual):
    return {
        "run_completed": True,
        "scenario_directory": "/s/member_0/event_index.0",
        "model_type": "swmm",
        "actual_nTasks": 1,
        "actual_omp_threads": actual,
        "actual_gpu_backend": "none",
        "run_mode": run_mode,
        "n_mpi_procs": 1,
        "n_omp_threads": n_omp_threads,
        "n_gpus": 0,
    }


@pytest.mark.parametrize("run_mode", ["serial", "mpi", "openmp", "hybrid", "gpu"])
def test_a_swmm_rows_expected_threads_equals_the_one_the_coupled_check_computes_for_every_valid_config(run_mode):
    """Property: for every V24-valid config, the SWMM-row expectation equals ``min(swmm_threads, omp_thread_count)``.

    Class: ``(run_mode, n_omp_threads, swmm_threads)`` with ``swmm_threads <= omp_thread_count(cfg)`` (what
    ``validation._validate_swmm_threads`` admits), the row built as ``analysis.df_status`` builds a SWMM row (run
    mode ``serial`` at one OpenMP thread and ``openmp`` above it, never the table's own). Why required: the two
    instruments (``consolidate_workflow.validate_resource_usage`` and ``analysis_validation.swmm_thread_issues``)
    must not disagree about one run. Killed by: reverting the SWMM-row expectation to the OpenMP count (the original
    red: ``n_omp_threads`` 3 with ``swmm_threads`` 1 expects 3). A second correct implementation, and a measured
    fact about the repair: ``expected_swmm_threads`` returning ``swmm_threads`` alone with no ``min`` bound passes
    this grid, because V24 already refuses ``swmm_threads`` above the bound, so the bound is redundant on every valid
    config and matters only on one V24 refuses.
    """
    checked = 0
    for n_omp, swmm_threads in itertools.product([1, 2, 3], [1, 2, 3]):
        cfg = SimpleNamespace(run_mode=run_mode, n_omp_threads=n_omp, swmm_threads=swmm_threads)
        if swmm_threads > omp_thread_count(cfg):
            continue  # refused at configuration time (V24), so no run of it exists
        row = _swmm_status_row(run_mode="serial" if n_omp == 1 else "openmp", n_omp_threads=n_omp, actual=99)
        analysis = SimpleNamespace(cfg_analysis=cfg)
        _, issues = validate_resource_usage(analysis, df_status=pd.DataFrame([row]))
        expected = [i["expected"] for i in issues if i["resource"] == "OMP threads"]
        assert expected == [min(swmm_threads, omp_thread_count(cfg))], (run_mode, n_omp, swmm_threads, expected)
        checked += 1
    assert checked >= 1  # the grid is not empty for any mode
