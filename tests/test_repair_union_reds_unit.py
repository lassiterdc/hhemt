"""Laptop-tier checks for the repair of the union-suite reds caused by the 0.3.0 release (change 1117-REPAIR1).

Each test asserts the absence of one defect over the population the item names, and each was red before its
repair. Nothing here compiles or runs a solver.

- Item 2 (``test_synth_04_multisim_with_snakemake``): where ``THREADS`` is written, and the one reader the synthetic
  test shares with this file (``threads_rows``).
- Item 3 (``test_synth_06_sensitivity_analysis_triton_and_swmm_only``): the OpenMP-thread expectation of a SWMM row.
- Item 4 (``report_renderers.system_overview``): the hydraulics panel with the dummy chain of a derived model.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import matplotlib
import pandas as pd
import pytest
from plotly.subplots import make_subplots

matplotlib.use("Agg")

from hhemt import swmm_utils  # noqa: E402
from hhemt.config.report import SystemMapConfig  # noqa: E402
from hhemt.consolidate_workflow import expected_swmm_threads, validate_resource_usage  # noqa: E402
from hhemt.report_renderers._provenance import ProvenanceLog  # noqa: E402
from hhemt.report_renderers.system_overview import (  # noqa: E402
    _draw_hydraulics_panel,
    _draw_hydraulics_panel_plotly,
)
from hhemt.swmm_runnable import (  # noqa: E402
    DUMMY_NODE_PREFIX,
    derive_role_model,
    set_threads,
    without_injected_dummy_chain,
)
from tests.fixtures.swmm_runnable_inputs import runnable_source_text  # noqa: E402
from tests.utils_for_testing import threads_rows  # noqa: E402

# ---------------------------------------------------------------------------
# Item 2: THREADS is written into the hydraulics and full roles, and into no other
# ---------------------------------------------------------------------------


def test_threads_rows_reader_finds_a_row_and_is_empty_without_one():
    # The control that must fire: a reader that returned [] for everything would pass every absence assertion below.
    assert threads_rows("[OPTIONS]\nFLOW_ROUTING DYNWAVE\nTHREADS              3\n") == ["3"]
    assert threads_rows("[OPTIONS]\nFLOW_ROUTING DYNWAVE\n") == []


@pytest.mark.parametrize("swmm_threads", [1, 3])
def test_write_swmm_role_model_writes_threads_only_into_hydraulics_and_full(tmp_path, monkeypatch, swmm_threads):
    """The population is the three role models of one scenario: two carry the ``swmm_threads`` row, hydrology none."""
    monkeypatch.setattr(swmm_utils, "swmm_role_text", lambda *a, **k: "[OPTIONS]\nFLOW_ROUTING DYNWAVE\n")
    monkeypatch.setattr(swmm_utils, "_swmm_fill_values", lambda scenario, destination: {})
    monkeypatch.setattr(swmm_utils, "rewrite_forcing", lambda text, fill: text)
    scenario = SimpleNamespace(
        _analysis=SimpleNamespace(cfg_analysis=SimpleNamespace(swmm_threads=swmm_threads)),
        _system=SimpleNamespace(cfg_system=object()),
        log=MagicMock(),
    )
    written = {}
    for role in ("hydrology", "hydraulics", "full"):
        destination = tmp_path / f"{role}.inp"
        swmm_utils.write_swmm_role_model(scenario, role, destination)
        written[role] = threads_rows(destination.read_text(encoding="utf-8"))
    assert written == {"hydrology": [], "hydraulics": [str(swmm_threads)], "full": [str(swmm_threads)]}


def test_derived_hydrology_role_keeps_the_base_models_threads_row_and_adds_none():
    """The base model's own ``THREADS`` row survives derivation into the hydrology role, unchanged and inert.

    Measured against the synthetic builder's model: it carries ``THREADS`` followed by 15 spaces then ``1``, and every
    derived role keeps that spacing. A test that looks for a literal with another run of spaces finds the row in none
    of the three roles, whatever value it holds.
    """
    source = runnable_source_text()
    assert threads_rows(source) == ["1"]  # the control that must fire: the model does carry a row
    assert threads_rows(derive_role_model(source, "hydrology", [])) == ["1"]
    literal = "THREADS              1"  # 14 spaces, the literal the synthetic test used to look for
    for role in ("hydrology", "hydraulics", "full"):
        text = set_threads(derive_role_model(source, role, []), 1)
        assert literal not in text
        assert threads_rows(text) == ["1"]


# ---------------------------------------------------------------------------
# Item 3: the OpenMP-thread expectation of a SWMM row
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("swmm_threads", "omp_threads", "expected"),
    [(1, 3, 1), (3, 3, 3), (4, 2, 2), (None, 3, 1), (2, None, 1)],
)
def test_expected_swmm_threads_is_swmm_threads_bounded_by_the_openmp_count(swmm_threads, omp_threads, expected):
    assert expected_swmm_threads(swmm_threads, omp_threads) == expected


def _status_row(*, model_type, run_mode, n_omp_threads, actual_omp_threads, **extra):
    return {
        "run_completed": True,
        "scenario_directory": "/sims/member_0/event_index.0",
        "model_type": model_type,
        "actual_nTasks": 1,
        "actual_omp_threads": actual_omp_threads,
        "actual_gpu_backend": "none",
        "run_mode": run_mode,
        "n_mpi_procs": 1,
        "n_omp_threads": n_omp_threads,
        "n_gpus": 0,
        **extra,
    }


def _omp_issues(row, *, swmm_threads):
    analysis = SimpleNamespace(cfg_analysis=SimpleNamespace(swmm_threads=swmm_threads))
    passed, issues = validate_resource_usage(analysis, df_status=pd.DataFrame([row]))
    return passed, [(i["expected"], i["actual"]) for i in issues if i["resource"] == "OMP threads"]


def test_swmm_row_expects_swmm_threads_not_the_openmp_count():
    """A SWMM-only row of ``n_omp_threads=3`` that ran ``swmm_threads=1`` is the 2026-11-17 Rivanna red."""
    row = _status_row(model_type="swmm", run_mode="openmp", n_omp_threads=3, actual_omp_threads=1)
    assert _omp_issues(row, swmm_threads=1) == (True, [])
    # The control that must fire: the same row measuring the OpenMP count while swmm_threads is 1 is a mismatch.
    assert _omp_issues({**row, "actual_omp_threads": 3}, swmm_threads=1) == (False, [(1, 3)])


def test_swmm_row_with_swmm_threads_three_expects_three():
    row = _status_row(model_type="swmm", run_mode="openmp", n_omp_threads=3, actual_omp_threads=3)
    assert _omp_issues(row, swmm_threads=3) == (True, [])
    assert _omp_issues({**row, "actual_omp_threads": 1}, swmm_threads=3) == (False, [(3, 1)])


def test_swmm_row_is_bounded_by_the_run_mode_openmp_count():
    """A serial run mode allows one OpenMP thread, so a SWMM row of that mode expects 1 whatever ``swmm_threads`` is."""
    row = _status_row(model_type="swmm", run_mode="serial", n_omp_threads=3, actual_omp_threads=1)
    assert _omp_issues(row, swmm_threads=3) == (True, [])


def test_swmm_row_reads_swmm_threads_from_the_sensitivity_column_when_the_row_carries_it():
    row = _status_row(
        model_type="swmm",
        run_mode="openmp",
        n_omp_threads=3,
        actual_omp_threads=2,
        **{"analysis.swmm_threads": 2},
    )
    # The analysis config says 1; the row's own column says 2, and the row wins.
    assert _omp_issues(row, swmm_threads=1) == (True, [])
    assert _omp_issues({**row, "actual_omp_threads": 1}, swmm_threads=1) == (False, [(2, 1)])


def test_triton_rows_still_expect_the_openmp_count():
    """The check is not weakened for TRITON rows: ``n_omp_threads=3`` measured as 1 is still a mismatch."""
    for model_type in ("tritonswmm", "triton"):
        row = _status_row(model_type=model_type, run_mode="openmp", n_omp_threads=3, actual_omp_threads=1)
        assert _omp_issues(row, swmm_threads=1) == (False, [(3, 1)])
        assert _omp_issues({**row, "actual_omp_threads": 3}, swmm_threads=1) == (True, [])


# ---------------------------------------------------------------------------
# Item 4: the hydraulics panel with the injected dummy chain
# ---------------------------------------------------------------------------


def _model_with_dummy_chain():
    """A derived model of one real junction, outfall and conduit plus the dummy chain, which has no coordinates."""
    j, c, o = (f"{DUMMY_NODE_PREFIX}{k}" for k in ("j", "c", "of"))
    return SimpleNamespace(
        inp=SimpleNamespace(
            coordinates=pd.DataFrame({"X": [10.0, 20.0], "Y": [5.0, 6.0]}, index=["J1", "O1"]),
            junctions=pd.DataFrame({"InvertElev": [1.0, 2.0], "MaxDepth": [3.0, 3.0]}, index=["J1", j]),
            outfalls=pd.DataFrame({"InvertElev": [0.5, 0.0]}, index=["O1", o]),
            conduits=pd.DataFrame(
                {"InletNode": ["J1", j], "OutletNode": ["O1", o], "Length": [10.0, 1.0]}, index=["C1", c]
            ),
        )
    )


def test_hydraulics_panel_draws_a_model_that_carries_the_dummy_chain():
    """The matplotlib panel raised ``KeyError: 'hhemt_dummy_j'`` on this model."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    try:
        prov = ProvenanceLog()
        _draw_hydraulics_panel(ax, _model_with_dummy_chain(), "h.inp", (0.0, 0.0, 30.0, 10.0), SystemMapConfig(), prov)
        notes = [a["note"] for a in prov.serialize()]
    finally:
        plt.close(fig)
    # The control that must fire: the real nodes are drawn, so the dummy filter is not simply drawing nothing.
    assert "junctions (1)" in notes and "outfalls (1)" in notes


@pytest.fixture(scope="module")
def _plotly_warm():
    """Pay plotly's one-time template initialisation (about 0.2 s, measured) in setup, not in the timed test."""
    make_subplots(rows=1, cols=1)


def test_plotly_hydraulics_panel_draws_a_model_that_carries_the_dummy_chain(_plotly_warm):
    """The Plotly panel (the report's interactive figure) raised the same ``KeyError`` at its junction loop."""
    fig = make_subplots(rows=1, cols=1)
    prov = ProvenanceLog()
    _draw_hydraulics_panel_plotly(fig, _model_with_dummy_chain(), "h.inp", SystemMapConfig(), prov, col=1, row=1)
    drawn = {t.name: list(t.x) for t in fig.data if t.name in ("Junctions", "Outfalls")}
    assert drawn == {"Junctions": [10.0], "Outfalls": [20.0]}


def test_without_injected_dummy_chain_drops_only_the_prefixed_rows():
    frame = pd.DataFrame({"v": [1, 2, 3]}, index=["J1", f"{DUMMY_NODE_PREFIX}j", "dummy_but_not_the_prefix"])
    assert list(without_injected_dummy_chain(frame).index) == ["J1", "dummy_but_not_the_prefix"]


def test_dem_resolution_junction_counts_exclude_the_injected_dummy_junction(tmp_path):
    """``Total junctions`` of the DEM-resolution table counted ``hhemt_dummy_j`` when every outfall is coupled.

    The model is the real derivation of the synthetic source with its only outfall coupled, which leaves no outfall in
    the role and so injects the chain. Control that must fire: the unfiltered junction table holds the dummy row.
    """
    import swmmio

    from hhemt.eda._dem_resolution_plots import _junction_counts_of_model

    path = tmp_path / "hydraulics.inp"
    path.write_text(derive_role_model(runnable_source_text(), "hydraulics", ["dummy_outfall"]), encoding="utf-8")
    model = swmmio.Model(str(path))
    raw = [str(n) for n in model.inp.junctions.index]
    assert f"{DUMMY_NODE_PREFIX}j" in raw
    _coupling, total = _junction_counts_of_model(model)
    assert total == len(raw) - 1
    assert f"{DUMMY_NODE_PREFIX}j" not in {str(n) for n in without_injected_dummy_chain(model.inp.junctions).index}
