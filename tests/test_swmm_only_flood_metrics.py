"""The standalone SWMM model's flood and water-balance additions.

No solver is compiled or run. The system series are read from the committed reference output
``test_data/swmm_binary_reference/hydraulics.out`` (one stock SWMM run, 12 report periods) and the Runoff Quantity
Continuity rows from the committed stock reports ``test_data/swmm_refactoring_reference/hydro_runoff_summary.rpt``
(a run with runoff) and ``test_data/swmm_binary_reference/hydraulics.rpt`` (a run that routes no runoff). The
consolidation rows extract the committed multi-sim golden per test, whose SWMM-only scenarios the golden fixture
completes with a system series store and the twelve system reductions (``add_swmm_only_system_stores``). Each row names
its must-fire control: the input class that makes a broken implementation read as correct.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

REFERENCE = Path(__file__).resolve().parent.parent / "test_data" / "swmm_binary_reference"
OUT = REFERENCE / "hydraulics.out"
HYDRAULICS_RPT = REFERENCE / "hydraulics.rpt"
RUNOFF_RPT = (
    Path(__file__).resolve().parent.parent / "test_data" / "swmm_refactoring_reference" / "hydro_runoff_summary.rpt"
)

#: stored name -> member of ``swmm.toolkit.shared_enum.SystemAttribute``.
SYSTEM_ATTRIBUTES = {
    "system_flooding_rate": "FLOOD_LOSSES",
    "system_outfall_flow": "OUTFALL_FLOWS",
    "system_stored_volume": "VOLUME_STORED",
    "system_lateral_inflow": "TOTAL_LATERAL_INFLOW",
    "system_runoff": "RUNOFF_FLOW",
    "system_rainfall": "RAINFALL",
}
SYSTEM_SUMMARY = [f"{name}_{reduction}" for name in SYSTEM_ATTRIBUTES for reduction in ("max", "last")]
RUNOFF_ROWS = {
    "runoff_continuity_total_precipitation": "Total Precipitation",
    "runoff_continuity_evaporation_loss": "Evaporation Loss",
    "runoff_continuity_infiltration_loss": "Infiltration Loss",
    "runoff_continuity_surface_runoff": "Surface Runoff",
    "runoff_continuity_final_storage": "Final Storage",
}
_SYSTEM_GROUP = "swmm_only/system/timeseries"
_SYSTEM_SUMMARY_GROUP = "swmm_only/system/summary"
_NODE_SUMMARY_GROUP = "swmm_only/swmm_node/summary"


# ---------------------------------------------------------------------------
# The system series, read from the binary output
# ---------------------------------------------------------------------------
def _reader():
    from hhemt.swmm_output_parser import return_system_time_series_results_from_outfile

    return return_system_time_series_results_from_outfile


def _toolkit_system_values(attribute_name: str) -> tuple[np.ndarray, int]:
    """The series for one system attribute, read period by period with ``get_system_result``.

    A different toolkit call from the windowed ``get_system_series`` the reader uses, so the comparison does not
    depend on the reader's own accessor."""
    from pyswmm import Output
    from swmm.toolkit import output, shared_enum

    attribute = getattr(shared_enum.SystemAttribute, attribute_name)
    with Output(str(OUT)) as out:
        n = int(output.get_times(out.handle, shared_enum.Time.NUM_PERIODS))
        # ``get_system_result`` returns every system attribute of one period, indexed by the attribute's value.
        values = [output.get_system_result(out.handle, i, 0)[attribute.value] for i in range(n)]
    return np.array(values, dtype=float), n


def test_the_six_system_series_have_the_files_full_period_count_and_equal_the_toolkits_for_each_attribute():
    """Property: on the committed reference output, the reader returns the six system series on the file's full
    period count, ending at the last report time, and each equals the toolkit's own values attribute by attribute,
    the final period included.

    Class: REGRESSION of E43 for a new accessor. Plausible wrong implementations: pyswmm's ``system_series`` (its
    default end drops the final period under pyswmm 2.1.0), a window ``(0, num_periods - 1)``, or a series stored
    under another attribute's name. Reachability: the real reader over the real file. Both-states anchor: the node
    and link reader of ``tests/test_swmm_binary_period_count.py`` is held to the same count. Must-fire control:
    ``system_stored_volume`` is not constant zero in this file, so a reader that returns zeros for every series
    fails the equality for it, and the attribute map is exact (six names, six distinct toolkit members). Killed by:
    those. Passes under a second correct implementation that reads each period with ``get_system_result``.
    """
    ds = _reader()(OUT)
    assert sorted(ds.data_vars) == sorted(SYSTEM_ATTRIBUTES)
    assert len(set(SYSTEM_ATTRIBUTES.values())) == 6
    for name, attribute in SYSTEM_ATTRIBUTES.items():
        want, n = _toolkit_system_values(attribute)
        assert ds.sizes["date_time"] == n == 12, (name, ds.sizes["date_time"], n)
        got = ds[name].values
        assert np.allclose(got, want, rtol=1e-9, atol=0.0, equal_nan=True), name
        assert np.isclose(got[-1], want[-1]), f"{name}: the final report period differs"
    assert float(np.abs(ds["system_stored_volume"].values).max()) > 0.0, "control: the reference holds stored volume"
    last_in_report = max(
        np.datetime64(f"{m.group(3)}-{m.group(1)}-{m.group(2)}T{m.group(4)}")
        for m in re.finditer(
            r"^\s*(\d\d)/(\d\d)/(\d{4}) (\d\d:\d\d:\d\d)\s", HYDRAULICS_RPT.read_text(encoding="latin-1"), re.M
        )
    )
    assert ds["date_time"].values[-1] == last_in_report


def test_a_system_series_one_period_short_is_refused(monkeypatch):
    """Property (must-fire control of the period rule): a toolkit that returns one period fewer than ``NUM_PERIODS``
    for a system series makes the reader raise ``ProcessingError`` naming the series, and the same reader over the
    unpatched toolkit returns the full count.

    Class: REGRESSION of E43. Plausible wrong implementation: no length check, which stores a truncated series
    quietly. Reachability: ``get_system_series`` is wrapped. Both-states anchor: the unpatched call below. Killed by:
    removing the length check. Passes under a second correct implementation that raises ``ValueError`` if the
    assertion is widened to the exception pair.
    """
    from swmm.toolkit import output

    from hhemt.exceptions import ProcessingError

    assert _reader()(OUT).sizes["date_time"] == 12
    real = output.get_system_series
    monkeypatch.setattr(
        output, "get_system_series", lambda handle, attr, start, end: real(handle, attr, start, end - 1)
    )
    with pytest.raises(ProcessingError, match="FLOOD_LOSSES"):
        _reader()(OUT)


def test_the_coupled_family_writes_no_system_series_store(golden_extractor):
    """Property (control of the standalone-only rule): in a coupled-only analysis the scenario defines no system
    series path, the series consolidation writes no ``swmm_only/system`` group, and the coupled reader returns no
    ``system_*`` variable for the same output file.

    Class: NEW CAPABILITY. Plausible wrong implementation: the system store written for every model that writes SWMM
    series (the coupled build holds flooded water as ponded volume, so its system flooding is about zero). Reachability:
    the real scenario paths and the real ``consolidate_timeseries`` over the golden with the coupled arm alone.
    Both-states anchor: ``test_with_the_switch_on_the_system_store_is_cleared_after_its_slice_passes`` below runs the
    standalone arm of the same golden and writes the group. Killed by: an unconditional path or mode.
    Passes under a second correct implementation that defines the path and leaves it unwritten.
    """
    from hhemt.scenario import TRITONSWMM_scenario
    from hhemt.swmm_output_parser import retrieve_SWMM_outputs_as_datasets

    a = _ready(golden_extractor, families=("tritonswmm",), clear=False)
    assert TRITONSWMM_scenario(0, a).scen_paths.output_swmm_only_system_time_series is None
    a.process.consolidate_timeseries()
    tree = xr.open_datatree(Path(a.analysis_paths.analysis_datatree_zarr), engine="zarr", consolidated=False)
    assert "system" not in tree["swmm_only"].children, "the golden holds the standalone groups; no system group"
    ds_nodes, _ = retrieve_SWMM_outputs_as_datasets(
        REFERENCE.parent / "swmm_refactoring_reference" / "hydraulics.inp", OUT, family="tritonswmm"
    )
    assert not [v for v in ds_nodes.data_vars if v.startswith("system_") and v != "system_flooding"]
    assert not [v for v in ds_nodes.data_vars if v.startswith("runoff_continuity_") and v != "runoff_continuity_error"]


# ---------------------------------------------------------------------------
# The Runoff Quantity Continuity rows, read from the report
# ---------------------------------------------------------------------------
def _printed_rows(rpt: Path) -> dict[str, float]:
    """The depth column of each Runoff Quantity Continuity row of ``rpt``, by an independent line regex."""
    text = rpt.read_text(encoding="latin-1")
    start = text.index("Runoff Quantity Continuity")
    block = text[start : text.index("Flow Routing Continuity", start)]
    rows = {}
    for label in RUNOFF_ROWS.values():
        match = re.search(rf"^\s*{label} \.+\s+(\S+)\s+(\S+)\s*$", block, re.M)
        assert match is not None, label
        rows[label] = float(match.group(2))
    return rows


def _scan(rpt: Path):
    from hhemt.swmm_output_parser import _REPORT_SECTION_HEADERS, _scan_metadata_and_summaries

    lines = rpt.read_text(encoding="latin-1").splitlines(True)
    return _scan_metadata_and_summaries(lines, _REPORT_SECTION_HEADERS, tail_lines=lines[-5:])[0]


def test_each_runoff_continuity_row_equals_its_printed_row_on_a_stock_report_with_runoff():
    """Property: on a stock SWMM report that routes runoff, each ``runoff_continuity_{term}`` equals the row's depth
    column (the second figure, ``mm``), not its volume column, and the five values are distinct numbers.

    Class: NEW CAPABILITY. Plausible wrong implementations: the first figure (hectare-m) read in place of the
    second, a row matched by a prefix of another row's label, or the Continuity Error percentage read as a row.
    Reachability: the real scan of the real report, then the real reader. Both-states anchor: the report prints
    ``Total Precipitation 3.757 20.000``, so the volume and the depth differ by a factor the equality catches.
    Must-fire control: ``test_a_report_that_routes_no_runoff_yields_nan_without_error`` below. Killed by: those.
    Passes under a second correct implementation that reads the report with its own table parser.
    """
    from hhemt.swmm_output_parser import read_runoff_quantity_continuity

    section = _scan(RUNOFF_RPT)["Runoff Quantity Continuity"]
    got = read_runoff_quantity_continuity(section)
    printed = _printed_rows(RUNOFF_RPT)
    assert sorted(got) == sorted(RUNOFF_ROWS)
    for name, label in RUNOFF_ROWS.items():
        assert got[name] == printed[label], (name, got[name], printed[label])
    assert got["runoff_continuity_total_precipitation"] == 20.0, "the depth, not the 3.757 hectare-m volume"
    assert len({v for v in got.values() if v != 0.0}) == 4


def test_a_report_that_routes_no_runoff_yields_nan_without_error():
    """Property (must-fire control of the row reader): the reference hydraulics report prints no Runoff Quantity
    Continuity table, and the reader returns NaN for all five rows without raising.

    Class: NEW CAPABILITY. Plausible wrong implementation: a missing table raising ``KeyError`` or yielding zero (a
    zero would read as a closed water balance). Reachability: the real scan of the real report. Both-states anchor: the
    test above reads the same five names from a report that prints them. Killed by: those.
    Passes under a second correct implementation that returns an empty dataset for such a report.
    """
    from hhemt.swmm_output_parser import read_runoff_quantity_continuity

    assert "Runoff Quantity Continuity" not in HYDRAULICS_RPT.read_text(encoding="latin-1")
    got = read_runoff_quantity_continuity(_scan(HYDRAULICS_RPT).get("Runoff Quantity Continuity", []))
    assert sorted(got) == sorted(RUNOFF_ROWS)
    assert all(np.isnan(v) for v in got.values())


def test_a_runoff_row_with_a_third_figure_is_refused():
    """Property: a Runoff Quantity Continuity row with a number of figures other than two raises
    ``ProcessingError`` naming the row, and the well-formed row beside it is read.

    Class: NEW CAPABILITY. Plausible wrong implementation: ``split()[-1]``, which reads a malformed row as a value.
    Reachability: the real reader over two constructed lines. Both-states anchor: the second line parses.
    Killed by: taking the last figure. Passes under a second correct implementation that refuses on any non-numeric
    token as well.
    """
    from hhemt.exceptions import ProcessingError
    from hhemt.swmm_output_parser import read_runoff_quantity_continuity

    good = "  Surface Runoff ...........         1.838         9.783\n"
    bad = "  Total Precipitation ......         3.757        20.000         1.000\n"
    assert read_runoff_quantity_continuity([good])["runoff_continuity_surface_runoff"] == 9.783
    with pytest.raises(ProcessingError, match="Total Precipitation"):
        read_runoff_quantity_continuity([bad, good])


def test_the_standalone_dataset_carries_the_runoff_rows_and_the_coupled_one_does_not(tmp_path):
    """Property: with the stock runoff table spliced into the reference report, the standalone reader returns the
    five ``runoff_continuity_`` variables equal to the printed depths, and the coupled reader returns none of them
    for the same pair of files.

    Class: NEW CAPABILITY. Plausible wrong implementation: the rows read for both families (the coupled hydraulics
    model routes no runoff, so its table is absent and every value would be NaN noise in a coupled store), or read
    for neither because the section list lacks the table's title. Reachability: ``retrieve_SWMM_outputs_as_datasets``
    over a copy of the reference output and a report with the table inserted before the Flow Routing Continuity
    table. Both-states anchor: the unspliced report gives NaN for the standalone family. Killed by: those.
    Passes under a second correct implementation that stores the rows in the link dataset.
    """
    from hhemt.swmm_output_parser import retrieve_SWMM_outputs_as_datasets

    inp = REFERENCE.parent / "swmm_refactoring_reference" / "hydraulics.inp"
    plain = tmp_path / "plain"
    spliced = tmp_path / "spliced"
    for folder in (plain, spliced):
        folder.mkdir()
        shutil.copy(OUT, folder / "hydraulics.out")
    shutil.copy(HYDRAULICS_RPT, plain / "hydraulics.rpt")
    lines = HYDRAULICS_RPT.read_text(encoding="latin-1").splitlines(True)
    flow_title = next(i for i, ln in enumerate(lines) if "Flow Routing Continuity" in ln)
    runoff_lines = RUNOFF_RPT.read_text(encoding="latin-1").splitlines(True)
    first = next(i for i, ln in enumerate(runoff_lines) if "Runoff Quantity Continuity" in ln) - 1
    last = next(i for i, ln in enumerate(runoff_lines) if "Flow Routing Continuity" in ln) - 1
    block = runoff_lines[first:last]
    (spliced / "hydraulics.rpt").write_text("".join(lines[: flow_title - 1] + block + lines[flow_title - 1 :]))
    printed = _printed_rows(spliced / "hydraulics.rpt")

    alone, _ = retrieve_SWMM_outputs_as_datasets(inp, spliced / "hydraulics.out", family="swmm")
    for name, label in RUNOFF_ROWS.items():
        assert float(alone[name]) == printed[label], name
    coupled, _ = retrieve_SWMM_outputs_as_datasets(inp, spliced / "hydraulics.out", family="tritonswmm")
    assert not [v for v in coupled.data_vars if v in RUNOFF_ROWS], "the coupled family must not carry the rows"
    unspliced, _ = retrieve_SWMM_outputs_as_datasets(inp, plain / "hydraulics.out", family="swmm")
    assert all(np.isnan(float(unspliced[name])) for name in RUNOFF_ROWS), "control: no table, NaN"


# ---------------------------------------------------------------------------
# The processing export
# ---------------------------------------------------------------------------
def _ready(golden_extractor, *, families, clear: bool, consolidate: bool = True):
    """Golden analysis enabling exactly ``families``, with the series consolidation and the clearing switch as given."""
    from tests.test_family_parity_consolidation import _ready as parity_ready

    return parity_ready(golden_extractor, families=families, clear=clear, consolidate=consolidate)


def _scen(a, event_iloc: int):
    from hhemt.scenario import TRITONSWMM_scenario

    return TRITONSWMM_scenario(event_iloc, a)


def _store(a) -> Path:
    return Path(a.analysis_paths.analysis_datatree_zarr)


def _group(a, name: str) -> xr.Dataset:
    return xr.open_zarr(_store(a), group=name, consolidated=False)


def test_the_standalone_export_writes_the_system_store_and_the_twelve_reductions(golden_extractor, tmp_path):
    """Property: processing a standalone SWMM scenario whose node and link stores are already written writes the
    system store from the binary output, equal to the toolkit's series, records it in the processing log, and the
    summary export then holds the twelve ``_max`` and ``_last`` reductions in the node summary store, each equal to
    the maximum and the last value of its series.

    Class: NEW CAPABILITY. Plausible wrong implementations: the early return that ignores the system arm (a
    scenario processed before the change never gets its store), the system store written for the coupled model, or
    reductions computed from the node store. Reachability: ``_export_SWMM_outputs`` and ``_export_SWMM_summary``
    over a golden scenario whose system store is removed and whose raw output is a COPY of the reference (a copy, so
    the raw-output clearing cannot touch the committed file). Both-states anchor: the arm above that finds the
    store already written does nothing. Must-fire control: with the store already recorded the export does not call
    the reader. Killed by: those. Passes under a second correct implementation that writes the store in its own
    export method.
    """
    from hhemt.process_simulation import TRITONSWMM_sim_post_processing

    a = _ready(golden_extractor, families=("swmm",), clear=False)
    scen = _scen(a, 0)
    raw = tmp_path / "raw"
    raw.mkdir()
    shutil.copy(OUT, raw / "hydraulics.out")
    shutil.copy(HYDRAULICS_RPT, raw / "hydraulics.rpt")
    paths = scen.scen_paths
    paths.swmm_full_out_file = raw / "hydraulics.out"
    system_store = Path(paths.output_swmm_only_system_time_series)
    shutil.rmtree(system_store)
    log = scen.get_log("swmm")
    proc = TRITONSWMM_sim_post_processing(scen.run, model_log=log)
    assert proc._swmm_system_outputs_processed() is False
    proc._export_SWMM_outputs("swmm", override_clear_raw="none")
    assert system_store.exists() and proc._swmm_system_outputs_processed() is True
    stored = xr.open_zarr(system_store, consolidated=False).load()
    for name, attribute in SYSTEM_ATTRIBUTES.items():
        want, _ = _toolkit_system_values(attribute)
        assert np.allclose(stored[name].values, want, equal_nan=True), name
    # control: with the store recorded as written, a second export does not read the binary output again
    from hhemt import process_simulation as ps

    def refuse(*args, **kwargs):
        raise AssertionError("the system series was read again")

    real = ps.return_system_time_series_results_from_outfile
    ps.return_system_time_series_results_from_outfile = refuse
    try:
        proc._export_SWMM_outputs("swmm", override_clear_raw="none")
    finally:
        ps.return_system_time_series_results_from_outfile = real

    summary_store = Path(paths.output_swmm_only_node_summary)
    shutil.rmtree(summary_store)
    log.processing_log.outputs.pop(summary_store.name, None)
    proc._export_SWMM_summaries("swmm")
    summary = xr.open_zarr(summary_store, consolidated=False).load()
    for name in SYSTEM_ATTRIBUTES:
        assert float(summary[f"{name}_max"].isel(event_iloc=0)) == float(stored[name].max()), name
        assert float(summary[f"{name}_last"].isel(event_iloc=0)) == float(stored[name].isel(date_time=-1)), name
    assert not [v for v in summary.data_vars if v in SYSTEM_ATTRIBUTES], "a series is not a summary variable"


# ---------------------------------------------------------------------------
# Consolidation: one group per value, and the clearing of the system store
# ---------------------------------------------------------------------------
def _one_group_violations(system: xr.Dataset, node: xr.Dataset, run_level: set[str]) -> list[str]:
    """What breaks the rule that each value has one group: the system group holds exactly the twelve reductions,
    the node group holds none of them, and no run-level variable is in the system group."""
    problems = []
    if sorted(system.data_vars) != sorted(SYSTEM_SUMMARY):
        problems.append(f"the system group holds {sorted(system.data_vars)}, not the twelve reductions")
    leaked = sorted(set(node.data_vars) & set(SYSTEM_SUMMARY))
    if leaked:
        problems.append(f"the node group also holds {leaked}")
    stray = sorted(set(system.data_vars) & run_level)
    if stray:
        problems.append(f"the system group holds the run-level variables {stray}")
    return problems


def _add_run_level_variable(a, name: str, value: float) -> None:
    """Write one run-level scalar (no node or link dimension) into every scenario's node summary store."""
    for event_iloc in a.df_sims.index:
        store = Path(_scen(a, int(event_iloc)).scen_paths.output_swmm_only_node_summary)
        existing = xr.open_zarr(store, consolidated=False)["event_iloc"].values
        xr.Dataset(
            {name: (("event_iloc",), np.full(len(existing), value), {"units": "percent"})},
            coords={"event_iloc": existing},
        ).to_zarr(store, mode="a")


def test_each_system_reduction_is_in_the_system_summary_group_and_in_no_other(golden_extractor):
    """Property: a standalone analysis whose node summary stores hold the six system series' reductions consolidates
    each of the twelve ``system_*_max`` and ``system_*_last`` variables into ``swmm_only/system/summary`` and none
    into ``swmm_only/swmm_node/summary``, a D11 run-level variable written beside them stays in the node group, and
    the system group holds no run-level variable.

    Class: NEW CAPABILITY. Plausible wrong implementations: the stem passed through whole to both groups (each value
    in two places, the duplicate D21 forbids), the node group left holding the reductions, or the run-level
    variables swept into the system group by a selection on the wrong side. Reachability: the real
    ``consolidate_to_datatree`` over the golden. Both-states anchor: the node group is checked to hold the run-level
    variable, so the empty-handed pass of a mode that selected nothing fails. Must-fire control: a run-level variable
    written into the system group, and a reduction left in the node group, each make the same check report a
    problem, so a check that always passes fails the controls. Killed by: those.
    Passes under a second correct implementation that splits the stem when the per-scenario store is written.
    """
    from hhemt.constants import (
        LST_OUTFALL_SYSTEM_VARIABLES,
        LST_ROUTING_CONTINUITY_VARIABLES,
        LST_RUN_LEVEL_CONTINUITY_VARIABLES,
    )

    run_level = {*LST_RUN_LEVEL_CONTINUITY_VARIABLES, *LST_ROUTING_CONTINUITY_VARIABLES, *LST_OUTFALL_SYSTEM_VARIABLES}
    a = _ready(golden_extractor, families=("swmm",), clear=False, consolidate=False)
    _add_run_level_variable(a, "flow_continuity_error", 0.25)
    a.process.consolidate_to_datatree()
    system, node = _group(a, _SYSTEM_SUMMARY_GROUP), _group(a, _NODE_SUMMARY_GROUP)
    assert _one_group_violations(system, node, run_level) == []
    assert "flow_continuity_error" in node.data_vars, "the run-level variable stays in the node group"
    assert [int(v) for v in system["event_iloc"].values] == [0, 1, 2]
    assert not set(system.dims) - {"event_iloc"}, "the system group has no spatial dimension"
    assert _one_group_violations(system.assign(flow_continuity_error=node["flow_continuity_error"]), node, run_level)
    assert _one_group_violations(system, node.assign({"system_runoff_max": system["system_runoff_max"]}), run_level)


def test_a_stem_with_no_system_reductions_writes_no_system_summary_group(golden_extractor):
    """Property (control of the selection): a standalone analysis whose node summary stores carry no system
    reduction (stores processed before the change) consolidates its node group unchanged and writes no empty
    ``swmm_only/system/summary`` group.

    Class: NEW CAPABILITY. Plausible wrong implementation: an empty group written whenever the mode is listed, which
    a reader opens as a system with no data. Reachability: the golden before the fixture's completion is restored by
    dropping the twelve variables. Both-states anchor: the test above writes the group from the same stems with the
    variables present. Killed by: the empty group. Passes under a second correct implementation that writes the group
    with NaN.
    """
    a = _ready(golden_extractor, families=("swmm",), clear=False, consolidate=False)
    import zarr

    for event_iloc in a.df_sims.index:
        group = zarr.open_group(str(_scen(a, int(event_iloc)).scen_paths.output_swmm_only_node_summary), mode="r+")
        for name in SYSTEM_SUMMARY:
            del group[name]
        zarr.consolidate_metadata(group.store)
    a.process.consolidate_to_datatree()
    tree = xr.open_datatree(_store(a), engine="zarr", consolidated=False)
    assert "swmm_only" in tree.children and "swmm_node" in tree["swmm_only"].children
    assert "system" not in tree["swmm_only"].children, sorted(tree["swmm_only"].children)


def test_with_the_switch_off_the_system_group_holds_every_event_and_the_store_is_kept(golden_extractor):
    """Property (control of the clearing arm below): with the clearing switch off, ``swmm_only/system/timeseries``
    holds every event with all six series equal to the scenario's own store on its own times, and every scenario
    keeps its ``SWMM_only_system_tseries`` store with no cleared record.

    Class: NEW CAPABILITY. Plausible wrong implementations: the system mode missing from the series mode table (the
    group is absent), a store cleared although the switch is off, or the series written under the node group's name.
    Reachability: the real ``consolidate_timeseries`` over the golden with the standalone arm alone. Both-states
    anchor: the arm below clears the same stores with the switch on. Killed by: those.
    Passes under a second correct implementation that writes the group by a region write.
    """
    a = _ready(golden_extractor, families=("swmm",), clear=False)
    before = {
        e: xr.open_zarr(_scen(a, e).scen_paths.output_swmm_only_system_time_series, consolidated=False).load()
        for e in (0, 1, 2)
    }
    a.process.consolidate_timeseries()
    stored = _group(a, _SYSTEM_GROUP).load()
    assert [int(v) for v in stored["event_iloc"].values] == [0, 1, 2]
    assert sorted(stored.data_vars) == sorted(SYSTEM_ATTRIBUTES)
    assert "elapsed_s" in stored.dims and "date_time" not in stored.dims
    for e in (0, 1, 2):
        for name in SYSTEM_ATTRIBUTES:
            got = stored[name].sel(event_iloc=e).isel(elapsed_s=slice(0, before[e].sizes["date_time"])).values
            assert np.array_equal(got, before[e][name].values, equal_nan=True), (e, name)
        assert Path(_scen(a, e).scen_paths.output_swmm_only_system_time_series).exists()
        field = _scen(a, e).get_log("swmm").consolidated_system_timeseries_cleared
        assert field is None or field.get() is not True


def test_with_the_switch_on_the_system_store_is_cleared_after_its_slice_passes(golden_extractor):
    """Property: with the clearing switch on, every standalone scenario's ``SWMM_only_system_tseries`` store is
    deleted and ``consolidated_system_timeseries_cleared`` is True in the standalone model's log (the pending record
    gone, and no such field in the coupled model's log) once its slice of ``swmm_only/system/timeseries`` has been
    read back equal, and the group still holds every event.

    Class: NEW CAPABILITY. Plausible wrong implementations: the system store left out of the clearing (a duplicate of
    the group, which D21 forbids), the record written to the coupled log, or the store deleted before its read-back.
    Reachability: the real ``consolidate_timeseries``. Both-states anchor: the switch-off arm above keeps the same
    stores, and the arm below keeps the store when the read-back fails. Killed by: those. Passes under a second
    correct implementation that clears the three stores of the model in one delete.
    """
    a = _ready(golden_extractor, families=("swmm",), clear=True)
    a.process.consolidate_timeseries()
    for e in (0, 1, 2):
        scen = _scen(a, e)
        assert not Path(scen.scen_paths.output_swmm_only_system_time_series).exists(), e
        log = scen.get_log("swmm")
        assert log.consolidated_system_timeseries_cleared.get() is True
        assert not log.consolidated_system_timeseries_clearing_pending.get()
        assert getattr(scen.get_log("tritonswmm"), "consolidated_system_timeseries_cleared", None) is None
    assert [int(v) for v in _group(a, _SYSTEM_GROUP)["event_iloc"].values] == [0, 1, 2]


def test_a_failed_readback_keeps_the_system_store_and_leaves_its_record_unset(golden_extractor, monkeypatch):
    """Property (must-fire control of the clearing): when the read-back of the system group fails for event 0, the
    consolidation raises ``ProcessingError``, the system store of event 0 exists and its cleared record is not True,
    and the node and link stores of the same scenario (verified before it, in the same model) are kept too, because a
    model's stores are cleared only once every one of its slices has passed.

    Class: NEW CAPABILITY. Plausible wrong implementation: a system store deleted without its own read-back, or one
    cleared by the node store's passing. Reachability: the real consolidation with ``_stored_slice_equals`` answering
    False for the system group alone. Both-states anchor: the test above clears it with the same code unpatched.
    Killed by: those. Passes under a second correct implementation that clears each store as its own slice passes,
    provided the failing store and its record are kept.
    """
    import hhemt.processing_analysis as pa

    cls = pa.TRITONSWMM_analysis_post_processing
    a = _ready(golden_extractor, families=("swmm",), clear=True)
    real = cls._stored_slice_equals

    def failing(store, group, expected, event_iloc):
        if group == _SYSTEM_GROUP:
            return False
        return real(store, group, expected, event_iloc)

    monkeypatch.setattr(cls, "_stored_slice_equals", staticmethod(failing))
    with pytest.raises(Exception) as err:
        a.process.consolidate_timeseries()
    assert type(err.value).__name__ == "ProcessingError"
    scen = _scen(a, 0)
    assert Path(scen.scen_paths.output_swmm_only_system_time_series).exists()
    assert scen.get_log("swmm").consolidated_system_timeseries_cleared is None or (
        scen.get_log("swmm").consolidated_system_timeseries_cleared.get() is not True
    )
    assert Path(scen.scen_paths.output_swmm_only_node_time_series).exists()
    assert Path(scen.scen_paths.output_swmm_only_link_time_series).exists()


def test_a_cleared_system_store_reads_as_processed_and_an_unset_record_does_not(golden_extractor):
    """Property: after a clearing consolidation the standalone model's processed-state check reads the system store
    as processed from the cleared record alone (the processing records are removed first), and with the record set
    False and nothing else recorded it reads not processed.

    Class: NEW CAPABILITY. Plausible wrong implementation: the system arm omitted from the predicate, which would
    re-derive a cleared store from a raw output the clearing may have removed. Reachability: the real consolidation,
    then the real predicate over the real log. Both-states anchor: the unset arm below. Killed by: that.
    Passes under a second correct implementation that reads a combined record.
    """
    from hhemt.process_simulation import TRITONSWMM_sim_post_processing

    a = _ready(golden_extractor, families=("swmm",), clear=True)
    a.process.consolidate_timeseries()
    scen = _scen(a, 1)
    log = scen.get_log("swmm")
    proc = TRITONSWMM_sim_post_processing(scen.run, model_log=log)
    log.processing_log.outputs.clear()
    assert proc._swmm_system_outputs_processed() is True
    log.consolidated_system_timeseries_cleared.set(False)
    assert proc._swmm_system_outputs_processed() is False
