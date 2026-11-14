"""The standalone SWMM system series, read against inputs the sibling test file cannot reach.

The committed reference output ``test_data/swmm_binary_reference/hydraulics.out`` holds zero in five of its six system
series, so a test that compares each stored series with the toolkit's values on that file cannot tell one of those five
from another. These tests give every system attribute its own non-zero value in every report period, by rewriting the
system block of a COPY of the reference output, and they read the consolidated system groups of a golden analysis.
No solver is compiled or run; every write is under ``tmp_path``.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import xarray as xr

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "test_data" / "swmm_binary_reference" / "hydraulics.out"
N_SYSTEM_VARS = 15
EPILOG_BYTES = 24

#: stored name -> member of ``swmm.toolkit.shared_enum.SystemAttribute``, written out from the ruled description of the
#: six series (flooding rate, outfall flow, stored volume, lateral inflow, runoff, rainfall).
EXPECTED_ATTRIBUTE = {
    "system_flooding_rate": "FLOOD_LOSSES",
    "system_outfall_flow": "OUTFALL_FLOWS",
    "system_stored_volume": "VOLUME_STORED",
    "system_lateral_inflow": "TOTAL_LATERAL_INFLOW",
    "system_runoff": "RUNOFF_FLOW",
    "system_rainfall": "RAINFALL",
}
#: stored name -> units of the series, and so of its ``_max`` and ``_last`` reductions.
EXPECTED_UNITS = {
    "system_flooding_rate": "m3 s-1",
    "system_outfall_flow": "m3 s-1",
    "system_stored_volume": "m3",
    "system_lateral_inflow": "m3 s-1",
    "system_runoff": "m3 s-1",
    "system_rainfall": "mm h-1",
}


def _attribute_index(attribute_name: str) -> int:
    """Position of one system attribute within a period's system block."""
    from swmm.toolkit import shared_enum

    return int(getattr(shared_enum.SystemAttribute, attribute_name).value)


def _expected_value(attribute_name: str, period: int) -> float:
    """The value the rewritten file holds for one attribute in one period: distinct per attribute and per period."""
    return 100.0 * (_attribute_index(attribute_name) + 1) + period


def _rewritten_copy(destination: Path) -> Path:
    """Copy the reference output with every system variable of every period set to ``_expected_value``.

    The system block is the last ``N_SYSTEM_VARS`` float32 values of each period, and the period size follows from the
    epilog (the output start position and the period count), so the geometry is derived from the file. Before any byte
    is written the block read from those offsets is compared with the toolkit's own reading of the unmodified file,
    and the stored volume (non-zero in the reference) is required to be non-zero, so a wrong offset fails here.
    """
    from pyswmm import Output
    from swmm.toolkit import output, shared_enum

    raw = bytearray(OUT.read_bytes())
    _id_pos, _inp_pos, out_pos, n_periods, _error, _magic = struct.unpack("<6i", bytes(raw[-EPILOG_BYTES:]))
    period_bytes, remainder = divmod(len(raw) - EPILOG_BYTES - out_pos, n_periods)
    assert remainder == 0, "the epilog geometry does not divide into whole periods"
    block_start = [out_pos + p * period_bytes + period_bytes - 4 * N_SYSTEM_VARS for p in range(n_periods)]
    with Output(str(OUT)) as out:
        assert int(output.get_times(out.handle, shared_enum.Time.NUM_PERIODS)) == n_periods
        for p in range(n_periods):
            from_bytes = struct.unpack_from(f"<{N_SYSTEM_VARS}f", raw, block_start[p])
            from_toolkit = output.get_system_result(out.handle, p, 0)
            assert np.allclose(from_bytes, from_toolkit, rtol=1e-6), f"period {p}: the offsets do not find the block"
    stored_volume = struct.unpack_from("<f", raw, block_start[0] + 4 * _attribute_index("VOLUME_STORED"))[0]
    assert stored_volume != 0.0, "control: the reference holds stored volume, so the offsets land on real data"
    for p in range(n_periods):
        for attribute_name in EXPECTED_ATTRIBUTE.values():
            struct.pack_into(
                "<f", raw, block_start[p] + 4 * _attribute_index(attribute_name), _expected_value(attribute_name, p)
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(bytes(raw))
    return destination


def _mapping_problems(ds: xr.Dataset) -> list[str]:
    """Stored names whose values are not the rewritten file's values for the attribute the name is ruled to hold."""
    problems = []
    if sorted(ds.data_vars) != sorted(EXPECTED_ATTRIBUTE):
        problems.append(f"variables {sorted(ds.data_vars)}")
    for name, attribute_name in EXPECTED_ATTRIBUTE.items():
        if name not in ds.data_vars:
            continue
        want = np.array([_expected_value(attribute_name, p) for p in range(ds.sizes["date_time"])])
        if not np.array_equal(ds[name].values, want):
            problems.append(f"{name}: {ds[name].values.tolist()} is not {attribute_name} {want.tolist()}")
    return problems


def test_each_system_series_name_holds_its_own_attribute_in_every_report_period(tmp_path, monkeypatch):
    """Property: on a binary output where every system attribute holds a different non-zero value in every period,
    each of the six stored series holds exactly the values of the attribute it is ruled to hold, over all periods.

    Class: the six stored system series (names against attributes), and the periods of the file. Why required: the
    committed reference output holds zero in five of the six attributes, so a pairing of a name with a wrong
    attribute among those five stores zeros either way and reads as correct, while the consolidated tree would carry
    a flooding rate that is really the runoff. Kills it: exchanging two attributes in
    ``constants.SWMM_SYSTEM_SERIES`` (the control arm below applies that change to the module the reader uses and
    requires the same check to report it); reading a window that drops the last period. Second correct
    implementation: one that reads each period with ``get_system_result``, which this file does not constrain.
    Reachability: the real reader over a real binary output whose system block was rewritten; the geometry is
    verified against the toolkit before any byte changes. Both-states anchor: values are compared with what the file
    holds, not with any name or message the change introduces. Invocation isolated: not isolatable from this tree -
    discrimination shown instead: the swap in the control arm turns this red.
    """
    from hhemt.swmm_output_parser import return_system_time_series_results_from_outfile

    rewritten = _rewritten_copy(tmp_path / "rewritten" / "hydraulics.out")
    ds = return_system_time_series_results_from_outfile(rewritten)
    assert ds.sizes["date_time"] == 12
    assert _mapping_problems(ds) == []

    import hhemt.swmm_output_parser as parser

    swapped = {"system_outfall_flow": "RUNOFF_FLOW", "system_runoff": "OUTFALL_FLOWS"}
    mutant = tuple((name, swapped.get(name, attribute)) for name, attribute in parser.SWMM_SYSTEM_SERIES)
    assert mutant != parser.SWMM_SYSTEM_SERIES, "control: the mutant differs from the real table"
    monkeypatch.setattr(parser, "SWMM_SYSTEM_SERIES", mutant)
    assert _mapping_problems(return_system_time_series_results_from_outfile(rewritten)), "control: the swap must show"


def test_the_system_series_time_axis_is_the_node_and_link_time_axis_of_the_same_output():
    """Property: for one binary output, the system series are on the same time axis as the node and the link series,
    value for value.

    Class: the three series kinds a standalone scenario stores from one output file. Why required: the consolidation
    appends every kind of a scenario on one ``elapsed_s`` axis built from the scenario's own times, and a system
    series offset from the node series by a report step would align a flooding rate with the wrong water depth.
    Kills it: building the system axis from the report start instead of the first report time; dropping or adding a
    period. Second correct implementation: one that returns the node dataset's axis for the system series.
    Reachability: the real readers over the committed output. Both-states anchor: axes are compared with each other,
    whatever the count. Invocation isolated: not isolatable from this tree - discrimination shown instead: a system
    reader that drops the first period turns the length comparison red.
    """
    from hhemt.swmm_output_parser import (
        return_node_time_series_results_from_outfile,
        return_system_time_series_results_from_outfile,
    )

    system = return_system_time_series_results_from_outfile(OUT)
    nodes, links = return_node_time_series_results_from_outfile(OUT)
    assert system.sizes["date_time"] == 12, "control: the reference holds twelve report periods"
    assert np.array_equal(system["date_time"].values, nodes["date_time"].values)
    assert np.array_equal(system["date_time"].values, links["date_time"].values)


def _consolidated_system_problems(summary: xr.Dataset, series: xr.Dataset) -> list[str]:
    """What breaks the rule that every variable of the consolidated system groups states the units of its series."""
    problems = []
    for name, units in EXPECTED_UNITS.items():
        if series[name].attrs.get("units") != units:
            problems.append(f"timeseries {name}: units {series[name].attrs.get('units')!r}, want {units!r}")
        for reduction in ("max", "last"):
            reduced = f"{name}_{reduction}"
            if summary[reduced].attrs.get("units") != units:
                problems.append(f"summary {reduced}: units {summary[reduced].attrs.get('units')!r}, want {units!r}")
        if summary[f"{name}_max"].attrs.get("cell_methods") != "time: maximum":
            problems.append(f"summary {name}_max: no 'time: maximum' cell method")
    return problems


def _rebuild_event_through_the_real_export(a, event_iloc: int, tmp_path: Path) -> None:
    """Replace one event's system store and node summary store with ones the real export writes from the reference.

    The golden's system stores and the twelve reductions in its summary stores are written by a test fixture, so the
    attributes they carry were never produced by the code under review. Every event is rebuilt because the
    concatenation drops only conflicting attributes: one scenario still carrying the fixture's units would supply a
    unit that a scenario written without it lacks. The raw output is a COPY of the reference, so the raw-output
    clearing cannot touch the committed file.
    """
    import shutil

    from hhemt.process_simulation import TRITONSWMM_sim_post_processing
    from hhemt.scenario import TRITONSWMM_scenario

    scen = TRITONSWMM_scenario(event_iloc, a)
    raw = tmp_path / f"raw_{event_iloc}"
    raw.mkdir()
    shutil.copy(OUT, raw / "hydraulics.out")
    shutil.copy(OUT.with_suffix(".rpt"), raw / "hydraulics.rpt")
    paths = scen.scen_paths
    paths.swmm_full_out_file = raw / "hydraulics.out"
    log = scen.get_log("swmm")
    proc = TRITONSWMM_sim_post_processing(scen.run, model_log=log)
    system_store = Path(paths.output_swmm_only_system_time_series)
    shutil.rmtree(system_store)
    log.processing_log.outputs.pop(system_store.name, None)
    proc._export_SWMM_outputs("swmm", override_clear_raw="none")
    summary_store = Path(paths.output_swmm_only_node_summary)
    shutil.rmtree(summary_store)
    log.processing_log.outputs.pop(summary_store.name, None)
    proc._export_SWMM_summaries("swmm")


def test_every_variable_of_the_consolidated_system_groups_states_the_units_of_its_series(golden_extractor, tmp_path):
    """Property: after every event's stores are written by the real export and a real consolidation runs, all six
    variables of ``swmm_only/system/timeseries`` and all twelve of ``swmm_only/system/summary`` carry the units ruled
    for their series, and each ``_max`` carries the maximum cell method.

    Class: the eighteen variables of the two consolidated system groups. Why required: the consolidation keeps the
    attributes the per-scenario store was written with, and re-applies a name map only to variables that have an
    entry; a variable the export wrote without units therefore reaches the tree with a name-derived long name and no
    units. The per-scenario stores of the golden carry units a fixture wrote, so only the export's own stores test
    this. Kills it: a CF map without the system names, or an export that merges the reductions after the
    attributes are applied. Second correct implementation: one that stamps the units again at consolidation.
    Reachability: the real export and consolidation over the golden. Both-states anchor: attributes are read from
    the stored groups. Invocation isolated: not isolatable from this tree - discrimination shown instead: removing
    the ``system_runoff_max`` entry of the CF map turns this red, and the control arm below removes one attribute
    and requires the same check to report it.
    """
    from tests.test_family_parity_consolidation import _ready

    a = _ready(golden_extractor, families=("swmm",), clear=False)
    for event_iloc in (0, 1, 2):
        _rebuild_event_through_the_real_export(a, event_iloc, tmp_path)
    a.process.consolidate_to_datatree()
    a.process.consolidate_timeseries()
    store = Path(a.analysis_paths.analysis_datatree_zarr)
    summary = xr.open_zarr(store, group="swmm_only/system/summary", consolidated=False).load()
    series = xr.open_zarr(store, group="swmm_only/system/timeseries", consolidated=False).load()
    assert _consolidated_system_problems(summary, series) == []

    stripped = summary.copy(deep=True)
    stripped["system_runoff_max"].attrs.pop("units")
    assert _consolidated_system_problems(stripped, series), "control: a missing unit must show"
