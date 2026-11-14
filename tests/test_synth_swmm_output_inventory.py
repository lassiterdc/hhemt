"""The binary-output series of both model families, on the synthetic model (compile tier; Rivanna scratch only).

The synthetic multisim is run by the solver, so a test that reaches ``synthetic_multisim_completed`` compiles and runs
TRITON-SWMM and belongs on Rivanna scratch (the venue rule), never on the laptop. This module asserts, on that tree:

* each family's node store carries ``volume`` and ``lateral_inflow`` and its link store ``link_volume``, on the
  ``date_time`` axis of ``depth``;
* the coupled ``inflow_flow`` is within the report's print rounding (0.0005 m3/s) of the value the report prints for
  the same node and time, which holds only if the series were read from the binary output and not from the report;
* and, as a must-fire control, at least one coupled ``inflow_flow`` value differs from its printed value by MORE
  than 0 (a series read from the report would equal it exactly).

The printed values are read by a helper in this module that splits the ``Node Time Series Results`` body on
whitespace; the toolkit's own report series reader is removed and may not be used here. The helper's own control
(``test_the_printed_series_helper_reproduces_the_reference_stores``) runs without a solver on the tracked reference
report.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import tests.utils_for_testing as tst_ut

REF_DIR = Path(__file__).resolve().parents[1] / "test_data" / "swmm_refactoring_reference"
PRINT_ROUNDING = 5e-4 + 1e-9


def _printed_node_inflow(rpt: Path) -> xr.DataArray:
    """The ``Inflow`` column of the report's Node Time Series Results body as a (node_id, date_time) array.

    A test-local reader: it splits each data row on whitespace and takes the first value column, which the report
    prints for every node and report step.
    """
    rows: dict[str, dict[pd.Timestamp, float]] = {}
    current = None
    in_node_section = False
    for line in rpt.read_text(encoding="latin-1").splitlines():
        if "Node Time Series Results" in line:
            in_node_section = True
        if "Link Time Series Results" in line:
            break
        if not in_node_section:
            continue
        match = re.match(r"\s*<<<\s+Node\s+(\S+)\s+>>>", line)
        if match:
            current = rows.setdefault(match.group(1), {})
            continue
        tokens = line.split()
        if current is not None and len(tokens) >= 6 and re.match(r"\d\d/\d\d/\d{4}$", tokens[0]):
            current[pd.Timestamp(f"{tokens[0]} {tokens[1]}")] = float(tokens[2])
    frame = pd.DataFrame(rows).T.sort_index()
    return xr.DataArray(
        frame.to_numpy(float),
        dims=("node_id", "date_time"),
        coords={"node_id": frame.index.to_numpy(str), "date_time": pd.DatetimeIndex(frame.columns)},
    )


def test_the_printed_series_helper_reproduces_the_reference_stores():
    """Property: the whitespace-splitting helper reproduces the committed reference ``inflow_flow`` store.

    Class: the tracked reference report (1135 nodes, 11 report steps). Why: the compile-tier comparison below is
    only as good as its reader of the printed values, and the toolkit's reader is removed. What kills it: a reader
    that takes another column or mis-aligns the steps. Second correct implementation: ``pandas.read_csv`` over the
    same rows. Control: a value offset by 0.001 no longer matches.
    """
    printed = _printed_node_inflow(REF_DIR / "hydraulics.rpt")
    reference = xr.open_dataset(REF_DIR / "SWMM_node_tseries.zarr", engine="zarr", consolidated=False)["inflow_flow"]
    aligned = printed.sel(node_id=reference["node_id"].values, date_time=reference["date_time"].values)
    assert np.allclose(aligned.values, reference.values, atol=PRINT_ROUNDING, equal_nan=True)
    shifted = aligned.values.copy()
    shifted.flat[0] += 0.001
    assert not np.allclose(shifted, reference.values, atol=PRINT_ROUNDING, equal_nan=True)


def _open_store(path: Path) -> xr.Dataset:
    if path.suffix == ".zarr":
        return xr.open_zarr(path, consolidated=False).load()
    return xr.open_dataset(path).load()


def _processed(analysis, event_iloc: int = 0):
    return analysis._retrieve_sim_run_processing_object(event_iloc)


@pytest.mark.slow  # solver-bearing: runs on the cluster; the laptop capture selects `not slow`
def test_both_families_carry_the_binary_output_series_on_the_axis_of_depth(synthetic_multisim_completed):
    """Property: each enabled SWMM family's node store carries ``volume`` and ``lateral_inflow`` and its link store
    ``link_volume``, each on the dimensions and ``date_time`` axis of ``depth``.

    Class: the coupled (``tritonswmm``) family and, when the multisim enables it, the standalone family, of the
    synthetic model. Why: the three series exist only in SWMM's binary output, and both families now read every
    series from it. What kills it: a family still reading the report's four series. Second correct implementation:
    reading the series through ``swmm.toolkit.output`` instead of pyswmm. Control: the capacity series is named
    ``capacity_setting`` for the coupled family and ``capacity`` for the standalone one, never both.
    """
    analysis = synthetic_multisim_completed
    proc = _processed(analysis)
    paths = proc.scen_paths
    families = [("tritonswmm", paths.output_tritonswmm_node_time_series, paths.output_tritonswmm_link_time_series)]
    if "swmm" in tst_ut.get_enabled_model_types(analysis):
        families.append(("swmm", paths.output_swmm_only_node_time_series, paths.output_swmm_only_link_time_series))
    assert families[0][1] is not None and Path(families[0][1]).exists(), "the coupled node store was not written"
    for family, node_path, link_path in families:
        nodes, links = _open_store(Path(node_path)), _open_store(Path(link_path))
        for name in ("volume", "lateral_inflow"):
            assert name in nodes, (family, name)
            assert nodes[name].dims == nodes["depth"].dims, (family, name)
            assert np.array_equal(nodes[name]["date_time"].values, nodes["depth"]["date_time"].values)
        assert "link_volume" in links and links["link_volume"].dims == links["flow"].dims
        capacity = "capacity_setting" if family == "tritonswmm" else "capacity"
        other = "capacity" if family == "tritonswmm" else "capacity_setting"
        assert capacity in links and other not in links, family


@pytest.mark.slow  # solver-bearing: runs on the cluster; the laptop capture selects `not slow`
def test_the_coupled_inflow_series_come_from_the_binary_output_not_the_report(synthetic_multisim_completed):
    """Property: the coupled ``inflow_flow`` is within 0.0005 m3/s of the report's printed value at every node and
    time, and at least one value differs from its printed value by more than 0.

    Class: every node and report step of the first scenario's coupled node store. Why: the report prints the binary
    value rounded to three decimals, so a series read from the binary file is within half a unit of the last printed
    digit of the report, and a series read from the report equals it exactly. What kills it: the coupled family
    still reading the report (the second assertion), or reading another variable (the first). Second correct
    implementation: comparing against the report through a pandas merge. Not established: a model whose every
    inflow is a multiple of 0.001 would make the difference zero everywhere; the synthetic model's inflows are not.
    """
    proc = _processed(synthetic_multisim_completed)
    paths = proc.scen_paths
    rpt = Path(paths.swmm_hydraulics_rpt)
    assert "Node Time Series Results" in rpt.read_text(encoding="latin-1"), (
        f"{rpt} has no time-series body (it was reclaimed), so the printed values cannot be read"
    )
    store = _open_store(Path(paths.output_tritonswmm_node_time_series))["inflow_flow"]
    printed = _printed_node_inflow(rpt)
    stored = store.sel(node_id=printed["node_id"].values, date_time=printed["date_time"].values)
    difference = np.abs(stored.values - printed.values)
    assert np.nanmax(difference) <= PRINT_ROUNDING, f"largest difference {np.nanmax(difference)}"
    assert np.nanmax(difference) > 0.0, (
        "every stored inflow equals its printed value: the store was read from the report"
    )
