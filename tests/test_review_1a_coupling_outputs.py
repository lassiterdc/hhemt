"""Tests for coupling outputs, with one section per chunk.

The first section covers ``hhemt.coupling_exchange``, the decoder of the coupled solver's exchange
side-file. Each test here quantifies over an input class that ``tests/test_coupling_exchange.py``
does not reach, and every expected value is worked out by hand in the docstring or computed by a
route that does not call the code under test.

The second section covers the coupled node outputs (``_attach_coupling_outputs``, the side-file reclaim guard,
the summary reduction, the CF flag metadata) and the Link Flow Summary reader against rows stock SWMM
printed. Nothing in either section compiles or runs TRITON-SWMM.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hhemt.coupling_exchange import FT3_TO_M3_FACTOR, read_exchange_sidefile, to_reporting_intervals

_V2_MAGIC = 0x53574D32
_CFS_TO_CMS = 0.028316847


def _write_side_file(path: Path, dt: np.ndarray, q: np.ndarray) -> Path:
    """Write a 64-bit side-file in the solver's V2 layout: 12-byte header, then (dt, q[0..n-1]) records."""
    header = np.array([_V2_MAGIC, q.shape[1], 8], dtype="<i4")
    records = np.column_stack([dt, q]).astype("<f8")
    path.write_bytes(header.tobytes() + records.tobytes())
    return path


@pytest.mark.parametrize("block_records", [1, 2, 1000])
@pytest.mark.parametrize("last_step_crosses_the_last_report_time", [False, True])
@pytest.mark.parametrize("steps_after_the_last_report_time", [1, 2, 3])
def test_steps_that_start_after_the_last_report_time_belong_to_no_interval(
    tmp_path, steps_after_the_last_report_time, last_step_crosses_the_last_report_time, block_records
):
    """Property: a side-file that runs past its last report time yields the same interval means as one that stops there.

    Input class: every side-file whose final steps begin at or after the last report time, with
    1 to 3 such steps, with and without a step that straddles the last report time, read in blocks
    of 1, 2 and 1000 steps. Plausible wrong implementations: indexing an interval that does not
    exist (an IndexError on the first such step), or clipping the interval index to the last
    interval, which pours the tail's volume into the final mean. Both make the last mean depend on
    how long the solver kept running after the last print.
    Why required: a window whose length is not a whole number of reporting steps leaves a partial
    interval at the end, and the solver's final steps can then begin after the last print. The tail
    flows are 100, -60 and 80 cfs against 2 to 10 cfs in the body, chosen so that no prefix of
    the tail sums to zero (two equal and opposite tail steps would cancel and hide a misattributed
    tail), so any share of the tail that reaches an interval moves a mean by far more than the
    comparison tolerance.
    Killed by: removing the `first_edge < n_interval` guard, or clipping the index. Passes under a
    second correct implementation that differences a cumulative volume curve evaluated at the
    report times, because that curve has no interval after the last report time either.

    Hand arithmetic, R = 120 s, report times 120, 240, 360 s. Body steps end at 50, 150, 220, 340 s
    at 2, 4, 6, 8 cfs; the next body step is 20 s at 10 cfs (ending at 360 s) or 60 s at 10 cfs
    (ending at 400 s, so it straddles the last report time and only its first 20 s count).
      (0, 120]   : 50x2 + 70x4                = 380
      (120, 240] : 30x4 + 70x6 + 20x8         = 700
      (240, 360] : 100x8 + 20x10              = 1000     (identical in both variants)
    Every tail step is 45 s long and lies wholly after 360 s; its flows are 100, -60 and 80 cfs in order.
    """
    last_body_step_s = 60.0 if last_step_crosses_the_last_report_time else 20.0
    body_dt = [50.0, 100.0, 70.0, 120.0, last_body_step_s]
    body_q = [2.0, 4.0, 6.0, 8.0, 10.0]
    tail_q = [100.0, -60.0, 80.0][:steps_after_the_last_report_time]
    dt = np.array(body_dt + [45.0] * steps_after_the_last_report_time)
    q = np.array(body_q + tail_q)[:, None]
    side = _write_side_file(tmp_path / "tail.bin", dt, q)
    nodes = tmp_path / "coupling_nodes.txt"
    nodes.write_text("A\n")
    series = read_exchange_sidefile(side, nodes)
    out = to_reporting_intervals(series, [120.0, 240.0, 360.0], block_records=block_records)
    assert _CFS_TO_CMS == FT3_TO_M3_FACTOR
    np.testing.assert_allclose(
        out.exchange_mean_cms[:, 0], np.array([380.0, 700.0, 1000.0]) / 120.0 * _CFS_TO_CMS, rtol=1e-12
    )
    # The volume the intervals hold is the volume of the step series over (0, 360] s and no more.
    np.testing.assert_allclose(out.exchange_mean_cms[:, 0].sum() * 120.0, 2080.0 * _CFS_TO_CMS, rtol=1e-12)


# ----------------------------------------------------------------------------------------------------
# Coupled node outputs
# ----------------------------------------------------------------------------------------------------

_NODE_LIST_ORDER = ("n3", "n1")  # file order differs from the SWMM node order n1, n2, n3 on purpose


def _coupled_node_dataset() -> xr.Dataset:
    """A node time-series dataset shaped like the coupled SWMM export: three nodes, three 120 s report times."""
    times = pd.to_datetime("2020-01-01") + pd.to_timedelta([120, 240, 360], unit="s")
    return xr.Dataset(
        {"depth": (("date_time", "node_id"), np.zeros((3, 3)))},
        coords={"date_time": times, "node_id": ["n1", "n2", "n3"]},
    )


def _coupled_scenario(tmp_path: Path, *, flux: bool, side_file: bool = True, names=_NODE_LIST_ORDER):
    """A post-processing object holding only what ``_attach_coupling_outputs`` reads, and its input file.

    The side-file holds four steps of 60, 60, 120, 120 s. Slot 0 (listed node n3) carries 10, 10, 0, 5 cfs and
    slot 1 (listed node n1) carries 1, -1, 2, 2 cfs, so the node order of the file and of the dataset differ.
    """
    from hhemt.process_simulation import TRITONSWMM_sim_post_processing

    swmm_dir = tmp_path / "out_tritonswmm" / "swmm"
    swmm_dir.mkdir(parents=True)
    rpt = swmm_dir / "hydraulics.rpt"
    rpt.write_text("  Starting Date ............ 01/01/2020 00:00:00\n")
    (swmm_dir / "coupling_nodes.txt").write_text("".join(f"{n}\n" for n in names))
    if side_file:
        _write_side_file(
            swmm_dir / "x_exchange_replay.bin",
            np.array([60.0, 60.0, 120.0, 120.0]),
            np.array([[10.0, 1.0], [10.0, -1.0], [0.0, 2.0], [5.0, 2.0]]),
        )
    proc = TRITONSWMM_sim_post_processing.__new__(TRITONSWMM_sim_post_processing)
    proc.scen_paths = SimpleNamespace(out_tritonswmm=tmp_path / "out_tritonswmm", swmm_hydraulics_rpt=rpt)
    proc._analysis = SimpleNamespace(cfg_analysis=SimpleNamespace(toggle_coupling_exchange_flow=flux))
    return proc, swmm_dir / "hydraulics.inp"


def test_the_flag_follows_the_listed_nodes_and_the_flux_follows_the_side_file_slots(tmp_path):
    """Property: with the flux on, each side-file slot lands on the dataset node the list names at that line, the
    flag is 1 exactly there, and every other node holds NaN in the flux and in all eight statistics.

    Input class: a coupling list whose line order differs from the dataset's node order (n3 then n1 against
    n1, n2, n3), which is the case whenever the list is not a prefix of the SWMM node order. Plausible wrong
    implementations: assigning the slots in dataset order or sorted order (every value lands on the wrong node);
    filling uncoupled nodes with 0 instead of NaN (a node that never exchanged reads as a measured zero).
    Why required: the flux and its statistics are read by node name downstream; a swapped pair is
    indistinguishable from a real result by any check on shape or units.
    Killed by: sorting the names before computing slots, or `np.zeros` for the fill. Passes under a second
    correct implementation that builds the flux with `xr.DataArray.reindex` on `node_id`.

    Hand arithmetic, R = 120 s, steps of 60, 60, 120, 120 s. n3: intervals 10, 0, 5 cfs; peak to the network 10 cfs
    reached by the step ending at 60 s (1 min). n1: intervals (60x1 + 60x-1)/120 = 0, 2, 2 cfs; peak to the network
    2 cfs reached by the step ending at 240 s (4 min); peak to the surface 1 cfs (step ending at 120 s, 2 min).
    """
    from hhemt.process_simulation import TRITONSWMM_sim_post_processing

    proc, inp = _coupled_scenario(tmp_path, flux=True)
    out = TRITONSWMM_sim_post_processing._attach_coupling_outputs(proc, _coupled_node_dataset(), inp)
    assert out["is_coupling_node"].dtype == np.int8
    assert out["is_coupling_node"].values.tolist() == [1, 0, 1]
    flow = out["coupling_exchange_flow"]
    assert flow.dims == ("date_time", "node_id")
    np.testing.assert_allclose(flow.sel(node_id="n3").values, np.array([10.0, 0.0, 5.0]) * _CFS_TO_CMS, rtol=1e-12)
    np.testing.assert_allclose(flow.sel(node_id="n1").values, np.array([0.0, 2.0, 2.0]) * _CFS_TO_CMS, rtol=1e-12)
    assert np.isnan(flow.sel(node_id="n2").values).all()
    stat_peak = out["coupling_exchange_peak_to_network"]
    np.testing.assert_allclose(stat_peak.values[[0, 2]], np.array([2.0, 10.0]) * _CFS_TO_CMS, rtol=1e-12)
    np.testing.assert_allclose(out["coupling_exchange_peak_to_surface"].values[0], 1.0 * _CFS_TO_CMS, rtol=1e-12)
    assert out["coupling_exchange_time_of_peak_to_network"].values[[0, 2]].tolist() == [4.0, 1.0]
    assert out["coupling_exchange_time_of_peak_to_surface"].values[0] == 2.0
    for name in (
        "coupling_exchange_peak_to_network",
        "coupling_exchange_peak_to_surface",
        "coupling_exchange_volume_to_network",
        "coupling_exchange_volume_to_surface",
        "coupling_exchange_volume_net",
        "coupling_exchange_time_of_peak_to_network",
        "coupling_exchange_time_of_peak_to_surface",
        "coupling_exchange_duration_to_surface",
    ):
        assert out[name].dims == ("node_id",)
        assert np.isnan(out[name].values[1]), name
    assert "interval" in out["coupling_exchange_flow"].attrs["cell_methods"]


def test_the_flag_alone_is_written_when_the_flux_is_off_even_without_a_side_file(tmp_path):
    """Property: with the flux off the dataset gains `is_coupling_node` and nothing else, and no side-file is read.

    Input class: a coupled scenario whose side-file was already reclaimed (none on disk) with the toggle off, the
    state of every run made before the flux existed. Plausible wrong implementation: decoding the side-file
    whatever the toggle says, which would raise on every such run. Must-fire control: the same scenario with the
    toggle ON raises `ProcessingError`, so the absence of an error above is the toggle's doing.
    Killed by: dropping the `if not self._coupling_exchange_flow_on(): return` branch. Passes under a second
    correct implementation that attaches the flux in a separate method called only when the toggle is on.
    """
    from hhemt.exceptions import ProcessingError
    from hhemt.process_simulation import TRITONSWMM_sim_post_processing

    proc, inp = _coupled_scenario(tmp_path / "off", flux=False, side_file=False)
    out = TRITONSWMM_sim_post_processing._attach_coupling_outputs(proc, _coupled_node_dataset(), inp)
    assert sorted(out.data_vars) == ["depth", "is_coupling_node"]
    assert out["is_coupling_node"].values.tolist() == [1, 0, 1]
    proc_on, inp_on = _coupled_scenario(tmp_path / "on", flux=True, side_file=False)
    with pytest.raises(ProcessingError):
        TRITONSWMM_sim_post_processing._attach_coupling_outputs(proc_on, _coupled_node_dataset(), inp_on)


def test_a_store_is_never_written_when_a_listed_node_or_the_one_side_file_is_missing(tmp_path):
    """Property: a listed coupling node absent from the SWMM output, and a flux request with no side-file or with
    two, are each refused with `ProcessingError` instead of returning a dataset without the variable.

    Input class: the three ways the attach can find its inputs wrong. Plausible wrong implementation: skipping the
    missing node or the missing file and returning the dataset unchanged, which writes a store that lacks the
    variable the analysis was configured to carry. Must-fire control: the identical scenario with every input
    present returns a dataset that has the flux, so each refusal is caused by the one thing removed.
    Killed by: replacing any `raise` with a silent return. Passes under a second correct implementation that
    raises a subclass of `ProcessingError`.
    """
    from hhemt.exceptions import ProcessingError
    from hhemt.process_simulation import TRITONSWMM_sim_post_processing

    attach = TRITONSWMM_sim_post_processing._attach_coupling_outputs
    proc, inp = _coupled_scenario(tmp_path / "ok", flux=True)
    assert "coupling_exchange_flow" in attach(proc, _coupled_node_dataset(), inp).data_vars

    proc, inp = _coupled_scenario(tmp_path / "missing_node", flux=True, names=("n3", "n9"))
    with pytest.raises(ProcessingError):
        attach(proc, _coupled_node_dataset(), inp)

    proc, inp = _coupled_scenario(tmp_path / "no_file", flux=True, side_file=False)
    with pytest.raises(ProcessingError):
        attach(proc, _coupled_node_dataset(), inp)

    proc, inp = _coupled_scenario(tmp_path / "two_files", flux=True)
    swmm_dir = inp.parent
    (swmm_dir / "y_exchange_replay.bin").write_bytes((swmm_dir / "x_exchange_replay.bin").read_bytes())
    with pytest.raises(ProcessingError):
        attach(proc, _coupled_node_dataset(), inp)


@pytest.mark.parametrize("flux_on", [False, True])
@pytest.mark.parametrize("node_store_written", [False, True])
def test_the_side_file_is_reclaimed_unless_the_flux_is_on_and_the_node_store_is_not_yet_written(
    tmp_path, flux_on, node_store_written
):
    """Property: `_clear_raw_outputs("tritonswmm")` removes the exchange side-file in every state except flux on
    with the SWMM node store unwritten, where it keeps it.

    Input class: the 2 x 2 of (flux toggle, node store written), driven through the real method on the minimal
    stub the existing clear-raw tests use. Plausible wrong implementations: no guard (the TRITON export runs
    first and deletes the file the SWMM export decodes later); a guard that ignores the store state (the file is
    kept forever, an 808 MB leak per fine-grid scenario); a guard that ignores the toggle (every run keeps it).
    Killed by: removing the guard, widening it to `flux_on` alone, or narrowing it to `not written` alone. Passes
    under a second correct implementation that moves the reclaim out of `_clear_raw_outputs` into the SWMM export.
    """
    from contextlib import nullcontext

    from hhemt.process_simulation import TRITONSWMM_sim_post_processing

    out_dir = tmp_path / "scenario" / "out_tritonswmm"
    (out_dir / "swmm").mkdir(parents=True)
    side = out_dir / "swmm" / "x_exchange_replay.bin"
    side.write_bytes(b"\x00" * 12)
    proc = TRITONSWMM_sim_post_processing.__new__(TRITONSWMM_sim_post_processing)
    proc.scen_paths = SimpleNamespace(
        out_tritonswmm=out_dir, out_triton=None, swmm_full_out_file=None, sim_folder=out_dir.parent
    )
    marker = SimpleNamespace(set=lambda _v: None)
    proc.log = SimpleNamespace(
        raw_TRITON_outputs_cleared=marker, raw_SWMM_outputs_cleared=marker, deferred_writes=nullcontext
    )
    proc._analysis = SimpleNamespace(
        analysis_paths=SimpleNamespace(analysis_dir=out_dir.parent),
        log=SimpleNamespace(multi_allocation_in_progress=SimpleNamespace(get=lambda: None)),
        cfg_analysis=SimpleNamespace(toggle_coupling_exchange_flow=flux_on),
    )
    proc._swmm_node_outputs_processed = lambda model: node_store_written
    proc._clear_raw_outputs("tritonswmm")
    assert side.exists() == (flux_on and not node_store_written)


def test_the_summary_drops_the_flux_series_and_keeps_the_flag_and_the_statistics():
    """Property: reducing a node dataset that holds the flux gives no `coupling_exchange_flow` or generic
    `_max`/`_last` pair of it, keeps `is_coupling_node` and the eight statistics on `(event_iloc, node_id)`
    unchanged, and still reduces an ordinary time-variant variable.

    Input class: the coupled node dataset after `_attach_coupling_outputs`. Plausible wrong implementations: no
    drop (a `coupling_exchange_flow_max` and `_last` pair enter the cross-simulation identity check), or a drop
    that also removes static variables with a node dimension. Must-fire control: `depth` in the same dataset
    gives `depth_max` and `depth_last`.
    Killed by: deleting the `drop_vars` line, or dropping every non-time variable. Passes under a second correct
    implementation that excludes the series inside the loop over time-variant variables.
    """
    from hhemt.process_simulation import summarize_swmm_simulation_results

    ds = _coupled_node_dataset()
    ds["depth"][:] = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    ds["is_coupling_node"] = ("node_id", np.array([1, 0, 1], dtype="int8"))
    ds["coupling_exchange_flow"] = (("date_time", "node_id"), np.ones((3, 3)))
    ds["coupling_exchange_peak_to_network"] = ("node_id", np.array([2.0, np.nan, 10.0]))
    out = summarize_swmm_simulation_results(ds, event_iloc=0)
    assert not [v for v in out.data_vars if v.startswith("coupling_exchange_flow")]
    assert out["is_coupling_node"].dims == ("event_iloc", "node_id")
    assert out["is_coupling_node"].values.tolist() == [[1, 0, 1]]
    assert out["coupling_exchange_peak_to_network"].values[0, [0, 2]].tolist() == [2.0, 10.0]
    assert out["depth_max"].values[0].tolist() == [7.0, 8.0, 9.0] and out["depth_last"].values[0].tolist() == [
        7.0,
        8.0,
        9.0,
    ]


def test_the_coupling_flag_values_attribute_has_the_type_of_the_flag_variable_in_a_netcdf_file(tmp_path):
    """Property: after the CF attributes are applied and the dataset is written as NetCDF, the `flag_values`
    attribute of `is_coupling_node` has the same type as the variable (int8) and the values 0 and 1; the zarr
    round trip keeps the same two values.

    Input class: the flag as `_attach_coupling_outputs` creates it (int8 on `node_id`), annotated under the
    coupled node mode. CF-1.13 section 3.5 requires `flag_values` to be of the same type as the variable it
    describes, and the flag is meant to be CF-conformant and to survive zarr and NetCDF identically.
    Plausible wrong implementation (the one this test was written against): `flag_values` held as the Python list
    `[0, 1]`, which NetCDF stores as int64, so a CF checker rejects the file. Must-fire control: the variable's
    own dtype and the zarr values, which hold under both the list and an int8 array.
    Killed by: a list attribute. Passes under a second correct implementation holding `np.array([0, 1], int8)`,
    or an `np.int8` tuple.
    """
    import netCDF4

    from hhemt.cf_conventions import apply_cf_attributes

    ds = xr.Dataset(
        {"is_coupling_node": ("node_id", np.array([1, 0, 1], dtype="int8"))}, coords={"node_id": ["a", "b", "c"]}
    )
    apply_cf_attributes(ds, "tritonswmm_swmm_node")
    ds.to_netcdf(tmp_path / "flag.nc")
    with netCDF4.Dataset(tmp_path / "flag.nc") as nc:
        variable = nc["is_coupling_node"]
        assert variable.dtype == np.int8
        flag_values = variable.getncattr("flag_values")
        assert np.asarray(flag_values).tolist() == [0, 1]
        assert flag_values.dtype == variable.dtype, (
            f"flag_values is {flag_values.dtype}, the variable is {variable.dtype}"
        )
    ds.to_zarr(tmp_path / "flag.zarr")
    assert list(xr.open_zarr(tmp_path / "flag.zarr")["is_coupling_node"].attrs["flag_values"]) == [0, 1]


def _stock_link_flow_rows() -> list[str]:
    """Link Flow Summary data rows exactly as stock SWMM 5.2.4 printed them for one model holding every link type.

    Produced with ``python -c "from swmm.toolkit import solver; solver.swmm_run('multi.inp', 'multi.rpt',
    'multi.out')"`` in the hhemt environment (pyswmm 2.0.1, report header "EPA STORM WATER MANAGEMENT MODEL -
    VERSION 5.2 (Build 5.2.4)"), CMS units, dynamic wave, copying the report's Link Flow Summary data lines. The
    model had conduits C1, `C1:C2`, a 31-character id and STEEP; an irregular-section channel CH1; a transverse
    weir W1; a side orifice OR1; a pump P1 with a Pump1 curve and an ideal pump P2 (curve `*`); and two outlet
    links OL1, OL2. No TRITON-SWMM solver ran. The ideal pump's row is followed by 26 blanks, as printed.
    """
    return [
        "  C1                   CONDUIT     0.900     0  00:30      1.57    0.53    1.00\n",
        "  C1:C2                CONDUIT     0.900     0  00:30      1.15    0.53    1.00\n",
        "  LONG_LINK_IDENTIFIER_0123456789 CONDUIT     0.001     0  02:00      0.28    0.00    0.02\n",
        "  CH1                  CHANNEL     0.001     0  02:00      0.25    0.00    0.03\n",
        "  STEEP                CONDUIT     0.001     0  02:00      0.53    0.01    0.05\n",
        "  W1                   WEIR        2.332     0  00:10                      1.00\n",
        "  OR1                  ORIFICE     0.400     0  00:16                      1.00\n",
        "  P1                   PUMP        0.050     0  00:05              1.00\n",
        "  P2                   PUMP        0.050     0  00:05" + " " * 26 + "\n",
        "  OL1                  DUMMY       0.004     0  00:05\n",
        "  OL2                  DUMMY       0.001     0  00:05\n",
    ]


def test_every_link_type_stock_swmm_prints_lands_each_cell_in_its_own_column():
    """Property: reading the stock-printed Link Flow Summary of a model with every link type gives, per row, the
    id and type whole and each printed cell in the column its header names, with every cell the type does not
    print empty.

    Input class: one row of each type stock SWMM prints (CONDUIT, CHANNEL, WEIR, ORIFICE, PUMP with a curve,
    PUMP without one, DUMMY for an outlet link), an id containing a colon, an id of 31 characters, and an id of
    five characters, all in one call so any state carried between rows shows. These are bytes the engine printed,
    not rows assembled from its format strings, and they include the trailing blanks of the ideal pump's row.
    Plausible wrong implementations: whitespace tokens (the weir's and orifice's depth ratio lands in the
    velocity column, the pump's ratio in the wrong one), cells measured from the left edge (the 31-character row
    shifts 11 characters), cells measured from the right edge (a pump's ratio lands in the depth column).
    Killed by: any of those. Passes under a second correct implementation that splits each row with one regular
    expression.
    """
    from hhemt.swmm_output_parser import format_link_flow_summary

    headers = ["link_id", "type", "max_flow", "time", "max_velocity", "max_over_full_flow", "max_over_full_depth"]
    frame = format_link_flow_summary(_stock_link_flow_rows(), headers)
    expected = [
        ("C1", "CONDUIT", "0.900", "0 00:30", "1.57", "0.53", "1.00"),
        ("C1:C2", "CONDUIT", "0.900", "0 00:30", "1.15", "0.53", "1.00"),
        ("LONG_LINK_IDENTIFIER_0123456789", "CONDUIT", "0.001", "0 02:00", "0.28", "0.00", "0.02"),
        ("CH1", "CHANNEL", "0.001", "0 02:00", "0.25", "0.00", "0.03"),
        ("STEEP", "CONDUIT", "0.001", "0 02:00", "0.53", "0.01", "0.05"),
        ("W1", "WEIR", "2.332", "0 00:10", "", "", "1.00"),
        ("OR1", "ORIFICE", "0.400", "0 00:16", "", "", "1.00"),
        ("P1", "PUMP", "0.050", "0 00:05", "", "1.00", ""),
        ("P2", "PUMP", "0.050", "0 00:05", "", "", ""),
        ("OL1", "DUMMY", "0.004", "0 00:05", "", "", ""),
        ("OL2", "DUMMY", "0.001", "0 00:05", "", "", ""),
    ]
    assert [tuple(row) for row in frame.itertuples(index=False, name=None)] == expected


def test_preflight_validate_calls_the_coupling_flux_check_with_both_configs_and_the_result():
    """Property: `preflight_validate` contains a call `_validate_coupling_exchange_flow(cfg_system, cfg_analysis,
    result)`, the three names in that order.

    Input class: the one entry point every run passes through. A test that calls the check directly leaves its
    wiring unobserved: a check that is correct and never called refuses nothing, and the run then
    finishes without the series the analysis asked for. Plausible wrong implementations: the call removed, or the
    two configs swapped (the rule then reads `toggle_tritonswmm_model` off the analysis config and raises).
    This is a structural fitness check on the function's syntax tree, stated without reference to the rule.
    Killed by: deleting the call or reordering its arguments. Passes under a second correct implementation that
    moves the call into a helper, only if that helper is the one named here; a helper rename would need this test
    to follow, which is the price of a wiring pin.
    """
    import ast
    import inspect
    import textwrap

    from hhemt.validation import preflight_validate

    tree = ast.parse(textwrap.dedent(inspect.getsource(preflight_validate)))
    calls = [
        [arg.id for arg in node.args if isinstance(arg, ast.Name)]
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_validate_coupling_exchange_flow"
    ]
    assert calls == [["cfg_system", "cfg_analysis", "result"]]
