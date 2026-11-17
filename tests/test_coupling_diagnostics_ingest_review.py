"""Reviewer tests for the coupling-diagnostics ingestion.

Chunk test: it guards one class of input the chunk's own tests do not reach, namely scenarios of unequal record
count across EVERY capture the ingestion table lists. It is a one-time test, to be promoted into a module test or
deleted at the run's final landing.

The raw files are written here from the byte layouts the solver's file format fixes, with hand-chosen values
that differ per event and per record, and are decoded by each capture's own reader before they are consolidated.
No solver runs.
"""

from __future__ import annotations

import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hhemt import coupling_diagnostics as decoders
from hhemt import processing_analysis as pa
from hhemt.cf_conventions import apply_cf_attributes
from hhemt.process_simulation import COUPLING_CAPTURES, coupling_capture_store_path
from hhemt.utils import deferred_consolidation, write_zarr

_HEADER = struct.Struct("<iiii")
_MAGICS = {
    "swmm_step": 0x44535031,
    "swmm_node_path3": 0x44504E33,
    "edge_flux": 0x44454631,
    "dt_limiter": 0x44444C31,
    "swmm_link_flow": 0x444C4631,
}
_BALANCE_HEADER = (
    "%t_end, d_storage_m3, runoff_m3, inflow_m3, extbc_m3, clamp_m3, swmm_exchange_m3, "
    "edge_west_m3, edge_east_m3, edge_south_m3, edge_north_m3, residual_m3"
)
_NODES = ("n1", "n2", "n3")
_LINKS = ("l1", "l2")


def _value(event: int, record: int, item: int = 0) -> float:
    """A hand value that differs per event, record and item, so a slice mixed up with another is visible."""
    return 1000.0 * (event + 1) + 10.0 * (record + 1) + item


def _write_raw(capture, directory: Path, event: int, n_records: int) -> Path:
    """Write the capture's raw file for ``event`` with ``n_records`` records, in the solver's byte layout."""
    path = directory / capture.raw_file
    name = capture.name
    if name == "swmm_step":
        # the count of non-converged nodes per step differs with the event, so the flat node list does too
        body = b""
        for k in range(n_records):
            bad = [100 * event + 10 * k + j for j in range(event + 1 + k % 2)]
            body += struct.pack("<ddiiddd", _value(event, k), 10.0, 3, len(bad), 1.0, -1.0, 5.0)
            body += struct.pack(f"<{len(bad)}i", *bad)
        path.write_bytes(_HEADER.pack(_MAGICS[name], 1, 8, 0) + body)
    elif name == "swmm_node_path3":
        n = len(_NODES)
        body = b""
        for k in range(n_records):
            body += struct.pack("<d", _value(event, k))
            body += struct.pack(f"<{n}d", *[_value(event, k, j) for j in range(n)])
            body += struct.pack(f"<{n}d", *[-_value(event, k, j) for j in range(n)])
            body += struct.pack(f"<{n}i", *[event + k + j for j in range(n)])
        path.write_bytes(_HEADER.pack(_MAGICS[name], 1, 8, n) + body)
        (directory / capture.sidecar).write_text("\n".join(_NODES) + "\n")
    elif name == "edge_flux":
        body = b"".join(
            struct.pack("<ddddd", _value(event, k), *[_value(event, k, j) for j in range(4)]) for k in range(n_records)
        )
        path.write_bytes(_HEADER.pack(_MAGICS[name], 1, 8, 4) + body)
    elif name == "dt_limiter":
        body = b"".join(
            struct.pack("<ddiiiidd", _value(event, k), 0.5, 0, 3 + k, 4 + event, 0, 1.5 + k, 2.5 + event)
            for k in range(n_records)
        )
        path.write_bytes(_HEADER.pack(_MAGICS[name], 1, 8, 0) + body)
    elif name == "swmm_link_flow":
        n = len(_LINKS)
        body = b"".join(
            struct.pack("<d", _value(event, k)) + struct.pack(f"<{n}d", *[_value(event, k, j) for j in range(n)])
            for k in range(n_records)
        )
        path.write_bytes(_HEADER.pack(_MAGICS[name], 1, 8, n) + body)
        (directory / capture.sidecar).write_text("\n".join(_LINKS) + "\n")
    elif name == "domain_balance":
        rows = [
            ", ".join(str(_value(event, k, j)) for j in range(12)) for k in range(n_records)
        ]  # column 0 is the interval end
        path.write_text(_BALANCE_HEADER + "\n" + "\n".join(rows) + "\n")
    elif name == "wet_cells":
        ranks = (0, 1)
        rows = [
            f"{k}, {_value(event, k)}, {r}, {_value(event, k, r)}, {_value(event, k, r) + 0.5}"
            for k in range(n_records)
            for r in ranks
        ]
        path.write_text("%print_id, simtime, rank, wet_mean, wet_at_print\n" + "\n".join(rows) + "\n")
    else:  # a capture the table gains without this writer is a finding, not a silent skip
        raise AssertionError(f"no byte writer for capture {name}; add one to this test")
    return path


def _decode(capture, tmp_path: Path, event: int, n_records: int) -> xr.Dataset:
    raw_dir = tmp_path / f"raw{event}"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw = _write_raw(capture, raw_dir, event, n_records)
    read = getattr(decoders, capture.reader)
    return read(raw) if capture.sidecar is None else read(raw, raw_dir / capture.sidecar)


def _consolidator(tmp_path: Path, capture, per_event: dict[int, xr.Dataset], monkeypatch):
    """A post-processing-analysis object over scenarios whose per-capture stores hold the given datasets."""
    scen = {}
    for ei, ds in per_event.items():
        scen[ei] = SimpleNamespace(output_tritonswmm_node_time_series=tmp_path / f"sim{ei}" / "processed" / "node.zarr")
        store = coupling_capture_store_path(scen[ei], capture.name, "zarr")
        store.parent.mkdir(parents=True)
        write_zarr(ds, store)
    monkeypatch.setattr(pa, "TRITONSWMM_scenario", lambda ei, analysis: SimpleNamespace(scen_paths=scen[ei]))
    analysis = SimpleNamespace(
        cfg_analysis=SimpleNamespace(target_processed_output_type="zarr"),
        df_sims=pd.DataFrame(index=list(per_event)),
        _get_enabled_model_types=lambda: ["tritonswmm"],
        _retrieve_weather_indexer_using_integer_index=lambda ei: {"event_id": ei},
    )
    post = pa.TRITONSWMM_analysis_post_processing.__new__(pa.TRITONSWMM_analysis_post_processing)
    post._analysis = analysis
    post._coupling_captures = lambda: (capture,)
    return post


@pytest.mark.parametrize("capture", COUPLING_CAPTURES, ids=lambda c: c.name)
def test_every_capture_consolidates_scenarios_of_unequal_record_count_and_keeps_each_scenarios_values(
    capture, tmp_path, monkeypatch
):
    """Property: for EVERY capture the ingestion table lists, two scenarios whose files hold a different number of
    records consolidate into one group, and each scenario's own values are still there, unmixed.

    Class: the capture table ``COUPLING_CAPTURES`` (7 captures), each given two scenarios of 2 and 3 records (the
    solver's files are ragged across events by nature, because a longer event writes more steps), and for
    ``swmm_step`` a different count of non-converged nodes per step, which makes the flat node list ragged too.
    Reachability: a real analysis has events of different duration, and a run in which every step converges is the
    only one in which the flat node list is equal in length across events; artifact: ``read_swmm_step_diag``
    returns ``nonconverged_node_index`` on the dimension ``nonconverged_entry``, which has no coordinate.

    Why it is required: the consolidation is the only route from the per-scenario stores to the analysis tree, so a
    capture that refuses two ordinary events makes the whole consolidation of an analysis that asks for it fail.

    What kills it: the outer join of ``_consolidate_coupling_diagnostics`` over a dimension that has no coordinate
    (xarray refuses to align unindexed dimensions of different size).

    A second correct implementation, differing in what the test asserts on: store the flat node list as a
    per-event offset into one concatenated array, or give ``nonconverged_entry`` an integer coordinate before the
    join. The assertions below read each variable's values back per event with the padding dropped, so both pass.
    """
    per_event = {0: _decode(capture, tmp_path, 0, 2), 1: _decode(capture, tmp_path, 1, 3)}
    post = _consolidator(tmp_path, capture, per_event, monkeypatch)
    store = tmp_path / "tree.zarr"
    with deferred_consolidation(store):
        written = post._consolidate_coupling_diagnostics(store)
    assert written == [f"tritonswmm/coupling_diagnostics/{capture.name}"]
    group = xr.open_zarr(store, group=written[0], consolidated=True, chunks=None)
    assert group["event_iloc"].values.tolist() == [0, 1]
    for event, original in per_event.items():
        for name in original.data_vars:
            stored = np.asarray(group[name].sel(event_iloc=event).values, dtype="float64").ravel()
            expected = np.asarray(original[name].values, dtype="float64").ravel()
            np.testing.assert_array_equal(stored[~np.isnan(stored)], expected, err_msg=f"{capture.name}:{name}")


def test_the_flat_node_list_of_the_step_capture_keeps_each_scenarios_own_entries(tmp_path, monkeypatch):
    """Property: the non-converged node indices of two scenarios with different totals are each scenario's own, as
    written down by hand: event 0 steps hold [0],[10,11] and event 1 steps hold [100,101],[110,111,112],[120,121].

    Class: one named input of the ragged class above, with hand-written expected values so the assertion does not
    read back from the code that wrote them. Why required: a consolidation that returns for a scenario another
    scenario's node list reports the wrong nodes as unstable. What kills it: padding the flat list at the front, or
    stacking the two lists on one axis. A second correct implementation: one concatenated array and an offset per
    event; the test reads the entries through ``swmm_step_nonconverged_nodes`` counts, which both layouts carry.
    """
    capture = next(c for c in COUPLING_CAPTURES if c.name == "swmm_step")
    per_event = {0: _decode(capture, tmp_path, 0, 2), 1: _decode(capture, tmp_path, 1, 3)}
    post = _consolidator(tmp_path, capture, per_event, monkeypatch)
    store = tmp_path / "tree.zarr"
    with deferred_consolidation(store):
        post._consolidate_coupling_diagnostics(store)
    group = xr.open_zarr(store, group="tritonswmm/coupling_diagnostics/swmm_step", consolidated=True, chunks=None)
    for event, counts, nodes in ((0, [1, 2], [0, 10, 11]), (1, [2, 3, 2], [100, 101, 110, 111, 112, 120, 121])):
        stored_counts = np.asarray(group["swmm_step_nonconverged_nodes"].sel(event_iloc=event).values, dtype="float64")
        assert stored_counts[~np.isnan(stored_counts)].astype(int).tolist() == counts
        entries = np.asarray(group["nonconverged_node_index"].sel(event_iloc=event).values, dtype="float64")
        assert entries[~np.isnan(entries)].astype(int).tolist() == nodes


@pytest.mark.parametrize("capture", COUPLING_CAPTURES, ids=lambda c: c.name)
def test_the_cf_stamp_of_the_write_path_keeps_the_units_every_capture_reader_declared(capture, tmp_path):
    """Property: after the write path's CF stamp, every variable of every capture dataset carries the same ``units``
    its reader declared.

    Class: every data variable of every capture in ``COUPLING_CAPTURES`` (7 datasets). The stamp is
    ``apply_cf_attributes(ds, mode)`` that ``_write_output`` runs, and a variable name that is also a key of the CF
    table is overwritten by that entry. Why required: the units are what a reader of the consolidated tree uses to
    convert, and a name collision changes them without an error. What kills it: a reader variable named like an
    existing CF entry with other units. A second correct implementation: any set of reader names that the CF table
    does not reuse, or entries that agree with the readers.
    """
    ds = _decode(capture, tmp_path, 0, 2)
    declared = {n: v.attrs.get("units") for n, v in ds.data_vars.items()}
    stamped = apply_cf_attributes(ds.copy(deep=True), "tritonswmm_coupling_diagnostics")
    for name, units in declared.items():
        if units is not None:
            assert stamped[name].attrs.get("units") == units, f"{capture.name}:{name}"
