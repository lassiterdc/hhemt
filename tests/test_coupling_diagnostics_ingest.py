"""Ingestion of the solver's coupling-diagnostic capture files.

Chunk test: it guards the per-scenario stores, the per-node path-3 account on the coupled node store, the
consolidated ``tritonswmm/coupling_diagnostics/*`` groups, the fingerprint and root attributes, and the
``coupling diagnostics captured`` validation check. It is promoted into a module test or deleted at the run's
final landing.

The raw files are written here from the solver's byte layout, with ``struct`` and hand-chosen values that
differ per field and per node. Expected numbers are written down in the tests and never read back from the code
under test. No solver runs: everything starts from the files the solver would have left.
"""

from __future__ import annotations

import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hhemt.cf_conventions import apply_cf_attributes, quantity_provenance
from hhemt.coupling_exchange import FT3_TO_M3_FACTOR
from hhemt.exceptions import ProcessingError
from hhemt.process_simulation import (
    COUPLING_CAPTURES,
    COUPLING_NODE_DIAGNOSTIC_VARIABLES,
    TRITONSWMM_sim_post_processing,
    coupling_capture_store_path,
    expected_coupling_captures,
    expected_coupling_raw_captures,
    summarize_swmm_simulation_results,
)
from hhemt.utils import write_zarr

NODE_MAGIC = 0x44504E33  # "DPN3", item 1 of the solver's file format
EDGE_MAGIC = 0x44454631  # "DEF1"
STEP_MAGIC = 0x44535031  # "DSP1"


def _cfg(*, captures=False, links=None, interval=1, lag=0, out_type="zarr", path3_step=False):
    return SimpleNamespace(
        toggle_coupling_diagnostics=captures,
        swmm_diagnostic_links=links,
        swmm_diagnostic_node_path3_per_step=path3_step,
        swmm_coupling_interval=interval,
        swmm_coupling_lag=lag,
        target_processed_output_type=out_type,
    )


def _node_file(path: Path, names: list[str], records: list[tuple]) -> None:
    """swmm_node_path3.bin: header then per interval {f64 t_end, f64[n] pos, f64[n] neg, i32[n] n_nonconv}."""
    n = len(names)
    body = b"".join(
        struct.pack("<d", t) + struct.pack(f"<{n}d", *pos) + struct.pack(f"<{n}d", *neg) + struct.pack(f"<{n}i", *bad)
        for t, pos, neg, bad in records
    )
    path.write_bytes(struct.pack("<iiii", NODE_MAGIC, 1, 8, n) + body)
    path.with_name("swmm_node_diag.nodes").write_text("\n".join(names) + "\n")


def _edge_file(path: Path, rows: list[tuple]) -> None:
    body = b"".join(struct.pack("<ddddd", *row) for row in rows)
    path.write_bytes(struct.pack("<iiii", EDGE_MAGIC, 1, 8, 4) + body)


def _step_file(path: Path, n_steps: int) -> None:
    body = b"".join(struct.pack("<ddiiddd", 10.0 * (k + 1), 10.0, 3, 0, 1.0, -1.0, 5.0) for k in range(n_steps))
    path.write_bytes(struct.pack("<iiii", STEP_MAGIC, 1, 8, 0) + body)


class _Log:
    """The slice of a model log ingestion touches: the success ledger ``_already_written`` reads."""

    def __init__(self):
        self.processing_log = SimpleNamespace(outputs={})
        self.entries: list[str] = []

    def add_sim_processing_entry(self, f_out, size, elapsed, success, notes=None):
        self.entries.append(Path(f_out).name)
        self.processing_log.outputs[Path(f_out).name] = SimpleNamespace(success=success)


def _proc(tmp_path: Path, cfg) -> TRITONSWMM_sim_post_processing:
    """A post-processing object holding only what ingestion and the node attach read."""
    proc = TRITONSWMM_sim_post_processing.__new__(TRITONSWMM_sim_post_processing)
    processed = tmp_path / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    proc.scen_paths = SimpleNamespace(
        out_tritonswmm=tmp_path / "out_tritonswmm",
        output_tritonswmm_node_time_series=processed / "TRITONSWMM_SWMM_node_tseries.zarr",
        swmm_hydraulics_rpt=tmp_path / "out_tritonswmm" / "swmm" / "hydraulics.rpt",
    )
    (tmp_path / "out_tritonswmm" / "swmm").mkdir(parents=True, exist_ok=True)
    proc.scen_paths.swmm_hydraulics_rpt.write_text("  Starting Date ............ 01/01/2020 00:00:00\n")
    proc._analysis = SimpleNamespace(cfg_analysis=cfg)
    proc.log = _Log()
    proc._written = []

    def write_output(ds, f_out, compression_level, verbose, mode):
        assert mode == "tritonswmm_coupling_diagnostics"
        proc._written.append(Path(f_out).name)
        write_zarr(ds, f_out)

    proc._write_output = write_output
    proc._open = lambda f: xr.open_dataset(f, engine="zarr", chunks=None)
    return proc


def _node_dataset(names=("n1", "outf", "n2"), n_times=4, step_s=120):
    times = pd.to_datetime("2020-01-01") + pd.to_timedelta([step_s * (k + 1) for k in range(n_times)], unit="s")
    return xr.Dataset(
        {"depth": (("date_time", "node_id"), np.zeros((n_times, len(names))))},
        coords={"date_time": times, "node_id": list(names)},
    )


# ---------------------------------------------------------------------------------------------
# which captures an analysis asks for
# ---------------------------------------------------------------------------------------------


def test_the_requested_captures_follow_the_two_fields_and_a_stand_in_config_asks_for_none():
    """Property: the toggle selects the six solver-side captures, the links field selects the link file, both
    select all seven, and a config that sets neither (or a stand-in that predates the fields) selects none.

    Class: the four config states. Plausible wrong implementation: one trigger for all seven (the link file
    written on every toggle, or never written when only the links field is set). Control: the empty state, so the
    non-empty results are not a default. Killed by: swapping the two triggers.
    """
    names = lambda cfg: [c.name for c in expected_coupling_captures(cfg)]  # noqa: E731
    assert names(_cfg()) == []
    assert names(SimpleNamespace()) == []
    assert names(_cfg(captures=True)) == [
        "swmm_step",
        "swmm_node_path3",
        "edge_flux",
        "domain_balance",
        "wet_cells",
        "dt_limiter",
    ]
    assert names(_cfg(links="ALL")) == ["swmm_link_flow"]
    assert len(names(_cfg(captures=True, links="ALL"))) == len(COUPLING_CAPTURES) == 7


def test_a_store_path_sits_in_a_subdirectory_of_processed_so_it_never_blocks_a_scenario_removal(tmp_path):
    """Property: the store is ``processed/coupling_diagnostics/{name}.{type}``, one directory below the folder the
    scenario-removal test scans for stores with a non-recursive glob.

    Plausible wrong implementation: ``processed/{name}.zarr``, which that glob would count as an unconsolidated
    store and refuse to remove the scenario. Control: the node store itself IS directly under ``processed/``.
    """
    paths = SimpleNamespace(output_tritonswmm_node_time_series=tmp_path / "processed" / "node.zarr")
    store = coupling_capture_store_path(paths, "edge_flux", "zarr")
    assert store == tmp_path / "processed" / "coupling_diagnostics" / "edge_flux.zarr"
    assert list((tmp_path / "processed").glob("*.zarr")) == [] and store.parent.name == "coupling_diagnostics"
    with pytest.raises(ProcessingError):
        coupling_capture_store_path(SimpleNamespace(output_tritonswmm_node_time_series=None), "edge_flux", "zarr")


# ---------------------------------------------------------------------------------------------
# per-scenario ingestion
# ---------------------------------------------------------------------------------------------


def test_ingestion_writes_each_requested_capture_once_and_a_second_call_writes_nothing(tmp_path):
    """Property: every requested capture whose file exists is decoded and written to its own store, and a second
    call, with the ledger now recording them, writes none.

    Class: the toggle-on, files-present state (two captures present of the six requested would be refused, so the
    config here asks for exactly the two present, via the links path and a custom list). Plausible wrong
    implementation: ingesting without the ledger gate (a reprocess after the raw file was reclaimed would then
    raise on the missing file). Killed by: dropping the ``_already_written`` gate.
    """
    out = tmp_path / "out_tritonswmm" / "perf"
    out.mkdir(parents=True)
    _edge_file(out / "edge_flux.bin", [(10.0, 1.0, 2.0, 3.0, 4.0), (20.0, 5.0, 6.0, 7.0, 8.0)])
    proc = _proc(tmp_path, _cfg(captures=True))
    # restrict to the two captures present by shadowing the expectation (the files for the others are not written)
    only = tuple(c for c in COUPLING_CAPTURES if c.name == "edge_flux")
    proc._coupling_captures_expected = lambda: only

    proc._export_coupling_diagnostics()
    store = tmp_path / "processed" / "coupling_diagnostics" / "edge_flux.zarr"
    assert proc._written == ["edge_flux.zarr"] and store.exists()
    ds = xr.open_dataset(store, engine="zarr", chunks=None)
    np.testing.assert_array_equal(ds["edge_flux"].values, [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    assert ds["edge"].values.tolist() == ["west", "east", "south", "north"]

    (out / "edge_flux.bin").unlink()  # the raw file reclaimed
    proc._written.clear()
    proc._export_coupling_diagnostics()  # must not raise on the missing raw file and must not rewrite
    assert proc._written == []


def test_a_toggle_that_is_on_with_its_file_absent_is_refused_and_one_that_is_off_is_not(tmp_path):
    """Property: rule 1 of the capture design. With the toggle on and no capture file the ingestion raises naming
    the file; with the toggle off the same tree is accepted; a links request without its file is refused too.

    Must-fire control: the toggle-off arm on the identical (empty) tree, so the refusal is the toggle's doing and
    not a missing directory's. Killed by: swallowing the absence, or refusing regardless of the toggle.
    """
    (tmp_path / "out_tritonswmm").mkdir(parents=True, exist_ok=True)
    on = _proc(tmp_path / "on", _cfg(captures=True))
    with pytest.raises(ProcessingError, match="swmm_step_diag.bin"):
        on._export_coupling_diagnostics()
    links = _proc(tmp_path / "links", _cfg(links="ALL"))
    with pytest.raises(ProcessingError, match="swmm_link_flow_diag.bin"):
        links._export_coupling_diagnostics()
    off = _proc(tmp_path / "off", _cfg())
    off._export_coupling_diagnostics()
    assert off._written == []


def test_two_files_of_one_name_and_a_missing_name_list_are_refused(tmp_path):
    """Property: a name found twice under the output directory is refused (not picked), and a per-node file whose
    name list is absent beside it is refused (its width is unknowable).

    Plausible wrong implementation: ``found[0]``, which ingests a file the run may not have written. Control: the
    single-file arm in the test above ingests normally.
    """
    only_edge = tuple(c for c in COUPLING_CAPTURES if c.name == "edge_flux")
    only_node = tuple(c for c in COUPLING_CAPTURES if c.name == "swmm_node_path3")
    proc = _proc(tmp_path / "dup", _cfg(captures=True))
    for sub in ("a", "b"):
        (tmp_path / "dup" / "out_tritonswmm" / sub).mkdir(parents=True)
        _edge_file(tmp_path / "dup" / "out_tritonswmm" / sub / "edge_flux.bin", [(10.0, 1.0, 2.0, 3.0, 4.0)])
    proc._coupling_captures_expected = lambda: only_edge
    with pytest.raises(ProcessingError, match="2 files"):
        proc._export_coupling_diagnostics()

    proc2 = _proc(tmp_path / "nolist", _cfg(captures=True))
    raw = tmp_path / "nolist" / "out_tritonswmm" / "x" / "swmm_node_path3.bin"
    raw.parent.mkdir(parents=True)
    _node_file(raw, ["n1"], [(120.0, [1.0], [-1.0], [0])])
    (raw.parent / "swmm_node_diag.nodes").unlink()
    proc2._coupling_captures_expected = lambda: only_node
    with pytest.raises(ProcessingError, match="swmm_node_diag.nodes"):
        proc2._export_coupling_diagnostics()


def test_the_per_step_path3_file_is_checked_for_but_never_ingested(tmp_path):
    """Property: with ``swmm_diagnostic_node_path3_per_step`` on, ingestion refuses a run that left no
    ``swmm_node_path3_step.bin`` or no name list beside it, accepts a run that left both, writes no store for it, and
    keeps it out of the ingested capture list; with the field off the same empty tree is accepted.

    Class: NEW CAPABILITY. Plausible wrong implementations: an entry in ``COUPLING_CAPTURES`` (the gigabyte file is
    then copied into a store, consolidated and counted by the validation check), or no presence check (a solver that
    predates the key passes unnoticed). Reachability: the three trees are the three states a run can leave.
    Both-states anchor: the refusal turns on the file's absence, with the off-field arm on the identical empty tree
    as the control. Killed by: either wrong implementation. Second correct implementation: checking in a separate
    method called from the same place; the test calls the public export.
    """
    assert tuple(c.name for c in expected_coupling_captures(_cfg(path3_step=True))) == ()
    assert tuple(c.raw_file for c in expected_coupling_raw_captures(_cfg(path3_step=True))) == (
        "swmm_node_path3_step.bin",
    )
    assert expected_coupling_raw_captures(_cfg()) == ()
    assert expected_coupling_raw_captures(SimpleNamespace()) == ()

    absent = _proc(tmp_path / "absent", _cfg(path3_step=True))
    with pytest.raises(ProcessingError, match="swmm_node_path3_step.bin"):
        absent._export_coupling_diagnostics()
    off = _proc(tmp_path / "off", _cfg())
    off._export_coupling_diagnostics()

    nolist = _proc(tmp_path / "nolist", _cfg(path3_step=True))
    raw_dir = tmp_path / "nolist" / "out_tritonswmm" / "x"
    raw_dir.mkdir(parents=True)
    (raw_dir / "swmm_node_path3_step.bin").write_bytes(struct.pack("<iiii", 0x44505333, 1, 8, 1))
    with pytest.raises(ProcessingError, match="swmm_node_diag.nodes"):
        nolist._export_coupling_diagnostics()

    ok = _proc(tmp_path / "ok", _cfg(path3_step=True))
    ok_dir = tmp_path / "ok" / "out_tritonswmm" / "x"
    ok_dir.mkdir(parents=True)
    (ok_dir / "swmm_node_path3_step.bin").write_bytes(
        struct.pack("<iiii", 0x44505333, 1, 8, 1) + struct.pack("<ddd", 10.0, 10.0, 2.5)
    )
    (ok_dir / "swmm_node_diag.nodes").write_text("n1\n")
    ok._export_coupling_diagnostics()
    assert ok._written == []


# ---------------------------------------------------------------------------------------------
# the per-node path-3 account on the coupled node store
# ---------------------------------------------------------------------------------------------


def _ingest_node_store(tmp_path: Path, records, names=("n2", "n1")):
    proc = _proc(tmp_path, _cfg(captures=True))
    raw = tmp_path / "out_tritonswmm" / "diag" / "swmm_node_path3.bin"
    raw.parent.mkdir(parents=True)
    _node_file(raw, list(names), records)
    proc._coupling_captures_expected = lambda: tuple(c for c in COUPLING_CAPTURES if c.name == "swmm_node_path3")
    proc._export_coupling_diagnostics()
    return proc


def test_each_print_interval_lands_on_its_report_row_and_the_outfall_stays_nan(tmp_path):
    """Property: with the report step equal to the print interval each record lands on the row whose time is its
    end, node by node (the file lists n2 then n1, the dataset n1, outf, n2), volumes arrive in m3, and the outfall
    the solver does not record is NaN in all three variables.

    Hand arithmetic: n2 records (pos, neg, bad) = (10, -1, 2) at 120 s and (20, -2, 0) at 240 s; n1 = (30, -3, 1)
    and (40, -4, 5). ft3 to m3 by FT3_TO_M3_FACTOR. The rows at 360 and 480 s have no record and stay NaN.
    Plausible wrong implementations: node order taken from the file instead of by name (n1 and n2 swapped);
    filling the outfall with 0; a missing record read as a measured zero. Killed by each of those.
    """
    records = [
        (120.0, [10.0, 30.0], [-1.0, -3.0], [2, 1]),
        (240.0, [20.0, 40.0], [-2.0, -4.0], [0, 5]),
    ]
    proc = _ingest_node_store(tmp_path, records)
    out = proc._attach_coupling_node_diagnostics(_node_dataset())
    f = FT3_TO_M3_FACTOR
    pos = out["path3_volume_pos"]
    assert pos.dims == ("date_time", "node_id")
    np.testing.assert_allclose(pos.sel(node_id="n2").values[:2], np.array([10.0, 20.0]) * f, rtol=1e-12)
    np.testing.assert_allclose(pos.sel(node_id="n1").values[:2], np.array([30.0, 40.0]) * f, rtol=1e-12)
    np.testing.assert_allclose(out["path3_volume_neg"].sel(node_id="n1").values[:2], np.array([-3.0, -4.0]) * f)
    assert out["nonconverged_steps"].sel(node_id="n2").values[:2].tolist() == [2.0, 0.0]
    assert out["nonconverged_steps"].sel(node_id="n1").values[:2].tolist() == [1.0, 5.0]
    for name in COUPLING_NODE_DIAGNOSTIC_VARIABLES:
        assert np.isnan(out[name].sel(node_id="outf").values).all(), name
        assert np.isnan(out[name].values[2:, :]).all(), name  # no record ends in those intervals


def test_records_finer_than_the_report_step_are_summed_into_the_report_interval(tmp_path):
    """Property: with a 240 s report step over 120 s print intervals each report row is the SUM of the records
    ending inside it: n1 pos 1 + 2 = 3 then 4 + 8 = 12, nonconverged 1 + 0 and 2 + 3.

    Class: the H7 case (report step independent of the print interval). Plausible wrong implementation: taking
    the last record, or the first, of each report interval. Control: the equal-step test above, where each row is
    a single record. Killed by `=` in place of the accumulation.
    """
    records = [
        (120.0, [1.0], [-1.0], [1]),
        (240.0, [2.0], [-2.0], [0]),
        (360.0, [4.0], [-4.0], [2]),
        (480.0, [8.0], [-8.0], [3]),
    ]
    proc = _ingest_node_store(tmp_path, records, names=("n1",))
    out = proc._attach_coupling_node_diagnostics(_node_dataset(names=("n1", "n2"), n_times=2, step_s=240))
    f = FT3_TO_M3_FACTOR
    np.testing.assert_allclose(out["path3_volume_pos"].sel(node_id="n1").values, np.array([3.0, 12.0]) * f)
    assert out["nonconverged_steps"].sel(node_id="n1").values.tolist() == [1.0, 5.0]
    assert np.isnan(out["path3_volume_pos"].sel(node_id="n2").values).all()


def test_a_record_after_the_last_report_time_and_an_unknown_node_are_refused(tmp_path):
    """Property: a record ending after the last report time, and a recorded node the SWMM output lacks, raise.

    Plausible wrong implementation: ``searchsorted`` index clipped to the last row, which silently folds a stray
    record into it. Control: the passing arms above. Killed by clipping.
    """
    proc = _ingest_node_store(tmp_path / "late", [(120.0, [1.0], [-1.0], [0]), (999.0, [1.0], [-1.0], [0])], ("n1",))
    with pytest.raises(ProcessingError, match="end after the last report time"):
        proc._attach_coupling_node_diagnostics(_node_dataset(names=("n1",), n_times=2))
    proc2 = _ingest_node_store(tmp_path / "unk", [(120.0, [1.0], [-1.0], [0])], ("ghost",))
    with pytest.raises(ProcessingError, match="ghost"):
        proc2._attach_coupling_node_diagnostics(_node_dataset(names=("n1",), n_times=2))


def test_the_node_attach_does_nothing_when_the_per_node_capture_is_not_requested(tmp_path):
    """Property: with the toggle off the dataset comes back unchanged and no store is read (none exists).

    Must-fire control: the same dataset with the toggle on and no store raises, so the quiet return is the
    toggle's doing.
    """
    ds = _node_dataset()
    off = _proc(tmp_path / "off", _cfg())
    assert sorted(off._attach_coupling_node_diagnostics(ds).data_vars) == ["depth"]
    on = _proc(tmp_path / "on", _cfg(captures=True))
    with pytest.raises(ProcessingError, match="swmm_node_path3 store is absent"):
        on._attach_coupling_node_diagnostics(_node_dataset())


def test_the_summary_drops_the_three_per_interval_series():
    """Property: the node summary carries no ``_max`` or ``_last`` of the per-interval path-3 series, and still
    summarises an ordinary series beside them (control).
    """
    ds = _node_dataset(names=("n1", "n2"), n_times=3)
    for name in COUPLING_NODE_DIAGNOSTIC_VARIABLES:
        ds[name] = (("date_time", "node_id"), np.ones((3, 2)))
    out = summarize_swmm_simulation_results(ds, 0)
    assert "depth_max" in out
    assert not [v for v in out.data_vars if v.startswith(("path3_", "nonconverged_"))]


def test_the_three_node_variables_carry_units_and_a_provenance_descriptor():
    """Property: the three names are stamped with their units from the map and each has a descriptor.

    Control: an unlisted name beside them gets no unit. Killed by deleting a map entry.
    """
    ds = xr.Dataset({n: (("node_id",), np.zeros(2)) for n in (*COUPLING_NODE_DIAGNOSTIC_VARIABLES, "path3_unlisted")})
    apply_cf_attributes(ds, "tritonswmm_swmm_node")
    assert ds["path3_volume_pos"].attrs["units"] == "m3" == ds["path3_volume_neg"].attrs["units"]
    assert ds["nonconverged_steps"].attrs["units"] == "1"
    assert "units" not in ds["path3_unlisted"].attrs
    for name in COUPLING_NODE_DIAGNOSTIC_VARIABLES:
        assert quantity_provenance(name) is not None, name


# ---------------------------------------------------------------------------------------------
# consolidation: groups, fingerprint, root attributes
# ---------------------------------------------------------------------------------------------


def _consolidator(tmp_path: Path, cfg, event_lengths: dict[int, int], monkeypatch):
    """A post-processing-analysis object over scenarios whose edge-flux stores hold the given record counts."""
    from hhemt import processing_analysis as pa

    scen_dirs = {}
    for ei, n in event_lengths.items():
        node_store = tmp_path / f"sim{ei}" / "processed" / "node.zarr"
        scen_dirs[ei] = SimpleNamespace(output_tritonswmm_node_time_series=node_store)
        if n:
            store = coupling_capture_store_path(scen_dirs[ei], "edge_flux", "zarr")
            store.parent.mkdir(parents=True)
            ds = xr.Dataset(
                {"edge_flux": (("step", "edge"), np.full((n, 4), float(ei + 1)))},
                coords={"step": np.arange(n), "edge": ["west", "east", "south", "north"]},
            )
            write_zarr(ds, store)
    monkeypatch.setattr(pa, "TRITONSWMM_scenario", lambda ei, analysis: SimpleNamespace(scen_paths=scen_dirs[ei]))
    analysis = SimpleNamespace(
        cfg_analysis=cfg,
        df_sims=pd.DataFrame(index=list(event_lengths)),
        _get_enabled_model_types=lambda: ["tritonswmm"],
        _retrieve_weather_indexer_using_integer_index=lambda ei: {"event_id": ei},
    )
    post = pa.TRITONSWMM_analysis_post_processing.__new__(pa.TRITONSWMM_analysis_post_processing)
    post._analysis = analysis
    return post


def test_a_group_per_capture_holds_every_scenario_on_event_iloc_and_pads_the_shorter_one(tmp_path, monkeypatch):
    """Property: the consolidated group ``tritonswmm/coupling_diagnostics/edge_flux`` carries both scenarios on
    ``event_iloc``, each at its own values, and the scenario with fewer steps is NaN-padded beyond them.

    Hand values: event 0 has 3 steps of 1.0, event 1 has 5 steps of 2.0. Plausible wrong implementation: an inner
    join (the 5-step scenario truncated to 3), or the scenarios stacked on the step axis. Control: the two
    unpadded rows are exactly the stored values, so NaN is padding and not a lost value.
    """
    from hhemt.utils import deferred_consolidation

    post = _consolidator(tmp_path, _cfg(captures=True), {0: 3, 1: 5}, monkeypatch)
    post._coupling_captures = lambda: tuple(c for c in COUPLING_CAPTURES if c.name == "edge_flux")
    store = tmp_path / "tree.zarr"
    with deferred_consolidation(store):
        written = post._consolidate_coupling_diagnostics(store)
    assert written == ["tritonswmm/coupling_diagnostics/edge_flux"]
    ds = xr.open_zarr(store, group=written[0], consolidated=True, chunks=None)
    assert ds["edge_flux"].dims == ("event_iloc", "step", "edge")
    assert ds.sizes["step"] == 5 and ds["event_iloc"].values.tolist() == [0, 1]
    np.testing.assert_array_equal(ds["edge_flux"].sel(event_iloc=1).values, np.full((5, 4), 2.0))
    np.testing.assert_array_equal(ds["edge_flux"].sel(event_iloc=0).values[:3], np.full((3, 4), 1.0))
    assert np.isnan(ds["edge_flux"].sel(event_iloc=0).values[3:]).all()
    assert ds["event_id"].values.tolist() == ["event_id.0", "event_id.1"]


def test_a_scenario_store_that_is_gone_keeps_an_existing_group_and_refuses_when_there_is_none(tmp_path, monkeypatch):
    """Property: a scenario whose store was removed leaves an existing group untouched (it was written while the
    store existed); with no group to fall back on the consolidation refuses with FileNotFoundError.

    Plausible wrong implementation: rewriting the group from the surviving scenarios, which drops the removed
    scenario's slice, the only copy. Killed by that. Control: the refusing arm.
    """
    from hhemt.utils import deferred_consolidation

    post = _consolidator(tmp_path, _cfg(captures=True), {0: 3, 1: 2}, monkeypatch)
    post._coupling_captures = lambda: tuple(c for c in COUPLING_CAPTURES if c.name == "edge_flux")
    store = tmp_path / "tree.zarr"
    with deferred_consolidation(store):
        post._consolidate_coupling_diagnostics(store)
    # scenario 1's store removed
    import shutil

    shutil.rmtree(tmp_path / "sim1" / "processed" / "coupling_diagnostics")
    with deferred_consolidation(store):
        assert post._consolidate_coupling_diagnostics(store) == []
    ds = xr.open_zarr(store, group="tritonswmm/coupling_diagnostics/edge_flux", consolidated=True, chunks=None)
    assert ds["event_iloc"].values.tolist() == [0, 1]  # the removed scenario's slice survived

    fresh = tmp_path / "fresh.zarr"
    with pytest.raises(FileNotFoundError, match="edge_flux"):
        with deferred_consolidation(fresh):
            post._consolidate_coupling_diagnostics(fresh)


def test_the_fingerprint_changes_only_for_a_departure_from_the_defaults(tmp_path, monkeypatch):
    """Property: an analysis that sets none of the fields contributes nothing (so its stamped tree is not rebuilt),
    and each of the toggle, the interval and the lag contributes its own entry.

    Control: the default arm is the empty dict, so the other three being non-empty is the field's doing.
    """

    def fingerprint(arm, cfg):
        post = _consolidator(tmp_path / arm, cfg, {0: 1}, monkeypatch)
        return post._coupling_diagnostics_fingerprint()

    assert fingerprint("default", _cfg()) == {}
    assert fingerprint("toggle", _cfg(captures=True))["captures"][0] == "swmm_step"
    assert fingerprint("interval", _cfg(interval=4)) == {"swmm_coupling_interval": 4}
    assert fingerprint("lag", _cfg(lag=1)) == {"swmm_coupling_lag": 1}


# ---------------------------------------------------------------------------------------------
# the validation check
# ---------------------------------------------------------------------------------------------


def _fake_analysis(tmp_path: Path, cfg, n_events=1):
    sims = tmp_path / "sims"
    return SimpleNamespace(
        cfg_analysis=cfg,
        analysis_paths=SimpleNamespace(
            simulation_directory=sims, analysis_datatree_zarr=tmp_path / "analysis_datatree.zarr"
        ),
        df_sims=pd.DataFrame(index=list(range(n_events))),
        _get_enabled_model_types=lambda: ["tritonswmm"],
        _retrieve_weather_indexer_using_integer_index=lambda i: {"event_id": i},
    )


def _store(tmp_path: Path, name: str, n_records: int, event_id="event_id.0") -> Path:
    from hhemt.scenario import compute_event_id_slug

    scen = tmp_path / "sims" / compute_event_id_slug({"event_id": 0})
    path = scen / "processed" / "coupling_diagnostics" / f"{name}.zarr"
    path.parent.mkdir(parents=True, exist_ok=True)
    record_dim = next(c.record_dim for c in COUPLING_CAPTURES if c.name == name)
    write_zarr(xr.Dataset({"v": ((record_dim,), np.zeros(n_records))}), path)
    return path


_ONLY_EDGE_CFG = _cfg(captures=True)


def _edge_only(monkeypatch):
    import hhemt.process_simulation as ps

    only = tuple(c for c in COUPLING_CAPTURES if c.name == "edge_flux")
    monkeypatch.setattr(ps, "expected_coupling_captures", lambda cfg: only if cfg.toggle_coupling_diagnostics else ())


def test_the_check_passes_for_a_present_non_empty_store_and_fails_for_empty_missing_or_unreadable(
    tmp_path, monkeypatch
):
    """Property: present and non-empty passes; an empty record dimension, a missing store under a live scenario
    directory, and an unreadable store each fail with a row that names the capture.

    Must-fire control: the passing arm on the same fixture with a non-empty store, so each failure is the store's
    state and not the fixture's. Killed by a presence-only check (the empty arm) or a size-only check (the missing
    arm).
    """
    from hhemt.analysis_validation import check_coupling_diagnostics

    _edge_only(monkeypatch)
    ok = tmp_path / "ok"
    _store(ok, "edge_flux", 3)
    res = check_coupling_diagnostics(_fake_analysis(ok, _ONLY_EDGE_CFG))
    assert res.passed and res.applicable and res.details == []

    empty = tmp_path / "empty"
    _store(empty, "edge_flux", 0)
    res = check_coupling_diagnostics(_fake_analysis(empty, _ONLY_EDGE_CFG))
    assert not res.passed and "holds no record on step" in res.details[0]["detail"]

    missing = tmp_path / "missing"
    from hhemt.scenario import compute_event_id_slug

    (missing / "sims" / compute_event_id_slug({"event_id": 0})).mkdir(parents=True)
    res = check_coupling_diagnostics(_fake_analysis(missing, _ONLY_EDGE_CFG))
    assert not res.passed and "no store at" in res.details[0]["detail"]

    broken = tmp_path / "broken"
    path = _store(broken, "edge_flux", 3)
    for child in path.iterdir():
        if child.is_file():
            child.write_bytes(b"{not json")
    res = check_coupling_diagnostics(_fake_analysis(broken, _ONLY_EDGE_CFG))
    assert not res.passed and "cannot be opened" in res.details[0]["detail"]


def test_a_removed_scenario_is_accounted_for_by_the_consolidated_tree_and_not_by_nothing(tmp_path, monkeypatch):
    """Property: with the scenario directory gone, the capture is accounted for when the consolidated tree holds
    its group and is a failure when it does not; a tree that exists without the group fails even while the
    per-scenario store is present.

    Controls: the two arms differ only in the group directory's existence.
    """
    from hhemt.analysis_validation import check_coupling_diagnostics

    _edge_only(monkeypatch)
    carried = tmp_path / "carried"
    (carried / "sims").mkdir(parents=True)
    (carried / "analysis_datatree.zarr" / "tritonswmm" / "coupling_diagnostics" / "edge_flux").mkdir(parents=True)
    assert check_coupling_diagnostics(_fake_analysis(carried, _ONLY_EDGE_CFG)).passed

    bare = tmp_path / "bare"
    (bare / "sims").mkdir(parents=True)
    assert not check_coupling_diagnostics(_fake_analysis(bare, _ONLY_EDGE_CFG)).passed

    stale_tree = tmp_path / "stale"
    _store(stale_tree, "edge_flux", 2)
    (stale_tree / "analysis_datatree.zarr").mkdir()
    res = check_coupling_diagnostics(_fake_analysis(stale_tree, _ONLY_EDGE_CFG))
    assert not res.passed and "no group" in res.details[0]["detail"]


def test_the_check_is_not_applicable_without_a_requested_capture_and_runs_in_the_report(tmp_path):
    """Property: an analysis that requests no capture is N/A and passes; the check is registered in
    ``validate_analysis`` (its row name appears in the module's check list source).

    Control: the requesting arm above is applicable. The registration is asserted on the source text because a
    full ``validate_analysis`` needs a live analysis.
    """
    import inspect

    from hhemt import analysis_validation as av

    res = av.check_coupling_diagnostics(_fake_analysis(tmp_path, _cfg()))
    assert res.applicable is False and res.passed is True
    assert "check_coupling_diagnostics(analysis)" in inspect.getsource(av.validate_analysis)


# ---------------------------------------------------------------------------------------------
# the keep-field selects the per-node path-3 variables like the exchange flow (decision F2, form A)
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", COUPLING_NODE_DIAGNOSTIC_VARIABLES)
def test_a_selection_naming_a_path3_variable_is_accepted_for_the_coupled_node_store_only(name):
    """Property: each of the three names is accepted in a coupled node keep-list and returned as written; an
    unknown name is still refused naming the domain; the standalone SWMM family, which never writes them, selects
    none of them.

    Class: the three names of the T1b account. Plausible wrong implementation: widening the domain of both node
    families (the standalone store has no such variable). Must-fire controls: the unknown name raises, and the
    standalone arm raises, so acceptance is the coupled domain's doing. Killed by removing a name from the domain.
    """
    from hhemt.timeseries_selection import resolve_keep

    assert resolve_keep("swmm_node", "tritonswmm", [name]) == (name,)
    assert name in resolve_keep("swmm_node", "tritonswmm", "all")
    with pytest.raises(ValueError, match="outside the swmm_node domain"):
        resolve_keep("swmm_node", "tritonswmm", ["path3_unknown"])
    assert resolve_keep("swmm_node", "swmm_only", [name]) == ()  # the standalone family keeps only names it writes


def test_a_keep_list_prunes_the_path3_variables_it_omits_and_keeps_the_ones_it_names():
    """Property: with a coupled store holding depth and the three variables, a list naming ``depth`` and one of
    them prunes the other two and keeps the named one, exactly as for ``coupling_exchange_flow``.

    Control: the exchange flow beside them is pruned by the same list, so the behaviour is the table's.
    """
    from hhemt.timeseries_selection import _to_prune, resolve_keep

    ds = _node_dataset(names=("n1",), n_times=2)
    for name in (*COUPLING_NODE_DIAGNOSTIC_VARIABLES, "coupling_exchange_flow"):
        ds[name] = (("date_time", "node_id"), np.ones((2, 1)))
    keep = resolve_keep("swmm_node", "tritonswmm", ["depth", "path3_volume_pos"])
    pruned = set(_to_prune(ds, "swmm_node", "tritonswmm", keep))
    assert pruned == {"path3_volume_neg", "nonconverged_steps", "coupling_exchange_flow"}
