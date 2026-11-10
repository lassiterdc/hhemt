"""Reviewer probes for the SWMM hydrology groups of the consolidated tree.

Written by the reviewing seat, separate from the coder's ``tests/test_hydrology_groups.py``. No solver: every test runs
on the committed multi-sim golden (three events, a 10 s hydrology report step, a 120 s window, one DEM cell per node).
Each probe targets a path the coder's file does not exercise: the full ``consolidate_to_datatree`` wiring and the
version-3 upgrade path, idempotence, the group-sourced hydrograph rebuild of an event SHORTER than the axis, a
gridcell holding two wet nodes, a node outside the fixed axis, a report capture that arrives after the inflow capture
was cleared, and the per-node summaries of a NaN-padded event.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tests.fixtures.golden_trees import construct_golden_analysis

_FAMILY = "tritonswmm"
_TS = f"{_FAMILY}/swmm_node_hydro/timeseries"
_NODE = f"{_FAMILY}/swmm_node_hydro/summary"
_SUB = f"{_FAMILY}/swmm_subcatchment_hydro/summary"
_FIELD = "consolidated_hydrology_cleared"
_STEP = 10
_COORDS = {
    "J1": (5.0, 5.0),
    "J2": (15.0, 5.0),
    "J3": (25.0, 5.0),
    "J4": (35.0, 5.0),
    "J5": (5.0, 15.0),
    "J6": (15.0, 15.0),
    "J7": (25.0, 15.0),
    "collector": (35.0, 15.0),
    "sewer_outflow": (5.0, 25.0),
    "B1_1": (15.0, 25.0),
    "B1_2": (25.0, 25.0),
    "B2_1": (35.0, 25.0),
    "B2_2": (5.0, 35.0),
}


def _ready(golden_extractor, *, clear: bool, full_tree: bool = False):
    """Golden analysis with series consolidation on; ``full_tree`` also prepares it for ``consolidate_to_datatree``."""
    system_dir = golden_extractor("multisim")
    cfg = system_dir / "analysis_config.yaml"
    flag = "toggle_clear_scenario_timeseries_after_consolidation: " + ("true" if clear else "false") + "\n"
    cfg.write_text(cfg.read_text() + "toggle_consolidate_timeseries: true\n" + flag)
    a = construct_golden_analysis(system_dir)
    if full_tree:
        from hhemt.scenario import TRITONSWMM_scenario, coupling_nodes_file_for

        shutil.rmtree(a.analysis_paths.analysis_datatree_zarr, ignore_errors=True)
        for event_iloc in a.df_sims.index:
            scen = TRITONSWMM_scenario(event_iloc, a)
            ts = xr.open_zarr(scen.scen_paths.output_tritonswmm_node_time_series, consolidated=False)
            nodes_file = coupling_nodes_file_for(scen.scen_paths.swmm_hydraulics_inp)
            nodes_file.parent.mkdir(parents=True, exist_ok=True)
            nodes_file.write_text("".join(f"{n}\n" for n in ts["node_id"].values[:3]))
    return a


def _scen(a, event_iloc: int):
    from hhemt.scenario import TRITONSWMM_scenario

    return TRITONSWMM_scenario(event_iloc, a)


def _processed(a, event_iloc: int) -> Path:
    return Path(_scen(a, event_iloc).scen_paths.sim_folder) / "processed"


def _write_capture(a, event_iloc: int, flows: dict[str, list[float]], coords: dict | None = None) -> Path:
    from hhemt.cf_conventions import apply_cf_attributes
    from hhemt.swmm_runoff_modeling import HydrologyNodeInflow, hydrology_capture_path, node_inflow_capture_dataset
    from hhemt.utils import write_zarr

    where = coords or _COORDS
    n = len(next(iter(flows.values())))
    node_inflow = HydrologyNodeInflow(
        {k: np.asarray(v, dtype="float64") for k, v in flows.items()},
        {k: where[k] for k in flows},
        np.arange(n) * _STEP / 3600.0,
        "CMS",
    )
    ds = node_inflow_capture_dataset(node_inflow, event_iloc)
    apply_cf_attributes(ds, "hydrology_inflow_summary")
    path = hydrology_capture_path(_scen(a, event_iloc))
    path.parent.mkdir(parents=True, exist_ok=True)
    write_zarr(ds, path, chunks=None, encoding={})
    return path


def _write_rpt_capture(a, event_iloc: int, ids: list[str]) -> Path:
    from hhemt.cf_conventions import apply_cf_attributes
    from hhemt.utils import write_zarr

    ds = xr.Dataset(
        {
            "total_runoff_volume": ("subcatchment_id", np.arange(len(ids), dtype="float64") + 1.0 + event_iloc),
            "peak_runoff": ("subcatchment_id", np.full(len(ids), 0.5)),
        },
        coords={"subcatchment_id": np.asarray(ids, dtype=str)},
        attrs={"flow_units": "cms", "runoff_continuity_error": 0.25 + event_iloc},
    )
    apply_cf_attributes(ds, "hydrology_rpt_summary")
    path = _processed(a, event_iloc) / "hydrology_rpt_summary.zarr"
    write_zarr(ds, path, chunks=None, encoding={})
    return path


def _group(a, name: str) -> xr.Dataset:
    return xr.open_zarr(a.analysis_paths.analysis_datatree_zarr, group=name, consolidated=False).load()


def _record(a, event_iloc: int):
    field = getattr(_scen(a, event_iloc).log, _FIELD, None)
    return None if field is None else field.get()


def test_full_consolidation_writes_the_hydrology_groups_stamps_version_4_and_upgrades_a_tree_stamped_3(
    golden_extractor,
):
    """Property: the public ``consolidate_to_datatree`` (not only ``consolidate_hydrology``) writes the three hydrology
    groups, leaves the root stamp at 4, and on a tree stamped 3 that lacks the groups appends them again.

    Class: INTEGRATION WIRING. Plausible wrong implementation: ``consolidate_hydrology`` correct in isolation and not
    reached from the tree builder, or reached only on the first build; or a version-3 tree whose stamp-3 rebuild path
    removes the groups and does not re-append them (D25 (g): the next consolidation appends them from the stores still
    on disk). Reachability: the real ``consolidate_to_datatree`` on the golden. Both-states anchor: after the first
    build the groups exist; after the stamp is set to 3 and the groups removed they exist again. Must-fire control:
    before the build there is no hydrology group on disk.
    Killed by: Not calling consolidate_hydrology from consolidate_to_datatree, or removing the groups on a stamp-3
        rebuild without re-appending them.
    Passes under a second correct implementation: A tree builder that appends the hydrology groups through a different
        public entry point, if the groups and the stamp are the same.
    Why required: the property is a consequence of D25 (c)/(d)/(e) as the code reads them (cited in the body above).
    """
    from hhemt.utils import deferred_consolidation, update_zarr_root_attrs, zarr_root_attrs

    a = _ready(golden_extractor, clear=False, full_tree=True)
    for event_iloc in range(3):
        _write_capture(a, event_iloc, {"J1": [1.0] * 12, "J2": [2.0] * 12})
        _write_rpt_capture(a, event_iloc, ["S1", "S2"])
    store = Path(a.analysis_paths.analysis_datatree_zarr)
    assert not (store / _FAMILY / "swmm_node_hydro").exists(), "control: nothing before the build"
    a.process.consolidate_to_datatree()
    for group in (_TS, _NODE, _SUB):
        assert (store / group).is_dir(), f"{group} is not reached from the public consolidation"
    assert zarr_root_attrs(store).get("consolidation_version") == 4
    shutil.rmtree(store / _FAMILY / "swmm_node_hydro")
    shutil.rmtree(store / _FAMILY / "swmm_subcatchment_hydro")
    with deferred_consolidation(store):
        update_zarr_root_attrs(store, {"consolidation_version": 3})
    a.process.consolidate_to_datatree()
    for group in (_TS, _NODE, _SUB):
        assert (store / group).is_dir(), f"{group} is not appended again on a tree stamped 3"
    assert zarr_root_attrs(store).get("consolidation_version") == 4
    assert _group(a, _TS)["event_iloc"].values.tolist() == [0, 1, 2]


def test_a_second_consolidation_changes_nothing_and_duplicates_no_event(golden_extractor):
    """Property: running ``consolidate_hydrology`` twice leaves every group identical and each event once.

    Class: IDEMPOTENCE. Plausible wrong implementation: an unconditional append on the second pass (the code comment
    in ``_hydrology_write_series`` names the hazard), or a summary group rewritten from NaN padding. Must-fire
    control: the first pass holds three events, so a duplicate would be visible as six.
    Killed by: An unconditional append in _hydrology_write_series or _hydrology_write_summary on the second pass.
    Passes under a second correct implementation: A pass that skips every event whose clearing record is done and
        verifies the rest by read-back.
    Why required: the property is a consequence of D25 (c)/(d)/(e) as the code reads them (cited in the body above).
    """
    a = _ready(golden_extractor, clear=False)
    for event_iloc in range(3):
        _write_capture(a, event_iloc, {"J1": [0.0, 1.0, 3.0] + [0.5] * 9, "J4": [2.0] * 12})
        _write_rpt_capture(a, event_iloc, ["S1", "S2", "S3"])
    a.process.consolidate_hydrology()
    before = {g: _group(a, g) for g in (_TS, _NODE, _SUB)}
    assert before[_TS]["event_iloc"].values.tolist() == [0, 1, 2]
    a.process.consolidate_hydrology()
    for g, was in before.items():
        now = _group(a, g)
        assert now["event_iloc"].values.tolist() == [0, 1, 2], f"{g} duplicated or lost an event"
        xr.testing.assert_equal(now.drop_vars("event_id", errors="ignore"), was.drop_vars("event_id", errors="ignore"))


def test_an_event_shorter_than_the_axis_is_rebuilt_from_the_group_at_its_own_length(golden_extractor):
    """Property: the hydrographs rebuilt from the group for an 11-step event on a 12-step axis have 11 rows (the NaN
    tail is not a row), a time column that starts at 0 and steps by the report step, and the same gridcell values the
    event carried, J2 at its cell and J5 at its cell.

    Class: NEW CAPABILITY. Plausible wrong implementation: a rebuild that keeps the padded tail (a 12th row of NaN or
    zero, which TRITON would read as a dry final step) or reads the full axis. The coder's own group-source arm runs on
    a FULL-length event, where the tail does not exist. Must-fire control: the same event consolidated beside a
    12-step event, so the axis really is 12 long (asserted).
    Killed by: hydrology_event_inflow_from_tree returning the padded tail (np.argmax(padding) replaced by the full
        axis length).
    Passes under a second correct implementation: A rebuild that takes the event length from the event window over the
        report step instead of the first NaN.
    Why required: the property is a consequence of D25 (c)/(d)/(e) as the code reads them (cited in the body above).
    """
    from hhemt.swmm_runoff_modeling import hydrograph_rebuild_source

    a = _ready(golden_extractor, clear=True)
    flows1 = {"J2": [2.0] * 11, "J5": [0.1] * 10 + [4.0]}
    _write_capture(a, 0, {"J1": [1.0] * 12})
    _write_capture(a, 1, flows1)
    _write_capture(a, 2, {"J1": [1.0] * 12})
    a.process.consolidate_hydrology()
    assert len(_group(a, _TS)["elapsed_s"]) == 12, "control: the axis is 12 long"
    assert not (_processed(a, 1) / "hydrology_inflow_summary.zarr").exists()
    scen = _scen(a, 1)
    hydro_out = Path(str(scen.scen_paths.swmm_hydro_inp).replace(".inp", ".out"))
    hydro_out.unlink()
    for p in (scen.scen_paths.hyg_timeseries, scen.scen_paths.hyg_locs):
        Path(p).unlink()
    assert hydrograph_rebuild_source(scen) == "group"
    scen._runoff_modeler.write_hydrograph_files()
    table = pd.read_csv(scen.scen_paths.hyg_timeseries, skiprows=2, header=None)
    locs = pd.read_csv(scen.scen_paths.hyg_locs)
    assert table.shape[0] == 11, f"the NaN-padded tail is a row: {table.shape}"
    assert not table.isna().any().any()
    np.testing.assert_allclose(table.iloc[:, 0].values, np.arange(11) * _STEP / 3600.0, rtol=1e-6)
    by_cell = {
        (float(r.iloc[0]), float(r.iloc[1])): table.iloc[:, 1 + i].values for i, (_, r) in enumerate(locs.iterrows())
    }
    np.testing.assert_allclose(by_cell[_COORDS["J2"]], flows1["J2"], rtol=1e-6)
    np.testing.assert_allclose(by_cell[_COORDS["J5"]], flows1["J5"], rtol=1e-6)


def test_the_hydrograph_table_sums_two_wet_nodes_of_one_gridcell_and_omits_a_dry_gridcell():
    """Property: ``hydrograph_frame`` writes ONE column for a gridcell holding two wet nodes, the sum of the two, one
    column per other wet gridcell in groupby order, and no column for a gridcell whose nodes are all dry.

    Class: REGRESSION against the landed sum (the coder moved it into a module function). The golden has one node per
    cell, so no landed test reaches the sum. Must-fire control: the dry cell's nodes are in the input, so an
    implementation that emitted a column per cell would add one.
    Killed by: hydrograph_frame emitting a column per gridcell, or taking the last node of a gridcell instead of the
        sum.
    Passes under a second correct implementation: A frame built with groupby-sum over a node-indexed DataFrame, if the
        column set and order are the same.
    Why required: the property is a consequence of D25 (c)/(d)/(e) as the code reads them (cited in the body above).
    """
    from hhemt.swmm_runoff_modeling import hydrograph_frame

    df = pd.DataFrame(
        {
            "node_key": ["a", "b", "c", "d", "e", "out"],
            "dem_x_coord": [5.0, 5.0, 15.0, 15.0, 25.0, 35.0],
            "dem_y_coord": [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
        }
    )
    inflow = {
        "a": np.array([1.0, 2.0, 3.0]),
        "b": np.array([0.5, 0.5, 0.5]),
        "c": np.zeros(3),
        "d": np.zeros(3),
        "e": np.array([0.0, 0.0, 4.0]),
    }
    frame = hydrograph_frame(inflow, df, ["out"], np.array([0.0, 1.0, 2.0]))
    assert list(frame.columns) == ["timestep", (5.0, 5.0), (25.0, 5.0)]
    np.testing.assert_allclose(frame[(5.0, 5.0)].values, [1.5, 2.5, 3.5])
    np.testing.assert_allclose(frame[(25.0, 5.0)].values, [0.0, 0.0, 4.0])


def test_a_captured_node_outside_the_fixed_axis_is_refused_naming_the_scenario_and_writes_nothing(golden_extractor):
    """Property: a capture holding a node the hydrology model does not have is refused with ``ProcessingError`` naming
    the scenario, and nothing of that event reaches the group or the log.

    Class: NEW CAPABILITY. Plausible wrong implementation: dropping the unknown node silently (its inflow is lost from
    the only copy once the capture is cleared). Must-fire control: the same event without the stray node consolidates.
    Killed by: Dropping the extra node in _hydrology_event_slices (reindex without the extra-set check).
    Passes under a second correct implementation: A refusal raised from the append step, provided it names the
        scenario and precedes any clearing.
    Why required: the property is a consequence of D25 (c)/(d)/(e) as the code reads them (cited in the body above).
    """
    a = _ready(golden_extractor, clear=True)
    coords = dict(_COORDS, ZZ=(5.0, 5.0))
    _write_capture(a, 0, {"J1": [1.0] * 12, "ZZ": [9.0] * 12}, coords=coords)
    with pytest.raises(Exception) as err:
        a.process.consolidate_hydrology()
    assert type(err.value).__name__ == "ProcessingError" and a.process._event_slug(0) in str(err.value)
    assert (_processed(a, 0) / "hydrology_inflow_summary.zarr").exists(), "the refused capture is kept"
    assert _record(a, 0) is None
    shutil.rmtree(_processed(a, 0) / "hydrology_inflow_summary.zarr")
    _write_capture(a, 0, {"J1": [1.0] * 12})
    for event_iloc in (1, 2):
        _write_capture(a, event_iloc, {"J1": [1.0] * 12})
    a.process.consolidate_hydrology()
    assert _record(a, 0)["state"] == "done"


def test_a_report_capture_that_arrives_after_the_inflow_capture_was_cleared_joins_the_record(golden_extractor):
    """Property: with the inflow capture already cleared and recorded, a later-arriving ``hydrology_rpt_summary`` store
    is consolidated and cleared on the next pass, the record then names BOTH stems, and the inflow slice is neither
    re-read nor changed.

    Class: NEW CAPABILITY (the record-merge arm of ``_clear_hydrology_stores``). Plausible wrong implementation: the
    second clearing overwrites the record with the report stem alone, which un-records the inflow capture, so the
    hydrographs reclaim and the group-sourced rebuild lose their gate. Must-fire control: after the first pass the
    record names the inflow stem alone.
    Killed by: _clear_hydrology_stores writing {**stems} instead of {**held, **stems}.
    Passes under a second correct implementation: A record kept per stem in two fields, if the inflow stem stays
        recorded after the report stem is cleared.
    Why required: the property is a consequence of D25 (c)/(d)/(e) as the code reads them (cited in the body above).
    """
    a = _ready(golden_extractor, clear=True)
    for event_iloc in range(3):
        _write_capture(a, event_iloc, {"J1": [1.0] * 12, "J3": [0.25] * 12})
    a.process.consolidate_hydrology()
    assert set(_record(a, 0)["stems"]) == {"hydrology_inflow_summary"}
    inflow_before = _group(a, _TS)
    for event_iloc in range(3):
        _write_rpt_capture(a, event_iloc, ["S1", "S2"])
    a.process.consolidate_hydrology()
    record = _record(a, 0)
    assert record["state"] == "done"
    assert set(record["stems"]) == {"hydrology_inflow_summary", "hydrology_rpt_summary"}
    assert not (_processed(a, 0) / "hydrology_rpt_summary.zarr").exists()
    xr.testing.assert_equal(
        _group(a, _TS).drop_vars("event_id", errors="ignore"), inflow_before.drop_vars("event_id", errors="ignore")
    )
    assert _group(a, _SUB)["event_iloc"].values.tolist() == [0, 1, 2]


def test_the_node_summary_of_a_padded_event_is_computed_over_its_own_steps(golden_extractor):
    """Property: for an 11-step event on a 12-step axis, ``total_inflow_volume`` is the trapezoid over its 11 points,
    ``inflow_max`` and ``time_of_inflow_max`` are those of the 11 points (the peak at the LAST own step is a time of
    11 steps, not NaN and not 12), and a node absent from the capture is 0 with a NaN time.

    Class: NEW CAPABILITY. Plausible wrong implementation: reductions over the padded axis (NaN-propagating, or a
    ``nanargmax`` over 12 that is right only when the peak is early). The coder's summary arm uses a single full-length
    event. Must-fire control: the peak is at the last own step, the position a padded-axis implementation moves.
    Killed by: Reducing over the padded axis (nanmax/nanargmax on 12 steps with the peak moved) or taking the time
        origin at the first period.
    Passes under a second correct implementation: A summary computed from the group slice after placement, if NaN
        padding is excluded from the reduction.
    Why required: the property is a consequence of D25 (c)/(d)/(e) as the code reads them (cited in the body above).
    """
    a = _ready(golden_extractor, clear=False)
    flows1 = {"J5": [0.1] * 10 + [4.0], "J2": [2.0] * 11}
    _write_capture(a, 0, {"J1": [1.0] * 12})
    _write_capture(a, 1, flows1)
    _write_capture(a, 2, {"J1": [1.0] * 12})
    a.process.consolidate_hydrology()
    s = _group(a, _NODE).sel(event_iloc=1)
    j5 = np.asarray(flows1["J5"])
    assert float(s["total_inflow_volume"].sel(node_id="J5")) == pytest.approx(
        float(np.sum((j5[:-1] + j5[1:]) / 2.0 * _STEP)), rel=1e-6
    )
    assert float(s["inflow_max"].sel(node_id="J5")) == pytest.approx(4.0)
    assert float(s["time_of_inflow_max"].sel(node_id="J5")) == pytest.approx(11 * _STEP / 60.0)
    assert float(s["inflow_max"].sel(node_id="J1")) == 0.0
    assert bool(s["time_of_inflow_max"].sel(node_id="J1").isnull())
    assert float(s["system_total_inflow_volume"]) == pytest.approx(float(s["total_inflow_volume"].sum()), rel=1e-9)


def _landed_reconstruction_block(source: str) -> str:
    """The reconstruction statements of the landed bit-identity test: from its ``df = pd.DataFrame(`` to the line
    before its shape assertion. Extracted from the landed source so the check runs THAT expression."""
    import textwrap

    lines = source.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.strip().startswith("df = pd.DataFrame("))
    end = next(i for i, ln in enumerate(lines) if ln.strip().startswith("assert rebuilt.shape"))
    return textwrap.dedent("\n".join(lines[start:end]))


def _run_landed_reconstruction(source: str):
    """Run the landed test's reconstruction on a constructed full-population capture with one dry gridcell.

    Returns ``(rebuilt, written)``: the landed expression's result and the hydrograph table's gridcell columns."""
    from hhemt.swmm_runoff_modeling import HydrologyNodeInflow, hydrograph_frame, node_inflow_capture_dataset

    node_df = pd.DataFrame(
        {
            "node_key": ["a", "b", "c", "out"],
            "dem_x_coord": [5.0, 15.0, 25.0, 35.0],
            "dem_y_coord": [5.0, 5.0, 5.0, 5.0],
        }
    )
    flows = {"a": np.array([1.0, 2.0]), "b": np.zeros(2), "c": np.array([0.5, 0.5])}
    where = {"a": (5.0, 5.0), "b": (15.0, 5.0), "c": (25.0, 5.0)}
    times = np.array([0.0, 10.0 / 3600.0])
    cap = node_inflow_capture_dataset(HydrologyNodeInflow(flows, where, times, "CMS"), 0).squeeze(
        "event_iloc", drop=True
    )
    written_cells = hydrograph_frame(flows, node_df, ["out"], times).iloc[:, 1:].to_numpy(dtype="float64")
    namespace = {"np": np, "pd": pd, "cap": cap}
    exec(_landed_reconstruction_block(source), namespace)  # noqa: S102 -- the landed test's own statements
    return namespace["rebuilt"], written_cells


def test_the_landed_bit_identity_reconstruction_holds_for_a_capture_that_covers_a_dry_gridcell():
    """Property: the reconstruction statements of ``tests/test_synth_reclaim_capture_and_arms.py::
    test_per_node_capture_is_bit_identical_to_the_written_hydrograph``, executed unchanged on a constructed capture
    that covers one dry gridcell, equal the hydrograph table's gridcell columns in shape and in value.

    Class: REGRESSION of a landed test outside the hydrology groups' own files: the capture holds every non-outfall node
    (D25 (c)) and the table omits a dry gridcell, so an unfiltered group-sum has an extra all-zero column. The check
    reads the landed source (``inspect.getsource``) and runs its own statements, so it is red on the pre-repair
    expression and green on the repaired one. Must-fire control: the next test runs the same driver on the source
    with the omission line removed and requires the shape mismatch.
    Killed by: removing the dry-gridcell omission from the landed expression.
    Passes under a second correct implementation: any omission rule that drops all-zero gridcells before comparing.
    Why required: otherwise the landed bit-identity test reds on any scenario with a wholly dry gridcell.
    """
    import inspect

    from tests import test_synth_reclaim_capture_and_arms as landed

    source = inspect.getsource(landed.test_per_node_capture_is_bit_identical_to_the_written_hydrograph)
    rebuilt, written = _run_landed_reconstruction(source)
    assert rebuilt.shape == written.shape, f"landed reconstruction {rebuilt.shape} != table {written.shape}"
    np.testing.assert_allclose(rebuilt, written, rtol=0, atol=1e-6)


def test_the_driver_is_red_on_the_landed_expression_without_its_dry_gridcell_omission():
    """Property (must-fire control of the test above): the same driver on the landed source with every line that
    filters by ``.sum(axis=0) > 0`` removed reports a shape mismatch (3 columns against 2).

    Class: CONTROL. Plausible wrong implementation: a driver that ignores the landed source and passes on its own
    arithmetic. Killed by: a driver that does not execute the landed statements.
    Passes under a second correct implementation: any driver that executes the extracted statements.
    Why required: separates a true green from a driver that cannot fail.
    """
    import inspect

    from tests import test_synth_reclaim_capture_and_arms as landed

    source = inspect.getsource(landed.test_per_node_capture_is_bit_identical_to_the_written_hydrograph)
    stripped = "\n".join(ln for ln in source.splitlines() if "sum(axis=0) > 0" not in ln)
    assert stripped != source, "control: the landed source carries the omission line"
    rebuilt, written = _run_landed_reconstruction(stripped)
    assert rebuilt.shape != written.shape
