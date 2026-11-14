"""The SWMM hydrology results are groups of the consolidated tree: ``swmm_node_hydro`` (series and summary) and
``swmm_subcatchment_hydro`` (summary), each rebuilt from the per-scenario hydrology stores.

No solver and no Snakemake: every test that needs an analysis runs on the committed multi-sim golden (a finished
tree whose three events carry a real ``swmm/hydro.inp``, ``swmm/hydro.out`` and ``swmm/hydro.rpt``, a 10 s report
step, a 120 s window and a 10 m DEM). Reading ``hydro.out`` through ``pyswmm.Output`` executes nothing. Each arm
names its control: the input class that makes a broken implementation read as correct.
"""

from __future__ import annotations

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
_STEP = 10  # the golden hydro.inp REPORT_STEP, in seconds

#: The golden hydro.inp's node positions (one DEM cell each, the cell centres of a 10 m DEM) and its one outfall.
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
_POPULATION = sorted(_COORDS)  # the 13 non-outfall nodes, by name, which is the fixed node axis


def _ready(golden_extractor, *, clear: bool):
    """Golden analysis with series consolidation on and the clearing switch on or off."""
    system_dir = golden_extractor("multisim")
    cfg = system_dir / "analysis_config.yaml"
    flag = "toggle_clear_scenario_timeseries_after_consolidation: " + ("true" if clear else "false") + "\n"
    cfg.write_text(cfg.read_text() + "toggle_consolidate_timeseries: true\n" + flag)
    return construct_golden_analysis(system_dir)


def _drop_captures(a) -> None:
    """Remove the per-node capture the golden extractor writes, so the scenarios are in the state of a tree prepared
    before the capture existed."""
    import shutil

    for event_iloc in a.df_sims.index:
        shutil.rmtree(_processed(a, int(event_iloc)) / "hydrology_inflow_summary.zarr", ignore_errors=True)


def _scen(a, event_iloc: int):
    from hhemt.scenario import TRITONSWMM_scenario

    return TRITONSWMM_scenario(event_iloc, a)


def _processed(a, event_iloc: int) -> Path:
    return Path(_scen(a, event_iloc).scen_paths.sim_folder) / "processed"


def _write_capture(a, event_iloc: int, flows: dict[str, list[float]], step: int = _STEP) -> Path:
    """Write the per-node capture of one event, in the form the preparation step writes it."""
    from hhemt.cf_conventions import apply_cf_attributes
    from hhemt.swmm_runoff_modeling import HydrologyNodeInflow, hydrology_capture_path, node_inflow_capture_dataset
    from hhemt.utils import write_zarr

    n = len(next(iter(flows.values())))
    node_inflow = HydrologyNodeInflow(
        {k: np.asarray(v, dtype="float64") for k, v in flows.items()},
        {k: _COORDS[k] for k in flows},
        np.arange(n) * step / 3600.0,
        "CMS",
    )
    ds = node_inflow_capture_dataset(node_inflow, event_iloc)
    apply_cf_attributes(ds, "hydrology_inflow_summary")
    path = hydrology_capture_path(_scen(a, event_iloc))
    path.parent.mkdir(parents=True, exist_ok=True)
    write_zarr(ds, path, chunks=None, encoding={})
    return path


def _write_rpt_capture(a, event_iloc: int, ids: list[str], **attrs) -> Path:
    from hhemt.cf_conventions import apply_cf_attributes
    from hhemt.utils import write_zarr

    ds = xr.Dataset(
        {
            "total_runoff_volume": ("subcatchment_id", np.arange(len(ids), dtype="float64") + 1.0 + event_iloc),
            "peak_runoff": ("subcatchment_id", np.full(len(ids), 0.5)),
        },
        coords={"subcatchment_id": np.asarray(ids, dtype=str)},
        attrs={"flow_units": "cms", "runoff_continuity_error": 0.25 + event_iloc, **attrs},
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


def test_the_capture_covers_the_full_node_population_zeros_included(monkeypatch):
    """Property: the per-node capture holds every non-outfall node of the hydrology model, a node with no inflow at
    every step included as all zeros, and the hydrograph table still omits the gridcells that received nothing.

    Class: REGRESSION against the landed writer (it kept only nodes whose inflow summed above zero). Plausible wrong
    implementation: that filter. Reachability: a constructed ``hydro.out`` (a stand-in for ``pyswmm.Output``) of five
    population nodes, two of them dry, and one outfall, through the real ``read_hydrology_node_inflow``,
    ``node_inflow_capture_dataset`` and ``hydrograph_frame``. Both-states anchor: the three wet nodes carry their
    constructed values in the capture and in the table. Must-fire control: the landed rule applied to the same
    input keeps three nodes, so the five here is the property and not the input.
    Killed by: restoring the filter. Passes under a second correct implementation that stores zeros from the
    ``hydro.out`` node list instead of the node table, if the population is the same five.
    """
    import hhemt.swmm_runoff_modeling as srm

    times = pd.to_datetime(["2000-01-01 00:00:10", "2000-01-01 00:00:20", "2000-01-01 00:00:30"])
    values = {
        "a": [0.0, 1.0, 2.0],
        "b": [0.0, 0.0, 0.0],
        "c": [0.5, 0.5, 0.5],
        "d": [0.0, 0.0, 0.0],
        "e": [3.0, 0.0, 1.0],
        "out": [9.0, 9.0, 9.0],
    }

    class FakeOutput:
        """A stand-in for ``pyswmm.Output`` that serves the constructed series and reports CMS."""

        units = {"system": "SI", "flow": "CMS", "pollutant": "NONE"}

        def __init__(self, path):
            self.path = path

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def node_series(self, key, attribute):
            """The constructed series of ``key``, keyed by report time, as ``pyswmm.Output.node_series`` returns it."""
            return dict(zip(times, values[key], strict=True))

    # The reader takes each series and the time axis from the SWMM parser's window readers, so those two are
    # stood in for with the constructed series, and the double lists the node positions the reader looks up.
    import hhemt.swmm_output_parser as sop

    FakeOutput.nodes = {k: i for i, k in enumerate(values)}
    monkeypatch.setattr(srm, "Output", FakeOutput)
    monkeypatch.setattr(
        sop, "_out_series_values", lambda out, kind, ident, index, attribute: np.asarray(values[ident], dtype=float)
    )
    monkeypatch.setattr(sop, "_out_time_axis", lambda out, f_outfile: list(times))
    df = pd.DataFrame(
        {
            "node_key": ["a", "b", "c", "d", "e", "out"],
            "dem_x_coord": [5.0, 15.0, 25.0, 35.0, 45.0, 55.0],
            "dem_y_coord": [5.0] * 6,
        }
    )
    node_inflow = srm.read_hydrology_node_inflow("unused.out", df, ["out"])
    ds = srm.node_inflow_capture_dataset(node_inflow, 7)
    assert [str(v) for v in ds["node_id"].values] == ["a", "b", "c", "d", "e"]
    assert ds["inflow"].isel(event_iloc=0).sel(node_id="b").values.tolist() == [0.0, 0.0, 0.0]
    assert ds["inflow"].isel(event_iloc=0).sel(node_id="e").values.tolist() == [3.0, 0.0, 1.0]
    landed_rule = [k for k, v in node_inflow.inflow_by_node.items() if np.asarray(v).sum() > 0]
    assert len(landed_rule) == 3, "control: the landed rule keeps three of the same five"
    frame = srm.hydrograph_frame(node_inflow.inflow_by_node, df, ["out"], node_inflow.times_hr)
    assert len(frame.columns) == 1 + 3, "the hydrograph table keeps one column per wet gridcell and the time column"
    assert np.allclose(frame[(45.0, 5.0)].values, [3.0, 0.0, 1.0])


def test_two_events_with_different_inflow_nodes_and_lengths_land_on_one_fixed_axis_pair(golden_extractor):
    """Property: two events whose captures hold different node sets and different numbers of steps land on ONE
    ``node_id`` axis (the 13 non-outfall nodes of ``hydro.inp``) and ONE ``elapsed_s`` axis at the report step, a
    node the capture lacks being 0 and the shorter event NaN-padded after its last step.

    Class: NEW CAPABILITY. Plausible wrong implementations: an axis taken from the first capture (the second
    event's extra node is then refused or dropped), and zero padding of the short event (its tail would read as a
    dry spell). Reachability: the real ``consolidate_hydrology`` over three golden events (the third rebuilt from
    ``hydro.out``). Both-states anchor: event 0 fills all 12 steps, event 1 fills 11 and is NaN at the 12th.
    Invocation isolation: nothing here depends on the working directory. Must-fire control: appending the second
    event's own capture, unreindexed, to the group raises ``ValueError``.
    Killed by: the two above. Passes under a second correct implementation that writes a group created whole from
    the axes and fills each event by a region write, if the axes and values are the same.
    """
    a = _ready(golden_extractor, clear=False)
    flows0 = {"J1": [0.0, 1.0, 3.0, 2.0] + [0.5] * 8, "J2": [1.0] * 12, "J3": [0.0] * 12}
    flows1 = {"J2": [2.0] * 11, "J5": [0.1] * 10 + [4.0]}
    _write_capture(a, 0, flows0)
    _write_capture(a, 1, flows1)
    a.process.consolidate_hydrology()
    ts = _group(a, _TS)
    assert [str(v) for v in ts["node_id"].values] == _POPULATION
    assert ts["elapsed_s"].values.tolist() == [_STEP * k for k in range(1, 13)]
    assert ts["elapsed_s"].dtype.kind == "i"
    inflow = ts["inflow"]
    got0 = inflow.sel(event_iloc=0, node_id="J1").values
    np.testing.assert_allclose(got0, np.asarray(flows0["J1"], dtype="float32"))
    assert (inflow.sel(event_iloc=0, node_id="B1_1").values == 0.0).all(), "an absent node is 0, never NaN"
    assert bool(inflow.sel(event_iloc=1, node_id="J2").isel(elapsed_s=11).isnull()), "event 1 is NaN-padded"
    assert float(inflow.sel(event_iloc=1, node_id="J5").isel(elapsed_s=9)) == pytest.approx(0.1)
    assert float(inflow.sel(event_iloc=1, node_id="J5").isel(elapsed_s=10)) == pytest.approx(4.0)
    assert not bool(inflow.sel(event_iloc=0, node_id="J2").isnull().any()), "control: event 0 has no padding"
    raw = xr.Dataset(
        {"inflow": (("event_iloc", "node_id", "elapsed_s"), np.zeros((1, 2, 12), dtype="float32"))},
        coords={"event_iloc": [9], "node_id": ["J2", "J5"], "elapsed_s": ts["elapsed_s"].values},
    )
    with pytest.raises(ValueError):
        raw.to_zarr(a.analysis_paths.analysis_datatree_zarr, group=_TS, mode="a", append_dim="event_iloc")


def test_the_node_summary_variables_equal_hand_computed_values_and_the_run_total_is_a_data_variable(golden_extractor):
    """Property: ``total_inflow_volume``, ``inflow_max``, ``time_of_inflow_max`` and ``system_total_inflow_volume``
    equal hand-computed values for a constructed event, ``time_of_inflow_max`` is NaN for a dry node and counts
    minutes from the event start (so its first period is one report step), and the run total is a data variable.

    Class: NEW CAPABILITY. Plausible wrong implementations: a maximum over the padded axis, a time origin at the
    first period instead of the event start, and the run total stored as an attribute. Reachability: the real
    consolidation. Both-states anchor: J1 peaks at its third step, so its time is 3 x 10 s = 0.5 min; J3 is dry.
    Must-fire control: written as an attribute, two events with different totals lose it under
    ``combine_attrs="drop_conflicts"``.
    Killed by: the three above. Passes under a second correct implementation that takes the volume from the
    trapezoid in seconds and not in hours times 3600, if the values agree.
    """
    a = _ready(golden_extractor, clear=False)
    flows0 = {"J1": [0.0, 1.0, 3.0, 2.0] + [0.5] * 8, "J2": [1.0] * 12, "J3": [0.0] * 12}
    _write_capture(a, 0, flows0)
    a.process.consolidate_hydrology()
    summary = _group(a, _NODE)
    j1 = np.asarray(flows0["J1"])
    by_hand = float(np.sum((j1[:-1] + j1[1:]) / 2.0 * _STEP))
    assert float(summary["total_inflow_volume"].sel(event_iloc=0, node_id="J1")) == pytest.approx(by_hand, rel=1e-6)
    assert float(summary["inflow_max"].sel(event_iloc=0, node_id="J1")) == pytest.approx(3.0)
    assert float(summary["time_of_inflow_max"].sel(event_iloc=0, node_id="J1")) == pytest.approx(3 * _STEP / 60.0)
    assert bool(summary["time_of_inflow_max"].sel(event_iloc=0, node_id="J3").isnull()), "a dry node has no time"
    assert float(summary["inflow_max"].sel(event_iloc=0, node_id="J3")) == 0.0
    total = float(summary["total_inflow_volume"].sel(event_iloc=0).sum())
    assert float(summary["system_total_inflow_volume"].sel(event_iloc=0)) == pytest.approx(total)
    assert "system_total_inflow_volume" in summary.data_vars and "system_total_inflow_volume" not in summary.attrs
    one = xr.Dataset({"x": ("event_iloc", [1.0])}, coords={"event_iloc": [0]}, attrs={"total": 3.0})
    two = xr.Dataset({"x": ("event_iloc", [1.0])}, coords={"event_iloc": [1]}, attrs={"total": 4.0})
    assert "total" not in xr.concat([one, two], dim="event_iloc", combine_attrs="drop_conflicts").attrs


def test_the_subcatchment_summary_lands_on_the_declared_axis_and_a_missing_subcatchment_is_nan(golden_extractor):
    """Property: a constructed ``hydrology_rpt_summary`` capture lands on ``swmm_subcatchment_hydro/summary`` on the
    11 subcatchments ``hydro.inp`` declares, the six the capture lacks hold NaN rather than vanishing, and the run
    level continuity error is a data variable.

    Class: NEW CAPABILITY. Plausible wrong implementations: an axis taken from the capture (5 rows), and the
    continuity error left an attribute (lost at the cross-event join). Reachability: the real consolidation of
    events 0 and 1, the second with a different continuity error. Both-states anchor: S1 to S5 carry the
    constructed values. Must-fire control: the second event's value survives beside the first's, which an
    attribute would not.
    Killed by: the two above. Passes under a second correct implementation that reindexes in the writer, if the
    axis is the declared one.
    """
    a = _ready(golden_extractor, clear=False)
    for event_iloc in range(3):
        _write_capture(a, event_iloc, {"J1": [1.0] * 12})
    _write_rpt_capture(a, 0, ["S1", "S2", "S3", "S4", "S5"])
    _write_rpt_capture(a, 1, ["S1", "S2", "S3", "S4", "S5"])
    a.process.consolidate_hydrology()
    sub = _group(a, _SUB)
    assert [str(v) for v in sub["subcatchment_id"].values] == [f"S{i}" for i in range(1, 12)]
    assert sub["event_iloc"].values.tolist() == [0, 1], "event 2 has no report capture and gets no slice"
    assert float(sub["total_runoff_volume"].sel(event_iloc=0, subcatchment_id="S2")) == pytest.approx(2.0)
    assert bool(sub["total_runoff_volume"].sel(event_iloc=0, subcatchment_id="S9").isnull()), "NaN, not a dropped row"
    assert sub["runoff_continuity_error"].values.tolist() == [0.25, 1.25]


@pytest.mark.parametrize(
    ("enabled", "expected"),
    [
        ({"tritonswmm", "triton"}, "tritonswmm"),
        ({"triton", "swmm"}, "triton_only"),
        ({"swmm"}, "swmm_only"),
    ],
)
def test_the_hydrology_groups_have_one_home_by_the_family_placement_rule(
    golden_extractor, monkeypatch, enabled, expected
):
    """Property: the three groups exist under ONE family: ``tritonswmm`` when the coupled model is enabled (the
    TRITON-only model beside it gets none), else ``triton_only``, else ``swmm_only``.

    Class: NEW CAPABILITY. Plausible wrong implementations: one copy per enabled family (a duplicate of one
    result), and a fixed ``tritonswmm`` home. Reachability: the real consolidation with the enabled model set
    changed. Both-states anchor: each parametrized row names its own home and the other two homes are absent.
    Must-fire control: the row with the coupled family off places the groups under ``triton_only``, which a fixed
    home could not.
    Killed by: those. Passes under a second correct implementation that places the groups from the system config.
    """
    a = _ready(golden_extractor, clear=False)
    monkeypatch.setattr(a, "_get_enabled_model_types", lambda: list(enabled))
    for event_iloc in range(3):
        _write_capture(a, event_iloc, {"J1": [1.0] * 12})
    a.process.consolidate_hydrology()
    store = Path(a.analysis_paths.analysis_datatree_zarr)
    homes = {f for f in ("tritonswmm", "triton_only", "swmm_only") if (store / f / "swmm_node_hydro").is_dir()}
    assert homes == {expected}


def test_the_stores_are_cleared_only_after_every_slice_reads_back_and_a_failed_readback_keeps_them(
    golden_extractor, monkeypatch
):
    """Property: with the switch on, both per-scenario hydrology stores are deleted and the log record
    ``consolidated_hydrology_cleared``
    is ``done`` on the scenario log only after every slice passes its read-back; an event whose inflow read-back
    fails keeps both stores and records nothing, while the event before it is cleared.

    Class: NEW CAPABILITY. Plausible wrong implementation: deleting the capture as soon as it is appended. Both-
    states anchor: event 0 is cleared with the record ``done`` and names both stems. Must-fire control: with the
    read-back of event 1 forced to disagree, its stores exist and its record is absent.
    Killed by: that. Passes under a second correct implementation that clears per store in either order.
    """
    import hhemt.processing_analysis as pa

    cls = pa.TRITONSWMM_analysis_post_processing
    a = _ready(golden_extractor, clear=True)
    for event_iloc in range(3):
        _write_capture(a, event_iloc, {"J1": [1.0] * 12, "J2": [2.0] * 12})
        _write_rpt_capture(a, event_iloc, ["S1", "S2"])
    real = cls._stored_slice_equals
    monkeypatch.setattr(
        cls,
        "_stored_slice_equals",
        staticmethod(lambda s, g, e, i: False if (i == 1 and g == _TS) else real(s, g, e, i)),
    )
    with pytest.raises(Exception) as err:
        a.process.consolidate_hydrology()
    assert type(err.value).__name__ == "ProcessingError"
    assert _processed(a, 1).joinpath("hydrology_inflow_summary.zarr").exists(), "an unverified slice lost its source"
    assert _processed(a, 1).joinpath("hydrology_rpt_summary.zarr").exists()
    assert _record(a, 1) is None
    assert not _processed(a, 0).joinpath("hydrology_inflow_summary.zarr").exists(), "control: event 0 was cleared"
    assert not _processed(a, 0).joinpath("hydrology_rpt_summary.zarr").exists()
    record = _record(a, 0)
    assert record["state"] == "done"
    assert set(record["stems"]) == {"hydrology_inflow_summary", "hydrology_rpt_summary"}
    assert record["stems"]["hydrology_inflow_summary"][_TS] == ["inflow"]
    assert "system_total_inflow_volume" in record["stems"]["hydrology_inflow_summary"][_NODE]


def test_a_pending_record_is_finished_from_the_groups_and_never_read_back_from_a_half_deleted_store(golden_extractor):
    """Property: a scenario recorded ``pending`` whose store is half deleted is finished from the groups (the store
    is deleted, the record set ``done``) and the half-deleted store is never compared; a pending record whose group
    lacks the event refuses naming the scenario and deletes nothing.

    Class: NEW CAPABILITY. Plausible wrong implementation: comparing the surviving chunks against the group, which
    fails and is answered by overwriting a verified slice. Both-states anchor: after the first consolidation event 0
    is cleared; the record is then reset to ``pending`` with a leftover store. Must-fire control: the same record
    with the group's event removed refuses and keeps the leftover.
    Killed by: that. Passes under a second correct implementation that finishes the delete before checking.
    """
    a = _ready(golden_extractor, clear=True)
    for event_iloc in range(3):
        _write_capture(a, event_iloc, {"J1": [1.0] * 12})
    a.process.consolidate_hydrology()
    record = _record(a, 0)
    assert record["state"] == "done"
    leftover = _write_capture(a, 0, {"J1": [99.0] * 12})
    _scen(a, 0).log.consolidated_hydrology_cleared.set({"state": "pending", "stems": record["stems"]})
    a.process.consolidate_hydrology()
    assert not leftover.exists() and _record(a, 0)["state"] == "done"
    assert float(_group(a, _TS)["inflow"].sel(event_iloc=0, node_id="J1").max()) == pytest.approx(1.0)
    leftover = _write_capture(a, 0, {"J1": [99.0] * 12})
    bad = {
        k: {g: v + ["not_a_variable"] if g == _TS else v for g, v in groups.items()}
        for k, groups in record["stems"].items()
    }
    _scen(a, 0).log.consolidated_hydrology_cleared.set({"state": "pending", "stems": bad})
    with pytest.raises(Exception) as err:
        a.process.consolidate_hydrology()
    assert type(err.value).__name__ == "ProcessingError" and a.process._event_slug(0) in str(err.value)
    assert leftover.exists(), "a refused finish deletes nothing"


def test_a_scenario_with_no_capture_is_consolidated_from_hydro_out_and_refused_without_it(golden_extractor):
    """Property: an older scenario with no capture store is consolidated by rebuilding the capture from ``hydro.out``
    in memory (no capture store is created), and with ``hydro.out`` also absent it is refused naming the scenario.

    Class: NEW CAPABILITY. Plausible wrong implementations: skipping the scenario (its slice would be NaN or
    absent), and writing the rebuilt capture to disk (one more copy of the data being consolidated). Reachability:
    the golden's real ``hydro.out`` through the real consolidation. Both-states anchor: the group's values equal
    ``read_hydrology_node_inflow`` of that file. Must-fire control: the same scenario with the file removed raises
    and the message carries the scenario's slug.
    Killed by: those. Passes under a second correct implementation that rebuilds into a temporary store.
    """
    from hhemt.scenario import return_df_of_nodes_grouped_by_DEM_gridcell
    from hhemt.swmm_runoff_modeling import read_hydrology_node_inflow

    a = _ready(golden_extractor, clear=False)
    _drop_captures(a)
    scen = _scen(a, 2)
    hydro_out = Path(str(scen.scen_paths.swmm_hydro_inp).replace(".inp", ".out"))
    df, outfalls = return_df_of_nodes_grouped_by_DEM_gridcell(
        scen.scen_paths.swmm_hydro_inp, a._system.sys_paths.dem_processed
    )
    expected = read_hydrology_node_inflow(hydro_out, df, outfalls)
    a.process.consolidate_hydrology()
    ts = _group(a, _TS)
    got = ts["inflow"].sel(event_iloc=2, node_id="J4").values
    np.testing.assert_allclose(got, np.asarray(expected.inflow_by_node["J4"], dtype="float32"))
    assert not (_processed(a, 2) / "hydrology_inflow_summary.zarr").exists(), "the rebuilt capture is not stored"
    hydro_out.unlink()
    b = _ready_again(a)
    with pytest.raises(Exception) as err:
        b.process.consolidate_hydrology()
    assert type(err.value).__name__ == "ProcessingError" and b.process._event_slug(2) in str(err.value)


def _ready_again(a):
    """The same analysis with its group deleted, so the consolidation starts from nothing."""
    import shutil

    shutil.rmtree(Path(a.analysis_paths.analysis_datatree_zarr) / _FAMILY / "swmm_node_hydro")
    return a


def test_a_400_step_event_at_120_s_lands_every_value_at_its_position_on_an_integer_axis(golden_extractor, monkeypatch):
    """Property: a constructed 400-step event at 120 s lands every value at its own position and ``elapsed_s`` is
    integer seconds, although the capture's float32 hours times 3600 are not.

    Class: NEW CAPABILITY. Plausible wrong implementation: converting the stored float32 hours to seconds and
    reindexing by label, which drops or misplaces periods. Reachability: the real ``_hydrology_event_slices`` on a
    constructed capture, the event window set to 400 x 120 s. Both-states anchor: every one of the 400 values
    sits at its index. Must-fire control: the label reindex of the same capture loses periods.
    Killed by: that. Passes under a second correct implementation that places by an index array, if the positions
    agree.
    """
    from hhemt.scenario import TRITONSWMM_scenario
    from hhemt.swmm_runoff_modeling import HydrologyNodeInflow, node_inflow_capture_dataset

    a = _ready(golden_extractor, clear=False)
    step, n = 120, 400
    values = np.linspace(0.1, 40.0, n)
    ds = node_inflow_capture_dataset(
        HydrologyNodeInflow({"J1": values}, {"J1": (5.0, 5.0)}, np.arange(n) * step / 3600.0, "CMS"), 0
    ).isel(event_iloc=0, drop=True)
    start = pd.Timestamp("2000-01-01")
    monkeypatch.setattr(
        TRITONSWMM_scenario, "_resolve_event_window", lambda self: (start, start + pd.Timedelta(seconds=step * n))
    )
    axis = step * np.arange(1, n + 1)
    series, _ = a.process._hydrology_event_slices(
        _scen(a, 0), 0, ds, (["J1"], np.array([5.0]), np.array([5.0])), float(step), axis
    )
    placed = series["inflow"].sel(event_iloc=0, node_id="J1").values
    np.testing.assert_allclose(placed, values.astype("float32"))
    assert series["elapsed_s"].dtype.kind == "i" and series["elapsed_s"].values.tolist() == axis.tolist()
    seconds = ds["timestep"].values.astype("float64") * 3600.0
    assert int((seconds != np.rint(seconds)).sum()) > 0, "control: the stored hours are not whole seconds"
    by_label = xr.DataArray(values, dims="elapsed_s", coords={"elapsed_s": seconds}).reindex(elapsed_s=axis - step)
    assert int(by_label.isnull().sum()) > 0, "control: reindexing by label loses periods"


def test_an_event_whose_times_are_not_whole_report_steps_is_refused_naming_the_scenario(golden_extractor):
    """Property (control of the axis arm): an event whose recorded times are shifted off the report step, or whose
    count lies more than one period from its window, is refused with ``ProcessingError`` naming the scenario and
    nothing is written.

    Class: NEW CAPABILITY. Plausible wrong implementation: placing by position without checking the times. Both-
    states anchor: the unshifted capture consolidates in the arms above. Must-fire control: the shifted capture
    raises; so does one of 9 steps against a 12 step window.
    Killed by: that. Passes under a second correct implementation that raises from the reindex.
    """
    a = _ready(golden_extractor, clear=False)
    _write_capture(a, 0, {"J1": [1.0] * 12}, step=7)
    with pytest.raises(Exception) as err:
        a.process.consolidate_hydrology()
    assert type(err.value).__name__ == "ProcessingError" and a.process._event_slug(0) in str(err.value)
    import shutil

    shutil.rmtree(_processed(a, 0) / "hydrology_inflow_summary.zarr")
    _write_capture(a, 0, {"J1": [1.0] * 9})
    with pytest.raises(Exception) as err:
        a.process.consolidate_hydrology()
    assert type(err.value).__name__ == "ProcessingError" and a.process._event_slug(0) in str(err.value)


def _reference_hyg(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, skiprows=2, header=None)


def test_the_gate_rebuilds_the_hydrographs_from_the_capture_then_from_the_group_and_raises_with_neither(
    golden_extractor,
):
    """Property: with the hydrograph files and ``hydro.out`` absent, ``write_hydrograph_files`` rebuilds
    ``tseries.hyg`` from the per-node capture, and once the capture is cleared into the group and recorded, from
    the event's slice of ``swmm_node_hydro/timeseries``, with the same per-gridcell columns; with none of the three
    sources it raises naming all three.

    Class: NEW CAPABILITY. Plausible wrong implementations: a gate that raises when ``hydro.out`` is absent (the
    landed one), and a rebuild that reads the group's padded NaN tail. Reachability: the real gate and writer on
    the golden scenario with ``hydro.out`` removed. Both-states anchor: the columns carry the constructed per-node
    values (every node is its own DEM cell here) and the time column starts at 0 and steps by 10 s. Must-fire
    control: with the capture gone and the record removed, the rebuild raises and the message names the output, the
    capture and the group.
    Killed by: those. Passes under a second correct implementation that sums the capture by gridcell in another
    order, if the column set and values agree.
    """
    from hhemt.swmm_runoff_modeling import hydrograph_rebuild_source

    a = _ready(golden_extractor, clear=True)
    flows = {"J1": [0.0, 1.0, 3.0] + [2.0] * 9, "J2": [0.0] * 12, "J4": [0.5] * 12}
    for event_iloc in range(3):
        _write_capture(a, event_iloc, flows)
    scen = _scen(a, 0)
    hydro_out = Path(str(scen.scen_paths.swmm_hydro_inp).replace(".inp", ".out"))
    hydro_out.unlink()
    for p in (scen.scen_paths.hyg_timeseries, scen.scen_paths.hyg_locs):
        Path(p).unlink()
    assert hydrograph_rebuild_source(scen) == "capture"
    scen._runoff_modeler.write_hydrograph_files()
    from_capture = _reference_hyg(scen.scen_paths.hyg_timeseries)
    assert from_capture.shape[1] == 1 + 2, "a time column and the two wet gridcells; the dry node is omitted"
    np.testing.assert_allclose(from_capture.iloc[:, 0].values, np.arange(12) * _STEP / 3600.0, rtol=1e-6)
    np.testing.assert_allclose(from_capture.iloc[:, 1].values, flows["J1"], rtol=1e-6)
    np.testing.assert_allclose(from_capture.iloc[:, 2].values, flows["J4"], rtol=1e-6)
    # clear the capture into the group, then rebuild from the group
    a.process.consolidate_hydrology()
    assert not (_processed(a, 0) / "hydrology_inflow_summary.zarr").exists()
    scen = _scen(a, 0)
    for p in (scen.scen_paths.hyg_timeseries, scen.scen_paths.hyg_locs):
        Path(p).unlink()
    assert hydrograph_rebuild_source(scen) == "group"
    scen._runoff_modeler.write_hydrograph_files()
    from_group = _reference_hyg(scen.scen_paths.hyg_timeseries)
    np.testing.assert_allclose(from_group.values, from_capture.values, rtol=1e-6)
    scen.log.consolidated_hydrology_cleared.clear()
    for p in (scen.scen_paths.hyg_timeseries, scen.scen_paths.hyg_locs):
        Path(p).unlink()
    assert hydrograph_rebuild_source(_scen(a, 0)) is None
    with pytest.raises(Exception) as err:
        _scen(a, 0)._runoff_modeler.write_hydrograph_files()
    assert type(err.value).__name__ == "ProcessingError"
    text = str(err.value)
    assert "hydro.out" in text and "hydrology_inflow_summary.zarr" in text and "swmm_node_hydro/timeseries" in text


def test_the_hydrographs_reclaim_accepts_a_recorded_consolidation_of_the_capture(golden_extractor):
    """Property: the ``hydrographs`` reclaim removes ``tseries.hyg`` and ``loc.txt`` when the capture is landed OR
    the scenario's log records the capture as consolidated; it keeps them when neither holds, and a record that
    names only the report capture does not count.

    Class: NEW CAPABILITY. Plausible wrong implementation: a gate that reads the capture store alone (it declines
    forever after the clearing). Reachability: the real ``reclaim_scenario_scoped_classes`` on the golden
    scenario. Both-states anchor: with the record, the two files are gone. Must-fire control: with no capture and
    no record, or a record of the report stem only, they remain.
    Killed by: that. Passes under a second correct implementation that checks the tree instead of the record.
    """
    from hhemt.process_simulation import reclaim_scenario_scoped_classes

    a = _ready(golden_extractor, clear=False)
    _drop_captures(a)
    scen = _scen(a, 0)
    hyg = [Path(scen.scen_paths.hyg_timeseries), Path(scen.scen_paths.hyg_locs)]
    assert all(p.exists() for p in hyg)
    reclaim_scenario_scoped_classes(scen, ("hydrographs",))
    assert all(p.exists() for p in hyg), "control: no capture and no record keeps the files"
    scen.log.consolidated_hydrology_cleared  # noqa: B018 -- the field exists and is None
    from hhemt.processing_analysis import TRITONSWMM_analysis_post_processing as pp

    pp._log_field(scen.log, _FIELD).set({"state": "done", "stems": {"hydrology_rpt_summary": {_SUB: ["x"]}}})
    reclaim_scenario_scoped_classes(_scen(a, 0), ("hydrographs",))
    assert all(p.exists() for p in hyg), "control: a record of the report stem alone does not satisfy the gate"
    pp._log_field(scen.log, _FIELD).set({"state": "done", "stems": {"hydrology_inflow_summary": {_TS: ["inflow"]}}})
    reclaim_scenario_scoped_classes(_scen(a, 0), ("hydrographs",))
    assert not any(p.exists() for p in hyg)


def test_every_new_variable_of_the_three_groups_carries_a_unit_and_the_log_field_is_additive(golden_extractor):
    """Property: every numeric data variable of the three groups carries a non-empty ``units`` and no name carries a
    unit token; a scenario log written before the field loads with it ``None``, and the record round-trips through
    the JSON the log writes.

    Class: NEW CAPABILITY. Plausible wrong implementation: a new variable with no CF entry (stored without a unit).
    Both-states anchor: the six variables below each carry their unit. Must-fire control: a stored name with no CF
    entry fails the same check.
    Killed by: that. Passes under a second correct implementation that stamps the unit in the writer.
    """
    import json
    import re

    a = _ready(golden_extractor, clear=False)
    _write_capture(a, 0, {"J1": [1.0] * 12})
    _write_rpt_capture(a, 0, ["S1"])
    a.process.consolidate_hydrology()
    units = {}
    for name in (_TS, _NODE, _SUB):
        for var, da in _group(a, name).data_vars.items():
            if np.issubdtype(da.dtype, np.number):
                units[str(var)] = da.attrs.get("units")
    assert units["inflow"] == "m3 s-1" and units["inflow_max"] == "m3 s-1"
    assert units["time_of_inflow_max"] == "min" and units["system_total_inflow_volume"] == "m3"
    assert units["total_inflow_volume"] == "m3" and units["total_runoff_volume"] == "10^6 L"
    assert all(isinstance(u, str) and u.strip() for u in units.values()), units
    assert not [v for v in units if re.search(r"(^|_)(m|min|cms|m3|hr|s|mm)(_|$)", v)]
    from hhemt.cf_conventions import apply_cf_attributes

    stray = xr.Dataset({"stored_without_an_entry": ("x", [1.0])})
    assert "units" not in apply_cf_attributes(stray, "hydrology_inflow_summary")["stored_without_an_entry"].attrs
    scen = _scen(a, 0)
    log_path = Path(scen.scen_paths.scenario_prep_log)
    payload = json.loads(log_path.read_text())
    payload.pop(_FIELD, None)
    log_path.write_text(json.dumps(payload))
    older = _scen(a, 0).log
    assert getattr(older, _FIELD) is None, "a log written before the field loads it as None"
    from hhemt.processing_analysis import TRITONSWMM_analysis_post_processing as pp

    pp._log_field(older, _FIELD).set({"state": "pending", "stems": {"hydrology_inflow_summary": {_TS: ["inflow"]}}})
    assert json.loads(log_path.read_text())[_FIELD]["state"] == "pending"
    assert _record(a, 0)["stems"]["hydrology_inflow_summary"] == {_TS: ["inflow"]}


def test_the_first_output_period_is_one_report_step_after_the_event_start_and_the_step_is_read_from_the_inp(
    golden_extractor,
):
    """Property: the first output period of the hydrology model's binary output is the simulation
    start plus ONE report step, so the group's ``elapsed_s`` axis starts at one step, and the step is the
    ``REPORT_STEP`` of ``hydro.inp`` read by ``inp_report_step_s``.

    Class: NEW CAPABILITY. Plausible wrong implementation: an axis starting at 0 (every value one step early).
    Reachability: the golden's real ``hydro.out`` through ``pyswmm.Output`` (a read; nothing executes). Both-states
    anchor: first time minus start equals the report step. Must-fire control: it does not equal zero, and a
    ``REPORT_STEP`` written as ``00:02:00`` reads 120.
    Killed by: that. Passes under a second correct implementation that reads the step from the output itself.
    """
    from pyswmm import Output

    from hhemt.swmm_runoff_modeling import inp_report_step_s

    a = _ready(golden_extractor, clear=False)
    scen = _scen(a, 0)
    inp = Path(scen.scen_paths.swmm_hydro_inp)
    assert inp_report_step_s(inp) == _STEP
    with Output(str(inp).replace(".inp", ".out")) as out:
        first, start = out.times[0], out.start
    assert (first - start).total_seconds() == _STEP != 0
    other = inp.with_name("other.inp")
    other.write_text("[OPTIONS]\nREPORT_STEP   00:02:00\n")
    assert inp_report_step_s(other) == 120
    other.write_text("[OPTIONS]\nFLOW_UNITS CMS\n")
    assert inp_report_step_s(other) is None
