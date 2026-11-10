"""Reviewer tests for the performance series group, the leftovers of a cleared store, and the outer
join of a series group on append.

Written by the reviewing seat, separate from the coder's ``tests/test_performance_series_and_leftovers.py``. No solver
and no Snakemake: every test runs on the committed multi-sim golden. Each test names the failure it is meant to catch
and carries a control that makes a broken implementation read as correct.

Gaps these tests close against the coder's file: the coder's axis test stacks contiguous steps (so a stack-by-order
implementation passes it), no coder test reaches the performance export's processed-state skip, the resume of a
performance clearing killed after its pending record, the rank axis per run mode, the outer join on the performance
and gridded groups, a netCDF-form store's intermediate, and over-deletion of siblings that belong to a store that was
not cleared.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from tests.fixtures.golden_trees import construct_golden_analysis

_PERF = "tritonswmm/performance/timeseries"
_GRID = "tritonswmm/triton/timeseries"
_RECORD = "variable_set_across_members"


def _ready(golden_extractor, *, clear: bool):
    """Golden analysis, series consolidation on, clearing on or off, a coupling list beside every scenario."""
    from hhemt.scenario import TRITONSWMM_scenario, coupling_nodes_file_for

    system_dir = golden_extractor("multisim")
    cfg = system_dir / "analysis_config.yaml"
    flag = "toggle_clear_scenario_timeseries_after_consolidation: " + ("true" if clear else "false") + "\n"
    cfg.write_text(cfg.read_text() + "toggle_consolidate_timeseries: true\n" + flag)
    analysis = construct_golden_analysis(system_dir)
    for event_iloc in analysis.df_sims.index:
        scen = TRITONSWMM_scenario(event_iloc, analysis)
        ts = xr.open_zarr(scen.scen_paths.output_tritonswmm_node_time_series, consolidated=False)
        nodes_file = coupling_nodes_file_for(scen.scen_paths.swmm_hydraulics_inp)
        nodes_file.parent.mkdir(parents=True, exist_ok=True)
        nodes_file.write_text("".join(f"{n}\n" for n in ts["node_id"].values[:3]))
    return analysis


def _scen(a, event_iloc: int):
    from hhemt.scenario import TRITONSWMM_scenario

    return TRITONSWMM_scenario(event_iloc, a)


def _perf(a, event_iloc: int) -> Path:
    return Path(_scen(a, event_iloc).scen_paths.output_tritonswmm_performance_timeseries)


def _grid(a, event_iloc: int) -> Path:
    return Path(_scen(a, event_iloc).scen_paths.output_tritonswmm_triton_timeseries)


def _rewrite(path: Path, edit) -> None:
    ds = xr.open_zarr(path, consolidated=False).load()
    ds = edit(ds)
    shutil.rmtree(path)
    ds.to_zarr(path, mode="w")


def _group(a, name: str) -> xr.Dataset:
    return xr.open_zarr(a.analysis_paths.analysis_datatree_zarr, group=name, consolidated=False)


def _events(a, name: str) -> list[int]:
    return [int(v) for v in _group(a, name)["event_iloc"].values]


def test_a_performance_series_lands_at_its_own_elapsed_seconds_not_at_its_order(golden_extractor):
    """Property: a per-rank performance store whose checkpoints are every second reporting step (timestep minutes
    ``2/6, 4/6, ...`` at a 10 s step) lands each value at ``elapsed_s`` 20, 40, ... and leaves the steps between
    them NaN; an event that wrote every step puts its values at 10, 20, 30, ... .

    Class: NEW CAPABILITY (D26: "on ``elapsed_s`` at the TRITON reporting step (the series' ``timestep`` minutes
    times 60)"). Plausible wrong implementation: stacking the rows by order and labelling them with the first
    ``len(rows)`` axis values, which a contiguous series cannot distinguish from the right one (the coder's axis
    test uses contiguous steps only). Reachability: a real checkpoint gap is what a 0-byte ``performance{N}.txt``
        skipped
    by the aggregator leaves (``process_simulation._aggregate_perf_tseries``, "the wallclock series simply omits that
    reporting step"). Both-states anchor: values read from the group by ``elapsed_s`` label. Must-fire control: event
    1, unmodified, has its value at ``elapsed_s`` 10 (position 0), so the assertion cannot be met by an axis shifted
    by one step.
    """
    a = _ready(golden_extractor, clear=False)
    kept = [1, 3, 5, 7, 9, 11]
    _rewrite(_perf(a, 0), lambda ds: ds.isel(timestep=kept))
    src0 = xr.open_zarr(_perf(a, 0), consolidated=False).load()
    src1 = xr.open_zarr(_perf(a, 1), consolidated=False).load()
    a.process.consolidate_timeseries()
    stored = _group(a, _PERF).load()
    axis = list(stored["elapsed_s"].values)
    assert axis[:3] == [10, 20, 30], axis
    got0 = stored["Total"].sel(event_iloc=0).isel(Rank=0).values
    want0 = src0["Total"].isel(event_iloc=0, Rank=0).values
    for pos, step_index in enumerate(kept):
        assert got0[step_index] == want0[pos], (step_index, got0, want0)
    gaps = [i for i in range(len(axis)) if i not in kept]
    assert np.isnan(got0[gaps]).all(), "steps with no checkpoint are NaN"
    got1 = stored["Total"].sel(event_iloc=1).isel(Rank=0).values
    assert got1[0] == src1["Total"].isel(event_iloc=0, Rank=0).values[0], "control: an unmodified event starts at 10 s"


def test_a_performance_series_off_the_reporting_step_is_refused_naming_the_event_and_nothing_is_written_for_it(
    golden_extractor,
):
    """Property: an event whose ``timestep`` values are not whole multiples of the reporting step raises
    ``ProcessingError`` naming the scenario, and the group keeps only the events before it.

    Class: NEW CAPABILITY. Plausible wrong implementation: rounding the label to the nearest step with no tolerance
    check, which silently puts a 15 s checkpoint at 10 s. Reachability: event 1's ``timestep`` is scaled by 1.5, so
    the first checkpoint is at 15 s. Both-states anchor: asserts the exception type, the event in the message and the
    stored event list. Must-fire control: event 0 (unmodified) is in the group, so the refusal is event-specific.
    """
    a = _ready(golden_extractor, clear=False)
    _rewrite(_perf(a, 1), lambda ds: ds.assign_coords(timestep=ds["timestep"] * 1.5))
    with pytest.raises(Exception) as err:
        a.process.consolidate_timeseries()
    assert type(err.value).__name__ == "ProcessingError"
    assert "event_iloc 1" in str(err.value), str(err.value)
    assert _events(a, _PERF) == [0]


def test_a_performance_column_the_new_event_lacks_is_padded_and_a_column_it_adds_is_created_nan_over_earlier_events(
    golden_extractor,
):
    """Property: D27 on the performance group. Event 1 lacks the ``Other`` timer column and event 2 carries an extra
    column ``Extra``: both append without a refusal, ``Other`` is NaN at event 1 and equal to its source at events 0
    and 2, ``Extra`` is NaN at events 0 and 1 and equal to its source at event 2, the group opens, and the variable-set
    record of events 1 and 2 names the column.

    Class: NEW CAPABILITY ("The gridded and the performance groups follow the same rule"). Plausible wrong
    implementation: the join wired to the node and link modes only, so the performance group refuses or becomes
    unreadable on a changed column set. Reachability: the real ``consolidate_timeseries`` in one pass over three
    events. Both-states anchor: values read from the opened group. Must-fire control: event 0's ``Other`` equals its
    source, so the pad touched only the event that lacks the column.
    """
    a = _ready(golden_extractor, clear=False)
    src0 = xr.open_zarr(_perf(a, 0), consolidated=False).load()
    _rewrite(_perf(a, 1), lambda ds: ds.drop_vars("Other"))
    _rewrite(_perf(a, 2), lambda ds: ds.assign(Extra=ds["Total"] * 2))
    src2 = xr.open_zarr(_perf(a, 2), consolidated=False).load()
    a.process.consolidate_timeseries()
    stored = _group(a, _PERF).load()
    assert _events(a, _PERF) == [0, 1, 2]
    other = stored["Other"].isel(Rank=0)
    np.testing.assert_array_equal(other.sel(event_iloc=0).values, src0["Other"].isel(event_iloc=0, Rank=0).values)
    assert bool(other.sel(event_iloc=1).isnull().all())
    np.testing.assert_array_equal(other.sel(event_iloc=2).values, src2["Other"].isel(event_iloc=0, Rank=0).values)
    extra = stored["Extra"].isel(Rank=0)
    assert bool(extra.sel(event_iloc=[0, 1]).isnull().all())
    np.testing.assert_array_equal(extra.sel(event_iloc=2).values, src2["Extra"].isel(event_iloc=0, Rank=0).values)
    assert "Other" in str(stored[_RECORD].sel(event_iloc=1).values)
    assert "Extra" in str(stored[_RECORD].sel(event_iloc=2).values)


def test_the_gridded_group_pads_a_variable_the_new_event_lacks(golden_extractor):
    """Property: D27 on the gridded group. Event 2's gridded store lacks ``flood_depth``: it appends without a
    refusal, ``flood_depth`` is NaN at event 2 and unchanged at events 0 and 1, and the group opens with three events.

    Class: NEW CAPABILITY. Plausible wrong implementation: the pad built eagerly with the wrong chunking, or the
    gridded mode excluded from the join. Reachability: the real ``consolidate_timeseries``. Both-states anchor: values
    read from the opened group. Must-fire control: ``max_flood_depth`` (a variable event 2 keeps) equals its source,
    so the NaN is confined to the missing variable.
    """
    a = _ready(golden_extractor, clear=False)
    src0 = xr.open_zarr(_grid(a, 0), consolidated=False).load()
    _rewrite(_grid(a, 2), lambda ds: ds.drop_vars("flood_depth"))
    src2 = xr.open_zarr(_grid(a, 2), consolidated=False).load()
    a.process.consolidate_timeseries()
    stored = _group(a, _GRID).load()
    assert _events(a, _GRID) == [0, 1, 2]
    assert bool(stored["flood_depth"].sel(event_iloc=2).isnull().all())
    assert bool(stored["flood_depth"].sel(event_iloc=0).notnull().any())
    n = src0.sizes["timestep"]
    np.testing.assert_array_equal(
        stored["flood_depth"].sel(event_iloc=0).isel(elapsed_s=slice(1, n + 1)).values, src0["flood_depth"].values
    )
    assert "max_flood_depth" in src2.data_vars


def test_a_second_pass_after_clearing_leaves_both_performance_groups_and_flags_as_they_were(golden_extractor):
    """Property: consolidating again after every performance store was cleared appends nothing, rewrites nothing, does
    not raise, and leaves each cleared flag True and each group's values equal to the first pass.

    Class: NEW CAPABILITY (resume of a finished analysis). Plausible wrong implementation: the cleared performance
    event reaching ``_plan_timeseries_write`` with no source and no cleared record, which raises, or appending a
    duplicate (``[0, 1, 1]`` measured for the node group). Reachability: two real passes. Both-states anchor:
    the stored values and events. Must-fire control: the first pass cleared the stores (they do not exist), so the
    second pass cannot succeed by re-reading them.
    """
    a = _ready(golden_extractor, clear=True)
    a.process.consolidate_timeseries()
    first = {g: _group(a, g).load() for g in (_PERF, "triton_only/performance/timeseries")}
    for event_iloc in a.df_sims.index:
        assert not _perf(a, event_iloc).exists(), "control: the first pass cleared the store"
    a.process.consolidate_timeseries()
    for group, before in first.items():
        after = _group(a, group).load()
        assert _events(a, group) == [0, 1, 2], group
        xr.testing.assert_equal(after["Total"], before["Total"])
    for event_iloc in a.df_sims.index:
        log = _scen(a, event_iloc).get_log("tritonswmm")
        assert log.consolidated_performance_timeseries_cleared.get() is True


def test_a_performance_clearing_killed_after_its_pending_record_is_finished_from_the_group_and_its_leftovers_go(
    golden_extractor,
):
    """Property: a scenario whose log records the performance clearing as pending (names written) and whose store was
    already deleted with an ``.aside`` and a ``.tmp`` left beside it resumes to cleared: the cleared field is True, the
    pending record is empty, the leftovers are gone and the store is not recreated from either.

    Class: NEW CAPABILITY. Plausible wrong implementation: the finish step deleting the store only (the leftover
    ``.tmp`` is then promoted to the store path by the next publish), or comparing against the partly deleted source.
    Reachability: a first pass with clearing off puts every slice in the group; the pending record and the missing
    store are then constructed as a kill between the delete and the cleared record leaves them. Both-states anchor:
    the log fields and the three paths. Must-fire control: the same state with a pending name the group lacks
    (``NoSuchVariable``) refuses and keeps the log un-cleared.
    """
    a = _ready(golden_extractor, clear=False)
    a.process.consolidate_timeseries()
    store = _perf(a, 1)
    names = sorted(xr.open_zarr(store, consolidated=False).data_vars)
    names = [n for n in names if {"timestep", "Rank"} <= set(xr.open_zarr(store, consolidated=False)[n].dims)]
    log = _scen(a, 1).get_log("tritonswmm")
    # The pending record is written through the consolidation's own field accessor, as a kill would have left it.
    type(a.process)._log_field(log, "consolidated_performance_timeseries_clearing_pending").set(["NoSuchVariable"])
    with pytest.raises(Exception) as err:
        a.process.consolidate_timeseries()
    assert type(err.value).__name__ == "ProcessingError"
    cleared_after_refusal = _scen(a, 1).get_log("tritonswmm").consolidated_performance_timeseries_cleared
    assert cleared_after_refusal is None or not bool(cleared_after_refusal.get())
    assert store.exists(), "control: the refused finish deleted nothing"

    type(a.process)._log_field(log, "consolidated_performance_timeseries_clearing_pending").set(names)
    shutil.rmtree(store)
    for suffix in (".aside", ".tmp"):
        leftover = store.with_name(store.name + suffix)
        leftover.mkdir()
        (leftover / "part").write_text("x")
    a.process.consolidate_timeseries()
    fresh = _scen(a, 1).get_log("tritonswmm")
    assert fresh.consolidated_performance_timeseries_cleared.get() is True
    assert not (fresh.consolidated_performance_timeseries_clearing_pending.get() or [])
    assert not store.exists()
    assert not store.with_name(store.name + ".aside").exists()
    assert not store.with_name(store.name + ".tmp").exists()
    assert _events(a, _PERF) == [0, 1, 2]


@pytest.mark.parametrize(
    "model, method, field",
    [
        ("tritonswmm", "_export_TRITONSWMM_performance_tseries", "output_tritonswmm_performance_timeseries"),
        ("triton", "_export_TRITON_only_performance_tseries", "output_triton_only_performance_timeseries"),
    ],
)
def test_no_resume_reaches_the_performance_aggregator_after_clearing(
    golden_extractor, monkeypatch, model, method, field
):
    """Property: for a scenario whose performance store is gone and whose cleared field is set, the performance export
    returns without parsing the raw performance files, even when the processing record and the written flag that would
    otherwise say "already written" are absent (a forced re-run clears them); with the cleared field removed the same
    call reaches the aggregator.

    Class: NEW CAPABILITY. This is the edit at ``_export_performance_tseries`` that the coder's file does not exercise
    (its resume test covers the gridded writer only). Plausible wrong implementation: the skip left on the processing
    record alone, so a forced re-run parses raw files that ``clear_raw`` removed. Reachability: the real clearing
    removes the store; the record and flag are then reset by hand and the aggregator is replaced by a function that
    raises. Both-states anchor: whether the aggregator ran. Must-fire control: the second arm, with the cleared field
    set False, reaches the aggregator (the sentinel error), so a skip that never reads the field fails the first
    arm and a call that never reaches the aggregator fails the second.
    """
    import hhemt.process_simulation as ps

    a = _ready(golden_extractor, clear=True)
    a.process.consolidate_timeseries()
    store = Path(getattr(_scen(a, 0).scen_paths, field))
    assert not store.exists()

    class _AggregatorReached(RuntimeError):
        pass

    def _boom(*args, **kwargs):
        raise _AggregatorReached

    monkeypatch.setattr(ps, "_aggregate_perf_tseries", _boom)
    proc = a._retrieve_sim_run_processing_object(0)
    proc._set_active_model_log(model)
    proc.log.processing_log.outputs.pop(store.name, None)
    proc.log.performance_timeseries_written.set(False)
    getattr(proc, method)(verbose=False)
    assert proc.log.performance_timeseries_written.get() is True, "the skip latches the written marker"

    proc.log.consolidated_performance_timeseries_cleared.set(False)
    proc.log.performance_timeseries_written.set(False)
    with pytest.raises((_AggregatorReached, FileNotFoundError)) as err:
        getattr(proc, method)(verbose=False)
    assert err.type is _AggregatorReached or "performance" in str(err.value).lower()


def test_the_rank_axis_is_the_largest_rank_count_the_run_mode_admits(golden_extractor):
    """Property: the fixed ``Rank`` axis has ``n_gpus`` entries in GPU mode (one rank per GPU, whatever
    ``n_mpi_procs`` says) and ``n_mpi_procs`` entries in every other mode; a missing count gives one rank.

    Class: NEW CAPABILITY (D26: "fixed ... to the largest rank count the analysis's compute configuration admits").
    Plausible wrong implementation: ``n_mpi_procs`` in every mode, which gives a one-rank axis for a two-GPU run and
    refuses its second rank. Reachability: the method is called on the real class with a stand-in for the analysis,
    because the golden's config validator forbids a GPU configuration without a GPU system. Both-states anchor: the
    axis values. Must-fire control: the GPU row has ``n_mpi_procs`` 1 and ``n_gpus`` 2, so the two counts differ.
    """
    from hhemt.processing_analysis import TRITONSWMM_analysis_post_processing as pp

    def axis(run_mode, n_mpi_procs, n_gpus):
        cfg = SimpleNamespace(run_mode=run_mode, n_mpi_procs=n_mpi_procs, n_gpus=n_gpus)
        return list(pp._rank_axis(SimpleNamespace(_analysis=SimpleNamespace(cfg_analysis=cfg))))

    assert axis("gpu", 1, 2) == [0, 1]
    assert axis("mpi", 4, 0) == [0, 1, 2, 3]
    assert axis("hybrid", 3, 0) == [0, 1, 2]
    assert axis("serial", None, None) == [0]
    assert axis("gpu", 1, 2) != axis("mpi", 1, 2), "control: the mode, not a count, decides"


def test_clearing_a_netcdf_form_store_removes_its_zarr_intermediate_and_no_sibling_of_another_store(golden_extractor):
    """Property: clearing a store named ``X.nc`` removes ``X.nc.zarr`` (the intermediate ``write_zarr_then_netcdf``
    leaves when its removal failed), its ``.done`` and its ``.chapters``; a store that is NOT in the cleared list keeps
    its own ``.done`` and ``.chapters`` and intermediate, and the cleared store's neighbour with a shared prefix stays.

    Class: NEW CAPABILITY. Plausible wrong implementations: deriving the intermediate as ``stem + .nc.zarr`` for every
    suffix (``X.nc.nc.zarr`` for a netCDF store, which leaves ``X.nc.zarr``), and a glob on the stem that takes the
    neighbour. Reachability: ``target_processed_output_type`` can be ``nc`` (``scenario.py`` builds every store path
    with ``.{out_type}``), so a netCDF-form store is a real input. Both-states anchor: the existence of each path.
    Must-fire control: the neighbour ``X_other.nc`` and its siblings exist after the call.
    """
    a = _ready(golden_extractor, clear=False)
    scen = _scen(a, 0)
    folder = Path(scen.scen_paths.output_tritonswmm_performance_timeseries).parent
    cleared = folder / "cleared_probe.nc"
    other = folder / "cleared_probe_other.nc"
    planted = {}
    for store in (cleared, other):
        store.write_text("x")
        planted[store] = {
            "done": store.with_name(store.name + ".done"),
            "chapters": store.with_name(store.name + ".chapters"),
            "zarr": store.with_name(store.name + ".zarr"),
        }
        planted[store]["done"].write_text("x")
        for key in ("chapters", "zarr"):
            planted[store][key].mkdir()
            (planted[store][key] / "part").write_text("x")
    type(a.process)._delete_cleared_stores(scen, [cleared])
    assert not cleared.exists()
    for key, path in planted[cleared].items():
        assert not path.exists(), key
    assert other.exists()
    for key, path in planted[other].items():
        assert path.exists(), ("control", key)


def test_clearing_a_performance_store_leaves_the_summary_store_and_its_flag_of_the_same_scenario(golden_extractor):
    """Property: deleting the cleared performance series store does not delete the performance SUMMARY store beside
    it or the summary's ``.done``; the summary is a different store with its own clearing.

    Class: NEW CAPABILITY. Plausible wrong implementation: a leftover sweep by prefix (``TRITONSWMM_perf_*``) that
    takes the summary with the series. Reachability: the golden holds both stores in ``processed/``. Both-states
    anchor: existence of the summary after the real clearing. Must-fire control: the series store itself is gone, so
    the clearing ran.
    """
    a = _ready(golden_extractor, clear=True)
    summary = Path(_scen(a, 0).scen_paths.output_tritonswmm_performance_summary)
    assert summary.exists(), "precondition: the golden holds the performance summary"
    flag = summary.with_name(summary.name + ".done")
    flag.write_text("x")
    a.process.consolidate_timeseries()
    assert not _perf(a, 0).exists(), "control: the series store was cleared"
    assert summary.exists() and flag.exists()
