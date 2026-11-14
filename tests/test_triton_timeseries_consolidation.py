"""The gridded TRITON series are consolidated into ``tritonswmm/triton/timeseries`` and cleared like the SWMM series.

No solver and no Snakemake: every test runs on the committed multi-sim golden (a
finished analysis tree written by the independent generator, whose per-scenario ``TRITONSWMM_TRITON_tseries.zarr``
holds four ``(timestep, y, x)`` variables on a 10 s step) with a gridded store rewritten where an arm needs a different
length or time origin. Each arm names its control: the input class that makes a broken implementation read as correct.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from tests.fixtures.golden_trees import construct_golden_analysis

_TRI = "tritonswmm/triton/timeseries"
_NODE = "tritonswmm/swmm_node/timeseries"
_LINK = "tritonswmm/swmm_link/timeseries"
_CLEARED = "consolidated_triton_timeseries_cleared"
_PENDING = "consolidated_triton_timeseries_clearing_pending"
_GRIDDED_VARS = ["flood_depth", "max_flood_depth", "unit_discharge_x", "unit_discharge_y"]


def _ready(golden_extractor, *, clear: bool, extra: str = ""):
    """Golden analysis with series consolidation on, clearing on or off, and a coupling list beside every scenario."""
    from hhemt.scenario import TRITONSWMM_scenario, coupling_nodes_file_for

    system_dir = golden_extractor("multisim")
    cfg = system_dir / "analysis_config.yaml"
    flag = "toggle_clear_scenario_timeseries_after_consolidation: " + ("true" if clear else "false") + "\n"
    cfg.write_text(cfg.read_text() + "toggle_consolidate_timeseries: true\n" + flag + extra)
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


def _own(a, event_iloc: int) -> Path:
    return Path(_scen(a, event_iloc).scen_paths.output_tritonswmm_triton_timeseries)


def _rewrite(path: Path, edit) -> None:
    """Replace the zarr store at ``path`` with ``edit(its dataset)``."""
    ds = xr.open_zarr(path, consolidated=False).load()
    ds = edit(ds)
    shutil.rmtree(path)
    ds.to_zarr(path, mode="w")


def _group(a, name: str = _TRI) -> xr.Dataset:
    return xr.open_zarr(a.analysis_paths.analysis_datatree_zarr, group=name, consolidated=False)


def _events(a, name: str = _TRI) -> list[int]:
    return [int(v) for v in _group(a, name)["event_iloc"].values]


def _flag(a, event_iloc: int, name: str):
    field = getattr(_scen(a, event_iloc).get_log("tritonswmm"), name, None)
    return None if field is None else field.get()


def _store_files(group_dir: Path) -> dict[str, bytes]:
    return {str(p.relative_to(group_dir)): p.read_bytes() for p in sorted(group_dir.rglob("*")) if p.is_file()}


def _only_gridded(monkeypatch) -> None:
    """Restrict the consolidation to the gridded mode, so that a test refusing a full ``.load()`` does not trip over
    the SWMM slice builder, which loads its small slice by design."""
    import hhemt.processing_analysis as pa

    cls = pa.TRITONSWMM_analysis_post_processing
    monkeypatch.setattr(
        cls, "_TIMESERIES_MODE_CONFIG", {"tritonswmm_triton_ts": cls._TIMESERIES_MODE_CONFIG["tritonswmm_triton_ts"]}
    )


def test_two_gridded_stores_of_different_length_and_origin_land_on_one_axis_and_a_misplaced_one_is_refused(
    golden_extractor,
):
    """Property: a gridded store that starts at ``timestep`` 0 and one that starts at one step, of different lengths,
    land in ``tritonswmm/triton/timeseries`` on ONE ``elapsed_s`` axis at the TRITON step that includes the instant
    0, each at its own seconds, the shorter NaN-padded; a store whose times are not whole multiples of the step is
    refused naming the scenario.

    Class: NEW CAPABILITY. Plausible wrong implementations: an axis that starts at one step (the origin-0 store is
    then refused or shifted by a step), an axis taken from the first store written (the next store of another
    length is refused), or minutes read as seconds. Reachability: the golden's three events consolidate through the
    real ``consolidate_timeseries``. Both-states anchor: event 0 carries the sentinel 7.0 at its instant 0 and its
    source rows after it; event 1 carries its first eight source rows then NaN. Invocation isolation: nothing here
    depends on the working directory. Control: event 2 with every time shifted by 0.01 min is refused, and its
    scenario slug is in the message.
    Killed by: the three above. Passes under a second correct implementation that writes each event by a region
    write into a group created whole from the axis; the axis and the values are the same.
    """
    a = _ready(golden_extractor, clear=False)
    original = {i: xr.open_zarr(_own(a, i), consolidated=False).load() for i in a.df_sims.index}
    step = float(a.cfg_analysis.TRITON_reporting_timestep_s)

    def with_initial_state(ds):
        first = xr.full_like(ds.isel(timestep=[0]), 7.0).assign_coords(timestep=[0.0])
        return xr.concat([first, ds], dim="timestep")

    _rewrite(_own(a, 0), with_initial_state)
    _rewrite(_own(a, 1), lambda ds: ds.isel(timestep=slice(0, 8)))
    a.process.consolidate_timeseries()
    stored = _group(a).load()
    axis = stored["elapsed_s"].values
    assert axis[0] == 0 and np.allclose(np.diff(axis), step), axis
    assert len(axis) > original[0].sizes["timestep"], "control: the axis is longer than the longest store"
    depth = stored["flood_depth"]
    assert (depth.sel(event_iloc=0).isel(elapsed_s=0).values == 7.0).all()
    n0 = original[0].sizes["timestep"]
    np.testing.assert_array_equal(
        depth.sel(event_iloc=0).isel(elapsed_s=slice(1, 1 + n0)).values, original[0]["flood_depth"].values
    )
    np.testing.assert_array_equal(
        depth.sel(event_iloc=1).isel(elapsed_s=slice(1, 9)).values, original[1]["flood_depth"].values[:8]
    )
    assert bool(depth.sel(event_iloc=1).isel(elapsed_s=slice(9, None)).isnull().all()), "event 1 must be NaN-padded"
    assert bool(depth.sel(event_iloc=1).isel(elapsed_s=0).isnull().all()), (
        "a store without an initial state is NaN at 0"
    )


def test_a_gridded_store_whose_times_are_not_whole_steps_is_refused_naming_the_scenario(golden_extractor):
    """Property (control of the axis arm): a gridded store whose times are shifted off the reporting step is refused
    with ``ProcessingError`` and the message carries the scenario's slug.

    Class: NEW CAPABILITY. Plausible wrong implementation: rounding every time to the nearest step, which places a
    misaligned store on the axis silently. Reachability: event 2 of the golden with every ``timestep`` + 0.01 min
    (0.6 s) goes through the real ``consolidate_timeseries``. Both-states anchor: the unshifted golden consolidates
    in the arm above. Killed by: that rounding. Passes under a second correct implementation that raises from the
    reindex instead, provided the scenario is named.
    """
    b = _ready(golden_extractor, clear=False)
    _rewrite(_own(b, 2), lambda ds: ds.assign_coords(timestep=ds["timestep"] + 0.01))
    with pytest.raises(Exception) as err:
        b.process.consolidate_timeseries()
    assert type(err.value).__name__ == "ProcessingError"
    assert b.process._event_slug(2) in str(err.value), "the refusal must name the scenario"


def test_the_gridded_slice_is_written_and_read_back_in_blocks_and_one_changed_block_keeps_the_store(
    golden_extractor, monkeypatch
):
    """Property: with every explicit load of a Dataset or DataArray refused, the gridded consolidation still writes
    and verifies the slice, no array larger than one block is ever materialized, and a single value changed in a
    LATE block after the write makes the read-back fail and keeps the scenario's store.

    Class: NEW CAPABILITY. Plausible wrong implementations: the landed whole-slice ``.load()`` (refused here), a
    read-back that realizes the whole slice with ``.values`` (the largest materialization then exceeds one block),
    and a read-back that compares only the first block (the changed value sits in the last). Reachability: the
    block size is shrunk to 4 steps through ``_TIMESERIES_BLOCK_BYTES`` so the golden's slice spans several
    blocks; the real ``consolidate_timeseries`` runs. Both-states anchor: the same consolidation without the
    damage completes and leaves the store cleared. Must-fire control: the damaged run raises ``ProcessingError``
    and the store exists.
    Killed by: the three above. Passes under a second correct implementation that compares by zarr region reads
    through another reader, provided its blocks are no larger than the stated size.
    """
    import dask.array.core as dcore

    import hhemt.processing_analysis as pa

    cls = pa.TRITONSWMM_analysis_post_processing
    _only_gridded(monkeypatch)
    monkeypatch.setattr(cls, "_TIMESERIES_BLOCK_BYTES", 400)
    a = _ready(golden_extractor, clear=True)

    def refuse(self, *args, **kwargs):
        raise AssertionError("the gridded consolidation loaded a whole slice")

    seen: list[int] = []
    real_array = dcore.Array.__array__

    def spy(self, *args, **kwargs):
        seen.append(int(self.size))
        return real_array(self, *args, **kwargs)

    with monkeypatch.context() as patched:
        for owner in (xr.Dataset, xr.DataArray):
            for method in ("load", "compute"):
                patched.setattr(owner, method, refuse)
        patched.setattr(dcore.Array, "__array__", spy)
        a.process.consolidate_timeseries()
    assert not _own(a, 0).exists(), "control: the clean run read back and cleared the store"
    axis_len = _group(a).sizes["elapsed_s"]
    cells = _group(a).sizes["y"] * _group(a).sizes["x"]
    assert seen, "control: the read-back realized something, so the spy observed it"
    assert max(seen) <= 4 * cells < axis_len * cells, (max(seen), 4 * cells, axis_len * cells)


def test_one_value_changed_in_a_late_block_after_the_write_fails_the_readback_and_keeps_the_store(
    golden_extractor, monkeypatch
):
    """Property (must-fire control of the block-wise read-back): one value changed in the LAST block of the stored
    gridded slice after the write makes the read-back fail with ``ProcessingError`` and keeps the scenario's store.

    Class: NEW CAPABILITY. Plausible wrong implementation: a read-back that compares only the first block (or
    samples blocks). Reachability: ``append_zarr`` is wrapped to overwrite one stored value through the zarr API
    right after the real append, with the block size shrunk to 4 steps. Both-states anchor: the clean run of the arm
    above clears the same store. Killed by: first-block-only comparison. Passes under a second correct
    implementation that compares every block through a different reader.
    """
    import zarr

    import hhemt.processing_analysis as pa

    cls = pa.TRITONSWMM_analysis_post_processing
    _only_gridded(monkeypatch)
    monkeypatch.setattr(cls, "_TIMESERIES_BLOCK_BYTES", 400)
    b = _ready(golden_extractor, clear=True)
    real_append = pa.append_zarr

    def damaging_append(ds, store, **kw):
        """Append as the real function does, then change one value in the LAST block of the stored flood depth."""
        real_append(ds, store, **kw)
        if kw.get("group") == _TRI and kw.get("append_dim") == "event_iloc":
            arr = zarr.open_group(store, path=_TRI, mode="r+")["flood_depth"]
            arr[arr.shape[0] - 1, 10, ...] = 999.0

    monkeypatch.setattr(pa, "append_zarr", damaging_append)
    with pytest.raises(Exception) as err:
        b.process.consolidate_timeseries()
    assert type(err.value).__name__ == "ProcessingError"
    assert _own(b, 1).exists(), "the store of the event whose slice was damaged must be kept"


def test_the_gridded_store_is_cleared_only_after_every_block_passes_and_a_failed_readback_keeps_it(
    golden_extractor, monkeypatch
):
    """Property: with clearing on, a scenario's gridded store is deleted and ``consolidated_triton_timeseries_cleared``
    set only after its slice passes the read-back; an event whose gridded read-back fails keeps its gridded store,
    its node and link stores, and has the flag unset, while the event before it is cleared.

    Class: NEW CAPABILITY. Plausible wrong implementation: deleting the gridded store as soon as it is appended,
    before the read-back, or clearing the SWMM stores while the gridded slice is unverified (the clearing is one act
    per scenario). Reachability: the real consolidation with the read-back replaced for one group of one event.
    Both-states anchor: event 0 is cleared with the flag True, event 1 is kept with the flag not True.
    Killed by: those. Passes under a second correct implementation that clears the three stores in either order.
    """
    import hhemt.processing_analysis as pa

    cls = pa.TRITONSWMM_analysis_post_processing
    a = _ready(golden_extractor, clear=True)
    real = cls._stored_slice_equals
    monkeypatch.setattr(
        cls,
        "_stored_slice_equals",
        staticmethod(lambda s, g, e, i: False if (i == 1 and g == _TRI) else real(s, g, e, i)),
    )
    with pytest.raises(Exception) as err:
        a.process.consolidate_timeseries()
    assert type(err.value).__name__ == "ProcessingError"
    assert _own(a, 1).exists(), "the gridded store was deleted over an unverified slice"
    assert Path(_scen(a, 1).scen_paths.output_tritonswmm_node_time_series).exists()
    assert _flag(a, 1, _CLEARED) is not True
    assert not _own(a, 0).exists() and _flag(a, 0, _CLEARED) is True, "control: event 0 passed and was cleared"
    assert _flag(a, 0, _PENDING) is None, "a finished clearing leaves no pending record"


def test_the_keep_list_prune_precedes_the_gridded_consolidation(golden_extractor):
    """Property: with ``timeseries_keep_triton_gridded: ["flood_depth"]`` and the real prune run, the gridded branch
    holds ``flood_depth`` only; with the field absent it holds every variable of the store that carries ``timestep``.

    Class: NEW CAPABILITY. Plausible wrong implementation: consolidating before the prune so the branch keeps the
    unkept series, or a slice builder that keeps a fixed variable list. Reachability: the config-driven
    ``_prune_timeseries_stores`` of each scenario, then the real consolidation. Control: the next arm.
    Killed by: those. Passes under a second correct implementation that applies the prune inside the consolidation.
    """
    a = _ready(golden_extractor, clear=False, extra="timeseries_keep_triton_gridded: [flood_depth]\n")
    for i in a.df_sims.index:
        a._retrieve_sim_run_processing_object(i)._prune_timeseries_stores("tritonswmm", "TRITON")
    a.process.consolidate_timeseries()
    assert sorted(_group(a).data_vars) == ["flood_depth"]


def test_with_no_keep_field_the_gridded_branch_holds_every_variable_of_the_store(golden_extractor):
    """Property (control of the keep-list arm): with no ``timeseries_keep_triton_gridded`` the gridded branch holds
    every variable of the store that carries ``timestep``.

    Class: NEW CAPABILITY. Plausible wrong implementation: a slice builder that keeps a fixed variable list, which
    would make the arm above pass for the wrong reason. Reachability: the real consolidation of the unpruned golden.
    Killed by: that. Passes under a second correct implementation that names the variables from the first store.
    """
    b = _ready(golden_extractor, clear=False)
    b.process.consolidate_timeseries()
    assert sorted(_group(b).data_vars) == sorted(_GRIDDED_VARS)


def test_a_triton_summary_rebuild_from_a_cleared_series_refuses_and_the_same_scenario_rebuilds_before_clearing(
    golden_extractor,
):
    """Property: a scenario whose gridded series was cleared and whose summary store is gone raises
    ``ProcessingError`` naming the cleared store and ``tritonswmm/triton/timeseries``; the same scenario before
    clearing rebuilds the summary; and the processed-state property reads the cleared record.

    Class: NEW CAPABILITY. Plausible wrong implementation: letting the missing series fall through to
    ``FileNotFoundError``, whose text says the processing failed when it did not. Reachability:
    ``write_summary_outputs`` of the real processing object. Control: the unconsolidated twin rebuilds.
    Killed by: that. Passes under a second correct implementation that raises ``ProcessingError`` earlier in the
    export, before the usable-summary test.
    """
    a = _ready(golden_extractor, clear=True)
    a.process.consolidate_timeseries()
    summary = Path(_scen(a, 0).scen_paths.output_tritonswmm_triton_summary)
    shutil.rmtree(summary)
    proc = a._retrieve_sim_run_processing_object(0)
    with pytest.raises(Exception) as err:
        proc.write_summary_outputs(which="TRITON", model_type="tritonswmm")
    assert type(err.value).__name__ == "ProcessingError"
    assert "tritonswmm/triton/timeseries" in str(err.value) and "TRITONSWMM_TRITON_tseries" in str(err.value)
    proc._set_active_model_log("tritonswmm")
    proc.log.TRITON_timeseries_written.set(False)
    assert proc.TRITON_outputs_processed is True, "the cleared record counts as processed"


def test_before_clearing_the_same_scenario_rebuilds_its_triton_summary(golden_extractor):
    """Property (control of the refusal arm): the unconsolidated twin rebuilds the summary from its series, and with
    no cleared record the written flag decides ``TRITON_outputs_processed``.

    Class: NEW CAPABILITY. Plausible wrong implementation: a refusal keyed on the toggle rather than on the record,
    which would refuse this rebuild too. Reachability: ``write_summary_outputs`` of the real processing object.
    Killed by: that. Passes under a second correct implementation that checks the record in the processing log.
    """
    b = _ready(golden_extractor, clear=False)
    summary_b = Path(_scen(b, 0).scen_paths.output_tritonswmm_triton_summary)
    shutil.rmtree(summary_b)
    b._retrieve_sim_run_processing_object(0).write_summary_outputs(which="TRITON", model_type="tritonswmm")
    assert summary_b.exists(), "control: before clearing the summary rebuilds from the series"
    proc_b = b._retrieve_sim_run_processing_object(0)
    proc_b._set_active_model_log("tritonswmm")
    proc_b.log.TRITON_timeseries_written.set(False)
    assert proc_b.TRITON_outputs_processed is False, "control: no cleared record, so the written flag decides"


def test_the_data_availability_check_reports_a_cleared_gridded_store_naming_the_branch(golden_extractor):
    """Property: after the consolidation clears the gridded stores, ``check_data_availability`` passes and its
    summary names ``tritonswmm/triton/timeseries`` as the branch holding the copy.

    Class: NEW CAPABILITY. Plausible wrong implementation: no entry for the new log field, so a cleared gridded store
    is silent. Reachability: the real check over the real consolidation. Control: the unrecorded deletion
    reads "No per-scenario artifacts were reclaimed".
    Killed by: that. Passes under a second correct implementation that derives the label from the branch map.
    """
    from hhemt.analysis_validation import check_data_availability

    a = _ready(golden_extractor, clear=True)
    a.process.consolidate_timeseries()
    result = check_data_availability(a)
    assert result.passed and "tritonswmm/triton/timeseries" in result.summary, result.summary


def test_a_gridded_store_removed_by_hand_with_no_record_is_not_reported_as_cleared(golden_extractor):
    """Property (control of the availability arm): a gridded store removed with no record is not claimed as cleared.

    Class: NEW CAPABILITY. Plausible wrong implementation: reporting any absent gridded store as cleared. Reachability:
    the real check over a consolidation with clearing off and one store deleted by hand. Killed by: that. Passes
    under a second correct implementation that also lists the absent store as missing, provided it does not call it
    cleared.
    """
    from hhemt.analysis_validation import check_data_availability

    b = _ready(golden_extractor, clear=False)
    b.process.consolidate_timeseries()
    shutil.rmtree(_own(b, 0))
    missing = check_data_availability(b)
    assert "tritonswmm/triton/timeseries" not in missing.summary
    assert "No per-scenario artifacts were reclaimed" in missing.summary, missing.summary


def test_a_clearing_killed_after_the_delete_resumes_to_done_when_the_slice_is_complete_and_refuses_when_it_is_not(
    golden_extractor, monkeypatch
):
    """Property: the clearing is recorded as pending, with the verified variable names, BEFORE the store is deleted;
    a kill after the delete and before ``done`` resumes to ``done`` when the event's slice is complete in its group;
    the same state with a variable missing from the group refuses naming the scenario and marks nothing done.

    Class: NEW CAPABILITY. Plausible wrong implementations: deleting first and recording after (the next pass finds
    no store and no record and refuses the scenario for good), and a resume that accepts a group missing a recorded
    variable. Reachability: the real ``delete_and_account`` is wrapped to read the on-disk log at the moment of the
    delete and then to raise as a kill would. Both-states anchor: the pending record read from disk during the
    delete equals the four verified names. Control: dropping ``flood_depth`` from the group makes the resume refuse.
    Killed by: both. Passes under a second correct implementation that records the names as a companion string
    instead of a list.
    """
    import hhemt.du_sentinels as du

    a = _ready(golden_extractor, clear=True)
    real = du.delete_and_account
    log_file = Path(_scen(a, 1).scen_paths.sim_folder) / "log_tritonswmm.json"
    observed: dict[str, object] = {}

    def killed_after_delete(paths, **kw):
        """Record what the log said at the delete, delete as the real tool does, then raise."""
        if any("TRITONSWMM_TRITON_tseries" in str(p) for p in paths) and "event_index.1" in str(paths[0]):
            observed["at_delete"] = json.loads(log_file.read_text()).get(_PENDING)
            real(paths, **kw)
            raise RuntimeError("killed after the delete")
        return real(paths, **kw)

    monkeypatch.setattr(du, "delete_and_account", killed_after_delete)
    with pytest.raises(RuntimeError, match="killed after the delete"):
        a.process.consolidate_timeseries()
    assert sorted(observed["at_delete"]) == sorted(_GRIDDED_VARS), observed
    assert not _own(a, 1).exists() and _flag(a, 1, _CLEARED) is not True
    monkeypatch.undo()
    a.process.consolidate_timeseries()
    assert _flag(a, 1, _CLEARED) is True and _flag(a, 1, _PENDING) is None


def test_a_recorded_clearing_whose_slice_lacks_a_variable_refuses_naming_the_scenario(golden_extractor):
    """Property (control of the resume arm): the same pending state with ``flood_depth`` missing from the group
    refuses with ``ProcessingError`` naming the scenario and marks nothing done.

    Class: NEW CAPABILITY. Plausible wrong implementation: a resume that checks only that the event is present in
    the group, which would mark a clearing done over a slice that lost a variable. Reachability: the pending record
    is written with the real field helper and the variable's array directory is removed from the group. Killed by:
    the event-only check. Passes under a second correct implementation that compares against the group's variables
    recorded elsewhere, provided the refusal names the scenario.
    """
    b = _ready(golden_extractor, clear=True)
    b.process.consolidate_timeseries()
    log = _scen(b, 1).get_log("tritonswmm")
    b.process._log_field(log, _CLEARED).set(False)
    b.process._log_field(log, _PENDING).set(list(_GRIDDED_VARS))
    shutil.rmtree(b.analysis_paths.analysis_datatree_zarr / _TRI / "flood_depth")
    with pytest.raises(Exception) as err:
        b.process.consolidate_timeseries()
    assert type(err.value).__name__ == "ProcessingError"
    assert b.process._event_slug(1) in str(err.value)
    assert _flag(b, 1, _CLEARED) is not True


def test_a_half_deleted_source_with_the_record_pending_is_finished_from_the_store_and_never_read_back(
    golden_extractor,
):
    """Property: with half of a scenario's gridded chunk files deleted and the clearing recorded pending, the resume
    leaves the group byte-identical, removes the rest of the store and sets ``done``; the same half-deleted store
    with NO record is never read back as verified.

    Class: NEW CAPABILITY. Plausible wrong implementation: a resume that compares the half-deleted source with the
    stored slice (zarr reads a missing chunk as fill value, so the comparison fails and the slice is then rewritten
    from the truncated source). Reachability: the real consolidation twice. Both-states anchor: every file of the
    group is compared before and after. Control: the plan for the same store without a record returns ``rewrite`` or
    raises, and never ``verified``.
    Killed by: that. Passes under a second correct implementation that checks the slice by a different reader.
    """
    a = _ready(golden_extractor, clear=False)
    a.process.consolidate_timeseries()
    group_dir = a.analysis_paths.analysis_datatree_zarr / _TRI
    before = _store_files(group_dir)
    source = _own(a, 1)
    chunks = sorted(p for p in source.rglob("*") if p.is_file() and p.name != "zarr.json")
    for victim in chunks[::2]:
        victim.unlink()

    starts, step, axis = a.process._timeseries_axis()
    try:
        action = a.process._plan_timeseries_write(
            _scen(a, 1), "tritonswmm_triton_ts", _TRI, 1, starts[1], step, axis, False
        )[0]
    except Exception as err:  # noqa: BLE001 -- a refusal is an acceptable answer for a source with no record
        action = type(err).__name__
    assert action != "verified", "a half-deleted source with no record was read back as verified"

    log = _scen(a, 1).get_log("tritonswmm")
    a.process._log_field(log, _PENDING).set(list(_GRIDDED_VARS))
    a.process.consolidate_timeseries()
    assert _store_files(group_dir) == before, "the resume rewrote the verified slice"
    assert not source.exists()
    assert _flag(a, 1, _CLEARED) is True and _flag(a, 1, _PENDING) is None
