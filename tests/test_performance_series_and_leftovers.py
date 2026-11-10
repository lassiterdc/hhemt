"""The per-rank performance series are consolidated, cleared stores take their leftovers with them, and a series group
joins outer on append.

No solver and no Snakemake: every test runs on the committed multi-sim golden (a finished analysis tree whose
per-scenario ``*_perf_tseries`` stores hold nine timer columns on ``(timestep, Rank)`` at a 10 s reporting step), with a
store rewritten where an arm needs another rank count or length. Each arm names its control: the input that makes a
broken implementation read as correct.

Rows: the performance group (shape row "the performance series group"), its clearing, the removal of a cleared store's
``.done``, ``.chapters``, ``.tmp``, ``.aside`` and stray ``.nc.zarr`` siblings, the processed-state check that keeps a
resume out of the gridded writer, and the two outer-join rows of a series group.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from tests.fixtures.golden_trees import construct_golden_analysis

_PERF = "tritonswmm/performance/timeseries"
_PERF_ONLY = "triton_only/performance/timeseries"
_NODE = "tritonswmm/swmm_node/timeseries"
_PERF_CLEARED = "consolidated_performance_timeseries_cleared"
_PERF_PENDING = "consolidated_performance_timeseries_clearing_pending"
_RECORD = "variable_set_across_members"


def _ready(golden_extractor, *, clear: bool, ranks: int | None = None):
    """Golden analysis with series consolidation on, clearing on or off, a coupling list beside every scenario and,
    when ``ranks`` is given, an MPI compute configuration admitting that many ranks."""
    from hhemt.scenario import TRITONSWMM_scenario, coupling_nodes_file_for

    system_dir = golden_extractor("multisim")
    cfg = system_dir / "analysis_config.yaml"
    text = cfg.read_text()
    if ranks is not None:
        assert 'run_mode: "serial"' in text, "the golden's own compute configuration is read here"
        text = text.replace('run_mode: "serial"', f'run_mode: "mpi"\nn_mpi_procs: {ranks}')
    flag = "toggle_clear_scenario_timeseries_after_consolidation: " + ("true" if clear else "false") + "\n"
    cfg.write_text(text + "toggle_consolidate_timeseries: true\n" + flag)
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


def _perf(a, event_iloc: int, family: str = "tritonswmm") -> Path:
    paths = _scen(a, event_iloc).scen_paths
    return Path(
        paths.output_tritonswmm_performance_timeseries
        if family == "tritonswmm"
        else paths.output_triton_only_performance_timeseries
    )


def _gridded(a, event_iloc: int) -> Path:
    return Path(_scen(a, event_iloc).scen_paths.output_tritonswmm_triton_timeseries)


def _node_store(a, event_iloc: int) -> Path:
    return Path(_scen(a, event_iloc).scen_paths.output_tritonswmm_node_time_series)


def _rewrite(path: Path, edit) -> None:
    """Replace the zarr store at ``path`` with ``edit(its dataset)``."""
    ds = xr.open_zarr(path, consolidated=False).load()
    ds = edit(ds)
    shutil.rmtree(path)
    ds.to_zarr(path, mode="w")


def _with_ranks(count: int, steps: int | None = None):
    """An edit giving a per-scenario performance store ``count`` ranks and, if ``steps`` is given, that many steps.

    Rank ``r`` carries the source column plus ``r``, so a value read back at a rank says which rank it came from.
    """

    def edit(ds):
        """The store's Dataset with ranks ``0`` to ``count - 1``, each the source column plus its rank number."""
        if steps is not None:
            ds = ds.isel(timestep=slice(0, steps))
        base = ds.isel(Rank=[0])
        return xr.concat([(base + r).assign_coords(Rank=[r]) for r in range(count)], dim="Rank")

    return edit


def open_coupling_list(a, event_iloc: int) -> list[str]:
    """The node ids listed in event ``event_iloc``'s ``coupling_nodes.txt``."""
    from hhemt.scenario import coupling_nodes_file_for

    nodes_file = coupling_nodes_file_for(_scen(a, event_iloc).scen_paths.swmm_hydraulics_inp)
    return nodes_file.read_text().splitlines()


def _group(a, name: str) -> xr.Dataset:
    return xr.open_zarr(a.analysis_paths.analysis_datatree_zarr, group=name, consolidated=False)


def _events(a, name: str) -> list[int]:
    return [int(v) for v in _group(a, name)["event_iloc"].values]


def _flag(a, event_iloc: int, name: str, model: str = "tritonswmm"):
    field = getattr(_scen(a, event_iloc).get_log(model), name, None)
    return None if field is None else field.get()


def test_two_performance_series_of_different_length_and_rank_count_land_on_one_axis_pair(golden_extractor):
    """Property: two per-rank performance series of different length and rank count land in
    ``tritonswmm/performance/timeseries`` on ONE ``elapsed_s`` axis (the shared one, integer seconds) and ONE
    ``Rank`` axis (``0`` to the configured rank count minus one), each at its own steps and ranks, every cell
    outside them NaN.

    Class: NEW CAPABILITY. Plausible wrong implementations: a ``Rank`` axis taken from the first store written (event
    1 has more ranks than event 0, so it would be refused or cut), an axis of the union of the stores' ranks (event 2's
    one rank would leave ranks 2 and 3 missing), and ``timestep`` minutes read as seconds. Reachability: the golden's
    three events go through the real ``consolidate_timeseries``; the compute configuration is ``mpi`` with 4 ranks.
    Both-states anchor: asserts values read from the group against the source stores, so it holds whatever the
    implementation does to the axes. Invocation isolation: nothing depends on the working directory.
    Killed by: the three above. Passes under a second correct implementation that writes the group whole from the
    axes and fills it by region writes; the axes and the values are the same.
    """
    a = _ready(golden_extractor, clear=False, ranks=4)
    source = {i: xr.open_zarr(_perf(a, i), consolidated=False).load() for i in a.df_sims.index}
    _rewrite(_perf(a, 0), _with_ranks(2, steps=8))
    _rewrite(_perf(a, 1), _with_ranks(4))
    expected = {i: xr.open_zarr(_perf(a, i), consolidated=False).load() for i in a.df_sims.index}
    a.process.consolidate_timeseries()
    stored = _group(a, _PERF).load()

    axis = a.process._timeseries_axis()[2]
    np.testing.assert_array_equal(stored["elapsed_s"].values, axis)
    assert list(stored["Rank"].values) == [0, 1, 2, 3], "the Rank axis is the configured count, not the largest store"
    assert stored["Total"].dims == ("event_iloc", "elapsed_s", "Rank")
    assert source[0].sizes["timestep"] == 12 and expected[0].sizes["timestep"] == 8, "precondition: lengths differ"
    for event_iloc, steps, n_ranks in ((0, 8, 2), (1, 12, 4), (2, 12, 1)):
        total = stored["Total"].sel(event_iloc=event_iloc).values
        np.testing.assert_array_equal(total[:steps, :n_ranks], expected[event_iloc]["Total"].isel(event_iloc=0).values)
        assert np.isnan(total[steps:, :]).all(), (event_iloc, "steps past the store's own are NaN")
        assert np.isnan(total[:, n_ranks:]).all(), (event_iloc, "ranks past the store's own are NaN")
    assert not np.isnan(stored["Total"].sel(event_iloc=0).values[:8, :2]).any(), "control: the covered cells are data"


def test_an_event_with_more_ranks_than_the_fixed_axis_is_refused_naming_it(golden_extractor):
    """Property (must-fire control of the row above): an event whose store holds more ranks than the compute
    configuration admits raises ``ProcessingError`` naming the scenario and the ``Rank`` axis, and the events before it
    stay as written.

    Class: NEW CAPABILITY. Plausible wrong implementation: a ``Rank`` axis grown to the largest store, which accepts
    the 5-rank event and so is not fixed before the first append. Reachability: event 2 is rewritten with 5 ranks under
    a 4-rank configuration. Both-states anchor: asserts the exception type, the event and the stored event list.
    Invocation isolation: nothing depends on the working directory.
    Killed by: that. Passes under a second correct implementation that refuses in the slice builder or before the
    first append; the message and the stored events are the same.
    """
    a = _ready(golden_extractor, clear=False, ranks=4)
    _rewrite(_perf(a, 2), _with_ranks(5))
    with pytest.raises(Exception) as err:
        a.process.consolidate_timeseries()
    assert type(err.value).__name__ == "ProcessingError"
    message = str(err.value)
    assert "event_iloc 2" in message and "Rank" in message and "4" in message, message
    assert _events(a, _PERF) == [0, 1], "the refused event must not have been written"


def test_the_performance_series_is_cleared_after_its_readback(golden_extractor):
    """Property: with the clearing switch on, each scenario's per-rank performance store is deleted, its cleared field
    is True in its own model's log, its pending record is cleared, ``_consolidated_timeseries_cleared("performance")``
    reads True, and the group still holds every slice, for both TRITON models.

    Class: NEW CAPABILITY. Plausible wrong implementation: recording the clearing in the coupled model's log for both
    families (the TRITON-only log stays unset and its processed state reads unprocessed). Reachability: the real
    ``consolidate_timeseries``. Both-states anchor: asserts file existence, the logged field and the group's events.
    Invocation isolation: nothing depends on the working directory.
    Killed by: that. Passes under a second correct implementation that records both fields in one write.
    """
    a = _ready(golden_extractor, clear=True)
    a.process.consolidate_timeseries()
    for event_iloc in a.df_sims.index:
        for family, model, group in (("tritonswmm", "tritonswmm", _PERF), ("triton_only", "triton", _PERF_ONLY)):
            assert not _perf(a, event_iloc, family).exists(), (event_iloc, family)
            assert _flag(a, event_iloc, _PERF_CLEARED, model) is True, (event_iloc, model)
            assert _flag(a, event_iloc, _PERF_PENDING, model) is None
            assert event_iloc in _events(a, group)
            proc = a._retrieve_sim_run_processing_object(event_iloc)
            proc._set_active_model_log(model)
            assert proc._consolidated_timeseries_cleared("performance") is True


def test_the_performance_series_is_kept_when_its_readback_fails(golden_extractor, monkeypatch):
    """Property (must-fire control of the row above): when the read-back of event 1's performance slice fails, event
    1's performance store is kept, its cleared field is not set and the error is a ``ProcessingError``, while event 0,
    already verified, is cleared.

    Class: NEW CAPABILITY. Plausible wrong implementation: clearing before the read-back (the only copy deleted on a
    write that did not take). Reachability: the comparison is replaced so that it fails for the performance group of
    event 1 only. Both-states anchor: asserts file existence and the logged field. Invocation isolation: nothing
    depends on the working directory.
    Killed by: clearing first. Passes under a second correct implementation that verifies through a second reader;
    the same files would remain.
    """
    from hhemt.processing_analysis import TRITONSWMM_analysis_post_processing as pp

    a = _ready(golden_extractor, clear=True)
    real = pp._stored_slice_equals
    monkeypatch.setattr(
        pp,
        "_stored_slice_equals",
        staticmethod(lambda s, g, e, i: False if (i == 1 and g == _PERF) else real(s, g, e, i)),
    )
    with pytest.raises(Exception) as err:
        a.process.consolidate_timeseries()
    assert type(err.value).__name__ == "ProcessingError"
    assert not _perf(a, 0).exists() and _flag(a, 0, _PERF_CLEARED) is True, "event 0 verified, so it is cleared"
    assert _perf(a, 1).exists(), "event 1's read-back failed, so its store is kept"
    assert _flag(a, 1, _PERF_CLEARED) is not True


def _plant_publish_leftovers(store: Path) -> dict[str, Path]:
    """The publish siblings of ``store`` (``.chapters``, ``.tmp``, ``.aside``, stray ``.nc.zarr``), each populated."""
    siblings = {
        "chapters": store.with_name(store.name + ".chapters"),
        "tmp": store.with_name(store.name + ".tmp"),
        "aside": store.with_name(store.name + ".aside"),
        "nc_zarr": store.with_name(store.stem + ".nc.zarr"),
    }
    for path in siblings.values():
        path.mkdir()
        (path / "part").write_text("x")
    return siblings


@pytest.mark.parametrize("clear", [True, False], ids=["cleared", "not-cleared-control"])
def test_a_cleared_stores_done_flag_and_chapters_are_removed_and_an_uncleared_stores_stay(golden_extractor, clear):
    """Property: when a gridded store is cleared its ``.done`` flag and its ``.chapters`` directory go with it; beside
    a store whose clearing is not recorded (clearing off) both stay.

    Class: NEW CAPABILITY. Plausible wrong implementation: deleting the store alone, which leaves a flag asserting a
    completion the directory no longer shows, and leaves the chapter parts. Reachability: the golden's gridded stores
    carry a ``.done`` flag; a ``.chapters`` directory is planted beside each. Both-states anchor: asserts existence of
    the store, the flag and the directory together. Invocation isolation: nothing depends on the working directory.
    Must-fire control: the ``not-cleared-control`` case keeps all three, so the assertion is not met by an
    implementation that deletes siblings unconditionally.
    Killed by: those. Passes under a second correct implementation that removes the siblings before the store.
    """
    a = _ready(golden_extractor, clear=clear)
    for event_iloc in a.df_sims.index:
        store = _gridded(a, event_iloc)
        assert store.exists() and store.with_name(store.name + ".done").exists(), "precondition: golden flag"
        _plant_publish_leftovers(store)
    a.process.consolidate_timeseries()
    for event_iloc in a.df_sims.index:
        store = _gridded(a, event_iloc)
        siblings = [store, store.with_name(store.name + ".done"), store.with_name(store.name + ".chapters")]
        assert [p.exists() for p in siblings] == [not clear] * 3, (event_iloc, siblings)


@pytest.mark.parametrize("clear", [True, False], ids=["cleared", "not-cleared-control"])
def test_a_cleared_stores_tmp_aside_and_stray_nc_zarr_are_removed_and_an_uncleared_stores_stay(golden_extractor, clear):
    """Property: when a store is cleared its ``.tmp``, ``.aside`` and stray ``.nc.zarr`` siblings are removed with it
    and the ``.aside`` is not promoted to the store; beside a store that is not cleared they remain.

    Class: NEW CAPABILITY. Plausible wrong implementations: removing only the flag and the chapters, and calling the
    publish recovery without deleting the store afterwards, which leaves ``.aside`` promoted to the store.
    Reachability: the siblings are planted beside a performance store (zarr suffix) and a gridded store. Both-states
    anchor: asserts the store path and every sibling path. Invocation isolation: nothing depends on the working
    directory. Must-fire control: the ``not-cleared-control`` case keeps every sibling.
    Killed by: those. Passes under a second correct implementation that deletes the siblings by glob.
    """
    a = _ready(golden_extractor, clear=clear)
    planted = _plant_publish_leftovers(_perf(a, 0)) | {
        f"grid_{k}": v for k, v in _plant_publish_leftovers(_gridded(a, 1)).items()
    }
    a.process.consolidate_timeseries()
    for name, path in planted.items():
        assert path.exists() is (not clear), f"{name}: {path}"
    assert _perf(a, 0).exists() is (not clear) and _gridded(a, 1).exists() is (not clear)


def test_an_aside_and_a_tmp_beside_an_absent_store_are_removed_and_the_store_is_not_recreated(golden_extractor):
    """Property: deleting a store that is already absent, with an ``.aside`` and a ``.tmp`` beside it (the state a
    clearing killed after the store's deletion and before its siblings' leaves), removes both and does not recreate the
    store from either.

    Class: NEW CAPABILITY. Plausible wrong implementation: the publish recovery run on the absent store and its result
    kept, which renames ``.tmp`` (or ``.aside``) to the store path. Reachability: the deletion helper of the real
    consolidation object called on a path the clearing has already emptied. Both-states anchor: asserts the three
    paths. Invocation isolation: nothing depends on the working directory.
    Killed by: that. Passes under a second correct implementation that removes the two siblings without the recovery.
    """
    a = _ready(golden_extractor, clear=True)
    a.process.consolidate_timeseries()
    absent = _perf(a, 2)
    assert not absent.exists(), "precondition: the clearing removed the store"
    for suffix in (".aside", ".tmp"):
        leftover = absent.with_name(absent.name + suffix)
        leftover.mkdir()
        (leftover / "part").write_text("x")
    type(a.process)._delete_cleared_stores(_scen(a, 2), [absent])
    assert not absent.with_name(absent.name + ".aside").exists()
    assert not absent.with_name(absent.name + ".tmp").exists()
    assert not absent.exists()


@pytest.mark.parametrize(
    "model, method", [("tritonswmm", "_export_TRITONSWMM_TRITON_outputs"), ("triton", "_export_TRITON_only_outputs")]
)
def test_no_resume_reaches_the_gridded_writer_after_clearing(golden_extractor, monkeypatch, model, method):
    """Property: for a scenario whose gridded store, ``.done`` flag and ``.chapters`` are gone and whose cleared field
    is set, the processed-state check reads processed and the TRITON export returns without calling the chapter
    writer, even when the processing record that would otherwise say "already written" is absent; with the cleared
    field removed the same call reaches the writer.

    Class: NEW CAPABILITY. Plausible wrong implementation: the skip left on the processing record alone. A forced
    re-run clears that record, and the writer's own guard (``.done`` present) no longer exists once the flag is
    removed with the store, so the writer would rebuild a store from whatever raw files survived. Reachability: the
    real clearing removes the store, its flag and chapters; the record is then removed by hand and the raw-frame
    builder and the writer are replaced, the writer by a function that raises. Both-states anchor: asserts the
    processed-state property and whether the writer ran. Invocation isolation: nothing depends on the working
    directory. Must-fire control: the same call with the cleared field removed reaches the writer (the sentinel
    error), so a skip that does not read the field fails the first arm and a call that never reaches the writer fails
    the second.
    Killed by: that. Passes under a second correct implementation that checks the field inside the writer's caller.
    """
    import pandas as pd

    import hhemt.process_simulation as ps

    a = _ready(golden_extractor, clear=True)
    a.process.consolidate_timeseries()
    store = Path(
        getattr(
            _scen(a, 0).scen_paths,
            "output_tritonswmm_triton_timeseries" if model == "tritonswmm" else "output_triton_only_timeseries",
        )
    )
    assert not store.exists() and not store.with_name(store.name + ".done").exists()
    assert not store.with_name(store.name + ".chapters").exists()

    class _WriterReached(RuntimeError):
        pass

    def _writer(*args, **kwargs):
        raise _WriterReached

    monkeypatch.setattr(ps, "triton_raw_frame_or_raise", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(ps.TRITONSWMM_sim_post_processing, "_streaming_chunked_zarr_write", _writer)
    proc = a._retrieve_sim_run_processing_object(0)
    proc._set_active_model_log(model)
    proc.log.processing_log.outputs.pop(store.name, None)
    assert proc.TRITON_outputs_processed is True
    getattr(proc, method)()

    proc.log.consolidated_triton_timeseries_cleared.set(False)
    proc.log.TRITON_timeseries_written.set(False)
    assert proc.TRITON_outputs_processed is False, "control: with no cleared record the written flag decides"
    with pytest.raises(_WriterReached):
        getattr(proc, method)()


def _consolidate_events_before_the_last(a) -> None:
    """Consolidate (and clear) the events that have stores, stopping at the one whose stores were moved away.

    The pass stops at event 2 with a ``FileNotFoundError``, which is how a scenario that has not been processed yet
    reads, and leaves events 0 and 1 in the group with their own stores cleared.
    """
    with pytest.raises(FileNotFoundError):
        a.process.consolidate_timeseries()


def _with_event_2_unprocessed(a) -> dict[Path, Path]:
    """Move event 2's node and link stores aside, returning ``{store: where it went}`` for ``_restore``."""
    hidden = {}
    for kind in ("node", "link"):
        store = Path(getattr(_scen(a, 2).scen_paths, f"output_tritonswmm_{kind}_time_series"))
        hidden[store] = store.with_name("hidden_" + store.name)
        store.rename(hidden[store])
    return hidden


def _restore(hidden: dict[Path, Path]) -> None:
    for store, away in hidden.items():
        away.rename(store)


def _after_a_clearing_with_a_widened_keep_list(golden_extractor):
    """An analysis whose events 0 and 1 are consolidated and cleared and whose event 2, not yet processed, then
    appears with one more time-variant variable (``extra_series``) than the group holds."""
    a = _ready(golden_extractor, clear=True)
    hidden = _with_event_2_unprocessed(a)
    _consolidate_events_before_the_last(a)
    assert _events(a, _NODE) == [0, 1]
    _restore(hidden)
    _rewrite(_node_store(a, 2), lambda ds: ds.assign(extra_series=ds["depth"] * 2))
    return a


def test_a_series_group_takes_a_new_variable_by_an_outer_join(golden_extractor):
    """Property: after events 0 and 1 are consolidated and cleared, a new scenario whose node series carries a variable
    the ``timeseries`` group lacks (a widened keep-list) appends without a refusal; the variable is NaN at every
    earlier event and equal to its source at the new one; every chunk file of the group's other variables that was
    already written is byte-identical to before; the new event's variable-set coordinate names the variable.

    Class: NEW CAPABILITY. Plausible wrong implementations: the landed refusal (``ProcessingError``), and appending
    the new variable without creating it first at the group's length, which leaves one array longer than the rest.
    Reachability: the real ``consolidate_timeseries`` in two passes, the second after event 2's stores are restored
    with an extra time-variant variable. Both-states anchor: asserts values read from the group and the bytes of
    files that were already written. Invocation isolation: nothing depends on the working directory.
    Killed by: those. Passes under a second correct implementation that creates the array by a region write over a
    group pre-sized to the analysis's event count.
    """
    a = _after_a_clearing_with_a_widened_keep_list(golden_extractor)
    group_dir = a.analysis_paths.analysis_datatree_zarr / _NODE
    before = {str(p.relative_to(group_dir)): p.read_bytes() for p in sorted(group_dir.rglob("*")) if p.is_file()}
    source = xr.open_zarr(_node_store(a, 2), consolidated=False).load()
    a.process.consolidate_timeseries()
    stored = _group(a, _NODE).load()
    assert _events(a, _NODE) == [0, 1, 2]
    assert "extra_series" in stored.data_vars
    assert bool(stored["extra_series"].sel(event_iloc=[0, 1]).isnull().all()), "NaN at every earlier event"
    n = source.sizes["date_time"]
    np.testing.assert_array_equal(
        stored["extra_series"].sel(event_iloc=2).values[:, :n],
        source["extra_series"].transpose("node_id", "date_time").values,
    )
    after = {str(p.relative_to(group_dir)): p.read_bytes() for p in sorted(group_dir.rglob("*")) if p.is_file()}
    assert before, "precondition: the group holds files"
    for name, content in before.items():
        if name.endswith("zarr.json"):
            continue
        assert after[name] == content, f"{name} changed although it is not a chunk of the new variable"
    assert "extra_series" in str(stored[_RECORD].sel(event_iloc=2).values)


def test_without_the_array_created_first_the_same_append_raises_and_leaves_the_group_unreadable(
    golden_extractor, monkeypatch
):
    """Property (must-fire control of the row above): with the creation of the NaN array replaced by a no-op, the same
    append raises ``ValueError`` and leaves the group unreadable (opening it raises ``ValueError`` on the conflicting
    ``event_iloc`` sizes of its arrays), so the creation is what keeps the group readable.

    Class: NEW CAPABILITY. Plausible wrong implementation: the arm above passing because the append tolerates a
    variable the group lacks. Reachability: the same two passes, the creation replaced on the class. Measured, not
    assumed: the append itself writes, and the raise is the read-back's, so the group is NOT left as it was. Both-states
    anchor: asserts the exception type of the pass and of the open. Invocation isolation: nothing depends on the
    working directory.
    Killed by: that. Passes under a second correct implementation that fails earlier with a ``ValueError``, if it also
    leaves the group readable (then only the first assertion applies and the second fails by design).
    """
    from hhemt.processing_analysis import TRITONSWMM_analysis_post_processing as pp

    a = _after_a_clearing_with_a_widened_keep_list(golden_extractor)
    monkeypatch.setattr(pp, "_ensure_series_variables", lambda self, store, group, ds: None)
    with pytest.raises(ValueError):
        a.process.consolidate_timeseries()
    with pytest.raises(ValueError, match="conflicting sizes"):
        _group(a, _NODE)


def test_a_series_group_pads_a_variable_the_new_event_lacks(golden_extractor, monkeypatch):
    """Property: a new scenario whose node series lacks a group variable (a narrowed keep-list) appends without a
    refusal, NaN at the new event, and the event's variable-set coordinate names the missing variable; a scenario
    lacking ``is_coupling_node`` gets it derived from its ``coupling_nodes.txt``, never NaN-filled.

    Class: NEW CAPABILITY. Plausible wrong implementations: the landed refusal, an append of the shorter variable set
    (the store becomes unreadable), and a flag NaN-filled by the pad instead of derived. Reachability: the real
    ``consolidate_timeseries`` in two passes, event 2's node store restored without ``depth`` and without the flag.
    Both-states anchor: asserts values read from the group, the coordinate text and the stored flag.
    Invocation isolation: nothing depends on the working directory. Must-fire control: the derived flag of event 2
    equals its listed nodes and is not NaN, and an earlier event's ``depth`` is unchanged, so the pad touched only
    the new event.
    Killed by: those. Passes under a second correct implementation that pads by a region write.
    """
    a = _ready(golden_extractor, clear=True)
    hidden = _with_event_2_unprocessed(a)
    _consolidate_events_before_the_last(a)
    depth_before = _group(a, _NODE)["depth"].load().sel(event_iloc=[0, 1])
    _restore(hidden)
    _rewrite(_node_store(a, 2), lambda ds: ds.drop_vars("depth"))
    a.process.consolidate_timeseries()
    stored = _group(a, _NODE).load()
    assert _events(a, _NODE) == [0, 1, 2]
    assert bool(stored["depth"].sel(event_iloc=2).isnull().all()), "NaN at the new event"
    np.testing.assert_array_equal(stored["depth"].sel(event_iloc=[0, 1]).values, depth_before.values)
    record = str(stored[_RECORD].sel(event_iloc=2).values)
    assert "depth" in record and "LACKS" in record, record
    listed = {ln.strip() for ln in open_coupling_list(a, 2)}
    flag = stored["is_coupling_node"].sel(event_iloc=2)
    assert not bool(flag.isnull().any()), "the flag is derived, not NaN-filled"
    assert {str(n) for n in flag["node_id"].values[flag.values.astype(bool)]} == listed


def _strip_record(a, group: str) -> None:
    """Remove the variable-set record array from a stored group, making it a group written before the record existed."""
    shutil.rmtree(a.analysis_paths.analysis_datatree_zarr / group / _RECORD)


def test_a_legacy_group_without_the_record_gets_it_on_its_first_heterogeneous_append(golden_extractor):
    """Property: a series group stored without the variable-set record, given a heterogeneous append, gets the record
    created first: blank over the earlier events, naming the variable at the new event; the group stays readable.

    Class: NEW CAPABILITY. Plausible wrong implementation: no record for an older group (NaN unexplained), or the
    record appended without being created first (the group becomes unreadable). Reachability: two passes; the record
    array is deleted between them to make the group a pre-record one. Both-states anchor: asserts values read from
    the group. Invocation isolation: nothing depends on the working directory.
    Killed by: those. Passes under a second correct implementation that rewrites the group whole.
    """
    a = _ready(golden_extractor, clear=True)
    hidden = _with_event_2_unprocessed(a)
    _consolidate_events_before_the_last(a)
    _strip_record(a, _NODE)
    _restore(hidden)
    _rewrite(_node_store(a, 2), lambda ds: ds.assign(extra_series=ds["depth"] * 2))
    a.process.consolidate_timeseries()
    record = _group(a, _NODE).load()[_RECORD]
    assert [str(v) for v in record.sel(event_iloc=[0, 1]).values] == ["", ""]
    assert "extra_series" in str(record.sel(event_iloc=2).values)


def test_a_uniform_append_to_a_legacy_group_creates_no_record(golden_extractor):
    """Property (must-fire control of the row above): a uniform append to a group without the record creates none.

    Class: NEW CAPABILITY. Plausible wrong implementation: creating the record on every append. Reachability: as above
    without the extra variable. Both-states anchor: asserts the group's variables. Invocation isolation: nothing
    depends on the working directory.
    Killed by: that. Passes under a second correct implementation that creates the record lazily elsewhere.
    """
    a = _ready(golden_extractor, clear=True)
    hidden = _with_event_2_unprocessed(a)
    _consolidate_events_before_the_last(a)
    _strip_record(a, _NODE)
    _restore(hidden)
    a.process.consolidate_timeseries()
    assert _RECORD not in _group(a, _NODE).variables
    assert _events(a, _NODE) == [0, 1, 2]
