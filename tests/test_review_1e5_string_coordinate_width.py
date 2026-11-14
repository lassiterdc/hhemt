"""Reviewer tests for fixed-width string coordinates.

A string column appended along ``event_iloc`` takes a value of another length.

No solver and no Snakemake. Every test runs the real writers over the committed multi-sim golden (a finished
analysis tree written by the independent generator) or over a Dataset the test builds, and replaces only the event
id slug, which is what differs in length on a real ensemble.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from tests.fixtures.golden_trees import construct_golden_analysis

NODE = "tritonswmm/swmm_node/timeseries"
HYDRO_SUMMARY = "tritonswmm/swmm_node_hydro/summary"
IDS = {
    "short": "year.8_event_type.rain_event_id.1",
    "long": "year.11_event_type.compound_event_id.3",
    "mid": "year.16_event_type.surge_event_id.4",
}


@pytest.fixture(autouse=True)
def _pin_clean_build(monkeypatch):
    """Force the build-stamp term of the consolidation reuse conjunction TRUE so a dirty checkout rebuilds nothing."""
    import hhemt.provenance as prov

    monkeypatch.setattr(prov, "store_build_mismatch", lambda store, stamp: None)


def _ready(golden_extractor, *, fresh: bool):
    """Golden analysis with series consolidation on and clearing off; ``fresh`` removes the layout-30 tree it ships."""
    import shutil

    from hhemt.scenario import TRITONSWMM_scenario, coupling_nodes_file_for

    system_dir = golden_extractor("multisim")
    cfg = system_dir / "analysis_config.yaml"
    cfg.write_text(
        cfg.read_text()
        + "toggle_consolidate_timeseries: true\ntoggle_clear_scenario_timeseries_after_consolidation: false\n"
    )
    analysis = construct_golden_analysis(system_dir)
    if fresh:
        shutil.rmtree(analysis.analysis_paths.analysis_datatree_zarr, ignore_errors=True)
    for event_iloc in analysis.df_sims.index:
        scen = TRITONSWMM_scenario(event_iloc, analysis)
        ts = xr.open_zarr(scen.scen_paths.output_tritonswmm_node_time_series, consolidated=False)
        nodes_file = coupling_nodes_file_for(scen.scen_paths.swmm_hydraulics_inp)
        nodes_file.parent.mkdir(parents=True, exist_ok=True)
        nodes_file.write_text("".join(f"{n}\n" for n in ts["node_id"].values[:3]))
    return analysis


def _name_events(monkeypatch, ids):
    """Make event ``i`` carry ``ids[i]`` wherever the series consolidation names it."""
    import hhemt.processing_analysis as pa

    monkeypatch.setattr(
        pa.TRITONSWMM_analysis_post_processing, "_event_slug", lambda self, event_iloc: ids[int(event_iloc)]
    )


@pytest.mark.parametrize(
    "order", [("short", "long", "mid"), ("long", "mid", "short")], ids=["increasing-first", "longest-first"]
)
def test_the_whole_consolidation_lands_event_ids_of_different_length_in_every_timeseries_group(
    golden_extractor, monkeypatch, order
):
    """Property: ``consolidate_to_datatree`` (summary branches, then every series group, then hydrology) over an
    ensemble whose event ids differ in length leaves EVERY ``timeseries`` group of the tree holding the ids in event
    order, whichever id comes first.

    Input class: all series groups the golden defines (the SWMM node, SWMM link, gridded TRITON and performance
    groups of each family, plus the hydrology inflow group), found by walking the tree for a ``timeseries`` group
    that carries ``event_id``. The chunk's own test names two of them; the property is over the class.
    Class: REGRESSION - red on the base tree with ``ValueError: Mismatched dtypes for variable event_id``. Reachability:
    the real entry the failed Norfolk run used, with only the slug replaced (E87's measured ``<U37`` against ``<U38``).
    Both-states anchor: asserts the ids read from the tree, which hold under any stored width. Invocation isolation:
    nothing depends on the working directory. Must-fire control: the walk finds at least five such groups, and
    at least one outside the node and performance groups; a walk that found none would pass vacuously.
    Killed by: a width taken from the first event, applied to any one group's writer. Passes under a second correct
    implementation that sizes every string coordinate to the longest id of the whole ensemble, which is known up front.
    """
    ids = [IDS[k] for k in order]
    a = _ready(golden_extractor, fresh=True)
    _name_events(monkeypatch, ids)
    a.process.consolidate_to_datatree()
    tree = xr.open_datatree(a.analysis_paths.analysis_datatree_zarr, engine="zarr", consolidated=False)
    checked = []
    for path, node in tree.subtree_with_keys:
        if path.rstrip("/").endswith("/timeseries") and "event_id" in node.dataset.coords:
            assert [str(v) for v in node.dataset["event_id"].values] == ids, path
            checked.append(path)
    assert len(checked) >= 5, checked
    assert any("swmm_link" in p or "/triton/" in p for p in checked), checked


def test_a_legacy_narrow_series_group_is_recreated_by_the_remedy_its_refusal_names(golden_extractor, monkeypatch):
    """Property: when a ``timeseries`` group created at a narrow width (the Norfolk store, ``<U37``) refuses a longer
    id, the remedy the refusal names is one that WORKS on that group: the consolidation then lands every event.

    Input class: a series group created before widths were fixed (built here by consolidating short ids with the
    width floor lowered to 10), then consolidated again with ids longer than 10.
    Class: NEW CAPABILITY. Plausible wrong implementation: the one under review, whose refusal says to raise
    ``consolidation_version`` so the consolidation recreates the group; a ``consolidation_version`` change rewrites
    only ``summary`` groups, and the ``timeseries`` branches are append-only by design, so following the advice
    refuses again (measured: the second run raises the same ``ProcessingError``, the group stays ``<U10`` holding the
    short ids). Reachability: Norfolk's own tree is this input. Both-states anchor: asserts the stored ids after the
    named remedy, and passes trivially for a refusal that names no remedy of this kind. Invocation isolation: nothing
    depends on the working directory. Must-fire control: the first run is asserted to leave the group narrow.
    Killed by: a refusal naming a remedy that does not rebuild the group. Passes under a second correct
    implementation that rebuilds a timeseries group on a version change, or that names a different remedy that
    works (then only the branch on the named remedy changes), or that widens the older group in place.
    """
    import hhemt.processing_analysis as pa

    cls = pa.TRITONSWMM_analysis_post_processing
    a = _ready(golden_extractor, fresh=True)
    n = len(a.df_sims.index)
    short = [f"s.{i}" for i in range(n)]
    longs = [f"year.{i}0_event_type.compound_event_id.{i}" for i in range(n)]
    current = {"ids": short}
    monkeypatch.setattr(cls, "_event_slug", lambda self, event_iloc: current["ids"][int(event_iloc)])
    with monkeypatch.context() as narrow:
        narrow.setattr(cls, "_EVENT_STRING_WIDTH_DEFAULT", 10)
        a.process.consolidate_timeseries()
    store = a.analysis_paths.analysis_datatree_zarr
    assert xr.open_zarr(store, group=NODE, consolidated=False)["event_id"].dtype.itemsize // 4 == 10, "precondition"
    current["ids"] = longs
    try:
        a.process.consolidate_to_datatree()
    except Exception as err:
        assert type(err).__name__ == "ProcessingError", repr(err)
        named = str(err)
        assert longs[0] in named or longs[-1] in named or "year." in named, named
    else:
        named = ""
    if "consolidation_version" in named:
        monkeypatch.setattr(cls, "CONSOLIDATION_VERSION", cls.CONSOLIDATION_VERSION + 1)
        try:
            a.process.consolidate_to_datatree()
        except Exception as again:
            pytest.fail(f"the remedy the refusal names (consolidation_version) does not recreate the group: {again}")
    stored = xr.open_zarr(store, group=NODE, consolidated=False)
    assert [str(v) for v in stored["event_id"].values] == longs


def test_a_legacy_narrow_series_group_rewrites_an_event_in_place_at_its_stored_width(golden_extractor, monkeypatch):
    """Property: an event present in a narrow (``<U10``) series group whose stored slice differs from its own store is
    rewritten in place by a region write, keeps the group at its stored width, and reads back equal to the source.

    Input class: the ``rewrite`` action of ``_plan_timeseries_write`` on a group whose ``event_id`` is narrower than
    the fixed width. The chunk's tests exercise create, append and the refusal; the region write is the fourth
    branch that touches a string coordinate and a region write raises ``Mismatched dtypes`` for a different width too
    (measured with xarray's region write on ``<U10`` against ``<U255``).
    Class: NEW CAPABILITY. Plausible wrong implementation: the width fit applied on the append branch only, so the
    region write carries ``<U255`` into a ``<U10`` array. Reachability: a stored value perturbed through the zarr API
    makes the stored slice unequal, which is exactly what plans a rewrite. Both-states anchor: asserts the values
    and the ids read back and the stored width. Invocation isolation: nothing depends on the working directory.
    Must-fire control: the perturbation is read back before the rerun and differs from the source.
    Killed by: dropping the fit from the planning step. Passes under a second correct implementation that widens the
    older array in place before the region write, if it keeps the ids and values.
    """
    import zarr

    import hhemt.processing_analysis as pa

    cls = pa.TRITONSWMM_analysis_post_processing
    a = _ready(golden_extractor, fresh=True)
    n = len(a.df_sims.index)
    short = [f"s.{i}" for i in range(n)]
    _name_events(monkeypatch, short)
    with monkeypatch.context() as narrow:
        narrow.setattr(cls, "_EVENT_STRING_WIDTH_DEFAULT", 10)
        a.process.consolidate_timeseries()
    store = a.analysis_paths.analysis_datatree_zarr
    before = xr.open_zarr(store, group=NODE, consolidated=False).load()
    variable = next(
        name
        for name, var in before.data_vars.items()
        if var.dtype.kind == "f" and var.dims[0] == "event_iloc" and var.sizes["event_iloc"] == n
    )
    arr = zarr.open_group(store, mode="r+")[f"{NODE}/{variable}"]
    arr[0] = np.asarray(arr[0]) + 1000.0
    perturbed = xr.open_zarr(store, group=NODE, consolidated=False).load()
    assert not np.array_equal(perturbed[variable].values, before[variable].values, equal_nan=True), "control"
    a.process.consolidate_timeseries()
    after = xr.open_zarr(store, group=NODE, consolidated=False).load()
    np.testing.assert_array_equal(after[variable].values, before[variable].values)
    assert [str(v) for v in after["event_id"].values] == short
    assert after["event_id"].dtype.itemsize // 4 == 10


@pytest.mark.parametrize("narrow_first", [True, False], ids=["narrow-first", "wide-first"])
def test_a_string_data_variable_along_event_iloc_takes_a_value_of_another_length(
    golden_extractor, monkeypatch, narrow_first
):
    """Property: a string DATA variable stored along ``event_iloc`` (the summary groups' per-node ``type``, ``<U8``
    for ``JUNCTION`` and ``<U7`` for ``OUTFALL``) is cast to the stored width or the append is refused naming the
    value, never left to the store's dtype check, whichever event comes first.

    Input class: an event whose longest ``type`` value is longer (or shorter) than the first event's. Errata E87
    states the class as "every string-typed coordinate AND DATA VARIABLE appended along event_iloc"; the golden's
    tree holds two such variables (``swmm_node/summary/type`` and ``swmm_link/summary/type``). The coordinate
    helpers read ``piece.coords`` only, so a data variable is outside both of them.
    Class: NEW CAPABILITY (E87's own text). Plausible wrong implementation: the one under review, which raises
    ``ValueError: Mismatched dtypes for variable type`` out of the store (measured). Reachability: stated by E87's
    class and by the enumerated variables; NOT established that real node or link sets give a different longest type
    within one analysis, since a model's node set is fixed. Both-states anchor: asserts the stored values, or the
    refusal text and the stored event count. Invocation isolation: nothing depends on the working directory.
    Must-fire control: the two arms differ in direction only (store ``<U7`` against ``<U8``, then ``<U8`` against
    ``<U7``), and both raise ``ValueError: Mismatched dtypes`` today, so neither is a case the store tolerates.
    Killed by: a fix that handles coordinates only. Passes under a second correct implementation that stores string
    data variables as one fixed width, or that refuses naming the value.
    """
    from hhemt.exceptions import ProcessingError
    from hhemt.utils import deferred_consolidation

    a = _ready(golden_extractor, fresh=False)
    ids = [IDS["short"], IDS["long"], IDS["mid"]]
    _name_events(monkeypatch, ids)
    narrow_types, wide_types = ["OUTFALL", "STORAGE"], ["JUNCTION", "STORAGE"]
    rows = [narrow_types, wide_types, narrow_types] if narrow_first else [wide_types, narrow_types, wide_types]
    store = a.analysis_paths.analysis_datatree_zarr
    refused = None
    for event_iloc, types in enumerate(rows):
        piece = xr.Dataset(
            {
                "total_inflow_volume": (("event_iloc", "node_id"), np.full((1, 2), float(event_iloc))),
                "type": (("event_iloc", "node_id"), np.asarray([types])),
            },
            coords={"event_iloc": [event_iloc], "node_id": ["a", "b"], "event_id": ("event_iloc", [ids[event_iloc]])},
        )
        try:
            with deferred_consolidation(store):
                a.process._hydrology_write_summary(store, HYDRO_SUMMARY, piece, event_iloc, 5)
        except ProcessingError as err:
            refused = (event_iloc, str(err))
            break
    stored = xr.open_zarr(store, group=HYDRO_SUMMARY, consolidated=False)
    if refused is None:
        assert stored["type"].values.tolist() == rows
    else:
        position, text = refused
        assert "JUNCTION" in text and HYDRO_SUMMARY in text, text
        assert [int(v) for v in stored["event_iloc"].values] == list(range(position)), "nothing written for the refusal"


def test_an_event_id_too_long_for_any_width_is_refused_naming_it_and_nothing_is_written(golden_extractor, monkeypatch):
    """Property: an event id longer than the fixed width is refused before any write, the refusal NAMES the id (E87:
    "refused naming the value when it does not fit"), and the events before it stay readable in the group.

    Input class: a value over the width floor, appended after two events that fit. The fit helper's refusal names the
    value, the group and the width; the widening helper's refusal names the coordinate and a character count only,
    so on an ensemble of hundreds of events it does not say which one.
    Class: NEW CAPABILITY. Plausible wrong implementation: the one under review (message
    "the value of coordinate event_id has N characters and its fixed width is 255"). Reachability: any id over the
    255 floor reaches the ``longest > floor`` branch; the golden's ids are short, so the id is constructed.
    Both-states anchor: asserts the exception type, the first forty characters of the id in the text, and the stored
    ids. Invocation isolation: nothing depends on the working directory. Must-fire control: the two events before
    the refused one landed.
    Killed by: a refusal that names no event. Passes under a second correct implementation that names the full id,
    a truncation of at least forty characters, or the id and the event position together.
    """
    a = _ready(golden_extractor, fresh=True)
    too_long = "year.99_event_type.compound_event_id." + "9" * 300
    ids = [IDS["short"], IDS["long"], too_long]
    _name_events(monkeypatch, ids)
    with pytest.raises(Exception) as err:
        a.process.consolidate_timeseries()
    assert type(err.value).__name__ == "ProcessingError", repr(err.value)
    assert too_long[:40] in str(err.value), str(err.value)
    stored = xr.open_zarr(a.analysis_paths.analysis_datatree_zarr, group=NODE, consolidated=False)
    assert [str(v) for v in stored["event_id"].values] == ids[:2]


# ----------------------------------------------------------------------------------------------------------------
# Round 2: the in-place widening of a stored label array (``_widen_stored_strings``)
# ----------------------------------------------------------------------------------------------------------------


def _leftovers(group_dir):
    return sorted(q.name for q in group_dir.iterdir() if q.name.endswith((".widening", ".widened_aside")))


def _data_chunk_bytes(group_dir):
    """Bytes of every file under a numeric data variable of the group (labels and every ``zarr.json`` excluded)."""
    skip = {"event_id", "variable_set_across_members", "event_iloc", "node_id"}
    return {
        str(f.relative_to(group_dir)): f.read_bytes()
        for f in group_dir.rglob("*")
        if f.is_file() and f.name != "zarr.json" and not (set(f.relative_to(group_dir).parts) & skip)
    }


def _narrow_node_group(a, monkeypatch, ids, width):
    """Series group of ``a`` created at ``width`` from events 0 and 1 (event 2's node store hidden), ids ``ids``."""
    from pathlib import Path

    import hhemt.processing_analysis as pa
    from tests.test_string_coordinate_width import _node_store

    cls = pa.TRITONSWMM_analysis_post_processing
    _name_events(monkeypatch, ids)
    node_store = Path(_node_store(a, 2))
    away = node_store.with_name("hidden_" + node_store.name)
    node_store.rename(away)
    with monkeypatch.context() as narrow:
        narrow.setattr(cls, "_EVENT_STRING_WIDTH_DEFAULT", width)
        with pytest.raises(FileNotFoundError):
            a.process.consolidate_timeseries()
    away.rename(node_store)
    store = a.analysis_paths.analysis_datatree_zarr
    assert xr.open_zarr(store, group=NODE, consolidated=False)["event_id"].dtype.itemsize // 4 == width, "precondition"
    return store


@pytest.mark.parametrize("window", ["before-first-rename", "between-the-renames", "before-aside-removal"])
def test_a_kill_inside_the_in_place_widening_is_settled_by_the_next_call_with_no_loss(
    golden_extractor, monkeypatch, window
):
    """Property: raising at each of the three steps of the swap (old to aside, building to name, aside removed) leaves
    the old label values recoverable on disk, and the NEXT consolidation call settles it: the group reads back with
    the earlier ids unchanged and the new one appended at the fixed width, no ``.widening`` or ``.widened_aside``
    entry remains, and the numeric data chunk files of the earlier events are byte-identical.

    Input class: a kill (an exception) at each step of ``_widen_stored_strings`` on a narrow older series group.
    Class: NEW CAPABILITY. Plausible wrong implementation: the one under review, whose settling of a leftover runs
    only for a column that still has an array under its name and is reached only through ``_fit_strings`` after the
    ``f"{group}/{name}" not in root`` skip, so a kill BETWEEN the renames (name absent, aside present) is never
    settled and the next append sees no label array at all. Reachability: any process kill or disk error at that
    instant; the injection raises ``OSError`` from the rename or the removal. Both-states anchor: asserts values and
    directory listings, which hold under any settling mechanism. Invocation isolation: nothing depends on the working
    directory. Must-fire control: the injected failure is asserted to have fired, and the old values are asserted
    recoverable from disk at the failure.
    Killed by: a settle that is skipped when the name is absent. Passes under a second correct implementation that
    uses a different swap, or that settles every leftover at the start of each consolidation.
    """
    import os
    import shutil

    import zarr

    a = _ready(golden_extractor, fresh=True)
    ids = ["a.1_x_id.1", "a.2_id.3", "a.20_event_id.30"]
    store = _narrow_node_group(a, monkeypatch, ids, 10)
    group_dir = store / NODE
    before = _data_chunk_bytes(group_dir)
    fired = []
    real_rename, real_rmtree = os.rename, shutil.rmtree

    def rename(src, dst, *args, **kwargs):
        name = os.path.basename(str(src))
        if not fired and (
            (window == "before-first-rename" and name == "event_id")
            or (window == "between-the-renames" and name == "event_id.widening")
        ):
            fired.append(name)
            raise OSError("injected kill")
        return real_rename(src, dst, *args, **kwargs)

    def rmtree(path, *args, **kwargs):
        if window == "before-aside-removal" and not fired and str(path).endswith(".widened_aside"):
            fired.append(str(path))
            raise OSError("injected kill")
        return real_rmtree(path, *args, **kwargs)

    with monkeypatch.context() as kill:
        kill.setattr(os, "rename", rename)
        kill.setattr(shutil, "rmtree", rmtree)
        with pytest.raises(OSError, match="injected kill"):
            a.process.consolidate_timeseries()
    assert fired, "control: the injected failure fired"
    recoverable = [
        np.asarray(zarr.open_group(store, mode="r")[f"{NODE}/{n}"][...]).tolist()
        for n in ("event_id", "event_id.widened_aside")
        if (group_dir / n).exists()
    ]
    assert ["a.1_x_id.1", "a.2_id.3"] in recoverable, f"the old labels are on disk: {recoverable}"
    a.process.consolidate_timeseries()
    stored = xr.open_zarr(store, group=NODE, consolidated=False)
    assert [str(v) for v in stored["event_id"].values] == ids
    assert stored["event_id"].dtype.itemsize // 4 == 255
    assert not _leftovers(group_dir)
    after = _data_chunk_bytes(group_dir)
    for name, content in before.items():
        assert after[name] == content, f"{name} changed"


def test_the_in_place_widening_refuses_a_non_string_or_sharded_array_naming_it_and_changes_nothing(tmp_path):
    """Property: asked to widen an array that is not a plain fixed-width string array (a float data variable, or a
    string array that is sharded) the widening raises ``ProcessingError`` naming the group and the array, leaves no
    ``.widening`` or ``.widened_aside`` entry, and leaves the stored bytes as they were.

    Input class: a float array; a sharded ``<U5`` array. Class: NEW CAPABILITY (refusal). Plausible wrong
    implementation: rewriting the array through ``create_array`` without the sharding, losing the shard layout, or
    casting a float array to a string. Reachability: ``_fit_strings`` passes only names whose piece column is a
    string, so the float arm is reached only by a caller naming it; the sharded arm needs a store whose label array
    was sharded, which nothing in this tree writes today - both are therefore guards on the function's own contract
    and not on a path a real run takes. Both-states anchor: asserts the exception, the message and the bytes.
    Invocation isolation: nothing depends on the working directory. Must-fire control: the plain string array of the
    same group IS widened by the same call afterwards.
    Killed by: removing the kind or shards check. Passes under a second correct implementation that widens a sharded
    array correctly, in which case only the sharded arm is dropped.
    """
    import zarr

    from hhemt.exceptions import ProcessingError
    from hhemt.processing_analysis import TRITONSWMM_analysis_post_processing as pa

    store = tmp_path / "s.zarr"
    root = zarr.open_group(store, mode="w", zarr_format=3)
    root.create_array("g/depth", shape=(4,), dtype="float64", chunks=(1,), dimension_names=("event_iloc",))
    root.create_array(
        "g/sharded", shape=(4,), dtype="<U5", chunks=(1,), shards=(4,), dimension_names=("event_iloc",), fill_value=""
    )
    root.create_array("g/plain", shape=(4,), dtype="<U5", chunks=(1,), dimension_names=("event_iloc",), fill_value="")

    def snapshot():
        return {str(f.relative_to(store)): f.read_bytes() for f in sorted(store.rglob("*")) if f.is_file()}

    before = snapshot()
    for name in ("depth", "sharded"):
        with pytest.raises(ProcessingError) as err:
            pa._widen_stored_strings(store, "g", {name: 255})
        assert name in str(err.value) and "g" in str(err.value), str(err.value)
        assert snapshot() == before, f"{name}: the store changed"
    pa._widen_stored_strings(store, "g", {"plain": 255})
    assert zarr.open_group(store, mode="r")["g/plain"].dtype.itemsize // 4 == 255, "control"


def test_a_norfolk_shaped_group_of_37_character_ids_takes_a_38_character_id_with_data_bytes_unchanged(
    golden_extractor, monkeypatch
):
    """Property: a series group created at width 37 holding two 37-character ids takes a third, 38-character id; the
    three ids read back, the group is at the fixed width, and every numeric data chunk file of the two earlier events
    is byte-identical to its bytes before the widening.

    Input class: Norfolk's measured widths (``<U37`` stored, ``<U38`` appended). Class: NEW CAPABILITY. Plausible
    wrong implementation: a widening that cuts the 38th character, or that re-chunks or rewrites a data variable.
    Reachability: E87's measurement. Both-states anchor: ids and bytes. Invocation isolation: nothing depends on the
    working directory. Must-fire control: the stored width is asserted 37 before and 255 after.
    Killed by: those. Passes under a second correct implementation that rebuilds the group.
    """
    a = _ready(golden_extractor, fresh=True)
    first = "year.8_event_type.rain_event_id.1" + "0" * (37 - len("year.8_event_type.rain_event_id.1"))
    second = "year.9_event_type.rain_event_id.2" + "0" * (37 - len("year.9_event_type.rain_event_id.2"))
    third = "year.11_event_type.rain_event_id.3" + "0" * (38 - len("year.11_event_type.rain_event_id.3"))
    ids = [first, second, third]
    assert [len(x) for x in ids] == [37, 37, 38]
    store = _narrow_node_group(a, monkeypatch, ids, 37)
    group_dir = store / NODE
    before = _data_chunk_bytes(group_dir)
    a.process.consolidate_timeseries()
    stored = xr.open_zarr(store, group=NODE, consolidated=False)
    assert [str(v) for v in stored["event_id"].values] == ids
    assert stored["event_id"].dtype.itemsize // 4 == 255
    assert not _leftovers(group_dir)
    after = _data_chunk_bytes(group_dir)
    for name, content in before.items():
        assert after[name] == content, f"{name} changed by the widening"


def test_a_cleared_summary_legacy_group_widens_labels_in_place_and_keeps_the_only_copy_of_its_data(
    golden_extractor, monkeypatch
):
    """Property: an append-only summary group whose first event is recorded cleared (its slice is the only copy) and
    whose labels were created narrow (``event_id`` and the string data variable ``type`` at width 10) takes a second
    event with a longer id and a longer ``type`` value: the cleared event's numeric data are byte-identical, its
    labels read back unchanged, the new event lands, and no swap entry is left.

    Input class: ``_write_summary_group_by_event`` with a non-empty ``cleared`` record over a narrow older group.
    Class: NEW CAPABILITY. Plausible wrong implementation: widening that reads or rewrites a data variable, or that
    loses the cleared event's label. Reachability: the append-only path is what a summary group takes once any
    scenario clears its summary store; an older narrow group is Norfolk's shape. Both-states anchor: values and
    bytes. Invocation isolation: nothing depends on the working directory. Must-fire control: the group is asserted
    narrow before and the cleared event's numeric chunk bytes are non-empty.
    Killed by: rewriting data. Passes under a second correct implementation that widens another way.
    """
    import hhemt.processing_analysis as pa
    from hhemt.utils import deferred_consolidation

    cls = pa.TRITONSWMM_analysis_post_processing
    a = _ready(golden_extractor, fresh=False)
    store = a.analysis_paths.analysis_datatree_zarr
    group = HYDRO_SUMMARY

    def piece(i, eid, types):
        return xr.Dataset(
            {
                "total_inflow_volume": (("event_iloc", "node_id"), np.full((1, 2), float(i + 1))),
                "type": (("event_iloc", "node_id"), np.asarray([types])),
            },
            coords={"event_iloc": [i], "node_id": ["a", "b"], "event_id": ("event_iloc", [eid])},
        )

    first = piece(0, "s.0", ["OUT", "OUT"])
    with monkeypatch.context() as narrow:
        narrow.setattr(cls, "_EVENT_STRING_WIDTH_DEFAULT", 10)
        narrow.setattr(cls, "_STRING_DATA_WIDTH_DEFAULT", 10, raising=False)
        prepared, encoding = cls._prepare_summary_group(first, 5)
        from hhemt.utils import append_zarr

        with deferred_consolidation(store):
            append_zarr(prepared, store, group=group, consolidate=False, mode="w", encoding=encoding)
    group_dir = store / group
    stored = xr.open_zarr(store, group=group, consolidated=False)
    widths = {n: stored[n].dtype.itemsize // 4 for n in ("event_id", "type")}
    assert max(widths.values()) <= 10 or widths["event_id"] == 10, widths
    before = _data_chunk_bytes(group_dir)
    assert before, "control: numeric chunk files exist"
    both = xr.concat(
        [first, piece(1, "year.11_event_type.compound_event_id.3", ["JUNCTION", "JUNCTION"])], dim="event_iloc"
    )
    with deferred_consolidation(store):
        a.process._write_summary_group_by_event(store, group, both, {0: ["total_inflow_volume", "type"]}, 5)
    got = xr.open_zarr(store, group=group, consolidated=False)
    assert [str(v) for v in got["event_id"].values] == ["s.0", "year.11_event_type.compound_event_id.3"]
    assert got["type"].values.tolist() == [["OUT", "OUT"], ["JUNCTION", "JUNCTION"]]
    assert not _leftovers(group_dir)
    after = _data_chunk_bytes(group_dir)
    for name, content in before.items():
        assert after[name] == content, f"{name} (the cleared event's only copy) changed"


def test_a_half_written_widening_array_is_discarded_and_the_next_call_lands_the_event(golden_extractor, monkeypatch):
    """Property: a kill inside the data write of the new label array leaves only ``event_id.widening`` (the old array
    untouched under its name); the next consolidation discards it and lands the long-id event with the earlier ids and
    numeric data bytes unchanged.

    Input class: a created-but-unwritten ``.widening`` array in a narrow group. Class: NEW CAPABILITY. Plausible wrong
    implementation: a reader that treats the leftover as a variable. Reachability: a signal between ``create_array``
    and the write. Both-states anchor: ids, width, listing and bytes. Invocation isolation: nothing depends on the
    working directory. Must-fire control: the leftover exists before the call.
    Killed by: not settling. Passes under any settle that removes or ignores the staging array.
    """
    import zarr

    a = _ready(golden_extractor, fresh=True)
    ids = ["a.1_x_id.1", "a.2_id.3", "a.20_event_id.30"]
    store = _narrow_node_group(a, monkeypatch, ids, 10)
    group_dir = store / NODE
    before = _data_chunk_bytes(group_dir)
    root = zarr.open_group(store, mode="r+")
    old = root[f"{NODE}/event_id"]
    root.create_array(
        f"{NODE}/event_id.widening",
        shape=old.shape,
        dtype="<U255",
        chunks=old.chunks,
        dimension_names=old.metadata.dimension_names,
        fill_value="",
    )
    assert _leftovers(group_dir) == ["event_id.widening"], "control"
    a.process.consolidate_timeseries()
    stored = xr.open_zarr(store, group=NODE, consolidated=False)
    assert [str(v) for v in stored["event_id"].values] == ids
    assert not _leftovers(group_dir)
    for name, content in before.items():
        assert _data_chunk_bytes(group_dir)[name] == content


def test_the_sensitivity_master_copy_settles_an_interrupted_widening_before_it_reads_the_group(tmp_path, monkeypatch):
    """Property: a master series group left between the two renames (``event_id`` absent, ``event_id.widened_aside``
    holding the old labels) is settled by ``_copy_series_group`` and takes the next event, ids intact, no leftover.

    Input class: the swap's middle window on the master copy path. Class: NEW CAPABILITY. Plausible wrong
    implementation: settling only on the consolidation path. Reachability: a kill in the middle window while the
    master copy widens. Both-states anchor: ids and listing. Invocation isolation: nothing depends on the working
    directory. Must-fire control: the group is asserted without ``event_id`` before the call.
    Killed by: not settling on this path. Passes under any settle reached from the copy.
    """
    import os

    from hhemt import sensitivity_analysis as sa
    from hhemt.processing_analysis import TRITONSWMM_analysis_post_processing as pa
    from tests.test_string_coordinate_width import _series_source

    first = _series_source(tmp_path, "first", ["e0"])
    second = _series_source(tmp_path, "second", ["e0", "year.11_event_type.compound_event_id.3"])
    master = tmp_path / "m.zarr"
    xr.Dataset().to_zarr(master, mode="w", consolidated=False)
    with monkeypatch.context() as old:
        old.setattr(pa, "_widen_event_strings", staticmethod(lambda ds, group="": ds))
        sa._copy_series_group(first, "g/timeseries", master, "m/g/timeseries", 5, member_id="0")
    group_dir = master / "m/g/timeseries"
    os.rename(group_dir / "event_id", group_dir / "event_id.widened_aside")
    assert not (group_dir / "event_id").exists(), "control"
    sa._copy_series_group(second, "g/timeseries", master, "m/g/timeseries", 5, member_id="0")
    stored = xr.open_zarr(master, group="m/g/timeseries", consolidated=False)
    assert [str(v) for v in stored["event_id"].values] == ["e0", "year.11_event_type.compound_event_id.3"]
    assert not _leftovers(group_dir)
