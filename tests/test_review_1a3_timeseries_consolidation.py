"""Tests for the consolidated time-series store, the clearing predicate and its re-run rule.

Every test runs on the committed multi-sim golden (no solver, no Snakemake) or on a stand-in object, as the
chunk's own tests do. The inputs here are ones that file does not reach: a read-back that fails for one of the two
stores of a scenario, a write that lands damaged, stored cells that differ from the scenario's only in NaN-ness,
an interrupted clearing pass that is then resumed, a reprocess that deletes the summary tree while the per-scenario
series are already cleared, the log predicates read for node and link independently, an elapsed axis taken from
the event windows when they exceed every stored series, and the sensitivity master's graft.
"""

from __future__ import annotations

import shutil
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from tests.fixtures.golden_trees import construct_golden_analysis

_NODE = "swmm_node_timeseries"
_LINK = "swmm_link_timeseries"


def _ready(golden_extractor, *, clear: bool):
    """Golden analysis with consolidation (and optionally clearing) on, and a coupling list beside every scenario."""
    from hhemt.scenario import TRITONSWMM_scenario, coupling_nodes_file_for

    system_dir = golden_extractor("multisim")
    cfg = system_dir / "analysis_config.yaml"
    extra = "toggle_clear_scenario_timeseries_after_consolidation: true\n" if clear else ""
    cfg.write_text(cfg.read_text() + "toggle_consolidate_timeseries: true\n" + extra)
    analysis = construct_golden_analysis(system_dir)
    for event_iloc in analysis.df_sims.index:
        scen = TRITONSWMM_scenario(event_iloc, analysis)
        ts = xr.open_zarr(scen.scen_paths.output_tritonswmm_node_time_series, consolidated=False)
        nodes_file = coupling_nodes_file_for(scen.scen_paths.swmm_hydraulics_inp)
        nodes_file.parent.mkdir(parents=True, exist_ok=True)
        nodes_file.write_text("".join(f"{n}\n" for n in ts["node_id"].values[:3]))
    return analysis


def _own_store(analysis, event_iloc: int, kind: str):
    from hhemt.scenario import TRITONSWMM_scenario

    return getattr(TRITONSWMM_scenario(event_iloc, analysis).scen_paths, f"output_tritonswmm_{kind}_time_series")


def _rewrite(path, edit) -> None:
    ds = xr.open_zarr(path, consolidated=False).load()
    ds = edit(ds)
    shutil.rmtree(path)
    ds.to_zarr(path, mode="w")


def _group(analysis, name: str) -> xr.Dataset:
    return xr.open_zarr(
        analysis.analysis_paths.analysis_timeseries_zarr, group=f"tritonswmm/{name}", consolidated=False
    )


def _events(analysis, name: str = _NODE) -> list[int]:
    return [int(v) for v in _group(analysis, name)["event_iloc"].values]


def test_a_scenario_is_cleared_only_after_BOTH_of_its_slices_read_back(golden_extractor, monkeypatch):
    """Property: when the link slice of an event fails its read-back, that event's NODE store is kept too, and the
    model log records no clearing for it.

    Input class: a read-back that fails for the link group of event 1 alone, with clearing on. Plausible wrong
    implementation: deleting each store as soon as its own slice verifies, so the node store of event 1 is gone
    while its link slice is unverified; the next pass then has half a scenario. A read-back that fails every group of
    event 1 cannot tell this from the correct order.
    Why required: a scenario's node and link time-series stores are cleared after its append is durable and read
    back, as one act.
    Killed by: moving the delete between the node and link appends. Passes under a second correct implementation
    that verifies both slices and then clears both, in either order.
    """
    from hhemt.processing_analysis import TRITONSWMM_analysis_post_processing as pp
    from hhemt.scenario import TRITONSWMM_scenario

    a = _ready(golden_extractor, clear=True)
    real = pp._stored_slice_equals
    monkeypatch.setattr(
        pp,
        "_stored_slice_equals",
        staticmethod(lambda s, g, e, i: False if (i == 1 and g.endswith(_LINK)) else real(s, g, e, i)),
    )
    with pytest.raises(Exception) as err:
        a.process.consolidate_timeseries()
    assert type(err.value).__name__ == "ProcessingError"
    assert _own_store(a, 1, "node").exists(), "the node store was deleted before the link slice was verified"
    assert _own_store(a, 1, "link").exists()
    log = TRITONSWMM_scenario(1, a).get_log("tritonswmm")
    for name in ("consolidated_node_timeseries_cleared", "consolidated_link_timeseries_cleared"):
        field = getattr(log, name, None)
        assert field is None or field.get() is not True, name
    assert not _own_store(a, 0, "node").exists(), "control: event 0 verified in both groups, so it is cleared"


def test_the_real_readback_decides_whether_a_scenario_is_cleared(golden_extractor, monkeypatch):
    """Property: if the slice that lands in the store differs from the data it was written from, the scenario's stores
    are kept and the pass raises `ProcessingError`, with the real comparison (not a replaced one) doing the deciding.

    Input class: an append whose written values are damaged right after it returns (a cell overwritten through the
    zarr API), with clearing on, for the first event. Plausible wrong implementation: a read-back that compares the
    in-memory data with itself, or one that is skipped; the stores are then deleted over a slice that did not take.
    The chunk's own test replaces the comparison and therefore cannot reach this.
    Killed by: a read-back that always answers True. Passes under a second correct implementation that verifies by
    re-reading the written arrays through any other reader.
    """
    import zarr

    import hhemt.processing_analysis as pa

    a = _ready(golden_extractor, clear=True)
    real_append = pa.append_zarr

    def damaging_append(ds, store, **kw):
        """Append as the real function does, then overwrite the last stored node depth row with 999."""
        real_append(ds, store, **kw)
        if kw.get("group", "").endswith(_NODE):
            arr = zarr.open_group(store, path=kw["group"], mode="r+")["depth"]
            arr[arr.shape[0] - 1, ...] = 999.0

    monkeypatch.setattr(pa, "append_zarr", damaging_append)
    with pytest.raises(Exception) as err:
        a.process.consolidate_timeseries()
    assert type(err.value).__name__ == "ProcessingError"
    assert _own_store(a, 0, "node").exists() and _own_store(a, 0, "link").exists()


def test_a_rerun_detects_damage_that_differs_only_in_nan_ness(golden_extractor):
    """Property: after a pass, a stored padded NaN overwritten by 0.0, and a stored 0.0 overwritten by NaN, are each
    found by the next pass's read-back and rewritten from the scenario's store.

    Input class: the two cells where a NaN-blind comparison agrees with the original: expected NaN against stored
    0.0 (event 0 is shorter than the axis, so its tail is padding), and expected 0.0 against stored NaN (an own-store
    cell set to 0.0). Plausible wrong implementations: comparing after `nan_to_num`, or only where both values are
    finite; either reads both damaged cells as equal and a clearing pass would delete the only copy over them.
    Why required: a NaN-aware comparison is required for every data variable, and padding makes NaN a real
    stored value.
    Killed by: either comparison. Passes under a second correct implementation using `xr.testing.assert_identical`.
    """
    import zarr

    a = _ready(golden_extractor, clear=False)
    _rewrite(_own_store(a, 0, "node"), lambda ds: ds.isel(date_time=slice(0, 8)))

    def zero_a_cell(ds):
        """Set one depth cell of the scenario store to exactly 0.0."""
        ds["depth"] = ds["depth"].copy()
        ds["depth"][dict(date_time=2, node_id=1)] = 0.0
        return ds

    _rewrite(_own_store(a, 1, "node"), zero_a_cell)
    a.process.consolidate_timeseries()
    depth = _group(a, _NODE)["depth"]
    assert bool(depth.isel(event_iloc=0, elapsed_s=slice(8, None)).isnull().all()), "event 0 must be padded"
    assert float(depth.isel(event_iloc=1, elapsed_s=2, node_id=1)) == 0.0
    arr = zarr.open_group(a.analysis_paths.analysis_timeseries_zarr, path=f"tritonswmm/{_NODE}", mode="r+")["depth"]
    dims = list(depth.dims)
    pad_index = [0] * len(dims)
    pad_index[dims.index("elapsed_s")] = 10
    zero_index = [1] * 0 + [0] * len(dims)
    zero_index[dims.index("event_iloc")] = 1
    zero_index[dims.index("elapsed_s")] = 2
    zero_index[dims.index("node_id")] = 1
    arr[tuple(pad_index)] = 0.0
    arr[tuple(zero_index)] = np.nan
    a.process.consolidate_timeseries()
    repaired = _group(a, _NODE)["depth"]
    assert _events(a) == [0, 1, 2]
    assert bool(repaired.isel(event_iloc=0, elapsed_s=slice(8, None)).isnull().all()), "padding cell was not rewritten"
    assert float(repaired.isel(event_iloc=1, elapsed_s=2, node_id=1)) == 0.0, "the zero cell was not rewritten"


def test_an_interrupted_clearing_pass_resumes_to_the_same_end_state(golden_extractor, monkeypatch):
    """Property: a clearing pass that stops at the third event leaves the first two events cleared and their slices
    readable from the unindexed store; running the pass again finishes without refusing the cleared events, clears
    the third, and leaves the store indexed with each event once.

    Input class: an exception at event 2's read-back with clearing on, then a second pass. Plausible wrong
    implementations: a resume that treats an absent own store as damage (refusing events 0 and 1 although the log
    records their verified clearing), or a first pass that leaves the cleared events unreadable.
    Why required: the store is written inside a block that holds no index until it exits normally; the
    guarantee that clearing never costs the only copy has to hold across exactly this interruption.
    Killed by: ignoring the log's cleared record on resume. Passes under a second correct implementation that
    rebuilds the index after each clearing.
    """
    from hhemt.processing_analysis import TRITONSWMM_analysis_post_processing as pp
    from hhemt.utils import zarr_store_has_index

    a = _ready(golden_extractor, clear=True)
    real = pp._stored_slice_equals
    monkeypatch.setattr(
        pp, "_stored_slice_equals", staticmethod(lambda s, g, e, i: False if i == 2 else real(s, g, e, i))
    )
    with pytest.raises(Exception) as err:
        a.process.consolidate_timeseries()
    assert type(err.value).__name__ == "ProcessingError"
    store = a.analysis_paths.analysis_timeseries_zarr
    assert not zarr_store_has_index(store), "the interrupted block must leave the store unindexed, not stale"
    assert _events(a) == [0, 1, 2]
    for event_iloc in (0, 1):
        assert not _own_store(a, event_iloc, "node").exists()
    assert np.isfinite(_group(a, _NODE)["depth"].isel(event_iloc=[0, 1]).values).any()
    assert _own_store(a, 2, "node").exists()

    monkeypatch.undo()
    a.process.consolidate_timeseries()
    assert zarr_store_has_index(store)
    assert _events(a) == [0, 1, 2] and _events(a, _LINK) == [0, 1, 2]
    for event_iloc in (0, 1, 2):
        assert not _own_store(a, event_iloc, "node").exists() and not _own_store(a, event_iloc, "link").exists()


def test_a_reprocess_that_deletes_the_summary_tree_rebuilds_it_from_cleared_scenarios(golden_extractor):
    """Property: with clearing on, deleting `analysis_datatree.zarr` after a consolidation (what a reprocess does) and
    consolidating again rebuilds the tree, and the time-series store keeps every event with unchanged values.

    Input class: the one routine operation the separate store exists to survive, run with the scenario stores
    already deleted. Plausible wrong implementation: a consolidation that re-reads each scenario's own series and
    fails on the cleared ones, or that rewrites the store from nothing.
    Killed by: either. Passes under a second correct implementation that skips the series stage when the log
    records every scenario cleared.
    """
    a = _ready(golden_extractor, clear=True)
    a.process.consolidate_to_datatree()
    tree = a.analysis_paths.analysis_datatree_zarr
    before = _group(a, _NODE)["depth"].load()
    shutil.rmtree(tree)
    a.process.consolidate_to_datatree()
    assert tree.exists()
    xr.testing.assert_identical(_group(a, _NODE)["depth"].load(), before)


def _stub_proc(*, node_cleared: bool, link_cleared: bool):
    from hhemt.process_simulation import TRITONSWMM_sim_post_processing

    flag = lambda value: SimpleNamespace(get=lambda: value, set=lambda _v: None)  # noqa: E731
    proc = TRITONSWMM_sim_post_processing.__new__(TRITONSWMM_sim_post_processing)
    proc.log = SimpleNamespace(
        consolidated_node_timeseries_cleared=flag(node_cleared),
        consolidated_link_timeseries_cleared=flag(link_cleared),
        SWMM_node_timeseries_written=flag(False),
        SWMM_link_timeseries_written=flag(False),
    )
    proc.scen_paths = SimpleNamespace(
        output_tritonswmm_node_time_series=None,
        output_tritonswmm_link_time_series=None,
        output_swmm_only_node_time_series=None,
        output_swmm_only_link_time_series=None,
    )
    proc._already_written = lambda _path: False
    return proc


@pytest.mark.parametrize("node_cleared, link_cleared", [(False, False), (True, False), (False, True), (True, True)])
def test_the_node_and_link_predicates_read_their_own_cleared_record_and_only_for_the_coupled_model(
    node_cleared, link_cleared
):
    """Property: with both stores absent, `_swmm_node_outputs_processed("tritonswmm")` is True exactly when the node
    cleared record is True, the link predicate exactly when the link record is True, and the SWMM-only model reads
    neither record.

    Input class: the 2 x 2 of cleared records with no store on disk. Plausible wrong implementations: one predicate
    reading the other's record, a single shared record, or the SWMM-only predicate honouring a record only the
    coupled consolidation writes (a SWMM-only scenario would then read as processed with nothing on disk).
    Must-fire control: the (False, False) row reads False for both, so a predicate that always answers True fails.
    Killed by: swapping the two record names, or dropping the model test. Passes under a second correct
    implementation that reads one combined record only if it is set for both stores together.
    """
    proc = _stub_proc(node_cleared=node_cleared, link_cleared=link_cleared)
    assert proc._swmm_node_outputs_processed("tritonswmm") is node_cleared
    assert proc._swmm_link_outputs_processed("tritonswmm") is link_cleared
    assert proc._swmm_node_outputs_processed("swmm") is False
    assert proc._swmm_link_outputs_processed("swmm") is False


def test_the_elapsed_axis_is_sized_from_the_event_windows_not_from_the_stored_series(golden_extractor, monkeypatch):
    """Property: when the longest event window is longer than every stored series, the shared `elapsed_s` axis has the
    window's length and every event is padded to it.

    Input class: event 2's window is 150 s against stored series of 12 steps of 10 s (120 s). Plausible wrong
    implementation: taking the axis from the longest series on disk, which is the wrong source (the extent is the
    analysis's own event window, not the length of whatever series happens to be on disk); a later event whose
    series is longer than the first appended would then not fit.
    Killed by: sizing from the stored series. Passes under a second correct implementation computing the extent
    from the configured simulation end.
    """
    import pandas as pd

    from hhemt.scenario import TRITONSWMM_scenario

    a = _ready(golden_extractor, clear=False)
    start = pd.Timestamp("2000-01-01")
    seconds = {0: 120, 1: 120, 2: 150}
    monkeypatch.setattr(
        TRITONSWMM_scenario,
        "_resolve_event_window",
        lambda self: (start, start + pd.Timedelta(seconds=seconds[int(self.event_iloc)])),
    )
    a.process.consolidate_timeseries()
    node = _group(a, _NODE)
    assert node.sizes["elapsed_s"] == 15
    assert int(node["elapsed_s"].values[-1]) == 150
    assert bool(node["depth"].isel(elapsed_s=slice(12, None)).isnull().all())


def test_the_sensitivity_master_grafts_each_members_store_and_skips_a_member_without_one(golden_extractor, tmp_path):
    """Property: `consolidate_sensitivity_timeseries` writes `sensitivity_timeseries.zarr` holding each member's node
    and link groups under `member_{id}/tritonswmm/...` with the member's values, leaves a member with no store out,
    and returns None when no member has one.

    Input class: two members whose store is a real consolidated store (copied from a golden consolidation), one
    member with no store, then a master where none has. The method is called on a stand-in carrying only what it
    reads. Plausible wrong implementations: grafting the root only, prefixing the wrong member name, writing the
    target when nothing was grafted, or raising on a member whose consolidation was skipped.
    Killed by: any of those. Passes under a second correct implementation that links the member stores without
    copying, provided the master opens with the same group paths.
    """
    from hhemt.sensitivity_analysis import TRITONSWMM_sensitivity_analysis as sens

    a = _ready(golden_extractor, clear=False)
    a.process.consolidate_timeseries()
    source = a.analysis_paths.analysis_timeseries_zarr
    for member in ("0", "1"):
        shutil.copytree(source, tmp_path / f"m{member}.zarr")

    def master(paths):
        """A stand-in master holding only what the graft reads, with one member per entry of `paths`."""
        return SimpleNamespace(
            analysis_paths=SimpleNamespace(sensitivity_timeseries_zarr=tmp_path / "sensitivity_timeseries.zarr"),
            experiment=SimpleNamespace(cfg_analysis=SimpleNamespace(analysis_id="exp")),
            member_prefix="member_",
            members={
                mid: SimpleNamespace(analysis_paths=SimpleNamespace(analysis_timeseries_zarr=p))
                for mid, p in paths.items()
            },
        )

    out = sens.consolidate_sensitivity_timeseries(
        master({"0": tmp_path / "m0.zarr", "1": tmp_path / "m1.zarr", "2": tmp_path / "absent.zarr"})
    )
    assert out == tmp_path / "sensitivity_timeseries.zarr"
    tree = xr.open_datatree(out, engine="zarr", consolidated=False)
    assert {"member_0", "member_1"} <= set(tree.children) and "member_2" not in tree.children
    for member in ("member_0", "member_1"):
        grafted = tree[f"{member}/tritonswmm/{_NODE}"].to_dataset()
        xr.testing.assert_equal(grafted["depth"].load(), _group(a, _NODE)["depth"].load())
        assert f"tritonswmm/{_LINK}" in {p.lstrip("/") for p, _ in tree[member].subtree_with_keys}

    empty = master({"0": tmp_path / "gone0.zarr"})
    empty.analysis_paths.sensitivity_timeseries_zarr = tmp_path / "none.zarr"
    assert sens.consolidate_sensitivity_timeseries(empty) is None
    assert not (tmp_path / "none.zarr").exists()


def test_the_sensitivity_datatree_consolidation_calls_the_time_series_graft_only_when_the_toggle_is_on():
    """Property: `consolidate_sensitivity_datatree` contains a call to `consolidate_sensitivity_timeseries` that sits
    under an `if` whose test reads `toggle_consolidate_timeseries`.

    Input class: the master's one write path. The graft is correct in isolation and unreachable if the call is
    missing, which no other test observes; an ungated call would write an empty-or-absent store for every master.
    This is a structural pin on the function's syntax tree.
    Killed by: deleting the call, or hoisting it out of the `if`. Passes under a second correct implementation that
    moves the gated call into a helper only if this test is pointed at that helper.
    """
    import ast
    import inspect
    import textwrap

    from hhemt.sensitivity_analysis import TRITONSWMM_sensitivity_analysis as sens

    tree = ast.parse(textwrap.dedent(inspect.getsource(sens.consolidate_sensitivity_datatree)))
    gated = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and "toggle_consolidate_timeseries" in ast.unparse(node.test):
            gated += [
                n
                for stmt in node.body
                for n in ast.walk(stmt)
                if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == "consolidate_sensitivity_timeseries"
            ]
    assert len(gated) == 1
