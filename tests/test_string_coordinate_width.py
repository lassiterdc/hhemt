"""String coordinates appended along ``event_iloc`` keep a fixed width, so event ids of different length stack.

A zarr string array takes the width of the first value written, and an append of another width raises
``Mismatched dtypes``. Real event ids differ in length (``year.8_..._event_id.1`` against
``year.11_..._compound_event_id.3``) and the goldens' ids do not, so nothing caught it. No solver and no Snakemake:
the committed multi-sim golden is consolidated in-process with the event-id slug replaced by the ids a row names.
"""

from __future__ import annotations

import pytest
import xarray as xr

from tests.fixtures.golden_trees import construct_golden_analysis

_NODE = "tritonswmm/swmm_node/timeseries"
_PERF = "tritonswmm/performance/timeseries"
_IDS = {
    "short": "year.8_event_type.rain_event_id.1",
    "long": "year.11_event_type.compound_event_id.3",
    "mid": "year.16_event_type.surge_event_id.4",
}


def _ready(golden_extractor):
    """Golden analysis with series consolidation on and clearing off, a coupling list beside every scenario."""
    from hhemt.scenario import TRITONSWMM_scenario, coupling_nodes_file_for

    system_dir = golden_extractor("multisim")
    cfg = system_dir / "analysis_config.yaml"
    cfg.write_text(
        cfg.read_text()
        + "toggle_consolidate_timeseries: true\ntoggle_clear_scenario_timeseries_after_consolidation: false\n"
    )
    analysis = construct_golden_analysis(system_dir)
    for event_iloc in analysis.df_sims.index:
        scen = TRITONSWMM_scenario(event_iloc, analysis)
        ts = xr.open_zarr(scen.scen_paths.output_tritonswmm_node_time_series, consolidated=False)
        nodes_file = coupling_nodes_file_for(scen.scen_paths.swmm_hydraulics_inp)
        nodes_file.parent.mkdir(parents=True, exist_ok=True)
        nodes_file.write_text("".join(f"{n}\n" for n in ts["node_id"].values[:3]))
    return analysis


def _name_events(monkeypatch, ids: list[str]) -> None:
    """Make event ``i`` carry ``ids[i]`` wherever the series consolidation names it."""
    import hhemt.processing_analysis as pa

    monkeypatch.setattr(
        pa.TRITONSWMM_analysis_post_processing, "_event_slug", lambda self, event_iloc: ids[int(event_iloc)]
    )


@pytest.mark.parametrize(
    "order",
    [("short", "long", "mid"), ("long", "mid", "short"), ("mid", "short", "long"), ("long", "short", "mid")],
    ids=["increasing-then-mid", "decreasing", "mid-first", "longest-first"],
)
def test_event_ids_of_different_length_land_in_any_order_and_read_back_equal(golden_extractor, monkeypatch, order):
    """Property: three real-shaped event ids of different length, appended in each of four orders (the first written
    is the shortest, the longest, or neither), land in the node and performance series groups and read back equal.

    Class: REGRESSION - red on the base tree with ``ValueError: Mismatched dtypes for variable event_id``. Reachability:
    the real ``consolidate_timeseries`` over the golden; only the slug is replaced. Both-states anchor: asserts the
    values read from the group, so it holds whatever width is stored. Invocation isolation: nothing depends on the
    working directory. Must-fire control: the orders place the longest id second and last, so a width fixed by the
    first value fails for at least three of the four.
    Killed by: width taken from the first event. Passes under a second correct implementation that widens every
    string coordinate to one constant.
    """
    ids = [_IDS[k] for k in order]
    a = _ready(golden_extractor)
    _name_events(monkeypatch, ids)
    a.process.consolidate_timeseries()
    for group in (_NODE, _PERF):
        stored = xr.open_zarr(a.analysis_paths.analysis_datatree_zarr, group=group, consolidated=False)
        assert [str(v) for v in stored["event_id"].values] == ids, group


def test_a_legacy_narrow_group_is_widened_in_place_and_an_id_no_width_holds_is_refused_naming_it(
    golden_extractor, monkeypatch
):
    """Property: a group created at a narrow width (a pre-fix group, built here with the floor lowered to 10) takes a
    longer id by being widened IN PLACE to the fixed width: the earlier ids read back unchanged, the data chunk files
    of the earlier events are byte-identical, and the new event lands. An id longer than the fixed width is refused
    before any write, naming the value, the group and the width.

    Class: NEW CAPABILITY. Plausible wrong implementations: refusing the older group with a remedy that does not
    work on it, a widening that rewrites data, and a cast that cuts the longer id. Reachability: the first pass uses
    the lowered floor over events 0 and 1 (event 2's node store hidden), the second runs with the real floor.
    Both-states anchor: asserts the stored ids and the bytes of the earlier events' data chunks. Invocation isolation:
    nothing depends on the working directory. Must-fire control: the group is asserted narrow before the second pass.
    Killed by: those. Passes under a second correct implementation that rebuilds the labels another way.
    """
    from pathlib import Path

    import hhemt.processing_analysis as pa

    cls = pa.TRITONSWMM_analysis_post_processing
    a = _ready(golden_extractor)
    ids = ["a.1_x_id.1", "a.2_id.3", "a.20_event_id.30"]
    _name_events(monkeypatch, ids)
    node_store = Path(_node_store(a, 2))
    away = node_store.with_name("hidden_" + node_store.name)
    node_store.rename(away)
    with monkeypatch.context() as narrow:
        narrow.setattr(cls, "_EVENT_STRING_WIDTH_DEFAULT", 10)
        with pytest.raises(FileNotFoundError):
            a.process.consolidate_timeseries()
    store = a.analysis_paths.analysis_datatree_zarr
    group_dir = store / _NODE
    stored = xr.open_zarr(store, group=_NODE, consolidated=False)
    assert stored["event_id"].dtype.itemsize // 4 == 10, "precondition: the group was created narrow"
    data_files = {
        str(f.relative_to(group_dir)): f.read_bytes()
        for f in group_dir.rglob("*")
        if f.is_file() and "event_id" not in f.parts and f.name != "zarr.json"
    }
    away.rename(node_store)
    a.process.consolidate_timeseries()
    stored = xr.open_zarr(store, group=_NODE, consolidated=False)
    assert [str(v) for v in stored["event_id"].values] == ids
    assert stored["event_id"].dtype.itemsize // 4 == 255
    for name, content in data_files.items():
        assert (group_dir / name).read_bytes() == content, f"{name} changed by the widening"
    assert not [p for p in group_dir.iterdir() if p.name.endswith((".widening", ".widened_aside"))]


def test_an_id_no_width_holds_is_refused_naming_the_value_and_the_group(golden_extractor, monkeypatch):
    """Property: an event id longer than the fixed width is refused before any write with the id, the group and the
    width in the message, and the events before it stay readable.

    Class: NEW CAPABILITY. Plausible wrong implementation: a cast that cuts the id, or a refusal naming a count only.
    Reachability: an id of 337 characters (constructed; real ids are 37 to 38) as the third event. Both-states anchor:
    asserts the message and the stored ids. Invocation isolation: nothing depends on the working directory.
    Must-fire control: the two events before the refused one landed.
    Killed by: those. Passes under a second correct implementation that names a truncation of at least 40 characters.
    """
    b = _ready(golden_extractor)
    too_long = "year.99_event_type.compound_event_id." + "9" * 300
    ids = [_IDS["short"], _IDS["long"], too_long]
    _name_events(monkeypatch, ids)
    with pytest.raises(Exception) as err:
        b.process.consolidate_timeseries()
    assert type(err.value).__name__ == "ProcessingError"
    text = str(err.value)
    assert too_long in text and _NODE in text and "255" in text, text
    stored = xr.open_zarr(b.analysis_paths.analysis_datatree_zarr, group=_NODE, consolidated=False)
    assert [str(v) for v in stored["event_id"].values] == ids[:2]


def _node_store(a, event_iloc: int):
    from hhemt.scenario import TRITONSWMM_scenario

    return TRITONSWMM_scenario(event_iloc, a).scen_paths.output_tritonswmm_node_time_series


def _series_source(root, name: str, ids: list[str]):
    """A one-group member tree ``{root}/{name}.zarr`` holding a ``timeseries`` group over events with ids ``ids``."""
    import numpy as np

    n = len(ids)
    ds = xr.Dataset(
        {"depth": (("event_iloc", "elapsed_s"), np.arange(float(n * 3)).reshape(n, 3))},
        coords={"event_iloc": list(range(n)), "elapsed_s": [10, 20, 30], "event_id": ("event_iloc", ids)},
    )
    path = root / f"{name}.zarr"
    ds.to_zarr(path, group="g/timeseries", mode="w", consolidated=False)
    return path


def test_the_master_copy_takes_event_ids_of_different_length_and_widens_a_narrow_legacy_group(tmp_path, monkeypatch):
    """Property: copying a member's series group into the sensitivity master twice, the second time with an extra
    event whose id is LONGER than the first copy's, lands the new event and reads back equal; against a master group
    created narrow (before widths were fixed) the longer id is taken by widening that group in place, and an id longer
    than the fixed width is refused with ``ProcessingError`` naming the value, the group and the width.

    Class: REGRESSION - red on the base tree with ``ValueError: Mismatched dtypes for variable event_id``.
    Reachability: the real ``_copy_series_group``; the narrow master is made by replacing the widening with the
    identity for the first copy. Both-states anchor: asserts the ids read from the master. Invocation isolation:
    nothing depends on the working directory. Must-fire control: the older master group is asserted narrow before
    the second copy, and an id over the fixed width is refused.
    Killed by: a copy that fits no width. Passes under a second correct implementation that rewrites the group.
    """
    from hhemt import sensitivity_analysis as sa
    from hhemt.exceptions import ProcessingError
    from hhemt.processing_analysis import TRITONSWMM_analysis_post_processing as pa

    first = _series_source(tmp_path, "first", ["e0"])
    second = _series_source(tmp_path, "second", ["e0", "year.11_event_type.compound_event_id.3"])

    master = tmp_path / "master_ok.zarr"
    xr.Dataset().to_zarr(master, mode="w", consolidated=False)
    sa._copy_series_group(first, "g/timeseries", master, "m/g/timeseries", 5, member_id="0")
    sa._copy_series_group(second, "g/timeseries", master, "m/g/timeseries", 5, member_id="0")
    stored = xr.open_zarr(master, group="m/g/timeseries", consolidated=False)
    assert [str(v) for v in stored["event_id"].values] == ["e0", "year.11_event_type.compound_event_id.3"]

    older = tmp_path / "master_legacy.zarr"
    xr.Dataset().to_zarr(older, mode="w", consolidated=False)
    with monkeypatch.context() as old:
        old.setattr(pa, "_widen_event_strings", staticmethod(lambda ds, group="": ds))
        sa._copy_series_group(first, "g/timeseries", older, "m/g/timeseries", 5, member_id="0")
    narrow = xr.open_zarr(older, group="m/g/timeseries", consolidated=False)
    assert narrow["event_id"].dtype.itemsize // 4 == 2, "control: the older master group is narrow"
    sa._copy_series_group(second, "g/timeseries", older, "m/g/timeseries", 5, member_id="0")
    widened = xr.open_zarr(older, group="m/g/timeseries", consolidated=False)
    assert [str(v) for v in widened["event_id"].values] == ["e0", "year.11_event_type.compound_event_id.3"]
    too_long = "x" * 300
    third = _series_source(tmp_path, "third", ["e0", "year.11_event_type.compound_event_id.3", too_long])
    with pytest.raises(ProcessingError) as err:
        sa._copy_series_group(third, "g/timeseries", older, "m/g/timeseries", 5, member_id="0")
    text = str(err.value)
    assert too_long in text and "m/g/timeseries" in text and "255" in text, text


def test_the_hydrology_summary_and_series_take_event_ids_of_different_length(golden_extractor, monkeypatch):
    """Property: the hydrology summary writer (group created by the first event) and the hydrology series writer take
    three events whose ids differ in length (short, long, mid) and read back with all three ids.

    Class: REGRESSION for the series writer (red on the base with ``Mismatched dtypes``); the summary create path
    already fixed its width and this row pins that. Reachability: the real ``_hydrology_write_summary`` and
    ``_hydrology_write_series`` on the golden's analysis object, called per event inside ``deferred_consolidation``.
    Both-states anchor: asserts the ids and one data value read back. Invocation isolation: nothing depends on the
    working directory. Must-fire control: the ids are in an order whose second member is longer than the first.
    Killed by: width taken from the first event. Passes under a second correct implementation that widens to a constant.
    """
    import numpy as np

    from hhemt.utils import deferred_consolidation

    a = _ready(golden_extractor)
    ids = [_IDS["short"], _IDS["long"], _IDS["mid"]]
    _name_events(monkeypatch, ids)
    store = a.analysis_paths.analysis_datatree_zarr
    proc = a.process
    for kind, group in (
        ("summary", "tritonswmm/swmm_node_hydro/summary"),
        ("series", "tritonswmm/swmm_node_hydro/timeseries"),
    ):
        for event_iloc in range(3):
            if kind == "summary":
                piece = xr.Dataset(
                    {"total_inflow_volume": (("event_iloc", "node_id"), np.full((1, 2), float(event_iloc)))},
                    coords={
                        "event_iloc": [event_iloc],
                        "node_id": ["a", "b"],
                        "event_id": ("event_iloc", [ids[event_iloc]]),
                    },
                )
                with deferred_consolidation(store):
                    proc._hydrology_write_summary(store, group, piece, event_iloc, 5)
            else:
                piece = xr.Dataset(
                    {"inflow": (("event_iloc", "elapsed_s", "node_id"), np.full((1, 3, 2), float(event_iloc)))},
                    coords={
                        "event_iloc": [event_iloc],
                        "elapsed_s": [10, 20, 30],
                        "node_id": ["a", "b"],
                        "event_id": ("event_iloc", [ids[event_iloc]]),
                    },
                )
                with deferred_consolidation(store):
                    proc._hydrology_write_series(store, group, piece, event_iloc)
        stored = xr.open_zarr(store, group=group, consolidated=False)
        assert [str(v) for v in stored["event_id"].values] == ids, group
