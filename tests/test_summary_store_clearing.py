"""The per-scenario summary stores are cleared after a verified copy, and the ``summary`` branch only gains events.

No solver and no Snakemake. Each test extracts the committed multi-sim golden (a finished analysis tree written by the
independent generator: three events, per-scenario summary and series stores), switches the clearing on or off in its
analysis config and runs the real consolidation. Every row names its class, its control (the input class that makes a
broken implementation read as correct), the change that kills it and a second correct implementation.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import zarr

from hhemt import processing_analysis as pa
from hhemt.exceptions import ConfigurationError, ProcessingError
from tests.fixtures.golden_trees import construct_golden_analysis

_PP = pa.TRITONSWMM_analysis_post_processing
_NODE = "tritonswmm/swmm_node/summary"
_TRITON = "tritonswmm/triton/summary"
_NODE_STEM = "TRITONSWMM_SWMM_node_summary"
_TRITON_STEM = "TRITONSWMM_TRITON_summary"
_FAMILY_TOGGLES = {
    "tritonswmm": "toggle_tritonswmm_model",
    "swmm": "toggle_swmm_model",
    "triton": "toggle_triton_model",
}


@pytest.fixture(autouse=True)
def _pin_clean_build(monkeypatch):
    """Force the build-stamp term of the consolidation reuse conjunction TRUE, so a dirty checkout does not make every
    consolidation rebuild for a reason unrelated to what these tests assert."""
    import hhemt.provenance as prov

    monkeypatch.setattr(prov, "store_build_mismatch", lambda store, stamp: None)


def _ready(golden_extractor, *, clear: bool, families=("tritonswmm",), cfg_extra: str = ""):
    """Golden analysis enabling exactly ``families``, series consolidation on, the clearing switch as given."""
    from hhemt.scenario import TRITONSWMM_scenario, coupling_nodes_file_for

    system_dir = golden_extractor("multisim")
    system_cfg = system_dir / "system_config.yaml"
    text = system_cfg.read_text()
    for family, key in _FAMILY_TOGGLES.items():
        text = re.sub(rf"^{key}: .*$", f"{key}: {'true' if family in families else 'false'}", text, flags=re.M)
    system_cfg.write_text(text)
    cfg = system_dir / "analysis_config.yaml"
    cfg.write_text(
        cfg.read_text()
        + "toggle_consolidate_timeseries: true\n"
        + f"toggle_clear_scenario_timeseries_after_consolidation: {'true' if clear else 'false'}\n"
        + cfg_extra
    )
    analysis = construct_golden_analysis(system_dir)
    shutil.rmtree(analysis.analysis_paths.analysis_datatree_zarr, ignore_errors=True)
    if "tritonswmm" in families:
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


def _store(a) -> Path:
    return Path(a.analysis_paths.analysis_datatree_zarr)


def _group(a, name: str) -> xr.Dataset:
    return xr.open_zarr(_store(a), group=name, consolidated=False, chunks=None).load()


def _slug(a, event_iloc: int) -> str:
    return a.process._event_slug(event_iloc)


def _own(a, event_iloc: int, stem: str) -> Path:
    return _scen(a, event_iloc).scen_paths.sim_folder / "processed" / f"{stem}.zarr"


def _record(a, event_iloc: int, model: str = "tritonswmm"):
    field = getattr(_scen(a, event_iloc).get_log(model), "consolidated_summaries_cleared", None)
    return None if field is None else field.get()


def _set_record(a, event_iloc: int, value, model: str = "tritonswmm") -> None:
    log = _scen(a, event_iloc).get_log(model)
    field = log.consolidated_summaries_cleared
    if field is None:
        _PP._log_field(log, "consolidated_summaries_cleared")
        field = log.consolidated_summaries_cleared
    field.clear() if value is None else field.set(value)


def _snapshot(a, tmp_path: Path) -> Path:
    """Copy every per-scenario ``*_summary.zarr`` store, so cleared values can be compared with what was there."""
    snap = tmp_path / "snapshot"
    for event_iloc in a.df_sims.index:
        sim = _scen(a, event_iloc).scen_paths.sim_folder
        for src in (sim / "processed").glob("*_summary.zarr"):
            shutil.copytree(src, snap / _slug(a, event_iloc) / src.name)
    return snap


def _restore(a, snap: Path, event_iloc: int) -> None:
    """Put an event's per-scenario summary stores back and remove its clearing record (the event is uncleared)."""
    sim = _scen(a, event_iloc).scen_paths.sim_folder
    for src in (snap / _slug(a, event_iloc)).glob("*.zarr"):
        shutil.rmtree(sim / "processed" / src.name, ignore_errors=True)
        shutil.copytree(src, sim / "processed" / src.name)
    _set_record(a, event_iloc, None)


def _rewrite_store(path: Path, edit) -> None:
    """Replace a per-scenario store by an edited copy of itself."""
    ds = xr.open_zarr(path, consolidated=False, chunks=None).load()
    shutil.rmtree(path)
    edit(ds).to_zarr(path, mode="w", consolidated=False)


def _event_chunks(store: Path, group: str, position: int) -> dict[str, str]:
    """Hash of every chunk file of ``group`` that holds only event ``position`` (event_iloc is the first axis).

    Unit: one chunk file. Matcher: the path component after ``c`` equals ``position``.
    """
    out = {}
    for p in sorted((store / group).rglob("*")):
        parts = p.relative_to(store / group).parts
        if p.is_file() and "c" in parts and parts[parts.index("c") + 1] == str(position):
            out[str(p.relative_to(store))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def _group_hashes(store: Path, group: str) -> dict[str, str]:
    return {
        str(p.relative_to(store)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((store / group).rglob("*"))
        if p.is_file()
    }


def _bump(store: Path, group: str, variable: str, position: int) -> None:
    """Add 1.0 to ``variable`` at one event of a stored group, behind the pipeline's back."""
    arr = zarr.open_group(store, mode="r+")[f"{group}/{variable}"]
    arr[position : position + 1] = np.asarray(arr[position : position + 1]) + 1.0


def _cleared(golden_extractor, tmp_path, **kwargs):
    """A golden analysis consolidated with the clearing on, and the snapshot taken before it."""
    a = _ready(golden_extractor, clear=True, **kwargs)
    snap = _snapshot(a, tmp_path)
    a.process.consolidate_to_datatree()
    return a, snap


def _add_flag(a) -> None:
    """Give every scenario's node summary the coupling-node flag, as a coupled summary written by a newer build has."""

    def edit(ds):
        return ds.assign(is_coupling_node=("node_id", np.zeros(ds.sizes["node_id"], dtype="int8")))

    for event_iloc in a.df_sims.index:
        _rewrite_store(_own(a, event_iloc, _NODE_STEM), edit)


def _two_then_three(golden_extractor, tmp_path, *, edit_new=None, with_flag=False, **kwargs):
    """Consolidate events 0 and 1 with the clearing on, then add event 2 (optionally with its node store edited)."""
    a = _ready(golden_extractor, clear=True, **kwargs)
    if with_flag:
        _add_flag(a)
    full = a.df_sims
    a.df_sims = full.loc[[0, 1]]
    a.process.consolidate_to_datatree()
    a.df_sims = full
    if edit_new is not None:
        _rewrite_store(_own(a, 2, _NODE_STEM), edit_new)
    return a


def _spy_writes(monkeypatch) -> list[tuple[str, str]]:
    """Record every ``(group, mode)`` the consolidation passes to ``append_zarr``."""
    seen: list[tuple[str, str]] = []
    real = pa.append_zarr

    def spy(ds, store, *, group, **kwargs):
        seen.append((str(group), str(kwargs.get("mode"))))
        return real(ds, store, group=group, **kwargs)

    monkeypatch.setattr(pa, "append_zarr", spy)
    return seen


# --------------------------------------------------------------------------------------------------------------------
# The clearing and its read-back
# --------------------------------------------------------------------------------------------------------------------


def test_with_the_switch_on_every_scenarios_summary_stores_are_deleted_after_the_read_back(golden_extractor, tmp_path):
    """Property: with the switch on, each scenario's per-scenario summary stores of the enabled model are deleted and
    its model log records ``done`` with each stem, the group it fed and the variable names the group received; the
    branch values equal the snapshot taken before.
    Class: NEW CAPABILITY. Plausible wrong implementation: no clearing, or a record without the variable names.
    Reachability: the real ``consolidate_to_datatree`` over the golden. Both-states anchor:
    ``test_with_the_switch_off_no_summary_store_is_cleared`` keeps the stores of the same golden. Killed by: either.
    Passes under a second correct implementation that records the names in a sidecar file.
    """
    a, snap = _cleared(golden_extractor, tmp_path)
    for event_iloc in a.df_sims.index:
        record = _record(a, event_iloc)
        assert record["state"] == "done", record
        assert set(record["stems"]) == {
            _TRITON_STEM,
            _NODE_STEM,
            "TRITONSWMM_SWMM_link_summary",
            "TRITONSWMM_perf_summary",
        }
        assert "max_flood_depth" in record["stems"][_TRITON_STEM][_TRITON]
        assert "depth_max" in record["stems"][_NODE_STEM][_NODE]
        for stem in record["stems"]:
            assert not _own(a, event_iloc, stem).exists(), f"{stem} survived the clearing"
        before = xr.open_zarr(snap / _slug(a, event_iloc) / f"{_NODE_STEM}.zarr", consolidated=False).load()
        stored = _group(a, _NODE).sel(event_iloc=[event_iloc])
        assert np.array_equal(
            stored["depth_max"].values.reshape(before["depth_max"].shape), before["depth_max"].values, equal_nan=True
        )


def test_a_slice_that_fails_its_read_back_keeps_its_stores_and_raises_naming_the_scenario(
    golden_extractor, tmp_path, monkeypatch
):
    """Property: one value of one event's stored ``summary`` slice changed after the write keeps that scenario's
    summary stores and raises ``ProcessingError`` naming the scenario; scenarios processed before it stay cleared.
    Class: NEW CAPABILITY. Plausible wrong implementation: deleting the stores without comparing, or comparing the
    group with itself. Must-fire control: the same corruption of the same value is the whole input; the uncorrupted
    run is ``test_with_the_switch_on_...``. Killed by: a clearing with no read-back.
    Passes under a second correct implementation that compares a content hash of the slice.
    """
    a = _ready(golden_extractor, clear=True)
    real = _PP._clear_summary_stores

    def corrupt_then_clear(self, store, **kwargs):
        _bump(store, _NODE, "depth_max", 1)
        return real(self, store, **kwargs)

    monkeypatch.setattr(_PP, "_clear_summary_stores", corrupt_then_clear)
    with pytest.raises(ProcessingError, match=re.escape(_slug(a, 1))):
        a.process.consolidate_to_datatree()
    assert _own(a, 1, _NODE_STEM).is_dir(), "the stores of the failing scenario were deleted"
    assert _record(a, 1) is None
    assert _record(a, 0)["state"] == "done"


def test_the_read_back_admits_the_outer_join_fill_and_nothing_else(golden_extractor, tmp_path, monkeypatch):
    """Property: a scenario whose node summary lacks a variable the group carries reads back equal when that variable
    is all-NaN at its event, and the clearing proceeds; the same scenario with a non-NaN value in that variable at its
    event fails the read-back.
    Class: NEW CAPABILITY. Plausible wrong implementation: requiring an identical variable set, which refuses the
    legitimate fill, or ignoring the variables the scenario lacks, which admits a stray value. Must-fire control: the
    second half. Killed by: either. Passes under a second correct implementation that compares the union of names.
    """
    a = _ready(golden_extractor, clear=True)
    _rewrite_store(_own(a, 2, _NODE_STEM), lambda ds: ds.drop_vars("max_total_inflow"))
    a.process.consolidate_to_datatree()
    assert _record(a, 2)["state"] == "done"
    assert "max_total_inflow" not in _record(a, 2)["stems"][_NODE_STEM][_NODE]
    assert _group(a, _NODE)["max_total_inflow"].sel(event_iloc=2).isnull().all()
    assert _group(a, _NODE)["max_total_inflow"].sel(event_iloc=0).notnull().any()

    b = _ready(_second(tmp_path), clear=True)
    _rewrite_store(_own(b, 2, _NODE_STEM), lambda ds: ds.drop_vars("max_total_inflow"))
    real = _PP._clear_summary_stores

    def stray_then_clear(self, store, **kwargs):
        zarr.open_group(store, mode="r+")[f"{_NODE}/max_total_inflow"][2:3] = 5.0
        return real(self, store, **kwargs)

    monkeypatch.setattr(_PP, "_clear_summary_stores", stray_then_clear)
    with pytest.raises(ProcessingError, match=re.escape(_slug(b, 2))):
        b.process.consolidate_to_datatree()
    assert _own(b, 2, _NODE_STEM).is_dir()


def _second(tmp_path: Path):
    """An extractor for a second copy of the golden in one test, under a different directory name."""
    from tests.fixtures.golden_trees import extract_golden

    def extractor(kind: str) -> Path:
        return extract_golden(kind, tmp_path / f"second_{kind}")

    return extractor


def test_with_the_switch_off_no_summary_store_is_cleared(golden_extractor, tmp_path):
    """Property: with the switch explicitly False every per-scenario summary store survives the consolidation and no
    record is written. Control: the switch on clears them (the first row).
    Class: NEW CAPABILITY. Plausible wrong implementation: clearing summaries whenever consolidation is on. Killed by:
    that. Passes under a second correct implementation that reads the switch at another site.
    """
    a = _ready(golden_extractor, clear=False)
    a.process.consolidate_to_datatree()
    for event_iloc in a.df_sims.index:
        assert _record(a, event_iloc) is None
        assert _own(a, event_iloc, _NODE_STEM).is_dir()
        assert _own(a, event_iloc, _TRITON_STEM).is_dir()


# --------------------------------------------------------------------------------------------------------------------
# The group rule
# --------------------------------------------------------------------------------------------------------------------


def test_the_whole_group_is_rewritten_only_while_every_store_exists(golden_extractor, tmp_path):
    """Property: with no event cleared a changed per-scenario value is rewritten whole and the variable-set
    heterogeneity coordinate is recomputed over all events; with one event cleared the same change takes the per-event
    rewrite of that event's slice and the cleared event's chunk files are byte-identical before and after.
    Class: NEW CAPABILITY. Plausible wrong implementation: always rewriting the group whole, which replaces a cleared
    event's only copy. Must-fire control: the two halves differ only in whether an event is cleared. Killed by: an
    unconditional whole-group write. Passes under a second correct implementation that always plans per event.
    """
    a = _ready(golden_extractor, clear=False)
    a.process.consolidate_to_datatree()
    _rewrite_store(_own(a, 2, _NODE_STEM), lambda ds: ds.drop_vars("max_total_inflow"))
    a.log.datatree_consolidation_complete.set(False)
    a.process.consolidate_to_datatree()
    coord = _group(a, _NODE)["variable_set_across_members"].values
    assert "max_total_inflow" in str(coord[2]) and "LACKS" in str(coord[2])

    b = _ready(_second(tmp_path), clear=True)
    snap = _snapshot(b, tmp_path)
    b.process.consolidate_to_datatree()
    for event_iloc in (1, 2):
        _restore(b, snap, event_iloc)
    before = _event_chunks(_store(b), _NODE, 0)
    assert before
    _rewrite_store(_own(b, 2, _NODE_STEM), lambda ds: ds.assign(depth_max=ds["depth_max"] + 1.0))
    b.log.datatree_consolidation_complete.set(False)
    b.process.consolidate_to_datatree()
    assert _event_chunks(_store(b), _NODE, 0) == before
    assert _record(b, 0)["state"] == "done"
    assert float(_group(b, _NODE)["depth_max"].sel(event_iloc=2).max()) > 0


def test_a_cleared_event_is_kept_and_a_fingerprint_change_only_restamps_the_root(
    golden_extractor, tmp_path, monkeypatch
):
    """Property: after clearing, a consolidation whose inputs fingerprint changed re-stamps the root attributes and
    writes no ``summary`` group (every summary chunk file byte-identical). Control: with the switch off the same
    change rewrites the groups (a whole-group write per summary group).
    Class: NEW CAPABILITY. Plausible wrong implementation: a whole-group rewrite from the branch, which is safe only
    by luck, or one that drops cleared events. Killed by: any summary-group write. Passes under a second correct
    implementation that skips the write when nothing differs.
    """
    a, _ = _cleared(golden_extractor, tmp_path)
    before = {
        g: _group_hashes(_store(a), g) for g in pa.TRITONSWMM_analysis_post_processing._MODE_TO_TREE_PATH.values()
    }
    monkeypatch.setattr(_PP, "_consolidation_inputs_fingerprint", lambda self: "changed")
    writes = _spy_writes(monkeypatch)
    a.process.consolidate_to_datatree()
    summary_writes = [w for w in writes if w[0].endswith("/summary")]
    assert not summary_writes, summary_writes
    for group, hashes in before.items():
        if hashes:
            assert _group_hashes(_store(a), group) == hashes
    assert zarr.open_group(_store(a), mode="r").attrs["consolidation_inputs_fingerprint"] == "changed"

    b = _ready(_second(tmp_path), clear=False)
    b.process.consolidate_to_datatree()
    monkeypatch.setattr(_PP, "_consolidation_inputs_fingerprint", lambda self: "changed again")
    writes_b = _spy_writes(monkeypatch)
    b.process.consolidate_to_datatree()
    assert [w for w in writes_b if w == (_NODE, "w")], writes_b


def test_appending_after_clearing_creates_a_new_variable_nan_over_the_cleared_events(
    golden_extractor, tmp_path, monkeypatch
):
    """Property: after a clearing, a new scenario whose summary carries a variable the append-only group lacks appends
    without a refusal; the variable is NaN at every cleared event and equal to its source at the new one, and every
    chunk file of the group's other variables for the cleared events is byte-identical. Must-fire control: the same
    append with the NaN-filled array not created first leaves a group that raises ``ValueError`` (conflicting
    sizes) when it is read.
    Class: NEW CAPABILITY. Plausible wrong implementation: appending the event as is, which raises on the mismatched
    sizes, or refusing the event. Killed by: either. Passes under a second correct implementation that rewrites
    the group with the variable added.
    """
    a = _two_then_three(
        golden_extractor,
        tmp_path,
        edit_new=lambda ds: ds.assign(brand_new=ds["depth_max"] * 2.0),
    )
    assert _record(a, 0)["state"] == "done"
    held = {p: _event_chunks(_store(a), _NODE, p) for p in (0, 1)}
    piece = xr.open_zarr(_own(a, 2, _NODE_STEM), consolidated=False, chunks=None).load()
    control = tmp_path / "control.zarr"
    shutil.copytree(_store(a), control)
    a.log.datatree_consolidation_complete.set(False)
    a.process.consolidate_to_datatree()
    grown = _group(a, _NODE)
    assert grown["brand_new"].sel(event_iloc=[0, 1]).isnull().all()
    assert float(grown["brand_new"].sel(event_iloc=2).max()) > 0
    for position, hashes in held.items():
        now = _event_chunks(_store(a), _NODE, position)
        assert {k: v for k, v in now.items() if "/brand_new/" not in k} == hashes

    piece = _PP._fit_strings(control, _NODE, piece.assign_coords(event_iloc=[2]).drop_vars("event_id", errors="ignore"))
    with pytest.raises(ValueError, match="conflicting sizes"):
        with pa.deferred_consolidation(control):
            pa.append_zarr(piece, control, group=_NODE, consolidate=False, mode="a", append_dim="event_iloc")
        xr.open_zarr(control, group=_NODE, consolidated=False).load()


def test_appending_after_clearing_pads_a_variable_the_new_event_lacks(golden_extractor, tmp_path):
    """Property: after a clearing, a new scenario lacking a variable the group carries appends without a refusal, NaN
    at the new event, and its heterogeneity coordinate names the missing variable. Must-fire control: the same
    scenario added to a group with no clearing recorded takes the whole-group rewrite and the coordinate names the
    same variable.
    Class: NEW CAPABILITY. Plausible wrong implementation: refusing a variable-set mismatch after clearing (the
    earlier ruling) or leaving the coordinate unset. Killed by: either. Passes under a second correct implementation
    that rewrites the new event's whole row.
    """
    drop = lambda ds: ds.drop_vars("max_total_inflow")  # noqa: E731
    a = _two_then_three(golden_extractor, tmp_path, edit_new=drop)
    a.log.datatree_consolidation_complete.set(False)
    a.process.consolidate_to_datatree()
    grown = _group(a, _NODE)
    assert grown["max_total_inflow"].sel(event_iloc=2).isnull().all()
    assert grown["max_total_inflow"].sel(event_iloc=[0, 1]).notnull().any()
    assert "max_total_inflow" in str(grown["variable_set_across_members"].values[2])

    b = _ready(_second(tmp_path), clear=False)
    _rewrite_store(_own(b, 2, _NODE_STEM), drop)
    b.process.consolidate_to_datatree()
    assert "max_total_inflow" in str(_group(b, _NODE)["variable_set_across_members"].values[2])


def test_every_summary_array_is_chunked_one_event_per_chunk_and_an_append_rewrites_no_earlier_chunk(
    golden_extractor, tmp_path
):
    """Property: every array of every ``summary`` group a consolidation writes has an ``event_iloc`` chunk length of
    1, and appending one event leaves every earlier chunk file byte-identical. Must-fire control: the same append to a
    copy of a group chunked 2 along ``event_iloc`` rewrites the chunk file that holds the last earlier event.
    Class: NEW CAPABILITY. Plausible wrong implementation: the default chunking of the dask arrays. Killed by: any
    chunk longer than 1. Passes under a second correct implementation that chunks by shard.
    """
    a = _two_then_three(golden_extractor, tmp_path)
    root = zarr.open_group(_store(a), mode="r")
    for group in pa.TRITONSWMM_analysis_post_processing._MODE_TO_TREE_PATH.values():
        if group not in root:
            continue
        for name, arr in root[group].arrays():
            dims = arr.metadata.dimension_names
            if dims and "event_iloc" in dims:
                assert arr.chunks[dims.index("event_iloc")] == 1, (group, name, arr.chunks)
    held = {p: _event_chunks(_store(a), _NODE, p) for p in (0, 1)}
    a.log.datatree_consolidation_complete.set(False)
    a.process.consolidate_to_datatree()
    assert {p: _event_chunks(_store(a), _NODE, p) for p in (0, 1)} == held

    ds = xr.Dataset({"v": (("event_iloc", "n"), np.arange(8.0).reshape(4, 2))}, coords={"event_iloc": [0, 1, 2, 3]})
    two = tmp_path / "two.zarr"
    ds.isel(event_iloc=slice(0, 3)).to_zarr(two, mode="w", encoding={"v": {"chunks": (2, 2)}}, consolidated=False)
    seen = {str(p): p.read_bytes() for p in two.rglob("*") if p.is_file() and "c" in p.parts}
    ds.isel(event_iloc=[3]).to_zarr(two, mode="a", append_dim="event_iloc", consolidated=False)
    after = {str(p): p.read_bytes() for p in two.rglob("*") if p.is_file() and "c" in p.parts}
    assert any(after[k] != seen[k] for k in seen), "a chunk of 2 events was left untouched by the append"


def test_a_cleared_event_that_leaves_the_analysis_refuses_and_a_stored_event_without_a_record_is_dropped(
    golden_extractor, tmp_path
):
    """Property: removing a cleared scenario from the analysis raises naming it and its group, and the group is
    unchanged. Control: removing a scenario whose stores exist, with no event cleared, rebuilds without it.
    Class: NEW CAPABILITY. Plausible wrong implementation: a whole-group rewrite that silently drops the cleared
    event, or a refusal of every removal. Killed by: either. Passes under a second correct implementation that
    refuses in the planner instead of before the write.
    """
    a, _ = _cleared(golden_extractor, tmp_path)
    before = _group_hashes(_store(a), _NODE)
    a.df_sims = a.df_sims.loc[[0, 1]]
    a.log.datatree_consolidation_complete.set(False)
    with pytest.raises(ProcessingError, match=re.escape(_slug_of_full(a, 2))):
        a.process.consolidate_to_datatree()
    assert _group_hashes(_store(a), _NODE) == before

    b = _ready(_second(tmp_path), clear=False)
    b.process.consolidate_to_datatree()
    b.df_sims = b.df_sims.loc[[0, 1]]
    b.log.datatree_consolidation_complete.set(False)
    b.process.consolidate_to_datatree()
    assert [int(v) for v in _group(b, _NODE)["event_iloc"].values] == [0, 1]


def _slug_of_full(a, event_iloc: int) -> str:
    return f"event_index.{event_iloc}"


# --------------------------------------------------------------------------------------------------------------------
# The completion predicate and its consumers
# --------------------------------------------------------------------------------------------------------------------


def test_the_completion_predicate_reads_the_record_and_constructs_no_scenario(golden_extractor, tmp_path, monkeypatch):
    """Property: ``scenario_summaries_present`` is True for a scenario whose stores are gone and whose model log
    records ``pending``, and for one recording ``done``, constructing no scenario; the same scenario with the record
    removed is False.
    Class: NEW CAPABILITY. Plausible wrong implementation: reading the record through ``TRITONSWMM_scenario`` (a
    constructor that creates directories) or not reading it. Must-fire control: the record removed. Killed by: either.
    Passes under a second correct implementation that reads the record from the analysis log.
    """
    from hhemt.scenario import TRITONSWMM_scenario
    from hhemt.summary_paths import analysis_summaries_complete, scenario_summaries_present

    a, _ = _cleared(golden_extractor, tmp_path)
    slug = _slug(a, 0)
    record = _record(a, 0)

    def refuse(self, *args, **kwargs):
        raise AssertionError("a scenario was constructed")

    monkeypatch.setattr(TRITONSWMM_scenario, "__init__", refuse)
    assert scenario_summaries_present(a, slug, ["tritonswmm"])
    assert analysis_summaries_complete(a, ["tritonswmm"])
    monkeypatch.undo()
    _set_record(a, 0, {"state": "pending", "stems": record["stems"]})
    assert scenario_summaries_present(a, slug, ["tritonswmm"])
    _set_record(a, 0, None)
    assert not scenario_summaries_present(a, slug, ["tritonswmm"])


def test_the_self_heal_deletes_no_flag_of_a_cleared_scenario(golden_extractor, tmp_path):
    """Property: ``_reconcile_stale_process_flags_against_summaries`` on an analysis whose scenarios' summaries are
    cleared returns an empty set and every ``d_process`` flag remains. Must-fire control: a scenario whose stores are
    gone with NO record has its flag deleted.
    Class: NEW CAPABILITY. Plausible wrong implementation: reading only the store paths, which re-processes every
    cleared scenario. Killed by: that. Passes under a second correct implementation that skips cleared scenarios
    before the flag test.
    """
    a, _ = _cleared(golden_extractor, tmp_path)
    status = Path(a.analysis_paths.analysis_dir) / "_status"
    status.mkdir(exist_ok=True)
    flags = {i: status / f"d_process_tritonswmm_evt-{_slug(a, i)}_complete.flag" for i in a.df_sims.index}
    for flag in flags.values():
        flag.write_text("")
    assert a._reconcile_stale_process_flags_against_summaries() == set()
    assert all(flag.exists() for flag in flags.values())
    _set_record(a, 1, None)
    assert a._reconcile_stale_process_flags_against_summaries() == {(_slug(a, 1), "tritonswmm")}
    assert not flags[1].exists() and flags[0].exists()


def test_the_reprocess_target_enumeration_and_the_analysis_reclaim_include_a_cleared_event(golden_extractor, tmp_path):
    """Property: ``reprocess_snakefile_generator._available_event_ids`` includes a cleared event and
    ``reclaim_unconsolidated_scenarios`` reaches a cleared event past its summaries skip. Must-fire control: the same
    event with its record removed is excluded from the first and skipped at the second.
    Class: NEW CAPABILITY. Plausible wrong implementation: a cleared event dropped from the report targets. Killed
    by: that. Passes under a second correct implementation that reads the predicate once per run.
    """
    from hhemt.reprocess_snakefile_generator import _available_event_ids

    a, _ = _cleared(golden_extractor, tmp_path)
    all_ids = [_slug(a, i) for i in a.df_sims.index]
    assert _available_event_ids(a, enabled_models=["tritonswmm"], all_event_ids=all_ids) == all_ids
    removed = _record(a, 2)
    _set_record(a, 2, None)
    assert _available_event_ids(a, enabled_models=["tritonswmm"], all_event_ids=all_ids) == all_ids[:2]
    _set_record(a, 2, removed)


def test_the_data_availability_check_reports_the_summary_clearing_as_a_reclaim(golden_extractor, tmp_path):
    """Property: ``check_data_availability`` on a cleared analysis passes and lists the summary-clearing class.
    Must-fire control: the same analysis with the records removed fails, reporting each scenario as damage.
    Class: NEW CAPABILITY. Plausible wrong implementation: not listing the class, or reading the cleared stores as
    absent damage. Killed by: either. Passes under a second correct implementation that labels the class differently.
    """
    from hhemt.analysis_validation import check_data_availability

    a, _ = _cleared(golden_extractor, tmp_path)
    verdict = check_data_availability(a)
    assert verdict.passed, verdict.details
    assert "per-scenario summary stores" in verdict.summary
    records = {i: _record(a, i) for i in a.df_sims.index}
    for i in records:
        _set_record(a, i, None)
    broken = check_data_availability(a)
    assert not broken.passed and len(broken.details) == len(records)


# --------------------------------------------------------------------------------------------------------------------
# The branch read
# --------------------------------------------------------------------------------------------------------------------


def test_the_combined_output_reads_a_cleared_event_from_the_branch(golden_extractor, tmp_path):
    """Property: for a mix of cleared and present events ``_retrieve_combined_output`` returns, per event, values
    equal to the pre-clearing per-scenario store's, with exactly the recorded variables. Control: a cleared event whose
    slice is missing from the group raises naming it.
    Class: NEW CAPABILITY. Plausible wrong implementation: raising ``FileNotFoundError`` for the missing store.
    Killed by: that. Passes under a second correct implementation that reads every event from the branch.
    """
    a = _ready(golden_extractor, clear=True)
    snap = _snapshot(a, tmp_path)
    a.process.consolidate_to_datatree()
    _restore(a, snap, 2)
    combined = a.process._retrieve_combined_output("tritonswmm_swmm_node")
    for event_iloc in (0, 1, 2):
        before = xr.open_zarr(snap / _slug(a, event_iloc) / f"{_NODE_STEM}.zarr", consolidated=False).load()
        got = combined.sel(event_iloc=[event_iloc])
        for name in _record(a, 0)["stems"][_NODE_STEM][_NODE]:
            want = before[name].values.reshape(got[name].shape)
            assert _PP._arrays_equal(np.asarray(got[name].values), np.asarray(want)), (event_iloc, name)
    assert "is_coupling_node" not in combined.data_vars
    _set_record(a, 0, {"state": "done", "stems": {_NODE_STEM: {_NODE: ["depth_max", "never_stored"]}}})
    with pytest.raises(FileNotFoundError, match=re.escape(_slug(a, 0))):
        a.process._retrieve_combined_output("tritonswmm_swmm_node")


def test_the_forcing_tail_check_reads_a_cleared_triton_summary(golden_extractor, tmp_path, monkeypatch):
    """Property: ``check_forcing_tail_influence`` on a cleared scenario examines it through the branch read. Must-fire
    control: with the branch read raising ``FileNotFoundError`` the same scenario reads indeterminate.
    Class: NEW CAPABILITY. Plausible wrong implementation: globbing the deleted store, which counts every cleared
    scenario indeterminate. Killed by: that. Passes under a second correct implementation that reads the group
    directly.
    """
    from hhemt.analysis_validation import check_forcing_tail_influence

    a, _ = _cleared(golden_extractor, tmp_path)
    sims = Path(a.analysis_paths.simulation_directory)
    if not (sims / _slug(a, 0) / "sim_weather.nc").exists():
        pytest.skip("the golden carries no sim_weather.nc, so the check has no forced extent to compare against")
    ok = check_forcing_tail_influence(a)
    assert "examined 3" in ok.summary or "examined 0" not in ok.summary, ok.summary

    def refuse(self, mode, **kwargs):
        raise FileNotFoundError("branch read refused")

    monkeypatch.setattr(_PP, "_retrieve_combined_output", refuse)
    broken = check_forcing_tail_influence(a)
    assert "3 indeterminate" in broken.summary, broken.summary


# --------------------------------------------------------------------------------------------------------------------
# Recorded before the delete, and the resume
# --------------------------------------------------------------------------------------------------------------------


def test_a_clearing_killed_after_the_delete_resumes_to_done_and_one_missing_variable_refuses(
    golden_extractor, tmp_path, monkeypatch
):
    """Property: a clearing killed after the delete and before ``done`` resumes to ``done`` when every recorded group
    holds the event with every recorded variable. Control: the same state with one group missing a recorded variable
    refuses naming the scenario and the group.
    Class: NEW CAPABILITY. Plausible wrong implementation: deleting before recording, so a kill leaves neither the
    stores nor a record. Killed by: that. Passes under a second correct implementation that writes the record twice.
    """
    a, _ = _cleared(golden_extractor, tmp_path)
    record = _record(a, 0)
    _set_record(a, 0, {"state": "pending", "stems": record["stems"]})
    a.log.datatree_consolidation_complete.set(False)
    a.process.consolidate_to_datatree()
    assert _record(a, 0)["state"] == "done"

    bad = json.loads(json.dumps(record["stems"]))
    bad[_NODE_STEM][_NODE].append("never_stored")
    _set_record(a, 0, {"state": "pending", "stems": bad})
    a.log.datatree_consolidation_complete.set(False)
    with pytest.raises((ProcessingError, FileNotFoundError), match=re.escape(_slug(a, 0))) as err:
        a.process.consolidate_to_datatree()
    assert _NODE in str(err.value)


def test_a_half_deleted_store_with_a_pending_record_is_finished_and_never_read_back(golden_extractor, tmp_path):
    """Property: with half of a summary store's files deleted and the record ``pending``, the resume leaves every
    ``summary`` slice byte-identical, removes the rest of the store and sets ``done``. Control: the same half-deleted
    store with NO record is never read back as verified (the consolidation refuses it as incomplete or rewrites it
    whole).
    Class: NEW CAPABILITY. Plausible wrong implementation: comparing the group with the partly deleted store, which
    fails and answers by overwriting a verified slice. Killed by: that. Passes under a second correct implementation
    that deletes the rest first.
    """
    a = _ready(golden_extractor, clear=True)
    snap = _snapshot(a, tmp_path)
    a.process.consolidate_to_datatree()
    before = {g: _group_hashes(_store(a), g) for g in (_NODE, _TRITON)}
    record = _record(a, 1)
    _restore(a, snap, 1)
    store = _own(a, 1, _NODE_STEM)
    files = sorted(p for p in store.rglob("*") if p.is_file() and "c" in p.parts)
    for p in files[: len(files) // 2]:
        p.unlink()
    _set_record(a, 1, {"state": "pending", "stems": record["stems"]})
    a.log.datatree_consolidation_complete.set(False)
    a.process.consolidate_to_datatree()
    assert _record(a, 1)["state"] == "done"
    assert not store.exists()
    assert {g: _group_hashes(_store(a), g) for g in (_NODE, _TRITON)} == before

    _restore(a, snap, 1)
    files = sorted(p for p in _own(a, 1, _NODE_STEM).rglob("*") if p.is_file() and "c" in p.parts)
    for p in files[: len(files) // 2]:
        p.unlink()
    a.log.datatree_consolidation_complete.set(False)
    try:
        a.process.consolidate_to_datatree()
    except (FileNotFoundError, ProcessingError, ValueError):
        pass
    verified = _record(a, 1)
    assert verified is None or _own(a, 1, _NODE_STEM).exists() is False


# --------------------------------------------------------------------------------------------------------------------
# Reprocess
# --------------------------------------------------------------------------------------------------------------------


def test_the_reprocess_delete_keeps_the_summary_branches_after_clearing(golden_extractor, tmp_path):
    """Property: ``delete_summary_branches`` on an analysis with a recorded clearing removes the root attributes and
    leaves every ``summary`` and ``timeseries`` chunk file byte-identical. Control: on an analysis with no recorded
    clearing it removes the ``summary`` branches.
    Class: NEW CAPABILITY. Plausible wrong implementation: deleting the branches that now hold the only copy. Killed
    by: that. Passes under a second correct implementation that checks the record in the caller.
    """
    from hhemt.delete_reprocess_zarr_runner import delete_summary_branches
    from hhemt.utils import zarr_root_attrs

    a, _ = _cleared(golden_extractor, tmp_path)
    store = _store(a)
    held = {g: _group_hashes(store, g) for g in (_NODE, _TRITON, "tritonswmm/swmm_node/timeseries")}
    removed = delete_summary_branches(store, Path(a.analysis_paths.analysis_dir))
    assert removed == []
    assert "consolidation_version" not in zarr_root_attrs(store)
    assert {g: _group_hashes(store, g) for g in held} == held

    b = _ready(_second(tmp_path), clear=False)
    b.process.consolidate_to_datatree()
    gone = delete_summary_branches(_store(b), Path(b.analysis_paths.analysis_dir))
    assert gone and not (_store(b) / _NODE).exists()


def test_a_forced_reprocess_of_a_cleared_scenario_without_its_raw_source_refuses(golden_extractor, tmp_path):
    """Property: with the raw rebuild source absent, a forced process-stage re-run of a cleared scenario raises
    ``ConfigurationError`` naming the scenario. Must-fire control: the same scenario with its raw source present does
    not raise.
    Class: NEW CAPABILITY. Plausible wrong implementation: letting the forced re-run delete the flag and fail deep in
    a job. Killed by: that. Passes under a second correct implementation that checks inside the force-rerun resolver.
    """
    from hhemt.workflow import ResolvedForceRerunSpec

    a, _ = _cleared(golden_extractor, tmp_path)
    spec = ResolvedForceRerunSpec(scope="all", tokens=(), stage="process", models=("tritonswmm",))
    pairs = a._cleared_summary_pairs_for_force(spec)
    assert pairs == {(_slug(a, i), "tritonswmm") for i in a.df_sims.index}
    scen = _scen(a, 0)
    raw = scen.scen_paths.out_tritonswmm
    shutil.rmtree(raw, ignore_errors=True)
    with pytest.raises(ConfigurationError, match=re.escape(_slug(a, 0))):
        a._assert_reprocess_rebuild_sources_present(pairs, field="override_force_rerun")
    (raw / "performance").mkdir(parents=True, exist_ok=True)
    (raw / "performance" / "performance0.txt").write_text("x")
    one = {(_slug(a, 0), "tritonswmm")}
    a._assert_reprocess_rebuild_sources_present(one, field="override_force_rerun")


def test_a_rebuilt_summary_resets_the_record_and_the_next_consolidation_clears_it_again(
    golden_extractor, tmp_path, monkeypatch
):
    """Property: writing a rebuilt per-scenario summary removes ``consolidated_summaries_cleared``, and the next
    consolidation rewrites that event's slice from the new store, records ``done`` again and changes no other event's
    chunk file. Must-fire control: a rebuilt store holding one changed value changes that event's slice.
    Class: NEW CAPABILITY. Plausible wrong implementation: leaving the record, so readers keep the stale slice.
    Killed by: that. Passes under a second correct implementation that resets the record at the start of processing.
    """
    a = _ready(golden_extractor, clear=True)
    snap = _snapshot(a, tmp_path)
    a.process.consolidate_to_datatree()
    other = _event_chunks(_store(a), _NODE, 0)
    _restore(a, snap, 1)
    _set_record(a, 1, {"state": "done", "stems": {_NODE_STEM: {_NODE: ["depth_max"]}}})
    from hhemt.process_simulation import TRITONSWMM_sim_post_processing

    proc = TRITONSWMM_sim_post_processing(_scen(a, 1).run)
    for name in dir(proc):
        if name.startswith("_export_"):
            monkeypatch.setattr(proc, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(proc, "_prune_timeseries_stores", lambda *args, **kwargs: None)
    proc.write_summary_outputs(which="both", model_type="tritonswmm")
    assert _record(a, 1) is None
    _rewrite_store(_own(a, 1, _NODE_STEM), lambda ds: ds.assign(depth_max=ds["depth_max"] + 1.0))
    a.log.datatree_consolidation_complete.set(False)
    a.process.consolidate_to_datatree()
    assert _record(a, 1)["state"] == "done"
    assert float(_group(a, _NODE)["depth_max"].sel(event_iloc=1).max()) > 0.03
    assert _event_chunks(_store(a), _NODE, 0) == other


def test_an_unforced_process_reprocess_leaves_every_flag_of_a_cleared_analysis(golden_extractor, tmp_path):
    """Property: the self-heal of a process-stage reprocess of a cleared analysis deletes no flag and so emits no
    process rule for a cleared scenario. Must-fire control: a scenario whose stores are gone with no record is
    reconciled (its flag deleted, its process rule re-emitted).
    Class: NEW CAPABILITY. Plausible wrong implementation: treating the absent stores as a divergence. Killed by:
    that. Passes under a second correct implementation that skips the self-heal for recorded scenarios.
    """
    a, _ = _cleared(golden_extractor, tmp_path)
    status = Path(a.analysis_paths.analysis_dir) / "_status"
    status.mkdir(exist_ok=True)
    flag = status / f"d_process_tritonswmm_evt-{_slug(a, 0)}_complete.flag"
    flag.write_text("")
    assert a._reconcile_stale_process_flags_against_summaries() == set() and flag.exists()
    _set_record(a, 0, None)
    assert a._reconcile_stale_process_flags_against_summaries() == {(_slug(a, 0), "tritonswmm")}
    assert not flag.exists()


# --------------------------------------------------------------------------------------------------------------------
# SWMM-only: a stem feeding two groups; the coupling-node flag's one home
# --------------------------------------------------------------------------------------------------------------------


def test_a_stem_feeding_two_groups_is_recorded_and_read_back_per_group(golden_extractor, tmp_path, monkeypatch):
    """Property: a SWMM-only scenario consolidated and cleared records both ``swmm_only/swmm_node/summary`` and
    ``swmm_only/system/summary`` for ``SWMM_only_node_summary``, each with its own variable names. Must-fire control:
    one changed value of a ``system_*_max`` variable in the system group's slice fails the read-back and keeps the
    store.
    Class: NEW CAPABILITY. Plausible wrong implementation: one record per stem, which loses the second group.
    Killed by: that. Passes under a second correct implementation that records per mode.
    """
    from hhemt.constants import LST_SWMM_SYSTEM_SUMMARY_VARIABLES

    a = _ready(golden_extractor, clear=True, families=("swmm",))
    stem = "SWMM_only_node_summary"
    node_g, system_g = "swmm_only/swmm_node/summary", "swmm_only/system/summary"
    has_system = set(LST_SWMM_SYSTEM_SUMMARY_VARIABLES) & set(
        xr.open_zarr(_own(a, 0, stem), consolidated=False).data_vars
    )
    if not has_system:
        pytest.skip("the golden's SWMM-only node summary carries no system reductions")
    a.process.consolidate_to_datatree()
    record = _record(a, 0, "swmm")
    assert set(record["stems"][stem]) == {node_g, system_g}, record
    assert set(record["stems"][stem][system_g]) <= set(LST_SWMM_SYSTEM_SUMMARY_VARIABLES)
    assert not set(record["stems"][stem][node_g]) & set(LST_SWMM_SYSTEM_SUMMARY_VARIABLES)

    b = _ready(_second(tmp_path), clear=True, families=("swmm",))
    real = _PP._clear_summary_stores
    name = sorted(has_system)[0]

    def bump_then_clear(self, store, **kwargs):
        _bump(store, system_g, name, 1)
        return real(self, store, **kwargs)

    monkeypatch.setattr(_PP, "_clear_summary_stores", bump_then_clear)
    with pytest.raises(ProcessingError, match=re.escape(_slug(b, 1))):
        b.process.consolidate_to_datatree()
    assert _own(b, 1, stem).is_dir()


def test_the_coupling_node_flag_has_one_home_under_clearing_and_keeps_it_when_the_switch_is_turned_off(
    golden_extractor, tmp_path
):
    """Property: with the switch on, ``is_coupling_node`` is in ``tritonswmm/swmm_node/timeseries`` and not in
    ``tritonswmm/swmm_node/summary``, and the record's variable list omits it; after a clearing, a new scenario
    consolidated with the switch explicitly False appends to the summary group without the flag and no refusal.
    Must-fire control: with the switch off and no clearing recorded the flag is in both groups.
    Class: NEW CAPABILITY. Plausible wrong implementation: keying the exclusion on the switch alone, which brings the
    flag back as a new array when the switch is turned off. Killed by: that. Passes under a second correct
    implementation that drops the flag from the series instead.
    """
    a = _two_then_three(golden_extractor, tmp_path, with_flag=True)
    assert "is_coupling_node" in _group(a, "tritonswmm/swmm_node/timeseries").data_vars
    assert "is_coupling_node" not in _group(a, _NODE).data_vars
    assert "is_coupling_node" not in _record(a, 0)["stems"][_NODE_STEM][_NODE]
    a.cfg_analysis.toggle_clear_scenario_timeseries_after_consolidation = False
    a.log.datatree_consolidation_complete.set(False)
    a.process.consolidate_to_datatree()
    assert "is_coupling_node" not in _group(a, _NODE).data_vars
    assert [int(v) for v in _group(a, _NODE)["event_iloc"].values] == [0, 1, 2]

    b = _ready(_second(tmp_path), clear=False)
    _add_flag(b)
    b.process.consolidate_to_datatree()
    assert "is_coupling_node" in _group(b, _NODE).data_vars
    assert "is_coupling_node" in _group(b, "tritonswmm/swmm_node/timeseries").data_vars


def test_the_state_machine_value_is_a_json_mapping_a_reader_without_hhemt_can_parse(golden_extractor, tmp_path):
    """Property: the record is stored in the model log as a plain JSON mapping with ``state`` and ``stems``, so the
    path-only predicate reads it with ``json`` alone. Control: a record with an unknown state reads as absent.
    Class: REGRESSION on the predicate's contract. Plausible wrong implementation: a LogField repr in the file.
    Killed by: that. Passes under a second correct implementation with another key spelling only if both readers move.
    """
    from hhemt.summary_paths import parse_summary_clearing_record, summary_clearing_record

    a, _ = _cleared(golden_extractor, tmp_path)
    sim = _scen(a, 0).scen_paths.sim_folder
    payload = json.loads((sim / "log_tritonswmm.json").read_text())
    assert payload["consolidated_summaries_cleared"]["state"] == "done"
    assert summary_clearing_record(sim, "tritonswmm") == payload["consolidated_summaries_cleared"]
    assert parse_summary_clearing_record({"state": "later", "stems": {}}) is None
    assert parse_summary_clearing_record({"state": "done"}) is None


# --------------------------------------------------------------------------------------------------------------------
# The sensitivity master: a real analysis stands in as the one member, the master is the stand-in of the master tests
# --------------------------------------------------------------------------------------------------------------------


def _master(a, monkeypatch):
    """A master over the analysis ``a`` as its only member, whose node name is the analysis id (no prefix)."""
    from tests.test_sensitivity_store_no_redundancy import _Master

    class _Flat(_Master):
        member_prefix = ""

    member_id = str(a.cfg_analysis.analysis_id)
    root = Path(a.analysis_paths.analysis_dir).parent.parent
    master = _Flat(root, {member_id: a})
    monkeypatch.setattr(
        "hhemt.sensitivity_analysis._first_scenario_paths", lambda analysis: _scen(analysis, 0).scen_paths
    )
    return master, member_id


def test_the_master_holds_the_members_summaries_after_clearing_and_the_member_keeps_nothing(
    golden_extractor, tmp_path, monkeypatch
):
    """Property: with the switch on, after the master copies the member, the member's per-scenario summaries and tree
    are gone and ``{member}/{family}/{kind}/summary`` holds values equal to copies taken before the clearing. Must-fire
    control: with the switch off the member tree and the stores remain.
    Class: NEW CAPABILITY. Plausible wrong implementation: the member's clearing leaving the master with a group the
    member tree alone held. Killed by: that. Passes under a second correct implementation that copies per event.
    """
    a = _ready(golden_extractor, clear=True)
    snap = _snapshot(a, tmp_path)
    a.process.consolidate_to_datatree()
    master, member_id = _master(a, monkeypatch)
    master.run()
    assert not _store(a).exists()
    assert all(not _own(a, i, _NODE_STEM).exists() for i in a.df_sims.index)
    held = xr.open_zarr(master.store, group=f"{member_id}/{_NODE}", consolidated=True, chunks=None).load()
    for i in a.df_sims.index:
        before = xr.open_zarr(snap / _slug(a, i) / f"{_NODE_STEM}.zarr", consolidated=False).load()
        assert _PP._arrays_equal(
            held["depth_max"].sel(event_iloc=i).values.ravel(), before["depth_max"].values.ravel()
        ), i

    b = _ready(_second(tmp_path), clear=False)
    b.process.consolidate_to_datatree()
    master_b, _ = _master(b, monkeypatch)
    master_b.run()
    assert _store(b).exists() and _own(b, 0, _NODE_STEM).is_dir()


def test_the_master_keeps_cleared_member_slices_on_a_rebuild_and_the_branch_read_follows_the_moved_member(
    golden_extractor, tmp_path, monkeypatch
):
    """Property: after clearing, a fingerprint change leaves every chunk file of the member's summary group in the
    master byte-identical, and ``_retrieve_combined_output`` on the moved member returns the pre-clearing values from
    the
    master's tree. Must-fire control: with the member's move record removed the same call raises naming the member's
    absent tree; with the switch off the same fingerprint change rewrites the group.
    Class: NEW CAPABILITY. Plausible wrong implementation: a whole-group rewrite of the member's group in the master,
    or a reader that looks only in the member's own tree. Killed by: either.
    Passes under a second correct implementation that rebuilds nothing when no event differs.
    """
    a = _ready(golden_extractor, clear=True)
    snap = _snapshot(a, tmp_path)
    a.process.consolidate_to_datatree()
    master, member_id = _master(a, monkeypatch)
    master.run()
    group = f"{member_id}/{_NODE}"
    before = _group_hashes(master.store, group)
    monkeypatch.setattr(a.process, "_consolidation_inputs_fingerprint", lambda: "changed")
    master.run()
    assert _group_hashes(master.store, group) == before

    combined = a.process._retrieve_combined_output("tritonswmm_swmm_node")
    snapped = xr.open_zarr(snap / _slug(a, 1) / f"{_NODE_STEM}.zarr", consolidated=False).load()
    assert _PP._arrays_equal(
        combined["depth_max"].sel(event_iloc=1).values.ravel(), snapped["depth_max"].values.ravel()
    )
    a.log.member_tree_moved_to_master.clear()
    with pytest.raises(FileNotFoundError, match="absent"):
        a.process._retrieve_combined_output("tritonswmm_swmm_node")


def test_a_partial_reprocess_removes_only_the_stems_it_rewrote_from_the_record(golden_extractor, tmp_path, monkeypatch):
    """Property: after ``write_summary_outputs(which="SWMM")`` on a cleared coupled scenario whose node and link
    stores are back, the record names only the TRITON and performance stems, ``scenario_summaries_present`` stays True
    and a forced re-consolidation completes and clears the returned stores again. Must-fire control: the record reset
    whole leaves the scenario with no record, ``scenario_summaries_present`` False and the TRITON store absent.
    Class: REGRESSION on the pre-change reset, which cleared the whole record. Plausible wrong implementation: the
    whole reset. Killed by: that. Passes under a second correct implementation that rewrites the record per stem.
    """
    from hhemt.process_simulation import TRITONSWMM_sim_post_processing
    from hhemt.summary_paths import scenario_summaries_present

    a = _ready(golden_extractor, clear=True)
    snap = _snapshot(a, tmp_path)
    a.process.consolidate_to_datatree()
    for stem in (_NODE_STEM, "TRITONSWMM_SWMM_link_summary"):
        shutil.copytree(snap / _slug(a, 1) / f"{stem}.zarr", _own(a, 1, stem))
    proc = TRITONSWMM_sim_post_processing(_scen(a, 1).run)
    for name in dir(proc):
        if name.startswith("_export_"):
            monkeypatch.setattr(proc, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(proc, "_prune_timeseries_stores", lambda *args, **kwargs: None)
    proc.write_summary_outputs(which="SWMM", model_type="tritonswmm")
    record = _record(a, 1)
    assert set(record["stems"]) == {_TRITON_STEM, "TRITONSWMM_perf_summary"}, record
    assert scenario_summaries_present(a, _slug(a, 1), ["tritonswmm"])
    a.log.datatree_consolidation_complete.set(False)
    a.process.consolidate_to_datatree()
    assert set(_record(a, 1)["stems"]) == {
        _TRITON_STEM,
        _NODE_STEM,
        "TRITONSWMM_SWMM_link_summary",
        "TRITONSWMM_perf_summary",
    }
    assert not _own(a, 1, _NODE_STEM).exists()

    _set_record(a, 1, None)
    assert not scenario_summaries_present(a, _slug(a, 1), ["tritonswmm"])
    assert not _own(a, 1, _TRITON_STEM).exists()


def test_a_forced_reprocess_names_the_cleared_pairs_of_sensitivity_members_from_their_own_sims(tmp_path):
    """Property: on a bare two-member layout, a sensitivity primary under scope ``all`` names every member's cleared
    pair as ``member_{id}/{event_id}``, a member-scoped spec names only that member's, and a pair whose member has no
    raw source refuses naming it. Must-fire control: the same layout with the records removed names no pair.
    Class: NEW CAPABILITY. Plausible wrong implementation: looking only in the primary's empty ``sims``, which names
    no pair. Killed by: that. Passes under a second correct implementation that asks each member object.
    """
    import json
    from types import SimpleNamespace

    from hhemt.analysis import TRITONSWMM_analysis
    from hhemt.workflow import ResolvedForceRerunSpec

    record = {
        "state": "done",
        "stems": {"TRITONSWMM_TRITON_summary": {"tritonswmm/triton/summary": ["max_flood_depth"]}},
    }
    for member in ("0", "1"):
        sim = tmp_path / "members" / f"member_{member}" / "sims" / "event_index.0"
        sim.mkdir(parents=True)
        (sim / "log_tritonswmm.json").write_text(json.dumps({"consolidated_summaries_cleared": record}))
    primary = SimpleNamespace(
        analysis_paths=SimpleNamespace(analysis_dir=tmp_path, simulation_directory=tmp_path / "sims"),
        df_sims=[],
        analysis_config_yaml=tmp_path / "analysis_config.yaml",
    )
    pairs = TRITONSWMM_analysis._cleared_summary_pairs_for_force
    assert_sources = TRITONSWMM_analysis._assert_reprocess_rebuild_sources_present
    allspec = ResolvedForceRerunSpec(scope="all", tokens=(), stage="process", models=("tritonswmm",))
    named = pairs(primary, allspec)
    assert named == {(f"member_{m}/event_index.0", "tritonswmm") for m in ("0", "1")}
    one = ResolvedForceRerunSpec(scope="member", tokens=("1",), stage="process", models=("tritonswmm",))
    assert pairs(primary, one) == {("member_1/event_index.0", "tritonswmm")}
    with pytest.raises(ConfigurationError, match="member_1/event_index.0"):
        assert_sources(primary, pairs(primary, one), field="override_force_rerun")
    raw = tmp_path / "members" / "member_1" / "sims" / "event_index.0" / "out_tritonswmm" / "performance"
    raw.mkdir(parents=True)
    (raw / "performance0.txt").write_text("x")
    assert_sources(primary, pairs(primary, one), field="override_force_rerun")
    for log_file in tmp_path.glob("members/*/sims/*/log_*.json"):
        log_file.write_text("{}")
    assert pairs(primary, allspec) == set()
