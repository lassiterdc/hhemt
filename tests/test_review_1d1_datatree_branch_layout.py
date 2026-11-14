"""Reviewer tests of the two branches per group in analysis_datatree.zarr.

No solver and no Snakemake: they run on the committed multi-sim golden through the coder's own builder helpers, or on
small zarr stores written here. Each test quantifies over a class the coder's 30 tests position differently.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from tests.test_datatree_branch_layout import (
    NODE_TS,
    _master,
    _member,
    _open,
    _ready,
    _series_hashes,
    _tree_path,
)

CALENDAR_GROUP = "tritonswmm/swmm_node/timeseries_legacy_calendar"


@pytest.fixture(autouse=True)
def _pin_clean_build(monkeypatch):
    """Force the build-stamp term of the consolidation reuse conjunction TRUE (a dirty checkout is not the subject)."""
    import hhemt.provenance as prov

    monkeypatch.setattr(prov, "store_build_mismatch", lambda store, stamp: None)


@pytest.fixture
def consolidated(golden_extractor):
    """A golden analysis rebuilt at layout 31 with summaries and series in the one tree."""
    a = _ready(golden_extractor)
    a.process.consolidate_to_datatree()
    return a


def _add_legacy_calendar_group(store: Path) -> None:
    """Give the tree the group V0031 leaves when an analysis holds a version-1 series and no newer copy."""
    from hhemt.utils import deferred_consolidation

    with deferred_consolidation(store):
        shutil.copytree(store / NODE_TS, store / CALENDAR_GROUP)


def test_a_group_other_than_summary_and_timeseries_never_reaches_the_system_tree(consolidated, tmp_path):
    """Property: the system tree and the sensitivity master's summary tree copy `summary` branches and nothing else,
    whatever other branch a group holds.
    Class: NEW CAPABILITY. Plausible wrong implementation: excluding the one branch NAME `timeseries` (the landed
        filters `rsplit("/", 1)[-1] == "timeseries"`) rather than selecting the `summary` branch; red under it, output
        pasted in the reviewer's scratch file.
    Input class: every analysis tree whose group holds a third branch. V0031 produces one for every analysis that
        held a version-1 calendar-axis series and no version-2 copy (`timeseries_legacy_calendar`, D13 layout text).
        Reachability: `_move_legacy_calendar_groups` renames a `tritonswmm/swmm_{kind}_timeseries` group to
        `{kind}/timeseries_legacy_calendar` when `{kind}/timeseries` is absent; the artifact is that function.
    Why required: the chunk brief says `processing_system.py` grafts `summary` branches only and a system or master
        tree carries no series; a calendar-axis series copied into either holds series data under a group the
        readers of the `summary` path never look at and that nothing clears.
    Killed by: a filter on the branch name `timeseries` only. Both-states anchor: the assertion is on the set of
        branch names copied, present before and after the change.
    Passes under a second correct implementation: a graft that selects nodes whose own name is `summary` through
        `DataTree.filter`, or one that walks `_MODE_TO_TREE_PATH` values: the copied set is the summary groups.
    Invocation isolation: the red is a function of the data the test writes into the tree, not of how pytest is
        invoked; no path, flag or environment variable is involved.
    """
    from hhemt.processing_system import TRITONSWMM_system_post_processing

    store = _tree_path(consolidated)
    _add_legacy_calendar_group(store)
    assert CALENDAR_GROUP in {p.lstrip("/") for p, n in _open(consolidated).subtree_with_keys if n.has_data}, "control"

    system_store = tmp_path / "system_datatree.zarr"
    system = SimpleNamespace(
        sys_paths=SimpleNamespace(system_datatree_zarr=system_store),
        _analysis=consolidated,
        log=SimpleNamespace(write=lambda: None),
    )
    consolidated.sensitivity = None
    TRITONSWMM_system_post_processing(system).consolidate_system_datatree()
    copied = {
        p.lstrip("/")
        for p, n in xr.open_datatree(system_store, engine="zarr", consolidated=True).subtree_with_keys
        if n.has_data
    }
    assert copied, "control: the system tree copies something"
    assert all(p.endswith("/summary") for p in copied), sorted(p for p in copied if not p.endswith("/summary"))


def test_a_group_other_than_summary_and_timeseries_never_reaches_the_master_summary_tree(consolidated):
    """Property: the sensitivity master's summary tree grafts `summary` branches and nothing else, whatever other
    branch a member's group holds.
    Class: NEW CAPABILITY. Plausible wrong implementation: the landed graft skips the one branch NAME `timeseries`
        (`rel.rsplit("/", 1)[-1] == "timeseries"`), so a `timeseries_legacy_calendar` branch is grafted into the
        master's summary tree; red under it, output pasted in the reviewer's scratch file.
    Input class and reachability: as the sibling test above (a member analysis migrated with a version-1 series
        and no version-2 copy); the artifact is `build_sensitivity_datatree`'s `subtree_with_keys` loop.
    Why required: the master's summary tree is read at `member_{id}/{family}/{kind}/summary`; the series go to
        `sensitivity_timeseries.zarr` (chunk brief, fifth bullet).
    Killed by: a filter on the name `timeseries` only.
    Passes under a second correct implementation: a graft that selects nodes named `summary`.
    Invocation isolation: the red is a function of the data written into the member tree, not of the invocation.
    """
    from hhemt.sensitivity_analysis import TRITONSWMM_sensitivity_analysis as sens

    store = _tree_path(consolidated)
    _add_legacy_calendar_group(store)
    master = _master({"0": _member(store), "1": _member(store)})
    tree = sens.build_sensitivity_datatree(master)
    grafted = {p.lstrip("/") for p, n in tree.subtree_with_keys if n.has_data and p.lstrip("/").startswith("member_")}
    assert grafted, "control: the master tree grafts something"
    assert all(p.endswith("/summary") for p in grafted), sorted(p for p in grafted if not p.endswith("/summary"))


def test_a_consolidation_killed_while_appending_the_series_is_completed_by_the_next_one_with_each_event_once(
    consolidated, monkeypatch
):
    """Property: a consolidation killed after its summary groups are written and while the second event's series is
    being appended leaves a tree with no stamp, and the next consolidation leaves every event once in the series
    and the summaries current and stamped.
    Class: NEW CAPABILITY. Plausible wrong implementation: stamping the root before the series block, or planning the
        series by group presence rather than by event, which appends the first event twice (`[0, 1, 1]`, the hazard
        the landed `_plan_timeseries_write` names); red under it, output pasted in the reviewer's scratch file.
    Input class: the kill position the coder's kill tests do not reach (theirs are inside the summary writes).
        Reachability: `consolidate_timeseries` runs inside `_write_summary_branches` after every summary group, and
        a SLURM time limit can land at any append; the artifact is `_write_summary_branches`'s block order.
    Why required: the stamp-last order is the only thing that makes the interrupted tree recognisable, and it must
        hold for the whole block, series included.
    Killed by: stamping before the series block; a plan that appends an event whose slice is already present.
    Passes under a second correct implementation: any writer whose root attributes are written after the last series
        append, whatever it does for the summary groups.
    Invocation isolation: the kill is a patched function on `processing_analysis.append_zarr`; the red does not
        depend on the checkout path or the pytest invocation.
    """
    import hhemt.processing_analysis as pa
    from hhemt.processing_analysis import TRITONSWMM_analysis_post_processing as pp
    from hhemt.utils import zarr_root_attrs

    store = _tree_path(consolidated)
    n_events = len(consolidated.df_sims.index)
    assert n_events >= 2, "control: the golden holds at least two events"
    # Start from a tree that holds no series, so the kill lands inside the first series append of a rebuild.
    shutil.rmtree(store)
    real = pa.append_zarr
    appended: list[str] = []

    def boom(ds, store_, **kw):
        """Pass every write through; raise on the second event-dimension append of the series."""
        if kw.get("append_dim") == "event_iloc":
            appended.append(kw.get("group", ""))
            if len(appended) == 2:
                raise RuntimeError("killed during the series append")
        return real(ds, store_, **kw)

    monkeypatch.setattr(pa, "append_zarr", boom)
    with pytest.raises(RuntimeError, match="killed during the series append"):
        consolidated.process.consolidate_to_datatree()
    assert "consolidation_version" not in zarr_root_attrs(store), "an interrupted tree must carry no stamp"
    assert (store / "tritonswmm" / "swmm_node" / "summary").is_dir(), "control: the summaries were written first"

    monkeypatch.setattr(pa, "append_zarr", real)
    consolidated.process.consolidate_to_datatree()
    node = _open(consolidated)[NODE_TS].to_dataset().load()
    assert list(node["event_iloc"].values) == list(consolidated.df_sims.index)
    assert zarr_root_attrs(store).get("consolidation_version") == pp.CONSOLIDATION_VERSION
    assert _series_hashes(store), "control: the series files exist"


def test_a_rebuild_removes_the_summary_of_a_mode_the_analysis_no_longer_produces_and_keeps_every_series(
    consolidated, monkeypatch
):
    """Property: a rebuild removes a `summary` group whose mode is no longer produced, leaves every other summary
    group readable, and leaves both timeseries branches byte-identical.
    Class: NEW CAPABILITY. Plausible wrong implementation: removing the stale mode's PARENT group
        (`{family}/{kind}`), which also removes its `timeseries` branch; red under it, output pasted in the
        reviewer's scratch file.
    Input class: a group with a `summary` of a mode the analysis stopped producing AND a series beside it
        (`triton_only/triton` is the planted example; the planted `timeseries` stands for the coupled series that a
        toggle change leaves behind).
    Reachability: `_write_summary_branches` deletes `set(_MODE_TO_TREE_PATH.values()) - written`; a model toggle
        turned off after a first consolidation reaches it.
    Why required: with clearing on, the series exist nowhere else; a stale-summary removal that reaches a series
        destroys the only copy.
    Killed by: deleting `Path(store) / family / kind` rather than the `summary` path.
    Passes under a second correct implementation: leaving the stale `summary` group in place and re-stamping, which
        fails the first assertion below, so it is NOT a second correct implementation; a removal through
        `delete_summary_branches` of that one path is.
    Invocation isolation: the red is a function of the planted directories, not of the invocation.
    """
    import hhemt.processing_analysis as pa
    from hhemt.processing_analysis import TRITONSWMM_analysis_post_processing as pp
    from hhemt.utils import deferred_consolidation

    store = _tree_path(consolidated)
    stale_mode = "swmm_only_link"
    stale_parent = store / "swmm_only" / "swmm_link"
    live = store / "tritonswmm" / "swmm_node" / "summary"
    assert (stale_parent / "summary").is_dir() and live.is_dir(), "control: the golden produced both modes"
    if not (stale_parent / "timeseries").exists():
        # The golden's own consolidation writes this branch for the SWMM-only family, so nothing is
        # planted; the series asserted kept below is then the real one.
        with deferred_consolidation(store):
            shutil.copytree(store / NODE_TS, stale_parent / "timeseries")
    planted = {
        str(p.relative_to(stale_parent)): p.read_bytes()
        for p in (stale_parent / "timeseries").rglob("*")
        if p.is_file()
    }
    series = _series_hashes(store)
    stale_attr = pp._MODE_CONFIG[stale_mode][0]
    # The stale mode's SERIES store is unbound too: with it bound, the rebuild would re-write the `timeseries` branch
    # from the per-scenario store, and a removal of the whole parent group would pass unseen.
    stale_series_attr = pp._TIMESERIES_MODE_CONFIG["swmm_only_swmm_link_ts"]
    real_scenario = pa.TRITONSWMM_scenario

    class _NoStaleStore:
        """Scenario paths whose stale mode store is unbound, as when its model was switched off."""

        def __init__(self, real):
            self._real = real

        def __getattr__(self, name):
            return None if name in (stale_attr, stale_series_attr) else getattr(self._real, name)

    class _ScenarioProxy:
        """Delegates to the real scenario except for ``scen_paths``."""

        def __init__(self, real):
            self._real = real
            self.scen_paths = _NoStaleStore(real.scen_paths)

        def __getattr__(self, name):
            return getattr(self._real, name)

    def scenario(*args, **kwargs):
        """A scenario whose path object reports the stale mode's store as unbound."""
        return _ScenarioProxy(real_scenario(*args, **kwargs))

    monkeypatch.setattr(pa, "TRITONSWMM_scenario", scenario)
    monkeypatch.setattr(pp, "_consolidation_inputs_fingerprint", lambda self: "inputs-after-a-toggle")
    consolidated.process.consolidate_to_datatree()
    assert not (stale_parent / "summary").exists(), "the stale mode's summary is removed"
    assert {
        str(p.relative_to(stale_parent)): p.read_bytes()
        for p in (stale_parent / "timeseries").rglob("*")
        if p.is_file()
    } == planted, "the series beside a stale summary is untouched"
    assert _series_hashes(store) == series
    assert (stale_parent / "zarr.json").is_file(), "the group that still holds a series stays a group"
    assert live.is_dir()


def test_the_migration_moves_every_family_and_kind_it_finds_and_a_second_run_changes_no_byte(tmp_path):
    """Property: V0031's tree restructure moves each populated `{family}/{kind}` group of ANY family to
    `{kind}/summary`, folds a layout-30 series store into `timeseries`, keeps a calendar-axis group beside it under
    `timeseries_legacy_calendar` only when no newer copy exists, lists every group through the default consolidated
    open, and a second run changes no byte.
    Class: NEW CAPABILITY. Plausible wrong implementation: moving the `tritonswmm` family only (the coder's fixtures
        populate that family and the master), which leaves `triton_only` and `swmm_only` groups at the layout-30
        path; red under it, output pasted in the reviewer's scratch file.
    Input class: a store with three families, a `performance` kind, a kind absent from one family, and a
        calendar-axis group in a tree whose series store holds the OTHER kind only.
    Reachability: `_TREE_FAMILIES` names the three families and an analysis that ran `triton_only` and `swmm_only`
        models writes them; the artifact is `_MODE_TO_TREE_PATH` in `processing_analysis.py`.
    Why required: a group left at the layout-30 path is read by no layout-31 reader and refused by the writer.
    Killed by: a loop over a subset of families or kinds.
    Passes under a second correct implementation: a move driven by walking the store's directories rather than by
        the named tuples.
    Invocation isolation: the store is built from numpy arrays in `tmp_path`; the result does not depend on the tree
        the run resolves from.
    """
    import hashlib

    from hhemt.utils import consolidate_zarr_store
    from hhemt.version_migration.context import _migrate_datatree_branches

    def ds(value: float) -> xr.Dataset:
        """A small summary-shaped dataset."""
        return xr.Dataset({"max_x": (("event_iloc", "k"), np.full((2, 3), value))}, coords={"event_iloc": [0, 1]})

    store = tmp_path / "analysis_datatree.zarr"
    layout30 = [
        "tritonswmm/triton",
        "tritonswmm/swmm_node",
        "tritonswmm/performance",
        "triton_only/triton",
        "triton_only/performance",
        "swmm_only/swmm_node",
        "swmm_only/swmm_link",
    ]
    for i, group in enumerate(layout30):
        ds(float(i)).to_zarr(store, group=group, mode="a", consolidated=False)
    ds(99.0).to_zarr(
        store, group="tritonswmm/swmm_link_timeseries", mode="a", consolidated=False
    )  # calendar axis, link
    ds(98.0).to_zarr(
        store, group="tritonswmm/swmm_node_timeseries", mode="a", consolidated=False
    )  # calendar axis, node
    series = tmp_path / "analysis_timeseries.zarr"
    ds(98.0).to_zarr(series, group="tritonswmm/swmm_node_timeseries", mode="w", consolidated=False)
    consolidate_zarr_store(series)
    consolidate_zarr_store(store)

    _migrate_datatree_branches(store, series_source=series, master=False)
    tree = xr.open_datatree(store, engine="zarr", consolidated=True)
    data_groups = sorted(p.lstrip("/") for p, n in tree.subtree_with_keys if n.has_data)
    assert [f"{g}/summary" for g in layout30] == [g for g in data_groups if g.endswith("/summary")] or sorted(
        f"{g}/summary" for g in layout30
    ) == [g for g in data_groups if g.endswith("/summary")]
    assert "tritonswmm/swmm_node/timeseries" in data_groups, "the series store was folded into the node timeseries"
    assert "tritonswmm/swmm_link/timeseries_legacy_calendar" in data_groups, "no newer link copy exists: kept"
    assert "tritonswmm/swmm_node/timeseries_legacy_calendar" not in data_groups, "a newer node copy exists: deleted"
    assert not series.exists()
    assert not [g for g in data_groups if "__v30_moving" in g or g in layout30]

    def digest() -> dict[str, str]:
        """Hash every file of the store."""
        return {
            str(p.relative_to(store)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(store.rglob("*"))
            if p.is_file()
        }

    first = digest()
    _migrate_datatree_branches(store, series_source=None, master=False)
    assert digest() == first
