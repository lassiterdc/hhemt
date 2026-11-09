"""Layout 31: every group of ``analysis_datatree.zarr`` holds a ``summary`` and a ``timeseries`` branch.

No solver and no Snakemake. The consolidation runs on the committed multi-sim golden (a finished analysis tree written
by the independent generator) or on stand-ins that carry only what the code under test reads. Each arm names its
control: the input class that makes a broken implementation read as correct.
"""

from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tests.fixtures.golden_trees import construct_golden_analysis

SRC = Path(__file__).resolve().parents[1] / "src" / "hhemt"
NODE_TS = "tritonswmm/swmm_node/timeseries"
LINK_TS = "tritonswmm/swmm_link/timeseries"

_PP = __import__("hhemt.processing_analysis", fromlist=["x"]).TRITONSWMM_analysis_post_processing


class _Writer:
    """The member's own group writer, borrowed from the real post-processing, for a member that reports no clearing."""

    _MODE_TO_TREE_PATH = _PP._MODE_TO_TREE_PATH
    _mode_of_group = _PP._mode_of_group
    _prepare_summary_group = staticmethod(_PP._prepare_summary_group)
    _write_summary_group = _PP._write_summary_group

    def _summary_records(self, mode):
        return {}


def _ready(golden_extractor, *, clear: bool = False):
    """Golden analysis with series consolidation on and a coupling list beside every scenario."""
    from hhemt.scenario import TRITONSWMM_scenario, coupling_nodes_file_for

    system_dir = golden_extractor("multisim")
    cfg = system_dir / "analysis_config.yaml"
    extra = (
        "toggle_clear_scenario_timeseries_after_consolidation: true\n"
        if clear
        else "toggle_clear_scenario_timeseries_after_consolidation: false\n"
    )
    cfg.write_text(cfg.read_text() + "toggle_consolidate_timeseries: true\n" + extra)
    analysis = construct_golden_analysis(system_dir)
    # The golden carries a finished tree written at layout 30. Layout 31 refuses to write a summary branch inside a
    # layout-30 group, so the stale tree is removed and the consolidation builds a layout-31 one.
    shutil.rmtree(analysis.analysis_paths.analysis_datatree_zarr, ignore_errors=True)
    for event_iloc in analysis.df_sims.index:
        scen = TRITONSWMM_scenario(event_iloc, analysis)
        ts = xr.open_zarr(scen.scen_paths.output_tritonswmm_node_time_series, consolidated=False)
        nodes_file = coupling_nodes_file_for(scen.scen_paths.swmm_hydraulics_inp)
        nodes_file.parent.mkdir(parents=True, exist_ok=True)
        nodes_file.write_text("".join(f"{n}\n" for n in ts["node_id"].values[:3]))
    return analysis


def _tree_path(a) -> Path:
    return a.analysis_paths.analysis_datatree_zarr


def _hashes(store: Path, group: str) -> dict[str, str]:
    return {
        str(p.relative_to(store)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((store / group).rglob("*"))
        if p.is_file()
    }


def _series_hashes(store: Path) -> dict[str, str]:
    return _hashes(store, NODE_TS) | _hashes(store, LINK_TS)


def _open(a, *, consolidated: bool = True) -> xr.DataTree:
    return xr.open_datatree(_tree_path(a), engine="zarr", consolidated=consolidated)


def _summary_paths(a) -> list[str]:
    from hhemt.processing_analysis import TRITONSWMM_analysis_post_processing as pp

    return sorted(pp._MODE_TO_TREE_PATH.values())


@pytest.fixture(autouse=True)
def _pin_clean_build(monkeypatch):
    """Force the build-stamp term of the consolidation reuse conjunction TRUE, so a dirty checkout does not make
    every consolidation rebuild for a reason unrelated to what these tests assert."""
    import hhemt.provenance as prov

    monkeypatch.setattr(prov, "store_build_mismatch", lambda store, stamp: None)


@pytest.fixture
def consolidated(golden_extractor):
    """A golden analysis, rebuilt at layout 31, with its summaries and series consolidated into the one tree."""
    a = _ready(golden_extractor)
    a.process.consolidate_to_datatree()
    return a


def test_every_summary_mode_writes_at_its_summary_path_and_no_variable_sits_at_the_parent(consolidated):
    """Property: after a consolidation every populated summary mode is a data-bearing group at {family}/{kind}/summary
    and its parent group holds no data variable.
    Class: NEW CAPABILITY. Plausible wrong implementation: writing the summary at both the old path and the new one,
        or only at the old path. Control inside the test: the parent is a plain group while the child holds
        variables; red under the wrong implementation was not run on the pre-change tree in this round.
    Why required: layout 31 moves every summary under a branch, and a reader left on the parent path would read
        nothing.
    Killed by: a writer that keeps the layout-30 path; a map value without the summary suffix.
    Passes under a second correct implementation: a writer that writes each summary through a single DataTree call
        instead of per-group in-place writes: the paths and the empty parent are the same.
    Not established: the red-on-the-pre-change-tree output was not captured this round; the in-test control is the
        substitute.
    """
    from hhemt.processing_analysis import TRITONSWMM_analysis_post_processing as pp

    assert all(p.endswith("/summary") for p in pp._MODE_TO_TREE_PATH.values())
    tree = _open(consolidated)
    written = [p for p in _summary_paths(consolidated) if p in {q.lstrip("/") for q, _ in tree.subtree_with_keys}]
    assert written, "control: the constructed analysis populates at least one summary mode"
    for path in written:
        assert tree[path].to_dataset().data_vars, path
        parent = path.rsplit("/", 1)[0]
        assert not tree[parent].to_dataset().data_vars, f"{parent} must hold no data variable"


def test_the_series_land_at_timeseries_inside_the_one_store_and_no_other_store_exists(consolidated):
    """Property: the consolidated node and link series are groups of analysis_datatree.zarr at
    {family}/{kind}/timeseries, equal NaN-aware to the per-scenario series, and no separate analysis-level
    series store or path field exists.
    Class: NEW CAPABILITY. Plausible wrong implementation: appending to a second store beside the tree, or leaving
        the path field bound. Both-states anchor: the series values come from the scenario's own store, not from
        wording the change introduces.
    Why required: the layout keeps every series in the one tree; a second store would hold a duplicate and break the
        no-duplication rule.
    Killed by: binding analysis_timeseries_zarr again; appending under the layout-30 group names.
    Passes under a second correct implementation: appending each scenario through a batch of region writes instead
        of one append per event: the group paths and values are identical.
    Not established: the red-on-the-pre-change-tree output was not captured this round; the in-test control is the
        substitute.
    """
    from hhemt.scenario import TRITONSWMM_scenario

    tree = _open(consolidated)
    node = tree[NODE_TS].to_dataset().load()
    link = tree[LINK_TS].to_dataset().load()
    assert list(node["event_iloc"].values) == list(consolidated.df_sims.index)
    assert "depth" in node.data_vars and "flow" in link.data_vars
    own = xr.open_zarr(
        TRITONSWMM_scenario(0, consolidated).scen_paths.output_tritonswmm_node_time_series, consolidated=False
    ).load()
    first = node["depth"].isel(event_iloc=0).dropna("elapsed_s", how="all")
    assert np.array_equal(first.values, own["depth"].values[: first.shape[0]], equal_nan=True)
    assert not (_tree_path(consolidated).parent / "analysis_timeseries.zarr").exists()
    assert not hasattr(consolidated.analysis_paths, "analysis_timeseries_zarr")


def test_a_summary_rebuild_never_touches_a_series_branch(consolidated, monkeypatch):
    """Property: a changed inputs fingerprint rewrites the summary groups and leaves every file of both timeseries
    branches byte-identical.
    Class: NEW CAPABILITY. Plausible wrong implementation: the earlier whole-store publish, which replaces the
        store and drops the series; the test runs that publish as its must-fire control and sees the series hashes
        change and the group vanish.
    Why required: with the clearing switch on the series exist nowhere else, so a rebuild that rewrites them
        destroys the only copy.
    Killed by: any rebuild that replaces or rewrites the store as a whole.
    Passes under a second correct implementation: a rebuild that rewrites each summary group through a different
        zarr write path, provided it never writes under a timeseries group.
    Not established: the red-on-the-pre-change-tree output was not captured this round; the in-test control is the
        substitute.
    """
    from hhemt.processing_analysis import TRITONSWMM_analysis_post_processing as pp
    from hhemt.utils import write_datatree_zarr

    store = _tree_path(consolidated)
    before = _series_hashes(store)
    assert before, "control: there are series files to compare"
    rebuilt = []
    real = pp._write_summary_branches
    monkeypatch.setattr(
        pp, "_write_summary_branches", lambda self, *a, **k: (rebuilt.append(1), real(self, *a, **k))[1]
    )
    monkeypatch.setattr(pp, "_consolidation_inputs_fingerprint", lambda self: "a-different-fingerprint")
    consolidated.process.consolidate_to_datatree()
    assert rebuilt, "the changed fingerprint must trigger a rebuild"
    assert _series_hashes(store) == before
    # the earlier route: build a tree of the summaries alone and publish it as the whole store
    tree = _open(consolidated)
    summaries_only = {
        f"/{p.lstrip('/')}": n.dataset.load()
        for p, n in tree.subtree_with_keys
        if n.has_data and p.lstrip("/").rsplit("/", 1)[-1] == "summary"
    }
    write_datatree_zarr(xr.DataTree.from_dict({"/": xr.Dataset(), **summaries_only}), store)
    assert _series_hashes(store) != before and not _hashes(store, NODE_TS), "the whole-store publish destroys them"


def test_a_summary_group_is_rewritten_in_place_and_an_interrupted_rewrite_is_completed(consolidated, monkeypatch):
    """Property: a changed per-scenario summary value lands in its summary group on rebuild, the other groups stay
    readable, and a writer killed after the first group leaves a stampless store the next consolidation
    completes with the series untouched.
    Class: NEW CAPABILITY. Plausible wrong implementation: a rebuild that writes the stamp before the groups, so a
        kill leaves a stamped half-written tree that the reuse gate trusts. The kill arm is the must-fire control.
    Why required: the store is rewritten group by group; the stamp-last order is what makes an interruption
        recognisable.
    Killed by: stamping first; swallowing the writer's exception.
    Passes under a second correct implementation: removing the stamp and writing the groups inside any other single
        deferred-index block.
    Not established: the red-on-the-pre-change-tree output was not captured this round; the in-test control is the
        substitute.
    """
    import hhemt.processing_analysis as pa
    from hhemt.processing_analysis import TRITONSWMM_analysis_post_processing as pp
    from hhemt.scenario import TRITONSWMM_scenario

    store = _tree_path(consolidated)
    series = _series_hashes(store)
    path = "tritonswmm/swmm_node/summary"
    var = sorted(_open(consolidated)[path].to_dataset().data_vars)[0]
    changed = TRITONSWMM_scenario(0, consolidated).scen_paths.output_tritonswmm_node_summary
    ds = xr.open_zarr(changed, consolidated=False).load()
    ds[var] = ds[var] + 1000.0
    shutil.rmtree(changed)
    ds.to_zarr(changed, mode="w")
    monkeypatch.setattr(pp, "_consolidation_inputs_fingerprint", lambda self: "changed-inputs-1")
    consolidated.process.consolidate_to_datatree()
    new = _open(consolidated)[path].to_dataset()[var].isel(event_iloc=0).load()
    assert np.allclose(new.values, ds[var].values.reshape(new.shape), equal_nan=True)
    other = [p for p in _summary_paths(consolidated) if p != path and (store / p).is_dir()]
    assert other and all(_open(consolidated)[p].to_dataset().data_vars for p in other)

    real = pa.append_zarr
    seen = []

    def boom(ds, store_, **kw):
        """Pass every write through, and raise at the second summary-group write."""
        if kw.get("mode") == "w":
            seen.append(kw["group"])
            if len(seen) == 2:
                raise RuntimeError("writer killed after the first group")
        return real(ds, store_, **kw)

    monkeypatch.setattr(pa, "append_zarr", boom)
    monkeypatch.setattr(pp, "_consolidation_inputs_fingerprint", lambda self: "changed-inputs-2")
    with pytest.raises(RuntimeError, match="killed after the first group"):
        consolidated.process.consolidate_to_datatree()
    assert "consolidation_version" not in __import__("hhemt.utils", fromlist=["x"]).zarr_root_attrs(store)
    monkeypatch.setattr(pa, "append_zarr", real)
    consolidated.process.consolidate_to_datatree()
    tree = _open(consolidated)
    for p in _summary_paths(consolidated):
        if (store / p).is_dir():
            assert tree[p].to_dataset().data_vars, p
    assert _series_hashes(store) == series


def test_the_reprocess_delete_keeps_the_series_and_removes_the_summaries_and_the_stamp(consolidated, tmp_path):
    """Property: the reprocess delete runner removes the root stamp and every summary branch of analysis_datatree.zarr,
    keeps every timeseries branch byte-identical, and deletes sensitivity_datatree.zarr whole.
    Class: NEW CAPABILITY. Plausible wrong implementation: deleting analysis_datatree.zarr outright, as the runner
        did before. Control: the same runner removes the planted sensitivity tree whole, so a runner that kept
        everything also fails.
    Why required: a reprocess delete that removes the tree destroys the only copy of a cleared series.
    Killed by: the old whole-tree delete; a delete that leaves a summary branch.
    Passes under a second correct implementation: a runner that deletes the summary branches with a different
        deletion primitive: the surviving files are the same.
    Not established: the red-on-the-pre-change-tree output was not captured this round; the in-test control is the
        substitute.
    """
    from hhemt import delete_reprocess_zarr_runner as runner
    from hhemt.utils import zarr_root_attrs

    store = _tree_path(consolidated)
    analysis_dir = store.parent
    series = _series_hashes(store)
    assert "consolidation_version" in zarr_root_attrs(store)
    (analysis_dir / "sensitivity_datatree.zarr").mkdir()
    (analysis_dir / "sensitivity_datatree.zarr" / "zarr.json").write_text("{}")
    assert runner.main(["--analysis-dir", str(analysis_dir)]) == 0
    assert "consolidation_version" not in zarr_root_attrs(store)
    assert not [p for p in store.glob("*/*/summary")], "every summary branch is removed"
    assert _series_hashes(store) == series
    assert not (analysis_dir / "sensitivity_datatree.zarr").exists()
    tree = xr.open_datatree(store, engine="zarr", consolidated=True)
    assert tree[NODE_TS].to_dataset().data_vars


def test_a_reprocess_delete_killed_part_way_is_followed_by_a_full_summary_rewrite(consolidated, monkeypatch):
    """Property: a delete killed after the stamp is removed and one summary group is half-deleted is followed by a
    consolidation that rewrites every summary group, series untouched; with the stamp present and one group
    removed by hand, no rebuild happens.
    Class: NEW CAPABILITY. Plausible wrong implementation: a delete that removes the groups before the stamp,
        leaving a stamped tree with a missing group that the gate reuses. The by-hand arm shows the stamp is the
        only signal.
    Why required: the order of the delete is what forces the rewrite after an interruption.
    Killed by: removing the stamp last; a gate that ignores the root stamp.
    Passes under a second correct implementation: a delete that removes the stamp and the groups in one block but in
        the same order.
    Not established: the red-on-the-pre-change-tree output was not captured this round; the in-test control is the
        substitute.
    """
    from hhemt import delete_reprocess_zarr_runner as runner
    from hhemt.processing_analysis import TRITONSWMM_analysis_post_processing as pp

    store = _tree_path(consolidated)
    series = _series_hashes(store)
    rebuilt = []
    real = pp._write_summary_branches
    monkeypatch.setattr(
        pp, "_write_summary_branches", lambda self, *a, **k: (rebuilt.append(1), real(self, *a, **k))[1]
    )

    # control: stamp present, inputs matching, one summary group removed by hand -> reused, no rebuild
    victim = sorted(store.glob("*/*/summary"))[0]
    saved = victim.parent / "_saved_summary"
    shutil.copytree(victim, saved)
    shutil.rmtree(victim)
    consolidated.process.consolidate_to_datatree()
    assert rebuilt == [], "with the stamp in place nothing signals the damaged group"
    shutil.copytree(saved, victim)
    shutil.rmtree(saved)

    killed = []

    def killed_delete(paths, **kw):
        """Half-delete the first branch it is asked to delete, then raise as a killed process would."""
        killed.append(paths)
        files = sorted(p for p in Path(paths[0]).rglob("*") if p.is_file())
        for p in files[: len(files) // 2]:
            p.unlink()
        raise OSError("killed during the delete")

    monkeypatch.setattr(runner, "delete_and_account", killed_delete)
    with pytest.raises(OSError, match="killed during the delete"):
        runner.delete_summary_branches(store, store.parent)
    monkeypatch.setattr(runner, "delete_and_account", runner.delete_and_account)
    assert killed
    from hhemt.utils import zarr_root_attrs

    assert "consolidation_version" not in zarr_root_attrs(store), "the stamp goes before any summary group"
    consolidated.process.consolidate_to_datatree()
    assert rebuilt == [1], "a store with no stamp is read as incomplete and rewritten"
    tree = _open(consolidated)
    for p in _summary_paths(consolidated):
        if (store / p).parent.is_dir():
            assert tree[p].to_dataset().data_vars, p
    assert _series_hashes(store) == series


def test_consolidation_version_3_rebuilds_a_tree_stamped_2_once_and_a_tree_stamped_3_not_at_all(
    consolidated, monkeypatch
):
    """Property: CONSOLIDATION_VERSION is 3; a tree whose root stamp reads 2 rewrites its summaries once and its series
    not at all, and a tree stamped 3 with matching inputs is not rewritten.
    Class: NEW CAPABILITY. Plausible wrong implementation: a gate that never reads the root stamp, or one that
        rewrites every call. The unchanged-tree arm is the control for the second.
    Why required: version 3 is the signal that every layout-2 tree rebuilds its summaries.
    Killed by: leaving the version at 2; a gate that rebuilds always.
    Passes under a second correct implementation: a gate that compares the stamp through the log field instead of
        the root attribute, provided a stale root stamp still forces the rewrite.
    Not established: the red-on-the-pre-change-tree output was not captured this round; the in-test control is the
        substitute.
    """
    from hhemt.processing_analysis import TRITONSWMM_analysis_post_processing as pp
    from hhemt.utils import deferred_consolidation, update_zarr_root_attrs

    assert pp.CONSOLIDATION_VERSION == 3
    store = _tree_path(consolidated)
    series = _series_hashes(store)
    rebuilt = []
    real = pp._write_summary_branches
    monkeypatch.setattr(
        pp, "_write_summary_branches", lambda self, *a, **k: (rebuilt.append(1), real(self, *a, **k))[1]
    )
    consolidated.process.consolidate_to_datatree()
    assert rebuilt == [], "control: a tree stamped 3 with matching inputs is not rebuilt"
    with deferred_consolidation(store):
        update_zarr_root_attrs(store, {"consolidation_version": 2})
    consolidated.process.consolidate_to_datatree()
    assert rebuilt == [1]
    assert _series_hashes(store) == series
    consolidated.process.consolidate_to_datatree()
    assert rebuilt == [1], "the rebuild restamped the tree, so it happens once"


def test_the_system_tree_carries_summaries_only(consolidated, tmp_path):
    """Property: the system tree built from an analysis holds summary groups and no timeseries group.
    Class: NEW CAPABILITY. Plausible wrong implementation: grafting every populated node, which copies the gridded
        series into the system tree. Control: the source analysis tree does hold a timeseries group.
    Why required: a system tree carries no series, and copying them would duplicate the largest store.
    Killed by: the old graft of every node.
    Passes under a second correct implementation: a graft that filters on the group's parent name instead of its own
        name: the groups copied are the same.
    Not established: the red-on-the-pre-change-tree output was not captured this round; the in-test control is the
        substitute.
    """
    from hhemt.processing_system import TRITONSWMM_system_post_processing

    store = tmp_path / "system_datatree.zarr"
    system = SimpleNamespace(
        sys_paths=SimpleNamespace(system_datatree_zarr=store),
        _analysis=consolidated,
        log=SimpleNamespace(write=lambda: None),
    )
    consolidated.sensitivity = None
    TRITONSWMM_system_post_processing(system).consolidate_system_datatree()
    groups = {p.lstrip("/") for p, _ in xr.open_datatree(store, engine="zarr", consolidated=True).subtree_with_keys}
    assert any(g.endswith("/timeseries") for g in {p.lstrip("/") for p, _ in _open(consolidated).subtree_with_keys})
    assert not [g for g in groups if g.endswith("/timeseries")]
    assert any(g.endswith("/summary") for g in groups)


def _member(tree_path: Path | None, layout_ok: bool = True):
    from hhemt.processing_analysis import TRITONSWMM_analysis_post_processing as pp

    return SimpleNamespace(
        _refresh_log=lambda: None,
        analysis_paths=SimpleNamespace(analysis_datatree_zarr=tree_path),
        process=SimpleNamespace(
            open_datatree=lambda: xr.open_datatree(tree_path, engine="zarr", consolidated=True),
            assert_summary_branch_layout=pp.assert_summary_branch_layout,
            _write_summary_group=_Writer()._write_summary_group,
        ),
    )


def _master(members: dict):
    return SimpleNamespace(
        member_prefix="member_",
        members=members,
        df_setup=pd.DataFrame({"run_mode": ["cpu"] * len(members)}, index=pd.Index(list(members), name="member_id")),
        experiment=SimpleNamespace(cfg_analysis=SimpleNamespace(analysis_id="exp")),
        analysis_paths=SimpleNamespace(sensitivity_datatree_zarr=None),
    )


def test_the_sensitivity_master_holds_member_summaries_and_series_in_one_store(consolidated, tmp_path):
    """Property: `build_sensitivity_datatree` returns members' `summary` branches only, the master's in-place write
    puts both the `summary` and the `timeseries` branches of each member into the one master store, and a member tree
    holding a layout-30 summary is refused naming layout 31.
    Class: NEW CAPABILITY. Plausible wrong implementation: grafting every node into the in-memory tree, or leaving the
        series out of the store. Control: the layout-30 member tree is refused.
    Why required: the master's summaries and series live in one store, `experiment_datatree.zarr`.
    Killed by: grafting without the timeseries filter; no layout refusal; no series copy.
    Passes under a second correct implementation: filtering on the node's group path instead of its last component.
    Not established: the red-on-the-pre-change-tree output was not captured this round; the in-test control is the
        substitute.
    """
    from hhemt import sensitivity_analysis as sa
    from hhemt.processing_analysis import ProcessingError
    from hhemt.sensitivity_analysis import TRITONSWMM_sensitivity_analysis as sens

    member_tree = _tree_path(consolidated)
    master = _master({"0": _member(member_tree), "1": _member(member_tree)})
    tree = sens.build_sensitivity_datatree(master)
    paths = {p.lstrip("/") for p, n in tree.subtree_with_keys if n.has_data}
    assert any(p == "member_0/tritonswmm/swmm_node/summary" for p in paths)
    assert not [p for p in paths if p.endswith("/timeseries")], "no series enters the master's summary tree"
    tree, sources = sa._build_master_tree(master)
    store = tmp_path / "experiment_datatree.zarr"
    sens._write_master_in_place(master, tree, sources, store)
    out = xr.open_datatree(store, engine="zarr", consolidated=True)
    assert "depth" in out[f"member_1/{NODE_TS}"].to_dataset().data_vars
    assert any(p.lstrip("/").endswith("/summary") for p, n in out.subtree_with_keys if n.has_data)

    flat = tmp_path / "flat.zarr"
    ds = xr.Dataset({"v": (("event_iloc",), [1.0])}, coords={"event_iloc": [0]})
    xr.DataTree.from_dict({"/": xr.Dataset(), "/tritonswmm/swmm_node": ds}).to_zarr(flat, consolidated=True)
    with pytest.raises(ProcessingError, match="layout 31"):
        sens.build_sensitivity_datatree(_master({"0": _member(flat)}))


def test_the_summary_readers_open_their_groups_at_the_new_paths(consolidated):
    """Property: every reader that carries a path constant (arm groups, intercomparison variables, performance groups,
    the benchmarking node lookup) names a summary branch that exists in the constructed tree, and the layout-30
    form finds nothing.
    Class: NEW CAPABILITY. Plausible wrong implementation: a reader left on the parent path. Control:
        _find_perf_node on a tree with the layout-30 path returns None.
    Why required: a reader on the old path reads an empty group without an error.
    Killed by: any constant left at {family}/{kind}.
    Passes under a second correct implementation: readers that join the branch name at run time instead of holding a
        full constant.
    Not established: the red-on-the-pre-change-tree output was not captured this round; the in-test control is the
        substitute.
    """
    from hhemt.bundle._combine import _INTERCOMPARISON_VARS, _PERFORMANCE_GROUP_BY_ARM
    from hhemt.report_renderers._model_arms import ARM_GROUPS
    from hhemt.report_renderers.sensitivity_benchmarking import _find_perf_node

    tree = _open(consolidated)
    present = {p.lstrip("/") for p, n in tree.subtree_with_keys if n.has_data}
    arm_paths = {g.lstrip("/") for groups in ARM_GROUPS.values() for g in groups.values()}
    assert arm_paths and all(p.endswith("/summary") for p in arm_paths)
    assert {p for p in arm_paths if p in present}, "control: at least one arm group exists in the constructed tree"
    for candidates, _var in _INTERCOMPARISON_VARS:
        assert all(c.endswith("/summary") for c in candidates)
    assert all(p.endswith("/summary") for p in _PERFORMANCE_GROUP_BY_ARM.values())
    perf = next(p for p in present if p.endswith("performance/summary"))
    master = xr.DataTree.from_dict({"/": xr.Dataset(), f"/member_x/{perf}": tree[perf].to_dataset().load()})
    assert _find_perf_node(master, "x") is not None
    flat = xr.DataTree.from_dict(
        {"/": xr.Dataset(), "/member_x/tritonswmm/performance": tree[perf].to_dataset().load()}
    )
    assert _find_perf_node(flat, "x") is None, "control: a reader on the flat performance path finds nothing"


#: Layout-30 path literals that remain in ``src`` by classification (a census of every reader of a group path),
#: each with the reason that justifies it. A literal not listed here, with no ``/summary`` after it, is an
#: unclassified reader.
_CLASSIFIED = {
    "utils.py": "group chain of the tier test: the plain group `{family}/{kind}` exists in both layouts",
    "report_renderers/cross_experiment_disk_utilization.py": "tier test through group-name prefixes, both layouts",
    "version_migration/context.py": "V0003/V0021 helpers that write the layout those migrations produced",
    "processing_analysis.py": "comments naming the node, or the layout-31 summary/timeseries paths themselves",
}


def test_no_summary_path_literal_remains_outside_the_classified_table():
    """Property: no source line carries a {family}/{kind} path literal without a summary or timeseries suffix outside
    the files classified as layout-neutral.
    Class: NEW CAPABILITY. Plausible wrong implementation: a reader re-pointed in one file and missed in another.
        The matcher's must-fire control is asserted in the test.
    Why required: the census of group-path readers is a superset matcher and this test keeps it true after later edits.
    Killed by: a new unsuffixed literal in an unlisted file.
    Passes under a second correct implementation: a literal built by string concatenation, which this matcher cannot
        see (a stated limit).
    Not established: the red-on-the-pre-change-tree output was not captured this round; the in-test control is the
        substitute.
    """
    pat = re.compile(
        r"(tritonswmm|triton_only|swmm_only)/(triton|swmm_node|swmm_link|performance)(?!/(summary|timeseries))"
    )
    assert pat.search("tritonswmm/swmm_node"), "must-fire control"
    assert not pat.search("tritonswmm/swmm_node/summary")
    stray = []
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        if rel.startswith("version_migration/versions/") or rel in _CLASSIFIED:
            continue
        for n, line in enumerate(path.read_text().splitlines(), 1):
            if pat.search(line):
                stray.append(f"{rel}:{n}: {line.strip()[:100]}")
    assert stray == [], "unclassified layout-30 path literals:\n" + "\n".join(stray)


def test_the_root_stamp_helpers_set_and_remove_attributes_without_touching_a_group(tmp_path):
    """Property: update_zarr_root_attrs sets and removes root attributes, never rewrites a group's data files, and a
    nested deferred_consolidation block leaves the one index rebuild to the outer block.
    Class: NEW CAPABILITY. Plausible wrong implementation: a helper that rewrites the store, or a nested block that
        rebuilds the index and unregisters the store so later appends are refused.
    Why required: the stamp is set last and removed first without rewriting any group.
    Killed by: rewriting attributes through a full group rewrite; an inner block that rebuilds.
    Passes under a second correct implementation: setting attributes through the zarr group API: the stored
        attributes are the same.
    Not established: the red-on-the-pre-change-tree output was not captured this round; the in-test control is the
        substitute.
    """
    from hhemt.utils import deferred_consolidation, update_zarr_root_attrs, zarr_root_attrs

    store = tmp_path / "s.zarr"
    ds = xr.Dataset({"v": ("n", [1.0, 2.0])}, coords={"n": [0, 1]})
    with deferred_consolidation(store):
        ds.to_zarr(store, group="a/b/summary", mode="w", consolidated=False)
        update_zarr_root_attrs(store, {"consolidation_version": 3, "x": np.int64(4)})
        with deferred_consolidation(store):  # nested: the outer block owns the one rebuild
            ds.to_zarr(store, group="a/b/timeseries", mode="w", consolidated=False)
    before = {p: p.read_bytes() for p in (store / "a").rglob("*") if p.is_file()}
    assert zarr_root_attrs(store) == {"consolidation_version": 3, "x": 4}
    with deferred_consolidation(store):
        update_zarr_root_attrs(store, remove=("consolidation_version",))
    assert zarr_root_attrs(store) == {"x": 4}
    assert {p: p.read_bytes() for p in (store / "a").rglob("*") if p.is_file() and p.name != "zarr.json"} == {
        p: b for p, b in before.items() if p.name != "zarr.json"
    }
    tree = xr.open_datatree(store, engine="zarr", consolidated=True)
    assert "a/b/summary" in {p.lstrip("/") for p, _ in tree.subtree_with_keys}
    assert "a/b/timeseries" in {p.lstrip("/") for p, _ in tree.subtree_with_keys}
