"""The sensitivity master holds every member's summaries and series, so with clearing on no zarr content is twice.

The master tree ``experiment_datatree.zarr`` holds ``member_{id}/{family}/{kind}/summary``
AND ``.../timeseries``; each member's tree is copied in one member at a time, every copied group is read back equal,
and only then is the member's ``analysis_datatree.zarr`` removed (clearing switch on), with the removal recorded in the
member's log BEFORE the delete.

No solver and no Snakemake: every member tree is constructed under ``tmp_path`` in the layout-31 form, and the members
and the master are small stand-ins that carry exactly the attributes the code under test reads. The provenance and
sentinel plumbing of ``consolidate_sensitivity_datatree`` is outside these rows; they exercise the three steps it runs
for the store (build the tree, write it in place, move the member trees).

Every row names its class, its control, the change that kills it and a second correct implementation.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

from hhemt import delete_member_reprocess_runner, delete_reprocess_zarr_runner
from hhemt import processing_analysis as pa
from hhemt import sensitivity_analysis as sa
from hhemt.exceptions import ProcessingError
from hhemt.utils import write_datatree_zarr, zarr_root_attrs

SUMMARY = "tritonswmm/swmm_node/summary"
SERIES = "tritonswmm/swmm_node/timeseries"


@pytest.fixture(autouse=True)
def _first_scenario_has_a_node_summary(monkeypatch):
    """The first scenario's per-scenario paths: the one summary mode the stand-in member produces is present."""
    monkeypatch.setattr(sa, "_first_scenario_paths", lambda analysis: SimpleNamespace(swmm_node_summary=Path("x")))


def _summary(scale: float) -> xr.Dataset:
    return xr.Dataset(
        {"depth_max": (("event_iloc", "node_id"), scale * np.arange(6.0).reshape(2, 3), {"units": "m"})},
        coords={"event_iloc": [0, 1], "node_id": ["a", "b", "c"], "event_id": ("event_iloc", ["e0", "e1"])},
    )


def _series(scale: float) -> xr.Dataset:
    depth = scale + np.arange(24.0).reshape(2, 4, 3)
    depth[0, 3, :] = np.nan
    return xr.Dataset(
        {"depth": (("event_iloc", "elapsed_s", "node_id"), depth, {"units": "m"})},
        coords={
            "event_iloc": [0, 1],
            "elapsed_s": np.array([60, 120, 180, 240], dtype="int64"),
            "node_id": ["a", "b", "c"],
            "event_id": ("event_iloc", ["e0", "e1"]),
        },
    )


class _Field:
    """Stand-in for a ``LogField``: ``get`` and ``set`` over one value."""

    def __init__(self, value=None):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class _Process:
    """Stand-in for the member's post-processing object: what the master reads of it."""

    _MODE_TO_TREE_PATH = {"tritonswmm_swmm_node": SUMMARY}
    _MODE_CONFIG = {"tritonswmm_swmm_node": ("swmm_node_summary",)}

    def __init__(self, member):
        self.member = member
        self.fingerprint = "fp-1"
        self.per_scenario_summary: xr.Dataset | None = None

    def open_datatree(self):
        return xr.open_datatree(self.member.analysis_paths.analysis_datatree_zarr, engine="zarr", consolidated=True)

    assert_summary_branch_layout = staticmethod(pa.TRITONSWMM_analysis_post_processing.assert_summary_branch_layout)

    # The master hands a member's group to the member's own writer. The stand-in borrows the real writer and its
    # helpers, and reports no scenario as cleared, so every group takes the whole-group write.
    _mode_of_group = pa.TRITONSWMM_analysis_post_processing._mode_of_group
    _prepare_summary_group = staticmethod(pa.TRITONSWMM_analysis_post_processing._prepare_summary_group)
    _write_summary_group = pa.TRITONSWMM_analysis_post_processing._write_summary_group

    def _summary_records(self, mode):
        return {}

    def _consolidation_inputs_fingerprint(self):
        return self.fingerprint

    def _retrieve_combined_output(self, mode):
        if self.per_scenario_summary is None:
            raise FileNotFoundError(f"Summary file not found for {mode}")
        return self.per_scenario_summary.copy(deep=True)


class _Member:
    def __init__(self, root: Path, member_id: str, *, clearing: bool, scale: float):
        self.member_id = member_id
        self.directory = root / "members" / f"member_{member_id}"
        self.directory.mkdir(parents=True)
        self.tree = self.directory / "analysis_datatree.zarr"
        write_datatree_zarr(
            xr.DataTree.from_dict(
                {
                    "/": xr.Dataset(attrs={"analysis_id": f"member_{member_id}", "consolidation_version": 3}),
                    f"/{SUMMARY}": _summary(scale),
                    f"/{SERIES}": _series(scale),
                }
            ),
            self.tree,
        )
        self.analysis_paths = SimpleNamespace(analysis_datatree_zarr=self.tree, analysis_dir=self.directory)
        self.cfg_analysis = SimpleNamespace(
            toggle_clear_scenario_timeseries_after_consolidation=clearing, toggle_consolidate_timeseries=True
        )
        self.log = SimpleNamespace(
            member_tree_moved_to_master=_Field(),
            member_tree_moved_groups=_Field(),
            datatree_consolidation_complete=_Field(True),
            consolidation_inputs_fingerprint=_Field("fp-1"),
        )
        self.process = _Process(self)

    def _refresh_log(self):
        return None


class _Master:
    """Carries the three methods of the master that write and move, bound to the attributes they read."""

    member_prefix = "member_"
    _write_master_in_place = sa.TRITONSWMM_sensitivity_analysis._write_master_in_place
    _move_member_trees = sa.TRITONSWMM_sensitivity_analysis._move_member_trees
    _move_member_tree = sa.TRITONSWMM_sensitivity_analysis._move_member_tree
    _assert_member_complete_in_master = sa.TRITONSWMM_sensitivity_analysis._assert_member_complete_in_master

    def __init__(self, root: Path, members: dict):
        self.root = root
        self.members = members
        self.store = root / "experiment_datatree.zarr"
        self.analysis_paths = SimpleNamespace(sensitivity_datatree_zarr=self.store)
        self.df_setup = pd.DataFrame(
            {"run_mode": ["cpu"] * len(members)}, index=pd.Index(list(members), name="member_id")
        )
        self.experiment = SimpleNamespace(cfg_analysis=SimpleNamespace(analysis_id="exp"))

    def run(self) -> None:
        """Build the tree, write it in place with each member's tree moved before the next member's copy begins."""
        tree, sources = sa._build_master_tree(self)
        tree.attrs["consolidation_version"] = pa.TRITONSWMM_analysis_post_processing.CONSOLIDATION_VERSION
        copied: dict = {}
        self._write_master_in_place(
            tree,
            sources,
            self.store,
            copied=copied,
            after_member=lambda member_id: self._move_member_tree(member_id, sources, copied),
        )
        self._move_member_trees(sources, copied)


def _build(root: Path, *, clearing: bool) -> _Master:
    members = {
        "0": _Member(root, "0", clearing=clearing, scale=1.0),
        "1": _Member(root, "1", clearing=clearing, scale=10.0),
    }
    return _Master(root, members)


def _master_group(master: _Master, member_id: str, group: str) -> xr.Dataset:
    return xr.open_zarr(master.store, group=f"member_{member_id}/{group}", consolidated=True, chunks=None).load()


def _digest(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(root)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def _equal(a: xr.Dataset, b: xr.Dataset) -> bool:
    return sa._datasets_equal(a.load(), b.load())


def _array_keys(root: Path, names: frozenset[str]) -> list[tuple[str, str, str]]:
    """Every ``(member, group, variable)`` array named in ``names`` under any ``.zarr`` store beneath ``root``.

    Unit: one array directory (a ``zarr.json`` whose node type is ``array``). Matcher: the document's ``node_type``
    and the directory name in ``names``. A store under ``members/member_X`` names its member by that directory; a
    master store names it by its first path component.
    """
    found = []
    for store in sorted(p for p in root.rglob("*.zarr") if p.is_dir()):
        for doc in sorted(store.rglob("zarr.json")):
            if doc.parent == store or json.loads(doc.read_text()).get("node_type") != "array":
                continue
            if doc.parent.name not in names:
                continue
            parts = doc.parent.relative_to(store).parts
            if parts[0].startswith("member_"):
                member, group = parts[0], "/".join(parts[1:-1])
            else:
                member, group = store.parent.name, "/".join(parts[:-1])
            found.append((member, group, doc.parent.name))
    return found


_SERIES_NAMES = frozenset({"depth", "depth_max"})


def test_the_master_holds_member_summaries_and_series_equal_to_each_members_and_no_separate_series_store(tmp_path):
    """Property: after the master consolidates two members with clearing off, ``member_{id}/{family}/{kind}/summary``
    and ``.../timeseries`` are in ``experiment_datatree.zarr`` with values equal to each member's, and no
    ``sensitivity_timeseries.zarr`` exists.
    Class: NEW CAPABILITY. Plausible wrong implementation: copying summaries only (summaries without series) or
        copying into a second series store. Reachability: both branches are written into every member tree built
        below. Both-states
        anchor: the assertion compares VALUES of branches both the member and the master hold. Control: the second
        member's values differ from the first's (scale 10), so a master that copied one member twice fails.
    Why required: the shape's D20 puts the series in the main store, at parity with an analysis.
    Killed by: dropping the timeseries copy; writing ``sensitivity_timeseries.zarr``.
    Passes under a second correct implementation: copying whole groups instead of one event at a time.
    Not established: the red on the pre-change tree was not captured; the in-test control is the substitute.
    """
    master = _build(tmp_path, clearing=False)
    master.run()
    for member_id, member in master.members.items():
        member_tree = xr.open_datatree(member.tree, engine="zarr", consolidated=True)
        assert _equal(_master_group(master, member_id, SUMMARY), member_tree[SUMMARY].to_dataset())
        assert _equal(_master_group(master, member_id, SERIES), member_tree[SERIES].to_dataset())
    assert not _equal(_master_group(master, "0", SERIES), _master_group(master, "1", SERIES))
    assert not (tmp_path / "sensitivity_timeseries.zarr").exists()


def test_a_member_tree_is_removed_only_after_its_verified_copy_and_a_failed_read_back_keeps_it(tmp_path, monkeypatch):
    """Property: with clearing on, each member's tree is deleted and ``member_tree_moved_to_master`` is ``done`` only
    after every copied group read back equal; a copy that does not read back equal keeps every tree and records no
    move.
    Class: NEW CAPABILITY. Plausible wrong implementation: deleting the tree after the copy without the read-back, or
        before it. Reachability: both members' trees exist and clearing is on. Both-states anchor: the tree's
        presence and the record, which the old code also had (the tree) or lacked (the record). Must-fire control:
        the series read-back answers False and the copy raises with both trees present.
    Why required: the member tree is the only source the copy was verified against.
    Killed by: deleting without the read-back; recording ``done`` before the delete.
    Passes under a second correct implementation: verifying with a content hash instead of a value comparison.
    Not established: the red on the pre-change tree was not captured; the in-test control is the substitute.
    """
    refused = _build(tmp_path / "refused", clearing=True)
    monkeypatch.setattr(pa.TRITONSWMM_analysis_post_processing, "_stored_slice_equals", staticmethod(lambda *a: False))
    with pytest.raises(ProcessingError, match="does not equal the member's slice"):
        refused.run()
    monkeypatch.undo()
    assert all(m.tree.is_dir() for m in refused.members.values())
    assert all(m.log.member_tree_moved_to_master.get() is None for m in refused.members.values())

    master = _build(tmp_path / "ok", clearing=True)
    master.run()
    for member in master.members.values():
        assert not member.tree.exists()
        assert member.log.member_tree_moved_to_master.get() == "done"
        assert SERIES in json.loads(member.log.member_tree_moved_groups.get())
    assert _master_group(master, "1", SERIES)["depth"].shape == (2, 4, 3)


def test_no_series_variable_is_stored_in_two_stores_under_clearing_and_the_walk_finds_the_trees_without_it(tmp_path):
    """Property: after consolidation with clearing on, no ``(member, group, variable)`` key of the series and summary
    variables exists in more than one ``.zarr`` store under the analysis directories; with clearing off the same walk
    reports every key twice.
    Class: NEW CAPABILITY. Plausible wrong implementation: copying into the master and leaving the member trees.
        Must-fire control: the clearing-off arm, where the walk must find duplicates, which proves the walk can
        see a member tree's arrays at all. Unit: array directories; matcher: ``zarr.json`` node type ``array`` named
        ``depth`` or ``depth_max`` (the helper above).
    Why required: D21's rule that with clearing on no store the pipeline writes holds data another holds.
    Killed by: leaving the member trees; a walk that cannot see the member trees (the control).
    Passes under a second correct implementation: moving a tree by rename instead of copy and delete.
    Not established: the red on the pre-change tree was not captured; the in-test control is the substitute.
    """
    on = _build(tmp_path / "on", clearing=True)
    on.run()
    keys_on = _array_keys(tmp_path / "on", _SERIES_NAMES)
    assert keys_on and len(keys_on) == len(set(keys_on)), "no key in two stores"
    off = _build(tmp_path / "off", clearing=False)
    off.run()
    keys_off = _array_keys(tmp_path / "off", _SERIES_NAMES)
    assert len(keys_off) == 2 * len(set(keys_off)), "control: the walk sees the member trees and finds every key twice"


def test_a_summary_rebuild_without_member_trees_rewrites_the_summaries_and_leaves_the_series_byte_identical(tmp_path):
    """Property: after the move, a changed inputs fingerprint rebuilds a member's ``summary`` group in the master from
    its per-scenario summary store and the ``timeseries`` branches are byte-identical before and after; with the
    per-scenario store gone the rebuild refuses naming the member.
    Class: NEW CAPABILITY. Plausible wrong implementation: re-consolidating the member (recreating its tree and
        failing on its cleared series) or rewriting the whole master. Reachability: the fingerprint of member 0 is
        changed after the move. Both-states anchor: the master's summary values and the series files' digest.
        Must-fire control: the second arm removes the per-scenario source and expects ``ProcessingError``.
    Why required: the master's summary for a member is rebuilt from per-scenario stores, never from a removed tree.
    Killed by: a rebuild that reads the removed tree; a master rewrite that touches the series.
    Passes under a second correct implementation: running the member's consolidation against the master as target.
    Not established: the red on the pre-change tree was not captured; the in-test control is the substitute.
    """
    master = _build(tmp_path, clearing=True)
    master.run()
    series_dir = master.store / "member_0" / SERIES
    before = _digest(series_dir)
    member = master.members["0"]
    member.process.fingerprint = "fp-2"
    member.process.per_scenario_summary = _summary(7.0)
    master.run()
    assert _equal(_master_group(master, "0", SUMMARY), _summary(7.0))
    assert _digest(series_dir) == before
    assert member.log.consolidation_inputs_fingerprint.get() == "fp-2"
    assert not member.tree.exists()

    member.process.fingerprint = "fp-3"
    member.process.per_scenario_summary = None
    with pytest.raises(ProcessingError, match="member 0"):
        master.run()


def test_a_master_consolidation_interrupted_after_one_members_move_completes_without_copying_that_member_again(
    tmp_path, monkeypatch
):
    """Property: a consolidation killed after the first member's move completes on re-run, copying only the second
    member; a member whose tree is gone, whose log marks it consolidated and which records no move is refused.
    Class: NEW CAPABILITY. Plausible wrong implementation: re-copying every member, which fails for the member whose
        tree is gone. Reachability: the first delete succeeds and the second raises. Must-fire control: the refusal
        arm, a member with no tree and no record.
    Why required: the master copy resumes per member.
    Killed by: copying a member whose record is ``done``; treating an absent tree with no record as a move.
    Passes under a second correct implementation: planning the copy per member before any write.
    Not established: the red on the pre-change tree was not captured; the in-test control is the substitute.
    """
    from hhemt import du_sentinels

    master = _build(tmp_path, clearing=True)
    real_delete = du_sentinels.delete_and_account
    deleted = []

    def kill_second(paths, **kw):
        if deleted:
            raise OSError("killed between members")
        deleted.append(paths)
        return real_delete(paths, **kw)

    monkeypatch.setattr(du_sentinels, "delete_and_account", kill_second)
    with pytest.raises(OSError, match="killed between members"):
        master.run()
    monkeypatch.undo()
    assert master.members["0"].log.member_tree_moved_to_master.get() == "done"
    assert master.members["1"].log.member_tree_moved_to_master.get() == "pending"
    copies = []
    real_copy = sa._copy_series_group
    monkeypatch.setattr(sa, "_copy_series_group", lambda *a, **k: (copies.append(a[1]), real_copy(*a, **k))[1])
    master.run()
    monkeypatch.undo()
    assert copies == [], "member 1 is pending, so the master confirms its groups and finishes the delete; no copy"
    assert [m.log.member_tree_moved_to_master.get() for m in master.members.values()] == ["done", "done"]
    assert not master.members["1"].tree.exists()

    lost = _build(tmp_path / "lost", clearing=True)
    shutil.rmtree(lost.members["1"].tree)
    with pytest.raises(ProcessingError, match="member 1"):
        lost.run()


def test_a_kill_after_the_tree_delete_and_before_done_resumes_to_done_with_an_equal_master(tmp_path, monkeypatch):
    """Property: a run killed after a member tree's delete and before ``done`` leaves ``pending`` and the resume sets
    ``done`` with the master unchanged; a member with no tree and no record refuses naming the member.
    Class: NEW CAPABILITY. Plausible wrong implementation: recording after the delete, which leaves no tree and no
        record. Reachability: the kill raises after the real delete. Must-fire control: the no-record arm.
    Why required: the record comes before the delete.
    Killed by: setting ``pending`` after the delete.
    Passes under a second correct implementation: recording in a journal file next to the tree.
    Not established: the red on the pre-change tree was not captured; the in-test control is the substitute.
    """
    from hhemt import du_sentinels

    master = _build(tmp_path, clearing=True)
    real_delete = du_sentinels.delete_and_account

    def delete_then_die(paths, **kw):
        real_delete(paths, **kw)
        raise OSError("killed after the delete")

    monkeypatch.setattr(du_sentinels, "delete_and_account", delete_then_die)
    with pytest.raises(OSError, match="killed after the delete"):
        master.run()
    monkeypatch.undo()
    first = master.members["0"]
    assert not first.tree.exists() and first.log.member_tree_moved_to_master.get() == "pending"
    before = _digest(master.store)
    master.run()
    assert first.log.member_tree_moved_to_master.get() == "done"
    assert _digest(master.store) == before or all(
        _equal(_master_group(master, m, g), xr.open_zarr(master.store, group=f"member_{m}/{g}", chunks=None))
        for m in ("0", "1")
        for g in (SUMMARY, SERIES)
    )


def test_a_half_deleted_member_tree_with_the_record_pending_leaves_the_master_byte_identical_and_finishes(tmp_path):
    """Property: with half of a member tree's files deleted and ``pending`` recorded, the resume leaves the master's
    ``member_{id}/`` groups byte-identical, removes the rest of the tree and sets ``done``; the same half-deleted tree
    with NO record is refused and never copied from.
    Class: NEW CAPABILITY. Plausible wrong implementation: reading the damaged tree as a verification or rewrite
        source, which overwrites a verified slice with fill values. Reachability: the damage removes metadata
        documents the tree's own index lists. Must-fire control: the no-record arm, refused naming the member.
    Why required: with the record pending the copy already read back equal, and the tree is no longer a source.
    Killed by: re-verifying against the tree; no completeness check of an unrecorded tree.
    Passes under a second correct implementation: moving the tree aside before deleting it.
    Not established: the red on the pre-change tree was not captured; the in-test control is the substitute.
    """
    master = _build(tmp_path / "pending", clearing=True)
    tree, sources = sa._build_master_tree(master)
    tree.attrs["consolidation_version"] = pa.TRITONSWMM_analysis_post_processing.CONSOLIDATION_VERSION
    copied = master._write_master_in_place(tree, sources, master.store)
    member = master.members["0"]
    member.log.member_tree_moved_groups.set(json.dumps(copied["0"]))
    member.log.member_tree_moved_to_master.set("pending")
    files = sorted(p for p in member.tree.rglob("*") if p.is_file())
    for p in files[: len(files) // 2]:
        p.unlink()
    before = _digest(master.store / "member_0")
    master.run()
    assert not member.tree.exists() and member.log.member_tree_moved_to_master.get() == "done"
    assert _digest(master.store / "member_0") == before

    incomplete = _build(tmp_path / "incomplete", clearing=True)
    tree2, sources2 = sa._build_master_tree(incomplete)
    copied2 = incomplete._write_master_in_place(tree2, sources2, incomplete.store)
    gone = incomplete.members["0"]
    gone.log.member_tree_moved_groups.set(json.dumps(copied2["0"]))
    gone.log.member_tree_moved_to_master.set("pending")
    shutil.rmtree(incomplete.store / "member_0" / SERIES)
    with pytest.raises(ProcessingError, match="member 0"):
        incomplete.run()
    assert gone.tree.exists(), "a pending move whose master group is missing deletes nothing"

    unrecorded = _build(tmp_path / "unrecorded", clearing=True)
    victim = unrecorded.members["0"]
    next(p for p in sorted(victim.tree.rglob("zarr.json")) if p.parent.name == "depth").unlink()
    with pytest.raises(ProcessingError, match="member 0"):
        unrecorded.run()
    assert not (unrecorded.store / "member_0" / SERIES).exists(), "nothing was copied from the damaged tree"


def test_the_member_reprocess_delete_with_the_tree_moved_exits_zero_and_touches_no_master(tmp_path):
    """Property: ``delete_member_reprocess_runner`` on a member whose tree was moved exits 0 and deletes nothing in
    the master; on a member whose tree is present it deletes the tree's ``summary`` branches and keeps its
    ``timeseries`` branches.
    Class: REGRESSION on the series (the runner deleted the whole tree, series included) and NEW CAPABILITY on the
        absent tree. Reachability: ``--analysis-dir`` is the member directory. Must-fire control: the present-tree
        arm, where the runner deletes the summary group, so an inert runner fails.
    Why required: with the series cleared the tree's ``timeseries`` branches are the only copy.
    Killed by: ``delete_and_account([tree])`` of the whole tree; an error on an absent tree.
    Passes under a second correct implementation: skipping the deletion when the move is recorded.
    Not established: the red on the pre-change tree was not captured; the in-test control is the substitute.
    """
    master = _build(tmp_path, clearing=True)
    master.run()
    before = _digest(master.store)
    moved = master.members["0"]
    assert delete_member_reprocess_runner.main(["--member-id", "0", "--analysis-dir", str(moved.directory)]) == 0
    assert _digest(master.store) == before

    present = _build(tmp_path / "present", clearing=False)
    kept = present.members["0"]
    assert delete_member_reprocess_runner.main(["--member-id", "0", "--analysis-dir", str(kept.directory)]) == 0
    tree = xr.open_datatree(kept.tree, engine="zarr", consolidated=False)
    paths = {p.lstrip("/") for p, n in tree.subtree_with_keys if n.has_data}
    assert SERIES in paths and SUMMARY not in paths


def test_the_masters_reprocess_delete_keeps_every_member_timeseries_branch_and_the_parameters(tmp_path):
    """Property: the reprocess delete of an ``experiment_datatree.zarr`` removes the root stamp and every
    ``member_*/{family}/{kind}/summary`` branch and keeps the ``timeseries`` branches and ``parameters``; the separate
    ``sensitivity_datatree.zarr`` store is still deleted whole.
    Class: NEW CAPABILITY. Plausible wrong implementation: deleting the master whole, as before, which removes the
        only copy of the series. Must-fire control: the separate store beside it IS deleted, so the runner ran.
    Why required: the master's series are append-only and, with the member trees gone, the only copy.
    Killed by: whole-store deletion of ``experiment_datatree.zarr``.
    Passes under a second correct implementation: moving the series aside while the summaries are deleted.
    Not established: the red on the pre-change tree was not captured; the in-test control is the substitute.
    """
    master = _build(tmp_path, clearing=True)
    master.run()
    unified = tmp_path / "sensitivity_datatree.zarr"
    write_datatree_zarr(xr.DataTree.from_dict({"/": xr.Dataset(attrs={"a": 1})}), unified)
    series_before = _digest(master.store / "member_1" / SERIES)
    assert zarr_root_attrs(master.store).get("consolidation_version") == 3
    assert delete_reprocess_zarr_runner.main(["--analysis-dir", str(tmp_path)]) == 0
    assert not unified.exists()
    assert "consolidation_version" not in zarr_root_attrs(master.store)
    assert not list(master.store.glob("*/*/*/summary"))
    assert _digest(master.store / "member_1" / SERIES) == series_before
    assert (master.store / "parameters" / "zarr.json").is_file()


def test_the_member_move_is_owed_for_a_pending_record_or_a_tree_under_clearing_and_for_nothing_else(tmp_path):
    """Property: ``_member_needs_move`` is True for a ``pending`` record and for a present tree with clearing on, and
    False for ``done``, for clearing off, and for an absent tree.
    Class: NEW CAPABILITY. Plausible wrong implementation: a reuse check that ignores the owed move and returns the
        stored master early. Reachability: the four member states are constructed. Control: the True arms.
    Why required: the master's early return must not skip a move the previous call did not finish.
    Killed by: constant False; constant True.
    Passes under a second correct implementation: deriving the answer from the master's log.
    Not established: the red on the pre-change tree was not captured; the in-test control is the substitute.
    """
    member = _build(tmp_path, clearing=True).members["0"]
    assert sa._member_needs_move(member) is True
    member.cfg_analysis.toggle_clear_scenario_timeseries_after_consolidation = False
    assert sa._member_needs_move(member) is False
    member.log.member_tree_moved_to_master.set("pending")
    assert sa._member_needs_move(member) is True
    member.log.member_tree_moved_to_master.set("done")
    assert sa._member_needs_move(member) is False


def _members_for(tmp_path: Path, monkeypatch, master_yaml: dict, sweep: dict) -> dict:
    """Run the real ``_create_members`` over one swept member and return the member config it wrote, as a dict."""
    from hhemt.config.analysis import analysis_config
    from tests.test_config_validation import _minimal_analysis_config_dict

    base = _minimal_analysis_config_dict(tmp_path)
    yaml_path = tmp_path / "master.yaml"
    yaml_path.write_text(yaml.safe_dump({**base, **master_yaml}))
    cfg = analysis_config.model_validate({**base, **master_yaml})
    captured = {}

    def fake_analysis(*, analysis_config_yaml, **kwargs):
        captured["yaml"] = analysis_config_yaml
        return SimpleNamespace()

    monkeypatch.setattr(sa.anlysis, "TRITONSWMM_analysis", fake_analysis)
    master = SimpleNamespace(
        unique_system_targets=[SimpleNamespace(analysis_ids=["0"], system="sys")],
        df_setup=pd.DataFrame({f"analysis.{k}": [v] for k, v in sweep.items()}, index=pd.Index(["0"])),
        member_prefix="member_",
        members_dir=tmp_path / "members",
        experiment=SimpleNamespace(cfg_analysis=cfg, analysis_config_yaml=yaml_path, hpc_system_config_yaml=None),
        _is_main_orchestrator=True,
        _skip_log_update=True,
    )
    master._master_states_clearing_explicitly = lambda: (
        sa.TRITONSWMM_sensitivity_analysis._master_states_clearing_explicitly(master)
    )
    sa.TRITONSWMM_sensitivity_analysis._create_members(master)
    return yaml.safe_load(Path(captured["yaml"]).read_text())


def test_a_member_sweeping_consolidation_off_resolves_clearing_false_and_passes_preflight(tmp_path, monkeypatch):
    """Property (errata E56): with consolidation on and clearing UNSET on the master, a member whose swept cell turns
    consolidation off gets clearing False and passes the clearing preflight; with the master's clearing explicitly
    True the same cell is refused.
    Class: REGRESSION against the member inheriting the master's resolved True. Reachability: ``_create_members`` is
        the real one, run over a one-row sweep. Must-fire control: the explicit-True master, refused by
        ``validation._validate_clear_scenario_timeseries`` with exactly one error.
    Why required: a swept ``toggle_consolidate_timeseries: false`` otherwise leaves the member with clearing True and
        consolidation False, which preflight refuses for a pair the user never wrote.
    Killed by: copying the master's resolved clearing value into every member.
    Passes under a second correct implementation: resolving the default after the overlay in the member's config.
    Not established: the red on the pre-change tree was not captured; the in-test control is the substitute.
    """
    from hhemt import validation
    from hhemt.config.analysis import analysis_config

    def refusals(member: dict) -> list:
        result = validation.ValidationResult(context="test")
        validation._validate_clear_scenario_timeseries(analysis_config.model_validate(member), result)
        return list(result.errors)

    unset = _members_for(
        tmp_path / "unset",
        monkeypatch,
        {"toggle_consolidate_timeseries": True},
        {"toggle_consolidate_timeseries": False},
    )
    assert unset["toggle_clear_scenario_timeseries_after_consolidation"] is False
    assert refusals(unset) == []

    explicit = _members_for(
        tmp_path / "explicit",
        monkeypatch,
        {"toggle_consolidate_timeseries": True, "toggle_clear_scenario_timeseries_after_consolidation": True},
        {"toggle_consolidate_timeseries": False},
    )
    assert explicit["toggle_clear_scenario_timeseries_after_consolidation"] is True
    assert len(refusals(explicit)) == 1
