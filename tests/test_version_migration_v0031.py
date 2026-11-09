"""V0031 (layout 30 to 31): every consolidated group holds a ``summary`` and a ``timeseries`` branch.

No solver and no Snakemake: every tree is constructed under ``tmp_path`` in the form layout 30 wrote (a summary
directly at ``{family}/{kind}`` of ``analysis_datatree.zarr``, the consolidated SWMM series in a separate
``analysis_timeseries.zarr``, a version-1 calendar-axis series group inside the tree), then migrated through the
runner. Each arm names its control: the input class that makes a broken implementation read as correct.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from hhemt.utils import write_datatree_zarr, zarr_root_attrs
from hhemt.version_migration import context as ctx_mod
from hhemt.version_migration import runner, state
from hhemt.version_migration.constants import LAYOUT_VERSION

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "legacy_layouts"
NODE_SERIES = "tritonswmm/swmm_node_timeseries"
LINK_SERIES = "tritonswmm/swmm_link_timeseries"
NODE_TS = "tritonswmm/swmm_node/timeseries"
LINK_TS = "tritonswmm/swmm_link/timeseries"


def _summary(scale: float = 1.0) -> xr.Dataset:
    return xr.Dataset(
        {"depth_max": (("event_iloc", "node_id"), scale * np.arange(6.0).reshape(2, 3), {"units": "m"})},
        coords={"event_iloc": [0, 1], "node_id": ["a", "b", "c"], "event_id": ("event_iloc", ["e0", "e1"])},
    )


def _series(offset: float = 0.0) -> xr.Dataset:
    depth = offset + np.arange(24.0).reshape(2, 4, 3)
    depth[0, 3, :] = np.nan  # event 0 is shorter than the axis: its tail is padding
    return xr.Dataset(
        {"depth": (("event_iloc", "elapsed_s", "node_id"), depth, {"units": "m"})},
        coords={
            "event_iloc": [0, 1],
            "elapsed_s": np.array([60, 120, 180, 240], dtype="int64"),
            "node_id": ["a", "b", "c"],
            "event_id": ("event_iloc", ["e0", "e1"]),
            "event_start_datetime": ("event_iloc", np.array(["2020-01-01", "2020-02-01"], dtype="datetime64[ns]")),
        },
    )


def _legacy_calendar() -> xr.Dataset:
    t = np.array(["2020-01-01T00:01:00", "2020-01-01T00:02:00"], dtype="datetime64[ns]")
    return xr.Dataset(
        {"depth": (("event_iloc", "date_time", "node_id"), np.ones((2, 2, 3)), {"units": "m"})},
        coords={"event_iloc": [0, 1], "date_time": t, "node_id": ["a", "b", "c"]},
    )


def _write(path: Path, groups: dict[str, xr.Dataset], attrs: dict | None = None) -> None:
    nodes = {"/": xr.Dataset(attrs=attrs or {"analysis_id": "t"})}
    nodes.update({f"/{g}": ds for g, ds in groups.items()})
    write_datatree_zarr(xr.DataTree.from_dict(nodes), path)


def _digest(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file() and not p.name.endswith(".lock"):
            h.update(str(p.relative_to(root)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def _files_under(root: Path, group: str) -> dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((root / group).rglob("*"))
        if p.is_file()
    }


@pytest.fixture
def v30_tree(tmp_path: Path) -> Path:
    """A stamped layout-30 analysis: summaries at ``{family}/{kind}`` and the series in ``analysis_timeseries.zarr``."""
    work = tmp_path / "v30"
    work.mkdir()
    _write(
        work / "analysis_datatree.zarr",
        {
            "tritonswmm/swmm_node": _summary(1.0),
            "tritonswmm/swmm_link": _summary(2.0),
            "tritonswmm/performance": xr.Dataset({"Total": ("event_iloc", [1.0, 2.0])}, coords={"event_iloc": [0, 1]}),
        },
    )
    _write(work / "analysis_timeseries.zarr", {NODE_SERIES: _series(0.0), LINK_SERIES: _series(100.0)})
    runner.baseline(work, 30)
    return work


def _migrate(work: Path, *, apply: bool = True):
    return runner.run_migration(work, target=31, apply=apply)


def _open(work: Path, *, consolidated: bool = True) -> xr.DataTree:
    return xr.open_datatree(work / "analysis_datatree.zarr", engine="zarr", consolidated=consolidated)


def test_the_constructed_tree_is_a_layout_30_tree_before_the_migration(v30_tree: Path):
    """Property: the constructed input really holds a summary at {family}/{kind} and a separate series store.
    Class: NEW CAPABILITY. Plausible wrong implementation: a fixture that already holds the layout-31 shape, which
        makes every test below vacuous.
    Why required: the control that gives every other test in this file its meaning.
    Killed by: a fixture built at the new layout.
    Passes under a second correct implementation: any builder of the layout-30 shape.
    Not established: the red-on-the-pre-change-tree output was not captured this round; the in-test control is the
        substitute.
    """
    tree = _open(v30_tree)
    assert "depth_max" in tree["tritonswmm/swmm_node"].to_dataset().data_vars
    assert "summary" not in tree["tritonswmm/swmm_node"].children
    assert (v30_tree / "analysis_timeseries.zarr").is_dir()


def test_the_summary_groups_move_and_the_series_fold_in_and_the_old_store_goes(v30_tree: Path):
    """Property: migrating a layout-30 tree puts each summary at {kind}/summary with unchanged values, copies the
    series to {kind}/timeseries equal NaN-aware, removes the old store, records the attribute and stamps 31; a
    default consolidated read lists the new groups.
    Class: NEW CAPABILITY. Plausible wrong implementation: moving groups without rebuilding the index, or deleting
        the source before the read-back.
    Why required: it is the migration's whole contract on the ordinary input.
    Killed by: no index rebuild; a copy that drops the NaN padding.
    Passes under a second correct implementation: a migration that copies through the zarr API instead of xarray.
    Not established: the red-on-the-pre-change-tree output was not captured this round; the in-test control is the
        substitute.
    """
    before_summary = _open(v30_tree)["tritonswmm/swmm_node"].to_dataset().load()
    before_series = xr.open_zarr(
        v30_tree / "analysis_timeseries.zarr", group=NODE_SERIES, consolidated=False, decode_timedelta=False
    ).load()
    result = _migrate(v30_tree)
    assert result.migrations_applied == ["V0031__datatree_summary_and_timeseries_branches"]
    tree = _open(v30_tree)  # DEFAULT consolidated read: the index was rebuilt after the moves
    for kind in ("swmm_node", "swmm_link", "performance"):
        assert "summary" in tree[f"tritonswmm/{kind}"].children, kind
        assert not tree[f"tritonswmm/{kind}"].to_dataset().data_vars, f"{kind}: nothing may stay at the old path"
    xr.testing.assert_identical(tree["tritonswmm/swmm_node/summary"].to_dataset().load(), before_summary)
    got = tree[NODE_TS].to_dataset().load()
    xr.testing.assert_equal(got, before_series)
    assert bool(np.isnan(got["depth"].values[0, 3]).all()), "the padded NaN tail survives the copy"
    assert not (v30_tree / "analysis_timeseries.zarr").exists()
    assert zarr_root_attrs(v30_tree / "analysis_datatree.zarr")["migrated_from_analysis_timeseries"] == "done"
    assert state.read_version_file(v30_tree).layout_version == 31 == LAYOUT_VERSION


def test_a_second_run_changes_no_byte(v30_tree: Path):
    """Property: a second run plans nothing and the migrated tree is byte-identical.
    Class: NEW CAPABILITY. Plausible wrong implementation: a migration that rewrites the index or a group on every
        run.
    Why required: idempotence is a contract of the migration runner.
    Killed by: any non-idempotent step.
    Passes under a second correct implementation: a migration that skips already-done states by a different test.
    Not established: the red-on-the-pre-change-tree output was not captured this round; the in-test control is the
        substitute.
    """
    _migrate(v30_tree)
    once = _digest(v30_tree / "analysis_datatree.zarr")
    result = _migrate(v30_tree)
    assert result.migrations_applied == [] and result.migrations_planned == []
    assert _digest(v30_tree / "analysis_datatree.zarr") == once


def test_a_dry_run_changes_no_byte(v30_tree: Path):
    """Property: a dry run plans V0031 and changes no byte of the tree.
    Class: NEW CAPABILITY. Plausible wrong implementation: executing the primitive's body at plan time.
    Why required: a dry run must not mutate.
    Killed by: mutating in the planning call.
    Passes under a second correct implementation: a primitive that records its plan lazily.
    Not established: the red-on-the-pre-change-tree output was not captured this round; the in-test control is the
        substitute.
    """
    before = _digest(v30_tree)
    result = _migrate(v30_tree, apply=False)
    assert result.migrations_planned == ["V0031__datatree_summary_and_timeseries_branches"]
    assert _digest(v30_tree) == before


def test_a_migration_made_to_fail_the_read_back_leaves_the_series_store_in_place(v30_tree: Path, monkeypatch):
    """Property: when the copy does not read back equal the series store and the layout stamp are left unchanged and no
    completion attribute is written.
    Class: NEW CAPABILITY. Plausible wrong implementation: deleting the source before or without the read-back. The
        comparison is replaced by one that answers False, which is the must-fire control.
    Why required: the source is the only copy until the read-back passes.
    Killed by: deleting first.
    Passes under a second correct implementation: a read-back through a different reader.
    Not established: the red-on-the-pre-change-tree output was not captured this round; the in-test control is the
        substitute.
    """
    monkeypatch.setattr(ctx_mod, "_read_group_equal", lambda a, b: False)
    with pytest.raises(ValueError, match="does not read back equal"):
        _migrate(v30_tree)
    assert (v30_tree / "analysis_timeseries.zarr").is_dir()
    assert "migrated_from_analysis_timeseries" not in zarr_root_attrs(v30_tree / "analysis_datatree.zarr")
    assert state.read_version_file(v30_tree).layout_version == 30


def _half_delete(store: Path) -> int:
    files = sorted(p for p in store.rglob("*") if p.is_file())
    victims = files[: len(files) // 2]
    for p in victims:
        p.unlink()
    return len(victims)


def test_a_kill_during_the_sources_deletion_reruns_to_completion_from_the_targets_attribute(
    v30_tree: Path, monkeypatch
):
    """Property: a kill inside the source delete leaves the done attribute on the target, and the re-run removes the
    rest of the source with the target's timeseries files unchanged.
    Class: NEW CAPABILITY. Plausible wrong implementation: writing the attribute after the delete, which leaves no
        record of a verified copy. The kill firing is asserted.
    Why required: record before delete is the rule for every verified deletion.
    Killed by: recording after the delete.
    Passes under a second correct implementation: recording the verification in a log field instead of a root
        attribute.
    Not established: the red-on-the-pre-change-tree output was not captured this round; the in-test control is the
        substitute.
    """
    real_rmtree = shutil.rmtree
    calls = []

    def killed(path, *a, **k):
        """Half-delete the series store and raise, as a process killed inside the delete would."""
        if Path(path).name == "analysis_timeseries.zarr":
            _half_delete(Path(path))
            calls.append(path)
            raise OSError("killed during the delete")
        return real_rmtree(path, *a, **k)

    monkeypatch.setattr(shutil, "rmtree", killed)
    with pytest.raises(OSError, match="killed during the delete"):
        _migrate(v30_tree)
    monkeypatch.undo()
    target = v30_tree / "analysis_datatree.zarr"
    assert zarr_root_attrs(target)["migrated_from_analysis_timeseries"] == "done", "record BEFORE the delete"
    assert (v30_tree / "analysis_timeseries.zarr").exists() and state.read_version_file(v30_tree).layout_version == 30
    before = _files_under(target, NODE_TS)
    _migrate(v30_tree)
    assert not (v30_tree / "analysis_timeseries.zarr").exists()
    assert _files_under(target, NODE_TS) == before
    assert state.read_version_file(v30_tree).layout_version == 31
    assert calls, "control: the kill fired"


def test_a_half_deleted_source_with_no_attribute_is_refused_and_never_used_as_a_copy_source(v30_tree: Path):
    """Property: a source that lost a metadata document and has no done attribute on the target is refused and nothing
    is copied.
    Class: NEW CAPABILITY. Plausible wrong implementation: copying from a damaged source, whose own read-back passes
        because zarr reads a missing chunk as fill. Control for the next test: the same damage with the attribute is
        finished.
    Why required: a copy of fill values would pass its own read-back and then the source would be deleted.
    Killed by: no completeness check of the source.
    Passes under a second correct implementation: a check against a manifest kept outside the store.
    Not established: the red-on-the-pre-change-tree output was not captured this round; the in-test control is the
        substitute.
    """
    source = v30_tree / "analysis_timeseries.zarr"
    victims = [p for p in sorted(source.rglob("zarr.json")) if p.parent != source][:1]
    victims[0].unlink()
    with pytest.raises(ValueError, match="not used as a copy source"):
        _migrate(v30_tree)
    assert source.exists()
    assert NODE_TS not in {p.lstrip("/") for p, _ in _open(v30_tree, consolidated=False).subtree_with_keys}


def test_a_half_deleted_source_after_the_attribute_is_finished_without_touching_the_target(v30_tree: Path, monkeypatch):
    """Property: with the done attribute on the target, a source half-deleted by a kill is removed without comparison
    and the target's timeseries files are unchanged.
    Class: NEW CAPABILITY. Plausible wrong implementation: re-verifying a half-deleted source against the target,
        which fails and blocks the migration.
    Why required: a partly deleted source cannot be compared against.
    Killed by: re-verifying after the attribute.
    Passes under a second correct implementation: removing the source in the same step as the attribute check.
    Not established: the red-on-the-pre-change-tree output was not captured this round; the in-test control is the
        substitute.
    """
    monkeypatch.setattr(shutil, "rmtree", lambda *a, **k: (_ for _ in ()).throw(OSError("kill before the delete")))
    with pytest.raises(OSError):
        _migrate(v30_tree)
    monkeypatch.undo()
    _half_delete(v30_tree / "analysis_timeseries.zarr")
    target = v30_tree / "analysis_datatree.zarr"
    before = _files_under(target, NODE_TS) | _files_under(target, LINK_TS)
    _migrate(v30_tree)
    assert _files_under(target, NODE_TS) | _files_under(target, LINK_TS) == before
    assert not (v30_tree / "analysis_timeseries.zarr").exists()


def _kill_after(monkeypatch, step: str):
    real_rename, real_group = ctx_mod.os.rename, ctx_mod._write_plain_group
    count = {"rename": 0}

    def rename(src, dst):
        """Count renames; raise before or after the second one, as the scenario's step requires."""
        count["rename"] += 1
        if step == "after_step_3" and count["rename"] == 2:
            real_rename(src, dst)
            raise OSError("killed after step 3")
        if step == "before_step_3" and count["rename"] == 2:
            raise OSError("killed before step 3")
        return real_rename(src, dst)

    def group(directory, **kw):
        """Raise where the scenario's step says the process was killed around the group-document write."""
        if step == "after_step_1":
            raise OSError("killed after step 1")
        if step == "between_mkdir_and_document":
            Path(directory).mkdir(parents=True, exist_ok=True)
            raise OSError("killed between mkdir and the group document")
        return real_group(directory, **kw)

    monkeypatch.setattr(ctx_mod.os, "rename", rename)
    monkeypatch.setattr(ctx_mod, "_write_plain_group", group)


@pytest.mark.parametrize("step", ["after_step_1", "between_mkdir_and_document", "before_step_3", "after_step_3"])
def test_a_summary_move_killed_at_each_step_completes_on_rerun(v30_tree: Path, monkeypatch, step: str):
    """Property: a summary move killed after step 1, between mkdir and the group document, before step 3 or after step
    3 completes on re-run with every array equal and the layout stamp 31.
    Class: NEW CAPABILITY. Plausible wrong implementation: reading step 2 as done from the directory alone, which
        renames into a directory that is not yet a group. The kill is asserted to fire in each case.
    Why required: a group cannot move into its own subdirectory, so the move is three directory moves and a
        write and every state must resume.
    Killed by: a resume that trusts the directory.
    Passes under a second correct implementation: a move done through a copy and a delete, if it resumes from any
        kill.
    Not established: the red-on-the-pre-change-tree output was not captured this round; the in-test control is the
        substitute.
    """
    expected = _open(v30_tree)["tritonswmm/swmm_node"].to_dataset().load()
    _kill_after(monkeypatch, step)
    with pytest.raises(OSError, match="killed"):
        _migrate(v30_tree)
    monkeypatch.undo()
    tree_dir = v30_tree / "analysis_datatree.zarr" / "tritonswmm"
    if step == "after_step_1":
        assert (tree_dir / "swmm_node__v30_moving").is_dir(), "control: the kill left a half-moved group"
    if step == "between_mkdir_and_document":
        assert (tree_dir / "swmm_node").is_dir() and not (tree_dir / "swmm_node" / "zarr.json").exists()
    assert state.read_version_file(v30_tree).layout_version == 30, "a killed migration is unstamped"
    _migrate(v30_tree)
    tree = _open(v30_tree)
    xr.testing.assert_identical(tree["tritonswmm/swmm_node/summary"].to_dataset().load(), expected)
    assert (tree_dir / "swmm_node" / "zarr.json").is_file()
    assert not any(p.name.endswith("__v30_moving") for p in tree_dir.iterdir())
    assert state.read_version_file(v30_tree).layout_version == 31


def test_a_rerun_with_every_move_done_but_the_pre_move_index_rebuilds_the_index(v30_tree: Path):
    """Property: a tree whose moves are complete but whose root index is the pre-move one lists the summary node after
    a re-run.
    Class: NEW CAPABILITY. Plausible wrong implementation: skipping the index rebuild when every move is already
        done. Control inside the test: the stale index lists no summary node.
    Why required: the rebuild runs on every run that reaches the end.
    Killed by: a rebuild only after a move.
    Passes under a second correct implementation: a rebuild done lazily at the first read.
    Not established: the red-on-the-pre-change-tree output was not captured this round; the in-test control is the
        substitute.
    """
    target = v30_tree / "analysis_datatree.zarr"
    pre_move_root = (target / "zarr.json").read_text()
    _migrate(v30_tree)
    stale = target / "zarr.json"
    stale.write_text(pre_move_root)
    assert state.read_version_file(v30_tree).layout_version == 31
    runner.baseline(v30_tree, 30, force=True)
    listed_before = {p.lstrip("/") for p, _ in _open(v30_tree).subtree_with_keys}
    assert "tritonswmm/swmm_node/summary" not in listed_before, "control: the stale index lists no summary node"
    _migrate(v30_tree)
    listed = {p.lstrip("/") for p, _ in _open(v30_tree).subtree_with_keys}
    assert "tritonswmm/swmm_node/summary" in listed
    assert state.read_version_file(v30_tree).layout_version == 31


def test_a_version_1_calendar_group_without_a_newer_copy_is_kept_and_the_stamp_is_written_last(
    tmp_path: Path, monkeypatch
):
    """Property: a version-1 calendar group with no version-2 copy is kept at timeseries_legacy_calendar with the
    reason attribute, and a migration killed before the stamp leaves the tree stamped 30 and re-runs.
    Class: NEW CAPABILITY. Plausible wrong implementation: deleting the only copy of a version-1 series, or stamping
        before the moves.
    Why required: the version-1 group is the only record of its series.
    Killed by: deleting it; stamping early.
    Passes under a second correct implementation: keeping it at another path with the same reason.
    Not established: the red-on-the-pre-change-tree output was not captured this round; the in-test control is the
        substitute.
    """
    work = tmp_path / "v30"
    work.mkdir()
    _write(work / "analysis_datatree.zarr", {"tritonswmm/swmm_node": _summary(), NODE_SERIES: _legacy_calendar()})
    runner.baseline(work, 30)
    expected = _legacy_calendar()
    # a kill AFTER every move and BEFORE the layout stamp is written
    real_record = state.record_migration

    def killed(*a, **k):
        """Stand in for the stamp write of a migration killed after its last move."""
        raise OSError("killed before the stamp")

    monkeypatch.setattr(state, "record_migration", killed)
    with pytest.raises(OSError, match="before the stamp"):
        _migrate(work)
    monkeypatch.setattr(state, "record_migration", real_record)
    assert state.read_version_file(work).layout_version == 30, (
        "control: a migration stopped before the stamp is unstamped"
    )
    _migrate(work)
    tree = _open(work)
    kept = tree["tritonswmm/swmm_node/timeseries_legacy_calendar"].to_dataset().load()
    xr.testing.assert_equal(kept, expected)
    assert not (work / "analysis_datatree.zarr" / NODE_SERIES).exists()
    assert "timeseries_legacy_calendar_reason" in zarr_root_attrs(work / "analysis_datatree.zarr")
    assert state.read_version_file(work).layout_version == 31


def test_a_version_1_calendar_group_is_deleted_when_a_version_2_copy_exists(v30_tree: Path):
    # add the version-1 group next to the tree's summaries; analysis_timeseries.zarr supplies the version-2 copy
    """Property: a version-1 calendar group is deleted when the tree holds a version-2 copy, and no calendar-axis group
    is kept.
    Class: NEW CAPABILITY. Plausible wrong implementation: keeping every version-1 group, which leaves a duplicate
        series.
    Why required: the no-duplication rule.
    Killed by: keeping both.
    Passes under a second correct implementation: deleting the version-1 group before copying, if the copy is verified
        first.
    Not established: the red-on-the-pre-change-tree output was not captured this round; the in-test control is the
        substitute.
    """
    v1_group = _legacy_calendar()
    v1_group.to_zarr(v30_tree / "analysis_datatree.zarr", group=NODE_SERIES, mode="w", consolidated=False)
    _migrate(v30_tree)
    tree = _open(v30_tree)
    assert not (v30_tree / "analysis_datatree.zarr" / NODE_SERIES).exists()
    assert "timeseries" in tree["tritonswmm/swmm_node"].children
    assert "timeseries_legacy_calendar" not in tree["tritonswmm/swmm_node"].children


def test_the_migration_deletes_no_per_scenario_summary_store(v30_tree: Path):
    """Property: a per-scenario summary store under sims is byte-identical and present after the migration.
    Class: NEW CAPABILITY. Plausible wrong implementation: a walk that reaches sims and clears summaries, which
        belongs to a later chunk.
    Why required: an existing analysis keeps every per-scenario summary.
    Killed by: a walk that enters sims.
    Passes under a second correct implementation: a walk that enters sims and ignores summary stores by name.
    Not established: the red-on-the-pre-change-tree output was not captured this round; the in-test control is the
        substitute.
    """
    store = v30_tree / "sims" / "event_id.0" / "processed" / "TRITONSWMM_SWMM_node_summary.zarr"
    _summary().to_zarr(store, mode="w")
    before = _digest(store)
    _migrate(v30_tree)
    assert store.is_dir() and _digest(store) == before


def test_the_migration_leaves_every_summary_store_and_writes_no_summary_clearing_record(v30_tree: Path):
    """Property: after migrating the layout-30 tree every per-scenario summary store under sims is present and
    byte-identical, and no model log carries a ``consolidated_summaries_cleared`` record.
    Class: NEW CAPABILITY. Plausible wrong implementation: a migration that backfills the clearing record, which would
        make a tree whose stores still exist read as cleared and send readers to a branch slice that was never
        read back. Control: the record belongs to the consolidation, which
        ``tests/test_summary_store_clearing.py::test_with_the_switch_on_every_scenarios_summary_stores_are_deleted_after_the_read_back``
        runs with the switch on; this migration test writes none.
    Why required: the migration reads only trees, and the first consolidation at the new version decides what to clear.
    Killed by: a backfill of the record; a walk that deletes a summary store.
    Passes under a second correct implementation: a migration that also touches the model logs and leaves the key out.
    Not established: the red-on-the-pre-change-tree output was not captured this round.
    """
    import json

    sims = v30_tree / "sims" / "event_id.0"
    stores = [
        sims / "processed" / f"{stem}.zarr" for stem in ("TRITONSWMM_SWMM_node_summary", "TRITONSWMM_TRITON_summary")
    ]
    for store in stores:
        _summary().to_zarr(store, mode="w")
    log_file = sims / "log_tritonswmm.json"
    log_file.write_text(json.dumps({"simulation_completed": True}))
    before = {store: _digest(store) for store in stores}
    _migrate(v30_tree)
    assert all(store.is_dir() and _digest(store) == digest for store, digest in before.items())
    for found in v30_tree.glob("sims/*/log_*.json"):
        assert "consolidated_summaries_cleared" not in json.loads(found.read_text())
    assert json.loads(log_file.read_text()) == {"simulation_completed": True}


def test_a_master_tree_gets_the_same_summary_rename_under_each_member(tmp_path: Path):
    """Property: a sensitivity master tree gets {kind}/summary under each member node and the group itself holds no
    data.
    Class: NEW CAPABILITY. Plausible wrong implementation: renaming only analysis trees.
    Why required: the master's readers are re-pointed to the summary branch.
    Killed by: skipping master trees.
    Passes under a second correct implementation: renaming through the master's graft at its next consolidation.
    Not established: the red-on-the-pre-change-tree output was not captured this round; the in-test control is the
        substitute.
    """
    work = tmp_path / "v30"
    work.mkdir()
    _write(
        work / "experiment_datatree.zarr",
        {
            "member_0/tritonswmm/performance": xr.Dataset(
                {"Total": ("event_iloc", [1.0, 2.0])}, coords={"event_iloc": [0, 1]}
            )
        },
    )
    runner.baseline(work, 30)
    _migrate(work)
    tree = xr.open_datatree(work / "experiment_datatree.zarr", engine="zarr", consolidated=True)
    node = tree["member_0/tritonswmm/performance"]
    assert "summary" in node.children and not node.to_dataset().data_vars


def test_the_layout_version_is_31_and_the_v31_fixture_is_a_byte_copy_of_v30():
    """Property: LAYOUT_VERSION is 31 and the v31 fixture is a byte copy of v30.
    Class: NEW CAPABILITY. Plausible wrong implementation: a v31 fixture that differs, or no bump. A non-empty copy
        is the control.
    Why required: the layout checker requires both fixtures.
    Killed by: a missing or altered fixture.
    Passes under a second correct implementation: a fixture built by another generator with the same files.
    Not established: the red-on-the-pre-change-tree output was not captured this round; the in-test control is the
        substitute.
    """
    import filecmp

    assert LAYOUT_VERSION == 31
    cmp = filecmp.dircmp(FIXTURE_ROOT / "v30", FIXTURE_ROOT / "v31")

    def walk(c: filecmp.dircmp) -> list[str]:
        """Every path that differs, appears on one side only, or cannot be compared, under ``c``."""
        bad = list(c.left_only) + list(c.right_only) + list(c.diff_files) + list(c.funny_files)
        for sub in c.subdirs.values():
            bad += walk(sub)
        return bad

    assert walk(cmp) == []
    assert len(list((FIXTURE_ROOT / "v31").rglob("*"))) > 90, "control: the copy is not empty"


# ---- The sensitivity master's separate series store ----

MASTER_NODE_SERIES = "member_0/tritonswmm/swmm_node_timeseries"
MASTER_LINK_SERIES = "member_1/tritonswmm/swmm_link/timeseries"
MASTER_NODE_TS = "member_0/tritonswmm/swmm_node/timeseries"
MASTER_LINK_TS = "member_1/tritonswmm/swmm_link/timeseries"


@pytest.fixture
def v30_master(tmp_path: Path) -> Path:
    """A stamped layout-30 sensitivity master: summaries at ``member_*/{family}/{kind}``, a separate series store.

    The series store names its groups both ways a master could carry them: the layout-30 ``{kind}_timeseries`` for
    member 0 and the ``{kind}/timeseries`` written by the current build for member 1, so the fold is exercised
    over both names.
    """
    work = tmp_path / "v30master"
    work.mkdir()
    _write(
        work / "experiment_datatree.zarr",
        {"member_0/tritonswmm/swmm_node": _summary(1.0), "member_1/tritonswmm/swmm_link": _summary(2.0)},
    )
    _write(work / "sensitivity_timeseries.zarr", {MASTER_NODE_SERIES: _series(0.0), MASTER_LINK_SERIES: _series(5.0)})
    runner.baseline(work, 30)
    return work


def _master_store(work: Path) -> Path:
    return work / "experiment_datatree.zarr"


def test_the_sensitivity_series_store_is_folded_into_the_master_read_back_equal_and_removed(v30_master: Path):
    """Property: a layout-30 sensitivity fixture's ``sensitivity_timeseries.zarr`` is copied into the master's
    ``member_{id}/{family}/{kind}/timeseries`` branches (from either source naming), read back equal, then removed,
    and the master root records ``migrated_from_sensitivity_timeseries: done``.
    Class: NEW CAPABILITY. Plausible wrong implementation: leaving the separate store (the master then holds the
        series twice) or removing it without copying. Reachability: the fixture holds both group names. Both-states
        anchor: the values of the copied groups against the values written into the source. Control: the second
        member's values differ from the first's, so a fold that copies one source twice fails.
    Why required: D20 retires ``sensitivity_timeseries.zarr`` and folds an existing one into the master.
    Killed by: not queueing the fold; handling only one source naming.
    Passes under a second correct implementation: copying through the shared context primitive.
    Not established: the red on the pre-change tree was not captured; the in-test control is the substitute.
    """
    source_node = xr.open_zarr(v30_master / "sensitivity_timeseries.zarr", group=MASTER_NODE_SERIES, chunks=None).load()
    source_link = xr.open_zarr(v30_master / "sensitivity_timeseries.zarr", group=MASTER_LINK_SERIES, chunks=None).load()
    _migrate(v30_master)
    assert not (v30_master / "sensitivity_timeseries.zarr").exists()
    tree = xr.open_datatree(_master_store(v30_master), engine="zarr", consolidated=True)
    assert ctx_mod._read_group_equal(tree[MASTER_NODE_TS].to_dataset().load(), source_node)
    assert ctx_mod._read_group_equal(tree[MASTER_LINK_TS].to_dataset().load(), source_link)
    assert not ctx_mod._read_group_equal(tree[MASTER_NODE_TS].to_dataset().load(), source_link)
    assert zarr_root_attrs(_master_store(v30_master))["migrated_from_sensitivity_timeseries"] == "done"
    assert "summary" in tree["member_0/tritonswmm/swmm_node"].children
    assert state.read_version_file(v30_master).layout_version == 31


def test_a_fold_made_to_fail_the_read_back_keeps_the_sensitivity_series_store(v30_master: Path, monkeypatch):
    """Property: when the copy does not read back equal the series store stays, no completion attribute is written and
    the layout stamp is unchanged.
    Class: NEW CAPABILITY. Plausible wrong implementation: deleting the source without the read-back. Must-fire
        control: the module's comparison is replaced by one that answers False.
    Why required: the source is the only copy until the read-back passes.
    Killed by: deleting first.
    Passes under a second correct implementation: a read-back through a different reader.
    Not established: the red on the pre-change tree was not captured; the in-test control is the substitute.
    """
    monkeypatch.setattr(ctx_mod, "_read_group_equal", lambda a, b: False)
    with pytest.raises(ValueError, match="does not read back equal"):
        _migrate(v30_master)
    assert (v30_master / "sensitivity_timeseries.zarr").is_dir()
    assert "migrated_from_sensitivity_timeseries" not in zarr_root_attrs(_master_store(v30_master))
    assert state.read_version_file(v30_master).layout_version == 30


def test_a_sensitivity_series_store_half_deleted_after_its_attribute_is_finished_without_touching_the_master(
    v30_master: Path, monkeypatch
):
    """Property: with ``migrated_from_sensitivity_timeseries`` on the master, a series store half-deleted by a kill is
    removed without comparison and the master's ``timeseries`` files are unchanged; the same half-deleted store with
    no attribute is refused and never used as a copy source.
    Class: NEW CAPABILITY. Plausible wrong implementation: re-verifying a half-deleted source against the master.
        Must-fire control: the no-attribute arm, where a metadata document of the source is missing.
    Why required: a partly deleted source cannot be compared against.
    Killed by: re-verifying after the attribute; no completeness check before the attribute.
    Passes under a second correct implementation: removing the source in the step that checks the attribute.
    Not established: the red on the pre-change tree was not captured; the in-test control is the substitute.
    """
    refused = v30_master.parent / "refused"
    shutil.copytree(v30_master, refused)
    victim = next(
        p for p in sorted((refused / "sensitivity_timeseries.zarr").rglob("zarr.json")) if p.parent.name == "depth"
    )
    victim.unlink()
    with pytest.raises(ValueError, match="not used as a copy source"):
        _migrate(refused)
    assert (refused / "sensitivity_timeseries.zarr").exists()

    real_rmtree = shutil.rmtree

    def killed(path, *a, **k):
        """Half-delete the series store and raise, as a process killed inside the delete would."""
        if Path(path).name == "sensitivity_timeseries.zarr":
            _half_delete(Path(path))
            raise OSError("killed during the delete")
        return real_rmtree(path, *a, **k)

    monkeypatch.setattr(shutil, "rmtree", killed)
    with pytest.raises(OSError, match="killed during the delete"):
        _migrate(v30_master)
    monkeypatch.undo()
    target = _master_store(v30_master)
    assert zarr_root_attrs(target)["migrated_from_sensitivity_timeseries"] == "done", "record BEFORE the delete"
    before = _files_under(target, MASTER_NODE_TS) | _files_under(target, MASTER_LINK_TS)
    _migrate(v30_master)
    assert not (v30_master / "sensitivity_timeseries.zarr").exists()
    assert _files_under(target, MASTER_NODE_TS) | _files_under(target, MASTER_LINK_TS) == before
