"""The experiment master copies every data group of a member tree, and refuses to remove a tree it does not hold.

Design DR-18 (chunk 1.8). The master used to copy a member's ``summary`` groups (merged) and its ``timeseries``
groups, chosen by the last path component, so the seven ``coupling_diagnostics`` capture groups never reached
``experiment_datatree.zarr`` and, with clearing on, were deleted with the member's tree. The rule is now: every data
node of the member tree that the summary merge does not already carry is copied, whatever its name, and the move
refuses to delete a member tree that holds a group the master lacks.

No solver and no Snakemake: the member tree is a zarr tree written by hand under ``tmp_path``, and the member and the
master are small stand-ins that carry exactly the attributes the code under test reads.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hhemt import processing_analysis as pa
from hhemt import sensitivity_analysis as sa
from hhemt.exceptions import ProcessingError
from hhemt.utils import write_datatree_zarr

SUMMARY = "tritonswmm/swmm_node/summary"
SERIES = "tritonswmm/swmm_node/timeseries"
EDGE_FLUX = "tritonswmm/coupling_diagnostics/edge_flux"
NODE_PATH3 = "tritonswmm/coupling_diagnostics/node_path3"
NOVEL = "tritonswmm/a_group_added_later/readings"
STATIC = "tritonswmm/a_group_added_later/static_lookup"
DATA_GROUPS = sorted([SUMMARY, SERIES, EDGE_FLUX, NODE_PATH3, NOVEL, STATIC])


@pytest.fixture(autouse=True)
def _first_scenario_has_a_node_summary(monkeypatch):
    monkeypatch.setattr(sa, "_first_scenario_paths", lambda analysis: SimpleNamespace(swmm_node_summary=Path("x")))


def _events(scale: float, trailing: tuple[tuple[str, int], ...]) -> xr.Dataset:
    """A group over ``event_iloc`` and the given trailing dimensions, one float variable ``v``."""
    dims = ("event_iloc", *(name for name, _ in trailing))
    shape = (2, *(size for _, size in trailing))
    coords: dict = {"event_iloc": [0, 1], "event_id": ("event_iloc", ["e0", "e1"])}
    for name, size in trailing:
        coords[name] = np.arange(size)
    return xr.Dataset({"v": (dims, scale + np.arange(float(np.prod(shape))).reshape(shape))}, coords=coords)


def _member_groups(scale: float) -> dict[str, xr.Dataset]:
    return {
        SUMMARY: _events(scale, (("node_id", 3),)),
        SERIES: _events(scale, (("elapsed_s", 4), ("node_id", 3))),
        EDGE_FLUX: _events(scale, (("record", 5),)),
        NODE_PATH3: _events(scale, (("record", 6), ("node_id", 3))),
        NOVEL: _events(scale, (("reading", 7),)),
        STATIC: xr.Dataset({"w": (("k",), scale + np.arange(4.0))}, coords={"k": np.arange(4)}),
    }


class _Field:
    def __init__(self, value=None):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class _Process:
    _MODE_TO_TREE_PATH = {"tritonswmm_swmm_node": SUMMARY}
    _MODE_CONFIG = {"tritonswmm_swmm_node": ("swmm_node_summary",)}

    def __init__(self, member):
        self.member = member

    def open_datatree(self):
        return xr.open_datatree(self.member.analysis_paths.analysis_datatree_zarr, engine="zarr", consolidated=True)

    assert_summary_branch_layout = staticmethod(pa.TRITONSWMM_analysis_post_processing.assert_summary_branch_layout)
    _mode_of_group = pa.TRITONSWMM_analysis_post_processing._mode_of_group
    _prepare_summary_group = staticmethod(pa.TRITONSWMM_analysis_post_processing._prepare_summary_group)
    _write_summary_group = pa.TRITONSWMM_analysis_post_processing._write_summary_group

    def _summary_records(self, mode):
        return {}

    def _consolidation_inputs_fingerprint(self):
        return "fp-1"


class _Member:
    def __init__(self, root: Path, member_id: str, scale: float):
        self.directory = root / "members" / f"member_{member_id}"
        self.directory.mkdir(parents=True)
        self.tree = self.directory / "analysis_datatree.zarr"
        groups = {"/": xr.Dataset(attrs={"analysis_id": f"member_{member_id}", "consolidation_version": 5})}
        groups.update({f"/{path}": ds for path, ds in _member_groups(scale).items()})
        write_datatree_zarr(xr.DataTree.from_dict(groups), self.tree)
        self.analysis_paths = SimpleNamespace(analysis_datatree_zarr=self.tree, analysis_dir=self.directory)
        self.cfg_analysis = SimpleNamespace(
            toggle_clear_scenario_timeseries_after_consolidation=True, toggle_consolidate_timeseries=True
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
    """Carries the methods of the master that write and move, bound to the attributes they read."""

    member_prefix = "member_"
    _write_master_in_place = sa.TRITONSWMM_sensitivity_analysis._write_master_in_place
    _move_member_tree = sa.TRITONSWMM_sensitivity_analysis._move_member_tree
    _assert_member_complete_in_master = sa.TRITONSWMM_sensitivity_analysis._assert_member_complete_in_master

    def __init__(self, root: Path):
        self.members = {"0": _Member(root, "0", 1.0), "1": _Member(root, "1", 10.0)}
        self.store = root / "experiment_datatree.zarr"
        self.analysis_paths = SimpleNamespace(experiment_datatree_zarr=self.store)
        self.df_setup = pd.DataFrame({"run_mode": ["cpu", "cpu"]}, index=pd.Index(["0", "1"], name="member_id"))
        self.experiment = SimpleNamespace(cfg_analysis=SimpleNamespace(analysis_id="exp"))

    def write(self, *, move: bool) -> tuple[dict, dict]:
        tree, sources = sa._build_master_tree(self)
        tree.attrs["consolidation_version"] = pa.TRITONSWMM_analysis_post_processing.CONSOLIDATION_VERSION
        copied: dict = {}
        hook = (lambda member_id: self._move_member_tree(member_id, sources, copied)) if move else None
        self._write_master_in_place(tree, sources, self.store, copied=copied, after_member=hook)
        return sources, copied


def _master_group(master: _Master, member_id: str, group: str) -> xr.Dataset:
    return xr.open_zarr(master.store, group=f"member_{member_id}/{group}", consolidated=True, chunks=None).load()


def test_the_master_holds_every_data_group_of_every_member(tmp_path):
    """AC-1.8 (a). Property: every data node of a member tree, whatever its name, is a group of the master and reads
    back equal to the member's. Input class: ``summary``, ``timeseries``, the coupling captures, a group with a novel
    name, and a group with no ``event_iloc``. Must-fire control: the member groups are asserted to be six, and the
    novel and no-``event_iloc`` groups are neither ``summary`` nor ``timeseries`` by name."""
    master = _Master(tmp_path)
    assert len(DATA_GROUPS) == 6 and NOVEL.rsplit("/", 1)[-1] not in ("summary", "timeseries")
    master.write(move=False)
    for member_id, scale in (("0", 1.0), ("1", 10.0)):
        for group, expected in _member_groups(scale).items():
            assert sa._datasets_equal(expected.load(), _master_group(master, member_id, group)), (member_id, group)


def test_the_move_record_lists_every_group_once_and_the_summary_is_written_once(tmp_path, monkeypatch):
    """AC-1.8 (b). Property: the record of the groups the master copied holds every data group exactly once, and the
    ``summary`` group is written to the master once and not also copied. Unit: one call of ``_write_group_verified``
    or ``_copy_data_group`` per group. Matcher: the ``group`` and ``src_group`` arguments."""
    master = _Master(tmp_path)
    written: list[str] = []
    copies: list[str] = []
    real_write, real_copy = sa._write_group_verified, sa._copy_data_group
    monkeypatch.setattr(
        sa,
        "_write_group_verified",
        lambda store, group, *a, **k: (written.append(group), real_write(store, group, *a, **k))[1],
    )
    monkeypatch.setattr(
        sa, "_copy_data_group", lambda src, rel, *a, **k: (copies.append(rel), real_copy(src, rel, *a, **k))[1]
    )
    _, copied = master.write(move=True)
    for member_id in ("0", "1"):
        assert copied[member_id] == DATA_GROUPS
        assert len(copied[member_id]) == len(set(copied[member_id]))
        assert json.loads(master.members[member_id].log.member_tree_moved_groups.get()) == DATA_GROUPS
        assert written.count(f"member_{member_id}/{SUMMARY}") == 1
    assert SUMMARY not in copies
    assert sorted(set(copies)) == sorted(set(DATA_GROUPS) - {SUMMARY}) and len(copies) == 2 * 5
    assert not any(m.tree.exists() for m in master.members.values()), "control: with every group held, the trees move"


def test_the_move_refuses_a_member_whose_group_the_master_lacks_and_keeps_the_tree(tmp_path):
    """AC-1.8 (c). Property: with one group removed from the master, the move refuses with text naming the member and
    that group, the member tree still exists, and nothing is recorded as moved. Must-fire control: with the group
    restored by a second write, the same move proceeds and the tree goes."""
    import shutil

    master = _Master(tmp_path)
    sources, copied = master.write(move=False)
    member = master.members["0"]
    shutil.rmtree(master.store / "member_0" / EDGE_FLUX)
    with pytest.raises(ProcessingError) as err:
        master._move_member_tree("0", sources, copied)
    text = str(err.value)
    assert "member 0" in text and EDGE_FLUX in text and "master holds no copy" in text, text
    assert member.tree.exists()
    assert member.log.member_tree_moved_to_master.get() is None
    assert member.log.member_tree_moved_groups.get() is None
    master.write(move=False)  # the copy step restores the group
    master._move_member_tree("0", sources, copied)
    assert not member.tree.exists() and member.log.member_tree_moved_to_master.get() == sa.MOVE_DONE


def test_the_consolidation_version_is_six():
    """AC-1.8: DR-18 raises the version from 5 to 6, so every tree stamped 5 rebuilds once."""
    assert pa.TRITONSWMM_analysis_post_processing.CONSOLIDATION_VERSION == 6
