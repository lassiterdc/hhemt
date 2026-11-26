"""Reviewer tests for chunk 1h.1 (DR-18): the master copies every data group and the move refuses to lose one.

The coder's stand-ins (``_Master``, ``_Member``) are reused so these tests exercise the same code path. Each test names
the property it holds and the input class it varies; none weakens a coder test.
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest
import xarray as xr

from hhemt import sensitivity_analysis as sa
from hhemt.exceptions import ProcessingError
from hhemt.utils import append_zarr
from tests import test_experiment_master_carries_every_member_group as coder
from tests.test_experiment_master_carries_every_member_group import (
    EDGE_FLUX,
    NODE_PATH3,
    NOVEL,
    STATIC,
    SUMMARY,
    _Master,
)

pytestmark = pytest.mark.usefixtures("_first_scenario_has_a_node_summary")
_first_scenario_has_a_node_summary = coder._first_scenario_has_a_node_summary


def _refusal(master: _Master, member_id: str, sources: dict, copied: dict) -> str:
    with pytest.raises(ProcessingError) as err:
        master._move_member_tree(member_id, sources, copied)
    return str(err.value)


def test_the_refusal_names_every_absent_group_not_only_the_first(tmp_path):
    """Property: the refusal text lists EVERY data group the master lacks (the shape's text is "the master holds no
    copy of {groups}"), the member tree stays, and nothing is recorded. Input class: three absent groups of three
    kinds: a merged ``summary``, a group with ``event_iloc``, a group without it."""
    master = _Master(tmp_path)
    sources, copied = master.write(move=False)
    member = master.members["0"]
    absent = [SUMMARY, EDGE_FLUX, STATIC]
    for group in absent:
        shutil.rmtree(master.store / "member_0" / group)
    text = _refusal(master, "0", sources, copied)
    assert "member 0" in text, text
    for group in absent:
        assert group in text, (group, text)
    for held in (NODE_PATH3, NOVEL):
        assert held not in text, (held, text)
    assert member.tree.exists()
    assert member.log.member_tree_moved_to_master.get() is None
    assert member.log.member_tree_moved_groups.get() is None


def test_the_refusal_fires_when_the_master_group_holds_no_variables(tmp_path):
    """Property: a group document present in the master with NO variables is a group the master lacks, so the move
    refuses and keeps the tree. Input class: the group document exists, its data is gone (a copy killed after the group
    was created and before it was filled)."""
    master = _Master(tmp_path)
    sources, copied = master.write(move=False)
    member = master.members["0"]
    append_zarr(xr.Dataset(), master.store, group=f"member_0/{NODE_PATH3}", mode="w")
    emptied = xr.open_zarr(master.store, group=f"member_0/{NODE_PATH3}", consolidated=True)
    assert not emptied.data_vars, "control: the stored group holds no variables"
    text = _refusal(master, "0", sources, copied)
    assert NODE_PATH3 in text and "member 0" in text, text
    assert member.tree.exists()


def test_the_refusal_fires_when_the_master_has_no_node_for_the_member(tmp_path):
    """Property: with the member's whole node absent from the master, the move refuses naming every data group of the
    member's tree, and keeps the tree. Input class: the node is missing, not one group of it."""
    master = _Master(tmp_path)
    sources, copied = master.write(move=False)
    member = master.members["0"]
    shutil.rmtree(master.store / "member_0")
    text = _refusal(master, "0", sources, copied)
    for group in coder.DATA_GROUPS:
        assert group in text, (group, text)
    assert member.tree.exists()


def test_a_member_group_that_holds_only_coordinates_is_copied_and_does_not_block_the_move(tmp_path, monkeypatch):
    """Property: the rule "every data node of the member tree" and the refusal's test of "the master holds a copy" agree
    on one input class, so a group the copy step wrote never makes the move refuse. Input class: a member group holding
    a coordinate and no data variable (xarray's ``DataTree.has_data`` is True for it, so the copy rule selects it).
    Must-fire control: the group is selected by the rule."""
    coords_only = "tritonswmm/a_group_added_later/axis_only"
    original = coder._member_groups
    monkeypatch.setattr(
        coder,
        "_member_groups",
        lambda scale: {**original(scale), coords_only: xr.Dataset(coords={"k": np.arange(4)})},
    )
    master = _Master(tmp_path)
    assert coords_only in sa._member_tree_data_groups(master.members["0"].tree), "control: the rule selects the group"
    master.write(move=True)
    assert not any(m.tree.exists() for m in master.members.values()), "the move must complete when every group is held"
