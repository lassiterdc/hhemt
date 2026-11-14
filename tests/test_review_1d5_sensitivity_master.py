"""Reviewer tests for the sensitivity master copies and removes member trees ONE MEMBER AT A TIME.

Approved shape D20 states the disk consequence of the removal order: "Peak disk: the members' trees plus the master,
minus each member tree as it moves, so at most one member tree is held twice at a time", and the chunk brief says the
master copies each member's branches "one member at a time with the verified read-back, then remove[s] the member's
`analysis_datatree.zarr` under the clearing switch". The rows here reuse the builder's stand-in harness
(``tests.test_sensitivity_store_no_redundancy``) so the two files observe the same three production steps.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from hhemt import sensitivity_analysis as sa
from tests.test_sensitivity_store_no_redundancy import _Master, _Member


@pytest.fixture(autouse=True)
def _first_scenario_has_a_node_summary(monkeypatch):
    """The stand-in members produce one summary mode, as the builder's file arranges."""
    monkeypatch.setattr(sa, "_first_scenario_paths", lambda analysis: SimpleNamespace(swmm_node_summary=Path("x")))


def _three_members(root: Path, *, clearing: bool) -> _Master:
    members = {str(i): _Member(root, str(i), clearing=clearing, scale=float(10**i)) for i in range(3)}
    return _Master(root, members)


def _trees_present_when_each_member_copy_starts(master: _Master, monkeypatch) -> dict[str, list[bool]]:
    """For each member id, whether each member's tree (in member order) exists at the instant that member's FIRST
    series-group copy begins. Unit: one first-copy call per member. Matcher: the ``member_id`` keyword of
    ``sa._copy_series_group`` and ``Path.exists`` of every member's ``analysis_datatree.zarr``."""
    seen: dict[str, list[bool]] = {}
    real = sa._copy_series_group

    def spy(*args, member_id, **kwargs):
        seen.setdefault(member_id, [m.tree.exists() for m in master.members.values()])
        return real(*args, member_id=member_id, **kwargs)

    monkeypatch.setattr(sa, "_copy_series_group", spy)
    master.run()
    monkeypatch.undo()
    return seen


def test_with_clearing_on_a_member_tree_is_gone_before_the_next_member_copy_begins(tmp_path, monkeypatch):
    """Property: for every member k > 0 of a clearing-on master, the trees of members 0..k-1 are absent when member k's
    series copy begins, so no more than one member tree is ever held beside the master's copy of it.
    Class: REGRESSION against the shape text, over the class "every member after the first" (members 1 and 2 of three),
        so a single-position fix (moving only the first tree) fails at member 2. Red on the chunk tree as it stands:
        the chunk copies every member and only then removes any tree, so all three trees exist when member 1 and
        member 2 begin. Reachability: ``Master.run`` is the builder's mirror of the three production steps and the
        master below has clearing on for all three members, so every member is copied from a present tree.
        Both-states anchor: the assertion turns on whether earlier members' trees exist at the instant a later
        member's copy starts, which is true or false under any implementation, not on wording.
    Why required: D20's peak-disk statement is the reason the removal is per member; with all trees held at once the
        peak is the members' series twice over (every member tree plus the master's copy of all of them), which is the
        footprint the clearing switch exists to avoid on a campaign whose member series are gridded.
    Killed by: moving the removal back after the whole copy loop.
    Passes under a second correct implementation: interleaving copy and removal inside ``_write_master_in_place``, or
        driving the three steps once per member (the control arm below does the latter).
    Not established: whether the next member's copy may begin before the previous member's removal is RECORDED as
        ``done`` (the test asks only that the tree is gone); and whether the production
        ``consolidate_sensitivity_datatree`` orders its calls as ``Master.run`` does (read from the diff, not
        executed).
    """
    seen = _trees_present_when_each_member_copy_starts(_three_members(tmp_path, clearing=True), monkeypatch)
    assert sorted(seen) == ["0", "1", "2"]
    for k, flags in seen.items():
        earlier_present = [present for idx, present in enumerate(flags) if idx < int(k)]
        assert earlier_present == [False] * int(k), f"member {k} began copying with earlier trees present: {flags}"


def test_the_observer_sees_every_tree_when_clearing_is_off_and_sees_none_held_when_driven_member_by_member(
    tmp_path, monkeypatch
):
    """Control arms for the row above, which vary the property under suspicion (the ORDER of copy and removal).
    Arm 1, clearing off: no tree is ever removed, so the observer must read every tree present at every copy start.
    Arm 2, clearing on and the master driven once per member (copy then removal of that member before the next): the
        observer must read earlier trees absent, which shows the row above discriminates an interleaved order from the
        batched one rather than failing on any ordering.
    Class: control. Reachability: both arms run the builder's ``Master.run``. Both-states anchor: tree existence at the
        first-copy instant. Killed by: an observer that always reads False (arm 1) or always reads True (arm 2).
    Passes under a second correct implementation: any orchestration that copies a member before removing it.
    """
    off = _trees_present_when_each_member_copy_starts(_three_members(tmp_path / "off", clearing=False), monkeypatch)
    assert all(all(flags) for flags in off.values()), off

    master = _three_members(tmp_path / "by_member", clearing=True)
    all_members = dict(master.members)
    seen: dict[str, list[bool]] = {}
    real = sa._copy_series_group

    def spy(*args, member_id, **kwargs):
        seen.setdefault(member_id, [m.tree.exists() for m in all_members.values()])
        return real(*args, member_id=member_id, **kwargs)

    monkeypatch.setattr(sa, "_copy_series_group", spy)
    for member_id, member in all_members.items():
        master.members = {member_id: member}
        master.run()
    master.members = all_members
    monkeypatch.undo()
    for k, flags in seen.items():
        assert [p for idx, p in enumerate(flags) if idx < int(k)] == [False] * int(k), (k, flags)
