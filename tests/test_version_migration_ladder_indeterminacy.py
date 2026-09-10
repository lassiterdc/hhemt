"""[Q292]: an undatable tree MUST NOT resolve to a plannable layout version.

A separate module from `test_version_migration_state.py` deliberately. That module pins the
legacy-iloc rung (`== 0`) and `test_version_migration_V0022.py` pins the unified-store rung
(`== 21`); NOTHING pins the zarr-attrs rung, which is why it returned an exact `4` from
evidence establishing only "post-V0004" for eighteen layout versions without a test noticing.
It is also where `A6` is owed, so a sibling module cannot collide with it.

EVERY MEMBER-TIER ASSERTION HERE IS MUTATION-ADEQUATE, and that is the point rather than a
bonus. Three plausible wrong implementations exist and each is caught by exactly one test:

    mutation                                        caught by
    return LAYOUT_VERSION, not master.layout_version  test_..._is_dated_from_its_master (master@20)
    member rung placed above the target's own record  test_a_members_own_record_still_wins
    member rung placed above the legacy-content rung  test_positive_legacy_content_outranks_...

The first is the mistake the neighbouring rung's comment forbids BY NAME ("Do not replace this
with LAYOUT_VERSION") because it was already made once and shipped a silent total skip. A test
that stamps the master AT `LAYOUT_VERSION` cannot see it, since the correct value and the bug's
value coincide -- so these tests stamp the master at a DIFFERENT version on purpose.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hhemt.version_migration import runner, state
from hhemt.version_migration.constants import LAYOUT_VERSION
from hhemt.version_migration.exceptions import BaselineRequiredError

#: A stamp value that is deliberately NOT `LAYOUT_VERSION`. Every member-tier test below
#: stamps the master here so that "read the master's record" and "return the module
#: constant" produce DIFFERENT answers; at `LAYOUT_VERSION` they coincide and the
#: assertion goes green under the forbidden substitution.
_OLDER_THAN_CURRENT = LAYOUT_VERSION - 2


def _datatree(target_dir: Path, *, conventions: str | None = "CF-1.13") -> Path:
    """A minimal readable zarr v3 group at the analysis-datatree name."""
    target_dir.mkdir(parents=True, exist_ok=True)
    store = target_dir / "analysis_datatree.zarr"
    store.mkdir()
    attrs: dict = {"analysis_id": "probe"}
    if conventions is not None:
        attrs["Conventions"] = conventions
    store.joinpath("zarr.json").write_text(
        json.dumps({"attributes": attrs, "node_type": "group", "zarr_format": 3}),
        encoding="utf-8",
    )
    return target_dir


def _legacy_sims(target_dir: Path) -> Path:
    """Pre-Phase-0 iloc-prefixed sims: positive, in-target evidence of layout 0."""
    (target_dir / "sims" / "0-event_id.0").mkdir(parents=True)
    return target_dir


def _stamp(target_dir: Path, version: int) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    target_dir.joinpath("_version.json").write_text(
        json.dumps(
            {
                "layout_version": version,
                "toolkit_version": "0.1.0",
                "created_at": "2026-01-01T00:00:00Z",
                "migration_history": [],
            }
        ),
        encoding="utf-8",
    )


def test_an_undatable_tree_does_not_resolve_to_a_plannable_version(tmp_path: Path) -> None:
    """THE [Q292] ASSERTION. A CF-1.13 datatree with no record establishes only post-V0004.

    Pre-fix this resolved to 4 and planned eighteen migrations against a CURRENT tree,
    including the vocabulary rename and the store rename against a tree that already
    carries both. The assertion is on the PLAN, not on any message, so it discriminates on
    behaviour in both the pre-fix and post-fix worlds.
    """
    work = _datatree(tmp_path / "undatable")

    with pytest.raises(BaselineRequiredError):
        runner.status(work)


def test_the_zarr_attrs_rung_reports_indeterminacy_rather_than_an_exact_version(
    tmp_path: Path,
) -> None:
    """The rung nothing pinned. Its evidence is a LOWER BOUND and it must say so."""
    work = _datatree(tmp_path / "post_v0004")

    assert state.infer_layout_version(work) is None


def test_a_pre_v0004_datatree_is_still_dated_exactly(tmp_path: Path) -> None:
    """The rung's one remaining exact answer: no Conventions attr means V0003, and that
    discriminator has not decayed because V0004 is what introduced the attribute."""
    work = _datatree(tmp_path / "pre_v0004", conventions=None)

    assert state.infer_layout_version(work) == 3


@pytest.mark.parametrize("container", ["members", "subanalyses"])
def test_a_member_tier_is_dated_from_its_master_not_from_its_content(tmp_path: Path, container: str) -> None:
    """A nested tier's version is not unknown -- the master's record states it.

    MUTATION-ADEQUATE BY CONSTRUCTION: the master is stamped at `_OLDER_THAN_CURRENT`, not
    at `LAYOUT_VERSION`, so returning the module constant instead of reading the record
    yields a DIFFERENT number and reddens this test. Measured: correct payload returns
    20 with a non-empty plan; the constant-returning bug returns 22 with an empty one.
    """
    master = tmp_path / "master"
    _stamp(master, _OLDER_THAN_CURRENT)
    member = _datatree(master / container / "member_0")

    assert state.infer_layout_version(member) == _OLDER_THAN_CURRENT
    # The plan must be NON-empty: the member is genuinely behind, and an empty plan is the
    # signature of the constant-returning bug reporting the tree as already current.
    assert runner.status(member).migrations_planned != []


def test_positive_legacy_content_outranks_an_inherited_master_stamp(tmp_path: Path) -> None:
    """PRECEDENCE, tier 2. Evidence read from the target beats a stamp read from elsewhere.

    A member carrying pre-Phase-0 iloc-prefixed sims is provably layout 0 from its own
    contents. An inherited stamp must not suppress that: placed above the legacy rung, the
    member rung reported 22 and planned ZERO of the 22 migrations the contents prove are
    needed -- a total silent skip, the same failure the unified-store rung records.
    """
    master = tmp_path / "master"
    _stamp(master, LAYOUT_VERSION)
    member = _legacy_sims(master / "members" / "member_0")

    assert state.infer_layout_version(member) == 0
    assert runner.status(member).migrations_planned != []


def test_a_member_under_an_unstamped_master_still_refuses(tmp_path: Path) -> None:
    """The member rung answers only where the master's record exists. With no record above
    it there is no evidence, and the honest rung below must still refuse."""
    member = _datatree(tmp_path / "master" / "members" / "member_0")

    assert state.infer_layout_version(member) is None
    with pytest.raises(BaselineRequiredError):
        runner.status(member)


def test_a_members_own_record_still_wins(tmp_path: Path) -> None:
    """PRECEDENCE, tier 1. The target's own record outranks the inherited stamp.

    MUTATION-ADEQUATE BY CONSTRUCTION: master and member are stamped at DIFFERENT versions,
    so the two candidate sources yield different numbers. Stamping both at the same value --
    as an earlier draft did -- makes this assertion hold whichever rung fires, and measured
    proof that it was vacuous: a member with NO record at all under a master stamped 22
    also returns 22. Measured here: correct precedence returns LAYOUT_VERSION; hoisting the
    member rung above the record rung returns the master's older value instead.
    """
    master = tmp_path / "master"
    _stamp(master, _OLDER_THAN_CURRENT)
    member = _datatree(master / "members" / "member_0")
    _stamp(member, LAYOUT_VERSION)

    assert state.infer_layout_version(member) == LAYOUT_VERSION


def test_the_unified_store_rung_is_unchanged(tmp_path: Path) -> None:
    """REGRESSION PIN. The rung repaired for this same bound-vs-exact error keeps its
    lower-bound answer; this module must not perturb it."""
    work = tmp_path / "unified"
    (work / "experiment_datatree.zarr").mkdir(parents=True)

    assert state.infer_layout_version(work) == 21
