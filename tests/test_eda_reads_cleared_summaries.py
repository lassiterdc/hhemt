"""The EDA readers follow a cleared summary into the `summary` branch of the consolidated tree.

No solver and no Snakemake. The identity check and the provenance declaration run on the committed sensitivity golden
(four members, one event) with the clearing switch turned on and each member consolidated, so the per-scenario summary
stores are deleted by the real clearing. The cross-arm pairing runs on small constructed arm roots. Each test names its
control: the input class that makes a broken implementation read as correct.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import zarr

from hhemt.eda.cross_sim_identity import (
    NOT_EVALUATED,
    _cleared_slice,
    _summary_paths,
    check_cross_sim_identity,
    compare_arms,
)
from tests.fixtures.golden_trees import construct_golden_analysis

TRITON_GROUP = "tritonswmm/triton/summary"
TRITON_STEM = "TRITONSWMM_TRITON_summary"
TREE = "analysis_datatree.zarr"


@pytest.fixture(autouse=True)
def _pin_clean_build(monkeypatch):
    """Force the build-stamp term of the consolidation reuse conjunction TRUE, so a dirty checkout does not make every
    consolidation rebuild for a reason unrelated to what these tests assert."""
    import hhemt.provenance as prov

    monkeypatch.setattr(prov, "store_build_mismatch", lambda store, stamp: None)


def _clear_every_member(analysis) -> None:
    """Consolidate each member of the sensitivity golden under the clearing switch, which deletes its summary stores."""
    from hhemt.scenario import TRITONSWMM_scenario, coupling_nodes_file_for

    for member in analysis.sensitivity.members.values():
        member.cfg_analysis.toggle_consolidate_timeseries = True
        member.cfg_analysis.toggle_clear_scenario_timeseries_after_consolidation = True
        shutil.rmtree(member.analysis_paths.analysis_datatree_zarr, ignore_errors=True)
        for event_iloc in member.df_sims.index:
            scen = TRITONSWMM_scenario(event_iloc, member)
            series = xr.open_zarr(scen.scen_paths.output_tritonswmm_node_time_series, consolidated=False)
            nodes_file = coupling_nodes_file_for(scen.scen_paths.swmm_hydraulics_inp)
            nodes_file.parent.mkdir(parents=True, exist_ok=True)
            nodes_file.write_text("".join(f"{n}\n" for n in series["node_id"].values[:3]))
        member.process.consolidate_to_datatree()


def _bump_first_finite(tree: Path, group: str, variable: str) -> None:
    """Add one to the first finite value of ``variable`` in ``group`` of the zarr store ``tree``, in place."""
    array = zarr.open_group(tree, path=group, mode="r+")[variable]
    values = array[...]
    index = tuple(int(i[0]) for i in np.nonzero(np.isfinite(values)))
    values[index] += 1.0
    array[...] = values


def test_the_identity_verdict_is_the_same_after_the_summaries_are_cleared_and_fails_on_one_changed_value(
    golden_extractor,
):
    """Property: `check_cross_sim_identity` on the sensitivity golden gives the same verdict before and after every
    member's per-scenario summaries are cleared, and fails once one value of one member's `summary` slice is changed.
    Class: NEW CAPABILITY. Plausible wrong implementation: a reader that finds no per-scenario summary after the
        clearing and so skips every member (the check then reports no member has summaries, a different verdict), or
        a reader that compares the tree's join-filled slices instead of the recorded variables.
    Must-fire control: the changed-value row, which differs from the passing row by one value in one member's branch.
    Why required: with the clearing switch on the identity check can read nothing else.
    Killed by: a reader left on the per-scenario stores.
    Passes under a second correct implementation: one that reads the slices straight from the tree.
    """
    analysis = construct_golden_analysis(golden_extractor("sensitivity"))
    before = check_cross_sim_identity(analysis)
    assert before.verdict.passed and not before.skipped, "control: the uncleared golden passes the check"

    _clear_every_member(analysis)
    member = analysis.sensitivity.members["0"]
    assert not list(Path(member.analysis_paths.analysis_dir).glob("sims/*/processed/*_summary.zarr")), (
        "control: the clearing deleted every per-scenario summary store"
    )
    after = check_cross_sim_identity(analysis)
    assert after.verdict.passed and not after.skipped
    assert after.verdict.summary == before.verdict.summary

    for sub in analysis.sensitivity.members.values():
        sub.__dict__.pop("_eda_mode_cache", None)
    _bump_first_finite(
        Path(analysis.sensitivity.members["1"].analysis_paths.analysis_datatree_zarr), TRITON_GROUP, "max_flood_depth"
    )
    changed = check_cross_sim_identity(analysis)
    assert not changed.verdict.passed


def test_summary_paths_declares_the_tree_for_a_cleared_event_and_the_stores_for_a_present_one(golden_extractor):
    """Property: for a member whose events are cleared `_summary_paths` returns the consolidated tree, and for the same
    member before the clearing it returns the per-scenario stores and not the tree.
    Class: NEW CAPABILITY. Plausible wrong implementation: the landed glob alone, which declares nothing for a cleared
        event so the bundle carries no provenance source for its summaries.
    Must-fire control: the present row returns per-scenario stores, so an empty cleared result is not a broken glob.
    Why required: a figure computed from a cleared summary must declare where the summary is.
    Killed by: the glob without the tree.
    Passes under a second correct implementation: one that declares the tree for every event and the stores beside it.
    """
    analysis = construct_golden_analysis(golden_extractor("sensitivity"))
    member = analysis.sensitivity.members["0"]
    present = _summary_paths(member)
    assert present and all(p.parent.name == "processed" for p in present)
    assert Path(member.analysis_paths.analysis_dir, TREE) not in present

    _clear_every_member(analysis)
    cleared = _summary_paths(member)
    assert cleared == [Path(member.analysis_paths.analysis_dir, TREE)]


# ---- the cross-arm pairing ---------------------------------------------------------------------------------------


def _arm(root: Path, events: list[str], *, drop: str | None = None, bump: float = 0.0) -> None:
    """An arm root with one member whose summaries are cleared: logs naming the clearing, and the consolidated tree.

    ``drop`` leaves that event out of the tree's `summary` group; ``bump`` is added to every stored value.
    """
    member = root / "members" / "member_0"
    record = {"state": "done", "stems": {TRITON_STEM: {TRITON_GROUP: ["max_flood_depth"]}}}
    for event in events:
        sim = member / "sims" / event
        sim.mkdir(parents=True)
        (sim / "log_tritonswmm.json").write_text(json.dumps({"consolidated_summaries_cleared": record}))
    kept = [e for e in events if e != drop]
    values = np.arange(len(kept) * 4, dtype="float64").reshape(len(kept), 2, 2) + bump
    summary = xr.Dataset(
        {"max_flood_depth": (("event_iloc", "y", "x"), values)},
        coords={"event_iloc": list(range(len(kept))), "event_id": ("event_iloc", kept)},
    )
    xr.DataTree.from_dict({"/": xr.Dataset(attrs={"triton_producing_sha": "abc"}), TRITON_GROUP: summary}).to_zarr(
        member / TREE, mode="w", consolidated=True
    )


EVENTS = ["event_index.0", "event_index.1", "event_index.2"]


def _compare(tmp_path: Path):
    return compare_arms(
        arm_root=tmp_path / "arm",
        arm_layout="members",
        reference_root=tmp_path / "reference",
        reference_layout="members",
    )


def test_the_pairing_compares_every_shared_event_of_two_cleared_roots(tmp_path):
    """Property: on two roots whose per-scenario summaries are all cleared the pairing compares the `summary` slice of
    every shared event, and the compared count equals the event count.
    Class: NEW CAPABILITY. Plausible wrong implementation: the landed per-scenario glob, which finds no shared artifact
        and reports the member not evaluated with nothing compared (red on the count).
    Must-fire control: the count is the event count, three, and not zero, and a value changed in one root's tree makes
        the verdict disagree, so the agreement is a comparison and not an absence.
    Why required: with the clearing switch on both arms of a cross-pin comparison are cleared.
    Killed by: a pairing that reads per-scenario stores only.
    Passes under a second correct implementation: one that reads the whole group once and compares event by event.
    """
    _arm(tmp_path / "arm", EVENTS)
    _arm(tmp_path / "reference", EVENTS)
    result = _compare(tmp_path)
    assert result.verdict == "AGREE", result.reason
    assert result.n_compared == len(EVENTS)

    shutil.rmtree(tmp_path / "arm")
    _arm(tmp_path / "arm", EVENTS, bump=1.0)
    assert _compare(tmp_path).verdict == "DISAGREE"


def test_an_event_missing_from_one_roots_branch_is_reported_not_evaluated(tmp_path):
    """Property: with one event's slice removed from one root's `summary` group that event is reported not evaluated
    and the others are still compared.
    Class: NEW CAPABILITY. Plausible wrong implementation: a pairing that drops the event silently, so the verdict
        agrees over fewer events than the roots hold.
    Must-fire control: the compared count is the other events, two, not zero, and the row names the event.
    Why required: a cleared slice is the only copy, and its absence must be visible rather than narrowed away.
    Killed by: a pairing that skips an unreadable slice without a row.
    Passes under a second correct implementation: one that refuses the whole member when any slice is missing.
    """
    _arm(tmp_path / "arm", EVENTS, drop=EVENTS[2])
    _arm(tmp_path / "reference", EVENTS)
    result = _compare(tmp_path)
    assert result.n_compared == len(EVENTS) - 1
    rows = [d for d in result.details if d.get("verdict") == NOT_EVALUATED and EVENTS[2] in str(d.get("artifact"))]
    assert rows, result.details


def test_a_slice_is_read_from_the_masters_tree_when_the_members_tree_moved(tmp_path):
    """Property: a cleared event of a member whose own tree is gone is read from the `member_{id}` groups of the
    master's experiment tree two directories above.
    Class: NEW CAPABILITY. Plausible wrong implementation: a reader that opens the member's own tree only and so
        raises for every member the master has absorbed.
    Must-fire control: the same call on a member with neither tree raises `FileNotFoundError`.
    Why required: with clearing on the master removes each member's tree after copying it.
    Killed by: a reader without the master fallback.
    Passes under a second correct implementation: one that is handed the master's path by its caller.
    """
    member = tmp_path / "exp" / "members" / "member_0"
    member.mkdir(parents=True)
    summary = xr.Dataset(
        {"max_flood_depth": (("event_iloc", "y", "x"), np.arange(8.0).reshape(2, 2, 2))},
        coords={"event_iloc": [0, 1], "event_id": ("event_iloc", EVENTS[:2])},
    )
    xr.DataTree.from_dict({"/": xr.Dataset(), f"member_0/{TRITON_GROUP}": summary}).to_zarr(
        tmp_path / "exp" / "experiment_datatree.zarr", mode="w", consolidated=True
    )
    piece = _cleared_slice(member, EVENTS[1], {TRITON_GROUP: ["max_flood_depth"]})
    assert piece["max_flood_depth"].values.tolist() == [[4.0, 5.0], [6.0, 7.0]]
    assert "event_iloc" not in piece.dims

    with pytest.raises(FileNotFoundError):
        _cleared_slice(tmp_path / "nowhere" / "members" / "member_0", EVENTS[1], {TRITON_GROUP: ["max_flood_depth"]})
