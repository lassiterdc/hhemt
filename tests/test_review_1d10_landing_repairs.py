"""Chunk 1d.10 reviewer tests: where the system tree's pointer for a sensitivity master resolves the master's store.

No solver and no Snakemake. The sensitivity cases run on stand-ins that carry only what `_pointer_target` reads; the
golden case runs on the committed sensitivity golden. Each test names its class, the plausible wrong implementation it
reddens, and a must-fire control.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import xarray as xr

from hhemt.processing_system import TRITONSWMM_system_post_processing
from hhemt.utils import EXPERIMENT_TREE_NAME, REGULAR_TREE_NAME, ROOT_TREE_NAMES

DATE = "2026-11-08"


def _pointer_target(analysis):
    """`_pointer_target` of a post-processing object over no system, which the method never reads."""
    return TRITONSWMM_system_post_processing(SimpleNamespace())._pointer_target(analysis)


def _own_tree_is_unreachable():
    raise AssertionError("a sensitivity master was read through its own `open_datatree`")


def _write_store(store: Path) -> None:
    xr.Dataset(attrs={"output_creation_date": DATE}).to_zarr(store, mode="w", consolidated=False)


def _master(root: Path) -> SimpleNamespace:
    """A master stand-in. The write field names the unified store, as `analysis.py` binds it, so a result that follows
    the field rather than the root's contents is distinguishable from one that follows the contents."""
    return SimpleNamespace(
        cfg_analysis=SimpleNamespace(toggle_sensitivity_analysis=True),
        analysis_paths=SimpleNamespace(
            analysis_dir=root,
            sensitivity_datatree_zarr=root / EXPERIMENT_TREE_NAME,
            analysis_datatree_zarr=None,
        ),
        process=SimpleNamespace(open_datatree=_own_tree_is_unreachable),
    )


@pytest.mark.parametrize("store_name", ROOT_TREE_NAMES)
def test_a_master_root_carrying_any_root_store_name_is_bound_to_that_store(tmp_path, store_name):
    """Property: a sensitivity master whose root carries exactly one root store, under any name in
    `ROOT_TREE_NAMES`, gets that store and its `output_creation_date` as the pointer target.
    Class: REGRESSION on the two retired names (red on the tree before chunk 1d.10, whose target was the write field and
        returned None for a root holding only a retired name); the unified-name row is green before and after.
    Reachability: a master analysis directory that predates the layout migration, or a bundle's root, carries a retired
        name -- the constant's own note says both exist ("a render bundle is a tarball no migration reaches").
    Both-states anchor: the assertion compares the returned store path and date, which both versions return on the
        unified row.
    Must-fire control: the unified-name row passes on every correct implementation, so a red on a retired row is the
        resolver's absence and not a broken stand-in.
    Why required: a pointer that returns None skips the master, so the system tree carries no group for an experiment
        whose store exists under a retired name.
    Killed by: replacing the resolver call with the unified filename joined onto the root.
    Passes under a second correct implementation: one that walks `ROOT_TREE_NAMES` itself and returns the first
        existing store.
    """
    root = tmp_path / "exp"
    root.mkdir()
    _write_store(root / store_name)
    assert _pointer_target(_master(root)) == (root / store_name, DATE)


def test_a_master_root_that_also_carries_the_regular_arms_store_is_bound_to_the_unified_store(tmp_path):
    """Property: a master root carrying both the unified store and the regular arm's store resolves to the unified
    one, because on the sensitivity arm the other is residue no writer produces.
    Class: held green. Before and after chunk 1d.10 both return the unified store; the named mutation reds it.
    Reachability: the stranded regular store is the state `V0025__retire_stranded_regular_store` exists to retire, so a
        master root can still carry it until that migration has run.
    Both-states anchor: the assertion turns on which of two existing stores is returned, a property both versions have.
    Must-fire control: a root carrying the regular store alone resolves to it (the row above), so the regular store is
        readable and the unified one is the only thing that differs here.
    Why required: the resolver returns the regular store for a root whose arm is not supplied, so a call that drops the
        arm points the master's pointer at the stale flat tree.
    Killed by: calling the resolver with `arm=False`.
    Passes under a second correct implementation: one that supplies the arm as the string `"sensitivity"`.
    """
    root = tmp_path / "exp"
    root.mkdir()
    _write_store(root / EXPERIMENT_TREE_NAME)
    _write_store(root / REGULAR_TREE_NAME)
    assert _pointer_target(_master(root)) == (root / EXPERIMENT_TREE_NAME, DATE)


def test_a_master_root_with_no_store_has_no_pointer_target(tmp_path):
    """Property: a master whose root carries no root store has no pointer target, so the consolidation skips it.
    Class: held green. Both versions return None; the named mutation reds it.
    Reachability: an experiment whose consolidation has not run has no store.
    Both-states anchor: the assertion is the returned value, None.
    Must-fire control: the same root with a store written resolves to it (the first test), so None is not the stand-in's
        default.
    Why required: a returned path to a store that does not exist is opened by the consolidation and fails there.
    Killed by: returning the resolver's unified-name fallback when nothing exists.
    Passes under a second correct implementation: one that checks `ROOT_TREE_NAMES` for existence before resolving.
    """
    root = tmp_path / "exp"
    root.mkdir()
    assert _pointer_target(_master(root)) is None


def _golden_with_member_trees_kept(golden_extractor):
    """The sensitivity golden with the clearing switch off, so each member keeps its own tree and `_pointer_target`
    has a tree to open for it."""
    from tests.fixtures.golden_trees import construct_golden_analysis

    system_dir = golden_extractor("sensitivity")
    cfg = system_dir / "analysis_config.yaml"
    cfg.write_text(cfg.read_text() + "toggle_clear_scenario_timeseries_after_consolidation: false\n")
    return construct_golden_analysis(system_dir)


def test_the_pointer_predicate_follows_the_write_fields_binding_over_every_analysis_of_a_real_sensitivity_golden(
    golden_extractor,
):
    """Property: over the master and the four members of the real sensitivity golden, the analyses `_pointer_target`
    treats as a master are exactly those whose `analysis_paths.sensitivity_datatree_zarr` is bound; the master's target
    is its experiment store and each member's is its own `analysis_datatree.zarr`.
    The clearing switch is off here (the golden names neither toggle, and the default is on, which moves each member's
        tree into the master and leaves a member with no tree of its own to point to); the next test covers that state.
    Class: held green. The previous implementation keyed on the field itself, so both versions agree on this input; the
        named mutations red it.
    Reachability: `analysis.py` binds the field exactly when `toggle_sensitivity_analysis` is set and
        `_create_members` clears the toggle on every member, so the golden holds one analysis of each kind.
    Both-states anchor: the assertion compares the predicate with the binding and the returned paths, none of which the
        change introduces.
    Must-fire control: the master's store is written before the call, so a None for the master is a defect of the
        predicate and not an absent store.
    Why required: the pointer predicate replaced a read of the binding with a read of the toggle, and a member that
        carried the toggle, or a master that did not, would be bound to the wrong tree.
    Killed by: a predicate that is always false or always true, or one keyed on a name other than the toggle.
    Passes under a second correct implementation: one that reads `is_experiment_member` instead of the toggle, since
        the golden's members carry it and its master does not.
    Not established: this row alone does not separate a predicate that is always true, because a member's own root
        also resolves to its own store; the `_pointer_target` tests of `test_system_tree_pointer.py` carry that mutant.
    """
    master = _golden_with_member_trees_kept(golden_extractor)
    sensitivity = master.sensitivity
    sensitivity.consolidate_sensitivity_datatree()
    post = TRITONSWMM_system_post_processing(SimpleNamespace(_analysis=master))

    analyses = post._analyses()
    members = [a for a in analyses if a is not master]
    assert len(members) == 4, "the golden's four members"
    for analysis in analyses:
        bound = analysis.analysis_paths.sensitivity_datatree_zarr is not None
        assert bool(analysis.cfg_analysis.toggle_sensitivity_analysis) is bound

    master_target = post._pointer_target(master)
    assert master_target is not None, "control: the master's store was written"
    assert master_target[0] == master.analysis_paths.sensitivity_datatree_zarr
    for member in members:
        target = post._pointer_target(member)
        assert target is not None
        assert target[0] == member.analysis_paths.analysis_datatree_zarr


def test_the_system_tree_skips_a_member_whose_tree_moved_into_the_master_and_never_asks_for_its_pointer(
    golden_sensitivity_analysis, monkeypatch
):
    """Property: with the clearing switch at its default (on), `consolidate_system_datatree` over the sensitivity
    golden never calls `_pointer_target` for a member whose tree moved into the master, writes no group for it, and
    writes the master's pointer.

    Class: NEW CAPABILITY of the default. Reachability: the real `consolidate_sensitivity_datatree` (moves the four
    member trees) then the real `consolidate_system_datatree`. Plausible wrong implementation: the member check
    placed after the pointer lookup, which opens a tree that is gone (`FileNotFoundError`). Must-fire control: the
    master IS asked for its pointer and the member trees are absent on disk, so the skip is what separates them.
    """
    master = golden_sensitivity_analysis
    master.sensitivity.consolidate_sensitivity_datatree()
    master._system._analysis = master
    post = TRITONSWMM_system_post_processing(master._system)
    members = [a for a in post._analyses() if a is not master]
    assert len(members) == 4
    assert all(not Path(m.analysis_paths.analysis_datatree_zarr).exists() for m in members), "control: trees moved"
    asked = []
    real = post._pointer_target

    def spy(analysis):
        asked.append(analysis)
        return real(analysis)

    monkeypatch.setattr(post, "_pointer_target", spy)
    system_zarr = post.consolidate_system_datatree()
    assert master in asked, "control: the master was asked for its pointer"
    assert not [a for a in asked if a in members], "a moved member must never be asked for its pointer"
    groups = sorted(p.name for p in Path(system_zarr).iterdir() if p.is_dir() and not p.name.startswith("."))
    assert str(master.cfg_analysis.analysis_id) in groups
    assert not [m for m in members if str(m.cfg_analysis.analysis_id) in groups]
