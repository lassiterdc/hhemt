"""Root consolidated-store resolution (S8b, shared resolver in ``hhemt.utils``).

`V0021__experiment_tree_unification` unified BOTH prior root stores into one
`experiment_datatree.zarr`. These tests pin resolution by EXISTENCE across the
unified and both retired names, the preference ORDER, and the absent case.

Fixture-free and solver-free: `tmp_path` only, no `conftest` fixture request, no
`compile_TRITON_SWMM` call, no zarr write.
"""

from pathlib import Path

import pytest

from hhemt.utils import EXPERIMENT_TREE_NAME, ROOT_TREE_NAMES, resolve_experiment_tree


def _root(tmp_path: Path, *names: str) -> Path:
    root = tmp_path / "analysis"
    root.mkdir()
    for n in names:
        (root / n).mkdir()
    return root


def test_unified_name_is_first_in_the_preference_order():
    """Order is the contract, not an accident: the unified name must win a tie."""
    assert ROOT_TREE_NAMES[0] == EXPERIMENT_TREE_NAME == "experiment_datatree.zarr"
    assert set(ROOT_TREE_NAMES) == {
        "experiment_datatree.zarr",
        "analysis_datatree.zarr",
        "sensitivity_datatree.zarr",
    }


@pytest.mark.parametrize("name", ROOT_TREE_NAMES)
def test_resolves_each_accepted_name_when_it_is_the_only_one(tmp_path, name):
    """Every accepted name resolves on its own.

    The two retired names are the differently-positioned SATISFYING arms: correct
    states that are not the one the fix was written against, which the widening
    must not redden.
    """
    assert resolve_experiment_tree(_root(tmp_path, name)).name == name


def test_prefers_the_unified_name_over_a_retired_one(tmp_path):
    """Mid-migration state: deterministic, and it does not depend on dir order."""
    root = _root(tmp_path, "sensitivity_datatree.zarr", "experiment_datatree.zarr")
    assert resolve_experiment_tree(root).name == "experiment_datatree.zarr"


def test_absent_case_returns_the_canonical_name_and_does_not_exist(tmp_path):
    """The resolver was widened, not disabled.

    Anchored on BEHAVIOUR that exists in both the pre-fix and post-fix worlds --
    whether the returned path exists -- so every caller's own absent-tree branch
    still fires, and it reports against the canonical name rather than a retired one.
    """
    resolved = resolve_experiment_tree(_root(tmp_path))
    assert not resolved.exists()
    assert resolved.name == EXPERIMENT_TREE_NAME


def test_accepts_a_str_root(tmp_path):
    """Callers pass both `Path` and `str` roots; the signature admits either."""
    root = _root(tmp_path, "experiment_datatree.zarr")
    assert resolve_experiment_tree(str(root)).name == "experiment_datatree.zarr"


def test_the_two_store_state_warns_and_the_warning_is_the_pin(tmp_path):
    """THE NON-VACUITY ARM for the two-store branch, whose mutation score is otherwise ZERO.

    Measured: the branch is distinguishable from its own deletion on exactly ONE of the
    eight subsets of `ROOT_TREE_NAMES`, and every other node in this module builds one of
    the seven on which the two are identical -- including the one two-store arm above,
    which is the WRONG PAIR. Delete the branch and this node reds; nothing else here does.

    WHAT THIS DELIBERATELY DOES NOT ASSERT: which store comes back. The two production
    routes to this subset invert which one is current while leaving the directory names
    identical, so an assertion on the return value would pin a position rather than the
    invariant. The WARNING is route-invariant -- it is a function of the subset alone --
    and it is what the branch adds.

    It also survives the repair. A widened guard keyed on MEMBERSHIP rather than on exact
    list equality still warns here, so this arm does not redden the fix for the disarm
    recorded in the next node.
    """
    root = _root(tmp_path, "experiment_datatree.zarr", "analysis_datatree.zarr")
    with pytest.warns(RuntimeWarning):
        resolved = resolve_experiment_tree(root)
    assert resolved.name in {"experiment_datatree.zarr", "analysis_datatree.zarr"}


@pytest.mark.xfail(
    strict=True,
    reason=(
        "KNOWN DEFECT, recorded as an arm rather than as prose. The guard is keyed on "
        "exact list equality, so a THIRD root name DISARMS it: the resolver returns the "
        "migrated store with no warning at all, which is strictly worse than the two-store "
        "case the guard was written for and is reached by ADDING a store rather than "
        "removing one. strict=True is load-bearing: when the guard is widened to a "
        "membership test this node XPASSes and reds the suite, which is the signal to "
        "flip the marker rather than a regression."
    ),
)
def test_a_third_root_name_must_not_silently_disarm_the_guard(tmp_path):
    """THE DISARM. Fails today by design; green is the repair landing."""
    root = _root(
        tmp_path,
        "experiment_datatree.zarr",
        "sensitivity_datatree.zarr",
        "analysis_datatree.zarr",
    )
    with pytest.warns(RuntimeWarning):
        resolve_experiment_tree(root)
