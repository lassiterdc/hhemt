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
