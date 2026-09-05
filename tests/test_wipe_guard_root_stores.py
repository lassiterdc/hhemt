"""The wipe guard and the delete runners must see EVERY accepted root store name.

V0021 unified both prior root stores into `experiment_datatree.zarr`. A guard or a
deleter whose name list predates that migration fails toward UNDER-ACTION: the guard
sees nothing to protect and the deleter leaves the tree behind, both silently.

Fixture-free and solver-free: `tmp_path` only, no `conftest` fixture request, no
`compile_TRITON_SWMM` call, no zarr write.
"""

from pathlib import Path

import pytest

from hhemt.utils import ROOT_TREE_NAMES
from hhemt.wipe_guard import summarize_wipe_cost


@pytest.mark.parametrize("store", ROOT_TREE_NAMES)
def test_wipe_guard_sees_every_accepted_root_store(tmp_path: Path, store: str) -> None:
    """The guard must fire for the unified name AND for both retired ones.

    The unified name is the VIOLATING arm (the guard is blind to it pre-fix); the two
    retired names are differently-positioned SATISFYING arms that the widening must
    not redden. Anchored on `is_empty`, a property that exists in both the pre-fix and
    post-fix worlds, never on the tuple's contents.
    """
    root = tmp_path / "analysis"
    root.mkdir()
    (root / store).mkdir()
    cost = summarize_wipe_cost(root)
    assert not cost.is_empty, f"wipe guard did not see {store}; a full-tree wipe would report nothing to lose"
    assert store in cost.consolidated_trees


def test_wipe_guard_still_reports_empty_on_a_pristine_tree(tmp_path: Path) -> None:
    """The guard was widened, not disabled: an analysis dir with no store is still empty."""
    root = tmp_path / "analysis"
    root.mkdir()
    assert summarize_wipe_cost(root).is_empty


def test_delete_runners_target_every_accepted_root_store() -> None:
    """Both reprocess/consolidation deleters must cover the same name set as the guard.

    Asserted on SET membership rather than on identity so a runner may legitimately
    delete more than the root stores (the consolidation runner also removes
    `system_datatree.zarr`, plots, and status files).
    """
    from hhemt.delete_consolidation_runner import _ANALYSIS_LEVEL_ARTIFACTS
    from hhemt.delete_reprocess_zarr_runner import _REPROCESS_ZARR_ARTIFACTS

    for name in ROOT_TREE_NAMES:
        assert name in _ANALYSIS_LEVEL_ARTIFACTS, f"consolidation deleter would leave {name} behind"
        assert name in _REPROCESS_ZARR_ARTIFACTS, f"reprocess deleter would leave {name} behind"
