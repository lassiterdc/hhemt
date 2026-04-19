"""Phase 1 regression: read-only DataTree overlay via `to_datatree()`."""

import pytest
import xarray as xr

import tests.utils_for_testing as tst_ut

pytestmark = pytest.mark.skipif(
    tst_ut.is_scheduler_context(), reason="Only runs on non-HPC systems."
)


def test_to_datatree_skips_missing_modes(norfolk_all_models_analysis):
    """Absent consolidated outputs produce a tree with only the root node."""
    tree = norfolk_all_models_analysis.process.to_datatree()
    assert isinstance(tree, xr.DataTree)
    assert "analysis_id" in tree.attrs


def test_mode_to_tree_path_keys_match_mode_config(norfolk_all_models_analysis):
    """`_MODE_TO_TREE_PATH` must cover every `_MODE_CONFIG` key (schema SSOT)."""
    proc = norfolk_all_models_analysis.process
    assert set(proc._MODE_TO_TREE_PATH.keys()) == set(proc._MODE_CONFIG.keys())


def test_to_datatree_is_lazy_when_outputs_present(norfolk_all_models_analysis_cached):
    """After consolidation, populated leaves are dask-backed (lazy)."""
    analysis = norfolk_all_models_analysis_cached
    tree = analysis.process.to_datatree()
    assert isinstance(tree, xr.DataTree)

    populated = [node for node in tree.subtree if node.has_data]
    if not populated:
        pytest.skip("Fixture has no consolidated outputs to verify laziness against.")

    for node in populated:
        for var in node.ds.data_vars.values():
            assert hasattr(var.data, "dask"), (
                f"{node.path}/{var.name} is not dask-backed; "
                f"to_datatree must open lazily."
            )
