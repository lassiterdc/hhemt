"""Root-tree resolution for a combine input bundle."""

from pathlib import Path

import pytest
import xarray as xr

from hhemt.bundle import _combine_merge as M


def _write_stub_bundle(tmp: Path, name: str) -> Path:
    root = tmp / name
    root.mkdir(parents=True)
    ds = xr.Dataset({"max_flood_depth": ("event_iloc", [1.0, 2.0])})
    xr.DataTree(dataset=ds).to_zarr(root / M.CONSOLIDATED_TREE_NAME)
    return root


def test_merge_missing_tree_raises(tmp_path):
    a = _write_stub_bundle(tmp_path, "exp_a")
    assert M._resolve_root_tree(a).name == M.CONSOLIDATED_TREE_NAME
    b = tmp_path / "exp_b"
    b.mkdir()  # no consolidated tree
    with pytest.raises(FileNotFoundError):
        M._resolve_root_tree(b)
