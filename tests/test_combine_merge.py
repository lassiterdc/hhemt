"""Phase-2 merge tests — DataTree-of-experiments shape + identity-on-coordinate."""

from pathlib import Path

import pytest
import xarray as xr
from xarray import DataTree

from hhemt.bundle import _combine_merge as M


def _write_stub_bundle(tmp: Path, name: str) -> Path:
    root = tmp / name
    root.mkdir(parents=True)
    ds = xr.Dataset({"max_wlevel_m": ("event_iloc", [1.0, 2.0])})
    xr.DataTree(dataset=ds).to_zarr(root / M.CONSOLIDATED_TREE_NAME)
    return root


def _write_hierarchical_stub_bundle(tmp: Path, name: str) -> Path:
    """A stub bundle whose consolidated tree has data-bearing child groups
    (``/triton``, ``/swmm``) like a real analysis_datatree.zarr."""
    root = tmp / name
    root.mkdir(parents=True)
    tree = DataTree.from_dict(
        {
            "/": xr.Dataset(attrs={"title": "analysis"}),
            "triton": xr.Dataset({"max_wlevel_m": ("event_iloc", [1.0, 2.0])}),
            "swmm": xr.Dataset({"max_flow_cms": ("event_iloc", [3.0, 4.0])}),
        }
    )
    tree.to_zarr(root / M.CONSOLIDATED_TREE_NAME)
    return root


def test_merge_shape_and_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "_experiment_id", lambda r: r.name)
    a = _write_stub_bundle(tmp_path, "exp_a")
    b = _write_stub_bundle(tmp_path, "exp_b")
    merged = M.merge_experiment_trees([a, b])
    assert set(merged.children) == {"experiment_exp_a", "experiment_exp_b"}
    # identity rides a coordinate, not attrs:
    assert merged["experiment_exp_a"].coords["experiment"].item() == "exp_a"


def test_merge_missing_tree_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "_experiment_id", lambda r: r.name)
    a = _write_stub_bundle(tmp_path, "exp_a")
    b = tmp_path / "exp_b"
    b.mkdir()  # no consolidated tree
    with pytest.raises(FileNotFoundError):
        M.merge_experiment_trees([a, b])


def test_identity_rides_every_child_group(tmp_path, monkeypatch):
    """The experiment coordinate must ride each data-bearing group node (not the
    experiment-root only) so it survives a later per-group concat (Phase 4)."""
    monkeypatch.setattr(M, "_experiment_id", lambda r: r.name)
    a = _write_hierarchical_stub_bundle(tmp_path, "exp_a")
    merged = M.merge_experiment_trees([a])
    child = merged["experiment_exp_a"]
    assert child.coords["experiment"].item() == "exp_a"
    for group in ("triton", "swmm"):
        assert child[group].coords["experiment"].item() == "exp_a"
    # laziness preserved — the scalar-coord stamp must not load data vars:
    assert hasattr(child["triton"]["max_wlevel_m"].data, "dask")
