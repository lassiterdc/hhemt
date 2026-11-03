"""Reviewer tests for chunk 2b.3: ``experiment_tree_summary`` against a full ``xr.open_datatree``.

The helper answers three questions (root attributes, ``/member_*`` groups, model tier) from store
metadata where the renderers used to call ``xr.open_datatree``. The contract is that it answers what
a full open answers, so every test here uses the full open, read in the test with the old renderer
logic, as its oracle, at store shapes the coder's file does not build: a tier directory that is an
array, an implicit group with no ``zarr.json``, a member directory that is not a group, names that
only look like members, a consolidated root, and non-ASCII and nested attributes.

Every test writes a small zarr store under ``tmp_path``; none compiles or runs a solver, so all run on
a laptop. Run as ``PYTHONPATH=${PWD}/src pytest tests/test_experiment_tree_summary_reviewer.py``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from hhemt.utils import experiment_tree_summary


def _ds(name: str = "v") -> xr.Dataset:
    return xr.Dataset({name: (("y",), np.arange(3, dtype="f4"))}, coords={"y": np.arange(3)})


def _write(store: Path, nodes: dict[str, xr.Dataset], attrs: dict | None = None, consolidated: bool = False) -> Path:
    nodes = dict(nodes)
    nodes["/"] = xr.Dataset(attrs=attrs if attrs is not None else {"k": "v"})
    xr.DataTree.from_dict(nodes).to_zarr(store, mode="w", consolidated=consolidated, zarr_format=3)
    return store


def _oracle(store: Path):
    """The three answers from a FULL open, with the logic the renderers used before the helper."""
    dt = xr.open_datatree(str(store), engine="zarr", consolidated=False)
    groups = set(dt.groups)
    members = tuple(sorted(g for g in groups if g.count("/") == 1 and g.startswith("/member_")))
    tier = None
    for g in members:
        if f"{g}/tritonswmm/triton" in groups:
            tier = "tritonswmm"
            break
        if f"{g}/triton_only/triton" in groups:
            tier = "triton_only"
            break
    return dict(dt.attrs), members, tier


def _same_as_full_open(store: Path) -> None:
    attrs, members, tier = _oracle(store)
    got = experiment_tree_summary(store)
    assert dict(got.attrs) == attrs
    assert tuple(got.members) == members
    assert got.tier == tier


def test_tier_leaf_that_is_an_array_not_a_group_is_not_a_tier(tmp_path):
    """Property: a tier path counts only when it is a GROUP, as it does in ``dt.groups``.

    Input class: ``member_a/tritonswmm/triton`` written as an ARRAY inside a group, with a later member
    carrying a real ``triton_only/triton`` group. A file-existence check counts the array's ``zarr.json``
    as a coupled tier and answers ``tritonswmm``; the full open answers ``triton_only``. Approved-shape
    row: section 5.1 row 2b.3, "inspecting every member" and the same answers as the full open.
    Why required: the model label printed in the combined report would flip to TRITON-SWMM. The change
    that kills it is a tier test by ``zarr.json`` existence. Second correct implementation: reading the
    leaf's ``node_type`` from its ``zarr.json``.
    """
    store = _write(
        tmp_path / "t.zarr",
        {
            "/member_a/tritonswmm": xr.Dataset({"triton": (("y",), np.arange(3.0))}),
            "/member_b/triton_only/triton": _ds(),
        },
    )
    assert _oracle(store)[2] == "triton_only"
    _same_as_full_open(store)


def test_implicit_tier_parent_without_its_own_document_is_not_a_tier(tmp_path):
    """Property: ``tritonswmm/triton`` counts only when its parent group is itself a stored group.

    Input class: a store whose ``tritonswmm`` directory has lost its ``zarr.json`` (an interrupted or
    partially copied tree) while ``triton/zarr.json`` remains. The full open does not see the subtree; a
    leaf-file check does. Approved-shape row: section 5.1 row 2b.3, same answers as the full open.
    Why required: a partial copy is the state a failed bundle transfer leaves, and the label must not
    claim a tier the open cannot read. Killed by checking only the leaf file. Second correct
    implementation: walking each level's ``zarr.json``.
    """
    store = _write(tmp_path / "t.zarr", {"/member_a/tritonswmm/triton": _ds(), "/member_b/triton_only/triton": _ds()})
    (store / "member_a" / "tritonswmm" / "zarr.json").unlink()
    assert _oracle(store)[2] == "triton_only"
    _same_as_full_open(store)


def test_member_directory_without_a_document_is_not_a_member(tmp_path):
    """Property: a ``member_*`` directory is listed only when it is a stored group, and a file is never one.

    Input class: a ``member_x`` directory with no ``zarr.json`` and a plain FILE named ``member_f``, next
    to real members and names that only resemble members (``members_y``, ``Member_z``, ``member``).
    Approved-shape row: section 5.1 row 2b.3, ``members`` as ``sorted(dt.groups)`` filtered gave. Why
    required: ``n_subs`` in the provenance table is ``len(members)``. Killed by listing directories by
    name alone. Second correct implementation: filtering ``zarr.json`` ``node_type == "group"``.
    """
    store = _write(
        tmp_path / "t.zarr",
        {
            "/member_a/tritonswmm/triton": _ds(),
            "/member_b/triton_only/triton": _ds(),
            "/members_y/q": _ds(),
            "/other/q": _ds(),
        },
    )
    (store / "member_x").mkdir()
    (store / "member_f").write_text("not a group")
    shutil.copytree(store / "other", store / "Member_z")
    shutil.copytree(store / "other", store / "member")
    _same_as_full_open(store)


@pytest.mark.parametrize("consolidated", [False, True])
def test_root_attributes_match_a_full_open_for_nested_non_ascii_and_consolidated_roots(tmp_path, consolidated):
    """Property: the attrs returned equal ``dt.attrs`` for nested, list-valued and non-ASCII attributes.

    Input class: a root whose attributes hold a nested mapping, a list, an integer, a float, a boolean
    and a non-ASCII string, written with and without consolidated metadata (which adds a second copy of
    the attributes to the root document). Approved-shape row: section 5.1 row 2b.3, same answers as the
    full open. Why required: the provenance row reads the producing shas from these attrs. Killed by a
    reader that decodes the document with a narrower encoding or reads the consolidated copy. Second
    correct implementation: ``zarr.open_group(store).attrs``.
    """
    attrs = {"triton_producing_sha": "éabc", "nested": {"a": [1, 2, {"b": None}]}, "n": 3, "f": 0.5, "t": True}
    store = _write(tmp_path / "t.zarr", {"/member_a/triton_only/triton": _ds()}, attrs=attrs, consolidated=consolidated)
    _same_as_full_open(store)


def test_member_order_is_code_point_order_not_directory_listing_order(tmp_path):
    """Property: members are returned sorted by code point (``member_10`` before ``member_2``), as the oracle sorts.

    Input class: member names whose natural and lexical orders differ, created in an order that is
    neither. Approved-shape row: section 5.1 row 2b.3 (``sorted``). Why required: the tier is taken
    from the FIRST sorted member that has one, so order decides the label. Killed by an unsorted or
    naturally sorted listing. Second correct implementation: sorting the names read from zarr's own
    member listing.
    """
    nodes = {f"/member_{i}/triton_only/triton": _ds() for i in (2, 10, 1, 20)}
    nodes["/member_10/tritonswmm/triton"] = _ds()
    store = _write(tmp_path / "t.zarr", nodes)
    assert _oracle(store)[1] == ("/member_1", "/member_10", "/member_2", "/member_20")
    _same_as_full_open(store)


def test_zarr_format_2_root_and_missing_store_follow_the_full_open(tmp_path):
    """Property: a zarr-format-2 root gives the full open's answers, and an absent path raises what a full open raises.

    Input class: a format-2 store with a tritonswmm member, and a path that does not exist. Approved-shape
    row: section 5.1 row 2b.3, "falling back to `xr.open_datatree` on a zarr-format-2 root". Why required:
    the slurm store is format 2 and the fallback is the one branch that keeps old trees readable. Killed by
    a fallback that returns an empty summary. Second correct implementation: reading ``.zgroup`` and
    ``.zattrs`` directly.
    """
    nodes = {"/member_a/tritonswmm/triton": _ds()}
    dt = xr.DataTree.from_dict({**nodes, "/": xr.Dataset(attrs={"k": "v2"})})
    store = tmp_path / "v2.zarr"
    dt.to_zarr(store, mode="w", consolidated=False, zarr_format=2)
    assert not (store / "zarr.json").exists()
    _same_as_full_open(store)
    try:
        xr.open_datatree(str(tmp_path / "absent.zarr"), engine="zarr", consolidated=False)
    except BaseException as exc:  # noqa: BLE001 - the oracle records what the full open raises
        expected = type(exc)
    else:
        raise AssertionError("fixture: a full open of an absent path did not raise")
    with pytest.raises(expected):
        experiment_tree_summary(tmp_path / "absent.zarr")
