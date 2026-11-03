"""Reviewer tests for chunk 3a.2: readers go through ``open_store``; the index check judges every stale class.

Written by the reviewing seat, not the coder. Each test builds a few kilobytes under ``tmp_path`` or
reads source text; none compiles or runs a solver.

Not established here: reads on a store larger than a few chunks, or on an object store.
"""

from __future__ import annotations

import ast
import re
import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import zarr

import hhemt
from hhemt.analysis_validation import (
    CONSOLIDATED_INDEX_REQUIRED_FROM_LAYOUT,
    check_consolidated_metadata_current,
)
from hhemt.utils import write_datatree_zarr, write_zarr
from hhemt.version_migration.constants import LAYOUT_VERSION

SRC_ROOT = Path(hhemt.__file__).resolve().parent
MIGRATIONS = SRC_ROOT / "version_migration" / "versions"


def _literal_false_consolidated(source: str) -> list[str]:
    """Spellings that hand a zarr opener a LITERAL ``consolidated`` False.

    Three: the keyword ``consolidated=False``; a subscript assignment ``kw["consolidated"] = False``;
    a dict literal ``{"consolidated": False}``. A variable (``consolidated=flag``) is not a literal.
    """
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.keyword) and node.arg == "consolidated":
            if isinstance(node.value, ast.Constant) and node.value.value is False:
                found.append("keyword consolidated=False")
        elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and node.value.value is False:
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.slice, ast.Constant)
                    and target.slice.value == "consolidated"
                ):
                    found.append('subscript ["consolidated"] = False')
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                if (
                    isinstance(key, ast.Constant)
                    and key.value == "consolidated"
                    and isinstance(value, ast.Constant)
                    and value.value is False
                ):
                    found.append('dict {"consolidated": False}')
    return found


def test_no_zarr_reader_in_src_hands_an_opener_a_literal_consolidated_false():
    """No module under ``src/hhemt`` states ``consolidated`` as a literal False; the census fires on each spelling.

    Input class: the spellings of one reader flag. Approved shape D13 item 3 routes every reader through
    ``utils.open_store``, and the coder's census counted the keyword form ``consolidated=False`` (41 lines to 2
    prose lines). Measured on the chunk tree, three readers state the flag through a dict item assignment
    instead and were not counted: ``processing_analysis.py`` lines 609 and 717 and ``process_simulation.py``
    line 267 (``open_kwargs["consolidated"] = False``), so those per-scenario reads ignore an index their own
    writers now produce. Enforces: 5.1 row 3a.2 and D13 item 3 ("one read path"). Why: the property "every
    reader uses the index" holds only while no reader spells the opt-out in a form the census cannot see. Both arms
    are in this test: the census returns a finding on a synthetic source for each spelling (control) and, on the
    real tree, must return none. Kills it: reintroducing ``open_kwargs["consolidated"] = False`` anywhere. A second
    correct implementation: a reader that computes the flag from ``zarr_store_has_index(path)`` instead of calling
    ``open_store`` passes too.
    """
    controls = {
        "xr.open_zarr(p, consolidated=False)\n": "keyword consolidated=False",
        'kw = {}\nkw["consolidated"] = False\n': 'subscript ["consolidated"] = False',
        'kw = {"consolidated": False, "chunks": "auto"}\n': 'dict {"consolidated": False}',
    }
    for source, spelling in controls.items():
        assert spelling in _literal_false_consolidated(source), f"control: census did not fire on {spelling!r}"
    assert _literal_false_consolidated("xr.open_zarr(p, consolidated=flag)\n") == [], "control: a variable fired"

    offenders: dict[str, list[str]] = {}
    for path in sorted(SRC_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        found = _literal_false_consolidated(path.read_text(encoding="utf-8"))
        if found:
            offenders[str(path.relative_to(SRC_ROOT))] = found
    assert offenders == {}, f"readers that opt out of the index in a form the census cannot see: {offenders}"


# ---- the index check, over every way an index can be wrong ------------------------------------------


def _tree_store(root: Path, name: str) -> Path:
    """A nested DataTree store with attributes, a NaN fill and a string variable, written by the toolkit helper."""
    ds = xr.Dataset(
        {"a": (("t", "y"), np.full((3, 4), np.nan, dtype="float32")), "s": (("t",), np.array(["x", "y", "z"]))},
        coords={"t": np.arange(3), "y": np.arange(4)},
        attrs={"k": 1.5, "n": "v"},
    )
    tree = xr.DataTree.from_dict({"/": xr.Dataset(attrs={"r": "r"}), "/sa_1/tri": ds, "/sa_1/swmm": ds.copy()})
    store = root / name
    write_datatree_zarr(tree, store)
    return store


def _stale_group_attr(p: Path) -> None:
    zarr.open_group(str(p), mode="r+", use_consolidated=False)["sa_1"].attrs["new"] = 1


def _stale_array_attr(p: Path) -> None:
    zarr.open_group(str(p), mode="r+", use_consolidated=False)["sa_1/swmm/a"].attrs["units"] = "m"


def _stale_shape(p: Path) -> None:
    zarr.open_group(str(p), mode="r+", use_consolidated=False)["sa_1/swmm/a"].resize((5, 4))


def _stale_added_group(p: Path) -> None:
    zarr.open_group(str(p), mode="r+", use_consolidated=False).create_group("extra")


def _stale_removed_group(p: Path) -> None:
    shutil.rmtree(p / "sa_1" / "swmm")


@pytest.mark.parametrize(
    "damage",
    [_stale_group_attr, _stale_array_attr, _stale_shape, _stale_added_group, _stale_removed_group],
    ids=lambda f: f.__name__.removeprefix("_stale_"),
)
def test_the_index_check_fails_on_each_way_a_store_changes_behind_its_index_and_passes_a_fresh_store(tmp_path, damage):
    """A store changed behind its index by any of five routes fails the check; the same store untouched passes.

    Input class: the five routes D13 item 4 names (a node added, a node removed, a group attribute updated in
    place, an array attribute updated in place, an array's shape changed), each applied through raw ``zarr``
    to a store the toolkit just wrote. Enforces: 5.1 row 3a.2, D13 item 4 ("including each node's attributes and
    each array's shape ... a node-list comparison alone is not enough"). Why: ``open_store`` trusts the index, so
    an undetected stale route is a silent wrong read. The must-fire control is the first assertion: the untouched
    store passes, so a check that fails everything cannot satisfy this test. Kills it: comparing node lists only.
    A second correct implementation: comparing a hash of each node's document passes too.
    """
    store = _tree_store(tmp_path, "t.zarr")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert check_consolidated_metadata_current(store).passed, "control: a fresh store must pass"
        damage(store)
        verdict = check_consolidated_metadata_current(store)
    assert not verdict.passed, f"{damage.__name__} was not reported"
    assert verdict.details and verdict.details[0]["node"], "a failure must name the first differing node"


def test_a_format_2_store_is_judged_by_its_zmetadata(tmp_path):
    """A format 2 store passes fresh, fails after an attribute changes behind ``.zmetadata``, and fails without it.

    Input class: the format 2 root (``slurm_jobs.zarr``), which the coordinator scoped in to the index check as
    form 1 (round 1 row C9). Enforces: the approved end state, "every zarr store the toolkit writes carries
    consolidated metadata", for the one store the toolkit writes at format 2. Why: its index is a different file
    (``.zmetadata``) with different keys, so a check written for ``zarr.json`` would pass or fail it for the wrong
    reason. Control: the fresh store passes. Kills it: a check that treats a missing root ``zarr.json`` as no index.
    A second correct implementation: delegating the comparison to ``zarr.open_consolidated`` passes too.
    """
    store = tmp_path / "v2.zarr"
    write_zarr(
        xr.Dataset({"a": (("x",), np.arange(3.0))}, attrs={"q": 1}), store, chunks=None, encoding={}, zarr_format=2
    )
    assert check_consolidated_metadata_current(store).passed, "control: a fresh format 2 store must pass"
    zarr.open_group(str(store), mode="r+", use_consolidated=False)["a"].attrs["z"] = 1
    assert not check_consolidated_metadata_current(store).passed
    (store / ".zmetadata").unlink()
    assert not check_consolidated_metadata_current(store).passed
    assert check_consolidated_metadata_current(store, index_required=False).applicable is False


# ---- the layout threshold is a literal that must follow the migration chunk 3a.3 lands -----------------


def test_the_layout_that_requires_an_index_is_the_layout_of_the_consolidation_migration():
    """The required-from layout equals the consolidation migration's version once it exists, else the next layout.

    Input class: the two states of the tree, before chunk 3a.3 (no ``V*__consolidate_zarr_metadata.py``) and after
    it. Enforces: D13 item 4, "an absent index on a store that should carry one (any store at or after the
    migration's layout version) is also an ERROR". Why: ``CONSOLIDATED_INDEX_REQUIRED_FROM_LAYOUT`` is a literal
    (27) with no link to ``LAYOUT_VERSION`` (26); if the migration lands at another number the absence check
    never fires, or fires on a legacy store, with no error. The review item 5 verdict: the rule is right and the
    unlinked literal is its weakness. Kills it: setting the constant to 28. A second correct implementation: the
    constant derived from ``LAYOUT_VERSION`` at import after 3a.3 passes too.
    """
    migrations = sorted(MIGRATIONS.glob("V*__consolidate_zarr_metadata.py"))
    if not migrations:
        assert CONSOLIDATED_INDEX_REQUIRED_FROM_LAYOUT == LAYOUT_VERSION + 1
        return
    assert len(migrations) == 1, f"expected one consolidation migration, found {[m.name for m in migrations]}"
    version = int(re.match(r"V(\d+)__", migrations[0].name).group(1))
    assert CONSOLIDATED_INDEX_REQUIRED_FROM_LAYOUT == version
    assert LAYOUT_VERSION >= version
