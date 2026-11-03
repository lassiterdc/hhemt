"""Reviewer tests for chunk 3a.3: V0027 rebuilds every stale or missing index, and the index stays true afterwards.

Written by the reviewing seat, not the coder. Each test builds a few kilobytes under ``tmp_path`` or reads
source text; none compiles or runs a solver. Not established here: a tree of thousands of stores, or a
store on an object store.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
import zarr

import hhemt
from hhemt.analysis_validation import check_consolidated_metadata_current
from hhemt.processing_system import TRITONSWMM_system_post_processing
from hhemt.utils import append_zarr, deferred_consolidation, write_datatree_zarr, write_zarr
from hhemt.version_migration import runner, state

SRC_ROOT = Path(hhemt.__file__).resolve().parent


def _tree(root: Path, name: str = "analysis_datatree.zarr") -> Path:
    """A nested DataTree store written by the toolkit helper, so it starts indexed and current."""
    ds = xr.Dataset({"a": (("t",), np.arange(4.0))}, coords={"t": np.arange(4)}, attrs={"k": 1})
    store = root / name
    store.parent.mkdir(parents=True, exist_ok=True)
    write_datatree_zarr(xr.DataTree.from_dict({"/": xr.Dataset(attrs={"r": 1}), "/g/n": ds}), store)
    return store


def _analysis(tmp_path: Path) -> Path:
    """An analysis directory stamped at layout 26 holding one current DataTree store."""
    root = tmp_path / "analysis"
    root.mkdir()
    _tree(root)
    state.write_version_file(root, state.VersionState.fresh(26, "reviewer-probe"))
    return root


def _migrate(root: Path) -> None:
    runner.run_migration(root, target=27, apply=True)


def _index_nodes(store: Path) -> set[str] | None:
    consolidated = json.loads((store / "zarr.json").read_text()).get("consolidated_metadata")
    return None if consolidated is None else set(consolidated["metadata"])


def _disk_nodes(store: Path) -> set[str]:
    return {str(p.parent.relative_to(store)) for p in store.rglob("zarr.json") if p.parent != store}


def _open(store: Path) -> zarr.Group:
    return zarr.open_group(str(store), mode="r+", use_consolidated=False)


def _array_attr(s: Path) -> None:
    _open(s)["g/n/a"].attrs["units"] = "m"


def _group_attr(s: Path) -> None:
    _open(s)["g"].attrs["new"] = 1


def _added_group(s: Path) -> None:
    _open(s).create_group("extra")


def _resized(s: Path) -> None:
    _open(s)["g/n/a"].resize((6,))


def _no_index(s: Path) -> None:
    doc = json.loads((s / "zarr.json").read_text())
    del doc["consolidated_metadata"]
    (s / "zarr.json").write_text(json.dumps(doc))


@pytest.mark.parametrize(
    "damage", [_array_attr, _group_attr, _added_group, _resized, _no_index], ids=lambda f: f.__name__
)
def test_the_migration_leaves_every_damaged_store_with_an_index_equal_to_its_documents(tmp_path, damage):
    """After `run_migration(target=27)` a store damaged behind its index passes the check and lists exactly its nodes.

    Input class: a legacy store whose index is absent or STALE in one of five ways (array attribute, group
    attribute, added group, resized array, no index). Enforces: 5.1 row 3a.3 and D13 item 5 ("runs
    `zarr.consolidate_metadata` on every zarr store ... idempotent: a store whose index already equals its
    listing is skipped"), and my round-4 routed item D5 (a migration can meet a stale index). Why: a migration that
    decides by "does the store carry an index" skips a stale one, and the 3a.2 check then fails the tree it just
    migrated. Control: the damage is asserted real before the run. Kills it: skipping any store whose root
    document has an index. A second correct implementation: rebuilding every store unconditionally passes too.
    """
    root = _analysis(tmp_path)
    store = root / "analysis_datatree.zarr"
    damage(store)
    assert not check_consolidated_metadata_current(store).passed, "control: the damage must be visible first"
    _migrate(root)
    assert check_consolidated_metadata_current(store).passed
    assert _index_nodes(store) == _disk_nodes(store)


def test_a_format_2_store_without_its_zmetadata_ends_carrying_it(tmp_path):
    """A format 2 store whose `.zmetadata` is missing has one after the migration, and its values are unchanged.

    Input class: the format 2 root (`slurm_jobs.zarr`), form 1 of round-1 row C9. Enforces: D13 end state
    "every zarr store the toolkit writes carries consolidated metadata". Why: the walk recognises a format 2
    root by `.zgroup`, a different file from format 3, so a walk written for `zarr.json` alone would skip it.
    Kills it: `_is_store_root` testing `zarr.json` only. A second correct implementation: a walk that tries
    `zarr.open_group` on every `.zarr` directory passes too.
    """
    root = _analysis(tmp_path)
    store = root / "slurm_jobs.zarr"
    write_zarr(xr.Dataset({"a": (("x",), np.arange(3.0))}), store, chunks=None, encoding={}, zarr_format=2)
    (store / ".zmetadata").unlink()
    _migrate(root)
    assert (store / ".zmetadata").is_file()
    assert check_consolidated_metadata_current(store).passed
    assert xr.open_zarr(store, consolidated=True)["a"].values.tolist() == [0.0, 1.0, 2.0]


def _fingerprint(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def test_a_second_run_changes_no_file_and_the_migration_changes_no_data_chunk(tmp_path):
    """A rerun at the same layout rewrites nothing except the stamp and its lock; no chunk file or array ever changes.

    Input class: the tree after a first migration, re-stamped at 26 so the migration is planned again (a run at 27
    plans nothing, which would pass vacuously). Enforces: 5.1 row 3a.3 idempotency and D13's "data values
    unchanged". Why: a rebuild that rewrote every store on each run would touch data files under readers.
    Kills it: a migration that rewrites a data chunk or any file but a store's own documents. It does NOT
    kill removal of the `passed` skip (measured: an unchanged-attribute rewrite reproduces identical bytes, so
    the rerun changes no file either way). A second correct implementation: one that never touches a current store.
    """
    root = _analysis(tmp_path)
    store = root / "analysis_datatree.zarr"
    _array_attr(store)
    chunks_before = {k: v for k, v in _fingerprint(store).items() if not k.endswith("zarr.json")}
    _migrate(root)
    chunks_after = {k: v for k, v in _fingerprint(store).items() if not k.endswith("zarr.json")}
    assert chunks_after == chunks_before, "a data chunk changed"
    state.write_version_file(root, state.VersionState.fresh(26, "reviewer-probe"))
    first = _fingerprint(root)
    _migrate(root)
    second = _fingerprint(root)
    changed = sorted(k for k in set(first) | set(second) if first.get(k) != second.get(k))
    assert [k for k in changed if not k.endswith(("_version.json", ".lock"))] == [], changed


@pytest.mark.parametrize(
    ("relpath", "expect_index"),
    [
        ("sims/e0/processed/TRITONSWMM_perf_summary.zarr", True),
        ("eda/plot_a.zarr", True),
        ("members/member_1/analysis_datatree.zarr", True),
        ("experiment_datatree.zarr", True),
        ("sims/e0/out_triton/stray.zarr", False),
        ("sims/e0/.snakemake/stray.zarr", False),
        ("sims/e0/processed/x.zarr.chapters/chapter_00000.zarr", False),
    ],
)
def test_the_walk_indexes_the_stores_readers_open_and_leaves_the_excluded_ones_alone(tmp_path, relpath, expect_index):
    """Stores where readers open them get an index; stores under `out_*`, `.snakemake` and `*.chapters` do not.

    Input class: store locations by role. The first four are written by `write_zarr` and read through
    `open_store` (per-scenario summary, EDA artifact, member tree, experiment tree); the last three are the
    walk's documented exclusions. Enforces: 5.1 row 3a.3 ("every zarr store under an analysis tree"). Why: a store
    a reader opens but the walk skips would have no index at layout 27 and fail the 3a.2 check; a store inside raw
    output that the walk entered would cost millions of directory reads. Kills it: adding `eda` to the skipped names.
    A second correct implementation: the same walk expressed with `Path.rglob` passes too.
    """
    root = _analysis(tmp_path)
    store = root / relpath
    store.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset({"v": (("x",), np.arange(2.0))}).to_zarr(store, mode="w", consolidated=False)
    _migrate(root)
    assert (_index_nodes(store) is not None) is expect_index


# ---- the index stays true: a route that removes a group behind it, and the system consolidation step -----


def test_pruning_an_analysis_from_the_system_store_leaves_no_index_that_lists_it(tmp_path):
    """After `prune_analysis` the system store's index is absent or equals the groups on disk.

    Input class: the system store built by the toolkit's own block, then one analysis group pruned. Enforces:
    D13A's crash-window rule applied to a route this package edits (`processing_system.py`): "no moment exists
    at which a trusted index lists fewer or more groups than the store holds". Why: `prune_analysis` removes the
    group with `shutil.rmtree` and rebuilds nothing, so `open_store` then serves an index that lists a group
    the store no longer holds (round-1 probe P2 measured that shape of stale index). Kills it: running the
    removal inside `deferred_consolidation(system_zarr)`. A second correct implementation: calling
    `consolidate_zarr_store` after the removal passes too.
    """
    system_zarr = tmp_path / "system_datatree.zarr"
    with deferred_consolidation(system_zarr):
        for analysis_id in ("a1", "a2"):
            append_zarr(
                xr.Dataset(attrs={"analysis_id": analysis_id}), system_zarr, group=analysis_id, consolidate=False
            )
            append_zarr(_leaf(), system_zarr, group=f"{analysis_id}/n", consolidate=False)
    assert _index_nodes(system_zarr) == _disk_nodes(system_zarr), "fixture: the built store must be current"
    post = object.__new__(TRITONSWMM_system_post_processing)
    post._system = SimpleNamespace(sys_paths=SimpleNamespace(system_datatree_zarr=system_zarr))
    post.prune_analysis("a1")
    assert "a1" not in _disk_nodes(system_zarr), "fixture: the group must be gone"
    index = _index_nodes(system_zarr)
    assert index is None or index == _disk_nodes(system_zarr), f"index lists {sorted(index)}"


def _leaf() -> xr.Dataset:
    return xr.Dataset({"v": (("x",), np.arange(2.0))})


def _calls_in_function(path: Path, function: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function:
            return {
                n.func.id if isinstance(n.func, ast.Name) else n.func.attr
                for n in ast.walk(node)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name | ast.Attribute)
            }
    raise AssertionError(f"{function} not found in {path}")


@pytest.mark.parametrize(
    ("module", "function"),
    [
        ("processing_analysis.py", "consolidate_to_datatree"),
        ("sensitivity_analysis.py", "consolidate_sensitivity_datatree"),
        ("processing_system.py", "consolidate_system_datatree"),
    ],
)
def test_each_consolidation_step_ends_by_checking_the_index_it_just_wrote(module, function):
    """Every consolidation step that writes a zarr store calls the end-of-step index check.

    Input class: the three consolidation steps (analysis, sensitivity master, system). Enforces: D13 item 4,
    "It runs at the end of each consolidation step and in `validate_analysis`", and the coder's stipulation
    draft, which repeats that sentence. Measured on the chunk tree: the analysis and sensitivity steps call
    `_assert_store_index_current`; `consolidate_system_datatree` calls nothing of the kind, and
    `check_consolidated_metadata` does not list the system store either. Why: the system store is the one
    written by a block of appends, the one place an interrupted write is plausible. Kills it: dropping the
    call from any step. A second correct implementation: calling `check_consolidated_metadata_current` directly.
    """
    names = _calls_in_function(SRC_ROOT / module, function)
    assert names & {"_assert_store_index_current", "check_consolidated_metadata_current"}, sorted(names)
