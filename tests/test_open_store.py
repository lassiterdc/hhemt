"""``utils.open_store``: one read path that trusts a store's consolidated index when it carries one.

Approved shape D13 item 3 (chunk 3a.2). Every zarr store the toolkit reads opens through
``open_store``, which passes ``consolidated=True`` when the store's root document carries an index
and ``consolidated=False`` otherwise, decided by reading that one document and never by xarray's
``consolidated=None`` fallback. A format 2 store's root ``.zmetadata`` is its index (the format 2
``slurm_jobs.zarr``, row C9 of the chunk 3a.1 review).

No test here compiles or runs a solver; each builds a few kilobytes under ``tmp_path``.

Not established here: read cost (measured by the cluster jobs of the approved shape) and behaviour
on an object store.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import hhemt
from hhemt.utils import (
    append_zarr,
    deferred_consolidation,
    open_store,
    write_zarr,
    zarr_store_has_index,
)


def _data(n: int = 4) -> xr.Dataset:
    return xr.Dataset({"a": (("x",), np.arange(float(n)))}, coords={"x": np.arange(n)}, attrs={"k": 1})


def _unindexed_store(path: Path) -> Path:
    _data().to_zarr(path, mode="w", consolidated=False)
    _data().to_zarr(path, group="g", mode="a", consolidated=False)
    return path


def _consolidated_kwarg_seen(monkeypatch) -> list:
    """Record the ``consolidated`` keyword every xarray zarr opener receives, still performing the open."""
    seen: list = []
    for name in ("open_datatree", "open_dataset", "open_zarr"):
        original = getattr(xr, name)

        def spy(*args, __original=original, **kwargs):
            seen.append(kwargs.get("consolidated", "ABSENT"))
            return __original(*args, **kwargs)

        monkeypatch.setattr(xr, name, spy)
    return seen


def test_a_store_with_and_without_an_index_reads_the_same_data(tmp_path):
    """``open_store`` returns the same data for an indexed store and an unindexed copy of it.

    Property: the root dataset values, attributes and the group ``g`` are equal across the two states, and
    the group opens by name. Class: REGRESSION -- the toolkit's readers all opened unindexed before, and the
    switch must not change what they read. Why: a reader that returned different data once an index existed
    would make the consolidated end state a behaviour change. Kills it: opening an indexed store with the
    wrong ``group`` or ``consolidated`` handling. A second correct implementation: always reading unindexed
    passes this test too, which is why the spy tests below pin WHICH flag is passed.
    """
    plain = _unindexed_store(tmp_path / "plain.zarr")
    indexed = tmp_path / "indexed.zarr"
    write_zarr(_data(), indexed, chunks=None, encoding={})
    append_zarr(_data(), indexed, group="g")
    assert not zarr_store_has_index(plain) and zarr_store_has_index(indexed)
    for kind in ("datatree", "zarr"):
        a, b = open_store(plain, kind=kind), open_store(indexed, kind=kind)
        root_a = a.dataset if kind == "datatree" else a
        root_b = b.dataset if kind == "datatree" else b
        np.testing.assert_array_equal(root_a["a"].values, root_b["a"].values)
        assert root_a.attrs == root_b.attrs
    for store in (plain, indexed):
        ds = open_store(store, kind="dataset", group="g", chunks=None)
        np.testing.assert_array_equal(ds["a"].values, np.arange(4.0))


def test_the_flag_follows_the_root_document_not_xarrays_fallback(tmp_path, monkeypatch):
    """``consolidated`` is True exactly when the root document carries an index, for all three openers.

    Property: the keyword xarray receives is the explicit boolean from the root document and is never absent
    or None. Class: NEW CAPABILITY -- plausible wrong implementation: passing ``consolidated=None`` (xarray's
    fallback warns on every open of an older store and ignores a stale index). Must-fire control: the same
    call on the unindexed store passes False, so the spy is not reading a constant. Kills it: removing the
    ``zarr_store_has_index`` decision. A second correct implementation: reading the document inside each
    caller instead of in the helper.
    """
    plain = _unindexed_store(tmp_path / "plain.zarr")
    indexed = tmp_path / "indexed.zarr"
    write_zarr(_data(), indexed, chunks=None, encoding={})
    seen = _consolidated_kwarg_seen(monkeypatch)
    for kind in ("datatree", "dataset", "zarr"):
        open_store(indexed, kind=kind)
    assert seen == [True, True, True], seen
    seen.clear()
    for kind in ("datatree", "dataset", "zarr"):
        open_store(plain, kind=kind)
    assert seen == [False, False, False], seen


def test_a_format_2_root_zmetadata_is_its_index(tmp_path, monkeypatch):
    """A format 2 store (root ``.zgroup`` and ``.zmetadata``, no root ``zarr.json``) is read through its index.

    Property: ``zarr_store_has_index`` is True for it and ``open_store`` passes ``consolidated=True``; with the
    ``.zmetadata`` removed both flip. Class: NEW CAPABILITY -- plausible wrong implementation: looking only at
    the format 3 root document, which would read ``slurm_jobs.zarr`` unindexed for ever (row C9, form 1).
    Must-fire control: the same store minus ``.zmetadata``. Kills it: dropping the ``.zmetadata`` branch.
    A second correct implementation: asking zarr itself whether consolidated metadata exists.
    """
    store = tmp_path / "v2.zarr"
    write_zarr(_data(), store, chunks=None, encoding={}, zarr_format=2)
    assert not (store / "zarr.json").exists() and (store / ".zmetadata").is_file()
    seen = _consolidated_kwarg_seen(monkeypatch)
    ds = open_store(store, kind="dataset", chunks="auto")
    np.testing.assert_array_equal(ds["a"].values, np.arange(4.0))
    assert seen == [True]
    (store / ".zmetadata").unlink()
    assert not zarr_store_has_index(store)
    seen.clear()
    open_store(store, kind="dataset", chunks="auto")
    assert seen == [False]


def test_the_slurm_jobs_store_is_read_through_its_index(tmp_path, monkeypatch):
    """``slurm_store.read_rows`` opens the format 2 ``slurm_jobs.zarr`` with ``consolidated=True`` after a write.

    Property: after ``write_store`` the reader passes True. Class: NEW CAPABILITY -- plausible wrong
    implementation: leaving the reader on ``consolidated=False``, so the index the write path now builds is
    never used. Kills it: reverting the reader's routing. A second correct implementation: opening
    through ``xr.open_zarr``.
    """
    from hhemt import slurm_store

    row = {
        "MainJobID": "1001",
        "StepKind": "job",
        "JobID": "1001",
        "Submit": "2026-08-16T10:00:00",
        "JobName": "triton.exe",
        "Elapsed": "00:01:00",
        "State": "COMPLETED",
    }
    path = slurm_store.write_store(tmp_path, [row])
    assert path is not None and (path / ".zmetadata").is_file()
    seen = _consolidated_kwarg_seen(monkeypatch)
    slurm_store.read_rows(tmp_path)
    assert seen and all(flag is True for flag in seen), seen


def test_a_group_opened_inside_a_deferred_block_is_read_unindexed(tmp_path, monkeypatch):
    """Inside ``deferred_consolidation`` the store has no index, so ``open_store`` reads it unindexed.

    Property: during the block a read passes False and lists a group appended in the block; after the block
    the same read passes True. Class: NEW CAPABILITY -- plausible wrong implementation: caching the flag
    from before the block, which would serve the pre-block node list. Why: the system build and V0004 read
    their own store while appending to it. Kills it: caching the decision. A second correct implementation:
    reading the document on every call, as this one does.
    """
    store = tmp_path / "s.zarr"
    write_zarr(_data(), store, chunks=None, encoding={})
    seen = _consolidated_kwarg_seen(monkeypatch)
    with deferred_consolidation(store):
        append_zarr(_data(), store, group="new", consolidate=False)
        tree = open_store(store)
        assert "/new" in tree.groups
        assert seen[-1] is False
    open_store(store)
    assert seen[-1] is True


def test_a_path_that_is_not_a_store_directory_has_no_index(tmp_path):
    """``zarr_store_has_index`` is False for a missing path, a file and an unreadable root document.

    Property: the safe answer (read unindexed) for anything that is not a directory with a parseable root.
    Class: NEW CAPABILITY -- plausible wrong implementation: raising on a path the caller then reads with
    its own error handling. Kills it: removing the ``try``. A second correct implementation: returning
    False for every exception.
    """
    assert not zarr_store_has_index(tmp_path / "nope.zarr")
    f = tmp_path / "file.zarr"
    f.write_text("x")
    assert not zarr_store_has_index(f)
    bad = tmp_path / "bad.zarr"
    bad.mkdir()
    (bad / "zarr.json").write_text("{not json")
    assert not zarr_store_has_index(bad)


def test_open_store_rejects_an_unknown_kind(tmp_path):
    """An unknown ``kind`` raises ``ValueError`` naming the accepted ones.

    Property: no silent fallback to a default opener. Class: NEW CAPABILITY. Kills it: a default branch.
    A second correct implementation: a ``Literal`` type with a runtime check.
    """
    with pytest.raises(ValueError, match="datatree"):
        open_store(tmp_path, kind="netcdf")


# ---- the reader census, as a fitness function over the source tree ---------------------------------


def _calls_with_consolidated_false(source: str) -> list[int]:
    return [
        node.lineno
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and any(
            k.arg == "consolidated" and isinstance(k.value, ast.Constant) and k.value.value is False
            for k in node.keywords
        )
    ]


def test_no_reader_in_src_opens_a_store_with_consolidated_false_directly():
    """No call under ``src/hhemt`` passes ``consolidated=False`` as a keyword except ``open_store`` itself.

    Property: every zarr read goes through ``open_store``, so a store's index is used wherever it exists.
    Class: REGRESSION against the pre-change tree, which had 38 such call lines (measured over
    ``git show HEAD:`` of every tracked module before the routing). Must-fire control: the matcher reports a
    synthetic call that passes the keyword. Why: a reader that bypasses the helper reads unindexed for ever,
    or worse reads a stale index by its own rule. Kills it: a new ``xr.open_*`` call with the keyword. A second
    correct implementation: a grep for the string, which would also flag the prose; the AST form is exact.
    """
    assert _calls_with_consolidated_false("import xarray as xr\nxr.open_zarr('p', consolidated=False)\n") == [2]
    root = Path(hhemt.__file__).resolve().parent
    offenders = []
    for path in sorted(root.rglob("*.py")):
        for line in _calls_with_consolidated_false(path.read_text(encoding="utf-8")):
            offenders.append(f"{path.relative_to(root).as_posix()}:{line}")
    assert not offenders, "readers that bypass utils.open_store:\n" + "\n".join(offenders)
