"""Reviewer tests for chunk 2b.2: the two extraction call sites and the deposit zip writer.

``extract_reprex_bundle`` and the HydroShare branch of ``TRITON_SWMM_experiment._fetch_deposit_files``
now extract through the threaded ``bundle._archive.extract_zip``, and ``publishing._resolve_upload``
writes a directory upload through ``write_zip`` under the moderate policy. The contract each site keeps
is what the stdlib call it replaced gave: the same files and directories, confinement of a hostile name,
the same exception on a bad archive, and replacement of a file already on disk. These tests hold that
contract at input positions the coder's own tests do not reach, with the stdlib call as the oracle.

Every test builds files under ``tmp_path`` and uses a fake HydroShare resource, so none compiles or runs a
solver, none touches the network, and all run on a laptop. Run from a worktree as
``PYTHONPATH=${PWD}/src pytest tests/test_extract_zip_sites_reviewer.py``.
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

import pytest

from hhemt.bundle._reprex import extract_reprex_bundle
from hhemt.experiments import TRITON_SWMM_experiment
from hhemt.publishing import _resolve_upload


def _tree(root: Path) -> dict[str, bytes | None]:
    """Every file under ``root`` as ``{relative path: bytes}`` and every directory as ``{path/: None}``."""
    out: dict[str, bytes | None] = {}
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root).as_posix()
        out[rel + "/" if p.is_dir() else rel] = None if p.is_dir() else p.read_bytes()
    return out


class _FakeResource:
    """A hsclient resource stand-in whose ``download`` copies a prepared zip into the target."""

    def __init__(self, source: Path):
        self._source = source

    def download(self, target: Path) -> str:
        out = Path(target) / "resource.zip"
        out.write_bytes(self._source.read_bytes())
        return str(out)


class _FakeHS:
    """A hsclient stand-in that hands out the one prepared resource whatever key is asked."""

    def __init__(self, source: Path):
        self._resource = _FakeResource(source)

    def resource(self, _key: str) -> _FakeResource:
        return self._resource


def _hydroshare_fetch(src_zip: Path, work: Path) -> Path:
    work.mkdir(exist_ok=True)
    return TRITON_SWMM_experiment._fetch_deposit_files(
        "hydroshare", res_identifier="abc", dest=work / "case", hs=_FakeHS(src_zip), validate=False
    )


def _extract_with(site: str, src_zip: Path, workdir: Path) -> Path:
    """Run the named call site on ``src_zip`` and return the directory its files landed in."""
    if site == "reprex":
        target = workdir / "reprex" / "bundle.zip"
        target.parent.mkdir()
        target.write_bytes(src_zip.read_bytes())
        return extract_reprex_bundle(target)
    return _hydroshare_fetch(src_zip, workdir / "hs")


# Entry names a bag or bundle zip may carry that a hand-rolled extractor can mishandle. The bag site
# needs one top-level folder, so its set keeps every name under `bag/`.
_REPREX_NAMES = [
    "a/./b.txt",
    "c//d.txt",
    "../up.txt",
    "../outside_dir/f.txt",
    "/abs.txt",
    "x/../../y.txt",
    "ünï.txt",
    "sp ace/x y",
    "e\\f",
    "g.",
    "d/",
]
_BAG_NAMES = [
    "bag/a/./b.txt",
    "bag//c.txt",
    "bag/x/../../y.txt",
    "bag/../../outside_dir/f.txt",
    "bag/ünï.txt",
    "bag/sp ace/x y",
    "bag/e\\f",
    "bag/g.",
    "bag/d/",
    "bag/plain.txt",
]


def _zip_of(path: Path, names: list[str]) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for index, name in enumerate(names):
            zf.writestr(zipfile.ZipInfo(name), b"" if name.endswith("/") else f"entry {index} of {name}\n".encode() * 3)


@pytest.mark.parametrize("site", ["reprex", "hydroshare"])
def test_site_lands_the_tree_extractall_lands_for_unusual_and_hostile_entry_names(tmp_path, site):
    """Property: the call site writes exactly the files and directories ``ZipFile.extractall`` writes.

    Input class: entry names with a dot segment, a doubled slash, a climbing segment, a leading slash,
    non-ASCII letters, a space, a backslash, a trailing dot and a directory entry (the bag site keeps all
    of them under one top-level folder, because it refuses a zip with several). The coder's tests use
    plain names. Approved-shape row: section 5.1 row 2b.2, "`extract_zip` at `bundle/_reprex.py:203` and
    `experiments.py:1245`", and 5.7 2b test (3), "extract through `extract_zip` with the same file set as
    `extractall`".

    Why required: both sites read zips made by other parties, and an extractor that does not confine a
    climbing name writes outside the destination. The oracle is the stdlib: the same zip extracted by
    ``extractall`` into a sibling directory. The change that kills it is an extractor that joins the
    entry name onto the destination without sanitising it. A second correct implementation is a loop
    over ``ZipFile.extract``, which sanitises the same way.

    The tree outside the destination is also compared, so a file written one level up cannot hide.
    """
    names = _REPREX_NAMES if site == "reprex" else _BAG_NAMES
    src = tmp_path / "src.zip"
    _zip_of(src, names)
    reference = tmp_path / "reference"
    with zipfile.ZipFile(src) as zf:
        zf.extractall(reference)
    work = tmp_path / "work"
    work.mkdir()
    dest = _extract_with(site, src, work)
    if site == "reprex":
        assert _tree(dest) == _tree(reference)
        assert sorted(p.name for p in (work / "reprex").iterdir()) == ["bundle", "bundle.zip"]
    else:
        # the fetch extracts into the destination's parent and renames the bag root to the destination
        assert _tree(dest) == _tree(reference / "bag")
        assert sorted(p.name for p in dest.parent.iterdir()) == ["case"]
        assert sorted(p.name for p in work.iterdir()) == ["hs"]


def _stored_zip(path: Path, count: int = 12) -> list[str]:
    """A zip of ``count`` stored entries; returns the names in order."""
    names = [f"d{i % 3}/f{i}.bin" for i in range(count)]
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as zf:
        for name in names:
            zf.writestr(name, name.encode() * 40)
    return names


def _flip_payload_byte(path: Path, name: str) -> None:
    """Flip one byte inside the stored payload of entry ``name``, so its CRC no longer matches."""
    raw = bytearray(path.read_bytes())
    with zipfile.ZipFile(path) as zf:
        info = zf.getinfo(name)
    start = info.header_offset + 30 + int.from_bytes(raw[info.header_offset + 26 : info.header_offset + 28], "little")
    start += int.from_bytes(raw[info.header_offset + 28 : info.header_offset + 30], "little")
    raw[start + 5] ^= 0xFF
    path.write_bytes(bytes(raw))


@pytest.mark.parametrize("site", ["reprex", "hydroshare"])
@pytest.mark.parametrize("damage", ["bad_crc_in_a_middle_member", "not_a_zip_at_all", "truncated_zip"])
def test_site_raises_the_exception_the_stdlib_raises_on_a_damaged_archive(tmp_path, site, damage):
    """Property: a damaged archive raises the same exception type, with the same message, as ``ZipFile`` does.

    Input class: a zip whose payload fails its CRC (twelve entries, so the threaded path runs), a file
    that is not a zip, and a zip cut off before its end record. Approved-shape row: section 5.1 row 2b.2
    ("keeps the behaviour it had, including what is raised on a bad archive") and the writer contract that
    ``extract_zip`` "raises what ``extractall`` raises". The coder's tests extract only valid zips.

    Why required: callers catch ``zipfile.BadZipFile`` to report a corrupt download; a threaded extractor
    that wraps the error (an exception group, a ``RuntimeError``) turns that report into a crash. The
    oracle is the stdlib on the same bytes. The change that kills it is a pool that collects errors into
    another type. A second correct implementation is a serial extractor, which raises the stdlib's own.
    """
    src = tmp_path / "src.zip"
    if damage == "bad_crc_in_a_middle_member":
        names = _stored_zip(src)
        _flip_payload_byte(src, names[6])
    elif damage == "not_a_zip_at_all":
        src.write_bytes(b"this is not a zip file" * 20)
    else:
        _stored_zip(src)
        src.write_bytes(src.read_bytes()[:-30])

    def _oracle() -> BaseException:
        try:
            with zipfile.ZipFile(src) as zf:
                zf.extractall(tmp_path / "reference")
        except BaseException as exc:  # noqa: BLE001 - the oracle records whatever the stdlib raises
            return exc
        raise AssertionError("fixture is not damaged: extractall succeeded")

    expected = _oracle()
    work = tmp_path / "work"
    work.mkdir()
    with pytest.raises(BaseException) as got:  # noqa: PT011 - the type is asserted below against the oracle
        _extract_with(site, src, work)
    assert type(got.value) is type(expected)
    assert str(got.value) == str(expected)


def test_hydroshare_fetch_overwrites_a_same_size_stale_file_already_in_the_target(tmp_path):
    """Property: a file already on disk at an entry's path is replaced by the entry's bytes, whatever its size.

    Input class: a stale file with the SAME size as the new entry but different bytes, and a stale file
    that is longer. ``extractall`` rewrites both; an extractor told to skip entries whose size already
    matches keeps the first. Approved-shape row: section 5.1 row 2b.2 ("keeps the behaviour it had") and
    ``extract_zip``'s own docstring, which says the skip option "trusts size alone, so a same-size edit is
    kept"; that option exists for ``report-from-bundle``'s reuse and is wrong here.

    Reachability: a fetch into a directory that holds an earlier, partly edited copy of the same bag is
    what an operator does after a failed validation; the call site passes ``dest.parent``, which is that
    directory. Why required: a stale same-size file would validate against nothing and be read as the
    deposit's. The change that kills it is passing ``only_missing_or_resized=True``. A second correct
    implementation is deleting the target files before extracting.
    """
    new_same = b"NEW  bytes!"
    src = tmp_path / "src.zip"
    with zipfile.ZipFile(src, "w") as zf:
        zf.writestr("mybag/data/same.txt", new_same)
        zf.writestr("mybag/data/longer.txt", b"short")
    work = tmp_path / "work"
    stale = work / "mybag" / "data"
    stale.mkdir(parents=True)
    (stale / "same.txt").write_bytes(b"OLD  bytes!")
    (stale / "longer.txt").write_bytes(b"a much longer stale file than the entry")
    assert len(new_same) == len(b"OLD  bytes!")
    dest = _hydroshare_fetch(src, work)
    assert (dest / "data" / "same.txt").read_bytes() == new_same
    assert (dest / "data" / "longer.txt").read_bytes() == b"short"


def _upload_names(archive: Path) -> list[str]:
    with zipfile.ZipFile(archive) as zf:
        return [info.filename for info in zf.infolist()]


def test_deposit_zip_has_the_old_writers_entry_set_order_and_bytes_for_awkward_names(tmp_path):
    """Property: the deposit zip names every FILE under the upload relative to its parent, in path-parts order.

    Input class: a store with sibling names differing in case and in a character below ``/``, an empty
    directory (which the old writer skipped), zero-byte files, and a non-ASCII name. The coder's store
    has four plain entries. Approved-shape row: section 5.1 row 2b.2, "`publishing._resolve_upload` writes
    through `write_zip` under the policy", with the entry set and order the old loop gave
    (``sorted(path.rglob("*"))``, files only, ``relative_to(path.parent)``).

    Why required: a deposit whose entry list differs from the store's file set is a different dataset. The
    oracle is the old loop's own expression recomputed here, not the writer. The change that kills it is a
    sort that folds case, or a writer that adds directory entries or drops zero-byte files. A second
    correct implementation is ``zf.write`` in that same order with its own stamps; names, order and bytes
    are the same.
    """
    store = tmp_path / "up" / "s.zarr"
    files = {
        "zarr.json": b"{}",
        "c/0/0": b"",
        "c/0/1": bytes(range(256)) * 5,
        "Plots/a.txt": b"A",
        "plots/a.txt": b"a",
        "plots-x/a.txt": b"x",
        "plots.x/a.txt": b"y",
        "é.txt": b"e",
        "empty.txt": b"",
    }
    for rel, payload in files.items():
        (store / rel).parent.mkdir(parents=True, exist_ok=True)
        (store / rel).write_bytes(payload)
    (store / "emptydir").mkdir()
    expected = [p.relative_to(store.parent).as_posix() for p in sorted(store.rglob("*")) if p.is_file()]
    folded = sorted(expected, key=lambda n: [part.casefold() for part in Path(n).parts])
    assert expected != folded, "fixture cannot tell a case-folded sort from the old order"
    staging = tmp_path / "staging"
    staging.mkdir()
    archive, key = _resolve_upload(store, staging)
    assert key == "s.zarr.zip"
    assert _upload_names(archive) == expected
    with zipfile.ZipFile(archive) as zf:
        assert zf.testzip() is None
        for name in expected:
            assert zf.read(name) == (store.parent / name).read_bytes(), name


def test_deposit_zip_classifies_stores_nested_below_the_upload_root(tmp_path):
    """Property: a chunk file is STORED wherever its ``.zarr`` directory sits below the upload root.

    Input class: an upload directory that is not itself a store and holds stores at depth one and two, a
    zarr-format-2 store (chunk files ``0.0``, documents ``.zarray`` and ``.zgroup``), a directory named
    ``c`` that is not under any ``.zarr``, and a plain file. The coder's store is the upload root itself,
    so a classifier that looks only at the upload's own name passes it. Approved-shape row: section 2B,
    "A zarr chunk file is any entry inside a directory named `*.zarr/` that is not a zarr metadata
    document", and 5.7 2b test (4), "stores zarr chunk entries and deflates the rest".

    Why required: the deposit holds whatever the caller passes, and a store nested in it is deflated for
    nothing if the classifier misses it. The oracle is the shape's definition written out here, over the
    entry name alone. The change that kills it is a classifier that inspects only the first path segment.
    A second correct implementation is ``zf.write`` with a per-entry method from the same definition.
    """
    root = tmp_path / "up"
    layout = {
        "a/s.zarr/zarr.json": zipfile.ZIP_DEFLATED,
        "a/s.zarr/c/0/0": zipfile.ZIP_STORED,
        "a/b/t.zarr/.zarray": zipfile.ZIP_DEFLATED,
        "a/b/t.zarr/.zgroup": zipfile.ZIP_DEFLATED,
        "a/b/t.zarr/0.0": zipfile.ZIP_STORED,
        "a/b/t.zarr/1.0": zipfile.ZIP_STORED,
        "a/plain/c/0": zipfile.ZIP_DEFLATED,
        "a/notes.txt": zipfile.ZIP_DEFLATED,
    }
    for rel in layout:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes((rel.encode() + b"\n") * 30)
    staging = tmp_path / "staging"
    staging.mkdir()
    archive, key = _resolve_upload(root, staging)
    assert key == "up.zip"
    with zipfile.ZipFile(archive) as zf:
        actual = {info.filename: info.compress_type for info in zf.infolist()}
    expected = {f"up/{rel}": method for rel, method in layout.items()}
    assert actual == expected


def test_deposit_zip_bytes_do_not_depend_on_the_sources_modification_times(tmp_path):
    """Property: two deposit zips of the same store are byte-identical even when every source mtime differs.

    Input class: the same store zipped twice, with every file's mtime moved between the two calls. The
    old writer stamped each entry with its file's mtime, so its two archives differ. Approved-shape row:
    section 5.1 row 2b.2 ("writes through `write_zip`") and section 2B's restated contract, "same entries,
    same order, same stamps, same content bytes always; same archive bytes under the same zlib", of which
    the fixed stamp is the part this input reaches. The coder's return states that the stamps changed and
    that it pinned nothing.

    Why required: a deposit re-run after a store is copied or touched must not appear as new content to a
    byte comparison of the deposit. This is NEW CAPABILITY, red on the pre-edit tree. The change that kills
    it is a writer that takes each entry's stamp from the file. A second correct implementation is
    ``zf.write`` with ``ZipInfo`` dates fixed by the caller.
    """
    store = tmp_path / "up" / "s.zarr"
    for rel in ("zarr.json", "c/0/0", "notes.txt"):
        (store / rel).parent.mkdir(parents=True, exist_ok=True)
        (store / rel).write_bytes((rel.encode() + b"\n") * 40)
    first_stage = tmp_path / "s1"
    second_stage = tmp_path / "s2"
    first_stage.mkdir()
    second_stage.mkdir()
    first, _ = _resolve_upload(store, first_stage)
    for p in store.rglob("*"):
        os.utime(p, (1_000_000_000, 1_000_000_000))
    second, _ = _resolve_upload(store, second_stage)
    assert first.read_bytes() == second.read_bytes()
