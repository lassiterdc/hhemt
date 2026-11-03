"""Tests for the two zip-extraction call sites that now go through ``bundle._archive.extract_zip``.

``extract_reprex_bundle`` (``bundle/_reprex.py``) and the HydroShare branch of
``TRITON_SWMM_experiment._fetch_deposit_files`` (``experiments.py``) used to call
``ZipFile.extractall``, which creates every entry one after another. They now call the threaded
``extract_zip``. What must not change is the file set and bytes that land on disk, the empty
directories, the removal of a stale destination, and the refusal of a bag with several top-level
folders.

Every test builds a small zip under ``tmp_path`` and uses a fake HydroShare resource object, so
none compiles or runs a solver, none touches the network, and all run on a laptop. Run from a
worktree as ``PYTHONPATH=${PWD}/src pytest tests/test_extract_zip_call_sites.py``.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from hhemt.bundle._reprex import extract_reprex_bundle
from hhemt.exceptions import ProcessingError
from hhemt.experiments import TRITON_SWMM_experiment


def _refuse_extractall(monkeypatch) -> None:
    """Make ``ZipFile.extractall`` raise; ``ZipFile.extract`` (what ``extract_zip`` uses) stays live."""

    def _raiser(*_args, **_kwargs):
        raise AssertionError("ZipFile.extractall reached: this call site must extract through extract_zip")

    monkeypatch.setattr(zipfile.ZipFile, "extractall", _raiser)


def _make_zip(path: Path, files: dict[str, bytes], empty_dirs: tuple[str, ...] = ()) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for name, payload in files.items():
            zf.writestr(name, payload)
        for name in empty_dirs:
            zf.writestr(name if name.endswith("/") else name + "/", b"")


def _tree(root: Path) -> dict[str, bytes | None]:
    """Every file under ``root`` as ``{relative path: bytes}`` and every directory as ``{path/: None}``."""
    out: dict[str, bytes | None] = {}
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root).as_posix()
        out[rel + "/" if p.is_dir() else rel] = None if p.is_dir() else p.read_bytes()
    return out


_BUNDLE_FILES = {
    "bundle_manifest.json": b'{"k": 1}\n',
    "plots/a.html": b"<html>a</html>" * 40,
    "eda/s.zarr/zarr.json": b"{}",
    "eda/s.zarr/c/0/0": bytes(range(256)),
}


def test_reprex_extract_keeps_file_set_bytes_and_empty_dirs_without_extractall(tmp_path, monkeypatch):
    """Property: ``extract_reprex_bundle`` yields the same tree as ``extractall`` and never calls it.

    Class: REGRESSION-shaped discriminator. Before this change the call reaches ``extractall`` and
    the refusing stub reds; the tree comparison holds before and after. The input class is any bundle
    zip, here one with nested directories, a zarr chunk and an EMPTY directory entry (a files-only
    extractor would drop it). The plausible wrong implementation is leaving ``extractall`` in place,
    or an extractor that skips directory entries. Why required: the call sits on the bundle consume
    path where entry creation is the cost.

    A second correct implementation is a hand-written loop over ``ZipFile.extract``, which never
    names ``extractall`` and yields the same tree.
    """
    zip_path = tmp_path / "bundle.zip"
    _make_zip(zip_path, _BUNDLE_FILES, empty_dirs=("logs/empty",))
    reference = tmp_path / "reference"
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(reference)  # taken BEFORE the stub is installed
    _refuse_extractall(monkeypatch)
    dest = extract_reprex_bundle(zip_path)
    assert dest == tmp_path / "bundle"
    assert _tree(dest) == _tree(reference)
    assert "logs/empty/" in _tree(dest)


def test_reprex_extract_replaces_a_stale_destination(tmp_path):
    """Property, differently positioned: a destination that already exists is replaced, not merged into.

    Class: REGRESSION (holds before and after). The satisfying input differs from the test above in
    that the destination holds a stale file the zip does not carry; the plausible wrong implementation
    is an extractor change that drops the ``rmtree`` and leaves the stale file. Why required: the
    reprex round-trip must see exactly the zip's contents. A second correct implementation is
    extracting into a fresh temporary directory and renaming it over the destination.
    """
    zip_path = tmp_path / "bundle.zip"
    _make_zip(zip_path, _BUNDLE_FILES)
    stale = tmp_path / "bundle" / "stale.txt"
    stale.parent.mkdir()
    stale.write_text("old")
    dest = extract_reprex_bundle(zip_path)
    assert not (dest / "stale.txt").exists()
    assert (dest / "bundle_manifest.json").read_bytes() == _BUNDLE_FILES["bundle_manifest.json"]


class _FakeResource:
    """A stand-in for an hsclient resource whose ``download`` writes a prepared zip into the target."""

    def __init__(self, zip_bytes_source: Path):
        self._source = zip_bytes_source

    def download(self, target: Path) -> str:
        out = Path(target) / "resource.zip"
        out.write_bytes(self._source.read_bytes())
        return str(out)


class _FakeHS:
    def __init__(self, zip_source: Path):
        self._resource = _FakeResource(zip_source)

    def resource(self, _key: str) -> _FakeResource:
        return self._resource


_BAG_FILES = {
    "mybag/bagit.txt": b"BagIt-Version: 0.97\n",
    "mybag/data/contents/a.csv": b"1,2,3\n" * 50,
    "mybag/data/contents/sub/b.bin": bytes(range(200)),
}


def test_hydroshare_fetch_extracts_the_bag_without_extractall(tmp_path, monkeypatch):
    """Property: the HydroShare fetch lands the bag's files under ``dest`` and never calls ``extractall``.

    Class: REGRESSION-shaped discriminator: before this change the fetch reaches ``extractall`` and the
    stub reds; the landed tree and the removal of the downloaded zip hold before and after. The input
    class is a one-top-level-folder bag zip with an empty directory entry; the plausible wrong
    implementation is leaving ``extractall`` or extracting into the wrong parent. Why required: a
    HydroShare bag holds many small files and creating them serially is the cost on a shared file system.

    A second correct implementation is a hand-written ``ZipFile.extract`` loop, which passes unchanged.
    """
    src = tmp_path / "src.zip"
    _make_zip(src, _BAG_FILES, empty_dirs=("mybag/data/contents/emptydir",))
    _refuse_extractall(monkeypatch)
    work = tmp_path / "work"
    work.mkdir()
    dest = work / "case"
    got = TRITON_SWMM_experiment._fetch_deposit_files(
        "hydroshare", res_identifier="abc", dest=dest, hs=_FakeHS(src), validate=False
    )
    assert got == dest
    tree = _tree(dest)
    assert tree["data/contents/a.csv"] == _BAG_FILES["mybag/data/contents/a.csv"]
    assert tree["data/contents/sub/b.bin"] == _BAG_FILES["mybag/data/contents/sub/b.bin"]
    assert "data/contents/emptydir/" in tree
    assert not (work / "resource.zip").exists(), "the downloaded zip must be removed"


def test_hydroshare_fetch_still_refuses_a_bag_with_several_top_level_folders(tmp_path):
    """Property, differently positioned: a zip with two top-level folders is refused with ``ProcessingError``.

    Class: REGRESSION (holds before and after). The satisfying input differs from the test above only in
    its top level; the plausible wrong implementation is an extractor change that picks a root silently.
    Why required: the bag root cannot be determined from such a zip. A second correct implementation is
    checking the names before extracting, which refuses the same input.
    """
    src = tmp_path / "src.zip"
    _make_zip(src, {"one/x.txt": b"x", "two/y.txt": b"y"})
    work = tmp_path / "work"
    work.mkdir()
    with pytest.raises(ProcessingError):
        TRITON_SWMM_experiment._fetch_deposit_files(
            "hydroshare", res_identifier="abc", dest=work / "case", hs=_FakeHS(src), validate=False
        )
