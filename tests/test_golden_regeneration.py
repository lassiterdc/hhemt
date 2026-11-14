"""The committed golden zips must equal a fresh generation; Zarr and NetCDF members are compared by CONTENT.

``tests/fixtures/goldens/multisim.zip`` and ``sensitivity.zip`` are goldens: each holds a finished analysis tree
and the files hhemt needs to construct an analysis object from it (``tests/goldens/SPEC.md`` section 1). They are
produced by running the generator (``python -m tests.goldens.generator``), which is independent of ``hhemt``
because that independence is the premise of ``tests/test_golden_conformance.py``: had the generator imported hhemt,
hhemt would be asked to accept trees hhemt itself wrote (``tests/test_golden_generator_independence.py``). Its output
is zipped with :func:`build_zip`. This test re-runs the generator into a temporary directory and compares it to the zip:

* the file set (including directories that hold no file) must be equal;
* inside a Zarr store (a directory named ``*.zarr``) the comparison is by array values, attributes and
  coordinates (``Dataset.identical`` / ``DataTree.identical``) plus equal variable dtypes, so Zarr's
  metadata ordering cannot fail it;
* a NetCDF file is compared the same way, because HDF5 stamps its own metadata into the bytes;
* every other file is compared byte for byte.

It prints each zip's size in bytes and file count (visible with ``pytest -s``) and asserts both positive.
No size ceiling is written: the size is measured, not budgeted. The test does not import ``hhemt``.

Regenerate the zips with ``python -m tests.test_golden_regeneration`` from the repository root.
"""

from __future__ import annotations

import shutil
import sys
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from tests.goldens.generator.__main__ import generate_multisim, generate_sensitivity

GOLDEN_DIR = Path(__file__).resolve().parent / "fixtures" / "goldens"
# Fixed so that the stored zip does not vary with the time it was built; extraction stamps its own time, so this value
# never reaches an extracted file.
ZIP_TIME = (2000, 1, 1, 0, 0, 0)
GENERATORS = {"multisim": generate_multisim, "sensitivity": generate_sensitivity}


def build_zip(root: Path, dest: Path) -> None:
    """Zip ``root`` with sorted members, fixed time, fixed permissions and no symlinked files.

    A symlink to a file is dropped; a symlink to a directory is written as an empty directory entry, because
    ``is_dir()`` is true for it. The committed goldens hold no symlinks (``tests/goldens/SPEC.md`` section 9).

    Directories that hold no file get an explicit directory entry, because an empty directory (a
    sensitivity member's ``_status/``) is part of the layout and a plain file listing would lose it."""
    root = Path(root)
    files = sorted(p for p in root.rglob("*") if p.is_file() and not p.is_symlink())
    with_files = {parent for p in files for parent in p.parents}
    empty_dirs = sorted(d for d in root.rglob("*") if d.is_dir() and d not in with_files)
    members: list[tuple[str, Path | None]] = [(p.relative_to(root).as_posix(), p) for p in files]
    members += [(d.relative_to(root).as_posix() + "/", None) for d in empty_dirs]
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for name, path in sorted(members):
            info = zipfile.ZipInfo(name, date_time=ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            if path is None:
                info.external_attr = (0o40755 << 16) | 0x10
                zf.writestr(info, b"")
            else:
                info.external_attr = 0o100644 << 16
                zf.writestr(info, path.read_bytes())


def _listing(root: Path) -> set[str]:
    """Every file and every empty directory under ``root``, as posix paths (directories end in ``/``).

    A symlinked file counts as a file here, unlike in :func:`build_zip`; a golden carries no symlinks
    (``tests/goldens/SPEC.md`` section 9), so the two do not differ on a real golden."""
    files = {p for p in root.rglob("*") if p.is_file()}
    with_files = {parent for p in files for parent in p.parents}
    out = {p.relative_to(root).as_posix() for p in files}
    out |= {d.relative_to(root).as_posix() + "/" for d in root.rglob("*") if d.is_dir() and d not in with_files}
    return out


def _zarr_store_roots(listing: set[str]) -> set[str]:
    roots: set[str] = set()
    for rel in listing:
        parts = rel.split("/")
        for i, part in enumerate(parts[:-1] if rel.endswith("/") else parts):
            if part.endswith(".zarr"):
                roots.add("/".join(parts[: i + 1]))
                break
    return roots


# A name containing "datatree" is a consolidated tier: a hierarchical DataTree. Every store, this tier included,
# carries a consolidated-metadata index since layout 27 (tests/goldens/SPEC.md section 8); this comparison opens it
# with consolidated=False, which reads the documents directly and does not compare the index. Not established here:
# why the flat stores are opened the same way, why chunks=None, and why warnings are silenced.
def _open_store(path: Path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if "datatree" in path.name:
            return xr.open_datatree(path, engine="zarr", chunks=None, consolidated=False).load()
        return xr.open_zarr(path, consolidated=False).load()


def _dtypes(obj) -> dict[str, dict[str, str]]:
    """Variable dtypes per node: ``identical`` compares values, attributes and coordinates but not dtype."""
    if isinstance(obj, xr.DataTree):
        return {n.path: {k: str(v.dtype) for k, v in n.to_dataset().variables.items()} for n in obj.subtree}
    return {"/": {k: str(v.dtype) for k, v in obj.variables.items()}}


def _difference(a, b) -> str | None:
    """``"content"`` or ``"dtype"`` when two opened stores or datasets differ, ``None`` when equal."""
    if not a.identical(b):
        return "content"
    if _dtypes(a) != _dtypes(b):
        return "dtype"
    return None


def compare_trees(fresh: Path, committed: Path) -> list[str]:
    """Differences between two extracted golden trees by content; an empty list means equal."""
    left, right = _listing(fresh), _listing(committed)
    problems = [f"only in fresh generation: {p}" for p in sorted(left - right)]
    problems += [f"only in committed zip: {p}" for p in sorted(right - left)]
    stores = _zarr_store_roots(left & right)
    for rel in sorted(stores):
        try:
            kind = _difference(_open_store(fresh / rel), _open_store(committed / rel))
        except Exception as exc:  # a store that cannot be read is a difference, with its reason
            problems.append(f"store {rel} could not be compared: {type(exc).__name__}: {exc}")
            continue
        if kind:
            problems.append(f"zarr store differs by {kind}: {rel}")
    for rel in sorted(p for p in left & right if not p.endswith("/")):
        if any(rel == s or rel.startswith(s + "/") for s in stores):
            continue
        a, b = fresh / rel, committed / rel
        if rel.endswith(".nc"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with xr.open_dataset(a) as da, xr.open_dataset(b) as db:
                    kind = _difference(da, db)
                    if kind:
                        problems.append(f"netcdf differs by {kind}: {rel}")
        elif a.read_bytes() != b.read_bytes():
            problems.append(f"file differs: {rel}")
    return problems


@pytest.fixture(scope="module")
def regenerated(tmp_path_factory: pytest.TempPathFactory):
    """A temporary directory for the regeneration test's two parametrised cases, removed afterwards.

    Generation and extraction happen in ``test_committed_zip_equals_a_fresh_generation``, not here."""
    base = tmp_path_factory.mktemp("golden_regeneration")
    try:
        yield base
    finally:
        shutil.rmtree(base, ignore_errors=True)


def _extract(zip_path: Path, dest: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            assert not name.startswith("/") and ".." not in Path(name).parts, f"unsafe member name {name!r}"
        zf.extractall(dest)


@pytest.mark.parametrize("kind", sorted(GENERATORS))
def test_committed_zip_equals_a_fresh_generation(kind: str, regenerated: Path) -> None:
    zip_path = GOLDEN_DIR / f"{kind}.zip"
    assert zip_path.is_file(), f"{zip_path} is not committed; run `python -m tests.test_golden_regeneration`"
    with zipfile.ZipFile(zip_path) as zf:
        infos = zf.infolist()
    n_files = sum(1 for i in infos if not i.is_dir())
    size = zip_path.stat().st_size
    print(f"{kind}.zip: {size} bytes, {n_files} files")
    assert size > 0
    assert n_files > 0
    assert {i.date_time for i in infos} == {ZIP_TIME}, "zip members must carry the fixed time"

    fresh = regenerated / f"{kind}_fresh"
    committed = regenerated / f"{kind}_zip"
    fresh.mkdir()
    GENERATORS[kind](fresh)
    _extract(zip_path, committed)
    assert compare_trees(fresh, committed) == []


@pytest.mark.parametrize("kind", sorted(GENERATORS))
def test_committed_zip_is_relocatable(kind: str) -> None:
    """No text member (.yaml, .json, .cfg, .csv, .txt, .inp, .rpt) names a path under /home/, /tmp/ or /Users/.

    At least one text member carries the ``@GOLDEN_ROOT@`` root token, so the absence is not an empty scan.
    The generator writes that token for the golden's own root so that a golden can be extracted into a different
    directory for every test; ``tests/fixtures/golden_trees.py`` replaces it on extraction (``tests/goldens/SPEC.md``
    section 5.2)."""
    token_seen = False
    with zipfile.ZipFile(GOLDEN_DIR / f"{kind}.zip") as zf:
        for name in zf.namelist():
            if name.endswith("/") or not name.endswith((".yaml", ".json", ".cfg", ".csv", ".txt", ".inp", ".rpt")):
                continue
            text = zf.read(name).decode("utf-8", errors="replace")
            assert "/home/" not in text and "/tmp/" not in text and "/Users/" not in text, name
            token_seen = token_seen or "@GOLDEN_ROOT@" in text
    assert token_seen


def _control_pair(tmp_path: Path, name: str) -> tuple[Path, Path]:
    """Two equal trees: a Zarr store with a variable attribute, a NetCDF file, a text file, an empty directory."""
    one, two = tmp_path / name / "one", tmp_path / name / "two"
    for root in (one, two):
        (root / "a.zarr").mkdir(parents=True)
        ds = xr.Dataset({"v": (("x",), [1.0, 2.0], {"units": "m"})})
        ds.to_zarr(root / "a.zarr", mode="w", consolidated=False)
        ds.to_netcdf(root / "w.nc")
        (root / "t.txt").write_text("same")
        (root / "gone.txt").write_text("x")
        (root / "keep").mkdir()
    return one, two


def test_the_comparison_can_fail(tmp_path: Path) -> None:
    """Must-fire control: each planted difference is reported by its problem string, and an equal pair reports none.

    Not established: the unreadable-store arm of ``compare_trees`` (``could not be compared``) has no planted case."""
    one, two = _control_pair(tmp_path, "equal")
    assert compare_trees(one, two) == []

    def planted(name: str, edit) -> list[str]:
        a, b = _control_pair(tmp_path, name)
        edit(a, b)
        return compare_trees(a, b)

    def value(a, b):
        xr.Dataset({"v": (("x",), [1.0, 3.0], {"units": "m"})}).to_zarr(b / "a.zarr", mode="w", consolidated=False)

    def text(a, b):
        (b / "t.txt").write_text("different")

    def dropped_file(a, b):
        (b / "gone.txt").unlink()

    def attribute(a, b):
        xr.Dataset({"v": (("x",), [1.0, 2.0])}).to_zarr(b / "a.zarr", mode="w", consolidated=False)

    def extra_file(a, b):
        (b / "extra.txt").write_text("x")

    def empty_dir(a, b):
        (a / "only_here").mkdir()

    def netcdf_value(a, b):
        (b / "w.nc").unlink()
        xr.Dataset({"v": (("x",), [1.0, 9.0], {"units": "m"})}).to_netcdf(b / "w.nc")

    def zarr_dtype(a, b):
        ds = xr.Dataset({"v": (("x",), np.array([1.0, 2.0], dtype="float32"), {"units": "m"})})
        ds.to_zarr(b / "a.zarr", mode="w", consolidated=False)

    def zarr_int_dtype(a, b):
        ds = xr.Dataset({"v": (("x",), np.array([1, 2], dtype="int64"), {"units": "m"})})
        ds.to_zarr(b / "a.zarr", mode="w", consolidated=False)

    def netcdf_dtype(a, b):
        (b / "w.nc").unlink()
        ds = xr.Dataset({"v": (("x",), np.array([1.0, 2.0], dtype="float32"), {"units": "m"})})
        ds.to_netcdf(b / "w.nc")

    assert "zarr store differs by content: a.zarr" in planted("value", value)
    assert "file differs: t.txt" in planted("text", text)
    assert "only in fresh generation: gone.txt" in planted("dropped", dropped_file)
    assert "zarr store differs by content: a.zarr" in planted("attribute", attribute)
    assert "only in committed zip: extra.txt" in planted("extra", extra_file)
    assert "only in fresh generation: only_here/" in planted("emptydir", empty_dir)
    assert "netcdf differs by content: w.nc" in planted("ncvalue", netcdf_value)
    assert "zarr store differs by dtype: a.zarr" in planted("dtype", zarr_dtype)
    assert "zarr store differs by dtype: a.zarr" in planted("intdtype", zarr_int_dtype)
    assert "netcdf differs by dtype: w.nc" in planted("ncdtype", netcdf_dtype)


def main() -> int:
    """Regenerate both committed zips."""
    import tempfile

    for kind, generate in GENERATORS.items():
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / kind
            root.mkdir()
            generate(root)
            build_zip(root, GOLDEN_DIR / f"{kind}.zip")
        print(f"wrote {GOLDEN_DIR / (kind + '.zip')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
