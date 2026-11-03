"""Tests for the deposit zip ``publishing._resolve_upload`` writes for a zarr-store directory.

A directory upload used to be zipped with every entry deflated at zlib's default level. It is now
written through ``bundle._archive.write_zip`` under the moderate policy: DEFLATE at level 6 for every
entry except zarr chunk files, which are STORED. A non-directory upload is passed through untouched.

Every test builds a small store under ``tmp_path`` and uses only the standard library and the function
under test, so none compiles or runs a solver and all run on a laptop. Run from a worktree as
``PYTHONPATH=${PWD}/src pytest tests/test_publish_zip_policy.py``.
"""

from __future__ import annotations

import random
import zipfile
import zlib
from pathlib import Path

from hhemt.publishing import _resolve_upload


def _numeric_csv(seed: int = 3, rows: int = 4000) -> bytes:
    """Gaussian CSV text whose deflate size differs at levels 1, 6 and 9."""
    rng = random.Random(seed)
    return "\n".join(",".join(f"{rng.gauss(0, 1):.3f}" for _ in range(6)) for _ in range(rows)).encode()


def _store(tmp_path: Path) -> Path:
    store = tmp_path / "up" / "s.zarr"
    (store / "c" / "0").mkdir(parents=True)
    (store / "zarr.json").write_bytes(b'{"zarr_format": 3}' * 50)
    (store / ".zattrs").write_bytes(_numeric_csv())
    (store / "c" / "0" / "0").write_bytes(bytes(range(256)) * 20)
    (store / "c" / "0" / "1").write_bytes(bytes(reversed(range(256))) * 20)
    return store


_EXPECTED_TYPE = {
    "s.zarr/.zattrs": zipfile.ZIP_DEFLATED,
    "s.zarr/c/0/0": zipfile.ZIP_STORED,
    "s.zarr/c/0/1": zipfile.ZIP_STORED,
    "s.zarr/zarr.json": zipfile.ZIP_DEFLATED,
}


def _wrong_types(zip_path: Path) -> list[str]:
    with zipfile.ZipFile(zip_path) as zf:
        actual = {i.filename: i.compress_type for i in zf.infolist()}
    return sorted(name for name, method in _EXPECTED_TYPE.items() if actual.get(name) != method)


def test_deposit_zip_stores_chunk_files_and_deflates_the_rest(tmp_path):
    """Property: chunk entries are STORED and every other entry DEFLATED, named relative to the store's parent.

    Class: NEW CAPABILITY and a discriminator: before this change every entry was DEFLATED, so the two
    chunk entries read ZIP_DEFLATED and the comparison reds. The plausible wrong implementation is the old
    all-DEFLATED writer or an all-STORED one. Why required: chunk files are already compressed, so
    deflating them spends CPU for about 2 percent, while the other entries shrink.

    The must-fire control is the next test. A second correct implementation is ``zf.write`` with a
    per-entry ``compress_type`` chosen from the name.
    """
    store = _store(tmp_path)
    staging = tmp_path / "staging"
    staging.mkdir()
    archive, key = _resolve_upload(store, staging)
    assert (archive, key) == (staging / "s.zarr.zip", "s.zarr.zip")
    assert _wrong_types(archive) == []
    with zipfile.ZipFile(archive) as zf:
        assert sorted(zf.namelist()) == sorted(_EXPECTED_TYPE)
        assert zf.testzip() is None
        for name in _EXPECTED_TYPE:
            assert zf.read(name) == (store.parent / name).read_bytes()


def test_type_comparison_reports_a_deflated_chunk_entry(tmp_path):
    """Must-fire control for the comparison above: an all-DEFLATED zip has exactly the two chunk entries wrong."""
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in _EXPECTED_TYPE:
            zf.writestr(name, b"payload" * 30)
    assert _wrong_types(bad) == ["s.zarr/c/0/0", "s.zarr/c/0/1"]


def test_deposit_zip_deflates_at_level_six(tmp_path):
    """Property: a deflated entry's stream is the one DEFLATE at zlib level 6 produces.

    Class: NEW CAPABILITY. The payload deflates to three different sizes at levels 1, 6 and 9 (asserted
    first). Not established: a writer that leaves the level unset passes, because zlib's default is
    also 6. The plausible wrong implementations are level 1 or 9. Why required: level 6 is the policy
    the size and CPU figures were taken at. A second correct implementation is ``writestr`` with
    ``compresslevel=6``, whose bytes are identical.
    """
    payload = _numeric_csv()

    def _raw(level: int) -> bytes:
        c = zlib.compressobj(level, zlib.DEFLATED, -15)
        return c.compress(payload) + c.flush()

    sizes = {level: len(_raw(level)) for level in (1, 6, 9)}
    assert len(set(sizes.values())) == 3
    archive, _ = _resolve_upload(_store(tmp_path), tmp_path)
    raw = archive.read_bytes()
    with zipfile.ZipFile(archive) as zf:
        info = zf.getinfo("s.zarr/.zattrs")
    name_len = int.from_bytes(raw[info.header_offset + 26 : info.header_offset + 28], "little")
    extra_len = int.from_bytes(raw[info.header_offset + 28 : info.header_offset + 30], "little")
    start = info.header_offset + 30 + name_len + extra_len
    assert raw[start : start + info.compress_size] == _raw(6)


def test_a_plain_file_upload_is_passed_through_unchanged(tmp_path):
    """Property, differently positioned: a path that is not a directory is returned as given, with its own name.

    Class: REGRESSION (holds before and after). The plausible wrong implementation is zipping every
    upload. Why required: single-file deposits must keep their filename. A second correct implementation
    is returning a hard link, which still equals the file by content and name.
    """
    f = tmp_path / "report.pdf"
    f.write_bytes(b"%PDF")
    staging = tmp_path / "staging"
    staging.mkdir()
    assert _resolve_upload(f, staging) == (f, "report.pdf")
    assert list(staging.iterdir()) == []
