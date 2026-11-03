"""Tests for the member bundle zip written by ``hhemt.bundle._emit._emit_bundle_zip``.

The member bundle zip used to be written STORED. It is now written through
``bundle._archive.write_zip`` under the moderate compression policy: DEFLATE at zlib level 6
for every entry except zarr chunk files, which stay STORED. What must NOT change is the
entry set, the entry order, the per-entry stamps and every entry's content bytes.

Every test builds a small staging tree under ``tmp_path`` and uses only the standard library
and the function under test, so none compiles or runs a solver and all run on a laptop. Run
from a worktree as ``PYTHONPATH=${PWD}/src pytest tests/test_emit_compression_policy.py``.

Expected values are written as literals or recomputed with ``zlib`` and ``zipfile`` directly;
no test calls the writer to produce its own expectation.
"""

from __future__ import annotations

import random
import zipfile
import zlib
from pathlib import Path

import pytest

from hhemt.bundle._emit import _emit_bundle_zip

_FIXED_DATE = (1980, 1, 1, 0, 0, 0)

# Everything below is the entry set of the fixture, in the order a sort by path PARTS gives.
# `plots-extra/x.txt` is the one that tells a sort by path PARTS from a sort of the joined
# string: "plots-extra/x.txt" < "plots/fig.png" as strings ('-' sorts before '/') but
# ["plots"] < ["plots-extra"] as parts.
_EXPECTED_ORDER = [
    "cfg_analysis.yaml",
    "eda/s.zarr/.zattrs",
    "eda/s.zarr/c/0/0",
    "eda/s.zarr/zarr.json",
    "harvest/notes.txt",
    "harvest/t.zarr/c/1",
    "harvest/t.zarr/zarr.json",
    "plots/fig.manifest.json",
    "plots/fig.png",
    "plots/series.csv",
    "plots-extra/x.txt",
]

# The compression method each entry must carry: zarr chunk files STORED, everything else DEFLATED.
_EXPECTED_TYPE = {
    "cfg_analysis.yaml": zipfile.ZIP_DEFLATED,
    "eda/s.zarr/.zattrs": zipfile.ZIP_DEFLATED,
    "eda/s.zarr/c/0/0": zipfile.ZIP_STORED,
    "eda/s.zarr/zarr.json": zipfile.ZIP_DEFLATED,
    "harvest/notes.txt": zipfile.ZIP_DEFLATED,
    "harvest/t.zarr/c/1": zipfile.ZIP_STORED,
    "harvest/t.zarr/zarr.json": zipfile.ZIP_DEFLATED,
    "plots/fig.manifest.json": zipfile.ZIP_DEFLATED,
    "plots/fig.png": zipfile.ZIP_DEFLATED,
    "plots/series.csv": zipfile.ZIP_DEFLATED,
    "plots-extra/x.txt": zipfile.ZIP_DEFLATED,
}


def _wrong_types(zip_path: Path) -> list[str]:
    """The entry names in ``zip_path`` whose compression method is not the one the policy gives them."""
    with zipfile.ZipFile(zip_path) as zf:
        actual = {i.filename: i.compress_type for i in zf.infolist()}
    return sorted(name for name, method in _EXPECTED_TYPE.items() if actual.get(name) != method)


def _numeric_csv(seed: int = 5, rows: int = 4000) -> bytes:
    """Gaussian numbers as CSV text, whose deflate size differs at levels 1, 6 and 9."""
    rng = random.Random(seed)
    lines = [",".join(f"{rng.gauss(0, 1):.3f}" for _ in range(6)) for _ in range(rows)]
    return "\n".join(lines).encode()


def _write(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _build_inputs(tmp_path: Path) -> tuple[Path, dict[str, Path], dict[str, bytes]]:
    """A staging tree, a streamed plan, and the bytes every archive name must carry.

    The streamed plan also names ``cfg_analysis.yaml`` with different bytes, to exercise the
    rule that staging wins an archive-name collision.
    """
    staging = tmp_path / "staging"
    harvested = tmp_path / "analysis"
    expected: dict[str, bytes] = {}

    def _staged(name: str, payload: bytes) -> None:
        _write(staging / name, payload)
        expected[name] = payload

    _staged("cfg_analysis.yaml", b"analysis_id: demo\nlevel: 6\n" * 200)
    _staged("eda/s.zarr/.zattrs", b'{"k": "v"}' * 50)
    _staged("eda/s.zarr/c/0/0", bytes(range(256)) * 40)
    _staged("eda/s.zarr/zarr.json", b'{"zarr_format": 3}' * 20)
    _staged("plots/fig.manifest.json", b'{"plot_id": "fig"}' * 30)
    _staged("plots/fig.png", b"\x89PNG not really a png " * 100)
    _staged("plots/series.csv", _numeric_csv())

    streamed: dict[str, Path] = {}

    def _stream(name: str, payload: bytes) -> None:
        streamed[name] = _write(harvested / name, payload)
        expected[name] = payload

    _stream("harvest/notes.txt", b"harvested note\n" * 300)
    _stream("harvest/t.zarr/c/1", bytes(reversed(range(256))) * 40)
    _stream("harvest/t.zarr/zarr.json", b'{"zarr_format": 3, "streamed": true}' * 20)
    _stream("plots-extra/x.txt", b"sorted by parts, not by string\n" * 50)
    # A streamed source under a name staging already holds: the staged bytes must win.
    streamed["cfg_analysis.yaml"] = _write(harvested / "cfg_analysis.yaml", b"STREAMED COPY MUST LOSE\n")
    return staging, streamed, expected


def test_member_zip_keeps_entries_order_stamps_and_content(tmp_path):
    """Property: the policy changes how entries are COMPRESSED and nothing else about the zip.

    Class: NEW CAPABILITY (compression is new; the entry set, order, stamps and content are
    regression-guarded by the same assertions). The input class is every staging-plus-streamed
    layout, here one with nested zarr stores, a name collision and a name that sorts differently
    by parts than by joined string. The plausible wrong implementation is a writer that sorts the
    joined string, drops the fixed date, lets the streamed copy win a collision, or stores bytes
    differently from the source. Why required: the archive is compared and shared by its entries,
    so a reorder or a changed stamp makes a re-emit differ without any content change.

    A second correct implementation is ``ZipFile.writestr`` with the same ZipInfo values in the
    same order, which yields the same entry listing and passes unchanged.
    """
    staging, streamed, expected_bytes = _build_inputs(tmp_path)
    out = tmp_path / "bundle.zip"
    _emit_bundle_zip(staging, out, streamed=streamed)

    with zipfile.ZipFile(out) as zf:
        assert zf.testzip() is None
        infos = zf.infolist()
        assert [i.filename for i in infos] == _EXPECTED_ORDER
        for info in infos:
            assert info.date_time == _FIXED_DATE, info.filename
            assert info.external_attr >> 16 == 0o644, info.filename
            assert zf.read(info.filename) == expected_bytes[info.filename], info.filename


def test_member_zip_stores_zarr_chunks_and_deflates_everything_else(tmp_path):
    """Property: zarr chunk entries are STORED and every other entry is DEFLATED.

    Class: NEW CAPABILITY, and a discriminator against today's stored writer: before this change
    every entry was STORED, so each non-chunk entry below reads ZIP_STORED and the assertion
    reds. The plausible wrong implementation is the old all-STORED writer, or an all-DEFLATED
    one that also compresses chunk files that are already Blosc-compressed. Why required: the
    deflated entries are what shrink a member zip, and the stored chunks are what spare CPU on
    bytes that cannot shrink.

    The must-fire control is ``test_type_comparison_reports_a_deflated_chunk_entry``, which runs
    the same comparison over a zip that deflated its chunk entries. A second correct
    implementation is any writer that picks the method per entry from the name, for example
    ``zf.write`` with a per-call ``compress_type``.
    """
    staging, streamed, _ = _build_inputs(tmp_path)
    out = tmp_path / "bundle.zip"
    _emit_bundle_zip(staging, out, streamed=streamed)

    assert _wrong_types(out) == []


def test_type_comparison_reports_a_deflated_chunk_entry(tmp_path):
    """Must-fire control for the compression-type assertion above, on a known-bad zip.

    The zip built here deflates every entry, so the two zarr chunk entries are the ones the
    comparison used by ``test_member_zip_stores_zarr_chunks_and_deflates_everything_else`` must
    report as wrong. Without this control a comparison that could never fail would read as a pass.
    """
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        for name in _EXPECTED_ORDER:
            zf.writestr(zipfile.ZipInfo(name, _FIXED_DATE), b"payload" * 20, compress_type=zipfile.ZIP_DEFLATED)
    assert _wrong_types(bad) == ["eda/s.zarr/c/0/0", "harvest/t.zarr/c/1"]


def _stored_stream(zip_path: Path, name: str) -> bytes:
    """The compressed bytes of entry ``name`` exactly as they sit in the archive file."""
    raw = zip_path.read_bytes()
    with zipfile.ZipFile(zip_path) as zf:
        info = zf.getinfo(name)
    # Local file header: 30 fixed bytes, the name length at offset 26 and the extra-field length at 28.
    offset = info.header_offset
    name_len = int.from_bytes(raw[offset + 26 : offset + 28], "little")
    extra_len = int.from_bytes(raw[offset + 28 : offset + 30], "little")
    start = offset + 30 + name_len + extra_len
    return raw[start : start + info.compress_size]


def test_member_zip_deflates_at_level_six(tmp_path):
    """Property: a deflated entry carries exactly the stream DEFLATE at zlib level 6 produces.

    Class: NEW CAPABILITY. The payload is Gaussian CSV text whose deflate size differs between
    levels 1, 6 and 9 (asserted first, so the test cannot pass on a payload that deflates the same
    at every level). The plausible wrong implementations are a writer that sets level 1 or level 9.
    Not established: a writer that leaves the level unset passes, because zlib's default level is
    also 6, and this test does not claim to tell those two apart. Why required: level 6 is the
    moderate policy the size and CPU figures were taken at.

    A second correct implementation is ``zf.writestr`` with ``compresslevel=6`` on a ZipFile built
    with that level; its compressed bytes are identical, so the test still passes.
    """
    payload = _numeric_csv()

    def _raw_deflate(level: int) -> bytes:
        compressor = zlib.compressobj(level, zlib.DEFLATED, -15)
        return compressor.compress(payload) + compressor.flush()

    sizes = {level: len(_raw_deflate(level)) for level in (1, 6, 9)}
    assert len(set(sizes.values())) == 3, f"payload does not discriminate levels: {sizes}"

    staging = tmp_path / "staging"
    _write(staging / "plots/series.csv", payload)
    out = tmp_path / "bundle.zip"
    _emit_bundle_zip(staging, out)
    with zipfile.ZipFile(out) as zf:
        assert zf.getinfo("plots/series.csv").compress_type == zipfile.ZIP_DEFLATED
    assert _stored_stream(out, "plots/series.csv") == _raw_deflate(6)


def test_two_emits_in_one_process_are_byte_identical(tmp_path):
    """Property: two emits of the same inputs in one process give identical archive bytes.

    Class: REGRESSION on today's stored writer (it holds before and after) and the reason the
    determinism contract stays stated as "same archive bytes under the same zlib build". The input
    class is any staging-plus-streamed layout; the plausible wrong implementation embeds a clock
    or a random value (for example by letting zipfile stamp the current time on an entry). Why
    required: repeat-emission tests across the suite compare archives whole.

    A second correct implementation is a writer that deflates in a different but fixed block size;
    two runs of it are still identical to each other, so the test passes.
    """
    staging, streamed, _ = _build_inputs(tmp_path)
    first, second = tmp_path / "a.zip", tmp_path / "b.zip"
    _emit_bundle_zip(staging, first, streamed=streamed)
    _emit_bundle_zip(staging, second, streamed=streamed)
    assert first.read_bytes() == second.read_bytes()


def test_a_failed_emit_leaves_the_destination_untouched_and_no_partial(tmp_path):
    """Property: an emit that fails part way leaves the old destination as it was and no ``.partial``.

    Class: NEW CAPABILITY, and a discriminator: today's writer opens ``output_path`` itself, so a
    failure after the first entry leaves a truncated zip there. The failure is injected by naming a
    streamed source that does not exist, which raises while the archive is being written (the
    source is read, and its size taken, entry by entry). The plausible wrong implementation is a
    writer that opens the destination directly. Why required: a half-written bundle that opens as a
    zip but misses entries is worse than none.

    A second correct implementation is one that writes to a temporary name in another directory and
    renames, which also leaves the destination untouched.
    """
    staging, streamed, _ = _build_inputs(tmp_path)
    streamed["harvest/missing.txt"] = tmp_path / "analysis" / "does_not_exist.txt"
    out = tmp_path / "bundle.zip"
    out.write_bytes(b"OLD DESTINATION")
    with pytest.raises(FileNotFoundError):
        _emit_bundle_zip(staging, out, streamed=streamed)
    assert out.read_bytes() == b"OLD DESTINATION"
    assert not (tmp_path / "bundle.zip.partial").exists()


def test_a_failed_emit_where_no_destination_existed_leaves_nothing(tmp_path):
    """Property, differently positioned: with no earlier destination a failed emit leaves no file at all.

    Class: NEW CAPABILITY and a discriminator (today's writer leaves the truncated destination
    behind). The satisfying input differs from the previous test only in that nothing existed at
    the destination, which is the case where a "restore the old file" implementation has nothing to
    restore and a "delete the partial" one has to delete the destination itself. Why required: the
    two arms of that guard fail independently. A second correct implementation is the same
    temporary-name-and-rename writer.
    """
    staging, streamed, _ = _build_inputs(tmp_path)
    streamed["harvest/missing.txt"] = tmp_path / "analysis" / "does_not_exist.txt"
    out = tmp_path / "bundle.zip"
    with pytest.raises(FileNotFoundError):
        _emit_bundle_zip(staging, out, streamed=streamed)
    assert not out.exists()
    assert not (tmp_path / "bundle.zip.partial").exists()
