"""Tests for ``hhemt.bundle._archive``: reproducible zip writing and threaded extraction.

Every test builds small files under ``tmp_path`` and uses only the standard library plus
the module under test, so none compiles or runs a solver and all run on a laptop. They are
run from a worktree as ``PYTHONPATH=${PWD}/src pytest tests/test_bundle_archive.py``.

Each test docstring names its property, its class (NEW CAPABILITY: the module is new, so
there is no pre-change tree to be red on), the plausible wrong implementation that turns it
red, and a second correct implementation under which it still passes. Expectations are
written as literals or recomputed with ``zipfile``/``zlib`` directly, never by calling the
function under test to produce its own expected value.
"""

from __future__ import annotations

import os
import random
import struct
import zipfile
import zlib
from pathlib import Path

import pytest

from hhemt.bundle._archive import (
    MODERATE,
    CompressionPolicy,
    copy_into_zip,
    extract_zip,
    is_zarr_chunk,
    render_set,
    write_zip,
)


def _make_tree(root: Path, files: dict[str, bytes]) -> dict[str, Path]:
    """Create ``files`` under ``root`` and return ``{arcname: path}`` for them."""
    out: dict[str, Path] = {}
    for name, payload in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        out[name] = path
    return out


def _compressible(seed: int, n: int = 40_000) -> bytes:
    """Text-like bytes that deflate well and deflate DIFFERENTLY at different levels."""
    words = [f"word{(seed * 7 + i * 13) % 97}" for i in range(n // 8)]
    return " ".join(words).encode()


def _numeric_csv(seed: int, rows: int = 4000) -> bytes:
    """Gaussian numbers as CSV text: deflate output differs at levels 1, 6 and 9 (word text does not)."""
    rng = random.Random(seed)
    lines = [",".join(f"{rng.gauss(0, 1):.3f}" for _ in range(6)) for _ in range(rows)]
    return "\n".join(lines).encode()


def _raw_payload(zip_path: Path, name: str) -> bytes:
    """The bytes of entry ``name`` exactly as stored in the zip (compressed if deflated)."""
    with zipfile.ZipFile(zip_path) as zf:
        info = zf.getinfo(name)
    with zip_path.open("rb") as fh:
        fh.seek(info.header_offset)
        header = fh.read(30)
        name_len, extra_len = struct.unpack("<HH", header[26:30])
        fh.seek(info.header_offset + 30 + name_len + extra_len)
        return fh.read(info.compress_size)


# ---------------------------------------------------------------- is_zarr_chunk


def test_is_zarr_chunk_classifies_literal_names() -> None:
    """Property: only files under a ``*.zarr`` directory that are not metadata documents are chunks.

    Class: NEW CAPABILITY. Wrong implementation: ``name.endswith(".zarr")`` or a check
    that ignores the metadata filenames; red because the table has a chunk path that
    does not end in ``.zarr`` and metadata paths that sit under one. Reachability: every
    name below is a path a bundle holds (v3 chunk ``c/0/0``, v2 chunk ``0.0``, a v2
    ``.zmetadata``). Both-states anchor: the assertions turn on the name's directory
    segments, not on any wording the module introduces. Second correct implementation: a
    regex on ``(^|/)[^/]*\\.zarr/`` over the directory part with the five filenames excluded.
    """
    expected = {
        "a.zarr/var/c/0/0": True,
        "a.zarr/var/0.0": True,
        "plots/a.zarr/x/c/1": True,
        "a.zarr/b/inner.zarr/v/c/0": True,
        "a.zarr/zarr.json": False,
        "a.zarr/var/zarr.json": False,
        "a.zarr/var/.zarray": False,
        "a.zarr/.zgroup": False,
        "a.zarr/.zattrs": False,
        "a.zarr/.zmetadata": False,
        "plots/fig.png": False,
        "notzarr/var/c/0": False,
        "a.zarray/c/0": False,
        "x.zarr": False,
        "dir/x.zarr": False,
    }
    got = {name: is_zarr_chunk(name) for name in expected}
    assert got == expected


def test_zarr_metadata_names_match_zarr_constants() -> None:
    """Property: the module's metadata-filename set equals the one zarr itself defines.

    Class: NEW CAPABILITY. Wrong implementation: a set missing ``.zmetadata`` or carrying
    a misspelling; red because the comparison is exact. Reachability: ``hhemt`` pins zarr
    3.x, whose ``zarr.core.common`` exports these constants. Both-states anchor: the
    expected set comes from zarr, not from this module. Second correct implementation:
    importing the constants from zarr inside the module instead of writing them out.
    """
    from zarr.core.common import ZARR_JSON, ZARRAY_JSON, ZATTRS_JSON, ZGROUP_JSON, ZMETADATA_V2_JSON

    from hhemt.bundle import _archive

    assert _archive._ZARR_METADATA_NAMES == {ZARR_JSON, ZARRAY_JSON, ZGROUP_JSON, ZATTRS_JSON, ZMETADATA_V2_JSON}


# ---------------------------------------------------------------- write_zip


def _compress_types(zip_path: Path) -> dict[str, int]:
    with zipfile.ZipFile(zip_path) as zf:
        return {i.filename: i.compress_type for i in zf.infolist()}


def test_write_zip_is_byte_identical_on_repeat_and_stores_exactly_the_zarr_chunks(tmp_path: Path) -> None:
    """Property: two writes give identical bytes; chunk entries are STORED and all others deflated.

    Class: NEW CAPABILITY. Wrong implementation: deflate everything, or store everything,
    or let a file timestamp into the entry; red on the stored-set equality and on the
    byte comparison. Reachability: the chunk and the text file below are both highly
    compressible, so a deflated chunk is visibly smaller than the stored original.
    Both-states anchor: the stored set is compared with a LITERAL set of names, not with
    ``is_zarr_chunk``. Invocation: not isolatable from this tree (the module is new);
    discrimination is the mutation that makes ``is_zarr_chunk`` return False. Second
    correct implementation: a writer that stores by a regex over the arcname.
    """
    chunk_payload = _compressible(1)
    files = _make_tree(
        tmp_path / "src",
        {
            "s.zarr/var/c/0/0": chunk_payload,
            "s.zarr/zarr.json": b'{"zarr_format": 3}' * 50,
            "report.html": _compressible(2),
            "sims/out_triton/bin/H_00.bin": _compressible(3),
        },
    )
    first = write_zip(files.items(), tmp_path / "one.zip")
    os.utime(files["report.html"], (1_000_000_000, 1_000_000_000))
    second = write_zip(files.items(), tmp_path / "two.zip")
    assert first.read_bytes() == second.read_bytes()

    types = _compress_types(first)
    stored = {n for n, t in types.items() if t == zipfile.ZIP_STORED}
    assert stored == {"s.zarr/var/c/0/0"}
    assert {n for n, t in types.items() if t == zipfile.ZIP_DEFLATED} == set(files) - stored
    assert _raw_payload(first, "s.zarr/var/c/0/0") == chunk_payload


def test_write_zip_detects_a_deflated_zarr_chunk_when_the_policy_asks_for_it(tmp_path: Path) -> None:
    """Must-fire control for the test above: the stored-set check reds on a deflating policy.

    Class: NEW CAPABILITY. Wrong implementation: none; this proves the instrument.
    The policy here deflates chunks, so the same detector must report the chunk entry as
    deflated and the stored set must be empty. Second correct implementation: n/a.
    """
    files = _make_tree(tmp_path / "src", {"s.zarr/var/c/0/0": _compressible(1)})
    out = write_zip(files.items(), tmp_path / "x.zip", policy=CompressionPolicy(level=6, store_zarr_chunks=False))
    types = _compress_types(out)
    assert types == {"s.zarr/var/c/0/0": zipfile.ZIP_DEFLATED}


def test_write_zip_orders_by_path_parts_and_stamps_a_fixed_date_and_mode(tmp_path: Path) -> None:
    """Property: entries are ordered by path PARTS and every entry has the fixed date and mode.

    Class: NEW CAPABILITY. Wrong implementation: ``sorted(arcnames)`` on the joined
    string, or a real file mtime in the entry; red because ``"a-b/x"`` sorts before
    ``"a/b"`` as a string but after it by parts. Reachability: both names are ordinary
    bundle paths. Both-states anchor: the expected order is written out as a literal.
    Second correct implementation: ``sorted(names, key=lambda n: tuple(n.split("/")))``.
    """
    files = _make_tree(tmp_path / "src", {"a-b/x": b"1", "a/b": b"2", "a/a/z": b"3", "z": b"4"})
    os.utime(files["z"], (1_700_000_000, 1_700_000_000))
    out = write_zip(files.items(), tmp_path / "o.zip")
    with zipfile.ZipFile(out) as zf:
        infos = zf.infolist()
    assert [i.filename for i in infos] == ["a/a/z", "a/b", "a-b/x", "z"]
    assert {i.date_time for i in infos} == {(1980, 1, 1, 0, 0, 0)}
    assert {i.external_attr >> 16 for i in infos} == {0o644}


def test_write_zip_gives_the_same_entries_for_path_bytes_and_zip_member_sources(tmp_path: Path) -> None:
    """Property: a path, the same bytes, and an entry of another zip produce an identical archive.

    Class: NEW CAPABILITY. Wrong implementation: a source kind that is read into the
    wrong place or skipped; red on the byte comparison. Reachability: the combine reads
    members from zips and a placeholder rewrite hands text over as bytes. Both-states
    anchor: three archives are compared with each other, not with a stored expectation.
    Second correct implementation: materialise each source to a temporary file first.
    """
    payload = _compressible(5)
    files = _make_tree(tmp_path / "src", {"d/f.txt": payload, "d/g.txt": b"g"})
    from_paths = write_zip(files.items(), tmp_path / "paths.zip")
    from_bytes = write_zip([("d/f.txt", payload), ("d/g.txt", b"g")], tmp_path / "bytes.zip")
    with zipfile.ZipFile(from_paths) as source:
        from_zip = write_zip(
            [("d/f.txt", (source, "d/f.txt")), ("d/g.txt", (source, "d/g.txt"))],
            tmp_path / "zip.zip",
        )
    assert from_paths.read_bytes() == from_bytes.read_bytes() == from_zip.read_bytes()


def test_write_zip_policy_level_reaches_the_compressor(tmp_path: Path) -> None:
    """Property: the policy's level is the zlib level the entry is compressed at.

    Class: NEW CAPABILITY. Wrong implementation: ignoring the level, or hard-coding
    another (``MODERATE.level`` read as 9, or ``info._compresslevel`` left unset and
    zlib's default taken); red because the level-1 and level-9 payloads differ from each
    other and ``MODERATE`` must equal a level-6 stream made by ``zlib`` directly.
    Reachability: the data is numeric CSV text, measured to deflate to different bytes at levels 1, 6 and 9.
    Both-states anchor: the oracle is ``zlib.compressobj`` called here. Second correct
    implementation: compressing each entry with ``zlib.compressobj(level, DEFLATED, -15)``
    and writing the raw stream.
    """
    payload = _numeric_csv(1)
    files = _make_tree(tmp_path / "src", {"t.txt": payload})

    def deflated(level: int, name: str) -> bytes:
        out = write_zip(files.items(), tmp_path / name, policy=CompressionPolicy(level=level, store_zarr_chunks=True))
        return _raw_payload(out, "t.txt")

    def oracle(level: int) -> bytes:
        c = zlib.compressobj(level, zlib.DEFLATED, -15)
        return c.compress(payload) + c.flush()

    assert deflated(1, "l1.zip") == oracle(1)
    assert deflated(9, "l9.zip") == oracle(9)
    assert oracle(1) != oracle(6) != oracle(9)
    assert MODERATE.level == 6
    assert _raw_payload(write_zip(files.items(), tmp_path / "m.zip"), "t.txt") == oracle(6)


def test_write_zip_leaves_no_dest_and_no_partial_on_failure_and_keeps_an_existing_dest(tmp_path: Path) -> None:
    """Property: a failed write removes ``.partial``, creates no ``dest``, and keeps an older ``dest`` intact.

    Class: NEW CAPABILITY. Wrong implementation: writing straight to ``dest`` (a failure
    then leaves a truncated zip, or replaces a good one); red on both assertions.
    Reachability: a source path that does not exist is an ordinary mid-write failure and
    is raised after earlier entries were already written. Both-states anchor: the check
    reads the filesystem, not the function's own account of what it did. Second correct
    implementation: writing to a temporary file in the same directory and renaming.
    """
    files = _make_tree(tmp_path / "src", {"a.txt": b"a", "z.txt": b"z"})
    dest = tmp_path / "out.zip"
    bad = [*files.items(), ("m.txt", tmp_path / "src" / "missing.txt")]
    with pytest.raises(FileNotFoundError):
        write_zip(bad, dest)
    assert not dest.exists()
    assert not Path(str(dest) + ".partial").exists()

    write_zip(files.items(), dest)
    good_bytes = dest.read_bytes()
    with pytest.raises(FileNotFoundError):
        write_zip(bad, dest)
    assert dest.read_bytes() == good_bytes
    assert not Path(str(dest) + ".partial").exists()


def test_write_zip_replaces_an_existing_dest_and_leaves_no_partial(tmp_path: Path) -> None:
    """Property: a successful write over an existing ``dest`` replaces it and leaves no ``.partial``.

    Class: NEW CAPABILITY. Wrong implementation: refusing to overwrite, or leaving the
    partial file behind after the rename; red on the content check and on the partial
    check. Reachability: a re-run of a combine writes onto its previous output. Both-states
    anchor: the new content is read back with ``zipfile``. Second correct implementation:
    ``os.rename`` after ``unlink`` of the old file.
    """
    dest = tmp_path / "out.zip"
    first = _make_tree(tmp_path / "s1", {"old.txt": b"old"})
    second = _make_tree(tmp_path / "s2", {"new.txt": b"new"})
    write_zip(first.items(), dest)
    write_zip(second.items(), dest)
    with zipfile.ZipFile(dest) as zf:
        assert zf.namelist() == ["new.txt"]
    assert not Path(str(dest) + ".partial").exists()


def test_write_zip_refuses_a_repeated_archive_name(tmp_path: Path) -> None:
    """Property: two entries with one arcname raise ``ValueError`` and write nothing.

    Class: NEW CAPABILITY. Wrong implementation: writing both (``zipfile`` only warns and
    the archive then has two entries of one name, one unreachable by name); red on the
    raise. Reachability: ``entries`` is a plain iterable, so a caller can repeat a name.
    Both-states anchor: the check is on the raised type and the absent file. Second
    correct implementation: building a dict from the pairs and raising when a key is already present.
    """
    files = _make_tree(tmp_path / "src", {"a.txt": b"a"})
    with pytest.raises(ValueError, match="more than once"):
        write_zip([("a.txt", files["a.txt"]), ("a.txt", b"other")], tmp_path / "o.zip")
    assert not (tmp_path / "o.zip").exists()


def test_copy_into_zip_stores_the_file_whole_with_the_fixed_stamps(tmp_path: Path) -> None:
    """Property: ``copy_into_zip`` writes the file STORED, byte for byte, with the fixed date and mode.

    Class: NEW CAPABILITY. Wrong implementation: letting the ``ZipFile``'s own compression
    apply; red because the zip below is opened with DEFLATED and the payload is
    compressible. Reachability: the file is itself a zip, the case the function exists
    for. Both-states anchor: the stored bytes are compared with the source file's bytes.
    Second correct implementation: ``zf.write(src, arcname, compress_type=ZIP_STORED)``
    followed by overwriting the date and mode on the ``ZipInfo``.
    """
    inner = tmp_path / "member.zip"
    with zipfile.ZipFile(inner, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("x.txt", _compressible(4))
    outer = tmp_path / "outer.zip"
    with zipfile.ZipFile(outer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        copy_into_zip(zf, "child_crates/e1.zip", inner)
    with zipfile.ZipFile(outer) as zf:
        info = zf.getinfo("child_crates/e1.zip")
        assert info.compress_type == zipfile.ZIP_STORED
        assert info.date_time == (1980, 1, 1, 0, 0, 0)
        assert info.external_attr >> 16 == 0o644
        assert zf.read("child_crates/e1.zip") == inner.read_bytes()


# ---------------------------------------------------------------- render_set


def test_render_set_omits_exactly_the_out_bin_entries() -> None:
    """Property: ``render_set`` drops names under ``.../out_*/bin/`` and keeps everything else in order.

    Class: NEW CAPABILITY. Wrong implementation: dropping any ``bin`` directory, any
    ``out_*`` directory, or a file named ``bin``; red because the table keeps a sibling
    ``bin/`` outside ``out_*``, an ``outer/bin/``, an ``out_x/binary/`` and a file called
    ``bin``. Reachability: these are layouts a scenario folder produces. Both-states
    anchor: the kept list is a literal. Second correct implementation: a regex
    ``(^|/)[^/]+/out_[^/]*/bin/`` search on the name.
    """
    names = [
        "analysis/sims/0/out_triton/bin/H_00_00.bin",
        "analysis/sims/0/out_tritonswmm/bin/QX_00_00.bin",
        "analysis/sims/0/out_swmm/bin/x/y.bin",
        "analysis/sims/0/out_triton/cfg/run.cfg",
        "analysis/sims/0/bin/tool",
        "analysis/sims/0/outer/bin/tool",
        "analysis/sims/0/out_triton/binary/w",
        "analysis/sims/0/out_triton/bin",
        "analysis/plots/fig.png",
    ]
    assert render_set(names) == [
        "analysis/sims/0/out_triton/cfg/run.cfg",
        "analysis/sims/0/bin/tool",
        "analysis/sims/0/outer/bin/tool",
        "analysis/sims/0/out_triton/binary/w",
        "analysis/sims/0/out_triton/bin",
        "analysis/plots/fig.png",
    ]


# ---------------------------------------------------------------- extract_zip


def _snapshot(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def _build_awkward_zip(zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for i in range(60):
            zf.writestr(f"d{i % 7}/sub{i % 3}/f{i}.txt", _compressible(i, 800))
        zf.writestr("emptydir/", b"")
        zf.writestr("../escape.txt", b"climbs")
        zf.writestr("/abs.txt", b"absolute")
        zf.writestr("a/./b/../c.txt", b"dots")


@pytest.mark.parametrize("threads", [1, 4])
def test_extract_zip_yields_the_file_set_and_bytes_of_extractall(tmp_path: Path, threads: int) -> None:
    """Property: extraction gives the same files, bytes and directories as ``ZipFile.extractall``,
    and an entry that climbs out of ``dest`` is confined as ``extractall`` confines it.

    Class: NEW CAPABILITY. Wrong implementation: ``dest / name`` as the target, or a
    hand-made sanitiser that raises or drops the entry; red because ``../escape.txt``
    would land outside ``dest`` and ``/abs.txt`` at the filesystem root. Reachability:
    the zip carries a climbing name, an absolute name, a dot segment and a directory
    entry, all of which a hostile or odd archive can hold; the control assertion shows
    the climbing name really does leave ``dest`` under naive joining. Both-states anchor:
    the expectation is ``extractall``'s own result on the same zip. Second correct
    implementation: calling ``ZipFile.extract`` for every member in one thread.
    """
    src = tmp_path / "in.zip"
    _build_awkward_zip(src)
    ref_root, got_root = tmp_path / "ref" / "d", tmp_path / "got" / "d"
    ref_root.mkdir(parents=True)
    with zipfile.ZipFile(src) as zf:
        zf.extractall(ref_root)
    extract_zip(src, got_root, threads=threads)

    assert _snapshot(got_root) == _snapshot(ref_root)
    assert (got_root / "emptydir").is_dir()
    assert sorted(p.name for p in got_root.parent.iterdir()) == ["d"], "an entry escaped dest"
    # control: naive joining of the climbing name does leave dest, so the check above can fire
    assert (got_root / "../escape.txt").resolve().parent != got_root.resolve()


def test_extract_zip_members_none_means_all_and_members_selects_only_those(tmp_path: Path) -> None:
    """Property: ``members`` limits extraction to the named entries; ``render_set`` omits exactly the raw output.

    Class: NEW CAPABILITY. Wrong implementation: ignoring ``members`` or extracting the
    union with directories created for omitted names; red on the file-set equality and
    on the absent ``out_triton/bin`` directory. Reachability: this is the combine's
    call, ``members=render_set(names)``. Both-states anchor: the expected set is the
    names list minus a literal pair. Second correct implementation: extracting the full
    zip and deleting the ``out_*/bin`` directories afterwards.
    """
    src = tmp_path / "m.zip"
    names = [
        "a/sims/0/out_triton/bin/H.bin",
        "a/sims/0/out_triton/cfg.txt",
        "a/sims/1/out_tritonswmm/bin/Q.bin",
        "a/plots/fig.png",
    ]
    with zipfile.ZipFile(src, "w") as zf:
        for n in names:
            zf.writestr(n, n.encode())
    out = tmp_path / "out"
    with zipfile.ZipFile(src) as zf:
        wanted = render_set(zf.namelist())
    extract_zip(src, out, members=wanted)
    assert set(_snapshot(out)) == {"a/sims/0/out_triton/cfg.txt", "a/plots/fig.png"}
    assert not (out / "a/sims/0/out_triton/bin").exists()


def test_extract_zip_only_missing_or_resized_skips_complete_files_and_restores_the_rest(tmp_path: Path) -> None:
    """Property: ``only_missing_or_resized`` keeps a right-sized file and rewrites a missing or resized one.

    Class: NEW CAPABILITY. Wrong implementation: always rewriting (the same-size edit
    below would be overwritten), or never rewriting a resized file; red on either.
    Reachability: a repeat ``report-from-bundle`` into a kept directory. Both-states
    anchor: the skipped file is detected by its edited bytes surviving, which depends on
    the file, not on any message. Second correct implementation: comparing each file's
    size through ``os.scandir`` before dispatching instead of ``Path.stat``.
    """
    src = tmp_path / "r.zip"
    with zipfile.ZipFile(src, "w") as zf:
        zf.writestr("keep.txt", b"AAAA")
        zf.writestr("resized.txt", b"BBBBBBBB")
        zf.writestr("gone.txt", b"CCCC")
    out = tmp_path / "out"
    extract_zip(src, out)
    (out / "keep.txt").write_bytes(b"XXXX")  # same size: must be kept
    (out / "resized.txt").write_bytes(b"B")  # wrong size: must be restored
    (out / "gone.txt").unlink()  # missing: must be restored
    extract_zip(src, out, only_missing_or_resized=True)
    assert (out / "keep.txt").read_bytes() == b"XXXX"
    assert (out / "resized.txt").read_bytes() == b"BBBBBBBB"
    assert (out / "gone.txt").read_bytes() == b"CCCC"
    extract_zip(src, out)
    assert (out / "keep.txt").read_bytes() == b"AAAA"


def test_extract_zip_raises_what_extractall_raises(tmp_path: Path) -> None:
    """Property: a corrupt entry raises ``BadZipFile`` and an absent member raises ``KeyError`` before any write.

    Class: NEW CAPABILITY. Wrong implementation: swallowing worker exceptions (the call
    returns normally with a truncated tree) or wrapping them in another type; red on
    ``pytest.raises``. Reachability: a flipped byte inside a stored entry fails its CRC
    on read, which is what a damaged transfer produces; a ``members`` name absent from
    the zip is what a stale list produces. Both-states anchor: ``extractall`` is run on
    the same corrupt zip and must raise the same type. Second correct implementation:
    collecting worker errors and re-raising the first from the calling thread.
    """
    src = tmp_path / "c.zip"
    with zipfile.ZipFile(src, "w", compression=zipfile.ZIP_STORED) as zf:
        for i in range(12):
            zf.writestr(f"f{i}.txt", b"payload-%d" % i * 20)
    raw = bytearray(src.read_bytes())
    at = raw.index(b"payload-7")
    raw[at] ^= 0xFF
    src.write_bytes(bytes(raw))

    with pytest.raises(zipfile.BadZipFile):
        with zipfile.ZipFile(src) as zf:
            zf.extractall(tmp_path / "ref")
    with pytest.raises(zipfile.BadZipFile):
        extract_zip(src, tmp_path / "got", threads=4)

    clean = tmp_path / "clean.zip"
    with zipfile.ZipFile(clean, "w") as zf:
        zf.writestr("a.txt", b"a")
    with pytest.raises(KeyError):
        extract_zip(clean, tmp_path / "none", members=["a.txt", "missing.txt"])
    assert not (tmp_path / "none" / "a.txt").exists()


def test_extract_zip_refuses_a_thread_count_below_one(tmp_path: Path) -> None:
    """Property: ``threads`` below 1 raises ``ValueError`` and writes nothing.

    Class: NEW CAPABILITY. Wrong implementation: ``ThreadPoolExecutor(max_workers=0)``
    surfacing its own error after directories were created, or treating 0 as serial;
    red on the raised type and on the absent destination. Reachability: ``threads`` is a
    caller-chosen number. Both-states anchor: the destination directory must not exist.
    Second correct implementation: validating ``threads`` through a pydantic ``PositiveInt``
    annotation before the call body runs.
    """
    src = tmp_path / "t.zip"
    with zipfile.ZipFile(src, "w") as zf:
        zf.writestr("a.txt", b"a")
    with pytest.raises(ValueError, match="threads"):
        extract_zip(src, tmp_path / "out", threads=0)
    assert not (tmp_path / "out").exists()
