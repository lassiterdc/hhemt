"""The member bundle zip is rewritten, scanned AND compressed, by one writer.

A bundle emit has two jobs that used to live in two zip writers: rewriting and scanning the TEXT
members for the producer's identity (a placeholder table and a zero-user-info scanner), and writing
the archive under the moderate compression policy (DEFLATE level 6, zarr chunks STORED, a fixed
stamp and mode, a ``.partial`` file renamed into place). ``_emit_bundle_zip`` hands the first job to
``bundle._archive.write_zip`` as a per-member ``transform`` and leaves the second job in
``write_zip``, so every member the archive carries is both rewritten AND compressed.

SOLVER-FREE: every arm builds members under ``tmp_path`` and calls ``_emit_bundle_zip`` or
``write_zip`` directly. No test here requests a ``synth_*`` fixture, compiles, or runs a solver.

Producer-looking values (a login name, per-user roots) are BUILT FROM FRAGMENTS, never written as
literals: this file is itself scanned by the repository's anonymization guard.

Expected bytes come from ``zlib``, ``zipfile`` and plain string replacement, never from the writer
under test.
"""

from __future__ import annotations

import contextlib
import random
import struct
import warnings
import zipfile
import zlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from hhemt.bundle import _emit
from hhemt.bundle import _placeholders as P
from hhemt.bundle._archive import write_zip
from hhemt.bundle._emit import _emit_bundle_zip
from hhemt.bundle._reprex_gate import ZeroUserInfoScanner
from hhemt.exceptions import ProcessingError

# --------------------------------------------------------------------------------------
# Producer-shaped values, built from fragments (see module docstring)
# --------------------------------------------------------------------------------------

LOGIN = "carol" + "ine"
HOME = "/ho" + "me/" + LOGIN
ANALYSIS_DIR = "/scr" + "atch/" + LOGIN + "/runs/a1"
UNLISTED_HOME = "/ho" + "me/" + "someone" + "/x"  # a per-user path the table does not hold

_FIXED_DATE = (1980, 1, 1, 0, 0, 0)


def _table() -> P.PlaceholderTable:
    return P.PlaceholderTable.from_values({P.TOKEN_ANALYSIS_DIR: [ANALYSIS_DIR], P.TOKEN_HOME: [HOME]})


def _numeric_csv(seed: int = 5, rows: int = 4000) -> bytes:
    """Gaussian numbers as CSV text, whose deflate size differs at zlib levels 1, 6 and 9."""
    rng = random.Random(seed)
    lines = [",".join(f"{rng.gauss(0, 1):.3f}" for _ in range(6)) for _ in range(rows)]
    return "\n".join(lines).encode()


def _with_path(path: str, seed: int = 5) -> bytes:
    """A member far larger than the sniff prefix whose LAST line names ``path``."""
    return _numeric_csv(seed) + f"\n# dir {path}\n".encode()


def _put(path: Path, data: bytes | str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data if isinstance(data, bytes) else data.encode())
    return path


def _raw_deflate(payload: bytes, level: int) -> bytes:
    compressor = zlib.compressobj(level, zlib.DEFLATED, -15)
    return compressor.compress(payload) + compressor.flush()


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


def _scanner(monkeypatch) -> ZeroUserInfoScanner:
    """A real scanner with the generic layers only (no private list configured)."""
    monkeypatch.delenv("HHEMT_REPREX_BLOCKLIST", raising=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ZeroUserInfoScanner.for_emit(SimpleNamespace())


def _world(tmp_path: Path):
    """Members of every class the emit distinguishes, with the bytes each one is built from.

    Returns (staging, streamed, declared, original): ``original`` maps archive name -> source bytes.
    """
    staging = tmp_path / "staging"
    src = tmp_path / "src"
    original: dict[str, bytes] = {}
    streamed: dict[str, Path] = {}

    def staged(name: str, data: bytes) -> None:
        _put(staging / name, data)
        original[name] = data

    def stream(name: str, data: bytes) -> None:
        streamed[name] = _put(src / name, data)
        original[name] = data

    staged("Snakefile.source", _with_path(ANALYSIS_DIR, 5))  # staged text: rewritten
    staged("external/weather.csv", _with_path(ANALYSIS_DIR, 6))  # declared input: scanned, never rewritten
    staged("binary.bin", b"\x00\x01" + _with_path(ANALYSIS_DIR, 7))  # NUL in the prefix: binary
    stream("harvest/model.inp", _with_path(ANALYSIS_DIR, 8))  # streamed text: rewritten
    stream("store.zarr/c/0", _with_path(ANALYSIS_DIR, 9))  # chunk-named: binary by NAME, plain text inside
    stream("store.zarr/zarr.json", b'{"attributes": {"source_file": "' + ANALYSIS_DIR.encode() + b'/in.nc"}}')
    return staging, streamed, frozenset({"external/weather.csv"}), original


def _expected_rewritten(data: bytes) -> bytes:
    return data.replace(ANALYSIS_DIR.encode(), P.TOKEN_ANALYSIS_DIR.encode())


class _CountingHandle:
    """A read-only proxy that adds up the bytes ``read`` hands back, so a test can tell a prefix read from none."""

    def __init__(self, handle, ledger: dict):
        self._handle = handle
        self._ledger = ledger

    def read(self, *args):
        data = self._handle.read(*args)
        self._ledger["bytes_read"] += len(data)
        return data

    def readinto(self, buffer):
        count = self._handle.readinto(buffer)
        self._ledger["bytes_read"] += count or 0
        return count

    def __getattr__(self, name):
        return getattr(self._handle, name)

    def __enter__(self):
        self._handle.__enter__()
        return self

    def __exit__(self, *exc):
        return self._handle.__exit__(*exc)


# --------------------------------------------------------------------------------------
# What a member holds and how it is compressed
# --------------------------------------------------------------------------------------


def test_a_rewritten_text_member_is_also_deflated_at_level_six(tmp_path):
    """Property: through ``_emit_bundle_zip`` with a table, a text member arrives with its placeholder AND
        carries ZIP_DEFLATED at level 6; a declared input and a NUL-prefixed member arrive byte-identical and
        DEFLATED; a zarr chunk arrives byte-identical and STORED; a zarr metadata document is rewritten.

    Class: NEW CAPABILITY - plausible wrong implementations: the run branch's loop (rewrites, writes STORED, so
        the DEFLATED arm reds) and develop's body (compresses, never rewrites, so the placeholder arm reds).
    Reachability: a real emit builds the table and passes it; the members are the census classes (staged text,
        streamed text, declared input, binary, zarr chunk, zarr metadata document).
    Both-states anchor: the assertions turn on each member's compression method and on the bytes read back,
        which any correct emitter fixes, not on wording this change introduced.
    Invocation isolated: ``_emit_bundle_zip`` reads only its arguments; the payload deflates differently at
        levels 1, 6 and 9 (asserted first) so a wrong level cannot pass.
    What kills it: a writer that stores, a writer that does not rewrite, a level other than 6, a rewrite of the
        declared input or of a chunk.
    Not established: a writer that leaves the level unset passes, since zlib's default level is also 6.
    Second correct implementation: ``ZipFile.writestr`` per member with ``compresslevel=6`` on rewritten bytes.
    """
    staging, streamed, declared, original = _world(tmp_path)
    sizes = {level: len(_raw_deflate(original["Snakefile.source"], level)) for level in (1, 6, 9)}
    assert len(set(sizes.values())) == 3, f"payload does not discriminate levels: {sizes}"

    out = tmp_path / "b.zip"
    _emit_bundle_zip(staging, out, streamed=streamed, placeholders=_table(), declared_arcnames=declared)

    with zipfile.ZipFile(out) as zf:
        got = {info.filename: zf.read(info.filename) for info in zf.infolist()}
        methods = {info.filename: info.compress_type for info in zf.infolist()}

    for name in ("Snakefile.source", "harvest/model.inp", "store.zarr/zarr.json"):
        assert got[name] == _expected_rewritten(original[name]), f"{name}: placeholder missing or value left in"
        assert P.TOKEN_ANALYSIS_DIR.encode() in got[name] and ANALYSIS_DIR.encode() not in got[name]
        assert methods[name] == zipfile.ZIP_DEFLATED
        assert _stored_stream(out, name) == _raw_deflate(got[name], 6), f"{name}: not DEFLATE level 6"
    for name in ("external/weather.csv", "binary.bin"):
        assert got[name] == original[name], f"{name}: must arrive byte-identical"
        assert methods[name] == zipfile.ZIP_DEFLATED
        assert _stored_stream(out, name) == _raw_deflate(original[name], 6)
    assert got["store.zarr/c/0"] == original["store.zarr/c/0"], "a chunk keeps its bytes though they are plain text"
    assert methods["store.zarr/c/0"] == zipfile.ZIP_STORED


def test_an_emit_with_no_table_is_develops_archive(tmp_path, monkeypatch):
    """Property: with neither a table nor a scanner (the call ``_combine`` makes), every member keeps its bytes,
        non-chunks are DEFLATED at level 6, chunks are STORED, and no member is sniffed or read whole.

    Class: REGRESSION - holds on develop's body and must hold after the transform; red on the run branch's
        loop, which stores every member. The no-sniff arm is NEW CAPABILITY: the plausible wrong implementation
        always passes a transform, which reads every text member whole into memory for nothing.
    Reachability: ``_combine`` calls ``_emit_bundle_zip`` with no table; the same members with a table ARE
        sniffed (the control at the end), so the spy counts.
    Both-states anchor: the assertion is on bytes and methods, true of any correct writer.
    Invocation isolated: ``_emit_bundle_zip`` reads only its arguments; the spy wraps the emitter's sniff.
    What kills it: defaulting the table to a built one; storing everything; an unconditional transform.
    Second correct implementation: a separate legacy function for table-less callers.
    """
    sniffed: list[int] = []
    real_sniff = _emit.prefix_is_text

    def spy(prefix: bytes) -> bool:
        sniffed.append(len(prefix))
        return real_sniff(prefix)

    monkeypatch.setattr(_emit, "prefix_is_text", spy)
    staging, streamed, _declared, original = _world(tmp_path)
    out = tmp_path / "b.zip"
    _emit_bundle_zip(staging, out, streamed=streamed)
    assert sniffed == [], "no table and no scanner: no member is sniffed"
    _emit_bundle_zip(staging, tmp_path / "with_table.zip", streamed=streamed, placeholders=_table())
    assert sniffed, "control: with a table the members ARE sniffed"
    with zipfile.ZipFile(out) as zf:
        for info in zf.infolist():
            assert zf.read(info.filename) == original[info.filename]
            expected_method = zipfile.ZIP_STORED if info.filename == "store.zarr/c/0" else zipfile.ZIP_DEFLATED
            assert info.compress_type == expected_method, info.filename
    assert _stored_stream(out, "Snakefile.source") == _raw_deflate(original["Snakefile.source"], 6)


def test_entry_order_stamp_mode_and_sizes_survive_the_transform(tmp_path, monkeypatch):
    """Property: with a table and a scanner, the entry list is sorted by path PARTS, every entry carries the
        1980-01-01 stamp and mode 0o644, each entry's recorded size equals the length of the bytes it holds, and
        no ``.partial`` file is left beside the destination.

    Class: REGRESSION on the contract both parents share; the property a merge that moved the rewrite into
        ``write_zip`` could lose (a rewritten member keeps its stat size in the header).
    Reachability: ``plots-extra/x.txt`` sorts differently by parts than by joined string ('-' before '/').
    Both-states anchor: order, stamp and mode are fixed by the archive contract and independent of the rewrite.
    Invocation isolated: ``_emit_bundle_zip`` reads only its arguments.
    What kills it: sorting the joined string, dropping the fixed date, a stat-sized header for a rewritten member.
    Second correct implementation: ``writestr`` with the same ``ZipInfo`` values in the same order.
    """
    staging, streamed, declared, _original = _world(tmp_path)
    # The scanner would refuse the planted per-user paths of the declared and binary members; this arm is
    # about the archive's shape, so those members are made clean and the rewritten ones stay rewritten.
    _put(staging / "external/weather.csv", b"declared clean\n")
    _put(staging / "binary.bin", b"\x00\x01binary clean\n")
    _put(staging / "plots-extra/x.txt", b"sorted by parts, not by string\n")
    _put(staging / "plots/fig.manifest.json", b'{"plot_id": "fig"}\n')
    out = tmp_path / "b.zip"
    _emit_bundle_zip(
        staging,
        out,
        streamed=streamed,
        placeholders=_table(),
        declared_arcnames=declared,
        scanner=_scanner(monkeypatch),
    )
    with zipfile.ZipFile(out) as zf:
        names = [info.filename for info in zf.infolist()]
        assert names == sorted(names, key=lambda n: n.split("/"))
        assert names.index("plots/fig.manifest.json") < names.index("plots-extra/x.txt")
        for info in zf.infolist():
            assert info.date_time == _FIXED_DATE
            assert info.external_attr == 0o644 << 16
            assert info.file_size == len(zf.read(info.filename)), f"{info.filename}: size is not the length"
    assert not out.with_name(out.name + ".partial").exists()


# --------------------------------------------------------------------------------------
# One open per path source; a chunk's prefix is never read
# --------------------------------------------------------------------------------------


def test_each_path_source_is_opened_once_and_a_chunk_is_read_only_to_stream_it(tmp_path, monkeypatch):
    """Property: with a table and a scanner, every path source is opened exactly once, and the bytes read from a
        zarr chunk equal its size (so its sniff prefix was never read and re-read).

    Class: NEW CAPABILITY - plausible wrong implementations: sniffing by a second ``open`` (count reds), and
        sniffing the chunk's prefix before streaming it (the chunk's byte count exceeds its size).
    Reachability: the real harvest hands this function hundreds of thousands of chunk files.
    Both-states anchor: opens and bytes read are properties of how members are read with and without a table.
    Invocation isolated: the counter wraps ``Path.open`` for this test's member paths only.
    What kills it: a second open per member; reading the chunk's first bytes.
    Second correct implementation: ``os.open`` / ``io.open`` instead of ``Path.open`` for the single read.
    """
    staging, streamed, declared, original = _world(tmp_path)
    members = {p for p in staging.rglob("*") if p.is_file()} | set(streamed.values())
    ledgers = {p: {"opens": 0, "bytes_read": 0} for p in members}
    real_open = Path.open

    def counting_open(self, *args, **kwargs):
        handle = real_open(self, *args, **kwargs)
        if self in ledgers and (args[:1] == ("rb",) or kwargs.get("mode") == "rb"):
            ledgers[self]["opens"] += 1
            return _CountingHandle(handle, ledgers[self])
        return handle

    monkeypatch.setattr(Path, "open", counting_open)
    scanner = _scanner(monkeypatch)
    with contextlib.suppress(ProcessingError):  # whether the scan refuses is not this arm's question
        _emit_bundle_zip(
            staging,
            tmp_path / "b.zip",
            streamed=streamed,
            placeholders=_table(),
            declared_arcnames=declared,
            scanner=scanner,
        )
    monkeypatch.setattr(Path, "open", real_open)

    assert {p: ledger["opens"] for p, ledger in ledgers.items()} == dict.fromkeys(members, 1), "unit: opens per member"
    chunk = streamed["store.zarr/c/0"]
    assert ledgers[chunk]["bytes_read"] == len(original["store.zarr/c/0"]), "unit: bytes read from the chunk"
    # Must-fire control: a text member IS read in full to sniff, rewrite and scan it, so the counter does count.
    assert ledgers[staging / "Snakefile.source"]["bytes_read"] >= len(original["Snakefile.source"])


# --------------------------------------------------------------------------------------
# The scanner's refusal, raised after the zip closes
# --------------------------------------------------------------------------------------


def test_a_scanner_hit_is_raised_after_the_zip_closes_and_names_every_offender(tmp_path, monkeypatch):
    """Property: a producer path left in a declared input and one in an unlisted staged text member are both
        named by one ``ProcessingError``; the archive is closed and readable when it is raised; the declared
        input is archived unrewritten; a text member whose value the table holds is rewritten and not named.

    Class: REGRESSION - the run branch refuses this way; develop's body never scans. The raise happening
        AFTER the writer returns is what lets one message name every member.
    Reachability: a declared input is archived byte for byte, so a producer path in it is the hit the scan
        exists to find; a path the table does not list survives the rewrite.
    Both-states anchor: the same emit with those two members clean does not raise (the control below).
    Invocation isolated: a real scanner with no private list; ``_emit_bundle_zip`` reads only its arguments.
    What kills it: raising at the first hit, not scanning declared inputs, rewriting them, raising before the
        zip is complete.
    Second correct implementation: collecting hits and raising from ``emit_bundle`` after calling the writer.
    """
    staging = tmp_path / "staging"
    held = _with_path(HOME + "/data")  # a path the table holds: rewritten, so clean after the rewrite
    _put(staging / "a_rewritten.txt", held)
    _put(staging / "b_unlisted.txt", _with_path(UNLISTED_HOME))
    declared_bytes = _with_path(HOME + "/data", 3)  # a table value, yet a declared input is never rewritten
    _put(staging / "external/d.csv", declared_bytes)
    out = tmp_path / "b.zip"
    declared = frozenset({"external/d.csv"})

    with pytest.raises(ProcessingError) as info:
        _emit_bundle_zip(staging, out, placeholders=_table(), declared_arcnames=declared, scanner=_scanner(monkeypatch))
    text = str(info.value)
    assert "b_unlisted.txt" in text and "external/d.csv" in text
    assert "a_rewritten.txt" not in text
    with zipfile.ZipFile(out) as zf:  # closed: reads clean at the moment of the raise
        assert zf.testzip() is None
        assert zf.read("external/d.csv") == declared_bytes
        assert P.TOKEN_HOME.encode() in zf.read("a_rewritten.txt")
    assert not out.with_name(out.name + ".partial").exists()

    # Control: the same members, offenders made clean, do not raise.
    _put(staging / "b_unlisted.txt", b"clean\n")
    _put(staging / "external/d.csv", b"clean\n")
    _emit_bundle_zip(
        staging, tmp_path / "ok.zip", placeholders=_table(), declared_arcnames=declared, scanner=_scanner(monkeypatch)
    )


# --------------------------------------------------------------------------------------
# write_zip's transform contract
# --------------------------------------------------------------------------------------


class _Boom(Exception):
    """Raised by a transform to stand for any failure the emitter's own transform could meet."""


@pytest.mark.parametrize("dest_existed", [True, False], ids=["old-destination", "no-destination"])
def test_a_transform_that_raises_propagates_and_leaves_destination_and_no_partial(tmp_path, dest_existed):
    """Property: when the transform raises for one member (not the first written), the exception reaches the
        caller unchanged, an existing destination keeps its exact bytes (an absent one stays absent) and no
        ``.partial`` file remains.

    Class: NEW CAPABILITY - plausible wrong implementations: swallowing the transform's error and writing the
        member untransformed (the exception never arrives), and calling the transform outside the guarded
        region (a partial is left behind). Red on the parent: ``write_zip`` has no ``transform`` keyword.
    Reachability: a transform that reads a member whole can fail on the filesystem or on a scan.
    Both-states anchor: the destination and partial state are fixed by ``write_zip``'s documented
        failure contract, with and without an earlier destination (the two arms fail independently).
    Invocation isolated: ``write_zip`` reads only its arguments.
    What kills it: a bare ``except`` around the transform call; unlinking the destination; no cleanup.
    Second correct implementation: building the archive in memory and writing it only when complete.
    """
    files = {name: _put(tmp_path / "src" / name, f"{name}\n") for name in ("a.txt", "b.txt", "c.txt")}
    dest = tmp_path / "x.zip"
    if dest_existed:
        dest.write_bytes(b"OLD DESTINATION")

    def transform(arcname, handle):
        if arcname == "b.txt":
            raise _Boom(arcname)
        return None

    with pytest.raises(_Boom) as info:
        write_zip(files.items(), dest, transform=transform)
    assert info.value.args == ("b.txt",)
    assert (dest.read_bytes() == b"OLD DESTINATION") if dest_existed else (not dest.exists())
    assert not dest.with_name(dest.name + ".partial").exists()


def test_the_transform_contract_none_streams_from_offset_zero_and_bytes_replace_the_entry(tmp_path, monkeypatch):
    """Property: a transform that READS then returns None gets the same entry bytes as no transform (the handle is
        streamed from offset 0, not from where the transform stopped); a ``bytes`` return is the entry, sized by
        its own length; only path sources reach the transform; ``transform=None`` equals the call without the
        keyword byte for byte.

    Class: NEW CAPABILITY - plausible wrong implementations: streaming from the handle's current position
        after the transform read (a truncated entry), passing bytes or ``(ZipFile, name)`` sources to the
        transform, sizing a replaced entry by the source's stat. Red on the parent: no ``transform`` keyword.
    Reachability: the emitter's sniff reads a prefix and returns None for a binary member.
    Both-states anchor: the identity arm compares two writes of the same entries, one with the keyword and one
        without, so it turns on the writer's determinism and not on the new wording.
    Invocation isolated: ``write_zip`` reads only its arguments.
    What kills it: no seek, the wrong source kinds reaching the transform, a stat-sized replaced entry.
    Second correct implementation: re-opening the path after the transform returns None (costs a second open).
    """
    src_zip = tmp_path / "src.zip"
    with zipfile.ZipFile(src_zip, "w") as zf:
        zf.writestr("inner.txt", b"from another zip\n" * 50)
    path_members = {
        "p/text.txt": _put(tmp_path / "src" / "text.txt", _numeric_csv(rows=500)),
        "p/bin.dat": _put(tmp_path / "src" / "bin.dat", b"\x00" + bytes(range(256)) * 100),
    }

    with zipfile.ZipFile(src_zip) as source_zip:
        entries = [
            *path_members.items(),
            ("b/bytes.txt", b"plain bytes\n" * 20),
            ("z/inner.txt", (source_zip, "inner.txt")),
        ]
        plain = write_zip(entries, tmp_path / "plain.zip")
        explicit_none = write_zip(entries, tmp_path / "none.zip", transform=None)
        seen: list[str] = []

        def reading_then_none(arcname, handle):
            seen.append(arcname)
            handle.read(100)  # consume part of the member, as a sniff does
            return None

        reading = write_zip(entries, tmp_path / "reading.zip", transform=reading_then_none)

        def replacing(arcname, handle):
            return b"R" * 7 if arcname == "p/text.txt" else None

        replaced = write_zip(entries, tmp_path / "replaced.zip", transform=replacing)

    assert plain.read_bytes() == explicit_none.read_bytes() == reading.read_bytes()
    assert sorted(seen) == ["p/bin.dat", "p/text.txt"], "only path sources reach the transform"
    with zipfile.ZipFile(replaced) as zf:
        assert zf.read("p/text.txt") == b"R" * 7
        assert zf.getinfo("p/text.txt").file_size == 7
        assert zf.getinfo("p/text.txt").compress_type == zipfile.ZIP_DEFLATED
        assert zf.read("p/bin.dat") == path_members["p/bin.dat"].read_bytes()
        assert zf.read("b/bytes.txt") == b"plain bytes\n" * 20
        assert zf.read("z/inner.txt") == b"from another zip\n" * 50

    # A replaced entry is OPENED at the replacement's size: a large file replaced by a few bytes needs no zip64
    # extra, which a stat-sized opening would add. The threshold is lowered so a 160 KB file stands in for 2 GiB.
    big = _put(tmp_path / "src" / "big.txt", _numeric_csv(rows=4000))
    assert big.stat().st_size * 1.05 > 5000
    monkeypatch.setattr(zipfile, "ZIP64_LIMIT", 5000)
    shrunk = write_zip([("big.txt", big)], tmp_path / "shrunk.zip", transform=lambda arcname, handle: b"tiny")
    raw = shrunk.read_bytes()
    name_len, extra_len = struct.unpack("<HH", raw[26:30])
    assert name_len == len(b"big.txt")
    assert extra_len == 0, "a zip64 extra in the local header means the entry was opened at the source's size"
