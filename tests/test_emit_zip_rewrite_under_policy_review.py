"""Reviewer tests for chunk 2a.11 (DR-17): the member zip writer's ``transform`` hook and its emitter caller.

The coder's file pins the headline contract (rewritten AND deflated, a chunk stored, one open per member,
the refusal after the close, a raising transform, ``transform=None``). These tests aim at what that file
leaves open, by moving the input to positions its arms do not occupy:

* a ZERO-BYTE member and a rewrite that FAILS MID-EMIT, both through ``_emit_bundle_zip`` (the run branch's
  loop and develop's body differ from the change on these, so they are red on the parent);
* a policy LEVEL other than 6 on BOTH transform answers (level 6 is zlib's default, so a writer that forgot
  to set the level passes every level-6 assertion);
* a FALSY ``bytes`` answer, call ORDER and handle STATE, a DUPLICATE arcname, a ``BaseException``, a
  NON-bytes answer, the refusal message's ORDER, and a rewrite that CROSSES the zip64 limit upward.

SOLVER-FREE: every test builds its members under ``tmp_path`` and calls ``write_zip`` or
``_emit_bundle_zip`` directly. None requests a ``synth_*`` fixture, compiles, or runs a solver.

Producer-looking values are BUILT FROM FRAGMENTS, never written as literals: this file is itself scanned
by the repository's anonymization guard. Expected bytes come from ``zlib``, ``zipfile`` and plain string
replacement, never from the writer under test.
"""

from __future__ import annotations

import array
import random
import warnings
import zipfile
import zlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from hhemt.bundle import _emit
from hhemt.bundle import _placeholders as P
from hhemt.bundle._archive import CompressionPolicy, write_zip
from hhemt.bundle._emit import _emit_bundle_zip
from hhemt.bundle._reprex_gate import ZeroUserInfoScanner
from hhemt.exceptions import ProcessingError

LOGIN = "dor" + "othea"
HOME = "/ho" + "me/" + LOGIN
UNLISTED_HOME = "/ho" + "me/" + "someoneelse" + "/x"  # a per-user path the table does not hold


def _table() -> P.PlaceholderTable:
    return P.PlaceholderTable.from_values({P.TOKEN_HOME: [HOME]})


def _csv(seed: int, rows: int = 3000) -> bytes:
    """Gaussian numbers as CSV text: its deflate output differs at zlib levels 1, 6 and 9."""
    rng = random.Random(seed)
    return "\n".join(",".join(f"{rng.gauss(0, 1):.3f}" for _ in range(6)) for _ in range(rows)).encode()


def _put(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _raw_deflate(payload: bytes, level: int) -> bytes:
    compressor = zlib.compressobj(level, zlib.DEFLATED, -15)
    return compressor.compress(payload) + compressor.flush()


def _entry_stream(zip_path: Path, name: str) -> bytes:
    """The compressed bytes of entry ``name`` exactly as they sit in the archive file."""
    raw = zip_path.read_bytes()
    with zipfile.ZipFile(zip_path) as zf:
        info = zf.getinfo(name)
    offset = info.header_offset
    name_len = int.from_bytes(raw[offset + 26 : offset + 28], "little")
    extra_len = int.from_bytes(raw[offset + 28 : offset + 30], "little")
    start = offset + 30 + name_len + extra_len
    return raw[start : start + info.compress_size]


def _local_extra_len(zip_path: Path, name: str) -> int:
    raw = zip_path.read_bytes()
    with zipfile.ZipFile(zip_path) as zf:
        offset = zf.getinfo(name).header_offset
    return int.from_bytes(raw[offset + 28 : offset + 30], "little")


def _scanner(monkeypatch) -> ZeroUserInfoScanner:
    monkeypatch.delenv("HHEMT_REPREX_BLOCKLIST", raising=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ZeroUserInfoScanner.for_emit(SimpleNamespace())


def _siblings(directory: Path) -> set[str]:
    return {entry.name for entry in directory.iterdir()}


# --------------------------------------------------------------------------------------
# Through _emit_bundle_zip: red on the parent commit by behaviour
# --------------------------------------------------------------------------------------


def test_a_zero_byte_text_member_is_an_empty_deflated_entry_through_the_emit(tmp_path):
    """Property: an EMPTY staged member, emitted with a table, arrives empty and DEFLATED like every other
        non-chunk member; its neighbour with a producer path is rewritten.

    Input class: zero-byte members (the sniff reads b"", which is text; the rewrite and the scan see b"").
    Class: REGRESSION - red on 075063f4, where the emitter stores every member (compress_type 0 for the
        empty one), output pasted in the reviewer's file.
    Reachability: a staged empty file (an empty `.gitkeep`-style marker, an empty log) reaches the transform
        with an empty prefix; nothing excludes it.
    Both-states anchor: the assertions turn on the entry's compression method and its read-back length, both
        defined by the archive format and by develop's policy, not on any wording this change introduced.
    Invocation isolated: not isolatable beyond printing ``hhemt.__file__``; discrimination shown instead - a
        transform that answers None for an empty prefix, or a writer that stores empty files, changes the
        method this test reads.
    What kills it: treating b"" as 'nothing to do' and bypassing the policy (STORED), or answering a
        truthiness check that drops the empty entry.
    Second correct implementation: ``ZipFile.writestr("name", b"", ZIP_DEFLATED)`` for every empty member.
    """
    staging = tmp_path / "staging"
    _put(staging / "empty.log", b"")
    _put(staging / "paths.txt", f"home {HOME}/x\n".encode())
    out = tmp_path / "b.zip"
    _emit_bundle_zip(staging, out, placeholders=_table())

    with zipfile.ZipFile(out) as zf:
        assert zf.testzip() is None
        empty = zf.getinfo("empty.log")
        assert zf.read("empty.log") == b"" and empty.file_size == 0
        assert empty.compress_type == zipfile.ZIP_DEFLATED, "an empty member must follow the policy, not be stored"
        assert zf.getinfo("paths.txt").compress_type == zipfile.ZIP_DEFLATED
        assert zf.read("paths.txt") == b"home " + P.TOKEN_HOME.encode() + b"/x\n"


@pytest.mark.parametrize("dest_existed", [True, False], ids=["old-destination", "no-destination"])
def test_a_rewrite_that_fails_midway_through_the_emit_leaves_the_destination_and_no_partial(
    tmp_path, monkeypatch, dest_existed
):
    """Property: when the rewrite of the SECOND member raises, ``_emit_bundle_zip`` propagates that exception,
        leaves an existing destination byte-identical (or absent if it was absent), and leaves no sibling file
        of any name in the destination's directory.

    Input class: a failure raised by the emitter's OWN transform body (the real ``rewrite_producer_paths``
        replaced by one that raises), rather than by a synthetic transform handed to ``write_zip`` directly.
    Class: REGRESSION - red on 075063f4, where the archive is opened for writing in place: an existing
        destination is truncated by the failed emit and a partial archive is left behind.
    Reachability: the rewrite runs member by member inside the zip writer; any member whose bytes make the
        rewrite raise (a regex failure, a decode problem, memory) reaches this branch after earlier entries
        were written.
    Both-states anchor: the assertions turn on the destination's bytes and the directory listing, which hold
        in any correct implementation whatever the exception text.
    Invocation isolated: not isolatable beyond printing ``hhemt.__file__``; discrimination shown instead - a
        writer that opens the destination directly, or one that does not clean its partial, reds it.
    What kills it: writing to the destination in place; leaving ``dest.partial`` on an exception.
    Second correct implementation: build the archive in a ``tempfile`` in the same directory and
        ``os.replace`` it on success, deleting it in a ``finally``.
    """
    staging = tmp_path / "staging"
    for name in ("a.txt", "b.txt", "c.txt"):
        _put(staging / name, f"{name} {HOME}/x\n".encode())
    out = tmp_path / "out" / "bundle.zip"
    out.parent.mkdir()
    if dest_existed:
        out.write_bytes(b"PRIOR-ARCHIVE-BYTES")

    calls = {"n": 0}
    real = _emit.rewrite_producer_paths

    def failing(data: bytes) -> bytes:
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("rewrite exploded on the second member")
        return real(data)

    monkeypatch.setattr(_emit, "rewrite_producer_paths", failing)
    with pytest.raises(RuntimeError, match="second member"):
        _emit_bundle_zip(staging, out, placeholders=_table())

    assert calls["n"] == 2, "the failure must have been raised from inside the emit, mid-archive"
    if dest_existed:
        assert out.read_bytes() == b"PRIOR-ARCHIVE-BYTES"
        assert _siblings(out.parent) == {"bundle.zip"}
    else:
        assert _siblings(out.parent) == set()


def test_the_callers_partial_name_gains_no_second_partial_beside_it(tmp_path):
    """Property: ``emit_bundle`` hands ``_emit_bundle_zip`` its own ``.partial`` name; on success the directory
        holds exactly that one file, so the writer's inner temporary is not left next to it.

    Input class: a destination name that already ends in ``.partial``.
    Class: REGRESSION (held) - green on the parent and on the change; it pins that the writer's own
        temporary (``dest`` plus ``.partial`` again) is renamed away, so nothing but the caller's name stays.
    Reachability: ``emit_bundle`` builds ``.{name}.partial`` and passes it as ``output_path``.
    Both-states anchor: the assertion is a directory listing, true of any correct writer.
    Invocation isolated: not isolatable beyond printing ``hhemt.__file__``; discrimination shown instead - a
        writer that copies its temporary to the destination and leaves the temporary behind adds a second
        file to the listing.
    What kills it: a writer that copies instead of renaming its temporary (named, not run as a mutation).
    Second correct implementation: write straight to the given name (the parent's behaviour).
    """
    staging = tmp_path / "staging"
    _put(staging / "a.txt", b"hello\n")
    out = tmp_path / ".bundle.zip.partial"
    _emit_bundle_zip(staging, out, placeholders=_table())
    assert _siblings(tmp_path) == {"staging", ".bundle.zip.partial"}


# --------------------------------------------------------------------------------------
# write_zip(transform=...) directly: positions the coder's arms do not occupy
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("level", [1, 9])
def test_the_policy_level_is_honoured_on_both_transform_answers(tmp_path, level):
    """Property: with a policy whose level is not 6, a path entry whose transform answers None AND a path
        entry whose transform answers bytes are both deflated at THAT level; a plain bytes source is too.

    Input class: policy level (1 and 9) x transform answer (None, bytes) - level 6 is zlib's default, so a
        writer that never sets the entry's level passes every level-6 assertion.
    Class: NEW CAPABILITY - plausible wrong implementation: `_entry_info` (or the new writer) leaves
        ``_compresslevel`` unset, so zipfile deflates at its default (6). Red under it, run on a scratch copy.
    Reachability: ``write_zip`` takes any ``CompressionPolicy``; `_emit_bundle_zip` today passes MODERATE
        (level 6) only, so this is reachable by a direct caller (publish and combine pass their own policy).
    Both-states anchor: the assertion turns on the raw deflate stream against ``zlib`` at the stated level.
    Invocation isolated: not isolatable beyond printing ``hhemt.__file__``; discrimination shown instead - the
        payload's deflate output at levels 1, 6 and 9 differs (asserted first).
    What kills it: dropping ``info._compresslevel = level`` from the entry header for either answer.
    Second correct implementation: ``zf.writestr(info, data, compresslevel=level)`` per entry.
    """
    kept = _csv(1)
    replaced_source = _csv(2)
    replacement = _csv(3)
    for payload in (kept, replacement):
        sizes = {lvl: len(_raw_deflate(payload, lvl)) for lvl in (1, 6, 9)}
        assert len(set(sizes.values())) == 3, f"payload does not discriminate levels: {sizes}"

    a = _put(tmp_path / "src" / "a.csv", kept)
    b = _put(tmp_path / "src" / "b.csv", replaced_source)
    out = tmp_path / "o.zip"
    policy = CompressionPolicy(level=level, store_zarr_chunks=True)
    write_zip(
        [("a.csv", a), ("b.csv", b), ("c.csv", replacement)],
        out,
        policy=policy,
        transform=lambda name, handle: replacement if name == "b.csv" else None,
    )
    assert _entry_stream(out, "a.csv") == _raw_deflate(kept, level), "None answer: wrong level"
    assert _entry_stream(out, "b.csv") == _raw_deflate(replacement, level), "bytes answer: wrong level"
    assert _entry_stream(out, "c.csv") == _raw_deflate(replacement, level), "bytes source: wrong level"
    with zipfile.ZipFile(out) as zf:
        assert zf.read("a.csv") == kept and zf.read("b.csv") == replacement


def test_an_empty_bytes_answer_is_an_empty_entry_and_a_bytearray_answer_is_its_content(tmp_path):
    """Property: a transform answering ``b""`` archives an EMPTY entry (it does not fall back to the file),
        and a ``bytearray`` answer archives its content.

    Input class: falsy-but-not-None answers.
    Class: NEW CAPABILITY - plausible wrong implementation: `if not replacement:` in place of
        `if replacement is None:` streams the original file for an empty answer. Red under it, run on a copy.
    Reachability: a rewrite of a member that reduces it to nothing, or an emitter answering with a mutable
        buffer, reaches both arms; DR-17 item 2 defines `None` and `bytes` as the two answers.
    Both-states anchor: the assertion is the entry's read-back content and size.
    What kills it: a truthiness test on the answer.
    Second correct implementation: compare ``replacement is None`` explicitly.
    """
    src = _put(tmp_path / "src" / "x.txt", b"original content that must not survive\n")
    src2 = _put(tmp_path / "src" / "y.txt", b"y original\n")
    out = tmp_path / "o.zip"
    answers = {"x.txt": b"", "y.txt": bytearray(b"y replaced\n")}
    write_zip([("x.txt", src), ("y.txt", src2)], out, transform=lambda name, handle: answers[name])
    with zipfile.ZipFile(out) as zf:
        assert zf.read("x.txt") == b"" and zf.getinfo("x.txt").file_size == 0
        assert zf.read("y.txt") == b"y replaced\n" and zf.getinfo("y.txt").file_size == len(b"y replaced\n")
        assert zf.testzip() is None


def test_the_transform_runs_once_per_path_source_in_archive_order_on_a_fresh_handle(tmp_path):
    """Property: the transform is called exactly once for each PATH source, in the writer's archive order
        (path-part order, not plain string order), never for bytes or zip-entry sources, and each call's handle
        is a readable, non-writable binary handle positioned at offset 0.

    Input class: names that order differently by parts and by string ("a/b" before "a-b/x" by parts), mixed
        with a bytes source and a zip-entry source.
    Class: NEW CAPABILITY - plausible wrong implementation: calling the transform from a pre-pass in dict
        order, calling it twice, or handing it a handle that a prior read advanced. Red under each.
    Reachability: the emitter feeds ``write_zip`` an arbitrary mix of staged and streamed names.
    Both-states anchor: the oracle order is ``sorted(names, key=lambda n: n.split("/"))``, computed here.
    Invocation isolated: not isolatable beyond printing ``hhemt.__file__``; discrimination shown instead - a
        pre-pass in insertion order, or a second call, changes the recorded list.
    What kills it: sorting by string; two calls per member; a handle left at a non-zero offset.
    Second correct implementation: collect the path sources, sort them, and call the transform in that order.
    """
    names = ["z.txt", "a-b/x.txt", "a/b.txt", "a.txt", "m/n/o.txt"]
    paths = {n: _put(tmp_path / "src" / n, f"content of {n}\n".encode()) for n in names}
    inner = tmp_path / "inner.zip"
    with zipfile.ZipFile(inner, "w") as zf:
        zf.writestr("from_zip.txt", b"zip entry")
    calls: list[tuple[str, int, bool, bool]] = []

    def transform(name, handle):
        calls.append((name, handle.tell(), handle.readable(), handle.writable()))
        handle.read(5)  # advance it; the writer must rewind
        return None

    out = tmp_path / "o.zip"
    with zipfile.ZipFile(inner) as inner_zf:
        entries = [(n, paths[n]) for n in names] + [("bytes.bin", b"raw"), ("from_zip.txt", (inner_zf, "from_zip.txt"))]
        write_zip(entries, out, transform=transform)

    oracle = sorted(names, key=lambda n: n.split("/"))
    assert [c[0] for c in calls] == oracle
    assert all(c[1] == 0 and c[2] is True and c[3] is False for c in calls), calls
    with zipfile.ZipFile(out) as zf:
        for n in names:
            assert zf.read(n) == f"content of {n}\n".encode(), f"{n}: not streamed whole from offset 0"
        assert zf.read("bytes.bin") == b"raw" and zf.read("from_zip.txt") == b"zip entry"


def test_a_duplicate_arcname_refuses_before_any_transform_call_and_touches_nothing(tmp_path):
    """Property: two entries with one arcname raise ``ValueError`` BEFORE the transform runs for any member,
        leave an existing destination byte-identical, and leave no partial.

    Input class: a duplicate that the emitter cannot produce (its entries are a dict) but ``write_zip``'s
        other callers can.
    Class: NEW CAPABILITY - plausible wrong implementation: checking duplicates lazily, inside the entry
        loop, so the transform has already read and recorded earlier members when the duplicate is met.
    Reachability: ``write_zip`` accepts an ``Iterable`` of pairs; the duplicate rule is its documented
        contract and the transform is a new consumer of everything before the check.
    Both-states anchor: the assertions are the exception type, the call count and the directory contents.
    Invocation isolated: not isolatable beyond printing ``hhemt.__file__``; discrimination shown instead - a
        check moved into the loop records transform calls before it raises.
    What kills it: moving the duplicate check after the first entry.
    Second correct implementation: build a ``set`` of names first, as the writer does today.
    """
    a = _put(tmp_path / "src" / "a.txt", b"a")
    b = _put(tmp_path / "src" / "b.txt", b"b")
    out = tmp_path / "out" / "o.zip"
    out.parent.mkdir()
    out.write_bytes(b"PRIOR")
    seen: list[str] = []
    with pytest.raises(ValueError, match="more than once"):
        write_zip(
            [("a.txt", a), ("b.txt", b), ("a.txt", b)],
            out,
            transform=lambda name, handle: seen.append(name),
        )
    assert seen == []
    assert out.read_bytes() == b"PRIOR" and _siblings(out.parent) == {"o.zip"}


def test_a_base_exception_in_the_transform_cleans_up_and_keeps_the_destination(tmp_path):
    """Property: a ``KeyboardInterrupt`` raised by the transform propagates as itself, removes the partial and
        leaves an existing destination byte-identical.

    Input class: a ``BaseException`` that is not an ``Exception`` (the interrupt arm of the writer's
        ``except BaseException``), raised from the new call site rather than from a copy loop.
    Class: NEW CAPABILITY - plausible wrong implementation: wrapping the transform call in
        ``except Exception``, or moving the call outside the guarded region. Red under both.
    Reachability: the emitter runs for minutes over hundreds of thousands of members; an interrupt arrives
        inside the transform far more often than anywhere else.
    Both-states anchor: the assertions are the exception type, the destination bytes and the directory.
    Invocation isolated: not isolatable beyond printing ``hhemt.__file__``; discrimination shown instead - a
        narrowed ``except`` leaves ``o.zip.partial`` behind.
    What kills it: catching ``Exception`` only; calling the transform before the ``try``.
    Second correct implementation: a ``try/finally`` that unlinks the partial unless the rename happened.
    """
    a = _put(tmp_path / "src" / "a.txt", b"a")
    b = _put(tmp_path / "src" / "b.txt", b"b")
    out = tmp_path / "out" / "o.zip"
    out.parent.mkdir()
    out.write_bytes(b"PRIOR")

    def transform(name, handle):
        if name == "b.txt":
            raise KeyboardInterrupt
        return None

    with pytest.raises(KeyboardInterrupt):
        write_zip([("a.txt", a), ("b.txt", b)], out, transform=transform)
    assert out.read_bytes() == b"PRIOR" and _siblings(out.parent) == {"o.zip"}


@pytest.mark.parametrize(
    "answer",
    ["a string", memoryview(array.array("H", [1, 2, 3, 4]))],
    ids=["str", "multi-byte-memoryview"],
)
def test_an_answer_that_is_not_bytes_is_refused_with_a_clean_destination(tmp_path, answer):
    """Property: a transform answer that is neither ``bytes``, ``bytearray`` nor ``None`` raises
        ``TypeError`` naming the entry, and the destination and directory are as they were.

    Input class: a ``str`` (length counts characters) and a multi-byte ``memoryview`` (``len`` counts items,
        not bytes), the two answers whose ``len()`` is not their byte size.
    Scope: this CHARACTERIZES the coder's TypeError guard, which no shape row states; it constrains nothing
        the shape decides and is removable if the shape owner rules the guard out.
    Class: NEW CAPABILITY - plausible wrong implementation: no type check, so the entry's declared size is
        ``len(answer)`` and does not match the bytes written. The coder's deviation is the guard under test.
    Reachability: a transform returning ``data.decode()`` by mistake is the reachable shape.
    Both-states anchor: the assertion is the exception type, its mention of the entry, and the directory.
    Invocation isolated: not isolatable beyond printing ``hhemt.__file__``; discrimination shown instead - a
        removed guard lets the str through to zipfile, which raises a different type or writes a bad size.
    What kills it: removing the `isinstance` guard.
    Second correct implementation: ``memoryview(answer).nbytes`` for the size, writing the buffer as is.
    """
    a = _put(tmp_path / "src" / "a.txt", b"a")
    out = tmp_path / "out" / "o.zip"
    out.parent.mkdir()
    out.write_bytes(b"PRIOR")
    with pytest.raises(TypeError, match="a.txt"):
        write_zip([("a.txt", a)], out, transform=lambda name, handle: answer)
    assert out.read_bytes() == b"PRIOR" and _siblings(out.parent) == {"o.zip"}


def test_every_entry_is_consistent_when_a_rewrite_grows_one_member_and_shrinks_another(tmp_path):
    """Property: when the transform grows one member and shrinks another, each entry's recorded size and CRC
        equal its content's, and the archive passes ``testzip``.

    Input class: a replacement longer than its file and a replacement shorter than its file, in one archive.
    Class: NEW CAPABILITY - plausible wrong implementation: sizing the entry from the SOURCE file's size
        after a bytes answer (the stat size), which leaves the header size inconsistent with the content.
    Reachability: the table's tokens are shorter than most paths and longer than some (`/tmp/x` to a token).
    Both-states anchor: ``ZipInfo.file_size`` and ``ZipInfo.CRC`` against ``len`` and ``zlib.crc32``.
    Invocation isolated: not isolatable beyond printing ``hhemt.__file__``; discrimination shown instead - a
        stat-sized header makes the grown entry's declared size smaller than its content.
    What kills it: `file_size` taken from the file rather than the answer.
    Second correct implementation: ``ZipFile.writestr`` of the answer.
    """
    small = _put(tmp_path / "src" / "small.txt", b"tiny")
    big = _put(tmp_path / "src" / "big.txt", b"x" * 5000)
    grown = b"G" * 20000
    shrunk = b"s"
    answers = {"small.txt": grown, "big.txt": shrunk}
    out = tmp_path / "o.zip"
    write_zip([("small.txt", small), ("big.txt", big)], out, transform=lambda name, handle: answers[name])
    with zipfile.ZipFile(out) as zf:
        assert zf.testzip() is None
        for name, content in answers.items():
            info = zf.getinfo(name)
            assert zf.read(name) == content
            assert info.file_size == len(content) and info.CRC == zlib.crc32(content)


def test_a_rewrite_that_grows_a_small_file_across_the_zip64_limit_is_written_as_zip64(tmp_path, monkeypatch):
    """Property: a file BELOW the zip64 limit whose replacement is ABOVE it is written with a zip64 header and
        reads back whole (the coder's arm covers the opposite direction, a large file replaced by 4 bytes).

    Input class: crossing the limit upward (stat size small, answer size large), with the limit lowered to
        100 bytes so no large file is needed.
    Class: NEW CAPABILITY - plausible wrong implementation: opening the replaced entry at the source file's
        stat size, so zipfile chooses the 32-bit form and then refuses the larger write. Red under it.
    Reachability: a rewrite that expands a member (a long token for a short path) of a member near 4 GiB is
        the real shape; the lowered limit is the same branch at test scale.
    Both-states anchor: the assertion is the read-back content, the recorded size and a non-empty local
        extra field, all fixed by the zip format.
    Invocation isolated: not isolatable beyond printing ``hhemt.__file__``; discrimination shown instead - a
        stat-sized header raises ``RuntimeError`` (file size too large) when the answer is written.
    What kills it: sizing the entry from ``stat``.
    Second correct implementation: ``ZipFile.writestr`` of the answer with ``force_zip64`` set.
    """
    monkeypatch.setattr(zipfile, "ZIP64_LIMIT", 100)
    small = _put(tmp_path / "src" / "s.txt", b"0123456789")
    replacement = bytes(range(256)) * 2  # 512 bytes, well over the lowered limit
    out = tmp_path / "o.zip"
    write_zip([("s.txt", small)], out, transform=lambda name, handle: replacement)
    with zipfile.ZipFile(out) as zf:
        assert zf.read("s.txt") == replacement and zf.getinfo("s.txt").file_size == len(replacement)
    assert _local_extra_len(out, "s.txt") > 0, "a zip64 entry carries the zip64 extra field in its local header"


# --------------------------------------------------------------------------------------
# The refusal message is deterministic
# --------------------------------------------------------------------------------------


def test_the_refusal_names_offenders_in_archive_order_not_string_order(tmp_path, monkeypatch):
    """Property: when two staged members both hold an unlisted per-user path, the single refusal lists them
        in the writer's archive order ("a/b.txt" before "a-b/x.txt"), after the zip has closed.

    Input class: two offenders whose names order differently by path parts and by plain string.
    Class: REGRESSION (held) - green on the parent and the change; the scanner records in write order, so the
        message order is the writer's sort order. It reds if the writer's sort key changes.
    Reachability: any bundle with offenders in sibling directories whose names differ at "-" versus "/".
    Both-states anchor: the assertion is the relative order of two member names in the message, true of any
        implementation that fixes one deterministic order.
    Invocation isolated: not isolatable beyond printing ``hhemt.__file__``; discrimination shown instead - a
        string-ordered writer flips the two indexes.
    What kills it: sorting entries by ``arcname`` instead of ``arcname.split("/")``.
    Second correct implementation: record hits, then sort them by the same key before rendering.
    """
    staging = tmp_path / "staging"
    for name in ("a-b/x.txt", "a/b.txt"):
        _put(staging / name, f"at {UNLISTED_HOME}\n".encode())
    out = tmp_path / "o.zip"
    with pytest.raises(ProcessingError) as raised:
        _emit_bundle_zip(staging, out, scanner=_scanner(monkeypatch))
    message = str(raised.value)
    assert "a/b.txt" in message and "a-b/x.txt" in message
    assert message.index("a/b.txt") < message.index("a-b/x.txt"), message
