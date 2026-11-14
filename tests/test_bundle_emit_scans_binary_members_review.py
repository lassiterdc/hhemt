"""Reviewer tests for the emit-time scan of binary members (work package 1a, chunk 1a.1).

The coder's module (``tests/test_bundle_emit_scans_binary_members.py``) pins the headline behaviours. These
tests constrain the CLASSES of input that module samples only once, each with a differently positioned
input: every text-bearing zarr data type (the coder tests three of the six), zarr format 2, values that
sit in a later chunk, in a 0-d array or in a 2-D array, the block seam in the two reading paths the coder
does not sweep (a decompressed stream and a byte-order-mark-decoded member), and a byte-order-marked text
entry inside each container kind.

SOLVER-FREE: ``tmp_path`` only, the real ``_emit_bundle_zip`` and the real scanner, no fixture that compiles
or runs a solver. Producer-looking values are BUILT FROM FRAGMENTS so this file passes the repository's
anonymization guard.

Every test here is GREEN on the tree it was written against. Each is admitted under B4's NEW CAPABILITY
class: the plausible wrong implementation each names turns it red, shown by running the test with that
change applied in-process (the change is to what the code DOES; nothing here edits a production path).
"""

from __future__ import annotations

import codecs
import gzip
import io
import json
import tarfile
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pytest
import zarr

from hhemt.bundle import _reprex_gate as G
from hhemt.bundle._emit import _emit_bundle_zip
from hhemt.exceptions import ProcessingError


def _unix(root: str, user: str, rest: str = "x") -> str:
    """A path under a per-user root, assembled so no literal of the form appears in this file."""
    return "/" + root + "/" + user + "/" + rest


PLANTED = _unix("home", "someone", "data")
PLANTED_BYTES = PLANTED.encode()


@pytest.fixture(autouse=True)
def _no_private_list(monkeypatch):
    """Layer (c) off: every arm here is decided by layer (a), the per-user path class."""
    monkeypatch.delenv("HHEMT_REPREX_BLOCKLIST", raising=False)


def _emit(staging: Path, tmp_path: Path, tag: str) -> str | None:
    """Run the emit scan over ``staging``; the refusal message, or None when the emit passed."""
    try:
        _emit_bundle_zip(staging, tmp_path / f"{tag}.zip", scanner=G.ZeroUserInfoScanner.for_consume())
    except ProcessingError as exc:
        assert PLANTED not in str(exc), "a refusal names the member and class, never the value"
        return str(exc)
    return None


def _quiet(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # zarr warns that some data types have no v3 specification
        return fn()


def _data_type_name(array_dir: Path) -> str:
    """The v3 ``data_type`` name the array's metadata document records."""
    raw = json.loads((array_dir / "zarr.json").read_text())["data_type"]
    return raw["name"] if isinstance(raw, dict) else raw


# ----------------------------------------------------------------------------------------------
# R-1: every text-bearing data type of the table decodes, not only the three the coder samples
# ----------------------------------------------------------------------------------------------


def _build_string(d: Path, values: list[str]) -> None:
    arr = zarr.create_array(store=str(d), shape=(len(values),), dtype="str", zarr_format=3)
    arr[:] = np.array(values, dtype=object)


def _build_utf32(d: Path, values: list[str]) -> None:
    zarr.create_array(store=str(d), data=np.array(values, dtype="U60"), zarr_format=3)


def _build_null_terminated(d: Path, values: list[str]) -> None:
    zarr.create_array(store=str(d), data=np.array([v.encode() for v in values], dtype="S60"), zarr_format=3)


def _build_variable_length_bytes(d: Path, values: list[str]) -> None:
    from zarr.core.dtype import VariableLengthBytes

    arr = zarr.create_array(store=str(d), shape=(len(values),), dtype=VariableLengthBytes(), zarr_format=3)
    arr[:] = np.array([v.encode() for v in values], dtype=object)


def _build_raw_bytes(d: Path, values: list[str]) -> None:
    arr = np.zeros(len(values), dtype="V64")
    for i, v in enumerate(values):
        arr[i] = np.void(v.encode().ljust(64, b"\0"))
    zarr.create_array(store=str(d), data=arr, zarr_format=3)


def _build_structured(d: Path, values: list[str]) -> None:
    rows = np.zeros(len(values), dtype=[("n", "i4"), ("label", "U60")])
    for i, v in enumerate(values):
        rows[i] = (i, v)
    zarr.create_array(store=str(d), data=rows, zarr_format=3)


#: (zarr registry name the array must record, builder). The six text-bearing rows of the data-type
#: table in ``_reprex_gate`` (``ZARR_TEXT_DATA_TYPES``), by name; the name is read back from the written
#: metadata so no arm can silently build a different type than the row it is named for.
_DECODE_ROWS = [
    ("string", _build_string),
    ("fixed_length_utf32", _build_utf32),
    ("null_terminated_bytes", _build_null_terminated),
    ("variable_length_bytes", _build_variable_length_bytes),
    ("raw_bytes", _build_raw_bytes),
    ("structured", _build_structured),
]


@pytest.mark.parametrize(("row", "build"), _DECODE_ROWS, ids=[r[0] for r in _DECODE_ROWS])
def test_every_text_bearing_zarr_data_type_refuses_a_planted_path_and_passes_benign_values(tmp_path, row, build):
    """Property: a top-level array of EACH text-bearing data type of the table refuses when a value is a
        per-user path and passes when every value is benign.

    Class: NEW CAPABILITY. Plausible wrong implementation: a decode row filed under the skip set (or left
        out of the decode set), so that one text type is treated as numeric and its values are never read.
        Red under it for ``variable_length_bytes``, ``raw_bytes`` and ``null_terminated_bytes`` (one failing
        arm each, the other 20 of this module's arms green); the coder's module samples ``string``,
        ``fixed_length_utf32`` and a structured array only and stays green under all three.
    Reachability: ``{row}`` is a data type a toolkit store can carry; xarray writes ``string``, a user's
        NetCDF-to-zarr conversion writes the byte types. Artifact: ``ZARR_TEXT_DATA_TYPES``, six rows.
    Both-states anchor: the assertion turns on refuse-versus-pass for the same array shape with a path
        and without one, a property present before and after any change to the table.
    Invocation: not isolatable from this tree; discrimination shown instead, by the named change to what the
        code DOES (the row's name moved to the skip set) turning the planted arm to a pass.
    What kills it: any change that classes one of the six text types as skippable.
    Second correct implementation: one that decodes EVERY array regardless of type and filters by the
        scanner's patterns alone (slower, same verdicts); it passes both arms.
    """
    planted = tmp_path / "planted"
    benign = tmp_path / "benign"
    _quiet(lambda: build(planted / "s.zarr" / "a", [PLANTED, "cpu", "gpu"]))
    _quiet(lambda: build(benign / "s.zarr" / "a", ["alpha", "cpu", "gpu"]))
    assert _data_type_name(planted / "s.zarr" / "a") == row, "the arm must build the row it is named for"
    refusal = _emit(planted, tmp_path, "planted")
    assert refusal is not None, f"a per-user path in a {row} array was carried unscanned"
    assert "s.zarr/a" in refusal
    assert _emit(benign, tmp_path, "benign") is None


# ----------------------------------------------------------------------------------------------
# R-2: zarr format 2 (.zarray) is classified by the numpy kind of its dtype string
# ----------------------------------------------------------------------------------------------


def _v2_str(d: Path, values: list[str]) -> None:
    arr = zarr.create_array(store=str(d), shape=(len(values),), dtype="str", zarr_format=2)
    arr[:] = np.array(values, dtype=object)


def _v2_utf(d: Path, values: list[str]) -> None:
    zarr.create_array(store=str(d), data=np.array(values, dtype="U60"), zarr_format=2)


def _v2_bytes(d: Path, values: list[str]) -> None:
    zarr.create_array(store=str(d), data=np.array([v.encode() for v in values], dtype="S60"), zarr_format=2)


_V2_ARMS = [("object_vlen_str", _v2_str), ("U", _v2_utf), ("S", _v2_bytes)]


@pytest.mark.parametrize(("kind", "build"), _V2_ARMS, ids=[a[0] for a in _V2_ARMS])
def test_a_zarr_format_2_text_array_refuses_a_planted_path(tmp_path, kind, build):
    """Property: an array written in zarr format 2 (a ``.zarray`` metadata document, classified by the numpy
        kind of its ``dtype`` string) with a per-user path among its values refuses; the same array with
        benign values passes.

    Class: NEW CAPABILITY. Plausible wrong implementation: the v2 branch of the classifier treats the kinds
        ``U``, ``S`` and ``O`` as skippable, or never reads ``.zarray``; red under it.
    Reachability: v2 stores are what xarray writes under ``zarr_format=2`` and what every store from before
        the v3 default holds; a declared input can be one. Artifact: the classifier's v2 branch, which the
        coder's module does not exercise (``grep -c zarr_format=2`` over it returns 0).
    Both-states anchor: refuse-versus-pass on identical shapes, independent of the classifier's wording.
    Invocation: not isolatable from this tree; discrimination shown by the named change to behaviour (the
        three kinds moved to the skip set) turning the planted arm to a pass.
    What kills it: dropping a text kind from the v2 decode set.
    Second correct implementation: one that opens every v2 array through zarr and decides on the decoded
        values' numpy kind instead of the metadata string.
    """
    planted = tmp_path / "planted"
    benign = tmp_path / "benign"
    _quiet(lambda: build(planted / "s.zarr" / "a", [PLANTED, "cpu", "gpu"]))
    _quiet(lambda: build(benign / "s.zarr" / "a", ["alpha", "cpu", "gpu"]))
    assert (planted / "s.zarr" / "a" / ".zarray").is_file(), "the arm must write zarr format 2"
    assert _emit(planted, tmp_path, "planted") is not None, (
        f"a per-user path in a v2 {kind} array was carried unscanned"
    )
    assert _emit(benign, tmp_path, "benign") is None


# ----------------------------------------------------------------------------------------------
# R-3: a value anywhere in the array is read: a later chunk, a 0-d array, a 2-D array
# ----------------------------------------------------------------------------------------------


def _multi_chunk(d: Path) -> None:
    values = [f"v{i}" for i in range(100)]
    values[95] = PLANTED
    zarr.create_array(store=str(d), data=np.array(values, dtype="U60"), chunks=(10,), zarr_format=3)


def _two_d(d: Path) -> None:
    grid = np.array([[f"r{i}c{j}" for j in range(3)] for i in range(4)], dtype="U60")
    grid[3, 2] = PLANTED
    zarr.create_array(store=str(d), data=grid, chunks=(1, 3), zarr_format=3)


def _zero_d(d: Path) -> None:
    arr = zarr.create_array(store=str(d), shape=(), dtype="str", zarr_format=3)
    arr[...] = PLANTED


_SHAPES = [("later_chunk_of_ten", _multi_chunk), ("two_d_last_row", _two_d), ("zero_d", _zero_d)]


@pytest.mark.parametrize(("shape", "build"), _SHAPES, ids=[s[0] for s in _SHAPES])
def test_a_planted_path_in_a_later_chunk_a_2d_row_or_a_scalar_array_refuses(tmp_path, shape, build):
    """Property: a per-user path is found wherever it sits in the array: in the 10th chunk of ten, in the
        last row of a 2-D array read a row at a time, and as the only value of a 0-d array.

    Class: NEW CAPABILITY. Plausible wrong implementation: a decoder that reads only the first chunk along
        axis 0 (an off-by-one in the chunk loop), or that indexes a 0-d array as if it had an axis; red
        under it for the multi-chunk and 2-D arms (a first-chunk reader leaves both unrefused). The 0-d arm is red
        under a decoder that returns no text for an empty shape (the coder's module stays green under it).
    Reachability: consolidated stores chunk along axis 0 by default for long arrays, and xarray writes scalar
        coordinates as 0-d arrays. Artifact: ``decode_zarr_array_text``'s loop over ``range(0, shape[0], step)``
        and its ``if not shape`` branch.
    Both-states anchor: the planted value is absent from the raw chunk bytes of the ``fixed_length_utf32``
        arms (asserted), so a refusal can only come from decoding that array's values.
    Invocation: not isolatable from this tree; discrimination shown by the named change to behaviour (read
        the first chunk only) turning the multi-chunk and 2-D arms to passes.
    What kills it: stopping the decode after the first chunk, or mishandling an empty shape.
    Second correct implementation: one that reads the whole array in a single ``array[...]`` call; it passes.
    """
    staging = tmp_path / "staging"
    _quiet(lambda: build(staging / "s.zarr" / "a"))
    if shape != "zero_d":
        raw = b"".join(
            f.read_bytes() for f in (staging / "s.zarr" / "a").rglob("*") if f.is_file() and f.name != "zarr.json"
        )
        assert PLANTED_BYTES not in raw, "the arm must need decoding: the path is in the raw chunk bytes"
    refusal = _emit(staging, tmp_path, "planted")
    assert refusal is not None, f"a per-user path in the {shape} array was carried unscanned"
    assert "s.zarr/a" in refusal


# ----------------------------------------------------------------------------------------------
# R-4: the block seam holds in every reading path, not only the raw one
# ----------------------------------------------------------------------------------------------

_BLOCK = 4096
_STRADDLE = range(_BLOCK - 40, _BLOCK + 40)


def _raw_member(offset: int, value: str) -> tuple[str, bytes]:
    return "opaque.bin", b"\x00" + b"a" * offset + value.encode() + b" z"


def _gzip_member(offset: int, value: str) -> tuple[str, bytes]:
    return "stream.gz", gzip.compress(b"a" * offset + value.encode() + b" z")


def _bom_member(offset: int, value: str) -> tuple[str, bytes]:
    return "wide.txt", codecs.BOM_UTF16_LE + ("a" * (offset // 2) + value + " z").encode("utf-16-le")


_SEAM_PATHS = [("raw_opaque", _raw_member), ("decompressed_stream", _gzip_member), ("bom_decoded_text", _bom_member)]


@pytest.mark.parametrize(("path", "make"), _SEAM_PATHS, ids=[p[0] for p in _SEAM_PATHS])
def test_a_path_straddling_a_block_boundary_is_found_in_every_reading_path(tmp_path, monkeypatch, path, make):
    """Property: at EVERY offset that puts the planted path across the boundary between two blocks, the
        path is found, in the raw scan of an opaque member, in a decompressed stream, and in a byte-order-mark
        decoded member.

    Class: NEW CAPABILITY. Plausible wrong implementation: a path that carries no overlap between blocks;
        the coder's module sweeps only the raw path. Run in-process with the overlap removed from the
        decoded-text path only, the coder's whole module stays green (measured: 26 passed) and this test
        reds on its byte-order-mark arm.
    Reachability: a declared input larger than one block (1 MiB) reaches a seam; a UTF-16 input over 1 MiB is
        the decoded path's case. Artifact: ``_blocks`` and ``_decoded_blocks``, two separate carry loops.
    Both-states anchor: the block size is shrunk to 4096 bytes with ``raising=False`` and the assertion is
        refuse-or-pass per offset, so an implementation with another block size sees the same inputs, one
        block each, and still refuses.
    Invocation: not isolatable from this tree; discrimination shown by the named change to behaviour
        (``overlap_chars`` returning 0) turning each path's arm red.
    What kills it: dropping or shortening the carry in any one of the reading loops.
    Second correct implementation: one that holds the whole member in memory when it is under a limit, or
        reads each stream with a sliding window one pattern longer than the longest match; both refuse every offset.
    """
    monkeypatch.setattr(G, "_BLOCK_BYTES", _BLOCK, raising=False)
    missed = []
    for offset in _STRADDLE:
        name, data = make(offset, PLANTED)
        staging = tmp_path / f"s{offset}"
        staging.mkdir()
        (staging / name).write_bytes(data)
        if _emit(staging, tmp_path, f"o{offset}") is None:
            missed.append(offset)
    assert missed == [], f"{path}: the path went unfound at {len(missed)} of {len(_STRADDLE)} straddling offsets"
    benign = tmp_path / "benign"
    benign.mkdir()
    name, data = make(_BLOCK - 5, "b" * len(PLANTED))
    (benign / name).write_bytes(data)
    assert _emit(benign, tmp_path, "benign") is None


# ----------------------------------------------------------------------------------------------
# R-5: a byte-order-marked text entry INSIDE a container is decoded as a top-level member is
# ----------------------------------------------------------------------------------------------


def _wrap_zip(inner: bytes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("inner.txt", inner)
    return buf.getvalue()


def _wrap_gzip(inner: bytes) -> bytes:
    return gzip.compress(inner, mtime=0)


def _wrap_tar(inner: bytes) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        info = tarfile.TarInfo("inner.txt")
        info.size = len(inner)
        tf.addfile(info, io.BytesIO(inner))
    return buf.getvalue()


_WRAPS = [("zip", _wrap_zip), ("gzip", _wrap_gzip), ("tar", _wrap_tar)]
_BOMS = [("utf16le", codecs.BOM_UTF16_LE, "utf-16-le"), ("utf32be", codecs.BOM_UTF32_BE, "utf-32-be")]


@pytest.mark.parametrize("wrap_name", [w[0] for w in _WRAPS])
@pytest.mark.parametrize("enc", [b[0] for b in _BOMS])
def test_a_byte_order_marked_text_entry_inside_a_container_refuses(tmp_path, wrap_name, enc):
    """Property: a UTF-16 or UTF-32 text file with a byte-order mark and a per-user path, carried as the
        entry of a zip, a tar or a gzip stream, refuses; with a benign body it passes.

    Class: NEW CAPABILITY. Plausible wrong implementation: the byte-order-mark decode applied to the top-level
        member only (a depth-0 guard, or the branch placed ahead of the container recursion); red under it
        for all six arms, while the coder's module (top-level members only for this table) stays green.
    Reachability: a vendor archive of UTF-16 text exports is the realistic carrier. Artifact: ``scan_binary_member``,
        which applies the byte-order-mark table to every entry it recurses into.
    Both-states anchor: the planted path's UTF-8 bytes are asserted ABSENT from the container's raw bytes, so
        only decoding through the container and then the mark can find it.
    Invocation: not isolatable from this tree; discrimination shown by the named change to behaviour
        (decode only when ``_depth == 0``) turning the six planted arms to passes.
    What kills it: restricting the mark table to depth 0.
    Second correct implementation: one that flattens a container to a temporary tree of members and applies
        the member rules to each file; it refuses all six.
    """
    _, mark, codec = next(b for b in _BOMS if b[0] == enc)
    wrap = dict(_WRAPS)[wrap_name]
    for body, expect_refusal in (("x " + PLANTED + " y", True), ("x alpha y", False)):
        staging = tmp_path / f"s_{expect_refusal}"
        staging.mkdir()
        data = wrap(mark + body.encode(codec))
        assert PLANTED_BYTES not in data, "the arm must need decoding: the path is in the container's raw bytes"
        (staging / f"member.{wrap_name}").write_bytes(data)
        refusal = _emit(staging, tmp_path, f"t_{expect_refusal}")
        assert (refusal is not None) is expect_refusal, f"{wrap_name}/{enc}: expected refusal={expect_refusal}"


# ----------------------------------------------------------------------------------------------
# R-6: a byte-order-marked member is decoded wherever its first NUL byte sits relative to the sniff window
# ----------------------------------------------------------------------------------------------

_SNIFF_CHARS = 8192 // 2  # CJK characters before the first ASCII character, in UTF-16 code units
_NUL_POSITIONS = [10, 2000, _SNIFF_CHARS - 20, _SNIFF_CHARS - 2, _SNIFF_CHARS - 1, _SNIFF_CHARS, _SNIFF_CHARS + 1, 6000]


@pytest.mark.parametrize("encoding", ["utf-16-le", "utf-16-be"])
def test_a_marked_utf16_member_is_decoded_wherever_its_first_nul_byte_sits(tmp_path, encoding):
    """Property: a UTF-16 member with a byte-order mark and a per-user path refuses whether its first NUL
        byte lies well inside the 8192-byte sniff window, on either side of its edge, or beyond it (a prefix of
        CJK characters holds no NUL); the same member with a benign tail passes.

    Class: NEW CAPABILITY. Plausible wrong implementation: the mark honoured only when the sniff would also have
        classed the member binary (the check placed after the NUL sniff, or restricted to a shorter window); red
        under it at every position whose first NUL lies beyond the window. The coder's module pins one position.
    Reachability: any marked UTF-16 file of non-Latin text with its first ASCII character late in the file.
        Artifact: ``rewrite_and_scan`` in ``_emit.py``, which tests ``has_byte_order_mark`` before ``prefix_is_text``.
    Both-states anchor: refuse-versus-pass for identical bytes except the path, independent of any wording; the
        arm builds the first-NUL position and asserts it relative to the window, so the arm cannot drift.
    Invocation: not isolatable from this tree; discrimination shown by the named change to behaviour (the mark
        honoured only when a NUL lies in the first 4096 bytes) turning the late-NUL positions to passes.
    What kills it: any ordering or window that lets the NUL sniff decide before the mark is tested.
    Second correct implementation: one that decodes every member whose first bytes are a mark and also every
        member the sniff classes binary, in either order; it refuses at every position.
    """
    mark = codecs.BOM_UTF16_LE if encoding == "utf-16-le" else codecs.BOM_UTF16_BE
    inside_window: set[bool] = set()
    for nul_at in _NUL_POSITIONS:
        for tail, expect in ((PLANTED, True), ("benign", False)):
            text = "あ" * nul_at + " " + "あ" * (6000 - min(nul_at, 6000)) + " " + tail
            data = mark + text.encode(encoding)
            first_nul = data.find(b"\x00")
            inside_window.add(first_nul < 8192)
            staging = tmp_path / f"s_{nul_at}_{expect}"
            staging.mkdir()
            (staging / "in.txt").write_bytes(data)
            assert (_emit(staging, tmp_path, f"t_{nul_at}_{expect}") is not None) is expect, (nul_at, expect)
    assert inside_window == {True, False}, "the arms must put the first NUL on both sides of the sniff window"


# ----------------------------------------------------------------------------------------------
# Chunk 1a.2, consume side: ``_scan_zero_user_info`` (the informational scan ``reprex()`` runs over an unpacked
# bundle root). Added by the chunk 1a.2 reviewer round. C-1 and C-2 are RED on the tree they were written against;
# the module docstring's "GREEN" sentence describes the chunk 1a.1 tests above and is left as written.
# ----------------------------------------------------------------------------------------------


def _consume(root: Path) -> list[str]:
    """The consume-side leak list for ``root``, with the scan's unset-carrier warning silenced."""
    from hhemt.bundle._reprex import _scan_zero_user_info

    return _quiet(lambda: _scan_zero_user_info(root))


def _write_zarr_metadata(directory: Path, name: str, body: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(body)


def _nested_zip(depth: int) -> bytes:
    data = b"inner " + PLANTED_BYTES
    for _ in range(depth):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("e", data)
        data = buf.getvalue()
    return data


def _refuser_corrupt_zip(d: Path) -> None:
    (d / "m_refuser.bin").write_bytes(b"PK\x03\x04" + b"\x00" * 30)


def _refuser_corrupt_gzip(d: Path) -> None:
    (d / "m_refuser.bin").write_bytes(b"\x1f\x8b\x08\x00" + b"\x00" * 6 + b"junk")


def _refuser_corrupt_bzip2(d: Path) -> None:
    (d / "m_refuser.bin").write_bytes(b"BZh91AY&SY" + b"\x00\x01junk")


def _refuser_corrupt_xz(d: Path) -> None:
    (d / "m_refuser.bin").write_bytes(b"\xfd7zXZ\x00" + b"\x00" * 20)


def _refuser_corrupt_tar(d: Path) -> None:
    (d / "m_refuser.bin").write_bytes(b"x" * 257 + b"ustar" + b"\x00" * 300)


def _refuser_too_deep(d: Path) -> None:
    (d / "m_refuser.bin").write_bytes(_nested_zip(G._MAX_CONTAINER_DEPTH + 1))


def _refuser_unknown_zarr_data_type(d: Path) -> None:
    body = json.dumps({"node_type": "array", "zarr_format": 3, "data_type": "a_type_zarr_does_not_have_yet"})
    _write_zarr_metadata(d / "m_arr", "zarr.json", body)


def _refuser_unreadable_zarr_metadata(d: Path) -> None:
    _write_zarr_metadata(d / "m_arr", "zarr.json", "{{{")


_REFUSERS = [
    ("corrupt_zip", _refuser_corrupt_zip),
    ("corrupt_gzip", _refuser_corrupt_gzip),
    ("corrupt_bzip2", _refuser_corrupt_bzip2),
    ("corrupt_xz", _refuser_corrupt_xz),
    ("corrupt_tar", _refuser_corrupt_tar),
    ("container_nested_past_the_bound", _refuser_too_deep),
    ("zarr_data_type_not_in_the_table", _refuser_unknown_zarr_data_type),
    ("unreadable_zarr_metadata_document", _refuser_unreadable_zarr_metadata),
]


@pytest.mark.parametrize("side", ["text_member_sorts_first", "text_member_sorts_last"])
@pytest.mark.parametrize(("refuser", "build"), _REFUSERS, ids=[r[0] for r in _REFUSERS])
def test_a_member_the_scan_cannot_read_does_not_hide_a_leak_in_a_text_member(tmp_path, refuser, build, side):
    """Property: a bundle root that holds one member the scan refuses to read (a container that will not open, a
        container nested past the bound, a zarr data type or metadata document it cannot classify) AND a text member
        carrying a per-user path still reports the text member's leak; the leak the pre-change gate gave for that text
        member is not lost to the refusal.

    Class: REGRESSION - red on this tree (chunk 1a.2 as the coder left it, uncommitted), green on the pre-change
        loop at ``8ab54844``, which could not refuse and reported the text member in all 16 arms. The scan holds its
        hits in the scanner until ``raise_if_hits`` at the end of the loop, so a ``ProcessingError`` raised by the
        helper mid-loop leaves the caller only the refusal's reason (``_scan_zero_user_info`` returns
        ``[exc.reason]``) and drops every hit already recorded and every member not yet read. Measured by the
        reviewer over these 16 arms: 16 of 16 return a list that does not name the text member.
    Reachability: ``reprex()`` runs this scan on every unpacked bundle root (``_reprex.py``,
        ``_scan_zero_user_info``); a bundle from another researcher can hold a truncated zip or an archive whose
        header is intact and whose body is damaged. The shape states the objective this serves: iteration 8 header,
        "the consume-side scan keeps every verdict it gave before this run".
    Both-states anchor: the assertion turns on whether the text member's NAME appears in the returned list - a
        property of the pre-change loop and of any correct fix - and on no wording of the refusal. The text-alone
        control in the test shows the instrument fires on the same text member with the refusing member absent.
    Invocation: not isolatable from this tree; discrimination shown by the named change to what the code DOES -
        the pre-change text pass run before the new passes and its hits raised - turning all 16 arms green.
    What kills it: any consume-side loop that lets a refusal raised for one member discard the hits recorded for
        another.
    Second correct implementation: one that catches the refusal per member, records its reason as an entry beside
        the hits and carries on, so the list holds both the refusal and the text leak; it passes (run in-process
        for the 12 container arms; the 4 zarr arms need the same catch in the classifier loop), and so does one
        that reads the text members first and raises their hits ahead of the binary pass (run in-process: all 16).
    """
    text_name = "a_notes.txt" if side == "text_member_sorts_first" else "z_notes.txt"
    alone = tmp_path / "text_alone"
    alone.mkdir()
    (alone / text_name).write_text("see " + PLANTED + " for the data\n")
    assert any(text_name in leak for leak in _consume(alone)), "the instrument must fire on the text member alone"
    refuser_alone = tmp_path / "refuser_alone"
    refuser_alone.mkdir()
    build(refuser_alone)
    assert _consume(refuser_alone), "the refusing member must be reported at all"
    both = tmp_path / "both"
    both.mkdir()
    build(both)
    (both / text_name).write_text("see " + PLANTED + " for the data\n")
    leaks = _consume(both)
    assert all(PLANTED not in leak for leak in leaks), "a leak names the member and class, never the value"
    assert any(text_name in leak for leak in leaks), (
        f"the leak in {text_name} was lost to the refusal of the {refuser} member; returned {len(leaks)} entry(ies)"
    )


#: (id, document name, body). The JSON value types are a CLOSED domain (object, array, string, number, boolean,
#: null); an object is the only valid document root, so the other five are the root arms, and the field arms put
#: each non-valid JSON type where the classifier reads a v3 ``data_type`` or a v2 ``dtype``.
_MALFORMED_ZARR_METADATA = [
    ("zarr_json_root_array", "zarr.json", "[]"),
    ("zarr_json_root_string", "zarr.json", '"x"'),
    ("zarr_json_root_number", "zarr.json", "3"),
    ("zarr_json_root_true", "zarr.json", "true"),
    ("zarr_json_root_null", "zarr.json", "null"),
    ("zarray_root_array", ".zarray", "[]"),
    ("zarray_root_string", ".zarray", '"x"'),
    ("zarray_root_number", ".zarray", "3"),
    ("zarray_root_true", ".zarray", "true"),
    ("zarray_root_null", ".zarray", "null"),
    ("v3_data_type_array", "zarr.json", json.dumps({"node_type": "array", "zarr_format": 3, "data_type": ["a"]})),
    (
        "v3_data_type_name_array",
        "zarr.json",
        json.dumps({"node_type": "array", "zarr_format": 3, "data_type": {"name": ["a"]}}),
    ),
    (
        "v3_data_type_name_number",
        "zarr.json",
        json.dumps({"node_type": "array", "zarr_format": 3, "data_type": {"name": 3}}),
    ),
    ("v3_data_type_number", "zarr.json", json.dumps({"node_type": "array", "zarr_format": 3, "data_type": 5})),
    ("v3_data_type_null", "zarr.json", json.dumps({"node_type": "array", "zarr_format": 3, "data_type": None})),
    ("v2_dtype_number", ".zarray", json.dumps({"zarr_format": 2, "dtype": 5})),
    ("v2_dtype_true", ".zarray", json.dumps({"zarr_format": 2, "dtype": True})),
    ("v2_dtype_null", ".zarray", json.dumps({"zarr_format": 2, "dtype": None})),
    ("v2_dtype_object", ".zarray", json.dumps({"zarr_format": 2, "dtype": {"a": 1}})),
    ("data_type_array_in_a_zarray_document", ".zarray", json.dumps({"zarr_format": 2, "data_type": ["a"]})),
]


@pytest.mark.parametrize(
    ("doc_name", "body"), [m[1:] for m in _MALFORMED_ZARR_METADATA], ids=[m[0] for m in _MALFORMED_ZARR_METADATA]
)
def test_a_zarr_metadata_document_of_any_json_shape_never_crashes_the_consume_scan(tmp_path, doc_name, body):
    """Property: ``_scan_zero_user_info`` returns a list for a bundle root holding a file named ``zarr.json`` or
        ``.zarray`` whose content is valid JSON of ANY shape, including shapes that are not a zarr array
        document; it never raises. The list's content is not constrained: a refusal naming the document and a
        plain text scan of it are both correct.

    Class: REGRESSION - red on this tree, green on the pre-change loop, which text-scanned such a file and could
        not raise. The scan is documented non-fatal ("returns the leak list rather than raising") and its sibling
        ``decode_zarr_array_text`` says "none may escape as a crash". Measured by the reviewer: 12 of these 20
        arms raise ``AttributeError`` or ``TypeError`` out of ``_scan_zero_user_info`` (root arms: ``.get`` on a
        list, string, number, true or null in ``zarr.json``, and the same but null in ``.zarray``; a ``data_type``
        that is a list or has a list for a name, hashed into a frozenset); the other 8 already return, and are
        kept as the arms that vary the property under suspicion (an object-shaped or invalid document is
        handled, a non-object one is not).
    Reachability: ``assert_bundle_zero_user_info`` calls ``classify_zarr_array`` for every file whose NAME is
        ``zarr.json`` or ``.zarray``, before any look at its content (``_reprex_gate.py``, the first loop of the
        function); ``reprex()`` calls the scan on every unpacked bundle root. Not shown: a real bundle that
        carries such a document - the input is a damaged or foreign file with a zarr name.
    Both-states anchor: the assertion is that a list comes back, a property of the pre-change loop and of every
        correct fix, whatever the list holds.
    Invocation: not isolatable from this tree; discrimination shown by the named change to what the code DOES -
        ``classify_zarr_array`` answering ``ProcessingError`` for a document it cannot read as an object - turning
        all 20 arms green.
    What kills it: any classifier path that lets a non-``ProcessingError`` exception out for a malformed document.
    Second correct implementation: one that treats a non-object document as not an array (returns None) and lets
        the text scan read it; it returns a list at every arm.
    """
    root = tmp_path / "bundle"
    _write_zarr_metadata(root / "arr", doc_name, body)
    try:
        leaks = _consume(root)
    except Exception as exc:  # noqa: BLE001 - the property under test is that nothing escapes
        pytest.fail(f"the consume scan raised {type(exc).__name__} for a {doc_name} holding {body!r}: {exc}")
    assert isinstance(leaks, list)


#: Every text-bearing arm the module already builds for the emit tests (six v3 data types, three v2 kinds),
#: reused here so the consume side is held to the same table row for row.
_CONSUME_ARMS = [(f"v3_{row}", build) for row, build in _DECODE_ROWS] + [
    (f"v2_{kind}", build) for kind, build in _V2_ARMS
]


@pytest.mark.parametrize("placement", ["under_a_zarr_named_directory", "under_a_plain_directory"])
@pytest.mark.parametrize(("arm", "build"), _CONSUME_ARMS, ids=[a[0] for a in _CONSUME_ARMS])
def test_the_consume_scan_decodes_every_text_bearing_array_wherever_it_sits(tmp_path, arm, build, placement):
    """Property: a text-bearing zarr array under the bundle root, of EACH data type or v2 kind the table
        decodes, in either zarr format, under a directory named ``*.zarr`` or under a plain directory, is
        reported when a value is a per-user path and not reported when every value is benign.

    Class: NEW CAPABILITY. Plausible wrong implementation: the consume wiring decodes one data type (the
        coder's module builds only ``fixed_length_utf32``), reads only v3 ``zarr.json``, or recognises an array
        only below a directory whose name ends ``.zarr``. Red under each of those three, measured in-process:
        14, 6 and 9 of this test's 18 arms respectively; the coder's module (5 tests) stays green under all three.
    Reachability: the emit tests name the same table; a consumer's unpacked bundle holds the arrays the emit
        carried, in the directory layout the producer's analysis used. Artifact: ``classify_zarr_array`` keys on
        the metadata document, not on the directory name, so a plain directory is reachable.
    Both-states anchor: reported-versus-not for identical array shapes with and without the path; no wording of
        the report is asserted beyond the array's directory being named.
    Invocation: not isolatable from this tree; discrimination shown by the named changes to what the code DOES
        (decode only ``fixed_length_utf32``; only v3 documents; only ``*.zarr``-named parents) turning exactly
        the arms they restrict to passes.
    What kills it: any narrowing of which arrays the consume loop decodes.
    Second correct implementation: one that decodes every array regardless of type and lets the scanner's
        patterns decide; it reports at every arm.
    """
    rel = Path("s.zarr") / "a" if placement == "under_a_zarr_named_directory" else Path("store") / "a"
    planted = tmp_path / "planted"
    benign = tmp_path / "benign"
    _quiet(lambda: build(planted / rel, [PLANTED, "cpu", "gpu"]))
    _quiet(lambda: build(benign / rel, ["alpha", "cpu", "gpu"]))
    leaks = _consume(planted)
    assert leaks and rel.as_posix() in leaks[0], f"a per-user path in a {arm} array was not reported"
    assert PLANTED not in leaks[0]
    assert _consume(benign) == []


# ----------------------------------------------------------------------------------------------
# Chunk 1a.2, round 2 (RR-A): the consumer wording must survive EVERY recursion branch of the container table.
# ----------------------------------------------------------------------------------------------

_PRODUCER_PHRASES = ("no archive was written", "--exclude-config")
_BROKEN_ZIP = b"PK\x03\x04" + b"\x00" * 30


def _wrap_bzip2(inner: bytes) -> bytes:
    import bz2

    return bz2.compress(inner)


def _wrap_xz(inner: bytes) -> bytes:
    import lzma

    return lzma.compress(inner)


_ALL_WRAPS = [*_WRAPS, ("bzip2", _wrap_bzip2), ("xz", _wrap_xz)]


@pytest.mark.parametrize(("wrap_name", "wrap"), _ALL_WRAPS, ids=[w[0] for w in _ALL_WRAPS])
def test_a_damaged_container_nested_in_any_container_kind_is_worded_for_the_consumer_and_the_emit_keeps_its_own(
    tmp_path, wrap_name, wrap
):
    """Property: a damaged zip nested inside a container of EACH of the five kinds the table opens (zip, gzip, tar,
        bzip2, xz) is reported by the consume scan with a reason that names the outer member and the cause ("cannot
        be opened") and carries neither "no archive was written" nor "--exclude-config"; the same input through the
        real ``_emit_bundle_zip`` refuses with BOTH phrases still present.

    Class: NEW CAPABILITY. Plausible wrong implementation: the consumer flag is threaded through the zip recursion
        and dropped at the tar branch's recursive call or at the call shared by the gzip, bzip2 and xz branches
        (three recursive call sites in all). Red under dropping it at either, measured on a copy of the module:
        1 arm (tar) and 3 arms (gzip, bzip2, xz) go red; the coder's RR-A arms nest zips only and stay green
        under both drops.
    Reachability: any consumer bundle holding an archive of any of the five kinds around a damaged member.
        Artifact: ``scan_binary_member`` recurses at four call sites, one per branch, each passing the flag by hand.
    Both-states anchor: consume-versus-emit wording for the SAME bytes, asserted on the phrases and the cause, not
        on a full sentence.
    Invocation: not isolatable from this tree; discrimination shown by the named change to what the code DOES
        (the flag dropped at one recursion site) turning that kind's arm red.
    What kills it: any recursion site that does not pass the flag down.
    Second correct implementation: a consumer flag held on the scanner object instead of an argument; it passes.
    """
    from hhemt.bundle._emit import _emit_bundle_zip
    from hhemt.bundle._reprex_gate import ZeroUserInfoScanner

    root = tmp_path / "bundle"
    root.mkdir()
    (root / "m.bin").write_bytes(wrap(_BROKEN_ZIP))
    leaks = _consume(root)
    assert len(leaks) == 1 and "m.bin" in leaks[0] and "cannot be opened" in leaks[0], leaks
    assert not any(p in leaks[0] for p in _PRODUCER_PHRASES), leaks[0]
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "m.bin").write_bytes(wrap(_BROKEN_ZIP))
    with pytest.raises(ProcessingError) as info:
        _emit_bundle_zip(staging, tmp_path / "out.zip", scanner=ZeroUserInfoScanner.for_consume())
    assert all(p in info.value.reason for p in _PRODUCER_PHRASES), info.value.reason


def _chain(depth: int, wrap) -> bytes:
    data = b"hello"
    for _ in range(depth):
        data = wrap(data)
    return data


@pytest.mark.parametrize(("wrap_name", "wrap"), _ALL_WRAPS, ids=[w[0] for w in _ALL_WRAPS])
def test_a_container_nested_past_the_bound_in_any_kind_states_its_cause_for_the_consumer(tmp_path, wrap_name, wrap):
    """Property: a chain of 9 containers of ONE kind (each of the five) is reported by the consume scan with a
        reason naming the member, stating the cause ("8 or more containers around it") and carrying neither producer
        phrase; the emit refuses the same chain with both phrases.

    Class: NEW CAPABILITY. Plausible wrong implementation: the consumer nesting reason drops the cause (names the
        member only), or the flag is lost along the chain of one kind, so the depth refusal reverts to the emit's
        words. The coder's nest arm covers zip chains only and does not assert the cause.
    Reachability: a chain of 9 is built by the arm; the bound is ``_MAX_CONTAINER_DEPTH`` = 8.
    Both-states anchor: consume versus emit wording for identical bytes.
    Invocation: not isolatable from this tree; discrimination shown by the named changes to behaviour.
    What kills it: a consumer reason without its cause; the flag dropped at a recursion site.
    Second correct implementation: the flag carried on the scanner; passes.
    """
    from hhemt.bundle._emit import _emit_bundle_zip
    from hhemt.bundle._reprex_gate import ZeroUserInfoScanner

    assert G._MAX_CONTAINER_DEPTH == 8
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "chain.bin").write_bytes(_chain(9, wrap))
    leaks = _consume(root)
    assert len(leaks) == 1 and "chain.bin" in leaks[0] and "8 or more containers" in leaks[0], leaks
    assert not any(p in leaks[0] for p in _PRODUCER_PHRASES), leaks[0]
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "chain.bin").write_bytes(_chain(9, wrap))
    with pytest.raises(ProcessingError) as info:
        _emit_bundle_zip(staging, tmp_path / "out.zip", scanner=ZeroUserInfoScanner.for_consume())
    assert all(p in info.value.reason for p in _PRODUCER_PHRASES), info.value.reason


@pytest.mark.parametrize(("wrap_name", "wrap"), _ALL_WRAPS, ids=[w[0] for w in _ALL_WRAPS])
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_a_planted_path_under_stacked_stream_or_archive_containers_of_one_kind_is_found_on_both_sides(
    tmp_path, wrap_name, wrap, depth
):
    """Property: a per-user path inside 1, 2 or 3 stacked containers of ONE kind (zip, gzip, tar, bzip2, xz) is
        reported by the consume scan and refused by the emit; a benign body in the same stack returns ``[]`` and
        passes. Neither side raises anything but the scan's own refusal.

    Class: REGRESSION for stacked gzip - red on this tree at depth 2 and 3 for gzip only (``AttributeError``), green
        for the other four kinds. The nested ``GzipFile`` is opened over another ``GzipFile``, whose ``mode``
        attribute is an int, and the constructor calls ``mode.startswith``. The pre-change consume loop never
        raised, so the scan is no longer non-fatal for such a member (M5: "Non-fatal").
    Reachability: a doubly gzipped file (a re-compressed log, a ``.gz`` inside a ``.gz`` from a double pipe) in a
        consumer bundle or a declared input. Measured by a separate probe over benign and planted bodies: gzip at depth
        2 and 3 raises on the consume side in 4 of 4 inputs; bzip2 and xz at depths 1 to 3 return in 12 of 12.
    Both-states anchor: found-versus-passed for identical stacks with and without the path, independent of wording.
    Invocation: not isolatable from this tree; discrimination shown by the named change to what the code DOES
        (opening the nested stream with an explicit ``mode="rb"``) turning the red arms green: 111 passed on a copy.
    What kills it: any stream opener that reads its mode from the underlying file object.
    Second correct implementation: opening the nested stream through ``gzip.open(fh, "rb")`` (not run).
    """
    from hhemt.bundle._emit import _emit_bundle_zip
    from hhemt.bundle._reprex_gate import ZeroUserInfoScanner

    def stack(body: bytes) -> bytes:
        data = body
        for _ in range(depth):
            data = wrap(data)
        return data

    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "m.bin").write_bytes(stack(("see " + PLANTED + " here").encode()))
    leaks = _consume(bad)
    assert leaks and "m.bin" in leaks[0] and PLANTED not in leaks[0], leaks
    ok = tmp_path / "ok"
    ok.mkdir()
    (ok / "m.bin").write_bytes(stack(b"benign text"))
    assert _consume(ok) == []
    with pytest.raises(ProcessingError) as info:
        _emit_bundle_zip(bad, tmp_path / "bad.zip", scanner=ZeroUserInfoScanner.for_consume())
    assert "m.bin" in info.value.reason
    _emit_bundle_zip(ok, tmp_path / "ok.zip", scanner=ZeroUserInfoScanner.for_consume())
