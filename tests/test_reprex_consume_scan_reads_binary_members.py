"""The consume-side zero-user-info scan reads what a binary member holds, as the emit-time check does.

``reprex()`` runs ``_scan_zero_user_info`` over an unpacked bundle root and returns an informational leak list.
It used to read only members the NUL sniff classes text, so a text zarr array, a zip, or a UTF-16 file under the
root read clean whatever it held. It now applies the shared helpers of ``hhemt.bundle._reprex_gate`` (the zarr
decoder, the container table, the byte-order-mark table and the overlapping-block raw scan), and tests for a
byte-order mark BEFORE the NUL sniff, so a marked member is never read as UTF-8 text.

SOLVER-FREE: every arm writes ``tmp_path`` files. Planted values are BUILT FROM FRAGMENTS, never written as
literals: this file is scanned by the repository's anonymization guard.

Each test docstring states Property, Class, Reachability, Both-states anchor, Invocation isolated, What kills it,
and a second correct implementation under which it still passes.
"""

from __future__ import annotations

import io
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pytest
import zarr

from hhemt.bundle._reprex import _scan_zero_user_info


def _planted() -> str:
    return "/" + "home" + "/" + "someone" + "/" + "data"


PLANTED = _planted()
PLANTED_BYTES = PLANTED.encode()
_CJK = "あ" * 6000


@pytest.fixture(autouse=True)
def _no_private_list(monkeypatch):
    monkeypatch.delenv("HHEMT_REPREX_BLOCKLIST", raising=False)


def _scan(root: Path) -> list[str]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the unset-carrier warning, and zarr's unstable-dtype warning
        return _scan_zero_user_info(root)


def _text_array(root: Path, values) -> Path:
    directory = root / "store.zarr" / "arr"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        zarr.create_array(store=str(directory), data=np.array(values, dtype="U60"), zarr_format=3)
    return directory


def _raw_chunks(directory: Path) -> bytes:
    return b"".join(f.read_bytes() for f in sorted(directory.rglob("*")) if f.is_file() and f.name != "zarr.json")


def _filler(n: int) -> str:
    import random

    rng = random.Random(5)
    return "".join(rng.choice("abcdefghijklmnopqrstuvwxyz \n") for _ in range(n))


def test_a_planted_path_in_a_text_zarr_array_is_reported(tmp_path):
    """Property: a per-user path stored as a value of a ``fixed_length_utf32`` zarr array under the bundle root is
        reported; the same root with benign values returns ``[]``.

    Class: REGRESSION - red before the change, where the chunk files are binary and unread.
    Reachability: ``reprex()`` calls the scan on every consumer's unpacked bundle.
    Both-states anchor: the planted path is asserted ABSENT from the raw chunk bytes (so the arm proves decoding).
    Invocation isolated: ``_scan_zero_user_info`` over a fresh root per case.
    What kills it: skipping binary members; byte-scanning chunk files.
    Second correct implementation: decoding through ``zarr.open_group`` and scanning every string leaf.
    """
    bad = tmp_path / "bad"
    array_dir = _text_array(bad, [PLANTED, "cpu"])
    assert PLANTED_BYTES not in _raw_chunks(array_dir)
    leaks = _scan(bad)
    assert leaks and "store.zarr/arr (decoded values)" in leaks[0] and PLANTED not in leaks[0]
    ok = tmp_path / "ok"
    _text_array(ok, ["cpu", "gpu"])
    assert _scan(ok) == []


def test_a_planted_path_in_a_deflated_zip_is_reported(tmp_path):
    """Property: a per-user path inside a ``ZIP_DEFLATED`` entry of a zip under the bundle root is reported; a zip
        of benign content returns ``[]``.

    Class: REGRESSION - red before the change, and red under a raw-block scan of the zip's compressed bytes.
    Reachability: a bundle can carry a declared input that is a zip.
    Both-states anchor: the path is asserted ABSENT from the zip's raw bytes, so the arm proves decompression.
    Invocation isolated: ``_scan_zero_user_info`` over a fresh root per case.
    What kills it: raw-scanning the container; skipping binary members.
    Second correct implementation: extracting the zip to a directory and scanning the tree.
    """

    def zip_bytes(text: str) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("inner/notes.txt", _filler(1000) + text + _filler(1000))
        return buf.getvalue()

    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "input.zip").write_bytes(zip_bytes(" " + PLANTED + " "))
    assert PLANTED_BYTES not in (bad / "input.zip").read_bytes()
    leaks = _scan(bad)
    assert leaks and "input.zip" in leaks[0]
    ok = tmp_path / "ok"
    ok.mkdir()
    (ok / "input.zip").write_bytes(zip_bytes(" "))
    assert _scan(ok) == []


def test_a_planted_path_in_a_marked_utf16_file_is_reported(tmp_path):
    """Property: a UTF-16 LE file with a byte-order mark holding a per-user path is reported; benign text returns
        ``[]``.

    Class: REGRESSION - red before the change, where the NULs interleave every pattern and the file is skipped.
    Reachability: Windows-origin text files in a bundle.
    Both-states anchor: the path's UTF-8 bytes are asserted ABSENT from the raw bytes.
    Invocation isolated: ``_scan_zero_user_info`` over a fresh root per case.
    What kills it: raw-scanning the bytes; skipping NUL-bearing files.
    Second correct implementation: trying every codec and scanning the first that decodes.
    """
    import codecs

    bad = tmp_path / "bad"
    bad.mkdir()
    data = codecs.BOM_UTF16_LE + ("x " + PLANTED + " y").encode("utf-16-le")
    assert PLANTED_BYTES not in data
    (bad / "input.txt").write_bytes(data)
    leaks = _scan(bad)
    assert leaks and "input.txt" in leaks[0]
    ok = tmp_path / "ok"
    ok.mkdir()
    (ok / "input.txt").write_bytes(codecs.BOM_UTF16_LE + "x cpu y".encode("utf-16-le"))
    assert _scan(ok) == []


def test_a_marked_utf16_file_with_no_nul_in_its_first_block_is_reported(tmp_path):
    """Property: a marked UTF-16 LE file of 6000 CJK characters, a space and a planted path - whose first 8192
        bytes hold NO NUL byte, so the NUL sniff would class it text - is reported; the same file without the
        path returns ``[]``.

    Class: REGRESSION for P2-2 - red when the text-or-binary split is the NUL sniff alone.
    Reachability: any UTF-16 file of non-Latin text longer than a block.
    Both-states anchor: the first 8192 bytes are asserted NUL-free in the test, and the path absent from the raw bytes.
    Invocation isolated: ``_scan_zero_user_info`` over a fresh root per case.
    What kills it: testing the mark after the NUL sniff.
    Second correct implementation: routing every member to the binary path and scanning text there too.
    """
    import codecs

    bad = tmp_path / "bad"
    bad.mkdir()
    data = codecs.BOM_UTF16_LE + (_CJK + " " + PLANTED).encode("utf-16-le")
    assert b"\x00" not in data[:8192] and PLANTED_BYTES not in data
    (bad / "input.txt").write_bytes(data)
    leaks = _scan(bad)
    assert leaks and "input.txt" in leaks[0]
    ok = tmp_path / "ok"
    ok.mkdir()
    (ok / "input.txt").write_bytes(codecs.BOM_UTF16_LE + _CJK.encode("utf-16-le"))
    assert _scan(ok) == []


def test_a_nul_free_zarr_chunk_file_is_scanned_as_text_as_before_and_a_binary_one_is_not(tmp_path):
    """Property: a NUL-free file ``s.zarr/c/0`` with no array metadata, whose bytes carry a planted per-user path,
        is reported (the pre-change loop scanned it as text and refused it); the same chunk without the path
        returns ``[]``; a binary chunk (a NUL first, then the path) returns ``[]`` because it is not text-classed.

    Class: REGRESSION for the C1-2 ruling - red against a rule that skips every chunk file, which gives a verdict
        the pre-change gate did not.
    Reachability: a consumer's unpacked store whose chunk bytes happen to hold no NUL byte.
    Both-states anchor: the first 8192 bytes of the planted chunk are asserted NUL-free in the test; the control
        that varies the position of the NUL (NUL first) is binary and unread.
    Invocation isolated: ``_scan_zero_user_info`` over a fresh root per case.
    What kills it: skipping every chunk file; raw-scanning a binary chunk (the false-positive class the skip is for).
    Second correct implementation: deciding text versus binary by the NUL sniff for every member and skipping only
        the binary chunks.
    """
    planted = (_filler(200) + " " + PLANTED + " " + _filler(200)).encode()
    assert b"\x00" not in planted[:8192]
    bad = tmp_path / "bad"
    (bad / "s.zarr" / "c").mkdir(parents=True)
    (bad / "s.zarr" / "c" / "0").write_bytes(planted)
    leaks = _scan(bad)
    assert leaks and "s.zarr/c/0" in leaks[0] and PLANTED not in leaks[0]
    clean = tmp_path / "clean"
    (clean / "s.zarr" / "c").mkdir(parents=True)
    (clean / "s.zarr" / "c" / "0").write_bytes(_filler(400).encode())
    assert _scan(clean) == []
    binary = tmp_path / "binary"
    (binary / "s.zarr" / "c").mkdir(parents=True)
    (binary / "s.zarr" / "c" / "0").write_bytes(b"\x00" + planted)
    assert _scan(binary) == []


# --------------------------------------------------------------------------------------
# RR-A: the consume side words its three refusal reasons for a consumer; the emit keeps its own
# --------------------------------------------------------------------------------------

#: The emit's reasons for the three inputs below, captured from the code BEFORE the RR-A change (byte for byte).
_EMIT_CONTAINER = (
    "m.bin begins like a zip container but cannot be opened (BadZipFile), so its content was not checked; "
    "no archive was written. Fix or remove the file, or leave it out of the bundle and reference it through "
    "--exclude-config"
)
_EMIT_NEST = (
    "chain.zip!z7.zip!z6.zip!z5.zip!z4.zip!z3.zip!z2.zip!z1.zip!z0.zip is a container with 8 or more containers "
    "around it, so its content was not read; no archive was written. Leave the file out of the bundle and "
    "reference it through --exclude-config"
)
_EMIT_ZARR = "the zarr array cannot be decoded for the check (RuntimeError); no archive was written"


def _broken_zip(root: Path) -> None:
    (root / "m.bin").write_bytes(b"PK\x03\x04" + b"\x00" * 30)


def _nine_zips(root: Path) -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("leaf.txt", "hello")
    data = buf.getvalue()
    for level in range(8):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"z{level}.zip", data)
        data = buf.getvalue()
    (root / "chain.zip").write_bytes(data)


def _undecodable_text_array(root: Path) -> None:
    directory = _text_array(root, ["x", "y"])
    chunk = next(f for f in directory.rglob("*") if f.is_file() and f.name != "zarr.json")
    chunk.write_bytes(b"\xff" * 7)


_RRA = [
    ("container", _broken_zip, "m.bin", _EMIT_CONTAINER),
    ("nest", _nine_zips, "chain.zip", _EMIT_NEST),
    ("zarr", _undecodable_text_array, "store.zarr/arr", _EMIT_ZARR),
]


@pytest.mark.parametrize(("label", "build", "member", "emit_reason"), _RRA, ids=[r[0] for r in _RRA])
def test_a_refusal_reason_is_worded_for_the_consumer_and_the_emit_keeps_its_own(
    tmp_path, label, build, member, emit_reason
):
    """Property: for a file that begins ``PK\\x03\\x04`` and does not open, a chain of 9 nested zips, and a text zarr
        array whose chunk does not decode, ``_scan_zero_user_info`` returns an entry naming the member whose text
        carries neither "no archive was written" nor "--exclude-config"; the same input through ``_emit_bundle_zip``
        refuses with the emit's wording, equal to the reason captured before the change.

    Class: REGRESSION for RR-A - red before the change (all three carry "no archive was written" today, the first two
        also "--exclude-config").
    Reachability: ``reprex()`` runs the consume scan on a consumer's machine, where no archive is being written.
    Both-states anchor: the consume and the emit reasons are read for the SAME input in one test.
    Invocation isolated: a root alone holds the one input, so the returned entry is that member's own reason (no
        hits exist to share it, and ``raise_if_hits``' sentence is not in play).
    What kills it: one wording shared by both sides; a context argument that changes the emit's bytes.
    Second correct implementation: building the consumer reason from the emit's by substitution.
    """
    from hhemt.bundle._emit import _emit_bundle_zip
    from hhemt.bundle._reprex_gate import ZeroUserInfoScanner
    from hhemt.exceptions import ProcessingError

    root = tmp_path / "bundle"
    root.mkdir()
    build(root)
    leaks = _scan(root)
    assert len(leaks) == 1 and member.split("/")[0] in leaks[0], leaks
    assert "no archive was written" not in leaks[0] and "--exclude-config" not in leaks[0], leaks[0]

    staging = tmp_path / "staging"
    staging.mkdir()
    build(staging)
    with pytest.raises(ProcessingError) as info:
        _emit_bundle_zip(staging, tmp_path / "out.zip", scanner=ZeroUserInfoScanner.for_consume())
    assert info.value.reason == emit_reason


def test_a_zarr_json_holding_an_empty_list_passes_the_emit_and_a_planted_text_member_still_refuses(tmp_path):
    """Property: a file named ``zarr.json`` whose content is ``[]`` (valid JSON, not an array object) passes the real
        ``_emit_bundle_zip``; the same tree with a text member carrying a planted path refuses, naming that member.

    Class: REGRESSION - the emit crashed with AttributeError on this input before the classifier treated a non-object
        document as not an array.
    Reachability: any carried file named ``zarr.json`` or ``.zarray`` with unexpected content.
    Both-states anchor: the same ``zarr.json`` in both emits; only the text member differs.
    Invocation isolated: ``_emit_bundle_zip`` over a fresh staging tree per arm.
    What kills it: a classifier that indexes the parsed document without checking it is an object.
    Second correct implementation: catching the error in the emit and treating the file as text.
    """
    from hhemt.bundle._emit import _emit_bundle_zip
    from hhemt.bundle._reprex_gate import ZeroUserInfoScanner
    from hhemt.exceptions import ProcessingError

    ok = tmp_path / "ok"
    ok.mkdir()
    (ok / "zarr.json").write_text("[]")
    scanner = ZeroUserInfoScanner.for_consume()
    _emit_bundle_zip(ok, tmp_path / "ok.zip", scanner=scanner)
    assert scanner.hits == []
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "zarr.json").write_text("[]")
    (bad / "a_notes.txt").write_text("see " + PLANTED + " here\n")
    with pytest.raises(ProcessingError) as info:
        _emit_bundle_zip(bad, tmp_path / "bad.zip", scanner=ZeroUserInfoScanner.for_consume())
    assert "a_notes.txt" in info.value.reason and PLANTED not in info.value.reason
