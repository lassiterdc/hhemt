"""Reviewer tests for chunk 2b.5: ``apply_post_process_surgery_to_zip`` copies entries and swaps one page.

The function used to extract the report zip, edit ``report.html`` and re-zip it. It now reads the zip
entry by entry, writes a new zip beside it with a COPY of each entry's ``ZipInfo`` and replaces only the
page's bytes, then renames it over the original. The contract is that every entry other than the page
is carried over unchanged. These tests hold that at entry shapes the coder's file does not build: extra
fields, comments, a zip64-sized entry, directory and empty entries, a duplicated member name, non-ASCII
names, a page that is not under the zip's stem, an interrupt that is not an ``Exception``, a stale
temporary file, and an unwritable directory. Every zip is built under ``tmp_path``; no real report zip
is touched and no solver runs. Run as ``PYTHONPATH=${PWD}/src pytest tests/test_react_surgery_zip_rewrite_reviewer.py``.
"""

from __future__ import annotations

import os
import warnings
import zipfile
from pathlib import Path

import pytest

from hhemt.report_renderers._react_surgery import apply_post_process_surgery, apply_post_process_surgery_to_zip

_PAGE = "<html><body>\n<script>x.sort((a, b) => a.localeCompare(b));</script>\n</body></html>\n"
_PAST = (2020, 3, 4, 5, 6, 8)


def _expected(page: str = _PAGE) -> bytes:
    return apply_post_process_surgery(page, bundle_mode=False).encode()


def _info(name: str, method: int = zipfile.ZIP_DEFLATED, **attrs) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, _PAST)
    info.compress_type = method
    info.external_attr = 0o640 << 16
    for key, value in attrs.items():
        setattr(info, key, value)
    return info


def _snapshot(path: Path) -> list[tuple]:
    """Everything about each entry that is not its payload, plus its payload."""
    with zipfile.ZipFile(path) as zf:
        return [
            (
                i.filename, i.date_time, i.compress_type, i.external_attr, i.internal_attr, i.create_system,
                i.comment, i.is_dir(), zf.read(i) if not i.is_dir() else b"",
            )
            for i in zf.infolist()
        ]  # fmt: skip


def _compressed_stream(path: Path, name: str) -> bytes:
    raw = path.read_bytes()
    with zipfile.ZipFile(path) as zf:
        info = zf.getinfo(name)
    nl = int.from_bytes(raw[info.header_offset + 26 : info.header_offset + 28], "little")
    el = int.from_bytes(raw[info.header_offset + 28 : info.header_offset + 30], "little")
    start = info.header_offset + 30 + nl + el
    return raw[start : start + info.compress_size]


def test_extra_field_comment_and_compressed_bytes_of_untouched_entries_are_carried_over(tmp_path):
    """Property: an entry that is not the page keeps its extra field, comment and compressed stream.

    Input class: entries carrying a custom extra field (an extended-timestamp style record), a per-entry
    comment and an internal-attribute value, all DEFLATED at zlib level 6 as Snakemake writes them.
    Approved-shape row: section 5.1 row 2b.5, "copy every entry with its own settings"; 5.7 test (8),
    "each entry keeps its compression type". The coder's input has no extra field or comment. Why
    required: a re-render must not alter what it did not edit; bytes that differ without a content
    change make two rewrites of one report compare unequal. The oracle is the input zip read by the
    stdlib. Killed by a writer that builds a fresh ``ZipInfo`` from name and date only. Second correct
    implementation: a raw copy of each entry's local record that never decompresses it.
    """
    src = tmp_path / "analysis_report.zip"
    extra = b"UT\x05\x00\x01\x00\x00\x00\x00"
    with zipfile.ZipFile(src, "w") as zf:
        zf.writestr(_info("analysis_report/report.html"), _PAGE.encode())
        for name in ("analysis_report/data/a.json", "analysis_report/data/b.json"):
            info = _info(name, comment=b"entry comment", extra=extra, internal_attr=1)
            zf.writestr(info, b'{"k": "v"}' * 200)
    before = _snapshot(src)
    streams = {n: _compressed_stream(src, n) for n in ("analysis_report/data/a.json", "analysis_report/data/b.json")}
    apply_post_process_surgery_to_zip(src)
    after = _snapshot(src)
    assert [row[:8] for row in after] == [row[:8] for row in before]
    assert [row[8] for row in after[1:]] == [row[8] for row in before[1:]]
    with zipfile.ZipFile(src) as zf:
        for name in streams:
            assert zf.getinfo(name).extra == extra, name
            assert _compressed_stream(src, name) == streams[name], name


def test_a_zip64_sized_entry_keeps_one_zip64_record_and_reads_back(tmp_path, monkeypatch):
    """Property: an entry that needs zip64 is rewritten into a zip that reads back with the same payload.

    Input class: the zip64 threshold lowered to 1000 bytes so a 5 KB entry carries a zip64 extra record
    in the central directory (a real 4 GiB entry is not built on a laptop). Approved-shape row: section
    5.1 row 2b.5 and its "zip64 boundary" check. Reachability: a combined report bundle over 4 GiB.
    Why required: copying the old extra field and then letting the writer append its own zip64 record
    leaves two, which some readers refuse. The oracle is the payload read back and the count of zip64
    records in the local header. Killed by copying ``extra`` unchanged. Second correct implementation:
    clearing ``extra`` of its zip64 record before writing.
    """
    monkeypatch.setattr(zipfile, "ZIP64_LIMIT", 1000)
    src = tmp_path / "analysis_report.zip"
    big = bytes(range(256)) * 20
    with zipfile.ZipFile(src, "w") as zf:
        zf.writestr(_info("analysis_report/report.html"), _PAGE.encode())
        zf.writestr(_info("analysis_report/data/big.bin", zipfile.ZIP_STORED), big)
    apply_post_process_surgery_to_zip(src)
    raw = src.read_bytes()
    with zipfile.ZipFile(src) as zf:
        assert zf.testzip() is None
        assert zf.read("analysis_report/data/big.bin") == big
        info = zf.getinfo("analysis_report/data/big.bin")
    nl = int.from_bytes(raw[info.header_offset + 26 : info.header_offset + 28], "little")
    el = int.from_bytes(raw[info.header_offset + 28 : info.header_offset + 30], "little")
    local_extra = raw[info.header_offset + 30 + nl : info.header_offset + 30 + nl + el]
    records, pos = [], 0
    while pos + 4 <= len(local_extra):
        tag = int.from_bytes(local_extra[pos : pos + 2], "little")
        size = int.from_bytes(local_extra[pos + 2 : pos + 4], "little")
        records.append(tag)
        pos += 4 + size
    assert records.count(1) <= 1, records


def test_directory_and_empty_entries_are_kept_in_place(tmp_path):
    """Property: directory entries and zero-byte files survive in their original positions.

    Input class: a zip with a directory entry, a zero-byte file and the page. Approved-shape row: section
    5.1 row 2b.5, "copy every entry", and the function's own docstring ("an entry the input did not
    carry is not added", so one it did carry is kept). Why required: an extractor that skips directory
    entries or empty files would drop them silently. Killed by a copy loop that skips ``is_dir`` or
    empty payloads. Second correct implementation: a raw entry copy.
    """
    src = tmp_path / "analysis_report.zip"
    with zipfile.ZipFile(src, "w") as zf:
        zf.writestr(_info("analysis_report/", zipfile.ZIP_STORED), b"")
        zf.writestr(_info("analysis_report/report.html"), _PAGE.encode())
        zf.writestr(_info("analysis_report/empty.txt"), b"")
        zf.writestr(_info("analysis_report/data/", zipfile.ZIP_STORED), b"")
    names_before = [row[0] for row in _snapshot(src)]
    apply_post_process_surgery_to_zip(src)
    after = _snapshot(src)
    assert [row[0] for row in after] == names_before
    assert [row[7] for row in after] == [True, False, False, True]
    assert after[2][8] == b""
    assert after[1][8] == _expected()


def test_a_duplicated_member_name_is_carried_both_times_and_the_first_page_is_edited(tmp_path):
    """Property: two entries with one name both survive, in order, with their own bytes.

    Input class: the page listed twice (an appended zip), plus a repeated data name. Approved-shape row:
    section 5.1 row 2b.5, "copy every entry with its own settings"; a rewrite that keys entries by name
    collapses them. The stdlib reads each by its own ``ZipInfo``, which is the oracle used here. Why
    required: a collapse loses data silently; the old extract path also lost the earlier of the two, so
    this pins what the new code does rather than what the old code did. Killed by a dict keyed by name.
    Second correct implementation: a raw copy of local records.
    """
    src = tmp_path / "analysis_report.zip"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(src, "w") as zf:
            zf.writestr(_info("analysis_report/report.html"), _PAGE.encode())
            zf.writestr(_info("analysis_report/d.txt"), b"first d")
            zf.writestr(_info("analysis_report/d.txt"), b"second d")
            zf.writestr(_info("analysis_report/report.html"), _PAGE.encode() + b"<!--second-->")
    apply_post_process_surgery_to_zip(src)
    with zipfile.ZipFile(src) as zf:
        infos = zf.infolist()
        assert [i.filename for i in infos] == ["analysis_report/report.html", "analysis_report/d.txt"] * 2 or [
            i.filename for i in infos
        ] == [
            "analysis_report/report.html",
            "analysis_report/d.txt",
            "analysis_report/d.txt",
            "analysis_report/report.html",
        ]
        assert [zf.read(i) for i in infos if i.filename.endswith("d.txt")] == [b"first d", b"second d"]
        assert zf.read(infos[0]) == _expected()
        assert zf.read(infos[-1]) == _PAGE.encode() + b"<!--second-->"


def test_non_ascii_names_and_a_non_ascii_page_round_trip(tmp_path):
    """Property: non-ASCII entry names keep their text and the page's non-ASCII text is preserved.

    Input class: a data entry named with accented and CJK letters and a page containing the same text.
    Approved-shape row: section 5.1 row 2b.5 ("replace only the page"). Why required: the name's UTF-8
    flag and the page's encoding are the two places a rewrite can corrupt text; the old path read and
    wrote the page as text. Killed by writing the page with a different encoding or by dropping the
    name flag. Second correct implementation: a raw copy plus UTF-8 text for the page.
    """
    page = _PAGE.replace("<body>", "<body>café 漢字")
    src = tmp_path / "analysis_report.zip"
    with zipfile.ZipFile(src, "w") as zf:
        zf.writestr(_info("analysis_report/report.html"), page.encode())
        zf.writestr(_info("analysis_report/déjà/漢.txt"), "é".encode())
    apply_post_process_surgery_to_zip(src)
    with zipfile.ZipFile(src) as zf:
        assert zf.namelist() == ["analysis_report/report.html", "analysis_report/déjà/漢.txt"]
        assert zf.read("analysis_report/report.html") == _expected(page)
        assert zf.read("analysis_report/déjà/漢.txt") == "é".encode()


def test_the_page_is_found_when_it_is_not_under_the_zip_stem(tmp_path):
    """Property: with no ``{stem}/report.html``, the first ``report.html`` by archive order is edited.

    Input class: a zip named ``x.zip`` whose page sits under ``analysis_report/`` (the stem does not
    match), with a result file also named ``report.html`` later. Approved-shape row: section 5.1 row 2b.5,
    "replace the page at its exact path" and the function's documented fallback. Why required: the
    combined and sensitivity callers pass zips whose names differ from the folder inside. Killed by
    requiring the stem path. Second correct implementation: a search by last path segment.
    """
    src = tmp_path / "x.zip"
    with zipfile.ZipFile(src, "w") as zf:
        zf.writestr(_info("analysis_report/report.html"), _PAGE.encode())
        zf.writestr(_info("analysis_report/results/report.html"), b"<html>result</html>")
    apply_post_process_surgery_to_zip(src)
    with zipfile.ZipFile(src) as zf:
        assert zf.read("analysis_report/report.html") == _expected()
        assert zf.read("analysis_report/results/report.html") == b"<html>result</html>"


def test_an_interrupt_that_is_not_an_exception_leaves_the_original_and_no_partial(tmp_path, monkeypatch):
    """Property: a ``KeyboardInterrupt`` while writing leaves the original byte-identical and no temporary file.

    Input class: an interrupt raised on the second ``writestr``. Approved-shape row: section 5.1 row
    2b.5, "written beside the zip and renamed over it" (a failure leaves the original untouched and no
    partial). The coder's failure test raises an ``Exception`` from the surgery. Why required: Ctrl-C
    during a report render is the common interrupt. Killed by cleanup in ``except Exception``. Second
    correct implementation: a ``try``/``finally`` with a success flag.
    """
    src = tmp_path / "analysis_report.zip"
    with zipfile.ZipFile(src, "w") as zf:
        zf.writestr(_info("analysis_report/report.html"), _PAGE.encode())
        zf.writestr(_info("analysis_report/a.txt"), b"a" * 100)
        zf.writestr(_info("analysis_report/b.txt"), b"b" * 100)
    original = src.read_bytes()
    real = zipfile.ZipFile.writestr
    calls = {"n": 0}

    def _writestr(self, *a, **k):
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt
        return real(self, *a, **k)

    monkeypatch.setattr(zipfile.ZipFile, "writestr", _writestr)
    with pytest.raises(KeyboardInterrupt):
        apply_post_process_surgery_to_zip(src)
    assert src.read_bytes() == original
    assert sorted(p.name for p in tmp_path.iterdir()) == ["analysis_report.zip"]


def test_a_stale_temporary_zip_does_not_survive_or_reach_the_result(tmp_path):
    """Property: a ``.surgery_tmp.zip`` left by a killed earlier run is replaced and none remains.

    Input class: a longer junk file already at the temporary name. Approved-shape row: section 5.1 row
    2b.5 ("written beside the zip and renamed over it"). Why required: a SLURM kill leaves exactly this.
    Killed by opening the temporary in append mode. Second correct implementation: a uniquely named
    temporary file.
    """
    src = tmp_path / "analysis_report.zip"
    with zipfile.ZipFile(src, "w") as zf:
        zf.writestr(_info("analysis_report/report.html"), _PAGE.encode())
    (tmp_path / "analysis_report.surgery_tmp.zip").write_bytes(b"\xff" * 20000)
    apply_post_process_surgery_to_zip(src)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["analysis_report.zip"]
    assert src.read_bytes()[:4] == b"PK\x03\x04", "stale temporary bytes were kept in front of the archive"
    with zipfile.ZipFile(src) as zf:
        assert zf.testzip() is None
        assert zf.read("analysis_report/report.html") == _expected()


def test_an_unwritable_directory_raises_and_leaves_the_original(tmp_path):
    """Property: when the zip's directory cannot take a new file the call raises and the zip is unchanged.

    Input class: the containing directory at mode 0o500 (skipped for a user the mode does not bind).
    Approved-shape row: section 5.1 row 2b.5 and its "read-only destination directory" check. Why
    required: a silent success would report a surgery that did not happen. Killed by swallowing
    ``OSError``. Second correct implementation: checking ``os.access`` first and raising.
    """
    d = tmp_path / "ro"
    d.mkdir()
    src = d / "analysis_report.zip"
    with zipfile.ZipFile(src, "w") as zf:
        zf.writestr(_info("analysis_report/report.html"), _PAGE.encode())
    original = src.read_bytes()
    d.chmod(0o500)
    try:
        if os.access(d, os.W_OK):
            pytest.skip("directory permissions are not enforced for this user")
        with pytest.raises(OSError):
            apply_post_process_surgery_to_zip(src)
        assert src.read_bytes() == original
    finally:
        d.chmod(0o700)
