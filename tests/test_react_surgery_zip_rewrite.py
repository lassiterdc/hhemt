"""Tests for ``apply_post_process_surgery_to_zip`` after it stopped extracting the report zip.

The function used to extract the zip into a temporary folder, edit ``report.html`` and re-zip with
``shutil.make_archive``, which stamped every entry with the time of the rewrite and added a directory
entry for every directory. It now copies each entry with its own date, mode and compression method
and replaces only the page's bytes. The input here is built the way Snakemake builds a report zip:
file entries only, no folder entries, every entry dated a fixed past date.

Every test builds its zip under ``tmp_path``; none compiles or runs a solver and all run on a
laptop. Run from a worktree as ``PYTHONPATH=${PWD}/src pytest tests/test_react_surgery_zip_rewrite.py``.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from hhemt.report_renderers._react_surgery import apply_post_process_surgery, apply_post_process_surgery_to_zip

_PAST = (2020, 3, 4, 5, 6, 8)
_PAGE = "<html><body>\n<script>x.sort((a, b) => a.localeCompare(b));</script>\n</body></html>\n"
_ENTRIES = {
    # name: (bytes, compression method)
    "analysis_report/report.html": (_PAGE.encode(), zipfile.ZIP_DEFLATED),
    "analysis_report/data/a.json": (b'{"a": 1}' * 40, zipfile.ZIP_DEFLATED),
    "analysis_report/data/raw.bin": (bytes(range(256)) * 4, zipfile.ZIP_STORED),
    "analysis_report/plots/report.html": (b"<html>a RESULT file also named report.html</html>", zipfile.ZIP_DEFLATED),
}


def _build(path: Path, entries=_ENTRIES) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name, (payload, method) in entries.items():
            info = zipfile.ZipInfo(name, _PAST)
            info.compress_type = method
            info.external_attr = 0o640 << 16
            zf.writestr(info, payload)
    return path


def _expected_page() -> bytes:
    return apply_post_process_surgery(_PAGE, bundle_mode=False).encode()


def test_entry_names_dates_modes_and_methods_survive_and_only_the_page_changes(tmp_path):
    """Property: every entry keeps its name, order, date, mode and method; only the page's bytes change.

    Class: discriminator against the extract-and-``make_archive`` path, which stamps the time of the
    rewrite on every entry (so the date assertion fails with no pause between runs) and adds folder
    entries (so the name list differs on this folder-free input). Zip dates have two-second resolution,
    so comparing two back-to-back rewrites would not discriminate; the fixed past date does. The
    plausible wrong implementation is the old path, or one that recompresses everything as DEFLATED.
    Why required: a faithful rewrite is what makes the zip repeatable. A second correct implementation
    is a raw entry copy that never decompresses the untouched entries.
    """
    zip_path = _build(tmp_path / "analysis_report.zip")
    apply_post_process_surgery_to_zip(zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        assert zf.testzip() is None
        infos = zf.infolist()
        assert [i.filename for i in infos] == list(_ENTRIES)
        for info in infos:
            payload, method = _ENTRIES[info.filename]
            assert info.date_time == _PAST, info.filename
            assert info.compress_type == method, info.filename
            assert info.external_attr >> 16 == 0o640, info.filename
            if info.filename != "analysis_report/report.html":
                assert zf.read(info) == payload, info.filename
        assert zf.read("analysis_report/report.html") == _expected_page()
        assert zf.read("analysis_report/report.html") != _PAGE.encode()


def test_the_page_edited_is_the_one_at_the_zip_stem_path_not_a_result_file(tmp_path):
    """Property: with a decoy ``report.html`` listed FIRST, the page at ``{stem}/report.html`` is the one edited.

    Class: NEW CAPABILITY, differently positioned from the test above: the decoy entry precedes the real
    page in archive order, so a "first report.html" implementation edits the decoy and leaves the page.
    Why required: the page is chosen by its exact path. A second correct implementation is selecting by
    the full path computed from the zip's stem.
    """
    decoy = "analysis_report/plots/report.html"
    entries = {decoy: (_PAGE.encode(), zipfile.ZIP_DEFLATED)}
    entries.update({k: v for k, v in _ENTRIES.items() if k != decoy})
    zip_path = _build(tmp_path / "analysis_report.zip", entries)
    apply_post_process_surgery_to_zip(zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        assert zf.read(decoy) == _PAGE.encode()
        assert zf.read("analysis_report/report.html") == _expected_page()


def test_two_rewrites_of_one_zip_give_the_same_bytes_and_a_second_pass_is_idempotent(tmp_path):
    """Property: rewriting copies of one zip twice gives identical bytes, and rewriting a rewritten zip changes nothing.

    Class: REGRESSION-shaped on the idempotence arm (the surgery already guards double injection) and a
    discriminator on byte repeatability: the old path stamped the rewrite time, so two rewrites seconds
    apart could differ. Holds on one zlib build. Why required: repeatable output. A second correct
    implementation is any copy that fixes every stamp.
    """
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a = _build(tmp_path / "a" / "analysis_report.zip")
    b = _build(tmp_path / "b" / "analysis_report.zip")
    apply_post_process_surgery_to_zip(a)
    apply_post_process_surgery_to_zip(b)
    assert a.read_bytes() == b.read_bytes()
    once = a.read_bytes()
    apply_post_process_surgery_to_zip(a)
    assert a.read_bytes() == once


def test_a_zip_without_a_report_page_and_a_missing_zip_are_left_alone(tmp_path):
    """Property, differently positioned: no page leaves the zip byte-identical; a missing path returns quietly.

    Class: REGRESSION (both behaviours hold before and after). The plausible wrong implementation is
    rewriting the zip anyway, or raising for a missing path. Why required: callers invoke the function
    unconditionally. A second correct implementation is checking names before opening for write.
    """
    z = _build(tmp_path / "analysis_report.zip", {"analysis_report/other.html": (b"x", zipfile.ZIP_DEFLATED)})
    before = z.read_bytes()
    apply_post_process_surgery_to_zip(z)
    assert z.read_bytes() == before
    assert apply_post_process_surgery_to_zip(tmp_path / "missing.zip") is None
    assert sorted(p.name for p in tmp_path.iterdir()) == ["analysis_report.zip"]


def test_a_failure_while_editing_leaves_the_original_and_no_partial_file(tmp_path, monkeypatch):
    """Property: if the surgery raises, the original zip is byte-identical and no ``.surgery_tmp.zip`` remains.

    Class: REGRESSION on the original (the old path wrote beside and moved), with the partial-file arm
    new. The failure is injected by patching ``apply_post_process_surgery`` to raise ``ProcessingError``
    (the drift error callers rely on propagating). The plausible wrong implementation is writing over
    the original in place. Why required: a half-written report zip is worse than the stale one. A second
    correct implementation is a temporary file in another directory renamed over the original.
    """
    from hhemt.exceptions import ProcessingError
    from hhemt.report_renderers import _react_surgery as mod

    z = _build(tmp_path / "analysis_report.zip")
    before = z.read_bytes()

    def _boom(*_a, **_k):
        raise ProcessingError(operation="surgery", filepath=None, reason="drift")

    monkeypatch.setattr(mod, "apply_post_process_surgery", _boom)
    with pytest.raises(ProcessingError):
        mod.apply_post_process_surgery_to_zip(z)
    assert z.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["analysis_report.zip"]


def test_other_extra_field_records_survive_and_a_zip64_record_is_not_duplicated(tmp_path):
    """Property: another extra record (extended timestamp) is kept byte for byte and no entry gets two zip64 records.

    Class: NEW CAPABILITY for the zip64 half (an input carrying a zip64 record is rewritten with one),
    REGRESSION for the other-record half. The input entry has an extended-timestamp record (tag 0x5455)
    and a zip64 record (tag 0x0001) in its extra field. The plausible wrong implementation drops the
    whole extra field, or keeps it whole and lets the writer append a second zip64 record. Why
    required: other records carry data a reader may use, and a duplicate zip64 record is invalid. A
    second correct implementation is a fresh ZipInfo that re-adds the timestamp record itself.
    """
    import struct

    stamp = struct.pack("<HHBI", 0x5455, 5, 1, 1_600_000_000)
    zip64 = struct.pack("<HHQQ", 0x0001, 16, 0, 0)
    z = tmp_path / "analysis_report.zip"
    with zipfile.ZipFile(z, "w") as zf:
        info = zipfile.ZipInfo("analysis_report/report.html", _PAST)
        info.extra = stamp
        zf.writestr(info, _PAGE.encode())
        other = zipfile.ZipInfo("analysis_report/data.bin", _PAST)
        other.extra = stamp + zip64
        zf.writestr(other, b"data")
    apply_post_process_surgery_to_zip(z)
    with zipfile.ZipFile(z) as zf:
        assert zf.testzip() is None
        assert zf.getinfo("analysis_report/report.html").extra == stamp
        assert zf.read("analysis_report/data.bin") == b"data"
        extra = zf.getinfo("analysis_report/data.bin").extra
        tags = []
        pos = 0
        while pos + 4 <= len(extra):
            tags.append(int.from_bytes(extra[pos : pos + 2], "little"))
            pos += 4 + int.from_bytes(extra[pos + 2 : pos + 4], "little")
        assert tags.count(0x0001) <= 1
        assert extra.startswith(stamp)
