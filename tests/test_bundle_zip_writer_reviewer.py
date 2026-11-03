"""Reviewer tests for ``hhemt.bundle._archive`` (work chunk 1a.1).

Each test builds small files under ``tmp_path`` with the standard library only, so none
compiles or runs a solver and all run on a laptop as
``PYTHONPATH=${PWD}/src pytest tests/test_bundle_zip_writer_reviewer.py``. The module is
new, so no test here has a pre-change tree to be red on; each names the wrong
implementation it reds under and a second correct implementation it passes under.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from hhemt.bundle._archive import extract_zip, write_zip

_MIB = 1 << 20


def _zip_with_colliding_names(zip_path: Path, per_side: int) -> None:
    """Write a zip whose two ``x.txt`` entries (``x.txt``, then ``../x.txt``) share one target.

    The pair sits at the boundary between the first and second halves of the entry list,
    padded by ``per_side - 1`` small files on each side, so that a split of the entries
    into contiguous runs puts the earlier entry at the end of one run and the later
    entry at the start of the next.
    """
    with zipfile.ZipFile(zip_path, "w") as zf:
        for i in range(per_side - 1):
            zf.writestr(f"first_half/{i}.txt", b"x")
        zf.writestr("x.txt", b"EARLIER")
        zf.writestr("../x.txt", b"LATER")
        for i in range(per_side - 1):
            zf.writestr(f"second_half/{i}.txt", b"y")


@pytest.mark.parametrize("threads", [2, 4])
def test_extract_zip_keeps_the_entry_extractall_keeps_when_two_entries_share_a_target(
    tmp_path: Path, threads: int
) -> None:
    """Property: where two entries sanitise to one file, ``extract_zip`` leaves the file ``extractall`` leaves.

    Input class: zips holding two or more entries whose confined target path is the same
    (``x.txt`` and ``../x.txt``, which ``extractall`` confines to one file; the later
    entry overwrites the earlier). Approved-shape row: section 5.1 row 1a.1, ``extract_zip``
    "raising what `extractall` raises, path-confined as `extractall` is", and 5.7 1a test
    (2), "yields the same file set and bytes as `extractall`".

    Why required: the docstring of ``extract_zip`` states that the resulting files and
    bytes are those ``ZipFile.extractall`` gives. Entries are dealt to threads in
    contiguous runs, so two entries with one target can land on different threads, and
    whichever thread writes last wins; the earlier entry then replaces the later one.
    A bundle written by ``write_zip`` never repeats an arcname, but a zip from any other
    writer, or a hostile one, can.

    Class: NEW CAPABILITY. Plausible wrong implementation: the run-splitting in
    ``extract_zip`` that does not look at target paths; red under it (measured 20 of 20
    runs: ``extractall`` leaves ``LATER``, ``extract_zip`` leaves ``EARLIER``).
    Reachability: the entry names are the climbing-name case the shape itself asks to be
    confined, placed at a run boundary. Both-states anchor: the expectation is
    ``extractall``'s own result on the same zip, a property of the zip and not of any
    wording. Invocation not isolatable from this tree - discrimination shown instead:
    extracting the colliding entries in one thread (``worker_count`` of 1) greens it.
    Second correct implementation: putting every entry that shares a target into one run,
    or extracting serially whenever two entries share a target.
    """
    per_side = 300
    src = tmp_path / "collide.zip"
    _zip_with_colliding_names(src, per_side)
    ref, got = tmp_path / "ref", tmp_path / "got"
    ref.mkdir()
    with zipfile.ZipFile(src) as zf:
        zf.extractall(ref)
    extract_zip(src, got, threads=threads)
    assert (ref / "x.txt").read_bytes() == b"LATER"  # control: the oracle really keeps the later entry
    assert (got / "x.txt").read_bytes() == (ref / "x.txt").read_bytes()


def test_write_zip_writes_an_entry_over_the_zip64_limit_so_it_reads_back(tmp_path: Path, monkeypatch) -> None:
    """Property: an entry larger than ``zipfile.ZIP64_LIMIT`` is written in the zip64 form and reads back whole.

    Input class: entries whose size exceeds the zip64 threshold, both STORED (a zarr
    chunk path) and DEFLATED (a text path). Approved-shape row: section 5.1 row 1a.1,
    ``write_zip`` "streamed in 1 MiB blocks", whose combine inputs include member zips
    of 1.13 GB (section 2B).

    Why required: ``zipfile`` decides between the plain and zip64 headers when the entry
    is opened, from the size it is told then. An entry streamed with no size known up
    front, once it passes the limit, raises ``RuntimeError`` mid-write, so the writer must
    state each entry's size before streaming. The limit is lowered to 1000 bytes so a
    laptop file stands in for a 2 GiB one; ``zipfile`` reads the limit at call time.

    Class: NEW CAPABILITY, held green. Plausible wrong implementation: ``_write_entry``
    that no longer sets ``info.file_size`` before opening the entry; red under it
    (``RuntimeError: File size too large, try using force_zip64``, measured).
    Reachability: any source file over 2 GiB. Both-states anchor: the check reads the
    entries back with ``zipfile`` and compares bytes. Invocation not isolatable from this
    tree - discrimination shown instead: removing the ``file_size`` assignment reds it.
    Second correct implementation: opening every entry with ``force_zip64=True``.
    """
    monkeypatch.setattr(zipfile, "ZIP64_LIMIT", 1000)
    payload = bytes(range(256)) * 20
    big = tmp_path / "big.bin"
    big.write_bytes(payload)
    out = write_zip([("s.zarr/v/c/0", big), ("t/big.txt", big)], tmp_path / "o.zip")
    with zipfile.ZipFile(out) as zf:
        assert zf.testzip() is None
        assert [zf.read(name) for name in zf.namelist()] == [payload, payload]


@pytest.mark.parametrize("source_kind", ["path", "zip_member"])
def test_write_zip_streams_a_large_entry_in_blocks_of_at_most_one_mib(
    tmp_path: Path, monkeypatch, source_kind: str
) -> None:
    """Property: no single write into an entry exceeds 1 MiB, and a 3 MiB entry takes several writes.

    Input class: entries from a path or from a member of an open zip whose size is a
    multiple of the block size or larger. Approved-shape row: section 5.1 row 1a.1,
    ``write_zip`` "streamed in 1 MiB blocks".

    Why required: the shape promises that no entry is held whole in memory, which is what
    lets the combine write a member of more than 1 GB on a 16 GB job. A bytes source is
    exempt because its bytes are already in memory, so only the two file-backed kinds
    are measured.

    Class: NEW CAPABILITY, held green. Plausible wrong implementation: reading the source
    whole (``Path.read_bytes()`` or ``ZipFile.read``) and writing it in one call; red
    under it (the ``max(writes) <= 1 MiB`` assertion fails, measured for both source
    kinds). Reachability: any file-backed entry over
    1 MiB. Both-states anchor: the check counts the sizes of the writes ``zipfile``
    receives, a property of the data path and not of wording. Invocation not isolatable
    from this tree - discrimination shown instead: replacing the block copy with a whole
    read reds it. Second correct implementation: 64 KiB blocks (the upper bound is the
    only constraint, so a smaller block passes).
    """
    payload = bytes(range(256)) * (3 * _MIB // 256)
    src_file = tmp_path / "src.bin"
    src_file.write_bytes(payload)
    writes: list[int] = []
    real_write = zipfile._ZipWriteFile.write  # noqa: SLF001

    def recording_write(self, data):
        """Record the size of each write into a zip entry, then pass the write through."""
        writes.append(len(data))
        return real_write(self, data)

    monkeypatch.setattr(zipfile._ZipWriteFile, "write", recording_write)  # noqa: SLF001
    if source_kind == "path":
        write_zip([("t/big.txt", src_file)], tmp_path / "o.zip")
    else:
        holder = tmp_path / "holder.zip"
        with zipfile.ZipFile(holder, "w") as zf:
            zf.writestr("t/big.txt", payload)
        writes.clear()
        with zipfile.ZipFile(holder) as zf:
            write_zip([("t/big.txt", (zf, "t/big.txt"))], tmp_path / "o.zip")
    assert sum(writes) == len(payload)
    assert max(writes) <= _MIB
    assert len(writes) >= 3
