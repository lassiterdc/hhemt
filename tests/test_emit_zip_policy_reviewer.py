"""Reviewer tests for chunk 2b.1: the member bundle zip written by ``_emit_bundle_zip``.

``_emit_bundle_zip`` writes through ``bundle._archive.write_zip`` under the moderate policy, and
the contract it keeps is: the same entries, in the same order, with the same stamps and the same
content bytes. These tests hold that contract at input positions the coder's own tests do not
reach: mixed-case and non-ASCII names, entry sizes either side of the 1 MiB streaming block, an
interrupt that is not an ``Exception``, and a ``.partial`` left behind by a killed earlier emit.

Every test builds files under ``tmp_path`` and calls the one private function, so none compiles
or runs a solver and all run on a laptop. Run from a worktree as
``PYTHONPATH=${PWD}/src pytest tests/test_emit_zip_policy_reviewer.py``.

Expected values are written as literals, or recomputed with ``sorted``, ``zipfile`` and ``os``
directly; no test calls the writer to produce its own expectation, except the stale-partial test,
whose oracle is a second emit to a clean path in the same process.
"""

from __future__ import annotations

import os
import random
import signal
import zipfile
from pathlib import Path

import pytest

from hhemt.bundle._emit import _emit_bundle_zip

_MIB = 1 << 20


def _write(path: Path, payload: bytes) -> Path:
    """Write ``payload`` to ``path``, creating its parent directories, and return ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _filenames_distinct(root: Path) -> bool:
    """Whether the filesystem under ``root`` keeps names that differ only in case apart."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "Case").write_bytes(b"1")
    (root / "case").write_bytes(b"2")
    return len(list(root.iterdir())) == 2


def test_member_zip_order_is_code_point_order_by_path_parts_for_mixed_case_and_non_ascii_names(tmp_path):
    """Property: the entry order is the code-point order of the path PARTS, whatever the names' case or script.

    Input class: staging layouts whose sibling names differ in case (``Plots`` against ``plots``),
    in a non-ASCII letter (``e`` with an acute accent against ``z``), or in a character that sorts
    below or above ``/`` (space, ``-``, ``.``, ``0``). The coder's fixture is all lowercase ASCII
    with one hyphenated sibling, so a sort that folds case or that follows the locale passes it.
    Approved-shape row: section 5.1 row 2b.1, "keeping its entry set, order, stamps", and 5.7 2b
    test (1), "the same entry names, order, stamps and extracted bytes as today's stored archive";
    today's order is ``sorted(staging.rglob("*"))``, which compares path parts as code points.

    Why required: a re-emit on a case-insensitive or locale-collating host must list entries in
    the order every other host lists them, or two emits of one analysis differ with no content
    change. The change that kills it is a sort keyed by ``str.casefold`` or by ``locale.strxfrm``.

    The fixture is checked to discriminate before the assertion: the code-point order must differ
    from the joined-string order and from the case-folded order, so the test cannot pass on a
    layout where the wrong orders agree with the right one. A second correct implementation is a
    sort keyed by ``name.replace("/", "\\0")``, which gives the same order for these names.
    """
    if not _filenames_distinct(tmp_path / "probe"):
        pytest.skip("filesystem folds case, so case-differing names cannot be created")
    staging = tmp_path / "staging"
    names = [
        "Plots/a.txt",
        "plots/a.txt",
        "plots-x/a.txt",
        "plots x/a.txt",
        "plots.x/a.txt",
        "plots0/a.txt",
        "plots_x/a.txt",
        "PLOTS/a.txt",
        "README.md",
        "analysis.yaml",
        "Z.txt",
        "z.txt",
        "é.txt",
    ]
    for name in names:
        _write(staging / name, name.encode())

    by_parts = sorted(names, key=lambda n: Path(n).parts)
    by_joined = sorted(names)
    by_folded = sorted(names, key=lambda n: [part.casefold() for part in Path(n).parts])
    assert by_parts != by_joined, "fixture cannot tell a joined-string sort from a parts sort"
    assert by_parts != by_folded, "fixture cannot tell a case-folded sort from a code-point sort"

    out = tmp_path / "bundle.zip"
    _emit_bundle_zip(staging, out)
    with zipfile.ZipFile(out) as zf:
        assert [info.filename for info in zf.infolist()] == by_parts
        for info in zf.infolist():
            assert zf.read(info.filename) == info.filename.encode(), info.filename


@pytest.mark.parametrize("size", [0, 1, _MIB - 1, _MIB, _MIB + 1, 2 * _MIB + 3])
@pytest.mark.parametrize(
    ("arcname", "method"),
    [("notes/blob.bin", zipfile.ZIP_DEFLATED), ("eda/s.zarr/c/0/0", zipfile.ZIP_STORED)],
    ids=["deflated-name", "stored-chunk-name"],
)
def test_member_zip_entry_bytes_round_trip_at_sizes_around_the_streaming_block(tmp_path, arcname, method, size):
    """Property: every entry reads back as exactly the bytes of its source, and carries its method.

    Input class: file sizes of zero, one byte, one byte under, equal to, one byte over and a
    non-multiple above the 1 MiB streaming block, for a name the policy deflates and for a name
    it stores. The writer-level tests read a 3 MiB entry, which is a whole number of blocks, so a
    reader that drops the last PARTIAL block, or an empty file, is not reached by them.
    Approved-shape row: section 5.1 row 2b.1 and 2B's "entry contents stay byte-identical
    everywhere", with 5.7 2b test (1), "extracted bytes".

    Why required: an entry short by a tail, or a zero-byte marker file dropped or mis-sized, is a
    bundle that opens as a zip and has lost data; the status flag files an analysis carries are
    zero-byte. The change that kills it is a copy loop that stops at the last full block, or one
    that skips a file whose size is zero.

    Content is seeded random bytes, so DEFLATE cannot shrink it to nothing and a truncated entry
    cannot hide behind a short stream. A second correct implementation is ``ZipFile.writestr``
    with the same ZipInfo values, which reads back the same bytes and carries the same method.
    """
    payload = random.Random(size).randbytes(size)
    staging = tmp_path / "staging"
    _write(staging / arcname, payload)
    out = tmp_path / "bundle.zip"
    _emit_bundle_zip(staging, out)
    with zipfile.ZipFile(out) as zf:
        assert zf.testzip() is None
        info = zf.getinfo(arcname)
        assert info.compress_type == method
        assert info.file_size == size
        assert zf.read(arcname) == payload


@pytest.mark.skipif(
    not (hasattr(os, "mkfifo") and hasattr(signal, "setitimer")),
    reason="needs a FIFO and an interval timer to interrupt a blocked read",
)
@pytest.mark.parametrize("position", ["first", "last"])
def test_an_interrupt_mid_emit_leaves_the_old_destination_and_no_partial(tmp_path, position):
    """Property: an interrupt that is not an ``Exception`` leaves the old destination and no ``.partial``.

    Input class: a ``KeyboardInterrupt`` (a ``BaseException``) delivered while the writer is
    blocked reading one entry's source, with that entry sorting first or last among the entries.
    The coder's failing-emit tests inject a ``FileNotFoundError``, which an ``except Exception``
    cleanup also handles. Approved-shape row: section 5.1 row 2b.1, "writes through
    ``bundle._archive.write_zip``", whose contract (``_archive.write_zip`` docstring) is that
    "a failure or an interrupt leaves ``dest`` exactly as it was and removes the partial file".

    Reachability: ``hhemt bundle`` is run from a terminal against analyses whose bundle emit takes
    minutes, and Ctrl-C at that point is a ``KeyboardInterrupt`` inside the entry loop. The block
    is produced here without patching any function: the streamed source is a FIFO with no writer,
    so opening it for reading blocks until a timer signal raises the interrupt.

    Why required: a ``.partial`` left beside a destination is read by the next emit as its own
    working file, and an old destination that was already replaced would hide the failure. The
    change that kills it is a cleanup under ``except Exception`` instead of ``BaseException``.
    A second correct implementation is a ``try``/``finally`` with a success flag; it removes the
    partial on every exit that is not success, so the test passes.
    """
    staging = tmp_path / "staging"
    _write(staging / "m/middle.txt", b"middle\n" * 100)
    fifo = tmp_path / "analysis" / "blocked.fifo"
    fifo.parent.mkdir()
    os.mkfifo(fifo)
    arcname = "0-first.fifo" if position == "first" else "zz-last.fifo"
    out = tmp_path / "bundle.zip"
    out.write_bytes(b"OLD DESTINATION")

    def _interrupt(signum, frame):
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGALRM, _interrupt)
    signal.setitimer(signal.ITIMER_REAL, 0.5)
    try:
        with pytest.raises(KeyboardInterrupt):
            _emit_bundle_zip(staging, out, streamed={arcname: fifo})
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    assert out.read_bytes() == b"OLD DESTINATION"
    assert not (tmp_path / "bundle.zip.partial").exists()


@pytest.mark.parametrize("destination", ["absent", "an earlier valid zip"])
def test_a_stale_partial_from_a_killed_emit_does_not_reach_the_next_archive(tmp_path, destination):
    """Property: an emit after a killed one writes the archive a clean emit writes, and no ``.partial``.

    Input class: a ``bundle.zip.partial`` file left by an earlier emit that was killed (an
    out-of-memory kill or a time limit leaves no cleanup), longer than the archive the next emit
    writes, with the destination absent or holding an earlier valid zip. The coder's tests start
    from no partial. Approved-shape row: section 5.1 row 2b.1, "writes through
    ``bundle._archive.write_zip``", whose partial is the one place a killed run leaves residue.

    Why required: the residue is found by the next emit on the same path, which is exactly the
    re-run an operator makes after a kill; an emit that appends to it, or refuses because it
    exists, turns a transient kill into a bundle that is wrong or cannot be rewritten. The oracle
    is a second emit of the same inputs to a clean path in the same process, so equal bytes show no
    stale byte survived, and the junk is longer than the archive so a writer that does not
    truncate leaves a tail. The change that kills it is opening the partial in append or
    exclusive-create mode. A second correct implementation is one that writes to a fresh
    temporary name and renames, which ignores the stale partial altogether.
    """
    staging = tmp_path / "staging"
    _write(staging / "cfg.yaml", b"analysis_id: demo\n" * 50)
    _write(staging / "eda/s.zarr/c/0", bytes(range(256)) * 8)

    clean = tmp_path / "clean" / "bundle.zip"
    clean.parent.mkdir()
    _emit_bundle_zip(staging, clean)
    assert clean.stat().st_size < 4096, "archive must be shorter than the stale junk"

    out = tmp_path / "run" / "bundle.zip"
    out.parent.mkdir()
    (tmp_path / "run" / "bundle.zip.partial").write_bytes(b"\xff" * 8192)
    if destination != "absent":
        earlier = tmp_path / "earlier_staging"
        _write(earlier / "old.txt", b"an earlier bundle\n")
        _emit_bundle_zip(earlier, out)
        assert out.read_bytes() != clean.read_bytes()
        (tmp_path / "run" / "bundle.zip.partial").write_bytes(b"\xff" * 8192)

    _emit_bundle_zip(staging, out)

    assert out.read_bytes() == clean.read_bytes()
    assert not (tmp_path / "run" / "bundle.zip.partial").exists()
    with zipfile.ZipFile(out) as zf:
        assert zf.testzip() is None
