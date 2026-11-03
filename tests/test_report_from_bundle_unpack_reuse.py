"""Tests for unpacking a member bundle zip beside itself, with a zip-identity record and reuse.

``unpack_member_zip`` is what ``hhemt report-from-bundle {member}.zip`` calls. A repeat call reuses a
folder it unpacked from the same zip, restores a missing or resized file, and starts over for a record
that is absent or names another zip. Every case builds a small zip under ``tmp_path`` with the standard
library, so no solver runs and all run on a laptop. Run as
``PYTHONPATH=${PWD}/src pytest tests/test_report_from_bundle_unpack_reuse.py``.

Each docstring names its property, class (NEW CAPABILITY: the record and reuse did not exist), the
wrong implementation that turns it red and a second correct implementation that stays green.
"""

from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

from hhemt.bundle._combine import unpack_member_zip

_RECORD = ".hhemt_unpack_source.json"


def _make_zip(path: Path, files: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, payload in files.items():
            zf.writestr(name, payload)
    return path


_FILES = {"bundle_manifest.json": b'{"a": 1}', "plots/fig.txt": b"figure-bytes", "zarr/c/0": b"0123456789"}


def test_a_first_unpack_extracts_every_file_and_writes_the_identity_record(tmp_path):
    """Property: the unpack yields exactly the zip's files plus the record naming the zip's size and mtime.

    Class: NEW CAPABILITY. Wrong implementation: no record, or a record before the files exist; red on
    the record check. Reachability: an ordinary member zip. Both-states anchor: file bytes compared
    with the literals the zip was built from. Second correct implementation: ``extractall`` then a
    record write.
    """
    z = _make_zip(tmp_path / "m.zip", _FILES)
    out = unpack_member_zip(z)
    assert out == (tmp_path / "m").resolve()
    for name, payload in _FILES.items():
        assert (out / name).read_bytes() == payload
    st = z.stat()
    assert json.loads((out / _RECORD).read_text()) == {"size": st.st_size, "mtime_ns": st.st_mtime_ns}


def test_a_repeat_unpack_keeps_a_complete_file_a_deleted_file_comes_back_and_a_truncated_one_is_restored(tmp_path):
    """Property: with the record matching, a right-sized file is kept, a deleted one restored, a truncated one restored.

    Class: NEW CAPABILITY. Wrong implementation: always rewriting (the same-size edit is lost) or never
    restoring; red on any of the three. Reachability: three files, each put in one state. Both-states
    anchor: the kept file is a same-size edit whose new bytes must survive. Second correct
    implementation: comparing sizes with ``os.scandir``.
    """
    z = _make_zip(tmp_path / "m.zip", _FILES)
    out = unpack_member_zip(z)
    (out / "bundle_manifest.json").write_bytes(b'{"a": 2}')  # same size: kept
    (out / "plots/fig.txt").unlink()  # deleted: restored
    (out / "zarr/c/0").write_bytes(b"01")  # truncated: restored
    unpack_member_zip(z)
    assert (out / "bundle_manifest.json").read_bytes() == b'{"a": 2}'
    assert (out / "plots/fig.txt").read_bytes() == b"figure-bytes"
    assert (out / "zarr/c/0").read_bytes() == b"0123456789"


def test_a_deleted_folder_is_unpacked_again(tmp_path):
    """Property: removing the unpack folder entirely gives a full unpack on the next call.

    Class: NEW CAPABILITY. Wrong implementation: trusting a record held elsewhere and extracting
    nothing; red on the missing files. Reachability: the folder, record included, is removed.
    Both-states anchor: file bytes. Second correct implementation: always starting from an empty folder.
    """
    import shutil

    z = _make_zip(tmp_path / "m.zip", _FILES)
    out = unpack_member_zip(z)
    shutil.rmtree(out)
    unpack_member_zip(z)
    assert (out / "zarr/c/0").read_bytes() == b"0123456789" and (out / _RECORD).is_file()


def test_a_replacement_zip_of_the_same_name_starts_from_an_empty_folder(tmp_path):
    """Property: a different zip under the same name drops the old folder's files, even ones of equal size.

    Class: NEW CAPABILITY. Wrong implementation: reusing by size alone (a same-size old file survives
    and a file only the old zip had stays behind); red on both. Reachability: the replacement keeps
    one name and size and changes the other file's name; its mtime is set apart explicitly. Both-states
    anchor: the old-only file is absent and the shared name holds the new bytes. Second correct
    implementation: hashing the zip for its identity.
    """
    z = _make_zip(tmp_path / "m.zip", {"keep.txt": b"AAAA", "old_only.txt": b"x"})
    out = unpack_member_zip(z)
    z.unlink()
    _make_zip(z, {"keep.txt": b"BBBB", "new_only.txt": b"y"})
    os.utime(z, (1_000_000_000, 1_000_000_000))
    unpack_member_zip(z)
    assert (out / "keep.txt").read_bytes() == b"BBBB"
    assert not (out / "old_only.txt").exists() and (out / "new_only.txt").read_bytes() == b"y"


def test_an_absent_or_unreadable_record_starts_from_an_empty_folder(tmp_path):
    """Property: with no record, or a record that is not JSON, the next unpack clears the folder first.

    Class: NEW CAPABILITY. Wrong implementation: treating a missing record as "same zip" (a stray old
    file stays); red on the stray file. Reachability: two folders, one record deleted, one corrupted.
    Both-states anchor: a stray file placed in each. Second correct implementation: refusing to
    reuse a folder it did not create.
    """
    for name, damage in (("a", "delete"), ("b", "corrupt")):
        z = _make_zip(tmp_path / f"{name}.zip", _FILES)
        out = unpack_member_zip(z)
        (out / "stray.txt").write_text("old")
        if damage == "delete":
            (out / _RECORD).unlink()
        else:
            (out / _RECORD).write_text("not json")
        unpack_member_zip(z)
        assert not (out / "stray.txt").exists()
        assert (out / "plots/fig.txt").read_bytes() == b"figure-bytes"
