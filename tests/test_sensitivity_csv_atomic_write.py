"""The synthetic sensitivity CSV is replaced atomically, so a reader never meets a truncated file (finding F3).

Every array element on every node writes the same ``_sensitivity_configs/{analysis_name}.csv`` (the slug runs root is
shared), and ``DataFrame.to_csv`` truncates its target before it writes. A reader that opens the file in that window
(``sensitivity_analysis.py`` reads it with ``pandas.read_csv``) gets an empty file and ``EmptyDataError``; counted
array 1's chunk 2 failed in setup that way. ``Local_TestCases._write_synth_sensitivity_csv`` now writes a complete
temporary file in the same directory and ``os.replace``s it onto the target. The tests below force the interleaving
rather than wait for it, with the runs root pointed at ``tmp_path`` so nothing is written to the shared cache. The
cause of the array failure is a hypothesis (the trace did not reproduce it): these tests pin the mechanism, not the
failure. Nothing here runs a solver.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pandas as pd
import pytest

from tests.fixtures import test_case_catalog as catalog

_NAME = "atomic_probe"
_OLD = pd.DataFrame({"member_id": [90, 91]})


@pytest.fixture
def csv_path(tmp_path, monkeypatch) -> Path:
    """The target path of the probe CSV, holding a valid OLD file, with the catalog's runs root set to `tmp_path`."""
    monkeypatch.setattr(catalog, "slug_runs_root", lambda _slug: tmp_path)
    path = tmp_path / "_sensitivity_configs" / f"{_NAME}.csv"
    path.parent.mkdir(parents=True)
    _OLD.to_csv(path, index=False)
    return path


def _write(extra_columns: dict | None = None) -> Path:
    """Call the function under test for the 4-row coupled subset, with optional extra columns (different bytes)."""
    return catalog.Local_TestCases._write_synth_sensitivity_csv(
        analysis_name=_NAME, model_subset="all", extra_columns=extra_columns
    )


def test_a_reader_between_the_truncation_and_the_write_still_sees_the_old_rows(csv_path, monkeypatch):
    """Property: when `to_csv` is interrupted at the moment it has truncated its own target, a `read_csv` of the FINAL
    path returns the OLD rows; afterwards the final path holds the NEW rows and the directory holds nothing else.

    Class: REGRESSION, red on the code before this change, whose `to_csv` target IS the final path, so the inner read
    raises `EmptyDataError` (the array's exception), deterministically. The interleaving is forced: `to_csv` is wrapped
    so that it opens its own target for writing first (the truncation), then reads the final path, records the result,
    and only then writes.

    Why it is required, checkable against the code: the function writes `tmp_path` then `os.replace`s, so the final
    path is whole at every instant.

    What kills it: a single `df.to_csv(csv_path)`. A write-a-temp-then-copy variant is killed by the `os.replace` test
    below, not by this one.

    A second correct implementation under which it passes: one that writes the temporary file in a subdirectory of
    the same filesystem and renames it, if the replace-spy assertion were relaxed to the same filesystem."""
    real = pd.DataFrame.to_csv
    seen: list = []

    def interleaved(self, path_or_buf=None, *args, **kwargs):
        """Truncate the target, read the final path, record it, then write; the in-memory build passes through."""
        if path_or_buf is None:
            return real(self, *args, **kwargs)
        open(path_or_buf, "w").close()
        try:
            seen.append(pd.read_csv(csv_path))
        except Exception as exc:  # noqa: BLE001 - the recorded exception is the observable
            seen.append(exc)
        return real(self, path_or_buf, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_csv", interleaved)
    _write()
    assert len(seen) == 1 and isinstance(seen[0], pd.DataFrame), seen
    assert seen[0]["member_id"].tolist() == [90, 91]
    assert pd.read_csv(csv_path)["member_id"].tolist() == [0, 1, 2, 3]
    assert sorted(p.name for p in csv_path.parent.iterdir()) == [csv_path.name]


def test_the_replace_takes_a_same_directory_temporary_onto_the_final_path(csv_path, monkeypatch):
    """Property: exactly one `os.replace`, from a file in the target's own directory, not ending in `.csv`, onto the
    target.

    Class: NEW CAPABILITY. Closes the gap of the test above, which a write-temp-then-COPY implementation passes (the
    copy truncates the target as well). Also pins the three properties of the name: same directory (rename stays
    atomic), not `*.csv` (nothing that lists CSVs sees it) and a leading `.` (the suite's write-surface classifier reads
    it as a coordination file).

    Why it is required, checkable against the code: the temporary is built from `dest_dir` and `os.replace` is the only
    step that touches the final path.

    What kills it: a copy instead of a replace; a temporary in another directory; a temporary named `*.csv`.

    A second correct implementation under which it passes: one that uses `os.rename` on POSIX, if the spy wrapped it."""
    calls: list[tuple[Path, Path]] = []
    real = os.replace

    def spy(src, dst, *args, **kwargs):
        """Record the arguments, then replace."""
        calls.append((Path(src), Path(dst)))
        return real(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "replace", spy)
    _write()
    assert len(calls) == 1
    src, dst = calls[0]
    assert dst == csv_path and src.parent == csv_path.parent and src != dst
    assert not src.name.endswith(".csv") and src.name.startswith(".")


def test_a_failed_write_removes_the_temporary_and_leaves_the_old_file_intact(csv_path, monkeypatch):
    """Property: when `to_csv` raises after creating the temporary, no stray file remains and the old CSV is unchanged.

    Class: NEW CAPABILITY. The wrong implementation never removes the temporary on failure, leaving a growing set of
    dot files in the shared directory, or replaces the target before the write finishes.

    Why it is required, checkable against the code: the function unlinks the temporary in an `except` that re-raises.

    What kills it: dropping the unlink; replacing before writing. Control: the success path leaves no temporary
    (the first test).

    A second correct implementation under which it passes: one that writes through a `tempfile.NamedTemporaryFile` in
    the same directory and relies on its cleanup."""
    real = pd.DataFrame.to_csv

    def partial_then_fail(self, path_or_buf=None, *args, **kwargs):
        """Write part of the file, then fail; the in-memory build passes through."""
        if path_or_buf is None:
            return real(self, *args, **kwargs)
        Path(path_or_buf).write_text("member_id\n0\n")
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_csv", partial_then_fail)
    with pytest.raises(OSError, match="disk full"):
        _write()
    monkeypatch.setattr(pd.DataFrame, "to_csv", real)
    assert sorted(p.name for p in csv_path.parent.iterdir()) == [csv_path.name]
    assert pd.read_csv(csv_path)["member_id"].tolist() == [90, 91]


def test_each_call_uses_its_own_temporary_name(csv_path, monkeypatch):
    """Property: two calls in one process write two differently named temporaries.

    Class: NEW CAPABILITY. The wrong implementation uses one fixed temporary name, so two writers share it and one
    can replace the other's half-written file.

    Why it is required, checkable against the code: the name carries the process id and a random token per call.

    What kills it: a constant suffix. Control: the two names are compared, not asserted unequal to a literal.

    A second correct implementation under which it passes: one that takes the name from `tempfile.mkstemp`."""
    real = pd.DataFrame.to_csv
    targets: list[str] = []

    def record(self, path_or_buf=None, *args, **kwargs):
        """Record the temporary's name, then write; the in-memory build passes through."""
        if path_or_buf is None:
            return real(self, *args, **kwargs)
        targets.append(Path(path_or_buf).name)
        return real(self, path_or_buf, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_csv", record)
    _write()
    _write({"extra": [1, 2, 3, 4]})  # different bytes: two equal calls now write once
    assert len(targets) == 2 and targets[0] != targets[1], targets


def test_two_processes_started_together_leave_one_valid_file(tmp_path):
    """Smoke control, NOT a discriminator: two processes call the function at once and one valid CSV remains.

    Class: control. It can pass against the code before this change, because the truncation window is a few
    milliseconds; the forced-interleaving test above is what discriminates. It is here to show the replace holds up
    when two real writers collide on one path and neither leaves a temporary.

    Why it is required, checkable against the code: both processes write the same target through distinct temporaries.

    What kills it: nothing specific; a crash or a leftover file in a real collision would.

    A second correct implementation under which it passes: any that ends with one whole file."""
    script = textwrap.dedent(
        """
        import sys
        from pathlib import Path
        from tests.fixtures import test_case_catalog as catalog
        root = Path(sys.argv[1])
        catalog.slug_runs_root = lambda _slug: root
        for _ in range(20):
            catalog.Local_TestCases._write_synth_sensitivity_csv(analysis_name="atomic_probe", model_subset="all")
        """
    )
    repo = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(repo), str(repo / "src")])}
    procs = [
        subprocess.Popen([sys.executable, "-c", script, str(tmp_path)], cwd=repo, env=env, stderr=subprocess.PIPE)
        for _ in range(2)
    ]
    codes = [p.wait(timeout=120) for p in procs]
    assert codes == [0, 0], [p.stderr.read().decode()[-500:] for p in procs]
    directory = tmp_path / "_sensitivity_configs"
    assert sorted(p.name for p in directory.iterdir()) == ["atomic_probe.csv"]
    assert pd.read_csv(directory / "atomic_probe.csv")["member_id"].tolist() == [0, 1, 2, 3]


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit], ids=["keyboard-interrupt", "system-exit"])
def test_an_interrupt_during_the_write_removes_the_temporary_and_leaves_the_old_file_intact(
    csv_path, monkeypatch, interrupt
):
    """Property: when a `KeyboardInterrupt` or `SystemExit` (not an `Exception`) is raised from `to_csv` after the
    temporary exists, the temporary is removed, the interrupt propagates and the old CSV is unchanged.

    Class: NEW CAPABILITY. The wrong implementation cleans up with `except Exception`, which an interrupt passes
    through, leaving the dot-file in the directory the whole array shares (a Ctrl-C or a scancel's SIGTERM turned into
    `SystemExit` during the write is the realistic input). Control: the sibling failure test, which raises an
    `Exception` and passes under both.

    Why it is required, checkable against the code: the function catches `BaseException` precisely so that the
    temporary does not outlive an interrupted writer; no other test raises a non-`Exception`.

    What kills it: `except Exception` in place of `except BaseException`.

    A second correct implementation under which it passes: one that cleans up in a `finally` guarded by a success
    flag."""

    real = pd.DataFrame.to_csv

    def partial_then_interrupt(self, path_or_buf=None, *args, **kwargs):
        """Create the temporary with a partial body, then raise the interrupt; the in-memory build passes through."""
        if path_or_buf is None:
            return real(self, *args, **kwargs)
        Path(path_or_buf).write_text("member_id\n0\n")
        raise interrupt()

    monkeypatch.setattr(pd.DataFrame, "to_csv", partial_then_interrupt)
    with pytest.raises(interrupt):
        _write()
    assert sorted(p.name for p in csv_path.parent.iterdir()) == [csv_path.name]
    assert pd.read_csv(csv_path)["member_id"].tolist() == [90, 91]


_NEW_BYTES = (
    b"member_id,analysis.run_mode,analysis.n_mpi_procs,analysis.n_omp_threads,analysis.n_gpus,analysis.n_nodes\n"
    b"0,mpi,2,1,0,1\n1,openmp,1,2,0,1\n2,hybrid,2,2,0,1\n3,serial,1,1,0,1\n"
)

_AUDITED_CALLS = """
import json, os, sys
from pathlib import Path
from tests.fixtures import test_case_catalog as catalog

root = Path(sys.argv[1])
catalog.slug_runs_root = lambda _slug: root
WRITE = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC


def call(**kw):
    return catalog.Local_TestCases._write_synth_sensitivity_csv(analysis_name="atomic_probe", model_subset="all", **kw)


path = call()
before = os.stat(path)
events, recording = [], [False]


def hook(event, args):
    if not recording[0]:
        return
    if event == "os.rename":
        events.append(["rename", str(args[0]), str(args[1])])
    elif event == "open":
        mode = args[1] if len(args) > 1 else None
        flags = args[2] if len(args) > 2 else 0
        if (isinstance(mode, str) and mode != "r") or (mode is None and flags & WRITE):
            events.append(["open", str(args[0]), str(mode)])


sys.addaudithook(hook)
recording[0] = True
call(**json.loads(sys.argv[2]))
recording[0] = False
after = os.stat(path)
print(json.dumps({"events": events, "inode_same": before.st_ino == after.st_ino,
                  "mtime_same": before.st_mtime_ns == after.st_mtime_ns, "bytes": path.read_bytes().decode()}))
"""


def _second_call(tmp_path: Path, extra: dict | None = None) -> dict:
    """Make one call, then a second under a real audit hook; return the second call's write events and file facts."""
    repo = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(repo), str(repo / "src")])}
    done = subprocess.run(
        [sys.executable, "-c", _AUDITED_CALLS, str(tmp_path), json.dumps({"extra_columns": extra} if extra else {})],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_a_second_identical_call_opens_nothing_for_write_and_leaves_the_file_untouched(tmp_path):
    """Property: under a real audit hook, the second of two identical calls raises NO write-mode `open` and NO
    `os.rename` event, and the file's inode and `st_mtime_ns` are unchanged.

    Class: REGRESSION, red on the writer of 3a.7, which writes a temporary and replaces it on every call. Input class:
    equal bytes at the target.

    Why it is required, checkable against the code: the function builds the bytes in memory and returns when
    `csv_path.read_bytes()` equals them. One call, no write: the target is not churned for readers in other chunks.

    What kills it: skipping the compare (today's behaviour), inverting it, an existence check in place of a byte
    compare (the different-bytes arm below), a compare on a different serialisation than the write (equal data never
    matches), an early return that still touches the file (`os.utime` moves the mtime; an append or truncating open is
    an event).

    A second correct implementation under which it passes: one that compares a hash kept beside the file."""
    facts = _second_call(tmp_path)
    assert facts["events"] == [], facts["events"]
    assert facts["inode_same"] and facts["mtime_same"]
    assert facts["bytes"].encode() == _NEW_BYTES


def test_a_second_call_with_different_bytes_writes_and_replaces(tmp_path):
    """Control: the same two calls with an added column write a dot-prefixed temporary and rename it onto the target;
    the inode changes; and the file's bytes afterwards are exactly the in-memory serialisation.

    Class: NEW CAPABILITY. The wrong implementation treats an existing file as current (an existence check), or never
    writes after the first call.

    Why it is required, checkable against the code: only equal bytes return early.

    What kills it: `if csv_path.is_file(): return`. A frame with an extra column changes the bytes.

    A second correct implementation under which it passes: one that always writes when a hash differs."""
    facts = _second_call(tmp_path, {"extra": [1, 2, 3, 4]})
    kinds = [e[0] for e in facts["events"]]
    assert kinds.count("rename") == 1 and kinds.count("open") >= 1, facts["events"]
    rename = next(e for e in facts["events"] if e[0] == "rename")
    assert rename[2].endswith("atomic_probe.csv") and Path(rename[1]).name.startswith(".")
    assert facts["inode_same"] is False
    assert facts["bytes"].splitlines()[0] == (
        "member_id,analysis.run_mode,analysis.n_mpi_procs,analysis.n_omp_threads,analysis.n_gpus,analysis.n_nodes,extra"
    )


def test_the_bytes_written_are_exactly_the_in_memory_bytes_and_the_parameters_match(csv_path, monkeypatch):
    """Property: after a write the file holds exactly the four-row serialisation (LF endings, UTF-8, no index), and
    the in-memory build and the file write are called with the SAME `index`, `lineterminator` and `encoding`.

    Class: NEW CAPABILITY. The compare and the write must come from one call signature, or the second call of two
    equal frames would never match what the first wrote. The exact bytes are a literal here, not recomputed.

    Why it is required, checkable against the code: platform defaults for the line terminator and encoding differ from
    these on some hosts, so the parameters are passed explicitly; this test pins that they are, which a default-
    matching run on Linux cannot show by bytes alone.

    What kills it: dropping `lineterminator` or `encoding` from the file write, or writing with a different frame.

    A second correct implementation under which it passes: one that writes the in-memory bytes themselves."""
    real = pd.DataFrame.to_csv
    calls: list[tuple[bool, dict]] = []

    def record(self, path_or_buf=None, *args, **kwargs):
        """Record whether the call is in memory and its keyword arguments, then do it."""
        calls.append((path_or_buf is None, dict(kwargs)))
        return real(self, path_or_buf, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_csv", record)
    _write()
    assert csv_path.read_bytes() == _NEW_BYTES
    in_memory = [kw for flag, kw in calls if flag]
    to_file = [kw for flag, kw in calls if not flag]
    assert len(in_memory) == 1 and len(to_file) == 1, calls
    for kw in (in_memory[0], to_file[0]):
        assert kw["index"] is False and kw["lineterminator"] == "\n"
    assert to_file[0]["encoding"] == "utf-8"


@pytest.mark.parametrize("error", [PermissionError, OSError], ids=["permission", "oserror"])
def test_an_unreadable_existing_target_counts_as_different_and_is_written(csv_path, monkeypatch, error):
    """Property: when the target exists and holds exactly the new bytes but cannot be read, the writer still writes.

    Class: NEW CAPABILITY. The wrong implementation handles only `FileNotFoundError` (a `PermissionError` then escapes
    from the read) or treats an unreadable target as equal (the call returns without writing). The file is seeded with
    the new bytes so that "equal" would otherwise be the answer.

    Why it is required, checkable against the code: the read is inside `except OSError`, which falls through to the
    write.

    What kills it: `except FileNotFoundError`; `except OSError: return csv_path`. Control: the replace spy sees the
    write.

    A second correct implementation under which it passes: one that opens the file for reading with `open` and catches
    the same family."""
    csv_path.write_bytes(_NEW_BYTES)
    real_read = Path.read_bytes
    replaced: list[tuple] = []
    real_replace = os.replace

    def failing_read(self):
        """Fail for the target only."""
        if self == csv_path:
            raise error("unreadable")
        return real_read(self)

    def spy(src, dst, *args, **kwargs):
        """Record the replace, then do it."""
        replaced.append((src, dst))
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", failing_read)
    monkeypatch.setattr(os, "replace", spy)
    assert _write() == csv_path
    assert len(replaced) == 1 and Path(replaced[0][1]) == csv_path


def test_a_directory_at_the_target_still_raises_from_the_replace_and_leaves_no_temporary(csv_path):
    """Property: a directory at the target raises `IsADirectoryError` from the `os.replace` line, as it did before this
    change, and no temporary file is left.

    Class: NEW CAPABILITY, a control for the read-error handling: the compare meets the directory, the read error is
    swallowed, and the write path runs unchanged. The wrong implementation (`except FileNotFoundError` only) raises a
    different exception from a different place (`IsADirectoryError` from `read_bytes`), which this test tells apart by
    asserting WHERE the exception was raised.

    Why it is required, checkable against the code: `is_file()` is false for a directory, so the read is not
    attempted, and the replace of a file onto a directory fails.

    What kills it: a compare that reads the target without the `is_file` guard and without `OSError` handling.

    A second correct implementation under which it passes: one that checks `is_dir` first and raises the same
    exception itself."""
    csv_path.unlink()
    csv_path.mkdir()
    with pytest.raises(IsADirectoryError) as excinfo:
        _write()
    assert "os.replace" in excinfo.traceback[-1].statement.__str__()
    assert sorted(p.name for p in csv_path.parent.iterdir()) == [csv_path.name]


def test_a_target_of_the_same_length_and_different_content_is_rewritten(csv_path):
    """Property: an existing file with exactly the payload's length and one different byte is rewritten to the payload.

    Class: NEW CAPABILITY. The wrong implementation compares lengths (or any summary weaker than the bytes), so a stale
    file of the right size is left in place.

    Why it is required, checkable against the code: the early return needs `read_bytes()` to EQUAL the payload.

    What kills it: `len(read_bytes()) == len(payload)`. Control: the file is first asserted to have the payload's
    length and to differ from it.

    A second correct implementation under which it passes: one that compares a content hash."""
    stale = _NEW_BYTES.replace(b"mpi", b"mpx")
    assert len(stale) == len(_NEW_BYTES) and stale != _NEW_BYTES
    csv_path.write_bytes(stale)
    _write()
    assert csv_path.read_bytes() == _NEW_BYTES


def test_a_target_that_is_the_payload_plus_trailing_bytes_is_rewritten(csv_path):
    """Property: an existing file that begins with the payload and carries extra trailing content is rewritten to
    exactly the payload.

    Class: NEW CAPABILITY. The wrong implementation compares a prefix (`startswith`), so an appended-to file reads as
    current and stays stale.

    Why it is required, checkable against the code: equality, not containment, decides the early return.

    What kills it: `read_bytes().startswith(payload)`. Control: the seeded file is asserted to start with the payload
    and to differ from it.

    A second correct implementation under which it passes: one that also compares the lengths."""
    stale = _NEW_BYTES + b"9,extra,0,0,0,0\n"
    assert stale.startswith(_NEW_BYTES) and stale != _NEW_BYTES
    csv_path.write_bytes(stale)
    _write()
    assert csv_path.read_bytes() == _NEW_BYTES


def test_both_branches_return_the_target_path(csv_path):
    """Property: the early-return branch (equal bytes) and the write branch (different bytes) both return `csv_path`.

    Class: NEW CAPABILITY. The callers in the catalog assign the result (`csv_path = ..._write_synth_sensitivity_csv`)
    and pass it on, so the branch that skips the write must return the same path as the one that writes.

    Why it is required, checkable against the code: both branches end in `return csv_path`.

    What kills it: an early `return None`. Control: the write branch is exercised first, by the old OLD file that
    differs from the payload, and the early branch second, by the identical call.

    A second correct implementation under which it passes: one that returns a `Path` equal to the target."""
    assert _write() == csv_path  # the old file differs: the write branch
    assert csv_path.read_bytes() == _NEW_BYTES
    assert _write() == csv_path  # equal bytes now: the early-return branch


def test_a_target_with_junk_before_the_payload_is_rewritten(csv_path):
    """Property: an existing file that ENDS with the payload but carries extra leading content is rewritten to exactly
    the payload.

    Class: NEW CAPABILITY, the mirror of the trailing-bytes test: the wrong implementation compares a suffix
    (`endswith`), so a file with junk in front reads as current and stays stale.

    Why it is required, checkable against the code: equality, not containment, decides the early return.

    What kills it: `read_bytes().endswith(payload)`. Control: the seeded file is asserted to end with the payload and
    to differ from it.

    A second correct implementation under which it passes: one that compares lengths and content hashes."""
    stale = b"junk,line\n" + _NEW_BYTES
    assert stale.endswith(_NEW_BYTES) and stale != _NEW_BYTES
    csv_path.write_bytes(stale)
    _write()
    assert csv_path.read_bytes() == _NEW_BYTES


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit], ids=["keyboard-interrupt", "system-exit"])
def test_an_interrupt_during_the_compare_read_propagates_and_writes_nothing(csv_path, monkeypatch, interrupt):
    """Property: a `KeyboardInterrupt` or `SystemExit` raised while the target is being read for the compare propagates
    out of the writer, and the writer does not go on to write or replace.

    Class: NEW CAPABILITY. The wrong implementation catches `BaseException` around the compare, swallows the interrupt
    and carries on to write, so a Ctrl-C or a scancel during the read is lost. Input class: a read that raises a
    non-`Exception`. The target is seeded with the new bytes so the early return would otherwise be taken.

    Why it is required, checkable against the code: the compare catches `OSError` only; an interrupt is not an
    `OSError`.

    What kills it: `except BaseException: pass` in place of `except OSError: pass`. Control: the replace spy records no
    call, and the interrupt type is the one asserted.

    A second correct implementation under which it passes: one that catches `(FileNotFoundError, PermissionError,
    IsADirectoryError)` only."""
    csv_path.write_bytes(_NEW_BYTES)
    real_read = Path.read_bytes
    replaced: list = []
    real_replace = os.replace

    def interrupting_read(self):
        """Raise the interrupt for the target only."""
        if self == csv_path:
            raise interrupt()
        return real_read(self)

    def spy(src, dst, *args, **kwargs):
        """Record any replace."""
        replaced.append(dst)
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", interrupting_read)
    monkeypatch.setattr(os, "replace", spy)
    with pytest.raises(interrupt):
        _write()
    assert replaced == []
    assert sorted(p.name for p in csv_path.parent.iterdir()) == [csv_path.name]


def test_a_named_pipe_at_the_target_is_replaced_and_never_read(csv_path):
    """Property: a FIFO at the target is not read (a read would block until a writer opens it) and is replaced by the
    payload file; the call returns promptly.

    Class: NEW CAPABILITY. This is the one input that tells `is_file()` apart from the `except OSError`: a directory or
    a missing file raises from `read_bytes` and is caught, but `open` on a FIFO BLOCKS. Input class: a non-regular
    file that can be opened for reading without error.

    Why it is required, checkable against the code: the guard `csv_path.is_file()` is false for a FIFO, so the read is
    never attempted and the write path replaces it.

    What kills it: dropping `csv_path.is_file() and` from the condition (the call then hangs in `read_bytes`; the test
    runs the call in a thread and releases it after a short wait so the run does not hang).

    A second correct implementation under which it passes: one that opens with `O_NONBLOCK`."""
    import threading

    csv_path.unlink()
    os.mkfifo(csv_path)
    done = threading.Event()

    def call():
        """Run the writer; mark completion."""
        try:
            _write()
        finally:
            done.set()

    thread = threading.Thread(target=call, daemon=True)
    thread.start()
    finished = done.wait(timeout=10)
    if not finished:  # release a blocked reader so the thread ends
        fd = os.open(csv_path, os.O_WRONLY | os.O_NONBLOCK)
        os.close(fd)
        thread.join(timeout=5)
    assert finished, "the writer blocked reading a FIFO at the target"
    assert csv_path.is_file() and csv_path.read_bytes() == _NEW_BYTES
