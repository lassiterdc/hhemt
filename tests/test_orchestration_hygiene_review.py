"""Reviewer's tests for change 1118-tk3: stale Snakemake locks and the cleanup-metadata message.

Written by the reviewing seat against the entry in the scope document, not against the coder's tests. No
solver, no SLURM and no Snakemake run: the analysis is built with ``object.__new__`` and ``subprocess.run``
is stubbed. ``REAL_SNAKEMAKE_STDERR`` is the text ``snakemake --cleanup-metadata x.txt y.txt`` printed on
exit 1 when captured once on the laptop (stdout empty, stderr below, the ``WorkflowError:`` header first).

Module guarded: ``hhemt.analysis`` (one-time review test, uncommitted, for the landing decision).
"""

from __future__ import annotations

import socket
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, cast

import pytest

from hhemt import analysis as analysis_mod
from hhemt import orchestrator_sentinels as osent
from hhemt.analysis import TRITONSWMM_analysis
from hhemt.exceptions import WorkflowError
from hhemt.workflow import SnakemakeWorkflowBuilder

REAL_SNAKEMAKE_STDERR = (
    "WorkflowError:\n"
    "Failed to clean up metadata for the following files because the metadata was not present.\n"
    "If this is expected, there is nothing to do.\n"
    "Otherwise, the reason might be file system latency or still running jobs.\n"
    "Consider running metadata cleanup again.\n"
    "Files:\n"
    "x.txt\n"
    "y.txt\n"
)
PROSE_LINES = ("Otherwise, the reason might be file system latency", "Consider running metadata cleanup again")


def _analysis(analysis_dir: Path) -> TRITONSWMM_analysis:
    a = object.__new__(TRITONSWMM_analysis)
    a.analysis_paths = types.SimpleNamespace(analysis_dir=analysis_dir)
    return a


def _stub_run(monkeypatch, *, returncode: int, stdout: str = "", stderr: str = "") -> None:
    def fake_run(cmd, *args, **kwargs):
        return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr=stderr)

    monkeypatch.setattr(subprocess, "run", fake_run)


# --------------------------------------------------------------------------------------------------
# (b) cleanup-metadata message
# --------------------------------------------------------------------------------------------------


def test_real_snakemake_output_is_success_and_the_warning_names_only_the_files(tmp_path, monkeypatch, capsys):
    """Invariant: Snakemake's own message, as printed, returns without raising and names exactly its files.

    Violating input: the captured output including the ``WorkflowError:`` header Snakemake prefixes.
    Satisfying input of a different shape: the same text without the header (second test below).
    """
    _stub_run(monkeypatch, returncode=1, stderr=REAL_SNAKEMAKE_STDERR)

    _analysis(tmp_path)._invoke_snakemake_cleanup_metadata(["x.txt", "y.txt"])

    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "x.txt" in out
    assert "y.txt" in out
    assert "2 file(s)" in out, "the count must be of files, not of every line after the message"
    for prose in PROSE_LINES:
        assert prose not in out, f"a sentence of Snakemake prose was reported as a file: {prose!r}"


def test_the_files_helper_returns_exactly_the_listed_files_for_the_real_output():
    """Property: the helper's list equals the files Snakemake listed under ``Files:``; prose is not a file."""
    assert analysis_mod._files_with_absent_cleanup_metadata(REAL_SNAKEMAKE_STDERR) == ["x.txt", "y.txt"]


@pytest.mark.parametrize(
    "text",
    [
        REAL_SNAKEMAKE_STDERR.replace("WorkflowError:\n", ""),
        REAL_SNAKEMAKE_STDERR.replace("\nIf this is expected, there is nothing to do.", ""),
        "\n" + REAL_SNAKEMAKE_STDERR + "\n\n",
        REAL_SNAKEMAKE_STDERR.replace("x.txt\ny.txt\n", "plots/a b.png\n"),
    ],
    ids=["no-header", "no-expected-sentence", "padded-with-blank-lines", "path-with-space"],
)
def test_message_variants_are_forgiven_with_only_files_named(text):
    """Variants of the same message: still a success, and the list never holds a sentence of prose."""
    files = analysis_mod._files_with_absent_cleanup_metadata(text)
    assert files, "a variant of the absent-metadata message was not forgiven"
    for f in files:
        for prose in PROSE_LINES:
            assert prose not in f


def test_the_two_streams_are_joined_so_the_message_on_stdout_is_also_forgiven(tmp_path, monkeypatch, capsys):
    """The caller joins stdout and stderr; the message must be recognised on either one."""
    _stub_run(monkeypatch, returncode=1, stdout=REAL_SNAKEMAKE_STDERR.replace("WorkflowError:\n", ""))
    _analysis(tmp_path)._invoke_snakemake_cleanup_metadata(["x.txt"])
    assert "x.txt" in capsys.readouterr().out


@pytest.mark.parametrize(
    "stderr",
    [
        "WorkflowError:\nSyntax error in Snakefile\n",
        "WorkflowError:\nLockException: directory is locked\n",
        "Traceback (most recent call last):\n  File x\nKeyError: 'a'\n",
        "snakemake: error: unrecognized arguments: --bogus\n",
        "",
    ],
    ids=["workflow-error", "lock-exception", "traceback", "usage-error", "empty"],
)
def test_an_unrelated_cleanup_error_still_fails(tmp_path, monkeypatch, stderr):
    """An exit 1 that does not carry the absent-metadata message raises ``WorkflowError`` as before."""
    _stub_run(monkeypatch, returncode=1, stderr=stderr)
    with pytest.raises(WorkflowError) as err:
        _analysis(tmp_path)._invoke_snakemake_cleanup_metadata(["x.txt"])
    assert err.value.phase == "cleanup_stale_metadata"


def test_the_message_beside_a_traceback_still_fails(tmp_path, monkeypatch):
    """The message is forgiven alone, not when a real failure is printed with it."""
    text = "Traceback (most recent call last):\n  File x\nValueError: boom\n" + REAL_SNAKEMAKE_STDERR
    _stub_run(monkeypatch, returncode=1, stderr=text)
    with pytest.raises(WorkflowError):
        _analysis(tmp_path)._invoke_snakemake_cleanup_metadata(["x.txt"])


def test_exit_zero_and_the_no_snakefile_path_are_untouched(tmp_path, monkeypatch, capsys):
    """Exit 0 prints no warning; ``No Snakefile found`` stays a silent skip."""
    _stub_run(monkeypatch, returncode=0)
    _analysis(tmp_path)._invoke_snakemake_cleanup_metadata(["x.txt"])
    _stub_run(monkeypatch, returncode=1, stderr="No Snakefile found, tried Snakefile\n")
    _analysis(tmp_path)._invoke_snakemake_cleanup_metadata(["x.txt"])
    assert capsys.readouterr().out == ""


# --------------------------------------------------------------------------------------------------
# (a) stale locks
# --------------------------------------------------------------------------------------------------


def _locks(analysis_dir: Path, names=("a.lock", "b.lock")) -> list[Path]:
    d = analysis_dir / ".snakemake" / "locks"
    d.mkdir(parents=True, exist_ok=True)
    paths = [d / n for n in names]
    for p in paths:
        p.write_text("x\n")
    return paths


def _real_gate_analysis(analysis_dir: Path) -> TRITONSWMM_analysis:
    builder = SnakemakeWorkflowBuilder.__new__(SnakemakeWorkflowBuilder)
    builder.analysis_paths = types.SimpleNamespace(analysis_dir=analysis_dir)
    builder.cfg_analysis = types.SimpleNamespace(hpc_total_job_duration_min=None, hpc_max_wait_for_inflight_min=60)
    a = _analysis(analysis_dir)
    a._workflow_builder = builder
    return a


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def test_no_sentinel_at_all_unlocks_and_logs_the_names(tmp_path, capsys):
    """No driver ever claimed the directory: the locks are stale, removed, and named in the log."""
    paths = _locks(tmp_path)
    removed = _real_gate_analysis(tmp_path)._unlock_stale_snakemake_locks_when_no_driver_is_live(
        dry_run=False, verbose=False
    )
    assert sorted(removed) == ["a.lock", "b.lock"]
    assert not any(p.exists() for p in paths)
    out = capsys.readouterr().out
    assert "a.lock" in out and "b.lock" in out


def test_a_dead_driver_sentinel_unlocks(tmp_path):
    paths = _locks(tmp_path)
    osent.write_orchestrator_sentinel(tmp_path, driver_id="dead", workflow_submission_mode="local", pid=_dead_pid())
    _real_gate_analysis(tmp_path)._unlock_stale_snakemake_locks_when_no_driver_is_live(dry_run=False, verbose=False)
    assert not any(p.exists() for p in paths)


def test_a_live_driver_keeps_every_lock_and_prints_nothing(tmp_path, capsys):
    """Must NOT unlock while a live driver (this process) holds the directory."""
    paths = _locks(tmp_path)
    osent.write_orchestrator_sentinel(tmp_path, driver_id="live", workflow_submission_mode="local")
    removed = _real_gate_analysis(tmp_path)._unlock_stale_snakemake_locks_when_no_driver_is_live(
        dry_run=False, verbose=False
    )
    assert removed == []
    assert all(p.exists() for p in paths)
    assert "unlocked" not in capsys.readouterr().out


def test_an_indeterminate_driver_keeps_the_locks(tmp_path):
    """A sentinel from another host cannot be probed from here: indeterminate counts as live."""
    paths = _locks(tmp_path)
    osent.write_orchestrator_sentinel(tmp_path, driver_id="foreign", workflow_submission_mode="local", pid=_dead_pid())
    sentinel = next((tmp_path / "_status" / "_orchestrator").glob("foreign.json"))
    import json

    data = json.loads(sentinel.read_text())
    data["hostname"] = socket.gethostname() + "-other-node"
    sentinel.write_text(json.dumps(data))
    _real_gate_analysis(tmp_path)._unlock_stale_snakemake_locks_when_no_driver_is_live(dry_run=False, verbose=False)
    assert all(p.exists() for p in paths)


def test_override_naming_a_dead_driver_does_not_unlock_past_another_live_driver(tmp_path):
    """The operator's assertion covers the one named sentinel only; a second, live driver still blocks."""
    paths = _locks(tmp_path)
    osent.write_orchestrator_sentinel(tmp_path, driver_id="dead", workflow_submission_mode="local", pid=_dead_pid())
    osent.write_orchestrator_sentinel(tmp_path, driver_id="live", workflow_submission_mode="local")
    _real_gate_analysis(tmp_path)._unlock_stale_snakemake_locks_when_no_driver_is_live(
        dry_run=False, verbose=False, override_live_driver="dead"
    )
    assert all(p.exists() for p in paths)


def test_dry_run_touches_nothing_and_does_not_ask_the_gate(tmp_path):
    paths = _locks(tmp_path)

    class Boom:
        def _orchestrator_liveness_gate(self, *a, **k):
            raise AssertionError("the gate must not be consulted on a dry run")

    a = _analysis(tmp_path)
    cast(Any, a)._workflow_builder = Boom()
    assert a._unlock_stale_snakemake_locks_when_no_driver_is_live(dry_run=True, verbose=False) == []
    assert all(p.exists() for p in paths)


def test_nothing_but_lock_files_are_removed(tmp_path):
    """Only ``*.lock`` under ``.snakemake/locks``; metadata and incomplete markers are not Snakemake locks."""
    _locks(tmp_path)
    meta = tmp_path / ".snakemake" / "metadata"
    inc = tmp_path / ".snakemake" / "incomplete"
    meta.mkdir(parents=True)
    inc.mkdir(parents=True)
    (meta / "m").write_text("m")
    (inc / "i").write_text("i")
    other = tmp_path / ".snakemake" / "locks" / "keep.txt"
    other.write_text("k")
    _real_gate_analysis(tmp_path)._unlock_stale_snakemake_locks_when_no_driver_is_live(dry_run=False, verbose=False)
    assert (meta / "m").exists() and (inc / "i").exists() and other.exists()
