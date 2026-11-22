"""Orchestration hygiene of ``TRITONSWMM_analysis`` (change 1118-tk3): stale locks and cleanup-metadata.

Two defects measured on Rivanna, 2026-11-19, are covered with no solver, no SLURM and no Snakemake
subprocess; the analysis is built with ``object.__new__`` and ``subprocess.run`` is replaced by a stub.

(a) After an orchestrator is killed, the next batch submit stopped at ``Stale Snakemake locks require
    confirmation, but stdin is not a TTY (batch submission)``. The analysis facade must unlock the stale
    locks, say what it unlocked and continue when no live driver holds the analysis directory, and must
    leave the locks alone whenever a live or indeterminate driver exists.
(b) ``snakemake --cleanup-metadata`` exits 1 with ``Failed to clean up metadata for the following files
    because the metadata was not present. If this is expected, there is nothing to do.`` and the toolkit
    raised ``WorkflowError``. That exact message is success with a warning that names the files; every
    other failure still raises.

Module guarded: ``hhemt.analysis`` (one-time chunk test for change 1118-tk3, to be promoted or deleted at
the landing).
"""

from __future__ import annotations

import ast
import inspect
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest

from hhemt import orchestrator_sentinels as osent
from hhemt.analysis import TRITONSWMM_analysis
from hhemt.exceptions import WorkflowError
from hhemt.workflow import _NON_INTERACTIVE_LOCK_CLEAR_ENV, SnakemakeWorkflowBuilder

# --------------------------------------------------------------------------------------------------
# (a) stale Snakemake locks
# --------------------------------------------------------------------------------------------------


def _lock_dir(analysis_dir: Path, names: tuple[str, ...]) -> list[Path]:
    locks = analysis_dir / ".snakemake" / "locks"
    locks.mkdir(parents=True, exist_ok=True)
    paths = [locks / n for n in names]
    for p in paths:
        p.write_text("lock\n")
    return paths


class _GateRecorder:
    """Stand-in for the builder's orchestrator-liveness gate; records how it was asked."""

    def __init__(self, verdict: WorkflowError | None) -> None:
        self.verdict = verdict
        self.calls: list[dict] = []

    def _orchestrator_liveness_gate(self, analysis_dir=None, *, exclude_driver_id=None, phase_label="reprocess"):
        self.calls.append({"analysis_dir": analysis_dir, "exclude_driver_id": exclude_driver_id, "phase": phase_label})
        return self.verdict


def _analysis_with_gate(analysis_dir: Path, gate) -> TRITONSWMM_analysis:
    a = object.__new__(TRITONSWMM_analysis)
    a.analysis_paths = types.SimpleNamespace(analysis_dir=analysis_dir)
    a._workflow_builder = gate
    return a


_LIVE = WorkflowError(phase="submit pre-submission orchestrator-liveness gate", return_code=1, stderr="live driver")


@pytest.mark.parametrize("names", [("0.lock",), ("0.lock", "1.lock", "2.lock")])
def test_no_live_driver_unlocks_every_stale_lock_and_names_them(tmp_path, capsys, names):
    """Property: with no live driver, every ``*.lock`` is removed and each one is named in the output.

    Class: any set of stale lock files under ``.snakemake/locks/`` in an analysis dir whose gate says clear.
    Killed by: skipping the removal, or removing without naming. A second correct implementation: shelling
    out to ``snakemake --unlock`` (it removes the same files) and printing the names it found first.
    """
    locks = _lock_dir(tmp_path, names)
    a = _analysis_with_gate(tmp_path, _GateRecorder(None))

    unlocked = a._unlock_stale_snakemake_locks_when_no_driver_is_live(dry_run=False, verbose=False)

    assert not any(p.exists() for p in locks)
    out = capsys.readouterr().out
    assert all(n in out for n in names)
    assert sorted(unlocked) == sorted(names)


@pytest.mark.parametrize("verdict", [_LIVE], ids=["live-or-indeterminate-driver"])
def test_a_live_driver_keeps_every_lock(tmp_path, capsys, verdict):
    """Property: when the gate reports a live (or indeterminate) driver, no lock is touched and nothing raises.

    Killed by: unlocking regardless of the gate. A second correct implementation: consult the sentinels
    directly instead of through the builder's gate and still refuse. The refusal itself is the driver
    claim's, which runs next in ``submit_workflow``; this helper only declines to unlock.
    """
    locks = _lock_dir(tmp_path, ("0.lock", "1.lock"))
    a = _analysis_with_gate(tmp_path, _GateRecorder(verdict))

    unlocked = a._unlock_stale_snakemake_locks_when_no_driver_is_live(dry_run=False, verbose=False)

    assert all(p.exists() for p in locks)
    assert unlocked == []
    assert "unlock" not in capsys.readouterr().out.lower()


def test_dry_run_unlocks_nothing_and_does_not_consult_the_gate(tmp_path):
    """Property: a dry run submits nothing, so it never removes a lock and never probes liveness.

    Killed by: dropping the dry-run early return. A second correct implementation: consult the gate and
    report would-unlock names without removing; this test asserts only that nothing is removed.
    """
    locks = _lock_dir(tmp_path, ("0.lock",))
    gate = _GateRecorder(None)
    a = _analysis_with_gate(tmp_path, gate)

    a._unlock_stale_snakemake_locks_when_no_driver_is_live(dry_run=True, verbose=False)

    assert locks[0].exists()


def test_no_lock_files_means_nothing_to_do(tmp_path, capsys):
    """Property: a tree with no lock files prints nothing and returns an empty list (the common case)."""
    a = _analysis_with_gate(tmp_path, _GateRecorder(None))
    assert a._unlock_stale_snakemake_locks_when_no_driver_is_live(dry_run=False, verbose=False) == []
    assert capsys.readouterr().out == ""


def test_the_operators_dead_driver_assertion_is_passed_to_the_gate(tmp_path):
    """Property: ``override_live_driver`` names a sentinel the operator found dead, and the gate must skip it.

    Without this the operator's assertion would be honoured by the driver claim but the stale locks would
    survive and stop the same submit one step later. Killed by: not forwarding the id.
    """
    _lock_dir(tmp_path, ("0.lock",))
    gate = _GateRecorder(None)
    a = _analysis_with_gate(tmp_path, gate)

    a._unlock_stale_snakemake_locks_when_no_driver_is_live(
        dry_run=False, verbose=False, override_live_driver="123-host-abcd1234"
    )

    assert gate.calls and gate.calls[0]["exclude_driver_id"] == "123-host-abcd1234"
    assert gate.calls[0]["phase"] == "submit"


def _real_builder_analysis(analysis_dir: Path) -> TRITONSWMM_analysis:
    """An analysis whose gate is the real ``SnakemakeWorkflowBuilder._orchestrator_liveness_gate``."""
    builder = SnakemakeWorkflowBuilder.__new__(SnakemakeWorkflowBuilder)
    builder.analysis_paths = types.SimpleNamespace(analysis_dir=analysis_dir)
    # The gate reads only these two fields, for the mtime bound on an unanswerable sentinel.
    builder.cfg_analysis = types.SimpleNamespace(hpc_total_job_duration_min=None, hpc_max_wait_for_inflight_min=60)
    return _analysis_with_gate(analysis_dir, builder)


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def test_real_gate_a_sentinel_of_a_dead_local_driver_does_not_keep_the_locks(tmp_path):
    """The measured defect end to end: a killed local driver leaves a sentinel and locks; both are stale.

    Reachability: the real ``_orchestrator_liveness_gate`` classifies a same-host ``local`` sentinel by
    ``ps -p {pid}``. A pid that has exited is DEAD, so the gate reclaims it and returns None.
    """
    locks = _lock_dir(tmp_path, ("0.lock",))
    osent.write_orchestrator_sentinel(
        tmp_path, driver_id="dead-driver", workflow_submission_mode="local", pid=_dead_pid()
    )
    a = _real_builder_analysis(tmp_path)

    a._unlock_stale_snakemake_locks_when_no_driver_is_live(dry_run=False, verbose=False)

    assert not locks[0].exists()


def test_real_gate_a_live_local_driver_keeps_the_locks(tmp_path):
    """Control for the test above: a sentinel whose pid is alive (this process) must keep the locks.

    Must-fire control: it varies the property under suspicion (driver liveness) and not the token.
    """
    locks = _lock_dir(tmp_path, ("0.lock",))
    osent.write_orchestrator_sentinel(tmp_path, driver_id="live-driver", workflow_submission_mode="local")
    a = _real_builder_analysis(tmp_path)

    a._unlock_stale_snakemake_locks_when_no_driver_is_live(dry_run=False, verbose=False)

    assert locks[0].exists()


def test_after_the_unlock_the_batch_lock_check_no_longer_refuses(tmp_path, monkeypatch):
    """The measured refusal itself: ``stdin is not a TTY (batch submission)`` is gone once the helper has run.

    Before the helper the builder's own lock check raises ``WorkflowError`` on a non-TTY stdin whenever a lock
    file exists; after it, the same check finds no lock and returns.
    """
    _lock_dir(tmp_path, ("0.lock",))
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
    monkeypatch.delenv(_NON_INTERACTIVE_LOCK_CLEAR_ENV, raising=False)
    a = _real_builder_analysis(tmp_path)
    builder = a._workflow_builder
    with pytest.raises(WorkflowError, match="not a TTY"):
        builder._check_and_clear_snakemake_lock(tmp_path / "Snakefile", dry_run=False, verbose=False)

    a._unlock_stale_snakemake_locks_when_no_driver_is_live(dry_run=False, verbose=False)

    builder._check_and_clear_snakemake_lock(tmp_path / "Snakefile", dry_run=False, verbose=False)


def test_submit_workflow_runs_the_unlock_before_the_driver_claim():
    """Wiring: the unlock helper is called in ``submit_workflow`` ahead of ``_acquire_submit_driver_claim``.

    Order matters: the claim writes this driver's own sentinel, which the gate would then read as a live
    driver. Class: the two call sites in the one method body, read from the AST (no submission is made).
    Not established: that the call is reached on every path of ``submit_workflow``; the sensitivity arm
    and the plain arm both pass through the statement.
    """
    src = textwrap.dedent(inspect.getsource(TRITONSWMM_analysis.submit_workflow))
    tree = ast.parse(src)
    lines: dict[str, list[int]] = {"unlock": [], "claim": []}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            if node.attr == "_unlock_stale_snakemake_locks_when_no_driver_is_live":
                lines["unlock"].append(node.lineno)
            elif node.attr == "_acquire_submit_driver_claim":
                lines["claim"].append(node.lineno)
    assert lines["unlock"], "submit_workflow never calls the unlock helper"
    assert lines["claim"], "control: the driver claim call must be found by the same matcher"
    assert min(lines["unlock"]) < min(lines["claim"])


# --------------------------------------------------------------------------------------------------
# (b) cleanup-metadata "metadata was not present"
# --------------------------------------------------------------------------------------------------

_ABSENT = "Failed to clean up metadata for the following files because the metadata was not present."
_EXPECTED = "If this is expected, there is nothing to do."


def _cleanup_analysis(analysis_dir: Path) -> TRITONSWMM_analysis:
    a = object.__new__(TRITONSWMM_analysis)
    a.analysis_paths = types.SimpleNamespace(analysis_dir=analysis_dir)
    return a


def _stub_snakemake(monkeypatch, *, returncode: int, stdout: str = "", stderr: str = "") -> None:
    def fake_run(cmd, *args, **kwargs):
        return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr=stderr)

    monkeypatch.setattr(subprocess, "run", fake_run)


@pytest.mark.parametrize(
    ("stream", "files"),
    [
        ("stderr", ["plots/system_overview.png"]),
        (
            "stderr",
            ["plots/system_overview.png", "plots/per_sim/e0/peak_flood_depth.png", "plots/per_sim/e0/conduit_flow.png"],
        ),
        ("stdout", ["plots/system_overview.png", "plots/per_sim/e1/conduit_flow.png"]),
    ],
)
def test_absent_metadata_message_is_success_with_a_warning_naming_the_files(
    tmp_path, monkeypatch, capsys, stream, files
):
    """Property: the exact absent-metadata message, with exit 1, returns normally and the warning names each file.

    Class: that message on stdout or stderr, one file or several. Killed by: raising on any non-zero exit
    (the defect), or succeeding without naming the files. A second correct implementation: parse the
    files from the stream with a regex instead of line splitting.
    """
    body = "\n".join(["WorkflowError:", _ABSENT, _EXPECTED, "Files:", *files])
    _stub_snakemake(monkeypatch, returncode=1, **{stream: body})
    a = _cleanup_analysis(tmp_path)

    a._invoke_snakemake_cleanup_metadata(files)

    err = capsys.readouterr()
    shown = err.out + err.err
    assert "WARNING" in shown
    assert all(f in shown for f in files)


def test_absent_metadata_message_names_only_the_files_the_message_lists(tmp_path, monkeypatch, capsys):
    """Property: the warning's file list is the message's list, not the whole requested list.

    Controls the over-naming arm: three paths were requested, Snakemake reported only one as absent.
    """
    requested = ["plots/a.png", "plots/b.png", "plots/c.png"]
    _stub_snakemake(monkeypatch, returncode=1, stderr="\n".join([_ABSENT, _EXPECTED, "Files:", "plots/b.png"]))
    a = _cleanup_analysis(tmp_path)

    a._invoke_snakemake_cleanup_metadata(requested)

    shown = "".join(capsys.readouterr())
    assert "plots/b.png" in shown
    assert "plots/a.png" not in shown
    assert "plots/c.png" not in shown


@pytest.mark.parametrize(
    "stderr",
    [
        "Error: something else broke\n",
        "WorkflowError:\nSyntax error in Snakefile\n",
        _ABSENT + "\n" + _EXPECTED + "\nplots/a.png\nTraceback (most recent call last):\n  boom\nKeyError: 'x'\n",
    ],
    ids=["unrelated-failure", "snakefile-error", "message-beside-a-traceback"],
)
def test_any_other_failure_still_raises(tmp_path, monkeypatch, stderr):
    """Property: only the absent-metadata message is forgiven; a different failure, or that message beside a
    traceback, still raises ``WorkflowError`` with the exit code.

    Killed by: forgiving every non-zero exit. A second correct implementation: forgive on an exact-line
    match of the message instead of a substring, which still raises on all three inputs here.
    """
    _stub_snakemake(monkeypatch, returncode=1, stderr=stderr)
    a = _cleanup_analysis(tmp_path)
    with pytest.raises(WorkflowError) as exc:
        a._invoke_snakemake_cleanup_metadata(["plots/a.png"])
    assert exc.value.return_code == 1


def test_no_snakefile_found_is_still_a_silent_no_op(tmp_path, monkeypatch, capsys):
    """Control: the pre-existing benign case keeps returning without a warning."""
    _stub_snakemake(monkeypatch, returncode=1, stderr="No Snakefile found, tried Snakefile\n")
    a = _cleanup_analysis(tmp_path)
    a._invoke_snakemake_cleanup_metadata(["plots/a.png"])
    assert capsys.readouterr().out == ""


def test_exit_zero_is_unchanged(tmp_path, monkeypatch, capsys):
    """Control: a clean exit prints no warning."""
    _stub_snakemake(monkeypatch, returncode=0)
    a = _cleanup_analysis(tmp_path)
    a._invoke_snakemake_cleanup_metadata(["plots/a.png"])
    assert "WARNING" not in capsys.readouterr().out


_REAL_SNAKEMAKE_STDERR = (
    "WorkflowError:\n"
    f"{_ABSENT}\n"
    f"{_EXPECTED}\n"
    "Otherwise, the reason might be file system latency or still running jobs.\n"
    "Consider running metadata cleanup again.\n"
    "Files:\n"
    "x.txt\n"
    "y.txt\n"
)


def test_the_real_snakemake_text_is_forgiven_and_names_exactly_its_files(tmp_path, monkeypatch, capsys):
    """Property: Snakemake's verbatim output (``WorkflowError:`` header, two prose lines, ``Files:`` label)
    is success and the warning names x.txt and y.txt, not the prose lines.

    Captured by running ``snakemake --cleanup-metadata x.txt y.txt`` (reviewer's measurement). Killed by a
    regex that rejects the header or a parser that takes every line after the message as a file.
    """
    _stub_snakemake(monkeypatch, returncode=1, stderr=_REAL_SNAKEMAKE_STDERR)
    _cleanup_analysis(tmp_path)._invoke_snakemake_cleanup_metadata(["x.txt", "y.txt"])
    shown = capsys.readouterr().out
    assert "2 file(s)" in shown
    assert shown.rstrip().endswith("Files: x.txt, y.txt")
    assert "Consider running" not in shown


@pytest.mark.parametrize(
    "stderr",
    [
        "WorkflowError:\nLockException:\nError: Directory cannot be locked.\n",
        "usage: snakemake [-h]\nsnakemake: error: unrecognized arguments: --bogus\n",
        "",
    ],
    ids=["lock-exception", "usage-error", "empty-output"],
)
def test_real_failures_beside_a_workflowerror_header_still_raise(tmp_path, monkeypatch, stderr):
    """Property: exempting the bare ``WorkflowError:`` header forgives nothing else."""
    _stub_snakemake(monkeypatch, returncode=1, stderr=stderr)
    with pytest.raises(WorkflowError):
        _cleanup_analysis(tmp_path)._invoke_snakemake_cleanup_metadata(["x.txt"])
