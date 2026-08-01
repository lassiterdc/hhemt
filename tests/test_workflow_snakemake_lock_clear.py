"""Unit tests for the non-interactive branch of
``SnakemakeWorkflowBuilder._check_and_clear_snakemake_lock``.

The branch was added in Phase 1 of synth-test-isolation-and-runtime
(Decision D1-Option-D). The end-to-end synth tests exercise the branch
indirectly; this module covers it directly per the Phase 1 unit-coverage
DoD.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from unittest import mock

import pytest

from hhemt.workflow import (
    _NON_INTERACTIVE_LOCK_CLEAR_ENV,
    SnakemakeWorkflowBuilder,
)


def _make_builder(analysis_dir):
    """Construct a SnakemakeWorkflowBuilder with a minimal stand-in analysis.

    The lock-clear helper only reads ``self.analysis_paths.analysis_dir``;
    bypassing ``__init__`` keeps the unit isolated from the broader builder
    construction graph.
    """
    builder = SnakemakeWorkflowBuilder.__new__(SnakemakeWorkflowBuilder)
    builder.analysis_paths = SimpleNamespace(analysis_dir=analysis_dir)
    return builder


def test_non_interactive_branch_clears_locks_and_incomplete(tmp_path, monkeypatch):
    """Env var set → locks/ and incomplete/ are rmtree'd, log/ is re-created."""
    analysis_dir = tmp_path / "an"
    snakemake_state = analysis_dir / ".snakemake"
    (snakemake_state / "locks").mkdir(parents=True)
    (snakemake_state / "locks" / "stale.lock").write_text("lock\n")
    (snakemake_state / "incomplete").mkdir(parents=True)
    (snakemake_state / "incomplete" / "job_xyz").write_text("incomplete\n")

    # log/ deliberately absent — branch must create it for the tee target.
    assert not (snakemake_state / "log").exists()

    monkeypatch.setenv(_NON_INTERACTIVE_LOCK_CLEAR_ENV, "1")

    builder = _make_builder(analysis_dir)
    # Pass a dummy snakefile_path — branch does not touch it.
    builder._check_and_clear_snakemake_lock(
        analysis_dir / "Snakefile", dry_run=False, verbose=False
    )

    assert not (snakemake_state / "locks").exists(), "locks/ should be removed"
    assert not (snakemake_state / "incomplete").exists(), "incomplete/ should be removed"
    assert (snakemake_state / "log").is_dir(), "log/ should be re-created"


def test_interactive_branch_when_env_unset(tmp_path, monkeypatch):
    """Env var unset → interactive prompt code path is invoked.

    Arranges a stale lock file and asserts ``input()`` is called. The user
    declines (``"n"``) so the helper raises ``WorkflowError`` — that's the
    canonical sentinel that the interactive branch fired (it never tries to
    rmtree silently when the user declines).
    """
    from hhemt.exceptions import WorkflowError

    analysis_dir = tmp_path / "an"
    locks_dir = analysis_dir / ".snakemake" / "locks"
    locks_dir.mkdir(parents=True)
    (locks_dir / "stale.lock").write_text("lock\n")

    monkeypatch.delenv(_NON_INTERACTIVE_LOCK_CLEAR_ENV, raising=False)

    # The interactive branch is reachable only with a TTY; stub one so this
    # test asserts the prompt under the condition it has always meant.
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)

    builder = _make_builder(analysis_dir)

    with mock.patch("builtins.input", return_value="n") as mock_input:
        with pytest.raises(WorkflowError):
            builder._check_and_clear_snakemake_lock(
                analysis_dir / "Snakefile", dry_run=False, verbose=False
            )

    assert mock_input.called, "interactive branch must call input() when env unset"


def test_non_tty_with_locks_refuses_with_actionable_error(tmp_path, monkeypatch):
    """Non-TTY + stale locks → WorkflowError naming the unlock command, not EOFError.

    This is the batch/sbatch submission path. It is unreachable from any
    ``--dry-run`` (the helper returns early on ``dry_run``), so this unit is
    the only coverage of it.
    """
    from hhemt.exceptions import WorkflowError

    analysis_dir = tmp_path / "an"
    locks_dir = analysis_dir / ".snakemake" / "locks"
    locks_dir.mkdir(parents=True)
    (locks_dir / "0.input.lock").write_text("lock\n")

    monkeypatch.delenv(_NON_INTERACTIVE_LOCK_CLEAR_ENV, raising=False)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)

    builder = _make_builder(analysis_dir)

    with mock.patch("builtins.input", side_effect=EOFError) as mock_input:
        with pytest.raises(WorkflowError) as excinfo:
            builder._check_and_clear_snakemake_lock(
                analysis_dir / "Snakefile", dry_run=False, verbose=False
            )

    assert not mock_input.called, "non-TTY must not reach input()"
    assert "--unlock" in str(excinfo.value), "error must carry the manual unlock command"


def test_non_tty_without_locks_returns_cleanly(tmp_path, monkeypatch):
    """Non-TTY + no locks → returns; the guard fires only when locks exist."""
    analysis_dir = tmp_path / "an"
    (analysis_dir / ".snakemake").mkdir(parents=True)

    monkeypatch.delenv(_NON_INTERACTIVE_LOCK_CLEAR_ENV, raising=False)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)

    builder = _make_builder(analysis_dir)

    with mock.patch("builtins.input", side_effect=EOFError):
        builder._check_and_clear_snakemake_lock(
            analysis_dir / "Snakefile", dry_run=False, verbose=False
        )
