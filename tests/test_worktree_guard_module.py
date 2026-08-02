"""VMS-F2: the importable worktree guard for NON-pytest entrypoints.

`tests/test_worktree_guard.py` covers the PYTEST tier end-to-end through pytester. This
file covers the shared module directly — the half that the CLI root callback consumes and
that no pytester run reaches.

The message body is asserted through BOTH labels because the two tiers are contract-pinned
to different names (`worktree-test-guard` for pytest, `worktree-guard` for the CLI) while
sharing one body. A test that only checked one label would let the tiers silently diverge,
which is the duplication F2c exists to remove.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from hhemt._worktree_guard import assert_worktree_source, worktree_mismatch_message


def _fake_checkout(root):
    """A directory that looks like an hhemt checkout to the guard's predicate."""
    (root / "src" / "hhemt").mkdir(parents=True)
    (root / "src" / "hhemt" / "__init__.py").write_text("", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True, capture_output=True)
    return root


# --------------------------------------------------------------------------- message


def test_message_is_none_when_the_import_resolves_under_expected_src(tmp_path):
    """SATISFYING input: resolved path lies under expected_src -> no message."""
    src = tmp_path / "src"
    (src / "hhemt").mkdir(parents=True)
    resolved = src / "hhemt" / "__init__.py"
    assert worktree_mismatch_message(expected_src=src, force_wrong_src=str(resolved)) is None


def test_message_names_the_resolved_path_when_it_lies_outside(tmp_path):
    """VIOLATING input: resolved path outside expected_src -> the message."""
    msg = worktree_mismatch_message(
        expected_src=tmp_path / "src", force_wrong_src="/tmp/not-the-worktree/src"
    )
    assert msg is not None
    assert "hhemt.__file__ = /tmp/not-the-worktree/src" in msg
    assert str(tmp_path / "src") in msg
    # It must say the run is not evidence about this checkout — that is the whole point.
    assert "not evidence about this checkout" in msg


@pytest.mark.parametrize("label", ["worktree-guard", "worktree-test-guard"])
def test_label_parameterizes_the_prefix_but_not_the_body(tmp_path, label):
    """Both tiers share one body; only the leading label differs.

    `tests/test_worktree_guard.py` asserts the `worktree-test-guard:` spelling verbatim, so
    this pins that the parameter actually reaches the emitted text.
    """
    msg = worktree_mismatch_message(
        expected_src=tmp_path / "src", force_wrong_src="/elsewhere/src", label=label
    )
    assert msg is not None
    assert msg.startswith(f"{label}: hhemt.__file__ = ")
    assert "shared conda env's editable install points elsewhere." in msg


# ------------------------------------------------------------------- assert_worktree


def test_returns_none_when_cwd_is_outside_any_hhemt_checkout(tmp_path, monkeypatch):
    """No expectation to check -> silent no-op. This is the deployed/HPC-scratch case."""
    monkeypatch.delenv("HHEMT_DISABLE_WORKTREE_GUARD", raising=False)
    monkeypatch.chdir(tmp_path)  # a bare dir: not a checkout carrying src/hhemt/__init__.py
    assert assert_worktree_source(strict=True) is None


def test_disable_env_short_circuits_before_any_git_call(tmp_path, monkeypatch):
    monkeypatch.setenv("HHEMT_DISABLE_WORKTREE_GUARD", "1")
    monkeypatch.chdir(_fake_checkout(tmp_path))
    assert assert_worktree_source(strict=True) is None


def test_exits_99_inside_a_checkout_the_import_did_not_honour(tmp_path, monkeypatch):
    """VIOLATING input: cwd IS a checkout, but hhemt resolves elsewhere -> SystemExit(99).

    99 is the conftest guard's code, deliberately shared so both tiers are one signal.
    """
    monkeypatch.delenv("HHEMT_DISABLE_WORKTREE_GUARD", raising=False)
    monkeypatch.delenv("HHEMT_ALLOW_INSTALLED", raising=False)
    monkeypatch.chdir(_fake_checkout(tmp_path))
    with pytest.raises(SystemExit) as exc:
        assert_worktree_source(strict=True)
    assert exc.value.code == 99


def test_allow_installed_downgrades_the_exit_to_a_warning(tmp_path, monkeypatch):
    monkeypatch.delenv("HHEMT_DISABLE_WORKTREE_GUARD", raising=False)
    monkeypatch.setenv("HHEMT_ALLOW_INSTALLED", "1")
    monkeypatch.chdir(_fake_checkout(tmp_path))
    with pytest.warns(UserWarning, match="hhemt.__file__"):
        out = assert_worktree_source(strict=True)
    assert out == (tmp_path / "src").resolve()


def test_non_strict_warns_rather_than_exiting(tmp_path, monkeypatch):
    monkeypatch.delenv("HHEMT_DISABLE_WORKTREE_GUARD", raising=False)
    monkeypatch.delenv("HHEMT_ALLOW_INSTALLED", raising=False)
    monkeypatch.chdir(_fake_checkout(tmp_path))
    with pytest.warns(UserWarning):
        assert assert_worktree_source(strict=False) == (tmp_path / "src").resolve()


def test_the_cli_root_callback_wires_the_guard(tmp_path, monkeypatch):
    """F2b: the Typer app is the single funnel, so --help must still render.

    Guards against the callback raising or swallowing the command tree — the failure that
    would break every CLI invocation at once.
    """
    from typer.testing import CliRunner

    from hhemt.cli import app

    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    assert "run" in result.output
