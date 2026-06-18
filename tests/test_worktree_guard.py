"""Deterministic validation of the worktree-aware test guard's firing and WARNING branches.

The outer pytest session loads the real rootdir conftest.py; these tests use the
pytester fixture to run *inner* pytest sessions with controlled environment
variables, exercising each branch of the guard without relying on filesystem
tricks or main-tree fallback behavior.
"""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

pytest_plugins = ["pytester"]

_WRONG_SRC = "/tmp/not-the-worktree/src"


def _copy_rootdir_conftest(pytester: pytest.Pytester) -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    src = repo_root / "conftest.py"
    dst = pytester.path / "conftest.py"
    dst.write_text(src.read_text())
    (pytester.path / "src").mkdir(exist_ok=True)
    return dst


def _make_trivial_test(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(test_trivial=dedent("""
        def test_noop():
            assert True
    """))


def test_guard_fires_on_wrong_src(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    """FORCE_WRONG_SRC outside worktree src/ → guard fires with exit 99."""
    _copy_rootdir_conftest(pytester)
    _make_trivial_test(pytester)
    monkeypatch.delenv("HHEMT_DISABLE_WORKTREE_GUARD", raising=False)
    monkeypatch.delenv("HHEMT_ALLOW_INSTALLED", raising=False)
    monkeypatch.setenv("HHEMT_FORCE_WRONG_SRC", _WRONG_SRC)
    result = pytester.runpytest_subprocess("--collect-only", "-s")
    assert result.ret == 99, f"expected exit 99, got {result.ret}"
    combined = "\n".join(result.outlines + result.errlines)
    assert f"worktree-test-guard: hhemt.__file__ = {_WRONG_SRC}" in combined, combined


def test_guard_downgrades_to_warning_with_allow_installed(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    """ALLOW_INSTALLED=1 + FORCE_WRONG_SRC → WARNING emitted, inner exit 0."""
    _copy_rootdir_conftest(pytester)
    _make_trivial_test(pytester)
    monkeypatch.delenv("HHEMT_DISABLE_WORKTREE_GUARD", raising=False)
    monkeypatch.setenv("HHEMT_FORCE_WRONG_SRC", _WRONG_SRC)
    monkeypatch.setenv("HHEMT_ALLOW_INSTALLED", "1")
    result = pytester.runpytest_subprocess("--collect-only", "-s")
    assert result.ret == 0, f"expected exit 0 with WARNING, got {result.ret}"
    combined = "\n".join(result.outlines + result.errlines)
    assert "[worktree-test-guard] WARNING:" in combined, combined
