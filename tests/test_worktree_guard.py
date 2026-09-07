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


def _root_conftest_plugin(pytestconfig):
    """The LOADED repo-root conftest, as a plugin object.

    Retrieved from the plugin manager rather than imported, so the test drives the
    conftest pytest actually loaded for this session.
    """
    for plugin in pytestconfig.pluginmanager.get_plugins():
        mod = getattr(plugin, "__file__", None)
        if mod and mod.endswith("/conftest.py") and "/tests/" not in mod:
            return plugin
    raise AssertionError("root conftest plugin not found")


def test_rootdir_conftest_stands_alone_without_the_tests_package(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ARM 1 -- the root conftest must collect in a dir containing only itself.

    `_copy_rootdir_conftest` writes the conftest and an empty `src/` and NOTHING
    else, which is the shape a copier produces. Anything the conftest imports
    unconditionally at collection time crashes here. Before the guard's arming
    hook was made conditional this returned INTERNALERROR (rc=3) on
    `ModuleNotFoundError: No module named 'tests'`.
    """
    _copy_rootdir_conftest(pytester)
    _make_trivial_test(pytester)
    monkeypatch.setenv("HHEMT_DISABLE_WORKTREE_GUARD", "1")
    monkeypatch.delenv("HHEMT_COMPILE_VENUE", raising=False)
    result = pytester.runpytest_subprocess("--collect-only", "-s")
    assert result.ret == 0, f"root conftest did not stand alone, got {result.ret}"


def test_rootdir_conftest_hook_still_arms_the_compile_guard(
    pytestconfig: pytest.Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ARM 2 -- the DISCRIMINATOR: it drives the HOOK, not the mechanism.

    Arm 1 only measures that nothing crashed, so a hook that returns
    unconditionally passes it while disarming the guard everywhere. This calls
    the loaded root conftest's `pytest_collection_modifyitems` and asserts the
    entry points were replaced -- which fails if the hook returns early.

    THE monkeypatch PRELUDE IS ALSO THE CLEANUP AND MUST NOT BE REMOVED.
    `arm_compile_guard` uses a bare `setattr` that no fixture undoes. Patching
    each entry point FIRST makes monkeypatch record the pre-test value, so
    teardown discards whatever the raw setattr left. Delete the prelude and this
    test poisons every later test in the session with compile refusals.
    """
    from hhemt.system import TRITONSWMM_system
    from tests.fixtures._compile_guard import COMPILE_ENTRY_POINTS

    monkeypatch.delenv("HHEMT_COMPILE_VENUE", raising=False)
    for name in COMPILE_ENTRY_POINTS:
        monkeypatch.setattr(TRITONSWMM_system, name, lambda *_a, **_k: None, raising=True)

    _root_conftest_plugin(pytestconfig).pytest_collection_modifyitems(config=pytestconfig, items=[])

    landed = {n: getattr(TRITONSWMM_system, n).__name__ for n in COMPILE_ENTRY_POINTS}
    assert all(v == "_raise" for v in landed.values()), f"hook did not arm: {landed}"


def _make_trivial_test(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        test_trivial=dedent("""
        def test_noop():
            assert True
    """)
    )


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


def test_guard_downgrades_to_warning_with_allow_installed(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
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
