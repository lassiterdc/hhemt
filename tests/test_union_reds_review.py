"""Reviewer measurements for change 1118-0359 (the union-red fixes).
LAPTOP-SAFE: nothing clones, compiles or runs a solver.

Two groups.

1. The hook-env exception. `tests/test_hook_env_review.py::_DELIBERATE_TAIL_CHANGES` turns one red test green by an
   exception, so the claim "no assertion was loosened" is measured here by perturbation: the exception must be NEEDED
   (without it exactly the one entry differs), and it must NOT launder a revert of that entry, another tail edit of
   that entry, an edit of another entry's tail, or an edit of another field of that entry.
2. The outfall tier. The ten tests are the same tests (AST of every `test_` function equal to the one on the base
   commit), and the warm target fails (never passes vacuously) when the canonical store is unavailable.
"""

from __future__ import annotations

import ast
import subprocess
import warnings

import pytest

from tests import test_hook_env_review as hook
from tests.fixtures import _triton_source_cache as tsc

_ENTRY = "scripts/hook_env.sh python scripts/check_layout_version.py check-a --range\n"
_ID = "check-layout-version-a-range"


def _grade(text: str, tmp_path, monkeypatch, exceptions=None):
    """Run the graded test against `text` as the pre-commit config; return its failure message,
    or None when it passes."""
    cfg = tmp_path / "pre-commit.yaml"
    cfg.write_text(text, encoding="utf-8")
    monkeypatch.setattr(hook, "_CONFIG", cfg)
    if exceptions is not None:
        monkeypatch.setattr(hook, "_DELIBERATE_TAIL_CHANGES", exceptions)
    try:
        hook.test_the_rewrite_changed_each_entry_only_by_its_prefix_and_nothing_else_about_any_hook()
    except AssertionError as exc:
        return str(exc)
    return None


def _live() -> str:
    return hook._CONFIG.read_text(encoding="utf-8")


def _once(text: str, old: str, new: str) -> str:
    assert text.count(old) == 1, (old, text.count(old))
    return text.replace(old, new)


def test_control_the_live_config_passes_and_the_matcher_finds_the_entry_once(tmp_path, monkeypatch):
    """Property: the perturbations below edit a real, unique line, and the unedited config is green."""
    text = _live()
    assert text.count(_ENTRY) == 1
    assert _grade(text, tmp_path, monkeypatch) is None


def test_the_exception_is_needed_for_exactly_one_entry(tmp_path, monkeypatch):
    """Property: with no exception, the one differing entry is the range entry and nothing else."""
    message = _grade(_live(), tmp_path, monkeypatch, exceptions={})
    assert message is not None and _ID in message, message
    others = [h for h in hook._local_hooks(_live()) if h["id"] != _ID and f"'{h['id']}'" in message]
    assert others == [], [h["id"] for h in others]


@pytest.mark.parametrize(
    ("label", "edit"),
    [
        ("revert the entry to the literal HEAD~1", lambda t: _once(t, _ENTRY, _ENTRY.rstrip("\n") + " HEAD~1\n")),
        ("another tail edit of that entry", lambda t: _once(t, _ENTRY, _ENTRY.rstrip("\n") + " HEAD~2\n")),
        (
            "a tail edit of ANOTHER entry",
            lambda t: _once(
                t,
                "scripts/hook_env.sh python scripts/check_layout_version.py check-b --range\n",
                "scripts/hook_env.sh python scripts/check_layout_version.py check-b --range --extra\n",
            ),
        ),
    ],
)
def test_the_exception_does_not_launder_any_other_tail_change(label, edit, tmp_path, monkeypatch):
    """Property: each edit is still reported, so the test still fails on it."""
    diffs = _grade(edit(_live()), tmp_path, monkeypatch)
    assert diffs, label


def test_the_exception_does_not_launder_a_change_of_another_field_of_that_entry(tmp_path, monkeypatch):
    """Property: `stages` of the range entry is still compared, so changing it still fails."""
    text = _live()
    head, sep, tail = text.partition(_ENTRY)
    assert sep
    changed = head + sep + _once(tail[:400], "stages: [pre-push]", "stages: [pre-commit]") + tail[400:]
    assert _grade(changed, tmp_path, monkeypatch), "changing stages of the range entry did not fail the test"


def test_the_ten_outfall_tests_are_unchanged_as_syntax():
    """Property: every `test_` function of the outfall file has the AST it had on the base commit."""
    base = "2ff8fe09"
    done = subprocess.run(
        ["git", "show", f"{base}:tests/test_synth_open_water_outfall_review.py"],
        cwd=hook._REPO,
        capture_output=True,
        text=True,
    )
    if done.returncode != 0:
        pytest.skip(f"base commit {base} is not reachable from this clone")
    now = (hook._REPO / "tests" / "test_synth_open_water_outfall_review.py").read_text(encoding="utf-8")

    def tests(src):
        return {
            n.name: ast.dump(n)
            for n in ast.walk(ast.parse(src))
            if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")
        }

    old, new = tests(done.stdout), tests(now)
    assert len(old) >= 9, len(old)  # must-be-nonzero control: the matcher found the tests
    assert old == new, sorted(k for k in old if old[k] != new.get(k))


def test_the_warm_target_fails_and_does_not_pass_when_the_canonical_store_is_unavailable(tmp_path, monkeypatch):
    """Property: no canonical store means the tier is absent, and the warm test reports that as a failure."""
    from tests import test_synth_00_compile_models as warm

    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv(tsc._DISABLE_ENV, raising=False)
    monkeypatch.delenv(tsc._SUITE_ROLE_ENV, raising=False)
    monkeypatch.setattr(tsc.cache_paths, "slug_runs_root", lambda slug: tmp_path / slug)

    def unavailable(*, pin=tsc.TRITON_PIN):
        raise OSError("no network")

    monkeypatch.setattr(tsc, "ensure_canonical", unavailable)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(AssertionError, match="is not a checkout at"):
            warm.test_provision_the_open_water_outfall_solver_tier()


def test_the_warm_target_skips_under_the_opt_out(monkeypatch):
    """Property: this is the one path where the warm test cannot report the tier absent; it skips, loudly."""
    from tests import test_synth_00_compile_models as warm

    monkeypatch.setenv(tsc._DISABLE_ENV, "1")
    with pytest.raises(pytest.skip.Exception):
        warm.test_provision_the_open_water_outfall_solver_tier()
