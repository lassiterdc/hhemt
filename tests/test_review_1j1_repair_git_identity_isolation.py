"""Reviewer tests for the 1j.1 repair round: the stamp-release tests must not depend on host git config.

The problem: tests/test_review_k1_release_stamp.py runs `git commit` inside throwaway repositories,
and a CI runner has no global git identity, so a host-dependent test passes on a laptop and fails
there with `Author identity unknown`. These tests run that module in a child pytest whose HOME and
git configuration are controlled, so the host's own identity cannot mask the dependence.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_MODULE = _REPO / "tests" / "test_review_k1_release_stamp.py"
_IDENTITY_KEYS = ("GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL", "GIT_COMMITTER_NAME", "GIT_COMMITTER_EMAIL")


def _child_pytest(home: Path, *, global_config: str) -> subprocess.CompletedProcess[str]:
    """Run the stamp module in a child pytest on a host whose git identity this call fully controls."""
    env = {k: v for k, v in os.environ.items() if k not in _IDENTITY_KEYS and not k.startswith("GIT_CONFIG")}
    env.update(HOME=str(home), GIT_CONFIG_GLOBAL=global_config, GIT_CONFIG_SYSTEM="/dev/null")
    env.pop("PYTEST_ADDOPTS", None)
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", str(_MODULE)],
        cwd=_REPO,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_stamp_module_is_green_on_a_host_with_no_git_identity_anywhere(tmp_path: Path) -> None:
    """Property: every stamp-module test passes when HOME is empty and global and system git config are /dev/null.

    Class: the four tests of tests/test_review_k1_release_stamp.py, each of which starts git inside a
    temporary repository, directly or through a recipe body run under bash. REGRESSION: red on the tree
    before the repair, where the stamp test died with `Author identity unknown`.
    Why required: the CI runner has no global identity, and a module that reads the developer's own
    config passes locally and fails there, so the laptop can never reveal the defect.
    What kills it: dropping the identity from the environment of any git process the module starts,
    for example returning `_run` to inheriting only the host environment.
    A second correct implementation that still passes: a per-repository `git config user.name` and
    `user.email` written after `git init`, or an autouse fixture that sets the four GIT_* variables.
    """
    home = tmp_path / "home"
    home.mkdir()
    done = _child_pytest(home, global_config="/dev/null")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "4 passed" in done.stdout, done.stdout


def test_stamp_module_is_green_when_the_host_carries_a_global_identity(tmp_path: Path) -> None:
    """Property: the same four tests pass on a host that does have a global identity, as the laptop does.

    Class: the same module, in the other correct host state. Why required: a repair that removed the
    dependence on host config by also forbidding it, for instance by making git ignore global config
    and then failing when the identity is absent from the repository, would pass the test above and
    break a developer host, so the two host states are checked separately.
    What kills it: any change that makes the module fail when `~/.gitconfig` supplies an identity.
    A second correct implementation that still passes: the repair as committed, whose environment
    variables outrank the global file, or a per-repository identity written after `git init`.
    """
    home = tmp_path / "home"
    home.mkdir()
    cfg = tmp_path / "gitconfig"
    cfg.write_text("[user]\n\tname = host\n\temail = host@host\n", encoding="utf-8")
    done = _child_pytest(home, global_config=str(cfg))
    assert done.returncode == 0, done.stdout + done.stderr
    assert "4 passed" in done.stdout, done.stdout
