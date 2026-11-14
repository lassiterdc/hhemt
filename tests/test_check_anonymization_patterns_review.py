"""Tests for the anonymization guards' refusal of an unusable carrier path.

One class is covered here: a carrier path on which the filesystem call itself raises an OS error
that ``pathlib`` does not swallow. Every value planted is built at run time from fragments, and no
producer-specific value appears in this file.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import scripts.check_anonymization as guard  # repo root is on sys.path under pytest

REPO_ROOT = Path(guard.__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _isolated_git_and_env(tmp_path, monkeypatch):
    """No developer carrier or git config leaks into a test."""
    monkeypatch.delenv("HHEMT_ANONYMIZATION_BLOCKLIST", raising=False)
    empty_cfg = tmp_path / "empty_gitconfig"
    empty_cfg.write_text("", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(empty_cfg))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("HOME", str(tmp_path / "home_dir"))


def _unusable_carrier_path(state: str, tmp_path: Path) -> tuple[Path, Path | None]:
    """A carrier path on which ``stat`` raises an errno that ``pathlib.Path.is_file`` re-raises.

    Returns the path and a directory whose permissions the caller must restore (or None).
    ``name_too_long`` raises ENAMETOOLONG and ``unsearchable_parent`` raises EACCES; ``pathlib``
    swallows only ENOENT, ENOTDIR, EBADF and ELOOP, so both escape as a bare OSError.
    """
    if state == "name_too_long":
        return tmp_path / ("a" * 300) / "carrier.txt", None
    locked = tmp_path / "locked"
    locked.mkdir()
    (locked / "carrier.txt").write_text("# synthetic\n" + "\n".join(f"ZZREVIEWTOKEN{i:02d}" for i in range(8)) + "\n")
    locked.chmod(0o000)
    return locked / "carrier.txt", locked


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    (root / "ok.txt").write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    return root


@pytest.mark.parametrize("state", ["name_too_long", "unsearchable_parent"])
def test_a_carrier_path_whose_stat_raises_is_a_named_refusal_in_both_guards(tmp_path, monkeypatch, state) -> None:
    """Property: a configured carrier whose path makes the filesystem call raise an OS error is a
    refusal that names the tier and the path, in the file guard and in the commit-message guard, and
    never an uncaught traceback.

    Class: every carrier path on which ``stat`` raises an errno that ``pathlib`` does not ignore. Two
    members are built: a path component longer than the filesystem allows (ENAMETOOLONG) and a path
    under a directory the process cannot search (EACCES, skipped for the superuser, which searches
    every directory). The approved carrier-resolution rule says a mistyped or unusable carrier is
    "never as a traceback or a bare OSError", and its 22-state space has no state that reaches this
    branch, so the generated 11,616-cell test cannot see it.

    Why required: a traceback and a refusal both exit non-zero, but only the refusal tells an
    operator which tier and which path to fix, and the pre-commit and CI venues show the traceback
    instead of the cause.

    Killed by: guarding ``is_file()`` in ``load_blocklist`` (and in the exemptions loader) with
    ``try/except OSError`` and routing the error to a refusal. A second correct implementation, which
    this test must also pass: calling ``os.stat`` inside a ``try`` and mapping every OSError to the same
    refusal text, because the assertions look at the refusal's tier, its path and the absence of an
    uncaught exception, not at its wording.

    Reachability: ``$HHEMT_ANONYMIZATION_BLOCKLIST`` set to such a path reaches ``load_blocklist``
    through ``_resolve_carrier``; without the guard, ``load_blocklist`` surfaces the bare OSError for both
    members.
    """
    if state == "unsearchable_parent" and os.geteuid() == 0:
        pytest.skip("the superuser searches every directory, so the state cannot be built")
    carrier, restore = _unusable_carrier_path(state, tmp_path)
    try:
        monkeypatch.setenv("HHEMT_ANONYMIZATION_BLOCKLIST", str(carrier))

        # In process: the refusal is a SystemExit naming the tier and the path.
        with pytest.raises(SystemExit) as exc:
            guard.main(["--root", str(_repo(tmp_path))])
        message = str(exc.value.code)
        assert guard.TIER_ENV in message and str(carrier) in message
        assert "ZZREVIEWTOKEN" not in message

        # Through the real scripts: a non-zero exit, the path in stderr, no traceback.
        msg_file = tmp_path / "msg.txt"
        msg_file.write_text("feat: tidy the thing\n", encoding="utf-8")
        for script, args in (
            ("check_anonymization.py", ["--root", str(tmp_path / "repo")]),
            ("check_commit_message_anonymization.py", [str(msg_file)]),
        ):
            proc = subprocess.run(
                [sys.executable, str(REPO_ROOT / "scripts" / script), *args],
                capture_output=True,
                text=True,
                cwd=tmp_path,
                env=dict(os.environ),
            )
            assert proc.returncode != 0, script
            assert "Traceback" not in proc.stderr, (script, proc.stderr[-300:])
            assert str(carrier) in proc.stderr, script
    finally:
        if restore is not None:
            restore.chmod(0o755)
