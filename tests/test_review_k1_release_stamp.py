"""Reviewer tests for chunk 1j.1 (approved shape F2): the release-date coupling.

`just` is not installed on every host, so each test extracts the recipe body from the justfile and
runs it with bash in a throwaway git repository holding the real `scripts/check_release_metadata.py`.
The `just` interpolation `{{VERSION}}` is substituted and `python` is pinned to this interpreter.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
TODAY = "2026-11-13"
YESTERDAY = "2026-11-12"


def _recipe_body(name: str) -> str:
    """Lines of the justfile recipe ``name`` with the shebang line, indentation and interpolation removed."""
    text = (_REPO / "justfile").read_text(encoding="utf-8")
    body = text.split(f"\n{name}:\n", 1)[1].split("\n\n", 1)[0]
    lines = [ln[4:] if ln.startswith("    ") else ln for ln in body.splitlines()]
    lines = [ln for ln in lines if not ln.startswith("#!")]
    return "\n".join(lines).replace("{{VERSION}}", "0.2.0").replace("python scripts/", f"{sys.executable} scripts/")


# A CI runner has no global git identity, so every git process this module starts, including the one
# inside a recipe body run through bash, takes its identity from the environment.
_GIT_IDENTITY_ENV = {
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@t",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@t",
}


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, **_GIT_IDENTITY_ENV}
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, env=env, check=False)


def _repo(tmp_path: Path, *, cff_date: str, heading_date: str) -> Path:
    (tmp_path / "scripts").mkdir()
    shutil.copy(_REPO / "scripts" / "check_release_metadata.py", tmp_path / "scripts" / "check_release_metadata.py")
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\nversion = "0.2.0"\n', encoding="utf-8")
    (tmp_path / "CITATION.cff").write_text(
        f'cff-version: 1.2.0\nversion: "0.2.0"\ndate-released: "{cff_date}"\n', encoding="utf-8"
    )
    (tmp_path / "HISTORY.md").write_text(
        f"## v0.2.0 ({heading_date})\n\nbody\n\n## v0.1.0 (2026-07-14)\n", encoding="utf-8"
    )
    assert _git(tmp_path, "init", "-q").returncode == 0
    _git(tmp_path, "add", "-A")
    assert _git(tmp_path, "commit", "-q", "-m", "base").returncode == 0
    return tmp_path


def _run(root: Path, script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", "set -euo pipefail\n" + script],
        cwd=root,
        capture_output=True,
        text=True,
        env={**os.environ, **_GIT_IDENTITY_ENV},
        check=False,
    )


def test_tag_commitment_guard_covers_the_history_file_the_check_now_reads(tmp_path: Path) -> None:
    """REGRESSION against the tip: `just tag` guards commitment of CITATION.cff and pyproject.toml only,
    but the check reads HISTORY.md too, so a saved-not-committed heading correction passes the check
    while `git tag` archives the stale committed heading. Control: the same uncommitted change to
    CITATION.cff IS refused by the guard, so a zero from the HISTORY.md arm is not a broken harness."""
    guard = next(ln for ln in _recipe_body("tag").splitlines() if ln.startswith("git diff --quiet HEAD --"))
    root = _repo(tmp_path, cff_date=TODAY, heading_date=YESTERDAY)

    cff = root / "CITATION.cff"
    cff.write_text(cff.read_text(encoding="utf-8").replace(TODAY, "2099-01-01"), encoding="utf-8")
    assert _run(root, guard).returncode != 0  # control: the guard fires on an uncommitted CITATION.cff
    _git(root, "checkout", "--", "CITATION.cff")

    history = root / "HISTORY.md"
    history.write_text(history.read_text(encoding="utf-8").replace(YESTERDAY, TODAY), encoding="utf-8")
    check = _run(root, f"{sys.executable} scripts/check_release_metadata.py --release-date {TODAY}")
    assert check.returncode == 0  # the working-tree copy satisfies the check
    assert _run(root, guard).returncode != 0, "uncommitted HISTORY.md heading passes the tag guard"


def test_stamp_release_recipe_commits_both_copies_in_one_commit(tmp_path: Path) -> None:
    """NEW CAPABILITY: wrong implementation = two commits, or a commit that also carries an unrelated
    staged file. Both copies stale; one new commit touches exactly CITATION.cff and HISTORY.md."""
    root = _repo(tmp_path, cff_date=YESTERDAY, heading_date=YESTERDAY)
    (root / "other.txt").write_text("staged\n", encoding="utf-8")
    _git(root, "add", "other.txt")
    done = _run(
        root,
        _recipe_body("stamp-release").replace(
            "check_release_metadata.py", f"check_release_metadata.py --release-date {TODAY}"
        ),
    )
    assert done.returncode == 0, done.stderr
    assert _git(root, "rev-list", "--count", "HEAD").stdout.strip() == "2"
    assert sorted(_git(root, "show", "--name-only", "--format=", "HEAD").stdout.split()) == [
        "CITATION.cff",
        "HISTORY.md",
    ]


def test_stamp_release_recipe_refuses_a_dirty_history_file(tmp_path: Path) -> None:
    """NEW CAPABILITY: wrong implementation = stamp over a dirty HISTORY.md, folding unrelated edits
    into the stamp commit. The commit count must stay 1."""
    root = _repo(tmp_path, cff_date=YESTERDAY, heading_date=YESTERDAY)
    (root / "HISTORY.md").write_text((root / "HISTORY.md").read_text(encoding="utf-8") + "\nedit\n", encoding="utf-8")
    done = _run(root, _recipe_body("stamp-release"))
    assert done.returncode != 0
    assert _git(root, "rev-list", "--count", "HEAD").stdout.strip() == "1"


def test_stamp_release_recipe_is_a_noop_commit_when_already_stamped(tmp_path: Path) -> None:
    """NEW CAPABILITY: wrong implementation = `git commit` on a clean tree, which exits 1 and fails the recipe."""
    root = _repo(tmp_path, cff_date=TODAY, heading_date=TODAY)
    done = _run(
        root,
        _recipe_body("stamp-release").replace(
            "check_release_metadata.py", f"check_release_metadata.py --release-date {TODAY}"
        ),
    )
    assert done.returncode == 0, done.stderr
    assert _git(root, "rev-list", "--count", "HEAD").stdout.strip() == "1"
