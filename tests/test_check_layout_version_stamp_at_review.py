"""Reviewer arms for `--stamp PATH --at REV` in `scripts/check_layout_version.py`.

These arms position themselves on inputs the arms in `tests/test_check_layout_version.py` do not reach. That
file's `--at` arms stamp a path that exists in BOTH the working tree and the named commit, from a working tree
whose file is the tip content. Two further input classes decide whether the stamp covers a pushed commit, and each
is a class a natural wrong implementation gets wrong while those arms stay green:

* a path the working tree no longer holds (a later commit deleted or renamed it), and
* a stamp taken while the working tree holds content that is not the commit the row must cover, followed by the
  separate-push route the range venue's failure text itself prescribes.

Laptop-safe: every root is under `tmp_path`, the checker is copied into `<repo>/scripts/` so that its `REPO_ROOT`
resolves to the fixture, and nothing here compiles a solver, runs a simulation or touches the network.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import yaml

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_layout_version.py"
MANIFEST = "_layout_relevant_files.yaml"
_GIT = ["git", "-c", "user.email=t@t", "-c", "user.name=t"]
_WHY = "NON-BREAKING. Comment-only edit; no on-disk change."
_GOVERNED = "layout_relevant:\n  paths:\n    - src/foo.py\n  globs: []\nnon_breaking_allowlist:\n"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run([*_GIT, *args], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()


def _run(repo: Path, *args: str) -> subprocess.CompletedProcess:
    """Run the checker the fixture carries, with the ref pair pre-commit exports removed from the environment."""
    target = repo / "scripts" / "check_layout_version.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(SCRIPT.read_text())
    env = {k: v for k, v in os.environ.items() if not k.startswith("PRE_COMMIT")}
    return subprocess.run([sys.executable, str(target), *args], cwd=repo, capture_output=True, text=True, env=env)


def _init(repo: Path) -> None:
    (repo / "src" / "hhemt" / "version_migration" / "versions").mkdir(parents=True)
    (repo / "tests" / "fixtures" / "legacy_layouts").mkdir(parents=True)
    (repo / MANIFEST).write_text(_GOVERNED)
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)


def _commit(repo: Path, version: int, foo: str | None, *paths: str) -> str:
    """Commit the layout version and the governed file's content (None deletes it), plus any extra paths."""
    (repo / "src" / "hhemt" / "version_migration" / "constants.py").write_text(f"LAYOUT_VERSION: int = {version}\n")
    foo_path = repo / "src" / "foo.py"
    if foo is None:
        foo_path.unlink(missing_ok=True)
    else:
        foo_path.write_text(foo)
    _git(repo, "add", "-A", "src", *paths)
    _git(repo, "commit", "--allow-empty", "-qm", "c")
    return _git(repo, "rev-parse", "HEAD")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _last_row(repo: Path) -> dict:
    return yaml.safe_load((repo / MANIFEST).read_text())["non_breaking_allowlist"][-1]


def test_stamp_at_covers_a_commit_whose_file_the_working_tree_no_longer_holds(tmp_path: Path) -> None:
    """Property: `--stamp PATH --at COMMIT` stamps the blob COMMIT carries when a later commit has deleted PATH, and
    the row it writes makes `check-b --range` pass; without `--at` the same working tree refuses.

    Class: a pushed history in which the governed file changes without a bump and a later bumped commit deletes it,
    so that the file is absent from the working tree at stamp time.
    Why required: the range venue grades each commit's own blob, so the file's absence from the working tree says
    nothing about whether the commit that changed it can be covered; a stamper whose path must exist on disk first
    cannot cover a commit whose file was later removed or renamed.
    Kills it: a stamp that refuses when PATH is absent from the working tree before it reads the named commit.
    Passes under: any implementation that judges existence only at the named commit, however it reads the blob.
    """
    repo = tmp_path / "gone"
    repo.mkdir()
    _init(repo)
    base = _commit(repo, 4, "# v0\n")
    changed = _commit(repo, 4, "# v1\n")
    _commit(repo, 5, None)
    assert not (repo / "src" / "foo.py").exists()

    refused = _run(repo, "--stamp", "src/foo.py", "--why", _WHY)
    assert refused.returncode == 1, refused.stdout + refused.stderr
    assert "does not exist" in refused.stderr, refused.stderr

    stamped = _run(repo, "--stamp", "src/foo.py", "--at", changed, "--why", _WHY)
    assert stamped.returncode == 0, stamped.stdout + stamped.stderr
    assert _last_row(repo)["layout_signature"] == _sha("# v1\n")
    _git(repo, "add", MANIFEST)
    _git(repo, "commit", "-qm", "row")
    graded = _run(repo, "check-b", base, "--range", "--to", "HEAD")
    assert graded.returncode == 0, graded.stdout + graded.stderr


def test_the_split_push_route_in_the_range_text_passes_one_push_at_a_time(tmp_path: Path) -> None:
    """Property: two un-bumped changes to one governed file, each followed directly by a row stamped `--at` that
    change while the working tree already holds the later content, pass `check-b --range` as two successive pushes;
    the same history graded as one push fails.

    Class: a history c1, row1, c2, row2 whose rows were stamped with the working tree dirty at a content other than
    the commit being covered, graded as two pushes and as one.
    Why required: the range venue's failure text tells the stamper this route, and a stamper who follows it gets a
    row per covered commit only if `--at` hashes the named commit and not the working tree; the one-push control
    varies the property the text states (one row covers one content) rather than the token.
    Kills it: a stamp that ignores `--at` (the row for the first change would carry the later content and the first
    push would fail); a range venue that stopped refusing the one-push history.
    Passes under: any implementation that hashes the named commit's blob and keeps one signature per path.
    """
    repo = tmp_path / "split"
    repo.mkdir()
    _init(repo)
    base = _commit(repo, 4, "# v0\n")
    first = _commit(repo, 4, "# v1\n")

    (repo / "src" / "foo.py").write_text("# v2\n")
    stamped = _run(repo, "--stamp", "src/foo.py", "--at", first, "--why", _WHY)
    assert stamped.returncode == 0, stamped.stdout + stamped.stderr
    assert _last_row(repo)["layout_signature"] == _sha("# v1\n")
    _git(repo, "add", MANIFEST)
    _git(repo, "commit", "-qm", "row 1")
    row1 = _git(repo, "rev-parse", "HEAD")

    second = _commit(repo, 4, "# v2\n")
    stamped = _run(repo, "--stamp", "src/foo.py", "--at", second, "--why", _WHY)
    assert stamped.returncode == 0, stamped.stdout + stamped.stderr
    _git(repo, "add", MANIFEST)
    _git(repo, "commit", "-qm", "row 2")
    row2 = _git(repo, "rev-parse", "HEAD")

    one = _run(repo, "check-b", base, "--range", "--to", row1)
    assert one.returncode == 0, one.stdout + one.stderr
    two = _run(repo, "check-b", row1, "--range", "--to", row2)
    assert two.returncode == 0, two.stdout + two.stderr

    together = _run(repo, "check-b", base, "--range", "--to", row2)
    assert together.returncode == 1, together.stdout + together.stderr
