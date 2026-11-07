"""Remote-exclusion arms for `check-b --range` (`PRE_COMMIT_REMOTE_NAME`).

WHAT THIS FILE PINS. At pre-push, `scripts/check_layout_version.py check-b --range` grades every
commit in `PRE_COMMIT_FROM_REF..PRE_COMMIT_TO_REF` against the allowlist loaded at the range's
upper end. A correct merge of `develop` into a branch therefore failed: a commit imported from
`develop` was signed by a row that a later row has since replaced under the loader's last-wins
rule. The repair grades only the commits NOT reachable from `refs/remotes/{name}/*` when
pre-commit names the remote, and grades the whole range when it does not.

THE SEVEN ARMS, and what each one separates. Each builds a `tmp_path` repository with a bare
remote named `origin` and runs the checker as a child process with `PRE_COMMIT_REMOTE_NAME` in
the child environment:

* A  a re-signed merge of a pushed `develop`, one file changed on BOTH sides  -> must NOT fire
* B  a merged blob that no row signs                                          -> must fire
* C  an unpushed side-branch commit with no signature                         -> must fire
* D  a side branch bumps LAYOUT_VERSION, then changes an unsigned layout file -> must fire
     (separates the per-commit remote exclusion from a first-parent walk, which would grade only
     the merge, see the bump, and skip it)
* E  arm B's history, every commit reachable from `origin/*`, variable UNSET  -> must fire
     (separates "grade everything" from the bare `--remotes` fallback)
* F  an unsigned commit reachable only from `upstream/*`, variable `origin`   -> must fire
     (separates `--remotes={name}` from bare `--remotes`)
* G  the first-push shape pre-commit produces, over a merge of a pushed `develop`
     -> must NOT fire

VENUE. Laptop-safe: every root is under `tmp_path`, the checker is copied into `<repo>/scripts/`
so its `REPO_ROOT` resolves to the fixture, and nothing here compiles or runs a solver or touches
the network (the remotes are bare repositories on local disk).
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_layout_version.py"

CONSTANTS_REL = "src/hhemt/version_migration/constants.py"
FOO = "src/foo.py"


def _sig(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _sentinel(*signed: str) -> str:
    """A sentinel whose allowlist appends one signature row per blob text, in order (last wins)."""
    head = "layout_relevant:\n  paths:\n    - src/foo.py\n  globs: []\n"
    if not signed:
        return head + "non_breaking_allowlist: []\n"
    rows = "".join(
        f"  - path: src/foo.py\n    justification: row {i}\n    layout_signature: {_sig(text)}\n"
        for i, text in enumerate(signed)
    )
    return head + "non_breaking_allowlist:\n" + rows


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
        cwd=repo,
        check=check,
        capture_output=True,
        text=True,
    )


def _rev(repo: Path, ref: str = "HEAD") -> str:
    return _git(repo, "rev-parse", ref).stdout.strip()


def _commit(repo: Path, msg: str, files: dict[str, str], *, version: int | None = None) -> str:
    """Write `files` (and, if given, the LAYOUT_VERSION constant), commit exactly those paths."""
    if version is not None:
        files = {**files, CONSTANTS_REL: f"LAYOUT_VERSION: int = {version}\n"}
    for rel, content in files.items():
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    _git(repo, "add", "--", *files)
    _git(repo, "commit", "-q", "--allow-empty", "-m", msg)
    return _rev(repo)


def _make_repo(tmp_path: Path) -> tuple[Path, str]:
    """`work` with commit c0 on `main` pushed to a bare `origin`; a bare `upstream` is also wired."""
    repo = tmp_path / "work"
    repo.mkdir()
    for name in ("origin", "upstream"):
        subprocess.run(["git", "init", "-q", "--bare", str(tmp_path / f"{name}.git")], check=True)
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    for name in ("origin", "upstream"):
        _git(repo, "remote", "add", name, str(tmp_path / f"{name}.git"))
    c0 = _commit(repo, "c0", {FOO: "base\n", "_layout_relevant_files.yaml": _sentinel()}, version=4)
    _git(repo, "push", "-q", "origin", "main")
    return repo, c0


def _checker(repo: Path, from_ref: str, to_ref: str, remote: str | None) -> subprocess.CompletedProcess:
    """Run `check-b --range` as pre-commit would: the ref pair and remote name in the environment.

    `remote=None` is explicitly BLANK rather than merely absent, because an ambient value in the
    outer environment (these tests may themselves run under a pre-push hook) would otherwise
    silently become the population under test.
    """
    target = repo / "scripts" / "check_layout_version.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(SCRIPT.read_text())
    env = {
        **os.environ,
        "PRE_COMMIT_FROM_REF": from_ref,
        "PRE_COMMIT_TO_REF": to_ref,
        "PRE_COMMIT_REMOTE_NAME": remote or "",
    }
    return subprocess.run(
        [sys.executable, str(target), "check-b", "--range"],
        cwd=repo,
        capture_output=True,
        text=True,
        env=env,
    )


def _merge_conflicting_foo(repo: Path, other: str, resolved_sentinel: str, merged: str = "merged\n") -> str:
    """Merge `other` into the current branch, resolving the foo + sentinel conflicts by hand."""
    _git(repo, "merge", "--no-commit", "--no-ff", other, check=False)
    (repo / FOO).write_text(merged)
    (repo / "_layout_relevant_files.yaml").write_text(resolved_sentinel)
    _git(repo, "add", "--", FOO, "_layout_relevant_files.yaml")
    _git(repo, "commit", "-q", "--no-edit")
    return _rev(repo)


def _diverged_history(repo: Path) -> dict[str, str]:
    """feature (pushed) and develop (pushed) both change foo; returns their shas."""
    _git(repo, "checkout", "-q", "-b", "feature")
    f = _commit(repo, "F", {FOO: "feat\n", "_layout_relevant_files.yaml": _sentinel("feat\n")})
    _git(repo, "push", "-q", "origin", "feature")
    _git(repo, "checkout", "-q", "-b", "develop", "main")
    d = _commit(repo, "D", {FOO: "dev\n", "_layout_relevant_files.yaml": _sentinel("dev\n")})
    _git(repo, "push", "-q", "origin", "develop")
    _git(repo, "checkout", "-q", "feature")
    return {"F": f, "D": d}


def test_arm_a_a_resigned_merge_of_a_pushed_develop_does_not_fire(tmp_path: Path) -> None:
    """Arm A. Property: a re-signed merge of a pushed `develop`, where one file changed on both sides, passes.

    Class: REGRESSION. Red on the pre-fix checker, which grades the
    imported `develop` commit against the tip allowlist whose last-wins row now signs the merged blob.
    Why required: that is the failure a correct merge of `develop` hit before this change. Killed by:
    removing `--not --remotes={name}` from `_commits_in_range`, or never passing the name. A second
    correct implementation that still passes: grading each commit against the allowlist at its OWN
    tree, since develop's commit signs its own blob.
    """
    repo, _ = _make_repo(tmp_path)
    h = _diverged_history(repo)
    m = _merge_conflicting_foo(repo, "develop", _sentinel("dev\n", "feat\n"))
    r = _commit(repo, "re-sign the merge", {"_layout_relevant_files.yaml": _sentinel("dev\n", "feat\n", "merged\n")})
    out = _checker(repo, h["F"], r, "origin")
    assert out.returncode == 0, out.stdout + out.stderr
    assert "skipped 1 commit(s)" in out.stdout, out.stdout
    assert "grading 2" in out.stdout, out.stdout
    for sha in (h["D"], m, r):
        assert sha[:9] not in out.stderr, out.stderr


def test_arm_b_a_merged_blob_that_no_row_signs_fires(tmp_path: Path) -> None:
    """Arm B. Property: a merged blob that no row signs still fails, and the pushed commit is not named.

    Class: NEW CAPABILITY. Plausible wrong implementation: skip the whole range when any commit is on
    a remote. Killed by: that implementation returning 0. A second correct implementation that still
    passes: grading only the merge commit's first-parent diff.
    """
    repo, _ = _make_repo(tmp_path)
    h = _diverged_history(repo)
    m = _merge_conflicting_foo(repo, "develop", _sentinel("dev\n", "feat\n"))
    out = _checker(repo, h["F"], m, "origin")
    assert out.returncode == 1, out.stdout + out.stderr
    assert m[:9] in out.stderr, out.stderr
    assert h["D"][:9] not in out.stderr, "the pushed develop commit must not be named: " + out.stderr


def test_arm_c_an_unpushed_side_branch_commit_without_a_signature_fires(tmp_path: Path) -> None:
    """Arm C. Property: an unpushed layout commit with no signature fires even with a remote named.

    Class: NEW CAPABILITY. Plausible wrong implementation: exclude by branch name rather than by
    reachability from the remote's tracking refs. Killed by: that implementation returning 0. A second
    correct implementation that still passes: excluding via `git merge-base --is-ancestor` per commit.
    """
    repo, c0 = _make_repo(tmp_path)
    _git(repo, "checkout", "-q", "-b", "feature")
    x = _commit(repo, "X unsigned", {FOO: "unsigned\n"})
    out = _checker(repo, c0, x, "origin")
    assert out.returncode == 1, out.stdout + out.stderr
    assert x[:9] in out.stderr, out.stderr


def test_arm_d_a_bump_on_a_side_branch_does_not_disarm_its_next_unsigned_commit(tmp_path: Path) -> None:
    """Arm D. Property: a bump on a merged side branch does not disarm the next unsigned commit on it.

    Class: NEW CAPABILITY. Plausible wrong implementation: a first-parent walk, which grades only the
    merge, sees the version bump and skips. Killed by: that implementation returning 0. A second correct
    implementation that still passes: walking every commit and excluding by remote reachability.
    """
    repo, c0 = _make_repo(tmp_path)
    _git(repo, "checkout", "-q", "-b", "side")
    s1 = _commit(repo, "S1 bump", {"unrelated.txt": "u\n"}, version=5)
    s2 = _commit(repo, "S2 unsigned layout change", {FOO: "side\n"})
    _git(repo, "checkout", "-q", "-b", "feature", "main")
    _commit(repo, "F1", {"other.txt": "o\n"})
    _git(repo, "merge", "--no-ff", "--no-edit", "side")
    m = _rev(repo)
    out = _checker(repo, c0, m, "origin")
    assert out.returncode == 1, out.stdout + out.stderr
    assert s2[:9] in out.stderr, out.stderr
    assert s1[:9] not in out.stderr, out.stderr
    # The merge itself carries the v4 -> v5 transition and is skipped as a bump. A first-parent
    # walk would grade ONLY the merge, take that skip, and pass: this is what the arm separates.
    assert f"check-b: {m[:9]}: LAYOUT_VERSION bumped" in out.stdout, out.stdout


def test_arm_e_every_commit_on_a_remote_ref_but_variable_unset_still_fires(tmp_path: Path) -> None:
    """Arm E. Property: with the variable unset nothing is excluded, even when every commit is on a remote ref.

    Class: NEW CAPABILITY. Plausible wrong implementation: a bare `--not --remotes` fallback when the
    name is empty, which would drop every commit here. Killed by: that implementation returning 0. A
    second correct implementation that still passes: raising when the variable is unset and a range is
    given, provided the one-commit default population is kept.
    """
    repo, _ = _make_repo(tmp_path)
    h = _diverged_history(repo)
    m = _merge_conflicting_foo(repo, "develop", _sentinel("dev\n", "feat\n"))
    _git(repo, "push", "-q", "origin", "feature")
    reachable = _git(repo, "rev-list", m, "--not", "--remotes=origin").stdout.strip()
    assert reachable == "", "the fixture must have every commit reachable from origin/*: " + reachable
    out = _checker(repo, h["F"], m, None)
    assert out.returncode == 1, out.stdout + out.stderr
    assert m[:9] in out.stderr, out.stderr
    assert h["D"][:9] in out.stderr, "nothing may be excluded, so the pushed develop commit is graded: " + out.stderr


def test_arm_f_a_commit_reachable_only_from_another_remote_still_fires(tmp_path: Path) -> None:
    """Arm F. Property: a commit reachable only from another remote's refs is still graded.

    Class: NEW CAPABILITY. Plausible wrong implementation: bare `--remotes` instead of
    `--remotes={name}`, which excludes the `upstream`-only commit. Killed by: that implementation
    returning 0. A second correct implementation that still passes: excluding by an explicit list of
    `refs/remotes/{name}/*` tips given to `rev-list`.
    """
    repo, c0 = _make_repo(tmp_path)
    _git(repo, "push", "-q", "upstream", "main")
    _git(repo, "checkout", "-q", "-b", "feature")
    u = _commit(repo, "U unsigned", {FOO: "unsigned\n"})
    _git(repo, "push", "-q", "upstream", "feature")
    bare = _git(repo, "rev-list", u, "--not", "--remotes").stdout.strip()
    assert bare == "", "the fixture must make the bare --remotes form exclude U: " + bare
    out = _checker(repo, c0, u, "origin")
    assert out.returncode == 1, out.stdout + out.stderr
    assert u[:9] in out.stderr, out.stderr


def test_arm_g_the_first_push_shape_over_a_merge_of_a_pushed_develop_does_not_fire(tmp_path: Path) -> None:
    """Arm G. Property: the first-push population over a merge of a pushed `develop` passes.

    Class: REGRESSION. Red on the pre-fix checker, which grades the two pushed `develop` commits
    against a tip allowlist whose last-wins row signs only the second. Why required: a new branch is
    the commonest push shape. Killed by: computing the exclusion only for an existing remote branch.
    A second correct implementation that still passes: the same exclusion expressed as an explicit
    commit list from `git rev-list ... --not --remotes={name}` passed to the walk.
    """
    repo, _ = _make_repo(tmp_path)
    _git(repo, "checkout", "-q", "-b", "develop")
    pushed_first = _commit(repo, "develop one", {FOO: "dev1\n", "_layout_relevant_files.yaml": _sentinel("dev1\n")})
    pushed_second = _commit(
        repo, "develop two", {FOO: "dev2\n", "_layout_relevant_files.yaml": _sentinel("dev1\n", "dev2\n")}
    )
    _git(repo, "push", "-q", "origin", "develop")
    _git(repo, "checkout", "-q", "-b", "feature", "main")
    _commit(repo, "F1", {"other.txt": "o\n"})
    _git(repo, "merge", "--no-ff", "--no-edit", "develop")
    m = _rev(repo)
    # pre-commit's new-branch population: the parent of the earliest commit on no origin ref.
    first = _git(repo, "rev-list", m, "--topo-order", "--reverse", "--not", "--remotes=origin").stdout.split()[0]
    from_ref = _rev(repo, f"{first}^")
    out = _checker(repo, from_ref, m, "origin")
    assert out.returncode == 0, out.stdout + out.stderr
    for sha in (pushed_first, pushed_second, m):
        assert sha[:9] not in out.stderr, out.stderr


def test_arm_h_variable_absent_from_the_environment_grades_everything(tmp_path: Path) -> None:
    """Arm H. Property: with PRE_COMMIT_REMOTE_NAME ABSENT (not blank) nothing is excluded and the run does not crash.

    Class: NEW CAPABILITY. Plausible wrong implementation: `os.environ[PRE_PUSH_REMOTE_ENV]`, which
    raises KeyError on every manual `pre-commit run --hook-stage pre-push` where pre-commit exports
    no remote name. Why required: the hook comment promises that a manual run keeps working and grades
    the whole range. Killed by: that implementation exiting with a traceback and naming no commit. A
    second correct implementation that still passes: `if (remote := os.environ.get(PRE_PUSH_REMOTE_ENV)):`.
    """
    repo, _ = _make_repo(tmp_path)
    h = _diverged_history(repo)
    m = _merge_conflicting_foo(repo, "develop", _sentinel("dev\n", "feat\n"))
    _git(repo, "push", "-q", "origin", "feature")
    target = repo / "scripts" / "check_layout_version.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(SCRIPT.read_text())
    env = {k: v for k, v in os.environ.items() if k != "PRE_COMMIT_REMOTE_NAME"}
    env.update({"PRE_COMMIT_FROM_REF": h["F"], "PRE_COMMIT_TO_REF": m})
    assert "PRE_COMMIT_REMOTE_NAME" not in env
    out = subprocess.run(
        [sys.executable, str(target), "check-b", "--range"], cwd=repo, capture_output=True, text=True, env=env
    )
    assert "Traceback" not in out.stderr, out.stderr
    assert out.returncode == 1, out.stdout + out.stderr
    assert m[:9] in out.stderr, out.stderr
    assert h["D"][:9] in out.stderr, out.stderr
