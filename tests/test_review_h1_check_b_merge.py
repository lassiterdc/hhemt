"""Merge-grading arms for `check-b --range` that the eight remote-exclusion arms do not reach.

WHAT THIS FILE PINS. `tests/test_check_layout_version_merge_range.py` covers a merge in which ONE
file changed on BOTH sides and the merged blob is a third content (arms A, B, E, H). It never builds
a merge in which a layout-relevant file changed on ONE side only, the `develop` side, and the
merge simply carries that change in. The checker graded such a merge against its FIRST parent
alone, so the carried-in blob counted as a change the merge introduced even though `develop`
had already been graded and signed it. When the feature branch's allowlist rows for that path
come last in the merged `_layout_relevant_files.yaml`, the loader's last-wins rule signs the
branch's content and not `develop`'s, and the correct merge failed with no way to repair it
other than appending a further row.

THE TWO ARMS.

* I  `develop` alone changed `src/foo.py` and signed it; the branch's own row for that path comes
     later in the merged allowlist -> must NOT fire
* J  the same history, but the merge hand-edits `src/foo.py` to a content in NEITHER parent
     -> must fire (separates "grade what the merge introduces" from "skip merges")

VENUE. Laptop-safe: every root is under `tmp_path`, the checker is copied into `<repo>/scripts/` so
its `REPO_ROOT` resolves to the fixture, and nothing here compiles, runs a solver or touches the
network.
"""

from __future__ import annotations

from pathlib import Path

from tests.test_check_layout_version_merge_range import (
    FOO,
    _checker,
    _commit,
    _git,
    _make_repo,
    _merge_conflicting_foo,
    _rev,
    _sentinel,
)


def _branch_row_shadows_a_develop_only_change(repo: Path) -> dict[str, str]:
    """`feature` signs `feat`, then reverts foo to base; `develop` alone changes foo to `dev` and signs it.

    Net effect on foo: unchanged on the branch relative to the merge base, changed on `develop`
    only. The branch keeps its `feat` row, which the merged allowlist will list AFTER develop's.
    """
    _git(repo, "checkout", "-q", "-b", "feature")
    _commit(repo, "F1 sign feat", {FOO: "feat\n", "_layout_relevant_files.yaml": _sentinel("feat\n")})
    f2 = _commit(repo, "F2 revert foo, keep the row", {FOO: "base\n"})
    _git(repo, "push", "-q", "origin", "feature")
    _git(repo, "checkout", "-q", "-b", "develop", "main")
    d = _commit(
        repo, "D develop-only change, signed", {FOO: "dev\n", "_layout_relevant_files.yaml": _sentinel("dev\n")}
    )
    _git(repo, "push", "-q", "origin", "develop")
    _git(repo, "checkout", "-q", "feature")
    return {"F2": f2, "D": d}


def test_arm_i_a_develop_only_change_carried_in_by_a_merge_does_not_fire(tmp_path: Path) -> None:
    """Arm I. Property: a merge whose blob equals one parent's is not graded as a change it introduced.

    Class: REGRESSION. Red on the first-parent-only checker, which reads `base -> dev` in the merge's
    first-parent diff and hashes it against the branch's last-wins `feat` row. Reachability:
    a merged `src/foo.py` equal to `develop`'s blob, an allowlist whose last row for the path signs
    other content; artifact: the merge M below, whose second parent holds that same blob. Both-states
    anchor: the verdict turns on which parents hold the merged blob, present in the old and new code
    alike. Why required: the merge is the ordinary way a branch takes `develop`, and a spurious
    failure there can only be answered by a row that signs content already signed. Killed by: grading
    a merge against its first parent alone. A second correct implementation that still passes:
    reading `git diff-tree --cc --name-only` for a merge, which lists only paths differing from
    every parent.
    """
    repo, _ = _make_repo(tmp_path)
    h = _branch_row_shadows_a_develop_only_change(repo)
    m = _merge_conflicting_foo(repo, "develop", _sentinel("dev\n", "feat\n"), merged="dev\n")
    assert _git(repo, "show", f"{m}:{FOO}").stdout == "dev\n"
    assert _git(repo, "show", f"{h['D']}:{FOO}").stdout == "dev\n"
    out = _checker(repo, h["F2"], m, "origin")
    assert out.returncode == 0, out.stdout + out.stderr
    assert m[:9] not in out.stderr, out.stderr


def test_arm_k_a_blob_carried_from_the_third_parent_of_an_octopus_merge_does_not_fire(tmp_path: Path) -> None:
    """Arm K. Property: "changed against EVERY parent" holds for a parent past the second.

    Input class: a merge commit with three parents whose `src/foo.py` equals the THIRD parent's
    blob and differs from the first two. Arms I and J build two-parent merges only, so an
    implementation that intersects with the second parent alone (`other_bases[:1]`) passes both
    and fails here: the second parent holds the base blob, so the path still reads as changed.
    Class: NEW CAPABILITY; the plausible wrong implementation is that second-parent-only
    intersection, red under it (run). Reachability: `git merge -s ours` accepts several heads and
    `git commit` then records every one as a parent; artifact: the `rev-list --parents` assert
    below. Both-states anchor: the verdict turns on which parents hold the merged blob, true under
    the old and the new code alike. Why required: the shape's rule is "every parent", and a
    quantifier the suite exercises at one value only is a property of that value. Killed by:
    intersecting with a subset of the later parents. A second correct implementation that still
    passes: `git diff-tree --cc --name-only`, which lists only paths differing from every parent.
    Enforces shape item C4 (1h.1: "compare against every parent").
    """
    repo, _ = _make_repo(tmp_path)
    h = _branch_row_shadows_a_develop_only_change(repo)
    _git(repo, "checkout", "-q", "-b", "side", "main")
    s = _commit(repo, "S unrelated side change", {"side.txt": "y\n"})
    _git(repo, "push", "-q", "origin", "side")
    _git(repo, "checkout", "-q", "feature")
    _git(repo, "merge", "--no-commit", "--no-ff", "-s", "ours", s, h["D"])
    (repo / FOO).write_text("dev\n")
    (repo / "side.txt").write_text("y\n")
    (repo / "_layout_relevant_files.yaml").write_text(_sentinel("dev\n", "feat\n"))
    _git(repo, "add", "--", FOO, "side.txt", "_layout_relevant_files.yaml")
    _git(repo, "commit", "-q", "--no-edit")
    m = _rev(repo)
    assert _git(repo, "rev-list", "--parents", "-n", "1", m).stdout.split()[1:] == [h["F2"], s, h["D"]]
    assert _git(repo, "show", f"{s}:{FOO}").stdout == "base\n"
    assert _git(repo, "show", f"{m}:{FOO}").stdout == _git(repo, "show", f"{h['D']}:{FOO}").stdout
    out = _checker(repo, h["F2"], m, "origin")
    assert out.returncode == 0, out.stdout + out.stderr
    assert m[:9] not in out.stderr, out.stderr


def test_arm_j_a_merge_that_hand_edits_the_file_to_a_content_in_neither_parent_still_fires(tmp_path: Path) -> None:
    """Arm J. Property: a merge whose blob equals NEITHER parent's is still graded and named.

    Class: NEW CAPABILITY. Plausible wrong implementation: skipping every merge commit, or every
    path changed on `develop`. Killed by: that implementation returning 0. Reachability: the merge's
    `src/foo.py` is `evil`, which neither `feat`-side nor `develop`-side holds, so it is a change
    only the merge introduced; artifact: the `_git show` asserts below. A second correct
    implementation that still passes: a combined diff (`--cc`), which lists a path differing from
    every parent.
    """
    repo, _ = _make_repo(tmp_path)
    h = _branch_row_shadows_a_develop_only_change(repo)
    m = _merge_conflicting_foo(repo, "develop", _sentinel("dev\n", "feat\n"), merged="evil\n")
    parents = _git(repo, "rev-list", "--parents", "-n", "1", m).stdout.split()[1:]
    assert len(parents) == 2
    for parent in parents:
        assert _git(repo, "show", f"{parent}:{FOO}").stdout != "evil\n"
    assert _rev(repo) == m
    out = _checker(repo, h["F2"], m, "origin")
    assert out.returncode == 1, out.stdout + out.stderr
    assert m[:9] in out.stderr, out.stderr
    assert h["D"][:9] not in out.stderr, out.stderr


def test_arm_l_a_develop_only_edit_with_no_signing_row_still_fires_on_the_merge_that_carries_it(tmp_path: Path) -> None:
    """Arm L. Property: a layout file carried in from another parent that NO allowlist row signs still fails.

    Class: REGRESSION, red on 5c2be4fa where the every-parent intersection drops the carried path and
    the merge passes at rc 0 (rc 1 before 5c2be4fa). Input class: an unsigned edit pushed to `develop`
    (a commit that bypassed develop's own gate), merged into a branch that changed another file.
    Reachability: the pushed commit is excluded from grading by the remote exclusion, so the merge is
    the only place the unsigned blob is seen; artifact: the `skipped 1 commit(s)` assert below. Both-states
    anchor: the verdict turns on whether any row for the path signs the carried blob, a property of the
    allowlist and the blob that holds under the old and new code. Why required: no CI workflow runs this
    check, so a develop commit pushed past the pre-push hook is otherwise never graded. Killed by:
    dropping carried paths without a signature check. A second correct implementation that still
    passes: grading a carried path against the signatures of every row, however they are collected.
    Arms I and K are the must-pass controls: they carry a signed blob and stay green.
    """
    repo, _ = _make_repo(tmp_path)
    _git(repo, "checkout", "-q", "-b", "feature")
    f = _commit(repo, "F1 other file", {"other.txt": "o\n"})
    _git(repo, "push", "-q", "origin", "feature")
    _git(repo, "checkout", "-q", "-b", "develop", "main")
    d = _commit(repo, "D unsigned edit", {FOO: "dev\n"})
    _git(repo, "push", "-q", "origin", "develop")
    _git(repo, "checkout", "-q", "feature")
    _git(repo, "merge", "--no-ff", "--no-edit", "develop")
    m = _rev(repo)
    out = _checker(repo, f, m, "origin")
    assert "skipped 1 commit(s)" in out.stdout, out.stdout
    assert out.returncode == 1, out.stdout + out.stderr
    assert m[:9] in out.stderr, out.stderr
    assert d[:9] not in out.stderr, out.stderr
