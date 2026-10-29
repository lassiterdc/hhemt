"""Reviewer-authored arms for the `--stamp` mode of `scripts/check_layout_version.py`.

These are the REVIEWING seat's tests for plan-and-implement run `10-29_0236` (small
correctness fixes), work package `2b`, chunk `2b.1` (item `1b.5`, W-A: the stamp
command). They are deliberately DISJOINT from the coder's
`tests/test_check_layout_version.py`: that file grades the implementation the coder
wrote, this one grades the approved criterion's behaviour on inputs the coder's arms
do not position themselves on.

WHY A SEPARATE FILE RATHER THAN MORE ARMS IN THE CODER'S. Two reasons, both about
which party the failure belongs to. The coder's four arms are the chunk's done-when
and must stay the coder's to green; a reviewer arm that landed beside them would join
a file the reviewer does not own, and the next round could not tell whose red it is.
And the reviewer's write set at Build is exactly one declared path — the reviewer test
path — so a second production path or a second test file would be a partition
violation rather than a contribution.

WHAT IS RAISED, AND WHAT IS NOT. Two properties are raised below, one test each; both
were measured RED against the tree this turn read, with a named discriminating change
to the production code run against a PATCHED COPY of the checker (never against the
coder's file) to show the arm greens under a correct implementation. The four
done-when properties the shape states for this chunk (a row that parses, resolves and
carries `--why`; refusal with no write when `--why` is absent; a textual edit that
preserves comments and block scalars; append rather than in-place mutation) were
verified independently — by the coder's own arms, by this seat's probes over a copy of
the real 4,411-line manifest, and by probes over mixed block styles and interleaved
comments — and are NOT duplicated here: one class, one test, and the coder's arms
already own them.

VENUE. Laptop-safe and nothing else: every root is under `tmp_path`, the checker is
copied into `<repo>/scripts/` so that its `REPO_ROOT = Path(__file__).resolve().parent.parent`
resolves to the fixture, and nothing here compiles a solver, runs a simulation,
touches the network, or needs git.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import yaml

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_layout_version.py"
MANIFEST_NAME = "_layout_relevant_files.yaml"

# Deliberately hex-with-letters rather than a quoted all-digit string: `_load_allowlist`
# refuses a `layout_signature` that YAML parsed as an int, so an all-digit fixture would
# be testing the loader's type gate instead of the behaviour under review.
_EARLIER_SIGNATURE = "5a5a5a5a" * 8

# A manifest in the shape the real one has: a header comment, a `layout_relevant` block,
# and an allowlist whose first row carries its own signature. The flow form `[]` is NOT
# used here on purpose -- `--stamp` appends a block item, so a `[]` fixture would make
# every arm below exercise the refusal path instead of the append path.
_MANIFEST = (
    "# Sentinel list for scripts/check_layout_version.py check-b.\n"
    "# A change to any path below WITHOUT a LAYOUT_VERSION bump fails CI.\n"
    "layout_relevant:\n"
    "  paths:\n"
    "    - src/foo.py\n"
    "  globs: []\n"
    "non_breaking_allowlist:\n"
    "  - path: src/foo.py\n"
    f"    layout_signature: {_EARLIER_SIGNATURE}\n"
    "    justification: >-\n"
    "      EARLIER CHANGE. Comment-only edit, so nothing on disk moved.\n"
)

_DEFAULT_FILES = {"src/foo.py": "# changed\n"}


def _sha256(path: Path) -> str:
    """The digest `--stamp` is required to record, computed here independently of it."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_stamp_repo(tmp_path: Path, files: dict[str, str] | None = None) -> Path:
    """Build the fixture repo and return its root.

    `--stamp` is pure file-in/file-out, so the fixture needs no git and no config: only
    a writable tree, the files the target paths name, and the manifest where the checker
    resolves it.
    """
    for rel, content in (files or _DEFAULT_FILES).items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    (tmp_path / MANIFEST_NAME).write_text(_MANIFEST)
    return tmp_path


def _run(repo: Path, *args: str) -> subprocess.CompletedProcess:
    """Invoke the checker as a subprocess from inside the fixture repo.

    The copy is refreshed on every call so an arm observes the tree under review rather
    than a stale copy left by an earlier arm in the same session.
    """
    target = repo / "scripts" / "check_layout_version.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(SCRIPT.read_text())
    return subprocess.run([sys.executable, str(target), *args], cwd=repo, capture_output=True, text=True)


def _last_row(repo: Path) -> dict:
    """The last `non_breaking_allowlist` row of the fixture manifest, parsed."""
    parsed = yaml.safe_load((repo / MANIFEST_NAME).read_text())
    rows = parsed.get("non_breaking_allowlist") or []
    assert rows, "the fixture manifest carries no allowlist rows"
    return rows[-1]


def test_stamp_records_a_multiline_why_verbatim(tmp_path: Path) -> None:
    """Property: a `--why` carrying a line break is recorded on the appended row as the
    same text, line breaks included.

    Class: any `--why` whose value contains a newline. The CLI accepts one — quoting a
    file's contents into `--why "$(cat note.txt)"` is the shape — so this is a class the
    suite can enumerate, not a single artifact.

    Why it is required, as a consequence a reader can check against the code: the
    justification is the whole audit trail a human reads when adjudicating a
    change-scoped exemption, and `--why`'s own help text says it is "the justification
    recorded on the row". A rendering that folds the supplied text into one line records
    a DIFFERENT string than the operator wrote, silently — and a justification whose
    bullet list has been run together is still non-empty, so `_load_allowlist` accepts
    it and nothing downstream reports the loss.

    What kills it: emitting the justification through a folded (`>-`) block scalar, which
    turns every single newline into a space and every blank line into one newline.

    A second correct implementation under which this still passes: a literal (`|-`) block
    scalar, or a double-quoted scalar with escaped newlines. Both parse back to exactly
    the supplied text, and both leave the coder's present assertions (single-line `--why`
    round-trips) green — verified by running this arm against a copy of the checker with
    `justification: >-` changed to `justification: |-`.
    """
    repo = _make_stamp_repo(tmp_path)
    why = (
        "NON-BREAKING. Re-stamping the manifest row for this module.\n"
        "- no on-disk name, path or group moves\n"
        "- the signature re-fires on the next content change"
    )

    out = _run(repo, "--stamp", "src/foo.py", "--why", why)
    assert out.returncode == 0, out.stdout + out.stderr

    row = _last_row(repo)
    assert row["path"] == "src/foo.py"
    assert row["justification"] == why, (
        "the recorded justification is not the text supplied on --why; the folded block scalar joined every line"
    )


def test_stamp_records_the_target_sha256_as_it_stands_after_the_write(tmp_path: Path) -> None:
    """Property: when `--stamp` exits 0, the appended row's `layout_signature` is the
    sha256 of the target file AS IT STANDS AFTER the command; when it exits non-zero, the
    manifest is unchanged.

    Class: every repo-relative stamp target — a relation the command's contract states
    over all of them. Two members are positioned differently on purpose: an ordinary
    source file (the control, which must pass) and the manifest itself (the instance that
    discriminates), because the append mutates the target only in the second case.

    Why it is required, as a consequence a reader can check against the code: check-b's
    exemption arm is `_file_content_hash(REPO_ROOT / rel) == expected_sig`, evaluated
    against the file at grading time. A signature taken BEFORE the command's own write
    records a value that never matches, so the command prints `stamped …` and exits 0
    while leaving the gate red on exactly the path it was asked to fix — and re-running it
    is not self-correcting, because each run appends a digest of the file as it was one
    append earlier. Measured on the tree this turn read: recorded `6a487c62…` against an
    actual `72787b6e…`, and the second run recorded `72787b6e…` against an actual
    `4863555c…`, exit 0 both times.

    What kills it: hashing the target and then appending to it with no check that the
    append mutated the target.

    A second correct implementation under which this still passes: refuse when the
    resolved target is the manifest (exit non-zero, nothing written), OR append first and
    record the post-append hash. The two differ in every observable this test measures
    except the invariant itself — verified by running this arm against a copy of the
    checker carrying a one-branch refusal for the manifest, which greens it.
    """
    # Control arm — an ordinary target. This is the must-pass half: without it, the
    # manifest arm below would also "hold" under an implementation that never stamps.
    repo = _make_stamp_repo(tmp_path / "control")
    out = _run(repo, "--stamp", "src/foo.py", "--why", "NON-BREAKING. Control arm.")
    assert out.returncode == 0, out.stdout + out.stderr
    assert _last_row(repo)["layout_signature"] == _sha256(repo / "src" / "foo.py")

    # Discriminating arm — the target is the file the command appends to.
    repo = _make_stamp_repo(tmp_path / "self")
    manifest = repo / MANIFEST_NAME
    before = manifest.read_bytes()
    out = _run(repo, "--stamp", MANIFEST_NAME, "--why", "NON-BREAKING. The manifest itself.")

    if out.returncode != 0:
        assert out.stderr.strip(), "a refusal must say why it refused"
        assert manifest.read_bytes() == before, "a refused stamp must write nothing"
    else:
        recorded = _last_row(repo)["layout_signature"]
        assert recorded == _sha256(manifest), (
            "the appended row records the target's digest as it was BEFORE the append, so "
            "the recorded signature is stale the moment the write completes"
        )


# --- chunk 2b.2: the venue move, the D151 gate and the range closure ---------
# Two held arms. Both are GREEN against the tree this round read and must REMAIN
# green; each carries, in its own docstring, the node id's named mutation that would
# red it (they are also enumerated in the reviewer's round-3 return).

_SENTINEL_RANGE = "layout_relevant:\n  paths:\n    - src/foo.py\n  globs: []\nnon_breaking_allowlist:\n"


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    """Run git in the fixture repo with an identity, so no arm depends on the host config."""
    return subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )


def _make_history(
    root: Path,
    specs: list[dict],
    *,
    sentinel: str = _SENTINEL_RANGE,
    branch_after: int | None = None,
) -> list[str]:
    """A LINEAR fixture repo, ONE COMMIT PER SPEC, oldest first; returns the shas.

    `branch_after=N` commits the first N specs on `main` and the rest on `feature` — the
    only way an arm can tell "grade one commit" from "grade the branch", because on a
    fixture whose every commit sits on `main` the two populations are the same set and the
    assertion cannot discriminate.
    """
    (root / "src" / "hhemt" / "version_migration" / "versions").mkdir(parents=True, exist_ok=True)
    (root / "tests" / "fixtures" / "legacy_layouts").mkdir(parents=True, exist_ok=True)
    (root / "_layout_relevant_files.yaml").write_text(sentinel)
    constants = root / "src" / "hhemt" / "version_migration" / "constants.py"
    _git(root, "init", "-q", "-b", "main")
    shas: list[str] = []
    for i, spec in enumerate(specs):
        if branch_after is not None and i == branch_after:
            _git(root, "checkout", "-q", "-b", "feature")
        constants.write_text(f"LAYOUT_VERSION: int = {spec['version']}\n")
        for rel, content in spec.get("files", {}).items():
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
        _git(root, "add", ".")
        _git(root, "commit", "--allow-empty", "-qm", f"c{i}")
        shas.append(_git(root, "rev-parse", "HEAD").stdout.strip())
    return shas


def _run_range(repo: Path, *args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    """Invoke the checker as the hook does: no ref pair in argv, the pair in the environment."""
    import os

    target = repo / "scripts" / "check_layout_version.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(SCRIPT.read_text())
    clean = {k: v for k, v in os.environ.items() if k not in ("PRE_COMMIT_FROM_REF", "PRE_COMMIT_TO_REF")}
    if env:
        clean.update(env)
    return subprocess.run([sys.executable, str(target), *args], cwd=repo, capture_output=True, text=True, env=clean)


def test_check_b_range_population_is_the_pushed_pair_not_a_wide_base(tmp_path: Path) -> None:
    """Property: with no ref pair supplied, `check-b --range` grades the ONE-COMMIT
    population `HEAD~1..HEAD`, not a base merged from the trunk.

    Class: every invocation of the range venue with no explicit base — the manual
    `pre-commit run --hook-stage pre-push` case, and any pre-push run whose environment
    does not carry the pair. The assertion quantifies over that invocation class, not over
    one fixture.

    Why it is required, as a consequence checkable against the code: a wide base
    (`merge-base(main, HEAD)..HEAD`) is W-B2 and is REFUTED — at this repository's state it
    returns PASS before its scan while `check-a --range main` blocks every push. The
    population is the hazard the venue move exists to close, so the DEFAULT is the thing to
    pin: a default that silently widened would restore the refuted mechanism while every
    other assertion still read green.

    Two arms, and the second is what makes the first mean anything: on a fixture where an
    un-declared layout change sits on `main` behind a clean feature HEAD, the default must
    PASS (one commit graded) while the SAME fixture with the wide pair supplied must FAIL —
    which proves the defect is present in the fixture and that the default is not the
    population that finds it.

    Node id and the mutation that would red it: this test reds under
    `PRE_PUSH_DEFAULT_FROM = "main"` in `scripts/check_layout_version.py` (a one-token
    change), which makes arm 1 fail on the offence behind the clean head.

    A second correct implementation under which it still passes: any default that is a
    single-commit population — `HEAD~1`, `@~1`, or a resolved
    `rev-parse HEAD^` written as the default — all satisfy both arms.
    """
    # The offence sits on the FEATURE branch, one commit behind a CLEAN head, and `main` is
    # the branch point. That placement is what makes the two populations differ here: the
    # one-commit default grades only the clean head, while `main..HEAD` reaches the offence.
    # Putting the offence on `main` instead would have been the mirror-image defect -- both
    # populations exclude it, and arm 1 would pass for the wrong reason (measured: the first
    # version of this arm did exactly that and survived its PRUNE mutation).
    specs = [
        {"version": 5, "files": {"other.txt": "0\n"}},
        {"version": 5, "files": {"other.txt": "branch point\n"}},
        {"version": 5, "files": {"src/foo.py": "# offence behind a clean head\n"}},
        {"version": 5, "files": {"other.txt": "2\n"}},
    ]
    repo = tmp_path / "repo"
    repo.mkdir()
    _make_history(repo, specs, branch_after=2)

    default = _run_range(repo, "check-b", "--range")
    assert default.returncode == 0, (
        "the default population graded a commit behind the branch point; the one-commit "
        "default has been widened: " + default.stdout + default.stderr
    )

    wide = _run_range(repo, "check-b", "--range", env={"PRE_COMMIT_FROM_REF": "main", "PRE_COMMIT_TO_REF": "HEAD"})
    assert wide.returncode == 1, (
        "the fixture's offence is not discoverable at all, so arm 1 proves nothing: " + wide.stdout + wide.stderr
    )
    assert "src/foo.py" in wide.stderr


def test_check_b_range_reads_the_graded_commit_not_a_dirty_worktree(tmp_path: Path) -> None:
    """Property: at the range venue the change-scoped exemption resolves the file's content
    from the GRADED COMMIT, so a dirty worktree cannot change the verdict.

    Class: every range-venue run over a worktree whose content differs from the graded
    commit — which is the NORMAL state at pre-push, because pre-commit reverts unstaged
    edits for the hook's duration and stashes nothing at all when the first unpushed
    ancestor is a root commit. The input is positioned differently from the coder's arm on
    the same property: that arm raises from a monkeypatched `_file_content_hash`, this one
    leaves the instrument alone and varies the DATA the worktree carries.

    Why it is required, as a consequence checkable against the code: if the exemption read
    the worktree, it would compare a recorded signature against content nobody is pushing —
    the recorded defect D151 — and it would do so silently in both directions (exempting a
    push that should fail, and failing a push that should pass). Arm 1 is the
    should-still-pass direction; the control arm is the should-still-fail direction, so the
    test cannot go green merely because the exemption stopped applying at all.

    Node id and the mutation that would red it: this test reds under passing
    `lambda rel: _file_content_hash(REPO_ROOT / rel)` as `_scan_changed_files`'s
    `content_hash` inside `_check_b_range` — the D151 regression, restored as a one-line
    change.

    A second correct implementation under which it still passes: reading the blob through
    `git cat-file blob {commit}:{rel}` into a streaming hash, or through a temporary
    checkout of the graded commit's tree — both resolve the same bytes and satisfy both arms.
    """
    committed = "# committed content\n"
    signature = hashlib.sha256(committed.encode()).hexdigest()
    sentinel = _SENTINEL_RANGE + (
        f'  - path: src/foo.py\n    justification: baseline\n    layout_signature: "{signature}"\n'
    )

    repo = tmp_path / "match"
    repo.mkdir()
    _make_history(
        repo,
        [{"version": 5, "files": {"src/foo.py": "#v0\n"}}, {"version": 5, "files": {"src/foo.py": committed}}],
        sentinel=sentinel,
    )
    (repo / "src" / "foo.py").write_text("# DIRTY worktree, not what is being pushed\n")
    out = _run_range(repo, "check-b", "--range")
    assert out.returncode == 0, (
        "the range venue's verdict followed the dirty worktree instead of the graded commit: " + out.stdout + out.stderr
    )

    # CONTROL — the graded blob itself does NOT match the signature, so the exemption must
    # re-fire. Without this arm, an implementation that never exempts anything passes above.
    repo2 = tmp_path / "mismatch"
    repo2.mkdir()
    _make_history(
        repo2,
        [
            {"version": 5, "files": {"src/foo.py": "#v0\n"}},
            {"version": 5, "files": {"src/foo.py": "#something else entirely\n"}},
        ],
        sentinel=sentinel,
    )
    out2 = _run_range(repo2, "check-b", "--range")
    assert out2.returncode == 1, (
        "the exemption did not re-fire on a graded blob that does not match its signature: " + out2.stdout + out2.stderr
    )
    assert "src/foo.py" in out2.stderr
