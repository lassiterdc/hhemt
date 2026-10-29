#!/usr/bin/env python
"""CI check enforcing the version-migration discipline.

Usage:
    python scripts/check_layout_version.py check-a [base_ref=main]
    python scripts/check_layout_version.py check-b [base_ref=main]
    python scripts/check_layout_version.py check-c [base_ref=main]
    python scripts/check_layout_version.py --stamp {path} --why {text}

Exit 0 = pass; exit 1 = enforcement failure with structured message.
Check C is warning-only and always exits 0.

`--stamp` is the remedy GATE rather than a check: it APPENDS one
`non_breaking_allowlist` row carrying the changed file's current sha256 and the
`--why` text as its justification, so the non-breaking re-declaration Check B asks
for is one command instead of a hand-copied 64-character digest. It takes no
subcommand and writes nothing unless every readiness predicate below holds.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONSTANTS_PATH = REPO_ROOT / "src" / "hhemt" / "version_migration" / "constants.py"
SENTINEL_PATH = REPO_ROOT / "_layout_relevant_files.yaml"
VERSIONS_DIR = REPO_ROOT / "src" / "hhemt" / "version_migration" / "versions"
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "legacy_layouts"
SCENARIO_PATH = REPO_ROOT / "src" / "hhemt" / "scenario.py"
SCENARIO_RELPATH = "src/hhemt/scenario.py"
SENTINEL_RELPATH = "_layout_relevant_files.yaml"
VERSIONS_RELPATH = "src/hhemt/version_migration/versions"
FIXTURES_RELPATH = "tests/fixtures/legacy_layouts"
SLUG_FUNC_NAME = "compute_event_id_slug"
SLUG_FUNC_SENTINEL = f"{SCENARIO_RELPATH}::{SLUG_FUNC_NAME}"

# Rename-transition fallbacks (hhemt-rename Phase 1). For any base ref that
# predates the src/TRITON_SWMM_toolkit -> src/hhemt package rename, the new-path
# `git show {ref}:src/hhemt/...` lookup raises CalledProcessError. These helpers
# fall back to the pre-rename path so the version/slug comparison is taken against
# the file's real pre-rename content instead of a spurious "absent => 0/None".
# Self-retiring: once the rename is in history, the new path exists at every base
# ref and the fallback branch is never reached. Safe to delete after the rename
# is well behind the active base-ref window.
_OLD_CONSTANTS_RELPATH = "src/TRITON_SWMM_toolkit/version_migration/constants.py"
_NEW_CONSTANTS_RELPATH = "src/hhemt/version_migration/constants.py"
_OLD_SCENARIO_RELPATH = "src/TRITON_SWMM_toolkit/scenario.py"

LAYOUT_VERSION_RE = re.compile(r"^LAYOUT_VERSION:\s*int\s*=\s*(\d+)\s*$", re.MULTILINE)
LAYOUT_SUSPICIOUS_SUBSTRINGS = (
    "scenario",
    "log",
    "config",
    "consolidation",
    "paths",
    "schema",
    "conventions",
)


def _git(*args: str, suppress_stderr: bool = False) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=str(REPO_ROOT),
        stderr=subprocess.DEVNULL if suppress_stderr else None,
    ).decode()


class _NonCommitBaseRef(ValueError):
    """`base_ref` names an object that EXISTS but does not peel to a commit."""


_NULL_REF = "0" * 40


def _is_null_ref(ref: str) -> bool:
    """True for pre-commit's "the remote has no such ref yet" sentinel: forty zeroes.

    A NEW-BRANCH PUSH is the reachable case, and the classification below is a MEASUREMENT
    rather than an argument: `git rev-parse --verify 0000.../40` SUCCEEDS (git accepts the null
    object id) while the `^{commit}` probe then fails, so the ordinary two-probe sequence routes
    a null ref into `_NonCommitBaseRef` and prints `misconfigured invocation` with exit 1 --
    blocking a legitimate first push of a branch, on a check whose whole obligation is to grade
    what leaves the machine. Treating it as ABSENCE OF HISTORY instead is the same category
    `_base_ref_or_verdict` already handles, and it is the safe direction: the range collapses to
    empty and nothing is graded, rather than a push being refused.
    """
    return len(ref) == 40 and set(ref) == {"0"}


def _resolve_base_ref(base_ref: str) -> str | None:
    """Classify `base_ref` into three categories and return the ref to use.

        NONEXISTENT        `--verify` fails              -> "HEAD" (absence of history)
        EXISTS-NOT-COMMIT  `--verify` ok, `^{commit}` no -> raise _NonCommitBaseRef
        OK                 both succeed                  -> `base_ref` unchanged

    Returns None when even HEAD is unborn. TWO probes, not one: a single
    `--verify {ref}^{commit}` collapses the first two categories, and they are
    different KINDS of thing. A base that does not exist is an ABSENCE OF HISTORY --
    a state, not a defect -- and falling back to HEAD is right there. A base that
    exists but is not a commit is a MISCONFIGURED INVOCATION: someone wired a hook or
    a CI argument with a tree-ish or a blob, and falling back would run the check
    against a base nobody chose, silently, forever, behind a notice that scrolls past
    once and is never seen again.

    Measured discriminator, and the last two rows are what make the refusal safe:

        OBJECT           --verify  ^{commit}   category
        nosuchref        1         1           NONEXISTENT
        null ref (40x0)  0         1           NONEXISTENT -- special-cased, see _is_null_ref
        empty tree       0         1           EXISTS-NOT-COMMIT
        blob             0         1           EXISTS-NOT-COMMIT
        annotated tag    0         0           OK
        lightweight tag  0         0           OK
        HEAD             0         0           OK

    Both tag forms peel cleanly, so the refusal provably cannot fire on a legitimate
    tag base. This is NOT a loud refusal on shallow clones: a shallow clone's HEAD~1
    does not EXIST, so it lands in the first category and falls back.

    WHY A GUARD IS NEEDED AT ALL. `_layout_version_at` swallows an unresolvable ref and
    returns 0; `_changed_files` swallows nothing, so `git diff {unresolvable}` raises an
    uncaught CalledProcessError. Before the check-b disarm became a pending-change
    predicate, an unresolvable base made `head_v != base_v` true and the disarm returned 0
    BEFORE `_changed_files` ever ran -- the disarm was ACCIDENTALLY SHIELDING the diff.
    The repaired disarm no longer fires there, so the shield is gone and the crash is
    reachable: measured on a repo's SECOND commit and on a `git clone --depth 1`.
    """
    try:
        _git("rev-parse", "--verify", "--quiet", base_ref, suppress_stderr=True)
    except subprocess.CalledProcessError:
        pass  # NONEXISTENT -- absence of history, fall through to the HEAD probe
    else:
        if _is_null_ref(base_ref):
            # NONEXISTENT, not EXISTS-NOT-COMMIT. See `_is_null_ref`: `--verify` accepts the
            # null object id, so without this branch a new-branch push is REFUSED as a
            # misconfigured invocation. Falls through to the same HEAD probe.
            pass
        else:
            try:
                _git("rev-parse", "--verify", "--quiet", f"{base_ref}^{{commit}}", suppress_stderr=True)
            except subprocess.CalledProcessError:
                raise _NonCommitBaseRef(
                    f"base ref {base_ref!r} exists but does not name a commit (it is a tree or a blob); "
                    f"this is a misconfigured invocation, not an absence of history"
                ) from None
            return base_ref
    try:
        _git("rev-parse", "--verify", "--quiet", "HEAD^{commit}", suppress_stderr=True)
    except subprocess.CalledProcessError:
        return None
    return "HEAD"


def _base_ref_or_verdict(base_ref: str, check_name: str) -> tuple[str | None, int | None]:
    """Shared preamble for check-a and check-b: `(ref, None)` to proceed, `(None, code)` to return.

    Both entry points need it because both are `always_run: true` in
    `.pre-commit-config.yaml`: a root-commit skip in one is defeated by a hard failure in
    the other, and the developer is blocked either way. Measured at a root commit before
    this guard, under EVERY base including the empty tree -- check-a
    `FAIL - LAYOUT_VERSION jumped from 0 to 19` while check-b printed
    `bumped (0->19); check-a covers this; pass`, which is false: check-a did not cover it,
    it failed with a nonsense expectation derived from a `base_v` that swallowed to 0.
    """
    try:
        resolved = _resolve_base_ref(base_ref)
    except _NonCommitBaseRef as exc:
        print(f"{check_name}: FAIL - {exc}", file=sys.stderr)
        return None, 1
    if resolved is None:
        # NOT a conceded fail-open, and the licensing reason is ACTIONABILITY rather than
        # the absence of a referent. At a root commit LAYOUT_VERSION is being ESTABLISHED,
        # so the remedy these checks exist to demand -- "bump the version and ship a
        # migration for it" -- names a bump that does not exist. Every layout-relevant file
        # in an initial import would be flagged with an instruction nobody can follow, and a
        # finding that cannot be acted on is worse than a skip. (Contrast pre-push, where the
        # predicate is inert but the remedy is still well defined; "no referent" alone would
        # argue for a different predicate there, not for a skip, so it is the wrong ground.)
        #
        # Do NOT "close" this by supplying a base that makes the diff non-empty. The only
        # such base is the empty tree, and it was measured wrong at every state: fail-OPEN
        # under the pre-repair disarm (EXIT=0 on a commit that genuinely modified a
        # layout-relevant file), and flagging files the commit never touched under the
        # repaired one. The one case this genuinely cannot see is a repo re-inited from an
        # existing tree, where on-disk analyses DO exist in the wild; no base closes that,
        # and the remedy is a one-time operator baseline, not a base choice.
        print(f"{check_name}: no in-repo predecessor (unborn HEAD / root commit); skipped")
        return None, 0
    if resolved != base_ref:
        print(
            f"{check_name}: base ref {base_ref!r} does not exist (shallow clone or early history); "
            f"falling back to {resolved!r}"
        )
    return resolved, None


def _git_show_with_rename_fallback(ref: str, new_relpath: str, old_relpath: str) -> str | None:
    """`git show {ref}:{new_relpath}`, falling back to the pre-rename path.

    Returns the file text, or None when the file is absent at `ref` under BOTH
    the post-rename and pre-rename paths. Lets the version/slug checks compare
    against a file's real pre-rename content across the package-dir rename
    instead of treating an absent new path as "version 0 / hash None".
    """
    for relpath in (new_relpath, old_relpath):
        try:
            return _git("show", f"{ref}:{relpath}", suppress_stderr=True)
        except subprocess.CalledProcessError:
            continue
    return None


def _layout_version_at(ref: str) -> int:
    text = _git_show_with_rename_fallback(ref, _NEW_CONSTANTS_RELPATH, _OLD_CONSTANTS_RELPATH)
    if text is None:
        return 0
    m = LAYOUT_VERSION_RE.search(text)
    if not m:
        raise SystemExit(f"check_layout_version: no LAYOUT_VERSION found at {ref}")
    return int(m.group(1))


def _layout_version_at_head() -> int:
    text = CONSTANTS_PATH.read_text()
    m = LAYOUT_VERSION_RE.search(text)
    if not m:
        raise SystemExit("check_layout_version: no LAYOUT_VERSION at HEAD")
    return int(m.group(1))


def _changed_files(base_ref: str, to_ref: str | None = None) -> list[Path]:
    """Changed paths from `base_ref` to `to_ref` (default: the WORKING TREE), EXCLUDING pure
    git renames/copies.

    `to_ref=None` grades `base_ref`..worktree, which is what pre-commit needs (see the range
    note below). `to_ref="HEAD"` grades the COMMIT RANGE `base_ref`..HEAD, which is what a
    range venue needs: at pre-push the worktree is not what is being pushed, and reading it
    lets an unrelated pending edit into the graded file set.

    A source-package move or import-rewrite that git detects as a rename (R) or
    copy (C) does NOT alter on-disk {analysis_dir} layout, so it must not trip
    the layout guard. `--name-status -M` reports R/C entries as
    `R<score>\t<old>\t<new>` (tab-delimited); all other entries (A/M/D/T) use
    `<status>\t<path>`. We skip R/C and take the last tab field as the path for
    the rest. Rename detection is on by default (diff.renames=true); -M makes it
    explicit and threshold-independent of repo config.

    Rationale + evidence: hhemt-rename Phase 1. All 143 src/TRITON_SWMM_toolkit
    -> src/hhemt moves are R-status at >=78% similarity even after the in-file
    import-string rewrite, so this skip suppresses the entire move set with zero
    allowlist additions. A real layout-logic change made in the SAME commit as a
    rename is the only blind spot; the slug-hash AST sentinel (path-independent)
    still catches the highest-risk such case (compute_event_id_slug), and the
    skip is one-commit-scoped (the next non-rename edit re-enters enforcement).
    """
    # {base}..HEAD is a COMMIT RANGE and stops at HEAD. At pre-commit the commit object
    # does not exist, so this range is the PREVIOUS, already-committed commit -- which
    # already passed this hook when it was made. Measured on a real merge: the range
    # returned 1 unrelated file while 8 layout-relevant paths sat in the pending working
    # tree, invisible. Diff against the WORKING TREE so the pending change is in scope,
    # matching _layout_version_at_head(), which already reads the working tree -- the two
    # previously described different commits inside one predicate. Pre-commit is the ONLY
    # venue this script runs in (branching-and-release-model.md records that it is wired
    # into no GitHub Actions workflow), so the range being wrong made the file-scan arm
    # inert everywhere rather than only at commit time.
    diff_args = ["diff", "--name-status", "-M", base_ref]
    if to_ref is not None:
        diff_args.append(to_ref)
    out = _git(*diff_args).strip()
    changed: list[Path] = []
    for line in out.splitlines():
        if not line.strip():
            continue
        fields = line.split("\t")
        status = fields[0]
        if status.startswith("R") or status.startswith("C"):
            continue
        changed.append(REPO_ROOT / fields[-1])
    return changed


def _added_files(base_ref: str) -> list[Path]:
    # Same {base}..HEAD vacuity as _changed_files above, and the same fix. The STAKES
    # differ and the difference is worth stating: check-c is invoked from nowhere (no
    # pre-commit hook id, no CI workflow -- only check-a and check-b are wired) and
    # returns 0 unconditionally, so its arm was inert for two independent reasons rather
    # than one. Repairing the range here is pre-emptive: it makes the function correct
    # for whenever check-c is wired, and changes nothing observable until then.
    out = _git("diff", "--name-only", "--diff-filter=A", base_ref).strip()
    return [REPO_ROOT / line for line in out.splitlines() if line.strip()]


def _load_sentinel(ref: str | None = None) -> dict:
    """The layout sentinel at `ref`, or from the WORKING TREE when `ref` is None.

    THE FOURTH WORKTREE READ, and the only one that failed OPEN. This file supplies BOTH
    halves of the scan's definition -- `layout_relevant` (what is governed) and
    `non_breaking_allowlist` (what is excused) -- so a range venue reading it from the
    working tree grades a COMMIT RANGE against an UNCOMMITTED rule set. Measured at
    32be36e2: with the two V002x paths added to the allowlist and NOT committed,
    `check-b HEAD~1` went from FAIL to `no layout-relevant changes; pass` with the graded
    range unchanged. That direction is the dangerous one -- the venue-inert disarm this
    module was repaired for produced a false BLOCK, which is loud; this produces a false
    PASS, which ships the un-migrated change the guard exists to stop. And the guard's own
    failure text tells the developer to edit this exact file, so the edit that defeats it
    is the edit it asks for.

    Absence at `ref` is a hard failure rather than a fallback: a range venue grades a
    committed rule set, and silently reverting to the working tree is the defect itself.

    OPERATIONAL CONSEQUENCE, named here so nobody meets it as a surprise: pushing a branch
    whose HEAD predates this file's introduction hard-fails the hook with the message below.
    That is the intended direction for a guard whose failure mode is silent on-disk
    corruption, and the message is actionable; it is not a bug report. Note also that the
    reachable input is a HEAD without the file -- `check_b` passes "HEAD", never the base ref
    -- so no property of the base commit can reach or avoid this branch.
    """
    if ref is None:
        return yaml.safe_load(SENTINEL_PATH.read_text())
    try:
        text = _git("show", f"{ref}:{SENTINEL_RELPATH}", suppress_stderr=True)
    except subprocess.CalledProcessError:
        raise SystemExit(
            f"check_layout_version: {SENTINEL_RELPATH} is absent at {ref}; a range venue grades a "
            f"committed rule set and must not fall back to the working tree"
        ) from None
    return yaml.safe_load(text)


def _tree_names_at(ref: str, relpath: str) -> set[str]:
    """Entry names directly under `relpath` in `ref`'s tree; empty when the path is absent.

    The range-venue counterpart to `Path.glob` / `Path.is_dir`. check_a's migration-module
    and fixture-directory probes are filesystem reads, so at a range venue an UNCOMMITTED
    module or fixture directory would satisfy a check about a pushed commit -- the same
    false-PASS direction as the sentinel read above, reached through a different instrument.
    """
    try:
        out = _git("ls-tree", "--name-only", f"{ref}:{relpath}", suppress_stderr=True)
    except subprocess.CalledProcessError:
        return set()
    return {line.strip() for line in out.splitlines() if line.strip()}


def _glob_to_regex(glob: str) -> str:
    """Translate a layout-sentinel glob to an anchored regex.

    Supports ``**`` (any depth, slash-crossing), ``*`` (single path segment),
    and ``?`` (single non-slash char). ``**/`` absorbs its trailing slash so it
    matches ZERO OR MORE directories — this is the fix for the ``fnmatch``
    failure where ``config/**/*.py`` did not match the direct child
    ``config/analysis.py``. Char-class ``[...]`` brackets are NOT supported
    (rendered literal via re.escape); no current glob uses them.
    """
    i, n, out = 0, len(glob), []
    while i < n:
        c = glob[i]
        if c == "*":
            if i + 1 < n and glob[i + 1] == "*":
                i += 2
                if i < n and glob[i] == "/":
                    out.append("(?:.*/)?")  # **/ matches zero or more dirs
                    i += 1
                else:
                    out.append(".*")  # trailing ** matches anything
            else:
                out.append("[^/]*")  # single segment
                i += 1
        elif c == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(c))
            i += 1
    return "".join(out)


def _layout_glob_match(rel: str, glob: str) -> bool:
    """True iff `rel` (a repo-relative POSIX path) matches `glob`."""
    return re.fullmatch(_glob_to_regex(glob), rel) is not None


@dataclass(frozen=True)
class AllowlistEntry:
    justification: str | None  # None for legacy bare-string entries
    # The Optional is the PARSE type only: _load_allowlist refuses any path whose RESOLVED
    # entry is None, so no caller can observe None here. `None => path-permanent` was the
    # content-independent fail-open that gate closed.
    layout_signature: str | None = None  # change-scoped hash


def _load_allowlist(sentinel: dict) -> dict[str, AllowlistEntry]:
    """Parse `non_breaking_allowlist` into path -> AllowlistEntry.

    Accepts two entry forms:
      - bare string `path`                    -> AllowlistEntry(None, None)
      - mapping {path, justification[, layout_signature]}
        (justification REQUIRED and non-empty for the mapping form)

    Both forms remain legal PER ROW so the append-log stays intact, but every path must
    RESOLVE to a layout_signature: a path whose RESOLVED entry is None raises. A MAPPING
    row that omits the field does NOT raise when an earlier row for the same path supplied
    one -- the merge below carries it forward, which is what keeps this an append-log. A
    BARE-STRING row is different: it clears rather than inherits, because the string branch
    `continue`s before reaching the merge, so it raises even after a signature row.

    Exits non-zero (SystemExit) on any malformed entry, AND on a well-formed row set in
    which some path resolves unscoped.
    """
    out: dict[str, AllowlistEntry] = {}
    for raw in sentinel.get("non_breaking_allowlist", []) or []:
        if isinstance(raw, str):
            out[raw] = AllowlistEntry(justification=None)
            continue
        if not isinstance(raw, dict) or "path" not in raw:
            raise SystemExit(f"check_layout_version: malformed allowlist entry: {raw!r}")
        unknown = set(raw) - {"path", "justification", "layout_signature"}
        if unknown:
            raise SystemExit(f"check_layout_version: unknown allowlist keys {sorted(unknown)} in {raw['path']!r}")
        just = raw.get("justification")
        if not isinstance(just, str) or not just.strip():
            raise SystemExit(
                f"check_layout_version: dict allowlist entry {raw['path']!r} needs a non-empty justification"
            )
        sig = raw.get("layout_signature")
        if sig is not None and not isinstance(sig, str):
            raise SystemExit(f"check_layout_version: layout_signature for {raw['path']!r} must be a string")
        # DUPLICATE PATHS ARE THE IDIOM, NOT A DEFECT. 10 of this file's 35 entries are
        # duplicates, and each occurrence carries its own justification naming its own
        # change. They are an append-LOG over a frequently-amended file and the
        # justifications are the audit trail, so agreeing occurrences MERGE rather than
        # being refused.
        #
        # A ROW THAT OMITS layout_signature IS NOT ASSERTING "never check this file again".
        # It is a row whose author did not supply an optional field while recording their
        # own change. Plain `out[path] = ...` is last-wins, so such a row silently converts
        # an earlier change-scoped exemption into a PERMANENT one -- the file becomes exempt
        # from check-b regardless of what it later becomes, and nothing in the yaml says so.
        # Measured on report_plot_ids.py: three rows, signature-scoped first, and its
        # signature was UNREACHABLE rather than stale.
        #
        # Keeping the signature is therefore the merge that matches what the rows mean. A
        # DELIBERATE downgrade to permanently-exempt is NO LONGER EXPRESSIBLE AT ALL: the
        # resolved-entry gate at the end of this function refuses any path that ends up with
        # no signature, whether by an omitting row, an explicit null, or a bare string.
        # Retiring a file's governance is now TWO acts, not one: DELETE its allowlist
        # row(s) -- the resolved-entry gate reads only non_breaking_allowlist -- and THEN
        # narrow layout_relevant so the file stops being captured. Narrowing on its own
        # changes nothing the gate sees.
        #
        # SCOPE: a path whose rows carry TWO OR MORE signatures is UNTOUCHED by this edit.
        # Those remain last-wins, exactly as before.
        prior = out.get(raw["path"])
        if sig is None and prior is not None and prior.layout_signature is not None:
            sig = prior.layout_signature
        out[raw["path"]] = AllowlistEntry(justification=just, layout_signature=sig)
    # PATH-PERMANENCE IS INEXPRESSIBLE. Checked over the RESOLVED entries, not per row,
    # so the append-log stays intact: a historical signature-less row is fine as long as
    # the path ends up scoped by some later row. This single gate covers ALL THREE
    # permanence-expressing forms -- a mapping that omits layout_signature, a mapping that
    # sets it to null, and a bare-string entry (which cannot carry one at all) -- because
    # every one of them resolves to None here. There is deliberately NO opt-in permanence
    # key. Retiring a file is TWO acts and this gate only sees the first: delete its
    # allowlist row(s), then narrow layout_relevant so it stops being captured.
    unscoped = sorted(p for p, e in out.items() if e.layout_signature is None)
    if unscoped:
        raise SystemExit(
            "check_layout_version: non_breaking_allowlist path(s) resolve to NO "
            f"layout_signature and would be exempt from Check B forever: {unscoped}. "
            "Add a layout_signature (the file's current sha256) to a row for each path. "
            "If a path should NOT be governed at all, a signature is the wrong remedy -- but "
            "so is narrowing layout_relevant on its own: this function reads "
            "non_breaking_allowlist and NOTHING else, so an un-governed path that still has "
            "a signature-less row still refuses here. DELETE that path's allowlist row(s) "
            "first. Then, to stop governing it, narrow layout_relevant -- which for a "
            "glob-captured path means replacing the glob with an explicit paths enumeration, "
            "since _glob_to_regex has no negation form."
        )
    return out


def _file_content_hash(path: Path) -> str | None:
    """Whole-file sha256 hexdigest, or None when the file is absent.

    Returning None (rather than raising on read_bytes) means a DELETED
    allowlisted path that carries a stamped layout_signature re-fires Check-B
    (None != the pinned hash -> the change-scoped gate falls through to path/glob
    enforcement) instead of crashing the guard with FileNotFoundError.

    THIS IS THE WORKTREE VENUE ONLY. The range venue must never call it -- see
    `_file_content_hash_at` and the D151 gate in `_scan_changed_files`.
    """
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_bytes(*args: str) -> bytes:
    """`git ...` as BYTES. `_git` decodes unconditionally, which is wrong for a digest:
    a blob is not guaranteed to be valid UTF-8, and a decode error or a replacement
    character would change the hash of content the gate is comparing.
    """
    return subprocess.check_output(["git", *args], cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL)


def _file_content_hash_at(ref: str, rel: str) -> str | None:
    """Whole-blob sha256 of `rel` AS IT STANDS AT `ref`, or None when absent there.

    THE RANGE VENUE'S COUNTERPART to `_file_content_hash`, and the D151 gate. At pre-push
    the worktree is neither HEAD nor the developer's tree: `pre_commit/commands/run.py`
    computes `stash = not all_files and not files`, which is not keyed on stage, so unstaged
    edits are reverted for the hook's duration while staged-uncommitted ones survive, and a
    push whose first unpushed ancestor is a root commit stashes nothing at all. Any worktree
    read there is therefore unsound in BOTH directions -- it can compare a signature against
    content nobody is pushing, and it can miss content that is.

    None for an absent path, matching `_file_content_hash`, so a file DELETED in the graded
    commit re-fires rather than `None == None` exempting it.
    """
    try:
        data = _git_bytes("show", f"{ref}:{rel}")
    except subprocess.CalledProcessError:
        return None
    return hashlib.sha256(data).hexdigest()


def _commits_in_range(from_ref: str, to_ref: str) -> list[str]:
    """Full shas of the commits in `from_ref..to_ref`, OLDEST FIRST.

    `--reverse` is load-bearing rather than cosmetic: the walk reports failures in the order
    a reviewer reading the push will meet them, and a reversal bug would make a multi-commit
    diagnosis read backwards. `from_ref` itself is EXCLUDED, which is `git rev-list`'s
    range semantics and is what makes `from..to` name the pushed commits.
    """
    out = _git("rev-list", "--reverse", f"{from_ref}..{to_ref}")
    return [line.strip() for line in out.splitlines() if line.strip()]


def _commit_parent(commit: str) -> str | None:
    """`commit^` as a commit sha, or None when `commit` is a root commit.

    None is the ROOT-COMMIT signal and the caller skips: with no in-repo predecessor there
    is no migration to demand and no predecessor version to compare against, which is the
    same reasoning `_base_ref_or_verdict` applies at an unborn HEAD.

    `^` is the FIRST PARENT, so for a merge this compares against the branch being merged
    into -- the same "what this commit introduced" definition `git show` uses, rather than a
    diff against every parent. `^{commit}` peels, in case `^` ever names a tag.
    """
    try:
        return _git("rev-parse", "--verify", "--quiet", f"{commit}^^{{commit}}", suppress_stderr=True).strip()
    except subprocess.CalledProcessError:
        return None


def _short(sha: str) -> str:
    """A commit sha at the abbreviation length the gate's own messages use."""
    return sha[:9]


def _slug_function_hash(source: str) -> str | None:
    """AST-normalized hash of (signature, body) for compute_event_id_slug.

    Returns None when the function is not present. Uses ast.unparse so that
    formatting / whitespace / comment changes do not register as drift, but
    real signature or logic changes do.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == SLUG_FUNC_NAME:
            normalized = ast.unparse(node)
            return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return None


def _slug_hash_at(ref: str) -> str | None:
    text = _git_show_with_rename_fallback(ref, SCENARIO_RELPATH, _OLD_SCENARIO_RELPATH)
    if text is None:
        return None
    return _slug_function_hash(text)


def _slug_hash_at_head() -> str | None:
    if not SCENARIO_PATH.exists():
        return None
    return _slug_function_hash(SCENARIO_PATH.read_text())


def _scan_changed_files(
    changed: list[Path],
    paths: set[str],
    globs: list[str],
    allow: dict[str, AllowlistEntry],
    content_hash,
) -> list[Path]:
    """Check B's file scan over an ALREADY-MATERIALISED change set.

    Factored out of `check_b` so the two venues can share one implementation of the scan and
    differ ONLY in `content_hash`, which is exactly where they must differ. `content_hash(rel)`
    is the sole venue-dependent input: the pending-change venue passes a WORKTREE read, and the
    range venue passes a blob read at the graded commit. Putting the difference in a parameter
    rather than in a branch is what makes the D151 defect un-reintroducible -- there is one
    place to pass the right function and no second place to forget to.
    """
    relevant: list[Path] = []
    for p in changed:
        rel = str(p.relative_to(REPO_ROOT))
        if rel in allow:
            expected_sig = allow[rel].layout_signature
            if expected_sig is None:
                # FAIL-CLOSED TRIPWIRE, not a reachable branch once _load_allowlist's
                # resolved-entry gate stands: that gate's predicate is `is None`, and it refuses
                # every path whose RESOLVED layout_signature is None, so this cannot fire while
                # it holds. (An EMPTY-STRING signature is deliberately NOT refused there -- it
                # is not None, so it flows to the comparison below and is FLAGGED on every
                # change, which is fail-closed already.) This branch is an EXECUTABLE
                # restatement of that invariant rather than a comment about it, because falling
                # through to the comparison below is NOT fail-closed in every case --
                # content_hash returns None for an ABSENT file, so `None == None` would
                # silently exempt a DELETED allowlisted path, which is the fail-open this
                # repair was convened to close.
                # BLAST RADIUS, stated because a reader infers otherwise from a branch sitting
                # inside a layout-relevance scan: `if rel in allow:` is tested BEFORE
                # layout-relevance, so with the gate weakened this fires for ANY changed file
                # carrying a signature-less allowlist row, governed or not. That is intended --
                # the tripwire's subject is "the loader gate is broken", which is true
                # regardless of whether the file that surfaced it is governed.
                raise SystemExit(
                    f"check_layout_version: {rel} resolves to NO layout_signature at the "
                    "Check B comparison. _load_allowlist should have refused this at load; "
                    "its resolved-entry gate has been weakened. Restore it -- do not exempt "
                    "the path here."
                )
            if content_hash(rel) == expected_sig:
                continue  # change-scoped exemption still valid for this content
            # signature drifted: exemption no longer covers this change -> re-fire Check-B
        if rel in paths or any(_layout_glob_match(rel, g) for g in globs):
            relevant.append(p)
    return relevant


def check_a(base_ref: str, *, range_mode: bool = False) -> int:
    resolved_base, verdict = _base_ref_or_verdict(base_ref, "check-a")
    if verdict is not None:
        return verdict
    base_v = _layout_version_at(resolved_base)
    # SAME VENUE SPLIT AS check_b, and it is not optional here either. Every read below is a
    # WORKING-TREE read at default settings: the version, the migration-module glob, and the
    # two fixture-directory probes. At a range venue that is wrong in both directions.
    # Measured at 32be36e2 with an UNCOMMITTED bump to 23 on a push carrying 21->22:
    # `check-a HEAD~1` reported `FAIL - LAYOUT_VERSION jumped from 21 to 23`, blocking a push
    # that carries neither 23 nor a jump -- a false BLOCK relocated from check_b rather than
    # removed. The module and fixture probes fail the other way: an UNCOMMITTED module or
    # fixture directory would satisfy a check about a pushed commit, which is a false PASS.
    head_v = _layout_version_at("HEAD") if range_mode else _layout_version_at_head()
    if head_v == base_v:
        print(f"check-a: LAYOUT_VERSION unchanged ({head_v}); pass")
        return 0
    if head_v != base_v + 1:
        print(
            f"check-a: FAIL - LAYOUT_VERSION jumped from {base_v} to {head_v}; "
            f"each bump must be exactly +1 (write {head_v - base_v} migrations instead)",
            file=sys.stderr,
        )
        return 1
    if range_mode:
        module_names = _tree_names_at("HEAD", VERSIONS_RELPATH)
        fixture_names = _tree_names_at("HEAD", FIXTURES_RELPATH)
        has_module = any(re.fullmatch(rf"V{head_v:04d}__.*\.py", n) for n in module_names)
        has_from_fixture = f"v{base_v}" in fixture_names
        has_to_fixture = f"v{head_v}" in fixture_names
    else:
        has_module = next(VERSIONS_DIR.glob(f"V{head_v:04d}__*.py"), None) is not None
        has_from_fixture = (FIXTURES_DIR / f"v{base_v}").is_dir()
        has_to_fixture = (FIXTURES_DIR / f"v{head_v}").is_dir()
    if not has_module:
        print(
            f"check-a: FAIL - LAYOUT_VERSION bumped to {head_v} but no migration module "
            f"V{head_v:04d}__*.py exists in {VERSIONS_DIR}",
            file=sys.stderr,
        )
        return 1
    if not has_from_fixture:
        print(
            f"check-a: FAIL - fixture {FIXTURES_DIR / f'v{base_v}'} (FROM side) missing",
            file=sys.stderr,
        )
        return 1
    if not has_to_fixture:
        print(
            f"check-a: FAIL - fixture {FIXTURES_DIR / f'v{head_v}'} (TO side) missing",
            file=sys.stderr,
        )
        return 1
    print(f"check-a: PASS - V{head_v:04d} migration + fixtures present")
    return 0


def check_b(base_ref: str, *, range_mode: bool = False, to_ref: str = "HEAD") -> int:
    sentinel = _load_sentinel(to_ref if range_mode else None)
    resolved_base, verdict = _base_ref_or_verdict(base_ref, "check-b")
    if verdict is not None:
        return verdict
    if range_mode:
        # THE RANGE VENUE IS A PER-COMMIT WALK, delegated so the pending-change venue below
        # keeps ONE code path that the walk cannot drift from. `_check_b_range` carries the
        # mechanism and the reason a wider base is not it.
        return _check_b_range(resolved_base, to_ref, sentinel)
    head_v = _layout_version_at_head()
    base_v = _layout_version_at(resolved_base)
    pending_base_v = _layout_version_at("HEAD")
    # DISARM ON AGREEMENT, and both halves of this condition are load-bearing.
    #
    # The literal "HEAD" is NOT a stale `base_ref` waiting to be tidied up. The disarm asks
    # "does THIS commit carry the bump", which is a property of the pending change and is
    # therefore base-INDEPENDENT; the base ref governs the FILE SET and nothing else.
    # Replacing "HEAD" with the base ref for consistency reinstates the fail-open this repair
    # removed: a bump that landed one commit ago makes `head_v != base_v` true, the disarm
    # fires, and check-b's file scan does not run at all for the whole next commit. Measured
    # on this repo at 48d4efd3, which modified src/hhemt/log.py, a layout_relevant.paths member.
    #
    # The `and head_v != base_v` conjunct is what stops the disarm handing off to a check-a
    # that cannot see what it is being handed. The two checks read DIFFERENT COMMITS -- check-b
    # here reads HEAD, check-a reads the base it resolved -- and when those disagree the handoff
    # is to a blind target. Measured: with HEAD~1=18, HEAD=19 and a worktree that backs the bump
    # out to 18, a HEAD-only disarm reports `bumped (19->18)` and returns 0 while check-a reports
    # `unchanged (18); pass`, so a pending layout-relevant change is neither scanned nor
    # validated. Requiring BOTH comparisons to see a pending bump makes the disarm fire only
    # when check-a will actually evaluate one.
    #
    # The false handoff is now unreachable from BOTH sides rather than suppressed on one. This
    # conjunct closes the version-churn instance; the shared `_base_ref_or_verdict` sentinel
    # closes the root-commit instance, where check-b used to print "check-a covers this" while
    # check-a failed with `jumped from 0 to N` off a `base_v` that had swallowed to 0.
    #
    # SCOPE, so nobody wires this into a venue where it is dead. This is now a PENDING-CHANGE
    # predicate: `_layout_version_at("HEAD")` reads the COMMIT and `_layout_version_at_head()`
    # reads the WORKTREE FILE, so on a clean tree they are equal by construction and the disarm
    # is structurally unreachable. Pre-commit, where the tree is dirty by definition, is the only
    # venue in which it can fire. A range-based venue -- pre-push, CI, or the documented
    # `check-b [base_ref=main]` default against a clean checkout -- gets a permanently-inert
    # disarm and must supply its own bump predicate rather than assuming this one works there.
    #
    # `--range` IS THAT PREDICATE, and it is a venue SELECTION rather than a relaxation. Under
    # it `head_v` reads the HEAD COMMIT, the file scan grades `base..HEAD`, and the slug hash is
    # taken at HEAD -- so the worktree is not read on any arm and the bump predicate collapses to
    # the base-relative `head_v != base_v`, which is exactly "the graded range contains a bump".
    # The pending conjunct is dropped there because it has no referent, NOT because it is
    # inconvenient: with no pending change there is no "this commit" distinct from HEAD.
    #
    # Do NOT re-derive the venue from tree dirtiness instead of declaring it. Measured: a pending
    # edit unrelated to the push makes `git diff HEAD` non-empty, the inference reads a pre-push
    # run as pre-commit, and the false failure returns on exactly the push this flag exists to
    # permit. The venue is a property of the CALLER and only the caller knows it.
    #
    # PRE-PUSH IS NOT A CLEAN-TREE GUARANTEE, and the earlier draft of this comment said it
    # was. `commands/run.py:344` computes `stash = not all_files and not files`, which is NOT
    # keyed on stage, and `hook_impl._pre_push_ns` supplies from_ref/to_ref on a branch
    # update -- so UNSTAGED edits are reverted to HEAD for the hook's duration while
    # STAGED-uncommitted ones survive, and a push whose first unpushed ancestor is a ROOT
    # commit takes the `all_files=True` path and stashes nothing at all. The worktree at
    # pre-push is therefore neither HEAD nor the developer's tree, and it differs between
    # those cases. That makes reading it unsound in BOTH directions, which is a stronger
    # reason for this flag than the one it replaces, not a weaker one.
    #
    # RESIDUAL, stated so nobody prices it as closure. `head_v != base_v` disarms the WHOLE
    # graded span, so a base spanning many commits is disarmed by ONE bump anywhere in it -- a
    # later un-bumped layout change inside the same span rides through. THAT RESIDUAL IS NOW
    # CLOSED AT THE RANGE VENUE and only there: `_check_b_range` walks the range one commit at
    # a time and grades each commit's own diff, so a bump in commit 2 cannot disarm the scan of
    # commit 3. It remains open on THIS path, which grades a single pending change, where there
    # is no span to coarsen over.
    if head_v != base_v and head_v > pending_base_v:
        print(
            f"check-b: LAYOUT_VERSION bumped ({pending_base_v}->{head_v}); "
            f"check-a covers this at the same base ({resolved_base}); pass"
        )
        return 0
    paths = set(sentinel["layout_relevant"]["paths"])
    globs = sentinel["layout_relevant"]["globs"]
    allow = _load_allowlist(sentinel)
    layout_relevant_changed = _scan_changed_files(
        _changed_files(resolved_base, None),
        paths,
        globs,
        allow,
        # THE PENDING-CHANGE VENUE'S CONTENT SOURCE, and it is the WORKTREE on purpose: at
        # commit time the pending edit IS the change being graded, so reading it from HEAD
        # would refuse every non-breaking exemption on the commit that introduces it.
        lambda rel: _file_content_hash(REPO_ROOT / rel),
    )
    failed = False
    if layout_relevant_changed:
        print(
            "check-b: FAIL - layout-relevant files changed without LAYOUT_VERSION bump:",
            file=sys.stderr,
        )
        for p in layout_relevant_changed:
            print(f"  - {p.relative_to(REPO_ROOT)}", file=sys.stderr)
        _print_resolution_paths(head_v)
        failed = True

    if SLUG_FUNC_SENTINEL not in allow:
        base_hash = _slug_hash_at(resolved_base)
        head_hash = _slug_hash_at_head()
        if base_hash is not None and head_hash is not None and base_hash != head_hash:
            print(
                f"check-b: FAIL - {SLUG_FUNC_SENTINEL} drift would invalidate V0001's "
                f"slug derivation; bump LAYOUT_VERSION and ship V{head_v + 1:04d} that "
                f"supersedes V0001's slug logic, OR add {SLUG_FUNC_SENTINEL} to "
                "non_breaking_allowlist with author justification for why the refactor "
                "preserves slug semantics.",
                file=sys.stderr,
            )
            failed = True

    if failed:
        return 1
    print("check-b: no layout-relevant changes; pass")
    return 0


def _print_resolution_paths(head_v: int) -> None:
    """Check B's two resolution paths, printed on failure at BOTH venues so the advice
    cannot drift between them."""
    print(
        "\nResolution paths:\n"
        f"  1. If breaking: bump LAYOUT_VERSION to {head_v + 1}, write "
        f"versions/V{head_v + 1:04d}__*.py, add fixtures v{head_v}/ and v{head_v + 1}/.\n"
        "  2. If non-breaking: add or update the file's non_breaking_allowlist entry in "
        "_layout_relevant_files.yaml. The row must leave the path RESOLVING to a layout_signature -- "
        "set it to the file's current sha256. Omitting the field on an appended row INHERITS the "
        "path's previous signature, which no longer matches the content you just changed, so this "
        "check keeps failing; a bare-string row clears the signature and the loader refuses at load.",
        file=sys.stderr,
    )


def _check_b_range(from_ref: str, to_ref: str, sentinel: dict) -> int:
    """Check B over a COMMIT RANGE, ONE COMMIT AT A TIME.

    THE POPULATION. `from_ref..to_ref` -- normally the `PRE_COMMIT_FROM_REF`/`PRE_COMMIT_TO_REF`
    pair pre-commit exports to a pre-push hook, so the graded set is exactly the commits being
    pushed rather than one commit of N.

    WHY A PER-COMMIT WALK, in the gate's own terms. The previous range arm compared
    `_layout_version_at(base)` against `_layout_version_at(HEAD)` ONCE and, on any difference,
    printed `bumped; check-a covers this` and returned 0 BEFORE the file scan. One bump
    anywhere in a wide range therefore disarmed the WHOLE range, so a later un-bumped layout
    change inside it rode through ungraded. That is a property of a RANGE-WIDE predicate, not
    of a bad comparison: there is no single version pair that can represent "some commit in
    this span bumped" and "this commit did not". Walking each commit and grading ITS OWN diff
    against ITS OWN version transition removes the coarseness rather than tuning it -- a bump
    in commit 2 cannot disarm the scan of commit 3, because commit 3 is evaluated from its own
    parent.

    WHY NOT A WIDER BASE. Grading `merge-base(main, HEAD)..HEAD` is W-B2 and is REFUTED:
    measured at this repository's state (`main` at LAYOUT_VERSION 16 against HEAD 26) it
    returns PASS before its scan while `check-a --range main` blocks every push. The
    population comes from the pushed ref pair; a wider base is not the mechanism.

    AND THE WORKTREE IS NEVER READ. The exemption arm gets `_file_content_hash_at(commit, rel)`
    -- the blob at the GRADED COMMIT -- because at pre-push the worktree is neither HEAD nor
    the developer's tree. That is the D151 gate and it is not optional here.

    ROOT COMMITS ARE SKIPPED. A commit with no in-repo predecessor has no predecessor version
    to compare and no migration to demand, the same reasoning `_base_ref_or_verdict` applies at
    an unborn HEAD.
    """
    paths = set(sentinel["layout_relevant"]["paths"])
    globs = sentinel["layout_relevant"]["globs"]
    allow = _load_allowlist(sentinel)
    to_v = _layout_version_at(to_ref)

    failures: list[tuple[str, list[Path]]] = []
    for commit in _commits_in_range(from_ref, to_ref):
        parent = _commit_parent(commit)
        if parent is None:
            print(f"check-b: {_short(commit)}: no in-repo predecessor (root commit); skipped")
            continue
        parent_v = _layout_version_at(parent)
        commit_v = _layout_version_at(commit)
        if commit_v > parent_v:
            # `check-a`'s range arm validates a bump; this check grades the commits that do
            # NOT carry one. An INCREASE is the only shape `check-a` accepts, so a DECREASE
            # must NOT be skipped here -- it is failed by check-a and its own files still
            # need grading, which is why this is `>` and not `!=`.
            print(
                f"check-b: {_short(commit)}: LAYOUT_VERSION bumped ({parent_v}->{commit_v}); "
                f"check-a covers this commit ({from_ref}..{to_ref})"
            )
            continue
        hits = _scan_changed_files(
            _changed_files(parent, commit),
            paths,
            globs,
            allow,
            # THE RANGE VENUE'S CONTENT SOURCE -- the graded commit's blob, never the worktree.
            lambda rel, c=commit: _file_content_hash_at(c, rel),
        )
        if hits:
            failures.append((commit, hits))

    failed = False
    if failures:
        print(
            "check-b: FAIL - layout-relevant files changed without LAYOUT_VERSION bump:",
            file=sys.stderr,
        )
        for commit, hits in failures:
            print(f"  in commit {_short(commit)} ({from_ref}..{to_ref}):", file=sys.stderr)
            for p in hits:
                print(f"  - {p.relative_to(REPO_ROOT)}", file=sys.stderr)
        _print_resolution_paths(to_v)
        failed = True

    if SLUG_FUNC_SENTINEL not in allow:
        base_hash = _slug_hash_at(from_ref)
        head_hash = _slug_hash_at(to_ref)
        if base_hash is not None and head_hash is not None and base_hash != head_hash:
            print(
                f"check-b: FAIL - {SLUG_FUNC_SENTINEL} drift would invalidate V0001's "
                f"slug derivation; bump LAYOUT_VERSION and ship V{to_v + 1:04d} that "
                f"supersedes V0001's slug logic, OR add {SLUG_FUNC_SENTINEL} to "
                "non_breaking_allowlist with author justification for why the refactor "
                "preserves slug semantics.",
                file=sys.stderr,
            )
            failed = True

    if failed:
        return 1
    print("check-b: no layout-relevant changes; pass")
    return 0


def check_c(base_ref: str) -> int:
    sentinel = _load_sentinel()
    paths = set(sentinel["layout_relevant"]["paths"])
    globs = sentinel["layout_relevant"]["globs"]
    allow = _load_allowlist(sentinel)
    suspicious: list[Path] = []
    for p in _added_files(base_ref):
        rel = str(p.relative_to(REPO_ROOT))
        if not rel.startswith("src/hhemt/") or not rel.endswith(".py"):
            continue
        if rel in paths or rel in allow:
            continue
        if any(_layout_glob_match(rel, g) for g in globs):
            continue
        name = p.name.lower()
        if any(s in name for s in LAYOUT_SUSPICIOUS_SUBSTRINGS):
            suspicious.append(p)
    for p in suspicious:
        rel = p.relative_to(REPO_ROOT)
        print(
            f"check-c: WARNING - layout-suspicious new file {rel} is not in "
            "_layout_relevant_files.yaml; either add it to layout_relevant.paths "
            "(if it touches on-disk state) or to non_breaking_allowlist (with "
            "justification).",
            file=sys.stderr,
        )
    if not suspicious:
        print("check-c: no layout-suspicious new files; pass")
    return 0


_ALLOWLIST_KEY = "non_breaking_allowlist"


def _top_level_key(line: str) -> str | None:
    """The top-level mapping key a column-0 line opens, or None when it opens none.

    Only a column-0, non-comment, non-sequence line opens a top-level key. A line
    beginning with whitespace or `-` continues a value already opened above it, and a
    `#` line is a comment. This is the whole of the YAML parsing this module does for
    the append POSITION -- deliberately not `yaml.safe_load` of the existing file, whose
    key ORDER is exactly what a loader discards.
    """
    if not line or line[0] in " \t#-":
        return None
    head, sep, _ = line.partition(":")
    if not sep or not head.strip():
        return None
    return head.strip()


def _render_allowlist_row(path: str, signature: str, why: str) -> str:
    """One `non_breaking_allowlist` mapping row in the file's own block style.

    Style is copied from the manifest rather than invented: `- path:` at two spaces,
    `layout_signature` then `justification`, and the justification as a block scalar at
    six spaces. Key ORDER is `path`, `layout_signature`, `justification`, which is the
    order 4,000+ of this file's rows already use.

    `|-` LITERAL, not `>-` folded, and the difference is the whole of this function's
    contract. A folded scalar turns every newline into a space and every blank line into
    a single newline, so `--why $'A\\nB'` would be recorded as `A B` -- a DIFFERENT string
    from the one the operator supplied, silently, and still non-empty, so `_load_allowlist`
    accepts it and nothing downstream reports the loss. The justification is the entire
    audit trail a human reads when adjudicating a change-scoped exemption, so a bullet
    list run together into one line destroys exactly what the row exists to carry.
    `|-` (literal, strip-chomping) parses back to the supplied text with its line breaks
    intact; strip-chomping is right because `why` is already `.strip()`ed by the caller.

    Rows already in the manifest use `>-` and are left alone -- this changes only what the
    command WRITES, which is the append-log's newest row. `|-` needs no escaping either:
    every `--why` line is emitted behind six spaces of indentation, so a colon, a `#`, a
    quote, or a leading `-` inside the prose is content rather than syntax. The one thing a
    block scalar cannot carry is a leading TAB, and that is caught downstream by the
    assembled-document parse rather than by a special case here -- see `_stamp_text`.
    """
    body = []
    for raw in why.splitlines():
        # No `.rstrip()` and no `.strip()` test: a line with trailing spaces is content, and
        # an all-whitespace line is content too. Emitting them verbatim is what "recorded
        # as the text supplied" means; only a genuinely EMPTY line becomes an empty line.
        body.append(("      " + raw) if raw else "")
    return (
        "\n".join(
            [
                f"  - path: {path}",
                f"    layout_signature: {signature}",
                "    justification: |-",
                *body,
            ]
        )
        + "\n"
    )


def _stamp_text(text: str, path: str, signature: str, why: str) -> str:
    """Return `text` with ONE new allowlist row appended at END OF FILE.

    TEXTUAL, and the word is load-bearing. The manifest is 4,400+ lines carrying 232
    `justification:` blocks and a header comment, so a `yaml.safe_load` +
    `yaml.safe_dump` round-trip destroys every comment and reflows every block scalar on
    its FIRST invocation. Nothing here parses to edit; the only parse is the
    read-only verification in `stamp_allowlist_row`, which reads the CANDIDATE text and
    never writes its representation back.

    APPEND, never mutate. The file is an append-log: one row per change, each carrying
    its own justification. Rewriting a pre-existing row's `layout_signature` in place
    would leave that row's justification describing an earlier change -- the gate would
    pass while the audit trail silently lied. Position does not affect RESOLUTION (the
    loader keys on `path` and is last-wins) but the file is grouped by path in places, so
    an interleaved insert would break that grouping for a human reader while changing
    nothing for the loader.
    """
    out = text if text.endswith("\n") else text + "\n"
    keys = [k for k in (_top_level_key(line) for line in out.splitlines()) if k is not None]
    if _ALLOWLIST_KEY not in keys:
        out += f"\n{_ALLOWLIST_KEY}:\n"
    out += _render_allowlist_row(path, signature, why)
    return out


def stamp_allowlist_row(path_arg: str, why: str | None) -> int:
    """`--stamp {path} --why {text}`: append one non-breaking row for `path`.

    Exits non-zero and writes NOTHING unless every readiness predicate holds. The order
    of the predicates is the order in which a violation would be MOST expensive to
    discover late:

    1. `--why` non-empty. A mapping row with no non-empty `justification` is REFUSED by
       `_load_allowlist` before any signature logic runs, so a signature-only append takes
       `check-b` AND `check-c` offline AT LOAD, install-wide, until a human edits the YAML
       by hand. This is the single cheapest way to break the gate, and it is why the
       absence of `--why` is a refusal rather than a defaulted justification.
    2. `path` resolves inside the repo and is NOT the manifest itself. See the refusal
       below for why this one is a refusal rather than a post-write re-hash.
    3. `path` EXISTS. `_file_content_hash` returns None for an absent file, and a row whose
       `layout_signature` is null on a path with no earlier row resolves to None -- which is
       the same install-wide load refusal as (1), reached by a different road.
    4. `non_breaking_allowlist` is the LAST top-level key in the file, or is absent. The
       row is appended at END OF FILE, so a manifest whose allowlist is followed by another
       top-level key would have the row land under THAT key -- a silent semantic
       relocation, not a syntax error.
    5. the assembled document PARSES and its last allowlist row is OURS. This is what makes
       the `non_breaking_allowlist: []` flow form, a leading tab in the prose, and any
       other assembly hazard a refusal instead of a corrupted 4,400-line audit trail.

    `why` is `None` rather than a defaulted empty string so that `--why ""` and an
    absent `--why` are the same refusal, for the same measured reason.
    """
    if why is None or not why.strip():
        print(
            "check_layout_version: --stamp requires a non-empty --why justification. "
            "A mapping row without one is REFUSED by _load_allowlist before any signature "
            "logic runs, which takes check-b and check-c offline at load, install-wide, "
            "until a human edits this YAML by hand. Nothing was written.",
            file=sys.stderr,
        )
        return 1

    rel = Path(path_arg)
    if rel.is_absolute() or ".." in rel.parts:
        print(
            f"check_layout_version: --stamp takes a repo-relative path; {path_arg!r} is "
            f"absolute or escapes the repo root. Nothing was written.",
            file=sys.stderr,
        )
        return 1
    # THE SELF-TARGET REFUSAL, and it is a refusal because the alternative is ARITHMETICALLY
    # IMPOSSIBLE rather than merely awkward. `--stamp` records sha256(target) and then
    # appends to the manifest. When the target IS the manifest, the append changes the very
    # file whose digest was just taken, so the recorded signature is stale the instant the
    # write completes: the command prints `stamped ...` and exits 0 while check-b stays red
    # on exactly the path it was asked to fix. Measured on a fixture: recorded 6a487c62...
    # against an actual 72787b6e..., and a SECOND run recorded 72787b6e... against an actual
    # 4863555c... -- exit 0 both times, staleness moved one append forward per run.
    #
    # "Append first, then record the post-append digest" is NOT a repair here: the row
    # carries the digest of a document that CONTAINS that digest, so a correct row would
    # need `S == sha256(document containing S)` -- a fixed point of sha256 over a structure
    # holding its own output, which cannot be found. That is why this branch refuses instead
    # of re-hashing, and the non-convergence is a property of the operand rather than a cost
    # that a second pass could pay down.
    #
    # The capability cost is ZERO, measured rather than assumed: `_layout_relevant_files.yaml`
    # is governed by neither `layout_relevant.paths` nor `layout_relevant.globs`, so it never
    # needs an exemption row of its own and there is no regress to break.
    if (REPO_ROOT / rel).resolve() == SENTINEL_PATH.resolve():
        print(
            f"check_layout_version: --stamp refuses to stamp {SENTINEL_RELPATH} itself. The "
            f"command records the target's sha256 and then appends to this file, so a "
            f"self-stamp records a digest that the append itself invalidates -- and no "
            f"re-run converges, because a correct row would have to contain the digest of a "
            f"document that contains it. Nothing was written. This file is governed by "
            f"neither layout_relevant.paths nor layout_relevant.globs, so it needs no "
            f"allowlist row.",
            file=sys.stderr,
        )
        return 1
    signature = _file_content_hash(REPO_ROOT / rel)
    if signature is None:
        print(
            f"check_layout_version: --stamp target {path_arg!r} does not exist under "
            f"{REPO_ROOT}. A row for an absent path carries no signature, which resolves to "
            f"None and is refused by _load_allowlist at load. Nothing was written.",
            file=sys.stderr,
        )
        return 1

    if not SENTINEL_PATH.exists():
        print(
            f"check_layout_version: --stamp found no {SENTINEL_RELPATH} at {REPO_ROOT}. Nothing was written.",
            file=sys.stderr,
        )
        return 1
    text = SENTINEL_PATH.read_bytes().decode("utf-8")
    keys = [k for k in (_top_level_key(line) for line in text.splitlines()) if k is not None]
    if _ALLOWLIST_KEY in keys and keys[-1] != _ALLOWLIST_KEY:
        print(
            f"check_layout_version: --stamp appends at END OF FILE, but {SENTINEL_RELPATH} "
            f"carries top-level key {keys[-1]!r} AFTER {_ALLOWLIST_KEY!r}. Appending would "
            f"file the new row under {keys[-1]!r}. Move {_ALLOWLIST_KEY!r} to the end of the "
            f"file, or add the row by hand. Nothing was written.",
            file=sys.stderr,
        )
        return 1

    new_text = _stamp_text(text, rel.as_posix(), signature, why.strip())
    try:
        parsed = yaml.safe_load(new_text)
    except yaml.YAMLError as exc:
        print(
            f"check_layout_version: refusing to write -- the assembled document does not "
            f"parse, so the appended row would corrupt {SENTINEL_RELPATH}: {exc}",
            file=sys.stderr,
        )
        return 1
    allow = (parsed or {}).get(_ALLOWLIST_KEY) or []
    last = allow[-1] if allow else None
    if not isinstance(last, dict) or last.get("path") != rel.as_posix() or last.get("layout_signature") != signature:
        print(
            f"check_layout_version: refusing to write -- the appended row did not resolve as "
            f"the last {_ALLOWLIST_KEY} entry with path {rel.as_posix()!r} and the current "
            f"sha256. This means the manifest's allowlist is not a plain block sequence "
            f"(a `[]` flow value is the common cause). Nothing was written.",
            file=sys.stderr,
        )
        return 1

    SENTINEL_PATH.write_bytes(new_text.encode("utf-8"))
    print(f"check-layout-version: stamped {rel.as_posix()} at {signature}")
    return 0


CHECK_B_DEFAULT_BASE = "main"
PRE_PUSH_FROM_ENV = "PRE_COMMIT_FROM_REF"
PRE_PUSH_TO_ENV = "PRE_COMMIT_TO_REF"
PRE_PUSH_DEFAULT_FROM = "HEAD~1"
PRE_PUSH_DEFAULT_TO = "HEAD"


def _resolve_check_b_population(base_ref: str | None, to_ref: str | None, range_mode: bool) -> tuple[str, str]:
    """`(from_ref, to_ref)` for check-b, with the RANGE venue's population from the environment.

    THE ENTRY CARRIES NO REF, and this function is why. pre-commit exports
    `PRE_COMMIT_FROM_REF`/`PRE_COMMIT_TO_REF` to a pre-push hook, and reading them HERE -- in
    the child process -- is correct whether or not pre-commit shell-expands `entry`, because
    environment inheritance holds under either exec model, whereas a literal
    `$PRE_COMMIT_FROM_REF` written into `entry` would be correct under only one of them.
    Nothing in this repository read either variable before this change, so the plumbing is new
    on both sides.

    THE DEFAULTS ARE PER VENUE, and the asymmetry is load-bearing rather than tidy. `main` is
    a sensible base for the pending-change venue (a developer comparing against the trunk) and
    a DISASTROUS one for `--range`, where it is precisely the refuted wide-base mechanism. When
    neither the CLI nor the environment supplies a range, both ends fall back to the safe
    ONE-COMMIT population, so a manual `pre-commit run --hook-stage pre-push` outside a push
    grades one commit rather than crashing or silently widening.
    """
    if range_mode:
        return (
            base_ref or os.environ.get(PRE_PUSH_FROM_ENV) or PRE_PUSH_DEFAULT_FROM,
            to_ref or os.environ.get(PRE_PUSH_TO_ENV) or PRE_PUSH_DEFAULT_TO,
        )
    return (base_ref or CHECK_B_DEFAULT_BASE, to_ref or PRE_PUSH_DEFAULT_TO)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stamp",
        metavar="PATH",
        default=None,
        help="Append a non_breaking_allowlist row for the repo-relative PATH, carrying the "
        "file's current sha256 as layout_signature and --why as the justification. Takes no "
        "subcommand. Appends at END OF FILE and never mutates a pre-existing row.",
    )
    parser.add_argument(
        "--why",
        metavar="TEXT",
        default=None,
        help="The justification recorded on the row --stamp appends, VERBATIM: a multi-line "
        "TEXT keeps its line breaks (a literal block scalar), so a bullet list stays a bullet "
        "list. Required with --stamp: a row without a non-empty justification is refused by "
        "the loader at load.",
    )
    # `required=False` is --stamp's cost, and the refusal below restores the guarantee it
    # removes. A `required=True` subparser would make `--stamp` unusable without a
    # subcommand, which is the whole point of the mode; resolving the requirement by hand
    # keeps every existing invocation's failure loud (`a subcommand is required`).
    sub = parser.add_subparsers(dest="cmd", required=False)
    a = sub.add_parser("check-a")
    a.add_argument("base_ref", nargs="?", default="main")
    a.add_argument(
        "--range",
        action="store_true",
        dest="range_mode",
        help="grade the COMMIT RANGE base..HEAD instead of base..worktree. Required at any "
        "range-based venue (pre-push, CI, a clean-checkout run): the default reads the "
        "version, the migration module and the fixtures from the WORKING TREE.",
    )
    b = sub.add_parser("check-b")
    b.add_argument("base_ref", nargs="?", default=None)
    b.add_argument(
        "--range",
        action="store_true",
        dest="range_mode",
        help="grade the COMMIT RANGE base_ref..--to instead of base_ref..worktree, walking each "
        "commit and grading its own diff. The population defaults to the ref pair pre-commit "
        "exports (PRE_COMMIT_FROM_REF/PRE_COMMIT_TO_REF), and to the safe one-commit "
        "HEAD~1..HEAD when those are unset.",
    )
    b.add_argument(
        "--to",
        metavar="REF",
        dest="to_ref",
        default=None,
        help="The range's upper end for --range. Defaults to PRE_COMMIT_TO_REF, then HEAD. Ignored outside --range.",
    )
    c = sub.add_parser("check-c")
    c.add_argument("base_ref", nargs="?", default="main")
    args = parser.parse_args()
    if args.stamp is not None:
        return stamp_allowlist_row(args.stamp, args.why)
    if args.why is not None:
        parser.error("--why is only meaningful with --stamp")
    if args.cmd is None:
        parser.error("a subcommand is required unless --stamp is given")
    if args.cmd == "check-a":
        return check_a(args.base_ref, range_mode=args.range_mode)
    elif args.cmd == "check-b":
        base, to = _resolve_check_b_population(args.base_ref, args.to_ref, args.range_mode)
        return check_b(base, range_mode=args.range_mode, to_ref=to)
    elif args.cmd == "check-c":
        return check_c(args.base_ref)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
