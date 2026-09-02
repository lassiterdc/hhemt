#!/usr/bin/env python
"""CI check enforcing the version-migration discipline.

Usage:
    python scripts/check_layout_version.py check-a [base_ref=main]
    python scripts/check_layout_version.py check-b [base_ref=main]
    python scripts/check_layout_version.py check-c [base_ref=main]

Exit 0 = pass; exit 1 = enforcement failure with structured message.
Check C is warning-only and always exits 0.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
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


def _changed_files(base_ref: str) -> list[Path]:
    """Changed paths in {base_ref}..HEAD, EXCLUDING pure git renames/copies.

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
    out = _git("diff", "--name-status", "-M", base_ref).strip()
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


def _load_sentinel() -> dict:
    return yaml.safe_load(SENTINEL_PATH.read_text())


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
    layout_signature: str | None = None  # change-scoped hash; None => path-permanent


def _load_allowlist(sentinel: dict) -> dict[str, AllowlistEntry]:
    """Parse `non_breaking_allowlist` into path -> AllowlistEntry.

    Accepts two entry forms:
      - bare string `path`                    -> AllowlistEntry(None, None)
      - mapping {path, justification[, layout_signature]}
        (justification REQUIRED and non-empty for the mapping form)
    Exits non-zero (SystemExit) on any malformed entry.
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
        out[raw["path"]] = AllowlistEntry(justification=just, layout_signature=sig)
    return out


def _file_content_hash(path: Path) -> str | None:
    """Whole-file sha256 hexdigest, or None when the file is absent.

    Returning None (rather than raising on read_bytes) means a DELETED
    allowlisted path that carries a stamped layout_signature re-fires Check-B
    (None != the pinned hash -> the change-scoped gate falls through to path/glob
    enforcement) instead of crashing the guard with FileNotFoundError.
    """
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def check_a(base_ref: str) -> int:
    resolved_base, verdict = _base_ref_or_verdict(base_ref, "check-a")
    if verdict is not None:
        return verdict
    base_v = _layout_version_at(resolved_base)
    head_v = _layout_version_at_head()
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
    expected_module = next(VERSIONS_DIR.glob(f"V{head_v:04d}__*.py"), None)
    if expected_module is None:
        print(
            f"check-a: FAIL - LAYOUT_VERSION bumped to {head_v} but no migration module "
            f"V{head_v:04d}__*.py exists in {VERSIONS_DIR}",
            file=sys.stderr,
        )
        return 1
    if not (FIXTURES_DIR / f"v{base_v}").is_dir():
        print(
            f"check-a: FAIL - fixture {FIXTURES_DIR / f'v{base_v}'} (FROM side) missing",
            file=sys.stderr,
        )
        return 1
    if not (FIXTURES_DIR / f"v{head_v}").is_dir():
        print(
            f"check-a: FAIL - fixture {FIXTURES_DIR / f'v{head_v}'} (TO side) missing",
            file=sys.stderr,
        )
        return 1
    print(f"check-a: PASS - V{head_v:04d} migration + fixtures present")
    return 0


def check_b(base_ref: str) -> int:
    sentinel = _load_sentinel()
    resolved_base, verdict = _base_ref_or_verdict(base_ref, "check-b")
    if verdict is not None:
        return verdict
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
    if head_v != pending_base_v and head_v != base_v:
        print(f"check-b: LAYOUT_VERSION bumped ({pending_base_v}->{head_v}); check-a covers this; pass")
        return 0
    paths = set(sentinel["layout_relevant"]["paths"])
    globs = sentinel["layout_relevant"]["globs"]
    allow = _load_allowlist(sentinel)
    layout_relevant_changed: list[Path] = []
    for p in _changed_files(resolved_base):
        rel = str(p.relative_to(REPO_ROOT))
        if rel in allow:
            expected_sig = allow[rel].layout_signature
            if expected_sig is None:
                continue  # legacy path-permanent exemption (no layout_signature)
            if _file_content_hash(REPO_ROOT / rel) == expected_sig:
                continue  # change-scoped exemption still valid for this content
            # signature drifted: exemption no longer covers this change -> re-fire Check-B
        if rel in paths or any(_layout_glob_match(rel, g) for g in globs):
            layout_relevant_changed.append(p)
    failed = False
    if layout_relevant_changed:
        print(
            "check-b: FAIL - layout-relevant files changed without LAYOUT_VERSION bump:",
            file=sys.stderr,
        )
        for p in layout_relevant_changed:
            print(f"  - {p.relative_to(REPO_ROOT)}", file=sys.stderr)
        print(
            "\nResolution paths:\n"
            f"  1. If breaking: bump LAYOUT_VERSION to {head_v + 1}, write "
            f"versions/V{head_v + 1:04d}__*.py, add fixtures v{head_v}/ and v{head_v + 1}/.\n"
            "  2. If non-breaking: add or update the file's non_breaking_allowlist entry in "
            "_layout_relevant_files.yaml (prefer the dict form {path, justification}). If the entry "
            "carries a layout_signature and this change is still non-breaking, re-stamp layout_signature "
            "with the file's current sha256.",
            file=sys.stderr,
        )
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


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("check-a")
    a.add_argument("base_ref", nargs="?", default="main")
    b = sub.add_parser("check-b")
    b.add_argument("base_ref", nargs="?", default="main")
    c = sub.add_parser("check-c")
    c.add_argument("base_ref", nargs="?", default="main")
    args = parser.parse_args()
    if args.cmd == "check-a":
        return check_a(args.base_ref)
    elif args.cmd == "check-b":
        return check_b(args.base_ref)
    elif args.cmd == "check-c":
        return check_c(args.base_ref)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
