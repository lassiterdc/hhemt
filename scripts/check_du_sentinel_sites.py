#!/usr/bin/env python3
"""CI check enforcing the DU-sentinel mutation-site restamp contract.

Implements the DELETION_NOT_ROUTED_THROUGH_TOOL audit rule (the route-based successor
of the retired MUTATION_SITE_MISSING_RESTAMP / FAST_RMTREE_MISSING_ANALYSIS_DIR rules)
under the stipulation `du sentinels written at every mutation site.md`. Pure-stdlib
ast.NodeVisitor; mirrors scripts/check_layout_version.py. Full-corpus scan of
src/hhemt/**/*.py (NOT git-diff-scoped). Exit 0 = clean, 1 = >=1
failure.

Rules:
  DELETION_NOT_ROUTED_THROUGH_TOOL  a fast_rmtree / .unlink / shutil.rmtree outside
                                    du_sentinels.py that is neither the tool nor exempt
  EXEMPT_MISSING_CATEGORY           bare `# EXEMPT-DU:` with no category
  EXEMPT_UNKNOWN_CATEGORY           category not in EXEMPT_CATEGORIES
  EXEMPT_ORPHAN                     exempt comment with no associated mutation (warn-only)
  RAW_RMTREE_UNMAINTAINED           raw shutil.rmtree(...) outside du_sentinels.py, un-annotated
                                    (warn-tier by design: raw shutil.rmtree is only ever used
                                    outside an analysis scope; annotate, never route)
  UNCLASSIFIED_MUTATION             a declared FS_MUTATORS name with no rule (warn-only)

status-flag: {scope}/_status/** bytes are never DU-counted at ANY depth (du_sentinels
clause 6), so unlinking a flag changes zero counted bytes at every scope -- not only the
top rollup.
"""

from __future__ import annotations

import argparse
import ast
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src" / "hhemt"

EXEMPT_CATEGORIES = frozenset(
    {
        "full-analysis-root-wipe",
        "cli-helper-full-wipe",
        "lock-file-cleanup",
        "status-dir-cleanup",
        "status-flag",
        "system-dir",
        "bundle-root",
        "delete-workflow-leaf",
        "test-example-fixture",
        "canonical-helper",
        # Added Phase 1 (2026-06-13) after the full-corpus Bucket-2 triton classification:
        # "du-handled-by-decrement" RETIRED 2026-09-13 (clause 9): the tool performs
        # the decrement, so every site that carried this category is now a tool call.
        "dry-run-trigger",  # a reprocess dry_run deletes report/plot artifacts as the rerun
        # trigger but MUST NOT write _du.json (reprocess-dry_run stipulation); the sentinel
        # is corrected by the real run.
        "transient-intermediate",  # write-staging intermediate (e.g. an intermediate .zarr in a
        # generic helper) deleted within its creating call, never
        # observed by a committed sentinel; durable DU computed downstream.
        "outside-analysis-tree",  # delete of a path with no analysis-tree DU semantics
        # (e.g. a HOME-dir credential cache).
        "migration-primitive",  # out-of-band version-migration primitive operating on arbitrary
        # paths; migrations own the on-disk layout (incl. _du.json) by contract.
    }
)
EXEMPT_TOKEN = "# EXEMPT-DU:"

#: Declared filesystem-mutating vocabulary. A resolved name in this set with no
#: branch in _check_stmt is reported UNCLASSIFIED_MUTATION rather than ignored --
#: the inversion that makes the NEXT blind spot loud instead of silent.
FS_MUTATORS: frozenset[str] = frozenset(
    {
        "fast_rmtree",
        "unlink",
        "shutil.rmtree",
        "shutil.copytree",
        "shutil.move",
        "os.remove",
        "os.unlink",
        "os.rmdir",
        "os.removedirs",
        "os.replace",
        "os.rename",
        "os.truncate",
        "shutil.copy",
        "shutil.copy2",
        "shutil.copyfile",
    }
)
#: The names _check_stmt branches on. Paired with DELIBERATELY_UNHANDLED by the
#: vocabulary self-test in tests/test_check_du_sentinel_sites_vocabulary.py.
_BRANCHED_NAMES: frozenset[str] = frozenset({"fast_rmtree", "unlink", "shutil.rmtree"})
#: Vocabulary members with no enforcing branch, each with the reason.
DELIBERATELY_UNHANDLED: dict[str, str] = {
    "os.replace": "atomic swap; net delta needs a post-replace restamp, not an adjacent one",
    "os.rename": "same as os.replace",
    "shutil.move": "relocation; one restamp is correct at one endpoint and wrong at the other",
    "shutil.copy": "growth; self-corrects at the next rollup",
    "shutil.copy2": "growth; self-corrects at the next rollup",
    "shutil.copyfile": "growth; self-corrects at the next rollup",
    "shutil.copytree": "growth; self-corrects at the next rollup",
    "os.truncate": "no occurrence in src/hhemt; vocabulary member for forward coverage",
    "os.removedirs": "no occurrence in src/hhemt; vocabulary member for forward coverage",
    "os.rmdir": "no occurrence in src/hhemt; vocabulary member for forward coverage",
    "os.remove": "no occurrence in src/hhemt; vocabulary member for forward coverage",
    "os.unlink": "no occurrence in src/hhemt; vocabulary member for forward coverage",
}

# (file-relpath basename, funcname) bodies that ARE the sanctioned implementation — never scanned.
_CANONICAL_HELPER_FUNCS = frozenset(
    {
        ("utils.py", "fast_rmtree"),
        ("du_sentinels.py", "delete_and_account"),
    }
)

WARN_ONLY_RULES = frozenset({"EXEMPT_ORPHAN", "UNCLASSIFIED_MUTATION", "RAW_RMTREE_UNMAINTAINED"})


@dataclass(frozen=True)
class Violation:
    path: str  # repo-relative
    line: int
    rule_id: str
    message: str

    def render(self) -> str:
        return f"{self.path}:{self.line}: {self.rule_id}: {self.message}"


def _build_exempt_map(text: str) -> dict[int, str | None]:
    """1-indexed lineno -> category (str) or None for a bare `# EXEMPT-DU:`.

    Uses tokenize (not a regex line-scan) so a `# EXEMPT-DU:` substring inside a
    string/docstring never registers as a suppression.
    """
    out: dict[int, str | None] = {}
    try:
        tokens = tokenize.generate_tokens(iter(text.splitlines(keepends=True)).__next__)
        for tok in tokens:
            if tok.type == tokenize.COMMENT and EXEMPT_TOKEN in tok.string:
                body = tok.string.split(EXEMPT_TOKEN, 1)[1].strip()
                out[tok.start[0]] = body or None  # None == bare/categoryless
    except tokenize.TokenError:
        pass
    return out


def _import_alias_map(tree: ast.AST) -> dict[str, str]:
    """local_name -> canonical_name for fast_rmtree / delete_and_account,
    plus module aliases (e.g. `import ...utils as u` -> 'u' -> '<module:utils>')."""
    aliases: dict[str, str] = {}
    canonical = {"fast_rmtree", "delete_and_account"}
    _FS_MODULES = {"os", "shutil", "subprocess", "pathlib"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for a in node.names:
                if a.name in canonical:
                    aliases[a.asname or a.name] = a.name
            if node.module in _FS_MODULES:
                for a in node.names:
                    aliases[a.asname or a.name] = f"{node.module}.{a.name}"
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.endswith(".utils") or a.name.endswith("du_sentinels"):
                    aliases[a.asname or a.name.split(".")[-1]] = f"<module:{a.name.split('.')[-1]}>"
            for a in node.names:
                if a.name in _FS_MODULES:
                    aliases[a.asname or a.name] = f"<fsmod:{a.name}>"
    return aliases


def _resolve_call_name(call: ast.Call, aliases: dict[str, str]) -> str | None:
    """Return canonical name ('fast_rmtree' / 'delete_and_account' /
    'unlink' / 'rm') for a Call node, or None if not a tracked mutation."""
    func = call.func
    if isinstance(func, ast.Name):
        return aliases.get(func.id, func.id)
    if isinstance(func, ast.Attribute):
        recv = func.value
        if isinstance(recv, ast.Name):
            bound = aliases.get(recv.id)
            if bound is not None and bound.startswith("<fsmod:"):
                return f"{bound[len('<fsmod:') : -1]}.{func.attr}"
        # p.unlink()  OR  module.fast_rmtree()  OR subprocess.run(["rm", ...])
        return func.attr
    return None


def _statement_call(stmt: ast.stmt) -> ast.Call | None:
    """Return the Call node when `stmt` is an expression-statement whose value is
    a Call (the mutation-site shape: `fast_rmtree(x)`, `p.unlink()`,
    `delete_and_account(...)`). Otherwise None.

    Mutation sites in this codebase are always bare `ast.Expr(Call)` statements,
    so detection at the statement level is exact.
    """
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
        return stmt.value
    return None


class _MutationSiteVisitor(ast.NodeVisitor):
    """Walks statement-body lists; each mutation statement is judged on its own
    (routed through du_sentinels.delete_and_account, or exempt), keyed against the
    exempt map."""

    def __init__(self, relpath: str, aliases: dict[str, str], exempt: dict[int, str | None]):
        self.relpath = relpath  # repo-relative path, used for Violation.path
        self.aliases = aliases
        self.exempt = exempt
        self.violations: list[Violation] = []
        self.used_exempt_lines: set[int] = set()
        self._basename = relpath.split("/")[-1]
        self._skip_funcs = {fn for (f, fn) in _CANONICAL_HELPER_FUNCS if f == self._basename}

    # ---- exempt-association helpers (FQ3) --------------------------------

    def _exempt_lines_for(self, stmt: ast.stmt) -> list[int]:
        """Return the exempt-map line(s) that suppress `stmt`, if any.

        A mutation at line L is suppressed by an exempt comment on the line
        directly above (own-line, L-1) or trailing on any line of the statement
        (L .. end_lineno) — mirroring the `# noqa` / `# type: ignore` dual
        same-line/next-line idiom, extended to multi-line statements.
        """
        end = getattr(stmt, "end_lineno", None) or stmt.lineno
        candidates = {stmt.lineno - 1, *range(stmt.lineno, end + 1)}
        return [ln for ln in candidates if ln in self.exempt]

    def _is_exempt(self, stmt: ast.stmt) -> bool:
        return bool(self._exempt_lines_for(stmt))

    def _mark_exempt_used(self, stmt: ast.stmt) -> None:
        for ln in self._exempt_lines_for(stmt):
            self.used_exempt_lines.add(ln)

    # ---- per-statement detection (route predicate) ------------------------

    def _check_stmt(self, stmt: ast.stmt) -> None:
        """Detect a direct fast_rmtree / .unlink / shutil.rmtree outside du_sentinels.py
        that is not exempt (DELETION_NOT_ROUTED_THROUGH_TOOL / RAW_RMTREE_UNMAINTAINED).
        Honor the exempt map on the statement's first line / line above / trailing
        line. Record used exempt lines so orphan-exempts can be reported.
        """
        call = _statement_call(stmt)
        if call is None:
            return
        name = _resolve_call_name(call, self.aliases)

        if name == "fast_rmtree":
            # RULING-5 FINDING, recorded here so it is not lost: the predicate this
            # replaces was `if "analysis_dir" in kwarg_names: return` -- a keyword-NAME
            # presence test that never read the value, so `fast_rmtree(p,
            # analysis_dir=None)` passed it and so did the per-file restamp storm. It
            # was fail-open on value. The predicate is now ROUTE-based: a deletion is
            # accounted for iff it goes through du_sentinels.delete_and_account.
            # INVARIANT this early return rests on: du_sentinels.py contains EXACTLY ONE
            # `fast_rmtree(` call, inside delete_and_account. Measuring command:
            #   grep -c "fast_rmtree(" src/hhemt/du_sentinels.py   -> 1
            # A second call there would be exempted by this line without review.
            if self.relpath.endswith("du_sentinels.py"):
                return  # the tool's own primitive call
            if self._is_exempt(stmt):
                self._mark_exempt_used(stmt)
                return
            self.violations.append(
                Violation(
                    self.relpath,
                    stmt.lineno,
                    "DELETION_NOT_ROUTED_THROUGH_TOOL",
                    "direct fast_rmtree(...) outside du_sentinels.py; route the deletion "
                    "through du_sentinels.delete_and_account(paths, scope_dir=..., scope=...) "
                    "or annotate the site with `# EXEMPT-DU: {category}`",
                )
            )
            return

        if name == "shutil.rmtree":
            if self._is_exempt(stmt):
                self._mark_exempt_used(stmt)
                return
            self.violations.append(
                Violation(
                    self.relpath,
                    stmt.lineno,
                    "RAW_RMTREE_UNMAINTAINED",
                    f"raw {name}(...) is not maintained; route the deletion through "
                    "du_sentinels.delete_and_account(...) if the path is DU-counted, or annotate "
                    "with `# EXEMPT-DU: {category}`",
                )
            )
            return

        if name in DELIBERATELY_UNHANDLED:
            if self._is_exempt(stmt):
                self._mark_exempt_used(stmt)
            else:
                self.violations.append(
                    Violation(
                        self.relpath,
                        stmt.lineno,
                        "UNCLASSIFIED_MUTATION",
                        f"`{name}` is a declared filesystem mutator with no DU rule "
                        f"({DELIBERATELY_UNHANDLED[name]}); classify it or annotate it",
                    )
                )
            return

        if name == "unlink":
            if self._is_exempt(stmt):
                self._mark_exempt_used(stmt)
                return
            self.violations.append(
                Violation(
                    self.relpath,
                    stmt.lineno,
                    "DELETION_NOT_ROUTED_THROUGH_TOOL",
                    ".unlink() outside du_sentinels.py; route through "
                    "du_sentinels.delete_and_account(...) or annotate with "
                    "`# EXEMPT-DU: {category}`",
                )
            )
            return

    # ---- body-bearing visitors ------------------------------------------

    def _scan_body(self, stmts: list[ast.stmt]) -> None:
        """Check each statement of a body. Under the route predicate a site is judged on
        its own (routed through the tool, or exempt); no sibling-adjacency is consulted."""
        for stmt in stmts:
            self._check_stmt(stmt)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.name not in self._skip_funcs:
            self._scan_body(node.body)
            self.generic_visit(node)
        # do not descend into a skipped canonical-helper body

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_If(self, node: ast.If) -> None:
        self._scan_body(node.body)
        self._scan_body(node.orelse)
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:
        self._scan_body(node.body)
        for h in node.handlers:
            self._scan_body(h.body)
        self._scan_body(node.orelse)
        self._scan_body(node.finalbody)
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self._scan_body(node.body)
        self._scan_body(node.orelse)
        self.generic_visit(node)

    visit_While = visit_For

    def visit_With(self, node: ast.With) -> None:
        self._scan_body(node.body)
        self.generic_visit(node)

    visit_AsyncWith = visit_With

    def visit_Module(self, node: ast.Module) -> None:
        self._scan_body(node.body)
        self.generic_visit(node)


def _check_file(path: Path) -> list[Violation]:
    rel = str(path.relative_to(REPO_ROOT))
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    aliases = _import_alias_map(tree)
    exempt = _build_exempt_map(text)
    visitor = _MutationSiteVisitor(rel, aliases, exempt)
    visitor.visit(tree)
    violations = list(visitor.violations)
    # EXEMPT self-lint: bad categories + orphans
    for lineno, cat in exempt.items():
        if cat is None:
            violations.append(
                Violation(
                    rel,
                    lineno,
                    "EXEMPT_MISSING_CATEGORY",
                    "`# EXEMPT-DU:` has no category; name one of: " + ", ".join(sorted(EXEMPT_CATEGORIES)),
                )
            )
        elif cat not in EXEMPT_CATEGORIES:
            violations.append(
                Violation(
                    rel,
                    lineno,
                    "EXEMPT_UNKNOWN_CATEGORY",
                    f"unknown EXEMPT-DU category `{cat}`; valid: " + ", ".join(sorted(EXEMPT_CATEGORIES)),
                )
            )
        elif lineno not in visitor.used_exempt_lines:
            violations.append(
                Violation(
                    rel, lineno, "EXEMPT_ORPHAN", "`# EXEMPT-DU:` comment has no associated mutation site (warn-only)"
                )
            )
    return violations


def _iter_target_files() -> list[Path]:
    return sorted(SRC_ROOT.rglob("*.py"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args(argv)
    if args.format == "json":
        raise SystemExit("check_du_sentinel_sites: --format json not yet implemented")
    all_violations: list[Violation] = []
    for py in _iter_target_files():
        all_violations.extend(_check_file(py))
    failures = [v for v in all_violations if v.rule_id not in WARN_ONLY_RULES]
    warnings = [v for v in all_violations if v.rule_id in WARN_ONLY_RULES]
    for v in warnings:
        print(f"WARNING {v.render()}", file=sys.stderr)
    if failures:
        print("DU-sentinel mutation-site check FAILED:", file=sys.stderr)
        for v in failures:
            print(f"  {v.render()}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
