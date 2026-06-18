#!/usr/bin/env python
"""CI check enforcing the DU-sentinel mutation-site restamp contract.

Implements the MUTATION_SITE_MISSING_RESTAMP audit rule deferred by the
stipulation `du sentinels written at every mutation site.md`. Pure-stdlib
ast.NodeVisitor; mirrors scripts/check_layout_version.py. Full-corpus scan of
src/hhemt/**/*.py (NOT git-diff-scoped). Exit 0 = clean, 1 = >=1
failure.

Rules:
  FAST_RMTREE_MISSING_ANALYSIS_DIR  PATTERN A: fast_rmtree(...) lacks analysis_dir=
  MUTATION_SITE_MISSING_RESTAMP     PATTERN B: .unlink() not adjacent-followed
                                    by restamp_parent_sentinels(...)
  EXEMPT_MISSING_CATEGORY           bare `# EXEMPT-DU:` with no category
  EXEMPT_UNKNOWN_CATEGORY           category not in EXEMPT_CATEGORIES
  EXEMPT_ORPHAN                     exempt comment with no associated mutation (warn-only)
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
        "du-handled-by-decrement",  # DU-counted, but maintained by decrement_scope_sentinel(...)
        # or a non-adjacent (e.g. `if not dry_run:`-gated) restamp the
        # AST checker cannot see as an adjacent sibling.
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

# (file-relpath basename, funcname) bodies that ARE the sanctioned implementation — never scanned.
_CANONICAL_HELPER_FUNCS = frozenset(
    {
        ("utils.py", "fast_rmtree"),
        ("utils.py", "_restamp_after_mutation"),
    }
)

WARN_ONLY_RULES = frozenset({"EXEMPT_ORPHAN"})


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
    """local_name -> canonical_name for fast_rmtree / restamp_parent_sentinels,
    plus module aliases (e.g. `import ...utils as u` -> 'u' -> '<module:utils>')."""
    aliases: dict[str, str] = {}
    canonical = {"fast_rmtree", "restamp_parent_sentinels"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for a in node.names:
                if a.name in canonical:
                    aliases[a.asname or a.name] = a.name
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.endswith(".utils") or a.name.endswith("du_sentinels"):
                    aliases[a.asname or a.name.split(".")[-1]] = f"<module:{a.name.split('.')[-1]}>"
    return aliases


def _resolve_call_name(call: ast.Call, aliases: dict[str, str]) -> str | None:
    """Return canonical name ('fast_rmtree' / 'restamp_parent_sentinels' /
    'unlink' / 'rm') for a Call node, or None if not a tracked mutation/restamp."""
    func = call.func
    if isinstance(func, ast.Name):
        return aliases.get(func.id, func.id)
    if isinstance(func, ast.Attribute):
        # p.unlink()  OR  module.fast_rmtree()  OR subprocess.run(["rm", ...])
        return func.attr
    return None


def _statement_call(stmt: ast.stmt) -> ast.Call | None:
    """Return the Call node when `stmt` is an expression-statement whose value is
    a Call (the mutation/restamp-site shape: `fast_rmtree(x)`, `p.unlink()`,
    `restamp_parent_sentinels(...)`). Otherwise None.

    Mutation sites in this codebase are always bare `ast.Expr(Call)` statements;
    detecting at the statement level is what makes PATTERN-B sibling-adjacency
    (FQ1) well-defined.
    """
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
        return stmt.value
    return None


class _MutationSiteVisitor(ast.NodeVisitor):
    """Walks statement-body lists; for each mutation statement checks the next
    sibling (PATTERN B) or kwargs (PATTERN A), keyed against the exempt map."""

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

    def _next_is_restamp(self, nxt: ast.stmt | None) -> bool:
        if nxt is None:
            return False
        call = _statement_call(nxt)
        if call is None:
            return False
        return _resolve_call_name(call, self.aliases) == "restamp_parent_sentinels"

    # ---- per-statement detection (PATTERN A + B) -------------------------

    def _check_stmt(self, stmt: ast.stmt, nxt: ast.stmt | None) -> None:
        """Detect PATTERN-A (fast_rmtree missing analysis_dir) and PATTERN-B
        (.unlink without an adjacent restamp). Honor the exempt map on the
        statement's first line / line above / trailing line. Record used exempt
        lines so orphan-exempts can be reported (linting-specialist H4 FQ1/FQ2).
        """
        call = _statement_call(stmt)
        if call is None:
            return
        name = _resolve_call_name(call, self.aliases)

        if name == "fast_rmtree":
            kwarg_names = {kw.arg for kw in call.keywords if kw.arg is not None}
            if "analysis_dir" in kwarg_names:
                return  # PATTERN A satisfied
            if self._is_exempt(stmt):
                self._mark_exempt_used(stmt)
                return
            self.violations.append(
                Violation(
                    self.relpath,
                    stmt.lineno,
                    "FAST_RMTREE_MISSING_ANALYSIS_DIR",
                    "fast_rmtree(...) call lacks analysis_dir= kwarg; pass "
                    "analysis_dir=<analysis root> so parent DU sentinels are "
                    "re-stamped, or annotate the site with `# EXEMPT-DU: {category}`",
                )
            )
            return

        if name == "unlink":
            if self._next_is_restamp(nxt):
                return  # PATTERN B satisfied
            if self._is_exempt(stmt):
                self._mark_exempt_used(stmt)
                return
            self.violations.append(
                Violation(
                    self.relpath,
                    stmt.lineno,
                    "MUTATION_SITE_MISSING_RESTAMP",
                    ".unlink() mutation is not immediately followed by a "
                    "restamp_parent_sentinels(...) call; add the restamp as the next "
                    "sibling statement (or as the sole/last statement of an "
                    "immediately-following finally), or annotate with "
                    "`# EXEMPT-DU: {category}`",
                )
            )
            return

    # ---- body-bearing visitors ------------------------------------------

    def _scan_body(self, stmts: list[ast.stmt], tail_next: ast.stmt | None = None) -> None:
        """Check each statement against its next sibling. `tail_next` supplies a
        synthetic next-sibling for the LAST statement (used by visit_Try to wire
        a try-body's trailing mutation to a finally-body restamp)."""
        for i, stmt in enumerate(stmts):
            nxt = stmts[i + 1] if i + 1 < len(stmts) else tail_next
            self._check_stmt(stmt, nxt)

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
        # PATTERN-B try/finally carve-out: a mutation in `body` whose restamp is
        # the sole/last stmt of `finalbody` is compliant. Wire the finally's
        # trailing restamp as the synthetic next-sibling for the try body.
        final_restamp: ast.stmt | None = None
        if node.finalbody:
            last_final = node.finalbody[-1]
            call = _statement_call(last_final)
            if call is not None and _resolve_call_name(call, self.aliases) == "restamp_parent_sentinels":
                final_restamp = last_final
        self._scan_body(node.body, tail_next=final_restamp)
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
