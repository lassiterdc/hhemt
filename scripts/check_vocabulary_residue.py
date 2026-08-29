#!/usr/bin/env python3
"""Static residue scan for a vocabulary rename: string constants AND the names bound to them.

WHY THIS EXISTS, and why grep is not enough. A rename sweep is normally verified by
counting residual occurrences of the retired token. That count is scoped by the
instrument, and two blind spots were measured on this repository:

  1. A NAME-token count reads ``ast.Name`` and therefore cannot see an attribute
     name written as a string, e.g. ``getattr(obj, "sub_analyses", None)``. Four
     such sites shipped as silent-default defects.
  2. A STRING-constant count reads the literal and therefore cannot see a consumer
     that binds the literal to a variable first. Measured instance, workflow.py:

         tag = "sa-" if spec.scope == "sa" else "evt-"          # line 1762
         return any(f"{tag}{v}_" in name for v in spec.tokens)  # line 1763

     Line 1763 consumes the dialect and contains no retired token at all.

This pass closes both: it reports every ``ast.Constant[str]`` carrying a retired
token (DIRECT), and every load of a name whose binding RHS contained one (INDIRECT).

WHAT IT STILL CANNOT SEE -- stated because a completeness instrument that
overstates its own reach is worse than none:
  * a token assembled by concatenation whose fragments each lack it ("sa" + "-")
  * a token whose value arrives at runtime from a config file, env var, or CSV
  * taint crossing a module boundary via import
  * taint flowing through a function parameter or a return value
  * taint held in a dict value or a container element reached by subscript
Each needs dataflow analysis this pass does not attempt. Treat a clean run as
"no residue of the two shapes above", never as "no residue".

USAGE. Two distinct uses, and they want different settings:
  * DISCOVERY, against a PRE-rename tree -- every hit is a real consumer, so no
    exception list is wanted and a non-zero exit is expected and meaningless.
  * CI GATE, against a POST-rename tree -- exit 1 means undeclared residue, and
    the declared-exception file carries the sites that legitimately keep the old
    vocabulary (historical migrations, frozen fixtures).
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

DEFAULT_TOKENS = ("sa_id", "sa-", "sa_", "subanalys", "sub_analys")


def _tainted_const(node: ast.AST, tokens: tuple[str, ...]) -> str | None:
    """Return the first retired token carried by a str Constant, else None."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        for tok in tokens:
            if tok in node.value:
                return tok
    return None


def _subtree_taint(node: ast.AST, tokens: tuple[str, ...]) -> str | None:
    """First retired token anywhere in an expression subtree.

    Walking the subtree rather than testing the node is what catches the shapes a
    bare-literal check misses: a ternary (``ast.IfExp``), a tuple or list literal,
    a concatenation, an f-string, and a call argument.
    """
    for sub in ast.walk(node):
        hit = _tainted_const(sub, tokens)
        if hit is not None:
            return hit
    return None


def _target_names(target: ast.AST) -> list[str]:
    """Bound names for an assignment target, including tuple-unpack and attributes."""
    out: list[str] = []
    for sub in ast.walk(target):
        if isinstance(sub, ast.Name):
            out.append(sub.id)
        elif isinstance(sub, ast.Attribute):
            out.append(sub.attr)
    return out


def scan_module(path: Path, tokens: tuple[str, ...]) -> list[tuple[int, str, str, str]]:
    """Return (lineno, kind, token, source_line) for every residue site in one module."""
    try:
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(path))
    except (SyntaxError, UnicodeDecodeError) as exc:
        return [(0, "UNPARSED", type(exc).__name__, str(exc)[:120])]

    lines = src.splitlines()
    findings: list[tuple[int, str, str, str]] = []
    tainted_names: dict[str, str] = {}

    # Pass 1 -- direct string constants, and the names those constants taint.
    for node in ast.walk(tree):
        hit = _tainted_const(node, tokens)
        if hit is not None:
            findings.append((node.lineno, "DIRECT", hit, lines[node.lineno - 1].strip()))
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            value = getattr(node, "value", None)
            if value is None:
                continue
            tok = _subtree_taint(value, tokens)
            if tok is None:
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for tgt in targets:
                for name in _target_names(tgt):
                    tainted_names.setdefault(name, tok)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defaults = list(node.args.defaults) + [d for d in node.args.kw_defaults if d is not None]
            for default in defaults:
                tok = _subtree_taint(default, tokens)
                if tok is not None:
                    for arg in node.args.args + node.args.kwonlyargs:
                        tainted_names.setdefault(arg.arg, tok)

    # Pass 2 -- loads of a tainted name on a line that carries no retired token itself.
    # The line filter is what makes an INDIRECT hit worth reading: a load on a line
    # that already greps positive tells the reader nothing grep did not.
    direct_lines = {ln for ln, kind, _, _ in findings if kind == "DIRECT"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id in tainted_names and node.lineno not in direct_lines:
                label = f"{node.id}<-{tainted_names[node.id]}"
                findings.append((node.lineno, "INDIRECT", label, lines[node.lineno - 1].strip()))
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
            if node.attr in tainted_names and node.lineno not in direct_lines:
                label = f".{node.attr}<-{tainted_names[node.attr]}"
                findings.append((node.lineno, "INDIRECT", label, lines[node.lineno - 1].strip()))

    seen: set[tuple[int, str, str]] = set()
    deduped: list[tuple[int, str, str, str]] = []
    for ln, kind, tok, text in sorted(findings):
        key = (ln, kind, tok)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((ln, kind, tok, text))
    return deduped


def load_exceptions(path: Path | None) -> set[tuple[str, str]]:
    """Load declared exceptions as a set of (path_substring, token) pairs.

    An EMPTY file is valid and yields an empty set: ``yaml.safe_load("")``
    returns ``None``, which this coerces rather than subscripting. That is the
    intended first state of the gate -- the file ships declared-but-empty so the
    first post-rename run reports every hit as undeclared rather than crashing.
    """
    if path is None:
        return set()
    if not path.is_file():
        raise FileNotFoundError(f"declared-exception file not found: {path}")
    import yaml

    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out: set[tuple[str, str]] = set()
    for entry in doc.get("exceptions") or []:
        out.add((str(entry["path"]), str(entry["token"])))
    return out


def _is_excepted(path: Path, token: str, declared: set[tuple[str, str]]) -> bool:
    return any(sub in str(path) and (tok == token or tok == "*") for sub, tok in declared)


def main(argv: list[str] | None = None) -> int:
    """Exit codes: 0 clean, 1 undeclared residue, 2 a module could not be parsed.

    2 is DISTINCT from 1 and from 0 on purpose. An unparseable module is not a
    clean run -- the pass could not read it, so it cannot vouch for it -- and it
    is not residue either. Returning 0 there would make the gate report success
    on a corpus it never inspected, which is the failure class this instrument
    exists to detect. Measured before the fix: a directory holding one syntax
    error returned `0 DIRECT, 0 INDIRECT, 1 UNPARSED` and exit 0.
    """
    parser = argparse.ArgumentParser(description="Vocabulary-residue AST scan.")
    parser.add_argument("roots", nargs="*", default=["src", "scripts", "tests"])
    parser.add_argument("--tokens", nargs="*", default=list(DEFAULT_TOKENS))
    parser.add_argument("--exclude", nargs="*", default=["version_migration/versions"])
    parser.add_argument("--kind", choices=["all", "direct", "indirect"], default="all")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument(
        "--exceptions",
        type=Path,
        default=None,
        help="declared-exception YAML; omit for DISCOVERY, supply for the CI GATE",
    )
    args = parser.parse_args(argv)

    declared = load_exceptions(args.exceptions)
    tokens = tuple(args.tokens)
    files: list[Path] = []
    for root in args.roots:
        files.extend(sorted(Path(root).rglob("*.py")))
    files = [f for f in files if not any(x in str(f) for x in args.exclude)]

    n_direct = n_indirect = n_unparsed = n_excepted = 0
    for path in files:
        for ln, kind, tok, text in scan_module(path, tokens):
            if kind == "UNPARSED":
                n_unparsed += 1
                print(f"UNPARSED  {path}: {tok}: {text}")
                continue
            bare = tok.split("<-")[-1]
            if _is_excepted(path, bare, declared):
                n_excepted += 1
                continue
            if kind == "DIRECT":
                n_direct += 1
            else:
                n_indirect += 1
            if args.kind != "all" and kind.lower() != args.kind:
                continue
            if not args.summary_only:
                print(f"{kind:<9} {path}:{ln}  [{tok}]  {text[:112]}")

    print(
        f"\nresidue scan: {len(files)} file(s); {n_direct} DIRECT, {n_indirect} INDIRECT, "
        f"{n_excepted} EXCEPTED, {n_unparsed} UNPARSED; tokens={list(tokens)}"
    )
    if n_unparsed:
        return 2
    return 1 if (n_direct or n_indirect) else 0


if __name__ == "__main__":
    sys.exit(main())
