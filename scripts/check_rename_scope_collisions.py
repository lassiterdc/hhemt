#!/usr/bin/env python3
"""Pre-flight scope-collision check for a proposed identifier rename map.

WHY THIS EXISTS. Every other check on a rename sweep asks whether it is COMPLETE.
This one asks whether it is CORRECT, and the failure it catches is the opposite
shape: the rename applies cleanly, greps clean afterwards, and silently changes
what a name denotes. If ``old`` and ``new`` are BOTH already bound in the same
scope, renaming ``old`` to ``new`` merges two distinct bindings into one. The edit
is textually valid, passes every anchor check, and the collision does not exist in
the pre-sweep tree, so nothing in the diff reveals it.

See the knowledge doc "a rename map can emit a name already bound in the same
scope and only an ast pass over the proposed map can see it".

WHAT IT CANNOT SEE, stated plainly:
  * a collision introduced through a star-import or a dynamic ``globals()`` write
  * a collision against a name bound only at runtime (``setattr``, ``exec``)
  * an ATTRIBUTE collision (``self.old`` vs ``self.new``), which is a different
    analysis over class bodies and is not attempted here
  * semantic shadowing that is legal but confusing (a parameter shadowing a
    module global), which is a lint concern rather than a rename defect

USAGE:
    python check_rename_scope_collisions.py --map sa_id=member_id --map sa_ids=member_ids src tests
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

_SCOPE_NODES = (
    ast.Module,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Lambda,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
)


def _scope_label(node: ast.AST) -> str:
    if isinstance(node, ast.Module):
        return "<module>"
    name = getattr(node, "name", None)
    if name:
        return f"{type(node).__name__}:{name}"
    return f"{type(node).__name__}@{getattr(node, 'lineno', 0)}"


def _bindings_in_scope(scope: ast.AST) -> dict[str, int]:
    """Names BOUND directly in this scope, mapped to first binding line.

    Descends into the scope body but stops at any nested scope, because a nested
    scope's bindings are not the enclosing scope's bindings and counting them
    would manufacture collisions that cannot occur.
    """
    bound: dict[str, int] = {}

    def record(name: str, lineno: int) -> None:
        bound.setdefault(name, lineno)

    def visit(node: ast.AST, is_root: bool) -> None:
        if not is_root and isinstance(node, _SCOPE_NODES):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                record(node.name, node.lineno)
            return
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store | ast.Del):
            record(node.id, node.lineno)
        elif isinstance(node, ast.arg):
            record(node.arg, node.lineno)
        elif isinstance(node, ast.Import | ast.ImportFrom):
            for alias in node.names:
                record((alias.asname or alias.name).split(".")[0], node.lineno)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            record(node.name, node.lineno)
        elif isinstance(node, ast.Global | ast.Nonlocal):
            for n in node.names:
                record(n, node.lineno)
        for child in ast.iter_child_nodes(node):
            visit(child, False)

    if isinstance(scope, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
        for arg in scope.args.posonlyargs + scope.args.args + scope.args.kwonlyargs:
            record(arg.arg, arg.lineno)
        for extra in (scope.args.vararg, scope.args.kwarg):
            if extra is not None:
                record(extra.arg, extra.lineno)
    visit(scope, True)
    return bound


def scan_module(path: Path, rename_map: dict[str, str]) -> list[tuple[str, int, str, str]]:
    """Return (scope_label, lineno, old, new) for every would-be collision."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        return []
    out: list[tuple[str, int, str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, _SCOPE_NODES):
            continue
        bound = _bindings_in_scope(node)
        for old, new in rename_map.items():
            if old in bound and new in bound:
                out.append((_scope_label(node), max(bound[old], bound[new]), old, new))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scope-collision pre-flight for a rename map.")
    parser.add_argument("roots", nargs="*", default=["src", "scripts", "tests"])
    parser.add_argument(
        "--map",
        action="append",
        required=True,
        metavar="OLD=NEW",
        help="one proposed rename; repeat for each entry in the map",
    )
    args = parser.parse_args(argv)

    rename_map: dict[str, str] = {}
    for entry in args.map:
        if "=" not in entry:
            parser.error(f"--map expects OLD=NEW, got {entry!r}")
        old, new = entry.split("=", 1)
        rename_map[old] = new

    files: list[Path] = []
    for root in args.roots:
        files.extend(sorted(Path(root).rglob("*.py")))

    n = 0
    for path in files:
        for scope, lineno, old, new in scan_module(path, rename_map):
            n += 1
            print(f"COLLISION {path}:{lineno}  scope={scope}  {old} -> {new} (both already bound)")

    print(f"\nscope-collision scan: {len(files)} file(s); {n} collision(s); map={rename_map}")
    return 1 if n else 0


if __name__ == "__main__":
    sys.exit(main())
