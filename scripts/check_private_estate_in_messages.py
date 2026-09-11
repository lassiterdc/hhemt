#!/usr/bin/env python3
"""Refuse a user-visible message that names the operator's private deployment repository.

D106 says error and status text names the thing the USER can act on -- a config, an input
dataset, a path, a missing value -- and never the private companion repo, because that repo
is not part of this software and pointing at it gives a reader nothing to fix.

WHY AN AST PASS AND NOT A GREP. The offending phrase in _runner.py's tree-mismatch
SystemExit spans two source lines as adjacent string literals, so a line-oriented grep for it
returns 0. Python's parser concatenates adjacent literals into one ast.Constant, so an AST
pass over message strings sees the joined text and finds it.

WHY NOT THE ANONYMIZATION BLOCKLIST. That guard (scripts/check_anonymization.py) matches
genuinely-private IDENTIFIERS anywhere in a tracked file, and by its own triage rule lists
only tokens that cannot be found in public documentation. The banned word here is an ordinary
English noun this repo uses correctly in docstrings and comments; blocklisting it would fire
on 57 tracked lines repo-wide, 43 of them inside this gate's own roots, against 2 message
sites. This gate keys on POSITION -- is the string emitted to a user -- not on token privacy.

THE EMITTING SET IS THE WHOLE CONTRACT, so it is enumerated rather than described: `raise`
(any exception), `assert` messages, `print`, `SystemExit`, `warnings.warn`, `sys.exit` and any
other `.exit(...)`, `click`/`typer` `echo` and `secho`, `sys.stdout.write` / `sys.stderr.write`,
`ArgumentParser(...)` and `add_argument(...)` (argparse prints both on --help and on usage
errors), and every logger method. A shape absent from that list is NOT checked.

THE __doc__ CHANNEL, which the emitting set alone does not reach. `ArgumentParser(
description=__doc__)` emits the MODULE DOCSTRING to --help, and the docstring is not a
constant inside the call, so no walk of the call can see it. When any emitting call in a file
references `__doc__`, this pass therefore scans the module docstring as emitted text. That is
one dataflow step and it is the only one taken; a docstring reaching --help by any other route
is not seen.

WHAT THIS STILL CANNOT SEE, stated because a clean run is not "D106 is satisfied":
  * a message whose literal is bound to a variable before it is emitted (measured: this
    pass returns 0 findings on `msg = "...word..."` followed by `raise SystemExit(msg)`).
    A token assembled from adjacent fragments IS caught -- `ast.walk` reaches each constant --
    so the residual is only a token split MID-WORD.
  * any emitter not in the enumerated set above, including a third-party console wrapper.
  * any non-Python emitter -- shell scripts, Snakefile shell lines, YAML, Jinja templates.
  * a D106 violation that names the private deployment WITHOUT this token ("the private
    harness", "the deployment repo"). The token is a proxy for D106, never a test of it.

Exit 0 = clean, 1 = >=1 finding.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Every Python-bearing root. `hooks/` is included because its three modules raise errors that
#: surface as a docs-build failure, which is user-visible text like any other.
DEFAULT_ROOTS = ("src", "scripts", "hooks")

BANNED = re.compile(r"\bestates?\b", re.IGNORECASE)

_LOG_METHODS = frozenset({"debug", "info", "warning", "warn", "error", "critical", "exception", "log"})
_EMITTING_NAMES = frozenset(
    {
        "print",
        "SystemExit",
        "warn",
        "exit",
        "echo",
        "secho",
        "ArgumentParser",
        "add_argument",
    }
)
_STD_STREAMS = frozenset({"stdout", "stderr"})


@dataclass(frozen=True)
class Finding:
    path: str  # repo-relative
    line: int
    excerpt: str

    def render(self) -> str:
        return (
            f"{self.path}:{self.line}: user-visible message names the operator's private "
            f"deployment repository: {self.excerpt!r}"
        )


def _callee(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _is_std_stream_write(node: ast.Call) -> bool:
    """True for sys.stdout.write(...) / sys.stderr.write(...) and nothing else.

    Guarded on the receiver rather than on the method name: a bare `.write` test would
    match every file handle in the repository.
    """
    func = node.func
    if not (isinstance(func, ast.Attribute) and func.attr == "write"):
        return False
    recv = func.value
    return isinstance(recv, ast.Attribute) and recv.attr in _STD_STREAMS


def _is_emitting_call(node: ast.Call) -> bool:
    name = _callee(node)
    if name in _EMITTING_NAMES:
        return True
    if _is_std_stream_write(node):
        return True
    return name in _LOG_METHODS and isinstance(node.func, ast.Attribute)


def _references_module_doc(node: ast.AST) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and sub.id == "__doc__":
            return True
        if isinstance(sub, ast.Attribute) and sub.attr == "__doc__":
            return True
    return False


def _message_strings(node: ast.AST):
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            yield sub.lineno, sub.value


def _emitting_nodes(tree: ast.AST):
    """Yield (node_to_scan, reaches_module_doc) for every user-visible emission site."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Raise):
            yield node, False
        elif isinstance(node, ast.Assert):
            # Only the message. The test may legitimately compare against the token.
            if node.msg is not None:
                yield node.msg, False
        elif isinstance(node, ast.Call) and _is_emitting_call(node):
            yield node, _references_module_doc(node)


def scan_file(path: Path, repo_root: Path) -> list[Finding]:
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError, OSError):
        return []
    module_doc = ast.get_docstring(tree, clean=False)
    seen: set[tuple[int, str]] = set()
    out: list[Finding] = []

    def record(lineno: int, text: str) -> None:
        if not BANNED.search(text):
            return
        excerpt = " ".join(text.split())[:120]
        key = (lineno, excerpt)
        if key in seen:
            return
        seen.add(key)
        out.append(Finding(str(path.relative_to(repo_root)), lineno, excerpt))

    doc_emitted = False
    for node, reaches_doc in _emitting_nodes(tree):
        for lineno, text in _message_strings(node):
            record(lineno, text)
        doc_emitted = doc_emitted or reaches_doc

    if doc_emitted and module_doc:
        # Attributed to line 1: the docstring IS the module's opening statement, and the
        # reader repairing it edits there.
        record(1, module_doc)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--root", action="append", dest="roots", default=None)
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    roots = args.roots or list(DEFAULT_ROOTS)

    findings: list[Finding] = []
    for rel in roots:
        base = repo_root / rel
        if not base.is_dir():
            continue
        for py in sorted(base.rglob("*.py")):
            findings.extend(scan_file(py, repo_root))

    for finding in sorted(findings, key=lambda f: (f.path, f.line)):
        print(finding.render())
    if findings:
        print(
            f"\n{len(findings)} user-visible message(s) name the operator's private deployment "
            "repository. Rewrite each to name what the reader can act on (D106).",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
