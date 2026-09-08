#!/usr/bin/env python3
"""CI check: no rendered public docstring is written in a dialect the configured
mkdocstrings handler does not parse.

WHY THIS IS NOT COVERED BY `check_autodoc_coverage.py`. That gate asserts two
facts -- the symbol renders a doc-object anchor, and its docstring is non-empty.
A Google-style ``Args:`` block under ``docstring_style: numpy`` satisfies both
and still ships broken: griffe does not recognise the header as a section, so
the parameter documentation renders as literal body prose. Measured 2026-09-07
on the built site, the facade's ``run`` renders nine parameters as one run-on
paragraph, and an ``Example:`` block's doctest lines render as running prose --
a runnable snippet destroyed as a code block, which is strictly worse than an
unstructured parameter list.

WHY THE FLAG CONDITION IS NOT "renders other than as authored". Because it does
render as authored: every authored character reaches the page. What is lost is
the STRUCTURE the convention was meant to produce, and a condition phrased
against the author's intent cannot be applied to a failure that preserves the
author's bytes. The referent has to be the CONFIGURED CONSUMER, which is what
this checks: an authored section header that the configured handler does not
parse as a section.

THE SECOND CONJUNCT IS LOAD-BEARING AND LOOKS REDUNDANT. A docstring is flagged
only when it carries a foreign header AND parses zero native sections. Without
that, a correct numpy docstring whose Examples block happens to contain the
literal text ``Args:`` false-fires. A checker that cries wolf on correct
docstrings is routed around, which is worse than no checker.

GROUND TRUTH IS DERIVED, NOT DECLARED. The module set comes from
``check_autodoc_coverage.public_modules()``, which parses ``docs/reference/api.md``
-- so this script and that one cannot diverge. That script's own history is the
reason: a hand-maintained tuple there mirrored the rendering page and silently
diverged at 6 entries against 8.

Exit 0 = clean. 1 = >=1 flagged symbol. 2 = usage/environment error. Pure stdlib;
imports no project code and needs no built site.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_autodoc_coverage import public_modules  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent

#: Every section header of each dialect. Enumerating ALL of them is the point:
#: a check written from the ``Args:``/``Returns:`` motivating example passes the
#: highest-severity live instance, which is an ``Example:`` block.
GOOGLE_HEADERS: tuple[str, ...] = (
    "Args",
    "Arguments",
    "Attributes",
    "Example",
    "Examples",
    "Keyword Args",
    "Keyword Arguments",
    "Note",
    "Notes",
    "Other Parameters",
    "Raises",
    "Returns",
    "Todo",
    "Warns",
    "Warnings",
    "Yields",
)
NUMPY_HEADERS: tuple[str, ...] = (
    "Attributes",
    "Examples",
    "Notes",
    "Other Parameters",
    "Parameters",
    "Raises",
    "Receives",
    "References",
    "Returns",
    "See Also",
    "Warnings",
    "Warns",
    "Yields",
)

GOOGLE_SECTION = re.compile(r"(?m)^[ \t]*(" + "|".join(GOOGLE_HEADERS) + r")[ \t]*:[ \t]*$")
NUMPY_SECTION = re.compile(r"(?m)^[ \t]*(" + "|".join(NUMPY_HEADERS) + r")[ \t]*\n[ \t]*-{3,}[ \t]*$")
STYLE_KEY = re.compile(r"^\s*docstring_style:\s*([A-Za-z]+)\s*$", re.M)


def configured_style(mkdocs_yml: Path) -> str:
    """The handler's ``docstring_style``, read from mkdocs.yml by regex.

    Regex rather than a YAML parse because `mkdocs.yml` carries a
    ``!!python/name:`` tag that `yaml.safe_load` refuses, and adding a
    dependency to read one scalar is the wrong trade for a gate that is
    otherwise pure stdlib.
    """
    m = STYLE_KEY.search(mkdocs_yml.read_text(encoding="utf-8"))
    if m is None:
        raise ValueError(f"{mkdocs_yml} declares no docstring_style — nothing to check against.")
    return m.group(1).strip().lower()


def _module_path(module: str, src: Path) -> Path | None:
    direct = src / (module.replace(".", "/") + ".py")
    if direct.is_file():
        return direct
    pkg = src / module.replace(".", "/") / "__init__.py"
    return pkg if pkg.is_file() else None


def _declared_all(tree: ast.Module) -> list[str] | None:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "__all__" for t in node.targets):
            try:
                return list(ast.literal_eval(node.value))
            except (ValueError, SyntaxError):
                return None
    return None


def _import_sources(module: str, tree: ast.Module) -> dict[str, tuple[str, str]]:
    """{local name: (defining module, original name)} for every ``from X import Y``.

    Needed because a package re-exports: ``hhemt.Toolkit`` reaches the API page
    through ``::: hhemt`` and its ``__all__``, while the class itself is defined
    in ``hhemt.toolkit``. Resolving only module-local definitions silently skips
    every re-exported symbol -- which on this corpus is the facade class and all
    nine public exceptions, i.e. exactly the highest-traffic half of the surface.
    """
    out: dict[str, tuple[str, str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level:
            parts = module.split(".")
            base = ".".join(parts[: len(parts) - node.level + 1] + ([node.module] if node.module else []))
        else:
            base = node.module or ""
        for alias in node.names:
            out[alias.asname or alias.name] = (base, alias.name)
    return out


def flagged(src: Path, api_page: Path, style: str) -> list[tuple[str, str]]:
    """(qualname, foreign-header) for every rendered symbol in the wrong dialect.

    Raises ValueError if no module named on the API page resolves to a file under
    ``src`` -- an environment error rather than a coverage miss, and an empty
    examined set would otherwise make this gate pass vacuously. This mirrors
    ``check_autodoc_coverage.public_modules``, which raises for the same reason on
    the other axis: that one guards an empty DIRECTIVE list, this one an empty
    RESOLVED list, and a gate needs both because either alone leaves a green over
    a population it never found.
    """
    foreign, native = (GOOGLE_SECTION, NUMPY_SECTION) if style == "numpy" else (NUMPY_SECTION, GOOGLE_SECTION)
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    examined = 0
    for module in public_modules(api_page):
        path = _module_path(module, src)
        if path is None:
            continue
        examined += 1
        tree = ast.parse(path.read_text(encoding="utf-8"))
        defined = {n.name: n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))}
        names = _declared_all(tree) or [n for n in defined if not n.startswith("_")]
        imported = _import_sources(module, tree)
        for name in names:
            node = defined.get(name)
            owner = module
            if node is None and name in imported:
                origin_module, origin_name = imported[name]
                origin_path = _module_path(origin_module, src)
                if origin_path is None:
                    continue
                origin_tree = ast.parse(origin_path.read_text(encoding="utf-8"))
                node = {
                    n.name: n
                    for n in origin_tree.body
                    if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                }.get(origin_name)
                owner = origin_module
                name = origin_name
            if node is None:
                continue
            nodes = [(f"{owner}.{name}", node)]
            if isinstance(node, ast.ClassDef):
                nodes += [
                    (f"{owner}.{name}.{c.name}", c)
                    for c in node.body
                    if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef)) and not c.name.startswith("_")
                ]
            for qualname, member in nodes:
                if qualname in seen:
                    continue
                seen.add(qualname)
                doc = ast.get_docstring(member)
                if doc is None:
                    continue
                hit = foreign.search(doc)
                if hit and not native.search(doc):
                    out.append((qualname, hit.group(1)))
    if not examined:
        raise ValueError(f"no module named on {api_page} resolved to a file under {src} -- nothing to check.")
    return sorted(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=_REPO_ROOT / "src")
    ap.add_argument("--api-page", type=Path, default=_REPO_ROOT / "docs" / "reference" / "api.md")
    ap.add_argument("--mkdocs-yml", type=Path, default=_REPO_ROOT / "mkdocs.yml")
    args = ap.parse_args(argv)

    for label, path in (("src", args.src), ("api page", args.api_page), ("mkdocs.yml", args.mkdocs_yml)):
        if not path.exists():
            print(f"ERROR: {label} not found: {path}", file=sys.stderr)
            return 2

    try:
        style = configured_style(args.mkdocs_yml)
        hits = flagged(args.src, args.api_page, style)
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if hits:
        print(f"docstring dialect check FAILED (configured style: {style}):", file=sys.stderr)
        for qualname, header in hits:
            print(f"  {qualname} — carries `{header}:` and parses zero {style} sections", file=sys.stderr)
        print(
            f"\n{len(hits)} symbol(s). Each renders its section as literal body prose rather than "
            f"a parsed section; an Example block renders its doctest lines as running text, "
            f"destroying a runnable snippet. Convert to {style} sections.",
            file=sys.stderr,
        )
        return 1

    print(f"docstring dialect OK — every rendered public docstring parses as {style}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
