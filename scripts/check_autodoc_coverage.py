#!/usr/bin/env python3
"""CI check: every public class/function in a public module's __all__ renders as a
mkdocstrings doc-object anchor in the built site (ADR-7 docs-accuracy proxy).

Pairs with docs/reference/api.md (the per-submodule ``:::`` directives) and the
`docs-build` GitHub Actions job, which builds the site this script inspects.

Ground truth = the union of every public module's ``__all__`` (leading-underscore
entries excluded to mirror mkdocs.yml ``filters: ["!^_"]``). The check keys on
rendered doc-object ANCHORS (``id="{module}.{symbol}"``), NOT a substring grep:
a substring match false-positives on type-annotation cross-references and source
snippets (e.g. ``TRITONSWMM_analysis`` appears inside ``Toolkit``'s rendered
signature without being a documented object). Only classes and functions are
required to render an anchor; bare module-level constants (e.g. LAYOUT_VERSION)
emit no heading anchor under the numpy-style default config and are excluded.

Two independent assertions, because an anchor and a docstring are different facts:
  1. every expected class/function renders a doc-object anchor (the symbol reached
     the page), and
  2. every public symbol carries a non-empty docstring (something is under the
     heading). Without (2) a symbol renders as a bare name and this gate is green.

Exit 0 = both hold. 1 = >=1 failure of either (enumerated separately).
2 = usage/environment error (site dir absent, import failure, api.md absent or
declaring no directives). Pure stdlib.

The population is bound to THIS checkout's ``src`` through ``scripts/local_src.py``
(prepend, purge, then a provenance assert on every imported module), and one
``population root:`` line printed immediately before the verdict names the tree the
symbols came from, the API page, and the site. A bare ``python`` run of this script
from a worktree therefore grades the worktree, not whichever checkout the shared
editable install last pointed at.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import sys
from html.parser import HTMLParser
from pathlib import Path
from types import ModuleType

sys.path.insert(0, str(Path(__file__).resolve().parent))
from local_src import assert_resolved_under, bind_local_src  # noqa: E402

# The documented surface is DERIVED from docs/reference/api.md's ``:::`` directives
# rather than declared here. That is deliberate and it removes a defect class rather
# than an instance of one: a hand-maintained tuple here mirrored a hand-maintained
# rendering surface there, and the two silently diverged — measured 2026-08-26 at 6
# entries here against 8 directives there, so `hhemt.experiment_bundle` and
# `hhemt.synthetic_experiment` rendered on the public API page while this gate
# certified a surface that excluded them.
#
# (`hhemt.toolkit` needs no directive: its sole export `Toolkit` is re-exported at
# the top level and rendered by ``::: hhemt`` as ``hhemt.Toolkit``.)
API_REFERENCE_PAGE = Path(__file__).resolve().parent.parent / "docs" / "reference" / "api.md"
SRC_ROOT = Path(__file__).resolve().parent.parent / "src"


def public_modules(api_page: Path = API_REFERENCE_PAGE) -> tuple[str, ...]:
    """Module names from the ``::: {module}`` directives on the API reference page.

    Raises FileNotFoundError if the page is absent, and ValueError if it declares
    no directives — both are environment errors rather than coverage misses, and
    an empty derived set would otherwise make this gate pass vacuously.
    """
    text = api_page.read_text(encoding="utf-8")
    mods = tuple(
        line.split(":::", 1)[1].strip()
        for line in text.splitlines()
        if line.startswith(":::") and line.split(":::", 1)[1].strip()
    )
    if not mods:
        raise ValueError(f"{api_page} declares no ``:::`` directives — nothing to check.")
    return mods


def _declared_public(mod) -> list[str]:
    """``__all__`` where the module declares one, else its own top-level defs.

    The fallback reads SOURCE, not ``dir(mod)``. A runtime sweep of a module that
    declares no ``__all__`` returns everything it IMPORTED as well, and the
    class/routine filter below cannot strain those out because ``Path`` IS a class
    and ``dataclass`` IS a routine -- measured at 72 symbols with 5 unrendered, two
    of them stdlib names, against 67 with 0 unrendered here. Only the two modules
    declaring no ``__all__`` reach the fallback at all; the other seven are read
    exactly as before, from the object.

    A module with neither ``__all__`` nor a readable ``__file__`` yields nothing.
    That is the old behaviour, it can only UNDER-report, and it is what keeps a
    synthetic module built by a test from crashing the gate.

    THIS IS THE SAME RULE `check_docstring_dialect.py` APPLIES, AND THE TWO ARE NOT
    COPIES. That gate has no class/routine filter and needs none -- a constant has
    no docstring to check -- so this rule yields 67 where the bare shared rule
    yields 71, the four extra being ``__all__``-listed constants that render no
    anchor. It also reads ``__all__`` from the AST while this reads it from the
    imported object, which differs for a dynamically built ``__all__``. So the two
    files hold two rules that AGREE on this corpus and differ in what they tolerate,
    and no diff can detect them drifting apart. The price is accepted deliberately:
    importing the sibling's helper inverts the import stack and breaks this gate,
    measured in both placements.
    """
    declared = getattr(mod, "__all__", None)
    if declared is not None:
        return [n for n in declared if not n.startswith("_")]
    source = getattr(mod, "__file__", None)
    if source is None:
        return []
    tree = ast.parse(Path(source).read_text(encoding="utf-8"))
    return [
        node.name
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_")
    ]


def _import_public_module(modname: str, src_root: Path) -> ModuleType:
    """The one import path both enumerators share: bind, import, then assert provenance.

    ``bind_local_src`` puts ``src_root`` first on ``sys.path`` and evicts any warm
    ``sys.modules`` entry of the module's package that resolved elsewhere, so the
    import below reaches THIS checkout whether the interpreter is cold or warm.
    ``assert_resolved_under`` then reports the one case the bind cannot repair, a
    ``src_root`` that supplies no such package, by raising into ``main()``'s exit-2
    branch. A module that declares no ``__file__`` passes both: the injected
    fixtures in this gate's tests are such modules and pin nothing about resolution.
    """
    bind_local_src(src_root, (modname,))
    mod = importlib.import_module(modname)
    assert_resolved_under(mod, src_root)
    return mod


def expected_qualnames(api_page: Path = API_REFERENCE_PAGE, src_root: Path = SRC_ROOT) -> set[str]:
    """{module}.{symbol} for every public class/function, by the ``__all__``-or-defs rule."""
    out: set[str] = set()
    for modname in public_modules(api_page):
        mod = _import_public_module(modname, src_root)
        for sym in _declared_public(mod):
            if sym.startswith("_"):
                continue  # mirrors mkdocs.yml filters: ["!^_"]
            obj = getattr(mod, sym, None)
            if inspect.isclass(obj) or inspect.isroutine(obj):
                out.add(f"{modname}.{sym}")
            # bare constants/data: no heading anchor under default config -> skip
    return out


def undocumented_symbols(api_page: Path = API_REFERENCE_PAGE, src_root: Path = SRC_ROOT) -> list[str]:
    """Public symbols that render an anchor but carry NO docstring.

    An anchor proves the symbol reached the page; it says nothing about whether
    anything is under the heading. A symbol with an empty docstring renders as a
    bare name — measured 2026-08-26 at 25 such symbols, including
    ``TRITONSWMM_analysis``, the toolkit's central class, while this gate was
    green. Checking anchor presence alone certifies the delivery mechanism
    rather than the content.

    Covers each ``__all__`` entry and, for classes, their public methods,
    properties, classmethods and staticmethods. ``classmethod``/``staticmethod``
    objects retrieved via ``vars()`` are NOT matched by ``inspect.isfunction``,
    so they are tested explicitly — omitting them silently undercounts.
    """
    missing: list[str] = []
    for modname in public_modules(api_page):
        mod = _import_public_module(modname, src_root)
        for sym in getattr(mod, "__all__", ()):
            if sym.startswith("_"):
                continue
            obj = getattr(mod, sym, None)
            if obj is None:
                continue
            if (inspect.isclass(obj) or inspect.isroutine(obj)) and not (inspect.getdoc(obj) or "").strip():
                missing.append(f"{modname}.{sym}")
            if inspect.isclass(obj):
                for attr, val in vars(obj).items():
                    if attr.startswith("_"):
                        continue
                    if isinstance(val, property):
                        doc = inspect.getdoc(val.fget) or ""
                    elif isinstance(val, classmethod | staticmethod):
                        doc = inspect.getdoc(val.__func__) or ""
                    elif inspect.isfunction(val):
                        doc = inspect.getdoc(val) or ""
                    else:
                        continue
                    if not doc.strip():
                        missing.append(f"{modname}.{sym}.{attr}")
    return missing


class _AnchorCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for key, val in attrs:
            if key == "id" and val:
                self.ids.add(val)


def rendered_anchors(site_dir: Path) -> set[str]:
    ids: set[str] = set()
    for html in site_dir.rglob("*.html"):
        parser = _AnchorCollector()
        parser.feed(html.read_text(encoding="utf-8", errors="ignore"))
        ids |= parser.ids
    return ids


def _provenance_line(site_dir: Path, *, import_failed: bool = False) -> str:
    """One line naming what the verdict below it was computed over.

    Printed on the verdict's own stream immediately before it, at every exit that
    reached the import, so a reader of either stream sees the roots beside the
    sentence they qualify. The resolved package directory is the measurement and
    ``SRC_ROOT`` is the intention; a reader can see the two agree.
    """
    package = sys.modules.get("hhemt")
    origin = None if import_failed or package is None else getattr(package, "__file__", None)
    resolved = f"hhemt resolved at {Path(origin).resolve().parent}" if origin else "hhemt import FAILED"
    return f"population root: {SRC_ROOT} ({resolved}); api page: {API_REFERENCE_PAGE}; site: {site_dir.resolve()}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--site-dir",
        type=Path,
        default=Path("site"),
        help="built mkdocs site directory (default: ./site)",
    )
    args = ap.parse_args(argv)
    if not args.site_dir.is_dir():
        print(f"ERROR: site dir not found: {args.site_dir} (run `mkdocs build` first)", file=sys.stderr)
        return 2
    try:
        expected = expected_qualnames()
    except Exception as exc:  # import failure is an env error, not a coverage miss
        print(_provenance_line(args.site_dir, import_failed=True), file=sys.stderr)
        print(f"ERROR: could not import public modules: {exc}", file=sys.stderr)
        return 2
    anchors = rendered_anchors(args.site_dir)
    missing = sorted(q for q in expected if q not in anchors)
    if missing:
        print(_provenance_line(args.site_dir), file=sys.stderr)
        print("autodoc coverage FAILED — public symbols with no rendered doc anchor:", file=sys.stderr)
        for qual in missing:
            print(f"  - {qual}", file=sys.stderr)
        print(
            f"\n{len(missing)}/{len(expected)} public symbols unrendered. Add a "
            f"`::: {{module}}` directive to docs/reference/api.md or check mkdocstrings "
            f"filters.",
            file=sys.stderr,
        )
        return 1

    undocumented = undocumented_symbols()
    if undocumented:
        print(_provenance_line(args.site_dir), file=sys.stderr)
        print("autodoc coverage FAILED — public symbols that render with NO docstring:", file=sys.stderr)
        for qual in undocumented:
            print(f"  - {qual}", file=sys.stderr)
        print(
            f"\n{len(undocumented)} public symbol(s) render as a bare name on the API "
            f"reference. An anchor proves the symbol reached the page; it does not "
            f"prove anything is under the heading. Add a docstring, or make the symbol "
            f"private if it is not part of the supported API.",
            file=sys.stderr,
        )
        return 1

    print(_provenance_line(args.site_dir))
    print(
        f"autodoc coverage OK — all {len(expected)} class/function symbols named by "
        f"`__all__`, or defined at module level where a module declares none, render "
        f"a doc anchor; each of those {len(expected)} carries a docstring. Neither "
        f"clause reaches a symbol outside that rule, and neither detects a module "
        f"dropped from docs/reference/api.md — see `public_modules`."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
