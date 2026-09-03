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
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import sys
from html.parser import HTMLParser
from pathlib import Path

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


def expected_qualnames() -> set[str]:
    """{module}.{symbol} for every non-underscore class/function in each __all__."""
    out: set[str] = set()
    for modname in public_modules():
        mod = importlib.import_module(modname)
        for sym in getattr(mod, "__all__", ()):
            if sym.startswith("_"):
                continue  # mirrors mkdocs.yml filters: ["!^_"]
            obj = getattr(mod, sym, None)
            if inspect.isclass(obj) or inspect.isroutine(obj):
                out.add(f"{modname}.{sym}")
            # bare constants/data: no heading anchor under default config -> skip
    return out


def undocumented_symbols() -> list[str]:
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
    for modname in public_modules():
        mod = importlib.import_module(modname)
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
        print(f"ERROR: could not import public modules: {exc}", file=sys.stderr)
        return 2
    anchors = rendered_anchors(args.site_dir)
    missing = sorted(q for q in expected if q not in anchors)
    if missing:
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

    print(
        f"autodoc coverage OK — all {len(expected)} public class/function symbols "
        f"rendered, and every public symbol carries a docstring."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
