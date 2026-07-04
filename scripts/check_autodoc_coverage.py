#!/usr/bin/env python
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

Exit 0 = every expected class/function has an anchor. 1 = >=1 missing (enumerated).
2 = usage/environment error (site dir absent, import failure). Pure stdlib.
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import sys
from html.parser import HTMLParser
from pathlib import Path

# Public modules whose __all__ defines the documented surface. Each MUST have a
# corresponding ``::: {module}`` directive in docs/reference/api.md. (hhemt.toolkit
# is intentionally omitted: its sole export Toolkit is re-exported at the top level
# and rendered by ``::: hhemt`` as ``hhemt.Toolkit``.)
PUBLIC_MODULES: tuple[str, ...] = (
    "hhemt",
    "hhemt.analysis",
    "hhemt.sensitivity_analysis",
    "hhemt.bundle",
    "hhemt.eda",
    "hhemt.version_migration",
)

def expected_qualnames() -> set[str]:
    """{module}.{symbol} for every non-underscore class/function in each __all__."""
    out: set[str] = set()
    for modname in PUBLIC_MODULES:
        mod = importlib.import_module(modname)
        for sym in getattr(mod, "__all__", ()):
            if sym.startswith("_"):
                continue  # mirrors mkdocs.yml filters: ["!^_"]
            obj = getattr(mod, sym, None)
            if inspect.isclass(obj) or inspect.isroutine(obj):
                out.add(f"{modname}.{sym}")
            # bare constants/data: no heading anchor under default config -> skip
    return out

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
        "--site-dir", type=Path, default=Path("site"),
        help="built mkdocs site directory (default: ./site)",
    )
    args = ap.parse_args(argv)
    if not args.site_dir.is_dir():
        print(f"ERROR: site dir not found: {args.site_dir} (run `mkdocs build` first)",
              file=sys.stderr)
        return 2
    try:
        expected = expected_qualnames()
    except Exception as exc:  # import failure is an env error, not a coverage miss
        print(f"ERROR: could not import public modules: {exc}", file=sys.stderr)
        return 2
    anchors = rendered_anchors(args.site_dir)
    missing = sorted(q for q in expected if q not in anchors)
    if missing:
        print("autodoc coverage FAILED — public symbols with no rendered doc anchor:",
              file=sys.stderr)
        for qual in missing:
            print(f"  - {qual}", file=sys.stderr)
        print(
            f"\n{len(missing)}/{len(expected)} public symbols unrendered. Add a "
            f"`::: {{module}}` directive to docs/reference/api.md or check mkdocstrings "
            f"filters.",
            file=sys.stderr,
        )
        return 1
    print(f"autodoc coverage OK — all {len(expected)} public class/function symbols rendered.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
