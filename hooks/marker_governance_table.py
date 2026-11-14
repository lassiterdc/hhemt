"""Publish the documentation gate's exemption vocabulary on the contributor page.

WHY A SUBSTITUTION RATHER THAN PROSE. `check_docs_content._declared_exemptions`
detects a marker by testing whether its NAME occurs in the page's unfenced text,
so a contributor page that wrote the names in its own prose would abort the gate
it documents. This hook keeps the names out of the page's bytes and puts them on
the rendered page instead: `on_page_markdown` substitutes in memory and never
writes to disk, and the gate reads what is on disk.

WHY THE `Exempts` COLUMN READS A CONSTANT AND NOT THE PAGES. The column is headed
per MARKER, so it must be a property of the marker. Unioning `_exempt_groups`
over the pages carrying it computes a per-PAGE aggregate: it folds in every OTHER
marker on those pages and folds again across pages, and it agrees with the
per-marker answer only while every live declaration happens to name the same
group. The module states each marker's declaration as a constant, so the cell
reads that constant through the module's own parser. Pages determine the
`Live pages` count and nothing else.

WHY THE CELLS ARE ASSERTED RATHER THAN CLAIMED. `hooks/eda_kind_table.py` states
that its cells are identifiers and that this is why it needs no lint pass. The
claim is the licence, so it is checked here rather than asserted in prose: every
DATA cell must match `_CELL_OK`, and the header row is a literal in this file.
The assertion is what licenses the absence of a lint pass; if a later edit emits
a free-text cell the build fails here instead of publishing it.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

#: The page carrying the marker, and the marker itself. The name is deliberately
#: NOT one of the exemption markers and carries no `exempt=`, so it is inert to
#: the gate: `_exempt_groups` refuses `exempt=` on a non-exemption name, and this
#: marker declares none.
TARGET_URI = "contributing.md"
MARKER = "<!-- hhemt:marker-governance -->"

_REPO_ROOT = Path(__file__).resolve().parent.parent

#: A data cell is a backticked identifier-or-marker-name, a backticked
#: comma-joined group list, or a bare integer. Nothing else may be emitted.
_CELL_OK = re.compile(r"^(?:`[A-Za-z_][A-Za-z_0-9:,-]*`|\d+)$")

_HEADER = (
    "| Marker | Exempts | Live pages |",
    "|---|---|---|",
)


def _load_gate():
    """The gate module, loaded by path the way `hooks/config_reference.py` does."""
    path = _REPO_ROOT / "scripts" / "check_docs_content.py"
    spec = importlib.util.spec_from_file_location("_hhemt_docs_gate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def marker_records() -> list[dict[str, object]]:
    """One record per exemption marker.

    `groups` is parsed from the marker's own declaration CONSTANT; `count` is the
    only value the live pages decide.

    THE GUARD BELOW CATCHES CONSTANT DRIFT, and that is the only thing that
    reaches it. `_declared_exemptions` raises on every malformed declaration --
    an empty `exempt=`, an unknown group, a name with no `exempt=` at all -- so
    none of those arrives here. Its one falsy return is `None`, and `None` means
    the NAME was absent from the text it was given. The text here is the
    marker's own declaration constant, so the reachable condition is exactly
    that the module's NAME constant and its DECLARATION constant have drifted
    apart. Nothing else in the module checks that the second contains the first.
    """
    gate = _load_gate()
    docs = _REPO_ROOT / "docs"
    markers = (
        (gate.GENERATED_MARKER_NAME, gate.GENERATED_MARKER, gate.generated_files),
        (gate.PERSONAL_VOICE_MARKER_NAME, gate.PERSONAL_VOICE_MARKER, gate.personal_voice_files),
        (gate.REPO_INTERNAL_MARKER_NAME, gate.REPO_INTERNAL_MARKER, gate.repo_internal_files),
    )
    out: list[dict[str, object]] = []
    for name, constant, finder in markers:
        declared = gate._declared_exemptions(constant, name)
        if not declared:
            raise RuntimeError(
                f"{name} does not appear in its own declaration constant {constant!r}, "
                f"so the module's NAME constant and its DECLARATION constant have "
                f"drifted apart. Every malformed declaration raises earlier; this is "
                f"the only condition that reaches here."
            )
        out.append({"marker": name, "groups": sorted(declared), "count": len(finder(docs))})
    return out


def render_table() -> str:
    """The markdown table, derived from the gate and from nothing else."""
    lines = list(_HEADER)
    for record in marker_records():
        lines.append(f"| `{record['marker']}` | `{','.join(record['groups'])}` | {record['count']} |")
    for line in lines[len(_HEADER) :]:
        for cell in (c.strip() for c in line.strip("|").split("|")):
            if not _CELL_OK.match(cell):
                raise RuntimeError(
                    f"marker_governance_table emitted a cell that is not an identifier, "
                    f"group list or count: {cell!r}. The absence of a lint pass in this "
                    f"hook is licensed by that property, so restore it or add the pass."
                )
    return "\n".join(lines)


def on_page_markdown(markdown, page, config, files):
    """Replace the marker on the contributor page with the derived table."""
    if page.file.src_uri != TARGET_URI:
        return None
    if MARKER not in markdown:
        raise RuntimeError(
            f"{TARGET_URI} no longer carries {MARKER}, so the exemption table has "
            f"nowhere to go. Restore the marker, or retire hooks/marker_governance_table.py "
            f"and hand-maintain the table."
        )
    return markdown.replace(MARKER, render_table())
