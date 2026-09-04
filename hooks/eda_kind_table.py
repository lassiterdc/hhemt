"""Substitute the EDA renderer-kind table into `how-to/running-eda.md` at build time.

WHY A BUILD-TIME SUBSTITUTION RATHER THAN A COMMITTED TABLE. Which renderer kinds
exist, and which of them a design pass has filled in, are two facts the code already
carries: `eda/_plotting.py::_EDA_RENDERERS` is the dispatch registry and
`eda/_sensitivity_figures.py::_PENDING_EDA_FIGURE_STEMS` is the not-yet-designed set.
A hand-maintained table restates both and falls behind whichever one moves first.
There is no artifact between builds, so the table cannot drift.

WHY A MARKER RATHER THAN A WHOLE GENERATED PAGE. `hooks/config_reference.py`
generates an entire page because every row of it is derived. Here the derived region
is one block inside a page that is otherwise authored prose, so the page stays
hand-written and this hook fills one marker.

WHY NOTHING IS WRAPPED IN try/except. The reason `hooks/config_reference.py` gives
holds unchanged: mkdocs fails loudly by design, and a hook that swallowed its own
exception would ship a page carrying the raw marker or no table at all, silently.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

#: The page carrying the marker, and the marker itself. A target page that has lost
#: its marker is NOT silently passed over: the substitution asserts the marker is
#: present, so a rename or a hand-edit that drops the comment aborts the build
#: instead of publishing a page with no table.
TARGET_URI = "how-to/running-eda.md"
MARKER = "<!-- hhemt:eda-kind-table -->"

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _bind_local_src() -> None:
    """Bind `import hhemt` to THIS checkout, reusing the config-reference binder.

    Loaded by path rather than imported: mkdocs execs each hook as a standalone
    module, so `hooks/` is not a package and `from config_reference import ...` does
    not resolve. This is the same `spec_from_file_location` mechanism
    `hooks/config_reference.py` already uses to load `scripts/check_docs_content.py`.
    Reusing it rather than restating it matters because the binder does two things a
    copy would get wrong: it prepends this checkout's `src` AND drops already-imported
    `hhemt` modules that an editable install's `.pth` resolved to a different tree.
    Whether that second half is load-bearing depends on the interpreter: measured on
    this machine, two environments carry mkdocs and their `.pth` files disagree, one
    pointing at the worktree and one at the main clone. Nothing in the repo pins which
    one runs the build, so the binder is what makes the answer not matter.
    """
    path = _REPO_ROOT / "hooks" / "config_reference.py"
    spec = importlib.util.spec_from_file_location("_hhemt_config_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._bind_local_src()


def kind_records() -> list[dict]:
    """One record per registered EDA renderer kind, sorted by kind.

    `backing` is the `eda/{stem}.zarr` the kind's calc member actually writes, which
    is the kind itself except where `_RENDERER_BACKING_ARTIFACT` declares otherwise.
    `designed` is False for exactly the stems `_PENDING_EDA_FIGURE_STEMS` names.
    """
    from hhemt.eda._plotting import _EDA_RENDERERS, _RENDERER_BACKING_ARTIFACT
    from hhemt.eda._sensitivity_figures import _PENDING_EDA_FIGURE_STEMS

    return [
        {
            "kind": kind,
            "backing": _RENDERER_BACKING_ARTIFACT.get(kind, kind),
            "designed": kind not in _PENDING_EDA_FIGURE_STEMS,
        }
        for kind in sorted(_EDA_RENDERERS)
    ]


def render_table() -> str:
    """The markdown table, derived from the two registries and from nothing else.

    Every cell is either an identifier read out of the code or one of two fixed
    words, so the docs-content vocabulary and punctuation rules cannot be violated
    by a value this function emits. That is why there is no lint pass here, where
    `hooks/config_reference.py` needs one over its own authored prose.
    """
    lines = [
        "| Renderer kind | Backing artifact | Figure |",
        "|---|---|---|",
    ]
    for record in kind_records():
        mark = "Designed" if record["designed"] else "Not yet designed"
        lines.append(f"| `{record['kind']}` | `eda/{record['backing']}.zarr` | {mark} |")
    return "\n".join(lines)


def on_page_markdown(markdown, page, config, files):
    """Replace the marker on the running-EDA page with the derived table."""
    if page.file.src_uri != TARGET_URI:
        return None
    if MARKER not in markdown:
        raise RuntimeError(
            f"{TARGET_URI} no longer carries {MARKER}, so the EDA kind table has "
            f"nowhere to go. Restore the marker, or retire hooks/eda_kind_table.py "
            f"and hand-maintain the table."
        )
    _bind_local_src()
    return markdown.replace(MARKER, render_table())
