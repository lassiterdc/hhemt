"""Regression tests for hooks/eda_kind_table.py — the build-time EDA kind table.

Mirrors tests/test_check_autodoc_coverage.py: loads the module by path (mkdocs execs
each hook as a standalone module, so `hooks/` is not a package) and runs pure, with no
fixture and no analysis tree.

WHAT THESE ASSERT, AND WHY IT IS NOT THE TABLE'S WORDING. Every assertion below is
DERIVED from the two registries at assert time rather than pinned to today's nine
kinds. A test that hardcoded the nine would pass today and would fail the moment a
design pass fills a figure in or a new kind is registered, which is the exact event
the generated table exists to survive. So the invariant under test is the RELATION
between the registries and the rendered rows, not the rows themselves.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_HOOK = Path(__file__).resolve().parent.parent / "hooks" / "eda_kind_table.py"


def _load_hook():
    spec = importlib.util.spec_from_file_location("eda_kind_table", _HOOK)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


hook = _load_hook()


def _page(src_uri: str):
    """The one attribute the hook reads off a mkdocs Page."""
    return SimpleNamespace(file=SimpleNamespace(src_uri=src_uri))


def _rendered_rows(table: str) -> list[str]:
    """The data rows of the rendered table, header and separator dropped."""
    return [ln for ln in table.splitlines() if ln.startswith("|")][2:]


def _kind_of(row: str) -> str:
    return row.split("|")[1].strip().strip("`")


def _mark_of(row: str) -> str:
    return row.split("|")[3].strip()


# --- the substitution behaviour -------------------------------------------------


def test_marker_is_substituted_on_the_target_page():
    page_md = f"# Run the exploratory analysis\n\nbefore\n\n{hook.MARKER}\n\nafter\n"
    out = hook.on_page_markdown(page_md, _page(hook.TARGET_URI), None, None)
    assert hook.MARKER not in out
    assert "| Renderer kind | Backing artifact | Figure |" in out
    # The surrounding prose is untouched: the hook substitutes, it does not rewrite.
    assert "before" in out and "after" in out


def test_a_non_target_page_is_returned_unchanged():
    """The differently-positioned satisfying state.

    This page ALSO carries no marker. An assertion written as "a page without the
    marker is a defect" would fire here and be wrong, which is the over-firing this
    test exists to pin down.
    """
    page_md = "# Installation\n\nno marker anywhere\n"
    assert hook.on_page_markdown(page_md, _page("how-to/installation.md"), None, None) is None


def test_target_page_without_the_marker_raises():
    """The violating state: the marker was renamed or hand-deleted off the target."""
    with pytest.raises(RuntimeError) as excinfo:
        hook.on_page_markdown("# Run the exploratory analysis\n", _page(hook.TARGET_URI), None, None)
    # Assert on a substring that is true in every future wording of this message:
    # it must name the file that lost the marker.
    assert hook.TARGET_URI in str(excinfo.value)


# --- the derivation contract ----------------------------------------------------


def test_rows_are_exactly_the_registered_kinds_in_sorted_order():
    from hhemt.eda._plotting import _EDA_RENDERERS

    rows = _rendered_rows(hook.render_table())
    assert [_kind_of(r) for r in rows] == sorted(_EDA_RENDERERS)


def test_the_mark_tracks_the_pending_set():
    from hhemt.eda._sensitivity_figures import _PENDING_EDA_FIGURE_STEMS

    for row in _rendered_rows(hook.render_table()):
        expected = "Not yet designed" if _kind_of(row) in _PENDING_EDA_FIGURE_STEMS else "Designed"
        assert _mark_of(row) == expected, _kind_of(row)


def test_the_backing_column_honours_the_override_map():
    """The one kind whose backing artifact is not its own name.

    Derived from `_RENDERER_BACKING_ARTIFACT` rather than naming `config_diff_maps`,
    so the test keeps meaning if the override set grows or empties.
    """
    from hhemt.eda._plotting import _RENDERER_BACKING_ARTIFACT

    backing = {r["kind"]: r["backing"] for r in hook.kind_records()}
    for kind, stem in _RENDERER_BACKING_ARTIFACT.items():
        assert backing[kind] == stem
    for kind, stem in backing.items():
        if kind not in _RENDERER_BACKING_ARTIFACT:
            assert stem == kind


def test_no_rendered_cell_can_trip_the_docs_content_gate():
    """The generated table bypasses `check_docs_content.py`, which reads what is on
    disk and therefore sees the marker rather than the table. This pins the property
    that made a build-time lint pass unnecessary: every emitted cell is an identifier
    or one of two fixed words."""
    table = hook.render_table()
    assert "—" not in table
    for banned in ("scaffold", "estate"):
        assert banned not in table.lower()
