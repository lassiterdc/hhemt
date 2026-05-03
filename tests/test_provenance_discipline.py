"""AST-level enforcement of the per-artist provenance discipline.

Walks every renderer module under `src/TRITON_SWMM_toolkit/report_renderers/`
(excluding files whose stem starts with `_`) and asserts that every artist-
creating matplotlib call inside the module is enclosed in a
`with <name>.artist(...)` context block. Additionally asserts that each
non-skipped renderer module contains at least one such block — guard against
alias-rebinds (e.g., `plot = ax.plot; plot(...)`) that would trivially satisfy
a per-call check.

The test surfaces a clear file:line error message on violation.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

# Curated list of matplotlib Axes methods that produce data-driven artists.
# Excluded: text-annotation methods (annotate, text, set_title, ...), colorbar,
# layout setters. The list is the enforcement surface; adding a method here
# expands the lint without code change.
_ARTIST_METHODS: frozenset[str] = frozenset({
    "plot",
    "scatter",
    "imshow",
    "add_collection",
    "add_patch",
    "fill_between",
    "fill_betweenx",
    "contour",
    "contourf",
    "quiver",
    "streamplot",
    "bar",
    "barh",
    "step",
    "stem",
    "hexbin",
    "pcolor",
    "pcolormesh",
    "matshow",
    "errorbar",
    "tricontour",
    "tricontourf",
    "tripcolor",
})


_RENDERERS_DIR = (
    Path(__file__).resolve().parents[1]
    / "src" / "TRITON_SWMM_toolkit" / "report_renderers"
)


def _renderer_files() -> list[Path]:
    return sorted(
        p for p in _RENDERERS_DIR.glob("*.py")
        if not p.stem.startswith("_")
    )


def _attach_parents(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node  # type: ignore[attr-defined]


def _is_artist_context_call(call: ast.Call) -> bool:
    """True if `call` looks like `<expr>.artist(...)`."""
    return (
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "artist"
    )


def _has_artist_with_ancestor(node: ast.AST) -> bool:
    """True if any `With` ancestor's items contain `<expr>.artist(...)`."""
    cur: ast.AST | None = getattr(node, "parent", None)
    while cur is not None:
        if isinstance(cur, ast.With):
            for item in cur.items:
                expr = item.context_expr
                if isinstance(expr, ast.Call) and _is_artist_context_call(expr):
                    return True
        cur = getattr(cur, "parent", None)
    return False


@pytest.mark.parametrize("path", _renderer_files(), ids=lambda p: p.name)
def test_artist_calls_enclosed_in_provenance_block(path: Path) -> None:
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    _attach_parents(tree)

    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in _ARTIST_METHODS:
            continue
        if not _has_artist_with_ancestor(node):
            violations.append(
                f"{path.name}:{node.lineno}: "
                f"`.{node.func.attr}(...)` is not enclosed in a "
                f"`with <name>.artist(...)` block"
            )

    if violations:
        msg = "\n".join(violations)
        raise AssertionError(
            f"Provenance discipline violations in {path.name}:\n{msg}"
        )


@pytest.mark.parametrize("path", _renderer_files(), ids=lambda p: p.name)
def test_renderer_module_has_provenance_block(path: Path) -> None:
    """Guard against alias-rebinds.

    Every renderer must contain at least one `with <name>.artist(...)` block,
    even if no direct artist methods are detected (e.g., when artists are
    produced by external helpers like `plot_continuous_raster`).
    """
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.With):
            for item in node.items:
                expr = item.context_expr
                if isinstance(expr, ast.Call) and _is_artist_context_call(expr):
                    found = True
                    break
            if found:
                break

    assert found, (
        f"{path.name}: no `with <name>.artist(...)` block found. Every "
        f"renderer module must bind a `ProvenanceLog` and wrap every artist "
        f"creation in a `with prov.artist(...)` block."
    )
