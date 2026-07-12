"""AST-level enforcement of the per-artist provenance discipline.

Walks every renderer module under `src/hhemt/report_renderers/`
(excluding files whose stem starts with `_`) and asserts that every artist-
creating matplotlib call inside the module is enclosed in a
`with <name>.artist(...)` context block. Additionally asserts that each
FIGURE-EMITTING renderer module contains at least one such block — guard
against alias-rebinds (e.g., `plot = ax.plot; plot(...)`) that would trivially
satisfy a per-call check.

Delegating-adapter exemption: the module-level existence check applies only to
renderers that OWN a terminal `emit_plot_with_sources(...)` call. A
`ProvenanceLog`'s only sink is `manifest_payload["artists"] = prov.serialize()`
inside that call (`_figure_emission.py`), so a renderer that owns no emit call
owns no manifest and has nowhere to thread a log — binding one would be a dead
object that satisfies the AST matcher and nothing else. Pure DELEGATING
adapters (e.g. `eda_compute_sensitivity.py`, which forwards to
`eda/_plotting.render_eda_plots`) are therefore exempt from this one check.
They remain fully subject to `test_artist_calls_enclosed_in_provenance_block`:
if such a module ever grows an artist call, the per-call check still fires.
Note this exemption is behavioral, NOT a filename list — a renderer cannot opt
out by being named, only by provably emitting no figure.

The test surfaces a clear file:line error message on violation.

Phase 1 extension: additionally enforces the same discipline for Plotly
trace constructions (``go.<TraceClass>(...)``). The lint requires that
``plotly.graph_objects`` is imported as ``import plotly.graph_objects as go``
(any other alias bypasses the matcher, an alias-rebind attack surface
analogous to ``plot = ax.plot; plot(...)`` on the matplotlib side).
"""

from __future__ import annotations

import ast
import textwrap
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


# Plotly trace classes (constructor names from `plotly.graph_objects`).
# Curated; extend as new trace types appear in renderer bodies. Phase 1
# substrate; Phases 2-5 add Plotly-using renderers that exercise this check.
_PLOTLY_TRACE_CLASSES: frozenset[str] = frozenset({
    "Heatmap",
    "Scatter",
    "Scattergl",
    "Scattergeo",
    "Scattermapbox",
    "Bar",
    "Box",
    "Histogram",
    "Histogram2d",
    "Contour",
    "Surface",
    "Violin",
    "Pie",
    "Choropleth",
    "Densitymapbox",
    "Image",
})


# go-import alias guard: every Plotly-using renderer must import as
# `import plotly.graph_objects as go` (no alternate alias). The provenance
# walker matches `go.<TraceClass>(...)` calls only; alias rebinds bypass
# the lint and are an alias-rebind attack surface analogous to
# `plot = ax.plot; plot(...)` on the matplotlib side.
_REQUIRED_GO_IMPORT_ALIAS = "go"


_RENDERERS_DIR = (
    Path(__file__).resolve().parents[1]
    / "src" / "hhemt" / "report_renderers"
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


def _is_plotly_trace_call(call: ast.Call) -> bool:
    """True if `call` looks like ``go.<TraceClass>(...)``."""
    return (
        isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == _REQUIRED_GO_IMPORT_ALIAS
        and call.func.attr in _PLOTLY_TRACE_CLASSES
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


def _check_plotly_import_alias(tree: ast.AST, filename: str) -> list[str]:
    """Detect plotly.graph_objects imports under a non-`go` alias.

    Walks Import / ImportFrom nodes. Rejects:

    - ``import plotly.graph_objects as <not_go>``
    - ``import plotly.graph_objects`` (no alias — would bind as
      ``plotly.graph_objects.<TraceClass>`` which the matcher doesn't see)
    - ``from plotly.graph_objects import <TraceClass>`` (bare name binds
      bypass the ``go.<TraceClass>`` matcher)
    """
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "plotly.graph_objects":
                    if alias.asname is None or alias.asname != _REQUIRED_GO_IMPORT_ALIAS:
                        violations.append(
                            f"{filename}:{node.lineno}: "
                            f"plotly.graph_objects must be imported as "
                            f"`import plotly.graph_objects as go`; got alias "
                            f"{alias.asname!r}"
                        )
        elif isinstance(node, ast.ImportFrom):
            if node.module == "plotly.graph_objects":
                violations.append(
                    f"{filename}:{node.lineno}: "
                    f"`from plotly.graph_objects import ...` is forbidden — "
                    f"use `import plotly.graph_objects as go` so the "
                    f"`go.<TraceClass>` matcher in the provenance lint sees "
                    f"the trace construction"
                )
    return violations


def _lint_source(src: str, filename: str = "<synthetic>") -> list[str]:
    """Run the provenance-discipline lint over a Python source string.

    Returns a list of human-readable violation messages. Empty list means
    all checks pass.
    """
    tree = ast.parse(src, filename=filename)
    _attach_parents(tree)

    violations: list[str] = []
    # First check: plotly alias guard (file-level static check).
    violations.extend(_check_plotly_import_alias(tree, filename))

    # Second check: matplotlib artist calls + Plotly trace calls inside
    # `with prov.artist(...)` blocks.
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # matplotlib branch: <expr>.<artist_method>(...)
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in _ARTIST_METHODS
        ):
            if not _has_artist_with_ancestor(node):
                violations.append(
                    f"{filename}:{node.lineno}: "
                    f"`.{node.func.attr}(...)` is not enclosed in a "
                    f"`with <name>.artist(...)` block"
                )
            continue
        # Plotly branch: go.<TraceClass>(...)
        if _is_plotly_trace_call(node):
            if not _has_artist_with_ancestor(node):
                violations.append(
                    f"{filename}:{node.lineno}: "
                    f"`go.{node.func.attr}(...)` is not enclosed in a "
                    f"`with <name>.artist(...)` block"
                )

    return violations


@pytest.mark.parametrize("path", _renderer_files(), ids=lambda p: p.name)
def test_artist_calls_enclosed_in_provenance_block(path: Path) -> None:
    source = path.read_text()
    violations = _lint_source(source, filename=path.name)
    if violations:
        msg = "\n".join(violations)
        raise AssertionError(
            f"Provenance discipline violations in {path.name}:\n{msg}"
        )


def _owns_figure_emission(tree: ast.AST) -> bool:
    """True if the module owns a terminal `emit_plot_with_sources(...)` call.

    This is the exemption predicate for the module-level existence check. A
    `ProvenanceLog` reaches a figure's manifest ONLY via the `provenance=`
    argument of `emit_plot_with_sources` (`_figure_emission.py`, which writes
    `manifest_payload["artists"] = provenance.serialize()`). A renderer that
    owns no emit call owns no manifest, so it has no sink for a log and no
    artists to describe — it is a pure DELEGATING adapter whose figure emission
    happens downstream (e.g. `eda_compute_sensitivity.py` -> `eda/_plotting`).

    NOTE the criterion is emit-ownership, NOT "creates zero artists": the
    pure-HTML table renderers create zero matplotlib artists yet correctly bind
    a log, because they own their emit call and therefore own a manifest.
    """
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "emit_plot_with_sources"
        ):
            return True
    return False


@pytest.mark.parametrize("path", _renderer_files(), ids=lambda p: p.name)
def test_renderer_module_has_provenance_block(path: Path) -> None:
    """Guard against alias-rebinds.

    Every FIGURE-EMITTING renderer must contain at least one
    `with <name>.artist(...)` block, even if no direct artist methods are
    detected (e.g., when artists are produced by external helpers like
    `plot_continuous_raster`).

    Pure delegating adapters — renderers that own no `emit_plot_with_sources`
    call — are exempt: they own no manifest, so a `ProvenanceLog` bound in them
    would never be serialized. See `_owns_figure_emission` and the module
    docstring. They stay subject to the per-call check above.
    """
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))

    if not _owns_figure_emission(tree):
        pytest.skip(
            f"{path.name}: delegating adapter (owns no `emit_plot_with_sources` "
            f"call, therefore no manifest to thread a ProvenanceLog into); "
            f"exempt from the module-level provenance-block check. Artist calls, "
            f"if any are ever added, are still enforced by "
            f"test_artist_calls_enclosed_in_provenance_block."
        )

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
        f"figure-emitting renderer module must bind a `ProvenanceLog` and wrap "
        f"every artist creation in a `with prov.artist(...)` block."
    )


# ============================================================================
# Phase 1 extension: synthetic-source tests for Plotly trace discipline
# ============================================================================


def test_plotly_trace_in_provenance_block_compliant() -> None:
    """Lint must accept ``go.Heatmap(...)`` wrapped in
    ``with prov.artist(...):``."""
    src = textwrap.dedent('''
        import plotly.graph_objects as go

        def render(analysis, report_cfg, output_path, **kwargs):
            prov = ProvenanceLog()
            fig = go.Figure()
            with prov.artist("depth_raster"):
                fig.add_trace(go.Heatmap(z=[[1, 2], [3, 4]]))
            return fig
    ''')
    violations = _lint_source(src)
    assert violations == [], f"Expected zero violations, got {violations}"


def test_plotly_trace_outside_provenance_block_rejected() -> None:
    """Lint must reject ``go.Heatmap(...)`` outside any
    ``with prov.artist(...):`` block."""
    src = textwrap.dedent('''
        import plotly.graph_objects as go

        def render(analysis, report_cfg, output_path, **kwargs):
            fig = go.Figure()
            fig.add_trace(go.Heatmap(z=[[1, 2], [3, 4]]))
            return fig
    ''')
    violations = _lint_source(src)
    assert any(
        "Heatmap" in v for v in violations
    ), f"Expected violation for unprotected go.Heatmap; got {violations}"


def test_plotly_alias_rebind_rejected() -> None:
    """Lint must reject ``import plotly.graph_objects as <not_go>`` in
    renderer modules — alias rebinds bypass the ``go.<TraceClass>`` matcher
    in :func:`_is_plotly_trace_call`. Parallel to the matplotlib-side
    alias-rebind guard the existing module documents (per its docstring)."""
    src = textwrap.dedent('''
        import plotly.graph_objects as go_alias

        def render(analysis, report_cfg, output_path, **kwargs):
            fig = go_alias.Figure()
            with prov.artist("depth_raster"):
                fig.add_trace(go_alias.Heatmap(z=[[1, 2], [3, 4]]))
            return fig
    ''')
    violations = _lint_source(src)
    assert any(
        "alias" in v.lower() or "go_alias" in v for v in violations
    ), f"Expected violation for non-`go` Plotly alias; got {violations}"


# ============================================================================
# ADR-6 Gate-B: static declared-check — reject literal-empty source_paths
# ============================================================================


def _emit_call_source_paths_arg(tree: ast.AST) -> list[ast.AST]:
    """Return the source_paths argument node of each emit_plot_with_sources call.

    Matches `emit_plot_with_sources(...)` called as a bare imported name; the
    source_paths argument is the `source_paths=` keyword if present, else the
    third positional argument.
    """
    args: list[ast.AST] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "emit_plot_with_sources"
        ):
            kw = {k.arg: k.value for k in node.keywords}
            if "source_paths" in kw:
                args.append(kw["source_paths"])
            elif len(node.args) >= 3:
                args.append(node.args[2])
    return args


@pytest.mark.parametrize("path", _renderer_files(), ids=lambda p: p.name)
def test_renderer_declares_nonliteral_empty_sources(path: Path) -> None:
    """Reject a renderer whose emit_plot_with_sources call passes a literal
    empty list/tuple as source_paths (ADR-6 Gate-B, the static half). A
    runtime-dynamic empty (closure returns []) is caught by the render-time
    gate (Gate-A in _figure_emission), not here."""
    tree = ast.parse(path.read_text(), filename=str(path))
    for arg in _emit_call_source_paths_arg(tree):
        is_empty_literal = (
            isinstance(arg, (ast.List, ast.Tuple)) and len(arg.elts) == 0
        )
        assert not is_empty_literal, (
            f"{path.name}: emit_plot_with_sources called with a literal "
            f"empty source_paths. Pass real sources or, for a genuinely "
            f"source-less figure, pass allow_empty_sources=True."
        )
