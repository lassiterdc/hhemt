"""AST-level enforcement of the per-artist provenance discipline.

Walks every renderer module under `src/hhemt/report_renderers/`
(excluding files whose stem starts with `_`) and asserts that every artist-
creating matplotlib call inside the module is enclosed in a
`with <name>.artist(...)` context block. Additionally asserts that each
non-skipped renderer module contains at least one such block — guard against
alias-rebinds (e.g., `plot = ax.plot; plot(...)`) that would trivially satisfy
a per-call check.

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
_ARTIST_METHODS: frozenset[str] = frozenset(
    {
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
    }
)


# Plotly trace classes (constructor names from `plotly.graph_objects`).
# Curated; extend as new trace types appear in renderer bodies. Phase 1
# substrate; Phases 2-5 add Plotly-using renderers that exercise this check.
_PLOTLY_TRACE_CLASSES: frozenset[str] = frozenset(
    {
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
    }
)


# go-import alias guard: every Plotly-using renderer must import as
# `import plotly.graph_objects as go` (no alternate alias). The provenance
# walker matches `go.<TraceClass>(...)` calls only; alias rebinds bypass
# the lint and are an alias-rebind attack surface analogous to
# `plot = ax.plot; plot(...)` on the matplotlib side.
_REQUIRED_GO_IMPORT_ALIAS = "go"


_RENDERERS_DIR = Path(__file__).resolve().parents[1] / "src" / "hhemt" / "report_renderers"


def _renderer_files() -> list[Path]:
    return sorted(p for p in _RENDERERS_DIR.glob("*.py") if not p.stem.startswith("_"))


def _attach_parents(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node  # type: ignore[attr-defined]


def _is_artist_context_call(call: ast.Call) -> bool:
    """True if `call` looks like `<expr>.artist(...)`."""
    return isinstance(call.func, ast.Attribute) and call.func.attr == "artist"


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
        if isinstance(node.func, ast.Attribute) and node.func.attr in _ARTIST_METHODS:
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
        raise AssertionError(f"Provenance discipline violations in {path.name}:\n{msg}")


# Renderers that create ZERO artists locally and therefore cannot satisfy the
# "at least one `with <name>.artist(...)` block" proxy below. This is a NAME LIST
# and there is no getting around that: a module that creates no artists is, at the
# AST level, indistinguishable from one that forgot to. What the list buys is that
# each entry's RATIONALE is machine-checked rather than merely asserted in a
# comment -- the two entries have DIFFERENT shapes and a single shared rationale
# would be false of one of them.
#
#   "pure_delegate"  -- owns NO provenance. Delegates figure emission AND its
#                       provenance to a module that carries the discipline itself.
#                       `eda_compute_sensitivity` (Gotcha 67d / R11) delegates to
#                       `hhemt.eda._plotting.render_eda_plots`, which emits via
#                       `emit_plot_with_sources` and owns the provenance block.
#                       Counter-assertion: zero artist calls AND binds NO
#                       ProvenanceLog (binding one would mean it owns provenance
#                       after all, i.e. it is the other kind).
#
#   "composer"       -- OWNS provenance. Binds a ProvenanceLog, threads it into
#                       delegate builders that write into it, and hands it to
#                       `emit_plot_with_sources(provenance=...)` itself.
#                       `per_sim_event_page` composes the per-model figures built
#                       by `per_sim_peak_flood_depth` / `per_sim_conduit_flow`.
#                       Counter-assertion: zero artist calls AND binds a
#                       ProvenanceLog AND threads it to a `provenance=` kwarg.
#
# KNOWN CEILING, stated so nobody prices this higher than it is: "zero artist
# calls" is an AST-visible property, and an alias-rebind (`plot = ax.plot;
# plot(...)`) produces zero AST-visible artist calls BY CONSTRUCTION. So the
# counter-assertion catches a DIRECT `ax.plot(...)` added to an exempt module --
# it does NOT catch the alias-rebind this test's docstring names. That limitation
# is inherent to the exemption and predates this entry; it is why the list is kept
# to modules whose composer/delegate shape is independently reviewed, not opened
# to a general structural predicate.
#
# Why NOT a general "accept any zero-artist module that owns a ProvenanceLog"
# rule: measured against the current tree, NINE other renderers (workflow_
# performance, metadata, errors_and_warnings, per_analysis_summary,
# scenario_status_appendix, disk_utilization, and the three cross_experiment_*
# modules) also have zero AST-visible artist calls and bind a threaded
# ProvenanceLog -- they pass today only because they additionally carry a
# `with prov.artist(...)` block declaring the provenance of work done inside HTML
# helper functions. A general predicate would stop REQUIRING those blocks,
# trading guard strength on nine modules to accommodate one.
_EXEMPT_RENDERER_KINDS: dict[str, str] = {
    "eda_compute_sensitivity.py": "pure_delegate",
    "per_sim_event_page.py": "composer",
}
_DELEGATING_RENDERERS: frozenset[str] = frozenset(_EXEMPT_RENDERER_KINDS)


def _binds_provenance_log(tree: ast.AST) -> set[str]:
    """Names bound to a `ProvenanceLog()` construction."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            func = node.value.func
            ctor = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
            if ctor == "ProvenanceLog":
                names.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return names


def _threads_provenance_kwarg(tree: ast.AST, names: set[str]) -> bool:
    """True if some call receives `provenance=<one of names>`."""
    return any(
        kw.arg == "provenance" and isinstance(kw.value, ast.Name) and kw.value.id in names
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for kw in node.keywords
    )


@pytest.mark.parametrize("path", _renderer_files(), ids=lambda p: p.name)
def test_renderer_module_has_provenance_block(path: Path) -> None:
    """Guard against alias-rebinds.

    Every renderer must contain at least one `with <name>.artist(...)` block,
    even if no direct artist methods are detected (e.g., when artists are
    produced by external helpers like `plot_continuous_raster`).

    Exception: a module listed in ``_EXEMPT_RENDERER_KINDS`` creates no artists
    locally, so it cannot satisfy this proxy. It must instead prove ZERO
    artist-creating calls AND satisfy its declared exemption KIND's own
    counter-assertion (``pure_delegate`` binds no ProvenanceLog; ``composer`` binds
    one and threads it to the emit).

    That is NOT strictly stronger than being wrapped in a provenance block, and the
    earlier wording here claimed it was. An alias-rebind (``plot = ax.plot;
    plot(...)``) produces a ``Call`` whose ``func`` is a ``Name`` rather than an
    ``Attribute``, so it yields zero AST-visible artist calls BY CONSTRUCTION --
    which is exactly what the zero-call counter-assertion accepts as proof of
    innocence. The counter-assertion catches a DIRECT ``ax.plot(...)`` added to an
    exempt module; it does not catch a rebind. See the ceiling note on
    ``_EXEMPT_RENDERER_KINDS``.
    """
    source = path.read_text()

    if path.name in _DELEGATING_RENDERERS:
        # Positive counter-assertion: the exemption is only valid while the module
        # genuinely creates no artists.
        tree = ast.parse(source, filename=str(path))
        # Same two predicates `_lint_source` uses: the inlined matplotlib branch
        # (`<expr>.<method>(...)` with method in _ARTIST_METHODS) and the plotly
        # branch (`go.<TraceClass>(...)`). Kept in lockstep with _lint_source: any
        # method added to _ARTIST_METHODS automatically tightens this too.
        artist_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and (
                (isinstance(node.func, ast.Attribute) and node.func.attr in _ARTIST_METHODS)
                or _is_plotly_trace_call(node)
            )
        ]
        kind = _EXEMPT_RENDERER_KINDS[path.name]
        assert not artist_calls, (
            f"{path.name} is listed in _EXEMPT_RENDERER_KINDS (exempt from the "
            f"provenance-block requirement) but creates "
            f"{len(artist_calls)} artist(s) at line(s) "
            f"{[n.lineno for n in artist_calls]}. A module that creates artists "
            f"MUST bind a ProvenanceLog and wrap them in `with prov.artist(...)`; "
            f"remove it from _EXEMPT_RENDERER_KINDS."
        )
        # Per-kind counter-assertion: the exemption's stated RATIONALE must hold,
        # not just the zero-artist precondition it shares with the other kind.
        log_names = _binds_provenance_log(tree)
        if kind == "pure_delegate":
            assert not log_names, (
                f"{path.name} is exempt as a 'pure_delegate' (owns NO provenance; "
                f"the delegate module carries the discipline) but binds a "
                f"ProvenanceLog to {sorted(log_names)}. A module that owns a "
                f"ProvenanceLog is a 'composer', not a pure delegate -- either "
                f"reclassify it in _EXEMPT_RENDERER_KINDS or drop the log."
            )
        elif kind == "composer":
            assert log_names, (
                f"{path.name} is exempt as a 'composer' (creates no artists "
                f"locally but OWNS the provenance its delegates write into) yet "
                f"binds no ProvenanceLog. A composer must bind one and hand it to "
                f"`emit_plot_with_sources(provenance=...)`; if provenance is "
                f"genuinely owned downstream, reclassify it as 'pure_delegate'."
            )
            assert _threads_provenance_kwarg(tree, log_names), (
                f"{path.name} is exempt as a 'composer' and binds a ProvenanceLog "
                f"{sorted(log_names)}, but never passes it as `provenance=` to an "
                f"emit call -- so nothing it collects reaches the manifest sidecar. "
                f"Thread it into `emit_plot_with_sources(..., provenance=prov)`."
            )
        else:  # pragma: no cover -- guarded by the mapping's own vocabulary
            raise AssertionError(
                f"{path.name}: unknown exemption kind {kind!r}; expected one of " f"{{'pure_delegate', 'composer'}}."
            )
        return

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


# ============================================================================
# Phase 1 extension: synthetic-source tests for Plotly trace discipline
# ============================================================================


def test_plotly_trace_in_provenance_block_compliant() -> None:
    """Lint must accept ``go.Heatmap(...)`` wrapped in
    ``with prov.artist(...):``."""
    src = textwrap.dedent("""
        import plotly.graph_objects as go

        def render(analysis, report_cfg, output_path, **kwargs):
            prov = ProvenanceLog()
            fig = go.Figure()
            with prov.artist("depth_raster"):
                fig.add_trace(go.Heatmap(z=[[1, 2], [3, 4]]))
            return fig
    """)
    violations = _lint_source(src)
    assert violations == [], f"Expected zero violations, got {violations}"


def test_plotly_trace_outside_provenance_block_rejected() -> None:
    """Lint must reject ``go.Heatmap(...)`` outside any
    ``with prov.artist(...):`` block."""
    src = textwrap.dedent("""
        import plotly.graph_objects as go

        def render(analysis, report_cfg, output_path, **kwargs):
            fig = go.Figure()
            fig.add_trace(go.Heatmap(z=[[1, 2], [3, 4]]))
            return fig
    """)
    violations = _lint_source(src)
    assert any("Heatmap" in v for v in violations), f"Expected violation for unprotected go.Heatmap; got {violations}"


def test_plotly_alias_rebind_rejected() -> None:
    """Lint must reject ``import plotly.graph_objects as <not_go>`` in
    renderer modules — alias rebinds bypass the ``go.<TraceClass>`` matcher
    in :func:`_is_plotly_trace_call`. Parallel to the matplotlib-side
    alias-rebind guard the existing module documents (per its docstring)."""
    src = textwrap.dedent("""
        import plotly.graph_objects as go_alias

        def render(analysis, report_cfg, output_path, **kwargs):
            fig = go_alias.Figure()
            with prov.artist("depth_raster"):
                fig.add_trace(go_alias.Heatmap(z=[[1, 2], [3, 4]]))
            return fig
    """)
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
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "emit_plot_with_sources":
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
        is_empty_literal = isinstance(arg, ast.List | ast.Tuple) and len(arg.elts) == 0
        assert not is_empty_literal, (
            f"{path.name}: emit_plot_with_sources called with a literal "
            f"empty source_paths. Pass real sources or, for a genuinely "
            f"source-less figure, pass allow_empty_sources=True."
        )
