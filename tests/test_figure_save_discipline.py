"""AST-level enforcement of the figure-save routing discipline.

Every ``report_renderers/*.py`` module (excluding leading-underscore helpers)
must route figure persistence through
``_figure_emission.emit_plot_with_sources(...)``. Direct calls to
``Figure.savefig(...)``, ``plt.savefig(...)``, or
``{path}.write_text({html}, ...)`` from a renderer module's ``render(...)``
body are rejected — the uniform helper is the single emit-site that produces
the preview-PNG sibling and the ``{stem}.manifest.json`` sidecar (per the
``report renderers accept uniform signature`` stipulation).

This complements ``test_provenance_discipline.py``: provenance-block
discipline governs artist creation; figure-save discipline governs artist
persistence.

The lint accepts BOTH branches of the extended helper (Phase 1 of the
interactive-report-renderers plan): a ``Figure``-typed first argument
(matplotlib branch) AND a ``str``-typed first argument (HTML branch).

Lint-scope note: rejection of ``.savefig(...)`` and ``.write_text(...)`` /
``.write_bytes(...)`` is scoped to call sites inside ``def render(...)``
bodies. Module-private helpers (e.g., ``_render_caption_rst``,
``_emit_model_type_skip_placeholder``) MAY persist non-figure or
degenerate-case artifacts via direct ``Path.write_text(...)`` /
``Figure.savefig(...)`` because they are reviewed at code-review time and
not expected to emit the canonical figure. The lint trusts module-private
helpers.

Relationship to test_provenance_discipline.py: the two modules enforce
orthogonal concerns. provenance_discipline governs artist-creation context
(every artist call sits inside ``with prov.artist(...)``);
figure_save_discipline governs artist persistence (every artist save
routes through emit_plot_with_sources). A renderer can pass one and fail
the other; both are enforced.
"""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

import pytest

_RENDERERS_DIR = (
    Path(__file__).resolve().parents[1]
    / "src" / "TRITON_SWMM_toolkit" / "report_renderers"
)

_FORBIDDEN_SAVE_METHODS: frozenset[str] = frozenset({"savefig"})
_FORBIDDEN_TEXT_WRITERS_IN_RENDER: frozenset[str] = frozenset(
    {"write_text", "write_bytes"}
)


def _renderer_files() -> list[Path]:
    return sorted(
        p for p in _RENDERERS_DIR.glob("*.py")
        if not p.stem.startswith("_")
    )


def _is_method_call(call: ast.Call, method_set: frozenset[str]) -> bool:
    return (
        isinstance(call.func, ast.Attribute)
        and call.func.attr in method_set
    )


def _walk_render_function_calls(tree: ast.AST):
    """Yield (call_node, enclosing_funcname) for every Call inside a
    ``def render(...)`` (or ``def render_*``) function body."""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and (
            node.name == "render" or node.name.startswith("render_")
        ):
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    yield child, node.name


def _lint_source(src: str, filename: str = "<synthetic>") -> list[str]:
    tree = ast.parse(src, filename=filename)
    violations: list[str] = []

    # Both savefig and write_text/write_bytes rejection are scoped to call
    # sites inside `def render(...)` bodies. Module-private helpers (e.g.,
    # `_emit_model_type_skip_placeholder`, `_render_caption_rst`) MAY persist
    # non-figure or degenerate-case artifacts via direct ``Path.write_text``
    # / ``Figure.savefig`` because they are reviewed at code-review time and
    # not expected to emit the canonical figure. The lint trusts module-private
    # helpers.
    for call, fname in _walk_render_function_calls(tree):
        if _is_method_call(call, _FORBIDDEN_SAVE_METHODS):
            violations.append(
                f"{filename}:{call.lineno}: "
                f"`.{call.func.attr}(...)` inside def {fname}(...) "
                f"bypasses emit_plot_with_sources"
            )
        if _is_method_call(call, _FORBIDDEN_TEXT_WRITERS_IN_RENDER):
            violations.append(
                f"{filename}:{call.lineno}: "
                f"`.{call.func.attr}(...)` inside def {fname}(...) "
                f"bypasses emit_plot_with_sources"
            )

    return violations


@pytest.mark.parametrize("path", _renderer_files(), ids=lambda p: p.name)
def test_renderer_does_not_bypass_emit_helper(path: Path) -> None:
    src = path.read_text()
    violations = _lint_source(src, filename=path.name)
    if violations:
        msg = "\n".join(violations)
        raise AssertionError(
            f"Figure-save discipline violations in {path.name}:\n{msg}"
        )


def test_emit_plot_with_sources_html_branch_compliant() -> None:
    src = textwrap.dedent("""
        from TRITON_SWMM_toolkit.report_renderers._figure_emission import emit_plot_with_sources

        def render(analysis, report_cfg, output_path, **kwargs):
            html_str = "<div>example</div>"
            return emit_plot_with_sources(
                html_str,
                output_path,
                [],
                analysis_dir=analysis.analysis_dir,
                output_format="html",
            )
    """)
    assert _lint_source(src) == []


def test_direct_path_write_text_html_rejected() -> None:
    src = textwrap.dedent("""
        def render(analysis, report_cfg, output_path, **kwargs):
            html_str = "<div>example</div>"
            output_path.write_text(html_str, encoding="utf-8")
            return output_path
    """)
    violations = _lint_source(src)
    assert any("write_text" in v for v in violations), violations


def test_direct_savefig_rejected() -> None:
    src = textwrap.dedent("""
        import matplotlib.pyplot as plt

        def render(analysis, report_cfg, output_path, **kwargs):
            fig, ax = plt.subplots()
            fig.savefig(output_path)
            return output_path
    """)
    violations = _lint_source(src)
    assert any("savefig" in v for v in violations), violations
