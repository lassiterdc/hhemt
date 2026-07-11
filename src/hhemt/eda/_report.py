"""EDA report assembler (ADR-10, repurposed under ADR-14): a GENERIC, reusable
scrollable-HTML composer.

`render_scrollable_report` is EDA-agnostic — a figure-collection + table-collection
assembler over a Jinja2 template — so the parked scrollable-singlepage-report idea
can reuse it. The Plotly bundle is inlined EXACTLY ONCE via `_figure_divs`
(first-figure-inline + rest include_plotlyjs=False; full_html=False).
`cross_sim_identity_figure_from_root` rebuilds the EDA figure from its carried
eda/<plot_id>.zarr (reused by the notebook seed cell, ADR-14). The former
`assemble_eda_report` wrapper was trimmed at Phase 5 — the notebook + best-effort
nbconvert HTML export supersedes the standalone-HTML doc path (ADR-14).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jinja2
import plotly.graph_objects as go
import plotly.io as pio


# FigureSpec + _figure_divs: the data-visualization-specialist FQ1 VMS (the
# first-figure-inline single-bundle mechanism). Applied verbatim from the master
# plan's Implementation Research; this is the single place the bundle-once
# invariant lives.
@dataclass
class FigureSpec:
    """One Plotly figure destined for the scrollable report."""

    figure: go.Figure
    title: str = ""
    config: dict[str, Any] = field(default_factory=dict)


def _figure_divs(figures: list[FigureSpec], *, plotly_js_mode: str) -> tuple[list[str], str]:
    """Render each FigureSpec to a self-contained <div> snippet, inlining the
    Plotly bundle EXACTLY ONCE.

    The first figure carries the bundle (`include_plotlyjs` resolved from
    `plotly_js_mode`: 'inline' -> True, 'cdn' -> 'cdn'); every subsequent
    figure is emitted with `include_plotlyjs=False`. `full_html=False` is
    MANDATORY on every figure so the snippets concatenate into one document.
    Plotly mints a fresh UUIDv4 div id per call, so ids never collide.

    Returns (list_of_div_html, js_mode_used). Caller drops the divs into the
    page template in order; the first div carries the <script> bundle.
    """
    first_js: bool | str = True if plotly_js_mode == "inline" else "cdn"
    divs: list[str] = []
    for idx, spec in enumerate(figures):
        # Toggle on INDEX, never on figure identity — a reordered/conditional
        # first figure must never silently drop the bundle.
        include_js: bool | str = first_js if idx == 0 else False
        # Ensure an explicit px height so the scrollable stack does not collapse
        # to zero on height:100%-of-auto-height parents.
        if spec.figure.layout.height is None:
            spec.figure.update_layout(height=500)
        divs.append(
            pio.to_html(
                spec.figure,
                include_plotlyjs=include_js,
                full_html=False,
                config={"displaylogo": False, **spec.config},
            )
        )
    return divs, plotly_js_mode


@dataclass
class TableSpec:
    """One Tabulator data table (e.g. the EDA datasets reference)."""

    title: str
    columns: list[str]
    rows: list[list[str]]


def render_scrollable_report(
    figures: list[FigureSpec],
    tables: list[TableSpec],
    *,
    title: str,
    brand: dict[str, Any] | None = None,
    plotly_js_mode: str = "inline",
    tabulator_js_mode: str = "inline",
) -> str:
    """Compose figures + tables into ONE self-contained scrollable HTML string.

    GENERIC (no EDA coupling). The Plotly bundle is inlined once across all
    figures (via _figure_divs); Tabulator loads as an independent global
    (inline or cdn). Returns the full HTML document text.
    """
    # _figure_divs returns (divs, js_mode_used). The Plotly bundle is embedded
    # INSIDE divs[0] (first figure carries include_plotlyjs=True), NOT returned
    # separately — so figure_divs[0] (emitted in <body>) carries the bundle.
    # Do NOT inject the second return value into <head>: it is the mode string,
    # not bundle HTML (master FQ1 VMS: _figure_divs -> tuple[list[str], str]).
    fig_divs, _js_mode_used = _figure_divs(figures, plotly_js_mode=plotly_js_mode)
    env = jinja2.Environment(
        loader=jinja2.PackageLoader("hhemt", "report_templates"),
        autoescape=jinja2.select_autoescape(["html"]),
    )
    template = env.get_template("eda_report.html.j2")
    return template.render(
        title=title,
        brand=brand or {},
        figure_divs=fig_divs,
        tables=tables,
        tabulator_js_mode=tabulator_js_mode,
    )


def config_diff_maps_figure_from_root(root: Path) -> go.Figure:
    """Re-build the config-diff-maps figure from the carried sensitivity_datatree.zarr.

    The scrollable doc is RE-RENDERED from the consolidated tree (NOT assembled from
    the saved standalone plots/eda/*.html fragments, which are full_html=True documents
    that cannot be concatenated). Delegates to the same builder the per-figure renderer
    uses so the doc figure matches the standalone artifact.
    """
    from hhemt.eda._config_diff import build_config_diff_figure

    return build_config_diff_figure(root)
