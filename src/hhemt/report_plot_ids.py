"""Canonical report plot-ID minting (ADR-2, single source of truth).

Every report figure carries ONE canonical plot ID minted here. The ID is
threaded three ways with no second source of truth:

  1. the figure-output file STEM (via `plot_output_template`),
  2. the `plot_id` field stamped into the `*.manifest.json` sidecar
     (via `canonical_plot_id`, called by `_figure_emission` at render time),
  3. the `expand()` / `rule all` / `render_report` input lists in all three
     rule-emission generators (workflow.py, bundle/snakefile_generator.py,
     reprocess_snakefile_generator.py) -- all derive from this module.

Grammar (ADR-2): {renderer_kind}[__{descriptor}][__sa.{sa_id}][__evt.{event_id}]
using "." as the within-segment separator. "-" is NOT used: it is absent
from the enforced wildcard charset ^[A-Za-z0-9_.]+$ (C-CHARSET).

This module is the ONLY layout-relevant artifact for figure stems (Decision
D2 option (b)): it is listed in _layout_relevant_files.yaml::layout_relevant.
paths so CI Check B fires on stem-grammar edits WITHOUT putting all of
workflow.py (the toolkit's largest module, overwhelmingly non-layout logic)
in the Check-B blast radius.
"""

from __future__ import annotations

from typing import Literal

#: Per-renderer output extension under each static backend. Part of the
#: on-disk figure filename, hence co-located with the ID grammar (both are
#: stem determinants governed by C-LAYOUT). Lifted verbatim from workflow.py.
_OUTPUT_EXT_BY_RENDERER: dict[str, dict[str, str]] = {
    # Plotly chart-renderer outputs are interactive HTML emitted via pio.to_html;
    # extension must be .html so Snakemake's report engine sets mime_type=text/html
    # and dispatches each figure via <iframe> (which loads HTML correctly under
    # both HTTP and file:// double-click). A .svg extension here triggers
    # mime_type=image/svg+xml and an <img> dispatch that fails to parse HTML.
    "system_overview": {"matplotlib": ".png", "plotly": ".html"},
    "per_sim_peak_flood_depth": {"matplotlib": ".png", "plotly": ".html"},
    "per_sim_conduit_flow": {"matplotlib": ".png", "plotly": ".html"},
    "per_sim_per_sa_peak_flood_depth": {"matplotlib": ".png", "plotly": ".html"},
    "per_sim_per_sa_conduit_flow": {"matplotlib": ".png", "plotly": ".html"},
    "sensitivity_benchmarking": {"matplotlib": ".png", "plotly": ".html"},
    "per_analysis_summary": {"matplotlib": ".html", "plotly": ".html"},
    "scenario_status_appendix": {"matplotlib": ".html", "plotly": ".html"},
    "errors_and_warnings": {"matplotlib": ".html", "plotly": ".html"},
    # Disk utilization is a table renderer -- emits HTML unconditionally
    # (no matplotlib raster branch). Matches per_analysis_summary /
    # scenario_status_appendix / errors_and_warnings.
    "disk_utilization": {"matplotlib": ".html", "plotly": ".html"},
    # Metadata (ADR-14 / C10) is an all-static table/prose page -- emits HTML
    # unconditionally, like the other table renderers above.
    "metadata": {"matplotlib": ".html", "plotly": ".html"},
}


def output_ext_for(static_backend: Literal["matplotlib", "plotly"], renderer_module: str) -> str:
    """Return the output extension for a renderer under the given static backend.

    Three-place output_ext coupling: rule output path, rule report() first arg,
    and rule_all / render_report input lists must all use this same extension.
    """
    return _OUTPUT_EXT_BY_RENDERER[renderer_module][static_backend]


def canonical_plot_id(
    renderer_kind: str,
    *,
    descriptor: str | None = None,
    sa_id: str | None = None,
    event_id: str | None = None,
) -> str:
    """Mint a canonical plot ID per the ADR-2 grammar.

    Segments are joined with "__"; within a segment the separator is ".".
    Order is fixed: renderer_kind, then optional descriptor, then optional
    sa.{sa_id}, then optional evt.{event_id}.

    Callers pass CONCRETE wildcard values (e.g. event_id="year.9_...") to get
    the display/manifest string; the rule-emission generators pass the literal
    brace token (e.g. event_id="{event_id}") to get a Snakefile path template
    via `plot_output_template`.
    """
    segments = [renderer_kind]
    if descriptor is not None:
        segments.append(descriptor)
    if sa_id is not None:
        segments.append(f"sa.{sa_id}")
    if event_id is not None:
        segments.append(f"evt.{event_id}")
    return "__".join(segments)


def plot_output_template(
    *,
    renderer_kind: str,
    subdir: str,
    descriptor: str | None = None,
    sa_id: str | None = None,
    event_id: str | None = None,
) -> str:
    """Return the Snakefile-relative output path template for a figure.

    `subdir` is the plots/ subdirectory the figure lives under (e.g.
    "plots", "plots/per_sim/{event_id}", "plots/sensitivity/benchmarking").
    The returned template embeds the canonical plot ID as the file STEM and
    ends in the literal "__OUTPUT_EXT__" token that `_emit_plot_rule`
    substitutes per-backend. Snakemake "{wildcard}" braces in `subdir` and in
    the minted ID survive unescaped (plain-string assembly, no .format()).
    """
    plot_id = canonical_plot_id(
        renderer_kind,
        descriptor=descriptor,
        sa_id=sa_id,
        event_id=event_id,
    )
    return f"{subdir}/{plot_id}__OUTPUT_EXT__"
