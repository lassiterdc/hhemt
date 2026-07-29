"""EDA plotting family (ADR-10 / ADR-2 / ADR-6).

Free functions shared by `analysis.eda()` and `Bundle.eda()` (they take a `root`
Path + the configs, NOT a TRITONSWMM_analysis, so the Bundle non-subclass boundary
is honored). Each EDA plot emits via `emit_plot_with_sources` (HTML branch) under
MASTER-ROOTED `{root}/plots/eda/<plot_id>.html` and declares its
`{root}/eda/<plot_id>.zarr` data-prep artifact as a source - so the existing
harvest chain carries the dataset into a render bundle (D1 Option A). EDA plots
MUST NOT emit under plots/sensitivity/per_sim/sa-{N}/ (harvest re-roots that subtree
against subanalyses/sa_{N}/, which has no eda/ dir; see the master-rooted-emission
stipulation).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import plotly.graph_objects as go

from hhemt.eda._dem_resolution_plots import (
    _render_dem_resolution_cost_error,
    _render_dem_resolution_coupling_table,
    _render_dem_resolution_diff_maps,
    _render_dem_resolution_error_ecdf,
)
from hhemt.report_plot_ids import canonical_plot_id
from hhemt.report_renderers._figure_emission import emit_plot_with_sources

if TYPE_CHECKING:
    from hhemt.config.analysis import analysis_config
    from hhemt.config.eda import eda_config


def render_eda_plots(
    root: Path,
    *,
    cfg_analysis: analysis_config,
    eda_cfg: eda_config,
    only_kinds: set[str] | None = None,
) -> list[Path]:
    """Render the plots in ``eda_cfg.enabled_plots`` to ``{root}/plots/eda/``.

    ``only_kinds`` is an OPTIONAL narrowing filter (the eda() facade passes the
    produced-kinds set). It MUST NOT be relied on for absence-safety:
    Bundle.eda(plots_only=True) passes ``only_kinds=None`` (render-all), so the
    absence gate below is CALLEE-side and both callers inherit it. Returns the
    list of emitted HTML paths. Unknown renderer-kind keys raise ValueError
    (fail-fast). ``root`` is the analysis_dir on the Analysis side and
    bundle.root on the Bundle side.
    """
    out: list[Path] = []
    for kind in eda_cfg.enabled_plots:
        if only_kinds is not None and kind not in only_kinds:
            continue
        renderer = _EDA_RENDERERS.get(kind)
        if renderer is None:
            raise ValueError(f"unknown EDA plot kind {kind!r}; known: {sorted(_EDA_RENDERERS)}")
        # Absence gate (bundle-safe, CALLEE-side). A calc member that skipped wrote no
        # backing zarr, so skip rather than open a non-existent store -- Bundle.eda(
        # plots_only=True) passes only_kinds=None (render-all) and would otherwise crash
        # on a grown enabled_plots default. Key on the kind's ACTUAL backing-artifact
        # stem, which is the kind itself for the eda_-prefixed sensitivity plots but
        # eda_cross_sim_identity for config_diff_maps (its calc member is
        # check_cross_sim_identity). LOUD ON SURPRISE: warn before skipping so a future
        # kind/stem drift surfaces instead of silently emptying the report (mirrors the
        # _harvest_and_copy_sources skip-with-warning precedent, Gotcha 50).
        backing_stem = _RENDERER_BACKING_ARTIFACT.get(kind, kind)
        artifact = root / "eda" / f"{backing_stem}.zarr"
        # The b4b kinds are ALWAYS enumerated by the b4b ReportingSet (workflow.py /
        # bundle snakefile_generator emit one report() target per template), and their
        # renderer emits an honest-degradation panel on absent/degraded data, so they must
        # NOT be absence-skipped here — a skip yields "marked for report but does not exist".
        # Non-b4b conditional members keep the skip (a skipped calc member wrote no zarr).
        if not artifact.exists() and kind not in _ALWAYS_EMIT_KINDS:
            warnings.warn(
                f"EDA plot kind {kind!r} is enabled but its backing artifact "
                f"{artifact} is absent -- skipping. If the calc member did NOT "
                f"legitimately skip, this is a kind/stem mismatch.",
                stacklevel=2,
            )
            continue
        out.append(renderer(root, cfg_analysis=cfg_analysis, eda_cfg=eda_cfg))
    return out


def _render_config_diff_maps(
    root: Path,
    *,
    cfg_analysis: analysis_config,
    eda_cfg: eda_config,
) -> Path:
    """Config-diff-maps EDA plot (redesign, /design-figure iteration 1).

    Reads the consolidated `{root}/sensitivity_datatree.zarr` directly (per-cell
    max_wlevel_m + per-conduit max_flow_cms + per-sub compute-config attrs) plus one
    sub's hydraulics.inp for conduit geometry, and renders: a cross-config
    identity+absolute-diff table, and per byte-identical config group the SIGNED diff
    and percent-diff maps (DEM cells + SWMM conduits) vs the serial-CPU baseline with
    serial reference maps. Compute-config labels are derived from config attrs (never
    the sa_id name). Emits under MASTER-ROOTED plots/eda/ as config_diff_maps.html.

    plot_id `config_diff_maps` (== on-disk stem, ADR-2). Existing bundles carrying the
    legacy `eda_cross_sim_identity` enabled_plots key are normalized to this kind by
    config/eda.py::_rewrite_legacy_eda_plot_kind (Bundle.eda() back-compat).
    """
    from hhemt.eda._config_diff import build_config_diff_figure, config_diff_source_paths

    plot_id = canonical_plot_id("config_diff_maps")
    fig = build_config_diff_figure(root)

    output_path = root / "plots" / "eda" / f"{plot_id}.html"
    html_text = _fig_to_html(fig, plotly_js_mode=eda_cfg.plotly_js_mode)
    return emit_plot_with_sources(
        html_text,
        output_path,
        source_paths=config_diff_source_paths(root),  # consolidated tree + one hydraulics.inp
        analysis_dir=root,
        output_format="html",
    )


def _render_rank_sensitivity(
    root: Path,
    *,
    cfg_analysis: analysis_config,
    eda_cfg: eda_config,
) -> Path:
    """Rank-sensitivity EDA plot (within-family rank-N vs rank-1). Figure body is a
    /eda-spinup seam; emits master-rooted plots/eda/eda_rank_sensitivity.html and declares
    BOTH eda/eda_rank_sensitivity.zarr AND its .verdict.json as sources."""
    from hhemt.eda._sensitivity_figures import (
        build_rank_sensitivity_figure,
        sensitivity_source_paths,
    )

    plot_id = canonical_plot_id("eda_rank_sensitivity")
    fig = build_rank_sensitivity_figure(root)
    output_path = root / "plots" / "eda" / f"{plot_id}.html"
    html_text = _fig_to_html(fig, plotly_js_mode=eda_cfg.plotly_js_mode)
    return emit_plot_with_sources(
        html_text,
        output_path,
        source_paths=sensitivity_source_paths(root, plot_id),  # {plot_id}.zarr + .verdict.json
        analysis_dir=root,
        output_format="html",
    )


def _render_resume_sensitivity(
    root: Path,
    *,
    cfg_analysis: analysis_config,
    eda_cfg: eda_config,
) -> Path:
    """Resume-sensitivity EDA plot (clean-vs-resume). Figure body is a /eda-spinup seam;
    emits master-rooted plots/eda/eda_resume_sensitivity.html and declares BOTH
    eda/eda_resume_sensitivity.zarr AND its .verdict.json as sources."""
    from hhemt.eda._sensitivity_figures import (
        build_resume_sensitivity_figure,
        sensitivity_source_paths,
    )

    plot_id = canonical_plot_id("eda_resume_sensitivity")
    fig = build_resume_sensitivity_figure(root)
    output_path = root / "plots" / "eda" / f"{plot_id}.html"
    html_text = _fig_to_html(fig, plotly_js_mode=eda_cfg.plotly_js_mode)
    return emit_plot_with_sources(
        html_text,
        output_path,
        source_paths=sensitivity_source_paths(root, plot_id),
        analysis_dir=root,
        output_format="html",
    )


def _render_cross_hardware_magnitude(
    root: Path,
    *,
    cfg_analysis: analysis_config,
    eda_cfg: eda_config,
) -> Path:
    """Cross-hardware-magnitude EDA plot (ADR-4 1-GPU vs 1-rank serial-CPU). Figure body
    is a /eda-spinup seam; emits master-rooted plots/eda/eda_cross_hardware_magnitude.html
    and declares BOTH eda/eda_cross_hardware_magnitude.zarr AND its .verdict.json."""
    from hhemt.eda._sensitivity_figures import (
        build_cross_hardware_magnitude_figure,
        sensitivity_source_paths,
    )

    plot_id = canonical_plot_id("eda_cross_hardware_magnitude")
    fig = build_cross_hardware_magnitude_figure(root)
    output_path = root / "plots" / "eda" / f"{plot_id}.html"
    html_text = _fig_to_html(fig, plotly_js_mode=eda_cfg.plotly_js_mode)
    return emit_plot_with_sources(
        html_text,
        output_path,
        source_paths=sensitivity_source_paths(root, plot_id),
        analysis_dir=root,
        output_format="html",
    )


def _render_b4b(
    root: Path,
    *,
    stem: str,
    title: str,
    eda_cfg: eda_config,
    show_boundaries: bool,
) -> Path:
    """Shared b4b render: reads eda/{stem}.zarr (written by
    raw_resume_identity.check_raw_b4b) and emits a heatmap y=compute_config, x=timestep_min,
    z=all-raw-types-identical (AND-reduced over raw_output_type), or an explicit R10
    honest-degradation panel when the artifact's `degraded` attr is set. Declares its own
    eda/{stem}.zarr as the source (raw dirs fail _validate_source_path; the producer owns
    raw-presence detection via the `degraded` attr, so this render reads only its own zarr
    and passes the Gotcha-53 renderer-IO audit)."""
    import xarray as xr

    plot_id = canonical_plot_id(stem)  # pass-through == stem
    artifact = root / "eda" / f"{plot_id}.zarr"
    output_path = root / "plots" / "eda" / f"{plot_id}.html"
    # Absent-artifact guard (Gate-6 honest-degradation). check_raw_b4b writes this zarr on
    # every analysis.eda() run, so an absent artifact means the b4b calc member did not run
    # before this render (a bundle emitted before eda(), or a bundle-carriage gap where the
    # figure was not harvested). Emit the honest-degradation placeholder rather than letting
    # xr.open_zarr raise / render_eda_plots skip, so an ENUMERATED b4b report target is never
    # "marked for report but does not exist".
    if not artifact.exists():
        ds = None
        degraded = True
        reason = "raw byte-for-byte backing artifact absent (the b4b calc member did not run before this render)"
        da = None
    else:
        ds = xr.open_zarr(artifact, consolidated=False)
        degraded = bool(int(ds.attrs.get("degraded", 0)))
        reason = str(ds.attrs.get("degraded_reason", ""))
        da = ds["identical"] if "identical" in ds else None
    if degraded or da is None or "compute_config" not in getattr(da, "dims", ()):
        fig = go.Figure()
        fig.add_annotation(
            text="Raw byte-for-byte comparison unavailable<br>" + (reason or "raw outputs were cleared"),
            showarrow=False,
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            font=dict(size=14),
        )
        fig.update_layout(title=title, height=260, xaxis=dict(visible=False), yaxis=dict(visible=False))
    else:
        z_all = da.min(dim="raw_output_type", skipna=True)  # 1 iff ALL raw types identical
        x = [float(t) for t in z_all["timestep_min"].values]
        y = [str(c) for c in z_all["compute_config"].values]
        z = [[None if v != v else int(v) for v in row] for row in z_all.values]  # NaN -> None
        fig = go.Figure(
            go.Heatmap(
                z=z,
                x=x,
                y=y,
                colorscale=[[0.0, "crimson"], [1.0, "seagreen"]],
                zmin=0,
                zmax=1,
                colorbar=dict(title="b4b (1=identical)"),
            )
        )
        if show_boundaries:
            for k, t in enumerate(ds.attrs.get("resume_boundaries_min", []) or [], start=1):
                fig.add_vline(
                    x=float(t),
                    line_dash="dash",
                    line_color="black",
                    annotation_text=f"r{k}",
                    annotation_position="top",
                )
        fig.update_layout(
            title=title,
            xaxis_title="reporting timestep (min)",
            yaxis_title="compute config",
            height=max(220, 60 + 26 * len(y)),
            margin=dict(l=140, r=90, t=50, b=40),
        )
    html_text = _fig_to_html(fig, plotly_js_mode=eda_cfg.plotly_js_mode)
    return emit_plot_with_sources(
        html_text,
        output_path,
        source_paths=[artifact],
        analysis_dir=root,
        output_format="html",
    )


def _render_b4b_clean_identity(root: Path, *, cfg_analysis: analysis_config, eda_cfg: eda_config) -> Path:
    """Clean-run raw byte-identity heatmap (b4b). Reads eda/b4b_clean_identity.zarr; R10
    honest-degradation panel when raw outputs were cleared."""
    return _render_b4b(
        root, stem="b4b_clean_identity", title="Clean-run raw byte identity", eda_cfg=eda_cfg, show_boundaries=False
    )


def _render_b4b_clean_vs_resume(root: Path, *, cfg_analysis: analysis_config, eda_cfg: eda_config) -> Path:
    """Clean-vs-resume raw byte-identity heatmap (b4b) with requested resume-boundary vlines.
    Reads eda/b4b_clean_vs_resume.zarr; R10 honest-degradation panel on `degraded`."""
    return _render_b4b(
        root,
        stem="b4b_clean_vs_resume",
        title="Clean vs resume raw byte identity",
        eda_cfg=eda_cfg,
        show_boundaries=True,
    )


def _fig_to_html(fig: go.Figure, *, plotly_js_mode: str) -> str:
    """Serialize one figure to an HTML fragment via the FQ1 single-bundle path.

    For a SINGLE figure the simple form is `pio.to_html(fig,
    include_plotlyjs=<True|cdn>, full_html=True)`. The MULTI-figure bundle-once
    composition lives in eda/_report.py::_figure_divs (Phase 2); this single-figure
    emit is what emit_plot_with_sources' HTML branch stores per plot.

    ``plotly_js_mode`` is the eda_config field value ('inline' | 'cdn'); map it to
    plotly's ``include_plotlyjs`` argument (which spells full-inline as ``True``,
    not the literal string 'inline').
    """
    import plotly.io as pio

    include_plotlyjs: bool | str = True if plotly_js_mode == "inline" else "cdn"
    return pio.to_html(fig, include_plotlyjs=include_plotlyjs, full_html=True)


#: renderer-kind -> renderer function. Keys are the eda_-PREFIXED plot-IDs the calc
#: members mint (compute_sensitivity.py:463,553,637); canonical_plot_id is a
#: pass-through, so those keys ARE the on-disk eda/{key}.zarr stems. Do not unprefix.
#: EDA kinds that are ALWAYS enumerated by their ReportingSet (the b4b family) and whose
#: renderer emits an honest-degradation panel on absent/degraded backing data. render_eda_plots
#: MUST NOT absence-skip these, or an enumerated report() target goes unproduced (WorkflowError).
_ALWAYS_EMIT_KINDS = frozenset({"b4b_clean_identity", "b4b_clean_vs_resume"})

_EDA_RENDERERS = {
    "config_diff_maps": _render_config_diff_maps,
    "eda_rank_sensitivity": _render_rank_sensitivity,
    "eda_resume_sensitivity": _render_resume_sensitivity,
    "eda_cross_hardware_magnitude": _render_cross_hardware_magnitude,
    "dem_resolution_cost_error": _render_dem_resolution_cost_error,
    "dem_resolution_diff_maps": _render_dem_resolution_diff_maps,
    "dem_resolution_error_ecdf": _render_dem_resolution_error_ecdf,
    "dem_resolution_coupling_table": _render_dem_resolution_coupling_table,
    "b4b_clean_identity": _render_b4b_clean_identity,
    "b4b_clean_vs_resume": _render_b4b_clean_vs_resume,
}

#: renderer-kind -> the on-disk eda/{stem}.zarr the kind's CALC member actually writes.
#: Absence-gate keys on THIS, not the kind, because the two diverge for config_diff_maps:
#: its calc member is check_cross_sim_identity, which mints
#: canonical_plot_id("eda_cross_sim_identity") -> eda/eda_cross_sim_identity.zarr (nothing
#: writes eda/config_diff_maps.zarr). The three sensitivity kinds are 1:1 (kind == stem),
#: so absent entries default to identity via .get(kind, kind); add an entry ONLY for a
#: renderer whose kind != its calc member's plot_id. Consistency note: analysis.eda()'s
#: _eda_results pairing (kind -> calc member, analysis.py) encodes the SAME config_diff_maps
#: exception from the facade side; keep the two in sync.
_RENDERER_BACKING_ARTIFACT = {
    "config_diff_maps": "eda_cross_sim_identity",
}
