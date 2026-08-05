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
    must_emit: set[str] | None = None,
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
        # ENUMERATE-IMPLIES-EMIT (the general rule; _ALWAYS_EMIT_KINDS is the b4b-specific
        # special case that predates it). `must_emit` carries the ENUMERATED report() targets
        # for THIS invocation -- the adapter passes the single rule's own output stem, which is
        # exactly what the ReportingSet enumerated. A kind the active set enumerated MUST be
        # emitted even with its backing artifact absent, or Snakemake raises
        # MissingOutputException ("marked for report but does not exist") and the whole workflow
        # dies. The renderer is then contractually responsible for an honest-degradation panel.
        # Render-all callers that carry no enumeration (Bundle.eda(plots_only=True),
        # eda/_notebook.py) pass must_emit=None and KEEP the skip-with-warning, so the warning
        # that surfaces genuine kind/stem drift is preserved exactly where it is still meaningful.
        if not artifact.exists() and kind in (must_emit or ()) and kind not in _ALWAYS_EMIT_KINDS:
            warnings.warn(
                f"EDA plot kind {kind!r} is an ENUMERATED report target but its backing "
                f"artifact {artifact} is absent -- rendering the degradation panel rather "
                f"than skipping (a skip would fail the workflow). If the calc member did "
                f"NOT legitimately skip, this is a kind/stem mismatch.",
                stacklevel=2,
            )
        elif not artifact.exists() and kind not in _ALWAYS_EMIT_KINDS:
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


#: Disclosure-regime physical band for the b4b max-abs-difference colorbar (data-viz round-3).
#: Ties to the disclosed dry-cell/nuisance floor τ (a-priori physical quantity, NEVER tuned to
#: the result); keep equal to compute_sensitivity._TAU_M. |Δ| below this reads ~white (honest
#: "no meaningful difference"); above it saturates honestly (proportional ink).
_B4B_TAU_M = 0.03
#: Okabe-Ito colorblind-safe pair (Moreland 2016): bluish-green = byte-identical (null state),
#: vermillion = magnitude ramp for differing cells. Redundant text/legend accompanies color.
_B4B_IDENTICAL_COLOR = "#009E73"
_B4B_DIFF_RAMP = [[0.0, "#FFFFFF"], [1.0, "#D55E00"]]


def _b4b_str_map(ds, name: str) -> dict:
    """compute_config (str) -> str value from a (compute_config,) DataArray; {} if absent."""
    if ds is None or name not in ds:
        return {}
    da = ds[name]
    return {str(c): str(v) for c, v in zip(da["compute_config"].values, da.values)}


def _b4b_bool_map(ds, name: str) -> dict:
    """compute_config (str) -> bool value from a (compute_config,) DataArray; {} if absent."""
    if ds is None or name not in ds:
        return {}
    da = ds[name]
    return {str(c): bool(v) for c, v in zip(da["compute_config"].values, da.values)}


def _b4b_category(fam: str) -> str:
    """Q4a (iter-2): collapse the per-hardware family token to a display CATEGORY — 'cpu' stays
    'cpu'; any GPU hardware token -> 'gpu'; the degraded single-panel sentinel 'all' -> 'all'.
    Panels group by this (one CPU panel, one GPU panel); each row keeps its OWN family's
    within-family reference math (max_abs_diff is precomputed per config in the zarr)."""
    if fam in ("cpu", "all"):
        return fam
    return "gpu"


def _b4b_category_title(cat: str) -> str:
    """Human panel title for a display CATEGORY: 'cpu' -> 'CPU'; 'gpu' -> 'GPU'; 'all' -> 'All configs'."""
    if cat == "cpu":
        return "CPU"
    if cat == "all":
        return "All configs"
    return "GPU"


def _b4b_family_title(fam: str) -> str:
    """Human panel title from the settled `family` token: 'cpu' -> 'CPU'; 'gpu' -> 'GPU'
    (N3: ONE GPU family); a legacy per-hardware token -> 'GPU {token}'; the degraded
    single-panel sentinel -> 'All configs'. The per-hardware branch is retained so a
    pre-N3 zarr still renders a sensible caption."""
    if fam == "cpu":
        return "CPU"
    if fam == "all":
        return "All configs"
    if fam == "gpu":
        return "GPU"
    return f"GPU {fam}"


def _b4b_faceted_figure(ds, da, *, title: str, baseline_caption: str, show_boundaries: bool) -> "go.Figure":
    """Disclosure-regime b4b figure: ONE subplot per hardware family (CPU + one per GPU
    hardware). Identical cells render solid Okabe-Ito bluish-green; differing cells are colored
    by max |Δ| vs the per-family baseline on a white->vermillion colorbar anchored to [0, τ].
    Reads the SETTLED per-compute_config DataArray contract directly: `config_label` (already
    _derive_config_label'd) as the y-label, `family` for panel grouping, `is_reference` to mark
    the baseline row, `max_abs_diff` (metres) for magnitude. Caption is built from the dataset
    attr `reference_config_by_family`. Degrades to a binary identical/differs encoding + raw
    labels + a single panel when max_abs_diff / config_label / family are absent. Positions and
    margins are nominal — smoke-render and nudge if needed."""
    import json

    import numpy as np
    from plotly.subplots import make_subplots

    # Q4b (iter-2): compare WATER LEVEL (wlevel_m = instantaneous H), NOT the AND of all four
    # raw types (which includes max_wlevel_m + velocities). Select the wlevel_m raw type when
    # present; fall back to the all-types AND only when wlevel_m is absent (older/partial data).
    _RAW_WLEVEL = "wlevel_m"
    if "raw_output_type" in getattr(da, "dims", ()) and _RAW_WLEVEL in [str(v) for v in da["raw_output_type"].values]:
        z_all = da.sel(raw_output_type=_RAW_WLEVEL)
    else:
        z_all = da.min(dim="raw_output_type", skipna=True)  # fallback: 1 iff ALL raw types identical
    x = [float(t) for t in z_all["timestep_min"].values]
    configs = [str(c) for c in z_all["compute_config"].values]
    identical = np.asarray(z_all.transpose("compute_config", "timestep_min").values, dtype=float)

    mad = ds["max_abs_diff"] if (ds is not None and "max_abs_diff" in ds) else None
    if mad is not None and "raw_output_type" in getattr(mad, "dims", ()):
        # Q4b: magnitude for the SAME variable the identity uses (wlevel_m), not the max over
        # all raw types. Fall back to the all-types max only when wlevel_m is absent.
        if _RAW_WLEVEL in [str(v) for v in mad["raw_output_type"].values]:
            mad = mad.sel(raw_output_type=_RAW_WLEVEL)
        else:
            mad = mad.max(dim="raw_output_type", skipna=True)
    mad_vals = np.asarray(mad.transpose("compute_config", "timestep_min").values, dtype=float) if mad is not None else None

    label_by_cfg = _b4b_str_map(ds, "config_label")   # already deterministic (F2)
    family_by_cfg = _b4b_str_map(ds, "family")        # "cpu" / "a6000" / "a100-80"
    is_ref_by_cfg = _b4b_bool_map(ds, "is_reference")

    def _label(cfg: str) -> str:
        base = label_by_cfg.get(cfg, cfg)
        return f"{base}  ★ ref" if is_ref_by_cfg.get(cfg) else base

    def _family(cfg: str) -> str:
        return family_by_cfg.get(cfg, "all")

    fam_to_rows: dict[str, list[int]] = {}
    for i, cfg in enumerate(configs):
        fam_to_rows.setdefault(_b4b_category(_family(cfg)), []).append(i)  # Q4a: one CPU + one GPU panel
    families = sorted(fam_to_rows, key=lambda f: (f != "cpu", f))  # CPU panel first
    n_panels = len(families)
    row_heights = [max(2, len(fam_to_rows[f]) ) for f in families]  # Q2: floor a 1-config panel

    fig = make_subplots(
        rows=n_panels, cols=1, shared_xaxes=True, row_heights=row_heights,
        subplot_titles=[_b4b_category_title(f) for f in families],
        vertical_spacing=min(0.08, 0.3 / max(n_panels, 1)),
    )
    shown_colorbar = False
    for r, fam in enumerate(families, start=1):
        rows = fam_to_rows[fam]
        y = [_label(configs[i]) for i in rows]
        ident = identical[rows, :]  # (n_rows, n_t)
        z_ident = [[1 if v == 1 else None for v in ident[a]] for a in range(len(rows))]
        fig.add_trace(
            go.Heatmap(
                z=z_ident, x=x, y=y,
                colorscale=[[0.0, _B4B_IDENTICAL_COLOR], [1.0, _B4B_IDENTICAL_COLOR]],
                showscale=False, xgap=1, ygap=1,
                hovertemplate="%{y}<br>t=%{x} min<br>byte-identical<extra></extra>",
            ),
            row=r, col=1,
        )
        if mad_vals is not None:
            mrows = mad_vals[rows, :]
            z_diff = [
                [None if (ident[a, b] == 1 or mrows[a, b] != mrows[a, b]) else float(mrows[a, b]) for b in range(len(x))]
                for a in range(len(rows))
            ]
            fig.add_trace(
                go.Heatmap(
                    z=z_diff, x=x, y=y, colorscale=_B4B_DIFF_RAMP, zmin=0.0, zmax=_B4B_TAU_M,
                    showscale=not shown_colorbar,
                    colorbar=dict(
                        title=f"max |Δ| vs reference (m)<br>scale 0–τ ({_B4B_TAU_M} m)",
                        len=0.5, thickness=14, x=1.02, xanchor="left",
                        y=0.85, yanchor="top",
                    ),
                    xgap=1, ygap=1,
                    hovertemplate="%{y}<br>t=%{x} min<br>max |Δ|=%{z:.3g} m<extra></extra>",
                ),
                row=r, col=1,
            )
            shown_colorbar = True
        else:
            z_diff = [[1 if ident[a, b] == 0 else None for b in range(len(x))] for a in range(len(rows))]
            fig.add_trace(
                go.Heatmap(
                    z=z_diff, x=x, y=y, colorscale=[[0.0, "#D55E00"], [1.0, "#D55E00"]],
                    showscale=False, xgap=1, ygap=1,
                    hovertemplate="%{y}<br>t=%{x} min<br>differs<extra></extra>",
                ),
                row=r, col=1,
            )
        if show_boundaries and ds is not None:
            for k, t in enumerate(ds.attrs.get("resume_boundaries_min", []) or [], start=1):
                fig.add_vline(
                    x=float(t), line_dash="dash", line_color="black",
                    annotation_text=f"r{k}", annotation_position="top", row=r, col=1,
                )

    # Legend proxy for the identical color (Heatmap emits no legend entry) — redundant channel.
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=12, color=_B4B_IDENTICAL_COLOR, symbol="square"),
            name="byte-identical (Δ = 0)", showlegend=True,
        ),
        row=1, col=1,
    )

    # Caption: generic prefix + the deterministic per-family reference list (Δ = 0 by definition).
    ref_map = ds.attrs.get("reference_config_by_family", "{}") if ds is not None else "{}"
    if isinstance(ref_map, str):
        try:
            ref_map = json.loads(ref_map)
        except Exception:
            ref_map = {}
    ref_bits = "; ".join(f"{_b4b_family_title(k)} → {v}" for k, v in sorted((ref_map or {}).items()))
    caption = baseline_caption
    if ref_bits:
        caption = f"{baseline_caption} Reference (Δ = 0 by definition) per family: {ref_bits}."

    # N4 / Gate-0 count-don't-eyeball: the collapsed rows hide a denominator, so state it.
    # Guarded on `in ds` because a legacy (pre-N4) zarr carries no n_replicates variable and
    # must render its caption unchanged.
    if ds is not None and "n_replicates" in ds:
        _nr = sorted({int(v) for v in ds["n_replicates"].values})
        if _nr and max(_nr) > 1:
            _rng = str(_nr[0]) if len(_nr) == 1 else f"{min(_nr)}–{max(_nr)}"
            caption += (
                f" Each row aggregates {_rng} replicate run(s) of that compute config,"
                " worst-case: a cell is green only if EVERY replicate was byte-identical,"
                " and |Δ| is the maximum over replicates."
            )

    total_rows = sum(row_heights)
    fig.update_layout(
        title=title,
        height=max(340, 90 + 26 * total_rows + 46 * n_panels),
        margin=dict(l=180, r=140, t=64, b=150),
        legend=dict(orientation="v", yanchor="top", y=1.0, x=1.02, xanchor="left"),
    )
    fig.update_xaxes(title_text="reporting timestep (min)", row=n_panels, col=1)
    fig.add_annotation(
        text=caption, showarrow=False, xref="paper", yref="paper",
        x=0.0, y=-0.14, yanchor="top", align="left", font=dict(size=11),
        width=400,
    )
    return fig


def _render_b4b(
    root: Path,
    *,
    stem: str,
    title: str,
    baseline_caption: str,
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
        import textwrap

        fig = go.Figure()
        fig.add_annotation(
            text="Raw byte-for-byte comparison unavailable<br>"
            + "<br>".join(textwrap.wrap(reason or "raw outputs were cleared", width=90)),
            showarrow=False,
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            font=dict(size=14),
        )
        fig.update_layout(title=title, height=260, xaxis=dict(visible=False), yaxis=dict(visible=False))
    else:
        fig = _b4b_faceted_figure(
            ds, da, title=title, baseline_caption=baseline_caption, show_boundaries=show_boundaries
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
    """Cross-hardware results comparison (b4b), reused per-family over each master's
    own subs (serves both the clean and resume masters). Reads eda/b4b_clean_identity.zarr;
    R10 honest-degradation panel when raw outputs were cleared."""
    return _render_b4b(
        root,
        stem="b4b_clean_identity",
        title=(
            "Per-timestep water-level byte-for-byte comparison vs the hardware-category reference"
        ),
        baseline_caption=(
            "Alternate configurations are compared against their hardware category's "
            "minimum-device reference (CPU → serial-CPU, GPU → 1-GPU). "
            "Identical cells (Δ = 0) are green; differing cells show max |Δ| vs the per-family "
            "reference, colorbar band 0–τ (τ = 0.03 m)."
        ),
        eda_cfg=eda_cfg,
        show_boundaries=False,
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
_ALWAYS_EMIT_KINDS = frozenset({"b4b_clean_identity"})

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
