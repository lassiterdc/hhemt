"""Sensitivity benchmarking renderer.

Dual-panel figure (left: Wall-clock hours, right: Compute-hours = wallclock × n_devices)
with a shared x-axis given by ``independent_var`` (typical: ``n_devices``). One
line+marker series per ``group_by_var`` value (typical: ``run_mode``).

Special line-drawing rules per the user-locked Phase 6 iter-2 spec:

- ``hybrid`` (or any group with multiple points sharing the same x-value): the line
  passes through the **minimum** y-value at each x; remaining points are still drawn
  as markers and (for hybrid) annotated with their ``n_mpi_procs`` value to highlight
  the most computationally efficient configuration when several share the same
  resource budget.
- ``serial`` / single-CPU group (always one point on the curve): rendered as a single
  larger distinguished marker, no connecting line.
- GPU runs (``n_gpus > 0``): distinct marker shape from CPU runs.
- All non-hybrid lines: dashed, thin.

DataTree-aware read pattern: ``performance.Total`` lives at
``/sa_{id}/tritonswmm/performance`` in the master ``sensitivity_datatree.zarr``,
dimensioned by ``event_iloc``. SWMM-only sub-analyses fall back to per-scenario
``.rpt`` parsing via :func:`hhemt.swmm_output_parser.parse_total_elapsed`.

Derived columns: when ``independent_var`` is ``n_devices`` and the column is absent
from the sensitivity CSV, the renderer computes it as
``n_gpus if run_mode == "gpu" else n_mpi_procs * n_omp_threads * n_nodes``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import xarray as xr
from plotly.subplots import make_subplots


# Q1 fungibility (iter-2): color must be STABLE per group_value across arms. A
# pure-TRITON arm and a TRITON-SWMM arm that differ in which groups are present
# would otherwise get shifted palette indices (color = palette[i % len]) and the
# same hardware family would read as two colors across the co-located figures.
# Canonical order pins the known families to fixed slots; unknowns append
# deterministically (sorted) after the known block.
#: Recognized group SPELLINGS. NOT the palette index source -- `_stable_group_color`
#: indexes `_CANONICAL_FAMILIES` below, because this tuple's alias slots shift every
#: family two positions right. Do not reintroduce it as an ordering.
_CANONICAL_GROUP_ORDER = ("serial", "single_cpu", "single-cpu", "cpu", "gpu", "hybrid")


#: Hardware FAMILIES, de-aliased. `_CANONICAL_GROUP_ORDER` spent six slots on four
#: concepts (`single_cpu`/`single-cpu` are spellings of `serial`), so indexing against
#: it directly pushes every family two positions right and lands gpu on #56B4E9 beside
#: serial's #0072B2 -- two blues in the figure whose whole job is CPU-vs-GPU comparison.
_CANONICAL_FAMILIES = ("serial", "cpu", "gpu", "hybrid")
_FAMILY_ALIASES = {"single_cpu": "serial", "single-cpu": "serial"}


def _stable_group_color(group_value, palette, all_groups=None):
    """Palette colour for ``group_value``, INDEPENDENT of which groups are present.

    The previous implementation filtered the canonical order by `all_groups`, which
    reintroduced exactly the set-sensitivity the header comment says this function
    exists to remove: a panel filtered to one hardware family resolved that family to
    index 0, so every singleton group rendered #0072B2 while the legend -- built from
    the unfiltered frame -- showed its true slot. Measured against the shipped
    Okabe-Ito palette, gpu read blue in the panel and green in the legend.

    `all_groups` is retained and IGNORED for signature compatibility with the four
    call sites; unknown groups still sort deterministically after the known block.
    """
    gv = _FAMILY_ALIASES.get(str(group_value), str(group_value))
    if gv in _CANONICAL_FAMILIES:
        idx = _CANONICAL_FAMILIES.index(gv)
    else:
        extras = sorted(
            {
                _FAMILY_ALIASES.get(str(x), str(x))
                for x in (all_groups or [])
            }
            - set(_CANONICAL_FAMILIES)
        )
        idx = len(_CANONICAL_FAMILIES) + (extras.index(gv) if gv in extras else 0)
    return palette[idx % len(palette)]

from hhemt.figure_caption import add_figure_caption, content_width_px
from hhemt.report_renderers._figure_emission import emit_plot_with_sources
from hhemt.report_renderers._provenance import ProvenanceLog, ProvenanceRef
from hhemt.swmm_output_parser import parse_total_elapsed

if TYPE_CHECKING:
    from hhemt.analysis import TRITONSWMM_analysis
    from hhemt.config.report import report_config
    from hhemt.config.static_plots import StaticPlotBaseConfig


@dataclass
class FacetConfig:
    """Configuration for multi-facet sensitivity-benchmarking layouts.

    Architectural scaffold (F4 of the kickoff figure-review) for future experiments
    that compare benchmark metrics across an additional categorical axis — typically
    DEM resolution (e.g., 1m vs 3.5m vs 10m) or GPU hardware (a6000 vs a100 vs h100).

    When ``facet=None`` (current default) the renderer emits the canonical
    ``rows=4, cols=1, shared_xaxes=True`` layout. When ``facet`` is provided, the
    renderer arranges the 4 metric panels per facet value across the grid shape
    declared by ``cols`` × ``rows`` (rows is implicit: ``len(facet_values) // cols``,
    rounded up — and the panel-row count multiplies by 4).

    Today's wiring: declared but not yet consumed by ``_render_plotly_branch``'s
    grid construction. The wiring lands when a user-side experiment requests it;
    the kwarg presence is the architectural breadcrumb the user requested.
    """

    facet_var: str = ""
    facet_values: list[Any] = field(default_factory=list)
    cols: int = 2
    label_format: str = "{facet_var}={facet_value}"


# Module-level styling constants moved to `report_cfg.sensitivity` per the
# config-driven refactor (see plan: full sweep — eliminate hardcoded params).
# Per-call: `sens_cfg = report_cfg.sensitivity` and read `sens_cfg.cpu_marker`,
# `sens_cfg.gpu_marker`, `sens_cfg.point_size`, `sens_cfg.line_style`,
# `sens_cfg.line_width`, `sens_cfg.palette`, `sens_cfg.independent_var_labels`.


def render(
    analysis: TRITONSWMM_analysis,
    report_cfg: report_config,
    output_path: Path,
    *,
    independent_var: str,
    static_cfg: StaticPlotBaseConfig | None = None,
    **kwargs,
) -> Path:
    """Render the dual-panel benchmarking figure for one independent variable.

    When ``static_cfg`` is provided (publication static-plots path, ADR-8) the
    matplotlib branch is FORCED (publication is matplotlib-only per ADR-3) and the
    figure geometry, the categorical series palette, the cpu/gpu markers, optional
    log-y, base typography, and emit format are driven by the
    SensitivityBenchmarkingStaticConfig rather than report_cfg. The Plotly branch
    is the interactive-report renderer and is NOT promoted to publication
    (KEEP-no-hybrid invariant). ``static_cfg=None`` (the report path) is
    byte-unchanged. ``**kwargs`` tolerates dispatcher-passed keywords this
    renderer does not consume.
    """
    from hhemt.report_renderers.system_overview import _apply_rcparams

    _apply_rcparams(report_cfg)

    if report_cfg.sensitivity is None:
        raise ValueError("report_cfg.sensitivity must be set for benchmarking rendering")

    sensitivity = analysis.sensitivity
    df_setup = sensitivity.df_setup_with_system_overlays.copy()

    df_setup = _ensure_n_devices_column(df_setup, independent_var)

    if independent_var not in df_setup.columns:
        raise ValueError(
            f"{independent_var!r} is not a resolvable benchmarking axis; "
            f"resolvable columns: {sorted(df_setup.columns)}"
        )

    dependent_var = report_cfg.sensitivity.dependent_var
    group_by_var = report_cfg.sensitivity.group_by_var

    rows, source_paths = _collect_rows(analysis, dependent_var)
    if not rows:
        raise RuntimeError(
            f"No data for benchmarking {independent_var} vs {dependent_var}"
        )

    df = pd.DataFrame(rows)
    # Wallclock-safe column allowlist (V0008+): only barrier-synchronized
    # cumulative columns can be interpreted as wallclock. Other performance.*
    # columns are per-category cost, not wallclock; raise rather than silently
    # mislabel.
    _WALLCLOCK_SAFE_COLS = {
        "performance.Total",
        "performance.Simulation",
        "performance.Init",
    }
    if dependent_var not in _WALLCLOCK_SAFE_COLS:
        raise ValueError(
            f"dependent_var {dependent_var!r} is not wallclock-safe. "
            f"Choose one of: {sorted(_WALLCLOCK_SAFE_COLS)}. Other performance.* "
            "columns are per-category cost and cannot be plotted as wallclock. "
            "See library/docs/stipulations/hhemt/wallclock reduction uses max over rank.md "
            "for the project rule on this."
        )
    df["wallclock_s"] = df["value"]
    df["wallclock_hr"] = df["wallclock_s"] / 3600.0
    df["indep_value"] = df["sa_id"].map(df_setup[independent_var])
    df["n_devices"] = df["sa_id"].map(df_setup["n_devices"])
    df["compute_hr"] = df["wallclock_hr"] * df["n_devices"]
    if group_by_var is not None:
        if group_by_var not in df_setup.columns:
            raise ValueError(
                f"group_by_var {group_by_var!r} is not a resolvable benchmarking axis; "
                f"resolvable columns: {sorted(df_setup.columns)}"
            )
        df["group_value"] = df["sa_id"].map(df_setup[group_by_var])
        # Iteration 4 (FQ2): replicate identity. The trailing `_rN` token is the
        # replicate marker and is
        # deliberately NOT part of config identity (_config_diff._derive_config_label:
        # "Replicate suffixes are NOT in the identity, so replicates share one label").
        # n_replicates drives marker FILL (open = this config has repeated runs) and the
        # line aggregation averages over replicates before the per-N min is taken, so the
        # min rule selects the best distinct CONFIG rather than the luckiest RUN.
        df["config_id"] = df["sa_id"].astype(str).str.replace(r"_r\d+$", "", regex=True)
        df["n_replicates"] = df["config_id"].map(
            df.groupby("config_id")["sa_id"].nunique()
        ).astype(int)
        # Iteration 4 (FQ3 + FQ9b): qualify GPU group names with their hardware token when
        # -- and only when -- more than one token is present. Unqualified, `gpu-a6000` and
        # `gpu-a100-80` collapse into ONE series, and because the line is drawn through
        # groupby(indep_value).min() its vertices are a100-80 at every device count while the
        # a6000 data contributes none. Splitting the key carries hardware on colour + legend
        # (strictly better than a fourth glyph channel, which cannot fix the line selection)
        # and gives _resolve_family_baselines a per-hardware family to anchor on.
        _part_col = _resolve_setup_col(df_setup, "hpc.partition")
        if _part_col is not None:
            _part = df["sa_id"].map(df_setup[_part_col]).astype(str)
            _is_gpu = df["group_value"].astype(str).str.lower().eq("gpu") & _part.str.startswith("gpu-")
            _tokens = _part[_is_gpu].str.replace(r"^gpu-", "", regex=True)
            if _tokens.nunique() > 1:
                df.loc[_is_gpu, "group_value"] = "gpu (" + _tokens + ")"
    else:
        df["group_value"] = "all"
    df["n_mpi_procs"] = df["sa_id"].map(
        df_setup[_resolve_setup_col(df_setup, "n_mpi_procs") or "n_mpi_procs"]
    )
    # F2: extra config columns for hover customdata (OMP threads, GPUs, Nodes).
    # Use .get() semantics so missing columns degrade gracefully — hovertemplate
    # only includes labels for columns that map successfully. Each is resolved
    # bare-or-`analysis.`-prefixed, then stored under its BARE key in df.
    for col in ("n_omp_threads", "n_gpus", "n_nodes"):
        resolved_col = _resolve_setup_col(df_setup, col)
        if resolved_col is not None:
            df[col] = df["sa_id"].map(df_setup[resolved_col])

    wall_unit, wall_factor = _adaptive_time_unit(df["wallclock_hr"].max())
    cost_unit, cost_factor = _adaptive_time_unit(df["compute_hr"].max())
    df["wallclock_disp"] = df["wallclock_hr"] * wall_factor
    df["compute_disp"] = df["compute_hr"] * cost_factor

    prov = ProvenanceLog()
    sens_cfg = report_cfg.sensitivity

    static_backend = getattr(
        getattr(report_cfg, "interactive", None),
        "static_backend",
        "plotly",
    )
    # Publication path (ADR-3): static_cfg FORCES the matplotlib branch regardless
    # of static_backend. The Plotly branch stays the interactive-report renderer and
    # is never promoted to publication (KEEP-no-hybrid invariant).
    use_plotly = False if static_cfg is not None else (static_backend == "plotly")
    if use_plotly:
        # Panels 3+4 carry TWO series per group and they answer different questions.
        # The LINE is the per-N envelope over replicate-AVERAGED configs, so a vertex
        # is the best distinct CONFIG at that device count rather than the luckiest RUN.
        # The MARKERS are every raw row, so the replicate spread stays visible. Before
        # this change the "line" was itself an all-rows series (measured: panel-3 line
        # x = [1,1,2,2,3,3], not [1,2,3]), so the min rule the comment claimed was never
        # applied on these two panels and the two series were the same points twice.
        #
        # Both series anchor on the SAME per-family baseline. They share an axis, a
        # colour and a legendgroup, which asserts to the reader that they are one
        # series; a global-serial anchor against a per-family anchor made that
        # assertion false for every GPU family and made the y-axis title's t_family(1)
        # true of the line only.
        #
        # THE ANCHOR IS RESOLVED FROM THE RAW FRAME. This reverses a deliberate earlier
        # choice, and the earlier reasoning is preserved because it is still true:
        # resolving from the raw frame divides averaged line values by an un-averaged
        # minimum, so the reference config no longer lands at EXACTLY 1.0 under its own
        # line -- it lands slightly below.
        #
        # That was traded away because of what the averaged anchor cost. _resolve_family_
        # baselines takes float(ref[t_col].min()) over the min-N rows of whatever frame it
        # is handed, so an AVERAGED frame yields `min over configs of (mean over
        # replicates)`. The line is then pinned to anchor/anchor = 1.0000 at min-N, while a
        # marker is anchor/T_i over RAW rows -- and a single replicate that beat the mean
        # including its slower sibling plots ABOVE 1.0. The line sat outside the spread of
        # the points it summarises. MEASURED on the e581fffb0b1c generation:
        # synth_cc_resume_triton panel x4/y4, gpu (a100-80), line 1.0000 vs BOTH markers at
        # 1.2586; across all four masters, 18 of 208 line-vs-marker points deviate >1%,
        # worst 25.85%.
        #
        # A raw anchor is <= every raw row at min-N by construction, so no marker can plot
        # above its own family's reference. A line beginning marginally under 1.0 says "the
        # averaged configuration is slower than the single best run", which is true and
        # readable; a line beneath which its own points float says something false about the
        # relationship between the two series. BM-2 ("use averages for drawing all lines")
        # is unaffected -- the LINE is still built from _df_avg; only the shared scalar it
        # divides by changes.
        #
        # This scalar feeds ALL FOUR metric calls below -- speedup line, speedup markers,
        # efficiency line, efficiency markers -- so this one argument moves BOTH panels.
        _df_avg = (
            df.groupby(["group_value", "n_devices", "config_id"], as_index=False)
            .agg(wallclock_s=("wallclock_s", "mean"), sa_id=("sa_id", "first"))
        )
        family_baselines = _resolve_family_baselines(
            df, t_col="wallclock_s", indep_col="n_devices", group_col="group_value",
        )
        if family_baselines:
            speedup_pg, strong_eff_pg = {}, {}
            speedup_all, efficiency_all = {}, {}
            for _gv, _anchor in family_baselines.items():
                _sub_avg = _df_avg[_df_avg["group_value"].astype(str) == _gv]
                _sub_raw = df[df["group_value"].astype(str) == _gv]
                if _sub_avg.empty:
                    continue
                # Line: per-N minimum over the replicate-averaged configs.
                _line_sub = _sub_avg.loc[
                    _sub_avg.groupby("n_devices")["wallclock_s"].idxmin()
                ]
                speedup_pg.update(_compute_metric_all_rows_per_group(
                    _line_sub, t_col="wallclock_s", indep_col="n_devices",
                    group_col="group_value", kind="speedup", anchor=_anchor,
                ))
                strong_eff_pg.update(_compute_metric_all_rows_per_group(
                    _line_sub, t_col="wallclock_s", indep_col="n_devices",
                    group_col="group_value", kind="efficiency", anchor=_anchor,
                ))
                # Markers: every raw row, same anchor.
                if _sub_raw.empty:
                    continue
                speedup_all.update(_compute_metric_all_rows_per_group(
                    _sub_raw, t_col="wallclock_s", indep_col="n_devices",
                    group_col="group_value", kind="speedup", anchor=_anchor,
                ))
                efficiency_all.update(_compute_metric_all_rows_per_group(
                    _sub_raw, t_col="wallclock_s", indep_col="n_devices",
                    group_col="group_value", kind="efficiency", anchor=_anchor,
                ))
            speedup_all = speedup_all or None
            efficiency_all = efficiency_all or None
        else:
            speedup_pg = _compute_speedup_per_group(
                df, t_col="wallclock_s", indep_col="n_devices",
                group_col="group_value", baseline_mode="serial",
            )
            strong_eff_pg = _compute_efficiency_per_group(
                df, t_col="wallclock_s", indep_col="n_devices",
                group_col="group_value", mode="strong", baseline_mode="serial",
            )
            serial_anchor = _resolve_serial_baseline(
                df, t_col="wallclock_s", group_col="group_value",
            )
            if serial_anchor is not None:
                speedup_all = _compute_metric_all_rows_per_group(
                    df, t_col="wallclock_s", indep_col="n_devices",
                    group_col="group_value", kind="speedup", anchor=serial_anchor,
                )
                efficiency_all = _compute_metric_all_rows_per_group(
                    df, t_col="wallclock_s", indep_col="n_devices",
                    group_col="group_value", kind="efficiency", anchor=serial_anchor,
                )
            else:
                speedup_all = None
                efficiency_all = None
        if analysis.cfg_analysis.sensitivity_analysis is not None:
            source_paths.append(Path(analysis.cfg_analysis.sensitivity_analysis))
        # F1: GPU hardware suffix (e.g., "gpu (a6000)"). Phase-4 (4c, D3): gpu_hardware
        # was retired off system_config to the partition axis; resolve it from each
        # sub-analysis's resolved partition's PartitionSpec.
        from hhemt.config.hpc_system import resolve_gpu_target

        # Phase 6 (DQ7c): under per-row partition, derive the distinct GPU hardware
        # across sub-analyses. Single-hardware experiments keep the master suffix
        # (byte-identical); multi-hardware experiments suppress the global suffix so
        # the per-group hardware is carried by the group label instead.
        _hw_values = set()
        _sens = getattr(analysis, "sensitivity", None)
        if _sens is not None:
            for _sub in _sens.sub_analyses.values():
                _hw = resolve_gpu_target(
                    _sub.cfg_hpc_system, _sub.cfg_analysis.hpc_ensemble_partition
                )[0]
                if _hw:
                    _hw_values.add(_hw)
        if len(_hw_values) == 1:
            gpu_hw = next(iter(_hw_values))
        elif len(_hw_values) == 0:
            gpu_hw = resolve_gpu_target(
                analysis.cfg_hpc_system, analysis.cfg_analysis.hpc_ensemble_partition
            )[0]
        else:
            gpu_hw = None  # multi-hardware: suffix carried per-group, not globally
        gpu_legend_suffix = f" ({gpu_hw})" if gpu_hw else ""
        # F-FU-6 / Q1: speedup panel range mode. Read from report_cfg if present,
        # default to `full_ideal`. Surface via kwarg for caller override (e.g.,
        # render-twice comparison during /design-figure iteration).
        speedup_range_mode = getattr(
            getattr(report_cfg, "sensitivity", None), "speedup_panel_range_mode", "full_ideal",
        )
        _model_arm = _resolve_model_arm(analysis)
        return _render_plotly_branch(
            df, speedup_pg, strong_eff_pg,
            model_arm=_model_arm,
            wall_unit=wall_unit, cost_unit=cost_unit,
            independent_var=independent_var, group_by_var=group_by_var,
            sens_cfg=sens_cfg,
            output_path=output_path, source_paths=source_paths,
            analysis_dir=analysis.analysis_paths.analysis_dir,
            plotly_js_mode=report_cfg.interactive.plotly_js_mode,
            prov=prov,
            gpu_legend_suffix=gpu_legend_suffix,
            speedup_all_rows=speedup_all,
            efficiency_all_rows=efficiency_all,
            speedup_range_mode=speedup_range_mode,
        )

    # Publication base typography (parity with the other static-plot renderers);
    # full per-element FontTarget threading is the deferred _core extraction.
    if static_cfg is not None:
        from hhemt.config.viz_vocabulary import FontTarget

        plt.rcParams["font.family"] = static_cfg.font_family
        plt.rcParams["font.size"] = static_cfg.font_sizes[FontTarget.axis_label]

    # Publication exact dimensions (data-viz OE-1) vs the report figsize.
    _figsize = (
        (static_cfg.figure_width_inches, static_cfg.figure_height_inches)
        if static_cfg is not None
        else tuple(sens_cfg.figsize_inches)
    )
    fig, (ax_wall, ax_cost, ax_speedup, ax_eff) = plt.subplots(
        4, 1, figsize=_figsize, sharex=True
    )
    _draw_panel(
        ax_wall, df, y_col="wallclock_disp", group_by_var=group_by_var,
        sens_cfg=sens_cfg, prov=prov, static_cfg=static_cfg,
    )
    _draw_panel(
        ax_cost, df, y_col="compute_disp", group_by_var=group_by_var,
        sens_cfg=sens_cfg, prov=prov, static_cfg=static_cfg,
    )
    # Optional publication log-y on the magnitude panels (wall-clock / compute-cost);
    # the speedup/efficiency panels are ratios and stay linear.
    if static_cfg is not None and static_cfg.log_y:
        ax_wall.set_yscale("log")
        ax_cost.set_yscale("log")

    speedup_per_group = _compute_speedup_per_group(
        df, t_col="wallclock_s", indep_col="n_devices", group_col="group_value",
        baseline_mode="global",
    )
    strong_eff_per_group = _compute_efficiency_per_group(
        df, t_col="wallclock_s", indep_col="n_devices", group_col="group_value", mode="strong",
        baseline_mode="global",
    )
    _draw_metric_panel(
        ax_speedup, speedup_per_group, df=df,
        x_max=df["n_devices"].max(),
        ideal_kind="linear", ideal_label="Ideal speedup (S=N)",
        sens_cfg=sens_cfg, prov=prov, static_cfg=static_cfg,
    )
    _draw_metric_panel(
        ax_eff, strong_eff_per_group, df=df,
        x_max=df["n_devices"].max(),
        ideal_kind="constant", ideal_value=1.0, ideal_label="Ideal efficiency (=1.0)",
        sens_cfg=sens_cfg, prov=prov, static_cfg=static_cfg,
    )

    xlabel_text = sens_cfg.independent_var_labels.get(independent_var, independent_var)
    ax_eff.set_xlabel(xlabel_text)  # bottom panel only under sharex=True
    ax_wall.set_ylabel(f"Wall-clock time ({wall_unit})")
    ax_cost.set_ylabel(f"Compute cost ({cost_unit} × devices)")
    ax_speedup.set_ylabel("Strong-Scaling Speedup\n" + r"$S(N) = t(1)\,/\,t(N)$")
    ax_eff.set_ylabel("Strong-Scaling Efficiency\n" + r"$E_s(N) = t(1)\,/\,(N \cdot t(N))$")
    for ax in (ax_wall, ax_cost, ax_speedup, ax_eff):
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        if report_cfg.sensitivity.show_gridlines:
            ax.grid(
                True, which="major", axis="both",
                color=sens_cfg.gridline_color,
                linewidth=sens_cfg.gridline_width,
                zorder=0,
            )

    if group_by_var is not None:
        # Asterisk after groups that get per-point n_mpi_procs annotations
        # (currently: hybrid). Connects the legend entry to the bottom-panel footnote.
        handles, labels = ax_wall.get_legend_handles_labels()
        starred = [f"{lab}*" if lab.lower() == "hybrid" else lab for lab in labels]
        ax_wall.legend(handles, starred, title=group_by_var, loc="upper right")

    # Title placed via ax_wall.set_title so it's anchored to the top panel's data
    # area (truly plot-centered horizontally, not figure-centered) and matplotlib
    # auto-reserves space for it. pad=4 keeps it close to the panel edge.
    ax_wall.set_title(
        sens_cfg.title,
        fontsize=sens_cfg.title_fontsize,
        pad=sens_cfg.title_pad,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 1.0])
    # Footnote uses axes-fraction coords on the bottom panel so it's truly centered
    # on the plot area (fig.text x=0.5 is figure-center, not plot-center, because
    # the left y-axis labels offset the plot area rightward of figure-center).
    ax_eff.text(
        0.5, -0.18,
        sens_cfg.footnote_text,
        transform=ax_eff.transAxes,
        ha="center", va="top", fontsize=sens_cfg.footnote_fontsize, style="italic",
    )

    if analysis.cfg_analysis.sensitivity_analysis is not None:
        source_paths.append(Path(analysis.cfg_analysis.sensitivity_analysis))

    return emit_plot_with_sources(
        fig,
        output_path,
        source_paths,
        analysis_dir=analysis.analysis_paths.analysis_dir,
        dpi=(static_cfg.savefig_dpi if static_cfg is not None else report_cfg.figure_defaults.savefig_dpi),
        output_format=(static_cfg.output_format if static_cfg is not None else "svg"),
        bbox_inches_tight=(static_cfg.bbox_inches_tight if static_cfg is not None else True),
        emit_preview=(static_cfg is None),
        provenance=prov,
    )


def _resolve_setup_col(df_setup: pd.DataFrame, bare: str) -> str | None:
    """Return the column in ``df_setup`` for ``bare``, tolerating the ``analysis.`` prefix.

    Sensitivity columns are BARE (``n_gpus``) for legacy suites or ``analysis.``-prefixed
    (``analysis.n_gpus``) for per-sub overlay suites (e.g. container-validation, which MUST
    use the prefixed form so each row applies to its sub-analysis). ``df_setup_with_system_overlays``
    preserves whichever the CSV used. Returns None if neither form is present. Used by every
    compute-column read in this renderer so a prefixed-column suite does not raise ``KeyError``.
    """
    if bare in df_setup.columns:
        return bare
    prefixed = f"analysis.{bare}"
    return prefixed if prefixed in df_setup.columns else None


def _ensure_n_devices_column(df_setup: pd.DataFrame, independent_var: str) -> pd.DataFrame:
    """Derive ``n_devices`` from ``run_mode`` × n_gpus / (n_mpi × n_omp × n_nodes) if absent.

    Resolves every compute column by bare-or-``analysis.``-prefixed name (see
    ``_resolve_setup_col``), else derivation silently no-ops on a prefixed-column suite and a
    later ``df_setup["n_devices"]`` raises ``KeyError``.
    """
    if "n_devices" in df_setup.columns:
        return df_setup

    resolved = {
        bare: _resolve_setup_col(df_setup, bare)
        for bare in ("n_mpi_procs", "n_omp_threads", "n_gpus", "n_nodes")
    }
    missing = sorted(bare for bare, col in resolved.items() if col is None)
    if missing:
        if independent_var == "n_devices":
            raise ValueError(
                "Cannot derive n_devices: sensitivity CSV is missing required columns "
                f"{missing} (checked bare and 'analysis.'-prefixed). Either declare "
                "n_devices explicitly or include the missing columns."
            )
        return df_setup

    run_mode_col = _resolve_setup_col(df_setup, "run_mode")
    run_mode = (
        df_setup[run_mode_col].astype(str).str.lower() if run_mode_col is not None else ""
    )
    n_gpus = df_setup[resolved["n_gpus"]]
    is_gpu = (run_mode == "gpu") | (n_gpus > 0)
    df_setup = df_setup.assign(
        n_devices=n_gpus.where(
            is_gpu,
            df_setup[resolved["n_mpi_procs"]]
            * df_setup[resolved["n_omp_threads"]]
            * df_setup[resolved["n_nodes"]],
        ).astype(int)
    )
    return df_setup


def _resolve_global_baseline(
    df: pd.DataFrame, *, t_col: str, indep_col: str
) -> float | None:
    """Return the minimum wallclock at the smallest N across all groups, or None
    if the dataframe is empty / has no positive wallclock at N_min.
    """
    if df.empty:
        return None
    n_min = df[indep_col].min()
    sub = df[df[indep_col] == n_min]
    if sub.empty:
        return None
    t_baseline = float(sub[t_col].min())
    if t_baseline <= 0:
        return None
    return t_baseline


def _hardware_family(gv: str) -> str:
    """The hardware COLUMN a group belongs to: every GPU token is its own family,
    every non-GPU group shares the single `cpu` family.

    Promoted from `_resolve_family_baselines`'s nested `_family_of` so the figure
    builder can partition columns by the SAME rule the baselines anchor on. Two
    rules would let a group be anchored against one family and drawn in another.
    """
    low = gv.lower()
    return gv if low.startswith("gpu") else "cpu"


def _resolve_family_baselines(
    df: pd.DataFrame, *, t_col: str, indep_col: str, group_col: str
) -> dict[str, float]:
    """Return ``{group_value: baseline_wallclock}``, anchored PER HARDWARE FAMILY.

    Mirrors the family rule the codebase already implements twice -- ``_config_diff``'s
    ``_hw_family_key`` / ``_device_count_key`` / ``_family_reference_group`` and
    ``raw_resume_identity``'s ``_b4b_family_key`` / ``_b4b_ref_key``: CPU configs anchor on
    the serial-CPU run, and each GPU hardware token anchors on ITS OWN minimum-device run.
    A third, differently-shaped implementation here would be the divergence those two
    already avoid, so the semantics are copied rather than re-derived.

    Why this is not a mode on ``_resolve_serial_baseline``: that function returns a SCALAR
    and feeds a scalar ``anchor=`` parameter, while the per-family answer is a MAPPING --
    there are three families on a two-GPU master (``cpu``, plus one per GPU token), not two.

    The global-serial anchor this replaces reports a6000 3-GPU as S = 400.30/265.20 = 1.51,
    which reads as "this scales"; the per-family anchor gives 67.64/265.20 = 0.255, the
    correct within-hardware statement. The in-source objection to a global N=1-minimum
    anchor does not reach this: a per-family anchor never compares across families, so the
    "N=1 GPU is faster than serial" failure it names cannot arise.

    Returns ``{}`` when no family resolves, so callers fall back to the serial anchor
    rather than silently comparing against an arbitrary group.
    """
    if df.empty or group_col not in df.columns or indep_col not in df.columns:
        return {}
    groups = [str(g) for g in df[group_col].dropna().unique()]
    if not groups:
        return {}

    fam_members: dict[str, list[str]] = {}
    for gv in groups:
        fam_members.setdefault(_hardware_family(gv), []).append(gv)

    out: dict[str, float] = {}
    for members in fam_members.values():
        sub = df[df[group_col].astype(str).isin(members)]
        if sub.empty:
            continue
        min_n = sub[indep_col].min()
        ref = sub[sub[indep_col] == min_n]
        if ref.empty:
            continue
        t_baseline = float(ref[t_col].min())
        if t_baseline <= 0:
            continue
        for gv in members:
            out[gv] = t_baseline
    return out


def _resolve_serial_baseline(
    df: pd.DataFrame, *, t_col: str, group_col: str, serial_group_name: str = "serial"
) -> float | None:
    """Return the wallclock of the serial group's fastest run (typically the single
    serial-at-N=1 entry). Strong-scaling speedup S(N) = t_serial / t(N) requires
    the serial baseline, not the global min — at N=1, a GPU run is typically much
    faster than a serial run, so global-min-at-smallest-N would anchor against GPU
    and produce nonsensical speedups.

    Returns None if the dataframe is empty or the serial group is absent.
    """
    if df.empty or group_col not in df.columns:
        return None
    sub = df[df[group_col].astype(str).str.lower() == serial_group_name.lower()]
    if sub.empty:
        return None
    t_baseline = float(sub[t_col].min())
    if t_baseline <= 0:
        return None
    return t_baseline


def _compute_speedup_per_group(
    df: pd.DataFrame, *, t_col: str, indep_col: str, group_col: str,
    baseline_mode: str = "per_group",
) -> dict[str, list[tuple[float, float, str]]]:
    """Compute strong-scaling speedup S(N) = t_baseline / t(N) for each group.

    Return shape: ``{group_value: [(n_devices, speedup, sa_id), ...]}``. The ``sa_id``
    is the identifier of the wallclock-minimum row at each N (the "best configuration
    at that resource level" — same row whose `t` was used to compute the speedup).
    Per-`sa_id` provenance enables hover-customdata population and per-point
    annotations downstream (F2, F3 in the kickoff figure-review).

    ``baseline_mode='per_group'``: each group anchors against its own N=1 wallclock
    (groups without N=1 are excluded — no anchor available).

    ``baseline_mode='global'``: all groups share a single anchor — the minimum
    wallclock at the smallest N across all groups. Groups without an N=1 entry
    are still included; their points are normalized against the global anchor.

    When a group has multiple sa rows at the same N, the minimum-wallclock entry
    wins (best configuration at that resource level).
    """
    if baseline_mode not in ("per_group", "global", "serial"):
        raise ValueError(f"baseline_mode must be 'per_group', 'global', or 'serial'; got {baseline_mode!r}")
    if df.empty:
        return {}
    if baseline_mode == "global":
        global_anchor = _resolve_global_baseline(df, t_col=t_col, indep_col=indep_col)
    elif baseline_mode == "serial":
        global_anchor = _resolve_serial_baseline(df, t_col=t_col, group_col=group_col)
    else:
        global_anchor = None
    if baseline_mode in ("global", "serial") and global_anchor is None:
        return {}
    out: dict[str, list[tuple[float, float, str]]] = {}
    for group_value, sub in df.groupby(group_col):
        # Keep the wallclock-min row per N so we can recover sa_id of the winning config.
        min_rows = sub.loc[sub.groupby(indep_col)[t_col].idxmin()]
        per_n_min = min_rows.set_index(indep_col)
        if baseline_mode == "per_group":
            if 1 not in per_n_min.index:
                continue
            anchor = float(per_n_min.loc[1, t_col])
        else:
            anchor = global_anchor  # type: ignore[assignment]
        if anchor is None or anchor <= 0:
            continue
        pts: list[tuple[float, float, str]] = []
        for n_val, row in per_n_min.iterrows():
            n = int(n_val) if float(n_val).is_integer() else float(n_val)
            pts.append((n, anchor / float(row[t_col]), str(row["sa_id"])))
        pts.sort(key=lambda r: r[0])
        out[str(group_value)] = pts
    return out


def _compute_efficiency_per_group(
    df: pd.DataFrame, *, t_col: str, indep_col: str, group_col: str, mode: str,
    baseline_mode: str = "per_group",
) -> dict[str, list[tuple[float, float, str]]]:
    """Compute scaling efficiency for each group.

    Return shape: ``{group_value: [(n_devices, efficiency, sa_id), ...]}``. See
    :func:`_compute_speedup_per_group` for the per-`sa_id` provenance rationale.

    - ``mode='strong'``: E_s(N) = S(N) / N = t_baseline / (N × t(N)). Ideal = 1.0.
    - ``mode='weak'``: E_w(N) = t_baseline / t(N). Ideal = 1.0.

    ``baseline_mode`` matches :func:`_compute_speedup_per_group` semantics.
    """
    if mode not in ("strong", "weak"):
        raise ValueError(f"mode must be 'strong' or 'weak'; got {mode!r}")
    if baseline_mode not in ("per_group", "global", "serial"):
        raise ValueError(f"baseline_mode must be 'per_group', 'global', or 'serial'; got {baseline_mode!r}")
    if df.empty:
        return {}
    if baseline_mode == "global":
        global_anchor = _resolve_global_baseline(df, t_col=t_col, indep_col=indep_col)
    elif baseline_mode == "serial":
        global_anchor = _resolve_serial_baseline(df, t_col=t_col, group_col=group_col)
    else:
        global_anchor = None
    if baseline_mode in ("global", "serial") and global_anchor is None:
        return {}
    out: dict[str, list[tuple[float, float, str]]] = {}
    for group_value, sub in df.groupby(group_col):
        min_rows = sub.loc[sub.groupby(indep_col)[t_col].idxmin()]
        per_n_min = min_rows.set_index(indep_col)
        if baseline_mode == "per_group":
            if 1 not in per_n_min.index:
                continue
            anchor = float(per_n_min.loc[1, t_col])
        else:
            anchor = global_anchor  # type: ignore[assignment]
        if anchor is None or anchor <= 0:
            continue
        pts: list[tuple[float, float, str]] = []
        for n_val, row in per_n_min.iterrows():
            n = int(n_val) if float(n_val).is_integer() else float(n_val)
            tN = float(row[t_col])
            if tN <= 0:
                continue
            if mode == "strong":
                eff = anchor / (n * tN)
            else:
                eff = anchor / tN
            pts.append((n, eff, str(row["sa_id"])))
        pts.sort(key=lambda r: r[0])
        out[str(group_value)] = pts
    return out


def _compute_metric_all_rows_per_group(
    df: pd.DataFrame, *, t_col: str, indep_col: str, group_col: str,
    kind: str, anchor: float,
) -> dict[str, list[tuple[float, float, str]]]:
    """Compute speedup or efficiency for EVERY row (not just the per-N min row).

    ``kind='speedup'``: y = anchor / t(N).
    ``kind='efficiency'``: y = anchor / (N * t(N)).

    Used to populate the all-points markers trace on panels 3+4 alongside the
    line trace (which draws through the per-N min — best configuration). This
    matches the panels 1+2 behavior where multi-point groups show ALL points as
    markers but only the per-N min as a connecting line. Critical for hybrid:
    a hybrid group can have multiple (n_mpi_procs, n_omp_threads) decompositions
    at the same n_devices, and the per-point spread carries information the
    min-only line hides.
    """
    if kind not in ("speedup", "efficiency"):
        raise ValueError(f"kind must be 'speedup' or 'efficiency'; got {kind!r}")
    if df.empty or anchor is None or anchor <= 0:
        return {}
    out: dict[str, list[tuple[float, float, str]]] = {}
    for group_value, sub in df.groupby(group_col):
        pts: list[tuple[float, float, str]] = []
        for _, row in sub.iterrows():
            tN = float(row[t_col])
            if tN <= 0:
                continue
            n_val = row[indep_col]
            n = int(n_val) if float(n_val).is_integer() else float(n_val)
            if kind == "speedup":
                y = anchor / tN
            else:
                y = anchor / (n * tN)
            pts.append((n, y, str(row["sa_id"])))
        pts.sort(key=lambda r: (r[0], r[1]))
        if pts:
            out[str(group_value)] = pts
    return out


def _draw_metric_panel(
    ax,
    metric_per_group: dict[str, list[tuple[float, float]]],
    *,
    df: pd.DataFrame,
    x_max: float,
    ideal_kind: str,
    sens_cfg,
    prov: ProvenanceLog,
    ideal_value: float = 1.0,
    ideal_label: str = "Ideal",
    static_cfg=None,
) -> None:
    """Draw a per-group line+marker series for speedup or efficiency.

    Each group is plotted in its own Okabe-Ito color (matching the wallclock and
    compute-cost panels via the same `_OKABE_ITO` palette and group ordering).
    A red ideal-reference line is overlaid at zorder=2 — above the gridlines (zorder=0)
    but below the data markers (zorder=3) so points always render in front.

    For hybrid groups (or any group with duplicate x-values), each marker is
    annotated with its `n_mpi_procs` value. Same convention as the wallclock /
    compute-cost panels.

    - ``ideal_kind='linear'``: y = x (the perfect-speedup S(N) = N reference).
    - ``ideal_kind='constant'``: y = ``ideal_value`` (perfect efficiency = 1.0).
    """
    groups = sorted(metric_per_group.keys(), key=str)
    # Dual-source publication style: static_cfg overrides palette + cpu marker.
    palette = static_cfg.series_palette if static_cfg is not None else sens_cfg.palette
    cpu_marker = static_cfg.cpu_marker if static_cfg is not None else sens_cfg.cpu_marker
    # Annotation lookup: map (group_value, n_devices) → n_mpi_procs at the MIN-y row.
    df_min = (
        df.loc[df.groupby(["group_value", "n_devices"])["wallclock_s"].idxmin()]
        if "wallclock_s" in df.columns and not df.empty
        else df
    )
    annotation_lookup = {
        (str(r["group_value"]), int(r["n_devices"])): int(r["n_mpi_procs"])
        for _, r in df_min.iterrows()
    }
    for i, gv in enumerate(groups):
        pts = metric_per_group[gv]
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        color = _stable_group_color(gv, palette, groups)
        with prov.artist(
            axes_id="ax_metric", kind="line",
            note=f"metric group {gv}",
        ) as a:
            a.add_channel("data", ProvenanceRef(source_path="sensitivity_datatree.zarr"))
            ax.plot(xs, ys, color=color, linestyle=sens_cfg.line_style, linewidth=sens_cfg.line_width, zorder=2)
        with prov.artist(
            axes_id="ax_metric", kind="scatter",
            note=f"metric points {gv}",
        ) as a:
            a.add_channel("data", ProvenanceRef(source_path="sensitivity_datatree.zarr"))
            ax.scatter(
                xs, ys, color=color, marker=cpu_marker, s=sens_cfg.point_size,
                edgecolor="black", linewidth=1.0, zorder=3,
            )
        if str(gv).lower() == "hybrid":
            for x, y in zip(xs, ys, strict=True):
                n_mpi = annotation_lookup.get((str(gv), int(x)))
                if n_mpi is not None:
                    ax.annotate(
                        str(n_mpi), xy=(x, y),
                        xytext=(6, 6), textcoords="offset points",
                        fontsize=8, color=color,
                    )
    if ideal_kind == "linear":
        with prov.artist(
            axes_id="ax_metric", kind="line",
            note="ideal-reference line",
        ) as a:
            a.add_channel("data", ProvenanceRef(source_path="sensitivity_datatree.zarr"))
            ax.plot(
                [1, x_max], [1, x_max],
                color=sens_cfg.ideal_line_color,
                linewidth=sens_cfg.ideal_line_width,
                zorder=2, label=ideal_label,
            )
    elif ideal_kind == "constant":
        ax.axhline(
            ideal_value,
            color=sens_cfg.ideal_line_color,
            linewidth=sens_cfg.ideal_line_width,
            zorder=2, label=ideal_label,
        )
    else:
        raise ValueError(f"ideal_kind must be 'linear' or 'constant'; got {ideal_kind!r}")


def _adaptive_time_unit(max_hours: float) -> tuple[str, float]:
    """Pick label + multiplicative factor for converting hours → display unit.

    Cascading rule per user spec: if max < 3 hr → minutes; if max < 3 min → seconds.
    """
    if max_hours < 3.0 / 60.0:  # < 3 minutes
        return "s", 3600.0
    if max_hours < 3.0:  # < 3 hours
        return "min", 60.0
    return "hrs", 1.0


def _draw_panel(
    ax, df: pd.DataFrame, *, y_col: str, group_by_var: str | None, sens_cfg, prov: ProvenanceLog, static_cfg=None,
) -> None:
    """Draw one panel of the dual-panel benchmarking figure."""
    groups = sorted(df["group_value"].dropna().unique(), key=str)
    # Dual-source publication style: static_cfg overrides palette + cpu/gpu markers.
    palette = static_cfg.series_palette if static_cfg is not None else sens_cfg.palette
    cpu_marker = static_cfg.cpu_marker if static_cfg is not None else sens_cfg.cpu_marker
    gpu_marker = static_cfg.gpu_marker if static_cfg is not None else sens_cfg.gpu_marker
    for i, gv in enumerate(groups):
        sub = df[df["group_value"] == gv].sort_values("indep_value")
        color = _stable_group_color(gv, palette, groups)
        is_gpu_group = str(gv).lower().startswith("gpu")
        is_hybrid_group = str(gv).lower() == "hybrid"
        marker = gpu_marker if is_gpu_group else cpu_marker
        is_single_point_group = str(gv).lower() in {"serial", "single_cpu", "single-cpu"}
        if is_single_point_group or len(sub) == 1:
            with prov.artist(
                axes_id="ax_panel", kind="scatter",
                note=f"single-point group {gv}",
            ) as a:
                a.add_channel("data", ProvenanceRef(source_path="sensitivity_datatree.zarr"))
                ax.scatter(
                    sub["indep_value"], sub[y_col],
                    color=color, marker=marker, s=sens_cfg.point_size,
                    edgecolor="black", linewidth=1.0, zorder=3, label=str(gv),
                )
            if is_hybrid_group:
                for _, r in sub.iterrows():
                    ax.annotate(
                        str(int(r["n_mpi_procs"])),
                        xy=(r["indep_value"], r[y_col]),
                        xytext=(6, 6), textcoords="offset points",
                        fontsize=8, color=color,
                    )
            continue
        # Multi-point group: line through MIN-y at each x-value, all points as markers.
        per_x_min = sub.groupby("indep_value", as_index=True)[y_col].min().sort_index()
        with prov.artist(
            axes_id="ax_panel", kind="line",
            note=f"multi-point line {gv}",
        ) as a:
            a.add_channel("data", ProvenanceRef(source_path="sensitivity_datatree.zarr"))
            ax.plot(
                per_x_min.index, per_x_min.values,
                color=color, linestyle=sens_cfg.line_style, linewidth=sens_cfg.line_width, zorder=2,
            )
        with prov.artist(
            axes_id="ax_panel", kind="scatter",
            note=f"multi-point markers {gv}",
        ) as a:
            a.add_channel("data", ProvenanceRef(source_path="sensitivity_datatree.zarr"))
            ax.scatter(
                sub["indep_value"], sub[y_col],
                color=color, marker=marker, s=sens_cfg.point_size,
                edgecolor="black", linewidth=1.0, zorder=3, label=str(gv),
            )
        # Hybrid: annotate every point with its n_mpi_procs value (per user spec).
        # Other groups: annotate only when duplicate x-values exist (helps disambiguate).
        if is_hybrid_group or sub["indep_value"].duplicated().any():
            for _, r in sub.iterrows():
                ax.annotate(
                    str(int(r["n_mpi_procs"])),
                    xy=(r["indep_value"], r[y_col]),
                    xytext=(6, 6), textcoords="offset points",
                    fontsize=8, color=color,
                )


def _collect_rows(
    analysis: TRITONSWMM_analysis, dependent_var: str
) -> tuple[list[dict[str, Any]], list[Path]]:
    """Collect (sa_id, event_iloc, value) rows + source paths for the dependent_var."""
    if not dependent_var.startswith("performance."):
        raise ValueError(
            f"dependent_var {dependent_var!r} must start with 'performance.' "
            f"(only performance metrics are supported in v1)"
        )
    col = dependent_var.split(".", 1)[1]
    sensitivity = analysis.sensitivity
    rows: list[dict[str, Any]] = []
    source_paths: list[Path] = []

    # RESUME CORRECTNESS. `performance.{col}` is the slowest-rank sum over the
    # performance{N}.txt checkpoints PRESENT ON DISK when the process rule ran. On a
    # hotstart-resumed sim that set is not the whole run, so the column under-reports
    # the run's wallclock -- and by a factor that is NOT constant, so it also reorders
    # configurations. Measured on delivered generation 01655abb60c2, n=28 per arm:
    # perf_Total / wall_clock_ledger_s = 0.975 (clean triton) / 0.985 (clean coupled)
    # vs 0.527 +/- 0.133 (resume triton, range 0.21-0.72) / 0.215 +/- 0.012 (resume
    # coupled). Spearman(perf_Total, ledger) falls from 0.996/0.9995 on the clean arms
    # to 0.909 on resume-triton -- i.e. the plotted metric disagrees with true wallclock
    # about which configuration is faster.
    #
    # `wall_clock_ledger_s` is the append-only per-attempt ledger summed by
    # run_simulation.read_walltime_ledger_total_s, the only kill-survivable per-attempt
    # wall source (the perf files and sim_run_time_minutes are overwrite-prone). It is
    # preferred for Total ONLY: it is a whole-process wall and is not a substitute for
    # the Init or Simulation decompositions, which stay datatree-sourced.
    _ledger: dict[tuple[str, int], float] = {}
    _csv = analysis.analysis_paths.analysis_dir / "scenario_status.csv"
    if col == "Total" and _csv.exists():
        _df_status = pd.read_csv(_csv)
        if {"sa_id", "event_iloc", "wall_clock_ledger_s"} <= set(_df_status.columns):
            for _r in _df_status.itertuples(index=False):
                _v = getattr(_r, "wall_clock_ledger_s", None)
                if _v is None or pd.isna(_v):
                    continue
                _ledger[(str(_r.sa_id), int(_r.event_iloc))] = float(_v)
            if _ledger:
                source_paths.append(_csv)

    datatree_path = analysis.analysis_paths.sensitivity_datatree_zarr
    tree: xr.DataTree | None = None
    if datatree_path is not None and datatree_path.exists():
        tree = xr.open_datatree(str(datatree_path), engine="zarr", consolidated=False)
        source_paths.append(datatree_path)

    for sa_id, sub_analysis in sensitivity.sub_analyses.items():
        node_ds = _find_perf_node(tree, sa_id) if tree is not None else None
        if node_ds is not None and col in node_ds.data_vars:
            for event_iloc in sub_analysis.df_sims.index:
                value = _ledger.get((str(sa_id), int(event_iloc)))
                if value is None:
                    value = _scalar_at_event(node_ds[col], int(event_iloc))
                if value is None:
                    continue
                rows.append({"sa_id": sa_id, "event_iloc": int(event_iloc), "value": value})
            continue
        enabled = sub_analysis._get_enabled_model_types()
        if "swmm" in enabled and len(enabled) == 1:
            for event_iloc in sub_analysis.df_sims.index:
                proc = sub_analysis._retrieve_sim_run_processing_object(int(event_iloc))
                rpt = proc.scen_paths.swmm_full_rpt_file
                if not rpt or not rpt.exists():
                    continue
                value = parse_total_elapsed(rpt)
                if value is None:
                    continue
                rows.append({"sa_id": sa_id, "event_iloc": int(event_iloc), "value": value})
                source_paths.append(rpt)
    return rows, source_paths


def _find_perf_node(tree: xr.DataTree, sa_id: str) -> xr.Dataset | None:
    """Locate the per-sa_id performance node, preferring tritonswmm over triton-only."""
    for model_subpath in ("tritonswmm/performance", "triton_only/performance"):
        path = f"/sa_{sa_id}/{model_subpath}"
        try:
            return tree[path].ds
        except KeyError:
            continue
    return None


def _resolve_model_arm(analysis) -> str | None:
    """Return the single TRITON model arm this benchmarking master carries.

    Phase 6 change (2). Under the sibling-master architecture each sensitivity
    master enables exactly one TRITON arm, so the figure is single-arm and the
    encoding is constant per figure (self-labeling). ``"coupled"`` == the
    TRITON-SWMM arm (``tritonswmm/performance``); ``"uncoupled"`` == the
    pure-TRITON arm (``triton_only/performance``). Returns ``None`` for a
    non-TRITON master (e.g. swmm-only) so the pre-change filled/dashed default is
    preserved byte-identically. The tritonswmm-first precedence mirrors
    ``_find_perf_node``'s probe order, so config-truth agrees with the plotted
    node under the single-arm invariant.
    """
    sens = getattr(analysis, "sensitivity", None)
    if sens is None:
        return None
    enabled: set[str] = set()
    for sub in sens.sub_analyses.values():
        enabled.update(sub._get_enabled_model_types())
    if "tritonswmm" in enabled:
        return "coupled"
    if "triton" in enabled:
        return "uncoupled"
    return None


def _scalar_at_event(da: xr.DataArray, event_iloc: int) -> float | None:
    """Extract a scalar value at the given event_iloc, returning None if absent."""
    if "event_iloc" in da.dims:
        try:
            return float(da.sel(event_iloc=event_iloc).values.item())
        except (KeyError, ValueError):
            return None
    try:
        return float(da.values.item())
    except (TypeError, ValueError):
        return None


def _build_sensitivity_benchmarking_figure(
    df: pd.DataFrame,
    speedup_per_group: dict,
    strong_eff_per_group: dict,
    *,
    wall_unit: str,
    cost_unit: str,
    independent_var: str,
    group_by_var: str | None,
    sens_cfg,
    output_path: Path,
    source_paths: list,
    analysis_dir,
    plotly_js_mode: str,
    prov: ProvenanceLog,
    gpu_legend_suffix: str = "",
    facet: FacetConfig | None = None,
    speedup_all_rows: dict | None = None,
    efficiency_all_rows: dict | None = None,
    speedup_range_mode: str = "full_ideal",
    model_arm: str | None = None,
):
    """Plotly MV port (pre-/design-figure): static 4-panel benchmarking figure.
    Wall-clock | Compute-cost | Strong-scaling speedup | Parallel efficiency,
    stacked rows=4, cols=1 with shared x-axis. One trace per group_by_var value
    per panel, sharing the Okabe-Ito palette (sens_cfg.palette) as Plotly's
    colorway. Informationally congruent with the matplotlib branch — no hover
    refinement, no line-toggle UX, no per-panel zoom/pan customization.
    """
    # Side-effect import: registers `triton_journal` Plotly template.
    from hhemt.report_renderers import _plotly_theme  # noqa: F401

    # BM-6: the two SCALING panels get one column per hardware family (CPU, then
    # one per GPU token); the wall-clock and compute-cost panels above stay full
    # width via colspan. Columns are DERIVED from the data, so a family absent
    # from this master produces no column rather than an empty one (P9).
    _hw_cols = sorted(
        {_hardware_family(str(gv)) for gv in df["group_value"].dropna().unique()},
        key=lambda f: (f != "cpu", f),
    )
    _n_hw = max(len(_hw_cols), 1)
    # P10: spacing is derived, never hard-coded. Plotly's own default is
    # 0.2/cols; a third of that keeps the per-column y-axis titles clear of the
    # neighbouring panel at the column counts this campaign produces.
    _h_space = min(0.09, 0.20 / _n_hw)
    fig = make_subplots(
        rows=4, cols=_n_hw, shared_xaxes=True,
        vertical_spacing=0.06, horizontal_spacing=_h_space,
        specs=[
            [{"colspan": _n_hw}] + [None] * (_n_hw - 1),
            [{"colspan": _n_hw}] + [None] * (_n_hw - 1),
            [{} for _ in range(_n_hw)],
            [{} for _ in range(_n_hw)],
        ],
        subplot_titles=[""] * 2 + [f"{f}" for f in _hw_cols] + [""] * _n_hw,
    )
    fig.update_layout(
        template="plotly_white",
        colorway=list(sens_cfg.palette),
        showlegend=True,
        legend=dict(
            title=group_by_var if group_by_var is not None else "",
            orientation="v", yanchor="top", y=1.0, xanchor="left", x=1.02,
        ),
        # No `b=` here: the bottom margin is DERIVED from the caption and set at the
        # add_figure_caption call site below. This b=80 was superseded by that call
        # (same figure, no intervening def) and survived only as a leftover -- exactly
        # the hand-tuned constant the geometry guard now detects.
        margin=dict(l=90, r=120, t=30),
        height=1000,
    )

    # ---- Panels 1 + 2: wallclock + compute-cost -------------------------
    for row, y_col, panel_id in (
        (1, "wallclock_disp", "ax_wall_plotly"),
        (2, "compute_disp", "ax_cost_plotly"),
    ):
        _plotly_metric_panel(
            fig, df, y_col=y_col, row=row, panel_id=panel_id,
            group_by_var=group_by_var, sens_cfg=sens_cfg, prov=prov,
            show_in_legend=(row == 1),
            gpu_legend_suffix=gpu_legend_suffix,
            model_arm=model_arm,
        )

    # ---- Panels 3 + 4: speedup + efficiency, one column per hardware family --
    # Groups are DISJOINT across families, so filtering the per-group dicts
    # partitions the data with no group drawn twice and no legend entry
    # duplicated. The ideal reference is emitted per column -- BM-6's "a single
    # reference case in each one to define perfect scaling" -- but only column 1
    # carries the legend entry, so the legend keeps ONE row for both red lines.
    for _ci, _fam in enumerate(_hw_cols, start=1):
        _df_c = df[df["group_value"].astype(str).map(_hardware_family) == _fam]
        if _df_c.empty:
            continue
        _sp_c = {k: v for k, v in speedup_per_group.items() if _hardware_family(str(k)) == _fam}
        _ef_c = {k: v for k, v in strong_eff_per_group.items() if _hardware_family(str(k)) == _fam}
        _sp_all_c = {
            k: v for k, v in (speedup_all_rows or {}).items() if _hardware_family(str(k)) == _fam
        } or None
        _ef_all_c = {
            k: v for k, v in (efficiency_all_rows or {}).items() if _hardware_family(str(k)) == _fam
        } or None
        _x_max_c = float(_df_c["n_devices"].max())
        _plotly_metric_panel_precomputed(
            fig, _sp_c, df_for_groups=_df_c, row=3, col=_ci,
            panel_id=f"ax_speedup_plotly_c{_ci}",
            color_groups=sorted(df["group_value"].dropna().unique(), key=str),
            ideal_kind="linear", x_max=_x_max_c,
            ideal_label="ideal speedup (S=N)<br>and efficiency (=1.0)",
            sens_cfg=sens_cfg, prov=prov, show_in_legend=False,
            gpu_legend_suffix=gpu_legend_suffix,
            all_rows_per_group=_sp_all_c,
            ideal_show_in_legend=(_ci == 1),
            model_arm=model_arm,
        )
        _plotly_metric_panel_precomputed(
            fig, _ef_c, df_for_groups=_df_c, row=4, col=_ci,
            panel_id=f"ax_efficiency_plotly_c{_ci}",
            color_groups=sorted(df["group_value"].dropna().unique(), key=str),
            ideal_kind="constant", ideal_value=1.0, x_max=_x_max_c,
            ideal_label="ideal speedup (S=N)<br>and efficiency (=1.0)",
            sens_cfg=sens_cfg, prov=prov, show_in_legend=False,
            gpu_legend_suffix=gpu_legend_suffix,
            all_rows_per_group=_ef_all_c,
            ideal_show_in_legend=False,
            model_arm=model_arm,
        )
    # F-FU-6 / Q1: speedup panel range mode. Default `full_ideal` shows the full
    # ideal line; `empirical_clipped` clips y to the empirical max for better
    # discrimination of low-speedup points (Kelleher Guideline 4) and adds a
    # corner annotation naming the ideal slope so the reader doesn't lose the
    # reference.
    if speedup_range_mode == "empirical_clipped" and speedup_all_rows:
        # BM-6: max_empirical stays GLOBAL -- computed over the unfiltered
        # speedup_all_rows, before the per-family column split. A per-column max
        # would give each hardware family its own y-scale, which makes the columns
        # silently non-comparable; that is worse than not clipping at all, because
        # the reader has no cue that the axes differ.
        max_empirical = max(
            (p[1] for pts in speedup_all_rows.values() for p in pts),
            default=None,
        )
        if max_empirical is not None and max_empirical > 0:
            for _ci in range(1, _n_hw + 1):
                fig.update_yaxes(range=[0, max_empirical * 1.1], row=3, col=_ci)
            # The ideal-reference line's truncation is communicated via the legend
            # entry "Ideal speedup (S=N)" rather than a corner annotation (v5
            # feedback); the legend entry stays visible at the clipped y-range,
            # the line itself extends off the panel.

    # ---- Axis labels + tickers ------------------------------------------
    xlabel_text = sens_cfg.independent_var_labels.get(independent_var, independent_var)
    fig.update_xaxes(title_text="", row=1, col=1)
    fig.update_xaxes(title_text="", row=2, col=1)
    for _ci in range(1, _n_hw + 1):
        fig.update_xaxes(title_text="", row=3, col=_ci)
        fig.update_xaxes(title_text=xlabel_text, row=4, col=_ci)
    fig.update_yaxes(title_text=f"Wall-clock ({wall_unit})", row=1, col=1)
    fig.update_yaxes(title_text=f"Compute cost ({cost_unit} × devices)", row=2, col=1)
    # OE-1: the numerator is the FAMILY baseline (CPU -> serial CPU; each GPU hardware ->
    # its own 1-GPU run), not the plotted series' own t(1) and not a global serial.
    # The `family` subscript distinguished baselines ACROSS panels; each column is now
    # one hardware family, so it distinguishes nothing within a panel. The cross-panel
    # fact it carried ("S = 1.0 is a different wall-clock on each curve") is stated in
    # full by the footnote below and is NOT weakened by this relabel -- but do not drop
    # that footnote in the same pass, because it is now the sole carrier.
    fig.update_yaxes(title_text="Strong-Scaling Speedup<br>S(N) = t(1) / t(N)", row=3, col=1)
    fig.update_yaxes(title_text="Strong-Scaling Efficiency<br>E<sub>s</sub>(N) = t(1) / (N · t(N))", row=4, col=1)
    # Adopt hhemt.figure_caption: this annotation predated the module (see the retired
    # v3/v4/v5 b= history in git). `y=-0.10` was a FRACTION of the plot area while the
    # x-axis-title band below it is a fixed pixel height, so the two converge as the
    # hardware-column count grows -- which is why the middle column's long
    # "Number of Devices (CPUs or GPUs)" title is the one that reaches the block first.
    _bench_caption = (
        "* number next to hybrid scenarios indicates number of MPI processes"
        " Speedup and efficiency are measured against each hardware family's own"
        " minimum-device run (CPU → serial CPU; each GPU → its own 1-GPU run),"
        " so S = 1.0 denotes a different wall-clock on each curve."
        " Hollow markers mark compute configurations run more than once;"
        " every replicate is plotted."
    )
    _bench_plot_h = 1000 - 30  # height minus top margin; bottom is what we are deriving
    _bench_b_px = add_figure_caption(
        fig,
        _bench_caption,
        content_w_px=content_width_px(fig, fallback_px=1000) - 90 - 120,
        plot_h_px=_bench_plot_h,
        font_px=10,
        axis_band_px=46,
    )
    fig.update_layout(margin=dict(l=90, r=120, t=30, b=_bench_b_px))
    if sens_cfg.show_gridlines:
        for r in range(1, 5):
            _cols = range(1, _n_hw + 1) if r >= 3 else range(1, 2)
            for _c in _cols:
                fig.update_xaxes(
                    showgrid=True, gridcolor=sens_cfg.gridline_color,
                    gridwidth=sens_cfg.gridline_width, row=r, col=_c,
                )
                fig.update_yaxes(
                    showgrid=True, gridcolor=sens_cfg.gridline_color,
                    gridwidth=sens_cfg.gridline_width, row=r, col=_c,
                )

    # ---- Emit -----------------------------------------------------------
    plotly_config = {
        "displayModeBar": True,
        "displaylogo": False,
        "modeBarButtonsToRemove": [
            "lasso2d", "select2d", "autoScale2d",
            "hoverCompareCartesian", "hoverClosestCartesian",
            "toggleSpikelines",
        ],
        "toImageButtonOptions": {
            "format": "svg", "filename": "sensitivity_benchmarking", "scale": 2,
        },
    }
    return fig, plotly_config


def _render_plotly_branch(
    df: pd.DataFrame,
    speedup_per_group: dict,
    strong_eff_per_group: dict,
    *,
    wall_unit: str,
    cost_unit: str,
    independent_var: str,
    group_by_var: str | None,
    sens_cfg,
    output_path: Path,
    source_paths: list,
    analysis_dir,
    plotly_js_mode: str,
    prov: ProvenanceLog,
    gpu_legend_suffix: str = "",
    facet: FacetConfig | None = None,
    speedup_all_rows: dict | None = None,
    efficiency_all_rows: dict | None = None,
    speedup_range_mode: str = "full_ideal",
    model_arm: str | None = None,
) -> Path:
    fig, plotly_config = _build_sensitivity_benchmarking_figure(
        df,
        speedup_per_group,
        strong_eff_per_group,
        wall_unit=wall_unit,
        cost_unit=cost_unit,
        independent_var=independent_var,
        group_by_var=group_by_var,
        sens_cfg=sens_cfg,
        output_path=output_path,
        source_paths=source_paths,
        analysis_dir=analysis_dir,
        plotly_js_mode=plotly_js_mode,
        prov=prov,
        gpu_legend_suffix=gpu_legend_suffix,
        facet=facet,
        speedup_all_rows=speedup_all_rows,
        efficiency_all_rows=efficiency_all_rows,
        speedup_range_mode=speedup_range_mode,
        model_arm=model_arm,
    )
    html_text = pio.to_html(
        fig, include_plotlyjs=plotly_js_mode,
        full_html=True, config=plotly_config,
    )

    try:
        fig.write_image(
            output_path.with_suffix(".svg"),
            engine="kaleido", width=1400, height=700, scale=1,
        )
    except Exception as exc:  # noqa: BLE001 — Kaleido failure is non-fatal
        import logging
        logging.getLogger(__name__).warning(
            "Kaleido SVG export skipped for %s: %s",
            output_path.with_suffix(".svg"), exc,
        )

    return emit_plot_with_sources(
        html_text, output_path, source_paths,
        analysis_dir=analysis_dir,
        output_format="html",
        manifest_data={
            "independent_var": independent_var,
            "group_by_var": group_by_var,
            "group_count": int(df["group_value"].nunique()),
            "data_point_count": int(len(df)),
            "wall_unit": wall_unit,
            "cost_unit": cost_unit,
        },
        provenance=prov,
    )


def _plotly_metric_panel(
    fig,
    df: pd.DataFrame,
    *,
    y_col: str,
    row: int,
    panel_id: str,
    group_by_var: str | None,
    sens_cfg,
    prov: ProvenanceLog,
    show_in_legend: bool,
    gpu_legend_suffix: str = "",
    model_arm: str | None = None,
) -> None:
    """Plot one of the wallclock/compute-cost panels (raw data per group)."""
    groups = sorted(df["group_value"].dropna().unique(), key=str)
    cfg_cols = ["n_mpi_procs", "n_omp_threads", "n_gpus", "n_nodes"]
    available_cfg_cols = [c for c in cfg_cols if c in df.columns]
    for i, gv in enumerate(groups):
        sub = df[df["group_value"] == gv].sort_values("indep_value")
        color = _stable_group_color(gv, sens_cfg.palette, groups)
        is_gpu_group = str(gv).lower().startswith("gpu")
        is_hybrid_group = str(gv).lower() == "hybrid"
        is_serial_group = str(gv).lower() in {"serial", "single_cpu", "single-cpu"}
        is_single_point_group = is_serial_group or len(sub) == 1
        if is_gpu_group:
            marker_symbol = "triangle-up"
        elif is_serial_group:
            marker_symbol = "star"
        else:
            marker_symbol = "circle"
        # Iteration 4: model-arm encoding REMOVED. This renderer is structurally
        # single-arm (see _resolve_model_arm's docstring: each sibling master enables
        # exactly one TRITON arm) and no cross_experiment_* renderer consumes it, so
        # the arm is carried by the figure's title and its position in the combined
        # report's paired_figures small-multiple. Arm-conditioned fill and dash made
        # the two arms' symbology DISJOINT by construction, which is what regressed
        # the standing "symbology identical across models" requirement. The freed
        # marker FILL channel now encodes replicate count; the connector dash is a
        # single constant style.
        arm_dash = "solid"
        if is_gpu_group:
            legend_name = f"{gv}{gpu_legend_suffix}"
        elif is_hybrid_group:
            legend_name = f"{gv}*"
        else:
            legend_name = str(gv)
        if not is_single_point_group:
            # Average replicates of one config first, THEN take the per-N min across
            # distinct configs (the documented "best configuration at that resource
            # level" rule). Without the first step the min is a min over RUNS, which on
            # the staged master discards a 159.3 s vs 275.9 s replicate spread silently.
            _by_cfg = (
                sub.groupby(["indep_value", "config_id"], as_index=False)[y_col].mean()
                if "config_id" in sub.columns
                else sub
            )
            per_x_min = _by_cfg.groupby("indep_value", as_index=True)[y_col].min().sort_index()
            with prov.artist(
                axes_id=panel_id, kind="line",
                note=f"multi-point line {gv} (panel {panel_id})",
            ) as a:
                a.add_channel("data", ProvenanceRef(source_path="sensitivity_datatree.zarr"))
                fig.add_trace(
                    go.Scatter(
                        x=per_x_min.index, y=per_x_min.values,
                        mode="lines",
                        line=dict(color=color, dash=arm_dash,
                                  width=sens_cfg.line_width),
                        legendgroup=str(gv), name=legend_name,
                        showlegend=False, hoverinfo="skip",
                    ),
                    row=row, col=1,
                )
        # Build hybrid n_mpi_procs annotations as marker text (matches matplotlib reference).
        marker_mode = "markers+text" if is_hybrid_group and "n_mpi_procs" in sub.columns else "markers"
        marker_text = (
            sub["n_mpi_procs"].fillna(0).astype(int).astype(str).tolist()
            if is_hybrid_group and "n_mpi_procs" in sub.columns
            else None
        )
        # Hover customdata: per-point MPI ranks, OMP threads, GPUs, Nodes (F2).
        if available_cfg_cols:
            customdata = sub[available_cfg_cols].fillna(0).astype(int).to_numpy()
        else:
            customdata = None
        hover_lines = [f"<b>{legend_name}</b>", "x: %{x}", "y: %{y:.3f}"]
        if customdata is not None:
            # `_cfg_col`, NOT `col`: this iterates DataFrame COLUMN NAMES, a
            # different sense of "col" from plotly's subplot-grid `col=`. The
            # sibling `_plotly_metric_panel_precomputed` took a `col` parameter and
            # this same loop shadowed it -- and because the loop sits BELOW the
            # first add_trace, only groups 2+ were corrupted, so a single-group
            # fixture would have passed. This function takes no `col` parameter
            # today; the rename is here so that adding one later cannot re-form the
            # collision silently.
            for j, _cfg_col in enumerate(available_cfg_cols):
                label = {"n_mpi_procs": "MPI ranks",
                         "n_omp_threads": "OMP threads",
                         "n_gpus": "GPUs",
                         "n_nodes": "Nodes"}.get(_cfg_col, _cfg_col)
                hover_lines.append(f"{label}: %{{customdata[{j}]}}")
        hovertemplate_str = "<br>".join(hover_lines) + "<extra></extra>"
        with prov.artist(
            axes_id=panel_id, kind="scatter",
            note=f"markers {gv} (panel {panel_id})",
        ) as a:
            a.add_channel("data", ProvenanceRef(source_path="sensitivity_datatree.zarr"))
            # BM-3: hollow marker == this config has repeated runs, matching the
            # encoding already applied on panels 3+4. No sa_id lookup is needed
            # here: the marker arrays ARE `sub`'s rows in `sub`'s order, so the
            # n_replicates column is already point-aligned. `rgba(0,0,0,0)` over
            # "white" keeps the plot background visible through the marker.
            _fill = color
            if "n_replicates" in sub.columns:
                _fill = [
                    "rgba(0,0,0,0)" if int(r) > 1 else color
                    for r in sub["n_replicates"].fillna(1)
                ]
            scatter_kwargs = dict(
                x=sub["indep_value"], y=sub[y_col],
                mode=marker_mode,
                text=marker_text,
                textposition="top right",
                textfont=dict(size=9, color=color),
                marker=dict(
                    symbol=marker_symbol,
                    size=max(int(sens_cfg.point_size ** 0.5), 6),
                    color=_fill, line=dict(color=color, width=1.4),
                ),
                legendgroup=str(gv), name=legend_name,
                showlegend=show_in_legend,
                hovertemplate=hovertemplate_str,
            )
            if customdata is not None:
                scatter_kwargs["customdata"] = customdata
            fig.add_trace(
                go.Scatter(**scatter_kwargs),
                row=row, col=1,
            )


def _plotly_metric_panel_precomputed(
    fig,
    per_group_data: dict,
    *,
    df_for_groups: pd.DataFrame,
    row: int,
    col: int = 1,
    panel_id: str,
    ideal_kind: str,
    x_max: float,
    ideal_label: str,
    sens_cfg,
    prov: ProvenanceLog,
    show_in_legend: bool,
    ideal_value: float = 1.0,
    gpu_legend_suffix: str = "",
    all_rows_per_group: dict | None = None,
    ideal_show_in_legend: bool = False,
    model_arm: str | None = None,
    color_groups: list | None = None,
) -> None:
    """Plot speedup / efficiency panel from precomputed per-group data.

    Accepts ``per_group_data`` as ``{gv: list[(x, y, sa_id), ...]}`` (current format
    returned by ``_compute_speedup_per_group`` / ``_compute_efficiency_per_group``,
    F2/F3 enriched), OR legacy ``{gv: list[(x, y), ...]}`` (older callers), OR
    ``{gv: {xs: [...], ys: [...]}}`` (legacy dict form for forward compat).

    When per-row `sa_id` is available, populates `customdata` for hover enrichment
    (n_mpi_procs / n_omp_threads / n_gpus / n_nodes) and per-point text annotations
    on hybrid markers (matplotlib reference parity for panels 3+4).
    """
    groups = sorted(df_for_groups["group_value"].dropna().unique(), key=str)
    # `_stable_group_color`'s third parameter is named `all_groups` and its whole
    # purpose is a palette index that does not shift when the group set shrinks.
    # `df_for_groups` here is ALREADY filtered to one hardware family, so passing
    # `groups` gives a GPU-only column order.index("gpu") == 0 while the legend --
    # built solely from row 1 on the UNFILTERED frame (show_in_legend=(row == 1)) --
    # gives order.index("gpu") == 2. Same series, two palette slots. `color_groups`
    # is the unfiltered list; it defaults to `groups` so single-panel callers are
    # unchanged.
    _color_groups = color_groups if color_groups is not None else groups
    # Per-sa_id config lookup for hover customdata + hybrid annotations (F2, F3).
    sa_cfg_cols = ["n_mpi_procs", "n_omp_threads", "n_gpus", "n_nodes"]
    available_cfg_cols = [c for c in sa_cfg_cols if c in df_for_groups.columns]
    if available_cfg_cols and "sa_id" in df_for_groups.columns:
        # Deduplicate to one row per sa_id (config doesn't vary within an sa_id).
        sa_cfg_lookup = (
            df_for_groups.drop_duplicates(subset=["sa_id"])
            .set_index("sa_id")[available_cfg_cols]
            .fillna(0).astype(int)
        )
    else:
        sa_cfg_lookup = None

    # BM-3: n_replicates drives marker FILL -- a config with repeated runs renders
    # hollow. The column is computed in render() at the `_r\d+`-strip and was never
    # read again; this is the lookup that consumes it. Built here rather than passed
    # in because `df_for_groups` IS the same frame render() computed it on, so a new
    # parameter would thread a value already in scope.
    if "n_replicates" in df_for_groups.columns and "sa_id" in df_for_groups.columns:
        n_rep_by_sa = (
            df_for_groups.drop_duplicates(subset=["sa_id"])
            .set_index("sa_id")["n_replicates"]
            .astype(int)
        )
    else:
        n_rep_by_sa = None

    def _extract_xyz(data):
        """Return (xs, ys, sa_ids) from one of the supported per-group data formats."""
        if isinstance(data, dict):
            return data.get("xs") or [], data.get("ys") or [], None
        if isinstance(data, list):
            if not data:
                return [], [], None
            xs_local = [p[0] for p in data]
            ys_local = [p[1] for p in data]
            sa_local = [str(p[2]) for p in data] if len(data[0]) >= 3 else None
            return xs_local, ys_local, sa_local
        return [], [], None

    def _build_customdata(sa_ids_local):
        if sa_ids_local is None or sa_cfg_lookup is None:
            return None
        try:
            return sa_cfg_lookup.reindex(sa_ids_local).to_numpy()
        except KeyError:
            return None

    for i, gv in enumerate(groups):
        if str(gv) not in per_group_data and gv not in per_group_data:
            continue
        data = per_group_data.get(str(gv), per_group_data.get(gv))
        line_xs, line_ys, line_sa = _extract_xyz(data)
        if not line_xs:
            continue
        # If all_rows_per_group is provided, use it for the markers trace; else fall
        # back to the per-N-min data (today's behavior — line and markers coincide).
        all_data = None
        if all_rows_per_group is not None:
            all_data = all_rows_per_group.get(str(gv), all_rows_per_group.get(gv))
        if all_data is None:
            marker_xs, marker_ys, marker_sa = line_xs, line_ys, line_sa
        else:
            marker_xs, marker_ys, marker_sa = _extract_xyz(all_data)
            if not marker_xs:
                # Empty all-rows fall back to line data for markers.
                marker_xs, marker_ys, marker_sa = line_xs, line_ys, line_sa
        color = _stable_group_color(gv, sens_cfg.palette, _color_groups)
        is_gpu_group = str(gv).lower().startswith("gpu")
        is_hybrid_group = str(gv).lower() == "hybrid"
        is_serial_group = str(gv).lower() in {"serial", "single_cpu", "single-cpu"}
        if is_gpu_group:
            marker_symbol = "triangle-up"
        elif is_serial_group:
            marker_symbol = "star"
        else:
            marker_symbol = "circle"
        # Iteration 4: model-arm encoding REMOVED (see _plotly_metric_panel for the
        # rationale). The ideal-reference line below is NOT a data connector and was
        # never arm-encoded.
        arm_dash = "solid"
        legend_name = f"{gv}{gpu_legend_suffix}" if is_gpu_group else (
            f"{gv}*" if is_hybrid_group else str(gv)
        )
        # Build hover customdata + hybrid annotation text from sa_id provenance.
        marker_customdata = _build_customdata(marker_sa)
        marker_text = None
        marker_mode = "markers"
        if is_hybrid_group and marker_customdata is not None and "n_mpi_procs" in available_cfg_cols:
            mpi_col_idx = available_cfg_cols.index("n_mpi_procs")
            marker_text = [str(int(row[mpi_col_idx])) for row in marker_customdata]
            marker_mode = "markers+text"
        hover_lines = [f"<b>{legend_name}</b>",
                       "x: %{x}",
                       "y: %{y:.3f}"]
        if marker_customdata is not None and available_cfg_cols:
            # `_cfg_col`, NOT `col`: this function now takes a `col` PARAMETER
            # (the subplot column), and a loop variable of that name rebinds it
            # to the last config field for every statement below. The free-name
            # check cannot see this -- the name is bound, just to the wrong thing.
            for j, _cfg_col in enumerate(available_cfg_cols):
                label = {"n_mpi_procs": "MPI ranks",
                         "n_omp_threads": "OMP threads",
                         "n_gpus": "GPUs",
                         "n_nodes": "Nodes"}.get(_cfg_col, _cfg_col)
                hover_lines.append(f"{label}: %{{customdata[{j}]}}")
        hovertemplate_str = "<br>".join(hover_lines) + "<extra></extra>"
        # Line trace through per-N min — dashed, no markers, no hover (line is connective only).
        if is_serial_group or len(line_xs) == 1:
            line_trace = None  # serial / single-point groups skip the line, render markers only
        else:
            line_trace = dict(
                x=line_xs, y=line_ys, mode="lines",
                line=dict(color=color, dash=arm_dash, width=sens_cfg.line_width),
                legendgroup=str(gv), name=legend_name,
                showlegend=False, hoverinfo="skip",
            )
        if line_trace is not None:
            with prov.artist(
                axes_id=panel_id, kind="line",
                note=f"metric min-line {gv} (panel {panel_id})",
            ) as a:
                a.add_channel("data", ProvenanceRef(source_path="sensitivity_datatree.zarr"))
                fig.add_trace(go.Scatter(**line_trace), row=row, col=col)
        # Markers trace — all-row points (or fall back to per-N-min if all-row not provided).
        # BM-3: hollow marker == this config has repeated runs. Per-POINT, not
        # per-trace: a group can mix replicated and single-run configs, so the fill
        # is a list aligned to marker_xs. `rgba(0,0,0,0)` rather than "white" keeps
        # the plot background visible through the marker on any template.
        _marker_fill = color
        if n_rep_by_sa is not None and marker_sa is not None:
            _reps = n_rep_by_sa.reindex(marker_sa)
            _marker_fill = [
                "rgba(0,0,0,0)" if (r == r and int(r) > 1) else color for r in _reps
            ]
        marker_kwargs = dict(
            x=marker_xs, y=marker_ys, mode=marker_mode,
            marker=dict(
                symbol=marker_symbol,
                size=max(int(sens_cfg.point_size ** 0.5), 6),
                color=_marker_fill, line=dict(color=color, width=1.4),
            ),
            legendgroup=str(gv), name=legend_name,
            showlegend=show_in_legend,
            hovertemplate=hovertemplate_str,
        )
        if marker_customdata is not None:
            marker_kwargs["customdata"] = marker_customdata
        if marker_text is not None:
            marker_kwargs["text"] = marker_text
            marker_kwargs["textposition"] = "top right"
            marker_kwargs["textfont"] = dict(size=9, color=color)
        with prov.artist(
            axes_id=panel_id, kind="scatter",
            note=f"metric markers {gv} (panel {panel_id})",
        ) as a:
            a.add_channel("data", ProvenanceRef(source_path="sensitivity_datatree.zarr"))
            fig.add_trace(go.Scatter(**marker_kwargs), row=row, col=col)
    # Ideal reference line: linear (S=N) or constant (E=1.0).
    if ideal_kind == "linear":
        ideal_x = [1.0, x_max]
        ideal_y = [1.0, x_max]
    elif ideal_kind == "constant":
        ideal_x = [1.0, x_max]
        ideal_y = [ideal_value, ideal_value]
    else:
        ideal_x = []
        ideal_y = []
    if ideal_x:
        with prov.artist(
            axes_id=panel_id, kind="line",
            note=f"ideal-reference line ({ideal_kind})",
        ):
            fig.add_trace(
                go.Scatter(
                    x=ideal_x, y=ideal_y, mode="lines",
                    line=dict(color=sens_cfg.ideal_line_color,
                              width=sens_cfg.ideal_line_width),
                    name=ideal_label,
                    legendgroup="ideal",
                    showlegend=ideal_show_in_legend,
                    hoverinfo="skip",
                ),
                row=row, col=col,
            )
