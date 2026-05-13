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
``.rpt`` parsing via :func:`TRITON_SWMM_toolkit.swmm_output_parser.parse_total_elapsed`.

Derived columns: when ``independent_var`` is ``n_devices`` and the column is absent
from the sensitivity CSV, the renderer computes it as
``n_gpus if run_mode == "gpu" else n_mpi_procs * n_omp_threads * n_nodes``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import xarray as xr
from plotly.subplots import make_subplots

from TRITON_SWMM_toolkit.report_renderers._figure_emission import emit_plot_with_sources
from TRITON_SWMM_toolkit.report_renderers._provenance import ProvenanceLog, ProvenanceRef
from TRITON_SWMM_toolkit.swmm_output_parser import parse_total_elapsed

if TYPE_CHECKING:
    from TRITON_SWMM_toolkit.analysis import TRITONSWMM_analysis
    from TRITON_SWMM_toolkit.config.report import report_config


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
) -> Path:
    """Render the dual-panel benchmarking figure for one independent variable."""
    from TRITON_SWMM_toolkit.report_renderers.system_overview import _apply_rcparams

    _apply_rcparams(report_cfg)

    if report_cfg.sensitivity is None:
        raise ValueError("report_cfg.sensitivity must be set for benchmarking rendering")

    sensitivity = analysis.sensitivity
    df_setup = sensitivity.df_setup.copy()

    df_setup = _ensure_n_devices_column(df_setup, independent_var)

    if independent_var not in df_setup.columns:
        raise ValueError(
            f"{independent_var} not in sensitivity CSV columns: {list(df_setup.columns)}"
        )

    dependent_var = report_cfg.sensitivity.dependent_var
    group_by_var = report_cfg.sensitivity.group_by_var

    rows, source_paths = _collect_rows(analysis, dependent_var)
    if not rows:
        raise RuntimeError(
            f"No data for benchmarking {independent_var} vs {dependent_var}"
        )

    df = pd.DataFrame(rows)
    df["wallclock_s"] = df["value"]
    df["wallclock_hr"] = df["wallclock_s"] / 3600.0
    df["indep_value"] = df["sa_id"].map(df_setup[independent_var])
    df["n_devices"] = df["sa_id"].map(df_setup["n_devices"])
    df["compute_hr"] = df["wallclock_hr"] * df["n_devices"]
    if group_by_var is not None:
        if group_by_var not in df_setup.columns:
            raise ValueError(
                f"group_by_var {group_by_var!r} not in sensitivity CSV columns: "
                f"{list(df_setup.columns)}"
            )
        df["group_value"] = df["sa_id"].map(df_setup[group_by_var])
    else:
        df["group_value"] = "all"
    df["n_mpi_procs"] = df["sa_id"].map(df_setup["n_mpi_procs"])

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
    if static_backend == "plotly":
        # Pre-compute speedup + efficiency (same call sites as matplotlib branch).
        speedup_pg = _compute_speedup_per_group(
            df, t_col="wallclock_s", indep_col="n_devices",
            group_col="group_value", baseline_mode="global",
        )
        strong_eff_pg = _compute_efficiency_per_group(
            df, t_col="wallclock_s", indep_col="n_devices",
            group_col="group_value", mode="strong", baseline_mode="global",
        )
        if analysis.cfg_analysis.sensitivity_analysis is not None:
            source_paths.append(Path(analysis.cfg_analysis.sensitivity_analysis))
        return _render_plotly_branch(
            df, speedup_pg, strong_eff_pg,
            wall_unit=wall_unit, cost_unit=cost_unit,
            independent_var=independent_var, group_by_var=group_by_var,
            sens_cfg=sens_cfg,
            output_path=output_path, source_paths=source_paths,
            analysis_dir=analysis.analysis_paths.analysis_dir,
            plotly_js_mode=report_cfg.interactive.plotly_js_mode,
            prov=prov,
        )

    fig, (ax_wall, ax_cost, ax_speedup, ax_eff) = plt.subplots(
        4, 1, figsize=tuple(sens_cfg.figsize_inches), sharex=True
    )
    _draw_panel(ax_wall, df, y_col="wallclock_disp", group_by_var=group_by_var, sens_cfg=sens_cfg, prov=prov)
    _draw_panel(ax_cost, df, y_col="compute_disp", group_by_var=group_by_var, sens_cfg=sens_cfg, prov=prov)

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
        sens_cfg=sens_cfg, prov=prov,
    )
    _draw_metric_panel(
        ax_eff, strong_eff_per_group, df=df,
        x_max=df["n_devices"].max(),
        ideal_kind="constant", ideal_value=1.0, ideal_label="Ideal efficiency (=1.0)",
        sens_cfg=sens_cfg, prov=prov,
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
            ax.grid(True, which="major", axis="both", color=sens_cfg.gridline_color, linewidth=sens_cfg.gridline_width, zorder=0)

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
        dpi=report_cfg.figure_defaults.savefig_dpi,
        output_format="svg",
        provenance=prov,
    )


def _ensure_n_devices_column(df_setup: pd.DataFrame, independent_var: str) -> pd.DataFrame:
    """Derive ``n_devices`` from ``run_mode`` × n_gpus / (n_mpi × n_omp × n_nodes) if absent."""
    if "n_devices" in df_setup.columns:
        return df_setup
    required = {"n_mpi_procs", "n_omp_threads", "n_gpus", "n_nodes"}
    missing = required - set(df_setup.columns)
    if missing:
        if independent_var == "n_devices":
            raise ValueError(
                "Cannot derive n_devices: sensitivity CSV is missing required columns "
                f"{sorted(missing)}. Either declare n_devices explicitly or include the "
                "missing columns."
            )
        return df_setup
    is_gpu = (df_setup.get("run_mode", "").astype(str).str.lower() == "gpu") | (df_setup["n_gpus"] > 0)
    df_setup = df_setup.assign(
        n_devices=df_setup["n_gpus"].where(
            is_gpu,
            df_setup["n_mpi_procs"] * df_setup["n_omp_threads"] * df_setup["n_nodes"],
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


def _compute_speedup_per_group(
    df: pd.DataFrame, *, t_col: str, indep_col: str, group_col: str,
    baseline_mode: str = "per_group",
) -> dict[str, list[tuple[float, float]]]:
    """Compute strong-scaling speedup S(N) = t_baseline / t(N) for each group.

    ``baseline_mode='per_group'``: each group anchors against its own N=1 wallclock
    (groups without N=1 are excluded — no anchor available).

    ``baseline_mode='global'``: all groups share a single anchor — the minimum
    wallclock at the smallest N across all groups. Groups without an N=1 entry
    are still included; their points are normalized against the global anchor.

    When a group has multiple sa rows at the same N, the minimum-wallclock entry
    wins (best configuration at that resource level).
    """
    if baseline_mode not in ("per_group", "global"):
        raise ValueError(f"baseline_mode must be 'per_group' or 'global'; got {baseline_mode!r}")
    if df.empty:
        return {}
    global_anchor = (
        _resolve_global_baseline(df, t_col=t_col, indep_col=indep_col)
        if baseline_mode == "global" else None
    )
    if baseline_mode == "global" and global_anchor is None:
        return {}
    out: dict[str, list[tuple[float, float]]] = {}
    for group_value, sub in df.groupby(group_col):
        per_n_min = sub.groupby(indep_col)[t_col].min()
        if baseline_mode == "per_group":
            if 1 not in per_n_min.index:
                continue
            anchor = float(per_n_min.loc[1])
        else:
            anchor = global_anchor  # type: ignore[assignment]
        if anchor is None or anchor <= 0:
            continue
        pts = [(int(n) if float(n).is_integer() else float(n), anchor / float(t))
               for n, t in per_n_min.items()]
        pts.sort(key=lambda r: r[0])
        out[str(group_value)] = pts
    return out


def _compute_efficiency_per_group(
    df: pd.DataFrame, *, t_col: str, indep_col: str, group_col: str, mode: str,
    baseline_mode: str = "per_group",
) -> dict[str, list[tuple[float, float]]]:
    """Compute scaling efficiency for each group.

    - ``mode='strong'``: E_s(N) = S(N) / N = t_baseline / (N × t(N)). Ideal = 1.0.
    - ``mode='weak'``: E_w(N) = t_baseline / t(N). Ideal = 1.0.

    ``baseline_mode`` matches :func:`_compute_speedup_per_group` semantics.
    """
    if mode not in ("strong", "weak"):
        raise ValueError(f"mode must be 'strong' or 'weak'; got {mode!r}")
    if baseline_mode not in ("per_group", "global"):
        raise ValueError(f"baseline_mode must be 'per_group' or 'global'; got {baseline_mode!r}")
    if df.empty:
        return {}
    global_anchor = (
        _resolve_global_baseline(df, t_col=t_col, indep_col=indep_col)
        if baseline_mode == "global" else None
    )
    if baseline_mode == "global" and global_anchor is None:
        return {}
    out: dict[str, list[tuple[float, float]]] = {}
    for group_value, sub in df.groupby(group_col):
        per_n_min = sub.groupby(indep_col)[t_col].min()
        if baseline_mode == "per_group":
            if 1 not in per_n_min.index:
                continue
            anchor = float(per_n_min.loc[1])
        else:
            anchor = global_anchor  # type: ignore[assignment]
        if anchor is None or anchor <= 0:
            continue
        pts: list[tuple[float, float]] = []
        for n_val, t in per_n_min.items():
            n = int(n_val) if float(n_val).is_integer() else float(n_val)
            tN = float(t)
            if tN <= 0:
                continue
            if mode == "strong":
                eff = anchor / (n * tN)
            else:
                eff = anchor / tN
            pts.append((n, eff))
        pts.sort(key=lambda r: r[0])
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
        color = sens_cfg.palette[i % len(sens_cfg.palette)]
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
                xs, ys, color=color, marker=sens_cfg.cpu_marker, s=sens_cfg.point_size,
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
            ax.plot([1, x_max], [1, x_max], color=sens_cfg.ideal_line_color, linewidth=sens_cfg.ideal_line_width, zorder=2, label=ideal_label)
    elif ideal_kind == "constant":
        ax.axhline(ideal_value, color=sens_cfg.ideal_line_color, linewidth=sens_cfg.ideal_line_width, zorder=2, label=ideal_label)
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


def _draw_panel(ax, df: pd.DataFrame, *, y_col: str, group_by_var: str | None, sens_cfg, prov: ProvenanceLog) -> None:
    """Draw one panel of the dual-panel benchmarking figure."""
    groups = sorted(df["group_value"].dropna().unique(), key=str)
    for i, gv in enumerate(groups):
        sub = df[df["group_value"] == gv].sort_values("indep_value")
        color = sens_cfg.palette[i % len(sens_cfg.palette)]
        is_gpu_group = str(gv).lower() == "gpu"
        is_hybrid_group = str(gv).lower() == "hybrid"
        marker = sens_cfg.gpu_marker if is_gpu_group else sens_cfg.cpu_marker
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

    datatree_path = analysis.analysis_paths.sensitivity_datatree_zarr
    tree: xr.DataTree | None = None
    if datatree_path is not None and datatree_path.exists():
        tree = xr.open_datatree(str(datatree_path), engine="zarr", consolidated=False)
        source_paths.append(datatree_path)

    for sa_id, sub_analysis in sensitivity.sub_analyses.items():
        node_ds = _find_perf_node(tree, sa_id) if tree is not None else None
        if node_ds is not None and col in node_ds.data_vars:
            for event_iloc in sub_analysis.df_sims.index:
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
    for model_subpath in ("tritonswmm/performance", "triton/performance"):
        path = f"/sa_{sa_id}/{model_subpath}"
        try:
            return tree[path].ds
        except KeyError:
            continue
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
) -> Path:
    """Plotly MV port (pre-/design-figure): static 4-panel benchmarking figure.
    Wall-clock | Compute-cost | Strong-scaling speedup | Parallel efficiency,
    stacked rows=4, cols=1 with shared x-axis. One trace per group_by_var value
    per panel, sharing the Okabe-Ito palette (sens_cfg.palette) as Plotly's
    colorway. Informationally congruent with the matplotlib branch — no hover
    refinement, no line-toggle UX, no per-panel zoom/pan customization.
    """
    # Side-effect import: registers `triton_journal` Plotly template.
    from TRITON_SWMM_toolkit.report_renderers import _plotly_theme  # noqa: F401

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        vertical_spacing=0.045,
        subplot_titles=(
            sens_cfg.title or "Sensitivity benchmarking",
            "Compute cost",
            "Strong-scaling speedup",
            "Parallel efficiency",
        ),
    )
    fig.update_layout(
        template="plotly_white",
        colorway=list(sens_cfg.palette),
        showlegend=True,
        legend=dict(
            title=group_by_var if group_by_var is not None else "",
            orientation="v", yanchor="top", y=1.0, xanchor="left", x=1.02,
        ),
        margin=dict(l=10, r=120, t=80, b=80),
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
        )

    # ---- Panels 3 + 4: speedup + efficiency -----------------------------
    _plotly_metric_panel_precomputed(
        fig, speedup_per_group, df_for_groups=df, row=3,
        panel_id="ax_speedup_plotly",
        ideal_kind="linear", x_max=float(df["n_devices"].max()),
        ideal_label="Ideal speedup (S=N)",
        sens_cfg=sens_cfg, prov=prov, show_in_legend=False,
    )
    _plotly_metric_panel_precomputed(
        fig, strong_eff_per_group, df_for_groups=df, row=4,
        panel_id="ax_efficiency_plotly",
        ideal_kind="constant", ideal_value=1.0, x_max=float(df["n_devices"].max()),
        ideal_label="Ideal efficiency (=1.0)",
        sens_cfg=sens_cfg, prov=prov, show_in_legend=False,
    )

    # ---- Axis labels + tickers ------------------------------------------
    xlabel_text = sens_cfg.independent_var_labels.get(independent_var, independent_var)
    fig.update_xaxes(title_text="", row=1, col=1)
    fig.update_xaxes(title_text="", row=2, col=1)
    fig.update_xaxes(title_text="", row=3, col=1)
    fig.update_xaxes(title_text=xlabel_text, row=4, col=1)
    fig.update_yaxes(title_text=f"Wall-clock ({wall_unit})", row=1, col=1)
    fig.update_yaxes(title_text=f"Compute cost ({cost_unit} × devices)", row=2, col=1)
    fig.update_yaxes(title_text="Strong-scaling speedup S(N) = t(1) / t(N)", row=3, col=1)
    fig.update_yaxes(title_text="Strong-scaling efficiency Es(N) = t(1) / (N · t(N))", row=4, col=1)
    if sens_cfg.show_gridlines:
        for r in range(1, 5):
            fig.update_xaxes(
                showgrid=True, gridcolor=sens_cfg.gridline_color,
                gridwidth=sens_cfg.gridline_width, row=r, col=1,
            )
            fig.update_yaxes(
                showgrid=True, gridcolor=sens_cfg.gridline_color,
                gridwidth=sens_cfg.gridline_width, row=r, col=1,
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
) -> None:
    """Plot one of the wallclock/compute-cost panels (raw data per group)."""
    groups = sorted(df["group_value"].dropna().unique(), key=str)
    for i, gv in enumerate(groups):
        sub = df[df["group_value"] == gv].sort_values("indep_value")
        color = sens_cfg.palette[i % len(sens_cfg.palette)]
        is_gpu_group = str(gv).lower() == "gpu"
        is_single_point_group = (
            str(gv).lower() in {"serial", "single_cpu", "single-cpu"}
            or len(sub) == 1
        )
        marker_symbol = "diamond" if is_gpu_group else "circle"
        if not is_single_point_group:
            per_x_min = sub.groupby("indep_value", as_index=True)[y_col].min().sort_index()
            with prov.artist(
                axes_id=panel_id, kind="line",
                note=f"multi-point line {gv} (panel {panel_id})",
            ) as a:
                a.add_channel("data", ProvenanceRef(source_path="sensitivity_datatree.zarr"))
                fig.add_trace(
                    go.Scatter(
                        x=per_x_min.index, y=per_x_min.values,
                        mode="lines",
                        line=dict(color=color, dash="dash",
                                  width=sens_cfg.line_width),
                        legendgroup=str(gv), name=str(gv),
                        showlegend=False, hoverinfo="skip",
                    ),
                    row=row, col=1,
                )
        with prov.artist(
            axes_id=panel_id, kind="scatter",
            note=f"markers {gv} (panel {panel_id})",
        ) as a:
            a.add_channel("data", ProvenanceRef(source_path="sensitivity_datatree.zarr"))
            fig.add_trace(
                go.Scatter(
                    x=sub["indep_value"], y=sub[y_col],
                    mode="markers",
                    marker=dict(
                        symbol=marker_symbol,
                        size=max(int(sens_cfg.point_size ** 0.5), 6),
                        color=color, line=dict(color="black", width=1.0),
                    ),
                    legendgroup=str(gv), name=str(gv),
                    showlegend=show_in_legend,
                    hovertemplate=(
                        f"<b>{gv}</b><br>"
                        "x: %{x}<br>y: %{y:.3f}<extra></extra>"
                    ),
                ),
                row=row, col=1,
            )


def _plotly_metric_panel_precomputed(
    fig,
    per_group_data: dict,
    *,
    df_for_groups: pd.DataFrame,
    row: int,
    panel_id: str,
    ideal_kind: str,
    x_max: float,
    ideal_label: str,
    sens_cfg,
    prov: ProvenanceLog,
    show_in_legend: bool,
    ideal_value: float = 1.0,
) -> None:
    """Plot speedup / efficiency panel from precomputed per-group data."""
    groups = sorted(df_for_groups["group_value"].dropna().unique(), key=str)
    for i, gv in enumerate(groups):
        if gv not in per_group_data:
            continue
        data = per_group_data[gv]
        xs = data.get("xs") if isinstance(data, dict) else None
        ys = data.get("ys") if isinstance(data, dict) else None
        if xs is None or ys is None:
            continue
        color = sens_cfg.palette[i % len(sens_cfg.palette)]
        is_gpu_group = str(gv).lower() == "gpu"
        marker_symbol = "diamond" if is_gpu_group else "circle"
        with prov.artist(
            axes_id=panel_id, kind="line",
            note=f"metric line {gv} (panel {panel_id})",
        ) as a:
            a.add_channel("data", ProvenanceRef(source_path="sensitivity_datatree.zarr"))
            fig.add_trace(
                go.Scatter(
                    x=xs, y=ys, mode="lines+markers",
                    line=dict(color=color, dash="dash",
                              width=sens_cfg.line_width),
                    marker=dict(
                        symbol=marker_symbol,
                        size=max(int(sens_cfg.point_size ** 0.5), 6),
                        color=color, line=dict(color="black", width=1.0),
                    ),
                    legendgroup=str(gv), name=str(gv),
                    showlegend=show_in_legend,
                    hovertemplate=(
                        f"<b>{gv}</b><br>"
                        "x: %{x}<br>y: %{y:.3f}<extra></extra>"
                    ),
                ),
                row=row, col=1,
            )
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
                    name=ideal_label, showlegend=False,
                    hoverinfo="skip",
                ),
                row=row, col=1,
            )
