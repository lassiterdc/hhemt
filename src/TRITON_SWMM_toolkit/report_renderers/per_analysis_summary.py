"""Per-analysis summary renderer: deterministic workflow-health table as matplotlib SVG.

Default rows: total simulations, n successful / pending / failed, enabled model
types, sensitivity-analysis mode (when applicable). Comprehensive diagnostic
content (continuity errors, performance breakdowns, conduit utilization,
sensitivity benchmarking) lives in the v2 comprehensive report catalog plan;
this table is a workflow-health placeholder for the v1 report.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd

if TYPE_CHECKING:
    from TRITON_SWMM_toolkit.analysis import TRITONSWMM_analysis
    from TRITON_SWMM_toolkit.config.report import report_config


def render(
    analysis: TRITONSWMM_analysis,
    report_cfg: report_config,
    output_path: Path,
) -> Path:
    """Render the analysis summary table to output_path."""
    from TRITON_SWMM_toolkit.report_renderers._figure_emission import (
        emit_plot_with_sources,
    )
    from TRITON_SWMM_toolkit.report_renderers._provenance import (
        ProvenanceLog,
        ProvenanceRef,
    )
    from TRITON_SWMM_toolkit.report_renderers.system_overview import _apply_rcparams

    _apply_rcparams(report_cfg)
    prov = ProvenanceLog()

    # Detect sensitivity-master scope: the analysis is the master sensitivity
    # analysis with sub-analyses populated. In that case, render a per-sa-row
    # table (one row per sub-analysis showing status counts) — Iteration 6
    # "show all sub-analyses" scope. Otherwise fall back to the regular
    # multisim single-scope table.
    is_sensitivity_master = (
        getattr(analysis.cfg_analysis, "toggle_sensitivity_analysis", False)
        and getattr(analysis, "sensitivity", None) is not None
        and len(analysis.sensitivity.sub_analyses) > 0
    )

    metrics = report_cfg.per_analysis_summary.metrics

    if is_sensitivity_master:
        # Multi-row table: one row per sub-analysis with status counts.
        per_sa_rows = []
        for sa_id, sub in analysis.sensitivity.sub_analyses.items():
            n = len(sub.df_sims.index)
            n_succ = sum(
                1 for i in sub.df_sims.index if _is_scenario_successful(sub, i)
            )
            n_pend = sum(
                1 for i in sub.df_sims.index if _is_scenario_pending(sub, i)
            )
            n_fail = n - n_succ - n_pend
            per_sa_rows.append({
                "sub-analysis": sa_id,
                "n sims": n,
                "successful": n_succ,
                "pending": n_pend,
                "failed": n_fail,
            })
        df = pd.DataFrame(per_sa_rows)
    else:
        n_sims = len(analysis.df_sims.index)
        n_successful = sum(
            1 for i in analysis.df_sims.index if _is_scenario_successful(analysis, i)
        )
        n_pending = sum(
            1 for i in analysis.df_sims.index if _is_scenario_pending(analysis, i)
        )
        n_failed = n_sims - n_successful - n_pending

        rows = []
        if "n_sims" in metrics:
            rows.append(("Total simulations", n_sims))
        if "n_successful" in metrics:
            rows.append(("Successful", n_successful))
        if "n_pending" in metrics:
            rows.append(("Pending", n_pending))
        if "n_failed" in metrics:
            rows.append(("Failed", n_failed))
        if "enabled_model_types" in metrics:
            enabled = analysis._get_enabled_model_types()
            rows.append(("Enabled model types", ", ".join(enabled) if enabled else "(none)"))
        if "sensitivity_mode" in metrics:
            sensitivity_cfg = getattr(report_cfg, "sensitivity", None)
            mode = getattr(sensitivity_cfg, "mode", None) if sensitivity_cfg else None
            if mode is not None:
                rows.append(("Sensitivity analysis mode", str(mode)))

        df = pd.DataFrame(rows, columns=["Metric", "Value"])
    fig, ax = plt.subplots(
        figsize=(8, 0.4 * len(df) + 0.7), layout="constrained"
    )
    ax.axis("off")

    # Source paths the parsers will read — captured first so we can record them.
    # In sensitivity-master mode iterate every sub-analysis's scenarios; otherwise
    # iterate the analysis's own df_sims (regular multisim case).
    enabled_model_types = analysis._get_enabled_model_types()
    source_paths: list[Path] = []
    if is_sensitivity_master:
        for sub in analysis.sensitivity.sub_analyses.values():
            for event_iloc in sub.df_sims.index:
                try:
                    scen = sub._retrieve_sim_runs(event_iloc)._scenario
                except Exception:
                    continue
                for mt in enabled_model_types:
                    log_file = scen.scen_paths.sim_folder / f"log_{mt}.json"
                    if log_file.exists():
                        source_paths.append(log_file)
    else:
        for event_iloc in analysis.df_sims.index:
            scen = analysis._retrieve_sim_runs(event_iloc)._scenario
            for mt in enabled_model_types:
                log_file = scen.scen_paths.sim_folder / f"log_{mt}.json"
                if log_file.exists():
                    source_paths.append(log_file)

    analysis_root = str(Path(analysis.analysis_paths.analysis_dir).resolve())
    import os

    rel_sources = [
        os.path.relpath(str(Path(p).resolve()), analysis_root) for p in source_paths
    ]

    with prov.artist(
        axes_id="ax_summary",
        kind="table",
        note="per-analysis workflow-health table (status counts + enabled model types)",
    ) as a:
        for rel in rel_sources:
            a.add_channel(
                "other",
                ProvenanceRef(
                    source_path=rel,
                    variable="run_complete",
                    attrs={},
                    transform="counted across event_iloc by status",
                ),
            )
        table = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            cellLoc="left",
            loc="center",
        )
    table.auto_set_font_size(False)
    table.set_fontsize(report_cfg.figure_defaults.font_size)
    table.scale(1, 1.5)

    return emit_plot_with_sources(
        fig,
        output_path,
        source_paths,
        analysis_dir=analysis.analysis_paths.analysis_dir,
        dpi=report_cfg.figure_defaults.savefig_dpi,
        output_format="svg",
        provenance=prov,
    )


def _is_scenario_successful(analysis, event_iloc: int) -> bool:
    scen = analysis._retrieve_sim_runs(event_iloc)._scenario
    return all(
        scen.model_run_completed(mt) for mt in analysis._get_enabled_model_types()
    )


def _is_scenario_pending(analysis, event_iloc: int) -> bool:
    scen = analysis._retrieve_sim_runs(event_iloc)._scenario
    return not any(
        scen.model_run_completed(mt) for mt in analysis._get_enabled_model_types()
    )
