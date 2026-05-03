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
        # Prefer scenario_status.csv (written by export_scenario_status.py as
        # Snakemake onsuccess/onerror hook) for pending/failed counts —
        # already aggregates the per-log inference. Fall back to per-log
        # iteration if the CSV is missing (e.g., workflow killed before the
        # hook ran).
        scenario_status_csv = Path(analysis.analysis_paths.analysis_dir) / "scenario_status.csv"
        if scenario_status_csv.exists():
            # CSV schema (per export_scenario_status.py): one row per
            # (event_iloc, model_type) with `run_completed` boolean column.
            # No explicit status column — derive: True → success; False with
            # `scenario_setup=True` → failed; everything else → pending.
            status_df = pd.read_csv(scenario_status_csv)
            # Aggregate per scenario (event_iloc): a scenario is successful
            # when ALL its model_type rows have run_completed=True; failed if
            # any row has scenario_setup=True but run_completed=False;
            # pending otherwise.
            success_per_event = status_df.groupby("event_iloc")["run_completed"].all()
            n_successful = int(success_per_event.sum())
            # Failed: at least one model_type row failed (run_completed False
            # but scenario was set up)
            failed_mask = (~status_df["run_completed"].fillna(False).astype(bool)) & status_df["scenario_setup"].fillna(False).astype(bool)
            failed_events = status_df[failed_mask]["event_iloc"].unique()
            # Exclude events that are otherwise successful (some model_types succeeded)
            failed_events = [e for e in failed_events if not success_per_event.get(e, False)]
            n_failed = len(failed_events)
            n_pending = max(0, n_sims - n_successful - n_failed)
        else:
            n_successful = sum(
                1 for i in analysis.df_sims.index if _is_scenario_successful(analysis, i)
            )
            n_pending = sum(
                1 for i in analysis.df_sims.index if _is_scenario_pending(analysis, i)
            )
            n_failed = n_sims - n_successful - n_pending

        # Derived expected total: n_weather_events × n_sensitivity_rows for
        # sensitivity analyses; n_weather_events otherwise. `analysis.df_setup`
        # is NOT a proxy for `analysis.sensitivity.df_setup` — verified by grep
        # of analysis.py + sensitivity_analysis.py. The renderer dispatches on
        # `cfg_analysis.toggle_sensitivity_analysis` to pick the right object.
        n_weather_events = len(analysis.df_sims.index)
        is_sensitivity = (
            getattr(analysis.cfg_analysis, "toggle_sensitivity_analysis", False)
            and getattr(analysis, "sensitivity", None) is not None
        )
        if is_sensitivity:
            n_sa_rows = len(analysis.sensitivity.df_setup.index)
            expected_total = n_weather_events * n_sa_rows
            expected_label = f"Expected total (derived: {n_weather_events} events × {n_sa_rows} sa rows)"
        else:
            expected_total = n_weather_events
            expected_label = f"Expected total (derived: {n_weather_events} weather events)"

        rows = []
        if "n_sims" in metrics:
            rows.append(("Total simulations", n_sims))
            rows.append((expected_label, expected_total))
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
    pas_cfg = report_cfg.per_analysis_summary
    fig, ax = plt.subplots(
        figsize=(
            pas_cfg.figure_width_inches,
            pas_cfg.figure_height_per_row_inches * len(df) + pas_cfg.figure_height_padding_inches,
        ),
        layout="constrained",
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
    table.scale(*report_cfg.per_analysis_summary.table_scale)

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
