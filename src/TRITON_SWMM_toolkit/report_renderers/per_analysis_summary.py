"""Per-analysis summary renderer: workflow-health table as a Tabulator data grid (HTML).

Migrated from a matplotlib `ax.table(...)` + SVG output to a self-contained
Tabulator HTML emit at Phase 6 of the interactive_report_renderers feature.
Default rows in regular multisim mode: total simulations, expected total,
n successful / pending / failed, enabled model types, sensitivity-analysis
mode (when applicable). Sensitivity-master mode renders one row per
sub-analysis with status counts.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from TRITON_SWMM_toolkit.report_renderers._tabulator_defaults import (
    build_columns_spec,
    build_html_document,
    build_options_dict,
)

if TYPE_CHECKING:
    from TRITON_SWMM_toolkit.analysis import TRITONSWMM_analysis
    from TRITON_SWMM_toolkit.config.report import report_config


def render(
    analysis: TRITONSWMM_analysis,
    report_cfg: report_config,
    output_path: Path,
) -> Path:
    """Render the analysis summary table to output_path as a Tabulator data grid."""
    from TRITON_SWMM_toolkit.report_renderers._figure_emission import (
        emit_plot_with_sources,
    )
    from TRITON_SWMM_toolkit.report_renderers._provenance import (
        ProvenanceLog,
        ProvenanceRef,
    )

    prov = ProvenanceLog()

    # Sensitivity-master scope: the analysis is the master sensitivity analysis
    # with sub-analyses populated. Render a per-sa-row table (one row per
    # sub-analysis showing status counts). Otherwise fall back to the regular
    # multisim single-scope metrics table.
    is_sensitivity_master = (
        getattr(analysis.cfg_analysis, "toggle_sensitivity_analysis", False)
        and getattr(analysis, "sensitivity", None) is not None
        and len(analysis.sensitivity.sub_analyses) > 0
    )

    metrics = report_cfg.per_analysis_summary.metrics

    if is_sensitivity_master:
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
        # Snakemake onsuccess/onerror hook). Fall back to per-log iteration
        # if the CSV is missing.
        scenario_status_csv = (
            Path(analysis.analysis_paths.analysis_dir) / "scenario_status.csv"
        )
        if scenario_status_csv.exists():
            status_df = pd.read_csv(scenario_status_csv)
            success_per_event = status_df.groupby("event_iloc")[
                "run_completed"
            ].all()
            n_successful = int(success_per_event.sum())
            failed_mask = (
                ~status_df["run_completed"].fillna(False).astype(bool)
            ) & status_df["scenario_setup"].fillna(False).astype(bool)
            failed_events = status_df[failed_mask]["event_iloc"].unique()
            failed_events = [
                e for e in failed_events if not success_per_event.get(e, False)
            ]
            n_failed = len(failed_events)
            n_pending = max(0, n_sims - n_successful - n_failed)
        else:
            n_successful = sum(
                1
                for i in analysis.df_sims.index
                if _is_scenario_successful(analysis, i)
            )
            n_pending = sum(
                1
                for i in analysis.df_sims.index
                if _is_scenario_pending(analysis, i)
            )
            n_failed = n_sims - n_successful - n_pending

        n_weather_events = len(analysis.df_sims.index)
        is_sensitivity = (
            getattr(analysis.cfg_analysis, "toggle_sensitivity_analysis", False)
            and getattr(analysis, "sensitivity", None) is not None
        )
        if is_sensitivity:
            n_sa_rows = len(analysis.sensitivity.df_setup.index)
            expected_total = n_weather_events * n_sa_rows
            expected_label = (
                f"Expected total (derived: {n_weather_events} events "
                f"× {n_sa_rows} sa rows)"
            )
        else:
            expected_total = n_weather_events
            expected_label = (
                f"Expected total (derived: {n_weather_events} weather events)"
            )

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
            rows.append(
                ("Enabled model types", ", ".join(enabled) if enabled else "(none)")
            )
        if "sensitivity_mode" in metrics:
            sensitivity_cfg = getattr(report_cfg, "sensitivity", None)
            mode = (
                getattr(sensitivity_cfg, "mode", None) if sensitivity_cfg else None
            )
            if mode is not None:
                rows.append(("Sensitivity analysis mode", str(mode)))

        df = pd.DataFrame(rows, columns=["Metric", "Value"])

    # Source-path declarations identical to legacy matplotlib renderer. Regular
    # multisim mode reads scenario_status.csv; sensitivity-master mode reads
    # nothing from disk (per-sa counts come from in-memory sub_analyses).
    source_paths: list[Path] = []
    scenario_status_csv_path = (
        Path(analysis.analysis_paths.analysis_dir) / "scenario_status.csv"
    )
    if scenario_status_csv_path.exists():
        source_paths.append(scenario_status_csv_path)

    analysis_root = str(Path(analysis.analysis_paths.analysis_dir).resolve())
    rel_sources = [
        os.path.relpath(str(Path(p).resolve()), analysis_root) for p in source_paths
    ]

    with prov.artist(
        axes_id="ax_summary",
        kind="table",
        note=(
            "per-analysis workflow-health table "
            "(status counts + enabled model types) — Tabulator data grid"
        ),
    ) as a:
        for rel in rel_sources:
            a.add_channel(
                "other",
                ProvenanceRef(
                    source_path=rel,
                    variable="run_completed",
                    attrs={},
                    transform=(
                        "aggregated by event_iloc to derive "
                        "successful/pending/failed counts"
                    ),
                ),
            )

    html_text = _build_tabulator_html(df, report_cfg)

    return emit_plot_with_sources(
        html_text,
        output_path,
        source_paths,
        analysis_dir=analysis.analysis_paths.analysis_dir,
        provenance=prov,
        output_format="html",
    )


def _build_tabulator_html(df: pd.DataFrame, report_cfg: report_config) -> str:
    """Build a self-contained Tabulator HTML document from the workflow-health DataFrame.

    Delegates construction to ``_tabulator_defaults``. No persistence-id
    derivation here — per_analysis_summary's data is fully aggregate (one
    row per metric in regular mode; one row per sub-analysis in
    sensitivity-master mode) so cross-analysis persistence collision is
    not a risk; ``persistence_id`` is wired only when the user explicitly
    sets it via config.
    """
    tab_cfg = report_cfg.per_analysis_summary.interactive

    columns_spec = build_columns_spec(
        df,
        visible_columns_default=tab_cfg.visible_columns_default,
        header_filter=tab_cfg.header_filter,
    )

    options = build_options_dict(
        df,
        columns_spec=columns_spec,
        table_height=tab_cfg.table_height,
        pagination_size=tab_cfg.pagination_size,
        persistence_id=tab_cfg.persistence_id,
    )

    js_mode = getattr(
        getattr(report_cfg, "interactive", None), "tabulator_js_mode", "cdn",
    )

    return build_html_document(
        title="Analysis summary",
        container_id="summary-table",
        body_heading_html="",
        options=options,
        js_mode=js_mode,
        renderer_name="per_analysis_summary",
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
