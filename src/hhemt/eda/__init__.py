"""eda/ — governed EDA data-prep subpackage (ADR-9).

EDA-supporting data-prep functions live here, one module per EDA family. Each
returns an ``EdaResult`` (a ``CheckResult`` verdict + a manifest-sidecar'd derived
artifact under ``{analysis_dir}/eda/``). The first member is the cross-sim
byte-for-byte identity verification (peak flood depth + conduit flow/ratios across
sims sharing an event iloc on a sensitivity master).

These functions are NOT methods on
``processing_analysis.TRITONSWMM_analysis_post_processing`` (no coupling of EDA
findings to the consolidation DAG) and are NOT pure ``analysis_validation.check_*``
functions (EDA prepares plottable data, not side-effect-free verdicts). The calc is
invoked in-process by the downstream ``analysis.eda()`` facade — there is no
Snakemake rule for the EDA loop.
"""

from __future__ import annotations

from hhemt.eda._context import EdaContext, load_eda_context
from hhemt.eda._plotting import render_eda_plots
from hhemt.eda._promote import (
    promote_eda_plot_to_static_config,
    register_eda_plot_in_reporting_set,
)
from hhemt.eda._report import (
    config_diff_maps_figure_from_root,
    dem_resolution_cost_error_figure_from_root,
    dem_resolution_coupling_table_figure_from_root,
    dem_resolution_diff_maps_figure_from_root,
    dem_resolution_error_ecdf_figure_from_root,
)
from hhemt.eda._result import EdaReportResult, EdaResult
from hhemt.eda.cross_sim_identity import check_cross_sim_identity

__all__ = [
    "EdaContext",
    "EdaReportResult",
    "EdaResult",
    "check_cross_sim_identity",
    "config_diff_maps_figure_from_root",
    "dem_resolution_coupling_table_figure_from_root",
    "dem_resolution_cost_error_figure_from_root",
    "dem_resolution_diff_maps_figure_from_root",
    "dem_resolution_error_ecdf_figure_from_root",
    "load_eda_context",
    "promote_eda_plot_to_static_config",
    "register_eda_plot_in_reporting_set",
    "render_eda_plots",
]
