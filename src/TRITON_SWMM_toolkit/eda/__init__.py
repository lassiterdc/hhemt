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

from TRITON_SWMM_toolkit.eda._plotting import render_eda_plots
from TRITON_SWMM_toolkit.eda._report import assemble_eda_report
from TRITON_SWMM_toolkit.eda._result import EdaReportResult, EdaResult
from TRITON_SWMM_toolkit.eda.cross_sim_identity import check_cross_sim_identity

__all__ = [
    "EdaReportResult",
    "EdaResult",
    "assemble_eda_report",
    "check_cross_sim_identity",
    "render_eda_plots",
]
