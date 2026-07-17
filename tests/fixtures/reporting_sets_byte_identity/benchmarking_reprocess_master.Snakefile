# Auto-generated reprocess-scoped master Snakefile for sensitivity analysis
# Re-runs downstream stages (per-sa consolidate + master consolidation +
# plots + render) against existing per-sa sim completion flags. No
# simulation or scenario-preparation rules are emitted.

import os
from datetime import datetime as _dt
from hhemt.report_renderers._figure_emission import format_sources_rst as _fmt_sources_rst

try:
    from importlib.metadata import version as _pkg_version
    _toolkit_version = _pkg_version("hhemt")
except Exception:
    _toolkit_version = "unknown"

config["analysis_id"] = 'synth_sensitivity'
config["toolkit_version"] = _toolkit_version
config["n_sims"] = 4
config["is_sensitivity"] = True
config["n_sub_analyses"] = 4
config["independent_vars"] = ['n_devices']
config["group_by_var"] = 'run_mode'
config["report"] = {"generated_at": _dt.now().isoformat(timespec="seconds")}

SA_EVENT_PAIRS_SA = []
SA_EVENT_PAIRS_EVT = []

report: "report/workflow_description.rst"

onstart:
    shell("mkdir -p _status {PYTEST_TMP}/test_reprocess_master_byte_ide0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/logs/sims {PYTEST_TMP}/test_reprocess_master_byte_ide0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/logs")

onerror:
    shell("""
        python -m hhemt.export_scenario_status \
            --system-config {PYTEST_TMP}/test_reprocess_master_byte_ide0/synthetic_test_runs/synth_sensitivity/system_config.yaml \
            --analysis-config {PYTEST_TMP}/test_reprocess_master_byte_ide0/synthetic_test_runs/synth_sensitivity/analysis_config.yaml \
            > {PYTEST_TMP}/test_reprocess_master_byte_ide0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/logs/export_scenario_status.log 2>&1
    """)


rule all:
    input:
        "_status/f_consolidate_master_complete.flag", "plots/system_overview.html", "plots/per_analysis/summary_table.html", "plots/appendix/scenario_status.html", "plots/errors_and_warnings/validation_report.html", "plots/disk_utilization.html", "plots/metadata.html", "scenario_status.csv", "workflow_summary.md", expand("plots/sensitivity/benchmarking/benchmarking__{independent_var}.vs.total.html", independent_var=['n_devices']), "analysis_report.zip"

rule master_consolidation:
    input: 
    output: "_status/f_consolidate_master_complete.flag"
    log: "{PYTEST_TMP}/test_reprocess_master_byte_ide0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/logs/master_consolidation.log"
    conda: "{REPO_ROOT}/workflow/envs/hhemt.yaml"
    resources:
        slurm_partition="None",
        runtime=30,
        tasks=1,
        cpus_per_task=1,
        mem_mb=12000,
        nodes=1
    shell:
        """
        mkdir -p {PYTEST_TMP}/test_reprocess_master_byte_ide0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/logs/sims {PYTEST_TMP}/test_reprocess_master_byte_ide0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/logs _status
        python -m hhemt.consolidate_workflow \
            --system-config {PYTEST_TMP}/test_reprocess_master_byte_ide0/synthetic_test_runs/synth_sensitivity/system_config.yaml \
            --analysis-config {PYTEST_TMP}/test_reprocess_master_byte_ide0/synthetic_test_runs/synth_sensitivity/analysis_config.yaml \
            --consolidate-sensitivity-analysis-outputs \
            --allow-incomplete \
            --which both \
            --compression-level 5 \
            --flag-output {output} \
            --rule-name master_consolidation \
            > {log} 2>&1
        """

rule plot_system_overview:
    input:
        consolidated = "_status/f_consolidate_master_complete.flag",
    output:
        report(
            "plots/system_overview.html",
            caption="report/captions/system_map.rst",
            category="System Information",
            labels={"figure": "System map"},
        )
    params:
        source_paths = [{'path': '../elevation_10.00m.dem', 'variables': []}, {'path': 'subanalyses/sa_0/sims/event_index.0/swmm/hydro.inp', 'variables': ['[SUBCATCHMENTS]', '[JUNCTIONS]', '[OUTFALLS]']}, {'path': 'subanalyses/sa_0/sims/event_index.0/swmm/hydraulics.inp', 'variables': ['[CONDUITS]', '[JUNCTIONS]', '[POLYGONS]']}, {'path': '../../../../../../..{SYNTH_MODELS}/cba1103fcbb013fa/boundary.geojson', 'variables': []}],
        source_paths_rst = '- ``../elevation_10.00m.dem``\n\n- ``subanalyses/sa_0/sims/event_index.0/swmm/hydro.inp``\n\n  - ``[SUBCATCHMENTS]``\n  - ``[JUNCTIONS]``\n  - ``[OUTFALLS]``\n\n- ``subanalyses/sa_0/sims/event_index.0/swmm/hydraulics.inp``\n\n  - ``[CONDUITS]``\n  - ``[JUNCTIONS]``\n  - ``[POLYGONS]``\n\n- ``../../../../../../..{SYNTH_MODELS}/cba1103fcbb013fa/boundary.geojson``\n',
    log: "logs/plots/system_overview.log"
    conda: "{REPO_ROOT}/workflow/envs/hhemt.yaml"
    resources: mem_mb=2000, time_min=10
    shell:
        """
        {PYTHON} -m hhemt.report_renderers._cli system_overview \
            --system-config {PYTEST_TMP}/test_reprocess_master_byte_ide0/synthetic_test_runs/synth_sensitivity/system_config.yaml \
            --analysis-config {PYTEST_TMP}/test_reprocess_master_byte_ide0/synthetic_test_runs/synth_sensitivity/analysis_config.yaml \
            --output {output} \
            > {log} 2>&1
        """

rule plot_per_analysis_summary_table:
    input:
        "scenario_status.csv",
        consolidated = "_status/f_consolidate_master_complete.flag",
    output:
        report(
            "plots/per_analysis/summary_table.html",
            caption="report/captions/per_analysis_summary_table.rst",
            category="Workflow Status",
            subcategory="Workflow Health Summary",
            labels={"figure": "Summary table"},
        )
    params:
        source_paths = [{'path': 'subanalyses/sa_0/sims/event_index.0/out_tritonswmm/swmm/hydraulics.rpt', 'variables': ['Flow Routing Continuity error (%)']}, {'path': 'subanalyses/sa_0/sims/event_index.0/log_tritonswmm.json', 'variables': ['model_run_completed[tritonswmm] (status flag for n_successful / n_pending counts)']}, {'path': 'subanalyses/sa_1/sims/event_index.0/out_tritonswmm/swmm/hydraulics.rpt', 'variables': ['Flow Routing Continuity error (%)']}, {'path': 'subanalyses/sa_1/sims/event_index.0/log_tritonswmm.json', 'variables': ['model_run_completed[tritonswmm] (status flag for n_successful / n_pending counts)']}, {'path': 'subanalyses/sa_2/sims/event_index.0/out_tritonswmm/swmm/hydraulics.rpt', 'variables': ['Flow Routing Continuity error (%)']}, {'path': 'subanalyses/sa_2/sims/event_index.0/log_tritonswmm.json', 'variables': ['model_run_completed[tritonswmm] (status flag for n_successful / n_pending counts)']}, {'path': 'subanalyses/sa_3/sims/event_index.0/out_tritonswmm/swmm/hydraulics.rpt', 'variables': ['Flow Routing Continuity error (%)']}, {'path': 'subanalyses/sa_3/sims/event_index.0/log_tritonswmm.json', 'variables': ['model_run_completed[tritonswmm] (status flag for n_successful / n_pending counts)']}],
        source_paths_rst = '- ``subanalyses/sa_0/sims/event_index.0/out_tritonswmm/swmm/hydraulics.rpt``\n\n  - ``Flow Routing Continuity error (%)``\n\n- ``subanalyses/sa_0/sims/event_index.0/log_tritonswmm.json``\n\n  - ``model_run_completed[tritonswmm] (status flag for n_successful / n_pending counts)``\n\n- ``subanalyses/sa_1/sims/event_index.0/out_tritonswmm/swmm/hydraulics.rpt``\n\n  - ``Flow Routing Continuity error (%)``\n\n- ``subanalyses/sa_1/sims/event_index.0/log_tritonswmm.json``\n\n  - ``model_run_completed[tritonswmm] (status flag for n_successful / n_pending counts)``\n\n- ``subanalyses/sa_2/sims/event_index.0/out_tritonswmm/swmm/hydraulics.rpt``\n\n  - ``Flow Routing Continuity error (%)``\n\n- ``subanalyses/sa_2/sims/event_index.0/log_tritonswmm.json``\n\n  - ``model_run_completed[tritonswmm] (status flag for n_successful / n_pending counts)``\n\n- ``subanalyses/sa_3/sims/event_index.0/out_tritonswmm/swmm/hydraulics.rpt``\n\n  - ``Flow Routing Continuity error (%)``\n\n- ``subanalyses/sa_3/sims/event_index.0/log_tritonswmm.json``\n\n  - ``model_run_completed[tritonswmm] (status flag for n_successful / n_pending counts)``\n',
    log: "logs/plots/per_analysis_summary_table.log"
    conda: "{REPO_ROOT}/workflow/envs/hhemt.yaml"
    resources: mem_mb=2000, time_min=5
    shell:
        """
        {PYTHON} -m hhemt.report_renderers._cli per_analysis_summary \
            --system-config {PYTEST_TMP}/test_reprocess_master_byte_ide0/synthetic_test_runs/synth_sensitivity/system_config.yaml \
            --analysis-config {PYTEST_TMP}/test_reprocess_master_byte_ide0/synthetic_test_runs/synth_sensitivity/analysis_config.yaml \
            --output {output} \
            > {log} 2>&1
        """

rule plot_scenario_status_appendix:
    input:
        "scenario_status.csv",
        consolidated = "_status/f_consolidate_master_complete.flag",
    output:
        report(
            "plots/appendix/scenario_status.html",
            caption="report/captions/scenario_status_appendix.rst",
            category="Appendix",
            subcategory="Scenario Status",
            labels={"figure": "Per-scenario status table"},
        )
    params:
        source_paths = [{'path': 'scenario_status.csv', 'variables': ['event_id', 'model_type', 'status', 'runtime_s', 'continuity_error_pct', 'notes']}],
        source_paths_rst = '- ``scenario_status.csv``\n\n  - ``event_id``\n  - ``model_type``\n  - ``status``\n  - ``runtime_s``\n  - ``continuity_error_pct``\n  - ``notes``\n',
    log: "logs/plots/scenario_status_appendix.log"
    conda: "{REPO_ROOT}/workflow/envs/hhemt.yaml"
    resources: mem_mb=1000, time_min=5
    shell:
        """
        {PYTHON} -m hhemt.report_renderers._cli scenario_status_appendix \
            --system-config {PYTEST_TMP}/test_reprocess_master_byte_ide0/synthetic_test_runs/synth_sensitivity/system_config.yaml \
            --analysis-config {PYTEST_TMP}/test_reprocess_master_byte_ide0/synthetic_test_runs/synth_sensitivity/analysis_config.yaml \
            --output {output} \
            > {log} 2>&1
        """

rule plot_errors_and_warnings:
    input:
        "scenario_status.csv",
        consolidated = "_status/f_consolidate_master_complete.flag",
    output:
        report(
            "plots/errors_and_warnings/validation_report.html",
            caption="report/captions/errors_and_warnings.rst",
            category="Errors and Warnings",
            subcategory="Validation Report",
            labels={"figure": "Validation report"},
        )
    params:
        source_paths = [{'path': 'scenario_status.csv', 'variables': ['scenario_setup', 'run_completed', 'actual_nTasks', 'actual_omp_threads', 'actual_total_gpus', 'actual_gpu_backend']}, {'path': 'sims/<event_id>/log_<model_type>.json', 'variables': ['simulation_completed (per scenario × model_type)']}, {'path': '../system_log.json', 'variables': ['compilation_successful', 'compilation_triton_only_successful', 'compilation_swmm_successful']}],
        source_paths_rst = '- ``scenario_status.csv``\n\n  - ``scenario_setup``\n  - ``run_completed``\n  - ``actual_nTasks``\n  - ``actual_omp_threads``\n  - ``actual_total_gpus``\n  - ``actual_gpu_backend``\n\n- ``sims/<event_id>/log_<model_type>.json``\n\n  - ``simulation_completed (per scenario × model_type)``\n\n- ``../system_log.json``\n\n  - ``compilation_successful``\n  - ``compilation_triton_only_successful``\n  - ``compilation_swmm_successful``\n',
    log: "logs/plots/errors_and_warnings.log"
    conda: "{REPO_ROOT}/workflow/envs/hhemt.yaml"
    resources: mem_mb=1000, time_min=5
    shell:
        """
        {PYTHON} -m hhemt.report_renderers._cli errors_and_warnings \
            --system-config {PYTEST_TMP}/test_reprocess_master_byte_ide0/synthetic_test_runs/synth_sensitivity/system_config.yaml \
            --analysis-config {PYTEST_TMP}/test_reprocess_master_byte_ide0/synthetic_test_runs/synth_sensitivity/analysis_config.yaml \
            --output {output} \
            > {log} 2>&1
        """

rule plot_disk_utilization:
    input:
        consolidated = "_status/f_consolidate_master_complete.flag",
    output:
        report(
            "plots/disk_utilization.html",
            caption="report/captions/disk_utilization.rst",
            category="System Information",
            labels={"figure": "Disk Utilization"},
        )
    params:
        source_paths = [{'path': '_status/_du.json', 'variables': ['disk_utilization_bytes', 'sub_path_breakdown']}],
        source_paths_rst = '- ``_status/_du.json``\n\n  - ``disk_utilization_bytes``\n  - ``sub_path_breakdown``\n',
    log: "logs/plots/disk_utilization.log"
    conda: "{REPO_ROOT}/workflow/envs/hhemt.yaml"
    resources: mem_mb=1000, time_min=5
    shell:
        """
        {PYTHON} -m hhemt.report_renderers._cli disk_utilization \
            --system-config {PYTEST_TMP}/test_reprocess_master_byte_ide0/synthetic_test_runs/synth_sensitivity/system_config.yaml \
            --analysis-config {PYTEST_TMP}/test_reprocess_master_byte_ide0/synthetic_test_runs/synth_sensitivity/analysis_config.yaml \
            --output {output} \
            > {log} 2>&1
        """

rule plot_metadata:
    input:
        consolidated = "_status/f_consolidate_master_complete.flag",
    output:
        report(
            "plots/metadata.html",
            caption="report/captions/metadata.rst",
            category="Metadata",
            labels={"figure": "Metadata"},
        )
    params:
        source_paths = [{'path': 'ro-crate-metadata.json', 'variables': ['provenance']}],
        source_paths_rst = '- ``ro-crate-metadata.json``\n\n  - ``provenance``\n',
    log: "logs/plots/metadata.log"
    conda: "{REPO_ROOT}/workflow/envs/hhemt.yaml"
    resources: mem_mb=1000, time_min=5
    shell:
        """
        {PYTHON} -m hhemt.report_renderers._cli metadata \
            --system-config {PYTEST_TMP}/test_reprocess_master_byte_ide0/synthetic_test_runs/synth_sensitivity/system_config.yaml \
            --analysis-config {PYTEST_TMP}/test_reprocess_master_byte_ide0/synthetic_test_runs/synth_sensitivity/analysis_config.yaml \
            --output {output} \
            > {log} 2>&1
        """

localrules: export_scenario_status

rule export_scenario_status:
    input: "_status/f_consolidate_master_complete.flag"
    output:
        csv = "scenario_status.csv",
        md  = "workflow_summary.md",
    log: "{PYTEST_TMP}/test_reprocess_master_byte_ide0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/logs/export_scenario_status.log"
    conda: "{REPO_ROOT}/workflow/envs/hhemt.yaml"
    resources:
        slurm_partition="None",
        runtime=10,
        tasks=1,
        cpus_per_task=1,
        mem_mb=1000,
        nodes=1
    shell:
        """
        {PYTHON} -m hhemt.export_scenario_status \
            --system-config {PYTEST_TMP}/test_reprocess_master_byte_ide0/synthetic_test_runs/synth_sensitivity/system_config.yaml \
            --analysis-config {PYTEST_TMP}/test_reprocess_master_byte_ide0/synthetic_test_runs/synth_sensitivity/analysis_config.yaml \
            > {log} 2>&1
        """

INDEPENDENT_VARS = ['n_devices']

def _sensitivity_source_paths(wildcards):
    from hhemt.report_renderers._figure_emission import (
        collect_sensitivity_source_paths,
    )
    return collect_sensitivity_source_paths(
        wildcards.independent_var,
        swmm_only_rpt_rel_paths=[],
    )

rule plot_sensitivity_benchmarking:
    input:
        master = "_status/f_consolidate_master_complete.flag",
    output:
        report(
            "plots/sensitivity/benchmarking/benchmarking__{independent_var}.vs.total.html",
            caption="report/captions/sensitivity_benchmarking.rst",
            category="Key Results",
            subcategory="Benchmarking",
            labels={"independent_var": "{independent_var}", "figure": "vs Total runtime"},
        )
    wildcard_constraints:
        independent_var="[A-Za-z0-9_.]+",
    params:
        source_paths = _sensitivity_source_paths,
        source_paths_rst = lambda w: _fmt_sources_rst(_sensitivity_source_paths(w)),
    log: "logs/plots/sensitivity_benchmarking_{independent_var}.log"
    conda: "{REPO_ROOT}/workflow/envs/hhemt.yaml"
    resources: mem_mb=4000, time_min=10
    shell:
        """
        {PYTHON} -m hhemt.report_renderers._cli sensitivity_benchmarking \
            --system-config {PYTEST_TMP}/test_reprocess_master_byte_ide0/synthetic_test_runs/synth_sensitivity/system_config.yaml \
            --analysis-config {PYTEST_TMP}/test_reprocess_master_byte_ide0/synthetic_test_runs/synth_sensitivity/analysis_config.yaml \
            --independent-var {wildcards.independent_var} \
            --output {output} \
            > {log} 2>&1
        """

rule render_report:
    input:
        "plots/system_overview.html",
        "plots/per_analysis/summary_table.html",
        "plots/appendix/scenario_status.html",
        "plots/errors_and_warnings/validation_report.html",
        "plots/disk_utilization.html",
        "plots/metadata.html",
        "scenario_status.csv",
        "workflow_summary.md",
        expand("plots/sensitivity/benchmarking/benchmarking__{independent_var}.vs.total.html", independent_var=['n_devices'])
    output:
        "analysis_report.{format}"
    wildcard_constraints:
        format="zip|html"
    log: "{PYTEST_TMP}/test_reprocess_master_byte_ide0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/logs/render_report_{format}.log"
    resources:
        slurm_partition="None",
        runtime=30,
        tasks=1,
        cpus_per_task=1,
        mem_mb=2000,
        nodes=1
    shell:
        """
        python -m hhemt.render_report_runner \
            --system-config {PYTEST_TMP}/test_reprocess_master_byte_ide0/synthetic_test_runs/synth_sensitivity/system_config.yaml \
            --analysis-config {PYTEST_TMP}/test_reprocess_master_byte_ide0/synthetic_test_runs/synth_sensitivity/analysis_config.yaml \
            --format {wildcards.format} \
            --reprocess \
            > {log} 2>&1
        """
