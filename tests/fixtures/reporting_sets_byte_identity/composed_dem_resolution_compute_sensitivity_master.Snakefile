# Auto-generated flattened master Snakefile for sensitivity analysis
# Each member simulation phase gets its own rule with appropriate resources

import os
from datetime import datetime as _dt
from hhemt.report_renderers._figure_emission import format_sources_rst as _fmt_sources_rst

try:
    from importlib.metadata import version as _pkg_version
    _toolkit_version = _pkg_version("hhemt")
except Exception:
    _toolkit_version = "unknown"

# Config dict consumed by report_templates/workflow_description.rst.j2
config["analysis_id"] = 'synth_sensitivity'
config["toolkit_version"] = _toolkit_version
config["n_sims"] = 4
config["is_sensitivity"] = True
config["n_members"] = 4
config["independent_vars"] = ['n_devices']
config["group_by_var"] = 'run_mode'
config["report"] = {"generated_at": _dt.now().isoformat(timespec="seconds")}

# Paired (member_id, event_id) lists for per-member per-event plot rules.
# Used by `expand(..., zip=True, member_id=MEMBER_EVENT_PAIRS_MEMBER, event_id=MEMBER_EVENT_PAIRS_EVT)`
# in the master `rule all` and in the per-member per-event plot rule definitions.
MEMBER_EVENT_PAIRS_MEMBER = ['0', '1', '2', '3']
MEMBER_EVENT_PAIRS_EVT = ['event_index.0', 'event_index.0', 'event_index.0', 'event_index.0']

report: "report/workflow_description.rst"

onstart:
    shell("mkdir -p _status {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/logs/sims {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/logs")

# onsuccess: removed — `rule export_scenario_status` (added below) now produces
# scenario_status.csv and workflow_summary.md on the success path via the
# Snakemake DAG.

onerror:
    shell("""
        python -m hhemt.export_scenario_status \
            --system-config {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/system_config.yaml \
            --analysis-config {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/analysis_config.yaml \
            > {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/logs/export_scenario_status.log 2>&1
    """)


rule all:
    input:
        "_status/a_setup_target_0_complete.flag", "_status/e_consolidate_member-0_complete.flag", "_status/e_consolidate_member-1_complete.flag", "_status/e_consolidate_member-2_complete.flag", "_status/e_consolidate_member-3_complete.flag", "_status/f_consolidate_experiment_complete.flag", "plots/system_overview.html", "plots/per_analysis/summary_table.html", "plots/appendix/scenario_status.html", "plots/errors_and_warnings/validation_report.html", "plots/disk_utilization.html", "plots/metadata.html", "plots/workflow_performance.html", "scenario_status.csv", "workflow_summary.md", expand("plots/sensitivity/per_sim/member-{member_id}/{event_id}/peak_flood_depth__member.{member_id}__evt.{event_id}.html", zip=True, member_id=MEMBER_EVENT_PAIRS_MEMBER, event_id=MEMBER_EVENT_PAIRS_EVT), expand("plots/sensitivity/per_sim/member-{member_id}/{event_id}/conduit_flow__member.{member_id}__evt.{event_id}.html", zip=True, member_id=MEMBER_EVENT_PAIRS_MEMBER, event_id=MEMBER_EVENT_PAIRS_EVT), expand("plots/sensitivity/benchmarking/benchmarking__{independent_var}.vs.total.html", independent_var=['n_devices']), "plots/eda/dem_resolution_cost_error.html", "plots/eda/dem_resolution_error_ecdf.html", "plots/eda/dem_resolution_diff_maps.html", "plots/eda/dem_resolution_coupling_table.html", "plots/eda/config_diff_maps.html", "analysis_report.zip"

rule setup_target_0:
    output: "_status/a_setup_target_0_complete.flag"
    log: "{PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/logs/setup_target_0.log"
    conda: "{REPO_ROOT}/workflow/envs/hhemt.yaml"
    resources:
        slurm_partition="None",
        runtime=60,
        tasks=1,
        cpus_per_task=1,
        mem_mb=12000,
        nodes=1
    shell:
        """
        mkdir -p {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/logs/sims {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/logs _status
        python -m hhemt.setup_workflow \
            --system-config {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/system_config.yaml \
            --analysis-config {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/analysis_config.yaml \
            \
            \
            --compile-triton-swmm \
            \
            \
            \
            --flag-output {output} \
            --rule-name setup_target_0 \
            --target-id 0 \
            2>&1 | tee {log}
        """

rule prepare_member_0_evt_event_index_0:
    input:
        "_status/a_setup_target_0_complete.flag",
        "_status/member-0_inputs.json"
    output: "_status/b_prepare_member-0_evt-event_index.0_complete.flag"
    log: "{PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/logs/sims/prepare_member_0_evt_event_index_0.log"
    conda: "{REPO_ROOT}/workflow/envs/hhemt.yaml"
    resources:
        slurm_partition="None",
        runtime=30,
        tasks=1,
        cpus_per_task=1,
        mem_mb=2000,
        nodes=1
    shell:
        """
        mkdir -p {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/logs/sims {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/logs _status
        python -m hhemt.prepare_scenario_runner \
            --event-iloc 0 \
            --system-config {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/system_config.yaml \
            --analysis-config {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/members/member_0/member_0.yaml \
            \
            \
            --flag-output {output} \
            --rule-name prepare_member_0_evt_event_index_0 \
            --member-id 0 \
            --event-id event_index.0 \
            2>&1 | tee {log}
        """

rule simulation_member_0_evt_event_index_0:
    input:
        "_status/b_prepare_member-0_evt-event_index.0_complete.flag",
        "_status/member-0_inputs.json"
    output: "_status/c_run_tritonswmm_member-0_evt-event_index.0_complete.flag"
    retries: 2
    log: "{PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/logs/sims/simulation_member_0_evt_event_index_0.log"
    conda: "{REPO_ROOT}/workflow/envs/hhemt.yaml"
    threads: 2
    resources:
        slurm_partition="None",
        runtime=60,
        tasks=2,
        cpus_per_task=1,
        mem_mb=4000,
        nodes=1,
        mpi=True
    shell:
        """
        mkdir -p {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/logs/sims {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/logs _status
        python -m hhemt.run_simulation_runner \
            --event-iloc 0 \
            --system-config {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/system_config.yaml \
            --analysis-config {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/members/member_0/member_0.yaml \
            --model-type tritonswmm \
            --member-id 0 \
            --pickup-where-leftoff \
            --flag-output {output} \
            --rule-name simulation_member_0_evt_event_index_0 \
            --event-id event_index.0 \
            2>&1 | tee {log}
        """

rule process_member_0_evt_event_index_0:
    input:
        "_status/c_run_tritonswmm_member-0_evt-event_index.0_complete.flag",
        "_status/member-0_inputs.json"
    output: "_status/d_process_tritonswmm_member-0_evt-event_index.0_complete.flag"
    log: "{PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/logs/sims/process_member_0_evt_event_index_0.log"
    conda: "{REPO_ROOT}/workflow/envs/hhemt.yaml"
    resources:
        slurm_partition="None",
        runtime=240,
        tasks=1,
        cpus_per_task=2,
        mem_mb=12000,
        nodes=1
    shell:
        """
        mkdir -p {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/logs/sims {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/logs _status
        python -m hhemt.process_timeseries_runner \
            --event-iloc 0 \
            --system-config {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/system_config.yaml \
            --analysis-config {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/members/member_0/member_0.yaml \
            --model-type tritonswmm \
            --which both \
            \
            --compression-level 5 \
            --flag-output {output} \
            --rule-name process_member_0_evt_event_index_0 \
            --member-id 0 \
            --event-id event_index.0 \
            2>&1 | tee {log}
        """

rule consolidate_member_0:
    input: "_status/d_process_tritonswmm_member-0_evt-event_index.0_complete.flag", "_status/member-0_inputs.json"
    output: "_status/e_consolidate_member-0_complete.flag"
    log: "{PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/logs/sims/consolidate_member_0.log"
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
        mkdir -p {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/logs/sims {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/logs _status
        python -m hhemt.consolidate_workflow \
            --system-config {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/system_config.yaml \
            --analysis-config {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/members/member_0/member_0.yaml \
            --which both \
            --compression-level 5 \
            --flag-output {output} \
            --rule-name consolidate_member_0 \
            --member-id 0 \
            2>&1 | tee {log}
        """

rule prepare_member_1_evt_event_index_0:
    input:
        "_status/a_setup_target_0_complete.flag",
        "_status/member-1_inputs.json"
    output: "_status/b_prepare_member-1_evt-event_index.0_complete.flag"
    log: "{PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/logs/sims/prepare_member_1_evt_event_index_0.log"
    conda: "{REPO_ROOT}/workflow/envs/hhemt.yaml"
    resources:
        slurm_partition="None",
        runtime=30,
        tasks=1,
        cpus_per_task=1,
        mem_mb=2000,
        nodes=1
    shell:
        """
        mkdir -p {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/logs/sims {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/logs _status
        python -m hhemt.prepare_scenario_runner \
            --event-iloc 0 \
            --system-config {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/system_config.yaml \
            --analysis-config {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/members/member_1/member_1.yaml \
            \
            \
            --flag-output {output} \
            --rule-name prepare_member_1_evt_event_index_0 \
            --member-id 1 \
            --event-id event_index.0 \
            2>&1 | tee {log}
        """

rule simulation_member_1_evt_event_index_0:
    input:
        "_status/b_prepare_member-1_evt-event_index.0_complete.flag",
        "_status/member-1_inputs.json"
    output: "_status/c_run_tritonswmm_member-1_evt-event_index.0_complete.flag"
    retries: 2
    log: "{PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/logs/sims/simulation_member_1_evt_event_index_0.log"
    conda: "{REPO_ROOT}/workflow/envs/hhemt.yaml"
    threads: 2
    resources:
        slurm_partition="None",
        runtime=60,
        tasks=1,
        cpus_per_task=2,
        mem_mb=4000,
        nodes=1
    shell:
        """
        mkdir -p {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/logs/sims {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/logs _status
        python -m hhemt.run_simulation_runner \
            --event-iloc 0 \
            --system-config {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/system_config.yaml \
            --analysis-config {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/members/member_1/member_1.yaml \
            --model-type tritonswmm \
            --member-id 1 \
            --pickup-where-leftoff \
            --flag-output {output} \
            --rule-name simulation_member_1_evt_event_index_0 \
            --event-id event_index.0 \
            2>&1 | tee {log}
        """

rule process_member_1_evt_event_index_0:
    input:
        "_status/c_run_tritonswmm_member-1_evt-event_index.0_complete.flag",
        "_status/member-1_inputs.json"
    output: "_status/d_process_tritonswmm_member-1_evt-event_index.0_complete.flag"
    log: "{PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/logs/sims/process_member_1_evt_event_index_0.log"
    conda: "{REPO_ROOT}/workflow/envs/hhemt.yaml"
    resources:
        slurm_partition="None",
        runtime=240,
        tasks=1,
        cpus_per_task=2,
        mem_mb=12000,
        nodes=1
    shell:
        """
        mkdir -p {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/logs/sims {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/logs _status
        python -m hhemt.process_timeseries_runner \
            --event-iloc 0 \
            --system-config {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/system_config.yaml \
            --analysis-config {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/members/member_1/member_1.yaml \
            --model-type tritonswmm \
            --which both \
            \
            --compression-level 5 \
            --flag-output {output} \
            --rule-name process_member_1_evt_event_index_0 \
            --member-id 1 \
            --event-id event_index.0 \
            2>&1 | tee {log}
        """

rule consolidate_member_1:
    input: "_status/d_process_tritonswmm_member-1_evt-event_index.0_complete.flag", "_status/member-1_inputs.json"
    output: "_status/e_consolidate_member-1_complete.flag"
    log: "{PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/logs/sims/consolidate_member_1.log"
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
        mkdir -p {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/logs/sims {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/logs _status
        python -m hhemt.consolidate_workflow \
            --system-config {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/system_config.yaml \
            --analysis-config {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/members/member_1/member_1.yaml \
            --which both \
            --compression-level 5 \
            --flag-output {output} \
            --rule-name consolidate_member_1 \
            --member-id 1 \
            2>&1 | tee {log}
        """

rule prepare_member_2_evt_event_index_0:
    input:
        "_status/a_setup_target_0_complete.flag",
        "_status/member-2_inputs.json"
    output: "_status/b_prepare_member-2_evt-event_index.0_complete.flag"
    log: "{PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/logs/sims/prepare_member_2_evt_event_index_0.log"
    conda: "{REPO_ROOT}/workflow/envs/hhemt.yaml"
    resources:
        slurm_partition="None",
        runtime=30,
        tasks=1,
        cpus_per_task=1,
        mem_mb=2000,
        nodes=1
    shell:
        """
        mkdir -p {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/logs/sims {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/logs _status
        python -m hhemt.prepare_scenario_runner \
            --event-iloc 0 \
            --system-config {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/system_config.yaml \
            --analysis-config {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/members/member_2/member_2.yaml \
            \
            \
            --flag-output {output} \
            --rule-name prepare_member_2_evt_event_index_0 \
            --member-id 2 \
            --event-id event_index.0 \
            2>&1 | tee {log}
        """

rule simulation_member_2_evt_event_index_0:
    input:
        "_status/b_prepare_member-2_evt-event_index.0_complete.flag",
        "_status/member-2_inputs.json"
    output: "_status/c_run_tritonswmm_member-2_evt-event_index.0_complete.flag"
    retries: 2
    log: "{PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/logs/sims/simulation_member_2_evt_event_index_0.log"
    conda: "{REPO_ROOT}/workflow/envs/hhemt.yaml"
    threads: 4
    resources:
        slurm_partition="None",
        runtime=60,
        tasks=2,
        cpus_per_task=2,
        mem_mb=8000,
        nodes=1,
        mpi=True
    shell:
        """
        mkdir -p {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/logs/sims {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/logs _status
        python -m hhemt.run_simulation_runner \
            --event-iloc 0 \
            --system-config {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/system_config.yaml \
            --analysis-config {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/members/member_2/member_2.yaml \
            --model-type tritonswmm \
            --member-id 2 \
            --pickup-where-leftoff \
            --flag-output {output} \
            --rule-name simulation_member_2_evt_event_index_0 \
            --event-id event_index.0 \
            2>&1 | tee {log}
        """

rule process_member_2_evt_event_index_0:
    input:
        "_status/c_run_tritonswmm_member-2_evt-event_index.0_complete.flag",
        "_status/member-2_inputs.json"
    output: "_status/d_process_tritonswmm_member-2_evt-event_index.0_complete.flag"
    log: "{PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/logs/sims/process_member_2_evt_event_index_0.log"
    conda: "{REPO_ROOT}/workflow/envs/hhemt.yaml"
    resources:
        slurm_partition="None",
        runtime=240,
        tasks=1,
        cpus_per_task=2,
        mem_mb=12000,
        nodes=1
    shell:
        """
        mkdir -p {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/logs/sims {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/logs _status
        python -m hhemt.process_timeseries_runner \
            --event-iloc 0 \
            --system-config {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/system_config.yaml \
            --analysis-config {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/members/member_2/member_2.yaml \
            --model-type tritonswmm \
            --which both \
            \
            --compression-level 5 \
            --flag-output {output} \
            --rule-name process_member_2_evt_event_index_0 \
            --member-id 2 \
            --event-id event_index.0 \
            2>&1 | tee {log}
        """

rule consolidate_member_2:
    input: "_status/d_process_tritonswmm_member-2_evt-event_index.0_complete.flag", "_status/member-2_inputs.json"
    output: "_status/e_consolidate_member-2_complete.flag"
    log: "{PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/logs/sims/consolidate_member_2.log"
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
        mkdir -p {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/logs/sims {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/logs _status
        python -m hhemt.consolidate_workflow \
            --system-config {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/system_config.yaml \
            --analysis-config {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/members/member_2/member_2.yaml \
            --which both \
            --compression-level 5 \
            --flag-output {output} \
            --rule-name consolidate_member_2 \
            --member-id 2 \
            2>&1 | tee {log}
        """

rule prepare_member_3_evt_event_index_0:
    input:
        "_status/a_setup_target_0_complete.flag",
        "_status/member-3_inputs.json"
    output: "_status/b_prepare_member-3_evt-event_index.0_complete.flag"
    log: "{PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/logs/sims/prepare_member_3_evt_event_index_0.log"
    conda: "{REPO_ROOT}/workflow/envs/hhemt.yaml"
    resources:
        slurm_partition="None",
        runtime=30,
        tasks=1,
        cpus_per_task=1,
        mem_mb=2000,
        nodes=1
    shell:
        """
        mkdir -p {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/logs/sims {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/logs _status
        python -m hhemt.prepare_scenario_runner \
            --event-iloc 0 \
            --system-config {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/system_config.yaml \
            --analysis-config {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/members/member_3/member_3.yaml \
            \
            \
            --flag-output {output} \
            --rule-name prepare_member_3_evt_event_index_0 \
            --member-id 3 \
            --event-id event_index.0 \
            2>&1 | tee {log}
        """

rule simulation_member_3_evt_event_index_0:
    input:
        "_status/b_prepare_member-3_evt-event_index.0_complete.flag",
        "_status/member-3_inputs.json"
    output: "_status/c_run_tritonswmm_member-3_evt-event_index.0_complete.flag"
    retries: 2
    log: "{PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/logs/sims/simulation_member_3_evt_event_index_0.log"
    conda: "{REPO_ROOT}/workflow/envs/hhemt.yaml"
    threads: 1
    resources:
        slurm_partition="None",
        runtime=60,
        tasks=1,
        cpus_per_task=1,
        mem_mb=2000,
        nodes=1
    shell:
        """
        mkdir -p {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/logs/sims {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/logs _status
        python -m hhemt.run_simulation_runner \
            --event-iloc 0 \
            --system-config {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/system_config.yaml \
            --analysis-config {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/members/member_3/member_3.yaml \
            --model-type tritonswmm \
            --member-id 3 \
            --pickup-where-leftoff \
            --flag-output {output} \
            --rule-name simulation_member_3_evt_event_index_0 \
            --event-id event_index.0 \
            2>&1 | tee {log}
        """

rule process_member_3_evt_event_index_0:
    input:
        "_status/c_run_tritonswmm_member-3_evt-event_index.0_complete.flag",
        "_status/member-3_inputs.json"
    output: "_status/d_process_tritonswmm_member-3_evt-event_index.0_complete.flag"
    log: "{PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/logs/sims/process_member_3_evt_event_index_0.log"
    conda: "{REPO_ROOT}/workflow/envs/hhemt.yaml"
    resources:
        slurm_partition="None",
        runtime=240,
        tasks=1,
        cpus_per_task=2,
        mem_mb=12000,
        nodes=1
    shell:
        """
        mkdir -p {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/logs/sims {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/logs _status
        python -m hhemt.process_timeseries_runner \
            --event-iloc 0 \
            --system-config {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/system_config.yaml \
            --analysis-config {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/members/member_3/member_3.yaml \
            --model-type tritonswmm \
            --which both \
            \
            --compression-level 5 \
            --flag-output {output} \
            --rule-name process_member_3_evt_event_index_0 \
            --member-id 3 \
            --event-id event_index.0 \
            2>&1 | tee {log}
        """

rule consolidate_member_3:
    input: "_status/d_process_tritonswmm_member-3_evt-event_index.0_complete.flag", "_status/member-3_inputs.json"
    output: "_status/e_consolidate_member-3_complete.flag"
    log: "{PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/logs/sims/consolidate_member_3.log"
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
        mkdir -p {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/logs/sims {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/logs _status
        python -m hhemt.consolidate_workflow \
            --system-config {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/system_config.yaml \
            --analysis-config {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/members/member_3/member_3.yaml \
            --which both \
            --compression-level 5 \
            --flag-output {output} \
            --rule-name consolidate_member_3 \
            --member-id 3 \
            2>&1 | tee {log}
        """

rule experiment_consolidation:
    input: "_status/e_consolidate_member-0_complete.flag", "_status/e_consolidate_member-1_complete.flag", "_status/e_consolidate_member-2_complete.flag", "_status/e_consolidate_member-3_complete.flag"
    output: "_status/f_consolidate_experiment_complete.flag"
    log: "{PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/logs/experiment_consolidation.log"
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
        mkdir -p {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/logs/sims {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/logs _status
        python -m hhemt.consolidate_workflow \
            --system-config {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/system_config.yaml \
            --analysis-config {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/analysis_config.yaml \
            --consolidate-sensitivity-analysis-outputs \
            --which both \
            --compression-level 5 \
            --flag-output {output} \
            --rule-name experiment_consolidation \
            2>&1 | tee {log}
        """

rule plot_system_overview:
    input:
        consolidated = "_status/f_consolidate_experiment_complete.flag",
    output:
        report(
            "plots/system_overview.html",
            caption="report/captions/system_map.rst",
            category="System Information",
            labels={"figure": "System map"},
        )
    params:
        source_paths = [{'path': '../elevation_10.00m.dem', 'variables': []}, {'path': 'members/member_0/sims/event_index.0/swmm/hydro.inp', 'variables': ['[SUBCATCHMENTS]', '[JUNCTIONS]', '[OUTFALLS]']}, {'path': 'members/member_0/sims/event_index.0/swmm/hydraulics.inp', 'variables': ['[CONDUITS]', '[JUNCTIONS]', '[POLYGONS]']}, {'path': '{SYNTH_MODELS}/{MODEL_KEY}/boundary.geojson', 'variables': []}],
        source_paths_rst = '- ``../elevation_10.00m.dem``\n\n- ``members/member_0/sims/event_index.0/swmm/hydro.inp``\n\n  - ``[SUBCATCHMENTS]``\n  - ``[JUNCTIONS]``\n  - ``[OUTFALLS]``\n\n- ``members/member_0/sims/event_index.0/swmm/hydraulics.inp``\n\n  - ``[CONDUITS]``\n  - ``[JUNCTIONS]``\n  - ``[POLYGONS]``\n\n- ``{SYNTH_MODELS}/{MODEL_KEY}/boundary.geojson``\n',
    log: "logs/plots/system_overview.log"
    conda: "{REPO_ROOT}/workflow/envs/hhemt.yaml"
    resources: mem_mb=2000, time_min=10
    shell:
        """
        {PYTHON} -m hhemt.report_renderers._cli system_overview \
            --system-config {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/system_config.yaml \
            --analysis-config {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/analysis_config.yaml \
            --output {output} \
            2>&1 | tee {log}
        """

rule plot_per_analysis_summary_table:
    input:
        "scenario_status.csv",
        consolidated = "_status/f_consolidate_experiment_complete.flag",
    output:
        report(
            "plots/per_analysis/summary_table.html",
            caption="report/captions/per_analysis_summary_table.rst",
            category="Workflow Status",
            subcategory="Workflow Health Summary",
            labels={"figure": "Summary table"},
        )
    params:
        source_paths = [{'path': 'members/member_0/sims/event_index.0/out_tritonswmm/swmm/hydraulics.rpt', 'variables': ['Flow Routing Continuity error (%)']}, {'path': 'members/member_0/sims/event_index.0/log_tritonswmm.json', 'variables': ['model_run_completed[tritonswmm] (status flag for n_successful / n_pending counts)']}, {'path': 'members/member_1/sims/event_index.0/out_tritonswmm/swmm/hydraulics.rpt', 'variables': ['Flow Routing Continuity error (%)']}, {'path': 'members/member_1/sims/event_index.0/log_tritonswmm.json', 'variables': ['model_run_completed[tritonswmm] (status flag for n_successful / n_pending counts)']}, {'path': 'members/member_2/sims/event_index.0/out_tritonswmm/swmm/hydraulics.rpt', 'variables': ['Flow Routing Continuity error (%)']}, {'path': 'members/member_2/sims/event_index.0/log_tritonswmm.json', 'variables': ['model_run_completed[tritonswmm] (status flag for n_successful / n_pending counts)']}, {'path': 'members/member_3/sims/event_index.0/out_tritonswmm/swmm/hydraulics.rpt', 'variables': ['Flow Routing Continuity error (%)']}, {'path': 'members/member_3/sims/event_index.0/log_tritonswmm.json', 'variables': ['model_run_completed[tritonswmm] (status flag for n_successful / n_pending counts)']}],
        source_paths_rst = '- ``members/member_0/sims/event_index.0/out_tritonswmm/swmm/hydraulics.rpt``\n\n  - ``Flow Routing Continuity error (%)``\n\n- ``members/member_0/sims/event_index.0/log_tritonswmm.json``\n\n  - ``model_run_completed[tritonswmm] (status flag for n_successful / n_pending counts)``\n\n- ``members/member_1/sims/event_index.0/out_tritonswmm/swmm/hydraulics.rpt``\n\n  - ``Flow Routing Continuity error (%)``\n\n- ``members/member_1/sims/event_index.0/log_tritonswmm.json``\n\n  - ``model_run_completed[tritonswmm] (status flag for n_successful / n_pending counts)``\n\n- ``members/member_2/sims/event_index.0/out_tritonswmm/swmm/hydraulics.rpt``\n\n  - ``Flow Routing Continuity error (%)``\n\n- ``members/member_2/sims/event_index.0/log_tritonswmm.json``\n\n  - ``model_run_completed[tritonswmm] (status flag for n_successful / n_pending counts)``\n\n- ``members/member_3/sims/event_index.0/out_tritonswmm/swmm/hydraulics.rpt``\n\n  - ``Flow Routing Continuity error (%)``\n\n- ``members/member_3/sims/event_index.0/log_tritonswmm.json``\n\n  - ``model_run_completed[tritonswmm] (status flag for n_successful / n_pending counts)``\n',
    log: "logs/plots/per_analysis_summary_table.log"
    conda: "{REPO_ROOT}/workflow/envs/hhemt.yaml"
    resources: mem_mb=2000, time_min=5
    shell:
        """
        {PYTHON} -m hhemt.report_renderers._cli per_analysis_summary \
            --system-config {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/system_config.yaml \
            --analysis-config {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/analysis_config.yaml \
            --output {output} \
            2>&1 | tee {log}
        """

rule plot_scenario_status_appendix:
    input:
        "scenario_status.csv",
        consolidated = "_status/f_consolidate_experiment_complete.flag",
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
            --system-config {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/system_config.yaml \
            --analysis-config {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/analysis_config.yaml \
            --output {output} \
            2>&1 | tee {log}
        """

rule plot_errors_and_warnings:
    input:
        "scenario_status.csv",
        "validation_report.json",
        consolidated = "_status/f_consolidate_experiment_complete.flag",
    output:
        report(
            "plots/errors_and_warnings/validation_report.html",
            caption="report/captions/errors_and_warnings.rst",
            category="Errors and Warnings",
            subcategory="Validation Report",
            labels={"figure": "Validation report"},
        )
    params:
        source_paths = [{'path': 'scenario_status.csv', 'variables': ['scenario_setup', 'run_completed', 'actual_nTasks', 'actual_omp_threads', 'actual_total_gpus', 'actual_gpu_backend']}, {'path': 'sims/<event_id>/log_<model_type>.json', 'variables': ['simulation_completed (per scenario × model_type)']}, {'path': '../system_log.json', 'variables': ['compilation_tritonswmm_cpu_successful', 'compilation_tritonswmm_gpu_successful', 'compilation_triton_cpu_successful', 'compilation_triton_gpu_successful', 'compilation_swmm_successful']}],
        source_paths_rst = '- ``scenario_status.csv``\n\n  - ``scenario_setup``\n  - ``run_completed``\n  - ``actual_nTasks``\n  - ``actual_omp_threads``\n  - ``actual_total_gpus``\n  - ``actual_gpu_backend``\n\n- ``sims/<event_id>/log_<model_type>.json``\n\n  - ``simulation_completed (per scenario × model_type)``\n\n- ``../system_log.json``\n\n  - ``compilation_tritonswmm_cpu_successful``\n  - ``compilation_tritonswmm_gpu_successful``\n  - ``compilation_triton_cpu_successful``\n  - ``compilation_triton_gpu_successful``\n  - ``compilation_swmm_successful``\n',
    log: "logs/plots/errors_and_warnings.log"
    conda: "{REPO_ROOT}/workflow/envs/hhemt.yaml"
    resources: mem_mb=1000, time_min=5
    shell:
        """
        {PYTHON} -m hhemt.report_renderers._cli errors_and_warnings \
            --system-config {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/system_config.yaml \
            --analysis-config {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/analysis_config.yaml \
            --output {output} \
            2>&1 | tee {log}
        """

rule plot_disk_utilization:
    input:
        consolidated = "_status/f_consolidate_experiment_complete.flag",
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
            --system-config {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/system_config.yaml \
            --analysis-config {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/analysis_config.yaml \
            --output {output} \
            2>&1 | tee {log}
        """

rule plot_metadata:
    input:
        consolidated = "_status/f_consolidate_experiment_complete.flag",
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
            --system-config {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/system_config.yaml \
            --analysis-config {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/analysis_config.yaml \
            --output {output} \
            2>&1 | tee {log}
        """

rule plot_workflow_performance:
    input:
        consolidated = "_status/f_consolidate_experiment_complete.flag",
    output:
        report(
            "plots/workflow_performance.html",
            caption="report/captions/workflow_performance.rst",
            category="Workflow performance",
            labels={"figure": "Workflow performance"},
        )
    params:
        source_paths = [],
        source_paths_rst = '',
    log: "logs/plots/workflow_performance.log"
    conda: "{REPO_ROOT}/workflow/envs/hhemt.yaml"
    resources: mem_mb=1000, time_min=5
    shell:
        """
        {PYTHON} -m hhemt.report_renderers._cli workflow_performance \
            --system-config {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/system_config.yaml \
            --analysis-config {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/analysis_config.yaml \
            --output {output} \
            2>&1 | tee {log}
        """

localrules: export_scenario_status

rule export_scenario_status:
    input: "_status/f_consolidate_experiment_complete.flag"
    output:
        csv = "scenario_status.csv",
        md  = "workflow_summary.md",
        # The read-model is written by THIS rule's shell (export_scenario_status.py:444
        # and :477) and was previously undeclared. Snakemake builds its DAG from declared
        # interfaces, so an undeclared side-effect write to a file that
        # plot_errors_and_warnings declares as an INPUT (:3268) cannot propagate: the plot
        # rule is evaluated against the JSON's mtime at BUILD time, classified up to date,
        # excluded -- and then this rule rewrites the JSON after the DAG is fixed, creating
        # an inversion with no scheduled consumer.
        #
        # SAFETY (measured, Round 19): two rules declaring one output is NOT a silent smell
        # -- Snakemake raises AmbiguousRuleException at DAG BUILD naming both rules. This
        # declaration does not create that case: consolidate_workflow.py:489 writes the same
        # file from its SHELL without declaring it. The invariant this rests on is that the
        # read-model stays declared in EXACTLY ONE rule; break it and the failure is loud.
        #
        # SCOPE, stated so this is not over-read: declaring the output is a real fix for a
        # real defect. It is NOT established as the cause of the 2026-08-13 failures --
        # run 3's job stats ("all 1, render_report 1, total 2") show this rule never ran,
        # so something else wrote the read-model at 01:44:50 and is still unidentified.
        read_model = "validation_report.json",
    log: "{PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/logs/export_scenario_status.log"
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
            --system-config {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/system_config.yaml \
            --analysis-config {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/analysis_config.yaml \
            2>&1 | tee {log}
        """

ILOC_BY_EVENT_ID_BY_MEMBER = {'0': {'event_index.0': 0}, '1': {'event_index.0': 0}, '2': {'event_index.0': 0}, '3': {'event_index.0': 0}}

from hhemt.report_plot_ids import (
    event_labels_from_status as _event_labels_from_status,
    report_label_value as _report_label_value,
    member_labels_from_status as _member_labels_from_status,
)

_EVENT_LABELS = _event_labels_from_status(workflow.basedir)
_MEMBER_LABELS = _member_labels_from_status(workflow.basedir)


def _per_sim_per_member_flood_depth_sources(wildcards):
    from hhemt.report_renderers._figure_emission import (
        collect_per_sim_source_paths,
    )
    return collect_per_sim_source_paths(
        "peak_flood_depth",
        wildcards.event_id,
        rainfall_datavar='RG_synth',
        storm_tide_datavar='water_level',
        member_id=wildcards.member_id,
    )

def _per_sim_per_member_conduit_flow_sources(wildcards):
    from hhemt.report_renderers._figure_emission import (
        collect_per_sim_source_paths,
    )
    return collect_per_sim_source_paths(
        "conduit_flow",
        wildcards.event_id,
        rainfall_datavar='RG_synth',
        storm_tide_datavar='water_level',
        member_id=wildcards.member_id,
    )

rule plot_per_sim_per_member_peak_flood_depth:
    input:
        master = "_status/f_consolidate_experiment_complete.flag",
    output:
        report(
            "plots/sensitivity/per_sim/member-{member_id}/{event_id}/peak_flood_depth__member.{member_id}__evt.{event_id}.html",
            caption="report/captions/per_sim_peak_flood_depth.rst",
            category="Per Simulation Results",
            labels=(lambda w: {"figure": "Peak flood depth", "member": _report_label_value(_MEMBER_LABELS, w.member_id, "member"), "event": _report_label_value(_EVENT_LABELS, w.event_id, "event")}),
        )
    wildcard_constraints:
        member_id="[A-Za-z0-9_.]+",
        event_id="[A-Za-z0-9_.]+",
    params:
        source_paths = _per_sim_per_member_flood_depth_sources,
        source_paths_rst = lambda w: _fmt_sources_rst(_per_sim_per_member_flood_depth_sources(w)),
        event_iloc = lambda w: ILOC_BY_EVENT_ID_BY_MEMBER[w.member_id][w.event_id],
    log: "logs/plots/per_sim_per_member_peak_flood_depth_member-{member_id}_{event_id}.log"
    conda: "{REPO_ROOT}/workflow/envs/hhemt.yaml"
    resources: mem_mb=4000, time_min=15
    shell:
        """
        {PYTHON} -m hhemt.report_renderers._cli per_sim_peak_flood_depth \
            --system-config {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/system_config.yaml \
            --analysis-config {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/analysis_config.yaml \
            --member-id {wildcards.member_id} \
            --event-iloc {params.event_iloc} \
            --output {output} \
            2>&1 | tee {log}
        """

rule plot_per_sim_per_member_conduit_flow:
    input:
        master = "_status/f_consolidate_experiment_complete.flag",
    output:
        report(
            "plots/sensitivity/per_sim/member-{member_id}/{event_id}/conduit_flow__member.{member_id}__evt.{event_id}.html",
            caption="report/captions/per_sim_conduit_flow.rst",
            category="Per Simulation Results",
            labels=(lambda w: {"figure": "Conduit flow", "member": _report_label_value(_MEMBER_LABELS, w.member_id, "member"), "event": _report_label_value(_EVENT_LABELS, w.event_id, "event")}),
        )
    wildcard_constraints:
        member_id="[A-Za-z0-9_.]+",
        event_id="[A-Za-z0-9_.]+",
    params:
        source_paths = _per_sim_per_member_conduit_flow_sources,
        source_paths_rst = lambda w: _fmt_sources_rst(_per_sim_per_member_conduit_flow_sources(w)),
        event_iloc = lambda w: ILOC_BY_EVENT_ID_BY_MEMBER[w.member_id][w.event_id],
    log: "logs/plots/per_sim_per_member_conduit_flow_member-{member_id}_{event_id}.log"
    conda: "{REPO_ROOT}/workflow/envs/hhemt.yaml"
    resources: mem_mb=4000, time_min=15
    shell:
        """
        {PYTHON} -m hhemt.report_renderers._cli per_sim_conduit_flow \
            --system-config {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/system_config.yaml \
            --analysis-config {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/analysis_config.yaml \
            --member-id {wildcards.member_id} \
            --event-iloc {params.event_iloc} \
            --output {output} \
            2>&1 | tee {log}
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
        "scenario_status.csv",
        master = "_status/f_consolidate_experiment_complete.flag",
    output:
        report(
            "plots/sensitivity/benchmarking/benchmarking__{independent_var}.vs.total.html",
            caption="report/captions/sensitivity_benchmarking.rst",
            category="Key Results",
            subcategory="Benchmarking",
            labels={"independent_var": "{independent_var}", "figure": "vs Total runtime", "models": "TRITON-SWMM"},
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
            --system-config {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/system_config.yaml \
            --analysis-config {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/analysis_config.yaml \
            --independent-var {wildcards.independent_var} \
            --output {output} \
            2>&1 | tee {log}
        """

rule plot_eda_dem_resolution_cost_error:
    input:
        master = "_status/f_consolidate_experiment_complete.flag",
    output:
        report(
            "plots/eda/dem_resolution_cost_error.html",
            caption="report/captions/eda_dem_resolution_cost_error.rst",
            category="Key Results",
            subcategory="DEM-resolution EDA",
            labels={"figure": "Cost vs error"},
        )
    params:
        source_paths = ['experiment_datatree.zarr'],
        source_paths_rst = '- ``experiment_datatree.zarr``\n',
    log: "_logs/plots/dem_resolution_cost_error.log"
    conda: "{REPO_ROOT}/workflow/envs/hhemt.yaml"
    resources: mem_mb=4000, time_min=10
    shell:
        """
        {PYTHON} -m hhemt.report_renderers._cli eda_compute_sensitivity \
            --system-config {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/system_config.yaml \
            --analysis-config {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/analysis_config.yaml \
            --output {output} \
            2>&1 | tee {log}
        """

rule plot_eda_dem_resolution_error_ecdf:
    input:
        master = "_status/f_consolidate_experiment_complete.flag",
    output:
        report(
            "plots/eda/dem_resolution_error_ecdf.html",
            caption="report/captions/eda_dem_resolution_error_ecdf.rst",
            category="Key Results",
            subcategory="DEM-resolution EDA",
            labels={"figure": "Depth-error ECDF"},
        )
    params:
        source_paths = ['experiment_datatree.zarr'],
        source_paths_rst = '- ``experiment_datatree.zarr``\n',
    log: "_logs/plots/dem_resolution_error_ecdf.log"
    conda: "{REPO_ROOT}/workflow/envs/hhemt.yaml"
    resources: mem_mb=4000, time_min=10
    shell:
        """
        {PYTHON} -m hhemt.report_renderers._cli eda_compute_sensitivity \
            --system-config {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/system_config.yaml \
            --analysis-config {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/analysis_config.yaml \
            --output {output} \
            2>&1 | tee {log}
        """

rule plot_eda_dem_resolution_diff_maps:
    input:
        master = "_status/f_consolidate_experiment_complete.flag",
    output:
        report(
            "plots/eda/dem_resolution_diff_maps.html",
            caption="report/captions/eda_dem_resolution_diff_maps.rst",
            category="Key Results",
            subcategory="DEM-resolution EDA",
            labels={"figure": "Depth-difference maps"},
        )
    params:
        source_paths = ['experiment_datatree.zarr'],
        source_paths_rst = '- ``experiment_datatree.zarr``\n',
    log: "_logs/plots/dem_resolution_diff_maps.log"
    conda: "{REPO_ROOT}/workflow/envs/hhemt.yaml"
    resources: mem_mb=4000, time_min=10
    shell:
        """
        {PYTHON} -m hhemt.report_renderers._cli eda_compute_sensitivity \
            --system-config {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/system_config.yaml \
            --analysis-config {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/analysis_config.yaml \
            --output {output} \
            2>&1 | tee {log}
        """

rule plot_eda_dem_resolution_coupling_table:
    input:
        master = "_status/f_consolidate_experiment_complete.flag",
    output:
        report(
            "plots/eda/dem_resolution_coupling_table.html",
            caption="report/captions/eda_dem_resolution_coupling_table.rst",
            category="Key Results",
            subcategory="DEM-resolution EDA",
            labels={"figure": "Resolution x coupling junctions"},
        )
    params:
        source_paths = ['experiment_datatree.zarr'],
        source_paths_rst = '- ``experiment_datatree.zarr``\n',
    log: "_logs/plots/dem_resolution_coupling_table.log"
    conda: "{REPO_ROOT}/workflow/envs/hhemt.yaml"
    resources: mem_mb=4000, time_min=10
    shell:
        """
        {PYTHON} -m hhemt.report_renderers._cli eda_compute_sensitivity \
            --system-config {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/system_config.yaml \
            --analysis-config {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/analysis_config.yaml \
            --output {output} \
            2>&1 | tee {log}
        """

rule plot_eda_compute_sensitivity:
    input:
        master = "_status/f_consolidate_experiment_complete.flag",
    output:
        report(
            "plots/eda/config_diff_maps.html",
            caption="report/captions/eda_compute_sensitivity.rst",
            category="Key Results",
            subcategory="Compute-config EDA",
            labels={"figure": "Config-diff maps"},
        )
    params:
        source_paths = ['experiment_datatree.zarr'],
        source_paths_rst = '- ``experiment_datatree.zarr``\n',
    log: "_logs/plots/eda_compute_sensitivity.log"
    conda: "{REPO_ROOT}/workflow/envs/hhemt.yaml"
    resources: mem_mb=4000, time_min=10
    shell:
        """
        {PYTHON} -m hhemt.report_renderers._cli eda_compute_sensitivity \
            --system-config {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/system_config.yaml \
            --analysis-config {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/analysis_config.yaml \
            --output {output} \
            2>&1 | tee {log}
        """

rule render_report:
    input:
        "plots/system_overview.html",
        "plots/per_analysis/summary_table.html",
        "plots/appendix/scenario_status.html",
        "plots/errors_and_warnings/validation_report.html",
        "plots/disk_utilization.html",
        "plots/metadata.html",
        "plots/workflow_performance.html",
        "scenario_status.csv",
        "workflow_summary.md",
        expand("plots/sensitivity/per_sim/member-{member_id}/{event_id}/peak_flood_depth__member.{member_id}__evt.{event_id}.html", zip=True, member_id=MEMBER_EVENT_PAIRS_MEMBER, event_id=MEMBER_EVENT_PAIRS_EVT),
        expand("plots/sensitivity/per_sim/member-{member_id}/{event_id}/conduit_flow__member.{member_id}__evt.{event_id}.html", zip=True, member_id=MEMBER_EVENT_PAIRS_MEMBER, event_id=MEMBER_EVENT_PAIRS_EVT),
        expand("plots/sensitivity/benchmarking/benchmarking__{independent_var}.vs.total.html", independent_var=['n_devices']),
        "plots/eda/dem_resolution_cost_error.html",
        "plots/eda/dem_resolution_error_ecdf.html",
        "plots/eda/dem_resolution_diff_maps.html",
        "plots/eda/dem_resolution_coupling_table.html",
        "plots/eda/config_diff_maps.html"
    output:
        "analysis_report.{format}"
    wildcard_constraints:
        format="zip|html"
    log: "{PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/synth_sensitivity/logs/render_report_{format}.log"
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
            --system-config {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/system_config.yaml \
            --analysis-config {PYTEST_TMP}/test_composed_master_byte_iden0/synthetic_test_runs/synth_sensitivity/analysis_config.yaml \
            --format {wildcards.format} \
            2>&1 | tee {log}
        """
