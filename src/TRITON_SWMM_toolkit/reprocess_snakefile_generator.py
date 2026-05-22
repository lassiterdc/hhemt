"""Reprocess-scoped Snakefile generator.

Emits a Snakefile containing only the downstream rules from ``start_with``
onward (``process_*`` / ``consolidate`` / ``plot_*`` / ``export_scenario_status``
/ ``render_report``), reusing the shared rule-emission helpers on
:class:`TRITON_SWMM_toolkit.workflow.SnakemakeWorkflowBuilder`.

Per-scenario simulation completion flags (``_status/c_run_*_complete.flag``)
are declared as plain ``input:`` files on downstream rules rather than as
outputs of ``rule run_*`` — so the reprocess driver never attempts a
simulation. ``SIM_IDS`` is filtered at generation time to scenarios whose
``c_run_*`` flag already exists on disk; live or never-started scenarios
are silently excluded.

Distinct from :mod:`TRITON_SWMM_toolkit.bundle.snakefile_generator`
(bundle-side, render-only); this is the wider source-side sibling.

Public surface
--------------

``generate_reprocess_snakefile(builder, *, start_with, overwrite=True) -> str``
    Return the reprocess Snakefile body as a string.

``write_reprocess_snakefile(builder, *, start_with, overwrite=True) -> Path``
    Write the Snakefile to ``{analysis_dir}/Snakefile.reprocess`` and return
    the path.

Stage → re-emitted rule families (from the reprocess phase plan):

* ``process``:     ``process_*`` + ``consolidate`` + plot + export + render
* ``consolidate``: ``consolidate`` + plot + export + render
* ``render``:      plot + export + render

For ``start_with="consolidate"``, the rule consolidate's ``input:`` references
the existing ``c_run_*`` sim flags directly rather than chaining through
``d_process_*`` flag outputs — process rules are excluded from the reprocess
Snakefile, so depending on their outputs would fail DAG resolution.

For ``start_with="render"``, no upstream rules are emitted; the render and
plot rules consume their existing inputs from disk and Snakemake's mtime
trigger re-fires only what the caller invalidated (typically the report
artifacts).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

from TRITON_SWMM_toolkit.workflow import _resolve_rule_all_extensions

if TYPE_CHECKING:
    from TRITON_SWMM_toolkit.workflow import SnakemakeWorkflowBuilder


START_STAGES = ("process", "consolidate", "render")


def _enabled_models(builder: SnakemakeWorkflowBuilder) -> list[str]:
    """Mirror ``generate_snakefile_content``'s enabled-model enumeration."""
    enabled: list[str] = []
    cfg_system = builder.system.cfg_system
    if cfg_system.toggle_triton_model:
        enabled.append("triton")
    if cfg_system.toggle_tritonswmm_model:
        enabled.append("tritonswmm")
    if cfg_system.toggle_swmm_model:
        enabled.append("swmm")
    if not enabled:
        raise ValueError(
            "No model types enabled! Enable at least one of: "
            "toggle_triton_model, toggle_tritonswmm_model, toggle_swmm_model"
        )
    return enabled


def _available_event_ids(
    analysis_dir: Path,
    *,
    enabled_models: list[str],
    all_event_ids: list[str],
) -> list[str]:
    """Filter ``all_event_ids`` to scenarios whose ``c_run_*`` flag exists.

    A scenario is "available" for reprocess when at least one enabled
    model's ``_status/c_run_{model}_evt-{event_id}_complete.flag`` exists.
    Scenarios missing every model's flag are silently dropped from the
    reprocess Snakefile's ``SIM_IDS`` — re-running them would require the
    simulation driver, which reprocess explicitly does not invoke.
    """
    status_dir = analysis_dir / "_status"
    available: list[str] = []
    for event_id in all_event_ids:
        for model in enabled_models:
            if (status_dir / f"c_run_{model}_evt-{event_id}_complete.flag").exists():
                available.append(event_id)
                break
    return available


def generate_reprocess_snakefile(
    builder: SnakemakeWorkflowBuilder,
    *,
    start_with: Literal["process", "consolidate", "render"],
) -> str:
    """Generate the reprocess Snakefile body as a string.

    Parameters
    ----------
    builder
        The analysis's ``SnakemakeWorkflowBuilder``. Provides config, paths,
        resource builders, and the rule-emission helpers shared with the
        source-side ``generate_snakefile_content``.
    start_with
        Stage to start reprocessing from. Determines which downstream rule
        families are emitted (see module docstring).

    Notes
    -----
    Per cleanup-rerun-delete-redesign Phase 3, the legacy ``overwrite`` flag
    (which baked ``--overwrite-outputs-if-already-created`` into rule shells)
    is retired. Reprocess relies on flag-invalidation
    (``_invalidate_downstream_flags``) to trigger re-runs; force-rerun against
    already-written outputs returns in Phase 4 via ``--override-force-rerun``.
    Reprocess never clears raw outputs — ``override_clear_raw`` is omitted from
    the emitted rule shells (equivalent to passing ``"none"`` at runtime).
    """
    if start_with not in START_STAGES:
        raise ValueError(f"start_with must be one of {START_STAGES!r}; got {start_with!r}")

    from TRITON_SWMM_toolkit.scenario import compute_event_id_slug

    cfg_analysis = builder.cfg_analysis
    analysis_dir = builder.analysis_paths.analysis_dir
    log_dir = builder.analysis_paths.analysis_log_directory
    log_dir_str = str(log_dir)
    conda_env_path = str(builder._get_conda_env_path())
    config_args = builder._get_config_args()
    compression_level = 5
    analysis_id_str = str(cfg_analysis.analysis_id)

    # Determine the enabled model set and full event list (mirror
    # generate_snakefile_content's prelude).
    enabled_models = _enabled_models(builder)
    n_sims = len(builder.analysis.df_sims)
    all_event_ids = [
        compute_event_id_slug(builder.analysis._retrieve_weather_indexer_using_integer_index(i)) for i in range(n_sims)
    ]
    iloc_by_event_id = {all_event_ids[i]: i for i in range(n_sims)}

    # Filter to scenarios whose sim flag exists. Reprocess never spawns sims,
    # so missing flags mean the scenario is silently excluded.
    available_event_ids = _available_event_ids(
        analysis_dir,
        enabled_models=enabled_models,
        all_event_ids=all_event_ids,
    )

    # Resource blocks (mirror source-side shapes; reprocess uses the same
    # processing/consolidation partitions because the work is identical).
    process_resources = builder._build_resource_block(
        partition=cfg_analysis.hpc_setup_and_analysis_processing_partition,
        runtime_min=120,
        mem_mb=cfg_analysis.hpc_mem_allocation_for_sim_output_processing_mb,
        nodes=1,
        tasks=1,
        cpus_per_task=2,
    )
    consolidate_resources = builder._build_resource_block(
        partition=cfg_analysis.hpc_setup_and_analysis_processing_partition,
        runtime_min=30,
        mem_mb=cfg_analysis.hpc_mem_allocation_for_analysis_output_consolidation_mb,
        nodes=1,
        tasks=1,
        cpus_per_task=2,
    )

    # Ensure log dirs exist before any rule fires.
    (analysis_dir / "_status").mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "sims").mkdir(parents=True, exist_ok=True)

    # Report extension resolution (matches source-side via _resolve_rule_all_extensions).
    static_backend = builder._get_report_cfg_static_backend()
    _ext = _resolve_rule_all_extensions(static_backend)
    report_formats: list[str] = ["zip"]
    render_targets_in_rule_all = "".join(f',\n        "analysis_report.{fmt}"' for fmt in report_formats)

    # Determine which `which` value to pass to consolidate's shell based on
    # which models are enabled (mirror generate_snakefile_content's logic).
    if "tritonswmm" in enabled_models:
        which = "both"
    elif "triton" in enabled_models:
        which = "TRITON"
    else:
        which = "SWMM"

    # ---- Preamble (imports + _toolkit_version + config dict + rule all + report) ----
    snakefile_content = f'''# Auto-generated by reprocess_snakefile_generator (start_with={start_with!r})
#
# Reprocess-scoped Snakefile: re-runs downstream stages against existing
# simulation outputs without re-running sims. Per-scenario c_run_*.flag files
# are declared as plain `input:` files (no rule produces them); SIM_IDS is
# filtered to scenarios whose flag already exists on disk.

import os
import glob
import subprocess
from datetime import datetime as _dt
from TRITON_SWMM_toolkit.report_renderers._figure_emission import format_sources_rst as _fmt_sources_rst

try:
    from importlib.metadata import version as _pkg_version
    _toolkit_version = _pkg_version("TRITON_SWMM_toolkit")
except Exception:
    _toolkit_version = "unknown"

# Config dict consumed by report_templates/workflow_description.rst.j2
config["analysis_id"] = {analysis_id_str!r}
config["toolkit_version"] = _toolkit_version
config["n_sims"] = {len(available_event_ids)}
config["is_sensitivity"] = False
config["report"] = {{"generated_at": _dt.now().isoformat(timespec="seconds")}}

report: "report/workflow_description.rst"

SIM_IDS = {available_event_ids!r}
ILOC_BY_EVENT_ID = {iloc_by_event_id!r}

rule all:
    input:
        "_status/e_consolidate_complete.flag",
        "scenario_status.csv",
        "workflow_summary.md",
        "plots/system_overview{_ext["system_overview"]}",
        expand("plots/per_sim/{{event_id}}/peak_flood_depth{_ext["per_sim_peak_flood_depth"]}", event_id=SIM_IDS),
        expand("plots/per_sim/{{event_id}}/conduit_flow{_ext["per_sim_conduit_flow"]}",     event_id=SIM_IDS),
        "plots/per_analysis/summary_table{_ext["per_analysis_summary"]}",
        "plots/appendix/scenario_status{_ext["scenario_status_appendix"]}",
        "plots/errors_and_warnings/validation_report{_ext["errors_and_warnings"]}"{render_targets_in_rule_all},

onerror:
    # Reprocess does not re-fire the sim driver, so the only failure modes are
    # downstream-rule failures. The export_scenario_status `onerror` fallback
    # mirrors source-side semantics.
    shell("""
        {builder.python_executable} -m TRITON_SWMM_toolkit.export_scenario_status \\
            {config_args} \\
            > {log_dir_str}/export_scenario_status.log 2>&1
    """)
'''

    # ---- Process rules (only when start_with == "process") ----
    if start_with == "process":
        for model_type in enabled_models:
            if model_type == "triton":
                which_arg = "TRITON"
            elif model_type == "tritonswmm":
                which_arg = "both"
            elif model_type == "swmm":
                which_arg = "SWMM"
            else:  # pragma: no cover — defensive
                raise ValueError(f"Unknown model_type: {model_type}")
            snakefile_content += builder._build_process_rule_block(
                model_type,
                which_arg=which_arg,
                config_args=config_args,
                log_dir_str=log_dir_str,
                conda_env_path=conda_env_path,
                process_resources=process_resources,
                compression_level=compression_level,
                override_clear_raw="none",  # reprocess never clears raw outputs (force-no regardless of cfg)
            )

    # ---- Consolidate rule (start_with in {"process", "consolidate"}) ----
    if start_with in ("process", "consolidate"):
        # When start_with == "process", consolidate's input chains through the
        # process rule outputs (d_process_*). When start_with == "consolidate",
        # process rules are NOT emitted — consolidate consumes c_run_* sim flags
        # directly to avoid a MissingRuleException at DAG planning.
        if start_with == "process":
            flag_prefix = "d_process"
        else:
            flag_prefix = "c_run"
        consolidate_inputs = []
        for model_type in enabled_models:
            flag_pattern = f"{flag_prefix}_{model_type}_evt-{{event_id}}_complete.flag"
            consolidate_inputs.append(f'expand("_status/{flag_pattern}", event_id=SIM_IDS)')
        consolidate_input_str = " + ".join(consolidate_inputs)
        snakefile_content += builder._build_consolidate_rule_block(
            consolidate_input_str=consolidate_input_str,
            which=which,
            config_args=config_args,
            log_dir_str=log_dir_str,
            conda_env_path=conda_env_path,
            consolidate_resources=consolidate_resources,
            compression_level=compression_level,
            allow_incomplete=True,
        )

    # ---- Plot rules (always emitted; their mtime trigger handles re-firing) ----
    snakefile_content += builder._build_plot_rule_block_system_overview()
    snakefile_content += builder._build_plot_rule_block_per_sim()
    snakefile_content += builder._build_plot_rule_block_per_analysis_summary()
    snakefile_content += builder._build_plot_rule_block_scenario_status_appendix()
    snakefile_content += builder._build_plot_rule_block_errors_and_warnings()

    # ---- Export scenario status (always emitted) ----
    snakefile_content += builder._build_export_scenario_status_rule(
        input_flag="_status/e_consolidate_complete.flag",
    )

    # ---- Render report rule (always emitted; mirror source-side shape) ----
    render_resources = builder._build_resource_block(
        partition=cfg_analysis.hpc_setup_and_analysis_processing_partition,
        runtime_min=30,
        mem_mb=2000,
        nodes=1,
        tasks=1,
        cpus_per_task=1,
    )
    snakefile_content += f'''
rule render_report:
    input:
        "plots/system_overview{_ext["system_overview"]}",
        expand("plots/per_sim/{{event_id}}/peak_flood_depth{_ext["per_sim_peak_flood_depth"]}", event_id=SIM_IDS),
        expand("plots/per_sim/{{event_id}}/conduit_flow{_ext["per_sim_conduit_flow"]}",     event_id=SIM_IDS),
        "plots/per_analysis/summary_table{_ext["per_analysis_summary"]}",
        "plots/appendix/scenario_status{_ext["scenario_status_appendix"]}",
        "plots/errors_and_warnings/validation_report{_ext["errors_and_warnings"]}",
        "scenario_status.csv",
    output:
        "analysis_report.{{format}}"
    wildcard_constraints:
        format="zip|html"
    log: "{log_dir_str}/render_report_{{format}}.log"
    resources:
{render_resources}
    shell:
        """
        {builder.python_executable} -m TRITON_SWMM_toolkit.render_report_runner \\
            {config_args} \\
            --format {{wildcards.format}} \\
            > {{log}} 2>&1
        """
'''

    return snakefile_content


def write_reprocess_snakefile(
    builder: SnakemakeWorkflowBuilder,
    *,
    start_with: Literal["process", "consolidate", "render"],
) -> Path:
    """Generate the reprocess Snakefile and write it to ``Snakefile.reprocess``.

    The destination is ``{analysis_dir}/Snakefile.reprocess`` (sibling of the
    normal ``Snakefile`` so the two can coexist). Overwrites any existing
    file at that path.
    """
    text = generate_reprocess_snakefile(builder, start_with=start_with)
    out = builder.analysis_paths.analysis_dir / "Snakefile.reprocess"
    out.write_text(text)
    return out
