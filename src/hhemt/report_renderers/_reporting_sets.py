"""Named reporting-set registry: which figures a report carries, and in what order.

An analysis report is assembled from many renderers, and different kinds of
analysis want different subsets of them -- an event ensemble wants per-event
maps, a compute-configuration sweep wants scaling plots, a cross-experiment
combine wants comparison tables. This module names each such subset once, as a
"reporting set", so every place that emits, enumerates or orders report figures
reads the same answer instead of carrying its own list.

Each set declares (a) an ordered renderer selection, which the workflow.py
dispatcher turns into plot rules; (b) a sidebar category order, which the
render_report facades pass to _react_surgery; and (c) a validator key plus the
analysis shape and sweep axes it accepts, which config.report checks at
analysis.run() entry.

Import discipline: this module MUST stay import-light -- it must NOT import
renderer modules or config.report at module load. config.report lazy-imports this
module to validate the active set, so a module-level import back into it closes
the cycle config.report -> report_renderers._reporting_sets -> config.report and
the import fails. That is why renderer selection is expressed as builder-key
strings, and a conditional renderer carries a predicate_key string the dispatcher
resolves to a callable, rather than either holding the callable itself.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuleSpecTemplate:
    """The Snakemake rule facts for ONE report figure, declared once.

    A renderer may produce several figures, so this is per FIGURE, not per renderer.
    Several generators need the same facts about a figure -- its rule name, output
    path, caption, category, wildcards. Deliberately NOT a copy per generator: copies
    drift, and a drifted category files one figure under different sidebar sections
    in the source-side report and the bundle-regenerated one. Code reaches a template
    either through a ReportingSet's ``renderer_selection`` -- however the set object
    was obtained: ``REPORTING_SETS``, ``get_reporting_set``, ``compose_reporting_sets``
    (called directly, or via config.report's ``resolve_active_reporting_set`` and
    ``validate_active_reporting_set``), or ``eda_rule_spec_templates`` for the EDA
    family -- or by importing a ``_TMPL_*`` name directly. Examples of readers, not a
    complete list: the bundle regeneration generator
    (`bundle/snakefile_generator.py`) builds its rules entirely from them, and the
    source-side EDA builder emits one rule per template instead of hardcoding its
    figures.

    renderer_module      : the renderer module the bundle rule shells out to.
    output_path_template : the rule's output path (may contain `{wildcard}`s).
    report_kwargs        : the `report(...)` kwargs template
                           (caption/category/subcategory/labels) — its
                           `category` MUST equal the `report(category=)` the
                           corresponding source-side `_build_plot_rule_block_*`
                           builder emits (guarded by a co-sourcing test).
    wildcards            : the rule's wildcard names, in order.
    resources_yaml       : the `resources:` body for the bundle rule.
    log_path_template    : the rule's log path (may contain `{wildcard}`s).
    """

    rule_name: str
    renderer_module: str
    output_path_template: str
    report_kwargs: dict
    wildcards: tuple[str, ...] = ()
    resources_yaml: str = "mem_mb=2000, time_min=10"
    log_path_template: str = ""
    #: Per-TEMPLATE gate, distinct from RendererSelection.predicate_key which gates a
    #: whole selection. A selection may carry N templates that are not all applicable
    #: to the same analysis -- per_sim_per_member carries peak_flood_depth (any model) and
    #: conduit_flow (SWMM-derived only). This field is read by the bundle regeneration
    #: generator (bundle/snakefile_generator.py); the source-side per_sim_per_member
    #: builder does not read it and hardcodes the same SWMM test instead. Splitting the
    #: selection into two same-key selections is UNSAFE: the dispatcher calls the
    #: builder once per selection, and that builder emits peak_flood_depth on every call
    #: (and conduit_flow on every call for a SWMM-bearing analysis), so a split would
    #: emit the same rule twice; for the eda_compute_sensitivity key,
    #: eda_rule_spec_templates() reads only the FIRST same-key selection, so the second
    #: is silently dropped. None = always emitted.
    predicate_key: str | None = None


@dataclass(frozen=True)
class RendererSelection:
    """One renderer in a set's ordered selection.

    builder_key        : key into workflow.py's dispatcher builders map, which names
                         the source-side method that emits this renderer's rules.
    predicate_key      : None for unconditional renderers; a string the dispatcher
                         resolves to a Callable[[inputs], bool] for conditional
                         renderers (e.g. "has_independent_vars",
                         "has_member_event_pairs").
    rule_spec_template : a TUPLE of RuleSpecTemplate, one per figure this renderer
                         produces -- most renderers produce one; per_sim_per_member
                         produces two (peak_flood_depth + conduit_flow) and the
                         dem-resolution EDA selection four. Empty only for a renderer
                         with no figure to declare. Its readers are listed on
                         RuleSpecTemplate.
    """

    builder_key: str
    predicate_key: str | None = None
    rule_spec_template: tuple[RuleSpecTemplate, ...] = ()


@dataclass(frozen=True)
class ReportingSet:
    name: str
    category_order: tuple[str, ...]
    renderer_selection: tuple[RendererSelection, ...]
    # Run-entry validator selector: "benchmarking" delegates to
    # validate_sensitivity_independent_vars; "none" is a no-op (the default set).
    validator_key: str = "none"
    #: The analysis SHAPE this set applies to, read at analysis.run() entry by the
    #: compatibility check: "event_ensemble" iff toggle_sensitivity_analysis is
    #: False, "sensitivity" iff True, "cross_experiment" never selectable from an
    #: analysis config (it is reached only by hhemt combine, which looks the set up
    #: by name). Also the homogeneity key for a LIST of sets: every named set must
    #: declare the same shape. The default is None -- UNDECLARED, not a shape -- so
    #: an undeclared set is refused by the run-entry check rather than silently
    #: admitted; a str default would make every sweep set read as an event ensemble
    #: until its declaration landed, which is fail-OPEN on the exact predicate this
    #: field exists to close.
    shape: str | None = None
    #: Sensitivity AXES this set's figures plot against, as ALL-of over the outer
    #: tuple and ANY-of within each inner tuple. Compared at run entry against the
    #: sweep's varied-axis set (sensitivity.analysis_independent_vars union
    #: system_independent_vars, both already canonical stripped field names), so a
    #: set whose figures need an x-axis the sweep does not vary is refused before
    #: compute is committed rather than rendering empty plots. An empty tuple means
    #: the set imposes no axis requirement -- a real declaration. None is meant to
    #: mean UNDECLARED, by analogy with shape. Not established: the run-entry check
    #: (config.report.validate_active_reporting_set) tests `if active.required_axes:`,
    #: so None is falsy and is ACCEPTED exactly as () is -- unlike an undeclared
    #: shape, which that check does refuse. Every registered set declares a value.
    required_axes: tuple[tuple[str, ...], ...] | None = None


# The sidebar category order every analysis-scoped set shares -- default,
# sensitivity, benchmarking and the three EDA sets; only "combined" declares its own.
# The benchmarking and EDA figures need no category of their own: they file under
# "Key Results", with "Benchmarking" and the EDA family names as SUBCATEGORIES. This
# is a ranking, not an inventory: a category absent from it -- the scenario-status
# appendix's "Appendix" -- still renders, and sorts after every ranked category,
# alphabetically, under _react_surgery's comparator.
_STANDARD_CATEGORY_ORDER: tuple[str, ...] = (
    "Workflow Status",
    "Errors and Warnings",
    "Key Results",
    "System Information",
    "Per Simulation Results",
    "Workflow performance",
    "Metadata",
)

# --- Figure RuleSpecTemplates ------------------------------------------------
# One template per figure, so the registry is the single source for each figure's
# category/caption/wildcards/output. Where a source-side `_build_plot_rule_block_*`
# still writes its own report(category=), that string is cross-checked against the
# template by tests/test_reporting_set_cosourcing.py, which fails on a mismatch. The
# seven unconditional templates from here through _TMPL_WORKFLOW_PERFORMANCE are
# defined once and referenced from every analysis-scoped selection; the
# cross-experiment and EDA templates after them are referenced only by the
# combined set and the EDA sets respectively.
_TMPL_SYSTEM_OVERVIEW = RuleSpecTemplate(
    rule_name="plot_system_overview",
    renderer_module="system_overview",
    output_path_template="plots/system_overview__OUTPUT_EXT__",
    report_kwargs={
        "caption": "report/captions/system_map.rst",
        "category": "System Information",
        "labels": '{"figure": "System map"}',
    },
    wildcards=(),
    resources_yaml="mem_mb=2000, time_min=10",
    log_path_template="_logs/plots/system_overview.log",
)
_TMPL_PER_ANALYSIS_SUMMARY = RuleSpecTemplate(
    rule_name="plot_per_analysis_summary_table",
    renderer_module="per_analysis_summary",
    output_path_template="plots/per_analysis/summary_table__OUTPUT_EXT__",
    report_kwargs={
        "caption": "report/captions/per_analysis_summary_table.rst",
        "category": "Workflow Status",
        "subcategory": "Workflow Health Summary",
        "labels": '{"figure": "Summary table"}',
    },
    wildcards=(),
    resources_yaml="mem_mb=2000, time_min=5",
    log_path_template="_logs/plots/per_analysis_summary_table.log",
)
_TMPL_SCENARIO_STATUS_APPENDIX = RuleSpecTemplate(
    rule_name="plot_scenario_status_appendix",
    renderer_module="scenario_status_appendix",
    output_path_template="plots/appendix/scenario_status__OUTPUT_EXT__",
    report_kwargs={
        "caption": "report/captions/scenario_status_appendix.rst",
        "category": "Appendix",
        "subcategory": "Scenario Status",
        "labels": '{"figure": "Per-scenario status table"}',
    },
    wildcards=(),
    resources_yaml="mem_mb=1000, time_min=5",
    log_path_template="_logs/plots/scenario_status_appendix.log",
)
_TMPL_ERRORS_AND_WARNINGS = RuleSpecTemplate(
    rule_name="plot_errors_and_warnings",
    renderer_module="errors_and_warnings",
    output_path_template="plots/errors_and_warnings/validation_report__OUTPUT_EXT__",
    report_kwargs={
        "caption": "report/captions/errors_and_warnings.rst",
        "category": "Errors and Warnings",
        "subcategory": "Validation Report",
        "labels": '{"figure": "Validation report"}',
    },
    wildcards=(),
    resources_yaml="mem_mb=1000, time_min=5",
    log_path_template="_logs/plots/errors_and_warnings.log",
)
# disk_utilization: facts mirror the source-side builder
# (workflow.py _build_plot_rule_block_disk_utilization), which writes its own copy;
# this template is what the bundle regeneration Snakefile emits.
_TMPL_DISK_UTILIZATION = RuleSpecTemplate(
    rule_name="plot_disk_utilization",
    renderer_module="disk_utilization",
    output_path_template="plots/disk_utilization__OUTPUT_EXT__",
    report_kwargs={
        "caption": "report/captions/disk_utilization.rst",
        "category": "System Information",
        "labels": '{"figure": "Disk Utilization"}',
    },
    wildcards=(),
    resources_yaml="mem_mb=1000, time_min=5",
    log_path_template="_logs/plots/disk_utilization.log",
)
# metadata: the Metadata page -- RO-Crate provenance summary + data availability +
# reprex reproduction guide (the SLURM efficiency table is on the workflow_performance
# page below, not here). Unconditional table renderer (emits .html under both static
# backends). Facts mirror the source-side builder
# (workflow.py _build_plot_rule_block_metadata); the `category` here is
# cross-checked against that builder's report(category=) by
# tests/test_reporting_set_cosourcing.py.
_TMPL_METADATA = RuleSpecTemplate(
    rule_name="plot_metadata",
    renderer_module="metadata",
    output_path_template="plots/metadata__OUTPUT_EXT__",
    report_kwargs={
        "caption": "report/captions/metadata.rst",
        "category": "Metadata",
        "labels": '{"figure": "Metadata"}',
    },
    wildcards=(),
    resources_yaml="mem_mb=1000, time_min=5",
    log_path_template="_logs/plots/metadata.log",
)
# workflow_performance: the run timeline + SLURM efficiency page, kept separate from
# metadata because both tables describe how the workflow RAN rather than what it
# produced or how to reproduce it. Unconditional table renderer (emits .html under both
# static backends). Facts mirror the source-side builder
# (workflow.py _build_plot_rule_block_workflow_performance); the `category` here is
# cross-checked against that builder's report(category=) by
# tests/test_reporting_set_cosourcing.py.
_TMPL_WORKFLOW_PERFORMANCE = RuleSpecTemplate(
    rule_name="plot_workflow_performance",
    renderer_module="workflow_performance",
    output_path_template="plots/workflow_performance__OUTPUT_EXT__",
    report_kwargs={
        "caption": "report/captions/workflow_performance.rst",
        "category": "Workflow performance",
        "labels": '{"figure": "Workflow performance"}',
    },
    wildcards=(),
    resources_yaml="mem_mb=1000, time_min=5",
    log_path_template="_logs/plots/workflow_performance.log",
)
# Cross-experiment templates, used only by the "combined" set below. INERT on the
# analysis-side generators: workflow.py has no builder for them and the single-bundle
# regeneration Snakefile never selects the combined set. Two readers, both on the
# combine path: bundle/_combine.py renders each figure directly at combine time, and
# bundle/combined_snakefile_generator.py emits a report() rule per template so the
# combined report indexes the already-rendered figure.
_TMPL_CROSS_EXPERIMENT_COMPATIBILITY = RuleSpecTemplate(
    rule_name="plot_cross_experiment_compatibility",
    renderer_module="cross_experiment_compatibility",
    output_path_template="plots/cross_experiment/compatibility__OUTPUT_EXT__",
    report_kwargs={
        "caption": "report/captions/cross_experiment_compatibility.rst",
        "category": "Combine Provenance & Compatibility",
        "labels": '{"figure": "Compatibility report"}',
    },
    wildcards=(),
    resources_yaml="mem_mb=1000, time_min=5",
    log_path_template="_logs/plots/cross_experiment_compatibility.log",
)

# Cross-experiment INTERCOMPARISON table. Same readers as the compatibility template
# above. Projects the combined_intercomparison.json read-model
# (clean-vs-resume per-compute-config identity, derived CROSS-BUNDLE by
# _combine._write_combined_intercomparison).
_TMPL_CROSS_EXPERIMENT_INTERCOMPARISON = RuleSpecTemplate(
    rule_name="plot_cross_experiment_intercomparison",
    renderer_module="cross_experiment_intercomparison",
    output_path_template="plots/cross_experiment/intercomparison__OUTPUT_EXT__",
    report_kwargs={
        "caption": "report/captions/cross_experiment_intercomparison.rst",
        "category": "Cross-Experiment Results",
        "labels": '{"figure": "Clean vs resume intercomparison"}',
    },
    wildcards=(),
    resources_yaml="mem_mb=4000, time_min=10",
    log_path_template="_logs/plots/cross_experiment_intercomparison.log",
)

# Cross-experiment INTERCOMPARISON MAPS. Same readers as the compatibility template.
# Re-reads child_crates/*/sensitivity_datatree.zarr at render time rather than
# persisting a derived artifact during the combine's emit step -- the step whose crate,
# read-models and manifest must be byte-identical across two combines of the same
# bundles. Lands under the SAME "Cross-Experiment Results" category as the scalar
# intercomparison table.
_TMPL_CROSS_EXPERIMENT_INTERCOMPARISON_MAPS = RuleSpecTemplate(
    rule_name="plot_cross_experiment_intercomparison_maps",
    renderer_module="cross_experiment_intercomparison_maps",
    output_path_template="plots/cross_experiment/intercomparison_maps__OUTPUT_EXT__",
    report_kwargs={
        "caption": "report/captions/cross_experiment_intercomparison_maps.rst",
        "category": "Cross-Experiment Results",
        "labels": '{"figure": "Clean vs resume spatial diff maps"}',
    },
    wildcards=(),
    resources_yaml="mem_mb=8000, time_min=15",
    log_path_template="_logs/plots/cross_experiment_intercomparison_maps.log",
)

# Cross-experiment ERRORS-AND-WARNINGS roll-up. Same readers as the compatibility
# template. Reads each child_crates/{eid}/validation_report.json at render time, for
# the same emit-determinism reason as the maps above. Gives the combined report ONE
# top-level health page: without it, each experiment's own errors_and_warnings figure
# is reachable only one level down, under "Errors and Warnings" in a subcategory whose
# name starts with that experiment's id (model-arm suffix stripped).
_TMPL_CROSS_EXPERIMENT_ERRORS_AND_WARNINGS = RuleSpecTemplate(
    rule_name="plot_cross_experiment_errors_and_warnings",
    renderer_module="cross_experiment_errors_and_warnings",
    output_path_template="plots/cross_experiment/errors_and_warnings__OUTPUT_EXT__",
    report_kwargs={
        "caption": "report/captions/cross_experiment_errors_and_warnings.rst",
        "category": "Cross-Experiment Errors and Warnings",
        "labels": '{"figure": "Per-experiment health roll-up"}',
    },
    wildcards=(),
    resources_yaml="mem_mb=1000, time_min=5",
    log_path_template="_logs/plots/cross_experiment_errors_and_warnings.log",
)

_TMPL_CROSS_EXPERIMENT_DISK_UTILIZATION = RuleSpecTemplate(
    rule_name="plot_cross_experiment_disk_utilization",
    renderer_module="cross_experiment_disk_utilization",
    output_path_template="plots/cross_experiment/disk_utilization__OUTPUT_EXT__",
    report_kwargs={
        "caption": "report/captions/cross_experiment_disk_utilization.rst",
        "category": "Cross-Experiment Disk Utilization",
        "labels": '{"figure": "Per-experiment disk utilization"}',
    },
    wildcards=(),
    resources_yaml="mem_mb=1000, time_min=5",
    log_path_template="_logs/plots/cross_experiment_disk_utilization.log",
)

# eda_compute_sensitivity: the in-report EDA adapter for the compute-sensitivity
# family. Conditional (predicate has_eda_artifact — gated on the master carrying an
# EDA artifact). Emits the config_diff_maps figure under master-rooted plots/eda/;
# lands under the existing "Key Results" category with its own subcategory rather
# than adding a sidebar category. The source-side builder
# (workflow.py _build_plot_rule_block_eda_compute_sensitivity) DERIVES its
# report_kwargs from this template, so category parity is structural rather than
# test-enforced. tests/test_reporting_set_cosourcing.py checks only that the b4b
# set carries this template and that its category is one the b4b set ranks; its
# builder-versus-template comparison covers the `default` and `benchmarking` sets,
# neither of which carries this template.
_TMPL_EDA_COMPUTE_SENSITIVITY = RuleSpecTemplate(
    rule_name="plot_eda_compute_sensitivity",
    renderer_module="eda_compute_sensitivity",
    output_path_template="plots/eda/config_diff_maps__OUTPUT_EXT__",
    report_kwargs={
        "caption": "report/captions/eda_compute_sensitivity.rst",
        "category": "Key Results",
        "subcategory": "Compute-config EDA",
        "labels": '{"figure": "Config-diff maps"}',
    },
    wildcards=(),
    resources_yaml="mem_mb=4000, time_min=10",
    log_path_template="_logs/plots/eda_compute_sensitivity.log",
)

# dem-resolution: the in-report EDA adapter for the DEM-resolution family.
# FOUR figures under ONE RendererSelection reusing builder key
# eda_compute_sensitivity -- the same one-selection-N-templates shape
# per_sim_per_member uses for its two figures. All four share renderer_module
# "eda_compute_sensitivity" (the _cli entrypoint and the _OUTPUT_EXT_BY_RENDERER
# key are per-MODULE, not per-figure), and differ only in plot ID, caption, label
# and log path. Tuple order is the figures' authored reading order (cost/error
# headline -> error distribution -> spatial diff -> coupling table); the report's
# within-category ordering is by output basename, so this order is the EMISSION
# order, not the display order.
_TMPL_DEM_RESOLUTION_COST_ERROR = RuleSpecTemplate(
    rule_name="plot_eda_dem_resolution_cost_error",
    renderer_module="eda_compute_sensitivity",
    output_path_template="plots/eda/dem_resolution_cost_error__OUTPUT_EXT__",
    report_kwargs={
        "caption": "report/captions/eda_dem_resolution_cost_error.rst",
        "category": "Key Results",
        "subcategory": "DEM-resolution EDA",
        "labels": '{"figure": "Cost vs error"}',
    },
    wildcards=(),
    resources_yaml="mem_mb=4000, time_min=10",
    log_path_template="_logs/plots/dem_resolution_cost_error.log",
)

_TMPL_DEM_RESOLUTION_ERROR_ECDF = RuleSpecTemplate(
    rule_name="plot_eda_dem_resolution_error_ecdf",
    renderer_module="eda_compute_sensitivity",
    output_path_template="plots/eda/dem_resolution_error_ecdf__OUTPUT_EXT__",
    report_kwargs={
        "caption": "report/captions/eda_dem_resolution_error_ecdf.rst",
        "category": "Key Results",
        "subcategory": "DEM-resolution EDA",
        "labels": '{"figure": "Depth-error ECDF"}',
    },
    wildcards=(),
    resources_yaml="mem_mb=4000, time_min=10",
    log_path_template="_logs/plots/dem_resolution_error_ecdf.log",
)

_TMPL_DEM_RESOLUTION_DIFF_MAPS = RuleSpecTemplate(
    rule_name="plot_eda_dem_resolution_diff_maps",
    renderer_module="eda_compute_sensitivity",
    output_path_template="plots/eda/dem_resolution_diff_maps__OUTPUT_EXT__",
    report_kwargs={
        "caption": "report/captions/eda_dem_resolution_diff_maps.rst",
        "category": "Key Results",
        "subcategory": "DEM-resolution EDA",
        "labels": '{"figure": "Depth-difference maps"}',
    },
    wildcards=(),
    resources_yaml="mem_mb=4000, time_min=10",
    log_path_template="_logs/plots/dem_resolution_diff_maps.log",
)

_TMPL_DEM_RESOLUTION_COUPLING_TABLE = RuleSpecTemplate(
    rule_name="plot_eda_dem_resolution_coupling_table",
    renderer_module="eda_compute_sensitivity",
    output_path_template="plots/eda/dem_resolution_coupling_table__OUTPUT_EXT__",
    report_kwargs={
        "caption": "report/captions/eda_dem_resolution_coupling_table.rst",
        "category": "Key Results",
        "subcategory": "DEM-resolution EDA",
        "labels": '{"figure": "Resolution x coupling junctions"}',
    },
    wildcards=(),
    resources_yaml="mem_mb=4000, time_min=10",
    log_path_template="_logs/plots/dem_resolution_coupling_table.log",
)

# The standard multisim set: its eight renderers, in emission order. The count is
# len(_STANDARD_SELECTION) below, not the intersection with _BENCHMARKING_SELECTION.
# Emitted by the `_emit_active_set_plot_rules` dispatch in `generate_snakefile_content`
# (workflow.py). per_sim declares one figure, the per-event page.
_STANDARD_SELECTION: tuple[RendererSelection, ...] = (
    RendererSelection("system_overview", rule_spec_template=(_TMPL_SYSTEM_OVERVIEW,)),
    RendererSelection(
        "per_sim",
        rule_spec_template=(
            RuleSpecTemplate(
                rule_name="plot_per_sim_event_page",
                renderer_module="per_sim_event_page",
                output_path_template="plots/per_sim/{event_id}/event_page__evt.{event_id}__OUTPUT_EXT__",
                report_kwargs={
                    "caption": "report/captions/per_sim_event_page.rst",
                    "category": "Per Simulation Results",
                    "labels": (
                        '(lambda w: {"figure": "Simulation results", '
                        '"event": _report_label_value(_EVENT_SIDEBAR_LABELS, w.event_id, "event")})'
                    ),
                },
                wildcards=("event_id",),
                resources_yaml="mem_mb=8000, time_min=30",
                log_path_template="_logs/plots/per_sim_event_page_{event_id}.log",
            ),
        ),
    ),
    RendererSelection("per_analysis_summary", rule_spec_template=(_TMPL_PER_ANALYSIS_SUMMARY,)),
    RendererSelection("scenario_status_appendix", rule_spec_template=(_TMPL_SCENARIO_STATUS_APPENDIX,)),
    RendererSelection("errors_and_warnings", rule_spec_template=(_TMPL_ERRORS_AND_WARNINGS,)),
    RendererSelection("disk_utilization", rule_spec_template=(_TMPL_DISK_UTILIZATION,)),
    RendererSelection("metadata", rule_spec_template=(_TMPL_METADATA,)),
    RendererSelection("workflow_performance", rule_spec_template=(_TMPL_WORKFLOW_PERFORMANCE,)),
)

# The SWEEP set: the seven unconditional renderers the master and reprocess-master
# generators share, plus ONE conditional renderer (per_sim_per_member). This tuple is
# the sensitivity-shaped bundle WITHOUT the benchmarking figure; _BENCHMARKING_SELECTION
# below extends it with that figure and is the tuple carrying two conditionals. Emitted
# by the `_emit_active_set_plot_rules` dispatch in `generate_master_snakefile_content`
# and in `generate_reprocess_master_snakefile_content` (workflow.py). per_sim_per_member
# declares two figures.
_SWEEP_SELECTION: tuple[RendererSelection, ...] = (
    RendererSelection("system_overview", rule_spec_template=(_TMPL_SYSTEM_OVERVIEW,)),
    RendererSelection("per_analysis_summary", rule_spec_template=(_TMPL_PER_ANALYSIS_SUMMARY,)),
    RendererSelection("scenario_status_appendix", rule_spec_template=(_TMPL_SCENARIO_STATUS_APPENDIX,)),
    RendererSelection("errors_and_warnings", rule_spec_template=(_TMPL_ERRORS_AND_WARNINGS,)),
    RendererSelection("disk_utilization", rule_spec_template=(_TMPL_DISK_UTILIZATION,)),
    RendererSelection("metadata", rule_spec_template=(_TMPL_METADATA,)),
    RendererSelection("workflow_performance", rule_spec_template=(_TMPL_WORKFLOW_PERFORMANCE,)),
    RendererSelection(
        "per_sim_per_member",
        predicate_key="has_member_event_pairs",
        rule_spec_template=(
            RuleSpecTemplate(
                rule_name="plot_per_sim_per_member_peak_flood_depth",
                # The REAL module; member-routing is via the --member-id flag, not a separate
                # module. Deliberately NOT a per_sim_per_member_* name to match rule_name:
                # no such module exists on disk, and this field is copied verbatim into the
                # bundle-side regen RuleSpec (bundle/snakefile_generator.py), whose rule
                # would then shell out to a missing module. Naming the real module changes
                # nothing else: besides that shell-out this field reaches only
                # output_ext_for, whose per_sim_* and per_sim_per_member_* entries are
                # identical under both backends, and plot IDs are minted from
                # renderer_kind, not from this value.
                renderer_module="per_sim_peak_flood_depth",
                output_path_template="plots/sensitivity/per_sim/member-{member_id}/{event_id}/peak_flood_depth__member.{member_id}__evt.{event_id}__OUTPUT_EXT__",
                report_kwargs={
                    "caption": "report/captions/per_sim_peak_flood_depth.rst",
                    "category": "Per Simulation Results",
                    "labels": (
                        '(lambda w: {"figure": "Peak flood depth", '
                        '"member": _report_label_value(_MEMBER_LABELS, w.member_id, "member"), '
                        '"event": _report_label_value(_EVENT_SIDEBAR_LABELS, w.event_id, "event")})'
                    ),
                },
                wildcards=("member_id", "event_id"),
                resources_yaml="mem_mb=4000, time_min=15",
                log_path_template="_logs/plots/per_sim_per_member_peak_flood_depth_member-{member_id}_{event_id}.log",
            ),
            RuleSpecTemplate(
                rule_name="plot_per_sim_per_member_conduit_flow",
                # The REAL module — see the note on the peak_flood_depth template above.
                renderer_module="per_sim_conduit_flow",
                output_path_template="plots/sensitivity/per_sim/member-{member_id}/{event_id}/conduit_flow__member.{member_id}__evt.{event_id}__OUTPUT_EXT__",
                report_kwargs={
                    "caption": "report/captions/per_sim_conduit_flow.rst",
                    "category": "Per Simulation Results",
                    "labels": (
                        '(lambda w: {"figure": "Conduit flow", '
                        '"member": _report_label_value(_MEMBER_LABELS, w.member_id, "member"), '
                        '"event": _report_label_value(_EVENT_SIDEBAR_LABELS, w.event_id, "event")})'
                    ),
                },
                wildcards=("member_id", "event_id"),
                resources_yaml="mem_mb=4000, time_min=15",
                log_path_template="_logs/plots/per_sim_per_member_conduit_flow_member-{member_id}_{event_id}.log",
                predicate_key="has_swmm_link_outputs",
            ),
        ),
    ),
)

# The benchmarking set is the sweep-shaped bundle plus the benchmarking figure,
# which plots against a compute-configuration axis. Split out so the `default`
# sentinel can resolve a sweep to _SWEEP_SELECTION, whose per_sim_per_member
# figures carry the (member_id, event_id) axis a master-scope report needs.
_BENCHMARKING_SELECTION: tuple[RendererSelection, ...] = _SWEEP_SELECTION + (
    RendererSelection(
        "sensitivity_benchmarking",
        predicate_key="has_independent_vars",
        rule_spec_template=(
            RuleSpecTemplate(
                rule_name="plot_sensitivity_benchmarking",
                renderer_module="sensitivity_benchmarking",
                output_path_template="plots/sensitivity/benchmarking/benchmarking__{independent_var}.vs.total__OUTPUT_EXT__",
                report_kwargs={
                    "caption": "report/captions/sensitivity_benchmarking.rst",
                    "category": "Key Results",
                    "subcategory": "Benchmarking",
                    # `__MODEL_ARM_LABEL__` is the whole-fragment sentinel `_emit_plot_rule`
                    # substitutes from `ctx.model_arm` (mirroring `__OUTPUT_EXT__`). It expands to
                    # `, "models": "TRITON-SWMM"` / `, "models": "TRITON"`, or to NOTHING on a
                    # master carrying neither TRITON arm. `models` is the same key the combined
                    # report already emits, so the two surfaces name one facet, not two synonyms.
                    "labels": '{"independent_var": "{independent_var}", '
                    '"figure": "vs Total runtime"__MODEL_ARM_LABEL__}',
                },
                wildcards=("independent_var",),
                resources_yaml="mem_mb=4000, time_min=10",
                log_path_template="_logs/plots/sensitivity_benchmarking_{independent_var}.log",
            ),
        ),
    ),
)

# The compute-sensitivity set: the benchmarking (sensitivity-master) selection plus
# the in-report EDA adapter, gated on has_eda_artifact. Same category_order as
# benchmarking, because the EDA figures land under the existing "Key Results"
# category rather than a new sidebar category. Selected via
# report_config.reporting_set="compute-sensitivity".
_COMPUTE_SENSITIVITY_SELECTION: tuple[RendererSelection, ...] = _BENCHMARKING_SELECTION + (
    RendererSelection(
        "eda_compute_sensitivity",
        predicate_key="has_eda_artifact",
        rule_spec_template=(_TMPL_EDA_COMPUTE_SENSITIVITY,),
    ),
)

# The dem-resolution set: the benchmarking (sensitivity-master) selection
# plus the DEM-resolution EDA family, gated on has_eda_artifact. Structurally
# identical to _COMPUTE_SENSITIVITY_SELECTION -- same builder key, same predicate,
# same category_order -- differing only in carrying FOUR rule_spec_templates where
# compute-sensitivity carries one. Selected via
# report_config.reporting_set="dem-resolution", paired with a
# system.target_dem_resolution sweep.
_DEM_RESOLUTION_SELECTION: tuple[RendererSelection, ...] = _BENCHMARKING_SELECTION + (
    RendererSelection(
        "eda_compute_sensitivity",
        predicate_key="has_eda_artifact",
        rule_spec_template=(
            _TMPL_DEM_RESOLUTION_COST_ERROR,
            _TMPL_DEM_RESOLUTION_ERROR_ECDF,
            _TMPL_DEM_RESOLUTION_DIFF_MAPS,
            _TMPL_DEM_RESOLUTION_COUPLING_TABLE,
        ),
    ),
)

# b4b (raw-output byte-for-byte comparison). Its one figure template below, plus the
# config_diff_maps template, sit under ONE RendererSelection reusing builder key
# eda_compute_sensitivity (the _cli entrypoint and _OUTPUT_EXT_BY_RENDERER key are
# per-MODULE, not per-figure), so the figure needs no generator edit of its own: the
# existing EDA builder emits one rule per template. predicate_key is
# has_preserved_raw_outputs (NOT has_eda_artifact) because the b4b figures compare raw
# solver outputs, which must have survived the post-processing clear for the figures
# to have anything to read. That predicate gates EMISSION only: the rule-all
# enumeration of EDA figures is a separate gate in workflow.py, on has_eda_artifact,
# whatever this predicate_key says (see the comment inside _B4B_SELECTION below).
_TMPL_B4B_CLEAN_IDENTITY = RuleSpecTemplate(
    rule_name="plot_raw_byte_identity",
    renderer_module="eda_compute_sensitivity",
    output_path_template="plots/eda/raw_byte_identity__OUTPUT_EXT__",
    report_kwargs={
        "caption": "report/captions/raw_byte_identity.rst",
        "category": "Key Results",
        "subcategory": "Byte-for-byte comparison",
        "labels": '{"figure": "Cross-hardware raw byte identity"}',
    },
    wildcards=(),
    resources_yaml="mem_mb=4000, time_min=10",
    log_path_template="_logs/plots/raw_byte_identity.log",
)

_B4B_SELECTION: tuple[RendererSelection, ...] = _BENCHMARKING_SELECTION + (
    RendererSelection(
        "eda_compute_sensitivity",
        predicate_key="has_preserved_raw_outputs",
        # config_diff_maps (the cross-sim byte-identity grouping vs the serial-CPU
        # baseline) joins the b4b set's single eda_compute_sensitivity selection so it
        # renders into analysis_report.html for the b4b masters. It MUST share this ONE
        # selection: eda_rule_spec_templates() below, which the workflow builder calls
        # for both emission and rule-all enumeration, returns only the FIRST
        # eda_compute_sensitivity selection's templates, so a second same-key selection
        # would be silently dropped from both emission and the rule-all enumeration. It
        # therefore inherits the has_preserved_raw_outputs predicate; that is inert here
        # because the b4b masters preserve raw outputs by contract (the b4b figures
        # require it), and the rule-all enumeration gate is has_eda_artifact, so both
        # gates hold together on a b4b run.
        rule_spec_template=(_TMPL_EDA_COMPUTE_SENSITIVITY, _TMPL_B4B_CLEAN_IDENTITY),
    ),
)

#: The compute-configuration sensitivity axes, as canonical stripped analysis_config
#: field names. A benchmarking-family figure plots against one of these, so a sweep
#: that varies none of them has no x-axis for it, and the run-entry check refuses the
#: set rather than letting it render empty. Compared at run entry, via required_axes,
#: against the sweep's varied axes (sensitivity.analysis_independent_vars union
#: system_independent_vars), whose entries are already stripped of their column prefix.
_COMPUTE_CONFIG_AXES: tuple[str, ...] = (
    "run_mode",
    "n_mpi_procs",
    "n_omp_threads",
    "n_gpus",
    "n_nodes",
    "hpc_ensemble_partition",
    # `mem_gb_per_cpu` is included as a JUDGEMENT, not a measurement: memory per CPU
    # changes node packing and bandwidth contention and therefore wall clock, which is
    # what the benchmarking figures plot. Do NOT cite synthetic_experiment.py's config
    # tuple as the ground -- it names this field because every row needs a memory
    # request, and across the whole default matrix the value is a function of
    # run_mode (8 on every gpu row, 2 on every cpu row) with no memory-only contrast.
    "mem_gb_per_cpu",
    # DELIBERATELY EXCLUDED: `hpc_time_min_per_sim`. A walltime REQUEST changes nothing
    # about how the computation runs, so a sweep varying only it has no performance
    # axis to plot.
)

REPORTING_SETS: dict[str, ReportingSet] = {
    "default": ReportingSet(
        name="default",
        category_order=_STANDARD_CATEGORY_ORDER,
        renderer_selection=_STANDARD_SELECTION,
        validator_key="none",
        shape="event_ensemble",
        required_axes=(),
    ),
    "benchmarking": ReportingSet(
        name="benchmarking",
        category_order=_STANDARD_CATEGORY_ORDER,
        renderer_selection=_BENCHMARKING_SELECTION,
        validator_key="benchmarking",
        shape="sensitivity",
        required_axes=(_COMPUTE_CONFIG_AXES,),
    ),
    "sensitivity": ReportingSet(
        name="sensitivity",
        category_order=_STANDARD_CATEGORY_ORDER,
        renderer_selection=_SWEEP_SELECTION,
        validator_key="none",
        shape="sensitivity",
        required_axes=(),
    ),
    "combined": ReportingSet(
        name="combined",
        # The rest of the combined sidebar is DYNAMIC: it depends on which child bundles
        # sit under child_crates/, known only at combine time. This static tuple carries
        # only the FIXED cross-experiment categories, which rank first. The combined
        # generator (render_combined_report_via_snakemake) APPENDS the children's own
        # figure categories -- each child set's category_order, filtered to categories one
        # of that set's templates declares, de-duplicated across children -- and threads the
        # full list into _react_surgery. A child's harvested figure keeps its ORIGINAL
        # category, so each experiment's errors_and_warnings figure sits under "Errors
        # and Warnings", in a subcategory whose name starts with that experiment's id.
        category_order=(
            "Combine Provenance & Compatibility",
            "Cross-Experiment Results",
            "Cross-Experiment Errors and Warnings",
            "Cross-Experiment Disk Utilization",
        ),
        renderer_selection=(
            RendererSelection(
                "cross_experiment_compatibility",
                rule_spec_template=(_TMPL_CROSS_EXPERIMENT_COMPATIBILITY,),
            ),
            RendererSelection(
                "cross_experiment_intercomparison",
                rule_spec_template=(_TMPL_CROSS_EXPERIMENT_INTERCOMPARISON,),
            ),
            RendererSelection(
                "cross_experiment_intercomparison_maps",
                rule_spec_template=(_TMPL_CROSS_EXPERIMENT_INTERCOMPARISON_MAPS,),
            ),
            RendererSelection(
                "cross_experiment_errors_and_warnings",
                rule_spec_template=(_TMPL_CROSS_EXPERIMENT_ERRORS_AND_WARNINGS,),
            ),
            RendererSelection(
                "cross_experiment_disk_utilization",
                rule_spec_template=(_TMPL_CROSS_EXPERIMENT_DISK_UTILIZATION,),
            ),
        ),
        validator_key="none",
        shape="cross_experiment",
        required_axes=(),
    ),
    "compute-sensitivity": ReportingSet(
        name="compute-sensitivity",
        category_order=_STANDARD_CATEGORY_ORDER,
        renderer_selection=_COMPUTE_SENSITIVITY_SELECTION,
        validator_key="benchmarking",
        shape="sensitivity",
        required_axes=(_COMPUTE_CONFIG_AXES,),
    ),
    "dem-resolution": ReportingSet(
        name="dem-resolution",
        category_order=_STANDARD_CATEGORY_ORDER,
        renderer_selection=_DEM_RESOLUTION_SELECTION,
        validator_key="benchmarking",
        shape="sensitivity",
        required_axes=(("target_dem_resolution",),),
    ),
    "b4b": ReportingSet(
        name="b4b",
        category_order=_STANDARD_CATEGORY_ORDER,
        renderer_selection=_B4B_SELECTION,
        validator_key="benchmarking",
        shape="sensitivity",
        required_axes=(_COMPUTE_CONFIG_AXES,),
    ),
}


def get_reporting_set(name: str) -> ReportingSet:
    """Return the registered ReportingSet for ``name`` or raise KeyError.

    Callers that need a validating lookup against config use
    config.report.validate_active_reporting_set (run-entry, lazy-imports this
    module). This bare accessor is for callers that already hold a registered name
    -- a hardcoded literal such as "combined" or "default", a name a resolver has
    already validated, or a key read from ``REPORTING_SETS`` itself -- so an
    unregistered name here is a programming error and surfaces as KeyError.
    """
    return REPORTING_SETS[name]


def renderer_active(builder_key: str, disabled: list[str] | None) -> bool:
    """Return False when ``builder_key`` is disabled for this invocation.

    The single source of truth for per-plot disable (report_config.disabled_renderers).
    Every emission site (the workflow.py dispatcher, the bundle regeneration generator)
    AND every rule all / render_report input-list site calls this — a site that
    filters emission without filtering enumeration yields MissingInputException;
    the inverse yields an orphan rule. An unknown ``builder_key`` (a typo) never
    matches a selection entry, so it silently drops nothing here; the run-entry
    ``validate_active_reporting_set`` is where such a key raises ConfigurationError.
    """
    return builder_key not in (disabled or ())


def set_carries(reporting_set: ReportingSet, builder_key: str) -> bool:
    """Return True when ``reporting_set`` carries a selection for ``builder_key``.

    The membership half of the parity contract ``renderer_active`` states above: a
    ``rule all`` entry gated only on ``renderer_active`` asks whether a renderer was
    DISABLED and never whether the active set carries it at all, so a set that omits
    the renderer still has its output enumerated and no rule produces it. Emission
    already consults membership by construction -- the dispatcher iterates
    ``renderer_selection`` -- so this is what lets an enumeration site ask the same
    question the emission site answers implicitly.

    Membership is by ``builder_key`` and not by rule_name, matching the dispatcher's
    own loop; a selection whose per-template predicates all fail still counts as
    carried, because emission and enumeration then agree via the predicate rather
    than via membership.
    """
    return any(sel.builder_key == builder_key for sel in reporting_set.renderer_selection)


def eda_rule_spec_templates(reporting_set: ReportingSet) -> tuple[RuleSpecTemplate, ...]:
    """Return ``reporting_set``'s ``eda_compute_sensitivity`` rule_spec_templates, or ().

    SINGLE SOURCE for the question "does this set enumerate EDA figures as report
    targets, and how many?" — shared by the workflow builder's
    ``_eda_rule_spec_templates`` (which drives the rule emission and the two rule all /
    render_report enumeration sites) and by
    ``analysis_validation.check_eda_calc_ran``, which must answer the same question at
    validate time with no workflow builder in hand. Lives here, on the registry, rather
    than on the builder, because the answer is a property of the SET.

    Returns the FIRST such selection's templates: a second same-key selection is
    silently dropped by every consumer (see the _B4B_SELECTION comment).
    """
    for sel in reporting_set.renderer_selection:
        if sel.builder_key == "eda_compute_sensitivity":
            return sel.rule_spec_template
    return ()


class ReportingSetCompositionError(ValueError):
    """Raised when a list of reporting-set names cannot be composed into one set."""


def compose_reporting_sets(names: list[str] | tuple[str, ...]) -> ReportingSet:
    """Compose a list of registered reporting-set names into ONE ReportingSet.

    A ONE-NAME list returns the registered object ITSELF, not a copy, so a
    single-set config reaches every consumer exactly as a bare set name does. The
    consumers that take the OBJECT (the workflow dispatcher, the bundle regeneration
    generator, analysis_validation, the render-path category-order blocks) need no
    change for a list, so on the registry side composing it here is the whole of it.
    The exception is the combine path: bundle/combined_snakefile_generator.py
    resolves each child bundle's set NAMES itself and raises ConfigurationError on a
    child run with a composed set, because it reads one set's templates.

    Merging is at the TEMPLATE level, not the selection level: two members
    contributing the same builder_key yield ONE selection carrying the union of
    their rule_spec_templates, because the dispatcher calls one builder per
    selection and eda_rule_spec_templates returns only the first same-key
    selection's templates.

    shape is the homogeneity key for a list, as declared on ReportingSet.shape.
    required_axes is ALL-of over the outer tuple, so composing concatenates the
    members' families and a composed sweep must satisfy every one of them.
    """
    names = tuple(names)
    if not names:
        raise ReportingSetCompositionError(
            "report.reporting_set resolved to an empty list; name at least one "
            f"registered set. Registered sets: {sorted(REPORTING_SETS)}."
        )
    duplicated = sorted({n for n in names if names.count(n) > 1})
    if duplicated:
        raise ReportingSetCompositionError(
            f"report.reporting_set names the same set more than once: {duplicated}. Each set may appear at most once."
        )
    members = [REPORTING_SETS[n] for n in names]
    if len(members) == 1:
        return members[0]

    shapes = {m.shape for m in members}
    if len(shapes) != 1:
        raise ReportingSetCompositionError(
            "report.reporting_set composes sets of differing analysis shapes: "
            f"{ {m.name: m.shape for m in members} }. Every named set must declare "
            "the same shape."
        )
    orders = {m.category_order for m in members}
    if len(orders) != 1:  # forward guard: unreachable while every set of one shape shares one order
        raise ReportingSetCompositionError(
            "report.reporting_set composes sets with differing sidebar category "
            f"orders: {sorted(m.name for m in members)}. A composed report has one "
            "sidebar, so every named set must declare the same category_order."
        )

    required: list[tuple[str, ...]] = []
    for m in members:
        for family in m.required_axes or ():
            if family not in required:
                required.append(family)

    key_order: list[str] = []
    contributors: dict[str, list[tuple[str, RendererSelection]]] = {}
    for m in members:
        for sel in m.renderer_selection:
            if sel.builder_key not in contributors:
                key_order.append(sel.builder_key)  # first-appearance order
                contributors[sel.builder_key] = []
            contributors[sel.builder_key].append((m.name, sel))

    merged: list[RendererSelection] = []
    for key in key_order:
        contribs = contributors[key]
        predicates = {sel.predicate_key for _, sel in contribs if sel.predicate_key}
        if len(predicates) > 1:
            raise ReportingSetCompositionError(
                f"report.reporting_set composes sets that gate renderer {key!r} on "
                f"different conditions: { {n: sel.predicate_key for n, sel in contribs} }. "
                "A composed report emits each renderer once, under one condition, so "
                "these sets cannot be combined. Name one of them."
            )
        templates: list[RuleSpecTemplate] = []
        claimed: dict[str, tuple[str, RuleSpecTemplate]] = {}
        for owner, sel in contribs:
            for tmpl in sel.rule_spec_template:
                prior = claimed.get(tmpl.rule_name)
                if prior is None:
                    claimed[tmpl.rule_name] = (owner, tmpl)
                    templates.append(tmpl)
                elif prior[1] != tmpl:  # forward guard: colliding names are equal today
                    raise ReportingSetCompositionError(
                        f"report.reporting_set composes sets whose rule "
                        f"{tmpl.rule_name!r} is declared differently by {prior[0]!r} "
                        f"and {owner!r}. One rule name must mean one rule."
                    )
        merged.append(
            RendererSelection(
                builder_key=key,
                predicate_key=(next(iter(predicates)) if predicates else None),
                rule_spec_template=tuple(templates),
            )
        )

    return ReportingSet(
        name="+".join(names),
        category_order=next(iter(orders)),
        renderer_selection=tuple(merged),
        validator_key=("benchmarking" if any(m.validator_key == "benchmarking" for m in members) else "none"),
        shape=next(iter(shapes)),
        required_axes=tuple(required),
    )
