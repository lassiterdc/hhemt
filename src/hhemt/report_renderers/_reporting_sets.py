"""Named reporting-set registry (ADR-5).

A "reporting set" is a first-class registered object keyed by name. Each set
declares: (a) an ordered renderer selection (consumed by the workflow.py
dispatcher in Phase P1b), (b) a sidebar category-order list (consumed by the
render_report facades + _react_surgery in Phase P1a), and (c) a set-specific
run-entry validator hook (consumed at analysis.run() entry).

Import discipline (stipulation "ReportingSet validated at run-entry"): this
module MUST stay import-light — it must NOT import renderer modules or
config.report at module load, or it creates the cycle
config.report -> report_renderers._reporting_sets -> config.report. Renderer
selection is expressed as builder-key strings; conditional renderers carry a
predicate_key string the dispatcher resolves to a callable (P1b).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuleSpecTemplate:
    """Bundle-side rule facts for one renderer (P1b).

    The bundle generator (`bundle/snakefile_generator.py`) builds stub-shaped
    `RuleSpec`s from the active set's selection rather than a hand-maintained
    literal list. Each conditional/unconditional figure renderer that the bundle
    emits carries the per-renderer facts here so BOTH the source-side builders and
    the bundle generator read category/caption/wildcards/output-path from one
    place (closing the source-vs-bundle drift). Consumed by the bundle generator
    AND -- since the dem-resolution phase's D13 registry-read generalization -- by
    the source-side `_build_plot_rule_block_eda_compute_sensitivity`, which iterates
    this tuple instead of hardcoding one figure. Selections that predate that change
    omit it, so it remains a backward-compatible add (P1a entries omit it).

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


@dataclass(frozen=True)
class RendererSelection:
    """One renderer in a set's ordered selection.

    builder_key        : key into workflow.py's dispatcher builders map (P1b).
    predicate_key      : None for unconditional renderers; a string the dispatcher
                         resolves to a Callable[[inputs], bool] for conditional
                         renderers (e.g. "has_independent_vars",
                         "has_sa_event_pairs").
    rule_spec_template : () for selections the bundle generator does not emit;
                         a TUPLE of RuleSpecTemplate (one per emitted bundle
                         figure — most renderers map to one, but per_sim and
                         per_sim_per_sa each expand to two: peak_flood_depth +
                         conduit_flow) carrying the bundle-side rule facts when the
                         bundle generator data-drives this renderer (P1b). Consumed
                         ONLY by the bundle generator.
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


# Historical sidebar category order (the pre-ADR-5 _CATEGORY_ORDER, now a per-set
# field). The default/standard set uses it verbatim; benchmarking reuses it
# (its figures land under existing categories — "Key Results"/"Benchmarking").
_STANDARD_CATEGORY_ORDER: tuple[str, ...] = (
    "Workflow Status",
    "Errors and Warnings",
    "Key Results",
    "System Information",
    "Simulation Health (placeholder)",
    "Per Simulation Results",
    "Metadata",
)

# --- Bundle-side RuleSpecTemplates (P1b) -------------------------------------
# These carry the regeneration-Snakefile rule facts the bundle generator
# (bundle/snakefile_generator.py) emits, so the registry is the single source for
# each figure's category/caption/wildcards/output. The `category` here is
# cross-checked against the source-side `_build_plot_rule_block_*`
# report(category=) by tests/test_reporting_set_cosourcing.py (drift guard). The
# five non-conditional figure templates below are SHARED by the default and
# benchmarking sets (defined once, referenced from both selections).
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
# disk_utilization: previously absent from the bundle (intentional +1 drift-fix,
# P1b BUNDLE DRIFT NOTE). Facts mirror the source-side builder
# (workflow.py _build_plot_rule_block_disk_utilization); regen-only metadata.
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
# metadata (ADR-14 / C10): the Metadata page -- RO-Crate provenance summary +
# reprex reproduction guide + SLURM efficiency. Unconditional table renderer
# (emits .html under both static backends); shared by the default and
# benchmarking sets. Facts mirror the source-side builder
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
# Cross-experiment combined set (PIP-1 Phase 4). INERT on source/bundle
# generators (no workflow.py builder, no bundle snakefile rule); consumed ONLY
# by bundle/_combine.py's emit-time direct-render dispatch (F-B Flag 1(c)).
_TMPL_CROSS_EXPERIMENT_COMPATIBILITY = RuleSpecTemplate(
    rule_name="plot_cross_experiment_compatibility",
    renderer_module="cross_experiment_compatibility",
    output_path_template="plots/cross_experiment/compatibility__OUTPUT_EXT__",
    report_kwargs={
        "caption": "report/captions/cross_experiment_compatibility.rst",
        "category": "Cross-Experiment Compatibility",
        "labels": '{"figure": "Compatibility report"}',
    },
    wildcards=(),
    resources_yaml="mem_mb=1000, time_min=5",
    log_path_template="_logs/plots/cross_experiment_compatibility.log",
)

# Cross-experiment INTERCOMPARISON set (Phase 5). Same INERT-on-source posture as
# the compatibility template above: consumed ONLY by bundle/_combine.py's emit-time
# direct-render dispatch. Projects the combined_intercomparison.json read-model
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

# Cross-experiment INTERCOMPARISON MAPS (b3, Phase 5). Same INERT-on-source posture;
# direct-rendered by _combine.py. Re-reads child_crates/*/sensitivity_datatree.zarr at
# render time (Option R — no emit-time artifact, CR4-safe). Lands under the SAME
# "Cross-Experiment Results" category as the scalar intercomparison table.
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

# Cross-experiment ERRORS-AND-WARNINGS roll-up (Phase 5, F2). Same INERT-on-source
# posture; direct-rendered by _combine.py. Reads each child_crates/{eid}/validation_report.json
# at render time (Option R -- no emit-time artifact, CR4-safe). Restores a DISCOVERABLE
# top-level E&W surface that v8/a2 removed (the per-experiment E&W stays a {eid} subcategory).
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

# eda_compute_sensitivity (R11): the in-report EDA adapter for the
# compute-sensitivity family. Conditional (predicate has_eda_artifact — gated on
# the master carrying an EDA artifact). Emits the config_diff_maps figure under
# master-rooted plots/eda/; lands under "Key Results" (Decision 1, no new sidebar
# category). The source-side builder
# (workflow.py _build_plot_rule_block_eda_compute_sensitivity) DERIVES its
# report_kwargs from this template, so category parity is structural rather than
# test-enforced. Note tests/test_reporting_set_cosourcing.py does NOT cover this
# template: its two tests exercise the `default` and `benchmarking` sets, neither
# of which carries the eda_compute_sensitivity selection.
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

# dem-resolution (D13): the in-report EDA adapter for the DEM-resolution family.
# FOUR figures under ONE RendererSelection reusing builder key
# eda_compute_sensitivity -- the same one-selection-N-templates shape per_sim and
# per_sim_per_sa already use. All four share renderer_module
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

# The standard multisim set: the six common renderers, in emission order
# (matches workflow.py:1913-1918 today). per_sim expands to two bundle figures.
_STANDARD_SELECTION: tuple[RendererSelection, ...] = (
    RendererSelection("system_overview", rule_spec_template=(_TMPL_SYSTEM_OVERVIEW,)),
    RendererSelection(
        "per_sim",
        rule_spec_template=(
            RuleSpecTemplate(
                rule_name="plot_per_sim_peak_flood_depth",
                renderer_module="per_sim_peak_flood_depth",
                output_path_template="plots/per_sim/{event_id}/peak_flood_depth__evt.{event_id}__OUTPUT_EXT__",
                report_kwargs={
                    "caption": "report/captions/per_sim_peak_flood_depth.rst",
                    "category": "Per Simulation Results",
                    "labels": '{"event_id": "{event_id}", "figure": "Peak Flood Depth"}',
                },
                wildcards=("event_id",),
                resources_yaml="mem_mb=4000, time_min=15",
                log_path_template="_logs/plots/per_sim_peak_flood_depth_{event_id}.log",
            ),
            RuleSpecTemplate(
                rule_name="plot_per_sim_conduit_flow",
                renderer_module="per_sim_conduit_flow",
                output_path_template="plots/per_sim/{event_id}/conduit_flow__evt.{event_id}__OUTPUT_EXT__",
                report_kwargs={
                    "caption": "report/captions/per_sim_conduit_flow.rst",
                    "category": "Per Simulation Results",
                    "labels": '{"event_id": "{event_id}", "figure": "Conduit Flow"}',
                },
                wildcards=("event_id",),
                resources_yaml="mem_mb=4000, time_min=15",
                log_path_template="_logs/plots/per_sim_conduit_flow_{event_id}.log",
            ),
        ),
    ),
    RendererSelection("per_analysis_summary", rule_spec_template=(_TMPL_PER_ANALYSIS_SUMMARY,)),
    RendererSelection("scenario_status_appendix", rule_spec_template=(_TMPL_SCENARIO_STATUS_APPENDIX,)),
    RendererSelection("errors_and_warnings", rule_spec_template=(_TMPL_ERRORS_AND_WARNINGS,)),
    RendererSelection("disk_utilization", rule_spec_template=(_TMPL_DISK_UTILIZATION,)),
    RendererSelection("metadata", rule_spec_template=(_TMPL_METADATA,)),
)

# The benchmarking (sensitivity-master) set: the five common renderers shared by
# the master/reprocess generators (workflow.py:6391-6415), plus the two
# conditional sensitivity renderers gated by predicate. per_sim_per_sa expands to
# two bundle figures.
_BENCHMARKING_SELECTION: tuple[RendererSelection, ...] = (
    RendererSelection("system_overview", rule_spec_template=(_TMPL_SYSTEM_OVERVIEW,)),
    RendererSelection("per_analysis_summary", rule_spec_template=(_TMPL_PER_ANALYSIS_SUMMARY,)),
    RendererSelection("scenario_status_appendix", rule_spec_template=(_TMPL_SCENARIO_STATUS_APPENDIX,)),
    RendererSelection("errors_and_warnings", rule_spec_template=(_TMPL_ERRORS_AND_WARNINGS,)),
    RendererSelection("disk_utilization", rule_spec_template=(_TMPL_DISK_UTILIZATION,)),
    RendererSelection("metadata", rule_spec_template=(_TMPL_METADATA,)),
    RendererSelection(
        "per_sim_per_sa",
        predicate_key="has_sa_event_pairs",
        rule_spec_template=(
            RuleSpecTemplate(
                rule_name="plot_per_sim_per_sa_peak_flood_depth",
                renderer_module="per_sim_per_sa_peak_flood_depth",
                output_path_template="plots/sensitivity/per_sim/sa-{sa_id}/{event_id}/peak_flood_depth__sa.{sa_id}__evt.{event_id}__OUTPUT_EXT__",
                report_kwargs={
                    "caption": "report/captions/per_sim_peak_flood_depth.rst",
                    "category": "Per Simulation Results",
                    "labels": '{"sa_id": "{sa_id}", "event_id": "{event_id}", "figure": "Peak Flood Depth"}',
                },
                wildcards=("sa_id", "event_id"),
                resources_yaml="mem_mb=4000, time_min=15",
                log_path_template="_logs/plots/per_sim_per_sa_peak_flood_depth_sa-{sa_id}_{event_id}.log",
            ),
            RuleSpecTemplate(
                rule_name="plot_per_sim_per_sa_conduit_flow",
                renderer_module="per_sim_per_sa_conduit_flow",
                output_path_template="plots/sensitivity/per_sim/sa-{sa_id}/{event_id}/conduit_flow__sa.{sa_id}__evt.{event_id}__OUTPUT_EXT__",
                report_kwargs={
                    "caption": "report/captions/per_sim_conduit_flow.rst",
                    "category": "Per Simulation Results",
                    "labels": '{"sa_id": "{sa_id}", "event_id": "{event_id}", "figure": "Conduit Flow"}',
                },
                wildcards=("sa_id", "event_id"),
                resources_yaml="mem_mb=4000, time_min=15",
                log_path_template="_logs/plots/per_sim_per_sa_conduit_flow_sa-{sa_id}_{event_id}.log",
            ),
        ),
    ),
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
                    "labels": '{"independent_var": "{independent_var}", "figure": "vs Total runtime"}',
                },
                wildcards=("independent_var",),
                resources_yaml="mem_mb=4000, time_min=10",
                log_path_template="_logs/plots/sensitivity_benchmarking_{independent_var}.log",
            ),
        ),
    ),
)

# The compute-sensitivity set (R11): the benchmarking (sensitivity-master)
# selection plus the in-report EDA adapter, gated on has_eda_artifact. Same
# category_order as benchmarking (the EDA figures land under the existing "Key
# Results" category — Decision 1, no new sidebar category). Selected via
# report_config.reporting_set="compute-sensitivity".
_COMPUTE_SENSITIVITY_SELECTION: tuple[RendererSelection, ...] = _BENCHMARKING_SELECTION + (
    RendererSelection(
        "eda_compute_sensitivity",
        predicate_key="has_eda_artifact",
        rule_spec_template=(_TMPL_EDA_COMPUTE_SENSITIVITY,),
    ),
)

# The dem-resolution set (D13): the benchmarking (sensitivity-master) selection
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

REPORTING_SETS: dict[str, ReportingSet] = {
    "default": ReportingSet(
        name="default",
        category_order=_STANDARD_CATEGORY_ORDER,
        renderer_selection=_STANDARD_SELECTION,
        validator_key="none",
    ),
    "benchmarking": ReportingSet(
        name="benchmarking",
        category_order=_STANDARD_CATEGORY_ORDER,
        renderer_selection=_BENCHMARKING_SELECTION,
        validator_key="benchmarking",
    ),
    "combined": ReportingSet(
        name="combined",
        # Option B: the per-experiment categories are DYNAMIC (one per child_crates/{eid},
        # known only at combine time). This static tuple carries only the FIXED bookend
        # categories; the combined generator (render_combined_report_via_snakemake) APPENDS
        # the sorted experiment ids after these fixed bookends and threads the full list
        # into _react_surgery. (v8/a2: the aggregate "Errors and Warnings" category was
        # RETIRED — each experiment carries its OWN errors_and_warnings figure natively
        # under its {eid} section via the per-experiment harvest.)
        category_order=(
            "Cross-Experiment Compatibility",
            "Cross-Experiment Results",
            "Cross-Experiment Errors and Warnings",
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
        ),
        validator_key="none",
    ),
    "compute-sensitivity": ReportingSet(
        name="compute-sensitivity",
        category_order=_STANDARD_CATEGORY_ORDER,
        renderer_selection=_COMPUTE_SENSITIVITY_SELECTION,
        validator_key="benchmarking",
    ),
    "dem-resolution": ReportingSet(
        name="dem-resolution",
        category_order=_STANDARD_CATEGORY_ORDER,
        renderer_selection=_DEM_RESOLUTION_SELECTION,
        validator_key="benchmarking",
    ),
}


def get_reporting_set(name: str) -> ReportingSet:
    """Return the registered ReportingSet for ``name`` or raise KeyError.

    Callers that need a validating lookup against config use
    config.report.validate_active_reporting_set (run-entry, lazy-imports this
    module). This bare accessor is for the render_report facades + dispatcher,
    which receive an already-validated name.
    """
    return REPORTING_SETS[name]


def renderer_active(builder_key: str, disabled: list[str] | None) -> bool:
    """Return False when ``builder_key`` is disabled for this invocation.

    The single source of truth for per-plot disable (report_config.disabled_renderers,
    Phase 3). Every emission site (the workflow.py dispatcher, the bundle harvest)
    AND every rule all / render_report input-list site calls this — a site that
    filters emission without filtering enumeration yields MissingInputException;
    the inverse yields an orphan rule. An unknown ``builder_key`` (a typo) never
    matches a selection entry, so it silently drops nothing here; the run-entry
    ``validate_active_reporting_set`` is where such a key raises ConfigurationError.
    """
    return builder_key not in (disabled or ())
