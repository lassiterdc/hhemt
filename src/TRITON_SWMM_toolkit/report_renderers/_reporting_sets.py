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
    place (closing the source-vs-bundle drift). This field is consumed ONLY by the
    bundle generator; the source-side workflow.py dispatcher ignores it, so it is
    a backward-compatible add (P1a entries omit it).

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
}


def get_reporting_set(name: str) -> ReportingSet:
    """Return the registered ReportingSet for ``name`` or raise KeyError.

    Callers that need a validating lookup against config use
    config.report.validate_active_reporting_set (run-entry, lazy-imports this
    module). This bare accessor is for the render_report facades + dispatcher,
    which receive an already-validated name.
    """
    return REPORTING_SETS[name]
