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
class RendererSelection:
    """One renderer in a set's ordered selection.

    builder_key   : key into workflow.py's _RENDERER_BUILDERS map (P1b).
    predicate_key : None for unconditional renderers; a string the dispatcher
                    resolves to a Callable[[builder], bool] for conditional
                    renderers (e.g. "has_independent_vars", "has_sa_event_pairs").
    """

    builder_key: str
    predicate_key: str | None = None


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

# The standard multisim set: the six common renderers, in emission order
# (matches workflow.py:1913-1918 today).
_STANDARD_SELECTION: tuple[RendererSelection, ...] = (
    RendererSelection("system_overview"),
    RendererSelection("per_sim"),
    RendererSelection("per_analysis_summary"),
    RendererSelection("scenario_status_appendix"),
    RendererSelection("errors_and_warnings"),
    RendererSelection("disk_utilization"),
)

# The benchmarking (sensitivity-master) set: the five common renderers shared by
# the master/reprocess generators (workflow.py:6391-6415), plus the two
# conditional sensitivity renderers gated by predicate.
_BENCHMARKING_SELECTION: tuple[RendererSelection, ...] = (
    RendererSelection("system_overview"),
    RendererSelection("per_analysis_summary"),
    RendererSelection("scenario_status_appendix"),
    RendererSelection("errors_and_warnings"),
    RendererSelection("disk_utilization"),
    RendererSelection("per_sim_per_sa", predicate_key="has_sa_event_pairs"),
    RendererSelection("sensitivity_benchmarking", predicate_key="has_independent_vars"),
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
