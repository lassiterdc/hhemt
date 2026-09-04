# Reporting sets

A **reporting set** is a named *bundle* of report renderers, plus a sidebar
category order. One set produces one report containing every renderer in the
bundle. Choosing a set decides which figures and tables a rendered report
contains, and in what order its sidebar lists them. You select a set with a
single config field, without editing code.

!!! tip "Most reports come from one set, and one set is usually enough"
    A report showing benchmarking results alongside compute-configuration
    comparisons has usually selected one set that bundles both kinds of
    renderer, rather than naming two. `b4b` is the clearest case: it carries
    ten renderers, including `sensitivity_benchmarking` and
    `eda_compute_sensitivity` together, so its report shows both. Reach for a
    list only when no single set carries everything you want. If you are
    trying to work out what produced a report you are looking at, read the
    bundle contents below rather than reasoning from the figures present.

## Selecting one or more

The field is `reporting_set`, on the `report` block of the analysis config. It
takes either one set name or a list of them. One name is the common case:

```yaml
report:
  reporting_set: benchmarking
```

Its default value is the one-element list `["default"]`, whose single member is
a sentinel rather than an ordinary set name. At `analysis.run()` entry that
sentinel resolves to `benchmarking` when `toggle_sensitivity_analysis` is true,
and to the standard `default` set otherwise. Setting any other registered name
selects that set directly.

Writing the field but leaving it empty means the same as omitting it. A bare
`reporting_set:` with nothing after it, its explicit form `null`, an empty
string, and an empty list are all read as absent values rather than as requests
for a report with no figures, so each of them falls back to the sentinel above.

The sentinel then resolves from the analysis rather than from the blank. On a
sensitivity analysis an empty field therefore produces the `benchmarking`
report, carrying its run-time, compute-cost and scaling figures.

The value must be a name or a list of names. An unordered container such as a
YAML set (`!!set`) is refused rather than quietly accepted, because the order
you write the names in decides the order their renderers merge, and an
unordered container carries no order to read.

The name is validated at `analysis.run()` entry against the registry. An
unregistered name raises `ConfigurationError` naming both what you asked for and
the registered names, so a typo fails before any compute is committed.

## Composing several sets

Naming more than one set composes them into a single report. A renderer carried
by more than one of them appears once, carrying the figures both contribute, so
listing `dem-resolution` alongside `compute-sensitivity` gives you both EDA
families rather than whichever one happens to be first:

```yaml
report:
  reporting_set:
    - compute-sensitivity
    - dem-resolution
```

Four rules apply, and all four are checked at `analysis.run()` entry, before any
compute is committed:

- **Every named set must describe the same kind of analysis.** You cannot mix an
  event-ensemble set with a sensitivity set, and `combined` is never selectable
  on an analysis at all.
- **Your sweep must vary an axis for each named set that requires one.** The
  requirements add up rather than replacing one another, so a list of
  `compute-sensitivity` and `dem-resolution` needs a sweep that varies both a
  compute-configuration field and `target_dem_resolution`.
- **`default` cannot be listed alongside another set.** It is a sentinel that
  resolves to whichever set suits the analysis, so it has no meaning as one
  member of a composition. Name the set you want instead.
- **Two sets that gate the same renderer on different conditions cannot be
  combined.** A composed report emits each renderer once, under one condition,
  so two sets that disagree about that condition have no single answer. `b4b`
  gates `eda_compute_sensitivity` on preserved raw outputs, while
  `compute-sensitivity` and `dem-resolution` gate it on an EDA artifact, so
  `b4b` composes with `benchmarking` and `sensitivity` but not with those two.

All four rules fail with a message naming both the set and the reason, so a
combination that cannot work is refused on the login node rather than after the
compute is spent.

## The registered sets

Seven sets ship, in three kinds. The kind tells you whether a set is even
selectable for the analysis you are running.

`src/hhemt/report_renderers/_reporting_sets.py::REPORTING_SETS` is the single
source of truth for what each set contains. This page describes what each is
**for**.

### Event-ensemble sets

For an analysis whose members are weather events rather than compute
configurations.

| Set | Use it for |
|---|---|
| `default` | An ordinary analysis. The general-purpose report: workflow status, system inputs, per-simulation results, validation findings, provenance. It is the only set carrying `per_sim`, the per-event figure pair. |

### Sensitivity-analysis sets

For a sweep, where members vary along a declared axis. All five replace
`per_sim` with `per_sim_per_member`; all but `sensitivity` also add
`sensitivity_benchmarking`.

| Set | Use it for |
|---|---|
| `benchmarking` | A sweep whose axis is compute configuration. Adds run-time, compute-cost and scaling figures. This is what `default` resolves to on a sensitivity analysis. |
| `sensitivity` | A sweep that wants per-member results without any specialized figure family: no run-time and scaling figures, no EDA comparison. It is the sweep-shaped bundle minus `sensitivity_benchmarking`, so it imposes no axis requirement and suits a sweep varying any axis. |
| `compute-sensitivity` | A compute-configuration sweep where the question is whether results are invariant to the compute config, rather than only how fast each one runs. Expects the compute-sensitivity EDA artifacts. |
| `dem-resolution` | A sweep whose axis is DEM cell size. Its figures compare peak flood depth across resolution rungs, so it expects a varying grid and rejects a mixed-resolution experiment. |
| `b4b` | A raw-output byte-for-byte identity study across compute configurations. The benchmarking bundle plus a per-timestep check on whether the raw TRITON rasters are bit-identical across compute configs within one device class (CPU or GPU). It is not a resume study: the clean-versus-resume comparison lives on the combined report. Requires preserved raw outputs. |

### Cross-experiment sets

| Set | Use it for |
|---|---|
| `combined` | The cross-experiment report emitted by `hhemt combine`. You do not select this one on an analysis. `combine` uses it to build one report over several finished experiments. |

Two properties matter before you pick one.

**The analysis-specific sets expect artifacts an ordinary run does not produce.**
`compute-sensitivity`, `dem-resolution` and `b4b` each assume the analysis was
shaped for their question. Selecting one on an analysis that was not shaped that
way produces a report with missing figures rather than a useful one.
Those artifacts are produced by the exploratory analysis step; see
[Run the exploratory analysis](../how-to/running-eda.md) for how to produce them.

**`combined` is the only set with its own sidebar order.** Every other set uses
the standard order. `combined` carries fixed cross-experiment categories, and
`hhemt combine` appends one category per input experiment at combine time, so
its final sidebar is known only once the inputs are known.

## What each set bundles

Every name below is droppable via `disabled_renderers`. These are the complete
bundles, not examples.

| Set | Renderers | Count |
|---|---|---|
| `default` | `disk_utilization`, `errors_and_warnings`, `metadata`, `per_analysis_summary`, `per_sim`, `scenario_status_appendix`, `system_overview`, `workflow_performance` | 8 |
| `benchmarking` | `disk_utilization`, `errors_and_warnings`, `metadata`, `per_analysis_summary`, `per_sim_per_member`, `scenario_status_appendix`, `sensitivity_benchmarking`, `system_overview`, `workflow_performance` | 9 |
| `sensitivity` | `disk_utilization`, `errors_and_warnings`, `metadata`, `per_analysis_summary`, `per_sim_per_member`, `scenario_status_appendix`, `system_overview`, `workflow_performance` | 8 |
| `compute-sensitivity` | `disk_utilization`, `eda_compute_sensitivity`, `errors_and_warnings`, `metadata`, `per_analysis_summary`, `per_sim_per_member`, `scenario_status_appendix`, `sensitivity_benchmarking`, `system_overview`, `workflow_performance` | 10 |
| `dem-resolution` | `disk_utilization`, `eda_compute_sensitivity`, `errors_and_warnings`, `metadata`, `per_analysis_summary`, `per_sim_per_member`, `scenario_status_appendix`, `sensitivity_benchmarking`, `system_overview`, `workflow_performance` | 10 |
| `b4b` | `disk_utilization`, `eda_compute_sensitivity`, `errors_and_warnings`, `metadata`, `per_analysis_summary`, `per_sim_per_member`, `scenario_status_appendix`, `sensitivity_benchmarking`, `system_overview`, `workflow_performance` | 10 |
| `combined` | `cross_experiment_compatibility`, `cross_experiment_disk_utilization`, `cross_experiment_errors_and_warnings`, `cross_experiment_intercomparison`, `cross_experiment_intercomparison_maps` | 5 |

The three sensitivity EDA sets bundle the same ten renderers. They differ in the
figures `eda_compute_sensitivity` emits, which follow from the analysis each set
expects.

## Dropping a renderer from a set

`disabled_renderers` removes named renderers from whichever set is active,
without switching sets. Any name from that set's bundle above is a valid entry:

```yaml
report:
  reporting_set: benchmarking
  disabled_renderers:
    - per_sim_per_member
```

An unknown name here raises `ConfigurationError` at run entry rather than
silently doing nothing, and the same filter applies both where figures are
emitted and where they are enumerated, so the two can never disagree.

??? note "Why the sensitivity sets carry an extra validation step"
    `benchmarking`, `compute-sensitivity`, `dem-resolution` and `b4b` all
    describe a sensitivity analysis, so each additionally validates that the
    sensitivity configuration declares the independent variables its figures
    plot against. That check runs at `analysis.run()` entry alongside the
    set-name check, which is why a mis-specified sweep fails on the login node
    rather than after the compute is spent.

## See also

- [The interactive analysis report](example-report.md): what a rendered report contains and how to open one.
- [Configuration schema](config-schema.md): the rest of the `report` block.
- [Running a synthetic compute-sensitivity experiment](../how-to/synthetic-compute-sensitivity-experiment.md): a worked run that selects `compute-sensitivity` and `dem-resolution`.
- [Exploratory analysis](../explanation/exploratory-analysis.md): what the exploratory figures are, and why the step that produces them sits outside the workflow graph.
