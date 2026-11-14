# Reporting sets

A **reporting set** is a named bundle of report renderers plus a sidebar category order. One set produces one report with every renderer in the bundle. The choice of set fixes the figures and tables of a report and the sidebar order. A single config field selects a set without code edits.

!!! tip "One set usually suffices"
    A report with benchmarking results next to compute-configuration comparisons usually selects one set that bundles both renderers. `compute_config_byte_identity` is the clearest case. The set carries `sensitivity_benchmarking` and `eda_compute_sensitivity` together, so the report shows both. A list is necessary only when no single set carries every renderer you want. To identify the set behind a report, read the bundle contents and skip any reasoning from the figures present.

## Selecting one or more

The field is `reporting_set` on the `report` block of the analysis config. The field takes a set name or a list of names. A single name is the common case.

```yaml
report:
  reporting_set: benchmarking
```

The default value is the one-element list `["default"]`. The single member is a sentinel and no ordinary set name. At `analysis.run()` entry the sentinel resolves to `sensitivity` when `toggle_sensitivity_analysis` is true and to the standard `default` set otherwise. Any other registered name selects that set directly.

An empty value means the same as an omitted field. A bare `reporting_set:` with no value, the explicit form `null`, an empty string, and an empty list all read as absent values and fall back to the sentinel. A request for a report with no figures is impossible.

The sentinel resolves from the analysis and not from the blank. On a sensitivity analysis an empty field therefore produces the `sensitivity` report. The report carries per-member results and omits the run-time, compute-cost, and scaling figures.

The value must be a name or a list of names. An unordered container such as a YAML set with `!!set` is refused. The order of the names decides the merge order of the renderers, and an unordered container has no order to read.

`analysis.run()` validates every name against the registry at entry. An unregistered name raises `ConfigurationError` that names the requested value and the registered names. A typo fails before any compute is committed.

## Composing several sets

A list of several names composes the sets into one report. A renderer in more than one set appears once with the figures from each, and the renderers merge in the order of the list.

```yaml
report:
  reporting_set:
    - sensitivity
    - benchmarking
```

??? note "Compositions that add figures, and the pair for separate sweeps"
    The list composes to the same bundle as `benchmarking` alone. Every pair that the rules admit behaves that way with one exception. In each of the other pairs a single set carries everything that the other set carries, so naming the larger set is enough. The exception is `compute_sensitivity` with `dem_resolution`. The pair passes every rule and still fails to render the added figures. `config_diff_maps` subtracts members cell by cell and refuses members on different grids with a `ProcessingError`. The `dem_resolution` figures need a ladder of grids. No single sweep serves both, and separate sweeps solve the conflict, as [Running a synthetic compute-sensitivity experiment](../how-to/synthetic-compute-sensitivity-experiment.md) shows.

`analysis.run()` checks each rule at entry, before any compute is committed.

- **Every named set must describe the same type of analysis.** An event-ensemble set and a sensitivity set never mix, and `combined` is never selectable on an analysis.
- **The sweep must vary an axis for each named set that requires an axis.** The requirements add up. A list of `compute_sensitivity` and `dem_resolution` needs a sweep that varies a compute-configuration field and `target_dem_resolution`.
- **`default` is unavailable in a list.** The name `default` is a sentinel that resolves to the set that suits the analysis, so the name has no meaning as one member of a composition. Name the set you want.
- **Sets that gate the same renderer on different conditions are unavailable together.** A composed report emits each renderer once under one condition, and sets that disagree about the condition have no single answer. `compute_config_byte_identity` gates `eda_compute_sensitivity` on preserved raw outputs. `compute_sensitivity` and `dem_resolution` gate the renderer on an EDA artifact. Therefore `compute_config_byte_identity` composes with `benchmarking` and `sensitivity` and refuses `compute_sensitivity` and `dem_resolution`.

Each rule fails with a message that names the set and the reason. A combination that fails is refused on the login node and wastes no compute.

## The registered sets

`src/hhemt/report_renderers/_reporting_sets.py::REPORTING_SETS` is the single source of truth for the contents of each set. The tables give the purpose of each set.

### Event-ensemble sets

| Set | Use |
|---|---|
| `default` | An ordinary analysis. The general-purpose report shows workflow status, system inputs, per-simulation results, validation findings, and provenance. The set is the only one carrying `per_sim`, the per-event figure pair. |

### Sensitivity-analysis sets

Every set here replaces `per_sim` with `per_sim_per_member`. Every set except `sensitivity` also adds `sensitivity_benchmarking`.

| Set | Use |
|---|---|
| `benchmarking` | A [benchmarking](glossary.md#benchmarking) sweep. The set adds run-time, compute-cost, and scaling figures. |
| `sensitivity` | A sweep that needs per-member results and no specialized figure family. The set omits the run-time and scaling figures and the EDA comparison. The set is the sweep-shaped bundle minus `sensitivity_benchmarking`, so the set imposes no axis requirement and suits a sweep over any axis. The name `default` resolves to the set on a sensitivity analysis. |
| `compute_sensitivity` | A compute-configuration sweep that tests whether results are invariant to the compute config and goes beyond the speed of each configuration. The set expects the compute-sensitivity EDA artifacts. |
| `dem_resolution` | A sweep over the DEM cell size. The figures compare peak flood depth across resolution rungs, so the set expects a varying grid and rejects a mixed-resolution experiment. |
| `compute_config_byte_identity` | A byte-for-byte identity check of raw outputs across compute configurations. The set bundles the benchmarking renderers plus a per-timestep check of whether the raw TRITON rasters are bit-identical across compute configs within the CPU device class or within the GPU device class. The combined report holds the clean-versus-resume comparison, and [Combining experiments](../how-to/combining-experiments.md) describes the comparison. The set requires preserved raw outputs. |

`compute_sensitivity`, `dem_resolution`, and `compute_config_byte_identity` expect artifacts that an ordinary sweep leaves out. Each set assumes that the analysis was shaped for the question of the set. A set selected on an unsuited analysis produces a report with missing figures. The exploratory analysis step produces the artifacts, and [Running the exploratory analysis](../how-to/running-eda.md) gives the procedure.

### Cross-experiment sets

| Set | Use |
|---|---|
| `combined` | The cross-experiment report that `hhemt combine` emits. An analysis never selects `combined`. The `combine` command uses the set to build one report over several finished experiments. |

`combined` is the only set with a nonstandard sidebar order. Every other set uses the standard order. `combined` carries fixed cross-experiment categories. `hhemt combine` appends a category per input experiment at combine time. The final sidebar is known once the inputs are known.

## Set bundles

`disabled_renderers` can drop any name that follows.

| Set | Renderers |
|---|---|
| `default` | `disk_utilization`, `errors_and_warnings`, `metadata`, `per_analysis_summary`, `per_sim`, `scenario_status_appendix`, `system_overview`, `workflow_performance` |
| `benchmarking` | `disk_utilization`, `errors_and_warnings`, `metadata`, `per_analysis_summary`, `per_sim_per_member`, `scenario_status_appendix`, `sensitivity_benchmarking`, `system_overview`, `workflow_performance` |
| `sensitivity` | `disk_utilization`, `errors_and_warnings`, `metadata`, `per_analysis_summary`, `per_sim_per_member`, `scenario_status_appendix`, `system_overview`, `workflow_performance` |
| `compute_sensitivity` | `disk_utilization`, `eda_compute_sensitivity`, `errors_and_warnings`, `metadata`, `per_analysis_summary`, `per_sim_per_member`, `scenario_status_appendix`, `sensitivity_benchmarking`, `system_overview`, `workflow_performance` |
| `dem_resolution` | `disk_utilization`, `eda_compute_sensitivity`, `errors_and_warnings`, `metadata`, `per_analysis_summary`, `per_sim_per_member`, `scenario_status_appendix`, `sensitivity_benchmarking`, `system_overview`, `workflow_performance` |
| `compute_config_byte_identity` | `disk_utilization`, `eda_compute_sensitivity`, `errors_and_warnings`, `metadata`, `per_analysis_summary`, `per_sim_per_member`, `scenario_status_appendix`, `sensitivity_benchmarking`, `system_overview`, `workflow_performance` |
| `combined` | `cross_experiment_compatibility`, `cross_experiment_disk_utilization`, `cross_experiment_errors_and_warnings`, `cross_experiment_intercomparison`, `cross_experiment_intercomparison_maps`, `cross_solver_comparison` |

`compute_sensitivity`, `dem_resolution`, and `compute_config_byte_identity` bundle the same renderers. The figures that `eda_compute_sensitivity` emits differ, and the figures follow from the analysis that each set expects.

## Dropping a renderer from a set

`disabled_renderers` subtracts named renderers from the bundle of the active set and keeps the set. The report contains every renderer of the bundle except the dropped ones, and any name from the bundle is a valid entry. A large event ensemble is the usual reason. The per-event figures, `per_sim` or `per_sim_per_member` on a sweep, add a page for every event.

```yaml
report:
  reporting_set: benchmarking
  disabled_renderers:
    - per_sim_per_member
```

An unknown name raises `ConfigurationError` at run entry and never passes silently. The same filter applies where figures are emitted and where figures are enumerated. Emission and enumeration always agree.

??? note "Extra validation step on some sets"
    `benchmarking`, `compute_sensitivity`, `dem_resolution`, and `compute_config_byte_identity` carry figures that plot against a sweep axis. Each set therefore validates that the sensitivity configuration declares the independent variables of the figures. The check runs at `analysis.run()` entry beside the set-name check, so a mis-specified sweep fails on the login node and wastes no compute.

## See also

- [The interactive analysis report](example-report.md) covers report production and the finished examples.
- [Configuration schema](config-schema.md) lists the rest of the `report` block.
- [Running a synthetic compute-sensitivity experiment](../how-to/synthetic-compute-sensitivity-experiment.md) shows a worked sweep with `compute_sensitivity` and a worked sweep with `dem_resolution`.
- [Exploratory analysis](../tutorials/exploratory-analysis.md) covers the exploratory figures and the placement of the step outside the workflow graph.
