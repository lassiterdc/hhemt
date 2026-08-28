# Reporting sets

A **reporting set** is a named selection of report renderers plus a sidebar
category order. Choosing one decides which figures and tables a rendered report
contains, and in what order its sidebar lists them. You select a set with a
single config field; you do not edit code to change what a report contains.

## Selecting one

The field is `reporting_set`, on the `report` block of the analysis config:

```yaml
report:
  reporting_set: benchmarking
```

Its default value is the literal string `default`, which is a sentinel rather
than a set name in the ordinary sense: at `analysis.run()` entry it resolves to
`benchmarking` when `toggle_sensitivity_analysis` is true, and to the standard
`default` set otherwise. Setting any other registered name selects that set
directly.

The name is validated at `analysis.run()` entry against the registry. An
unregistered name raises `ConfigurationError` naming both what you asked for and
the registered names, so a typo fails before any compute is committed.

## The registered sets

Six sets ship. `src/hhemt/report_renderers/_reporting_sets.py::REPORTING_SETS`
is the single source of truth for what each one contains; this page describes
what each is **for**, which stays true as individual renderers move between
them.

| Set | Use it for |
|---|---|
| `default` | An ordinary analysis. The general-purpose report: workflow status, system inputs, per-simulation results, validation findings, provenance. |
| `benchmarking` | A sensitivity analysis whose axis is compute configuration. Adds the run-time, compute-cost and scaling figures. This is what `default` resolves to on a sensitivity analysis. |
| `compute-sensitivity` | A compute-configuration sweep where the question is whether results are invariant to the compute config, not just how fast each one is. Analysis-specific: it expects the compute-sensitivity EDA artifacts. |
| `dem-resolution` | A sweep whose axis is DEM cell size. Analysis-specific: its figures compare peak flood depth across resolution rungs, so it expects a varying grid and rejects a mixed-resolution master. |
| `b4b` | A clean-versus-resume study. The benchmarking selection plus the bit-for-bit identity check on whether a resumed simulation reproduces its clean run exactly. |
| `combined` | The cross-experiment report emitted by `hhemt combine`. You do not select this one on an analysis: `combine` uses it to build one report over several finished experiments. |

Two properties are worth knowing before you pick one.

**The analysis-specific sets expect artifacts an ordinary run does not produce.**
`compute-sensitivity`, `dem-resolution` and `b4b` each assume the analysis was
shaped for their question. Selecting one on an analysis that was not produces a
report with missing figures rather than a useful one.

**`combined` is the only set with its own sidebar order.** Every other set uses
the standard order. `combined` carries fixed cross-experiment categories, and
`hhemt combine` appends one category per input experiment at combine time, so
its final sidebar is known only once the inputs are known.

## Dropping a renderer from a set

`disabled_renderers` removes named renderers from whichever set is active,
without switching sets:

```yaml
report:
  reporting_set: benchmarking
  disabled_renderers:
    - per_sim
```

An unknown name here is a `ConfigurationError` at run entry, not a silent no-op,
and the same filter applies both where figures are emitted and where they are
enumerated, so the two can never disagree.

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
