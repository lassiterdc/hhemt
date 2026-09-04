# Exploratory analysis

**Exploratory analysis** is the step that asks a question a finished report
cannot answer on its own: is this result a property of the model, or a property
of how it happened to be computed? It runs after an analysis has completed and
consolidated, compares that analysis against itself along whatever axis the
experiment varied, and leaves behind both a verdict you can gate on and a
notebook you can keep working in.

It is deliberately outside the workflow graph. `analysis.eda()` is an in-process
facade with no Snakemake rule behind it, so running it costs no allocation, and
re-running it re-derives everything from artifacts that are already on disk.

## The question it answers

A sweep varies one thing on purpose. A compute-configuration sweep varies MPI
rank counts, run modes and partitions; a DEM-resolution sweep varies cell size.
In both cases the interesting result is not only how long each member took but
whether the members agree, and by how much they disagree when they do not.

That comparison is not something the per-simulation figures can make, because
each of them describes one member. Exploratory analysis is the layer that reads
across members and reduces the comparison to something a reader can act on: a
pass or fail verdict, a magnitude, and a figure showing where the disagreement
is.

## Two surfaces from one calculation

Every run produces two things, and it is worth being clear about which is
authoritative.

The **notebook** is the source of truth. Each run writes a fresh seeded
`.ipynb` and never modifies an existing one, so anything you author in a
previous notebook survives. Its cells call the same installed functions the
toolkit itself calls, and they re-derive their variables on execution rather
than replaying values pickled at emit time. That is what lets the notebook
travel inside a render bundle and still work.

The **HTML export** is a convenience. It is produced by executing the notebook
and converting the result, and it is best-effort: a missing kernel, a cell
error or a timeout degrades it to a warning rather than failing the run. If the
export is absent, the notebook is not.

Alongside both, the figures render to standalone files under `plots/eda/`, and
those are what a reporting set picks up when the figure is promoted into the
main analysis report.

## Calculation and figure are separate

Each member of the exploratory analysis has two halves that fail independently.

The **calculation** reads the analysis, writes a backing dataset under `eda/`,
and returns a verdict. The verdict is an `analysis_validation.CheckResult`, the
same record every other toolkit check returns, so it merges into
`validation_report.json` and surfaces in the report's Errors and Warnings
section without any separate plumbing.

The **figure** reads that backing dataset and nothing else. If the calculation
did not apply to your experiment shape, it writes no dataset, records a
not-applicable verdict, and the figure is skipped rather than drawn from
nothing.

Keeping the two apart is what makes a partial result honest. A member that
could not run says so; it does not render an empty figure that looks like a
negative finding.

## Some figures are not yet drawn

Three renderer kinds are registered and reachable but their figure design has
not been done: the within-family rank comparison, the clean-versus-resume
comparison, and the cross-hardware magnitude panel. Their calculations run and
their verdicts are real. Their figures currently render a titled placeholder
carrying the text `(figure pending /eda-spinup design)` and no panels.

This is visible rather than hidden, and that is the point: the artifact
discloses its own state, so a reader who opens one is not misled about what
they are looking at. The [how-to guide](../how-to/running-eda.md) marks each
kind so you can tell before you enable one.

## How these figures reach a report

Exploratory analysis writes its figures to disk unconditionally. Whether they
appear in `analysis_report.html` is a separate decision, made by the
[reporting set](../reference/reporting-sets.md) the analysis selects. Three
sets carry the exploratory adapter: `compute-sensitivity`, `dem-resolution` and
`b4b`, and each expects an analysis shaped for its own question.

Selecting one of them on an analysis that was not shaped that way produces a
report with degraded panels rather than a useful one, which is why the set name
and the enabled figures are chosen together.

## What it does not do

It does not run simulations, and it does not re-consolidate. It reads a
completed analysis and derives from it.

It does not decide whether a difference matters. A verdict reports whether the
comparison found a difference and how large it was, at a stated detection
floor; whether that is acceptable for your study is a judgment the toolkit does
not make for you.

It does not compare across experiments. A single run reads one analysis
directory. Comparing two finished experiments against each other is
[`hhemt combine`](../how-to/combining-experiments.md), which builds one report
over several of them.

## See also

- [Run the exploratory analysis](../how-to/running-eda.md): the commands, the
  outputs, and how to choose which figures render.
- [Reporting sets](../reference/reporting-sets.md): which sets carry the
  exploratory figures, and what each set is for.
- [The interactive analysis report](../reference/example-report.md): what the
  main report contains and how to open one.
- [Combining experiments](../how-to/combining-experiments.md): the
  cross-experiment comparison that a single run cannot make.
