# The interactive analysis report

Every completed analysis renders a self-contained `analysis_report.html`. This
page documents what that report contains and how to open one; it is a reference
for the artifact, not a live sample.

## Producing a report

The report is rendered from completed-workflow outputs. It is idempotent:
re-rendering does not re-execute any simulation rule.

```python
analysis.render_report()          # multisim
sensitivity.render_report()       # sensitivity master
```

The container is chosen by the `format` argument, which defaults to `"zip"`.
`render_report()` with no argument writes `{analysis_dir}/analysis_report.zip`,
containing `report.html` plus per-figure `data/*.html` entries. Pass
`format="html"` for a single self-contained `analysis_report.html` with every
figure inlined. There is no size-based selection between them.

## Sidebar sections

Which sections appear is driven by the active reporting set, selected with
`report.reporting_set` (see [Reporting sets](reporting-sets.md)). The default
set renders, in sidebar order:

| Section | Contents |
|---|---|
| Workflow Status | Per-scenario completion state across every phase |
| Errors and Warnings | The post-completion validation report: system-level checks, aggregate per-scenario checks, granular per-scenario failures, and resource-utilisation mismatches |
| System Information | DEM, boundary conditions, SWMM network elements, and disk utilisation |
| Per Simulation Results | One figure group per simulated event |
| Workflow performance | Per-rule scheduler and runtime accounting |
| Metadata | RO-Crate provenance, the reproduction guide, and SLURM efficiency |
| Appendix | The `scenario_status.csv` table (emitted by the default set, but not yet declared in its sidebar order) |

The `b4b`, `benchmarking`, `combined`, `compute-sensitivity` and
`dem-resolution` sets render different selections; each declares its own
sidebar order.

## Viewing a report you did not generate

A rendered report is portable; it carries its own JavaScript and needs no
server. Open the HTML file directly in a browser, or unzip the `.zip` form and
open `report.html` from the extracted directory.

## Embedding a report in your own documentation

MkDocs copies everything under `docs/` into the built site verbatim, so a
report dropped under `docs/` is embeddable directly:

```html
<iframe
  src="bundles/my-report.html"
  width="100%"
  height="600"
  style="border: 1px solid #ccc;"
  title="Interactive analysis report">
</iframe>
```

Reports are large: a full analysis report with inlined Plotly figures runs to
several megabytes, so weigh the repository cost before committing one.
