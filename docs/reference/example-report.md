# The interactive analysis report

A completed analysis renders a self-contained report as `analysis_report.html` or `analysis_report.zip`. The report renders from the outputs of the completed workflow. Rendering is idempotent, and a repeat render executes no simulation rule. The `hhemt eda` command writes a separate report and notebook. See [Exploratory analysis](../tutorials/exploratory-analysis.md).

## Producing a report

`render_report()` renders a single analysis, and the same method of the sensitivity facade renders a sensitivity primary.

The container follows the `report.interactive.report_html_mode` setting of the report config the analysis used. The config is the file passed as `report_config=` to `analysis.run()` when the call gave one, and the `report` block of the analysis config otherwise. The setting defaults to `auto`.

- `html` writes a single self-contained `analysis_report.html` with every figure inlined. The file is larger than the zip and opens without an unzip step.
- `zip` writes `analysis_report.zip`, containing `report.html` plus per-figure `data/*.html` entries.
- `auto` adds up the bytes of the figure files under `plots/` when the render starts. The render writes the HTML file when the total is below 15 MB, or 15,000,000 bytes, and writes the zip at that size or above. A workflow that renders under `auto` also writes `analysis_report.format`, a file that names the container of the render.

The `format` argument of `render_report()` overrides the setting for one render. The `format` argument of `Bundle.regenerate_report` and the `--format` option of `hhemt report-from-bundle` work the same way. Without an override, a single-analysis render reads the setting. A combined bundle carries no analysis config and renders under `auto`.

## Sidebar sections

The active reporting set, selected with `report.reporting_set`, decides the sections. [Reporting sets](reporting-sets.md) lists the sets. The default set renders the sections below in sidebar order.

| Section | Contents |
|---|---|
| Workflow Status | Per-scenario completion state across every phase |
| Errors and Warnings | The post-completion validation report with system-level checks, aggregate per-scenario checks, resource-utilization mismatches, and a table of granular per-scenario failures when any exist |
| System Information | DEM, Manning's roughness, boundary conditions, SWMM network elements, and disk utilization |
| Per Simulation Results | A figure group per simulated event |
| Workflow Performance | The timeline of workflow rules and the SLURM efficiency report |
| Metadata | RO-Crate provenance summary, data availability, and the reproduction guide |
| Appendix | The `scenario_status.csv` table that the default set emits and the declared sidebar order omits, so the table sorts last |

The other reporting sets render different selections. Only `combined` has a distinct sidebar order, because the categories depend on the experiments that `hhemt combine` receives. See [Cross-experiment sets](reporting-sets.md#cross-experiment-sets).

## Example reports

Finished reports are committed with the documentation and open from this site. Each report needs no sibling files. The figure and table pages load the Plotly and Tabulator libraries from a CDN. Figures and tables stay blank without a network connection.

- [Compute-configuration sweep report](bundles/norfolk-compute-config-sweep-report.html) belongs to the tutorial [Compute configurations](../tutorials/norfolk-compute-config-sweep.md) and shows run time, compute cost, speedup, and efficiency per configuration.
- [Observed events report](bundles/norfolk-observed-events-report.html) belongs to the tutorial [Observed events](../tutorials/norfolk-observed-events.md) and has one page per event.

## Viewing a report you did not generate

A rendered report is portable and needs no server, because the report is a static HTML file or a zip of static files. Open the HTML file in a browser, or unzip the `.zip` form and open `report.html` from the extracted directory.

## Embedding a report in your own documentation

MkDocs copies everything under `docs/` into the built site verbatim, so a report placed under `docs/` embeds directly.

```html
<iframe
  src="bundles/my-report.html"
  width="100%"
  height="600"
  style="border: 1px solid #ccc;"
  title="Interactive analysis report">
</iframe>
```

A report embeds every figure, and each example report on this page is roughly 100 MB. Weigh the repository cost before committing a report.
