# Run the exploratory analysis

Run the exploratory analysis on a completed analysis to produce the seeded EDA
notebook, the standalone figures under `plots/eda/`, and the verdicts that feed
the report's Errors and Warnings section. For what the step is for and why it
sits outside the workflow graph, see
[Exploratory analysis](../explanation/exploratory-analysis.md).

## Before you start

- **A completed, consolidated analysis directory.** The step reads the
  consolidated outputs and the per-scenario summaries; it does not run
  simulations and it does not consolidate.
- **`cfg_analysis.yaml` and `cfg_system.yaml` inside that directory**, if you
  intend to drive it by directory rather than by config paths. Both are written
  by a prior `analysis.run()`, and this step rewrites them itself on every run.
- **No allocation.** The loop runs in process on whatever machine you are on.

## Run it

By directory, which re-derives both configs from the directory itself:

```bash
hhemt eda --analysis-dir runs/synth_cc/master/
```

By explicit config paths, which is the form to use when the directory does not
yet carry its persisted configs:

```bash
hhemt eda \
    --system-config system.yaml \
    --analysis-config analysis.yaml
```

From Python, which is the form to use inside a notebook or a driver script:

```python
from hhemt import Toolkit

tk = Toolkit.from_configs("system.yaml", "analysis.yaml")
result = tk.analysis.eda()

result.notebook_path   # the seeded notebook this run wrote
result.report_path     # the HTML export, or None when the export was skipped
result.plot_paths      # the standalone figures written under plots/eda/
result.verdicts        # one CheckResult per calculation that ran
```

To try a different figure selection without editing the analysis config, pass a
YAML file carrying only the `eda` block:

```bash
hhemt eda --analysis-dir runs/synth_cc/master/ \
    --override-eda-config my_eda_selection.yaml
```

## What it writes

Everything lands under the analysis directory.

| Path | What it is |
|---|---|
| `eda/{stem}.zarr` | The backing dataset one calculation produced. |
| `eda/{stem}.verdict.json` | That calculation's verdict, in `CheckResult` shape. |
| `eda/{stem}.manifest.json` | The provenance sidecar naming the files the calculation read. |
| `plots/eda/{kind}.html` | One standalone figure per rendered kind. |
| `eda.ipynb` | The seeded notebook. A second run writes `eda_1.ipynb`, a third `eda_2.ipynb`, and so on. |
| `eda_report/eda_report.html` | The executed-and-converted notebook. Best-effort. |
| `eda_local/` | Your own EDA package, outside the toolkit source tree. |
| `cfg_analysis.yaml`, `cfg_system.yaml` | The configs, persisted so the notebook can re-derive its variables. |

The verdicts are also merged into `validation_report.json`, so they appear in
the report's Errors and Warnings section without a separate step.

## Choose which figures render

Every calculation runs on every invocation. Rendering is what you select, with
`eda.enabled_plots` on the analysis config:

```yaml
eda:
  enabled_plots:
    - config_diff_maps
  plotly_js_mode: inline
```

The default is the single entry `config_diff_maps`. A kind renders when it is
listed here **and** its calculation produced a backing dataset on this run, so
listing a kind whose calculation did not apply costs you a warning rather than
a broken figure.

The DEM-resolution family is the other shipped selection, and it is mutually
exclusive with `config_diff_maps`: pair it with
`report.reporting_set: dem-resolution` and a `system.target_dem_resolution`
sweep.

```yaml
eda:
  enabled_plots:
    - dem_resolution_cost_error
    - dem_resolution_error_ecdf
    - dem_resolution_diff_maps
    - dem_resolution_coupling_table
```

These are the registered kinds. The table is derived from the renderer registry
at documentation build time, so it cannot fall behind the code:

<!-- hhemt:eda-kind-table -->

A kind marked `Not yet designed` renders a titled figure carrying no panels.
Its calculation and its verdict are real; only the drawing is outstanding.

The same list is available at runtime, which is the form to reach for when you
are working against an installed version rather than reading this page:

```python
from hhemt.eda._plotting import _EDA_RENDERERS

sorted(_EDA_RENDERERS)
```

Naming an unregistered kind raises `ValueError` at render time, listing the
registered names.

## Read a verdict

Each verdict is a JSON serialization of `analysis_validation.CheckResult`:

```python
import json
from pathlib import Path

verdict = json.loads(
    Path("runs/synth_cc/master/eda/b4b_clean_identity.verdict.json").read_text()
)
verdict["passed"]           # bool
verdict["applicable"]       # False when the calculation did not apply
verdict["summary"]          # a sentence naming what was compared and the result
verdict["instrument"]       # "raw_rasters" or "summary_tier"
verdict["detection_floor"]  # the smallest difference that instrument can resolve
verdict["details"]          # per-row detail, where the check populates it
```

Read `applicable` before `passed`. A calculation that examined nothing returns
`passed` true with `applicable` false, and treating that as a green result is
the one way to misread these files.

`details` is populated by the cross-simulation and compute-sensitivity checks.
The raw byte-for-byte check leaves it empty on both outcomes, so a gate written
against `b4b_clean_identity`'s `details` reads empty on a failing run. Gate on
`passed` and report `summary`.

That check compares each compute configuration against its own within-family
reference inside a single analysis. It is produced under the `b4b` reporting
set; see [Reporting sets](../reference/reporting-sets.md) for when to select
it, and for where the clean-versus-resume comparison lives instead.

## Keep exploring in the notebook

Open the newest `eda_*.ipynb` and run the first cell. It binds `ctx`, which
carries the re-derived experiment: `datatree`, `sensitivity_datatree`,
`cfg_analysis`, `cfg_system`, `scenario_status`, `swmm_features`, `triton_dem`,
`performance` and `is_bundle`. A field is `None` when this root does not carry
that artifact.

Everything below the seeded cells is yours. A later run writes a new file
rather than touching the one you are working in, so open the newest file when
you want refreshed seed cells and keep your own work where it is.

Author reusable functions in `eda_local/` rather than in the toolkit source
tree. It imports the installed toolkit with a plain `import hhemt`, so a
function written there ports cleanly if it is later promoted into the report.

## When something is missing

**The HTML export was skipped.** The command says so and names the notebook.
The export executes the notebook against a `python3` kernel, and any failure
there degrades to a warning by design. Open the notebook, which is the source
of truth, and run it yourself to see the underlying error.

**A figure is enabled but absent, with a warning naming its backing artifact.**
Its calculation did not apply to this analysis, so it wrote no dataset. Check
the matching `eda/{stem}.verdict.json`: `applicable` false with a summary
naming the reason is the expected case.

**A report figure renders as a degraded panel.** The reporting set enumerated
an exploratory figure but the calculation never ran for it. Run this step on
the analysis before rendering the report.

## See also

- [Exploratory analysis](../explanation/exploratory-analysis.md): what the step
  is for, and why it sits outside the workflow graph.
- [Reporting sets](../reference/reporting-sets.md): which sets carry the
  exploratory figures into `analysis_report.html`.
- [Running a synthetic compute-sensitivity experiment](synthetic-compute-sensitivity-experiment.md):
  a worked sweep that these figures describe.
- [Combining experiments](combining-experiments.md): the cross-experiment
  comparison, including clean versus resume.
