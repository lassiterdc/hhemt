# Run the exploratory analysis

## Goal

Produce the seeded EDA notebook, the standalone figures under `plots/eda/`, and the verdicts that the report reads under Errors and Warnings, from a completed analysis.

## Prerequisites

- A consolidated analysis directory. The step reads the consolidated outputs and the per-scenario summaries, runs no simulation, and does no re-consolidation.
- `cfg_analysis.yaml` and `cfg_system.yaml` inside the directory, when you drive the step by directory and not by config paths. A prior `analysis.run()` writes both files, and the step rewrites both on every invocation.
- No allocation is needed. The loop runs in process on the machine you use.

## Key references

- [Exploratory analysis of an event ensemble](../tutorials/exploratory-analysis.md) runs the step on the observed-event result and draws difference plots from the example result.
- [Reporting sets](../reference/reporting-sets.md) describes the renderer bundles that put the exploratory figures into `analysis_report.html`.
- [Running a synthetic compute-sensitivity experiment](synthetic-compute-sensitivity-experiment.md) walks through the sweep whose figures this page describes.
- [Combining experiments](combining-experiments.md) covers the cross-experiment comparison, including clean versus resume.

## Run the step

Run by directory. The command re-derives the system and analysis configs from the directory itself.

```bash
hhemt eda --analysis-dir {analysis_dir}
```

Run by explicit config paths when the directory holds no persisted configs.

```bash
hhemt eda \
    --system-config system.yaml \
    --analysis-config analysis.yaml
```

In Python, call the method from a notebook or a driver script.

```python
from hhemt import Toolkit

tk = Toolkit.from_configs("system.yaml", "analysis.yaml")
result = tk.analysis.eda()

result.notebook_path   # the seeded notebook this invocation wrote
result.report_path     # the HTML export, or None when the export was skipped
result.plot_paths      # the standalone figures written under plots/eda/
result.verdicts        # one CheckResult per calculation that ran
```

To try a different figure selection without editing the analysis config, pass a YAML file that carries only the `eda` block.

```bash
hhemt eda --analysis-dir {analysis_dir} \
    --override-eda-config my_eda_selection.yaml
```

## Files written

Every path lands under the analysis directory.

| Path | Content |
|---|---|
| `eda/{stem}.zarr` | The backing dataset that one calculation produced |
| `eda/{stem}.verdict.json` | The verdict of the calculation, in `CheckResult` shape |
| `eda/{stem}.manifest.json` | The provenance sidecar that names the files the calculation read |
| `plots/eda/{kind}.html` | One standalone figure per rendered plot |
| `eda.ipynb` | The seeded notebook. A second invocation writes `eda_1.ipynb`, a third writes `eda_2.ipynb`, and so on |
| `eda_report/eda_report.html` | The executed and converted notebook, best effort |
| `eda_local/` | Your own EDA package, outside the toolkit source tree |
| `cfg_analysis.yaml`, `cfg_system.yaml` | The configs, persisted so that the notebook can re-derive the variables |

As [Calculation and figure](../tutorials/exploratory-analysis.md#calculation-and-figure) sets out, the report lists the verdicts under Errors and Warnings.

## Choose the figures to render

Every calculation runs on every invocation. Select the rendering with `eda.enabled_plots` on the analysis config.

```yaml
eda:
  enabled_plots:
    - config_diff_maps
  plotly_js_mode: inline
```

The default is the single entry `config_diff_maps`. A figure renders when the figure is listed here **and** the calculation behind the figure produced a backing dataset in this invocation, so listing a figure whose calculation did not apply costs a warning and never a broken figure.

The DEM-resolution family is the other shipped selection. The family needs a separate sweep, apart from `config_diff_maps`, and [Composing several sets](../reference/reporting-sets.md#composing-several-sets) gives the reason. Pair the family with `report.reporting_set: dem_resolution` and a `system.target_dem_resolution` sweep.

```yaml
eda:
  enabled_plots:
    - dem_resolution_cost_error
    - dem_resolution_error_ecdf
    - dem_resolution_diff_maps
    - dem_resolution_coupling_table
```

The table lists the registered plots. The table derives from the renderer registry at documentation build time.

<!-- hhemt:eda-kind-table -->

Naming an unregistered plot raises `ValueError` at render time and lists the registered names.

## Read a verdict

Each verdict serializes `analysis_validation.CheckResult` as JSON.

```python
import json
from pathlib import Path

verdict = json.loads(
    Path("{analysis_dir}/eda/raw_byte_identity.verdict.json").read_text()
)
verdict["passed"]           # bool
verdict["applicable"]       # False when the calculation did not apply
verdict["summary"]          # a sentence naming what was compared and the result
verdict["instrument"]       # "raw_rasters" or "summary_tier"
verdict["detection_floor"]  # the smallest difference that instrument can resolve
verdict["details"]          # per-row detail, where the check populates it
```

Read `applicable` before `passed`. A calculation that examined no data returns `passed` true with `applicable` false. Treating that result as green is the sole way to misread the files.

`details` is populated by the cross-simulation and compute-sensitivity checks. The raw byte-for-byte check leaves `details` empty on both outcomes, so a gate written against `details` of `raw_byte_identity` reads empty on a failing analysis. Gate on `passed` and report `summary`.

The raw byte-for-byte check compares each compute configuration against the within-family reference of that configuration, inside a single analysis. The `compute_config_byte_identity` reporting set produces the check. [Reporting sets](../reference/reporting-sets.md) gives the criteria for selecting the set. [Combining experiments](combining-experiments.md) produces the clean-versus-resume comparison.

## Keep exploring in the notebook

Open the newest `eda_*.ipynb` and run the first cell. The cell binds `ctx`, an object that carries the re-derived experiment as the fields `datatree`, `sensitivity_datatree`, `cfg_analysis`, `cfg_system`, `scenario_status`, `swmm_features`, `triton_dem`, `performance`, and `is_bundle`. A field is `None` when the root carries no such artifact.

Everything below the seeded cells is yours. A later invocation writes a new file and leaves the file you work in untouched, so open the newest file for refreshed seed cells and keep your own work in place.

Author reusable functions in `eda_local/` and keep the toolkit source tree unchanged. Code there imports the installed toolkit with a plain `import hhemt`, so a function written there ports cleanly if the function is later promoted into the report.

## Missing outputs

**The HTML export was skipped.** The command names the notebook. Open the notebook and run the cells to see the underlying error. [The seeded notebook](../tutorials/exploratory-analysis.md#the-seeded-notebook) gives the reasons that the export is best effort.

**A figure is enabled but absent, with a warning that names the backing artifact.** The calculation for the figure did not apply to the analysis, so the calculation wrote no dataset. Check the matching `eda/{stem}.verdict.json`. A false `applicable` with a summary that names the reason is the expected case.

**A report figure renders as a degraded panel.** The reporting set enumerated an exploratory figure, but the calculation for the figure never ran. Run this step on the analysis before rendering the report.
