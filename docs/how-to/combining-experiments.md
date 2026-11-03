# Combining experiments into one cross-experiment report

`hhemt combine` ingests **two or more** completed render bundles from different
experiments, checks that they are compatible, and emits a single standalone
**combined bundle** with one cross-experiment report, itself a valid RO-Crate
containing each input bundle intact.

Use this when you have run the same (or a compatible) analysis under different
conditions (e.g. two compute configurations, or two clusters) and want one
report that presents them side by side.

## Prerequisites

--8<-- "prerequisites-base.md"
- **Two or more emitted bundles on disk.** Produce each with `hhemt bundle`
  after a run that has rendered its report. `hhemt bundle --system-config
  my_system.yaml --analysis-config my_analysis.yaml` writes the archive; see the
  [CLI reference](../reference/cli.md) for its optional flags.
- Each input is a bundle **zip** (as `hhemt bundle` writes it) or an **unpacked
  bundle directory**; give them as they are, with no unzip step. A bundle zip is
  compressed entry by entry: zarr chunk files are stored (the toolkit already
  compresses them) and every other entry is deflated at zlib level 6. A zip made by an
  earlier toolkit stores every entry uncompressed, and is accepted too. The inputs are
  never modified.
- Combine operates on **single-analysis** bundles (each ships its consolidated
  `analysis_datatree.zarr` at the bundle root) **and on sensitivity-primary
  bundles** (each ships `experiment_datatree.zarr` at the bundle root; a bundle
  produced before the store unification ships `analysis_datatree.zarr` or
  `sensitivity_datatree.zarr` and is still accepted). The
  combine step resolves whichever consolidated tree a bundle ships. It records
  the divergences its compatibility check finds in the combined bundle's
  `combined_compatibility.json`, and the cross-experiment report presents, for a
  clean-vs-resume sensitivity-primary pair, a
  `cross_experiment_intercomparison` panel derived cross-bundle from the two
  consolidated trees (see *Compatibility gates* below).

## Combine the bundles

```bash
hhemt combine bundle_a.zip bundle_b.zip -o combined
```

- Positional arguments: two or more bundle zips or bundle directories.
- `-o` / `--output` (optional): the combined bundle is written as one zip,
  `{output}.zip` (a value that already ends in `.zip` is used as given). Defaults to
  `{first_bundle}/../combined_{n}bundles_{git_sha}.zip`.
- `--unzipped` (optional): write a folder at `--output` instead, holding each
  experiment's full bundle under `child_crates/{experiment_id}/`.
- `--workdir` (optional): the directory under which the combine makes its private working
  folder. Defaults to the system temporary directory. The folder holds each input's
  render set (everything except the raw `out_*/bin` solver output, which no combine step
  reads) and is removed when the combine ends, including on SIGTERM.

The command (1) refuses an input whose consolidated tree was carried without its data,
(2) extracts each input's render set into the working folder, (3) runs a
metadata-compatibility check, (4) renders one cross-experiment report, and (5) writes the
combined bundle at the output path, replacing any earlier output. It **aborts with an
error** if the bundles carry a *blocking* divergence (see below). The combined zip holds
the root files (crate, read-models, manifest, report, `plots/`) and, per experiment,
`child_crates/{experiment_id}.zip` (that experiment's own bundle zip) and
`child_crates/{experiment_id}.refreshed/` (the two child figures the combine re-rendered).

## Read the compatibility report

The combined bundle's report opens with a **Combine Provenance & Compatibility**
section. Its table, *What Was Combined (data provenance)*, lists each combined
experiment with its role, model, member count, and the toolkit and solver builds
that produced it. The compatibility check itself runs before the report is built.
It compares a fixed set of identity, sensitivity and compute fields, plus the
RO-Crate schema version, and leaves out the analysis id and the toolkit version. It
classifies each difference it finds by severity, and it records the result in the
combined bundle's `combined_compatibility.json`. A non-blocking
divergence that the combine's intended structure does not account for appears as a
warning row under **Cross-Experiment Errors and Warnings**:

| Severity | Meaning | Effect |
|----------|---------|--------|
| `informational` | An expected divergence (e.g. HPC/compute config). | Recorded only in `combined_compatibility.json`; non-blocking. |
| `warning` | A surfaced difference that may matter (e.g. a sensitivity-axis field, or an RO-Crate schema-version skew). | Recorded in `combined_compatibility.json`, and also listed under **Cross-Experiment Errors and Warnings** unless the combine expects it (a model toggle between paired arms, `sensitivity_analysis`, or `TRITONSWMM_branch_key`); non-blocking. |
| `blocking` | A different experiment entirely (e.g. differing weather events or enabled models). | **Aborts** the combine. |

??? note "How this severity differs from the bug-registry severity"
    `CompatibilitySeverity` (combine-admissibility) is distinct from the
    bug-registry `severity` (output-invalidation). They answer different questions:
    one decides whether two bundles may be combined, the other whether a recorded
    bug invalidates an output.

### Compatibility gates

`hhemt combine` refuses to combine bundles from two different case studies: a
`case_name` divergence (sourced from each bundle's `case.yaml`) is classified
**blocking** and aborts the combine. A compute-config divergence, a difference in the
`partitions` map or `gpu_allocation_flavor`, sourced from each bundle's scrubbed
`hpc_system_config.identity.yaml`, is classified **informational**: it is expected, so
it is recorded only in the combined bundle's `combined_compatibility.json` and does not
abort, because comparing the same experiment across UVA and Frontier is a supported use. The combined report additionally carries a
clean-vs-resume `cross_experiment_intercomparison` figure under the "Cross-Experiment
Results" category, projected from the `combined_intercomparison.json` read-model, so a
local re-render of the combined report picks it up with no re-merge.

The enabled models come from the system config's toggles. Bundles from the same
experiment agree on them; the paired model arms of one experiment are the one
divergence the combine admits as a warning rather than a block. A single-model arm
sets, for example:

```yaml
toggle_triton_model: false
toggle_tritonswmm_model: true
toggle_swmm_model: false
```

## Regenerate or inspect the combined report

`hhemt report-from-bundle combined.zip` re-renders a combined bundle. It unpacks the
render set beside the zip, into `combined/`, and renders there; a second run reuses the
folder while the zip is unchanged. In Python, `CombinedBundle.from_zip(path)` does the
unpack and `CombinedBundle.from_directory(folder)` opens a folder, and either handle's
`regenerate_report()` renders.

Each input experiment is preserved intact: as `child_crates/{experiment_id}.zip` in the
combined zip, and as `child_crates/{experiment_id}/` in a `--unzipped` folder or after unpacking.
A child is a complete single-analysis bundle, so `hhemt report-from-bundle` renders
one once every figure's recorded build stamp matches the running build. It refuses with
a `StalePlotsError` otherwise, and also when the figures carry no stamp at all or
disagree with each other, and the override is Python-only:
`Bundle.from_directory(child).regenerate_report(declare_stale_plots=True)`.
A child's EDA surface is the per-experiment one, reached through the Python API
(`Bundle.from_directory(child).eda()`), which carries no build guard. A combined
bundle has no aggregate EDA surface.

??? note "How the combined report is laid out"
    The combined report is one cohesive Snakemake `--report`. Its sidebar opens
    with four cross-experiment sections, in this order:
    **Combine Provenance & Compatibility**, **Cross-Experiment Results** (the
    clean-vs-resume intercomparison, as a table and as maps),
    **Cross-Experiment Errors and Warnings** (each experiment's validation checks,
    side by side), and **Cross-Experiment Disk Utilization**. Below them, the
    combined experiments' own figures are filed under the section each figure has
    in a single-experiment report (for example System Information). The
    sub-heading names the experiment, followed by the figure's own sub-heading when
    it has one. The name is the experiment id without its trailing
    model token, so `synth_cc_clean_tritonswmm` and `synth_cc_clean_triton` both
    appear as `synth_cc_clean`. An experiment's TRITON-SWMM and pure-TRITON runs
    share one page, with the pure-TRITON figure below the TRITON-SWMM one. A figure
    that no section claims is listed under Other. These pages are composed at
    combine time under `paired_figures/` from each experiment's figures in
    `child_crates/{experiment_id}/plots/`, so there is no click-through to a
    separate document. There is no `index.html` front door: everything is
    navigable inside the single `analysis_report.html` (or the size-friendly
    `analysis_report.zip`).

The combined report is a first-class Snakemake `--report` (shared chrome, nav, and CSS
with the single-experiment reports), and re-rendering it locally from the bundled
read-models needs no re-merge.
