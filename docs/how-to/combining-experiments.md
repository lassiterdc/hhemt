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
- **Two or more emitted bundles on disk.** Produce each with
  `hhemt bundle` (or `Analysis.bundle_report_data()` / the sensitivity-master
  `TRITONSWMM_sensitivity_analysis.bundle_report_data()`) after a completed
  `analysis.run()` + `render_report()`.
- The bundles must be **unpacked directories**, not the emitted `.zip`.
  `hhemt bundle` writes a `ZIP_STORED` archive; unzip each before combining:

  ```bash
  unzip bundle_a.zip -d bundle_a/
  unzip bundle_b.zip -d bundle_b/
  ```

- Combine operates on **single-analysis** bundles (each ships its consolidated
  `analysis_datatree.zarr` at the bundle root) **and on sensitivity-master
  bundles** (each ships `experiment_datatree.zarr` at the bundle root; a bundle
  produced before the store unification ships `analysis_datatree.zarr` or
  `sensitivity_datatree.zarr` and is still accepted). The
  combine step resolves whichever consolidated tree a bundle ships. The
  cross-experiment report presents the compatibility table across the combined
  set plus, for a clean-vs-resume sensitivity-master pair, a
  `cross_experiment_intercomparison` panel derived cross-bundle from the two
  consolidated trees (see *Compatibility gates* below).

## Combine the bundles

```bash
hhemt combine bundle_a/ bundle_b/ -o combined/
```

- Positional arguments: two or more bundle directories.
- `-o` / `--output` (optional): the target directory for the combined bundle.
  Defaults to `{first_bundle}/../combined_{n}bundles_{git_sha}/`.

The command (1) runs a metadata-compatibility check, (2) merges the bundles'
consolidated trees, (3) renders one cross-experiment report, and (4) writes a
standalone combined bundle at the output path. It **aborts with an error** if the
bundles carry a *blocking* divergence (see below).

## Read the compatibility report

The combined bundle's report opens with a **Combine Provenance & Compatibility**
section. Its table, *What Was Combined (data provenance)*, lists each combined
experiment with its role, model, member count, and the toolkit and solver builds
that produced it. The compatibility check itself runs before the report is built.
It classifies every field on which the bundles diverge by severity, and it records
the result in the combined bundle's `combined_compatibility.json`. A non-blocking
divergence that the combine's intended structure does not account for appears as a
warning row under **Cross-Experiment Errors and Warnings**:

| Severity | Meaning | Effect |
|----------|---------|--------|
| `informational` | An expected divergence (e.g. HPC/compute config). | Surfaced; non-blocking. |
| `warning` | A surfaced difference that may matter (e.g. a sensitivity-axis field, or an RO-Crate schema-version skew). | Surfaced; non-blocking. |
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
`hpc_system_config.identity.yaml`, is classified **informational**: it is surfaced as a
divergence row and does not abort, because comparing the same experiment across UVA
and Frontier is a supported use. The combined report additionally carries a
clean-vs-resume `cross_experiment_intercomparison` figure under the "Cross-Experiment
Results" category, projected from the `combined_intercomparison.json` read-model, so
`CombinedBundle.regenerate_report()` re-renders it locally with no re-merge.

## Regenerate or inspect the combined report

The combined bundle mirrors a single-analysis bundle:

```python
from hhemt.bundle import CombinedBundle

cb = CombinedBundle.from_directory("combined/")
report = cb.regenerate_report(format="zip")  # or format="html"
```

Each input experiment is preserved intact under `combined/child_crates/{experiment_id}/`;
run `Bundle.from_directory(...).eda()` on a child directory for a per-experiment
EDA surface (a combined bundle has no aggregate EDA surface).

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
with the single-experiment reports); `CombinedBundle.from_directory(dir).regenerate_report()`
re-renders it locally from the bundled read-models with no re-merge.
