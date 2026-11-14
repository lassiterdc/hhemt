# Combining experiments into one cross-experiment report

## Goal

Merge completed render bundles of compatible experiments into one standalone **combined bundle** with a single cross-experiment report. The combined bundle is a valid RO-Crate that contains each input bundle intact. Combine experiments that ran the same analysis, or compatible analyses, under different conditions such as compute configurations or clusters, and read the results side by side.

## Prerequisites

- A working [installation](installation.md).
- More than one emitted bundle on disk. `hhemt bundle --system-config my_system.yaml --analysis-config my_analysis.yaml` writes the archive for an analysis with a rendered report, and the [CLI reference](../reference/cli.md) lists the optional flags.
- Each input is a bundle **zip**, as `hhemt bundle` writes the zip, or an **unpacked bundle directory**. Give the inputs without unzipping them. The toolkit never modifies the inputs. A bundle zip is compressed entry by entry. Zarr chunk files are stored because the toolkit compresses them already, and every other entry is deflated at zlib level 6. A zip that stores every entry uncompressed is accepted too.
- Single-analysis bundles ship the consolidated `analysis_datatree.zarr` at the bundle root, and sensitivity-primary bundles ship `experiment_datatree.zarr` at the bundle root. Combine operates on both. A bundle with `analysis_datatree.zarr` or `sensitivity_datatree.zarr` at the root is accepted too. Combine resolves whichever consolidated tree a bundle ships and records the divergences that the compatibility check finds in `combined_compatibility.json` of the combined bundle. For a clean-versus-resume sensitivity-primary pair, the report presents a `cross_experiment_intercomparison` panel that the combine derives from both consolidated trees, as [Compatibility gates](#compatibility-gates) describes.

## Key references

- The [CLI reference](../reference/cli.md#inspecting-and-reporting) lists the flags of `hhemt combine` and `hhemt report-from-bundle`.
- The [Bundles API reference](../reference/api-bundles.md) documents `CombinedBundle`.
- [Producing a report](../reference/example-report.md#producing-a-report) describes the report container.

## Combine the bundles

```bash
hhemt combine bundle_a.zip bundle_b.zip -o combined
```

- The positional arguments name the bundle zips or bundle directories of several bundles.
- `-o` or `--output`, optional, writes the combined bundle as the zip `{output}.zip`, and a value that already ends in `.zip` is used as given. The default is `{first_bundle}/../combined_{n}bundles_{git_sha}.zip`.
- `--unzipped`, optional, writes a folder at `--output` in place of a zip, with the full bundle of each experiment under `child_crates/{experiment_id}/`.
- `--workdir`, optional, sets the directory that holds the private working folder of the combine. The default is the system temporary directory. The folder holds the render set of each input, meaning everything except the raw `out_*/bin` solver output that the combine ignores. The combine removes the folder on exit, including on SIGTERM.

The command runs the steps in order.

1. Refuse an input whose consolidated tree was carried without data.
2. Extract the render set of each input into the working folder.
3. Run a metadata-compatibility check.
4. Render one cross-experiment report.
5. Write the combined bundle at the output path, replacing any earlier output.

The command **aborts with an error** when the bundles carry a *blocking* divergence, as the severity table describes. The combined zip holds the root files, meaning the crate, the read-models, the manifest, the report, and `plots/`. Per experiment, the zip also holds `child_crates/{experiment_id}.zip`, the bundle zip of the experiment, and `child_crates/{experiment_id}.refreshed/`, the child figures that the combine re-rendered.

## Read the compatibility report

The report of the combined bundle opens with a **Combine Provenance & Compatibility** section. The provenance table lists each combined experiment with the role, the model, the member count, the toolkit build, and the solver build that produced the experiment. The compatibility check runs before the report is built. The check compares a fixed set of identity, sensitivity, and compute fields, plus the RO-Crate schema version, and leaves out the analysis id and the toolkit version. The check classifies each difference by severity and records the result in `combined_compatibility.json` of the combined bundle. A non-blocking divergence that the intended structure of the combine leaves unexplained appears as a warning row under **Cross-Experiment Errors and Warnings**.

| Severity | Meaning | Effect |
|----------|---------|--------|
| `informational` | An expected divergence, such as the HPC compute config. | Recorded in `combined_compatibility.json` only. Non-blocking. |
| `warning` | A surfaced difference that may matter, such as a sensitivity-axis field or an RO-Crate schema-version skew. | Recorded in `combined_compatibility.json` and listed under **Cross-Experiment Errors and Warnings**, unless the combine expects the difference. A model toggle between paired arms, `sensitivity_analysis`, and `TRITONSWMM_branch_key` are expected. Non-blocking. |
| `blocking` | A different experiment entirely, such as differing weather events or enabled models. | **Aborts** the combine. |

??? note "Compatibility severity versus bug-registry severity"
    `CompatibilitySeverity` decides whether bundles may be combined. The bug-registry `severity` decides whether a recorded bug invalidates an output.

### Compatibility gates

`hhemt combine` refuses to combine bundles from different case studies. A `case_name` divergence sourced from the `case.yaml` of each bundle is classified **blocking** and aborts the combine. A compute-config divergence, meaning a difference in the `partitions` map or `gpu_allocation_flavor` sourced from the scrubbed `hpc_system_config.identity.yaml` of each bundle, is classified **informational**. The divergence is expected, so the combine records the divergence only in `combined_compatibility.json` of the combined bundle and leaves the combine running, because comparing the same experiment across clusters is a supported use. The combined report also carries a clean-versus-resume `cross_experiment_intercomparison` figure under the "Cross-Experiment Results" category, projected from the `combined_intercomparison.json` read-model, so a local re-render of the combined report picks up the figure with no re-merge.

The enabled models come from the toggles of the system config. Bundles from the same experiment agree on the toggles. The paired model arms of a single experiment are the sole divergence that the combine admits as a warning and never as a block. A single-model arm sets the toggles as follows.

```yaml
toggle_triton_model: false
toggle_tritonswmm_model: true
toggle_swmm_model: false
```

## Regenerate or inspect the combined report

`hhemt report-from-bundle` regenerates a combined report. Point the command at the combined zip or at the combined folder.

```bash
hhemt report-from-bundle combined.zip
```

The command unpacks the render set of a zip beside the zip, into `combined/`, and renders there. A second invocation reuses the folder while the zip is unchanged. A bundle is a combined bundle when the bundle holds `child_crates` and `bundle_manifest.json` and carries no root `cfg_analysis.yaml`. A combined bundle has no analysis config, so the report container defaults to `auto`. `--format html` or `--format zip` chooses the container, as [Producing a report](../reference/example-report.md#producing-a-report) describes. The render refuses with a `StalePlotsError` when the build stamp of a figure differs from the running build, and `--declare-stale-plots` overrides the refusal, as the [CLI reference](../reference/cli.md#inspecting-and-reporting) describes. In Python, `CombinedBundle.from_zip(path)` unpacks a zip, `CombinedBundle.from_directory(folder)` opens a folder, and `regenerate_report()` on either handle renders. The [Bundles API reference](../reference/api-bundles.md) documents both.

??? note "Rendering or inspecting a single child of a combined bundle"
    Each input experiment stays intact, as `child_crates/{experiment_id}.zip` in the combined zip and as `child_crates/{experiment_id}/` in an `--unzipped` folder or after unpacking. A child is a complete single-analysis bundle, so `hhemt report-from-bundle` renders a child once the recorded build stamp of every figure matches the running build. The command refuses with a `StalePlotsError` otherwise, and when the figures carry no stamp or disagree with each other. The override is `hhemt report-from-bundle --declare-stale-plots` on the directory of the child, or in Python `Bundle.from_directory(child).regenerate_report(declare_stale_plots=True)`. The EDA surface of a child is the per-experiment surface, reached through the Python API with `Bundle.from_directory(child).eda()`, and that call carries no build guard. A combined bundle has no aggregate EDA surface.

??? note "Layout of the combined report"
    The combined report is one cohesive Snakemake `--report`. The sidebar opens with the cross-experiment sections in order.

    - **Combine Provenance & Compatibility**
    - **Cross-Experiment Results**, the clean-versus-resume intercomparison as a table and as maps
    - **Cross-Experiment Errors and Warnings**, the validation checks of each experiment side by side
    - **Cross-Experiment Disk Utilization**

    Below the sections, the figures of the combined experiments are filed under the section that each figure has in a single-experiment report, such as System Information. The sub-heading names the experiment, followed by the sub-heading of the figure, when the figure carries a sub-heading. The name is the experiment id without the trailing model token. The ids `synth_cc_clean_tritonswmm` and `synth_cc_clean_triton` both appear as `synth_cc_clean`. The TRITON-SWMM and pure-TRITON analyses of an experiment share one page, with the pure-TRITON figure below the TRITON-SWMM figure. A figure that no section claims is listed under Other. The combine composes the pages under `paired_figures/` from the figures of each experiment in `child_crates/{experiment_id}/plots/`, so no click-through to a separate document exists. No `index.html` front door exists, and everything is navigable inside the single `analysis_report.html`, or the size-friendly `analysis_report.zip`.

The combined report is a Snakemake `--report` with the navigation and styling of the single-experiment reports. Re-rendering the report locally from the bundled read-models needs no re-merge.
