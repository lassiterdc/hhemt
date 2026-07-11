# Combining experiments into one cross-experiment report

`hhemt combine` ingests **two or more** completed render bundles from different
experiments, checks that they are compatible, and emits a single standalone
**combined bundle** with one cross-experiment report — itself a valid RO-Crate
containing each input bundle intact.

Use this when you have run the same (or a compatible) analysis under different
conditions — e.g. two compute configurations, or two clusters — and want one
report that presents them side by side.

## Prerequisites

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
  bundles** (each ships `sensitivity_datatree.zarr` at the bundle root). The
  combine step resolves whichever consolidated tree a bundle ships. The
  cross-experiment report presents the compatibility table across the combined
  set; a cross-experiment byte-identity data panel over the deeper
  sensitivity-master tree shape is a future addition.

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

The combined bundle's report opens with a **Cross-Experiment Compatibility**
panel summarizing every field on which the bundles diverge, each classified by
severity:

| Severity | Meaning | Effect |
|----------|---------|--------|
| `informational` | An expected divergence (e.g. HPC/compute config). | Surfaced; non-blocking. |
| `warning` | A surfaced difference that may matter (e.g. a sensitivity-axis field, or an RO-Crate schema-version skew). | Surfaced; non-blocking. |
| `blocking` | A different experiment entirely (e.g. differing weather events or enabled models). | **Aborts** the combine. |

`CompatibilitySeverity` (combine-admissibility) is distinct from the ADR-17
bug-registry `severity` (output-invalidation) — they answer different questions.
See the decision doc *"CompatibilitySeverity is orthogonal to ADR-17 severity"*.

> **Note:** compute-config / HPC identity is not currently serialized into a
> bundle, so a pure compute-config difference produces **no** divergence row (the
> bundles read as identical to the checker). Making HPC divergence a visible
> `informational` row is a planned enhancement.

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
