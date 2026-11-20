---
type: workflow
title: Reporting, render bundles, and experiment combination
description: Select report renderer sets, consume portable bundles without rerunning simulations, and combine compatible completed experiments.
tags: [reporting, bundles, portability]
verified:
  - by: openwiki/0.9.0
    at: 2026-11-20T07:08:41.116Z
sources:
  - id: openwiki-source-c26c3d0fa341d015d80eb178
    resource: repo://docs/reference/reporting-sets.md
  - id: openwiki-source-1e6bf8fb0554c9f030b787ad
    resource: repo://src/hhemt/bundle/__init__.py
  - id: openwiki-source-faeebb5f7c329d37d48cc997
    resource: repo://src/hhemt/bundle/_combine.py
  - id: openwiki-source-8ff8b8461603e28cd76778f9
    resource: repo://src/hhemt/bundle/_path_policy.py
  - id: openwiki-source-0939f8f536ea2d1f3219be0f
    resource: repo://src/hhemt/config/report.py
generated: { by: "omp", at: "2026-11-20T07:08:41.116Z" }
---

# Reporting, render bundles, and experiment combination

## Select a report before computation

`report.reporting_set` selects one registered set or an ordered list of sets. An omitted or empty selection resolves through the `default` sentinel: ordinary analyses use the default report, sensitivity analyses use `sensitivity`. Sets determine renderer membership and sidebar ordering; `disabled_renderers` subtracts named renderers rather than selecting a different report. See `docs/reference/reporting-sets.md` for the selection recipes.

Run-entry validation checks each set's analysis shape and required varied axes, then composes the selection. `combined` belongs to cross-experiment reporting and cannot be selected on an analysis. Unknown disabled renderer keys fail against the composed selection. Benchmarking validation runs once when required by the composed set. These checks prevent an unsuitable report configuration from silently proceeding to compute.

## Consume results, not an executable analysis

`Bundle` is deliberately separate from `TRITONSWMM_analysis`: a completed bundle supports report regeneration and EDA, not `Analysis.run()`. Typical Python consumption is:

```python
from hhemt.bundle import Bundle
bundle = Bundle.from_directory("unpacked_bundle")
bundle.regenerate_report()
```

Construction requires a manifest and validated `cfg_analysis.yaml`. Bundle schema compatibility is exact: newer bundle schemas require upgrading the reader; older schemas require re-emission from the source analysis. A differing producing toolkit SHA is separately warned about as a provenance caveat, not treated as a schema failure.

Portable archived configuration uses explicit per-field path policies. Carried inputs become bundle-relative paths, roots become `.` where required, and host-specific binary paths can become `None`. Consumers rebase relative `Path` fields against the bundle root before model validation. This validation checks archive shape, not universal file existence: each consuming operation owns its actual input requirements. In particular, regeneration checks a declared branding file before opening it.

## Regenerate a report locally

Regeneration checks bundled plot build stamps before rendering; `declare_stale_plots=True` explicitly permits stale plots. This override does not rebuild solver results. It stages report artifacts and generates a report-only Snakefile, then uses Snakemake touch/report subprocesses rooted at the bundle. No simulation, processing, or consolidation rules are generated on this path.

The report format defaults to the bundled configuration. `auto` chooses HTML below 15 million figure bytes and ZIP otherwise; explicit HTML or ZIP overrides that choice. Python callers must resolve stale Snakemake locks themselves: regeneration refuses nonempty `.snakemake/locks`. A missing declared render input indicates an incomplete bundle and requires re-emission, not silently using defaults.

## Combine completed experiments

```bash
hhemt combine bundle_a.zip bundle_b.zip -o combined
hhemt report-from-bundle combined.zip
```

`combine_bundle` requires at least two directory or ZIP inputs. Before staging, it refuses a root consolidated tree carried metadata-only. It extracts each input's render set into a private working directory, checks compatibility and solver provenance, verifies root consolidated trees, writes cross-experiment read-models, and renders the `combined` report. It does not modify the inputs.

Blocking compatibility differences abort. `declare_solver_split` permits a deliberate differing solver pin, but does not waive the check that bundles sharing one branch key agree on producing solver SHA and commit date. Comparing experiments across compute configurations is distinct from accepting inconsistent solver provenance.

The default output is a ZIP, with child bundle archives and refreshed child figures; `--unzipped` produces a directory with child crates. Existing output is replaced. Temporary working state is removed in a `finally` block, with SIGTERM cleanup also installed. Use `--workdir` when the default temporary filesystem cannot hold the staged render set.

## Boundaries and further reading

- [Output data model](../concepts/output-data-model.md): consolidated trees consumed by reports.
- [Provenance and publication](../integrations/provenance-and-publication.md): identity and publication contracts.
- [Experiment reproduction](experiment-reproduction.md): executing an experiment rather than merely rendering its outputs.
- [Exploratory analysis](exploratory-analysis.md): EDA artifact production and bundle consumption.

This page documents source behavior; no solver execution, report rendering, remote API call, or test execution was performed while writing it.
