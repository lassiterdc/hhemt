# H&H Ensemble Modeling Toolkit

Running one coupled flood simulation is a solved problem. Running four hundred of
them, across a sweep of events and compute configurations, and getting back a single
analysis-ready dataset you can defend in a paper, is not.

The H&H Ensemble Modeling Toolkit (hhemt) orchestrates coupled TRITON-SWMM flood
simulations as embarrassingly-parallel ensembles across local workstations and HPC
clusters, using configurable CPU or GPU resources per simulation on both NVIDIA and
AMD hardware. It manages the full lifecycle, from preprocessing and compilation
through execution and post-processing, and produces consolidated datasets and an
interactive report.

--8<-- "platform-support.md"

## What a run looks like

Two YAML configs describe the study, and the toolkit does the rest:

```python
from hhemt.experiments import NorfolkIreneExperiment

norfolk = NorfolkIreneExperiment.load()          # fetches the case-study data once
result = norfolk.analysis.run(from_scratch=False, execution_mode="auto")
norfolk.analysis.render_report()                 # self-contained interactive report
```

That is the whole worked path for the shipped example. The same two calls run a
four-hundred-member ensemble on a cluster; what changes is the configuration, not
the code you write.

**[Start with the Quickstart](tutorials/quickstart.md)**, which takes a fresh clone
to a finished report.

## What it is for

Three classes of study, from one codebase:

- **Model comparison.** Run the same event through TRITON, SWMM, and the coupled
  TRITON-SWMM, and measure what the two-way coupling changes.
- **Ensemble flood studies.** Many events in parallel, consolidated into
  hierarchical zarr stores with CF-1.13 metadata, ready for analysis across the
  whole population.
- **Sensitivity analysis and benchmarking.** A parameter table turns each row into
  a scenario, so you can vary the hydraulic model, roughness, grid resolution, or
  the compute configuration itself.

[Capabilities](explanation/capabilities.md) covers each in depth, and
[Limitations](explanation/limitations.md) is candid about where the toolkit stops.

## How to cite

If you use hhemt in published work, cite it:

```text
Lassiter, D. (2026). H&H Ensemble Modeling Toolkit (hhemt) (Version 0.1.0)
[Computer software]. https://doi.org/10.5281/zenodo.21359152
```

Two DOIs exist, and which one you want depends on what you are claiming:

| DOI | Resolves to | Use it when |
|---|---|---|
| `10.5281/zenodo.21359152` | version 0.1.0, permanently | **Citing in a paper.** It pins the exact version your results came from. |
| `10.5281/zenodo.21359151` | always the latest version | Pointing a reader at the software in general, rather than at the version you ran. |

`CITATION.cff` in the repository root carries the same metadata in a
machine-readable form, and GitHub renders it as a "Cite this repository" button.

## Where to go next

- **[Installation](how-to/installation.md)**: environment setup, and the pip-only
  alternative.
- **[Tutorials](tutorials/index.md)**: learning-oriented walkthroughs. Start with
  the [Quickstart](tutorials/quickstart.md) if you are new.
- **[How-To Guides](how-to/index.md)**: task-oriented recipes for a specific job,
  from configuring a run to publishing a dataset.
- **[Reference](reference/index.md)**: the API surface, the config schema, and what
  the toolkit produces.
- **[Explanation](explanation/index.md)**: why the design is what it is, and what it
  can and cannot do.
