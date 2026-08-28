# H&H Ensemble Modeling Toolkit

The H&H Ensemble Modeling Toolkit (hhemt) orchestrates coupled TRITON-SWMM flood simulations as embarrassingly-parallel ensembles across local workstations and HPC clusters, using configurable CPU or GPU resources per simulation on both NVIDIA and AMD hardware. It manages the full lifecycle (preprocessing, compilation, execution, and post-processing), producing analysis-ready datasets and an interactive report.

## Where to start

- **[Tutorials](tutorials/index.md)**: learning-oriented walkthroughs. Start
  with the [Quickstart](tutorials/quickstart.md) if you are new.
- **[How-To Guides](how-to/index.md)**: task-oriented recipes for a specific
  job: installing, configuring, running on HPC, publishing, reproducing.
- **[Reference](reference/index.md)**: the API surface, the config schema, and
  what the toolkit produces.
- **[Explanation](explanation/index.md)**: why the design is what it is, and
  what it can and cannot do.
