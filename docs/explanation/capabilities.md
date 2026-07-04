# Capabilities

The H&H Ensemble Modeling Toolkit (hhemt) orchestrates coupled TRITON-SWMM flood simulations as embarrassingly-parallel ensembles across local workstations and HPC clusters, using configurable CPU or GPU resources per simulation on both NVIDIA and AMD hardware. It manages the full lifecycle — preprocessing, compilation, execution, and post-processing — producing analysis-ready datasets and an interactive report.

## Three classes of study

From a single codebase, the toolkit makes three classes of study tractable: **model comparison** across TRITON, SWMM, and the coupled TRITON-SWMM; **ensemble flood studies** over many events; and **sensitivity analysis**, including the fully built-out benchmarking study type. A fourth, cross-cutting capability — **hardware portability** — lets any of these run unchanged on CPU or on NVIDIA or AMD GPUs, on a workstation or on HPC.

## Model comparison across TRITON, SWMM, and coupled TRITON-SWMM

The toolkit compiles and runs any combination of the three model types — TRITON alone, SWMM alone, or the coupled TRITON-SWMM — as independently toggled options, keeping each model's outputs in its own store. Running the same event through more than one model produces a direct intermodel comparison, so you can measure how much the tight two-way coupling between surface flooding and stormwater hydraulics changes results — and decide whether the added cost of coupling is warranted.

## Ensemble flood studies

The toolkit parallelizes simulations rather than running them in sequence, so large ensembles of events — for flood studies and uncertainty assessment — are tractable. Outputs consolidate into hierarchical `DataTree` zarr stores stamped with CF-1.13 conventions, ready for objective-function computation and analysis across the full population of simulated events. Snakemake, the open-source workflow manager, maximizes utilization of the allocated resources and skips work that has already completed, so re-running an analysis never repeats finished steps.

## Hardware portability

GPU code compiles against either the CUDA (NVIDIA) or HIP/ROCm (AMD) backend, selected through configuration YAML files; CPU-only execution is also supported. The toolkit runs on both ORNL's Frontier (AMD) and UVA's Rivanna (NVIDIA), which both use the SLURM scheduler. It can run a large ensemble as one big SLURM allocation (typical on Frontier) or as a batch of many fit-to-purpose SLURM jobs, one per workflow step (typical on Rivanna).

## Sensitivity analysis and benchmarking

A parameter table you provide turns each row into a modeled scenario, so almost any input can be varied across a study — the SWMM hydraulic model, Manning's roughness, the grid resolution, and the compute configuration among them. Benchmarking is the first fully built-out sensitivity-analysis type: it produces a report with standard verification metrics and hydraulic-output visualizations, plus benchmarking-specific figures for run time, compute cost, and strong-scaling speedup. Run a benchmarking study on your own machine or HPC system to find the compute configuration that best fits your hardware and the system you intend to simulate.

## See also

- [Quickstart](../tutorials/quickstart.md) — try it.
- [Example interactive report](../reference/example-report.md) — see a rendered report.
