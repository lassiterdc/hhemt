# Capabilities

## Supported experiments

The `toggle_sensitivity_analysis` field of the analysis configuration selects between an **[event ensemble](../reference/glossary.md#event-ensemble)**, whose members are weather events, and a **[sensitivity analysis](../reference/glossary.md#sensitivity-analysis)**, whose members are the rows of a parameter table. A [benchmarking](../reference/glossary.md#benchmarking) experiment is a sensitivity analysis.

```yaml
toggle_sensitivity_analysis: false  # event ensemble; true selects a sensitivity analysis
```

Both types run the same event through any model type and run unchanged on any supported hardware.

### Ensemble flood studies

Simulations run in parallel. Outputs consolidate into hierarchical `DataTree` zarr stores stamped with CF-1.13 conventions, ready for objective-function computation and analysis across the full population of simulated events. Snakemake, the open-source workflow manager, uses the allocated resources and skips completed work, so re-running an analysis repeats only the steps whose inputs changed or whose outputs are missing.

### Sensitivity analysis

A parameter table turns each row into a [member](../reference/glossary.md#member), a separate analysis. A sensitivity analysis can vary the SWMM hydraulic model, Manning's roughness, the grid resolution, the compute configuration, and other inputs. [Benchmarking](../reference/glossary.md#benchmarking) produces a report with standard verification metrics and hydraulic-output visualizations, plus benchmarking-specific figures for run time, compute cost, and strong-scaling speedup. A benchmarking experiment on your own machine or HPC system finds the compute configuration that best fits your hardware and the system you intend to simulate.

### Inter-model comparison

The toolkit compiles and runs any combination of TRITON alone, SWMM alone, and the coupled TRITON-SWMM as independently toggled options, and keeps the outputs of each model in a separate store. Running the same event through more than one model gives a direct comparison. The comparison measures how much the bidirectional coupling between surface flooding and stormwater hydraulics changes results and shows whether coupling justifies the added cost.

## Hardware portability

GPU code compiles against the CUDA backend for NVIDIA or the HIP/ROCm backend for AMD, selected through configuration YAML files. CPU-only execution is also supported. The toolkit runs on SLURM clusters with AMD GPUs and on SLURM clusters with NVIDIA GPUs. A large ensemble runs as one big SLURM allocation through `multi_sim_run_method: 1_job_many_srun_tasks`, or as a batch of many fit-to-purpose SLURM jobs, one per workflow step, through `multi_sim_run_method: batch_job`.

## See also

- The [Quickstart](../quickstart.md) starts a first simulation.
- The [example interactive report](../reference/example-report.md) shows a rendered report.
