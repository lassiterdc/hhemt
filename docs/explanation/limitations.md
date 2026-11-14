# Limitations and constraints

## The solvers contain the science

The toolkit is an **orchestration framework**. TRITON supplies the 2D hydrodynamics and
SWMM supplies the stormwater network. hhemt compiles the solvers, prepares their
inputs, runs them at scale, and consolidates their outputs.

- **A solver's limitations are hhemt's limitations.** Any physics that TRITON omits is
  absent from every hhemt ensemble.
- **Solver version is part of your result.** The toolkit clones and builds TRITON at a
  pinned commit, and analyses built from different pins can differ. The provenance of
  your consolidated output records the pin. Setup verifies a set pin and stops on a
  mismatch.
- **Coupled and standalone SWMM are different builds.** The coupled model uses a SWMM
  vendored inside TRITON that travels with the TRITON pin. The standalone model is built
  from a separate tag.

## Environment constraints

**Python 3.11 to 3.12 only.** The `requires-python` bound is `>= 3.11, < 3.13`. A
transitive dependency forces the upper bound by pinning an older pydantic with no wheel
for 3.13. The bound lifts when that dependency relaxes.

**The SWMM engine is version-gated at runtime.** Before executing SWMM the toolkit
checks the installed `pyswmm` and `swmm-toolkit` against the certified pairing
and refuses to run against any other engine build. `pyproject.toml` pins that pairing directly, so a pip install resolves a
stack the guard accepts. A refusal names the offending versions, and setting
`HHEMT_ALLOW_UNVALIDATED_SWMM_STACK=1` overrides the guard. Using the override means
accepting a result from an unchecked engine. A pip-only environment also lacks the
exact conda-resolved graph that `environment.yaml` reproduces. Conda remains the
recommended path because `environment.yaml` pins the whole HPC stack, including the
Snakemake SLURM executor plugins. See [Installation](../how-to/installation.md).

## Scale and hardware

**Per-row partition variation requires `batch_job` dispatch.** A SLURM allocation
covers one partition, so a sensitivity analysis that varies hardware across rows needs
`batch_job` dispatch. Preflight rejects a single-allocation dispatch method in that case
before any simulation starts.

**Only junctions are eligible for coupling.** Outfalls, storage nodes, and dividers are
not coupled. [Coupled inflow nodes](../reference/coupled-inflow-nodes.md) describes the candidate
modes and the optional limit of one junction per DEM cell. The limit applies only to the
mode that selects nodes from `[INFLOWS]` lines.

**Coupled water quality is limited.** In a coupled analysis no subcatchment washoff
reaches the network, so only the sources the hydraulics model declares produce pollutant
mass. The sources are dry-weather and external inflow concentrations, RDII, and
initial concentrations. Pollutant mass that leaves for the 2D surface is counted in
`Quality Routing Continuity` External Outflow and is not transported. See the
[output data model](../reference/output-data-model.md).

**SWMM file paths are limited to 259 characters.** SWMM keeps an input, report, or
binary-output file name only up to 259 characters. A coupled simulation whose SWMM input,
report, or binary-output path is longer stops at startup with an error naming the path
and the path length. The flooding-debug report is written only when the path is at
most 259 characters. Keep the analysis directory shallow enough that a simulation's
`swmm/hydraulics.inp` and `out_tritonswmm/swmm/hydraulics.rpt` stay within the limit.
Standalone SWMM models are outside that refusal. SWMM itself shortens a longer name
without an error.

**Walltime bounds a simulation.** A simulation killed at walltime resumes from the most
recent checkpoint. The solver's checkpoint cadence sets the resumption point, and the
toolkit has no control over that cadence. Size allocations accordingly.

## Reproducibility boundaries

**GPU and CPU results differ slightly.** The consistency checking of the toolkit reports
within-hardware-family reproducibility and cross-hardware comparison separately. A GPU
result and a serial-CPU result of the same simulation are expected to differ by a small
amount. A serial result and a decomposed result on CPUs, with MPI, OpenMP, or both, are also
expected to differ slightly, because floating-point addition is not associative. The checking
reports the GPU and CPU difference as a characterized divergence.

**A render bundle's schema version must match.** A bundle emitted by a different toolkit
version is refused. Re-emit the bundle from source.

## See also

- [Capabilities](capabilities.md) covers the supported experiments and hardware.
- [Rerun triggers](../reference/rerun-triggers.md) lists the triggers behind a full
  re-execution.
- [Installation](../how-to/installation.md) states the environment contract and the
  failure modes.
