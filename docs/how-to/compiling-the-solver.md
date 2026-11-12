# Compile the solver

## Goal

Build the TRITON-SWMM solver once on the machine that runs your simulations, so that the setup step of every analysis finds the solver.

## Prerequisites

- A working [installation](installation.md) with the `hhemt` environment activated.
- `git` and `make` on your `PATH`. The operating system provides both. The conda environment provides the compiler, `cmake`, and MPI.

## Key references

- [Diagnosing a failed analysis](diagnosing-a-failed-run.md) reads the `_status/` ladder and the logs.
- [HPC-profile setup](hpc-profile-setup.md) describes the GPU hardware entry and the modules that a build uses.
- [Configuration schema](../reference/config-schema.md) lists the `toggle_*` and `TRITONSWMM_*` fields.

## Compilation as a separate step

In native mode, the default `execution_environment: native`, an analysis executes the solver under `TRITONSWMM_software_directory` of your system config. The setup step of every analysis checks that each enabled model is compiled there and stops with an error when a model is missing. The setup step never compiles. You run the build once per machine and once per GPU target, and later calls skip a backend whose build succeeded.

The system config fields that select the models and the source location look like this.

```yaml
toggle_triton_model: false
toggle_tritonswmm_model: true
toggle_swmm_model: false
TRITONSWMM_software_directory: /path/to/triton
TRITONSWMM_git_URL: https://code.ornl.gov/hydro/triton.git
```

In container mode the solver ships inside the image, and [Building container images](building-container-images.md) covers the images.

## On a workstation

Compile with the standalone setup command. The command takes the same system and analysis configs as an analysis.

```bash
python -m hhemt.setup_workflow \
    --system-config my_system.yaml \
    --analysis-config my_analysis.yaml \
    --compile-triton-swmm
```

On first use the command clones the pinned TRITON-SWMM source into `TRITONSWMM_software_directory`, writes a build script named `compile_cpu.sh` into your system directory, and runs the script. A build takes a few minutes. The command prints `[CPU] ✓ Compilation successful!` when the build finishes. After a failure the command prints `[CPU] ✗ Compilation failed` and the path of the log to read. An existing build makes the command print `[CPU] Already compiled successfully (skipping)` and return at once.

Pass one flag for each model that the system config enables.

| Toggle | Command-line flag |
|---|---|
| `toggle_tritonswmm_model` | `--compile-triton-swmm` |
| `toggle_triton_model` | `--compile-triton-only` |
| `toggle_swmm_model` | `--compile-swmm` |

Run the same command again to list the built backends. The command skips and reports a backend whose build succeeded.

## On a cluster

An analysis on a GPU partition needs a GPU build keyed to the partition. The partition entry in your HPC system profile sets the GPU architecture and backend, as [Choose a partition](hpc-profile-setup.md#choose-a-partition) describes. The build script loads the environment modules that the profile lists. The command reads the profile only when you pass `--hpc-system-config`, and without the flag the command builds the CPU backend alone. A cluster build takes the profile and the partition that `hpc_ensemble_partition` of your analysis config names.

```bash
python -m hhemt.setup_workflow \
    --system-config my_system.yaml \
    --analysis-config my_analysis.yaml \
    --hpc-system-config hpc_system_config_uva.yaml \
    --target-partition gpu-a6000 \
    --compile-triton-swmm
```

Add `--compile-triton-only` or `--compile-swmm` for the other models that you enabled. The command builds the CPU backend and, when the partition declares a `gpu_compilation_backend`, the GPU backend as well. Each backend gets a directory under `TRITONSWMM_software_directory`, named `build_tritonswmm_cpu` or `build_tritonswmm_gpu_{gpu_hardware}`. Run the command once per GPU partition that you submit to. A CPU partition needs only the CPU build. A second call also lists the built backends, because the command skips a backend whose build succeeded.

An [experiment definition](running-an-experiment-definition.md) supplies the config paths as the resolved copies that `hhemt run-experiment --dry-run` writes to `$SCRATCH_DIR/resolved_configs/`.

## Recompilation triggers

The check of an analysis reads the log that the build left in the build directory and asks whether the build succeeded. The check compares no value against your config or the source pin, so a change to either needs your action.

- **You changed `TRITONSWMM_branch_key`.** Remove the clone at `TRITONSWMM_software_directory` and compile again. The build directories live inside the clone. A compile refuses to proceed while the checkout of the clone disagrees with the pin, and the message names the directory to remove.
- **You changed `TRITONSWMM_git_URL`.** Remove the clone the same way. No check compares the remote of the clone against the config, so a clone from the old URL with a checkout that matches the pin is reused silently.
- **You will submit to a GPU partition whose `gpu_hardware` has no build.** Compile with that partition as `--target-partition`. The new build lands in a separate directory, and the existing builds stay untouched.
- **You want an existing backend rebuilt** after a change to the modules that the profile loads, or because a build is suspect. Pass `--recompile-if-already-done` to the command.

No other change calls for a rebuild. Updating hhemt, changing `execution_mode` or `multi_sim_run_method`, editing the analysis config, and `hhemt run --from-scratch` all leave the build in place. `--from-scratch` clears the analysis directory. The build lives under the software directory.

## If an analysis stops at setup

An analysis stops at the first step when the setup check finds an uncompiled model. [Diagnosing a failed analysis](diagnosing-a-failed-run.md#setup-says-a-model-is-enabled-but-not-compiled) shows the line that the setup check leaves in the setup log and the remedy. The compile on this page for your machine satisfies the check.
