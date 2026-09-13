# Compile the solver

**Goal:** build the TRITON-SWMM solver once on the machine that will run your
simulations, so that the setup step of every run finds it.

**Prerequisites:** a working install ([Installation](installation.md)) with the
`hhemt` environment activated, and `git` and `make` on your `PATH`. The compiler,
`cmake` and MPI come with the conda environment; `git` and `make` come from the
operating system.

## Why this is a separate step

In native mode (`execution_environment: native`, the default) a run executes the
solver it finds under your system config's `TRITONSWMM_software_directory`. The
setup step of every run checks that each enabled model is already compiled there
and stops with an error when one is missing. It never compiles. The build is yours
to run, once per machine and once per GPU target, and it is cached afterwards: a
second call skips a backend whose build already succeeded.

In container mode the solver ships inside the image and there is nothing to
compile; that path is described with the container recipes in `containers/README.md`.

## On a laptop or workstation

Compile from the same Python session you run from. For the Norfolk example:

```python
from hhemt.experiments import NorfolkIreneExperiment
norfolk = NorfolkIreneExperiment.load()
norfolk.system.compile_TRITON_SWMM()
```

For your own configs, the system object hangs off the toolkit:

```python
from hhemt import Toolkit
tk = Toolkit.from_configs("my_system.yaml", "my_analysis.yaml")
tk.system.compile_TRITON_SWMM()
```

The call clones the pinned TRITON-SWMM source into `TRITONSWMM_software_directory`
on first use, writes a build script into your system directory (`compile_cpu.sh`),
and runs it. Expect a few minutes. It prints `[CPU] ✓ Compilation successful!` when
done, or `[CPU] ✗ Compilation failed` followed by the path of the log to read. When
the build already exists it prints `[CPU] Already compiled successfully (skipping)`
and returns at once.

Call one method for each model you have enabled in the system config:

| Toggle | Method | Command-line flag |
|---|---|---|
| `toggle_tritonswmm_model` | `system.compile_TRITON_SWMM()` | `--compile-triton-swmm` |
| `toggle_triton_model` | `system.compile_TRITON_only()` | `--compile-triton-only` |
| `toggle_swmm_model` | `system.compile_SWMM()` | `--compile-swmm` |

To see what is built, call `system.print_compilation_status()`.

## On a cluster

A run on a GPU partition needs a GPU build, and that build is keyed to the
partition: the GPU architecture and backend come from the partition's entry in your
[HPC system profile](hpc-profile-setup.md), and the environment modules the profile
lists are loaded inside the build script. The Python calls above never read the
profile. On a cluster use the command-line form, which reads it, and pass the
partition your analysis config names in `hpc_ensemble_partition`:

```bash
python -m hhemt.setup_workflow \
    --system-config my_system.yaml \
    --analysis-config my_analysis.yaml \
    --hpc-system-config hpc_system_config_uva.yaml \
    --target-partition gpu-a6000 \
    --compile-triton-swmm
```

Add `--compile-triton-only` or `--compile-swmm` for the other models you have
enabled. The command builds the CPU backend and, when the partition declares a
`gpu_compilation_backend`, the GPU backend as well, each into its own directory
under `TRITONSWMM_software_directory` (`build_tritonswmm_cpu` and
`build_tritonswmm_gpu_{gpu_hardware}`). Run it once for each GPU partition you will
submit to; a CPU partition needs only the CPU build. Running it a second time is
also the way to check what is built, because a backend whose build already
succeeded is skipped.

For an [experiment bundle](running-an-experiment-bundle.md), the two config paths
are the resolved copies that `hhemt run-experiment --dry-run` writes to
`$SCRATCH_DIR/resolved_configs/`.

## When to compile again

The check a run performs reads the log the build left in its build directory and
asks whether that build succeeded. It compares nothing against your config or the
source pin, so a change to either is yours to act on:

- **You changed `TRITONSWMM_branch_key`.** Remove the clone at
  `TRITONSWMM_software_directory` (the build directories live inside it) and
  compile again. A compile refuses to proceed while the clone's checkout disagrees
  with the pin, and its message says what to remove.
- **You changed `TRITONSWMM_git_URL`.** Remove the clone the same way. Nothing
  compares the clone's remote against the config, so a clone from the old URL whose
  checkout still matches the pin is reused in silence.
- **You will submit to a GPU partition whose `gpu_hardware` you have not built
  for.** Compile with that partition as `--target-partition`; the new build lands in
  its own directory and the existing ones are untouched.
- **You want an existing backend rebuilt**, after a change to the modules the
  profile loads, or because a build is suspect. Pass
  `recompile_if_already_done_successfully=True` to the method, or
  `--recompile-if-already-done` to the command.

Nothing else calls for a rebuild. Updating hhemt itself, changing `execution_mode` or
`multi_sim_run_method`, editing the analysis config, and
`analysis.run(from_scratch=True)` all leave the build in place: `from_scratch`
clears the analysis directory, and the build lives under the software directory.

## If a run stops at setup

A run stops at its first step when the setup check finds an uncompiled model. It
writes no `a_setup_*` flag under `{analysis_dir}/_status/`, and `logs/setup.log` in
the analysis directory (`logs/setup_target_{N}.log` on an experiment) ends with a
line naming the model:

```text
TRITON-SWMM is enabled but not compiled and --compile-triton-swmm not specified
```

The TRITON-only and SWMM checks write the same line with their own names and
flags. The flag the line names belongs to the command above; `hhemt run` and
`analysis.run()` do not take it. Compile as shown for your machine, then start the
run again: setup work that completed is kept.

## See also

- [Diagnosing a failed run](diagnosing-a-failed-run.md): reading the `_status/`
  ladder and the logs.
- [HPC-profile setup](hpc-profile-setup.md): where the GPU hardware and the modules
  a build uses come from.
- [Configuration schema](../reference/config-schema.md): the `toggle_*` and
  `TRITONSWMM_*` fields.
