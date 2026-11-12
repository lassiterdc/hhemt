# Running an experiment definition

## Goal

Run an [experiment definition](../reference/glossary.md#experiment-definition) with `hhemt run-experiment`. The `experiment.yaml` descriptor declares the system and analysis configs, the input datasets, the per-cluster HPC profile, the container, and a resolvable toolkit pin. The descriptor is the single config, so the command takes no positional config arguments, and any CLI flag that overrides a descriptor-declared value needs confirmation first.

## Prerequisites

- A working [installation](installation.md).
- A [compiled solver](compiling-the-solver.md#on-a-cluster) for the target partition.
- An experiment definition directory.

## Key references

- [Config filling](config-filling.md) and [HPC-profile setup](hpc-profile-setup.md) describe the system, analysis, and `hpc_system_config` contents.
- [Forcing and suppressing re-runs](forcing-reruns.md) describes `--override-force-rerun`.
- The [CLI reference](../reference/cli.md#exit-codes) lists the exit codes.

## The experiment definition layout

A conformant definition is a directory with at least the entries below. The `ExperimentConfig` descriptor model validates the directory, as [Verifying an experiment definition conforms](#verifying-an-experiment-definition-conforms) describes.

```
experiments/my_experiment/
├── experiment.yaml          # the descriptor (schema: hhemt.config.experiment_definition.ExperimentConfig)
├── README.md                # instructions to reproduce the experiment (required)
├── rerun.sh                 # re-run driver (required)
└── configs/
    ├── system_config_uva.yaml
    └── analysis_config_uva.yaml
```

The descriptor carries no identity key. The directory names the experiment definition, and the analysis config supplies the analysis identity. A minimal `experiment.yaml` follows.

```yaml
description: One-line description.
system_config: configs/system_config_uva.yaml    # definition-relative
analysis_config: configs/analysis_config_uva.yaml # definition-relative
hpc_system_config:
  uva: ../../hpc/hpc_system_config_uva.yaml   # absolute, or relative to this experiment definition directory (the one holding experiment.yaml)
inputs:
  - name: weather
    local_path: "${HHEMT_DATA_ROOT}/weather/forcing.nc"   # ${VAR}-templated, never a literal operator path
    deposit: true                                          # these bytes are part of the publish payload
    destinations:
      uva: /scratch/$USER/my_experiment/weather/forcing.nc # the per-cluster path that staging tools read
toolkit_pin:
  version: "1.0.0"                      # PyPI version: the durable, installable identifier
container:
  sha256_source: ro-crate              # the SIF digest's authoritative home is the RO-Crate
```

The `container:` block names no recipe. A descriptor that carries the key `def_recipe` is refused by name when the descriptor loads, with a message that names `container.sif_root` as the replacement. Images are addressed by identity under `sif_root` and built by `hhemt build-sifs`. `sha256_source` stays because the RO-Crate is the authoritative home of the digest.

## Run the experiment

```bash
# Plan only, building the DAG and writing nothing:
hhemt run-experiment --experiment experiments/my_experiment --cluster uva --dry-run

# Execute:
hhemt run-experiment --experiment experiments/my_experiment --cluster uva
```

`--cluster` selects the `hpc_system_config[<cluster>]` entry. The verb loads and validates `experiment.yaml`, resolves the HPC profile, and hands the system and analysis configs to the toolkit.

Compile the solver for the partition that the experiment targets before the first execution. [Compile the solver](compiling-the-solver.md#on-a-cluster) covers the command and the resolved config paths that `--dry-run` writes.

## Run modes and the wipe guard

`--mode` selects the treatment of an analysis directory that already holds work. The default is `resume`. A second invocation, or a SLURM requeue, picks up where the last invocation stopped, and `resume` never deletes completed simulations.

| `--mode` | Behaviour |
|----------|-----------|
| `resume`, the default | Continue from the last checkpoint and keep completed simulations. |
| `fresh` | Delete the whole analysis directory first and rebuild everything. The wipe guard below applies. |

The parser refuses any other value with exit 2, and `hhemt run-experiment --help` lists the accepted values.

```bash
# Start over, deleting the analysis directory (refused if it holds completed work):
hhemt run-experiment --experiment experiments/my_experiment --cluster uva --mode fresh

# Start over even though the directory holds completed work:
hhemt run-experiment --experiment experiments/my_experiment --cluster uva --mode fresh --override-wipe-nonempty
```

The wipe guard refuses `--mode fresh` with exit 2 when the analysis directory still holds completed simulations marked by `_status/c_run_*.flag` files, consolidated output, in-flight submission sentinels under `_status/_submitted` or `_status/_queued`, or an orchestrator sentinel under `_status/_orchestrator`. The refusal names the finding of the guard. `hhemt run --from-scratch` applies the same guard and raises `ConfigurationError`. Pass `--override-wipe-nonempty` to delete the work on purpose.

`--override-wipe-nonempty` differs from `--yes`. `--yes` accepts the descriptor-override table, described under [The override gate](#the-override-gate), and authorizes no deletion. `--override-wipe-nonempty` authorizes the deletion and accepts no override. To remove an analysis, use `hhemt delete`, as noted in [Add scenarios or events to a running sweep](forcing-reruns.md#add-scenarios-or-events-to-a-running-sweep).

Keep `--mode resume` and pass `--override-force-rerun` to re-run completed scenarios without deleting any file. [Forcing and suppressing re-runs](forcing-reruns.md#force-a-full-re-run-of-specific-scenarios) lists the accepted values and the stage floor.

```bash
# Re-run every completed scenario in place, no wipe:
hhemt run-experiment --experiment experiments/my_experiment --cluster uva --override-force-rerun all
```

## Waiting for completion

For an experiment dispatched through SLURM, `--wait` and `--no-wait` control whether the verb blocks until the workflow finishes. With neither flag, the verb waits inside an sbatch allocation and returns immediately on a login node. Inside an allocation `$SLURM_JOB_ID` is set, and the detached orchestrator would otherwise die with the allocation. A detached invocation exits `0` as soon as the workflow is submitted. The exit code reports the outcome of the workflow only when the verb waits.

## `${VAR}` placeholders in configs

The git-tracked system and analysis configs carry portable `${VAR}` placeholders such as `${DATA_DIR}/dem.tif` and `${SCRATCH_DIR}/system`. Export the referenced variables, typically from your submit script, before running. An **unset** variable fails fast with a `ConfigurationError` that names the placeholder.

??? note "Location of the expanded configs"
    `run-experiment` expands the placeholders against the environment before execution and materializes resolved copies in `$SCRATCH_DIR/resolved_configs/`. The location is a shared filesystem, so `batch_job` rules dispatched to other compute nodes can read the copies.

## The override gate

`experiment.yaml` is the single source of truth. A CLI flag that overrides a value the descriptor declares, currently `--hpc-system-config`, makes the verb print a side-by-side `descriptor (config)` and `CLI override` table and require confirmation.

```bash
# Non-interactive contexts (a submit script) must pass --yes to accept an override:
hhemt run-experiment --experiment experiments/my_experiment --cluster uva --yes
```

Without `--yes`, a non-interactive invocation that would override the descriptor **refuses** and never silently prefers the CLI. When the CLI adds no value beyond the descriptor, no confirmation is needed, and the common one-config path falls under this case. `--yes` accepts only the table. Deleting completed work with `--mode fresh` goes through a separate gate with a separate flag, described under [Run modes and the wipe guard](#run-modes-and-the-wipe-guard).

## Exit codes

`hhemt run-experiment` exits with the codes in the [CLI reference exit-code table](../reference/cli.md#exit-codes). The page also lists the cases that each code covers for this command.

## Verifying an experiment definition conforms

The descriptor model ships in the wheel, so an installed copy validates an experiment definition directly. Run the snippet with the interpreter that holds your `hhemt` install. Inside that environment `python3` resolves to that interpreter, and outside the environment no interpreter on `PATH` imports `hhemt`, so the snippet fails.

```bash
python3 -c "
import sys, yaml
from hhemt.config.experiment_definition import ExperimentConfig
ExperimentConfig.model_validate(yaml.safe_load(open(sys.argv[1] + '/experiment.yaml')))
print('OK')" experiments/my_experiment
```

From a repository checkout, the fuller checker also verifies that the declared `system_config` and `analysis_config` paths exist on disk and that `README.md` and `rerun.sh` are present.

```bash
scripts/hook_env.sh python scripts/check_experiment_structure.py experiments/my_experiment
```

Exit 0 means conformant. The wheel omits the `scripts/` directory, so the second form requires a clone and no `pip install`.
