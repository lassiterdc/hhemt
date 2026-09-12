# Running an experiment bundle

`hhemt run-experiment` runs a **self-describing experiment bundle**: a directory whose
`experiment.yaml` declares everything the run needs: the system and analysis configs, the
input datasets, the per-cluster HPC profile, the container, and a resolvable toolkit pin. The
descriptor is the *single config*: there are no positional config arguments to line up, and any
CLI flag that would override a descriptor-declared value must be confirmed first.

## The bundle layout

A conformant bundle (validated against the `ExperimentConfig` descriptor model, see
[Verifying a bundle conforms](#verifying-a-bundle-conforms)) is a directory containing at least:

```
experiments/my_experiment/
├── experiment.yaml          # the descriptor (schema: hhemt.config.experiment_bundle.ExperimentConfig)
├── README.md                # runbook (required)
├── rerun.sh                 # re-run driver (required)
└── configs/
    ├── system_config_uva.yaml
    └── analysis_config_uva.yaml
```

The descriptor carries no identity key: the bundle is named by its directory, and the
analysis identity comes from the analysis config. A minimal `experiment.yaml`:

```yaml
description: One-line description.
system_config: configs/system_config_uva.yaml    # bundle-relative
analysis_config: configs/analysis_config_uva.yaml # bundle-relative
hpc_system_config:
  uva: hpc/hpc_system_config_uva.yaml   # relative to $HHEMT_DEPLOYMENT_CONFIG, or to the directory two levels above this bundle
inputs:
  - name: weather
    local_path: "${HHEMT_DATA_ROOT}/weather/forcing.nc"   # ${VAR}-templated, never a literal operator path
    deposit: true                                          # these bytes are part of the publish payload
    destinations:
      uva: /scratch/$USER/my_experiment/weather/forcing.nc # where the provisioning stages it on-cluster
toolkit_pin:
  version: "0.1.0"                      # PyPI version: the durable, installable identifier
container:
  def_recipe: containers/uva-cuda.def    # bundle-relative, or ${VAR}-rooted for a shared recipe
  sha256_source: ro-crate              # the SIF digest's authoritative home is the RO-Crate
```

See the [config-filling](config-filling.md) and [HPC-profile setup](hpc-profile-setup.md)
guides for the system/analysis and `hpc_system_config` contents.

`def_recipe` declares its own root. A bare relative value such as `containers/uva-cuda.def`
is resolved against the bundle directory; a `${VAR}`-rooted value such as
`${HHEMT_TOOLKIT}/containers/uva-cuda.def` names one recipe shared across several
experiments. An absolute path, a `~`-rooted path, or an unbraced `$VAR` is refused when the
descriptor loads, because such a value is rooted on one operator's machine and cannot be
reproduced by anyone else.

## Run it

```bash
# Plan only, building the DAG and writing nothing:
hhemt run-experiment --bundle experiments/my_experiment --cluster uva --dry-run

# Execute:
hhemt run-experiment --bundle experiments/my_experiment --cluster uva
```

`--cluster` selects which `hpc_system_config[<cluster>]` and which per-cluster `destinations`
apply. The verb loads and validates `experiment.yaml`, resolves the HPC profile, then hands the
two configs to the toolkit.

## Run modes and the wipe guard

`--mode` selects what happens to an analysis directory that already holds work. The default
is `resume`: a second invocation, or a SLURM requeue, picks up where the last one left off,
and completed simulations are never deleted under `resume`.

| `--mode` | Behaviour |
|----------|-----------|
| `resume` (default) | Continue from the last checkpoint; completed simulations are kept. |
| `fresh` | Delete the whole analysis directory first, then rebuild everything. Guarded; see below. |
| `overwrite` | Accepted; currently behaves as `resume`, continuing from the last checkpoint. |

```bash
# Start over, deleting the analysis directory (refused if it holds completed work):
hhemt run-experiment --bundle experiments/my_experiment --cluster uva --mode fresh

# Start over even though the directory holds completed work:
hhemt run-experiment --bundle experiments/my_experiment --cluster uva --mode fresh --override-wipe-nonempty
```

The wipe guard refuses `--mode fresh` (exit 2) when the analysis directory still holds
completed simulations, consolidated output, in-flight submission sentinels, or an orchestrator
sentinel; the refusal names what it found. Pass `--override-wipe-nonempty` to delete that work
deliberately. This flag is not `--yes`: `--yes` accepts the descriptor-override table (see
[The override gate](#the-override-gate)) and authorizes no deletion, and
`--override-wipe-nonempty` authorizes the deletion and accepts no override. To remove an
analysis rather than re-run it, use `hhemt delete`, which carries its own confirmation.

## Waiting for completion

For a SLURM-dispatched run, `--wait` / `--no-wait` control whether the verb blocks until the
workflow finishes. With neither flag given, the verb waits when running inside an sbatch
allocation (`$SLURM_JOB_ID` is set, so the detached orchestrator would otherwise die with the
allocation) and returns immediately on a login node. A detached invocation exits `0` as soon
as the workflow is submitted; the exit code reports the workflow's outcome only when the verb
waits.

## `${VAR}` placeholders in configs

The git-tracked system/analysis configs carry portable `${VAR}` placeholders (e.g.
`${DATA_DIR}/dem.tif`, `${SCRATCH_DIR}/system`). Export the referenced variables
(typically from your submit script) before running: an **unset** variable fails fast with
a `ConfigurationError` naming the placeholder.

??? note "Where the expanded configs are written, and why there"
    `run-experiment` expands the placeholders against the environment before the run,
    materializing resolved copies to `$SCRATCH_DIR/resolved_configs/`. That location is
    a shared filesystem, so `batch_job` rules dispatched to other compute nodes can read
    them.

## The override gate

`experiment.yaml` is meant to be the single source of truth. If you pass a CLI flag that
overrides a value the descriptor already declares (currently `--hpc-system-config`), the verb
prints a side-by-side `descriptor (config)` vs `CLI override` table and requires confirmation:

```bash
# Non-interactive contexts (a submit script) must pass --yes to accept an override:
hhemt run-experiment --bundle experiments/my_experiment --cluster uva --yes
```

Without `--yes`, a non-interactive invocation that would override the descriptor **refuses**
rather than silently preferring the CLI. When the CLI adds nothing the descriptor does not
already say, no confirmation is needed: that is the common one-config path. `--yes` accepts
only this table. Deleting completed work with `--mode fresh` goes through a separate gate with
its own flag; see [Run modes and the wipe guard](#run-modes-and-the-wipe-guard).

## Exit codes

| Code | Meaning |
|------|---------|
| 0 | success (or `--dry-run` planned cleanly) |
| 1 | the run finished without succeeding: Snakemake exited non-zero, `sbatch` refused the submission, or a rule failed permanently; the console output printed just before the exit names the cause |
| 2 | configuration error (bad `experiment.yaml`, unset `${VAR}`, missing/placeholder `default_account` or `container.sif_path`, declined override gate, `--mode fresh` refused by the wipe guard) |
| 5 | workflow / processing / simulation error |
| 10 | unexpected error |

## Verifying a bundle conforms

The descriptor model ships in the wheel, so an installed copy can validate a bundle
directly. Run this with the interpreter you installed `hhemt` into: inside that
environment, `python3` resolves to it; from outside one, no interpreter on `PATH` will
import `hhemt` and the snippet cannot work.

```bash
python3 -c "
import sys, yaml
from hhemt.config.experiment_bundle import ExperimentConfig
ExperimentConfig.model_validate(yaml.safe_load(open(sys.argv[1] + '/experiment.yaml')))
print('OK')" experiments/my_experiment
```

Working from a repo checkout, the fuller checker additionally verifies that the declared
`system_config`/`analysis_config` paths exist on disk and that `README.md` + `rerun.sh` are
present:

```bash
uv run --locked python scripts/check_experiment_structure.py experiments/my_experiment
```

Exit 0 = conformant. Note that `scripts/` is not distributed in the wheel, so this second form
requires a clone rather than a `pip install`.
