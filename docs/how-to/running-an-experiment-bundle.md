# Running an experiment bundle

`hhemt run-experiment` runs a **self-describing experiment bundle** — a directory whose
`experiment.yaml` declares everything the run needs: the system and analysis configs, the
input datasets, the per-cluster HPC profile, the container, and a resolvable toolkit pin. The
descriptor is the *single config*: there are no positional config arguments to line up, and any
CLI flag that would override a descriptor-declared value must be confirmed first.

## The bundle layout

A conformant bundle (validated against the `ExperimentBundle` descriptor model — see
[Verifying a bundle conforms](#verifying-a-bundle-conforms)) is a directory containing at least:

```
experiments/my_experiment/
├── experiment.yaml          # the descriptor (schema: hhemt.config.experiment_bundle.ExperimentBundle)
├── README.md                # runbook (required)
├── rerun.sh                 # re-run driver (required)
└── configs/
    ├── system_config_uva.yaml
    └── analysis_config_uva.yaml
```

A minimal `experiment.yaml`:

```yaml
experiment_id: my_experiment          # must equal the directory name
description: One-line description.
system_config: configs/system_config_uva.yaml    # bundle-relative
analysis_config: configs/analysis_config_uva.yaml # bundle-relative
hpc_system_config:
  uva: hpc/hpc_system_config_uva.yaml   # estate-relative (resolved against $HHEMT_DEPLOYMENT_CONFIG or the estate root)
inputs:
  - name: weather
    local_path: "${HHEMT_DATA_ROOT}/weather/forcing.nc"   # ${VAR}-templated — never a literal operator path
    deposit: true                                          # these bytes are part of the publish payload
    destinations:
      uva: /scratch/$USER/my_experiment/weather/forcing.nc # where the provisioning stages it on-cluster
toolkit_pin:
  version: "0.1.0"                      # PyPI version — the durable, installable identifier
container:
  def_recipe: containers/uva-cuda.def
  sha256_source: ro-crate              # the SIF digest's authoritative home is the RO-Crate
```

See the [config-filling](config-filling.md) and [HPC-profile setup](hpc-profile-setup.md)
guides for the system/analysis and `hpc_system_config` contents.

## Run it

```bash
# Plan only — build the DAG, write nothing:
hhemt run-experiment --bundle experiments/my_experiment --cluster uva --dry-run

# Execute:
hhemt run-experiment --bundle experiments/my_experiment --cluster uva
```

`--cluster` selects which `hpc_system_config[<cluster>]` and which per-cluster `destinations`
apply. The verb loads and validates `experiment.yaml`, resolves the HPC profile, then hands the
two configs to the toolkit.

## `${VAR}` placeholders in configs

The git-tracked system/analysis configs carry portable `${VAR}` placeholders (e.g.
`${DATA_DIR}/dem.tif`, `${SCRATCH_DIR}/system`). `run-experiment` expands them against the
environment before the run, materializing resolved copies to `$SCRATCH_DIR/resolved_configs/`
(a shared filesystem, so `batch_job` rules dispatched to other compute nodes can read them). An
**unset** variable fails fast with a `ConfigurationError` naming the placeholder — export the
referenced variables (typically from your submit script) before running.

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
already say, no confirmation is needed — that is the common one-config path.

## Exit codes

| Code | Meaning |
|------|---------|
| 0 | success (or `--dry-run` planned cleanly) |
| 2 | configuration error (bad `experiment.yaml`, unset `${VAR}`, missing/placeholder `default_account` or `container.sif_path`, declined override gate) |
| 5 | workflow / processing / simulation error |
| 10 | unexpected error |

## Verifying a bundle conforms

The descriptor model ships in the wheel, so an installed copy can validate a bundle directly:

```bash
python -c "
import sys, yaml
from hhemt.config.experiment_bundle import ExperimentBundle
ExperimentBundle.model_validate(yaml.safe_load(open(sys.argv[1] + '/experiment.yaml')))
print('OK')" experiments/my_experiment
```

Working from a repo checkout, the fuller checker additionally verifies that `experiment_id`
matches the directory name, that the declared `system_config`/`analysis_config` paths exist,
and that `README.md` + `rerun.sh` are present:

```bash
python scripts/check_experiment_structure.py experiments/my_experiment
```

Exit 0 = conformant. Note that `scripts/` is not distributed in the wheel, so this second form
requires a clone rather than a `pip install`.
