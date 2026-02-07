# `tests_and_case_studies.yaml` Specification (Draft)

## Purpose

Define how testcase and case-study runs inherit and override HPC/runtime settings
without relying on hardcoded values in curated examples.

## File Role

`tests_and_case_studies.yaml` acts as a profile catalog for:

- discoverable testcase definitions
- discoverable case-study definitions
- shared defaults for HPC/runtime parameters
- profile-specific overrides

## Draft Schema

```yaml
version: 1

defaults:
  hpc:
    platform_config: null
    scheduler: slurm
    partition: null
    account: null
    qos: null
    nodes: 1
    ntasks_per_node: 1
    cpus_per_task: 1
    gpus_per_node: 0
    walltime: "01:00:00"
  workflow:
    jobs: 1
    which: both
    model: auto

testcases:
  norfolk_smoke:
    description: "Fast install/runtime verification"
    case_root: test_data/norfolk_coastal_flooding/tests/smoke
    system_config: test_data/norfolk_coastal_flooding/template_system_config.yaml
    analysis_config: test_data/norfolk_coastal_flooding/template_analysis_config.yaml
    hpc:
      partition: debug
      walltime: "00:20:00"
    workflow:
      jobs: 1
      event_ilocs: [0]

case_studies:
  norfolk_coastal_flooding:
    description: "Reference case-study workflow"
    case_root: test_data/norfolk_coastal_flooding
    system_config: test_data/norfolk_coastal_flooding/template_system_config.yaml
    analysis_config: test_data/norfolk_coastal_flooding/template_analysis_config.yaml
    hpc:
      nodes: 2
      ntasks_per_node: 4
      cpus_per_task: 2
      walltime: "02:00:00"
```

## Merge and Precedence Rules

Resolved values should be computed in this order (highest first):

1. CLI explicit arguments
2. Selected testcase/case-study entry in `tests_and_case_studies.yaml`
3. `defaults` section in `tests_and_case_studies.yaml`
4. analysis config values
5. system config values
6. toolkit defaults

### Notes
- `null` means “unspecified” and should not overwrite lower-priority values.
- Merge should be **field-level** (deep merge for `hpc` and `workflow`).
- Final resolved config should be emitted in logs for transparency.

## CLI Mapping

- `--profile testcase --testcase <name>` → `testcases.<name>`
- `--profile case-study --case-study <name>` → `case_studies.<name>`
- `--tests-case-config PATH` selects alternate catalog file.

## Validation Rules

1. `version` required and supported.
2. Selected profile entry must exist.
3. Profile entries must provide or resolve `system_config` and `analysis_config`.
4. Numeric HPC fields must be positive integers.
5. `walltime` must match `HH:MM:SS`.

## Operational Guarantees

- Users can run curated assets without editing source-controlled hardcoded HPC settings.
- Cluster-specific overrides are possible at invocation time.
- Testcase/case-study runs remain reproducible via profile snapshots.
