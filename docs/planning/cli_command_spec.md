# Single-Command CLI Specification (Snakemake Primary)

## Goal

Specify the canonical CLI contract for running TRITON-SWMM Toolkit workflows via
a single Snakemake-backed command.

---

## Canonical Command

```bash
triton-swmm run \
  --profile production \
  --system-config /path/to/system_config.yml \
  --analysis-config /path/to/analysis_config.yml \
  [options]
```

> Note: command name can be adjusted at implementation time (`tritonswmm`,
> `triton-swmm`, or package-provided entrypoint), but behavior should remain the
> same.

---

## Required Arguments

- `--profile {production,testcase,case-study}`
  - `production`: standard project workflow execution
  - `testcase`: quick environment and workflow verification
  - `case-study`: curated example execution path

- `--system-config PATH`
  - Path to system configuration file.
  - Must exist and be readable.

- `--analysis-config PATH`
  - Path to analysis/workflow configuration file.
  - Must exist and be readable.

---

## Optional Arguments (v1)

### Execution Control

- `--from-scratch`
  - Clear run artifacts/state required for a fresh execution.
  - Mutually exclusive with `--resume`.

- `--resume` (default behavior)
  - Continue from completed state using logs/checkpoints.

- `--overwrite`
  - Recreate processed outputs even if completion logs indicate success.

- `--dry-run`
  - Validate configs and render intended workflow actions without execution.

### Model/Processing Scope

- `--model {auto,triton,swmm,tritonswmm}`
  - `auto` uses enabled model toggles from configuration.

- `--which {TRITON,SWMM,both}`
  - Processing scope, typically passed through to output processing stage.

### Scenario/Event Selection

- `--event-ilocs CSV`
  - Example: `--event-ilocs 0,1,2,10`

- `--event-range START:END`
  - Example: `--event-range 0:100`
  - Inclusive/exclusive semantics should be documented explicitly in implementation.

### Testcase / Case-Study Selection

- `--testcase NAME`
  - Required when `--profile testcase` unless default testcase is configured.

- `--case-study NAME`
  - Required when `--profile case-study` unless default case study is configured.

- `--tests-case-config PATH`
  - Optional path to `tests_and_case_studies.yaml` containing profile catalog and
    HPC inheritance defaults.
  - If omitted, use toolkit default location.

- `--list-testcases`
  - Print available testcase entries and exit.

- `--list-case-studies`
  - Print available case-study entries and exit.

### HPC Override Arguments

Representative examples (exact field names may evolve with implementation):

- `--platform-config NAME`
- `--partition NAME`
- `--account NAME`
- `--qos NAME`
- `--nodes INT`
- `--ntasks-per-node INT`
- `--cpus-per-task INT`
- `--gpus-per-node INT`
- `--walltime HH:MM:SS`

These values should override any hardcoded testcase/case-study HPC defaults.

### Runtime / Workflow Engine

- `--jobs INT`
  - Parallel jobs for workflow execution.

- `--workflow-target TARGET`
  - Optional explicit Snakemake target/rule group (advanced users).

- `--snakemake-arg TEXT` (repeatable)
  - Pass-through for advanced Snakemake flags not yet first-class in toolkit CLI.

### Tool Provisioning

- `--redownload {none,triton,swmm,all}`
  - Optional pre-run bootstrap behavior for external tool binaries.

### Logging & UX

- `--verbose`
- `--quiet`
- `--log-level {DEBUG,INFO,WARNING,ERROR}`

---

## Validation Rules (v1)

1. Config paths are required and must exist.
2. `--from-scratch` and `--resume` are mutually exclusive.
3. `--event-ilocs` and `--event-range` are mutually exclusive unless an explicit
   merge policy is implemented.
4. `--which` must be compatible with resolved model mode.
5. `--redownload` is allowed in all modes but should log exactly what was changed.
6. `--profile testcase` must resolve a testcase selection.
7. `--profile case-study` must resolve a case-study selection.
8. `--list-testcases` and `--list-case-studies` should be no-run informational
   actions.

---

## Precedence Rules

CLI behavior should follow this precedence (highest first):

1. Explicit CLI arguments
2. Selected profile in `tests_and_case_studies.yaml` (for testcase/case-study runs)
3. Analysis config values
4. System config values
5. Internal defaults

Any resolved value should be visible in startup logs for transparency.

---

## Exit Code Guidelines

- `0`: success
- `2`: argument/config validation errors
- `3`: workflow planning/build errors
- `4`: simulation execution failure
- `5`: output processing/summarization failure
- `10+`: unexpected internal errors

---

## Example Commands

### Baseline reproducible run
```bash
triton-swmm run \
  --profile production \
  --system-config ./config/system.yml \
  --analysis-config ./config/analysis.yml
```

### Fresh run with selected events and explicit jobs
```bash
triton-swmm run \
  --profile production \
  --system-config ./config/system.yml \
  --analysis-config ./config/analysis.yml \
  --from-scratch \
  --event-ilocs 0,1,2 \
  --jobs 8
```

### Redownload SWMM binary before execution
```bash
triton-swmm run \
  --profile production \
  --system-config ./config/system.yml \
  --analysis-config ./config/analysis.yml \
  --redownload swmm
```

### Run a testcase with user HPC override
```bash
triton-swmm run \
  --profile testcase \
  --testcase norfolk_smoke \
  --system-config ./config/system.yml \
  --analysis-config ./config/analysis.yml \
  --partition debug \
  --nodes 1 \
  --walltime 00:20:00
```

### Run a case study with profile catalog file
```bash
triton-swmm run \
  --profile case-study \
  --case-study norfolk_coastal_flooding \
  --tests-case-config ./config/tests_and_case_studies.yaml \
  --system-config ./config/system.yml \
  --analysis-config ./config/analysis.yml
```
