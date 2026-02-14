# CLI Vision (Snakemake-First)

## Purpose

Define a clear, stable command-line vision for TRITON-SWMM Toolkit where
Snakemake workflows are the primary user-facing CLI mechanism.

## North Star

Provide **one primary command surface** that:

1. accepts system + analysis configuration paths,
2. controls workflow behavior through explicit arguments,
3. supports reproducible and resumable execution, and
4. maps cleanly to the same orchestration used by the Python API.

The single command should support three primary usage profiles:

- `production` (normal project execution)
- `testcase` (quick verification that installs/environment/workflow work)
- `case-study` (curated end-to-end starter runs)

## User Personas

### 1) Workflow Operator (primary)
- Runs scenarios from terminal/HPC scheduler.
- Cares about reproducibility, resumability, and transparent logs.

### 2) Applied Researcher
- Uses CLI for baseline runs; switches to notebooks for deeper analysis.
- Expects CLI behavior to match Python API behavior.

### 3) Pipeline Integrator
- Embeds toolkit commands in larger automation systems.
- Needs predictable argument contracts and non-zero exit codes on failure.

## CLI Principles

1. **Single command, explicit arguments**
   - Keep command surface small and discoverable.

2. **Resumable by default**
   - Prefer log/state-aware skip semantics over blind overwrite.

3. **Fail fast with actionable errors**
   - Validate config files, paths, and mode compatibility before heavy work.

4. **Deterministic behavior**
   - Establish and document clear precedence among defaults/config/CLI args.

5. **Snakemake as execution engine**
   - CLI acts as a thin wrapper around workflow targets/rules and options.

6. **Fast verification paths for new users**
   - First-class testcase/case-study execution modes should help users confirm
     installation and cluster/runtime compatibility quickly.

7. **HPC settings are user-overridable**
   - Testcases/case studies must not lock users into hardcoded platform values.
   - User-provided HPC settings should inherit into these profiles through a
     dedicated profile config.

## Primary Outcome

Users should be able to run most work with one command pattern and a known set
of arguments, while still having the Python API available for interactive and
custom workflows.
