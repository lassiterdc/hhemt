# Plan: Implement `local_gpus_for_workflow` Support in Snakemake Local Mode

## Context

`analysis_config` already defines:

- `local_cpu_cores_for_workflow`
- `local_gpus_for_workflow`

Today, local Snakemake execution effectively honors CPU limits but does **not** enforce a user-defined GPU budget for the workflow. This plan describes how to implement GPU-aware local workflow limits in a way that is consistent, testable, and backwards compatible.

---

## Goals

1. Respect `local_gpus_for_workflow` during local Snakemake runs.
2. Preserve existing behavior when `local_gpus_for_workflow` is unset.
3. Keep CPU/GPU limiting logic config-driven (no ad-hoc submit-time CLI passthrough).
4. Ensure local dry-runs and real runs both use the same resource limit semantics.

---

## Current Behavior (Gap)

- `run_snakemake_local` builds a Snakemake command for local mode.
- CPU control is now passed via `--cores` (when `local_cpu_cores_for_workflow > 1`).
- No equivalent `--resources gpu=<N>` is passed for local mode.
- Snakefile rules can request GPU resources (`resources: gpu=...`), but without a top-level global GPU budget, Snakemake cannot enforce an overall local GPU cap.

---

## Proposed Design

### 1) Treat local GPU budget as a global Snakemake resource

In `run_snakemake_local`:

- Read `self.cfg_analysis.local_gpus_for_workflow`.
- If it is an `int` and `> 0`, append:

```bash
--resources gpu=<local_gpus_for_workflow>
```

to the local Snakemake command.

### 2) Preserve backwards compatibility

- If `local_gpus_for_workflow` is `None` or `0`, do not append GPU resource flags.
- Existing CPU-only local workflows continue unchanged.
- Existing mixed-model workflows still run; GPU-limited behavior only activates when user opts in.

### 3) Keep profile simple; enforce limits via command line

- Continue generating local profile config as before.
- Keep runtime resource limit knobs (`--cores`, `--resources`) applied in command construction.
- This mirrors current single-job mode patterns and avoids duplicating resource logic across profile and CLI.

---

## Validation and Guardrails

### Config validation (optional but recommended)

Add a validator in `analysis_config`:

- `local_gpus_for_workflow` must be `>= 0` when provided.

This avoids invalid values like `-1` reaching workflow execution.

### Runtime assertions

In local run path, avoid hard failure when GPU value is unset.

- Assert type only when value is not `None`.
- Gracefully skip GPU flag when not specified.

---

## Code Touchpoints

1. `src/TRITON_SWMM_toolkit/workflow.py`
   - `run_snakemake_local`: append `--resources gpu=<N>` when configured.
   - Ensure dry-run path uses same command builder logic.

2. `src/TRITON_SWMM_toolkit/config.py` (optional but recommended)
   - Add validator for non-negative `local_gpus_for_workflow`.

3. Tests
   - Add/adjust tests to verify command construction includes/excludes GPU resource flags as expected.

---

## Test Plan

### Unit/behavioral tests (recommended)

1. **GPU flag included when configured**
   - Set `local_gpus_for_workflow=2`
   - Verify command includes `--resources gpu=2`.

2. **GPU flag omitted when unset**
   - Set `local_gpus_for_workflow=None`
   - Verify no `--resources gpu=...` is passed.

3. **GPU flag omitted when zero**
   - Set `local_gpus_for_workflow=0`
   - Verify no GPU resource flag is passed.

4. **Dry-run consistency**
   - In dry-run mode with GPU configured, ensure command still contains `--resources gpu=<N>`.

5. **Negative value validation** (if validator added)
   - `local_gpus_for_workflow=-1` should raise validation error.

---

## Risks / Edge Cases

1. **Rule resource naming mismatch**
   - Must use the same resource key (`gpu`) consistently in rule definitions and CLI (`--resources gpu=...`).

2. **Mixed CPU/GPU model workflows**
   - SWMM-only rules do not request GPU; they should continue to schedule normally.
   - GPU-requiring rules will be throttled by global GPU budget.

3. **Systems with no GPUs**
   - Users should leave `local_gpus_for_workflow` unset/0 to avoid artificial resource declarations.

---

## Agentic Implementation Checklist

- [ ] Add local GPU CLI resource injection in `run_snakemake_local`.
- [ ] Ensure dry-run and normal local runs share the same resource flag behavior.
- [ ] Add config validator for non-negative `local_gpus_for_workflow` (optional but preferred).
- [ ] Add/adjust tests for inclusion/omission of `--resources gpu=<N>`.
- [ ] Run targeted workflow tests.
- [ ] Update docs/changelog notes describing local GPU workflow control.

---

## Acceptance Criteria

1. Local workflow command includes `--resources gpu=<N>` when `local_gpus_for_workflow=N>0`.
2. Local workflow command omits GPU resource flags when config value is unset or zero.
3. Existing tests for local workflow behavior remain green.
4. New tests covering local GPU workflow limits pass.
