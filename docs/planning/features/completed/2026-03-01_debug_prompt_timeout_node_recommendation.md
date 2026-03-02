# Feature: Timeout-Aware Node Recommendation in HPC Debugging Skill

**Written**: 2026-03-01
**Status**: Completed

---

## Summary

When the `/debug-hpc-analysis` skill diagnoses a `1_job_many_srun_tasks` run where the **only** failure mode is SLURM time limit cancellation (no errors, all active sims were running), the agent should:

1. **Ask the user** whether there is a partition time limit (e.g., Frontier `batch` partition = 2-hour max).
2. **Recommend an `override_hpc_total_nodes` value** for the re-run, sized to avoid over-allocating nodes for the remaining incomplete simulations.

---

## Motivation

The `frontier_sensitivity_suite` debugging session (2026-03-01) surfaced this gap: after a time-limit cancellation, the user needs to re-run with only the 7 incomplete (low-parallelism) sims. Those sims need far fewer nodes than the original 50-node allocation. Without this guidance, the user either:
- wastes 50 nodes running 7 serial/single-GPU tasks, or
- manually calculates the right node count themselves.

---

## Desired Behavior

### Trigger condition

All of the following must be true:
- `multi_sim_run_method == "1_job_many_srun_tasks"`
- All failures are time-limit cancellations (no Python errors, no SLURM resource errors)
- At least one simulation is incomplete

### Agent actions

**Step 1 — Ask about time limits:**
> "Is there a per-job wall time limit on this partition (e.g., 2 hours on Frontier `batch`)? If so, what is it?"

**Step 2 — Recommend node count for re-run:**

For each incomplete simulation, determine its resource requirements from `sensitivity_analysis_definition.csv` (or `cfg_analysis.yaml` for regular analyses):
- `n_mpi_procs` × `n_nodes` gives the node requirement per sim
- For GPU runs: `ceil(n_gpus / hpc_gpus_per_node)` gives nodes needed

Take the **maximum** of all incomplete sims' node requirements as the minimum needed. Add a small buffer (e.g., ×1.5, rounded up) if there are multiple incomplete sims that can run concurrently. Cap at the total nodes that could actually be useful (i.e., number of incomplete sims × per-sim node requirement).

Present as:
> "For the N incomplete sub-analyses, the maximum per-sim node requirement is X. Recommend `override_hpc_total_nodes=Y`."

**For regular (non-sensitivity) analyses:**
- Simply count incomplete simulations and recommend enough nodes to run them all concurrently (each sim uses `n_nodes` from config).

---

## Implementation Notes

- This logic lives in the `/debug-hpc-analysis` skill (`~/dev/claude-workspace/projects/TRITON-SWMM_toolkit/skills/debug-hpc-analysis/SKILL.md`) inline — there is no separate `.prompts/debugging_hpc_analysis.md` file. Not in Python source code.
- The recommendation must reference the new `override_hpc_total_nodes` parameter (requires that feature to be implemented first — see `docs/planning/features/2026-03-01_override_hpc_total_nodes.md`).
- The agent should make a concrete recommendation, not just explain the formula.
- If the sensitivity definition CSV is not available (excluded from transfer), the agent should note that it cannot make a node recommendation and ask the user to provide the resource requirements manually.

---

## Dependency

Requires `override_hpc_total_nodes` feature (`docs/planning/features/2026-03-01_override_hpc_total_nodes.md`) to be implemented and merged before this prompt update makes sense to deploy.

---

## Definition of Done

- [x] `/debug-hpc-analysis` SKILL.md updated with time-limit-specific branch in Step 7 (Root Cause) and Step 8 (Recommended Fixes)
- [x] Agent asks about partition time limit when all failures are time-limit cancellations
- [x] Agent computes and recommends `override_hpc_total_nodes` value with explanation
- [x] Recommendation covers both sensitivity analyses (per-subanalysis resource lookup) and regular analyses (per-sim node count × incomplete count)
