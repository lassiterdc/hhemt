# Implementation Ideas

A persistent record of implementation ideas, enhancement proposals, and future directions for this project. Entries are not dated planning docs — they persist until explicitly pursued (create a dated feature doc) or dropped.

---

### Group: HPC Resilience / Unattended Recovery

## Idea 1: Implement retries for batch_job and 1_big_job_approach

**Surfaced**: 2026-03-03
**Priority**: Medium
**Description**: When analyses fail due to HPC timeouts, the current recovery workflow is costly: Globus transfer, full `/debug-hpc-analysis` skill run, manual diagnosis, and manual re-submission. Implementing automatic retries (configurable via a config field and an `analysis.run()` override argument, similar to the existing HPC nodes override) would eliminate most of this overhead for the common timeout case.
**Approach notes**: Two known sticking points: (1) The `1_big_job` approach often leaves Snakemake lock files behind after a timeout, requiring a currently-manual unlock step — this will need to be automated for unattended re-submission. (2) For `1_big_job` mode, the HPC nodes override on retry should be computed from the number of remaining incomplete simulations to avoid wasting node allocations. The `batch_job` mode is expected to be simpler. Biggest risk: retrying blindly when the failure cause is something other than a timeout (e.g., a bug or bad config), which would waste HPC allocation.
**Related ideas**: none

## Idea 2: Auto-unlock Snakemake when no jobs are in queue

**Surfaced**: 2026-03-03
**Priority**: High
**Description**: Loading an analysis on Frontier can take several minutes, during which the existing lock detection prompts the user to hit `y` and Enter before proceeding. When `squeue -u $USER` returns empty, there is no risk of unlocking a live job — the manual gate is unnecessary. Auto-unlocking in this case eliminates the wait.
**Approach notes**: none
**Related ideas**: Idea 1 (automatic retries — this is a prerequisite for unattended retry); `features/completed/2026-02-28_snakemake-lock-detection-and-auto-unlock.md` (existing lock detection infrastructure, direct extension point); `bugs/completed/2026-02-13_fix_unlock_snakemake_if_interrupted.md` (earlier context)
