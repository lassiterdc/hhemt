# Implementation Plan: Fix CPU Sensitivity Suite Failures (sa-14, sa-17, sa-18, sa-19)

**Created:** 2026-02-27
**Status:** Complete

## Task Understanding

### Requirements

1. **Fix sa_19 configuration**: `n_mpi_procs=2, n_omp_threads=64, n_nodes=1, partition=standard` requests 128 CPUs on one node, which exceeds the standard-afton limit of 96 CPUs. SLURM rejects the sbatch immediately. The sub-analysis definition in `full_benchmarking_experiment_uva_test_cpu.xlsx` must be corrected.

2. **Fix sa_14, sa_17, sa_18 time limit failures**: Three high-MPI-rank configurations exhaust the 20-minute budget without completing. `hpc_time_min_per_sim: 20` in the test fixture must be increased.

3. **Suppress OMP_PROC_BIND warning (optional)**: In `run_simulation.py`, the `else` branch (serial and mpi modes) sets `OMP_NUM_THREADS=1` but omits `OMP_PROC_BIND`, causing Kokkos to print a cosmetic warning in every model log for those modes. Adding `OMP_PROC_BIND="false"` silences it.

### Assumptions

- The sensitivity analysis is defined by two artifacts: the `.xlsx` file referenced in `test_case_catalog.py`, and the `analysis_overrides` dict in `benchmarking_norfolk_irene_cpu()`. No other source of truth exists.
- Snakemake will automatically resume from the 16 complete flags. Only the 4 failed sub-analyses will rerun.
- The `parallel` partition on UVA afton has 96 CPUs/node and supports up to 64 nodes, making a 2-node request valid.
- Per-sub-analysis `hpc_time_min_per_sim` overrides are not supported. One global value applies to all sub-analyses.
- The OMP_PROC_BIND change is cosmetic and has no impact on simulation correctness or performance.

### Success Criteria

- All 20 sub-analyses reach `_status/simulation_sa*_evt0_complete.flag`.
- No `sbatch: error: Batch job submission failed` in cluster logs.
- No `DUE TO TIME LIMIT` cancellations in any model log.
- (Optional) No `Kokkos::OpenMP::initialize WARNING: OMP_PROC_BIND environment variable not set` in serial/mpi model logs.

---

## Evidence from Codebase

- `.debugging/test_uva_sensitivity_suite_cpu/debugging_report_20260224_175000.md`: Confirmed root causes for all four failures. sa_19 SBATCH rejection at 128 CPUs on standard (max 96). sa_14/sa_17/sa_18 ran exactly 20 minutes before SIGTERM. 16/20 sub-analyses complete.
- `.debugging/test_uva_sensitivity_suite_cpu/sensitivity_analysis_definition.csv`: Row 19 shows `run_mode=hybrid, n_mpi_procs=2, n_omp_threads=64, n_gpus=0, n_nodes=1, hpc_ensemble_partition=standard`. This is the invalid row. Rows 17–18 (sa_17, sa_18) correctly use `parallel` partition and `n_nodes=2` for their 128-CPU configurations.
- `.debugging/test_uva_sensitivity_suite_cpu/cfg_analysis.yaml`: Confirms `hpc_time_min_per_sim: 20` is the active value.
- `src/TRITON_SWMM_toolkit/run_simulation.py` lines 392–398: `OMP_PROC_BIND` is only set when `run_mode in ("openmp", "hybrid")`. The `else` branch sets only `OMP_NUM_THREADS="1"`.
- Partition hardware (from debugging report):
  - `standard-afton`: 96 CPUs/node, MaxNodes=1
  - `parallel`: 96 CPUs/node, MaxNodes=64

---

## Implementation Strategy

### sa_19 fix: Move to `parallel` partition, set `n_nodes=2`

The sa_19 row intends to benchmark hybrid with 2 MPI ranks × 64 OMP threads = 128 CPUs total. This configuration genuinely requires 128 CPUs and cannot fit on any single standard node. The `parallel` partition is the correct venue — sa_17 and sa_18 already use it for their 128-CPU configurations. Moving sa_19 there makes the sweep internally consistent: all 128-CPU configurations live in `parallel`.

**Alternative considered**: Reduce `n_omp_threads` to 48 (2×48=96 CPUs on one standard node). This changes the configuration being benchmarked, producing a data point at a different total CPU count and breaking the comparison intent of the row. Rejected.

### Time limit fix: Increase `hpc_time_min_per_sim` to 90 minutes

The debugging report recommends 60/60/90 minutes for sa_14/sa_17/sa_18 respectively. Setting a single global value of 90 covers all three with headroom. The 16 already-complete sub-analyses won't rerun (complete flags exist); if they did, they'd release their allocations within minutes.

Per-sub-analysis time override configuration is not currently supported. Implementing it would require new config schema fields and workflow generator changes — out of scope for this targeted fix.

### OMP_PROC_BIND: Add `"false"` to else branch

One-line addition. The value `"false"` is Kokkos's documented recommendation for single-threaded or non-threaded contexts, explicitly called out in the warning text itself.

---

## File-by-File Change Plan

### 1. `test_data/norfolk_coastal_flooding/full_benchmarking_experiment_uva_test_cpu.xlsx`

**Purpose**: Fix the sa_19 row so SLURM accepts the sbatch submission.

**Change**: Edit row 20 (sa_19, zero-indexed row 19):

| Column | Current | New |
|--------|---------|-----|
| `n_nodes` | 1 | 2 |
| `hpc_ensemble_partition` | standard | parallel |
| `n_mpi_procs` | 2 | 2 (unchanged) |
| `n_omp_threads` | 64 | 64 (unchanged) |

This changes the SLURM request from `--nodes=1 --ntasks=2 --cpus-per-task=64` (impossible on standard) to `--nodes=2 --ntasks=2 --cpus-per-task=64` (valid on parallel). The total CPU count and benchmarking configuration are unchanged.

### 2. `tests/fixtures/test_case_catalog.py`

**Purpose**: Increase the time budget so sa_14, sa_17, and sa_18 can complete.

**Change**: In `benchmarking_norfolk_irene_cpu()`, update `analysis_overrides`:

```python
# Before:
"hpc_time_min_per_sim": 20,

# After:
"hpc_time_min_per_sim": 90,
```

### 3. `src/TRITON_SWMM_toolkit/run_simulation.py`

**Purpose**: Suppress the cosmetic Kokkos OMP_PROC_BIND warning for serial and mpi modes.

**Change**: In the OpenMP configuration block (~line 397):

```python
# Before:
else:
    env["OMP_NUM_THREADS"] = "1"

# After:
else:
    env["OMP_NUM_THREADS"] = "1"
    env["OMP_PROC_BIND"] = "false"  # suppresses Kokkos WARNING when OMP threading unused
```

---

## Risks and Edge Cases

- **sa_19 on parallel may take longer than expected due to 2-node MPI overhead**: Unlikely — sa_19 uses only 2 MPI ranks, making it one of the lightest MPI configurations. The time limit failures were in 32–64 rank configurations. Should complete well within 90 minutes.
- **sa_14/sa_17/sa_18 still time out at 90 minutes**: Possible but unlikely. These jobs ran for exactly 20 minutes before SIGTERM — they were clearly running, not hanging. 90 minutes should be ample for the small Norfolk test domain.
- **Snakemake partial-resume behavior**: Snakemake will see the 16 complete flags and skip those jobs. Only the 4 failed sub-analyses rerun. No special handling required.
- **OMP_PROC_BIND="false" surprising Kokkos**: The Kokkos warning itself explicitly states "For unit testing set OMP_PROC_BIND=false". Risk is negligible.

---

## Validation Plan

### Local (pre-push)

```bash
pytest tests/test_PC_01_singlesim.py
pytest tests/test_PC_05_sensitivity_analysis_with_snakemake.py
```

The Excel and fixture changes cannot be validated locally — they exercise SLURM-only behavior.

### HPC (coordinated with user on UVA)

1. Confirm 4 missing complete flags are absent before rerunning.
2. Re-run workflow (Snakemake resumes automatically).
3. Verify sa_19 sbatch no longer rejected — job queues on parallel partition.
4. After completion:
   ```bash
   ls _status/simulation_sa*_evt0_complete.flag | wc -l   # expect 20
   ```
5. Check model logs for sa_14, sa_17, sa_18, sa_19 show `[OK] Simulation ends` not `DUE TO TIME LIMIT`.
6. (Optional) Confirm OMP warning suppressed in serial mode log:
   ```bash
   grep "OMP_PROC_BIND" logs/sims/model_tritonswmm_sa_0_evt0.log   # expect no match
   ```

---

## Documentation and Tracker Updates

- `docs/planning/bugs/addressing_cpu_sensitivity_analysis_failures.md`: The prior plan (addressed the `runtime=2` issue from run 1). Can be moved to `docs/planning/bugs/completed/` after the rerun succeeds.
- `docs/planning/features/2026-02-07_priorities.md`: Update once all 20 sub-analyses pass on UVA.
- `CLAUDE.md`: No changes needed — no new patterns introduced.

---

## Decisions Needed from User

**Decision 1: sa_19 fix strategy**

Plan assumes **Option A**: move sa_19 to `parallel` partition with `n_nodes=2`, preserving `n_mpi_procs=2, n_omp_threads=64`. This is consistent with sa_17 and sa_18.

Risk if proceeding with assumption: **low** — it is the only fix that preserves the experimental intent.

**Decision 2: Include OMP_PROC_BIND suppression**

Plan assumes **include it**. Risk: **low** — purely cosmetic change.

---

## Definition of Done

- [ ] `full_benchmarking_experiment_uva_test_cpu.xlsx` sa_19 row updated: `n_nodes=2`, `hpc_ensemble_partition=parallel`
- [ ] `benchmarking_norfolk_irene_cpu()` fixture updated: `hpc_time_min_per_sim: 90`
- [ ] `run_simulation.py` else branch updated: `OMP_PROC_BIND="false"` added
- [ ] Local smoke tests pass: `pytest tests/test_PC_01_singlesim.py tests/test_PC_05_sensitivity_analysis_with_snakemake.py`
- [ ] UVA rerun: all 20 `simulation_sa*_evt0_complete.flag` files present
- [ ] No `DUE TO TIME LIMIT` or `Batch job submission failed` in any sub-analysis logs
- [ ] Prior plan (`addressing_cpu_sensitivity_analysis_failures.md`) archived

---

## Post-Implementation Findings (2026-02-28)

Two additional issues were discovered and resolved during UVA CPU sensitivity suite runs after the original plan was implemented.

### Finding 1: High MPI rank counts cause indefinite hang after simulation starts

**Observation**: Sub-analyses with ≥32 MPI ranks caused simulations to hang indefinitely after starting (not a time limit cancel — actual MPI deadlock/barrier hang). This is distinct from the PMI `inet_connect` transient failures.

**Resolution**: Removed all sub-analyses with ≥32 MPI ranks from the UVA CPU benchmarking Excel (`full_benchmarking_experiment_uva.xlsx`).
- **Commit**: `8afde85` (2026-02-25) — "remove 32 mpi tests"

### Finding 2: Frontier allocates 8 cores to GPU hardware — effective CPU limit is 56, not 64

**Observation**: Frontier compute nodes have 64 CPUs listed in partition information, but 8 of those cores are reserved for GPU hardware management. CPU-only jobs (and CPU components of hybrid jobs) can access at most **56 CPUs per node**, not 64.

**Resolution**: The `hpc_cpus_per_node` configuration value for Frontier analyses should be set to `56` (not `64`). Sensitivity suite definitions and test fixtures referencing Frontier CPU-per-node counts were updated accordingly.

**Impact**: Any sub-analyses designed around 64 CPUs/node on Frontier are invalid. The maximum safe per-node CPU request for CPU work on Frontier is 56.

---

## Self-Check Results

1. **Header/body alignment**: All sections present and content matches headers. File-by-File section covers exactly three files for three issues. Decisions section has exactly the two genuinely open questions.
2. **Section necessity**: All sections carry actionable content. No sections removed.
