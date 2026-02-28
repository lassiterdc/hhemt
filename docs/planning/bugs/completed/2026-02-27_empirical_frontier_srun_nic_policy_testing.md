# Empirical Testing: Frontier srun NIC Policy & Node Distribution

**Date**: 2026-02-27
**Context**: Run 4 of `frontier_sensitivity_suite` (Job 4153992) completed 31/36 sub-analyses.
**Reference**: `debugging_report_20260227_000000.md` in the analysis directory.

The 5 failures divide into two hypotheses:

| Hypothesis | Affected SAs | Proposed Fix |
|------------|-------------|--------------|
| **H1**: `MPICH_OFI_NIC_POLICY=NUMA` (default) aborts MPI_Init when any rank's CPU set spans a NUMA domain boundary | sa_29, sa_30, sa_34, sa_35 | Set `MPICH_OFI_NIC_POLICY=BLOCK` in srun environment |
| **H2**: `ntasks=4, nodes=3` forces 2 tasks onto one node → 64 CPUs > 56 allocatable limit | sa_33 | Change `n_nodes` from 3 to 4 in sensitivity definition |

**This document contains the exact commands to run on Frontier to confirm or refute each hypothesis before any code changes are made.**

---

## Setup: Get an interactive allocation

All tests require live compute nodes. Use `salloc` so each `srun` result is immediately visible.

```bash
salloc -A ***REMOVED*** -p batch -N 4 --gres=gpu:8 -t 00:30:00 -q debug
```

Expected: Allocation granted, you drop into an interactive shell with `SLURM_JOB_ID` set.
If rejected: Remove `-q debug` (debug queue may be busy) or reduce to `-N 2`.

```
# OUTPUT:
salloc: Pending job allocation 4154797
salloc: job 4154797 queued and waiting for resources
salloc: job 4154797 has been allocated resources
salloc: Granted job allocation 4154797
salloc: Waiting for resource configuration
salloc: Nodes frontier[06449-06450,06477-06478] are ready for job
```

Once allocated, load the same modules the SBATCH script uses:

```bash
module load PrgEnv-amd Core/24.07 craype-accel-amd-gfx90a miniforge3/23.11.0-0 libfabric/1.22.0
```

Expected: No errors.

```
# OUTPUT:
(base) ***REMOVED***@frontier06449:~> module load PrgEnv-amd Core/24.07 craype-accel-amd-gfx90a miniforge3/23.11.0-0 libfabric/1.22.0

Lmod is automatically replacing "cce/18.0.1" with "amd/6.2.4".


Lmod is automatically replacing "PrgEnv-cray/8.6.0" with "PrgEnv-amd/8.6.0".


Due to MODULEPATH changes, the following have been reloaded:
  1) cray-libsci/24.11.0     2) cray-mpich/8.1.31     3) darshan-runtime/3.4.6-mpi     4) tmux/3.4

The following have been reloaded with a version change:
  1) Core/25.03 => Core/24.07

Deactivating conda environments

```

---

## Baseline: Confirm NUMA topology

Before testing hypotheses, verify the NUMA layout matches what the debugging report assumed.

### B1 — Count allocatable CPUs per node

```bash
srun -N 1 --ntasks=1 --cpus-per-task=1 bash -c 'echo "Node: $(hostname); CPUs: $(nproc)"'
```

Expected: `CPUs: 56` (64 physical − 8 reserved by `-S 8` core specialization).

```
# OUTPUT:
(base) ***REMOVED***@frontier06449:~> srun -N 1 --ntasks=1 --cpus-per-task=1 bash -c 'echo "Node: $(hostname); CPUs: $(nproc)"'
Node: frontier06449; CPUs: 1

```

### B2 — Inspect NUMA domain layout

```bash
srun -N 1 --ntasks=1 --cpus-per-task=1 numactl --hardware
```

Expected: 4 NUMA nodes, each covering 16 physical CPUs (CPUs 0–15, 16–31, 32–47, 48–63). Allocatable subset will be the 14 non-specialized cores in each domain.

```
# OUTPUT:
(base) ***REMOVED***@frontier06449:~> srun -N 1 --ntasks=1 --cpus-per-task=1 numactl --hardware
available: 4 nodes (0-3)
node 0 cpus: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79
node 0 size: 127712 MB
node 0 free: 123781 MB
node 1 cpus: 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95
node 1 size: 129014 MB
node 1 free: 125814 MB
node 2 cpus: 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111
node 2 size: 129014 MB
node 2 free: 125799 MB
node 3 cpus: 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127
node 3 size: 128960 MB
node 3 free: 126089 MB
node distances:
node   0   1   2   3
  0:  10  12  12  12
  1:  12  10  12  12
  2:  12  12  10  12
  3:  12  12  12  10

```

### B3 — Confirm current MPICH NIC policy default

```bash
srun -N 1 --ntasks=1 --cpus-per-task=1 \
  bash -c 'echo "MPICH_OFI_NIC_POLICY=${MPICH_OFI_NIC_POLICY:-[not set — using compiled default NUMA]}"'
```

Expected: `[not set — using compiled default NUMA]` (env var absent, Cray MPICH defaults to NUMA policy).

```
# OUTPUT:
***REMOVED***@frontier06449:~> srun -N 1 --ntasks=1 --cpus-per-task=1 \
>   bash -c 'echo "MPICH_OFI_NIC_POLICY=${MPICH_OFI_NIC_POLICY:-[not set — using compiled default NUMA]}"'

MPICH_OFI_NIC_POLICY=NUMA

```

---

## Hypothesis 1: MPICH NIC_POLICY NUMA abort

### H1-A — Reproduce the failure (sa_29 equivalent: 2 tasks × 32 cpus, 2 nodes)

This should **FAIL** with the MPICH NIC_POLICY NUMA error.

```bash
srun -N 2 --ntasks=2 --cpus-per-task=32 --cpu-bind=cores --overlap \
  bash -c 'echo "Rank $SLURM_PROCID on $(hostname)"'
```

Expected: `MPICH ERROR: Unable to use a NIC_POLICY of 'NUMA'. Rank X is not confined to a single NUMA node.`

```
# OUTPUT:
***REMOVED***@frontier06449:~> srun -N 2 --ntasks=2 --cpus-per-task=32 --cpu-bind=cores --overlap \
>   bash -c 'echo "Rank $SLURM_PROCID on $(hostname)"'

Rank 0 on frontier06449
Rank 1 on frontier06478
***REMOVED***@frontier06449:~>

```

### H1-B — Test fix: same config WITH BLOCK policy

```bash
export MPICH_OFI_NIC_POLICY=BLOCK
srun -N 2 --ntasks=2 --cpus-per-task=32 --cpu-bind=cores --overlap \
  bash -c 'echo "Rank $SLURM_PROCID on $(hostname)"'
unset MPICH_OFI_NIC_POLICY
```

Expected: Both ranks print successfully. No MPICH abort.

```
# OUTPUT:
***REMOVED***@frontier06449:~> export MPICH_OFI_NIC_POLICY=BLOCK
***REMOVED***@frontier06449:~> srun -N 2 --ntasks=2 --cpus-per-task=32 --cpu-bind=cores --overlap \
>   bash -c 'echo "Rank $SLURM_PROCID on $(hostname)"'
t MPICH_OFI_NIC_POLICYRank 0 on frontier06449

Rank 1 on frontier06450
***REMOVED***@frontier06449:~>

```

### H1-C — Reproduce the failure (sa_30 equivalent: 4 tasks × 16 cpus, 2 nodes)

The debugging report hypothesizes `cpus-per-task=16` also fails because 14 allocatable cores per NUMA domain < 16 requested.

```bash
srun -N 2 --ntasks=4 --cpus-per-task=16 --cpu-bind=cores --overlap \
  bash -c 'echo "Rank $SLURM_PROCID on $(hostname)"'
```

Expected: Fails with NIC_POLICY NUMA abort.

```
# OUTPUT:
***REMOVED***@frontier06449:~> srun -N 2 --ntasks=4 --cpus-per-task=16 --cpu-bind=cores --overlap \
>   bash -c 'echo "Rank $SLURM_PROCID on $(hostname)"'
Rank 3 on frontier06478
Rank 2 on frontier06478
Rank 0 on frontier06477
Rank 1 on frontier06477
***REMOVED***@frontier06449:~>

```

### H1-D — Test fix for sa_30 with BLOCK policy

```bash
export MPICH_OFI_NIC_POLICY=BLOCK
srun -N 2 --ntasks=4 --cpus-per-task=16 --cpu-bind=cores --overlap \
  bash -c 'echo "Rank $SLURM_PROCID on $(hostname)"'
unset MPICH_OFI_NIC_POLICY
```

Expected: All 4 ranks print successfully.

```
# OUTPUT:
***REMOVED***@frontier06449:~> export MPICH_OFI_NIC_POLICY=BLOCK
***REMOVED***@frontier06449:~> srun -N 2 --ntasks=4 --cpus-per-task=16 --cpu-bind=cores --overlap \
>   bash -c 'echo "Rank $SLURM_PROCID on $(hostname)"'

Rank 1 on frontier06449
Rank 0 on frontier06449
Rank 2 on frontier06450
Rank 3 on frontier06450

```

### H1-E — Reproduce the failure (sa_35 equivalent: 16 tasks × 8 cpus, 3 nodes — uneven distribution)

The most subtle case: sa_35 uses `cpus-per-task=8`, which passed in sa_31 at 2 nodes, but fails at 3 nodes because uneven task distribution causes non-NUMA-aligned placement of the extra task.

```bash
srun -N 3 --ntasks=16 --cpus-per-task=8 --cpu-bind=cores --overlap \
  bash -c 'echo "Rank $SLURM_PROCID on $(hostname)"'
```

Expected: Fails with NIC_POLICY NUMA abort on at least one rank.

```
# OUTPUT:
***REMOVED***@frontier06449:~> srun -N 3 --ntasks=16 --cpus-per-task=8 --cpu-bind=cores --overlap \
>   bash -c 'echo "Rank $SLURM_PROCID on $(hostname)"'
Rank 9 on frontier06477
Rank 3 on frontier06449
Rank 0 on frontier06449
Rank 8 on frontier06477
Rank 13 on frontier06478
Rank 14 on frontier06478
Rank 5 on frontier06449
Rank 10 on frontier06477
Rank 6 on frontier06477
Rank 1 on frontier06449
Rank 15 on frontier06478
Rank 7 on frontier06477
Rank 12 on frontier06478
Rank 4 on frontier06449
Rank 2 on frontier06449
Rank 11 on frontier06478

```

### H1-F — Test fix for sa_35 with BLOCK policy

```bash
export MPICH_OFI_NIC_POLICY=BLOCK
srun -N 3 --ntasks=16 --cpus-per-task=8 --cpu-bind=cores --overlap \
  bash -c 'echo "Rank $SLURM_PROCID on $(hostname)"'
unset MPICH_OFI_NIC_POLICY
```

Expected: All 16 ranks print successfully.

```
# OUTPUT:
***REMOVED***@frontier06449:~> export MPICH_OFI_NIC_POLICY=BLOCK
***REMOVED***@frontier06449:~> srun -N 3 --ntasks=16 --cpus-per-task=8 --cpu-bind=cores --overlap \
>   bash -c 'echo "Rank $SLURM_PROCID on $(hostname)"'
t MPICH_OFI_NIC_POLICY
Rank 1 on frontier06449
Rank 0 on frontier06449
Rank 7 on frontier06450
Rank 4 on frontier06449
Rank 3 on frontier06449
Rank 6 on frontier06450
Rank 10 on frontier06450
Rank 2 on frontier06449
Rank 5 on frontier06449
Rank 9 on frontier06450
Rank 8 on frontier06450
Rank 13 on frontier06477
Rank 14 on frontier06477
Rank 11 on frontier06477
Rank 12 on frontier06477
Rank 15 on frontier06477
***REMOVED***@frontier06449:~>

```

### H1-G — Regression check: passing case still passes with BLOCK

sa_31 (`8 tasks × 8 cpus, 2 nodes`) passed without any NIC policy override. Confirm BLOCK doesn't break it (it should be strictly more permissive than NUMA).

```bash
export MPICH_OFI_NIC_POLICY=BLOCK
srun -N 2 --ntasks=8 --cpus-per-task=8 --cpu-bind=cores --overlap \
  bash -c 'echo "Rank $SLURM_PROCID on $(hostname)"'
unset MPICH_OFI_NIC_POLICY
```

Expected: All 8 ranks print successfully.

```
# OUTPUT:
***REMOVED***@frontier06449:~> export MPICH_OFI_NIC_POLICY=BLOCK
***REMOVED***@frontier06449:~> srun -N 2 --ntasks=8 --cpus-per-task=8 --cpu-bind=cores --overlap \
>   bash -c 'echo "Rank $SLURM_PROCID on $(hostname)"'
 MPICH_OFI_NIC_POLICY
Rank 3 on frontier06449
Rank 4 on frontier06478
Rank 7 on frontier06478
Rank 5 on frontier06478
Rank 6 on frontier06478
Rank 0 on frontier06449
Rank 1 on frontier06449
Rank 2 on frontier06449
***REMOVED***@frontier06449:~>

```

---

## Hypothesis 2: Uneven task distribution exceeds per-node CPU limit

### H2-A — Diagnose task distribution (4 tasks across 3 nodes)

First, check that SLURM actually places 2 tasks on one node (the assumed 2+1+1 distribution), using a cheap 1-CPU-per-task request so resource limits don't interfere.

```bash
srun -N 3 --ntasks=4 --cpus-per-task=1 --overlap \
  bash -c 'echo "Rank $SLURM_PROCID on $(hostname)"'
```

Expected: One hostname appears twice, the other two once each (2+1+1 distribution).

```
# OUTPUT:
***REMOVED***@frontier06449:~> srun -N 3 --ntasks=4 --cpus-per-task=1 --overlap \
>   bash -c 'echo "Rank $SLURM_PROCID on $(hostname)"'
Rank 2 on frontier06450
Rank 1 on frontier06449
Rank 0 on frontier06449
Rank 3 on frontier06477
***REMOVED***@frontier06449:~>

```

### H2-B — Reproduce the failure (sa_33 equivalent: 4 tasks × 32 cpus, 3 nodes)

With 2 tasks on one node: 2 × 32 = 64 CPUs requested on that node, exceeding the 56-core limit.

```bash
srun -N 3 --ntasks=4 --cpus-per-task=32 --cpu-bind=cores --overlap \
  bash -c 'echo "Rank $SLURM_PROCID on $(hostname)"'
```

Expected: `srun: error: Unable to create step for job XXXXXX: More processors requested than permitted`

```
# OUTPUT:
***REMOVED***@frontier06449:~> srun -N 3 --ntasks=4 --cpus-per-task=32 --cpu-bind=cores --overlap \
>   bash -c 'echo "Rank $SLURM_PROCID on $(hostname)"'
srun: error: Unable to create step for job 4154797: More processors requested than permitted

```

### H2-C — Test fix: n_nodes=4 (1 task per node × 32 cpus = 32 CPUs/node)

```bash
srun -N 4 --ntasks=4 --cpus-per-task=32 --cpu-bind=cores --overlap \
  bash -c 'echo "Rank $SLURM_PROCID on $(hostname)"'
```

Expected: All 4 ranks print successfully, each on a different node.

```
# OUTPUT:
***REMOVED***@frontier06449:~> srun -N 4 --ntasks=4 --cpus-per-task=32 --cpu-bind=cores --overlap \
>   bash -c 'echo "Rank $SLURM_PROCID on $(hostname)"'
Rank 0 on frontier06449
Rank 1 on frontier06450
Rank 2 on frontier06477
Rank 3 on frontier06478

```

---

## Combined: BLOCK policy + n_nodes=4 for sa_33

sa_33 needs both fixes: correct node count (H2) AND `cpus-per-task=32` spans 2 NUMA domains (H1). Test them together:

```bash
export MPICH_OFI_NIC_POLICY=BLOCK
srun -N 4 --ntasks=4 --cpus-per-task=32 --cpu-bind=cores --overlap \
  bash -c 'echo "Rank $SLURM_PROCID on $(hostname)"'
unset MPICH_OFI_NIC_POLICY
```

Expected: All 4 ranks print successfully, each on a different node.

```
# OUTPUT:
***REMOVED***@frontier06449:~> export MPICH_OFI_NIC_POLICY=BLOCK
***REMOVED***@frontier06449:~> srun -N 4 --ntasks=4 --cpus-per-task=32 --cpu-bind=cores --overlap \
>   bash -c 'echo "Rank $SLURM_PROCID on $(hostname)"'

Rank 1 on frontier06450
Rank 3 on frontier06478
Rank 2 on frontier06477
Rank 0 on frontier06449

```

---

## Summary Table

Fill in after running the tests above:

| Test | Config | NIC Policy | Expected | Actual Result | Pass/Fail |
|------|--------|-----------|----------|---------------|-----------|
| B1 | N=1, ntasks=1, cpt=1 | — | `CPUs: 56` | | |
| B2 | N=1, ntasks=1, cpt=1 | — | 4 NUMA nodes, 16 CPUs each | | |
| B3 | N=1, ntasks=1, cpt=1 | — | MPICH_OFI_NIC_POLICY not set | | |
| H1-A | N=2, ntasks=2, cpt=32 | default (NUMA) | FAIL: NIC abort | | |
| H1-B | N=2, ntasks=2, cpt=32 | BLOCK | PASS | | |
| H1-C | N=2, ntasks=4, cpt=16 | default (NUMA) | FAIL: NIC abort | | |
| H1-D | N=2, ntasks=4, cpt=16 | BLOCK | PASS | | |
| H1-E | N=3, ntasks=16, cpt=8 | default (NUMA) | FAIL: NIC abort | | |
| H1-F | N=3, ntasks=16, cpt=8 | BLOCK | PASS | | |
| H1-G | N=2, ntasks=8, cpt=8 | BLOCK | PASS (regression) | | |
| H2-A | N=3, ntasks=4, cpt=1 | — | 2+1+1 distribution | | |
| H2-B | N=3, ntasks=4, cpt=32 | — | FAIL: too many procs | | |
| H2-C | N=4, ntasks=4, cpt=32 | — | PASS | | |
| Combined | N=4, ntasks=4, cpt=32 | BLOCK | PASS | | |

---

## Interpretation guide

### If H1-A passes (no NIC abort without BLOCK):
The root cause is not `MPICH_OFI_NIC_POLICY=NUMA`. The failure may be in task binding or resource contention rather than NIC assignment. Re-run H1-A with `--cpu-bind=verbose` to inspect task placement, then re-examine the actual logs from sa_29.

### If H1-B fails (still aborts with BLOCK):
`BLOCK` is not overriding the Cray MPICH compiled default. Try `ROUND-ROBIN`:

```bash
export MPICH_OFI_NIC_POLICY=ROUND-ROBIN
srun -N 2 --ntasks=2 --cpus-per-task=32 --cpu-bind=cores --overlap \
  bash -c 'echo "Rank $SLURM_PROCID on $(hostname)"'
```

Also check whether a site-level value is already set in the SBATCH environment:

```bash
env | grep MPICH
```

### If H2-A shows even distribution (not 2+1+1):
SLURM's default task distribution may have changed. In this case H2-B likely won't reproduce either. Verify the actual CPUs-per-node with:

```bash
srun -N 3 --ntasks=4 --cpus-per-task=32 --overlap \
  bash -c 'echo "Rank $SLURM_PROCID on $(hostname): $(nproc) CPUs visible"'
```

### If H2-C fails (n_nodes=4 still rejected):
Check `scontrol show job $SLURM_JOB_ID | grep -i cpu` to see how SLURM is counting the step's CPUs.

---

## Next steps after testing

**If all hypotheses confirmed (all expected FAIL/PASS results match):**
1. Add `MPICH_OFI_NIC_POLICY=BLOCK` to the `env` dict in `run_simulation.py:prepare_simulation_command()` alongside the other OMP env vars (~line 397)
2. Update the sensitivity definition: change sa_33 row from `n_nodes=3` to `n_nodes=4` in the source `.xlsx`
3. Re-run with `--rerun-incomplete` — only the 5 failed sims need to rerun

**If any hypothesis refuted:**
Fill in the "Interpretation guide" section above and return to root cause analysis before making code changes.
