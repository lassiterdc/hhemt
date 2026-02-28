# Bug Investigation: srun Step Scheduling Latency Under High Concurrency (sa_26, 4050s)

**Date**: 2026-02-27
**Priority**: Medium (affects job efficiency and allocation budget in 1_job_many_srun_tasks mode)
**Related debugging report**: `frontier_sensitivity_suite/debugging_docs/debugging_report_20260227_2056.md` (Run 7)
**Status**: SHELVED (2026-02-28) — anomaly never recurred; empirical Test 5 not obtained; frontier_sensitivity_suite completed successfully without further sa_26 issues. Reopen if 4000s+ latency anomaly is observed again in future runs.

---

## Observation

In Run 7 (Job 4155594), sa_26 (`hybrid, 8 MPI, 4 OMP, 1 node`) completed correctly but took **4050 seconds (67 minutes)** elapsed time. All 36 simulation rules were dispatched concurrently at ~17:37:25. Comparable hybrid simulations on 2–4 nodes completed in 75–220 seconds.

| SA | n_mpi | n_omp | n_nodes | srun CPUs/node | elapsed_s |
|----|-------|-------|---------|----------------|-----------|
| sa_25 | 4 | 8 | 4 | 8 | 84 |
| sa_30 | 4 | 16 | 2 | 32 | 75 |
| sa_33 | 4 | 32 | 4 | 32 | 75 |
| **sa_26** | 8 | 4 | **1** | **32** | **4050** |

TRITON's own model log shows a clean 120-second simulation — confirming the 3930-second remainder was scheduling/queue latency, not compute time.

---

## Hypothesis History

### Original hypothesis (from `snakemake-specialist`) — **REFUTED**

`--cpu-bind=cores` was proposed to enforce strict CPU binding at SLURM step admission time, causing serialization when 36 concurrent srun steps all request high per-node CPU counts. This hypothesis was incorrect.

**Why it was wrong** (from `slurm-specialist` SLURM 24.11.5 source analysis):

- `--cpu-bind=cores` (`cpu_bind_type = CPU_BIND_TO_CORES`) is **never consulted by step admission code** in `stepmgr/stepmgr.c`. It is passed to `slurmstepd` and enforced post-fork via `plugins/task/affinity/dist_tasks.c:190`. A step cannot block at slurmctld due to `cpu_bind_type`.
- `--overlap` does **not** merely bypass node exclusivity. It maps to `SSF_OVERLAP_FORCE`, which bypasses **all** per-node CPU and core slot accounting at admission: the `cpus_used` deduction (`stepmgr.c:1281-1308`), `core_bitmap_used` marking (`stepmgr.c:2037`), and `cpus_used` increment in `_step_alloc_lps()` (`stepmgr.c:2654`) are all guarded by `!(step_ptr->flags & SSF_OVERLAP_FORCE)` and are skipped entirely.
- The per-node CPU demand asymmetry (sa_26 needs 32 CPUs from 1 node vs. multi-node steps needing 8 CPUs/node) is **irrelevant** under `SSF_OVERLAP_FORCE` — both pass step admission immediately.

**Consequence**: removing `--cpu-bind=cores` would have zero effect on step admission latency and is **not a fix for this anomaly**.

### Current hypothesis — **slurmctld RPC rate limiting**

When 36 `srun` processes fire `REQUEST_STEP_CREATE` RPCs to slurmctld nearly simultaneously, Frontier's rate limiter (`rl_enable`, `rl_bucket_size=350`, `rl_refill_rate=64`) throttles some of them. Throttled srun clients receive a retryable error and re-enter a wait loop (`srun/launch.c:1212-1296`). The retry interval is:

```
step_wait = ((getpid() % 10) + MIN(300, MAX(60, slurmctld_timeout))) * 1000 ms
```

With default `slurmctld_timeout=120s`, individual retry intervals span **60–310 seconds**. Over multiple retries, a step that was throttled on first attempt could wait hours before being admitted — especially if it keeps timing out rather than being woken by `srun_step_signal()` (which only applies to steps already admitted as `SLURM_PENDING_STEP`; throttled steps are retrying from scratch each time).

The 4050-second wait is consistent with 3–5 throttle-retry cycles at 60–310 seconds each.

---

## Investigation Plan

### Test 5: Capture srun stderr for RPC throttling messages

Run 36 concurrent `--overlap` srun steps inside an active allocation on Frontier and capture srun stderr. If RPC rate limiting is the cause, srun will emit "step creation temporarily disabled, retrying" messages (source: `srun/launch.c:1266`).

```bash
# On Frontier, inside salloc or SBATCH with ≥2 nodes
# Launch 36 concurrent --overlap srun steps; log each step's stderr and start/end time
mkdir -p /tmp/srun_test_logs
for i in $(seq 1 36); do
    (
        start=$(date +%s%3N)
        srun -N 1 --ntasks=8 --cpus-per-task=4 --overlap --kill-on-bad-exit=1 \
            sleep 10 2>/tmp/srun_test_logs/step_${i}.err
        exit_code=$?
        end=$(date +%s%3N)
        wait_ms=$(( (end - start) - 10000 ))
        echo "step $i: waited ${wait_ms}ms before launch, exit=$exit_code"
    ) &
done
wait
echo "--- srun stderr with 'retry' messages ---"
grep -i "retry\|disabled\|throttle\|EAGAIN\|timeout" /tmp/srun_test_logs/*.err 2>/dev/null || echo "(none found)"
```

**Expected output if RPC throttling is the cause**:
- At least one step shows `waited >60000ms` (60+ second pre-launch wait)
- `step_N.err` files contain "step creation temporarily disabled, retrying" or similar
```
# Output (fill in after running on Frontier):

```

**Expected output if RPC throttling is NOT the cause**:
- All steps launch within a few seconds of each other
- No retry messages in stderr
```
# Output (fill in after running on Frontier):

```

### Test 6 (optional): Check slurmctld logs via Frontier support

If Test 5 confirms throttling at the srun client, follow up by asking the Frontier support team whether slurmctld rate-limiting events (RL_THROTTLE or equivalent) appear in the controller log for Job 4155594. This would provide server-side confirmation.

---

## Potential Fixes (pending Test 5 confirmation)

### Option D: Reduce Snakemake `--jobs` concurrency

Cap the number of simultaneously-launched srun steps to stay within `rl_bucket_size=350` / `rl_refill_rate=64`. Reducing `--jobs` from 453 (current ThreadPoolExecutor default) to 20–24 would spread `REQUEST_STEP_CREATE` RPCs over time and avoid triggering the rate limiter.

**Change location**: wherever `--jobs` is passed to Snakemake in `1_job_many_srun_tasks` mode (workflow launch logic).

**Risk**: Reduces the parallelism benefit of the `1_job_many_srun_tasks` mode slightly; simulations queue rather than all starting at once. Acceptable if the alternative is random 60–300s throttle delays.

### Option E: Contact Frontier support to increase rate limiter config

Ask OLCF whether `rl_bucket_size` and `rl_refill_rate` can be increased, or whether the rate limiter can be exempted for step-creation RPCs originating from within a batch job. This is a site configuration change — outside our control but worth raising if Test 5 confirms this is the cause.

### Option F: No change — accept sporadic latency

If sa_26-class anomalies are rare (only 1 in ~180 completed simulations across Runs 1–7) and the latency does not exceed the job's wall-clock budget, the current behavior may be acceptable. The wall-clock budget concern is mitigated by `--kill-on-bad-exit=1` (now implemented), which prevents the *other* class of hang (PMI partial launch failure).

---

## Decision criteria

- **Implement Option D** if: Test 5 confirms srun retry messages appear for some steps
- **Pursue Option E** if: Option D is confirmed as the cause and reducing concurrency is undesirable
- **Implement Option F (no change)** if: Test 5 shows no retry messages (anomaly was a one-off or has a different unexplained cause)

---

## SLURM Source Analysis

**SLURM version**: 24.11.5 (submodule tag `slurm-24-11-5-1`, matching Frontier). All citations are to `slurm/src/` paths relative to `/home/***REMOVED***/dev/slurm-workspace/`.

---

### Q1: Is `--cpu-bind=cores` evaluated at step admission time or post-admission?

**Answer: Post-admission. `--cpu-bind=cores` plays no role in step admission.**

`--cpu-bind=cores` (parsed to `cpu_bind_type = CPU_BIND_TO_CORES`) is carried in the step request message but is never consulted by any code in `stepmgr/stepmgr.c` that gates step admission. The step manager has no branch that returns `ESLURM_NODES_BUSY` or any other deferral code based on `cpu_bind_type`.

The `cpu_bind_type` is stored in the step record and shipped to `slurmstepd` via `launch_tasks_request_msg_t`. It is consumed entirely in the `slurmstepd` / task plugin layer, after the step has already been admitted and tasks are being launched:
- `stepmgr/slurmstepd/slurmstepd_job.c:328`: step record receives `msg->cpu_bind_type`
- `stepmgr/slurmstepd/task.c:347`: passed into `step->envtp->cpu_bind_type`
- `plugins/task/affinity/dist_tasks.c:190` and `plugins/task/cgroup/task_cgroup_cpuset.c:472,487,925`: CPU binding is enforced there via `CPU_BIND_TO_CORES` after tasks are forked

**`--cpu-bind=cores` cannot block a step at slurmctld.** A step can only block (return `ESLURM_NODES_BUSY` and enter the pending-step wait loop) due to checks in `_pick_step_nodes()` and `_step_alloc_lps()` — neither of which reads `cpu_bind_type`.

---

### Q2: What does `--overlap` actually bypass?

**Answer: Under 24.11.5, `--overlap` maps unconditionally to `SSF_OVERLAP_FORCE` and bypasses essentially all per-node CPU and memory slot accounting at admission time.**

The mapping chain:
- `common/slurm_opt.c:2800`: `arg_set_overlap()` sets `opt->srun_opt->overlap_force = true`
- `srun/launch.c:702-703`: `if (srun_opt->overlap_force) step_req->flags |= SSF_OVERLAP_FORCE;`

`SSF_OVERLAP_FORCE` bypasses the following checks in `_pick_step_nodes()`:

1. **Per-node CPU-used tracking** (`stepmgr/stepmgr.c:1281-1308`): The block that subtracts `cpus_used[node_inx]` from `usable_cpu_cnt[i]` is skipped entirely. With `SSF_OVERLAP_FORCE`, `usable_cpu_cnt[i]` remains equal to the full `job_resrcs_ptr->cpus[node_inx]` for every node, regardless of how many CPUs other running steps have consumed on that node.

2. **Memory-used tracking** (`stepmgr/stepmgr.c:1340,1364`): Memory accounting against `memory_used[node_inx]` is skipped.

3. **`cpus_used` increment in `_step_alloc_lps()`** (`stepmgr/stepmgr.c:2654-2664`): The allocated CPUs are not added to `cpus_used[job_node_inx]`, so other steps' `usable_cpu_cnt` calculations remain unaffected.

4. **`memory_used` increment in `_step_alloc_lps()`** (`stepmgr/stepmgr.c:2688-2704`): Memory used by this step is not counted against the job's memory budget.

5. **`core_bitmap_used` marking in `_pick_step_core()`** (`stepmgr/stepmgr.c:2037-2042`): Under `SSF_OVERLAP_FORCE`, the `core_bitmap_used` bit is not set, so the step's core selections are invisible to subsequent steps' `_pick_step_cores()` calls.

6. **GRES accounting** (`stepmgr/stepmgr.c:2507-2508`): `gres_stepmgr_step_alloc()` is called with `need_alloc=false` under `SSF_OVERLAP_FORCE`.

**The one check `SSF_OVERLAP_FORCE` does NOT bypass** is the node-list availability check at `stepmgr/stepmgr.c:1488`:

```c
if (!bit_super_set(selected_nodes, nodes_avail)) { ... *return_code = ESLURM_NODES_BUSY; ... }
```

However, because the `cpus_used` deduction block at 1281-1308 is skipped, `nodes_avail` retains all job nodes (none are cleared at line 1312 for insufficient CPUs). So a specifically-requested node list under `SSF_OVERLAP_FORCE` will always pass this check unless the node is DOWN or powered off.

**Conclusion: `--overlap` (`SSF_OVERLAP_FORCE`) bypasses CPU slot availability entirely at admission time.** The per-node CPU tracking that the prior hypothesis relied upon does not apply when `SSF_OVERLAP_FORCE` is set.

---

### Q3: Is there a mechanism by which a 1-node step with high per-node CPU demand could queue behind multi-node steps under `--overlap`?

**Answer: Not through the per-node CPU or core-binding mechanisms. The hypothesis is incorrect. However, a step can still block under `--overlap` via two other paths.**

Given that `SSF_OVERLAP_FORCE` skips all per-node CPU accounting, the per-node demand asymmetry (sa_26 needs 32 CPUs from 1 node vs. multi-node steps needing 8 CPUs per node) is irrelevant — both pass immediately under `SSF_OVERLAP_FORCE`.

**Paths that CAN block step admission even under `SSF_OVERLAP_FORCE`:**

**Path A: `nodes_avail` reduced by DOWN/NO_RESPOND nodes.**
If a job node goes unresponsive during the run, it is cleared from `nodes_avail` before the node-list check. A single-node step targeting that specific node would get `ESLURM_NODES_BUSY`. Multi-node steps targeting 4 of 8 nodes might be able to pick an alternative. This is unlikely to explain a 4050-second wait in an otherwise clean run.

**Path B: slurmctld RPC rate limiting.**
Frontier's `slurmctld` is configured with rate limiting (`rl_enable`, `rl_bucket_size=350`, `rl_refill_rate=64`; `slurm-workspace/CLAUDE.md`). With 36 srun processes issuing `REQUEST_STEP_CREATE` RPCs simultaneously, some may be throttled. Throttled RPCs return an error that `launch_common_step_retry_errno()` recognizes as retryable (`EAGAIN` or `SLURM_PROTOCOL_SOCKET_IMPL_TIMEOUT`; `srun/launch.c:1113-1121`). The srun client then re-attempts the step creation in a loop (`srun/launch.c:1212-1296`), waiting up to `step_wait` milliseconds between attempts (`srun/launch.c:1228-1231`): `step_wait = ((getpid() % 10) + MIN(300, MAX(60, slurmctld_timeout))) * 1000`. With default `slurmctld_timeout=120s`, `step_wait` ranges from 60 to 310 seconds per attempt. This could serialize step creation in a way that is unrelated to the step content.

**Path C: `_wake_pending_steps` throttling.**
When a step returns `ESLURM_NODES_BUSY` and registers a pending step (`stepmgr/stepmgr.c:3528-3531`), it waits on `poll()` on a socket for slurmctld to signal it (`srun/step_ctx.c:170-200`). `_wake_pending_steps()` is called whenever a step completes (`stepmgr/stepmgr.c:379`), and it signals at most `config_start_count` pending steps per invocation (default: 8; tunable via `step_retry_count=` in `sched_params`). Pending steps older than `config_max_age` seconds (default: 60) are preferentially signaled. Under `SSF_OVERLAP_FORCE`, steps should not be entering the pending state (since per-node CPU checks are bypassed), so this path is unlikely unless something else is causing `ESLURM_NODES_BUSY`.

---

### Q4: If `--cpu-bind=cores` is post-admission only, what is the more likely explanation for the 4050s wait?

**Answer: The most likely explanation is slurmctld RPC rate limiting (Path B above) combined with the srun step-wait timeout mechanism.**

When 36 `srun` processes fire `REQUEST_STEP_CREATE` RPCs to slurmctld nearly simultaneously, Frontier's rate limiter (`rl_bucket_size=350`, `rl_refill_rate=64`) may throttle some of them. The throttled `srun` processes receive a retryable error and re-enter the wait loop.

The key timing observation is that `step_wait` is `((getpid() % 10) + slurmctld_timeout) * 1000` milliseconds. If `slurmctld_timeout=120s`, individual srun retry intervals span 60–130 seconds. Over multiple retries, a step that was unlucky on its first few attempts could wait hours before being admitted — particularly if it keeps timing out instead of being woken by `srun_step_signal()`. The `srun_step_signal()` mechanism only applies to steps that were admitted as `SLURM_PENDING_STEP`; a step that was rate-limited and never made it to the pending-step queue is simply retrying from scratch each time.

A secondary but plausible contributor: **network fabric (OFI/libfabric/CXI) initialization contention**. When 36 MPI jobs all initialize PMIx/OFI simultaneously on 8 shared nodes, the NIC and fabric may contend on connection setup. This would manifest as post-admission compute time, not as SLURM scheduler latency — but if the TRITON model log captures only steady-state compute (not initialization), the 120s "clean simulation" might exclude a lengthy OFI bootstrap phase. [UNVERIFIED — depends on what TRITON's log.out timestamps capture.]

**The 4050-second wait is more consistent with srun being unable to re-register a pending step with slurmctld after repeated timeouts, than with a deliberate per-CPU-slot scheduling queue.**

---

### Q5: Would removing `--cpu-bind=cores` change step admission behavior?

**Answer: No. Removing `--cpu-bind=cores` would have zero effect on step admission latency, since `cpu_bind_type` is never consulted by the step admission code path.**

`--cpu-bind=cores` is enforced entirely in `slurmstepd` after admission. Removing it would:
- Have no effect on step admission speed
- Have no effect on how long srun waits for the `REQUEST_STEP_CREATE` RPC to succeed
- Only affect CPU affinity of tasks after they are launched

If the root cause is slurmctld rate limiting, removing `--cpu-bind=cores` does not address it. If the root cause is OFI initialization contention, removing `--cpu-bind=cores` might slightly change NUMA placement (tasks could be placed on different cores) but would not reduce contention.

---

### Assessment of the Original Hypothesis

**The hypothesis is incorrect.** The prior snakemake-specialist analysis assumed that `--overlap` only bypasses "node exclusivity" while leaving per-node CPU slot tracking active. The source code shows the opposite: `--overlap` (`SSF_OVERLAP_FORCE`) bypasses all per-node CPU and core accounting at step admission time. Specifically:

- The `cpus_used` subtraction from `usable_cpu_cnt` is guarded by `!(step_spec->flags & SSF_OVERLAP_FORCE)` at `stepmgr/stepmgr.c:1281-1282`
- The `core_bitmap_used` bit-set in `_pick_step_core()` is guarded by `!(step_ptr->flags & SSF_OVERLAP_FORCE)` at `stepmgr/stepmgr.c:2037`
- The `cpus_used` increment in `_step_alloc_lps()` is guarded by `!(step_ptr->flags & SSF_OVERLAP_FORCE)` at `stepmgr/stepmgr.c:2654`

The per-node CPU demand asymmetry between 1-node and multi-node steps cannot cause scheduling serialization under `--overlap`. The 4050-second wait must have a different root cause.

---

### Recommendation

**Do not remove `--cpu-bind=cores` as a fix for this latency.** Source analysis confirms it is post-admission and cannot affect scheduling latency. The change would be inert with respect to the anomaly.

Investigate slurmctld rate limiting as the primary candidate via **Test 5** above. If confirmed, **Option D** (reduce Snakemake `--jobs` concurrency) is the appropriate fix.
