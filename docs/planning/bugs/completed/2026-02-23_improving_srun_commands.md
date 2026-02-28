# Improving `srun` Commands for TRITON-SWMM

**Created:** 2026-02-23

## Status

- **Type**: Implemented — pending field validation on UVA cluster
- **Goal**: Make `srun` launch arguments safer and easier to diagnose in production SLURM environments
- **Scope**:
  - `src/TRITON_SWMM_toolkit/run_simulation.py` — command construction and pre-launch validation
  - `src/TRITON_SWMM_toolkit/resource_management.py` — GPU env var parsing utility
  - `src/TRITON_SWMM_toolkit/run_simulation_runner.py` — enriched srun error handling post-launch

---

## Summary of Agreed Direction

Based on current discussion, we will:

1. **Keep GPU launch simple**:
   - Keep `--ntasks-per-gpu=1`
   - **Do not add** `--gpus-per-task` for now (treat as potentially redundant in our workflow)

2. **Remove `--overlap`** from default `srun` launches.

3. **Switch CPU binding to cores**:
   - Use `--cpu-bind=cores` instead of `--cpu-bind=none`
   - Keep OpenMP settings (`OMP_PROC_BIND=true`, `OMP_PLACES=cores`)

4. **Add GPU allocation pre-checks** analogous to current CPU allocation checks.

5. **Add defensive diagnostics** around `srun` launch failures (especially MPI/PMI/cgroup binding issues), without overcomplicating defaults.

---

## Why this direction

### 1) Why keep only `--ntasks-per-gpu=1`?

In this code path, GPU mode already sets:

- `--ntasks={n_gpus}`
- `--ntasks-per-gpu=1`
- `--cpus-per-task={n_omp_threads}`

This establishes 1 rank per GPU and is usually sufficient on modern SLURM clusters when combined with scheduler-assigned GPUs. Adding `--gpus-per-task=1` can be helpful on some systems, but it can also create policy conflicts with site-level sbatch defaults and is not required for this phase.

### 2) Why `--cpu-bind=cores`?

`--cpu-bind=cores` typically improves CPU locality and reduces thread migration. It aligns better with:

- `OMP_PROC_BIND=true`
- `OMP_PLACES=cores`

Compared with `--cpu-bind=none`, this is more deterministic and generally better for hybrid MPI/OpenMP performance.

---

## What could go wrong with `--cpu-bind=cores`?

Yes, it can cause issues on some sites/configurations. Typical failure modes:

1. **Immediate `srun` launch error** (best case)
   - Example classes of messages:
     - "Unable to satisfy cpu bind request"
     - "Task launch failed"
     - cgroup/cpuset mismatch messages

2. **Job starts but poor performance / imbalance**
   - Symptoms:
     - throughput regression vs baseline
     - one or more ranks lagging badly
     - unusual CPU utilization patterns

3. **Binding conflict with site policy**
   - Some clusters force binding behavior via scheduler config or cgroups.

### How to diagnose quickly

Use these commands during debugging:

```bash
scontrol show job "$SLURM_JOB_ID"
srun --cpu-bind=verbose -n 2 -c 4 /bin/hostname
env | grep -E 'SLURM|OMP|CUDA_VISIBLE_DEVICES|HIP_VISIBLE_DEVICES'
```

If launch fails, capture and persist the exact stderr from `srun` in scenario logs. That message is usually enough to determine whether the issue is PMI, binding, or resource mismatch.

---

## Defensive coding pattern (recommended)

We want behavior that is strict by default but diagnosable:

1. **Preflight checks** (fail fast before launch):
   - CPU: keep existing `expected_cpus <= SLURM_NTASKS * SLURM_CPUS_PER_TASK`
   - GPU: add analogous checks against allocated GPUs

2. **Single canonical launch string**:
   - Remove `--overlap`
   - Set `--cpu-bind=cores`
   - Keep `--ntasks-per-gpu=1` in GPU mode

3. **Structured launch error handling**:
   - On non-zero return code, parse known `srun` stderr patterns and raise actionable RuntimeError guidance
   - Include key env snapshot (`SLURM_*`, run mode, ntasks, cpus-per-task, gpu count)

4. **Optional fallback (disabled by default)**:
   - If future evidence shows site-specific bind incompatibility, allow config-level fallback from `cores` to `none`.

---

## Production-ready code chunks (proposed)

> These snippets are designed to be dropped into `TRITONSWMM_run.prepare_simulation_command` after final review.

### A) Utility: parse allocated GPU count from SLURM env

**Location**: `resource_management.py` (module-level private function). This is where existing SLURM GPU env var parsing already lives (lines 324-330); placing the utility here avoids duplication.

```python
def _parse_slurm_allocated_gpus(env: dict[str, str]) -> int:
    """Best-effort parse of allocated GPU count from common SLURM env vars.

    Tries variables in priority order. If none yield a usable value, prints
    a diagnostic to stdout (captured in runner logs) listing what was tried.

    Returns
    -------
    int
        Total GPUs visible from SLURM metadata for this job allocation,
        or 0 if not detectable.
    """
    import re

    tried: dict[str, str | None] = {}

    # Most reliable if present
    val = env.get("SLURM_GPUS")
    tried["SLURM_GPUS"] = val
    if val:
        if val.isdigit():
            return int(val)
        parts = [p for p in re.split(r"[,\s]+", val.strip()) if p]
        if parts:
            return len(parts)

    # Alternative variable sometimes present
    val = env.get("SLURM_GPUS_ON_NODE")
    tried["SLURM_GPUS_ON_NODE"] = val
    if val and val.isdigit():
        return int(val)

    # Fallback: comma-separated list of GPU IDs (e.g. "0,1,2,3")
    val = env.get("SLURM_JOB_GPUS")
    tried["SLURM_JOB_GPUS"] = val
    if val:
        parts = [p for p in val.split(",") if p.strip()]
        if parts:
            return len(parts)

    # Nothing usable found — emit diagnostic so logs show what was checked
    print(
        f"[GPU-PREFLIGHT] Could not detect allocated GPU count from SLURM environment. "
        f"Tried: {tried}. Skipping GPU preflight check.",
        flush=True,
    )
    return 0
```

### B) GPU preflight validation (analogous to CPU validation)

```python
if using_srun and "SLURM_JOB_ID" in os.environ and run_mode == "gpu":
    allocated_gpus = _parse_slurm_allocated_gpus(os.environ)
    expected_gpus = int(n_gpus)

    # If we can detect allocation and it's insufficient, fail fast.
    if allocated_gpus > 0 and allocated_gpus < expected_gpus:
        raise RuntimeError(
            "SLURM GPU allocation mismatch: "
            f"configuration requires {expected_gpus} GPUs but allocation appears to "
            f"provide {allocated_gpus}. "
            "Refusing launch to avoid hanging/oversubscription. "
            "Inspect SLURM_GPUS/SLURM_GPUS_ON_NODE/SLURM_JOB_GPUS and sbatch request."
        )
```

### C) Updated launch command construction

```python
if run_mode != "gpu":
    if using_srun:
        launch_cmd_str = (
            f"srun "
            f"-N {n_nodes_per_sim} "
            f"--ntasks={n_mpi_procs} "
            f"--cpus-per-task={n_omp_threads} "
            f"--cpu-bind=cores "
            f"{exe} {cfg}"
        )
    elif run_mode in ("serial", "openmp"):
        launch_cmd_str = f"{exe} {cfg}"
    elif run_mode in ("mpi", "hybrid"):
        launch_cmd_str = f"mpirun -np {n_mpi_procs} {exe} {cfg}"
elif run_mode == "gpu":
    if using_srun:
        launch_cmd_str = (
            f"srun "
            f"-N {n_nodes_per_sim} "
            f"--ntasks={n_gpus} "
            f"--cpus-per-task={n_omp_threads} "
            f"--ntasks-per-gpu=1 "
            f"--cpu-bind=cores "
            f"{exe} {cfg}"
        )
    else:
        launch_cmd_str = f"{exe} {cfg}"
else:
    raise ValueError(f"Unknown run_mode: {run_mode}")
```

### D) Error enrichment wrapper (diagnostic-first)

**Location**: `run_simulation_runner.py`. The runner script is where the subprocess return code is checked after `proc.wait()`. At that point `model_logfile` already has TRITON's output; srun launch failures mean TRITON never started, so diagnostics should go via `logger.error(...)` (captured in the runner's stderr → Snakemake/SLURM job logs), not into `model_logfile`. The enrichment function raises a `RuntimeError` which propagates to the runner's outer `except Exception` block (line 235), which calls `logger.error(traceback.format_exc())` automatically.

**Note on `--mpi=pmix`**: Do not add `--mpi=` flags preemptively. The PMI hint below is sufficient to surface the symptom when it occurs; the fix (adding an explicit config toggle) should only happen after a confirmed failure.

```python
def _raise_enriched_srun_error(stderr_text: str, *, run_mode: str, cmd: str) -> None:
    hints = []
    s = (stderr_text or "").lower()

    if "pmi" in s or "pmix" in s or "pmi2" in s:
        hints.append(
            "MPI/PMI handshake issue detected. Check cluster MPI integration and consider configurable --mpi=<pmix|pmi2>."
        )
    if "cpu bind" in s or "cpuset" in s or "cgroup" in s:
        hints.append(
            "CPU binding/cgroup issue detected. Verify cpus-per-task, OMP settings, and site binding policy."
        )
    if "invalid generic resource" in s or "gres" in s or "gpu" in s:
        hints.append(
            "GPU/GRES mismatch detected. Verify sbatch GPU request and SLURM GPU environment variables."
        )

    detail = "\n".join(f"- {h}" for h in hints) if hints else "- No known pattern matched."
    raise RuntimeError(
        "srun launch failed.\n"
        f"Run mode: {run_mode}\n"
        f"Command: {cmd}\n"
        f"Known diagnostics:\n{detail}\n"
        f"Raw stderr:\n{stderr_text}"
    )
```

---

## Implementation Plan (when ready to code)

1. **`run_simulation.py`** — update launch argument defaults:
   - Remove `--overlap` (both CPU and GPU command strings)
   - Switch `--cpu-bind=none` → `--cpu-bind=cores` (both command strings)

2. **`resource_management.py`** — add `_parse_slurm_allocated_gpus()` utility (Section A above).

3. **`run_simulation.py`** — add GPU preflight validation block (Section B above), after the existing CPU preflight (lines 461-503). Import `_parse_slurm_allocated_gpus` from `resource_management`.

4. **`run_simulation_runner.py`** — add `_raise_enriched_srun_error()` (Section D above). Change `stderr=subprocess.STDOUT` → `stderr=subprocess.PIPE` in the `Popen` call, add `stderr_text = proc.stderr.read().decode("utf-8", errors="replace")` after `proc.wait()`, then call `_raise_enriched_srun_error(stderr_text, run_mode=run_mode, cmd=" ".join(cmd))` when `_rc != 0`.

5. **Tests** — add/extend for:
   - CPU mode command string (assert no `--overlap`, assert `--cpu-bind=cores`)
   - GPU mode command string (assert no `--overlap`, assert `--cpu-bind=cores`, assert `--ntasks-per-gpu=1`)
   - CPU mismatch raises `RuntimeError`
   - GPU mismatch raises `RuntimeError` (when SLURM env vars are present)
   - `_parse_slurm_allocated_gpus()` returns correct count for each var variant
   - `_parse_slurm_allocated_gpus()` prints diagnostic when no vars found

6. **Field validation** on UVA cluster:
   - Run one CPU/hybrid case — confirm no regression
   - Run one GPU case — confirm no regression
   - Check runner logs for `[GPU-PREFLIGHT]` message to identify which SLURM GPU vars are present

---

## Acceptance Criteria

- `srun` command no longer includes `--overlap`.
- Default binding is `--cpu-bind=cores`.
- GPU mode keeps `--ntasks-per-gpu=1` and does not require `--gpus-per-task`.
- Preflight fails fast on detectable CPU/GPU under-allocation with actionable errors.
- If `srun` fails, raised error contains enough context to diagnose (binding vs PMI vs GPU/GRES).

---

## Open Questions — Status

1. **Which SLURM GPU env vars are present on UVA partitions?**
   Deferred to field validation. `_parse_slurm_allocated_gpus()` will emit a `[GPU-PREFLIGHT]` print to stdout on the first run listing exactly which variables were tried and what they returned. Check runner logs after the first GPU job to resolve this.

2. **Does `srun` require `--mpi=pmix` on any target partition?**
   No known requirement. Do not add preemptively. `_raise_enriched_srun_error()` will surface a PMI hint if a failure occurs. Only add `--mpi=<value>` as an explicit config toggle after a confirmed failure.

3. **Does `--cpu-bind=cores` conflict with any site-enforced affinity policy?**
   Unknown — requires field testing. `_raise_enriched_srun_error()` will surface a binding hint if the launch fails. If a conflict is confirmed, add a `cpu_bind` config field (default `"cores"`) as a targeted override.

If issues appear, we keep defaults strict and add **explicit config toggles** rather than silently mutating launch behavior.
