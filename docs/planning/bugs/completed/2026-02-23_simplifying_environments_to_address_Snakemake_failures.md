# Bug Fix: Trim Subprocess Environment to Prevent ARG_MAX Overflow on HPC

**Date**: 2026-02-23
**Status**: Partially implemented — see Outcome section

---

## Problem

On UVA HPC, `module load` + `conda activate` accumulate a very large `PATH` and
`LD_LIBRARY_PATH` in the shell environment. The `snakemake-executor-plugin-slurm`
calls `scontrol` inheriting this full environment, which exceeds Linux's `ARG_MAX`
kernel limit and crashes with:

```
OSError: [Errno 7] Argument list too long
```

A secondary failure (`AttributeError: 'str' object has no attribute 'is_storage'`
in the `snakemake-executor-plugin-slurm` stack) was investigated and not reproduced —
likely a transient state issue.

---

## What Was Built

### Fix 1: `workflow.py` — PATH trim for Snakemake process (KEPT)

Both the tmux script generator and the single-job script generator wrap the
`python -m snakemake` invocation with an `env` prefix:

```bash
SLURM_BIN=$(dirname "$(command -v scontrol 2>/dev/null || echo "/opt/slurm/current/bin/scontrol")")
env PATH="${CONDA_PREFIX}/bin:${SLURM_BIN}:/usr/local/bin:/usr/bin:/usr/sbin:/bin" \
    LD_LIBRARY_PATH="${CONDA_PREFIX}/lib" \
    ${CONDA_PREFIX}/bin/python -m snakemake ...
```

This scopes a trimmed environment to just the Snakemake process, preventing the
`scontrol` ARG_MAX crash. The surrounding bash script retains the full environment.

**Preflight version snapshot**: Both generated scripts also write
`logs/snakemake_versions.txt` before each run (package versions + `env_size_bytes`
+ `path_length_chars`) for future debugging diagnosability.

**Commit**: `b688c51` (2026-02-23)

### Fix 2: `run_simulation.py` — PATH omission from env dict (KEPT)

`env["PATH"]` is intentionally omitted from the simulation subprocess env dict.
The `bash -lc` (login shell) rebuilds PATH from `/etc/profile`; the `module load`
inside the command string adds HPC paths correctly. Copying `os.environ["PATH"]`
would propagate the full accumulated module environment as a shell argument.

**Commit**: `b688c51` (2026-02-23)

### Fix 3: `run_simulation.py` — LD_LIBRARY_PATH (PARTIALLY REVERTED)

The original fix set `env["LD_LIBRARY_PATH"] = str(swmm_path)` (SWMM bin only).
This was **partially reverted** in commit `8cf071d` (2026-02-26) because it stripped
rocm/libfabric paths that Frontier compute nodes require but don't re-add via
`module load` when `LD_LIBRARY_PATH` is pre-set in `bash -lc`.

**Current behavior** (post-revert): `env["LD_LIBRARY_PATH"]` prepends `swmm_path`
to the full `og_env["LD_LIBRARY_PATH"]` (the SBATCH environment's full library path
set, already containing `/opt/rocm-6.2.4/lib`, `/opt/cray/libfabric/.../lib64`,
etc.). This is passed via `env=` dict to `Popen` — not as a shell argument — so it
does not hit ARG_MAX (env vars go through `execve()` environment vector, not the
argument list).

**Why the revert was correct**: The SBATCH script loads all required modules before
launching Snakemake, so `og_env["LD_LIBRARY_PATH"]` already contains the correct
HPC library paths. Stripping it to just SWMM bin broke Frontier runs.

**Commit implementing revert**: `8cf071d` (2026-02-26)

---

## Outcome

| Fix | Status |
|-----|--------|
| `workflow.py` PATH trim for Snakemake process | ✅ Kept — resolves `scontrol` ARG_MAX crash |
| `workflow.py` version snapshot (`snakemake_versions.txt`) | ✅ Kept |
| `run_simulation.py` PATH omission from env dict | ✅ Kept |
| `run_simulation.py` LD_LIBRARY_PATH trim to SWMM-only | ❌ Reverted — broke Frontier; full LD_LIBRARY_PATH inheritance restored |

The ARG_MAX crash was resolved for UVA HPC. Frontier required a different strategy:
full `LD_LIBRARY_PATH` inheritance via the `env=` dict (which bypasses ARG_MAX since
env vars don't go through the shell argument list).
