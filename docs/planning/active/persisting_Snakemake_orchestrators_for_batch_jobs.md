# Persisting Snakemake Orchestrators for Batch Jobs

**Status:** Implemented (Options A + E)
**Priority:** Medium
**Created:** 2026-02-20
**Implemented:** 2026-02-20

---

## Problem Statement

When `multi_sim_run_method = batch_job`, the toolkit launches Snakemake inside a **detached tmux session** on the HPC login node. The session name is persisted in the analysis log so users can reattach later. However, most HPC clusters — including UVA HPC via `login.hpc.virginia.edu` — use a **round-robin load balancer** across multiple login nodes.

This means:

1. User submits the workflow on `login1.hpc.virginia.edu` → tmux session is created there
2. User later SSHs to `login.hpc.virginia.edu` → lands on `login2.hpc.virginia.edu`
3. `tmux list-sessions` returns nothing — the session exists on a different machine
4. User cannot reattach, and the post-submission reattach command is silently useless

The toolkit currently emits reattach commands of the form:
```
tmux attach -t triton_swmm_norfolk_20260220_143022
```

This only works if the user is already on the correct login node. There is no mechanism to determine which node to target.

---

## Current Architecture Summary

### Submission Flow

`_submit_tmux_workflow()` (workflow.py ~line 1873) orchestrates the following:

1. Generates a unique session name (e.g., `triton_swmm_{study_name}_{timestamp}`)
2. Constructs a `tmux new-session -d -s {name}` command
3. Sends the Snakemake invocation into the tmux pane
4. Persists the session name to `analysis.log` via `tmux_session_name` log field
5. Emits post-submission user hints (~line 2126) including the reattach command

### Log Persistence

In `TRITONSWMM_analysis_log` (log.py ~line 574):

```python
tmux_session_name: LogField[str] = Field(default_factory=LogField)
```

This field is populated at submission time and can be read back on subsequent calls to find an active session.

### What Is Missing

- **No record of which login node** the session was started on
- Post-submission hints use a bare `tmux attach` command — correct only if the user is already on the same node
- No validation warning that round-robin load balancing may cause reattach to fail

---

## Solution Options

### Option A: Login-Node Pinning via Config Field

Add an optional `hpc_login_node: Optional[str]` config field (e.g., `"login1.hpc.virginia.edu"`).

- Reattach commands become: `ssh login1.hpc.virginia.edu -t tmux attach -t {session}`
- Post-submission hints use the config value if set
- If field unset → falls back to current behavior with a warning

**Pros:**
- Simple to implement
- Transparent to the user
- No architectural change

**Cons:**
- Requires user to know and specify their login node
- Not zero-config; easy to forget

---

### Option B: Run Snakemake Itself Inside a SLURM Job (Deprecated / Not Recommended)

Submit the Snakemake orchestrator via `sbatch` as a long-running SLURM job (not a compute job — just an orchestrator on a login or interactive partition). The orchestrator job runs Snakemake which submits child jobs.

> **Note:** Snakemake officially discourages running the orchestrator inside a SLURM job — it can interfere with environment variable inheritance, proper resource accounting, and child-job submission. The supporting functions exist in the codebase but are left deprecated and should not be used.

**Pros:**
- Fully node-agnostic; no tmux needed
- Survives login node restarts or disconnections
- Most portable across clusters

**Cons:**
- Snakemake recommends against it — environment inheritance and resource allocation can break
- Significant architectural change if ever revisited
- Orchestrator job must target a partition with sufficient wall time and no compute resource constraints

---

### Option C: nohup Background Process

Replace tmux with `nohup python -m snakemake ... &` on the login node. Store the PID in a `snakemake.pid` file.

**Pros:**
- No tmux dependency
- Simpler process management
- Survives logout

**Cons:**
- Still tied to the originating login node
- Cannot interactively attach to monitor progress
- Harder to kill or inspect

---

### Option D: FastX / GUI Session

UVA HPC supports FastX for persistent desktop sessions. Snakemake would run in a FastX terminal that persists across reconnections.

**Pros:**
- User-friendly for interactive use
- Persistent across reconnections (FastX sessions are stable)

**Cons:**
- Cluster-specific (not portable to Frontier or other systems)
- Requires manual GUI interaction
- Not automatable by the toolkit

---

### Option E: Auto-Detect and Store Login Node Hostname (Recommended Short Term)

At the moment of `_submit_tmux_workflow()` invocation, capture the current hostname via `socket.gethostname()` and persist it in `analysis.log` as a new `workflow_submission_node` log field. Use this stored hostname in post-submission reattach commands and when reconstructing reattach hints on subsequent toolkit calls.

No config field is required. The node name is captured automatically at submission time.

**Pros:**
- Zero-config; works transparently
- Requires minimal code changes (log field + one `socket.gethostname()` call + updated hint formatting)
- Reattach commands are always correct if user follows the hint

**Cons:**
- Node name must be captured at submission time; if the log is read on a fresh session without the original node available, user must SSH to the right node manually
- Does not prevent the disconnect — just makes the correct reattach command visible

---

## Recommended Approach

### Short Term (Low Effort)

**Option E** — store the login node hostname in `analysis.log` at submission time.

```python
import socket
# In _submit_tmux_workflow():
submission_node = socket.gethostname()
self.log.workflow_submission_node.set(submission_node)
```

Then emit reattach commands that include the node:

```
ssh login1.hpc.virginia.edu -t tmux attach -t triton_swmm_norfolk_20260220_143022
```

This requires no user configuration and no architectural change.

### Medium Term

**Option A** — add optional `hpc_login_node` config field so users can proactively pin to a specific node. The toolkit uses this value for reattach hints and (optionally) when creating the initial session via SSH-forwarded commands. When both `hpc_login_node` (config) and `workflow_submission_node` (log) are available, prefer `hpc_login_node` for future sessions and `workflow_submission_node` for existing ones.

### Long Term

**Option B** (SLURM-based orchestrator) — only if HPC policies outright ban long-running processes on login nodes. Snakemake discourages this pattern due to environment inheritance and resource allocation issues; supporting functions exist in the codebase but are deprecated.

---

## Config Field (Implemented — Option A)

In `analysis_config.yaml`:

```yaml
hpc_login_node: "login1.hpc.virginia.edu"  # optional; only needed for tmux reattach
```

Pydantic field in `config/analysis.py` (implemented):

```python
hpc_login_node: Optional[str] = Field(
    default=None,
    description=(
        "Specific HPC login node hostname for tmux session reattach. "
        "Only needed if the cluster uses round-robin login load balancing "
        "(e.g., UVA HPC). Example: 'login1.hpc.virginia.edu'. "
        "If unset, the submission node is auto-detected at workflow launch time."
    ),
)
```

**Validation behavior:** emits a warning (not error) if `multi_sim_run_method == "batch_job"` and `hpc_login_node` is None.

---

## Log Field (Implemented — Option E)

In `TRITONSWMM_analysis_log` (log.py), alongside `tmux_session_name` (implemented):

```python
workflow_submission_node: LogField[str] = Field(default_factory=LogField)  # login node hostname at submission time
```

Populated in `_submit_tmux_workflow()`:

```python
import socket
submission_node = socket.gethostname()
self.analysis.log.workflow_submission_node.set(submission_node)
```

---

## Reattach Commands (Implemented)

**On HPC (when `module load tmux` is required):**
```
[Snakemake]   Attach to session: ssh login1.hpc.virginia.edu -t 'module load tmux && tmux attach -t triton_swmm_norfolk_20260220_143022'
[Snakemake]   Kill this session: ssh login1.hpc.virginia.edu -t 'module load tmux && tmux kill-session -t triton_swmm_norfolk_20260220_143022'
[Snakemake]   List all sessions: ssh login1.hpc.virginia.edu -t 'module load tmux && tmux list-sessions'
```

The `module load tmux &&` prefix is included because a fresh SSH session on HPC typically has no modules loaded, and tmux is often not in the default PATH.

Node resolution order: `hpc_login_node` config → `socket.gethostname()` at submission time (auto-detected).

**On local machine (no module load needed):**
```
[Snakemake]   Attach to session: tmux attach -t triton_swmm_norfolk_20260220_143022
```

---

## Implementation Checklist

- [x] Add `workflow_submission_node: LogField[str]` to `TRITONSWMM_analysis_log` in `log.py`
- [x] Import `socket` and populate `workflow_submission_node` with `socket.gethostname()` in `_submit_tmux_workflow()` (workflow.py)
- [x] Update post-submission user message formatting to use node-aware reattach command (including `module load tmux` on HPC)
- [x] Add optional `hpc_login_node: Optional[str]` field to `AnalysisConfig` in `config/analysis.py`
- [x] Add validation warning in `_validate_hpc_configuration()` (validation.py) when `batch_job` mode + `hpc_login_node` is None
- [ ] Update `docs/planning/priorities.md` with this work item (note: item resolved/implemented)

---

## Relevant Files

| File | Relevance |
|------|-----------|
| `src/TRITON_SWMM_toolkit/workflow.py` | `_submit_tmux_workflow()` ~line 1873; post-submission hints ~line 2126 |
| `src/TRITON_SWMM_toolkit/log.py` | `TRITONSWMM_analysis_log`; `tmux_session_name` ~line 574 |
| `src/TRITON_SWMM_toolkit/config/analysis.py` | Analysis config model; batch_job validators ~line 372 |
| `src/TRITON_SWMM_toolkit/validation.py` | `_validate_hpc_configuration()` ~line 563 |
| `docs/planning/priorities.md` | Development priorities tracking |
