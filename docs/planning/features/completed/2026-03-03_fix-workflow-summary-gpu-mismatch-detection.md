# Remove Resource Allocation Diagnostics Section from workflow_summary.md

**Created**: 2026-03-03 12:12
**Last edited**: 2026-03-03 14:05 — completed

---

## What Was Built

Deleted `gather_resource_allocation_diagnostics()` (97 lines) from `export_scenario_status.py` and its 6-line call site in `write_workflow_summary_md()`. The function parsed simulation rules from the Snakefile and flagged `threads ≠ tasks × cpus_per_task` as `⚠️ MISMATCH` — a false alarm for GPU rules where `threads=n_gpus` is structurally correct (`threads` is overridden by `cpus_per_task=1` in the SLURM executor's CPU priority chain). Removed rather than patched to eliminate ongoing maintenance burden.

Also fixed 4 pre-existing ruff violations encountered while touching the file: import ordering, `Optional[X]→X|None`, f-string without placeholder, line too long.

`gather_hpc_partition_info()` and `parse_partition_limits()` are untouched.

---

## Files Changed

| File | Change |
|------|--------|
| `src/TRITON_SWMM_toolkit/export_scenario_status.py` | Deleted `gather_resource_allocation_diagnostics()` and its call site; fixed 4 pre-existing ruff violations |

---

## Definition of Done

- [x] `gather_resource_allocation_diagnostics()` deleted from `export_scenario_status.py`
- [x] Call site removed from `write_workflow_summary_md()`
- [x] `ruff check` and `ruff format` pass
- [x] PC_04 smoke test passes (confirmed via other agent run 2026-03-03 14:00)
