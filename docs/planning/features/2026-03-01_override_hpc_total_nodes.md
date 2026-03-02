# Implementation Plan: `override_hpc_total_nodes` Parameter for `submit_workflow()`

**Written**: 2026-03-01
**Last edited**: 2026-03-01 — codebase verification pass; added assert clarification for `_generate_single_job_submission_script()` and delegation note for `SensitivityAnalysisWorkflowBuilder`

---

## Task Understanding

### Requirements

Add an `override_hpc_total_nodes: int | None = None` parameter to `submit_workflow()` on both `SnakemakeWorkflowBuilder` and `SensitivityAnalysisWorkflowBuilder`. When provided, it replaces `cfg_analysis.hpc_total_nodes` for the purpose of sbatch script generation — without mutating the config object or the YAML on disk.

If `override_hpc_total_nodes` is passed when `multi_sim_run_method != "1_job_many_srun_tasks"`, the method must raise a `ConfigurationError` immediately (fail-fast).

### Assumptions

- `None` (default) means "use config as-is" — no change to current behavior.
- The override is passed through to `_generate_single_job_submission_script()`, which is the only place `hpc_total_nodes` is read.
- The override does **not** affect `_validate_single_job_dry_run()`, which uses `hpc_cpus_per_node * hpc_total_nodes` for CPU budget checks. Since the dry-run validation is a pre-flight guard (not the actual submission), it should also respect the override — otherwise the dry-run check will be wrong. The override must be threaded through to the dry-run validator as well.
- No persistence: the YAML on disk and the `cfg_analysis` object are not modified.
- Default of `None` is appropriate here because the flag has no meaning outside `1_job_many_srun_tasks` mode — it is intentionally mode-gated.

### Success Criteria

- `submit_workflow(override_hpc_total_nodes=5)` generates `#SBATCH --nodes=5` even when `cfg_analysis.hpc_total_nodes=50`.
- `submit_workflow(override_hpc_total_nodes=5)` when `multi_sim_run_method="local"` raises `ConfigurationError` immediately, before any workflow generation.
- No change to behavior when `override_hpc_total_nodes=None` (default).
- Both `SnakemakeWorkflowBuilder` and `SensitivityAnalysisWorkflowBuilder` support the parameter.
- Test coverage added.

---

## Evidence from Codebase

- **`workflow.py:753`** — `_generate_single_job_submission_script(snakefile_path, config_dir)`: reads `self.cfg_analysis.hpc_total_nodes` at line 781. This is the single write point for `--nodes=` in the sbatch script.
- **`workflow.py:2425`** — `SnakemakeWorkflowBuilder.submit_workflow()`: 18 parameters. Calls `_submit_single_job_workflow()` for `1_job_many_srun_tasks` mode.
- **`workflow.py:3088`** — `SensitivityAnalysisWorkflowBuilder.submit_workflow()`: separate class (not a subclass of `SnakemakeWorkflowBuilder`), holds `_base_builder` instance. Calls `self._base_builder._validate_single_job_dry_run()` at line 3219 and `self._base_builder._submit_single_job_workflow()` at line 3231.
- **`workflow.py:1054`** — `_validate_single_job_dry_run()`: uses `hpc_cpus_per_node * hpc_total_nodes` for CPU budget at line 1098. Must also receive the override.
- **`workflow.py:1519`** — `_submit_single_job_workflow()`: calls `_generate_single_job_submission_script()` at line 1569. This is the call chain threading point.
- **`config/analysis.py:60`** — `hpc_total_nodes: Optional[int]` — field definition.
- **`exceptions.py`** — `ConfigurationError(field, message, config_path)` — the right exception type for mode/parameter mismatch.
- **`tests/test_workflow_1job_sbatch_generation.py`** — existing tests for sbatch generation. Test for `override_hpc_total_nodes` belongs here.

---

## Implementation Strategy

### Chosen approach: thread `override_hpc_total_nodes` through the call chain

1. Add `override_hpc_total_nodes: int | None = None` to both `submit_workflow()` methods.
2. At the top of `submit_workflow()`, before any workflow generation, validate: if the override is set and `multi_sim_run_method != "1_job_many_srun_tasks"`, raise `ConfigurationError`.
3. Pass the override down the call chain: `submit_workflow()` → `_submit_single_job_workflow()` → `_validate_single_job_dry_run()` and `_generate_single_job_submission_script()`.
4. In `_generate_single_job_submission_script()`, replace:
   ```python
   total_nodes = self.cfg_analysis.hpc_total_nodes
   ```
   with:
   ```python
   total_nodes = override_hpc_total_nodes if override_hpc_total_nodes is not None else self.cfg_analysis.hpc_total_nodes
   ```
5. Apply the same pattern in `_validate_single_job_dry_run()` where it uses `hpc_total_nodes`.

### Alternatives considered

- **Mutate config before calling** (`workflow.cfg_analysis.hpc_total_nodes = 5`): Already works today. Rejected because it mutates live state unexpectedly and isn't discoverable.
- **Separate `resubmit()` method**: Cleaner conceptually but more surface area for a narrow use case.

### Trade-offs

The override parameter must be threaded through `_submit_single_job_workflow()` and into both internal methods. This adds one extra parameter to two private methods, which is a minor increase in internal complexity — but keeps the public surface clean and the override non-destructive.

---

## File-by-File Change Plan

### `src/TRITON_SWMM_toolkit/workflow.py`

**1. `SnakemakeWorkflowBuilder.submit_workflow()`**
- Add `override_hpc_total_nodes: int | None = None` to signature.
- Add early validation guard (before any snakefile generation):
  ```python
  if override_hpc_total_nodes is not None and self.cfg_analysis.multi_sim_run_method != "1_job_many_srun_tasks":
      raise ConfigurationError(
          field="override_hpc_total_nodes",
          message=f"override_hpc_total_nodes is only valid when multi_sim_run_method='1_job_many_srun_tasks', "
                  f"but current method is '{self.cfg_analysis.multi_sim_run_method}'.",
          config_path=None,
      )
  ```
- Pass `override_hpc_total_nodes` to `_submit_single_job_workflow()`.

**2. `SensitivityAnalysisWorkflowBuilder.submit_workflow()`**
- Same changes as above.
- Note: this builder does not inherit from `SnakemakeWorkflowBuilder` — it holds a `_base_builder` instance. The dry-run and single-job calls inside this method are `self._base_builder._validate_single_job_dry_run(...)` and `self._base_builder._submit_single_job_workflow(...)`. Pass `override_hpc_total_nodes` at both call sites.

**3. `_submit_single_job_workflow()`**
- Add `override_hpc_total_nodes: int | None = None` to signature.
- Pass to both `_validate_single_job_dry_run()` and `_generate_single_job_submission_script()`.

**4. `_validate_single_job_dry_run()`**
- Add `override_hpc_total_nodes: int | None = None` to signature.
- Replace `self.cfg_analysis.hpc_total_nodes` reads with:
  ```python
  total_nodes = override_hpc_total_nodes if override_hpc_total_nodes is not None else self.cfg_analysis.hpc_total_nodes
  ```

**5. `_generate_single_job_submission_script()`**
- Add `override_hpc_total_nodes: int | None = None` to signature.
- At line 781–782, the existing code is:
  ```python
  total_nodes = self.cfg_analysis.hpc_total_nodes
  assert isinstance(total_nodes, int), "hpc_total_nodes required for 1_job_many_srun_tasks mode"
  ```
  Replace with:
  ```python
  total_nodes = override_hpc_total_nodes if override_hpc_total_nodes is not None else self.cfg_analysis.hpc_total_nodes
  assert isinstance(total_nodes, int), "hpc_total_nodes required for 1_job_many_srun_tasks mode"
  ```
  The assert must follow the resolved value, not precede it.

### `tests/test_workflow_1job_sbatch_generation.py`

Add tests:
1. **`test_override_hpc_total_nodes`**: Verify that passing `override_hpc_total_nodes=3` when `hpc_total_nodes=50` generates `#SBATCH --nodes=3`.
2. **`test_override_hpc_total_nodes_wrong_mode`**: Verify `ConfigurationError` is raised immediately when `override_hpc_total_nodes` is passed with `multi_sim_run_method="local"`.

---

## Risks and Edge Cases

- **Dry-run CPU budget mismatch**: If `_validate_single_job_dry_run()` uses `hpc_total_nodes` for the CPU budget check but the override is not passed through, the validation will compute the wrong total. The override must reach the validator. *(Mitigated by threading it through `_submit_single_job_workflow()`.)*
- **`override_hpc_total_nodes=0`**: Falsy but not `None`. The `if override_hpc_total_nodes is not None` check handles this correctly — though `0` nodes would fail SLURM submission anyway. No special handling needed.
- **Sensitivity analysis**: `SensitivityAnalysisWorkflowBuilder` has its own `submit_workflow()` override. Both must be updated consistently.

---

## Validation Plan

No local smoke tests required — this change is isolated to HPC-only paths (SLURM script generation). Tests in `test_workflow_1job_sbatch_generation.py` run locally (they mock the SLURM environment via fixture).

```bash
# Run targeted test file
conda run -n triton_swmm_toolkit pytest tests/test_workflow_1job_sbatch_generation.py -v

# Ruff check
conda run -n triton_swmm_toolkit ruff check src/TRITON_SWMM_toolkit/workflow.py
conda run -n triton_swmm_toolkit ruff format src/TRITON_SWMM_toolkit/workflow.py --check
```

---

## Documentation and Tracker Updates

- No `architecture.md` update needed (no structural changes).
- Docstring on `submit_workflow()` should note that `override_hpc_total_nodes` is only valid for `1_job_many_srun_tasks` mode.

---

## Definition of Done

- [ ] `override_hpc_total_nodes` added to both `submit_workflow()` methods
- [ ] Early `ConfigurationError` guard added for wrong-mode usage
- [ ] Override threaded through `_submit_single_job_workflow()` → `_validate_single_job_dry_run()` and `_generate_single_job_submission_script()`
- [ ] `#SBATCH --nodes=` in generated script reflects the override value
- [ ] Two new tests added and passing in `test_workflow_1job_sbatch_generation.py`
- [ ] `ruff check` and `ruff format` pass
- [ ] `architecture.md` unchanged (no structural change)
- [ ] Before closing this plan: loop in developer, then implement the debugging skill update (`docs/planning/features/2026-03-01_debug_prompt_timeout_node_recommendation.md`) — that feature depends on this parameter existing
