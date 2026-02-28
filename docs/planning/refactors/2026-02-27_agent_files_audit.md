# Agent Files Audit and Refresh Plan

**Written**: 2026-02-27
**Last Edit**: 2026-02-27 — initial draft

---

## Task Understanding

### Requirements

Audit all eight `.claude/agents/` specialist agent files for staleness and structural deficiencies, then define a standard structure and phased update plan. The secondary goal is to add explicit "startup reads" to agents for which curated, always-current file content matters most — HPC/SLURM integration and Snakemake workflow are the priority candidates.

### Assumptions

- Agent files become the system prompt for subagents spawned via the Task tool. Stale content in these prompts causes agents to give wrong advice and erodes developer trust.
- "Startup reads" means the agent body explicitly instructs the spawned subagent to read specific source files at the start of every invocation. This is the primary mechanism for giving agents always-current specialist context without bloating CLAUDE.md.
- The developer will review and approve this plan before any files are edited.
- Backward compatibility is explicitly NOT a concern per project philosophy; cruft should be deleted.

### Success Criteria

1. Each agent file has a consistent, agreed-upon structure (front-matter, "When to invoke", startup reads, curated reference content).
2. Stale claims (wrong class names, wrong API surface, wrong philosophy guidance) are corrected or eliminated.
3. At least two pilot agents are rewritten as exemplars before rolling out to the rest.
4. Developer confidence in invoking agents is restored.

---

## Evidence from Codebase

### Key Findings

**Naming mismatches (actual code vs. agent text):**

| Agent | What Agent Says | Actual Code |
|---|---|---|
| `pydantic-config-specialist` | `SystemConfig`, `AnalysisConfig` as class names | Actual classes are snake_case module names in `config/system.py` and `config/analysis.py`; Pydantic model is `cfgBaseModel` |
| `hpc-slurm-integration` | `many_jobs_1_srun_task_each` as an execution mode | Does not exist. Actual Literal: `"local" \| "batch_job" \| "1_job_many_srun_tasks"` in `analysis_config` |
| `snakemake-workflow` | "Three-Phase Workflow Structure" | CLAUDE.md documents five distinct phases (Setup, Scenario Preparation, Simulation Execution, Output Processing, Consolidation) |
| `snakemake-workflow` | `generate_sensitivity_analysis_workflow()` as a method | Does not exist. Actual class is `SensitivityAnalysisWorkflowBuilder` (line 2657 of `workflow.py`) |
| `snakemake-workflow` | `rule simulate` in code snippet | Actual templates use `rule run_{model_type}` (e.g., `run_tritonswmm`, `run_triton`, `run_swmm`) |
| `swmm-model-generation` | `config.py` as module path for toggles | Config is a package: `config/analysis.py` |
| `triton-debugger` | `log_context` context manager | Not found in `log.py`; actual pattern is `LogField[T]` JSON-persistence |

**Philosophy violations (agent gives wrong advice per project philosophy):**

| Agent | Violation |
|---|---|
| `pydantic-config-specialist` | "Backward Compatibility Checklist" includes "Renamed fields should support aliases temporarily" and "Removed fields should emit deprecation warnings first" — directly contradicts project philosophy |
| `swmm-model-generation` | "Don't break existing model generation without migration path" — same violation |
| `output-processing` | "ensure backwards compatibility when possible" — same violation |

**Missing critical context:**

- `hpc-slurm-integration`: Missing `MPICH_OFI_NIC_POLICY=BLOCK` requirement for Frontier MPI/hybrid runs (recent hard-won fix). Missing `platform_configs.py` and `profile_catalog.py`.
- `snakemake-workflow`: Missing `SnakemakeWorkflowBuilder` class name. Missing multi-model rule pattern (`run_triton`, `run_tritonswmm`, `run_swmm`). No mention of `subprocess_utils.py` and `run_subprocess_with_tee()` pattern.
- `sensitivity-analysis`: Missing actual class name `TRITONSWMM_sensitivity_analysis`. Missing GPU constraint gotcha from CLAUDE.md.
- All agents: No agent currently instructs the spawned subagent to read any source files at startup. All knowledge is static text that becomes stale as code evolves.

**What agents do well:**
- Front-matter `description` blocks with `<example>` tags are well-written and useful — keep these.
- High-level operational intent is generally sound.
- Domain knowledge content (SWMM .inp format, dask/zarr patterns) is accurate.
- `triton-test-suite` is the most accurate agent relative to current codebase.

---

## Standard Agent File Structure

Every agent file should follow this structure after the refresh:

```
---
name: <agent-name>
description: "<When to invoke description with examples>"
model: sonnet
---

## When to Invoke
[2-4 sentence summary of what tasks this agent handles.]

## Startup Reads
Before proceeding with any task, read these files to get current codebase state:
- `<absolute path>` — [why this file matters]
[3-6 files max; quality over quantity]

## Key Patterns and Conventions
[Curated, accurate reference content: class names, method signatures, config field names,
gotchas, architectural decisions.]

## Operational Checklist
[Short numbered list of "before you submit changes, verify..." items.]

## What NOT to Do
[Explicit anti-patterns specific to this subsystem, including philosophy violations.]
```

**Rationale for each section:**
- "Startup Reads" is the key new addition — gives agents access to ground-truth current code, making them durable against codebase evolution.
- "Key Patterns and Conventions" replaces long prose blocks with tight, verifiable reference material.
- "What NOT to Do" captures philosophy violations as explicit prohibitions so agents cannot drift into bad advice.

---

## Per-Agent Assessment Table

| Agent File | Lines | Staleness | Primary Issues | Startup Reads | Action |
|---|---|---|---|---|---|
| `hpc-slurm-integration.md` | 141 | HIGH | Wrong mode name (`many_jobs_1_srun_task_each`). Missing Frontier NIC policy. Missing `platform_configs.py`. | `execution.py`, `resource_management.py`, `platform_configs.py`, `config/analysis.py` | **REWRITE (Pilot 1)** |
| `snakemake-workflow.md` | 149 | HIGH | Wrong phase count (3 vs 5). Wrong rule name. Fabricated method name. No class names. | `workflow.py` (first 250 lines), `subprocess_utils.py` | **REWRITE (Pilot 2)** |
| `pydantic-config-specialist.md` | 114 | MEDIUM | Wrong class names. Backward compat philosophy violation. | `config/base.py`, `config/system.py`, `config/analysis.py`, `config/loaders.py` | **UPDATE (Phase 2)** |
| `sensitivity-analysis.md` | 105 | MEDIUM | Missing class name `TRITONSWMM_sensitivity_analysis`. Missing GPU constraint gotcha. | `sensitivity_analysis.py` (first 100 lines) | **UPDATE (Phase 2)** |
| `triton-debugger.md` | 117 | MEDIUM | Stale `log_context` reference. Otherwise good structure. | `log.py`, `exceptions.py` | **UPDATE (Phase 2)** |
| `triton-test-suite.md` | 176 | LOW | Most accurate agent. Missing some newer assertion helpers. | `tests/utils_for_testing.py` | **LIGHT UPDATE (Phase 2)** |
| `swmm-model-generation.md` | 212 | LOW | Backward compat philosophy violation. Config module path wrong. | `swmm_full_model.py`, `scenario_inputs.py`, `swmm_runoff_modeling.py` | **LIGHT UPDATE (Phase 2)** |
| `output-processing.md` | 81 | LOW | Backward compat philosophy violation. Otherwise accurate. | `swmm_output_parser.py`, `process_simulation.py`, `processing_analysis.py` | **LIGHT UPDATE (Phase 2)** |

---

## File-by-File Change Plan

### Phase 1: Pilot Rewrites

**`hpc-slurm-integration.md` — Full Rewrite**

Remove:
- `many_jobs_1_srun_task_each` mode description (mode name does not exist in code).
- Generic SLURM environment variable table (generic HPC knowledge, not codebase-specific).
- "Communication Style" section (low value).

Add:
- Startup reads: `execution.py`, `resource_management.py`, `config/analysis.py`, `platform_configs.py`.
- Correct execution mode names from `analysis_config` Literal: `"local"`, `"batch_job"`, `"1_job_many_srun_tasks"`.
- Note `batch_job` = Snakemake submitting SBATCH per sim (not a SLURM array).
- Frontier gotcha: `MPICH_OFI_NIC_POLICY=BLOCK` required for `mpi`/`hybrid` run modes.
- Mention `profile_catalog.py` and `platform_configs.py` as cluster-specific config files.
- "What NOT to Do": no deprecation wrappers, no backward compat shims.

**`snakemake-workflow.md` — Full Rewrite**

Remove:
- "Three-Phase Workflow Structure" section.
- `rule simulate` code snippet.
- `generate_sensitivity_analysis_workflow()` method name.
- "Code Quality Standards" section (generic advice).

Add:
- Startup reads: `workflow.py` (first ~250 lines for class overview), `subprocess_utils.py`.
- `SnakemakeWorkflowBuilder` and `SensitivityAnalysisWorkflowBuilder` (line 2657) as the two key classes.
- Five-phase workflow accurately described: Setup, Scenario Preparation, Simulation Execution (per model type), Output Processing (per model type), Consolidation.
- Multi-model rule naming pattern: `rule run_triton`, `rule run_tritonswmm`, `rule run_swmm` dynamically generated via f-string with `{model_type}`.
- "What NOT to Do": do not invent method names; read the actual class before proposing changes.

---

### Phase 2: Remaining Six Agents

- **`pydantic-config-specialist`**: Fix class names. Remove entire "Backward Compatibility Checklist". Add startup reads. Add "What NOT to Do".
- **`sensitivity-analysis`**: Add class name `TRITONSWMM_sensitivity_analysis`. Add GPU constraint gotcha. Add startup reads.
- **`triton-debugger`**: Remove/correct stale `log_context` reference; replace with `LogField[T]` pattern. Add startup reads.
- **`triton-test-suite`**: Add startup reads. Expand assertion helper table with newer helpers.
- **`swmm-model-generation`**: Fix config module path. Remove backward compat item. Add startup reads and "What NOT to Do".
- **`output-processing`**: Remove backward compat line. Add startup reads and "What NOT to Do". Expand from 81 lines with more concrete reference material.

---

## Risks and Edge Cases

- **Startup reads slow agent startup**: `workflow.py` is 3400 lines. Mitigation: use selective line ranges; read class overviews only.
- **Startup reads become stale on file renames**: A broken path is easier to notice than a wrong fact in prose. Mitigation: update "When to Update Agent Documentation" in CLAUDE.md to explicitly include startup read paths.
- **Removing content that was actually correct**: Some "generic" content may have been intentional. Mitigation: developer reviews each Phase 1 deletion before finalizing.

---

## Validation Plan

After each pilot rewrite:
1. Manually verify: no stale mode names, no backward compat advice, startup reads point to real files, five workflow phases in snakemake agent.
2. Invoke the rewritten agent on a real task:
   - `hpc-slurm-integration`: Ask it to explain the srun command structure for a Frontier `1_job_many_srun_tasks` run.
   - `snakemake-workflow`: Ask it to describe what rules are generated for a multi-model analysis.
3. Verify the agent reads startup files before answering.

After Phase 2:
4. Run `grep -r "many_jobs_1_srun_task_each\|SystemConfig\|AnalysisConfig\|three-phase\|backwards compat" .claude/agents/` to confirm all known staleness markers are removed.

No smoke tests needed — documentation-only change.

---

## Decisions Needed from User

1. **Startup read granularity for `workflow.py`**: File is 3400 lines. Read whole file (thorough but slow) or targeted subset (lines 1-250 for class overview plus specific method ranges)? *Recommendation: targeted subset — class overviews are sufficient for most tasks.* (Risk: low)

2. **Pilot approval gate**: Should the developer review and explicitly approve each pilot rewrite before Phase 2 begins, or is one review at the end of Phase 1 sufficient? *Recommendation: approve each pilot — getting the template right matters for all subsequent agents.* (Risk: medium)

3. **Phase 2 order**: Suggested order: `pydantic-config-specialist` and `sensitivity-analysis` first (most concrete errors), then `triton-debugger`, then light updates. Confirm or reorder. (Risk: low)

---

## CLAUDE.md Content Awaiting Agent Migration

The following sections are currently in CLAUDE.md but belong in agent files once those files are refreshed. They should stay in CLAUDE.md until the relevant agent is updated, then be trimmed from CLAUDE.md.

| CLAUDE.md Section | Target Agent | Notes |
|---|---|---|
| Multi-Model Integration — directory tree | `snakemake-workflow.md` | Per-scenario directory structure belongs with workflow documentation |
| Multi-Model Integration — Workflow Rules pseudocode | `snakemake-workflow.md` | Rule naming pattern (`run_triton`, `run_tritonswmm`, `run_swmm`) |
| Multi-Model Integration — Status Tracking | `snakemake-workflow.md` | `df_status["model_types_enabled"]` pattern |
| HPC 1-Job-Many-srun-Tasks details | `hpc-slurm-integration.md` | Currently trimmed to 2 lines in CLAUDE.md; expand in agent |

---

## Future Consideration: `architecture.md` Modular Approach

**Idea**: Extract all architecture content from CLAUDE.md into a dedicated `.prompts/architecture.md`, then hook it from both CLAUDE.md (via the "Read before beginning" directive) and `conventions.md`. This would:
- Keep CLAUDE.md focused on tooling, commands, and quick-reference material
- Make architecture.md independently loadable for deep dives
- Reduce CLAUDE.md length significantly

**When to act**: Evaluate after agent file audit is complete. If CLAUDE.md is still bloated after agent migration, create a formal plan for architecture.md.

---

## Definition of Done

### Phase 1 — Pilot Rewrites
- [ ] Standard agent file template agreed upon (this document)
- [ ] `hpc-slurm-integration.md` rewritten; verified no stale mode names, no backward compat advice, startup reads present
- [ ] `snakemake-workflow.md` rewritten; verified five-phase workflow, correct class names, correct rule pattern, startup reads present
- [ ] Developer has invoked each pilot agent on a real task and confirmed improved quality

### Phase 2 — Remaining Six Agents
- [ ] All six remaining agents updated with correct class/method names, startup reads, and no backward compat advice
- [ ] `grep` check confirms all known staleness markers removed from `.claude/agents/`
- [ ] CLAUDE.md "When to Update Agent Documentation" updated to mention startup read paths
- [ ] CLAUDE.md staleness notice in "Specialized Agent Documentation" removed
- [ ] Sections listed in "CLAUDE.md Content Awaiting Agent Migration" trimmed from CLAUDE.md
- [ ] `docs/planning/features/2026-02-07_priorities.md` updated
