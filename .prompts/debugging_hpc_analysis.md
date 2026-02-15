# HPC Analysis Debugging Protocol

## Context

You are debugging a TRITON-SWMM analysis that failed on an HPC cluster (UVA or Frontier). The user has transferred analysis outputs via Globus to `.debugging/`, **EXCLUDING** `sims/` and `subanalyses/` directories due to size constraints (raw TRITON/TRITON-SWMM outputs can be multi-GB per simulation).

### What's Available

- **Configuration**: `cfg_system.yaml`, `cfg_analysis.yaml`
- **Status tracking**: `scenario_status.csv`, `_status/*.flag` completion markers
- **Master workflow logs**: `logs/_slurm_logs/workflow_batch_*.out` (SLURM stdout for entire workflow)
- **Snakemake logs**: `.snakemake/log/*.snakemake.log` (Snakemake orchestrator logs)
- **Rule-specific logs**: `logs/sims/*.log` (prepare, simulation, processing logs)
- **Per-rule SLURM logs**: `.snakemake/slurm_logs/rule_*/[event_iloc]/[job_id].log`
- **Performance reports**: `efficiency_report_*.csv`

**Note**: `dry_run_report.md` should NOT be used for debugging as it may pre-date the failed run.

### What's NOT Available (Unless Requested)

- `sims/` directory - individual simulation folders with model inputs/outputs
- `subanalyses/` directory - for sensitivity studies, each sub-analysis's full directory

**If needed**: You can request specific simulation folders. Ask user to transfer individual `sims/{event_id}/` directories, **excluding** `out_triton*/` and `out_tritonswmm/` subdirectories to reduce size.

---

## Systematic Debugging Workflow

### Step 1: Identify Execution Mode and System Setup

Read `cfg_analysis.yaml` and `cfg_system.yaml`:
- Execution mode: `multi_sim_run_method` (local, batch_job, 1_job_many_srun_tasks)
- Run mode: `run_mode` (serial, openmp, mpi, gpu, hybrid)
- Model types enabled: Check `cfg_system.yaml` for `toggle_triton_model`, `toggle_tritonswmm_model`, `toggle_swmm_model`
- For sensitivity studies: Note if this is a master config or sub-analysis config

### Step 2: Check Most Recent Master Workflow Log

**Primary source**: `logs/_slurm_logs/workflow_batch_[DATETIME]_[JOBID].out`

- Find the **most recent** file (latest datetime in filename)
- This is the master Snakefile log showing all rule executions and failures
- Look for:
  - Job submission messages: `Job X has been submitted with SLURM jobid Y`
  - Rule failures: `Error in rule`, `CalledProcessError`, `[FAILED]`
  - Incomplete jobs: Rules that started but never completed
  - Time limit errors: `CANCELLED DUE TO TIME LIMIT`

**Cross-reference**: The corresponding `.snakemake/log/[DATETIME].snakemake.log` should have the same datetime and contain similar information. If both exist, prefer `logs/_slurm_logs/*.out` as it's easier to read.

### Step 3: Identify Incomplete Rules

Check `_status/` directory for missing completion flags:

```bash
# Count completions by phase
ls _status/setup_complete.flag 2>/dev/null
ls _status/sims/*_prepared.flag 2>/dev/null | wc -l      # Scenario preparation
ls _status/sims/*_complete.flag 2>/dev/null | wc -l       # Simulations (triton, tritonswmm, swmm)
ls _status/sims/*_processed.flag 2>/dev/null | wc -l      # Processing
ls _status/consolidate_complete.flag 2>/dev/null          # Consolidation
```

**For sensitivity analyses**: Flags use pattern `prepare_sa{N}_evt{M}_complete.flag`, `simulation_sa{N}_evt{M}_complete.flag`, etc.

**Corroborate** with `scenario_status.csv`:

```python
import pandas as pd
df = pd.read_csv('scenario_status.csv')

# Basic status
print(f"Total scenarios: {len(df)}")
print(f"Setup complete: {df['scenario_setup'].sum()}")
print(f"Run complete: {df['run_completed'].sum()}")

# For sensitivity studies, check by sub-analysis
if 'sub_analysis_iloc' in df.columns:
    summary = df.groupby('sub_analysis_iloc').agg({
        'scenario_setup': 'sum',
        'run_completed': 'sum',
        'run_mode': 'first'
    })
    print(summary[summary['run_completed'] < summary['scenario_setup']])
```

### Step 4: Examine Rule-Specific Logs

For incomplete rules identified in Step 3, check `logs/sims/*.log`:

**Regular analysis naming**:
- `prepare_{event_iloc}.log`
- `triton_{event_iloc}.log` or `tritonswmm_{event_iloc}.log` or `swmm_{event_iloc}.log`
- `process_triton_{event_iloc}.log` or `process_tritonswmm_{event_iloc}.log` or `process_swmm_{event_iloc}.log`

**Sensitivity analysis naming**:
- `prepare_sa{N}_evt{M}.log`
- `simulation_sa{N}_evt{M}.log`
- `process_sa{N}_evt{M}.log`
- `consolidate_sa_{N}.log`

Generally good for finding Python errors and exceptions (`Traceback`, `Error:`)

### Step 5: Check Per-Rule SLURM Logs

Also check `.snakemake/slurm_logs/` (particularly relevant if SLURM errors are the cause of failure)

**Structure**: `.snakemake/slurm_logs/rule_{rule_name}/{event_iloc}/{job_id}.log`

**Regular analysis examples**:
- `.snakemake/slurm_logs/rule_run_triton/33/9322252.log`
- `.snakemake/slurm_logs/rule_process_triton/19/9322253.log`

**Sensitivity analysis examples**:
- `.snakemake/slurm_logs/rule_simulation_sa6_evt0/8974623.log`
- `.snakemake/slurm_logs/rule_prepare_sa37_evt0/8974183.log`

**Common failure patterns**:
- `CANCELLED ... DUE TO TIME LIMIT` → increase time allocation
- `Unable to allocate resources` → resource constraint (CPUs/GPUs unavailable)
- `QOSMaxGRESPerUser` → May indicate GPU model unavailability (not always a QOS limit)
- `srun: error:` → SLURM execution error
- `Provided cores: N` + `Rules claiming more threads will be scaled down` → **CRITICAL**: srun request may exceed SLURM allocation, causing infinite hang
- Empty log or truncated → job never started or was killed
- `CPU Efficiency: 0.0%` (in efficiency reports) → job timed out before productive execution

**CRITICAL CHECK FOR TIMEOUT FAILURES**:

When jobs timeout with 0% CPU efficiency, **always verify srun request matches SLURM allocation**:

1. Check SLURM log: `.snakemake/slurm_logs/rule_*/[job_id].log`
   ```
   Provided cores: N  # What SLURM actually allocated
   threads: M         # What Snakemake requested
   ```

2. Check simulation log for srun command: `logs/sims/simulation_*.log`
   ```bash
   srun --ntasks=X --cpus-per-task=Y  # Needs X*Y cores total
   ```

3. **If srun needs more cores than SLURM provided**: Job hangs indefinitely waiting for resources
   - Example: SLURM allocated 4 cores, srun requests 2×4=8 cores → infinite wait → timeout
   - Common with `standard` partition (MaxNodes=1) when nodes have limited availability
   - Solution: Use partition with guaranteed resources or reduce resource requests

### Step 6: Diagnose Root Cause

Based on evidence from Steps 1-5, determine:

1. **What failed**: Which rules/phases (preparation, simulation, processing, consolidation)
2. **How many failed**: Single event, subset, or all events?
3. **Failure pattern**:
   - Time limits
   - Resource unavailability
   - Configuration errors
   - Deterministic bugs
4. **For sensitivity studies**: Are failures concentrated in specific sub-analyses with particular resource configurations?

---

## Standard Operating Procedure

### Step 0: Check for Previous Debugging Reports

**ALWAYS check first** if previous debugging reports exist:

```bash
ls -lt debugging_report_*.md | head -5
```

If previous reports exist:
1. Read the most recent report to understand prior issues
2. Reference previous findings in your new report
3. Note which issues have been fixed vs. persist
4. Track progress across debugging rounds

### Step 1-6: Follow Systematic Debugging Workflow (see above)

### Step 7: Write Debugging Report

**REQUIRED**: After completing diagnosis, write a comprehensive markdown report to:

```
{analysis_dir}/debugging_report_{YYYYMMDD_HHMMSS}.md
```

The report **must** include:

#### Required Sections

1. **Executive Summary**
   - Status: Pass/Fail with completion statistics
   - Primary failure modes (1-3 sentence summary each)
   - Scope of impact

2. **Configuration Summary**
   - Relevant config excerpts (YAML format)
   - Execution mode, run mode, model types, time limits

3. **Detailed Failure Analysis**
   - One subsection per failure mode
   - **Include exact text/code excerpts from logs** (representative examples)
   - Pattern analysis (which configurations failed/succeeded)
   - Status tables showing affected sub-analyses

4. **Root Cause Analysis**
   - Hypotheses with supporting evidence
   - Investigation of deeper patterns (e.g., why hybrid jobs fail)

5. **Investigation Commands** (if applicable)
   - Provide actual commands user can run on login node
   - Group by investigation type (QOS limits, partition configs, test submissions)

6. **Recommended Fixes**
   - Specific, actionable changes with code snippets
   - Rationale and expected impact for each fix

7. **Next Steps**
   - Immediate actions before re-run
   - Post-run verification steps

8. **Appendix: Representative Log Excerpts**
   - Full, unedited excerpts showing each failure mode
   - Include file paths for traceability

#### Report Header Template

```markdown
# HPC Debugging Report: {Analysis Name}

**Report Date**: {YYYY-MM-DD HH:MM:SS}
**Analysis ID**: `{analysis_id}`
**Workflow Run Date**: {date of failed run}
**Previous Reports**: {list previous report dates or "None (initial report)"}

---

## Executive Summary
...
```

#### Referencing Previous Reports

If prior reports exist, add a section:

```markdown
## Changes Since Previous Report ({date})

### Issues Resolved
- Issue 1: Brief description
- Issue 2: Brief description

### Persistent Issues
- Issue 3: Still occurring, see Section X for details

### New Issues
- Issue 4: Newly discovered in this run
```

---

## Output Format

Structure your **verbal response** as (before writing report):

### 1. Initial Assessment
- Execution mode and system setup (from configs)
- Scope of failure (which phases, how many scenarios affected)

### 2. Evidence Summary
- Key findings from master workflow log
- Incomplete rules (from `_status/` and `scenario_status.csv`)
- Specific error messages from logs

### 3. Root Cause Analysis
- Most likely cause based on evidence
- Supporting details from logs

### 4. Recommended Fixes
- Specific configuration changes needed
- Whether full or partial re-run is required

### 5. Investigation Commands
- Provide commands user can run on login node to investigate further
- Group by investigation type (partition configs, QOS limits, resource tests)

### 6. Additional Files Needed (If Applicable)
If diagnosis is incomplete, request specific files:
- Simulation files or folders: `sims/{event_id}/` (exclude `out_*` directories)
- Configuration files from subanalyses

**Then write the debugging report as described in Step 7.**

---

## Example Usage

**User:**
```
@debugging_hpc_analysis.md, .debugging/uva_sensitivity_suite/
```
---

## Important Notes

- **Always start with master workflow log** (`logs/_slurm_logs/*.out`) - it reveals most failures immediately
- **Don't assume compilation issues** - if setup succeeded, compilation worked
- **Focus on most recent run** - ignore older logs with different datetimes
- **For time limits**: Check if failures correlate with specific resource configs (low CPU counts often timeout)
- **Keep it focused**: Provide actionable diagnosis, not exhaustive investigation
