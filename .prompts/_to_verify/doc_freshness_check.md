# Documentation Freshness Check

Check if documentation needs updating based on recent code changes.

## What to Check

Per `scripts/check_doc_freshness.py` and `CLAUDE.md` guidelines:

1. **CLAUDE.md** — if touching:
   - Architecture (system.py, analysis.py, scenario.py)
   - Key modules table
   - Configuration fields
   - Build/test commands
   - HPC execution modes
   - Runner script architecture

2. **Agent docs** (`.claude/agents/*.md`) — if touching:
   - `pydantic-config-specialist.md` — config validation patterns
   - `snakemake-workflow.md` — workflow generation, DAG structure
   - `hpc-slurm-integration.md` — SLURM modes, resource management
   - `output-processing.md` — SWMM/TRITON output parsing
   - `swmm-model-generation.md` — SWMM .inp generation patterns
   - `triton-test-suite.md` — test utilities, fixtures

3. **Planning docs** — if completing planned work:
   - Mark phases as complete in plan documents
   - Update priorities.md
   - Archive completed planning docs if appropriate

## Questions to Ask

- Does this change affect architecture described in CLAUDE.md?
- Are there new "gotchas" or non-obvious behaviors to document?
- Do build/test commands still work as documented?
- Should any implementation notes be archived?

## Expected Output

- List of docs that need updating
- Specific sections/content that became stale
- Recommended updates or archival actions
