# Scripts Directory

This directory contains utility scripts for the TRITON-SWMM toolkit.

## Documentation Maintenance Scripts

### `check_doc_freshness.py`

Python script that checks if CLAUDE.md or agent documentation needs updating based on file changes.

**Usage:**
```bash
# Called automatically by pre-commit hook
# Or run manually:
python scripts/check_doc_freshness.py
```

**How it works:**
- Maps source files to their corresponding agent documentation
- Detects staged changes in git
- Prints reminders (non-blocking) when docs might be stale

**File Mapping:**
| Source File | Documentation |
|-------------|---------------|
| `config.py` | `pydantic-config-specialist.md` |
| `workflow.py` | `snakemake-workflow.md` |
| `execution.py`, `resource_management.py` | `hpc-slurm-integration.md` |
| `sensitivity_analysis.py` | `sensitivity-analysis.md` |
| `swmm_*.py`, `scenario_inputs.py` | `swmm-model-generation.md` |
| `swmm_output_parser.py`, `process_simulation.py` | `output-processing.md` |
| `conftest.py`, `utils_for_testing.py` | `triton-test-suite.md` |
| `system.py`, `analysis.py`, `scenario.py` | `CLAUDE.md` |

### `update_docs_check.sh`

Bash script for manual documentation freshness check (doesn't require git staging).

**Usage:**
```bash
# Check if docs need updating based on current changes
./scripts/update_docs_check.sh
```

**When to run:**
- Before committing changes to core files
- During code review
- When refactoring major components

## Setting Up Pre-commit Hooks

To automatically check documentation freshness on every commit:

```bash
# Install pre-commit (if not already installed)
pip install pre-commit

# Install the hooks
pre-commit install

# Test it (optional)
pre-commit run --all-files
```

Once installed, the documentation check will run automatically whenever you commit changes to core files. It will print a reminder but won't block your commit.

## Updating the File Mapping

If you add new modules or reorganize code, update the mappings in:
1. `check_doc_freshness.py` - `AGENT_MAPPING` and `CLAUDE_MD_TRIGGERS` dicts
2. `update_docs_check.sh` - `claude_md_files` array and `agent_files` associative array
3. `.github/workflows/doc-check.yml` - `paths` list
