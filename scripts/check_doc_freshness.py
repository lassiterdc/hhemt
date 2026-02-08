#!/usr/bin/env python3
"""
Check if CLAUDE.md or agent documentation might need updating based on staged file changes.

This script is called by pre-commit to remind developers when documentation may be stale.
It does NOT block commits - it only prints reminders.

Exit codes:
  0 - Always (non-blocking reminder)
"""

import subprocess
import sys
from pathlib import Path

# Map source files to their corresponding agent documentation
AGENT_MAPPING = {
    "config.py": ".claude/agents/pydantic-config-specialist.md",
    "workflow.py": ".claude/agents/snakemake-workflow.md",
    "execution.py": ".claude/agents/hpc-slurm-integration.md",
    "resource_management.py": ".claude/agents/hpc-slurm-integration.md",
    "sensitivity_analysis.py": ".claude/agents/sensitivity-analysis.md",
    "swmm_utils.py": ".claude/agents/swmm-model-generation.md",
    "swmm_runoff_modeling.py": ".claude/agents/swmm-model-generation.md",
    "swmm_full_model.py": ".claude/agents/swmm-model-generation.md",
    "scenario_inputs.py": ".claude/agents/swmm-model-generation.md",
    "swmm_output_parser.py": ".claude/agents/output-processing.md",
    "process_simulation.py": ".claude/agents/output-processing.md",
    "conftest.py": ".claude/agents/triton-test-suite.md",
    "utils_for_testing.py": ".claude/agents/triton-test-suite.md",
    "log.py": "CLAUDE.md",  # LogField pattern documented in main file
    "system.py": "CLAUDE.md",  # Three-layer architecture
    "analysis.py": "CLAUDE.md",  # Three-layer architecture
    "scenario.py": "CLAUDE.md",  # Three-layer architecture
}

# Files that should trigger CLAUDE.md update check
CLAUDE_MD_TRIGGERS = {
    "pyproject.toml",  # Build/test commands
    "setup.py",
    "config.py",  # Critical configuration fields
    "workflow.py",  # Architecture changes
    "execution.py",  # Execution modes
}


def get_staged_files():
    """Get list of staged files relevant to documentation freshness checks."""
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
            capture_output=True,
            text=True,
            check=True,
        )
        staged = [f.strip() for f in result.stdout.splitlines() if f.strip()]

        # Include Python files for agent mappings and explicit trigger files
        # (e.g., pyproject.toml) for CLAUDE.md checks.
        trigger_files = set(CLAUDE_MD_TRIGGERS)
        return [f for f in staged if f.endswith(".py") or Path(f).name in trigger_files]
    except subprocess.CalledProcessError:
        return []


def check_documentation_freshness():
    """Check if documentation might need updating and print reminders."""
    staged_files = get_staged_files()

    if not staged_files:
        return 0  # No relevant files staged, nothing to check

    # Track which docs might need updates
    docs_to_check = set()
    claude_md_update = False

    for file_path in staged_files:
        filename = Path(file_path).name

        # Check if this file maps to an agent
        if filename in AGENT_MAPPING:
            docs_to_check.add(AGENT_MAPPING[filename])

        # Check if this should trigger CLAUDE.md review
        if filename in CLAUDE_MD_TRIGGERS:
            claude_md_update = True

    # Print reminders (non-blocking)
    if docs_to_check or claude_md_update:
        print("\n" + "=" * 70)
        print("📝 DOCUMENTATION REMINDER")
        print("=" * 70)

        if claude_md_update:
            print("\n⚠️  You've modified files that may affect CLAUDE.md:")
            print("   Please review if CLAUDE.md needs updates:")
            print("   - Architecture changes (three-layer hierarchy)")
            print("   - New build/test commands")
            print("   - Critical configuration fields")
            print("   - New execution modes or gotchas")

        if docs_to_check:
            print("\n⚠️  You've modified files covered by these agent docs:")
            for doc in sorted(docs_to_check):
                print(f"   - {doc}")
            print("\n   Please review if these agents need pattern updates.")

        print(
            "\n💡 See CLAUDE.md 'Maintaining This Documentation' section for guidance."
        )
        print("=" * 70 + "\n")

    return 0  # Always succeed (non-blocking)


if __name__ == "__main__":
    sys.exit(check_documentation_freshness())
