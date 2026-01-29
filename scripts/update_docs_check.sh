#!/bin/bash
# Manual documentation freshness check
# Run this before committing to see if docs might need updating

set -e

echo "=================================================="
echo "  TRITON-SWMM Documentation Freshness Check"
echo "=================================================="
echo ""

# Check if running in git repo
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "❌ Not in a git repository"
    exit 1
fi

# Get list of modified files (staged + unstaged)
modified_files=$(git diff --name-only HEAD)

if [ -z "$modified_files" ]; then
    echo "✅ No modified files detected"
    echo ""
    exit 0
fi

echo "Modified files:"
echo "$modified_files" | sed 's/^/  - /'
echo ""

# Files that might affect CLAUDE.md
claude_md_files=(
    "pyproject.toml"
    "setup.py"
    "src/TRITON_SWMM_toolkit/config.py"
    "src/TRITON_SWMM_toolkit/workflow.py"
    "src/TRITON_SWMM_toolkit/execution.py"
    "src/TRITON_SWMM_toolkit/system.py"
    "src/TRITON_SWMM_toolkit/analysis.py"
    "src/TRITON_SWMM_toolkit/scenario.py"
)

# Files that might affect agent docs
declare -A agent_files=(
    ["src/TRITON_SWMM_toolkit/config.py"]=".claude/agents/pydantic-config-specialist.md"
    ["src/TRITON_SWMM_toolkit/workflow.py"]=".claude/agents/snakemake-workflow.md"
    ["src/TRITON_SWMM_toolkit/execution.py"]=".claude/agents/hpc-slurm-integration.md"
    ["src/TRITON_SWMM_toolkit/resource_management.py"]=".claude/agents/hpc-slurm-integration.md"
    ["src/TRITON_SWMM_toolkit/sensitivity_analysis.py"]=".claude/agents/sensitivity-analysis.md"
    ["src/TRITON_SWMM_toolkit/swmm_utils.py"]=".claude/agents/swmm-model-generation.md"
    ["src/TRITON_SWMM_toolkit/swmm_runoff_modeling.py"]=".claude/agents/swmm-model-generation.md"
    ["src/TRITON_SWMM_toolkit/swmm_full_model.py"]=".claude/agents/swmm-model-generation.md"
    ["src/TRITON_SWMM_toolkit/scenario_inputs.py"]=".claude/agents/swmm-model-generation.md"
    ["src/TRITON_SWMM_toolkit/swmm_output_parser.py"]=".claude/agents/output-processing.md"
    ["src/TRITON_SWMM_toolkit/process_simulation.py"]=".claude/agents/output-processing.md"
    ["tests/conftest.py"]=".claude/agents/triton-test-suite.md"
    ["tests/utils_for_testing.py"]=".claude/agents/triton-test-suite.md"
)

# Check for CLAUDE.md updates
claude_md_needs_update=false
for file in "${claude_md_files[@]}"; do
    if echo "$modified_files" | grep -q "^$file$"; then
        claude_md_needs_update=true
        break
    fi
done

# Check for agent doc updates
declare -A agents_to_check
for file in "${!agent_files[@]}"; do
    if echo "$modified_files" | grep -q "^$file$"; then
        agent="${agent_files[$file]}"
        agents_to_check["$agent"]=1
    fi
done

# Print results
if [ "$claude_md_needs_update" = true ] || [ ${#agents_to_check[@]} -gt 0 ]; then
    echo "⚠️  DOCUMENTATION MAY NEED UPDATES"
    echo ""

    if [ "$claude_md_needs_update" = true ]; then
        echo "📄 Consider reviewing CLAUDE.md for:"
        echo "   - Architecture changes"
        echo "   - New build/test commands"
        echo "   - Critical configuration fields"
        echo "   - New gotchas"
        echo ""
    fi

    if [ ${#agents_to_check[@]} -gt 0 ]; then
        echo "🤖 Consider reviewing these agent docs:"
        for agent in "${!agents_to_check[@]}"; do
            echo "   - $agent"
        done
        echo ""
    fi

    echo "💡 See CLAUDE.md 'Maintaining This Documentation' for guidance"
    echo ""
else
    echo "✅ No documentation-sensitive files modified"
    echo ""
fi

echo "=================================================="
