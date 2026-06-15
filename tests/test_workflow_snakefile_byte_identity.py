"""Master-Snakefile byte-identity test.

Asserts that the source-side ``SnakemakeWorkflowBuilder.generate_snakefile_content``
and ``SensitivityAnalysisWorkflowBuilder.generate_master_snakefile_content``
emit byte-identical Snakefile output to the captured goldens. The goldens
were re-baselined at Plan Phase 5 to reflect the cfg-controlled
``static_backend`` default (``"plotly"`` per Decision 4); the builders read
the report-cfg field via ``_get_report_cfg_static_backend()`` and emit
plotly-branch rules when no ``--report-config`` overrides the default.

If byte-identity fails, the source-side Snakefile generation introduced
silent drift and must be fixed before merge — NOT papered over by
regenerating the golden.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from fixtures.test_case_catalog import Local_TestCases  # noqa: E402

from TRITON_SWMM_toolkit.sensitivity_analysis import (  # noqa: E402
    TRITONSWMM_sensitivity_analysis,
)
from TRITON_SWMM_toolkit.workflow import (  # noqa: E402
    SensitivityAnalysisWorkflowBuilder,
    SnakemakeWorkflowBuilder,
)

GOLDENS_DIR = Path(__file__).parent / "fixtures" / "golden_snakefiles"

# Phase 4 (4a — byte-identity foundation): thread an hpc_system_config into the
# byte-identity cases so cfg_hpc_system is non-None BEFORE the legacy
# None-fallbacks in the resolution helpers are deleted in 4c/4d (those helpers
# are called unconditionally during Snakefile generation). The Norfolk cases are
# LOCAL mode with all hpc_* selectors null, so the config is byte-identity-neutral
# (its partition is never looked up; default_account appears only in the profile
# config.yaml, not the Snakefile). See the example config's header for the rationale.
EXAMPLE_HPC_CONFIG = Path(__file__).parent / "fixtures" / "hpc_system_config_test.yaml"

# Pin sys.executable for byte-identity comparison: pytest's shebang resolves
# to `python3.11` whereas direct invocation resolves to `python`. The
# captured goldens use `python` (symlink-preserved); pin to match.
@pytest.fixture(autouse=True)
def _pin_sys_executable(monkeypatch: pytest.MonkeyPatch) -> None:
    pinned = re.sub(r"/python3\.\d+$", "/python", sys.executable)
    monkeypatch.setattr(sys, "executable", pinned)


def _unified_diff_excerpt(want: str, got: str, max_lines: int = 80) -> str:
    import difflib

    diff = list(
        difflib.unified_diff(
            want.splitlines(keepends=True),
            got.splitlines(keepends=True),
            fromfile="golden",
            tofile="emitted",
            n=2,
        )
    )
    return "".join(diff[:max_lines])


def test_multi_sim_snakefile_byte_identity() -> None:
    """Source-side multi-sim Snakefile byte-identical to golden."""
    tc = Local_TestCases.retrieve_norfolk_multi_sim_test_case(
        start_from_scratch=False,
        download_if_exists=False,
        hpc_system_config_yaml=EXAMPLE_HPC_CONFIG,
    )
    builder = SnakemakeWorkflowBuilder(tc.analysis)
    got = builder.generate_snakefile_content()
    want = (GOLDENS_DIR / "multi_sim.Snakefile.golden").read_text()
    assert got == want, _unified_diff_excerpt(want, got)


def test_master_snakefile_byte_identity() -> None:
    """Source-side sensitivity-master Snakefile byte-identical to golden."""
    tc = Local_TestCases.retrieve_norfolk_cpu_config_sensitivity_case(
        start_from_scratch=False,
        download_if_exists=False,
        hpc_system_config_yaml=EXAMPLE_HPC_CONFIG,
    )
    sens = TRITONSWMM_sensitivity_analysis(tc.analysis)
    builder = SensitivityAnalysisWorkflowBuilder(sens)
    got = builder.generate_master_snakefile_content()
    want = (GOLDENS_DIR / "sensitivity_master.Snakefile.golden").read_text()
    assert got == want, _unified_diff_excerpt(want, got)


# ========== Phase 3b: Snakemake `group:` directive assertions (R8) ==========

# Per-rule `cpus_per_task` for the three process_* rules. Snakemake's
# GroupResources default-sums resources across jobs in the same group, so
# the per-event group's aggregate `cpus_per_task` equals the sum.
_EXPECTED_CPUS_PER_TASK_PER_PROCESS_RULE = 2
_EXPECTED_PROCESS_RULES_PER_EVENT = 3  # triton, tritonswmm, swmm
_EXPECTED_GROUP_CPUS_PER_TASK = (
    _EXPECTED_CPUS_PER_TASK_PER_PROCESS_RULE * _EXPECTED_PROCESS_RULES_PER_EVENT
)
# Conservative ceiling guarding against accidental cpus_per_task inflation
# (Snakemake architecture Gotcha 9): if a future edit bumps the per-rule
# value, the aggregate could over-subscribe a CI runner. 16 leaves
# headroom over the current 6.
_MAX_GROUP_CPUS_PER_TASK = 16


def test_process_rules_emit_group_directive() -> None:
    """All three process_* rules carry `group: "process_evt_{event_id}"` so
    Snakemake's DAG planner collapses them into a single per-event job-group,
    deduplicating subprocess-startup overhead (Phase 3b, R8)."""
    tc = Local_TestCases.retrieve_norfolk_multi_sim_test_case(
        start_from_scratch=False,
        download_if_exists=False,
        hpc_system_config_yaml=EXAMPLE_HPC_CONFIG,
    )
    builder = SnakemakeWorkflowBuilder(tc.analysis)
    got = builder.generate_snakefile_content()
    for rule_name in ("process_triton", "process_tritonswmm", "process_swmm"):
        pattern = (
            rf"rule {rule_name}:[\s\S]*?^\s*group:\s*\"process_evt_\{{event_id\}}\""
        )
        assert re.search(pattern, got, re.MULTILINE), (
            f"rule {rule_name} missing `group: \"process_evt_{{event_id}}\"` "
            f"directive; Snakemake will not collapse the three process rules "
            f"into a per-event job-group."
        )


def test_process_rule_group_resources_do_not_overallocate() -> None:
    """GroupResources sums resources across grouped jobs by default. Verify
    the per-event aggregate `cpus_per_task` (sum across the three process_*
    rules) stays within a sane ceiling (architecture Gotcha 9 for Snakemake;
    Phase 3b, R8). This is a static check on the emitted Snakefile."""
    tc = Local_TestCases.retrieve_norfolk_multi_sim_test_case(
        start_from_scratch=False,
        download_if_exists=False,
        hpc_system_config_yaml=EXAMPLE_HPC_CONFIG,
    )
    builder = SnakemakeWorkflowBuilder(tc.analysis)
    got = builder.generate_snakefile_content()

    per_rule_cpus: list[int] = []
    for rule_name in ("process_triton", "process_tritonswmm", "process_swmm"):
        block = re.search(rf"rule {rule_name}:[\s\S]*?(?=\nrule |\Z)", got)
        assert block is not None, f"rule {rule_name} not found in Snakefile"
        m = re.search(r"cpus_per_task=(\d+)", block.group(0))
        assert m is not None, (
            f"rule {rule_name} does not declare `cpus_per_task=`; cannot "
            f"verify group-resource sum."
        )
        per_rule_cpus.append(int(m.group(1)))

    assert len(per_rule_cpus) == _EXPECTED_PROCESS_RULES_PER_EVENT
    aggregate = sum(per_rule_cpus)
    assert aggregate == _EXPECTED_GROUP_CPUS_PER_TASK, (
        f"Per-event group cpus_per_task aggregate is {aggregate}; expected "
        f"{_EXPECTED_GROUP_CPUS_PER_TASK} "
        f"({_EXPECTED_CPUS_PER_TASK_PER_PROCESS_RULE} × "
        f"{_EXPECTED_PROCESS_RULES_PER_EVENT}). If a per-rule cpus_per_task "
        f"changed deliberately, update the constants and audit "
        f"--group-components scheduling."
    )
    assert aggregate <= _MAX_GROUP_CPUS_PER_TASK, (
        f"Per-event group cpus_per_task aggregate {aggregate} exceeds the "
        f"ceiling {_MAX_GROUP_CPUS_PER_TASK}; risk of over-subscribing CI "
        f"runners under default --group-components scheduling."
    )
