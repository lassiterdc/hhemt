"""Utilities for strict parsing of generated Snakemake Snakefiles.

This module intentionally follows a fail-fast strategy:
- Missing Snakefile => error
- Missing expected run/simulation rules => error
- Missing required resource keys (tasks/cpus_per_task) => error

The parsed outputs are used for workflow diagnostics (intended vs allocated vs actual).
"""

from __future__ import annotations

from pathlib import Path
import re


class SnakefileParsingError(RuntimeError):
    """Raised when generated Snakefile parsing fails validation."""


_RULE_RE = re.compile(r"^\s*rule\s+([A-Za-z0-9_]+)\s*:\s*$")
_TASKS_RE = re.compile(r"\btasks\s*=\s*(\d+)\b")
_CPUS_RE = re.compile(r"\bcpus_per_task\s*=\s*(\d+)\b")
_GPU_RE = re.compile(r"\bgpu\s*=\s*(\d+)\b")
_SA_SIM_RE = re.compile(r"^simulation_sa(\d+)_evt(\d+)$")


def _read_snakefile_text(snakefile_path: Path) -> str:
    if not snakefile_path.exists():
        raise FileNotFoundError(
            f"Snakefile not found at {snakefile_path}. Cannot parse Snakemake resource allocations."
        )
    return snakefile_path.read_text()


def _extract_rule_blocks(snakefile_text: str) -> dict[str, str]:
    lines = snakefile_text.splitlines()
    blocks: dict[str, list[str]] = {}
    current_rule: str | None = None

    for line in lines:
        rule_match = _RULE_RE.match(line)
        if rule_match:
            rule_name = rule_match.group(1)
            current_rule = rule_name
            blocks[rule_name] = [line]
            continue
        if current_rule is not None:
            blocks[current_rule].append(line)

    if not blocks:
        raise SnakefileParsingError("No Snakemake rules found while parsing Snakefile.")

    return {name: "\n".join(rule_lines) for name, rule_lines in blocks.items()}


def _parse_rule_resources(rule_name: str, rule_block: str) -> dict[str, int]:
    tasks_match = _TASKS_RE.search(rule_block)
    cpus_match = _CPUS_RE.search(rule_block)
    gpu_match = _GPU_RE.search(rule_block)

    if tasks_match is None:
        raise SnakefileParsingError(
            f"Rule '{rule_name}' is missing required Snakemake resource key: tasks"
        )
    if cpus_match is None:
        raise SnakefileParsingError(
            f"Rule '{rule_name}' is missing required Snakemake resource key: cpus_per_task"
        )

    tasks = int(tasks_match.group(1))
    cpus_per_task = int(cpus_match.group(1))
    gpus = int(gpu_match.group(1)) if gpu_match is not None else 0

    return {
        "snakemake_allocated_nTasks": tasks,
        "snakemake_allocated_omp_threads": cpus_per_task,
        "snakemake_allocated_gpus": gpus,
        "snakemake_allocated_total_cpus": tasks * cpus_per_task,
    }


def parse_regular_workflow_model_allocations(
    snakefile_path: Path,
    enabled_model_types: list[str],
) -> dict[str, dict[str, int]]:
    """Parse model allocations from regular (non-sensitivity) Snakefile.

    Parameters
    ----------
    snakefile_path : Path
        Path to generated Snakefile.
    enabled_model_types : list[str]
        Enabled model types expected to have corresponding `run_<model>` rules.
    """
    snakefile_text = _read_snakefile_text(snakefile_path)
    rule_blocks = _extract_rule_blocks(snakefile_text)

    allocations: dict[str, dict[str, int]] = {}
    for model_type in enabled_model_types:
        rule_name = f"run_{model_type}"
        if rule_name not in rule_blocks:
            raise SnakefileParsingError(
                f"Expected Snakemake rule '{rule_name}' not found for enabled model '{model_type}'."
            )
        allocations[model_type] = _parse_rule_resources(
            rule_name=rule_name,
            rule_block=rule_blocks[rule_name],
        )

    return allocations


def parse_sensitivity_analysis_workflow_model_allocations(
    snakefile_path: Path,
    expected_subanalysis_ids: list[int] | None = None,
) -> dict[int, dict[str, int]]:
    """Parse per-subanalysis allocations from flattened sensitivity Snakefile.

    The flattened sensitivity Snakefile contains rules named like:
    ``simulation_sa{sa_id}_evt{event_id}``.

    This parser extracts Snakemake resources from those simulation rules and
    returns one allocation per subanalysis. If multiple event rules for the
    same subanalysis disagree on resources, it raises ``SnakefileParsingError``.
    """
    snakefile_text = _read_snakefile_text(snakefile_path)
    rule_blocks = _extract_rule_blocks(snakefile_text)

    allocations_by_sa: dict[int, dict[str, int]] = {}

    for rule_name, rule_block in rule_blocks.items():
        match = _SA_SIM_RE.match(rule_name)
        if match is None:
            continue

        sa_id = int(match.group(1))
        parsed_alloc = _parse_rule_resources(rule_name=rule_name, rule_block=rule_block)

        if sa_id in allocations_by_sa and allocations_by_sa[sa_id] != parsed_alloc:
            raise SnakefileParsingError(
                "Inconsistent Snakemake resources found across sensitivity simulation "
                f"rules for subanalysis sa_{sa_id}."
            )

        allocations_by_sa[sa_id] = parsed_alloc

    if not allocations_by_sa:
        raise SnakefileParsingError(
            "No sensitivity simulation rules found. Expected rules matching "
            "'simulation_sa{sa_id}_evt{event_id}'."
        )

    if expected_subanalysis_ids is not None:
        missing = sorted(set(expected_subanalysis_ids) - set(allocations_by_sa.keys()))
        if missing:
            raise SnakefileParsingError(
                "Missing expected sensitivity simulation allocations for subanalysis ids: "
                f"{missing}"
            )

    return allocations_by_sa
