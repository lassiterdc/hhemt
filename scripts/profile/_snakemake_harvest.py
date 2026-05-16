"""Harvest per-rule timing from .snakemake/metadata/ JSON records.

The harvester runs strictly post-hoc after pytest exits. It walks each test's
tmp_path via rglob("Snakefile"), checks for a sibling .snakemake/metadata/
directory, parses every JSON record there, and groups records by
(rule, job_hash) to deduplicate multi-output rules.

Statistical-reporting discipline (per hpc-performance-specialist refresh):
durations are summarized via min/mean/max/total here; median + IQR rollups
are emitter-side. No artificial floor is applied to sub-second rules — they
report as 0.0 and weight out correctly when aggregated.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

MIN_RECORD_VERSION = 5
MAX_VALIDATED_VERSION = 6
_ZERO_DURATION_THRESHOLD = 1e-6

_NORMALIZE = [
    (re.compile(r"_sa_\d+"), "_sa_N"),
    (re.compile(r"_evt_\d+"), "_evt_N"),
]


def normalize_rule(name: str) -> str:
    r"""Collapse sensitivity-rule wildcards into a normalized form.

    Currently collapses ``_sa_\d+`` -> ``_sa_N`` and ``_evt_\d+`` -> ``_evt_N``
    to match the toolkit's ``SnakemakeWorkflowBuilder`` rule-naming conventions.

    NOTE: If ``workflow.py`` ever adds a third wildcard (e.g., ``_year_\d+``,
    ``_replicate_\d+``, ``_iloc_\d+``), the normalization will silently fail to
    collapse it. Extend ``_NORMALIZE`` regex list when adding new wildcard
    naming conventions.
    """
    for pat, repl in _NORMALIZE:
        name = pat.sub(repl, name)
    return name


@dataclass
class HarvestedRule:
    rule: str
    rule_normalized: str
    job_count: int
    total_s: float
    mean_s: float
    min_s: float
    max_s: float
    zero_duration_job_count: int
    test_origin: str = ""


@dataclass
class HarvestDiagnostics:
    snakefiles_found: int = 0
    snakefiles_with_metadata: int = 0
    snakefiles_dry_run_only: int = 0
    snakefiles_zero_records: int = 0
    total_records: int = 0
    parser_warnings: list[str] = field(default_factory=list)


def discover_analysis_dirs(tmp_path_root: Path) -> list[Path]:
    """Find every directory under tmp_path_root containing a Snakefile + .snakemake/metadata/.

    Returns sorted, deduplicated list of analysis directories. Empty list when
    no metadata-bearing Snakefiles found (e.g., dry-run-only tests).

    Assumption: each profile-run uses fresh pytest-managed tmp_paths so stale
    .snakemake/ state from prior runs cannot leak into the harvest. Re-using a
    tmp_path across runs would silently corrupt timing data; that contract is
    enforced by the orchestrator, not by this function.
    """
    found: set[Path] = set()
    for snakefile in tmp_path_root.rglob("Snakefile"):
        parent = snakefile.parent
        if (parent / ".snakemake" / "metadata").is_dir():
            found.add(parent)
    return sorted(found)


def _parse_metadata_dir(metadata_dir: Path) -> tuple[list[tuple[str, int, float]], list[str]]:
    """Read every JSON record under metadata_dir; return (records, warnings).

    records is a list of (rule, job_hash, duration_s). Warnings are emitted for
    record_format_version drift on either side of the validated range.
    """
    records: list[tuple[str, int, float]] = []
    warnings: list[str] = []
    for path in metadata_dir.rglob("*"):
        if not path.is_file():
            continue
        try:
            obj = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        version = obj.get("record_format_version", 0)
        if version < MIN_RECORD_VERSION:
            warnings.append(
                f"legacy metadata format v{version} at {path.name} — parser may misread fields"
            )
            continue
        if version > MAX_VALIDATED_VERSION:
            warnings.append(
                f"metadata format v{version} newer than parser was validated against "
                f"(max validated: v{MAX_VALIDATED_VERSION}); verify timing fields"
            )
        rule = obj.get("rule")
        job_hash = obj.get("job_hash")
        starttime = obj.get("starttime")
        endtime = obj.get("endtime")
        if rule is None or job_hash is None or starttime is None or endtime is None:
            continue
        duration = max(0.0, float(endtime) - float(starttime))
        records.append((rule, job_hash, duration))
    return records, warnings


def harvest(tmp_path_root: Path) -> tuple[list[HarvestedRule], HarvestDiagnostics]:
    """Walk tmp_path_root, harvest .snakemake/metadata/ records, return (rules, diagnostics).

    Public entry point. Group records by (rule, job_hash) to dedupe multi-output
    rules (same job_hash, identical timestamps). Per-rule aggregates use
    sum/mean/min/max over distinct job_hashes; the emitter further rolls up
    across reps with median + IQR.
    """
    out: list[HarvestedRule] = []
    diagnostics = HarvestDiagnostics()
    seen_dirs: set[Path] = set()
    for snakefile in tmp_path_root.rglob("Snakefile"):
        diagnostics.snakefiles_found += 1
        parent = snakefile.parent
        if parent in seen_dirs:
            continue
        seen_dirs.add(parent)
        md_dir = parent / ".snakemake" / "metadata"
        if not md_dir.is_dir():
            diagnostics.snakefiles_dry_run_only += 1
            continue
        diagnostics.snakefiles_with_metadata += 1
        records, warnings = _parse_metadata_dir(md_dir)
        diagnostics.parser_warnings.extend(warnings)
        diagnostics.total_records += len(records)
        if not records:
            diagnostics.snakefiles_zero_records += 1
            continue
        per_rule_jobs: dict[str, dict[int, float]] = defaultdict(dict)
        for rule, job_hash, duration in records:
            per_rule_jobs[rule][job_hash] = duration
        try:
            origin = str(parent.relative_to(tmp_path_root))
        except ValueError:
            origin = str(parent)
        for rule, jobs in per_rule_jobs.items():
            durations = list(jobs.values())
            if not durations:
                continue
            out.append(HarvestedRule(
                rule=rule,
                rule_normalized=normalize_rule(rule),
                job_count=len(durations),
                total_s=sum(durations),
                mean_s=sum(durations) / len(durations),
                min_s=min(durations),
                max_s=max(durations),
                zero_duration_job_count=sum(1 for d in durations if d <= _ZERO_DURATION_THRESHOLD),
                test_origin=origin,
            ))
    return out, diagnostics
