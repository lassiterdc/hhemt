"""Say which jobs Snakemake would schedule for a golden analysis tree, without running any of them.

A rerun-trigger test needs to know which jobs a dry run schedules after some outputs change, not what a real run
does to file times. The dry run answers that with no solver: it compares modification times under the ``mtime``
rerun trigger and prints the jobs it would run. This module wraps the golden-tree helpers in
``tests/fixtures/golden_trees.py`` (``construct_golden_analysis`` and ``dry_run_plan``) so a test passes a tree and
a set of modification times and receives the scheduled job names.

Usage in a test (``golden_extractor`` extracts a golden and lets the analysis constructor accept a working tree
under edit)::

    root = golden_extractor("multisim")
    untouched = planned_jobs(root)                                  # the report-stage jobs only
    stale = planned_jobs(root, older_than_input(root, run_flag, prepare_flag))
    assert stale > untouched                                        # the stale scenario's jobs are added

Job names. One job is named ``rule`` when the rule has no wildcards and ``rule[k=v,...]`` when it has, so the
three ``plot_per_sim_event_page`` jobs are three members of the set and a second scenario going stale adds that
scenario's names rather than leaving the set unchanged. Names are read from the ``rule NAME:`` blocks and their
``wildcards:`` lines in the dry-run log, because the ``Job stats`` table counts jobs per rule and cannot say which
instance is scheduled. That table's ``total`` line is used as a cross-check: ``planned_jobs`` raises if the parsed
names do not number the total, so a log in an unrecognised format cannot read as "nothing scheduled".

No solver is compiled or run: a dry run only plans.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path

from tests.fixtures.golden_trees import TIER_SECONDS, construct_golden_analysis, dry_run_plan

__all__ = ["job_names_from_log", "older_than_input", "planned_jobs", "stats_total"]

_RULE_RE = re.compile(r"^(?:local)?rule\s+([A-Za-z0-9_]+):\s*$")
_WILDCARDS_RE = re.compile(r"^\s+wildcards:\s*(\S.*?)\s*$")
_TOTAL_RE = re.compile(r"^total\s+(\d+)\s*$")


def job_names_from_log(lines: list[str]) -> frozenset[str]:
    """Parse the jobs a Snakemake dry-run log schedules into ``rule`` / ``rule[wildcards]`` names.

    Each scheduled job is one ``rule NAME:`` block (``localrule`` for a local rule); its ``wildcards:`` line, if
    any, is part of the name. A job with no wildcards is named by its rule alone."""
    names: set[str] = set()
    current: str | None = None
    wildcards: str | None = None
    for line in lines:
        rule = _RULE_RE.match(line.strip())
        if rule:
            if current is not None:
                names.add(f"{current}[{wildcards}]" if wildcards else current)
            current, wildcards = rule.group(1), None
            continue
        if current is not None:
            match = _WILDCARDS_RE.match(line)
            if match:
                wildcards = match.group(1).replace(", ", ",")
    if current is not None:
        names.add(f"{current}[{wildcards}]" if wildcards else current)
    return frozenset(names)


def stats_total(lines: list[str]) -> int | None:
    """The ``total`` row of the ``Job stats`` table of a dry-run log, or ``None`` when the log has no such row."""
    in_table = False
    for line in lines:
        stripped = line.strip()
        if stripped == "Job stats:":
            in_table = True
        elif in_table:
            match = _TOTAL_RE.match(stripped)
            if match:
                return int(match.group(1))
    return None


def older_than_input(root: Path, output_rel: str, input_rel: str) -> dict[str, float]:
    """The mtime entry that makes ``output_rel`` one tier older than ``input_rel`` (both relative to ``root``).

    Under the ``mtime`` rerun trigger an output older than its input is out of date, so the dry run schedules the
    job that makes it, and every job downstream of it."""
    return {output_rel: (root / input_rel).stat().st_mtime - TIER_SECONDS}


def planned_jobs(system_dir: Path, mtimes: Mapping[str, float] | None = None) -> frozenset[str]:
    """Apply ``mtimes`` to an extracted golden tree, dry-run the workflow, return the scheduled job names.

    ``system_dir`` is the extraction directory (what ``golden_extractor(kind)`` returns); ``mtimes`` maps a path
    relative to it to the modification time to give that file or directory (``None`` or empty leaves the tree as
    extracted, with every stage current). The tree is mutated in place, so a test that plans twice from one tree
    sees the earlier times; extract again for an independent plan. The dry run also writes the Snakefile."""
    for rel, stamp in (mtimes or {}).items():
        target = system_dir / rel
        if not target.exists():
            raise FileNotFoundError(f"cannot set the modification time of {rel!r}: {target} does not exist")
        os.utime(target, (stamp, stamp))
    analysis = construct_golden_analysis(system_dir)
    dry_run_plan(analysis)
    log = Path(analysis.analysis_paths.analysis_dir) / "logs" / "snakemake_master_dry_run.log"
    lines = log.read_text().splitlines()
    names = job_names_from_log(lines)
    total = stats_total(lines)
    if total is None or total != len(names):
        raise ValueError(
            f"the dry-run log {log} reports a job total of {total} but {len(names)} job name(s) were parsed from it; "
            "the log format is not the one this parser reads, so the plan cannot be trusted"
        )
    return names
