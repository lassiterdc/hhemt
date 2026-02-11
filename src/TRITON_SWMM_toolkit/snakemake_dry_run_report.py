from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import re

from TRITON_SWMM_toolkit.utils import current_datetime_string


def _parse_job_stats(lines: list[str]) -> list[tuple[str, int]]:
    stats: list[tuple[str, int]] = []
    in_table = False
    for line in lines:
        stripped = line.strip()
        if stripped == "Job stats:":
            in_table = True
            continue
        if not in_table:
            continue
        if not stripped:
            if stats:
                break
            continue
        if stripped.startswith("job") or stripped.startswith("---"):
            continue
        m = re.match(r"^([A-Za-z0-9_]+)\s+(\d+)$", stripped)
        if m:
            stats.append((m.group(1), int(m.group(2))))
    return stats


def _parse_reason_components(lines: list[str]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("reason:"):
            continue
        reason_text = stripped.split("reason:", 1)[1].strip()
        for part in reason_text.split(";"):
            part = part.strip()
            if part:
                counter[part] += 1
    return counter


def _summarize_reason_categories(lines: list[str]) -> dict[str, Counter[str]]:
    """Summarize Snakemake reason lines by rule and reason category.

    Categories:
        - missing_output
        - input_updated
        - other
    """
    summary: dict[str, Counter[str]] = defaultdict(Counter)
    current_rule: str | None = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("rule ") and stripped.endswith(":"):
            current_rule = stripped.split()[1].rstrip(":")
            continue
        if not stripped.startswith("reason:") or current_rule is None:
            continue
        reason_text = stripped.split("reason:", 1)[1].strip()
        for part in reason_text.split(";"):
            part = part.strip()
            if not part:
                continue
            if part.startswith("Missing output files"):
                summary[current_rule]["missing_output"] += 1
            elif part.startswith("Input files updated by another job"):
                summary[current_rule]["input_updated"] += 1
            else:
                summary[current_rule]["other"] += 1
    return summary


def _format_missing_ids(ids: list[int], max_ids: int = 10) -> str:
    if not ids:
        return ""
    if len(ids) <= max_ids:
        return ", ".join(str(i) for i in ids)
    return ", ".join(str(i) for i in ids[:max_ids]) + f" (+{len(ids) - max_ids} more)"


def _parse_footer_reason_map(lines: list[str]) -> dict[str, list[str]]:
    reason_map: dict[str, list[str]] = defaultdict(list)
    in_reasons = False
    current_reason: str | None = None

    for line in lines:
        stripped = line.strip()
        if stripped == "Reasons:":
            in_reasons = True
            continue
        if not in_reasons:
            continue
        if not stripped:
            continue
        if stripped.startswith("This was a dry-run"):
            break
        if stripped.startswith("(check individual jobs above"):
            continue
        if stripped.endswith(":") and not line.startswith("        "):
            current_reason = stripped[:-1]
            reason_map.setdefault(current_reason, [])
            continue
        if current_reason and line.startswith("        "):
            reason_map[current_reason].append(stripped)

    return dict(reason_map)


def _parse_missing_status_flags(lines: list[str]) -> dict[str, list[int]]:
    missing: dict[str, set[int]] = {
        "scenario_prepared": set(),
        "triton_complete": set(),
        "triton_processed": set(),
    }
    pattern = re.compile(
        r"_status/sims/(scenario|triton)_(\d+)_(prepared|complete|processed)\.flag:\s+False"
    )

    for line in lines:
        m = pattern.search(line)
        if not m:
            continue
        prefix, sim_id_str, suffix = m.groups()
        sim_id = int(sim_id_str)

        if prefix == "scenario" and suffix == "prepared":
            missing["scenario_prepared"].add(sim_id)
        elif prefix == "triton" and suffix == "complete":
            missing["triton_complete"].add(sim_id)
        elif prefix == "triton" and suffix == "processed":
            missing["triton_processed"].add(sim_id)

    return {k: sorted(v) for k, v in missing.items()}


def _extract_metadata(lines: list[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("Using profile ") and " for setting default" in stripped:
            metadata["profile"] = stripped
        elif stripped.startswith("host:"):
            metadata["host"] = stripped.split(":", 1)[1].strip()
    metadata["dry_run_marker_found"] = str(
        any("This was a dry-run" in line for line in lines)
    )
    return metadata


def generate_dry_run_report_markdown(
    snakemake_logfile: Path,
    analysis_dir: Path,
    verbose: bool = True,
) -> Path:
    lines = snakemake_logfile.read_text().splitlines()

    job_stats = _parse_job_stats(lines)
    reason_components = _parse_reason_components(lines)
    reason_summary = _summarize_reason_categories(lines)
    footer_reason_map = _parse_footer_reason_map(lines)
    missing_flags = _parse_missing_status_flags(lines)
    metadata = _extract_metadata(lines)

    report_path = analysis_dir / "dry_run_report.md"
    written_at = current_datetime_string()

    md_lines: list[str] = [
        "# Snakemake Dry Run Report",
        "",
        f"- Written at: `{written_at}`",
        f"- Source log: `{snakemake_logfile}`",
        "",
        "## Metadata",
        "",
    ]
    for key in ("profile", "host", "dry_run_marker_found"):
        if key in metadata:
            md_lines.append(f"- {key}: `{metadata[key]}`")

    md_lines.extend(["", "## Job Statistics", ""])
    if job_stats:
        md_lines.extend(
            [
                "| rule | count |",
                "|---|---:|",
                *[f"| {rule} | {count} |" for rule, count in job_stats],
            ]
        )
    else:
        md_lines.append("No job stats table found in logfile.")

    md_lines.extend(["", "## Job Status Summary", ""])
    if reason_summary:
        md_lines.extend(
            [
                "| rule | missing outputs | inputs updated | other |",
                "|---|---:|---:|---:|",
            ]
        )
        for rule, counts in sorted(reason_summary.items()):
            md_lines.append(
                "| {rule} | {missing} | {updated} | {other} |".format(
                    rule=rule,
                    missing=counts.get("missing_output", 0),
                    updated=counts.get("input_updated", 0),
                    other=counts.get("other", 0),
                )
            )
    elif reason_components:
        md_lines.append("No per-rule reason lines found.")
    else:
        md_lines.append("No reason summary available.")

    md_lines.extend(["", "## Reasons Footer (from Snakemake summary)", ""])
    if footer_reason_map:
        for reason, rules in footer_reason_map.items():
            md_lines.append(f"- **{reason}**")
            if rules:
                for rule_name in rules:
                    md_lines.append(f"  - `{rule_name}`")
    else:
        md_lines.append("No Reasons footer section found.")

    md_lines.extend(["", "## Missing Status Flags Observed", ""])
    for key in ("scenario_prepared", "triton_complete", "triton_processed"):
        ids = missing_flags.get(key, [])
        md_lines.append(f"- `{key}` missing count: **{len(ids)}**")
        if ids:
            md_lines.append(f"  - IDs: {_format_missing_ids(ids)}")

    md_content = "\n".join(md_lines) + "\n"
    report_path.write_text(md_content)

    if verbose:
        print("\n=== Dry Run Report ===")
        print(md_content)
        print(f"Report written to: {report_path}")

    return report_path
