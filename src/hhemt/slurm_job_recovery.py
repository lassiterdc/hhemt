"""Recover the sacct rows the SLURM-executor plugin's parsing discards (Stage A of [Q153]).

WHY THIS EXISTS. `snakemake_executor_plugin_slurm/efficiency_report.py` asks sacct for
everything needed (`:46-47`), then drops two row classes while parsing:

    :115  df = df[~df["JobName"].str.contains("batch|extern", na=False)]
    :135  df = job_steps.copy()

Line 115 drops `.batch`, which carries the ONLY nonzero `TotalCPU` -- measured on job
18396137, every step reads `00:00:00` except `.batch` at `00:03.744`. That single drop is
why the rendered CPU-efficiency column reads 0 on every row. Line 135 drops the main job
rows, which carry the allocation's own `Elapsed` -- 00:02:00 on that job, which is neither
the max (00:01:51) nor the sum (00:01:14) of its steps, so it is not reconstructible from
what survives.

hhemt discards nothing: it passes `--slurm-efficiency-report-path` and later reads the CSV,
downstream of both drops. So this module does not fix a capture bug we own -- it recovers
rows a third-party parser dropped, using the job ids that parser DID keep.

WHAT THIS MODULE DELIBERATELY DOES NOT DO. It performs no reduction. It recovers rows and
writes them; deciding what a job's Elapsed/CPU/memory IS across steps is Stage B, where the
reducer travels with the declaration that names it. Keeping the boundary here is what makes
Stage A invisible: no rendered byte moves, because no renderer reads this yet.

Leaf module by construction, mirroring `slurm_liveness.py`: stdlib only, no toolkit imports,
so a runner subprocess can use it without pulling in the workflow builder. Every failure
mode degrades to "recovered nothing" rather than raising -- a missing sacct, a timeout, or
an unparsable line must never take down a workflow whose real work already succeeded.
"""

from __future__ import annotations

import csv
import io
import subprocess
import sys
from pathlib import Path

#: Written beside the plugin's own reports. Deliberately NOT matching the
#: `slurm_efficiency_report_*.csv` glob `metadata._resolve_all_efficiency_csvs` consumes,
#: so this file is never mistaken for an efficiency report and parsed as one.
RECOVERY_FILENAME = "job_level_recovery.csv"

_EFF_RELDIR = ("logs", "slurm_efficiency_report")
_EFF_GLOB = "slurm_efficiency_report_*.csv"
_EFF_INNER_GLOB = "efficiency_report_*.csv"

#: sacct fields recovered. `JobID` first because it carries the step suffix that decides
#: which class a row belongs to; `State` last because Stage B's per-attempt disclosure needs
#: it (the CANCELLED/COMPLETED breakdown of a resumed sim's solver steps).
_SACCT_FIELDS = ("JobID", "JobName", "Elapsed", "TotalCPU", "MaxRSS", "NNodes", "NCPUS", "ReqMem", "State")

#: Header of the emitted CSV. `StepKind` is DERIVED here rather than left to the reader:
#: the job/batch distinction is the whole point of the file, and re-deriving it downstream
#: from a suffix regex would be a second place that knowledge lives.
RECOVERY_HEADER = ("MainJobID", "StepKind", *_SACCT_FIELDS)

#: sacct is invoked in chunks. 771 ids at ~9 chars is well under ARG_MAX, but a campaign
#: that grows is exactly the case nobody re-measures, so the bound is explicit.
_CHUNK = 500


def _step_kind(job_id: str) -> str | None:
    """Classify a sacct JobID into the row class this module recovers, or None.

    `18396137`       -> "job"    (the allocation; carries Elapsed)
    `18396137.batch` -> "batch"  (carries the only nonzero TotalCPU)
    `18396137.0`     -> None     (a numeric step; the plugin already kept these)
    `18396137.extern`-> None     (never carries anything the table reports)
    """
    if "." not in job_id:
        return "job"
    suffix = job_id.split(".", 1)[1]
    return "batch" if suffix == "batch" else None


def main_job_ids_from_efficiency_csvs(analysis_dir: Path) -> list[str]:
    """Every distinct `MainJobID` the plugin's own reports kept, sorted numerically.

    Sourced from the plugin's output rather than from `_status/*.flag.json`, deliberately:
    the flag files retain one job id per rule (last-wins) and cover ~11% of the CSV's rows,
    while the CSV's `MainJobID` column is the full population this recovery must match. Any
    id the CSV does not carry is an id the rendered table has no row for.
    """
    eff_dir = analysis_dir.joinpath(*_EFF_RELDIR)
    if not eff_dir.is_dir():
        return []
    seen: set[str] = set()
    for match in sorted(eff_dir.glob(_EFF_GLOB)):
        # The plugin writes the real CSV INSIDE a `.csv`-NAMED DIRECTORY, so a bare
        # read_text() on the glob match raises IsADirectoryError (metadata.py documents
        # this at _resolve_all_efficiency_csvs). Handle both layouts.
        files = [match] if match.is_file() else sorted(match.glob(_EFF_INNER_GLOB))
        for path in files:
            if not path.is_file():
                continue
            try:
                text = path.read_text()
            except OSError:
                continue
            for row in csv.DictReader(io.StringIO(text)):
                value = (row.get("MainJobID") or "").strip()
                if value.isdigit():
                    seen.add(value)
    return sorted(seen, key=int)


def recover_rows(job_ids: list[str], *, timeout_s: float = 60.0) -> list[dict[str, str]]:
    """Query sacct for `job_ids` and return only the job and `.batch` rows.

    Returns [] on any failure -- missing sacct, timeout, non-zero exit. A partial recovery
    is returned as-is: sacct silently omits ids it no longer retains, and reporting the
    subset it DID return is more useful than discarding the batch. The caller reports
    coverage; this function does not decide that a shortfall is fatal.
    """
    if not job_ids:
        return []
    out: list[dict[str, str]] = []
    for start in range(0, len(job_ids), _CHUNK):
        chunk = job_ids[start : start + _CHUNK]
        try:
            proc = subprocess.run(
                ["sacct", "-j", ",".join(chunk), "-n", "-P", "-o", ",".join(_SACCT_FIELDS)],
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            print(
                f"[job-recovery] WARNING: sacct timed out after {timeout_s}s "
                f"on {len(chunk)} job id(s); recovered nothing for this chunk",
                file=sys.stderr,
                flush=True,
            )
            continue
        except FileNotFoundError:
            print(
                "[job-recovery] WARNING: sacct not found on PATH — recovery skipped",
                file=sys.stderr,
                flush=True,
            )
            return []
        if proc.returncode != 0:
            print(
                f"[job-recovery] WARNING: sacct exited {proc.returncode}; "
                f"recovered nothing for this chunk",
                file=sys.stderr,
                flush=True,
            )
            continue
        for line in proc.stdout.splitlines():
            if not line.strip():
                continue
            parts = line.split("|")
            if len(parts) != len(_SACCT_FIELDS):
                continue
            record = dict(zip(_SACCT_FIELDS, parts, strict=True))
            kind = _step_kind(record["JobID"])
            if kind is None:
                continue
            record["MainJobID"] = record["JobID"].split(".", 1)[0]
            record["StepKind"] = kind
            out.append(record)
    # Sorted so the emitted file is deterministic regardless of sacct's ordering or the
    # chunk boundaries -- a byte-stable file is what makes the compare-and-write below
    # meaningful rather than accidentally re-writing on every run.
    out.sort(key=lambda r: (int(r["MainJobID"]), r["StepKind"]))
    return out


def write_recovery_csv(analysis_dir: Path, rows: list[dict[str, str]]) -> Path | None:
    """Compare-and-write the recovery CSV. Returns the path, or None when nothing was written.

    Compare-and-write rather than unconditional write, for the same reason the DU sentinels
    use it: this file will be declared as a renderer source in Stage B, and an mtime bump on
    byte-identical content is exactly what re-fires an mtime-triggered consumer and what
    produced the StaleReadModelError class. It is also what makes the idempotence property
    testable -- running twice must leave the file byte-identical AND mtime-unchanged.
    """
    eff_dir = analysis_dir.joinpath(*_EFF_RELDIR)
    if not rows:
        return None
    eff_dir.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(RECOVERY_HEADER), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in RECOVERY_HEADER})
    payload = buf.getvalue()
    path = eff_dir / RECOVERY_FILENAME
    if path.exists():
        try:
            if path.read_text() == payload:
                return path
        except OSError:
            pass
    path.write_text(payload)
    return path


def backfill(analysis_dir: Path, *, timeout_s: float = 60.0) -> dict[str, int]:
    """Recover and write for one analysis dir. Returns a coverage report.

    The report is the point, not a side note: a PARTIAL recovery changes what Stage B can
    promise, so the caller gets counts rather than a boolean. `ids_missing` is the number of
    MainJobIDs the CSVs carry for which sacct returned no job row -- jobs aged out of the
    accounting database.
    """
    ids = main_job_ids_from_efficiency_csvs(analysis_dir)
    rows = recover_rows(ids, timeout_s=timeout_s)
    with_job = {r["MainJobID"] for r in rows if r["StepKind"] == "job"}
    with_batch = {r["MainJobID"] for r in rows if r["StepKind"] == "batch"}
    write_recovery_csv(analysis_dir, rows)
    return {
        "ids_in_csv": len(ids),
        "rows_recovered": len(rows),
        "ids_with_job_row": len(with_job),
        "ids_with_batch_row": len(with_batch),
        "ids_missing": len([i for i in ids if i not in with_job]),
    }


def main(argv: list[str] | None = None) -> int:
    """One-shot back-fill CLI: `python -m hhemt.slurm_job_recovery {analysis_dir} [...]`."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print("usage: python -m hhemt.slurm_job_recovery ANALYSIS_DIR [ANALYSIS_DIR ...]", file=sys.stderr)
        return 2
    for raw in args:
        analysis_dir = Path(raw)
        report = backfill(analysis_dir)
        print(
            f"{analysis_dir}: ids_in_csv={report['ids_in_csv']} "
            f"rows={report['rows_recovered']} "
            f"job_rows={report['ids_with_job_row']} "
            f"batch_rows={report['ids_with_batch_row']} "
            f"missing={report['ids_missing']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
