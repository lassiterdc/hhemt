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
#:
#: `TotalCPU` is retained for continuity but MUST NOT be the source of a CPU-efficiency
#: figure: measured on Rivanna, it reads `00:00:00` for any work performed in an `srun`
#: step, so on every simulation job it carries the batch-step wrapper's CPU only. The true
#: per-step CPU time is `TRESUsageInTot`'s `cpu=` key, which is why that field is recovered
#: alongside it. `TRESUsageInTot` is also the only place `gres/gpuutil` and `gres/gpumem`
#: appear; it is populated on STEP rows and empty on the job row, and its values are SUMS
#: over the step's tasks (`TRESUsageInMax` is the per-task max), so a per-GPU reading is
#: `Tot / NTasks` — never `Tot / AllocTRES gres/gpu`, which under the whole-node
#: `--exclusive` path counts GPUs the step never bound.
#:
#: `Partition`, `AllocTRES`, `Planned`, `Timelimit` and `NodeList` are job-row fields
#: populated on 100% of rows. They matter because the toolkit-side join is a last-wins
#: snapshot that leaves most historical rows unlabelled; SLURM's own record recovers the
#: partition (the experiment's hardware axis), the granted GPU/CPU/memory, and the queue
#: wait for those rows. `NTasks` is required to normalize the gres keys above.
_SACCT_FIELDS = (
    "JobID",
    "JobName",
    "Elapsed",
    "TotalCPU",
    "MaxRSS",
    "NNodes",
    "NCPUS",
    "NTasks",
    "ReqMem",
    "Partition",
    "Timelimit",
    "Planned",
    "AllocTRES",
    "NodeList",
    "TRESUsageInTot",
    # `Submit` is the requeue discriminator and is REQUIRED, not decorative. `sacct -j`
    # returns only the MOST RECENT instance of a job id, so a requeued job's earlier
    # execution is reachable only under `-D` -- and once `-D` is on, `JobID` alone no
    # longer identifies a row. Measured on Rivanna: `JobRequeue = 1`, and job 18583265
    # ran 07:54:22 on 8 CPUs, hit NODE_FAIL, and was requeued into a cancelled instance;
    # the default query returns ONLY the second, reporting `Elapsed=00:00:00, NCPUS=0`
    # for a job that ran eight hours.
    #
    # `Start`/`End` make the step TOPOLOGY computable from the stored rows: whether a
    # job's solver steps ran SEQUENTIALLY (attempts of one simulation) or CONCURRENTLY
    # (different simulations sharing an allocation) is the fork `RunMethod` declares, and
    # without these two fields nothing stored beside that field can contradict it.
    "Submit",
    "Start",
    "End",
    "State",
)

#: Header of the emitted CSV. `StepKind` is DERIVED here rather than left to the reader:
#: the job/batch distinction is the whole point of the file, and re-deriving it downstream
#: from a suffix regex would be a second place that knowledge lives.
#: `RunMethod` is hhemt's own, not sacct's, and it is REQUIRED by the retention promise
#: rather than merely useful. Under `batch_job` a job's solver steps are ATTEMPTS of one
#: simulation; under `1_job_many_srun_tasks` they are DIFFERENT SIMULATIONS sharing one
#: allocation -- and the accounting rows are structurally identical in the two cases, so a
#: reducer reading sacct alone cannot tell them apart. Without this field the promise that a
#: future aggregation bug is fixable from the stored rows does not hold for that mode: the
#: rows would be there and would still be un-interpretable. It is supplied by the CALLER
#: rather than read here, which keeps this module stdlib-only with no toolkit imports.
RECOVERY_HEADER = ("MainJobID", "StepKind", "RunMethod", *_SACCT_FIELDS)

#: sacct is invoked in chunks. 771 ids at ~9 chars is well under ARG_MAX, but a campaign
#: that grows is exactly the case nobody re-measures, so the bound is explicit.
_CHUNK = 500


def _step_kind(job_id: str) -> str | None:
    """Classify a sacct JobID into the row class this module recovers, or None.

    `18396137`       -> "job"    (the allocation; carries Elapsed)
    `18396137.batch` -> "batch"  (carries the only nonzero TotalCPU)
    `18396137.0`     -> "0"      (a numeric step; recovered for its TRESUsageInTot)
    `18396137.extern`-> None     (never carries anything the table reports)

    NUMERIC STEPS ARE RECOVERED, and the module docstring's "rows the plugin dropped"
    framing is corrected here rather than annotated around. The plugin KEEPS numeric step
    rows, so re-reading them looks like duplication -- but it keeps them WITHOUT
    `TRESUsageInTot`, and that field is the only place a solver step's CPU time exists
    (`TotalCPU` reads `00:00:00` for anything run in an `srun` step). What this module
    supplies is therefore whatever the plugin's CSV does not carry, whether that is a
    missing ROW or a missing FIELD on a row it kept; the original wording named only the
    first axis because only the first axis was in view. The recovered step rows are MERGED
    onto the plugin's own step rows by JobID downstream, never appended -- appending would
    put each solver step in the reduction twice and double a summed CPU figure.

    The numeric suffix is returned AS the kind so `_load_job_recovery`'s
    `{main_job_id: {kind: row}}` map absorbs several steps per job without collision: a
    suffix can never equal "job" or "batch". `backfill`'s per-class counters compare
    against those two literals and so keep their existing meaning.
    """
    if "." not in job_id:
        return "job"
    suffix = job_id.split(".", 1)[1]
    if suffix == "batch":
        return "batch"
    return suffix if suffix.isdigit() else None


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


def _job_ids_from_job_index(analysis_dir: Path) -> set[str]:
    """Job ids from `_status/_job_index.json` -- the executor's own per-job log tree.

    INDEPENDENT of the plugin's CSVs, which is the entire point: it is harvested from
    `.snakemake/slurm_logs/rule_{name}/{wildcards}/{jobid}.log`, where the rule is the
    directory and the job id is the filename, so it covers every job the executor submitted
    rather than every job one downstream parser chose to keep.

    Stdlib only, preserving this module's leaf property. Degrades to an empty set on any
    read or parse failure -- a roster that is short is recoverable on the next capture,
    whereas a raise here would take down a back-fill whose real work is unrelated.
    """
    import json as _json_ji

    path = analysis_dir / "_status" / "_job_index.json"
    if not path.is_file():
        return set()
    try:
        payload = _json_ji.loads(path.read_text())
    except (OSError, ValueError):
        return set()
    if not isinstance(payload, dict):
        return set()
    return {str(k) for k in payload if str(k).isdigit()}


def _job_ids_from_existing_store(analysis_dir: Path) -> set[str]:
    """Job ids already in the store, so an aged-out job stays in the roster.

    Without this the store is a SNAPSHOT of whatever sacct still retains, not an amended
    product: sacct silently omits ids past its retention window, so a later capture would
    quietly shrink the population. Re-querying an aged-out id costs one entry in a batched
    sacct call and returns nothing; the row already held is what survives, via the
    field-wise merge in `write_recovery_csv`.
    """
    path = analysis_dir.joinpath(*_EFF_RELDIR) / RECOVERY_FILENAME
    if not path.is_file():
        return set()
    try:
        text = path.read_text()
    except OSError:
        return set()
    out: set[str] = set()
    for row in csv.DictReader(io.StringIO(text)):
        value = (row.get("MainJobID") or "").strip()
        if value.isdigit():
            out.add(value)
    return out


def recover_rows(job_ids: list[str], *, timeout_s: float = 60.0) -> list[dict[str, str]]:
    """Query sacct for `job_ids` and return the job, `.batch` and numeric-step rows.

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
                # `-D/--duplicates` is REQUIRED, not defensive. When job ids are supplied
                # with `-j`, sacct returns only the MOST RECENT instance of each id; a
                # requeued job's earlier execution is otherwise never returned, and this
                # cluster requeues on node failure by default (`JobRequeue = 1`). Measured:
                # over one campaign window `sacct -D` returned 5221 job rows against 5219
                # without it, and the surviving row for one of the two reported
                # `Elapsed=00:00:00, NCPUS=0` for a job that ran 07:54:22 on 8 CPUs.
                # This flag is only safe alongside the composite `(JobID, Submit)` merge
                # key below -- with the bare `JobID` key the two instances fuse field-wise
                # and the cancelled one overwrites the one that did the work.
                ["sacct", "-D", "-j", ",".join(chunk), "-n", "-P", "-o", ",".join(_SACCT_FIELDS)],
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
    path_existing = eff_dir / RECOVERY_FILENAME
    # AMEND, never replace. The prior form wrote whatever this capture returned, which makes
    # the file a snapshot of sacct's current retention rather than a data product: a job aged
    # out of the accounting database vanished from the store on the next capture, taking a
    # measurement that was already safely recorded with it. Merge field-wise on the full step
    # `JobID`, and NEVER let an empty new value overwrite a non-empty stored one -- that
    # asymmetry is the whole mechanism. A re-run job legitimately updates its own fields; a
    # job sacct no longer knows about contributes nothing and keeps what it had.
    #
    # The key is the PAIR (step JobID, Submit), never JobID alone. One job id can carry
    # several INSTANCES -- a requeue re-submits the same id, and `recover_rows` now passes
    # `-D` so both arrive. Under a bare-JobID key the field-wise rule above fuses them and
    # the later instance wins every non-empty field, so a NODE_FAIL execution's real
    # Elapsed is overwritten by the requeued instance's `00:00:00`. Measured on Rivanna:
    # job 18583265 ran 07:54:22 on 8 CPUs before NODE_FAIL, and its requeued instance ran
    # not at all. Both are real rows and the store keeps both.
    stored: dict[tuple[str, str], dict[str, str]] = {}
    if path_existing.is_file():
        try:
            for prior in csv.DictReader(io.StringIO(path_existing.read_text())):
                key = ((prior.get("JobID") or "").strip(), (prior.get("Submit") or "").strip())
                if key[0]:
                    stored[key] = {k: (prior.get(k) or "") for k in RECOVERY_HEADER}
        except OSError:
            stored = {}
    for row in rows:
        key = ((row.get("JobID") or "").strip(), (row.get("Submit") or "").strip())
        if not key[0]:
            continue
        target = stored.setdefault(key, {k: "" for k in RECOVERY_HEADER})
        for field in RECOVERY_HEADER:
            value = (row.get(field) or "").strip()
            if value:
                target[field] = value
    if not stored:
        return None
    merged_rows = sorted(
        stored.values(),
        key=lambda r: (
            int(r["MainJobID"]) if r.get("MainJobID", "").isdigit() else 0,
            r.get("StepKind", ""),
            r.get("Submit", ""),
        ),
    )
    eff_dir.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(RECOVERY_HEADER), lineterminator="\n")
    writer.writeheader()
    for row in merged_rows:
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


def backfill(analysis_dir: Path, *, timeout_s: float = 60.0, run_method: str = "") -> dict[str, int]:
    """Recover and write for one analysis dir. Returns a coverage report.

    The report is the point, not a side note: a PARTIAL recovery changes what Stage B can
    promise, so the caller gets counts rather than a boolean. `ids_missing` is the number of
    MainJobIDs the CSVs carry for which sacct returned no job row -- jobs aged out of the
    accounting database.
    """
    # The roster is the architectural lever. Sourcing it from the plugin's CSVs alone caps
    # this store at "whatever the plugin already kept", so it can never become the SOLE
    # source -- it is definitionally a subset of one of the sources it is meant to replace.
    # Union three independent rosters instead:
    #   - the executor's own per-job log tree, via _status/_job_index.json, which covers every
    #     job submitted including the ones a later submission's flag-sidecar overwrite forgot;
    #   - the store's OWN existing keys, so a job sacct has since aged out stays in the roster
    #     and keeps its already-captured row rather than silently leaving the population;
    #   - the plugin CSVs, retained so this is additive and nothing that works today regresses.
    ids = sorted(
        {
            *main_job_ids_from_efficiency_csvs(analysis_dir),
            *_job_ids_from_job_index(analysis_dir),
            *_job_ids_from_existing_store(analysis_dir),
        },
        key=int,
    )
    rows = recover_rows(ids, timeout_s=timeout_s)
    # `run_method` is supplied by the CALLER rather than read here, which keeps this module
    # stdlib-only. Stamped on every row because the retention promise depends on it: the step
    # axis means ATTEMPTS under `batch_job` and DIFFERENT SIMULATIONS under
    # `1_job_many_srun_tasks`, and the accounting rows are structurally identical in the two
    # cases, so a stored row that omits the mode cannot be re-aggregated correctly later.
    for _r in rows:
        _r["RunMethod"] = run_method
    with_job = {r["MainJobID"] for r in rows if r["StepKind"] == "job"}
    with_batch = {r["MainJobID"] for r in rows if r["StepKind"] == "batch"}
    write_recovery_csv(analysis_dir, rows)
    # The CONSOLIDATED store, folding this capture together with the executor plugin's own
    # CSVs and the job-to-rule index into ONE dataset. Written here rather than at render
    # time because that is the whole point: the three sources were joined in the renderer,
    # so a disagreement between them surfaced as a wrong cell rather than as an error and
    # no single file could be pointed at as the source of truth.
    #
    # Imported INSIDE the function, deliberately. This module's contract is stdlib-only, no
    # toolkit imports, so a runner subprocess can use it without pulling in the workflow
    # builder (it mirrors `slurm_liveness.py` for that reason). `slurm_store` needs xarray,
    # so a module-level import here would break that contract for every caller including
    # the ones that never consolidate.
    #
    # Best-effort by the same rule that governs the rest of this module: every failure mode
    # degrades to "recovered nothing" rather than raising, because a store write must never
    # take down a workflow whose real work already succeeded. The CSV is already on disk at
    # this point, so a failure here loses the consolidation, not the capture.
    store_written = False
    try:
        from hhemt.slurm_store import consolidate

        store_written = consolidate(analysis_dir, recovery_rows=rows) is not None
    except Exception as exc:  # noqa: BLE001 -- a store write must not fail a workflow
        print(
            f"[job-recovery] WARNING: consolidated store not written ({type(exc).__name__}: {exc}); "
            f"the recovery CSV is unaffected",
            file=sys.stderr,
            flush=True,
        )
    return {
        "store_written": int(store_written),
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
