#!/usr/bin/env python3
"""Read a container smoke run's verdict from two layers, and exit 0 only when both hold.

THE PROBLEM. A launcher's exit status reports that a PROCESS finished. For a container
smoke run that is not the question: the scheduler step can exit 0 while the simulation never
wrote a result (a solver that logged an error and returned, a rule whose shell swallowed the
status), and the toolkit's own ``WorkflowResult.success`` is derived from the workflow's
view of itself rather than from the scheduler's view of the step. So neither number is the
verdict.

THE VERDICT IS A CONJUNCTION OF TWO INDEPENDENT RECORDS, and neither can make the other
true:

  1. the SCHEDULER's record of the simulation step -- ``sacct`` State ``COMPLETED`` and
     ExitCode ``0:0`` for that step's job id; and
  2. the TOOLKIT's completion record -- the per-model log the simulation rule writes, whose
     ``simulation_completed`` field is set once the solver has produced its outputs. This is
     the same log-based completion the toolkit itself reads (``_already_written``), so the
     smoke and the workflow agree on what "finished" means.

A run whose scheduler step is COMPLETED and whose marker is missing FAILED: the step ended
without producing the state a completed simulation leaves. A run with a marker and a
non-zero ExitCode FAILED too: the process reported a failure even though a record exists.

HOW TO USE IT, locally or on the cluster. The two inputs are files, so the reader can be
exercised with no cluster, no scheduler and no solver:

  printf 'COMPLETED|0:0\n' > /tmp/sacct_row.txt        # or paste a real `sacct -P` row
  python scripts/check_container_smoke.py \
      --sacct-row-file /tmp/sacct_row.txt \
      --marker path/to/log_tritonswmm.json

``--sacct-row-file`` takes one ``sacct -P`` line, ``STATE|EXITCODE``, and the two lines above
are the whole of what the local exercise needs: a row you write by hand stands in for the
scheduler, and the marker is the only file the check reads. A record that cannot be read at
all -- a malformed row, a missing file, a host with no ``sacct`` -- exits 2, so it can never
be mistaken for a run that ran and failed (1). Without the row file the script queries the
scheduler itself for ``--job-id``:

  python scripts/check_container_smoke.py --job-id 12345678 --marker {analysis_dir}/...

The launcher's own exit status is accepted with ``--launcher-exit`` and RECORDED in the
output; it never enters the verdict.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REQUIRED_STATE = "COMPLETED"
REQUIRED_EXIT_CODE = "0:0"


def read_sacct_row(row: str) -> tuple[str, str]:
    """Split one ``sacct -P`` row (``STATE|EXITCODE``) into its two fields.

    Empty fields are refused rather than defaulted: ``sacct`` prints ``State`` and
    ``ExitCode`` as empty strings for a job it has no record of, and an absent record must
    not read as a satisfied one.
    """
    parts = [p.strip() for p in row.strip().split("|")]
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"expected one 'STATE|EXITCODE' row with both fields present, got {row!r}")
    return parts[0], parts[1]


def query_sacct(job_id: str) -> tuple[str, str]:
    """Ask the scheduler for a job's State and ExitCode, first row only.

    Uses the accounting database rather than the queue, because this runs after the step has
    finished. Raises rather than defaulting when ``sacct`` cannot be run or prints nothing:
    an unreadable record is an unanswered question, and a verdict function that answers an
    unanswered question is the failure this script exists to remove.
    """
    try:
        done = subprocess.run(
            ["sacct", "-j", str(job_id), "-n", "-X", "-P", "-o", "State,ExitCode"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"sacct could not report on job {job_id}: {exc}") from exc
    lines = [line for line in done.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"sacct reported no row for job {job_id}")
    return read_sacct_row(lines[0])


def read_completion_marker(path: Path, field: str = "simulation_completed") -> bool:
    """Whether the toolkit's completion record says the simulation finished.

    Accepts both shapes the toolkit writes for a completion field: a bare boolean, and the
    ``{"value": ..., ...}`` wrapper the log module stores beside its metadata. A missing or
    unreadable file is NOT an exception -- it is the absence of the record, which is a
    verdict (False), because the caller's question is whether the marker is there.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(payload, dict):
        return False
    value = payload.get(field)
    if isinstance(value, dict):
        value = value.get("value")
    return value is True


def verdict(state: str, exit_code: str, marker_present: bool) -> tuple[bool, list[str]]:
    """The two-layer PASS: both records must hold, and each failing layer is named.

    Returning the failing layers rather than only a boolean is what makes a red run
    diagnosable without re-running it: "the step completed and the marker is missing" and
    "the step failed but a marker exists" are different defects with different owners.
    """
    reasons: list[str] = []
    if state != REQUIRED_STATE:
        reasons.append(f"scheduler state {state!r} != {REQUIRED_STATE!r}")
    if exit_code != REQUIRED_EXIT_CODE:
        reasons.append(f"scheduler exit code {exit_code!r} != {REQUIRED_EXIT_CODE!r}")
    if not marker_present:
        reasons.append("the toolkit's completion marker is absent")
    return (not reasons), reasons


def main(argv: list[str] | None = None) -> int:
    """Read the two records, print the verdict and the reasons, and exit 0 only on a PASS."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--sacct-row-file", type=Path, help="a file holding one 'STATE|EXITCODE' row")
    src.add_argument("--sacct-row", help="one 'STATE|EXITCODE' row inline")
    src.add_argument("--job-id", help="query sacct for this job id")
    ap.add_argument("--marker", type=Path, required=True, help="the simulation rule's model log")
    ap.add_argument(
        "--marker-field",
        default="simulation_completed",
        help="the completion field the marker carries (default: simulation_completed)",
    )
    ap.add_argument(
        "--launcher-exit",
        type=int,
        default=None,
        help="the launcher's own exit status: recorded, never used as the verdict",
    )
    args = ap.parse_args(argv)

    try:
        if args.sacct_row_file is not None:
            rows = [line for line in args.sacct_row_file.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not rows:
                print(f"{args.sacct_row_file} holds no sacct row: nothing to read the scheduler record from")
                return 2
            state, exit_code = read_sacct_row(rows[0])
        elif args.sacct_row is not None:
            state, exit_code = read_sacct_row(args.sacct_row)
        else:
            state, exit_code = query_sacct(args.job_id)
    except (OSError, ValueError, RuntimeError) as exc:
        # UNANSWERED, not FAILED: 2 is the code this script already uses for "nothing to read",
        # so a caller can tell "the run failed" (1) from "the record could not be read" (2).
        print(f"could not read the scheduler record: {exc}")
        return 2

    marker_present = read_completion_marker(args.marker, args.marker_field)
    passed, reasons = verdict(state, exit_code, marker_present)

    print(f"sacct state={state} exit_code={exit_code} (expected {REQUIRED_STATE} / {REQUIRED_EXIT_CODE})")
    print(f"completion marker at {args.marker}: {'present' if marker_present else 'ABSENT'}")
    if args.launcher_exit is not None:
        print(f"launcher exit (recorded, not the verdict): {args.launcher_exit}")
    for reason in reasons:
        print(f"  - {reason}")
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
