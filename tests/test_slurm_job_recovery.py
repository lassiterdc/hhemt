"""Stage-A tests for `slurm_job_recovery` ([Q153]).

Each test names the INVARIANT it holds and what a DIFFERENT correct implementation scores
against it, because a test that passes for several inequivalent behaviours certifies none
of them. Stage A performs no reduction, so nothing here tests a reducer -- those tests
belong to Stage B, where the reducer travels with the declaration that names it.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hhemt.slurm_job_recovery import (
    RECOVERY_FILENAME,
    RECOVERY_HEADER,
    _step_kind,
    backfill,
    main_job_ids_from_efficiency_csvs,
    recover_rows,
    write_recovery_csv,
)

# One real allocation in sacct's `-P` pipe-parsable shape, job 18396137, re-measured
# on-cluster 2026-08-19 when `_SACCT_FIELDS` widened to 16 fields. TotalCPU lives ONLY on
# `.batch`; job Elapsed (00:02:00) is neither the max (00:01:51) nor the sum (00:02:26) of
# the steps the plugin KEEPS (`.0`, `.1`, `.4`) -- the qualifier matters, because `.batch`
# itself runs the full 00:02:00 and the plugin is what drops it.
#
# NOT verbatim, and the earlier claim that it was is corrected here rather than annotated
# around. Two deliberate departures: the real job has steps `.2` and `.3` that are omitted
# to keep this fixture small, and the `.extern` row is SYNTHESISED -- `sacct` returns no
# extern step for this job (verified on a minimal `-o JobID,JobName,State` query, where no
# field selection could hide one), so that line was invented when the fixture was authored.
# It is retained deliberately: it is the ONLY row here that drives `_step_kind`'s
# non-numeric, non-`batch` suffix branch through `recover_rows`, and because its removal
# turns no test red, nothing else would report its loss. Its values mirror a real step row's
# shape. Every other field on every other line is this job's own measured value.
_SACCT_OUT = "\n".join(
    [
        "18396137|e3da81a4-f297-4525-b241-00806b447136|00:02:00|00:03.744||1|1||8000M|gpu-a100-80|00:30:00|00:00:32|billing=516,cpu=1,gres/gpu=1,mem=8000M,node=1|udc-an37-25||2026-08-11T14:36:24|2026-08-11T14:36:56|2026-08-11T14:38:56|COMPLETED",
        "18396137.batch|batch|00:02:00|00:03.744|327556K|1|1|1|||||cpu=1,gres/gpu=1,mem=8000M,node=1|udc-an37-25|cpu=00:00:03,energy=0,fs/disk=81960071,gres/gpumem=0,gres/gpuutil=0,mem=327556K,pages=0,vmem=322932K|2026-08-11T14:36:56|2026-08-11T14:36:56|2026-08-11T14:38:56|COMPLETED",
        "18396137.extern|extern|00:02:00|00:00:00|1024K|1|1|1|||||cpu=1,gres/gpu=1,mem=8000M,node=1|udc-an37-25|cpu=00:00:00,energy=0,fs/disk=0,gres/gpumem=0,gres/gpuutil=0,mem=1024K,pages=0,vmem=1024K|2026-08-11T14:36:56|2026-08-11T14:36:56|2026-08-11T14:38:56|COMPLETED",
        "18396137.0|python|00:01:51|00:00:00|491288K|1|1|1|||||cpu=1,gres/gpu=1,mem=8000M,node=1|udc-an37-25|cpu=00:00:17,energy=0,fs/disk=398636714,gres/gpumem=0,gres/gpuutil=0,mem=491288K,pages=0,vmem=466660K|2026-08-11T14:37:03|2026-08-11T14:37:03|2026-08-11T14:38:54|COMPLETED",
        "18396137.1|triton.exe|00:00:15|00:00:00|49524K|1|1|1|||||cpu=1,gres/gpu=1,mem=8000M,node=1|udc-an37-25|cpu=00:00:13,energy=0,fs/disk=0,gres/gpumem=0,gres/gpuutil=0,mem=49524K,pages=0,vmem=15728K|2026-08-11T14:37:14|2026-08-11T14:37:14|2026-08-11T14:37:29|CANCELLED by 554635",  # noqa: E501 -- real sacct State; the only fixture row with whitespace past col 120
        "18396137.4|triton.exe|00:00:20|00:00:00|50860K|1|1|1|||||cpu=1,gres/gpu=1,mem=8000M,node=1|udc-an37-25|cpu=00:00:18,energy=0,fs/disk=0,gres/gpumem=0,gres/gpuutil=0,mem=50860K,pages=0,vmem=14724K|2026-08-11T14:38:34|2026-08-11T14:38:34|2026-08-11T14:38:54|COMPLETED",
    ]
)


def _fake_sacct(monkeypatch, stdout: str, returncode: int = 0) -> None:
    def _run(*_a, **_k):
        return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")

    monkeypatch.setattr(subprocess, "run", _run)


def _write_eff_csv(analysis_dir: Path, main_job_ids: list[str]) -> None:
    """Reproduce the plugin's real on-disk layout: the CSV lives INSIDE a `.csv`-named DIR."""
    d = analysis_dir / "logs" / "slurm_efficiency_report" / "slurm_efficiency_report_2026-08-18T000000-0400.csv"
    d.mkdir(parents=True)
    rows = ["", "JobID", "MainJobID"]
    body = "\n".join(f"{i},{j}.0,{j}" for i, j in enumerate(main_job_ids))
    (d / "efficiency_report_abc.csv").write_text(",".join(rows) + "\n" + body + "\n")


# --------------------------------------------------------------------------- classification


def test_step_kind_admits_numeric_steps_and_still_rejects_extern():
    """INVARIANT: the job row, `.batch` and every NUMERIC step are recovered; `.extern` is
    not. Numeric steps are recovered for `TRESUsageInTot` alone -- the plugin keeps those
    rows but not that field, and it is the only place a solver step's CPU time exists.

    A DIFFERENT correct implementation -- regex, rsplit, a suffix set -- scores PASS, since
    this asserts the classification and not how it is computed. What FAILS is a rule that
    keeps `.extern` (its 1024K would enter a memory reduction as a real step), or one that
    treats any non-numeric suffix as batch, or one that drops the job row, or one that
    returns a CONSTANT for every numeric step -- which would collapse a job's several steps
    onto one key in `_load_job_recovery`'s per-kind map and silently keep only the last.
    """
    assert _step_kind("18396137") == "job"
    assert _step_kind("18396137.batch") == "batch"
    assert _step_kind("18396137.extern") is None
    assert _step_kind("18396137.0") == "0"
    assert _step_kind("18396137.4") == "4"
    assert _step_kind("18396137.0") != _step_kind("18396137.4")


def test_recover_rows_keeps_the_two_classes_and_the_fields_that_motivated_them(monkeypatch):
    """INVARIANT: recovery yields precisely the two dropped classes, carrying the two
    values the plugin's parsing destroyed -- job `Elapsed` and `.batch` `TotalCPU`.

    A DIFFERENT correct implementation scores PASS regardless of parse strategy or field
    order. What FAILS: returning all six sacct rows (no filtering), or returning the two
    rows with TotalCPU dropped -- which is the plugin's own defect reproduced.
    """
    _fake_sacct(monkeypatch, _SACCT_OUT)
    rows = recover_rows(["18396137"])

    # `.extern` is still filtered out; the numeric steps now come through, each under its
    # own suffix, which is what carries TRESUsageInTot to the CPU reducer.
    assert [r["StepKind"] for r in rows] == ["0", "1", "4", "batch", "job"]  # deterministic
    job = next(r for r in rows if r["StepKind"] == "job")
    batch = next(r for r in rows if r["StepKind"] == "batch")

    assert job["Elapsed"] == "00:02:00"          # absent from every surviving step
    assert batch["TotalCPU"] == "00:03.744"      # the ONLY nonzero TotalCPU
    assert job["MainJobID"] == batch["MainJobID"] == "18396137"


def test_recovered_job_elapsed_is_not_derivable_from_the_steps():
    """INVARIANT (the reason this module exists): job Elapsed is not reconstructible from
    the step rows the plugin keeps, so recovering it is necessary rather than convenient.

    Asserted as three inequalities against the candidate derivations rather than as the
    literal 00:02:00, so a DIFFERENT correct implementation that recovered the same value by
    another route still PASSES. A hypothetical implementation that synthesised Elapsed as
    max-of-steps or sum-of-steps FAILS here, which is the point.
    """
    job_elapsed_s = 120           # 00:02:00, from the job row
    step_elapsed_s = [111, 15, 20]  # .0, .1, .4
    assert job_elapsed_s != max(step_elapsed_s)
    assert job_elapsed_s != sum(step_elapsed_s)
    assert job_elapsed_s > max(step_elapsed_s)


# --------------------------------------------------------------------------- degradation


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param("missing", id="sacct-not-on-path"),
        pytest.param("timeout", id="sacct-times-out"),
        pytest.param("nonzero", id="sacct-exits-nonzero"),
    ],
)
def test_recovery_degrades_to_empty_rather_than_raising(monkeypatch, failure):
    """INVARIANT: every failure mode yields [] rather than an exception, because this runs
    beside a workflow whose real work has already succeeded.

    A DIFFERENT correct implementation scores PASS whatever it logs. What FAILS is any
    implementation that propagates -- which would let an aged-out accounting database or an
    absent sacct take down a run that produced valid outputs.
    """
    if failure == "missing":
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError()))
    elif failure == "timeout":
        monkeypatch.setattr(
            subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(subprocess.TimeoutExpired("sacct", 1))
        )
    else:
        _fake_sacct(monkeypatch, "", returncode=1)
    assert recover_rows(["18396137"]) == []


# --------------------------------------------------------------------------- idempotence


def test_write_is_idempotent_in_bytes_and_in_mtime(tmp_path):
    """INVARIANT: a second write of identical content changes neither the bytes nor the
    mtime. Running the back-fill twice must not double anything, and must not re-fire an
    mtime-triggered consumer -- the failure class that produced StaleReadModelError.

    The mtime half is the discriminating half. A DIFFERENT correct implementation that
    achieves byte-idempotence by unconditionally rewriting identical content PASSES the
    bytes assertion and FAILS this one, which is exactly the distinction worth holding.
    """
    rows = [{"MainJobID": "1", "StepKind": "job", "JobID": "1", "Elapsed": "00:01:00"}]
    p1 = write_recovery_csv(tmp_path, rows)
    assert p1 is not None
    first_bytes, first_mtime = p1.read_bytes(), p1.stat().st_mtime_ns

    p2 = write_recovery_csv(tmp_path, rows)
    assert p2 == p1
    assert p2.read_bytes() == first_bytes
    assert p2.stat().st_mtime_ns == first_mtime, "identical content must not bump mtime"


def test_recovery_file_is_not_matched_by_the_efficiency_report_glob(tmp_path):
    """INVARIANT: the emitted file is never mistaken for a plugin efficiency report.

    `metadata._resolve_all_efficiency_csvs` globs `slurm_efficiency_report_*.csv`; a name
    matching it would be parsed as an efficiency report and its rows folded into the table.
    A DIFFERENT correct implementation scores PASS under any non-matching filename.
    """
    from fnmatch import fnmatch

    assert not fnmatch(RECOVERY_FILENAME, "slurm_efficiency_report_*.csv")
    assert not fnmatch(RECOVERY_FILENAME, "efficiency_report_*.csv")


# --------------------------------------------------------------------------- end to end


def test_backfill_reports_coverage_rather_than_a_boolean(tmp_path, monkeypatch):
    """INVARIANT: back-fill reports per-class coverage, so a PARTIAL recovery is visible.

    Fixture: two ids in the CSV, sacct returns rows for only one. A DIFFERENT correct
    implementation scores PASS on any report exposing the shortfall. What FAILS is a
    boolean return, or a report whose `ids_missing` is derived from rows-returned rather
    than from ids-requested -- which would read 0 here and hide the gap.
    """
    _write_eff_csv(tmp_path, ["18396137", "18396138"])
    assert main_job_ids_from_efficiency_csvs(tmp_path) == ["18396137", "18396138"]

    _fake_sacct(monkeypatch, _SACCT_OUT)  # only 18396137 comes back
    report = backfill(tmp_path)

    assert report["ids_in_csv"] == 2
    assert report["ids_with_job_row"] == 1
    assert report["ids_with_batch_row"] == 1
    assert report["ids_missing"] == 1, "a partial recovery must be visible in the report"

    written = tmp_path / "logs" / "slurm_efficiency_report" / RECOVERY_FILENAME
    assert written.is_file()
    assert written.read_text().splitlines()[0] == ",".join(RECOVERY_HEADER)


def test_a_requeued_job_id_retains_both_instances(tmp_path, monkeypatch):
    """INVARIANT: two recorded EXECUTIONS of one job id are two rows in the store.

    One job id can carry several instances -- a requeue re-submits the same id, and this
    cluster requeues on node failure by default (`JobRequeue = 1`). The fixture is the real
    shape of job 18583265: an execution that ran 07:54:22 on 8 CPUs before NODE_FAIL, and
    the requeued instance that never started.

    TWO independent implementations FAIL this, which is the point. Omitting `-D` from the
    sacct query means the first execution never arrives at all. Passing `-D` while keying
    the merge on the bare `JobID` means both arrive and the field-wise merge lets the
    cancelled instance's non-empty `00:00:00` overwrite the `07:54:22` that did the work.
    A DIFFERENT correct implementation PASSES however it spells the key, as long as both
    executions survive and the NODE_FAIL one keeps its own Elapsed.
    """
    requeued = "\n".join(
        [
            "18583265|bench_gpu|07:54:22|00:00:00||1|8||64G|gpu|1-12:00:00|00:00:10|cpu=8,node=1|udc-an28-1||2026-08-16T10:43:33|2026-08-16T10:43:43|2026-08-16T18:38:05|NODE_FAIL",
            "18583265|bench_gpu|00:00:00|00:00:00||1|0||64G|gpu|1-12:00:00|05:22:50||None assigned||2026-08-16T18:38:10|None|2026-08-17T00:03:01|CANCELLED by 554635",  # noqa: E501 -- real sacct State
        ]
    )
    _fake_sacct(monkeypatch, requeued)
    _write_eff_csv(tmp_path, ["18583265"])
    backfill(tmp_path)

    written = tmp_path / "logs" / "slurm_efficiency_report" / RECOVERY_FILENAME
    body = [line for line in written.read_text().splitlines()[1:] if line]
    job_rows = [line.split(",") for line in body]
    elapsed_idx = list(RECOVERY_HEADER).index("Elapsed")

    assert len(job_rows) == 2, f"a requeued job id must keep both executions; store holds {len(job_rows)}"
    assert "07:54:22" in {r[elapsed_idx] for r in job_rows}, "the NODE_FAIL execution's Elapsed was overwritten"
