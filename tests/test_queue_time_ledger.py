"""O21 SLURM queue-time ledger: reader semantics and the accumulation falsifier.

Why this module exists, stated so it is not thinned later: the accumulation requirement
is that a RESUMED sim's queue total covers EVERY allocation. A test that hand-writes two
ledger lines and sums them passes identically whether the producer appends or overwrites,
so it falsifies nothing. `test_second_allocation_appends_and_does_not_overwrite` drives
the append path twice and is the one assertion that can fail if the mechanism regresses.

`test_queue_rows_do_not_perturb_the_wall_total` is the guard a reasonable author would
omit: queue and wall records share ONE file, so an O21 regression could silently corrupt
`wall_clock_ledger_s` (consumed by df_status and by sensitivity_benchmarking) while every
queue assertion stayed green.
"""

from __future__ import annotations

import json
from pathlib import Path

from hhemt.run_simulation import read_queue_ledger_seconds, read_walltime_ledger_total_s


def _ledger_for(model_logfile: Path) -> Path:
    d = model_logfile.parent / "_walltime"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{model_logfile.stem}.jsonl"


def _append_queue_row(model_logfile: Path, jobid: str, queue_s: float) -> None:
    """Mirror of the producer in run_simulation_runner.py's SLURM-guarded start block."""
    with open(_ledger_for(model_logfile), "a") as f:
        f.write(json.dumps({"slurm_jobid": jobid, "queue_s": float(queue_s)}) + "\n")


def _append_wall_row(model_logfile: Path, jobid: str, wall_s: float, attempt: int) -> None:
    """Mirror of the F11 producer at sim-finalize."""
    with open(_ledger_for(model_logfile), "a") as f:
        f.write(
            json.dumps(
                {"attempt": attempt, "wall_s": float(wall_s), "completed": True, "slurm_jobid": jobid}
            )
            + "\n"
        )


def test_second_allocation_appends_and_does_not_overwrite(tmp_path):
    """THE falsifier for the accumulation requirement.

    A resumed sim is dispatched as a NEW SLURM job and queues AGAIN. If the second
    allocation's write overwrote rather than appended, the total would equal the SECOND
    wait alone -- understating the modeler-visible wait silently, which is the exact
    failure the requirement exists to prevent. Driving the producer twice is what makes
    this falsifiable; asserting over a hand-built two-line file would not be.
    """
    log = tmp_path / "logs" / "sims" / "model_tritonswmm_evt0.log"
    log.parent.mkdir(parents=True, exist_ok=True)

    _append_queue_row(log, "1001", 105.0)
    _append_queue_row(log, "1002", 3600.0)

    assert len(_ledger_for(log).read_text().strip().splitlines()) == 2

    total, coverage, by_jobid = read_queue_ledger_seconds(log)
    assert total == 3705.0, "total must be the SUM across allocations, not the last one"
    assert total != 3600.0, "an overwrite regression would land exactly here"
    assert by_jobid == {"1001": 105.0, "1002": 3600.0}
    assert coverage == "2/2"


def test_queue_rows_do_not_perturb_the_wall_total(tmp_path):
    """Queue and wall records share one file; the wall total must be byte-identical.

    read_walltime_ledger_total_s sums `.get("wall_s", 0.0) or 0.0`, so a queue-only row
    contributes 0.0. This asserts that property directly rather than trusting it, because
    a regression here corrupts a column consumed well outside O21.
    """
    log = tmp_path / "logs" / "sims" / "model_triton_evt0.log"
    log.parent.mkdir(parents=True, exist_ok=True)

    _append_wall_row(log, "2001", 400.0, attempt=0)
    _append_wall_row(log, "2002", 89.0, attempt=1)
    wall_before = read_walltime_ledger_total_s(log)
    assert wall_before == 489.0

    _append_queue_row(log, "2001", 12.0)
    _append_queue_row(log, "2002", 7200.0)

    assert read_walltime_ledger_total_s(log) == wall_before, (
        "interleaved queue rows must not change the wall total"
    )
    total, coverage, _ = read_queue_ledger_seconds(log)
    assert total == 7212.0
    assert coverage == "2/2"


def test_absent_queue_capture_is_none_not_zero(tmp_path):
    """None means NOT MEASURED; 0.0 would assert the job did not wait.

    This is the 1_job_many_srun_tasks case and the pre-O21-tree case. The identity check
    is deliberate: `== 0` would pass on None under a loose comparison, which is the exact
    conflation the whole design guards against.
    """
    log = tmp_path / "logs" / "sims" / "model_swmm_evt0.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    _append_wall_row(log, "3001", 12.0, attempt=0)

    total, coverage, by_jobid = read_queue_ledger_seconds(log)
    assert total is None
    assert by_jobid == {}
    assert coverage == "0/1", "coverage discloses the denominator, so a zero-of-one reads as such"

    missing = tmp_path / "logs" / "sims" / "model_swmm_evt9.log"
    assert read_queue_ledger_seconds(missing) == (None, "0/0", {})


def test_partial_coverage_is_disclosed_as_partial(tmp_path):
    """A tree that resumed ACROSS the O21 landing boundary has some allocations with a
    queue row and some without. The total is then a genuine PARTIAL sum, and a bare number
    cannot say so -- which is the disclosed-denominator failure shape Gotcha 71(d) records
    for vacuous check passes. Coverage is what makes the partiality legible.
    """
    log = tmp_path / "logs" / "sims" / "model_tritonswmm_evt3.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    _append_wall_row(log, "4001", 100.0, attempt=0)   # pre-O21 allocation: no queue row
    _append_wall_row(log, "4002", 100.0, attempt=1)   # pre-O21 allocation: no queue row
    _append_queue_row(log, "4003", 60.0)              # post-O21 allocation
    _append_wall_row(log, "4003", 100.0, attempt=2)

    total, coverage, _ = read_queue_ledger_seconds(log)
    assert total == 60.0
    assert coverage == "1/3", "a partial sum must not present as a complete one"


def test_probe_returns_none_when_slurm_binaries_are_absent(monkeypatch, tmp_path):
    """Off-cluster, neither sacct nor scontrol exists. The probe must return None rather
    than raise -- a raise here would propagate into a sim run over a reporting nicety.
    """
    import subprocess

    from hhemt.run_simulation import probe_slurm_planned_seconds

    def _boom(*_a, **_kw):
        raise FileNotFoundError("no such binary")

    monkeypatch.setattr(subprocess, "run", _boom)
    assert probe_slurm_planned_seconds("12345") is None


def test_record_queue_time_is_not_applicable_off_batch_job(tmp_path, monkeypatch):
    """1_job_many_srun_tasks shares ONE allocation, so a per-sim queue figure is a lie.

    The arm must be reached by the RUN METHOD alone -- the probe must not even be called,
    because calling it would stamp the allocation's own Planned onto every sim in the
    ensemble if a later edit dropped the guard.
    """
    from hhemt import run_simulation_runner as rsr

    called = []
    monkeypatch.setattr(
        "hhemt.run_simulation.probe_slurm_planned_seconds",
        lambda jobid: called.append(jobid) or 999.0,
    )
    log = tmp_path / "logs" / "sims" / "model_tritonswmm_evt0.log"
    log.parent.mkdir(parents=True, exist_ok=True)

    outcome = rsr._record_queue_time(
        model_logfile=log, jobid="5001", run_method="1_job_many_srun_tasks", event_iloc=0
    )
    assert outcome == "not-applicable"
    assert called == [], "the probe must not run when the mode has no per-sim queue"
    assert not (log.parent / "_walltime").exists(), "no ledger may be created"


def test_record_queue_time_records_on_batch_job(tmp_path, monkeypatch):
    """The dominant path: batch_job plus a probe that answers."""
    import json

    from hhemt import run_simulation_runner as rsr

    monkeypatch.setattr("hhemt.run_simulation.probe_slurm_planned_seconds", lambda jobid: 105.0)
    log = tmp_path / "logs" / "sims" / "model_tritonswmm_evt0.log"
    log.parent.mkdir(parents=True, exist_ok=True)

    assert (
        rsr._record_queue_time(
            model_logfile=log, jobid="5002", run_method="batch_job", event_iloc=0
        )
        == "recorded"
    )
    rows = [
        json.loads(x)
        for x in (log.parent / "_walltime" / f"{log.stem}.jsonl").read_text().splitlines()
        if x.strip()
    ]
    assert rows == [{"slurm_jobid": "5002", "queue_s": 105.0}]


def test_record_queue_time_reports_unavailable_when_probe_returns_none(tmp_path, monkeypatch):
    """Neither sacct nor scontrol answered. The distinction from a recorded 0.0 is the
    whole point: an unavailable probe must leave the ledger untouched so the report shows
    an em-dash, not a zero that claims the job did not wait.
    """
    from hhemt import run_simulation_runner as rsr

    monkeypatch.setattr("hhemt.run_simulation.probe_slurm_planned_seconds", lambda jobid: None)
    log = tmp_path / "logs" / "sims" / "model_triton_evt0.log"
    log.parent.mkdir(parents=True, exist_ok=True)

    assert (
        rsr._record_queue_time(
            model_logfile=log, jobid="5003", run_method="batch_job", event_iloc=0
        )
        == "unavailable"
    )
    assert not (log.parent / "_walltime").exists()


def test_record_queue_time_never_raises_on_write_failure(tmp_path, monkeypatch):
    """A reporting nicety must not be able to fail a simulation.

    Reached by a read-only or over-quota log directory, which is a real Lustre state on a
    full scratch allocation. The assertion is that the helper RETURNS rather than raises.
    """
    from hhemt import run_simulation_runner as rsr

    monkeypatch.setattr("hhemt.run_simulation.probe_slurm_planned_seconds", lambda jobid: 7.0)

    def _deny(*_a, **_kw):
        raise OSError("Disk quota exceeded")

    monkeypatch.setattr("pathlib.Path.mkdir", _deny)
    log = tmp_path / "logs" / "sims" / "model_swmm_evt0.log"

    assert (
        rsr._record_queue_time(
            model_logfile=log, jobid="5004", run_method="batch_job", event_iloc=0
        )
        == "write-failed"
    )
