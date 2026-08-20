"""Stage-B reducer tests ([Q153]): the `seff` reduction at job grain.

Each test names its INVARIANT and states what a DIFFERENT correct implementation scores.
The CPU fixture is SYNTHETIC and deliberately so: on the real job 18396137 `TotalCPU` is
nonzero on `.batch` ONLY, so sum, max, first-nonzero and last-nonzero all return the same
number and certify nothing. That job discriminates fine for memory and elapsed and is used
verbatim for both.
"""

from __future__ import annotations

from hhemt.report_renderers.metadata import (
    _EFF_COLUMNS,
    _JOB_RECORD,
    _MAX_STEPS,
    _SUM_STEPS,
    _aggregate_jobs,
    _attempt_details_html,
    _reduce_cpu_sum,
    _reduce_rss_max,
    _slurm_rss_mb,
    _slurm_seconds,
)

# Real job 18396137: five steps, TotalCPU nonzero on `.batch` only, peak memory on the
# `python` wrapper rather than the solver, job Elapsed neither max nor sum of steps.
_REAL_STEPS = {
    # TRESUsageInTot values are the REAL captured ones for this job (see
    # tests/test_slurm_job_recovery.py): TotalCPU reads zero on every srun step while the
    # recorded usage shows the CPU that was actually consumed.
    # `JobName` is the step's IDENTITY and the reducer classifies on it, never on the step
    # index. This job carries the MAJORITY shape -- `python` at `.0` -- which is 4,771 of
    # the campaign's 5,198 jobs; the 319-job minority whose `.0` IS a solver step is a
    # different fixture, because a majority-shape fixture cannot reach that branch.
    "18396137.0": {"JobID": "18396137.0", "MainJobID": "18396137", "Elapsed": "00:01:51",
                   "JobName": "python",
                   "TotalCPU": "00:00:00", "MaxRSS": "491288K", "RequestedMem_MB": "4000",
                   "TRESUsageInTot": "cpu=00:00:17,energy=0,mem=491288K"},
    "18396137.1": {"JobID": "18396137.1", "MainJobID": "18396137", "Elapsed": "00:00:15",
                   "JobName": "triton.exe",
                   "TotalCPU": "00:00:00", "MaxRSS": "49524K", "State": "CANCELLED",
                   "TRESUsageInTot": "cpu=00:00:13,energy=0,mem=49524K"},
    "18396137.4": {"JobID": "18396137.4", "MainJobID": "18396137", "Elapsed": "00:00:20",
                   "JobName": "triton.exe",
                   "TotalCPU": "00:00:00", "MaxRSS": "50860K", "State": "COMPLETED"},
}
_REAL_RECOVERY = {
    "18396137": {
        "job": {"JobID": "18396137", "Elapsed": "00:02:00", "TotalCPU": "00:03.744",
                "NNodes": "1", "NCPUS": "1"},
        "batch": {"JobID": "18396137.batch", "Elapsed": "00:02:00", "TotalCPU": "00:03.744",
                  "MaxRSS": "327556K"},
    }
}


def test_cpu_is_summed_and_the_fixture_actually_discriminates_a_sum():
    """INVARIANT: CPU is the arithmetic SUM over every step, including `.batch`.

    SYNTHETIC fixture with THREE nonzero steps (1 + 2 + 4 = 7), chosen so the sum is distinct
    from max (4), first (1), last (4) and mean (~2.33). A DIFFERENT correct implementation
    PASSES however it iterates; `max`, `first`, `last` and `mean` each FAIL. Using the real
    job here would pass for all four, which is the non-discriminating shape this guards.
    """
    steps = [
        {"JobID": "9.batch", "TotalCPU": "00:00:01", "TRESUsageInTot": "cpu=00:00:01,mem=1K"},
        {"JobID": "9.0", "TotalCPU": "00:00:02", "TRESUsageInTot": "cpu=00:00:02,mem=1K"},
        {"JobID": "9.1", "TotalCPU": "00:00:04", "TRESUsageInTot": "cpu=00:00:04,mem=1K"},
    ]
    value, provenance = _reduce_cpu_sum(steps, {})
    assert float(value) == 7.0
    for wrong in (4.0, 1.0, 7.0 / 3):
        assert float(value) != wrong
    assert "summed over 3 of 3 step(s)" in provenance
    assert "batch" in provenance, "the batch step must be visible as a contributor"


def test_cpu_reduction_recovers_the_column_that_reads_zero_today():
    """INVARIANT: CPU is nonzero for a job whose every step reports `TotalCPU=00:00:00`.

    AMENDED. This test previously asserted `3.744` and called that "the defect closure" --
    but 3.744 s is the BATCH WRAPPER's own CPU, and folding it in replaced an obvious zero
    with a plausible-looking 3% on a job whose steps actually consumed 30 s (~27%). The
    test passed and certified a false number, which is worse than a red test.

    `slurm_job_recovery.py` states the contract directly: `TotalCPU` "MUST NOT be the
    source of a CPU-efficiency figure". The numerator is each step's recorded usage
    (`TRESUsageInTot`'s `cpu=`), which is why that field is recovered.

    A DIFFERENT correct implementation PASSES iff it reads recorded usage; reading
    `TotalCPU` FAILS, whichever steps it includes.
    """
    rows = _aggregate_jobs(_REAL_STEPS, _REAL_RECOVERY)
    assert float(rows[0]["cpu_seconds"]) == 30.0
    assert float(rows[0]["cpu_seconds"]) != 3.744, "batch-wrapper CPU is not the job's CPU"
    assert all(_slurm_seconds(s["TotalCPU"]) == 0 for s in _REAL_STEPS.values())


def test_memory_is_the_max_and_names_the_step_that_produced_it():
    """INVARIANT: memory is the MAX across steps, and the cell discloses which step.

    The real job discriminates: the peak (491288K, the `python` wrapper) is neither the first
    step in iteration order nor the last, and the sum (~919 MB) differs from it. A DIFFERENT
    correct max PASSES; `sum`, `first`, `last` and solver-only each FAIL. Solver-only failing
    is the point -- it would report ~50 MB for a job that needed ~480 MB.
    """
    steps = [_REAL_RECOVERY["18396137"]["batch"], *_REAL_STEPS.values()]
    value, provenance = _reduce_rss_max(steps, {})
    assert round(float(value), 1) == 479.8
    assert "from step 0" in provenance, provenance
    assert round(float(value), 1) != round(_slurm_rss_mb("49524K"), 1)  # not solver-only


def test_elapsed_is_read_from_the_job_row_and_is_not_step_derivable():
    """INVARIANT: Elapsed comes from the job's own record, not from any step reduction.

    Asserted against BOTH candidate derivations rather than against the literal, so a
    DIFFERENT correct implementation reaching 00:02:00 by another route PASSES. Step-max
    (00:01:51) and step-sum (00:02:26) both FAIL, and neither equals the job's 00:02:00.
    """
    rows = _aggregate_jobs(_REAL_STEPS, _REAL_RECOVERY)
    job_s = _slurm_seconds(rows[0]["Elapsed"])
    step_s = [_slurm_seconds(s["Elapsed"]) for s in _REAL_STEPS.values()]
    assert job_s == 120.0
    assert job_s != max(step_s)
    assert job_s != sum(step_s)


def test_grain_is_one_row_per_job_and_the_float_is_gone():
    """INVARIANT ([Q145] + {27}): one row per job, and `Job ID` carries no step suffix.

    A DIFFERENT correct implementation PASSES under any job-grained output. Step grain FAILS
    on the row count; rendering the raw step `JobID` FAILS on the suffix assertion.
    """
    rows = _aggregate_jobs(_REAL_STEPS, _REAL_RECOVERY)
    assert len(rows) == 1, "three step rows must collapse to one job row"
    assert rows[0]["JobID"] == "18396137"
    assert "." not in rows[0]["JobID"]


def test_per_attempt_disclosure_survives_job_grain():
    """INVARIANT ([Q143]/[Q153]): the CANCELLED/COMPLETED breakdown is not folded to a count.

    A DIFFERENT correct implementation PASSES under any shape carrying per-attempt state --
    a roster, a nested table, a tooltip. What FAILS is a bare count, which is the cost the
    ruling exists to avoid, and a roster that omits `State`.
    """
    rows = _aggregate_jobs(_REAL_STEPS, _REAL_RECOVERY)
    html = _attempt_details_html(rows[0]["_attempts"])
    assert "CANCELLED" in html and "COMPLETED" in html
    # Each entry keeps its true STEP index. Called with no ledger, as here, the roster must
    # NOT also assert an attempt number: the step suffix is an srun index and the attempt
    # number is a fact about the run's history, and deriving the second from the first would
    # fabricate one for every step the ledger does not record.
    assert "step 1" in html and "step 4" in html
    assert "attempt not recorded" in html, "no ledger was supplied, so no attempt may be named"
    assert "00:00:15" in html, "each attempt's own cost must survive"


def test_wrapper_step_is_not_counted_as_a_resume_attempt():
    """INVARIANT: a numeric step is an attempt iff its NAME is not a wrapper or bookkeeping
    step. Step INDEX is not step IDENTITY.

    This fixture is the MAJORITY shape (`python` at `.0`, 4,771 of 5,198 campaign jobs), so
    an index rule and a name rule agree on it and it cannot discriminate them on its own --
    the minority shape whose `.0` IS a solver step lives in its own test.

    What this test still pins is that a wrapper is excluded BY NAME: `python` at `.0` is not
    an attempt, and the two `triton.exe` steps are. A DIFFERENT correct implementation PASSES
    however it spells the name set. Counting the wrapper FAILS -- it would report 3 attempts
    for a job that made 2.

    The previous assertion `all(... rsplit(".", 1)[1] != "0" ...)` was DELETED rather than
    repaired: it asserted the refuted premise directly and universally, and no fixture makes
    it true in general, because on 319 of 5,198 campaign jobs a `.0` step IS the solver.
    """
    rows = _aggregate_jobs(_REAL_STEPS, _REAL_RECOVERY)
    assert rows[0]["attempts"] == "2"
    assert [s["JobName"] for s in rows[0]["_attempts"]] == ["triton.exe", "triton.exe"]


def test_every_column_declares_a_reduction_and_the_rule_is_stated_once():
    """INVARIANT ([Q153] single source): each column carries a reduction, and each distinct
    rule sentence exists in exactly ONE declaration -- headers/tooltips/caption render it.

    A DIFFERENT correct implementation PASSES under any declaration shape whose rule text is
    not duplicated across records. What FAILS is two `_Reduction` objects carrying the same
    sentence, which is the restatement the standing mandate forbids.
    """
    assert all(col.reduction is not None for col in _EFF_COLUMNS)
    rules = [r.rule for r in {col.reduction for col in _EFF_COLUMNS}]
    assert len(rules) == len(set(rules)), "a rule sentence must not be stated twice"
    # The symbol is not a restatement: it is meaningless without the rule that expands it.
    assert _SUM_STEPS.tag not in _SUM_STEPS.rule
    assert _MAX_STEPS.tag not in _MAX_STEPS.rule
    assert _JOB_RECORD.tag not in _JOB_RECORD.rule


def test_memory_requested_is_adjacent_to_memory_used():
    """INVARIANT ([Q144]): `Req mem` sits next to `Mem used`, and both are retained.

    A DIFFERENT correct implementation PASSES with either ordering of the adjacent pair.
    What FAILS is separating them, or dropping either on universality grounds -- the ruling's
    retain test is truthfulness.
    """
    keys = [c.key for c in _EFF_COLUMNS]
    assert abs(keys.index("RequestedMem_MB") - keys.index("mem_used_pct")) == 1
