"""CPU efficiency is sourced from recorded step usage, never from `TotalCPU`.

`slurm_job_recovery.py` states the contract its own field set was chosen to satisfy:
`TotalCPU` "MUST NOT be the source of a CPU-efficiency figure", because it reads
`00:00:00` for any work performed in an `srun` step and therefore carries only the batch
wrapper's CPU on a simulation job. The consumer read exactly that field, which produced a
FALSE SMALL efficiency rather than an obviously broken one.

The step rows below are REAL captured `sacct` output for job 18396137, copied from
`tests/test_slurm_job_recovery.py`, so these tests discriminate against measured data
rather than against a constructed example.
"""

from hhemt.report_renderers.metadata import _reduce_cpu_sum, _tres_cpu_seconds

# Real sacct rows, job 18396137: TotalCPU is zero on every srun step while the recorded
# per-step usage shows the work that actually happened.
_BATCH = {
    "JobID": "18396137.batch",
    "TotalCPU": "00:03.744",
    "TRESUsageInTot": "cpu=00:00:03,energy=0,fs/disk=81960071,gres/gpumem=0,mem=327556K",
}
_PYTHON = {
    "JobID": "18396137.0",
    "TotalCPU": "00:00:00",
    "TRESUsageInTot": "cpu=00:00:17,energy=0,fs/disk=398636714,gres/gpumem=0,mem=491288K",
}
_SOLVER = {
    "JobID": "18396137.1",
    "TotalCPU": "00:00:00",
    "TRESUsageInTot": "cpu=00:00:13,energy=0,fs/disk=0,gres/gpumem=0,mem=49524K",
}


def test_solver_step_cpu_is_recovered_where_totalcpu_reads_zero():
    """The discriminating row: TotalCPU says the solver used no CPU; usage says 13 s."""
    assert _SOLVER["TotalCPU"] == "00:00:00"
    assert _tres_cpu_seconds(_SOLVER["TRESUsageInTot"]) == 13.0


def test_sum_counts_the_solver_not_only_the_batch_wrapper():
    """Pre-fix this summed TotalCPU to 3.744 s (batch only); the true total is 33 s.

    On a 120 s / 1-CPU job that is ~3% versus ~27% -- an order of magnitude, and in the
    direction that reads as a real, terrible measurement rather than as a broken column.
    """
    value, prov = _reduce_cpu_sum([_BATCH, _PYTHON, _SOLVER], {})
    assert float(value) == 33.0
    assert "3.744" not in value
    assert "3 of 3" in prov


def test_absent_usage_reads_blank_rather_than_zero():
    """An unpopulated recovery CSV must render an em-dash, never assert 0% efficiency."""
    rows = [{"JobID": "1.batch", "TotalCPU": "00:03.744", "TRESUsageInTot": ""}]
    value, prov = _reduce_cpu_sum(rows, {})
    assert value == ""
    assert "no step reported CPU time" in prov


def test_a_measured_zero_is_kept_distinct_from_an_absent_measurement():
    """cpu=00:00:00 is a measurement; a missing cpu= key is not. The sum must include the
    former (as 0) and exclude the latter, or the two become indistinguishable."""
    measured_zero = {"JobID": "1.0", "TRESUsageInTot": "cpu=00:00:00,mem=1024K"}
    absent = {"JobID": "1.1", "TRESUsageInTot": "mem=1024K"}
    assert _tres_cpu_seconds(measured_zero["TRESUsageInTot"]) == 0.0
    assert _tres_cpu_seconds(absent["TRESUsageInTot"]) is None

    value, prov = _reduce_cpu_sum([measured_zero, absent], {})
    assert value == "0.000"
    assert "1 of 2" in prov
    assert "all zero" in prov


def test_parser_tolerates_malformed_and_missing_fields():
    for junk in ("", "   ", "mem=1024K", "cpu", "cpu=", "no-equals-sign"):
        assert _tres_cpu_seconds(junk) is None


# --- item 1: the Attempts scope limit is authored once and rendered everywhere ---------


def test_attempts_scope_limit_is_one_declaration_rendered_in_tooltip_and_caption():
    """[Q153] forbids restating a definition; the header tooltip and the caption must both
    READ the same `_Reduction.rule` rather than each carry their own prose.

    Pre-fix the caption filtered out join-class reductions, so a caveat on `Attempts`
    reached the reader on hover only -- which is the same true-looking-number-with-unstated-
    scope failure the caveat exists to prevent.
    """
    from hhemt.report_renderers.metadata import _ATTEMPTS_JOIN, _EFF_COLUMNS, _reduction_caption

    attempts = next(c for c in _EFF_COLUMNS if c.key == "attempts")
    assert attempts.reduction is _ATTEMPTS_JOIN, "Attempts must not share the generic join"

    caption = _reduction_caption()
    # The caption renders the rule itself, not a paraphrase of it.
    assert "1_job_many_srun_tasks" in caption
    assert "not recoverable" in caption

    # And the tooltip source is the same object, so there is exactly one place to edit.
    tooltips = tuple(c.reduction.rule for c in _EFF_COLUMNS)
    assert _ATTEMPTS_JOIN.rule in tooltips
    assert caption.count("one shared allocation") == 1, "stated once, not repeated"
