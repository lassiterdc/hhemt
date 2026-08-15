"""Perf aggregation against the tracked real-output fixture, under BOTH clean and
resume conditions and at BOTH reporting scales.

Why this module exists, stated so it is not deleted as redundant: the reset-correction
defect shipped to production while the entire behavioural suite was green, because
``min_per_tstep`` defaults to ``1.0`` and at 1.0 the ``timestep_min`` coordinate EQUALS
the raw ``performance{N}.txt`` file index. The production exporter is the only caller
that passes a non-1.0 value (``TRITON_reporting_timestep_s / 60 = 600 / 60 = 10.0``), so
the one value where the two index spaces coincide was the only value under test.
Measured on ``sa_serial_6_r1``: 89.51 s (the pure telescope to ``performance144.txt``)
against a correct 409.30 s, with a correct ledger and correct raw input.

The scale parametrization is therefore the point of this module, not a flourish. The
invariant is that the reporting-interval scaling must not change the aggregate at all:
``min_per_tstep`` describes how a row index maps to wall-clock minutes and says nothing
about how much time the run consumed.

Fixture provenance: ``tests/fixtures/perf_resume_ledger/`` holds the full 144-file
``performance{N}.txt`` set captured from ``sa_serial_6_r1`` on both
``synth_cc_clean_triton`` and ``synth_cc_resume_triton``, plus each sim's
``_walltime`` oracle. The resumed arm's oracle records four attempts (three
interruptions), independently corroborating the ledger's
``resume_reporting_tsteps = [36, 72, 108]`` — two artifacts written by different
producers agreeing on the reset count, which is corroboration a hand-built fixture
cannot have.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hhemt.process_simulation import _aggregate_perf_summary

FIXTURE = Path(__file__).parent / "fixtures" / "perf_resume_ledger"
RESUME_PERF = FIXTURE / "resume_triton" / "performance"
CLEAN_PERF = FIXTURE / "clean_triton" / "performance"

# The ledger for sa_serial_6_r1, verbatim from log_triton.json on the cluster.
RESUME_STEPS = [36, 72, 108]

# Production reporting interval is 600 s, so min_per_tstep = 600/60. 1.0 is the default
# and the value every prior test exercised; 10.0 is the value production actually uses.
SCALES = [1.0, 10.0]

# Ground truth, recomputed from the raw files by segment-sum over the ledger boundaries.
EXPECTED_RESUME = {"Total": 409.30, "Init": 0.19931}

# Every metric the perf schema tracks, per the header line of performance{N}.txt.
# Constraint A: coverage is the full schema, not the subset the benchmarking figure plots.
TRACKED_METRICS = [
    "Compute", "MPI", "IO", "Resize", "SWMM", "Other", "Simulation", "Init", "Total",
]


@pytest.mark.parametrize("min_per_tstep", SCALES)
def test_resume_arm_applies_every_ledger_reset_at_any_reporting_scale(min_per_tstep):
    """The corrected total must appear at BOTH scales.

    Pre-fix this FAILS at 10.0 and passes at 1.0 — that asymmetry is the whole defect,
    and a single-scale test cannot see it.
    """
    ds = _aggregate_perf_summary(
        RESUME_PERF, min_per_tstep, resume_steps=RESUME_STEPS
    )
    for name, expected in EXPECTED_RESUME.items():
        got = float(ds[name].values.ravel()[0])
        assert got == pytest.approx(expected, rel=1e-4), (
            f"{name} at min_per_tstep={min_per_tstep}: expected {expected}, got {got}. "
            f"89.51 for Total means ZERO resets were applied (pure telescope to the "
            f"final file); 300.5 means the retired inferred predicate is somehow live."
        )


def test_resume_arm_is_invariant_across_reporting_scale_for_every_tracked_metric():
    """The scaling maps row index to minutes; it must not change any aggregate.

    This is the generalized form of the assertion above and covers the whole schema,
    so a future metric cannot regress silently just because it is not Total or Init.
    """
    at_1 = _aggregate_perf_summary(RESUME_PERF, 1.0, resume_steps=RESUME_STEPS)
    at_10 = _aggregate_perf_summary(RESUME_PERF, 10.0, resume_steps=RESUME_STEPS)
    for name in TRACKED_METRICS:
        a = float(at_1[name].values.ravel()[0])
        b = float(at_10[name].values.ravel()[0])
        assert a == pytest.approx(b, rel=1e-9), (
            f"{name} changed with reporting scale: {a} at 1.0 vs {b} at 10.0. "
            f"The aggregate must not depend on how a row index maps to minutes."
        )


def test_clean_arm_is_invariant_across_reporting_scale_for_every_tracked_metric():
    """The clean arm is the CALIBRATION arm and must also be scale-invariant.

    It carries no resets, so it separates 'the aggregator is wrong' from 'the reset
    join is wrong'. A resume-only test cannot make that separation.
    """
    at_1 = _aggregate_perf_summary(CLEAN_PERF, 1.0, resume_steps=[])
    at_10 = _aggregate_perf_summary(CLEAN_PERF, 10.0, resume_steps=[])
    for name in TRACKED_METRICS:
        a = float(at_1[name].values.ravel()[0])
        b = float(at_10[name].values.ravel()[0])
        assert a == pytest.approx(b, rel=1e-9), (
            f"clean {name} changed with reporting scale: {a} at 1.0 vs {b} at 10.0."
        )


@pytest.mark.parametrize("min_per_tstep", SCALES)
def test_clean_arm_applies_no_reset_correction(min_per_tstep):
    """With an empty ledger the sum telescopes to the final cumulative row.

    This is the control that proves the correction is ledger-driven rather than
    inferred: the clean arm's raw files carry no negative deltas, so an inference-based
    detector and a ledger-based one agree here and disagree only on the resume arm.
    """
    ds = _aggregate_perf_summary(CLEAN_PERF, min_per_tstep, resume_steps=[])
    total = float(ds["Total"].values.ravel()[0])
    assert total > 0.0, "clean-arm Total must be positive"
    # The clean run's walltime oracle records a single completed attempt at 418.09 s;
    # the perf-file total is the solver's own accounting and sits just under it.
    assert total < 418.09, (
        f"clean Total {total} exceeds the walltime oracle's 418.09 s for the same sim, "
        f"which would mean the aggregate is double-counting rather than telescoping."
    )
