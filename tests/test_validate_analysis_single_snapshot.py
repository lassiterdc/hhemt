"""The aggregator builds the whole-population status table ONCE, and reuses a supplied one.

Unit tier: a counting stub, no solver compile, no simulation, no synthetic-model fixture.
It exercises `analysis_validation` the way `test_coupled_resume_validity.py` does — with a
lightweight stub whose attributes are permissive — and asserts a CALL COUNT rather than a
verdict, so it tolerates a maximally degenerate stub in which most checks fail.

Why a count and not a verdict: every check in `validate_analysis` is exception-tolerant and
returns a failed `CheckResult` rather than propagating, so how many checks succeed against
the stub is irrelevant to what is being measured. The count degrades safely in the only
direction that matters -- if the stub is too thin for one read to be reached, the pre-fix
count is 3 rather than 4, and 3 != 1 still fails.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from hhemt.analysis_validation import persist_validation_report, validate_analysis


def _frame() -> pd.DataFrame:
    """The minimum shape the df_status-reading checks gate on."""
    return pd.DataFrame(
        {
            "model_type": ["tritonswmm", "triton"],
            "n_resumes": [0, 0],
            "event_iloc": [0, 0],
            "scenario_directory": ["/nonexistent/sims/evt-0", "/nonexistent/sims/evt-0"],
            "run_completed": [True, True],
        }
    )


class _PermissiveConfig(SimpleNamespace):
    """A config stub whose UNMODELLED fields resolve rather than raise.

    The explicitly-set fields are the ones a check BRANCHES on, so they carry real
    values; everything else is a field some check reads without branching, and a
    MagicMock lets the check complete and record its own verdict. Without this the
    first unmodelled read aborts the aggregator before the count can be taken, which
    measures the stub rather than the code under test.
    """

    def __getattr__(self, name):
        return MagicMock()


class _CountingAnalysis:
    """Counts every evaluation of `df_status`; permissive for everything else."""

    def __init__(self, analysis_dir):
        self.df_status_reads = 0
        self.analysis_paths = _PermissiveConfig(
            analysis_dir=analysis_dir,
            analysis_datatree_zarr=analysis_dir / "analysis_datatree.zarr",
            sensitivity_datatree_zarr=None,
        )
        self.cfg_analysis = _PermissiveConfig(
            toggle_sensitivity_analysis=False,
            is_experiment_member=False,
            analysis_id="unit-stub",
            execution_environment="native",
            run_mode="serial",
            n_mpi_procs=1,
            n_omp_threads=1,
            n_gpus=0,
            resume_interruption_schedule=None,
            TRITON_reporting_timestep_s=60,
        )
        self._system = _PermissiveConfig(
            cfg_system=_PermissiveConfig(
                toggle_triton_model=True,
                toggle_tritonswmm_model=True,
                toggle_swmm_model=False,
            )
        )

    @property
    def df_status(self):
        self.df_status_reads += 1
        return _frame()

    def __getattr__(self, name):
        # Any surface a check reaches for that this stub does not model explicitly.
        return MagicMock()


@pytest.fixture
def stub(tmp_path):
    return _CountingAnalysis(tmp_path)


def test_validate_analysis_builds_df_status_exactly_once(stub):
    """The aggregator invariant: one build per invocation, not one per consuming check."""
    validate_analysis(stub)
    assert stub.df_status_reads == 1, (
        f"validate_analysis evaluated analysis.df_status {stub.df_status_reads} times; "
        "it must resolve ONE snapshot and pass it to every check that needs it. Each "
        "extra evaluation is a full whole-population walk."
    )


def test_persist_validation_report_reuses_a_supplied_snapshot(stub):
    """The pass-through invariant every B1 spec implements: a supplied frame is reused."""
    persist_validation_report(stub, df_status=_frame())
    assert stub.df_status_reads == 0, (
        f"persist_validation_report evaluated analysis.df_status {stub.df_status_reads} "
        "times despite being handed a frame; the supplied snapshot must reach every "
        "consumer without any rebuild."
    )
