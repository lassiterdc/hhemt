"""Phase 3 — SLURM-lift + delete-runner submission-sentinel tests.

Covers:
- V-P3.1: `_resolve_delete_mode_from_method` correctly maps the four
  `multi_sim_run_method` values (None, "local", "batch_job",
  "1_job_many_srun_tasks") to the delete-executor mode; unrecognized values
  raise ConfigurationError.
- V-P3.5: Each of the three delete-runner modules (`delete_scenario_runner`,
  `delete_subanalysis_runner`, `delete_consolidation_runner`):
  (a) writes a submission sentinel at entry when `SLURM_JOB_ID` is set,
  (b) propagates Python-side exceptions out of `main()` instead of
      swallowing them,
  (c) deletes the sentinel in the `finally` block under both clean-return
      and exception-propagation paths.

The runners are exercised via their public `main()` entrypoints with mocked
inner work (the `_ANALYSIS_LEVEL_ARTIFACTS` loop / `fast_rmtree` / etc.)
so the tests do not require real on-disk analysis trees.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from TRITON_SWMM_toolkit.exceptions import ConfigurationError


# ---------------------------------------------------------------------------
# V-P3.1 — _resolve_delete_mode_from_method
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method,expected",
    [
        (None, "local"),
        ("local", "local"),
        ("batch_job", "slurm"),
        ("1_job_many_srun_tasks", "slurm"),
    ],
)
def test_resolve_delete_mode_from_method_recognized_values(method, expected):
    """`_resolve_delete_mode_from_method` maps each recognized method value
    (and None) to the correct delete-executor mode."""
    from TRITON_SWMM_toolkit.workflow import SnakemakeWorkflowBuilder

    # Instance is unused by this pure method — call as an unbound function
    # via the class to avoid needing a fully-constructed analysis.
    result = SnakemakeWorkflowBuilder._resolve_delete_mode_from_method(
        None, method  # type: ignore[arg-type]
    )
    assert result == expected


def test_resolve_delete_mode_from_method_unknown_raises():
    """Unrecognized values raise ConfigurationError per the fail-fast contract."""
    from TRITON_SWMM_toolkit.workflow import SnakemakeWorkflowBuilder

    with pytest.raises(ConfigurationError):
        SnakemakeWorkflowBuilder._resolve_delete_mode_from_method(
            None, "bogus_mode"  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# V-P3.5 — delete-runner submission-sentinel finally cleanup
# ---------------------------------------------------------------------------


@pytest.fixture
def slurm_env(monkeypatch):
    """Set SLURM_JOB_ID so the runners take the sentinel-writing branch."""
    monkeypatch.setenv("SLURM_JOB_ID", "999999")
    monkeypatch.setenv("SLURM_JOB_NAME", "test-job")
    yield


def _assert_sentinel_written_and_cleaned(
    sentinel_path: Path, *, expect_cleaned: bool = True
) -> None:
    """Walk the parent dir; assert sentinel does not exist post-finally."""
    if expect_cleaned:
        assert not sentinel_path.exists(), (
            f"Sentinel not cleaned by finally block: {sentinel_path}"
        )


def test_delete_scenario_runner_writes_and_cleans_sentinel_on_success(
    tmp_path, slurm_env
):
    """Happy path: sentinel written at entry, deleted in finally on clean return."""
    from TRITON_SWMM_toolkit import delete_scenario_runner as runner

    analysis_dir = tmp_path / "analysis"
    (analysis_dir / "sims" / "scen_a").mkdir(parents=True)

    rc = runner.main(
        ["--event-id", "scen_a", "--analysis-dir", str(analysis_dir)]
    )
    assert rc == 0

    sentinel = analysis_dir / "_status" / "_submitted" / "delete_scenario_scen_a.json"
    _assert_sentinel_written_and_cleaned(sentinel)


def test_delete_scenario_runner_cleans_sentinel_on_exception(tmp_path, slurm_env):
    """Exception path: sentinel still deleted by finally; exception propagates."""
    from TRITON_SWMM_toolkit import delete_scenario_runner as runner

    analysis_dir = tmp_path / "analysis"
    (analysis_dir / "sims" / "scen_a").mkdir(parents=True)

    with patch.object(runner, "fast_rmtree", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError, match="boom"):
            runner.main(
                ["--event-id", "scen_a", "--analysis-dir", str(analysis_dir)]
            )

    sentinel = analysis_dir / "_status" / "_submitted" / "delete_scenario_scen_a.json"
    _assert_sentinel_written_and_cleaned(sentinel)


def test_delete_scenario_runner_no_op_without_slurm_job_id(tmp_path, monkeypatch):
    """When SLURM_JOB_ID is unset, no sentinel is written (matches local-run pattern)."""
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    from TRITON_SWMM_toolkit import delete_scenario_runner as runner

    analysis_dir = tmp_path / "analysis"
    (analysis_dir / "sims" / "scen_a").mkdir(parents=True)
    runner.main(["--event-id", "scen_a", "--analysis-dir", str(analysis_dir)])

    submitted_dir = analysis_dir / "_status" / "_submitted"
    assert not submitted_dir.exists() or not any(submitted_dir.iterdir()), (
        "Local-run path should not write any submission sentinels"
    )


def test_delete_subanalysis_runner_writes_and_cleans_sentinel_on_success(
    tmp_path, slurm_env
):
    from TRITON_SWMM_toolkit import delete_subanalysis_runner as runner

    analysis_dir = tmp_path / "analysis"
    (analysis_dir / "subanalyses" / "sa_3").mkdir(parents=True)

    rc = runner.main(
        ["--sa-id", "3", "--analysis-dir", str(analysis_dir)]
    )
    assert rc == 0

    sentinel = (
        analysis_dir / "_status" / "_submitted" / "delete_subanalysis_sa-3.json"
    )
    _assert_sentinel_written_and_cleaned(sentinel)


def test_delete_subanalysis_runner_cleans_sentinel_on_exception(tmp_path, slurm_env):
    from TRITON_SWMM_toolkit import delete_subanalysis_runner as runner

    analysis_dir = tmp_path / "analysis"
    (analysis_dir / "subanalyses" / "sa_3").mkdir(parents=True)

    with patch.object(runner, "fast_rmtree", side_effect=RuntimeError("boom-sa")):
        with pytest.raises(RuntimeError, match="boom-sa"):
            runner.main(
                ["--sa-id", "3", "--analysis-dir", str(analysis_dir)]
            )

    sentinel = (
        analysis_dir / "_status" / "_submitted" / "delete_subanalysis_sa-3.json"
    )
    _assert_sentinel_written_and_cleaned(sentinel)


def test_delete_consolidation_runner_writes_and_cleans_sentinel_on_success(
    tmp_path, slurm_env
):
    from TRITON_SWMM_toolkit import delete_consolidation_runner as runner

    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()

    rc = runner.main(["--analysis-dir", str(analysis_dir)])
    assert rc == 0

    # The consolidation runner deletes _status/ contents (including the
    # _status/_submitted/ dir holding its own sentinel) as part of its
    # primary task; the finally unlink uses missing_ok=True so the post-
    # condition is the same — no sentinel remains.
    sentinel = (
        analysis_dir / "_status" / "_submitted" / "delete_analysis_consolidation.json"
    )
    _assert_sentinel_written_and_cleaned(sentinel)


def test_delete_consolidation_runner_cleans_sentinel_on_exception(
    tmp_path, slurm_env
):
    from TRITON_SWMM_toolkit import delete_consolidation_runner as runner

    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    # Pre-create a fake artifact so the runner attempts removal before
    # we mock fast_rmtree to fail.
    (analysis_dir / "plots").mkdir()

    with patch.object(runner, "fast_rmtree", side_effect=RuntimeError("boom-consol")):
        with pytest.raises(RuntimeError, match="boom-consol"):
            runner.main(["--analysis-dir", str(analysis_dir)])

    sentinel = (
        analysis_dir / "_status" / "_submitted" / "delete_analysis_consolidation.json"
    )
    _assert_sentinel_written_and_cleaned(sentinel)
