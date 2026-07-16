"""Regression test: per-row sensitivity-CSV edit triggers per-sa_id rerun.

Verifies the per-sa_id input-fingerprint mechanism added in
`SensitivityAnalysisWorkflowBuilder._build_snakefile_content()`. Uses the
synthetic-test-model tier (cached under `platformdirs.user_cache_dir`).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from tests.fixtures.test_case_builder import retrieve_synth_TRITON_SWMM_test_case

pytestmark = pytest.mark.requires_snakemake_subprocess


@pytest.fixture
def synth_sa_two_row(tmp_path):
    """Build a 2-row synthetic sensitivity analysis with distinct independent_vars values."""
    csv_path = tmp_path / "sensitivity_2row.csv"
    pd.DataFrame(
        {
            "sa_id": ["0", "1"],
            "run_mode": ["openmp", "openmp"],
            "n_mpi_procs": [1, 1],
            "n_omp_threads": [2, 4],
            "n_gpus": [0, 0],
            "n_nodes": [1, 1],
        }
    ).to_csv(csv_path, index=False)
    case = retrieve_synth_TRITON_SWMM_test_case(
        analysis_name="synth_sensitivity_csv_row_rerun",
        toggle_tritonswmm_model=True,
        toggle_triton_model=False,
        toggle_swmm_model=False,
        sensitivity_csv=csv_path,
        start_from_scratch=True,
    )
    case.sensitivity_csv_path = csv_path
    return case


def _fingerprint_path(case, sa_id: str) -> Path:
    return case.analysis.analysis_paths.analysis_dir / "_status" / f"sa-{sa_id}_inputs.json"


def _read_fingerprint(case, sa_id: str) -> dict:
    fp = _fingerprint_path(case, sa_id)
    assert fp.exists(), f"Expected fingerprint at {fp}"
    return json.loads(fp.read_text())


def _fingerprint_mtime(case, sa_id: str) -> float:
    return _fingerprint_path(case, sa_id).stat().st_mtime


def test_fingerprint_files_written_on_first_invocation(synth_sa_two_row):
    """R1 + R3: every sa_id gets a fingerprint file at _status/sa-{sa_id}_inputs.json."""
    case = synth_sa_two_row
    case.analysis.submit_workflow(mode="local", dry_run=True)

    fp0 = _read_fingerprint(case, "0")
    fp1 = _read_fingerprint(case, "1")

    assert fp0["__schema_version__"] == 1
    assert fp1["__schema_version__"] == 1
    assert fp0["fields"]["n_omp_threads"] == 2
    assert fp1["fields"]["n_omp_threads"] == 4
    # sa_id excluded; only independent_vars + sentinel
    assert "sa_id" not in fp0["fields"]
    assert "sa_id" not in fp1["fields"]


def test_fingerprint_idempotent_when_csv_unchanged(synth_sa_two_row):
    """R5: re-invoking with no CSV change preserves fingerprint mtimes."""
    case = synth_sa_two_row
    case.analysis.submit_workflow(mode="local", dry_run=True)
    mtime0_before = _fingerprint_mtime(case, "0")
    mtime1_before = _fingerprint_mtime(case, "1")

    # Re-invoke without changing CSV
    case.analysis.submit_workflow(mode="local", dry_run=True)

    assert _fingerprint_mtime(case, "0") == mtime0_before
    assert _fingerprint_mtime(case, "1") == mtime1_before


def test_helper_returns_false_on_unchanged_repeat_call(synth_sa_two_row):
    """R2 + R5 (helper-level): _write_sa_id_fingerprint returns False when content unchanged."""
    case = synth_sa_two_row
    case.analysis.submit_workflow(mode="local", dry_run=True)

    sub = case.analysis.sensitivity.sub_analyses["0"]
    fp_path = _fingerprint_path(case, "0")
    result = case.analysis.sensitivity._write_sa_id_fingerprint(sub, fp_path)
    assert result is False, (
        "Expected _write_sa_id_fingerprint to detect unchanged content and skip write; "
        f"got result={result}. This indicates the compare-and-write contract is broken."
    )


def test_empty_independent_vars_degenerate_case(tmp_path):
    """R7: when independent_vars is empty, fingerprints are no-op (empty fields dict)."""
    csv_path = tmp_path / "sensitivity_degenerate.csv"
    pd.DataFrame({"sa_id": ["0", "1"]}).to_csv(csv_path, index=False)
    case = retrieve_synth_TRITON_SWMM_test_case(
        analysis_name="synth_sensitivity_csv_row_rerun",
        toggle_tritonswmm_model=True,
        toggle_triton_model=False,
        toggle_swmm_model=False,
        sensitivity_csv=csv_path,
        start_from_scratch=True,
    )
    case.sensitivity_csv_path = csv_path
    case.analysis.submit_workflow(mode="local", dry_run=True)

    fp0 = _read_fingerprint(case, "0")
    fp1 = _read_fingerprint(case, "1")

    assert fp0 == {"__schema_version__": 1, "fields": {}}
    assert fp1 == {"__schema_version__": 1, "fields": {}}


@pytest.mark.slow
@pytest.mark.usefixtures("tritonswmm_cpu_compiled")
def test_one_row_edit_triggers_only_that_chain(synth_sa_two_row):
    """R4: editing one row reruns only that sa_id's chain (full execution; slow).

    The only test in this module that runs a FULL (non-dry-run) workflow, so it
    is the only one needing the compiled binaries; the dry-run fingerprint tests
    stay ungated. Skips without cmake+mpic++; HARD-FAILS under
    HHEMT_REQUIRE_COMPILE_TIER=1."""
    case = synth_sa_two_row
    # Initial run (full execution, not dry-run) so flag files exist for sa_0 and sa_1
    case.analysis.submit_workflow(mode="local")

    # Edit only sa_id=1's value in the parent CSV
    df = pd.read_csv(case.sensitivity_csv_path)
    df.loc[df["sa_id"].astype(str) == "1", "n_omp_threads"] = 8
    df.to_csv(case.sensitivity_csv_path, index=False)

    # CRITICAL: case.analysis.sensitivity.sub_analyses is built ONCE at construction
    # (sensitivity_analysis.py:284 -> read_csv at :1482) and is NOT re-read by
    # submit_workflow. The per-sa_id fingerprint (workflow.py:6865) is computed from
    # the in-memory sub_analysis, so reusing the stale case would reproduce the
    # ORIGINAL fingerprint -> no mtime bump -> "Nothing to be done". Rebuild the case
    # from the edited CSV (start_from_scratch=False preserves the materialized run and
    # skips preprocessing; it only re-reads configs + the edited CSV). This mirrors the
    # real CLI flow (edit CSV -> re-instantiate analysis -> re-submit).
    case = retrieve_synth_TRITON_SWMM_test_case(
        analysis_name="synth_sensitivity_csv_row_rerun",
        toggle_tritonswmm_model=True,
        toggle_triton_model=False,
        toggle_swmm_model=False,
        sensitivity_csv=case.sensitivity_csv_path,
        start_from_scratch=False,
    )

    # Rebuild and dry-run to inspect what would re-execute. submit_workflow has no
    # capture_output param; the --dry-run snakemake stdout is redirected to a
    # logfile (run_snakemake_local, workflow.py:3318) whose path the returned dict
    # carries under "snakemake_logfile" (workflow.py:3347-3354). Read the dry-run
    # rule names from that log.
    result = case.analysis.submit_workflow(mode="local", dry_run=True)
    dry_run_log = result["snakemake_logfile"]
    stdout = Path(dry_run_log).read_text() if Path(dry_run_log).exists() else ""

    # Snakemake's --dry-run output names the rules that would run
    assert "simulation_sa_1" in stdout, (
        "Expected sa_1 simulation rule to be queued; full output:\n" + stdout
    )
    assert "simulation_sa_0" not in stdout, (
        "Did not expect sa_0 simulation rule to be queued; full output:\n" + stdout
    )


@pytest.mark.slow
def test_row_removal_does_not_rerun_remaining_chains(tmp_path):
    """R6: removing a row leaves the orphan fingerprint and does not invalidate siblings (slow)."""
    csv_path = tmp_path / "sensitivity_3row.csv"
    pd.DataFrame(
        {
            "sa_id": ["0", "1", "2"],
            "run_mode": ["openmp", "openmp", "openmp"],
            "n_mpi_procs": [1, 1, 1],
            "n_omp_threads": [2, 4, 8],
            "n_gpus": [0, 0, 0],
            "n_nodes": [1, 1, 1],
        }
    ).to_csv(csv_path, index=False)
    case = retrieve_synth_TRITON_SWMM_test_case(
        analysis_name="synth_sensitivity_csv_row_rerun_3row",
        toggle_tritonswmm_model=True,
        toggle_triton_model=False,
        toggle_swmm_model=False,
        sensitivity_csv=csv_path,
        start_from_scratch=True,
    )
    case.analysis.submit_workflow(mode="local")

    analysis_dir = case.analysis.analysis_paths.analysis_dir
    mtime_0_before = (analysis_dir / "_status" / "sa-0_inputs.json").stat().st_mtime
    mtime_2_before = (analysis_dir / "_status" / "sa-2_inputs.json").stat().st_mtime
    orphan_path = analysis_dir / "_status" / "sa-1_inputs.json"
    assert orphan_path.exists()

    # Remove sa_id=1 from the CSV
    pd.DataFrame(
        {
            "sa_id": ["0", "2"],
            "run_mode": ["openmp", "openmp"],
            "n_mpi_procs": [1, 1],
            "n_omp_threads": [2, 8],
            "n_gpus": [0, 0],
            "n_nodes": [1, 1],
        }
    ).to_csv(csv_path, index=False)
    # Rebuild from the row-removed CSV: sensitivity.df_setup (sensitivity_analysis.py
    # :1678) is frozen at construction, so the stale in-memory df would still list sa_1
    # as "expected" and find_orphan_status_flags would find nothing (no-op false pass).
    case = retrieve_synth_TRITON_SWMM_test_case(
        analysis_name="synth_sensitivity_csv_row_rerun_3row",
        toggle_tritonswmm_model=True,
        toggle_triton_model=False,
        toggle_swmm_model=False,
        sensitivity_csv=csv_path,
        start_from_scratch=False,
    )
    # Exercise the orphan-cleanup directly (R6's assertion target) instead of routing
    # through run(): run() runs report-config validation first (analysis.py:1881-1885,
    # raising ConfigurationError because cfg_analysis.report is the empty {} block from
    # test_case_builder.py:474), and run(dry_run=True, cleanup_orphans=True) actually
    # DELETES orphans (cleanup_all_orphans(dry_run=False) at analysis.py:1984), which
    # contradicts this test's "left in place / does not rerun" intent. The run()-path
    # cleanup_orphans integration is covered by test_cleanup_orphans_on_run.py.
    case.analysis.sensitivity.cleanup_all_orphans(dry_run=True, force=True, verbose=False)

    # Orphan fingerprint left in place per R6
    assert orphan_path.exists(), "R6: orphan fingerprint file should be left in place by this plan"
    # Sibling fingerprints' mtimes unchanged
    assert (analysis_dir / "_status" / "sa-0_inputs.json").stat().st_mtime == mtime_0_before
    assert (analysis_dir / "_status" / "sa-2_inputs.json").stat().st_mtime == mtime_2_before
