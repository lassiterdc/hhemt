"""Regression test: per-row sensitivity-CSV edit triggers per-member_id rerun.

Verifies the per-member_id input-fingerprint mechanism added in
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
def synth_member_two_row(tmp_path, monkeypatch):
    """Build a 2-row synthetic sensitivity analysis with distinct independent_vars values.

    PINS THE RUNS ROOT, AND THE PLACEMENT IS THE WHOLE POINT. The consuming test
    rebuilds this same `analysis_name` with `start_from_scratch=False`; without a pin
    the two builds resolve to DIFFERENT roots, because test_case_builder.py:373-375
    sends a `start_from_scratch=True` build with no override to a private mkdtemp while
    a `False` build keeps the slug root. The rebuild then sees none of this build's
    flags and every rule re-queues.

    The setenv MUST live here and not in the consuming test body. A fixture body runs
    BEFORE the test body, so a pin written in the test has not executed when THIS build
    runs. Measured both ways: pin-in-test-body leaves the two builds disagreeing,
    pin-here makes them agree, and monkeypatch's function scope carries the value
    through the test body so the rebuild is pinned too. Same construction as
    tests/conftest.py:1088.

    tmp_path is per-test-item, so the four consumers of this fixture each get their own
    root and no longer wipe one another under one analysis_name.
    """
    monkeypatch.setenv("HHEMT_TEST_RUNS_ROOT_OVERRIDE", str(tmp_path))
    csv_path = tmp_path / "sensitivity_2row.csv"
    pd.DataFrame(
        {
            "member_id": ["0", "1"],
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


def _fingerprint_path(case, member_id: str) -> Path:
    return case.analysis.analysis_paths.analysis_dir / "_status" / f"member-{member_id}_inputs.json"


def _read_fingerprint(case, member_id: str) -> dict:
    fp = _fingerprint_path(case, member_id)
    assert fp.exists(), f"Expected fingerprint at {fp}"
    return json.loads(fp.read_text())


def _fingerprint_mtime(case, member_id: str) -> float:
    return _fingerprint_path(case, member_id).stat().st_mtime


def test_fingerprint_files_written_on_first_invocation(synth_member_two_row):
    """R1 + R3: every member_id gets a fingerprint file at _status/member-{member_id}_inputs.json."""
    case = synth_member_two_row
    case.analysis.submit_workflow(mode="local", dry_run=True)

    fp0 = _read_fingerprint(case, "0")
    fp1 = _read_fingerprint(case, "1")

    assert fp0["__schema_version__"] == 1
    assert fp1["__schema_version__"] == 1
    assert fp0["fields"]["n_omp_threads"] == 2
    assert fp1["fields"]["n_omp_threads"] == 4
    # member_id excluded; only independent_vars + sentinel
    assert "member_id" not in fp0["fields"]
    assert "member_id" not in fp1["fields"]


def test_fingerprint_idempotent_when_csv_unchanged(synth_member_two_row):
    """R5: re-invoking with no CSV change preserves fingerprint mtimes."""
    case = synth_member_two_row
    case.analysis.submit_workflow(mode="local", dry_run=True)
    mtime0_before = _fingerprint_mtime(case, "0")
    mtime1_before = _fingerprint_mtime(case, "1")

    # Re-invoke without changing CSV
    case.analysis.submit_workflow(mode="local", dry_run=True)

    assert _fingerprint_mtime(case, "0") == mtime0_before
    assert _fingerprint_mtime(case, "1") == mtime1_before


def test_helper_returns_false_on_unchanged_repeat_call(synth_member_two_row):
    """R2 + R5 (helper-level): _write_member_id_fingerprint returns False when content unchanged."""
    case = synth_member_two_row
    case.analysis.submit_workflow(mode="local", dry_run=True)

    sub = case.analysis.sensitivity.members["0"]
    fp_path = _fingerprint_path(case, "0")
    result = case.analysis.sensitivity._write_member_id_fingerprint(sub, fp_path)
    assert result is False, (
        "Expected _write_member_id_fingerprint to detect unchanged content and skip write; "
        f"got result={result}. This indicates the compare-and-write contract is broken."
    )


def test_empty_independent_vars_degenerate_case(tmp_path):
    """R7: when independent_vars is empty, fingerprints are no-op (empty fields dict)."""
    csv_path = tmp_path / "sensitivity_degenerate.csv"
    pd.DataFrame({"member_id": ["0", "1"]}).to_csv(csv_path, index=False)
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
def test_one_row_edit_triggers_only_that_chain(synth_member_two_row):
    """R4: editing one row reruns only that member_id's chain (full execution; slow).

    The only test in this module that runs a FULL (non-dry-run) workflow, so it
    is the only one needing the compiled binaries; the dry-run fingerprint tests
    stay ungated. Skips without cmake+mpic++; HARD-FAILS under
    HHEMT_REQUIRE_COMPILE_TIER=1."""
    case = synth_member_two_row
    # Initial run (full execution, not dry-run) so flag files exist for member_0 and member_1
    case.analysis.submit_workflow(mode="local")

    # Edit only member_id=1's value in the parent CSV
    df = pd.read_csv(case.sensitivity_csv_path)
    df.loc[df["member_id"].astype(str) == "1", "n_omp_threads"] = 8
    df.to_csv(case.sensitivity_csv_path, index=False)

    # CRITICAL: case.analysis.sensitivity.members is built ONCE at construction
    # (sensitivity_analysis.py:284 -> read_csv at :1482) and is NOT re-read by
    # submit_workflow. The per-member_id fingerprint (workflow.py:6865) is computed from
    # the in-memory member, so reusing the stale case would reproduce the
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
    assert "simulation_member_1" in stdout, "Expected member_1 simulation rule to be queued; full output:\n" + stdout
    assert "simulation_member_0" not in stdout, (
        "Did not expect member_0 simulation rule to be queued; full output:\n" + stdout
    )


@pytest.mark.slow
def test_dry_run_orphan_cleanup_does_not_mutate(tmp_path, monkeypatch):
    """FORBIDS: cleanup_all_orphans(dry_run=True) mutating anything. That, and only that.

    RENAMED because the old name claimed more than the body checked. The call below is a
    DRY run, and all three assertions are "nothing changed", so they hold whatever the
    deleting path does -- the test had no power over row-removal invalidation (its title)
    or over delete-everything (its assertion message). Those two are now
    test_row_removal_preserves_surviving_fingerprints and
    test_orphan_cleanup_deletes_only_the_orphan, each with a call that can fail it.

    What survives here is real: a regression that made dry_run=True mutate would redden
    all three assertions, and that is the guarantee this test now names.

    THE PIN GOES IN THE TEST BODY HERE, unlike synth_member_two_row's, because BOTH of
    this test's builds are in the body -- so the placement that is wrong for a
    fixture/test pair is the correct one for this shape.
    """
    monkeypatch.setenv("HHEMT_TEST_RUNS_ROOT_OVERRIDE", str(tmp_path))
    csv_path = tmp_path / "sensitivity_3row.csv"
    pd.DataFrame(
        {
            "member_id": ["0", "1", "2"],
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
    mtime_0_before = (analysis_dir / "_status" / "member-0_inputs.json").stat().st_mtime
    mtime_2_before = (analysis_dir / "_status" / "member-2_inputs.json").stat().st_mtime
    orphan_path = analysis_dir / "_status" / "member-1_inputs.json"
    assert orphan_path.exists()

    # Remove member_id=1 from the CSV
    pd.DataFrame(
        {
            "member_id": ["0", "2"],
            "run_mode": ["openmp", "openmp"],
            "n_mpi_procs": [1, 1],
            "n_omp_threads": [2, 8],
            "n_gpus": [0, 0],
            "n_nodes": [1, 1],
        }
    ).to_csv(csv_path, index=False)
    # Rebuild from the row-removed CSV: sensitivity.df_setup (sensitivity_analysis.py
    # :1678) is frozen at construction, so the stale in-memory df would still list member_1
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

    # ASSERT ON THE TREE THAT WAS ACTED ON. `analysis_dir` was bound from the FIRST build
    # and `case` was rebound before the call above, so the two were different directories
    # whenever the roots diverged -- the action could not reach the files being asserted.
    # Re-deriving from the acting object makes them the same tree BY CONSTRUCTION rather
    # than by coincidence, and the equality below is what says so out loud.
    acted_dir = case.analysis.analysis_paths.analysis_dir
    assert acted_dir == analysis_dir, (
        "the rebuild resolved to a different tree than the first build; the assertions "
        "below would be reading a directory this test never touched"
    )
    assert (acted_dir / "_status" / "member-1_inputs.json").exists(), (
        "a dry run must leave the orphan fingerprint in place"
    )
    assert (acted_dir / "_status" / "member-0_inputs.json").stat().st_mtime == mtime_0_before
    assert (acted_dir / "_status" / "member-2_inputs.json").stat().st_mtime == mtime_2_before


@pytest.mark.slow
@pytest.mark.usefixtures("tritonswmm_cpu_compiled")
@pytest.mark.xfail(
    strict=True,
    raises=NotImplementedError,
    reason=(
        "BODY NOT YET WRITTEN. Completing it needs a full non-dry-run 3-row workflow, "
        "which compiles a solver and is simulation-bearing, so it needs the permission "
        "that governs solver-executing runs. strict=True is deliberate: when the body "
        "lands and passes, XPASS fails this test until the marker is removed."
    ),
)
def test_row_removal_preserves_surviving_fingerprints(tmp_path, monkeypatch):
    """FORBIDS: a row removal invalidating the fingerprints of the members that remain.

    This is the intent the old test's TITLE claimed and its body never exercised: it
    asserted unchanged mtimes across a dry-run cleanup, which mutates nothing, so the
    assertion held whatever row removal did. Here the mtimes are read after the REBUILD
    -- the path that actually rewrites fingerprints -- so a compare-and-write regression
    that bumped every member's mtime would redden this and nothing else.
    """
    monkeypatch.setenv("HHEMT_TEST_RUNS_ROOT_OVERRIDE", str(tmp_path))
    # Build the 3-row case, run it, record survivors' fingerprint mtimes, remove one row,
    # rebuild and re-submit, then assert only the removed member's chain moved.
    raise NotImplementedError("body follows the 3-row construction of test_dry_run_orphan_cleanup_does_not_mutate")


@pytest.mark.slow
@pytest.mark.usefixtures("tritonswmm_cpu_compiled")
@pytest.mark.xfail(
    strict=True,
    raises=NotImplementedError,
    reason=(
        "BODY NOT YET WRITTEN. Completing it needs cleanup_all_orphans(dry_run=False) "
        "over a real 3-row run, which compiles a solver and is simulation-bearing, so it "
        "needs the permission that governs solver-executing runs. strict=True is "
        "deliberate: when the body lands and passes, XPASS fails this test until the "
        "marker is removed."
    ),
)
def test_orphan_cleanup_deletes_only_the_orphan(tmp_path, monkeypatch):
    """FORBIDS: orphan cleanup deleting anything beyond the orphan.

    POSITIVE AND MIXED, because all-negative assertions cannot express this: the orphan
    MUST be gone and the survivors MUST remain, and a test that only asserted absence of
    change would fail on the first half while a test that only asserted deletion would
    pass on a delete-everything regression. This is the call the old test could not make
    -- dry_run=False -- and is why it cannot share a test with the dry-run arm.
    """
    monkeypatch.setenv("HHEMT_TEST_RUNS_ROOT_OVERRIDE", str(tmp_path))
    raise NotImplementedError(
        "body follows the 3-row construction of test_dry_run_orphan_cleanup_does_not_mutate, then "
        "case.analysis.sensitivity.cleanup_all_orphans(dry_run=False, force=True, verbose=False)"
    )
