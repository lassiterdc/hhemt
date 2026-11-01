"""Regression tests: a per-row sensitivity-CSV edit reruns only that member_id's chain,
and orphan cleanup after a row is removed.

The per-member_id input fingerprint, `_status/member-{member_id}_inputs.json`, is
written when the master Snakefile is generated
(`SensitivityAnalysisWorkflowBuilder.generate_master_snakefile_content`) and is an
input of that member's rules. Uses the synthetic-test-model tier. Every build pins its
runs root under `tmp_path` except the one in `test_empty_independent_vars_degenerate_case`,
a `start_from_scratch=True` build with no pin, which `retrieve_synth_TRITON_SWMM_test_case`
sends to a private temporary root. The compiled software stays under the toolkit's shared
user cache root (`hhemt.cache.paths.cache_root()`) in every case.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest
import yaml

from tests.fixtures.golden_trees import construct_golden_analysis
from tests.fixtures.test_case_builder import _TEST_RUNS_ROOT_OVERRIDE_ENV, retrieve_synth_TRITON_SWMM_test_case
from tests.utils_dry_run_plan import planned_jobs

pytestmark = pytest.mark.requires_snakemake_subprocess


@pytest.fixture
def synth_member_two_row(tmp_path, monkeypatch):
    """Build a 2-row synthetic sensitivity analysis with distinct independent_vars values.

    PINS THE RUNS ROOT, AND THE PLACEMENT IS THE WHOLE POINT. The consuming test
    rebuilds this same `analysis_name` with `start_from_scratch=False`; without a pin
    the two builds resolve to DIFFERENT roots, because `retrieve_synth_TRITON_SWMM_test_case`
    sends a `start_from_scratch=True` build with no override to a private mkdtemp while
    a `False` build keeps the slug root. The rebuild then sees none of this build's
    flags and every rule re-queues.

    The setenv MUST live here and not in the consuming test body. A fixture body runs
    BEFORE the test body, so a pin written in the test has not executed when THIS build
    runs, and the two builds would disagree. Pinned here, monkeypatch's function scope
    carries the value through the test body, so the rebuild is pinned too. Same
    construction as the function-scoped builders in tests/conftest.py, such as
    `synth_multi_sim_analysis`.

    tmp_path is per-test-item, so the four consumers of this fixture each get their own
    root and no longer wipe one another under one analysis_name.
    """
    monkeypatch.setenv(_TEST_RUNS_ROOT_OVERRIDE_ENV, str(tmp_path))
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
    """Every member_id gets a fingerprint file at _status/member-{member_id}_inputs.json."""
    case = synth_member_two_row
    case.analysis.submit_workflow(mode="local", dry_run=True)

    fp0 = _read_fingerprint(case, "0")
    fp1 = _read_fingerprint(case, "1")

    assert fp0["__schema_version__"] == 1
    assert fp1["__schema_version__"] == 1
    assert fp0["fields"]["n_omp_threads"] == 2
    assert fp1["fields"]["n_omp_threads"] == 4
    # member_id is not a field: "fields" holds only the analysis independent vars,
    # beside the top-level "__schema_version__" sentinel
    assert "member_id" not in fp0["fields"]
    assert "member_id" not in fp1["fields"]


def test_fingerprint_idempotent_when_csv_unchanged(synth_member_two_row):
    """Re-invoking with no CSV change preserves fingerprint mtimes."""
    case = synth_member_two_row
    case.analysis.submit_workflow(mode="local", dry_run=True)
    mtime0_before = _fingerprint_mtime(case, "0")
    mtime1_before = _fingerprint_mtime(case, "1")

    # Re-invoke without changing CSV
    case.analysis.submit_workflow(mode="local", dry_run=True)

    assert _fingerprint_mtime(case, "0") == mtime0_before
    assert _fingerprint_mtime(case, "1") == mtime1_before


def test_helper_returns_false_on_unchanged_repeat_call(synth_member_two_row):
    """At helper level, _write_member_id_fingerprint returns False when content unchanged."""
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
    """When independent_vars is empty, each fingerprint holds an empty fields dict."""
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


# ---- plan rows: the sensitivity golden is planned with a dry run, so no solver is compiled or run ------------

_STATUS_REPORT_JOBS = frozenset(
    {
        "all",
        "export_scenario_status",
        "plot_disk_utilization",
        "plot_errors_and_warnings",
        "plot_metadata",
        "plot_per_analysis_summary_table",
        "plot_scenario_status_appendix",
        "plot_system_overview",
        "plot_workflow_performance",
        "render_report[format=zip]",
    }
    | {
        f"plot_per_sim_per_member_{kind}[member_id={m},event_id=event_index.0]"
        for kind in ("conduit_flow", "peak_flood_depth")
        for m in range(4)
    }
)


def _golden_member_csv(root: Path) -> Path:
    """The sensitivity CSV an extracted sensitivity golden points at."""
    return Path(yaml.safe_load((root / "analysis_config.yaml").read_text())["sensitivity_analysis"])


def _chain_of_member(member_id: int) -> frozenset[str]:
    """The jobs of one member's chain that a stale fingerprint schedules, plus the experiment consolidation."""
    return frozenset(
        {
            f"prepare_member_{member_id}_evt_event_index_0",
            f"simulation_member_{member_id}_evt_event_index_0",
            f"process_member_{member_id}_evt_event_index_0",
            f"consolidate_member_{member_id}",
            "experiment_consolidation",
        }
    )


def test_one_row_edit_triggers_only_that_chain(golden_extractor):
    """Editing one row's value schedules only that member_id's chain (a dry-run plan over the sensitivity golden).

    The untouched golden plans only its report-stage jobs. Editing member 1's ``n_omp_threads`` in the CSV
    changes member 1's input fingerprint, which is an input of that member's rules, so the plan adds member 1's
    prepare, simulation, process and consolidation jobs and the experiment consolidation, and no job of
    members 0, 2 or 3."""
    root = golden_extractor("sensitivity")
    untouched = planned_jobs(root)
    assert untouched == _STATUS_REPORT_JOBS

    csv_path = _golden_member_csv(root)
    df = pd.read_csv(csv_path)
    df.loc[df["member_id"].astype(str) == "1", "n_omp_threads"] = 8
    df.to_csv(csv_path, index=False)

    edited = planned_jobs(root)
    assert edited - untouched == _chain_of_member(1)
    assert untouched <= edited


def test_dry_run_orphan_cleanup_does_not_mutate(golden_extractor):
    """FORBIDS: cleanup_all_orphans(dry_run=True) deleting the orphan fingerprint or rewriting a survivor's.

    The sensitivity golden is built from a CSV with member 1 removed, so member 1 is an orphan; a DRY-RUN
    cleanup is called; the full CSV is then restored (bytes and time) and the tree is planned. A dry run that
    deleted the orphan's fingerprint or flags, or rewrote any survivor's fingerprint, makes the plan schedule
    that member's chain; a dry run that mutated nothing leaves the plan at the report-stage jobs only."""
    root = golden_extractor("sensitivity")
    csv_path = _golden_member_csv(root)
    stamp = csv_path.stat().st_mtime
    full_csv = csv_path.read_bytes()
    pd.read_csv(csv_path).query("member_id != 1").to_csv(csv_path, index=False)
    os.utime(csv_path, (stamp, stamp))

    row_removed = construct_golden_analysis(root)
    row_removed.sensitivity.cleanup_all_orphans(dry_run=True, force=True, verbose=False)

    csv_path.write_bytes(full_csv)
    os.utime(csv_path, (stamp, stamp))
    assert planned_jobs(root) == _STATUS_REPORT_JOBS


@pytest.mark.slow
@pytest.mark.usefixtures("tritonswmm_cpu_compiled")
@pytest.mark.xfail(
    strict=True,
    raises=NotImplementedError,
    reason=(
        "BODY NOT YET WRITTEN. Completing it needs a full non-dry-run 3-row workflow, "
        "which compiles a solver and runs simulations. strict=True is deliberate: when "
        "the body lands and passes, XPASS fails this test until the marker is removed."
    ),
)
def test_row_removal_preserves_surviving_fingerprints(tmp_path, monkeypatch):
    """FORBIDS: a row removal invalidating the fingerprints of the members that remain.

    This is the intent the old test's TITLE claimed and its body never exercised: it
    asserted unchanged mtimes across a dry-run cleanup, so the assertion held whatever
    row removal did. The planned body reads the mtimes after the REBUILD -- the path
    that rewrites fingerprints -- so that a compare-and-write regression bumping every
    member's mtime would redden it.

    Not yet in force: the body raises NotImplementedError under a strict xfail.
    """
    monkeypatch.setenv(_TEST_RUNS_ROOT_OVERRIDE_ENV, str(tmp_path))
    # Planned body: build the 3-row case, run it, record survivors' fingerprint mtimes,
    # remove one row, rebuild and re-submit, then assert only the removed member's chain moved.
    raise NotImplementedError("body follows the 3-row construction of test_dry_run_orphan_cleanup_does_not_mutate")


@pytest.mark.slow
@pytest.mark.usefixtures("tritonswmm_cpu_compiled")
@pytest.mark.xfail(
    strict=True,
    raises=NotImplementedError,
    reason=(
        "BODY NOT YET WRITTEN. Completing it needs cleanup_all_orphans(dry_run=False) "
        "over a real 3-row run, which compiles a solver and runs simulations. strict=True "
        "is deliberate: when the body lands and passes, XPASS fails this test until the "
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

    Not yet in force: the body raises NotImplementedError under a strict xfail.
    """
    monkeypatch.setenv(_TEST_RUNS_ROOT_OVERRIDE_ENV, str(tmp_path))
    raise NotImplementedError(
        "body follows the 3-row construction of test_dry_run_orphan_cleanup_does_not_mutate, then "
        "case.analysis.sensitivity.cleanup_all_orphans(dry_run=False, force=True, verbose=False)"
    )
