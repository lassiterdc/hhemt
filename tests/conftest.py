import contextlib
import hashlib
import os
from pathlib import Path

import pytest

import tests.fixtures.test_case_catalog as cases
from hhemt.workflow import _NON_INTERACTIVE_LOCK_CLEAR_ENV

_SYNTH_SENSITIVITY_REPORT_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs" / "reports" / "synth_sensitivity_report_config.yaml"
)


# census-green-up Phase 1 — builder cache-wipe isolation.
# start_from_scratch=True builder fixtures whose analysis_name collides with a
# session *_completed cache (synth_multi_sim / synth_sensitivity) must NOT
# fast_rmtree-wipe that shared on-disk cache mid-suite (it is the copy-on-read
# source for the session fixtures and their _isolated clones). The builder reads
# HHEMT_TEST_RUNS_ROOT_OVERRIDE (test_case_builder.py) and nests its runs_root
# under that per-test path instead. Function-scoped builders set it via
# monkeypatch.setenv; session-scoped builders use this context manager around
# construction (monkeypatch is function-scoped and unavailable at session scope).
@contextlib.contextmanager
def _runs_root_override_env(path):
    key = "HHEMT_TEST_RUNS_ROOT_OVERRIDE"
    old = os.environ.get(key)
    os.environ[key] = str(path)
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old


# import tests.fixtures.test_case_catalog as cases


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_snakemake_subprocess: test launches Snakemake as a subprocess; "
        "incompatible with pytest-xdist parallel workers (nested parallelism)",
    )


@pytest.fixture(scope="session", autouse=True)
def _pytest_uses_non_interactive_snakemake_lock_clear():
    """Route this pytest session through the non-interactive branch of
    ``_check_and_clear_snakemake_lock`` so ``.snakemake/locks/`` and
    ``.snakemake/incomplete/`` are silently cleared before every snakemake
    invocation. Production / CLI users do not set the sentinel env var, so
    the interactive prompt remains the default outside the test suite.

    Phase 1 of synth-test-isolation-and-runtime (Decision D1-Option-D): the
    helper-body branch is the single firing point for the unconditional
    pre-snakemake clear required by R4. Fixtures do not clear separately;
    this fixture just toggles policy.
    """
    prior = os.environ.get(_NON_INTERACTIVE_LOCK_CLEAR_ENV)
    os.environ[_NON_INTERACTIVE_LOCK_CLEAR_ENV] = "1"
    yield
    if prior is None:
        os.environ.pop(_NON_INTERACTIVE_LOCK_CLEAR_ENV, None)
    else:
        os.environ[_NON_INTERACTIVE_LOCK_CLEAR_ENV] = prior


@pytest.fixture
def norfolk_single_sim_analysis():
    case = cases.Local_TestCases.retrieve_norfolk_all_models_test_case(start_from_scratch=True)
    return case.analysis


@pytest.fixture
def norfolk_single_sim_analysis_cached():
    case = cases.Local_TestCases.retrieve_norfolk_all_models_test_case(start_from_scratch=False)
    return case.analysis


@pytest.fixture
def norfolk_multi_sim_analysis():
    case = cases.Local_TestCases.retrieve_norfolk_multi_sim_test_case(start_from_scratch=True)
    return case.analysis


@pytest.fixture
def norfolk_multi_sim_analysis_cached():
    case = cases.Local_TestCases.retrieve_norfolk_multi_sim_test_case(start_from_scratch=False)
    return case.analysis


@pytest.fixture
def norfolk_sensitivity_analysis():
    case = cases.Local_TestCases.retrieve_norfolk_cpu_config_sensitivity_case(start_from_scratch=True)
    return case.analysis


@pytest.fixture
def norfolk_sensitivity_analysis_cached():
    case = cases.Local_TestCases.retrieve_norfolk_cpu_config_sensitivity_case(start_from_scratch=False)
    return case.analysis


@pytest.fixture
def norfolk_sensitivity_triton_only():
    """TRITON-only sensitivity analysis varying n_omp_threads."""
    case = cases.Local_TestCases.retrieve_norfolk_cpu_config_sensitivity_case_triton_only(start_from_scratch=True)
    return case.analysis


@pytest.fixture
def norfolk_sensitivity_swmm_only():
    """SWMM-only sensitivity analysis varying n_omp_threads."""
    case = cases.Local_TestCases.retrieve_norfolk_cpu_config_sensitivity_case_swmm_only(start_from_scratch=True)
    return case.analysis


# ========== Multi-Model Test Fixtures ==========


@pytest.fixture
def norfolk_triton_only_analysis():
    """TRITON-only analysis (no SWMM coupling)."""
    case = cases.Local_TestCases.retrieve_norfolk_triton_only_test_case(start_from_scratch=True)
    return case.analysis


@pytest.fixture
def norfolk_triton_only_analysis_cached():
    """TRITON-only analysis (cached - for faster iteration)."""
    case = cases.Local_TestCases.retrieve_norfolk_triton_only_test_case(start_from_scratch=False)
    return case.analysis


@pytest.fixture
def norfolk_swmm_only_analysis():
    """SWMM-only analysis (standalone EPA SWMM)."""
    case = cases.Local_TestCases.retrieve_norfolk_swmm_only_test_case(start_from_scratch=True)
    return case.analysis


@pytest.fixture
def norfolk_swmm_only_analysis_cached():
    """SWMM-only analysis (cached - for faster iteration)."""
    case = cases.Local_TestCases.retrieve_norfolk_swmm_only_test_case(start_from_scratch=False)
    return case.analysis


@pytest.fixture
def norfolk_all_models_analysis():
    """Analysis with all models enabled (TRITON, TRITON-SWMM, SWMM)."""
    case = cases.Local_TestCases.retrieve_norfolk_all_models_test_case(start_from_scratch=True)
    return case.analysis


@pytest.fixture
def norfolk_all_models_analysis_cached():
    """Analysis with all models (cached - for faster iteration)."""
    case = cases.Local_TestCases.retrieve_norfolk_all_models_test_case(start_from_scratch=False)
    return case.analysis


@pytest.fixture
def norfolk_triton_and_tritonswmm_analysis():
    """Analysis with TRITON and TRITON-SWMM (no standalone SWMM)."""
    case = cases.Local_TestCases.retrieve_norfolk_triton_and_tritonswmm_test_case(start_from_scratch=True)
    return case.analysis


@pytest.fixture
def norfolk_triton_and_tritonswmm_analysis_cached():
    """Analysis with TRITON and TRITON-SWMM (cached)."""
    case = cases.Local_TestCases.retrieve_norfolk_triton_and_tritonswmm_test_case(start_from_scratch=False)
    return case.analysis


# ========== Synthetic Test Fixtures ==========


@pytest.fixture
def synth_all_models_analysis():
    case = cases.Local_TestCases.retrieve_synth_all_models_test_case(start_from_scratch=True)
    return case.analysis


@pytest.fixture
def synth_all_models_analysis_cached():
    case = cases.Local_TestCases.retrieve_synth_all_models_test_case(start_from_scratch=False)
    return case.analysis


@pytest.fixture
def synth_multi_sim_analysis(tmp_path, monkeypatch):
    # census-green-up Phase 1: isolate this start_from_scratch=True wipe under
    # tmp_path so it cannot fast_rmtree the shared synth_multi_sim session cache.
    monkeypatch.setenv("HHEMT_TEST_RUNS_ROOT_OVERRIDE", str(tmp_path))
    case = cases.Local_TestCases.retrieve_synth_multi_sim_test_case(start_from_scratch=True)
    return case.analysis


@pytest.fixture
def synth_multi_sim_analysis_cached():
    case = cases.Local_TestCases.retrieve_synth_multi_sim_test_case(start_from_scratch=False)
    return case.analysis


@pytest.fixture(scope="session")
def synth_multi_sim_builder(tmp_path_factory):
    """Configured-but-not-run multi-sim analysis (Phase 2,
    synth-test-isolation-and-runtime). No DEM/landuse preprocessing, no compile,
    no scenario prep — sufficient for `generate_snakefile_content`-only tests.
    Session-scoped so the 4 collapsed symmetry tests share one builder."""
    # census-green-up Phase 1: isolate this start_from_scratch=True wipe under a
    # per-session tmp dir so it cannot fast_rmtree the shared synth_multi_sim cache.
    with _runs_root_override_env(tmp_path_factory.mktemp("synth_multi_sim_builder")):
        case = cases.Local_TestCases.retrieve_synth_multi_sim_test_case(
            start_from_scratch=True,
            skip_run=True,
        )
    return case.analysis


@pytest.fixture(scope="session")
def synth_sensitivity_builder(tmp_path_factory):
    """Configured-but-not-run sensitivity analysis (Phase 2,
    synth-test-isolation-and-runtime). Same contract as `synth_multi_sim_builder`
    but consumes the sensitivity-CSV factory."""
    # census-green-up Phase 1: isolate this start_from_scratch=True wipe under a
    # per-session tmp dir so it cannot fast_rmtree the shared synth_sensitivity cache.
    with _runs_root_override_env(tmp_path_factory.mktemp("synth_sensitivity_builder")):
        case = cases.Local_TestCases.retrieve_synth_cpu_config_sensitivity_case(
            start_from_scratch=True,
            skip_run=True,
        )
    return case.analysis


@pytest.fixture
def synthetic_multisim_builder(tmp_path, monkeypatch):
    """Synthetic-tier SnakemakeWorkflowBuilder for at-most-once-guard unit tests.

    Yields ``analysis._workflow_builder`` from a fresh synth multisim
    analysis (start_from_scratch=True) so ``_status/`` starts empty and
    ``_status/_submitted/`` is writable. No simulations are executed —
    Phase 1 unit tests construct sentinel files directly and exercise the
    reconciliation guard's classification logic in isolation.

    The builder exposes ``analysis_paths``, ``_reconcile_inflight_submissions``,
    and ``_recover_inflight_via_comment`` — the exact surface the guard's
    test cases monkeypatch against.
    """
    # census-green-up Phase 1: isolate this start_from_scratch=True wipe under
    # tmp_path so it cannot fast_rmtree the shared synth_multi_sim session cache.
    monkeypatch.setenv("HHEMT_TEST_RUNS_ROOT_OVERRIDE", str(tmp_path))
    case = cases.Local_TestCases.retrieve_synth_multi_sim_test_case(start_from_scratch=True)
    return case.analysis._workflow_builder


@pytest.fixture
def synth_triton_only_analysis():
    case = cases.Local_TestCases.retrieve_synth_triton_only_test_case(start_from_scratch=True)
    return case.analysis


@pytest.fixture
def synth_triton_only_analysis_cached():
    case = cases.Local_TestCases.retrieve_synth_triton_only_test_case(start_from_scratch=False)
    return case.analysis


@pytest.fixture
def synth_swmm_only_analysis():
    case = cases.Local_TestCases.retrieve_synth_swmm_only_test_case(start_from_scratch=True)
    return case.analysis


@pytest.fixture
def synth_swmm_only_analysis_cached():
    case = cases.Local_TestCases.retrieve_synth_swmm_only_test_case(start_from_scratch=False)
    return case.analysis


@pytest.fixture
def synth_triton_and_tritonswmm_analysis():
    case = cases.Local_TestCases.retrieve_synth_triton_and_tritonswmm_test_case(start_from_scratch=True)
    return case.analysis


@pytest.fixture
def synth_triton_and_tritonswmm_analysis_cached():
    case = cases.Local_TestCases.retrieve_synth_triton_and_tritonswmm_test_case(start_from_scratch=False)
    return case.analysis


@pytest.fixture
def synth_sensitivity_analysis(tmp_path, monkeypatch):
    # census-green-up Phase 1: isolate this start_from_scratch=True wipe under
    # tmp_path so it cannot fast_rmtree the shared synth_sensitivity session cache.
    monkeypatch.setenv("HHEMT_TEST_RUNS_ROOT_OVERRIDE", str(tmp_path))
    case = cases.Local_TestCases.retrieve_synth_cpu_config_sensitivity_case(start_from_scratch=True)
    return case.analysis


@pytest.fixture
def synth_sensitivity_analysis_cached():
    case = cases.Local_TestCases.retrieve_synth_cpu_config_sensitivity_case(start_from_scratch=False)
    return case.analysis


@pytest.fixture
def synth_sensitivity_triton_only():
    case = cases.Local_TestCases.retrieve_synth_cpu_config_sensitivity_case_triton_only(start_from_scratch=True)
    return case.analysis


@pytest.fixture
def synth_sensitivity_swmm_only():
    case = cases.Local_TestCases.retrieve_synth_cpu_config_sensitivity_case_swmm_only(start_from_scratch=True)
    return case.analysis


@pytest.fixture
def synth_sensitivity_with_system_overlay():
    case = cases.Local_TestCases.retrieve_synth_cpu_config_sensitivity_case_with_system_overlay(start_from_scratch=True)
    return case.analysis


@pytest.fixture
def synth_sensitivity_mutex_violation():
    case = cases.Local_TestCases.retrieve_synth_cpu_config_sensitivity_case_mutex_violation(start_from_scratch=True)
    return case.analysis


@pytest.fixture
def synth_sensitivity_invalid_overlay():
    case = cases.Local_TestCases.retrieve_synth_cpu_config_sensitivity_case_invalid_overlay(start_from_scratch=True)
    return case.analysis


@pytest.fixture
def synth_sensitivity_legacy_gpu_hardware_override():
    case = cases.Local_TestCases.retrieve_synth_cpu_config_sensitivity_case_legacy_gpu_hardware_override(
        start_from_scratch=True
    )
    return case.analysis


@pytest.fixture
def synth_sensitivity_typo_in_prefixed_column():
    case = cases.Local_TestCases.retrieve_synth_cpu_config_sensitivity_case_typo_in_prefixed_column(
        start_from_scratch=True
    )
    return case.analysis


@pytest.fixture
def synth_sensitivity_with_partition_axis():
    case = cases.Local_TestCases.retrieve_synth_cpu_config_sensitivity_case_with_partition_axis(
        start_from_scratch=True
    )
    return case.analysis


@pytest.fixture
def synth_sensitivity_multi_partition_fanout():
    case = cases.Local_TestCases.retrieve_synth_cpu_config_sensitivity_case_multi_partition_fanout(
        start_from_scratch=True
    )
    return case.analysis


@pytest.fixture
def synth_sensitivity_all_analysis_prefixed():
    case = cases.Local_TestCases.retrieve_synth_cpu_config_sensitivity_case_all_analysis_prefixed(
        start_from_scratch=True
    )
    return case.analysis


@pytest.fixture
def synth_sensitivity_mixed_prefixed_columns():
    case = cases.Local_TestCases.retrieve_synth_cpu_config_sensitivity_case_mixed_prefixed_columns(
        start_from_scratch=True
    )
    return case.analysis


@pytest.fixture(scope="session")
def synthetic_multisim_completed(tritonswmm_cpu_compiled):
    """Yield a synth multisim TRITONSWMM_analysis with sim outputs produced.

    Used by Phase 2 reprocess tests. Session-scoped: the first invocation
    in a pytest session runs the synth multisim through
    ``analysis.run(from_scratch=False, ...)`` if the analysis is not
    already at the ``e_consolidate_complete.flag`` state; subsequent
    invocations reuse the materialized analysis from the test-case cache.

    Stale ``.snakemake/locks/`` / ``.snakemake/incomplete/`` (and the
    reprocess-side ``.snakemake_reprocess/.snakemake/locks/`` /
    ``.snakemake_reprocess/.snakemake/incomplete/``) directories from
    prior interrupted runs are silently cleared before yielding.
    Follow-up: integrate this clear into the broader
    ``_pytest_uses_non_interactive_snakemake_lock_clear`` autouse fixture
    described by stipulation
    ``library/docs/stipulations/hhemt/snakemake fixture setup clears locks and incomplete.md``
    (cross-plan synth-test-isolation work; not yet implemented).
    """
    import shutil
    from pathlib import Path as _Path

    from tests.fixtures.test_case_catalog import Local_TestCases

    case = Local_TestCases.retrieve_synth_multi_sim_test_case(start_from_scratch=False)
    analysis = case.analysis
    analysis_dir = analysis.analysis_paths.analysis_dir

    # Clear stale lock + incomplete subtrees on both the run and reprocess
    # working dirs so a leftover state from an interrupted prior run does
    # not interfere with the fixture's analysis.run() (if it fires) or
    # with the test body's analysis.reprocess() invocations.
    for sub_root_name in (".snakemake", ".snakemake_reprocess"):
        sub_root = analysis_dir / sub_root_name
        if sub_root.exists():
            shutil.rmtree(sub_root / "locks", ignore_errors=True)
            shutil.rmtree(sub_root / "incomplete", ignore_errors=True)
            (sub_root / "log").mkdir(parents=True, exist_ok=True)

    # Ensure the analysis is in a "post-consolidate" state. If
    # ``e_consolidate_complete.flag`` is absent, run() once to materialize.
    consolidate_flag = analysis_dir / "_status" / "e_consolidate_complete.flag"
    if not consolidate_flag.exists():
        report_config = (
            _Path(__file__).resolve().parents[0].parent
            / "tests"
            / "configs"
            / "reports"
            / "synth_multisim_report_config.yaml"
        )
        analysis.run(
            from_scratch=False,
            report_config=report_config if report_config.exists() else None,
        )

    return analysis


@pytest.fixture(scope="session")
def synthetic_sensitivity_completed(tritonswmm_cpu_compiled):
    """Yield a synth sensitivity master analysis with all sims + consolidations produced.

    Used by Phase 3 sensitivity-reprocess tests. Session-scoped: the first
    invocation in a pytest session runs the synth sensitivity master through
    ``sensitivity.submit_workflow(mode="local")`` if the analysis is not
    already at the ``f_consolidate_master_complete.flag`` state; subsequent
    invocations reuse the materialized analysis from the test-case cache.

    Returns the ``TRITONSWMM_sensitivity_analysis`` object (master analysis
    accessible via ``.master_analysis``). Stale ``.snakemake/locks/`` /
    ``.snakemake/incomplete/`` (and the reprocess-side
    ``.snakemake_reprocess/.snakemake/locks/`` /
    ``.snakemake_reprocess/.snakemake/incomplete/``) directories from prior
    interrupted runs are silently cleared before yielding, mirroring the
    Phase 2 fixture pattern (``synthetic_multisim_completed``).

    Canonical EDA-loop fixture (ADR-9/ADR-10). EDA functions
    (check_cross_sim_identity, analysis.eda()) take a TRITONSWMM_analysis;
    pass synthetic_sensitivity_completed.master_analysis (NOT the fixture object
    itself, which is the sensitivity wrapper). Warm-cache precondition: the first
    session invocation against a clean cache pays the full compile + run +
    consolidate cost; a cold-cache run appears to hang for minutes.
    """
    import shutil

    from tests.fixtures.test_case_catalog import Local_TestCases

    case = Local_TestCases.retrieve_synth_cpu_config_sensitivity_case(start_from_scratch=False)
    master_analysis = case.analysis
    sensitivity = master_analysis.sensitivity
    analysis_dir = master_analysis.analysis_paths.analysis_dir

    # Clear stale lock + incomplete subtrees on both the run and reprocess
    # working dirs so a leftover state from an interrupted prior run does
    # not interfere with the fixture's submit_workflow() (if it fires) or
    # with the test body's sensitivity.reprocess() invocations.
    for sub_root_name in (".snakemake", ".snakemake_reprocess"):
        sub_root = analysis_dir / sub_root_name
        if sub_root.exists():
            shutil.rmtree(sub_root / "locks", ignore_errors=True)
            shutil.rmtree(sub_root / "incomplete", ignore_errors=True)
            (sub_root / "log").mkdir(parents=True, exist_ok=True)

    # Ensure the master analysis is in a "post-master-consolidate" state. If
    # ``f_consolidate_master_complete.flag`` is absent, run the master
    # sensitivity workflow once locally to materialize per-sa flags + the
    # master flag + the sensitivity_datatree.zarr.
    master_flag = analysis_dir / "_status" / "f_consolidate_master_complete.flag"
    if not master_flag.exists():
        sensitivity.submit_workflow(mode="local")

    return sensitivity


# ========== D1 copy-on-read isolation (census-green-up Phase 1) ==========
# Per-test isolated clones of the session-scoped ``*_completed`` fixtures, built
# on the in-tree clone helper (``tests/_failing_fixture_helpers.clone_analysis_to_tmp``),
# which copies the FULL system_directory (configs + nested analysis_dir +
# ``subanalyses/``) and re-roots ``system_directory`` in the cloned config — the
# config files live in the system_directory (the parent of analysis_dir), NOT in
# analysis_dir. Mutating consumers depend on these wrappers so a full
# ``pytest tests/`` run yields the same per-test verdict as an isolated ``-k`` run.
@pytest.fixture
def synthetic_multisim_completed_isolated(synthetic_multisim_completed, tmp_path):
    """Per-test isolated copy of ``synthetic_multisim_completed`` (D1)."""
    from tests._failing_fixture_helpers import clone_analysis_to_tmp

    return clone_analysis_to_tmp(synthetic_multisim_completed, tmp_path)


@pytest.fixture
def synthetic_sensitivity_completed_isolated(synthetic_sensitivity_completed, tmp_path):
    """Per-test isolated copy of ``synthetic_sensitivity_completed`` (D1) — clones
    the master system_directory (with ``subanalyses/``) so master AND per-sa
    reprocess paths re-derive under tmp_path."""
    from tests._failing_fixture_helpers import clone_analysis_to_tmp

    master_clone = clone_analysis_to_tmp(
        synthetic_sensitivity_completed.master_analysis, tmp_path
    )
    return master_clone.sensitivity


def test_isolated_fixture_does_not_perturb_session_tree(
    synthetic_multisim_completed, synthetic_multisim_completed_isolated
):
    """Lock the D1 copy-on-read contract: a destructive op on the isolated clone
    leaves the shared session tree's _status flags intact."""
    session_dir = synthetic_multisim_completed.analysis_paths.analysis_dir
    flag = session_dir / "_status" / "e_consolidate_complete.flag"
    before = hashlib.sha1(flag.read_bytes()).hexdigest() if flag.exists() else None
    a = synthetic_multisim_completed_isolated
    a.reprocess(
        start_with="consolidate",
        execution_mode="local",
        regenerate_existing=True,
        verbose=False,
    )
    after = hashlib.sha1(flag.read_bytes()).hexdigest() if flag.exists() else None
    assert before == after, "isolated reprocess leaked into the session tree"
    assert a.analysis_paths.analysis_dir != session_dir, "clone shares the session dir"


@pytest.fixture(scope="session")
def tritonswmm_cpu_compiled():
    """Pre-compile TRITON-SWMM CPU once per test session for each test-case
    family used by the coupled-mode tests.

    Required by coupled-mode tests whose `prepare_scenario` gate at
    scenario.py:800-811 checks `_system.compilation_cpu_successful`.
    The property reads the in-memory system log; a fresh system init
    shows compile-not-yet-done even when a valid build artifact exists
    on disk. Each test-case family has its own `_software_root`, so the
    fixture iterates over the 3 families touched by the marked tests
    (synth_all_models, synth_multi_sim, norfolk_multi_sim).

    Process-safety note: this fixture writes to ~/.cache/.../_software/
    or test_data/.../triton/ and assumes no concurrent test sessions are
    running an actual compile against the same cache dir.
    """
    from tests.utils_for_testing import compile_toolchain_unavailable

    if compile_toolchain_unavailable():
        pytest.skip(
            "TRITON-SWMM CPU compile toolchain (cmake + mpic++) not on PATH; "
            "run compile-dependent tests under the hhemt conda env "
            "(e.g. `conda run -n hhemt uv run --active --extra test pytest ...`).",
            allow_module_level=False,
        )

    from tests.fixtures.test_case_catalog import Local_TestCases

    for retrieve in (
        Local_TestCases.retrieve_synth_all_models_test_case,
        Local_TestCases.retrieve_synth_multi_sim_test_case,
        Local_TestCases.retrieve_norfolk_multi_sim_test_case,
        Local_TestCases.retrieve_norfolk_single_sim_test_case,
    ):
        case = retrieve(start_from_scratch=False)
        case.analysis._system.compile_TRITON_SWMM(
            backends=["cpu"],
            recompile_if_already_done_successfully=False,
        )
        if case.analysis._system.cfg_system.toggle_swmm_model:
            case.analysis._system.compile_SWMM(
                recompile_if_already_done_successfully=False,
            )


# ========== Phase 3a: Session-scope rendered_synth_* fixtures (R7) ==========


def _mtime_sha_snapshot(analysis_dir: Path) -> dict[str, tuple[float, str]]:
    """Return ``{relative_path: (mtime, sha1)}`` for ``_status/*.flag`` and
    every ``*.manifest.json`` under ``plots/``. Used by the session-scope
    rendered-fixture finalizers to enforce read-only consumption (R-CP-3:
    SHA-1 is the durable invariant; mtime-only drift is a warning)."""
    snapshot: dict[str, tuple[float, str]] = {}
    status_dir = analysis_dir / "_status"
    if status_dir.exists():
        for f in status_dir.glob("*.flag"):
            snapshot[str(f.relative_to(analysis_dir))] = (
                f.stat().st_mtime,
                hashlib.sha1(f.read_bytes()).hexdigest(),
            )
    plots_dir = analysis_dir / "plots"
    if plots_dir.exists():
        for f in plots_dir.rglob("*.manifest.json"):
            snapshot[str(f.relative_to(analysis_dir))] = (
                f.stat().st_mtime,
                hashlib.sha1(f.read_bytes()).hexdigest(),
            )
    return snapshot


def _assert_no_sha_drift(
    label: str,
    initial: dict[str, tuple[float, str]],
    final: dict[str, tuple[float, str]],
) -> None:
    """Compare two snapshots from :func:`_mtime_sha_snapshot`. Hard-fails on
    SHA-changed or deleted entries (R7 read-only invariant); mtime-only
    differences are downgraded to a pytest warning (R-CP-3 mitigation:
    read-write `open()` without writes still bumps mtime on some
    filesystems)."""
    drift: list[str] = []
    for k, (m0, sha0) in initial.items():
        if k not in final:
            drift.append(f"DELETED: {k}")
            continue
        m1, sha1 = final[k]
        if sha0 != sha1:
            drift.append(f"SHA-CHANGED: {k}")
        elif m0 != m1:
            drift.append(f"MTIME-ONLY-DRIFT: {k} (warning)")
    sha_drift = [d for d in drift if not d.startswith("MTIME-ONLY-DRIFT")]
    if sha_drift:
        pytest.fail(
            f"{label} session fixture was mutated:\n  " + "\n  ".join(sha_drift)
        )


@pytest.fixture(scope="session")
def rendered_synth_multi_sim():
    """Session-scope: build, run, and render the synth multisim analysis once.

    Promoted from function-scope per Phase 3a (R7,
    synth-test-isolation-and-runtime). Builds its own case from
    ``Local_TestCases`` (a session-scope fixture cannot depend on the
    function-scope ``synth_multi_sim_analysis``). Renders both ``html`` and
    ``zip`` formats so bundle round-trip tests find both artifacts. The
    session finalizer asserts no consumer mutated ``_status/*.flag`` or
    ``plots/**/*.manifest.json`` via SHA-1 (mtime-only drift tolerated)."""
    case = cases.Local_TestCases.retrieve_synth_multi_sim_test_case(
        start_from_scratch=True,
    )
    case.analysis.run()
    case.analysis.render_report(format="html")
    case.analysis.render_report(format="zip")
    snapshot = _mtime_sha_snapshot(case.analysis.analysis_paths.analysis_dir)
    yield case.analysis
    final = _mtime_sha_snapshot(case.analysis.analysis_paths.analysis_dir)
    _assert_no_sha_drift("rendered_synth_multi_sim", snapshot, final)


@pytest.fixture(scope="session")
def rendered_synth_sensitivity():
    """Session-scope: build, run, and render the synth sensitivity analysis
    once. See ``rendered_synth_multi_sim`` for contract details. Uses
    ``_SYNTH_SENSITIVITY_REPORT_CONFIG`` (relocated from
    ``test_synth_08_bundle_round_trip.py`` in Phase 3a)."""
    case = cases.Local_TestCases.retrieve_synth_cpu_config_sensitivity_case(
        start_from_scratch=True,
    )
    case.analysis.run(report_config=_SYNTH_SENSITIVITY_REPORT_CONFIG)
    case.analysis.render_report(format="html")
    case.analysis.render_report(format="zip")
    snapshot = _mtime_sha_snapshot(case.analysis.analysis_paths.analysis_dir)
    yield case.analysis
    final = _mtime_sha_snapshot(case.analysis.analysis_paths.analysis_dir)
    _assert_no_sha_drift("rendered_synth_sensitivity", snapshot, final)
