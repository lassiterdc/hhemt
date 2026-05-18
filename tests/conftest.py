import hashlib
import os
from pathlib import Path

import pytest

import tests.fixtures.test_case_catalog as cases
from TRITON_SWMM_toolkit.workflow import _NON_INTERACTIVE_LOCK_CLEAR_ENV

_SYNTH_SENSITIVITY_REPORT_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs" / "reports" / "synth_sensitivity_report_config.yaml"
)

# import tests.fixtures.test_case_catalog as cases


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
    case = cases.Local_TestCases.retrieve_norfolk_all_models_test_case(
        start_from_scratch=True
    )
    return case.analysis


@pytest.fixture
def norfolk_single_sim_analysis_cached():
    case = cases.Local_TestCases.retrieve_norfolk_all_models_test_case(
        start_from_scratch=False
    )
    return case.analysis


@pytest.fixture
def norfolk_multi_sim_analysis():
    case = cases.Local_TestCases.retrieve_norfolk_multi_sim_test_case(
        start_from_scratch=True
    )
    return case.analysis


@pytest.fixture
def norfolk_multi_sim_analysis_cached():
    case = cases.Local_TestCases.retrieve_norfolk_multi_sim_test_case(
        start_from_scratch=False
    )
    return case.analysis


@pytest.fixture
def norfolk_sensitivity_analysis():
    case = cases.Local_TestCases.retrieve_norfolk_cpu_config_sensitivity_case(
        start_from_scratch=True
    )
    return case.analysis


@pytest.fixture
def norfolk_sensitivity_analysis_cached():
    case = cases.Local_TestCases.retrieve_norfolk_cpu_config_sensitivity_case(
        start_from_scratch=False
    )
    return case.analysis


@pytest.fixture
def norfolk_sensitivity_triton_only():
    """TRITON-only sensitivity analysis varying n_omp_threads."""
    case = (
        cases.Local_TestCases.retrieve_norfolk_cpu_config_sensitivity_case_triton_only(
            start_from_scratch=True
        )
    )
    return case.analysis


@pytest.fixture
def norfolk_sensitivity_swmm_only():
    """SWMM-only sensitivity analysis varying n_omp_threads."""
    case = cases.Local_TestCases.retrieve_norfolk_cpu_config_sensitivity_case_swmm_only(
        start_from_scratch=True
    )
    return case.analysis


@pytest.fixture
def norfolk_frontier_multisim_analysis():
    case = cases.Frontier_TestCases.retrieve_norfolk_frontier_multisim_cpu_serial_case(
        start_from_scratch=True
    )
    return case.analysis


@pytest.fixture
def norfolk_frontier_multisim_analysis_cached():
    case = cases.Frontier_TestCases.retrieve_norfolk_frontier_multisim_cpu_serial_case(
        start_from_scratch=False
    )
    return case.analysis


@pytest.fixture
def norfolk_frontier_multisim_gpu_analysis():
    case = cases.Frontier_TestCases.retrieve_norfolk_frontier_multisim_gpu_case(
        start_from_scratch=True
    )
    return case.analysis


@pytest.fixture
def norfolk_frontier_sensitivity_analysis():
    case = cases.Frontier_TestCases.retrieve_norfolk_frontier_sensitivity_minimal(
        start_from_scratch=True
    )
    return case.analysis


@pytest.fixture
def norfolk_frontier_sensitivity_analysis_cached():
    case = cases.Frontier_TestCases.retrieve_norfolk_frontier_sensitivity_minimal(
        start_from_scratch=False
    )
    return case.analysis


@pytest.fixture
def norfolk_uva_multisim_analysis():
    case = cases.UVA_TestCases.retrieve_norfolk_UVA_multisim_1cpu_case(
        start_from_scratch=True
    )
    return case.analysis


@pytest.fixture
def norfolk_uva_multisim_analysis_cached():
    case = cases.UVA_TestCases.retrieve_norfolk_UVA_multisim_1cpu_case(
        start_from_scratch=False
    )
    return case.analysis


@pytest.fixture
def norfolk_uva_sensitivity_analysis():
    case = cases.UVA_TestCases.retrieve_norfolk_UVA_sensitivity_minimal(
        start_from_scratch=True
    )
    return case.analysis


@pytest.fixture
def norfolk_uva_sensitivity_analysis_cached():
    case = cases.UVA_TestCases.retrieve_norfolk_UVA_sensitivity_minimal(
        start_from_scratch=False
    )
    return case.analysis


@pytest.fixture
def norfolk_uva_sensitivity_full_ensemble_analysis():
    case = cases.UVA_TestCases.retrieve_norfolk_UVA_sensitivity_CPU_full_ensemble_short_sims(
        start_from_scratch=True
    )
    return case.analysis


# ========== Multi-Model Test Fixtures ==========


@pytest.fixture
def norfolk_triton_only_analysis():
    """TRITON-only analysis (no SWMM coupling)."""
    case = cases.Local_TestCases.retrieve_norfolk_triton_only_test_case(
        start_from_scratch=True
    )
    return case.analysis


@pytest.fixture
def norfolk_triton_only_analysis_cached():
    """TRITON-only analysis (cached - for faster iteration)."""
    case = cases.Local_TestCases.retrieve_norfolk_triton_only_test_case(
        start_from_scratch=False
    )
    return case.analysis


@pytest.fixture
def norfolk_swmm_only_analysis():
    """SWMM-only analysis (standalone EPA SWMM)."""
    case = cases.Local_TestCases.retrieve_norfolk_swmm_only_test_case(
        start_from_scratch=True
    )
    return case.analysis


@pytest.fixture
def norfolk_swmm_only_analysis_cached():
    """SWMM-only analysis (cached - for faster iteration)."""
    case = cases.Local_TestCases.retrieve_norfolk_swmm_only_test_case(
        start_from_scratch=False
    )
    return case.analysis


@pytest.fixture
def norfolk_all_models_analysis():
    """Analysis with all models enabled (TRITON, TRITON-SWMM, SWMM)."""
    case = cases.Local_TestCases.retrieve_norfolk_all_models_test_case(
        start_from_scratch=True
    )
    return case.analysis


@pytest.fixture
def norfolk_all_models_analysis_cached():
    """Analysis with all models (cached - for faster iteration)."""
    case = cases.Local_TestCases.retrieve_norfolk_all_models_test_case(
        start_from_scratch=False
    )
    return case.analysis


@pytest.fixture
def norfolk_triton_and_tritonswmm_analysis():
    """Analysis with TRITON and TRITON-SWMM (no standalone SWMM)."""
    case = cases.Local_TestCases.retrieve_norfolk_triton_and_tritonswmm_test_case(
        start_from_scratch=True
    )
    return case.analysis


@pytest.fixture
def norfolk_triton_and_tritonswmm_analysis_cached():
    """Analysis with TRITON and TRITON-SWMM (cached)."""
    case = cases.Local_TestCases.retrieve_norfolk_triton_and_tritonswmm_test_case(
        start_from_scratch=False
    )
    return case.analysis


# ========== Phase 6b.2.1: Unified Fixture API (Pilot - Local Only) ==========
# Proof-of-concept for parametrized fixtures before full platform expansion.
# Keeps all existing fixtures untouched during validation.


@pytest.fixture(params=[pytest.param("local", id="local")])
def platform_pilot(request):
    """Platform selection for parametrized fixtures (pilot: local only).

    This is a pilot implementation to validate the parametrization pattern
    before expanding to UVA and Frontier platforms.

    Future expansion (Phase 6b.2.2):
        @pytest.fixture(params=[
            pytest.param("local", id="local"),
            pytest.param("uva", marks=pytest.mark.skipif(...), id="uva"),
            pytest.param("frontier", marks=pytest.mark.skipif(...), id="frontier"),
        ])
    """
    return request.param


@pytest.fixture
def norfolk_multi_sim_unified(platform_pilot):
    """Multi-simulation analysis (unified API, pilot: local only).

    This fixture demonstrates the unified API pattern that will replace
    platform-specific fixtures once validated.

    Replaces (in future):
        - norfolk_multi_sim_analysis (local)
        - norfolk_uva_multisim_analysis (UVA)
        - norfolk_frontier_multisim_analysis (Frontier)

    Usage:
        def test_workflow(norfolk_multi_sim_unified):
            analysis = norfolk_multi_sim_unified
            # Test logic runs once per platform param
    """
    # Currently only supports local platform (pilot phase)
    if platform_pilot == "local":
        case = cases.Local_TestCases.retrieve_norfolk_multi_sim_test_case(
            start_from_scratch=True
        )
    else:
        pytest.fail(f"Unsupported platform in pilot: {platform_pilot}")

    return case.analysis


@pytest.fixture
def norfolk_multi_sim_unified_cached(platform_pilot):
    """Multi-simulation analysis (unified API, cached, pilot: local only).

    Cached variant of norfolk_multi_sim_unified for tests that don't need
    fresh setup each time.
    """
    if platform_pilot == "local":
        case = cases.Local_TestCases.retrieve_norfolk_multi_sim_test_case(
            start_from_scratch=False
        )
    else:
        pytest.fail(f"Unsupported platform in pilot: {platform_pilot}")

    return case.analysis


# ========== Synthetic Test Fixtures ==========


@pytest.fixture
def synth_all_models_analysis():
    case = cases.Local_TestCases.retrieve_synth_all_models_test_case(
        start_from_scratch=True
    )
    return case.analysis


@pytest.fixture
def synth_all_models_analysis_cached():
    case = cases.Local_TestCases.retrieve_synth_all_models_test_case(
        start_from_scratch=False
    )
    return case.analysis


@pytest.fixture
def synth_multi_sim_analysis():
    case = cases.Local_TestCases.retrieve_synth_multi_sim_test_case(
        start_from_scratch=True
    )
    return case.analysis


@pytest.fixture
def synth_multi_sim_analysis_cached():
    case = cases.Local_TestCases.retrieve_synth_multi_sim_test_case(
        start_from_scratch=False
    )
    return case.analysis


@pytest.fixture(scope="session")
def synth_multi_sim_builder():
    """Configured-but-not-run multi-sim analysis (Phase 2,
    synth-test-isolation-and-runtime). No DEM/landuse preprocessing, no compile,
    no scenario prep — sufficient for `generate_snakefile_content`-only tests.
    Session-scoped so the 4 collapsed symmetry tests share one builder."""
    case = cases.Local_TestCases.retrieve_synth_multi_sim_test_case(
        start_from_scratch=True,
        skip_run=True,
    )
    return case.analysis


@pytest.fixture(scope="session")
def synth_sensitivity_builder():
    """Configured-but-not-run sensitivity analysis (Phase 2,
    synth-test-isolation-and-runtime). Same contract as `synth_multi_sim_builder`
    but consumes the sensitivity-CSV factory."""
    case = cases.Local_TestCases.retrieve_synth_cpu_config_sensitivity_case(
        start_from_scratch=True,
        skip_run=True,
    )
    return case.analysis


@pytest.fixture
def synth_triton_only_analysis():
    case = cases.Local_TestCases.retrieve_synth_triton_only_test_case(
        start_from_scratch=True
    )
    return case.analysis


@pytest.fixture
def synth_triton_only_analysis_cached():
    case = cases.Local_TestCases.retrieve_synth_triton_only_test_case(
        start_from_scratch=False
    )
    return case.analysis


@pytest.fixture
def synth_swmm_only_analysis():
    case = cases.Local_TestCases.retrieve_synth_swmm_only_test_case(
        start_from_scratch=True
    )
    return case.analysis


@pytest.fixture
def synth_swmm_only_analysis_cached():
    case = cases.Local_TestCases.retrieve_synth_swmm_only_test_case(
        start_from_scratch=False
    )
    return case.analysis


@pytest.fixture
def synth_triton_and_tritonswmm_analysis():
    case = cases.Local_TestCases.retrieve_synth_triton_and_tritonswmm_test_case(
        start_from_scratch=True
    )
    return case.analysis


@pytest.fixture
def synth_triton_and_tritonswmm_analysis_cached():
    case = cases.Local_TestCases.retrieve_synth_triton_and_tritonswmm_test_case(
        start_from_scratch=False
    )
    return case.analysis


@pytest.fixture
def synth_sensitivity_analysis():
    case = cases.Local_TestCases.retrieve_synth_cpu_config_sensitivity_case(
        start_from_scratch=True
    )
    return case.analysis


@pytest.fixture
def synth_sensitivity_analysis_cached():
    case = cases.Local_TestCases.retrieve_synth_cpu_config_sensitivity_case(
        start_from_scratch=False
    )
    return case.analysis


@pytest.fixture
def synth_sensitivity_triton_only():
    case = cases.Local_TestCases.retrieve_synth_cpu_config_sensitivity_case_triton_only(
        start_from_scratch=True
    )
    return case.analysis


@pytest.fixture
def synth_sensitivity_swmm_only():
    case = cases.Local_TestCases.retrieve_synth_cpu_config_sensitivity_case_swmm_only(
        start_from_scratch=True
    )
    return case.analysis


@pytest.fixture
def synth_sensitivity_with_system_overlay():
    case = cases.Local_TestCases.retrieve_synth_cpu_config_sensitivity_case_with_system_overlay(
        start_from_scratch=True
    )
    return case.analysis


@pytest.fixture
def synth_sensitivity_mutex_violation():
    case = cases.Local_TestCases.retrieve_synth_cpu_config_sensitivity_case_mutex_violation(
        start_from_scratch=True
    )
    return case.analysis


@pytest.fixture
def synth_sensitivity_invalid_overlay():
    case = cases.Local_TestCases.retrieve_synth_cpu_config_sensitivity_case_invalid_overlay(
        start_from_scratch=True
    )
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
def synth_sensitivity_with_system_gpu_hardware_override():
    case = cases.Local_TestCases.retrieve_synth_cpu_config_sensitivity_case_with_system_gpu_hardware_override(
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
