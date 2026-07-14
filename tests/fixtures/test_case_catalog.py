"""
Test case catalog for TRITON-SWMM toolkit.

Provides a collection of pre-configured test cases for different scenarios:
- Local single/multi simulation tests
- Platform-specific HPC tests (UVA, Frontier)
- Sensitivity analysis tests with various configurations

Each method returns a retrieve_TRITON_SWMM_test_case instance with:
- Synthetic weather data
- Platform-appropriate HPC configurations
- Short simulation durations for fast testing

"""

import os
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import platformdirs
import pytest

import hhemt.constants as cnst
from hhemt.experiments import NorfolkIreneExperiment
from tests.fixtures import worktree_slug

# Import from test fixtures
from tests.fixtures.test_case_builder import (
    retrieve_synth_TRITON_SWMM_test_case,
    retrieve_TRITON_SWMM_test_case,
)


def _require_cpu_cores_for_sensitivity(min_cores: int = 4) -> None:
    """Skip the current test if the host has fewer than ``min_cores`` CPU cores.

    The synth sensitivity CSV includes a hybrid row (2 MPI x 2 OMP = 4 threads);
    on machines with fewer cores the honored-thread validator would flag a
    mismatch only after a full workflow run. Failing fast here costs seconds,
    not minutes.
    """
    n = os.cpu_count() or 1
    if n < min_cores:
        pytest.skip(f"synth sensitivity suite requires >={min_cores} CPU cores; found {n}")


def _load_norfolk_example_or_skip(**kwargs) -> "NorfolkIreneExperiment":
    """Acquire the Norfolk Irene example, or skip when the data is unavailable.

    Every Norfolk-example consumer in the suite (the conftest ``norfolk_*``
    fixtures, the ``retrieve_norfolk_*`` case methods, and the ``ex_Nrflk``
    sensitivity path) funnels through this helper, so gating here converts an
    absent-data ERROR into a SKIP for all of them at a single point.

    Mirrors the canonical gate in ``tests/test_case_study_catalog.py`` (which
    wraps the identical ``NorfolkIreneExperiment.load``): under
    ``HHEMT_REQUIRE_EXAMPLE_DATA=1`` (CI runners that cache the example data) a
    load failure is re-raised as a hard error instead of a silent skip, so a
    data-required run cannot pass vacuously. Bare ``pytest`` (test.yml) does not
    set the flag, so a runner without the Norfolk data SKIPs rather than ERRORs.
    """
    try:
        return NorfolkIreneExperiment.load(**kwargs)
    except Exception as exc:
        if os.environ.get("HHEMT_REQUIRE_EXAMPLE_DATA") == "1":
            raise AssertionError(
                f"Norfolk Irene example data required (HHEMT_REQUIRE_EXAMPLE_DATA=1) but load failed: {exc!r}"
            ) from exc
        pytest.skip(f"Norfolk Irene example data not available locally: {exc!r}")


@dataclass
class all_experiments:
    from hhemt.experiments import TRITON_SWMM_experiment

    @staticmethod
    def ex_Nrflk(download_if_exists: bool = False) -> TRITON_SWMM_experiment:
        return _load_norfolk_example_or_skip(download_if_exists=download_if_exists)


class GetTS_TestCases:
    """
    Test case catalog for TRITON-SWMM toolkit.

    Provides factory methods to create test cases with:
    - Synthetic weather data
    - Short simulation durations
    - Platform-specific HPC configurations (via analysis/system overlay dicts)
    - Isolated test directories

    Platform-specific methods apply centralized analysis/system overlay dicts
    (defined in case_study_catalog.py) to eliminate configuration duplication.

    Caching Strategy:
        Use start_from_scratch=False to reuse processed inputs from previous runs,
        significantly speeding up test execution. Set to True when you need a clean slate.
    """

    def __init__(
        self,
    ) -> None:
        pass

    @classmethod
    def _retrieve_norfolk_case(
        cls,
        n_events: int,
        analysis_name: str,
        start_from_scratch: bool,
        download_if_exists=False,
        analysis_overlay: dict | None = None,
        system_overlay: dict | None = None,
        example_data_dir: Path | None = None,
        analysis_overrides: dict | None = None,
        system_overrides: dict | None = None,
        n_reporting_tsteps_per_sim=cnst.TEST_N_REPORTING_TSTEPS_PER_SIM,
        TRITON_reporting_timestep_s=cnst.TEST_TRITON_REPORTING_TIMESTEP_S,
        test_system_dirname=cnst.TEST_SYSTEM_DIRNAME,
        hpc_system_config_yaml: Path | None = None,
    ) -> retrieve_TRITON_SWMM_test_case:
        """
        Internal helper to create Norfolk test cases.

        Applies a base analysis/system overlay (analysis_overlay / system_overlay),
        then per-call overrides (analysis_overrides / system_overrides) take precedence.

        Args:
            analysis_name: Name for the test analysis
            n_events: Number of weather events
            n_reporting_tsteps_per_sim: Timesteps per simulation
            TRITON_reporting_timestep_s: Reporting interval in seconds
            start_from_scratch: Whether to reprocess inputs
            download_if_exists: Whether to re-download HydroShare data
            analysis_overlay: Base analysis-config overlay dict
            system_overlay: Base system-config overlay dict
            analysis_overrides: Per-call analysis-config overrides (win over the overlay)
            system_overrides: Per-call system-config overrides (win over the overlay)
            example_data_dir: Override for data directory location
            hpc_system_config_yaml: Optional path to an hpc_system_config YAML

        Returns:
            retrieve_TRITON_SWMM_test_case instance with configured system
        """

        # Example-platform overlay is the base; per-call overrides win (same
        # precedence as the retired PlatformConfig.to_*_dict() | overrides).
        # example_data_dir is now an explicit param (was a UVA-preset field).
        final_analysis_configs = (analysis_overlay or {}) | (analysis_overrides or {})
        final_system_configs = (system_overlay or {}) | (system_overrides or {})

        example = _load_norfolk_example_or_skip(
            download_if_exists=download_if_exists, example_data_dir=example_data_dir
        )

        nrflk_test = retrieve_TRITON_SWMM_test_case(
            example=example,
            analysis_name=analysis_name,
            n_events=n_events,
            n_reporting_tsteps_per_sim=n_reporting_tsteps_per_sim,
            TRITON_reporting_timestep_s=TRITON_reporting_timestep_s,
            test_system_dirname=test_system_dirname,
            start_from_scratch=start_from_scratch,
            additional_analysis_configs=final_analysis_configs,
            additional_system_configs=final_system_configs,
            hpc_system_config_yaml=hpc_system_config_yaml,
        )
        return nrflk_test


class Local_TestCases:
    cpu_sensitivity = "cpu_benchmarking_analysis.xlsx"
    cpu_sensitivity_swmm = "cpu_benchmarking_analysis_swmm.xlsx"

    @classmethod
    def retrieve_norfolk_cpu_config_sensitivity_case(
        cls,
        start_from_scratch: bool = False,
        download_if_exists: bool = False,
        hpc_system_config_yaml: Path | None = None,
    ) -> retrieve_TRITON_SWMM_test_case:
        """Local CPU configuration sensitivity analysis test."""
        analysis_name = "cpu_config_sensitivity"
        sensitivity = all_experiments.ex_Nrflk().test_case_directory / cls.cpu_sensitivity
        analysis_overrides = {
            "toggle_sensitivity_analysis": True,
            "sensitivity_analysis": sensitivity,
            # Inject the benchmarking sensitivity report config (the same
            # report.sensitivity shape the synth tier uses) so
            # cfg_analysis.report.sensitivity is populated. Without it the
            # report falls back to {} -> sensitivity None -> ConfigurationError
            # when PC_05/PC_06 render the sensitivity benchmarking figure.
            "report": cls._load_synth_sensitivity_report_dict(),
        }

        return GetTS_TestCases._retrieve_norfolk_case(
            analysis_name=analysis_name,
            start_from_scratch=start_from_scratch,
            download_if_exists=download_if_exists,
            n_events=1,
            analysis_overrides=analysis_overrides,
            hpc_system_config_yaml=hpc_system_config_yaml,
        )

    @classmethod
    def retrieve_norfolk_cpu_config_sensitivity_case_triton_only(
        cls, start_from_scratch: bool = False, download_if_exists: bool = False
    ) -> retrieve_TRITON_SWMM_test_case:
        """Local CPU configuration sensitivity analysis test."""
        analysis_name = "cpu_config_sensitivity_triton_only"
        sensitivity = all_experiments.ex_Nrflk().test_case_directory / cls.cpu_sensitivity
        analysis_overrides = {
            "toggle_sensitivity_analysis": True,
            "sensitivity_analysis": sensitivity,
            # Inject the benchmarking sensitivity report config (the same
            # report.sensitivity shape the synth tier uses) so
            # cfg_analysis.report.sensitivity is populated. Without it the
            # report falls back to {} -> sensitivity None -> ConfigurationError
            # when PC_05/PC_06 render the sensitivity benchmarking figure.
            "report": cls._load_synth_sensitivity_report_dict(),
        }
        system_overrides = {
            "toggle_triton_model": True,
            "toggle_tritonswmm_model": False,
            "toggle_swmm_model": False,
        }

        return GetTS_TestCases._retrieve_norfolk_case(
            analysis_name=analysis_name,
            start_from_scratch=start_from_scratch,
            download_if_exists=download_if_exists,
            n_events=1,
            analysis_overrides=analysis_overrides,
            system_overrides=system_overrides,
        )

    @classmethod
    def retrieve_norfolk_cpu_config_sensitivity_case_swmm_only(
        cls, start_from_scratch: bool = False, download_if_exists: bool = False
    ) -> retrieve_TRITON_SWMM_test_case:
        """Local CPU configuration sensitivity analysis test."""
        analysis_name = "cpu_config_sensitivity_swmm_only"
        sensitivity = all_experiments.ex_Nrflk().test_case_directory / cls.cpu_sensitivity_swmm
        analysis_overrides = {
            "toggle_sensitivity_analysis": True,
            "sensitivity_analysis": sensitivity,
            # Inject the benchmarking sensitivity report config (the same
            # report.sensitivity shape the synth tier uses) so
            # cfg_analysis.report.sensitivity is populated. Without it the
            # report falls back to {} -> sensitivity None -> ConfigurationError
            # when PC_05/PC_06 render the sensitivity benchmarking figure.
            "report": cls._load_synth_sensitivity_report_dict(),
        }
        system_overrides = {
            "toggle_triton_model": False,
            "toggle_tritonswmm_model": False,
            "toggle_swmm_model": True,
        }

        return GetTS_TestCases._retrieve_norfolk_case(
            analysis_name=analysis_name,
            start_from_scratch=start_from_scratch,
            download_if_exists=download_if_exists,
            n_events=1,
            analysis_overrides=analysis_overrides,
            system_overrides=system_overrides,
        )

    @classmethod
    def retrieve_norfolk_single_sim_test_case(
        cls, start_from_scratch: bool = False, download_if_exists: bool = False
    ) -> retrieve_TRITON_SWMM_test_case:
        """Local single simulation test - fastest test case."""
        analysis_name = "single_sim"
        return GetTS_TestCases._retrieve_norfolk_case(
            analysis_name=analysis_name,
            start_from_scratch=start_from_scratch,
            download_if_exists=download_if_exists,
            n_events=1,
        )

    @classmethod
    def retrieve_norfolk_multi_sim_test_case(
        cls,
        start_from_scratch: bool = False,
        download_if_exists: bool = False,
        hpc_system_config_yaml: Path | None = None,
    ) -> retrieve_TRITON_SWMM_test_case:
        """Local multi-simulation test with 2 events."""
        analysis_name = "multi_sim"
        system_overrides = {
            "toggle_triton_model": True,
            "toggle_tritonswmm_model": True,
            "toggle_swmm_model": True,
        }

        return GetTS_TestCases._retrieve_norfolk_case(
            analysis_name=analysis_name,
            start_from_scratch=start_from_scratch,
            download_if_exists=download_if_exists,
            n_events=2,
            system_overrides=system_overrides,
            hpc_system_config_yaml=hpc_system_config_yaml,
        )

    # ========== Multi-Model Test Cases ==========

    @classmethod
    def retrieve_norfolk_triton_only_test_case(
        cls, start_from_scratch: bool = False, download_if_exists: bool = False
    ) -> retrieve_TRITON_SWMM_test_case:
        """Local TRITON-only test (no SWMM coupling)."""
        analysis_name = "triton_only"
        system_overrides = {
            "toggle_triton_model": True,
            "toggle_tritonswmm_model": False,
            "toggle_swmm_model": False,
        }
        return GetTS_TestCases._retrieve_norfolk_case(
            analysis_name=analysis_name,
            start_from_scratch=start_from_scratch,
            download_if_exists=download_if_exists,
            n_events=1,
            system_overrides=system_overrides,
        )

    @classmethod
    def retrieve_norfolk_swmm_only_test_case(
        cls, start_from_scratch: bool = False, download_if_exists: bool = False
    ) -> retrieve_TRITON_SWMM_test_case:
        """Local SWMM-only test (EPA SWMM without TRITON coupling)."""
        analysis_name = "swmm_only"
        system_overrides = {
            "toggle_triton_model": False,
            "toggle_tritonswmm_model": False,
            "toggle_swmm_model": True,
        }
        return GetTS_TestCases._retrieve_norfolk_case(
            analysis_name=analysis_name,
            start_from_scratch=start_from_scratch,
            download_if_exists=download_if_exists,
            n_events=1,
            system_overrides=system_overrides,
        )

    @classmethod
    def retrieve_norfolk_all_models_test_case(
        cls, start_from_scratch: bool = False, download_if_exists: bool = False
    ) -> retrieve_TRITON_SWMM_test_case:
        """Local test with all models enabled (TRITON, TRITON-SWMM, SWMM)."""
        analysis_name = "all_models"
        system_overrides = {
            "toggle_triton_model": True,
            "toggle_tritonswmm_model": True,
            "toggle_swmm_model": True,
        }
        return GetTS_TestCases._retrieve_norfolk_case(
            analysis_name=analysis_name,
            start_from_scratch=start_from_scratch,
            download_if_exists=download_if_exists,
            n_events=1,
            system_overrides=system_overrides,
        )

    @classmethod
    def retrieve_norfolk_triton_and_tritonswmm_test_case(
        cls, start_from_scratch: bool = False, download_if_exists: bool = False
    ) -> retrieve_TRITON_SWMM_test_case:
        """Local test with TRITON and TRITON-SWMM (no standalone SWMM)."""
        analysis_name = "triton_and_tritonswmm"
        system_overrides = {
            "toggle_triton_model": True,
            "toggle_tritonswmm_model": True,
            "toggle_swmm_model": False,
        }
        return GetTS_TestCases._retrieve_norfolk_case(
            analysis_name=analysis_name,
            start_from_scratch=start_from_scratch,
            download_if_exists=download_if_exists,
            n_events=1,
            system_overrides=system_overrides,
        )

    # ========== Synthetic Test Cases ==========

    @staticmethod
    def retrieve_synth_all_models_test_case(start_from_scratch: bool = False):
        return retrieve_synth_TRITON_SWMM_test_case(
            analysis_name="synth_all_models",
            n_events=1,
            toggle_tritonswmm_model=True,
            toggle_triton_model=True,
            toggle_swmm_model=True,
            start_from_scratch=start_from_scratch,
        )

    @staticmethod
    def retrieve_synth_multi_sim_test_case(
        start_from_scratch: bool = False,
        skip_run: bool = False,
    ):
        return retrieve_synth_TRITON_SWMM_test_case(
            analysis_name="synth_multi_sim",
            # Iter-2 of `per_sim_peak_flood_depth` (2026-04-28): bumped from 2
            # to 3 so the per-event peak-flood-depth maps cover the three
            # forcing-mechanism scenarios encoded in weather.py
            # (event 0: hydro-only, event 1: BC-only, event 2: both).
            n_events=3,
            start_from_scratch=start_from_scratch,
            skip_run=skip_run,
        )

    @staticmethod
    def retrieve_synth_triton_only_test_case(start_from_scratch: bool = False):
        return retrieve_synth_TRITON_SWMM_test_case(
            analysis_name="synth_triton_only",
            toggle_tritonswmm_model=False,
            toggle_triton_model=True,
            toggle_swmm_model=False,
            start_from_scratch=start_from_scratch,
        )

    @staticmethod
    def retrieve_synth_swmm_only_test_case(start_from_scratch: bool = False):
        return retrieve_synth_TRITON_SWMM_test_case(
            analysis_name="synth_swmm_only",
            toggle_tritonswmm_model=False,
            toggle_triton_model=False,
            toggle_swmm_model=True,
            start_from_scratch=start_from_scratch,
        )

    @staticmethod
    def retrieve_synth_triton_and_tritonswmm_test_case(
        start_from_scratch: bool = False,
    ):
        return retrieve_synth_TRITON_SWMM_test_case(
            analysis_name="synth_triton_and_tritonswmm",
            toggle_tritonswmm_model=True,
            toggle_triton_model=True,
            toggle_swmm_model=False,
            start_from_scratch=start_from_scratch,
        )

    @staticmethod
    def _load_synth_sensitivity_report_dict() -> dict:
        """Load tests/fixtures/synthetic_model/report_config_synth_sensitivity.yaml
        as a dict so it can be injected inline into cfg_analysis.report (post-F2,
        R1 — required field). Pre-F2, this file was passed via the legacy
        `report_config_path` kwarg that Phase 3 deleted."""
        import yaml

        peer_path = Path(__file__).parent / "synthetic_model" / "report_config_synth_sensitivity.yaml"
        return yaml.safe_load(peer_path.read_text())

    @staticmethod
    def retrieve_synth_cpu_config_sensitivity_case(
        start_from_scratch: bool = False,
        skip_run: bool = False,
    ):
        _require_cpu_cores_for_sensitivity()
        csv_path = Local_TestCases._write_synth_sensitivity_csv(
            analysis_name="synth_sensitivity",
            model_subset="all",
        )
        return retrieve_synth_TRITON_SWMM_test_case(
            analysis_name="synth_sensitivity",
            toggle_tritonswmm_model=True,
            toggle_triton_model=False,
            toggle_swmm_model=False,
            sensitivity_csv=csv_path,
            start_from_scratch=start_from_scratch,
            skip_run=skip_run,
        )

    @staticmethod
    def retrieve_synth_cpu_config_sensitivity_case_triton_only(
        start_from_scratch: bool = False,
    ):
        _require_cpu_cores_for_sensitivity()
        csv_path = Local_TestCases._write_synth_sensitivity_csv(
            analysis_name="synth_sensitivity_triton_only",
            model_subset="triton",
        )
        return retrieve_synth_TRITON_SWMM_test_case(
            analysis_name="synth_sensitivity_triton_only",
            toggle_tritonswmm_model=False,
            toggle_swmm_model=False,
            sensitivity_csv=csv_path,
            start_from_scratch=start_from_scratch,
        )

    @staticmethod
    def retrieve_synth_cpu_config_sensitivity_case_swmm_only(
        start_from_scratch: bool = False,
    ):
        csv_path = Local_TestCases._write_synth_sensitivity_csv(
            analysis_name="synth_sensitivity_swmm_only",
            model_subset="swmm",
        )
        return retrieve_synth_TRITON_SWMM_test_case(
            analysis_name="synth_sensitivity_swmm_only",
            toggle_tritonswmm_model=False,
            toggle_triton_model=False,
            sensitivity_csv=csv_path,
            start_from_scratch=start_from_scratch,
        )

    @staticmethod
    def retrieve_synth_cpu_config_sensitivity_case_with_system_overlay(
        start_from_scratch: bool = False,
    ):
        """Phase 1 R1/R5/R6 — `system.target_dem_resolution` overlay → two UniqueSystemTargets."""
        _require_cpu_cores_for_sensitivity()
        csv_path = Local_TestCases._write_synth_sensitivity_csv(
            analysis_name="synth_sensitivity_with_system_overlay",
            model_subset="all",
            extra_columns={
                "system.target_dem_resolution": [1.0, 1.0, 2.0, 2.0],
            },
        )
        return retrieve_synth_TRITON_SWMM_test_case(
            analysis_name="synth_sensitivity_with_system_overlay",
            toggle_tritonswmm_model=True,
            toggle_triton_model=False,
            toggle_swmm_model=False,
            sensitivity_csv=csv_path,
            start_from_scratch=start_from_scratch,
        )

    @staticmethod
    def retrieve_synth_cpu_config_sensitivity_case_mutex_violation(
        start_from_scratch: bool = False,
    ):
        """Phase 1 R3 — row with both system_config_yaml AND system.* → ConfigurationError."""
        _require_cpu_cores_for_sensitivity()
        dest_dir = (
            Path(platformdirs.user_cache_dir("hhemt"))
            / "synthetic_test_runs"
            / worktree_slug()
            / "_sensitivity_configs"
        )
        dest_dir.mkdir(parents=True, exist_ok=True)
        per_sa_yaml = dest_dir / "synth_mutex_violation_row0_system.yaml"
        per_sa_yaml.write_text("# placeholder per-sa system YAML for mutex-violation test\n")
        csv_path = Local_TestCases._write_synth_sensitivity_csv(
            analysis_name="synth_sensitivity_mutex_violation",
            model_subset="all",
            extra_columns={
                "system_config_yaml": [str(per_sa_yaml), "", "", ""],
                "system.target_dem_resolution": [1.0, None, None, None],
            },
        )
        return retrieve_synth_TRITON_SWMM_test_case(
            analysis_name="synth_sensitivity_mutex_violation",
            toggle_tritonswmm_model=True,
            toggle_triton_model=False,
            toggle_swmm_model=False,
            sensitivity_csv=csv_path,
            start_from_scratch=start_from_scratch,
        )

    @staticmethod
    def retrieve_synth_cpu_config_sensitivity_case_invalid_overlay(
        start_from_scratch: bool = False,
    ):
        """Phase 1 R4 — `system.target_dem_resolution='WRONG'` → Pydantic float-coercion failure.

        Retargeted from the retired `system.gpu_compilation_backend` overlay (Phase-4
        moved GPU backend off system_config) to the still-existing top-level typed
        `target_dem_resolution: float` field. `"WRONG"` fails float coercion, re-firing
        `system_config.model_validate` → the `"SystemConfig validation"` ConfigurationError
        wrapper (sensitivity_analysis.py), which the test asserts via
        `match="SystemConfig validation"`.
        """
        _require_cpu_cores_for_sensitivity()
        csv_path = Local_TestCases._write_synth_sensitivity_csv(
            analysis_name="synth_sensitivity_invalid_overlay",
            model_subset="all",
            extra_columns={
                "system.target_dem_resolution": ["WRONG", None, None, None],
            },
        )
        return retrieve_synth_TRITON_SWMM_test_case(
            analysis_name="synth_sensitivity_invalid_overlay",
            toggle_tritonswmm_model=True,
            toggle_triton_model=False,
            toggle_swmm_model=False,
            sensitivity_csv=csv_path,
            start_from_scratch=start_from_scratch,
        )

    @staticmethod
    def retrieve_synth_cpu_config_sensitivity_case_legacy_gpu_hardware_override(
        start_from_scratch: bool = False,
    ):
        """Phase 1 R8/R-X-2 — legacy `gpu_hardware_override` column → migration ConfigurationError."""
        _require_cpu_cores_for_sensitivity()
        csv_path = Local_TestCases._write_synth_sensitivity_csv(
            analysis_name="synth_sensitivity_legacy_gpu_hw_override",
            model_subset="all",
            extra_columns={
                "gpu_hardware_override": ["a6000", "a6000", "a6000", "a6000"],
            },
        )
        return retrieve_synth_TRITON_SWMM_test_case(
            analysis_name="synth_sensitivity_legacy_gpu_hw_override",
            toggle_tritonswmm_model=True,
            toggle_triton_model=False,
            toggle_swmm_model=False,
            sensitivity_csv=csv_path,
            start_from_scratch=start_from_scratch,
        )

    @staticmethod
    def retrieve_synth_cpu_config_sensitivity_case_typo_in_prefixed_column(
        start_from_scratch: bool = False,
    ):
        """Phase 1 R9 — unknown column `system.target_dem_reslution` (typo) → ConfigurationError."""
        _require_cpu_cores_for_sensitivity()
        csv_path = Local_TestCases._write_synth_sensitivity_csv(
            analysis_name="synth_sensitivity_typo_in_prefixed_column",
            model_subset="all",
            extra_columns={
                "system.target_dem_reslution": [1.0, 1.0, 2.0, 2.0],  # intentional typo
            },
        )
        return retrieve_synth_TRITON_SWMM_test_case(
            analysis_name="synth_sensitivity_typo_in_prefixed_column",
            toggle_tritonswmm_model=True,
            toggle_triton_model=False,
            toggle_swmm_model=False,
            sensitivity_csv=csv_path,
            start_from_scratch=start_from_scratch,
        )

    @staticmethod
    def retrieve_synth_cpu_config_sensitivity_case_with_partition_axis(
        start_from_scratch: bool = False,
    ):
        """Phase-5 partition-axis re-enablement — `analysis.hpc_ensemble_partition`
        overlay resolves gpu_hardware per-partition (replaces the retired
        `system.gpu_hardware` overlay; T13 equivalence). The synthetic
        `hpc_system_config_test.yaml` declares `test_partition` -> gpu_hardware
        `a6000`; the MASTER `hpc_ensemble_partition` is set to `test_partition`
        because the as-built single-master resolution reads the master selector
        for compile-dedup + GRES (per-row generalization is deferred to Phase 6)."""
        _require_cpu_cores_for_sensitivity()
        csv_path = Local_TestCases._write_synth_sensitivity_csv(
            analysis_name="synth_sensitivity_with_partition_axis",
            model_subset="all",
            extra_columns={
                "analysis.hpc_ensemble_partition": [
                    "test_partition",
                    "test_partition",
                    "test_partition",
                    "test_partition",
                ],
            },
        )
        return retrieve_synth_TRITON_SWMM_test_case(
            analysis_name="synth_sensitivity_with_partition_axis",
            toggle_tritonswmm_model=True,
            toggle_triton_model=False,
            toggle_swmm_model=False,
            sensitivity_csv=csv_path,
            hpc_system_config_yaml=(Path(__file__).parent / "hpc_system_config_test.yaml"),
            start_from_scratch=start_from_scratch,
            additional_analysis_configs={
                "hpc_ensemble_partition": "test_partition",
            },
        )

    @staticmethod
    def retrieve_synth_cpu_config_sensitivity_case_multi_partition_fanout(
        start_from_scratch: bool = False,
    ):
        """Phase-6 cross-hardware fan-out — the `hpc.partition` overlay varies the
        ensemble partition across rows (gpu-a6000 + gpu-a100), so the compile-dedup
        produces TWO distinct UniqueSystemTarget builds with distinct
        partition-derived gpu_hardware. (Tests force `n_gpus>0` per sub
        post-construction so the GPU directive renders — setting it in the CSV
        would trip the analysis-config MPI-only-mode validator.)
        `hpc_system_config_multipartition.yaml` declares both partitions; the
        MASTER `hpc_ensemble_partition` is gpu-a6000."""
        _require_cpu_cores_for_sensitivity()
        csv_path = Local_TestCases._write_synth_sensitivity_csv(
            analysis_name="synth_sensitivity_multi_partition_fanout",
            model_subset="all",
            extra_columns={
                "hpc.partition": [
                    "gpu-a6000",
                    "gpu-a100",
                    "gpu-a6000",
                    "gpu-a100",
                ],
            },
        )
        return retrieve_synth_TRITON_SWMM_test_case(
            analysis_name="synth_sensitivity_multi_partition_fanout",
            toggle_tritonswmm_model=True,
            toggle_triton_model=False,
            toggle_swmm_model=False,
            sensitivity_csv=csv_path,
            hpc_system_config_yaml=(Path(__file__).parent / "hpc_system_config_multipartition.yaml"),
            start_from_scratch=start_from_scratch,
            additional_analysis_configs={
                "hpc_ensemble_partition": "gpu-a6000",
            },
        )

    @staticmethod
    def retrieve_synth_cpu_config_sensitivity_case_mixed_cpu_gpu_fanout(
        start_from_scratch: bool = False,
    ):
        """synth_cc friction regression (2026-07-08): a mixed CPU/GPU per-row-partition
        fan-out — the `hpc.partition` overlay varies across rows between a CPU partition
        (`standard`, resolves to (None, None)) and a GPU partition (`gpu-a6000`). The
        dedup produces TWO distinct UniqueSystemTargets: a CPU/standard target (backend
        None) and a gpu-a6000 target (a6000/CUDA). The MASTER `hpc_ensemble_partition`
        is `gpu-a6000`, so the pre-fix sensitivity constructor overwrote the standard
        target's system backend with the master a6000/CUDA in the setup_target runner
        (is_main_orchestrator=False), crashing the CPU compile in the GPU branch.
        `hpc_system_config_multipartition.yaml` declares `standard` + `gpu-a6000`."""
        _require_cpu_cores_for_sensitivity()
        csv_path = Local_TestCases._write_synth_sensitivity_csv(
            analysis_name="synth_sensitivity_mixed_cpu_gpu_fanout",
            model_subset="all",
            extra_columns={
                "hpc.partition": [
                    "standard",
                    "gpu-a6000",
                    "standard",
                    "gpu-a6000",
                ],
            },
        )
        return retrieve_synth_TRITON_SWMM_test_case(
            analysis_name="synth_sensitivity_mixed_cpu_gpu_fanout",
            toggle_tritonswmm_model=True,
            toggle_triton_model=False,
            toggle_swmm_model=False,
            sensitivity_csv=csv_path,
            hpc_system_config_yaml=(Path(__file__).parent / "hpc_system_config_multipartition.yaml"),
            start_from_scratch=start_from_scratch,
            additional_analysis_configs={
                "hpc_ensemble_partition": "gpu-a6000",
            },
        )

    @staticmethod
    def retrieve_synth_cpu_config_sensitivity_case_hpc_gpu_hardware_rejected(
        start_from_scratch: bool = False,
    ):
        """Phase-6 DQ4 — a direct `hpc.gpu_hardware` overlay column is allowlist-
        REJECTED (gpu_hardware is derived-only, R7). Constructing this case raises
        a ConfigurationError pointing the user to `hpc.partition`."""
        _require_cpu_cores_for_sensitivity()
        csv_path = Local_TestCases._write_synth_sensitivity_csv(
            analysis_name="synth_sensitivity_hpc_gpu_hardware_rejected",
            model_subset="all",
            extra_columns={
                "hpc.gpu_hardware": ["a6000", "a100", "a6000", "a100"],
            },
        )
        return retrieve_synth_TRITON_SWMM_test_case(
            analysis_name="synth_sensitivity_hpc_gpu_hardware_rejected",
            toggle_tritonswmm_model=True,
            toggle_triton_model=False,
            toggle_swmm_model=False,
            sensitivity_csv=csv_path,
            hpc_system_config_yaml=(Path(__file__).parent / "hpc_system_config_multipartition.yaml"),
            start_from_scratch=start_from_scratch,
            additional_analysis_configs={
                "hpc_ensemble_partition": "gpu-a6000",
            },
        )

    @staticmethod
    def retrieve_synth_cpu_config_sensitivity_case_all_analysis_prefixed(
        start_from_scratch: bool = False,
    ):
        """Phase 2 R2 — all analysis-config columns use the canonical `analysis.` prefix."""
        _require_cpu_cores_for_sensitivity()
        csv_path = Local_TestCases._write_synth_sensitivity_csv(
            analysis_name="synth_sensitivity_all_analysis_prefixed",
            model_subset="all",
            drop_columns=["run_mode", "n_mpi_procs", "n_omp_threads", "n_gpus", "n_nodes"],
            extra_columns={
                "analysis.run_mode": ["mpi", "openmp", "hybrid", "serial"],
                "analysis.n_mpi_procs": [2, 1, 2, 1],
                "analysis.n_omp_threads": [1, 2, 2, 1],
                "analysis.n_gpus": [0, 0, 0, 0],
                "analysis.n_nodes": [1, 1, 1, 1],
            },
        )
        return retrieve_synth_TRITON_SWMM_test_case(
            analysis_name="synth_sensitivity_all_analysis_prefixed",
            toggle_tritonswmm_model=True,
            toggle_triton_model=False,
            toggle_swmm_model=False,
            sensitivity_csv=csv_path,
            start_from_scratch=start_from_scratch,
        )

    @staticmethod
    def retrieve_synth_cpu_config_sensitivity_case_mixed_prefixed_columns(
        start_from_scratch: bool = False,
    ):
        """Phase 2 R10 — mixed bare + `analysis.` + `system.` columns."""
        _require_cpu_cores_for_sensitivity()
        csv_path = Local_TestCases._write_synth_sensitivity_csv(
            analysis_name="synth_sensitivity_mixed_prefixed_columns",
            model_subset="all",
            drop_columns=["n_mpi_procs"],
            extra_columns={
                # `n_omp_threads` retained as bare (deprecated path); `analysis.n_mpi_procs`
                # introduced as canonical prefixed replacement.
                "analysis.n_mpi_procs": [2, 1, 2, 1],
                "system.target_dem_resolution": [1.0, 1.0, 2.0, 2.0],
            },
        )
        return retrieve_synth_TRITON_SWMM_test_case(
            analysis_name="synth_sensitivity_mixed_prefixed_columns",
            toggle_tritonswmm_model=True,
            toggle_triton_model=False,
            toggle_swmm_model=False,
            sensitivity_csv=csv_path,
            start_from_scratch=start_from_scratch,
        )

    @staticmethod
    def _write_synth_sensitivity_csv(
        analysis_name: str,
        model_subset: str,
        extra_columns: dict[str, list] | None = None,
        drop_columns: list[str] | None = None,
    ) -> Path:
        # Sibling dir (not under runs_root/<analysis_name>) so constructor's
        # start_from_scratch wipe of the analysis dir does not delete the CSV.
        # Per-worktree rooting matches test_case_builder so concurrent runs in
        # sibling worktrees do not race on the sensitivity CSV.
        runs_root = Path(platformdirs.user_cache_dir("hhemt")) / "synthetic_test_runs" / worktree_slug()
        dest_dir = runs_root / "_sensitivity_configs"
        dest_dir.mkdir(parents=True, exist_ok=True)
        csv_path = dest_dir / f"{analysis_name}.csv"
        # Row structure mirrors Norfolk's cpu_benchmarking_analysis*.xlsx so the
        # synth sensitivity tier exercises the same run_mode combinations as the
        # Norfolk tier. Coupled/triton subsets get the 4-row MPI/OMP/HYB/serial
        # matrix; swmm-only subset gets the 3-row thread-count ladder because
        # SWMM has no MPI path.
        if model_subset in ("all", "triton"):
            df = pd.DataFrame(
                {
                    "sa_id": [0, 1, 2, 3],
                    "run_mode": ["mpi", "openmp", "hybrid", "serial"],
                    "n_mpi_procs": [2, 1, 2, 1],
                    "n_omp_threads": [1, 2, 2, 1],
                    "n_gpus": [0, 0, 0, 0],
                    "n_nodes": [1, 1, 1, 1],
                }
            )
        elif model_subset == "swmm":
            df = pd.DataFrame(
                {
                    "sa_id": [0, 1, 2],
                    "run_mode": ["openmp", "openmp", "serial"],
                    "n_mpi_procs": [1, 1, 1],
                    # SWMM 5.2 clamps NumThreads=1 when Nobjects[LINK] < 4*NumThreads
                    # (project.c:269). The synth full SWMM model has 12 conduits, so it
                    # honors at most 3 threads (12 >= 4*3); 4 would silently clamp to 1
                    # and fail the resource-match check. sa_0=3 gives a genuinely
                    # multithreaded run whose actual matches expected.
                    "n_omp_threads": [3, 2, 1],
                    "n_gpus": [0, 0, 0],
                    "n_nodes": [1, 1, 1],
                }
            )
        else:
            raise ValueError(f"model_subset must be 'all', 'triton', or 'swmm'; got {model_subset!r}")
        if extra_columns:
            for col_name, col_values in extra_columns.items():
                if len(col_values) != len(df):
                    raise ValueError(
                        f"extra_columns[{col_name!r}] has {len(col_values)} values; df has {len(df)} rows."
                    )
                df[col_name] = col_values
        if drop_columns:
            df = df.drop(columns=[c for c in drop_columns if c in df.columns])
        assert all(re.fullmatch(r"[A-Za-z0-9_.]+", str(s)) for s in df["sa_id"]), (
            "sa_id values must match ^[A-Za-z0-9_.]+$"
        )
        df.to_csv(csv_path, index=False)
        return csv_path
