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

Usage:
    from tests.fixtures.test_case_catalog import GetTS_TestCases as tst

    # Local test
    case = tst.retrieve_norfolk_single_sim_test_case(start_from_scratch=True)
    system = case.system

    # HPC test
    case = tst.retrieve_norfolk_UVA_multisim_1cpu_case(start_from_scratch=False)
"""

import os
from pathlib import Path
from typing import Optional, TYPE_CHECKING
from importlib.resources import files
import TRITON_SWMM_toolkit.constants as cnst

# Import from production package
# from TRITON_SWMM_toolkit.constants import (
#     APP_NAME,
#     NORFOLK_EX,
#     NORFOLK_SYSTEM_CONFIG,
#     NORFOLK_CASE_CONFIG,
# )

# Import from test fixtures
from tests.fixtures.test_case_builder import retrieve_TRITON_SWMM_test_case
from TRITON_SWMM_toolkit.examples import NorfolkExample

if TYPE_CHECKING:
    from TRITON_SWMM_toolkit.platform_configs import PlatformConfig
    from TRITON_SWMM_toolkit.examples import TRITON_SWMM_example


class GetTS_TestCases:
    """
    Test case catalog for TRITON-SWMM toolkit.

    Provides factory methods to create test cases with:
    - Synthetic weather data
    - Short simulation durations
    - Platform-specific HPC configurations (via PlatformConfig)
    - Isolated test directories

    Platform-specific methods use centralized PlatformConfig instances from
    TRITON_SWMM_toolkit._testing.platform_configs to eliminate configuration duplication.

    Caching Strategy:
        Use start_from_scratch=False to reuse processed inputs from previous runs,
        significantly speeding up test execution. Set to True when you need a clean slate.
    """

    # test_system_dirname = "tests"
    # n_reporting_tsteps_per_sim = 12
    # TRITON_reporting_timestep_s = 10

    # Local test data paths

    # Frontier sensitivity analysis paths

    # UVA paths and sensitivity analysis

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
        platform_config: Optional["PlatformConfig"] = None,
        analysis_overrides: Optional[dict] = None,
        system_overrides: Optional[dict] = None,
        n_reporting_tsteps_per_sim=cnst.TEST_N_REPORTING_TSTEPS_PER_SIM,
        TRITON_reporting_timestep_s=cnst.TEST_TRITON_REPORTING_TIMESTEP_S,
        test_system_dirname=cnst.TEST_SYSTEM_DIRNAME,
    ) -> retrieve_TRITON_SWMM_test_case:
        """
        Internal helper to create Norfolk test cases.

        Supports both old-style (additional_*_configs) and new-style (platform_config + overrides)
        for backward compatibility during refactoring.

        Args:
            analysis_name: Name for the test analysis
            n_events: Number of weather events
            n_reporting_tsteps_per_sim: Timesteps per simulation
            TRITON_reporting_timestep_s: Reporting interval in seconds
            start_from_scratch: Whether to reprocess inputs
            download_if_exists: Whether to re-download HydroShare data
            platform_config: PlatformConfig instance (new style)
            analysis_overrides: Analysis config overrides (new style)
            system_overrides: System config overrides (new style)
            additional_analysis_configs: Analysis configs (old style)
            additional_system_configs: System configs (old style)
            example_data_dir: Override for data directory location

        Returns:
            retrieve_TRITON_SWMM_test_case instance with configured system
        """

        example_data_dir = None
        if platform_config is not None:
            if platform_config.example_data_dir:
                example_data_dir = platform_config.example_data_dir
            analysis_overrides = analysis_overrides or {}
            system_overrides = system_overrides or {}
            final_analysis_configs = (
                platform_config.to_analysis_dict() | analysis_overrides
            )
            final_system_configs = platform_config.to_system_dict() | system_overrides
        else:
            # When platform_config is None, use overrides directly or empty dicts
            final_analysis_configs = analysis_overrides or {}
            final_system_configs = system_overrides or {}

        example = NorfolkExample.load(
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
        )
        return nrflk_test


from dataclasses import dataclass


@dataclass
class all_examples:
    ex_Nrflk = NorfolkExample.load(download_if_exists=False)


class UVA_TestCases:
    sensitivity_UVA_cpu_minimal = "benchmarking_uva_cpus_minimal.xlsx"
    sensitivity_UVA_cpu_full = "benchmarking_uva_cpus.xlsx"

    @classmethod
    def retrieve_norfolk_UVA_multisim_1cpu_case(
        cls, start_from_scratch: bool = False, download_if_exists: bool = False
    ):
        analysis_overrides = {
            "run_mode": "serial",
            "n_mpi_procs": 1,
            "n_omp_threads": 1,
            "n_gpus": 0,
            "n_nodes": 1,
            "hpc_time_min_per_sim": 2,
        }

        """UVA HPC multi-simulation test with 1 CPU (serial mode)."""
        analysis_name = "UVA_multisim"
        return GetTS_TestCases._retrieve_norfolk_case(
            analysis_name=analysis_name,
            start_from_scratch=start_from_scratch,
            download_if_exists=download_if_exists,
            n_events=8,
            platform_config=cnst.UVA_DEFAULT_PLATFORM_CONFIG,
            analysis_overrides=analysis_overrides,
        )

    @classmethod
    def retrieve_norfolk_UVA_sensitivity_CPU_full_ensemble_short_sims(
        cls, start_from_scratch: bool = False, download_if_exists: bool = False
    ):
        """UVA HPC CPU sensitivity analysis with full ensemble configuration."""
        sensitivity_analysis = (
            all_examples.ex_Nrflk.test_case_directory / cls.sensitivity_UVA_cpu_full
        )
        analysis_overrides = {
            "toggle_sensitivity_analysis": True,
            "sensitivity_analysis": sensitivity_analysis,
            "hpc_time_min_per_sim": 10,
            "run_mode": "serial",
            "n_mpi_procs": 1,
            "n_omp_threads": 1,
            "n_gpus": 0,
            "n_nodes": 1,
        }
        analysis_name = "UVA_sensitivity_CPU_full_ensemble_short_sims"
        return GetTS_TestCases._retrieve_norfolk_case(
            analysis_name=analysis_name,
            start_from_scratch=start_from_scratch,
            download_if_exists=download_if_exists,
            n_events=1,
            platform_config=cnst.UVA_DEFAULT_PLATFORM_CONFIG,
            analysis_overrides=analysis_overrides,
        )

    @classmethod
    def retrieve_norfolk_UVA_sensitivity_CPU_minimal(
        cls, start_from_scratch: bool = False, download_if_exists: bool = False
    ):
        """UVA HPC CPU sensitivity analysis with minimal configuration."""
        sensitivity_analysis = (
            all_examples.ex_Nrflk.test_case_directory / cls.sensitivity_UVA_cpu_minimal
        )

        analysis_overrides = {
            "toggle_sensitivity_analysis": True,
            "sensitivity_analysis": sensitivity_analysis,
            "hpc_time_min_per_sim": 2,
            "run_mode": "serial",
            "n_mpi_procs": 1,
            "n_omp_threads": 1,
            "n_gpus": 0,
            "n_nodes": 1,
        }
        analysis_name = "UVA_sensitivity_CPU"
        return GetTS_TestCases._retrieve_norfolk_case(
            analysis_name=analysis_name,
            start_from_scratch=start_from_scratch,
            download_if_exists=download_if_exists,
            n_events=1,
            platform_config=cnst.UVA_DEFAULT_PLATFORM_CONFIG,
            analysis_overrides=analysis_overrides,
        )

    # ========== Frontier HPC Test Cases ==========


class Frontier_TestCases:
    sensitivity_frontier_all_configs_minimal = "benchmarking_frontier_minimal.xlsx"

    @classmethod
    def retrieve_norfolk_frontier_multisim_gpu_case(
        cls, start_from_scratch: bool = False, download_if_exists: bool = False
    ):
        """Frontier HPC multi-simulation test with GPU acceleration."""
        analysis_overrides = {
            "run_mode": "gpu",
            "n_gpus": 1,
            "hpc_time_min_per_sim": 2,
            "n_mpi_procs": 1,
            "n_omp_threads": 1,
            "n_nodes": 1,
            "hpc_total_nodes": 1,
            "hpc_total_job_duration_min": 30,
            "hpc_gpus_per_node": 8,
            "additional_SBATCH_params": ["-q debug"],
        }
        analysis_name = "frontier_multisim_GPU"
        return GetTS_TestCases._retrieve_norfolk_case(
            analysis_name=analysis_name,
            start_from_scratch=start_from_scratch,
            download_if_exists=download_if_exists,
            n_events=20,
            platform_config=cnst.FRONTIER_DEFAULT_PLATFORM_CONFIG,
            analysis_overrides=analysis_overrides,
        )

    @classmethod
    def retrieve_norfolk_frontier_multisim_cpu_serial_case(
        cls, start_from_scratch: bool = False, download_if_exists: bool = False
    ):
        """Frontier HPC multi-simulation test with serial CPU execution."""
        analysis_overrides = {
            "run_mode": "serial",
            "hpc_time_min_per_sim": 2,
            "n_mpi_procs": 1,
            "n_omp_threads": 1,
            "n_nodes": 1,
            "n_gpus": 0,
            "hpc_total_nodes": 1,
            "hpc_total_job_duration_min": 30,
            "hpc_gpus_per_node": 8,
            "additional_SBATCH_params": ["-q debug"],
        }
        analysis_name = "frontier_multisim_CPU"
        return GetTS_TestCases._retrieve_norfolk_case(
            analysis_name=analysis_name,
            start_from_scratch=start_from_scratch,
            download_if_exists=download_if_exists,
            n_events=20,
            platform_config=cnst.FRONTIER_DEFAULT_PLATFORM_CONFIG,
            analysis_overrides=analysis_overrides,
        )

    @classmethod
    def retrieve_norfolk_frontier_sensitivity_minimal(
        cls, start_from_scratch: bool = False, download_if_exists: bool = False
    ):
        """Frontier HPC sensitivity analysis with minimal configuration."""
        analysis_name = "frontier_sensitivity_minimal"
        sensitivity = (
            all_examples.ex_Nrflk.test_case_directory
            / cls.sensitivity_frontier_all_configs_minimal
        )
        analysis_overrides = {
            "toggle_sensitivity_analysis": True,
            "sensitivity_analysis": sensitivity,
            "run_mode": "serial",
            "hpc_time_min_per_sim": 2,
            "n_mpi_procs": 1,
            "n_omp_threads": 1,
            "n_nodes": 1,
            "n_gpus": 0,
            "hpc_total_nodes": 1,
            "hpc_total_job_duration_min": 30,
            "hpc_gpus_per_node": 8,
            "additional_SBATCH_params": ["-q debug"],
        }
        return GetTS_TestCases._retrieve_norfolk_case(
            analysis_name=analysis_name,
            start_from_scratch=start_from_scratch,
            download_if_exists=download_if_exists,
            n_events=1,
            platform_config=cnst.FRONTIER_DEFAULT_PLATFORM_CONFIG,
            analysis_overrides=analysis_overrides,
        )

    # ========== Local Test Cases ==========


class Local_TestCases:
    cpu_sensitivity = "cpu_benchmarking_analysis.xlsx"

    @classmethod
    def retrieve_norfolk_cpu_config_sensitivity_case(
        cls, start_from_scratch: bool = False, download_if_exists: bool = False
    ):
        """Local CPU configuration sensitivity analysis test."""
        analysis_name = "cpu_config_sensitivity"
        sensitivity = all_examples.ex_Nrflk.test_case_directory / cls.cpu_sensitivity
        analysis_overrides = {
            "toggle_sensitivity_analysis": True,
            "sensitivity_analysis": sensitivity,
        }

        return GetTS_TestCases._retrieve_norfolk_case(
            analysis_name=analysis_name,
            start_from_scratch=start_from_scratch,
            download_if_exists=download_if_exists,
            n_events=1,
            analysis_overrides=analysis_overrides,
        )

    @classmethod
    def retrieve_norfolk_single_sim_test_case(
        cls, start_from_scratch: bool = False, download_if_exists: bool = False
    ):
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
        cls, start_from_scratch: bool = False, download_if_exists: bool = False
    ):
        """Local multi-simulation test with 2 events."""
        analysis_name = "multi_sim"

        return GetTS_TestCases._retrieve_norfolk_case(
            analysis_name=analysis_name,
            start_from_scratch=start_from_scratch,
            download_if_exists=download_if_exists,
            n_events=2,
        )
