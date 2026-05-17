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
from typing import TYPE_CHECKING, Optional

import pandas as pd
import platformdirs
import pytest

import TRITON_SWMM_toolkit.constants as cnst

# Import from test fixtures
from tests.fixtures.test_case_builder import (
    retrieve_synth_TRITON_SWMM_test_case,
    retrieve_TRITON_SWMM_test_case,
)
from TRITON_SWMM_toolkit.examples import NorfolkIreneExample

if TYPE_CHECKING:
    from TRITON_SWMM_toolkit.platform_configs import PlatformConfig


def _require_cpu_cores_for_sensitivity(min_cores: int = 4) -> None:
    """Skip the current test if the host has fewer than ``min_cores`` CPU cores.

    The synth sensitivity CSV includes a hybrid row (2 MPI x 2 OMP = 4 threads);
    on machines with fewer cores the honored-thread validator would flag a
    mismatch only after a full workflow run. Failing fast here costs seconds,
    not minutes.
    """
    n = os.cpu_count() or 1
    if n < min_cores:
        pytest.skip(
            f"synth sensitivity suite requires >={min_cores} CPU cores; found {n}"
        )


@dataclass
class all_examples:
    from TRITON_SWMM_toolkit.examples import TRITON_SWMM_example

    @staticmethod
    def ex_Nrflk(download_if_exists: bool = False) -> TRITON_SWMM_example:
        return NorfolkIreneExample.load(download_if_exists=download_if_exists)


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
        analysis_overrides: dict | None = None,
        system_overrides: dict | None = None,
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

        example = NorfolkIreneExample.load(
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


class UVA_TestCases:
    sensitivity_UVA_cpu_minimal = "benchmarking_uva_minimal.xlsx"
    sensitivity_analysis_uva_suite_cpu = (
        "full_benchmarking_experiment_uva_test_cpu.xlsx"
    )
    sensitivity_analysis_uva_suite_gpu = (
        "full_benchmarking_experiment_uva_test_gpu.xlsx"
    )
    sensitivity_analysis_uva_suite_swmm = "full_benchmarking_experiment_uva_swmm.xlsx"
    sensitivity_analysis_uva_suite = (
        "full_benchmarking_experiment_uva_test_all_configs.xlsx"
    )

    @classmethod
    def retrieve_norfolk_UVA_multisim_1cpu_case(
        cls, start_from_scratch: bool = False, download_if_exists: bool = False
    ) -> retrieve_TRITON_SWMM_test_case:
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
    def retrieve_norfolk_UVA_sensitivity_minimal(
        cls, start_from_scratch: bool = False, download_if_exists: bool = False
    ) -> retrieve_TRITON_SWMM_test_case:
        """UVA HPC CPU sensitivity analysis with minimal configuration."""
        sensitivity_analysis = (
            all_examples.ex_Nrflk().test_case_directory
            / cls.sensitivity_UVA_cpu_minimal
        )

        analysis_overrides = {
            "toggle_sensitivity_analysis": True,
            "sensitivity_analysis": sensitivity_analysis,
            "hpc_time_min_per_sim": 20,
            "run_mode": "gpu",
            "n_mpi_procs": 1,
            "n_omp_threads": 1,
            "n_gpus": 1,
            "n_nodes": 1,
        }
        analysis_name = "UVA_sensitivity_minimal"
        return GetTS_TestCases._retrieve_norfolk_case(
            analysis_name=analysis_name,
            start_from_scratch=start_from_scratch,
            download_if_exists=download_if_exists,
            n_events=1,
            platform_config=cnst.UVA_DEFAULT_PLATFORM_CONFIG,
            analysis_overrides=analysis_overrides,
        )

    @classmethod
    def benchmarking_norfolk_irene_cpu(
        cls, start_from_scratch: bool = False, download_if_exists: bool = False
    ):
        """UVA HPC sensitivity analysis."""
        # example_name = "norfolk_irene"
        analysis_name = "test_uva_sensitivity_suite_cpu"
        sensitivity = (
            all_examples.ex_Nrflk().test_case_directory
            / cls.sensitivity_analysis_uva_suite_cpu
        )

        analysis_overrides = {
            "toggle_sensitivity_analysis": True,
            "sensitivity_analysis": sensitivity,
            "run_mode": "serial",
            "hpc_time_min_per_sim": 20,
            "n_mpi_procs": 1,
            "n_omp_threads": 1,
            "n_nodes": 1,
            "n_gpus": 0,
            "mem_gb_per_cpu": 2,
            "hpc_max_simultaneous_sims": 100,
            "hpc_total_job_duration_min": 60 * 72,
        }

        system_overrides = {
            "toggle_triton_model": False,
            "toggle_tritonswmm_model": True,
            "toggle_swmm_model": False,
            "gpu_compilation_backend": "CUDA",
        }

        return GetTS_TestCases._retrieve_norfolk_case(
            analysis_name=analysis_name,
            start_from_scratch=start_from_scratch,
            download_if_exists=download_if_exists,
            n_events=1,
            platform_config=cnst.UVA_DEFAULT_PLATFORM_CONFIG,
            analysis_overrides=analysis_overrides,
            system_overrides=system_overrides,
        )

    @classmethod
    def benchmarking_norfolk_irene_gpu(
        cls, start_from_scratch: bool = False, download_if_exists: bool = False
    ):
        """UVA HPC sensitivity analysis."""
        # example_name = "norfolk_irene"
        analysis_name = "test_uva_sensitivity_suite_gpu"
        sensitivity = (
            all_examples.ex_Nrflk().test_case_directory
            / cls.sensitivity_analysis_uva_suite_gpu
        )

        analysis_overrides = {
            "toggle_sensitivity_analysis": True,
            "sensitivity_analysis": sensitivity,
            "run_mode": "serial",
            "hpc_time_min_per_sim": 20,
            "n_mpi_procs": 1,
            "n_omp_threads": 1,
            "n_nodes": 1,
            "n_gpus": 0,
            "mem_gb_per_cpu": 2,
            "hpc_max_simultaneous_sims": 100,
            "hpc_total_job_duration_min": 60 * 72,
        }

        system_overrides = {
            "toggle_triton_model": False,
            "toggle_tritonswmm_model": True,
            "toggle_swmm_model": False,
            "gpu_compilation_backend": "CUDA",
        }

        return GetTS_TestCases._retrieve_norfolk_case(
            analysis_name=analysis_name,
            start_from_scratch=start_from_scratch,
            download_if_exists=download_if_exists,
            n_events=1,
            platform_config=cnst.UVA_DEFAULT_PLATFORM_CONFIG,
            analysis_overrides=analysis_overrides,
            system_overrides=system_overrides,
        )

    @classmethod
    def benchmarking_norfolk_irene_full_suite(
        cls, start_from_scratch: bool = False, download_if_exists: bool = False
    ):
        """UVA HPC sensitivity analysis."""
        # example_name = "norfolk_irene"
        analysis_name = "test_uva_sensitivity_suite_full_suite"
        sensitivity = (
            all_examples.ex_Nrflk().test_case_directory
            / cls.sensitivity_analysis_uva_suite
        )

        analysis_overrides = {
            "toggle_sensitivity_analysis": True,
            "sensitivity_analysis": sensitivity,
            "run_mode": "serial",
            "hpc_time_min_per_sim": 20,
            "n_mpi_procs": 1,
            "n_omp_threads": 1,
            "n_nodes": 1,
            "n_gpus": 0,
            "mem_gb_per_cpu": 2,
            "hpc_max_simultaneous_sims": 100,
            "hpc_total_job_duration_min": 60 * 72,
        }

        system_overrides = {
            "toggle_triton_model": False,
            "toggle_tritonswmm_model": True,
            "toggle_swmm_model": False,
            "gpu_compilation_backend": "CUDA",
        }

        return GetTS_TestCases._retrieve_norfolk_case(
            analysis_name=analysis_name,
            start_from_scratch=start_from_scratch,
            download_if_exists=download_if_exists,
            n_events=1,
            platform_config=cnst.UVA_DEFAULT_PLATFORM_CONFIG,
            analysis_overrides=analysis_overrides,
            system_overrides=system_overrides,
        )

    @classmethod
    def benchmarking_norfolk_irene_triton_only(
        cls, start_from_scratch: bool = False, download_if_exists: bool = False
    ):
        """UVA HPC sensitivity analysis."""
        example_name = "norfolk_irene"
        analysis_name = "uva_sensitivity_suite_triton_only"
        sensitivity = (
            all_examples.ex_Nrflk().test_case_directory
            / cls.sensitivity_analysis_uva_suite
        )

        analysis_overrides = {
            "toggle_sensitivity_analysis": True,
            "sensitivity_analysis": sensitivity,
            "run_mode": "serial",
            "hpc_time_min_per_sim": 20,
            "n_mpi_procs": 1,
            "n_omp_threads": 1,
            "n_nodes": 1,
            "n_gpus": 0,
            "mem_gb_per_cpu": 2,
            "hpc_max_simultaneous_sims": 100,
            "hpc_total_job_duration_min": 60 * 72,
        }

        system_overrides = {
            "toggle_triton_model": True,
            "toggle_tritonswmm_model": False,
            "toggle_swmm_model": False,
            "gpu_compilation_backend": "CUDA",
        }

        return GetTS_TestCases._retrieve_norfolk_case(
            download_if_exists=download_if_exists,
            analysis_name=analysis_name,
            start_from_scratch=start_from_scratch,
            n_events=1,
            platform_config=cnst.UVA_DEFAULT_PLATFORM_CONFIG,
            analysis_overrides=analysis_overrides,
            system_overrides=system_overrides,
        )

    @classmethod
    def benchmarking_norfolk_irene_swmm_only(
        cls, start_from_scratch: bool = False, download_if_exists: bool = False
    ):
        """UVA HPC sensitivity analysis."""
        example_name = "norfolk_irene"
        analysis_name = "uva_sensitivity_suite_swmm_only"
        sensitivity = (
            all_examples.ex_Nrflk().test_case_directory
            / cls.sensitivity_analysis_uva_suite_swmm
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
            "mem_gb_per_cpu": 2,
            "hpc_max_simultaneous_sims": 100,
            "hpc_total_job_duration_min": 60 * 72,
        }

        system_overrides = {
            "toggle_triton_model": False,
            "toggle_tritonswmm_model": False,
            "toggle_swmm_model": True,
        }

        return GetTS_TestCases._retrieve_norfolk_case(
            download_if_exists=download_if_exists,
            analysis_name=analysis_name,
            start_from_scratch=start_from_scratch,
            n_events=1,
            platform_config=cnst.UVA_DEFAULT_PLATFORM_CONFIG,
            analysis_overrides=analysis_overrides,
            system_overrides=system_overrides,
        )


class Frontier_TestCases:
    sensitivity_frontier_all_configs_minimal = "benchmarking_frontier_minimal.xlsx"
    sensitivity_frontier_suite = "full_benchmarking_experiment_frontier.xlsx"

    @classmethod
    def retrieve_norfolk_frontier_multisim_gpu_case(
        cls, start_from_scratch: bool = False, download_if_exists: bool = False
    ) -> retrieve_TRITON_SWMM_test_case:
        """Frontier HPC multi-simulation test with GPU acceleration."""
        analysis_overrides = {
            "run_mode": "gpu",
            "n_gpus": 1,
            "hpc_time_min_per_sim": 40,
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
    ) -> retrieve_TRITON_SWMM_test_case:
        """Frontier HPC multi-simulation test with serial CPU execution."""
        analysis_overrides = {
            "run_mode": "serial",
            "hpc_time_min_per_sim": 40,
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
    ) -> retrieve_TRITON_SWMM_test_case:
        """Frontier HPC sensitivity analysis with minimal configuration."""
        analysis_name = "frontier_sensitivity_minimal"
        sensitivity = (
            all_examples.ex_Nrflk().test_case_directory
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

    @classmethod
    def retrieve_norfolk_frontier_sensitivity_suite(
        cls, start_from_scratch: bool = False, download_if_exists: bool = False
    ) -> retrieve_TRITON_SWMM_test_case:
        """Frontier HPC sensitivity analysis with minimal configuration."""
        analysis_name = "frontier_sensitivity_suite"
        sensitivity = (
            all_examples.ex_Nrflk().test_case_directory / cls.sensitivity_frontier_suite
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
            "hpc_total_nodes": 8,
            "hpc_total_job_duration_min": 120,
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
    cpu_sensitivity_swmm = "cpu_benchmarking_analysis_swmm.xlsx"

    @classmethod
    def retrieve_norfolk_cpu_config_sensitivity_case(
        cls, start_from_scratch: bool = False, download_if_exists: bool = False
    ) -> retrieve_TRITON_SWMM_test_case:
        """Local CPU configuration sensitivity analysis test."""
        analysis_name = "cpu_config_sensitivity"
        sensitivity = all_examples.ex_Nrflk().test_case_directory / cls.cpu_sensitivity
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
    def retrieve_norfolk_cpu_config_sensitivity_case_triton_only(
        cls, start_from_scratch: bool = False, download_if_exists: bool = False
    ) -> retrieve_TRITON_SWMM_test_case:
        """Local CPU configuration sensitivity analysis test."""
        analysis_name = "cpu_config_sensitivity_triton_only"
        sensitivity = all_examples.ex_Nrflk().test_case_directory / cls.cpu_sensitivity
        analysis_overrides = {
            "toggle_sensitivity_analysis": True,
            "sensitivity_analysis": sensitivity,
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
        sensitivity = (
            all_examples.ex_Nrflk().test_case_directory / cls.cpu_sensitivity_swmm
        )
        analysis_overrides = {
            "toggle_sensitivity_analysis": True,
            "sensitivity_analysis": sensitivity,
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
        cls, start_from_scratch: bool = False, download_if_exists: bool = False
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
    def retrieve_synth_multi_sim_test_case(start_from_scratch: bool = False):
        return retrieve_synth_TRITON_SWMM_test_case(
            analysis_name="synth_multi_sim",
            # Iter-2 of `per_sim_peak_flood_depth` (2026-04-28): bumped from 2
            # to 3 so the per-event peak-flood-depth maps cover the three
            # forcing-mechanism scenarios encoded in weather.py
            # (event 0: hydro-only, event 1: BC-only, event 2: both).
            n_events=3,
            start_from_scratch=start_from_scratch,
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
        peer_path = (
            Path(__file__).parent
            / "synthetic_model"
            / "report_config_synth_sensitivity.yaml"
        )
        return yaml.safe_load(peer_path.read_text())

    @staticmethod
    def retrieve_synth_cpu_config_sensitivity_case(
        start_from_scratch: bool = False,
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
            additional_analysis_configs={
                "report": Local_TestCases._load_synth_sensitivity_report_dict()
            },
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
            additional_analysis_configs={
                "report": Local_TestCases._load_synth_sensitivity_report_dict()
            },
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
            additional_analysis_configs={
                "report": Local_TestCases._load_synth_sensitivity_report_dict()
            },
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
            additional_analysis_configs={
                "report": Local_TestCases._load_synth_sensitivity_report_dict()
            },
        )

    @staticmethod
    def retrieve_synth_cpu_config_sensitivity_case_mutex_violation(
        start_from_scratch: bool = False,
    ):
        """Phase 1 R3 — row with both system_config_yaml AND system.* → ConfigurationError."""
        _require_cpu_cores_for_sensitivity()
        dest_dir = (
            Path(platformdirs.user_cache_dir("TRITON_SWMM_toolkit"))
            / "synthetic_test_runs"
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
            additional_analysis_configs={
                "report": Local_TestCases._load_synth_sensitivity_report_dict()
            },
        )

    @staticmethod
    def retrieve_synth_cpu_config_sensitivity_case_invalid_overlay(
        start_from_scratch: bool = False,
    ):
        """Phase 1 R4 — `system.gpu_compilation_backend='WRONG'` → Pydantic Literal failure."""
        _require_cpu_cores_for_sensitivity()
        csv_path = Local_TestCases._write_synth_sensitivity_csv(
            analysis_name="synth_sensitivity_invalid_overlay",
            model_subset="all",
            extra_columns={
                "system.gpu_compilation_backend": ["WRONG", None, None, None],
            },
        )
        return retrieve_synth_TRITON_SWMM_test_case(
            analysis_name="synth_sensitivity_invalid_overlay",
            toggle_tritonswmm_model=True,
            toggle_triton_model=False,
            toggle_swmm_model=False,
            sensitivity_csv=csv_path,
            start_from_scratch=start_from_scratch,
            additional_analysis_configs={
                "report": Local_TestCases._load_synth_sensitivity_report_dict()
            },
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
            additional_analysis_configs={
                "report": Local_TestCases._load_synth_sensitivity_report_dict()
            },
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
            additional_analysis_configs={
                "report": Local_TestCases._load_synth_sensitivity_report_dict()
            },
        )

    @staticmethod
    def retrieve_synth_cpu_config_sensitivity_case_with_system_gpu_hardware_override(
        start_from_scratch: bool = False,
    ):
        """Phase 1 R8 (T13 equivalence) — `system.gpu_hardware='override-test-gpu'` overlay."""
        _require_cpu_cores_for_sensitivity()
        csv_path = Local_TestCases._write_synth_sensitivity_csv(
            analysis_name="synth_sensitivity_with_system_gpu_hardware_override",
            model_subset="all",
            extra_columns={
                "system.gpu_hardware": [
                    "override-test-gpu", "override-test-gpu",
                    "override-test-gpu", "override-test-gpu",
                ],
            },
        )
        return retrieve_synth_TRITON_SWMM_test_case(
            analysis_name="synth_sensitivity_with_system_gpu_hardware_override",
            toggle_tritonswmm_model=True,
            toggle_triton_model=False,
            toggle_swmm_model=False,
            sensitivity_csv=csv_path,
            start_from_scratch=start_from_scratch,
            additional_analysis_configs={
                "report": Local_TestCases._load_synth_sensitivity_report_dict()
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
                "analysis.run_mode":      ["mpi", "openmp", "hybrid", "serial"],
                "analysis.n_mpi_procs":   [2, 1, 2, 1],
                "analysis.n_omp_threads": [1, 2, 2, 1],
                "analysis.n_gpus":        [0, 0, 0, 0],
                "analysis.n_nodes":       [1, 1, 1, 1],
            },
        )
        return retrieve_synth_TRITON_SWMM_test_case(
            analysis_name="synth_sensitivity_all_analysis_prefixed",
            toggle_tritonswmm_model=True,
            toggle_triton_model=False,
            toggle_swmm_model=False,
            sensitivity_csv=csv_path,
            start_from_scratch=start_from_scratch,
            additional_analysis_configs={
                "report": Local_TestCases._load_synth_sensitivity_report_dict()
            },
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
            additional_analysis_configs={
                "report": Local_TestCases._load_synth_sensitivity_report_dict()
            },
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
        runs_root = (
            Path(platformdirs.user_cache_dir("TRITON_SWMM_toolkit"))
            / "synthetic_test_runs"
        )
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
                    "sa_id":         [0,      1,        2,        3],
                    "run_mode":      ["mpi",  "openmp", "hybrid", "serial"],
                    "n_mpi_procs":   [2,      1,        2,        1],
                    "n_omp_threads": [1,      2,        2,        1],
                    "n_gpus":        [0,      0,        0,        0],
                    "n_nodes":       [1,      1,        1,        1],
                }
            )
        elif model_subset == "swmm":
            df = pd.DataFrame(
                {
                    "sa_id":         [0,        1,        2],
                    "run_mode":      ["openmp", "openmp", "serial"],
                    "n_mpi_procs":   [1,        1,        1],
                    "n_omp_threads": [4,        2,        1],
                    "n_gpus":        [0,        0,        0],
                    "n_nodes":       [1,        1,        1],
                }
            )
        else:
            raise ValueError(
                f"model_subset must be 'all', 'triton', or 'swmm'; got {model_subset!r}"
            )
        if extra_columns:
            for col_name, col_values in extra_columns.items():
                if len(col_values) != len(df):
                    raise ValueError(
                        f"extra_columns[{col_name!r}] has {len(col_values)} values; "
                        f"df has {len(df)} rows."
                    )
                df[col_name] = col_values
        if drop_columns:
            df = df.drop(columns=[c for c in drop_columns if c in df.columns])
        assert all(
            re.fullmatch(r"[A-Za-z0-9_.]+", str(s)) for s in df["sa_id"]
        ), "sa_id values must match ^[A-Za-z0-9_.]+$"
        df.to_csv(csv_path, index=False)
        return csv_path
