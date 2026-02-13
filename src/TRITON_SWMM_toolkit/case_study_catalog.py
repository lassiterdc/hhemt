from pathlib import Path
from typing import TYPE_CHECKING, Optional, Literal

import yaml

import TRITON_SWMM_toolkit.constants as cnst
from TRITON_SWMM_toolkit.analysis import TRITONSWMM_analysis
from TRITON_SWMM_toolkit.config.analysis import analysis_config
from TRITON_SWMM_toolkit.examples import (
    NorfolkIreneExample,
    TRITON_SWMM_example,
    NorfolkObservedExample,
)
from TRITON_SWMM_toolkit.utils import fast_rmtree
from TRITON_SWMM_toolkit.system import TRITONSWMM_system

if TYPE_CHECKING:
    from TRITON_SWMM_toolkit.platform_configs import PlatformConfig


class all_examples:
    @staticmethod
    def norfolk_irene(download_if_exists: bool = False) -> TRITON_SWMM_example:
        return NorfolkIreneExample.load(download_if_exists=download_if_exists)

    @staticmethod
    def norfolk_observed(download_if_exists: bool = False) -> TRITON_SWMM_example:
        return NorfolkObservedExample.load(download_if_exists=download_if_exists)


class CaseStudyBuilder:
    def __init__(
        self,
        example_name: Literal["norfolk_irene", "norfolk_observed_ensemble"],
        download_if_exists: bool,  # whether to download the example data from scratch
        analysis_name: str,
        start_from_scratch: bool,
        case_system_dirname: str = cnst.CASE_SYSTEM_DIRNAME,
        platform_config: Optional["PlatformConfig"] = None,
        analysis_overrides: dict | None = None,
        system_overrides: dict | None = None,
    ):

        #
        if example_name == "norfolk_irene":
            example = all_examples.norfolk_irene(download_if_exists=download_if_exists)
        elif example_name == "norfolk_observed_ensemble":
            example = all_examples.norfolk_observed(
                download_if_exists=download_if_exists
            )

        self.example = example
        self.system = example.system
        self.analysis = example.analysis

        # define analysis and system configs
        if platform_config is not None:
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

        # Fix mutable default arguments

        for key, val in final_system_configs.items():
            setattr(self.system.cfg_system, key, val)

        # update system directory
        self.system.cfg_system.system_directory = (
            self.system.cfg_system.system_directory.parent / case_system_dirname
        )
        anlysys_dir = self.system.cfg_system.system_directory / analysis_name

        if start_from_scratch and anlysys_dir.exists():
            fast_rmtree(anlysys_dir)
        anlysys_dir.mkdir(parents=True, exist_ok=True)

        new_system_config_yaml = (
            self.system.cfg_system.system_directory / f"{case_system_dirname}.yaml"
        )

        new_system_config_yaml.write_text(
            yaml.safe_dump(
                self.system.cfg_system.model_dump(mode="json"),
                sort_keys=False,  # .dict() for pydantic v1
            )
        )
        # udpate system
        self.system = TRITONSWMM_system(new_system_config_yaml)

        # load single sim analysis
        cfg_analysis = self.example.analysis.cfg_analysis.model_copy()

        # update analysis attributes
        cfg_analysis.analysis_id = analysis_name

        # add additional fields
        for key, val in final_analysis_configs.items():
            setattr(cfg_analysis, key, val)

        cfg_analysis = analysis_config.model_validate(cfg_analysis)
        # write analysis as yaml
        cfg_anlysys_yaml = anlysys_dir / f"{analysis_name}.yaml"
        cfg_anlysys_yaml.write_text(
            yaml.safe_dump(
                cfg_analysis.model_dump(mode="json"),
                sort_keys=False,  # .dict() for pydantic v1
            )
        )
        # update analysis
        self.analysis = TRITONSWMM_analysis(cfg_anlysys_yaml, self.system)


class UVACaseStudies:

    sensitivity_analysis_uva_suite = "full_benchmarking_experiment_uva.xlsx"

    @classmethod
    def observed_ensemble_triton_only(
        cls, start_from_scratch: bool = False, download_if_exists: bool = False
    ):
        """UVA observed TRITON-only observed simulations"""
        example_name = "norfolk_observed_ensemble"
        analysis_name = "uva_observed_triton_only_3.7m_res"

        analysis_overrides = {
            "run_mode": "hybrid",
            "hpc_time_min_per_sim": 4 * 60,
            "n_mpi_procs": 2,
            "n_omp_threads": 2,
            "n_nodes": 1,
            "mem_gb_per_cpu": 2,
            "hpc_max_simultaneous_sims": 100,
            "hpc_total_job_duration_min": 60 * 12,
        }
        system_overrides = {
            "toggle_triton_model": True,
            "toggle_tritonswmm_model": False,
            "toggle_swmm_model": False,
            "gpu_compilation_backend": None,
            "target_dem_resolution": 3.6567656220319873,
        }

        return CaseStudyBuilder(
            example_name=example_name,
            download_if_exists=download_if_exists,
            analysis_name=analysis_name,
            start_from_scratch=start_from_scratch,
            platform_config=cnst.UVA_DEFAULT_PLATFORM_CONFIG,
            analysis_overrides=analysis_overrides,
            system_overrides=system_overrides,
        )

    @classmethod
    def benchmarking(
        cls, start_from_scratch: bool = False, download_if_exists: bool = False
    ):
        """UVA HPC sensitivity analysis."""
        example_name = "norfolk_irene"
        analysis_name = "uva_sensitivity_suite"
        sensitivity = (
            all_examples.norfolk_irene().test_case_directory
            / cls.sensitivity_analysis_uva_suite
        )

        analysis_overrides = {
            "toggle_sensitivity_analysis": True,
            "sensitivity_analysis": sensitivity,
            "run_mode": "serial",
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
            "toggle_tritonswmm_model": True,
            "toggle_swmm_model": False,
            "gpu_compilation_backend": None,
        }

        return CaseStudyBuilder(
            example_name=example_name,
            download_if_exists=download_if_exists,
            analysis_name=analysis_name,
            start_from_scratch=start_from_scratch,
            platform_config=cnst.UVA_DEFAULT_PLATFORM_CONFIG,
            analysis_overrides=analysis_overrides,
            system_overrides=system_overrides,
        )


class FrontierCaseStudies:
    sensitivity_frontier_suite = "full_benchmarking_experiment_frontier.xlsx"

    @classmethod
    def retrieve_norfolk_frontier_sensitivity_suite(
        cls, start_from_scratch: bool = False, download_if_exists: bool = False
    ):
        """Frontier HPC sensitivity analysis."""
        example_name = "norfolk_irene"
        analysis_name = "frontier_sensitivity_suite"
        sensitivity = (
            all_examples.norfolk_irene().test_case_directory
            / cls.sensitivity_frontier_suite
        )
        analysis_overrides = {
            "toggle_sensitivity_analysis": True,
            "sensitivity_analysis": sensitivity,
            "hpc_ensemble_partition": "batch",  # or batch or extended, see https://docs.olcf.ornl.gov/systems/frontier_user_guide.html
            "run_mode": "serial",
            "hpc_time_min_per_sim": 120,  # 60 * 6,
            "n_mpi_procs": 1,
            "n_omp_threads": 1,
            "n_nodes": 1,
            "n_gpus": 0,
            "hpc_total_nodes": 50,  # extended partition is limited to 64 nodes
            "hpc_total_job_duration_min": 120,  # 60 * 12 (2 hour limit for batch partition)
            "hpc_gpus_per_node": 8,
            "mem_gb_per_cpu": 2,
        }
        return CaseStudyBuilder(
            example_name=example_name,
            analysis_name=analysis_name,
            start_from_scratch=start_from_scratch,
            download_if_exists=download_if_exists,
            platform_config=cnst.FRONTIER_DEFAULT_PLATFORM_CONFIG,
            analysis_overrides=analysis_overrides,
        )
