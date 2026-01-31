import os
from pathlib import Path
from typing import Optional, TYPE_CHECKING
from importlib.resources import files
import TRITON_SWMM_toolkit.constants as cnst

import pandas as pd
import numpy as np
import shutil
import yaml
from pathlib import Path
from typing import Optional, TYPE_CHECKING
import TRITON_SWMM_toolkit.constants as cnst

# Import from production package
from TRITON_SWMM_toolkit.config import (
    load_analysis_config,
    analysis_config,
)
from TRITON_SWMM_toolkit.system import TRITONSWMM_system
from TRITON_SWMM_toolkit.analysis import TRITONSWMM_analysis

from TRITON_SWMM_toolkit.examples import TRITON_SWMM_example

# Import from test fixtures
from tests.fixtures.test_case_builder import retrieve_TRITON_SWMM_test_case
from TRITON_SWMM_toolkit.examples import NorfolkIreneExample


if TYPE_CHECKING:
    from TRITON_SWMM_toolkit.platform_configs import PlatformConfig
    from TRITON_SWMM_toolkit.examples import TRITON_SWMM_example
from dataclasses import dataclass


@dataclass
class all_examples:
    from TRITON_SWMM_toolkit.examples import TRITON_SWMM_example

    @staticmethod
    def ex_Nrflk(download_if_exists: bool = False) -> TRITON_SWMM_example:
        return NorfolkIreneExample.load(download_if_exists=download_if_exists)


class CaseStudyBuilder:
    def __init__(
        self,
        example: TRITON_SWMM_example,
        case_system_dirname: str,
        analysis_name: str,
        start_from_scratch: bool = False,
        additional_analysis_configs: Optional[dict] = None,
        additional_system_configs: Optional[dict] = None,
    ):
        # Fix mutable default arguments
        additional_analysis_configs = additional_analysis_configs or {}
        additional_system_configs = additional_system_configs or {}

        # load system
        self.system = example.system
        self.analysis = example.analysis

        for key, val in additional_system_configs.items():
            setattr(self.system.cfg_system, key, val)

        # update system directory
        self.system.cfg_system.system_directory = (
            self.system.cfg_system.system_directory.parent / case_system_dirname
        )
        anlysys_dir = self.system.cfg_system.system_directory / analysis_name

        if start_from_scratch and anlysys_dir.exists():
            shutil.rmtree(anlysys_dir)
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

        # load single sime analysis
        cfg_analysis = example.analysis.cfg_analysis.model_copy()

        # update analysis attributes
        cfg_analysis.analysis_id = analysis_name

        # add additional fields
        for key, val in additional_analysis_configs.items():
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


class getTS_CaseStudy:
    def __init__(
        self,
    ) -> None:
        pass

    @classmethod
    def _retrieve_norfolk_irene_case(
        cls,
        analysis_name: str,
        start_from_scratch: bool,
        platform_config: Optional["PlatformConfig"] = None,
        example_data_dir: Optional[Path] = None,
        analysis_overrides: Optional[dict] = None,
        system_overrides: Optional[dict] = None,
        download_if_exists: bool = False,
    ):
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

        case = CaseStudyBuilder(
            example=example,
            case_system_dirname=cnst.CASE_SYSTEM_DIRNAME,
            analysis_name=analysis_name,
            start_from_scratch=start_from_scratch,
            additional_analysis_configs=final_analysis_configs,
            additional_system_configs=final_system_configs,
        )

        return case


class FrontierCaseStudies:
    sensitivity_frontier_suite = "full_benchmarking_experiment_frontier.xlsx"

    @classmethod
    def retrieve_norfolk_frontier_sensitivity_suite(
        cls, start_from_scratch: bool = False, download_if_exists: bool = False
    ):
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
        }
        return getTS_CaseStudy._retrieve_norfolk_irene_case(
            analysis_name=analysis_name,
            start_from_scratch=start_from_scratch,
            download_if_exists=download_if_exists,
            platform_config=cnst.FRONTIER_DEFAULT_PLATFORM_CONFIG,
            analysis_overrides=analysis_overrides,
        )
