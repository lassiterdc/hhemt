from pathlib import Path
from typing import Literal

import yaml

import hhemt.constants as cnst
from hhemt.analysis import TRITONSWMM_analysis
from hhemt.config.analysis import analysis_config
from hhemt.config.loaders import yaml_to_model
from hhemt.config.report import report_config as report_config_model
from hhemt.config.system import system_config
from hhemt.examples import (
    NorfolkIreneExample,
    NorfolkObservedExample,
    TRITON_SWMM_example,
)
from hhemt.system import TRITONSWMM_system
from hhemt.utils import fast_rmtree

# Example HPC platform overlays (Phase-4 4b: inlined from the retired
# platform_configs.PlatformConfig presets — the data is preserved, only the
# PlatformConfig dataclass is deleted). The literal account/module/login values
# are the Phase-5 anonymization-scrub target (see the public-release
# anonymization plan + the `private identifier occurrences in public tree`
# knowledge doc). 4c/4d will prune the retiring/moving keys (gpu_*,
# additional_modules_*, preferred_slurm_option_*, hpc_account, hpc_login_node,
# hpc_gpus_per_node, hpc_cpus_per_node, hpc_max_simultaneous_sims) from these
# dicts as those fields retire. None-valued preset keys are omitted (a None
# setattr/overlay is a no-op against the field default).
_UVA_ANALYSIS_OVERLAY: dict = {
    "hpc_ensemble_partition": "standard",
    "hpc_setup_and_analysis_processing_partition": "standard",
    "hpc_account": "***REMOVED***",
    "multi_sim_run_method": "batch_job",
    "hpc_gpus_per_node": 8,
    "hpc_max_simultaneous_sims": 1000,
    "hpc_total_job_duration_min": 60 * 8,
    "target_processed_output_type": "zarr",
    "hpc_login_node": "login1.hpc.virginia.edu",
}
_UVA_SYSTEM_OVERLAY: dict = {
    "additional_modules_needed_to_run_TRITON_SWMM_on_hpc": "miniforge gompi/11.4.0_4.1.4 cuda/12.4.1",
    "gpu_compilation_backend": "CUDA",
    "gpu_hardware": "a6000",
    "toggle_triton_model": False,
    "toggle_tritonswmm_model": True,
    "toggle_swmm_model": False,
    "preferred_slurm_option_for_allocating_gpus": "gres",
}
_FRONTIER_ANALYSIS_OVERLAY: dict = {
    "hpc_ensemble_partition": "batch",
    "hpc_setup_and_analysis_processing_partition": "batch",
    "hpc_account": "***REMOVED***",
    "multi_sim_run_method": "1_job_many_srun_tasks",
    "hpc_gpus_per_node": 8,
    "hpc_cpus_per_node": 64,
    "target_processed_output_type": "zarr",
}
_FRONTIER_SYSTEM_OVERLAY: dict = {
    "additional_modules_needed_to_run_TRITON_SWMM_on_hpc": (
        "PrgEnv-amd Core/24.07 craype-accel-amd-gfx90a "
        "miniforge3/23.11.0-0 libfabric/1.22.0"
    ),
    "gpu_compilation_backend": "HIP",
    "toggle_triton_model": False,
    "toggle_tritonswmm_model": True,
    "toggle_swmm_model": False,
    "preferred_slurm_option_for_allocating_gpus": "gpus",
}


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
        analysis_overlay: dict | None = None,
        system_overlay: dict | None = None,
        analysis_overrides: dict | None = None,
        system_overrides: dict | None = None,
        report_config_yaml: Path | None = None,
    ):

        #
        if example_name == "norfolk_irene":
            example = all_examples.norfolk_irene(download_if_exists=download_if_exists)
        elif example_name == "norfolk_observed_ensemble":
            example = all_examples.norfolk_observed(download_if_exists=download_if_exists)

        self.example = example
        self.system = example.system
        self.analysis = example.analysis

        # define analysis and system configs. The example-platform overlay is the
        # base; per-call overrides win (same precedence as the retired
        # PlatformConfig.to_*_dict() | overrides).
        final_analysis_configs = (analysis_overlay or {}) | (analysis_overrides or {})
        final_system_configs = (system_overlay or {}) | (system_overrides or {})

        # Per the per-row-overlay-uses-model_validate stipulation: re-validate the
        # overlaid system config rather than raw-setattr (which skipped validation).
        self.system.cfg_system = system_config.model_validate(
            {**self.system.cfg_system.model_dump(), **final_system_configs}
        )

        # update system directory
        self.system.cfg_system.system_directory = self.system.cfg_system.system_directory.parent / case_system_dirname
        anlysys_dir = self.system.cfg_system.system_directory / analysis_name

        if start_from_scratch and anlysys_dir.exists():
            # EXEMPT-DU: full-analysis-root-wipe
            fast_rmtree(anlysys_dir)
        anlysys_dir.mkdir(parents=True, exist_ok=True)

        new_system_config_yaml = anlysys_dir / "cfg_system.yaml"

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

        # Inline source-side report_config.yaml into cfg_analysis.report when the
        # case method supplies one. The template_analysis_config.yaml ships with
        # report: {} (an empty default) — V0005 retroactively populated this field
        # for already-on-disk analyses but did not update CaseStudyBuilder to do
        # the same at instantiation, so sensitivity cases (whose run-time validator
        # requires report.sensitivity) hit ConfigurationError unless their canonical
        # standalone report_config_*.yaml is threaded here.
        if report_config_yaml is not None:
            cfg_analysis.report = yaml_to_model(Path(report_config_yaml), report_config_model)

        # add additional fields (per-row-overlay-uses-model_validate stipulation:
        # one validating overlay, not setattr-then-validate).
        cfg_analysis = analysis_config.model_validate(
            {**cfg_analysis.model_dump(), **final_analysis_configs}
        )
        # write analysis as yaml
        cfg_anlysys_yaml = anlysys_dir / "cfg_analysis.yaml"
        cfg_anlysys_yaml.write_text(
            yaml.safe_dump(
                cfg_analysis.model_dump(mode="json"),
                sort_keys=False,  # .dict() for pydantic v1
            )
        )
        # update analysis
        self.analysis = TRITONSWMM_analysis(cfg_anlysys_yaml, self.system)

        # Export sensitivity analysis definition if enabled
        if cfg_analysis.toggle_sensitivity_analysis:
            self.analysis.sensitivity.export_sensitivity_definition_csv()


class UVACaseStudies:
    sensitivity_analysis_uva_suite = "full_benchmarking_experiment_uva.xlsx"
    sensitivity_analysis_uva_suite_swmm = "full_benchmarking_experiment_uva_swmm.xlsx"

    @classmethod
    def observed_ensemble_triton_only(cls, start_from_scratch: bool = False, download_if_exists: bool = False):
        """UVA observed TRITON-only observed simulations"""
        example_name = "norfolk_observed_ensemble"
        analysis_name = "uva_observed_triton_only_3.7m_res"

        analysis_overrides = {
            "run_mode": "hybrid",
            "hpc_time_min_per_sim": 6 * 60,
            "n_mpi_procs": 2,
            "n_omp_threads": 2,
            "n_nodes": 1,
            "mem_gb_per_cpu": 2,
            "hpc_max_simultaneous_sims": 100,
            "hpc_total_job_duration_min": 60 * 72,
        }
        system_overrides = {
            "toggle_triton_model": True,
            "toggle_tritonswmm_model": False,
            "toggle_swmm_model": False,
            "gpu_compilation_backend": None,
            "target_dem_resolution": 3.6567656220319873,
            "processed_xllcorner": 3696703.833599999,
            "processed_yllcorner": 1059880.7176479595,
            "ncols": 526,
            "nrows": 513,
        }

        return CaseStudyBuilder(
            example_name=example_name,
            download_if_exists=download_if_exists,
            analysis_name=analysis_name,
            start_from_scratch=start_from_scratch,
            analysis_overlay=_UVA_ANALYSIS_OVERLAY,
            system_overlay=_UVA_SYSTEM_OVERLAY,
            analysis_overrides=analysis_overrides,
            system_overrides=system_overrides,
            case_system_dirname="case_og_dem_res_3.7m",
        )

    @classmethod
    def benchmarking_norfolk_irene(cls, start_from_scratch: bool = False, download_if_exists: bool = False):
        """UVA HPC sensitivity analysis."""
        example_name = "norfolk_irene"
        analysis_name = "uva_sensitivity_suite"
        example_dir = all_examples.norfolk_irene().test_case_directory
        sensitivity = example_dir / cls.sensitivity_analysis_uva_suite
        report_config_yaml = example_dir / "report_config_uva_benchmarking_norfolk_irene.yaml"

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
            "toggle_triton_model": False,
            "toggle_tritonswmm_model": True,
            "toggle_swmm_model": False,
            "gpu_compilation_backend": "CUDA",
        }

        return CaseStudyBuilder(
            example_name=example_name,
            download_if_exists=download_if_exists,
            analysis_name=analysis_name,
            start_from_scratch=start_from_scratch,
            analysis_overlay=_UVA_ANALYSIS_OVERLAY,
            system_overlay=_UVA_SYSTEM_OVERLAY,
            analysis_overrides=analysis_overrides,
            system_overrides=system_overrides,
            report_config_yaml=report_config_yaml,
        )

    @classmethod
    def benchmarking_norfolk_irene_triton_only(cls, start_from_scratch: bool = False, download_if_exists: bool = False):
        """UVA HPC sensitivity analysis."""
        example_name = "norfolk_irene"
        analysis_name = "uva_sensitivity_suite_triton_only"
        example_dir = all_examples.norfolk_irene().test_case_directory
        sensitivity = example_dir / cls.sensitivity_analysis_uva_suite
        report_config_yaml = example_dir / "report_config_uva_benchmarking_norfolk_irene.yaml"

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
            "toggle_tritonswmm_model": False,
            "toggle_swmm_model": False,
            "gpu_compilation_backend": "CUDA",
        }

        return CaseStudyBuilder(
            example_name=example_name,
            download_if_exists=download_if_exists,
            analysis_name=analysis_name,
            start_from_scratch=start_from_scratch,
            analysis_overlay=_UVA_ANALYSIS_OVERLAY,
            system_overlay=_UVA_SYSTEM_OVERLAY,
            analysis_overrides=analysis_overrides,
            system_overrides=system_overrides,
            report_config_yaml=report_config_yaml,
        )

    @classmethod
    def benchmarking_norfolk_irene_swmm_only(cls, start_from_scratch: bool = False, download_if_exists: bool = False):
        """UVA HPC sensitivity analysis."""
        example_name = "norfolk_irene"
        analysis_name = "uva_sensitivity_suite_swmm_only"
        example_dir = all_examples.norfolk_irene().test_case_directory
        sensitivity = example_dir / cls.sensitivity_analysis_uva_suite_swmm
        report_config_yaml = example_dir / "report_config_uva_benchmarking_norfolk_irene_swmm.yaml"

        analysis_overrides = {
            "toggle_sensitivity_analysis": True,
            "sensitivity_analysis": sensitivity,
            "run_mode": "serial",
            "n_mpi_procs": 1,
            "hpc_time_min_per_sim": 60 * 6,
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

        return CaseStudyBuilder(
            example_name=example_name,
            download_if_exists=download_if_exists,
            analysis_name=analysis_name,
            start_from_scratch=start_from_scratch,
            analysis_overlay=_UVA_ANALYSIS_OVERLAY,
            system_overlay=_UVA_SYSTEM_OVERLAY,
            analysis_overrides=analysis_overrides,
            system_overrides=system_overrides,
            report_config_yaml=report_config_yaml,
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
        example_dir = all_examples.norfolk_irene().test_case_directory
        sensitivity = example_dir / cls.sensitivity_frontier_suite
        report_config_yaml = example_dir / "report_config_frontier_norfolk_sensitivity_suite.yaml"
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
            analysis_overlay=_FRONTIER_ANALYSIS_OVERLAY,
            system_overlay=_FRONTIER_SYSTEM_OVERLAY,
            analysis_overrides=analysis_overrides,
            report_config_yaml=report_config_yaml,
        )
