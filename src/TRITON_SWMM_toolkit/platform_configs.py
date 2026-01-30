"""
HPC platform configuration presets for TRITON-SWMM testing.

This module centralizes platform-specific configurations to eliminate duplication
across test cases. Each PlatformConfig dataclass defines:
- SLURM partition and account settings
- Execution method (batch_job vs 1_job_many_srun_tasks)
- Module loading requirements for HPC environments
- GPU backend (HIP for Frontier, CUDA for UVA)

Usage:
    from TRITON_SWMM_toolkit._testing.platform_configs import FRONTIER

    # In test case retrieval:
    platform_config = FRONTIER
    analysis_overrides = {'run_mode': 'gpu', 'n_gpus': 1}
    additional_analysis_configs = platform_config.to_analysis_dict() | analysis_overrides
"""

import sys
from dataclasses import dataclass, field
from typing import Dict
from pathlib import Path
from typing import Optional


@dataclass
class PlatformConfig:
    """
    HPC platform configuration preset.

    Attributes:
        name: Platform identifier (e.g., "frontier", "uva")
        hpc_ensemble_partition: SLURM partition for parallel simulations
        hpc_setup_and_analysis_processing_partition: SLURM partition for setup/processing
        hpc_account: SLURM account to charge
        multi_sim_run_method: Execution strategy (batch_job, 1_job_many_srun_tasks)
        additional_modules: Space-separated module names to load on HPC
        gpu_backend: GPU compilation backend (HIP, CUDA, or empty string for CPU-only)
        python_path: Path to Python interpreter (defaults to current sys.executable)
    """

    name: str

    # Analysis configuration fields
    hpc_ensemble_partition: str
    hpc_setup_and_analysis_processing_partition: str
    hpc_account: str
    multi_sim_run_method: str

    # System configuration fields
    additional_modules: str
    gpu_backend: str

    # Optional fields with defaults
    python_path: str = field(default_factory=lambda: sys.executable)
    example_data_dir: Optional[Path] = None

    def to_analysis_dict(self) -> Dict:
        """
        Convert platform config to analysis configuration dictionary.

        Returns:
            Dictionary with keys compatible with analysis_config fields.
        """
        return {
            "hpc_ensemble_partition": self.hpc_ensemble_partition,
            "hpc_setup_and_analysis_processing_partition": self.hpc_setup_and_analysis_processing_partition,
            "hpc_account": self.hpc_account,
            "multi_sim_run_method": self.multi_sim_run_method,
            "python_path": self.python_path,
        }

    def to_system_dict(self) -> Dict:
        """
        Convert platform config to system configuration dictionary.

        Returns:
            Dictionary with keys compatible with system_config fields.
        """
        return {
            "additional_modules_needed_to_run_TRITON_SWMM_on_hpc": self.additional_modules,
            "gpu_compilation_backend": self.gpu_backend,
        }
