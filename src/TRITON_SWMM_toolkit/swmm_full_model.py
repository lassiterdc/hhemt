"""
SWMM Full Model Preparation Module (Placeholder)

This module is a placeholder for future functionality to create and run full SWMM models
(combined hydrology + hydraulics) as part of scenario preparation.

CURRENT STATUS: NOT IMPLEMENTED
This functionality is not currently built out. The foundations exist to create full SWMM
models from templates, but the complete workflow is not yet implemented.

TODO: To fully implement this module, the following would need to be completed:
0. Decide whether to eliminate redundancy with swmm_runoff_modeling.py (currently
 their create-from-template method has a lot of redundancy)
1. Implement full SWMM model execution (currently only template creation exists)
2. Add output processing for full SWMM results
3. Create comparison/validation methods between full SWMM and TRITON-SWMM results
4. Add configuration options for when to use full SWMM vs TRITON-SWMM
5. Document use cases where full SWMM modeling is beneficial
6. Add tests for full SWMM workflow

The primary use case for full SWMM models would be:
- Validation/comparison with TRITON-SWMM results
- Scenarios where TRITON-SWMM coupling is not needed
- Benchmarking and performance comparison studies
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import TYPE_CHECKING
import TRITON_SWMM_toolkit.utils as utils

if TYPE_CHECKING:
    from .scenario import TRITONSWMM_scenario


class SWMMFullModelBuilder:
    """
    Placeholder class for building and running full SWMM models.

    This class provides the foundation for creating full SWMM models (combined
    hydrology + hydraulics) from templates. However, the complete workflow for
    running and processing full SWMM models is not yet implemented.

    Attributes
    ----------
    scenario : TRITONSWMM_scenario
        Reference to the parent scenario object
    cfg_analysis : AnalysisConfig
        Analysis configuration settings
    system : TRITONSWMM_system
        System configuration and paths
    """

    def __init__(self, scenario: "TRITONSWMM_scenario") -> None:
        """
        Initialize the SWMMFullModelBuilder.

        Parameters
        ----------
        scenario : TRITONSWMM_scenario
            The parent scenario object containing configuration and paths
        """
        self.scenario = scenario
        self.cfg_analysis = scenario._analysis.cfg_analysis
        self.system = scenario._system

    def create_full_model_from_template(
        self, swmm_model_template, destination: Path
    ) -> None:
        """
        Create full SWMM model from template file.

        This method creates a full SWMM model (combined hydrology + hydraulics) from
        a template file. The template is filled with scenario-specific values including
        time series data, rain gauges, simulation timing, and reporting intervals.

        NOTE: This only creates the .inp file. Running and processing the full SWMM
        model is not yet implemented.

        Parameters
        ----------
        swmm_model_template : Path
            Path to the full SWMM template file
        destination : Path
            Path where the filled template should be written (typically full.inp)

        TODO:
        - Add method to run full SWMM model
        - Add method to process full SWMM outputs
        - Add comparison methods with TRITON-SWMM results
        """
        from .swmm_utils import create_swmm_inp_from_template

        create_swmm_inp_from_template(self.scenario, swmm_model_template, destination)
        return

    # TODO: Add methods for running and processing full SWMM models
    # def run_full_swmm_model(self, rerun_if_exists: bool = False, verbose: bool = False) -> None:
    #     """Execute full SWMM model."""
    #     pass
    #
    # def process_full_swmm_outputs(self) -> None:
    #     """Process outputs from full SWMM model."""
    #     pass
    #
    # def compare_with_triton_swmm(self) -> None:
    #     """Compare full SWMM results with TRITON-SWMM results."""
    #     pass
