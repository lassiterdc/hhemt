"""
Full SWMM model preparation.

This module creates a full SWMM model (combined hydrology and hydraulics) as an .inp file
from a template, using the scenario's time series, rain gauges, simulation timing and
reporting intervals.
"""

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .scenario import TRITONSWMM_scenario


class SWMMFullModelBuilder:
    """
    Builder that writes a full SWMM model (combined hydrology + hydraulics) from a template.

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

    def create_full_model_from_template(self, swmm_model_template, destination: Path) -> None:
        """
        Create full SWMM model from template file.

        The template is filled with scenario-specific values including time series data,
        rain gauges, simulation timing, and reporting intervals, and the result is written
        to ``destination`` as an .inp file.

        Parameters
        ----------
        swmm_model_template : Path
            Path to the full SWMM template file
        destination : Path
            Path where the filled template should be written (typically full.inp)

        """
        from .swmm_utils import create_swmm_inp_from_template

        create_swmm_inp_from_template(self.scenario, swmm_model_template, destination)
        return
