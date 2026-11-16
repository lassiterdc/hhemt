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

    def create_full_model(self, destination: Path) -> None:
        """
        Write the full SWMM role model of this scenario to ``destination``.

        The role is derived from ``system_config.SWMM_model`` and only the forcing the toolkit owns
        (simulation window, rain gauges, rain and water-level series) is rewritten for this scenario.
        """
        from .swmm_utils import write_swmm_role_model

        write_swmm_role_model(self.scenario, "full", destination)
        return
