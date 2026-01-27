
import TRITON_SWMM_toolkit.utils as utils
import TRITON_SWMM_toolkit.plot_utils as plt_utils

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .analysis import TRITONSWMM_analysis


class TRITONSWMM_analysis_plotting:
    def __init__(self, analysis: "TRITONSWMM_analysis") -> None:
        self._analysis = analysis
        self._system = analysis._system
        self.sys_paths = analysis._system.sys_paths

    @property
    def triton_ds(self):
        return self._analysis.TRITON_summary

    @property
    def swmm_node_ds(self):
        return self._analysis.SWMM_node_summary

    @property
    def swmm_link_ds(self):
        return self._analysis.SWMM_link_summary

    def max_wlevel(self, event_iloc):
        da = self.triton_ds["max_wlevel_m"].isel(event_iloc=event_iloc)
        watershed_shapefile = self._system.cfg_system.watershed_gis_polygon
        mask = utils.create_mask_from_shapefile(da, watershed_shapefile)
        plt_utils.plot_continuous_raster(
            da.where(mask & (da > 0)),
            cbar_lab="max_wlevel_m",
            watershed_shapefile=watershed_shapefile,
        )
