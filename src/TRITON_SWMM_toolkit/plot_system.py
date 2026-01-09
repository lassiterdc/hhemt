from TRITON_SWMM_toolkit.process_simulation import (
    convert_coords_to_dtype,
    convert_datavars_to_dtype,
)
import sys
import pandas as pd
import xarray as xr
import numpy as np
from TRITON_SWMM_toolkit.utils import (
    write_zarr,
    write_zarr_then_netcdf,
    paths_to_strings,
    current_datetime_string,
    get_file_size_MiB,
)
from typing import Literal, List
from typing import TYPE_CHECKING
from pathlib import Path
import time
from TRITON_SWMM_toolkit.plot_utils import plot_discrete_raster, plot_continuous_raster
from matplotlib.axes import Axes
from typing import Optional, Literal
import rioxarray as rxr
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from .system import TRITONSWMM_system


class TRITONSWMM_system_plotting:
    def __init__(self, system: "TRITONSWMM_system") -> None:
        self._system = system
        self.cfg_system = system.cfg_system
        self.sys_paths = system.sys_paths

    def processed_dem(self, ax=None):

        dem_outside_watershed_height = self.cfg_system.dem_outside_watershed_height
        dem_building_height = self.cfg_system.dem_building_height
        dem_processed = self.sys_paths.dem_processed
        dem_unprocessed = self.cfg_system.DEM_fullres
        watershed_shapefile = self.cfg_system.watershed_gis_polygon
        res = self.cfg_system.target_dem_resolution
        ax_title = f"DEM ({res}m)"

        rds_dem_unprocessed = rxr.open_rasterio(dem_unprocessed)
        rds_dem_processed = rxr.open_rasterio(dem_processed)

        rds_dem_fullres_for_plotting = self._process_dem_for_plotting(
            rds_dem_unprocessed, dem_outside_watershed_height, dem_building_height
        )
        vmin = rds_dem_fullres_for_plotting.min()  # type: ignore
        vmax = rds_dem_fullres_for_plotting.max()  # type: ignore
        if ax is None:
            fig, ax = plt.subplots(1, 2, figsize=(5, 4), layout="constrained")
        ax2 = plot_continuous_raster(
            rds_dem_processed,  # type: ignore
            cbar_lab="elevation",
            cmap="terrain",
            watershed_shapefile=watershed_shapefile,
            watershed_shapefile_color="red",
            vmin=vmin,
            vmax=vmax,
            ax=ax,
            set_over="white",
        )
        ax2.set_title(ax_title)
        return ax

    def processed_mannings(self, ax=None):

        rds_mannings = self._system.open_processed_mannings_as_rds()
        vmin = rds_mannings.min()  # type: ignore
        vmax = rds_mannings.max()  # type: ignore
        watershed_shapefile = self.cfg_system.watershed_gis_polygon
        res = self.cfg_system.target_dem_resolution
        ax_title = f"Mannings ({res}m)"
        if ax is None:
            fig, ax = plt.subplots(1, 2, figsize=(5, 4), layout="constrained")
        ax2 = plot_continuous_raster(
            rds_mannings,  # type: ignore
            cbar_lab="mannings",
            cmap="viridis",
            watershed_shapefile=watershed_shapefile,
            watershed_shapefile_color="red",
            vmin=vmin,
            vmax=vmax,
            ax=ax,
        )
        ax2.set_title(ax_title)
        return ax

    def dem_and_mannings(self):
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
        self.processed_dem(ax=axes[0])
        self.processed_mannings(ax=axes[1])

    def _process_dem_for_plotting(
        self, rds_dem, dem_out_of_watershed=None, dem_building_height=None
    ):
        """
        Docstring for _process_dem_for_plotting

        :param rds_dem: dem dataset
        :param dem_out_of_watershed: height assigned to DEM outside of watershed
        :param dem_building_height: height assigned to DEM where it overlaps buildings

        This assigns n/a to all dem grid cells that are outside of the watershed or represent buildings
        """
        rds_dem_plot_ready = rds_dem.copy()
        if dem_out_of_watershed is not None:
            mask_not_out_of_shed = rds_dem != dem_out_of_watershed
            rds_dem_plot_ready = rds_dem_plot_ready.where(mask_not_out_of_shed)
        if dem_building_height is not None:
            mask_not_building = rds_dem != dem_building_height
            rds_dem_plot_ready = rds_dem_plot_ready.where(mask_not_building)
        return rds_dem_plot_ready

    def landuse_raster(
        self,
        landuse_raster,
        landuse_lookup,
        landuse_colname,
        landuse_description_colname,
        landuse_plot_color_colname,
        watershed_shapefile,
        watershed_shapefile_color="black",
        ax: Optional[Axes] = None,
    ):
        rds = rxr.open_rasterio(landuse_raster)
        df_lu_lookup = pd.read_csv(landuse_lookup).set_index(landuse_colname)
        labs = df_lu_lookup[landuse_description_colname]
        colors = df_lu_lookup[landuse_plot_color_colname]

        ax = plot_discrete_raster(
            rds,
            cbar_lab="",
            colors=colors,
            labs=labs,
            watershed_shapefile=watershed_shapefile,
            watershed_shapefile_color=watershed_shapefile_color,
            ax=ax,
        )
        return ax
