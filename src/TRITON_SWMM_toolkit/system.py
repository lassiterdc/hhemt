# %%
from TRITON_SWMM_toolkit.config import load_system_config
import pandas as pd
import rioxarray as rxr
import numpy as np
import xarray as xr
from pathlib import Path
from rasterio.enums import Resampling
import sys
from TRITON_SWMM_toolkit.utils import read_header, read_text_file_as_string
import tempfile
from TRITON_SWMM_toolkit.paths import SysPaths
from typing import Optional
from TRITON_SWMM_toolkit.analysis import TRITONSWMM_analysis
from TRITON_SWMM_toolkit.plot_system import TRITONSWMM_system_plotting


class TRITONSWMM_system:
    def __init__(self, system_config_yaml: Path) -> None:
        self.system_config_yaml = system_config_yaml
        self.cfg_system = load_system_config(system_config_yaml)
        # define additional paths
        self.sys_paths = SysPaths(
            dem_processed=self.cfg_system.system_directory / "elevation.dem",
            mannings_processed=self.cfg_system.system_directory / "mannings.dem",
        )
        self._analysis: Optional["TRITONSWMM_analysis"] = None
        self.plot = TRITONSWMM_system_plotting(self)

    @property
    def analysis(self) -> "TRITONSWMM_analysis":
        if self._analysis is None:
            raise RuntimeError("No analysis defined. Call add_analysis() first.")
        return self._analysis

    def add_analysis(self, analysis_config_yaml: Path):
        # from TRITON_SWMM_toolkit.analysis import TRITONSWMM_analysis

        exp = TRITONSWMM_analysis(analysis_config_yaml, self)
        exp_name = exp.cfg_analysis.analysis_id
        self._analysis = exp
        return

    def process_system_level_inputs(
        self, overwrite_if_exists: bool = False, verbose: bool = False
    ):
        self.create_dem_for_TRITON(overwrite_if_exists, verbose)
        if not self.cfg_system.toggle_use_constant_mannings:
            self.create_mannings_file_for_TRITON(overwrite_if_exists, verbose)

    def create_dem_for_TRITON(
        self, overwrite_if_exists: bool = False, verbose: bool = False
    ):
        dem_processed = self.sys_paths.dem_processed
        if dem_processed.exists() and not overwrite_if_exists:
            out = "DEM file already exists. Not rewriting."
        rds_dem_coarse = self._coarsen_dem()
        self._write_raster_formatted_for_TRITON(
            rds_dem_coarse, dem_processed, include_metadata=True
        )
        out = f"wrote {str(dem_processed)}"
        if verbose:
            print(out)
        return

    def create_mannings_file_for_TRITON(
        self, overwrite_if_exists: bool = False, verbose: bool = False
    ):
        mannings_processed = self.sys_paths.mannings_processed
        if mannings_processed.exists() and not overwrite_if_exists:
            out = "Mannings file already exists. Not rewriting."
        include_metadata = False
        rds_mannings_coarse = self._create_mannings_raster_matching_dem()
        self._write_raster_formatted_for_TRITON(
            rds_mannings_coarse,
            mannings_processed,
            include_metadata=include_metadata,
        )
        out = f"wrote {str(mannings_processed)}"
        if verbose:
            print(out)
        return

    def open_processed_mannings_as_rds(self):  # mannings_processed, dem_processed):
        mannings_processed = self.sys_paths.mannings_processed
        dem_processed = self.sys_paths.dem_processed

        dem_header = "".join(read_header(dem_processed, 6))
        mannings_header = "".join(read_header(mannings_processed, 6))
        if dem_header != mannings_header:
            mannings_data = read_text_file_as_string(mannings_processed)
            mannings_with_header = dem_header + mannings_data
            with tempfile.NamedTemporaryFile(suffix=".asc") as tmp:
                tmp.write(mannings_with_header.encode("utf-8"))
                tmp.flush()
                rds_mannings_processed = rxr.open_rasterio(tmp.name).load()  # type: ignore
        else:
            rds_mannings_processed = rxr.open_rasterio(mannings_processed)
        return rds_mannings_processed

    @property
    def processed_dem_rds(self):
        return rxr.open_rasterio(self.sys_paths.dem_processed)

    def _create_mannings_raster(self):
        landuse_lookup_file = self.cfg_system.landuse_lookup_file
        landuse_raster = self.cfg_system.landuse_raster
        landuse_colname = self.cfg_system.landuse_lookup_class_id_colname
        mannings_colname = self.cfg_system.landuse_lookup_mannings_colname

        df_lu_lookup = pd.read_csv(landuse_lookup_file).loc[  # type: ignore
            :, [landuse_colname, mannings_colname]
        ]
        rds_lu = rxr.open_rasterio(landuse_raster)
        assert isinstance(rds_lu, xr.DataArray)
        arr = rds_lu.data
        unique_values = np.unique(arr[~np.isnan(arr)])
        no_data_value = rds_lu.rio.nodata
        # create dataframe from landuse vals in the landuse raster
        df_lu_vals = pd.Series(index=unique_values, name="placeholder").to_frame()
        df_lu_vals.index.name = landuse_colname
        # join the landuse values present in the raster with the lookup table
        df_lu_vals = df_lu_vals.join(
            df_lu_lookup.set_index(landuse_colname), how="left"
        )
        s_lu_mannings_mapping = df_lu_vals[mannings_colname].copy()
        dict_s_lu_mannings = s_lu_mannings_mapping.to_dict()

        rds_mannings_og = xr.apply_ufunc(
            lambda x: dict_s_lu_mannings.get(
                x, no_data_value
            ),  # Replace with mapped value, or keep original if not in dict
            rds_lu,
            keep_attrs=True,  # Keep original raster attributes
            vectorize=True,
        )
        return rds_mannings_og

    def _coarsen_dem(self):  # dem_unprocessed, target_resolution):
        dem_unprocessed = self.cfg_system.DEM_fullres
        target_resolution = self.cfg_system.target_dem_resolution

        rds_dem = rxr.open_rasterio(dem_unprocessed)
        # crs = rds_dem.rio.crs  # type: ignore
        # og_dem_res_xy, og_dem_avg_gridsize = compute_grid_resolution(rds_dem)
        if (rds_dem.data < -100).sum() > 0:  # type: ignore
            sys.exit(
                "Error - gaps found in DEM. Consider interpolating elevations using method = 'nearest' (see below in this function)"
            )
            rds_dem = rds_dem.rio.interpolate_na(method="nearest")
        # coarsen
        rds_dem_coarse = coarsen_georaster(rds_dem, target_resolution)
        return rds_dem_coarse

    def _create_mannings_raster_matching_dem(self, fillna_val=-9999):
        dem_unprocessed = self.cfg_system.DEM_fullres
        target_resolution = self.cfg_system.target_dem_resolution

        rds_mannings = self._create_mannings_raster()
        rds_dem = rxr.open_rasterio(dem_unprocessed)
        crs = rds_dem.rio.crs  # type: ignore
        assert rds_mannings.rio.crs == rds_dem.rio.crs  # type: ignore
        # resample mannings to og dem resolution to ensure exact alignment of final output
        rds_mannings = spatial_resampling(
            rds_mannings, rds_dem, missingfillval=fillna_val
        ).rio.write_crs(crs)
        assert (
            np.isclose(rds_mannings.rio.resolution(), rds_dem.rio.resolution())  # type: ignore
        ).sum() == 2
        rds_mannings_coarse = coarsen_georaster(rds_mannings, target_resolution)
        assert rds_mannings.min().values > 0
        return rds_mannings_coarse

    def _write_raster_formatted_for_TRITON(
        self, rds, output: Path, include_metadata: bool, fillna_val=-9999
    ):
        __, og_avg_gridsize = compute_grid_resolution(rds)
        ncols = rds.x.shape[0]
        nrows = rds.y.shape[0]
        xllcorner = (
            rds.x.values.min() - og_avg_gridsize / 2
        )  # adjusted from center to corner
        yllcorner = (
            rds.y.values.min() - og_avg_gridsize / 2
        )  # adjusted from center to corner
        # define DEM
        raster_metadata = {
            "ncols         ": ncols,
            "nrows         ": nrows,
            "xllcorner     ": xllcorner,
            "yllcorner     ": yllcorner,
            "cellsize      ": og_avg_gridsize,
            "NODATA_value  ": fillna_val,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        if include_metadata:
            self._write_raster(output, rds, raster_metadata)
        else:
            self._write_raster(output, rds)

    def _write_raster(self, fpath_raster, rds, raster_metadata=None):
        if raster_metadata is not None:
            data_write_mode = "a"
            f = open(fpath_raster, "w")
            for key in raster_metadata:
                f.write(key + str(raster_metadata[key]) + "\n")
            f.close()
        else:
            data_write_mode = "w"
        # create dataframe with the right shape
        df_long = (
            rds.to_dataframe("elevation").reset_index().loc[:, ["x", "y", "elevation"]]
        )
        df = df_long.pivot(index="y", columns="x", values="elevation")
        # ensure y is DESCENDING down and x is ASCENDING to the right
        df = df.sort_index(ascending=False)
        cols_sorted = df.columns.sort_values(ascending=True)
        df = df.loc[:, cols_sorted]
        # pad with zeros to achieve consistent spacing in the resulting file
        target_decimal_places = 5
        longest_num = (
            len(str(df.abs().max().max()).split(".")[0]) + target_decimal_places + 1
        )
        df_padded = df.apply(
            self._flt_to_str_certain_num_of_characters,
            args=(target_decimal_places, longest_num),
        )
        # df_padded = df_padded.astype(float)
        df_padded.to_csv(
            fpath_raster, mode=data_write_mode, index=False, header=False, sep=" "
        )

    def _flt_to_str_certain_num_of_characters(
        self, flt, target_decimal_places, longest_num
    ):
        flt = round(flt, target_decimal_places)  # type: ignore
        str_flt = flt.astype(str)
        str_flt = str_flt.apply(lambda x: x.ljust(longest_num, "0"))
        return str_flt


def spatial_resampling(xds_to_resample, xds_target, missingfillval=-9999):
    from rasterio.enums import Resampling
    import xarray as xr

    # resample
    ## https://corteva.github.io/rioxarray/stable/rioxarray.html#rioxarray.raster_dataset.RasterDataset.reproject_match
    ## (https://rasterio.readthedocs.io/en/stable/api/rasterio.enums.html#rasterio.enums.Resampling)
    xds_to_resampled = xds_to_resample.rio.reproject_match(  # type: ignore
        xds_target, resampling=Resampling.average
    )
    # fill missing values with prespecified val (this should just corresponds to areas where one dataset has pieces outside the other)
    xds_to_resampled = xr.where(
        xds_to_resampled >= 3.403e37, x=missingfillval, y=xds_to_resampled
    )
    return xds_to_resampled


def compute_grid_resolution(rds):
    res_xy = rds.rio.resolution()
    mean_grid_size = np.sqrt(abs(res_xy[0]) * abs(res_xy[1]))
    return res_xy, mean_grid_size


def coarsen_georaster(rds, target_resolution):
    crs = rds.rio.crs
    __, og_avg_gridsize = compute_grid_resolution(rds)
    res_multiplier = target_resolution / og_avg_gridsize
    target_res = og_avg_gridsize * res_multiplier
    rds_coarse = rds.rio.reproject(  # type: ignore
        crs, resolution=target_res, resampling=Resampling.average
    )  # aggregate cells
    __, coarse_avg_gridsize = compute_grid_resolution(rds_coarse)
    assert np.isclose(coarse_avg_gridsize, target_resolution)
    return rds_coarse
