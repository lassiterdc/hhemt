# %%
from TRITON_SWMM_toolkit.config import system_config
import pandas as pd
import rioxarray as rxr
import numpy as np
import xarray as xr
from pathlib import Path
from rasterio.enums import Resampling
import sys


def define_system_paths(system_directory):
    dem_processed = system_directory / "elevation.dem"
    mannings_processed = system_directory / "mannings.dem"
    ## combine into dic
    sys_paths = dict(
        dem_processed=dem_processed,
        mannings_processed=mannings_processed,
    )
    return sys_paths


def create_mannings_raster(
    landuse_lookup: Path,
    landuse_raster: Path,
    landuse_colname: str,
    mannings_colname: str,
):
    df_lu_lookup = pd.read_csv(landuse_lookup).loc[
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
    df_lu_vals = df_lu_vals.join(df_lu_lookup.set_index(landuse_colname), how="left")
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


def flt_to_str_certain_num_of_characters(flt, target_decimal_places, longest_num):
    flt = round(flt, target_decimal_places)
    str_flt = flt.astype(str)
    str_flt = str_flt.apply(lambda x: x.ljust(longest_num, "0"))
    return str_flt


def spatial_resampling(xds_to_resample, xds_target, missingfillval=-9999):
    from rasterio.enums import Resampling
    import xarray as xr

    # resample
    ## https://corteva.github.io/rioxarray/stable/rioxarray.html#rioxarray.raster_dataset.RasterDataset.reproject_match
    ## (https://rasterio.readthedocs.io/en/stable/api/rasterio.enums.html#rasterio.enums.Resampling)
    xds_to_resampled = xds_to_resample.rio.reproject_match(
        xds_target, resampling=Resampling.average
    )
    # fill missing values with prespecified val (this should just corresponds to areas where one dataset has pieces outside the other)
    xds_to_resampled = xr.where(
        xds_to_resampled >= 3.403e37, x=missingfillval, y=xds_to_resampled
    )
    return xds_to_resampled


def write_raster(fpath_raster, rds, raster_metadata=None):
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
        flt_to_str_certain_num_of_characters, args=(target_decimal_places, longest_num)
    )
    # df_padded = df_padded.astype(float)
    df_padded.to_csv(
        fpath_raster, mode=data_write_mode, index=False, header=False, sep=" "
    )


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


def coarsen_dem(dem_unprocessed, target_resolution):
    rds_dem = rxr.open_rasterio(dem_unprocessed)
    crs = rds_dem.rio.crs  # type: ignore
    og_dem_res_xy, og_dem_avg_gridsize = compute_grid_resolution(rds_dem)
    if (rds_dem.data < -100).sum() > 0:  # type: ignore
        sys.exit(
            "Error - gaps found in DEM. Consider interpolating elevations using method = 'nearest' (see below in this function)"
        )
        rds_dem = rds_dem.rio.interpolate_na(method="nearest")
    # coarsen
    rds_dem_coarse = coarsen_georaster(rds_dem, target_resolution)
    return rds_dem_coarse


def create_mannings_raster_matching_dem(
    landuse_lookup,
    landuse_raster,
    landuse_colname,
    mannings_colname,
    dem_unprocessed,
    target_resolution,
    fillna_val=-9999,
):
    rds_mannings = create_mannings_raster(
        landuse_lookup, landuse_raster, landuse_colname, mannings_colname
    )
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


def write_raster_formatted_for_TRITON(
    rds, output: Path, include_metadata: bool, fillna_val=-9999
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
        write_raster(output, rds, raster_metadata)
    else:
        write_raster(output, rds)


def create_dem_for_TRITON(dem_unprocessed, target_resolution, system_directory):
    rds_dem_coarse = coarsen_dem(dem_unprocessed, target_resolution)
    sys_paths = define_system_paths(system_directory)
    write_raster_formatted_for_TRITON(
        rds_dem_coarse, sys_paths["dem_processed"], include_metadata=True
    )
    return rds_dem_coarse


def create_mannings_file_for_TRITON(
    landuse_lookup,
    landuse_raster,
    landuse_colname,
    mannings_colname,
    dem_unprocessed,
    target_resolution,
    system_directory,
):
    rds_mannings_coarse = create_mannings_raster_matching_dem(
        landuse_lookup,
        landuse_raster,
        landuse_colname,
        mannings_colname,
        dem_unprocessed,
        target_resolution,
    )
    sys_paths = define_system_paths(system_directory)
    write_raster_formatted_for_TRITON(
        rds_mannings_coarse, sys_paths["mannings_processed"], include_metadata=True
    )
    return rds_mannings_coarse
