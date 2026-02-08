#!/usr/bin/env python3
"""
Simple test to reproduce netcdf output issues.
"""
import tempfile
import shutil
from pathlib import Path
import xarray as xr
import numpy as np
import pandas as pd

# Import the write_netcdf function
from TRITON_SWMM_toolkit.utils import write_netcdf, write_zarr

# Create a simple test dataset
def create_test_dataset():
    """Create a simple test dataset similar to TRITON outputs."""
    n_timesteps = 5
    n_x = 10
    n_y = 10

    timesteps = pd.date_range("2000-01-01", periods=n_timesteps, freq="1min")
    x = np.arange(n_x)
    y = np.arange(n_y)

    ds = xr.Dataset(
        {
            "wlevel_m": (["timestep_min", "y", "x"], np.random.rand(n_timesteps, n_y, n_x)),
            "velocity_x_mps": (["timestep_min", "y", "x"], np.random.rand(n_timesteps, n_y, n_x)),
            "velocity_y_mps": (["timestep_min", "y", "x"], np.random.rand(n_timesteps, n_y, n_x)),
        },
        coords={
            "timestep_min": timesteps,
            "x": x,
            "y": y,
        }
    )
    ds.attrs["test_attr"] = "test_value"
    return ds

def test_netcdf_write():
    """Test writing to netcdf format."""
    print("Creating test dataset...")
    ds = create_test_dataset()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test zarr (should work)
        zarr_path = Path(tmpdir) / "test_output.zarr"
        print(f"Writing zarr to {zarr_path}...")
        try:
            write_zarr(ds, zarr_path, compression_level=5)
            print("✓ Zarr write successful")
        except Exception as e:
            print(f"✗ Zarr write failed: {e}")

        # Test netcdf (may fail)
        nc_path = Path(tmpdir) / "test_output.nc"
        print(f"Writing netcdf to {nc_path}...")
        try:
            write_netcdf(ds, nc_path, compression_level=5)
            print("✓ NetCDF write successful")

            # Try to read it back
            print("Reading netcdf back...")
            ds_read = xr.open_dataset(nc_path, engine="h5netcdf")
            print(f"✓ NetCDF read successful, dims: {ds_read.dims}")
            ds_read.close()
        except Exception as e:
            print(f"✗ NetCDF write/read failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_netcdf_write()
