"""Phase 1 smoke tests — validate the synthetic model generator in isolation."""

from __future__ import annotations

import numpy as np
import pytest
import rioxarray  # noqa: F401  (registers .rio accessor)
import swmmio
import xarray as xr

from tests.fixtures.synthetic_model import (
    DEFAULT_PARAMS,
    get_or_build_synthetic_case,
)


def test_build_is_idempotent():
    first = get_or_build_synthetic_case(DEFAULT_PARAMS)
    second = get_or_build_synthetic_case(DEFAULT_PARAMS)
    assert first.cache_dir == second.cache_dir
    assert (first.cache_dir / "build.complete").exists()


def test_all_artifacts_exist():
    arts = get_or_build_synthetic_case(DEFAULT_PARAMS)
    for field_path in [
        arts.dem, arts.landuse, arts.landuse_lookup,
        arts.watershed, arts.boundary,
        arts.swmm_hydraulics, arts.swmm_hydrology, arts.swmm_full,
        arts.subcatchment_raingage_mapping, arts.weather, arts.tritonswmm_cfg,
    ]:
        assert field_path.exists(), f"missing artifact {field_path}"


@pytest.mark.parametrize("variant", ["swmm_hydraulics", "swmm_hydrology", "swmm_full"])
def test_swmm_template_parses(variant):
    arts = get_or_build_synthetic_case(DEFAULT_PARAMS)
    path = getattr(arts, variant)
    model = swmmio.Model(str(path))
    assert model.inp is not None


def test_weather_has_nonzero_variance():
    arts = get_or_build_synthetic_case(DEFAULT_PARAMS)
    ds = xr.open_dataset(arts.weather, engine="h5netcdf")
    assert float(ds["RG_synth"].std()) > 0
    assert float(ds["water_level"].std()) > 0


def test_dem_extent_matches_params():
    arts = get_or_build_synthetic_case(DEFAULT_PARAMS)
    da = rioxarray.open_rasterio(arts.dem)
    assert da.shape[-2:] == (DEFAULT_PARAMS.n_rows, DEFAULT_PARAMS.n_cols)
    # A-4a/A-4b: coord arrays must agree with the on-disk Affine transform at pixel centers.
    cs = DEFAULT_PARAMS.cell_size_m
    expected_y = DEFAULT_PARAMS.yllcorner + cs * (
        np.arange(DEFAULT_PARAMS.n_rows - 1, -1, -1) + 0.5
    )
    expected_x = DEFAULT_PARAMS.xllcorner + cs * (np.arange(DEFAULT_PARAMS.n_cols) + 0.5)
    np.testing.assert_allclose(da.y.values, expected_y, atol=1e-6)
    np.testing.assert_allclose(da.x.values, expected_x, atol=1e-6)
    assert da.rio.crs.to_epsg() == DEFAULT_PARAMS.epsg
