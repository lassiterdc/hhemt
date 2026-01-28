"""
Test suite for SWMM output parser refactoring.

This module verifies that refactored code produces identical outputs to the
reference data stored in test_data/swmm_refactoring_reference/.

The tests ensure:
1. Output equivalence - refactored code produces same results as original
2. Warning suppression - no UnstableSpecificationWarning during execution
3. Performance - no significant regression in execution time
"""

import pytest
import warnings
import time
from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
from importlib.resources import files
from TRITON_SWMM_toolkit.examples import GetTS_TestCases as tst

import tempfile
from TRITON_SWMM_toolkit.utils import write_zarr

from TRITON_SWMM_toolkit.swmm_output_parser import (
    retrieve_SWMM_outputs_as_datasets,
    convert_swmm_tdeltas_to_minutes,
    return_swmm_outputs,
    return_node_time_series_results_from_rpt,
    format_rpt_section_into_dataframe,
    return_data_from_rpt,
)
from TRITON_SWMM_toolkit.constants import (
    LST_COL_HEADERS_NODE_FLOOD_SUMMARY,
    LST_COL_HEADERS_NODE_FLOW_SUMMARY,
    LST_COL_HEADERS_LINK_FLOW_SUMMARY,
    APP_NAME,
)


# =============================================================================
# Test Configuration
# =============================================================================


REFERENCE_DATA_DIR = (
    files(APP_NAME).parents[1] / "test_data" / "swmm_refactoring_reference"  # type: ignore
)

# Reference files
REF_INP = REFERENCE_DATA_DIR / "hydraulics.inp"
REF_HYDRAULICS_RPT = REFERENCE_DATA_DIR / "hydraulics.rpt"

# Reference zarr outputs
REF_LINK_TSERIES_ZARR = REFERENCE_DATA_DIR / "SWMM_link_tseries.zarr"
REF_NODE_TSERIES_ZARR = REFERENCE_DATA_DIR / "SWMM_node_tseries.zarr"

# Tolerance for numeric comparisons
RTOL = 1e-5  # Relative tolerance
ATOL = 1e-8  # Absolute tolerance

# Baseline performance (seconds) for retrieve_SWMM_outputs_as_datasets
# Captured on 2026-01-27 using:
#   ds_nodes, ds_links = retrieve_SWMM_outputs_as_datasets(REF_INP, REF_HYDRAULICS_RPT)
BASELINE_RETRIEVE_SECONDS = 20.295690


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def test_case_analysis():
    nrflk_multisim_ensemble = tst.retrieve_norfolk_multi_sim_test_case(
        start_from_scratch=False
    )
    analysis = nrflk_multisim_ensemble.system.analysis
    return analysis


@pytest.fixture(scope="module")
def reference_link_tseries():
    """Load reference link time series dataset."""
    ds_links = xr.open_dataset(REF_LINK_TSERIES_ZARR, engine="zarr", consolidated=False)
    drop_vars = {
        "Barrels",
        "CrestHeight",
        "DischCoeff",
        "FlapGate",
        "Geom1",
        "Geom2",
        "Geom3",
        "Geom4",
        "InitFlow",
        "InletNode",
        "InOffset",
        "Length",
        "MaxFlow",
        "OpenCloseTime",
        "OrificeType",
        "OutletNode",
        "OutOffset",
        "Roughness",
        "Shape",
    }
    return ds_links.drop_vars([var for var in drop_vars if var in ds_links.data_vars])


@pytest.fixture(scope="module")
def reference_node_tseries():
    """Load reference node time series dataset."""
    ds_nodes = xr.open_dataset(REF_NODE_TSERIES_ZARR, engine="zarr", consolidated=False)
    drop_vars = {
        "Coefficient",
        "Constant",
        "Exponent",
        "InitDepth",
        "InvertElev",
        "MaxD",
        "MaxDepth",
        "OutfallType",
        "PondedArea",
        "SurchargeDepth",
        "StageOrTimeseries",
        "StorageCurve",
    }
    return ds_nodes.drop_vars([var for var in drop_vars if var in ds_nodes.data_vars])


def wrap_retrieve_SWMM_outputs_as_datasets(test_case_analysis):
    ds_nodes, ds_links = retrieve_SWMM_outputs_as_datasets(
        REF_INP,
        REF_HYDRAULICS_RPT,
    )
    return ds_nodes, ds_links


@pytest.fixture(scope="module")
def parsed_outputs(test_case_analysis):
    """Parse SWMM outputs using the current implementation."""
    ds_nodes, ds_links = wrap_retrieve_SWMM_outputs_as_datasets(test_case_analysis)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        nodes_path = tmpdir / "nodes.zarr"
        links_path = tmpdir / "links.zarr"

        write_zarr(ds_nodes, nodes_path, compression_level=5)
        write_zarr(ds_links, links_path, compression_level=5)

        ds_nodes = xr.open_dataset(nodes_path, engine="zarr", consolidated=False).load()
        ds_links = xr.open_dataset(links_path, engine="zarr", consolidated=False).load()
    return ds_nodes, ds_links


# =============================================================================
# Helper Functions
# =============================================================================


def compare_zarr_datasets(
    ds_new: xr.Dataset, ds_ref: xr.Dataset, rtol: float = RTOL, atol: float = ATOL
) -> tuple[bool, dict]:
    """
    Compare two xarray Datasets for equivalence.

    Parameters
    ----------
    ds_new : xr.Dataset
        The newly generated dataset
    ds_ref : xr.Dataset
        The reference dataset
    rtol : float
        Relative tolerance for numeric comparisons
    atol : float
        Absolute tolerance for numeric comparisons

    Returns
    -------
    tuple
        (is_equivalent: bool, differences: dict)
    """
    differences = {}

    # Check dimensions match
    if ds_new.sizes != ds_ref.sizes:
        differences["dims"] = {
            "new": set(ds_new.dims),
            "ref": set(ds_ref.dims),
            "missing_in_new": set(ds_ref.dims) - set(ds_new.dims),
            "extra_in_new": set(ds_new.dims) - set(ds_ref.dims),
        }

    # Check coordinates match
    for coord in ds_ref.coords:
        if coord not in ds_new.coords:
            differences[f"missing_coord_{coord}"] = True
            continue

        new_vals = ds_new[coord].values
        ref_vals = ds_ref[coord].values

        # Handle different dtypes
        if new_vals.dtype != ref_vals.dtype:
            # Try to compare as strings if dtypes differ
            try:
                new_str = np.array(new_vals, dtype=str)
                ref_str = np.array(ref_vals, dtype=str)
                if not np.array_equal(new_str, ref_str):
                    differences[f"coord_{coord}"] = {
                        "reason": "values differ (compared as strings)",
                        "new_dtype": str(new_vals.dtype),
                        "ref_dtype": str(ref_vals.dtype),
                    }
            except Exception as e:
                differences[f"coord_{coord}"] = {
                    "reason": f"dtype mismatch and comparison failed: {e}",
                    "new_dtype": str(new_vals.dtype),
                    "ref_dtype": str(ref_vals.dtype),
                }
        elif not np.array_equal(new_vals, ref_vals):
            differences[f"coord_{coord}"] = "values differ"

    # Check data variables
    for var in ds_ref.data_vars:
        if var not in ds_new.data_vars:
            differences[f"missing_var_{var}"] = True
            continue

        new_vals = ds_new[var].values
        ref_vals = ds_ref[var].values

        # Handle numeric vs string comparison
        if np.issubdtype(ref_vals.dtype, np.number):
            # Numeric comparison with tolerance
            # Create mask for valid (non-NaN) values in both arrays
            new_nan = (
                np.isnan(new_vals)
                if np.issubdtype(new_vals.dtype, np.floating)
                else np.zeros_like(new_vals, dtype=bool)
            )
            ref_nan = (
                np.isnan(ref_vals)
                if np.issubdtype(ref_vals.dtype, np.floating)
                else np.zeros_like(ref_vals, dtype=bool)
            )

            # Check NaN positions match
            if not np.array_equal(new_nan, ref_nan):
                differences[f"var_{var}"] = "NaN positions differ"
                continue

            # Compare non-NaN values
            mask = ~ref_nan
            if mask.any():
                if not np.allclose(
                    new_vals[mask], ref_vals[mask], rtol=rtol, atol=atol
                ):
                    max_diff = np.max(np.abs(new_vals[mask] - ref_vals[mask]))
                    differences[f"var_{var}"] = (
                        f"numeric values differ (max diff: {max_diff})"
                    )
        else:
            # String/object comparison
            try:
                new_str = np.array(new_vals, dtype=str)
                ref_str = np.array(ref_vals, dtype=str)
                if not np.array_equal(new_str, ref_str):
                    # Find first difference for debugging
                    diff_mask = new_str != ref_str
                    if diff_mask.any():
                        idx = np.argwhere(diff_mask)[0]
                        differences[f"var_{var}"] = {
                            "reason": "string values differ",
                            "first_diff_idx": idx.tolist(),
                            "new_val": str(new_str[tuple(idx)]),
                            "ref_val": str(ref_str[tuple(idx)]),
                        }
            except Exception as e:
                differences[f"var_{var}"] = f"comparison failed: {e}"

    # Check for extra variables in new dataset
    for var in ds_new.data_vars:
        if var not in ds_ref.data_vars:
            differences[f"extra_var_{var}"] = True

    return len(differences) == 0, differences


def assert_datasets_equivalent(
    ds_new: xr.Dataset, ds_ref: xr.Dataset, name: str = "dataset"
):
    """
    Assert that two datasets are equivalent, with detailed error messages.

    Parameters
    ----------
    ds_new : xr.Dataset
        The newly generated dataset
    ds_ref : xr.Dataset
        The reference dataset
    name : str
        Name for error messages
    """
    is_equivalent, differences = compare_zarr_datasets(ds_new, ds_ref)

    if not is_equivalent:
        diff_summary = "\n".join(f"  - {k}: {v}" for k, v in differences.items())
        pytest.fail(f"{name} differs from reference:\n{diff_summary}")


# =============================================================================
# Output Equivalence Tests
# =============================================================================


class TestOutputEquivalence:
    """Tests to verify refactored output matches reference data."""

    def test_node_timeseries_equivalence(self, parsed_outputs, reference_node_tseries):
        """Verify node time series output matches reference."""
        ds_nodes, _ = parsed_outputs

        assert_datasets_equivalent(ds_nodes, reference_node_tseries, "Node time series")

    def test_link_timeseries_equivalence(self, parsed_outputs, reference_link_tseries):
        """Verify link time series output matches reference."""
        _, ds_links = parsed_outputs

        assert_datasets_equivalent(ds_links, reference_link_tseries, "Link time series")

    def test_numeric_values_not_nan_where_reference_has_values(
        self, parsed_outputs, reference_node_tseries, reference_link_tseries
    ):
        """Ensure we don't introduce NaN values where reference has valid data."""
        ds_nodes, ds_links = parsed_outputs

        # Check node data
        for var in reference_node_tseries.data_vars:
            if var in ds_nodes.data_vars:
                ref_vals = reference_node_tseries[var].values
                new_vals = ds_nodes[var].values

                if np.issubdtype(ref_vals.dtype, np.number):
                    ref_valid = ~np.isnan(ref_vals)
                    new_nan = np.isnan(new_vals)

                    # Check if any reference valid values became NaN
                    invalid_nans = ref_valid & new_nan
                    if invalid_nans.any():
                        count = invalid_nans.sum()
                        pytest.fail(
                            f"Node variable '{var}': {count} values became NaN "
                            f"that were valid in reference"
                        )

        # Check link data
        for var in reference_link_tseries.data_vars:
            if var in ds_links.data_vars:
                ref_vals = reference_link_tseries[var].values
                new_vals = ds_links[var].values

                if np.issubdtype(ref_vals.dtype, np.number):
                    ref_valid = ~np.isnan(ref_vals)
                    new_nan = np.isnan(new_vals)

                    invalid_nans = ref_valid & new_nan
                    if invalid_nans.any():
                        count = invalid_nans.sum()
                        pytest.fail(
                            f"Link variable '{var}': {count} values became NaN "
                            f"that were valid in reference"
                        )


# =============================================================================
# Warning Suppression Tests
# =============================================================================


class TestWarningSuppression:
    """Tests to verify Zarr warnings are properly suppressed."""

    def test_no_unstable_specification_warning(self, tmp_path):
        """Verify no UnstableSpecificationWarning during zarr write."""
        from TRITON_SWMM_toolkit.utils import write_zarr

        # Create a simple dataset with string coordinates
        ds = xr.Dataset(
            data_vars={
                "values": (["node_id", "time"], np.random.rand(3, 5)),
            },
            coords={
                "node_id": ["node_001", "node_002", "node_003"],
                "time": pd.date_range("2020-01-01", periods=5, freq="h"),
            },
        )

        output_path = tmp_path / "test_output.zarr"

        # Capture warnings during write
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            write_zarr(ds, output_path, compression_level=5)

        assert len(caught_warnings) == 0, (
            f"Found {len(caught_warnings)} warning(s): "
            f"{[str(w.message) for w in caught_warnings]}"
        )

    def test_full_pipeline_no_warnings(self, test_case_analysis):
        """Verify full parsing pipeline produces no warnings."""
        # if not REF_INP.exists() or not REF_HYDRAULICS_RPT.exists():
        #     pytest.skip("Reference input files not found")

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")

            wrap_retrieve_SWMM_outputs_as_datasets(test_case_analysis)

        assert len(caught_warnings) == 0, (
            f"Found {len(caught_warnings)} warning(s): "
            f"{[str(w.message) for w in caught_warnings]}"
        )


# =============================================================================
# Function-Level Tests
# =============================================================================


class TestConvertSwmmTdeltasToMinutes:
    """Tests for the convert_swmm_tdeltas_to_minutes function."""

    def test_basic_conversion(self):
        """Test basic time delta conversion."""
        test_input = pd.Series(["0  05:30", "1  12:00", "0  00:15"])
        # Expected: days*1440 + hours*60 + minutes
        # "0  05:30" = 0*1440 + 5*60 + 30 = 330
        # "1  12:00" = 1*1440 + 12*60 + 0 = 1440 + 720 = 2160
        # "0  00:15" = 0*1440 + 0*60 + 15 = 15
        expected = [330.0, 2160.0, 15.0]

        result = convert_swmm_tdeltas_to_minutes(test_input)

        assert len(result) == len(expected)
        for r, e in zip(result, expected):
            assert abs(r - e) < 0.01, f"Expected {e}, got {r}"

    def test_handles_nan_values(self):
        """Test that NaN/None values are handled correctly."""
        test_input = pd.Series(["0  05:30", None, "0  00:15"])

        result = convert_swmm_tdeltas_to_minutes(test_input)

        assert len(result) == 3
        assert result[0] == 330.0
        assert np.isnan(result[1])
        assert result[2] == 15.0

    def test_empty_input(self):
        """Test empty input returns empty list."""
        result = convert_swmm_tdeltas_to_minutes(pd.Series([], dtype=str))
        assert result == []

    def test_multi_day_values(self):
        """Test values spanning multiple days."""
        test_input = pd.Series(["2  00:00", "10  12:30"])
        expected = [2 * 1440, 10 * 1440 + 12 * 60 + 30]

        result = convert_swmm_tdeltas_to_minutes(test_input)

        for r, e in zip(result, expected):
            assert abs(r - e) < 0.01


class TestReturnDataFromRpt:
    """Tests for the return_data_from_rpt function."""

    def test_parses_simple_lines(self):
        """Test parsing of simple whitespace-separated lines."""
        lines = [
            "node_001  0.5  1.2  3.4\n",
            "node_002  0.6  1.3  3.5\n",
        ]

        result = return_data_from_rpt(lines)

        assert len(result) == 2
        # return_data_from_rpt returns a dict, values include newlines
        assert result[0] == ["node_001", "0.5", "1.2", "3.4\n"]
        assert result[1] == ["node_002", "0.6", "1.3", "3.5\n"]

    def test_handles_multiple_spaces(self):
        """Test that multiple spaces between values are handled."""
        lines = [
            "node_001    0.5    1.2    3.4\n",
        ]

        result = return_data_from_rpt(lines)

        assert len(result) == 1
        # return_data_from_rpt returns a dict, values include newlines
        assert result[0] == ["node_001", "0.5", "1.2", "3.4\n"]


# =============================================================================
# Performance Tests
# =============================================================================


class TestPerformance:
    """Tests to ensure no performance regression."""

    @pytest.mark.slow
    def test_retrieve_swmm_outputs_baseline(self, test_case_analysis):
        """Track retrieve_SWMM_outputs_as_datasets baseline timing across phases."""
        start_time = time.time()
        wrap_retrieve_SWMM_outputs_as_datasets(test_case_analysis)
        elapsed = time.time() - start_time

        savings = BASELINE_RETRIEVE_SECONDS - elapsed
        savings_pct = (savings / BASELINE_RETRIEVE_SECONDS) * 100
        print(
            "\nBaseline timing (retrieve_SWMM_outputs_as_datasets): "
            f"{elapsed:.2f}s | Baseline: {BASELINE_RETRIEVE_SECONDS:.2f}s | "
            f"Savings: {savings:.2f}s ({savings_pct:.1f}%)"
        )

        assert elapsed <= BASELINE_RETRIEVE_SECONDS, (
            "retrieve_SWMM_outputs_as_datasets exceeded baseline time: "
            f"{elapsed:.2f}s > {BASELINE_RETRIEVE_SECONDS:.2f}s"
        )

    def test_tdelta_conversion_performance(self):
        """Test that vectorized tdelta conversion is fast."""
        # Create large test input
        n = 10000
        test_input = pd.Series(
            [f"{i % 10}  {(i % 24):02d}:{(i % 60):02d}" for i in range(n)]
        )

        start_time = time.time()
        result = convert_swmm_tdeltas_to_minutes(test_input)
        elapsed = time.time() - start_time

        assert len(result) == n
        # Relaxed timing constraint - current implementation is iterative
        # Will be improved in Phase 1 with vectorization
        assert (
            elapsed < 5.0
        ), f"Conversion of {n} values took {elapsed:.2f}s (should be < 5s)"


# =============================================================================
# Marker Configuration
# =============================================================================


def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
