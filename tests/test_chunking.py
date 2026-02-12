"""
Unit tests for processing_analysis._chunk_for_writing

These tests verify the chunk size calculation logic for writing xarray datasets,
including handling of sparse multi-dimensional coordinates (sensitivity analysis).
"""

import os
import signal

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from TRITON_SWMM_toolkit.processing_analysis import (
    TRITONSWMM_analysis_post_processing,
    prev_power_of_two,
    ds_memory_req_MiB,
)


# ========== Helper Functions ==========


def create_mock_analysis():
    """Create a minimal mock analysis object for testing."""

    class MockConfig:
        target_processed_output_type = "zarr"

    class MockAnalysis:
        cfg_analysis = MockConfig()

    return MockAnalysis()


def create_spatial_dataset_2d(n_events=10, nx=512, ny=512, n_vars=1, dtype=np.float64):
    """
    Create a 2D spatial dataset (x, y coordinates).

    Parameters
    ----------
    n_events : int
        Number of events (event_iloc dimension)
    nx, ny : int
        Spatial grid dimensions
    n_vars : int
        Number of data variables
    dtype : numpy dtype
        Data type for variables
    """
    data_vars = {}
    for i in range(n_vars):
        data_vars[f"var_{i}"] = (
            ["event_iloc", "y", "x"],
            np.random.randn(n_events, ny, nx).astype(dtype),
        )

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "event_iloc": np.arange(n_events),
            "x": np.arange(nx, dtype=np.float64),
            "y": np.arange(ny, dtype=np.float64),
        },
    )
    return ds


def create_sensitivity_dataset(
    n_sub_analysis=6,
    n_run_modes=5,
    n_gpus=3,
    n_mpi=2,
    n_omp=2,
    n_events=1,
    nx=551,
    ny=537,
    dtype=np.float64,
):
    """
    Create a sensitivity analysis dataset with multiple configuration dimensions.

    Mirrors the user's problematic dataset dimensions and variable layout while
    keeping arrays lazy (dask) to avoid large memory use in tests.
    """
    spatial_dims = (
        "n_gpus",
        "n_omp_threads",
        "n_mpi_procs",
        "run_mode",
        "sub_analysis_iloc",
        "event_iloc",
        "x",
        "y",
    )
    nonspatial_dims = (
        "n_gpus",
        "n_omp_threads",
        "n_mpi_procs",
        "run_mode",
        "sub_analysis_iloc",
        "event_iloc",
    )
    spatial_shape = (
        n_gpus,
        n_omp,
        n_mpi,
        n_run_modes,
        n_sub_analysis,
        n_events,
        nx,
        ny,
    )
    nonspatial_shape = (n_gpus, n_omp, n_mpi, n_run_modes, n_sub_analysis, n_events)
    spatial_chunks = (1, 1, 1, 1, 1, 1, nx, ny)
    nonspatial_chunks = (1, 1, 1, 1, 1, 1)

    spatial_vars = [
        "wlevel_m_last_tstep",
        "max_velocity_mps",
        "velocity_x_mps_at_time_of_max_velocity",
        "max_wlevel_m",
        "time_of_max_velocity_min",
        "time_of_max_wlevel_min",
        "velocity_y_mps_at_time_of_max_velocity",
    ]
    nonspatial_vars = ["final_surface_flood_volume_cm"]

    data_vars = {
        name: (
            spatial_dims,
            da.zeros(spatial_shape, chunks=spatial_chunks, dtype=dtype),
        )
        for name in spatial_vars
    }
    data_vars.update(  # type: ignore
        {  # type: ignore
            name: (
                nonspatial_dims,
                da.zeros(nonspatial_shape, chunks=nonspatial_chunks, dtype=dtype),
            )
            for name in nonspatial_vars
        }
    )

    run_mode_labels = np.array(["gpu", "cpu", "seq", "mpi", "hybrid"])

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "sub_analysis_iloc": np.arange(n_sub_analysis),
            "run_mode": run_mode_labels,
            "n_gpus": np.arange(n_gpus),
            "n_mpi_procs": np.arange(n_mpi),
            "n_omp_threads": np.arange(1, n_omp + 1),
            "event_iloc": np.arange(n_events),
            "x": np.linspace(3.697e6, 3.698e6, nx, dtype=np.float64),
            "y": np.linspace(1.06e6, 1.061e6, ny, dtype=np.float64),
        },
    )
    return ds


# ========== Timeout Fixture ==========


class _TestTimeoutError(RuntimeError):
    """Raised when a test exceeds the allowed runtime."""


@pytest.fixture(autouse=True)
def enforce_test_timeout():
    """Fail any test that exceeds 5 seconds."""
    if os.name != "posix":
        yield
        return

    def _handle_timeout(_signum, _frame):
        raise _TestTimeoutError("Test exceeded 5 second timeout")

    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.alarm(5)
    try:
        yield
    finally:
        signal.alarm(0)


# ========== Tests for Helper Functions ==========


def test_prev_power_of_two():
    """Test prev_power_of_two function."""
    assert prev_power_of_two(1) == 1
    assert prev_power_of_two(2) == 2
    assert prev_power_of_two(3) == 2
    assert prev_power_of_two(4) == 4
    assert prev_power_of_two(100) == 64
    assert prev_power_of_two(1000) == 512


def test_ds_memory_req_MiB():
    """Test ds_memory_req_MiB function."""
    ds = create_spatial_dataset_2d(n_events=10, nx=100, ny=100, n_vars=1)
    mem_MiB = ds_memory_req_MiB(ds)
    assert mem_MiB > 0


# ========== Tests for _chunk_for_writing ==========


@pytest.fixture
def proc():
    """Create a processing object for testing."""
    mock_analysis = create_mock_analysis()
    return TRITONSWMM_analysis_post_processing(mock_analysis)  # type: ignore


class TestChunkMemoryConstraint:
    """Tests that chunks actually respect memory constraints."""

    def test_chunk_memory_under_limit_simple(self, proc):
        """Test that chunks stay under memory limit for simple dataset."""
        ds = create_spatial_dataset_2d(n_events=10, nx=256, ny=256, n_vars=1)

        max_mem = 50  # 50 MiB limit
        chunks = proc._chunk_for_writing(
            ds, spatial_coords=["x", "y"], max_mem_usage_MiB=max_mem
        )

        # Test actual memory usage
        test_slice = {coord: slice(0, chunks[coord]) for coord in chunks}
        chunk_mem = ds_memory_req_MiB(ds.isel(test_slice))

        # Should be within tolerance
        assert chunk_mem <= max_mem

    def test_chunk_memory_sensitivity_dataset(self, proc):
        """Test chunking with realistic sensitivity analysis dataset."""
        # This mirrors the user's failing case
        ds = create_sensitivity_dataset()

        max_mem = 200  # 200 MiB limit
        chunks = proc._chunk_for_writing(
            ds, spatial_coords=["x", "y"], max_mem_usage_MiB=max_mem
        )

        # Test actual memory usage
        test_slice = {coord: slice(0, chunks[coord]) for coord in chunks}
        chunk_mem = ds_memory_req_MiB(ds.isel(test_slice))

        print(f"Chunks: {chunks}")
        print(f"Chunk memory: {chunk_mem:.1f} MiB (limit: {max_mem} MiB)")

        # This is the key test - should not exceed limit
        assert chunk_mem <= max_mem


class TestChunkForWriting2DSpatial:
    """Tests for 2D spatial datasets (x, y)."""

    def test_basic_2d_chunking(self, proc):
        """Test basic chunking for 2D spatial dataset."""
        ds = create_spatial_dataset_2d(n_events=10, nx=512, ny=512)

        chunks = proc._chunk_for_writing(ds, spatial_coords=["x", "y"])

        assert isinstance(chunks, dict)
        assert "x" in chunks
        assert "y" in chunks
        assert "event_iloc" in chunks

    def test_none_spatial_coords(self, proc):
        """Test with spatial_coords=None (returns 'auto')."""
        ds = create_spatial_dataset_2d(n_events=10, nx=100, ny=100)
        chunks = proc._chunk_for_writing(ds, spatial_coords=None)
        assert chunks == "auto"

    def test_missing_spatial_coord(self, proc):
        """Test error when spatial coord doesn't exist in dataset."""
        ds = create_spatial_dataset_2d(n_events=10, nx=100, ny=100)

        with pytest.raises(ValueError, match="not found in dataset"):
            proc._chunk_for_writing(ds, spatial_coords=["x", "y", "z"])


class TestChunkingEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_power_of_two_chunking(self, proc):
        """Verify non-spatial chunks use power-of-2 when subdividing dimensions."""
        # Use large event count to force chunking
        ds = create_spatial_dataset_2d(n_events=1000, nx=256, ny=256, n_vars=1)

        chunks = proc._chunk_for_writing(
            ds, spatial_coords=["x", "y"], max_mem_usage_MiB=100
        )

        # When a dimension is chunked (not using full length), it should be power of 2
        event_chunk = chunks["event_iloc"]
        event_len = len(ds["event_iloc"])

        if event_chunk < event_len:
            # If chunked, should be a power of 2
            assert event_chunk > 0 and (event_chunk & (event_chunk - 1)) == 0
        else:
            # If not chunked, uses full dimension (which is fine)
            assert event_chunk == event_len

    def test_mixed_dtype_dataset(self, proc):
        """Test chunking with mixed dtypes (float32 and float64)."""
        # Create dataset with different dtypes
        nx, ny = 256, 256
        n_events = 10

        ds = xr.Dataset(
            data_vars={
                "var_float32": (
                    ["event_iloc", "y", "x"],
                    np.random.randn(n_events, ny, nx).astype(np.float32),
                ),
                "var_float64": (
                    ["event_iloc", "y", "x"],
                    np.random.randn(n_events, ny, nx).astype(np.float64),
                ),
            },
            coords={
                "event_iloc": np.arange(n_events),
                "x": np.arange(nx, dtype=np.float64),
                "y": np.arange(ny, dtype=np.float64),
            },
        )

        max_mem = 100
        chunks = proc._chunk_for_writing(
            ds, spatial_coords=["x", "y"], max_mem_usage_MiB=max_mem
        )

        # Verify chunks respect memory limit
        test_slice = {coord: slice(0, chunks[coord]) for coord in chunks}
        chunk_mem = ds_memory_req_MiB(ds.isel(test_slice))
        assert chunk_mem <= max_mem

    def test_very_small_memory_limit(self, proc):
        """Test that algorithm handles very small memory limits gracefully."""
        ds = create_spatial_dataset_2d(n_events=10, nx=512, ny=512, n_vars=3)

        # Set unreasonably small memory limit (should still produce valid chunks)
        max_mem = 10  # 10 MiB
        chunks = proc._chunk_for_writing(
            ds, spatial_coords=["x", "y"], max_mem_usage_MiB=max_mem
        )

        # Verify chunks still respect the limit
        test_slice = {coord: slice(0, chunks[coord]) for coord in chunks}
        chunk_mem = ds_memory_req_MiB(ds.isel(test_slice))
        assert chunk_mem <= max_mem

        # All chunk sizes should be at least 1
        for coord, size in chunks.items():
            assert size >= 1

    def test_spatial_chunk_size_target(self, proc):
        """Test that spatial chunks respect the spatial_coord_size parameter."""
        ds = create_spatial_dataset_2d(n_events=5, nx=1000, ny=1000, n_vars=1)

        # Request smaller spatial chunks
        spatial_target = 16384  # 128x128 instead of default 256x256
        chunks = proc._chunk_for_writing(
            ds,
            spatial_coords=["x", "y"],
            spatial_coord_size=spatial_target,
            max_mem_usage_MiB=200,
        )

        # Verify spatial chunk size is approximately the target
        spatial_chunk_points = chunks["x"] * chunks["y"]
        assert spatial_chunk_points <= spatial_target
        # Should be close to target (within factor of 2 due to rounding)
        assert spatial_chunk_points >= spatial_target // 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
