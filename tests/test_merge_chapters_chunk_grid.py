"""Chunk-grid contract for `utils.merge_chapters_to_unified` (no solver, no fixture).

The merge concatenates chapter stores opened by `xr.open_zarr`, which chunks each
chapter on its STORED grid rather than its extent. A chapter whose stored chunk
length does not divide its timestep count therefore contributes a SHORT TAIL into
the INTERIOR of the concatenation, which zarr forbids -- it permits a short FINAL
chunk only.

These assertions check the ON-DISK GRID, not merely that the write succeeded. That
distinction is the whole point: a remedy that drops the encoding and rechunks the
time axis to a single chunk WRITES SUCCESSFULLY and is a memory hazard at Norfolk
scale, and a success-only assertion passes it.

The production shape below is measured, not invented: two chapters of 132 and 12
timesteps at stored time-chunks 17 and 3, which is what produced the reported
"Dask chunks at position 7 and 8" overlap (index 7 is chapter 0's short tail of 13,
index 8 is chapter 1's leading 3).
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from hhemt.utils import (
    chapter_flag_for,
    chapter_store_for,
    merge_chapters_to_unified,
    unified_flag_for,
)


def _write_chapters(root, specs, ny: int = 8, nx: int = 10) -> None:
    """One chapter store per (n_timesteps, stored_time_chunk) pair, each flagged done."""
    root.mkdir(parents=True, exist_ok=True)
    first = 0
    for index, (n_t, time_chunk) in enumerate(specs):
        ds = xr.Dataset(
            {"wlevel_m": (("timestep_min", "y", "x"), np.zeros((n_t, ny, nx), dtype="float32"))},
            coords={
                "timestep_min": np.arange(first, first + n_t),
                "y": np.arange(ny),
                "x": np.arange(nx),
            },
        )
        ds["wlevel_m"].encoding["chunks"] = (time_chunk, ny, nx)
        # Use the module's OWN naming helpers rather than constructing the names.
        # `chapter_flag_for` emits `chapter_{k:05d}.done`; a hand-built
        # `{store}.done` still MATCHES `completed_chapters`' `chapter_*.done` glob,
        # so it reaches the index parse and dies there on `int("00000.zarr")` --
        # a setup error that never lets an assertion run.
        store = chapter_store_for(root, index)
        ds.to_zarr(store, mode="w", consolidated=False)
        chapter_flag_for(root, index).write_text("ok", encoding="utf-8")
        first += n_t


@pytest.mark.parametrize(
    ("specs", "expected_time_chunk", "total"),
    [
        pytest.param([(132, 17), (12, 3)], 17, 144, id="measured-production-shape"),
        pytest.param([(66, 22), (22, 22)], 22, 88, id="already-aligned-different-length"),
    ],
)
def test_merged_store_keeps_the_inherited_time_chunk_grid(tmp_path, specs, expected_time_chunk, total):
    """The unified store's time axis is uniform at the INHERITED chunk length.

    Two cases on purpose. The first is the shape that fails today. The second is a
    DIFFERENT satisfying state at a different chunk length, so the assertion is
    tested against the invariant rather than against one position that satisfies it.
    """
    chapters = tmp_path / "tseries.zarr.chapters"
    _write_chapters(chapters, specs)
    final = tmp_path / "tseries.zarr"

    merge_chapters_to_unified(chapters, final)

    assert unified_flag_for(final).exists(), "merge did not publish its completion flag"
    merged = xr.open_zarr(final, consolidated=False)
    assert merged.sizes["timestep_min"] == total
    grid = merged["wlevel_m"].chunks[0]
    assert set(grid[:-1]) == {expected_time_chunk}, (
        f"interior time chunks {grid[:-1]} are not uniform at the inherited "
        f"{expected_time_chunk}; a non-uniform interior is what zarr refuses"
    )
    assert len(grid) > 1, (
        f"time axis collapsed to a single chunk of {grid[0]} over {total} steps -- "
        "the write succeeds and the store is a memory hazard, which is exactly the "
        "failure a success-only assertion cannot see"
    )
