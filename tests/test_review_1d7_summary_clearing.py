"""Reviewer test: an event appended to an append-only ``summary`` group is never refused for the length
of its label.

No solver, no Snakemake and no analysis tree: the test calls the group writer's two static helpers on a bare zarr
store, exactly as ``_write_summary_group`` and ``_write_summary_group_by_event`` call them.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from hhemt import processing_analysis as pa
from hhemt.utils import deferred_consolidation

_PP = pa.TRITONSWMM_analysis_post_processing


@pytest.mark.parametrize("first_length", [10, 63, 64, 100, 254])
def test_an_appended_event_identifier_one_character_longer_than_every_stored_one_is_appended_not_refused(
    tmp_path, first_length
):
    """Property: a ``summary`` group first written with event identifiers of ``first_length`` characters accepts an
    appended event whose identifier is ONE character longer, and reads that identifier back whole.

    Class: event identifiers of 65 to 255 characters (a scenario folder is named by the identifier, so 255 is the file
    system's bound). Reachability: the first write sizes the stored string to the longest identifier present or 64,
    whichever is larger, so a longer later identifier needs only an analysis whose identifiers run past 64 characters,
    which the weather indexer columns allow; artifact: ``_prepare_summary_group`` and ``_fit_strings`` in
    ``processing_analysis.py``. Why it is required: an append that refuses on the length of a label refuses a
    legitimate new scenario after earlier consolidations cleared their per-scenario stores, and the only remedy left
    is re-running simulations. Class label: NEW CAPABILITY. The plausible wrong implementation is a stored width fixed
    at the longest value of the first write; the pre-change tree has no append path to run it against. Both-states
    anchor: the assertion turns on whether the appended label is stored whole, not on any message. Invocation not
    isolatable from this tree: discrimination shown instead (a stored-width floor of 255 characters greens the rows
    64, 100 and 254). Killed by: a fixed width equal to the first write's longest value. Passes under a second correct
    implementation that stores identifiers at the file system's bound, or as variable-length strings.
    Must-fire control: the rows 10 and 63 append inside the 64-character floor and pass under the current code, so the
    helper flow itself works and only the length decides the rest.
    """
    store = tmp_path / "tree.zarr"

    def frame(positions, ids):
        return xr.Dataset(
            {"v": ("event_iloc", np.asarray(positions, dtype=float))},
            coords={"event_iloc": list(positions), "event_id": ("event_iloc", ids)},
        )

    first = frame([0, 1], np.array(["a" * first_length] * 2))
    prepared, encoding = _PP._prepare_summary_group(first, 5)
    later = "b" * (first_length + 1)
    xr.Dataset().to_zarr(store, mode="w", consolidated=False)
    with deferred_consolidation(store):
        pa.append_zarr(prepared, store, group="g", consolidate=False, mode="w", encoding=encoding)
        fitted = _PP._fit_strings(store, "g", frame([2], np.array([later])))
        pa.append_zarr(fitted, store, group="g", consolidate=False, mode="a", append_dim="event_iloc")

    stored = xr.open_zarr(store, group="g", consolidated=False, chunks=None).load()
    assert stored["event_id"].values.tolist()[-1] == later
