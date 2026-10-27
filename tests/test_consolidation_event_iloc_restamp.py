"""Consolidation must not trust a per-scenario store's stamped event_iloc.

event_iloc is a POSITION in the analysis's current event list, not an identity. After a
multi-sim rerun that removes one event and adds another, a scenario that was not
reprocessed keeps the position it was stamped with, while consolidation iterates the
CURRENT positions. Concatenating on the embedded value then publishes a duplicated
coordinate -- event_iloc [0, 2, 2] was measured in a consolidated tree on an HPC run --
so both loaders in ``processing_analysis`` re-stamp each loaded store from its loop
position before the concat.

Mock-only: the scenario class is replaced at its module attribute, so no solver, no
compile and no source clone is involved.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import xarray as xr

import hhemt.processing_analysis as pa

# Position -> the event_iloc each store was stamped with. Position 1 holds a store that
# was not reprocessed after the removal and still carries 2; position 2 holds the added
# event's fresh store, also stamped 2. This is the add/remove arrangement that published
# [0, 2, 2].
_STALE = {0: 0, 1: 2, 2: 2}


def _write_store(path, stale_iloc, *, timeseries):
    if timeseries:
        ds = xr.Dataset(
            {"flow_cms": (("event_iloc", "time", "node"), np.ones((1, 2, 2)))},
            coords={"event_iloc": [stale_iloc], "time": [0.0, 60.0], "node": ["J1", "J2"]},
        )
    else:
        ds = xr.Dataset(
            {"max_wlevel_m": (("event_iloc", "y", "x"), np.ones((1, 2, 2)))},
            coords={"event_iloc": [stale_iloc], "x": [0.0, 1.0], "y": [0.0, 1.0]},
        )
    ds.to_zarr(path, consolidated=False)


def _post_processor(tmp_path, monkeypatch, *, attr, timeseries):
    """A post-processor over three positions whose stores carry ``_STALE``'s stamps."""
    paths = {ei: tmp_path / f"store_{ei}.zarr" for ei in _STALE}
    for ei, stale in _STALE.items():
        _write_store(paths[ei], stale, timeseries=timeseries)

    class _Scen:
        def __init__(self, ei, _analysis):
            self.scen_paths = SimpleNamespace(**{attr: paths[ei]})

        def get_log(self, _model_type):
            """Report every store as successfully written, so the summary loader's gate admits it."""
            ok = SimpleNamespace(success=True)
            return SimpleNamespace(processing_log=SimpleNamespace(outputs={p.name: ok for p in paths.values()}))

    monkeypatch.setattr(pa, "TRITONSWMM_scenario", _Scen)
    analysis = Mock()
    analysis.df_sims = pd.DataFrame(index=pd.RangeIndex(len(_STALE)))
    analysis.cfg_analysis.target_processed_output_type = "zarr"
    analysis._retrieve_weather_indexer_using_integer_index = lambda ei: {"event_index": ei}
    return pa.TRITONSWMM_analysis_post_processing(analysis)


def test_the_summary_loader_restamps_event_iloc_from_the_loop_position(tmp_path, monkeypatch):
    """Property: ``_retrieve_combined_output`` publishes event_iloc equal to the current positions.

    Class: every set of per-scenario summaries whose stamped event_iloc disagrees with the
    current position of the scenario that produced it -- here a stale 2 at position 1.

    Why required: the consolidated tree is keyed on event_iloc, so a duplicated value makes
    ``.sel(event_iloc=2)`` return two events and leaves position 1 unaddressable.

    What kills it: concatenating the stores as opened, on their embedded event_iloc.

    A second correct implementation under which it still passes: one that drops each store's
    event_iloc and assigns ``range(len(df_sims))`` once after the concat; the assertion
    reads only the published coordinate.
    """
    attr = pa.TRITONSWMM_analysis_post_processing._MODE_CONFIG["tritonswmm_triton"][0]
    pp = _post_processor(tmp_path, monkeypatch, attr=attr, timeseries=False)
    out = pp._retrieve_combined_output("tritonswmm_triton")
    assert out["event_iloc"].values.tolist() == [0, 1, 2]


def test_the_timeseries_loader_restamps_event_iloc_from_the_loop_position(tmp_path, monkeypatch):
    """Property: ``_retrieve_combined_timeseries`` publishes event_iloc equal to the current positions.

    Class: the same disagreement as the summary case, on a timeseries store -- a store whose
    own event_iloc coordinate is stale.

    Why required: the two loaders feed sibling nodes of one consolidated tree, and a
    timeseries node keyed differently from its summary node would pair the wrong events.

    What kills it: appending the timeseries stores as opened.

    A second correct implementation under which it still passes: one that assigns the
    positions once after the concat; the assertion reads only the published coordinate.

    Not established: that a production timeseries store carries event_iloc. The SWMM
    timeseries writer emits none today, so on a production store this re-stamp adds an
    index equal to the positions the concat already produced; this test reaches the
    defect only through a store built to carry a stale value.
    """
    ts_mode = "tritonswmm_swmm_node_ts"
    attr = pa.TRITONSWMM_analysis_post_processing._TIMESERIES_MODE_CONFIG[ts_mode]
    pp = _post_processor(tmp_path, monkeypatch, attr=attr, timeseries=True)
    out = pp._retrieve_combined_timeseries(ts_mode)
    assert out["event_iloc"].values.tolist() == [0, 1, 2]


def test_a_store_already_at_its_position_is_published_unchanged(tmp_path, monkeypatch):
    """Property: when every stamp equals its position, the published coordinate is that position.

    Class: the differently-positioned satisfying input for the two tests above -- no stale
    stamp anywhere, which is every analysis that has not had events added or removed.

    Why required: the re-stamp must not perturb the common case it was not written for.

    What kills it: a re-stamp that writes anything other than the loop position (an
    off-by-one, or the row's label from a non-RangeIndex).

    A second correct implementation under which it still passes: not re-stamping a store
    whose stamp already equals its position; the assertion reads only the result.
    """
    attr = pa.TRITONSWMM_analysis_post_processing._MODE_CONFIG["tritonswmm_triton"][0]
    monkeypatch.setitem(_STALE, 1, 1)
    pp = _post_processor(tmp_path, monkeypatch, attr=attr, timeseries=False)
    out = pp._retrieve_combined_output("tritonswmm_triton")
    assert out["event_iloc"].values.tolist() == [0, 1, 2]
