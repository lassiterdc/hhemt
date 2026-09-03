"""Paired tests for the clear-raw/reclaim division-of-labor specs (S3-S10).

TWO-ARM POSTURE, stated per function rather than claimed once. The trailer test
has a real behavioral differential (today's truncator emits output from which
Total elapsed time cannot be parsed). The capture and arm-list tests fail
pre-fix by ImportError/AttributeError, which proves only that the symbol is
absent -- a weaker signal, and the honest reading of it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_RPT_TAIL = (
    "  Analysis begun on:  Tue Jan 27 20:09:58 2026\n"
    "  Analysis ended on:  Tue Jan 27 20:10:17 2026\n"
    "  Total elapsed time: 00:00:19\n"
)


def _truncate(rpt_path):
    """Drive _truncate_coupled_rpt without a built analysis.

    The method needs only `rpt_path`, `analysis_dir` and `verbose`, and re-stamps DU
    sentinels against `analysis_dir` -- so a tmp dir standing in for the analysis root
    exercises the real code path with no fixture. Bound HERE rather than named as a
    seam for "the applier", which is what round 6 halted on.
    """
    from hhemt.process_simulation import TRITONSWMM_sim_post_processing as _P

    # No shim: _truncate_coupled_rpt is a @staticmethod (it binds no instance state),
    # so the unbound call takes the three real arguments and nothing else.
    return _P._truncate_coupled_rpt(rpt_path, rpt_path.parent, False)


def _write_fixture_rpt(tmp_path: Path, *, finalized: bool = True) -> Path:
    rpt = tmp_path / "full.rpt"
    body = "  Node Time Series Results\n" + "   01/01/2000 00:00:10  0.0\n" * 50
    tail = _RPT_TAIL if finalized else "   01/01/2000 00:09:00  0.0\n"
    rpt.write_text("  Element Count\n  Flow Units ............... CMS\n" + body + tail)
    return rpt


def test_truncated_rpt_still_yields_total_elapsed_time(tmp_path, monkeypatch):
    """PRE-FIX: FAILS -- the one-line trailer drops 'Total elapsed time'."""
    from hhemt.swmm_output_parser import (
        parse_total_elapsed,
        retrieve_swmm_performance_stats_from_rpt,
    )

    rpt = _write_fixture_rpt(tmp_path)
    _truncate(rpt)
    assert parse_total_elapsed(rpt) == 19.0
    assert retrieve_swmm_performance_stats_from_rpt(rpt)["wall_time_s"] == 19.0


def test_truncation_still_refuses_an_unfinalized_rpt(tmp_path):
    """The fail-closed refusal must survive the trailer-block change (S3a)."""
    rpt = _write_fixture_rpt(tmp_path, finalized=False)
    before = rpt.read_text()
    assert _truncate(rpt) is False
    assert rpt.read_text() == before


def test_hydrology_rpt_summary_captures_runoff_volume_and_continuity():
    """S4: the capture must carry both the volume column and the continuity attrs."""
    from hhemt.swmm_output_parser import parse_hydrology_rpt_summary

    src = Path("test_data/norfolk_coastal_flooding/tests/single_sim/sims/event_id.0/swmm/hydro.rpt")
    if not src.exists():
        pytest.skip("real-data hydro.rpt not present in this checkout")
    ds = parse_hydrology_rpt_summary(src)
    assert "total_runoff_10e6_ltr" in ds.data_vars
    assert ds.sizes["subcatchment_id"] > 0
    assert ds.attrs["runoff_continuity_error_perc"] == pytest.approx(-1.234)
    assert ds.attrs["flow_units"] == "cms"


def test_per_node_capture_is_bit_identical_to_the_written_hydrograph(synth_prepared_scenario):
    """S5/S24: the float32 capture must RECONSTRUCT tseries.hyg, not merely match its shape.

    This is the correctness core of the `hydrographs` reclaim class, which
    DELETES strmflow/ on the strength of this capture. The first draft asserted
    only that the column COUNT agreed, which would have passed against a capture
    holding the wrong values -- coverage in a summary line, proof of nothing.

    The reconstruction is exact rather than approximate: tseries.hyg's columns
    are per-GRIDCELL sums of the same per-node series stored here, so summing
    the capture over each (dem_x_coord, dem_y_coord) group reproduces them.
    float32 is lossless w.r.t. SWMM's REAL4, so the only tolerance owed is the
    float64 accumulation order of the sum itself.
    """
    import numpy as np
    import pandas as pd
    import xarray as xr

    scen = synth_prepared_scenario
    cap_path = scen.scen_paths.sim_folder / "processed" / "hydrology_inflow_summary.zarr"
    # chunks=None keeps this eager: a chunked groupby needs explicit `labels`,
    # and the arrays here are one scenario's worth.
    cap = xr.open_zarr(cap_path, chunks=None).squeeze("event_iloc", drop=True)
    assert cap["inflow_cms"].dtype == np.dtype("float32")

    written = pd.read_csv(scen.scen_paths.hyg_timeseries, skiprows=2, header=None)
    # column 0 is time_hr; the rest are gridcells in (x, y)-sorted order, which
    # is the order write_hydrograph_files' groupby produced.
    written_cells = written.iloc[:, 1:].to_numpy(dtype="float64")

    df = pd.DataFrame(
        {
            "x": cap["dem_x_coord"].values,
            "y": cap["dem_y_coord"].values,
            "i": np.arange(cap.sizes["node_id"]),
        }
    )
    inflow = cap["inflow_cms"].values.astype("float64")
    rebuilt = np.column_stack([inflow[g["i"].to_numpy()].sum(axis=0) for _, g in df.groupby(["x", "y"], sort=True)])

    assert (
        rebuilt.shape == written_cells.shape
    ), f"gridcell reconstruction shape {rebuilt.shape} != written {written_cells.shape}"
    np.testing.assert_allclose(rebuilt, written_cells, rtol=0, atol=1e-6)


def test_capture_landed_is_false_for_absent_and_for_unopenable(tmp_path):
    """S25: BOTH negative arms, reachable now that the gate is a staticmethod."""
    from hhemt.process_simulation import TRITONSWMM_sim_post_processing as P

    assert P._capture_landed(None) is False
    assert P._capture_landed(tmp_path / "missing.zarr") is False
    corrupt = tmp_path / "corrupt.zarr"
    corrupt.mkdir()
    assert P._capture_landed(corrupt) is False


def test_peak_flood_depth_arm_groups_carry_both_arms_when_both_enabled():
    """RE-POINTED in round 5: arm resolution landed in _model_arms.groups_for.

    The assertion is unchanged in substance -- a three-model analysis must yield
    BOTH depth arms and never collapse to the coupled one -- but `_ARM_GROUPS` was
    never introduced, because `9e4a1af` shipped the shared module instead.
    """
    from hhemt.report_renderers._model_arms import groups_for

    enabled = ["tritonswmm", "triton", "swmm"]
    assert groups_for("peak_flood_depth", enabled) == [
        "/tritonswmm/triton",
        "/triton_only/triton",
    ]


def test_conduit_flow_arm_groups_carry_both_arms_when_both_enabled():
    """Mirror: conduit flow carries tritonswmm + swmm_only, never triton_only."""
    from hhemt.report_renderers._model_arms import groups_for

    enabled = ["tritonswmm", "triton", "swmm"]
    assert groups_for("conduit_flow", enabled) == [
        "/tritonswmm/swmm_link",
        "/swmm_only/swmm_link",
    ]
