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


class _FakeScenario:
    """The nine attributes `reclaim_scenario_scoped_classes` reads off `scen`.

    Nine, measured by an AST census rather than counted by eye: `event_iloc` plus eight
    `scen_paths` members. The count is stated because an earlier draft said "four", which
    was the number this module's own tests happened to need rather than the number the
    function reads -- and a stand-in that under-declares its contract fails inside the
    function instead of at construction.

    Deliberately NOT the `synth_prepared_scenario` fixture: that family compiles and runs
    a solver, which this module's predicate tests have no need of. Deliberately NOT a bare
    SimpleNamespace either -- the attribute set is the contract, so naming it here makes a
    future signature change fail at construction instead of inside the function.

    `extbc_tseries` is a real Path rather than None, and that is not cosmetic:
    `reclaim_scenario_scoped_classes` builds its prep_inputs tuple containing
    `scen.scen_paths.extbc_tseries.parent`, so `.parent` is dereferenced while the tuple is
    CONSTRUCTED -- before the `if _p is not None` guard one line below ever runs. With None
    there, `classes=("prep_inputs",)` raises AttributeError, which is a trap for the next
    test that reuses this stand-in for a different reclaim class.
    """

    def __init__(self, root, hydro_inp):
        from types import SimpleNamespace

        self.event_iloc = 0
        self.scen_paths = SimpleNamespace(
            swmm_full_rpt_file=None,
            swmm_hydro_inp=hydro_inp,
            sim_folder=root,
            hyg_timeseries=None,
            hyg_locs=None,
            dir_weather_datfiles=None,
            extbc_tseries=Path(root) / "extbc" / "tseries.csv",
            weather_timeseries=None,
        )


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
    """S4: the capture must carry both the volume column and the continuity attrs.

    The input is a TRACKED 85-line excerpt of a real Norfolk hydrology report, beside the
    tracked real hydraulics.rpt that the sibling parser regression test already consumes.
    It replaces an untracked path under test_data/*/tests/, which was the OUTPUT directory
    of a retired test tier: nothing regenerated it, it was absent on every fresh clone, and
    this test therefore skipped on both required CI checks while reading machine-local
    residue by a cwd-relative path.

    Two properties of the excerpt are load-bearing and must survive any re-trim. It keeps
    BOTH continuity blocks with DISTINCT values (-1.234 runoff, 100.000 flow routing),
    because the parser assigns a `Continuity Error (%)` line to runoff purely by whether a
    `Flow Routing Continuity` header has been seen yet -- so an excerpt carrying one block,
    or two with the same value, would pass while proving nothing about that split. And it
    slices data rows from the line AFTER the second dashed rule; starting one line earlier
    loses a row to the section walker's header terminator.
    """
    from importlib.resources import files

    from hhemt.constants import APP_NAME
    from hhemt.swmm_output_parser import parse_hydrology_rpt_summary

    src = (
        files(APP_NAME).parents[1]  # type: ignore[operator]
        / "test_data"
        / "swmm_refactoring_reference"
        / "hydro_runoff_summary.rpt"
    )
    ds = parse_hydrology_rpt_summary(Path(str(src)))
    assert "total_runoff_10e6_ltr" in ds.data_vars
    # Dim-NAME aware, and that is the point: parse_hydrology_rpt_summary's legal empty
    # return skips df.set_index, so the dimension is named `index` and a bare
    # ds.sizes["subcatchment_id"] raises KeyError instead of failing as an assertion.
    assert ds.sizes.get("subcatchment_id", 0) == 20, (
        f"expected the excerpt's 20 subcatchment rows, got dims {dict(ds.sizes)}"
    )
    assert ds.attrs["runoff_continuity_error_perc"] == pytest.approx(-1.234)
    assert ds.attrs["flow_continuity_error_perc"] == pytest.approx(100.0)
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

    assert rebuilt.shape == written_cells.shape, (
        f"gridcell reconstruction shape {rebuilt.shape} != written {written_cells.shape}"
    )
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


def _write_hydro_pair(tmp_path, *, subcatchments, include_summary=True):
    """Write a minimal (hydro.inp, hydro.rpt) pair exercising the capture predicate.

    `subcatchments` is a list of (id, area) pairs written into [SUBCATCHMENTS]; the .rpt's
    Subcatchment Runoff Summary carries one row per NONZERO-area entry, which is what SWMM
    itself emits (its loop skips a row when Subcatch[j].area == 0.0, because that value is
    the divisor of every depth column). `include_summary=False` drops the whole summary
    section, standing in for a report whose section is unrecognizable or truncated.

    The four .rpt metadata lines are mandatory, not decoration: _build_system_results
    RAISES on a missing Flow Units, flow-routing Continuity Error, Flooding Loss or
    Analysis-ended line. Both continuity blocks are present with DISTINCT values so the
    parser's positional runoff-vs-flow split is exercised rather than assumed.
    """
    swmm = tmp_path / "swmm"
    swmm.mkdir(parents=True, exist_ok=True)
    inp = swmm / "hydro.inp"
    rows = "\n".join(f"  {sid}  RG1  N1  {area}  50  100  0.5  0" for sid, area in subcatchments)
    inp.write_text(f"[OPTIONS]\nFLOW_UNITS  CMS\n\n[SUBCATCHMENTS]\n;;Name  Rain Gage  Outlet  Area\n{rows}\n")

    rpt = swmm / "hydro.rpt"
    body = [
        "  Flow Units ............... CMS",
        "",
        "  **************************        Volume         Depth",
        "  Runoff Quantity Continuity     hectare-m            mm",
        "  **************************     ---------       -------",
        "  Continuity Error (%) .....        -1.234",
        "",
        "  **************************        Volume        Volume",
        "  Flow Routing Continuity        hectare-m      10^6 ltr",
        "  **************************     ---------     ---------",
        "  Flooding Loss ............         0.000         0.000",
        "  Continuity Error (%) .....       100.000",
        "",
    ]
    if include_summary:
        rule = "  " + "-" * 126
        body += [
            "  ***************************",
            "  Subcatchment Runoff Summary",
            "  ***************************",
            "",
            rule,
            "                            Total      Total      Total      Total     Imperv"
            "       Perv      Total       Total     Peak  Runoff",
            "                           Precip      Runon       Evap      Infil     Runoff"
            "     Runoff     Runoff      Runoff   Runoff   Coeff",
            "  Subcatchment                 mm         mm         mm         mm         mm"
            "         mm         mm    10^6 ltr      CMS",
            rule,
        ]
        for sid, area in subcatchments:
            if float(area) != 0.0:
                body.append(
                    f"  {sid:<20}      20.00       0.00       0.00       1.75      15.28"
                    "       0.00      15.28        0.02     0.01   0.764"
                )
        body.append("  ")
    body += [
        "",
        "  Analysis begun on:  Tue Jan 27 20:09:58 2026",
        "  Analysis ended on:  Tue Jan 27 20:10:17 2026",
        "  Total elapsed time: 00:00:19",
    ]
    rpt.write_text("\n".join(body))
    return inp, rpt


def test_declared_nonzero_counts_only_nonzero_area_rows(tmp_path):
    """The operand is nonzero-area rows, never bare rows.

    A bare row count over-predicts SWMM's summary on any model carrying a zero-area
    subcatchment, and a legal model would then be declined forever. Measured on the real
    Norfolk pair before this test existed: 869 declared rows / 0 zero-area / 869 summary
    rows, so equality held there and the divergence could not have been observed from any
    artifact in the tree -- which is why it is asserted here on a constructed pair.
    """
    from hhemt.process_simulation import TRITONSWMM_sim_post_processing as P

    inp, _ = _write_hydro_pair(tmp_path, subcatchments=[("S1", "1.0"), ("S2", "0"), ("S3", "2.5")])
    assert P._declared_nonzero_subcatchments(inp) == 2


def test_declared_nonzero_is_none_when_the_discriminator_is_unavailable(tmp_path):
    """UNAVAILABLE is None, never 0 -- absence of evidence must not read as zero declared."""
    from hhemt.process_simulation import TRITONSWMM_sim_post_processing as P

    assert P._declared_nonzero_subcatchments(tmp_path / "nope.inp") is None
    no_section = tmp_path / "bare.inp"
    no_section.write_text("[OPTIONS]\nFLOW_UNITS  CMS\n")
    assert P._declared_nonzero_subcatchments(no_section) is None


def test_capture_is_declined_when_the_summary_section_is_unrecognizable(tmp_path):
    """PRE-FIX: FAILS -- the capture is written empty, accepted, and hydro.rpt deleted."""
    from hhemt.process_simulation import reclaim_scenario_scoped_classes

    inp, rpt = _write_hydro_pair(tmp_path, subcatchments=[("S1", "1.0"), ("S2", "2.0")], include_summary=False)
    scen = _FakeScenario(tmp_path, inp)
    out = reclaim_scenario_scoped_classes(scen, ("standalone_rpt",), tmp_path, verbose=False)
    assert rpt.exists(), "hydro.rpt must be KEPT when its capture cannot be verified"
    assert not (tmp_path / "processed" / "hydrology_rpt_summary.zarr").exists()
    assert out["standalone_rpt"] is False


def test_capture_and_delete_still_proceed_on_a_legal_zero_subcatchment_model(tmp_path):
    """The arm a bare non-emptiness check would break.

    A model declaring only zero-area subcatchments legitimately yields an empty summary,
    so captured == declared_nonzero == 0 and the reclaim MUST still fire. A `captured > 0`
    predicate passes both refusal arms above and fails only here.
    """
    from hhemt.process_simulation import reclaim_scenario_scoped_classes

    inp, rpt = _write_hydro_pair(tmp_path, subcatchments=[("S1", "0"), ("S2", "0")])
    scen = _FakeScenario(tmp_path, inp)
    out = reclaim_scenario_scoped_classes(scen, ("standalone_rpt",), tmp_path, verbose=False)
    assert not rpt.exists(), "a legal zero-subcatchment model must still reclaim its report"
    assert out["standalone_rpt"] is True
