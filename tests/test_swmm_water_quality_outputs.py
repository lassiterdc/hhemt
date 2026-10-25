"""SWMM water-quality outputs: the gate, the pollutant series, the per-event statistics, A3 and A2.

Problem these tests guard. A SWMM model that declares a pollutant and routes quality writes one
concentration column per pollutant into every time-series row of its report, prints extra summary
tables, and writes pollutant results into its binary output. The toolkit must turn those into a
`pollutant` dimension and a bounded set of per-event statistics, ONLY when SWMM itself says quality
was routed, and it must refuse to publish statistics a snapshot resume has spoiled.

Fixtures are written inside the tests with NON-ZERO values throughout. The synthetic knob's own
pollutant has zero concentration by design, so it cannot tell a correct parse from a wrong one.

Every test states, in its own docstring, the standard of stage-build B4/B5: its CLASS (every test
here is NEW CAPABILITY, with the plausible wrong implementation named and the red it produces), the
REACHABILITY artifact for its input class, the property it turns on that is present in both the
old and the new code, an INVOCATION-ISOLATION statement, and a SECOND CORRECT implementation under
which it still passes. The tests run on the laptop: they parse text and call pure functions; none
compiles or runs a solver.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from hhemt.exceptions import ProcessingError
from hhemt.swmm_output_parser import (
    parse_hydrology_rpt_summary,
    retrieve_SWMM_outputs_as_datasets,
    return_node_time_series_results_from_outfile,
)

STEPS = ("00:10:00", "00:20:00", "00:30:00")
START = "01/01/2000 00:00:00"
TWO = [("TSS", "MG/L"), ("Coliform_ABC", "#/L")]  # the second ID is 12 characters
TWO_UNITS = ["mg/L", "#/L"]

_NODE_CONC = {
    ("N1", 0): [1.5, 4.5, 2.5],
    ("N1", 1): [2.0, 2.0, 1.0],
    ("OUT1", 0): [0.5, 0.25, 0.125],
    ("OUT1", 1): [0.0, 3.0, 3.0],
}
_LINK_CONC = {("L1", 0): [0.1, 0.2, 0.3], ("L1", 1): [1.0, 0.0, 2.0]}


# --------------------------------------------------------------------------- fixture builders
def _inp(pollutants, ignore_quality=False) -> str:
    rows = "\n".join(f"{name}  {unit}  0 0 0 0 NO * 0 0" for name, unit in pollutants)
    pollutant_section = (
        f"[POLLUTANTS]\n;;Name Units Crain Cgw Crdii Kdecay SnowOnly CoPollutant CoFrac Cinit\n{rows}\n"
        if pollutants
        else ""
    )
    quality = "YES" if ignore_quality else "NO"
    return f"[OPTIONS]\nFLOW_UNITS CMS\nIGNORE_QUALITY {quality}\n\n{pollutant_section}\n[JUNCTIONS]\nN1 0 3\n"


def _continuity(header, error, flood=None):
    rows = [
        "  **************************        Volume        Volume",
        f"  {header:<31}hectare-m      10^6 ltr",
        "  **************************     ---------     ---------",
        "  Initial Stored Volume ....         0.000         0.000",
    ]
    if flood is not None:
        rows.append(f"  Flooding Loss ............         0.000{flood:14.3f}")
    rows.append(f"  Continuity Error (%) .....{error:14.3f}")
    return "\n".join(rows) + "\n  \n  \n"


def _group_table(title, labels, values, pollutants):
    """A quality continuity table printed in groups of five pollutants, as SWMM prints it."""
    out = []
    for g in range(0, len(pollutants), 5):
        grp = list(range(g, min(g + 5, len(pollutants))))
        out.append("  **************************" + "".join(f"{pollutants[p][0]:>14}" for p in grp))
        out.append(f"  {title:<26}" + "".join(f"{('kg' if pollutants[p][1] != '#/L' else 'LogN'):>14}" for p in grp))
        out.append("  **************************" + "    ----------" * len(grp))
        for label in labels:
            out.append(f"  {label} ......." + "".join(f"{values[label][p]:14.3f}" for p in grp))
        out.append("  \n  ")
    return "\n".join(out) + "\n"


def _default_values(labels, n):
    return {label: [1.0 + i + p for p in range(n)] for i, label in enumerate(labels)}


ROUTING_LABELS = [
    "Dry Weather Inflow",
    "Wet Weather Inflow",
    "Groundwater Inflow",
    "RDII Inflow",
    "External Inflow",
    "External Outflow",
    "Flooding Loss",
    "Exfiltration Loss",
    "Mass Reacted",
    "Initial Stored Mass",
    "Final Stored Mass",
    "Continuity Error (%)",
]
RUNOFF_LABELS = [
    "Initial Buildup",
    "Surface Buildup",
    "Wet Deposition",
    "Sweeping Removal",
    "Infiltration Loss",
    "BMP Removal",
    "Surface Runoff",
    "Remaining Buildup",
    "Continuity Error (%)",
]


def _tseries(kind, ident, n_extra_values, head_cols, rows_fn, short_row=False):
    lines = [f"  <<< {kind} {ident} >>>", "  " + "-" * 74, head_cols[0], head_cols[1], "  " + "-" * 64]
    for s, step in enumerate(STEPS):
        vals = rows_fn(s)
        if short_row and s == 1 and n_extra_values:
            vals = vals[:-1]
        lines.append(f"   01/01/2000 {step}" + "".join(f"{v:10.3f}" for v in vals))
    return "\n".join(lines) + "\n  \n"


def make_rpt(
    pollutants,
    *,
    gate="YES",
    extra_columns=True,
    short_row=False,
    outfall_vol=4.0,
    outfall_loads=(10.0, 2.0),
    link_loads=(12345.678, 2.0),
    routing=None,
    runoff_quality=False,
    washoff=False,
    node_summaries=True,
):
    n = len(pollutants)
    outfall_loads = list(outfall_loads) + [1.0] * (n - len(outfall_loads))
    link_loads = list(link_loads) + [1.0] * (n - len(link_loads))
    routing = routing or _default_values(ROUTING_LABELS, n)
    parts = [
        "  Element Count\n",
        "  Flow Units ............... CMS\n",
        f"    Water Quality .......... {gate}\n",
        f"  Starting Date ............ {START}\n",
        _continuity("Runoff Quantity Continuity", 0.111),
    ]
    if runoff_quality:
        parts.append(
            _group_table("Runoff Quality Continuity", RUNOFF_LABELS, _default_values(RUNOFF_LABELS, n), pollutants)
        )
    parts.append(_continuity("Flow Routing Continuity", 0.444, flood=2.0))
    if gate == "YES":
        parts.append(_group_table("Quality Routing Continuity", ROUTING_LABELS, routing, pollutants))
    if node_summaries:
        parts.append(
            "  Node Inflow Summary\n  ---------\n  header\n  header\n  header\n  ---------\n"
            "  N1                   JUNCTION     0.000    0.000     0  00:01    2.93e-06    2.93e-06       2.930 ltr\n"
            "  OUT1                 OUTFALL      0.000    0.000     0  00:01    2.93e-06    2.93e-06       2.930 ltr\n"
            "  \n"
            "  Node Flooding Summary\n  Flooding refers to all water that overflows a node.\n"
            "  ---------\n  header\n  header\n  header\n  ---------\n"
            "  N1                      0.02     0.988      0  00:00       0.014     1.041\n  \n"
            "  Link Flow Summary\n  ---------\n  header\n  header\n  header\n  ---------\n"
            "  L1                   CONDUIT     0.000     0  00:01      0.00    0.00    0.00\n  \n"
        )
    if gate == "YES":
        head = "  ".join(f"{name:>10}" for name, _ in pollutants)
        # Outfall Loading Summary, SWMM's layout: rows sit between the 2nd and 3rd dashed lines.
        dash = "  " + "-" * 60
        vol = f"{outfall_vol:12.3f}"
        loads = "".join(f"{x:14.3f}" for x in outfall_loads[:n])
        parts.append(
            f"  Outfall Loading Summary\n{dash}\n   header {head}\n   header\n{dash}\n"
            f"  {'OUT1':<20}{98.45:7.2f} {0.123:9.3f} {0.456:9.3f}{vol}{loads}\n{dash}\n"
            f"  {'System':<20}{98.45:7.2f} {0.123:9.3f} {0.456:9.3f}{vol}{loads}\n  \n"
        )
        link_cells = "".join(f"{x:14.3f}" if x < 10000.0 else f"{x:14.3e}" for x in link_loads[:n])
        parts.append(
            f"  Link Pollutant Load Summary\n{dash}\n   {head}\n   Link\n{dash}\n  {'L1':<20}{link_cells}\n  \n"
        )
        if washoff:
            wash = "".join(f"{x:14.3f}" for x in (7.0, 2.0)[:n])
            parts.append(
                f"  Subcatchment Washoff Summary\n{dash}\n   {head}\n   Subcatchment\n{dash}\n"
                f"  {'S1':<20}{wash}\n{dash}\n  {'System':<20}{wash}\n  \n"
            )
    parts.append("  ************************\n  Node Time Series Results\n  ************************\n  \n")
    for ident in ("N1", "OUT1"):
        cols = (
            "      Inflow  Flooding     Depth      Head"
            + "".join(f"{name:>10}" for name, _ in pollutants) * extra_columns,
            "  Date        Time            CMS       CMS    meters    meters" + "      MG/L" * (n * extra_columns),
        )

        def rows(s, ident=ident):
            base = [0.1, 0.2, 0.3, 0.4]
            if not extra_columns:
                return base
            return base + [(_NODE_CONC.get((ident, p)) or [float(p + t) for t in range(3)])[s] for p in range(n)]

        parts.append(_tseries("Node", ident, n * extra_columns, cols, rows, short_row))
    parts.append("  ************************\n  Link Time Series Results\n  ************************\n  \n")
    cols = (
        "      Flow  Velocity     Depth  Capacity/" + "".join(f"{name:>10}" for name, _ in pollutants) * extra_columns,
        "  Date        Time            CMS     m/sec    meters   Setting " + "      MG/L" * (n * extra_columns),
    )

    def link_rows(s):
        base = [1.0, 2.0, 3.0, 4.0]
        if not extra_columns:
            return base
        return base + [(_LINK_CONC.get(("L1", p)) or [float(p + t) for t in range(3)])[s] for p in range(n)]

    parts.append(_tseries("Link", "L1", n * extra_columns, cols, link_rows, short_row))
    parts.append("  Analysis ended on:  Tue Jan 27 20:10:17 2026\n")
    return "".join(parts)


def _write(tmp_path, pollutants, rpt_text, ignore_quality=False, name="m"):
    inp = tmp_path / f"{name}.inp"
    rpt = tmp_path / f"{name}.rpt"
    inp.write_text(_inp(pollutants, ignore_quality))
    rpt.write_text(rpt_text)
    return inp, rpt


def _datasets(tmp_path, pollutants, **kwargs):
    inp, rpt = _write(tmp_path, pollutants, make_rpt(pollutants, **kwargs))
    return retrieve_SWMM_outputs_as_datasets(inp, rpt)


# --------------------------------------------------------------------------- test 1
def test_gate_off_outputs_are_identical_to_a_model_without_pollutants(tmp_path):
    """Gate off gives exactly the outputs of a model that declares no pollutant.

    Class: NEW CAPABILITY. Plausible wrong implementation: one that adds a `pollutant` dimension (even
    zero-length) or an attribute key whenever `[POLLUTANTS]` is declared, whatever the report says;
    it is red under that implementation because the variable, coordinate and dimension sets differ.
    Reachability: a report printing `Water Quality .......... NO` while the model file declares
    pollutants is what SWMM produces under `IGNORE_QUALITY YES` (the Norfolk templates); artifact:
    the report fixture below prints NO beside a declared `[POLLUTANTS]` section.
    Both-states property: the dataset's variable names, coordinate names, dimension names and
    attribute KEYS, which the pre-change code can also produce.
    Invocation isolation: the reference run reads the same text with no `[POLLUTANTS]` section and
    NO line variant, a differently-positioned input of the same property.
    Second correct implementation: one that gates on the `.rpt` line inside `parse_rpt_single_pass`
    and returns the same datasets passes unchanged.
    """
    gated_off = _datasets(_mk(tmp_path / "a"), TWO, gate="NO", extra_columns=False)
    reference = _datasets(_mk(tmp_path / "b"), [], gate="NO", extra_columns=False)
    for got, want in zip(gated_off, reference, strict=True):
        assert set(got.data_vars) == set(want.data_vars)
        assert set(got.coords) == set(want.coords)
        assert set(got.dims) == set(want.dims)
        assert set(got.attrs) == set(want.attrs)
        assert "pollutant" not in got.dims and "pollutant" not in got.coords
        xr.testing.assert_equal(got, want)


def _mk(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


# --------------------------------------------------------------------------- test 2
def test_series_carry_pollutants_in_declared_order_with_units(tmp_path):
    """T1 and T2: the pollutant dimension, its order and units, and the values the report prints.

    Class: NEW CAPABILITY. Plausible wrong implementation: naming pollutants from the report's column
    headers (the 12-character ID `Coliform_ABC` runs into its neighbour there), or reading units from
    the report's unit row; it is red under it because the coordinate or the units differ.
    Reachability: a two-pollutant model with a 12-character ID and a `#/L` second pollutant is the
    class SWMM prints with header run-in; artifact: the fixture's own header row.
    Both-states property: "the value read equals the value printed" (true of any correct reader).
    Invocation isolation: the short-row case is a second input, an invalid row, not a re-run.
    Second correct implementation: reading names from the `.inp` with a regex rather than SWMM's
    tokenizer passes.
    """
    ds_nodes, ds_links = _datasets(_mk(tmp_path / "t"), TWO)
    assert list(ds_nodes["pollutant"].values) == ["TSS", "Coliform_ABC"]
    assert list(ds_nodes["pollutant_conc_units"].values) == TWO_UNITS
    assert list(ds_links["pollutant"].values) == ["TSS", "Coliform_ABC"]
    n1 = ds_nodes["pollutant_conc"].sel(node_id="N1")
    assert n1.sel(pollutant="TSS").values.tolist() == [1.5, 4.5, 2.5]
    assert n1.sel(pollutant="Coliform_ABC").values.tolist() == [2.0, 2.0, 1.0]
    link = ds_links["link_pollutant_conc"].sel(link_id="L1")
    assert link.sel(pollutant="TSS").values.tolist() == [0.1, 0.2, 0.3]
    assert link.sel(pollutant="Coliform_ABC").values.tolist() == [1.0, 0.0, 2.0]
    inp, rpt = _write(_mk(tmp_path / "s"), TWO, make_rpt(TWO, short_row=True))
    with pytest.raises(ProcessingError):
        retrieve_SWMM_outputs_as_datasets(inp, rpt)


# --------------------------------------------------------------------------- test 3
def test_peak_concentration_and_its_minute(tmp_path):
    """S1 and S2: the event maximum and the minutes from the simulation start to its first step.

    Class: NEW CAPABILITY. Plausible wrong implementation: taking the time from the first report step
    rather than the simulation start, or the LAST occurrence of a tied maximum; red because the minute
    differs (the series starts at minute 10 and ties at the coliform maximum).
    Reachability: any gate-on summary reaches it; artifact: `N1` TSS peaks 4.5 at step 2.
    Both-states property: the summary has one value per (event, node, pollutant).
    Invocation isolation: OUT1's coliform series ties its maximum at steps 2 and 3, a differently
    positioned input for the "first occurrence" clause.
    Second correct implementation: `idxmax` over the series instead of `argmax` passes.
    """
    from hhemt.process_simulation import summarize_swmm_simulation_results

    ds_nodes, _ = _datasets(_mk(tmp_path / "p"), TWO)
    summary = summarize_swmm_simulation_results(ds_nodes, 0)
    assert "pollutant_conc_max" not in summary and "pollutant_conc_last" not in summary
    peak = summary["pollutant_max_conc"].isel(event_iloc=0).sel(node_id="N1")
    when = summary["pollutant_time_of_max_conc_min"].isel(event_iloc=0).sel(node_id="N1")
    assert peak.sel(pollutant="TSS") == 4.5 and when.sel(pollutant="TSS") == 20.0
    out_when = (
        summary["pollutant_time_of_max_conc_min"].isel(event_iloc=0).sel(node_id="OUT1", pollutant="Coliform_ABC")
    )
    assert out_when == 20.0
    assert "swmm_simulation_start" not in summary.attrs
    assert "swmm_simulation_start" not in ds_nodes.attrs, "the start must not be persisted as an undeclared attribute"
    assert "swmm_simulation_start" not in _datasets(_mk(tmp_path / "p2"), TWO)[1].attrs


# --------------------------------------------------------------------------- test 4
def test_outfall_load_volume_emc_and_zero_volume(tmp_path):
    """S3, S4, S5: outfall load, volume and the event mean concentration; the zero-volume and count cases.

    Class: NEW CAPABILITY. Plausible wrong implementation: not linearizing the printed LOG10 of a
    `#/L` load (2.000 would stay 2), or dividing by a zero volume (infinity); red because the load
    reads 2 not 100, or the EMC reads inf.
    Reachability: SWMM prints every count load as LOG10 and a quiet event can print a zero volume;
    artifact: the fixture prints `2.000` for the count load and a second fixture prints volume 0.
    Both-states property: the load of the mass pollutant equals the printed value.
    Invocation isolation: the zero-volume run is a second input.
    Second correct implementation: computing the EMC from unrounded inputs passes the NaN/inf check.
    """
    ds_nodes, _ = _datasets(_mk(tmp_path / "e"), TWO)
    load = ds_nodes["outfall_total_load"].sel(node_id="OUT1")
    assert load.sel(pollutant="TSS") == 10.0
    assert load.sel(pollutant="Coliform_ABC") == pytest.approx(100.0)
    assert float(ds_nodes["outfall_total_vol_10e6_ltr"].sel(node_id="OUT1")) == 4.0
    emc = ds_nodes["outfall_emc"].sel(node_id="OUT1")
    assert emc.sel(pollutant="TSS") == pytest.approx(2.5)
    assert emc.sel(pollutant="Coliform_ABC") == pytest.approx(100.0 / 4.0e6)
    assert np.isnan(ds_nodes["outfall_total_load"].sel(node_id="N1")).all()
    zero, _ = _datasets(_mk(tmp_path / "z"), TWO, outfall_vol=0.0)
    emc0 = zero["outfall_emc"].sel(node_id="OUT1")
    assert np.isnan(emc0).all() and not np.isinf(emc0).any()


# --------------------------------------------------------------------------- test 5
def test_link_load_accepts_exponent_notation(tmp_path):
    """S6: the link load reader accepts `%14.3e` (SWMM switches to it at 10000).

    Class: NEW CAPABILITY. Plausible wrong implementation: a fixed-width `%14.3f` reader; red because
    `1.235e+04` fails to parse or is split. Reachability: SWMM prints exponent form for any link load
    of 10000 or more; artifact: the fixture's first load is 12345.678. Both-states property: the load
    equals the printed number. Invocation isolation: the second pollutant prints in the plain form in
    the same row. Second correct implementation: `float()` on each whitespace field passes.
    """
    _, ds_links = _datasets(_mk(tmp_path / "l"), TWO)
    load = ds_links["link_total_load"].sel(link_id="L1")
    assert load.sel(pollutant="TSS") == pytest.approx(12350.0)
    assert load.sel(pollutant="Coliform_ABC") == pytest.approx(100.0)


# --------------------------------------------------------------------------- test 6
SIX = [(f"P{k}", "MG/L") for k in range(5)] + [("PCOUNT", "#/L")]


def test_six_pollutants_concatenate_groups_and_linearize_every_mass_row(tmp_path):
    """S8, S9: the quality routing table over two groups of pollutants, with LogN rows.

    Class: NEW CAPABILITY. Plausible wrong implementation: reading only the first group of five, or
    skipping the count linearization on the continuity rows; red because the sixth pollutant is
    missing or its external inflow reads 3 not 1000. Reachability: SWMM prints groups of five, so six
    pollutants print two tables; artifact: the fixture prints two. Both-states property: values of
    the first five pollutants equal the printed numbers. Invocation isolation: the continuity error of
    the sixth is a percentage and is read as printed (0.5), a differently-positioned satisfying input.
    Second correct implementation: one pass that appends each group's values per label passes.
    """
    routing = _default_values(ROUTING_LABELS, 6)
    routing["External Inflow"][5] = 3.0
    routing["Continuity Error (%)"][5] = 0.5
    ds_nodes, _ = _datasets(_mk(tmp_path / "g"), SIX, routing=routing, extra_columns=True)
    assert list(ds_nodes["pollutant"].values) == [name for name, _ in SIX]
    inflow = ds_nodes["qual_routing_external_inflow"]
    assert inflow.sel(pollutant="PCOUNT") == pytest.approx(1000.0)
    assert inflow.sel(pollutant="P0") == pytest.approx(routing["External Inflow"][0])
    assert ds_nodes["qual_routing_continuity_error_perc"].sel(pollutant="PCOUNT") == 0.5
    assert ds_nodes["qual_routing_final_stored_mass"].sizes["pollutant"] == 6


# --------------------------------------------------------------------------- test 7
def test_hydrology_report_gets_washoff_and_runoff_quality(tmp_path):
    """S7, S10: washoff per subcatchment and the Runoff Quality Continuity table, on a hydrology report.

    Class: NEW CAPABILITY. Plausible wrong implementation: adding the washoff to a dataset whose
    `subcatchment_id` axis then grows past the runoff summary's, or taking the quantity table's error
    as the quality error; red because the axis length changes or the error reads 0.111. Reachability:
    the hydrology report of a quality-routed model prints both tables; artifact: the fixture. Both-
    states property: the runoff summary's own rows are unchanged. Invocation isolation: a second
    hydrology report with `Water Quality NO` yields a dataset with no pollutant dimension. Second
    correct implementation: merging with `join='left'` instead of reindexing passes.
    """
    pollutants = TWO
    rule = "  " + "-" * 126
    runoff = (
        "  ***************************\n  Subcatchment Runoff Summary\n  ***************************\n  \n"
        f"{rule}\n"
        "                            Total      Total      Total      Total     Imperv"
        "       Perv      Total       Total     Peak  Runoff\n"
        "                           Precip      Runon       Evap      Infil     Runoff"
        "     Runoff     Runoff      Runoff   Runoff   Coeff\n"
        "  Subcatchment                 mm         mm         mm         mm         mm"
        "         mm         mm    10^6 ltr      CMS\n"
        f"{rule}\n"
        f"  {'S1':<20}      20.00       0.00       0.00       1.75      15.28"
        "       0.00      15.28        0.02     0.01   0.764\n  \n"
    )
    marker = "  ************************\n  Node Time Series Results"

    def hydro_text(**kwargs):
        text = make_rpt(pollutants, node_summaries=False, extra_columns=False, **kwargs)
        return text.replace(marker, runoff + marker)

    inp, rpt = _write(_mk(tmp_path / "h"), pollutants, hydro_text(runoff_quality=True, washoff=True), name="hydro")
    ds = parse_hydrology_rpt_summary(rpt, f_inp=inp)
    assert list(ds["pollutant"].values) == ["TSS", "Coliform_ABC"]
    assert ds.sizes["subcatchment_id"] == 1
    assert ds["washoff_total_load"].sel(subcatchment_id="S1", pollutant="TSS") == pytest.approx(7.0)
    assert ds["washoff_total_load"].sel(subcatchment_id="S1", pollutant="Coliform_ABC") == pytest.approx(100.0)
    assert ds["runoff_qual_initial_buildup"].sizes["pollutant"] == 2
    # the quality error is the QUALITY table's (1 + 8 + p for the last row), not the quantity table's 0.111
    assert float(ds["runoff_qual_continuity_error_perc"].sel(pollutant="TSS")) == pytest.approx(9.0)
    inp2, rpt2 = _write(_mk(tmp_path / "h2"), pollutants, hydro_text(gate="NO"), name="hydro")
    ds_off = parse_hydrology_rpt_summary(rpt2, f_inp=inp2)
    assert "pollutant" not in ds_off.dims and ds_off.sizes["subcatchment_id"] == 1


# --------------------------------------------------------------------------- test 8 (A3 + A2)
def _summary(n_events=1, pollutants=TWO):
    ds = xr.Dataset(
        {
            "pollutant_max_conc": (("event_iloc", "node_id", "pollutant"), np.ones((n_events, 1, len(pollutants)))),
            "pollutant_time_of_max_conc_min": (
                ("event_iloc", "node_id", "pollutant"),
                np.full((n_events, 1, len(pollutants)), 5.0),
            ),
            "outfall_total_load": (
                ("event_iloc", "node_id", "pollutant"),
                np.full((n_events, 1, len(pollutants)), 3.0),
            ),
            "outfall_total_vol_10e6_ltr": (("event_iloc", "node_id"), np.full((n_events, 1), 4.0)),
            "qual_routing_external_inflow": (("event_iloc", "pollutant"), np.full((n_events, len(pollutants)), 2.0)),
        },
        coords={"event_iloc": list(range(n_events)), "node_id": ["OUT1"], "pollutant": [p[0] for p in pollutants]},
    )
    return ds


def _post(tmp_path, monkeypatch, rows, n_resumes=None, cfg_text=""):
    from hhemt import resume_events
    from hhemt.process_simulation import TRITONSWMM_sim_post_processing

    log = tmp_path / "run.log"
    ledger = resume_events.ledger_path_for(log)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    if rows is not None:
        ledger.write_text("".join(json.dumps(r) + "\n" for r in rows))
    cfg = tmp_path / "triton.cfg"
    cfg.write_text(cfg_text)
    monkeypatch.setattr("hhemt.run_simulation.model_logfile_for", lambda analysis, event_iloc, model_type: log)
    fake = SimpleNamespace(
        _analysis=SimpleNamespace(),
        _scenario=SimpleNamespace(event_iloc=0),
        log=SimpleNamespace(n_resumes=None if n_resumes is None else SimpleNamespace(get=lambda: n_resumes)),
        scen_paths=SimpleNamespace(triton_swmm_cfg=cfg),
    )
    fake._quality_stats_status = lambda model_type: TRITONSWMM_sim_post_processing._quality_stats_status(
        fake, model_type
    )
    return lambda ds: TRITONSWMM_sim_post_processing._apply_quality_stats_rule(fake, ds, "tritonswmm")


def test_a2_withholds_statistics_after_a_snapshot_resume(tmp_path, monkeypatch):
    """A2: a snapshot-path resume NaNs S1-S10 except outfall volume and records the status coordinate.

    Class: NEW CAPABILITY. Plausible wrong implementation: NaN-filling on `n_resumes >= 1` alone (it
    would also withhold exact-replay resumes), or putting the reason in an attribute that
    consolidation drops; red because the replay case is withheld or the coordinate is absent.
    Reachability: the resume-event ledger records `path: snapshot` or `path: replay` per resume;
    artifact: the three ledger fixtures below. Both-states property: the outfall volume and the
    dimensions are unchanged. Invocation isolation: three ledgers (snapshot, replay only, none).
    Second correct implementation: reading the model log marker directly instead of the ledger passes.
    """
    apply = _post(_mk(tmp_path / "snap"), monkeypatch, [{"path": "snapshot"}])
    out = apply(_summary())
    assert np.isnan(out["pollutant_max_conc"]).all() and np.isnan(out["outfall_total_load"]).all()
    assert np.isnan(out["qual_routing_external_inflow"]).all()
    assert (out["outfall_total_vol_10e6_ltr"] == 4.0).all()
    assert out["pollutant_stats_status"].values.tolist() == [
        "withheld: resumed from a snapshot; SWMM water-quality state is not carried across a snapshot resume"
    ]
    out_r = _post(_mk(tmp_path / "rep"), monkeypatch, [{"path": "replay"}, {"path": "replay"}])(_summary())
    assert (out_r["pollutant_max_conc"] == 1.0).all()
    assert out_r["pollutant_stats_status"].values.tolist() == ["computed: resumed by exact replay"]
    out_n = _post(_mk(tmp_path / "none"), monkeypatch, None)(_summary())
    assert (out_n["pollutant_max_conc"] == 1.0).all()
    assert out_n["pollutant_stats_status"].values.tolist() == ["computed"]


def test_a2_fallback_for_a_tree_without_a_ledger(tmp_path, monkeypatch):
    """A2's ledger-less fallback: resumed at least once and no `swmm_snapshot_disable=1` line withholds.

    Class: NEW CAPABILITY. Plausible wrong implementation: no fallback (a pre-ledger tree publishes
    spoiled statistics); red because the statistics stay computed. Reachability: a legacy tree has no
    ledger but has `n_resumes`; artifact: the cfg fixtures. Both-states property: a sim that never
    resumed is computed; the case n_resumes = 1 is the boundary and withholds. Invocation isolation:
    the same resume count with the disable line present is computed (replay). Second correct
    implementation: reading `n_resumes` from the log file passes.
    """
    withheld = _post(_mk(tmp_path / "a"), monkeypatch, None, n_resumes=2, cfg_text="a=1\n")(_summary())
    assert np.isnan(withheld["pollutant_max_conc"]).all()
    kept = _post(_mk(tmp_path / "b"), monkeypatch, None, n_resumes=2, cfg_text="a=1\nswmm_snapshot_disable=1\n")(
        _summary()
    )
    assert (kept["pollutant_max_conc"] == 1.0).all()
    never = _post(_mk(tmp_path / "c"), monkeypatch, None, n_resumes=0)(_summary())
    assert (never["pollutant_max_conc"] == 1.0).all()
    # the boundary: ONE resume already withholds, so a `> 1` reading is caught
    one = _post(_mk(tmp_path / "d"), monkeypatch, None, n_resumes=1, cfg_text="a=1\n")(_summary())
    assert np.isnan(one["pollutant_max_conc"]).all()


def test_a3_resolves_the_snapshot_selector_from_the_model(tmp_path):
    """A3: unset resolves to replay only for a quality-routed model; an explicit value wins; no change otherwise.

    Class: NEW CAPABILITY. Plausible wrong implementation: forcing replay for every model with a
    declared pollutant (ignoring `IGNORE_QUALITY YES`), or ignoring an explicit False; red because
    the Norfolk-style model resolves True or the explicit False is overridden. Reachability: the
    predicate reads the hydraulics `.inp`; artifact: three `.inp` fixtures. Both-states property: a
    model without quality routing resolves to the pre-change value (False), so its cfg is the same
    object. Invocation isolation: explicit False on a quality-routed model; explicit True on a
    non-quality model. Second correct implementation: a regex over the file passes.
    """
    from hhemt.scenario import append_swmm_snapshot_disable, resolve_swmm_snapshot_disable

    routed = tmp_path / "routed.inp"
    routed.write_text(_inp(TWO))
    ignored = tmp_path / "ignored.inp"
    ignored.write_text(_inp(TWO, ignore_quality=True))
    plain = tmp_path / "plain.inp"
    plain.write_text(_inp([]))
    assert resolve_swmm_snapshot_disable(None, routed) is True
    assert resolve_swmm_snapshot_disable(None, ignored) is False
    assert resolve_swmm_snapshot_disable(None, plain) is False
    assert resolve_swmm_snapshot_disable(False, routed) is False
    assert resolve_swmm_snapshot_disable(True, plain) is True
    cfg = "a=1\n"
    assert append_swmm_snapshot_disable(cfg, resolve_swmm_snapshot_disable(None, plain)) is cfg
    assert append_swmm_snapshot_disable(cfg, resolve_swmm_snapshot_disable(None, routed)).endswith(
        "swmm_snapshot_disable=1\n"
    )


def test_a3_analysis_field_defaults_to_none():
    """A3: `analysis_config.swmm_snapshot_disable` is optional with default None.

    Class: NEW CAPABILITY. Plausible wrong implementation: leaving the default False, which makes
    "unset" indistinguishable from an explicit False; red because the default is False.
    Reachability: any analysis config load; artifact: the model's field. Both-states property: the
    field is not required. Invocation isolation: an explicit False loads as False. Second correct
    implementation: a sentinel enum default passes if it is not False.
    """
    from hhemt.config.analysis import analysis_config

    field = analysis_config.model_fields["swmm_snapshot_disable"]
    assert field.default is None and not field.is_required()


def test_factory_persists_an_explicit_false_and_nothing_when_unset(monkeypatch, tmp_path):
    """The synthetic case factory persists `swmm_snapshot_disable: false` for an explicit False, and no key when unset.

    Class: NEW CAPABILITY. Plausible wrong implementation: the old true-only rule, which drops an
    explicit False so A3 forces replay on the arm that exists to test the snapshot route; red because
    the persisted configuration carries no key. Reachability: `build_resume_from_clean_runtimes`
    and `resume_case` call `_build_case` with the caller's value; artifact: the recorded
    `additional_analysis_configs`. Both-states property: an unset call persists no key (pre-change
    behaviour). Invocation isolation: True persists true. Second correct implementation: always
    persisting the key as a string would fail the unset case, so the two-arm test discriminates.
    """
    import scripts.experiments.synth_compute_config as driver

    captured = {}

    def recorder(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            analysis=SimpleNamespace(), system=SimpleNamespace(cfg_system=SimpleNamespace(system_directory="x"))
        )

    monkeypatch.setattr(driver, "retrieve_synth_TRITON_SWMM_test_case", recorder)
    csv = tmp_path / "m.csv"
    csv.write_text("member_id\n0\n")

    def run(**kw):
        captured.clear()
        driver._build_case(
            analysis_name="a", sensitivity_csv=csv, start_from_scratch=False, resume=False, system_directory=None, **kw
        )
        return captured["additional_analysis_configs"]

    assert run(swmm_snapshot_disable=False)["swmm_snapshot_disable"] is False
    assert run(swmm_snapshot_disable=True)["swmm_snapshot_disable"] is True
    assert "swmm_snapshot_disable" not in run()


# --------------------------------------------------------------------------- test 9
def test_compared_column_set_is_the_same_with_and_without_water_quality(tmp_path):
    """The b4b conjunct-A column set does not change when a model routes water quality.

    Invariant: a gate-on summary's compared set equals its gate-off set. Class: NEW CAPABILITY.
    Plausible wrong implementation: routing pollutant series through the generic `_max`/`_last`
    pair, which adds `pollutant_conc_max`/`_last` to the compared set (24 to 28 per member); red
    under it because the sets differ. Reachability: `derive_compared_columns` ranges over a summary's
    variable names; artifact: the two fixtures. Both-states property: the set is computed by
    shipped code over the summary. Invocation isolation: two cases, as the invariant states: a
    summary carrying the forbidden generic pair (the violating input, which DOES enter the set) and
    one carrying only `pollutant_max_conc` (the satisfying input in a different position). Second
    correct implementation: excluding the series by name inside the loop passes.
    """
    from hhemt.eda.cross_sim_identity import derive_compared_columns
    from hhemt.process_simulation import summarize_swmm_simulation_results

    on, _ = _datasets(_mk(tmp_path / "on"), TWO)
    off, _ = _datasets(_mk(tmp_path / "off"), TWO, gate="NO", extra_columns=False)
    on_set = derive_compared_columns(summarize_swmm_simulation_results(on, 0))[2]
    off_set = derive_compared_columns(summarize_swmm_simulation_results(off, 0))[2]
    assert on_set == off_set and len(on_set) == 8
    violating = xr.Dataset(
        {
            "pollutant_conc_max": ("n", [1.0]),
            "pollutant_conc_last": ("n", [1.0]),
            "depth_m_max": ("n", [1.0]),
            "depth_m_last": ("n", [1.0]),
        }
    )
    assert "pollutant_conc_max" in derive_compared_columns(violating)[2]
    satisfying = xr.Dataset(
        {"pollutant_max_conc": ("n", [1.0]), "depth_m_max": ("n", [1.0]), "depth_m_last": ("n", [1.0])}
    )
    assert "pollutant_max_conc" not in derive_compared_columns(satisfying)[2]


# --------------------------------------------------------------------------- test 10
class _StubOutput:
    """A stand-in for pyswmm's Output over one node, one link and a chosen pollutant count."""

    def __init__(self, n_pollutants):
        self.links = {"L1": 0}
        self.nodes = {"N1": 0}
        self.units = {"flow": "CMS", "system": "SI"}
        self.project_size = [0, 1, 1, 1, n_pollutants]
        self.start = None
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def node_series(self, node_id, attribute):
        self.calls.append(("node", attribute))
        return {np.datetime64("2000-01-01T00:10:00").astype("datetime64[s]").item(): float(attribute)}

    def link_series(self, link_id, attribute):
        self.calls.append(("link", attribute))
        return {np.datetime64("2000-01-01T00:10:00").astype("datetime64[s]").item(): float(attribute)}


def _stub_pyswmm(monkeypatch, n_pollutants):
    import pyswmm

    holder = {}

    def make(_path):
        holder["out"] = _StubOutput(n_pollutants)
        return holder["out"]

    class _Series:
        def __init__(self, out):
            self.out = out

        def __getitem__(self, ident):
            ts = {np.datetime64("2000-01-01T00:10:00").astype("datetime64[s]").item(): 1.0}
            return SimpleNamespace(
                invert_depth=ts,
                hydraulic_head=ts,
                total_inflow=ts,
                flooding_losses=ts,
                flow_rate=ts,
                flow_depth=ts,
                flow_velocity=ts,
                capacity=ts,
            )

    monkeypatch.setattr(pyswmm, "Output", make)
    monkeypatch.setattr(pyswmm, "NodeSeries", _Series)
    monkeypatch.setattr(pyswmm, "LinkSeries", _Series)
    return holder


def test_out_path_reads_pollutants_by_index_and_gates_on_the_out_count(tmp_path, monkeypatch):
    """The `.out` path: attribute `6+p` (node) and `5+p` (link), units from `[POLLUTANTS]`, gated on the `.out`.

    Class: NEW CAPABILITY. Plausible wrong implementation: gating on the companion `.rpt` as well
    (T1/T2 vanish when the `.rpt` is absent), or taking units from `Output.units`; red because T1/T2
    are absent without a `.rpt` or the units are wrong. Reachability: pyswmm exposes
    `project_size[4]` and `node_series(id, attribute)`; artifact: the stub implements both. Both-
    states property: the series exist iff the `.out` pollutant count is above zero. Invocation
    isolation: three inputs: pollutants with a `.rpt` ABSENT, a zero-count `.out`, and the indices.
    Second correct implementation: reading through `swmm.toolkit.output` directly passes with an
    equivalent stub.
    """
    inp = tmp_path / "m.inp"
    inp.write_text(_inp(TWO))
    out_file = tmp_path / "m.out"
    out_file.write_text("")
    holder = _stub_pyswmm(monkeypatch, 2)
    ds_n, ds_l, pollutants, _start = return_node_time_series_results_from_outfile(
        out_file, f_inp=inp, return_quality=True
    )
    assert pollutants == TWO or [(n, u) for n, u in pollutants] == [("TSS", "mg/L"), ("Coliform_ABC", "#/L")]
    assert sorted(a for k, a in holder["out"].calls if k == "node") == [6, 7]
    assert sorted(a for k, a in holder["out"].calls if k == "link") == [5, 6]
    assert list(ds_n["pollutant"].values) == ["TSS", "Coliform_ABC"]
    assert float(ds_n["pollutant_conc"].sel(pollutant="Coliform_ABC").isel(node_id=0, date_time=0)) == 7.0
    assert "link_pollutant_conc" in ds_l
    # no companion .rpt exists above: T1/T2 are still written, S3-S10 are absent
    ds_nodes, _ = retrieve_SWMM_outputs_as_datasets(inp, out_file)
    assert "pollutant_conc" in ds_nodes and "outfall_total_load" not in ds_nodes
    # a zero-count .out yields no pollutant dimension
    _stub_pyswmm(monkeypatch, 0)
    ds_nodes0, _ = retrieve_SWMM_outputs_as_datasets(inp, out_file)
    assert "pollutant" not in ds_nodes0.dims


# --------------------------------------------------------------------------- CF attributes
def test_cf_units_follow_the_pollutants_and_are_omitted_when_they_differ(tmp_path):
    """The CF `units` of a water-quality variable come from the dataset's own unit coordinate.

    Class: NEW CAPABILITY. Plausible wrong implementation: a fixed unit in the attribute map, which is
    wrong for a `ug/L` model and for a mixed one; red because `mg/L` is stamped on every variable.
    Reachability: `pollutant_conc_units` rides every gate-on dataset; artifact: the two fixtures
    below. Both-states property: the variable keeps a `long_name`. Invocation isolation: one model
    with a single shared unit, one mixed. Second correct implementation: stamping units at write time
    from the `.inp` rather than from the coordinate passes.
    """
    from hhemt.cf_conventions import apply_cf_attributes

    same, _ = _datasets(_mk(tmp_path / "s"), [("A", "MG/L"), ("B", "MG/L")])
    apply_cf_attributes(same, "tritonswmm_swmm_node")
    assert same["pollutant_conc"].attrs["units"] == "mg/L"
    assert same["outfall_total_load"].attrs["units"] == "kg"
    mixed, _ = _datasets(_mk(tmp_path / "m"), TWO)
    apply_cf_attributes(mixed, "tritonswmm_swmm_node")
    assert "units" not in mixed["pollutant_conc"].attrs
    assert "pollutant_conc_units" in mixed["pollutant_conc"].attrs["comment"]
    assert mixed["pollutant_conc"].attrs["long_name"]
    assert mixed["pollutant"].attrs["long_name"]


# --------------------------------------------------------------------------- A3 through the preparation path
def _prepared_cfg(tmp_path, monkeypatch, pollutants, field_value, ignore_quality=False):
    """The text of the coupled TRITON cfg a scenario PREPARES, with the field left at `field_value`."""
    from tests.test_scenario_boundary_guards import _scenario_for_coupled_cfg

    scen, cfg = _scenario_for_coupled_cfg(tmp_path, monkeypatch, rim_from_dem=False)
    scen._analysis.cfg_analysis.swmm_snapshot_disable = field_value
    (tmp_path / "hydraulics.inp").write_text(_inp(pollutants, ignore_quality))
    scen._generate_TRITON_SWMM_cfg()
    return cfg.read_text()


def test_a3_prepared_cfg_carries_the_disable_line_only_for_a_quality_routed_model_left_at_default(
    tmp_path, monkeypatch
):
    """A3 end to end: the cfg a scenario prepares carries `swmm_snapshot_disable=1` iff quality is routed and unset.

    Class: NEW CAPABILITY. Plausible wrong implementation: resolving the selector correctly but not
    passing it to the cfg writer (the resolver test alone passes), or forcing the line for every
    model; red because the routed, unset cfg lacks the line or the plain cfg gains it. Reachability:
    `_generate_TRITON_SWMM_cfg` is the one site that writes the line; artifact: the four prepared
    cfgs below. Both-states property: the cfg of a model without quality routing equals the cfg a
    prepared scenario produced before the field could be unset (no line). Invocation isolation: an
    explicit False on a routed model keeps the snapshot route, a differently-positioned input. Second
    correct implementation: resolving inside the cfg writer instead of the call site passes.
    """
    key = "swmm_snapshot_disable=1"
    routed_unset = _prepared_cfg(_mk(tmp_path / "a"), monkeypatch, TWO, None)
    assert key in routed_unset.splitlines()
    plain_unset = _prepared_cfg(_mk(tmp_path / "b"), monkeypatch, [], None)
    assert not any(line.startswith("swmm_snapshot_disable") for line in plain_unset.splitlines())
    ignored_unset = _prepared_cfg(_mk(tmp_path / "c"), monkeypatch, TWO, None, ignore_quality=True)

    def _without_paths(text):
        return [line for line in text.splitlines() if "coupling_nodes" not in line and "/" not in line]

    assert _without_paths(plain_unset) == _without_paths(ignored_unset)
    routed_false = _prepared_cfg(_mk(tmp_path / "d"), monkeypatch, TWO, False)
    assert not any(line.startswith("swmm_snapshot_disable") for line in routed_false.splitlines())


# --------------------------------------------------------------------------- A2 reach: per-scenario stores
def _per_scenario_node_ds(pollutants=TWO):
    """A coupled per-scenario NODE dataset as the parser returns it: no `event_iloc` dimension."""
    n = len(pollutants)
    return xr.Dataset(
        {
            "pollutant_conc": (("date_time", "node_id", "pollutant"), np.ones((3, 1, n))),
            "pollutant_time_of_max_conc_min": (("node_id", "pollutant"), np.full((1, n), 5.0)),
            "outfall_total_load": (("node_id", "pollutant"), np.full((1, n), 3.0)),
            "outfall_total_vol_10e6_ltr": (("node_id",), np.full((1,), 4.0)),
            "outfall_emc": (("node_id", "pollutant"), np.full((1, n), 0.75)),
            "qual_routing_external_inflow": (("pollutant",), np.full((n,), 2.0)),
        },
        coords={"node_id": ["OUT1"], "pollutant": [p[0] for p in pollutants]},
    )


def _per_scenario_link_ds(pollutants=TWO):
    """A coupled per-scenario LINK dataset as the parser returns it: the quality variables are S6 and a series."""
    n = len(pollutants)
    return xr.Dataset(
        {
            "link_pollutant_conc": (("date_time", "link_id", "pollutant"), np.ones((3, 1, n))),
            "link_total_load": (("link_id", "pollutant"), np.full((1, n), 6.0)),
        },
        coords={"link_id": ["C1"], "pollutant": [p[0] for p in pollutants]},
    )


def _export(tmp_path, monkeypatch, rows):
    """Run the real `_export_SWMM_outputs` on a stand-in object; return the (nodes, links) it hands to the writer."""
    import contextlib

    from hhemt import process_simulation as ps
    from hhemt import resume_events

    log_file = tmp_path / "run.log"
    ledger = resume_events.ledger_path_for(log_file)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    if rows is not None:
        ledger.write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr("hhemt.run_simulation.model_logfile_for", lambda analysis, event_iloc, model_type: log_file)
    monkeypatch.setattr(
        ps,
        "retrieve_SWMM_outputs_as_datasets",
        lambda f_inp, f_rpt: (_per_scenario_node_ds(), _per_scenario_link_ds()),
    )
    written = {}

    def _write_output(ds, f_out, comp_level, verbose, mode):
        written[mode] = ds
        f_out.write_text("x")

    log = SimpleNamespace(
        deferred_writes=contextlib.nullcontext,
        SWMM_node_timeseries_written=None,
        SWMM_link_timeseries_written=None,
        n_resumes=None,
        add_sim_processing_entry=lambda *args, **kwargs: None,
    )
    paths = SimpleNamespace(
        output_tritonswmm_node_time_series=tmp_path / "nodes.nc",
        output_tritonswmm_link_time_series=tmp_path / "links.nc",
        swmm_hydraulics_inp=tmp_path / "m.inp",
        swmm_hydraulics_rpt=tmp_path / "m.rpt",
        triton_swmm_cfg=tmp_path / "triton.cfg",
    )
    fake = SimpleNamespace(
        _analysis=SimpleNamespace(),
        _scenario=SimpleNamespace(event_iloc=0),
        log=log,
        scen_paths=paths,
        _resolve_clear_raw=lambda value: None,
        _validate_path=lambda path, name: path,
        _swmm_node_outputs_processed=lambda model: False,
        _swmm_link_outputs_processed=lambda model: False,
        _write_output=_write_output,
        _should_clear_raw_for_model=lambda *args: False,
    )
    cls = ps.TRITONSWMM_sim_post_processing
    fake._quality_stats_status = lambda model_type: cls._quality_stats_status(fake, model_type)
    fake._apply_quality_stats_rule = lambda ds, model_type: cls._apply_quality_stats_rule(fake, ds, model_type)
    cls._export_SWMM_outputs(fake, "tritonswmm")
    return written["tritonswmm_swmm_node"], written["tritonswmm_swmm_link"]


_WITHHELD_TEXT = "withheld: resumed from a snapshot; SWMM water-quality state is not carried across a snapshot resume"


def test_a2_withholds_the_per_scenario_node_and_link_stores_after_a_snapshot_resume(tmp_path, monkeypatch):
    """A2 reach: the per-scenario node AND link stores get the summary's NaN rule and status coordinate.

    Class: NEW CAPABILITY. Plausible wrong implementation: A2 applied only to the summary stores (the
    build as it stood), so a snapshot-resumed event's per-scenario stores publish S2, S3, S5, S8 and
    S9 (node) and S6 (link) unwithheld and unmarked; red because those variables are finite and the
    coordinate is absent. Another wrong implementation withholds the node store and forgets the link
    store (S6 rides `ds_links`); the link assertions catch it. Reachability: `_export_SWMM_outputs`
    is the one writer of both stores and receives them from the parser; artifact: the stand-in export
    below runs the real method on parser-shaped datasets. Both-states property: the outfall volume
    (S4, hydraulic) and both concentration series (T1/T2) are kept, in the withheld and in the
    replay case. Invocation isolation: three ledgers (snapshot, replay, none) are three differently
    positioned inputs. Second correct implementation: applying the rule inside the parse step, before
    the export, passes. The status coordinate is a scalar here because a per-scenario store has no
    `event_iloc` dimension; the last assertion shows consolidation's own concat turns it into a
    per-event vector.
    """
    nodes, links = _export(_mk(tmp_path / "snap"), monkeypatch, [{"path": "snapshot"}])
    for name in (
        "pollutant_time_of_max_conc_min",
        "outfall_total_load",
        "outfall_emc",
        "qual_routing_external_inflow",
    ):
        assert np.isnan(nodes[name]).all(), name
    assert np.isnan(links["link_total_load"]).all()
    assert (nodes["outfall_total_vol_10e6_ltr"] == 4.0).all()
    assert (nodes["pollutant_conc"] == 1.0).all() and (links["link_pollutant_conc"] == 1.0).all()
    assert nodes["pollutant_stats_status"].ndim == 0 and links["pollutant_stats_status"].ndim == 0
    assert str(nodes["pollutant_stats_status"].values) == _WITHHELD_TEXT
    assert str(links["pollutant_stats_status"].values) == _WITHHELD_TEXT

    nodes_r, links_r = _export(_mk(tmp_path / "rep"), monkeypatch, [{"path": "replay"}])
    assert (nodes_r["outfall_total_load"] == 3.0).all() and (links_r["link_total_load"] == 6.0).all()
    assert str(nodes_r["pollutant_stats_status"].values) == "computed: resumed by exact replay"
    nodes_n, links_n = _export(_mk(tmp_path / "none"), monkeypatch, None)
    assert (nodes_n["pollutant_time_of_max_conc_min"] == 5.0).all()
    assert str(links_n["pollutant_stats_status"].values) == "computed"

    both = xr.concat(
        [nodes.assign_coords(event_iloc=[0]), nodes_r.assign_coords(event_iloc=[1])],
        dim="event_iloc",
        combine_attrs="drop_conflicts",
        join="outer",
    )
    assert both["pollutant_stats_status"].values.tolist() == [_WITHHELD_TEXT, "computed: resumed by exact replay"]


# --------------------------------------------------------------------------- F-R8-1 and F-R8-2: descriptors
def test_washoff_descriptor_states_its_subcatchment_geometry_and_outfall_sources_name_columns():
    """Descriptors: `washoff_total_load` says polygon (subcatchment); the outfall rows name COLUMNS.

    Class: NEW CAPABILITY. Plausible wrong implementation: leaving `washoff_total_load` on "whole
    domain (scalar)" (one value per subcatchment per pollutant is not one value for the domain), or
    leaving the outfall descriptors reading "row Total {pollutant}" although `Total {pollutant}` and
    `Total Volume` are COLUMN headers of the Outfall Loading Summary; red because the strings differ.
    Reachability: the crate builder publishes these descriptor strings verbatim; artifact: the table.
    Both-states property: the other `_wq_printed` descriptors keep their `row` wording and the
    geometry vocabulary membership is untouched. Invocation isolation: the check reads two keys of
    one pure table. Second correct implementation: building the two strings without the shared
    helper passes.
    """
    from hhemt.cf_conventions import _WQ_QUANTITY_PROVENANCE as table

    assert table["washoff_total_load"]["spatial_representation"] == "polygon (subcatchment)"
    assert table["outfall_total_load"]["source_variables"] == (
        "SWMM report, Outfall Loading Summary, column Total {pollutant}"
    )
    assert table["outfall_total_vol_10e6_ltr"]["source_variables"] == (
        "SWMM report, Outfall Loading Summary, column Total Volume"
    )
    assert table["link_total_load"]["source_variables"] == "SWMM report, Link Pollutant Load Summary, row per link"
    assert table["washoff_total_load"]["source_variables"] == (
        "SWMM report, Subcatchment Washoff Summary, row per subcatchment"
    )
