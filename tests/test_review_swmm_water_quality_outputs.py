"""Water-quality mass values and the replay-selector decision, checked over their whole classes.

Problem. SWMM prints every MASS value of a count pollutant (units `#/L`) as a base-10 logarithm --
in the outfall loading, link load and subcatchment washoff tables and in every mass row of the
two quality continuity tables -- while it prints the continuity percentage errors and the mass
values of a concentration pollutant (`MG/L`, `UG/L`) as plain numbers. A reader that linearizes
some of those rows and not others returns a count off by orders of magnitude with no error.
Separately, a coupled run whose model routes water quality must resume by exact replay unless the
user says otherwise, and that decision is read from the model file by SWMM's own section and
token rules, so it must give the same answer for every spelling SWMM accepts.

The report fixtures follow the line layout of SWMM's report writer (`report.c` and `statsrpt.c`
in SWMM 5.2): continuity rows are a label padded with dots to column 26 followed by one `%14.3f`
value per pollutant; a pollutant table's data rows sit between its second and third dashed lines.
"""

import pytest

from hhemt.scenario import ordered_inp_pollutants, resolve_swmm_snapshot_disable
from hhemt.swmm_output_parser import parse_rpt_quality_summaries

POLLUTANTS = [("TSS", "mg/L"), ("FC", "#/L")]
ROUTING_TERMS = {
    "Dry Weather Inflow": "dry_weather_inflow",
    "Wet Weather Inflow": "wet_weather_inflow",
    "Groundwater Inflow": "groundwater_inflow",
    "RDII Inflow": "rdii_inflow",
    "External Inflow": "external_inflow",
    "External Outflow": "external_outflow",
    "Flooding Loss": "flooding_loss",
    "Exfiltration Loss": "exfiltration_loss",
    "Mass Reacted": "mass_reacted",
    "Initial Stored Mass": "initial_stored_mass",
    "Final Stored Mass": "final_stored_mass",
}
RUNOFF_TERMS = {
    "Initial Buildup": "initial_buildup",
    "Surface Buildup": "surface_buildup",
    "Wet Deposition": "wet_deposition",
    "Sweeping Removal": "sweeping_removal",
    "Infiltration Loss": "infiltration_loss",
    "BMP Removal": "bmp_removal",
    "Surface Runoff": "surface_runoff",
    "Remaining Buildup": "remaining_buildup",
}
# Printed values: the mass pollutant prints a plain number, the count pollutant prints log10.
# Row i prints 1.5 + i for TSS and 2.0 + 0.25 * i for FC; the first FC row prints 0.000.
MASS_PRINTED = 1.5
COUNT_PRINTED = 2.0
ERROR_PCT = (0.125, 0.75)


def _row(label, values):
    return f"  {label} {'.' * (26 - len(label) - 3)}" + "".join(f"{v:14.3f}" for v in values)


def _continuity_table(header, terms):
    lines = [
        "  **************************" + "".join(f"{p:>14}" for p, _ in POLLUTANTS),
        f"  {header:<26}" + "".join(f"{u:>14}" for u in ("kg", "LogN")),
        "  **************************" + "    ----------" * len(POLLUTANTS),
    ]
    for i, label in enumerate(terms):
        count = 0.0 if i == 0 else COUNT_PRINTED + 0.25 * i
        lines.append(_row(label, (MASS_PRINTED + i, count)))
    lines.append(_row("Continuity Error (%)", ERROR_PCT))
    return lines + ["  "]


def _pollutant_table(title, rows, *, closing_system_row):
    dash = "  " + "-" * 60
    lines = [f"  {title}", dash, "                     header", "                     header", dash]
    lines += rows
    if closing_system_row:
        lines += [dash, "  System               header"]
    return lines + ["  "]


def _report_lines():
    lines = [
        "  Water Quality .......... YES",
        "  ",
    ]
    lines += _continuity_table("Runoff Quality Continuity", RUNOFF_TERMS)
    lines += [
        "  **************************        Volume        Volume",
        "  Flow Routing Continuity        hectare-m      10^6 ltr",
        "  **************************     ---------     ---------",
        "  Flooding Loss ............         0.200         2.000",
        "  Continuity Error (%) .....         0.444",
        "  ",
    ]
    lines += _continuity_table("Quality Routing Continuity", ROUTING_TERMS)
    lines += _pollutant_table(
        "Subcatchment Washoff Summary",
        [f"  {'S1':<20}{MASS_PRINTED:14.3f}{COUNT_PRINTED:14.3f}", f"  {'S2':<20}{0.0:14.3f}{0.0:14.3f}"],
        closing_system_row=True,
    )
    lines += _pollutant_table(
        "Outfall Loading Summary",
        [f"  {'OUT1':<20}{98.45:7.2f} {0.123:9.3f} {0.456:9.3f}{4.0:12.3f}{MASS_PRINTED:14.3f}{COUNT_PRINTED:14.3f}"],
        closing_system_row=True,
    )
    lines += _pollutant_table(
        "Link Pollutant Load Summary",
        [f"  {'L1':<20}{MASS_PRINTED:14.3f}{COUNT_PRINTED:14.3f}"],
        closing_system_row=False,
    )
    return lines


def _expected_mass(printed_mass, printed_count):
    return [printed_mass, 0.0 if printed_count == 0.0 else 10.0**printed_count]


def _mass_cases():
    yield pytest.param("washoff_total_load", {"subcatchment_id": "S1"}, (MASS_PRINTED, COUNT_PRINTED), id="S7-washoff")
    yield pytest.param("washoff_total_load", {"subcatchment_id": "S2"}, (0.0, 0.0), id="S7-washoff-zero")
    yield pytest.param("outfall_total_load", {"node_id": "OUT1"}, (MASS_PRINTED, COUNT_PRINTED), id="S3-outfall")
    yield pytest.param("link_total_load", {"link_id": "L1"}, (MASS_PRINTED, COUNT_PRINTED), id="S6-link")
    for prefix, terms in (("qual_routing", ROUTING_TERMS), ("runoff_qual", RUNOFF_TERMS)):
        for i, stem in enumerate(terms.values()):
            count = 0.0 if i == 0 else COUNT_PRINTED + 0.25 * i
            yield pytest.param(f"{prefix}_{stem}", {}, (MASS_PRINTED + i, count), id=f"{prefix}_{stem}")


@pytest.mark.parametrize(("name", "where", "printed"), list(_mass_cases()))
def test_every_mass_value_of_a_count_pollutant_is_linearized_and_no_other_value_is(name, where, printed):
    """Property: every mass variable reads the mass pollutant's printed value unchanged and the count
    pollutant's printed `x` as `10**x`, with a printed `0.000` reading 0.

    Class: every mass variable the reader produces from a quality-routed report -- the outfall load,
    the link load, the subcatchment washoff, the eleven Quality Routing Continuity mass rows and the
    eight Runoff Quality Continuity mass rows -- each with a non-zero and (first row, washoff S2) a
    zero count value.

    Why it is required: SWMM converts every one of these to log10 for a count pollutant, so a row the
    reader leaves unconverted reports, for a printed 2.0, a count of 2 where SWMM integrated 100, and
    a row converted for a mass pollutant reports 10**1.5 kg where SWMM integrated 1.5 kg.

    What kills it: dropping any one table or row family from the linearization, or applying it by
    pollutant position rather than by the pollutant's declared unit.

    A second correct implementation that still passes: one that linearizes each table at the point
    it is read, row by row, instead of through one shared helper.
    """
    out = parse_rpt_quality_summaries(_report_lines(), POLLUTANTS, include_runoff_quality=True, filepath="<fixture>")
    assert name in out
    got = out[name].sel(**where) if where else out[name]
    assert got.sel(pollutant="TSS").item() == pytest.approx(_expected_mass(*printed)[0])
    assert got.sel(pollutant="FC").item() == pytest.approx(_expected_mass(*printed)[1])


@pytest.mark.parametrize("prefix", ["qual_routing", "runoff_qual"])
def test_continuity_percentages_are_read_as_printed_for_every_pollutant(prefix):
    """Property: the two quality continuity errors are read as printed, for the count pollutant too.

    Class: both quality continuity tables, both pollutant kinds.

    Why it is required: SWMM computes the percentage before it converts a count pollutant's masses to
    log10, so the printed percentage is already linear; converting it reports 10**0.75 = 5.6 % for a
    printed 0.75 %.

    What kills it: routing the `Continuity Error (%)` row through the mass linearization.

    A second correct implementation that still passes: one that reads the error row with a separate
    pattern from the mass rows.
    """
    out = parse_rpt_quality_summaries(_report_lines(), POLLUTANTS, include_runoff_quality=True, filepath="<fixture>")
    got = out[f"{prefix}_continuity_error_perc"]
    assert got.sel(pollutant="TSS").item() == pytest.approx(ERROR_PCT[0])
    assert got.sel(pollutant="FC").item() == pytest.approx(ERROR_PCT[1])


_WATER_QUALITY_NAMES = (
    ["pollutant_conc", "link_pollutant_conc", "pollutant_max_conc", "pollutant_time_of_max_conc_min"]
    + ["outfall_total_load", "outfall_total_vol_10e6_ltr", "outfall_emc", "link_total_load", "washoff_total_load"]
    + [f"qual_routing_{stem}" for stem in ROUTING_TERMS.values()]
    + ["qual_routing_continuity_error_perc"]
    + [f"runoff_qual_{stem}" for stem in RUNOFF_TERMS.values()]
    + ["runoff_qual_continuity_error_perc"]
)


@pytest.mark.parametrize("name", _WATER_QUALITY_NAMES)
def test_each_water_quality_variable_is_advertised_only_through_the_gated_union_and_described_once(name):
    """Property: every water-quality variable can be listed in a deposit's `variableMeasured` only on a
    store that holds it, carries exactly one descriptor, and is absent from the ungated base map.

    Class: all 30 water-quality variable names the processed outputs can carry.

    Why it is required: the crate builder lists the gated union filtered to a store's own variables,
    but lists the base CF map WHOLE when it cannot see the store, so a name in the base map is claimed
    for stores that do not hold it; and the mode-blind descriptor read merges three tables, so a name
    described in two of them is published with whichever table merges last.

    What kills it: moving the names into the base CF map, dropping a name from the advertised union, or
    giving a name a descriptor in a second table.

    A second correct implementation that still passes: one advertising these names from a dedicated
    gated set that the crate builder unions in, with the descriptors kept in the base descriptor table
    under the same keys.
    """
    from hhemt import cf_conventions as cf

    assert name in cf._ADVERTISABLE_VARIABLES
    assert name not in cf._CF_VARIABLE_MAP
    tables = (cf._QUANTITY_PROVENANCE, cf._PERF_QUANTITY_PROVENANCE, cf._WQ_QUANTITY_PROVENANCE)
    assert sum(name in table for table in tables) == 1
    assert name in cf._DESCRIBED_VARIABLES


@pytest.mark.parametrize("engine", ["zarr", "nc"])
def test_a_per_store_status_survives_the_write_and_the_timeseries_concat_per_event(tmp_path, engine):
    """Property: the per-event water-quality status a per-scenario store carries comes back, after a real
    write, a reopen and the timeseries consolidation's concat along `event_iloc`, as one value per event.

    Class: a per-scenario store written in either processed-output format, holding a scalar status, for
    a pair of events whose statuses differ (one withheld, one computed).

    Why it is required: the status is the only record of WHY an event's statistics are NaN, and a
    per-scenario store has no `event_iloc` dimension, so the status rides as a scalar and becomes
    per-event only through the concat. A concat that keeps the first dataset's scalar for every event
    labels a computed event withheld, or hides a withheld event's reason.

    What kills it: the concat keeping one scalar for all events -- which is what xarray's announced
    future defaults (`coords="minimal"`, `compat="override"`) do to a coordinate the call leaves
    implicit -- or a writer that drops a scalar string coordinate.

    A second correct implementation that still passes: a store that carries the status on a length-one
    `event_iloc` dimension, so the concat stacks it directly.
    """
    import numpy as np
    import xarray as xr

    from hhemt.utils import write_zarr, write_zarr_then_netcdf

    statuses = ["withheld: resumed from a snapshot", "computed"]
    reopened = []
    for event_iloc, status in enumerate(statuses):
        ds = xr.Dataset(
            {"outfall_total_load": (("node_id", "pollutant"), np.full((2, 1), np.nan if event_iloc == 0 else 1.0))},
            coords={"node_id": ["N1", "OUT1"], "pollutant": ["TSS"]},
        ).assign_coords(pollutant_stats_status=status)
        path = tmp_path / f"node_{event_iloc}.{engine}"
        if engine == "zarr":
            write_zarr(ds, path, 5)
            back = xr.open_dataset(path, engine="zarr", consolidated=False, chunks="auto", decode_timedelta=False)
        else:
            write_zarr_then_netcdf(ds, path, 5)
            back = xr.open_dataset(path, chunks="auto", decode_timedelta=False)
        reopened.append(back.assign_coords(event_iloc=[event_iloc]))
    combined = xr.concat(reopened, dim="event_iloc", combine_attrs="drop_conflicts", join="outer")
    assert combined["pollutant_stats_status"].dims == ("event_iloc",)
    assert [str(v) for v in combined["pollutant_stats_status"].values] == statuses


def _inp(tmp_path, *, options, pollutants_section):
    path = tmp_path / "hydraulics.inp"
    path.write_text(f"[TITLE]\nreview fixture\n\n{options}\n{pollutants_section}\n[JUNCTIONS]\nN1 0 3\n")
    return path


_POLLUTANTS_ONE = (
    "[POLLUTANTS]\n;;Name Units Crain Cgw Crdii Kdecay SnowOnly CoPollutant CoFrac Cinit\nTSS MG/L 0 0 0 0 NO * 0 0\n"
)


@pytest.mark.parametrize(
    ("options", "pollutants_section", "expected"),
    [
        pytest.param("[OPTIONS]\nFLOW_UNITS CMS\n", _POLLUTANTS_ONE, True, id="pollutant-no-ignore-line"),
        pytest.param("[OPTIONS]\nIGNORE_QUALITY NO\n", _POLLUTANTS_ONE, True, id="ignore-NO"),
        pytest.param("[OPTIONS]\nIGNORE_QUALITY YES\n", _POLLUTANTS_ONE, False, id="ignore-YES"),
        pytest.param("[options]\nignore_quality yes\n", _POLLUTANTS_ONE, False, id="ignore-yes-lowercase"),
        pytest.param(
            "[OPTIONS]\nIGNORE_QUALITY YES ; trailing comment\n", _POLLUTANTS_ONE, False, id="ignore-YES-comment"
        ),
        pytest.param("[OPTIONS]\nFLOW_UNITS CMS\n", "", False, id="no-pollutants-section"),
        pytest.param(
            "[OPTIONS]\nFLOW_UNITS CMS\n",
            "[POLLUTANTS]\n;;Name Units\n;TSS MG/L 0 0 0 0 NO * 0 0\n",
            False,
            id="pollutant-line-commented-out",
        ),
        pytest.param(
            "[OPTIONS]\nFLOW_UNITS CMS\n",
            "[POLLUTANT]\nTSS MG/L 0 0 0 0 NO * 0 0\n",
            True,
            id="section-header-abbreviated",
        ),
        pytest.param(
            "",
            _POLLUTANTS_ONE + "\n[OPTIONS]\nIGNORE_QUALITY YES\n",
            False,
            id="options-after-pollutants",
        ),
    ],
)
def test_unset_selector_resolves_from_whether_the_model_routes_quality(tmp_path, options, pollutants_section, expected):
    """Property: with the selector left unset, a coupled scenario resumes by replay exactly when its
    model routes water quality, and an explicit value is always returned as given.

    Class: model files spelling the two inputs (a declared pollutant, `IGNORE_QUALITY`) in each way
    SWMM accepts -- either case, a trailing comment, a commented-out pollutant, an abbreviated
    section header, `[OPTIONS]` after `[POLLUTANTS]` -- and models that do not route quality.

    Why it is required: SWMM decides quality routing from these same lines, so a model SWMM routes
    quality for but the toolkit reads as not routing it resumes from a snapshot that drops the node
    and link quality state, and the reverse forces an unnecessary replay.

    What kills it: matching the section header or the option keyword case-sensitively, or reading a
    commented-out pollutant line as a declaration.

    A second correct implementation that still passes: one that asks a SWMM parser library whether
    the model declares a pollutant and ignores quality, rather than tokenizing the file itself.
    """
    inp = _inp(tmp_path, options=options, pollutants_section=pollutants_section)
    assert resolve_swmm_snapshot_disable(None, inp) is expected
    assert resolve_swmm_snapshot_disable(True, inp) is True
    assert resolve_swmm_snapshot_disable(False, inp) is False


def test_pollutant_units_map_case_insensitively_in_declared_order(tmp_path):
    """Property: pollutants come back in `[POLLUTANTS]` order with SWMM's unit keywords mapped to
    `mg/L`, `ug/L` and `#/L` whatever their case.

    Class: the three SWMM concentration units in upper and lower case, in a non-alphabetical order.

    Why it is required: every pollutant column SWMM writes follows declaration order, and the unit
    decides whether a mass value is linearized; a lower-case `#/l` read as an unknown unit would
    leave a count pollutant's log10 masses unconverted.

    What kills it: sorting the pollutants, or matching unit keywords case-sensitively.

    A second correct implementation that still passes: a regex reader over the section's lines.
    """
    inp = _inp(
        tmp_path,
        options="[OPTIONS]\nFLOW_UNITS CMS\n",
        pollutants_section=(
            "[POLLUTANTS]\nZn ug/l 0 0 0 0 NO * 0 0\nFC #/l 0 0 0 0 NO * 0 0\nAl MG/L 0 0 0 0 NO * 0 0\n"
        ),
    )
    assert ordered_inp_pollutants(inp) == [("Zn", "ug/L"), ("FC", "#/L"), ("Al", "mg/L")]
