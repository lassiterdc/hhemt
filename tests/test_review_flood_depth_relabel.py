"""Reviewer tests for TRITON's depth variables: they carry flood-depth names and attributes.

The solver writes `H` (depth at a print) and `MH` (its running maximum of depth). The toolkit stores them as
`flood_depth`, `max_flood_depth`, `flood_depth_last_tstep` and `time_of_max_flood_depth`, with the CF
`standard_name` `flood_water_thickness` (a depth above the bed, where `sea_surface_height_above_geoid` would name a
height above the geoid), and states in the provenance text that `max_flood_depth` is the solver's running maximum,
not a maximum over prints. These tests hold that from the outside: each one names the input class it quantifies
over, why the property is required, and the change to production code that turns it red.

Every test here neither compiles nor executes a solver, so all of them run on the laptop.
"""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

REPO = Path(__file__).resolve().parents[1]

DEPTH_NAMES = ("max_flood_depth", "flood_depth", "flood_depth_last_tstep", "time_of_max_flood_depth")
RETIRED_DEPTH_NAMES = ("max_wlevel_m", "wlevel_m", "wlevel_m_last_tstep", "time_of_max_wlevel_min")


def test_raw_prefix_map_names_h_and_mh_as_flood_depth():
    """Property: the solver prefixes `MH` and `H` map to the flood-depth names; `QX` and `QY` do not move.

    Class: the whole raw-prefix table (four keys) and the canonical column order derived from it.
    Why required: this dict mints the timeseries column names, and `CANONICAL_TRITON_VARS`, the required-column
    check and the EDA variable list all derive from it, so one wrong value mislabels every downstream store.
    Kills it: `"MH": "max_wlevel_m"` or `"H": "wlevel_m"` restored, or the two being swapped.
    Passes under: an implementation that builds the dict from a helper or a constant table.
    """
    from hhemt.process_simulation import CANONICAL_TRITON_VARS, TRITON_RAW_VAR_BY_PREFIX

    assert TRITON_RAW_VAR_BY_PREFIX == {
        "MH": "max_flood_depth",
        "H": "flood_depth",
        "QX": "unit_discharge_x",
        "QY": "unit_discharge_y",
    }
    assert CANONICAL_TRITON_VARS == ("max_flood_depth", "flood_depth", "unit_discharge_x", "unit_discharge_y")


def _depth_timeseries() -> xr.Dataset:
    """A tseries-shaped Dataset whose depth extremes are known. Dims (timestep, y, x) = (6, 4, 5), 10 m cells.

    Cell (y=2, x=2): printed depth peaks at 3.0 on the 4th print (t=30) and is 1.0 at the last print; the solver's
    running maximum at the last print (`max_flood_depth[-1]`) is 3.0079, ABOVE every printed depth, the
    between-prints case.
    Cell (3, 3): the running maximum at the last print is 0.0003 BELOW the largest printed depth, the coupled
    manhole case at solver commits before b3820a44.
    Cell (1, 1): depth all NaN.
    """
    rng = np.random.default_rng(20261103)
    nt, ny, nx = 6, 4, 5
    h = rng.uniform(0.2, 1.5, size=(nt, ny, nx))
    h[:, 1, 1] = np.nan
    h[3, 2, 2], h[5, 2, 2] = 3.0, 1.0
    mh = np.maximum.accumulate(np.nan_to_num(h, nan=0.0), axis=0)
    mh[:, 1, 1] = np.nan
    mh[-1, 2, 2] = 3.0079
    mh[-1, 3, 3] = np.nanmax(h[:, 3, 3]) - 0.0003
    q = rng.uniform(-1.0, 1.0, size=(nt, ny, nx))
    dims = ("timestep", "y", "x")
    return xr.Dataset(
        {
            "max_flood_depth": (dims, mh),
            "flood_depth": (dims, h),
            "unit_discharge_x": (dims, q),
            "unit_discharge_y": (dims, q[::-1]),
        },
        coords={
            "timestep": np.arange(nt) * 10.0,
            "y": np.array([35.0, 25.0, 15.0, 5.0]),
            "x": np.array([5.0, 15.0, 25.0, 35.0, 45.0]),
        },
    )


def test_summary_writes_the_depth_names_with_d3_values():
    """Property: the summary's depth variables have the flood-depth names and the running-maximum meaning, cell by cell.

    Class: every cell of a 4x5 grid, including a running maximum above the printed maximum, one below it, and an
    all-NaN cell.
    Why required: `max_flood_depth` is as the solver's value at the last print and
    `time_of_max_flood_depth` as the print of the largest PRINTED depth; the streaming-helper tests do not call
    `summarize_triton_simulation_results`, so nothing else holds which input feeds which output.
    Kills it: a stale name; `max_flood_depth` recomputed as the maximum over prints (cell (2, 2) would read 3.0
    and cell (3, 3) would read the printed maximum, not 0.0003 below it); `time_of_max_flood_depth` taken from
    the running maximum; `flood_depth_last_tstep` taken from a print other than the last.
    Passes under: an implementation that skips the streaming helper, provided names and values agree.
    """
    from hhemt.process_simulation import summarize_triton_simulation_results

    ds = _depth_timeseries()
    out = summarize_triton_simulation_results(ds, 0, 10.0, chunksize_mb=0.001).isel(event_iloc=0)
    assert set(out.data_vars) == {
        *(n for n in DEPTH_NAMES if n != "flood_depth"),  # the timeseries name is not a summary variable
        "max_unit_discharge",
        "time_of_max_unit_discharge",
        "unit_discharge_x_at_time_of_max_unit_discharge",
        "unit_discharge_y_at_time_of_max_unit_discharge",
        "final_surface_flood_volume",  # the unit is `units: m3`
        # the flood velocity |q| / h: the summary carries these four too
        "max_flood_velocity",
        "time_of_max_flood_velocity",
        "x_flood_velocity_at_time_of_max_flood_velocity",
        "y_flood_velocity_at_time_of_max_flood_velocity",
    }
    for forbidden in RETIRED_DEPTH_NAMES:
        assert forbidden not in out.data_vars, forbidden

    h = ds["flood_depth"].values
    np.testing.assert_allclose(out["max_flood_depth"].values, ds["max_flood_depth"].values[-1], equal_nan=True)
    assert float(out["max_flood_depth"].values[2, 2]) == pytest.approx(3.0079)
    assert float(out["max_flood_depth"].values[3, 3]) == pytest.approx(np.nanmax(h[:, 3, 3]) - 0.0003)
    first_peak = np.argmax(np.nan_to_num(h, nan=-np.inf), axis=0)
    ref_t = np.where(np.isnan(h).all(axis=0), np.nan, ds["timestep"].values[first_peak])
    np.testing.assert_allclose(out["time_of_max_flood_depth"].values, ref_t, equal_nan=True)
    assert float(out["time_of_max_flood_depth"].values[2, 2]) == 30.0
    np.testing.assert_allclose(out["flood_depth_last_tstep"].values, h[-1], equal_nan=True)
    assert "final_surface_flood_volume_m3" not in out.data_vars  # the forbidden unit-token name
    assert float(out["final_surface_flood_volume"].values) == pytest.approx(np.nansum(h[-1]) * 100.0)


@pytest.mark.parametrize("case", ["no_running_maximum", "static_running_maximum"])
def test_summary_depth_branches_without_a_timeseries_running_maximum(case):
    """Property: the two other input shapes of the depth summary still write the flood-depth names with right values.

    Class: a timeseries with NO `max_flood_depth` variable (the maximum must come from the printed depths), and
    one whose `max_flood_depth` has no time dimension (it is passed through).
    Why required: the function branches three ways on whether and how the running maximum is present, and the
    rename touched a helper key (`max_` + the primary variable) that only the first-absent branch reads, so a
    test on the common branch alone would not see a broken key.
    Kills it: the absent branch reading a forbidden helper key (a `KeyError`), writing the maximum under a forbidden
    name, or returning the last-print depth instead of the maximum over prints; the static branch dropping or
    renaming the variable.
    Passes under: an implementation that merges the branches into one that takes a precomputed maximum.
    """
    from hhemt.process_simulation import summarize_triton_simulation_results

    ds = _depth_timeseries()
    h = ds["flood_depth"].values
    if case == "no_running_maximum":
        ds = ds.drop_vars("max_flood_depth")
        want = np.nanmax(np.where(np.isnan(h), -np.inf, h), axis=0)
        want = np.where(np.isinf(want), np.nan, want)
    else:
        static = np.full(h.shape[1:], 7.5)
        ds = ds.drop_vars("max_flood_depth").assign(max_flood_depth=(("y", "x"), static))
        want = static
    out = summarize_triton_simulation_results(ds, 0, 10.0, chunksize_mb=0.001).isel(event_iloc=0)
    np.testing.assert_allclose(out["max_flood_depth"].values, want, equal_nan=True)
    assert float(out["time_of_max_flood_depth"].values[2, 2]) == 30.0
    assert not any(r in out.data_vars for r in RETIRED_DEPTH_NAMES)


def test_summary_depth_variables_get_flood_depth_attributes_through_the_real_stamp():
    """Property: after `apply_cf_attributes` the four depth variables carry the flood-depth attribute columns.

    Class: the four depth variables of a summary stamped in a TRITON mode.
    Why required: the writer stamps before it writes, so this is what a reader of the store sees; a
    `sea_surface_height_above_geoid` here publishes a height above the geoid for a depth above the bed.
    Kills it: any of the three depth variables regaining the sea-surface name, a wrong unit, a missing
    `time: point` or `time: maximum`, or a `standard_name` on the time variable.
    Passes under: an implementation that stamps in the summary function instead of the map.
    """
    from hhemt.cf_conventions import apply_cf_attributes
    from hhemt.process_simulation import summarize_triton_simulation_results

    out = summarize_triton_simulation_results(_depth_timeseries(), 0, 10.0, chunksize_mb=0.001)
    out = apply_cf_attributes(out, mode="tritonswmm_triton")
    expect = {
        "max_flood_depth": ("m", "flood_water_thickness", "time: maximum"),
        "flood_depth_last_tstep": ("m", "flood_water_thickness", "time: point"),
        "time_of_max_flood_depth": ("min", None, None),
    }
    for name, (units, std, cm) in expect.items():
        a = out[name].attrs
        assert a["units"] == units, (name, a)
        assert a.get("standard_name") == std, (name, a)
        assert a.get("cell_methods") == cm, (name, a)
        assert "water level" not in a["long_name"].lower(), (name, a)


def test_cf_and_provenance_entries_match_section_1_2_and_d3():
    """Property: the four names have CF and provenance entries that state the flood-depth meaning, and no
    forbidden key remains.

    Class: the four depth names (and `final_surface_flood_volume`'s source) in `_CF_VARIABLE_MAP` and
    `_QUANTITY_PROVENANCE`, plus the four forbidden keys, which must be absent from both.
    Why required: the crate advertises from the CF map and the report renders the provenance table; the
    `max_flood_depth` provenance must state the solver's running maximum and the pre-b3820a44 coupled caveat, and
    the old text "maximum over all reported timesteps" is false for `MH`.
    Kills it: the old operation text; the running-maximum caveat dropped; a forbidden key left in either table;
    the volume still sourced from the forbidden name.
    Passes under: an implementation that words the caveat differently but keeps the three facts asserted.
    """
    from hhemt.cf_conventions import _CF_VARIABLE_MAP, _QUANTITY_PROVENANCE

    cols = {
        "max_flood_depth": ("flood_water_thickness", "m", "time: maximum"),
        "flood_depth": ("flood_water_thickness", "m", None),
        "flood_depth_last_tstep": ("flood_water_thickness", "m", "time: point"),
        "time_of_max_flood_depth": (None, "min", None),
    }
    for name, (std, units, cm) in cols.items():
        e = _CF_VARIABLE_MAP[name]
        assert (e["standard_name"], e["units"], e["cell_methods"]) == (std, units, cm), name
        assert "water level" not in e["long_name"].lower(), name
        assert name in _QUANTITY_PROVENANCE, name
        assert _QUANTITY_PROVENANCE[name]["spatial_representation"] == "grid cell", name
    for forbidden in RETIRED_DEPTH_NAMES:
        assert forbidden not in _CF_VARIABLE_MAP, forbidden
        assert forbidden not in _QUANTITY_PROVENANCE, forbidden

    op = _QUANTITY_PROVENANCE["max_flood_depth"]["operation"]
    assert "running maximum" in op and "b3820a44" in op and "below a printed depth" in op, op
    assert "all reported timesteps" not in op and "over prints" not in op, op
    top = _QUANTITY_PROVENANCE["time_of_max_flood_depth"]["operation"]
    assert "largest" in top and "printed" in top, top
    assert _QUANTITY_PROVENANCE["flood_depth_last_tstep"]["source_variables"] == "flood_depth"
    assert _QUANTITY_PROVENANCE["final_surface_flood_volume"]["source_variables"] == "flood_depth_last_tstep"
    # the volume carries no unit token, and no table keeps the forbidden key
    assert "final_surface_flood_volume_m3" not in _CF_VARIABLE_MAP
    assert "final_surface_flood_volume_m3" not in _QUANTITY_PROVENANCE
    assert _CF_VARIABLE_MAP["final_surface_flood_volume"]["units"] == "m3"


def test_no_depth_name_is_shared_by_the_triton_and_swmm_tiers():
    """Property: the four TRITON depth names are written by no SWMM-tier code.

    Class: every name the timeseries and summary writers produce, against the `.rpt` header registries and the
    `.out`-derived node and link names,
    including the SWMM node `depth`, which no TRITON variable may share.
    Why required: `_CF_VARIABLE_MAP` is consulted by name for every mode, so a shared name gets one description
    for two quantities.
    Kills it: naming the TRITON depth `depth` or `head` (the SWMM node names).
    Passes under: an implementation that keys the TRITON names in a separate table per mode.
    """
    from hhemt import constants

    swmm = set()
    for attr in dir(constants):
        if attr.startswith("LST_COL_HEADERS_"):
            swmm |= set(getattr(constants, attr))
    # the SWMM tseries names, which the TRITON tier must not write
    swmm |= {"depth", "head", "flow", "velocity", "link_depth", "capacity", "inflow_flow", "flooding"}
    from hhemt.process_simulation import TRITON_RAW_VAR_BY_PREFIX, summarize_triton_simulation_results

    written = set(TRITON_RAW_VAR_BY_PREFIX.values()) | set(
        summarize_triton_simulation_results(_depth_timeseries(), 0, 10.0, chunksize_mb=0.001).data_vars
    )
    assert set(DEPTH_NAMES) <= written, set(DEPTH_NAMES) - written  # the names under test are what the code writes
    assert not (written & swmm), written & swmm
    assert "max_velocity" in swmm  # control: the registries are non-empty and reachable (the link maximum velocity)


@pytest.mark.parametrize("golden", ["multisim", "sensitivity"])
def test_golden_zip_carries_the_depth_names_and_no_retired_name(golden):
    """Property: the regenerated golden holds the four names with the map's attributes and no forbidden name.

    Class: every array in the golden zip whose leaf name is one of the four, and every `zarr.json` for a forbidden
    name or for the sea-surface standard name.
    Why required: the goldens are the fixture other tests and the migration read, written by an independent
    generator, so only a read of the zip shows what a store carries.
    Kills it: a stale zip; `sea_surface_height_above_geoid` on a depth array; a wrong unit; a forbidden name in an
    array name or in any array's metadata.
    Passes under: a golden regenerated by a different but equivalent generator with the same names.
    """
    from hhemt.cf_conventions import _CF_VARIABLE_MAP

    z = zipfile.ZipFile(REPO / "tests" / "fixtures" / "goldens" / f"{golden}.zip")
    names = z.namelist()
    metas = [n for n in names if n.endswith("/zarr.json")]
    assert len(metas) > 100  # control: the walk reaches array metadata at all
    forbidden = re.compile("|".join((*RETIRED_DEPTH_NAMES, "sea_surface_height_above_geoid")))
    assert [n for n in names if forbidden.search(n)] == []
    assert [n for n in metas if forbidden.search(z.read(n).decode("utf-8", "replace"))] == []
    seen = set()
    for n in metas:
        leaf = n.rsplit("/", 2)[-2]
        if leaf in DEPTH_NAMES:
            seen.add(leaf)
            attrs = json.loads(z.read(n)).get("attributes", {})
            assert attrs["units"] == _CF_VARIABLE_MAP[leaf]["units"], (n, attrs)
            assert attrs.get("standard_name") == _CF_VARIABLE_MAP[leaf]["standard_name"], (n, attrs)
    assert seen == set(DEPTH_NAMES), seen


def test_docs_table_rows_for_the_depth_variables():
    """Property: the TRITON table has one row per depth name with its unit and CF name, and carries the
    running-maximum caveat.

    Class: the four rows, the four forbidden names, and the `max_flood_depth` meaning text.
    Why required: this page is the published specification the goldens are pinned to, and the
    running-maximum caveat belongs in the published text as well as the provenance table.
    Kills it: a missing row, a forbidden row, `sea_surface_height_above_geoid` left in the section, or the caveat
    dropped from the `max_flood_depth` row.
    Passes under: a table rendered from the CF map by a generator, with the same rows.
    """
    text = (REPO / "docs" / "reference" / "output-data-model.md").read_text()
    start = text.index("2D surface results (TRITON)")
    section = text[start : text.index("### Network results (SWMM)", start)]
    rows = {
        m.group(1): [c.strip().strip("`") for c in m.group(0).strip("|\n").split("|")]
        for m in re.finditer(r"^\| `([a-z0-9_]+)` .*$", section, re.M)
    }
    want = {
        "max_flood_depth": ("m", "flood_water_thickness"),
        "flood_depth": ("m", "flood_water_thickness"),
        "flood_depth_last_tstep": ("m", "flood_water_thickness"),
        "time_of_max_flood_depth": ("min", "n/a"),
    }
    for name, (units, std) in want.items():
        assert name in rows, name
        assert (rows[name][1], rows[name][2]) == (units, std), (name, rows[name])
    for forbidden in RETIRED_DEPTH_NAMES:
        assert forbidden not in rows and forbidden not in section, forbidden
    assert "sea_surface_height_above_geoid" not in section
    meaning = rows["max_flood_depth"][3]
    assert "running maximum" in meaning and "b3820a44" in meaning and "below a printed depth" in meaning, meaning


def test_eda_units_map_names_the_depth_rows_in_metres():
    """Property: the EDA byte-identity dataset's units attribute maps the new depth names to `m`.

    Class: `_b4b_dataset` built over the four canonical rows by the real function.
    Why required: the EDA renderer labels each raw-output row from this map; a row whose name is absent from it
    is unlabelled or mislabelled.
    Kills it: the map keeping a forbidden key, or a depth row missing from it.
    Passes under: an implementation that derives the map from the CF table.
    """
    from hhemt.eda.raw_byte_identity import TRITON_VARS, _b4b_dataset

    ts = np.array([0.0, 10.0])
    pair = (
        xr.DataArray([True, True], dims="timestep", coords={"timestep": ts}),
        xr.DataArray([0.0, 0.0], dims="timestep", coords={"timestep": ts}),
    )
    meta = {
        "serial": {
            "config_label": "serial",
            "family": "cpu",
            "is_reference": True,
            "run_mode": "serial",
            "hpc_partition": "standard",
            "n_gpus": 0,
            "n_mpi": 1,
            "n_omp": 1,
            "n_nodes": 1,
        }
    }
    ds = _b4b_dataset(
        {"serial": {v: pair for v in TRITON_VARS}},
        meta,
        boundaries=[],
        raw_out_type="bin",
        interval=600.0,
        degraded=False,
        degraded_reason="",
        reference_config_by_family={"cpu": "serial"},
    )
    units = json.loads(ds.attrs["units_by_raw_output_type"])
    assert units["max_flood_depth"] == "m" and units["flood_depth"] == "m", units
    assert set(units) >= set(TRITON_VARS), (units, TRITON_VARS)
    assert list(ds["raw_output_type"].values) == list(TRITON_VARS)


def _recorded_refs(node):
    """Yield every dict under `node` that records a source variable with its attrs (a manifest `ref`)."""
    if isinstance(node, dict):
        if "variable" in node and "attrs" in node:
            yield node
        for v in node.values():
            yield from _recorded_refs(v)
    elif isinstance(node, list):
        for v in node:
            yield from _recorded_refs(v)


def test_bundle_fixture_manifests_record_the_stored_attributes_of_the_renamed_depth():
    """Property: each bundle-fixture manifest that names a depth variable records that variable's current CF attrs.

    Class: every `peak_flood_depth.manifest.json` under `tests/fixtures/bundles` (the seven fixtures), every `ref`
    whose `variable` is one of the four depth names.
    Why required: a manifest is the figure's provenance record of what it read; one that says
    `max_flood_depth` is a `sea_surface_height_above_geoid` called "Maximum water level over simulation"
    reproduces, inside a fixture, a mislabel, and a bundle copied from it would carry it.
    Kills it: a manifest that changes `variable` only, leaving the forbidden `standard_name` and `long_name`.
    Passes under: manifests regenerated from a real render, which records the stamped store attributes.
    """
    from hhemt.cf_conventions import _CF_VARIABLE_MAP

    paths = sorted((REPO / "tests" / "fixtures" / "bundles").rglob("peak_flood_depth.manifest.json"))
    assert len(paths) == 7, [p.relative_to(REPO).as_posix() for p in paths]  # control: the seven fixtures
    checked = 0
    for p in paths:
        for ref in _recorded_refs(json.loads(p.read_text())):
            assert ref["variable"] not in RETIRED_DEPTH_NAMES, (p.name, ref["variable"])
            if ref["variable"] in DEPTH_NAMES:
                checked += 1
                want = _CF_VARIABLE_MAP[ref["variable"]]
                for key in ("standard_name", "long_name", "units"):
                    assert ref["attrs"].get(key) == want[key], (p.relative_to(REPO).as_posix(), ref["variable"], key)
    assert checked >= 7 * 2, checked  # control: the walk reached the refs


def test_cf_conventions_line_citations_resolve_to_the_cited_expression():
    """Property: every `process_simulation.py:N` citation in `cf_conventions.py` names a line that holds the
    selection the comment describes.

    Class: all such citations in the module (two at the time of writing, both describing the final-print
    selection `.sel(timestep=tsteps.max())`).
    Why required: a change that moves a cited line must re-verify the citation, and
    the published-docs gate keys on the token, so a stale number sends a reader to the wrong statement.
    Kills it: a citation left at a line that is a neighbouring statement.
    Passes under: a comment that cites the function name instead of a line, or no citation at all.
    """
    src = (REPO / "src" / "hhemt" / "cf_conventions.py").read_text()
    ps = (REPO / "src" / "hhemt" / "process_simulation.py").read_text().splitlines()
    cites = [int(m.group(1)) for m in re.finditer(r"process_simulation\.py:(\d+)", src)]
    for n in cites:
        assert "tsteps.max()" in ps[n - 1], (n, ps[n - 1].strip())


# ---------------------------------------------------------------------------------------------
# Census of the forbidden names in every form, with a control that varies the property
# ---------------------------------------------------------------------------------------------

_EXACT = r"\bwlevel_m\b|max_wlevel_m|wlevel_m_last_tstep|time_of_max_wlevel_min"
#: Wildcard and fragment forms of the forbidden family, which a name-by-name matcher does not reach.
_WILD = r"wlevel_[*{]|[*}]_wlevel\b|\*wlevel|_wlevel_(?:m|min)\b|wlevel \(m\)|wlevel[_ ]m\b"
#: Prose forms of the forbidden quantity name, and the forbidden CF standard name.
_PROSE = (
    r"(?i:(?:peak|maximum|max) water[- ]level|water[- ]level (?:timeseries|at (?:the )?final)"
    r"|Water level over simulation)|sea_surface_height_above_geoid"
)
_EXACT_RE = re.compile(_EXACT)
_CENSUS = re.compile("|".join((_EXACT, _WILD, _PROSE)))
_SKIP_PARTS = ("legacy_layouts", "docs/reference/bundles", "__pycache__", ".zarr", "fixtures/goldens")
#: The files allowed to quote a forbidden name: the migration and its two tests, plus the two files that must name
#: them (the vocabulary-freeze manifest and the `test_cf_conventions.py` soundness test) and the three reviewer files
#: that assert absence. Measured 2026-11-02: the census is red on the freeze manifest unless it is allowed, because
#: that manifest lists the forbidden tokens.
_MIGRATION_AND_ITS_TESTS = (
    "src/hhemt/version_migration/versions/V0028__flood_depth_and_unit_discharge_names.py",
    "tests/test_version_migration_v0028.py",
    "tests/test_review_v0028_migration.py",
)
_ALSO_ALLOWED = (
    "scripts/vocabulary_freeze.yaml",
    "tests/test_cf_conventions.py",
    "tests/test_review_unit_discharge_relabel.py",
    "tests/test_review_flood_depth_relabel.py",
    "tests/test_review_swmm_tier_names.py",
)
#: `origin/develop`'s report-text style tests list the phrase "peak water level" as a FORBIDDEN word, to assert that no
#: report text prints it; a negative assertion has to quote the phrase it forbids. They are
#: allowed for the prose forms only in the sense that this census, which also reads prose, cannot tell a forbidden-word
#: list from a use; the exact-name forms are still read in them.
_RETIRED_WORD_LISTS = (
    "tests/test_report_text_eda.py",
    "tests/test_report_text_maps.py",
    "tests/test_report_text_status_pages.py",
)
_QUOTING_FILES = {*_MIGRATION_AND_ITS_TESTS, *_ALSO_ALLOWED}
#: A line that describes the solver's RAW output file prefixes in the solver's own words (`H` water level, `MH`
#: maximum water level). That is a statement about the solver's file format, not about a stored toolkit variable.
_RAW_SOLVER_WORDING = "`MH` (maximum water level)"
_SUFFIXES = {".py", ".md", ".rst", ".yaml", ".yml", ".json", ".txt", ".cfg", ".toml", ".j2", ".csv"}


def _census_hits(root: Path) -> list[str]:
    """Every `path:line: text` under src, scripts, tests and docs of `root` that the census matcher flags."""
    hits = []
    for sub in ("src", "scripts", "tests", "docs", "test_data"):
        if not (root / sub).is_dir():
            continue
        for p in sorted((root / sub).rglob("*")):
            rel = p.relative_to(root).as_posix()
            if not p.is_file() or p.suffix not in _SUFFIXES or rel in _QUOTING_FILES:
                continue
            exact_only = rel in _RETIRED_WORD_LISTS
            if any(s in rel for s in _SKIP_PARTS):
                continue
            for lineno, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
                if (_EXACT_RE if exact_only else _CENSUS).search(line) and _RAW_SOLVER_WORDING not in line:
                    hits.append(f"{rel}:{lineno}: {line.strip()[:90]}")
    return hits


def test_census_matcher_fires_on_the_property_not_the_token(tmp_path):
    """Property: the census flags the wildcard, prose and CF-attribute forms, and passes the names that stay.

    Class: a planted tree with a wildcard line, a prose line, a JSON attribute line, an exact-name line, and four
    lines that must NOT match (the function `return_fpath_wlevels`, the in-memory key `"wlevel"`, the boundary
    condition `bc_water_level`, and a new name).
    Why required: a census returning zero is evidence only if the same matcher is shown non-zero on a violating
    input differing from the live one in the property under suspicion (a form, not a token) and zero on a clean
    input in a different correct state, so a broken matcher and a clean tree cannot read alike.
    Kills it: narrowing the matcher to the exact names (the three non-exact planted lines are missed), or widening
    it to flag `wlevel` alone (the three bystander lines are flagged).
    Passes under: a matcher implemented as a tokenizer over identifiers and comments instead of a regex.
    """
    for d in ("src", "scripts", "tests", "docs"):
        (tmp_path / d).mkdir()
    (tmp_path / "src" / "a.py").write_text('x = "max_wlevel_m"\n')
    (tmp_path / "docs" / "b.md").write_text("variables `wlevel_*` and the Peak Water Level map\n")
    (tmp_path / "tests" / "c.json").write_text('{"standard_name": "sea_surface_height_above_geoid"}\n')
    (tmp_path / "src" / "ok.py").write_text(
        'f = return_fpath_wlevels\nd = {"wlevel": 1}\nbc_water_level = 2\nn = "flood_depth_last_tstep"\n'
    )
    found = sorted({h.split(":")[0] for h in _census_hits(tmp_path)})
    assert found == ["docs/b.md", "src/a.py", "tests/c.json"], found


def test_no_retired_depth_name_in_any_form_remains():
    """Property: no forbidden depth name, wildcard fragment, prose form or sea-surface standard name remains.

    Class: every text file under src, scripts, tests, docs and test_data, minus the bundle docs, the
    `legacy_layouts` fixtures, the goldens and this file; the wildcard, prose and JSON-attribute forms are the
    members of the class an exact-name census matcher (exact names, `src scripts tests docs`) does not reach.
    Why required: an exact-name census closes on names; a manifest recording the forbidden `standard_name`
    under the new variable name, or a docstring saying "peak water level", is invisible to it.
    Kills it: any forbidden name or form left in a file.
    Passes under: a tree with every such line rewritten to the new names or to neutral prose.
    """
    assert _census_hits(REPO) == []


def test_the_exclusion_list_admits_the_migration_files_only_and_flags_a_retired_name_anywhere_else(tmp_path):
    """Property: the depth census excludes the migration and its two tests and nothing near them.

    Class: one forbidden depth name planted in each allowed path (not flagged) and in four near-miss paths (flagged).
    Why required: `V0028` quotes the forbidden names on purpose; an exclusion wider than its exact path would hide a
    stray occurrence beside it.
    Kills it: an exclusion by directory or prefix; a dropped file.
    Passes under: any exclusion exact on the relative path.
    """
    name = "max_wlevel" + "_m = 1\n"
    flagged = [
        "src/hhemt/version_migration/versions/V0027__other.py",
        "src/hhemt/version_migration/context.py",
        "tests/test_version_migration_v0027.py",
        "tests/test_review_v0028_migration_extra.py",
    ]
    for rel in [*_MIGRATION_AND_ITS_TESTS, *flagged]:
        f = tmp_path / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(name)
    assert sorted(h.split(":")[0] for h in _census_hits(tmp_path)) == sorted(flagged)
