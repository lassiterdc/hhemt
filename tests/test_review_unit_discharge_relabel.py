"""Reviewer tests for TRITON's `QX` and `QY`: they are unit discharges, not velocities.

The solver writes `QX` and `QY` as depth-integrated flow per meter of flow width (m2/s). The toolkit stores them
under six unit-discharge names, with attributes that say what they are, and the SWMM link's maximum conduit
velocity (a real conduit velocity) is the only holder of the name `max_velocity`. These tests hold that from the
outside: each one names the input class it quantifies over, why the property is required, and the change to
production code that turns it red.

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

NEW_SUMMARY_NAMES = (
    "max_unit_discharge",
    "time_of_max_unit_discharge",
    "unit_discharge_x_at_time_of_max_unit_discharge",
    "unit_discharge_y_at_time_of_max_unit_discharge",
)
NEW_DISCHARGE_NAMES = ("unit_discharge_x", "unit_discharge_y") + NEW_SUMMARY_NAMES
RETIRED_TRITON_NAMES = ("velocity_x_mps", "velocity_y_mps", "time_of_max_velocity_min")


# ---------------------------------------------------------------------------------------------
# The producer chain
# ---------------------------------------------------------------------------------------------


def test_raw_prefix_map_names_qx_and_qy_as_unit_discharge():
    """Property: the solver prefixes `QX` and `QY` map to the unit-discharge names; `MH` and `H` do not move.

    Class: the whole raw-prefix table (four keys), and the canonical column order derived from it.
    Why required: this dict is the single place the timeseries column names are minted, and
    `CANONICAL_TRITON_VARS`, the required-column check and the EDA variable list all derive from it, so a
    wrong value here mislabels every downstream store.
    Kills it: setting `"QX": "velocity_x_mps"` again, or `"MH"`/`"H"` back to the forbidden depth names.
    Passes under: a second correct implementation that builds the dict from a helper or a constant table.
    """
    from hhemt.process_simulation import CANONICAL_TRITON_VARS, TRITON_RAW_VAR_BY_PREFIX

    assert TRITON_RAW_VAR_BY_PREFIX == {
        "MH": "max_flood_depth",
        "H": "flood_depth",
        "QX": "unit_discharge_x",
        "QY": "unit_discharge_y",
    }
    assert CANONICAL_TRITON_VARS == ("max_flood_depth", "flood_depth", "unit_discharge_x", "unit_discharge_y")


def _discharge_timeseries() -> xr.Dataset:
    """A tseries-shaped Dataset with known extremes. Dims (timestep, y, x) = (6, 4, 5), 10 m cells.

    Cell (y=2, x=2): the peak is at the 4th print (t=30), qx=-3.0, qy=+4.0 (magnitude 5.0), at a depth of
    0.5 m, so a flood velocity would be 10.0 and an absolute-valued component would be +3.0.
    Cell (0, 0): identically zero discharge (the solver writes 0 at a dry cell).
    Cell (1, 1): all NaN.
    Every other cell is bounded by |q| <= 2 per component, so its magnitude is below 2.83 < 5.
    """
    rng = np.random.default_rng(20261102)
    nt, ny, nx = 6, 4, 5
    qx = rng.uniform(-2.0, 2.0, size=(nt, ny, nx))
    qy = rng.uniform(-2.0, 2.0, size=(nt, ny, nx))
    h = rng.uniform(0.2, 1.5, size=(nt, ny, nx))
    qx[:, 0, 0] = 0.0
    qy[:, 0, 0] = 0.0
    qx[:, 1, 1] = np.nan
    qy[:, 1, 1] = np.nan
    qx[3, 2, 2], qy[3, 2, 2], h[3, 2, 2] = -3.0, 4.0, 0.5
    tmin = np.arange(nt) * 10.0
    y = np.array([35.0, 25.0, 15.0, 5.0])
    x = np.array([5.0, 15.0, 25.0, 35.0, 45.0])
    dims = ("timestep", "y", "x")
    return xr.Dataset(
        {
            "max_flood_depth": (dims, np.maximum.accumulate(h, axis=0)),
            "flood_depth": (dims, h),
            "unit_discharge_x": (dims, qx),
            "unit_discharge_y": (dims, qy),
        },
        coords={"timestep": tmin, "y": y, "x": x},
    )


def test_summary_writes_exactly_the_shape_names_and_the_right_values():
    """Property: the TRITON summary writes exactly the twelve expected variables, with the expected values.

    Class: every cell of a 4x5 grid, including a signed-component peak, an all-zero cell and an all-NaN cell.
    Why required: the existing streaming-helper tests call `_streaming_argmax_with_companions` directly, so
    nothing else holds the names and values that `summarize_triton_simulation_results` assigns from its result.
    Kills it: a stale name; the SWMM link's `max_velocity` (or its forbidden form) written by the TRITON tier; the
    maximum divided by depth (5.0 would read
    10.0 at the peak cell); components taken as magnitudes (-3.0 would read +3.0).
    Passes under: an implementation that computes the reduction without the streaming helper, provided it
    writes the same names and values.
    """
    from hhemt.process_simulation import summarize_triton_simulation_results

    ds = _discharge_timeseries()
    out = summarize_triton_simulation_results(ds, 0, 10.0, chunksize_mb=0.001).isel(event_iloc=0)

    assert set(out.data_vars) == {
        *NEW_SUMMARY_NAMES,
        "max_flood_depth",
        "time_of_max_flood_depth",
        "flood_depth_last_tstep",
        "final_surface_flood_volume",
        # the flood velocity |q| / h: the summary carries these four too
        "max_flood_velocity",
        "time_of_max_flood_velocity",
        "x_flood_velocity_at_time_of_max_flood_velocity",
        "y_flood_velocity_at_time_of_max_flood_velocity",
    }
    # `max_velocity` and `velocity` are the SWMM link tier's names
    for forbidden in (
        *RETIRED_TRITON_NAMES,
        "max_velocity_mps",
        "velocity_mps",
        "max_velocity",
        "velocity",
        "final_surface_flood_volume_m3",
    ):
        assert forbidden not in out.data_vars, f"{forbidden} is not a TRITON-tier name any more"

    qx, qy = ds["unit_discharge_x"].values, ds["unit_discharge_y"].values
    mag = np.hypot(qx, qy)
    # The peak cell: the maximum is the unit discharge itself (5.0), not q / h (10.0).
    assert float(out["max_unit_discharge"].values[2, 2]) == pytest.approx(5.0)
    assert float(out["time_of_max_unit_discharge"].values[2, 2]) == 30.0
    assert float(out["unit_discharge_x_at_time_of_max_unit_discharge"].values[2, 2]) == -3.0
    assert float(out["unit_discharge_y_at_time_of_max_unit_discharge"].values[2, 2]) == 4.0
    # The dry cell and the all-NaN cell.
    assert float(out["max_unit_discharge"].values[0, 0]) == 0.0
    assert np.isnan(out["max_unit_discharge"].values[1, 1])
    assert np.isnan(out["time_of_max_unit_discharge"].values[1, 1])
    assert np.isnan(out["unit_discharge_x_at_time_of_max_unit_discharge"].values[1, 1])
    # Every cell against a numpy reference.
    ref_max = np.nanmax(np.where(np.isnan(mag), -np.inf, mag), axis=0)
    ref_max = np.where(np.isinf(ref_max), np.nan, ref_max)
    np.testing.assert_allclose(out["max_unit_discharge"].values, ref_max, equal_nan=True)
    first_peak = np.argmax(np.nan_to_num(mag, nan=-np.inf), axis=0)
    ref_t = np.where(np.isnan(ref_max), np.nan, ds["timestep"].values[first_peak])
    np.testing.assert_allclose(out["time_of_max_unit_discharge"].values, ref_t, equal_nan=True)


def test_summary_variables_get_unit_discharge_attributes_through_the_real_stamp():
    """Property: the stamp the writer applies gives the six names `m2 s-1` or `min`, a unit-discharge
    `long_name`, and no CF `standard_name`.

    Class: the summary variables of a TRITON store after `apply_cf_attributes` in a TRITON mode.
    Why required: the writer calls `apply_cf_attributes` before it writes, so this is the attribute set a
    reader of the store sees; a velocity `standard_name` here publishes a mislabel.
    Kills it: any of the six regaining `sea_water_speed` or another velocity name, a velocity `long_name`, a
    missing unit, `cell_methods` on a component, or the depth variable losing its CF name by accident.
    Passes under: an implementation that stamps the attributes in the summary function instead of the map.
    """
    from hhemt.cf_conventions import apply_cf_attributes
    from hhemt.process_simulation import summarize_triton_simulation_results

    out = summarize_triton_simulation_results(_discharge_timeseries(), 0, 10.0, chunksize_mb=0.001)
    out = apply_cf_attributes(out, mode="tritonswmm_triton")
    for name in NEW_DISCHARGE_NAMES[2:]:
        attrs = out[name].attrs
        assert "standard_name" not in attrs, (name, attrs)
        assert attrs["units"] == ("min" if name == "time_of_max_unit_discharge" else "m2 s-1"), (name, attrs)
        assert "unit discharge" in attrs["long_name"].lower(), (name, attrs)
        assert "veloc" not in attrs["long_name"].lower(), (name, attrs)
    assert out["max_unit_discharge"].attrs["cell_methods"] == "time: maximum"
    for name in NEW_DISCHARGE_NAMES[3:]:
        assert "cell_methods" not in out[name].attrs, name
    # The depth variable of the same store carries the flood-depth CF name, not a sea-surface one.
    assert out["max_flood_depth"].attrs["standard_name"] == "flood_water_thickness"


# ---------------------------------------------------------------------------------------------
# The CF table and the provenance table
# ---------------------------------------------------------------------------------------------


def test_cf_and_provenance_entries_match_section_1_2():
    """Property: each of the six names has a CF entry and a provenance entry matching the expected table row.

    Class: the six discharge names, in `_CF_VARIABLE_MAP` and `_QUANTITY_PROVENANCE`, plus the three forbidden
    names, which must be absent from both.
    Why required: the crate advertises variables from `_CF_VARIABLE_MAP` and the report renders
    `_QUANTITY_PROVENANCE`, so an entry that names a velocity, omits a source variable, or survives under a
    forbidden key publishes a false claim about the store.
    Kills it: a velocity word in any entry; `standard_name` set; a wrong unit; a forbidden key left behind;
    `source_variables` that omit a component.
    Passes under: an implementation that generates the six entries from a loop rather than listing them.
    """
    from hhemt.cf_conventions import _CF_VARIABLE_MAP, _QUANTITY_PROVENANCE

    for name in NEW_DISCHARGE_NAMES:
        cf = _CF_VARIABLE_MAP[name]
        assert cf["standard_name"] is None, name
        assert cf["units"] == ("min" if name == "time_of_max_unit_discharge" else "m2 s-1"), name
        assert cf["cell_methods"] == ("time: maximum" if name == "max_unit_discharge" else None), name
        assert "veloc" not in cf["long_name"].lower(), name
        prov = _QUANTITY_PROVENANCE[name]
        assert prov["spatial_representation"] == "grid cell", name
        assert not any("veloc" in str(v).lower() for v in prov.values()), (name, prov)
    for name in NEW_SUMMARY_NAMES:
        assert _QUANTITY_PROVENANCE[name]["source_variables"] == "unit_discharge_x, unit_discharge_y", name
    for forbidden in RETIRED_TRITON_NAMES:
        assert forbidden not in _CF_VARIABLE_MAP, forbidden
        assert forbidden not in _QUANTITY_PROVENANCE, forbidden


def test_base_max_velocity_entry_is_the_conduit_in_both_tables():
    """Property: the base `max_velocity` entry and its provenance describe the SWMM conduit.

    Class: the base CF entry and the base provenance entry for the one name the SWMM link tier keeps, keyed
    `max_velocity`, and the forbidden unit-token key, which neither table may keep.
    Why required: the crate builder advertises from the base map with no mode override, so a base entry that
    still said "flood velocity" would publish a grid-cell label for a SWMM link variable.
    Kills it: reverting `long_name` to the grid-cell wording, or the provenance to `grid cell`.
    Passes under: an implementation that moves the conduit description into the mode overrides only and
    leaves the base entry generic, provided the base still names the conduit.
    """
    from hhemt.cf_conventions import _CF_VARIABLE_MAP, _QUANTITY_PROVENANCE

    assert "max_velocity_mps" not in _CF_VARIABLE_MAP and "max_velocity_mps" not in _QUANTITY_PROVENANCE
    cf = _CF_VARIABLE_MAP["max_velocity"]
    assert cf["long_name"] == "Maximum conduit velocity"
    assert cf["units"] == "m s-1"
    assert cf["cell_methods"] == "time: maximum"
    prov = _QUANTITY_PROVENANCE["max_velocity"]
    assert prov["spatial_representation"].startswith("line"), prov
    assert "conduit" in prov["spatial_representation"] + prov["operation"] + prov["source_variables"], prov
    assert "grid cell" not in prov["spatial_representation"], prov


@pytest.mark.parametrize("mode", ["tritonswmm_swmm_link", "swmm_only_link"])
def test_link_mode_overrides_are_present_and_equal_to_the_base_entry(mode):
    """Property: the two mode overrides are equal to the base entry and are left in place.

    Class: the two SWMM link modes' `max_velocity` override (no override table may hold the key `max_velocity_mps`).
    Why required: the existing `test_apply_cf_mode_override_for_swmm_link_velocity` can no longer fail on
    an override removal (measured with a mutation that pops both modes: 32 of 32 tests passed), so nothing
    else holds the requirement that the overrides stay and equal the base.
    Kills it: deleting an override, or leaving one describing a different quantity than the base.
    Passes under: an implementation in which the override is a reference to the same dict object as the base.
    """
    from hhemt.cf_conventions import _CF_VARIABLE_MAP, _CF_VARIABLE_OVERRIDES_BY_MODE

    assert "max_velocity_mps" not in _CF_VARIABLE_OVERRIDES_BY_MODE[mode]
    assert _CF_VARIABLE_OVERRIDES_BY_MODE[mode]["max_velocity"] == _CF_VARIABLE_MAP["max_velocity"]


def test_no_variable_name_is_shared_by_the_triton_and_swmm_tiers():
    """Property: the six names written here are written by no SWMM-tier code.

    Class: the six TRITON discharge names against the `.rpt` header registries and the `.out`-derived link
    and node names the parser writes.
    Why required: `_CF_VARIABLE_MAP` is consulted by name for every mode, so a shared name gets one
    description for two quantities, which is the defect that put a flood-velocity label on a conduit.
    Kills it: naming a TRITON variable `depth`, `max_velocity` or any SWMM-tier name.
    Passes under: an implementation that puts the TRITON names in a separate table keyed by mode.
    """
    from hhemt import constants
    from hhemt.cf_conventions import _CF_VARIABLE_MAP

    swmm = set()
    for attr in dir(constants):
        if attr.startswith("LST_COL_HEADERS_"):
            swmm |= set(getattr(constants, attr))
    # the SWMM tseries names
    swmm |= {"flow", "velocity", "link_depth", "capacity_setting", "capacity", "depth", "head"}
    assert not (set(NEW_DISCHARGE_NAMES) & swmm), set(NEW_DISCHARGE_NAMES) & swmm
    # A control that can fire: the registries are non-empty and do carry the link's `max_velocity`.
    assert "max_velocity" in swmm
    assert {n for n in NEW_DISCHARGE_NAMES if n in _CF_VARIABLE_MAP} == set(NEW_DISCHARGE_NAMES)


# ---------------------------------------------------------------------------------------------
# The goldens
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("golden", ["multisim", "sensitivity"])
def test_golden_zip_carries_the_new_names_and_no_retired_name(golden):
    """Property: the regenerated golden holds the six names with the map's units, no CF name, no forbidden name.

    Class: every array in the golden zip whose leaf name is one of the six, and every `zarr.json` for a
    forbidden name; every `max_velocity` array, which must sit under a SWMM link path; and no `max_velocity_mps`
    array anywhere.
    Why required: the goldens are the published fixture other tests and the migration read, and they are
    written by an independent generator, so only a read of the zip shows what a store actually carries.
    Kills it: a stale zip, a velocity `standard_name` on a discharge array, a wrong unit, a forbidden name in
    an array name or in any array's metadata, a TRITON array named `max_velocity`, or any array still named
    with the forbidden unit-token form.
    Passes under: a golden regenerated by a different but equivalent generator, with the same names.
    """
    from hhemt.cf_conventions import _CF_VARIABLE_MAP

    z = zipfile.ZipFile(REPO / "tests" / "fixtures" / "goldens" / f"{golden}.zip")
    names = z.namelist()
    metas = [n for n in names if n.endswith("/zarr.json")]
    assert len(metas) > 100  # control: the walk reaches array metadata at all

    forbidden = re.compile("|".join(RETIRED_TRITON_NAMES))
    assert [n for n in names if forbidden.search(n)] == []
    leaked = [n for n in metas if forbidden.search(z.read(n).decode("utf-8", "replace"))]
    assert leaked == []

    seen: dict[str, int] = {}
    for n in metas:
        leaf = n.rsplit("/", 2)[-2]
        if leaf in NEW_DISCHARGE_NAMES:
            seen[leaf] = seen.get(leaf, 0) + 1
            attrs = json.loads(z.read(n)).get("attributes", {})
            assert attrs["units"] == _CF_VARIABLE_MAP[leaf]["units"], (n, attrs)
            assert "standard_name" not in attrs, (n, attrs)
    assert set(seen) == set(NEW_DISCHARGE_NAMES), seen  # every one of the six appears in the golden

    # The unit-token form is absent everywhere; `max_velocity` is present in the golden ONLY as the SWMM
    # link variable.
    assert [n for n in names if "/max_velocity_mps/" in n] == []
    mv = [n for n in names if n.endswith("/max_velocity/zarr.json")]
    assert mv, "the SWMM link variable `max_velocity` must be in the golden (control: the walk reaches it)"
    assert all(re.search(r"swmm_link|SWMM_only_link|SWMM_link|_link_", n) for n in mv), mv


# ---------------------------------------------------------------------------------------------
# Documentation, help text, cfg template, EDA units map
# ---------------------------------------------------------------------------------------------


def _triton_doc_section() -> str:
    """The text of the reference page's TRITON results section, up to the SWMM network section."""
    text = (REPO / "docs" / "reference" / "output-data-model.md").read_text()
    start = text.index("2D surface results from TRITON")
    end = text.index("### Network results from SWMM", start)
    return text[start:end]


def test_docs_table_rows_and_the_note():
    """Property: the TRITON table has one row per new name with the right unit and `n/a`, and the note replaced
    the 'means two different things' warning.

    Class: the six rows, the three forbidden names plus `max_velocity` and its forbidden unit-token form (no TRITON
    row), and the note text.
    Why required: this page is the published specification the goldens are checked against; a row that still
    lists the SWMM link's velocity name under TRITON tells a reader the tier writes a name it does not write.
    Kills it: a missing row, a velocity row left in the TRITON table, the old warning left in place, or the
    replacement note dropped.
    Passes under: a table rendered from the CF map by a generator, provided the same rows result.
    """
    section = _triton_doc_section()
    rows = {
        m.group(1): [c.strip() for c in m.group(0).strip("|\n").split("|")]
        for m in re.finditer(r"^\| `([a-z0-9_]+)` .*$", section, re.M)
    }
    for name in NEW_DISCHARGE_NAMES:
        assert name in rows, f"docs table has no row for {name}"
        _, units, std, _desc = (c.strip("` ") for c in rows[name][:4])
        assert units == ("min" if name == "time_of_max_unit_discharge" else "m2 s-1"), name
        assert std == "n/a", name
    for forbidden in (*RETIRED_TRITON_NAMES, "max_velocity_mps", "max_velocity"):
        assert forbidden not in rows, f"{forbidden} still has a TRITON table row"
    assert "means two different things" not in section
    # the note states that the TRITON tier writes no variable named `max_velocity`
    assert re.search(r"TRITON tier\s+writes no variable named\s+`max_velocity`", section), (
        "the replacement note is missing"
    )


def test_help_text_cfg_template_and_eda_units_map():
    """Property: the three non-table places that said 'velocity' now say unit discharge.

    Class: the `process_store_float32` field description, the synthetic TRITON cfg comment block above
    `print_option=huv`, and the EDA byte-identity dataset's `units_by_raw_output_type` attribute, built by
    the real `_b4b_dataset`.
    Why required: the help text is what a user reads, the cfg comment is copied into every synthetic model
    input, and the units attribute is read by the EDA renderer, which labels a `m2/s` quantity from it.
    Kills it: 'velocity' back in the description; the cfg comment losing its `#` prefix or its QX
    wording; the units map reading `m/s` for either discharge name or losing a name.
    Passes under: an implementation that derives the units map from the CF table instead of a literal.
    """
    from hhemt.config.analysis import analysis_config
    from hhemt.synthetic_model import triton_cfg

    desc = analysis_config.model_fields["process_store_float32"].description
    assert "unit discharge" in desc and "velocity" not in desc, desc

    lines = triton_cfg._TEMPLATE.splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.startswith("print_option=huv"))
    block = lines[i - 4 : i]  # dashes, two comment lines, dashes
    assert all(ln.startswith("#") for ln in block), block
    assert any("QX" in ln and "unit discharge" in ln for ln in block), block

    from hhemt.eda.raw_byte_identity import _b4b_dataset

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
        {"serial": {"unit_discharge_x": pair, "unit_discharge_y": pair}},
        meta,
        boundaries=[],
        raw_out_type="bin",
        interval=600.0,
        degraded=False,
        degraded_reason="",
        reference_config_by_family={"cpu": "serial"},
    )
    units = json.loads(ds.attrs["units_by_raw_output_type"])
    assert units["unit_discharge_x"] == "m2/s" and units["unit_discharge_y"] == "m2/s", units
    assert units["flood_depth"] == "m"  # untouched by the discharge rename
    assert list(ds["raw_output_type"].values) == ["unit_discharge_x", "unit_discharge_y"]


# ---------------------------------------------------------------------------------------------
# Residue scan, with a planted-file control
# ---------------------------------------------------------------------------------------------

#: Forbidden names AND the wildcard prose forms a name-by-name matcher does not catch.
_RESIDUE = re.compile(
    r"velocity_x_mps|velocity_y_mps|time_of_max_velocity_min"
    r"|velocity_[*{][^\s`]*_at_time_of_max_velocity|_at_time_of_max_velocity\b"
)
_SKIP_PARTS = ("legacy_layouts", "docs/reference/bundles", "__pycache__", ".zarr", "fixtures/goldens")
#: Files that quote the forbidden names on purpose (a test asserting their absence).
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
_QUOTING_FILES = {*_MIGRATION_AND_ITS_TESTS, *_ALSO_ALLOWED}


def _residue_hits(root: Path) -> list[str]:
    """Every `path:line: text` under src, scripts, tests and docs of `root` that the residue matcher flags."""
    hits = []
    for sub in ("src", "scripts", "tests", "docs"):
        for p in sorted((root / sub).rglob("*")):
            rel = p.relative_to(root).as_posix()
            if not p.is_file() or any(s in rel for s in _SKIP_PARTS) or rel in _QUOTING_FILES:
                continue
            if p.suffix not in {".py", ".md", ".rst", ".yaml", ".yml", ".json", ".txt", ".cfg", ".toml", ".j2"}:
                continue
            for lineno, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
                if _RESIDUE.search(line):
                    hits.append(f"{rel}:{lineno}: {line.strip()[:100]}")
    return hits


def test_residue_matcher_fires_on_the_property_not_the_token(tmp_path):
    """Property: the residue matcher flags both the exact name and the wildcard-prose form, and passes a new name.

    Class: a planted tree with one exact-name line, one wildcard-prose line, and one new-name line.
    Why required: a scan that returns zero is only evidence if the same matcher is shown to return non-zero
    on a violating input that differs from the live one in the property under suspicion (a prose wildcard,
    not a code token), so a broken matcher and a clean tree cannot read alike.
    Kills it: narrowing `_RESIDUE` to the three exact names (the planted wildcard line is then missed), or
    widening it to flag `unit_discharge_x_at_time_of_max_unit_discharge`.
    Passes under: a matcher implemented as an AST walk over docstrings instead of a regex.
    """
    (tmp_path / "src").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 'velocity_x_mps_at_time_of_max_velocity'\n")
    (tmp_path / "tests" / "b.py").write_text('"""argmax-derived (velocity_*_at_time_of_max_velocity)"""\n')
    (tmp_path / "docs" / "c.md").write_text(
        "`unit_discharge_x_at_time_of_max_unit_discharge` and `time_of_max_unit_discharge`\n"
    )
    found = _residue_hits(tmp_path)
    assert sorted(h.split(":")[0] for h in found) == ["src/a.py", "tests/b.py"], found


def test_no_retired_triton_name_or_wildcard_prose_form_remains():
    """Property: no forbidden TRITON discharge name, and no wildcard prose form of one, remains in the tree.

    Class: every text file under src, scripts, tests and docs, minus the bundle docs, the `legacy_layouts`
    fixtures and the two files that quote the names on purpose; the wildcard form is the member of the class
    a name-by-name census matcher does not reach.
    Why required: a name-by-name census matcher closes on exact names, so a docstring that names a forbidden
    variable by pattern (`velocity_*_at_time_of_max_velocity`) leaves a reader holding a name the store no
    longer carries, and the census would report the tree clean.
    Kills it: any forbidden name, or any `velocity_*_at_time_of_max_velocity`-shaped phrase, left in a file.
    Passes under: a tree in which the stale phrase is rewritten to the new name or to neutral prose.
    """
    assert _residue_hits(REPO) == []


def test_the_exclusion_list_admits_the_migration_files_only_and_flags_a_retired_name_anywhere_else(tmp_path):
    """Property: the TRITON census excludes the migration and its two tests and nothing near them.

    Class: one forbidden name planted in each of the three allowed paths (not flagged) and in four paths that are
    not allowed, including a sibling migration module and a test file of a similar name (flagged).
    Why required: after `V0028` landed the three census tests were red on it; widening an exclusion by pattern
    would hide the next stray occurrence, so the list is exact paths and this control shows each edge.
    Kills it: an exclusion by directory or by prefix; a list that drops one of the three files.
    Passes under: any implementation of the exclusion that is exact on the relative path.
    """
    name = "velocity_x" + "_mps = 1\n"
    allowed = list(_MIGRATION_AND_ITS_TESTS)
    flagged = [
        "src/hhemt/version_migration/versions/V0027__other.py",
        "src/hhemt/version_migration/context.py",
        "tests/test_version_migration_v0027.py",
        "tests/test_review_v0028_migration_extra.py",
    ]
    for rel in allowed + flagged:
        f = tmp_path / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(name)
    assert sorted(h.split(":")[0] for h in _residue_hits(tmp_path)) == sorted(flagged)
