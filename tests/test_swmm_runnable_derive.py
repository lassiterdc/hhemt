"""Derivation of the three SWMM role models from one runnable model (``hhemt.swmm_runnable``).

Pure text, laptop tier. The engine-open arms at the end open a derived role in SWMM (``swmm_open`` and
``swmm_start`` with no routing step, never ``swmm_step``), which the developer ruled admissible on the laptop
while each arm takes under 0.15 s (DR-10).
"""

from __future__ import annotations

import re

import pytest
from swmm.toolkit import solver

from hhemt.exceptions import ConfigurationError, ProcessingError
from hhemt.swmm_runnable import (
    DUMMY_NODE_PREFIX,
    FORCING_OPTION_KEYS,
    SECTION_ROLE_TABLE,
    SWMM_ROLES,
    _section_name,
    derive_role_model,
    rewrite_forcing,
)
from tests.fixtures.swmm_runnable_inputs import (
    EVENT_FILL,
    FIXED_WINDOW,
    LEVEL_SERIES,
    MATCHED_OUTFALL,
    RAIN_GAUGE,
    SOURCE_RAINGAGES,
    SOURCE_TIMESERIES,
    runnable_source_text,
    runnable_source_with_extras,
)

# The 57 entries of ``SectWords[]`` (external/swmm/src/solver/keywords.c, identical in the standalone SWMM
# 5.2.4 build and TRITON at faba2232), each resolved through text.h, in SectWords order.
SECTWORDS = (
    "[TITLE [OPTION [FILE [RAINGAGE [TEMPERATURE [EVAP [SUBCATCHMENT [SUBAREA [INFIL [AQUIFER [GROUNDWATER [SNOWPACK "
    "[JUNC [OUTFALL [STORAGE [DIVIDER [CONDUIT [PUMP [ORIFICE [WEIR [OUTLET [XSECT [TRANSECT [LOSS [CONTROL [POLLUT "
    "[LANDUSE [BUILDUP [WASHOFF [COVERAGE [INFLOW [DWF [PATTERN [RDII [HYDROGRAPH [LOADING [TREATMENT [CURVE "
    "[TIMESERIES [REPORT [COORDINATE [VERTICES [POLYGON [LABEL [SYMBOL [BACKDROP [TAG [PROFILE [MAP [LID_CONTROL "
    "[LID_USAGE [GWF [ADJUSTMENT [EVENT [STREET [INLET_USAGE [INLET"
).split()

# The section table of the approved shape, read from its markdown by code: the first letter of each cell.
SHAPE_CODES = """\
TITLE K K K|OPTIONS R R R|FILES K K K|RAINGAGES R D R|TEMPERATURE K K K|EVAPORATION K K K|SUBCATCHMENTS K D K|\
SUBAREAS K D K|INFILTRATION K D K|AQUIFERS K D K|GROUNDWATER K D K|SNOWPACKS K D K|JUNCTIONS K C C|OUTFALLS K C C|\
STORAGE K K D|DIVIDERS K K D|CONDUITS K C D|PUMPS K K D|ORIFICES K K D|WEIRS K K D|OUTLETS K K D|XSECTIONS K C D|\
TRANSECTS K K D|LOSSES K K D|CONTROLS K K D|POLLUTANTS K K K|LANDUSES K K K|BUILDUP K K K|WASHOFF K K K|\
COVERAGES K D K|INFLOWS K K D|DWF K K D|PATTERNS K K K|RDII K D D|HYDROGRAPHS K D D|LOADINGS K D K|TREATMENT K K K|\
CURVES K K K|TIMESERIES R F R|REPORT K F F|COORDINATES K K K|VERTICES K K D|POLYGONS K D K|LABELS K K K|\
SYMBOLS K D K|BACKDROP K K K|TAGS K F F|PROFILES K K D|MAP K K K|LID_CONTROLS K D K|LID_USAGE K D K|GWF K D K|\
ADJUSTMENTS K F K|EVENTS K K K|STREETS K K D|INLET_USAGE K K D|INLETS K K D"""
CODES: dict[str, dict[str, str]] = {}
for _row in SHAPE_CODES.split("|"):
    _name, *_letters = _row.split()
    CODES[_name] = dict(zip(SWMM_ROLES, _letters, strict=True))

# Data rows per section of the synthetic builder's own hydraulics and hydrology variants at base commit 0a6b2dc0
# (``build_templates(SyntheticModelParams(), tmp)``, one run, rows counted as non-blank non-comment lines).
BUILDER_ROWS = {
    "hydraulics": {
        "TITLE": 1, "OPTIONS": 27, "EVAPORATION": 2, "RAINGAGES": 1, "SUBCATCHMENTS": 0, "SUBAREAS": 0,
        "INFILTRATION": 0, "JUNCTIONS": 13, "OUTFALLS": 1, "CONDUITS": 12, "XSECTIONS": 12, "INFLOWS": 13,
        "CURVES": 0, "TIMESERIES": 1, "REPORT": 5, "TAGS": 0, "MAP": 2, "COORDINATES": 14,
    },
    "hydrology": {
        "TITLE": 1, "OPTIONS": 27, "EVAPORATION": 2, "RAINGAGES": 1, "SUBCATCHMENTS": 11, "SUBAREAS": 11,
        "INFILTRATION": 11, "JUNCTIONS": 13, "OUTFALLS": 1, "CONDUITS": 0, "XSECTIONS": 0, "INFLOWS": 0,
        "CURVES": 0, "TIMESERIES": 1, "REPORT": 5, "TAGS": 0, "MAP": 2, "COORDINATES": 14, "POLYGONS": 44,
    },
}  # fmt: skip
# Where a derived role differs from the builder's variant, and why. Absent: the builder keeps an empty (or
# placeholder-only) header the table drops. Rows: the builder keeps a line the table filters.
BUILDER_ABSENT = {
    "hydraulics": {"RAINGAGES", "SUBCATCHMENTS", "SUBAREAS", "INFILTRATION"},
    "hydrology": {"CONDUITS", "XSECTIONS", "INFLOWS"},
}
BUILDER_ROW_DIFFS = {
    ("hydraulics", "TIMESERIES"): (1, 0),  # builder: the ${TIMESERIES} placeholder row; derived: rain dropped
    ("hydraulics", "REPORT"): (5, 4),  # builder keeps `SUBCATCHMENTS ALL` over a model with none
    ("hydrology", "REPORT"): (5, 4),  # builder keeps `LINKS ALL` over a model with none
}


def _blocks(text: str) -> dict[str, list[str]]:
    """Section name -> the text of each occurrence, header to the next header. Independent of the module."""
    out: dict[str, list[str]] = {}
    current = None
    for line in text.splitlines(keepends=True):
        if line.strip().startswith("["):
            current = line.split()[0].strip("[]").upper()
            out.setdefault(current, []).append(line)
        elif current is not None:
            out[current][-1] += line
    return out


def _rows(block: str) -> list[list[str]]:
    lines = block.splitlines()[1:]
    return [ln.split(";")[0].split() for ln in lines if ln.strip() and not ln.lstrip().startswith(";")]


def _all_rows(text: str, name: str) -> list[list[str]]:
    return [row for block in _blocks(text).get(name, []) for row in _rows(block)]


def _row_counts(text: str) -> dict[str, int]:
    return {name: sum(len(_rows(b)) for b in blocks) for name, blocks in _blocks(text).items()}


def _options(text: str) -> dict[str, str]:
    return {row[0].upper(): " ".join(row[1:]) for row in _all_rows(text, "OPTIONS")}


def _sink() -> str:
    """One header and one comment per SWMM section, in SectWords order: no data row anywhere."""
    return "".join(f"[{name}]\n;; marker {name}\n\n" for name in CODES)


def _assert_follows_table(source: str, derived: str, role: str) -> None:
    src, got = _blocks(source), _blocks(derived)
    for name, by_role in CODES.items():
        letter = by_role[role]
        if name not in src:
            continue
        if letter == "K":
            assert got.get(name) == src[name], f"[{name}] is K in {role} and must pass through byte for byte"
        elif letter == "D":
            if name in ("CONDUITS", "XSECTIONS") and role == "hydrology" and name in got:
                assert all(row[0].startswith(DUMMY_NODE_PREFIX) for row in _all_rows(derived, name)), name
            else:
                assert name not in got, f"[{name}] is D in {role} and must be dropped"
        else:
            assert name in got, f"[{name}] is {letter} in {role} and must stay"


def test_section_table_is_total_over_sectwords_and_equals_the_shape():
    assert len(SECTWORDS) == 57
    resolved = [_section_name(prefix + "S]") for prefix in SECTWORDS]
    assert resolved == list(CODES), "SectWords order and the recorded section order differ"
    assert list(SECTION_ROLE_TABLE) == list(CODES)
    for name, by_role in SECTION_ROLE_TABLE.items():
        assert {role: cell[0] for role, cell in by_role.items()} == CODES[name], name
    # SWMM matches a header by prefix, case-insensitively, and tries INLET_USAGE before INLET.
    assert _section_name("[junc]") == "JUNCTIONS"
    assert _section_name("[Xsections]") == "XSECTIONS"
    assert _section_name("[INLET_USAGE]") == "INLET_USAGE"
    assert _section_name("[INLETS]") == "INLETS"
    assert _section_name("[FOOBAR]") is None


@pytest.mark.parametrize("role", ["hydraulics", "hydrology"])
def test_every_section_follows_its_table_code(role):
    sink = _sink()
    derived = derive_role_model(sink, role, [])
    _assert_follows_table(sink, derived, role)
    # No outfall row, so the dummy chain is added: one node of each kind and a link, all named hhemt_dummy_*.
    assert [r[0] for r in _all_rows(derived, "JUNCTIONS")] == [f"{DUMMY_NODE_PREFIX}j"]
    assert [r[0] for r in _all_rows(derived, "OUTFALLS")] == [f"{DUMMY_NODE_PREFIX}of"]
    real = runnable_source_with_extras()
    _assert_follows_table(real, derive_role_model(real, role, []), role)


def test_full_role_is_the_source_byte_for_byte_and_ignores_the_list():
    src = runnable_source_with_extras()
    assert derive_role_model(src, "full", []) == src
    assert derive_role_model(src, "full", ["dummy_outfall", "not_an_outfall"]) == src
    crlf = src.replace("\n", "\r\n")
    assert derive_role_model(crlf, "full", []) == crlf


@pytest.mark.parametrize("role", ["hydraulics", "hydrology"])
def test_derived_roles_match_the_builder_variants_at_base_commit(role):
    got = _row_counts(derive_role_model(runnable_source_text(), role, []))
    want = BUILDER_ROWS[role]
    assert set(want) - set(got) == BUILDER_ABSENT[role]
    assert set(got) <= set(want)
    for name, n in got.items():
        expected = BUILDER_ROW_DIFFS.get((role, name), (n, n))
        assert (want[name], n) == expected, f"[{name}] rows in {role}: builder {want[name]}, derived {n}"


def test_hydrology_routes_kinwave_and_every_other_option_is_unchanged():
    src = runnable_source_text()
    base = _options(src)
    assert base["FLOW_ROUTING"] == "DYNWAVE"
    assert _options(derive_role_model(src, "hydraulics", [])) == base
    hydrology = _options(derive_role_model(src, "hydrology", []))
    assert hydrology["FLOW_ROUTING"] == "KINWAVE"
    assert {k: v for k, v in hydrology.items() if k != "FLOW_ROUTING"} == {
        k: v for k, v in base.items() if k != "FLOW_ROUTING"
    }


@pytest.mark.parametrize("role", ["hydraulics", "hydrology"])
def test_listed_outfall_becomes_a_junction_and_the_dummy_chain_follows(role):
    src = runnable_source_text()
    invert = next(r[1] for r in _all_rows(src, "OUTFALLS") if r[0] == "dummy_outfall")
    highest = max(float(r[1]) for name in ("JUNCTIONS", "OUTFALLS") for r in _all_rows(src, name))
    out = derive_role_model(src, role, ["dummy_outfall"])
    assert "dummy_outfall" not in [r[0] for r in _all_rows(out, "OUTFALLS")]
    junctions = {r[0]: r[1:] for r in _all_rows(out, "JUNCTIONS")}
    assert junctions["dummy_outfall"] == [invert, "0", "0", "0", "0"]
    assert ["dummy_outfall"] == [r[0] for r in _all_rows(out, "COORDINATES") if r[0] == "dummy_outfall"]
    # The chain: junction at H+1001 with MaxDepth 1, outfall at H+1000 FREE, a 10 m conduit, a circular section.
    j, c, o = (f"{DUMMY_NODE_PREFIX}{k}" for k in ("j", "c", "of"))
    assert float(junctions[j][0]) == pytest.approx(highest + 1001) and junctions[j][1] == "1"
    assert [(r[0], float(r[1]), r[2], r[3]) for r in _all_rows(out, "OUTFALLS")] == [
        (o, pytest.approx(highest + 1000), "FREE", "NO")
    ]
    assert [r for r in _all_rows(out, "CONDUITS") if r[0] == c] == [[c, j, o, "10", "0.01", "0", "0", "0", "0"]]
    assert [r for r in _all_rows(out, "XSECTIONS") if r[0] == c] == [[c, "CIRCULAR", "1", "0", "0", "0", "1"]]
    assert not any(r[0] == c for name in ("COORDINATES",) for r in _all_rows(out, name))
    # Hydraulics keeps its 12 conduits beside the dummy; hydrology keeps only the dummy.
    assert len(_all_rows(out, "CONDUITS")) == (13 if role == "hydraulics" else 1)


@pytest.mark.parametrize("role", ["hydraulics", "hydrology"])
def test_an_unlisted_outfall_stays_and_no_dummy_is_added(role):
    out = derive_role_model(runnable_source_text(), role, [])
    assert [r[0] for r in _all_rows(out, "OUTFALLS")] == ["dummy_outfall"]
    assert DUMMY_NODE_PREFIX not in out


@pytest.mark.parametrize("role", ["hydraulics", "hydrology"])
def test_a_listed_name_that_is_not_an_outfall_is_refused(role):
    with pytest.raises(ConfigurationError, match="not_an_outfall"):
        derive_role_model(runnable_source_text(), role, ["dummy_outfall", "not_an_outfall"])
    with pytest.raises(ConfigurationError, match="J1"):  # a junction is not an outfall
        derive_role_model(runnable_source_text(), role, ["J1"])


def test_outfall_names_match_ascii_case_insensitively_as_swmm_matches_them():
    src = runnable_source_text().replace("[OUTFALLS]\n", "[OUTFALLS]\nPond_Outfall  5.0 FREE NO\n", 1)
    out = derive_role_model(src, "hydraulics", ["pond_outfall"])
    assert re.search(r"^Pond_Outfall +5\.0 +0 +0 +0 +0$", _blocks(out)["JUNCTIONS"][0], re.M)
    assert "Pond_Outfall" not in _blocks(out)["OUTFALLS"][0]
    assert [r[0] for r in _all_rows(out, "OUTFALLS")] == ["dummy_outfall"]  # one outfall remains, so no dummy chain


def test_unknown_section_and_unknown_role_are_refused():
    with pytest.raises(ProcessingError, match=r"\[FOOBAR\]"):
        derive_role_model(runnable_source_text() + "\n[FOOBAR]\nx\n", "hydraulics", [])
    with pytest.raises(ProcessingError, match="role"):
        derive_role_model(runnable_source_text(), "coupled", [])


def test_crlf_input_never_gains_a_bare_newline():
    crlf = runnable_source_with_extras().replace("\n", "\r\n")
    for role in ("hydraulics", "hydrology"):
        derived = derive_role_model(crlf, role, ["dummy_outfall"])
        assert "\n" not in derived.replace("\r\n", ""), role
        assert "\n" not in rewrite_forcing(derived, EVENT_FILL).replace("\r\n", ""), role


def test_filtered_sections_drop_only_the_rows_that_name_a_dropped_object():
    src = runnable_source_with_extras().replace("[TAGS]\n", "[TAGS]\nNode J1 a\nSubcatch S1 b\nLink C1 c\n")
    hydraulics = derive_role_model(src, "hydraulics", [])
    hydrology = derive_role_model(src, "hydrology", [])
    assert _all_rows(src, "REPORT") == [
        ["INPUT", "NO"],
        ["CONTROLS", "NO"],
        ["SUBCATCHMENTS", "ALL"],
        ["NODES", "ALL"],
        ["LINKS", "ALL"],
    ]
    assert [r[0] for r in _all_rows(hydraulics, "REPORT")] == ["INPUT", "CONTROLS", "NODES", "LINKS"]
    assert [r[0] for r in _all_rows(hydrology, "REPORT")] == ["INPUT", "CONTROLS", "SUBCATCHMENTS", "NODES"]
    assert [r[0] for r in _all_rows(hydraulics, "TAGS")] == ["Node", "Link"]
    assert [r[0] for r in _all_rows(hydrology, "TAGS")] == ["Node", "Subcatch"]
    # N-PERV names a subcatchment, so only the hydraulics role (which has none) drops it.
    assert [r[0] for r in _all_rows(hydraulics, "ADJUSTMENTS")] == ["RAINFALL"]
    assert [r[0] for r in _all_rows(hydrology, "ADJUSTMENTS")] == ["RAINFALL", "N-PERV"]
    # Rain series drop in hydraulics; other series, in either occurrence of the section, stay.
    assert {r[0] for r in _all_rows(hydraulics, "TIMESERIES")} == {"tmp_ts"}
    assert {r[0] for r in _all_rows(hydrology, "TIMESERIES")} == {RAIN_GAUGE, "tmp_ts"}


def test_storage_and_dividers_are_retyped_to_junctions_in_hydrology_only():
    extra = (
        "\n[STORAGE]\n;;Name Elev MaxDepth InitDepth Shape Params\nST1 2.5 3.0 0.5 FUNCTIONAL 0 0 1000 0 0\n"
        "\n[DIVIDERS]\n;;Name Elev DivLink Type Params\nDV1 1.5 C1 CUTOFF 2.0\n"
        "DV2 1.25 C2 WEIR 1.0 2.0 3.0 4.0 0.5 0.1 7.0\n"
    )
    src = runnable_source_text() + extra
    hydrology = derive_role_model(src, "hydrology", [])
    assert "STORAGE" not in _blocks(hydrology) and "DIVIDERS" not in _blocks(hydrology)
    junctions = {r[0]: r[1:] for r in _all_rows(hydrology, "JUNCTIONS")}
    assert junctions["ST1"] == ["2.5", "3.0", "0.5", "0", "0"]
    assert junctions["DV1"] == ["1.5", "0", "0", "0", "0"]
    assert junctions["DV2"] == ["1.25", "4.0", "0.5", "0.1", "7.0"]
    hydraulics = derive_role_model(src, "hydraulics", [])
    assert _blocks(hydraulics)["STORAGE"] == _blocks(src)["STORAGE"]
    assert "ST1" not in {r[0] for r in _all_rows(hydraulics, "JUNCTIONS")}


def _with_matched_outfall(role: str) -> str:
    """The role of the one-outfall model whose ``SYNTH_OUTFALL`` row exists and reads the record-0 level series."""
    source = runnable_source_with_extras().replace(
        "[OUTFALLS]\n", f"[OUTFALLS]\n{MATCHED_OUTFALL}  0.5 TIMESERIES {LEVEL_SERIES}\n", 1
    )
    return derive_role_model(source, role, [])


@pytest.mark.parametrize("role", ["full", "hydraulics", "hydrology"])
def test_rewrite_forcing_rewrites_the_owned_forcing_and_nothing_else(role):
    derived = _with_matched_outfall(role)
    out = rewrite_forcing(derived, EVENT_FILL)
    options = _options(out)
    assert {k: options[k] for k in FORCING_OPTION_KEYS} == {k: EVENT_FILL[k] for k in FORCING_OPTION_KEYS}
    series = [" ".join(r) for r in _all_rows(out, "TIMESERIES")]
    level = [s for s in series if s.startswith(LEVEL_SERIES)]
    assert level == [f"{LEVEL_SERIES} 03/04/2021 06:00 0.2", f"{LEVEL_SERIES} 03/04/2021 07:00 0.9"]
    assert "tmp_ts 01/01/2020 00:00 10.0" in series  # a series the toolkit does not own stays
    if role == "hydraulics":
        assert "RAINGAGES" not in _blocks(out) and not any(s.startswith(RAIN_GAUGE) for s in series)
    else:
        assert [" ".join(r) for r in _all_rows(out, "RAINGAGES")] == [EVENT_FILL["RAINGAGES"]]
        assert [s for s in series if s.startswith(RAIN_GAUGE)] == [
            f"{RAIN_GAUGE} 03/04/2021 06:{m} {v}" for m, v in (("00", "0.0"), ("20", "80.0"), ("40", "0.0"))
        ]
        assert SOURCE_TIMESERIES not in series and SOURCE_RAINGAGES not in out

    def unowned(text: str) -> list[str]:
        owned = (*FORCING_OPTION_KEYS, LEVEL_SERIES, RAIN_GAUGE)
        keep = [ln for ln in text.splitlines() if not ln.split(";")[0].split()[:1] or ln.split()[0] not in owned]
        return [ln for ln in keep if not re.match(r"^\S+ +INTENSITY ", ln)]

    assert unowned(out) == unowned(derived)
    assert rewrite_forcing(out, EVENT_FILL) == out  # the same event twice is the same text


@pytest.mark.parametrize("role", ["full", "hydraulics", "hydrology"])
def test_a_fill_without_water_level_rows_leaves_no_water_level_series(role):
    leftover = (
        runnable_source_text()
        + f"\n[TIMESERIES]\n{LEVEL_SERIES} 01/01/2020 00:00 0.4\n{LEVEL_SERIES} 01/01/2020 00:30 0.6\n"
        + "other_level 01/01/2020 00:00 0.1\n"
    )
    derived = derive_role_model(leftover, role, [])
    assert any(r[0] == LEVEL_SERIES for r in _all_rows(derived, "TIMESERIES"))  # the source carries a level series
    no_level = {**EVENT_FILL, "TIMESERIES": f"{RAIN_GAUGE} 03/04/2021 06:00 0.0"}
    out = rewrite_forcing(derived, no_level)
    assert not any(r[0] == LEVEL_SERIES for r in _all_rows(out, "TIMESERIES"))
    # Control that varies the property (the series is not the toolkit's prefix), not the call: it stays.
    assert any(r[0] == "other_level" for r in _all_rows(out, "TIMESERIES"))
    assert ("RAINGAGES" in _blocks(out)) == (role != "hydraulics")


def test_rewrite_forcing_keeps_the_spacing_and_comment_of_a_rewritten_option_row():
    derived = derive_role_model(runnable_source_text(), "hydraulics", []).replace(
        "END_TIME              00:30:00", "END_TIME              00:30:00 ; the sim end"
    )
    line = next(ln for ln in rewrite_forcing(derived, EVENT_FILL).splitlines() if ln.startswith("END_TIME"))
    assert line.startswith("END_TIME              07:00:00 ; the sim end")


def test_rewrite_forcing_refuses_a_fill_that_lacks_an_owned_key():
    derived = derive_role_model(runnable_source_text(), "hydrology", [])
    for key in (*FORCING_OPTION_KEYS, "RAINGAGES", "TIMESERIES"):
        fill = {k: v for k, v in EVENT_FILL.items() if k != key}
        with pytest.raises(ProcessingError, match=key):
            rewrite_forcing(derived, fill)


def _opens(text: str, tmp_path) -> None:
    inp = tmp_path / "m.inp"
    inp.write_text(text)
    try:
        solver.swmm_open(str(inp), str(tmp_path / "m.rpt"), str(tmp_path / "m.out"))
        solver.swmm_start(0)
        solver.swmm_end()
    finally:
        solver.swmm_close()


def test_every_derived_role_opens_and_starts_in_swmm(tmp_path):
    """The engine settles the SYMBOLS, ADJUSTMENTS and TEMPERATURE rows: the extras model carries all three."""
    for outfalls in ([], ["dummy_outfall"]):
        for role in SWMM_ROLES:
            text = rewrite_forcing(
                derive_role_model(runnable_source_with_extras(), role, outfalls),
                FIXED_WINDOW | {"RAINGAGES": SOURCE_RAINGAGES, "TIMESERIES": SOURCE_TIMESERIES},
            )
            _opens(text, tmp_path)


def test_the_open_check_can_fail(tmp_path):
    """Controls: the engine refuses a kept row that names a dropped subcatchment, and an undefined node."""
    hydraulics = derive_role_model(runnable_source_with_extras(), "hydraulics", [])
    _opens(hydraulics, tmp_path)
    with pytest.raises(Exception, match="(?i)error"):
        _opens(hydraulics.replace("[ADJUSTMENTS]\n", "[ADJUSTMENTS]\nN-PERV   S1 npat\n"), tmp_path)
    broken = re.sub(r"^(C1\s+J1\s+)J2", r"\1J_nope", hydraulics, count=1, flags=re.M)
    assert broken != hydraulics
    with pytest.raises(Exception, match="(?i)error"):
        _opens(broken, tmp_path)
