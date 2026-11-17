"""Derive the three SWMM role models from ONE plain runnable SWMM model.

A system config names a single runnable ``.inp`` (``SWMM_model``). This module turns that text into
the three models a run needs, as pure text in and text out (no file I/O, no solver, no config):

``full``
    The standalone SWMM comparison model. The source text, unchanged.
``hydraulics``
    The model the coupled TRITON-SWMM run reads: the drainage network without the subcatchment side.
``hydrology``
    The model SWMM runs first to produce runoff hydrographs: the subcatchment side without the
    network, routed KINWAVE.

``derive_role_model(source_text, role, outfalls_to_couple)`` applies :data:`SECTION_ROLE_TABLE`, one row
per SWMM section, and converts each listed outfall into a junction so it can exchange water with the TRITON
surface. ``rewrite_forcing(role_text, fill)`` then rewrites only the forcing the toolkit owns, per event.

The section domain is CLOSED. It is the 57 entries of ``SectWords[]`` in SWMM's ``keywords.c`` (identical in
the standalone 5.2.4 build and the copy vendored in TRITON at the ``triton-swmm`` head), each resolved through
``text.h``. SWMM matches a section header by PREFIX, case-insensitively (``input.c::match``), so
``[JUNCTIONS]`` and ``[junc]`` both open the junction section, and this module matches the same way.

Codes of the table: K keep, D drop, R rewrite (toolkit-owned keys or rows), F filter (drop the rows that name
an object a dropped section defines), C convert (rows added or removed by outfall conversion, storage and
divider retyping, or the dummy chain). The parenthesised text on a cell names what the code does beyond its
letter, and the first character of a cell is always the code.

Everything outside a section this module changes passes through byte for byte, line endings included.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from hhemt.exceptions import ConfigurationError, ProcessingError

SwmmRole = Literal["full", "hydraulics", "hydrology"]
SWMM_ROLES: tuple[SwmmRole, ...] = ("full", "hydraulics", "hydrology")

# Every node and link the dummy chain adds carries this prefix, so a reader that walks the nodes of a derived
# model (for example the hydrograph writer) can skip them by name.
DUMMY_NODE_PREFIX = "hhemt_dummy_"


def without_injected_dummy_chain(df):
    """``df`` without the rows (indexed by node or link name) of the dummy chain a derived model may carry.

    The chain has no ``[COORDINATES]`` rows and is a solver scaffold, so a reader that draws or counts the nodes of a
    derived model (the system overview, the DEM-resolution junction count) takes its frames through this.
    """
    return df.loc[~df.index.astype(str).str.startswith(DUMMY_NODE_PREFIX)]


# The keys of ``[OPTIONS]`` the toolkit rewrites per event. ``fill`` must carry each of them.
FORCING_OPTION_KEYS: tuple[str, ...] = (
    "START_DATE",
    "START_TIME",
    "REPORT_START_DATE",
    "REPORT_START_TIME",
    "END_DATE",
    "END_TIME",
    "REPORT_STEP",
)
# The two multi-row fill values: ``[RAINGAGES]`` rows and ``[TIMESERIES]`` rows of the rain and water-level series.
FORCING_SECTION_KEYS: tuple[str, ...] = ("RAINGAGES", "TIMESERIES")

# Each water-level record's series is written under this prefix plus the record's index. The prefix is
# reserved: a base model holds no series or node whose name begins ``hhemt_`` (V11).
WATER_LEVEL_SERIES_PREFIX = "hhemt_water_level_"

# Header prefixes in the order of SWMM's ``SectWords[]``. SWMM takes the FIRST prefix a header token starts
# with, so the order is part of the data (``[INLET_USAGE]`` must be tried before ``[INLET``).
_SECTION_PREFIXES: tuple[tuple[str, str], ...] = (
    ("[TITLE", "TITLE"),
    ("[OPTION", "OPTIONS"),
    ("[FILE", "FILES"),
    ("[RAINGAGE", "RAINGAGES"),
    ("[TEMPERATURE", "TEMPERATURE"),
    ("[EVAP", "EVAPORATION"),
    ("[SUBCATCHMENT", "SUBCATCHMENTS"),
    ("[SUBAREA", "SUBAREAS"),
    ("[INFIL", "INFILTRATION"),
    ("[AQUIFER", "AQUIFERS"),
    ("[GROUNDWATER", "GROUNDWATER"),
    ("[SNOWPACK", "SNOWPACKS"),
    ("[JUNC", "JUNCTIONS"),
    ("[OUTFALL", "OUTFALLS"),
    ("[STORAGE", "STORAGE"),
    ("[DIVIDER", "DIVIDERS"),
    ("[CONDUIT", "CONDUITS"),
    ("[PUMP", "PUMPS"),
    ("[ORIFICE", "ORIFICES"),
    ("[WEIR", "WEIRS"),
    ("[OUTLET", "OUTLETS"),
    ("[XSECT", "XSECTIONS"),
    ("[TRANSECT", "TRANSECTS"),
    ("[LOSS", "LOSSES"),
    ("[CONTROL", "CONTROLS"),
    ("[POLLUT", "POLLUTANTS"),
    ("[LANDUSE", "LANDUSES"),
    ("[BUILDUP", "BUILDUP"),
    ("[WASHOFF", "WASHOFF"),
    ("[COVERAGE", "COVERAGES"),
    ("[INFLOW", "INFLOWS"),
    ("[DWF", "DWF"),
    ("[PATTERN", "PATTERNS"),
    ("[RDII", "RDII"),
    ("[HYDROGRAPH", "HYDROGRAPHS"),
    ("[LOADING", "LOADINGS"),
    ("[TREATMENT", "TREATMENT"),
    ("[CURVE", "CURVES"),
    ("[TIMESERIES", "TIMESERIES"),
    ("[REPORT", "REPORT"),
    ("[COORDINATE", "COORDINATES"),
    ("[VERTICES", "VERTICES"),
    ("[POLYGON", "POLYGONS"),
    ("[LABEL", "LABELS"),
    ("[SYMBOL", "SYMBOLS"),
    ("[BACKDROP", "BACKDROP"),
    ("[TAG", "TAGS"),
    ("[PROFILE", "PROFILES"),
    ("[MAP", "MAP"),
    ("[LID_CONTROL", "LID_CONTROLS"),
    ("[LID_USAGE", "LID_USAGE"),
    ("[GWF", "GWF"),
    ("[ADJUSTMENT", "ADJUSTMENTS"),
    ("[EVENT", "EVENTS"),
    ("[STREET", "STREETS"),
    ("[INLET_USAGE", "INLET_USAGE"),
    ("[INLET", "INLETS"),
)

# One row per SWMM 5.2.4 section: (section, full, hydraulics, hydrology).
_TABLE_ROWS: tuple[tuple[str, str, str, str], ...] = (
    ("TITLE", "K", "K", "K"),
    ("OPTIONS", "R", "R", "R (+ FLOW_ROUTING KINWAVE)"),
    ("FILES", "K", "K", "K"),
    ("RAINGAGES", "R", "D", "R"),
    ("TEMPERATURE", "K", "K", "K"),
    ("EVAPORATION", "K", "K", "K"),
    ("SUBCATCHMENTS", "K", "D", "K"),
    ("SUBAREAS", "K", "D", "K"),
    ("INFILTRATION", "K", "D", "K"),
    ("AQUIFERS", "K", "D", "K"),
    ("GROUNDWATER", "K", "D", "K"),
    ("SNOWPACKS", "K", "D", "K"),
    ("JUNCTIONS", "K", "C", "C (+ storage and divider rows retyped)"),
    ("OUTFALLS", "K", "C", "C"),
    ("STORAGE", "K", "K", "D (retyped to JUNCTIONS)"),
    ("DIVIDERS", "K", "K", "D (retyped to JUNCTIONS)"),
    ("CONDUITS", "K", "C (+ dummy)", "D (+ dummy only)"),
    ("PUMPS", "K", "K", "D"),
    ("ORIFICES", "K", "K", "D"),
    ("WEIRS", "K", "K", "D"),
    ("OUTLETS", "K", "K", "D"),
    ("XSECTIONS", "K", "C (+ dummy)", "D (+ dummy only)"),
    ("TRANSECTS", "K", "K", "D"),
    ("LOSSES", "K", "K", "D"),
    ("CONTROLS", "K", "K", "D"),
    ("POLLUTANTS", "K", "K", "K"),
    ("LANDUSES", "K", "K", "K"),
    ("BUILDUP", "K", "K", "K"),
    ("WASHOFF", "K", "K", "K"),
    ("COVERAGES", "K", "D", "K"),
    ("INFLOWS", "K", "K", "D"),
    ("DWF", "K", "K", "D"),
    ("PATTERNS", "K", "K", "K"),
    ("RDII", "K", "D", "D"),
    ("HYDROGRAPHS", "K", "D", "D"),
    ("LOADINGS", "K", "D", "K"),
    ("TREATMENT", "K", "K", "K"),
    ("CURVES", "K", "K", "K"),
    ("TIMESERIES", "R", "F (rain series dropped; hhemt_water_level_ rows rewritten)", "R"),
    ("REPORT", "K", "F (SUBCATCHMENTS lines dropped)", "F (LINKS lines dropped)"),
    ("COORDINATES", "K", "K", "K"),
    ("VERTICES", "K", "K", "D"),
    ("POLYGONS", "K", "D", "K"),
    ("LABELS", "K", "K", "K"),
    ("SYMBOLS", "K", "D", "K"),
    ("BACKDROP", "K", "K", "K"),
    ("TAGS", "K", "F (subcatchment rows dropped)", "F (link rows dropped)"),
    ("PROFILES", "K", "K", "D"),
    ("MAP", "K", "K", "K"),
    ("LID_CONTROLS", "K", "D", "K"),
    ("LID_USAGE", "K", "D", "K"),
    ("GWF", "K", "D", "K"),
    ("ADJUSTMENTS", "K", "F (subcatchment rows dropped)", "K"),
    ("EVENTS", "K", "K", "K"),
    ("STREETS", "K", "K", "D"),
    ("INLET_USAGE", "K", "K", "D"),
    ("INLETS", "K", "K", "D"),
)

SECTION_ROLE_TABLE: dict[str, dict[str, str]] = {
    section: {"full": full, "hydraulics": hydraulics, "hydrology": hydrology}
    for section, full, hydraulics, hydrology in _TABLE_ROWS
}

_CANONICAL_ORDER = {name: i for i, (_prefix, name) in enumerate(_SECTION_PREFIXES)}

# The two comment lines SWMM writes under a section header. The coupled TRITON parser skips exactly two lines
# after each header, so a section this module creates carries two.
_NEW_SECTION_COMMENTS: dict[str, tuple[str, str]] = {
    "JUNCTIONS": (
        ";;Name           Elevation  MaxDepth   InitDepth  SurDepth   Aponded",
        ";;-------------- ---------- ---------- ---------- ---------- ----------",
    ),
    "OUTFALLS": (
        ";;Name           Elevation  Type       Stage Data       Gated    Route To",
        ";;-------------- ---------- ---------- ---------------- -------- ----------------",
    ),
    "CONDUITS": (
        ";;Name           From Node        To Node          Length   Roughness InOffset OutOffset InitFlow MaxFlow",
        ";;-------------- ---------------- ---------------- ------ --------- -------- --------- -------- -------",
    ),
    "XSECTIONS": (
        ";;Link           Shape        Geom1            Geom2      Geom3      Geom4      Barrels    Culvert",
        ";;-------------- ------------ ---------------- ---------- ---------- ---------- ---------- ----------",
    ),
    "RAINGAGES": (
        ";;Name           Format    Interval SCF      Source",
        ";;-------------- --------- ------ ------ ----------",
    ),
    "TIMESERIES": (
        ";;Name           Date       Time       Value",
        ";;-------------- ---------- ---------- ----------",
    ),
}

_LINE_SPLIT = re.compile(r"[^\r\n]*(?:\r\n|\n|\r)|[^\r\n]+")
_OPTION_ROW = re.compile(
    r"^(?P<lead>\s*)(?P<key>\S+)(?P<gap>\s+)(?P<val>[^;\r\n]*?)(?P<tail>\s*(?:;[^\r\n]*)?)(?P<eol>\r\n|\n|\r|)$"
)


@dataclass
class _Section:
    """One header and the lines under it. ``name`` is None for text before the first header."""

    name: str | None
    header: str
    lines: list[str] = field(default_factory=list)

    @property
    def eol(self) -> str:
        for line in [self.header, *self.lines]:
            for ending in ("\r\n", "\n", "\r"):
                if line.endswith(ending):
                    return ending
        return "\n"


def _fold(text: str) -> str:
    """Upper-case ASCII ``a``-``z`` and nothing else, as SWMM compares names and keywords.

    Deliberately not ``str.upper()``: it also folds non-ASCII letters, which SWMM keeps distinct.
    """
    return "".join(chr(ord(c) - 32) if "a" <= c <= "z" else c for c in text)


def _tokens(line: str) -> list[str]:
    """Split one line into tokens the way SWMM's ``getTokens`` does.

    Text from the first ``;`` is dropped first, even inside quotes. Tokens are separated by space, tab, CR or LF.
    A token that begins with ``"`` runs to the next ``"`` or the end of the line, and neither quote is part of it.
    """
    text = line.split(";", 1)[0]
    tokens: list[str] = []
    i, n = 0, len(text)
    while i < n:
        if text[i] in " \t\r\n":
            i += 1
        elif text[i] == '"':
            close = text.find('"', i + 1)
            stop = n if close == -1 else close
            tokens.append(text[i + 1 : stop])
            i = stop + 1
        else:
            j = i
            while j < n and text[j] not in " \t\r\n":
                j += 1
            tokens.append(text[i:j])
            i = j
    return tokens


def _is_data_row(line: str) -> bool:
    return bool(line.strip()) and not line.lstrip().startswith(";")


def _prefix_match(token: str, keyword: str) -> bool:
    """SWMM's ``match``: ``token`` begins with ``keyword``, ignoring case."""
    return _fold(token).startswith(_fold(keyword))


def _section_name(header_token: str) -> str | None:
    for prefix, name in _SECTION_PREFIXES:
        if _prefix_match(header_token, prefix):
            return name
    return None


def _parse(text: str) -> list[_Section]:
    sections = [_Section(name=None, header="")]
    for line in _LINE_SPLIT.findall(text):
        stripped = line.strip()
        if stripped.startswith("["):
            token = stripped.split()[0]
            name = _section_name(token)
            if name is None:
                raise ProcessingError(
                    operation="derive_role_model",
                    reason=(
                        f"section header {token!r} is not one of the 57 sections of SWMM 5.2.4, so SWMM would "
                        f"refuse this model with ERR_KEYWORD."
                    ),
                )
            sections.append(_Section(name=name, header=line))
        else:
            sections[-1].lines.append(line)
    return sections


def _render(sections: Iterable[_Section]) -> str:
    return "".join(s.header + "".join(s.lines) for s in sections)


def _first(sections: list[_Section], name: str) -> _Section | None:
    return next((s for s in sections if s.name == name), None)


def _all(sections: list[_Section], name: str) -> list[_Section]:
    return [s for s in sections if s.name == name]


def _ensure_section(sections: list[_Section], name: str, eol: str) -> _Section:
    """The first section called ``name``, created in SWMM's section order when the model has none."""
    existing = _first(sections, name)
    if existing is not None:
        return existing
    comments = _NEW_SECTION_COMMENTS.get(name, ())
    created = _Section(name=name, header=f"[{name}]{eol}", lines=[c + eol for c in comments] + [eol])
    rank = _CANONICAL_ORDER[name]
    at = next(
        (i for i, s in enumerate(sections) if s.name is not None and _CANONICAL_ORDER[s.name] > rank),
        len(sections),
    )
    sections.insert(at, created)
    if at > 0:
        _separate(sections[at - 1], eol)
    return created


def _separate(section: _Section, eol: str) -> None:
    """End ``section`` with a newline so rows appended after it never join its last line."""
    if section.lines:
        if not section.lines[-1].endswith(("\n", "\r")):
            section.lines[-1] += eol
    elif section.header and not section.header.endswith(("\n", "\r")):
        section.header += eol


def _append_rows(section: _Section, rows: Sequence[str], eol: str) -> None:
    """Add ``rows`` after the last non-blank line of the section, ahead of its trailing blank lines."""
    if not rows:
        return
    at = len(section.lines)
    while at > 0 and not section.lines[at - 1].strip():
        at -= 1
    if at > 0 and not section.lines[at - 1].endswith(("\n", "\r")):
        section.lines[at - 1] += eol
    if at == 0 and section.header and not section.header.endswith(("\n", "\r")):
        section.header += eol
    section.lines[at:at] = [row + eol for row in rows]


def _number(token: str) -> float | None:
    try:
        return float(token)
    except ValueError:
        return None


def _format_number(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _node_rows(sections: list[_Section], name: str) -> list[list[str]]:
    return [_tokens(line) for s in _all(sections, name) for line in s.lines if _is_data_row(line)]


def _highest_invert(sections: list[_Section]) -> float:
    """The highest node invert of the source, over junctions, outfalls, storage and dividers."""
    inverts = [
        v
        for name in ("JUNCTIONS", "OUTFALLS", "STORAGE", "DIVIDERS")
        for toks in _node_rows(sections, name)
        if len(toks) > 1 and (v := _number(toks[1])) is not None
    ]
    return max(inverts) if inverts else 0.0


def _rain_series_names(sections: list[_Section]) -> set[str]:
    """Series a ``[RAINGAGES]`` row of source ``TIMESERIES`` names: ``Name Format Interval SCF TIMESERIES Series``."""
    names: set[str] = set()
    for toks in _node_rows(sections, "RAINGAGES"):
        if len(toks) > 5 and _prefix_match(toks[4], "TIMESERIES"):
            names.add(_fold(toks[5]))
    return names


def _retyped_junction_row(kind: str, toks: list[str]) -> str:
    """A ``[JUNCTIONS]`` row for a storage or divider node, which the hydrology role keeps as a plain node.

    Storage: ``Name Elev MaxDepth InitDepth ...``. Divider: ``Name Elev DivLink Type Params [MaxDepth InitDepth
    SurDepth Aponded]``, where the parameter count follows the divider type.
    """
    name, elevation = toks[0], toks[1]
    depths = ["0", "0", "0", "0"]
    if kind == "STORAGE":
        depths[0:2] = [toks[2] if len(toks) > 2 else "0", toks[3] if len(toks) > 3 else "0"]
    elif len(toks) > 4:
        n_params = {"CUTOFF": 1, "OVERFLOW": 0, "TABULAR": 1, "WEIR": 3}.get(_fold(toks[3]), 0)
        tail = toks[4 + n_params :]
        depths[: len(tail[:4])] = tail[:4]
    return _junction_row(name, elevation, *depths)


def _junction_row(
    name: str, invert: str, max_depth: str = "0", init: str = "0", surcharge: str = "0", ponded: str = "0"
) -> str:
    return f"{name:<16} {invert:<10} {max_depth:<10} {init:<10} {surcharge:<10} {ponded}"


def derive_role_model(
    source_text: str,
    role: str,
    outfalls_to_couple: Sequence[str] = (),
    *,
    node_depths: Mapping[str, float] | None = None,
    series_pointers: Mapping[str, str] | None = None,
    stage_series: Mapping[str, str] | None = None,
) -> str:
    """The text of one role model derived from the runnable source model.

    ``role`` is ``full``, ``hydraulics`` or ``hydrology``. ``outfalls_to_couple`` names ``[OUTFALLS]`` rows of
    the source that become junctions in the hydraulics and hydrology roles, so they can exchange water with the
    TRITON surface. The full role never converts. When no outfall remains in a role after conversion, a dummy
    chain (junction, conduit, free outfall, all named ``hhemt_dummy_*``) is added, because SWMM needs one.
    The timing options and forcing rows are left as the source has them: ``rewrite_forcing`` owns those.

    ``node_depths`` maps a node's case-folded name to the ``MaxDepth`` its junction row takes (the ground at its DEM
    cell minus its invert). A converted outfall absent from it keeps ``MaxDepth 0``, and a junction of the source the
    mapping names takes the depth in the hydraulics and hydrology roles.
    ``series_pointers`` maps a case-folded series name to the file its ``[TIMESERIES]`` rows are replaced by, so a
    derived model holds no series data (every role, the full role included). ``stage_series`` maps a case-folded
    outfall name to the series its stage-data token is rewritten to, in the roles that keep the outfall.

    Raises
    ------
    ProcessingError
        A section header is not one of SWMM's 57 sections, or ``role`` is not a role.
    ConfigurationError
        A listed outfall is not an ``[OUTFALLS]`` row of the source (hydraulics and hydrology only).
    """
    if role not in SWMM_ROLES:
        raise ProcessingError(operation="derive_role_model", reason=f"role {role!r} is not one of {list(SWMM_ROLES)}.")
    sections = _parse(source_text)
    eol = next((s.eol for s in sections if s.name is not None), "\n")
    depths = {_fold(k): v for k, v in (node_depths or {}).items()}
    stages = {_fold(k): v for k, v in (stage_series or {}).items()}
    if role == "full":
        _retarget_stage_data(sections, stages)
        _externalize_series(sections, series_pointers or {}, eol)
        return _render(sections)

    eol = next((s.eol for s in sections if s.name is not None), "\n")
    listed = list(dict.fromkeys(outfalls_to_couple))
    listed_folded = {_fold(n) for n in listed}
    rain_series = _rain_series_names(sections)
    highest = _highest_invert(sections)
    retype_rows: list[str] = []
    converted_rows: list[str] = []
    outfall_names_left: list[str] = []
    converted: set[str] = set()

    for section in list(sections):
        name = section.name
        if name is None:
            continue
        cell = SECTION_ROLE_TABLE[name][role]
        code = cell[0]
        if name == "OUTFALLS":
            kept: list[str] = []
            for line in section.lines:
                toks = _tokens(line)
                if toks and _is_data_row(line) and _fold(toks[0]) in listed_folded:
                    converted.add(_fold(toks[0]))
                    depth = depths.get(_fold(toks[0]))
                    converted_rows.append(
                        _junction_row(
                            toks[0], toks[1] if len(toks) > 1 else "0", "0" if depth is None else _format_number(depth)
                        )
                    )
                    continue
                if toks and _is_data_row(line):
                    outfall_names_left.append(toks[0])
                kept.append(line)
            section.lines = kept
        elif code == "D":
            if name in ("STORAGE", "DIVIDERS"):
                kind = "STORAGE" if name == "STORAGE" else "DIVIDERS"
                retype_rows.extend(
                    _retyped_junction_row(kind, toks)
                    for line in section.lines
                    if _is_data_row(line) and len(toks := _tokens(line)) > 1
                )
            sections[:] = [s for s in sections if s is not section]
        elif code == "F":
            section.lines = _filter_rows(name, role, section.lines, rain_series)
        elif name == "OPTIONS" and role == "hydrology":
            _set_option(_all(sections, "OPTIONS"), "FLOW_ROUTING", "KINWAVE", eol)

    missing = [n for n in listed if _fold(n) not in converted]
    if missing:
        raise ConfigurationError(
            field="swmm_outfalls_to_couple",
            message=f"{missing} {'is' if len(missing) == 1 else 'are'} not an [OUTFALLS] row of the SWMM model.",
            fix_hint="List only names that appear in the [OUTFALLS] section of SWMM_model.",
        )

    _set_junction_depths(sections, depths)
    junction_rows = [*retype_rows, *converted_rows]
    dummy = not outfall_names_left
    if dummy:
        j, c, o = (f"{DUMMY_NODE_PREFIX}{k}" for k in ("j", "c", "of"))
        junction_rows.append(_junction_row(j, _format_number(highest + 1001), "1"))
        _append_rows(
            _ensure_section(sections, "OUTFALLS", eol),
            [f"{o:<16} {_format_number(highest + 1000):<10} FREE       NO"],
            eol,
        )
        conduits = _ensure_section(sections, "CONDUITS", eol)
        _append_rows(
            conduits, [f"{c:<16} {j:<16} {o:<16} {'10':<10} {'0.01':<10} 0          0          0          0"], eol
        )
        xsections = _ensure_section(sections, "XSECTIONS", eol)
        _append_rows(xsections, [f"{c:<16} {'CIRCULAR':<12} {'1':<16} 0          0          0          1"], eol)
    _append_rows(_ensure_section(sections, "JUNCTIONS", eol), junction_rows, eol)
    _retarget_stage_data(sections, stages)
    _externalize_series(sections, series_pointers or {}, eol)
    return _render(sections)


def _set_junction_depths(sections: list[_Section], depths: Mapping[str, float]) -> None:
    """Write ``MaxDepth``, the third token of a ``[JUNCTIONS]`` row, for each junction ``depths`` names."""
    if not depths:
        return
    for section in _all(sections, "JUNCTIONS"):
        for i, line in enumerate(section.lines):
            toks = _tokens(line) if _is_data_row(line) else []
            if len(toks) > 2 and _fold(toks[0]) in depths:
                depth = _format_number(depths[_fold(toks[0])])
                section.lines[i] = re.sub(r"^(\s*\S+\s+\S+\s+)\S+", lambda m, d=depth: m.group(1) + d, line, count=1)


def _retarget_stage_data(sections: list[_Section], stage_series: Mapping[str, str]) -> None:
    """Rewrite the stage-data token of each named ``TIMESERIES`` outfall row to its new series name."""
    if not stage_series:
        return
    for section in _all(sections, "OUTFALLS"):
        for i, line in enumerate(section.lines):
            toks = _tokens(line) if _is_data_row(line) else []
            if len(toks) > 3 and _fold(toks[0]) in stage_series and _prefix_match(toks[2], "TIMESERIES"):
                section.lines[i] = re.sub(
                    r"(?<=\s)" + re.escape(toks[3]) + r"(?=\s|;|$)", stage_series[_fold(toks[0])], line, count=1
                )


def _externalize_series(sections: list[_Section], pointers: Mapping[str, str], eol: str) -> None:
    """Replace every ``[TIMESERIES]`` series with a pointer row ``{name} FILE "{path}"`` where ``pointers`` names it.

    The first row of a series becomes the pointer and its other rows go, so a derived model holds no series
    data. A series ``pointers`` does not name is left as it is.
    """
    if not pointers:
        return
    folded = {_fold(k): v for k, v in pointers.items()}
    for section in _all(sections, "TIMESERIES"):
        seen: set[str] = set()
        kept: list[str] = []
        for line in section.lines:
            toks = _tokens(line) if _is_data_row(line) else []
            key = _fold(toks[0]) if toks else None
            if key is None or key not in folded:
                kept.append(line)
            elif key not in seen:
                seen.add(key)
                kept.append(f'{toks[0]} FILE "{folded[key]}"' + (eol if line.endswith(("\n", "\r")) else ""))
        section.lines = kept


def _filter_rows(name: str, role: str, lines: list[str], rain_series: set[str]) -> list[str]:
    """Drop the rows of an F section that name an object a dropped section defines."""
    kept: list[str] = []
    for line in lines:
        toks = _tokens(line) if _is_data_row(line) else []
        if toks and _drops_row(name, role, toks, rain_series):
            continue
        kept.append(line)
    return kept


def _drops_row(name: str, role: str, toks: list[str], rain_series: set[str]) -> bool:
    if name == "TIMESERIES":
        return _fold(toks[0]) in rain_series
    if name == "REPORT":
        return _prefix_match(toks[0], "SUBCATCH" if role == "hydraulics" else "LINK")
    if name == "TAGS":
        return _prefix_match(toks[0], "SUBCATCH" if role == "hydraulics" else "LINK")
    if name == "ADJUSTMENTS":
        # N-PERV, DSTORE and INFIL name a subcatchment (``climate_readAdjustments``); the others are monthly rows.
        return any(_prefix_match(toks[0], key) for key in ("N-PERV", "DSTORE", "INFIL"))
    return False


def _set_option(options: list[_Section], key: str, value: str, eol: str) -> None:
    """Set ``key`` to ``value`` in the ``[OPTIONS]`` sections, keeping the row's spacing and comment.

    A key found in any section is set there. A key found in none is added to the first.
    """
    found = False
    for section in options:
        for i, line in enumerate(section.lines):
            toks = _tokens(line)
            if not (_is_data_row(line) and toks and _fold(toks[0]) == key):
                continue
            match = _OPTION_ROW.match(line)
            if match is None:
                continue
            section.lines[i] = f"{match['lead']}{match['key']}{match['gap']}{value}{match['tail']}{match['eol']}"
            found = True
    if not found:
        _append_rows(options[0], [f"{key:<21} {value}"], eol)


def _stage_series_names(sections: list[_Section]) -> set[str]:
    """The series ``[OUTFALLS]`` rows of ``TIMESERIES`` type read, case-folded (the stage-data token of each)."""
    return {
        _fold(toks[3])
        for toks in _node_rows(sections, "OUTFALLS")
        if len(toks) > 3 and _prefix_match(toks[2], "TIMESERIES")
    }


def rewrite_forcing(role_text: str, fill: Mapping[str, str]) -> str:
    """Rewrite only the forcing the toolkit owns, in a derived role model, for one event.

    Owned: the timing keys of ``[OPTIONS]`` (:data:`FORCING_OPTION_KEYS`), ``[RAINGAGES]`` wholesale, and the
    ``[TIMESERIES]`` rows of the rain series and of the water-level series, the ones whose name begins
    :data:`WATER_LEVEL_SERIES_PREFIX`. ``fill`` carries one value per owned key (the nine keys of
    :data:`FORCING_OPTION_KEYS` and :data:`FORCING_SECTION_KEYS`); ``RAINGAGES`` and ``TIMESERIES`` are blocks of
    rows, one per line. A water-level row is written only into a role whose ``[OUTFALLS]`` rows read that series, so
    the roles that converted the outfall carry no row for it, and a role with no subcatchments and no rain gauges
    (the hydraulics role) takes no rain row. Every row whose name begins the prefix is removed first, so a fill
    without water-level rows leaves no water-level series. Every other series is kept.

    Raises
    ------
    ProcessingError
        ``fill`` lacks an owned key.
    """
    needed = [*FORCING_OPTION_KEYS, *FORCING_SECTION_KEYS]
    absent = [k for k in needed if k not in fill]
    if absent:
        raise ProcessingError(
            operation="rewrite_forcing",
            reason=f"fill values missing: {absent}. Expected: {needed}.",
        )
    sections = _parse(role_text)
    eol = next((s.eol for s in sections if s.name is not None), "\n")

    _ensure_section(sections, "OPTIONS", eol)
    for key in FORCING_OPTION_KEYS:
        _set_option(_all(sections, "OPTIONS"), key, str(fill[key]), eol)

    has_gauges = _first(sections, "RAINGAGES") is not None or _first(sections, "SUBCATCHMENTS") is not None
    old_rain_series = _rain_series_names(sections)
    if has_gauges:
        raingages = _ensure_section(sections, "RAINGAGES", eol)
        for section in _all(sections, "RAINGAGES"):
            section.lines = [ln for ln in section.lines if not _is_data_row(ln)]
        _append_rows(raingages, _block_rows(fill["RAINGAGES"]), eol)

    prefix = _fold(WATER_LEVEL_SERIES_PREFIX)
    read_here = _stage_series_names(sections)
    series_rows = [
        row
        for row in _block_rows(fill["TIMESERIES"])
        if (_fold(_tokens(row)[0]).startswith(prefix) and _fold(_tokens(row)[0]) in read_here)
        or (has_gauges and not _fold(_tokens(row)[0]).startswith(prefix))
    ]
    # The fill is authoritative for the water level: every source row of the prefix goes, and the fill's rows (possibly
    # none, when the water-level boundary is off) replace them, so a run never reads a water level the toolkit
    # did not write.
    owned = {_fold(_tokens(r)[0]) for r in series_rows}
    if has_gauges:
        owned |= old_rain_series
    timeseries = _ensure_section(sections, "TIMESERIES", eol) if series_rows else _first(sections, "TIMESERIES")
    for section in _all(sections, "TIMESERIES"):
        section.lines = [
            ln
            for ln in section.lines
            if not (
                _is_data_row(ln)
                and (_fold((_tokens(ln) or [""])[0]) in owned or _fold((_tokens(ln) or [""])[0]).startswith(prefix))
            )
        ]
    if timeseries is not None:
        _append_rows(timeseries, series_rows, eol)
    return _render(sections)


def set_threads(role_text: str, threads: int) -> str:
    """Set ``THREADS`` in the ``[OPTIONS]`` of a role model, adding the line when the model has none (DR-36).

    Written into the ``hydraulics`` and ``full`` roles only: SWMM threads only its dynamic-wave loops, and the
    hydrology role routes by kinematic wave over one link, so a thread count there is inert.
    """
    sections = _parse(role_text)
    eol = next((s.eol for s in sections if s.name is not None), "\n")
    _ensure_section(sections, "OPTIONS", eol)
    _set_option(_all(sections, "OPTIONS"), "THREADS", str(threads), eol)
    return _render(sections)


def link_count(source_text: str) -> int:
    """The number of links (conduits, pumps, orifices, weirs, outlets) of a SWMM model, for the V25 bound."""
    sections = _parse(source_text)
    return sum(len(_node_rows(sections, name)) for name in ("CONDUITS", "PUMPS", "ORIFICES", "WEIRS", "OUTLETS"))


def _block_rows(block: str) -> list[str]:
    return [line for line in block.splitlines() if _tokens(line)]
