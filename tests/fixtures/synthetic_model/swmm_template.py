"""Synthesize SWMM .inp templates via swmmio + subcatchment->raingage mapping.

Generation strategy:
    1. Bespoke minimal STARTER_INP is written to disk.
    2. swmmio.Model loads the starter, DataFrames are set for each populated
       section, and inp.save() writes SWMM-semantically-correct content.
    3. `_inject_double_comment_separators` post-processes the output to insert
       a `;;---- ----` separator line after each `;;column-names` comment line.
       swmmio's writer emits only ONE comment line; the TRITON-SWMM C++ parser
       at swmm_triton.h:174-240 unconditionally skips TWO lines after each
       section header (expecting the SWMM-auto-formatted double comment), so
       the post-process restores the format the coupled parser expects.

Topology (6 junctions + 1 outfall, designed for 2 intentional DEM-cell overlaps):

    S1 (impervious upper) --> J1a --C1a(5m)--> J1b --C1b(~100m)--\\
                                                                 J_merge --Cm(~80m)--> J_out --Co(~20m)--> OUT1
    S2 (pervious lower)   --> J2a --C2a(5m)--> J2b --C2b(~100m)--/

    Short C1a and C2a pipes span 5 m at 10 m DEM resolution, so J1a/J1b and
    J2a/J2b each collapse to one TRITON cell. This exercises the toolkit's
    `scenario_inputs.py` overlap-handling code path with real data rather
    than synthesising empty edge cases.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import swmmio

# ---------------------------------------------------------------------------
# Bespoke minimal starter .inp — all sections the coupled parser and
# scenario_inputs.py may scan for must be present (even if empty). OPTIONS and
# REPORT carry default values; every other data section is empty and gets
# populated via swmmio DataFrame setters.
# ---------------------------------------------------------------------------
_STARTER_INP = """\
[TITLE]
Synthetic TRITON-SWMM test model

[OPTIONS]

[EVAPORATION]
;;Data Source    Parameters
;;-------------- ----------------
CONSTANT         0.0
DRY_ONLY         NO

[RAINGAGES]
${RAINGAGES}

[SUBCATCHMENTS]

[SUBAREAS]

[INFILTRATION]

[JUNCTIONS]

[OUTFALLS]

[CONDUITS]

[XSECTIONS]

[INFLOWS]

[CURVES]

[TIMESERIES]
${TIMESERIES}

[REPORT]
INPUT      NO
CONTROLS   NO
SUBCATCHMENTS ALL
NODES ALL
LINKS ALL

[TAGS]

[MAP]
DIMENSIONS 0.000 0.000 10000.000 10000.000
Units      None

[COORDINATES]
"""


# ---------------------------------------------------------------------------
# Topology definition — centralized so `build_templates` and coord helpers
# stay in sync.
# ---------------------------------------------------------------------------
# (name, col, row) where (col, row) is the DEM cell index from the bottom-left
# of the 20x30 default grid. All nodes are placed at cell centers. The two
# overlap pairs (J1a/J1b and J2a/J2b) share a cell deliberately. Intermediate
# nodes J1bi{N}, J2bi{N}, Jmi{N}, Joi{N} subdivide the long conduits so the
# model has enough links (19 conduits) to exceed EPA SWMM 5.2's internal
# OpenMP threshold (Nobjects[LINK] >= 4 * NumThreads).
_NODES = [
    # Subcatchment outlets + overlap pairs (unchanged; anchor for hydrology->TRITON coupling)
    ("J1a",     6,  22),
    ("J1b",     6,  22),   # overlap pair 1 with J1a
    ("J2a",     13, 22),
    ("J2b",     13, 22),   # overlap pair 2 with J2a
    # Intermediate junctions along J1b -> J_merge (4 new; interpolated between (6,22) and (10,12))
    ("J1bi1",   7,  20),
    ("J1bi2",   8,  18),
    ("J1bi3",   8,  16),
    ("J1bi4",   9,  14),
    # Intermediate junctions along J2b -> J_merge (4 new; interpolated between (13,22) and (10,12))
    ("J2bi1",   12, 20),
    ("J2bi2",   12, 18),
    ("J2bi3",   11, 16),
    ("J2bi4",   11, 14),
    # Merge point (unchanged)
    ("J_merge", 10, 12),
    # Intermediate junctions along J_merge -> J_out (4 new; interpolated between (10,12) and (10,4))
    ("Jmi1",    10, 11),
    ("Jmi2",    10, 10),
    ("Jmi3",    10, 8),
    ("Jmi4",    10, 6),
    # Outlet junction (unchanged)
    ("J_out",   10, 4),
    # Intermediate junction along J_out -> OUT1 (1 new)
    ("Joi1",    10, 3),
    # Outfall (unchanged)
    ("OUT1",    10, 2),
]

# Junction invert elevations (metres). Slope downward from upstream to outfall.
# Upstream (J1b/J2b) elevation 8.0 -> J_merge elevation 5.0 interpolated over
# 5 conduit segments: 8.0, 7.4, 6.8, 6.2, 5.6, 5.0.
# J_merge elevation 5.0 -> J_out elevation 2.0 interpolated over 5 segments:
# 5.0, 4.4, 3.8, 3.2, 2.6, 2.0.
# J_out elevation 2.0 -> OUT1 elevation 0.0 interpolated over 2 segments:
# J_out = 2.0, Joi1 = 1.0, OUT1 = 0.0.
_JUNCTION_INVERTS = {
    "J1a":     8.0,
    "J1b":     8.0,
    "J1bi1":   7.4,
    "J1bi2":   6.8,
    "J1bi3":   6.2,
    "J1bi4":   5.6,
    "J2a":     8.0,
    "J2b":     8.0,
    "J2bi1":   7.4,
    "J2bi2":   6.8,
    "J2bi3":   6.2,
    "J2bi4":   5.6,
    "J_merge": 5.0,
    "Jmi1":    4.4,
    "Jmi2":    3.8,
    "Jmi3":    3.2,
    "Jmi4":    2.6,
    "J_out":   2.0,
    "Joi1":    1.0,
}

# (name, from_node, to_node, length_m). 19 conduits total; exceeds the EPA
# SWMM 5.2 threshold Nobjects[LINK] >= 4 * NumThreads for NumThreads up to 4.
_CONDUITS = [
    # Short overlap-causing conduits (unchanged; preserve the DEM-cell overlap
    # behavior that exercises scenario_inputs.py's overlap-handling path)
    ("C1a",    "J1a",     "J1b",     5.0),
    ("C2a",    "J2a",     "J2b",     5.0),
    # J1b -> J_merge (5 segments, 20m each, total 100m — unchanged aggregate length)
    ("C1b_1",  "J1b",     "J1bi1",   20.0),
    ("C1b_2",  "J1bi1",   "J1bi2",   20.0),
    ("C1b_3",  "J1bi2",   "J1bi3",   20.0),
    ("C1b_4",  "J1bi3",   "J1bi4",   20.0),
    ("C1b_5",  "J1bi4",   "J_merge", 20.0),
    # J2b -> J_merge (5 segments, 20m each, total 100m)
    ("C2b_1",  "J2b",     "J2bi1",   20.0),
    ("C2b_2",  "J2bi1",   "J2bi2",   20.0),
    ("C2b_3",  "J2bi2",   "J2bi3",   20.0),
    ("C2b_4",  "J2bi3",   "J2bi4",   20.0),
    ("C2b_5",  "J2bi4",   "J_merge", 20.0),
    # J_merge -> J_out (5 segments, 16m each, total 80m — unchanged aggregate length)
    ("Cm_1",   "J_merge", "Jmi1",    16.0),
    ("Cm_2",   "Jmi1",    "Jmi2",    16.0),
    ("Cm_3",   "Jmi2",    "Jmi3",    16.0),
    ("Cm_4",   "Jmi3",    "Jmi4",    16.0),
    ("Cm_5",   "Jmi4",    "J_out",   16.0),
    # J_out -> OUT1 (2 segments, 10m each, total 20m)
    ("Co_1",   "J_out",   "Joi1",    10.0),
    ("Co_2",   "Joi1",    "OUT1",    10.0),
]

# (subcatchment_id, outlet_node)
_SUBCATCHMENTS = [
    ("S1", "J1a"),
    ("S2", "J2a"),
]


# ---------------------------------------------------------------------------
# DataFrame builders
# ---------------------------------------------------------------------------
def _options_df() -> pd.DataFrame:
    """OPTIONS DataFrame. Timing keys use ``${NAME}`` placeholders that the
    toolkit's ``swmm_utils.create_swmm_inp_from_template`` fills at scenario
    prep via ``string.Template.safe_substitute``. Non-timing keys are literal.
    """
    return pd.DataFrame(
        {
            "Value": [
                "CMS", "HORTON", "DYNWAVE", "DEPTH", "0", "NO", "NO",
                "${START_DATE}", "${START_TIME}",
                "${REPORT_START_DATE}", "${REPORT_START_TIME}",
                "${END_DATE}", "${END_TIME}",
                "01/01", "12/31", "0",
                "${REPORT_STEP}", "00:00:10", "00:01:00", "00:00:01",
                "PARTIAL", "0.75", "0", "0", "BOTH", "H-W", "1",
            ],
        },
        index=pd.Index(
            [
                "FLOW_UNITS", "INFILTRATION", "FLOW_ROUTING", "LINK_OFFSETS",
                "MIN_SLOPE", "ALLOW_PONDING", "SKIP_STEADY_STATE",
                "START_DATE", "START_TIME", "REPORT_START_DATE",
                "REPORT_START_TIME", "END_DATE", "END_TIME", "SWEEP_START",
                "SWEEP_END", "DRY_DAYS", "REPORT_STEP", "WET_STEP",
                "DRY_STEP", "ROUTING_STEP", "INERTIAL_DAMPING",
                "VARIABLE_STEP", "LENGTHENING_STEP", "MIN_SURFAREA",
                "NORMAL_FLOW_LIMITED", "FORCE_MAIN_EQUATION", "THREADS",
            ],
            name="Key",
        ),
    )


def _junctions_df() -> pd.DataFrame:
    names = [n for n, *_ in _NODES if n.startswith("J")]
    return pd.DataFrame(
        {
            "InvertElev": [_JUNCTION_INVERTS[n] for n in names],
            "MaxDepth":   [2.0 for _ in names],
            "InitDepth":  [0 for _ in names],
            "SurchargeDepth": [0 for _ in names],
            "PondedArea": [0 for _ in names],
        },
        index=pd.Index(names, name="Name"),
    )


def _outfalls_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "InvertElev": [0.0],
            "OutfallType": ["FREE"],
            "StageOrTimeseries": ["NO"],
        },
        index=pd.Index(["OUT1"], name="Name"),
    )


def _conduits_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "InletNode":  [from_node for _, from_node, _, _ in _CONDUITS],
            "OutletNode": [to_node for _, _, to_node, _ in _CONDUITS],
            "Length":     [length for _, _, _, length in _CONDUITS],
            "Roughness":  [0.013 for _ in _CONDUITS],
            "InOffset":   [0 for _ in _CONDUITS],
            "OutOffset":  [0 for _ in _CONDUITS],
            "InitFlow":   [0 for _ in _CONDUITS],
            "MaxFlow":    [0 for _ in _CONDUITS],
        },
        index=pd.Index([name for name, *_ in _CONDUITS], name="Name"),
    )


def _xsections_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Shape":   ["CIRCULAR" for _ in _CONDUITS],
            "Geom1":   [1.0 for _ in _CONDUITS],
            "Geom2":   [0 for _ in _CONDUITS],
            "Geom3":   [0 for _ in _CONDUITS],
            "Geom4":   [0 for _ in _CONDUITS],
            "Barrels": [1 for _ in _CONDUITS],
        },
        index=pd.Index([name for name, *_ in _CONDUITS], name="Link"),
    )


def _inflows_df() -> pd.DataFrame:
    """One FLOW inflow entry per junction so TRITON-SWMM's coupling layer
    reads the full set of coupling nodes from [INFLOWS]."""
    names = [n for n, *_ in _NODES if n.startswith("J")]
    return pd.DataFrame(
        {
            "Constituent": ["FLOW" for _ in names],
            "Time Series": ['""' for _ in names],
            "Type":        ["FLOW" for _ in names],
            "Mfactor":     [1.0 for _ in names],
            "Sfactor":     [1 for _ in names],
            "Baseline":    [0 for _ in names],
        },
        index=pd.Index(names, name="Node"),
    )


def _coordinates_df(params) -> pd.DataFrame:
    """All nodes at DEM cell centers. Watershed is inset one cell from the
    DEM extent; all placements lie strictly inside the watershed polygon."""
    cs = params.cell_size_m
    x0 = params.xllcorner
    y0 = params.yllcorner
    def pt(col, row):
        return (x0 + (col + 0.5) * cs, y0 + (row + 0.5) * cs)
    xs = []
    ys = []
    names = []
    for name, col, row in _NODES:
        x, y = pt(col, row)
        names.append(name)
        xs.append(x)
        ys.append(y)
    return pd.DataFrame(
        {"X": xs, "Y": ys},
        index=pd.Index(names, name="Node"),
    )


def _subcatchments_df(params) -> pd.DataFrame:
    # S1 occupies the upper (impervious) half; S2 occupies the lower (pervious)
    # half. Areas are 1.5 ha each — coarse but fine for synthesised flows.
    return pd.DataFrame(
        {
            "Raingage":   ["RG_synth" for _ in _SUBCATCHMENTS],
            "Outlet":     [outlet for _, outlet in _SUBCATCHMENTS],
            "Area":       [1.5 for _ in _SUBCATCHMENTS],
            "PercImperv": [100, 0],
            "Width":      [50 for _ in _SUBCATCHMENTS],
            "PercSlope":  [params.slope_ns * 100.0 for _ in _SUBCATCHMENTS],
            "CurbLength": [0 for _ in _SUBCATCHMENTS],
        },
        index=pd.Index([s for s, _ in _SUBCATCHMENTS], name="Name"),
    )


def _subareas_df(params) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "N-Imperv":  [params.impervious_mannings for _ in _SUBCATCHMENTS],
            "N-Perv":    [params.pervious_mannings for _ in _SUBCATCHMENTS],
            "S-Imperv":  [0.05 for _ in _SUBCATCHMENTS],
            "S-Perv":    [0.05 for _ in _SUBCATCHMENTS],
            "PctZero":   [25 for _ in _SUBCATCHMENTS],
            "RouteTo":   ["OUTLET" for _ in _SUBCATCHMENTS],
        },
        index=pd.Index([s for s, _ in _SUBCATCHMENTS], name="Subcatchment"),
    )


def _infiltration_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "MaxRate":  [3.0 for _ in _SUBCATCHMENTS],
            "MinRate":  [0.5 for _ in _SUBCATCHMENTS],
            "Decay":    [4.0 for _ in _SUBCATCHMENTS],
            "DryTime":  [7 for _ in _SUBCATCHMENTS],
            "MaxInfil": [0 for _ in _SUBCATCHMENTS],
        },
        index=pd.Index([s for s, _ in _SUBCATCHMENTS], name="Subcatchment"),
    )


# ---------------------------------------------------------------------------
# Post-process: inject `;;---- ----` separator lines
# ---------------------------------------------------------------------------
# TRITON-SWMM's C++ coupled parser (`swmm_triton.h:174-240`) does
# `std::getline × 2` unconditionally after each section header, expecting the
# SWMM-auto-formatted double-comment pattern (column-name line + dashes
# separator). swmmio's writer emits only the first `;;` comment line. Without
# the dashes separator, the parser mis-skips data rows. This post-process
# inserts a dashes line after every `;;column-name` line that is followed by
# a data line (rather than another comment or blank).
_SECTIONS_NEEDING_DASHES = {
    "JUNCTIONS", "OUTFALLS", "CONDUITS", "XSECTIONS", "INFLOWS",
    "COORDINATES", "SUBCATCHMENTS", "SUBAREAS", "INFILTRATION",
    "RAINGAGES", "TIMESERIES", "CURVES",
}


def _dashes_for(comment_line: str) -> str:
    """Build a `;;---- ----` line matching the column widths of a `;;Col1 Col2 ...`
    comment line."""
    # Strip the leading `;;` then replace each non-space run with dashes.
    payload = comment_line[2:]
    return ";;" + re.sub(r"\S+", lambda m: "-" * len(m.group(0)), payload)


def _inject_double_comment_separators(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    out_lines = []
    current_section: str | None = None
    comment_line_idx = -1  # index in out_lines of the last emitted `;;` comment
    for line in text.splitlines():
        stripped = line.strip()
        # Section header?
        m = re.match(r"^\[([A-Za-z_]+)\]\s*$", line)
        if m:
            current_section = m.group(1).upper()
            comment_line_idx = -1
            out_lines.append(line)
            continue
        if current_section in _SECTIONS_NEEDING_DASHES:
            if stripped.startswith(";;"):
                # Only insert a dashes line after the FIRST `;;` comment in a
                # section; skip if this comment IS a dashes line already.
                if "-" in stripped and set(stripped.replace(";", "").strip()) <= {"-", " "}:
                    # already a dashes separator
                    out_lines.append(line)
                    comment_line_idx = -1
                else:
                    out_lines.append(line)
                    if comment_line_idx < 0:
                        comment_line_idx = len(out_lines) - 1
                continue
            # Non-comment line inside a data section — if we just emitted a
            # `;;column-names` line and haven't yet added a dashes line, inject
            # one now (before this data/blank line).
            if comment_line_idx >= 0 and stripped != "":
                dashes = _dashes_for(out_lines[comment_line_idx])
                out_lines.append(dashes)
                comment_line_idx = -1
        out_lines.append(line)
    path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def _write_variant(
    params,
    dest: Path,
    include_hydrology: bool,
    include_hydraulics: bool,
) -> Path:
    # Write the starter to disk — swmmio needs a real file path.
    dest.write_text(_STARTER_INP, encoding="utf-8")
    m = swmmio.Model(str(dest))
    # OPTIONS / REPORT — always written so synthesised run params take effect.
    m.inp.options = _options_df()
    # JUNCTIONS / OUTFALLS / COORDINATES — always present. Even hydrology-only
    # .inp needs junctions so SUBCATCHMENTS can reference outlet nodes.
    m.inp.junctions = _junctions_df()
    m.inp.outfalls = _outfalls_df()
    m.inp.coordinates = _coordinates_df(params)
    if include_hydraulics:
        m.inp.conduits = _conduits_df()
        m.inp.xsections = _xsections_df()
        m.inp.inflows = _inflows_df()
    if include_hydrology:
        m.inp.subcatchments = _subcatchments_df(params)
        m.inp.subareas = _subareas_df(params)
        m.inp.infiltration = _infiltration_df()
    m.inp.save()
    _inject_double_comment_separators(dest)
    return dest


def build_templates(params, cache_dir: Path):
    """Write swmm_hydraulics.inp, swmm_hydrology.inp, swmm_full.inp.

    Each variant is produced by a separate `swmmio.Model.save()` on a fresh
    copy of the starter, then post-processed to add `;;---- ----` separator
    lines required by the TRITON-SWMM C++ coupled parser.
    """
    hydraulics = cache_dir / "swmm_hydraulics.inp"
    hydrology = cache_dir / "swmm_hydrology.inp"
    full = cache_dir / "swmm_full.inp"
    _write_variant(params, hydraulics, include_hydrology=False, include_hydraulics=True)
    _write_variant(params, hydrology, include_hydrology=True, include_hydraulics=False)
    _write_variant(params, full, include_hydrology=True, include_hydraulics=True)
    return hydraulics, hydrology, full


def build_subcatchment_raingage_mapping(params, dest: Path) -> Path:
    df = pd.DataFrame(
        {
            "subcatchment_id": [s for s, _ in _SUBCATCHMENTS],
            "raingage_id":     ["RG_synth" for _ in _SUBCATCHMENTS],
            "mrms_col":        ["RG_synth" for _ in _SUBCATCHMENTS],
        }
    )
    df.to_csv(dest, index=False)
    return dest
