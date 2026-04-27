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

Topology (iteration-2 mid-iteration revision 2026-04-26 — 5 junctions + 1 dummy outfall):

    S1 (upper-left)  --> J1 --C1(100m)--\\
    S2 (upper-right) --> J2 --C2(100m)---> J3 --C3(80m)--> J4 (no subcatchment;
    S3 (middle-left) --> J3                                 just upstream of sea
                                                            wall) --C4(20m,
                                                            culvert)--> sewer_outflow

    `sewer_outflow` is a regular JUNCTION sitting in the dropoff/BC zone
    (per user feedback 2026-04-26 — was an outfall in the prior iter-2 spec).
    It is the terminal node of the drainage chain; flow surcharges out of its
    rim into TRITON's 2-D surface solver per the scenario constraint.

    `dummy_outfall` is the only SWMM-side OUTFALL element — disconnected and
    placed in an inconspicuous interior cell. SWMM requires at least one
    outfall for parsing semantics.

    Pipe geometry (Geom1 = 0.2 m circular) is intentionally undersized so
    runoff from S1/S2/S3 surcharges the network and forces flow into TRITON
    via the rim/cell coupling — per iteration-2 scenario constraints.
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
# (name, col, row_from_bottom). DEM is 20×30; walls occupy matrix_rows 0..1
# (top), cols 0..1 + 18..19 (sides), and matrix_row 26 = sea-wall row (interior
# peak between gradual slope rows 2..25 and dropoff/BC rows 27..29).
# Iteration-2 topology (mid-iteration revision 2026-04-26): 5 junctions
# (J1..J4 + sewer_outflow) + 1 dummy outfall. `sewer_outflow` is a regular
# junction (per user feedback) sitting in the dropoff/BC zone — flow exits
# to TRITON's 2-D surface solver via rim surcharge from this junction. The
# only SWMM-side OUTFALL element is the disconnected `dummy_outfall` (SWMM
# requires at least one outfall for parsing semantics).
_NODES = [
    ("J1",             6,  22),   # top-left  (S1 drains here)
    ("J2",             13, 22),   # top-right (S2 drains here)
    ("J3",             10, 13),   # middle    (S3 drains here; J1+J2 merge here)
    ("J4",             10,  4),   # just upstream of sea wall — NO subcatchment;
                                  # receives J3's flow and forwards via culvert
                                  # to sewer_outflow on the BC side.
    ("sewer_outflow",  10,  2),   # JUNCTION in dropoff/BC zone (terminal node;
                                  # surcharge here exits to TRITON 2-D solver
                                  # per the iter-2 scenario constraint).
    ("dummy_outfall",  17, 26),   # disconnected SWMM outfall, required for
                                  # parsing.
]

# Node-type partition: nodes whose names appear in `_OUTFALL_NAMES` go into
# the [OUTFALLS] section; everything else in `_NODES` is a junction. Allows
# arbitrary node names (e.g. `sewer_outflow`) without the prior
# name-prefix-based filter.
_OUTFALL_NAMES = {"dummy_outfall"}

# Rim elevation for every node equals the DEM cell-center elevation at that
# node's (col, row) — enforced by the rim==DEM assertion in cache.py. Invert
# elevations are pinned to rim - depth. Depth = 0.5 m keeps every invert
# non-negative (J4 at row_from_bottom=4 sits at gradient elev 1.5 m; the
# `sewer_outflow` outfall at row 2 sits at dropoff elev 0.5 m).
_JUNCTION_DEPTH_M = 0.5

# (name, from_node, to_node, length_m). C4 is the culvert that crosses the
# sea wall row to deliver flow from the upstream gradual-slope node J4 to
# `sewer_outflow` on the BC side.
_CONDUITS = [
    ("C1", "J1", "J3",            100.0),
    ("C2", "J2", "J3",            100.0),
    ("C3", "J3", "J4",             80.0),
    ("C4", "J4", "sewer_outflow",  20.0),
]

# Conduit cross-section diameter (m). Intentionally small so runoff from
# S1/S2/S3 surcharges the network — forces flow into the TRITON 2-D solver
# via the rim/cell coupling. Iter-3 conduit_flow feedback: 0.2 m kept C4 at
# only 0.94 max-over-full and C1/C2/C3 well below 1.0 at 500 mm/hr rainfall;
# 0.1 m cap forces all four conduits to surcharge (max-over-full >= 1.0).
_CONDUIT_DIAMETER_M = 0.1

# (col_min, row_from_bottom_min, col_max, row_from_bottom_max). Per iteration-2
# feedback: identically-sized 5×5-cell polygons. S1/S2/S3 only — S4 removed
# (J4 has no subcatchment in iteration-2). Positioned so each polygon sits
# near its outlet without overlapping the connecting conduits.
_SUBCATCHMENT_POLYGON_CELL_BOUNDS = {
    "S1": (3,  23, 8,  28),   # 5×5, upper-left, near J1 (col 6, row 22)
    "S2": (12, 23, 17, 28),   # 5×5, upper-right, near J2 (col 13, row 22)
    "S3": (3,   9, 8,  14),   # 5×5, middle-left of J3 (col 10, row 13)
}

_SUBCATCHMENTS = [
    ("S1", "J1"),
    ("S2", "J2"),
    ("S3", "J3"),
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


def _junctions_df(params) -> pd.DataFrame:
    """Junction invert = DEM(col, row_from_bottom) - _JUNCTION_DEPTH_M, so rim
    (= invert + MaxDepth) lands exactly on the DEM surface at the node cell.
    Junctions = all `_NODES` whose name is NOT in `_OUTFALL_NAMES` (iter-2
    mid-iteration: `sewer_outflow` is a regular junction)."""
    from .geometry import dem_elev_at

    rows = [(n, c, r) for n, c, r in _NODES if n not in _OUTFALL_NAMES]
    depth = _JUNCTION_DEPTH_M
    inverts = [round(dem_elev_at(params, c, r) - depth, 3) for _, c, r in rows]
    names = [n for n, _, _ in rows]
    return pd.DataFrame(
        {
            "InvertElev": inverts,
            "MaxDepth":   [depth for _ in names],
            "InitDepth":  [0 for _ in names],
            "SurchargeDepth": [0 for _ in names],
            "PondedArea": [0 for _ in names],
        },
        index=pd.Index(names, name="Name"),
    )


def _outfalls_df(params) -> pd.DataFrame:
    """Outfall rim matches DEM at its cell; invert pinned below rim. Iter-2
    mid-iteration: only `dummy_outfall` is in `_OUTFALL_NAMES`; all other
    nodes are junctions."""
    from .geometry import dem_elev_at

    rows = [(n, c, r) for n, c, r in _NODES if n in _OUTFALL_NAMES]
    inverts = []
    for _name, c, r in rows:
        rim = dem_elev_at(params, c, r)
        inverts.append(round(rim - _JUNCTION_DEPTH_M, 3))
    return pd.DataFrame(
        {
            "InvertElev": inverts,
            "OutfallType": ["FREE" for _ in rows],
            "StageOrTimeseries": ["NO" for _ in rows],
        },
        index=pd.Index([n for n, _, _ in rows], name="Name"),
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
            "Geom1":   [_CONDUIT_DIAMETER_M for _ in _CONDUITS],
            "Geom2":   [0 for _ in _CONDUITS],
            "Geom3":   [0 for _ in _CONDUITS],
            "Geom4":   [0 for _ in _CONDUITS],
            "Barrels": [1 for _ in _CONDUITS],
        },
        index=pd.Index([name for name, *_ in _CONDUITS], name="Link"),
    )


def _inflows_df() -> pd.DataFrame:
    """One FLOW inflow entry per junction so TRITON-SWMM's coupling layer
    reads the full set of coupling nodes from [INFLOWS]. Excludes outfalls."""
    names = [n for n, *_ in _NODES if n not in _OUTFALL_NAMES]
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
    # Alternate impervious fraction across the 3 subcatchments so test runs
    # exercise mixed surface response. Area is derived from the actual polygon
    # bounds so `[SUBCATCHMENTS].Area` matches `[POLYGONS]`. Iter-2: S4 removed
    # (J4 has no subcatchment); S1/S2/S3 polygons are identical 5×5 rectangles.
    # iter-3 feedback (conduit_flow): all 100% impervious to max runoff and
    # ensure every downstream conduit reaches >= max-over-full flow. The earlier
    # 80/20/60 split was for visual variety, retained as the comment for record.
    perc_imperv_map = {"S1": 100, "S2": 100, "S3": 100}
    names = [s for s, _ in _SUBCATCHMENTS]
    outlets = [outlet for _, outlet in _SUBCATCHMENTS]
    areas_ha = []
    widths_m = []
    for name in names:
        x_min, y_min, x_max, y_max = _subcatchment_world_bounds(params, name)
        width_m = x_max - x_min
        height_m = y_max - y_min
        areas_ha.append((width_m * height_m) / 10_000.0)
        # Use the polygon's long-axis dimension as SWMM `Width` (characteristic
        # overland flow width).
        widths_m.append(max(width_m, height_m))
    return pd.DataFrame(
        {
            "Raingage":   ["RG_synth" for _ in names],
            "Outlet":     outlets,
            "Area":       [round(a, 3) for a in areas_ha],
            "PercImperv": [perc_imperv_map[n] for n in names],
            "Width":      [round(w, 1) for w in widths_m],
            "PercSlope":  [round(params.slope_ns * 100.0, 3) for _ in names],
            "CurbLength": [0 for _ in names],
        },
        index=pd.Index(names, name="Name"),
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


def _subcatchment_world_bounds(params, name: str) -> tuple[float, float, float, float]:
    """Return (xmin, ymin, xmax, ymax) in world coords for subcatchment `name`.

    Cell-index bounds from `_SUBCATCHMENT_POLYGON_CELL_BOUNDS` are converted
    using `(xllcorner + col*cs, yllcorner + row*cs)` — so polygons snap to
    DEM cell boundaries, not cell centers.
    """
    cs = params.cell_size_m
    cmin, rmin, cmax, rmax = _SUBCATCHMENT_POLYGON_CELL_BOUNDS[name]
    return (
        params.xllcorner + cmin * cs,
        params.yllcorner + rmin * cs,
        params.xllcorner + cmax * cs,
        params.yllcorner + rmax * cs,
    )


def _polygons_df(params) -> pd.DataFrame:
    """SWMM `[POLYGONS]` DataFrame: each row is one polygon vertex, indexed
    by subcatchment name. Rectangles here; swmmio accepts arbitrary counts.
    """
    rows: list[dict] = []
    index: list[str] = []
    for name, _ in _SUBCATCHMENTS:
        xmin, ymin, xmax, ymax = _subcatchment_world_bounds(params, name)
        verts = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
        for vx, vy in verts:
            rows.append({"X": round(vx, 3), "Y": round(vy, 3)})
            index.append(name)
    return pd.DataFrame(rows, index=pd.Index(index, name="Name"))


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
    "RAINGAGES", "TIMESERIES", "CURVES", "POLYGONS",
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
    m.inp.junctions = _junctions_df(params)
    m.inp.outfalls = _outfalls_df(params)
    m.inp.coordinates = _coordinates_df(params)
    if include_hydraulics:
        m.inp.conduits = _conduits_df()
        m.inp.xsections = _xsections_df()
        m.inp.inflows = _inflows_df()
    if include_hydrology:
        m.inp.subcatchments = _subcatchments_df(params)
        m.inp.subareas = _subareas_df(params)
        m.inp.infiltration = _infiltration_df()
        m.inp.polygons = _polygons_df(params)
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
