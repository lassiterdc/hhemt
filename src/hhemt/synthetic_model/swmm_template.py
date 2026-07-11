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
# Iter-8 peak_flood_depth (2026-04-28): nodes spread to make the Y branches
# of the conduit network visibly diagonal in figures. With DEM 16 cols × 30
# rows (n_cols 20→16 in iter-8), J1 sits 4 cols left of J3 and J2 sits 4
# cols right of J3, producing diagonal C1/C2 covering 4 cols × 9 rows each.
# J3, J4, sewer_outflow share col 7 (centered), so C3+C4 form the vertical
# stem of the Y. dummy_outfall stays disconnected at col 7.
# Number of in-line coupling junctions. MUST be >= the largest MPI rank count
# the experiment matrix exercises (8, from hhemt.synthetic_experiment's rank_sweep
# default (2,4,8) max = 8 / hybrid rows) so that under a row-strip static
# decomposition EVERY rank owns >= 1 coupling node and participates in the
# TRITON-SWMM ENSIFY_COMM_WORLD collective. Fewer nodes than ranks => a
# node-free top rank => coupling-collective deadlock (triton.h:2363-2404).
# (2026-06-14: replaced the hardcoded-for-16x30 _NODES list with this
# n_rows-driven generator — the old list clustered all nodes in the bottom
# ~22% of the 64x120 experiment grid, deadlocking every multi-rank run.)
_N_COUPLING_NODES = 8


def _centerline_col(params) -> int:
    """Interior centerline column for the conduit chain."""
    return (params.n_cols - 1) // 2


def _node_matrix_rows(params) -> list[int]:
    """Matrix-rows (0=top) of the in-line coupling junctions.

    Rule (triton-specialist 2026-06-14, generalized 2026-06-15): spread the
    ``_N_COUPLING_NODES`` junctions EVENLY across the interior conveyance zone
    ``[_WALL_THICKNESS, _sea_wall_mr - 1]`` — node ``i`` is centered in its 1/n
    slice — so (a) no node lands on a wall or in the BC shelf and (b) every
    TRITON top-to-bottom row-strip rank owns >= 1 node for the configured rank
    counts. Spreading over the INTERIOR (not the full grid) is what makes the
    rule grid-agnostic; the previous floor(n_rows/n) band rule placed a node on
    the top wall and was unsatisfiable when n_rows was not a multiple of n
    (e.g. the 16x30 default). 64x120 => {9,23,37,51,65,79,93,107} (every
    {2,3,4,8}-rank strip owns >=1 node); 16x30 => {3,6,9,12,14,17,20,23}.
    """
    from .geometry import _WALL_THICKNESS, _sea_wall_mr

    n = _N_COUPLING_NODES
    lo = _WALL_THICKNESS
    hi = _sea_wall_mr(params) - 1  # last interior row above the sea wall
    span = hi - lo + 1
    assert span >= n, (
        f"interior conveyance zone ({span} rows) too small for {n} coupling "
        f"nodes; raise n_rows or lower _N_COUPLING_NODES"
    )
    rows = [lo + int((i + 0.5) * span / n) for i in range(n)]
    # Even-spread tripwire: max gap (incl. interior edges) <= achievable ceil(span/n).
    gaps = [rows[0] - lo] + [rows[i] - rows[i - 1] for i in range(1, n)] + [hi - rows[-1]]
    assert max(gaps) <= -(-span // n), (
        f"node row gaps {gaps} exceed ceil(span/n)={-(-span // n)}; nodes not "
        f"evenly spread -> a row-strip rank could own zero coupling nodes"
    )
    return rows


# --- Lateral tributary branches (2026-06-15) --------------------------------
# A couple of off-centerline tributaries feeding mid-stem junctions, alternating
# left/right. NOT required for MPI rank coverage (TRITON's row-strip split is
# top-to-bottom, so the centerline stem already covers every rank — mpi_utils.h
# create_local_dims); these add river-like realism and spread the hydrology to
# both sides. The floodplain buffer (geometry.py) carves a tributary valley
# along each branch conduit automatically.
_N_BRANCHES = 2
_BRANCH_N_NODES = 2  # junctions per tributary (-> an (N+1)-node path incl. the confluence)
_BRANCH_BOUNDARY_MARGIN_COLS = 2  # cells the outer branch junction sits inside the side wall
_BRANCH_ROW_RISE = 6  # rows the outer end rises north of its attach (downhill into stem)


def _branches(params):
    """Lateral tributaries, alternating sides, each a chain of ``_BRANCH_N_NODES``
    junctions running from near the side boundary INWARD to a mid-stem attach
    junction (so each tributary is a ``_BRANCH_N_NODES + 1``-node path incl. the
    confluence).

    Returns ``[(junctions, attach_name), ...]`` where ``junctions`` is an ordered
    list of ``(name, col, row_from_bottom)`` from the OUTER (near-boundary,
    northmost) end inward toward the stem.
    """
    from .geometry import _WALL_THICKNESS  # lazy import — avoids a module import cycle

    col0 = _centerline_col(params)
    n_cols = params.n_cols
    mrs = _node_matrix_rows(params)
    n_stem = len(mrs)  # J1..J(n-1) + collector
    mid = n_stem // 2
    n = _BRANCH_N_NODES
    branches = []
    for i in range(_N_BRANCHES):
        attach_idx = mid - 1 + i
        attach_name = f"J{attach_idx + 1}" if attach_idx < n_stem - 1 else "collector"
        attach_mr = mrs[attach_idx]
        side = -1 if i % 2 == 0 else 1
        outer_col = (
            _WALL_THICKNESS + _BRANCH_BOUNDARY_MARGIN_COLS
            if side < 0
            else n_cols - 1 - _WALL_THICKNESS - _BRANCH_BOUNDARY_MARGIN_COLS
        )
        junctions = []
        for k in range(n):  # k=0 = outer (near boundary, north); k=n-1 = innermost (near stem)
            col = round(outer_col + (col0 - outer_col) * (k / n))
            mr = attach_mr - round(_BRANCH_ROW_RISE * (n - k) / n)
            junctions.append((f"B{i + 1}_{k + 1}", col, params.n_rows - 1 - mr))
        branches.append((junctions, attach_name))
    return branches


def _nodes(params):
    """``(name, col, row_from_bottom)`` for the in-line N->S coupling chain.

    Node 0 is northernmost (top); node ``_N_COUPLING_NODES-1`` is southernmost
    (``sewer_outflow``, the terminal junction whose rim surcharge feeds TRITON's
    2-D solver toward the southern boundary condition). All chain nodes sit on
    the conduit centerline column. ``dummy_outfall`` is the disconnected
    SWMM-required OUTFALL, parked one cell off-centerline at the northern end so
    it never joins the chain.
    """
    from .geometry import _sea_wall_mr  # lazy import — avoids a module import cycle

    col = _centerline_col(params)
    mrs = _node_matrix_rows(params)
    rfb = [params.n_rows - 1 - mr for mr in mrs]  # matrix-row -> row_from_bottom
    # 8 upstream coupling junctions spread one-per-band (MPI rank coverage):
    # J1..J7 + `collector` (southernmost, just upstream of the full-width sea wall).
    names = [f"J{i + 1}" for i in range(len(mrs) - 1)] + ["collector"]
    nodes = [(names[i], col, rfb[i]) for i in range(len(mrs))]
    # Interaction junction: in the BC shelf, just DOWNSTREAM of the sea wall. The
    # storm-tide BC floods its cell (rim == _BC_FLOOR < BC level) and it surcharges
    # backwater up the culvert (collector -> sewer_outflow) into the chain, so flow
    # propagates north and pours out of every upstream node's rim. One row below the
    # wall keeps it inside the watershed inset (row_from_bottom >= _WALL_THICKNESS).
    inter_mr = _sea_wall_mr(params) + 1
    nodes.append(("sewer_outflow", col, params.n_rows - 1 - inter_mr))
    # Disconnected SWMM-required OUTFALL, parked off-centerline at the north end.
    nodes.append(("dummy_outfall", max(col - 2, 1), rfb[0]))
    # Lateral tributary junctions (each branch is a chain near the boundary -> stem).
    for junctions, _attach in _branches(params):
        nodes.extend(junctions)
    return nodes


# Node-type partition: nodes whose names appear in `_OUTFALL_NAMES` go into
# the [OUTFALLS] section; everything else in `_NODES` is a junction. Allows
# arbitrary node names (e.g. `sewer_outflow`) without the prior
# name-prefix-based filter.
_OUTFALL_NAMES = {"dummy_outfall"}

# Iter-8 peak_flood_depth (2026-04-28): with the sloped upstream gradient
# (3.0 m at top → 1.5 m at sea-wall row) and the 0.5 m buffer-lowering on
# swale cells, swale rims at each junction are:
#   J1, J2 (matrix_row 7):     2.174  (gradient 2.674 − 0.5)
#   J3     (matrix_row 16):    1.587  (gradient 2.087 − 0.5)
#   J4     (matrix_row 25):    1.000  (gradient 1.500 − 0.5)
#   sewer_outflow (mr 27):     0.000  (dropoff 0.500 − 0.5)
#   dummy_outfall (matrix_row 3, disconnected): 2.370 (gradient 2.870 − 0.5)
# Junction depths cascade so all inverts step down across the network;
# slopes 2.0% on C1/C2/C3, 2.0% on C4. All within the user's 1–3% range.
#   sewer_outflow rim 0.0,   depth 0.5,   invert −0.5  (iter-7..11)
#   J4            rim 1.0,   depth 1.1,   invert −0.1
#   J3            rim 1.587, depth 1.087, invert  0.5
#   J1, J2        rim 2.174, depth 1.074, invert  1.1
#   dummy_outfall rim 2.370, depth 0.5,   invert  1.870 (disconnected)
#
# Iter-13 (2026-04-28): pipe diameter 0.2 → 1.0 m, inverts cascaded for 2.0 %
# slopes throughout. (Pre-iter-15 cascade — replaced below.)
#
# Iter-17 (2026-04-29): dropoff zone lowered 1.0 → 0.5 m to match the minimum
# upstream elevation (the buffer-lowered swale at the bottom corridor row).
# Sewer_outflow now sits in a 0.5 m dropoff zone, so its depth re-cascades.
# Other junctions' rims unchanged from iter-16; their depths preserved.
#   J1, J2 rim 0.891, depth 0.500, invert  0.391
#   J3     rim 0.696, depth 0.800, invert −0.104   C1, C2 (0.391 − −0.104)/30 = 1.65 %
#   J4     rim 0.500, depth 1.100, invert −0.600   C3     (−0.104 − −0.600)/30 = 1.65 %
#   sewer  rim 0.500, depth 1.400, invert −0.900   C4     (−0.600 − −0.900)/20 = 1.50 %
#   dummy_outfall depth 0.500 (disconnected; rim auto-pins to DEM at its cell).
# 2026-06-14: emptied to the uniform 0.5 m fallback (_JUNCTION_DEPTH_M) for the
# n_rows-driven 8-node chain — the per-name depths above were tuned for the
# retired 5-node 16x30 layout and no longer correspond to the new node names.
# With a uniform depth, inverts follow the DEM corridor slope (rim==DEM,
# invert = rim - 0.5). Tune per-node here if the review plots show conduit
# slopes outside the 1-3% target (see the main-agent tuning guide).
_NODE_DEPTHS_M: dict[str, float] = {}
# Backwards-compat default for any code path that still imports
# `_JUNCTION_DEPTH_M` (now a fallback for nodes not in `_NODE_DEPTHS_M`).
_JUNCTION_DEPTH_M = 0.5


def _node_depth(name: str) -> float:
    return _NODE_DEPTHS_M.get(name, _JUNCTION_DEPTH_M)


def max_node_rim_elev(params) -> float:
    """Maximum rim elevation across all SWMM nodes (junctions + outfalls).

    Used by the synthetic weather builder (`weather.py`) to set the BC-only
    and combined-event water-level forcing as `max_rim + 0.10` per the iter-4
    user feedback for `per_sim_peak_flood_depth`. Reading from the same
    `dem_elev_at` source that pins junction rims keeps the value
    deterministic across fixture rebuilds.
    """
    from .geometry import dem_elev_at

    return float(max(dem_elev_at(params, col, row_from_bottom) for _name, col, row_from_bottom in _nodes(params)))


# (name, from_node, to_node, length_m). C4 is the culvert that crosses the
# sea wall row to deliver flow from the upstream gradual-slope node J4 to
# `sewer_outflow` on the BC side.
# Iter-5 peak_flood_depth (2026-04-28): C1/C2/C3 lengths reduced from
# 100/100/80 m to 50 m so all conduits maintain ≥1% slope under the
# rim==DEM invariant. Pre-iter-5 lengths produced 0.59% slope on
# C1/C2 and 0.73% on C3 — below the 1% threshold the user required for
# valid backwater hydraulics. Slopes after change:
#   C1: (2.174 - 1.587) / 50 = 1.174%
#   C2: (2.174 - 1.587) / 50 = 1.174%
#   C3: (1.587 - 1.000) / 50 = 1.174%
#   C4: (1.000 - 0.000) / 20 = 5.000%
# C4 is already 5% — no change. Lengths are decoupled from the actual
# (col, row) cell distance in this fixture; they're treated as
# friction-only parameters by SWMM dynamic-wave routing.
# Iter-6 peak_flood_depth (2026-04-28): C1/C2/C3 lengths reduced 50→30 m so
# slopes match the user's 1–3 % range with iter-6's varied-depth inverts.
# Slopes after change (using inverts from `_NODE_DEPTHS_M`):
#   C1: (1.30 - 0.85) / 30 = 1.50 %
#   C2: (1.30 - 0.85) / 30 = 1.50 %
#   C3: (0.85 - 0.40) / 30 = 1.50 %
#   C4: (0.40 - 0.00) / 20 = 2.00 %
def _conduits(params):
    """N->S stem chain (J1 -> ... -> collector -> sewer_outflow; the last reach
    is the culvert crossing the sea wall) plus the lateral tributary reaches
    (each branch -> its mid-stem attach junction).

    Friction length is fixed at 30 m (decoupled from the actual cell distance —
    SWMM dynamic-wave routing treats [CONDUITS].Length as a friction parameter);
    the per-reach slope derives from the DEM-pinned inverts (rim==DEM).
    """
    stem = [n for n, _c, _r in _nodes(params) if n.startswith("J") or n in ("collector", "sewer_outflow")]
    reaches = [(f"C{i + 1}", stem[i], stem[i + 1], 30.0) for i in range(len(stem) - 1)]
    for i, (junctions, attach) in enumerate(_branches(params)):
        chain = [j[0] for j in junctions] + [attach]  # outer -> ... -> inner -> stem
        for k in range(len(chain) - 1):
            reaches.append((f"CB{i + 1}_{k + 1}", chain[k], chain[k + 1], 30.0))
    return reaches


# Iter-6 peak_flood_depth (2026-04-28): bumped 0.1 → 0.2 m so 30-min sim
# with BC peak 5 m can deliver enough backwater volume to flood the
# (now narrow) channel visibly.
# Iter-13 peak_flood_depth (2026-04-28): bumped 0.2 → 1.0 m so event 1
# backwater equilibrates within the canonical 180-min sim. At 0.2 m, τ at
# the upstream Y tips was ~68 h (95 % equilibrium ≈ 8.5 d). Pipe area
# scales D², so 1.0/0.2 = 25× area; τ should shrink by roughly the same
# factor → 95 % at J1/J2 in ~8 h sim. The conduit_flow surcharge story
# stays alive because the rainfall is constant 100 mm/hr — runoff still
# overwhelms each conduit's max-over-full at the peak.
_CONDUIT_DIAMETER_M = 1.0


# Iter-8 peak_flood_depth (2026-04-28): subcatchments aligned with the
# spread-Y junction layout. S1 sits along the C1 (left) branch near J1,
# S2 along the C2 (right) branch near J2, S3 along the C3 stem near J3.
# Polygons are 5×5 cell rectangles. They overlap the Y corridor (intended)
# and the side walls (visual artifact, ignored — SWMM routing uses the
# [SUBCATCHMENTS] Outlet field, not polygon coordinates).
# 2026-06-14: param-driven. 3 subcatchments draining to the first 3 chain
# junctions (J1/J2/J3), each a 5x5-cell polygon centered on its outlet node so
# it overlaps the now-full-height conduit corridor. (Was hardcoded for the
# retired 16x30 layout.)
def _subcatchments(params):
    """One subcatchment per upstream coupling junction (S_i -> J_i) — distributes
    hydrology inputs down the floodplain. Excludes the `collector`, the BC-side
    `sewer_outflow` interaction node, and the disconnected `dummy_outfall`."""
    outlets = [n for n, _c, _r in _nodes(params) if n.startswith("J") or n.startswith("B")]
    return [(f"S{i + 1}", outlets[i]) for i in range(len(outlets))]


def _subcatchment_polygon_cell_bounds(params) -> dict[str, tuple[int, int, int, int]]:
    """5x5-cell ``(cmin, rmin, cmax, rmax)`` polygon centered on each
    subcatchment's outlet junction, clamped to the interior (>= 1)."""
    node_cell = {n: (c, r) for n, c, r in _nodes(params)}
    bounds: dict[str, tuple[int, int, int, int]] = {}
    for name, outlet in _subcatchments(params):
        col, rfb = node_cell[outlet]
        cmin = max(col - 2, 1)
        rmin = max(rfb - 2, 1)
        bounds[name] = (cmin, rmin, cmin + 4, rmin + 4)
    return bounds


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
                "CMS",
                "HORTON",
                "DYNWAVE",
                "DEPTH",
                "0",
                "NO",
                "NO",
                "${START_DATE}",
                "${START_TIME}",
                "${REPORT_START_DATE}",
                "${REPORT_START_TIME}",
                "${END_DATE}",
                "${END_TIME}",
                "01/01",
                "12/31",
                "0",
                "${REPORT_STEP}",
                "00:00:10",
                "00:01:00",
                "00:00:01",
                "PARTIAL",
                "0.75",
                "0",
                "0",
                "BOTH",
                "H-W",
                "1",
            ],
        },
        index=pd.Index(
            [
                "FLOW_UNITS",
                "INFILTRATION",
                "FLOW_ROUTING",
                "LINK_OFFSETS",
                "MIN_SLOPE",
                "ALLOW_PONDING",
                "SKIP_STEADY_STATE",
                "START_DATE",
                "START_TIME",
                "REPORT_START_DATE",
                "REPORT_START_TIME",
                "END_DATE",
                "END_TIME",
                "SWEEP_START",
                "SWEEP_END",
                "DRY_DAYS",
                "REPORT_STEP",
                "WET_STEP",
                "DRY_STEP",
                "ROUTING_STEP",
                "INERTIAL_DAMPING",
                "VARIABLE_STEP",
                "LENGTHENING_STEP",
                "MIN_SURFAREA",
                "NORMAL_FLOW_LIMITED",
                "FORCE_MAIN_EQUATION",
                "THREADS",
            ],
            name="Key",
        ),
    )


def _junctions_df(params) -> pd.DataFrame:
    """Junction invert = DEM(col, row_from_bottom) - per-node depth, so rim
    (= invert + MaxDepth) lands exactly on the DEM surface at the node cell.
    Iter-6: depths now vary per-node via `_NODE_DEPTHS_M` so inverts cascade
    with 1.0–2.5 % conduit slopes under a flat-channel DEM (rim==DEM still
    holds because each junction's rim is its cell's DEM elevation, regardless
    of how deep its invert sits below)."""
    from .geometry import dem_elev_at

    rows = [(n, c, r) for n, c, r in _nodes(params) if n not in _OUTFALL_NAMES]
    depths = [_node_depth(n) for n, _, _ in rows]
    inverts = [round(dem_elev_at(params, c, r) - d, 3) for (_, c, r), d in zip(rows, depths, strict=True)]
    names = [n for n, _, _ in rows]
    return pd.DataFrame(
        {
            "InvertElev": inverts,
            "MaxDepth": depths,
            "InitDepth": [0 for _ in names],
            "SurchargeDepth": [0 for _ in names],
            "PondedArea": [0 for _ in names],
        },
        index=pd.Index(names, name="Name"),
    )


def _outfalls_df(params) -> pd.DataFrame:
    """Outfall rim matches DEM at its cell; invert pinned below rim by the
    per-node depth (`_NODE_DEPTHS_M`). Iter-2 mid-iteration: only
    `dummy_outfall` is in `_OUTFALL_NAMES`; all other nodes are junctions."""
    from .geometry import dem_elev_at

    rows = [(n, c, r) for n, c, r in _nodes(params) if n in _OUTFALL_NAMES]
    inverts = []
    for name, c, r in rows:
        rim = dem_elev_at(params, c, r)
        inverts.append(round(rim - _node_depth(name), 3))
    return pd.DataFrame(
        {
            "InvertElev": inverts,
            "OutfallType": ["FREE" for _ in rows],
            "StageOrTimeseries": ["NO" for _ in rows],
        },
        index=pd.Index([n for n, _, _ in rows], name="Name"),
    )


def _conduits_df(params) -> pd.DataFrame:
    conduits = _conduits(params)
    return pd.DataFrame(
        {
            "InletNode": [from_node for _, from_node, _, _ in conduits],
            "OutletNode": [to_node for _, _, to_node, _ in conduits],
            "Length": [length for _, _, _, length in conduits],
            "Roughness": [0.013 for _ in conduits],
            "InOffset": [0 for _ in conduits],
            "OutOffset": [0 for _ in conduits],
            "InitFlow": [0 for _ in conduits],
            "MaxFlow": [0 for _ in conduits],
        },
        index=pd.Index([name for name, *_ in conduits], name="Name"),
    )


def _xsections_df(params) -> pd.DataFrame:
    conduits = _conduits(params)
    return pd.DataFrame(
        {
            "Shape": ["CIRCULAR" for _ in conduits],
            "Geom1": [_CONDUIT_DIAMETER_M for _ in conduits],
            "Geom2": [0 for _ in conduits],
            "Geom3": [0 for _ in conduits],
            "Geom4": [0 for _ in conduits],
            "Barrels": [1 for _ in conduits],
        },
        index=pd.Index([name for name, *_ in conduits], name="Link"),
    )


def _inflows_df(params) -> pd.DataFrame:
    """One FLOW inflow entry per junction so TRITON-SWMM's coupling layer
    reads the full set of coupling nodes from [INFLOWS]. Excludes outfalls."""
    names = [n for n, *_ in _nodes(params) if n not in _OUTFALL_NAMES]
    return pd.DataFrame(
        {
            "Constituent": ["FLOW" for _ in names],
            "Time Series": ['""' for _ in names],
            "Type": ["FLOW" for _ in names],
            "Mfactor": [1.0 for _ in names],
            "Sfactor": [1 for _ in names],
            "Baseline": [0 for _ in names],
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
    for name, col, row in _nodes(params):
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
    subs = _subcatchments(params)
    names = [s for s, _ in subs]
    outlets = [outlet for _, outlet in subs]
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
            "Raingage": ["RG_synth" for _ in names],
            "Outlet": outlets,
            "Area": [round(a, 3) for a in areas_ha],
            "PercImperv": [100 for _ in names],
            "Width": [round(w, 1) for w in widths_m],
            "PercSlope": [round(params.slope_ns * 100.0, 3) for _ in names],
            "CurbLength": [0 for _ in names],
        },
        index=pd.Index(names, name="Name"),
    )


def _subareas_df(params) -> pd.DataFrame:
    subs = _subcatchments(params)
    return pd.DataFrame(
        {
            "N-Imperv": [params.impervious_mannings for _ in subs],
            "N-Perv": [params.pervious_mannings for _ in subs],
            "S-Imperv": [0.05 for _ in subs],
            "S-Perv": [0.05 for _ in subs],
            "PctZero": [25 for _ in subs],
            "RouteTo": ["OUTLET" for _ in subs],
        },
        index=pd.Index([s for s, _ in subs], name="Subcatchment"),
    )


def _infiltration_df(params) -> pd.DataFrame:
    subs = _subcatchments(params)
    return pd.DataFrame(
        {
            "MaxRate": [3.0 for _ in subs],
            "MinRate": [0.5 for _ in subs],
            "Decay": [4.0 for _ in subs],
            "DryTime": [7 for _ in subs],
            "MaxInfil": [0 for _ in subs],
        },
        index=pd.Index([s for s, _ in subs], name="Subcatchment"),
    )


def _subcatchment_world_bounds(params, name: str) -> tuple[float, float, float, float]:
    """Return (xmin, ymin, xmax, ymax) in world coords for subcatchment `name`.

    Cell-index bounds from `_SUBCATCHMENT_POLYGON_CELL_BOUNDS` are converted
    using `(xllcorner + col*cs, yllcorner + row*cs)` — so polygons snap to
    DEM cell boundaries, not cell centers.
    """
    cs = params.cell_size_m
    cmin, rmin, cmax, rmax = _subcatchment_polygon_cell_bounds(params)[name]
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
    for name, _ in _subcatchments(params):
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
    "JUNCTIONS",
    "OUTFALLS",
    "CONDUITS",
    "XSECTIONS",
    "INFLOWS",
    "COORDINATES",
    "SUBCATCHMENTS",
    "SUBAREAS",
    "INFILTRATION",
    "RAINGAGES",
    "TIMESERIES",
    "CURVES",
    "POLYGONS",
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
        m.inp.conduits = _conduits_df(params)
        m.inp.xsections = _xsections_df(params)
        m.inp.inflows = _inflows_df(params)
    if include_hydrology:
        m.inp.subcatchments = _subcatchments_df(params)
        m.inp.subareas = _subareas_df(params)
        m.inp.infiltration = _infiltration_df(params)
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
    subs = _subcatchments(params)
    df = pd.DataFrame(
        {
            "subcatchment_id": [s for s, _ in subs],
            "raingage_id": ["RG_synth" for _ in subs],
            "mrms_col": ["RG_synth" for _ in subs],
        }
    )
    df.to_csv(dest, index=False)
    return dest
