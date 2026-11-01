"""Raw-layer writer for the golden analysis trees: the solver-output files of each event directory.

The golden trees are written by the generator package in ``tests/goldens/generator/``, which is
independent of hhemt: that independence is the premise of ``tests/test_golden_conformance.py``, which
asks hhemt's own readers whether the trees are finished analyses, and a generator that imported hhemt
would have hhemt asked to accept trees hhemt itself wrote (see
``tests/test_golden_generator_independence.py``). A golden is a finished analysis tree plus the files
hhemt needs to construct an analysis object from it (``tests/goldens/SPEC.md`` section 1). This module
writes only the raw layer; the completion records and input files are written by ``records.py`` and the
summary layer by ``summary.py``.

Writes what a finished solver run leaves in ``out_triton/``, ``out_tritonswmm/`` and
``out_swmm/`` of one event directory, plus the runoff-only model's ``hydro.out`` and ``hydro.rpt``
in that event directory's ``swmm/``, at dimensions reduced from the observed 30 by 16 cell grid and
1080 reporting steps (``tests/goldens/SPEC.md`` section 2). It follows SPEC.md section 7 and the
solver sources SPEC.md names; it imports the standard library and ``numpy`` and nothing else
(SPEC.md header: the generator imports no hhemt code and nothing from the rest of ``tests/``).

Specification map (every layout statement below carries the SPEC.md section and the solver
source it follows):

* TRITON binary raster, SPEC 7.1, TRITON ``src/output.h`` lines 1016-1030 and 1083-1094:
  two float64 (rows, cols) then the interior values (ghost cells removed, SPEC.md section 7.1)
  row by row, little-endian float64.
* SWMM binary ``.out``, SPEC 7.2, EPA SWMM v5.2.4 ``src/solver/output.c`` ``output_open``
  (lines 120-380), ``output_saveResults`` (457-505) and ``output_end`` (515-530), with
  constants from ``src/solver/consts.h`` (``VERSION`` 52004, ``MAGICNUMBER`` 516114522),
  ``src/solver/enums.h`` and the ``SMO_CMS`` enumerator of
  ``src/outfile/include/swmm_output_enums.h`` (flow-unit code 3).
* SWMM ``.rpt``, SPEC 7.2: the section list (open item OI-R2, SPEC section 10) is decided from EPA SWMM
  v5.2.4's ``src/solver/report.c``, not from any hhemt parser (SPEC.md section 7.2).

Everything is deterministic: closed-form values, no random numbers, no clock.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------------------
# Reduced dimensions (SPEC section 2)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Dimensions:
    """The reduced grid and step count of SPEC section 2.

    ``n_steps`` and ``report_step_s`` are the number and spacing of the TRITON output steps and of
    the SWMM reporting periods alike: both writers read the same two fields."""

    rows: int = 6
    cols: int = 4
    n_steps: int = 12
    report_step_s: int = 10
    # SWMM start date as an Excel-style serial day (days since 1899-12-30): 2000-01-01.
    swmm_start_serial: float = 36526.0


# --------------------------------------------------------------------------------------
# SWMM network (SPEC section 2: the observed 14 nodes, 12 links and 11 subcatchments)
# --------------------------------------------------------------------------------------

# Element codes of EPA SWMM src/solver/enums.h: NodeType JUNCTION=0, OUTFALL=1;
# LinkType CONDUIT=0.
NODE_JUNCTION = 0
NODE_OUTFALL = 1
LINK_CONDUIT = 0


@dataclass(frozen=True)
class SubcatchmentSpec:
    id: str
    area_ha: float


@dataclass(frozen=True)
class NodeSpec:
    id: str
    type_code: int
    invert_m: float
    max_depth_m: float

    @property
    def type_name(self) -> str:
        return "OUTFALL" if self.type_code == NODE_OUTFALL else "JUNCTION"


@dataclass(frozen=True)
class LinkSpec:
    id: str
    from_node: str
    to_node: str
    length_m: float
    max_depth_m: float


@dataclass(frozen=True)
class Network:
    subcatchments: tuple[SubcatchmentSpec, ...]
    nodes: tuple[NodeSpec, ...]
    links: tuple[LinkSpec, ...]


def default_network() -> Network:
    """The network's element ids and counts (SPEC section 2: 14 nodes, 12 links, 11
    subcatchments). The invert levels, depths, lengths and areas are fixed values or closed-form in
    the element's position in the list. The ids are nodes J1-J7, collector, sewer_outflow, B1_1,
    B1_2, B2_1, B2_2, dummy_outfall; links C1-C8, CB1_1, CB1_2, CB2_1, CB2_2; subcatchments S1-S11.
    SPEC.md fixes the counts and does not list the ids; the input-file writers must use this function
    so the ids agree."""
    node_ids = ["J1", "J2", "J3", "J4", "J5", "J6", "J7", "collector", "sewer_outflow", "B1_1", "B1_2", "B2_1", "B2_2"]
    nodes = []
    for i, nid in enumerate(node_ids):
        nodes.append(NodeSpec(nid, NODE_JUNCTION, invert_m=0.5 - 0.14 * i, max_depth_m=1.0))
    nodes.append(NodeSpec("dummy_outfall", NODE_OUTFALL, invert_m=-1.0, max_depth_m=0.0))
    link_pairs = [
        ("C1", "J1", "J2"),
        ("C2", "J2", "J3"),
        ("C3", "J3", "J4"),
        ("C4", "J4", "J5"),
        ("C5", "J5", "J6"),
        ("C6", "J6", "J7"),
        ("C7", "J7", "collector"),
        ("C8", "collector", "sewer_outflow"),
        ("CB1_1", "B1_1", "B1_2"),
        ("CB1_2", "B1_2", "J4"),
        ("CB2_1", "B2_1", "B2_2"),
        ("CB2_2", "B2_2", "J5"),
    ]
    links = tuple(LinkSpec(lid, a, b, length_m=10.0, max_depth_m=0.6) for lid, a, b in link_pairs)
    subs = tuple(SubcatchmentSpec(f"S{i}", area_ha=0.05 + 0.01 * i) for i in range(1, 12))
    return Network(subs, tuple(nodes), links)


# --------------------------------------------------------------------------------------
# TRITON raw outputs (SPEC 7.1)
# --------------------------------------------------------------------------------------

TRITON_PREFIXES = ("H", "QX", "QY", "MH")

_PERF_COLUMNS = ("Compute", "MPI", "IO", "Resize", "SWMM", "Other", "Simulation", "Init", "Total")


def _triton_field(prefix: str, step: int, dims: Dimensions, scale: float) -> np.ndarray:
    """Closed-form water-level / momentum fields; ``MH`` is the running maximum of ``H``.

    At the default 12 steps ``H`` rises at every step, so ``MH`` equals ``H`` in the written files;
    the running maximum only differs from ``H`` once ``n_steps`` is large enough for ``H`` to fall."""
    r = np.arange(dims.rows, dtype=np.float64)[:, None]
    c = np.arange(dims.cols, dtype=np.float64)[None, :]
    shape = 1.0 + 0.1 * r + 0.05 * c

    def h_at(k: int) -> np.ndarray:
        return scale * shape * 0.01 * k * (1.0 + 0.05 * math.sin(0.5 * k))

    if prefix == "H":
        return h_at(step)
    if prefix == "MH":
        return np.maximum.reduce([h_at(k) for k in range(1, step + 1)])
    if prefix == "QX":
        return 0.001 * scale * shape * step
    if prefix == "QY":
        return -0.0005 * scale * shape * step
    raise ValueError(prefix)


def write_triton_raster(path: Path, values: np.ndarray) -> None:
    """One TRITON binary raster: float64 rows, float64 cols, then values row by row (SPEC 7.1)."""
    rows, cols = values.shape
    body = np.asarray(values, dtype="<f8", order="C").tobytes()
    path.write_bytes(struct.pack("<dd", float(rows), float(cols)) + body)


def _performance_text(step: int, dims: Dimensions, coupled: bool) -> str:
    """Cumulative per-rank timing text; header and layout per SPEC 7.1 (one rank, then Average).

    ``Compute``, ``IO``, ``SWMM`` and ``Other`` are fixed fractions of the elapsed time ``t``
    (0.70, 0.25, 0.20 when ``coupled`` and 0 otherwise, and 0.05); ``MPI`` and ``Resize`` are always
    0. ``Simulation`` is the sum of those four, so it is 1.2 times ``t`` in the coupled arm and ``t``
    in the TRITON-only arm, and ``Total`` adds the constant ``Init`` of 0.01."""
    t = dims.report_step_s * step * 1.0e-3  # cumulative seconds, closed form
    compute = 0.70 * t
    io = 0.25 * t
    swmm = 0.20 * t if coupled else 0.0
    other = 0.05 * t
    simulation = compute + io + swmm + other
    init = 0.01
    total = simulation + init
    vals = [compute, 0.0, io, 0.0, swmm, other, simulation, init, total]
    row = ", ".join(f"{v:.4g}" for v in vals)
    return "%Rank, " + ", ".join(_PERF_COLUMNS) + f"\n0, {row}\nAverage, {row}\n"


def write_triton_arm(out_dir: Path, *, dims: Dimensions, coupled: bool, n_tasks: int = 1, omp_threads: int = 1) -> None:
    """Write ``bin/``, ``performance/``, ``performance.txt`` and ``log.out`` for one TRITON arm.

    ``performance/performance{step}.txt`` holds one file per step and ``performance.txt`` repeats the
    final step's text; both are in the observed tree (SPEC.md section 7.1).

    ``coupled`` is the TRITON-SWMM arm (non-zero SWMM column); the TRITON-only arm has none.
    ``cfg/`` is not written: SPEC 7.1 leaves its need ``[OPEN]`` (OI-D2, SPEC section 10) and its content holds
    absolute paths that SPEC does not specify for this file.

    ``n_tasks`` and ``omp_threads`` are the MPI tasks and OpenMP threads per task the run states in ``log.out``;
    hhemt reads them back as the run's actual resources and compares them with the configured ones."""
    scale = 1.15 if coupled else 1.0
    bin_dir = out_dir / "bin"
    perf_dir = out_dir / "performance"
    bin_dir.mkdir(parents=True, exist_ok=True)
    perf_dir.mkdir(parents=True, exist_ok=True)
    for step in range(1, dims.n_steps + 1):
        for prefix in TRITON_PREFIXES:
            name = f"{prefix}_{step:02d}_00.out"
            write_triton_raster(bin_dir / name, _triton_field(prefix, step, dims, scale))
        (perf_dir / f"performance{step}.txt").write_text(_performance_text(step, dims, coupled), encoding="utf-8")
    (out_dir / "performance.txt").write_text(_performance_text(dims.n_steps, dims, coupled), encoding="utf-8")
    sim_s = dims.n_steps * dims.report_step_s
    (out_dir / "log.out").write_text(
        "\n---- TRITON RUN INFO ----\n"
        "Machine : golden\n"
        f"nTasks : {n_tasks}\n"
        f"OMP threads per task : {omp_threads}\n"
        "GPUs per task : 0 (CPU-only)\n"
        "Build type : CPU+OMP\n"
        "----------------------------\n"
        f"Simulation ends at t = {sim_s} s\n",
        encoding="utf-8",
    )


# --------------------------------------------------------------------------------------
# SWMM binary .out (SPEC 7.2; EPA SWMM v5.2.4 src/solver/output.c)
# --------------------------------------------------------------------------------------

_MAGICNUMBER = 516114522  # src/solver/consts.h:20
_VERSION = 52004  # src/solver/consts.h:19
_FLOW_UNITS_CMS = 3  # SMO_CMS, src/outfile/include/swmm_output_enums.h (CFS,GPM,MGD,CMS)
_MAX_SYS_RESULTS = 15  # src/solver/enums.h:224
# Result-variable counts with no pollutants: MAX_*_RESULTS - 1 (output.c output_open).
_N_SUBCATCH_VARS = 8
_N_NODE_VARS = 6
_N_LINK_VARS = 5
# enum InputDataType in output.c: INPUT_TYPE_CODE, INPUT_AREA, INPUT_INVERT, INPUT_MAX_DEPTH,
# INPUT_OFFSET, INPUT_LENGTH
_IN_TYPE, _IN_AREA, _IN_INVERT, _IN_MAXDEPTH, _IN_OFFSET, _IN_LENGTH = range(6)


def _i4(*vals: int) -> bytes:
    return struct.pack(f"<{len(vals)}i", *vals)


def _f4(vals) -> bytes:
    return np.asarray(vals, dtype="<f4").tobytes()


def _id(s: str) -> bytes:
    raw = s.encode("ascii")
    return _i4(len(raw)) + raw  # output_saveID: INT4 length then the characters


def build_swmm_out(
    network: Network,
    dims: Dimensions,
    *,
    with_subcatchments: bool,
    with_nodes: bool,
    with_links: bool,
    scale: float = 1.0,
) -> bytes:
    """Return the bytes of a SWMM ``.out`` file, laid out as ``output.c`` writes it.

    Order: header (magic, version, flow units, counts of subcatchments, nodes, links,
    pollutants), element ids, input records, result-variable code tables, start date and report
    step, then one block per reporting period (REAL8 date, subcatchment, node and link results,
    15 system results), then the six-INT4 epilogue (id, input and output start positions, period
    count, error code, magic).

    ``with_subcatchments``, ``with_nodes`` and ``with_links`` choose which element kinds are
    written; an excluded kind has count 0, no ids, no per-element input records and no results.
    ``scale`` multiplies the ramp the per-period result values follow, which grows linearly with
    the period number."""
    subs = network.subcatchments if with_subcatchments else ()
    nodes = network.nodes if with_nodes else ()
    links = network.links if with_links else ()
    out = bytearray()
    out += _i4(_MAGICNUMBER, _VERSION, _FLOW_UNITS_CMS, len(subs), len(nodes), len(links), 0)
    id_start = len(out)
    for s in subs:
        out += _id(s.id)
    for n in nodes:
        out += _id(n.id)
    for lk in links:
        out += _id(lk.id)
    input_start = len(out)
    # subcatchment inputs: count 1, code INPUT_AREA, then one REAL4 area each
    out += _i4(1, _IN_AREA)
    out += _f4([s.area_ha for s in subs])
    # node inputs: count 3, codes, then per node INT4 type + REAL4 invert + REAL4 max depth
    out += _i4(3, _IN_TYPE, _IN_INVERT, _IN_MAXDEPTH)
    for n in nodes:
        out += _i4(n.type_code) + _f4([n.invert_m, n.max_depth_m])
    # link inputs: count 5, codes (TYPE, OFFSET, OFFSET, MAX_DEPTH, LENGTH), then per link
    # INT4 type + 4 REAL4 (offset1, offset2, max depth, length)
    out += _i4(5, _IN_TYPE, _IN_OFFSET, _IN_OFFSET, _IN_MAXDEPTH, _IN_LENGTH)
    for lk in links:
        out += _i4(LINK_CONDUIT) + _f4([0.0, 0.0, lk.max_depth_m, lk.length_m])
    # result-variable code tables (SubcatchResultType 0-7, NodeResultType 0-5,
    # LinkResultType 0-4, system 0-14)
    out += _i4(_N_SUBCATCH_VARS, *range(_N_SUBCATCH_VARS))
    out += _i4(_N_NODE_VARS, *range(_N_NODE_VARS))
    out += _i4(_N_LINK_VARS, *range(_N_LINK_VARS))
    out += _i4(_MAX_SYS_RESULTS, *range(_MAX_SYS_RESULTS))
    # starting report date (REAL8) and report step (INT4)
    out += struct.pack("<d", dims.swmm_start_serial) + _i4(dims.report_step_s)
    output_start = len(out)

    step_days = dims.report_step_s / 86400.0
    n_links = len(links)
    for k in range(1, dims.n_steps + 1):
        ramp = scale * k / dims.n_steps
        out += struct.pack("<d", dims.swmm_start_serial + k * step_days)
        for j in range(len(subs)):
            # rainfall, snow depth, evap, infil, runoff, gw flow, gw elev, soil moisture
            out += _f4([5.0 * ramp, 0.0, 0.0, 0.5 * ramp, 0.001 * ramp * (1 + 0.1 * j), 0.0, 0.0, 0.0])
        for j, n in enumerate(nodes):
            depth = 0.0 if n.type_code == NODE_OUTFALL else 0.02 * ramp * (1 + 0.05 * j)
            # depth, head, volume, lateral inflow, total inflow, overflow
            out += _f4([depth, n.invert_m + depth, 0.1 * depth, 0.0005 * ramp, 0.001 * ramp, 0.0])
        for j in range(n_links):
            q = 0.001 * ramp * (1 + 0.05 * j)
            # flow, depth, velocity, volume, capacity
            out += _f4([q, 0.01 * ramp, 0.1 * ramp, 0.05 * ramp, 0.02 * ramp])
        sys_vals = [0.0] * _MAX_SYS_RESULTS
        sys_vals[1] = 5.0 * ramp  # rainfall
        sys_vals[4] = 0.001 * ramp * len(subs)  # runoff
        sys_vals[8] = sys_vals[4]  # total lateral inflow
        out += _f4(sys_vals)
    out += _i4(id_start, input_start, output_start, dims.n_steps, 0, _MAGICNUMBER)
    return bytes(out)


# --------------------------------------------------------------------------------------
# SWMM .rpt (SPEC 7.2; open item OI-R2, SPEC section 10): section list decided from EPA SWMM's report.c
# --------------------------------------------------------------------------------------

_ENDED = "Analysis ended on:  Sat Jan  1 00:00:12 2000"
_BEGUN = "Analysis begun on:  Sat Jan  1 00:00:00 2000"


def _banner(title: str) -> str:
    stars = "*" * len(title)
    return f"  {stars}\n  {title}\n  {stars}\n"


def build_swmm_rpt(network: Network, dims: Dimensions, *, kind: str) -> str:
    """Return a SWMM report. ``kind``: ``"full"`` (runoff and routing), ``"hydraulics"``
    (routing only, no subcatchments) or ``"hydro"`` (runoff only). The text always ends with the
    ``Analysis begun on`` and ``Analysis ended on`` lines followed by ``Total elapsed time``, and no
    line has ``ERROR`` as its first non-blank text.

    The numeric cells are fixed literals, mostly zero, except the maximum-HGL column of the node depth
    table, which holds each node's invert level; none is computed from the values in the ``.out``
    written for the same arm."""
    if kind not in {"full", "hydraulics", "hydro"}:
        raise ValueError(kind)
    routing = kind != "hydro"
    runoff = kind != "hydraulics"
    end_h = dims.n_steps * dims.report_step_s
    lines: list[str] = []
    lines.append("")
    lines.append("  EPA STORM WATER MANAGEMENT MODEL - VERSION 5.2 (Build 5.2.4)")
    lines.append("  ------------------------------------------------------------")
    lines.append("")
    lines.append("  Synthetic golden model")
    lines.append("  ")
    lines.append(_banner("Analysis Options").rstrip("\n"))
    lines.append("  Flow Units ............... CMS")
    lines.append("  Process Models:")
    lines.append(f"    Rainfall/Runoff ........ {'YES' if runoff else 'NO'}")
    lines.append("    RDII ................... NO")
    lines.append("    Snowmelt ............... NO")
    lines.append("    Groundwater ............ NO")
    lines.append(f"    Flow Routing ........... {'YES' if routing else 'NO'}")
    if routing:
        lines.append("    Ponding Allowed ........ NO")
    lines.append("    Water Quality .......... NO")
    lines.append("  Infiltration Method ...... HORTON")
    if routing:
        lines.append("  Flow Routing Method ...... DYNWAVE")
        lines.append("  Surcharge Method ......... EXTRAN")
    lines.append("  Starting Date ............ 01/01/2000 00:00:00")
    lines.append(f"  Ending Date .............. 01/01/2000 00:{end_h // 60:02d}:{end_h % 60:02d}")
    lines.append("  Antecedent Dry Days ...... 0.0")
    lines.append(f"  Report Time Step ......... 00:00:{dims.report_step_s:02d}")
    lines.append("  Wet Time Step ............ 00:00:10")
    lines.append("  Dry Time Step ............ 00:01:00")
    if routing:
        lines.append("  Routing Time Step ........ 1.00 sec")
        lines.append("  Variable Time Step ....... YES")
        lines.append("  Maximum Trials ........... 8")
        lines.append("  Number of Threads ........ 1")
        lines.append("  Head Tolerance ........... 0.001524 m")
    lines += ["  ", "  "]
    if runoff:
        lines.append("  **************************        Volume         Depth")
        lines.append("  Runoff Quantity Continuity     hectare-m            mm")
        lines.append("  **************************     ---------       -------")
        for label in (
            "Total Precipitation",
            "Evaporation Loss",
            "Infiltration Loss",
            "Surface Runoff",
            "Final Storage",
        ):
            lines.append(f"  {label + ' ':.<25s}         0.000         0.000")
        lines.append("  Continuity Error (%) .....         0.000")
        lines += ["  ", "  "]
    if routing:
        lines.append("  **************************        Volume        Volume")
        lines.append("  Flow Routing Continuity        hectare-m      10^6 ltr")
        lines.append("  **************************     ---------     ---------")
        for label in (
            "Dry Weather Inflow",
            "Wet Weather Inflow",
            "Groundwater Inflow",
            "RDII Inflow",
            "External Inflow",
            "External Outflow",
            "Flooding Loss",
            "Evaporation Loss",
            "Exfiltration Loss",
            "Initial Stored Volume",
            "Final Stored Volume",
        ):
            lines.append(f"  {label + ' ':.<25s}         0.000         0.000")
        lines.append("  Continuity Error (%) .....         0.000")
        lines += ["  ", "  "]
        lines += _banner("Time-Step Critical Elements").splitlines() + ["  None", "  ", "  "]
        lines += _banner("Highest Flow Instability Indexes").splitlines() + ["  All links are stable.", "  ", "  "]
        lines += _banner("Most Frequent Nonconverging Nodes").splitlines()
        lines += ["  Convergence obtained at all time steps.", "  ", "  "]
        lines += _banner("Routing Time Step Summary").splitlines()
        lines += [
            "  Minimum Time Step           :     1.00 sec",
            "  Average Time Step           :     1.00 sec",
            "  Maximum Time Step           :     1.00 sec",
            "  % of Time in Steady State   :     0.00",
            "  Average Iterations per Step :     2.00",
            "  % of Steps Not Converging   :     0.00",
            "  ",
            "  ",
        ]
    if runoff:
        lines += _banner("Subcatchment Runoff Summary").splitlines()
        lines.append("  ")
        rule = "  " + "-" * 126
        lines.append(rule)
        lines.append(
            "                            Total      Total      Total      Total     Imperv       Perv      Total"
            "       Total     Peak  Runoff"
        )
        lines.append(
            "                           Precip      Runon       Evap      Infil     Runoff     Runoff     Runoff"
            "      Runoff   Runoff   Coeff"
        )
        lines.append(
            "  Subcatchment                 mm         mm         mm         mm         mm         mm         mm"
            "    10^6 ltr      CMS"
        )
        lines.append(rule)
        for s in network.subcatchments:
            lines.append(f"  {s.id:<22s}" + "      0.00" * 8 + "     0.00   0.000")
        lines += ["  ", "  "]
    if routing:
        lines += _banner("Node Depth Summary").splitlines()
        lines.append("  ")
        rule = "  " + "-" * 81
        lines += [
            rule,
            "                               Average  Maximum  Maximum  Time of Max    Reported",
            "                                 Depth    Depth      HGL   Occurrence   Max Depth",
            "  Node                 Type       Meters   Meters   Meters  days hr:min      Meters",
            rule,
        ]
        for n in network.nodes:
            lines.append(
                f"  {n.id:<20s} {n.type_name:<8s}     0.00     0.00 {n.invert_m:8.2f}     0  00:00        0.00"
            )
        lines += ["  ", "  "]
        lines += _banner("Node Inflow Summary").splitlines()
        lines.append("  ")
        rule = "  " + "-" * 97
        lines += [
            rule,
            "                                Maximum  Maximum                  Lateral       Total        Flow",
            "                                Lateral    Total  Time of Max      Inflow      Inflow     Balance",
            "                                 Inflow   Inflow   Occurrence      Volume      Volume       Error",
            "  Node                 Type           CMS      CMS  days hr:min    10^6 ltr    10^6 ltr     Percent",
            rule,
        ]
        for n in network.nodes:
            lines.append(
                f"  {n.id:<20s} {n.type_name:<8s}"
                "     0.000    0.000     0  00:00           0           0       0.000 ltr"
            )
        lines += ["  ", "  "]
        lines += _banner("Node Surcharge Summary").splitlines() + ["  ", "  No nodes were surcharged.", "  ", "  "]
        lines += _banner("Node Flooding Summary").splitlines() + ["  ", "  No nodes were flooded.", "  ", "  "]
        lines += _banner("Outfall Loading Summary").splitlines()
        lines.append("  ")
        rule = "  " + "-" * 59
        lines += [
            rule,
            "                         Flow       Avg       Max       Total",
            "                         Freq      Flow      Flow      Volume",
            "  Outfall Node           Pcnt       CMS       CMS    10^6 ltr",
            rule,
        ]
        for n in network.nodes:
            if n.type_code == NODE_OUTFALL:
                lines.append(f"  {n.id:<20s}      0.00     0.000     0.000       0.000")
        lines += [rule, "  System                 0.00     0.000     0.000       0.000", "  ", "  "]
        lines += _banner("Link Flow Summary").splitlines()
        lines.append("  ")
        rule = "  " + "-" * 77
        lines += [
            rule,
            "                               Maximum  Time of Max   Maximum    Max/    Max/",
            "                                |Flow|   Occurrence   |Veloc|    Full    Full",
            "  Link                 Type          CMS  days hr:min     m/sec    Flow   Depth",
            rule,
        ]
        for lk in network.links:
            lines.append(f"  {lk.id:<20s} CONDUIT     0.000     0  00:00      0.00    0.00    0.00")
        lines += ["  ", "  "]
        lines += _banner("Flow Classification Summary").splitlines()
        lines.append("  ")
        rule = "  " + "-" * 85
        lines += [
            rule,
            "                      Adjusted    ---------- Fraction of Time in Flow Class ---------- ",
            "                       /Actual         Up    Down  Sub   Sup   Up    Down  Norm  Inlet ",
            "  Conduit               Length    Dry  Dry   Dry   Crit  Crit  Crit  Crit  Ltd   Ctrl  ",
            rule,
        ]
        for lk in network.links:
            lines.append(f"  {lk.id:<20s}      1.00   1.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00")
        lines += ["  ", "  "]
        lines += _banner("Conduit Surcharge Summary").splitlines() + ["  ", "  No conduits were surcharged.", "  "]
    lines += ["", f"  {_BEGUN}", f"  {_ENDED}", "  Total elapsed time: 00:00:12"]
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# One event
# --------------------------------------------------------------------------------------


def write_event_raw(
    event_dir: Path,
    *,
    arms: tuple[str, ...] = ("triton", "tritonswmm", "swmm"),
    network: Network | None = None,
    dims: Dimensions | None = None,
    n_tasks: int = 1,
    omp_threads: int = 1,
) -> None:
    """Write the raw layer of one event directory (SPEC section 3 and 7).

    ``triton``    -> ``out_triton/``
    ``tritonswmm``-> ``out_tritonswmm/`` and ``out_tritonswmm/swmm/hydraulics.{out,rpt}``
    ``swmm``      -> ``out_swmm/full.{out,rpt}``
    and, when ``swmm`` or ``tritonswmm`` is enabled, ``swmm/hydro.{out,rpt}`` (the runoff-only
    model: subcatchments and nodes, no links, over the same network as the other arms). SPEC.md section 4.8
    lists ``swmm/hydro.out`` and ``swmm/hydro.rpt`` among the prepared inputs that the raw-layer readers
    and the scenario-completion checks read.

    ``n_tasks`` and ``omp_threads`` are written into each TRITON arm's ``log.out``."""
    net = network or default_network()
    d = dims or Dimensions()
    event_dir = Path(event_dir)
    unknown = set(arms) - {"triton", "tritonswmm", "swmm"}
    if unknown:
        raise ValueError(f"unknown arm(s): {sorted(unknown)}")
    if "triton" in arms:
        write_triton_arm(event_dir / "out_triton", dims=d, coupled=False, n_tasks=n_tasks, omp_threads=omp_threads)
    if "tritonswmm" in arms:
        out = event_dir / "out_tritonswmm"
        write_triton_arm(out, dims=d, coupled=True, n_tasks=n_tasks, omp_threads=omp_threads)
        (out / "swmm").mkdir(parents=True, exist_ok=True)
        (out / "swmm" / "hydraulics.out").write_bytes(
            build_swmm_out(net, d, with_subcatchments=False, with_nodes=True, with_links=True, scale=1.15)
        )
        (out / "swmm" / "hydraulics.rpt").write_text(build_swmm_rpt(net, d, kind="hydraulics"), encoding="ascii")
    if "swmm" in arms:
        out = event_dir / "out_swmm"
        out.mkdir(parents=True, exist_ok=True)
        (out / "full.out").write_bytes(
            build_swmm_out(net, d, with_subcatchments=True, with_nodes=True, with_links=True)
        )
        (out / "full.rpt").write_text(build_swmm_rpt(net, d, kind="full"), encoding="ascii")
    if "swmm" in arms or "tritonswmm" in arms:
        sw = event_dir / "swmm"
        sw.mkdir(parents=True, exist_ok=True)
        (sw / "hydro.out").write_bytes(
            build_swmm_out(net, d, with_subcatchments=True, with_nodes=True, with_links=False)
        )
        (sw / "hydro.rpt").write_text(build_swmm_rpt(net, d, kind="hydro"), encoding="ascii")
