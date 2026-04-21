"""Synthesize minimal SWMM .inp templates and subcatchment->raingage mapping.

Topology:
    raingage RG_synth (timeseries-driven, read from weather .nc via scenario wiring)
    subcatchments S1 (impervious top half), S2 (pervious bottom half) -> both to RG_synth
    junctions J1, J2, J3 running north->south along valley
    conduits C1 (J1->J2), C2 (J2->J3)
    outfall OUT1 at southern boundary

Three template variants are written:
    swmm_hydraulics.inp — hydraulics-only, no [SUBCATCHMENTS]/[SUBAREAS]/[INFILTRATION]
    swmm_hydrology.inp  — hydrology-only, no [CONDUITS]/[XSECTIONS]
    swmm_full.inp       — all sections
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_COMMON_HEADER = """\
[TITLE]
Synthetic TRITON-SWMM test model

[OPTIONS]
FLOW_UNITS           CMS
INFILTRATION         HORTON
FLOW_ROUTING         DYNWAVE
LINK_OFFSETS         DEPTH
MIN_SLOPE            0
START_DATE           01/01/2000
START_TIME           00:00:00
REPORT_START_DATE    01/01/2000
REPORT_START_TIME    00:00:00
END_DATE             01/01/2000
END_TIME             01:00:00
SWEEP_START          01/01
SWEEP_END            12/31
DRY_DAYS             0
REPORT_STEP          00:00:10
WET_STEP             00:00:10
DRY_STEP             00:01:00
ROUTING_STEP         00:00:01
ALLOW_PONDING        NO
INERTIAL_DAMPING     PARTIAL
VARIABLE_STEP        0.75
LENGTHENING_STEP     0
MIN_SURFAREA         0
NORMAL_FLOW_LIMITED  BOTH
SKIP_STEADY_STATE    NO
FORCE_MAIN_EQUATION  H-W

[RAINGAGES]
;;Name         Format    Interval SCF   Source
RG_synth       INTENSITY 0:01     1.0   TIMESERIES  RG_synth_ts

[TIMESERIES]
;;Name          Date        Time     Value
RG_synth_ts     01/01/2000  00:00    0
"""

_SUBCATCHMENTS_BLOCK = """\
[SUBCATCHMENTS]
;;Name  Raingage  Outlet  Area   %Imperv  Width   %Slope  CurbLen
S1      RG_synth  J1      1.5    100      50      1.0     0
S2      RG_synth  J2      1.5    0        50      1.0     0

[SUBAREAS]
;;Subcatchment  N-Imperv  N-Perv  S-Imperv  S-Perv  PctZero  RouteTo  PctRouted
S1              0.015     0.035   0.05      0.05    25       OUTLET
S2              0.015     0.035   0.05      0.05    25       OUTLET

[INFILTRATION]
;;Subcatchment  MaxRate  MinRate  Decay  DryTime  MaxInfil
S1              3.0      0.5      4.0    7        0
S2              3.0      0.5      4.0    7        0
"""

_LINKS_BLOCK = """\
[JUNCTIONS]
;;Name  Elev   MaxDepth  InitDepth  SurDepth  Aponded
J1      8.0    2.0       0          0         0
J2      5.0    2.0       0          0         0
J3      2.0    2.0       0          0         0

[OUTFALLS]
;;Name  Elev  Type   Gated  RouteTo
OUT1    0.0   FREE   NO

[CONDUITS]
;;Name  FromNode  ToNode  Length  Roughness  InOffset  OutOffset  InitFlow  MaxFlow
C1      J1        J2      150     0.013      0         0          0         0
C2      J2        J3      150     0.013      0         0          0         0
C3      J3        OUT1    50      0.013      0         0          0         0

[XSECTIONS]
;;Link  Shape     Geom1  Geom2  Geom3  Geom4  Barrels  Culvert
C1      CIRCULAR  1.0    0      0      0      1
C2      CIRCULAR  1.0    0      0      0      1
C3      CIRCULAR  1.0    0      0      0      1

[INFLOWS]
;;Node  Parameter  TimeSeries  Type  Mfactor  Sfactor  Baseline  Pattern

[CURVES]
;;Name  Type  X-Value  Y-Value
"""

_COORDS_BLOCK_TEMPLATE = """\
[COORDINATES]
;;Node  X-Coord  Y-Coord
{coord_rows}

[REPORT]
INPUT      NO
CONTINUITY YES
FLOWSTATS  YES
CONTROLS   NO
SUBCATCHMENTS ALL
NODES ALL
LINKS ALL
"""


def _coord_rows(params) -> str:
    """Place J1/J2/J3 on real DEM cells along the central valley.

    All nodes sit at cell centers (offset by 0.5 * cell_size) and strictly
    inside the watershed polygon, which is itself inset one cell from the DEM
    extent. Outer-edge cells would be rejected by TRITON-SWMM's coupling
    step with a `node_swmm ... is out of bounds` error.
    """
    cs = params.cell_size_m
    x_center = params.xllcorner + (params.n_cols // 2 + 0.5) * cs
    # Rows counted from the southern edge. Stay two cells inside the top,
    # two cells inside the bottom. For the default 30-row grid that's rows
    # 27, 15, and 2. OUT1 sits one cell south of J3 to give it a valid
    # downstream cell on the southern boundary of the watershed.
    y_top = params.yllcorner + (params.n_rows - 3 + 0.5) * cs
    y_mid = params.yllcorner + (params.n_rows // 2 + 0.5) * cs
    y_bot = params.yllcorner + (2 + 0.5) * cs
    # Put OUT1 in the same DEM cell as J3. This is intentional: the toolkit's
    # scenario_inputs.py step that groups nodes by DEM cell uses
    # pd.concat(lst_grps_more_than_1_node) which raises "No objects to
    # concatenate" when every node is in a distinct cell. On the 537×551
    # Norfolk grid overlaps happen naturally; on the 20×30 synth grid they
    # don't unless we force one. C3 (J3->OUT1) remains a 50 m conduit — the
    # co-location is a topology-level hack, not a physics change.
    y_out = y_bot
    return "\n".join([
        f"J1  {x_center:.2f}  {y_top:.2f}",
        f"J2  {x_center:.2f}  {y_mid:.2f}",
        f"J3  {x_center:.2f}  {y_bot:.2f}",
        f"OUT1 {x_center:.2f}  {y_out:.2f}",
    ])


def build_templates(params, cache_dir: Path):
    """Write swmm_hydraulics.inp, swmm_hydrology.inp, swmm_full.inp."""
    coords = _COORDS_BLOCK_TEMPLATE.format(coord_rows=_coord_rows(params))
    header = _COMMON_HEADER

    hydraulics = cache_dir / "swmm_hydraulics.inp"
    hydrology = cache_dir / "swmm_hydrology.inp"
    full = cache_dir / "swmm_full.inp"

    hydraulics.write_text(header + _LINKS_BLOCK + coords, encoding="utf-8")
    # Hydrology file needs JUNCTIONS + OUTFALLS so SUBCATCHMENTS can reference
    # J1/J2/OUT1 as outlet nodes. Including _LINKS_BLOCK in full gives
    # CONDUITS/XSECTIONS too — harmless for hydrology-only runs; SWMM tolerates
    # unreferenced link definitions.
    hydrology.write_text(header + _SUBCATCHMENTS_BLOCK + _LINKS_BLOCK + coords, encoding="utf-8")
    full.write_text(header + _SUBCATCHMENTS_BLOCK + _LINKS_BLOCK + coords, encoding="utf-8")

    return hydraulics, hydrology, full


def build_subcatchment_raingage_mapping(params, dest: Path) -> Path:
    df = pd.DataFrame(
        {
            "subcatchment_id": ["S1", "S2"],
            "raingage_id": ["RG_synth", "RG_synth"],
            "mrms_col": ["RG_synth", "RG_synth"],
        }
    )
    df.to_csv(dest, index=False)
    return dest
