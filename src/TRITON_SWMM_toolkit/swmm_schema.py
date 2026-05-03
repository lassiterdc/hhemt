"""EPA SWMM-5 canonical column names as used by `swmmio.Model.inp.*` DataFrames.

Single source of truth for the section column-name strings the renderers and
the GIS-layer exporter read from a parsed `.inp` file. swmmio mirrors EPA's
SWMM-5 manual exactly (see EPA SWMM 5.2 Reference Manual, Volume II — User's
Manual, App. D INP File Format).

Group constants by section name. SECTION_NAMES are the bracketed `[SECTION]`
identifiers (used as dict keys / source-paths sub-bullets in caption RSTs).
"""

from __future__ import annotations

from typing import Final

# ---- [COORDINATES] ---------------------------------------------------------
COORDS_X: Final = "X"
COORDS_Y: Final = "Y"

# ---- [JUNCTIONS] -----------------------------------------------------------
JUNC_INVERT_ELEV: Final = "InvertElev"
JUNC_MAX_DEPTH: Final = "MaxDepth"

# ---- [OUTFALLS] ------------------------------------------------------------
OUTFALL_INVERT_ELEV: Final = "InvertElev"
OUTFALL_TYPE: Final = "OutfallType"

# ---- [SUBCATCHMENTS] -------------------------------------------------------
SUBCATCH_OUTLET: Final = "Outlet"
SUBCATCH_AREA: Final = "Area"
SUBCATCH_PERC_IMPERV: Final = "PercImperv"
SUBCATCH_WIDTH: Final = "Width"
SUBCATCH_PERC_SLOPE: Final = "PercSlope"
SUBCATCH_RAINGAGE: Final = "Raingage"

# Properties to copy from the [SUBCATCHMENTS] row onto a subcatchment GeoJSON
# feature (lowercased) — used by `_swmm_gis_layers._write_subcatchments`.
SUBCATCHMENT_FEATURE_PROPS: Final[tuple[str, ...]] = (
    SUBCATCH_OUTLET,
    SUBCATCH_AREA,
    SUBCATCH_PERC_IMPERV,
    SUBCATCH_WIDTH,
    SUBCATCH_PERC_SLOPE,
    SUBCATCH_RAINGAGE,
)

# ---- [CONDUITS] ------------------------------------------------------------
CONDUIT_INLET_NODE: Final = "InletNode"
CONDUIT_OUTLET_NODE: Final = "OutletNode"
CONDUIT_LENGTH: Final = "Length"
CONDUIT_ROUGHNESS: Final = "Roughness"

# ---- Bracketed [SECTION] identifiers (caption RST source bullets) ---------
SECTION_CONDUITS: Final = "[CONDUITS]"
SECTION_COORDINATES: Final = "[COORDINATES]"
SECTION_JUNCTIONS: Final = "[JUNCTIONS]"
SECTION_OUTFALLS: Final = "[OUTFALLS]"
SECTION_SUBCATCHMENTS: Final = "[SUBCATCHMENTS]"
SECTION_POLYGONS: Final = "[POLYGONS]"
