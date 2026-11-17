"""Render the TRITON(-SWMM) cfg file from the toolkit's own inputs.

There is no cfg template for a user to supply: every key is either rendered from a config field or
scenario path (the ``${NAME}`` placeholders), or is a toolkit constant fixed in the text below
(``print_option``, ``input_format``, ``outfile_pattern`` and the others the processor and the input
writers depend on). ``render_triton_cfg`` returns the text; the scenario then adds the keys that
are toolkit-owned and appended (``output_folder``, ``swmm_coupling_nodes``, the snapshot keys).

The text is today's synthetic ``_TEMPLATE`` moved here with its comments as they were. With comments and
blank lines removed it holds the same keys and values as the Norfolk example's published cfg template.

The same module holds the two optional TRITON inputs a user supplies: observation points (a CSV the toolkit
checks and rewrites into the location file TRITON reads) and initial-condition rasters (checked against the
processed DEM's grid and rewritten into the header-less form TRITON's reader takes). ``triton_input_settings`` is
the one entry the scenario calls for both.
"""

from __future__ import annotations

import csv
import functools
import hashlib
import io
import math
from dataclasses import dataclass
from pathlib import Path
from string import Template

import numpy as np

from hhemt.exceptions import ConfigurationError, ProcessingError

_TEMPLATE = """\
#---------------------------------------------------------------------------
# TRITON-SWMM config file - ${CASE_DESC}
#---------------------------------------------------------------------------

#---------------------------------------------------------------------------
# Mandatory input files
#---------------------------------------------------------------------------
dem_filename="${DEM}"
inp_filename="${SWMM}"
#---------------------------------------------------------------------------
# Format=BIN or ASC
# Option=PAR or SEQ
#---------------------------------------------------------------------------
input_format=ASC
input_option=SEQ
outfile_pattern="%s/%s/%s_%02d_%02d"
output_format=${OUT_FORMAT}
output_option=SEQ

#---------------------------------------------------------------------------
# TRITON-SWMM specific inputs
#---------------------------------------------------------------------------
manhole_diameter=${MH_DIAM}
manhole_loss=${MH_LOSS}

#---------------------------------------------------------------------------
# Hydrograph and flow locations information
#---------------------------------------------------------------------------
num_sources=${NUM_SOURCES}
hydrograph_filename="${HYDROGRAPH}"
src_loc_file="${HYDO_SRC_LOC}"

#---------------------------------------------------------------------------
# Manning input files
#---------------------------------------------------------------------------
${MAN_FILE_TOGGLE}n_infile="${MANNINGS}"
${CONST_MAN_TOGGLE}const_mann=${CONST_MAN}

#---------------------------------------------------------------------------
# Runoff related information
#---------------------------------------------------------------------------
num_runoffs=0
#runoff_filename=""
#runoff_file=""

#---------------------------------------------------------------------------
# External boundaries
#---------------------------------------------------------------------------
num_extbc=${NUM_EXT_BC}
extbc_dir="${EXTBC_DIR}"
extbc_file="${EXTBC_FILE}"

#---------------------------------------------------------------------------
# time_series_flag=1 to activate, 0 to deactivate
#---------------------------------------------------------------------------
observation_loc_file="${OBSERVATION_LOC_FILE}"
time_series_flag=${TIME_SERIES_FLAG}

#---------------------------------------------------------------------------
# print_option=h to output just the h, huv to output h plus QX and QY (unit discharge in m2/s, h*u and h*v;
# the solver's own comment calls these u and v)
#---------------------------------------------------------------------------
print_option=huv
max_value_print_option=h

#---------------------------------------------------------------------------
# Start and duration of simulation time in seconds
#---------------------------------------------------------------------------
sim_start_time=0
sim_duration=${SIM_DUR_S}

#---------------------------------------------------------------------------
# If checkpoint_id is 0 that means a clean start
#---------------------------------------------------------------------------
checkpoint_id=0

#---------------------------------------------------------------------------
# time_increment_fixed=${FIXED_TSTEP} is for variable dt, 1 for constant dt
#---------------------------------------------------------------------------
time_step=${TSTEP_S}
time_increment_fixed=${FIXED_TSTEP}

#---------------------------------------------------------------------------
# Print interval in seconds of simulation time
#---------------------------------------------------------------------------
print_interval=${REPORTING_TSTEP_S}

#---------------------------------------------------------------------------
# Initial input files
#---------------------------------------------------------------------------
#h_infile="input/inith/asc/case05.inith"
#qx_infile="input/initqx/asc/case05.initqx"
#qy_infile="input/initqy/asc/case05.initqy"

#---------------------------------------------------------------------------
# Other variables
#---------------------------------------------------------------------------
it_count=0
courant=${COURANT}
hextra=${HEXTRA}
gpu_direct_flag=${GPU_DIRECT}
domain_decomposition=${DOMAIN_DECOMP}
open_boundaries=${OPEN_BOUNDARIES}
"""


# The values a toolkit-owned placeholder takes when the caller gives none. COURANT, HEXTRA and FIXED_TSTEP
# are config fields, and the caller supplies them; the defaults are the values the cfg has always carried.
_DEFAULT_IT_PRINT = 10000
_DEFAULTS = {
    "FIXED_TSTEP": 0,
    "GPU_DIRECT": 0,
    "DOMAIN_DECOMP": "static",
    "TIME_SERIES_FLAG": 0,
    "OBSERVATION_LOC_FILE": "",
}
# The two settings with no default anywhere: the analysis config requires them, so the mapping must carry them.
_REQUIRED = ("COURANT", "HEXTRA")

# Each optional initial-condition raster: its mapping key, the cfg key TRITON reads, and the analysis-config field
# that carries it. One table, so the renderer, the grid check and the field names cannot drift apart.
_INITIAL_CONDITIONS = (
    ("H_INFILE", "h_infile", "TRITON_initial_depth_raster"),
    ("QX_INFILE", "qx_infile", "TRITON_initial_qx_raster"),
    ("QY_INFILE", "qy_infile", "TRITON_initial_qy_raster"),
)

# The cfg keys of the coupling-diagnostics group, which the toolkit writes only into the COUPLED cfg and only when the
# analysis config departs from the default. The spelling is load-bearing in the solver repository, which reads each
# through `argsd(key, argmap, ...)`: a key spelled differently silently switches nothing on and no cfg parse rejects
# it. One toggle writes the four capture keys together (the uniform set of the experiment); the links, the interval
# and the lag are separate because each has its own value.
DIAG_CAPTURE_KEYS = ("swmm_diag_steps", "diag_domain_balance", "diag_wet_cells", "diag_dt_limiter")
SWMM_DIAG_LINKS_KEY = "swmm_diag_links"
SWMM_DIAG_NODE_PATH3_KEY = "swmm_diag_node_path3_per_step"
SWMM_COUPLING_INTERVAL_KEY = "swmm_coupling_interval"
SWMM_COUPLING_LAG_KEY = "swmm_coupling_lag"
#: The value of ``swmm_diagnostic_links`` that records every link; any other value is the path of a file naming them.
DIAG_LINKS_ALL = "ALL"

# TRITON's `print_option`/`inp_filename` handling for a TRITON-only cfg: the coupled input is commented out.
_DISABLED_INP = 'inp_filename="#DISABLED_FOR_TRITON_ONLY"'
_TRITON_ONLY_INP = '#inp_filename=""  # TRITON-only mode (no SWMM coupling)'


def render_triton_cfg(mapping: dict, *, coupled: bool) -> str:
    """The cfg text for one scenario.

    ``mapping`` carries one value per ``${NAME}`` placeholder of the text. For ``coupled=False`` the SWMM
    input line is commented out, and ``mapping["SWMM"]`` must be ``#DISABLED_FOR_TRITON_ONLY``.

    Beyond the placeholders, the mapping may carry the optional keys ``DIAG_CAPTURES``, ``DIAG_LINKS``,
    ``DIAG_NODE_PATH3``, ``COUPLING_INTERVAL`` and ``COUPLING_LAG`` (see ``triton_cfg_settings``). Each adds a
    line to a COUPLED cfg only when it departs from the solver's default, so a mapping without them renders the
    cfg it always rendered.
    """
    absent = [key for key in _REQUIRED if key not in mapping]
    if absent:
        raise ProcessingError(
            operation="render_triton_cfg", reason=f"the mapping lacks {absent}, which have no default"
        )
    values = {**_DEFAULTS, **mapping}
    text = Template(_TEMPLATE).safe_substitute(values)
    if not coupled:
        text = text.replace(_DISABLED_INP, _TRITON_ONLY_INP)
    # Keys written only when they differ from what TRITON reads by default, so the default cfg carries no line a
    # plain render lacks: `it_print` (default 10000), `factor_interval_domain_decomposition` (TRITON reads it only
    # under dynamic decomposition), `print_observation` (default: the reporting step) and the three
    # initial-condition rasters (absent: TRITON starts dry).
    extra = []
    if values.get("IT_PRINT") not in (None, _DEFAULT_IT_PRINT):
        extra.append(f"it_print={values['IT_PRINT']}")
    if values["DOMAIN_DECOMP"] == "dynamic" and values.get("DECOMP_CHECK_FACTOR") is not None:
        extra.append(f"factor_interval_domain_decomposition={values['DECOMP_CHECK_FACTOR']}")
    if values.get("PRINT_OBSERVATION") is not None:
        extra.append(f"print_observation={_number_text(values['PRINT_OBSERVATION'])}")
    for placeholder, key, _field in _INITIAL_CONDITIONS:
        if values.get(placeholder):
            extra.append(f'{key}="{values[placeholder]}"')
    # The coupling-diagnostics group acts on the SWMM coupling surface, so the TRITON-only cfg never carries it.
    # Each line is written only when its value departs from the solver's own default, so a run that names a default
    # explicitly renders the same bytes as one that leaves it unset.
    if coupled:
        if values.get("DIAG_CAPTURES"):
            extra.extend(f"{key}=1" for key in DIAG_CAPTURE_KEYS)
        if values.get("DIAG_NODE_PATH3"):
            extra.append(f"{SWMM_DIAG_NODE_PATH3_KEY}=1")
        links = values.get("DIAG_LINKS")
        if links is not None:
            written = links if links == DIAG_LINKS_ALL else f'"{links}"'
            extra.append(f"{SWMM_DIAG_LINKS_KEY}={written}")
        if values.get("COUPLING_INTERVAL") not in (None, 1):
            extra.append(f"{SWMM_COUPLING_INTERVAL_KEY}={values['COUPLING_INTERVAL']}")
        if values.get("COUPLING_LAG") not in (None, 0):
            extra.append(f"{SWMM_COUPLING_LAG_KEY}={values['COUPLING_LAG']}")
    if extra:
        text = text.rstrip("\n") + "\n" + "\n".join(extra) + "\n"
    return text


def triton_cfg_settings(cfg_analysis, cfg_hpc_system) -> dict:
    """The TRITON cfg settings that are config fields, by the placeholder or key each one renders to.

    A run with no HPC system config (a local run) takes the setting an HPC config at its defaults gives.

    The coupling-diagnostics group (``DIAG_CAPTURES``, ``DIAG_LINKS``, ``COUPLING_INTERVAL``, ``COUPLING_LAG``)
    appears in the result only for a field that departs from its default, so a run that leaves the group unset gets
    exactly the seven keys above.
    """
    gpu_aware = cfg_hpc_system is not None and bool(cfg_hpc_system.gpu_aware_mpi)
    return {
        "COURANT": cfg_analysis.TRITON_courant_number,
        "HEXTRA": cfg_analysis.TRITON_hextra_m,
        "FIXED_TSTEP": int(cfg_analysis.TRITON_fixed_timestep),
        "DOMAIN_DECOMP": cfg_analysis.TRITON_domain_decomposition,
        "DECOMP_CHECK_FACTOR": cfg_analysis.TRITON_domain_decomposition_check_factor,
        "IT_PRINT": cfg_analysis.TRITON_progress_print_iterations,
        "GPU_DIRECT": int(gpu_aware),
        **_coupling_diagnostic_settings(cfg_analysis),
    }


def _coupling_diagnostic_settings(cfg_analysis) -> dict:
    """The coupling-diagnostics placeholders that depart from their defaults, and no others.

    Absent keys, not defaulted ones: ``render_triton_cfg`` writes a line only for a key present with a non-default
    value, and a mapping without the keys is the mapping every older caller built, so a run that leaves the group
    unset hands the renderer exactly the dict it always did. The fields are read with ``getattr`` because several
    tests drive the cfg generators with stand-in configs that predate them; the real model always carries them.
    """
    settings: dict = {}
    if getattr(cfg_analysis, "toggle_coupling_diagnostics", False):
        settings["DIAG_CAPTURES"] = 1
    links = getattr(cfg_analysis, "swmm_diagnostic_links", None)
    if links is not None:
        settings["DIAG_LINKS"] = links
    if getattr(cfg_analysis, "swmm_diagnostic_node_path3_per_step", False):
        settings["DIAG_NODE_PATH3"] = 1
    interval = getattr(cfg_analysis, "swmm_coupling_interval", 1)
    if interval != 1:
        settings["COUPLING_INTERVAL"] = interval
    lag = getattr(cfg_analysis, "swmm_coupling_lag", 0)
    if lag != 0:
        settings["COUPLING_LAG"] = lag
    return settings


def _number_text(value: float) -> str:
    """``value`` as plain positional text: ``120.0`` reads ``120`` and ``1e6`` reads ``1000000``.

    A coordinate or an interval is written without an exponent because TRITON's own parsers read the text back,
    and ``repr`` of a large float is the exponent form.
    """
    return np.format_float_positional(float(value), trim="-")


# ---------------------------------------------------------------------------------------------------------------
# Observation points
# ---------------------------------------------------------------------------------------------------------------

#: The file TRITON reads, written into each scenario's folder beside its cfg.
OBSERVATION_FILE_NAME = "observation_points.txt"
_OBSERVATION_HEADER = ("id", "x", "y")


@dataclass(frozen=True)
class ObservationPoint:
    """One point of the user's observation file: a stable ``id`` and ``x``, ``y`` in the DEM's horizontal CRS."""

    id: str
    x: float
    y: float


def parse_observation_points(text: str) -> tuple[list[ObservationPoint], list[str]]:
    """The points of an ``id,x,y`` CSV and every problem found, each problem naming the point by its ``id``.

    A file is refused when its header is anything but ``id,x,y`` (TRITON's location file has no header, so a header
    row mistaken for a point would be read as one), when it lists no point (TRITON reads the first line
    unconditionally), when an ``id`` is empty or repeated, or when ``x`` or ``y`` is not a finite number.
    """
    rows = [row for row in csv.reader(io.StringIO(text)) if any(cell.strip() for cell in row)]
    if not rows:
        return [], ["the file is empty; it must start with the header 'id,x,y'"]
    header = tuple(cell.strip() for cell in rows[0])
    if header != _OBSERVATION_HEADER:
        return [], [f"the header is '{','.join(header)}'; it must be exactly 'id,x,y'"]
    points: list[ObservationPoint] = []
    problems: list[str] = []
    first_row_of: dict[str, int] = {}
    for n, row in enumerate(rows[1:], start=1):
        if len(row) != 3:
            problems.append(f"data row {n} has {len(row)} fields; each row is id,x,y")
            continue
        point_id = row[0].strip()
        if not point_id:
            problems.append(f"data row {n} has an empty id")
            continue
        if point_id in first_row_of:
            problems.append(f"id '{point_id}' appears in data rows {first_row_of[point_id]} and {n}")
            continue
        first_row_of[point_id] = n
        coordinates = []
        for axis, raw in zip("xy", row[1:], strict=True):
            try:
                number = float(raw)
            except ValueError:
                number = math.nan
            if not math.isfinite(number):
                problems.append(f"point '{point_id}': {axis} is '{raw.strip()}'; it must be a finite number")
            coordinates.append(number)
        if all(math.isfinite(c) for c in coordinates):
            points.append(ObservationPoint(point_id, coordinates[0], coordinates[1]))
    if not rows[1:]:
        problems.append("the file lists no points")
    return points, problems


def _read_observation_text(path: Path) -> str:
    """The text of the observation file, or one ``ConfigurationError`` naming the file when it is not UTF-8.

    A spreadsheet saves a CSV in the system's code page by default, so an accented site name is an ordinary way
    to reach this. The decode error would carry no file name and would abort the accumulating preflight.
    """
    try:
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as error:
        raise ConfigurationError(
            field="observation_points",
            message=f"{path.name} is not UTF-8 text: {error.reason} at byte {error.start}",
            fix_hint="Save the file as UTF-8 (in a spreadsheet, choose 'CSV UTF-8')",
        ) from None


def observation_point_problems(cfg_system) -> list[tuple[str, str, str]]:
    """``(field, message, fix_hint)`` per problem of the system's observation-point file; empty when none is set.

    The file's own format only: whether each point lies on the DEM is judged at scenario preparation, because the
    processed DEM does not exist until system setup has run.
    """
    path = getattr(cfg_system, "observation_points", None)
    if path is None:
        return []
    path = Path(path)
    if not path.is_file():
        return [("system.observation_points", f"Path does not exist: {path}", "Correct the path or unset the field")]
    try:
        text = _read_observation_text(path)
    except ConfigurationError as error:
        return [("system.observation_points", str(error), "Save the file as UTF-8 and check it again")]
    __, problems = parse_observation_points(text)
    return [
        ("system.observation_points", f"{problem} ({path.name})", "Fix the file; its header is 'id,x,y'")
        for problem in problems
    ]


def observation_cell(point: ObservationPoint, header: dict) -> tuple[int, int]:
    """The ``(row, col)`` of the DEM cell TRITON assigns ``point``, by TRITON's own formula.

    Mirrors ``calc_src_col`` and ``calc_src_row`` of the solver, including their ``1e-16`` nudge, so a point on a cell
    edge lands where the solver puts it. ``header`` is the processed DEM's grid header.
    """
    cellsize = header["cellsize"]
    col = math.ceil((point.x - header["xllcorner"]) / cellsize + 1e-16) - 1
    row = math.ceil(header["nrows"] - (point.y - header["yllcorner"]) / cellsize + 1e-16) - 1
    return row, col


@functools.lru_cache(maxsize=8)
def _dem_point_problems(
    dem_path: str,
    dem_identity: tuple[int, int],
    points: tuple[ObservationPoint, ...],
    outside_watershed_height: float | None,
    building_height: float | None,
) -> tuple[str, ...]:
    """Problems of ``points`` against the processed DEM: outside its extent, or on a cell TRITON cannot compute on.

    Cached on the DEM's ``(mtime, size)``: the answer depends only on the points and the DEM, and one scenario asks
    twice (the coupled cfg and the TRITON-only cfg). ``dem_identity`` is the cache key and is not read.

    A cell is refused when it carries the DEM's NODATA value, or when its elevation is at or above the smaller of the
    two fill heights that are set: the outside-watershed height and the building height. Those are the values the
    system burns into cells that are not terrain, the same rule ``dem_outside_watershed_height`` and
    ``dem_building_height`` state for the report's wall mask, less that mask's display buffer. Not established: a
    cell whose resampled elevation falls below both heights but is still mostly building is accepted.
    """
    # Lazy: ``hhemt.system`` imports the analysis, which imports the scenario, which imports this module.
    from hhemt.system import read_triton_grid_header

    header = read_triton_grid_header(dem_path)
    extent = (
        f"x {header['xllcorner']} to {header['xllcorner'] + header['ncols'] * header['cellsize']}, "
        f"y {header['yllcorner']} to {header['yllcorner'] + header['nrows'] * header['cellsize']}"
    )
    problems: list[str] = []
    cells: dict[str, tuple[int, int]] = {}
    for point in points:
        row, col = observation_cell(point, header)
        if 0 <= row < header["nrows"] and 0 <= col < header["ncols"]:
            cells[point.id] = (row, col)
        else:
            problems.append(
                f"point '{point.id}' (x={_number_text(point.x)}, y={_number_text(point.y)}) is outside the "
                f"processed DEM, whose extent is {extent}"
            )
    if not cells:
        return tuple(problems)
    wanted = {row for row, __ in cells.values()}
    fields_of_row: dict[int, list[str]] = {}
    with open(dem_path) as handle:
        lines = [handle.readline() for __ in range(6)]
        nodata = float(lines[5].split()[1])
        for index, line in enumerate(handle):
            if index in wanted:
                fields_of_row[index] = line.split()
            if index >= max(wanted):
                break
    fills = [h for h in (outside_watershed_height, building_height) if h is not None]
    fill_floor = min(fills) if fills else None
    by_id = {point.id: point for point in points}
    for point_id, (row, col) in cells.items():
        value = float(fields_of_row[row][col])
        if value == nodata:
            reason = f"the DEM's NODATA value {_number_text(nodata)}"
        elif fill_floor is not None and value >= fill_floor:
            reason = f"at or above the fill height {_number_text(fill_floor)} burned into cells that are not terrain"
        else:
            continue
        point = by_id[point_id]
        problems.append(
            f"point '{point_id}' (x={_number_text(point.x)}, y={_number_text(point.y)}) falls in DEM cell "
            f"(row {row}, col {col}) of elevation {_number_text(value)}, {reason}, so TRITON computes no water there"
        )
    return tuple(problems)


# ---------------------------------------------------------------------------------------------------------------
# Initial-condition rasters
# ---------------------------------------------------------------------------------------------------------------

_GRID_KEYS = ("ncols", "nrows", "xllcorner", "yllcorner", "cellsize")
_HEADER_KEYS = frozenset((*_GRID_KEYS, "xllcenter", "yllcenter", "nodata_value"))

# The closed set of length units a water-surface-elevation raster may be given in, each with its factor to meters.
# `ftUS` is the US survey foot, 1200/3937 m. The analysis config's `TRITON_initial_depth_raster_units` Literal names
# the same six keys, and a test holds the two equal.
LEVEL_UNIT_FACTORS = {"m": 1.0, "cm": 0.01, "mm": 0.001, "in": 0.0254, "ft": 0.3048, "ftUS": 1200 / 3937}

INITIAL_CONDITIONS_DIR_NAME = "triton_initial_conditions"
_FILL_LINE = "Write: TRITON_initial_depth_nodata_fill: 0"
_WSE = "water_surface_elevation"


@dataclass(frozen=True)
class AsciiGrid:
    """An ESRI ASCII grid as parsed text: its header keys (lower-case, values as written) and its value array."""

    header: dict
    values: np.ndarray


def parse_ascii_grid(raster_path, label: str) -> tuple[AsciiGrid | None, list[str]]:
    """Parse an ESRI ASCII grid the way TRITON's reader cannot: by tokens, so layout does not matter.

    Returns the grid and the problems found, each prefixed with ``label``. The grid is ``None`` when a problem
    stops the parse. The header is every leading line whose first word is a grid key. The value rows are split on
    any run of white space, so tabs, runs of spaces, a leading space and CRLF line ends read as the cells they
    hold, and a value such as ``1e-05`` reads as a number. A byte-order mark at the start is accepted, as the
    observation-file reader accepts one. Rows of unequal length, a word that is not a number, a file with no value
    rows and a file that is not UTF-8 text are each reported.

    The file is read line by line into an array that grows by doubling, so the peak is the array plus one line
    and not every word of the file as a Python string, which is about ten times the file size.

    Deliberately NOT ``np.loadtxt``: it names no row and no word in its message, and the label and the first bad
    row are what lets a user find the cell.
    """
    header: dict = {}
    values = None
    width = count = 0
    try:
        with open(raster_path, encoding="utf-8-sig") as handle:
            in_header = True
            for number, line in enumerate(handle, start=1):
                words = line.split()
                if not words:
                    continue
                if in_header and words[0].lower() in _HEADER_KEYS:
                    header[words[0].lower()] = " ".join(words[1:])
                    continue
                if in_header:
                    in_header = False
                    width = len(words)
                    capacity = 1024
                    declared = header.get("nrows", "")
                    if declared.isdigit() and 0 < int(declared) <= 10**7:
                        capacity = int(declared)
                    values = np.empty((capacity, width))
                if len(words) != width:
                    return None, [
                        f"{label}: line {number} holds {len(words)} values and the first value row holds {width}"
                    ]
                if count == len(values):
                    values = np.concatenate([values, np.empty_like(values)])
                try:
                    values[count] = np.array(words, dtype=float)
                except ValueError:
                    bad = next(word for word in words if not _is_number(word))
                    return None, [f"{label}: line {number} holds '{bad}', which is not a number"]
                count += 1
    except UnicodeDecodeError:
        return None, [f"{label}: the file is not UTF-8 text, so it is not an ESRI ASCII grid"]
    if values is None:
        return None, [f"{label}: the file has no value rows after its header"]
    return AsciiGrid(header, values[:count]), []


def _is_number(word: str) -> bool:
    """True when ``word`` parses as a float, so a failed row can name its first word that does not."""
    try:
        float(word)
    except ValueError:
        return False
    return True


def _grid_header_problems(label: str, grid: AsciiGrid, dem_header: dict) -> list[str]:
    """Problems of one grid's header against the processed DEM's, each prefixed with ``label``.

    TRITON reads the raster as a grid laid over the DEM, so a header that differs in any of the five grid keys
    places the initial water in the wrong cells or reads past the file. Corner-registered headers are required
    because the DEM's header names its corner; a grid that names its cell centres (``xllcenter``) is refused rather
    than silently shifted by half a cell. The value rows must also fill the grid the header describes.
    """
    problems = []
    seen = grid.header
    if "xllcenter" in seen or "yllcenter" in seen:
        problems.append(f"{label}: the header names cell centres; TRITON needs xllcorner/yllcorner")
    for key in _GRID_KEYS:
        if key not in seen:
            problems.append(f"{label}: the header has no '{key}'")
            continue
        want = dem_header[key]
        try:
            got = int(seen[key]) if key in ("ncols", "nrows") else float(seen[key])
        except ValueError:
            problems.append(f"{label}: header key '{key}' is '{seen[key]}', not a number")
            continue
        if not math.isclose(got, want, rel_tol=1e-9, abs_tol=1e-9):
            problems.append(f"{label}: header key '{key}' is {seen[key]}, the processed DEM's is {want}")
    if not problems and grid.values.shape != (dem_header["nrows"], dem_header["ncols"]):
        rows, cols = grid.values.shape
        problems.append(
            f"{label}: the file holds {rows} value rows of {cols} values; the header says "
            f"{dem_header['nrows']} rows of {dem_header['ncols']}"
        )
    return problems


def _nodata_mask(grid: AsciiGrid) -> np.ndarray:
    """The no-data cells of a grid: equal to the header's ``NODATA_value``, or not finite.

    Detected on the source values, before any conversion, so a no-data cell is never mistaken for a number that a
    unit factor or an offset has moved. A header with no ``NODATA_value`` marks no cell by value, and only the cells
    that are not finite are no-data.
    """
    mask = ~np.isfinite(grid.values)
    declared = grid.header.get("nodata_value")
    if declared is not None:
        try:
            mask |= grid.values == float(declared)
        except ValueError:
            pass
    return mask


def _first_cell(mask: np.ndarray) -> str:
    """The first cell of ``mask`` that is set, in reading order, as ``row r, col c`` counted from the north-west."""
    row, col = np.argwhere(mask)[0]
    return f"row {row}, col {col}"


def _depth_values(
    label: str, grid: AsciiGrid, nodata: np.ndarray, options: dict, dem_path: Path
) -> tuple[np.ndarray | None, list[str], str]:
    """The depth array TRITON reads for the depth raster, the problems found, and the log line for what was changed.

    Under ``depth`` a negative cell is refused. Under ``water_surface_elevation`` each cell is converted to meters,
    taken above the DEM, and clipped at 0, so a cell below the ground is dry. The no-data cells then take the fill
    as a depth and are exempt from both the conversion and the subtraction; with no fill set they are refused.
    """
    kind, fill = options["kind"], options["fill"]
    values = np.where(nodata, 0.0, grid.values)
    problems: list[str] = []
    notes = []
    if kind == "depth":
        negative = (values < 0) & ~nodata
        if negative.any():
            problems.append(
                f"{label}: {int(negative.sum())} cells are negative, the first at ({_first_cell(negative)}) with "
                f"value {_number_text(float(values[tuple(np.argwhere(negative)[0])]))}; the kind is 'depth', so a "
                f"negative depth is refused and not clipped. If the grid holds elevations, set "
                f"TRITON_initial_depth_raster_kind: {_WSE}"
            )
    elif kind == _WSE:
        units, offset = options["units"], options["offset"]
        if units not in LEVEL_UNIT_FACTORS or offset is None:
            problems.append(
                f"{label}: the kind '{_WSE}' needs TRITON_initial_depth_raster_units (one of "
                f"{', '.join(LEVEL_UNIT_FACTORS)}) and TRITON_initial_depth_raster_datum_offset (meters)"
            )
        else:
            dem, dem_problems = parse_ascii_grid(dem_path, f"the processed DEM {dem_path.name}")
            if dem is None or dem.values.shape != values.shape:
                problems += dem_problems or [
                    f"{label}: the processed DEM {dem_path.name} does not match the raster's grid"
                ]
            else:
                dem_missing = _nodata_mask(dem)
                depth = values * LEVEL_UNIT_FACTORS[units] + offset - dem.values
                dry = ((depth < 0) | dem_missing) & ~nodata
                depth = np.where(dry, 0.0, depth)
                notes.append(f"{int(dry.sum())} cells below the ground set dry")
                values = depth
    else:
        problems.append(f"{label}: TRITON_initial_depth_raster_kind is {kind!r}; write 'depth' or '{_WSE}'")
    if nodata.any():
        if fill is None:
            problems.append(
                f"{label}: {int(nodata.sum())} no-data cells, the first at ({_first_cell(nodata)}). {_FILL_LINE}"
            )
        else:
            values = np.where(nodata, float(fill), values)
            notes.append(f"{int(nodata.sum())} no-data cells set to {_number_text(float(fill))}")
    return (None if problems else values), problems, "; ".join(notes)


def _prepare_initial_raster(label: str, grid: AsciiGrid, *, is_depth: bool, options: dict, dem_path: Path):
    """The array to write for one initial-condition raster, the problems found, and the log line."""
    nodata = _nodata_mask(grid)
    if is_depth:
        return _depth_values(label, grid, nodata, options, dem_path)
    values = np.where(nodata, 0.0, grid.values)
    note = f"{int(nodata.sum())} no-data cells set to 0" if nodata.any() else ""
    return values, [], note


def _file_digest(path: Path):
    """A sha256 of the file's bytes, read in 1 MiB chunks so a large DEM is never held whole."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest


def _initial_raster_file(field: str, source: Path, options: dict, dem_path: Path, dem_header: dict) -> Path:
    """The path of the normalized copy of one raster: named by a hash of everything that shapes its values.

    The hash covers the source bytes, the depth settings, the processed DEM's grid and, under the elevation kind,
    the DEM's own bytes, because the subtraction reads them. A file already present was written only after every
    check passed on exactly these inputs, so a later scenario finds it and skips both the check and the write.
    """
    digest = _file_digest(source)
    digest.update(repr(sorted(options.items(), key=lambda item: item[0])).encode())
    digest.update(repr(sorted(dem_header.items())).encode())
    if options["kind"] == _WSE and field == "TRITON_initial_depth_raster":
        digest.update(_file_digest(dem_path).digest())
    return dem_path.parent / INITIAL_CONDITIONS_DIR_NAME / f"{field}_{digest.hexdigest()[:12]}.txt"


# ---------------------------------------------------------------------------------------------------------------
# The scenario's entry
# ---------------------------------------------------------------------------------------------------------------


def triton_input_settings(cfg_system, cfg_analysis, *, dem_path, observation_file) -> dict:
    """The cfg mapping values for observation points and initial-condition rasters, after checking both.

    Returns ``TIME_SERIES_FLAG`` and ``OBSERVATION_LOC_FILE`` (the template's two observation placeholders),
    ``PRINT_OBSERVATION`` (None unless an interval is set) and ``H_INFILE``, ``QX_INFILE``, ``QY_INFILE`` (None
    unless set). With nothing set it touches no file, so a scenario that uses neither gets the defaults and the
    cfg is unchanged on every line TRITON reads. With observation points set it writes the ``x,y`` location file
    to ``observation_file`` in the file's own order, which is the point identity the processing step relies on.

    An initial-condition raster is never handed to TRITON as the user wrote it. TRITON reads these files without a
    header, with a reader that splits on single spaces and reads a token without a decimal point as an integer, so
    the file is parsed here and a normalized copy is written under ``triton_initial_conditions/`` beside the
    processed DEM; the cfg names the copy. No-data cells take the depth fill, or 0 for a discharge, and a
    water-surface-elevation depth raster is turned into depths.

    Raises ``ConfigurationError`` naming every offending point or raster when a point lies outside the processed
    DEM or on a cell TRITON cannot compute on, or a raster's grid differs from the DEM's, or a raster holds a
    value the settings do not allow.

    The config attributes are read with ``getattr`` because several tests drive the cfg generators with stand-in
    configs that predate these fields; the real models always carry them, and a test pins the names against the
    models' fields so a misspelling cannot read as "unset".
    """
    settings = {
        "TIME_SERIES_FLAG": 0,
        "OBSERVATION_LOC_FILE": "",
        "PRINT_OBSERVATION": None,
        **{placeholder: None for placeholder, __, __ in _INITIAL_CONDITIONS},
    }
    points_path = getattr(cfg_system, "observation_points", None)
    # Literal names, so a reader of the config models finds each setting read here.
    rasters = {
        "H_INFILE": getattr(cfg_analysis, "TRITON_initial_depth_raster", None),
        "QX_INFILE": getattr(cfg_analysis, "TRITON_initial_qx_raster", None),
        "QY_INFILE": getattr(cfg_analysis, "TRITON_initial_qy_raster", None),
    }
    depth_options = {
        "kind": getattr(cfg_analysis, "TRITON_initial_depth_raster_kind", None),
        "units": getattr(cfg_analysis, "TRITON_initial_depth_raster_units", None),
        "offset": getattr(cfg_analysis, "TRITON_initial_depth_raster_datum_offset", None),
        "fill": getattr(cfg_analysis, "TRITON_initial_depth_nodata_fill", None),
    }
    if points_path is None and not any(rasters.values()):
        return settings
    dem_path = Path(dem_path)
    if not dem_path.is_file():
        raise ConfigurationError(
            field="observation_points" if points_path is not None else "TRITON_initial_*_raster",
            message=f"the processed DEM {dem_path} does not exist, so the inputs cannot be checked against it",
            fix_hint="Run system setup before preparing scenarios",
        )
    problems: list[str] = []
    fields: list[str] = []
    if points_path is not None:
        points, parse_problems = parse_observation_points(_read_observation_text(Path(points_path)))
        problems += parse_problems
        if not parse_problems:
            stat = dem_path.stat()
            problems += _dem_point_problems(
                str(dem_path),
                (stat.st_mtime_ns, stat.st_size),
                tuple(points),
                getattr(cfg_system, "dem_outside_watershed_height", None),
                getattr(cfg_system, "dem_building_height", None),
            )
        if problems:
            fields.append("observation_points")
        else:
            observation_file = Path(observation_file)
            observation_file.parent.mkdir(parents=True, exist_ok=True)
            observation_file.write_text("".join(f"{_number_text(p.x)},{_number_text(p.y)}\n" for p in points))
            settings["TIME_SERIES_FLAG"] = 1
            settings["OBSERVATION_LOC_FILE"] = str(observation_file)
            interval = getattr(cfg_analysis, "TRITON_observation_interval_s", None)
            settings["PRINT_OBSERVATION"] = interval
    if any(rasters.values()):
        # Lazy: see _dem_point_problems.
        from hhemt.system import read_triton_grid_header, write_triton_ascii_grid

        dem_header = read_triton_grid_header(dem_path)
        for placeholder, __, field in _INITIAL_CONDITIONS:
            source = rasters[placeholder]
            if source is None:
                continue
            source = Path(source)
            is_depth = placeholder == "H_INFILE"
            options = depth_options if is_depth else {"kind": None, "units": None, "offset": None, "fill": None}
            label = f"{field} ({source.name})"
            if is_depth and options["kind"] is None:
                fields.append(field)
                problems.append(f"{label}: TRITON_initial_depth_raster_kind is not set; write 'depth' or '{_WSE}'")
                continue
            target = _initial_raster_file(field, source, options, dem_path, dem_header)
            if target.is_file():
                settings[placeholder] = str(target)
                continue
            grid, raster_problems = parse_ascii_grid(source, label)
            if grid is not None:
                raster_problems = _grid_header_problems(label, grid, dem_header)
            values, note = None, ""
            if grid is not None and not raster_problems:
                values, raster_problems, note = _prepare_initial_raster(
                    label, grid, is_depth=is_depth, options=options, dem_path=dem_path
                )
            if raster_problems:
                fields.append(field)
                problems += raster_problems
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            write_triton_ascii_grid(target, values)
            print(f"[TRITON inputs] {field}: wrote {target.name}" + (f"; {note}" if note else ""), flush=True)
            settings[placeholder] = str(target)
    if problems:
        raise ConfigurationError(
            field=", ".join(fields),
            message="\n  ".join(problems),
            fix_hint="Correct the input; the processed DEM's grid and the points' CRS must match",
        )
    return settings
