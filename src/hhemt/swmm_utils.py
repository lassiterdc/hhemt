"""
SWMM Utility Functions

This module contains shared utility functions for filling SWMM role models.
These functions are used across multiple SWMM model creation modules to avoid code duplication.
"""

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from hhemt.swmm_runnable import WATER_LEVEL_SERIES_PREFIX, derive_role_model, rewrite_forcing, set_threads

if TYPE_CHECKING:
    from .scenario import TRITONSWMM_scenario

# SWMM copies file names into buffers of this many characters (consts.h:
# `#define MAXFNAME 259 // Max. # characters in file name`). For the two kinds this
# module measures, a longer name is truncated: a time-series FILE name is opened by its
# truncated copy, and the binary output is written under its truncated copy. The input
# and report names are opened from the caller's full strings and are not affected.
_SWMM_MAXFNAME = 259


def _swmm_fill_values(
    scenario: "TRITONSWMM_scenario",
    destination: Path,
) -> dict:
    """
    The forcing values one scenario writes into a derived SWMM role model.

    Shared by the hydrology, hydraulics and full role writes. Returns the mapping
    ``swmm_runnable.rewrite_forcing`` takes: time series data, rain gauges, simulation
    timing and the reporting interval. Before returning, it refuses any file name SWMM
    would truncate (see Raises).

    Parameters
    ----------
    scenario : TRITONSWMM_scenario
        The scenario object containing configuration and paths
    destination : Path
        Path of the role model the values are written into (selects which output name is measured)

    Raises
    ------
    ValueError
        If rainfall units are not 'mm/hr' or 'mm'
    ProcessingError
        Before anything is written, if any time-series ``FILE`` path the input names
        (one per rain gauge, one per ``water_level`` record of the external boundary, and the extracted file of
        every series the base model defines inline) is over 259 bytes, SWMM's file-name limit. SWMM would open a
        truncated name and fail with an error that names no length.
    ProcessingError
        Before anything is written, if the binary output name SWMM will be handed for
        this input is over 259 bytes, since SWMM would write its results under a
        shortened name. That name is: for ``swmm_hydro_inp``, the destination with
        ``.inp`` replaced by ``.out``; for ``swmm_full_inp``, ``swmm_full_out_file``
        (not checked when it is ``None``); for ``swmm_hydraulics_inp``, when
        ``swmm_hydraulics_rpt`` is set (the coupled model is on) and the run is native,
        ``swmm_hydraulics_rpt`` with a ``.out`` suffix. It is not checked in container
        mode, where TRITON hands SWMM a name inside the image, nor for any other
        destination. Input and report names are never checked.
    """
    cfg_analysis = scenario._analysis.cfg_analysis
    weather_time_series_timestep_dimension_name = cfg_analysis.weather_time_series_timestep_dimension_name

    ds_event_ts = scenario.ds_event_ts

    # Calculate timestep interval
    tstep_seconds = (
        ds_event_ts[weather_time_series_timestep_dimension_name].to_series().diff().mode().iloc[0].total_seconds()
    )
    interval = scenario.seconds_to_hhmmss(tstep_seconds)

    # Determine rainfall format
    if cfg_analysis.rainfall_units == "mm/hr":
        format = "INTENSITY"
    elif cfg_analysis.rainfall_units == "mm":
        format = "DEPTH"
    else:
        raise ValueError(
            f"Invalid rainfall units specified. Expecting mm/hr or mm but got {cfg_analysis.rainfall_units}"
        )

    # Create time series and rain gauge sections
    fs = scenario.log.swmm_rainfall_dat_files.get()

    lst_tseries_section = []
    rain_gages_section = []
    for key, val in fs.items():
        lst_tseries_section.append(f'{key} FILE "{val}"')
        row = f"{key} {format} {interval} 1.0 TIMESERIES {key}"
        rain_gages_section.append(row)

    # One series per water_level record of the external boundary, named for its record. A role keeps a row only if
    # its [OUTFALLS] read that series (rewrite_forcing decides).
    file_paths = list(fs.items())
    if cfg_analysis.toggle_external_boundaries:
        for k, level_path in sorted((scenario.log.water_level_boundary_dats.get() or {}).items()):
            name = f"{WATER_LEVEL_SERIES_PREFIX}{k}"
            lst_tseries_section.append(f'{name} FILE "{level_path}"')
            file_paths.append((name, level_path))
    # The pointers of the base model's own series sit in every role, so SWMM opens them by the same name limit.
    file_paths.extend(
        (f"{name} (extracted)", pointer) for name, pointer in _series_pointers(scenario._system.cfg_system).items()
    )

    # Refuse before writing anything: SWMM truncates a FILE name longer than
    # _SWMM_MAXFNAME and then fails with "ERROR 361: could not open external file",
    # an error that names no length. Length is the byte count SWMM's strlen sees.
    from hhemt.exceptions import ProcessingError

    for key, path in file_paths:
        n_bytes = len(os.fsencode(str(path)))
        if n_bytes > _SWMM_MAXFNAME:
            raise ProcessingError(
                operation="_swmm_fill_values",
                filepath=Path(path),
                reason=(
                    f"SWMM input {destination}: the FILE path for time series {key!r} is {n_bytes} bytes "
                    f"long, over SWMM's {_SWMM_MAXFNAME}-character file-name limit: {path}. SWMM cannot "
                    f"open a file name longer than {_SWMM_MAXFNAME} characters. Move the analysis "
                    f"directory to a shorter path."
                ),
            )

    # The binary output name SWMM is handed for this input is held in a buffer of the
    # same size, and an over-long one is written under a shortened name with no error.
    # The input and report names are opened from the caller's full strings, so they
    # are not measured here.
    scen_paths = scenario.scen_paths
    out_name = None
    if destination == scen_paths.swmm_hydro_inp:
        # pyswmm derives the output name this way when it is given none.
        out_name = str(destination).replace(".inp", ".out")
    elif destination == scen_paths.swmm_full_inp:
        if scen_paths.swmm_full_out_file is not None:
            out_name = str(scen_paths.swmm_full_out_file)
    elif destination == scen_paths.swmm_hydraulics_inp and scen_paths.swmm_hydraulics_rpt is not None:
        from hhemt.config.hpc_system import resolve_container_spec

        # In a container TRITON hands SWMM an output name under the image's install
        # root, not under the host analysis directory, so no host path is measured.
        in_container = (
            cfg_analysis.execution_environment == "container"
            and resolve_container_spec(scenario._analysis.cfg_hpc_system) is not None
        )
        if not in_container:
            out_name = str(scen_paths.swmm_hydraulics_rpt.with_suffix(".out"))
    if out_name is not None:
        n_bytes = len(os.fsencode(out_name))
        if n_bytes > _SWMM_MAXFNAME:
            raise ProcessingError(
                operation="_swmm_fill_values",
                filepath=Path(out_name),
                reason=(
                    f"SWMM input {destination}: the binary output name SWMM will be handed is "
                    f"{n_bytes} bytes long, over SWMM's {_SWMM_MAXFNAME}-character file-name limit: "
                    f"{out_name}. SWMM would write its results under a shortened name. Move the "
                    f"analysis directory to a shorter path."
                ),
            )

    mapping = dict()
    mapping["TIMESERIES"] = "\n\n".join(lst_tseries_section)
    mapping["RAINGAGES"] = "\n".join(rain_gages_section)

    # The window is the USER'S, read back rather than derived from the clipped coordinate: a derivation
    # would agree with the declared window only by coincidence of the clip, and would collapse Spec 8's
    # three-source guard (the window must stay independent of the forcing row count) into a tautology.
    first_tstep, last_tstep = scenario._resolve_event_window()

    # Add timing parameters
    mapping["START_DATE"] = first_tstep.strftime("%m/%d/%Y")
    mapping["START_TIME"] = first_tstep.strftime("%H:%M:%S")
    mapping["REPORT_START_DATE"] = mapping["START_DATE"]
    mapping["REPORT_START_TIME"] = mapping["START_TIME"]
    mapping["END_DATE"] = last_tstep.strftime("%m/%d/%Y")
    mapping["END_TIME"] = last_tstep.strftime("%H:%M:%S")
    # The coupled (hydraulics) model alone takes the independent report step; every other role keeps the TRITON
    # reporting step, which is also the value of the coupled model while ``swmm_report_step_s`` is None.
    report_step_s = cfg_analysis.TRITON_reporting_timestep_s
    independent_step_s = getattr(cfg_analysis, "swmm_report_step_s", None)
    if independent_step_s is not None and destination == scen_paths.swmm_hydraulics_inp:
        report_step_s = independent_step_s
    mapping["REPORT_STEP"] = scenario.seconds_to_hhmmss(report_step_s)

    return mapping


def _read_model_text(path: Path) -> str:
    """The text of a SWMM model, line endings untouched. UTF-8, else Latin-1 (SWMM reads bytes)."""
    for encoding in ("utf-8", "latin-1"):
        try:
            with open(path, encoding=encoding, newline="") as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    raise AssertionError("latin-1 decodes every byte string")  # pragma: no cover


def processed_dem_path(cfg_system) -> Path:
    """The processed (coarsened) DEM TRITON loads, in the system directory."""
    return Path(cfg_system.system_directory) / f"elevation_{cfg_system.target_dem_resolution:.2f}m.dem"


def _series_pointers(cfg_system) -> dict[str, str]:
    """Every series of the base model by case-folded name, mapped to the file its pointer names.

    An inline series is written once to ``{system_directory}/swmm_model_series/{model_sha12}/{index}.dat``, and a
    ``FILE`` series keeps its file, so no derived model holds series data.
    """
    path = Path(cfg_system.SWMM_model)
    return dict(_cached_series_pointers(str(path), path.stat().st_mtime_ns, str(cfg_system.system_directory)))


@lru_cache(maxsize=16)
def _cached_series_pointers(path: str, mtime_ns: int, system_directory: str) -> tuple[tuple[str, str], ...]:
    from hhemt.water_level_boundary import extract_model_series

    return tuple(extract_model_series(Path(path), Path(system_directory)).items())


def open_water_outfalls(cfg_analysis) -> tuple[str, ...]:
    """The outfalls ``swmm_open_water_outfalls`` lists, a repeated name once, in the order listed.

    Empty when the analysis has no such field or lists none. Read by attribute and by type: the configuration
    stand-ins of the tests carry only the fields their cases set.
    """
    names = getattr(cfg_analysis, "swmm_open_water_outfalls", None) if cfg_analysis is not None else None
    return tuple(dict.fromkeys(names)) if isinstance(names, list | tuple) else ()


def open_water_representation(cfg_analysis) -> str:
    """``driven_stage`` or ``crown_rim``: how each listed open-water outfall is represented."""
    value = getattr(cfg_analysis, "open_water_outfall_representation", "driven_stage")
    return value if value in ("driven_stage", "crown_rim") else "driven_stage"


def driven_stage_outfalls(cfg_analysis) -> tuple[str, ...]:
    """The listed open-water outfalls that stay outfalls and take a stage TRITON sets (empty under ``crown_rim``)."""
    return open_water_outfalls(cfg_analysis) if open_water_representation(cfg_analysis) == "driven_stage" else ()


def crown_rim_outfalls(cfg_analysis) -> tuple[str, ...]:
    """The listed open-water outfalls converted to coupled junctions at the outlet conduit's crown (empty under
    ``driven_stage``)."""
    return open_water_outfalls(cfg_analysis) if open_water_representation(cfg_analysis) == "crown_rim" else ()


def converted_outfalls(cfg_system, cfg_analysis=None) -> tuple[str, ...]:
    """The outfalls that become coupled junctions.

    ``swmm_outfalls_to_couple``, plus those a water_level record names, except that a record does not convert an
    outfall listed in ``swmm_open_water_outfalls`` while the representation is ``driven_stage`` (the outfall stays
    an outfall whose stage TRITON sets), plus every listed outfall while the representation is ``crown_rim``.
    With no listed outfall the result is what it was before the list existed.
    """
    from hhemt.swmm_runnable import _fold
    from hhemt.water_level_boundary import boundary_records_for, matched_outfall_names

    matched: tuple[str, ...] = ()
    if cfg_analysis is not None:
        records, _problems = boundary_records_for(cfg_analysis, cfg_system)
        matched = matched_outfall_names(records)
    listed = open_water_outfalls(cfg_analysis)
    if listed:
        if open_water_representation(cfg_analysis) == "driven_stage":
            driven = {_fold(n) for n in listed}
            matched = tuple(n for n in matched if _fold(n) not in driven)
        else:
            matched = (*matched, *listed)
    return tuple(dict.fromkeys([*cfg_system.swmm_outfalls_to_couple, *matched]))


#: One foot in meters: a US-unit SWMM model (flow units CFS, GPM or MGD) gives its cross-sections in feet.
_FEET_TO_METERS = 0.3048


@dataclass(frozen=True)
class CrownRimOutfall:
    """The comparison representation of one listed open-water outfall, sized from its outlet conduit.

    ``conduit`` is the ``[CONDUITS]`` row whose outlet node is the outfall, ``barrels`` its barrel count, ``shape``
    its cross-section keyword. ``area_m2`` is the full flow area of every barrel together (``scenario.calc_area``,
    which multiplies by the barrels), ``diameter_m`` the diameter of the circle of that area, ``crown_m`` the
    height of the conduit's crown above the outfall's invert (outlet offset plus full height), which is the depth
    SWMM raises the node to, or None when the offset is not a number. ``loss_coefficient`` is
    ``manhole_loss_coefficient``.
    """

    name: str
    conduit: str
    barrels: float
    shape: str
    area_m2: float
    diameter_m: float
    crown_m: float | None
    loss_coefficient: float


def _outlet_conduit_sizing(sections, name: str, length_scale: float, loss: float) -> "CrownRimOutfall | str":
    """The sizing of the one conduit whose outlet node is ``name``, or the reason there is none to read."""
    import math
    from types import SimpleNamespace

    from hhemt.scenario import calc_area
    from hhemt.swmm_runnable import _fold, _node_rows

    key = _fold(name)
    into = [t for t in _node_rows(sections, "CONDUITS") if len(t) > 2 and _fold(t[2]) == key]
    if not into:
        return f"{name} has no conduit whose outlet node is {name}, so there is no pipe to size it from"
    if len(into) > 1:
        return (
            f"{name} has {len(into)} conduits whose outlet node is {name} ({', '.join(t[0] for t in into)}); "
            "the rule reads one outlet conduit"
        )
    conduit = into[0]
    rows = {_fold(t[0]): t for t in _node_rows(sections, "XSECTIONS") if len(t) > 2}
    xs = rows.get(_fold(conduit[0]))
    if xs is None:
        return f"conduit {conduit[0]} (the outlet conduit of {name}) has no [XSECTIONS] row, so it has no size"

    def number(tokens, i):
        """The ``i``-th token as a float, or None when the row is shorter or the token is not a number."""
        try:
            return float(tokens[i])
        except (IndexError, ValueError):
            return None

    area = calc_area(
        SimpleNamespace(
            Shape=xs[1],
            Geom1=number(xs, 2),
            Geom2=number(xs, 3),
            Geom3=number(xs, 4),
            Geom4=number(xs, 5),
            Barrels=number(xs, 6),
        )
    )
    if area is None or not math.isfinite(float(area)) or float(area) <= 0:
        return (
            f"conduit {conduit[0]} (the outlet conduit of {name}) has the cross-section {xs[1]} with "
            f"dimensions {xs[2:6]}, whose flow area the toolkit cannot size"
        )
    height = number(xs, 2)
    offset = number(conduit, 6)
    # The crown above the outfall's invert, which is the depth SWMM raises the node to (``fullDepth = offset + yFull``).
    # Under LINK_OFFSETS ELEVATION the offset is an elevation, and SWMM first converts it to a height above the node's
    # invert (``link_convertOffsets``, ``offset -= node invert``, link.c), so the invert is subtracted here too.
    # Informational: written to the crown_rim_outfalls.json record and used in no cfg line, manhole row or refusal.
    elevation_offsets = any(
        len(t) > 1 and _fold(t[0]) == "LINK_OFFSETS" and _fold(t[1]).startswith("ELEV")
        for t in _node_rows(sections, "OPTIONS")
    )
    invert = next((number(t, 1) for t in _node_rows(sections, "OUTFALLS") if t and _fold(t[0]) == key), None)
    crown = None
    if height is not None and len(conduit) > 6 and conduit[6] == "*":
        crown = height * length_scale  # a '*' offset is 0 above the node's invert in either mode
    elif height is not None and offset is not None:
        if not elevation_offsets:
            crown = (offset + height) * length_scale
        elif invert is not None:
            crown = ((offset - invert) + height) * length_scale
    area_m2 = float(area) * length_scale**2
    return CrownRimOutfall(
        name=name,
        conduit=conduit[0],
        barrels=number(xs, 6) or 1.0,
        shape=str(xs[1]).strip().upper(),
        area_m2=area_m2,
        diameter_m=math.sqrt(4.0 * area_m2 / math.pi),
        crown_m=crown,
        loss_coefficient=float(loss),
    )


def crown_rim_sizing(cfg_system, cfg_analysis, skip=()) -> "tuple[dict[str, CrownRimOutfall], list[str]]":
    """Size each ``crown_rim`` open-water outfall from its outlet conduit: ``({name: sizing}, [reason, ...])``.

    The diameter is the area-equivalent one, ``d = sqrt(4 A / pi)`` with ``A`` the total flow area of every barrel
    of the outlet conduit, and the loss is ``manhole_loss_coefficient``. A US-unit model (flow units CFS, GPM or
    MGD) holds its cross-sections in feet and the diameter is returned in meters. The reasons name every listed
    outfall that cannot be sized. Empty for a ``driven_stage`` analysis. ``skip`` names outfalls left out of the
    sizing (the ones the caller has already refused as no ``[OUTFALLS]`` row).
    """
    from hhemt.swmm_runnable import _fold, _parse
    from hhemt.swmm_runoff_modeling import _US_FLOW_UNITS, flow_units_from_inp_text

    skipped = {_fold(n) for n in skip}
    listed = [n for n in crown_rim_outfalls(cfg_analysis) if _fold(n) not in skipped]
    if not listed:
        return {}, []
    text = _read_model_text(Path(cfg_system.SWMM_model))
    sections = _parse(text)
    scale = _FEET_TO_METERS if flow_units_from_inp_text(text) in _US_FLOW_UNITS else 1.0
    sized: dict[str, CrownRimOutfall] = {}
    reasons: list[str] = []
    for name in listed:
        result = _outlet_conduit_sizing(sections, name, scale, cfg_analysis.manhole_loss_coefficient)
        if isinstance(result, str):
            reasons.append(result)
        else:
            sized[name] = result
    return sized, reasons


def open_water_outfall_problems(cfg_system, cfg_analysis) -> list:
    """Every rule the open-water outfall list breaks, as ``Problem`` rows naming each offender.

    The one function the run-entry validation (``validation._validate_open_water_outfalls``) and scenario
    preparation (``require_valid_open_water_outfalls``) both ask, so the two cannot apply different rules:

    * a listed name is an ``[OUTFALLS]`` row of ``SWMM_model`` and appears once (after SWMM's case fold);
    * a listed outfall is not also in ``swmm_outfalls_to_couple`` or ``swmm_outfalls_to_keep`` (one disposition per
      outfall);
    * a listed outfall has no row in ``manhole_table`` (a driven outfall has no exchange slot, and a ``crown_rim``
      node's row is the toolkit's);
    * under ``crown_rim``, a listed outfall has exactly one conduit whose outlet node it is, and that conduit's
      cross-section has a flow area the toolkit can size.

    A listed outfall named by a water-level record is allowed: the record drives TRITON's boundary line. An
    unreadable model returns ``[]`` (its existence is judged elsewhere).
    """
    from hhemt.swmm_runnable import _fold, _parse
    from hhemt.water_level_boundary import Problem, model_outfalls

    raw = getattr(cfg_analysis, "swmm_open_water_outfalls", None)
    if not isinstance(raw, list | tuple) or not raw:
        return []
    field = "analysis.swmm_open_water_outfalls"
    problems: list[Problem] = []
    folded = [_fold(n) for n in raw]
    repeated = sorted({n for n, f in zip(raw, folded, strict=True) if folded.count(f) > 1})
    if repeated:
        problems.append(
            Problem(
                "CFG",
                field,
                f"outfall(s) {repeated} are listed more than once in swmm_open_water_outfalls (SWMM folds case)",
                list(raw),
                fix_hint="List each outfall once",
            )
        )
    for other_field, group, where in (
        ("system.swmm_outfalls_to_couple", getattr(cfg_system, "swmm_outfalls_to_couple", None), "system config"),
        ("analysis.swmm_outfalls_to_keep", getattr(cfg_analysis, "swmm_outfalls_to_keep", None), "analysis config"),
    ):
        names = {_fold(n) for n in group} if isinstance(group, list | tuple) else set()
        both = sorted(n for n in dict.fromkeys(raw) if _fold(n) in names)
        if both:
            leaf = other_field.split(".", 1)[1]
            problems.append(
                Problem(
                    "CFG",
                    field,
                    f"outfall(s) {both} are listed in swmm_open_water_outfalls and also in {leaf} ({where}); "
                    "an outfall has one disposition",
                    list(raw),
                    fix_hint=f"Remove each name from swmm_open_water_outfalls or from {leaf}",
                )
            )
    table = getattr(cfg_analysis, "manhole_table", None)
    if table is not None and Path(table).is_file():
        from hhemt.exceptions import ConfigurationError
        from hhemt.scenario import read_manhole_table

        try:
            table_rows = read_manhole_table(table)
        except ConfigurationError:
            table_rows = []  # the table's own check names its defect
        in_table = sorted(r.node_id for r in table_rows if _fold(r.node_id) in set(folded))
        if in_table:
            representation = open_water_representation(cfg_analysis)
            why = (
                "a driven outfall is not a coupling node and has no exchange slot"
                if representation == "driven_stage"
                else "the toolkit writes a crown_rim node's diameter and loss itself, from its outlet conduit"
            )
            problems.append(
                Problem(
                    "CFG",
                    "analysis.manhole_table",
                    f"manhole_table {table} has row(s) for open-water outfall(s) {in_table}; {why}",
                    str(table),
                    fix_hint="Remove those rows from manhole_table",
                )
            )
    try:
        text = _read_model_text(Path(cfg_system.SWMM_model))
        sections = _parse(text)
    except Exception:  # unreadable or malformed model: swmm_model_problems names it
        return problems
    known = model_outfalls(sections)
    unknown = [n for n in dict.fromkeys(raw) if _fold(n) not in known]
    if unknown:
        problems.append(
            Problem(
                "CFG",
                field,
                f"{unknown} {'is' if len(unknown) == 1 else 'are'} not an [OUTFALLS] row of {cfg_system.SWMM_model}",
                list(raw),
                fix_hint="List only names that appear in the [OUTFALLS] section of SWMM_model",
            )
        )
    if open_water_representation(cfg_analysis) == "crown_rim":
        _sized, reasons = crown_rim_sizing(cfg_system, cfg_analysis, skip=unknown)
        if reasons:
            problems.append(
                Problem(
                    "CFG",
                    "analysis.open_water_outfall_representation",
                    "crown_rim sizes each listed outfall from the one conduit that ends at it, and "
                    + "; ".join(reasons),
                    "crown_rim",
                    fix_hint="Use driven_stage, which needs no conduit, or list only outfalls with one sizable "
                    "outlet conduit",
                )
            )
    return problems


def require_valid_open_water_outfalls(cfg_system, cfg_analysis) -> None:
    """Refuse, naming the field and every offender, an open-water outfall list that breaks a rule of
    ``open_water_outfall_problems``. The scenario-preparation form of the run-entry check."""
    from hhemt.exceptions import ConfigurationError

    problems = open_water_outfall_problems(cfg_system, cfg_analysis)
    if problems:
        raise ConfigurationError(
            field="swmm_open_water_outfalls",
            message="swmm_open_water_outfalls is refused:\n    - " + "\n    - ".join(p.message for p in problems),
            fix_hint="; ".join(p.fix_hint for p in problems if p.fix_hint),
        )


@lru_cache(maxsize=2)
def _loaded_raster(path: str, mtime_ns: int):
    """The raster at ``path`` read once into memory, cached on its path and mtime like the derived role text."""
    import rioxarray as rxr

    return rxr.open_rasterio(path).load()


@lru_cache(maxsize=2)
def _coarsened_fallback(full_path: str, mtime_ns: int, resolution: float, xll, yll, ncols, nrows):
    """The grid setup will write, built in memory from the full-resolution DEM, cached on the inputs that shape it."""
    from hhemt.system import coarsen_georaster

    return coarsen_georaster(
        _loaded_raster(full_path, mtime_ns), resolution, xllcorner=xll, yllcorner=yll, ncols=ncols, nrows=nrows
    )


def _mtime(path) -> int:
    return Path(path).stat().st_mtime_ns


def processed_dem_grid(cfg_system):
    """The processed DEM TRITON loads, as ``(raster, provisional)``.

    When the file does not exist yet (preflight, before setup), the grid setup will write is built in memory the
    way setup builds it, and the result is marked provisional. Each read is cached on the file's path and mtime, so
    the raster is not read again on every call.
    """
    processed = processed_dem_path(cfg_system)
    if processed.is_file():
        return _loaded_raster(str(processed), _mtime(processed)), False
    full = Path(cfg_system.DEM_fullres)
    return (
        _coarsened_fallback(
            str(full),
            _mtime(full),
            cfg_system.target_dem_resolution,
            cfg_system.processed_xllcorner,
            cfg_system.processed_yllcorner,
            cfg_system.ncols,
            cfg_system.nrows,
        ),
        True,
    )


def rim_dem(cfg_system, cfg_analysis):
    """The DEM that gives node depths, as ``(raster, provisional)``, or ``(None, False)`` when none is chosen.

    ``dem_rim_resolution`` picks the full-resolution DEM or the processed one.
    """
    resolution = getattr(cfg_analysis, "dem_rim_resolution", None) if cfg_analysis is not None else None
    if resolution is None:
        return None, False
    if resolution == "full":
        full = Path(cfg_system.DEM_fullres)
        return _loaded_raster(str(full), _mtime(full)), False
    return processed_dem_grid(cfg_system)


def _junction_nodes(sections, names) -> dict:
    """``{name: (x, y, invert)}`` for each of ``names`` that is an outfall or junction row with a coordinate."""
    from hhemt.swmm_runnable import _fold, _node_rows
    from hhemt.water_level_boundary import model_coordinates

    coordinates = model_coordinates(sections)
    rows = {_fold(t[0]): t for sec in ("JUNCTIONS", "OUTFALLS") for t in _node_rows(sections, sec) if len(t) > 1}
    nodes = {}
    for name in names:
        key = _fold(name)
        if key in rows and key in coordinates:
            try:
                nodes[rows[key][0]] = (*coordinates[key], float(rows[key][1]))
            except ValueError:
                continue
    return nodes


def _node_depths(path: Path, converted: tuple[str, ...], cfg_system, cfg_analysis):
    """The ``MaxDepth`` of each node the DEM sets, the V23 problems, and whether the DEM is provisional.

    The converted outfalls always take the depth of the chosen DEM. With ``swmm_node_rim_from_dem`` on and the full
    DEM chosen, every coupling node does too, and the TRITON key is not written (TRITON would overwrite the rims from
    the processed grid).
    """
    from hhemt.swmm_runnable import _fold, _parse
    from hhemt.water_level_boundary import converted_node_depths

    resolution = getattr(cfg_analysis, "dem_rim_resolution", None)
    all_nodes = bool(getattr(cfg_analysis, "swmm_node_rim_from_dem", False)) and resolution == "full"
    # A crown_rim open-water outfall takes no depth from the DEM: its SWMM depth is written as 0 and SWMM raises it to
    # the outlet conduit's crown, so the DEM check (V23) is not asked of it, and of no other node.
    crown = {_fold(n) for n in crown_rim_outfalls(cfg_analysis)}
    depth_names = [n for n in converted if _fold(n) not in crown]
    if (not depth_names and not all_nodes) or resolution is None:
        return {}, [], False
    sections = _parse(_read_model_text(path))
    names = list(depth_names)
    if all_nodes:
        from hhemt.scenario import InpText, coupling_candidates

        base = InpText(derive_role_model(_read_model_text(path), "hydraulics", converted))
        names.extend(
            n
            for n in coupling_candidates(cfg_system, inp_path=base, cfg_analysis=cfg_analysis)
            if _fold(n) not in crown
        )
    nodes = _junction_nodes(sections, dict.fromkeys(names))
    if not nodes:
        return {}, [], False
    rds, provisional = rim_dem(cfg_system, cfg_analysis)
    depths, problems = converted_node_depths(rds, nodes)
    return depths, problems, provisional


def node_depth_problems(cfg_system, cfg_analysis):
    """The V23 problems of the converted outfalls' DEM depths, and whether they are provisional (preflight form)."""
    path = Path(cfg_system.SWMM_model)
    _depths, problems, provisional = _node_depths(
        path, converted_outfalls(cfg_system, cfg_analysis), cfg_system, cfg_analysis
    )
    return problems, provisional


#: How far below SWMM's depth a DEM rim may sit and still pass, in meters. A DEM is stored in single precision, whose
#: spacing at an elevation of 10 to 100 m is 1e-6 to 8e-6 m, so a cell authored to give exactly SWMM's depth reads
#: back off it by that much; 1e-5 m is above that noise and far below any shortfall a pipe could feel.
DEM_RIM_TOLERANCE_M = 1e-5


@dataclass(frozen=True)
class SwmmNodeDepth:
    """The minimum depth SWMM allows one node, in meters, or the reason the toolkit cannot compute it.

    ``depth_m`` is the crown floor: the highest ``offset + yFull`` over the link ends that SWMM extends the node to
    (``link_validate``, link.c). The authored ``[JUNCTIONS]`` ``MaxDepth`` is not part of it.
    ``depth_m`` is None, and ``reason`` set, when a link
    end that counts has a cross-section or offset the toolkit does not read.
    """

    name: str
    invert_m: float | None
    depth_m: float | None
    reason: str | None


def swmm_assigned_node_depths(text: str, names) -> dict[str, SwmmNodeDepth]:
    """The depth SWMM assigns each of ``names``, read from the model ``text``, in meters.

    Mirrors the floor SWMM forces on every node, the minimum viable rim (``link_validate`` raises the node's depth to
    ``offset + yFull`` of every conduit end at the node, whatever the authored ``MaxDepth``; the ``MaxDepth`` token is
    deliberately not read, since a node whose DEM rim is below an authored ``MaxDepth`` but above its crown is viable),
    and of the upstream end of an orifice, weir or outlet,
    where ``yFull`` is the first geometry of the cross-section. Under ``LINK_OFFSETS ELEVATION`` an offset is an
    elevation and SWMM first subtracts the node's invert (``link_getOffsetHeight``, never below 0); a ``*`` offset is
    0. A US-unit model is converted to meters by 0.3048, as the solver does.

    Not computable, so reported with a reason and never guessed: a node that is no ``[JUNCTIONS]`` row, a conduit
    end whose offset is not a number or whose cross-section is IRREGULAR, DUMMY or has a non-positive or non-numeric
    first geometry, and the upstream end of an orifice, weir or outlet (not a bottom orifice), whose ``yFull`` the
    toolkit does not read. A pump never raises a node.
    """
    from hhemt.swmm_runnable import _fold, _node_rows, _number, _parse
    from hhemt.swmm_runoff_modeling import _US_FLOW_UNITS, flow_units_from_inp_text

    sections = _parse(text)
    scale = _FEET_TO_METERS if flow_units_from_inp_text(text) in _US_FLOW_UNITS else 1.0
    elevation_offsets = any(
        len(t) > 1 and _fold(t[0]) == "LINK_OFFSETS" and _fold(t[1]).startswith("ELEV")
        for t in _node_rows(sections, "OPTIONS")
    )
    inverts = {
        _fold(t[0]): _number(t[1])
        for sec in ("JUNCTIONS", "OUTFALLS", "STORAGE", "DIVIDERS")
        for t in _node_rows(sections, sec)
        if len(t) > 1
    }
    junctions = {_fold(t[0]): t for t in _node_rows(sections, "JUNCTIONS") if len(t) > 1}
    conduits = [t for t in _node_rows(sections, "CONDUITS") if len(t) > 2]
    xsections = {_fold(t[0]): t for t in _node_rows(sections, "XSECTIONS") if len(t) > 2}
    upstream_only = [
        (kind, t)
        for kind, section in (("orifice", "ORIFICES"), ("weir", "WEIRS"), ("outlet", "OUTLETS"))
        for t in _node_rows(sections, section)
        if len(t) > 2
    ]

    def height(token: str | None, invert: float | None) -> float | None:
        """An offset as a height above the node's invert (SWMM's ``link_getOffsetHeight``), None if unreadable."""
        if token is None or token == "*":
            return 0.0
        value = _number(token)
        if value is None:
            return None
        if elevation_offsets:
            if invert is None:
                return None
            value -= invert
        return max(value, 0.0)

    depths: dict[str, SwmmNodeDepth] = {}
    for name in names:
        key = _fold(name)
        row = junctions.get(key)
        invert = inverts.get(key)
        if row is None or invert is None:
            depths[name] = SwmmNodeDepth(name, None, None, f"{name} is not a [JUNCTIONS] row of the model")
            continue
        full = 0.0  # the crown floor only: an authored MaxDepth above the crown does not raise the minimum
        reason: str | None = None
        for link in conduits:
            for end, offset_index in ((1, 5), (2, 6)):
                if _fold(link[end]) != key:
                    continue
                xs = xsections.get(_fold(link[0]))
                y_full = _number(xs[2]) if xs is not None and len(xs) > 2 else None
                shape = _fold(xs[1]) if xs is not None else None
                rise = height(link[offset_index] if len(link) > offset_index else None, invert)
                if y_full is None or y_full <= 0 or shape in ("IRREGULAR", "DUMMY") or rise is None:
                    reason = (
                        f"conduit {link[0]} ends at {name} with "
                        + (
                            f"the cross-section {shape} (first geometry {None if xs is None else xs[2]})"
                            if rise is not None
                            else f"the offset {link[offset_index]!r}"
                        )
                        + ", whose crown rise the toolkit cannot read"
                    )
                    continue
                full = max(full, rise + y_full)
        for kind, link in upstream_only:
            if _fold(link[1]) == key and not (kind == "orifice" and len(link) > 3 and _fold(link[3]) == "BOTTOM"):
                reason = f"{kind} {link[0]} leaves {name}, and its crown rise is not one the toolkit reads"
        depths[name] = SwmmNodeDepth(
            name,
            invert * scale,
            None if reason else full * scale,
            reason,
        )
    return depths


@dataclass(frozen=True)
class DemRimFinding:
    """One coupled node whose DEM rim the solver would not take as it is.

    ``cell_m`` is the DEM cell elevation (None when the cell is off the grid), ``dem_rim_m`` the cell minus the invert
    (None when the cell gives no rim), ``swmm_depth_m`` the depth SWMM assigns (None when not computable) and
    ``shortfall_m`` that depth minus the DEM rim, set only when both exist. ``reason`` is one sentence naming why.
    """

    name: str
    invert_m: float | None
    swmm_depth_m: float | None
    cell_m: float | None
    dem_rim_m: float | None
    shortfall_m: float | None
    reason: str


def dem_rim_findings(text: str, names, rds_dem) -> list[DemRimFinding]:
    """The nodes among ``names`` whose DEM rim is not at or above the depth SWMM assigns, in the order of ``names``.

    For each node, with ``text`` the model the solver reads and ``rds_dem`` the DEM it loads: the DEM rim the solver
    would set is ``cell elevation - invert`` (the cell is TRITON's own, ``dem_cell_elevations``), and the node is
    returned when the cell is off the grid, no-data, not finite or at or below the invert (the cases the solver
    refuses at run start), when SWMM's depth is not computable, or when the DEM rim is below SWMM's depth by more than
    ``DEM_RIM_TOLERANCE_M`` (the band the solver accepts silently, where the rim becomes shallower than the node's own
    pipe). A node at exactly SWMM's depth, or above it, is not returned.
    """
    import math

    from hhemt.swmm_runnable import _fold, _parse
    from hhemt.water_level_boundary import dem_cell_elevations, model_coordinates

    names = list(names)
    if not names:
        return []
    assigned = swmm_assigned_node_depths(text, names)
    coordinates = model_coordinates(_parse(text))
    located = {
        n: (*coordinates[_fold(n)], assigned[n].invert_m if assigned[n].invert_m is not None else 0.0)
        for n in names
        if _fold(n) in coordinates
    }
    cells = dem_cell_elevations(rds_dem, located)
    nodata = rds_dem.rio.nodata
    findings: list[DemRimFinding] = []
    for name in names:
        depth = assigned[name]
        invert = depth.invert_m
        if name not in located:
            findings.append(
                DemRimFinding(name, invert, depth.depth_m, None, None, None, f"{name} has no [COORDINATES] row")
            )
            continue
        z = cells[name]
        if z is None:
            findings.append(
                DemRimFinding(name, invert, depth.depth_m, None, None, None, "its coordinates fall outside the DEM")
            )
        elif not math.isfinite(z) or (nodata is not None and z == float(nodata)):
            findings.append(DemRimFinding(name, invert, depth.depth_m, z, None, None, "its DEM cell has no value"))
        elif invert is None or z <= invert:
            findings.append(
                DemRimFinding(name, invert, depth.depth_m, z, None, None, "its DEM cell is at or below the invert")
            )
        elif depth.depth_m is None:
            findings.append(
                DemRimFinding(
                    name, invert, None, z, z - invert, None, f"SWMM's depth is not computable: {depth.reason}"
                )
            )
        elif (z - invert) < depth.depth_m - DEM_RIM_TOLERANCE_M:
            short = depth.depth_m - (z - invert)
            findings.append(
                DemRimFinding(
                    name,
                    invert,
                    depth.depth_m,
                    z,
                    z - invert,
                    short,
                    f"its DEM rim is {short:.4f} m below the depth SWMM assigns",
                )
            )
    return findings


def swmm_role_text(
    cfg_system, role: str, cfg_analysis=None, *, strict_depths: bool = False, externalize: bool = True
) -> str:
    """The text of one role model derived from ``cfg_system.SWMM_model``.

    ``cfg_analysis`` names the outfalls a water_level record converts, the series each matched outfall reads, the
    DEM that gives a converted node its depth, and nothing else; without it only ``swmm_outfalls_to_couple`` converts
    and nothing is matched. The result is cached on the model's path and mtime, the converted and matched sets, the
    series each matched outfall reads, and the path and mtime of the DEM. ``strict_depths`` raises when the DEM gives
    a converted node no usable depth (V23), which scenario preparation requires before it writes any role.
    ``externalize=False`` leaves the base model's series rows inline and writes no extraction file: the readers that
    only judge the model's nodes and links (run-entry checks, preflight) take it, and nothing they return is written
    to a scenario.
    """
    from hhemt.exceptions import ProcessingError
    from hhemt.water_level_boundary import boundary_records_for

    path = Path(cfg_system.SWMM_model)
    converted = converted_outfalls(cfg_system, cfg_analysis)
    stage: tuple[tuple[str, str], ...] = ()
    if (
        cfg_analysis is not None
        and getattr(cfg_analysis, "water_level_boundary_series_source", None) == "weather_time_series"
    ):
        records, _problems = boundary_records_for(cfg_analysis, cfg_system)
        stage = tuple((n, r.series_name) for r in records if r.kind == "water_level" for n in r.outfalls)
    depths, problems, _provisional = _node_depths(path, converted, cfg_system, cfg_analysis)
    if problems and strict_depths:
        raise ProcessingError(operation="swmm_role_text", reason="; ".join(p.message for p in problems))
    dem = None
    if getattr(cfg_analysis, "dem_rim_resolution", None) is not None:
        dem = (
            processed_dem_path(cfg_system)
            if cfg_analysis.dem_rim_resolution == "processed"
            else Path(cfg_system.DEM_fullres)
        )
    dem_mtime = dem.stat().st_mtime_ns if dem is not None and dem.exists() else 0
    return _derived_role_text(
        str(path),
        path.stat().st_mtime_ns,
        role,
        converted,
        stage,
        tuple(sorted(depths.items())),
        tuple(sorted(_series_pointers(cfg_system).items())) if externalize else (),
        str(dem),
        dem_mtime,
        driven_stage_outfalls(cfg_analysis) if role == "hydraulics" else (),
    )


@lru_cache(maxsize=16)
def _derived_role_text(
    path: str,
    mtime_ns: int,
    role: str,
    outfalls: tuple[str, ...],
    stage: tuple[tuple[str, str], ...],
    depths: tuple[tuple[str, float], ...],
    pointers: tuple[tuple[str, str], ...],
    dem: str,
    dem_mtime_ns: int,
    driven: tuple[str, ...] = (),
) -> str:
    """The cached derivation. ``dem`` and ``dem_mtime_ns`` are part of the key so a changed DEM re-derives.

    ``driven`` names the open-water outfalls the hydraulics role writes as FIXED outfalls (empty for every other
    role and for every analysis that lists none, which leave the derived text untouched).
    """
    text = derive_role_model(
        _read_model_text(Path(path)),
        role,
        outfalls,
        node_depths=dict(depths),
        series_pointers=dict(pointers),
        stage_series=dict(stage),
    )
    return fixed_outfall_text(text, driven) if driven else text


def fixed_outfall_text(role_text: str, names: tuple[str, ...]) -> str:
    """``role_text`` with each outfall of ``names`` rewritten as a FIXED outfall whose stage is its invert.

    TRITON overwrites a driven outfall's stage before every SWMM step (the solver key ``swmm_driven_outfalls``), so
    the value written here is only the stage a step would read before TRITON's first write: the outfall's invert,
    which the solver also uses for a dry cell. The row keeps its name, invert, gate flag and route-to; a FREE or
    NORMAL row gains the stage token and a TIDAL, TIMESERIES or FIXED row has its stage token replaced. Every other
    line is untouched, line endings included.

    Raises
    ------
    ConfigurationError
        A name of ``names`` is not an ``[OUTFALLS]`` row of ``role_text``.
    """
    from hhemt.exceptions import ConfigurationError
    from hhemt.swmm_runnable import _all, _fold, _is_data_row, _parse, _prefix_match, _render, _tokens

    wanted = {_fold(n) for n in names}
    sections = _parse(role_text)
    found: set[str] = set()
    for section in _all(sections, "OUTFALLS"):
        for i, line in enumerate(section.lines):
            toks = _tokens(line) if _is_data_row(line) else []
            if len(toks) < 3 or _fold(toks[0]) not in wanted:
                continue
            found.add(_fold(toks[0]))
            has_stage = any(_prefix_match(toks[2], kind) for kind in ("FIXED", "TIDAL", "TIMESERIES"))
            rest = toks[4:] if has_stage else toks[3:]
            eol = line[len(line.rstrip("\r\n")) :]
            tail = (" " + " ".join(rest)) if rest else ""
            section.lines[i] = f"{toks[0]:<16} {toks[1]:<10} {'FIXED':<10} {toks[1]}{tail}{eol}"
    missing = [n for n in names if _fold(n) not in found]
    if missing:
        raise ConfigurationError(
            field="swmm_open_water_outfalls",
            message=f"{missing} {'is' if len(missing) == 1 else 'are'} not an [OUTFALLS] row of the SWMM model.",
            fix_hint="List only names that appear in the [OUTFALLS] section of SWMM_model.",
        )
    return _render(sections)


def write_swmm_role_model(scenario: "TRITONSWMM_scenario", role: str, destination: Path) -> None:
    """Write the ``role`` model of this scenario: the derived role with the scenario's forcing rewritten in.

    The hydraulics and full roles take ``swmm_threads``; the hydrology role routes by kinematic wave over one link, so a
    thread count there is inert and none is written.
    """
    cfg_analysis = scenario._analysis.cfg_analysis
    text = rewrite_forcing(
        swmm_role_text(scenario._system.cfg_system, role, cfg_analysis, strict_depths=True),
        _swmm_fill_values(scenario, destination),
    )
    if role in ("hydraulics", "full"):
        text = set_threads(text, cfg_analysis.swmm_threads)
        scenario.log.swmm_threads_written.set({role: cfg_analysis.swmm_threads})
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8", newline="") as f:
        f.write(text)


def swmm_model_problems(cfg_system) -> list[tuple[str, str, str]]:
    """Every reason ``cfg_system.SWMM_model`` cannot be derived into the three role models.

    Returns ``(field, message, fix_hint)`` triples, each naming the offender. The reasons: a ``${...}``
    placeholder (templates are not used); a listed outfall that is not an ``[OUTFALLS]`` row; a listed outfall
    with no conduit at an end, with the kinds of the links it has (only a conduit end crown-raises a node
    recorded at MaxDepth 0, which the coupled solver needs for a converted node's rim); a rain-gauge mapping
    that lacks the configured gage-id column; and a ``[SUBCATCHMENTS]`` row naming a rain gauge that is not in
    ``subcatchment_raingage_mapping``. An unreadable model returns ``[]``:
    path existence is judged elsewhere.
    """
    from hhemt.swmm_runnable import _fold, _node_rows, _parse

    path = Path(cfg_system.SWMM_model)
    try:
        text = _read_model_text(path)
    except OSError:
        return []
    problems: list[tuple[str, str, str]] = []
    if "${" in text:
        problems.append(
            (
                "system.SWMM_model",
                f"{path} carries a ${{...}} placeholder; a runnable model carries no placeholders; "
                "templates are not used",
                "Fill the placeholders: SWMM_model is one plain runnable .inp the toolkit derives the roles from",
            )
        )
    try:
        sections = _parse(text)
    except Exception as exc:  # an unknown section header; derivation names it again at run time
        return [*problems, ("system.SWMM_model", str(exc), "Use only the 57 SWMM 5.2 sections")]
    outfall_names = {_fold(r[0]): r[0] for r in _node_rows(sections, "OUTFALLS") if r}
    from hhemt.water_level_boundary import missing_series_files, model_series, reserved_prefix_names

    catalog = model_series(sections, path.parent)
    reserved = reserved_prefix_names(sections, catalog)
    if reserved:
        problems.append(
            (
                "system.SWMM_model",
                f"{path} has node or series name(s) {reserved} that begin the reserved prefix {'hhemt_'!r}; "
                "the toolkit writes objects under that prefix",
                "Rename each object so its name does not begin with hhemt_",
            )
        )
    missing_files = missing_series_files(catalog)
    if missing_files:
        problems.append(
            (
                "system.SWMM_model",
                f"{path} has FILE series whose file does not exist (series, resolved path): {missing_files}",
                "Create each file, or correct its name, which SWMM resolves against the directory of SWMM_model",
            )
        )
    for name in cfg_system.swmm_outfalls_to_couple:
        key = _fold(name)
        if key not in outfall_names:
            problems.append(
                (
                    "system.swmm_outfalls_to_couple",
                    f"{name!r} is not an [OUTFALLS] row of {path}",
                    "List only names that appear in the [OUTFALLS] section of SWMM_model",
                )
            )
    mapping = cfg_system.subcatchment_raingage_mapping
    colname = cfg_system.subcatchment_raingage_mapping_gage_id_colname
    if mapping is not None and colname and Path(mapping).exists():
        import pandas as pd

        df_mapping = pd.read_csv(mapping)
        if colname not in df_mapping.columns:
            problems.append(
                (
                    "system.subcatchment_raingage_mapping_gage_id_colname",
                    f"{colname!r} is not a column of {mapping}; its columns are {list(df_mapping.columns)}",
                    "Set subcatchment_raingage_mapping_gage_id_colname to a column the mapping has",
                )
            )
            return problems
        mapped = {_fold(str(g)) for g in df_mapping[colname].unique()}
        used = {_fold(r[1]): r[1] for r in _node_rows(sections, "SUBCATCHMENTS") if len(r) > 1}
        unmapped = sorted(v for k, v in used.items() if k not in mapped)
        if unmapped:
            problems.append(
                (
                    "system.subcatchment_raingage_mapping",
                    f"[SUBCATCHMENTS] of {path} name rain gauge(s) {unmapped} that are not in the "
                    f"{colname!r} column of {mapping}",
                    "Add each gauge to subcatchment_raingage_mapping, or point the subcatchments at a mapped gauge",
                )
            )
    return problems
