"""
SWMM Utility Functions

This module contains shared utility functions for filling SWMM role models.
These functions are used across multiple SWMM model creation modules to avoid code duplication.
"""

import os
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


def converted_outfalls(cfg_system, cfg_analysis=None) -> tuple[str, ...]:
    """The outfalls that become coupled junctions: ``swmm_outfalls_to_couple`` plus those a water_level record names."""
    from hhemt.water_level_boundary import boundary_records_for, matched_outfall_names

    matched: tuple[str, ...] = ()
    if cfg_analysis is not None:
        records, _problems = boundary_records_for(cfg_analysis, cfg_system)
        matched = matched_outfall_names(records)
    return tuple(dict.fromkeys([*cfg_system.swmm_outfalls_to_couple, *matched]))


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
    from hhemt.swmm_runnable import _parse
    from hhemt.water_level_boundary import converted_node_depths

    resolution = getattr(cfg_analysis, "dem_rim_resolution", None)
    all_nodes = bool(getattr(cfg_analysis, "swmm_node_rim_from_dem", False)) and resolution == "full"
    if (not converted and not all_nodes) or resolution is None:
        return {}, [], False
    sections = _parse(_read_model_text(path))
    names = list(converted)
    if all_nodes:
        from hhemt.scenario import InpText, coupling_candidates

        base = InpText(derive_role_model(_read_model_text(path), "hydraulics", converted))
        names.extend(coupling_candidates(cfg_system, inp_path=base, cfg_analysis=cfg_analysis))
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
) -> str:
    """The cached derivation. ``dem`` and ``dem_mtime_ns`` are part of the key so a changed DEM re-derives."""
    return derive_role_model(
        _read_model_text(Path(path)),
        role,
        outfalls,
        node_depths=dict(depths),
        series_pointers=dict(pointers),
        stage_series=dict(stage),
    )


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
