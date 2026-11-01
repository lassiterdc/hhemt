"""Completion-record, configuration and input-file writers for the golden analysis trees.

A golden is a finished analysis tree plus the files hhemt needs to construct an analysis object from it
(``tests/goldens/SPEC.md`` section 1); ``tests/test_golden_conformance.py`` asks hhemt's own readers whether
the trees this generator writes are finished analyses.

Follows ``tests/goldens/SPEC.md`` sections 3 to 6 (layout, completion records, configuration and
inputs with the ``@GOLDEN_ROOT@`` relocation token, the minimal software root). A value SPEC.md leaves open
is marked ``[CHOICE]`` at the site that picks it, for the choices that matter to a reader; the marker is not
exhaustive, and an unmarked literal is not thereby fixed by SPEC.md.

Imports: standard library, ``numpy``, ``xarray`` and the sibling ``raw`` module only. No YAML library is
imported (it is not on the import allow-list that ``tests/test_golden_generator_independence.py`` enforces,
which exists so that hhemt is never asked to accept trees hhemt itself wrote); configuration files are
emitted by a small writer for the flat mappings, lists and scalars they hold, with every string double-quoted.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from .raw import NODE_OUTFALL, Dimensions, Network

# SPEC.md 5.2: extraction replaces the token with the extraction directory's path, in the text file types listed there.
TOKEN = "@GOLDEN_ROOT@"
# SPEC.md 3: the value at the commit that regenerates the golden (26 at b7e963b2). A literal, because this module
# imports nothing from hhemt.
LAYOUT_VERSION = 26
CELL_SIZE_M = 10.0  # SPEC.md 2 (cell size); the config field target_dem_resolution is in the SPEC.md 5.1 table
FIXED_TIME = "2000-01-01T00:00:00+00:00"
MODELS = ("triton", "tritonswmm", "swmm")  # model_type values of the flag and log names
ARM_MODEL = {"triton": "triton", "tritonswmm": "tritonswmm", "swmm": "swmm"}

# [CHOICE] Compute columns of the four sensitivity members (SPEC.md 5.1).
SENSITIVITY_MEMBERS: dict[int, dict[str, Any]] = {
    0: {"run_mode": "mpi", "n_mpi_procs": 2, "n_omp_threads": 1, "n_gpus": 0, "n_nodes": 1},
    1: {"run_mode": "openmp", "n_mpi_procs": 1, "n_omp_threads": 2, "n_gpus": 0, "n_nodes": 1},
    2: {"run_mode": "hybrid", "n_mpi_procs": 2, "n_omp_threads": 2, "n_gpus": 0, "n_nodes": 1},
    3: {"run_mode": "serial", "n_mpi_procs": 1, "n_omp_threads": 1, "n_gpus": 0, "n_nodes": 1},
}


def tok(*parts: str) -> str:
    """A path inside the golden as the relocation token followed by ``/``-joined relative parts."""
    return TOKEN + "".join("/" + p for p in parts)


# --------------------------------------------------------------------------------------
# Plain file helpers
# --------------------------------------------------------------------------------------


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, obj: Any, *, indent: int | None = 2) -> None:
    write_text(path, json.dumps(obj, indent=indent) + "\n")


def _scalar(v: Any) -> str:
    # None, bool, int and float are written bare (a float by ``repr``); every other value is written as the
    # double-quoted ``str(v)``, so a mapping inside a list comes out as a quoted string and not as nested YAML.
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int | float):
        return repr(v)
    return json.dumps(str(v))  # a JSON string is a valid YAML double-quoted scalar


def yaml_text(obj: dict[str, Any], indent: int = 0) -> str:
    """Emit a mapping of scalars, lists of scalars and nested mappings as block YAML.

    A list element is written by ``_scalar``. An empty list is written as a bare ``key:`` line, which YAML
    reads as null and not as an empty list; an empty mapping is written as ``key: {}``."""
    pad = " " * indent
    lines: list[str] = []
    for key, val in obj.items():
        if isinstance(val, dict):
            if val:
                lines.append(f"{pad}{key}:")
                lines.append(yaml_text(val, indent + 2).rstrip("\n"))
            else:
                lines.append(f"{pad}{key}: {{}}")
        elif isinstance(val, list | tuple):
            lines.append(f"{pad}{key}:")
            lines += [f"{pad}- {_scalar(x)}" for x in val]
        else:
            lines.append(f"{pad}{key}: {_scalar(val)}")
    return "\n".join(lines) + "\n"


def dir_size_mib(path: Path) -> float:
    total = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return round(total / (1024 * 1024), 4)


# --------------------------------------------------------------------------------------
# Layout stamp, software root, DEM, weather and tabular inputs (SPEC 4.1, 5, 6)
# --------------------------------------------------------------------------------------


def write_version_stamp(directory: Path) -> None:
    """``_version.json`` (SPEC.md 4.1); times and version fixed so regeneration is deterministic."""
    write_json(
        directory / "_version.json",
        {
            "created_at": "2000-01-01T00:00:00Z",
            "layout_version": LAYOUT_VERSION,
            "migration_history": [],
            "toolkit_version": "0.0.0+golden",
        },
    )


def dem_text(dims: Dimensions, value: float) -> str:
    """ESRI ASCII grid of the processed DEM or Mannings raster (SPEC.md 2 and 3)."""
    head = (
        f"ncols         {dims.cols}\nnrows         {dims.rows}\nxllcorner     0.0\nyllcorner     0.0\n"
        "cellsize      10.0\nNODATA_value  -9999\n"
    )
    row = " ".join([f"{value:.5f}"] * dims.cols)
    return head + "\n".join([row] * dims.rows) + "\n"


def write_software_root(root: Path) -> None:
    """The minimal software root: compilation logs only, no executable (SPEC.md section 6).

    hhemt reads these logs to decide that each build succeeded (SPEC.md 6): the coupled TRITON-SWMM log must
    hold both ``Built target swmm5`` and ``[100%] Built target triton.exe``, the TRITON-only log needs only
    the second line, and the standalone SWMM log needs ``Built target runswmm`` or ``Built target swmm5``."""
    soft = root / "_software"
    both = (
        "[  4%] Building C object swmm5.c.o\n[ 96%] Built target swmm5\n"
        "[ 98%] Building CXX object triton.cpp.o\n[100%] Built target triton.exe\n"
    )
    write_text(soft / "triton" / "build_tritonswmm_cpu" / "compilation.log", both)
    write_text(soft / "triton" / "build_triton_cpu" / "compilation.log", "[100%] Built target triton.exe\n")
    write_text(
        soft / "swmm" / "compilation.log",
        "[ 11%] Built target swmm-output\n[ 96%] Built target swmm5\n[100%] Linking C executable runswmm\n"
        "[100%] Built target runswmm\n",
    )


def node_coordinates(count: int, dims: Dimensions) -> list[tuple[float, float]]:
    """One ``[COORDINATES]`` point per node (junctions, then the outfall), each at the centre of its own TRITON cell.

    hhemt refuses to plan a coupled system with no junction carrying a ``[COORDINATES]`` row and refuses a
    coupling node outside the grid (``analysis.py`` ``_refuse_coupled_system_with_no_coupling_candidate`` and
    ``_refuse_coupling_nodes_outside_triton_grid``; SPEC.md 2). The grid spans x 0 to ``cols * 10 m`` and
    y 0 to ``rows * 10 m`` (SPEC.md 2); node ``i`` takes the centre of cell ``(i // ncols, i % ncols)``.
    ``count`` is the length of ``Network.nodes``: the 13 junctions in file order, then the outfall as the last
    node (SPEC.md 2; ``raw.default_network``)."""
    if count > dims.rows * dims.cols:
        raise ValueError(f"{count} nodes do not fit one per cell of a {dims.rows} by {dims.cols} grid")
    return [((i % dims.cols + 0.5) * CELL_SIZE_M, (i // dims.cols + 0.5) * CELL_SIZE_M) for i in range(count)]


def inp_text(network: Network, dims: Dimensions, *, template: bool) -> str:
    """A small SWMM input file carrying ``FLOW_UNITS CMS`` (SPEC.md 5.1) and the network of
    ``raw.default_network`` (the sibling module's function, which the caller passes in as ``network``).

    ``template=True`` leaves the ``${...}`` placeholders hhemt substitutes at preparation; otherwise the
    values are filled with the reduced run's dates (2000-01-01, 120 s: 12 reporting steps of 10 s, SPEC.md 2).
    The rain gage is ``RG_synth``, the rainfall variable name of SPEC.md 5.1."""
    # ``end_h`` holds seconds, not hours. END_TIME below renders it as 00:MM:SS, which is
    # valid only for a run under one hour (120 s here).
    end_h = dims.n_steps * dims.report_step_s
    if template:
        dates = {
            "START_DATE": "${START_DATE}",
            "START_TIME": "${START_TIME}",
            "END_DATE": "${END_DATE}",
            "END_TIME": "${END_TIME}",
            "REPORT_STEP": "${REPORT_STEP}",
        }
    else:
        dates = {
            "START_DATE": "01/01/2000",
            "START_TIME": "00:00:00",
            "END_DATE": "01/01/2000",
            "END_TIME": f"00:{end_h // 60:02d}:{end_h % 60:02d}",
            "REPORT_STEP": f"00:00:{dims.report_step_s:02d}",
        }
    out = [
        "[TITLE]",
        "Synthetic golden model",
        "",
        "[OPTIONS]",
        "FLOW_UNITS            CMS",
        "INFILTRATION          HORTON",
        "FLOW_ROUTING          DYNWAVE",
        f"START_DATE            {dates['START_DATE']}",
        f"START_TIME            {dates['START_TIME']}",
        f"END_DATE              {dates['END_DATE']}",
        f"END_TIME              {dates['END_TIME']}",
        f"REPORT_STEP           {dates['REPORT_STEP']}",
        "WET_STEP              00:00:10",
        "DRY_STEP              00:01:00",
        "ROUTING_STEP          00:00:01",
        "",
        "[JUNCTIONS]",
    ]
    out += [
        f"{n.id}  {n.invert_m:.3f}  {n.max_depth_m:.3f}  0  0  0" for n in network.nodes if n.type_code != NODE_OUTFALL
    ]
    out += ["", "[OUTFALLS]"]
    out += [f"{n.id}  {n.invert_m:.3f}  FREE" for n in network.nodes if n.type_code == NODE_OUTFALL]
    out += ["", "[CONDUITS]"]
    out += [f"{lk.id}  {lk.from_node}  {lk.to_node}  {lk.length_m:.1f}  0.013  0  0  0  0" for lk in network.links]
    # [CHOICE] Every subcatchment drains to the node J1 and carries the trailing numbers 50, 10, 0.5, 0.5; SPEC.md
    # does not state them.
    out += ["", "[SUBCATCHMENTS]"]
    out += [f"{s.id}  RG_synth  J1  {s.area_ha:.3f}  50  10  0.5  0.5" for s in network.subcatchments]
    out += ["", "[COORDINATES]"]
    out += [
        f"{n.id}  {x:.1f}  {y:.1f}"
        for n, (x, y) in zip(network.nodes, node_coordinates(len(network.nodes), dims), strict=True)
    ]
    return "\n".join(out) + "\n"


def _weather_dataset(event_indices: list[int], dims: Dimensions) -> xr.Dataset:
    t0 = np.datetime64("2000-01-01T00:00:00", "ns")
    times = np.array([t0 + np.timedelta64(k * dims.report_step_s, "s") for k in range(dims.n_steps + 1)])
    ev = np.array(event_indices, dtype="int32")
    ramp = np.linspace(0.0, 1.0, dims.n_steps + 1, dtype="float32")
    # [CHOICE] The rain ramps from 0 to 5.0 (scaled by 1 + 0.1 per event index) and the water level from 0 to 0.2
    # (plus 0.01 per event index); SPEC.md 5.1 fixes this file's dimensions, variables and times, not these values.
    rain = np.stack([5.0 * ramp * np.float32(1 + 0.1 * e) for e in ev]).astype("float32")
    level = np.stack([0.2 * ramp + np.float32(0.01 * e) for e in ev]).astype("float32")
    return xr.Dataset(
        {"RG_synth": (("event_index", "time"), rain), "water_level": (("event_index", "time"), level)},
        coords={"event_index": ev, "time": times},
        attrs={"title": "synthetic weather for the golden analysis"},
    )


def write_system_inputs(root: Path, network: Network, dims: Dimensions, *, event_indices: list[int]) -> None:
    """Every file the configs name except ``inputs/sensitivity.csv`` (which ``write_configs`` writes), plus the
    processed DEM and Mannings grids (SPEC.md 3 and 5.1).

    ``event_indices`` are the event index values written to ``weather.nc`` and to
    ``weather_events_to_simulate.csv``."""
    inputs = root / "inputs"
    # [CHOICE] DEM elevation 50.0 everywhere; SPEC.md does not state it. The Mannings value 0.035 is
    # ``constant_mannings`` of SPEC.md 5.1.
    write_text(root / "elevation_10.00m.dem", dem_text(dims, 50.0))
    write_text(root / "mannings_10.00m.dem", dem_text(dims, 0.035))
    write_text(inputs / "dem_fullres.asc", dem_text(dims, 50.0))
    # The polygon is the DEM extent for the grid of 6 rows by 4 columns of 10 m cells (x 0 to 40 m, y 0 to 60 m,
    # SPEC.md 2), written as literals and not derived from ``dims``.
    poly = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0.0, 0.0], [40.0, 0.0], [40.0, 60.0], [0.0, 60.0], [0.0, 0.0]]],
                },
            }
        ],
    }
    line = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {"type": "LineString", "coordinates": [[0.0, 0.0], [40.0, 0.0]]},
            }
        ],
    }
    write_json(inputs / "watershed.geojson", poly)
    write_json(inputs / "boundary.geojson", line)
    for name in ("swmm_hydraulics", "swmm_hydrology", "swmm_full"):
        write_text(inputs / f"{name}.inp", inp_text(network, dims, template=True))
    rows = "".join(f"{s.id},RG_synth\n" for s in network.subcatchments)
    write_text(inputs / "subcatchment_raingage_mapping.csv", "subcatchment_id,raingage_id\n" + rows)
    write_text(
        inputs / "tritonswmm.cfg",
        "# TRITON-SWMM config file template - ${CASE_DESC}\nprint_option=huv\nmax_value_print_option=h\n"
        "time_step=1.0\ncourant=0.5\nhextra=0.001\n",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _weather_dataset(event_indices, dims).to_netcdf(inputs / "weather.nc")
    write_text(
        root / "weather_events_to_simulate.csv",
        "event_index\n" + "".join(f"{i}\n" for i in event_indices),
    )
    write_software_root(root)


# --------------------------------------------------------------------------------------
# Configuration files (SPEC.md 5.1)
# --------------------------------------------------------------------------------------


def system_config_dict(arms: tuple[str, ...] = MODELS) -> dict[str, Any]:
    return {
        "system_directory": TOKEN,
        "watershed_gis_polygon": tok("inputs", "watershed.geojson"),
        "DEM_fullres": tok("inputs", "dem_fullres.asc"),
        "SWMM_hydraulics": tok("inputs", "swmm_hydraulics.inp"),
        "TRITONSWMM_software_directory": tok("_software", "triton"),
        "TRITONSWMM_git_URL": "https://github.com/lassiterdc/triton.git",
        "SWMM_software_directory": tok("_software", "swmm"),
        "triton_swmm_configuration_template": tok("inputs", "tritonswmm.cfg"),
        "toggle_tritonswmm_model": "tritonswmm" in arms,
        "toggle_triton_model": "triton" in arms,
        "toggle_swmm_model": "swmm" in arms,
        "toggle_use_swmm_for_hydrology": True,
        "toggle_use_constant_mannings": True,
        "constant_mannings": 0.035,
        "target_dem_resolution": 10.0,
        "crs": {"horizontal_epsg": 3857, "vertical_epsg": 5703},
        "SWMM_hydrology": tok("inputs", "swmm_hydrology.inp"),
        "subcatchment_raingage_mapping": tok("inputs", "subcatchment_raingage_mapping.csv"),
        "subcatchment_raingage_mapping_gage_id_colname": "raingage_id",
        "SWMM_full": tok("inputs", "swmm_full.inp"),
    }


def analysis_config_dict(analysis_id: str, *, sensitivity: bool) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "analysis_id": analysis_id,
        "hhemt_sha": "0" * 40,
        "weather_event_indices": ["event_index"],
        "weather_timeseries": tok("inputs", "weather.nc"),
        "weather_time_series_timestep_dimension_name": "time",
        "rainfall_units": "mm/hr",
        "run_mode": "serial",
        "toggle_sensitivity_analysis": sensitivity,
        "toggle_storm_tide_boundary": True,
        "weather_events_to_simulate": tok("weather_events_to_simulate.csv"),
        "manhole_diameter": 1.2,
        "manhole_loss_coefficient": 0.1,
        "hydraulic_timestep_s": 1.0,
        "TRITON_reporting_timestep_s": 10.0,
        "open_boundaries": 1,
        "storm_tide_boundary_line_gis": tok("inputs", "boundary.geojson"),
        "weather_time_series_storm_tide_datavar": "water_level",
        "weather_time_series_spatial_mean_rainfall_datavar": "RG_synth",
        "storm_tide_units": "m",
        "multi_sim_run_method": "local",
        "target_processed_output_type": "zarr",
        "local_cpu_cores_for_workflow": 2,
        "report": {},
        "clear_raw": "none",
        "force_rerun": "none",
    }
    if sensitivity:
        cfg["sensitivity_analysis"] = tok("inputs", "sensitivity.csv")
    return cfg


def member_config_dict(member_id: int) -> dict[str, Any]:
    base = analysis_config_dict(f"member_{member_id}", sensitivity=False)
    base.update(SENSITIVITY_MEMBERS[member_id])
    base["dataset_license"] = "CC0-1.0"
    return base


def write_configs(root: Path, analysis_id: str, *, sensitivity: bool, arms: tuple[str, ...] = MODELS) -> None:
    write_text(root / "system_config.yaml", yaml_text(system_config_dict(arms)))
    write_text(root / "analysis_config.yaml", yaml_text(analysis_config_dict(analysis_id, sensitivity=sensitivity)))
    if sensitivity:
        header = "member_id,run_mode,n_mpi_procs,n_omp_threads,n_gpus,n_nodes\n"
        rows = "".join(
            f"{m},{c['run_mode']},{c['n_mpi_procs']},{c['n_omp_threads']},{c['n_gpus']},{c['n_nodes']}\n"
            for m, c in SENSITIVITY_MEMBERS.items()
        )
        write_text(root / "inputs" / "sensitivity.csv", header + rows)


# --------------------------------------------------------------------------------------
# Logs (SPEC.md 4.2 to 4.4)
# --------------------------------------------------------------------------------------


def system_log(root: Path, dims: Dimensions) -> dict[str, Any]:
    # ``root`` is not used by this function.
    shape = [1, dims.rows, dims.cols]
    return {
        "logfile": tok("system_log.json"),
        "dem_processed": True,
        "dem_shape": shape,
        "mannings_processed": True,
        "mannings_shape": shape,
        "compilation_tritonswmm_cpu_successful": True,
        "compilation_tritonswmm_gpu_successful": None,
        "compilation_triton_cpu_successful": True,
        "compilation_triton_gpu_successful": None,
        "compilation_swmm_successful": True,
        # SPEC.md 4.2: triton_head_sha, sif_sha256 and the producing-sha fields are null in a golden because it
        # has no real build.
        "triton_head_sha": None,
        "sif_sha256": None,
        "tritonswmm_producing_sha": None,
        "triton_only_producing_sha": None,
        "standalone_swmm_producing_version": None,
        "standalone_swmm_producing_sha": None,
        "system_datatree_consolidation_complete": None,
        "dem_crs_epsg": 3857,
        "vertical_crs_epsg": 5703,
    }


def scenario_log(event_rel: str, event_iloc: int, event_id: str) -> dict[str, Any]:
    # ``event_id`` is not used by this function.
    folder = tok(*event_rel.split("/"))
    return {
        "logfile": f"{folder}/scenario_prep_log.json",
        "event_iloc": event_iloc,
        "event_idx": {"event_index": event_iloc},
        "simulation_folder": folder,
        "swmm_rainfall_dat_files": {"RG_synth": f"{folder}/dats/grid-indRG_synth.dat"},
        "storm_tide_for_swmm": f"{folder}/dats/waterlevel.dat",
        "scenario_creation_complete": True,
        "inp_hydraulics_model_created_successfully": True,
        "inp_full_model_created_successfully": True,
        "inp_hydro_model_created_successfully": True,
        "hydro_swmm_sim_completed": True,
        "extbc_tseries_created": True,
        "extbc_loc_created": True,
        "hyg_timeseries_created": True,
        "hyg_locs_created": True,
        "inflow_nodes_in_hydraulic_inp_assigned": None,
        "triton_swmm_cfg_created": True,
        "triton_cfg_created": True,
        "sim_tritonswmm_executable_copied": True,
        "triton_backend_used": "cpu",
    }


# The written-markers of a model log. SPEC.md 4.3 lists only the markers set true per model (``_MODEL_TRUE``) and
# gives the rule that unmeasured optional fields are null. Whether the ten markers it does not list fall under that
# rule is not established here; this file writes them as None, and no marker as False.
_MODEL_FIELDS = (
    "performance_timeseries_written",
    "performance_summary_written",
    "TRITON_timeseries_written",
    "TRITON_summary_written",
    "raw_TRITON_outputs_cleared",
    "full_TRITON_timeseries_cleared",
    "SWMM_node_timeseries_written",
    "SWMM_link_timeseries_written",
    "SWMM_node_summary_written",
    "SWMM_link_summary_written",
    "raw_SWMM_outputs_cleared",
    "full_SWMM_timeseries_cleared",
    "raw_SWMM_binaries_reclaimed",
    "coupled_rpt_truncated",
    "hydro_out_reclaimed",
    "prep_inputs_reclaimed",
    "hydrographs_reclaimed",
    "standalone_rpt_reclaimed",
)
_MODEL_TRUE = {
    "tritonswmm": {
        "performance_timeseries_written",
        "performance_summary_written",
        "TRITON_timeseries_written",
        "TRITON_summary_written",
        "SWMM_node_timeseries_written",
        "SWMM_link_timeseries_written",
        "SWMM_node_summary_written",
        "SWMM_link_summary_written",
    },
    "triton": {
        "performance_timeseries_written",
        "performance_summary_written",
        "TRITON_timeseries_written",
        "TRITON_summary_written",
    },
    "swmm": {
        "SWMM_node_timeseries_written",
        "SWMM_link_timeseries_written",
        "SWMM_node_summary_written",
        "SWMM_link_summary_written",
    },
}


def model_log(event_rel: str, event_iloc: int, model: str, event_dir: Path, stores: dict[str, Path]) -> dict[str, Any]:
    """``log_{model}.json`` (SPEC.md 4.3): the written-markers and the processing-log entries of the
    model's stores. ``stores`` maps the store name WITHOUT its ``.zarr`` extension to the written path (the
    name is used as ``{name}.zarr``). ``event_dir`` is not used by this function."""
    folder = tok(*event_rel.split("/"))
    outputs: dict[str, Any] = {}
    for name, path in stores.items():
        outputs[f"{name}.zarr"] = {
            "filepath": f"{folder}/processed/{name}.zarr",
            "size_MiB": dir_size_mib(path),
            "time_elapsed_s": 1.0,
            "success": True,
            "notes": "",
            "warnings": "",
        }
    doc: dict[str, Any] = {
        "logfile": f"{folder}/log_{model}.json",
        "event_iloc": event_iloc,
        "event_idx": {"event_index": event_iloc},
        "simulation_folder": folder,
        "simulation_completed": True,
        "sim_run_time_minutes": 0.01,
        "n_resumes": None,
        "resume_reporting_tsteps": None,
        "processing_log": {"outputs": outputs},
    }
    for f in _MODEL_FIELDS:
        doc[f] = True if f in _MODEL_TRUE[model] else None
    return doc


def analysis_log(analysis_rel: str, store_name: str, *, experiment: bool, fingerprint: str) -> dict[str, Any]:
    """``log.json`` of an analysis, a member, or the experiment root (SPEC.md 4.4).

    ``fingerprint`` is the caller-supplied ``consolidation_inputs_fingerprint``, a 40-hex string
    (SPEC.md 4.4). ``experiment`` selects which consolidation-complete field is True: the sensitivity one
    when True, the single-analysis one when False; the other is null."""
    parts = analysis_rel.split("/") if analysis_rel else []
    base = tok(*parts)
    return {
        "logfile": f"{base}/log.json",
        "datatree_consolidation_complete": None if experiment else True,
        "consolidation_version": 1,
        "consolidation_inputs_fingerprint": fingerprint,
        "consolidation_build_stamp": None,
        "sensitivity_datatree_consolidation_complete": True if experiment else None,
        "cpu_backend_available": False,
        "gpu_backend_available": False,
        "processing_log": {
            "outputs": {
                store_name: {
                    "filepath": f"{base}/{store_name}",
                    "size_MiB": 0.0,
                    "time_elapsed_s": 1.0,
                    "success": True,
                    "notes": "",
                    "warnings": "",
                }
            }
        },
        "tmux_session_name": None,
        "snakemake_pid": None,
        "workflow_submission_time": None,
        "workflow_submission_mode": None,
        "workflow_submission_node": None,
        "orchestrator_slurm_jobid": None,
        "workflow_canceled": None,
        "workflow_cancellation_time": None,
        "multi_allocation_in_progress": None,
    }


# --------------------------------------------------------------------------------------
# Flags (SPEC.md 4.5 and 4.6)
# --------------------------------------------------------------------------------------


def write_flag(
    status_dir: Path,
    name: str,
    *,
    rule_name: str,
    model_type: str | None = None,
    member_id: str | None = None,
    event_id: str | None = None,
) -> None:
    """A zero-byte ``NAME.flag`` plus its JSON sidecar ``NAME.flag.json`` (SPEC.md 4.5).

    The sidecar holds ``rule_name``, ``written_at`` and ``slurm_job_id`` (null) and, when given, ``model_type``,
    ``sa_id`` and ``event_id``; the ``member_id`` parameter is written under the key ``sa_id``.

    [CHOICE] ``rule_name`` values: SPEC.md does not list them."""
    status_dir.mkdir(parents=True, exist_ok=True)
    (status_dir / name).write_bytes(b"")
    payload: dict[str, Any] = {"rule_name": rule_name, "written_at": FIXED_TIME, "slurm_job_id": None}
    if model_type is not None:
        payload["model_type"] = model_type
    if member_id is not None:
        payload["sa_id"] = member_id
    if event_id is not None:
        payload["event_id"] = event_id
    write_json(status_dir / f"{name}.json", payload)


def write_multisim_flags(status: Path, event_ids: list[str], arms: tuple[str, ...]) -> None:
    # Flag names are the multi-sim families of SPEC.md 4.5. The order of writing carries no meaning: the generator
    # sets no modification time (SPEC.md 4.5).
    write_flag(status, "a_setup_complete.flag", rule_name="setup")
    for eid in event_ids:
        write_flag(status, f"b_prepare_evt-{eid}_complete.flag", rule_name="prepare_scenario", event_id=eid)
        for arm in arms:
            m = ARM_MODEL[arm]
            write_flag(status, f"c_run_{m}_evt-{eid}_complete.flag", rule_name=f"run_{m}", model_type=m, event_id=eid)
            write_flag(
                status,
                f"d_process_{m}_evt-{eid}_complete.flag",
                rule_name=f"process_{m}",
                model_type=m,
                event_id=eid,
            )
        write_flag(
            status,
            f"f_consolidate_scenario_evt-{eid}_complete.flag",
            rule_name="consolidate_scenario",
            event_id=eid,
        )
    write_flag(status, "e_consolidate_complete.flag", rule_name="consolidate")


def write_sensitivity_flags(status: Path, members: list[int], event_ids: list[str], arms: tuple[str, ...]) -> None:
    # Flag names are the sensitivity families of SPEC.md 4.5; ``target_0`` is the one setup target of the
    # four-member golden (SPEC.md 4.5).
    write_flag(status, "a_setup_target_0_complete.flag", rule_name="setup_target")
    for m in members:
        cfg = SENSITIVITY_MEMBERS[m]
        # hhemt's compare-and-write byte form (sensitivity_analysis.py, `json.dumps(payload, sort_keys=True,
        # separators=(",", ":")) + "\n"`): any other form is rewritten by a plan, which makes every member stale.
        # Payload shape: SPEC.md 4.5.
        payload = {"__schema_version__": 1, "fields": {k: cfg[k] for k in sorted(cfg)}}
        write_text(
            status / f"member-{m}_inputs.json", json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        )
        for eid in event_ids:
            write_flag(
                status,
                f"b_prepare_member-{m}_evt-{eid}_complete.flag",
                rule_name="prepare_member",
                member_id=str(m),
                event_id=eid,
            )
            for arm in arms:
                mt = ARM_MODEL[arm]
                write_flag(
                    status,
                    f"c_run_{mt}_member-{m}_evt-{eid}_complete.flag",
                    rule_name=f"run_{mt}_member",
                    model_type=mt,
                    member_id=str(m),
                    event_id=eid,
                )
                write_flag(
                    status,
                    f"d_process_{mt}_member-{m}_evt-{eid}_complete.flag",
                    rule_name=f"process_{mt}_member",
                    model_type=mt,
                    member_id=str(m),
                    event_id=eid,
                )
        write_flag(status, f"e_consolidate_member-{m}_complete.flag", rule_name="consolidate_member", member_id=str(m))
    write_flag(status, "f_consolidate_experiment_complete.flag", rule_name="consolidate_experiment")


def write_done_flags(event_dir: Path, arms: tuple[str, ...]) -> None:
    """``{store}.zarr.done`` for the per-scenario TRITON timeseries stores; content is ``ok`` (SPEC.md 4.6).

    ``event_dir / "processed"`` must already exist: nothing here creates it."""
    proc = event_dir / "processed"
    names = []
    if "tritonswmm" in arms:
        names.append("TRITONSWMM_TRITON_tseries.zarr.done")
    if "triton" in arms:
        names.append("TRITON_only_tseries.zarr.done")
    for n in names:
        (proc / n).write_text("ok", encoding="utf-8")


# --------------------------------------------------------------------------------------
# Per-event records and prepared inputs (SPEC.md 4.3, 4.7, 4.8)
# --------------------------------------------------------------------------------------


def _cfg_text(kind: str, event_rel: str, dims: Dimensions) -> str:
    # ``kind`` is ``TRITON`` or ``TRITONSWMM``. It selects the header comment and ``output_folder`` (``out_triton``
    # or ``out_tritonswmm``, the raw-layer directories of SPEC.md 3); every other line is the same for both.
    folder = tok(*event_rel.split("/"))
    out = "out_triton" if kind == "TRITON" else "out_tritonswmm"
    lines = [
        f"# {kind} config file (golden)",
        f'output_folder="{out}"',
        f'dem_filename="{tok("elevation_10.00m.dem")}"',
        f'n_infile="{tok("mannings_10.00m.dem")}"',
        f'hydrograph_filename="{folder}/strmflow/tseries.hyg"',
        f'src_loc_file="{folder}/strmflow/loc.txt"',
        f'extbc_dir="{folder}/extbc"',
        f'extbc_file="{folder}/extbc/loc.extbc"',
        "output_format=BIN",
        "print_option=huv",
        "max_value_print_option=h",
        "sim_start_time=0",
        f"sim_duration={dims.n_steps * dims.report_step_s}",
        f"print_interval={float(dims.report_step_s)}",
    ]
    return "\n".join(lines) + "\n"


def write_event_records(
    root: Path,
    analysis_rel: str,
    sims_rel: str,
    event_id: str,
    event_iloc: int,
    arms: tuple[str, ...],
    stores: dict[str, Path],
    network: Network,
    dims: Dimensions,
    *,
    model_log_dir: Path,
    log_analysis_id: str = "",
) -> None:
    """Logs, prepared inputs and the analysis-level run logs of one event.

    ``sims_rel`` is the ``sims`` directory relative to the golden root; ``model_log_dir`` is the directory
    holding the analysis-level run logs (``{analysis_dir}/logs/sims``). ``log_analysis_id`` is put in the run-log
    file names of a sensitivity member (SPEC.md 4.7) and is empty otherwise. ``analysis_rel`` is not used by this
    function. The store-name prefixes ``TRITONSWMM_``, ``TRITON_only_`` and ``SWMM_only_`` are those of the
    SPEC.md 4.3 table."""
    event_rel = f"{sims_rel}/{event_id}"
    event_dir = root / event_rel
    write_json(event_dir / "scenario_prep_log.json", scenario_log(event_rel, event_iloc, event_id))
    for arm in arms:
        m = ARM_MODEL[arm]
        prefix = {"tritonswmm": "TRITONSWMM_", "triton": "TRITON_only_", "swmm": "SWMM_only_"}[arm]
        own = {n: p for n, p in stores.items() if n.startswith(prefix)}
        write_json(event_dir / f"log_{m}.json", model_log(event_rel, event_iloc, m, event_dir, own))
        # The run-log marker lines hhemt looks for (SPEC.md 4.7): ``Simulation ends`` for triton and tritonswmm,
        # ``EPA SWMM completed`` for swmm.
        marker = "EPA SWMM completed in 0.01 seconds.\n" if m == "swmm" else "Simulation ends\n"
        tag = f"{log_analysis_id}_" if log_analysis_id else ""
        write_text(model_log_dir / f"model_{m}_{tag}evt-{event_id}.log", marker)
    write_done_flags(event_dir, arms)
    # prepared inputs (SPEC.md 4.8): [CHOICE] minimal contents
    for name in ("hydro", "hydraulics", "full"):
        write_text(event_dir / "swmm" / f"{name}.inp", inp_text(network, dims, template=False))
    write_text(event_dir / "extbc" / "tseries.txt", "0.0 0.0\n")
    write_text(event_dir / "extbc" / "loc.extbc", "0.0 0.0\n")
    write_text(event_dir / "strmflow" / "tseries.hyg", "0.0 0.0\n")
    write_text(event_dir / "strmflow" / "loc.txt", "0.0 0.0\n")
    write_text(event_dir / "dats" / "grid-indRG_synth.dat", "RG_synth 2000 1 1 0 0 0.0\n")
    write_text(event_dir / "dats" / "waterlevel.dat", "2000 1 1 0 0 0.0\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wx = _weather_dataset([event_iloc], dims).isel(event_index=0, drop=True)
        wx.to_netcdf(event_dir / "sim_weather.nc")
    if "triton" in arms:
        write_text(event_dir / "TRITON.cfg", _cfg_text("TRITON", event_rel, dims))
    if "tritonswmm" in arms:
        write_text(event_dir / "TRITONSWMM.cfg", _cfg_text("TRITONSWMM", event_rel, dims))
