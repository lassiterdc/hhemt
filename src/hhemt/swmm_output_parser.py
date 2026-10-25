import io
import re
import warnings
from collections import Counter
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple, cast

import numpy as np
import pandas as pd
import xarray as xr

try:  # pragma: no cover - used by line_profiler
    profile  # type: ignore[name-defined]  # noqa: B018 - probing for line_profiler's injected global; the bare name IS the check
except NameError:  # pragma: no cover

    def profile(func):  # type: ignore[no-redef]
        return func


from hhemt.constants import (
    LST_COL_HEADERS_LINK_FLOW_SUMMARY,
    LST_COL_HEADERS_NODE_FLOOD_SUMMARY,
    LST_COL_HEADERS_NODE_FLOW_SUMMARY,
)

TDELTA_PATTERN = re.compile(r"^\s*(\d+)\s+(\d+):(\d+)")
RPT_DATETIME_FORMAT = "%m/%d/%Y %H:%M:%S"

_ADJACENT_VALUE_SOLUTION = (
    "Two values in the rpt were right next to each other and couldn't be parsed "
    "using spacing. Parsing by referencing a normal line."
)

# Bulk-parse regex patterns for parse_rpt_single_pass.
# Each <<< Node|Link {id} >>> block has the structure:
#   <<< (Node|Link) {id} >>>
#   ------------ (separator 1)
#   {column-name lines}
#   ------------ (separator 2)
#   {numeric rows: " MM/DD/YYYY HH:MM:SS  v1  v2  v3  v4"}
#   (blank line, blank line)
# The numeric block ends at the next <<< marker or two consecutive blank lines.
_RE_TSERIES_BLOCK = re.compile(
    r"<<<\s+(?P<kind>Node|Link)\s+(?P<id>\S+)\s+>>>"
    r".*?"  # column-name lines + first separator (non-greedy; DOTALL spans newlines)
    r"^\s*-{30,}\s*$"  # second separator (anchored to line)
    # Data iterations use `[^\n]*` (not `.*`) so each iteration captures ONE line
    # even under re.DOTALL — otherwise `.*` would greedily span the whole section.
    r"(?P<data>(?:\n[ \t]+\d{1,2}/\d{1,2}/\d{4}[ \t]+\d{1,2}:\d{2}:\d{2}[^\n]*)+)",
    re.DOTALL | re.MULTILINE,
)
_RE_NODE_TSERIES_SECTION = re.compile(r"Node Time Series Results")
_RE_LINK_TSERIES_SECTION = re.compile(r"Link Time Series Results")


def retrieve_swmm_performance_stats_from_rpt(
    report_file_path: Path | None,
) -> dict[str, Any]:
    """Extract performance/resource metadata from a SWMM ``.rpt`` file.

    Parameters
    ----------
    report_file_path : Path | None
        Path to SWMM report file.

    Returns
    -------
    dict[str, Any]
        Dictionary containing:
        - wall_time_s: float | None
        - actual_omp_threads: int | None
    """
    result: dict[str, Any] = {
        "wall_time_s": None,
        "actual_omp_threads": None,
    }

    if report_file_path is None or not report_file_path.exists():
        result["wall_time_s"] = "rpt does not exist"
        result["actual_omp_threads"] = "rpt does not exist"
        return result

    decode_errors: list[str] = []
    content: str | None = None
    attempted_encodings = ("utf-8", "cp1252")

    for encoding in attempted_encodings:
        try:
            with open(report_file_path, encoding=encoding) as f:
                content = f.read()
            break
        except UnicodeDecodeError as e:
            decode_errors.append(f"{encoding}: {e}")

    if content is None:
        attempted = ", ".join(attempted_encodings)
        details = " | ".join(decode_errors)
        raise UnicodeError(
            f"Failed to decode SWMM report file {report_file_path} using encodings [{attempted}]. Errors: {details}"
        )

    match_hms = re.search(r"Total elapsed time:\s*(\d{1,2}):(\d{2}):(\d{2})", content)
    match_lt_1s = re.search(r"Total elapsed time:\s*<\s*1\s*sec", content)
    if match_hms is not None:
        hours = int(match_hms.group(1))
        minutes = int(match_hms.group(2))
        seconds = int(match_hms.group(3))
        result["wall_time_s"] = float(hours * 3600 + minutes * 60 + seconds)
    elif match_lt_1s is not None:
        result["wall_time_s"] = 1.0
    else:
        raise ValueError(f"Could not find supported elapsed-time format in SWMM report file {report_file_path}")

    thread_match = re.search(
        r"Number of Threads\s*\.{2,}\s*(\d+)",
        content,
        flags=re.IGNORECASE,
    )
    if thread_match is not None:
        result["actual_omp_threads"] = int(thread_match.group(1))

    return result


def retrieve_SWMM_outputs_as_datasets(
    f_swmm_inp: Path,
    swmm_timeseries_result_file: Path,
):
    # TODO - reindex nodes and links by entity type; this will likely resolve parsing errors as well
    # TODO - record warnings as attributes in dataset and in sim processing log
    ds_nodes, ds_links = return_swmm_outputs(
        f_swmm_inp,
        swmm_timeseries_result_file,
        LST_COL_HEADERS_NODE_FLOOD_SUMMARY,
        LST_COL_HEADERS_NODE_FLOW_SUMMARY,
        LST_COL_HEADERS_LINK_FLOW_SUMMARY,
    )

    return ds_nodes, ds_links


def return_swmm_outputs(
    f_swmm_inp: Path,
    swmm_timeseries_result_file: Path,
    lst_col_headers_node_flood_summary,
    lst_col_headers_node_flow_summary,
    lst_col_headers_link_flow_summary,
):
    dict_system_results = {}
    lst_node_fld_summary = []
    lst_node_flow_summary = []
    lst_link_flow_summary = []
    if swmm_timeseries_result_file.name.split(".")[-1] == "rpt":
        use_rpt_for_tseries = True
    elif swmm_timeseries_result_file.name.split(".")[-1] == "out":
        use_rpt_for_tseries = False
    else:
        raise ValueError(
            f"SWMM output file not recognized while parsing time series. File passed: {swmm_timeseries_result_file}"
        )

    quality_vars: dict[str, xr.DataArray] = {}
    pollutants: list[tuple[str, str]] = []
    simulation_start: str | None = None
    if use_rpt_for_tseries:
        (
            dict_section_lines,
            ds_node_tseries,
            ds_link_tseries,
            dict_system_results,
            valid,
        ) = parse_rpt_single_pass(swmm_timeseries_result_file)
        # The coupled model's report is read as m3/s with no conversion, and SWMM writes its
        # flows in the model's own unit, so a report in any other unit is refused here --
        # the guard for reports the preparation check never saw, such as a reprocess of an
        # existing analysis. Deliberately NOT in `_build_system_results`, which the
        # hydrology report summary also reaches, and the hydrology model accepts LPS and
        # MLD because its flows are converted.
        if dict_system_results["flow_units"] != "cms":
            from hhemt.exceptions import ProcessingError

            raise ProcessingError(
                operation="return_swmm_outputs",
                filepath=Path(swmm_timeseries_result_file),
                reason=(
                    f"SWMM report flow units are {dict_system_results['flow_units'].upper()!r}; "
                    "the toolkit reads the coupled model's report flows as m3/s without "
                    "conversion, so the hydraulics model's [OPTIONS] FLOW_UNITS must be CMS. "
                    "Fix it in the SWMM template named by system_config."
                ),
            )
        if valid is False:
            print(f"The RPT file seems to not contain any information: {swmm_timeseries_result_file}")
        # make sure the data arrays are data type float
        ds_node_tseries = convert_datavars_to_dtype(ds_node_tseries, lst_dtypes_to_try=[float])
        ds_link_tseries = convert_datavars_to_dtype(ds_link_tseries, lst_dtypes_to_try=[float])
        # make sure the coordinates are the right data type
        ds_node_tseries = convert_coords_to_dtype(
            ds_node_tseries,
            lst_dtypes_to_try=[str],
            coords_to_coerce=["node_id", "link_id"],
        )
        ds_link_tseries = convert_coords_to_dtype(
            ds_link_tseries,
            lst_dtypes_to_try=[str],
            coords_to_coerce=["node_id", "link_id"],
        )
        # Water quality: only when the report itself says quality was routed.
        if rpt_prints_water_quality(swmm_timeseries_result_file):
            pollutants = _declared_pollutants(f_swmm_inp)
            da_node_conc, da_link_conc = parse_rpt_pollutant_tseries(swmm_timeseries_result_file, pollutants)
            if da_node_conc is not None:
                quality_vars["pollutant_conc"] = da_node_conc
            if da_link_conc is not None:
                quality_vars["link_pollutant_conc"] = da_link_conc
            rpt_content = Path(swmm_timeseries_result_file).read_text(encoding="latin-1")
            simulation_start = _rpt_starting_date(rpt_content)
            quality_vars.update(
                parse_rpt_quality_summaries(
                    rpt_content.splitlines(),
                    pollutants,
                    include_runoff_quality=False,
                    filepath=swmm_timeseries_result_file,
                )
            )
            del rpt_content
    else:
        ds_node_tseries, ds_link_tseries, pollutants, out_start = return_node_time_series_results_from_outfile(
            swmm_timeseries_result_file, f_inp=f_swmm_inp, return_quality=True
        )
        if pollutants:
            simulation_start = out_start
            quality_vars["pollutant_conc"] = ds_node_tseries["pollutant_conc"]
            quality_vars["link_pollutant_conc"] = ds_link_tseries["link_pollutant_conc"]
            ds_node_tseries = ds_node_tseries.drop_vars("pollutant_conc")
            ds_link_tseries = ds_link_tseries.drop_vars("link_pollutant_conc")
        rpt_file = swmm_timeseries_result_file.with_suffix(".rpt")
        if rpt_file.exists():
            try:
                with open(rpt_file, encoding="latin-1") as file:
                    rpt_lines = file.readlines()
                if pollutants and lines_print_water_quality(rpt_lines):
                    quality_vars.update(
                        parse_rpt_quality_summaries(
                            rpt_lines, pollutants, include_runoff_quality=True, filepath=rpt_file
                        )
                    )
                lst_node_fld_summary = return_lines_for_section_of_rpt("Node Flooding Summary", lines=rpt_lines)
                lst_node_flow_summary = return_lines_for_section_of_rpt("Node Inflow Summary", lines=rpt_lines)
                lst_link_flow_summary = return_lines_for_section_of_rpt("Link Flow Summary", lines=rpt_lines)
                dict_system_results = return_swmm_system_outputs(rpt_lines)
            except Exception as exc:
                warnings.warn(
                    f"Failed to parse SWMM RPT summaries from {rpt_file}: {exc}",
                    UserWarning,
                    stacklevel=2,
                )
    #
    if use_rpt_for_tseries:
        lst_node_fld_summary = dict_section_lines.get("Node Flooding Summary", [])
        lst_node_flow_summary = dict_section_lines.get("Node Inflow Summary", [])
        lst_link_flow_summary = dict_section_lines.get("Link Flow Summary", [])
    df_node_flood_summary = format_rpt_section_into_dataframe(lst_node_fld_summary, lst_col_headers_node_flood_summary)
    df_node_flow_summary = format_rpt_section_into_dataframe(lst_node_flow_summary, lst_col_headers_node_flow_summary)
    df_link_flow_summary = format_rpt_section_into_dataframe(lst_link_flow_summary, lst_col_headers_link_flow_summary)
    # combine event summary dataframes into an xarray dataset
    df_node_flood_summary.set_index("node_id", inplace=True)
    df_node_flow_summary.set_index("node_id", inplace=True)
    #
    df_node_summaries = df_node_flood_summary.join(df_node_flow_summary, how="outer")
    # process time stuff
    df_node_summaries["time_of_max_flood_min"] = convert_swmm_tdeltas_to_minutes(
        df_node_summaries["time_of_max_flood_d_hr_mn"]
    )
    df_node_summaries["time_of_max_flow_min"] = convert_swmm_tdeltas_to_minutes(
        df_node_summaries["time_of_max_flow_d_hr_mn"]
    )
    #
    df_node_summaries = df_node_summaries.drop(columns=["time_of_max_flood_d_hr_mn", "time_of_max_flow_d_hr_mn"])
    # remove spaces from node id column
    lst_node_ids = []
    for val in df_node_summaries.index.values:
        lst_val_substrings = val.split(" ")
        for substring in lst_val_substrings:
            if len(substring) > 0:
                lst_node_ids.append(substring)
                break
    df_node_summaries.index = lst_node_ids
    df_node_summaries.index.name = "node_id"
    # process link stuff
    df_link_flow_summary["time_of_max_flow_min"] = convert_swmm_tdeltas_to_minutes(
        df_link_flow_summary["time_of_max_flow_d_hr_mn"]
    )
    df_link_flow_summary = df_link_flow_summary.drop(columns=["time_of_max_flow_d_hr_mn"])

    def _clean_link_id(link_id):
        """Clean a single link_id value."""
        try:
            return str(int(float(link_id)))
        except (ValueError, TypeError):
            if isinstance(link_id, str):
                parts = link_id.split()
                return parts[0] if parts else link_id
            return str(link_id)

    df_link_flow_summary["link_id"] = df_link_flow_summary["link_id"].apply(_clean_link_id)
    df_link_flow_summary.set_index("link_id", inplace=True)

    ds_node_summaries = df_node_summaries.to_xarray()
    ds_link_flow_summary = df_link_flow_summary.to_xarray()
    #
    ds_node_summaries = convert_coords_to_dtype(
        ds_node_summaries,
        lst_dtypes_to_try=[str],
        coords_to_coerce=["node_id", "link_id"],
    )
    ds_node_summaries = convert_datavars_to_dtype(ds_node_summaries, lst_dtypes_to_try=[float, str])
    ds_link_flow_summary = convert_coords_to_dtype(
        ds_link_flow_summary,
        lst_dtypes_to_try=[str],
        coords_to_coerce=["node_id", "link_id"],
    )
    ds_link_flow_summary = convert_datavars_to_dtype(ds_link_flow_summary, lst_dtypes_to_try=[float, str])

    ds_nodes = xr.merge([ds_node_summaries, ds_node_tseries], join="outer")
    ds_links = xr.merge([ds_link_flow_summary, ds_link_tseries], join="outer")
    #
    if quality_vars:
        link_only = ("link_pollutant_conc", "link_total_load")
        node_vars = {k: v for k, v in quality_vars.items() if k not in link_only and k != "washoff_total_load"}
        link_vars = {k: v for k, v in quality_vars.items() if k in link_only}
        ds_nodes = xr.merge([ds_nodes, xr.Dataset(node_vars)], join="outer")
        if "washoff_total_load" in quality_vars:
            ds_nodes = xr.merge([ds_nodes, xr.Dataset({"washoff_total_load": quality_vars["washoff_total_load"]})])
        ds_links = xr.merge([ds_links, xr.Dataset(link_vars)], join="outer")
        ds_nodes = _attach_pollutant_coords(ds_nodes, pollutants)
        ds_links = _attach_pollutant_coords(ds_links, pollutants)
    #
    ds_nodes.attrs = dict_system_results
    ds_links.attrs = dict_system_results
    if "pollutant_conc" in ds_nodes.data_vars:
        # The peak's time is measured here, where the simulation start is known, and stored as the declared
        # variable it is. The start itself is NOT stored: a dataset attribute carrying it would be persisted
        # in every quality-routed node store as a key the output data model does not declare.
        ds_nodes["pollutant_time_of_max_conc_min"] = pollutant_time_of_max_conc_min(
            ds_nodes["pollutant_conc"], simulation_start
        )
    #
    return ds_nodes, ds_links


# --- Water quality (SWMM pollutants) ---------------------------------------------------------
#
# Everything below is written ONLY for a model whose report prints `Water Quality .......... YES`
# (SWMM prints NO whenever `IGNORE_QUALITY` is on or no pollutant is declared). On the `.out` path
# the same predicate is the `.out`'s own pollutant count being above zero (`output.c`: it is zero
# when quality is ignored). For every other model none of these names, coordinates, dimensions or
# attribute keys exist, so its processed outputs are byte-identical to what they were.

_RE_WATER_QUALITY = re.compile(r"^\s*Water Quality\s*\.+\s*(YES|NO)\s*$", re.MULTILINE)
_RE_START_DATE = re.compile(r"^\s*Starting Date\s*\.+\s*(\S+\s+\S+)\s*$", re.MULTILINE)
_RE_CONTINUITY_ROW = re.compile(r"^\s*(?P<label>[A-Za-z][^.]*?)\s*\.{2,}\s*(?P<vals>[-+0-9.eE\s]+?)\s*$")

#: `Quality Routing Continuity` row labels to the variable-name stem each is stored under.
_QUAL_ROUTING_TERMS = {
    "Dry Weather Inflow": "dry_weather_inflow",
    "Wet Weather Inflow": "wet_weather_inflow",
    "Groundwater Inflow": "groundwater_inflow",
    "RDII Inflow": "rdii_inflow",
    "External Inflow": "external_inflow",
    "External Outflow": "external_outflow",
    "Flooding Loss": "flooding_loss",
    "Exfiltration Loss": "exfiltration_loss",
    "Mass Reacted": "mass_reacted",
    "Initial Stored Mass": "initial_stored_mass",
    "Final Stored Mass": "final_stored_mass",
}
#: `Runoff Quality Continuity` row labels to the variable-name stem each is stored under.
_RUNOFF_QUAL_TERMS = {
    "Initial Buildup": "initial_buildup",
    "Surface Buildup": "surface_buildup",
    "Wet Deposition": "wet_deposition",
    "Sweeping Removal": "sweeping_removal",
    "Infiltration Loss": "infiltration_loss",
    "BMP Removal": "bmp_removal",
    "Surface Runoff": "surface_runoff",
    "Remaining Buildup": "remaining_buildup",
}


def lines_print_water_quality(lines) -> bool:
    """Whether a report's text (any iterable of lines) prints ``Water Quality .......... YES``."""
    for line in lines:
        match = _RE_WATER_QUALITY.match(line.rstrip("\r\n"))
        if match is not None:
            return match.group(1) == "YES"
    return False


def rpt_prints_water_quality(f_rpt: Path) -> bool:
    """Whether the report file prints ``Water Quality .......... YES``, reading only as far as needed.

    SWMM writes that line in the options block near the top of the report, so the scan stops at
    the first match; a multi-gigabyte coupled report is not read in full to answer this. A file
    that carries no such line (empty, truncated before the options block) answers False.
    """
    with open(f_rpt, encoding="latin-1") as fh:
        for line in fh:
            match = _RE_WATER_QUALITY.match(line.rstrip("\r\n"))
            if match is not None:
                return match.group(1) == "YES"
    return False


def _declared_pollutants(f_inp):
    """The model's pollutants as ``[(name, unit)]`` in ``[POLLUTANTS]`` order, refusing a model with none.

    A report that prints ``Water Quality YES`` without the model file declaring a pollutant cannot
    happen for a consistent pair, so an empty list is a mismatched pair and is refused rather than
    silently producing no pollutant variables.
    """
    from hhemt.exceptions import ProcessingError
    from hhemt.scenario import ordered_inp_pollutants

    pollutants = ordered_inp_pollutants(f_inp)
    if not pollutants:
        raise ProcessingError(
            operation="return_swmm_outputs",
            filepath=Path(f_inp),
            reason="the SWMM report says water quality was routed but the model file declares no [POLLUTANTS].",
        )
    return pollutants


def _rpt_starting_date(text: str) -> str | None:
    """The report's ``Starting Date`` value as ``MM/DD/YYYY HH:MM:SS`` text, or None when absent."""
    match = _RE_START_DATE.search(text)
    return match.group(1) if match else None


def pollutant_time_of_max_conc_min(conc, simulation_start, tstep_dimname="date_time"):
    """Minutes from the simulation start to the first step at which each series reaches its maximum.

    ``conc`` is a concentration DataArray whose last-but-one axis is the time dimension; the result drops
    that dimension. ``simulation_start`` is the report's ``Starting Date`` text (``MM/DD/YYYY HH:MM:SS``);
    when it is None the origin is the first reported step. The unit and origin match ``time_of_max_flow_min``.
    The first occurrence of a tied maximum is used; a series with no value gives NaN.
    """
    times = conc[tstep_dimname].values
    origin = (
        pd.to_datetime(simulation_start, format=RPT_DATETIME_FORMAT) if simulation_start else pd.Timestamp(times.min())
    )
    first = conc.fillna(-np.inf).argmax(dim=tstep_dimname)
    when = times[first.values]
    minutes = (when - np.datetime64(origin)) / np.timedelta64(1, "m")
    has_value = conc.notnull().any(dim=tstep_dimname).values
    return xr.DataArray(
        np.where(has_value, minutes, np.nan),
        dims=first.dims,
        coords={d: first.coords[d] for d in first.dims if d in first.coords},
    )


def _attach_pollutant_coords(ds, pollutants):
    """Return ``ds`` with the ``pollutant`` and ``pollutant_conc_units`` coordinates along ``pollutant``."""
    names = [name for name, _ in pollutants]
    units = [unit for _, unit in pollutants]
    return ds.assign_coords(
        pollutant=("pollutant", names),
        pollutant_conc_units=("pollutant", units),
    )


def _linearize_mass(values, pollutants):
    """Undo SWMM's ``LOG10`` on the mass values of a count (``#/L``) pollutant.

    ``values`` has the pollutant as its LAST axis. SWMM prints every mass value of a count
    pollutant as ``LOG10(x)`` and its macro returns ``x`` unchanged for ``x <= 0``, so a printed
    ``0.000`` means zero (or one organism, which is negligible) and maps to 0; any other printed
    value ``x`` is ``10**x``. Pollutants in a mass unit are returned as printed.
    """
    out = np.array(values, dtype=float, copy=True)
    for k, (_, unit) in enumerate(pollutants):
        if unit == "#/L":
            col = out[..., k]
            out[..., k] = np.where(col == 0.0, 0.0, 10.0**col)
    return out


def _rpt_table_rows(lines, title, *, until_dash):
    """Token lists of the data rows of the pollutant table titled ``title``, or None when absent.

    The table's rows sit after its SECOND dashed line. They end at the next dashed line when
    ``until_dash`` (outfall loading and subcatchment washoff, which close with a dashed line
    and a ``System`` row) or at the first blank line otherwise (link pollutant loads).
    """
    start = next((i for i, line in enumerate(lines) if title in line), None)
    if start is None:
        return None
    dashes = 0
    rows = []
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if stripped.startswith("---"):
            dashes += 1
            if dashes > 2 and until_dash:
                break
        elif dashes >= 2:
            if not stripped:
                break
            rows.append(stripped.split())
    return rows


def _rpt_quality_continuity(lines, table_header, n_pollutants, filepath):
    """Rows of one quality continuity table as ``{label: [value per pollutant]}``, scoped by header.

    SWMM prints the table in groups of at most five pollutants, repeating the header for each
    group, so the values of every group are concatenated in print order, which is the
    ``[POLLUTANTS]`` order. A value count different from ``n_pollutants`` is refused.
    """
    rows: dict[str, list[float]] = {}
    current = None
    for line in lines:
        opened = _continuity_table_opened_by(line)
        if opened is not None:
            current = opened
        if current != table_header:
            continue
        match = _RE_CONTINUITY_ROW.match(line.rstrip("\r\n"))
        if match is not None:
            rows.setdefault(match.group("label").strip(), []).extend(float(v) for v in match.group("vals").split())
    for label, vals in rows.items():
        if len(vals) != n_pollutants:
            from hhemt.exceptions import ProcessingError

            raise ProcessingError(
                operation="parse_rpt_quality_summaries",
                filepath=Path(filepath),
                reason=(
                    f"{table_header} row {label!r} carries {len(vals)} value(s) but the model declares "
                    f"{n_pollutants} pollutant(s)."
                ),
            )
    return rows


def _refuse_row_width(filepath, title, n_found, n_expected):
    """Raise the ProcessingError for a pollutant table row whose field count is wrong."""
    from hhemt.exceptions import ProcessingError

    raise ProcessingError(
        operation="parse_rpt_quality_summaries",
        filepath=Path(filepath),
        reason=f"a {title} row has {n_found} field(s); {n_expected} were expected for the declared pollutants.",
    )


def parse_rpt_quality_summaries(lines, pollutants, *, include_runoff_quality, filepath="<rpt>"):
    """Read SWMM's water-quality summary tables from report ``lines`` into labelled arrays.

    Returns ``{variable name: xr.DataArray}``, containing only the tables the report prints:

    - ``outfall_total_load`` (node_id, pollutant), ``outfall_total_vol_10e6_ltr`` (node_id) and
      ``outfall_emc`` (node_id, pollutant) from ``Outfall Loading Summary``. The event mean
      concentration is the total load over the total volume in the pollutant's concentration
      unit, and NaN where the volume is zero.
    - ``link_total_load`` (link_id, pollutant) from ``Link Pollutant Load Summary``.
    - ``washoff_total_load`` (subcatchment_id, pollutant) from ``Subcatchment Washoff Summary``.
    - ``qual_routing_{term}`` and ``qual_routing_continuity_error_perc`` (pollutant) from
      ``Quality Routing Continuity``; and, when ``include_runoff_quality``,
      ``runoff_qual_{term}`` and ``runoff_qual_continuity_error_perc`` from ``Runoff Quality
      Continuity``. Each is read from its own table (the continuity tables are header-scoped).

    Every mass value of a count pollutant is linearized (see ``_linearize_mass``); concentrations
    of the outfall table and the percentage errors are not masses and are used as printed.
    """
    names = [name for name, _ in pollutants]
    units = [unit for _, unit in pollutants]
    n = len(names)
    out: dict[str, xr.DataArray] = {}

    def _two_d(index_name, ids, matrix):
        return xr.DataArray(matrix, dims=(index_name, "pollutant"), coords={index_name: list(ids), "pollutant": names})

    rows = _rpt_table_rows(lines, "Outfall Loading Summary", until_dash=True)
    if rows:
        for row in rows:
            if len(row) != 5 + n:
                _refuse_row_width(filepath, "Outfall Loading Summary", len(row), 5 + n)
        ids = [row[0] for row in rows]
        volume = np.array([float(row[4]) for row in rows])
        loads = _linearize_mass(np.array([[float(v) for v in row[5:]] for row in rows]), pollutants)
        out["outfall_total_load"] = _two_d("node_id", ids, loads)
        out["outfall_total_vol_10e6_ltr"] = xr.DataArray(volume, dims=("node_id",), coords={"node_id": ids})
        emc = np.full(loads.shape, np.nan)
        for k, unit in enumerate(units):
            scale = {"mg/L": 1.0, "ug/L": 1000.0}.get(unit)
            with np.errstate(divide="ignore", invalid="ignore"):
                if scale is not None:
                    col = loads[:, k] / volume * scale
                else:
                    col = loads[:, k] / (volume * 1.0e6)
            emc[:, k] = np.where(volume > 0.0, col, np.nan)
        out["outfall_emc"] = _two_d("node_id", ids, emc)

    rows = _rpt_table_rows(lines, "Link Pollutant Load Summary", until_dash=False)
    if rows:
        for row in rows:
            if len(row) != 1 + n:
                _refuse_row_width(filepath, "Link Pollutant Load Summary", len(row), 1 + n)
        out["link_total_load"] = _two_d(
            "link_id",
            [row[0] for row in rows],
            _linearize_mass(np.array([[float(v) for v in row[1:]] for row in rows]), pollutants),
        )

    rows = _rpt_table_rows(lines, "Subcatchment Washoff Summary", until_dash=True)
    if rows:
        for row in rows:
            if len(row) != 1 + n:
                _refuse_row_width(filepath, "Subcatchment Washoff Summary", len(row), 1 + n)
        out["washoff_total_load"] = _two_d(
            "subcatchment_id",
            [row[0] for row in rows],
            _linearize_mass(np.array([[float(v) for v in row[1:]] for row in rows]), pollutants),
        )

    def _continuity(header, prefix, terms):
        table = _rpt_quality_continuity(lines, header, n, filepath)
        for label, stem in terms.items():
            if label in table:
                mass = _linearize_mass(np.array(table[label]).reshape(1, n), pollutants)[0]
                out[f"{prefix}_{stem}"] = xr.DataArray(mass, dims=("pollutant",), coords={"pollutant": names})
        if "Continuity Error (%)" in table:
            out[f"{prefix}_continuity_error_perc"] = xr.DataArray(
                np.array(table["Continuity Error (%)"]), dims=("pollutant",), coords={"pollutant": names}
            )

    _continuity("Quality Routing Continuity", "qual_routing", _QUAL_ROUTING_TERMS)
    if include_runoff_quality:
        _continuity("Runoff Quality Continuity", "runoff_qual", _RUNOFF_QUAL_TERMS)
    return out


def _bulk_parse_pollutant_section(section_text, kind_filter, idx_colname, pollutants, filepath):
    """Pollutant concentration columns of one rpt time-series section, as (idx, date_time, pollutant).

    Every data row of a block must carry ``2 + 4 + n`` fields (date, time, the four hydraulic
    columns, one concentration per declared pollutant); anything else raises a ``ProcessingError``
    naming the file and ``n``. The pollutant names come from the model's ``[POLLUTANTS]`` section,
    never from the report's column headers, which run together for identifiers of ten or more
    characters. Returns None when the section holds no block of ``kind_filter``.
    """
    n = len(pollutants)
    expected = 2 + 4 + n
    ids, row_counts, parts = [], [], []
    for match in _RE_TSERIES_BLOCK.finditer(section_text):
        if match.group("kind") != kind_filter:
            continue
        data_text = match.group("data")
        n_rows = data_text.count("\n")
        if n_rows == 0:
            continue
        for row in data_text.split("\n"):
            if row.strip() and len(row.split()) != expected:
                from hhemt.exceptions import ProcessingError

                raise ProcessingError(
                    operation="parse_rpt_pollutant_tseries",
                    filepath=Path(filepath),
                    reason=(
                        f"a {kind_filter} time-series row carries {len(row.split())} field(s); {expected} "
                        f"were expected for {n} declared pollutant(s) (date, time, 4 hydraulic, {n} concentration)."
                    ),
                )
        ids.append(match.group("id"))
        row_counts.append(n_rows)
        parts.append(data_text)
    if not parts:
        return None
    cols = [f"c{k}" for k in range(n)]
    df = pd.read_csv(
        io.StringIO("".join(parts)),
        sep=r"\s+",
        header=None,
        names=["date", "time", *cols],
        usecols=[0, 1, *range(6, 6 + n)],
        engine="c",
        dtype={"date": str, "time": str},
    )
    df["date_time"] = pd.to_datetime(df["date"] + " " + df["time"], format=RPT_DATETIME_FORMAT, errors="coerce")
    if df["date_time"].isna().any():
        raise ValueError(
            f"Parsed RPT date_time values contained NaT. Verify the RPT datetime format matches {RPT_DATETIME_FORMAT}."
        )
    df[idx_colname] = np.repeat(ids, row_counts)
    ds = df.set_index([idx_colname, "date_time"])[cols].to_xarray()
    da = ds.to_dataarray(dim="pollutant").transpose(idx_colname, "date_time", "pollutant")
    return da.assign_coords(pollutant=[name for name, _ in pollutants]).astype(float)


def parse_rpt_pollutant_tseries(f_rpt: Path, pollutants):
    """Pollutant concentration series of a coupled report: ``(node DataArray, link DataArray)``.

    ``pollutant_conc`` is (node_id, date_time, pollutant) and ``link_pollutant_conc`` is
    (link_id, date_time, pollutant). The hydraulic columns are read by ``_bulk_parse_tseries``
    and are untouched here; this is a second ``read_csv`` over the concentration columns only.
    """
    content = Path(f_rpt).read_text(encoding="latin-1")
    m_node = _RE_NODE_TSERIES_SECTION.search(content)
    m_link = _RE_LINK_TSERIES_SECTION.search(content)
    node_end = m_link.start() if m_link else len(content)
    node_section = content[m_node.end() : node_end] if m_node else ""
    link_section = content[m_link.end() :] if m_link else ""
    da_node = _bulk_parse_pollutant_section(node_section, "Node", "node_id", pollutants, f_rpt)
    da_link = _bulk_parse_pollutant_section(link_section, "Link", "link_id", pollutants, f_rpt)
    if da_node is not None:
        da_node.name = "pollutant_conc"
    if da_link is not None:
        da_link.name = "link_pollutant_conc"
    return da_node, da_link


# The continuity tables SWMM prints in a `.rpt`, by the header line that opens each. A `Continuity Error (%)`
# line and a `Flooding Loss` line mean different things in different tables, so the scans below take a figure
# only from the table that owns it. Taking the LAST such line, as the scans did before, returns the water-quality
# or groundwater table's value whenever a report prints one (SWMM prints them after the quantity tables).
_CONTINUITY_TABLE_HEADERS = (
    "Runoff Quantity Continuity",
    "Runoff Quality Continuity",
    "Groundwater Continuity",
    "Flow Routing Continuity",
    "Quality Routing Continuity",
)


def _continuity_table_opened_by(line):
    """Return the continuity table a header `line` opens, or None when it opens none."""
    for header in _CONTINUITY_TABLE_HEADERS:
        if header in line:
            return header
    return None


def return_swmm_system_outputs(rpt_lines):
    """Scan a full `.rpt` line list for the system-level metadata lines and
    delegate construction to the guarded `_build_system_results`.

    Every line this scan looks for is CONDITIONAL in the SWMM engine, so a
    legitimate `.rpt` may omit any of them:

    - "Flow Routing Continuity" — gated on `Nobjects[NODE] > 0 && !IgnoreRouting`
      (massbal.c:299), so a link-free or `IGNORE_ROUTING YES` model omits it.
    - "Analysis ended on" — written by `swmm_close` on every run that reaches the
      close, a clean finish or a run closed after an input or start error; a run
      killed or aborted before the close omits it.
    - "Flow Units" / "Flooding Loss" — present on any complete `.rpt`
      (report.c:267 is unconditional inside `report_writeOptions()`), but absent
      from a zero-byte or header-truncated file, which `Path.exists()` does not
      screen out.
    - "Continuity Error (%)" VALUE — printed with a fixed `%14.3f` format
      (report.c:570/730), so SWMM's near-zero-balance sentinel
      (`FlowTotals.pctError = TINY` when `|totalInflow - totalOutflow| < 1.0`,
      massbal.c:888) renders as `0.000` and `float(...)` parses it without
      special handling. This single-value read does not aggregate across
      scenarios, so no TINY-sentinel stripping is required.

    Guarding is therefore NOT optional. Do not re-inline the construction here:
    `_build_system_results` is the single guarded builder shared with
    `parse_rpt_single_pass`, and duplicating its body is what left four
    dereferences unguarded (three raising `UnboundLocalError`, one
    `AttributeError`) until 2026-07-15.
    """
    current_continuity_table = None
    line_flw_units = None
    runoff_continuity_error_line = None
    flow_continuity_error_line = None
    system_flood_loss_line = None
    analysis_end_line = None
    for line in rpt_lines:
        if "Flow Units" in line:
            line_flw_units = line
        opened = _continuity_table_opened_by(line)
        if opened is not None:
            current_continuity_table = opened
        if "Continuity Error (%)" in line:
            # Each error belongs to its own table: runoff to the quantity table, flow to the flow table.
            if current_continuity_table == "Runoff Quantity Continuity":
                runoff_continuity_error_line = line
            elif current_continuity_table == "Flow Routing Continuity":
                flow_continuity_error_line = line
        if "Flooding Loss" in line and current_continuity_table == "Flow Routing Continuity":
            system_flood_loss_line = line
        if "Analysis ended on" in line:
            analysis_end_line = line
    return _build_system_results(
        line_flw_units,
        runoff_continuity_error_line,
        flow_continuity_error_line,
        system_flood_loss_line,
        analysis_end_line,
    )


@profile
def parse_rpt_single_pass(f_rpt: Path):
    """
    Parse an RPT file in a single pass, extracting summaries and time series.

    Returns section lines for summary tables, node/link time series datasets,
    system-level outputs, and a validity flag.

    Bulk-parse path: reads the full file once, slices time-series sections via
    pre-compiled regex, and parses numeric rows with `pd.read_csv` (C engine).
    Summary tables remain on the per-line classifier (low-volume, irregular rows
    handled by ``format_rpt_section_into_dataframe``).
    """
    content = Path(f_rpt).read_text(encoding="latin-1")

    summary_section_headers = (
        "Node Flooding Summary",
        "Node Inflow Summary",
        "Link Flow Summary",
    )

    # Single-shot metadata + summary-table extraction over splitlines.
    # The summary-table classifier mirrors the legacy state machine exactly so
    # output equivalence holds for the orifice/irregular-row edge cases.
    lines = content.splitlines(keepends=True)
    (
        dict_section_lines,
        line_flw_units,
        runoff_continuity_error_line,
        flow_continuity_error_line,
        system_flood_loss_line,
        analysis_end_line,
        valid,
    ) = _scan_metadata_and_summaries(lines, summary_section_headers)

    # Bulk tseries extraction via regex slicing + pd.read_csv per entity.
    ds_node_tseries, ds_link_tseries = _bulk_parse_tseries(content)

    dict_system_results = _build_system_results(
        line_flw_units,
        runoff_continuity_error_line,
        flow_continuity_error_line,
        system_flood_loss_line,
        analysis_end_line,
    )
    return (
        dict_section_lines,
        ds_node_tseries,
        ds_link_tseries,
        dict_system_results,
        valid,
    )


def _scan_metadata_and_summaries(lines, summary_section_headers):
    """Single-pass scan for single-shot metadata lines + summary-section row lists.

    Skips the time-series section entirely — that work is delegated to
    ``_bulk_parse_tseries``. The result is identical to the legacy parser's
    summary-table extraction.
    """
    dict_section_lines = {header: [] for header in summary_section_headers}
    valid = False
    current_continuity_table = None
    line_flw_units = None
    runoff_continuity_error_line = None
    flow_continuity_error_line = None
    system_flood_loss_line = None
    analysis_end_line = None
    current_summary_section = None
    summary_begin_header = False
    summary_end_header = False
    tseries_section_started = False

    for line_num, line in enumerate(lines):
        if "Element Count" in line:
            valid = True
        if "Flow Units" in line:
            line_flw_units = line
        opened = _continuity_table_opened_by(line)
        if opened is not None:
            current_continuity_table = opened
        if "Continuity Error (%)" in line:
            if current_continuity_table == "Runoff Quantity Continuity":
                runoff_continuity_error_line = line
            elif current_continuity_table == "Flow Routing Continuity":
                flow_continuity_error_line = line
        if "Flooding Loss" in line and current_continuity_table == "Flow Routing Continuity":
            system_flood_loss_line = line
        if "Analysis ended on" in line:
            analysis_end_line = line

        # Skip summary-section extraction once we enter the time-series section.
        if not tseries_section_started and "Node Time Series Results" in line:
            tseries_section_started = True
        if tseries_section_started:
            continue

        for header in summary_section_headers:
            if header in line:
                current_summary_section = header
                summary_begin_header = False
                summary_end_header = False
                break
        if current_summary_section is None:
            continue
        if "No nodes were flooded." in line:
            current_summary_section = None
            continue
        if "------------" in line:
            if not summary_begin_header:
                summary_begin_header = True
                continue
            if not summary_end_header:
                summary_end_header = True
                continue
        if not (summary_begin_header and summary_end_header):
            continue
        if line_num == 0:
            continue
        if len(line.split(" ")) <= 3:
            current_summary_section = None
            continue
        dict_section_lines[current_summary_section].append(line)

    return (
        dict_section_lines,
        line_flw_units,
        runoff_continuity_error_line,
        flow_continuity_error_line,
        system_flood_loss_line,
        analysis_end_line,
        valid,
    )


def _bulk_parse_tseries(content: str):
    """Slice node/link time-series blocks via regex and parse via a single pd.read_csv.

    Replaces the per-line state machine + per-row Python split that previously
    drove the time-series ingest. All per-section numeric rows are concatenated
    and parsed in ONE `pd.read_csv` (C engine) call to amortize fixed setup cost.
    Entity ids are reconstructed via `np.repeat` against per-block row counts.
    """
    m_node = _RE_NODE_TSERIES_SECTION.search(content)
    m_link = _RE_LINK_TSERIES_SECTION.search(content)

    node_section_end = m_link.start() if m_link else len(content)
    node_section = content[m_node.end() : node_section_end] if m_node else ""
    link_section = content[m_link.end() :] if m_link else ""

    ds_node_tseries = _bulk_parse_section(
        node_section,
        kind_filter="Node",
        idx_colname="node_id",
        value_cols=["inflow_flow_cms", "flooding_cms", "depth_m", "head_m"],
    )
    ds_link_tseries = _bulk_parse_section(
        link_section,
        kind_filter="Link",
        idx_colname="link_id",
        value_cols=["flow_cms", "velocity_mps", "link_depth_m", "capacity_setting"],
    )
    return ds_node_tseries, ds_link_tseries


def _bulk_parse_section(section_text, kind_filter, idx_colname, value_cols):
    """Single-pass: collect all matching blocks' data into one buffer + parse once.

    Returns an xarray Dataset with dims (idx_colname, date_time) — the same shape
    the legacy `create_tseries_ds` produced for time-series blocks.

    Only the date, the time and the declared `value_cols` are read. When a model
    declares pollutants and routes quality, SWMM appends one concentration column
    per pollutant to every row; those extra columns are not read, because with the
    fixed `names=` below they would shift the fields and break the date parse.
    """
    ids = []
    row_counts = []
    parts = []
    for match in _RE_TSERIES_BLOCK.finditer(section_text):
        if match.group("kind") != kind_filter:
            continue
        data_text = match.group("data")
        # The regex's data group starts with `\n` and each iteration begins with
        # `\n` + one row, so a leading newline always exists. count("\n") equals
        # the number of rows captured.
        n_rows = data_text.count("\n")
        if n_rows == 0:
            continue
        ids.append(match.group("id"))
        row_counts.append(n_rows)
        parts.append(data_text)

    if not parts:
        empty_df = pd.DataFrame(columns=[idx_colname, "date_time", *value_cols])
        empty_df["date_time"] = pd.to_datetime([], format=RPT_DATETIME_FORMAT)
        empty_df = empty_df.set_index([idx_colname, "date_time"])
        return empty_df.to_xarray()

    buf = io.StringIO("".join(parts))
    df = pd.read_csv(
        buf,
        sep=r"\s+",
        header=None,
        names=["date", "time", *value_cols],
        usecols=list(range(2 + len(value_cols))),
        engine="c",
        dtype={"date": str, "time": str},
    )
    df["date_time"] = pd.to_datetime(df["date"] + " " + df["time"], format=RPT_DATETIME_FORMAT, errors="coerce")
    if df["date_time"].isna().any():
        raise ValueError(
            f"Parsed RPT date_time values contained NaT. Verify the RPT datetime format matches {RPT_DATETIME_FORMAT}."
        )
    df = df.drop(columns=["date", "time"])
    df[idx_colname] = np.repeat(ids, row_counts)
    df = df.set_index([idx_colname, "date_time"])
    df = df[value_cols]
    return df.to_xarray()


def _iter_rpt_lines(f_rpt: Path):
    """Yield lines from an RPT file."""
    with open(f_rpt, encoding="latin-1") as file:
        yield from file


def rpt_is_complete(f_rpt: Path) -> bool:
    """Return True iff an SWMM `.rpt` exists, is non-empty, carries the trailer and carries no SWMM error line.

    SWMM writes the ``Analysis ended on`` trailer from ``swmm_close`` (through
    ``report_writeSysTime``) on EVERY run that reaches the close: a clean finish, and
    also a run closed after an input or start error, whose report carries an
    ``ERROR`` block beside the trailer. So a finalized report carrying an ``ERROR``
    block is NOT complete, and the trailer alone is not a success signal. The trailer
    is absent only from a run killed or aborted before the close (e.g. a
    walltime-killed hotstart attempt); SWMM has no report-resume or report-append
    mechanism. ``Flow Units`` and ``Element Count`` are written near ``swmm_start``
    and can survive in a truncated `.rpt`; the trailer does not survive a kill
    before the close.

    Consumed by coupled-sim completion detection to reject a false-positive
    "completed" marker on a resumed run whose embedded SWMM engine never emitted
    its report. Do NOT loosen the summary parser to tolerate an incomplete `.rpt`
    — a SWMM-less resume would invalidate any clean-vs-resume comparison.
    """
    f_rpt = Path(f_rpt)
    if not f_rpt.exists() or f_rpt.stat().st_size == 0:
        return False
    _st = rpt_status(f_rpt)
    return _st.finalized and not _st.has_errors


class RptStatus(NamedTuple):
    """Both facts a SWMM `.rpt` can testify to, read in ONE pass.

    `finalized` is the terminal `Analysis ended on` trailer -- the engine reached
    `swmm_close`, after a clean finish or after an error. `has_errors` is an
    `  ERROR {n}:` block -- the run reported a fatal input or runtime error.

    THEY ARE INDEPENDENT, and that is the whole reason this returns a pair rather
    than a bool. A run that aborts during input validation writes the trailer AND
    an error block: the measured 813-byte damaged rpt carries `Analysis ended on`
    beside seven `ERROR 361: could not open external file`. So `finalized` alone
    cannot distinguish a clean run from a failed one, which is why a caller
    adjudicating COMPLETION must require `finalized and not has_errors`.

    Errors are matched anchored (`\\s*ERROR \\d+`) because SWMM writes warnings
    under a separate WARNING token and the continuity tables say
    `Continuity Error (%)` in mixed case -- an unanchored `ERROR` search would
    match neither, but an anchored one cannot match either by accident.
    """

    finalized: bool
    has_errors: bool


def rpt_status(f_rpt: Path) -> RptStatus:
    """Read a SWMM `.rpt` ONCE and report both the trailer and the error block.

    Single owner of the `Analysis ended on` literal, which was previously stated
    here and in the summary parser's `analysis_end_line`. A missing or empty file
    testifies to nothing: `RptStatus(False, False)`.

    Reads to EOF rather than short-circuiting on the first error, because the
    trailer is at the END and both facts are wanted. That costs nothing on the
    population that matters: an errored rpt is ~800 bytes (the run aborted before
    writing a body), while a valid one is ~1 MB and has no error line to exit on.
    """
    f_rpt = Path(f_rpt)
    if not f_rpt.exists() or f_rpt.stat().st_size == 0:
        return RptStatus(False, False)
    finalized = False
    has_errors = False
    with open(f_rpt, encoding="latin-1") as fh:
        for line in fh:
            if not finalized and "Analysis ended on" in line:
                finalized = True
            elif not has_errors and re.match(r"\s*ERROR \d+", line):
                has_errors = True
    return RptStatus(finalized, has_errors)


def _build_system_results(
    line_flw_units,
    runoff_continuity_error_line,
    flow_continuity_error_line,
    system_flood_loss_line,
    analysis_end_line,
):
    """Build system-level result attributes from parsed RPT metadata."""
    if line_flw_units is None:
        raise ValueError("Flow units line not found in RPT file.")
    if flow_continuity_error_line is None:
        raise ValueError("Flow continuity error line not found in RPT file.")
    if system_flood_loss_line is None:
        raise ValueError("System flooding loss line not found in RPT file.")
    if analysis_end_line is None:
        raise ValueError("Analysis end line not found in RPT file.")
    flow_units = line_flw_units.split(".")[-1].split("\n")[0].split(" ")[-1].lower()

    if runoff_continuity_error_line is not None:
        runoff_continuity_error_perc = float(runoff_continuity_error_line.split(" ")[-1].split("\n")[0])
    else:
        runoff_continuity_error_perc = np.nan
    flow_continuity_error_perc = float(flow_continuity_error_line.split(" ")[-1].split("\n")[0])
    system_flooding = float(system_flood_loss_line.split(" ")[-1].split("\n")[0])
    analysis_end_datetime = _parse_analysis_end_line(analysis_end_line)
    dict_system_results = dict(
        analysis_end_datetime=str(analysis_end_datetime),
        flow_units=flow_units,
        runoff_continuity_error_perc=runoff_continuity_error_perc,
        flow_continuity_error_perc=flow_continuity_error_perc,
        system_flooding_10e6_ltr=system_flooding,
    )
    return dict_system_results


#: Column headers for the Subcatchment Runoff Summary section, in file order.
#: MEASURED against a real Norfolk hydrology rpt (2026-08-17): the section header
#: spans three physical lines and the data rows are fixed-width, so this tuple is
#: authored against a real file, never inferred from the header text.
LST_COL_HEADERS_SUBCATCH_RUNOFF_SUMMARY = (
    "subcatchment_id",
    "total_precip_mm",
    "total_runon_mm",
    "total_evap_mm",
    "total_infil_mm",
    "imperv_runoff_mm",
    "perv_runoff_mm",
    "total_runoff_mm",
    "total_runoff_10e6_ltr",
    "peak_runoff_cms",
    "runoff_coeff",
)


def parse_hydrology_rpt_summary(f_rpt: Path, f_inp: Path | None = None):
    """Capture everything a hydrology-only ``.rpt`` carries that nothing else does.

    WHY THIS EXISTS, and why it is not a duplicate of `_build_system_results`.
    A hydrology-only rpt has NO `Node Time Series Results` section, so the
    reclaim truncator is a provable no-op on it (it returns early on a missing
    body sentinel) and the file's 289 KB is recoverable only by deletion. But
    39% of it is the Subcatchment Runoff Summary, whose `Total Runoff 10^6 ltr`
    column is the per-subcatchment runoff volume the user asked to preserve, and
    beside it sits the Runoff Quantity Continuity error -- the only hydrology
    mass-balance record on disk. Deleting without this capture discards both
    silently.

    Reuses `return_lines_for_section_of_rpt` + `format_rpt_section_into_dataframe`,
    the SAME pair the three existing summary sections use, so this adds a column
    tuple rather than a second parsing idiom, and takes NO new dependency
    (`swmmio` / `pyswmm` are not needed and must not be introduced here).

    Returns an `xr.Dataset` indexed on `subcatchment_id` whose attrs carry the two
    continuity errors and `flow_units`. Raises nothing on a missing section: an
    absent Subcatchment Runoff Summary yields an empty Dataset with the attrs
    still populated, because a hydrology model with zero subcatchments is legal.
    """
    with open(f_rpt, encoding="latin-1") as fh:
        rpt_lines = fh.readlines()
    dict_system_results = return_swmm_system_outputs(rpt_lines)
    lst_rows = return_lines_for_section_of_rpt("Subcatchment Runoff Summary", lines=rpt_lines)
    df = format_rpt_section_into_dataframe(lst_rows, list(LST_COL_HEADERS_SUBCATCH_RUNOFF_SUMMARY))
    if not df.empty:
        df = df.set_index("subcatchment_id")
    ds = df.to_xarray()
    ds = convert_datavars_to_dtype(ds, lst_dtypes_to_try=[float, str])
    if lines_print_water_quality(rpt_lines):
        # WATER QUALITY, only when the report says quality was routed: the subcatchment washoff
        # loads and the Runoff Quality Continuity table, with the pollutants named from the
        # hydrology model file (`f_inp`, by default the `.inp` beside the report).
        from hhemt.exceptions import ProcessingError

        inp = Path(f_inp) if f_inp is not None else Path(f_rpt).with_suffix(".inp")
        if not inp.exists():
            raise ProcessingError(
                operation="parse_hydrology_rpt_summary",
                filepath=Path(f_rpt),
                reason=(
                    f"the report says water quality was routed but the model file {inp} "
                    "is not there to name the pollutants."
                ),
            )
        pollutants = _declared_pollutants(inp)
        quality = parse_rpt_quality_summaries(rpt_lines, pollutants, include_runoff_quality=True, filepath=f_rpt)
        kept = {k: v for k, v in quality.items() if k == "washoff_total_load" or k.startswith("runoff_qual_")}
        if "washoff_total_load" in kept and "subcatchment_id" in ds.coords:
            # The washoff table lists the same subcatchments as the runoff summary; reindexing keeps
            # this dataset's subcatchment_id axis exactly the runoff summary's.
            kept["washoff_total_load"] = kept["washoff_total_load"].reindex(
                subcatchment_id=ds["subcatchment_id"].values
            )
        elif "washoff_total_load" in kept:
            del kept["washoff_total_load"]
        ds = _attach_pollutant_coords(xr.merge([ds, xr.Dataset(kept)], join="left"), pollutants)
    ds.attrs = dict_system_results
    return ds


def _parse_analysis_end_line(end_line: str):
    """Parse the analysis end datetime line from the RPT file."""
    # parse analysis end time
    lst_end_line = end_line.split("on:")[-1].split(" ")
    lst_info_in_line = []
    for substring in lst_end_line:
        if len(substring) > 0:
            lst_info_in_line.append(substring)
    month = lst_info_in_line[1]
    day = lst_info_in_line[2]
    assumed_year = datetime.today().year
    time = lst_info_in_line[3]
    datetime_string = f"{month}-{day}-{assumed_year} {time}"
    return pd.to_datetime(datetime_string, format="%b-%d-%Y %H:%M:%S")


def _build_tseries_datasets(dict_nodes, dict_links):
    """Create node/link time series datasets."""
    ds_node_tseries = create_tseries_ds(
        dict_nodes,
        [
            "date_time",
            "inflow_flow_cms",
            "flooding_cms",
            "depth_m",
            "head_m",
        ],
        "node_id",
    )
    ds_link_tseries = create_tseries_ds(
        dict_links,
        [
            "date_time",
            "flow_cms",
            "velocity_mps",
            "link_depth_m",
            "capacity_setting",
        ],
        "link_id",
    )
    return ds_node_tseries, ds_link_tseries


def return_lines_for_section_of_rpt(section_header, f_rpt=None, lines=None):
    lst_section_lines = []
    if lines is None:
        with open(f_rpt, encoding="latin-1") as file:  # type: ignore
            # Read all lines from the file
            lines = file.readlines()
    line_num = -1
    encountered_header = False
    encountered_end = False
    begin_header_line_num = None
    end_header_line_num = None
    for line in lines:
        line_num += 1
        # return node flooding summaries
        if section_header in line:
            # print("encountered header")
            encountered_header = True
        if encountered_header is False:
            continue
        if "No nodes were flooded." in line:
            break
        if begin_header_line_num is None and "------------" in line:
            begin_header_line_num = line_num
            continue
        if begin_header_line_num is not None and "------------" in line:
            end_header_line_num = line_num
        if (begin_header_line_num is None) or (end_header_line_num is None):
            continue
        if line_num <= end_header_line_num:  # skip one more line
            continue
        if len(line.split(" ")) <= 3:
            encountered_end = True
            break
        if encountered_end is False:
            lst_section_lines.append(line)
    return lst_section_lines


@profile
def format_rpt_section_into_dataframe(lst_section_lines, lst_col_headers):
    if len(lst_section_lines) == 0:
        return pd.DataFrame(columns=lst_col_headers)
    dict_line_contents_aslist = return_data_from_rpt(lst_section_lines)
    records = []
    col_count = len(lst_col_headers)
    for _, lst_substrings_with_content in dict_line_contents_aslist.items():
        row = [""] * col_count
        idx_val = 0
        for substring in lst_substrings_with_content:
            if "\n" in substring:
                substring = substring.split("\n")[0]
            if substring == "ltr":
                continue
            if ":" in substring:
                if idx_val > 0:
                    row[idx_val - 1] = f"{row[idx_val - 1]} {substring}".strip()
                continue
            if idx_val < col_count:
                row[idx_val] = substring
            idx_val += 1
        records.append(row)
    return pd.DataFrame.from_records(records, columns=lst_col_headers)


def return_node_time_series_results_from_rpt(
    section_header="Node Time Series Results",
    f_rpt=None,
    lines=None,
):
    # section_header, lines = "Node Time Series Results", rpt_lines
    # lst_section_lines = []
    if lines is None:
        with open(f_rpt, encoding="latin-1") as file:  # type: ignore
            # Read all lines from the file
            lines = file.readlines()
    encountered_header = False
    dict_lst_node_time_series = {}
    dict_lst_link_time_series = {}
    begin_header_line_num = None
    end_header_line_num = None
    encountered_end = False
    key = None
    is_link = False
    is_node = False
    lst_vals = []
    for line_num, line in enumerate(lines):
        # return node flooding summaries
        if not encountered_header:
            if section_header in line:
                encountered_header = True
            else:
                continue
        if "<<<" in line:
            # define new list for values and the node name
            key = line.split("<<<")[1].split(" ")[2]
            lower_line = line.lower()
            is_link = "link" in lower_line
            is_node = "node" in lower_line
            lst_vals = []
            begin_header_line_num = None
            end_header_line_num = None
            encountered_end = False
            continue
        # find the start of the next node flow section
        if begin_header_line_num is None and "------------" in line:
            begin_header_line_num = line_num
            continue
        if begin_header_line_num is not None and "------------" in line:
            end_header_line_num = line_num
        if (begin_header_line_num is None) or (end_header_line_num is None):
            continue
        if line_num <= end_header_line_num:  # skip one more line
            continue
        if len(line.split(" ")) <= 3:
            encountered_end = True
            if key is not None:
                if is_link:
                    dict_lst_link_time_series[key] = lst_vals
                if is_node:
                    dict_lst_node_time_series[key] = lst_vals
        if encountered_end is False:
            # dict_lst_node_time_series[node_id].append(line)
            lst_vals.append(line)
    #
    ds_node_tseries, ds_link_tseries = _build_tseries_datasets(
        dict_lst_node_time_series,
        dict_lst_link_time_series,
    )
    #
    return ds_node_tseries, ds_link_tseries


def return_node_time_series_results_from_outfile(f_outfile, f_inp=None, return_quality=False):
    """Read node and link time series from a SWMM ``.out`` file.

    With ``return_quality`` the function returns ``(node ds, link ds, pollutants, start)`` where
    ``pollutants`` is the model's ``[(name, unit)]`` list when the ``.out`` holds pollutant results
    (its pollutant count is above zero, which SWMM makes zero whenever quality is ignored) and
    ``[]`` otherwise, and ``start`` is the simulation start as ``MM/DD/YYYY HH:MM:SS``. Under that
    gate the datasets carry ``pollutant_conc`` and ``link_pollutant_conc`` read from attribute index
    ``6 + p`` (node) and ``5 + p`` (link), with names and units taken from ``f_inp``'s
    ``[POLLUTANTS]`` section and never from the ``.out``'s unit flags, which report only the
    first pollutant's unit. Without ``return_quality`` the two datasets are returned alone.
    """
    from pyswmm import LinkSeries, NodeSeries, Output

    pollutants: list[tuple[str, str]] = []
    out_start = None
    with Output(str(f_outfile)) as out:
        d_links = out.links
        d_nodes = out.nodes
        units = out.units
        # The FLOW unit, not the unit system: every series below is stored as m3/s with no
        # conversion, and SWMM writes them in the model's own flow unit, so an SI model in
        # LPS or MLD would pass a unit-system test and be read 1000 or 86.4 times too large.
        # This is the standalone model's guard for outputs the preparation check never saw,
        # such as a reprocess of an existing analysis.
        if units["flow"] != "CMS":
            # RAISE, never sys.exit. SystemExit derives from BaseException, so no
            # `except Exception` in any caller catches it and the process dies with
            # no traceback -- the un-diagnosable shape that made an unrelated
            # 123-scenario failure take a full session to attribute. ProcessingError
            # is this module's own family (CLI exit 5) and carries the file.
            from hhemt.exceptions import ProcessingError

            raise ProcessingError(
                operation="return_node_time_series_results_from_outfile",
                filepath=Path(f_outfile),
                reason=(
                    f"SWMM outputs are not in cubic metres per second (flow unit reported: "
                    f"{units['flow']!r}, unit system {units['system']!r}). The toolkit reads the "
                    "standalone model's flows as m3/s without conversion, so its [OPTIONS] "
                    "FLOW_UNITS must be CMS; an ABSENT FLOW_UNITS line defaults to CFS. Fix it "
                    "in the SWMM template named by system_config, not in the generated "
                    "per-scenario .inp, which inherits it."
                ),
            )
        # PROCESSING NODES
        dic_dfs = dict(depth_m=[], head_m=[], inflow_flow_cms=[], flooding_cms=[])
        for node_id in d_nodes.keys():
            ts_depth = NodeSeries(out)[node_id].invert_depth
            ts_head = NodeSeries(out)[node_id].hydraulic_head
            ts_inflow = NodeSeries(out)[node_id].total_inflow
            ts_flooding = NodeSeries(out)[node_id].flooding_losses
            # create dataframes
            df_depth = convert_pyswmm_output_to_df(
                pyswmm_tseries=ts_depth,
                valname="depth_m",
                item_idx_name="node_id",
                item_idx=node_id,
                idx_name="date_time",
            )
            df_head = convert_pyswmm_output_to_df(
                pyswmm_tseries=ts_head,
                valname="head_m",
                item_idx_name="node_id",
                item_idx=node_id,
                idx_name="date_time",
            )
            df_inflow = convert_pyswmm_output_to_df(
                pyswmm_tseries=ts_inflow,
                valname="inflow_flow_cms",
                item_idx_name="node_id",
                item_idx=node_id,
                idx_name="date_time",
            )
            df_flooding = convert_pyswmm_output_to_df(
                pyswmm_tseries=ts_flooding,
                valname="flooding_cms",
                item_idx_name="node_id",
                item_idx=node_id,
                idx_name="date_time",
            )
            # add dataframes to dictionary
            dic_dfs["depth_m"].append(df_depth)
            dic_dfs["head_m"].append(df_head)
            dic_dfs["inflow_flow_cms"].append(df_inflow)
            dic_dfs["flooding_cms"].append(df_flooding)
        #
        lst_dfs = []
        for key in dic_dfs.keys():
            df = pd.concat(dic_dfs[key]).reset_index().set_index(["node_id", "date_time"])
            lst_dfs.append(df)
        #
        ds_node_tseries = pd.concat(lst_dfs, axis=1).to_xarray()
        #
        # PROCESSING LINKS
        dic_dfs = dict(flow_cms=[], velocity_mps=[], link_depth_m=[], capacity=[])
        for link_id in d_links.keys():
            ts_fr = LinkSeries(out)[link_id].flow_rate
            ts_dpth = LinkSeries(out)[link_id].flow_depth
            ts_vel = LinkSeries(out)[link_id].flow_velocity
            ts_cap = LinkSeries(out)[link_id].capacity
            # create dataframes
            df_fr = convert_pyswmm_output_to_df(
                pyswmm_tseries=ts_fr,
                valname="flow_cms",
                item_idx_name="link_id",
                item_idx=link_id,
                idx_name="date_time",
            )
            df_dpth = convert_pyswmm_output_to_df(
                pyswmm_tseries=ts_dpth,
                valname="link_depth_m",
                item_idx_name="link_id",
                item_idx=link_id,
                idx_name="date_time",
            )
            df_vel = convert_pyswmm_output_to_df(
                pyswmm_tseries=ts_vel,
                valname="velocity_mps",
                item_idx_name="link_id",
                item_idx=link_id,
                idx_name="date_time",
            )
            df_cap = convert_pyswmm_output_to_df(
                pyswmm_tseries=ts_cap,
                valname="capacity",
                item_idx_name="link_id",
                item_idx=link_id,
                idx_name="date_time",
            )
            # add dataframes to dictionary
            dic_dfs["flow_cms"].append(df_fr)
            dic_dfs["velocity_mps"].append(df_vel)
            dic_dfs["link_depth_m"].append(df_dpth)
            dic_dfs["capacity"].append(df_cap)
        #
        lst_dfs = []
        for key in dic_dfs.keys():
            df = pd.concat(dic_dfs[key]).reset_index().set_index(["link_id", "date_time"])
            lst_dfs.append(df)
        #
        ds_link_tseries = pd.concat(lst_dfs, axis=1).to_xarray()
        # WATER QUALITY, gated on the .out's own pollutant count (zero when quality is ignored).
        n_out_pollutants = int(out.project_size[4]) if return_quality else 0
        if n_out_pollutants > 0:
            from hhemt.exceptions import ProcessingError

            if f_inp is None:
                raise ProcessingError(
                    operation="return_node_time_series_results_from_outfile",
                    filepath=Path(f_outfile),
                    reason="the .out holds pollutant results but no model file was given to name them.",
                )
            pollutants = _declared_pollutants(f_inp)
            if len(pollutants) != n_out_pollutants:
                raise ProcessingError(
                    operation="return_node_time_series_results_from_outfile",
                    filepath=Path(f_outfile),
                    reason=(f"the .out holds {n_out_pollutants} pollutant(s) but {f_inp} declares {len(pollutants)}."),
                )
            ds_node_tseries["pollutant_conc"] = _out_pollutant_series(
                out.node_series, list(d_nodes.keys()), 6, "node_id", pollutants
            )
            ds_link_tseries["link_pollutant_conc"] = _out_pollutant_series(
                out.link_series, list(d_links.keys()), 5, "link_id", pollutants
            )
        if return_quality:
            start = getattr(out, "start", None)
            out_start = start.strftime(RPT_DATETIME_FORMAT) if start is not None else None
    #
    if return_quality:
        return ds_node_tseries, ds_link_tseries, pollutants, out_start
    return ds_node_tseries, ds_link_tseries


def _out_pollutant_series(series_reader, entity_ids, first_attribute, idx_colname, pollutants):
    """Concentration series for every entity and pollutant as (idx, date_time, pollutant).

    ``series_reader(entity_id, attribute_index)`` is pyswmm's ``Output.node_series`` or
    ``Output.link_series``; pollutant ``p`` is attribute ``first_attribute + p``.
    """
    frames = []
    for p, (name, _) in enumerate(pollutants):
        rows = []
        for entity_id in entity_ids:
            series = series_reader(entity_id, first_attribute + p)
            idx = pd.DatetimeIndex(list(series.keys()), name="date_time")
            rows.append(
                pd.Series(
                    list(series.values()),
                    index=pd.MultiIndex.from_arrays(
                        [np.repeat(entity_id, len(idx)), idx], names=[idx_colname, "date_time"]
                    ),
                ).rename(name)
            )
        frames.append(pd.concat(rows))
    df = pd.concat(frames, axis=1)
    da = df.to_xarray().to_dataarray(dim="pollutant").transpose(idx_colname, "date_time", "pollutant")
    return da.assign_coords(pollutant=[name for name, _ in pollutants]).astype(float)


def create_tseries_ds(dict_lst_time_series, lst_col_headers, idx_colname):
    # dict_lst_time_series, lst_col_headers, idx_colname = dict_lst_node_time_series, lst_col_headers, "link_id"
    # dict_lst_time_series, lst_col_headers, idx_colname = dict_lst_node_time_series, lst_col_headers, "node_id"
    lst_dfs = []
    for key, lst_section_lines in dict_lst_time_series.items():
        df_tseries = format_rpt_section_into_dataframe(lst_section_lines, lst_col_headers)
        if df_tseries.empty:
            continue
        df_tseries[idx_colname] = key
        lst_dfs.append(df_tseries)
    if not lst_dfs:
        empty_df = pd.DataFrame(columns=[idx_colname, *lst_col_headers])
        if "date_time" not in empty_df.columns:
            empty_df["date_time"] = pd.to_datetime([], format=RPT_DATETIME_FORMAT)
        empty_df = empty_df.set_index([idx_colname, "date_time"])
        return empty_df.to_xarray()
    df_tseries = pd.concat(lst_dfs, ignore_index=True)
    df_tseries["date_time"] = pd.to_datetime(df_tseries["date_time"], format=RPT_DATETIME_FORMAT, errors="coerce")
    if df_tseries["date_time"].isna().any():
        raise ValueError(
            f"Parsed RPT date_time values contained NaT. Verify the RPT datetime format matches {RPT_DATETIME_FORMAT}."
        )
    df_tseries = df_tseries.set_index([idx_colname, "date_time"])
    return df_tseries.to_xarray()


def convert_pyswmm_output_to_df(pyswmm_tseries, valname, item_idx_name, item_idx, idx_name="date_time"):
    df = pd.DataFrame.from_dict(pyswmm_tseries, orient="index", columns=[valname])
    df.index.name = idx_name
    df[item_idx_name] = str(item_idx)
    return df


def convert_swmm_tdeltas_to_minutes(s_tdelta):
    """
    Convert SWMM time delta strings (e.g., "0  05:30") to minutes.

    Vectorized implementation using pandas string methods for improved performance.

    Parameters
    ----------
    s_tdelta : pd.Series or list-like
        Series of time delta strings in format "D  HH:MM" where D is days

    Returns
    -------
    list
        List of time deltas in minutes (float), with NaN for invalid entries
    """
    if not isinstance(s_tdelta, pd.Series):
        s_tdelta = pd.Series(s_tdelta)

    if len(s_tdelta) == 0:
        return []

    extracted = s_tdelta.astype(str).str.extract(TDELTA_PATTERN)

    days = pd.to_numeric(extracted[0], errors="coerce")
    hours = pd.to_numeric(extracted[1], errors="coerce")
    minutes = pd.to_numeric(extracted[2], errors="coerce")

    total_minutes = days * 1440 + hours * 60 + minutes

    return total_minutes.tolist()


@profile
def return_data_from_rpt(lst_section_lines):
    lst_substrings_to_ignore = ["ltr\n"]
    # initialize vars
    dict_line_contents_aslist = {}
    dict_content_lengths = {}
    # extract and parse data in each line using spaces
    line_contents = []
    line_strings = []
    line_lengths = []
    for line in lst_section_lines:
        lst_substrings_with_content = [
            substring for substring in line.split(" ") if substring and substring not in lst_substrings_to_ignore
        ]
        if lst_substrings_with_content:
            line_contents.append(lst_substrings_with_content)
            line_strings.append(line)
            line_lengths.append(len(lst_substrings_with_content))
    dict_line_contents_aslist = {idx: contents for idx, contents in enumerate(line_contents)}
    dict_content_lengths = {idx: length for idx, length in enumerate(line_lengths)}
    # make sure the lines all have the same lengths
    if not line_lengths:
        return dict_line_contents_aslist
    target_length = Counter(line_lengths).most_common(1)[0][0]
    idx_problem_rows = [idx for idx, length in dict_content_lengths.items() if length != target_length]
    # if there is an issue
    if len(idx_problem_rows) > 0:
        normal_row, normal_row_list = _select_normal_row(
            line_strings, line_lengths, idx_problem_rows, dict_line_contents_aslist
        )
        # loop through the problem rows and correct known issues
        for idx in idx_problem_rows:
            idx_int = cast(int, idx)
            problem_row = str(line_strings[idx_int])
            problem_row_list = dict_line_contents_aslist[idx_int]
            solution = None
            problem_row_lower = problem_row.lower()
            for prob_index, val in enumerate(problem_row_list):  # noqa: B007 - prob_index is read AFTER this loop at the normal_row_list lookups
                if val.count(".") > 1:
                    solution = _ADJACENT_VALUE_SOLUTION
                    print("##################################")
                    print(f"Found problem. {solution}")
                    print("Normal row vs. problem row:")
                    print(normal_row)
                    print(problem_row)
                    break
                if "orifice" in problem_row_lower:
                    solution = (
                        "Orifice conduits do not return max velocity or max over full flow. Filling with empty string"
                    )
                    print("##################################")
                    print(f"Found problem. {solution}")
                    print("Normal row vs. problem row:")
                    print(normal_row)
                    print(problem_row)
                    break
            if solution == _ADJACENT_VALUE_SOLUTION:
                # problem_val = problem_row_list[prob_index]
                normal_val_at_index = normal_row_list[prob_index]
                normal_next_val_at_index = normal_row_list[prob_index + 1]
                # identify string parsing location
                ## deal with the possibility that there are multiple substrings with the same value;
                ## find the pair with the smallest difference between the current val and next
                closest_end_loc_of_val_at_index = 9999
                dif_between_locs = 9999
                for normal_val_at_index_string_ilocs in find_substring_indices(normal_row, normal_val_at_index):
                    if normal_val_at_index != extract_substring(normal_row, normal_val_at_index_string_ilocs):
                        print("WARNING: There is a string indexing issue")
                    end_loc_prev = max(normal_val_at_index_string_ilocs)
                    for normal_val_at_next_index_string_ilocs in find_substring_indices(
                        normal_row, normal_next_val_at_index
                    ):
                        begin_loc_next = min(normal_val_at_next_index_string_ilocs)
                        if (begin_loc_next - end_loc_prev) < dif_between_locs:
                            dif_between_locs = begin_loc_next - end_loc_prev
                            closest_end_loc_of_val_at_index = end_loc_prev
                #
                split_loc = closest_end_loc_of_val_at_index
                # split_at_index(normal_row, split_loc+1)
                lst_row_split = split_at_index(problem_row, split_loc + 1)
                lst_substrings_corrected = []
                # lst_values = []
                # isolate strings with relevant values
                for line in lst_row_split:
                    for substring in line.split(" "):
                        if (len(substring) > 0) and (
                            substring not in lst_substrings_to_ignore
                        ):  # the latter part is to deal with issues
                            lst_substrings_corrected.append(substring)
                dict_line_contents_aslist[idx_int] = lst_substrings_corrected
                dict_content_lengths[idx_int] = len(lst_substrings_corrected)
                print(f"Properly parsed values:\n{lst_substrings_corrected}")
            elif (
                solution
                == "Orifice conduits do not return max velocity or max over full flow. Filling with empty string"
            ):
                problem_row_list.insert(5, "")
                problem_row_list.insert(6, "")
                dict_line_contents_aslist[idx_int] = problem_row_list
                print(f"Properly parsed values:\n{dict_line_contents_aslist[idx_int]}")
            else:
                print("####################################################################")
                print(
                    "WARNING: There is an issue with swmm outputs read from an rpt file that I have not yet encountered"
                )
                print("Here is an example of a normal row:")
                print(normal_row)
                print(f"There are {len(idx_problem_rows)} problem rows. Here are examples:")
                for idx in idx_problem_rows[:5]:
                    print(line_strings[idx])
                print("####################################################################")
    return dict_line_contents_aslist


def extract_substring(main_string, indices):
    """
    Extracts a substring from a string using a tuple of (start, end) indices.

    Parameters:
    main_string (str): The original string.
    indices (tuple): A tuple (start, end) representing the start and end indices.

    Returns:
    str: The extracted substring.
    """
    start, end = indices  # Unpack the tuple
    return main_string[start : end + 1]  # Use slicing to extract the substring


def _select_normal_row(line_strings, line_lengths, idx_problem_rows, dict_line_contents_aslist):
    str_lengths = [len(line) for line in line_strings]
    str_length_mode = Counter(str_lengths).most_common(1)[0][0]
    normal_idx = None
    for idx, (str_len, _content_len) in enumerate(zip(str_lengths, line_lengths, strict=False)):
        if str_len == str_length_mode and idx not in idx_problem_rows:
            normal_idx = idx
            break
    if normal_idx is None:
        normal_idx = 0
    normal_idx_int = cast(int, normal_idx)
    normal_row = line_strings[normal_idx]
    normal_row_list = dict_line_contents_aslist[normal_idx_int]
    return normal_row, normal_row_list


def find_substring_indices(main_string, substring):
    """
    Finds all occurrences of a substring in a string.

    Parameters:
    main_string (str): The string to search within.
    substring (str): The substring to search for.

    Returns:
    list of tuples: A list of (start, end) tuples representing the start and end index of each occurrence.
    """
    return [(m.start(), m.end() - 1) for m in _substring_pattern(substring).finditer(main_string)]


@lru_cache(maxsize=256)
def _substring_pattern(substring: str) -> re.Pattern:
    return re.compile(re.escape(substring))


def split_at_index(main_string, index):
    """
    Splits a string into two parts at a given index.

    Parameters:
    main_string (str): The string to split.
    index (int): The index at which to split the string.

    Returns:
    tuple: A tuple containing the two parts of the split string.
    """
    part1 = main_string[:index]  # From the beginning to the index (exclusive)
    part2 = main_string[index:]  # From the index to the end
    return [part1, part2]


def convert_coords_to_dtype(
    ds,
    lst_dtypes_to_try=(int, str),
    coords_to_coerce=("node_id", "link_id", "model", "simtype"),
):
    for coord in ds.coords:
        if (ds[coord].dtype == object) or (coord in coords_to_coerce):
            converted = False
            for dtype in lst_dtypes_to_try:
                # break the loop if it is already the desired data type
                if ds[coord].dtype == dtype:
                    converted = True
                    break
                if dtype is int:
                    numeric = pd.to_numeric(ds[coord].values, errors="coerce")
                    if np.isnan(numeric).any():
                        continue
                    if np.any(numeric % 1 != 0):
                        dtype = float
                    ds[coord] = ds[coord].astype(dtype)
                    converted = True
                    break
                try:
                    ds[coord] = ds[coord].astype(dtype)
                    converted = True
                    # print(f"converted {coord} to {dtype}")
                    break
                except Exception:
                    continue
            if not converted:
                print(f"{coord} unable to be converted to either {lst_dtypes_to_try}")
    return ds


def convert_datavars_to_dtype(ds, lst_dtypes_to_try=(str,), lst_vars_to_convert=None):
    # ds, lst_dtypes_to_try=[float, str]
    if lst_vars_to_convert is None:  # convert all variables
        lst_vars_to_convert = ds.data_vars
    for var in lst_vars_to_convert:
        if ds[var].size == 0:
            continue
        converted = False
        first_attempt = True
        for dtype in lst_dtypes_to_try:
            # break if it alread is the resired data type
            if ds[var].dtype == dtype:
                converted = True
                break
            # deal with common problem in SWMM results
            if (dtype is float) or (dtype is int):
                sample = isel_first_and_slice_longest(ds[var], n=10)
                sample_values = sample.values
                if sample.dtype == object:
                    sample_values = np.where(sample_values == "", "0", sample_values)
                sample_numeric = pd.to_numeric(sample_values.ravel(), errors="coerce")
                if np.isnan(sample_numeric).any():
                    non_null_mask = ~pd.isna(sample_values.ravel())
                    invalid_mask = non_null_mask & np.isnan(sample_numeric)
                    if invalid_mask.any():
                        first_attempt = False
                        continue
                if dtype is int:
                    if np.any(sample_numeric % 1 != 0):
                        first_attempt = False
                        continue
                data_to_convert = ds[var]
                if data_to_convert.dtype == object:
                    data_to_convert = data_to_convert.astype(str)
                    data_to_convert = xr.where(data_to_convert == "", "0", data_to_convert)
                try:
                    ds[var] = data_to_convert.astype(dtype)
                    converted = True
                except (ValueError, TypeError):
                    first_attempt = False
                    continue
            else:
                try:
                    ds[var] = ds[var].astype(dtype)
                    converted = True
                except (ValueError, TypeError):
                    first_attempt = False
                    continue
            if converted:
                if not first_attempt:
                    print(f"Converted variable to datatype = {var}, {dtype}")
                break
            first_attempt = False
        if not converted:
            print(f"{var} unable to be converted to either {lst_dtypes_to_try}")
    return ds


def isel_first_and_slice_longest(ds, n=5):
    """
    Select the first element for all dimensions, but slice the first `n` elements
    for the longest dimension.
    """
    if any(ds.sizes[dim] == 0 for dim in ds.dims):
        return ds
    # Find the longest dimension
    longest_dim = max(ds.dims, key=lambda d: ds.sizes[d])
    # Build the `isel` dictionary: first element for all dims, slice for the longest
    isel_dict = {dim: 0 for dim in ds.dims}  # Default to first index
    isel_dict[longest_dim] = slice(  # type: ignore
        0, min(ds.sizes[longest_dim], n)
    )  # Slice longest dim
    # Apply the `isel` operation using the dictionary
    return ds.isel(isel_dict)


def parse_total_elapsed(rpt_path: Path) -> float | None:
    """Parse 'Total elapsed time' from a SWMM .rpt and return seconds.

    Handles both canonical forms emitted by SWMM 5.2.x:
      - "Total elapsed time: HH:MM:SS" (long-run form; lowercase, colon-space separator)
      - "Total elapsed time: < 1 sec"  (short-run form; returns 0.5 by convention)

    Returns None if the file is missing, unreadable, or the line is absent.

    Reference: vendored SWMM solver `text.h` FMT21 = "  Total elapsed time: ".
    """
    if not rpt_path or not rpt_path.exists():
        return None
    try:
        text = rpt_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    m = re.search(r"Total elapsed time:\s*(\d+):(\d+):(\d+)", text)
    if m:
        h, mi, s = (int(m.group(i)) for i in (1, 2, 3))
        return float(h * 3600 + mi * 60 + s)
    if re.search(r"Total elapsed time:\s*<\s*1\s*sec", text):
        return 0.5
    return None
