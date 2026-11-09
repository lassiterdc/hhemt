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
    FLOW_ROUTING_CONTINUITY_TERMS,
    LST_COL_HEADERS_CONDUIT_SURCHARGE_SUMMARY,
    LST_COL_HEADERS_FLOW_CLASSIFICATION_SUMMARY,
    LST_COL_HEADERS_LINK_FLOW_SUMMARY,
    LST_COL_HEADERS_NODE_DEPTH_SUMMARY,
    LST_COL_HEADERS_NODE_FLOOD_SUMMARY,
    LST_COL_HEADERS_NODE_FLOW_SUMMARY,
    LST_COL_HEADERS_NODE_SURCHARGE_SUMMARY,
    LST_COL_HEADERS_OUTFALL_LOADING_SUMMARY,
    LST_COL_HEADERS_PUMPING_SUMMARY,
    LST_COL_HEADERS_STORAGE_VOLUME_SUMMARY,
    LST_OUTFALL_SYSTEM_VARIABLES,
    LST_RUN_LEVEL_CONTINUITY_VARIABLES,
    ROUTING_STEP_FREQUENCY_VARIABLE,
    RUNOFF_QUANTITY_CONTINUITY_TERMS,
    SWMM_STAT_RANK_LENGTH,
    SWMM_SYSTEM_SERIES,
)

TDELTA_PATTERN = re.compile(r"^\s*(\d+)\s+(\d+):(\d+)")
RPT_DATETIME_FORMAT = "%m/%d/%Y %H:%M:%S"

_ADJACENT_VALUE_SOLUTION = (
    "Two values in the rpt were right next to each other and couldn't be parsed "
    "using spacing. Parsing by referencing a normal line."
)

# The banner that opens a report's per-timestep body. Everything the toolkit reads from a report precedes it, apart
# from the trailer (`Analysis ended on`), which follows both time-series sections and is read from the file's last
# lines. The same sentinel is the truncator's body start (`process_simulation._RPT_BODY_START_SENTINEL`), so the
# reader and the reclaim agree on where the head ends.
_RPT_TSERIES_BANNER = "Node Time Series Results"
# SWMM writes at most three lines after the last time-series row (begun, ended, elapsed); this window holds them with
# room to spare and is the only part of a multi-hundred-megabyte coupled report read after the banner.
_RPT_TAIL_BYTES = 8192
# A table SWMM did not print says so in one sentence instead of its header (`statsrpt.c`: "No nodes were flooded.",
# "No nodes were surcharged.", "No conduits were surcharged.").
_RE_NO_ROWS_SENTENCE = re.compile(r"^\s*No (?:nodes|conduits) were \w+\.")
# Report sections that are a title, one rule line and then lines of text up to a blank line, not a dashed table.
_TEXT_BLOCK_SECTION_TITLES = frozenset(
    {
        "Runoff Quantity Continuity",
        "Flow Routing Continuity",
        "Highest Continuity Errors",
        "Time-Step Critical Elements",
        "Highest Flow Instability Indexes",
        "Most Frequent Nonconverging Nodes",
        "Routing Time Step Summary",
    }
)
#: Every section title `_scan_metadata_and_summaries` collects the lines of: the dashed tables (the three the toolkit
#: has always read, then the ones added with the binary-output series) and the text blocks above.
_REPORT_SECTION_HEADERS = (
    "Node Flooding Summary",
    "Node Inflow Summary",
    "Link Flow Summary",
    "Node Depth Summary",
    "Node Surcharge Summary",
    "Storage Volume Summary",
    "Outfall Loading Summary",
    "Flow Classification Summary",
    "Conduit Surcharge Summary",
    "Pumping Summary",
    *sorted(_TEXT_BLOCK_SECTION_TITLES),
)


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


def node_flooding_ponded_field(lines) -> str | None:
    """The LAST field of the Node Flooding Summary's third header line: ``Depth`` or ``Volume``, else None.

    SWMM heads the sixth column of that table ``Depth`` when the model is routed by Dynamic Wave and ``Volume``
    otherwise (``statsrpt.c``, "Ponded"), so the column is a depth in meters under one routing method and a
    ponded volume under the others. The header block is the lines between the table's two dashed rules; its
    third line ends in that word. The WHOLE line is not tested for ``Volume``: the Dynamic-Wave line also
    contains ``Volume`` (the flood-volume column's heading), which is why only its last field decides.
    ``lines`` may be any iterable of report lines, including an open file, and is read only as far as the
    header block."""
    in_section = False
    n_rules = 0
    header: list[str] = []
    for line in lines:
        if not in_section:
            in_section = "Node Flooding Summary" in line
            continue
        if "No nodes were flooded." in line:
            return None
        if "------------" in line:
            n_rules += 1
            if n_rules == 2:
                break
            continue
        if n_rules == 1 and line.strip():
            header.append(line)
    if len(header) < 3 or not header[2].split():
        return None
    return header[2].split()[-1]


def _apply_ponded_depth_rule(df_node_flood_summary: pd.DataFrame, ponded_field: str | None, rpt_name) -> None:
    """Keep ``max_ponded_depth`` only where the report heads that column ``Depth``; else NaN and warn (in place)."""
    if ponded_field == "Depth" or df_node_flood_summary.empty:
        return
    df_node_flood_summary["max_ponded_depth"] = np.nan
    warnings.warn(
        f"{rpt_name}: the Node Flooding Summary's last column is headed {ponded_field!r}, not 'Depth' "
        "(SWMM prints a depth only under Dynamic Wave routing), so max_ponded_depth is NaN for every node.",
        UserWarning,
        stacklevel=2,
    )


def format_node_inflow_summary(lst_section_lines, lst_col_headers) -> pd.DataFrame:
    """The Node Inflow Summary as a frame whose last field is split by the unit SWMM printed after it.

    SWMM prints the node's flow balance error as a percent, or, where the node's outflow volume is below one
    cubic foot, as an absolute error in litres followed by ``ltr`` (``statsrpt.c``). The shared section reader
    drops that token, which leaves one column holding two units. Here each row's last value goes to
    ``flow_balance_error`` (percent) where no ``ltr`` followed it and to ``flow_balance_error_volume`` (litres)
    where one did, the other being NaN on that row."""
    df = format_rpt_section_into_dataframe(lst_section_lines, lst_col_headers, section_name="Node Inflow Summary")
    has_ltr = [
        line.split()[-1] == "ltr" for line in lst_section_lines if [t for t in line.split(" ") if t and t != "ltr\n"]
    ]
    if len(has_ltr) != len(df):
        raise ValueError(
            f"Node Inflow Summary: {len(df)} parsed rows but {len(has_ltr)} report lines to read the unit from"
        )
    flag = pd.Series(has_ltr, index=df.index, dtype=bool)
    df["flow_balance_error_volume"] = df["flow_balance_error"].where(flag)
    df["flow_balance_error"] = df["flow_balance_error"].where(~flag)
    return df


def retrieve_SWMM_outputs_as_datasets(
    f_swmm_inp: Path,
    swmm_timeseries_result_file: Path,
    *,
    family: str,
):
    """Read one SWMM run into a node dataset and a link dataset.

    The per-step series of BOTH families come from the binary output ``swmm_timeseries_result_file`` (a ``.out``),
    and the summary tables, continuity figures and ranked lists from the report beside it (same stem, ``.rpt``).
    ``family`` is REQUIRED and names the model that wrote them: ``"tritonswmm"`` for the coupled model (its link
    capacity series is ``capacity_setting``) or ``"swmm"`` for standalone SWMM (``capacity``); omitting
    it is a TypeError, so a coupled caller cannot silently get the standalone name. A coupled run's report is
    required; a standalone run without a report yields the series alone.
    """
    # TODO - reindex nodes and links by entity type; this will likely resolve parsing errors as well
    # TODO - record warnings as attributes in dataset and in sim processing log
    ds_nodes, ds_links = return_swmm_outputs(
        f_swmm_inp,
        swmm_timeseries_result_file,
        LST_COL_HEADERS_NODE_FLOOD_SUMMARY,
        LST_COL_HEADERS_NODE_FLOW_SUMMARY,
        LST_COL_HEADERS_LINK_FLOW_SUMMARY,
        family=family,
    )

    return ds_nodes, ds_links


def _read_report_head_and_tail(f_rpt: Path) -> tuple[list[str], list[str]]:
    """The report lines before the time-series banner, and the file's last lines when a banner cut the read short.

    A coupled report is mostly its per-timestep body (82 to 99 percent of its lines), which no reader parses, so the
    body is never held in memory: the file is read line by line up to the banner and the trailer, which follows
    both time-series sections, is read from the last ``_RPT_TAIL_BYTES`` of the file. A report with no banner (a
    truncated one, or a model that reports no node series) is read whole and its tail is empty, because its
    trailer is already among the lines.
    """
    head: list[str] = []
    banner_seen = False
    with open(f_rpt, encoding="latin-1") as fh:
        for line in fh:
            if _RPT_TSERIES_BANNER in line:
                banner_seen = True
                break
            head.append(line)
    if not banner_seen:
        return head, []
    with open(f_rpt, "rb") as fh:
        fh.seek(0, 2)
        fh.seek(max(0, fh.tell() - _RPT_TAIL_BYTES))
        tail = fh.read().decode("latin-1").splitlines(keepends=True)
    return head, tail


def _clean_link_id(link_id):
    """Clean a single link_id value."""
    try:
        return str(int(float(link_id)))
    except (ValueError, TypeError):
        if isinstance(link_id, str):
            parts = link_id.split()
            return parts[0] if parts else link_id
        return str(link_id)


def return_swmm_outputs(
    f_swmm_inp: Path,
    swmm_timeseries_result_file: Path,
    lst_col_headers_node_flood_summary,
    lst_col_headers_node_flow_summary,
    lst_col_headers_link_flow_summary,
    *,
    family: str,
):
    """Assemble the node and link datasets of one run: series from the binary output, tables from the report.

    One reader serves both model families (see `retrieve_SWMM_outputs_as_datasets`). The report is read by ONE scan
    that stops at the time-series banner, so the per-timestep body is never parsed; a report the scan cannot read
    RAISES, because a coupled report failure that warned instead would write a store whose summary variables are
    silently absent.
    """
    from hhemt.exceptions import ProcessingError

    if family not in ("swmm", "tritonswmm"):
        raise ValueError(f"family must be 'swmm' or 'tritonswmm', not {family!r}")
    out_file = Path(swmm_timeseries_result_file)
    if out_file.suffix != ".out":
        raise ProcessingError(
            operation="return_swmm_outputs",
            filepath=out_file,
            reason=(
                "the SWMM time series are read from the binary output, not from a report; pass the sibling "
                f"{out_file.with_suffix('.out').name} instead (the report beside it supplies the summary tables)."
            ),
        )
    quality_vars: dict[str, xr.DataArray] = {}
    simulation_start: str | None = None
    ds_node_tseries, ds_link_tseries, pollutants, out_start = return_node_time_series_results_from_outfile(
        out_file,
        f_inp=f_swmm_inp,
        return_quality=True,
        capacity_name="capacity_setting" if family == "tritonswmm" else "capacity",
    )
    if pollutants:
        simulation_start = out_start
        quality_vars["pollutant_conc"] = ds_node_tseries["pollutant_conc"]
        quality_vars["link_pollutant_conc"] = ds_link_tseries["link_pollutant_conc"]
        ds_node_tseries = ds_node_tseries.drop_vars("pollutant_conc")
        ds_link_tseries = ds_link_tseries.drop_vars("link_pollutant_conc")

    rpt_file = out_file.with_suffix(".rpt")
    dict_system_results: dict[str, Any] = {}
    dict_section_lines: dict[str, list[str]] = {}
    head_lines: list[str] | None = None
    if rpt_file.exists():
        head_lines, tail_lines = _read_report_head_and_tail(rpt_file)
        (
            dict_section_lines,
            line_flw_units,
            runoff_continuity_error_line,
            flow_continuity_error_line,
            system_flood_loss_line,
            analysis_end_line,
            _valid,
        ) = _scan_metadata_and_summaries(head_lines, _REPORT_SECTION_HEADERS, tail_lines=tail_lines)
        dict_system_results = _build_system_results(
            line_flw_units,
            runoff_continuity_error_line,
            flow_continuity_error_line,
            system_flood_loss_line,
            analysis_end_line,
        )
        # The coupled model's flows are read as m3/s with no conversion, so a report in any other unit is refused even
        # when the binary output beside it passed its own unit check: the guard for reports the preparation check
        # never saw, such as a reprocess of an existing analysis. Deliberately NOT in `_build_system_results`, which
        # the hydrology report summary also reaches, and the hydrology model accepts LPS and MLD because its flows
        # are converted.
        if family == "tritonswmm" and dict_system_results["flow_units"] != "cms":
            raise ProcessingError(
                operation="return_swmm_outputs",
                filepath=rpt_file,
                reason=(
                    f"SWMM report flow units are {dict_system_results['flow_units'].upper()!r}; the toolkit reads "
                    "the coupled model's flows as m3/s without conversion, so the hydraulics model's [OPTIONS] "
                    "FLOW_UNITS must be CMS. Fix it in the SWMM template named by system_config."
                ),
            )
        if pollutants and lines_print_water_quality(head_lines):
            quality_vars.update(
                parse_rpt_quality_summaries(
                    head_lines, pollutants, include_runoff_quality=(family == "swmm"), filepath=rpt_file
                )
            )
    elif family == "tritonswmm":
        raise ProcessingError(
            operation="return_swmm_outputs",
            filepath=rpt_file,
            reason="the coupled SWMM report is missing; the summary tables and continuity figures are read from it.",
        )
    df_node_flood_summary = format_rpt_section_into_dataframe(
        dict_section_lines.get("Node Flooding Summary", []),
        lst_col_headers_node_flood_summary,
        section_name="Node Flooding Summary",
    )
    _ponded_field = node_flooding_ponded_field(head_lines) if head_lines is not None else None
    _apply_ponded_depth_rule(df_node_flood_summary, _ponded_field, rpt_file.name)
    df_node_flow_summary = format_node_inflow_summary(
        dict_section_lines.get("Node Inflow Summary", []), lst_col_headers_node_flow_summary
    )
    df_link_flow_summary = format_link_flow_summary(
        dict_section_lines.get("Link Flow Summary", []), lst_col_headers_link_flow_summary
    )
    # combine event summary dataframes into an xarray dataset
    df_node_flood_summary.set_index("node_id", inplace=True)
    df_node_flow_summary.set_index("node_id", inplace=True)
    #
    df_node_summaries = df_node_flood_summary.join(df_node_flow_summary, how="outer")
    # process time stuff
    df_node_summaries["time_of_max_flood"] = convert_swmm_tdeltas_to_minutes(
        df_node_summaries["time_of_max_flood_d_hr_mn"]
    )
    df_node_summaries["time_of_max_total_inflow"] = convert_swmm_tdeltas_to_minutes(
        df_node_summaries["time_of_max_total_inflow_d_hr_mn"]
    )
    #
    df_node_summaries = df_node_summaries.drop(
        columns=["time_of_max_flood_d_hr_mn", "time_of_max_total_inflow_d_hr_mn"]
    )
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
    df_link_flow_summary["time_of_max_flow"] = convert_swmm_tdeltas_to_minutes(
        df_link_flow_summary["time_of_max_flow_d_hr_mn"]
    )
    df_link_flow_summary = df_link_flow_summary.drop(columns=["time_of_max_flow_d_hr_mn"])
    df_link_flow_summary["link_id"] = df_link_flow_summary["link_id"].apply(_clean_link_id)
    df_link_flow_summary.set_index("link_id", inplace=True)

    # The tables beyond the first three, each outer-joined onto the frame of its element on the id exactly as the Node
    # Flooding Summary is: an element a table does not print is NaN in that table's columns.
    new_node_frames, new_link_frames, outfall_system_max_flow = _read_new_report_tables(
        dict_section_lines, head_lines, rpt_file
    )
    for frame in new_node_frames:
        df_node_summaries = df_node_summaries.join(frame, how="outer")
    for frame in new_link_frames:
        df_link_flow_summary = df_link_flow_summary.join(frame, how="outer")

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

    ds_node_run, ds_link_run = _run_level_datasets(
        dict_section_lines,
        dict_system_results,
        outfall_system_max_flow,
        rpt_file,
        has_report=head_lines is not None,
        family=family,
    )
    ds_nodes = xr.merge([ds_node_summaries, ds_node_tseries, ds_node_run], join="outer")
    ds_links = xr.merge([ds_link_flow_summary, ds_link_tseries, ds_link_run], join="outer")
    #
    if quality_vars:
        # Both frames above read the Outfall Loading Summary's volume column, so the quality path's copy of it is
        # the same printed number and is not merged a second time.
        quality_vars.pop("outfall_total_vol", None)
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
    # `flow_units` and `analysis_end_datetime` are equal for every event of an analysis, so they survive
    # consolidation's `drop_conflicts`; the continuity figures that DIFFER by event are data variables instead.
    stable_attrs = {
        k: dict_system_results[k] for k in ("flow_units", "analysis_end_datetime") if k in dict_system_results
    }
    ds_nodes.attrs = dict(stable_attrs)
    ds_links.attrs = dict(stable_attrs)
    if "pollutant_conc" in ds_nodes.data_vars:
        # The peak's time is measured here, where the simulation start is known, and stored as the declared
        # variable it is. The start itself is NOT stored: a dataset attribute carrying it would be persisted
        # in every quality-routed node store as a key the output data model does not declare.
        ds_nodes["pollutant_time_of_max_conc"] = pollutant_time_of_max_conc(
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


def read_rpt_starting_date(rpt_file) -> str | None:
    """The ``Starting Date`` of a SWMM report file, read line by line, or None when no line carries it.

    The value sits in the ``Analysis Options`` block, which follows every input warning SWMM printed, so its
    line number scales with the model: a coupled Norfolk report holds it at line 4147, and a fixed-size head
    read misses it. The file is read only as far as the first match.
    """
    with open(rpt_file, encoding="latin-1") as rpt:
        for line in rpt:
            match = _RE_START_DATE.search(line)
            if match:
                return match.group(1)
    return None


def pollutant_time_of_max_conc(conc, simulation_start, tstep_dimname="date_time"):
    """Minutes from the simulation start to the first step at which each series reaches its maximum.

    ``conc`` is a concentration DataArray whose last-but-one axis is the time dimension; the result drops
    that dimension. ``simulation_start`` is the report's ``Starting Date`` text (``MM/DD/YYYY HH:MM:SS``);
    when it is None the origin is the first reported step. The unit and origin match
    ``time_of_max_total_inflow`` and ``time_of_max_flow``.
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

    - ``outfall_total_load`` (node_id, pollutant), ``outfall_total_vol`` (node_id) and
      ``outfall_emc`` (node_id, pollutant) from ``Outfall Loading Summary``. The event mean
      concentration is the total load over the total volume in the pollutant's concentration
      unit, and NaN where the volume is zero.
    - ``link_total_load`` (link_id, pollutant) from ``Link Pollutant Load Summary``.
    - ``washoff_total_load`` (subcatchment_id, pollutant) from ``Subcatchment Washoff Summary``.
    - ``qual_routing_{term}`` and ``qual_routing_continuity_error`` (pollutant) from
      ``Quality Routing Continuity``; and, when ``include_runoff_quality``,
      ``runoff_qual_{term}`` and ``runoff_qual_continuity_error`` from ``Runoff Quality
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
        out["outfall_total_vol"] = xr.DataArray(volume, dims=("node_id",), coords={"node_id": ids})
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
            out[f"{prefix}_continuity_error"] = xr.DataArray(
                np.array(table["Continuity Error (%)"]), dims=("pollutant",), coords={"pollutant": names}
            )

    _continuity("Quality Routing Continuity", "qual_routing", _QUAL_ROUTING_TERMS)
    if include_runoff_quality:
        _continuity("Runoff Quality Continuity", "runoff_qual", _RUNOFF_QUAL_TERMS)
    return out


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


def parse_rpt_single_pass(f_rpt: Path):
    """
    Parse a SWMM report's head in a single pass, extracting summaries and the system-level figures.

    Returns ``(section_lines, system_results, valid)``: the data lines of every section in
    ``_REPORT_SECTION_HEADERS``, the dictionary `_build_system_results` builds, and a flag that is True when the
    report carries an ``Element Count`` block. The per-timestep body is never read (the series come from the binary
    output), and the trailer is read from the file's last lines.
    """
    head_lines, tail_lines = _read_report_head_and_tail(Path(f_rpt))
    (
        dict_section_lines,
        line_flw_units,
        runoff_continuity_error_line,
        flow_continuity_error_line,
        system_flood_loss_line,
        analysis_end_line,
        valid,
    ) = _scan_metadata_and_summaries(head_lines, _REPORT_SECTION_HEADERS, tail_lines=tail_lines)

    dict_system_results = _build_system_results(
        line_flw_units,
        runoff_continuity_error_line,
        flow_continuity_error_line,
        system_flood_loss_line,
        analysis_end_line,
    )
    return dict_section_lines, dict_system_results, valid


def _scan_metadata_and_summaries(lines, summary_section_headers, tail_lines=None):
    """Single-pass scan for single-shot metadata lines + section line lists.

    Reads ``lines`` (any iterable of report lines) and STOPS at the ``Node Time Series Results`` banner, so the
    per-timestep body is never tested, let alone parsed. Everything the scan needs precedes the banner except the
    ``Analysis ended on`` trailer, which follows both time-series sections: when the scan stopped before reaching it,
    it is taken from ``tail_lines`` (the file's last lines), or from the last lines of ``lines`` when that is a list.

    A header in ``summary_section_headers`` that is a dashed table collects the lines after the table's second
    rule (a rule that follows the header block is skipped, which keeps the Outfall Loading Summary's System row);
    a header in ``_TEXT_BLOCK_SECTION_TITLES`` collects the non-blank lines after its single rule line.
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
    block_title = None
    block_rule_pending = False

    for line_num, line in enumerate(lines):
        # The literal, not `_RPT_TSERIES_BANNER`: `tests/test_synth_remove_after_processing.py` derives the markers the
        # reclaim truncator must preserve by reading the string constants of THIS function, and the body-start
        # sentinel it must find here is the one marker truncation is allowed to consume.
        if "Node Time Series Results" in line:
            break
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

        if block_title is not None:
            if block_rule_pending:
                block_rule_pending = False
                continue
            if not line.strip():
                block_title = None
                continue
            dict_section_lines[block_title].append(line)
            continue

        for header in summary_section_headers:
            if header in line:
                if header in _TEXT_BLOCK_SECTION_TITLES:
                    block_title = header
                    block_rule_pending = True
                    current_summary_section = None
                else:
                    current_summary_section = header
                    summary_begin_header = False
                    summary_end_header = False
                break
        if block_title is not None or current_summary_section is None:
            continue
        if _RE_NO_ROWS_SENTENCE.match(line):
            current_summary_section = None
            continue
        if "------------" in line:
            if not summary_begin_header:
                summary_begin_header = True
                continue
            if not summary_end_header:
                summary_end_header = True
                continue
            continue
        if not (summary_begin_header and summary_end_header):
            continue
        if line_num == 0:
            continue
        if len(line.split(" ")) <= 3:
            current_summary_section = None
            continue
        dict_section_lines[current_summary_section].append(line)

    if analysis_end_line is None:
        if tail_lines is None and isinstance(lines, list):
            tail_lines = lines[-16:]
        for line in tail_lines or ():
            if "Analysis ended on" in line:
                analysis_end_line = line

    return (
        dict_section_lines,
        line_flw_units,
        runoff_continuity_error_line,
        flow_continuity_error_line,
        system_flood_loss_line,
        analysis_end_line,
        valid,
    )


# --- The report tables beyond the first three (depth, surcharge, storage, outfall, flow class, pumps, ranked lists) --
#
# Every table prints an id with `%-20s` and then a FIXED number of cells (`statsrpt.c`, one `write*` function per
# table), and SWMM ids contain no whitespace, so each row is split on whitespace and read by token position. A row
# whose token count is not the table's fixed count is REFUSED (naming the table, the report and the row) rather than
# read with a shifted column: a shifted column is a wrong number, and a refusal is not. Deliberately NOT
# `format_rpt_section_into_dataframe`: its colon rule joins any token containing ':' onto the previous cell, which
# misreads an id containing ':' and has no count check.


class _TableSpec(NamedTuple):
    """One report table: its title, the element it describes, its fixed token count, and its stored columns."""

    title: str
    kind: str
    n_tokens: int
    names: list[str]
    slots: tuple


#: `slots` aligns one entry with every name after the id: a token index, or a ``(days, clock)`` pair of token
#: indexes for a ``days hr:min`` cell, which is stored as minutes. Token 0 is the id. Positions are the `fprintf`
#: formats of `statsrpt.c`; `tests/test_swmm_output_inventory.py` holds each against a printed row.
_TABLE_SPECS: tuple[_TableSpec, ...] = (
    # id, type, avg depth, max depth, max HGL, days, clock, reported max depth (left out)
    _TableSpec("Node Depth Summary", "node", 8, LST_COL_HEADERS_NODE_DEPTH_SUMMARY, (2, 3, 4, (5, 6))),
    # id, type, hours surcharged, height above crown, depth below rim
    _TableSpec("Node Surcharge Summary", "node", 5, LST_COL_HEADERS_NODE_SURCHARGE_SUMMARY, (2, 3, 4)),
    # id, avg volume, avg pct, evap pct, exfil pct, max volume, max pct, days, clock, max outflow
    _TableSpec(
        "Storage Volume Summary",
        "node",
        10,
        LST_COL_HEADERS_STORAGE_VOLUME_SUMMARY,
        (1, 2, 3, 4, 5, 6, (7, 8), 9),
    ),
    # id, adjusted/actual length, seven flow classes, normal-flow-limited, inlet-controlled
    _TableSpec(
        "Flow Classification Summary",
        "link",
        11,
        LST_COL_HEADERS_FLOW_CLASSIFICATION_SUMMARY,
        tuple(range(1, 11)),
    ),
    # id, hours full both ends / upstream / downstream, above full normal flow, capacity limited
    _TableSpec("Conduit Surcharge Summary", "link", 6, LST_COL_HEADERS_CONDUIT_SURCHARGE_SUMMARY, (1, 2, 3, 4, 5)),
    # id, utilized, start-ups, min flow (left out), avg flow, max flow (left out), volume, energy, off-curve low/high
    _TableSpec("Pumping Summary", "link", 10, LST_COL_HEADERS_PUMPING_SUMMARY, (1, 2, 4, 6, 7, 8, 9)),
)
_TABLE_SPEC_BY_TITLE = {spec.title: spec for spec in _TABLE_SPECS}
#: Outfall Loading Summary: id, flow frequency, average flow, maximum flow, total volume, then one load per pollutant.
_OUTFALL_FIXED_TOKENS = 5
#: The header rule is 59 dashes plus 14 per pollutant column (`statsrpt.c`, `writeOutfallLoads`).
_OUTFALL_RULE_BASE = 59
_OUTFALL_RULE_PER_POLLUTANT = 14


def _refuse_table_row(filepath, title, row, n_expected):
    """Raise the ProcessingError for a report table row whose token count is wrong."""
    from hhemt.exceptions import ProcessingError

    raise ProcessingError(
        operation="read_report_table",
        filepath=Path(filepath),
        reason=(
            f"a {title} row ({row[0] if row else '<blank>'!r}) has {len(row)} field(s); {n_expected} were expected, "
            "so a column would be misread."
        ),
    )


def _float_token(token, filepath, title):
    """A printed number as a float, refusing an unreadable one by table."""
    try:
        return float(token)
    except ValueError:
        from hhemt.exceptions import ProcessingError

        raise ProcessingError(
            operation="read_report_table",
            filepath=Path(filepath),
            reason=f"a {title} cell {token!r} is not a number.",
        ) from None


def _clock_to_minutes(days_token, clock_token):
    """Minutes for a printed ``days hr:min`` pair (the report's time-of-occurrence cell), NaN when unreadable.

    The same arithmetic as `convert_swmm_tdeltas_to_minutes`, done on the tokens directly: that function builds a
    pandas Series per call, which costs milliseconds a row on a table of a thousand rows.
    """
    hours, _, minutes = clock_token.partition(":")
    try:
        return float(int(days_token) * 1440 + int(hours) * 60 + int(minutes))
    except ValueError:
        return float("nan")


def read_report_table(title: str, section_lines, filepath="<rpt>") -> pd.DataFrame:
    """One of the dashed tables of `_TABLE_SPECS` as a float frame indexed by its element id.

    ``section_lines`` are the data lines `_scan_metadata_and_summaries` collected for ``title``. Cells are read by
    token position; a row of the wrong token count is refused. A table the report did not print (no lines) yields
    an empty frame with the table's columns, which the caller's outer join turns into NaN for every element.
    """
    spec = _TABLE_SPEC_BY_TITLE[title]
    id_name = spec.names[0]
    rows = []
    for line in section_lines:
        tokens = line.split()
        if not tokens:
            continue
        if len(tokens) != spec.n_tokens:
            _refuse_table_row(filepath, title, tokens, spec.n_tokens)
        record: dict[str, Any] = {id_name: tokens[0]}  # the id exactly as printed, which is the binary output's key
        for name, slot in zip(spec.names[1:], spec.slots, strict=True):
            if isinstance(slot, tuple):
                _float_token(tokens[slot[0]], filepath, title)  # the days cell must be a number
                record[name] = _clock_to_minutes(tokens[slot[0]], tokens[slot[1]])
            else:
                record[name] = _float_token(tokens[slot], filepath, title)
        rows.append(record)
    frame = pd.DataFrame.from_records(rows, columns=spec.names)
    return frame.set_index(id_name)


def outfall_pollutant_columns(head_lines) -> int:
    """The number of pollutant columns the Outfall Loading Summary's header prints (zero when it prints none).

    ``writeOutfallLoads`` prints one column per DECLARED pollutant whether or not quality is routed, so the model
    file's ``[POLLUTANTS]`` decides, not the ``Water Quality`` option. The count is read from the header rule's
    width (59 dashes plus 14 per pollutant), which is the report's own statement of it.
    """
    in_section = False
    for line in head_lines:
        if not in_section:
            in_section = "Outfall Loading Summary" in line
            continue
        stripped = line.strip()
        if stripped.startswith("---"):
            extra = len(stripped) - _OUTFALL_RULE_BASE
            if extra < 0 or extra % _OUTFALL_RULE_PER_POLLUTANT:
                from hhemt.exceptions import ProcessingError

                raise ProcessingError(
                    operation="read_report_table",
                    filepath=Path("<rpt>"),
                    reason=(
                        f"the Outfall Loading Summary header rule is {len(stripped)} characters wide, which is not "
                        f"{_OUTFALL_RULE_BASE} plus a whole number of {_OUTFALL_RULE_PER_POLLUTANT}-character "
                        "pollutant columns."
                    ),
                )
            return extra // _OUTFALL_RULE_PER_POLLUTANT
    return 0


def read_outfall_loading_summary(section_lines, n_pollutants: int, filepath="<rpt>"):
    """The Outfall Loading Summary's flow columns and its System row's maximum flow.

    Returns ``(frame, system_max_flow)``: a float frame indexed by outfall id with the four stored flow columns,
    and the System row's Max Flow cell (NaN when the report printed no System row). Each row has
    ``5 + n_pollutants`` tokens, the System row included; the pollutant loads are read by the quality path from
    the same rows and are not read here. The cell kept from the System row is the maximum over steps of the SUM of
    outfall flows, which no per-outfall column gives.
    """
    title = "Outfall Loading Summary"
    names = LST_COL_HEADERS_OUTFALL_LOADING_SUMMARY
    n_tokens = _OUTFALL_FIXED_TOKENS + n_pollutants
    rows = []
    system_max_flow = float("nan")
    for line in section_lines:
        tokens = line.split()
        if not tokens:
            continue
        if len(tokens) != n_tokens:
            _refuse_table_row(filepath, title, tokens, n_tokens)
        if tokens[0] == "System":
            system_max_flow = _float_token(tokens[3], filepath, title)
            continue
        rows.append(
            {
                names[0]: tokens[0],
                names[1]: _float_token(tokens[1], filepath, title),
                names[2]: _float_token(tokens[2], filepath, title),
                names[3]: _float_token(tokens[3], filepath, title),
                names[4]: _float_token(tokens[4], filepath, title),
            }
        )
    return pd.DataFrame.from_records(rows, columns=names).set_index(names[0]), system_max_flow


def _read_new_report_tables(dict_section_lines, head_lines, filepath):
    """Frames for every table beyond the first three: ``(node frames, link frames, outfall system max flow)``."""
    node_frames, link_frames = [], []
    for spec in _TABLE_SPECS:
        frame = read_report_table(spec.title, dict_section_lines.get(spec.title, []), filepath)
        (node_frames if spec.kind == "node" else link_frames).append(frame)
    n_pollutants = outfall_pollutant_columns(head_lines) if head_lines is not None else 0
    outfall_frame, system_max_flow = read_outfall_loading_summary(
        dict_section_lines.get("Outfall Loading Summary", []), n_pollutants, filepath
    )
    node_frames.append(outfall_frame)
    return node_frames, link_frames, system_max_flow


# --- Run-level and ranked measures: data variables, not attributes ------------------------------------------------

_RE_ROUTING_STEP_LABELS = {
    "Minimum Time Step": "routing_step_smallest",
    "Average Time Step": "routing_step_mean",
    "Maximum Time Step": "routing_step_largest",
    "% of Time in Steady State": "routing_steady_state_time",
    "Average Iterations per Step": "routing_iterations_mean",
    "% of Steps Not Converging": "routing_nonconverging_steps",
}
_RE_STEP_FREQUENCY = re.compile(r"^\s*(?P<upper>[\d.]+)\s*-\s*(?P<lower>[\d.]+)\s*sec\s*:\s*(?P<pct>[-\d.]+)\s*%")
#: The Routing Time Step Summary prints this many step-size bins (`report_RouteStepFreq`: `TIMELEVELS` = 6 edges).
_N_ROUTING_STEP_BINS = 5
_RE_RANKED = re.compile(r"^\s*(?P<kind>Node|Link)\s+(?P<id>\S+)\s+\((?P<value>-?[\d.]+)%?\)\s*$")


def read_flow_routing_continuity(section_lines, filepath="<rpt>") -> dict[str, float]:
    """The ten Flow Routing Continuity rows the toolkit did not keep, from their ``10^6 L`` column.

    Each row prints a hectare-meter figure and then a ``10^6 L`` figure; the second is read. A row the report does
    not print is NaN; a row with any other number of figures is refused.
    """
    from hhemt.exceptions import ProcessingError

    found: dict[str, float] = {}
    for line in section_lines:
        match = _RE_CONTINUITY_ROW.match(line.rstrip("\r\n"))
        if match is None:
            continue
        label = match.group("label").strip()
        stem = FLOW_ROUTING_CONTINUITY_TERMS.get(label)
        if stem is None:
            continue
        values = match.group("vals").split()
        if len(values) != 2:
            raise ProcessingError(
                operation="read_report_table",
                filepath=Path(filepath),
                reason=f"the Flow Routing Continuity row {label!r} has {len(values)} figure(s); 2 were expected.",
            )
        found[stem] = float(values[1])
    return {
        f"routing_continuity_{stem}": found.get(stem, float("nan")) for stem in FLOW_ROUTING_CONTINUITY_TERMS.values()
    }


def read_runoff_quantity_continuity(section_lines, filepath="<rpt>") -> dict[str, float]:
    """The Runoff Quantity Continuity rows the toolkit keeps, from their depth column.

    Each row prints a ``hectare-m`` volume and then a ``mm`` depth; the second is read. A row the report does not
    print is NaN, which is also the whole answer for a model that routes no runoff (its table prints one sentence
    in place of the rows); a row with any other number of figures is refused.
    """
    from hhemt.exceptions import ProcessingError

    found: dict[str, float] = {}
    for line in section_lines:
        match = _RE_CONTINUITY_ROW.match(line.rstrip("\r\n"))
        if match is None:
            continue
        label = match.group("label").strip()
        stem = RUNOFF_QUANTITY_CONTINUITY_TERMS.get(label)
        if stem is None:
            continue
        values = match.group("vals").split()
        if len(values) != 2:
            raise ProcessingError(
                operation="read_report_table",
                filepath=Path(filepath),
                reason=f"the Runoff Quantity Continuity row {label!r} has {len(values)} figure(s); 2 were expected.",
            )
        found[stem] = float(values[1])
    return {
        f"runoff_continuity_{stem}": found.get(stem, float("nan")) for stem in RUNOFF_QUANTITY_CONTINUITY_TERMS.values()
    }


def read_routing_time_step_summary(section_lines, filepath="<rpt>"):
    """The Routing Time Step Summary: ``(scalars, upper bounds, lower bounds, frequencies)``.

    The scalars are keyed by stored name. The frequency table prints ``upper - lower sec : pct %`` for five
    step-size bins; when the report prints none (it does so only for Dynamic Wave routing with a Courant factor)
    the three arrays are NaN. A bin count other than five is refused.
    """
    from hhemt.exceptions import ProcessingError

    scalars = dict.fromkeys(_RE_ROUTING_STEP_LABELS.values(), float("nan"))
    uppers, lowers, freqs = [], [], []
    for line in section_lines:
        freq = _RE_STEP_FREQUENCY.match(line)
        if freq is not None:
            uppers.append(float(freq.group("upper")))
            lowers.append(float(freq.group("lower")))
            freqs.append(float(freq.group("pct")))
            continue
        label, sep, rest = line.partition(":")
        name = _RE_ROUTING_STEP_LABELS.get(label.strip())
        if name is not None and sep:
            scalars[name] = float(rest.split()[0])
    if freqs and len(freqs) != _N_ROUTING_STEP_BINS:
        raise ProcessingError(
            operation="read_report_table",
            filepath=Path(filepath),
            reason=(
                f"the Routing Time Step Summary prints {len(freqs)} frequency bin(s); "
                f"{_N_ROUTING_STEP_BINS} were expected."
            ),
        )
    nan = [float("nan")] * _N_ROUTING_STEP_BINS
    return scalars, np.array(uppers or nan), np.array(lowers or nan), np.array(freqs or nan)


def read_ranked_list(section_lines, *, filepath="<rpt>", title="ranked list"):
    """One of SWMM's lists of at most five elements as ``(kinds, ids, values)``, each padded to five.

    A line is ``Node|Link {id} ({value})``. The all-clear sentences (``All links are stable.``, ``Convergence
    obtained at all time steps.``) and ``None`` yield an EMPTY list, which is read as all-NaN and never as zero:
    SWMM prints the sentence whenever the top element's INDEX is 0, which is also the case of a nonzero value at the
    first node or link of the file, so the sentence does not prove the quantity was zero. An unprinted rank holds the
    empty string for a kind or an id and NaN for a value.
    """
    from hhemt.exceptions import ProcessingError

    kinds, ids, values = [], [], []
    for line in section_lines:
        stripped = line.strip()
        if not stripped or stripped in ("None", "All links are stable.", "Convergence obtained at all time steps."):
            continue
        match = _RE_RANKED.match(line)
        if match is None:
            raise ProcessingError(
                operation="read_report_table",
                filepath=Path(filepath),
                reason=f"a {title} line {stripped!r} is not 'Node|Link id (value)'.",
            )
        kinds.append(match.group("kind").lower())
        ids.append(match.group("id"))
        values.append(float(match.group("value")))
    if len(ids) > SWMM_STAT_RANK_LENGTH:
        raise ProcessingError(
            operation="read_report_table",
            filepath=Path(filepath),
            reason=f"a {title} prints {len(ids)} element(s); SWMM prints at most {SWMM_STAT_RANK_LENGTH}.",
        )
    pad = SWMM_STAT_RANK_LENGTH - len(ids)
    return (
        np.array(kinds + [""] * pad, dtype=str),
        np.array(ids + [""] * pad, dtype=str),
        np.array(values + [float("nan")] * pad, dtype=float),
    )


def _run_level_datasets(
    dict_section_lines, dict_system_results, outfall_system_max_flow, filepath, *, has_report, family="tritonswmm"
):
    """The run-level and ranked variables as ``(node dataset, link dataset)``.

    The flow instability list names links and goes to the link dataset; every other variable goes to the node
    dataset. A run with no report yields two empty datasets. A table the report did not print yields NaN (or the
    empty string for an id) for every variable it feeds. The Runoff Quantity Continuity rows are read for the
    standalone SWMM family alone (``family == "swmm"``), because the coupled hydraulics model routes no runoff.
    """
    if not has_report:
        return xr.Dataset(), xr.Dataset()
    nan = float("nan")
    node_vars: dict[str, Any] = {}
    for name in LST_RUN_LEVEL_CONTINUITY_VARIABLES:
        node_vars[name] = ((), float(dict_system_results.get(name, nan)))
    if family == "swmm":
        for name, value in read_runoff_quantity_continuity(
            dict_section_lines.get("Runoff Quantity Continuity", []), filepath
        ).items():
            node_vars[name] = ((), value)
    for name, value in read_flow_routing_continuity(
        dict_section_lines.get("Flow Routing Continuity", []), filepath
    ).items():
        node_vars[name] = ((), value)
    scalars, uppers, lowers, freqs = read_routing_time_step_summary(
        dict_section_lines.get("Routing Time Step Summary", []), filepath
    )
    for name, value in scalars.items():
        node_vars[name] = ((), value)
    node_vars[ROUTING_STEP_FREQUENCY_VARIABLE] = (("routing_step_bin",), freqs)
    node_vars[LST_OUTFALL_SYSTEM_VARIABLES[0]] = ((), float(outfall_system_max_flow))

    rank = ("swmm_stat_rank",)
    _kinds, ids, values = read_ranked_list(
        dict_section_lines.get("Most Frequent Nonconverging Nodes", []),
        filepath=filepath,
        title="Most Frequent Nonconverging Nodes list",
    )
    node_vars["nonconverging_node_id"] = (rank, ids)
    node_vars["nonconverging_node_frequency"] = (rank, values)
    kinds, ids, values = read_ranked_list(
        dict_section_lines.get("Time-Step Critical Elements", []),
        filepath=filepath,
        title="Time-Step Critical Elements list",
    )
    node_vars["courant_critical_element_id"] = (rank, ids)
    node_vars["courant_critical_element_kind"] = (rank, kinds)
    node_vars["courant_critical_frequency"] = (rank, values)
    _kinds, ids, values = read_ranked_list(
        dict_section_lines.get("Highest Continuity Errors", []),
        filepath=filepath,
        title="Highest Continuity Errors list",
    )
    node_vars["continuity_error_node_id"] = (rank, ids)
    node_vars["continuity_error_node_relative"] = (rank, values)
    ds_node_run = xr.Dataset(
        node_vars,
        coords={
            "routing_step_bin": np.arange(_N_ROUTING_STEP_BINS),
            "routing_step_bin_upper": ("routing_step_bin", uppers),
            "routing_step_bin_lower": ("routing_step_bin", lowers),
            "swmm_stat_rank": np.arange(SWMM_STAT_RANK_LENGTH),
        },
    )
    _kinds, ids, values = read_ranked_list(
        dict_section_lines.get("Highest Flow Instability Indexes", []),
        filepath=filepath,
        title="Highest Flow Instability Indexes list",
    )
    ds_link_run = xr.Dataset(
        {"flow_instability_link_id": (rank, ids), "flow_instability_index": (rank, values)},
        coords={"swmm_stat_rank": np.arange(SWMM_STAT_RANK_LENGTH)},
    )
    return ds_node_run, ds_link_run


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
        runoff_continuity_error = float(runoff_continuity_error_line.split(" ")[-1].split("\n")[0])
    else:
        runoff_continuity_error = np.nan
    flow_continuity_error = float(flow_continuity_error_line.split(" ")[-1].split("\n")[0])
    system_flooding = float(system_flood_loss_line.split(" ")[-1].split("\n")[0])
    analysis_end_datetime = _parse_analysis_end_line(analysis_end_line)
    dict_system_results = dict(
        analysis_end_datetime=str(analysis_end_datetime),
        flow_units=flow_units,
        runoff_continuity_error=runoff_continuity_error,
        runoff_continuity_error_units="percent",
        flow_continuity_error=flow_continuity_error,
        flow_continuity_error_units="percent",
        system_flooding=system_flooding,
        system_flooding_units="10^6 L",
    )
    return dict_system_results


#: Column headers for the Subcatchment Runoff Summary section, in file order.
#: MEASURED against a real Norfolk hydrology rpt (2026-08-17): the section header
#: spans three physical lines and the data rows are fixed-width, so this tuple is
#: authored against a real file, never inferred from the header text.
LST_COL_HEADERS_SUBCATCH_RUNOFF_SUMMARY = (
    "subcatchment_id",
    "total_precip",
    "total_runon",
    "total_evap",
    "total_infil",
    "imperv_runoff",
    "perv_runoff",
    "total_runoff_depth",
    "total_runoff_volume",
    "peak_runoff",
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
    df = format_rpt_section_into_dataframe(
        lst_rows, list(LST_COL_HEADERS_SUBCATCH_RUNOFF_SUMMARY), section_name="Subcatchment Runoff Summary"
    )
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


# Width of each cell of a Link Flow Summary row, counted from the END of the id field. SWMM prints the id
# with `%-20s`, which pads short ids to 20 characters and leaves longer ones whole, so the id field is
# `max(20, len(id))` wide and every later cell sits at a fixed offset from its end: type 9 characters,
# largest flow 9 (`%9.3f` or `%9.2f`), time of occurrence 13 (`  %4d  %02d:%02d`), velocity 10
# (`   %7.2f` or `    >50.00`), flow ratio 8 and depth ratio 8 (`  %6.2f`).
_LINK_FLOW_CELL_WIDTHS = (9, 9, 13, 10, 8, 8)


def format_link_flow_summary(lst_section_lines, lst_col_headers):
    """Read the Link Flow Summary into a frame by fixed columns measured from the end of the id field.

    Parameters
    ----------
    lst_section_lines : list of str
        The section's data rows as `return_lines_for_section_of_rpt` returns them.
    lst_col_headers : list of str
        The seven column names, in print order (`LST_COL_HEADERS_LINK_FLOW_SUMMARY`).

    Returns
    -------
    pandas.DataFrame
        One row per link; a cell the row's link type does not print, including trailing cells a row
        ends before, is the empty string.

    Notes
    -----
    Fixed columns rather than whitespace tokens, because the tokens of different link types do not line
    up: a pump with a curve prints its flow ratio under the velocity cell's neighbour, a weir prints only
    a depth ratio, an outlet or dummy link stops after the time cell, and an id containing a colon is
    one token that the shared whitespace reader would join to the previous cell. Anchoring at the left
    edge instead of the end of the id field would shift every cell of a link whose id is longer than 20
    characters (`statsrpt.c` pads the id but does not truncate it); anchoring at the right edge would put a
    pump's ratio in the depth column, because rows end at different cells by type.
    """
    records = []
    for raw_line in lst_section_lines:
        body = raw_line.rstrip("\r\n")[2:]
        if not body.strip():
            continue
        link_id = body.split(None, 1)[0]
        id_field_width = max(20, len(link_id))
        cursor = id_field_width
        cells = []
        for width in _LINK_FLOW_CELL_WIDTHS:
            cells.append(" ".join(body[cursor : cursor + width].split()))
            cursor += width
        records.append([link_id, *cells])
    return pd.DataFrame.from_records(records, columns=lst_col_headers)


@profile
def format_rpt_section_into_dataframe(lst_section_lines, lst_col_headers, section_name=None):
    if len(lst_section_lines) == 0:
        return pd.DataFrame(columns=lst_col_headers)
    dict_line_contents_aslist = return_data_from_rpt(lst_section_lines, section_name=section_name)
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


def _out_series_values(out, kind: str, ident: str, index: int, attribute) -> np.ndarray:
    """One element's series over EVERY report period, read straight from ``swmm.toolkit.output``.

    ``kind`` is ``"node"`` or ``"link"``; ``ident`` is the element's id (used only in the refusal message and by
    test stand-ins) and ``index`` its position in the output; ``attribute`` is a pyswmm attribute name
    (``"invert_depth"``) or a toolkit attribute index.

    The read does not go through pyswmm's ``NodeSeries`` / ``Output.node_series``
    because pyswmm 2.1.0 resolves their default end to the INCLUSIVE index ``num_periods - 1`` and hands it to the
    toolkit's END-EXCLUSIVE ``get_node_series``, so the final report period is silently dropped (the same file
    reads whole under pyswmm 2.0.1). The window ``(0, num_periods)`` is passed here explicitly, so the result does
    not depend on how any pyswmm version resolves a default end. A series shorter or longer than the output's
    period count raises, so a future accessor change cannot truncate the stores quietly.
    """
    from swmm.toolkit import output as swmm_output
    from swmm.toolkit import shared_enum

    from hhemt.exceptions import ProcessingError

    n_periods = int(swmm_output.get_times(out.handle, shared_enum.Time.NUM_PERIODS))
    group = shared_enum.NodeAttribute if kind == "node" else shared_enum.LinkAttribute
    attr = getattr(group, attribute.upper()) if isinstance(attribute, str) else attribute
    reader = swmm_output.get_node_series if kind == "node" else swmm_output.get_link_series
    values = np.asarray(reader(out.handle, index, attr, 0, n_periods), dtype=float)
    if len(values) != n_periods:
        raise ProcessingError(
            operation="read SWMM output series",
            filepath=Path(getattr(out, "binfile", "") or "."),
            reason=f"the {kind} series of {ident!r} has {len(values)} values for {n_periods} report periods.",
        )
    return values


def _out_time_axis(out, f_outfile) -> list:
    """The report times of ``out`` (``Output.times``), refused unless there is one per ``NUM_PERIODS`` period.

    The count check ties the axis to the same period count the series are read over, so a time axis one step
    shorter than the series (or the reverse) cannot be paired with them.
    """
    from swmm.toolkit import output as swmm_output
    from swmm.toolkit import shared_enum

    from hhemt.exceptions import ProcessingError

    times = list(out.times)
    n_periods = int(swmm_output.get_times(out.handle, shared_enum.Time.NUM_PERIODS))
    if len(times) != n_periods:
        raise ProcessingError(
            operation="return_node_time_series_results_from_outfile",
            filepath=Path(f_outfile),
            reason=f"the output's time axis has {len(times)} steps for {n_periods} report periods.",
        )
    return times


def _out_series_dataset(out, kind, entity_index, spec, idx_colname, f_outfile) -> xr.Dataset:
    """One dataset of ``(idx, date_time)`` series for every entity, read over every report period.

    ``entity_index`` maps entity id to its index in the output (pyswmm's ``Output.nodes`` or ``Output.links``);
    ``spec`` is ``(stored name, pyswmm attribute)`` pairs. Every entity of a SWMM output shares the run's report
    steps, so each variable is stacked into one array instead of being built entity by entity as a frame, which is
    what made the read cost scale with entities times variables. The time axis comes from ``_out_time_axis``. The
    ids and the steps come out sorted, the order a MultiIndex ``to_xarray`` gave them, so stores written before
    this reader existed compare equal.
    """
    entity_ids = list(entity_index.keys())
    ids = [str(i) for i in entity_ids]
    if not ids:
        return xr.Dataset()
    times = _out_time_axis(out, f_outfile)
    id_order = np.argsort(np.array(ids, dtype=str), kind="stable")
    time_index = pd.DatetimeIndex(times)
    time_order = np.argsort(time_index.values, kind="stable")
    data_vars = {
        name: (
            (idx_colname, "date_time"),
            np.array([_out_series_values(out, kind, ident, entity_index[ident], attribute) for ident in entity_ids])[
                id_order
            ][:, time_order],
        )
        for name, attribute in spec
    }
    return xr.Dataset(
        data_vars,
        coords={idx_colname: np.array(ids, dtype=str)[id_order], "date_time": time_index[time_order]},
    )


#: pyswmm's per-element series for the NODE variables of SWMM's binary output (`enums.h`: NODE_DEPTH, NODE_HEAD,
#: NODE_INFLOW, NODE_OVERFLOW, then NODE_VOLUME and NODE_LATFLOW), as (stored name, pyswmm
#: attribute). `ponded_volume` is pyswmm's name for NODE_VOLUME, which holds the node's WHOLE stored volume, ponded
#: volume included, so the stored name is the plain `volume`.
_OUT_NODE_SERIES = (
    ("depth", "invert_depth"),
    ("head", "hydraulic_head"),
    ("inflow_flow", "total_inflow"),
    ("flooding", "flooding_losses"),
    ("volume", "ponded_volume"),
    ("lateral_inflow", "lateral_inflow"),
)
#: The LINK variables (LINK_FLOW, LINK_VELOCITY, LINK_DEPTH, LINK_CAPACITY, LINK_VOLUME). The capacity series takes
#: its stored name from the caller, because the two model families name it differently (see below).
_OUT_LINK_SERIES = (
    ("flow", "flow_rate"),
    ("velocity", "flow_velocity"),
    ("link_depth", "flow_depth"),
    (None, "capacity"),
    ("link_volume", "flow_volume"),
)


def _require_cms_output(units, f_outfile, operation: str) -> None:
    """Refuse a binary output whose flow unit is not CMS, naming the file and the operation that read it.

    The FLOW unit, not the unit system: every series is stored as m3/s with no conversion, and SWMM writes them in
    the model's own flow unit, so an SI model in LPS or MLD would pass a unit-system test and be read 1000 or 86.4
    times too large. This is the standalone model's guard for outputs the preparation check never saw, such as a
    reprocess of an existing analysis. It RAISES, never ``sys.exit``: SystemExit derives from BaseException, so no
    ``except Exception`` in any caller catches it and the process dies with no traceback -- the un-diagnosable shape
    that made an unrelated 123-scenario failure take a full session to attribute. ``ProcessingError`` is this
    module's own family (CLI exit 5) and carries the file.
    """
    if units["flow"] == "CMS":
        return
    from hhemt.exceptions import ProcessingError

    raise ProcessingError(
        operation=operation,
        filepath=Path(f_outfile),
        reason=(
            f"SWMM outputs are not in cubic meters per second (flow unit reported: "
            f"{units['flow']!r}, unit system {units['system']!r}). The toolkit reads the "
            "standalone model's flows as m3/s without conversion, so its [OPTIONS] "
            "FLOW_UNITS must be CMS; an ABSENT FLOW_UNITS line defaults to CFS. Fix it "
            "in the SWMM template named by system_config, not in the generated "
            "per-scenario .inp, which inherits it."
        ),
    )


def return_node_time_series_results_from_outfile(f_outfile, f_inp=None, return_quality=False, capacity_name="capacity"):
    """Read node and link time series from a SWMM ``.out`` file.

    This is the one series reader of both model families. Node series: ``depth``, ``head``, ``inflow_flow``,
    ``flooding``, ``volume`` (the node's stored volume, ponded volume included) and ``lateral_inflow``. Link series:
    ``flow``, ``velocity``, ``link_depth``, the capacity series and ``link_volume``. ``capacity_name`` is
    ``"capacity_setting"`` for the coupled family and ``"capacity"`` for standalone SWMM: for a conduit the series is
    the filled fraction of the section, and for a pump, orifice, weir or outlet it is the control setting.

    With ``return_quality`` the function returns ``(node ds, link ds, pollutants, start)`` where
    ``pollutants`` is the model's ``[(name, unit)]`` list when the ``.out`` holds pollutant results
    (its pollutant count is above zero, which SWMM makes zero whenever quality is ignored) and
    ``[]`` otherwise, and ``start`` is the simulation start as ``MM/DD/YYYY HH:MM:SS``. Under that
    gate the datasets carry ``pollutant_conc`` and ``link_pollutant_conc`` read from attribute index
    ``6 + p`` (node) and ``5 + p`` (link), with names and units taken from ``f_inp``'s
    ``[POLLUTANTS]`` section and never from the ``.out``'s unit flags, which report only the
    first pollutant's unit. Without ``return_quality`` the two datasets are returned alone.
    """
    from pyswmm import Output

    pollutants: list[tuple[str, str]] = []
    out_start = None
    with Output(str(f_outfile)) as out:
        d_links = out.links
        d_nodes = out.nodes
        units = out.units
        _require_cms_output(units, f_outfile, "return_node_time_series_results_from_outfile")
        if not d_nodes or not d_links:
            from hhemt.exceptions import ProcessingError

            raise ProcessingError(
                operation="return_node_time_series_results_from_outfile",
                filepath=Path(f_outfile),
                reason=(
                    f"the binary output holds {len(d_nodes)} node(s) and {len(d_links)} link(s); SWMM writes series "
                    "only "
                    "for objects the model's [REPORT] section lists, so add `NODES ALL` and `LINKS ALL` to [REPORT] "
                    "in the SWMM template named by system_config."
                ),
            )
        # PROCESSING NODES AND LINKS
        ds_node_tseries = _out_series_dataset(out, "node", d_nodes, _OUT_NODE_SERIES, "node_id", f_outfile)
        link_series_spec = tuple((capacity_name if name is None else name, attr) for name, attr in _OUT_LINK_SERIES)
        ds_link_tseries = _out_series_dataset(out, "link", d_links, link_series_spec, "link_id", f_outfile)
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
            ds_node_tseries["pollutant_conc"] = _out_pollutant_series(out, "node", d_nodes, 6, "node_id", pollutants)
            ds_link_tseries["link_pollutant_conc"] = _out_pollutant_series(
                out, "link", d_links, 5, "link_id", pollutants
            )
        if return_quality:
            start = getattr(out, "start", None)
            out_start = start.strftime(RPT_DATETIME_FORMAT) if start is not None else None
    #
    if return_quality:
        return ds_node_tseries, ds_link_tseries, pollutants, out_start
    return ds_node_tseries, ds_link_tseries


def _out_system_series_values(out, attribute_name: str) -> np.ndarray:
    """One system series over EVERY report period, read straight from ``swmm.toolkit.output``.

    ``attribute_name`` is a member of ``swmm.toolkit.shared_enum.SystemAttribute`` (``"FLOOD_LOSSES"``). The read
    passes the window ``(0, num_periods)`` itself, for the reason ``_out_series_values`` gives (the period rule of
    E43): pyswmm 2.1.0 resolves a default end to the inclusive index ``num_periods - 1`` and its toolkit call is
    END-EXCLUSIVE, which drops the final report period. A series of any other length raises.
    """
    from swmm.toolkit import output as swmm_output
    from swmm.toolkit import shared_enum

    from hhemt.exceptions import ProcessingError

    n_periods = int(swmm_output.get_times(out.handle, shared_enum.Time.NUM_PERIODS))
    attribute = getattr(shared_enum.SystemAttribute, attribute_name)
    values = np.asarray(swmm_output.get_system_series(out.handle, attribute, 0, n_periods), dtype=float)
    if len(values) != n_periods:
        raise ProcessingError(
            operation="read SWMM output series",
            filepath=Path(getattr(out, "binfile", "") or "."),
            reason=f"the system series {attribute_name!r} has {len(values)} values for {n_periods} report periods.",
        )
    return values


def return_system_time_series_results_from_outfile(f_outfile) -> xr.Dataset:
    """The six system series of a standalone SWMM ``.out`` as one dataset on ``date_time``.

    The variables are ``system_flooding_rate`` (water lost to flooding, m3 s-1), ``system_outfall_flow`` (m3 s-1),
    ``system_stored_volume`` (m3), ``system_lateral_inflow`` (m3 s-1), ``system_runoff`` (m3 s-1) and
    ``system_rainfall`` (area-weighted mean intensity, mm h-1), read over every report period of the file. Only the
    standalone SWMM family stores them: the coupled hydraulics model routes no runoff and holds flooded water as
    ponded volume, so its system flooding is about zero. A binary output whose flow unit is not CMS is refused.
    """
    from pyswmm import Output

    with Output(str(f_outfile)) as out:
        _require_cms_output(out.units, f_outfile, "return_system_time_series_results_from_outfile")
        times = _out_time_axis(out, f_outfile)
        time_index = pd.DatetimeIndex(times)
        time_order = np.argsort(time_index.values, kind="stable")
        data_vars = {
            name: (("date_time",), _out_system_series_values(out, attribute_name)[time_order])
            for name, attribute_name in SWMM_SYSTEM_SERIES
        }
    return xr.Dataset(data_vars, coords={"date_time": time_index[time_order]})


def _out_pollutant_series(out, kind, entity_index, first_attribute, idx_colname, pollutants):
    """Concentration series for every entity and pollutant as (idx, date_time, pollutant).

    Pollutant ``p`` is toolkit attribute ``first_attribute + p``; every series is read over every report period
    by ``_out_series_values`` (see its note on pyswmm 2.1.0's dropped final period).
    """
    idx = pd.DatetimeIndex(_out_time_axis(out, "the SWMM binary output"), name="date_time")
    frames = []
    for p, (name, _) in enumerate(pollutants):
        rows = []
        for entity_id, position in entity_index.items():
            values = _out_series_values(out, kind, entity_id, position, first_attribute + p)
            rows.append(
                pd.Series(
                    values,
                    index=pd.MultiIndex.from_arrays(
                        [np.repeat(entity_id, len(idx)), idx], names=[idx_colname, "date_time"]
                    ),
                ).rename(name)
            )
        frames.append(pd.concat(rows))
    df = pd.concat(frames, axis=1)
    da = df.to_xarray().to_dataarray(dim="pollutant").transpose(idx_colname, "date_time", "pollutant")
    return da.assign_coords(pollutant=[name for name, _ in pollutants]).astype(float)


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
def return_data_from_rpt(lst_section_lines, section_name=None):
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
                    f"WARNING: the parser cannot interpret rows of the SWMM report section "
                    f"{section_name if section_name is not None else '(unnamed)'!r}"
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
