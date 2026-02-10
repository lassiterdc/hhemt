import sys
import re
import warnings
from collections import Counter
from functools import lru_cache
from typing import Any, cast
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr


try:  # pragma: no cover - used by line_profiler
    profile  # type: ignore[name-defined]
except NameError:  # pragma: no cover

    def profile(func):  # type: ignore[no-redef]
        return func


from TRITON_SWMM_toolkit.constants import (
    LST_COL_HEADERS_NODE_FLOOD_SUMMARY,
    LST_COL_HEADERS_NODE_FLOW_SUMMARY,
    LST_COL_HEADERS_LINK_FLOW_SUMMARY,
)


TDELTA_PATTERN = re.compile(r"^\s*(\d+)\s+(\d+):(\d+)")
RPT_DATETIME_FORMAT = "%m/%d/%Y %H:%M:%S"


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
        raise ValueError("rpt could not be found")

    decode_errors: list[str] = []
    content: str | None = None
    attempted_encodings = ("utf-8", "cp1252")

    for encoding in attempted_encodings:
        try:
            with open(report_file_path, "r", encoding=encoding) as f:
                content = f.read()
            break
        except UnicodeDecodeError as e:
            decode_errors.append(f"{encoding}: {e}")

    if content is None:
        attempted = ", ".join(attempted_encodings)
        details = " | ".join(decode_errors)
        raise UnicodeError(
            f"Failed to decode SWMM report file {report_file_path} using encodings "
            f"[{attempted}]. Errors: {details}"
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
        raise ValueError(
            f"Could not find supported elapsed-time format in SWMM report file "
            f"{report_file_path}"
        )

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

    if use_rpt_for_tseries:
        (
            dict_section_lines,
            ds_node_tseries,
            ds_link_tseries,
            dict_system_results,
            valid,
        ) = parse_rpt_single_pass(swmm_timeseries_result_file)
        if valid is False:
            print(
                "The RPT file seems to not contain any information: {}".format(
                    swmm_timeseries_result_file
                )
            )
        # make sure the data arrays are data type float
        ds_node_tseries = convert_datavars_to_dtype(
            ds_node_tseries, lst_dtypes_to_try=[float]
        )
        ds_link_tseries = convert_datavars_to_dtype(
            ds_link_tseries, lst_dtypes_to_try=[float]
        )
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
    else:
        ds_node_tseries, ds_link_tseries = return_node_time_series_results_from_outfile(  # type: ignore
            swmm_timeseries_result_file
        )
        rpt_file = swmm_timeseries_result_file.with_suffix(".rpt")
        if rpt_file.exists():
            try:
                with open(rpt_file, "r", encoding="latin-1") as file:
                    rpt_lines = file.readlines()
                lst_node_fld_summary = return_lines_for_section_of_rpt(
                    "Node Flooding Summary", lines=rpt_lines
                )
                lst_node_flow_summary = return_lines_for_section_of_rpt(
                    "Node Inflow Summary", lines=rpt_lines
                )
                lst_link_flow_summary = return_lines_for_section_of_rpt(
                    "Link Flow Summary", lines=rpt_lines
                )
                dict_system_results = return_swmm_system_outputs(rpt_lines)
            except Exception as exc:
                warnings.warn(
                    f"Failed to parse SWMM RPT summaries from {rpt_file}: {exc}",
                    UserWarning,
                )
    #
    if use_rpt_for_tseries:
        lst_node_fld_summary = dict_section_lines.get("Node Flooding Summary", [])
        lst_node_flow_summary = dict_section_lines.get("Node Inflow Summary", [])
        lst_link_flow_summary = dict_section_lines.get("Link Flow Summary", [])
    df_node_flood_summary = format_rpt_section_into_dataframe(
        lst_node_fld_summary, lst_col_headers_node_flood_summary
    )
    df_node_flow_summary = format_rpt_section_into_dataframe(
        lst_node_flow_summary, lst_col_headers_node_flow_summary
    )
    df_link_flow_summary = format_rpt_section_into_dataframe(
        lst_link_flow_summary, lst_col_headers_link_flow_summary
    )
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
    df_node_summaries = df_node_summaries.drop(
        columns=["time_of_max_flood_d_hr_mn", "time_of_max_flow_d_hr_mn"]
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
    df_link_flow_summary["time_of_max_flow_min"] = convert_swmm_tdeltas_to_minutes(
        df_link_flow_summary["time_of_max_flow_d_hr_mn"]
    )
    df_link_flow_summary = df_link_flow_summary.drop(
        columns=["time_of_max_flow_d_hr_mn"]
    )

    def _clean_link_id(link_id):
        """Clean a single link_id value."""
        try:
            return str(int(float(link_id)))
        except (ValueError, TypeError):
            if isinstance(link_id, str):
                parts = link_id.split()
                return parts[0] if parts else link_id
            return str(link_id)

    df_link_flow_summary["link_id"] = df_link_flow_summary["link_id"].apply(
        _clean_link_id
    )
    df_link_flow_summary.set_index("link_id", inplace=True)

    ds_node_summaries = df_node_summaries.to_xarray()
    ds_link_flow_summary = df_link_flow_summary.to_xarray()
    #
    ds_node_summaries = convert_coords_to_dtype(
        ds_node_summaries,
        lst_dtypes_to_try=[str],
        coords_to_coerce=["node_id", "link_id"],
    )
    ds_node_summaries = convert_datavars_to_dtype(
        ds_node_summaries, lst_dtypes_to_try=[float, str]
    )
    ds_link_flow_summary = convert_coords_to_dtype(
        ds_link_flow_summary,
        lst_dtypes_to_try=[str],
        coords_to_coerce=["node_id", "link_id"],
    )
    ds_link_flow_summary = convert_datavars_to_dtype(
        ds_link_flow_summary, lst_dtypes_to_try=[float, str]
    )

    ds_nodes = xr.merge([ds_node_summaries, ds_node_tseries], join="outer")
    ds_links = xr.merge([ds_link_flow_summary, ds_link_tseries], join="outer")
    #
    ds_nodes.attrs = dict_system_results
    ds_links.attrs = dict_system_results
    #
    return ds_nodes, ds_links


def return_swmm_system_outputs(rpt_lines):
    line_num = -1
    encountered_flow_routing_continuity = False
    runoff_continuity_error_line = None
    flow_continuity_error_line = None
    for line in rpt_lines:
        line_num += 1
        # if "Runoff Quantity Continuity" in line:
        #     encountered_runoff_quantity_continuity = True
        if "Flow Units" in line:
            line_flw_untits = line
        if "Flow Routing Continuity" in line:
            encountered_flow_routing_continuity = True
        if "Continuity Error (%)" in line:
            # runoff routing is reported BEFORE flow routing
            if encountered_flow_routing_continuity is False:
                runoff_continuity_error_line = line
            else:
                flow_continuity_error_line = line
        # return system flood statistic
        if "Flooding Loss" in line:
            system_flood_loss_line = line
    flow_units = line_flw_untits.split(".")[-1].split("\n")[0].split(" ")[-1].lower()

    if runoff_continuity_error_line is not None:
        runoff_continuity_error_perc = float(
            runoff_continuity_error_line.split(" ")[-1].split("\n")[0]
        )
    else:
        runoff_continuity_error_perc = np.nan
    flow_continuity_error_perc = float(
        flow_continuity_error_line.split(" ")[-1].split("\n")[0]  # type: ignore
    )
    # return system flood losses
    system_flooding = float(system_flood_loss_line.split(" ")[-1].split("\n")[0])
    analysis_end_datetime = return_analysis_end_date(rpt_lines)
    dict_system_results = dict(
        analysis_end_datetime=str(analysis_end_datetime),
        flow_units=flow_units,
        runoff_continuity_error_perc=runoff_continuity_error_perc,
        flow_continuity_error_perc=flow_continuity_error_perc,
        system_flooding_10e6_ltr=system_flooding,
    )
    return dict_system_results


@profile
def parse_rpt_single_pass(f_rpt: Path):
    """
    Parse an RPT file in a single pass, extracting summaries and time series.

    Returns section lines for summary tables, node/link time series datasets,
    system-level outputs, and a validity flag.
    """
    summary_section_headers = (
        "Node Flooding Summary",
        "Node Inflow Summary",
        "Link Flow Summary",
    )
    dict_section_lines = {header: [] for header in summary_section_headers}
    dict_lst_node_time_series = {}
    dict_lst_link_time_series = {}

    valid = False
    encountered_flow_routing_continuity = False
    line_flw_units = None
    runoff_continuity_error_line = None
    flow_continuity_error_line = None
    system_flood_loss_line = None
    analysis_end_line = None

    current_summary_section = None
    summary_begin_header = False
    summary_end_header = False

    tseries_started = False
    tseries_key = None
    tseries_is_link = False
    tseries_is_node = False
    tseries_begin_header = False
    tseries_end_header = False
    tseries_vals = []

    for line_num, line in enumerate(_iter_rpt_lines(f_rpt)):
        line_split = line.split(" ")
        line_split_len = len(line_split)
        if "Element Count" in line:
            valid = True
        if "Flow Units" in line:
            line_flw_units = line
        if "Flow Routing Continuity" in line:
            encountered_flow_routing_continuity = True
        if "Continuity Error (%)" in line:
            if encountered_flow_routing_continuity is False:
                runoff_continuity_error_line = line
            else:
                flow_continuity_error_line = line
        if "Flooding Loss" in line:
            system_flood_loss_line = line
        if "Analysis ended on" in line:
            analysis_end_line = line

        if not tseries_started and "Node Time Series Results" in line:
            tseries_started = True

        if tseries_started:
            if "<<<" in line:
                tseries_key = line.split("<<<")[1].split(" ")[2]
                lower_line = line.lower()
                tseries_is_link = "link" in lower_line
                tseries_is_node = "node" in lower_line
                tseries_vals = []
                tseries_begin_header = False
                tseries_end_header = False
                continue
            if tseries_key is not None:
                if "------------" in line:
                    if not tseries_begin_header:
                        tseries_begin_header = True
                        continue
                    if not tseries_end_header:
                        tseries_end_header = True
                        continue
                if not (tseries_begin_header and tseries_end_header):
                    continue
                if line_split_len <= 3:
                    if tseries_is_link:
                        dict_lst_link_time_series[tseries_key] = tseries_vals
                    if tseries_is_node:
                        dict_lst_node_time_series[tseries_key] = tseries_vals
                    tseries_key = None
                    tseries_vals = []
                    continue
                tseries_vals.append(line)

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
        if line_split_len <= 3:
            current_summary_section = None
            continue
        dict_section_lines[current_summary_section].append(line)

    if tseries_key is not None and tseries_vals:
        if tseries_is_link:
            dict_lst_link_time_series[tseries_key] = tseries_vals
        if tseries_is_node:
            dict_lst_node_time_series[tseries_key] = tseries_vals

    dict_system_results = _build_system_results(
        line_flw_units,
        runoff_continuity_error_line,
        flow_continuity_error_line,
        system_flood_loss_line,
        analysis_end_line,
    )
    ds_node_tseries, ds_link_tseries = _build_tseries_datasets(
        dict_lst_node_time_series,
        dict_lst_link_time_series,
    )
    return (
        dict_section_lines,
        ds_node_tseries,
        ds_link_tseries,
        dict_system_results,
        valid,
    )


def _iter_rpt_lines(f_rpt: Path):
    """Yield lines from an RPT file."""
    with open(f_rpt, "r", encoding="latin-1") as file:
        for line in file:
            yield line


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
        runoff_continuity_error_perc = float(
            runoff_continuity_error_line.split(" ")[-1].split("\n")[0]
        )
    else:
        runoff_continuity_error_perc = np.nan
    flow_continuity_error_perc = float(
        flow_continuity_error_line.split(" ")[-1].split("\n")[0]  # type: ignore
    )
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
    datetime_string = "{}-{}-{} {}".format(month, day, assumed_year, time)
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
        with open(f_rpt, "r", encoding="latin-1") as file:  # type: ignore
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
            first_line = line_num
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


def return_analysis_end_date(rpt_lines):
    for line in rpt_lines:
        if "Analysis ended on" in line:
            end_line = line
    return _parse_analysis_end_line(end_line)


def return_node_time_series_results_from_rpt(
    section_header="Node Time Series Results",
    f_rpt=None,
    lines=None,
):
    # section_header, lines = "Node Time Series Results", rpt_lines
    # lst_section_lines = []
    if lines is None:
        with open(f_rpt, "r", encoding="latin-1") as file:  # type: ignore
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


def return_node_time_series_results_from_outfile(f_outfile):
    from pyswmm import Output, NodeSeries, LinkSeries

    with Output(str(f_outfile)) as out:
        d_links = out.links
        d_nodes = out.nodes
        units = out.units
        if units["system"] != "SI":  # type: ignore
            sys.exit("SWMM outputs are not in SI units!")
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
            df = (
                pd.concat(dic_dfs[key])
                .reset_index()
                .set_index(["node_id", "date_time"])
            )
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
            df = (
                pd.concat(dic_dfs[key])
                .reset_index()
                .set_index(["link_id", "date_time"])
            )
            lst_dfs.append(df)
        #
        ds_link_tseries = pd.concat(lst_dfs, axis=1).to_xarray()
    #
    return ds_node_tseries, ds_link_tseries


def create_tseries_ds(dict_lst_time_series, lst_col_headers, idx_colname):
    # dict_lst_time_series, lst_col_headers, idx_colname = dict_lst_node_time_series, lst_col_headers, "link_id"
    # dict_lst_time_series, lst_col_headers, idx_colname = dict_lst_node_time_series, lst_col_headers, "node_id"
    lst_dfs = []
    for key, lst_section_lines in dict_lst_time_series.items():
        df_tseries = format_rpt_section_into_dataframe(
            lst_section_lines, lst_col_headers
        )
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
    df_tseries["date_time"] = pd.to_datetime(
        df_tseries["date_time"], format=RPT_DATETIME_FORMAT, errors="coerce"
    )
    if df_tseries["date_time"].isna().any():
        raise ValueError(
            "Parsed RPT date_time values contained NaT. "
            "Verify the RPT datetime format matches "
            f"{RPT_DATETIME_FORMAT}."
        )
    df_tseries = df_tseries.set_index([idx_colname, "date_time"])
    return df_tseries.to_xarray()


def convert_pyswmm_output_to_df(
    pyswmm_tseries, valname, item_idx_name, item_idx, idx_name="date_time"
):
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
    dict_line_contents_asline = {}
    dict_content_lengths = {}
    # extract and parse data in each line using spaces
    line_contents = []
    line_strings = []
    line_lengths = []
    for line in lst_section_lines:
        lst_substrings_with_content = [
            substring
            for substring in line.split(" ")
            if substring and substring not in lst_substrings_to_ignore
        ]
        if lst_substrings_with_content:
            line_contents.append(lst_substrings_with_content)
            line_strings.append(line)
            line_lengths.append(len(lst_substrings_with_content))
    dict_line_contents_aslist = {
        idx: contents for idx, contents in enumerate(line_contents)
    }
    dict_line_contents_asline = {idx: line for idx, line in enumerate(line_strings)}
    dict_content_lengths = {idx: length for idx, length in enumerate(line_lengths)}
    # make sure the lines all have the same lengths
    if not line_lengths:
        return dict_line_contents_aslist
    target_length = Counter(line_lengths).most_common(1)[0][0]
    idx_problem_rows = [
        idx for idx, length in dict_content_lengths.items() if length != target_length
    ]
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
            for prob_index, val in enumerate(problem_row_list):
                if val.count(".") > 1:
                    solution = "Two values in the rpt were right next to each other and couldn't be parsed using spacing. Parsing by referencing a normal line."
                    print("##################################")
                    print(f"Found problem. {solution}")
                    print("Normal row vs. problem row:")
                    print(normal_row)
                    print(problem_row)
                    break
                if "orifice" in problem_row_lower:  # type: ignore
                    solution = "Orifice conduits do not return max velocity or max over full flow. Filling with empty string"
                    print("##################################")
                    print(f"Found problem. {solution}")
                    print("Normal row vs. problem row:")
                    print(normal_row)
                    print(problem_row)
                    break
            if (
                solution
                == "Two values in the rpt were right next to each other and couldn't be parsed using spacing. Parsing by referencing a normal line."
            ):
                # problem_val = problem_row_list[prob_index]
                normal_val_at_index = normal_row_list[prob_index]
                normal_next_val_at_index = normal_row_list[prob_index + 1]
                # identify string parsing location
                ## deal with the possibility that there are multiple substrings with the same value; find the pair with the smallest difference between the current val and next val
                closest_end_loc_of_val_at_index = 9999
                closest_begin_loc_of_next_val = -9999
                dif_between_locs = 9999
                for normal_val_at_index_string_ilocs in find_substring_indices(
                    normal_row, normal_val_at_index
                ):
                    if normal_val_at_index != extract_substring(
                        normal_row, normal_val_at_index_string_ilocs
                    ):
                        print("WARNING: There is a string indexing issue")
                    end_loc_prev = max(normal_val_at_index_string_ilocs)
                    for normal_val_at_next_index_string_ilocs in find_substring_indices(
                        normal_row, normal_next_val_at_index
                    ):
                        begin_loc_next = min(normal_val_at_next_index_string_ilocs)
                        if (begin_loc_next - end_loc_prev) < dif_between_locs:
                            dif_between_locs = begin_loc_next - end_loc_prev
                            closest_begin_loc_of_next_val = begin_loc_next
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
                print(
                    "####################################################################"
                )
                print(
                    "WARNING: There is an issue with swmm outputs read from an rpt file that I have not yet encountered"
                )
                print("Here is an example of a normal row:")
                print(normal_row)
                print(
                    f"There are {len(idx_problem_rows)} problem rows. Here are examples:"
                )
                for idx in idx_problem_rows[:5]:
                    print(line_strings[idx])
                print(
                    "####################################################################"
                )
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


def _select_normal_row(
    line_strings, line_lengths, idx_problem_rows, dict_line_contents_aslist
):
    str_lengths = [len(line) for line in line_strings]
    str_length_mode = Counter(str_lengths).most_common(1)[0][0]
    normal_idx = None
    for idx, (str_len, content_len) in enumerate(zip(str_lengths, line_lengths)):
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
    return [
        (m.start(), m.end() - 1)
        for m in _substring_pattern(substring).finditer(main_string)
    ]


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
    lst_dtypes_to_try=[int, str],
    coords_to_coerce=["node_id", "link_id", "model", "simtype"],
):
    for coord in ds.coords:
        if (ds[coord].dtype == object) or (coord in coords_to_coerce):
            converted = False
            for dtype in lst_dtypes_to_try:
                # break the loop if it is already the desired data type
                if ds[coord].dtype == dtype:
                    converted = True
                    break
                if dtype == int:
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


def convert_datavars_to_dtype(ds, lst_dtypes_to_try=[str], lst_vars_to_convert=None):
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
            if (dtype == float) or (dtype == int):
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
                if dtype == int:
                    if np.any(sample_numeric % 1 != 0):
                        first_attempt = False
                        continue
                data_to_convert = ds[var]
                if data_to_convert.dtype == object:
                    data_to_convert = data_to_convert.astype(str)
                    data_to_convert = xr.where(
                        data_to_convert == "", "0", data_to_convert
                    )
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
