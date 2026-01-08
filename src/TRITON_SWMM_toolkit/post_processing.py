import sys
import time
import xarray as xr
import pandas as pd
import numpy as np
from glob import glob
import shutil
import zarr
import rioxarray as rxr
from typing import Literal
import warnings
from pathlib import Path
from TRITON_SWMM_toolkit.utils import (
    write_zarr,
    write_zarr_then_netcdf,
    paths_to_strings,
)
from TRITON_SWMM_toolkit.utils import current_datetime_string
from TRITON_SWMM_toolkit.running_a_simulation import TRITONSWMM_run
import re
from datetime import datetime


class TRITONSWMM_post_processing:
    def __init__(self, run: TRITONSWMM_run) -> None:
        self._run = run
        self._scenario = run._scenario
        self._experiment = run._scenario._experiment
        self._system = run._scenario._experiment._system
        self.log = self._scenario.log
        self.scen_paths = self._scenario.scen_paths

        if not self._scenario.sim_run_completed:
            raise RuntimeError(
                f"Simulation not completed. Log: {self._scenario.latest_simlog}"
            )

    def _already_written(self, f_out):
        proc_log = self.log.processing_log.outputs
        already_written = False
        if f_out.name in proc_log.keys():
            if proc_log[f_out.name].success == True:
                already_written = True
        return already_written

    def export_TRITON_outputs(
        self,
        overwrite_if_exist: bool = False,
        verbose: bool = False,
        compression_level: int = 5,
    ):
        fname_out = self.scen_paths.output_triton_timeseries
        # dir_outputs = self._run._triton_swmm_raw_output_directory()
        fldr_out_triton = self._run.raw_triton_output_dir
        raw_out_type = self._experiment.cfg_exp.TRITON_raw_output_type
        reporting_interval_s = self._experiment.cfg_exp.TRITON_reporting_timestep_s
        rds_dem = self._system.open_processed_dem_as_rds()

        if self._already_written(fname_out) and not overwrite_if_exist:
            if verbose:
                print(f"{fname_out.name} already written. Not overwriting.")
            return

        start_time = time.time()

        # load the dem in order to extract the spatial coordinates and assign them to the triton outputs
        bm_time = time.time()
        # out_type = "bin"
        df_outputs = return_fpath_wlevels(fldr_out_triton, reporting_interval_s)
        lst_ds_vars = []
        for varname, files in df_outputs.items():
            lst_ds = []
            for tstep_min, f in files.items():
                ds_triton_output = load_triton_output_w_xarray(
                    rds_dem, f, varname, raw_out_type
                )
                lst_ds.append(ds_triton_output)
            ds_var = xr.concat(lst_ds, dim=df_outputs.index)
            lst_ds_vars.append(ds_var)
        if verbose:
            print(
                f"Time to load {raw_out_type} triton outputs (min) {(time.time()-bm_time)/60:.2f}"
            )
        ds_combined = xr.merge(lst_ds_vars)
        self._write_output(ds_combined, fname_out, compression_level, verbose)  # type: ignore

        elapsed_s = time.time() - start_time
        self.log.add_sim_processing_entry(fname_out, elapsed_s, True)
        return

    def _write_output(
        self,
        ds: xr.Dataset | xr.DataArray,
        f_out: Path,
        compression_level: int,
        verbose: bool,
    ):
        processed_out_type = self._experiment.cfg_exp.TRITON_processed_output_type

        ds.attrs["sim_date"] = self._scenario.latest_sim_date(astype="str")
        ds.attrs["output_creation_date"] = current_datetime_string()

        ds.attrs["sim_log"] = paths_to_strings(self.log.as_dict())
        ds.attrs["paths"] = paths_to_strings(
            self._experiment.dict_of_all_sim_files(self._scenario.sim_iloc)
        )
        ds.attrs["configuration"] = paths_to_strings(
            {
                "system": self._system.cfg_system.model_dump(),
                "experiment": self._experiment.cfg_exp.model_dump(),
            }
        )

        if processed_out_type == "nc":
            write_zarr_then_netcdf(ds, f_out, compression_level)
        else:
            write_zarr(ds, f_out, compression_level)
        if verbose:
            print(f"finished writing {f_out}")

    def export_SWMM_outputs(
        self,
        overwrite_if_exist: bool = False,
        comp_level: int = 5,
        verbose: bool = False,
    ):
        start_time = time.time()
        f_out_nodes = self.scen_paths.output_swmm_node_time_series
        f_out_links = self.scen_paths.output_swmm_link_time_series

        f_inp = self.scen_paths.inp_hydraulics
        swmm_timeseries_result_file = self._run.raw_swmm_output

        nodes_already_written = self._already_written(f_out_nodes)
        links_already_written = self._already_written(f_out_links)

        if (nodes_already_written and links_already_written) and not overwrite_if_exist:
            if verbose:
                print(
                    f"{f_out_nodes.name} and {f_out_links.name} already written. Not overwriting."
                )
            return

        ds_nodes, ds_links = retrieve_SWMM_outputs_as_datasets(
            f_inp, swmm_timeseries_result_file
        )
        # WRITE NODES
        if nodes_already_written and not overwrite_if_exist:
            if verbose:
                print(f"{f_out_nodes.name} already written. Not overwriting.")
            pass
        else:
            elapsed_s = time.time() - start_time
            self._write_output(ds_nodes, f_out_nodes, comp_level, verbose)
            self.log.add_sim_processing_entry(f_out_nodes, elapsed_s, True)
        # WRITE LINKS
        if links_already_written and not overwrite_if_exist:
            if verbose:
                print(f"{f_out_links.name} already written. Not overwriting.")
            pass
        else:
            elapsed_s = time.time() - start_time
            self._write_output(ds_links, f_out_links, comp_level, verbose)
            self.log.add_sim_processing_entry(
                f_out_links,
                elapsed_s,
                True,
                notes="links are written after nodes so time elapsed reflecs writing both link AND node time series",
            )

        return


# %% WORK - SWMM STUFF

from TRITON_SWMM_toolkit.constants import (
    LST_COL_HEADERS_NODE_FLOOD_SUMMARY,
    LST_COL_HEADERS_NODE_FLOW_SUMMARY,
    LST_COL_HEADERS_LINK_FLOW_SUMMARY,
)


def retrieve_SWMM_outputs_as_datasets(f_swmm_inp, swmm_timeseries_result_file: Path):
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
    if swmm_timeseries_result_file.name.split(".")[-1] == "rpt":
        use_rpt_for_tseries = True
    elif swmm_timeseries_result_file.name.split(".")[-1] == "out":
        use_rpt_for_tseries = False
    else:
        raise ValueError(
            f"SWMM output file not recognized while parsing time series. File passed: {swmm_timeseries_result_file}"
        )

    # from pyswmm import Nodes
    import swmmio

    if use_rpt_for_tseries:
        with open(swmm_timeseries_result_file, "r", encoding="latin-1") as file:
            # Read all lines from the file
            rpt_lines = file.readlines()
        # file.close()
        # verify validity of rpt file
        valid = False
        for line in rpt_lines:
            if "Element Count" in line:
                valid = True
                break
        if valid == False:
            print(
                "The RPT file seems to not contain any information: {}".format(
                    swmm_timeseries_result_file
                )
            )
        section_header = "Node Time Series Results"
        lines = rpt_lines
        ds_node_tseries, ds_link_tseries = return_node_time_series_results_from_rpt(
            section_header=section_header, lines=lines
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
    #
    dict_system_results = return_swmm_system_outputs(rpt_lines)
    # create dataframes of node and link outputs
    lst_node_fld_summary = return_lines_for_section_of_rpt(
        section_header="Node Flooding Summary", lines=rpt_lines
    )
    df_node_flood_summary = format_rpt_section_into_dataframe(
        lst_node_fld_summary, lst_col_headers_node_flood_summary
    )
    #
    lst_node_flow_summary = return_lines_for_section_of_rpt(
        section_header="Node Inflow Summary", lines=rpt_lines
    )
    df_node_flow_summary = format_rpt_section_into_dataframe(
        lst_node_flow_summary, lst_col_headers_node_flow_summary
    )
    #
    lst_link_flow_summary = return_lines_for_section_of_rpt(
        section_header="Link Flow Summary", lines=rpt_lines
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
    for idx, row in df_link_flow_summary.iterrows():
        link_id = row.link_id
        try:
            link_id = str(int(link_id))
        except:
            lst_val_substrings = link_id.split(" ")
            for substring in lst_val_substrings:
                if len(substring) > 0:
                    link_id = substring
                    break
        df_link_flow_summary.loc[idx, "link_id"] = link_id
    df_link_flow_summary.set_index("link_id", inplace=True)
    #
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = swmmio.Model(str(f_swmm_inp))
        nodes = model.nodes.geodataframe
        nodes.index.name = "node_id"
        links = model.links.geodataframe
        links.index.name = "link_id"
        #
        # from shapely.geometry import LineString
        lines = links.geometry
        lines_wkt = np.array([line.wkt for line in lines])
        links["geometry_wkt"] = lines_wkt
        links = links.drop(columns="geometry")
        #
        points = nodes.geometry
        points_wkt = np.array([point.wkt for point in points])
        nodes["geometry_wkt"] = points_wkt
        nodes = nodes.drop(columns="geometry")
        # define how to create geodataframe from the saved geometry_wkt attribute
        converting_to_gpd = """from shapely import wkt
        df_geom = ds.geometry_wkt.to_dataframe().geometry_wkt
        features = [wkt.loads(wkt_str) for wkt_str in df_geom]
        gdf = gpd.GeoDataFrame(pd.DataFrame(index = df_geom.index, data=dict(geometry=features)), geometry="geometry")"""
    #
    # assign proper data types to coordinates and data variables
    ds_node_summaries = df_node_summaries.to_xarray()
    ds_node_characteristics = nodes.to_xarray()
    ds_link_flow_summary = df_link_flow_summary.to_xarray()
    ds_link_characteristics = links.to_xarray()
    #
    dict_ds = dict(
        ds_node_summaries=ds_node_summaries,
        ds_node_characteristics=ds_node_characteristics,
        ds_link_flow_summary=ds_link_flow_summary,
        ds_link_characteristics=ds_link_characteristics,
    )
    for key in dict_ds:
        dict_ds[key] = convert_coords_to_dtype(
            dict_ds[key],
            lst_dtypes_to_try=[str],
            coords_to_coerce=["node_id", "link_id"],
        )
        ds, lst_dtypes_to_try = dict_ds[key], [float, str]
        dict_ds[key] = convert_datavars_to_dtype(
            dict_ds[key], lst_dtypes_to_try=[float, str]
        )

    ds_nodes = xr.merge(
        [
            dict_ds["ds_node_summaries"],
            ds_node_tseries,
            dict_ds["ds_node_characteristics"],
        ]
    )
    ds_links = xr.merge(
        [
            dict_ds["ds_link_flow_summary"],
            ds_link_tseries,
            dict_ds["ds_link_characteristics"],
        ]
    )
    #
    dict_system_results["converting_to_gpd"] = converting_to_gpd
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
            if encountered_flow_routing_continuity == False:
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


def return_analysis_end_date(rpt_lines):
    for line in rpt_lines:
        if "Analysis ended on" in line:
            end_line = line
    # parse analysis end time
    lst_end_line = end_line.split("on:")[-1].split(" ")
    lst_info_in_line = []
    for substring in lst_end_line:
        if len(substring) > 0:
            lst_info_in_line.append(substring)
    # day_of_week = lst_info_in_line[0]
    month = lst_info_in_line[1]
    day = lst_info_in_line[2]
    assumed_year = datetime.today().year
    time = lst_info_in_line[3]
    datetime_string = "{}-{}-{} {}".format(month, day, assumed_year, time)
    analysis_end_datetime = pd.to_datetime(datetime_string, format="%b-%d-%Y %H:%M:%S")


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
        if encountered_header == False:
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
        if encountered_end == False:
            lst_section_lines.append(line)
    return lst_section_lines


def convert_swmm_tdeltas_to_minutes(s_tdelta):
    lst_tdeltas_min = []
    for val in s_tdelta:
        if pd.Series(val).isna()[0]:
            lst_tdeltas_min.append(np.nan)
            continue
        lst_val_substrings_all = val.split(" ")
        lst_val_substring_data = []
        for val in lst_val_substrings_all:
            if len(val) > 0:
                lst_val_substring_data.append(val)

        days = int(lst_val_substring_data[0])
        hh_mm = lst_val_substring_data[-1]
        hr = int(hh_mm.split(":")[0])
        min = int(hh_mm.split(":")[1])
        tdelta = (
            pd.Timedelta(days, unit="D")
            + pd.Timedelta(hr, unit="hr")
            + pd.Timedelta(min, unit="min")
        )
        lst_tdeltas_min.append(tdelta.total_seconds() / 60)
    return lst_tdeltas_min


def return_node_time_series_results_from_outfile(f_outfile):
    from pyswmm import Output, NodeSeries, LinkSeries

    with Output(f_outfile) as out:
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


def convert_pyswmm_output_to_df(
    pyswmm_tseries, valname, item_idx_name, item_idx, idx_name="date_time"
):
    df = pd.DataFrame.from_dict(pyswmm_tseries, orient="index", columns=[valname])
    df.index.name = idx_name
    df[item_idx_name] = str(item_idx)
    return df


def return_node_time_series_results_from_rpt(
    section_header="Node Time Series Results", f_rpt=None, lines=None
):
    # section_header, lines = "Node Time Series Results", rpt_lines
    # lst_section_lines = []
    if lines is None:
        with open(f_rpt, "r", encoding="latin-1") as file:  # type: ignore
            # Read all lines from the file
            lines = file.readlines()
    line_num = -1
    encountered_header = False
    dict_lst_node_time_series = {}
    dict_lst_link_time_series = {}
    begin_header_line_num = None
    end_header_line_num = None
    encountered_end = False
    for line in lines:
        line_num += 1
        # return node flooding summaries
        if section_header in line:
            section_start_line = line_num
            encountered_header = True
        if encountered_header == False:
            continue
        if "<<<" in line:
            # define new list for values and the node name
            key = line.split("<<<")[1].split(" ")[2]
            is_link = is_node = False
            if "link" in line.lower():
                is_link = True
            if "node" in line.lower():
                is_node = True
            lst_vals = []
            begin_header_line_num = None
            end_header_line_num = None
            encountered_end = False
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
            if is_link:
                dict_lst_link_time_series[key] = lst_vals
            if is_node:
                dict_lst_node_time_series[key] = lst_vals
        if encountered_end == False:
            # dict_lst_node_time_series[node_id].append(line)
            lst_vals.append(line)
    #
    lst_col_headers = [
        "date_time",
        "inflow_flow_cms",
        "flooding_cms",
        "depth_m",
        "head_m",
    ]
    ds_node_tseries = create_tseries_ds(
        dict_lst_node_time_series, lst_col_headers, "node_id"
    )
    #
    lst_col_headers = [
        "date_time",
        "flow_cms",
        "velocity_mps",
        "link_depth_m",
        "capacity_setting",
    ]
    ds_link_tseries = create_tseries_ds(
        dict_lst_link_time_series, lst_col_headers, "link_id"
    )
    #
    return ds_node_tseries, ds_link_tseries


def create_tseries_ds(dict_lst_time_series, lst_col_headers, idx_colname):
    # dict_lst_time_series, lst_col_headers, idx_colname = dict_lst_node_time_series, lst_col_headers, "link_id"
    # dict_lst_time_series, lst_col_headers, idx_colname = dict_lst_node_time_series, lst_col_headers, "node_id"
    lst_dfs = []
    for key in dict_lst_time_series:
        lst_section_lines = dict_lst_time_series[key]
        df_tseries = format_rpt_section_into_dataframe(
            lst_section_lines, lst_col_headers
        )
        df_tseries[idx_colname] = key
        df_tseries["date_time"] = pd.to_datetime(df_tseries.date_time)
        df_tseries = df_tseries.set_index([idx_colname, "date_time"])
        lst_dfs.append(df_tseries)
    df_tseries = pd.concat(lst_dfs)
    ds_tseries = df_tseries.to_xarray()
    return ds_tseries


def format_rpt_section_into_dataframe(lst_section_lines, lst_col_headers):
    lst_series = []
    df_rpt_section = pd.DataFrame(columns=lst_col_headers)
    if len(lst_section_lines) > 0:
        dict_line_contents_aslist = return_data_from_rpt(lst_section_lines)
        for line_idx_with_vals in dict_line_contents_aslist:
            lst_substrings_with_content = dict_line_contents_aslist[line_idx_with_vals]
            # datetime = f"{lst_substrings_with_content[0]} {lst_substrings_with_content[1]}"
            # lst_values.append(datetime)
            s_vals = pd.Series(index=lst_col_headers).astype(str)
            idx_val = 0
            for idx_str, substring in enumerate(lst_substrings_with_content):
                if "\n" in substring:
                    substring = substring.split("\n")[0]
                if substring == "ltr":
                    continue
                # handling times
                if (
                    ":" in substring
                ):  # this is either a datetime or time of max occurence; this and the previous value should be combined
                    # update previous value
                    s_vals.iloc[idx_val - 1] = (
                        f"{str(s_vals.iloc[idx_val-1])} {substring}"
                    )
                    continue
                s_vals.iloc[idx_val] = substring
                idx_val += 1
                # lst_values.append(val)
            new_row = s_vals.to_frame().T
            lst_series.append(new_row)
        df_rpt_section = pd.concat(lst_series, ignore_index=True)
    return df_rpt_section


def return_data_from_rpt(lst_section_lines):
    lst_substrings_to_ignore = ["ltr\n"]
    # initialize vars
    dict_line_contents_aslist = {}
    dict_line_contents_asline = {}
    dict_content_lengths = {}
    line_idx = 0
    # extract and parse data in each line using spaces
    for i, line in enumerate(lst_section_lines):
        lst_substrings_with_content = []
        # lst_values = []
        # isolate strings with relevant values
        for substring in line.split(" "):
            if (len(substring) > 0) and (
                substring not in lst_substrings_to_ignore
            ):  # the latter part is to deal with issues
                lst_substrings_with_content.append(substring)
        if len(lst_substrings_with_content) > 0:
            dict_line_contents_aslist[line_idx] = lst_substrings_with_content
            dict_line_contents_asline[line_idx] = line
            dict_content_lengths[line_idx] = len(lst_substrings_with_content)
            line_idx += 1
    # make sure the lines all have the same lengths
    s_lengths = pd.Series(dict_content_lengths)
    target_length = s_lengths.mode().iloc[0]
    idx_problem_rows = s_lengths[s_lengths != target_length].index
    # if there is an issue
    if len(idx_problem_rows) > 0:
        s_lines = pd.Series(dict_line_contents_asline)
        s_str_lengths = s_lines.str.len()
        # identify a normal row
        idx_odd_stringlengths = s_lines[
            s_str_lengths != s_str_lengths.mode().iloc[0]
        ].index
        for normal_idx in s_lengths.index:
            if (normal_idx not in idx_odd_stringlengths) and (
                normal_idx not in idx_problem_rows
            ):
                break
        normal_row = s_lines.loc[normal_idx]
        normal_row_list = dict_line_contents_aslist[normal_idx]
        # loop through the problem rows and correct known issues
        for idx, problem_row in s_lines.loc[idx_problem_rows].items():
            problem_row_list = dict_line_contents_aslist[idx]
            solution = None
            for prob_index, val in enumerate(problem_row_list):
                if len(val.split(".")) > 2:
                    solution = "Two values in the rpt were right next to each other and couldn't be parsed using spacing. Parsing by referencing a normal line."
                    print("##################################")
                    print(f"Found problem. {solution}")
                    print("Normal row vs. problem row:")
                    print(normal_row)
                    print(problem_row)
                    break
                if "orifice" in problem_row.lower():  # type: ignore
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
                dict_line_contents_aslist[idx] = lst_substrings_corrected
                dict_content_lengths[idx] = len(lst_substrings_corrected)
                print(f"Properly parsed values:\n{lst_substrings_corrected}")
            elif (
                solution
                == "Orifice conduits do not return max velocity or max over full flow. Filling with empty string"
            ):
                problem_row_list.insert(5, "")
                problem_row_list.insert(6, "")
                dict_line_contents_aslist[idx] = problem_row_list
                print(f"Properly parsed values:\n{dict_line_contents_aslist[idx]}")
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
                    f"There are {len(s_lines.loc[idx_problem_rows])} problem rows. Here are examples:"
                )
                for idx, problem_row in s_lines.loc[idx_problem_rows].head(5).items():
                    print(problem_row)
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


def find_substring_indices(main_string, substring):
    """
    Finds all occurrences of a substring in a string.

    Parameters:
    main_string (str): The string to search within.
    substring (str): The substring to search for.

    Returns:
    list of tuples: A list of (start, end) tuples representing the start and end index of each occurrence.
    """
    return [(m.start(), m.end() - 1) for m in re.finditer(substring, main_string)]


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
                try:
                    if dtype == int:
                        # make sure this doesn't change its value from a float
                        invalid_conversion = ~(
                            (ds[coord].astype(dtype) % 1)
                            == (ds[coord].astype(float) % 1)
                        )
                        if sum(invalid_conversion) > 0:
                            # use float datatype instead
                            dtype = float
                    ds[coord] = ds[coord].astype(dtype)
                    converted = True
                    # print(f"converted {coord} to {dtype}")
                    break
                except:
                    continue
            if not converted:
                print(f"{coord} unable to be converted to either {lst_dtypes_to_try}")
    return ds


def convert_datavars_to_dtype(ds, lst_dtypes_to_try=[str], lst_vars_to_convert=None):
    # ds, lst_dtypes_to_try=[float, str]
    if lst_vars_to_convert is None:  # convert all variables
        lst_vars_to_convert = ds.data_vars
    for var in lst_vars_to_convert:
        converted = False
        first_attempt = True
        for dtype in lst_dtypes_to_try:
            # break if it alread is the resired data type
            if ds[var].dtype == dtype:
                converted = True
                break
            try:
                # deal with common problem in SWMM results
                if (dtype == float) or (dtype == int):
                    if ds[var].dtype == object:
                        # first coerce to string
                        ds[var] = ds[var].astype(str)
                        # convert "" to "0"
                        ds[var] = xr.where(ds[var] == "", "0", ds[var])
                    ds[var] = ds[var].astype(dtype)
                # verify conversion
                sample = isel_first_and_slice_longest(ds[var], n=10).values
                test = np.array(sample, dtype=dtype)
                ds[var] = ds[var].astype(dtype)
                converted = True
                if not first_attempt:
                    print(f"Converted variable to datatype = {var}, {dtype}")
                break
            except Exception as e:
                # print(f"Failed to convert variable to datatype = {var}, {dtype}. Trying next datatype. Error encountered: {e}")
                first_attempt = False
                pass
        if not converted:
            print(f"{var} unable to be converted to either {lst_dtypes_to_try}")
    return ds


def isel_first_and_slice_longest(ds, n=5):
    """
    Select the first element for all dimensions, but slice the first `n` elements
    for the longest dimension.
    """
    # Find the longest dimension
    longest_dim = max(ds.dims, key=lambda d: ds.sizes[d])
    # Build the `isel` dictionary: first element for all dims, slice for the longest
    isel_dict = {dim: 0 for dim in ds.dims}  # Default to first index
    isel_dict[longest_dim] = slice(  # type: ignore
        0, min(ds.sizes[longest_dim], n)
    )  # Slice longest dim
    # Apply the `isel` operation using the dictionary
    return ds.isel(isel_dict)


# %% END WORK
def return_filelist_by_tstep(
    fldr_out_triton: Path, fpattern_prefix, min_per_tstep, varname
):
    lst_f_out = list(fldr_out_triton.glob(f"{fpattern_prefix}*"))
    if len(lst_f_out) == 0:
        return None
    lst_reporting_tstep_min = []
    for f in lst_f_out:
        tstep_parts = f.name.split(f"{fpattern_prefix}_")[-1].split(".")[0].split("_")
        reporting_tstep_iloc = int(tstep_parts[0])
        if int(tstep_parts[1]) != 0:
            sys.exit(
                f"problem parsing reporting timestep for file {f}\nnot expecting nonzero values behind the last underscore"
            )
        reporting_tstep_min = reporting_tstep_iloc * min_per_tstep
        lst_reporting_tstep_min.append(reporting_tstep_min)
    s_outputs = pd.Series(index=lst_reporting_tstep_min, data=lst_f_out)
    s_outputs.index.name = "timestep_min"
    s_outputs.name = varname
    s_outputs = s_outputs.sort_index()
    return s_outputs


def return_fpath_wlevels(fldr_out_triton: Path, reporting_interval_s: int | float):
    ## retrive the reporting time interval from the cfg file
    min_per_tstep = reporting_interval_s / 60
    # associate filepaths to timesteps
    s_outputs_mh = return_filelist_by_tstep(
        fldr_out_triton, "MH", min_per_tstep, "max_wlevel_m"
    )
    s_outputs_h = return_filelist_by_tstep(
        fldr_out_triton, "H", min_per_tstep, "wlevel_m"
    )
    s_outputs_qx = return_filelist_by_tstep(
        fldr_out_triton, "QX", min_per_tstep, "velocity_x_mps"
    )
    s_outputs_qy = return_filelist_by_tstep(
        fldr_out_triton, "QY", min_per_tstep, "velocity_y_mps"
    )
    lst_out = [s_outputs_mh, s_outputs_h, s_outputs_qx, s_outputs_qy]
    non_empty_dfs = [s for s in lst_out if s is not None]
    df_outputs = pd.concat(non_empty_dfs, axis=1)
    return df_outputs


def load_triton_output_w_xarray(rds_dem, f_triton_output, varname, raw_out_type):
    if raw_out_type == "asc":
        df_triton_output = pd.read_csv(f_triton_output, sep=" ", header=None)
    elif raw_out_type == "bin":
        # Load the binary file into a NumPy array
        data = np.fromfile(f_triton_output, dtype=np.float64)
        y_dim = int(data[0])  # 513 # type: ignore
        x_dim = int(data[1])  # 526 # type: ignore
        data_values = data[2:]  # type: ignore
        # confirm these first two values are dimensions
        if len(data_values) != y_dim * x_dim:
            raise ValueError("Data size does not match the expected shape.")
        df_triton_output = pd.DataFrame(data_values.reshape((y_dim, x_dim)))
    else:
        sys.exit(
            f"load_triton_output_w_xarray failed because raw_out_type wasn't recognized ({raw_out_type})"
        )
    df_triton_output.columns = rds_dem.x.values
    df_triton_output = df_triton_output.set_index(rds_dem.y.values)
    df_triton_output.index.name = "y"
    df_triton_output = (
        pd.melt(df_triton_output, ignore_index=False, var_name="x", value_name=varname)
        .reset_index()
        .set_index(["x", "y"])
    )
    ds_triton_output = df_triton_output.to_xarray()
    return ds_triton_output
