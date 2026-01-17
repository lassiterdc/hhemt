# %%
import pandas as pd
import rioxarray as rxr
import numpy as np
import xarray as xr
from pathlib import Path
import sys
import shutil
from pyswmm import Simulation, Output
import geopandas as gpd
import swmmio
import warnings
from swmm.toolkit.shared_enum import NodeAttribute
from scipy.stats import rankdata
import TRITON_SWMM_toolkit.utils as utils
from datetime import datetime
from TRITON_SWMM_toolkit.log import TRITONSWMM_scenario_log
from TRITON_SWMM_toolkit.paths import ScenarioPaths
from typing import TYPE_CHECKING, Literal
from contextlib import redirect_stdout, redirect_stderr
import threading
import traceback
from TRITON_SWMM_toolkit.log import log_function_to_file
import logging

import sys

lock = threading.Lock()

if TYPE_CHECKING:
    from .analysis import TRITONSWMM_analysis

    from .run_simulation import TRITONSWMM_run


class TRITONSWMM_scenario:
    def __init__(self, event_iloc: int, analysis: "TRITONSWMM_analysis") -> None:
        self.event_iloc = event_iloc
        self._analysis = analysis
        self._system = analysis._system
        self.weather_event_indexers = (
            self._analysis._retrieve_weather_indexer_using_integer_index(event_iloc)
        )
        from TRITON_SWMM_toolkit.run_simulation import TRITONSWMM_run

        # define sim specific filepaths
        analysis_simulations_folder = self._analysis.analysis_paths.simulation_directory
        self.sim_id_str = self._retrieve_sim_id_str()
        sim_folder = analysis_simulations_folder / self.sim_id_str
        swmm_folder = sim_folder / "swmm"
        swmm_folder.mkdir(parents=True, exist_ok=True)

        self.scen_paths = ScenarioPaths(
            sim_folder=sim_folder,
            f_log=sim_folder / "log.json",
            weather_timeseries=sim_folder / "sim_weather.nc",
            # swmm time series
            dir_weather_datfiles=sim_folder / "dats",
            # swmm models
            inp_hydro=swmm_folder / "hydro.inp",
            inp_hydraulics=swmm_folder / "hydraulics.inp",
            inp_full=swmm_folder / "full.inp",
            # external boundary conditions
            extbc_tseries=sim_folder / "extbc" / f"tseries.txt",
            extbc_loc=sim_folder / "extbc" / f"loc.extbc",
            # inflow hydrographs
            hyg_timeseries=sim_folder / "strmflow" / "tseries.hyg",
            hyg_locs=sim_folder / "strmflow" / "loc.txt",
            # TRITON-SWMM
            triton_swmm_cfg=sim_folder / f"TRITONSWMM.cfg",
            sim_tritonswmm_executable=sim_folder / "build" / "triton",
            tritonswmm_logfile_dir=sim_folder / "tritonswmm_sim_logfiles",
            # OUTPUTS
            output_triton_timeseries=sim_folder
            / f"TRITON_tseries.{self._analysis.cfg_analysis.TRITON_processed_output_type}",
            output_swmm_link_time_series=sim_folder
            / f"SWMM_link_tseries.{self._analysis.cfg_analysis.TRITON_processed_output_type}",
            output_swmm_node_time_series=sim_folder
            / f"SWMM_node_tseries.{self._analysis.cfg_analysis.TRITON_processed_output_type}",
        )
        self._create_directories()
        if self.scen_paths.f_log.exists():
            self.log = TRITONSWMM_scenario_log.from_json(self.scen_paths.f_log)
        else:
            self.log = TRITONSWMM_scenario_log(
                event_iloc=self.event_iloc,
                event_idx=self.weather_event_indexers,
                simulation_folder=self.scen_paths.sim_folder,
                logfile=self.scen_paths.f_log,
            )
        self.run = TRITONSWMM_run(self)

    @property
    def latest_simlog(self) -> dict:
        dic_logs = self.log.sim_log.model_dump()["run_attempts"]
        if not dic_logs:
            return {"status": "no sim run attempts made"}
        latest_key = max(
            dic_logs.keys(),
            key=lambda k: utils.string_to_datetime(k),
        )
        return dic_logs[latest_key]

    @property
    def sim_compute_time_min(self) -> float:
        """
        Docstring for sim_compute_time

        :param self: Adds up total compute time even if the simulatoin required restarting
          one or more times from a hotstart file.
        :return:
        :rtype: float
        """
        conversion = 1 / 60

        dic_full_sim = dict()
        dic_logs = self.log.sim_log.model_dump()["run_attempts"].copy()
        gathering_current_simlogs = True
        while gathering_current_simlogs:
            latest_key = max(
                dic_logs.keys(),
                key=lambda k: utils.string_to_datetime(k),
            )
            dic_full_sim[latest_key] = dic_logs[latest_key]
            del dic_logs[latest_key]
            if dic_full_sim[latest_key]["sim_start_reporting_tstep"] == 0:
                gathering_current_simlogs = False
                break
        # add up compute time
        total_compute_time = 0
        for sim_datetime, sim_dict in dic_full_sim.items():
            total_compute_time += sim_dict["time_elapsed_s"]

        return total_compute_time * conversion

    @property
    def sim_run_completed(self) -> bool:
        success = self.run.sim_run_completed
        self.log.simulation_completed.set(success)
        return success

    def _latest_sim_status(self) -> str:
        simlog = self.latest_simlog
        return simlog["status"]

    def latest_sim_date(self, astype: Literal["dt", "str"] = "dt") -> datetime:
        simlog = self.latest_simlog
        if simlog["status"] == "no sim run attempts made":
            return datetime.min
        else:
            dt_str = simlog["sim_datetime"]
            if astype == "dt":
                return utils.string_to_datetime(dt_str)
            else:
                return dt_str

    def _create_directories(self):
        self.scen_paths.dir_weather_datfiles.mkdir(parents=True, exist_ok=True)
        self.scen_paths.extbc_tseries.parent.mkdir(parents=True, exist_ok=True)
        self.scen_paths.hyg_timeseries.parent.mkdir(parents=True, exist_ok=True)
        self.scen_paths.tritonswmm_logfile_dir.mkdir(parents=True, exist_ok=True)
        return

    def _retrieve_sim_id_str(self):
        sim_id_str = "_".join(
            f"{idx}.{val}" for idx, val in self.weather_event_indexers.items()
        )
        return f"{self.event_iloc}-{sim_id_str}"

    def _write_swmm_rainfall_dat_files(self):

        weather_timeseries = self._analysis.cfg_analysis.weather_timeseries
        weather_event_indexers = self.weather_event_indexers
        subcatchment_raingage_mapping = (
            self._system.cfg_system.subcatchment_raingage_mapping
        )
        subcatchment_raingage_mapping_gage_id_colname = (
            self._system.cfg_system.subcatchment_raingage_mapping_gage_id_colname
        )
        rainfall_units = self._analysis.cfg_analysis.rainfall_units

        df_sub_raingage_mapping = pd.read_csv(subcatchment_raingage_mapping)  # type: ignore
        sim_id_str = self.sim_id_str

        # retreieve dataframe of rainfall time series
        df_allrain = (
            self.ds_event_ts[
                df_sub_raingage_mapping[subcatchment_raingage_mapping_gage_id_colname]
                .unique()
                .astype(str)
            ]
            .reset_coords(drop=True)
            .to_dataframe()
            .dropna()
        )
        dic_rain_paths = dict()

        for gage in df_allrain:
            s_rain = df_allrain[gage]
            df_rain = pd.DataFrame(
                dict(
                    date=df_allrain.index.strftime("%m/%d/%Y"),  # type: ignore
                    time=df_allrain.index.time,  # type: ignore
                    rain=s_rain,
                )
            )
            # define filepaths and write
            fname_raindat = "grid-ind{}.dat".format(gage)
            f_out_swmm_rainfall = self.scen_paths.dir_weather_datfiles / fname_raindat
            with open(f_out_swmm_rainfall, "w+") as file:
                file.write(f";;rain gage {gage} for sim {sim_id_str} \n")
                file.write(f";;Rainfall ({rainfall_units})\n")
            df_rain.to_csv(
                f_out_swmm_rainfall, sep="\t", index=False, header=False, mode="a"
            )
            dic_rain_paths[str(gage)] = f_out_swmm_rainfall

        self.log.swmm_rainfall_dat_files.set(dic_rain_paths)
        return

    def _write_swmm_waterlevel_dat_files(self):
        storm_tide_units = self._analysis.cfg_analysis.storm_tide_units
        weather_time_series_storm_tide_datavar = (
            self._analysis.cfg_analysis.weather_time_series_storm_tide_datavar
        )
        weather_timeseries = self._analysis.cfg_analysis.weather_timeseries
        weather_event_indexers = self.weather_event_indexers

        # ds_event_ts = ds_event_weather_series.sel(weather_event_indexers)
        # df_sub_raingage_mapping = pd.read_csv(subcatchment_raingage_mapping)
        sim_id_str = self.sim_id_str

        s_wlevel = (
            self.ds_event_ts[weather_time_series_storm_tide_datavar]
            .reset_coords(drop=True)
            .to_dataframe()
            .dropna()
        )[weather_time_series_storm_tide_datavar]

        fname_wleveldat = "waterlevel.dat"

        f_out_swmm_wlevel = self.scen_paths.dir_weather_datfiles / fname_wleveldat

        # create data frame with proper formatting to be read in SWMM
        df_wlevel = pd.DataFrame(
            dict(
                date=s_wlevel.index.strftime("%m/%d/%Y"),  # type: ignore
                time=s_wlevel.index.time,  # type: ignore
                water_level_m=s_wlevel,
            )
        )

        with open(f_out_swmm_wlevel, "w+") as file:
            file.write(f";;water level for sim {sim_id_str}\n")
            file.write(f";;Water Level ({storm_tide_units})\n")

        df_wlevel.to_csv(
            f_out_swmm_wlevel, sep="\t", index=False, header=False, mode="a"
        )
        self.log.storm_tide_for_swmm.set(f_out_swmm_wlevel)
        return

    def _create_swmm_model_from_template(self, swmm_model_template, destination):
        weather_timeseries = self._analysis.cfg_analysis.weather_timeseries
        weather_event_indexers = self.weather_event_indexers
        weather_time_series_timestep_dimension_name = (
            self._analysis.cfg_analysis.weather_time_series_timestep_dimension_name
        )

        ds_event_ts = self.ds_event_ts

        tstep_seconds = (
            ds_event_ts[weather_time_series_timestep_dimension_name]
            .to_series()
            .diff()
            .mode()
            .iloc[0]
            .total_seconds()  # type:ignore
        )
        # interval = self.seconds_to_hhmm(tstep_seconds)
        interval = self.seconds_to_hhmmss(tstep_seconds)

        # ds_event_ts = ds_event_weather_series.sel(weather_event_indexers)
        if self._analysis.cfg_analysis.rainfall_units == "mm/hr":
            format = "INTENSITY"
        elif self._analysis.cfg_analysis.rainfall_units == "mm":
            format = "DEPTH"
        else:
            raise ValueError(
                f"Invalid rainfall units specified. Expecting mm/hr or mm but got {self._analysis.cfg_analysis.rainfall_units}"
            )

        # create time series attributes
        fs = self.log.swmm_rainfall_dat_files.get()

        lst_tseries_section = []
        rain_gages_section = []
        for key, val in fs.items():
            lst_tseries_section.append(f'{key} FILE "{val}"')
            row = f"{key} {format} {interval} 1.0 TIMESERIES {key}"
            rain_gages_section.append(row)
        if self._analysis.cfg_analysis.toggle_storm_tide_boundary:
            lst_tseries_section.append(
                f'water_level FILE "{self.log.storm_tide_for_swmm.get()}"'
            )

        mapping = dict()
        mapping["TIMESERIES"] = "\n\n".join(lst_tseries_section)
        mapping["RAINGAGES"] = "\n".join(rain_gages_section)
        template_keys = utils.find_all_keys_in_template(swmm_model_template)

        first_tstep = (
            ds_event_ts[weather_time_series_timestep_dimension_name].to_series().min()
        )
        last_tstep = (
            ds_event_ts[weather_time_series_timestep_dimension_name].to_series().max()
        )

        mapping["START_DATE"] = first_tstep.strftime("%m/%d/%Y")
        mapping["START_TIME"] = first_tstep.strftime("%H:%M:%S")
        mapping["REPORT_START_DATE"] = mapping["START_DATE"]
        mapping["REPORT_START_TIME"] = mapping["START_TIME"]
        mapping["END_DATE"] = last_tstep.strftime("%m/%d/%Y")
        mapping["END_TIME"] = last_tstep.strftime("%H:%M:%S")
        mapping["REPORT_STEP"] = self.seconds_to_hhmmss(
            self._analysis.cfg_analysis.TRITON_reporting_timestep_s
        )

        for key in template_keys:
            missing_keys = []
            if key not in mapping.keys():
                missing_keys.append(key)
            if len(missing_keys) > 0:
                print(
                    f"One or more keys were not found in the dictionary defining template fill values."
                )
                print(f"Missing keys: {missing_keys}")
                print(f"All expected keys: {template_keys}")
                print(f"All keys accounted for: {mapping.keys()}")
                sys.exit()

        utils.create_from_template(swmm_model_template, mapping, destination)
        return

    def seconds_to_hhmm(self, seconds):
        seconds = int(seconds)
        h, rem = divmod(int(seconds), 3600)
        return f"{h}:{rem//60:02d}"

    def seconds_to_hhmmss(self, seconds: int | float) -> str:
        seconds = int(seconds)
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _run_swmm_hydro_model(self, rerun_if_exists=False, verbose=False):
        sim_complete = self.log.hydro_swmm_sim_completed.get() == True
        if (not sim_complete) or rerun_if_exists:
            self.log.hydro_swmm_sim_completed.set(False)
            with Simulation(str(self.scen_paths.inp_hydro)) as sim:
                sim.execute()
            self.log.hydro_swmm_sim_completed.set(True)
        else:
            if verbose:
                print("Hydrology-only SWMM model already executed. Not re-running.")
        return

    def _create_external_boundary_condition_files(self):
        weather_timeseries = self._analysis.cfg_analysis.weather_timeseries
        weather_event_indexers = self.weather_event_indexers
        weather_time_series_storm_tide_datavar = (
            self._analysis.cfg_analysis.weather_time_series_storm_tide_datavar
        )
        simulation_folders = self._analysis.analysis_paths.simulation_directory
        storm_tide_units = self._analysis.cfg_analysis.storm_tide_units

        dem_processed = self._system.sys_paths.dem_processed
        storm_tide_boundary_line_gis = (
            self._analysis.cfg_analysis.storm_tide_boundary_line_gis
        )
        ds_event_ts = self.ds_event_ts
        # ds_event_ts = ds_event_weather_series.sel(weather_event_indexers)
        df_water_levels = (
            ds_event_ts[weather_time_series_storm_tide_datavar]
            .reset_coords(drop=True)
            .to_dataframe()
            .dropna()
        )[weather_time_series_storm_tide_datavar].to_frame()

        tseries_diff_hrs = pd.Series(df_water_levels.index).diff().dt.seconds / 60 / 60  # type: ignore
        tseries_diff_hrs.loc[0] = 0

        df_water_levels["time_hr"] = tseries_diff_hrs.cumsum().values
        df_water_levels.set_index("time_hr", inplace=True)

        sim_id_str = self.sim_id_str

        wlevel_first_line = f"%{sim_id_str} Water Level Boundary Condition\n"
        wlevel_second_line = f"%Time(hr) water_elevation ({storm_tide_units})\n"

        with open(self.scen_paths.extbc_tseries, "w") as f:
            f.write(wlevel_first_line + wlevel_second_line)

        df_water_levels.to_csv(self.scen_paths.extbc_tseries, mode="a", header=False)

        self.log.extbc_tseries_created.set(True)
        # write external boundary condition location file
        rds_dem = rxr.open_rasterio(dem_processed)
        gdf_bc = gpd.read_file(storm_tide_boundary_line_gis)  # type: ignore
        str_line1 = "% BC Type, X1, Y1, X2, Y2, BC"
        gdf_row = gdf_bc.loc[0, :]
        vertices = extract_vertex_coordinates(gdf_row.geometry)
        lst_x = []
        lst_y = []
        for vertex in vertices:  # type: ignore
            # print(vertex)
            lst_x.append(vertex[0])
            lst_y.append(vertex[1])
        # find x and ys at edge of DEM representing the boundary condition
        min_x = min(lst_x)
        min_y = min(lst_y)
        max_x = max(lst_x)
        max_y = max(lst_y)
        BC_side = infer_side(rds_dem, min_x, max_x, min_y, max_y)
        x1, y1 = find_closest_dem_coord(min_x, min_y, BC_side, rds_dem)
        x2, y2 = find_closest_dem_coord(max_x, max_y, BC_side, rds_dem)
        BC_with_quotes = f'"{self.scen_paths.extbc_tseries.name}"'
        str_line2 = "{},{},{},{},{},{}".format("1", x1, y1, x2, y2, BC_with_quotes)
        # write file

        fpath_extbc = self.scen_paths.extbc_loc
        with open(fpath_extbc, "w") as f:
            f.write(str_line1 + "\n")
            f.write(str_line2 + "\n")
        self.log.extbc_loc_created.set(True)
        return

    def _write_hydrograph_files(self):
        dem_processed = self._system.sys_paths.dem_processed

        sim_id_str = self.sim_id_str
        hydro_outfile = str(self.scen_paths.inp_hydro).replace(".inp", ".out")
        rds_dem = rxr.open_rasterio(dem_processed)
        df_node_locs, lst_outfalls = return_df_of_nodes_grouped_by_DEM_gridcell(
            self.scen_paths.inp_hydro, dem_processed
        )

        d_time_series = dict()
        lst_nodes_with_inflow = []
        with Output(hydro_outfile) as out:
            flow_units = out.units["flow"]  # type: ignore
            inflow_first_line = f"%Runoff for sim {sim_id_str}\n"
            inflow_second_line = f"%Time(hr) Discharge ({flow_units})\n"
            # for key in out.nodes:
            need_to_create_time_series = True
            for coords, group in df_node_locs.groupby(["dem_x_coord", "dem_y_coord"]):
                keys = list(
                    group.node_key
                )  # list of node ids that fall within a single gridcell
                d_flows = {}
                for key in keys:
                    if key not in lst_outfalls:
                        d_inflow = pd.Series(
                            out.node_series(key, NodeAttribute.TOTAL_INFLOW)  # type: ignore
                        )
                        if (
                            need_to_create_time_series
                        ):  # create first column with time in hours
                            # tstep_seconds = pd.Series(d_inflow.index).diff().mode().dt.seconds.astype(float)
                            tseries = pd.Series(d_inflow.index).diff().dt.seconds / 60 / 60  # type: ignore
                            tseries.iloc[0] = 0
                            d_time_series["time_hr"] = tseries.cumsum().values
                            need_to_create_time_series = False
                        # create dataframe with the flow of all nodes within the gridcell
                        if d_inflow.sum() > 0:  # type: ignore
                            lst_nodes_with_inflow.append(key)
                            d_flows[key] = d_inflow.values
                # combine time series into a dataframe
                if len(d_flows) > 0:
                    df_flows = pd.DataFrame(d_flows)
                    d_time_series[coords] = df_flows.sum(axis=1)
            # write hydrograph file
            df_node_inflow = pd.DataFrame(d_time_series)

            with open(self.scen_paths.hyg_timeseries, "w") as f:
                f.write(inflow_first_line + inflow_second_line)
            df_node_inflow.to_csv(
                self.scen_paths.hyg_timeseries, mode="a", index=False, header=False
            )
            self.log.hyg_timeseries_created.set(True)
            # write hydrograph location file
            str_first_line = "%X-Location,Y-Location"
            with open(self.scen_paths.hyg_locs, "w") as f:
                f.write(str_first_line + "\n")
                for col in df_node_inflow.columns:
                    if "time" in col:  # skip column named time
                        continue
                    x = col[0]
                    y = col[1]
                    f.write("{},{}\n".format(x, y))
            self.log.hyg_locs_created.set(True)
            # verifying that all nodes are within the DEM
            xllcorner = rds_dem.x.values.min()  # type: ignore
            yllcorner = rds_dem.y.values.min()  # type: ignore
            df_xylocs = pd.read_csv(
                self.scen_paths.hyg_locs, header=0, names=["x", "y"]
            )
            # print("min x node: {}, min x DEM: {}".format(df_xylocs.x.min(), xllcorner))
            # print("min y node: {}, min y DEM: {}".format(df_xylocs.y.min(), yllcorner))
            if df_xylocs.x.min() < xllcorner:
                print("problem with x's")
            elif df_xylocs.y.min() < yllcorner:
                print("problem with y's")
            else:
                pass
            # check to make sure dimensions are correct
            df_hyg_loc = pd.read_csv(self.scen_paths.hyg_locs)
            df_hyg_test = pd.read_csv(self.scen_paths.hyg_timeseries, skiprows=2)
            if ((df_hyg_test.shape[1] - 1) - df_hyg_loc.shape[0]) != 0:
                print("ERROR ENCOUNTERED IN SETTING UP INPUTS")
                print(
                    "The shapes of the hydrograph file and the hydrograph location file do not match up."
                )
                print(f"{Path(self.scen_paths.hyg_locs).parent}")
                print("df_hyg_test.shape")
                print(df_hyg_test.shape)
                print("df_hyg_loc.shape")
                print(df_hyg_loc.shape)
                print("df_hyg_test.head()")
                print(df_hyg_test.head())
                print("df_hyg_loc.head()")
                print(df_hyg_loc.head())
                sys.exit()
        return

    def _update_hydraulics_model_to_have_1_inflow_node_per_DEM_gridcell(
        self, verbose=False
    ):
        dem_processed = self._system.sys_paths.dem_processed

        # simulation_folders = self._analysis.analysis_paths.simulation_directory
        # weather_event_indexers = self.weather_event_indexers
        """
        Makes the most downstream node in each TRITON grid cell into an 'inflow' node by assigning a dummy zerod out time series.
        """

        df_node_locs, lst_outfalls = return_df_of_nodes_grouped_by_DEM_gridcell(
            self.scen_paths.inp_hydraulics, dem_processed, verbose=verbose
        )
        lst_grps_more_than_1_node = []
        # gridcell_id = []
        count = -1
        for coords, group in df_node_locs.groupby(["dem_x_coord", "dem_y_coord"]):
            keys = list(group.node_key)
            if len(group) > 1:
                count += 1
                group["grid_id"] = count
                lst_grps_more_than_1_node.append(group)
        if verbose:
            print(
                "There are {} gridcells with more than 1 node.".format(
                    len(lst_grps_more_than_1_node)
                )
            )
        df_overlapping_nodes = pd.concat(lst_grps_more_than_1_node)
        # loop through to identify overlapping nodes
        # count = -1
        model = swmmio.Model(str(self.scen_paths.inp_hydraulics))
        links = model.links.dataframe
        nodes = model.nodes.dataframe
        inflows = model.inp.inflows
        all_upstream_nodes_to_drop = []
        lst_ambiguous_nodes = []
        for grid_id, group in df_overlapping_nodes.groupby(["grid_id"]):
            keys = list(group.node_key)
            # count += 1
            downstream_links = links[links.InletNode.isin(keys)]
            upstream_nodes_to_drop = (
                []
            )  # initiate list of nodes to remove as inflow nodes
            if len(downstream_links) == 0:
                # this means that these two nodes do not have downstream conduits
                node_to_keep = inflows[inflows.index.isin(keys)].index.values
                if len(node_to_keep) == 1:
                    # unambiguous
                    upstream_nodes_to_drop = upstream_nodes_to_drop + list(
                        np.asarray(keys)[np.asarray(keys) != node_to_keep[0]]
                    )  # drop all other nodes
                else:
                    node_to_keep = find_lowest_inv(node_to_keep, nodes)
                    if len(node_to_keep) > 1:
                        if verbose:
                            print("###########################################")
                            print(
                                "Warning: node selection ambiguous for nodes {} even after trying to select based on the lowest invert elevation. Choose the first node in the list...".format(
                                    list(node_to_keep)
                                )
                            )
                            print("###########################################")
                        if node_to_keep not in lst_ambiguous_nodes:
                            lst_ambiguous_nodes.append(node_to_keep)
                    node_to_keep = node_to_keep[0]
            else:
                keys = list(
                    downstream_links.InletNode
                )  # this ensures the order of keys and the order of the links are identical
                # compute link area
                lst_areas = []
                for id, row in downstream_links.iterrows():
                    lst_areas.append(calc_area(row))
                # rank pipe area
                ranks = rankdata(
                    lst_areas, method="min"
                )  # largest number will have highest rank
                # find unique ranks
                df_rank_counts = pd.DataFrame(np.unique(ranks, return_counts=True)).T
                df_rank_counts.columns = ["rank", "count"]
                df_rank_counts = df_rank_counts.set_index("rank")
                # if there is more than 1 link tied for largest area (/highest rank), try to find the most downstream link
                # add smaller nodes to the list of nodes to drop
                upstream_nodes_to_drop += list(np.asarray(keys)[ranks != max(ranks)])
                if df_rank_counts.loc[max(ranks), "count"] > 1:
                    downstream_links_tied = downstream_links[ranks == max(ranks)]
                    # if a node appears as an outlet node, drop it because that means the other node is downstream (want to use the most downstream)
                    downstream_links_w_outlets = downstream_links_tied[
                        downstream_links_tied.OutletNode.isin(keys)
                    ]
                    upstream_nodes_to_drop += list(
                        downstream_links_w_outlets.InletNode.values
                    )
                    # remove nodes that don't have an outlet node (meaning there is a more downstream node in the gridcell)
                    remaining_dwnstrm_links = downstream_links_tied[
                        ~downstream_links_tied.InletNode.isin(upstream_nodes_to_drop)
                    ]
                    node_to_keep = remaining_dwnstrm_links.InletNode.values
                    if len(node_to_keep) > 1:
                        # try to use the node with the lowest elevation
                        # print("Warning: node selection ambiguous for nodes {}. Attempting to use a node at lower elevation...".format(list(node_to_keep)))
                        node_to_keep = find_lowest_inv(node_to_keep, nodes)
                        if len(node_to_keep) > 1:
                            if verbose:
                                print("###########################################")
                                print(
                                    "Warning: node selection ambiguous for nodes {} even after trying to select based on the lowest invert elevation. Choose the first node in the list...".format(
                                        list(node_to_keep)
                                    )
                                )
                                print("###########################################")
                        node_to_keep = node_to_keep[0]
                        upstream_nodes_to_drop = downstream_links[
                            downstream_links.InletNode != node_to_keep
                        ].InletNode.values
                else:  # if there is a link with the largest area, use the upstream node as the inflow node
                    idx_largest_pipe = np.argmax(lst_areas)
                    node_to_keep = keys[idx_largest_pipe]
                    upstream_nodes_to_drop = downstream_links[
                        downstream_links.InletNode != node_to_keep
                    ].InletNode.values
            all_upstream_nodes_to_drop = all_upstream_nodes_to_drop + list(
                upstream_nodes_to_drop
            )
        if verbose:
            print("Removing {} inflow nodes.".format(len(all_upstream_nodes_to_drop)))
        # write new swmm model, removing outflow that won't be used by TRITON-SWMM
        with open(self.scen_paths.inp_hydraulics, "r") as fp:
            # read an store all lines into list
            lines = fp.readlines()
        for idx_line, line in enumerate(lines):
            if "[INFLOWS]" in line:
                inflows_section_first_line_num = idx_line
            if "[CURVES]" in line:
                inflows_section_last_line_num = idx_line
        line_nums_to_remove = []
        lines_to_remove = []
        for inflow_line in np.arange(
            inflows_section_first_line_num, inflows_section_last_line_num + 1
        ):
            line = lines[inflow_line]
            node_id = line.split("    ")[0]
            if node_id in all_upstream_nodes_to_drop:
                line_nums_to_remove.append(inflow_line)
                lines_to_remove.append(line)
        # overwrite hydaulics model
        with open(self.scen_paths.inp_hydraulics, "w") as fp:
            for number, line in enumerate(lines):
                if number not in line_nums_to_remove:
                    fp.write(line)
        self.log.inflow_nodes_in_hydraulic_inp_assigned = True
        return

    def _generate_TRITON_SWMM_cfg(self):
        use_constant_mannings = self._system.cfg_system.toggle_use_constant_mannings
        dem_processed = self._system.sys_paths.dem_processed
        manhole_diameter = self._analysis.cfg_analysis.manhole_diameter
        manhole_loss_coefficient = self._analysis.cfg_analysis.manhole_loss_coefficient
        TRITON_raw_output_type = self._analysis.cfg_analysis.TRITON_raw_output_type
        mannings_processed = self._system.sys_paths.mannings_processed
        constant_mannings = self._system.cfg_system.constant_mannings
        hydraulic_timestep_s = self._analysis.cfg_analysis.hydraulic_timestep_s
        TRITON_reporting_timestep_s = (
            self._analysis.cfg_analysis.TRITON_reporting_timestep_s
        )
        open_boundaries = self._analysis.cfg_analysis.open_boundaries
        triton_swmm_configuration_template = (
            self._system.cfg_system.triton_swmm_configuration_template
        )

        if use_constant_mannings:
            const_man_toggle = ""
            man_file_toggle = "#"
        else:
            const_man_toggle = "#"
            man_file_toggle = ""

        swmmmodel = swmmio.Model(str(self.scen_paths.inp_hydro))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim_options = swmmmodel.inp.options
        start_datetime = pd.to_datetime(
            sim_options.Value.START_DATE + " " + sim_options.Value.START_TIME
        )
        end_datetime = pd.to_datetime(
            sim_options.Value.END_DATE + " " + sim_options.Value.END_TIME
        )
        sim_dur_s = int((end_datetime - start_datetime) / np.timedelta64(1, "s"))

        df_extbc_loc = pd.read_csv(self.scen_paths.extbc_loc)
        num_ext_bc = len(df_extbc_loc)

        df_src_loc = pd.read_csv(self.scen_paths.hyg_locs)
        num_srcs = len(df_src_loc)

        sim_id_str = self.sim_id_str

        mapping = dict(
            CASE_DESC=sim_id_str,
            DEM=dem_processed,
            SWMM=self.scen_paths.inp_hydraulics,
            MH_DIAM=manhole_diameter,
            MH_LOSS=manhole_loss_coefficient,
            NUM_SOURCES=num_srcs,
            OUT_FORMAT=TRITON_raw_output_type.upper(),
            HYDROGRAPH=self.scen_paths.hyg_timeseries,
            HYDO_SRC_LOC=self.scen_paths.hyg_locs,
            MANNINGS=mannings_processed,
            CONST_MAN_TOGGLE=const_man_toggle,
            MAN_FILE_TOGGLE=man_file_toggle,
            CONST_MAN=constant_mannings,
            NUM_EXT_BC=num_ext_bc,
            EXTBC_DIR=str(self.scen_paths.extbc_loc.parent),
            EXTBC_FILE=self.scen_paths.extbc_loc,
            SIM_DUR_S=sim_dur_s,
            TSTEP_S=hydraulic_timestep_s,
            REPORTING_TSTEP_S=TRITON_reporting_timestep_s,
            OPEN_BOUNDARIES=open_boundaries,
        )
        utils.create_from_template(
            triton_swmm_configuration_template, mapping, self.scen_paths.triton_swmm_cfg
        )
        self.log.triton_swmm_cfg_created.set(True)
        return

    def _copy_tritonswmm_build_folder_to_sim(self):
        compiled_software_directory = (
            self._analysis.analysis_paths.compiled_software_directory
        )
        sim_tritonswmm_executable = self.scen_paths.sim_tritonswmm_executable

        src_build_fpath = compiled_software_directory / "build/"
        target_build_fpath = sim_tritonswmm_executable.parent
        if target_build_fpath.exists():
            shutil.rmtree(target_build_fpath)
        shutil.copytree(src_build_fpath, target_build_fpath)
        self.log.sim_tritonswmm_executable_copied.set(True)
        return

    def _write_sim_weather_nc(self):
        weather_timeseries = self._analysis.cfg_analysis.weather_timeseries
        weather_event_indexers = self.weather_event_indexers
        with lock:
            with xr.open_dataset(
                weather_timeseries, engine="h5netcdf"
            ) as ds_event_weather_series:
                ds_event_ts = ds_event_weather_series.sel(weather_event_indexers).load()
                utils.write_netcdf(
                    ds_event_ts,
                    self.scen_paths.weather_timeseries,
                    compression_level=5,
                    chunks="auto",
                )

    @property
    def ds_event_ts(self):
        if not self.scen_paths.weather_timeseries.exists():
            self._write_sim_weather_nc()
        return xr.open_dataset(self.scen_paths.weather_timeseries, engine="h5netcdf")

    def prepare_scenario(
        self,
        overwrite_scenario: bool = False,
        rerun_swmm_hydro_if_outputs_exist: bool = False,
    ):

        # Halt if scenario already complete
        if self.log.scenario_creation_complete.get() and not overwrite_scenario:
            print(  # type: ignore
                "Simulation already successfully created. "
                "If you wish to overwrite it, re-run with overwrite_scenario=True.",
                flush=True,
            )
            return

        # Main scenario setup
        self._write_sim_weather_nc()
        self._write_swmm_rainfall_dat_files()
        self._write_swmm_waterlevel_dat_files()
        self._create_swmm_model_from_template(
            self._system.cfg_system.SWMM_hydraulics,
            self.scen_paths.inp_hydraulics,
        )
        self.log.inp_hydraulics_model_created_successfully.set(True)

        if self._system.cfg_system.toggle_full_swmm_model:
            self._create_swmm_model_from_template(
                self._system.cfg_system.SWMM_full,
                self.scen_paths.inp_full,
            )
            self.log.inp_full_model_created_successfully.set(True)

        if self._system.cfg_system.toggle_use_swmm_for_hydrology:
            self._create_swmm_model_from_template(
                self._system.cfg_system.SWMM_hydrology,
                self.scen_paths.inp_hydro,
            )
            self._run_swmm_hydro_model(
                rerun_if_exists=rerun_swmm_hydro_if_outputs_exist,
                verbose=False,
            )
            self.log.inp_hydro_model_created_successfully.set(True)

        self._create_external_boundary_condition_files()
        self._write_hydrograph_files()
        self._update_hydraulics_model_to_have_1_inflow_node_per_DEM_gridcell(
            verbose=False
        )
        self._generate_TRITON_SWMM_cfg()
        self._copy_tritonswmm_build_folder_to_sim()

        self.log.scenario_creation_complete.set(True)
        print("Scenario preparation complete", flush=True)

        return

    def _create_subprocess_prepare_scenario_launcher(
        self,
        overwrite_scenario: bool = False,
        rerun_swmm_hydro_if_outputs_exist: bool = False,
        verbose: bool = False,
    ):
        """
        Create a launcher function that runs scenario preparation in a subprocess.

        This isolates PySwmm to a separate process, avoiding MultiSimulationError
        when preparing multiple scenarios concurrently.

        Parameters
        ----------
        event_iloc : int
            Integer index of the scenario to prepare
        overwrite_scenario : bool
            If True, overwrite existing scenario
        rerun_swmm_hydro_if_outputs_exist : bool
            If True, rerun SWMM hydrology model even if outputs exist
        verbose : bool
            If True, print progress messages

        Returns
        -------
        callable
            A launcher function that executes the subprocess
        """
        import os
        import subprocess

        event_iloc = self.event_iloc
        scenario_logfile = self.log.logfile.parent / f"scenario_prep_{event_iloc}.log"

        # Detect SLURM environment
        in_slurm = "SLURM_JOB_ID" in os.environ.copy()

        # Build command
        if in_slurm:
            # Use srun for single-core scenario preparation on HPC
            cmd = [
                "srun",
                "--ntasks=1",
                "--cpus-per-task=1",
                "python",
                "-m",
                "TRITON_SWMM_toolkit.prepare_scenario_runner",
                "--event-iloc",
                str(event_iloc),
                "--analysis-config",
                str(self._analysis.analysis_config_yaml),
                "--system-config",
                str(self._system.system_config_yaml),
            ]
        else:
            # Direct Python execution on desktop
            cmd = [
                "python",
                "-m",
                "TRITON_SWMM_toolkit.prepare_scenario_runner",
                "--event-iloc",
                str(event_iloc),
                "--analysis-config",
                str(self._analysis.analysis_config_yaml),
                "--system-config",
                str(self._system.system_config_yaml),
            ]

        # Add optional flags
        if overwrite_scenario:
            cmd.append("--overwrite-scenario")
        if rerun_swmm_hydro_if_outputs_exist:
            cmd.append("--rerun-swmm-hydro")

        def launcher():
            """Execute scenario preparation in a subprocess."""
            if verbose:
                print(
                    f"[Scenario {event_iloc}] Launching subprocess: {' '.join(cmd)}",
                    flush=True,
                )

            # Open log file for subprocess output
            with open(scenario_logfile, "w") as lf:
                proc = subprocess.Popen(
                    cmd,
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                    env=os.environ.copy(),
                )

                # Wait for subprocess to complete
                rc = proc.wait()

                if verbose:
                    if rc == 0:
                        print(
                            f"[Scenario {event_iloc}] Subprocess completed successfully",
                            flush=True,
                        )
                    else:
                        print(
                            f"[Scenario {event_iloc}] Subprocess failed with return code {rc}",
                            flush=True,
                        )

                if rc != 0:
                    # Log the error
                    if scenario_logfile.exists():
                        with open(scenario_logfile, "r") as f:
                            error_output = f.read()
                        if verbose:
                            print(
                                f"[Scenario {event_iloc}] Subprocess output:\n{error_output}",
                                flush=True,
                            )

        return launcher


def return_tstep_in_hrs(time_indexed_pd_obj):
    tstep_sim_tseries = pd.Series(time_indexed_pd_obj.index.diff()).mode()[0]
    tstep_sim_tseries_h = tstep_sim_tseries / np.timedelta64(1, "h")
    return tstep_sim_tseries_h


def return_tstep_in_hrs_for_weather_time_series(
    ds_tseries, weather_time_series_timestep_dimension_name
):
    time_indexed_pd_obj = ds_tseries[
        weather_time_series_timestep_dimension_name
    ].to_dataframe()
    return return_tstep_in_hrs(time_indexed_pd_obj)


def extract_vertex_coordinates(geometry):
    # Ensure the geometry is a LineString or MultiLineString
    if geometry.geom_type in ["LineString", "MultiLineString"]:
        return list(geometry.coords)
    else:
        return None


def infer_side(dem, min_x, max_x, min_y, max_y):
    dem_min_x = dem.x.values.min()
    dem_max_x = dem.x.values.max()
    dem_min_y = dem.y.values.min()
    dem_max_y = dem.y.values.max()
    if abs(min_x - max_x) > abs(min_y - max_y):
        loc = "top_or_bottom"
        if abs(max_y - dem_max_y) > abs(min_y - dem_min_y):
            loc = "bottom"
        else:
            loc = "top"
    else:
        loc = "left_or_right"
        if abs(max_x - dem_max_x) > abs(min_x - dem_min_x):
            loc = "left"
        else:
            loc = "right"
    return loc


def find_closest_dem_coord(x_val, y_val, BC_side, rds_dem):
    dem_xs = rds_dem.x.values  # + cellsize/2
    dem_ys = rds_dem.y.values
    if BC_side == "left":
        x_coord = min(dem_xs)
        y_coord = dem_ys[np.argmin(np.abs(dem_ys - y_val))]
    elif BC_side == "right":
        x_coord = max(dem_xs)
        y_coord = dem_ys[np.argmin(np.abs(dem_ys - y_val))]
    elif BC_side == "top":
        x_coord = dem_xs[np.argmin(np.abs(dem_xs - x_val))]
        y_coord = max(dem_ys)
    elif BC_side == "bottom":
        x_coord = dem_xs[np.argmin(np.abs(dem_xs - x_val))]
        y_coord = min(dem_ys)
    else:
        print("boundary condition location not defined")
    if (x_coord < min(dem_xs)) or (x_coord > max(dem_xs)):
        sys.exit("This x coordinate falls outside the domain of the DEM")
    if (y_coord < min(dem_ys)) or (y_coord > max(dem_ys)):
        sys.exit("This y coordinate falls outside the domain of the DEM")
    return x_coord, y_coord


def find_lowest_inv(node_to_keep, nodes):
    from scipy.stats import rankdata

    lst_invs = []
    for node_id in node_to_keep:
        row = nodes.loc[node_id, :]
        inv_elev = row.InvertElev
        lst_invs.append(inv_elev)
    ranks_inv = rankdata(lst_invs, method="min")
    # subset the nodes that have the lowest elevation
    node_to_keep = node_to_keep[ranks_inv == min(ranks_inv)]
    node_to_keep = list(np.unique(node_to_keep))  # type: ignore
    return node_to_keep


def return_df_of_nodes_grouped_by_DEM_gridcell(f_inp, dem_processed, verbose=False):
    rds_dem = rxr.open_rasterio(dem_processed)
    model = swmmio.Model(str(f_inp))
    warnings.filterwarnings(
        "ignore", category=UserWarning, module=r"swmmio\.utils\.dataframes"
    )
    node_coords = model.nodes.geodataframe["geometry"]
    dem_xs = rds_dem.x.values  # type: ignore
    dem_ys = rds_dem.y.values  # type: ignore
    d_node_locs = dict(node_key=[], dem_x_coord=[], dem_y_coord=[])
    lst_outfalls = list(model.nodes.geodataframe["OutfallType"].dropna().index)
    ## creating a row for each group of nodes associated with a single DEM cell (this is to make sure there is only 1 inflow node per gridcell)
    for node_id in node_coords.index:
        # verify that the node is within the dem
        node = node_coords[node_id]
        x_coord = node.x
        y_coord = node.y
        closest_dem_cell_x_ind = pd.Series(abs(dem_xs - x_coord)).idxmin()
        closest_dem_cell_y_ind = pd.Series(abs(dem_ys - y_coord)).idxmin()
        d_node_locs["node_key"].append(node_id)
        d_node_locs["dem_x_coord"].append(dem_xs[closest_dem_cell_x_ind])
        d_node_locs["dem_y_coord"].append(dem_ys[closest_dem_cell_y_ind])
        lst_out_of_bounds_nodes = []
        if (
            (x_coord < dem_xs.min())
            or (x_coord > dem_xs.max())
            or (y_coord < dem_ys.min())
            or (y_coord > dem_ys.max())
        ):
            if verbose:
                print("WARNING: node out bounds. Node ID: {}".format(node_id))
                print(
                    "dem lower left: ({},{}) | dem upper right: ({}, {})".format(
                        dem_xs.min(), dem_ys.min(), dem_xs.max(), dem_ys.max()
                    )
                )
                print("node coords: {}, {}".format(x_coord, y_coord))
            lst_out_of_bounds_nodes.append(node_id)
    ## create dataframe with node key and associated dem x and y coordinate for grouping
    df_node_locs = pd.DataFrame(d_node_locs)
    return df_node_locs, lst_outfalls


def calc_area(row):
    """calculate the cross-sectional area of a sewer segment. If the segment
    is multi-barrel, the area will reflect the total of all barrels"""
    if row.Shape == "ARCH":  # TREATING AS RECTANGULAR FOR SIMPLICITY
        h = row.Geom1
        w = row.Geom2
        area = h * w
        # print("Encountered arch cross sectional shape. Currently calculating a rectangular area assuming it's close enough.")
        return area * row.Barrels
    elif row.Shape in [
        "CIRCULAR",
        "HORIZ_ELLIPSE",
    ]:  # assuming horizontal ellipse is circular area
        d = row.Geom1
        area = 3.1415 * (d * d) / 4
        return round((area * row.Barrels), 2)
    elif "RECT" in row.Shape:
        # assume triangular bottom sections (geom3) deepens the excavated box
        return (row.Geom1 + row.Geom3) * float(row.Geom2) * row.Barrels
    elif row.Shape == "EGG":
        # assume geom1 is the span
        return row.Geom1 * 1.5 * row.Barrels
    else:
        print("shape not recognized in calc_area")
    return


# %%
