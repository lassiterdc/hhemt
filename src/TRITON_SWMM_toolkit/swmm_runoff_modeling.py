"""
SWMM Runoff Modeling Module

This module handles SWMM hydrology-only modeling to generate runoff hydrographs
that serve as inputs to TRITON-SWMM. This includes:
- Creating rainfall and water level input files for SWMM
- Creating and running SWMM hydrology-only models
- Extracting runoff hydrographs from SWMM output for TRITON-SWMM input
"""

import pandas as pd
import sys
from pathlib import Path
from pyswmm import Simulation, Output
from swmm.toolkit.shared_enum import NodeAttribute
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .scenario import TRITONSWMM_scenario


class SWMMRunoffModeler:
    """
    Handles SWMM hydrology modeling for generating runoff inputs to TRITON-SWMM.

    This class manages the complete workflow for using SWMM to model rainfall-runoff
    processes and generate hydrograph inputs for TRITON-SWMM:
    1. Creates rainfall and water level .dat files for SWMM input
    2. Creates SWMM hydrology-only model from template
    3. Executes SWMM hydrology model
    4. Extracts runoff hydrographs and formats them for TRITON-SWMM

    Attributes
    ----------
    scenario : TRITONSWMM_scenario
        Reference to the parent scenario object
    cfg_analysis : AnalysisConfig
        Analysis configuration settings
    system : TRITONSWMM_system
        System configuration and paths
    """

    def __init__(self, scenario: "TRITONSWMM_scenario") -> None:
        """
        Initialize the SWMMRunoffModeler.

        Parameters
        ----------
        scenario : TRITONSWMM_scenario
            The parent scenario object containing configuration and paths
        """
        self.scenario = scenario
        self.cfg_analysis = scenario._analysis.cfg_analysis
        self.system = scenario._system

    def write_swmm_rainfall_dat_files(self) -> None:
        """
        Generate rainfall input files from weather data for SWMM.

        Creates .dat files for each rain gauge with time series data formatted
        for SWMM input. Files are written to the scenario's weather data directory.
        Updates the scenario log with paths to created files.
        """
        weather_timeseries = self.cfg_analysis.weather_timeseries
        weather_event_indexers = self.scenario.weather_event_indexers
        subcatchment_raingage_mapping = (
            self.system.cfg_system.subcatchment_raingage_mapping
        )
        subcatchment_raingage_mapping_gage_id_colname = (
            self.system.cfg_system.subcatchment_raingage_mapping_gage_id_colname
        )
        rainfall_units = self.cfg_analysis.rainfall_units

        df_sub_raingage_mapping = pd.read_csv(subcatchment_raingage_mapping)  # type: ignore
        sim_id_str = self.scenario.sim_id_str

        # retreieve dataframe of rainfall time series
        df_allrain = (
            self.scenario.ds_event_ts[
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
            f_out_swmm_rainfall = (
                self.scenario.scen_paths.dir_weather_datfiles / fname_raindat
            )
            with open(f_out_swmm_rainfall, "w+") as file:
                file.write(f";;rain gage {gage} for sim {sim_id_str} \n")
                file.write(f";;Rainfall ({rainfall_units})\n")
            df_rain.to_csv(
                f_out_swmm_rainfall, sep="\t", index=False, header=False, mode="a"
            )
            dic_rain_paths[str(gage)] = f_out_swmm_rainfall

        self.scenario.log.swmm_rainfall_dat_files.set(dic_rain_paths)
        return

    def write_swmm_waterlevel_dat_files(self) -> None:
        """
        Generate water level input files for SWMM boundary conditions.

        Creates a .dat file with storm tide/water level time series formatted
        for SWMM input. Updates the scenario log with the path to the created file.
        """
        storm_tide_units = self.cfg_analysis.storm_tide_units
        weather_time_series_storm_tide_datavar = (
            self.cfg_analysis.weather_time_series_storm_tide_datavar
        )
        weather_timeseries = self.cfg_analysis.weather_timeseries
        weather_event_indexers = self.scenario.weather_event_indexers

        sim_id_str = self.scenario.sim_id_str

        s_wlevel = (
            self.scenario.ds_event_ts[weather_time_series_storm_tide_datavar]
            .reset_coords(drop=True)
            .to_dataframe()
            .dropna()
        )[weather_time_series_storm_tide_datavar]

        fname_wleveldat = "waterlevel.dat"

        f_out_swmm_wlevel = (
            self.scenario.scen_paths.dir_weather_datfiles / fname_wleveldat
        )

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
        self.scenario.log.storm_tide_for_swmm.set(f_out_swmm_wlevel)
        return

    def create_hydrology_model_from_template(
        self, swmm_model_template, destination: Path
    ) -> None:
        """
        Create SWMM hydrology-only model from template file.

        Fills in template placeholders with scenario-specific values including
        time series data, rain gauges, simulation timing, and reporting intervals.

        Parameters
        ----------
        swmm_model_template : Path
            Path to the SWMM hydrology template file
        destination : Path
            Path where the filled template should be written (typically hydro.inp)
        """
        from .swmm_utils import create_swmm_inp_from_template

        create_swmm_inp_from_template(self.scenario, swmm_model_template, destination)
        return

    def run_swmm_hydro_model(
        self, rerun_if_exists: bool = False, verbose: bool = False
    ) -> None:
        """
        Execute SWMM hydrology-only model to generate runoff hydrographs.

        Runs the hydrology-only SWMM model to generate runoff hydrographs.
        Can skip execution if outputs already exist.

        Parameters
        ----------
        rerun_if_exists : bool, optional
            If True, rerun even if outputs exist (default: False)
        verbose : bool, optional
            If True, print status messages (default: False)
        """
        sim_complete = self.scenario.log.hydro_swmm_sim_completed.get() is True
        if (not sim_complete) or rerun_if_exists:
            self.scenario.log.hydro_swmm_sim_completed.set(False)
            with Simulation(str(self.scenario.scen_paths.inp_hydro)) as sim:
                sim.execute()
            self.scenario.log.hydro_swmm_sim_completed.set(True)
        else:
            if verbose:
                print("Hydrology-only SWMM model already executed. Not re-running.")
        return

    def write_hydrograph_files(self) -> None:
        """
        Extract runoff hydrographs from SWMM output and format for TRITON-SWMM.

        Extracts runoff from SWMM hydrology model output and creates:
        1. Time series file with discharge hydrographs for each DEM grid cell
        2. Location file mapping hydrographs to DEM coordinates

        These files serve as inflow boundary conditions for TRITON-SWMM.
        Updates the scenario log to indicate files were created successfully.
        """
        import rioxarray as rxr
        from .scenario import return_df_of_nodes_grouped_by_DEM_gridcell

        dem_processed = self.system.sys_paths.dem_processed

        sim_id_str = self.scenario.sim_id_str
        hydro_outfile = str(self.scenario.scen_paths.inp_hydro).replace(".inp", ".out")
        rds_dem = rxr.open_rasterio(dem_processed)
        df_node_locs, lst_outfalls = return_df_of_nodes_grouped_by_DEM_gridcell(
            self.scenario.scen_paths.inp_hydro, dem_processed
        )

        d_time_series = dict()
        lst_nodes_with_inflow = []
        with Output(hydro_outfile) as out:
            flow_units = out.units["flow"]  # type: ignore
            inflow_first_line = f"%Runoff for sim {sim_id_str}\n"
            inflow_second_line = f"%Time(hr) Discharge ({flow_units})\n"
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

            with open(self.scenario.scen_paths.hyg_timeseries, "w") as f:
                f.write(inflow_first_line + inflow_second_line)
            df_node_inflow.to_csv(
                self.scenario.scen_paths.hyg_timeseries,
                mode="a",
                index=False,
                header=False,
            )
            self.scenario.log.hyg_timeseries_created.set(True)
            # write hydrograph location file
            str_first_line = "%X-Location,Y-Location"
            with open(self.scenario.scen_paths.hyg_locs, "w") as f:
                f.write(str_first_line + "\n")
                for col in df_node_inflow.columns:
                    if "time" in col:  # skip column named time
                        continue
                    x = col[0]
                    y = col[1]
                    f.write("{},{}\n".format(x, y))
            self.scenario.log.hyg_locs_created.set(True)
            # verifying that all nodes are within the DEM
            xllcorner = rds_dem.x.values.min()  # type: ignore
            yllcorner = rds_dem.y.values.min()  # type: ignore
            df_xylocs = pd.read_csv(
                self.scenario.scen_paths.hyg_locs, header=0, names=["x", "y"]
            )
            if df_xylocs.x.min() < xllcorner:
                print("problem with x's")
            elif df_xylocs.y.min() < yllcorner:
                print("problem with y's")
            else:
                pass
            # check to make sure dimensions are correct
            df_hyg_loc = pd.read_csv(self.scenario.scen_paths.hyg_locs)
            df_hyg_test = pd.read_csv(
                self.scenario.scen_paths.hyg_timeseries, skiprows=2
            )
            if ((df_hyg_test.shape[1] - 1) - df_hyg_loc.shape[0]) != 0:
                print("ERROR ENCOUNTERED IN SETTING UP INPUTS")
                print(
                    "The shapes of the hydrograph file and the hydrograph location file do not match up."
                )
                print(f"{Path(self.scenario.scen_paths.hyg_locs).parent}")
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
