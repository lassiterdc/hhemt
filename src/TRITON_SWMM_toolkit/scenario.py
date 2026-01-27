# %%
import pandas as pd
import rioxarray as rxr
import numpy as np
import xarray as xr
import sys
import shutil
import swmmio
import warnings
import TRITON_SWMM_toolkit.utils as utils
from datetime import datetime
from TRITON_SWMM_toolkit.log import TRITONSWMM_scenario_log
from TRITON_SWMM_toolkit.paths import ScenarioPaths
from typing import TYPE_CHECKING, Literal
import threading
from TRITON_SWMM_toolkit.subprocess_utils import run_subprocess_with_tee
from TRITON_SWMM_toolkit.scenario_inputs import ScenarioInputGenerator
from TRITON_SWMM_toolkit.swmm_runoff_modeling import SWMMRunoffModeler
from TRITON_SWMM_toolkit.swmm_full_model import SWMMFullModelBuilder


lock = threading.Lock()

if TYPE_CHECKING:
    from .analysis import TRITONSWMM_analysis



class TRITONSWMM_scenario:
    log: TRITONSWMM_scenario_log

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
            sim_tritonswmm_executable=sim_folder / "build" / "triton.exe",
            tritonswmm_logfile_dir=sim_folder / "tritonswmm_sim_logfiles",
            # OUTPUTS
            output_tritonswmm_performance_timeserie=sim_folder
            / f"TRITONSWMM_perf_tseries.{self._analysis.cfg_analysis.TRITON_processed_output_type}",
            output_tritonswmm_performance_summary=sim_folder
            / f"TRITONSWMM_perf_summary.{self._analysis.cfg_analysis.TRITON_processed_output_type}",
            output_triton_timeseries=sim_folder
            / f"TRITON_tseries.{self._analysis.cfg_analysis.TRITON_processed_output_type}",
            output_swmm_link_time_series=sim_folder
            / f"SWMM_link_tseries.{self._analysis.cfg_analysis.TRITON_processed_output_type}",
            output_swmm_node_time_series=sim_folder
            / f"SWMM_node_tseries.{self._analysis.cfg_analysis.TRITON_processed_output_type}",
            output_triton_summary=sim_folder
            / f"TRITON_summary.{self._analysis.cfg_analysis.TRITON_processed_output_type}",
            output_swmm_node_summary=sim_folder
            / f"SWMM_node_summary.{self._analysis.cfg_analysis.TRITON_processed_output_type}",
            output_swmm_link_summary=sim_folder
            / f"SWMM_link_summary.{self._analysis.cfg_analysis.TRITON_processed_output_type}",
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

        # Initialize scenario preparation components
        self._input_generator = ScenarioInputGenerator(self)
        self._runoff_modeler = SWMMRunoffModeler(self)
        self._full_model_builder = SWMMFullModelBuilder(self)

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
        src_build_fpath = self._system.sys_paths.TRITON_build_dir
        sim_tritonswmm_executable = self.scen_paths.sim_tritonswmm_executable

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

        # SWMM runoff modeling - generates hydrograph inputs
        self._runoff_modeler.write_swmm_rainfall_dat_files()
        self._runoff_modeler.write_swmm_waterlevel_dat_files()

        # Create SWMM hydraulics model - direct TRITON-SWMM input
        self._input_generator.create_hydraulics_model_from_template(
            self._system.cfg_system.SWMM_hydraulics,
            self.scen_paths.inp_hydraulics,
        )
        self.log.inp_hydraulics_model_created_successfully.set(True)

        # Optional: Full SWMM model (placeholder functionality)
        if self._system.cfg_system.toggle_full_swmm_model:
            self._full_model_builder.create_full_model_from_template(
                self._system.cfg_system.SWMM_full,
                self.scen_paths.inp_full,
            )
            self.log.inp_full_model_created_successfully.set(True)

        # SWMM hydrology for runoff generation
        if self._system.cfg_system.toggle_use_swmm_for_hydrology:
            self._runoff_modeler.create_hydrology_model_from_template(
                self._system.cfg_system.SWMM_hydrology,
                self.scen_paths.inp_hydro,
            )
            self._runoff_modeler.run_swmm_hydro_model(
                rerun_if_exists=rerun_swmm_hydro_if_outputs_exist,
                verbose=False,
            )
            self.log.inp_hydro_model_created_successfully.set(True)

        # Create TRITON inputs
        self._input_generator.create_external_boundary_condition_files()
        self._runoff_modeler.write_hydrograph_files()
        self._input_generator.update_hydraulics_model_to_have_1_inflow_node_per_DEM_gridcell(
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

        event_iloc = self.event_iloc
        scenario_logfile = self.log.logfile.parent / f"scenario_prep_{event_iloc}.log"

        # Build command - always use direct Python execution (no srun)
        cmd = [
            f"{self._analysis._python_executable}",
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

            # Use tee logging to write to both file and stdout
            proc = run_subprocess_with_tee(
                cmd=cmd,
                logfile=scenario_logfile,
                env=None,  # Uses os.environ by default
                echo_to_stdout=True,
            )

            rc = proc.returncode

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
