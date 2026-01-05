# %% tests/test_TRITON_SWMM_toolkit.py
# load all libraries and constants
from TRITON_SWMM_toolkit.constants import (
    NORFOLK_EX,
    NORFOLK_CASE_CONFIG,
    NORFOLK_SYSTEM_CONFIG,
    NORFOLK_SINGLE_SIM_EXP_CONFIG,
)
from TRITON_SWMM_toolkit.examples import (
    load_config_file_as_dic,
    get_norfolk_data_and_package_directory_mapping_dict,
    load_config_filepath,
    return_filled_template_yaml_dictionary,
    load_norfolk_system_config,
    load_norfolk_single_sim_experiment,
)
from TRITON_SWMM_toolkit.config import load_system_config, load_experiment_config
from pathlib import Path
from TRITON_SWMM_toolkit.system_setup import define_system_paths
from TRITON_SWMM_toolkit.prepare_an_experiment import define_experiment_paths
from TRITON_SWMM_toolkit.prepare_a_simulation import (
    retrieve_weather_indexer_using_integer_index,
    initialize_sim_logfile,
    define_simulation_paths,
)
from TRITON_SWMM_toolkit.utils_for_testing import (
    create_reduced_weather_file_for_testing_if_it_does_not_exist,
)


# PARAMETERS AND CONSTANTS
TST_DIR_SUFFIX = "test"
DUR_MIN = 10  # for testing

# LOADING FROM SYSTEM CONFIG
cfg_system = load_norfolk_system_config(download_if_exists=False)

cfg_system.system_directory = (  # update system directory for testing
    cfg_system.system_directory.parent / f"sys_{TST_DIR_SUFFIX}"
)
landuse_lookup = cfg_system.landuse_lookup_file
landuse_raster = cfg_system.landuse_raster
landuse_colname = cfg_system.landuse_lookup_class_id_colname
mannings_colname = cfg_system.landuse_lookup_mannings_colname
watershed_shapefile = cfg_system.watershed_gis_polygon
system_directory = cfg_system.system_directory
dem_unprocessed = cfg_system.DEM_fullres
target_resolution = cfg_system.target_dem_resolution
dem_outside_watershed_height = cfg_system.dem_outside_watershed_height
dem_building_height = cfg_system.dem_building_height
TRITONSWMM_software_directory = cfg_system.TRITONSWMM_software_directory
TRITON_SWMM_software_compilation_script = (
    cfg_system.TRITON_SWMM_software_compilation_script
)
subcatchment_raingage_mapping = cfg_system.subcatchment_raingage_mapping
subcatchment_raingage_mapping_gage_id_colname = (
    cfg_system.subcatchment_raingage_mapping_gage_id_colname
)
SWMM_hydrology = cfg_system.SWMM_hydrology
SWMM_hydraulics = cfg_system.SWMM_hydraulics
SWMM_full = cfg_system.SWMM_full
use_constant_mannings = cfg_system.toggle_use_constant_mannings
constant_mannings = cfg_system.constant_mannings
triton_swmm_configuration_template = cfg_system.triton_swmm_configuration_template

# LOADING FROM EXPERIMENT CONFIG
cfg_exp_1sim = load_norfolk_single_sim_experiment()
experiment_id = cfg_exp_1sim.experiment_id
TRITON_SWMM_make_command = cfg_exp_1sim.TRITON_SWMM_make_command
weather_time_series_storm_tide_datavar = (
    cfg_exp_1sim.weather_time_series_storm_tide_datavar
)
weather_time_series_timestep_dimension_name = (
    cfg_exp_1sim.weather_time_series_timestep_dimension_name
)
rainfall_units = cfg_exp_1sim.rainfall_units
storm_tide_units = cfg_exp_1sim.storm_tide_units
storm_tide_boundary_line_gis = cfg_exp_1sim.storm_tide_boundary_line_gis
TRITON_output_type = cfg_exp_1sim.TRITON_output_type
manhole_diameter = cfg_exp_1sim.manhole_diameter
manhole_loss_coefficient = cfg_exp_1sim.manhole_loss_coefficient
hydraulic_timestep_s = cfg_exp_1sim.hydraulic_timestep_s
TRITON_reporting_timestep_s = cfg_exp_1sim.TRITON_reporting_timestep_s
open_boundaries = cfg_exp_1sim.open_boundaries
weather_events_to_simulate = cfg_exp_1sim.weather_events_to_simulate
weather_event_indices = cfg_exp_1sim.weather_event_indices

# SUBSETTING WEATHER DATA FOR TESTING
og_weather_timeseries = cfg_exp_1sim.weather_timeseries
weather_timeseries = system_directory / "weather_subset.nc"
create_reduced_weather_file_for_testing_if_it_does_not_exist(
    og_weather_timeseries,
    weather_timeseries,
    weather_events_to_simulate,
    weather_event_indices,
    weather_time_series_timestep_dimension_name,
    dur_min=DUR_MIN,
)


# LOADING VARS BASED ON CONFIGS
exp_paths = define_experiment_paths(experiment_id, system_directory)
simulation_folders = exp_paths["simulation_directory"]
compiled_software_directory = exp_paths["compiled_software_directory"]

sys_paths = define_system_paths(system_directory)
mannings_processed = sys_paths["mannings_processed"]
dem_processed = sys_paths["dem_processed"]

weather_event_indexers = retrieve_weather_indexer_using_integer_index(
    0, weather_events_to_simulate, weather_event_indices
)
sim_paths = define_simulation_paths(simulation_folders, weather_event_indexers)
sim_tritonswmm_executable = sim_paths["sim_tritonswmm_executable"]


def test_load_system_config():
    # norfolk system
    case_details = load_config_file_as_dic(NORFOLK_EX, NORFOLK_CASE_CONFIG)
    res_identifier = case_details["res_identifier"]  # will come from the case yaml
    mapping = get_norfolk_data_and_package_directory_mapping_dict()
    cfg_template = load_config_filepath(NORFOLK_EX, NORFOLK_SYSTEM_CONFIG)
    cfg_filled_yaml = return_filled_template_yaml_dictionary(cfg_template, mapping)
    cfg_system = load_system_config(cfg_filled_yaml)


def test_load_experiment_config():
    # norfolk single sim experiment
    mapping = get_norfolk_data_and_package_directory_mapping_dict()
    cfg_template = load_config_filepath(NORFOLK_EX, NORFOLK_SINGLE_SIM_EXP_CONFIG)
    cfg_filled_yaml = return_filled_template_yaml_dictionary(cfg_template, mapping)
    cfg_experiment = load_experiment_config(cfg_filled_yaml)


def test_create_mannings_file_for_TRITON():
    from TRITON_SWMM_toolkit.system_setup import create_mannings_file_for_TRITON

    rds_dem_coarse = create_mannings_file_for_TRITON(
        landuse_lookup,
        landuse_raster,
        landuse_colname,
        mannings_colname,
        dem_unprocessed,
        target_resolution,
        system_directory,
        include_metadata=False,  # sometimes the header in the DEM file causes TRITON-SWMM to misread the file
    )


def test_create_dem_for_TRITON():
    from TRITON_SWMM_toolkit.system_setup import create_dem_for_TRITON

    rds_mannings_coarse = create_dem_for_TRITON(
        dem_unprocessed, target_resolution, system_directory
    )


def test_compile_TRITONSWMM_for_cpu_sims():
    from TRITON_SWMM_toolkit.prepare_an_experiment import compile_TRITON_SWMM

    compilation_log = compile_TRITON_SWMM(
        experiment_id,
        system_directory,
        TRITONSWMM_software_directory,
        TRITON_SWMM_make_command,
        TRITON_SWMM_software_compilation_script,
    )
    assert "[100%] Built target runswmm" in compilation_log
    assert "Building finished: triton" in compilation_log


def test_initialize_sim_logfile():
    log = initialize_sim_logfile(simulation_folders, weather_event_indexers)
    assert Path(log["f_log"]).exists()


def test_write_swmm_rainfall_dat_files():
    from TRITON_SWMM_toolkit.prepare_a_simulation import write_swmm_rainfall_dat_files

    log, dic_rain_path = write_swmm_rainfall_dat_files(
        weather_timeseries,
        weather_event_indexers,
        subcatchment_raingage_mapping,
        subcatchment_raingage_mapping_gage_id_colname,
        rainfall_units,
        simulation_folders,
    )
    for key, dat in log["swmm_rainfall_dat_files"].items():
        assert Path(dat).exists()


def test_write_swmm_waterlevel_dat_files():
    from TRITON_SWMM_toolkit.prepare_a_simulation import write_swmm_waterlevel_dat_files

    log = write_swmm_waterlevel_dat_files(
        storm_tide_units,
        weather_time_series_storm_tide_datavar,
        weather_timeseries,
        weather_event_indexers,
        simulation_folders,
    )
    assert Path(log["water_level"]).exists()


def test_create_swmm_model_from_template():
    from TRITON_SWMM_toolkit.prepare_a_simulation import create_swmm_model_from_template

    log = create_swmm_model_from_template(
        SWMM_hydrology,
        "inp_hydro",
        simulation_folders,
        weather_timeseries,
        weather_event_indexers,
        weather_time_series_timestep_dimension_name,
    )
    assert Path(log["inp_hydro"]).exists()

    log = create_swmm_model_from_template(
        SWMM_hydraulics,
        "inp_hydraulics",
        simulation_folders,
        weather_timeseries,
        weather_event_indexers,
        weather_time_series_timestep_dimension_name,
    )
    assert Path(log["inp_hydraulics"]).exists()
    log = create_swmm_model_from_template(
        SWMM_full,
        "inp_full",
        simulation_folders,
        weather_timeseries,
        weather_event_indexers,
        weather_time_series_timestep_dimension_name,
    )
    assert Path(log["inp_full"]).exists()


def test_run_swmm_hydro_model():
    from TRITON_SWMM_toolkit.prepare_a_simulation import run_swmm_hydro_model

    log = run_swmm_hydro_model(
        simulation_folders,
        weather_event_indexers,
        destination_sim_path_key="inp_hydro",
        rerun_if_exists=True,
    )
    outfiles = list(Path(log["inp_hydro"]).parent.glob("*.out"))
    assert len(outfiles) == 1


def test_create_external_boundary_condition_files():
    from TRITON_SWMM_toolkit.prepare_a_simulation import (
        create_external_boundary_condition_files,
    )

    log = create_external_boundary_condition_files(
        weather_timeseries,
        weather_event_indexers,
        weather_time_series_storm_tide_datavar,
        simulation_folders,
        storm_tide_units,
        dem_processed,
        storm_tide_boundary_line_gis,
    )
    assert Path(log["extbc_tseries"]).exists()
    assert Path(log["extbc_loc"]).exists()


def test_write_hydrograph_files():
    from TRITON_SWMM_toolkit.prepare_a_simulation import write_hydrograph_files

    log = write_hydrograph_files(
        weather_event_indexers, dem_processed, simulation_folders
    )
    assert Path(log["hyg_timeseries"]).exists()
    assert Path(log["hyg_locs"]).exists()


def test_update_hydraulics_model_to_have_1_inflow_node_per_DEM_gridcell():
    from TRITON_SWMM_toolkit.prepare_a_simulation import (
        update_hydraulics_model_to_have_1_inflow_node_per_DEM_gridcell,
    )

    log = update_hydraulics_model_to_have_1_inflow_node_per_DEM_gridcell(
        dem_processed, simulation_folders, weather_event_indexers, verbose=False
    )
    assert log["inflow_nodes_in_hydraulic_inp_assigned"] == True


def test_generate_TRITON_SWMM_cfg():
    from TRITON_SWMM_toolkit.prepare_a_simulation import generate_TRITON_SWMM_cfg

    log = generate_TRITON_SWMM_cfg(
        simulation_folders,
        weather_event_indexers,
        use_constant_mannings,
        dem_processed,
        manhole_diameter,
        manhole_loss_coefficient,
        TRITON_output_type,
        mannings_processed,
        constant_mannings,
        hydraulic_timestep_s,
        TRITON_reporting_timestep_s,
        open_boundaries,
        triton_swmm_configuration_template,
    )
    assert Path(log["triton_swmm_cfg"]).exists()


def test_copy_tritonswmm_build_folder_to_sim():
    from TRITON_SWMM_toolkit.prepare_a_simulation import (
        copy_tritonswmm_build_folder_to_sim,
    )

    log = copy_tritonswmm_build_folder_to_sim(
        compiled_software_directory,
        sim_tritonswmm_executable,
        simulation_folders,
        weather_event_indexers,
    )
    assert Path(log["sim_tritonswmm_executable"]).exists()


def test_prepare_all_sims():
    from TRITON_SWMM_toolkit.prepare_a_simulation import prepare_all_sims

    overwrite_sim = True

    log = prepare_all_sims(
        weather_events_to_simulate,
        weather_event_indices,
        overwrite_sim,
        system_directory,
        experiment_id,
        weather_timeseries,
        subcatchment_raingage_mapping,
        subcatchment_raingage_mapping_gage_id_colname,
        rainfall_units,
        storm_tide_units,
        weather_time_series_storm_tide_datavar,
        SWMM_hydrology,
        weather_time_series_timestep_dimension_name,
        SWMM_hydraulics,
        SWMM_full,
        storm_tide_boundary_line_gis,
        use_constant_mannings,
        manhole_diameter,
        manhole_loss_coefficient,
        TRITON_output_type,
        constant_mannings,
        hydraulic_timestep_s,
        TRITON_reporting_timestep_s,
        open_boundaries,
        triton_swmm_configuration_template,
    )

    for key, dat in log["swmm_rainfall_dat_files"].items():
        assert Path(dat).exists()
    assert Path(log["water_level"]).exists()
    assert Path(log["inp_hydro"]).exists()
    assert Path(log["inp_hydraulics"]).exists()
    assert Path(log["inp_full"]).exists()
    assert Path(log["extbc_tseries"]).exists()
    assert Path(log["extbc_loc"]).exists()
    assert Path(log["hyg_timeseries"]).exists()
    assert Path(log["hyg_locs"]).exists()
    assert log["inflow_nodes_in_hydraulic_inp_assigned"] == True
    assert Path(log["triton_swmm_cfg"]).exists()
    assert Path(log["sim_tritonswmm_executable"]).exists()


def test_run_singlecore_simulation():
    from TRITON_SWMM_toolkit.running_a_simulation import (
        run_singlecore_simulation,
        check_simulation_run_status,
    )

    tritonswmm_log, log = run_singlecore_simulation(
        experiment_id,
        system_directory,
        weather_event_indexers,
        pickup_where_leftoff=False,
        verbose=True,
    )

    sim_status, __ = check_simulation_run_status(
        system_directory, experiment_id, weather_event_indexers
    )
    assert sim_status == "simulation completed"
