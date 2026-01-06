# %% work
from TRITON_SWMM_toolkit.examples import (
    load_norfolk_single_sim_experiment,
    load_norfolk_system_config,
)
from TRITON_SWMM_toolkit.scenario import (
    retrieve_weather_indexer_using_integer_index,
)

cfg_system = load_norfolk_system_config(download_if_exists=False)
cfg_exp_1sim = load_norfolk_single_sim_experiment()
system_directory = cfg_system.system_directory
experiment_id = cfg_exp_1sim.experiment_id
weather_events_to_simulate = cfg_exp_1sim.weather_events_to_simulate
weather_event_indices = cfg_exp_1sim.weather_event_indices

weather_event_indexers = retrieve_weather_indexer_using_integer_index(
    0, weather_events_to_simulate, weather_event_indices
)
# %% end work
from TRITON_SWMM_toolkit.scenario import (
    combine_sys_exp_and_sim_paths,
    retrieve_sim_id_str,
)


sim_master_paths = combine_sys_exp_and_sim_paths(
    system_directory, experiment_id, weather_event_indexers
)
