import pytest
import tests.utils_for_testing as tst_ut

pytestmark = pytest.mark.skipif(not tst_ut.on_UVA_HPC(), reason="Only runs on UVA HPC")

# ijob \
#   -A ***REMOVED*** \
#   -p standard \
#   --time=08:00:00 \
#   -N 1 \
#  --cpus-per-task=1 \
#  --ntasks-per-node=96

# ijob \
#   -A ***REMOVED*** \
#   -p interactive \
#   --time=08:00:00 \
#   -N 1 \
#  --cpus-per-task=1 \
#  --ntasks-per-node=24

#   --gres=gpu:1 \

# module purge
# module load gompi/14.2.0_5.0.7 miniforge
# conda activate hhemt
# export PYTHONNOUSERSITE=1


def test_load_system_and_analysis(norfolk_uva_multisim_analysis):
    analysis = norfolk_uva_multisim_analysis
    tst_ut.assert_file_exists(
        analysis.analysis_paths.simulation_directory, "simulation directory"
    )


def test_create_dem_for_TRITON(norfolk_uva_multisim_analysis_cached):
    analysis = norfolk_uva_multisim_analysis_cached
    analysis._system.create_dem_for_TRITON()
    rds = analysis._system.processed_dem_rds
    assert rds.shape == (1, 537, 551)  # type: ignore


def test_create_mannings_file_for_TRITON(norfolk_uva_multisim_analysis_cached):
    analysis = norfolk_uva_multisim_analysis_cached
    analysis._system.create_mannings_file_for_TRITON()
    rds = analysis._system.mannings_rds
    assert rds.shape == (1, 537, 551)  # type: ignore


def test_compile_TRITONSWMM_for_cpu_sims(norfolk_uva_multisim_analysis_cached):
    analysis = norfolk_uva_multisim_analysis_cached
    analysis._system.compile_TRITON_SWMM(
        verbose=True,
        redownload_triton_swmm_if_exists=True,
        recompile_if_already_done_successfully=True,
    )
    assert analysis._system.compilation_successful


def test_prepare_scenarios(norfolk_uva_multisim_analysis_cached):
    analysis = norfolk_uva_multisim_analysis_cached
    prepare_scenario_launchers = analysis.retrieve_prepare_scenario_launchers(
        overwrite_scenario_if_already_set_up=True, verbose=True
    )
    analysis.run_python_functions_concurrently(
        prepare_scenario_launchers,
        verbose=True,
    )
    tst_ut.assert_scenarios_setup(analysis)
