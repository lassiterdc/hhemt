import os
import pytest
import socket
from TRITON_SWMM_toolkit.examples import GetTS_TestCases as tst
from tests.utils import on_UVA_HPC

pytestmark = pytest.mark.skipif(not on_UVA_HPC(), reason="Only runs on UVA HPC")

# module load miniforge
# conda activate triton_swmm_toolkit


def test_load_system_and_analysis():
    nrflk_multisim_ensemble = tst.retreive_norfolk_UVA_multisim_1cpu_case(
        start_from_scratch=True
    )
    assert (
        nrflk_multisim_ensemble.system.analysis.analysis_paths.simulation_directory.exists()
    )


def test_compile_TRITONSWMM_for_cpu_sims():
    nrflk_multisim_ensemble = tst.retreive_norfolk_UVA_multisim_1cpu_case(
        start_from_scratch=False
    )
    nrflk_multisim_ensemble.system.analysis.compile_TRITON_SWMM()
    assert nrflk_multisim_ensemble.system.analysis.compilation_successful


def test_run_and_process_sims():
    import subprocess
    import time
    from TRITON_SWMM_toolkit.examples import GetTS_TestCases as tst

    nrflk_multisim_ensemble = tst.retreive_norfolk_UVA_multisim_1cpu_case(
        start_from_scratch=False
    )
    analysis = nrflk_multisim_ensemble.system.analysis
    # Submit SLURM job array
    script_path, job_id = analysis.submit_SLURM_job_array(
        prepare_scenarios=True, process_timeseries=True, verbose=True, which="TRITON"
    )
    while True:
        result = subprocess.run(
            ["squeue", "-j", job_id], capture_output=True, text=True
        )
        # Check if job_id appears in the output (job still running)
        if job_id not in result.stdout:
            break
        time.sleep(5)  # Check every 5 seconds
    analysis._update_log()

    if analysis.log.all_sims_run.get() != True:
        sims_not_run = "\n".join(analysis.scenarios_not_run)
        pytest.fail(
            f"Running TRITONSWMM ensemble failed. Scenarios not run: \n{sims_not_run}"
        )
    success_processing = analysis.log.all_TRITON_timeseries_processed.get()
    if not success_processing:
        analysis._update_log()
        analysis.log.print()
        pytest.fail(f"Processing TRITON time series failed.")


def test_consolidate_analysis_outputs():
    nrflk_multisim_ensemble = tst.retreive_norfolk_UVA_multisim_1cpu_case(
        start_from_scratch=False
    )
    analysis = nrflk_multisim_ensemble.system.analysis
    analysis.consolidate_TRITON_simulation_summaries(overwrite_if_exist=True)
    assert analysis.TRITON_analysis_summary_created
