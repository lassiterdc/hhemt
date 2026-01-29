import pytest

from TRITON_SWMM_toolkit.examples import GetTS_TestCases as tst


@pytest.fixture
def norfolk_single_sim_analysis():
    case = tst.retrieve_norfolk_single_sim_test_case(start_from_scratch=True)
    return case.system.analysis


@pytest.fixture
def norfolk_single_sim_analysis_cached():
    case = tst.retrieve_norfolk_single_sim_test_case(start_from_scratch=False)
    return case.system.analysis


@pytest.fixture
def norfolk_multi_sim_analysis():
    case = tst.retrieve_norfolk_multi_sim_test_case(start_from_scratch=True)
    return case.system.analysis


@pytest.fixture
def norfolk_multi_sim_analysis_cached():
    case = tst.retrieve_norfolk_multi_sim_test_case(start_from_scratch=False)
    return case.system.analysis


@pytest.fixture
def norfolk_sensitivity_analysis():
    case = tst.retrieve_norfolk_cpu_config_sensitivity_case(start_from_scratch=True)
    return case.system.analysis


@pytest.fixture
def norfolk_sensitivity_analysis_cached():
    case = tst.retrieve_norfolk_cpu_config_sensitivity_case(start_from_scratch=False)
    return case.system.analysis


@pytest.fixture
def norfolk_frontier_multisim_analysis():
    case = tst.retrieve_norfolk_frontier_multisim_cpu_serial_case(
        start_from_scratch=True
    )
    return case.system.analysis


@pytest.fixture
def norfolk_frontier_multisim_analysis_cached():
    case = tst.retrieve_norfolk_frontier_multisim_cpu_serial_case(
        start_from_scratch=False
    )
    return case.system.analysis


@pytest.fixture
def norfolk_frontier_multisim_gpu_analysis():
    case = tst.retrieve_norfolk_frontier_multisim_gpu_case(start_from_scratch=True)
    return case.system.analysis


# @pytest.fixture
# def norfolk_frontier_all_configs_analysis():
#     case = tst.retrieve_norfolk_frontier_all_configs(start_from_scratch=True)
#     return case.system.analysis


# @pytest.fixture
# def norfolk_frontier_all_configs_analysis_cached():
#     case = tst.retrieve_norfolk_frontier_all_configs(start_from_scratch=False)
#     return case.system.analysis


@pytest.fixture
def norfolk_uva_multisim_analysis():
    case = tst.retrieve_norfolk_UVA_multisim_1cpu_case(start_from_scratch=True)
    return case.system.analysis


@pytest.fixture
def norfolk_uva_multisim_analysis_cached():
    case = tst.retrieve_norfolk_UVA_multisim_1cpu_case(start_from_scratch=False)
    return case.system.analysis


@pytest.fixture
def norfolk_uva_sensitivity_analysis():
    case = tst.retrieve_norfolk_UVA_sensitivity_CPU_minimal(start_from_scratch=True)
    return case.system.analysis


@pytest.fixture
def norfolk_uva_sensitivity_analysis_cached():
    case = tst.retrieve_norfolk_UVA_sensitivity_CPU_minimal(start_from_scratch=False)
    return case.system.analysis


@pytest.fixture
def norfolk_uva_sensitivity_full_ensemble_analysis():
    case = tst.retrieve_norfolk_UVA_sensitivity_CPU_full_ensemble_short_sims(
        start_from_scratch=True
    )
    return case.system.analysis


@pytest.fixture
def norfolk_frontier_sensitivity_analysis():
    case = tst.retrieve_norfolk_frontier_sensitivity_minimal(start_from_scratch=True)
    return case.system.analysis


@pytest.fixture
def norfolk_frontier_sensitivity_analysis_cached():
    case = tst.retrieve_norfolk_frontier_sensitivity_minimal(start_from_scratch=False)
    return case.system.analysis
