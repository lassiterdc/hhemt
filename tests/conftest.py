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
