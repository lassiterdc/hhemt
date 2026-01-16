from TRITON_SWMM_toolkit.examples import GetTS_TestCases as tst


def test_retrieve_test():
    nrflk_cpu_sensitivity = tst.retreive_norfolk_cpu_config_sensitivity_case(
        start_from_scratch=True
    )


def test_compile():
    nrflk_cpu_sensitivity = tst.retreive_norfolk_cpu_config_sensitivity_case(
        start_from_scratch=False
    )
    analysis = nrflk_cpu_sensitivity.system.analysis
    analysis.sensitivity.compile_TRITON_SWMM_for_sensitivity_analysis()
    assert analysis.compilation_successful == True


def test_prepare_scenarios():
    nrflk_cpu_sensitivity = tst.retreive_norfolk_cpu_config_sensitivity_case(
        start_from_scratch=False
    )
    analysis = nrflk_cpu_sensitivity.system.analysis
    analysis.sensitivity.prepare_scenarios_in_each_subanalysis(concurrent=True)
    assert analysis.log.all_scenarios_created.get() == True


def test_run_all_sims():
    nrflk_cpu_sensitivity = tst.retreive_norfolk_cpu_config_sensitivity_case(
        start_from_scratch=False
    )
    analysis = nrflk_cpu_sensitivity.system.analysis
    analysis.sensitivity.run_all_sims(pickup_where_leftoff=False, concurrent=True)
    assert analysis.log.all_sims_run.get() == True


def test_consolidate_outputs():
    nrflk_cpu_sensitivity = tst.retreive_norfolk_cpu_config_sensitivity_case(
        start_from_scratch=False
    )
    analysis = nrflk_cpu_sensitivity.system.analysis
    analysis.sensitivity.process_simulation_timeseries_concurrently(
        overwrite_if_exist=True
    )
    analysis.sensitivity.consolidate_TRITON_outputs_for_analysis()
    assert analysis.log.TRITON_analysis_summary_created.get() == True
    analysis.sensitivity.consolidate_SWMM_outputs_for_analysis()
    assert analysis.log.SWMM_node_analysis_summary_created.get() == True
    assert analysis.log.SWMM_link_analysis_summary_created.get() == True
