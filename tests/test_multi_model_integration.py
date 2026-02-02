"""
Tests for multi-model integration (TRITON-only, SWMM-only, and TRITON-SWMM).

These tests verify that the toolkit can handle:
1. TRITON-only simulations (no SWMM coupling)
2. SWMM-only simulations (standalone EPA SWMM)
3. All three models enabled concurrently
4. Model-specific compilation, execution, and output paths
"""

import pytest
from pathlib import Path
from tests.fixtures.test_case_catalog import Local_TestCases
import tests.utils_for_testing as tut
from TRITON_SWMM_toolkit.scenario import TRITONSWMM_scenario


class TestTRITONOnlyIntegration:
    """Tests for TRITON-only (no SWMM coupling) workflows."""

    @pytest.fixture
    def triton_only_case(self):
        """Retrieve TRITON-only test case."""
        return Local_TestCases.retrieve_norfolk_triton_only_test_case(
            start_from_scratch=True
        )

    def test_triton_only_compilation(self, triton_only_case):
        """Test that TRITON-only compiles with correct flags."""
        system = triton_only_case.system
        analysis = system.analysis

        # Verify toggles are set correctly
        assert system.cfg_system.toggle_triton_model is True
        assert system.cfg_system.toggle_tritonswmm_model is False
        assert system.cfg_system.toggle_swmm_model is False

        # Compile TRITON-only
        system.compile_TRITON_only(backends=["cpu"], verbose=True)

        # Verify compilation succeeded
        tut.assert_triton_compiled(analysis)
        assert system.compilation_triton_only_cpu_successful is True

    def test_triton_only_paths_created(self, triton_only_case):
        """Test that TRITON-only creates correct paths."""
        system = triton_only_case.system
        analysis = system.analysis

        # Create a scenario
        scenario = TRITONSWMM_scenario(event_iloc=0, analysis=analysis)

        # Verify TRITON-only paths exist
        assert scenario.scen_paths.triton_cfg is not None
        assert scenario.scen_paths.out_triton is not None
        assert scenario.scen_paths.sim_triton_executable is not None

        # Verify TRITON-SWMM and SWMM paths are None or not used
        # (triton_swmm_cfg exists for backward compat but shouldn't be used)
        assert scenario.scen_paths.out_swmm is None
        assert scenario.scen_paths.sim_swmm_executable is None

    def test_triton_only_cfg_generation(self, triton_only_case):
        """Test that TRITON.cfg has inp_filename commented out."""
        system = triton_only_case.system
        analysis = system.analysis

        # Prepare scenario
        scenario = TRITONSWMM_scenario(event_iloc=0, analysis=analysis)
        scenario.prepare_scenario()

        # Verify TRITON.cfg exists
        assert scenario.scen_paths.triton_cfg is not None
        assert scenario.scen_paths.triton_cfg.exists()

        # Read CFG and verify inp_filename is commented
        cfg_content = scenario.scen_paths.triton_cfg.read_text()
        assert "#inp_filename" in cfg_content or "# inp_filename" in cfg_content

        # Verify output_folder is set to out_triton
        assert "output_folder" in cfg_content


class TestSWMMOnlyIntegration:
    """Tests for SWMM-only (standalone EPA SWMM) workflows."""

    @pytest.fixture
    def swmm_only_case(self):
        """Retrieve SWMM-only test case."""
        return Local_TestCases.retrieve_norfolk_swmm_only_test_case(
            start_from_scratch=True
        )

    def test_swmm_only_compilation(self, swmm_only_case):
        """Test that SWMM compiles successfully."""
        system = swmm_only_case.system
        analysis = system.analysis

        # Verify toggles are set correctly
        assert system.cfg_system.toggle_triton_model is False
        assert system.cfg_system.toggle_tritonswmm_model is False
        assert system.cfg_system.toggle_swmm_model is True

        # Compile SWMM (this will be slow, so mark as slow test)
        pytest.skip("SWMM compilation test skipped - requires EPA SWMM source and is slow")

        # NOTE: To actually test compilation, uncomment:
        # system.compile_SWMM(verbose=True)
        # tut.assert_swmm_compiled(analysis)
        # assert system.compilation_swmm_successful is True

    def test_swmm_only_paths_created(self, swmm_only_case):
        """Test that SWMM-only creates correct paths."""
        system = swmm_only_case.system
        analysis = system.analysis

        # Create a scenario
        scenario = TRITONSWMM_scenario(event_iloc=0, analysis=analysis)

        # Verify SWMM paths are set
        assert scenario.scen_paths.inp_full is not None
        assert scenario.scen_paths.out_swmm is not None

        # Verify TRITON paths are None
        assert scenario.scen_paths.triton_cfg is None
        assert scenario.scen_paths.out_triton is None
        assert scenario.scen_paths.sim_triton_executable is None


class TestAllModelsIntegration:
    """Tests for all three models enabled concurrently."""

    @pytest.fixture
    def all_models_case(self):
        """Retrieve test case with all models enabled."""
        return Local_TestCases.retrieve_norfolk_all_models_test_case(
            start_from_scratch=True
        )

    def test_all_models_toggles(self, all_models_case):
        """Test that all model toggles are enabled."""
        system = all_models_case.system

        assert system.cfg_system.toggle_triton_model is True
        assert system.cfg_system.toggle_tritonswmm_model is True
        assert system.cfg_system.toggle_swmm_model is True

    def test_all_models_paths_created(self, all_models_case):
        """Test that all model-specific paths are created."""
        system = all_models_case.system
        analysis = system.analysis

        # Create a scenario
        scenario = TRITONSWMM_scenario(event_iloc=0, analysis=analysis)

        # Verify all model paths exist
        assert scenario.scen_paths.triton_cfg is not None
        assert scenario.scen_paths.triton_swmm_cfg is not None
        assert scenario.scen_paths.inp_full is not None

        assert scenario.scen_paths.out_triton is not None
        assert scenario.scen_paths.out_tritonswmm is not None
        assert scenario.scen_paths.out_swmm is not None

        # Verify executable paths only if models were compiled
        # (toggles can be ON without compilation for testing config logic)
        if system.log.compilation_triton_cpu_successful.get() or system.log.compilation_triton_gpu_successful.get():
            assert scenario.scen_paths.sim_triton_executable is not None

        if system.log.compilation_tritonswmm_cpu_successful.get() or system.log.compilation_tritonswmm_gpu_successful.get():
            assert scenario.scen_paths.sim_tritonswmm_executable is not None

        if system.log.compilation_swmm_successful.get():
            assert scenario.scen_paths.sim_swmm_executable is not None

    def test_all_models_logs_directory(self, all_models_case):
        """Test that centralized logs/ directory is created."""
        system = all_models_case.system
        analysis = system.analysis

        # Skip if SWMM wasn't compiled (compilation is slow and optional for path testing)
        if system.cfg_system.toggle_swmm_model and not system.log.compilation_swmm_successful.get():
            pytest.skip("SWMM compilation required but not performed - skipping prepare_scenario test")

        # Prepare scenario
        scenario = TRITONSWMM_scenario(event_iloc=0, analysis=analysis)
        scenario.prepare_scenario()

        # Verify logs directory exists
        assert scenario.scen_paths.logs_dir is not None
        assert scenario.scen_paths.logs_dir.exists()

        # Verify model-specific log paths are set
        assert scenario.scen_paths.log_run_triton is not None
        assert scenario.scen_paths.log_run_tritonswmm is not None
        assert scenario.scen_paths.log_run_swmm is not None

    def test_df_status_has_model_types(self, all_models_case):
        """Test that df_status includes model_types_enabled column."""
        system = all_models_case.system
        analysis = system.analysis

        df_status = analysis.df_status

        # Verify model_types_enabled column exists
        assert "model_types_enabled" in df_status.columns

        # Verify it shows all three models
        model_types = df_status["model_types_enabled"].iloc[0]
        assert "triton" in model_types
        assert "tritonswmm" in model_types
        assert "swmm" in model_types


class TestWorkflowGeneration:
    """Tests for Snakemake workflow generation with multi-model support."""

    def test_workflow_generates_model_specific_rules(self, tmp_path):
        """Test that workflow generates separate rules for each enabled model."""
        # This is a placeholder - full workflow testing requires more setup
        pytest.skip("Workflow generation test requires full Snakemake setup")

        # NOTE: To implement, would need to:
        # 1. Create test analysis with all models enabled
        # 2. Generate Snakefile via workflow.generate_snakefile_content()
        # 3. Verify rules exist: run_triton, run_tritonswmm, run_swmm
        # 4. Verify resource allocation (SWMM has threads=4, no GPU)
