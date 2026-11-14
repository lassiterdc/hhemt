"""
Tests for multi-model integration (TRITON-only, SWMM-only, and TRITON-SWMM).

These tests verify that the toolkit can handle:
1. TRITON-only simulations (no SWMM coupling)
2. SWMM-only simulations (standalone EPA SWMM)
3. All three models enabled concurrently
4. Model-specific compilation, execution, and output paths
"""

import pytest

import tests.utils_for_testing as tut
from hhemt.scenario import TRITONSWMM_scenario
from tests.fixtures.test_case_catalog import Local_TestCases

pytestmark = [
    pytest.mark.slow,
    pytest.mark.usefixtures("tritonswmm_cpu_compiled"),
    pytest.mark.skipif(
        tut.compile_toolchain_unavailable(),
        reason=(
            "TRITON-SWMM CPU compile toolchain (cmake + mpic++) not on PATH. "
            "The surviving tests call prepare_scenario(), whose CPU gate at "
            "scenario.py:1274-1279 requires a successful compile; run them "
            "under the hhemt conda env. NOTE: this reason names only the "
            "toolchain gate. The tritonswmm_cpu_compiled fixture this module "
            "also requests can skip for a SECOND reason, "
            "swmm_stack_version_mismatch() (utils_for_testing.py:87), and the "
            "first truthy skipif supplies the junit message — so a "
            "pyswmm-version skip of this module reports a toolchain reason."
        ),
    ),
]
# ESCALATION-SHADOWING, LATENT NOT LIVE: tritonswmm_cpu_compiled escalates to
# pytest.fail under HHEMT_REQUIRE_COMPILE_TIER=1 rather than skipping, and a
# collection-time module skipif pre-empts fixture setup entirely, bypassing that
# anti-masking gate. Measured 2026-09-06: compile-tests.yml's explicit allowlist
# does NOT include this module, so nothing is masked today. It becomes live the
# moment this file is added to that allowlist — read this comment before doing so.


class TestTRITONOnlyIntegration:
    """Tests for TRITON-only (no SWMM coupling) workflows."""

    @pytest.fixture
    def triton_only_case(self):
        """Retrieve TRITON-only test case."""
        return Local_TestCases.retrieve_synth_triton_only_test_case(start_from_scratch=True)

    @pytest.fixture
    def triton_cfg_case(self):
        """All-models synth case used by the TRITON-cfg test.

        NOT the triton-only case: prepare_scenario's CPU gate selects
        compilation_triton_only_cpu_successful when toggle_tritonswmm_model is
        False (scenario.py:1257, 1275-1279), and no fixture sets that marker for
        a synth family. _generate_TRITON_cfg gates only on toggle_triton_model
        (scenario.py:787-788), so the TRITON .cfg assertions hold identically on
        an all-models case.
        """
        return Local_TestCases.retrieve_synth_all_models_test_case(start_from_scratch=True)

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
        assert scenario.scen_paths.sim_swmm_executable is None

    @pytest.mark.usefixtures("triton_only_cpu_compiled")
    def test_triton_only_cfg_generation(self, triton_cfg_case):
        """Test that TRITON.cfg has inp_filename commented out."""
        system = triton_cfg_case.system
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
        return Local_TestCases.retrieve_synth_swmm_only_test_case(start_from_scratch=True)

    def test_swmm_only_toggles(self, swmm_only_case):
        """The SWMM-only case enables exactly the SWMM toggle."""
        system = swmm_only_case.system
        _ = system.analysis  # system.analysis is a property that RAISES when unset; the bind IS the check

        # Verify toggles are set correctly
        assert system.cfg_system.toggle_triton_model is False
        assert system.cfg_system.toggle_tritonswmm_model is False
        assert system.cfg_system.toggle_swmm_model is True

    def test_swmm_only_paths_created(self, swmm_only_case):
        """Test that SWMM-only creates correct paths."""
        system = swmm_only_case.system
        analysis = system.analysis

        # Create a scenario
        scenario = TRITONSWMM_scenario(event_iloc=0, analysis=analysis)

        # Verify SWMM paths are set
        assert scenario.scen_paths.swmm_full_inp is not None

        # Verify TRITON paths are None
        assert scenario.scen_paths.triton_cfg is None
        assert scenario.scen_paths.out_triton is None
        assert scenario.scen_paths.sim_triton_executable is None


class TestAllModelsIntegration:
    """Tests for all three models enabled concurrently."""

    @pytest.fixture
    def all_models_case(self):
        """Retrieve test case with all models enabled."""
        # start_from_scratch=True is LOAD-BEARING here, not a default.
        # True is what makes
        # test_case_builder.py:463 run process_system_level_inputs, which
        # produces system_dir/elevation_{res:.2f}m.dem (system.py:172) that
        # prepare_scenario opens at scenario.py:1521. It also keeps the
        # mkdtemp isolation (test_case_builder.py:373-375), so this build
        # never wipes a tree a concurrent session is reading.
        # The compile-marker problem that False was reaching for is closed
        # instead by Specs 8f and 8g, which move both SWMM guards off the
        # system_dir-derived LOG FIELD and onto the PROPERTY, which is
        # file-derived from the SHARED _software_root and therefore survives
        # a fresh system_directory. Do NOT switch this to False to reach the
        # slug-root system_log.json: False also skips preprocessing
        # (test_case_builder.py:463 gates both on the same flag), leaving these
        # tests to error on a missing DEM instead of skipping on a missing log.
        return Local_TestCases.retrieve_synth_all_models_test_case(start_from_scratch=True)

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
        assert scenario.scen_paths.swmm_full_inp is not None

        assert scenario.scen_paths.out_triton is not None
        assert scenario.scen_paths.out_tritonswmm is not None

        # THE THREE ASSERTIONS BELOW DO NOT CURRENTLY ASSERT ANYTHING, AND THEY
        # DO NOT ALL FAIL FOR THE SAME REASON. Measured 2026-09-06 and left in
        # place deliberately; read the split before concluding anything.
        #   * COMMON CAUSE, all three: the guards read LOG FIELDS, which are
        #     per-case (system_dir, per system.py:205) and empty for a freshly
        #     created system_directory. So all three branches are NOT TAKEN and
        #     the test passes having asserted nothing. A skip prints an 's'; an
        #     un-taken branch prints nothing.
        #   * TWO ARE STRUCTURALLY VACUOUS and no guard change recovers them.
        #     sim_tritonswmm_executable is unconditional (scenario.py:425) and
        #     sim_triton_executable is `... if cfg_sys.toggle_triton_model else
        #     None` (:426); both toggles are True on the all-models case
        #     (test_case_catalog.py:308-310), so `is not None` cannot fail on
        #     either arm. Making these guards property-based would convert a
        #     silent no-op into a loud one, nothing more.
        #   * ONE IS LIVE AND MERELY SUPPRESSED — this is the important half.
        #     sim_swmm_executable is `self._system.swmm_executable if
        #     cfg_sys.toggle_swmm_model else None` (:427), and swmm_executable
        #     (system.py:2180-2198) is NOT a constant: it returns None when
        #     SWMM_build_dir is None and otherwise probes runswmm/swmm5/swmm
        #     under bin/ and the build root, falling through to None when SWMM
        #     is not built. So this assertion genuinely CAN fail, and moving its
        #     guard to the property (as Specs 8f/8g do at the two skip sites
        #     below) WOULD recover real coverage. It is scoped out here as a
        #     deliberate bound, not because it is vacuous.
        #   * WHY BOUNDED: reviving it introduces a new failure mode into a batch
        #     whose purpose is a Norfolk-to-synth repoint, and the instrument
        #     that should gate the revival has not been run. That instrument is
        #     below; run it first.
        #   * ENUMERATING THE FOUR SITES. A bare
        #     `grep -c '.log.compilation_*.get()'` on this file does NOT do it:
        #     it returns 5, because this comment block and the guard comment
        #     below both name the pattern in prose. Use the comment-excluding
        #     form, which prose cannot satisfy:
        #       grep '\.log\.compilation_[a-z_]*\.get()' \
        #         tests/test_multi_model_integration.py | grep -vc '^\s*#'
        #     It returns 4 and they are the reads immediately below.
        #   * TERMINAL CONDITION for whoever takes these four sites. Do NOT use
        #     a grep count of `.log.compilation_*.get()` — it measures guard
        #     shape, and a cosmetic rewrite satisfies it while changing nothing
        #     a test can observe. Instead, PER ASSERTION: mutate the asserted
        #     fact and require the owning test to report `failed` — not
        #     `passed`, and specifically not `error`. Concretely: force
        #     scen_paths.sim_swmm_executable to None and require
        #     test_all_models_paths_created to FAIL. The failed-not-error
        #     distinction is load-bearing: an attribute-error mutation exits
        #     non-zero while never reaching the assertion. This check is
        #     self-partitioning — on a box with SWMM built it flags site 3 and
        #     not sites 1 and 2, so it sorts live from vacuous without anyone
        #     classifying by reading.
        # (toggles can be ON without compilation for testing config logic)
        if system.log.compilation_triton_cpu_successful.get() or system.log.compilation_triton_gpu_successful.get():
            assert scenario.scen_paths.sim_triton_executable is not None

        if (
            system.log.compilation_tritonswmm_cpu_successful.get()
            or system.log.compilation_tritonswmm_gpu_successful.get()
        ):
            assert scenario.scen_paths.sim_tritonswmm_executable is not None

        if system.log.compilation_swmm_successful.get():
            assert scenario.scen_paths.sim_swmm_executable is not None

    @pytest.mark.usefixtures("triton_only_cpu_compiled")
    def test_all_models_cfg_output_folders(self, all_models_case):
        """Test that CFG files have correct output_folder directives."""
        system = all_models_case.system
        analysis = system.analysis

        # Skip if SWMM wasn't compiled. Read the PROPERTY, never
        # system.log.compilation_swmm_successful.get(). The two operands live on
        # DIFFERENT TIERS and that is the whole point: the log field is read from
        # system_dir/"system_log.json" (system.py:205), which is per-case and is
        # empty for any case whose system_directory was freshly created, so
        # .get() returns None (log.py:76-78) and this skips unconditionally. The
        # property is read from SWMM_build_dir/compilation.log (system.py:2148,
        # 2162), and SWMM_build_dir is SWMM_software_directory =
        # _software_root/"swmm" (test_case_builder.py:484), which is pinned to
        # the shared slug root and is where the compile actually lands. This
        # comment deliberately states the tier property rather than any
        # particular fixture's start_from_scratch value: the guard must be
        # correct for every caller, not for the one above it today.
        if system.cfg_system.toggle_swmm_model and not system.compilation_swmm_successful:
            pytest.skip("SWMM compilation required but not performed - skipping prepare_scenario test")

        # Prepare scenario
        scenario = TRITONSWMM_scenario(event_iloc=0, analysis=analysis)
        scenario.prepare_scenario()

        # Verify TRITON.cfg has output_folder="out_triton"
        if scenario.scen_paths.triton_cfg and scenario.scen_paths.triton_cfg.exists():
            triton_cfg_content = scenario.scen_paths.triton_cfg.read_text()
            assert "output_folder" in triton_cfg_content
            assert 'output_folder="out_triton"' in triton_cfg_content

        # Verify TRITONSWMM.cfg has output_folder="out_tritonswmm"
        if scenario.scen_paths.triton_swmm_cfg and scenario.scen_paths.triton_swmm_cfg.exists():
            tritonswmm_cfg_content = scenario.scen_paths.triton_swmm_cfg.read_text()
            assert "output_folder" in tritonswmm_cfg_content
            assert 'output_folder="out_tritonswmm"' in tritonswmm_cfg_content

    @pytest.mark.usefixtures("triton_only_cpu_compiled")
    def test_all_models_logs_directory(self, all_models_case):
        """Test that centralized logs/ directory is created."""
        system = all_models_case.system
        analysis = system.analysis

        # Skip if SWMM wasn't compiled. Read the PROPERTY, not the log field:
        # the log field is per-case, read from system_dir/"system_log.json"
        # (system.py:205); the property is read from the shared
        # _software_root/"swmm"/compilation.log (system.py:2162,
        # test_case_builder.py:484), which is where the compile lands. Stated in
        # full rather than cross-referencing the identical guard in
        # test_all_models_cfg_output_folders: a comment that defers to another
        # comment is a second copy of a fact with nothing keeping the two in
        # step, and it keeps pointing at that explanation after it is edited.
        if system.cfg_system.toggle_swmm_model and not system.compilation_swmm_successful:
            pytest.skip("SWMM compilation required but not performed - skipping prepare_scenario test")

        # Prepare scenario
        scenario = TRITONSWMM_scenario(event_iloc=0, analysis=analysis)
        scenario.prepare_scenario()

        # Verify logs directory exists
        assert scenario.scen_paths.logs_dir is not None
        assert scenario.scen_paths.logs_dir.exists()

        # Verify model-specific log paths are set
        # log_run_triton / log_run_tritonswmm / log_run_swmm retired: they declared paths
        # nothing writes (see paths.py::ScenarioPaths). Asserting a dead field is not None
        # is what made the convention look real. The live per-sim log path is asserted by
        # tests/test_coupled_resume_validity.py::test_model_logfile_method_delegates_to_free_function.
