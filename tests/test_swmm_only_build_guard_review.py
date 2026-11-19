"""Reviewer's tests for the standalone-SWMM native-build gate of scenario preparation.

Problem under test. ``TRITONSWMM_scenario._verify_native_build_or_skip_in_container`` decides which
compiled build a scenario needs before it prepares. A system that runs only standalone SWMM compiles
just the SWMM build, so the gate must name that build and no TRITON build, while every system that
carries a TRITON toggle keeps the gate it always had. These tests stress the two places a one-site
repair of that gate can go wrong and that the repair's own tests leave unstated: the model-toggle
combinations other than the three single-model ones, and the log file the gate reads against the log
the setup phase writes. Every test runs on a stub system; nothing compiles, runs a solver or calls
SLURM.

Chunk test (change 1118-swmmonly, reviewer): guards the native-build gate of ``hhemt.scenario``. It
proves one change once; at the landing it is promoted into the scenario module's tests or deleted.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from types import SimpleNamespace

import pytest

from hhemt.exceptions import CompilationError, ConfigurationError
from hhemt.scenario import TRITONSWMM_scenario
from hhemt.system import TRITONSWMM_system

_ABSENT_LOG = Path("/nonexistent/compilation.log")

#: Every model-toggle combination that carries a TRITON toggle, as (triton_only, coupled, swmm).
#: ``validation.py`` refuses only the all-False combination, so each of these is a system a user can run.
_WITH_A_TRITON_TOGGLE = [combo for combo in itertools.product((False, True), repeat=3) if (combo[0] or combo[1])]


def _toggles_id(combo) -> str:
    triton, coupled, swmm = combo
    return f"triton={int(triton)}-coupled={int(coupled)}-swmm={int(swmm)}"


class _SwmmBuildSystem:
    """A system stub whose SWMM-build verdict is the REAL ``compilation_swmm_successful`` property.

    The property reads ``SWMM_build_dir / "compilation.log"`` for the markers ``compile_SWMM`` leaves,
    which is the one file the setup phase writes. Borrowing the real property keeps the gate and the
    setup phase on one definition of "the SWMM build exists".
    """

    compilation_swmm_successful = TRITONSWMM_system.compilation_swmm_successful

    def __init__(self, swmm_build_dir: Path, toggles) -> None:
        triton, coupled, swmm = toggles
        self.cfg_system = SimpleNamespace(
            toggle_triton_model=triton, toggle_tritonswmm_model=coupled, toggle_swmm_model=swmm
        )
        self.system_config_yaml = Path("system.yaml")
        self.log = SimpleNamespace(compilation_swmm_successful=None)
        self.sys_paths = SimpleNamespace(
            SWMM_build_dir=swmm_build_dir,
            compilation_logfile_cpu=_ABSENT_LOG,
            compilation_logfile_gpu=_ABSENT_LOG,
        )
        # No TRITON build exists on this system, which is what a standalone-SWMM system looks like.
        self.compilation_triton_only_cpu_successful = False
        self.compilation_triton_only_gpu_successful = False
        self.compilation_cpu_successful = False
        self.compilation_gpu_successful = False

    def _sync_compilation_log_field(self, *_args, **_kwargs) -> None:
        return None


def _scenario_on(system, backend: str, execution_environment: str = "native") -> TRITONSWMM_scenario:
    scen = TRITONSWMM_scenario.__new__(TRITONSWMM_scenario)
    scen.backend = backend
    scen._system = system
    scen._analysis = SimpleNamespace(cfg_analysis=SimpleNamespace(execution_environment=execution_environment))
    return scen


def _flag_stub_scenario(toggles, backend: str, *, swmm_ok: bool, builds_ok: bool, execution_environment="native"):
    """A scenario whose build verdicts are plain facts: the SWMM build, and every TRITON build together."""
    triton, coupled, swmm = toggles
    system = SimpleNamespace(
        cfg_system=SimpleNamespace(toggle_triton_model=triton, toggle_tritonswmm_model=coupled, toggle_swmm_model=swmm),
        system_config_yaml=Path("system.yaml"),
        compilation_swmm_successful=swmm_ok,
        compilation_triton_only_cpu_successful=builds_ok,
        compilation_triton_only_gpu_successful=builds_ok,
        compilation_cpu_successful=builds_ok,
        compilation_gpu_successful=builds_ok,
        sys_paths=SimpleNamespace(
            SWMM_build_dir=Path("/nonexistent/swmm_build"),
            compilation_logfile_cpu=_ABSENT_LOG,
            compilation_logfile_gpu=_ABSENT_LOG,
        ),
    )
    return _scenario_on(system, backend, execution_environment)


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
@pytest.mark.parametrize("combo", _WITH_A_TRITON_TOGGLE, ids=_toggles_id)
@pytest.mark.parametrize("swmm_build_ok", [False, True], ids=["swmm-build-absent", "swmm-build-present"])
def test_a_system_with_a_triton_toggle_is_gated_on_its_triton_build_whatever_the_swmm_build_says(
    combo, swmm_build_ok, backend
):
    """Property: any system carrying a TRITON toggle is refused when its TRITON build is absent and passes when present.

    Class: every (triton_only, coupled, swmm) toggle combination with at least one TRITON toggle, on
    both backends, with the SWMM build present and absent. This is the part of the guard the repair
    must leave byte-for-byte as it was, held here over the combinations the repair's own tests do not
    reach (TRITON-only with SWMM, coupled with SWMM, all three models together).
    Reachability: ``validation.py`` refuses only the all-False combination, so each of these systems
    is accepted at run entry. A system with all three toggles is the ``synth_all_models`` case.
    Why required: a SWMM-only branch whose condition reads the SWMM toggle without ruling out both
    TRITON toggles would let such a system start on a SWMM build alone and fail inside the solver
    launch instead of at preparation.
    What kills it: dropping either ``not ...toggle_triton_model`` or ``not ...toggle_tritonswmm_model``
    from the SWMM-only condition, or routing on ``toggle_swmm_model`` alone.
    Second correct implementation: one helper that maps the three toggles to the required build and is
    called by the guard; the result for every combination is the same.
    The unit is one toggle combination per backend and SWMM-build state; the refusal is judged by the
    exception type alone, never by its wording.
    """
    scen_without = _flag_stub_scenario(combo, backend, swmm_ok=swmm_build_ok, builds_ok=False)
    with pytest.raises((CompilationError, ConfigurationError)):
        scen_without._verify_native_build_or_skip_in_container()

    scen_with = _flag_stub_scenario(combo, backend, swmm_ok=swmm_build_ok, builds_ok=True)
    scen_with._verify_native_build_or_skip_in_container()


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
@pytest.mark.parametrize("log_state", ["no-log", "failed-log"])
def test_a_swmm_only_system_is_not_satisfied_by_triton_builds_when_its_swmm_build_is_absent(
    backend, log_state, tmp_path
):
    """Property: with the SWMM build absent or failed, a SWMM-only system is refused whatever TRITON builds exist.

    Class: NEW CAPABILITY, the plausible wrong implementation is a guard that accepts "some build is
    compiled" (the TRITON-only or coupled build of a neighbouring system sharing a software tree).
    Differently positioned from the repair's own refusal test: there every TRITON build is absent too,
    so a guard that passes on any build still refuses it; here the TRITON builds are present and only
    the SWMM build is missing. The pre-change guard PASSES this input, so the test is red before the repair
    as well as under the wrong implementation.
    Reachability: a standalone-SWMM system whose ``TRITONSWMM_software_directory`` already holds the
    builds of another system, which is the shared-software-tree layout of the synthetic tier
    (``tests/conftest.py::tritonswmm_cpu_compiled`` builds all three into one tree).
    Why required: the SWMM executable is the only binary a standalone-SWMM scenario runs; a guard that
    does not name it defers a missing ``runswmm`` into command construction at run time.
    What kills it: replacing the SWMM-only refusal with ``compilation_swmm_successful or <any TRITON build>``.
    Second correct implementation: raising ``ConfigurationError`` for both log states, since the verdict
    here is judged by exception type alone.
    """
    swmm_build_dir = tmp_path / "swmm_build"
    swmm_build_dir.mkdir()
    if log_state == "failed-log":
        (swmm_build_dir / "compilation.log").write_text("CMake Error: no compiler\n")
    system = _SwmmBuildSystem(swmm_build_dir, (False, False, True))
    system.compilation_triton_only_cpu_successful = True
    system.compilation_triton_only_gpu_successful = True
    system.compilation_cpu_successful = True
    system.compilation_gpu_successful = True

    scen = _scenario_on(system, backend)
    with pytest.raises((CompilationError, ConfigurationError)):
        scen._verify_native_build_or_skip_in_container()


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_the_gate_reads_the_log_file_the_setup_phase_writes(backend, tmp_path):
    """Property: the gate's verdict and the path it names are those of ``SWMM_build_dir / "compilation.log"``.

    Class: the three states of that one file a build directory can be in, judged against the REAL
    ``compilation_swmm_successful`` property. A success log (the markers ``compile_SWMM`` leaves on
    Rivanna: ``Built target swmm5`` and ``Built target runswmm``) passes with no TRITON build present;
    a failure log is a ``CompilationError`` carrying exactly that file; an absent log under an existing
    build directory is a ``ConfigurationError`` naming exactly that path.
    Reachability: the failed Rivanna bundle's own SWMM build wrote this log into its ``SWMM_software_directory``
    with the two markers above; the file is the one place the gate and ``compile_SWMM`` meet.
    Why required: if the gate looked at any other path, a SWMM-only system whose setup succeeded would be
    refused, which is the defect this change repairs, displaced one directory over.
    What kills it: reading ``SWMM_build_dir / "swmm_build" / "compilation.log"`` or the coupled build's log.
    Second correct implementation: deriving the log path from the system's own property instead of joining
    ``compilation.log`` onto the build directory in the gate; the observable path is the same.
    """
    build_dir = tmp_path / "swmm_cd"
    build_dir.mkdir()
    log = build_dir / "compilation.log"

    system = _SwmmBuildSystem(build_dir, (False, False, True))
    scen = _scenario_on(system, backend)

    # State 1: no log yet under an existing build directory -> never built.
    with pytest.raises(ConfigurationError) as never_built:
        scen._verify_native_build_or_skip_in_container()
    assert str(log) in str(never_built.value)

    # State 2: a failed build leaves a log without a success marker.
    log.write_text("CMake Error at CMakeLists.txt:1\n")
    with pytest.raises(CompilationError) as failed:
        scen._verify_native_build_or_skip_in_container()
    assert failed.value.logfile == log

    # State 3: the markers a successful build leaves.
    log.write_text("[ 96%] Built target swmm5\n[100%] Built target runswmm\n")
    scen._verify_native_build_or_skip_in_container()


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_the_link_gate_that_follows_the_guard_also_passes_a_swmm_only_system(backend):
    """Property: after the guard passes, the second build gate of preparation takes the same SWMM-only system.

    Class: the two build-state reads ``prepare_scenario`` makes in sequence on one system:
    ``_verify_native_build_or_skip_in_container`` and then ``_link_native_builds_into_sim``. A SWMM-only
    system with the SWMM build and no TRITON build passes both; the same system without the SWMM build
    is refused by the second even if the first were bypassed.
    Reachability: ``prepare_scenario`` calls the two in this order for every scenario of every system.
    Why required: repairing only the first gate would move the refusal from the guard to the link step,
    where it would surface as the same unprepared scenario with a different message.
    What kills it: adding a TRITON-build requirement to the SWMM branch of ``_link_native_builds_into_sim``.
    Second correct implementation: merging the two gates into one function that both call.
    """
    toggles = (False, False, True)
    ready = _flag_stub_scenario(toggles, backend, swmm_ok=True, builds_ok=False)
    ready._verify_native_build_or_skip_in_container()
    ready._link_native_builds_into_sim()

    unbuilt = _flag_stub_scenario(toggles, backend, swmm_ok=False, builds_ok=False)
    with pytest.raises(RuntimeError):
        unbuilt._link_native_builds_into_sim()


@pytest.mark.parametrize("combo", [(False, False, True), *_WITH_A_TRITON_TOGGLE], ids=_toggles_id)
@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_container_mode_is_untouched_for_every_toggle_combination_with_nothing_built(combo, backend):
    """Property: in container mode the gate raises for no toggle combination, with no build of any kind present.

    Class: every model-toggle combination including the SWMM-only one, both backends. Held test: green
    before and after the repair; it fixes the ORDER of the two checks. Reachability: a container run
    of a SWMM-only system is a supported configuration (the SIF carries ``runswmm``, see the container
    branch of ``prepare_simulation_command``'s SWMM arm).
    Why required: the SWMM-only dispatch must sit after the container early return, or a container run
    would demand a native SWMM build that setup deliberately never compiled.
    What kills it: moving the SWMM-only dispatch above the ``execution_environment == "container"`` return.
    Second correct implementation: evaluating the container condition inside ``_verify_swmm_only_build``.
    """
    scen = _flag_stub_scenario(combo, backend, swmm_ok=False, builds_ok=False, execution_environment="container")
    scen._verify_native_build_or_skip_in_container()


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_a_failed_swmm_only_gate_names_the_cpu_backend_and_an_unset_directory_says_so(backend, tmp_path):
    """Property: a refused SWMM-only system never names a GPU build and never prints a path ``None``.

    Class: both backends, two refusals. A failing log raises ``CompilationError`` whose backend is
    ``cpu`` whatever the analysis backend is (SWMM has no GPU build, and ``compile_SWMM`` names ``cpu``);
    an unset ``SWMM_software_directory`` raises ``ConfigurationError`` that says the directory is not
    configured and prints no ``None`` path.
    Reachability: a SWMM-only analysis on a GPU partition has ``backend == "gpu"``; ``system.py`` sets
    ``SWMM_build_dir`` to None when the directory is unset.
    Why required: the message is what a user reads to decide what to rebuild.
    What kills it: passing ``self.backend`` to the error, or joining the log name onto a None directory.
    Second correct implementation: any wording that names the unset directory without the word ``None``.
    """
    build_dir = tmp_path / "swmm_cd"
    build_dir.mkdir()
    (build_dir / "compilation.log").write_text("CMake Error at CMakeLists.txt:1\n")
    with pytest.raises(CompilationError) as failed:
        _scenario_on(
            _SwmmBuildSystem(build_dir, (False, False, True)), backend
        )._verify_native_build_or_skip_in_container()
    assert failed.value.backend == "cpu"

    with pytest.raises(ConfigurationError) as unset:
        _scenario_on(_SwmmBuildSystem(None, (False, False, True)), backend)._verify_native_build_or_skip_in_container()
    assert "not configured" in str(unset.value)
    assert "None" not in str(unset.value)
