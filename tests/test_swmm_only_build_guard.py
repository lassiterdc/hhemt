"""A standalone-SWMM system is gated on the SWMM build, never on a TRITON build.

Problem under test. Scenario preparation refuses to run when the build its system needs is not
compiled. ``_verify_native_build_or_skip_in_container`` chose which build to require from
``toggle_tritonswmm_model`` alone: with that toggle False it named the "triton_only" model and
required a TRITON-only CPU or GPU build. A system that runs standalone SWMM only
(``toggle_swmm_model`` True, both TRITON toggles False) compiles just the SWMM build, so every one
of its scenarios stopped with ``No triton_only CPU build is present`` (measured on Rivanna,
2026-11-19, bundle ``phase_b_cpu``). These tests pin the repaired choice on a stub system, so
nothing compiles, runs a solver or calls SLURM.

Chunk test (change 1118-swmmonly): guards the native-build gate of ``hhemt.scenario``. It proves
one change once; at the landing it is promoted into the scenario module's tests or deleted.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from hhemt.exceptions import CompilationError, ConfigurationError

_SRC = Path(__file__).resolve().parents[1] / "src" / "hhemt"

#: The model combinations a system can request, as (triton, tritonswmm, swmm) toggles.
_SWMM_ONLY = (False, False, True)
_TRITON_ONLY = (True, False, False)
_COUPLED = (False, True, False)


def _scenario(toggles, backend, *, swmm_ok=False, swmm_log="absent", triton_only_ok=False, coupled_ok=False):
    """A scenario whose system answers the build questions from the given facts.

    ``swmm_log`` is ``"absent"`` (no SWMM compilation log on disk, the build was never attempted)
    or a path to a log the test wrote (a build was attempted and did not succeed).
    """
    from hhemt.scenario import TRITONSWMM_scenario

    triton, tritonswmm, swmm = toggles
    swmm_build_dir = None if swmm_log == "absent" else Path(swmm_log).parent
    absent_log = Path("/nonexistent/compilation.log")
    system = SimpleNamespace(
        cfg_system=SimpleNamespace(
            toggle_triton_model=triton,
            toggle_tritonswmm_model=tritonswmm,
            toggle_swmm_model=swmm,
        ),
        system_config_yaml=Path("system.yaml"),
        compilation_swmm_successful=swmm_ok,
        compilation_triton_only_cpu_successful=triton_only_ok,
        compilation_triton_only_gpu_successful=triton_only_ok,
        compilation_cpu_successful=coupled_ok,
        compilation_gpu_successful=coupled_ok,
        sys_paths=SimpleNamespace(
            SWMM_build_dir=swmm_build_dir if swmm_build_dir is not None else Path("/nonexistent/swmm_build"),
            compilation_logfile_cpu=absent_log,
            compilation_logfile_gpu=absent_log,
        ),
    )
    if swmm_log == "absent":
        system.sys_paths.SWMM_build_dir = None
    scen = TRITONSWMM_scenario.__new__(TRITONSWMM_scenario)
    scen.backend = backend
    scen._system = system
    scen._analysis = SimpleNamespace(cfg_analysis=SimpleNamespace(execution_environment="native"))
    return scen


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_swmm_only_system_with_the_swmm_build_and_no_triton_build_passes(backend):
    """Property: a standalone-SWMM system with a compiled SWMM build is not refused on either backend.

    Class: REGRESSION. Red on the pre-change tree: it raised ``ConfigurationError`` (cpu) or
    ``CompilationError`` (gpu) naming ``triton_only``. Reachability: the input is the measured
    ``phase_b_cpu`` system (SWMM toggle only, SWMM build present, no TRITON build).
    Why required: the guard runs in ``prepare_scenario`` before any simulation. What kills it:
    selecting the TRITON-only build from ``toggle_tritonswmm_model`` alone. A second correct
    implementation: one that reads all three toggles through a single label helper.
    """
    scen = _scenario(_SWMM_ONLY, backend, swmm_ok=True)
    scen._verify_native_build_or_skip_in_container()


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_swmm_only_system_without_the_swmm_build_is_refused_naming_swmm_not_triton(backend, tmp_path):
    """Property: with no SWMM build the refusal names SWMM, and never a TRITON build.

    Class: NEW CAPABILITY; the plausible wrong implementation is the old guard, which raised for a
    missing TRITON-only build, so a green here with the SWMM build ABSENT would not separate it
    from the old one. The assertion therefore pins the model named in the error. Both arms of the
    refusal are covered: no log at all (ConfigurationError) and a log that records no success
    (CompilationError).
    """
    scen = _scenario(_SWMM_ONLY, backend, swmm_ok=False, swmm_log="absent")
    with pytest.raises(ConfigurationError) as never_built:
        scen._verify_native_build_or_skip_in_container()
    assert "SWMM software directory is not configured" in str(never_built.value)
    assert "SWMM_software_directory" in str(never_built.value)
    assert "None" not in str(never_built.value)
    assert "triton_only" not in str(never_built.value).lower()

    failed_log = tmp_path / "compilation.log"
    failed_log.write_text("CMake Error: no compiler\n")
    scen = _scenario(_SWMM_ONLY, backend, swmm_ok=False, swmm_log=str(failed_log))
    with pytest.raises(CompilationError) as failed:
        scen._verify_native_build_or_skip_in_container()
    assert failed.value.model_type == "swmm"
    assert failed.value.backend == "cpu"
    assert failed.value.logfile == failed_log


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_triton_only_system_still_demands_the_triton_only_build(backend):
    """Property: a TRITON-only system is gated on the TRITON-only build, and the SWMM build is irrelevant.

    Class: REGRESSION (green before and after; it holds the old behaviour). Kill: routing a
    TRITON-only system through the SWMM branch, which a compiled SWMM build would then satisfy.
    Second correct implementation: any gate that keys the TRITON-only build on ``toggle_triton_model``.
    """
    refused = _scenario(_TRITON_ONLY, backend, swmm_ok=True, triton_only_ok=False)
    with pytest.raises((CompilationError, ConfigurationError)) as err:
        refused._verify_native_build_or_skip_in_container()
    assert "triton_only" in str(err.value).lower()
    _scenario(_TRITON_ONLY, backend, swmm_ok=False, triton_only_ok=True)._verify_native_build_or_skip_in_container()


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_coupled_system_still_demands_the_coupled_build(backend):
    """Property: a coupled system is gated on the coupled build, whatever the other builds say.

    Class: REGRESSION (holds the old behaviour). Kill: a SWMM-only test written as "SWMM build
    present" without the model toggles, which would let a coupled system pass on a SWMM build.
    """
    refused = _scenario(_COUPLED, backend, swmm_ok=True, triton_only_ok=True, coupled_ok=False)
    with pytest.raises((CompilationError, ConfigurationError)) as err:
        refused._verify_native_build_or_skip_in_container()
    assert "tritonswmm" in str(err.value).lower()
    _scenario(_COUPLED, backend, coupled_ok=True)._verify_native_build_or_skip_in_container()


# --- The population: every function of src/hhemt that gates a TRITON build on the coupled toggle ----

#: The properties of the system that report whether a TRITON build exists (the SWMM build is not one).
_TRITON_BUILD_PROPERTIES = frozenset(
    {
        "compilation_successful",
        "compilation_cpu_successful",
        "compilation_gpu_successful",
        "compilation_triton_only_successful",
        "compilation_triton_only_cpu_successful",
        "compilation_triton_only_gpu_successful",
    }
)
_MODEL_TOGGLES = frozenset({"toggle_triton_model", "toggle_tritonswmm_model", "toggle_swmm_model"})


def _toggles_in_decisions(function: ast.AST) -> set[str]:
    """The model toggles a function reads inside a DECISION: an if/while/ternary test or a boolean operation.

    A toggle read that only feeds a log line, a return value or a list is a mention, not a decision,
    so it cannot tell a standalone-SWMM system from a TRITON-only one.
    """
    decided: set[str] = set()
    for node in ast.walk(function):
        if isinstance(node, (ast.If, ast.IfExp, ast.While)):
            root: ast.AST = node.test
        elif isinstance(node, ast.BoolOp):
            root = node
        else:
            continue
        decided |= {n.attr for n in ast.walk(root) if isinstance(n, ast.Attribute)} & _MODEL_TOGGLES
    return decided


def _toggle_and_build_readers(source: str) -> dict[str, set[str]]:
    """Map each function that reads a TRITON build property AND the coupled toggle to the toggles it decides on.

    The unit is a function definition (qualified by its enclosing class), matched by attribute name
    over the function's whole body, nested functions included.
    """
    found: dict[str, set[str]] = {}

    def visit(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                visit(child, f"{prefix}{child.name}.")
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names = {n.attr for n in ast.walk(child) if isinstance(n, ast.Attribute)}
                names |= {n.value for n in ast.walk(child) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
                if names & _TRITON_BUILD_PROPERTIES and "toggle_tritonswmm_model" in names:
                    found[f"{prefix}{child.name}"] = names & _MODEL_TOGGLES & _toggles_in_decisions(child)
                visit(child, f"{prefix}{child.name}.")

    visit(ast.parse(source), "")
    return found


def _census() -> dict[str, set[str]]:
    """The reader map over every module of src/hhemt except the system module that defines the properties.

    ``config/`` is excluded because it declares the toggles and reads no build status; the system
    module is excluded because it is where the build properties are DEFINED, not gated on.
    """
    population: dict[str, set[str]] = {}
    for path in sorted(_SRC.rglob("*.py")):
        if path.name == "system.py" or "config" in path.relative_to(_SRC).parts[:1]:
            continue
        for qualified, toggles in _toggle_and_build_readers(path.read_text()).items():
            population[f"{path.relative_to(_SRC).as_posix()}::{qualified}"] = toggles
    return population


#: The closed population as of this change; each member decides on the SWMM toggle (checked below).
_DISPOSITIONS = frozenset(
    {
        "analysis_validation.py::check_system_setup",
        "scenario.py::TRITONSWMM_scenario._verify_native_build_or_skip_in_container",
        "scenario.py::TRITONSWMM_scenario._link_native_builds_into_sim",
        "setup_workflow.py::main",
    }
)


def test_no_function_gates_a_triton_build_on_the_coupled_toggle_alone():
    """Property: for every function that reads both a TRITON build property and ``toggle_tritonswmm_model``,
    it also DECIDES on ``toggle_swmm_model`` (an if/ternary test or boolean
    operation), so it can tell a standalone-SWMM system from a TRITON-only one.

    Class: NEW CAPABILITY. D = every function of src/hhemt (system module and config excluded) that
    reads a TRITON build-status property and ``toggle_tritonswmm_model``; P = it reads
    ``toggle_swmm_model`` too. The unit of D is the function definition and the matcher is the
    attribute-name walk in ``_toggle_and_build_readers``. The plausible wrong implementation is a
    one-site repair of ``scenario.py``: it would still turn the SWMM-only guard green and leave
    another function choosing a TRITON build from the coupled toggle, which this test reds on.
    Second correct implementation: moving the three-toggle choice into one helper that every site
    calls (the helper reads all three toggles, the callers read none).
    """
    population = _census()
    assert "scenario.py::TRITONSWMM_scenario._verify_native_build_or_skip_in_container" in population, (
        "the census must reach the function the change repaired; an empty or short population "
        "would pass this test whatever the code does"
    )
    alone = sorted(site for site, toggles in population.items() if "toggle_swmm_model" not in toggles)
    assert set(population) == _DISPOSITIONS, (
        "a function now reads a TRITON build property and the coupled toggle, or one stopped: give it a row "
        f"in _DISPOSITIONS; population is {sorted(population)}"
    )
    assert alone == [], (
        f"functions that gate a TRITON build on the coupled toggle without reading the SWMM toggle: {alone}"
    )


def test_the_census_matcher_fires_on_the_defect_shape():
    """Property: the census matcher reports the old guard's shape, so a zero from it is a measurement.

    Class: NEW CAPABILITY (a must-fire control over the property under suspicion). The source below
    is the old guard reduced: it picks a build from the coupled toggle alone. The matcher must list
    it with the SWMM toggle absent; if it did not, the previous test's empty list would not mean
    "no such function exists".
    """
    defect = (
        "class S:\n"
        "    def guard(self):\n"
        "        is_c = self._system.cfg_system.toggle_tritonswmm_model\n"
        "        sys = self._system\n"
        "        return sys.compilation_cpu_successful if is_c else sys.compilation_triton_only_cpu_successful\n"
    )
    assert _toggle_and_build_readers(defect) == {"S.guard": set()}


def test_a_mention_of_the_swmm_toggle_is_not_a_decision_on_it():
    """Property: a function that picks a TRITON build from the coupled toggle and only logs the SWMM toggle is flagged.

    Class: NEW CAPABILITY, the differently-positioned violating input the reviewer measured: the old
    guard's shape plus an unrelated read of ``toggle_swmm_model``. The earlier mention predicate
    reported it clean; the decision predicate reports the SWMM toggle absent. The control is the same
    function with the SWMM toggle in an ``if`` test, which must report it present.
    """
    mention = (
        "class S:\n"
        "    def guard(self):\n"
        "        cfg = self._system.cfg_system\n"
        "        print(cfg.toggle_swmm_model)\n"
        "        sys = self._system\n"
        "        c, t = sys.compilation_cpu_successful, sys.compilation_triton_only_successful\n"
        "        ok = c if cfg.toggle_tritonswmm_model else t\n"
        "        return ok\n"
    )
    decision = mention.replace("print(cfg.toggle_swmm_model)", "if cfg.toggle_swmm_model:\n            return None")
    assert "toggle_swmm_model" not in _toggle_and_build_readers(mention)["S.guard"]
    assert "toggle_swmm_model" in _toggle_and_build_readers(decision)["S.guard"]
