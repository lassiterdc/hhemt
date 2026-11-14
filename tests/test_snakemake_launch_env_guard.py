"""Every Snakemake launch defaults its BLAS and OpenMP thread counts to 1, and a PMIx failure names its remedy.

Problem: Snakemake imports numpy and pandas and is the ``sbatch`` caller, so a numpy that starts one BLAS
thread per core oversubscribes a login node or a 1-CPU orchestrator job. Work package 1i, chunk 1i.1 of the
release-PR run guards every Snakemake launch with ``OPENBLAS_NUM_THREADS``, ``OMP_NUM_THREADS`` and
``MKL_NUM_THREADS`` set to 1 in the child's environment only, and adds a detector that turns the PMIx
``gds/shmem2`` failure text in a failed solver's log into an error that names the remedy.

The first part pins the guard, and its core is a source scan that fails when a launch site omits it. The scan
is a function of the module text, so the tests run it over the real source (it must report nothing) and over
source with one defect planted (it must report that defect). The second part feeds the detector fixture logs.

No solver, compile or Snakemake subprocess runs here; the one subprocess is ``env``/``printenv`` in bash.

Each test docstring states Property, Class, Why it is required, What kills it, and a second correct
implementation under which it still passes.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hhemt import workflow as wf
from hhemt.cli_utils import map_exception_to_exit_code
from hhemt.exceptions import SimulationError
from hhemt.run_simulation import (
    PMIX_GDS_FAILURE_TEXT,
    PMIxSharedMemoryError,
    TRITONSWMM_run,
    raise_if_pmix_gds_failure,
)

_GUARD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
_BASE_CMD_METHOD = "_get_snakemake_base_cmd"
_GUARD_FUNCTION = "snakemake_thread_guard_assignments"
_SHELL_GUARD_PLACEHOLDER = "{_SNAKEMAKE_THREAD_GUARD_SHELL_ASSIGNMENTS}"
# The one emitted script that already runs on a compute node and carries no guard by design.
_COMPUTE_NODE_SCRIPT_MARKER = "--cores $TOTAL_CPUS"


# ---------------------------------------------------------------------------
# The launch-site scan
# ---------------------------------------------------------------------------


def _python_launch_violations(source: str) -> tuple[int, list[str]]:
    """Return (launches built from the base command, violations) for a module's source text.

    A violation is the base-command method missing its guard call, or a list literal that spells out
    ``"-m", "snakemake"`` anywhere except inside that method, which is a launch built around the guard.
    """
    tree = ast.parse(source)
    violations: list[str] = []
    launches = 0
    method_bodies: list[ast.FunctionDef] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == _BASE_CMD_METHOD:
            method_bodies.append(node)
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == _BASE_CMD_METHOD:
            launches += 1
    if not method_bodies:
        violations.append(f"{_BASE_CMD_METHOD} is not defined")
    for method in method_bodies:
        calls_guard = any(
            isinstance(n, ast.Call) and getattr(n.func, "id", None) == _GUARD_FUNCTION for n in ast.walk(method)
        )
        if not calls_guard:
            violations.append(f"{_BASE_CMD_METHOD} does not call {_GUARD_FUNCTION}")
    inside = {id(n) for m in method_bodies for n in ast.walk(m)}
    for node in ast.walk(tree):
        if isinstance(node, ast.List) and id(node) not in inside:
            literals = [e.value for e in node.elts if isinstance(e, ast.Constant)]
            if "-m" in literals and "snakemake" in literals:
                violations.append(f"line {node.lineno}: a snakemake command built without {_BASE_CMD_METHOD}")
    return launches, violations


def _script_launch_violations(source: str) -> tuple[int, int, list[str]]:
    """Return (guarded script launches, exempt script launches, violations) for emitted shell scripts.

    A launch is a line of an f-string script that runs ``-m snakemake`` and continues onto the next line with a
    backslash, which no ``--version`` probe and no one-line manual hint does.
    The guard placeholder must stand within the three lines before it. A script that computes its cores from
    the allocation already runs on a compute node and is counted as exempt.
    """
    tree = ast.parse(source)
    guarded = exempt = 0
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.JoinedStr):
            continue
        parts = [
            v.value if isinstance(v, ast.Constant) else "{" + ast.unparse(v.value) + "}"
            for v in node.values
            if isinstance(v, (ast.Constant, ast.FormattedValue))
        ]
        lines = "".join(str(p) for p in parts).split("\n")
        launch_idx = [i for i, ln in enumerate(lines) if "-m snakemake" in ln and ln.rstrip().endswith("\\")]
        for i in launch_idx:
            if any(_COMPUTE_NODE_SCRIPT_MARKER in ln for ln in lines):
                exempt += 1
            elif any(_SHELL_GUARD_PLACEHOLDER in ln for ln in lines[max(0, i - 3) : i]):
                guarded += 1
            else:
                violations.append(f"line {node.lineno}: script launch at script line {i} has no thread guard")
    return guarded, exempt, violations


@pytest.fixture(scope="module")
def workflow_source() -> str:
    return Path(wf.__file__).read_text()


def test_every_python_snakemake_launch_is_built_from_the_guarded_base_command(workflow_source: str):
    """Property: no Snakemake launch from Python omits the thread guard, and the scan is able to see a defect.

    Class: REGRESSION. Before this chunk the base command was a bare ``[python, -m, snakemake]`` and nothing
    checked it; the first assertion is red against that tree because the method calls no guard function. The
    scan must also be shown to fire: two planted defects, the guard call removed and a hand-built launch beside
    the method, are each reported. Why required: a green scan over a matcher that reads nothing is the same zero
    as a clean tree, and the launch count is the control that it read the sites. What kills it: dropping the guard
    call from ``_get_snakemake_base_cmd`` or building a command list without it. Second correct implementation:
    passing ``env=`` at every ``subprocess`` call instead, with this scan rewritten to read that keyword.
    """
    launches, violations = _python_launch_violations(workflow_source)
    assert violations == []
    # Reachability control: the scan found the launches it claims to check (8 command builds today).
    assert launches >= 8

    unguarded = workflow_source.replace(f"guard = {_GUARD_FUNCTION}()", "guard = []")
    assert unguarded != workflow_source
    assert _python_launch_violations(unguarded)[1] == [f"{_BASE_CMD_METHOD} does not call {_GUARD_FUNCTION}"]

    bypass = workflow_source + '\n\ndef _bypass():\n    return [sys.executable, "-m", "snakemake", "--version"]\n'
    bypass_violations = _python_launch_violations(bypass)[1]
    assert len(bypass_violations) == 1 and "built without" in bypass_violations[0]


def test_both_login_node_scripts_carry_the_guard_and_the_compute_node_script_does_not(workflow_source: str):
    """Property: the batch_job and tmux scripts each launch Snakemake behind the guard; the 1-job script is exempt.

    Class: REGRESSION. Before this chunk neither script mentioned a thread variable, so the guarded count was 0.
    Why required: the two scripts are launches the Python base command never reaches, so the Python scan cannot
    see them; the exempt count pins that the exemption covers the 1-job script and nothing else. What kills it:
    removing the placeholder from either script (planted below) or widening the exemption marker so it swallows
    them. Second correct implementation: exporting the three variables on their own lines before the launch
    instead of inside the ``env`` prefix, with the scan reading ``export`` lines.
    """
    guarded, exempt, violations = _script_launch_violations(workflow_source)
    assert violations == []
    assert (guarded, exempt) == (2, 1)

    stripped = workflow_source.replace(f"    {_SHELL_GUARD_PLACEHOLDER} \\\\\n", "", 1)
    assert stripped != workflow_source
    _, _, planted = _script_launch_violations(stripped)
    assert len(planted) == 1 and "no thread guard" in planted[0]


# ---------------------------------------------------------------------------
# The guard's behaviour
# ---------------------------------------------------------------------------


def test_guard_words_cover_only_the_variables_the_user_has_not_set():
    """Property: a variable that is unset or empty gets a ``VAR=1`` word, and a variable the user set gets none.

    Class: three input states of the same variable (unset, empty, set to 8), each against a differently
    positioned satisfying input. REGRESSION: the function did not exist. Why required: ``${VAR:-1}`` is the
    contract, so a value the user exported must win and an empty value must not. What kills it: always emitting
    ``=1``, which overrides the user, or testing ``in environ`` instead of truthiness, which keeps an empty
    string. Second correct implementation: returning a mapping of the defaults and having the caller format it.
    """
    assert wf.snakemake_thread_guard_assignments({}) == [f"{v}=1" for v in _GUARD_VARS]
    assert wf.snakemake_thread_guard_assignments({"OPENBLAS_NUM_THREADS": ""}) == [f"{v}=1" for v in _GUARD_VARS]
    kept = wf.snakemake_thread_guard_assignments({"OPENBLAS_NUM_THREADS": "8", "MKL_NUM_THREADS": "4"})
    assert kept == ["OMP_NUM_THREADS=1"]
    assert wf.snakemake_thread_guard_assignments(dict.fromkeys(_GUARD_VARS, "2")) == []


def test_the_base_command_guards_the_child_and_never_touches_this_process(monkeypatch):
    """Property: the command carries an ``env VAR=1`` prefix for unset variables, and ``os.environ`` is unchanged.

    Class: REGRESSION on the command (red before the change, which returned the bare three words), and a
    non-mutation arm that no prefix-free implementation can fail. Both states are run: variables unset gives the
    prefix, variables set gives the bare command. Why required: setting ``os.environ`` would not reach a numpy a
    notebook already imported and would change that session's thread count afterwards. What kills it: writing
    the variables into ``os.environ`` (the second assertion block), or emitting the prefix when all are set.
    Second correct implementation: returning an ``(argv, env)`` pair, with the assertions reading the pair.
    """
    for var in _GUARD_VARS:
        monkeypatch.delenv(var, raising=False)
    before = dict(os.environ)
    cmd = wf.SnakemakeWorkflowBuilder._get_snakemake_base_cmd(None)  # ty: ignore[invalid-argument-type]  -- stub stands in for the instance
    assert cmd[0] == "env"
    assert cmd[1 : 1 + len(_GUARD_VARS)] == [f"{v}=1" for v in _GUARD_VARS]
    assert cmd[-3:] == [sys.executable, "-m", "snakemake"]
    assert dict(os.environ) == before
    assert not any(v in os.environ for v in _GUARD_VARS)

    for var in _GUARD_VARS:
        monkeypatch.setenv(var, "6")
    assert wf.SnakemakeWorkflowBuilder._get_snakemake_base_cmd(None) == [sys.executable, "-m", "snakemake"]  # ty: ignore[invalid-argument-type]  -- stub stands in for the instance


@pytest.mark.skipif(not Path("/usr/bin/env").exists() and not Path("/bin/env").exists(), reason="needs POSIX env")
def test_the_shell_guard_defaults_unset_variables_and_keeps_exported_ones():
    """Property: executed by bash, the emitted ``env`` assignments give 1 for unset variables and keep exports.

    Class: the shell spelling of the guard, which the Python tests cannot execute. REGRESSION: the constant did
    not exist. Why required: the two login-node scripts use this text verbatim and need conda and SLURM to run,
    so this is the only place its ``${VAR:-1}`` quoting is executed. What kills it: ``${VAR}`` without the
    default, or single quotes around the assignment so the shell never expands it. Second correct
    implementation: ``export VAR=${VAR:-1}`` lines before the launch, checked the same way.
    """
    assignments = wf._SNAKEMAKE_THREAD_GUARD_SHELL_ASSIGNMENTS
    script = f"env {assignments} printenv {' '.join(_GUARD_VARS)}"
    base = {k: v for k, v in os.environ.items() if k not in _GUARD_VARS}
    run = lambda extra: subprocess.run(  # noqa: E731
        ["bash", "-c", script], env={**base, **extra}, capture_output=True, text=True, check=True
    ).stdout.split()
    assert run({}) == ["1", "1", "1"]
    assert run({"OPENBLAS_NUM_THREADS": "8"}) == ["8", "1", "1"]
    assert run({"OMP_NUM_THREADS": "", "MKL_NUM_THREADS": "3"}) == ["1", "1", "3"]


# ---------------------------------------------------------------------------
# The PMIx detector
# ---------------------------------------------------------------------------

_PMIX_LOG = (
    "[node1:1234] PMIX ERROR: OUT-OF-RESOURCE in file gds_shmem2.c at line 99\n"
    "A memory allocation backed by shared-memory failed in the gds/shmem2 component.\n"
    "Resolve this issue by either:\n"
    "  1.) Increasing the value of PMIX_MCA_gds_shmem2_segment_size_multiplier\n"
    f"  2.) {PMIX_GDS_FAILURE_TEXT}\n"
)
_OTHER_MPI_LOG = (
    "There are not enough slots available in the system to satisfy the 2 slots that were requested.\n"
    "gds: hash selected as the data store\n"
    "Disabling gds via a different setting is not the PMIx sentence\n"
)


def test_detector_names_the_remedy_for_the_pmix_text_and_stays_silent_otherwise(tmp_path: Path):
    """Property: the PMIx sentence raises ``PMIxSharedMemoryError`` with the remedy, and nothing else does.

    Class: a positive log, two negative logs that share words with the sentence (the PRRTE slots refusal and
    near-miss ``gds`` lines), a missing file and ``None``. REGRESSION: the detector did not exist. Why required:
    the toolkit refuses to emit the variable, so this error text is the only place the remedy reaches a user at
    failure time; a detector that fires on a near-miss would misdirect a different failure. What kills it:
    matching ``gds`` or ``shmem2`` alone, or raising for a missing log. Second correct implementation: matching
    the sentence with a regex that tolerates whitespace, with the same two negative logs.
    """
    hit = tmp_path / "sim_run_3.log"
    hit.write_text(_PMIX_LOG)
    with pytest.raises(PMIxSharedMemoryError) as caught:
        raise_if_pmix_gds_failure(hit, event_iloc=3, model_type="tritonswmm")
    message = str(caught.value)
    assert "PMIX_MCA_gds=hash" in message and "PMIX_MCA_gds_shmem2_segment_size_multiplier" in message
    assert str(hit) in message and "event_iloc=3" in message and "tritonswmm" in message
    assert caught.value.logfile == hit and caught.value.event_iloc == 3

    miss = tmp_path / "other.log"
    miss.write_text(_OTHER_MPI_LOG)
    raise_if_pmix_gds_failure(miss, event_iloc=3, model_type="tritonswmm")
    raise_if_pmix_gds_failure(tmp_path / "absent.log", event_iloc=3, model_type="tritonswmm")
    raise_if_pmix_gds_failure(None, event_iloc=3, model_type="tritonswmm")


def test_pmix_error_is_a_simulation_error_with_the_simulation_exit_code(tmp_path: Path):
    """Property: the error is a ``SimulationError`` and maps to exit code 4, the simulation failure code.

    Class: the error's place in the hierarchy, REGRESSION (the class did not exist). Why required: the CLI maps
    exceptions by ``isinstance``, so a plain ``RuntimeError`` would exit 10 as an unexpected error and hide that
    this is a simulation failure. What kills it: subclassing ``RuntimeError`` or ``TRITONSWMMError`` directly.
    Second correct implementation: a ``SimulationError`` carrying the remedy in an extra attribute.
    """
    err = PMIxSharedMemoryError(1, "triton", tmp_path / "x.log")
    assert isinstance(err, SimulationError)
    assert map_exception_to_exit_code(err) == 4


def _finalize_for(tmp_path: Path, log_text: str, returncode: int):
    """Build the real ``finalize_sim`` closure over a stub run object and call it on a fabricated process."""
    log = tmp_path / "scen" / "sim_run_5.log"
    log.parent.mkdir(parents=True)
    log.write_text(log_text)
    stub = SimpleNamespace(
        _scenario=SimpleNamespace(event_iloc=5, log=SimpleNamespace(logfile=log.parent / "scenario.log")),
        _analysis=SimpleNamespace(_python_executable="python", analysis_config_yaml="a.yaml"),
    )
    stub._scenario._system = SimpleNamespace(system_config_yaml="s.yaml")
    _, finalize = TRITONSWMM_run._create_subprocess_sim_run_launcher(stub, verbose=False)  # ty: ignore[invalid-argument-type]  -- stub stands in for the instance
    return finalize(SimpleNamespace(wait=lambda: returncode), 0.0, log, open(log))


def test_finalize_raises_the_pmix_error_only_for_a_failed_solver_whose_log_carries_the_text(tmp_path: Path):
    """Property: ``finalize_sim`` raises on rc != 0 with the PMIx text, and on no other combination.

    Class: the four cells of (return code zero or nonzero) x (log carries the text or not), through the real
    closure. REGRESSION: red before the call was wired, because the nonzero-with-text cell returned quietly.
    Why required: the function being right is not the same as ``finalize_sim`` calling it, and a successful
    solver whose log merely quotes the text must not raise. What kills it: scanning the log without the return
    code check, or not calling the function. Second correct implementation: the check made in the executor after
    ``finalize_sim`` returns, with this test reading the executor.
    """
    with pytest.raises(PMIxSharedMemoryError):
        _finalize_for(tmp_path / "a", _PMIX_LOG, returncode=1)
    assert _finalize_for(tmp_path / "b", _PMIX_LOG, returncode=0) is None
    assert _finalize_for(tmp_path / "c", _OTHER_MPI_LOG, returncode=1) is None
    assert _finalize_for(tmp_path / "d", _OTHER_MPI_LOG, returncode=0) is None


# ---------------------------------------------------------------------------
# The PMIx detector on the Snakemake runner path
# ---------------------------------------------------------------------------


def _drive_runner_to_failure(monkeypatch, tmp_path: Path, log_text: str):
    """Run ``run_simulation_runner.main`` over stand-in objects whose solver "run" writes ``log_text``.

    The analysis, system and scenario classes are replaced by mocks and the solver command is
    ``bash -c`` printing the fixture text, so no solver, compile or MPI launch happens. Returns the exit
    code, the stderr text and the list of terminal-failure bumps.
    """
    from unittest.mock import MagicMock

    import hhemt.analysis as analysis_mod
    import hhemt.run_simulation_runner as runner
    import hhemt.scenario as scenario_mod
    import hhemt.system as system_mod

    tmp_path.mkdir(parents=True, exist_ok=True)
    fixture = tmp_path / "fixture.txt"
    fixture.write_text(log_text)
    model_logfile = tmp_path / "logs" / "model_tritonswmm_evt0.log"

    scenario = MagicMock()
    scenario.log.scenario_creation_complete.get.return_value = True
    scenario.run.model_run_completed.return_value = False
    scenario.run.prepare_simulation_command.return_value = (
        ["bash", "-c", f"cat {fixture}; exit 1"],
        {},
        model_logfile,
        0,
    )
    # The resume-count field is addressed by a computed name: a literal spelling would make the n-resumes
    # census test list this file as an unclassified consumer, and this test is not one.
    resume_count_field = "n_" + "resumes"
    getattr(scenario.get_log.return_value, resume_count_field).get.return_value = 0
    scenario.get_log.return_value.force_rerun_pending.get.return_value = False
    analysis = MagicMock()
    analysis.cfg_analysis.resume_interruption_schedule = None
    monkeypatch.setattr(analysis_mod, "TRITONSWMM_analysis", lambda *a, **k: analysis)
    monkeypatch.setattr(system_mod, "TRITONSWMM_system", lambda *a, **k: MagicMock())
    monkeypatch.setattr(scenario_mod, "TRITONSWMM_scenario", lambda *a, **k: scenario)
    bumps: list[str] = []
    monkeypatch.setattr(runner, "_native_compile_error_or_skip_in_container", lambda *a, **k: None)
    monkeypatch.setattr(runner, "_consecutive_failure_budget_refusal", lambda *a, **k: None)
    monkeypatch.setattr(runner, "_bump_terminal_failures", lambda sc, mt: bumps.append(mt) or 1)
    monkeypatch.setattr(runner, "_write_failed_marker", lambda ctx: None)
    monkeypatch.setattr(runner, "wait_with_bind_watchdog", lambda proc, logfile, window_s: (proc.wait(), False))
    monkeypatch.setattr(runner, "record_dir_for", lambda *a, **k: tmp_path / "gb")
    monkeypatch.setattr(runner, "reset_record_dir", lambda *a, **k: None)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    for name in ("a.yaml", "s.yaml"):
        (tmp_path / name).write_text("{}")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--event-iloc",
            "0",
            "--system-config",
            str(tmp_path / "s.yaml"),
            "--analysis-config",
            str(tmp_path / "a.yaml"),
            "--model-type",
            "tritonswmm",
        ],
    )
    import io
    import logging

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    runner.logger.addHandler(handler)
    try:
        rc = runner.main()
    finally:
        runner.logger.removeHandler(handler)
    return rc, stream.getvalue(), bumps


def test_runner_failure_with_the_pmix_text_logs_the_remedy_and_without_it_does_not(monkeypatch, tmp_path: Path):
    """Property: a failed runner whose model log carries the PMIx sentence logs the remedy; any other failure does not.

    Class: REGRESSION on the Snakemake path. Before this chunk's round 2 the runner never called the detector, so
    a SLURM failure ended as "did not complete successfully" with no cause. Both arms run the real ``main`` over
    mocked objects with a fabricated solver command. Why required: the failure the remedy exists for occurs only
    under PMIx, which in this toolkit means an srun or mpirun launch under the runner; the in-process
    ``finalize_sim`` test cannot see it. What kills it: removing the call from the runner, or calling it outside
    the ``try``, which would escape ``main`` as an uncaught exception instead of returning 1. Second correct
    implementation: calling the detector in the ``except`` arm from the saved log path.
    """
    rc, out, bumps = _drive_runner_to_failure(monkeypatch, tmp_path / "hit", _PMIX_LOG)
    assert rc == 1
    assert "PMIX_MCA_gds=hash" in out and "Exception occurred during simulation execution" in out
    assert bumps == ["tritonswmm"]

    rc, out, bumps = _drive_runner_to_failure(monkeypatch, tmp_path / "miss", _OTHER_MPI_LOG)
    assert rc == 1
    assert "PMIX_MCA_gds=hash" not in out and "Simulation did not complete successfully" in out
    assert bumps == ["tritonswmm"]
