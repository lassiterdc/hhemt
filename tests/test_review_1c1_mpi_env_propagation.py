"""Reviewer tests for work chunk 1c.1, repair round 3: the compile job's Open MPI pin reaches the solver launch.

PROBLEM. On a GitHub runner whose host exposes the MANA verbs device, Open MPI selects the `cm` PML and
the solver's first local message aborts. The repair pins `OMPI_MCA_pml` and `OMPI_MCA_btl` at job level in
`compile-tests.yml`. A job-level variable only helps if it survives every hop between the runner's
environment and the `mpirun` the toolkit starts: the toolkit builds its own launch environment, and a hop
that rebuilt it from scratch would drop the pin while every workflow check stayed green. These tests hold
that propagation, the probe workflow's agreement with the pin, and the fingerprint step's behaviour on the
new lines. They start no solver and compile nothing: the only subprocesses are bash and a stub `mpirun`.

Each test docstring states Property, Class, Why it is required, What kills it, and a second correct
implementation under which it still passes.
"""

from __future__ import annotations

import ast
import os
import re
import shutil
import stat
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from tests.test_srun_command_construction import _make_run

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
COMPILE_WORKFLOW = WORKFLOWS / "compile-tests.yml"
PROBE_WORKFLOW = WORKFLOWS / "mpi-transport-probe.yml"
RUNNER_SRC = REPO_ROOT / "src" / "hhemt" / "run_simulation_runner.py"
LAUNCH_SRC = REPO_ROOT / "src" / "hhemt" / "run_simulation.py"
REAL_BASH = "/bin/bash"


def _compile_job() -> dict:
    """The compile-synth-render-gate job mapping of compile-tests.yml."""
    return yaml.safe_load(COMPILE_WORKFLOW.read_text())["jobs"]["compile-synth-render-gate"]


def _job_ompi_pins() -> dict[str, str]:
    """Every OMPI_MCA_ variable the compile job declares at job level, as {key: value}."""
    return {k: str(v) for k, v in (_compile_job().get("env") or {}).items() if k.startswith("OMPI_MCA_")}


def _probe_cases() -> list[tuple[str, str, str]]:
    """The probe matrix's cases as (label, env assignments, mpirun extra args), in file order."""
    doc = yaml.safe_load(PROBE_WORKFLOW.read_text())
    step = next(s for s in doc["jobs"]["probe"]["steps"] if str(s.get("name", "")).startswith("Probe matrix"))
    cases = []
    for line in step["run"].splitlines():
        m = re.match(r'^\s*probe\s+(c\d+_\w+)\s+"([^"]*)"\s+"([^"]*)"', line)
        if m:
            cases.append((m.group(1), m.group(2), m.group(3)))
    return cases


_PINS = _job_ompi_pins()


def test_the_compile_job_pins_at_least_one_ompi_mca_variable():
    """Property: the job-level env carries OMPI_MCA_ keys, so the parametrized tests below are not vacuous.

    Class: the set of OMPI_MCA_ keys in the compile job's env (the population every other test here
    ranges over). A parametrization over an empty set collects zero cases and passes silently, so this
    test is the must-be-nonzero control for all of them.

    Why required: removing the pin turns every parametrized test below into a zero-case pass.

    What kills it: deleting the job-level `env:` mapping or renaming its keys off the OMPI_MCA_ prefix.

    A second correct implementation: any other YAML location for the pin that `jobs.<job>.env` still names.
    """
    assert _PINS, "compile-tests.yml declares no job-level OMPI_MCA_ variable"


@pytest.mark.parametrize("key", sorted(_PINS))
def test_every_job_level_ompi_mca_pin_reaches_the_mpirun_process(key, tmp_path):
    """Property: a job-level OMPI_MCA_ variable set in the parent process reaches the process `mpirun` runs as.

    Class: every OMPI_MCA_ key the compile job pins, driven through the toolkit's real
    `prepare_simulation_command` (run_mode mpi, local) and launched in the exact form
    `run_simulation_runner.py` uses, `Popen(cmd, env={**os.environ, **env})` with `cmd` the
    `bash -lc` command string. `mpirun` is a stub that prints the OMPI_MCA_ variable it sees.

    Reachability: a GitHub runner exports the job env to every step, pytest and snakemake inherit it, and
    the runner process builds `{**os.environ, **env}`; the toolkit's `env` carries LD_LIBRARY_PATH, OMP_*
    and MPICH_OFI_NIC_POLICY only (run_simulation.py, `env["LD_LIBRARY_PATH"]` and the OMP_ lines), so a
    pinned key reaches this launch on every compile-gate run.

    Both-states anchor: the assertion turns on the stub's report of the variable's VALUE, a property of
    the launch that holds before and after any rewording of the workflow comment.

    Invocation isolated: the stub run is the same command from a second tree-independent directory (the
    test's tmp_path holds the stub, HOME is tmp_path so no user profile applies); the control run with the
    variable removed must print nothing, so the printed value is caused by the parent environment.

    Why required: a launch that rebuilt its environment without `os.environ` would drop the pin and the
    compile gate would keep aborting on a MANA host with every workflow check green.

    What kills it: `env=env` in place of `env={**os.environ, **env}`, an `env -i` prefix in the command
    string, or a login-shell reset of the variable.

    A second correct implementation: the toolkit adding the same key through an `export` line in the
    command string, provided the exported value equals the job's.
    """
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    stub = stub_dir / "mpirun"
    stub.write_text(f'#!{REAL_BASH}\necho "SEEN {key}=${{{key}-<unset>}}"\n')
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)

    run = _make_run("mpi", n_mpi_procs=2, n_omp_threads=1, in_slurm=False)
    with patch.object(run, "_analysis_level_model_logfile", return_value=Path("/fake/run.log")):
        with patch.object(
            run, "_retrieve_hotstart_file_for_incomplete_triton_or_tritonswmm_simulation", return_value=None
        ):
            cmd, env, _logfile, _ = run.prepare_simulation_command(pickup_where_leftoff=False, verbose=False)
    assert cmd[:2] == ["bash", "-lc"]

    base = {k: v for k, v in os.environ.items() if not k.startswith("OMPI_MCA_")}
    base["PATH"] = f"{stub_dir}:{base.get('PATH', '')}"
    base["HOME"] = str(tmp_path)

    def launch(parent_env: dict[str, str]) -> str:
        proc = subprocess.Popen(
            cmd, env={**parent_env, **env}, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        out = proc.communicate(timeout=60)[0]
        assert proc.returncode == 0, out
        return out

    control = launch(base)
    assert f"SEEN {key}=<unset>" in control, f"control run saw a value without the variable set: {control!r}"
    treated = launch({**base, key: _PINS[key]})
    assert f"SEEN {key}={_PINS[key]}" in treated, f"{key} did not reach mpirun: {treated!r}"


def _popen_env_defects(source: str) -> list[str]:
    """Describe every Popen call in `source` whose env= argument is a fresh dict not merged over os.environ.

    A call with no env= argument inherits the parent environment and is clean. A call whose env= is
    `os.environ.copy()`, `dict(os.environ, ...)`, or a dict display unpacking `**os.environ` is clean.
    Anything else (a bare name, a dict display with no os.environ unpack) is reported with its line.
    """
    defects = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name != "Popen":
            continue
        env_kw = next((kw.value for kw in node.keywords if kw.arg == "env"), None)
        if env_kw is None:
            continue
        text = ast.unparse(env_kw)
        if "os.environ" not in text:
            defects.append(f"line {node.lineno}: Popen env={text}")
    return defects


_SOURCES = {"run_simulation_runner.py": RUNNER_SRC, "run_simulation.py": LAUNCH_SRC}


@pytest.mark.parametrize("module", sorted(_SOURCES))
def test_no_solver_launch_popen_builds_its_env_without_os_environ(module):
    """Property: no Popen in a solver-launch module hands its child an env that omits os.environ.

    Class: every `subprocess.Popen` call in `run_simulation_runner.py` and `run_simulation.py`, the two
    modules that start `bash -lc ... mpirun` (the runner) and the runner itself (the in-process launcher).
    The behavioural test above replicates the runner's Popen form; this scan binds it to the real source.

    Reachability: both modules are on the compile gate's launch path (snakemake rule -> runner ->
    `bash -lc` -> mpirun); a Popen added with `env=env` there is reachable by any later edit.

    Both-states anchor: the scan turns on whether `os.environ` appears in the env argument, not on any
    comment or name.

    Why required: the behavioural test cannot see a second Popen added elsewhere in these modules.

    What kills it: `env=env`, `env=dict(env)`, or `env={"PATH": ...}` in a Popen call here.

    A second correct implementation: a call with no env argument at all, which inherits the parent.
    """
    source = _SOURCES[module].read_text()
    assert "Popen(" in source, f"{module} has no Popen call; the scan would read zero for the wrong reason"
    assert _popen_env_defects(source) == []


@pytest.mark.parametrize(
    "planted, expected_defects",
    [
        ("proc = subprocess.Popen(cmd, env=env)", 1),
        ("proc = subprocess.Popen(cmd, env={'PATH': '/x'})", 1),
        ("proc = subprocess.Popen(cmd, env={**os.environ, **env})", 0),
        ("proc = subprocess.Popen(cmd, env=os.environ.copy())", 0),
        ("proc = subprocess.Popen(cmd, env=dict(os.environ, A='1'))", 0),
        ("proc = subprocess.Popen(cmd)", 0),
    ],
)
def test_popen_env_scan_flags_the_dropping_shapes_and_clears_the_inheriting_ones(planted, expected_defects):
    """Property: the scan fires on each env shape that drops the parent environment and on no inheriting shape.

    Class: six Popen shapes, two violating and four satisfying in three different correct states
    (dict-merge, `.copy()`, `dict(os.environ, ...)`, and no env argument at all).

    Why required: the scan above passes on the real source, so without a violating arm its green cannot
    be told from a scan that matches nothing.

    What kills it: matching only the `{**os.environ, ...}` spelling, which would flag the `.copy()` form.

    A second correct implementation: a regex over the call text, provided it clears all four inheriting shapes.
    """
    assert len(_popen_env_defects(planted)) == expected_defects


def test_probe_env_form_cases_use_exactly_the_job_pin_and_no_mca_arguments():
    """Property: the probe's environment-variable cases carry the compile job's pin verbatim, with no --mca.

    Class: every probe case that assigns an OMPI_MCA_ variable (c6 and c7 today). The CI confirmation is
    only a confirmation if c6 runs the configuration the compile job pins and does so through the
    environment, the form the job uses; a case that also passed `--mca` would test the argument form and
    pass for a reason the job does not share.

    Reachability: the probe runs on 20 runners, and a MANA host drawn there runs c6; the compile job's env
    is read from the same repository state, so a later edit of either file reaches this check.

    Both-states anchor: the assertion compares the two files' assignment sets, which holds before and
    after any rewording of either file's comments.

    Why required: the confirming dispatch is the only evidence the pin works on a MANA host; if c6 drifts
    from the pin the evidence is about a different configuration.

    What kills it: changing the job's `OMPI_MCA_btl` without c6, a typo in a c6 key, or adding `--mca pml ob1` to c6.

    A second correct implementation: generating c6's assignments from the job env at run time.
    """
    cases = _probe_cases()
    assert len(cases) >= 8, f"probe parse found {len(cases)} cases; the matcher is not reading the matrix"
    env_form = [(label, envs, margs) for label, envs, margs in cases if "OMPI_MCA_" in envs]
    assert env_form, "no probe case assigns an OMPI_MCA_ variable"
    for label, _envs, margs in env_form:
        assert "--mca" not in margs, f"{label} mixes environment assignments with --mca arguments"
    full = [envs for label, envs, margs in env_form if label == "c6_env_fix"]
    assert len(full) == 1
    assignments = dict(tok.split("=", 1) for tok in full[0].split())
    assert assignments == _PINS, f"c6 assigns {assignments}, the compile job pins {_PINS}"


def _fingerprint_run_text() -> str:
    """The Host fingerprint step's run text of compile-tests.yml, found by its name."""
    steps = [s for s in _compile_job()["steps"] if "fingerprint" in str(s.get("name", "")).lower()]
    assert len(steps) == 1
    return steps[0]["run"]


def _fingerprint_problems(
    run_text: str, tmp_path: Path, ompi_info: str | None, parent_env: dict[str, str], mana_host: bool = False
) -> list[str]:
    """Run `run_text` the way the runner does and return the ways it misbehaves, empty when it behaves.

    The PATH holds real coreutils plus stubs: `mpirun` exits 0, `ompi_info` is absent (`ompi_info` None)
    or prints the given text. The step's `/sys/class/infiniband` is redirected to a directory under
    `tmp_path` that is absent, or (`mana_host`) holds a `mana_0` entry, so the result does not depend on
    the test host's own hardware.
    """
    ib_dir = tmp_path / "infiniband"
    if mana_host:
        ib_dir.mkdir()
        (ib_dir / "mana_0").touch()
    run_text = run_text.replace("/sys/class/infiniband", str(ib_dir))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool in ("tee", "timeout", "head", "ls", "env", "grep", "sort", "cat", "nproc", "lscpu", "uname"):
        found = shutil.which(tool)
        if found:
            (bin_dir / tool).symlink_to(found)
    mpirun = bin_dir / "mpirun"
    mpirun.write_text(f"#!{REAL_BASH}\nexit 0\n")
    mpirun.chmod(mpirun.stat().st_mode | stat.S_IEXEC)
    if ompi_info is not None:
        stub = bin_dir / "ompi_info"
        stub.write_text(f"#!{REAL_BASH}\ncat <<'EOT'\n{ompi_info}\nEOT\n")
        stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    summary = tmp_path / "summary.md"
    summary.write_text("")
    env = {"PATH": str(bin_dir), "GITHUB_STEP_SUMMARY": str(summary), "HOME": str(tmp_path), **parent_env}
    proc = subprocess.run(
        [REAL_BASH, "-el", "-c", run_text],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    text = summary.read_text()
    problems = []
    if proc.returncode != 0:
        problems.append(f"step exited {proc.returncode}")
    if "mpirun_np2_rc=0" not in text:
        problems.append("the two-rank launch status line is missing after the new probes")
    ib_marker = "mana_0" if mana_host else "No such file or directory"
    ib_section = text.split("infiniband", 1)[-1].split("--- Open MPI", 1)[0] if "infiniband" in text else ""
    if ib_marker not in ib_section:
        problems.append(f"the infiniband section does not show {ib_marker!r}")
    if "OMPI_MCA_ environment" not in text:
        problems.append("the OMPI_MCA_ environment section is missing")
    for key, value in parent_env.items():
        if key.startswith("OMPI_MCA_") and f"{key}={value}" not in text:
            problems.append(f"{key} is set on the host but the log does not show it")
    if ompi_info is None and "components=unavailable" not in text:
        problems.append("an absent ompi_info is not reported")
    return problems


_COMPONENT_LINES = (
    'mca:pml:ob1:version:"component:5.0.10"\n'
    'mca:pml:cm:version:"component:5.0.10"\n'
    'mca:mtl:ofi:version:"component:5.0.10"\n'
    'mca:btl:sm:version:"component:5.0.10"\n'
    "mca:btl:sm:param:btl_sm_priority:value:30"
)


@pytest.mark.parametrize(
    "ompi_info, parent_env, mana_host",
    [
        (None, {}, False),  # ompi_info absent: no components, no OMPI vars, no verbs directory
        ("ompi_info: nothing parseable", {}, False),  # present, grep matches zero lines (status 1)
        (_COMPONENT_LINES, {}, False),  # healthy host, fix inactive
        (_COMPONENT_LINES, {"OMPI_MCA_pml": "ob1", "OMPI_MCA_btl": "self,sm"}, False),  # fix active
        (_COMPONENT_LINES, {"OMPI_MCA_pml": "ob1", "OMPI_MCA_btl": "self,sm"}, True),  # MANA host, fix active
    ],
    ids=["no-ompi-info", "no-components", "healthy-unpinned", "healthy-pinned", "mana-pinned"],
)
def test_fingerprint_new_probes_report_every_host_state_and_never_fail_the_step(
    tmp_path, ompi_info, parent_env, mana_host
):
    """Property: with the infiniband listing, the component grep and the OMPI_MCA_ print added, the step still
    exits 0 on every host state, prints the two-rank launch status, and names each probe's result.

    Class: five host states over (ompi_info absent | present with no matching line | present with component
    lines), (OMPI_MCA_ pin unset | set) and (verbs directory absent | holding mana_0); the step's
    `/sys/class/infiniband` is redirected to a directory under tmp_path so the result is hardware-independent.

    Reachability: a runner without the verbs directory makes `ls` exit 2 and an empty component grep exits
    1, both under `set +e`; every compile-gate run on a non-MANA host reaches these statuses.

    Both-states anchor: the assertions turn on the exit status and on named lines, not on the added
    commands' exact wording.

    Why required: the fingerprint is the only record of whether a failing run was a MANA host; a step that
    exited non-zero would also hide the pytest result that follows it.

    What kills it: dropping `set +e`, guarding the `ls` behind `[ -d ... ]` so a
    non-MANA host prints nothing, or putting the new lines after the launch status line's `$?` read.

    A second correct implementation: moving the added probes into a script the step calls, provided the
    step's own output names the same results.
    """
    assert _fingerprint_problems(_fingerprint_run_text(), tmp_path, ompi_info, parent_env, mana_host) == []


@pytest.mark.parametrize(
    "mutate",
    [
        lambda t: t.replace("set +e\n", "set -e\n", 1),
        lambda t: t.replace(
            "ls -l /sys/class/infiniband 2>&1", "[ -d /sys/class/infiniband ] && ls -l /sys/class/infiniband", 1
        ),
        lambda t: t.replace("env | grep '^OMPI_MCA_' | sort", "true", 1),
    ],
    ids=["set-e", "guarded-ls", "no-env-print"],
)
def test_fingerprint_checks_detect_each_planted_defect(tmp_path, mutate):
    """Property: the host-state checks above report a problem for each plausible wrong implementation.

    Class: three planted defects of the fingerprint text (`set -e`, an `ls` guarded by a directory test,
    the OMPI_MCA_ print removed), each run under the host state where it differs from the real text (a
    pinned host with no verbs directory). A fourth, deleting the final `exit 0`, is deliberately absent: the
    step's last command is the `| tee` pipeline under a shell without pipefail, so that deletion changes
    no observable and a test that demanded it be caught would grade an equivalent mutant.

    Why required: the test above is green on the real text; without a violating arm a checker that
    returned an empty list for everything would pass it.

    What kills it: a checker that tests only the exit status, which misses all three (with `set -e` the
    group dies at the first failing `ls` but the `| tee` pipeline still exits 0).

    A second correct implementation: grading the same four mutants through the step's rendered summary
    with a different problem vocabulary.
    """
    real = _fingerprint_run_text()
    mutated = mutate(real)
    assert mutated != real, "the planted defect did not change the text; the mutation matcher is stale"
    problems = _fingerprint_problems(
        mutated, tmp_path, _COMPONENT_LINES, {"OMPI_MCA_pml": "ob1", "OMPI_MCA_btl": "self,sm"}
    )
    assert problems, "the planted defect passed every check"
