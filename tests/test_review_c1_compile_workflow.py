"""Reviewer tests for work chunk 1c.1: compile-tests.yml host fingerprint and failure upload.

PROBLEM. The compile-gate job failed on the release PR and kept nothing that named the
failing rule, so the failure could not be told apart between two hypotheses (H1 the runner's
MPI refuses two ranks, H2 two multi-sim members fail). Chunk 1c.1 adds a fingerprint step
that must report the MPI verdict on every run, and an upload step that must carry the logs
and `_status` flags of a red run. These tests hold the properties that make those two steps
evidence rather than decoration. They read `.github/workflows/` only and run no solver.

Each test names its CLASS, the approved-shape row it enforces is named in the reviewer's
round file, and each states the change that kills it and a second correct implementation
under which it still passes.
"""

from __future__ import annotations

import glob
import stat
import subprocess
from pathlib import Path

import pytest
import yaml

from hhemt.cache import paths as cache_paths

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
COMPILE_WORKFLOW = WORKFLOWS / "compile-tests.yml"

REAL_BASH = "/bin/bash"
SYSTEM_TOOLS = ("tee", "timeout", "head")  # tools the step uses that are not under test


def _compile_job_steps() -> list[dict]:
    """The ordered steps of the compile-synth-render-gate job."""
    doc = yaml.safe_load(COMPILE_WORKFLOW.read_text())
    return doc["jobs"]["compile-synth-render-gate"]["steps"]


def _fingerprint_step() -> dict:
    """The step the shape calls the host fingerprint step, found by its name."""
    matches = [s for s in _compile_job_steps() if "fingerprint" in str(s.get("name", "")).lower()]
    assert len(matches) == 1, f"expected exactly one fingerprint step, found {len(matches)}"
    return matches[0]


def _upload_step() -> dict:
    """The compile job's artifact upload step."""
    matches = [s for s in _compile_job_steps() if str(s.get("uses", "")).startswith("actions/upload-artifact@")]
    assert len(matches) == 1, f"expected exactly one upload-artifact step, found {len(matches)}"
    return matches[0]


def _stub_bin(tmp_path: Path, behaviours: dict[str, int | None]) -> Path:
    """A PATH directory holding only the tools a behaviour names, plus the untested system tools.

    `behaviours[tool]` is an exit status for a stub that prints a marker line and exits with it,
    or None for a tool that is absent from PATH altogether.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool in SYSTEM_TOOLS:
        (bin_dir / tool).symlink_to(Path("/usr/bin") / tool)
    for tool, rc in behaviours.items():
        if rc is None:
            continue
        stub = bin_dir / tool
        stub.write_text(f"#!{REAL_BASH}\necho 'stub {tool} output'\nexit {rc}\n")
        stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    return bin_dir


def _run_fingerprint(tmp_path: Path, behaviours: dict[str, int | None]) -> tuple[int, str, str]:
    """Run the fingerprint step's `run` text the way the runner does, with controlled tools.

    Returns (exit status, stdout, text appended to GITHUB_STEP_SUMMARY).
    """
    bin_dir = _stub_bin(tmp_path, behaviours)
    summary = tmp_path / "summary.md"
    summary.write_text("")
    env = {"PATH": str(bin_dir), "GITHUB_STEP_SUMMARY": str(summary), "HOME": str(tmp_path)}
    proc = subprocess.run(
        [REAL_BASH, "-el", "-c", _fingerprint_step()["run"]],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    return proc.returncode, proc.stdout, summary.read_text()


_ALL_TOOLS = ("nproc", "lscpu", "hwloc-calc", "ompi_info", "mpirun")


@pytest.mark.parametrize(
    "mpirun_rc, others_rc",
    [
        (0, 0),  # control: a healthy host
        (1, 0),  # MPI refuses two ranks (H1's signature)
        (124, 0),  # the two-rank launch hangs and `timeout` kills it
        (127, 0),  # mpirun absent from PATH
        (1, 1),  # every probe tool fails at once
    ],
)
def test_fingerprint_never_fails_and_reports_the_mpirun_status_it_observed(tmp_path, mpirun_rc, others_rc):
    """Property: the fingerprint step exits 0 and prints the two-rank launch's own status.

    Class: every combination of (two-rank launch status) x (the other probe tools succeeding,
    failing or absent). A refused `mpirun` is the finding H1 is decided by, so the step must
    print `mpirun_np2_rc=<that status>` and must not turn red, because a red here would hide
    the pytest result that follows it on the same run.

    Why required: OI-1 closes on a fingerprint showing the launch refused. A step that
    swallowed the status, or printed a constant, closes it on nothing; a step that exited
    non-zero would fail the job before pytest and lose the simulation evidence.

    What kills it: printing a fixed `mpirun_np2_rc=0`, reading `$?` after another command, or
    dropping `set +e` / the final `exit 0`.

    A second correct implementation: moving the body to a script under `scripts/` that the
    step calls, because the test runs the step's `run` text with the repository as cwd.
    """
    behaviours = {tool: others_rc for tool in _ALL_TOOLS if tool != "mpirun"}
    behaviours["mpirun"] = None if mpirun_rc == 127 else mpirun_rc
    rc, stdout, summary = _run_fingerprint(tmp_path, behaviours)
    assert rc == 0, f"fingerprint step exited {rc}; stdout={stdout!r}"
    assert f"mpirun_np2_rc={mpirun_rc}" in summary, f"status {mpirun_rc} not reported; summary={summary!r}"


def test_fingerprint_reports_a_missing_probe_tool_instead_of_staying_silent(tmp_path):
    """Property: an absent probe tool leaves a named line, so absence is not read as success.

    Class: each of hwloc-calc and ompi_info absent from PATH. `hwloc_cores` and the
    `ompi_info` section are how H1 is separated from a host that merely has fewer cores.

    What kills it: guarding the probe with `|| true` and printing nothing.

    A second correct implementation: any wording that names the tool and says it is unavailable.
    """
    behaviours = {"nproc": 0, "lscpu": 0, "hwloc-calc": None, "ompi_info": None, "mpirun": 0}
    rc, _, summary = _run_fingerprint(tmp_path, behaviours)
    assert rc == 0
    assert "hwloc-calc" in summary and "unavailable" in summary
    assert "ompi_info" in summary and "unavailable" in summary
    assert "mpirun_np2_rc=0" in summary


def _expand(pattern: str, home: Path, tmp_root: Path) -> list[Path]:
    """Expand one upload path pattern against a fake HOME and a fake /tmp the way the action does.

    Python's recursive glob with include_hidden matches `**` against zero or more directories
    and walks dot-directories, which is what `include-hidden-files: true` asks of the action.
    """
    if pattern.startswith("~/"):
        pattern = str(home / pattern[2:])
    elif pattern.startswith("/tmp/"):
        pattern = str(tmp_root / pattern[len("/tmp/") :])
    return [Path(p) for p in glob.glob(pattern, recursive=True, include_hidden=True) if Path(p).is_file()]


def test_upload_patterns_reach_every_diagnostic_file_at_every_analysis_depth(tmp_path):
    """Property: the upload patterns match the files that name a failing rule or member.

    Class: every file under a `logs/` or `_status/` directory at any depth below a synthetic
    run tree, in either root the multi-sim and sensitivity fixtures write to (the cache root
    and pytest's basetemp). Depth varies on purpose: the master Snakemake log sits one level
    below the analysis directory, member logs and flags sit one or two levels deeper, and a
    single-level glob such as `logs/*` reaches the first only.

    Why required: OI-1 closes on a `_status/` listing naming members {1, 2} (H2) and the
    master log carries the `Error in rule` blocks. A pattern that reaches neither uploads a
    green-looking empty artifact, which is the failure the step exists to end.

    What kills it: `logs/*` in place of `logs/**`, dropping a root, or dropping `_status`.

    A second correct implementation: the shape's own wider `**/_status/**` form, which still
    reaches every file below.
    """
    home = tmp_path / "home"
    tmp_root = tmp_path / "tmp"
    # The directory name comes from the product definition, so this module carries no second spelling of the
    # runs-root literal (tests/test_synth_triton_source_cache.py::test_single_sourced_runs_root_literal).
    cache_run = home / ".cache" / "hhemt" / cache_paths.synthetic_runs_root().name / "addr" / "analysis_a"
    base_run = tmp_root / "pytest-of-runner" / "pytest-0" / "synth_multi_sim_builder0" / "analysis_b"
    diagnostics: list[Path] = []
    for run in (cache_run, base_run):
        diagnostics += [
            run / "logs" / "snakemake_master.log",
            run / "logs" / "sims" / "evt1.log",
            run / "_status" / "scenario_prep_evt-1.flag",
            run / "_status" / "_submitted" / "run_evt-1.json",
            run / "sa_1" / "logs" / "snakemake_master.log",
            run / "sa_1" / "_status" / "run_evt-2.flag",
            run / "sa_1" / "scenarios" / "evt-2" / "logs" / "run_tritonswmm.log",
        ]
    for path in diagnostics:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")

    patterns = [line.strip() for line in str(_upload_step()["with"]["path"]).splitlines() if line.strip()]
    matched = {p for pattern in patterns for p in _expand(pattern, home, tmp_root)}
    missed = sorted(str(p.relative_to(tmp_path)) for p in diagnostics if p not in matched)
    assert matched, "control failed: no pattern matched any file, so the expansion is not measuring anything"
    assert not missed, f"diagnostic files no upload pattern reaches: {missed}"


def test_every_upload_step_that_reads_a_dot_directory_includes_hidden_files():
    """Property: an upload path under a dot-directory is paired with `include-hidden-files: true`.

    Class: every `actions/upload-artifact` step in every workflow whose `path` has a segment
    beginning with `.`. upload-artifact v4 and later skips such paths by default and reports
    success on zero files, so the omission is silent; the run's cache root is `~/.cache/...`.

    What kills it: deleting the `include-hidden-files` line from the compile job's step.

    A second correct implementation: uploading from a non-hidden staging directory, which has
    no dot segment and is outside the class.
    """
    steps_checked = 0
    violations: list[str] = []
    for workflow in sorted(WORKFLOWS.glob("*.yml")):
        doc = yaml.safe_load(workflow.read_text())
        for job_name, job in (doc.get("jobs") or {}).items():
            for step in job.get("steps") or []:
                if not str(step.get("uses", "")).startswith("actions/upload-artifact@"):
                    continue
                steps_checked += 1
                paths = [p.strip() for p in str(step["with"].get("path", "")).splitlines() if p.strip()]
                dotted = any(seg.startswith(".") and seg not in (".", "..") for p in paths for seg in p.split("/"))
                if dotted and step["with"].get("include-hidden-files") is not True:
                    violations.append(f"{workflow.name}:{job_name}:{step.get('name')}")
    assert steps_checked >= 1, "control failed: no upload-artifact step found in any workflow"
    assert not violations, f"upload steps reading a dot-directory without include-hidden-files: {violations}"
