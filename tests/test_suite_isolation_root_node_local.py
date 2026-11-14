"""The per-chunk isolation root is node-local, removed on every exit the wrapper survives, and harvested on failure.

Work package 1d, chunk 1d.3 of the approved shape. Under ``--isolate-trees-per-chunk`` the chunk wrapper exports a
per-chunk ``HHEMT_TEST_RUNS_ROOT_OVERRIDE``. It used to sit under ``run_dir / "tree_isolation"`` on shared storage and
was never removed. It now sits beside the chunk's pytest basetemp under the system temp directory, the wrapper removes
it on pytest's own exit, on pytest killed and on SIGTERM to the wrapper, a failed chunk's harvest walks it under the
existing caps, and its whole trees are copied to ``run_dir/tree_isolation_failed/chunk-NN`` only with the plan flag
``--keep-failed-trees``. Residual, named: a SIGKILL of the wrapper or a SLURM KillWait expiry leaves the root to the
node's job-end temporary-directory cleanup.

The real wrapper (``python -m hhemt.suite._runner --chunk``) is started as a subprocess against a fixture run
directory, with a stand-in executable in place of the interpreter that runs pytest; no solver is involved, and
signals go only to the wrapper this module starts and the stand-in it records.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

import hhemt.suite._runner as R

_STANDIN = """\
#!/usr/bin/env bash
# Stand-in for the interpreter that runs pytest: records the override, plants a log, then behaves per STANDIN_MODE.
if [ "$1" = "-c" ]; then
  if [ -n "$STANDIN_PROBE_FILE" ]; then echo started > "$STANDIN_PROBE_FILE"; sleep "${STANDIN_PROBE_SLEEP:-0}"; fi
  exit 0
fi
root="$HHEMT_TEST_RUNS_ROOT_OVERRIDE"
echo "$root" > "$STANDIN_ROOT_FILE"
[ -d "$root" ] && echo yes > "$STANDIN_SAW_ROOT_FILE"
mkdir -p "$root/tree/a"
echo "tree log" > "$root/tree/a/build.log"
case "$STANDIN_MODE" in
  pass) exit 0 ;;
  fail) exit 1 ;;
  bulk) mkdir -p "$root/tree/bulk"; seq 1 30000 | sed "s|^|$root/tree/bulk/f|" | xargs touch; exit 1 ;;
  unreadable) echo secret > "$root/tree/secret"; chmod 000 "$root/tree/secret"; exit 1 ;;
  kill) echo $$ > "$STANDIN_PID_FILE"; kill -9 $$ ;;
  hang) echo $$ > "$STANDIN_PID_FILE"; exec sleep 120 ;;
esac
"""


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t" + "@" + "t.invalid", *args],
        check=True,
        capture_output=True,
    )


def _fixture(tmp_path: Path, *, keep: bool) -> dict:
    """Build a repo, a run directory with a one-chunk isolated manifest, and the stand-in executable."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "f.txt").write_text("x\n")
    _git(repo, "add", "f.txt")
    _git(repo, "commit", "-q", "-m", "init")
    run_dir = tmp_path / "run"
    (run_dir / "chunks").mkdir(parents=True)
    (run_dir / "chunks" / "00.txt").write_text("tests/x.py::test_a\n")
    manifest = {"run_id": "isotest", "chunk_count": 1, "tree_isolation_per_chunk": True, "keep_failed_trees": keep}
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    standin = tmp_path / "standin.sh"
    standin.write_text(_STANDIN)
    standin.chmod(0o755)
    node_tmp = tmp_path / "node_tmp"
    node_tmp.mkdir()
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTEST_", "HHEMT_"))}
    env.update(
        TMPDIR=str(node_tmp),
        PYTHONPATH=str(Path(R.__file__).resolve().parents[2]),
        STANDIN_ROOT_FILE=str(tmp_path / "root.txt"),
        STANDIN_SAW_ROOT_FILE=str(tmp_path / "saw.txt"),
        STANDIN_PID_FILE=str(tmp_path / "pid.txt"),
    )
    return {"repo": repo, "run_dir": run_dir, "standin": standin, "env": env, "node_tmp": node_tmp, "tmp": tmp_path}


def _wrapper_cmd(fx: dict) -> list[str]:
    return [
        sys.executable,
        "-m",
        "hhemt.suite._runner",
        "--chunk",
        "0",
        "--run-dir",
        str(fx["run_dir"]),
        "--toolkit",
        str(fx["repo"]),
        "--python",
        str(fx["standin"]),
    ]


def _run(tmp_path: Path, mode: str, *, keep: bool = False) -> dict:
    """Run the real wrapper to completion against the stand-in in ``mode``; return the fixture plus results."""
    fx = _fixture(tmp_path, keep=keep)
    env = {**fx["env"], "STANDIN_MODE": mode}
    done = subprocess.run(_wrapper_cmd(fx), env=env, capture_output=True, text=True, timeout=120, check=False)
    fx["done"] = done
    fx["status"] = json.loads((fx["run_dir"] / "chunk-00.status.json").read_text())
    fx["root"] = Path((tmp_path / "root.txt").read_text().strip())
    return fx


def test_isolation_root_is_node_local_beside_the_basetemp_and_never_under_the_run_dir(tmp_path):
    """The root the stand-in saw is the node-local sibling of the basetemp and is not under the run directory.

    Input class: a chunk of an isolated plan. Required because the node-local configuration guard and the
    local-disk saving both depend on the location. Killed by a wrapper that still roots at
    ``run_dir / "tree_isolation"``. A second correct implementation passes: any root under the basetemp's parent.
    """
    fx = _run(tmp_path, "pass")
    expected = R._chunk_isolation_root("isotest", 0)
    assert fx["root"].parent == fx["node_tmp"]
    assert fx["root"].name == expected.name
    assert fx["root"].parent == Path(fx["status"]["basetemp"]).parent
    assert not fx["root"].is_relative_to(fx["run_dir"])
    assert fx["status"]["runs_root_override"] == str(fx["root"])
    assert (tmp_path / "saw.txt").read_text().strip() == "yes", "the root must exist before pytest starts"


@pytest.mark.parametrize("mode", ["pass", "fail", "kill"])
def test_wrapper_removes_the_root_when_pytest_exits_fails_or_is_killed(tmp_path, mode):
    """The root is gone after pytest exits 0, exits 1 and is killed by SIGKILL.

    Input class: the three ways the child can end that leave the wrapper alive. Required because an unremoved
    root fills the node's temporary directory across array elements. Killed by a wrapper without the cleanup
    ``finally``. A second correct implementation passes: removal in an ``atexit`` hook of the wrapper.
    """
    fx = _run(tmp_path, mode)
    assert not fx["root"].exists(), f"the isolation root survived a {mode} chunk"
    assert (tmp_path / "saw.txt").exists(), "control: the stand-in did see the root while it ran"


def test_wrapper_removes_the_root_on_sigterm_to_the_wrapper(tmp_path):
    """SIGTERM to the wrapper ends the child, removes the root and records a non-zero chunk exit.

    Input class: a wrapper terminated while its child is still running. Required because SLURM sends SIGTERM
    before a hard kill. Killed by a wrapper with no SIGTERM handler (it dies at once and leaves the root).
    The signal goes only to the wrapper this test started; the stand-in's pid is read from the file it wrote.
    """
    fx = _fixture(tmp_path, keep=False)
    env = {**fx["env"], "STANDIN_MODE": "hang"}
    wrapper = subprocess.Popen(_wrapper_cmd(fx), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    pid_file = tmp_path / "pid.txt"
    try:
        deadline = time.monotonic() + 60
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.1)
        assert pid_file.exists(), "the stand-in never started"
        root = Path((tmp_path / "root.txt").read_text().strip())
        assert root.is_dir()
        wrapper.send_signal(signal.SIGTERM)
        wrapper.wait(timeout=60)
    finally:
        if wrapper.poll() is None:
            wrapper.kill()
            wrapper.wait()
        if pid_file.exists():
            try:
                os.kill(int(pid_file.read_text().strip()), signal.SIGKILL)
            except ProcessLookupError:
                pass
    assert not root.exists(), "the isolation root survived SIGTERM to the wrapper"
    status = json.loads((fx["run_dir"] / "chunk-00.status.json").read_text())
    assert status["pytest_exit"] != 0
    assert not (fx["run_dir"] / "tree_isolation_failed").exists()


def test_harvest_walks_the_isolation_root_under_the_existing_caps(tmp_path, monkeypatch):
    """A log under the isolation root is harvested beside the basetemp's, and the file-count cap binds across both.

    Input class: a basetemp holding one log and an isolation root holding three. Required because the heavy
    fixtures' logs now live under the root. Killed by a harvest that walks the basetemp only (the planted log is
    missing) or that applies the cap per root (four files harvested against a cap of two).
    """
    base = tmp_path / "base"
    iso = tmp_path / "iso"
    (base / "d").mkdir(parents=True)
    (iso / "tree").mkdir(parents=True)
    (base / "d" / "pytest.log").write_text("b")
    for i in range(3):
        (iso / "tree" / f"build{i}.log").write_text(f"i{i}")
    unbounded = R._harvest_diagnostics(base, tmp_path / "out1", extra_roots=(iso,))
    names = sorted(h["as"] for h in unbounded["harvested"])
    assert any(n.startswith("isolation_root__tree__build") for n in names), names
    assert len(names) == 4 and not unbounded["dropped"]
    monkeypatch.setattr(R, "HARVEST_MAX_FILES", 2)
    capped = R._harvest_diagnostics(base, tmp_path / "out2", extra_roots=(iso,))
    assert len(capped["harvested"]) == 2
    assert [d["cap"] for d in capped["dropped"]] == ["max_files", "max_files"]


def test_failed_chunk_harvest_includes_the_root_log_in_the_run_dir(tmp_path):
    """The real wrapper's diagnostics bundle for a failed chunk carries the log planted under the root.

    Input class: a chunk that exits 1 after writing a ``*.log`` under the isolation root. Required because the
    root is removed on exit, so the bundle is the only copy of that log. Killed by a wrapper that harvests the
    basetemp alone. A second correct implementation passes: any bundle naming the file.
    """
    fx = _run(tmp_path, "fail")
    bundle = fx["run_dir"] / "chunk-00.diagnostics"
    assert (bundle / "isolation_root__tree__a__build.log").read_text().strip() == "tree log"
    assert fx["status"]["diagnostics"]["harvested"] >= 1


def test_failed_chunk_without_the_flag_writes_nothing_under_tree_isolation_failed(tmp_path):
    """Without ``--keep-failed-trees`` a failed chunk leaves no ``tree_isolation_failed`` directory.

    Input class: a failing chunk of a plan whose manifest has ``keep_failed_trees`` false. Required because the
    trees are large. Killed by a wrapper that copies on every non-zero exit. Control: the passing chunk with the
    flag set also writes nothing, so the directory is created only by flag AND failure together.
    """
    fx = _run(tmp_path / "off", "fail", keep=False)
    assert fx["status"]["pytest_exit"] == 1
    assert not (fx["run_dir"] / "tree_isolation_failed").exists()
    passing = _run(tmp_path / "pass_with_flag", "pass", keep=True)
    assert not (passing["run_dir"] / "tree_isolation_failed").exists()


def test_failed_chunk_with_the_flag_copies_the_whole_trees(tmp_path):
    """With ``--keep-failed-trees`` a failed chunk's whole trees are copied to ``tree_isolation_failed/chunk-NN``.

    Input class: a failing chunk of a plan whose manifest has ``keep_failed_trees`` true. Required so a
    developer can inspect the trees after the node-local root is removed. Killed by a wrapper that ignores the
    flag. The copy path is also recorded in the chunk status.
    """
    fx = _run(tmp_path, "fail", keep=True)
    kept = fx["run_dir"] / "tree_isolation_failed" / "chunk-00"
    assert (kept / "tree" / "a" / "build.log").read_text().strip() == "tree log"
    assert fx["status"]["kept_failed_trees"] == str(kept)
    assert not fx["root"].exists()


def test_plan_flag_is_recorded_in_the_manifest_and_off_by_default():
    """``--keep-failed-trees`` parses to False by default and True when given.

    Input class: the two spellings of the drive command line. Required because the plan records the flag in the
    manifest the chunk reads. Killed by a parser without the flag, or one that defaults it on.
    """
    parser = R.build_parser()
    assert parser.parse_args([]).keep_failed_trees is False
    assert parser.parse_args(["--keep-failed-trees"]).keep_failed_trees is True


def _stray(pattern: str) -> int:
    """Number of processes whose command line contains ``pattern``, excluding the ``ps`` that lists them.

    ``-ww`` is required: without a terminal ``ps`` cuts each line at 80 columns, so a pattern that sits beyond column
    80 (a long temporary path) would never be seen and the count would be a vacuous 0."""
    out = subprocess.run(["ps", "-eww", "-o", "args"], capture_output=True, text=True, check=True).stdout
    return sum(1 for line in out.splitlines() if pattern in line and "ps -eww" not in line)


def _start_marker_process(*marks: str) -> subprocess.Popen:
    """Start ONE process that sleeps for 30 s and carries ``marks`` in its arguments, and never forks.

    A ``bash -c "sleep 30; :"`` helper forks a child that runs ``sleep`` and, between that fork and the exec, the
    child shows the parent's arguments too, so a count of processes carrying the marks can read 2 for a moment
    (measured on loaded cluster nodes: 13 and 22 of 2,400 rounds in two probe jobs). A count of 1 after the parent
    was killed was also observed on the bash form (6 of the 22 anomalies of the second job): a process still carried
    the marks after the parent was killed, and was gone within a second. The interpreter sleeping in-process has no
    such child (0 of 2,400 rounds in the second job)."""
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)", *marks])


def _standin_pattern(tmp_path: Path) -> str:
    """The text that marks this test's own wrapper process: the stand-in path the fixture passes as ``--python``."""
    return str(tmp_path / "standin.sh")


def _child_pid_pattern(tmp_path: Path) -> str:
    """The text that marks this test's own driver child: the pid file its script writes in ``tmp_path``."""
    return f"{tmp_path}/child.pid"


def test_the_scoped_stray_check_counts_this_tests_own_standin_process(tmp_path):
    """A process carrying this test's own ``{tmp_path}/standin.sh`` is counted by the scoped stray check.

    Input class: one helper process whose arguments name the test's own stand-in path. Required as the must-fire
    control for the ``standin.sh`` pattern, so a typo in it cannot pass the stray assertions vacuously. Killed by a
    pattern that does not match the path the wrapper is started with. The helper is always ended and reaped.
    """
    pattern = _standin_pattern(tmp_path)
    helper = _start_marker_process("own", str(tmp_path / "standin.sh"))
    try:
        deadline = time.monotonic() + 10
        while _stray(pattern) == 0 and time.monotonic() < deadline:
            time.sleep(0.05)
        assert _stray(pattern) == 1
    finally:
        helper.kill()
        helper.wait()
    assert _stray(pattern) == 0


def test_the_scoped_stray_check_counts_this_tests_own_child_pid_process(tmp_path):
    """A process carrying this test's own ``{tmp_path}/child.pid`` is counted by the scoped stray check.

    Input class: one helper process whose arguments name the test's own child-pid path. Required as the must-fire
    control for the ``child.pid`` pattern. Killed by a pattern that does not match the path the driver script writes.
    The helper is always ended and reaped.
    """
    pattern = _child_pid_pattern(tmp_path)
    helper = _start_marker_process("own", f"{tmp_path}/child.pid")
    try:
        deadline = time.monotonic() + 10
        while _stray(pattern) == 0 and time.monotonic() < deadline:
            time.sleep(0.05)
        assert _stray(pattern) == 1
    finally:
        helper.kill()
        helper.wait()
    assert _stray(pattern) == 0


def test_the_scoped_stray_check_ignores_a_process_carrying_another_path(tmp_path):
    """A decoy whose arguments name ANOTHER test's paths is not counted by this test's scoped pattern.

    Input class: a decoy carrying a sibling directory's ``standin.sh`` and ``child.pid`` paths. Required because the
    repair exists to stop a sibling test's process being blamed. Killed by the bare ``standin.sh`` / ``child.pid``
    patterns, which count it. The decoy is always ended and reaped.
    """
    other = tmp_path.parent / "another_test"
    decoy = _start_marker_process("decoy", str(other / "standin.sh"), f"{other}/child.pid")
    try:
        deadline = time.monotonic() + 10
        while _stray(str(other / "standin.sh")) == 0 and time.monotonic() < deadline:
            time.sleep(0.05)
        assert _stray(str(other / "standin.sh")) == 1, "the decoy was not visible to ps"
        assert _stray(_standin_pattern(tmp_path)) == 0
        assert _stray(_child_pid_pattern(tmp_path)) == 0
    finally:
        decoy.kill()
        decoy.wait()


def _start_wrapper(fx: dict, mode: str, **extra: str) -> subprocess.Popen:
    """Start the real wrapper against the stand-in in ``mode`` and return the running process."""
    env = {**fx["env"], "STANDIN_MODE": mode, **extra}
    return subprocess.Popen(_wrapper_cmd(fx), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _wait_for(path: Path, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    return path.exists()


def test_sigterm_during_the_start_up_probe_leaves_no_isolation_root(tmp_path):
    """SIGTERM while the wrapper runs its blocking interpreter probe leaves no node-local root behind.

    Input class: a wrapper terminated in the seconds between reading its manifest and starting pytest (the probe of
    ``_sibling_resolution`` is a blocking subprocess there). Required because the root must be created and guarded
    by one ``try``, with no gap. Killed by a wrapper that creates the root before the probe and guards it later (the
    reviewer's reproduction: root left, exit -15). The stand-in probe sleeps 3 s so the signal lands inside it.
    """
    fx = _fixture(tmp_path, keep=False)
    wrapper = _start_wrapper(fx, "pass", STANDIN_PROBE_FILE=str(tmp_path / "probe.txt"), STANDIN_PROBE_SLEEP="3")
    try:
        assert _wait_for(tmp_path / "probe.txt", 60), "the probe never started"
        wrapper.send_signal(signal.SIGTERM)
        wrapper.wait(timeout=60)
    finally:
        if wrapper.poll() is None:
            wrapper.kill()
            wrapper.wait()
    time.sleep(3.5)  # the probe's own sleep child outlives the wrapper by design; let it end before the ps check
    root = fx["node_tmp"] / R._chunk_isolation_root("isotest", 0).name
    assert not root.exists(), "a root exists after SIGTERM during the probe"
    assert _stray(_standin_pattern(tmp_path)) == 0


def test_sigterm_during_the_kept_trees_copy_removes_the_root_and_writes_a_terminated_status(tmp_path):
    """SIGTERM in the post-child phase removes the root and still writes a status naming the termination.

    Input class: a failing chunk with ``--keep-failed-trees`` whose root holds 30,000 small files, so the
    ``copytree`` is still running when the test sends the signal (sent as soon as ``tree_isolation_failed/``
    appears). Required because the wrapper is still the wrapper after its child ends. Killed by a wrapper that
    restores the default SIGTERM action after the child (the reviewer's reproduction: exit -15, root left, no
    status). The status keeps the child's exit 1 and records the termination in the existing free-text field
    ``kept_failed_trees_error``.
    """
    fx = _fixture(tmp_path, keep=True)
    wrapper = _start_wrapper(fx, "bulk")
    kept = fx["run_dir"] / "tree_isolation_failed" / "chunk-00"
    try:
        assert _wait_for(kept, 60), "the kept-trees copy never started"
        wrapper.send_signal(signal.SIGTERM)
        wrapper.wait(timeout=60)
    finally:
        if wrapper.poll() is None:
            wrapper.kill()
            wrapper.wait()
    root = Path((tmp_path / "root.txt").read_text().strip())
    assert not root.exists(), "the isolation root survived SIGTERM during the copy"
    status = json.loads((fx["run_dir"] / "chunk-00.status.json").read_text())
    assert status["pytest_exit"] == 1
    assert "terminated by SIGTERM during the kept-trees copy phase" in status["kept_failed_trees_error"]
    assert status["kept_failed_trees"] is None
    assert _stray(_standin_pattern(tmp_path)) == 0


@pytest.mark.skipif(os.geteuid() == 0, reason="file permissions do not stop root from reading the file")
def test_a_copytree_failure_is_recorded_without_changing_the_verdict(tmp_path):
    """A kept-trees copy that raises is recorded in the status and the chunk still ends with pytest's exit code.

    Input class: a failing chunk with the flag whose root holds an unreadable file. Required because the copy is
    best effort and must not decide the verdict. Killed by a wrapper that lets the copy error escape (no status).
    """
    fx = _run(tmp_path, "unreadable", keep=True)
    assert fx["status"]["pytest_exit"] == 1
    assert fx["status"]["kept_failed_trees"] is None
    assert fx["status"]["kept_failed_trees_error"], "the copy failure was not recorded"
    assert not fx["root"].exists()


_CHILD_DRIVER = """\
import sys
from pathlib import Path
import hhemt.suite._runner as R
rc = R._run_pytest_child(["bash", "-c", sys.argv[1]], Path("."), dict(__import__("os").environ), term_grace_s=1.0)
print("RC=%d" % rc, flush=True)
"""


def _drive_child(tmp_path: Path, script: str, ready: Path) -> tuple[str, Path]:
    """Run ``_run_pytest_child`` in a subprocess over a bash child, SIGTERM that subprocess, return its output."""
    driver = tmp_path / "driver.py"
    driver.write_text(_CHILD_DRIVER)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTEST_", "HHEMT_"))}
    env["PYTHONPATH"] = str(Path(R.__file__).resolve().parents[2])
    proc = subprocess.Popen(
        [sys.executable, str(driver), script], env=env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
    )
    pid_file = tmp_path / "child.pid"
    try:
        assert _wait_for(ready, 60), "the child never started"
        proc.send_signal(signal.SIGTERM)
        out, _ = proc.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        # The child may still hold the driver's stdout pipe, so end it BEFORE draining the pipe.
        if pid_file.exists():
            try:
                os.kill(int(pid_file.read_text().strip()), signal.SIGKILL)
            except ProcessLookupError:
                pass
        proc.communicate()
        out = "TIMEOUT"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        if pid_file.exists():
            try:
                os.kill(int(pid_file.read_text().strip()), signal.SIGKILL)
            except ProcessLookupError:
                pass
    return out, pid_file


def test_a_child_that_exits_zero_after_the_wrapper_was_terminated_is_reported_as_143(tmp_path):
    """A terminated chunk whose child exits 0 reports 143, never a pass.

    Input class: a child that traps SIGTERM and exits 0. Required because a terminated chunk must not read as
    green. Killed by a wrapper that returns the child's 0 (mutant: the 143 mapping removed).
    """
    ready = tmp_path / "ready"
    script = f"echo $$ > {tmp_path}/child.pid; trap 'exit 0' TERM; touch {ready}; while true; do sleep 0.1; done"
    out, _ = _drive_child(tmp_path, script, ready)
    assert "RC=143" in out, out
    assert _stray(_child_pid_pattern(tmp_path)) == 0


def test_a_child_that_ignores_sigterm_is_killed_after_the_grace_period(tmp_path):
    """A child that ignores SIGTERM is killed once ``term_grace_s`` has passed since the wrapper's SIGTERM.

    Input class: a child with SIGTERM ignored, grace 1 s. Required because an unresponsive pytest must not hold the
    SLURM allocation until its hard kill. Killed by a wrapper without the grace kill (mutant: the kill removed),
    which waits forever; the driver then times out and the test fails with output ``TIMEOUT``.
    """
    ready = tmp_path / "ready"
    script = f"echo $$ > {tmp_path}/child.pid; trap '' TERM; touch {ready}; while true; do sleep 0.1; done"
    out, _ = _drive_child(tmp_path, script, ready)
    assert "RC=-9" in out, out
    assert _stray(_child_pid_pattern(tmp_path)) == 0
