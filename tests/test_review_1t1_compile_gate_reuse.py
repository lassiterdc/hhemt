"""Reviewer tests for work chunk 1t.1: the compile gate reuses a pass record.

PROBLEM. `compile-tests.yml` ran its compile tier in full on every push to `develop`, including
pushes that changed only prose. Chunk 1t.1 makes the job look up a one-file pass record keyed on
the compile-relevant content of the commit and skip the heavy STEPS on a hit, while the job still
runs, keeps its name and concludes success, so the required context `compile-synth-render-gate`
is always reported by a job that ran. These tests hold the properties that make the skip safe:
every heavy step honours the miss condition, the failure-diagnostics upload stays failure-only,
a record is written only after a full pass, a key that cannot be computed fails the job instead of
becoming an empty key, and the record the reuse step prints is the record the write step wrote.
They read `.github/workflows/compile-tests.yml` and run only `bash` over step text with stub
tools. They compile nothing and run no solver.

Each test names its CLASS, the approved-shape row it enforces is named in the reviewer's round
file, and each states the change that kills it and a second correct implementation under which
it still passes.
"""

from __future__ import annotations

import re
import stat
import subprocess
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
COMPILE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "compile-tests.yml"
JOB = "compile-synth-render-gate"
REAL_BASH = "/bin/bash"

MISS = "steps.pass_record.outputs.cache-hit != 'true'"
HIT = "steps.pass_record.outputs.cache-hit == 'true'"
STATUS_FUNCTION = re.compile(r"\b(success|failure|always|cancelled)\(\)")


def _workflow() -> dict:
    """The parsed compile-tests workflow."""
    return yaml.safe_load(COMPILE_WORKFLOW.read_text())


def _steps() -> list[dict]:
    """The ordered steps of the compile-synth-render-gate job."""
    return _workflow()["jobs"][JOB]["steps"]


def _condition(step: dict) -> str:
    """A step's `if` text with the `${{ }}` wrapper removed, or the empty string when absent."""
    text = str(step.get("if", "")).strip()
    return re.sub(r"^\$\{\{\s*(.*?)\s*\}\}$", r"\1", text)


def _lookup_index(steps: list[dict]) -> int:
    """The index of the cache-restore step the shape calls the lookup."""
    found = [i for i, s in enumerate(steps) if s.get("id") == "pass_record"]
    assert len(found) == 1, f"expected exactly one step with id pass_record, found {len(found)}"
    return found[0]


def _miss_offenders(steps: list[dict]) -> list[str]:
    """Names of post-lookup steps (bar the reuse step) with no status function and no miss condition."""
    start = _lookup_index(steps)
    offenders = []
    for step in steps[start + 1 :]:
        condition = _condition(step)
        if HIT in condition:
            continue  # the reuse step: it runs on a hit, by design
        if STATUS_FUNCTION.search(condition):
            continue  # a status function takes the condition out of the implicit-success() class
        if MISS not in condition:
            offenders.append(str(step.get("name", step)))
    return offenders


def _upload_ok(step: dict) -> bool:
    """True when an upload step's condition keeps `failure()`, so a red run still uploads."""
    return "failure()" in _condition(step)


def _run_step(step: dict, tmp_path: Path, env: dict[str, str], stubs: dict[str, str]) -> tuple[int, str]:
    """Run a step's `run` text under `bash -e` with stub tools, as the runner runs it.

    The runner's shell is `bash -el {0}`; `-l` only sources the login profile, so `-e` is the
    property under test and `-l` is left off to keep the host profile out of the PATH.
    `stubs` maps a tool name to the body of a stub script placed first on PATH.
    Returns (exit status, combined output).
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    for tool, body in stubs.items():
        stub = bin_dir / tool
        stub.write_text(f"#!{REAL_BASH}\n{body}\n")
        stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    run_env = {"PATH": f"{bin_dir}:/usr/bin:/bin", **env}
    done = subprocess.run(
        [REAL_BASH, "-e", "-c", step["run"]],
        cwd=tmp_path,
        env=run_env,
        capture_output=True,
        text=True,
        check=False,
    )
    return done.returncode, done.stdout + done.stderr


def _key_step(steps: list[dict]) -> dict:
    """The step whose `run` text invokes the key script."""
    found = [s for s in steps if "compile_gate_key.py" in str(s.get("run", ""))]
    assert len(found) == 1, f"expected exactly one step running compile_gate_key.py, found {len(found)}"
    return found[0]


def test_every_post_lookup_step_without_a_status_function_carries_the_miss_condition() -> None:
    """Every step after the lookup, bar the reuse step, that names no status function carries the miss condition.

    CLASS: the steps of the job that follow the lookup step. REACHABILITY: a hit with a step
    missing the condition runs that step on a reused commit, which is the full-gate cost the
    chunk exists to remove, or worse a heavy step run against a runner that was never provisioned
    (a provisioning step skipped, a test step run) and fails the job. WHY: an `if:` that names
    no status function gets an implicit `success()`, so a step that omits the miss condition runs
    on every hit. KILLED BY: deleting the miss condition from any heavy step. SECOND CORRECT
    IMPLEMENTATION: a condition written `success() && steps.pass_record.outputs.cache-hit !=
    'true'` (excluded by the status-function clause, as the shape's own selection excludes it) or
    the miss condition wrapped in `${{ }}` still passes. Control: the selection must be non-empty,
    and a planted step list with one heavy step missing the condition must redden.
    """
    steps = _steps()
    after = steps[_lookup_index(steps) + 1 :]
    selected = [s for s in after if HIT not in _condition(s) and not STATUS_FUNCTION.search(_condition(s))]
    assert len(selected) >= 5, f"selection is nearly empty ({len(selected)}); the matcher is broken, not the workflow"
    assert _miss_offenders(steps) == []

    planted = [dict(s) for s in steps]
    victim = next(s for s in planted[_lookup_index(planted) + 1 :] if _condition(s) == MISS)
    victim["if"] = "true"
    assert _miss_offenders(planted) == [victim["name"]], "the control must redden exactly the planted step"


def test_failure_diagnostics_upload_keeps_failure_and_is_not_success_only() -> None:
    """The upload-artifact step's condition contains `failure()`.

    CLASS: the compile job's upload-artifact steps. REACHABILITY: a red full run reaches the
    upload; a miss condition written alone (`if: <miss>`) is an `if:` with no status function, so
    GitHub applies an implicit `success()` and the diagnostics of a failed run are never uploaded,
    which is the evidence loss the 1c.1 chunk repaired. WHY: the upload exists only for a red
    run. KILLED BY: replacing `failure()` with the miss condition. SECOND CORRECT IMPLEMENTATION:
    `failure() && steps.pass_record.outputs.cache-hit != 'true'` still passes. Control: a planted
    upload step carrying only the miss condition must redden.
    """
    uploads = [s for s in _steps() if str(s.get("uses", "")).startswith("actions/upload-artifact@")]
    assert len(uploads) == 1
    assert _upload_ok(uploads[0])
    assert not _upload_ok({"if": MISS}), "the control must redden a success-only upload condition"


def test_pass_record_is_saved_only_after_a_full_passing_run_on_a_miss() -> None:
    """The step using `actions/cache/save` requires `success()` and the miss condition, and the write step does too.

    CLASS: every step that writes or saves the record. REACHABILITY: a failed or cancelled full
    run reaches the save step unless `success()` guards it; a hit reaches it unless the miss
    condition does. WHY: a saved record is the claim "the allowlisted tests ran and passed on
    this tree", and a cache entry cannot be changed once written, so one record saved after a red
    run would reuse that red for the rest of the week. KILLED BY: dropping `success()` from the
    save step. SECOND CORRECT IMPLEMENTATION: folding write and save into one step with the same
    two-part condition passes, because the selection is by action name and by the record path.
    """
    steps = _steps()
    saves = [s for s in steps if str(s.get("uses", "")).startswith("actions/cache/save@")]
    assert len(saves) == 1
    writers = [s for s in steps if "> .compile-gate-pass/record.txt" in str(s.get("run", ""))]
    assert len(writers) == 1, "expected exactly one step writing the record file"
    for step in (saves[0], writers[0]):
        condition = _condition(step)
        assert "success()" in condition and MISS in condition, f"{step.get('name')}: {condition!r}"


def test_restore_and_save_share_one_exact_key_computed_once() -> None:
    """The restore and save steps take the same `key` expression, and the restore step has no `restore-keys`.

    CLASS: the two cache steps. REACHABILITY: a run that crosses an ISO-week boundary between
    lookup and save would store the record under a different week if the key were computed twice.
    WHY: "exact key and no restore-keys" is the shape's reuse condition; a prefix match would reuse
    a pass for a different tree. KILLED BY: adding `restore-keys:` or computing the key inline in
    one step. SECOND CORRECT IMPLEMENTATION: the key held in a job-level `env` and read by both
    steps fails this selection only if the two `key` strings differ, so it passes when they match.
    """
    steps = _steps()
    restore = next(s for s in steps if str(s.get("uses", "")).startswith("actions/cache/restore@"))
    save = next(s for s in steps if str(s.get("uses", "")).startswith("actions/cache/save@"))
    assert "restore-keys" not in restore["with"]
    assert restore["with"]["key"] == save["with"]["key"]
    assert restore["with"]["path"] == save["with"]["path"] == ".compile-gate-pass/record.txt"


def test_job_keeps_its_required_context_name_and_is_the_only_job() -> None:
    """The workflow has exactly one job, `compile-synth-render-gate`, with no `name:` override.

    CLASS: the workflow's jobs. REACHABILITY: a branch ruleset matches a required context by the
    job's display name; a renamed or second job leaves the context Pending and blocks a release
    PR. WHY: "only steps are skipped, never the job" is what lets the context report success on a
    hit. KILLED BY: renaming the job or adding a `name:`. SECOND CORRECT IMPLEMENTATION: moving the
    steps into a reusable workflow whose caller job keeps this key passes only if the file still
    declares this job, so this selection is the file's, not the implementation's.
    """
    jobs = _workflow()["jobs"]
    assert list(jobs) == [JOB]
    assert "name" not in jobs[JOB]


def test_force_full_run_is_a_false_default_boolean_and_gates_the_lookup() -> None:
    """`workflow_dispatch` declares boolean `force_full_run` (default false) and it gates the lookup step.

    CLASS: the dispatch inputs and the lookup step. REACHABILITY: a developer dispatching the
    workflow with the input true reaches the lookup. WHY: the bypass is how a release walk forces a
    full compile. KILLED BY: removing the lookup condition, or defaulting the input to true.
    SECOND CORRECT IMPLEMENTATION: `if: inputs.force_full_run != true` on the lookup passes, since
    the selection reads that the condition names the input and not its exact spelling.
    """
    doc = _workflow()
    # PyYAML reads the bare key `on` as the boolean True.
    triggers = doc.get("on", doc.get(True))
    declared = triggers["workflow_dispatch"]["inputs"]["force_full_run"]
    assert declared["type"] == "boolean"
    assert declared["default"] is False
    steps = _steps()
    assert "inputs.force_full_run" in _condition(steps[_lookup_index(steps)])


def test_a_key_script_failure_fails_the_key_step(tmp_path: Path) -> None:
    """The key step exits non-zero when the key script exits 2, and writes the key when it exits 0.

    CLASS: every step whose `run` text invokes `scripts/compile_gate_key.py`. REACHABILITY: the
    script exits 2 on a git error or an empty listing (the shape's own ground: one constant key for
    four commits); the step is reached on every push, so any git failure on the runner reaches it.
    WHY: `echo "key=$(cmd)" >> "$GITHUB_OUTPUT"` returns the status of `echo`, not of `cmd`, so a
    failing script writes `key=` and the step succeeds; the step's own comment says it fails the
    job, and what happens next depends on how `actions/cache` treats an empty key rather than on
    this workflow. KILLED BY: capturing the key in an assignment (`key=$(python3 ...)`) or a
    redirect, then echoing it. SECOND CORRECT IMPLEMENTATION: `python3 scripts/compile_gate_key.py
    > key.txt` followed by `echo "key=$(cat key.txt)" >> "$GITHUB_OUTPUT"` also passes, because the
    selection is the step's exit status under `bash -e`. Control: the same step with a succeeding
    script must exit 0 and write the key line, so a harness that cannot succeed is not a pass.
    """
    key_step = _key_step(_steps())
    for tmp_name, script_rc, expect_ok in (("fail", 2, False), ("ok", 0, True)):
        case_dir = tmp_path / tmp_name
        case_dir.mkdir()
        out = case_dir / "github_output"
        body = (
            f"echo compile-gate-pass-v1-Linux-2026-W46-abc\nexit {script_rc}" if expect_ok else "echo boom >&2\nexit 2"
        )
        rc, output = _run_step(key_step, case_dir, {"GITHUB_OUTPUT": str(out)}, {"python3": body})
        if expect_ok:
            assert rc == 0, output
            assert out.read_text().strip() == "key=compile-gate-pass-v1-Linux-2026-W46-abc"
        else:
            assert rc != 0, (
                f"the key step exited 0 although the key script exited 2; GITHUB_OUTPUT now holds {out.read_text()!r}"
            )


def test_the_reuse_step_prints_the_record_the_write_step_wrote(tmp_path: Path) -> None:
    """The reuse step's log and summary carry the commit, run id, attempt and date the write step recorded.

    CLASS: the (write step, reuse step) pair read over four record fields. REACHABILITY: a hit
    restores the file the write step produced on an earlier run. WHY: RR-7 form A rests the release
    gate on the reuse step naming the run it rests on; a field the writer omits or the reader does
    not print makes `/release-project` item (c) cite nothing. KILLED BY: dropping a field from the
    write step or reading a differently named key in the reuse step. SECOND CORRECT IMPLEMENTATION:
    a record in JSON that the reuse step prints with `jq` passes, since the selection is the four
    values appearing in the summary, not the file format.
    """
    steps = _steps()
    write = next(s for s in steps if "> .compile-gate-pass/record.txt" in str(s.get("run", "")))
    reuse = next(s for s in steps if HIT in _condition(s))
    write_step = dict(write, run=write["run"].replace("${{ steps.gate_key.outputs.key }}", "KEY-UNDER-TEST"))
    rc, output = _run_step(
        write_step,
        tmp_path,
        {"GITHUB_SHA": "c0ffee00", "GITHUB_RUN_ID": "987654321", "GITHUB_RUN_ATTEMPT": "2"},
        {},
    )
    assert rc == 0, output
    summary = tmp_path / "summary.md"
    rc, output = _run_step(
        reuse,
        tmp_path,
        {
            "GITHUB_STEP_SUMMARY": str(summary),
            "GITHUB_SERVER_URL": "https://github.example",
            "GITHUB_REPOSITORY": "owner/repo",
        },
        {},
    )
    assert rc == 0, output
    for text in (summary.read_text(), output):
        for needle in (
            "c0ffee00",
            "987654321",
            "attempt=2",
            "date=20",
            "https://github.example/owner/repo/actions/runs/987654321",
        ):
            assert needle in text, f"{needle!r} missing from {text!r}"
