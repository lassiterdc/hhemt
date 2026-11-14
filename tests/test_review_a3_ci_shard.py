"""The sharded `build` check: the shard option partitions the fast tier and the fan-in cannot report a red shard green.

Problem this file pins. `.github/workflows/test.yml` runs the fast tier across a matrix of runners, each selecting the
test files whose stable hash falls in its residue class (`--ci-shard K/N`, `tests/conftest.py`), and a fan-in job
named exactly `build (ubuntu-latest, 3.12)` is the one required status context. Three failures are silent and share
one shape, a check that reads green over a broken partition: a shard key that differs between processes leaves some
file in no shard; a fan-in that GitHub skips when a shard fails is reported as passing; a partition check that
accepts an overlap, a gap or a node id nobody collected certifies a broken matrix. Nothing red names any of them, so
each is pinned by running the thing that would have to refuse it.

Every test below carries its CLASS (REGRESSION is red on the tree before chunk 1a.3, where the option and the shard
job do not exist; NEW CAPABILITY names the plausible wrong implementation instead), the input class it quantifies
over, why the property is required, the production change that turns it red, and a second correct implementation
under which it still passes.

Not established: the GitHub-side semantics (`if: always()` keeping a job alive after a failed `needs`, the matrix
aggregate `needs.shard.result`). Nothing here starts an Actions run, so those two facts rest on GitHub's documented
behaviour and on the shape; the tests pin that the workflow is written to rely on them, not that GitHub honours it.
"""

from __future__ import annotations

import fnmatch
import itertools
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

import scripts.check_ruleset_sync as ruleset_guard  # repo root is on sys.path under pytest

pytest_plugins = ["pytester"]

_REPO = Path(__file__).resolve().parents[1]
_WORKFLOW = _REPO / ".github" / "workflows" / "test.yml"

REQUIRED_CONTEXT = "build (ubuntu-latest, 3.12)"
# The fast-tier population, spelled as the approved shape spells it. It is the population the previous single job
# ran, so the shard step and the fan-in's unsharded collection must both carry exactly this expression.
FAST_TIER_MARKER_EXPRESSION = "not slow and not compile_tier and not requires_snakemake_subprocess"


def _workflow() -> dict:
    return yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))


def _fan_in_and_shard_jobs() -> tuple[dict, dict[str, dict]]:
    """The job named `build` (the fan-in) and every other job, by job id."""
    jobs = _workflow()["jobs"]
    assert "build" in jobs, f"no job `build` in {_WORKFLOW}: the fan-in must keep the job id the required check used"
    return jobs["build"], {job_id: job for job_id, job in jobs.items() if job_id != "build"}


# ---------------------------------------------------------------------------------------------------------------
# The fan-in's shape
# ---------------------------------------------------------------------------------------------------------------


def test_exactly_one_job_produces_the_required_context_and_it_is_the_fan_in():
    """Property: across every job and every matrix combination, the required check name has ONE producer, the fan-in.

    Class: NEW CAPABILITY. The plausible wrong implementation is a shard job named
    `build (${{ matrix.os }}, ${{ matrix.python-version }})`, which renders to the required context on every shard
    (the old job's name was derived exactly so), so one green shard would satisfy the required context while the
    others run unwatched. A literal `name: build (ubuntu-latest, 3.12)` on a matrix job does NOT collide, because the
    repository's instrument appends the matrix values; that arm was run and stays green.
    Input class: every job in test.yml, expanded over its matrix, using the repository's own check-name instrument
    (`scripts/check_ruleset_sync.job_check_names`) rather than a second expansion written here.
    Why required: the required context is read by name; two producers make the context report whichever finishes
    first, and zero producers hang the release PR at "Expected -- Waiting for status to be reported".
    What kills it: a shard job name templated to the required context, or giving the fan-in a templated name.
    Second correct implementation: the fan-in could be a different job id with `name: build (ubuntu-latest, 3.12)`;
    the test reads display names, not job ids.
    """
    jobs = _workflow()["jobs"]
    producers = {
        job_id: names
        for job_id, job in jobs.items()
        if (names := ruleset_guard.job_check_names(job_id, job)) is not None and REQUIRED_CONTEXT in names
    }
    assert list(producers) == ["build"], f"producers of {REQUIRED_CONTEXT!r}: {producers}"
    # Must-fire control for the instrument: the guard reports a name for every job, so a None would have hidden one.
    assert all(ruleset_guard.job_check_names(job_id, job) for job_id, job in jobs.items())


def test_fan_in_waits_on_every_other_job_and_runs_even_when_one_failed():
    """Property: the fan-in `needs` every non-fan-in job and carries a job-level `if` containing `always()`.

    Class: NEW CAPABILITY. The plausible wrong implementation is `needs: [shard]` with no `if:`, which GitHub
    resolves as skipped when a shard fails, and a skipped required job is reported as passing.
    Input class: every job id other than `build`.
    Why required: the required check must go red on a red shard; a skipped fan-in is the one state that cannot.
    What kills it: deleting the job-level `if: always()`, or adding a job without adding it to `needs`.
    Second correct implementation: `if: ${{ always() }}` or `if: always() && github.event_name != 'x'`; only the
    substring is read.
    """
    fan_in, others = _fan_in_and_shard_jobs()
    assert others, "the fan-in has nothing to fan in"
    needs = fan_in.get("needs")
    needs = [needs] if isinstance(needs, str) else list(needs or [])
    assert set(needs) == set(others), f"needs {needs} must be exactly the other jobs {sorted(others)}"
    assert "always()" in str(fan_in.get("if", "")), f"fan-in job-level `if` is {fan_in.get('if')!r}"


def _all_success_steps(fan_in: dict) -> list[dict]:
    """Steps of the fan-in that read a `needs.*.result` expression through `env:` (the way the workflow reads it)."""
    return [
        step for step in fan_in["steps"] if any("needs." in str(value) for value in (step.get("env") or {}).values())
    ]


def test_the_all_success_step_runs_after_a_failure_and_exits_one_unless_the_result_is_success():
    """Property: ONE fan-in step reads every shard job's `needs` result, is allowed to run after a failure, and its
    script exits 0 for `success` and 1 for every other result a `needs` entry can take.

    Class: NEW CAPABILITY. Plausible wrong implementations: a step without `if: always()` (skipped after the first
    failed step, so it never reaches its `exit 1`); a script that exits 1 only for `failure` (a cancelled or skipped
    shard then passes); a script comparing against `skipped` alone.
    Input class: the four non-success results GitHub gives a `needs` entry (failure, cancelled, skipped) plus the
    empty string, against the one success value. The script is EXECUTED under the shell flags GitHub uses for `run`.
    Why required: a cancelled or skipped shard means part of the tier never ran, and the required check must not pass.
    What kills it: `[ "${SHARD_RESULT}" != "success" ]` rewritten as `= "failure"`; or deleting the step's `if:`.
    Second correct implementation: any env variable name, `test "$X" = success || exit 1`, or `[[ ... ]]`; every env
    variable that carries a `needs.` expression is set to the result, and the script text is not matched.
    Not covered: a fan-in that fails through an `if:` expression with a bare `run: exit 1` and no env read; that
    shape is a different design from the one step the shape names, and this test would report it as no step found.
    """
    fan_in, others = _fan_in_and_shard_jobs()
    steps = _all_success_steps(fan_in)
    assert len(steps) == 1, f"expected exactly one all-success step, found {[s.get('name') for s in steps]}"
    step = steps[0]
    assert "always()" in str(step.get("if", "")), f"the all-success step's `if` is {step.get('if')!r}"
    env_names = [name for name, value in step["env"].items() if "needs." in str(value)]
    for job_id in others:
        assert any(f"needs.{job_id}.result" in str(step["env"][n]) for n in env_names), (
            f"no env variable of the all-success step reads needs.{job_id}.result"
        )

    def run_with(result: str) -> int:
        env = {"PATH": os.environ["PATH"], **{name: result for name in env_names}}
        return subprocess.run(
            ["bash", "--noprofile", "--norc", "-e", "-o", "pipefail", "-c", step["run"]],
            env=env,
            capture_output=True,
            text=True,
        ).returncode

    assert run_with("success") == 0
    # The must-fire arm for the exit-0 above: every non-success result must be refused.
    refused = {result: run_with(result) for result in ("failure", "cancelled", "skipped", "")}
    assert refused == {"failure": 1, "cancelled": 1, "skipped": 1, "": 1}, refused


# ---------------------------------------------------------------------------------------------------------------
# The cross-step contract: what a shard records, what is uploaded, what the fan-in reads
# ---------------------------------------------------------------------------------------------------------------


def _commands_only(script: str) -> str:
    """The shell text without whole-line comments, so a comment that names an option is not read as using it."""
    return "\n".join(line for line in script.splitlines() if not line.lstrip().startswith("#"))


def _shard_job() -> dict:
    _, others = _fan_in_and_shard_jobs()
    assert len(others) == 1, f"expected one shard job, found {sorted(others)}"
    return next(iter(others.values()))


def _step_with(job: dict, needle: str, key: str) -> dict:
    hits = [step for step in job["steps"] if needle in str(step.get(key, ""))]
    assert len(hits) == 1, f"steps whose {key!r} contains {needle!r}: {[s.get('name') for s in hits]}"
    return hits[0]


def test_the_matrix_lists_every_residue_label_once():
    """Property: `matrix.shard` is exactly 1..shard-count, and `shard-count` has one value.

    Class: NEW CAPABILITY. Plausible wrong implementation: `shard: [1, 2, 3]` left behind when `shard-count` was
    raised to 8, which leaves five residue classes in no job; the fan-in would catch it only after the run.
    Input class: the matrix as written.
    Why required: shard K owns residue K mod N, so a missing label is a file class nobody runs.
    What kills it: dropping a label, or listing two shard counts.
    Second correct implementation: N = 4 with `shard: [1, 2, 3, 4]`; the test never reads a particular N.
    """
    matrix = _shard_job()["strategy"]["matrix"]
    assert len(matrix["shard-count"]) == 1, matrix["shard-count"]
    assert sorted(matrix["shard"]) == list(range(1, matrix["shard-count"][0] + 1)), matrix


def test_shard_step_and_the_unsharded_collection_select_the_same_population_the_old_job_ran():
    """Property: the shard step's `-m` expression and the fan-in's unsharded collection's are both the fast-tier
    expression, and both pass `tests` as a path argument.

    Class: NEW CAPABILITY. Plausible wrong implementation: the fan-in collecting with a looser expression (a
    population wider than the shards ran), so the partition check reports every excluded test as missing, or a
    narrower one, so it reports every shard test as extra; or either command omitting `tests`, which leaves
    `--ci-shard` unrecognised (measured by the coder: a bare run does not load tests/conftest.py before option
    parsing).
    Input class: every `pytest` invocation in the two jobs' `run` scripts.
    Why required: the partition check compares two selections; they are comparable only when the expression is one.
    What kills it: editing one of the two `-m` strings.
    Second correct implementation: moving the expression into an env variable or a pytest.ini marker default would
    change what is read, so it is NOT second-correct here; the test reads the expression in the shell text.
    """
    fan_in, _ = _fan_in_and_shard_jobs()
    shard_run = _commands_only(_step_with(_shard_job(), "--ci-shard", "run")["run"])
    collect_run = _commands_only(_step_with(fan_in, "--collect-only", "run")["run"])
    for label, text in (("shard step", shard_run), ("unsharded collection", collect_run)):
        expressions = re.findall(r'-m\s+"([^"]*)"', text)
        assert expressions == [FAST_TIER_MARKER_EXPRESSION], f"{label}: {expressions}"
        assert re.search(r"\bpytest\s+tests\b", text), f"{label} does not pass `tests` as a path argument"
    assert "--ci-shard" not in collect_run, "the unsharded collection must not be sharded"


def test_recorded_ids_are_uploaded_downloaded_and_read_under_one_name_and_layout():
    """Property: the file the shard records is under the directory it uploads, the artifact name matches the fan-in's
    download pattern, and the partition-check script reads `shard-artifacts/ci-shard-K-of-N/nodeids.txt`.

    Class: NEW CAPABILITY. Plausible wrong implementation: renaming the upload (`shard-K`) while the download keeps
    `ci-shard-*`, so the fan-in downloads nothing and the script sees zero shards.
    Input class: the four steps that carry the contract (record argument, upload, download, script).
    Why required: each half is correct alone; only the join is wrong, and the failure is a red fan-in that names no
    cause the developer can read from the shards.
    What kills it: changing the upload `name:` template, the download `pattern:`/`path:`, or the record path.
    Second correct implementation: a different artifact prefix changed in all three places at once passes.
    """
    fan_in, _ = _fan_in_and_shard_jobs()
    shard = _shard_job()
    shard_run = _step_with(shard, "--ci-shard-record", "run")["run"]
    recorded = re.search(r"--ci-shard-record\s+(\S+)", shard_run).group(1)
    upload = _step_with(shard, "upload-artifact", "uses")
    download = _step_with(fan_in, "download-artifact", "uses")
    assert recorded.startswith(upload["with"]["path"].rstrip("/") + "/"), (recorded, upload["with"]["path"])
    script_run = _step_with(fan_in, "PYEOF", "run")["run"]
    assert Path(recorded).name in script_run, f"the partition script never reads {Path(recorded).name}"
    matrix = shard["strategy"]["matrix"]
    rendered = re.sub(
        r"\$\{\{\s*matrix\.([\w-]+)\s*\}\}",
        lambda m: str(matrix[m.group(1)][0]),
        upload["with"]["name"],
    )
    assert fnmatch.fnmatch(rendered, download["with"]["pattern"]), (rendered, download["with"]["pattern"])
    assert re.fullmatch(r"ci-shard-\d+-of-\d+", rendered), rendered
    assert download["with"]["path"] in script_run, f"the partition script never reads {download['with']['path']}"


# ---------------------------------------------------------------------------------------------------------------
# The partition check, executed
# ---------------------------------------------------------------------------------------------------------------


def _partition_script() -> str:
    fan_in, _ = _fan_in_and_shard_jobs()
    run = _step_with(fan_in, "PYEOF", "run")["run"]
    match = re.search(r"<<'PYEOF'\n(.*?)\nPYEOF\b", run, re.S)
    assert match, "the fan-in's partition step is no longer an inline `python - <<'PYEOF'` script"
    return match.group(1)


_UNSHARDED_IDS = [f"tests/test_f{i}.py::test_{j}" for i in range(6) for j in range(3)]


def _write_unsharded(directory: Path, ids: list[str], footer: str | None = None) -> None:
    """The shape of `pytest --collect-only -q` stdout: reaper notices, one id per line, a blank, a footer."""
    footer = footer if footer is not None else f"{len(ids)} tests collected in 0.10s"
    noise = ["[run-dir reaper] kept 3 runs", "[run-dir reaper] removed 1 run"]
    (directory / "unsharded-collection.txt").write_text("\n".join([*noise, *ids, "", footer]) + "\n", encoding="utf-8")


def _write_shards(directory: Path, shards: dict[int, list[str]], count: int | None = None) -> None:
    count = count if count is not None else max(shards)
    for label, ids in shards.items():
        shard_dir = directory / "shard-artifacts" / f"ci-shard-{label}-of-{count}"
        shard_dir.mkdir(parents=True)
        (shard_dir / "nodeids.txt").write_text("".join(f"{i}\n" for i in ids), encoding="utf-8")
        (shard_dir / "wall_seconds.txt").write_text("42\n", encoding="utf-8")
        (shard_dir / "junit.xml").write_text(
            "<testsuites><testsuite>"
            '<testcase classname="c" name="test_a[x@y]" file="tests/test_heavy.py" time="9.0"/>'
            '<testcase classname="c" name="test_b@grp_big" file="tests/test_light.py" time="3.0"/>'
            '<testcase classname="c" name="test_c@grp_small" file="tests/test_light.py" time="1.0"/>'
            "</testsuite></testsuites>",
            encoding="utf-8",
        )


def _run_partition_check(tmp_path: Path, shards: dict[int, list[str]], unsharded=None, **kwargs):
    (tmp_path / "script.py").write_text(_partition_script(), encoding="utf-8")
    _write_unsharded(tmp_path, _UNSHARDED_IDS if unsharded is None else unsharded, kwargs.pop("footer", None))
    _write_shards(tmp_path, shards, kwargs.pop("count", None))
    return subprocess.run(
        [sys.executable, "-I", "script.py"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env={"PATH": os.environ["PATH"]},
    )


def _three_way_split() -> dict[int, list[str]]:
    return {1: _UNSHARDED_IDS[0:6], 2: _UNSHARDED_IDS[6:12], 3: _UNSHARDED_IDS[12:18]}


def test_partition_check_passes_a_true_partition_and_reports_the_duration_floor(tmp_path):
    """Property: shard sets that are pairwise disjoint with union equal to the unsharded selection exit 0, and the
    heaviest file and group are printed heaviest first.

    Class: NEW CAPABILITY. Plausible wrong implementation: a check that exits 1 on every input (a green-path control
    for every refusal below) or that prints no duration report.
    Input class: a three-way split of 18 ids whose unsharded footer is the `N tests collected` form.
    Why required: a check that refuses everything cannot be told from one that refuses the broken matrix; this is the
    arm that must NOT fire, and the duration report is what the shape's V3 asks the first sharded run to produce.
    What kills it: exiting 1 on a clean partition; dropping the report.
    Second correct implementation: set arithmetic in place of the Counter, different message text; only the exit
    status and the ORDER of the heaviest entries are asserted.
    """
    result = _run_partition_check(tmp_path, _three_way_split())
    assert result.returncode == 0, result.stdout + result.stderr
    out = result.stdout
    assert out.index("tests/test_heavy.py") < out.index("tests/test_light.py"), out
    assert out.index("grp_big") < out.index("grp_small"), out
    # The `@` inside the parameter id `test_a[x@y]` is not an xdist group, so `y` must not appear as a group row.
    assert not re.search(r"^\s+[\d.]+\s+y$", out, re.M), out


@pytest.mark.parametrize(
    "case",
    ["overlap", "missing", "extra", "suffixed_id", "label_gap", "footer_disagrees", "empty_shard_file"],
)
def test_partition_check_refuses_every_broken_matrix(tmp_path, case):
    """Property: for every shape of a broken matrix, the partition check exits 1.

    Class: NEW CAPABILITY. Plausible wrong implementations: comparing only totals (the sum of shard sizes equals the
    unsharded count in `overlap`+`missing` pairs that cancel); comparing as sets (misses a duplicated id);
    ignoring ids no shard recorded; accepting an xdist-suffixed id.
    Input class, one case each, differently positioned: an id in two shards while another is in none (totals agree);
    an id in no shard; an id no collection printed; the `@group` suffix on a recorded id (the coder's measured
    first-version defect); a residue label with no job while every id is still recorded once; an unsharded footer
    that disagrees with its id lines; a shard that recorded nothing.
    Why required: each is a way the matrix runs a different set of tests than the fast tier while every shard is green.
    What kills it: loosening any comparison in the script (Counter to set, dropping the `extra` arm, dropping the
    label check, dropping the footer cross-check).
    Second correct implementation: any script that exits 1 on these seven shapes and 0 on the clean split.
    """
    shards = _three_way_split()
    unsharded = None
    kwargs: dict = {}
    if case == "overlap":
        shards[1] = shards[1] + [shards[3][0]]  # an id now in two shards
        shards[2] = shards[2][1:]  # and another in none: sizes 7, 5, 6 still sum to 18
    elif case == "missing":
        shards[2] = shards[2][1:]
    elif case == "extra":
        shards[2] = shards[2] + ["tests/test_f9.py::test_never_collected"]
    elif case == "suffixed_id":
        shards[2] = [shards[2][0] + "@shared_runs_root:g", *shards[2][1:]]
    elif case == "label_gap":
        # Shard 2's ids are absorbed by the others, so every id is still recorded exactly once: only the label
        # check can see that a residue class has no job.
        shards = {1: _UNSHARDED_IDS[0:9], 3: _UNSHARDED_IDS[9:18]}
        kwargs["count"] = 3
    elif case == "footer_disagrees":
        kwargs["footer"] = "999 tests collected in 0.10s"
    elif case == "empty_shard_file":
        shards[3] = []
    result = _run_partition_check(tmp_path, shards, unsharded, **kwargs)
    assert result.returncode == 1, (case, result.returncode, result.stdout, result.stderr)


def test_partition_check_totals_agree_on_the_overlap_case_so_a_totals_only_check_would_pass_it():
    """Control for the `overlap` arm above: its shard sizes sum to the unsharded count, so a check that compares
    totals alone would accept it; the refusal in the parametrized test is therefore about identity, not size.

    Class: REGRESSION-style control of the instrument (the matrix input is built here, not read from production).
    Second correct implementation: any construction whose duplicated and dropped ids cancel in the total.
    """
    shards = _three_way_split()
    shards[1] = shards[1] + [shards[3][0]]
    shards[2] = shards[2][1:]
    assert sum(len(v) for v in shards.values()) == len(_UNSHARDED_IDS)
    assert len(set(itertools.chain.from_iterable(shards.values()))) < len(_UNSHARDED_IDS)


def test_partition_check_accepts_the_deselected_footer_form(tmp_path):
    """Property: a footer of the form `X/Y tests collected (Z deselected)` is read as X, the selected count.

    Class: NEW CAPABILITY. Plausible wrong implementation: a footer regex that reads only `N tests collected` and
    reports a disagreement on every real run, where `-m` deselects tens of thousands of tests and pytest prints the
    `selected/total` form.
    Input class: `-m`-deselected collection output.
    Second correct implementation: reading the footer's selected count by a different regex.
    """
    result = _run_partition_check(
        tmp_path,
        _three_way_split(),
        footer=f"{len(_UNSHARDED_IDS)}/{len(_UNSHARDED_IDS) + 40} tests collected (40 deselected) in 0.10s",
    )
    assert result.returncode == 0, result.stdout + result.stderr


# ---------------------------------------------------------------------------------------------------------------
# The shard option, executed over a toy tree in subprocess mode
# ---------------------------------------------------------------------------------------------------------------

_FILE_COUNT = 24
_SHARD_COUNT = 3

_TOY_FILE = textwrap.dedent(
    """
    import pytest


    def test_plain():
        pass


    @pytest.mark.parametrize("value", [1, 2])
    def test_param(value):
        pass


    @pytest.mark.xdist_group("toy_group")
    def test_grouped():
        pass
    """
)

# The toy conftest takes the REAL hooks from tests/conftest.py, so what runs here is the production option.
_TOY_CONFTEST = "from tests.conftest import pytest_addoption, pytest_configure  # noqa: F401\n"

_NODE_ID = re.compile(r"^\S+\.py::.*$")


@pytest.fixture
def toy(pytester, monkeypatch):
    """A toy tree of 24 files (residue split 4, 7, 13 for N = 3, so no shard is empty) under the production hooks."""
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(_REPO), str(_REPO / "src")]))
    monkeypatch.setenv("HHEMT_FORBID_COMPILE", "1")
    pytester.makeconftest(_TOY_CONFTEST)
    for index in range(_FILE_COUNT):
        pytester.makepyfile(**{f"test_toy_{index}": _TOY_FILE})
    return pytester


def _collect(toy, monkeypatch, *args, seed: str = "0") -> list[str]:
    """Node ids printed by `--collect-only -q` under PYTHONHASHSEED=seed."""
    monkeypatch.setenv("PYTHONHASHSEED", seed)
    result = toy.runpytest_subprocess("--collect-only", "-q", "-p", "no:cacheprovider", *args)
    assert result.ret == 0, "\n".join(result.outlines + result.errlines)
    return [line for line in result.outlines if _NODE_ID.match(line)]


def _recorded(toy, monkeypatch, shard: str, seed: str, *extra: str) -> list[str]:
    path = toy.path / f"record_{shard.replace('/', '_')}_{seed}.txt"
    monkeypatch.setenv("PYTHONHASHSEED", seed)
    result = toy.runpytest_subprocess(
        "--collect-only", "-q", "-p", "no:cacheprovider", "--ci-shard", shard, f"--ci-shard-record={path}", *extra
    )
    assert result.ret == 0, "\n".join(result.outlines + result.errlines)
    return path.read_text(encoding="utf-8").splitlines()


def test_every_shard_selection_partitions_the_unsharded_collection_under_different_hash_seeds(toy, monkeypatch):
    """Property: running `--ci-shard K/N` for K = 1..N, EACH under a different PYTHONHASHSEED, yields pairwise-disjoint
    selections whose union is the unsharded collection, every file lying wholly in one shard, no shard empty.

    Class: NEW CAPABILITY. Plausible wrong implementation: the built-in `hash(file)` for the key. It is stable inside
    one process and differs between processes, so shards started under different seeds disagree about which files are
    theirs and the partition fails (red under it: seeds 0, 1, 2 give three unrelated residue maps).
    Input class: a toy tree of 24 files x 4 test ids (plain, two parametrized, one xdist_group), N = 3.
    Why required: the shards run as separate processes on separate runners; a key that is not process-independent
    puts some file in no shard, and every shard is green.
    What kills it: replacing `hashlib.sha256` in `_ci_shard_residue` by `hash()`; or `K` instead of `K % N` as the
    residue (K = N then selects nothing).
    Second correct implementation: `zlib.crc32` or `hashlib.md5` of the path, or LPT assignment by measured duration
    held as a deterministic table; every one is a partition independent of the process, and none is read here.
    The hash seeds are three distinct values, so a mutated key would have to land every file in exactly one shard by
    chance three independent times.
    """
    unsharded = _collect(toy, monkeypatch)
    assert len(unsharded) == _FILE_COUNT * 4, unsharded
    seeds = {1: "11", 2: "222", 3: "3333"}
    shards = {k: _recorded(toy, monkeypatch, f"{k}/{_SHARD_COUNT}", seeds[k]) for k in range(1, _SHARD_COUNT + 1)}
    flat = [i for ids in shards.values() for i in ids]
    assert len(flat) == len(set(flat)), "a node id is in more than one shard"
    assert sorted(flat) == sorted(unsharded), "the union of the shards is not the unsharded collection"
    # Must-fire control for the partition equality: a shard selection that is the whole tree would also satisfy a
    # union check, so the property that varies here is that the work is actually divided.
    assert all(shards.values()), {k: len(v) for k, v in shards.items()}
    assert all(len(ids) < len(unsharded) for ids in shards.values())
    files_by_shard = {k: {i.partition("::")[0] for i in ids} for k, ids in shards.items()}
    for first, second in itertools.combinations(files_by_shard, 2):
        assert not files_by_shard[first] & files_by_shard[second], f"a file is split across shards {first} and {second}"


def test_a_shard_selection_does_not_depend_on_the_hash_seed(toy, monkeypatch):
    """Property: one `--ci-shard K/N` selects the same ids under two different PYTHONHASHSEED values.

    Class: NEW CAPABILITY. Plausible wrong implementation: built-in `hash()` (red under it: the two runs select
    different files). This is the same property as the partition test seen from one shard, so a failure here
    localises the cause to the key.
    Input class: K = 2 of 3, seeds 5 and 6.
    Second correct implementation: any process-independent digest.
    """
    first = _recorded(toy, monkeypatch, f"2/{_SHARD_COUNT}", "5")
    second = _recorded(toy, monkeypatch, f"2/{_SHARD_COUNT}", "6")
    assert first and first == second


def test_one_shard_of_one_selects_everything_and_the_option_alone_records_without_deselecting(toy, monkeypatch):
    """Property: `--ci-shard 1/1` selects the whole collection (K = N owns residue 0 and N = 1 has one residue), and
    `--ci-shard-record` without `--ci-shard` records the whole collection and deselects nothing.

    Class: NEW CAPABILITY. Plausible wrong implementation: residue `K` instead of `K % N`, which selects nothing for
    `1/1` (residue 1 of a modulus 1 never occurs) and for `N/N` in general.
    Input class: the two degenerate configurations, differently positioned from the 3-way split.
    Why required: the label scheme is 1..N, so `N/N` is the label the off-by-one drops; the fan-in's check would
    report its files missing, but only after a full run.
    Second correct implementation: 0-based labels would change the CLI contract and are not second-correct here.
    """
    unsharded = _collect(toy, monkeypatch)
    assert sorted(_recorded(toy, monkeypatch, "1/1", "0")) == sorted(unsharded)
    path = toy.path / "record_only.txt"
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    result = toy.runpytest_subprocess("--collect-only", "-q", "-p", "no:cacheprovider", f"--ci-shard-record={path}")
    assert result.ret == 0
    assert sorted(path.read_text(encoding="utf-8").splitlines()) == sorted(unsharded)


@pytest.mark.parametrize("value", ["0/3", "4/3", "a/b", "3", "1/0", "1/2/3", "-1/3", ""])
def test_a_malformed_shard_value_is_a_usage_error(toy, monkeypatch, value):
    """Property: every `--ci-shard` value outside `K/N` with integers and 1 <= K <= N exits with pytest's usage code 4
    and selects and runs nothing.

    Class: NEW CAPABILITY. Plausible wrong implementation: a parser that accepts `0/3` or `4/3` (residue arithmetic
    then wraps the value into a valid shard and the run silently covers the wrong files), or raises a raw ValueError
    (exit 1, indistinguishable from a test failure).
    Input class: out-of-range, non-numeric, missing-slash, zero-denominator, extra-slash, negative and empty values.
    Why required: the workflow passes the value from the matrix; a typo there must fail the shard, not skew it.
    What kills it: relaxing the range check or the digit check in `_parse_ci_shard`.
    Second correct implementation: argparse `choices`, or a click-style callback; the exit code is the contract.
    """
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    result = toy.runpytest_subprocess("--collect-only", "-q", "-p", "no:cacheprovider", f"--ci-shard={value}")
    assert result.ret == 4, (value, result.ret, result.errlines)


def test_under_xdist_loadgroup_the_recorded_ids_are_the_bare_ids_the_unsharded_collection_prints(toy, monkeypatch):
    """Property: with `-n 2 --dist loadgroup` and an `xdist_group` marker, the recorded ids of a shard equal that
    shard's ids from a non-xdist collection, none carrying an `@group` suffix.

    Class: REGRESSION against the coder's measured first version, which keyed a dict by the item and recorded
    `test_f5.py::test_a@grp1` (pytest-xdist rewrites the node id after the hook that recorded it). Plausible wrong
    implementation: reading `item.nodeid` in `pytest_collection_finish` without capturing it earlier; red under it.
    Input class: a tree with grouped tests run under real xdist workers (differently positioned from the
    single-process partition test: the id is rewritten by a second plugin between the two hooks).
    Why required: the fan-in compares recorded ids with `pytest --collect-only -q` output, which has no suffix; a
    suffix makes every grouped test missing and extra at once.
    What kills it: removing the `tryfirst` marker or the `item.stash` capture in `_CiShardPlugin`.
    Second correct implementation: recording in a `tryfirst` `pytest_collection_modifyitems` instead of
    `pytest_collection_finish`; only the recorded file is read. Under pytest 9.2 xdist no longer writes a suffix at
    all (see the knowledge doc on the 9.2.0 node id), and this test passes there for the stronger reason.
    """
    pytest.importorskip("xdist")
    expected = sorted(_recorded(toy, monkeypatch, f"2/{_SHARD_COUNT}", "0"))
    assert expected
    path = toy.path / "record_xdist.txt"
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    result = toy.runpytest_subprocess(
        "-p",
        "no:cacheprovider",
        "-n",
        "2",
        "--dist",
        "loadgroup",
        "--ci-shard",
        f"2/{_SHARD_COUNT}",
        f"--ci-shard-record={path}",
    )
    assert result.ret == 0, "\n".join(result.outlines + result.errlines)
    recorded = path.read_text(encoding="utf-8").splitlines()
    assert not [i for i in recorded if "@" in i], [i for i in recorded if "@" in i]
    assert sorted(recorded) == expected
