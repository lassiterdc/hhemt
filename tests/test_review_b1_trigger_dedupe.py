"""Reviewer tests for work chunk 1b.2: each heavy required check runs once per SHA, and the
release PR proves that the tree that lands is the tree that was tested.

PROBLEM. A develop -> main release PR used to run every required workflow twice for one head
SHA (once on `pull_request` into main, once on the develop `push`), and the merge result was
never compared with the tested head. Chunk 1b.2 narrows the heavy workflows to develop and adds
a PR-only job, `release-pr-tree-identity`, that must fail whenever the merge ref's tree differs
from the PR head's tree. `scripts/check_ruleset_sync.py --static` only proves that a producing
job EXISTS for each required context; its docstring says it does not read triggers, so a
workflow retargeted at main (or path-filtered so no push run exists for the head SHA) still
passes it. These tests hold the two properties that gap leaves open. They read
`.github/` and run `git` in throwaway repositories; no solver runs.

Each test names its CLASS, the approved-shape row it enforces is named in the reviewer's round
file, and each states the change that kills it and a second correct implementation under which
it still passes.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
RULESET = REPO_ROOT / ".github" / "rulesets" / "main.json"
TREE_IDENTITY = "release-pr-tree-identity"

_GIT_ENV = {
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@t",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@t",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_SYSTEM": os.devnull,
}


def _required_contexts() -> list[str]:
    """The `required_status_checks` contexts of the committed main ruleset, in file order."""
    ruleset = json.loads(RULESET.read_text())
    return [
        c["context"]
        for rule in ruleset["rules"]
        if rule["type"] == "required_status_checks"
        for c in rule["parameters"]["required_status_checks"]
    ]


def _workflow_docs() -> dict[str, dict]:
    """Every workflow file under `.github/workflows/`, parsed, keyed by file name."""
    return {p.name: yaml.safe_load(p.read_text()) for p in sorted(WORKFLOWS.glob("*.yml"))}


def _triggers(doc: dict) -> dict:
    """The `on:` mapping of a workflow. PyYAML reads the bare key `on` as the boolean True."""
    on = doc["on"] if "on" in doc else doc[True]
    return on if isinstance(on, dict) else {name: None for name in ([on] if isinstance(on, str) else on)}


def _producers(context: str, docs: dict[str, dict]) -> list[str]:
    """Workflow files with a job whose check-run name is `context` (job id, or `id (matrix...)`)."""
    found = []
    for fname, doc in docs.items():
        for job_id, job in doc.get("jobs", {}).items():
            name = str(job.get("name", job_id))
            if context == name or context.startswith(name + " ("):
                found.append(fname)
    return found


def _branches(trigger: dict | None) -> list[str]:
    """The `branches:` list of one `push` or `pull_request` trigger; empty when unfiltered."""
    return list((trigger or {}).get("branches", []))


def test_required_check_producers_run_on_develop_push_and_never_on_main():
    """Property: each workflow producing a required context, other than the tree-identity job,
    fires on a push to develop, has no path filter on that push, and names main in neither its
    push nor its pull_request branch filter.

    Class: REGRESSION -- red on the pre-change tree (eight producers named main), output pasted
    in the reviewer's round file. Input class: any workflow file that produces a context listed
    in `.github/rulesets/main.json`, including a file added after this chunk.

    Why required: the release PR reads its required contexts from the develop push run on the
    head SHA. A producer that does not run on that push leaves the context at "Expected --
    Waiting for status to be reported" with nothing red; a producer that also runs on main
    brings back the second run per SHA. `check_ruleset_sync.py --static` reads neither.

    What kills it: retargeting any producer's `push.branches` away from develop, re-adding
    "main" to a producer's `push` or `pull_request` filter, or adding `paths:` to a producer's
    `push` trigger.

    Second correct implementation: a producer that triggers on `push` to a develop pattern such
    as `["develop", "release/**"]`, or one that moves its filter into a reusable workflow's
    caller, still passes -- the test asserts the trigger's effect, not its spelling.
    """
    docs = _workflow_docs()
    contexts = [c for c in _required_contexts() if c != TREE_IDENTITY]
    assert contexts, "the ruleset lists no heavy required context; the matcher read the wrong file"
    problems = []
    for context in contexts:
        producers = _producers(context, docs)
        if not producers:
            problems.append(f"{context}: produced by no workflow")
        for fname in producers:
            triggers = _triggers(docs[fname])
            push = triggers.get("push")
            if "push" not in triggers or "develop" not in _branches(push):
                problems.append(f"{context} ({fname}): does not trigger on push to develop")
            if push and ("paths" in push or "paths-ignore" in push):
                problems.append(f"{context} ({fname}): push is path-filtered, so some develop SHAs have no run")
            for event in ("push", "pull_request"):
                if "main" in _branches(triggers.get(event)):
                    problems.append(f"{context} ({fname}): {event} names main")
    assert not problems, "\n".join(problems)


def test_tree_identity_workflow_is_pr_into_main_only_and_its_job_key_is_the_context():
    """Property: the workflow producing `release-pr-tree-identity` triggers on pull_request into
    main and on no other event, and its job carries no job-level `name:` that would rename the
    check-run away from the context string the ruleset requires.

    Class: NEW CAPABILITY -- plausible wrong implementation is a tree-identity workflow that
    also triggers on push to main, or names its job differently from the context; red under it.

    Why required: a push trigger on main would run after the merge has already landed and
    certify nothing; a renamed job leaves the required context unreported on every release PR.

    What kills it: adding a `push:` trigger, dropping the `branches: [main]` filter, or adding
    `name: Tree identity` to the job.

    Second correct implementation: a job id equal to the context with the comparison moved into
    a composite action passes; so does adding `workflow_dispatch`, which is not read here.
    """
    docs = _workflow_docs()
    producers = _producers(TREE_IDENTITY, docs)
    assert len(producers) == 1, f"expected exactly one producer of {TREE_IDENTITY}, found {producers}"
    doc = docs[producers[0]]
    triggers = _triggers(doc)
    assert set(triggers) - {"workflow_dispatch"} == {"pull_request"}, f"unexpected events: {sorted(triggers)}"
    assert _branches(triggers["pull_request"]) == ["main"]
    assert "name" not in doc["jobs"][TREE_IDENTITY]


def _tree_identity_script() -> str:
    """The shell text of the step that compares the merge ref's tree with the PR head's tree."""
    producers = _producers(TREE_IDENTITY, _workflow_docs())
    steps = _workflow_docs()[producers[0]]["jobs"][TREE_IDENTITY]["steps"]
    runs = [s["run"] for s in steps if "run" in s]
    assert len(runs) == 1, f"expected exactly one run step in the tree-identity job, found {len(runs)}"
    return runs[0]


def _git(repo: Path, *args: str) -> str:
    """Run git in `repo` with a fixed identity and no user or system config; return stdout."""
    done = subprocess.run(
        ["git", *args], cwd=repo, env={**os.environ, **_GIT_ENV}, check=True, capture_output=True, text=True
    )
    return done.stdout.strip()


def _commit_file(repo: Path, name: str, text: str, message: str) -> None:
    """Write `text` to `name`, stage it and commit."""
    (repo / name).write_text(text)
    _git(repo, "add", name)
    _git(repo, "commit", "-q", "-m", message)


def _merge_ref_checkout(tmp_path: Path, main_side: list[tuple[str, str]], develop_side: list[tuple[str, str]]):
    """Build a repo, then return (repo, develop_sha) with HEAD on a no-ff merge of develop into main.

    The two sides start from one shared base commit; each side then commits its `(name, text)`
    pairs. HEAD is left on the merge commit, which is what `actions/checkout` produces for a
    `pull_request` event (`refs/pull/N/merge`).
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _commit_file(repo, "shared.txt", "line1\nline2\nline3\nline4\nline5\n", "base")
    _git(repo, "checkout", "-q", "-b", "develop")
    for name, text in develop_side:
        _commit_file(repo, name, text, f"develop {name}")
    develop_sha = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "main")
    for name, text in main_side:
        _commit_file(repo, name, text, f"main {name}")
    _git(repo, "checkout", "-q", "-b", "mergeref")
    _git(repo, "merge", "-q", "--no-ff", "-m", "merge ref", "develop")
    return repo, develop_sha


def _run_tree_identity(repo: Path, develop_sha: str) -> subprocess.CompletedProcess:
    """Execute the workflow's own comparison step in `repo` with PR_HEAD_SHA set."""
    return subprocess.run(
        ["bash", "-c", _tree_identity_script()],
        cwd=repo,
        env={**os.environ, **_GIT_ENV, "PR_HEAD_SHA": develop_sha},
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    ("label", "main_side", "develop_side", "expect_rc"),
    [
        ("main is an ancestor of develop", [], [("d.txt", "d\n")], 0),
        ("main and develop each add the identical file by separate commits", [("x.txt", "x\n")], [("x.txt", "x\n")], 0),
        ("main adds a file develop lacks", [("hotfix.txt", "h\n")], [("d.txt", "d\n")], 1),
        (
            "main edits a shared file in a region develop did not touch",
            [("shared.txt", "LINE1\nline2\nline3\nline4\nline5\n")],
            [("shared.txt", "line1\nline2\nline3\nline4\nLINE5\n")],
            1,
        ),
    ],
)
def test_tree_identity_step_verdict_follows_tree_equality_not_ancestry(
    tmp_path, label, main_side, develop_side, expect_rc
):
    """Property: the tree-identity step exits 0 exactly when the merge ref's tree equals the PR
    head's tree, whatever the commit history that produced it.

    Class: NEW CAPABILITY -- plausible wrong implementation is a step that compares against the
    merge ref's FIRST parent (main) instead of the PR head, or one that tests ancestry of main
    in develop; red under either. Input class: a merge commit built from any pair of branch
    histories from one base, including one where main's change reaches develop by a different
    commit (equal trees, no ancestry).

    Why required: the job exists so the develop push run, which certified the head commit,
    certifies what lands. The two rows that exit 1 are the case it must refuse; the two rows
    that exit 0 are the cases it must not refuse, and the identical-file row is the one an
    ancestry test mislabels. The rows ran against the step text in the committed workflow, not a
    copy, so an edit to the step is read here.

    What kills it: changing `HEAD^{tree}` to `HEAD^1^{tree}`, swapping `!=` for `==`, or
    dropping `exit 1` after the error line.

    Second correct implementation: `git diff --quiet "$PR_HEAD_SHA" HEAD` as the test, exiting
    non-zero on a difference, passes all four rows.
    """
    repo, develop_sha = _merge_ref_checkout(tmp_path, main_side, develop_side)
    done = _run_tree_identity(repo, develop_sha)
    assert done.returncode == expect_rc, f"{label}: rc={done.returncode}\n{done.stdout}\n{done.stderr}"
    if expect_rc == 1:
        assert "::error::" in done.stdout
