"""Tests for scripts/check_ruleset_sync.py, the ruleset sync guard.

The guard compares `.github/rulesets/main.json` with the workflow jobs that must produce its
required checks (static half) and with the ruleset GitHub enforces (live half). Static tests drive
`main(["--repo-root", tmp])` over a throwaway tree. Live tests replace `_gh_json`, the one function
that shells out, so no network and no `gh` are used. Two tests read the real repository files.

Each test carries its CLASS (REGRESSION: red on the tree before this chunk; NEW CAPABILITY: the
module did not exist, so a plausible wrong implementation is named instead) and a second correct
implementation under which it still passes.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

import scripts.check_ruleset_sync as guard  # repo root is on sys.path under pytest

_REPO = Path(__file__).resolve().parent.parent


def _ruleset(contexts: list[str], **pull_request_extra: Any) -> dict[str, Any]:
    return {
        "name": "main-branch-protection",
        "target": "branch",
        "enforcement": "active",
        "bypass_actors": [],
        "conditions": {"ref_name": {"include": ["refs/heads/main"], "exclude": []}},
        "rules": [
            {
                "type": "pull_request",
                "parameters": {"allowed_merge_methods": ["squash", "rebase"], **pull_request_extra},
            },
            {"type": "required_linear_history"},
            {
                "type": "required_status_checks",
                "parameters": {"required_status_checks": [{"context": c} for c in contexts]},
            },
        ],
    }


def _tree(tmp_path: Path, contexts: list[str], workflows: dict[str, str]) -> Path:
    (tmp_path / ".github" / "rulesets").mkdir(parents=True)
    (tmp_path / ".github" / "workflows").mkdir()
    (tmp_path / ".github" / "rulesets" / "main.json").write_text(json.dumps(_ruleset(contexts)), encoding="utf-8")
    for name, text in workflows.items():
        (tmp_path / ".github" / "workflows" / name).write_text(text, encoding="utf-8")
    return tmp_path


def _static(root: Path) -> int:
    return guard.main(["--static", "--repo-root", str(root)])


# --------------------------------------------------------------------------------------
# static half
# --------------------------------------------------------------------------------------

_PLAIN_JOB = "name: wf\njobs:\n  docs-build:\n    runs-on: ubuntu-latest\n"
_MATRIX_JOB = (
    "name: wf\njobs:\n  build:\n    strategy:\n      matrix:\n"
    "        os: [ubuntu-latest]\n        python-version: ['3.12']\n    runs-on: ${{ matrix.os }}\n"
)


def test_plain_job_id_is_a_check_name(tmp_path: Path) -> None:
    """Property: a job with no `name:` and no matrix produces a check named by its job id.

    Class: NEW CAPABILITY. Plausible wrong implementation: keying on the WORKFLOW's `name:` (here `wf`)
    instead of the job -- red under it, because `docs-build` would then be reported missing.
    Why required: GitHub reports check runs by job name; matching the workflow name would pass a
    ruleset whose checks never appear. Second correct implementation: derive names by running
    `act --list` over the workflow instead of parsing it.
    """
    root = _tree(tmp_path, ["docs-build"], {"docs.yml": _PLAIN_JOB})
    assert _static(root) == 0


def test_matrix_job_is_named_with_its_value_suffix(tmp_path: Path) -> None:
    """Property: a matrix job's checks are `{id} ({v1}, {v2})` in matrix-key order.

    Class: NEW CAPABILITY. Wrong implementation: ignoring the matrix and emitting the bare job id --
    red under it, because the repository's real required context `build (ubuntu-latest, 3.12)` would
    then be reported missing. Why required: that context is the one the release PR is blocked on.
    Second correct implementation: expand the matrix with a different cartesian-product routine.
    """
    root = _tree(tmp_path, ["build (ubuntu-latest, 3.12)"], {"test.yml": _MATRIX_JOB})
    assert _static(root) == 0


def test_bare_id_does_not_satisfy_a_matrix_context(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Property: requiring the bare id `build` fails when `build` only exists as a matrix job.

    Class: NEW CAPABILITY, the violating arm paired with the satisfying arm above. Wrong
    implementation: prefix matching (`startswith`) -- red under it, because `build` would match
    `build (ubuntu-latest, 3.12)`. Why required: a required context `build` would hang the PR.
    Second correct implementation: a set lookup over fully rendered names.
    """
    root = _tree(tmp_path, ["build"], {"test.yml": _MATRIX_JOB})
    assert _static(root) == 1
    assert "produced by no job" in capsys.readouterr().err


def test_fan_in_job_with_literal_name_satisfies_a_context_its_shards_do_not(tmp_path: Path) -> None:
    """Property: a non-matrix job whose `name:` is the literal context produces it; shard jobs do not.

    Class: NEW CAPABILITY. Wrong implementation: appending a matrix-style suffix to every `name:`
    -- red under it, because the fan-in `build (ubuntu-latest, 3.12)` would become
    `build (ubuntu-latest, 3.12) ()`. Why required: a fan-in job with a literal name is a shape
    the ruleset must keep matching unchanged. Second correct implementation: treat a
    literal `name:` as authoritative whenever the job has no `strategy.matrix`.
    """
    workflow = (
        "name: wf\njobs:\n  shard:\n    strategy:\n      matrix:\n        shard: [1, 2]\n    runs-on: ubuntu-latest\n"
        "  fan-in:\n    name: build (ubuntu-latest, 3.12)\n    needs: [shard]\n    runs-on: ubuntu-latest\n"
    )
    root = _tree(tmp_path, ["build (ubuntu-latest, 3.12)"], {"test.yml": workflow})
    assert _static(root) == 0
    # The shard jobs alone (fan-in removed) must NOT satisfy it: their names are `shard (1)` and `shard (2)`.
    root2 = _tree(tmp_path / "b", ["build (ubuntu-latest, 3.12)"], {"test.yml": workflow.split("  fan-in:")[0]})
    assert _static(root2) == 1


def test_matrix_expression_in_name_is_substituted(tmp_path: Path) -> None:
    """Property: a `name:` template referencing `${{ matrix.KEY }}` renders per combination, with no suffix.

    Class: NEW CAPABILITY. Wrong implementation: appending the suffix on top of the template --
    red under it, because the context `lint-3.12` would instead be `lint-3.12 (3.12)`.
    Why required: GitHub does not suffix a name that already names the matrix.
    Second correct implementation: a single regex substitution pass over the template.
    """
    workflow = (
        "name: wf\njobs:\n  lint:\n    name: lint-${{ matrix.py }}\n    strategy:\n      matrix:\n"
        "        py: ['3.11', '3.12']\n    runs-on: ubuntu-latest\n"
    )
    root = _tree(tmp_path, ["lint-3.12", "lint-3.11"], {"lint.yml": workflow})
    assert _static(root) == 0


def test_a_job_that_cannot_be_derived_fails_loudly_rather_than_passing(tmp_path: Path) -> None:
    """Property: an expression-valued matrix produces no derivable name, so its context is reported missing.

    Class: NEW CAPABILITY. Wrong implementation: treating an un-derivable job as a wildcard that
    satisfies any context -- red under it, because the run would exit 0. Why required: a guard that
    passes what it could not read is the silent-pass defect this repository has measured repeatedly.
    Second correct implementation: raise an environment error (exit 2) for such a workflow instead.
    """
    workflow = (
        "name: wf\njobs:\n  dyn:\n    strategy:\n      matrix: ${{ fromJson(needs.x.outputs.m) }}\n"
        "    runs-on: ubuntu-latest\n"
    )
    root = _tree(tmp_path, ["dyn (1)"], {"d.yml": workflow})
    assert _static(root) in (1, 2)


def test_ruleset_without_required_checks_is_a_finding(tmp_path: Path) -> None:
    """Property: a file with no required_status_checks rule is reported, not vacuously accepted.

    Class: NEW CAPABILITY. Wrong implementation: iterating an empty context list and exiting 0 --
    red under it. Why required: an empty gate set is the state a bad edit to main.json produces.
    Second correct implementation: exit 2 (environment error) with the same message.
    """
    root = _tree(tmp_path, [], {"docs.yml": _PLAIN_JOB})
    ruleset = _ruleset([])
    ruleset["rules"] = [r for r in ruleset["rules"] if r["type"] != "required_status_checks"]
    (root / ".github" / "rulesets" / "main.json").write_text(json.dumps(ruleset), encoding="utf-8")
    assert _static(root) in (1, 2)


def test_no_half_selected_is_a_usage_error() -> None:
    """Property: running with neither --static nor --live exits via argparse (2), never silently OK.

    Class: NEW CAPABILITY. Wrong implementation: defaulting to a no-op that returns 0 -- red under it.
    Why required: `just ruleset-check` spelling a half wrong must not look like a passing check.
    Second correct implementation: default to running the static half and say so.
    """
    with pytest.raises(SystemExit) as exc:
        guard.main([])
    assert exc.value.code == 2


# --------------------------------------------------------------------------------------
# live half (the only shell-out, `_gh_json`, is replaced)
# --------------------------------------------------------------------------------------


def _fake_gh(monkeypatch: pytest.MonkeyPatch, live: dict[str, Any], listing_extra: list[dict[str, Any]] | None = None):
    def fake(path: str) -> Any:
        if path.endswith("/rulesets"):
            return [{"id": 7, "name": live["name"]}, *(listing_extra or [])]
        assert path.endswith("/rulesets/7"), path
        return live

    monkeypatch.setattr(guard, "_gh_json", fake)


def _live_tree(tmp_path: Path, file_ruleset: dict[str, Any]) -> Path:
    (tmp_path / ".github" / "rulesets").mkdir(parents=True)
    (tmp_path / ".github" / "rulesets" / "main.json").write_text(json.dumps(file_ruleset), encoding="utf-8")
    return tmp_path


_SERVER = {
    "id": 7,
    "source_type": "Repository",
    "source": "o/r",
    "node_id": "x",
    "created_at": "t",
    "updated_at": "t",
    "current_user_can_bypass": "never",
    "_links": {},
}


def test_live_equal_after_normalisation_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Property: server-assigned fields and list ordering do not count as drift.

    Class: NEW CAPABILITY. Wrong implementation: comparing raw JSON -- red under it, because the live
    payload carries `id`, `_links`, timestamps, and orders rules and checks differently.
    Why required: a guard that fires on every run is turned off. Second correct implementation:
    compare after projecting both sides onto the file's own keys.
    """
    file_rs = _ruleset(["a", "b"], required_reviewers=[])
    live = json.loads(json.dumps(file_rs)) | _SERVER
    live["rules"].reverse()
    for rule in live["rules"]:
        if rule["type"] == "required_status_checks":
            rule["parameters"]["required_status_checks"].reverse()
    _fake_gh(monkeypatch, live)
    assert guard.main(["--live", "--repo-root", str(_live_tree(tmp_path, file_rs))]) == 0


def test_live_key_missing_from_the_file_is_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Property: a key set only on the live ruleset is reported, naming the key.

    Class: REGRESSION on the data this chunk repairs: the live ruleset carried
    `require_extra_approval_for_unattributed_changes: true` that the file lacked, and the guard
    must fire on exactly that. Wrong implementation: projecting live onto the file's keys before
    comparing -- red under it, because the extra key would vanish and the run would exit 0.
    Second correct implementation: compare sorted key paths first, then values.
    """
    file_rs = _ruleset(["a"])
    live = _ruleset(["a"], require_extra_approval_for_unattributed_changes=True) | _SERVER
    _fake_gh(monkeypatch, live)
    rc = guard.main(["--live", "--repo-root", str(_live_tree(tmp_path, file_rs))])
    err = capsys.readouterr().err
    assert rc == 1
    assert "require_extra_approval_for_unattributed_changes" in err


def test_live_value_difference_is_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Property: a changed value (not only an added key) is drift.

    Class: NEW CAPABILITY. Wrong implementation: comparing key sets only -- red under it, because
    the two required-check lists have the same keys. Why required: the release-blocking drift is
    usually a context list edited in the web UI. Second correct implementation: a deep-equal.
    """
    file_rs = _ruleset(["a", "b"])
    live = _ruleset(["a", "c"]) | _SERVER
    _fake_gh(monkeypatch, live)
    assert guard.main(["--live", "--repo-root", str(_live_tree(tmp_path, file_rs))]) == 1


def test_gh_failure_is_an_environment_error_not_a_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Property: when the live ruleset cannot be read the exit is 2 and says why.

    Class: NEW CAPABILITY. Wrong implementation: catching the failure and returning 0 ("nothing to
    compare") -- red under it. Why required: `just tag` would then tag against an unverified
    ruleset. Second correct implementation: exit 1 with the same message.
    """

    def boom(path: str) -> Any:
        raise guard.EnvironmentProblem("`gh` is not on PATH")

    monkeypatch.setattr(guard, "_gh_json", boom)
    rc = guard.main(["--live", "--repo-root", str(_live_tree(tmp_path, _ruleset(["a"])))])
    assert rc == 2
    assert "gh" in capsys.readouterr().err


def test_ruleset_absent_live_is_an_environment_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Property: no live ruleset with the file's name is exit 2, and two with that name is exit 2.

    Class: NEW CAPABILITY. Wrong implementation: comparing against the first listing entry whatever
    its name -- red under it, because an unrelated ruleset would then be compared. Why required:
    the comparison is only meaningful against the same-named ruleset. Second correct implementation:
    take the id from the repository ruleset the apply script created, read from a state file.
    """
    live = _ruleset(["a"]) | _SERVER | {"name": "some-other"}

    def fake(path: str) -> Any:
        return [{"id": 9, "name": "some-other"}] if path.endswith("/rulesets") else live

    monkeypatch.setattr(guard, "_gh_json", fake)
    assert guard.main(["--live", "--repo-root", str(_live_tree(tmp_path, _ruleset(["a"])))]) == 2


# --------------------------------------------------------------------------------------
# the real repository files
# --------------------------------------------------------------------------------------


def test_real_ruleset_file_is_statically_satisfiable() -> None:
    """Property: every required context in the committed main.json is produced by a real workflow job.

    Class: REGRESSION target: a required context added to main.json with
    no producing job reds this test. Wrong implementation: matching only against `.yml` files --
    red under it if a future workflow is `.yaml`. Why required: this is the release gate's wiring.
    Second correct implementation: query the GitHub API for check-run names of a recent run.
    """
    assert guard.main(["--static", "--repo-root", str(_REPO)]) == 0


def test_real_ruleset_file_carries_the_two_live_keys() -> None:
    """Property: the file states `required_reviewers: []` and `require_extra_approval_for_unattributed_changes: true`.

    Class: REGRESSION, red on the tree before this chunk (the file lacked both keys). Wrong
    implementation: adding the keys with other values -- red under it. Why required: the live
    ruleset carries them, and an `apply_main_ruleset.sh` run from a file without them would
    reset them. Second correct implementation: a spelling-independent read of the JSON.
    """
    ruleset = json.loads((_REPO / ".github" / "rulesets" / "main.json").read_text(encoding="utf-8"))
    params = next(r["parameters"] for r in ruleset["rules"] if r["type"] == "pull_request")
    assert params["required_reviewers"] == []
    assert params["require_extra_approval_for_unattributed_changes"] is True


def test_just_tag_runs_the_ruleset_check_before_it_tags() -> None:
    """Property: the `tag` recipe invokes `ruleset-check` on a line before the `git tag` line.

    Class: REGRESSION, red on the tree before this chunk (no such recipe, no such call). Wrong
    implementation: calling the check AFTER the tag -- red under it, because a drifted ruleset would
    then be reported after the release is frozen. Why required: shape F1 places the live comparison
    at tag time. Second correct implementation: a `tag` dependency (`tag: ruleset-check`).
    """
    text = (_REPO / "justfile").read_text(encoding="utf-8")
    assert re.search(r"(?m)^ruleset-check:\n\s+python scripts/check_ruleset_sync\.py --static --live\s*$", text)
    tag_body = text.split("\ntag:\n", 1)[1].split("\n\n", 1)[0].splitlines()
    check_line = next(i for i, line in enumerate(tag_body) if "ruleset-check" in line)
    tag_line = next(i for i, line in enumerate(tag_body) if "git tag" in line)
    assert check_line < tag_line
