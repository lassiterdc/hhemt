#!/usr/bin/env python3
"""Ruleset sync guard: `.github/rulesets/main.json` must be satisfiable and must equal the live ruleset.

WHAT PROBLEM THIS SOLVES. The branch-protection ruleset lives in two places that nothing
compared: the committed file `.github/rulesets/main.json` (applied by
`scripts/apply_main_ruleset.sh`) and the ruleset GitHub actually enforces. Two failures
follow from that, and they fail differently:

  * A required status check that no workflow job produces never reports, so the release
    PR sits at "Expected -- Waiting for status to be reported" with nothing red to read.
    The STATIC half catches this before the ruleset is applied.
  * The live ruleset drifts from the file (a key set in the web UI, a key a newer API
    version adds). A later `apply_main_ruleset.sh` then silently reverts or re-asserts it.
    Measured at the time this guard was written: the live ruleset carried
    `required_reviewers: []` and `require_extra_approval_for_unattributed_changes: true`,
    which the file lacked. The LIVE half catches this.

TWO HALVES, selected by flag (at least one is required):

  --static   Every `required_status_checks[].context` in the file is the check-run name of
             some job in `.github/workflows/*.yml`. A job's check-run name is its `name:`
             (else its job id); a job with a `strategy.matrix` yields one check per matrix
             combination, named `{name} ({value1}, {value2}, ...)` with the matrix values in
             key order, or the `name:` template with `${{ matrix.KEY }}` substituted when the
             template references the matrix. Pure file reading; runs in CI
             (`.github/workflows/ruleset-sync.yml`).
  --live     The normalised file equals `gh api repos/{owner}/{repo}/rulesets/{id}`, the id
             found by listing the repository's rulesets and selecting the file's `name`. Run
             by `just ruleset-check`, which `just tag` calls before tagging.

WHY THE LIVE HALF IS NOT IN CI. It is compared against the live ruleset, and a pull request
that edits `main.json` is DRIFTED from live until `apply_main_ruleset.sh` runs after the
merge, so a live comparison on that PR would be red by design. The live half is therefore
reachable in CI only through the manual `workflow_dispatch` job in `ruleset-sync.yml`, which
exports `GH_TOKEN` from the workflow token because `gh api` refuses to run without a token
(exit 4 with no token and an empty config directory). Separately, an unauthenticated `curl` of
`GET /repos/lassiterdc/hhemt/rulesets` and `GET .../rulesets/18486873` returned HTTP 200 for this
public repository; that is a `curl` measurement and says nothing about `gh` or the CI token.

NORMALISATION. Both sides drop the server-assigned fields (`SERVER_ASSIGNED`), then order the
lists whose order GitHub does not preserve (rules by type, required checks by context, merge
methods, ref-name patterns). Every other key survives on both sides, so a key present on only
one side is drift and is reported, never normalised away.

WHAT THIS DOES NOT CHECK, stated so the guard does not read broader than it is:

  * The static half checks that a producing job EXISTS, not that its workflow triggers on the
    events the ruleset needs. A workflow retargeted away from `pull_request` still "produces"
    the context here and still hangs the release PR.
  * Jobs that call a reusable workflow (`uses:`) are not modelled: their check-run names are
    `{caller} / {callee job}`. A required context produced that way is reported missing, which
    fails loudly instead of passing silently.
  * A matrix given as an expression (`${{ fromJson(...) }}`) cannot be expanded without
    running the workflow, so such a job produces no context here for the same fail-loud reason.
  * `include:` entries are merged by the documented rule (extend every original combination
    the entry does not overwrite, else add a new combination); GitHub's exact expansion for
    exotic mixtures is not reproduced.

Exit 0 = the selected halves agree. 1 = a finding (a context nothing produces, or drift),
named. 2 = usage or environment error (file missing, `gh` absent or failing, PyYAML missing,
ruleset not found by name). An unverifiable live half is exit 2 and never a silent pass.
"""

from __future__ import annotations

import argparse
import difflib
import itertools
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RULESET_FILE = Path(".github") / "rulesets" / "main.json"
_WORKFLOW_DIR = Path(".github") / "workflows"

# Mirrors the constants in scripts/apply_main_ruleset.sh, which is the other consumer of the file.
DEFAULT_REPO = "lassiterdc/hhemt"

# Fields GitHub assigns or refreshes server-side. They are never in the committed file and are
# dropped from BOTH sides before comparing, so only a field a human can set is compared.
SERVER_ASSIGNED = frozenset(
    {"id", "source_type", "source", "node_id", "created_at", "updated_at", "current_user_can_bypass", "_links"}
)

_MATRIX_EXPR = re.compile(r"\$\{\{\s*matrix\.([A-Za-z0-9_-]+)\s*\}\}")
_ANY_EXPR = re.compile(r"\$\{\{.*?\}\}")


class EnvironmentProblem(Exception):
    """The guard could not run (missing file, `gh` failure, ...). Maps to exit 2, never to a pass."""


# --------------------------------------------------------------------------------------
# static half
# --------------------------------------------------------------------------------------


def required_contexts(ruleset: dict[str, Any]) -> list[str]:
    """The `context` of every required status check the ruleset declares, in file order."""
    contexts: list[str] = []
    for rule in ruleset.get("rules", []):
        if rule.get("type") == "required_status_checks":
            for check in rule.get("parameters", {}).get("required_status_checks", []):
                contexts.append(check["context"])
    return contexts


def _render_value(value: Any) -> str:
    """A matrix value as GitHub prints it in a check name (booleans are lower-case)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def expand_matrix(matrix: Any) -> list[dict[str, Any]] | None:
    """The matrix combinations, or None when the matrix is an expression that cannot be expanded.

    The cartesian product of every list-valued key other than `include`/`exclude`, minus the
    `exclude` matches, plus `include` entries merged by GitHub's documented rule.
    """
    if not isinstance(matrix, dict):
        return None  # a bare expression string such as ${{ fromJson(...) }}
    axes = {k: v for k, v in matrix.items() if k not in ("include", "exclude")}
    if any(not isinstance(v, list) for v in axes.values()):
        return None
    keys = list(axes)
    combos: list[dict[str, Any]] = (
        [dict(zip(keys, values, strict=True)) for values in itertools.product(*axes.values())] if keys else []
    )
    for ex in matrix.get("exclude", []) or []:
        combos = [c for c in combos if not all(k in c and c[k] == v for k, v in ex.items())]
    for inc in matrix.get("include", []) or []:
        original_keys = [k for k in inc if k in axes]
        matched = [c for c in combos if all(c[k] == inc[k] for k in original_keys)]
        if matched:
            for combo in matched:
                for k, v in inc.items():
                    if k not in axes:
                        combo[k] = v
        else:
            combos.append(dict(inc))
    return combos


def job_check_names(job_id: str, job: dict[str, Any]) -> list[str] | None:
    """The check-run names one job produces, or None when they cannot be derived statically."""
    if "uses" in job:
        return None  # reusable-workflow call: named `{caller} / {callee}`, not modelled
    name = job.get("name", job_id)
    if not isinstance(name, str):
        name = str(name)
    matrix = job.get("strategy", {}).get("matrix") if isinstance(job.get("strategy"), dict) else None
    if matrix is None:
        return None if _ANY_EXPR.search(name) else [name]
    combos = expand_matrix(matrix)
    if combos is None:
        return None
    if not combos:
        return None if _ANY_EXPR.search(name) else [name]
    names: list[str] = []
    for combo in combos:
        if _MATRIX_EXPR.search(name):
            rendered = _MATRIX_EXPR.sub(lambda m, c=combo: _render_value(c.get(m.group(1), "")), name)
        else:
            rendered = f"{name} ({', '.join(_render_value(v) for v in combo.values())})"
        if _ANY_EXPR.search(rendered):
            return None
        names.append(rendered)
    return names


def produced_check_names(repo_root: Path) -> dict[str, list[str]]:
    """Map of check-run name -> the `workflow-file:job-id` pairs that produce it."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - PyYAML is a declared dependency
        raise EnvironmentProblem("PyYAML is not importable; the static half reads workflow YAML with it.") from exc

    workflow_dir = repo_root / _WORKFLOW_DIR
    if not workflow_dir.is_dir():
        raise EnvironmentProblem(f"{workflow_dir} not found -- there are no workflows to match against.")
    produced: dict[str, list[str]] = {}
    for path in sorted([*workflow_dir.glob("*.yml"), *workflow_dir.glob("*.yaml")]):
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for job_id, job in (doc.get("jobs") or {}).items():
            names = job_check_names(str(job_id), job if isinstance(job, dict) else {})
            for name in names or []:
                produced.setdefault(name, []).append(f"{path.name}:{job_id}")
    return produced


def static_findings(ruleset: dict[str, Any], repo_root: Path) -> list[str]:
    """One finding per required context that no workflow job produces."""
    contexts = required_contexts(ruleset)
    if not contexts:
        return [
            f"{_RULESET_FILE} declares no required_status_checks rule, so no required context exists to verify. "
            f"The ruleset is meant to gate on status checks; restore the rule or retire this guard deliberately."
        ]
    produced = produced_check_names(repo_root)
    findings: list[str] = []
    for context in contexts:
        if context not in produced:
            findings.append(
                f"required context {context!r} is produced by no job in {_WORKFLOW_DIR}/. A required check nothing "
                f"reports leaves the PR 'Expected -- Waiting for status to be reported'. Either a job was "
                f"given another name or removed, or the context is misspelled. Check names this guard derived: "
                f"{sorted(produced)}"
            )
    return findings


# --------------------------------------------------------------------------------------
# live half
# --------------------------------------------------------------------------------------


def normalise(ruleset: dict[str, Any]) -> dict[str, Any]:
    """The comparable form of a ruleset: server fields dropped, unordered lists ordered."""
    out = {k: v for k, v in json.loads(json.dumps(ruleset)).items() if k not in SERVER_ASSIGNED}
    out.setdefault("bypass_actors", [])
    cond = out.get("conditions", {}).get("ref_name", {})
    for key in ("include", "exclude"):
        if isinstance(cond.get(key), list):
            cond[key] = sorted(cond[key])
    rules = out.get("rules", [])
    for rule in rules:
        params = rule.get("parameters")
        if not isinstance(params, dict):
            continue
        if isinstance(params.get("allowed_merge_methods"), list):
            params["allowed_merge_methods"] = sorted(params["allowed_merge_methods"])
        if isinstance(params.get("required_status_checks"), list):
            params["required_status_checks"] = sorted(params["required_status_checks"], key=lambda c: c["context"])
    out["rules"] = sorted(rules, key=lambda r: (r.get("type", ""), json.dumps(r, sort_keys=True)))
    return out


def _gh_json(path: str) -> Any:
    """`gh api {path}` parsed as JSON; any failure is an EnvironmentProblem, never a pass."""
    try:
        proc = subprocess.run(
            ["gh", "api", path, "-H", "Accept: application/vnd.github+json"],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
    except FileNotFoundError as exc:
        raise EnvironmentProblem("`gh` is not on PATH; the live half reads the ruleset with `gh api`.") from exc
    except subprocess.TimeoutExpired as exc:
        raise EnvironmentProblem(f"`gh api {path}` timed out after 60 s.") from exc
    if proc.returncode != 0:
        raise EnvironmentProblem(f"`gh api {path}` exited {proc.returncode}: {proc.stderr.strip()}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise EnvironmentProblem(f"`gh api {path}` did not return JSON: {exc}") from exc


def fetch_live(repo: str, name: str) -> dict[str, Any]:
    """The live ruleset named `name` on `repo`: list to find its id, then GET it in full."""
    listing = _gh_json(f"repos/{repo}/rulesets")
    ids = [r["id"] for r in listing if r.get("name") == name]
    if not ids:
        raise EnvironmentProblem(
            f"no ruleset named {name!r} on {repo} (found {sorted(r.get('name') for r in listing)}). "
            f"Apply the file with scripts/apply_main_ruleset.sh first."
        )
    if len(ids) > 1:
        raise EnvironmentProblem(f"{len(ids)} rulesets named {name!r} on {repo} (ids {ids}); expected exactly one.")
    return _gh_json(f"repos/{repo}/rulesets/{ids[0]}")


def live_findings(ruleset: dict[str, Any], repo: str) -> list[str]:
    """A unified diff (file -> live) when the normalised file and live ruleset differ."""
    live = fetch_live(repo, ruleset.get("name", ""))
    file_text = json.dumps(normalise(ruleset), indent=2, sort_keys=True).splitlines()
    live_text = json.dumps(normalise(live), indent=2, sort_keys=True).splitlines()
    if file_text == live_text:
        return []
    diff = "\n".join(difflib.unified_diff(file_text, live_text, fromfile="file", tofile="live", lineterm="", n=2))
    return [
        f"the live ruleset on {repo} differs from {_RULESET_FILE}. '-' lines are only in the file, '+' lines only "
        f"live. Reconcile by editing the file (live is right) or by running scripts/apply_main_ruleset.sh "
        f"(the file is right):\n{diff}"
    ]


# --------------------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run the selected halves and return the exit code (0 agree, 1 finding, 2 could not run).

    A caller must select at least one of `--static` / `--live`; selecting neither is an argparse
    usage error (exit 2) so a misspelt recipe never reads as a passing check.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--static", action="store_true", help="every required context has a producing workflow job")
    ap.add_argument("--live", action="store_true", help="the normalised file equals the live ruleset (needs `gh`)")
    ap.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    ap.add_argument("--repo", default=DEFAULT_REPO, help="owner/name for the live half (default: %(default)s)")
    args = ap.parse_args(argv)
    if not (args.static or args.live):
        ap.error("select at least one half: --static and/or --live")

    ruleset_path = args.repo_root / _RULESET_FILE
    try:
        if not ruleset_path.is_file():
            raise EnvironmentProblem(f"{ruleset_path} not found -- nothing to check.")
        ruleset = json.loads(ruleset_path.read_text(encoding="utf-8"))
        findings: list[str] = []
        halves: list[str] = []
        if args.static:
            halves.append("static")
            findings += static_findings(ruleset, args.repo_root)
        if args.live:
            halves.append("live")
            findings += live_findings(ruleset, args.repo)
    except (EnvironmentProblem, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if findings:
        print("ruleset sync check FAILED:", file=sys.stderr)
        for finding in findings:
            print(f"  {finding}", file=sys.stderr)
        print(f"\n{len(findings)} finding(s) in the {'+'.join(halves)} half.", file=sys.stderr)
        return 1
    print(f"ruleset sync OK -- {'+'.join(halves)} half agrees for {_RULESET_FILE}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
