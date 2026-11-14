"""The ledger record's field set and the `owed` verb's parameter set are both CLOSED.

Property: `policy.ledger_entry` produces a record whose key set EQUALS the field set this
package's prose enumerates, with every value inside its stated domain, and `policy.owed`
accepts exactly the four kernel parameters `{toolkit, runs_root, interval_hours, now}`.

Class: NEW CAPABILITY. There is no prior ledger to regress against, so each arm names the
plausible wrong implementation it is red under: a record widened with a changed-file list, and
a verb that takes the change set as an argument.

Why it is required: this run deliberately delivers NOTHING for diff-driven per-landing test
selection, and that deferral is only checkable if the artifacts cannot carry the question. A
record with a `changed_files` key, or a verb with a `diff` parameter, would make "is a full run
owed for this change" answerable by accident -- and an accidental answer of that kind fails
GREEN in the direction that looks like success. Both arms therefore range over the PROPERTY
(no artifact can carry a change set) rather than over container types: a duck-typed "accepts no
path, collection or iterable" assertion cannot be satisfied by a correct module, because the
house pattern's own first parameter is an introspectable string.

What kills it: adding any field to `LEDGER_FIELDS` without amending the criterion, any record
key outside the enumerated set, a `scope` or `verdict` outside its two-value domain, or any
one of the four kernel parameters being renamed, added to, or replaced.

A second correct implementation under which it still passes: the record could be assembled by a
dataclass or a TypedDict and serialised at append time, and `owed` could take its four kernel
parameters positionally rather than keyword-only. Neither changes what is asserted, because
both arms read NAMES and DOMAINS rather than structure or call convention.

The six `owed` fixtures at the bottom are the other half of the criterion: each is one distinct
owed/not-owed case, and together they are why a single one-sided arm would be insufficient --
a verb that always answered "owed" passes four of them and a verb that always answered "not
owed" passes one.
"""

from __future__ import annotations

import inspect
import json
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from hhemt.cache.keys import toolkit_src_digest
from hhemt.suite import policy

#: The criterion's enumeration, COPIED BY HAND rather than imported, so a change to
#: `policy.LEDGER_FIELDS` cannot satisfy this arm by editing one side. Each member is one fact
#: about one run; none is a summary of a summary, and none can name a set of files.
_ENUMERATED_FIELDS: tuple[str, ...] = (
    "run_id",
    "covered_sha",
    "toolkit_src_digest",
    "verdict",
    "problem_count",
    "advisory_count",
    "missing_conformance_sites",
    "wall_clock",
    "environment_name",
    "environment_digest",
    "compile_venue_permitted",
    "compile_tier_required",
    "scope",
    "estate_harness_sha",
)

_NOW = datetime(2026, 10, 30, 12, 0, tzinfo=UTC)
_RECENT = (_NOW - timedelta(hours=1)).isoformat()
_OLD = (_NOW - timedelta(days=30)).isoformat()
_INTERVAL_HOURS = 24.0


def _change_set_keys(names: Any) -> set[str]:
    """The names that could carry a change set. The instrument both arms' absence rests on."""
    return {name for name in names if policy.CHANGE_SET_KEY_RE.search(name)}


def _git(repo: Path, *args: str) -> str:
    """Run git in `repo` and return its stdout, with commit identity supplied per call.

    `--allow-empty` commits are enough here and are chosen rather than forced: `owed` compares
    the record against the checkout's `HEAD` and its WORKING-TREE digest, so three distinct
    commits give three distinct shas without staging anything. Nothing in this module's
    behaviour reads a commit's contents.
    """
    proc = subprocess.run(  # noqa: S603 -- fixed argv, no shell
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=ledger-schema-test",
            "-c",
            "user.email=ledger@test.invalid",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def _toolkit(base: Path, *, commits: int = 3) -> Path:
    """A throwaway checkout: `src/hhemt/` plus `commits` commits, so `HEAD~2` is meaningful."""
    repo = base / "toolkit"
    (repo / "src" / "hhemt").mkdir(parents=True)
    (repo / "src" / "hhemt" / "mod.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "init", "-q")
    for index in range(commits):
        _git(repo, "commit", "-q", "--allow-empty", "-m", f"c{index}")
    return repo


def _record(
    *,
    run_id: str,
    covered_sha: str,
    digest: str,
    scope: str = "union",
    verdict: str = "GREEN",
    wall_clock: str = _RECENT,
    missing_sites: tuple[str, ...] = (),
    compile_venue: bool = True,
    compile_tier: bool = True,
) -> dict[str, Any]:
    """One ledger line, built from the fields the record's contract states.

    Deliberately spelled out rather than derived from `policy.LEDGER_FIELDS`: a fixture derived
    from the implementation under test cannot detect a field the implementation stopped
    emitting, because both sides move together. A field missing here surfaces as `owed` reading
    an absent value, which is the non-discharging state -- a loud failure, not a silent pass.
    """
    return {
        "run_id": run_id,
        "covered_sha": covered_sha,
        "toolkit_src_digest": digest,
        "verdict": verdict,
        "problem_count": 0,
        "advisory_count": 0,
        "missing_conformance_sites": list(missing_sites),
        "wall_clock": wall_clock,
        "environment_name": "hhemt",
        "environment_digest": "0" * 64,
        "compile_venue_permitted": compile_venue,
        "compile_tier_required": compile_tier,
        "scope": scope,
        "estate_harness_sha": "e" * 40,
    }


def _write_ledger(runs_root: Path, entries: list[dict[str, Any]]) -> None:
    runs_root.mkdir(parents=True, exist_ok=True)
    lines = "".join(json.dumps(entry) + "\n" for entry in entries)
    (runs_root / policy.LEDGER_NAME).write_text(lines, encoding="utf-8")


def _summary(runs_root: Path, run_id: str, complement: list[str]) -> None:
    """A run's own `summary.json`, carrying the declared complement `owed` has to name."""
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps({"declared_complement": complement}), encoding="utf-8")


# ---------------------------------------------------------------------------------------
# Arm 1 -- the record
# ---------------------------------------------------------------------------------------
def _aggregated_run(base: Path) -> tuple[Path, dict[str, Any]]:
    """A run dir carrying the artifacts `ledger_entry` reads, plus the `aggregate()` result."""
    repo = _toolkit(base)
    run_id = "20261030T100000Z_abcdef123456"
    run_dir = base / "runs" / run_id
    run_dir.mkdir(parents=True)
    sha = _git(repo, "rev-parse", "HEAD")
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "source_sha": sha,
                "chunk_count": 1,
                "compile_precondition": {"compile_venue_permitted": True, "compile_tier_required": True},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "version_expectation.json").write_text(
        json.dumps({"schema": 1, "sha": sha, "tree": str(repo)}), encoding="utf-8"
    )
    # The roster is satisfied (drive, verdict, chunk-00) so the missing-site list is EMPTY by
    # measurement rather than by the run having written no declaration at all.
    conformance = run_dir / "_conformance"
    conformance.mkdir()
    for site in ("drive", "verdict", "chunk-00"):
        (conformance / f"{site}.json").write_text("{}", encoding="utf-8")
    result = {
        "run_id": run_id,
        "source_sha": sha,
        "verdict": "GREEN",
        "scope": "union",
        "problems": [],
        "advisories": ["[disclosure] something a reader should see"],
        "counts": {"collected": 1},
    }
    return run_dir, result


def test_ledger_record_key_set_is_the_enumerated_field_set(tmp_path: Path) -> None:
    """The record carries exactly the enumerated fields, in their stated domains, and no more."""
    run_dir, result = _aggregated_run(tmp_path)

    record = policy.ledger_entry(run_dir, result)

    assert set(record) == set(_ENUMERATED_FIELDS), (
        "the ledger record's key set moved away from the criterion's enumeration; a field added "
        "here is an amendment to the shape, not an edit to this test: got "
        f"{sorted(set(record) ^ set(_ENUMERATED_FIELDS))}"
    )
    assert policy.LEDGER_FIELDS == _ENUMERATED_FIELDS, "the module's declared field set drifted from the criterion"

    # Three value-domain assertions. The two two-value domains are the ones `owed` branches on,
    # and a third value in either would make a discharge reachable from a state the criterion
    # says cannot discharge.
    assert record["scope"] in {"array", "union"}
    assert record["verdict"] in {"GREEN", "NOT-GREEN"}
    assert record["verdict"] == "GREEN"
    assert record["advisory_count"] == 1, "the advisory count must count the result's advisories, not a constant"
    assert record["missing_conformance_sites"] == [], (
        "a run whose roster is fully reported must record an EMPTY site list; a non-empty list "
        f"here means the roster was computed over the wrong directory: {record['missing_conformance_sites']}"
    )

    # No key can carry a change set -- and the instrument is proven to fire before its zero is
    # believed. The control varies the PROPERTY (a name that can carry a file list) rather than
    # the token: a matcher that matched nothing would return the same empty set here.
    assert _change_set_keys(_ENUMERATED_FIELDS) == set()
    assert _change_set_keys({*_ENUMERATED_FIELDS, "changed_files"}) == {"changed_files"}


# ---------------------------------------------------------------------------------------
# Arm 2 -- the verb
# ---------------------------------------------------------------------------------------
def test_owed_signature_accepts_no_change_set() -> None:
    """The kernel's parameter set is an ALLOW-LIST, and it targets the kernel, not an entry point.

    `_cli._owed` and `aggregate`'s writer are the two layers that delegate here; either entry
    point would flag a correct module, because `_cli._owed` carries a Typer `OptionInfo` default
    for every parameter and `aggregate.main` carries `argv`. The kernel is the callable whose
    signature is a statement about the question the verb can answer.
    """
    names = set(inspect.signature(policy.owed).parameters)

    assert names == {"toolkit", "runs_root", "interval_hours", "now"}, (
        "the kernel's parameter set is the criterion's own; adding a FIFTH name is an amendment "
        f"to the shape row, not an edit to this test. Got {sorted(names)}"
    )
    assert _change_set_keys(names) == set()
    # The control the zero above needs: a name that DOES match is rejected by the same
    # instrument, so the empty set is a measurement rather than a matcher that never fires.
    assert _change_set_keys({*names, "diff"}) == {"diff"}


# ---------------------------------------------------------------------------------------
# The six owed fixtures
# ---------------------------------------------------------------------------------------
def test_fixture_1_never_run_is_owed(tmp_path: Path) -> None:
    """No ledger at all: nothing has ever been aggregated for this runs root."""
    repo = _toolkit(tmp_path)
    runs_root = tmp_path / "runs"

    verdict = policy.owed(toolkit=repo, runs_root=runs_root, interval_hours=_INTERVAL_HOURS, now=_NOW)

    assert verdict["owed"] is True
    assert "never-run" in verdict["reason"]


def test_fixture_2_union_and_green_at_head_inside_the_interval_is_not_owed(tmp_path: Path) -> None:
    """The discharging case: every conjunct holds, so the obligation is satisfied."""
    repo = _toolkit(tmp_path)
    runs_root = tmp_path / "runs"
    _write_ledger(
        runs_root,
        [
            _record(
                run_id="20261030T100000Z_abcdef123456",
                covered_sha=_git(repo, "rev-parse", "HEAD"),
                digest=toolkit_src_digest(repo / "src" / "hhemt"),
            )
        ],
    )

    verdict = policy.owed(toolkit=repo, runs_root=runs_root, interval_hours=_INTERVAL_HOURS, now=_NOW)

    assert verdict["owed"] is False, verdict["reason"]
    assert verdict["run_id"] == "20261030T100000Z_abcdef123456"


def test_fixture_3_union_green_at_an_ancestor_outside_the_interval_is_owed(tmp_path: Path) -> None:
    """A real ancestor green, two commits back and recorded long ago, refuses to discharge.

    The record is a green over a DIFFERENT tree state. It is the case a naive "there is a green
    in the ledger" implementation answers wrongly, and it is why the sha and the interval are
    two named reasons rather than one folded check.
    """
    repo = _toolkit(tmp_path)
    ancestor = _git(repo, "rev-parse", "HEAD~2")
    runs_root = tmp_path / "runs"
    _write_ledger(
        runs_root,
        [
            _record(
                run_id="20260930T100000Z_abcdef123456",
                covered_sha=ancestor,
                digest=toolkit_src_digest(repo / "src" / "hhemt"),
                wall_clock=_OLD,
            )
        ],
    )

    verdict = policy.owed(toolkit=repo, runs_root=runs_root, interval_hours=_INTERVAL_HOURS, now=_NOW)

    assert verdict["owed"] is True
    assert "is not the live HEAD" in verdict["reason"], verdict["reason"]
    assert "older than the" in verdict["reason"], verdict["reason"]


def test_fixture_4_array_scoped_green_is_owed_and_names_its_complement(tmp_path: Path) -> None:
    """A `scope=array` green never discharges, and the refusal names what the array could not cover."""
    repo = _toolkit(tmp_path)
    runs_root = tmp_path / "runs"
    run_id = "20261030T100000Z_abcdef123456"
    complement = [
        "tests/test_doi_roundtrip_e2e.py::test_live_deposit_run_proof",
        "tests/test_synth_09_from_doi_run_proof.py::test_local_coupled_run_proof",
    ]
    _write_ledger(
        runs_root,
        [
            _record(
                run_id=run_id,
                covered_sha=_git(repo, "rev-parse", "HEAD"),
                digest=toolkit_src_digest(repo / "src" / "hhemt"),
                scope="array",
            )
        ],
    )
    _summary(runs_root, run_id, complement)

    verdict = policy.owed(toolkit=repo, runs_root=runs_root, interval_hours=_INTERVAL_HOURS, now=_NOW)

    assert verdict["owed"] is True
    assert "never discharges" in verdict["reason"], verdict["reason"]
    assert verdict["complement"] == complement, "an array refusal that does not name its complement is unactionable"


def test_fixture_5_union_green_with_a_differing_working_tree_digest_is_owed(tmp_path: Path) -> None:
    """A green at the right COMMIT says nothing about the tree when the tree has moved under it."""
    repo = _toolkit(tmp_path)
    runs_root = tmp_path / "runs"
    sha = _git(repo, "rev-parse", "HEAD")
    recorded_digest = toolkit_src_digest(repo / "src" / "hhemt")
    _write_ledger(
        runs_root,
        [_record(run_id="20261030T100000Z_abcdef123456", covered_sha=sha, digest=recorded_digest)],
    )
    # An UNCOMMITTED edit: the sha is unchanged and the working-tree digest is not. That is the
    # whole reason the record carries a digest beside the sha.
    (repo / "src" / "hhemt" / "mod.py").write_text("VALUE = 2\n", encoding="utf-8")
    assert _git(repo, "rev-parse", "HEAD") == sha, "the fixture must keep the sha fixed to isolate the digest"
    assert toolkit_src_digest(repo / "src" / "hhemt") != recorded_digest

    verdict = policy.owed(toolkit=repo, runs_root=runs_root, interval_hours=_INTERVAL_HOURS, now=_NOW)

    assert verdict["owed"] is True
    assert "toolkit_src_digest differs" in verdict["reason"], verdict["reason"]


@pytest.mark.parametrize(
    ("compile_venue", "compile_tier", "conjunct"),
    [(False, True, "capability"), (True, False, "anti-silence")],
)
def test_fixture_6_union_green_from_a_non_compiling_session_is_owed(
    tmp_path: Path, compile_venue: bool, compile_tier: bool, conjunct: str
) -> None:
    """A green whose session could not compile is owed, and the missing conjunct is NAMED.

    Both arms are exercised because neither conjunct implies the other: capability without
    anti-silence produces the quiet `N skipped`, and anti-silence without capability produces a
    loud red rather than a green. A record that folded them into one boolean would admit one.
    """
    repo = _toolkit(tmp_path)
    runs_root = tmp_path / "runs"
    _write_ledger(
        runs_root,
        [
            _record(
                run_id="20261030T100000Z_abcdef123456",
                covered_sha=_git(repo, "rev-parse", "HEAD"),
                digest=toolkit_src_digest(repo / "src" / "hhemt"),
                compile_venue=compile_venue,
                compile_tier=compile_tier,
            )
        ],
    )

    verdict = policy.owed(toolkit=repo, runs_root=runs_root, interval_hours=_INTERVAL_HOURS, now=_NOW)

    assert verdict["owed"] is True
    assert verdict["missing_conjuncts"] == [conjunct], verdict["missing_conjuncts"]
    assert conjunct in verdict["reason"], verdict["reason"]
