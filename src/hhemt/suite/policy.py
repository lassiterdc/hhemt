"""Whether a full suite run is OWED, and the durable record every aggregated run appends.

PROBLEM THIS FILE SOLVES. A green suite run is evidence for ONE commit on ONE working tree,
and until now nothing durable remembered it: an operator asking "may I land this?" had to
reconstruct the answer by hand from a directory of run dirs. That reconstruction silently
answers a different question than the one asked, in three ways, and none of the three is
visible in a bare verdict word:

* a ``scope=array`` green covers only what a scheduler allocation can evaluate, and it
  reports ``GREEN`` for its own complement -- so it is not evidence about the suite;
* a green whose session could not compile SKIPS the compile tier and reports the same
  ``GREEN`` as one that built it, because a skip is not a failure;
* a green recorded against one working tree says nothing about the tree on disk now, and
  the identity that would distinguish them is a DIGEST while the run record carried a PATH.

So the record is written once, at aggregate time, into one append-only file per runs root,
and one verb reads it. The verb REFUSES rather than records: a non-discharging green does
not weaken into a softer "not owed", it returns owed together with the uncovered part.

WHAT A RUNS ROOT IS, stated because it bounds the multi-machine property. The ledger lives
at ``{runs_root}/_ledger.jsonl``, and a runs root is per-machine unless it sits on shared
scratch. On the HPC venue it does, so one venue's runs answer for each other; on a laptop
they do not. That is a property of the substrate rather than a defect this module can
repair, and the alternative -- a central store -- would put a network dependency inside the
release gate, which is the one place that must answer offline and unattended.

DELIBERATELY NOT A ``grep``-OVER-RUN-DIRS IMPLEMENTATION: a walk has to guess which
directory is a run, has to re-derive the verdict from a summary that may be absent, and
cannot answer at all about a run whose directory has been reclaimed. An append-only line is
the smallest artifact that survives all three.
"""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

#: The ledger file, one JSON object per line, inside the runs root.
LEDGER_NAME = "_ledger.jsonl"

#: The record's field set, CLOSED. Every member is computable from something the run has
#: already written; nothing here is a summary of a summary.
#:
#: WHY IT IS CLOSED AND WHY THIS TUPLE IS THE DECLARATION: `tests/test_suite_ledger_schema.py`
#: asserts the produced record's key set EQUAL to this tuple AND this tuple equal to its own
#: literal copy, which is what stops a later editor from widening the record by editing one
#: side. The widened field that matters is a changed-file list: a ledger keyed on a change set
#: would answer "is a full run owed for THIS diff", a question this run does not deliver and
#: which `CHANGE_SET_KEY_RE` below names so the absence is a decision rather than an oversight.
LEDGER_FIELDS: tuple[str, ...] = (
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

#: Every key name that could carry a CHANGE SET rather than a fact about one run. No ledger
#: field may match it. Held here rather than in the test because it states the record's
#: contract rather than one assertion about it, and the test imports it so the two cannot
#: drift -- with a must-fire control in that test, because a regex that matches nothing
#: passes this assertion over any key set whatsoever.
CHANGE_SET_KEY_RE = re.compile(r"diff|paths|changed|files|since|base")

#: Scopes that APPEND a ledger line. `triage` and `complement` do not answer the suite-level
#: question at all, and a record that made no suite-level claim would be read by `owed` as one
#: that failed to. Omitting them keeps the ledger's population homogeneous.
LEDGERED_SCOPES = frozenset({"array", "union"})

#: Scopes whose green can discharge the obligation. Exactly one, and it is the same
#: restriction `aggregate.py` states on the verdict line: only `union` supports a suite claim.
DISCHARGING_SCOPES = frozenset({"union"})

#: The run id's leading stamp, `{YYYYmmddTHHMMSSZ}_{sha12}` -- `_runner.plan` mints it, so
#: parsing it reads the plan-time instant without a second artifact.
_RUN_ID_STAMP = re.compile(r"^(\d{8}T\d{6}Z)_")

#: The two compile conjuncts, by the ledger key that records each. Neither implies the other,
#: so they are reported by name rather than folded into one boolean: an operator told only
#: "owed" cannot tell which export to set.
_COMPILE_CONJUNCTS: tuple[tuple[str, str], ...] = (
    ("capability", "compile_venue_permitted"),
    ("anti-silence", "compile_tier_required"),
)


def ledger_path(runs_root: str | os.PathLike[str]) -> Path:
    """The one file this module appends to and reads from."""
    return Path(runs_root) / LEDGER_NAME


def read_ledger(runs_root: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Every record in the runs root, in append order. An absent ledger reads as empty.

    A MALFORMED LINE IS SKIPPED RATHER THAN FATAL. This file is appended by concurrent
    aggregations on a shared runs root, so a torn final line is a real possibility after a node
    dies; raising there would make every later question unanswerable because of one dead run.
    The skip is silent by necessity -- `owed` has no channel to report a line it could not
    parse -- and the cost of it is bounded: a skipped line can only make the answer MORE owed.
    """
    path = ledger_path(runs_root)
    if not path.is_file():
        return []
    entries: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict):
            entries.append(record)
    return entries


def append_ledger_entry(runs_root: str | os.PathLike[str], entry: dict[str, Any]) -> Path:
    """Append one record, and return the file it landed in.

    ONE `write` CALL PER RECORD, because that is what makes a concurrent append safe: the
    whole line reaches the file in a single write below the pipe buffer, so two aggregations
    on a shared runs root interleave at line granularity rather than mid-record. A loop over
    `json.dump` pieces would not have that property.

    `sort_keys=True` so a record's rendering is a function of its content rather than of the
    dict's insertion order -- two runs with equal facts then produce equal bytes, which is what
    makes a hand diff of two lines readable.
    """
    path = ledger_path(runs_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
    return path


# ---------------------------------------------------------------------------------------
# Building the record
# ---------------------------------------------------------------------------------------
def _read_json(path: Path) -> dict[str, Any]:
    """A JSON object from `path`, or an empty dict. Absence reads as absence, never as zero."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _wall_clock_from_run_id(run_id: str) -> str:
    """The plan-time instant, ISO-8601 Z, parsed from the run id's leading stamp.

    WALL CLOCK AS AN INSTANT, not as a duration, and the choice is forced by the only question
    anything asks of it: `owed` compares it against `now - interval_hours`, and a duration
    cannot answer "how long ago". A run id that does not carry the stamp yields the empty
    string, which `owed` reports as an unreadable wall clock rather than treating as recent.
    """
    match = _RUN_ID_STAMP.match(run_id)
    if not match:
        return ""
    try:
        stamp = datetime.strptime(match.group(1), "%Y%m%dT%H%M%SZ")
    except ValueError:
        return ""
    return stamp.replace(tzinfo=UTC).isoformat()


def _environment_name() -> str:
    """The environment the aggregation ran in, by the name a caller would re-create it from.

    `CONDA_PREFIX` is the fallback rather than the primary because `conda run` sets both and
    only `CONDA_DEFAULT_ENV` carries the NAME the operator typed; a bare prefix path would
    record a hash-suffixed env directory instead.
    """
    name = os.environ.get("CONDA_DEFAULT_ENV", "")
    if name:
        return name
    prefix = os.environ.get("CONDA_PREFIX", "")
    if prefix:
        return Path(prefix).name
    return Path(os.environ.get("VIRTUAL_ENV", "")).name


def _environment_digest_or_empty() -> str:
    """The SOLVER-tier environment digest, or the empty string when it cannot be resolved.

    THE COST, stated because this runs once per aggregated run: two `--version` subprocesses
    (`cmake`, the C++ compiler) plus package-metadata reads. Against hours of suite wall clock
    that is proportionate, and the alternative -- recording the environment NAME alone -- does
    not identify a solver stack, since two machines can both call the env `hhemt` and build
    different binaries. `keys.environment_values` is process-cached, so a second aggregation in
    one process pays nothing.

    An import-time dependency on `hhemt.cache.keys` is avoided deliberately: that package's
    `__init__` pulls `clear.py`, which reaches `filelock` and `hhemt.utils`, and this module is
    imported by the aggregate CLI whose import path should stay cheap.
    """
    try:
        from hhemt.cache.keys import environment_digest

        return environment_digest("solver")
    except Exception:  # noqa: BLE001 -- a digest that cannot be resolved must not lose the record
        return ""


def _estate_harness_sha(run_dir: Path) -> str:
    """The estate harness's own sha, as the venue step recorded it, or the empty string.

    WRITTEN BY THE VENUE because only the venue knows its own sha: the array element is the
    participant that lives in the estate repository, and the aggregator runs from the toolkit.
    One file per chunk (`estate_harness_chunk-{NN}.json`, written by `submit_suite_uva.sh`)
    rather than one shared file, so N array elements do not race a single path. They all carry
    the same value; the first in sorted order is read.

    `UNRESOLVED` is the venue's own sentinel for "the estate is not a git tree", and it is
    mapped to the empty string here so a non-git estate does not masquerade as a sha, exactly
    as `keys.toolkit_sha` treats an absent checkout.
    """
    for path in sorted(run_dir.glob("estate_harness_chunk*.json")):
        sha = _read_json(path).get("sha") or ""
        return "" if sha == "UNRESOLVED" else str(sha)
    return ""


def _missing_conformance_sites(run_dir: Path, manifest: dict[str, Any]) -> list[str]:
    """The conformance sites the run implies and no record names.

    THE FIELD THAT SEPARATES TWO KINDS OF UNION-SCOPED GREEN. "union-scoped green over all
    chunks" and "union-scoped green over the chunks that happened to report" render identically
    on the verdict line; `missing_conformance_sites` is the whole difference, and `owed`
    refuses a discharge while it is non-empty.

    The roster is sized from `manifest.json` and is only meaningful for a run that wrote a
    version declaration, so a run without one reports no sites rather than a false shortfall of
    every site -- the same NO_EXPECTATION exemption `aggregate.main()` applies.
    """
    chunk_count = manifest.get("chunk_count")
    if not isinstance(chunk_count, int) or not (run_dir / "version_expectation.json").is_file():
        return []
    try:
        from hhemt.suite import missing_conformance_sites

        return missing_conformance_sites(run_dir, chunk_count=chunk_count)
    except Exception:  # noqa: BLE001 -- a roster that cannot be sized must not lose the record
        return []


def ledger_entry(run_dir: str | os.PathLike[str], result: dict[str, Any]) -> dict[str, Any]:
    """The ledger record for one aggregated run, built from what the run already wrote.

    `result` is `aggregate()`'s return dict as it stands AFTER every post-aggregate blocking
    append, so the verdict recorded is the one the operator was shown. Building this from the
    summary on disk instead would make the record a derivation of a derivation and would fail
    for the one run whose summary could not be written.

    Every value is either read from an artifact or computed from the tree; NONE is a judgement.
    A value that cannot be resolved is the empty string (an absent digest) or the empty list
    (no missing sites) rather than a plausible-looking zero, and `owed` treats each absence as
    the non-discharging state.
    """
    run_dir = Path(run_dir)
    manifest = _read_json(run_dir / "manifest.json")
    expectation = _read_json(run_dir / "version_expectation.json")
    precondition = manifest.get("compile_precondition") or {}
    tree = str(expectation.get("tree") or "")
    digest = ""
    if tree:
        try:
            from hhemt.cache.keys import toolkit_src_digest

            digest = toolkit_src_digest(Path(tree) / "src" / "hhemt")
        except Exception:  # noqa: BLE001 -- an unresolvable digest is recorded as absent
            digest = ""
    return {
        "run_id": str(result.get("run_id") or manifest.get("run_id") or ""),
        "covered_sha": str(result.get("source_sha") or manifest.get("source_sha") or ""),
        "toolkit_src_digest": digest,
        "verdict": str(result.get("verdict") or ""),
        "problem_count": len(result.get("problems") or []),
        "advisory_count": len(result.get("advisories") or []),
        "missing_conformance_sites": _missing_conformance_sites(run_dir, manifest),
        "wall_clock": _wall_clock_from_run_id(str(result.get("run_id") or manifest.get("run_id") or "")),
        "environment_name": _environment_name(),
        "environment_digest": _environment_digest_or_empty(),
        "compile_venue_permitted": precondition.get("compile_venue_permitted"),
        "compile_tier_required": precondition.get("compile_tier_required"),
        "scope": str(result.get("scope") or ""),
        "estate_harness_sha": _estate_harness_sha(run_dir),
    }


# ---------------------------------------------------------------------------------------
# Reading the record
# ---------------------------------------------------------------------------------------
def _parse_instant(value: Any) -> datetime | None:
    """An aware UTC instant from an ISO-8601 string, or None when it is absent/unreadable."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if not isinstance(value, str) or not value:
        return None
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _live_covered_sha(toolkit: Path) -> str:
    """`HEAD` of the checkout under question, or the empty string outside one."""
    from hhemt.cache.keys import toolkit_sha

    return toolkit_sha(toolkit)


def _live_toolkit_src_digest(toolkit: Path) -> str:
    """The digest the live working tree would be recorded under, from its OWN bytes."""
    from hhemt.cache.keys import toolkit_src_digest

    return toolkit_src_digest(toolkit / "src" / "hhemt")


def _complement_of(entry: dict[str, Any], runs_root: Path) -> list[str] | None:
    """The covered run's declared complement, read from its own summary. None when unreadable.

    A `scope=array` green is refused BECAUSE of its complement, so refusing it without naming
    that complement leaves an operator with a verdict and no way to act on it. The value is the
    run's own `declared_complement` (the plan-time set), with the disclosure's excluded-node
    union as the fallback for a summary written before that key existed.

    None is returned rather than an empty list when neither can be read: "the complement is
    empty" and "the summary is gone" are different facts, and an empty list would state the
    first.
    """
    run_id = str(entry.get("run_id") or "")
    if not run_id:
        return None
    summary = _read_json(runs_root / run_id / "summary.json")
    declared = summary.get("declared_complement")
    if isinstance(declared, list):
        return [str(node) for node in declared]
    excluded = ((summary.get("disclosure") or {}).get("excluded_nodes") or {}).get("nodes")
    if isinstance(excluded, list):
        return [str(node) for node in excluded]
    return None


def _classify(
    entry: dict[str, Any],
    *,
    live_sha: str,
    live_digest: str,
    reference: datetime,
    interval_hours: float,
    runs_root: Path,
) -> dict[str, Any]:
    """Why this record does or does not discharge, enumerated rather than ranked.

    EVERY FAILING CONJUNCT IS NAMED, not only the first: a record can be stale, array-scoped
    and non-compiling at once, and an operator who fixes only the reported one pays a full
    suite run to discover the next. The conjunction is the criterion, so the report is its
    unsatisfied terms.
    """
    reasons: list[str] = []
    scope = str(entry.get("scope") or "")
    if scope not in DISCHARGING_SCOPES:
        reasons.append(f"scope={scope or 'absent'} never discharges a suite-level obligation")
    verdict = str(entry.get("verdict") or "")
    if verdict != "GREEN":
        reasons.append(f"verdict={verdict or 'absent'}")

    recorded_sha = str(entry.get("covered_sha") or "")
    if not recorded_sha:
        reasons.append("the record carries no covered sha")
    elif recorded_sha != live_sha:
        reasons.append(f"recorded sha {recorded_sha[:12]} is not the live HEAD {live_sha[:12] or 'UNRESOLVED'}")

    recorded_digest = str(entry.get("toolkit_src_digest") or "")
    if not recorded_digest:
        reasons.append("the record carries no toolkit_src_digest, so the working tree it ran on is unknown")
    elif recorded_digest != live_digest:
        reasons.append("the live working tree's toolkit_src_digest differs from the recorded one")

    missing_sites = entry.get("missing_conformance_sites") or []
    if missing_sites:
        reasons.append(
            f"{len(missing_sites)} conformance site(s) never reported, so the run's coverage of "
            f"its own chunks is unknown: {sorted(map(str, missing_sites))}"
        )

    missing_conjuncts = [name for name, key in _COMPILE_CONJUNCTS if entry.get(key) is not True]
    if missing_conjuncts:
        reasons.append(
            "the session lacked the compile conjunct(s) "
            + ", ".join(missing_conjuncts)
            + ", so a green may be one whose compile tier skipped"
        )

    instant = _parse_instant(entry.get("wall_clock"))
    if instant is None:
        reasons.append("the record carries no readable wall clock, so its age is unknown")
    elif reference - instant > timedelta(hours=float(interval_hours)):
        reasons.append(f"recorded {instant.isoformat()} is older than the {interval_hours}h interval")

    return {
        "reasons": reasons,
        "complement": _complement_of(entry, runs_root) if scope not in DISCHARGING_SCOPES else None,
        "missing_conjuncts": missing_conjuncts,
        "run_id": str(entry.get("run_id") or "") or None,
        "covered_sha": recorded_sha or None,
    }


def owed(
    *,
    toolkit: str | os.PathLike[str],
    runs_root: str | os.PathLike[str],
    interval_hours: float,
    now: datetime | str | None = None,
) -> dict[str, Any]:
    """Is a full suite run OWED for `toolkit`? Assertion-first, and it REFUSES on a weak green.

    FOUR PARAMETERS, AND THE SET IS AN ALLOW-LIST RATHER THAN A DENY-LIST. `tests/
    test_suite_ledger_schema.py::test_owed_signature_accepts_no_change_set` asserts these
    EXACT names, and the reason is a property rather than a style: no argument can carry a
    change set, so no future caller can make this verb answer "which tests must run for this
    diff". That capability is deferred and this run delivers none of it, and the four names are
    the whole of what bounds the question.

    `toolkit` IS REQUIRED rather than tolerated, and it is not a change set. The question needs
    the repository twice: once for the LIVE digest to compare against the record's, and once
    for the live `HEAD` sha. Passing the repository rather than reading a process-wide working
    directory is what closes the cwd route -- a caller in any directory gets the answer for the
    tree it named, and `aggregate`'s own `--run-dir` has no cwd dependency to inherit.

    `now` is an injection point for the interval test and for nothing else; production callers
    omit it. It is named `now` because that is what it is.

    Returns a dict rather than a boolean. "owed" alone cannot be acted on: the caller needs the
    reason (which conjunct failed), the covering sha when one exists, and for an array-scoped
    green the complement that green did not cover. A boolean would force every caller to
    re-derive those from the ledger, which is the hand-reconstruction this module removes.
    """
    toolkit_path = Path(toolkit)
    runs_root_path = Path(runs_root)
    reference = _parse_instant(now) or datetime.now(UTC)
    live_sha = _live_covered_sha(toolkit_path)
    live_digest = _live_toolkit_src_digest(toolkit_path)
    entries = read_ledger(runs_root_path)

    if not entries:
        return {
            "owed": True,
            "reason": f"never-run: {ledger_path(runs_root_path)} carries no aggregated run",
            "covering_sha": None,
            "run_id": None,
            "complement": None,
            "missing_conjuncts": [],
        }

    # A DISCHARGE IS FOUND BY SCANNING, NOT BY READING THE NEWEST RECORD. A newer array-scoped
    # run must not hide an older union-scoped one that still satisfies the interval; the newest
    # record only supplies the REASON when nothing discharges.
    for entry in reversed(entries):
        classified = _classify(
            entry,
            live_sha=live_sha,
            live_digest=live_digest,
            reference=reference,
            interval_hours=interval_hours,
            runs_root=runs_root_path,
        )
        if not classified["reasons"]:
            return {
                "owed": False,
                "reason": (
                    "union-scoped GREEN at the live HEAD and working tree, inside the interval, "
                    "with both compile conjuncts and no missing conformance site"
                ),
                "covering_sha": classified["covered_sha"],
                "run_id": classified["run_id"],
                "complement": None,
                "missing_conjuncts": [],
            }

    newest = _classify(
        entries[-1],
        live_sha=live_sha,
        live_digest=live_digest,
        reference=reference,
        interval_hours=interval_hours,
        runs_root=runs_root_path,
    )
    return {
        "owed": True,
        "reason": "; ".join(newest["reasons"]),
        "covering_sha": None,
        "run_id": newest["run_id"],
        "complement": newest["complement"],
        "missing_conjuncts": newest["missing_conjuncts"],
    }
