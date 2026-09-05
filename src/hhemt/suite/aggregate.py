"""Aggregate per-chunk artifacts into one verdict for an hhemt suite run.

Two vocabularies, at two granularities, and they are not interchangeable.

CHUNK state
    PASS     ran to completion with no failing or unevaluated test
    FAIL     ran to completion; at least one test failed
    VOID     ran, but its own preconditions were violated, so its numbers are
             not evidence (source moved under it, two processes in the job
             resolved hhemt differently, the import guard exited 99, pytest
             reported INTERRUPTED/INTERNAL_ERROR/USAGE_ERROR, or SLURM killed it)
    MISSING  the manifest declares this chunk and no status sidecar exists

TEST outcome
    PASSED               the body ran and asserted true
    FAILED               the body ran and asserted false
    UNEVALUATED          setup or teardown raised, so the body never ran
    SKIPPED_STRUCTURAL   the environment forbids the assertion, so this site can never
                         evaluate it -- a scheduler-gated test inside a SLURM allocation
                         is the standing case. NOT benign: it is the complement the run
                         did not cover, and it is reported by node id.
    SKIPPED_INCIDENTAL   skipped for a local, non-structural reason (an optional
                         dependency absent). Benign.
    ABSENT               the manifest collected it and no junit entry mentions it

A run reports a SCOPE and never a bare verdict. `scope=array` means the run covered only
what a scheduler context can evaluate; `scope=union` is the only form that supports a
suite-level claim. Collapsing structural and incidental skips into one `skipped` count is
the same error as collapsing VOID into FAIL: it hides that two runs did not attempt the
same work.

MISSING is enumerated from the manifest rather than from the directory listing,
because a listing cannot report the absence of something it does not know to
expect. UNEVALUATED is kept apart from FAILED because a dead session fixture
takes down every test co-located with it, and "which test broke" is the wrong
question to invite about tests that never ran.
"""

from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

#: pytest exit codes that mean the session did not deliver a verdict.
NON_VERDICT_EXITS = {2, 3, 4}

#: Exit code the toolkit's repo-root import guard uses.
IMPORT_GUARD_EXIT = 99

#: Skip reasons that mean THIS SITE CAN NEVER EVALUATE THE TEST, as opposed to a
#: local, incidental absence. Exact-match on the junit skip message. Adding a pattern
#: widens the declared complement; it must never be widened to silence a real skip.
#:
#: EXACT-MATCH IS A CROSS-REPO COUPLING TO TOOLKIT PROSE, AND IT HAS ALREADY BROKEN ONCE.
#: The first entry below matched 17 tests until the toolkit deleted every mark carrying
#: it (see `tests/utils_for_testing.py::on_scheduler_node`, whose docstring records the
#: deletion). From 2026-08-25 this tuple matched ZERO tests, so `structurally_excluded`
#: read 0 and `summary.md` printed "None -- every collected test was evaluable at this
#: scope" while a scheduler-gated test sat in the INCIDENTAL bucket. The failure is
#: silent in the reassuring direction: a narrowed pattern makes the run look MORE
#: complete, not less. Nothing in this repo can detect the next rename; when a toolkit
#: `skipif` reason changes, this tuple must change with it.
STRUCTURAL_SKIP_REASONS: tuple[str, ...] = (
    "Only runs on non-HPC systems.",
    "Local coupled run-proof; do not launch on an HPC scheduler node.",
    "live-deposit run-proof; do not launch on an HPC scheduler node.",
)

#: SLURM states that invalidate a chunk regardless of what its junit says.
KILLED_STATES = {"TIMEOUT", "PREEMPTED", "CANCELLED", "NODE_FAIL", "OUT_OF_MEMORY"}


def _node_id(tc) -> tuple[str, bool]:
    """Reconstruct the pytest node id from a junit testcase element.

    pytest's junit splits an id across TWO attributes: `file` carries the path and
    `classname` carries the dotted MODULE path plus any enclosing class chain, while
    `name` carries only the final component. Building the id from `file` + `name`
    therefore DROPS every class segment, so `tests/x.py::TestC::test_m` is
    reconstructed as `tests/x.py::test_m` and matches no manifest entry.

    Returns (node_id, derivable). `derivable` is False when `classname` does not
    extend the module prefix implied by `file` -- the caller records those rather
    than letting an unreconstructable id fall silently into ABSENT, which is the
    failure this function exists to end.
    """
    f = tc.get("file") or ""
    name = tc.get("name") or ""
    cls = tc.get("classname") or ""
    if not f:
        return name, False
    prefix = str(Path(f).with_suffix("")).replace(os.sep, ".")
    if cls == prefix:
        return f"{f}::{name}", True
    if cls.startswith(prefix + "."):
        chain = cls[len(prefix) + 1 :].split(".")
        return f"{f}::" + "::".join([*chain, name]), True
    return f"{f}::{name}", False


def parse_junit(path: Path) -> dict[str, str]:
    """Map node id -> PASSED | FAILED | UNEVALUATED | SKIPPED_STRUCTURAL | SKIPPED_INCIDENTAL."""
    outcomes: dict[str, str] = {}
    root = ET.parse(path).getroot()
    for tc in root.iter("testcase"):
        node_id, _ = _node_id(tc)
        kinds = {c.tag for c in tc}
        if "error" in kinds:
            outcomes[node_id] = "UNEVALUATED"
        elif "failure" in kinds:
            outcomes[node_id] = "FAILED"
        elif "skipped" in kinds:
            sk = tc.find("skipped")
            msg = (sk.get("message") or "") if sk is not None else ""
            outcomes[node_id] = "SKIPPED_STRUCTURAL" if msg in STRUCTURAL_SKIP_REASONS else "SKIPPED_INCIDENTAL"
        else:
            outcomes[node_id] = "PASSED"
    return outcomes


def unevaluated_reasons(path: Path) -> dict[str, list[str]]:
    """Group unevaluated node ids by the first line of their error text."""
    groups: dict[str, list[str]] = defaultdict(list)
    root = ET.parse(path).getroot()
    for tc in root.iter("testcase"):
        err = tc.find("error")
        if err is None:
            continue
        node_id, _ = _node_id(tc)
        msg = (err.get("message") or (err.text or "")).strip().splitlines()
        groups[msg[0] if msg else "unknown"].append(node_id)
    return {k: sorted(v) for k, v in groups.items()}


#: What a slug-root entry IS, decided from raw evidence rather than from a name list.
#: A name list is a derivation with a horizon -- a staging dir added later would fall
#: through it and be reported as an analysis-tree race, which is the same over-report
#: this classifier exists to end, one size smaller. The discriminator is the toolkit's
#: OWN layout stamp: every analysis and system tree carries `_version.json`; the shared
#: compiled tier and the config staging dir do not. Measured on the slug cache
#: 2026-08-23: 13/13 trees carry it, 0/2 shared-by-design dirs do.
#:
#: The coupling is real and is made VISIBLE rather than assumed -- every classification
#: is reported WITH the evidence that produced it, so a stamp-convention change shows up
#: as a wrong label a reader can see, not as a silent reclassification. `aggregate` also
#: reports the all-shared-by-design case as an anomaly for the same reason.
ENTRY_ANALYSIS_TREE = "ANALYSIS_TREE"
ENTRY_SHARED_BY_DESIGN = "SHARED_BY_DESIGN"
ENTRY_TRANSIENT = "TRANSIENT"
ENTRY_UNKNOWN = "UNKNOWN"


def _is_coordination_path(rel: str) -> bool:
    """True when a write sample is a lock/marker rather than a build product.

    NAME-BASED, and that is a real limitation rather than an oversight -- it is the same
    fragile-string-coupling class this module already documents for
    STRUCTURAL_SKIP_REASONS. It is tolerable here because it fails in the SAFE direction:
    an unrecognised name is treated as PAYLOAD, so a new lock spelled without a leading
    dot produces a false BLOCKING (loud, investigable) rather than a false advisory
    (silent, and the direction that would defeat the verdict).

    Measured against the live corpus on 20260904T074157Z_cd4d8f6d9a91: the two lock paths
    actually written under `_software` are `.triton.provision.lock` and
    `triton/.build_tritonswmm_cpu.compile.lock`, and both match on either clause.
    """
    base = os.path.basename(rel)
    return base.startswith(".") or base.endswith(".lock")


def classify_slug_entry(rec: dict) -> str:
    """ANALYSIS_TREE | SHARED_BY_DESIGN | TRANSIENT | UNKNOWN, from recorded evidence."""
    is_dir = rec.get("is_dir")
    if is_dir is None:
        return ENTRY_UNKNOWN
    if not is_dir:
        # A plain file at the slug root: a `*.lock`, a stray config. Never a race surface.
        # `_version.json.lock` written by every chunk is the LOCK WORKING.
        return ENTRY_TRANSIENT
    stamp = rec.get("has_version_stamp")
    if stamp is None:
        return ENTRY_UNKNOWN
    return ENTRY_ANALYSIS_TREE if stamp else ENTRY_SHARED_BY_DESIGN


def classify_chunk(
    status: dict | None,
    junit: Path | None,
    pinned_sha: str,
    expected_fixtures: list[str] | None = None,
) -> tuple[str, list[str]]:
    if status is None:
        return "MISSING", ["no status sidecar for a chunk the manifest declares"]

    reasons: list[str] = []

    # A session fixture the manifest expected but that never completed setup is a
    # lost PRECONDITION, so the chunk is VOID and the response is to re-run it. A
    # test whose own function-scoped fixture raised is a defect, so the chunk stays
    # FAIL and the run is NOT-GREEN through the UNEVALUATED set instead.
    ok = set(status.get("session_fixtures_ok") or [])
    dead = sorted(set(expected_fixtures or []) - ok)
    if dead:
        reasons.append(f"expected session fixture(s) did not complete setup: {dead}")
    if status.get("source_sha_start") != pinned_sha:
        reasons.append(f"source_sha_start {status.get('source_sha_start')} != pinned {pinned_sha}")
    if status.get("source_sha_end") != status.get("source_sha_start"):
        reasons.append("source sha changed while the chunk was running")
    if status.get("dirty_start") or status.get("dirty_end"):
        reasons.append("source tree was dirty")

    resolutions = {v for v in (status.get("resolved_hhemt") or {}).values() if v}
    if len(resolutions) > 1:
        reasons.append(f"processes disagreed on hhemt.__file__: {sorted(resolutions)}")
    if not resolutions:
        reasons.append("no process recorded a resolved hhemt.__file__")

    rc = status.get("pytest_exit")
    if rc == IMPORT_GUARD_EXIT:
        reasons.append("import guard exited 99")
    if rc in NON_VERDICT_EXITS:
        reasons.append(f"pytest exit {rc} is INTERRUPTED/INTERNAL_ERROR/USAGE_ERROR")
    if str(status.get("slurm_state", "")).upper() in KILLED_STATES:
        reasons.append(f"SLURM state {status.get('slurm_state')}")
    if junit is None or not junit.exists():
        reasons.append("chunk produced no junit file")

    if reasons:
        return "VOID", reasons
    if rc == 0:
        return "PASS", []
    return "FAIL", []


def aggregate(run_dir: Path, scope: str = "array") -> dict:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    pinned = manifest["source_sha"]

    # SCOPE IS BOUND BY THE MANIFEST, NEVER BY THE CALLER ALONE. The run that BUILT this
    # directory recorded what it intended to cover; a downstream reader passing --scope
    # cannot know better, and the operator path proves it -- rerun.sh's --aggregate and
    # its verdict job pass no --scope at all, so a triage run aggregated through the
    # normal entry point would otherwise report `scope=array` and write `summary.json`,
    # defeating both unmisreadability properties at once. Binding here makes the property
    # structural rather than dependent on remembering a flag: there is no invocation of
    # this module that can produce a suite-shaped verdict from a triage manifest.
    scope_problems: list[str] = []
    declared = manifest.get("scope_intent")
    if declared == "triage" and scope != "triage":
        if scope == "union":
            scope_problems.append(
                "caller asked for scope=union on a manifest declaring scope_intent=triage; "
                "a triage run covers only the prior run's failing set and can never support "
                "a suite-level claim. Scope forced to triage."
            )
        scope = "triage"
    collected = set(manifest["collected"])

    chunk_rows = []
    statuses: dict[int, dict] = {}
    outcomes: dict[str, str] = {}
    reasons_by_fixture: dict[str, list[str]] = {}

    for c in manifest["chunks"]:
        cid = c["chunk_id"]
        sp = run_dir / f"chunk-{cid:02d}.status.json"
        jp = run_dir / f"chunk-{cid:02d}.junit.xml"
        status = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else None
        statuses[cid] = status or {}
        state, why = classify_chunk(status, jp if jp.exists() else None, pinned, c.get("expected_fixtures"))
        if jp.exists():
            outcomes.update(parse_junit(jp))
            for k, v in unevaluated_reasons(jp).items():
                reasons_by_fixture.setdefault(k, []).extend(v)
        # `diagnostics` is threaded so render_summary_md can NAME the directory. The
        # harvest has always written it and the summary has never mentioned it, which is
        # the whole defect: an unnamed directory gets read past, and that is what an
        # unnamed directory gets rather than a lapse by the reader.
        chunk_rows.append(
            {
                "chunk_id": cid,
                "kind": c["kind"],
                "state": state,
                "reasons": why,
                "diagnostics": (status or {}).get("diagnostics"),
            }
        )

    # MECHANISM READABILITY. A chunk that did not report contributes no observations, so
    # every cross-chunk number computed from the surviving chunks is a LOWER BOUND -- and
    # a low `derivation under-connected` is exactly what a missing chunk produces. Measured
    # 2026-08-23: a run whose 1013-test chunk timed out reported `under-connected: 1` over
    # five of six chunks; had it reported 0 it would have read as a pass and meant nothing.
    #
    # No new chunk state is minted: MISSING and VOID already say precisely this, and a
    # fifth token would duplicate them. What was missing is the DERIVED consequence and
    # its surfacing, so `mechanism_readable` is a predicate over the existing vocabulary.
    REPORTING_STATES = ("PASS", "FAIL")
    non_reporting = [
        {"chunk_id": r["chunk_id"], "state": r["state"]} for r in chunk_rows if r["state"] not in REPORTING_STATES
    ]
    mechanism_readable = not non_reporting

    def _of(kind: str) -> list[str]:
        return sorted(n for n, o in outcomes.items() if o == kind and n in collected)

    passed = _of("PASSED")
    failed = _of("FAILED")
    uneval = _of("UNEVALUATED")
    structural = _of("SKIPPED_STRUCTURAL")
    incidental = _of("SKIPPED_INCIDENTAL")
    absent = sorted(collected - set(outcomes))
    # Junit entries naming tests the manifest does not contain. Previously discarded
    # by the `n in collected` filter above, which is what made the id-shape defect
    # invisible: the same tests were counted ABSENT on one side and dropped on the
    # other, and neither number was reported next to its twin.
    unmatched_junit = sorted(set(outcomes) - collected)

    # Two-arm shared-tree check. Arm A is partition.py's manifest-time DERIVATION;
    # arm B is each chunk's runtime OBSERVATION. A tree seen by two chunks is a real
    # race -- concurrent sessions mutating one on-disk analysis tree. A tree OBSERVED
    # across chunks that arm A did not predict means the derivation under-connected,
    # which is the failure a deeper parse cannot remove: measured 2026-08-23, the reach
    # ran file -> fixture -> fixture -> helper module. Arm B is false-negative-safe (it
    # can miss a reach, never invent one), so a finding here is always real.
    #
    # CLASSIFY BEFORE JOINING. Arm B records slug-root ENTRIES, and the slug root holds
    # more than analysis trees. Joining before classifying is what produced the 5-for-1
    # over-report on the 2026-08-23 run: two of the five `concurrent BUILD` findings were
    # not directories at all, and one was the compiled tier that is shared BY DESIGN.
    observed_by: dict[str, set[int]] = {}
    written_by: dict[str, set[int]] = {}
    entry_evidence: dict[str, dict] = {}
    have_entries = False
    legacy_chunks: list[int] = []
    for c in manifest["chunks"]:
        st = statuses.get(c["chunk_id"], {})
        entries = st.get("reach_entries") or {}
        if entries:
            have_entries = True
            # KEYED ON THE ABSOLUTE ENTITY PATH the producer resolved -- `tree_root` when
            # the layout stamp resolved, else `p2`. NOT on the label: two chunks contend
            # only when they touched the SAME directory, and a label is not unique across
            # slugs or across per-session override roots. The `or p2` fallback is
            # load-bearing rather than defensive: `tree_root` is None for every unstamped
            # entry BY DEFINITION, so a bare `tree_root` key would collapse `_software`
            # and `_sensitivity_configs` into one bucket and report contention on a
            # directory that does not exist -- and it would do so in the quietest bucket.
            for key, rec in entries.items():
                ev = entry_evidence.setdefault(key, {})
                ev.setdefault("write_samples_by_chunk", {})
                ev.setdefault("votes", {})
                ev["label"] = rec.get("label") or os.path.basename(key)
                ev["root_shape"] = rec.get("root_shape")
                # Per-chunk votes are KEPT rather than merged away. A chunk that saw the
                # root vanish and one that did not now disagree VISIBLY, instead of one
                # False silently demoting the entry out of the contention check.
                ev["votes"][str(c["chunk_id"])] = {
                    k: rec.get(k) for k in ("is_dir", "has_version_stamp", "existed_at_first_open", "exists_at_end")
                }
                for k in ("is_dir", "has_version_stamp"):
                    prior = ev.get(k)
                    cur = rec.get(k)
                    # TRUE WINS across chunks, because each vote is now about the SAME
                    # absolute directory: a positive observation is evidence, a negative
                    # one may only mean the root was gone when that chunk stat'd it.
                    ev[k] = cur if prior is None else (bool(prior) or bool(cur) if cur is not None else prior)
                if rec.get("write_samples"):
                    ev["write_samples_by_chunk"][str(c["chunk_id"])] = rec["write_samples"]
                if rec.get("read"):
                    observed_by.setdefault(key, set()).add(c["chunk_id"])
                if rec.get("written"):
                    written_by.setdefault(key, set()).add(c["chunk_id"])
        else:
            if st.get("observed_trees"):
                legacy_chunks.append(c["chunk_id"])
            for tree in st.get("observed_trees") or []:
                observed_by.setdefault(tree, set()).add(c["chunk_id"])
            for tree in st.get("written_trees") or []:
                written_by.setdefault(tree, set()).add(c["chunk_id"])

    entry_class = {
        n: (classify_slug_entry(entry_evidence[n]) if n in entry_evidence else ENTRY_UNKNOWN)
        for n in sorted(set(observed_by) | set(written_by))
    }
    observed_cross = {t: sorted(ids) for t, ids in sorted(observed_by.items()) if len(ids) > 1}

    # THE JOIN. A cross-chunk reach alone does not say what KIND of race it is, and that
    # distinction cost a full round to reconstruct by hand: two chunks READING one tree
    # is benign, a chunk WRITING a tree another chunk reads is the race, and two chunks
    # WRITING it is a concurrent build. Recorded by observation, so it cannot inherit
    # arm A's horizon.
    def _contention(names) -> dict:
        return {
            t: {
                "readers": observed_cross[t],
                "writers": sorted(written_by.get(t, set())),
                "write_samples_by_chunk": (entry_evidence.get(t) or {}).get("write_samples_by_chunk", {}),
            }
            for t in names
            if t in observed_cross and written_by.get(t)
        }

    # The race class the harness exists to catch: two sessions mutating ONE analysis tree.
    contended_writes = _contention([t for t in observed_cross if entry_class.get(t) == ENTRY_ANALYSIS_TREE])
    # RECLASSIFIED, NOT EXCLUDED. The compiled tier is shared across every chunk by
    # design, so it is not an analysis-tree race -- but properties 1 and 2 say it is
    # written ONCE, by the warm, so a chunk writing it is still a finding and excluding it
    # would make that question permanently unanswerable. It gets its own label, its own
    # section, and the write samples that say WHAT was written.
    shared_by_design_writes = _contention([t for t in observed_cross if entry_class.get(t) == ENTRY_SHARED_BY_DESIGN])
    ignored_transient = sorted(t for t in observed_cross if entry_class.get(t) == ENTRY_TRANSIENT)
    unknown_entries = sorted(t for t in observed_cross if entry_class.get(t) == ENTRY_UNKNOWN)
    # An unclassified record still carries a real observation. Dropping it would trade an
    # over-report for an under-report, which is the worse direction: the reader would see a
    # notice and no findings and could reasonably conclude there were none.
    unclassified_writes = _contention(unknown_entries)

    # Arm A predicts TREES, so the under-connection check must compare against trees only.
    # Comparing the whole observed set against it made every shared-by-design entry and
    # every lock file read as "the derivation missed an edge" -- the same over-report,
    # a second consumer.
    predicted = {e["tree"] for e in manifest.get("shared_tree_exposure") or []}
    derivation_under_connected = sorted(
        t for t in observed_cross if entry_class.get(t) == ENTRY_ANALYSIS_TREE and t not in predicted
    )

    # Display names disambiguate ONLY when they need to. The absolute path is the
    # identity and is always in the write-surface table; repeating it in every problem
    # line buries the finding in noise, while dropping it entirely would hide exactly the
    # distinction this change bought -- two same-named directories are two entries.
    _label_counts: dict[str, int] = {}
    for _k, _e in entry_evidence.items():
        _label_counts[_e.get("label") or os.path.basename(_k)] = (
            _label_counts.get(_e.get("label") or os.path.basename(_k), 0) + 1
        )

    def _disp(key: str) -> str:
        lab = (entry_evidence.get(key) or {}).get("label") or os.path.basename(key)
        return lab if _label_counts.get(lab, 0) <= 1 else f"{lab} ({key})"

    entry_problems: list[str] = []
    # ADVISORY accumulator, sibling of entry_problems and NOT consulted by the verdict.
    # It exists because three findings below fire on a HEALTHY run and each of them says
    # so in its own comment. Measured 2026-09-04 against a synthetic run directory with 4
    # passing tests, both chunks `pytest_exit: 0`, failed=0 unevaluated=0 absent=0:
    # `verdict=NOT-GREEN`. An alarm that fires on every healthy run is not an alarm, and
    # the cost is not cosmetic -- a suite green is the gate on an ensemble launch, so a
    # constant NOT-GREEN forces the reader to evaluate the counts and ignore the word,
    # which is exactly the habit that makes a REAL NOT-GREEN invisible.
    entry_advisories: list[str] = []
    if legacy_chunks:
        entry_problems.append(
            f"reach records for chunk(s) {legacy_chunks} predate the write-surface "
            "classification (no 'reach_entries'), so their slug-root entries cannot be told "
            "apart from analysis trees. Their contention is listed under "
            "`unclassified_writes` and is "
            "known to over-report: locks, the shared compiled tier, and staging dirs all "
            "appear as trees. Re-run those chunks to classify."
        )
    if unknown_entries:
        entry_problems.append(
            f"{len(unknown_entries)} slug-root entr(ies) could not be classified: "
            f"{unknown_entries}. An unstattable entry is not a benign one; those with "
            f"cross-chunk writes are listed under `unclassified_writes` "
            f"({sorted(unclassified_writes)})."
        )
    disagreements = {
        k: e["votes"]
        for k, e in sorted(entry_evidence.items())
        if len({(v["is_dir"], v["has_version_stamp"]) for v in (e.get("votes") or {}).values()}) > 1
    }
    for k, votes in disagreements.items():
        entry_problems.append(
            f"chunks disagreed on what {_disp(k)} IS: {votes}. Resolved TRUE-wins (each vote is "
            "about the same absolute directory). A root seen as a stamped dir by one chunk "
            "and as absent by another was created or deleted mid-run; a difference in "
            "`existed_at_first_open` vs `exists_at_end` names which."
        )
    if non_reporting:
        _ids = [f"{d['chunk_id']:02d} ({d['state']})" for d in non_reporting]
        entry_problems.append(
            f"MECHANISM UNREADABLE: chunk(s) {', '.join(_ids)} produced no usable record, so "
            "every cross-chunk number below is a LOWER BOUND computed over the chunks that "
            "did report. `derivation under-connected` and the contended-write sets in "
            "particular CANNOT be read as a pass -- a missing chunk lowers both. Re-run the "
            "non-reporting chunk(s) before reading the mechanism."
        )
    dirs = [n for n, e in entry_evidence.items() if e.get("is_dir")]
    if have_entries and dirs and not any(e.get("has_version_stamp") for e in entry_evidence.values()):
        # ADVISORY. This is a self-check on the CLASSIFIER, not a finding about the run --
        # its own text says so ("this classifier is reporting zero analysis trees for the
        # wrong reason"). It also fires on a SINGLE-chunk run and on a run where nothing was
        # written at all, because it is gated on neither `observed_cross` nor `written_by`;
        # measured 2026-09-04, it is the reason even the triage shape could not reach GREEN.
        # What is given up: a genuine `_version.json` relocation no longer fails a run. It
        # still PRINTS, and a stamp move would break the toolkit's own layout-version CI
        # first, so this is the cheapest of the three demotions.
        entry_advisories.append(
            f"[classifier-self-check] no observed slug-root directory carries "
            f"`{'_version.json'}` -- every one of {sorted(dirs)} classified "
            "SHARED_BY_DESIGN. If the layout stamp moved, this classifier is reporting "
            "zero analysis trees for the wrong reason. ADVISORY: this describes the "
            "instrument, not the run."
        )

    problems: list[str] = list(scope_problems) + list(entry_problems)
    # Everything BELOW that stays in `problems` is verdict-bearing and unchanged. Only the
    # three sites this spec set names move, and each moves for a reason its own comment
    # already gave. `scope_problems` is NOT demoted: a caller asking for a scope the
    # manifest forbids is a misuse of the instrument, which is a different thing from the
    # instrument reporting on itself.
    advisories: list[str] = list(entry_advisories)
    for r in chunk_rows:
        if r["state"] != "PASS":
            problems.append(f"chunk {r['chunk_id']:02d} is {r['state']}: {'; '.join(r['reasons']) or 'see junit'}")
    if failed:
        problems.append(f"{len(failed)} test(s) FAILED")
    if uneval:
        problems.append(f"{len(uneval)} test(s) UNEVALUATED (setup or teardown raised; the body never ran)")
    if absent:
        problems.append(f"{len(absent)} collected test(s) ABSENT from every junit file")
    # A test cannot be simultaneously MISSING from the junit and EXTRA in it. When both
    # sets are non-empty the two sides are naming the same tests in different id shapes,
    # which is a REPORTING defect, not a coverage one -- and it is self-concealing,
    # because `absent` can never reach zero and the verdict stays NOT-GREEN forever with
    # no failing test to chase.
    # A cross-chunk reach with NO observed writer is concurrent READING, which is
    # benign, so it is recorded in the result and does NOT fail the run. Round 18 made
    # any cross-chunk reach a problem, which was correct only while the reach was
    # undifferentiated: failing every concurrent read would make the verdict fire for a
    # reason other than the one it names -- the defect this harness exists to close.
    # CAVEAT kept in view: arm B is false-negative-safe, so "no writer observed" is not
    # "no writer"; a write through a C-level open is invisible to it.
    for t, d in contended_writes.items():
        kind = (
            "concurrent BUILD (two or more chunks wrote it)"
            if len(d["writers"]) > 1
            else "write/read race (one chunk wrote it while others read)"
        )
        problems.append(f"shared-tree {kind}: {_disp(t)} -- writers {d['writers']}, touched by {d['readers']}")
    # Reported, never silently dropped, and NOT as an analysis-tree race. It is also not
    # a run failure: the tier is shared by construction, and whether these writes are a
    # concurrent build or mere lock acquisition is what the samples are for. Failing the
    # run on an undiagnosed reading would block the green gate on instrument noise -- the
    # thing this change exists to stop.
    #
    # THE COMMENT ABOVE AND THE CODE BELOW DISAGREED UNTIL 2026-09-04: the comment said
    # "not a run failure", the code appended to `problems`, and a non-empty `problems` is
    # NOT-GREEN. The split below makes the code say what the comment always said.
    #
    # THE DISCRIMINATOR IS WHAT WAS WRITTEN, NOT HOW MANY CHUNKS WROTE IT, and the writer
    # count is disqualified by measurement rather than by taste. On BOTH the
    # 20260904T074157Z_cd4d8f6d9a91 and 20260904T145255Z_2853a077a3ee runs, `_software`
    # has FOUR writers and `_sensitivity_configs` has two -- so a `len(writers) > 1`
    # threshold marks the steady state of every healthy multi-chunk run as BLOCKING and
    # reproduces the constant NOT-GREEN this whole change removes.
    #
    # The samples say what the count cannot. Measured on the same runs: all four
    # `_software` writers wrote ONLY `.triton.provision.lock` and
    # `triton/.build_tritonswmm_cpu.compile.lock` -- pure lock acquisition, which is
    # `system.py::_compile_backend` taking a lock and skipping at the completion marker
    # inside it. `_sensitivity_configs`' two writers wrote DISJOINT `.csv` files.
    #
    # Hence BOTH halves, because each alone is refuted by that data. Same-path alone marks
    # concurrent LOCK acquisition as a collision (all four wrote the identical lock path).
    # Not-a-lock alone marks two chunks materializing DIFFERENT config files as a race.
    # A real concurrent build -- two chunks both writing
    # `triton/build_tritonswmm_cpu/compilation.log` -- collides on a payload path and is
    # the case properties 1 and 2 forbid, so it stops the run.
    for t_, d in shared_by_design_writes.items():
        _by_chunk = d["write_samples_by_chunk"] or {}
        _payload = {cid: {s for s in ss if not _is_coordination_path(s)} for cid, ss in _by_chunk.items()}
        _collided = sorted(
            {
                s
                for cid, ss in _payload.items()
                for s in ss
                if any(s in other for oc, other in _payload.items() if oc != cid)
            }
        )
        # FAIL CLOSED ON ABSENT EVIDENCE. A writer with no recorded sample cannot be
        # exonerated: the whole predicate above reasons from WHAT was written, so a writer
        # that contributed nothing to reason from is unclassified, not benign. Without this
        # clause an empty `write_samples_by_chunk` makes `_collided` empty and the entry
        # reads ADVISORY -- measured 2026-09-04, `[[]] * 4` returned GREEN.
        #
        # NOT REACHABLE TODAY, and that is stated rather than hidden so the next reader does
        # not waste an afternoon constructing it: `run_suite.py`'s audit hook appends a
        # sample on the FIRST write, so `written=True` implies at least one sample, and the
        # LEGACY reach shape keys `written_by` on the bare NAME while the modern shape keys
        # on the ABSOLUTE PATH -- measured, they never join, so a legacy writer cannot
        # attach to a SHARED_BY_DESIGN entity. This clause is therefore a CONTRACT test on
        # an undocumented cross-module invariant (a hook in run_suite.py guaranteeing an
        # input property this consumer relies on), not a regression test on an observed
        # failure. It is kept because that invariant is undocumented, lives in a different
        # module from its consumer, and would break silently: a future hook that records a
        # write without a path turns a concurrent build into an advisory.
        _unwitnessed = sorted(c for c in d["writers"] if not _by_chunk.get(str(c)))
        _sbd = (
            f"shared-by-design entry written by chunk(s) {d['writers']} (touched by "
            f"{d['readers']}): {_disp(t_)} -- shared across chunks by construction, so NOT an "
            "analysis-tree race; a chunk writing it still wants explaining (for `_software`, "
            "properties 1 and 2 say the compiled tier is written ONCE, by the warm). "
            f"Samples: {d['write_samples_by_chunk'] or '(none recorded)'}"
        )
        if _collided or _unwitnessed:
            _why = (
                f"two or more chunks wrote the SAME non-coordination path(s) {_collided}"
                if _collided
                else f"chunk(s) {_unwitnessed} are recorded as writers with NO write sample, "
                "so this entry cannot be shown to be lock acquisition"
            )
            problems.append(
                f"[concurrent-build] {_sbd} BLOCKING: {_why} under a tier properties 1 and 2 say is written once."
            )
        else:
            advisories.append(
                f"[shared-by-design-write] {_sbd} ADVISORY: no two chunks wrote the same "
                "non-coordination path, so these are lock acquisitions and/or disjoint "
                "writes rather than a concurrent build. NOTE: `write_samples` is capped at "
                "5 per chunk and records no flag when the cap binds, so a collision beyond "
                "the cap is not visible here."
            )
    # DEMOTED VISIBLY, and the visibility is the point rather than a courtesy. Arm B is the
    # run's ONLY detector of an under-connected partition, so demoting it removes a real
    # capability and the emitted summary must say so where a reader will meet it -- not in
    # this comment, which no operator reads. The reason for the demotion is a suspected
    # instrument defect rather than a judgement that under-connection stopped mattering:
    # arm A extracts tree names from SYMBOLS (`retrieve_synth_TRITON_SWMM_test_case` yields
    # `synth_TRITON_SWMM`) while arm B observes `analysis_name=` LITERALS (`synth_multi_sim`),
    # and on the 20260904T074157Z_cd4d8f6d9a91 baseline those two sets were DISJOINT -- so
    # `t not in predicted` can be true for every observed tree for a reason that has nothing
    # to do with the partition. Until that is settled this fires on healthy runs.
    # RESTORE TO BLOCKING once arm A and arm B are shown to share a vocabulary; the text
    # below names that condition so the restoration is not left to memory.
    if derivation_under_connected:
        advisories.append(
            f"[derivation-under-connected] {len(derivation_under_connected)} tree(s) were "
            f"OBSERVED across chunks that the manifest did not predict -- "
            f"{[_disp(x) for x in derivation_under_connected]}. Arm A's static scan missed an edge; the "
            "reach is real because arm B observes rather than derives. "
            "ADVISORY, AND THIS IS A CAPABILITY GAP, NOT A CLEAN BILL: arm B is the only "
            "under-connection detector this run has, and it is advisory only because arm A "
            "and arm B may not share a tree vocabulary (arm A reads symbol names, arm B "
            "reads `analysis_name=` literals; measured disjoint on the "
            "20260904T074157Z_cd4d8f6d9a91 baseline). While this line is advisory, an "
            "under-connected partition CANNOT fail a run."
        )
    if absent and unmatched_junit:
        problems.append(
            f"id-shape mismatch: {len(absent)} collected test(s) have no junit entry AND "
            f"{len(unmatched_junit)} junit entr(ies) name tests absent from the manifest. "
            "These are the same tests under different id shapes; the tests RAN. "
            f"e.g. collected-only {absent[0]!r} vs junit-only {unmatched_junit[0]!r}"
        )

    # Every collected node id lands in exactly one bucket. A violation means the
    # buckets no longer describe the universe, so no verdict computed from them is
    # trustworthy -- it is itself a NOT-GREEN reason rather than an assertion error.
    total = len(passed) + len(failed) + len(uneval) + len(structural) + len(incidental) + len(absent)
    if total != len(collected):
        problems.append(f"partition identity violated: buckets sum to {total} but {len(collected)} were collected")

    # BLOCKING ONLY. `advisories` is deliberately absent from this expression: that absence
    # IS the fix, and it is why `advisories=` is emitted on the verdict line below rather
    # than left to the summary. The adjacency rule is the one this module already states
    # for `scope` at the CLI print -- a reader cannot see the verdict word without seeing
    # the advisory count, because ONE f-string produces both. A third verdict token
    # (`GREEN-WITH-ADVISORIES`) was rejected: it would solve the same hazard by a different
    # mechanism than the one this file already uses for it, and it would read as a failure
    # to anyone who has only ever seen two tokens.
    verdict = "GREEN" if not problems else "NOT-GREEN"
    return {
        "scope": scope,
        "verdict_line": (
            f"verdict={verdict} scope={scope} run_id={manifest['run_id']} "
            f"covered={len(passed)} structurally_excluded={len(structural)} "
            f"failed={len(failed)} unevaluated={len(uneval)} absent={len(absent)} "
            f"collected={len(collected)} "
            # ALWAYS emitted, including at zero, exactly as `mechanism=` is. A count that
            # appears only when non-zero teaches a reader to skim for its absence, and an
            # absent count is indistinguishable from a reporting layer too old to emit it
            # -- the same defect the README records for a bare `GREEN`.
            f"advisories={len(advisories)} "
            f"mechanism={'readable' if mechanism_readable else 'UNREADABLE'}"
        ),
        "mechanism_readable": mechanism_readable,
        "non_reporting_chunks": non_reporting,
        "run_id": manifest["run_id"],
        "source_sha": pinned,
        "verdict": verdict,
        "problems": problems,
        # Persisted as its own key rather than merged into `problems`, so a later reader of
        # summary.json can tell a demoted finding from a blocking one WITHOUT re-deriving
        # the classification from the text. The `[kind]` prefixes each advisory carries are
        # for humans; this key is for programs.
        "advisories": advisories,
        "chunks": chunk_rows,
        "counts": {
            "collected": len(collected),
            "covered": len(passed),
            "structurally_excluded": len(structural),
            "incidental_skips": len(incidental),
            "failed": len(failed),
            "unevaluated": len(uneval),
            "absent": len(absent),
            "unmatched_junit": len(unmatched_junit),
        },
        "unmatched_junit": unmatched_junit,
        "observed_cross_chunk_trees": observed_cross,
        "slug_entry_classification": entry_class,
        "slug_entry_evidence": entry_evidence,
        "contended_writes": contended_writes,
        "shared_by_design_writes": shared_by_design_writes,
        "ignored_transient_entries": ignored_transient,
        "unclassified_entries": unknown_entries,
        "unclassified_writes": unclassified_writes,
        "entry_disagreements": disagreements,
        "derivation_under_connected": derivation_under_connected,
        "structurally_excluded": structural,
        "incidental_skips": incidental,
        "failed": failed,
        "unevaluated_now_passing": sorted(set(manifest.get("from_unevaluated") or []) & set(passed)),
        "unevaluated": uneval,
        "unevaluated_by_cause": {k: sorted(v) for k, v in sorted(reasons_by_fixture.items())},
        "absent": absent,
    }


def render_summary_md(result: dict) -> str:
    triage = result["scope"] == "triage"
    lines = [
        # The heading is part of the verdict surface. "suite run" over a 29-test triage
        # is the same misreading the scope token exists to prevent, one line higher up.
        f"# hhemt {'TRIAGE run (NOT a suite result)' if triage else 'suite run'} {result['run_id']}",
        "",
        "```text",
        result["verdict_line"],
        "```",
        "",
        f"- source sha: `{result['source_sha']}`",
        f"- counts: {result['counts']}",
        "",
    ]
    # The banner goes ABOVE the chunk table, because a reader who scrolls to the mechanism
    # numbers has already passed this point. It states what the numbers ARE rather than
    # only that something went wrong -- lower bounds, not results.
    if not result.get("mechanism_readable", True):
        _nr = ", ".join(f"{d['chunk_id']:02d} ({d['state']})" for d in result.get("non_reporting_chunks") or [])
        lines += [
            "",
            "> ## MECHANISM UNREADABLE — do not read the cross-chunk numbers as a result",
            ">",
            f"> Chunk(s) **{_nr}** produced no usable record. Every cross-chunk number in this",
            "> summary — `derivation under-connected`, the contended-write sets, the observed",
            "> shared-cache surface — was computed over the chunks that DID report, and is",
            "> therefore a **lower bound**, not a measurement.",
            ">",
            "> This matters most where it looks best: a missing chunk contributes no",
            "> observations, so it can only LOWER an under-connection count. A `0` here would",
            "> be indistinguishable from a clean run and would mean nothing. Re-run the",
            "> non-reporting chunk(s) before reading the mechanism.",
        ]
    lines += [
        "",
        "## Chunks",
        "",
        "| chunk | kind | state | reasons |",
        "|---|---|---|---|",
    ]
    for r in result["chunks"]:
        lines.append(f"| {r['chunk_id']:02d} | {r['kind']} | {r['state']} | {'; '.join(r['reasons'])} |")
    # WHERE THE REASONS ARE. Emitted before Problems, because a reader who has just seen a
    # failure needs the path more than the restatement. Gated on presence: a run with no
    # failing chunk harvests nothing and says nothing.
    diag = [r for r in result["chunks"] if r.get("diagnostics")]
    if diag:
        lines += [
            "",
            "## Failure diagnostics captured",
            "",
            "Node-local logs copied out of each failing chunk's basetemp before the node",
            "released it. **The root cause of a chunk failure normally lives here, not in",
            "this summary** — a rule's own log is what a snakemake error points at.",
            "",
            "| chunk | dir | files | truncated | dropped | note |",
            "|---|---|---|---|---|---|",
        ]
        for r in diag:
            d = r["diagnostics"]
            # NAME A DIRECTORY ONLY WHEN ONE EXISTS. `_harvest_diagnostics` returns EARLY on
            # an absent basetemp -- before the `dest.mkdir` at its tail -- so a chunk that
            # failed without any test having used tmp_path has a diagnostics RECORD and no
            # diagnostics DIRECTORY. Printing the path unconditionally would send a reader to
            # a path that does not exist, which is the same defect this section exists to fix,
            # one level in. Caught by running it: chunk 00 of the verification run.
            wrote = bool(d.get("harvested"))
            where = f"`{d.get('dir', '?')}/`" if wrote else "— (nothing written)"
            note = (
                d.get("error")
                or d.get("note")
                or (f"**{d.get('dropped')} dropped to a cap**" if d.get("dropped") else "")
            )
            lines.append(
                f"| {r['chunk_id']:02d} | {where} | {d.get('harvested', 0)} "
                f"| {d.get('truncated', 0)} | {d.get('dropped', 0)} | {note} |"
            )
        lines += [
            "",
            "Filenames are path-flattened (`/` -> `__`) against the chunk basetemp; the",
            "`manifest.json` beside them carries each file's ORIGINAL path, and records any",
            "cap that bound rather than applying it silently.",
        ]

    if result["problems"]:
        lines += ["", "## Problems", ""]
        lines += [f"- {p}" for p in result["problems"]]
    # RENDERED UNCONDITIONALLY when non-empty, with a standing preamble. The preamble is
    # not decoration: an advisory section that lists findings without saying what is
    # therefore NOT being detected reads as "nothing to look at", which is the same
    # failure class as the constant NOT-GREEN this split removes.
    if result.get("advisories"):
        lines += [
            "",
            "## Advisories — reported, not verdict-bearing",
            "",
            "These did NOT affect the verdict. Two of the three kinds below describe the "
            "INSTRUMENT rather than the run, and one names a capability that is currently "
            "switched off:",
            "",
            "- `[classifier-self-check]` — the slug-entry classifier reporting on itself.",
            "- `[shared-by-design-write]` — ONE chunk wrote a shared-by-construction entry "
            "(a lock being taken). TWO or more writing it is BLOCKING and appears under "
            "Problems instead.",
            "- `[derivation-under-connected]` — **arm B is this run's only detector of an "
            "under-connected partition, and while it is advisory an under-connected "
            "partition cannot fail a run.** It is advisory because arm A and arm B may not "
            "share a tree vocabulary, not because under-connection stopped mattering.",
            "",
        ]
        lines += [f"- {a}" for a in result["advisories"]]
    if result["unevaluated_by_cause"]:
        lines += ["", "## Unevaluated, grouped by cause", ""]
        for cause, nodes in result["unevaluated_by_cause"].items():
            lines.append(f"- **{cause}** ({len(nodes)} test(s))")

    if triage:
        now_passing = result.get("unevaluated_now_passing") or []
        if now_passing:
            lines += [
                "",
                f"## Unevaluated in the source run, PASSING here — {len(now_passing)} test(s)",
                "",
                "```text",
                "N test(s) were UNEVALUATED in the source run and PASS here. A session fixture that",
                "does not fail in isolation is not evidence the fixture is fixed -- it failed in a",
                "concurrent array, and triage runs one session. Confirm in a full run.",
                "```",
                "",
                "```text",
                *now_passing,
                "```",
            ]
        lines += [
            "",
            "## What this triage result does NOT establish",
            "",
            "- Nothing about the tests it did not run, including whether the fix under test broke one.",
            "- Nothing about concurrency: triage runs a SINGLE session, so it cannot reproduce a",
            "  cross-chunk shared-tree race at all -- that entire class is invisible to it by construction.",
            "- Nothing durable: it is a statement about these node ids, at this sha, in this configuration.",
            "",
            "The claim it DOES support is narrow and worth having: the tests that were failing are no",
            "longer failing. Only `scope=union` supports a suite-level claim; this run is `scope=triage`.",
        ]

    # The write surface, with the EVIDENCE beside each label. Printing the reason is what
    # keeps the classifier from becoming its own silent horizon: a stamp-convention change
    # shows up here as a visibly wrong label rather than as a quietly reclassified entry.
    if result.get("slug_entry_classification"):
        lines += [
            "",
            "## Shared-cache write surface",
            "",
            "| entry | shape | class | evidence | readers | writers |",
            "|---|---|---|---|---|---|",
        ]
        ev_all = result.get("slug_entry_evidence") or {}
        cw = result.get("contended_writes") or {}
        sbd = result.get("shared_by_design_writes") or {}
        for name, klass in sorted(result["slug_entry_classification"].items()):
            ev = ev_all.get(name) or {}
            why = f"is_dir={ev.get('is_dir')}, _version.json={ev.get('has_version_stamp')}"
            row = cw.get(name) or sbd.get(name) or {}
            # Label AND path. The label is what a reader recognizes; the path is what the
            # join actually keyed on, and printing only the label would hide exactly the
            # distinction that made two same-named directories one bucket.
            label = ev.get("label") or os.path.basename(name)
            lines.append(
                f"| `{label}`<br/>`{name}` | {ev.get('root_shape', '-')} | {klass} | {why} "
                f"| {row.get('readers', '')} | {row.get('writers', '')} |"
            )
        # THE CAVEAT IS GENERATED WITH THE FINDING IT QUALIFIES, not documented elsewhere.
        # A rule in a README is a rule someone has to remember to apply, which is the same
        # shape as a guard nobody reads; this one is rendered where the reader is already
        # standing and cannot be skipped.
        #
        # GATED ON THE CLASS, not on `root_shape == "indeterminate"`. Those are not the
        # same set: `indeterminate` means neither candidate carried the stamp, which is
        # ALSO true of every TRANSIENT entry (a plain file has no `_version.json` under
        # it) -- measured, a `_version.json.lock` classifies TRANSIENT with
        # root_shape=indeterminate. Gating on the shape would print this caveat on a run
        # whose only indeterminate rows are lock files, qualifying nothing.
        #
        # The named examples below are PROSE, telling the reader which rows have
        # independent backing. They are NOT a classifier input and must never become one:
        # a name list is the derivation-with-a-horizon this classifier was built to avoid,
        # and wiring these two names into the classification would reintroduce it.
        if any(k == ENTRY_SHARED_BY_DESIGN for k in result["slug_entry_classification"].values()):
            lines += [
                "",
                "> **`SHARED_BY_DESIGN` is an ABSENCE of evidence, not evidence of design.**",
                "> A row carries that label because no `_version.json` was found under it — which is",
                "> also what a mis-resolved or mis-stat'd entry looks like. Two rows are corroborated",
                "> INDEPENDENTLY of this classifier: `_software` (this README states the compiled tier",
                "> is deliberately shared, and `test_case_builder.py` pins `_software_root` to the slug",
                "> root outside the branch that chooses the runs-root) and `_sensitivity_configs` (a",
                "> config staging dir, `test_case_catalog.py:843`). **Any OTHER entry above with this",
                "> label is UNCLASSIFIED, not benign** — treat it as a finding awaiting explanation.",
                ">",
                "> There is no structural marker for *shared by design* the way `_version.json` is one",
                "> for *analysis tree*, because the class is defined by the absence of a marker. This",
                "> caveat is the honest ceiling, not a placeholder for a stronger control.",
            ]

        if result.get("ignored_transient_entries"):
            lines += [
                "",
                "Ignored as transient (a plain file at the slug root, not a race surface): "
                + ", ".join(f"`{n}`" for n in result["ignored_transient_entries"])
                + ". A `*.lock` written by every chunk is the lock WORKING.",
            ]
        for name, d in (result.get("shared_by_design_writes") or {}).items():
            lines += [
                "",
                f"### `{(result.get('slug_entry_evidence') or {}).get(name, {}).get('label') or name}`"
                f" — shared by design, written by chunk(s) {d['writers']}",
                "",
                "```text",
                *[f"chunk {cid}: {paths}" for cid, paths in sorted(d.get("write_samples_by_chunk", {}).items())],
                "```",
                "",
                "Samples are capped, so this says WHAT was written, not everything that was.",
            ]

    # The complement, BY NODE ID. When scope != union this list is the only record of
    # what the gate did not cover, so it is written in full and never summarized to a
    # count: a count cannot be re-run, and a marker expression is not the same set.
    lines += ["", f"## Complement — {len(result['structurally_excluded'])} test(s) this scope cannot evaluate", ""]
    if result["structurally_excluded"]:
        lines += [
            "Run these where the structural exclusion does not apply. Copy the block verbatim:",
            "",
            "```text",
            *result["structurally_excluded"],
            "```",
        ]
    else:
        lines.append("None — every collected test was evaluable at this scope.")

    if result["incidental_skips"]:
        lines += [
            "",
            f"## Incidental skips — {len(result['incidental_skips'])} test(s), benign",
            "",
            "```text",
            *result["incidental_skips"],
            "```",
        ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Aggregate one hhemt suite run into a verdict.")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument(
        "--scope",
        default="array",
        choices=("array", "complement", "union", "triage"),
        help="what this run covered; only 'union' supports a suite-level claim",
    )
    ap.add_argument(
        "--allow-not-green",
        action="store_true",
        help="exit 0 even when the verdict is NOT-GREEN (reporting only)",
    )
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir).resolve()
    result = aggregate(run_dir, scope=args.scope)
    # THE FILENAME SPLIT, keyed on the BOUND scope rather than on args.scope, because the
    # manifest can override the caller. Without it a triage run in a fresh dir writes a
    # file that any later reader -- including run_suite.py's own --from-run default
    # selection -- picks up as a suite summary. The split makes that impossible rather
    # than unlikely, and it is the second of the two independent arms guarding that path
    # (the first is the `.triage` run-dir suffix).
    stem = "summary.triage" if result["scope"] == "triage" else "summary"
    (run_dir / f"{stem}.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / f"{stem}.md").write_text(render_summary_md(result), encoding="utf-8")

    # The scope token is emitted on the SAME line as the verdict, always. A bare
    # GREEN is unprintable by construction: there is no format string that produces
    # one, because verdict_line is built with scope= adjacent to verdict=.
    print(result["verdict_line"])
    print(f"counts={result['counts']}")
    for p in result["problems"]:
        print(f"  problem: {p}")
    # Printed on the SAME surface as the problems, under a label that cannot be mistaken
    # for one. An advisory the operator never sees is a suppression wearing a different
    # name, and the whole argument for demoting rather than deleting these findings is
    # that they stay visible.
    for a in result.get("advisories") or []:
        print(f"  advisory: {a}")
    print(f"summary={run_dir / (stem + '.md')}")
    if result["verdict"] == "GREEN" or args.allow_not_green:
        return 0
    return 1


if __name__ == "__main__":  # pragma: no cover - module-execution path
    # `main()` returns 0 (GREEN, or --allow-not-green) / 1 (NOT-GREEN). The Typer
    # wrapper must convert with `raise typer.Exit(code=...)`; a bare `return` exits 0.
    raise SystemExit(main())
