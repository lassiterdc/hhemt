"""Login-node driver and chunk executor for the hhemt pytest suite on SLURM.

Two modes in one file, because the second mode is also a pytest plugin and the
resolution record it writes is the whole reason this harness exists.

  --plan (default)  On the LOGIN node: pin the source, warm the shared borrow and
                    compile SERIALLY, collect the test universe, build the chunk
                    manifest, and PRINT the run dir and chunk count. It does NOT
                    call sbatch: the submit script owns cluster knowledge (account,
                    partition, module stack) that this module has no business
                    duplicating, and the chunk count crosses that seam as DATA read
                    from the manifest rather than as a parameter either side may
                    choose. A count chosen independently of the manifest silently
                    reverts fixture-closure chunking to a count-based slice.
  --chunk N         Inside one array element: run chunk N's node ids and write a
                    status sidecar recording, among other things, the resolved
                    hhemt.__file__ of BOTH the pytest session and a sibling
                    process that did not load the repo-root conftest.

Why the two resolutions. The toolkit's repo-root conftest prepends its own src to
sys.path BEFORE it imports hhemt to check the resolution, so inside pytest the
check cannot fail; a sibling process in the same job has no such prepend and can
resolve elsewhere. Recording both and requiring agreement is what makes a
cross-process divergence visible. aggregate.py VOIDs a chunk whose recorded
resolutions disagree.

Cache invariant: one run, one slug, one _software. This driver never sets
XDG_CACHE_HOME and never runs a chunk from a per-chunk directory, because either
would relocate the hhemt cache root and restore a per-chunk clone and compile.

Ordering invariant: A RUN DIR EXISTING IMPLIES A WARM SUCCEEDED. The run id is
minted only after the warm returns clean, so a failed warm leaves no directory at
all rather than a manifest-less one. That distinction is not tidiness: aggregate.py
reads manifest.json first, so a directory without it raises rather than reporting,
and the failure that has no verdict is the one nobody sees.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

#: xunit1 is required: xunit2 (the pytest default) omits the `file` attribute, and
#: without it a junit entry cannot be mapped back to a collected node id.
JUNIT_ARGS = ("-o", "junit_family=xunit1")

#: Diagnostics harvested from a FAILING chunk's pytest basetemp. The heavy fixtures
#: nest their working trees under `tmp_path_factory`, which is node-local, so a
#: snakemake log that explains a dry-run failure dies with the compute node and its
#: REASON is unreachable from the run dir. Harvest is failure-only: a passing chunk
#: copies nothing.
#: WIDENED from `("snakemake*.log",)` 2026-08-23. The narrow glob captured the snakemake
#: log and NOT the rule log that log points at: a failing rule renders
#: `Error in rule render_report: log: .../logs/render_report_zip.log`, which does not match
#: `snakemake*.log`. Measured on a real cluster failure -- 40 files harvested, 0 dropped, and
#: the root cause absent. A captured log whose entire content is a pointer to a file that was
#: NOT captured is worse than no artifact, because it reads as diagnostic and is not.
#:
#: Broad rather than precise, on FAILURE DIRECTION rather than cost. Following the named
#: `log:` path would capture exactly the right file and fail SILENTLY whenever a rule has no
#: log declaration, snakemake's error format changes, or the useful log belongs to an
#: adjacent rule -- and in each case the diagnostics dir looks complete. It would also add a
#: format parse: a second instrument with its own way of being wrong for a reason other than
#: the one it names. The glob's only failure mode is volume, volume is capped, and every cap
#: that binds is RECORDED with the file it dropped. Measured headroom: 40 of a 200-file cap.
#:
#: `snakemake*.log` is subsumed by `*.log` and `found` is deduped, so listing it separately
#: buys nothing; HARVEST_PRIORITY below still names the snakemake tiers, which is where the
#: intent belongs. HARVEST_PRIORITY needs NO edit: `_rank`'s `for...else` already assigns an
#: unmatched file `i = len(HARVEST_PRIORITY)`, so a rule log sits in the last tier today and
#: sits in the last tier after -- verified by simulating both orderings, which is what turned
#: this from a several-line change into a one-line one. Within that tier the tie-break is
#: most-recently-modified, and the failing rule's log is normally the last thing written.
HARVEST_GLOBS: tuple[str, ...] = ("*.log",)

#: PRIORITY ORDER, most diagnostic first. Under a cap, ORDER IS POLICY: a cap applied
#: to a lexicographic listing drops whatever sorts last, which has nothing to do with
#: what explains the failure. Measured on the first test of this harvest -- 205 filler
#: logs named `snakemake_extra_*` sorted AHEAD of `snakemake_master_dry_run.log`, so
#: the count cap kept 200 useless files and dropped the only one carrying the reason,
#: while every cap was respected and the manifest was honest. A file is ranked by the
#: index of the FIRST pattern it matches; ties break on most-recently-modified, since
#: the failure is normally the last thing written.
HARVEST_PRIORITY: tuple[str, ...] = (
    "*dry_run*.log",
    "snakemake_master*.log",
    "snakemake*.log",
)

#: Caps, grounded on the 24 snakemake logs measured on the maintainer host
#: (min 525 B, median 53,712 B, max 71,457 B). The per-file cap is ~29x the largest
#: ever observed, so a file that trips it is pathological rather than merely large;
#: truncation keeps head AND tail so both the invocation banner and the failure
#: survive. The count cap binds first in every realistic case (25 MiB is ~475 logs at
#: the measured median); the byte cap is the backstop.
HARVEST_MAX_FILE_BYTES = 2 * 1024 * 1024
HARVEST_MAX_FILES = 200
HARVEST_MAX_TOTAL_BYTES = 25 * 1024 * 1024

#: Where the chunk-mode pytest plugin writes the session's resolved module path.
RESOLUTION_ENV = "HHEMT_SUITE_RESOLUTION_OUT"

#: Written by the plugin below; consumed by _write_status.
FIXTURES_ENV = "HHEMT_SUITE_FIXTURES_OUT"

#: Where the chunk records the shared analysis trees it ACTUALLY touched.
REACH_ENV = "HHEMT_SUITE_REACH_OUT"

#: Where the COLLECT-time plugin writes each test's resolved fixture closure. ARM A's new
#: input: pytest's own resolver, not a substring guess at it. `--collect-only` populates
#: `item._fixtureinfo` without running anything, and the closure is TRANSITIVE -- measured,
#: a fixture two hops from the test appears in it -- which is exactly the indirection the
#: literal scan could not see.
CLOSURE_ENV = "HHEMT_SUITE_CLOSURE_OUT"

#: Where the chunk-mode plugin appends one durable JSON record per test report, as the
#: session runs. Every OTHER per-chunk sidecar is written at `pytest_sessionfinish` and is
#: therefore ABSENT -- not truncated -- when a chunk is cancelled or hits walltime; the
#: junit is the surprising member of that set, because `LogXML` accumulates in memory and
#: opens its logfile with mode `w` inside that same hook. Measured on the 2026-08-25 cancel:
#: the chunk left this file's `pytest_configure` sibling and its `-v` stdout, and left NO
#: junit, fixtures or reach sidecar. `-v` carries identity and outcome live but carries no
#: traceback at all, so a killed run has said WHICH tests are red and nothing about why.
LOGREPORT_ENV = "HHEMT_SUITE_LOGREPORT_OUT"

#: Per-record cap on the rendered failure text. Measured ceiling on a deliberately deep
#: 13-frame chain was 1320 bytes, so this is generous; it exists so one pathological repr
#: carrying large locals cannot dominate the file.
LOGREPORT_MAX_LONGREPR_CHARS = 8192

#: The slug-cache path segment. A test reaches a shared tree by OPENING a file under
#: it, and the tree name is the component after the slug.
SHARED_CACHE_MARK = "synthetic_test_runs"

#: Observed reach, populated by the audit hook installed in pytest_configure. ARM B of
#: a two-arm check: partition.py DERIVES which chunks reach which trees, and a
#: derivation has a HORIZON. Observation has a DIFFERENT blind spot (opens that bypass
#: Python's `open`), so an edge missed by BOTH arms needs two unrelated failures.
#:
#: KEYED ON AN ABSOLUTE PATH, NOT A NAME. The previous version keyed on `parts[i + 2]`
#: -- the component after the cache marker -- which is wrong twice over. (a) It assumes
#: the shape `.../synthetic_test_runs/{slug}/{entry}/...`, but tests/fixtures/
#: test_case_builder.py builds the runs-root THREE ways and only ONE has a slug: the
#: else-branch uses `slug_runs_root(worktree_slug())`, while the override branch (:342)
#: and the scratch branch (:386) both build `Path(...)/"synthetic_test_runs"` with NO
#: slug component. Under those, `parts[i + 2]` is a component INSIDE a tree, so
#: `subanalyses` was recorded as though it were a top-level entry and the tree's own
#: subdirectory was stat'd in place of the tree. (b) A name is not an identity: the
#: root was memoized on FIRST SIGHTING and every later root under the same name
#: discarded, while read/write flags and samples kept accumulating -- so one record
#: could be a union over several directories, described by a stat of whichever was seen
#: first. Measured 2026-08-23: the same entry classified `is_dir=True stamp=True` in one
#: chunk and `is_dir=False` in two others; one False bucketed it TRANSIENT and dropped
#: two genuinely racing trees out of the contention check entirely.
#:
#: The record now carries BOTH candidate prefixes, the stat at BOTH observation times,
#: the resolved root, and the shape -- so the artifact can diagnose its own
#: misclassification, which the previous one could not: `_reach_roots` was module-local
#: and never serialized, which is why neither candidate mechanism could be excluded.
_reach_records: dict[str, dict] = {}

#: The toolkit's own layout stamp. Every analysis and system tree carries one
#: (`version_migration`); the shared compiled tier and the config staging dir do not.
#: Measured on the local slug cache 2026-08-23: 13/13 analysis trees carry it, 0/2
#: shared-by-design dirs do. This is the discriminator INSTEAD of a curated name list,
#: because a name list is the derivation-with-a-horizon this harness rejected elsewhere.
#: It is now ALSO the boundary resolver: whichever candidate carries the stamp is the
#: entity, so the shape is discovered rather than assumed.
_TREE_STAMP = "_version.json"

#: Writes are sampled, not enumerated, so the record stays small. Held ABSOLUTE in the
#: hook and relativized to the resolved entity at session end -- the hook cannot know
#: which candidate is the entity, so relativizing there would bake in the assumption
#: this change exists to remove.
_WRITE_SAMPLE_CAP = 5

#: os.open passes mode=None, so intent lives in the flags. Anything but a pure read.
#: Safe against binary modes: CPython normalizes the audit event's mode argument, so
#: `open(p, "rb")` arrives here as `'r'` and `open(p, "wb")` as `'w'` -- measured, since
#: a write over-report here would be indistinguishable in the output from the
#: tree-mislabelling this change fixes.
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC


def _stat_candidate(path: str) -> tuple[bool | None, bool | None]:
    """(is_dir, carries_stamp) for one candidate prefix; (None, None) if unstattable."""
    try:
        return os.path.isdir(path), os.path.isfile(os.path.join(path, _TREE_STAMP))
    except OSError:
        return None, None


def _either(cand: dict, field: str) -> bool | None:
    """True if the property held at EITHER observation; None only if both are unknown.

    A root that existed when it was opened and was deleted before session end is still a
    real directory that a chunk really touched -- collapsing that to the session-end
    value is the failure direction that demoted two racing trees to TRANSIENT.
    """
    a, b = cand[f"{field}_first"], cand[f"{field}_end"]
    if a is None and b is None:
        return None
    return bool(a) or bool(b)


def _reach_hook(event, args):  # pragma: no cover - audit hook
    if event != "open" or not args:
        return
    p = args[0]
    if not isinstance(p, str) or SHARED_CACHE_MARK not in p:
        return
    parts = p.split(os.sep)
    try:
        i = parts.index(SHARED_CACHE_MARK)
    except ValueError:
        return
    if len(parts) <= i + 2:
        return
    # BOTH candidates, no offset assumption. p1 is the entity under a slugless override
    # root; p2 is the entity under the slugged shared root. Which one is resolved at
    # session end from the layout stamp.
    p1 = os.sep.join(parts[: i + 2])
    p2 = os.sep.join(parts[: i + 3])
    rec = _reach_records.get(p2)
    if rec is None:
        # Two stats, MEMOIZED per distinct 2-prefix -- bounded by trees x slugs, not by
        # opens. The per-open budget forbids a syscall PER OPEN; taking the only stat at
        # session end was an over-correction that made the record describe session-end
        # state rather than the state when the write happened.
        d1, s1 = _stat_candidate(p1)
        d2, s2 = _stat_candidate(p2)
        rec = _reach_records[p2] = {
            "p1": p1,
            "p2": p2,
            "read": False,
            "written": False,
            "write_samples": [],
            "candidates": {
                "p1": {"path": p1, "dir_first": d1, "stamp_first": s1, "dir_end": None, "stamp_end": None},
                "p2": {"path": p2, "dir_first": d2, "stamp_first": s2, "dir_end": None, "stamp_end": None},
            },
        }
    rec["read"] = True
    mode, flags = (args[1] if len(args) > 1 else None), (args[2] if len(args) > 2 else 0)
    if (isinstance(mode, str) and mode != "r") or (mode is None and (flags & _WRITE_FLAGS)):
        rec["written"] = True
        samples = rec["write_samples"]
        if p not in samples and len(samples) < _WRITE_SAMPLE_CAP:
            samples.append(p)


def _finalize_reach_entries() -> dict[str, dict]:
    """Resolve each record's entity, re-stat both candidates, and merge to entity keys.

    Returns a mapping keyed on the ENTITY PATH -- `tree_root` when the stamp resolved,
    else `p2`. The fallback is not cosmetic: `tree_root` is None for every unstamped
    entry BY DEFINITION (that is what SHARED_BY_DESIGN means), so keying on it alone
    would collapse `_software` and `_sensitivity_configs` into one bucket and report
    contention on a directory that does not exist -- a new conflation replacing the one
    this change removes, landing in the quietest bucket. `p2` is exact for every
    shared-shape entry, and an override/scratch root is minted per session
    (`mkdtemp` / `tmp_path`), so two chunks can never share one and the case where `p2`
    is one level too deep can produce no cross-chunk join to get wrong.
    """
    for rec in _reach_records.values():
        for cand in rec["candidates"].values():
            cand["dir_end"], cand["stamp_end"] = _stat_candidate(cand["path"])
        c1, c2 = rec["candidates"]["p1"], rec["candidates"]["p2"]
        if _either(c2, "stamp"):
            rec["tree_root"], rec["root_shape"], entity = c2["path"], "shared", c2
        elif _either(c1, "stamp"):
            rec["tree_root"], rec["root_shape"], entity = c1["path"], "override", c1
        else:
            rec["tree_root"], rec["root_shape"], entity = None, "indeterminate", c2
        rec["entity_path"] = rec["tree_root"] or rec["p2"]
        rec["label"] = os.path.basename(rec["entity_path"])
        rec["is_dir"] = _either(entity, "dir")
        rec["has_version_stamp"] = _either(entity, "stamp")
        rec["existed_at_first_open"] = entity["dir_first"]
        rec["exists_at_end"] = entity["dir_end"]

    merged: dict[str, dict] = {}
    for rec in _reach_records.values():
        key = rec["entity_path"]
        cur = merged.get(key)
        if cur is None:
            rec["write_samples"] = _relativize(rec["write_samples"], key)
            merged[key] = rec
            continue
        # Several p2 keys resolve to ONE entity under the override shape
        # (.../{tree}/sims and .../{tree}/subanalyses both resolve to .../{tree}).
        cur["read"] = cur["read"] or rec["read"]
        cur["written"] = cur["written"] or rec["written"]
        for s in _relativize(rec["write_samples"], key):
            if s not in cur["write_samples"] and len(cur["write_samples"]) < _WRITE_SAMPLE_CAP:
                cur["write_samples"].append(s)
    return merged


def _relativize(paths: list[str], root: str) -> list[str]:
    out = []
    for a in paths:
        try:
            out.append(os.path.relpath(a, root))
        except ValueError:  # pragma: no cover - different drives, Windows only
            out.append(a)
    return out


# --------------------------------------------------------------------------
# pytest plugin half (active only when loaded via `-p run_suite`)
# --------------------------------------------------------------------------
def pytest_configure(config):  # noqa: D103 - pytest hook
    # Arm B installs FIRST and on its OWN channel. It used to sit after the resolution
    # write, behind that block's `if not out: return`, so a session with REACH_ENV set and
    # RESOLUTION_ENV unset installed no audit hook and reported an empty reach -- a
    # detector silently not firing under a condition its own comment ("gated on the
    # channel") denied. run_chunk sets both, so production was unaffected and the coupling
    # was invisible; it surfaced the first time the hook was exercised on its own.
    # Installed here rather than at import so a plugin load that is not a chunk run pays
    # nothing.
    if os.environ.get(REACH_ENV):
        sys.addaudithook(_reach_hook)

    out = os.environ.get(RESOLUTION_ENV)
    if not out:
        return
    try:
        import hhemt

        resolved = str(Path(hhemt.__file__).resolve())
    except Exception as exc:  # noqa: BLE001 - recorded, never raised into the session
        resolved = f"IMPORT-FAILED: {type(exc).__name__}: {exc}"
    Path(out).write_text(json.dumps({"pytest": resolved}) + "\n", encoding="utf-8")


def pytest_collection_modifyitems(items):  # noqa: D103 - pytest hook
    out = os.environ.get(CLOSURE_ENV)
    if not out:
        return
    closures = {}
    for it in items:
        info = getattr(it, "_fixtureinfo", None)
        # FAIL LOUD, at collect time. `_fixtureinfo` is a pytest internal; if a version
        # renames or removes it, an empty closure is indistinguishable downstream from a
        # test that genuinely uses no fixtures, and the partition would silently
        # under-connect -- the same class this derivation exists to end.
        if info is None:
            raise RuntimeError(
                f"pytest item {it.nodeid} exposes no `_fixtureinfo`; the fixture-closure "
                "derivation cannot be computed. Refusing to emit a partial closure."
            )
        closures[it.nodeid] = sorted(info.name2fixturedefs)
    Path(out).write_text(json.dumps(closures) + "\n", encoding="utf-8")


_LOGREPORT_FAILED = False


def _logreport_emit(out, payload):  # noqa: D103 - internal writer
    # Append-only, one JSON object per line, flushed AND fsynced. The three properties
    # each answer a distinct way this instrument can fail. Append (never mode "w") means a
    # second write cannot destroy the first. One object per line means a process killed
    # mid-write leaves at most ONE unparseable trailing line, which a reader skips; any
    # format with a closing delimiter is unreadable while incomplete, which is to say
    # unreadable at the only moment it matters. flush() is what survives a scancel --
    # measured, a SIGKILLed session left a flush-only file fully parseable while its junit
    # was absent entirely -- because killing the PROCESS does not reclaim bytes already in
    # the kernel page cache. fsync() buys node-level durability on top, and it is
    # affordable here ONLY because the record is scoped to non-passing reports plus one
    # per passing call: measured 3.263 ms per fsynced record against 0.024 ms flush-only,
    # a 136x ratio that would be prohibitive at three records per test and is about three
    # seconds per thousand-test chunk at this scope.
    global _LOGREPORT_FAILED
    if _LOGREPORT_FAILED:
        return
    try:
        with open(out, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
    except OSError as exc:  # noqa: BLE001 - an instrument may not break what it measures
        # Latch off and say so ONCE. An unwritable path (quota, permissions, a run dir
        # removed underneath) must degrade to a missing record, never to a failed suite.
        _LOGREPORT_FAILED = True
        print(f"run_suite: durable report log disabled: {exc}", file=sys.stderr, flush=True)


def pytest_sessionstart(session):  # noqa: D103 - pytest hook
    # The header exists so that "no failure records" is unambiguous. Without it, an
    # all-green chunk and a chunk whose writer never loaded produce the same empty file.
    out = os.environ.get(LOGREPORT_ENV)
    if not out:
        return
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    _logreport_emit(
        out,
        {
            "t": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "kind": "session_start",
            "pid": os.getpid(),
        },
    )


def pytest_runtest_logreport(report):  # noqa: D103 - pytest hook
    # Scoped to the COMPLEMENT of what is already durable. The chunk's `-v` stdout is
    # unbuffered and already carries identity and outcome for every test as it lands, so
    # duplicating the green setup/teardown phases here buys nothing. What no existing
    # surface carries is (a) the failure reason -- pytest renders `longrepr` only in the
    # end-of-run summary, so a killed chunk has none of it -- and (b) the per-test
    # duration, which `--durations=0` also renders only in that summary. The second is the
    # one with the larger consequence: submit_suite_uva.sh records that the duration
    # ranking "yields ONLY from a chunk that completes" and is the input the
    # fixture-closure re-partition needs, and two runs have now died reaching for it.
    out = os.environ.get(LOGREPORT_ENV)
    if not out:
        return
    if report.outcome == "passed" and report.when != "call":
        return
    rec = {
        "t": datetime.now(UTC).isoformat(timespec="milliseconds"),
        "kind": "report",
        "nodeid": report.nodeid,
        "when": report.when,
        "outcome": report.outcome,
        "duration": round(getattr(report, "duration", 0.0), 4),
    }
    if report.outcome != "passed":
        # `when` is retained above precisely so a reader can separate a test whose
        # assertion failed from one that never ran because its precondition died. A wall
        # of setup-phase records sharing one first line is ONE lost precondition with N
        # dependents, which aggregate.unevaluated_reasons already knows how to collapse.
        text = report.longreprtext or ""
        if len(text) > LOGREPORT_MAX_LONGREPR_CHARS:
            text = text[:LOGREPORT_MAX_LONGREPR_CHARS] + "\nTRUNCATED by run_suite\n"
            rec["longrepr_truncated"] = True
        rec["longrepr"] = text
    _logreport_emit(out, rec)


def pytest_sessionfinish(session, exitstatus):  # noqa: D103 - pytest hook
    reach_out = os.environ.get(REACH_ENV)
    if reach_out:
        entries = _finalize_reach_entries()
        # The flat `observed`/`written` lists are RETAINED so a pre-fix reader and a
        # pre-fix run dir both keep working -- carrying LABELS, which is what those lists
        # always meant. `entries` is keyed on the ABSOLUTE entity path, because that is
        # the identity the cross-chunk join needs and the label is not unique.
        Path(reach_out).write_text(
            json.dumps(
                {
                    "observed": sorted({r["label"] for r in entries.values()}),
                    "written": sorted({r["label"] for r in entries.values() if r["written"]}),
                    "entries": entries,
                }
            )
            + "\n",
            encoding="utf-8",
        )
    out = os.environ.get(FIXTURES_ENV)
    if not out:
        return
    names = sorted(getattr(session, "_hhemt_suite_fixtures_ok", set()))
    Path(out).write_text(json.dumps(names) + "\n", encoding="utf-8")


try:  # the plugin half needs pytest; the driver half must import without it
    import pytest as _pytest

    @_pytest.hookimpl(hookwrapper=True)
    def pytest_fixture_setup(fixturedef, request):  # noqa: D103 - pytest hook
        outcome = yield
        if not os.environ.get(FIXTURES_ENV) or fixturedef.scope != "session":
            return
        if outcome.excinfo is not None:
            return
        session = request.session
        ok = getattr(session, "_hhemt_suite_fixtures_ok", None)
        if ok is None:
            ok = set()
            session._hhemt_suite_fixtures_ok = ok
        ok.add(fixturedef.argname)

except ImportError:  # pragma: no cover - driver-only import path
    pass


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------
def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()


def _sha_and_dirty(repo: Path) -> tuple[str, bool]:
    return _git(repo, "rev-parse", "HEAD"), bool(_git(repo, "status", "--porcelain"))


def _pytest_cmd(python: str) -> list[str]:
    return [python, "-m", "pytest", "-p", "no:cacheprovider"]


# --------------------------------------------------------------------------
# drive mode
# --------------------------------------------------------------------------
def collect(repo: Path, python: str) -> tuple[list[str], dict[str, list[str]]]:
    """Return (node ids, per-test fixture closures), refusing on any collection error.

    The closure rides THIS invocation rather than a second one: the collection pass was
    already being paid, and `--collect-only` populates `_fixtureinfo` without executing a
    test. Measured on the real corpus: 2566 tests collected in 7.3 s.
    """
    closure_out = Path(tempfile.mkdtemp(prefix="hhemt-closure-")) / "closures.json"
    env = dict(os.environ)
    env[CLOSURE_ENV] = str(closure_out)
    # `-p run_suite` is imported by the pytest CHILD, which does not inherit this process's
    # sys.path; PYTHONPATH is the only channel that survives the `conda run` wrapper.
    _harness = str(Path(__file__).resolve().parent)
    env["PYTHONPATH"] = f"{_harness}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else _harness
    proc = subprocess.run(
        [*_pytest_cmd(python), "--collect-only", "-q", "-p", "_runner"],
        cwd=repo,
        capture_output=True,
        text=True,
        env=env,
    )
    tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    if proc.returncode != 0 or " error" in tail:
        raise SystemExit(
            "refusing to submit: collection did not complete cleanly.\n"
            f"  exit={proc.returncode}\n  summary={tail!r}\n"
            "A partial collection yields a manifest that reconciles perfectly "
            "against itself while the uncollected tests never run."
        )
    node_ids = [ln.strip() for ln in proc.stdout.splitlines() if "::" in ln]
    if not closure_out.is_file():
        raise SystemExit(
            "refusing to submit: the collect-time closure plugin wrote nothing to "
            f"{closure_out}. Without it the partition would fall back to a derivation that "
            "under-connects on 28 known files, and the failure would surface as a flaky "
            "suite rather than as an error."
        )
    closures = json.loads(closure_out.read_text(encoding="utf-8"))
    return node_ids, closures


def warm(repo: Path, python: str, target: str) -> None:
    """Serially borrow the shared object store and compile into the run slug."""
    proc = subprocess.run([*_pytest_cmd(python), "-q", target], cwd=repo)
    if proc.returncode != 0:
        raise SystemExit(
            f"refusing to submit: warm step failed (exit {proc.returncode}).\n"
            "  Nothing was dispatched and no run id was minted."
        )


def _assert_external_warm(warm_log: str, warm_result: str = "", expect_pin: str = "") -> dict:
    """Verify an externally-performed warm left a durable record, and describe it.

    When the warm runs as its own awaited job, "it was warmed" arrives here as a
    CLAIM. This turns it into a check: the log the warm was required to write must
    exist and be non-empty. That is deliberately weaker than a cache-freshness
    predicate -- which submit-side bash owns, because only it decides whether to
    submit the warm at all -- and it is the half this module can verify without
    duplicating the toolkit's cache-path derivation, whose in-source note warns that
    a second copy would let the two roots drift apart.
    """
    if not warm_log:
        raise SystemExit(
            "refusing to submit: --warm-performed-externally requires --warm-log.\n"
            "  The warm's recorded output IS the failure-locality property; without a\n"
            "  path to it, 'the warm succeeded' is an unverifiable assertion."
        )
    p = Path(warm_log).resolve()
    if not p.is_file():
        raise SystemExit(f"refusing to submit: warm log {p} does not exist. Nothing was dispatched.")
    size = p.stat().st_size
    if size == 0:
        raise SystemExit(
            f"refusing to submit: warm log {p} is 0 bytes.\n"
            "  An empty log is the signature of a killed or unflushed writer, not of a\n"
            "  clean warm; it is the exact shape that destroyed a 57-minute run earlier."
        )
    digest = hashlib.sha256(p.read_bytes()).hexdigest()

    # The log is the HUMAN record and is not sufficient evidence. Measured: a warm that
    # skipped all five compile tests wrote `5 skipped, 1 warning in 1.26s` -- present,
    # non-empty, exit 0, and it had compiled nothing. Every log-shaped check above
    # passed on it. The ARTIFACT probe is what actually answers the question.
    if not warm_result:
        raise SystemExit(
            "refusing to submit: --warm-performed-externally requires --warm-result.\n"
            "  A log records that a process ran; only the artifact probe records that a\n"
            "  binary exists. This guard passed on a warm that compiled nothing when it\n"
            "  had only the log."
        )
    rp = Path(warm_result).resolve()
    if not rp.is_file():
        raise SystemExit(f"refusing to submit: warm result {rp} does not exist. Nothing was dispatched.")
    try:
        rec = json.loads(rp.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"refusing to submit: warm result {rp} is not valid JSON ({exc}).") from exc

    if rec.get("compiled") is not True:
        detail = [f"borrow_healthy={rec.get('borrow_healthy')}"]
        for m in rec.get("markers", []):
            if not m.get("matched"):
                detail.append(
                    f"no marker matched in {m.get('path')} "
                    f"(exists={m.get('exists')}, wanted any of {m.get('required_any_of')})"
                )
        raise SystemExit(
            "refusing to submit: the warm did not produce a compiled tier.\n"
            + "".join(f"  {d}\n" for d in detail)
            + f"  full record: {rp}\n  human log:   {p}"
        )

    pin = rec.get("triton_pin")
    if expect_pin and pin != expect_pin:
        raise SystemExit(
            f"refusing to submit: the compiled tier is at TRITON pin {pin}, not the\n"
            f"  expected {expect_pin}. A tier compiled at a DIFFERENT pin satisfies every\n"
            "  marker check and every exit code; only this comparison catches it."
        )

    return {
        "mode": "external",
        "log": str(p),
        "log_bytes": size,
        "log_sha256": digest,
        "result": str(rp),
        "compiled": True,
        "triton_pin": pin,
        "borrow_healthy": rec.get("borrow_healthy"),
    }


def _file_durations(run_dir: Path) -> dict[str, float]:
    """file -> summed reported duration, from a prior run's per-test JSONL records.

    Reads `chunk-*.reports.jsonl`, which is the ONE sidecar written AS THE SESSION RUNS --
    every other per-chunk artifact is a pytest_sessionfinish product and is simply ABSENT
    when a chunk is cancelled or times out. That is what makes a CANCELLED run a usable
    cost source: measured on 20260904T074157Z_cd4d8f6d9a91, chunk 0 left 572 records and no
    junit at all. Coverage is therefore PARTIAL by construction; build_manifest records
    which files were covered, and split_heavy_component imputes the rest at a stated rate.
    """
    out: dict[str, float] = {}
    for p in sorted(run_dir.glob("chunk-*.reports.jsonl")):
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("kind") != "report" or "nodeid" not in rec:
                continue
            out[rec["nodeid"].split("::")[0]] = out.get(rec["nodeid"].split("::")[0], 0.0) + float(
                rec.get("duration") or 0.0
            )
    if not out:
        raise SystemExit(
            f"refusing to submit: no per-test duration records under {run_dir}. A heavy split "
            "balanced on an empty table is a node-count split wearing a duration label."
        )
    return out


def plan(args: argparse.Namespace) -> int:
    repo = Path(args.toolkit).resolve()
    sha, dirty = _sha_and_dirty(repo)
    if dirty and not args.allow_dirty:
        raise SystemExit(f"refusing to submit: {repo} is dirty; pass --allow-dirty to override")

    # WARM FIRST. Nothing below this point runs until the warm is known good, and
    # nothing above it writes anything, so a failed warm leaves the runs_root
    # untouched -- no run id, no directory, no manifest, nothing for the aggregator
    # to find and choke on.
    if args.warm_performed_externally:
        warm_provenance = _assert_external_warm(args.warm_log, args.warm_result, args.expect_triton_pin)
    else:
        warm(repo, args.python, args.warm_target)
        warm_provenance = {"mode": "inline", "target": args.warm_target}

    run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{sha[:12]}"
    run_dir = Path(args.runs_root).resolve() / run_id

    # DECLARE FIRST, then conform to the declaration like every other site. Drive is the
    # site that PARSES the test tree (partition.py reads {repo_root}/tests/** to decide
    # chunk membership), so a drive resolving `hhemt` from a different tree than it parses
    # runs one version's parser over another version's source at the step that decides what
    # each chunk contains. A detector keyed on the importing package reports the parser and
    # is silent about the parsed -- which is why the declaration names the TREE.
    from hhemt.suite import verify_version_conformance, write_version_expectation

    write_version_expectation(run_dir, sha=sha, tree=str(repo))
    _status, _rec = verify_version_conformance(run_dir, site="drive")
    # MISMATCH *and* UNRESOLVABLE. The two non-MATCH states are NOT symmetric, and treating
    # them alike in either direction is a defect: NO_EXPECTATION is a property of the RUN --
    # a pre-floor run dir legitimately has no declaration and refusing there would make old
    # runs unusable -- whereas UNRESOLVABLE is a property of THIS PROCESS, and here the
    # expectation was written moments earlier BY THIS SAME PROCESS. A drive that cannot
    # determine its own tree is not meeting a legacy artifact; it cannot certify anything
    # about the run it is creating. Exempting it would wave through the most obviously broken
    # configuration this floor exists for: a wheel-installed hhemt driving a source --toolkit,
    # i.e. the parser deciding chunk membership being a different version from the tree parsed.
    if _status in ("MISMATCH", "UNRESOLVABLE"):
        if _status == "UNRESOLVABLE":
            raise SystemExit(
                "refusing to plan: this process cannot determine which source tree it resolved "
                "hhemt from (no `src` component -- a wheel or non-src layout), so it cannot "
                "certify the run it is about to create. Re-invoke with PYTHONPATH={--toolkit}/src."
            )
        raise SystemExit(
            f"refusing to plan: drive resolved hhemt from {_rec['resolved_tree']!r} but this "
            f"run is declared against {_rec['expected_tree']!r}. The manifest would be derived "
            "by one version and executed by another. IF YOU REACHED THIS FROM THE ESTATE "
            "HARNESS this is expected and the fix is one line: rerun.sh invokes this driver "
            "as `python -m hhemt.suite._runner` with NO PYTHONPATH, deliberately and by a "
            "stated design predating this floor (its comment at rerun.sh:79 reads `-m` rather "
            "than a path so the estate needs no PYTHONPATH), so it resolves hhemt from the "
            "conda env rather than from --toolkit. Export PYTHONPATH={--toolkit}/src at that "
            "call site, exactly as submit_suite_uva.sh:72 already does for the array elements."
        )

    node_ids, closures = collect(repo, args.python)
    # LAZY, and load-bearing. This module is loaded as a pytest PLUGIN in the child
    # (`-p _runner`), and a plugin module is imported BEFORE the toolkit's repo-root
    # conftest runs its `sys.path.insert(0, _SRC)` -- measured. A module-level
    # `from hhemt.suite import partition` therefore binds `hhemt` in sys.modules before
    # the guard can point it at the checkout under test, and the guard then exits 99 on
    # every chunk whose --toolkit differs from the installed toolkit. Keep it in here.
    from hhemt.suite import partition as _partition

    manifest = _partition.build_manifest(
        repo_root=repo,
        node_ids=node_ids,
        source_sha=sha,
        run_id=run_id,
        closures=closures,
        cheap_bins=args.cheap_bins,
        heavy_split_budget_s=(None if args.heavy_split_budget_min is None else args.heavy_split_budget_min * 60.0),
        durations=_file_durations(Path(args.durations_from)) if args.durations_from else None,
        tree_isolation_per_chunk=bool(args.isolate_trees_per_chunk),
    )
    manifest["warm"] = warm_provenance
    _partition.write_manifest(manifest, run_dir)

    n = manifest["chunk_count"]
    (run_dir / "chunk_count").write_text(f"{n}\n", encoding="utf-8")
    print(f"run_id={run_id}")
    print(f"run_dir={run_dir}")
    print(f"chunk_count={n}")
    print(f"cheap_bins={args.cheap_bins}")
    print(f"tree_isolation_per_chunk={bool(args.isolate_trees_per_chunk)}")
    print(f"collected={len(node_ids)}")
    return 0


# --------------------------------------------------------------------------
# chunk mode
# --------------------------------------------------------------------------
def _chunk_basetemp(run_id: str, cid: int) -> Path:
    """The pytest basetemp this chunk will use -- CHOSEN, never discovered.

    Keyed by run AND chunk because pytest REMOVES an explicit basetemp if it exists,
    so two array elements co-scheduled on one node would otherwise wipe each other.

    Deliberately NODE-LOCAL: derived from TMPDIR and never assigned to it. Redirecting
    TMPDIR would move every heavy fixture's working tree onto shared storage, taxing
    the chunk that already dominates wall clock on every run for a benefit that accrues
    only on failing ones, and -- because `tmp_path_retention_policy = "failed"` retains
    exactly the trees a debugging loop produces most of -- would accumulate without
    bound where the node no longer reclaims them.
    """
    tmp = os.environ.get("TMPDIR") or tempfile.gettempdir()
    return Path(tmp) / f"hhemt-suite-{run_id}-{cid:02d}"


def _truncate(raw: bytes, limit: int) -> bytes:
    """Keep the head and the tail, with an explicit marker naming what was dropped."""
    half = limit // 2
    dropped = len(raw) - 2 * half
    marker = (
        f"\n\n... [hhemt-suite] {dropped} bytes elided from the middle of a "
        f"{len(raw)}-byte file (per-file cap {limit}) ...\n\n"
    ).encode()
    return raw[:half] + marker + raw[-half:]


def _harvest_diagnostics(basetemp: Path, dest: Path) -> dict:
    """Copy failure diagnostics out of a node-local basetemp, under caps.

    Never raises into the chunk result: a diagnostics failure must not change the
    verdict of the run it is diagnosing. Every cap that binds is RECORDED -- a
    silently partial bundle is the defect class this harness exists to close, and
    hitting a cap is itself a finding.
    """
    manifest: dict = {
        "basetemp": str(basetemp),
        "globs": list(HARVEST_GLOBS),
        "priority_order": list(HARVEST_PRIORITY),
        "caps": {
            "max_file_bytes": HARVEST_MAX_FILE_BYTES,
            "max_files": HARVEST_MAX_FILES,
            "max_total_bytes": HARVEST_MAX_TOTAL_BYTES,
        },
        "harvested": [],
        "truncated": [],
        "dropped": [],
        "dropped_bytes": 0,
        "total_bytes_written": 0,
    }
    try:
        if not basetemp.is_dir():
            manifest["note"] = "basetemp absent; pytest may have died before creating it"
            return manifest

        found: list[Path] = []
        for g in HARVEST_GLOBS:
            found.extend(basetemp.rglob(g))

        def _rank(f: Path) -> tuple[int, float, str]:
            # noqa justified: `i` is not used in the loop BODY but escapes it -- the
            # for-else sets it when nothing matched and `return (i, ...)` consumes it.
            for i, pat in enumerate(HARVEST_PRIORITY):  # noqa: B007
                if fnmatch.fnmatch(f.name, pat):
                    break
            else:
                i = len(HARVEST_PRIORITY)
            try:
                mtime = f.stat().st_mtime
            except OSError:
                mtime = 0.0
            return (i, -mtime, str(f))

        found = sorted({f.resolve() for f in found if f.is_file()}, key=_rank)

        dest.mkdir(parents=True, exist_ok=True)
        written = 0
        for src in found:
            try:
                size = src.stat().st_size
            except OSError:
                continue
            rel = src.relative_to(basetemp) if src.is_relative_to(basetemp) else Path(src.name)
            flat = str(rel).replace(os.sep, "__")

            if len(manifest["harvested"]) >= HARVEST_MAX_FILES:
                manifest["dropped"].append({"path": str(rel), "bytes": size, "cap": "max_files"})
                manifest["dropped_bytes"] += size
                continue
            keep = min(size, HARVEST_MAX_FILE_BYTES)
            if written + keep > HARVEST_MAX_TOTAL_BYTES:
                manifest["dropped"].append({"path": str(rel), "bytes": size, "cap": "max_total_bytes"})
                manifest["dropped_bytes"] += size
                continue
            try:
                raw = src.read_bytes()
            except OSError:
                continue
            if len(raw) > HARVEST_MAX_FILE_BYTES:
                raw = _truncate(raw, HARVEST_MAX_FILE_BYTES)
                manifest["truncated"].append({"path": str(rel), "original_bytes": size})
            (dest / flat).write_bytes(raw)
            written += len(raw)
            manifest["harvested"].append({"path": str(rel), "as": flat, "bytes": len(raw)})

        manifest["total_bytes_written"] = written
    except Exception as exc:  # noqa: BLE001 - diagnostics never decide a verdict
        manifest["error"] = f"{type(exc).__name__}: {exc}"
    try:
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except OSError:
        pass
    return manifest


#: Triage run dirs are suffixed so nothing can mistake one for a suite run, and so a
#: later --from-run default cannot select one as a source of truth. This is the SECOND
#: of two independent arms: a triage run also writes `summary.triage.json` rather than
#: `summary.json`, so `_select_source_run`'s own `summary.json` filter already excludes
#: it. Two unrelated mechanisms, so neither one being wrong is sufficient to leak a
#: triage run into a suite reader.
TRIAGE_SUFFIX = ".triage"


def _select_source_run(runs_root: Path, from_run: str) -> Path:
    """Pick the run whose failures are being re-tested, and say which one aloud."""
    if from_run:
        p = (runs_root / from_run).resolve()
        if not (p / "summary.json").is_file():
            raise SystemExit(f"refusing: {p}/summary.json does not exist.")
        return p
    cands = sorted(
        (d for d in runs_root.iterdir() if (d / "summary.json").is_file() and not d.name.endswith(TRIAGE_SUFFIX)),
        key=lambda d: d.stat().st_mtime,
    )
    if not cands:
        raise SystemExit(f"refusing: no run under {runs_root} carries a summary.json.")
    return cands[-1]


def triage(args: argparse.Namespace) -> int:
    """Re-run only the prior run's FAILED and UNEVALUATED tests, as one session.

    Answers the operator's iteration question -- did the failing set become empty --
    without re-running the whole suite. It is NOT a suite result and cannot be read as
    one: the manifest declares `scope_intent="triage"`, which aggregate.py BINDS as the
    scope regardless of what the caller passed, so the verdict line carries scope=triage
    adjacent to verdict=, its `collected` is the triage set size rather than the suite's,
    and its summary is written under a distinct filename that no later reader can pick up
    as a suite summary.
    """
    repo = Path(args.toolkit).resolve()
    sha, dirty = _sha_and_dirty(repo)
    runs_root = Path(args.runs_root).resolve()
    src = _select_source_run(runs_root, args.from_run)
    summary = json.loads((src / "summary.json").read_text(encoding="utf-8"))
    src_sha = str(summary.get("source_sha") or "?")
    print(f"triage source run: {src.name}  (sha {src_sha[:12]})")

    # A summary predating the id-shape fix has an INCOMPLETE failed list: class-based
    # failures landed in `absent` then, so triaging from it would silently omit every
    # failing test in a class. The fix added `unmatched_junit`; its absence dates the
    # artifact. Refuse rather than warn -- a warned-past incomplete input is exactly
    # the shape this harness exists to make impossible. This refusal is safe to keep
    # sharp because it fires on an OBSOLETE artifact, never on a legitimate use: every
    # aggregate.py since the fix emits the key unconditionally.
    if "unmatched_junit" not in summary:
        raise SystemExit(
            f"refusing: {src / 'summary.json'} predates the id-shape fix (no "
            "'unmatched_junit' key), so its failed list omits every class-based test. "
            "Re-aggregate that run, or triage from a newer one."
        )
    if dirty and not args.allow_dirty:
        raise SystemExit(f"refusing: {repo} is dirty; pass --allow-dirty to override")

    # SHA DRIFT IS REPORTED, NEVER REFUSED. Triage exists to answer "did my fix work",
    # so the source run is at the PRE-fix sha by construction and a refusal keyed on
    # "the shas differ" would fire on 100% of legitimate uses -- a guard nobody reads,
    # whose override becomes reflex. The condition that actually makes the input the
    # wrong instrument is narrower and is checked below: node ids the source run named
    # that NO LONGER EXIST at HEAD, because the fix renamed, reparametrized, or deleted
    # them. That check discriminates; a sha comparison does not.
    if src_sha != sha:
        print(
            f"NOTE: source run is at {src_sha[:12]} and the toolkit is at {sha[:12]}. "
            "Expected -- the failure list predates the fix under test. The triage set is "
            "verified against the CURRENT collection below."
        )

    failed = list(summary.get("failed") or [])
    uneval = list(summary.get("unevaluated") or [])
    requested = sorted(set(failed) | set(uneval))
    if not requested:
        print("triage set is EMPTY -- the source run had no failed or unevaluated tests.")
        return 0

    # The real drift check. pytest ERRORS OUT ENTIRELY (exit 4) on a single unmatched
    # node id, so one test renamed by the fix would take the whole triage run down as a
    # VOID chunk with an opaque reason. Collecting first turns that into a named,
    # legible line -- and a vanished id is usually GOOD NEWS, since the fix removed or
    # renamed the failing test. Vanished ids are DROPPED and REPORTED, never silently
    # omitted; the run refuses only when nothing survives.
    # BIND the closures rather than discarding them with `[0]`. The triage manifest needs the
    # same closure-derived `expected_fixtures` the normal path gets, and this collect pass has
    # already computed them -- `collect()` returns `(node_ids, closures)`.
    live_ids, closures = collect(repo, args.python)
    live = set(live_ids)
    node_ids = [n for n in requested if n in live]
    vanished = [n for n in requested if n not in live]
    if vanished:
        print(f"{len(vanished)} node id(s) from the source run do not exist at {sha[:12]}:")
        for n in vanished:
            print(f"  vanished: {n}")
        print("  (renamed, reparametrized, or deleted by the fix -- dropped from the set)")
    if not node_ids:
        raise SystemExit(
            f"refusing: none of the {len(requested)} node id(s) in the source run's "
            f"failed+unevaluated set still exist at {sha[:12]}. There is nothing to "
            "re-test; run the full suite instead."
        )

    run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{sha[:12]}{TRIAGE_SUFFIX}"
    run_dir = runs_root / run_id
    (run_dir / "chunks").mkdir(parents=True, exist_ok=True)

    # DECLARE HERE TOO, AND DECLARE EARLY. triage() hand-builds its run dir and never calls
    # plan(), so without this the single triage chunk has nothing to conform to -- and the
    # roster check in the aggregator would count `drive` and `chunk-00` as silent on every
    # triage run. `repo` (:890), `sha` (:891) and `run_dir` (:961) are all bound above this
    # line; nothing new is computed.
    #
    # THE PLACEMENT IS THE POINT, and it was wrong in the previous form of this spec, which
    # inserted after the manifest and chunk_count writes. plan()'s check earns a STRUCTURAL
    # bound -- its raise precedes collect() and long precedes the manifest write, so a
    # mis-resolved drive never produces a manifest and the roster denominator a later
    # aggregate reads is therefore only ever written by a drive whose own gate held. Placed
    # after the manifest write, triage AUTHORS that denominator and then refuses, losing the
    # bound for its own runs. Triage cannot check as early as plan() does -- it must collect
    # to learn which ids survive, and run_dir does not exist until run_id is composed -- so
    # this line is the EARLIEST point at which the record can be written, and it is early
    # enough that a refusing triage leaves no manifest behind.
    from hhemt.suite import verify_version_conformance, write_version_expectation

    write_version_expectation(run_dir, sha=sha, tree=str(repo))
    # AND CONFORM. Declaring without checking makes triage the one writer the roster trusts
    # and never verifies: a triage process resolving a different tree than --toolkit would
    # declare THAT tree, run its chunk against it, and every record would agree -- peer
    # agreement with a population of one, which is exactly what a declaration exists to
    # replace. The referent is NOT self-supplied: `repo = Path(args.toolkit).resolve()` comes
    # from the CLI, so the comparison is "where this process imported hhemt from" against
    # "the tree the operator named", identically to plan(). triage() is structurally the
    # DRIVE of its own run, so it takes the drive's policy: MISMATCH and UNRESOLVABLE refuse,
    # NO_EXPECTATION cannot occur here because the declaration is written on the line above.
    _tstatus, _trec = verify_version_conformance(run_dir, site="drive")
    if _tstatus in ("MISMATCH", "UNRESOLVABLE"):
        raise SystemExit(
            f"refusing to triage: resolved hhemt from {_trec['resolved_tree']!r} against a run "
            f"declared for {_trec['expected_tree']!r} ({_tstatus})."
        )
    (run_dir / "chunks" / "00.txt").write_text("\n".join(node_ids) + "\n", encoding="utf-8")
    # LAZY, and load-bearing. This module is loaded as a pytest PLUGIN in the child
    # (`-p _runner`), and a plugin module is imported BEFORE the toolkit's repo-root
    # conftest runs its `sys.path.insert(0, _SRC)` -- measured. A module-level
    # `from hhemt.suite import partition` therefore binds `hhemt` in sys.modules before
    # the guard can point it at the checkout under test, and the guard then exits 99 on
    # every chunk whose --toolkit differs from the installed toolkit. Keep it in here.
    from hhemt.suite import partition as _partition

    # CLOSURE-DERIVED, matching partition.py's enrichment loop. A substring scan would keep
    # the defect alive on exactly the path an operator uses most while iterating toward green:
    # `test_partition_split.py` mentions `tritonswmm_cpu_compiled` seven times in string
    # literals, the scan reads all seven as requests, and the single triage chunk VOIDs.
    #
    # REFUSE BY NAME rather than subscripting blind OR defaulting to empty. `node_ids` is a
    # subset of `live` and `collect()` builds both halves in one invocation, so a miss is
    # excluded by construction TODAY -- but only while the two halves agree on ID SHAPE:
    # `node_ids` is parsed from `--collect-only -q` STDOUT, `closures` is keyed on
    # `item.nodeid`. The stdout half accepts ANY line containing `::` (_runner.py:544), so a
    # warnings-summary entry naming a nodeid, or a plugin banner, enters `node_ids` while
    # `closures` -- keyed strictly on `item.nodeid` -- cannot contain it. Same invocation,
    # genuine divergence, dependent on pytest's stdout shape rather than on this code.
    # On the NORMAL path such a divergence is caught legibly by build_manifest's wholesale
    # guard; THE TRIAGE PATH HAND-BUILDS ITS MANIFEST AND NEVER CALLS build_manifest, so it
    # has nothing above it. This guard is therefore NOT redundant -- do not delete it by
    # analogy with the redundant per-node guard in partition.py's enrichment loop.
    _unkeyed = [n for n in node_ids if n not in closures]
    if _unkeyed:
        raise SystemExit(
            f"refusing to triage: {len(_unkeyed)} collected node id(s) have no fixture closure, "
            f"first {_unkeyed[0]!r}. The two halves of collect() disagree on id shape "
            "(stdout-parsed ids vs item.nodeid); expected_fixtures cannot be derived from "
            "absent evidence."
        )
    expected = sorted({fx for n in node_ids for fx in closures[n] if fx in _partition.RECORDED_FIXTURES})
    manifest = {
        "run_id": run_id,
        "source_sha": sha,
        "scope_intent": "triage",
        "from_run": src.name,
        "from_sha": src_sha,
        "from_failed": sorted(failed),
        "from_unevaluated": sorted(uneval),
        "vanished_at_head": vanished,
        "collected": node_ids,
        "chunk_count": 1,
        "shared_tree_exposure": [],
        "chunks": [
            {
                "chunk_id": 0,
                "kind": "triage",
                "files": sorted({n.split("::")[0] for n in node_ids}),
                "node_ids": node_ids,
                "expected_fixtures": expected,
            }
        ],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "chunk_count").write_text("1\n", encoding="utf-8")
    print(f"run_id={run_id}")
    print(f"run_dir={run_dir}")
    print("chunk_count=1")
    print(f"collected={len(node_ids)}  (failed {len(failed)}, unevaluated {len(uneval)}, vanished {len(vanished)})")
    # The bill, printed before the run rather than discovered during it: triage costs
    # the DISTINCT heavy fixture chains its set touches, plus the tests themselves.
    print(f"fixture closure: {expected or '(none -- pure unit set)'}")
    return 0


def _sibling_resolution(repo: Path, python: str) -> str:
    proc = subprocess.run(
        [python, "-c", "import hhemt, pathlib; print(pathlib.Path(hhemt.__file__).resolve())"],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip() if proc.returncode == 0 else f"IMPORT-FAILED: {proc.stderr.strip()}"


def run_chunk(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).resolve()

    # Conform BEFORE reading the manifest. A chunk that resolves a different tree than the
    # one the manifest was derived against would execute node ids selected by a parser it
    # is not running -- and would do it silently, because every downstream artifact is
    # keyed on the run dir rather than on the tree.
    from hhemt.suite import chunk_site, verify_version_conformance

    _status, _rec = verify_version_conformance(run_dir, site=chunk_site(args.chunk))
    # MISMATCH *and* UNRESOLVABLE -- see U2-2 for why the two non-MATCH states are not
    # symmetric. NO_EXPECTATION stays non-fatal: a hand re-invocation of run_chunk against a
    # pre-floor run dir is a legitimate operator action and is untouched by this.
    if _status in ("MISMATCH", "UNRESOLVABLE"):
        if _status == "UNRESOLVABLE":
            raise SystemExit(
                f"refusing to run chunk {args.chunk}: this process cannot determine which source "
                "tree it resolved hhemt from (no `src` component). Re-invoke with "
                "PYTHONPATH={--toolkit}/src."
            )
        raise SystemExit(
            f"refusing to run chunk {args.chunk}: resolved hhemt from "
            f"{_rec['resolved_tree']!r} against a run declared for {_rec['expected_tree']!r}."
        )
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if args.n_chunks is not None and args.n_chunks != manifest["chunk_count"]:
        raise SystemExit(
            f"refusing to run: --n-chunks {args.n_chunks} disagrees with the manifest's "
            f"chunk_count {manifest['chunk_count']}. The chunk count is DERIVED from the "
            "fixture closure and the collected universe; it is not a free parameter. "
            "Read it from the run dir's chunk_count file instead of defaulting it."
        )
    repo = Path(args.toolkit).resolve()
    cid = args.chunk
    node_file = run_dir / "chunks" / f"{cid:02d}.txt"
    node_ids = [ln for ln in node_file.read_text(encoding="utf-8").splitlines() if ln.strip()]

    sha_start, dirty_start = _sha_and_dirty(repo)
    res_path = run_dir / f"chunk-{cid:02d}.resolution.json"
    fix_path = run_dir / f"chunk-{cid:02d}.fixtures.json"
    reach_path = run_dir / f"chunk-{cid:02d}.reach.json"
    junit = run_dir / f"chunk-{cid:02d}.junit.xml"

    env = dict(os.environ)
    env[RESOLUTION_ENV] = str(res_path)
    env[FIXTURES_ENV] = str(fix_path)
    env[REACH_ENV] = str(reach_path)
    # Named `.reports.jsonl` rather than `.log` because it is a machine-read record, and
    # sited in the run dir alongside its siblings so a cancelled chunk's evidence is where
    # a reader already looks. Unlike them it is written AS THE SESSION RUNS, which is the
    # whole point: every other sidecar here is a `pytest_sessionfinish` artifact and is
    # simply absent when the chunk is cancelled or hits walltime.
    env[LOGREPORT_ENV] = str(run_dir / f"chunk-{cid:02d}.reports.jsonl")
    # Latent, NOT the measured cause. tests/conftest.py:77 runs a run-dir reaper against
    # the SHARED slug cache at every session end, and each chunk is a controller rather
    # than an xdist worker, so an early-finishing chunk sweeps while siblings still run.
    # It is refuted as this run's mechanism -- the sweep skips anything younger than
    # ttl_days=7 and checks liveness first -- but a session-end sweep of shared state
    # under concurrent readers is a hazard the TTL happens to prevent rather than one
    # the design excludes. One env var, defence in depth.
    env["HHEMT_DISABLE_RUN_DIR_REAPER"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    # THE LEVER THE README NAMES AND THE HARNESS DID NOT PULL. rerun.sh:242 and :389 unset
    # HHEMT_TEST_RUNS_ROOT_OVERRIDE, and a repo-wide grep finds no site that ever set it --
    # so until now every chunk shared one analysis-tree root and a split component would have
    # raced. Setting it HERE rather than in rerun.sh is deliberate: the login-node unset stays
    # (a chunk must never INHERIT a private root from an operator's shell), and the only thing
    # that grants one is the manifest the partition was planned under.
    #
    # This is NOT either prohibited lever. XDG_CACHE_HOME and a per-chunk working directory
    # are prohibited because each relocates the hhemt cache root and restores an N-times clone
    # and compile; this one does not. `_software_root` is pinned to the SLUG runs_root at
    # tests/fixtures/test_case_builder.py:354, OUTSIDE the if/else at :329-332 that consumes
    # the override, so one borrow / one compile and the 403 ceiling are untouched. It composes
    # with the three privately-rooted session fixtures because conftest.py's
    # _runs_root_override_env saves and RESTORES the prior value rather than clearing it.
    #
    # The cost it buys, stated: synthetic_multisim_completed and synthetic_sensitivity_completed
    # gate on _marker_state, which is never "complete" against a fresh root, so each isolated
    # chunk pays its own analysis.run() / submit_workflow(). Upper bound measured at 2570 s for
    # the whole setup tier of chunk 0 (20535 s wall minus 17964.7 s of reported call duration).
    if manifest.get("tree_isolation_per_chunk"):
        _iso = run_dir / "tree_isolation" / f"chunk-{cid:02d}"
        _iso.mkdir(parents=True, exist_ok=True)
        env["HHEMT_TEST_RUNS_ROOT_OVERRIDE"] = str(_iso)
    else:
        # Absent isolation the chunk must not inherit one from the submitting shell either:
        # a stray override would silently move this chunk's trees off the root every OTHER
        # chunk uses, which is the same race with the blame pointing the wrong way.
        env.pop("HHEMT_TEST_RUNS_ROOT_OVERRIDE", None)
    # `-p run_suite` is imported by the pytest CHILD, which does not inherit this
    # process's sys.path. The module-level insert above runs here and never reaches
    # it, so the plugin -- and with it the whole per-process resolution record --
    # silently failed to load. PYTHONPATH is the only channel that survives the
    # `conda run` wrapper the submit script uses. Prepended, not replaced: the
    # toolkit's repo-root conftest rewrites PYTHONPATH to put its own src first and
    # preserves the remaining entries, so this survives that rewrite.
    _harness = str(Path(__file__).resolve().parent)
    _existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{_harness}{os.pathsep}{_existing}" if _existing else _harness

    sibling = _sibling_resolution(repo, args.python)

    basetemp = _chunk_basetemp(manifest["run_id"], cid)
    cmd = [
        *_pytest_cmd(args.python),
        "-v",
        *JUNIT_ARGS,
        f"--junitxml={junit}",
        f"--basetemp={basetemp}",
        "-p",
        "_runner",
        "--durations=0",
        *node_ids,
    ]
    proc = subprocess.run(cmd, cwd=repo, env=env)
    rc = proc.returncode

    # Failure-only harvest. The reason a chunk failed lives in a node-local log that
    # dies with the node; copying it out on rc != 0 is what makes the reason reachable
    # from the run dir. A passing chunk pays nothing at all -- not a walk, not a stat.
    harvest = None
    if rc != 0:
        harvest = _harvest_diagnostics(basetemp, run_dir / f"chunk-{cid:02d}.diagnostics")

    sha_end, dirty_end = _sha_and_dirty(repo)
    pytest_res = ""
    if res_path.exists():
        pytest_res = json.loads(res_path.read_text(encoding="utf-8")).get("pytest", "")
    else:
        # The plugin did not run. Say so IN the record rather than leaving the field
        # empty, because an empty string and "the instrument never loaded" are the
        # same bytes downstream, and aggregate.py VOIDs a chunk with no recorded
        # resolution -- which is the correct outcome and needs a legible reason.
        pytest_res = "PLUGIN-DID-NOT-LOAD: no resolution written by -p _runner"
    fixtures_ok = json.loads(fix_path.read_text(encoding="utf-8")) if fix_path.exists() else []
    _reach = json.loads(reach_path.read_text(encoding="utf-8")) if reach_path.exists() else {}
    observed_trees = _reach.get("observed", []) if isinstance(_reach, dict) else _reach
    written_trees = _reach.get("written", []) if isinstance(_reach, dict) else []
    reach_entries = _reach.get("entries", {}) if isinstance(_reach, dict) else {}

    status = {
        "chunk_id": cid,
        "run_id": manifest["run_id"],
        "pytest_exit": rc,
        "node_count": len(node_ids),
        "source_sha_start": sha_start,
        "source_sha_end": sha_end,
        "dirty_start": dirty_start,
        "dirty_end": dirty_end,
        "resolved_hhemt": {"pytest": pytest_res, "sibling": sibling},
        "session_fixtures_ok": fixtures_ok,
        "observed_trees": observed_trees,
        "reach_entries": reach_entries,
        "written_trees": written_trees,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
        "slurm_state": os.environ.get("HHEMT_SUITE_SLURM_STATE", ""),
        "basetemp": str(basetemp),
        "runs_root_override": env.get("HHEMT_TEST_RUNS_ROOT_OVERRIDE", ""),
        "diagnostics": (
            None
            if harvest is None
            else {
                "dir": f"chunk-{cid:02d}.diagnostics",
                "harvested": len(harvest.get("harvested", [])),
                "truncated": len(harvest.get("truncated", [])),
                "dropped": len(harvest.get("dropped", [])),
                "dropped_bytes": harvest.get("dropped_bytes", 0),
                "bytes_written": harvest.get("total_bytes_written", 0),
                "error": harvest.get("error"),
                "note": harvest.get("note"),
            }
        ),
    }
    (run_dir / f"chunk-{cid:02d}.status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return rc


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--toolkit",
        default=os.environ.get("HHEMT_TOOLKIT", os.getcwd()),
        help="path to the pinned hhemt checkout; defaults to $HHEMT_TOOLKIT then cwd",
    )
    p.add_argument("--python", default=sys.executable, help="interpreter that runs pytest")
    p.add_argument("--runs-root", default=os.environ.get("SCRATCH_DIR", "."), help="parent of the run directory")
    p.add_argument("--cheap-bins", type=int, default=1, help="bins for the non-heavy remainder")
    # NOT a chunk COUNT. The count is derived from the budget, because makespan saturates
    # at the heaviest indivisible unit -- measured 10,343.5 s at FILE granularity and
    # 5,178.3 s at NODE granularity on run 20260904T145255Z_2853a077a3ee -- so a caller
    # choosing K directly would inflate chunk_count and pay one more isolated fixture build
    # per part for zero wall-clock gain.
    p.add_argument(
        "--heavy-split-budget-min",
        type=float,
        default=None,
        help="split each heavy component into time-balanced parts of about this many "
        "minutes; requires --durations-from and --isolate-trees-per-chunk",
    )
    p.add_argument(
        "--durations-from",
        default=None,
        help="run dir whose chunk-*.reports.jsonl supplies the per-file duration table",
    )
    p.add_argument(
        "--isolate-trees-per-chunk",
        action="store_true",
        help="export a per-chunk HHEMT_TEST_RUNS_ROOT_OVERRIDE so a split component's parts "
        "do not share an analysis tree; _software stays on the slug root either way "
        "(test_case_builder.py:354). NOTE: the FIRST run using this is not red-set-comparable "
        "to an earlier one -- isolation flips the *_completed session fixtures from adopt to "
        "REBUILD, so a new red needs a second isolated run before it is a regression.",
    )
    p.add_argument("--warm-target", default="tests/test_synth_00_compile_models.py")
    p.add_argument("--allow-dirty", action="store_true")
    p.add_argument(
        "--warm-performed-externally",
        action="store_true",
        help="the borrow+compile ran as its own awaited job; verify its log instead of compiling here",
    )
    p.add_argument(
        "--warm-log",
        default="",
        help="path to the external warm's recorded output; required with --warm-performed-externally",
    )
    p.add_argument(
        "--warm-result",
        default="",
        help="path to the warm's artifact-probe result.json; required with --warm-performed-externally",
    )
    p.add_argument(
        "--expect-triton-pin",
        default="",
        help="when given, the probed tier's triton_pin must equal this or the run is refused",
    )
    p.add_argument(
        "--n-chunks",
        type=int,
        default=None,
        help="cross-check only: rejected when it disagrees with the manifest",
    )
    p.add_argument("--chunk", type=int, default=None, help="chunk mode: run this chunk id")
    p.add_argument(
        "--triage",
        action="store_true",
        help="build a run from the prior run's failed+unevaluated tests; NOT a suite result",
    )
    p.add_argument(
        "--from-run",
        default="",
        help="triage source run id; default is the newest non-triage run carrying a summary.json",
    )
    p.add_argument("--run-dir", default="", help="chunk mode: the run directory")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.triage:
        return triage(args)
    if args.chunk is not None:
        if not args.run_dir:
            raise SystemExit("--chunk requires --run-dir")
        return run_chunk(args)
    return plan(args)


if __name__ == "__main__":  # pragma: no cover - module-execution path
    # `main()` RETURNS pytest's exit code; SystemExit is what makes it the PROCESS's.
    # Retained for `python -m hhemt.suite._runner` after the move. The Typer command
    # in `suite/_cli.py` does the equivalent conversion with `raise typer.Exit(code=...)`
    # -- a Typer command body that merely RETURNS the code exits 0 (measured), which
    # would report COMPLETED to SLURM for a failed chunk.
    raise SystemExit(main())
