"""The write-surface observer sees renames, so an atomic write is neither invisible nor a false collision.

Chunk 3a.7 made the shared sensitivity CSV write atomic with `os.replace`. A rename raises the `os.rename` audit event
and no `open` event, so an observer that records only opens stopped seeing that write, and the real BLOCKING
`[concurrent-build]` problem on `_sensitivity_configs/synth_sensitivity.csv` in counted array 1 would have vanished by
blindness. `_runner._reach_hook` now records a rename's destination in `replace_samples`, counts write-mode opens of
dot-prefixed basenames in `coordination_writes` instead of sampling them, and `aggregate` reads the two.

Every observer arm runs a `python` subprocess that installs the hook with a real `sys.addaudithook`, performs real file
operations under a temporary `.../synthetic_test_runs/{slug}/_sensitivity_configs/`, and prints
`_finalize_reach_entries()` as JSON, so the audit events are CPython's and not a stand-in's. The aggregator arms build
run directories from such real reach records. Nothing here runs a solver.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from hhemt.suite import aggregate as A
from tests.test_suite_tail_element import _SHA, _junit, _status

_CFG = "_sensitivity_configs"
_CSV = "synth_sensitivity.csv"

_HARNESS = """
import json, os, sys
from pathlib import Path
import hhemt.suite._runner as R
from hhemt.cache.paths import synthetic_runs_root
RUNS = synthetic_runs_root().name
assert RUNS == R.SHARED_CACHE_MARK, (RUNS, R.SHARED_CACHE_MARK)
sys.addaudithook(R._reach_hook)
base = Path(sys.argv[1])
cfg = base / RUNS / "slug" / "_sensitivity_configs"
cfg.mkdir(parents=True, exist_ok=True)
outside = base / "elsewhere"
outside.mkdir(exist_ok=True)
{ops}
print(json.dumps(R._finalize_reach_entries()))
"""


def _stdout(tmp_path: Path, ops: str) -> list[str]:
    """Run `ops` (real file operations, using `cfg` and `outside`) under the real hook; return the stdout lines."""
    script = _HARNESS.format(ops=textwrap.dedent(ops))
    done = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)], capture_output=True, text=True, check=True, timeout=120
    )
    return done.stdout.strip().splitlines()


def _reach(tmp_path: Path, ops: str) -> dict:
    """The reach entries `_finalize_reach_entries()` printed after `ops`."""
    return json.loads(_stdout(tmp_path, ops)[-1])


def _the_entry(entries: dict) -> dict:
    """The one reach record for the `_sensitivity_configs` entity."""
    assert len(entries) == 1, entries
    return next(iter(entries.values()))


_ATOMIC = """
tmp = cfg / ".synth_sensitivity.csv.1.abc.tmp"
tmp.write_text("a,b\\n1,2\\n")
os.replace(tmp, cfg / "synth_sensitivity.csv")
"""


def test_an_atomic_write_records_the_destination_and_counts_the_temporary(tmp_path):
    """Property: a dot-prefixed temporary opened for write and `os.replace`d onto `x.csv` records `x.csv` in
    `replace_samples`, no temporary in `write_samples`, and `coordination_writes` of 1.

    Class: REGRESSION, red on the hook before this change, which has no `replace_samples` (the rename is invisible)
    and puts the temporary in `write_samples`.

    Why it is required, checkable against the code: `_reach_hook` handles the `os.rename` event for the destination and
    routes a write-mode open of a dot-prefixed basename to the counter.

    What kills it: a hook that records only opens, or one that samples the temporary. The source of the rename is not
    recorded: a hook that recorded the source instead of the destination records the temporary and reds here.

    A second correct implementation under which it passes: one that records the destination relative to the entity at
    hook time instead of at finalisation."""
    rec = _the_entry(_reach(tmp_path, _ATOMIC))
    assert rec["replace_samples"] == [_CSV]
    assert rec["write_samples"] == []
    assert rec["coordination_writes"] == 1
    assert rec["written"] is True and rec["replace_cap_bound"] is False


def test_the_event_reaches_the_hook_for_a_path_object_and_not_for_bytes(tmp_path):
    """Property: `os.replace` with `pathlib.Path` arguments is recorded (CPython normalises them to `str` in the audit
    event), while `bytes` arguments are not (they stay `bytes` and fail the `str` gate).

    Class: NEW CAPABILITY, a control taken from the probe rather than assumed: a first subprocess records the raw
    argument types the event carries, so the claim is measured on this interpreter and not asserted.

    Why it is required, checkable against the code: the hook requires `isinstance(args[1], str)`; the writer in
    `test_case_catalog.py` passes `Path` objects.

    What kills it: a hook that tests for `os.PathLike`, or that decodes bytes, would change the second half; a hook
    that rejects `Path` would red the first.

    A second correct implementation under which it passes: one that accepts bytes by decoding them, if the bytes
    expectation were changed with it."""
    lines = _stdout(
        tmp_path / "probe",
        """
        seen = []
        def note(event, args):
            if event == "os.rename":
                seen.append([type(args[0]).__name__, type(args[1]).__name__])
        sys.addaudithook(note)
        (cfg / ".p.tmp").write_text("x")
        os.replace(cfg / ".p.tmp", cfg / "p.csv")
        (cfg / ".q.tmp").write_text("x")
        os.replace(os.fsencode(cfg / ".q.tmp"), os.fsencode(cfg / "q.csv"))
        print(json.dumps(seen))
        """,
    )
    assert json.loads(lines[0]) == [["str", "str"], ["bytes", "bytes"]], lines[0]
    rec = _the_entry(
        _reach(
            tmp_path / "records",
            """
            (cfg / ".p.tmp").write_text("x")
            os.replace(cfg / ".p.tmp", cfg / "p.csv")
            (cfg / ".q.tmp").write_text("x")
            os.replace(os.fsencode(cfg / ".q.tmp"), os.fsencode(cfg / "q.csv"))
            """,
        )
    )
    assert rec["replace_samples"] == ["p.csv"], "a Path replace must be recorded, a bytes replace must not"


def test_many_temporaries_do_not_crowd_a_payload_path_out_of_the_samples(tmp_path):
    """Property: six atomic writes with six distinct temporaries and then one plain write-mode open of `y.yaml` keep
    `y.yaml` in `write_samples`.

    Class: REGRESSION, red on the hook before this change, whose five `write_samples` slots fill with temporaries and
    drop `y.yaml`.

    Why it is required, checkable against the code: dot-prefixed opens are counted, not sampled.

    What kills it: sampling dot-prefixed opens. Control in the test: the counter holds all six temporaries.

    A second correct implementation under which it passes: one that evicts temporaries from the samples at close."""
    ops = """
    for i in range(6):
        tmp = cfg / f".t{i}.tmp"
        tmp.write_text("x")
        os.replace(tmp, cfg / "x.csv")
    (cfg / "y.yaml").write_text("y")
    """
    rec = _the_entry(_reach(tmp_path, ops))
    assert rec["write_samples"] == ["y.yaml"]
    assert rec["coordination_writes"] == 6 and rec["replace_samples"] == ["x.csv"]


def test_the_replace_cap_binds_and_says_so(tmp_path):
    """Property: six distinct destinations keep five `replace_samples` and set `replace_cap_bound`.

    Class: NEW CAPABILITY. A cap that drops a destination silently would let a collision beyond it read as absent.

    Why it is required, checkable against the code: `_REPLACE_SAMPLE_CAP` is 5 and the flag is set when a distinct
    destination is dropped. Control: five distinct destinations leave the flag unset.

    What kills it: a cap with no flag.

    A second correct implementation under which it passes: a larger cap, if the test's count followed it."""
    six = "".join(f'(cfg / ".t{i}").write_text("x"); os.replace(cfg / ".t{i}", cfg / "d{i}.csv")\n' for i in range(6))
    five = "".join(f'(cfg / ".t{i}").write_text("x"); os.replace(cfg / ".t{i}", cfg / "d{i}.csv")\n' for i in range(5))
    capped = _the_entry(_reach(tmp_path / "six", six))
    assert len(capped["replace_samples"]) == 5 and capped["replace_cap_bound"] is True
    fits = _the_entry(_reach(tmp_path / "five", five))
    assert len(fits["replace_samples"]) == 5 and fits["replace_cap_bound"] is False


def test_a_rename_whose_destination_is_outside_the_cache_mark_records_nothing(tmp_path):
    """Control: a rename onto a path outside `synthetic_test_runs` is ignored, as an open outside it is.

    Class: NEW CAPABILITY. The wrong implementation records every rename.

    Why it is required, checkable against the code: the destination must contain `SHARED_CACHE_MARK`.

    What kills it: dropping the mark test.

    A second correct implementation under which it passes: one that keys on the resolved path."""
    entries = _reach(
        tmp_path,
        """
        (outside / ".t").write_text("x")
        os.replace(outside / ".t", outside / "z.csv")
        """,
    )
    assert entries == {}


_NAMES = [".a", ".a.b", "..x"]


@pytest.mark.parametrize("name", _NAMES)
def test_every_leading_dot_basename_the_hook_counts_is_a_coordination_path_to_the_aggregator(tmp_path, name):
    """Property: a write-mode open of a leading-`.` basename is counted by the hook, not sampled, and the aggregator's
    `_is_coordination_path` agrees it is coordination.

    Class: NEW CAPABILITY, the pin of the duplicated clause: `_runner.py` repeats the first clause of
    `aggregate._is_coordination_path` because it imports nothing from its siblings, and the two must not drift.
    Several names are tried (`.a`, `.a.b`, `..x`) so a divergence on a name the test did not try cannot pass.

    Why it is required, checkable against the code: a name the hook counts but the aggregator treats as payload would
    be unwitnessed, and one the hook samples but the aggregator exempts would hide a collision.

    What kills it: a hook test of `endswith(".lock")` or of `".lock" in name`.

    A second correct implementation under which it passes: both sides calling one shared function."""
    rec = _the_entry(_reach(tmp_path, f'(cfg / "{name}").write_text("x")\n'))
    assert rec["coordination_writes"] == 1 and rec["write_samples"] == []
    assert A._is_coordination_path(name)


def test_a_lock_suffix_without_a_leading_dot_is_a_sample_to_the_hook_and_still_coordination_to_the_aggregator(tmp_path):
    """Control for the pin above: `x.lock` does not start with a dot, so the hook samples it (unchanged), while the
    aggregator's second clause still reads it as coordination; the two copies differ on a `.lock` suffix only.

    Class: NEW CAPABILITY. It shows the pinned arms are not satisfied by a hook that counts everything.

    Why it is required, checkable against the code: the hook repeats only the first clause.

    What kills it: a hook that also counts `.lock` suffixes (the sample would vanish).

    A second correct implementation under which it passes: one that counts both clauses, if this control were
    reversed with it."""
    rec = _the_entry(_reach(tmp_path, '(cfg / "x.lock").write_text("x")\n'))
    assert rec["write_samples"] == ["x.lock"] and rec["coordination_writes"] == 0
    assert A._is_coordination_path("x.lock")


def test_the_dot_basename_itself_is_coordination_in_both_copies():
    """Property: the bare name `.` (not openable for write) is coordination in both copies.

    Class: NEW CAPABILITY, covering the name the subprocess arms cannot open. Both copies are called directly.

    Why it is required, checkable against the code: `"."` starts with a dot.

    What kills it: a hook copy that requires a second character.

    A second correct implementation under which it passes: any that tests `startswith('.')`."""
    import hhemt.suite._runner as R

    for name in (".", ".a", ".a.b", "..x"):
        assert R._is_coordination_basename(f"/c/{name}") and A._is_coordination_path(name)
    assert not R._is_coordination_basename("/c/x.lock") and A._is_coordination_path("x.lock")


# --------------------------------------------------------------------------------------------------------------
# the aggregator over run directories built from real reach records
# --------------------------------------------------------------------------------------------------------------


def _records(tmp_path: Path, scenarios: dict[int, str]) -> dict[int, dict]:
    """Real reach entries per chunk: each scenario's file operations run under the real hook in ONE shared directory."""
    shared = tmp_path / "shared"
    return {cid: _reach(shared, ops) for cid, ops in scenarios.items()}


def _run_dir(root: Path, records: dict[int, dict]) -> Path:
    """A run directory whose chunks carry the given reach entries; each chunk passes one test."""
    chunks = []
    for cid in sorted(records):
        node = f"tests/test_x.py::test_{cid}"
        chunks.append(
            {
                "chunk_id": cid,
                "kind": "heavy",
                "files": ["tests/test_x.py"],
                "node_ids": [node],
                "expected_fixtures": [],
            }
        )
    nodes = [c["node_ids"][0] for c in chunks]
    manifest = {
        "run_id": "FAKE",
        "source_sha": _SHA,
        "collected": nodes,
        "chunk_count": len(chunks),
        "cheap_bins": 1,
        "chunks": chunks,
        "shared_tree_exposure": [],
        "declared_complement": [],
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for cid, entries in records.items():
        (root / f"chunk-{cid:02d}.status.json").write_text(json.dumps(_status(cid, entries=entries)), encoding="utf-8")
        (root / f"chunk-{cid:02d}.junit.xml").write_text(_junit([f"tests/test_x.py::test_{cid}"]), encoding="utf-8")
    return root


_OPEN_CSV = '(cfg / "synth_sensitivity.csv").write_text("x")\n'


def _problems_about_cfg(result: dict) -> list[str]:
    """The problems that name the `_sensitivity_configs` entity."""
    return [p for p in result["problems"] if _CFG in p]


def test_two_chunks_that_only_replace_the_same_path_give_the_atomic_replace_advisory(tmp_path):
    """Property: two chunks that each REPLACE `synth_sensitivity.csv` and open it for write in neither give the
    `[atomic-replace]` advisory with the verbatim text, and no `[concurrent-build]` problem.

    Class: REGRESSION, red on the aggregator before this change, which emits no entry naming that path.

    Why it is required, checkable against the code: the shared-by-design rule reads `replace_samples_by_chunk`.

    What kills it: an aggregator that ignores replace samples (no advisory), or one that treats a replaced path like an
    opened one (a BLOCKING problem).

    A second correct implementation under which it passes: one that emits the advisory from a separate pass over
    the entries."""
    (tmp_path / "run").mkdir()
    recs = _records(tmp_path, {0: _ATOMIC, 1: _ATOMIC})
    result = A.aggregate(_run_dir(tmp_path / "run", recs), scope="array")
    ours = [a for a in result["advisories"] if a.startswith("[atomic-replace]")]
    assert len(ours) == 1 and ours[0] == (
        f'[atomic-replace] "{_CSV}" replaced atomically by chunks [0, 1]; identity of content not checked. This is '
        "not a clearance: a reader never meets a partial file, but the last writer's content wins. Before shape "
        "iteration 29 this observer could not see a rename, and a path written this way by two chunks produced no "
        "entry at all; a truncating open of the same path by two chunks is still BLOCKING."
    )
    assert _problems_about_cfg(result) == [], result["problems"]


def test_two_chunks_that_each_open_the_path_for_write_are_still_blocking(tmp_path):
    """Control, green before this change and pinned: two chunks that each OPEN `synth_sensitivity.csv` for write give
    `[concurrent-build] ... BLOCKING`.

    Class: NEW CAPABILITY. The wrong implementation turns every shared write into an advisory.

    Why it is required, checkable against the code: the plain-open collision rule is unchanged.

    What kills it: exempting a path because the observer now knows about renames.

    A second correct implementation under which it passes: any that keeps the open-open rule."""
    (tmp_path / "run").mkdir()
    recs = _records(tmp_path, {0: _OPEN_CSV, 1: _OPEN_CSV})
    result = A.aggregate(_run_dir(tmp_path / "run", recs), scope="array")
    blocking = _problems_about_cfg(result)
    assert len(blocking) == 1 and "[concurrent-build]" in blocking[0] and "BLOCKING" in blocking[0]
    assert not [a for a in result["advisories"] if a.startswith("[atomic-replace]")]


def test_one_chunk_opening_a_path_another_replaces_is_blocking(tmp_path):
    """Property: one chunk opens `synth_sensitivity.csv` for write and another replaces it: BLOCKING.

    Class: NEW CAPABILITY, red against an implementation that exempts every replaced path. A truncating open is what a
    reader of the replaced file can meet.

    Why it is required, checkable against the code: `_opened_and_replaced` is a payload sample of one chunk found in
    another chunk's replace samples.

    What kills it: exempting replaced paths. Control: the reversed assignment of the two scenarios is also blocking.

    A second correct implementation under which it passes: one that checks the pair in both directions."""
    for order, (a, b) in enumerate(((_OPEN_CSV, _ATOMIC), (_ATOMIC, _OPEN_CSV))):
        root = tmp_path / f"case{order}"
        (root / "run").mkdir(parents=True)
        recs = _records(root, {0: a, 1: b})
        result = A.aggregate(_run_dir(root / "run", recs), scope="array")
        blocking = _problems_about_cfg(result)
        assert len(blocking) == 1 and "[concurrent-build]" in blocking[0] and "BLOCKING" in blocking[0], order
        assert "opened for write a path another chunk replaced" in blocking[0]


def test_a_writer_with_only_a_coordination_write_is_witnessed_and_one_with_no_evidence_is_not(tmp_path):
    """Pins of the fail-closed clause: a writer chunk whose only record is `coordination_writes` is not unwitnessed;
    one with `written` true and no sample of any kind is.

    Class: NEW CAPABILITY. The clause is reachable since the observer stopped sampling dot-prefixed writes.

    Why it is required, checkable against the code: `_unwitnessed` requires no write sample, no replace sample and no
    coordination write.

    What kills it: the old clause (a coordination-only writer then reads as unwitnessed and BLOCKING), or a clause
    that exempts a writer with no evidence at all.

    A second correct implementation under which it passes: one that treats `coordination_writes` as a sample."""
    coord = '(cfg / ".lock_like").write_text("x")\n'
    (tmp_path / "ok").mkdir()
    ok = A.aggregate(_run_dir(tmp_path / "ok", _records(tmp_path / "okrec", {0: coord, 1: coord})), scope="array")
    assert _problems_about_cfg(ok) == [], ok["problems"]
    assert any("[shared-by-design-write]" in a for a in ok["advisories"])

    recs = _records(tmp_path / "badrec", {0: coord, 1: coord})
    for entries in recs.values():
        rec = _the_entry(entries)
        rec.update(write_samples=[], replace_samples=[], coordination_writes=0)
    (tmp_path / "bad").mkdir()
    bad = A.aggregate(_run_dir(tmp_path / "bad", recs), scope="array")
    blocking = _problems_about_cfg(bad)
    assert len(blocking) == 1 and "NO write, replace or coordination evidence" in blocking[0]


def test_a_reach_record_written_before_the_observer_saw_renames_aggregates_as_before(tmp_path):
    """Control: a record without `replace_samples`, `replace_cap_bound` and `coordination_writes` reads as an empty
    list, False and 0, and aggregates as it did.

    Class: NEW CAPABILITY. The reach schema change is additive; a run recorded before it must not be re-read wrongly.

    Why it is required, checkable against the code: the aggregator reads each key with `rec.get`.

    What kills it: a bare `rec["replace_samples"]`.

    A second correct implementation under which it passes: one that fills the keys at load time."""
    (tmp_path / "run").mkdir()
    recs = _records(tmp_path, {0: _OPEN_CSV, 1: _OPEN_CSV})
    for entries in recs.values():
        for key in ("replace_samples", "replace_cap_bound", "coordination_writes"):
            _the_entry(entries).pop(key)
    result = A.aggregate(_run_dir(tmp_path / "run", recs), scope="array")
    blocking = _problems_about_cfg(result)
    assert len(blocking) == 1 and "BLOCKING" in blocking[0]


def test_a_writer_whose_only_evidence_is_a_replace_sample_is_witnessed(tmp_path):
    """Property: a writer chunk with a replace sample, no write sample and no coordination write is not unwitnessed.

    Class: NEW CAPABILITY. The fail-closed clause is widened to the replace witness; a clause that kept only the write
    sample and the coordination count would read an atomic writer as evidence-less and BLOCK.

    Why it is required, checkable against the code: `_unwitnessed` lists a writer only when all three witnesses are
    absent. Control: the same chunks with the replace samples cleared too are BLOCKING (the evidence-less arm above).

    What kills it: dropping `and not _replace_by_chunk.get(str(c))` from `_unwitnessed`.

    A second correct implementation under which it passes: one that treats a replace sample as a write sample."""
    recs = _records(tmp_path / "rec", {0: _ATOMIC, 1: _ATOMIC})
    for entries in recs.values():
        rec = _the_entry(entries)
        rec.update(write_samples=[], coordination_writes=0)
        assert rec["replace_samples"] == [_CSV]
    (tmp_path / "run").mkdir()
    result = A.aggregate(_run_dir(tmp_path / "run", recs), scope="array")
    assert not [p for p in _problems_about_cfg(result) if "NO write, replace or coordination" in p], result["problems"]


def test_a_path_replaced_by_one_chunk_only_gives_no_atomic_replace_advisory(tmp_path):
    """Property: `[atomic-replace]` needs TWO OR MORE replacing chunks; one chunk replacing a path, with another chunk
    reaching the entity, gives none.

    Class: NEW CAPABILITY. The advisory says a path was replaced "by chunks {ids}" and a single chunk is not a
    collision.

    Why it is required, checkable against the code: `_atomic_only` keeps a path only when `len(ids) > 1`.

    What kills it: `len(ids) > 0`. Control: the two-replacers arm above still gives the advisory.

    A second correct implementation under which it passes: one that counts distinct chunks from the evidence dict."""
    coord = '(cfg / ".lock_like").write_text("x")\n'
    (tmp_path / "run").mkdir()
    result = A.aggregate(_run_dir(tmp_path / "run", _records(tmp_path, {0: _ATOMIC, 1: coord})), scope="array")
    assert not [a for a in result["advisories"] if a.startswith("[atomic-replace]")], result["advisories"]
    assert _problems_about_cfg(result) == [], result["problems"]


def test_a_path_one_chunk_opens_and_two_chunks_replace_is_blocking_and_not_an_atomic_replace_advisory(tmp_path):
    """Property: one chunk opens `synth_sensitivity.csv` for write and two chunks replace it: the BLOCKING problem is
    present and no `[atomic-replace]` advisory names the path.

    Class: NEW CAPABILITY. A path that any chunk opens for write is not atomic on both sides, so the advisory
    (which reads as a softer finding) must not be emitted beside the BLOCKING one.

    Why it is required, checkable against the code: `_atomic_only` excludes any path in `_opened_any`.

    What kills it: dropping `and s not in _opened_any`.

    A second correct implementation under which it passes: one that suppresses the advisory whenever the entry is
    blocking."""
    (tmp_path / "run").mkdir()
    recs = _records(tmp_path, {0: _OPEN_CSV, 1: _ATOMIC, 2: _ATOMIC})
    result = A.aggregate(_run_dir(tmp_path / "run", recs), scope="array")
    blocking = _problems_about_cfg(result)
    assert len(blocking) == 1 and "BLOCKING" in blocking[0] and "[concurrent-build]" in blocking[0]
    assert not [a for a in result["advisories"] if a.startswith("[atomic-replace]")], result["advisories"]


# Two records for one entity key: a stamped analysis tree under an override root, entered through two sub-directories
# (`sims`, `subanalyses`), which resolve to ONE entity and so take the merge branch of `_finalize_reach_entries`.
# `RUNS` is the runs-root directory name the harness script derives from the product's single definition.
_TWO_RECORDS_SETUP = """
tree = base / RUNS / "tree"
(tree / "sims").mkdir(parents=True)
(tree / "subanalyses").mkdir()
(tree / "_version.json").write_text("{}")
def atomic(directory, count, prefix):
    for i in range(count):
        (directory / f".t{prefix}{i}").write_text("x")
        os.replace(directory / f".t{prefix}{i}", directory / f"{prefix}{i}.csv")
"""


def _merged(tmp_path: Path, first: str, second: str) -> dict:
    """The one merged record after the hook saw `first` and then `second` (calls of `atomic`) under one entity."""
    entries = _reach(tmp_path, f"{_TWO_RECORDS_SETUP}{first}\n{second}\n")
    return _the_entry(entries)


def test_two_records_for_one_entity_merge_their_replace_samples_flag_and_coordination_counts(tmp_path):
    """Property: when two records resolve to one entity, `_finalize_reach_entries` unions the replace samples under
    the cap, ORs `replace_cap_bound` whichever record came first, and SUMS `coordination_writes`.

    Class: NEW CAPABILITY. Reaches the `cur is not None` branch honestly: a real hook run that touches `sims` and
    `subanalyses` of one stamped tree gives two records with the same entity key.

    Why it is required, checkable against the code: the merge branch had no test. Arms: a record with six replace
    destinations (cap bound) read second after a small one (the flag must come from the OR, since the union fits the
    cap), the reverse order, and two small records whose union overflows (the flag must be set by the overflow). The
    coordination counts are 1 per atomic write, so 2 + 3 must give 5.

    What kills it: dropping the OR of the flag or the sum of the counts in the merge branch.

    A second correct implementation under which it passes: one that merges by re-recording the samples through the hook
    path."""
    small_big = _merged(
        tmp_path / "a",
        'atomic(tree / "sims", 2, "s")',
        'atomic(tree / "subanalyses", 6, "b")',
    )
    big_small = _merged(
        tmp_path / "b",
        'atomic(tree / "sims", 6, "b")',
        'atomic(tree / "subanalyses", 2, "s")',
    )
    for rec in (small_big, big_small):
        assert rec["replace_cap_bound"] is True and len(rec["replace_samples"]) == 5
        assert rec["coordination_writes"] == 8
    # The first record has no replace sample at all, so the union fits the cap and only the OR of the flag can set it.
    for order, (first, second) in enumerate(
        (
            ('(tree / "sims" / ".only").write_text("x")', 'atomic(tree / "subanalyses", 6, "b")'),
            ('atomic(tree / "sims", 6, "b")', '(tree / "subanalyses" / ".only").write_text("x")'),
        )
    ):
        rec = _merged(tmp_path / f"or{order}", first, second)
        assert rec["replace_cap_bound"] is True and len(rec["replace_samples"]) == 5, order
        assert rec["coordination_writes"] == 7, order
    fits = _merged(tmp_path / "c", 'atomic(tree / "sims", 2, "s")', 'atomic(tree / "subanalyses", 3, "t")')
    assert fits["replace_cap_bound"] is False and len(fits["replace_samples"]) == 5
    assert fits["coordination_writes"] == 5
    overflow = _merged(tmp_path / "d", 'atomic(tree / "sims", 3, "s")', 'atomic(tree / "subanalyses", 3, "t")')
    assert overflow["replace_cap_bound"] is True and len(overflow["replace_samples"]) == 5
    assert overflow["coordination_writes"] == 6


def _sidecar_record(samples: list[str], bound: bool, coordination: int) -> dict:
    """The shape of one finalised reach record, as far as the cross-worker merge reads it."""
    return {
        "read": True,
        "written": True,
        "write_samples": [],
        "replace_samples": list(samples),
        "replace_cap_bound": bound,
        "coordination_writes": coordination,
    }


@pytest.mark.parametrize("first_is_bound", [True, False], ids=["bound-first", "bound-second"])
def test_the_cross_worker_merge_unions_replace_samples_ors_the_flag_and_sums_the_counts(first_is_bound):
    """Property: `_runner._merge_reach_entries` (the xdist sidecar merge) keeps the union of replace samples under the
    cap, ORs `replace_cap_bound` in both orders, and sums `coordination_writes`. The small record carries no samples,
    so the union fits the cap and the flag can come only from the OR.

    Class: NEW CAPABILITY. Two worker records for one entity are fed to the function directly; the shape is the one
    the hook and `_finalize_reach_entries` produce.

    Why it is required, checkable against the code: a worker's replace evidence reaches the chunk's record only through
    this merge; before this chunk it merged `write_samples` alone.

    What kills it: dropping the sample union, the flag OR or the count sum there. Control: a record with no new keys
    (a worker running older code) merges without error and adds nothing.

    A second correct implementation under which it passes: one that merges all list-valued keys generically."""
    import hhemt.suite._runner as R

    bound = _sidecar_record(["a0", "a1", "a2", "a3", "a4"], True, 2)
    small = _sidecar_record([], False, 3)
    into = {"k": dict(bound if first_is_bound else small)}
    R._merge_reach_entries(into, {"k": dict(small if first_is_bound else bound)})
    merged = into["k"]
    assert merged["replace_cap_bound"] is True, "the flag is ORed whichever record is first"
    assert merged["coordination_writes"] == 5
    assert sorted(merged["replace_samples"]) == ["a0", "a1", "a2", "a3", "a4"], "the union fits the cap exactly"
    older = {"read": True, "written": True, "write_samples": ["w"]}
    R._merge_reach_entries(into, {"k": older})
    assert into["k"]["coordination_writes"] == 5 and into["k"]["replace_cap_bound"] is True


def test_the_cross_worker_merge_enforces_the_replace_cap_on_a_union_that_overflows_it():
    """Property: two worker records with five DISTINCT replace samples each, neither flagged, merge to exactly
    `_REPLACE_SAMPLE_CAP` samples, the first record's five, and the merged record says the cap bound.

    Class: NEW CAPABILITY. The existing merge arm feeds a union that fits the cap exactly, so a merge that enforces no
    cap passes it. Input class: a union larger than the cap with no flag set on either side.

    Why it is required, checkable against the code: `_merge_reach_entries` appends a destination only while
    `len(replace_samples) < _REPLACE_SAMPLE_CAP`; an uncapped list would make the evidence unbounded in a sidecar.

    What kills it: `if True:` in place of the cap test.

    A second correct implementation under which it passes: one that merges through a shared helper with
    `_finalize_reach_entries`."""
    import hhemt.suite._runner as R

    first = _sidecar_record(["a0", "a1", "a2", "a3", "a4"], False, 0)
    second = _sidecar_record(["b0", "b1", "b2", "b3", "b4"], False, 0)
    into = {"k": dict(first)}
    R._merge_reach_entries(into, {"k": dict(second)})
    assert into["k"]["replace_samples"] == ["a0", "a1", "a2", "a3", "a4"]
    assert len(into["k"]["replace_samples"]) == R._REPLACE_SAMPLE_CAP


def test_the_cross_worker_merge_sets_the_cap_flag_itself_when_it_drops_a_destination():
    """Property: when the merge drops a distinct destination for lack of room, `replace_cap_bound` is True in the merged
    record although NEITHER input record carried the flag.

    Class: NEW CAPABILITY. The existing arm gets its True from an input record's own flag, so a merge that never sets
    the flag itself passes it. Control: a union that fits the cap leaves the flag False in the same call shape.

    Why it is required, checkable against the code: a destination dropped here is evidence the aggregator cannot see;
    the flag is how the `[atomic-replace]` advisory says `(replace samples capped)`.

    What kills it: replacing the `else:` branch that sets the flag with `pass`.

    A second correct implementation under which it passes: one that derives the flag from the count of dropped
    destinations after the loop."""
    import hhemt.suite._runner as R

    fits = {"k": _sidecar_record(["a0", "a1"], False, 0)}
    R._merge_reach_entries(fits, {"k": _sidecar_record(["b0", "b1"], False, 0)})
    assert fits["k"]["replace_cap_bound"] is False, "control: a union under the cap sets no flag"
    over = {"k": _sidecar_record(["a0", "a1", "a2", "a3", "a4"], False, 0)}
    R._merge_reach_entries(over, {"k": _sidecar_record(["b0"], False, 0)})
    assert over["k"]["replace_cap_bound"] is True
    assert over["k"]["replace_samples"] == ["a0", "a1", "a2", "a3", "a4"]


def test_a_rename_from_outside_the_cache_mark_into_it_marks_the_entry_written_and_records_the_destination(tmp_path):
    """Property: a file created OUTSIDE the cache mark and renamed into `_sensitivity_configs` leaves the entity
    `written` with the destination in `replace_samples` and no coordination count, although no `open` event was raised
    under the mark.

    Class: NEW CAPABILITY, closing the gap the coder's arms leave: every atomic write they run opens its temporary under
    the mark first, which already sets `written`, so a rename that does not set it passes them all. Input class: a
    source the hook never sees opened (a `shutil.move` or `os.replace` from a staging directory elsewhere).

    Why it is required, checkable against the code: the aggregator takes writers from `written`; without it the
    replacing chunk is not a writer of the entity and its replace evidence is never read.

    What kills it: dropping `rec["written"] = True` from the `os.rename` branch of `_reach_hook`.

    A second correct implementation under which it passes: one that marks written when it records a replace sample."""
    ops = """
    src = outside / "staged.csv"
    src.write_text("a\\n1\\n")
    os.replace(src, cfg / "synth_sensitivity.csv")
    """
    entry = _the_entry(_reach(tmp_path, ops))
    assert entry["written"] is True
    assert [Path(s).name for s in entry["replace_samples"]] == [_CSV]
    assert entry["coordination_writes"] == 0 and entry["write_samples"] == []


def test_one_chunk_that_opens_and_replaces_the_same_path_is_not_blocking_by_itself(tmp_path):
    """Property: a single chunk that opens `synth_sensitivity.csv` for write AND replaces it, beside a second chunk that
    writes an unrelated path, produces no problem naming `_sensitivity_configs`.

    Class: NEW CAPABILITY, a different position of the BLOCKING rule's invariant: it compares one chunk's open with
    ANOTHER chunk's replace. A rule that also compares a chunk with itself blames a chunk for racing its own write.

    Why it is required, checkable against the code: `_opened_and_replaced` skips the replacing chunk equal to the
    opening chunk (`oc != cid`).

    What kills it: dropping that inequality. Control: the previous BLOCKING arm, where the two roles sit in different
    chunks, still blocks.

    A second correct implementation under which it passes: one that builds the sets per chunk and intersects across
    chunks."""
    root = tmp_path / "case"
    (root / "run").mkdir(parents=True)
    recs = _records(root, {0: _OPEN_CSV + _ATOMIC, 1: '(cfg / "other.yaml").write_text("x")\n'})
    result = A.aggregate(_run_dir(root / "run", recs), scope="array")
    assert _problems_about_cfg(result) == []
