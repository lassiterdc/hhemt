"""REVIEWER findings for chunk `3a.1` (item `1b.4`), the model-log re-key.

This file is the reviewing seat's deliverable. Nothing here was written by the author of the
change under review. Every FFinding test below is RED against the chunk as it stands; the single
HELD test is GREEN now and must remain green, and its named mutation is recorded beside it.

Laptop-safe: no solver compiled or executed, no network, no `uv sync`; every path is under
`tmp_path`.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]

_MEMBER = "member_gpu_0_r1"

#: The relative events-CSV pointer this repository's own bundle fixtures carry
#: (`tests/fixtures/bundles/{multi_sim,sensitivity_master}/cfg_analysis.yaml:33`), so it is not a
#: contrived value. `weather_events_to_simulate` is a plain `Path` with no validator that
#: absolutises it, so a member yaml dumps it verbatim.
_RELATIVE_CSV = "weather_events.csv"

_RESUME_BODY = "SWMM exchange history replayed to t=600 s (3 steps); resuming live segment\n"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _events_csv(path: Path, *values: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("event\n" + "\n".join(values) + "\n", encoding="utf-8")
    return path


def _sub_yaml(sub_dir: Path, *, master: Path, csv_value: str) -> Path:
    """A member yaml shaped like the one `_atomic_write_analysis_yaml` dumps."""
    sub_dir.mkdir(parents=True, exist_ok=True)
    p = sub_dir / f"{_MEMBER}.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "analysis_id": _MEMBER,
                "is_experiment_member": True,
                "TRITON_reporting_timestep_s": 600.0,
                "experiment_cfg_yaml": str(master / "analysis_config.yaml"),
                "weather_events_to_simulate": csv_value,
                "weather_event_indices": ["event"],
            }
        )
    )
    return p


def _seed_log(logdir: Path, *, event_slug: str | None, ordinal: int | None, body: str) -> Path:
    logdir.mkdir(parents=True, exist_ok=True)
    if event_slug is not None:
        p = logdir / f"model_tritonswmm_{_MEMBER}_evt-{event_slug}.log"
    else:
        p = logdir / f"model_tritonswmm_{_MEMBER}_evt{ordinal}.log"
    p.write_text(body, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# FINDING 1 — the chunk reds 22 tests in three files the approved shape declared out of scope
# ---------------------------------------------------------------------------


def test_the_three_files_the_shape_declared_out_of_scope_are_green():
    """Property: the chunk leaves the suite green in the files its change makes reachable.

    Class: the TEST IDS of `tests/test_resume_verdict_disclosed_denominator.py`,
    `tests/test_resume_stepback_from_cut_off_snapshot.py` and
    `tests/test_debug_flag_keep_previous_snapshots.py`.

    Why required: measured on this tree, `22 failed, 48 passed` — verdict-expressed as
    `22` FAILING TEST IDS (instrument: pytest's own summary line, pasted in the reviewer's
    scratch), every failure
    `AttributeError: 'types.SimpleNamespace' object has no attribute
    '_retrieve_weather_indexer_using_integer_index'` at `src/hhemt/run_simulation.py:140`.
    Per file: 3 / 10 / 9. The production consequence is larger than the test count:
    `analysis_validation.check_resume_event_evidence` resolves its ledger inside
    `try: ... except Exception: indeterminate += 1`, so any analysis object without the
    projection is now reported `applicable=False, passed=True` ("No resumed sim was
    examined"). Pre-change the same object WAS examined, and the check could FAIL. A gate whose
    failure mode becomes a silent N/A is a weakened gate.

    What kills it: giving each of those analyses the weather-index projection the resolver now
    needs, and seeding each ledger at the event-identity name. Named change: pass an analysis
    object carrying `_retrieve_weather_indexer_using_integer_index` in `_analysis()` at
    `test_resume_verdict_disclosed_denominator.py:116-120` (and the equivalent stubs in the
    other two files), and rename `_write_ledger`'s seed from `model_tritonswmm_evt0.log` to
    `model_tritonswmm_evt-event.0.log`.

    Second correct implementation under which it still passes: any suite in which those three
    files' stubs supply the projection by a different route — a shared `_stub_analysis` helper
    in `tests/`, or binding the production method off `TRITONSWMM_analysis` as
    `tests/test_model_log_identity_key.py:65-67` does.

    REGRESSION. Pre-change, `model_logfile_for` read two attributes and could not raise, so
    these three files were green; the reviewer demonstrated the verdict file's pre-change
    pass by running `check_resume_event_evidence` against the retired resolver body (probe and
    both arms pasted in the reviewer's scratch).

    ONE NAMED EXEMPTION, and it is an ENVIRONMENTAL red rather than a code one.
    `test_both_builders_emit_debug_on_every_run_shell_exactly_when_debug[multisim]` and
    `[sensitivity]` fail on any DIRTY checkout with
    `ConfigurationError: Configuration error in field 'hhemt_sha'` raised at
    `src/hhemt/analysis.py:592` ("the running toolkit checkout is at 13e6045b but has modified
    tracked files"). Both reach `TRITONSWMM_analysis.__init__` through
    `_multisim_builder`/`_sensitivity_builder`, which no line this chunk edits is on the path
    to — the chunk's only edit in that file is six added lines in `_run`, a stub factory those
    two tests do not call. They therefore cannot be greened or reddened by this chunk and are
    allowed to be red HERE while every OTHER failure in the three files is not. The assertion is
    a SUBSET test over failing ids rather than a deselect list, so a new failure in those same
    two files still reds this test.
    """
    files = [
        "tests/test_resume_verdict_disclosed_denominator.py",
        "tests/test_resume_stepback_from_cut_off_snapshot.py",
        "tests/test_debug_flag_keep_previous_snapshots.py",
    ]
    #: Known ENVIRONMENTAL reds: the dirty-tree `hhemt_sha` guard refuses analysis construction.
    allowed_environmental = {
        "tests/test_debug_flag_keep_previous_snapshots.py::"
        "test_both_builders_emit_debug_on_every_run_shell_exactly_when_debug[multisim]",
        "tests/test_debug_flag_keep_previous_snapshots.py::"
        "test_both_builders_emit_debug_on_every_run_shell_exactly_when_debug[sensitivity]",
    }
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *files, "-q", "--tb=no", "-p", "no:cacheprovider"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    failed = {
        line.split(" - ", 1)[0][len("FAILED ") :].strip()
        for line in (proc.stdout or "").splitlines()
        if line.startswith("FAILED ")
    }
    unexpected = sorted(failed - allowed_environmental)
    tail = (proc.stdout or "").strip().splitlines()[-1:] or [""]
    assert not unexpected, (
        "failures in the three files the shape declared out of scope, minus the two known "
        f"dirty-tree-guard reds: {unexpected}; last pytest line: {tail[0]!r}"
    )


# ---------------------------------------------------------------------------
# FINDING 2 — the reader prefers a POSITION-keyed record over an IDENTITY-keyed one
# ---------------------------------------------------------------------------


def test_the_reader_prefers_the_identity_named_log_over_an_ordinal_one(tmp_path, monkeypatch):
    """Property: when BOTH names exist on disk, `read_sub_resume_context` returns the
    identity-named log.

    Class: any member tree whose config resolves the event identity while `logs/sims/` holds
    both an `evt-{event_id}` log and a legacy `evt{ordinal}` log — the state left by an arm
    re-run across the re-key.

    Why required: Decision 5 ranks "never attributing one event's record to another" FIRST, and
    this is that rule at the smallest scale — the correct record is on disk and a reader must not
    hand back the position-keyed one.

    The events CSV is ABSOLUTE here DELIBERATELY, so this test says nothing about which base
    directory a relative pointer resolves against: that is a separate property with a separate
    test (`test_the_reader_is_cwd_independent_for_a_relative_events_csv`), and coupling the two
    would let a defensible base choice fail this arm for a reason it does not assert.

    What kills it: offering the legacy candidate alongside an identity candidate that DID
    resolve, or searching roots before names so a position-keyed record in one location outranks
    an identity-keyed one in another.

    Second correct implementation under which it still passes: a reader that resolves the events
    CSV through a cached row-indexer dict, or one that takes the indexer dict from the sub yaml
    instead of re-reading the CSV.

    NEW CAPABILITY. Plausible wrong implementation: append the legacy candidate unconditionally
    (the shape of the chunk's first cut, red under it, output pasted in the reviewer's scratch).
    """
    from hhemt.eda.raw_byte_identity import read_sub_resume_context

    master = tmp_path / "master"
    logdir = master / "logs" / "sims"
    csv = _events_csv(master / _RELATIVE_CSV, "A", "B")
    identity = _seed_log(logdir, event_slug="event.A", ordinal=None, body=_RESUME_BODY)
    _seed_log(logdir, event_slug=None, ordinal=0, body="LEGACY-ORDINAL-LOG\n")

    sub = tmp_path / "scratch" / "members" / _MEMBER
    _sub_yaml(sub, master=master, csv_value=str(csv))  # ABSOLUTE: base choice is not under test

    monkeypatch.chdir(tmp_path)
    log, _interval, _schedule = read_sub_resume_context(sub, _MEMBER, 0)

    assert log is not None, "the reader returned no log at all"
    assert log.name == identity.name, (
        f"the reader returned the position-keyed record {log.name!r} while the identity-keyed {identity.name!r} exists"
    )


def test_the_reader_is_cwd_independent_for_a_relative_events_csv(tmp_path, monkeypatch):
    """Property: one on-disk tree resolves to the SAME log path from any working directory.

    Class: every member tree declaring a relative `weather_events_to_simulate` — the class the
    repository's own fixtures belong to.

    Why required: measured before the repair, the reader returned the identity-named log from
    the directory the CSV happened to resolve against and the legacy ordinal name (or no log)
    from any other. A reader whose answer depends on the caller's cwd is not reproducible, and
    the divergence is silent: the caller cannot tell "this arm has no log" from "you launched me
    from the wrong place".

    The CSV is seeded BESIDE THE YAML **and** beside the master config, so this test asserts
    determinism and identity-preference together WITHOUT prescribing which of the two defensible
    bases a relative pointer resolves against. Put it in only one place and the test silently
    becomes an assertion about a base choice it does not name.

    What kills it: resolving the CSV against the process cwd, in either direction.

    Second correct implementation under which it still passes: resolving lazily and caching the
    minted slug on the returned tuple, or accepting an absolute pointer and returning a uniform
    None when it is not absolute.
    """
    from hhemt.eda.raw_byte_identity import read_sub_resume_context

    master = tmp_path / "master"
    logdir = master / "logs" / "sims"
    _events_csv(master / _RELATIVE_CSV, "A", "B")
    identity = _seed_log(logdir, event_slug="event.A", ordinal=None, body=_RESUME_BODY)

    sub = tmp_path / "scratch" / "members" / _MEMBER
    _sub_yaml(sub, master=master, csv_value=_RELATIVE_CSV)
    # The same CSV beside the member yaml too, so EITHER defensible base resolves it.
    _events_csv(sub / _RELATIVE_CSV, "A", "B")

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    monkeypatch.chdir(elsewhere)
    from_elsewhere, _, _ = read_sub_resume_context(sub, _MEMBER, 0)
    monkeypatch.chdir(master)  # the cwd in which the RELATIVE pointer does resolve
    from_master, _, _ = read_sub_resume_context(sub, _MEMBER, 0)

    assert (from_elsewhere.name if from_elsewhere else None) == (from_master.name if from_master else None), (
        "the resolved log depends on the caller's cwd"
    )
    # Determinism alone is satisfied by resolving nothing everywhere, so the answer is also
    # pinned to the identity name: both bases carry the CSV, so a reader that resolves the
    # pointer at all resolves it to the same event.
    assert from_elsewhere is not None and from_elsewhere.name == identity.name, (
        "with the events CSV beside the yaml and beside the master config, the reader must "
        f"return {identity.name!r} from any cwd; it returned "
        f"{from_elsewhere.name if from_elsewhere else None!r}"
    )


# ---------------------------------------------------------------------------
# FINDING 3 — the legacy ordinal candidate re-admits the option Decision 5 REFUSED
# ---------------------------------------------------------------------------


def test_a_position_keyed_log_is_not_read_when_the_event_identity_is_known(tmp_path):
    """Property: with the event identity resolvable from the config, a POSITION-keyed log is
    not returned as that event's record.

    Class: any member tree carrying a legacy `evt{ordinal}` log whose owning analysis knows its
    events CSV — i.e. every pre-re-key arm on a tree whose config still points at the CSV.

    Why required: Decision 5 refused option 3 ("a read-time fallback — try the new name, then
    the old") in these words — "a shifted ordinal still hands one event another's record —
    after the fix, which is worse than before it" — and its optimization priority (1) is
    "never attributing one event's record to another". `_model_log_candidates` implements
    option 3. The coder's ground, that this reader "gates NO execution", answers priorities (2)
    and (4); it does not answer (1), which is ranked above both and does not depend on whether
    anything executes.

    What kills it: deleting the `names.append(f"model_tritonswmm_{member_id}_evt{event_iloc}.log")`
    line so the reader has cold-start behaviour (Decision 5 option 1, the approved option) —
    missing log -> None -> the figure loses its vlines, which the reader's own contract already
    states as its cost.

    Second correct implementation under which it still passes: keeping the legacy candidate but
    BOUNDED to the case where the event identity is NOT resolvable, which is exactly what this
    test's fixture excludes (the CSV is absolute here).

    NEW CAPABILITY. Plausible wrong implementation: append the legacy candidate unconditionally
    — what the chunk does; red under it (output pasted in the reviewer's scratch).
    """
    from hhemt.eda.raw_byte_identity import read_sub_resume_context

    master = tmp_path / "master"
    logdir = master / "logs" / "sims"
    csv = _events_csv(master / _RELATIVE_CSV, "A", "B")
    _seed_log(logdir, event_slug=None, ordinal=0, body="LEGACY-ORDINAL-LOG\n")

    sub = tmp_path / "scratch" / "members" / _MEMBER
    _sub_yaml(sub, master=master, csv_value=str(csv))  # ABSOLUTE: identity is knowable

    log, _interval, _schedule = read_sub_resume_context(sub, _MEMBER, 0)

    assert log is None, (
        "with the event identity resolvable the reader must not attribute a POSITION-keyed "
        f"record to it; it returned {log.name if log else None!r}"
    )


# ---------------------------------------------------------------------------
# HELD — GREEN now, must REMAIN green. Its mutation is named so the hold is checkable.
# ---------------------------------------------------------------------------


def test_the_reader_finds_an_identity_named_log_when_the_csv_is_absolute(tmp_path, monkeypatch):
    """HELD. Property: with an absolute events CSV the reader mints and finds the identity name.

    Class: any member tree with an identity-named log and an absolute
    `weather_events_to_simulate` — the shape item-4 configuration.

    Why required: this is the must-fire control bounding FINDINGS 2 and 3. Without it a reader
    that resolved nothing would satisfy every None-expecting assertion, and a broken reader and
    a cold-start reader would be indistinguishable.

    What kills it (the named mutation the coder re-runs to evidence the hold): delete the
    `if event_id is not None: names.append(model_logfile_name(...))` branch from
    `_model_log_candidates`, so no identity name is ever offered. The test then goes red.

    Second correct implementation under which it still passes: any reader that resolves the
    identity name from the yaml's pointer, an absolute path, or a cached indexer dict.
    """
    from hhemt.eda.raw_byte_identity import read_sub_resume_context

    master = tmp_path / "master"
    logdir = master / "logs" / "sims"
    csv = _events_csv(master / _RELATIVE_CSV, "A", "B")
    identity = _seed_log(logdir, event_slug="event.A", ordinal=None, body=_RESUME_BODY)

    sub = tmp_path / "scratch" / "members" / _MEMBER
    _sub_yaml(sub, master=master, csv_value=str(csv))

    monkeypatch.chdir(tmp_path)
    log, _interval, _schedule = read_sub_resume_context(sub, _MEMBER, 0)
    assert log is not None and log.name == identity.name
