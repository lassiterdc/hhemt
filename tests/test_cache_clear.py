"""Scope, refusal and liveness of `hhemt cache clear`, plus the compile-tier single reader.

Every destructive assertion here runs against an injected :class:`CacheRoots` pointing
into ``tmp_path``. That is not tidiness: a test that reached the real cache root would be
one bug away from deleting the developer's cache, and the injected value object is what
makes the destructive path testable at all.

Nothing in this module compiles a solver or executes a simulation, so it carries no
``compile_tier`` mark and runs in the fast tier.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hhemt._filelock_compat import resolve_filelock
from hhemt.cache.clear import (
    CacheRoots,
    apply_clear,
    detect_liveness,
    plan_clear,
)
from hhemt.suite._runner import _compile_precondition

REPO_ROOT = Path(__file__).resolve().parents[1]
# ASSEMBLED, never spelled: a single literal here would match the very scan this module
# runs, so the test would report its own source as a second reader. The scan asserts on
# the contiguous expression, and concatenation is what keeps this constant out of it.
REQUIRE_TIER_READER = "os.environ.get(" + '"HHEMT_REQUIRE_COMPILE_TIER"' + ")"


def _roots(tmp_path: Path) -> CacheRoots:
    """Four synthetic roots, laid out exactly as the real ones nest.

    T2 is nested INSIDE T1 because that nesting is the property two of the tests below
    turn on: a clear that removed T1 wholesale would take `_software` with it and defeat
    the venue gate by a side effect.
    """
    cache = tmp_path / "cache"
    analysis_trees = cache / "synthetic_test_runs" / "path-0123456789abcdef"
    return CacheRoots(
        model_artifacts=cache / "synthetic_test_models",
        analysis_trees=analysis_trees,
        compiled_builds=analysis_trees / "_software",
        source_checkout=cache / "_triton_canonical" / "triton",
    )


def _populate(roots: CacheRoots) -> None:
    """Give every root a file, so 'removed' is distinguishable from 'never existed'."""
    for path in (
        roots.model_artifacts,
        roots.analysis_trees / "some_case",
        roots.compiled_builds / "triton",
        roots.source_checkout,
    ):
        path.mkdir(parents=True, exist_ok=True)
        (path / "payload.txt").write_text("x", encoding="utf-8")


@pytest.fixture
def unpermitted(monkeypatch):
    """No permitted compile venue -- the state the destructive-clear hazard describes."""
    monkeypatch.delenv("HHEMT_COMPILE_VENUE", raising=False)


@pytest.fixture
def permitted(monkeypatch):
    """A permitted compile venue, so the compiled-build tier is in scope."""
    monkeypatch.setenv("HHEMT_COMPILE_VENUE", "toolchain")


def test_bare_invocation_lists_four_roots_and_removes_nothing(tmp_path, unpermitted):
    """Property: an unapplied plan touches nothing, even when the venue gate refuses.

    Class: NEW CAPABILITY. The plausible wrong implementation consults the gate only on
    the ``--apply`` path and materializes a root while reporting on it -- ``cache_root()``
    creates nothing, so the wrong version is a plan builder that calls ``mkdir`` to
    measure sizes. Red under it: ``_populate`` would no longer be needed for the
    'nothing removed' assertion, and the absent roots would become directories.

    Why required: a listing verb that changes the thing it describes manufactures the
    state it is reporting, and the operator would then be unable to see a genuinely cold
    machine.

    What kills it: any write during ``plan_clear``.

    Second correct implementation: a plan built from ``os.scandir`` with no size
    accounting at all still satisfies this test, because it asserts only on existence.
    """
    roots = _roots(tmp_path)
    _populate(roots)
    every_root = (roots.model_artifacts, roots.analysis_trees, roots.compiled_builds, roots.source_checkout)
    before = {p: p.exists() for p in every_root}

    plan = plan_clear(address="path-0123456789abcdef", roots=roots, liveness={})

    assert [d.tier for d in plan.decisions] == ["T0", "T1", "T2", "T3"]
    assert {p: p.exists() for p in before} == before, "plan_clear modified the tree"


def test_unpermitted_venue_clears_t0_and_t1_and_leaves_software(tmp_path, unpermitted):
    """Property: with no permitted venue, T0 and T1 go, ``_software`` survives, exit non-zero.

    Class: REGRESSION. This arm was red on the module's FIRST shape, which refused
    wholesale and removed nothing: that version raised ``PermissionError`` and left
    ``model_artifacts`` in place, so the two assertions below both failed. It pins the
    required behaviour in its own terms -- ``--apply`` removes T0 and T1, and T2 only
    when the compile venue is permitted.

    Why required: a wholesale refusal makes the two cheap tiers unclearable on exactly
    the machines where clearing them is safe, and the only route back to them is
    ``--include-compile-tier``, which destroys the tier the gate protects. That defect
    arrives THROUGH the gate, which is why it needs its own arm.

    What kills it: raising instead of clearing, or clearing anything under ``_software``.

    Second correct implementation: a verb that deletes T0 and T1 and reports the refused
    tier through any channel still passes, because the assertions are on the tree and on
    ``plan.refused``, not on the message.
    """
    roots = _roots(tmp_path)
    _populate(roots)
    plan = plan_clear(address="path-0123456789abcdef", roots=roots, liveness={})

    assert plan.refused
    assert "HHEMT_COMPILE_VENUE" in plan.refusal
    assert [d.tier for d in plan.targets] == ["T0", "T1"]

    removed = apply_clear(plan)

    assert [tier for tier, _bytes in removed] == ["T0", "T1"]
    assert not roots.model_artifacts.exists(), "T0 survived a gate that does not cover it"
    assert not (roots.analysis_trees / "some_case").exists(), "T1 survived a gate that does not cover it"
    assert roots.compiled_builds.exists(), "cleared a tier the venue gate exists to protect"
    assert (roots.compiled_builds / "triton" / "payload.txt").exists()
    assert roots.source_checkout.exists()


def test_permitted_venue_clears_t0_t1_and_t2_but_not_t3(tmp_path, permitted):
    """Property: T0, T1 and T2 go; T3 survives because its flag defaults off.

    Class: NEW CAPABILITY. The plausible wrong implementation defaults
    ``--include-source-clone`` on, which costs every later worktree a full re-clone of the
    TRITON object store; red under it because ``source_checkout`` would be gone.

    Why required: re-cloning is network-bound and the estate harness has a measured
    rate-limit hazard on concurrent borrows.

    What kills it: deleting T3 without the flag, or leaving any of T0/T1/T2 behind.

    Second correct implementation: removing T1 by renaming it aside and deleting lazily
    satisfies this, which asserts on final existence only.
    """
    roots = _roots(tmp_path)
    _populate(roots)
    plan = plan_clear(address="path-0123456789abcdef", roots=roots, liveness={})

    assert not plan.refused
    apply_clear(plan)

    assert not roots.model_artifacts.exists()
    assert not (roots.analysis_trees / "some_case").exists()
    assert not roots.compiled_builds.exists()
    assert roots.source_checkout.exists()


def test_include_compile_tier_overrides_the_venue_refusal(tmp_path, unpermitted):
    """Property: overriding the refusal removes T2 while T0/T1 also go.

    Class: NEW CAPABILITY. The plausible wrong implementation honours
    ``--include-compile-tier`` by widening the SCOPE test but leaving the refusal set; red
    under it because ``apply_clear`` would still raise.

    Why required: the override is the operator's only route to a genuinely cold machine.

    What kills it: an override that changes the flag but not the refusal.

    Second correct implementation: an override implemented as a separate plan-builder call
    satisfies this, because the assertion is on the resulting tree.
    """
    roots = _roots(tmp_path)
    _populate(roots)
    plan = plan_clear(
        include_compile_tier=True,
        address="path-0123456789abcdef",
        roots=roots,
        liveness={},
    )

    assert not plan.refused
    apply_clear(plan)
    assert not roots.compiled_builds.exists()
    assert not (roots.analysis_trees / "some_case").exists()
    assert not roots.model_artifacts.exists()


def test_live_compile_lock_skips_t2_and_still_clears_the_others(tmp_path, permitted):
    """Property: a held T2 lock skips that root alone; the ungated tiers still clear.

    Class: NEW CAPABILITY. The plausible wrong implementation treats any liveness hit as
    a plan-wide refusal; red under it because ``model_artifacts`` would survive.

    Why required: a busy cache is not a failed clear, and refusing everything because one
    build is in flight makes the verb unusable on a working machine.

    What kills it: clearing a root whose lock is held.

    Second correct implementation: a caller that filters live roots in the CLI instead of
    inside plan_clear satisfies this, because the assertion is on the tree.
    """
    roots = _roots(tmp_path)
    _populate(roots)
    plan = plan_clear(
        address="path-0123456789abcdef",
        roots=roots,
        liveness={"T2": ("/somewhere/.build.compile.lock",), "T3": ()},
    )

    assert not plan.refused
    apply_clear(plan)
    assert roots.compiled_builds.exists(), "cleared a tier whose compile lock is held"
    assert not roots.model_artifacts.exists()
    assert not (roots.analysis_trees / "some_case").exists()


def test_source_clone_flag_adds_t3(tmp_path, permitted):
    """Property: T3 is removed only when its flag is passed.

    Class: NEW CAPABILITY. The plausible wrong implementation keys T3 on the existence of
    its lock file rather than on the flag; red under it because ``{}`` liveness would
    leave T3 in place.

    Why required: T3 is a shared object store, so removing it costs every worktree.

    What kills it: an in-scope T3 with the flag absent.

    Second correct implementation: a plan that lists T3 but marks it out of scope
    satisfies this, because the assertion is on existence after apply.
    """
    roots = _roots(tmp_path)
    _populate(roots)
    plan = plan_clear(
        include_source_clone=True,
        address="path-0123456789abcdef",
        roots=roots,
        liveness={},
    )
    apply_clear(plan)
    assert not roots.source_checkout.exists()


def test_detect_liveness_reports_a_held_compile_lock(tmp_path):
    """Property: a HELD per-build-dir compile lock makes the tier-2 arm fire.

    Class: NEW CAPABILITY. The plausible wrong implementation returns the empty tuple
    unconditionally -- and a dead detector is green across every other test in this
    module, because each injects its own ``liveness`` dict and none invokes
    ``detect_liveness`` at all. Red under that implementation because ``live["T2"]``
    would be empty.

    Why required: this arm is the only thing standing between the verb and deleting a
    build directory a concurrent session is compiling into.

    What kills it: a detector that does not probe the lock, or that probes the wrong
    root -- ``_software`` rather than a named build dir.

    Second correct implementation: a detector that shells out to ``lsof`` on the build
    tree satisfies this, because the assertion is on the reported tuple, not on the
    probing mechanism.
    """
    roots = _roots(tmp_path)
    lock_path = roots.compiled_builds / "triton" / ".build_tritonswmm_cpu.compile.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    holder = resolve_filelock(str(lock_path), timeout=10)

    with holder:
        live = detect_liveness(roots)

    assert live["T2"], "a held compile lock did not make the tier-2 arm fire"
    assert any(path.endswith(".compile.lock") for path in live["T2"])


def test_detect_liveness_ignores_an_unheld_compile_lock(tmp_path):
    """Property: a lock FILE that nobody holds does not make the tier-2 arm fire.

    Class: NEW CAPABILITY. The plausible wrong implementation tests the lock file's
    EXISTENCE rather than its heldness; red under it because the file exists and
    ``live["T2"]`` must still be empty. This is the mirror of the held-lock arm, and it
    is the one that separates a real probe from a file-existence test.

    Why required: compile locks are left on disk after a build finishes, so an
    existence test would skip tier 2 on every machine that had ever compiled.

    What kills it: reporting a stale lock file as a live holder.

    Second correct implementation: a detector that reads the lock file's PID and checks
    that process satisfies this, because nothing holds the lock and no PID is recorded.
    """
    roots = _roots(tmp_path)
    lock_path = roots.compiled_builds / "triton" / ".build_tritonswmm_cpu.compile.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.touch()
    assert lock_path.exists(), "the fixture did not create the lock file, so the arm proves nothing"

    live = detect_liveness(roots)

    assert live["T2"] == ()


def test_detect_liveness_reports_a_held_canonical_lock(tmp_path):
    """Property: a HELD canonical-store lock makes the tier-3 arm fire.

    Class: NEW CAPABILITY. The plausible wrong implementation probes the lock inside the
    store rather than beside it; red under it because clearing T3 deletes the store, so a
    lock placed within it cannot survive the operation it guards.

    Why required: the canonical object store is shared by every worktree, and deleting it
    costs each of them a full re-clone.

    What kills it: a probe sited too deep, or one keyed on the wrong directory.

    Second correct implementation: a detector that consults the store's `.git` lock
    satisfies this, because the assertion is on the tier that fired.
    """
    roots = _roots(tmp_path)
    roots.source_checkout.mkdir(parents=True, exist_ok=True)
    lock_path = roots.source_checkout.parent / ".triton.canonical.lock"
    holder = resolve_filelock(str(lock_path), timeout=10)

    with holder:
        live = detect_liveness(roots)

    assert live["T3"], "a held canonical lock did not make the tier-3 arm fire"
    assert any(path.endswith(".triton.canonical.lock") for path in live["T3"])


def test_detect_liveness_ignores_an_unheld_canonical_lock(tmp_path):
    """Property: an unheld canonical lock file leaves the tier-3 arm silent.

    Class: NEW CAPABILITY. The plausible wrong implementation reports existence; red
    under it because the file is present and the arm must stay empty.

    Why required: the canonical lock is created once and left behind, so treating it as
    a liveness signal would make ``--include-source-clone`` refuse forever after the
    first borrow.

    What kills it: reporting a stale lock file as a live holder.

    Second correct implementation: a PID-reading detector satisfies this, because no
    process is recorded against the file.
    """
    roots = _roots(tmp_path)
    roots.source_checkout.mkdir(parents=True, exist_ok=True)
    lock_path = roots.source_checkout.parent / ".triton.canonical.lock"
    lock_path.touch()

    live = detect_liveness(roots)

    assert live["T3"] == ()


class _Args:
    """Minimal argparse stand-in for the one attribute the precondition reads."""

    declare_non_compile_bearing = False


def _marks(*node_ids: str, marked: bool) -> dict[str, list[str]]:
    return {n: (["compile_tier"] if marked else []) for n in node_ids}


@pytest.fixture
def no_compile_conjuncts(monkeypatch):
    monkeypatch.delenv("HHEMT_COMPILE_VENUE", raising=False)
    monkeypatch.delenv("HHEMT_REQUIRE_COMPILE_TIER", raising=False)


def test_non_compile_bearing_plan_is_never_refused(no_compile_conjuncts):
    """Property: a plan covering no compile_tier node proceeds with BOTH conjuncts false.

    Class: NEW CAPABILITY. The plausible wrong implementation unions the conjuncts with
    the env values alone, refusing every plan on a machine without a venue; red under it
    because this call would raise.

    Why required: the gate is about what a run was asked to COVER, so a suite subset
    that touches no compile node must remain runnable on a laptop.

    What kills it: a refusal keyed on the environment rather than on coverage.

    Second correct implementation: a caller that asks pytest for the mark set instead of
    reading the collect payload satisfies this, because the assertion is on the outcome.
    """
    record = _compile_precondition(_Args(), ["a::b", "a::c"], _marks("a::b", "a::c", marked=False))

    assert record["compile_bearing"] is False
    assert record["satisfied"] is True
    assert record["compile_venue_permitted"] is False
    assert record["compile_tier_required"] is False


def test_compile_bearing_plan_names_both_missing_conjuncts(no_compile_conjuncts):
    """Property: refusal names BOTH conjuncts when neither holds.

    Class: NEW CAPABILITY. The plausible wrong implementation reports a bare "refused";
    red under it because neither `capability` nor `anti-silence` would appear.

    Why required: an operator told only that the run was refused cannot tell which export
    to set, so the diagnostic is the whole value of the check.

    What kills it: a message naming one conjunct when two are missing.

    Second correct implementation: a caller raising two chained SystemExits still satisfies
    this, because the assertion is on the combined text.
    """
    with pytest.raises(SystemExit) as excinfo:
        _compile_precondition(_Args(), ["a::b"], _marks("a::b", marked=True))

    message = str(excinfo.value)
    assert "capability" in message
    assert "anti-silence" in message
    assert "1 compile-bearing node(s)" in message


def test_compile_bearing_plan_names_only_the_missing_conjunct(monkeypatch):
    """Property: with the venue permitted, only the anti-silence conjunct is named.

    Class: NEW CAPABILITY. The plausible wrong implementation emits the same message for
    every miss; red under it because `capability` would appear where it does not apply.

    Why required: naming a conjunct that already holds sends the operator to the wrong
    export.

    What kills it: a message that always names both.

    Second correct implementation: a refusal that prints a table of both conjuncts with
    their values satisfies this, since the assertion is on the missing one being named.
    """
    monkeypatch.setenv("HHEMT_COMPILE_VENUE", "toolchain")
    monkeypatch.delenv("HHEMT_REQUIRE_COMPILE_TIER", raising=False)

    with pytest.raises(SystemExit) as excinfo:
        _compile_precondition(_Args(), ["a::b"], _marks("a::b", marked=True))

    message = str(excinfo.value)
    assert "anti-silence" in message
    assert "capability" not in message


def test_compile_bearing_plan_records_the_resolved_pair(permitted, monkeypatch):
    """Property: both conjuncts true admits the run and records the pair it read.

    Class: NEW CAPABILITY. The plausible wrong implementation records the resolved values
    as constants or omits them; red under it because the asserted keys would be absent.

    Why required: the disclosure and the ledger read this pair to decide whether a green
    run may discharge the compile obligation.

    What kills it: a record that omits either conjunct or reports a stale value.

    Second correct implementation: reading the values once in `plan` and passing them in
    satisfies this, because the assertion is on the returned mapping.
    """
    monkeypatch.setenv("HHEMT_REQUIRE_COMPILE_TIER", "1")

    marks = _marks("a::b", "a::c", marked=False) | {"a::b": ["compile_tier"]}
    record = _compile_precondition(_Args(), ["a::b", "a::c"], marks)

    assert record["satisfied"] is True
    assert record["compile_bearing"] is True
    assert record["compile_tier_node_count"] == 1
    assert record["compile_venue_permitted"] is True
    assert record["compile_tier_required"] is True


def test_declared_non_compile_bearing_overrides(no_compile_conjuncts):
    """Property: an explicit declaration admits a compile-bearing plan and is recorded.

    Class: NEW CAPABILITY. The plausible wrong implementation infers non-compile-bearing
    from an empty mark set only, ignoring the declaration; red under it because the call
    would raise.

    Why required: "covered no compile nodes" and "chose not to cover them" are different
    facts, and only the second is a decision worth recording.

    What kills it: a declaration that is accepted but not recorded.

    Second correct implementation: recording the declaration as a string reason rather
    than a boolean satisfies this only if the flag is truthy -- so the wrong version is
    red, which is the discrimination this arm exists for.
    """
    args = _Args()
    args.declare_non_compile_bearing = True

    record = _compile_precondition(args, ["a::b"], _marks("a::b", marked=True))

    assert record["declared_non_compile_bearing"] is True
    assert record["satisfied"] is True
    assert record["compile_venue_permitted"] is False


def test_require_compile_tier_has_exactly_one_reader_and_it_is_in_src():
    """Property: the reader string occurs in exactly one repo file, under src/hhemt/.

    Class: REGRESSION -- RECORDED, not self-demonstrated. On the pre-change tree the scan
    below returned one file and it was the wrong one, so the assertion failed on its first
    element:

        E  AssertionError: assert ['tests/utils_for_testing.py'] == ['src/hhemt/_...ile_venue.py']
        E    At index 0 diff: 'tests/utils_for_testing.py' != 'src/hhemt/_compile_venue.py'

    Measured against that tree rather than recalled: at the commit that moved the read
    (`6fbf1d3c`, the most recent commit touching `tests/utils_for_testing.py`), that file
    carries the reader and the toolkit module does not -- which is why the set was a
    singleton sitting in the TEST tree.

    Recorded rather than re-demonstrated here, and the reason is a measurement. A query
    against this checkout's own history cannot be relied on in this repository's venues:
    CI fetches at depth 1 (``.github/workflows/test.yml`` uses ``actions/checkout`` with no
    ``fetch-depth``) and ``main`` takes squash merges only, so the move commit is neither
    present nor an ancestor there, and under ``check=True`` an unresolvable object raises
    rather than degrading. A history this test BUILDS would be worse than either, because
    it would assert on content the test itself authored -- an assertion that cannot fail as
    a function of this repository is not evidence about it.

    Why required: two readers are free to diverge on the spelling of the arming value, and
    a runner that accepts a compile-bearing plan while the fixture skips the tier yields a
    green run whose output never says it could not compile.

    What kills it: any second environment read of the compile-tier token, anywhere -- a
    second reader in the toolkit, or one reintroduced under `tests/`. The scan asserts on
    the contiguous reader expression and this docstring deliberately does not spell it, or
    this module would report itself.

    Second correct implementation: moving the read into another `src/` module would fail
    the path assertion, which is deliberate -- the single-reader module is named by the
    shape, not by preference.
    """
    hits = sorted(
        str(path.relative_to(REPO_ROOT))
        for base in (REPO_ROOT / "src", REPO_ROOT / "tests")
        for path in base.rglob("*.py")
        if REQUIRE_TIER_READER in path.read_text(encoding="utf-8", errors="ignore")
    )
    assert hits == ["src/hhemt/_compile_venue.py"]


def test_require_compile_tier_delegates():
    """Property: the test helper holds no read of its own and returns the src predicate.

    Class: REGRESSION (red on the pre-change tree, where the body performed the read).

    Why required: `plan` and the fixtures must resolve one value from one place.

    What kills it: a body that reads the environment, even if it agrees today.

    Second correct implementation: a helper that returns
    `compile_tier_required()` through a local alias satisfies this; the assertion is on
    the absence of the reader string and the presence of the delegation call.
    """
    import inspect

    from tests.utils_for_testing import require_compile_tier

    source = inspect.getsource(require_compile_tier)
    body = source.split('"""')[-1]
    assert "compile_tier_required()" in body
    assert REQUIRE_TIER_READER not in body


def test_clear_verb_exits_two_when_the_compile_gate_skips_tier_two(tmp_path, unpermitted, monkeypatch):
    """Property: the VERB exits 2 on the gate path, and the tree shows T0/T1 gone, T2 intact.

    Class: REGRESSION. Red on the round-1 tree, which refused wholesale: that version
    raised out of ``apply_clear``, so the CLI exited non-zero for the WRONG reason and
    left T0 and T1 in place -- both the ``exit_code == 2`` and the "T0 gone" assertion
    fail there. Red also on any later edit that drops the ``raise typer.Exit(code=2)``
    line, which is the failure mode this arm exists for: without it every other test in
    this module stays green while the requirement's named instrument stops holding.

    Why required: the process exit status is the instrument the requirement names, and a
    library-level test cannot observe it. Nothing else in the suite invokes this verb
    through the CLI, so this is the only automated arm over it.

    What kills it: dropping the exit statement, printing the notice without it, or
    taking T2 on the refusal path.

    Second correct implementation: a verb whose gate path routes through a shared
    exit-code mapper instead of ``typer.Exit`` satisfies this, because the assertion is
    on the observed process status and the resulting tree.

    Invocation isolation: arm one is unreachable here -- the cache root is per-worktree
    by design, so a second tree resolves a different tier -- and the discrimination is
    shown instead: deleting the ``raise typer.Exit(code=2)`` statement greens every
    other arm in this module and reds only this one.
    """
    from typer.testing import CliRunner

    import hhemt.cache.paths as cache_paths
    from hhemt.cache.paths import tier_address_for
    from hhemt.cli import app

    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(cache_paths, "cache_root", lambda: cache_dir)

    address = tier_address_for(Path.cwd())
    model_artifacts = cache_dir / "synthetic_test_models"
    analysis_trees = cache_dir / "synthetic_test_runs" / address
    compiled_builds = analysis_trees / "_software"
    source_checkout = cache_dir / "_triton_canonical" / "triton"
    for path in (model_artifacts, analysis_trees / "some_case", compiled_builds / "triton", source_checkout):
        path.mkdir(parents=True, exist_ok=True)
        (path / "payload.txt").write_text("x", encoding="utf-8")

    result = CliRunner().invoke(app, ["cache", "clear", "--apply"])

    assert result.exit_code == 2, result.output
    assert not model_artifacts.exists(), "T0 survived a gate that does not cover it"
    assert not (analysis_trees / "some_case").exists(), "T1 survived a gate that does not cover it"
    assert (compiled_builds / "triton" / "payload.txt").exists(), "cleared the tier the gate protects"
    assert source_checkout.exists()
    assert "HHEMT_COMPILE_VENUE" in result.output, "the non-zero exit did not name the gate"
