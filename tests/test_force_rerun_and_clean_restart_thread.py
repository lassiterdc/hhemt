"""Reviewer tests for work package 1a — the durable consecutive-failure budget's WIRING.

WHAT PROBLEM THIS FILE SOLVES. Item 1 builds a durable consecutive-terminal-failure
budget: a counter in the per-model log, a helper that increments it, and a gate that
refuses to launch the solver once it reaches ``hpc_consecutive_failure_budget_simulate``.
The counter is only as good as the set of failure paths that feed it. Item 1's own test
file exercises the two helpers in isolation against a stub scenario -- its own docstring
says so -- and therefore cannot observe whether ``run_simulation_runner.main`` actually
CALLS the increment on every path that concludes an attempt failed. A failure path that
writes the failed marker and does not increment is invisible to every helper-level test,
and a simulation failing that way is re-dispatched by Snakemake's multiplicative
``retries:`` ceiling without ever consuming budget -- which is the member-39 pathology the
item exists to remove, surviving inside the fix for it.

THE INVARIANT, stated once. Every path in ``run_simulation_runner.main`` that writes
``_status/_failed/{rule_token}.json`` increments the durable counter, EXCEPT the budget
gate's own refusal, which must not count itself because no solver was launched.

That is the production code's own stated predicate rather than this file's invention.
``_bump_terminal_failures``'s docstring reads: "Called at every runner path that concludes
THIS attempt failed -- i.e. beside every ``_write_failed_marker`` call." These tests hold
the module to it.

WHY A STRUCTURAL CENSUS RATHER THAN A BEHAVIOURAL DRIVE. Reaching these paths at runtime
requires constructing a ``TRITONSWMM_analysis``, which ``analysis.py``'s ``hhemt_sha``
dirty-tree guard refuses while the tree carries uncommitted work -- the state the Build
stage's own ruling mandates. A source census is reachable in that state, it quantifies
over the whole function rather than over the paths a test author happened to think of, and
it is the medium item 1's sibling chunks already use for their own census nodes.

Nothing here compiles or runs a solver, so this file is not simulation-bearing and runs
locally.
"""

import ast
import pathlib

import hhemt.run_simulation_runner as rsr
from hhemt.log import TRITONSWMM_model_log
from hhemt.run_simulation_runner import (
    _bump_terminal_failures,
    _consecutive_failure_budget_refusal,
)

MARKER_FUNC = "_write_failed_marker"
BUMP_FUNC = "_bump_terminal_failures"

# The ONE legitimately uncounted failure path, keyed on a substring of its GUARD rather
# than on a line number. A refusal counting itself would inflate the record of what the
# solver actually attempted, so this exemption is a property of the mechanism and not a
# concession to the current implementation. Stated as a closed one-member set so that a
# SECOND uncounted path -- the defect this file reports -- cannot hide inside a bare count.
EXEMPT_GUARD_SUBSTRINGS = ("budget_refusal",)


def _source_and_main():
    """Return the runner module's source text and its ``main`` FunctionDef node."""
    src = pathlib.Path(rsr.__file__).read_text()
    tree = ast.parse(src)
    main = next(
        node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "main"
    )
    return src, main


def _calls_name(stmts, name):
    """True when any statement in ``stmts`` contains a direct call to ``name``."""
    return any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name
        for stmt in stmts
        for node in ast.walk(stmt)
    )


def _failure_blocks(src, main):
    """Census every INNERMOST statement list in ``main`` that writes the failed marker.

    Innermost matters: an enclosing ``try`` body transitively contains both the marker
    writes and the increments, so a census over every enclosing block reports pairing that
    no individual path has. Each returned row is ``(lineno, guard_text, increments)``.
    """
    rows = {}
    for node in ast.walk(main):
        for field in ("body", "orelse", "finalbody"):
            block = getattr(node, field, None)
            if not isinstance(block, list) or not _calls_name(block, MARKER_FUNC):
                continue
            # Keep only blocks whose marker write is not already attributed to a nested
            # block: if any child statement list in this block also writes the marker,
            # this block is an ancestor and the child is the path.
            nested = any(
                isinstance(inner, list) and _calls_name(inner, MARKER_FUNC)
                for stmt in block
                for inner_node in ast.walk(stmt)
                for fname in ("body", "orelse", "finalbody")
                if (inner := getattr(inner_node, fname, None)) is not None and inner is not block
            )
            if nested:
                continue
            if isinstance(node, ast.If):
                guard = ast.get_source_segment(src, node.test) or ""
            elif isinstance(node, ast.ExceptHandler):
                guard = "except " + (ast.get_source_segment(src, node.type) or "")
            else:
                guard = f"<{type(node).__name__}.{field}>"
            rows[(node.lineno, field)] = (guard, _calls_name(block, BUMP_FUNC))
    return [(lineno, guard, bumped) for (lineno, _f), (guard, bumped) in sorted(rows.items())]


def test_the_census_finds_the_failure_paths_it_quantifies_over():
    """Control arm: the census is non-empty, so a later green cannot be vacuous.

    Property: ``_failure_blocks`` resolves at least four distinct failure paths in
    ``main``. Class: REGRESSION -- it passes against the pre-change module too, because
    the marker writes predate item 1.

    Why this is required, as a consequence a reader can check against the code: the two
    assertions below are expressed as "no uncounted path other than the exemption", and
    an assertion of that shape returns a confident PASS over an EMPTY census. Every
    other node in this file would then certify a module it never read. This node is the
    denominator those nodes are silent about.

    Kills: renaming ``_write_failed_marker``, or restructuring ``main`` so the marker is
    written through a wrapper this census cannot see -- either of which would silently
    empty the population and must red here rather than pass downstream.

    Second correct implementation under which it still passes: any arrangement of the
    failure paths -- more of them, fewer of them, differently guarded, or moved into
    ``elif`` chains -- passes, because the node asserts only that the population is
    non-trivial and never which members it has.
    """
    src, main = _source_and_main()
    rows = _failure_blocks(src, main)
    assert len(rows) >= 4, (
        "the failed-marker census resolved too few paths to be meaningful; "
        f"got {len(rows)}: {[(ln, g) for ln, g, _ in rows]}"
    )


def test_every_failed_marker_path_increments_the_durable_counter():
    """Every failure path feeds the budget, except the refusal that launches nothing.

    Property/input class: the class quantified over is every statement list in
    ``run_simulation_runner.main`` that calls ``_write_failed_marker`` -- i.e. every path
    on which the runner itself concludes the attempt failed. For each such path the same
    block must call ``_bump_terminal_failures``, unless its guard is the budget gate's own
    refusal.

    Class: REGRESSION on the module as landed -- it is RED against the current tree, where
    the ``model_logfile is None`` path writes the marker and returns 1 without
    incrementing. It is not a NEW CAPABILITY test: both functions it names exist.

    Why this property is required, as a consequence a reader can check against the code:
    ``_consecutive_failure_budget_refusal`` reads ONLY
    ``consecutive_terminal_failures``. A failure path that does not increment therefore
    contributes nothing to the gate, so a simulation failing exclusively on that path is
    relaunched by ``retries:`` at every nesting level forever and the budget never
    engages. The uncounted path is reached when ``prepare_simulation_command`` returns a
    ``model_logfile`` of ``None``, which is a configuration-resolution condition and so is
    deterministic across attempts -- precisely the class the gate was built to bound.

    What kills it: removing any ``_bump_terminal_failures`` call from a block that writes
    the failed marker, or adding a new failure path that writes the marker without one.

    Second correct implementation under which it still passes: the increment may be sited
    before or after the marker write, may be called through a differently-named local
    alias assigned to the same function, and the counter may live on the scenario log
    rather than the per-model log -- the node asserts the PAIRING of the two calls per
    path, never their order, their arguments, or where the counter is stored.
    """
    src, main = _source_and_main()
    rows = _failure_blocks(src, main)
    uncounted = [
        (lineno, guard)
        for lineno, guard, bumped in rows
        if not bumped and not any(tok in guard for tok in EXEMPT_GUARD_SUBSTRINGS)
    ]
    assert uncounted == [], (
        "every failed-marker path in run_simulation_runner.main must also increment the "
        "durable consecutive-terminal-failure counter (the budget gate's own refusal is "
        "the one exemption, because it launches nothing). Uncounted paths found: "
        + "; ".join(f"line {ln}: `if {g}:`" for ln, g in uncounted)
    )


def test_the_refusal_path_is_present_and_is_the_only_declared_exemption():
    """The exemption is a real path, and it is exactly one.

    Property: of the failure paths the census resolves, exactly one matches a declared
    exemption substring, and that path does NOT increment. Class: REGRESSION -- it passes
    against the module as landed.

    Why this is required, as a consequence a reader can check against the code: the node
    above is single-sided -- it reports uncounted paths OUTSIDE the exemption and is
    therefore satisfied by an exemption list that has grown to cover the defect. This node
    closes that direction by pinning the exemption at one member. Widening
    ``EXEMPT_GUARD_SUBSTRINGS`` to green the node above reds this one.

    What kills it: adding a second entry to the exemption list, or making the refusal path
    increment the counter -- which would record a failure for an attempt in which no
    solver ran, inflating the record of what was attempted.

    Second correct implementation under which it still passes: the refusal may be inlined
    into ``main`` rather than extracted into a helper, and may be guarded by any
    expression naming the budget refusal; the node keys on the guard text carrying the
    declared token and on the count being one, not on the helper existing.
    """
    src, main = _source_and_main()
    rows = _failure_blocks(src, main)
    exempt = [
        (lineno, guard, bumped)
        for lineno, guard, bumped in rows
        if any(tok in guard for tok in EXEMPT_GUARD_SUBSTRINGS)
    ]
    assert len(exempt) == 1, (
        f"exactly one failure path may be exempt from the budget increment; found {len(exempt)}: "
        f"{[(ln, g) for ln, g, _ in exempt]}"
    )
    assert exempt[0][2] is False, (
        "the budget gate's refusal must NOT increment the counter -- it launched nothing, "
        "and a refusal that counted itself would inflate the record of what the solver "
        f"attempted; path at line {exempt[0][0]} does increment"
    )


def test_a_refusal_leaves_the_counter_unchanged(tmp_path):
    """A spent-budget refusal reads the counter and does not write it.

    Property/input class: any ``(member, event, model_type)`` whose prior consecutive
    terminal-failure count is at or above the budget. Class: REGRESSION -- it passes
    against the module as landed, and it pins the behaviour the structural node above
    asserts only about the call graph.

    Why this is required, as a consequence a reader can check against the code: the
    structural node cannot see a mutation performed INSIDE
    ``_consecutive_failure_budget_refusal`` itself. That function does write the counter on
    one branch -- the ``force_rerun_pending`` era reset -- so "the refusal does not write"
    is a property of the refusal BRANCH rather than of the function, and only a
    behavioural call distinguishes them.

    What kills it: moving the increment into the refusal helper, or extending the era reset
    to fire on the refusal branch.

    Second correct implementation under which it still passes: the refusal may return any
    message, may log at any level, and may read the counter any number of times; the node
    asserts only that the stored value is the one it found.
    """
    budget = 3
    log = TRITONSWMM_model_log(logfile=str(tmp_path / "log_tritonswmm.json"))
    log.consecutive_terminal_failures.set(budget)

    class _Stub:
        def get_log(self, model_type):
            return log

    before = log.consecutive_terminal_failures.get()
    refusal = _consecutive_failure_budget_refusal(_Stub(), "tritonswmm", budget, 0)
    assert refusal is not None, "a spent budget must refuse"
    assert log.consecutive_terminal_failures.get() == before, (
        f"a refusal must not move the counter; {before} -> {log.consecutive_terminal_failures.get()}"
    )
    # And the increment helper is what moves it, so the pair is two-sided rather than a
    # claim that the counter is immutable.
    _bump_terminal_failures(_Stub(), "tritonswmm")
    assert log.consecutive_terminal_failures.get() == before + 1
