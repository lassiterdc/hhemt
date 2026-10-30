"""Submit-path driver claim + generated-Snakefile write safety.

Two mechanisms, both introduced after the 2026-08-31 `norfolk_stochastic`
double-launch in which three drivers stacked on one analysis tree:

1. ``SnakemakeWorkflowBuilder._acquire_submit_driver_claim`` -- the submit-path
   sibling of ``_acquire_reprocess_driver_claim``. The submit path WROTE an
   orchestrator sentinel and never READ the gate; writing is not gating.
2. ``_write_snakefile_atomic`` / ``_resolve_snakefile_path`` -- the write is made
   atomic and one-deep archiving, and a DRY RUN never displaces an existing
   production Snakefile.

The discriminating tests are ``test_unmatched_override_raises_and_unlinks_nothing``
and ``test_override_is_scoped_to_the_named_driver``: an implementation that treats
the override as a blanket skip passes every happy path and fails only those two.
"""

from __future__ import annotations

import os
import re
import socket
import subprocess
import types

import pytest

from hhemt import orchestrator_sentinels as osent
from hhemt import slurm_liveness as sl
from hhemt import workflow as wf
from hhemt.exceptions import WorkflowError
from hhemt.workflow import (
    SnakemakeWorkflowBuilder,
    _resolve_snakefile_path,
    _write_snakefile_atomic,
)


class _StubBuilder:
    """Minimal stand-in exposing only what the claim helper touches."""

    def __init__(self, analysis_dir):
        self.analysis_paths = types.SimpleNamespace(analysis_dir=analysis_dir)
        # _max_plausible_job_lifetime_min(cfg) reads hpc_total_job_duration_min.
        self.cfg_analysis = types.SimpleNamespace(hpc_total_job_duration_min=60)

    _acquire_submit_driver_claim = SnakemakeWorkflowBuilder._acquire_submit_driver_claim
    _orchestrator_liveness_gate = SnakemakeWorkflowBuilder._orchestrator_liveness_gate
    _tmux_session_is_live = SnakemakeWorkflowBuilder._tmux_session_is_live
    _get_module_load_prefix = lambda self: ""  # noqa: E731


@pytest.fixture()
def builder(tmp_path):
    return _StubBuilder(tmp_path)


def _write_live_local_sentinel(analysis_dir, driver_id):
    """A same-host `local` sentinel whose pid is alive (ours) -> gate says ALIVE."""
    return osent.write_orchestrator_sentinel(
        analysis_dir,
        driver_id=driver_id,
        workflow_submission_mode="local",
        pid=os.getpid(),
    )


def _write_foreign_sentinel(analysis_dir, driver_id, monkeypatch, *, jobid, mode="batch_job"):
    """A detached sentinel whose ORIGIN HOST is another node.

    That is the ordinary case rather than an edge: a batch_job/tmux driver is
    launched from a compute node, so any later resume from the login node (or
    from a different allocation) sees a foreign origin. The origin is stubbed on
    the WRITER's module only (``osent.socket``), so the sentinel records
    "other-node" while ``workflow.socket.gethostname()`` still returns this host
    and the gate therefore sees a genuinely foreign origin.
    """
    monkeypatch.setattr(osent, "socket", types.SimpleNamespace(gethostname=lambda: "other-node"))
    return osent.write_orchestrator_sentinel(
        analysis_dir,
        driver_id=driver_id,
        workflow_submission_mode=mode,
        slurm_jobid=jobid,
    )


# --------------------------------------------------------------------------- #
# claim: gate behaviour
# --------------------------------------------------------------------------- #


def test_dry_run_claims_nothing_and_writes_no_sentinel(builder, tmp_path):
    assert builder._acquire_submit_driver_claim(tmp_path, workflow_submission_mode="batch_job", dry_run=True) is None
    assert not osent.orchestrator_dir(tmp_path).exists()


def test_clean_tree_claims_and_records_the_real_submission_mode(builder, tmp_path):
    did = builder._acquire_submit_driver_claim(tmp_path, workflow_submission_mode="batch_job", dry_run=False)
    assert did
    payloads = osent.read_orchestrator_sentinels(tmp_path)
    assert len(payloads) == 1
    # The submit path records its REAL mode, not a synthetic "local".
    assert payloads[0]["workflow_submission_mode"] == "batch_job"
    assert payloads[0]["hostname"] == socket.gethostname()


def test_live_same_host_driver_refuses_with_an_actionable_message(builder, tmp_path):
    _write_live_local_sentinel(tmp_path, f"{os.getpid()}-{socket.gethostname()}-deadbeef")
    with pytest.raises(WorkflowError) as excinfo:
        builder._acquire_submit_driver_claim(tmp_path, workflow_submission_mode="local", dry_run=False)
    msg = str(excinfo.value)
    assert "--override-live-driver" in msg
    assert "ORIGIN HOST" in msg
    # No second claim was written.
    assert len(osent.read_orchestrator_sentinels(tmp_path)) == 1


# --------------------------------------------------------------------------- #
# claim: the refusal's NEXT STEPS remedy ladder
# --------------------------------------------------------------------------- #

_LADDER_HEADER = "NEXT STEPS, in order of preference:"


def _refusal_ladder(builder, analysis_dir):
    """Trigger the gate's refusal and return the printed NEXT STEPS text.

    One extractor for every ladder arm: the ladder is a property of the
    REFUSAL, so an arm that built its own message would be asserting on a
    string it also wrote. The gate's prepended stderr is dropped because the
    ladder is the text this chunk owns.
    """
    _write_live_local_sentinel(analysis_dir, f"{os.getpid()}-{socket.gethostname()}-ladder000")
    with pytest.raises(WorkflowError) as excinfo:
        builder._acquire_submit_driver_claim(analysis_dir, workflow_submission_mode="local", dry_run=False)
    msg = str(excinfo.value)
    assert _LADDER_HEADER in msg, "the refusal must carry the remedy ladder at all"
    return msg[msg.index(_LADDER_HEADER) :]


def test_ssh_remedy_is_gated_on_the_measured_partitions_discriminator(builder, tmp_path):
    """The ssh remedy is offered only after the origin host's class is MEASURED.

    Property: within the ladder, the `scontrol show node`/`Partitions=` rung
    precedes the ssh remedy, and rung 2 is not itself an unconditional
    cross-node ssh remedy.

    Class: NEW CAPABILITY -- the plausible wrong implementation is the rung
    keyed on the HOSTNAME PREFIX (UVA login and compute nodes share `udc-`)
    instead of on the `Partitions=` field; under it the `` `Partitions=` ``
    assertion reds while every other arm stays green, which is exactly how that
    implementation passes review.

    Reachability: {a live same-host `local` sentinel} reaches the refusal and
    therefore the ladder; artifact: the gate's own `WorkflowError` is the only
    carrier of this text, so no other input class can drive it.

    Both-states anchor: the assertion turns on the ORDER of two substrings and
    on which rung number the ssh sentence sits at -- properties measurable in
    the pre-change text as well (where the discriminator is absent and the ssh
    remedy IS rung 2), never on wording this change introduces.

    Invocation isolated: not isolatable from this tree -- discrimination shown
    instead: the named change to what the code DOES on this input is "gate the
    ssh remedy behind a measured host class", which greens it; no path, flag or
    interpreter change touches it.

    Why required: a `batch_job` driver runs ON a compute node, so its sentinel's
    origin host is a compute node that refuses ssh and is reallocated once the
    allocation ends -- the ssh remedy is unreachable for the one mode that
    writes such a hostname. Measured 2026-10-14 on `benchmarking_gpu_a100_uva`.

    What kills it: dropping the discriminator rung, moving the ssh remedy back
    ahead of it, or keying the branch on the hostname pattern.

    Second correct implementation: one rung that prints both host classes as a
    table and marks which arm applies still places `scontrol show node` before
    the ssh sentence and passes unchanged.
    """
    ladder = _refusal_ladder(builder, tmp_path)

    discriminator_at = ladder.find("scontrol show node")
    assert discriminator_at != -1, (
        "the ladder must MEASURE the origin host's class before offering a remedy; "
        "`scontrol show node` is the only instrument that reads the `Partitions=` field"
    )
    assert "`Partitions=`" in ladder, (
        "the discriminator is the `Partitions=` field: UVA login and compute nodes share "
        "one hostname prefix, so a hostname pattern cannot separate them"
    )
    ssh_at = ladder.find("retry until `hostname` matches")
    assert ssh_at != -1, "the login-node ssh remedy must survive -- it is the only route to a host-local probe"
    assert discriminator_at < ssh_at, "the ssh remedy must be gated BEHIND the measurement, never offered before it"
    assert not re.search(r"^\s*2\. .*ssh to that origin host", ladder, re.MULTILINE), (
        "rung 2 must not be an unconditional cross-node ssh remedy"
    )


def test_next_steps_is_a_five_rung_ladder_ending_in_the_override_assertion(builder, tmp_path):
    """The ladder has five rungs and the override assertion is the last of them.

    Property: rungs are numbered 1..5 in order, and `--override-live-driver`
    appears after the discriminator rung rather than as an early shortcut.

    Class: REGRESSION -- red on the pre-change tree, which numbers three rungs;
    output pasted in the build return.

    Reachability: {a live same-host `local` sentinel} reaches the refusal, the
    only carrier of the ladder; artifact: the gate's `WorkflowError`.

    Both-states anchor: the assertion turns on the rung NUMBERING and on the
    relative order of two substrings -- both measurable in the pre-change text,
    which numbers 1..3 and places the override third.

    Invocation isolated: not isolatable from this tree -- discrimination shown
    instead: the named change to what the code DOES on this input is "insert the
    two measurement rungs ahead of the override", which greens it.

    Why required: the override is an ASSERTION the operator makes from evidence
    (`_acquire_submit_driver_claim`'s docstring), so the ladder must not let it
    read as the second thing to try when a measurement is available.

    What kills it: dropping either measurement rung, or moving the override to
    an earlier number.

    Second correct implementation: emitting the ladder from a list of
    `(rung, text)` pairs and numbering by position greens this identically.
    """
    ladder = _refusal_ladder(builder, tmp_path)

    starts = [m.start() for m in re.finditer(r"^\s*\d\. ", ladder, re.MULTILINE)]
    numbers = [re.match(r"\s*(\d)", ladder[s:]).group(1) for s in starts]
    assert numbers == ["1", "2", "3", "4", "5"], f"the ladder is five rungs; measured numbering was {numbers}"
    blocks = [ladder[s:e] for s, e in zip(starts, starts[1:] + [len(ladder)], strict=True)]
    assert "--override-live-driver" in blocks[-1], "the override assertion is the last resort, never an earlier rung"
    assert "An unresolved driver is NOT a dead driver" in ladder, (
        "the 2026-08-31 warning is why the ladder exists; every revision must keep it"
    )


def test_ladder_states_the_polarity_and_scopes_the_ssh_rung_to_login_nodes(builder, tmp_path):
    """The ladder's two load-bearing SENTENCES, not only its tokens and their order.

    Property: rung 3 states the discriminator's POLARITY -- a non-empty
    ``Partitions=`` is the compute-node condition -- and the one rung that
    offers the ssh remedy scopes it to login nodes.

    Class: NEW CAPABILITY -- the plausible wrong implementations are (a) the
    polarity inverted, ``An ABSENT``/``a NON-EMPTY`` swapped, which still names
    ``scontrol show node`` and ``Partitions=`` in the right order and at the
    right rung, and (b) the measurement present but not gating, with the ssh
    rung's scoping dropped. Both pass every token-and-order assertion in the
    sibling tests above; both were measured against the delivered suite on
    2026-10-29 in isolated mutant trees and returned 18 passed each.

    Reachability: {a live same-host ``local`` sentinel} reaches the refusal and
    therefore the ladder; artifact: the gate's ``WorkflowError`` is the only
    carrier of this text, so no other input class can drive it.

    Both-states anchor: NOT AVAILABLE, and said so rather than faked -- the
    sentences under test are introduced by this chunk, so no pre-change
    property of them exists. The nearest both-states properties (the single
    ``NEXT STEPS`` carrier and the 2026-08-31 warning) are asserted by the
    siblings above; this test is declared under condition 1's NEW CAPABILITY
    arm, whose bar is the mutant red pasted beside it.

    Invocation isolated: not isolatable from this tree -- discrimination shown
    instead: the named change to what the code DOES on this input is "state the
    measured polarity and scope the ssh remedy to login nodes", which greens
    both assertions; no path, flag or interpreter change touches them.

    Why required: an inverted polarity sends an operator to ssh a COMPUTE node,
    which refuses interactive ssh, and an unscoped ssh rung restores the
    unconditional cross-node remedy the ladder exists to withdraw -- the exact
    defect item 1e.4 removes, reintroduced below a measurement that no longer
    decides anything.

    What kills it: swapping NON-EMPTY and ABSENT, or dropping the login-node
    scoping from the rung that offers ssh.

    Second correct implementation: emitting the same two sentences from a
    ``(rung, text)`` table greens this identically -- the assertion binds the
    operator-facing CONTRACT TEXT, which this chunk ships, not how the message
    is assembled.
    """
    ladder = _refusal_ladder(builder, tmp_path)

    assert "A NON-EMPTY `Partitions=` means a COMPUTE node" in ladder, (
        "rung 3 must state the discriminator's POLARITY, not merely name the field: an "
        "inverted polarity keeps `scontrol show node` and `Partitions=` in the right order "
        "and still sends an operator to ssh a COMPUTE node"
    )
    starts = [m.start() for m in re.finditer(r"^\s*\d\. ", ladder, re.MULTILINE)]
    blocks = [ladder[s:e] for s, e in zip(starts, starts[1:] + [len(ladder)], strict=True)]
    ssh_blocks = [b for b in blocks if "ssh to that origin host" in b]
    assert len(ssh_blocks) == 1, f"exactly one rung offers the ssh remedy; measured {len(ssh_blocks)}"
    assert "LOGIN node only" in ssh_blocks[0], (
        "the ssh rung must be scoped to login nodes: offered unscoped it restores the "
        "cross-node remedy the measurement rung declares unreachable"
    )


def test_rung_three_states_what_the_measurement_decides_for_a_compute_node_origin(builder, tmp_path):
    """The measurement DECIDES, in both directions, and says what it closes.

    Property: rung 3 states the discriminator as a two-way CLASSIFICATION (a
    non-empty ``Partitions=`` is the compute condition AND an absent field is
    the login condition) and states the CONSEQUENCE that follows for the
    compute arm -- that the ssh route is closed to it, naming the rung it
    closes -- ahead of the rung that offers that route.

    Class: NEW CAPABILITY -- two plausible wrong implementations, each measured
    and pasted in round 2's return: (M2) the rung reports the class and decides
    nothing, its consequence clause absent; (M4) the polarity is stated in one
    direction only, so an operator whose field is absent is never told what
    that means. Both pass every token, order, numbering and polarity-sentence
    assertion in the sibling tests above.

    Reachability: {a live same-host ``local`` sentinel} reaches the refusal and
    therefore the ladder; artifact: the gate's ``WorkflowError`` is the only
    carrier of this text, so no other input class can drive it.

    Both-states anchor: NOT AVAILABLE, and said so rather than faked -- the
    sentences under test are introduced by this chunk, so no pre-change property
    of them exists. Declared under condition 1's NEW CAPABILITY arm, whose bar
    is the mutant red pasted beside it.

    Invocation isolated: not isolatable from this tree -- discrimination shown
    instead: the named change to what the code DOES on this input is "say what
    the measurement decides, in both directions, and name the rung it closes",
    which greens all four assertions; no path, flag or interpreter change
    touches them.

    Why required: a classification with no consequence leaves the compute-node
    operator to infer that the very next rung is closed to them, and a
    one-directional classification leaves the login-node operator unable to
    identify their own arm -- the two ways a measurement can be present and
    still decide nothing.

    What kills it: dropping the consequence clause, dropping either half of the
    classification, or moving the consequence below the ssh rung.

    Second correct implementation: emitting the same four clauses from a
    ``(rung, text)`` table greens this identically -- the assertion binds the
    operator-facing CONTRACT TEXT this chunk ships, not how it is assembled.

    Position, against the sibling test that also touches polarity: that test
    reads the ladder as TEXT (one polarity sentence present, one rung carrying
    ``LOGIN node only``); this one reads it as a DECISION (both halves of the
    classification, plus what the compute branch's measurement closes). M4
    below is the measured case where the difference bites.
    """
    ladder = _refusal_ladder(builder, tmp_path)

    compute_half = ladder.find("NON-EMPTY `Partitions=` means a COMPUTE node")
    login_half = ladder.find("ABSENT field means a LOGIN node")
    assert compute_half != -1, "rung 3 must name the compute condition: a non-empty `Partitions=` means a COMPUTE node"
    assert login_half != -1, (
        "rung 3 must name the OTHER arm too: an absent `Partitions=` means a LOGIN node, or an "
        "operator cannot tell which arm they are in and the login remedy is unreachable by "
        "instruction"
    )
    assert compute_half < login_half, (
        "the two halves of the classification must read compute-first, in the order the rungs are walked"
    )

    consequence_at = ladder.find("cross-node ssh is closed")
    assert consequence_at != -1, (
        "the compute arm must state the CONSEQUENCE, not merely the class: without it the "
        "operator is left to infer that the next rung is closed to them"
    )
    assert "rung 4 is UNREACHABLE BY CONSTRUCTION" in ladder, (
        "the consequence must NAME the rung it closes, so the decision attaches to the remedy "
        "it withdraws rather than to the ladder in general"
    )
    ssh_at = ladder.find("retry until `hostname` matches")
    assert consequence_at < ssh_at, (
        "the consequence must be stated BEFORE the rung that offers ssh, otherwise the "
        "measurement follows the remedy it was supposed to gate"
    )


def test_consequence_clause_attaches_to_the_compute_arm(builder, tmp_path):
    """The consequence belongs to the arm it is TRUE of, not merely present somewhere.

    Property: the clause that states ``cross-node ssh is closed`` names the
    COMPUTE arm and does not name the LOGIN arm -- the decision is ATTRIBUTED
    to the class it governs, rather than stated in the ladder and attached to
    whichever arm a re-author put it beside.

    Class: NEW CAPABILITY -- the plausible wrong implementation is the ladder
    that keeps every pinned sentence and every ordering while attaching the
    consequence to the LOGIN arm ("an ABSENT field means a LOGIN node, where
    cross-node ssh is closed ..."). Measured 2026-10-29 as mutant M7: the
    suite's other 20 tests stay GREEN under it (`20 passed`) and this test is
    the only one that reds.

    Reachability: {a live same-host ``local`` sentinel} reaches the refusal and
    therefore the ladder; artifact: the gate's ``WorkflowError`` is the only
    carrier of this text, so no other input class can drive it.

    Both-states anchor: NOT AVAILABLE, and said so rather than faked -- the
    sentences under test are introduced by this chunk. Declared under condition
    1's NEW CAPABILITY arm, whose bar is the mutant red beside it.

    Invocation isolated: not isolatable from this tree -- discrimination shown
    instead: the named change to what the code DOES on this input is "attach
    the consequence to the compute arm", which greens this assertion and leaves
    every invocation untouched.

    Why required: a clause attached to the wrong arm is WORSE than an absent
    one. An absent consequence leaves the compute-node operator to guess; a
    misattached one tells the login operator -- the only operator whose ssh
    route works -- that their route is closed, while still leaving the compute
    operator to attempt the route that cannot work. Every token, order and
    polarity assertion in the siblings above passes on that ladder.

    What kills it: moving the consequence clause onto the LOGIN arm.

    Second correct implementation: splitting the sentence at the semicolon so
    the consequence closes its own sentence on the compute arm -- measured
    green (`20 passed`) on that reworded tree -- leaves the clause on the right
    arm and passes unchanged. The assertion is CLAUSE-scoped, not
    adjacency-scoped, so it does not bind the punctuation a re-author chooses.
    """
    ladder = _refusal_ladder(builder, tmp_path)

    carrying = [seg for seg in re.split(r"(?<=[.;])\s+", ladder) if "cross-node ssh is closed" in seg]
    assert len(carrying) == 1, f"exactly one clause states the consequence; measured {len(carrying)}"
    assert "LOGIN" not in carrying[0], (
        "the consequence must attach to the COMPUTE arm: on the LOGIN arm it tells the operator "
        "whose ssh works that their route is closed, and still leaves the compute operator to "
        "attempt the one that cannot work"
    )


# --------------------------------------------------------------------------- #
# gate: the host-agnostic job-id arm
# --------------------------------------------------------------------------- #


def test_foreign_host_sentinel_with_a_dead_jobid_proceeds(builder, tmp_path, monkeypatch):
    """A resume can clear a detached driver whose recorded job id is not live.

    Property: for a sentinel whose origin is a foreign host, a recorded job id
    that squeue does not report live is DEAD EVIDENCE — reclaim and proceed —
    rather than a reason to hold the sentinel to the age cap.

    Class: REGRESSION — red on the pre-change tree, where the sentinel fell to
    the foreign-origin branch, answered UNKNOWN, and the gate refused.

    Reachability: {batch_job or tmux sentinel, foreign origin, non-live recorded
    job id} reaches this arm; artifact: the assertion below reds at HEAD, which
    is where the arm is absent.

    Both-states anchor: the assertion turns on whether the gate RETURNS None, a
    property of both the pre-change and the proposed code; it does not assert
    any wording the change introduces.

    Invocation isolated: not isolatable from this tree — discrimination shown
    instead: the named change that greens it is "resolve the recorded job id
    instead of short-circuiting on a foreign origin", a change to what the code
    DOES on this input, not to how it is invoked.

    Why required: a compute-node origin is the NORMAL case for a detached
    driver, so this is the ordinary resume path. Measured 2026-09-13 on
    norfolk_stochastic_multimodel, one such sentinel blocked a resume for the
    full 4350-minute cap.

    What kills it: removing the job-id arm, or narrowing its guard so a foreign
    origin short-circuits before the job id is read.

    Second correct implementation: a helper `_jobid_verdict(sentinel)` consulted
    ahead of the foreign-origin branch, returning False here, greens it
    identically.
    """
    sentinel = _write_foreign_sentinel(tmp_path, "12345-other-node-aaaaaaaa", monkeypatch, jobid="12345")
    monkeypatch.setattr(wf, "_slurm_job_is_live", lambda _jid: False)
    # The D1 corroboration (chunk 2b.3) is PINNED, not left to run: left real it
    # answers NOT-KNOWN on a box without sacct, which is the other branch of this
    # chunk. This test's subject is the reclaim that a CONFIRMED death admits.
    monkeypatch.setattr(wf, "_corroborate_not_live", lambda _jid: "dead")

    assert builder._orchestrator_liveness_gate(tmp_path) is None
    assert not sentinel.exists(), "a non-live recorded job id is DEAD evidence: reclaim it, never hold it"


def test_foreign_host_dead_jobid_with_an_unconfirmable_second_query_is_held(builder, tmp_path, monkeypatch):
    """The SECOND host-agnostic arm corroborates too: an unconfirmable second query holds.

    Property: for a sentinel whose origin is a foreign host, a recorded job id that
    squeue does not report live is NOT on its own dead evidence. The second query
    is asked, and when it cannot answer the arm resolves UNKNOWN — held to the
    gate's existing mtime-age cap — rather than reclaimed.

    Class: NEW CAPABILITY — red on the pre-change tree, which reclaimed on the
    single squeue answer; artifact: the assertion below reds at HEAD.

    Reachability: {batch_job sentinel, foreign origin, recorded job id, squeue
    silent, sacct silent} reaches this arm.

    Both-states anchor: the assertion turns on the gate REFUSING and on the
    sentinel surviving — properties of both the pre-change and the proposed code.
    'UNKNOWN/held' is pre-existing wording, not wording this change introduces.

    Why required: this arm's disposition differs from the delete site's. Here an
    unconfirmable answer must HOLD (a false reclaim discards a live driver's
    sentinel and admits the concurrent consolidate-zarr write this gate prevents),
    where the delete preflight INCLUDES the job in its refusal set.

    What kills it: dropping the corroboration from this arm, treating 'unknown' as
    dead here, or uniformising this site to the delete site's inclusion rule.
    """
    sentinel = _write_foreign_sentinel(tmp_path, "12345-other-node-aabbbbbb", monkeypatch, jobid="12345")
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(wf, "_slurm_job_is_live", lambda _jid: False)
    monkeypatch.setattr(wf, "_corroborate_not_live", lambda _jid: "unknown")

    err = builder._orchestrator_liveness_gate(tmp_path)
    assert isinstance(err, WorkflowError), "an unconfirmable second query is UNKNOWN, never a reclaim"
    # Operator-visible signal: the LISTING entry names the verdict and the cap.
    # `"UNKNOWN/held"` alone is NOT discriminating — the refusal's static explanatory
    # paragraph contains that phrase unconditionally — so this keys on the entry tail.
    assert "UNKNOWN/held" in err.stderr
    assert "min cap, sentinel=" in err.stderr
    assert sentinel.exists(), "UNKNOWN is held, never reclaimed"


def test_foreign_host_dead_jobid_corroborated_alive_keeps_the_sentinel(builder, tmp_path, monkeypatch):
    """A non-terminal second-query row on the job-id arm is ALIVE, and the operator is told so.

    Property: the corroboration does not merely convert a reclaim into a hold; when
    sacct carries a non-terminal row the verdict is ALIVE, which is a DIRECT
    measurement rather than an inability to answer, and the message distinguishes
    the two.

    Class: NEW CAPABILITY — red on the pre-change tree, which reclaimed.
    """
    sentinel = _write_foreign_sentinel(tmp_path, "12345-other-node-ccdddddd", monkeypatch, jobid="12345")
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(wf, "_slurm_job_is_live", lambda _jid: False)
    monkeypatch.setattr(wf, "_corroborate_not_live", lambda _jid: "alive")

    err = builder._orchestrator_liveness_gate(tmp_path)
    assert isinstance(err, WorkflowError)
    assert "liveness=ALIVE" in err.stderr
    # The held LISTING entry is `liveness=UNKNOWN/held {n}min < {n}min cap, sentinel={p}`.
    # The refusal's static explanatory paragraph mentions "UNKNOWN/held" unconditionally,
    # so the negative half keys on the listing tail rather than on that phrase.
    assert "min cap, sentinel=" not in err.stderr
    assert sentinel.exists()


def test_requeue_carve_out_is_not_corroborated(builder, tmp_path, monkeypatch):
    """The requeue carve-out answers DEAD WITHOUT asking the corroborator.

    Property: the carve-out's DEAD verdict is an INFERENCE from a shared allocation
    id — the record was written by a PRIOR invocation inside the operator's OWN
    allocation — and not a probe's "not running" answer, so D1's corroboration does
    not apply to it. Corroborating it would re-REFUSE a genuine SLURM requeue,
    which is the recovery path these campaigns rely on.

    Class: REGRESSION GUARD — the corroborator is pinned to answer "dead" on
    purpose, so a wrongly-corroborated carve-out still RECLAIMS and the gate still
    proceeds identically. Only the call ledger separates the two, which is why this
    test's instrument is the recorded call list and not the return value.

    Reachability: {detached sentinel, foreign origin, recorded job id == this
    process's SLURM_JOB_ID} reaches the carve-out.

    What kills it: gating the corroboration on the arm rather than on whether a
    PROBE answered — the natural simplification once the carve-out and the probe
    both set ``alive = False`` in the same branch.
    """
    sentinel = _write_foreign_sentinel(tmp_path, "777-other-node-ffffffff", monkeypatch, jobid="777", mode="tmux")
    monkeypatch.setenv("SLURM_JOB_ID", "777")
    corroborated: list[str] = []

    def _record_corroboration(jid):
        corroborated.append(jid)
        return "dead"

    monkeypatch.setattr(wf, "_slurm_job_is_live", lambda _jid: True)  # squeue still reports it ALIVE
    monkeypatch.setattr(wf, "_corroborate_not_live", _record_corroboration)

    assert builder._orchestrator_liveness_gate(tmp_path) is None
    assert corroborated == [], "the requeue carve-out is an inference, not a probe result: never corroborate it"
    assert not sentinel.exists()


def test_requeued_jobid_inside_my_own_allocation_is_dead_without_asking_squeue(builder, tmp_path, monkeypatch):
    """A SLURM requeue keeps the job id, so my own id must not read as ALIVE.

    Property: when the recorded job id equals THIS process's `SLURM_JOB_ID` and
    the mode is detached, the previous invocation ran serially in my own
    allocation and has already returned — so the sentinel is DEAD on evidence
    and squeue is never consulted.

    Class: REGRESSION — red on the pre-change tree, where the foreign-origin
    branch answered UNKNOWN and refused the requeue, i.e. refused the very
    recovery path these campaigns rely on.

    Reachability: {detached sentinel, foreign origin, job id == my own} reaches
    the carve-out; artifact: the assertion below reds at HEAD.

    Both-states anchor: the assertion turns on the gate's RETURN value and on
    whether the liveness probe was CALLED — both properties of the pre-change
    and proposed code alike.

    Invocation isolated: not isolatable from this tree — discrimination shown
    instead: the named change that greens it is "treat my own allocation's job
    id as already-returned rather than probing it".

    Why required: `JobRequeue=1` on UVA re-runs the batch script from the top
    and KEEPS the job id, so without this carve-out the recorded id reads ALIVE
    and the requeue is refused.

    What kills it: widening the carve-out to every mode, or dropping the
    `jobid == my_jobid` conjunct.

    Second correct implementation: comparing against a pre-computed
    `_my_jobid = os.environ.get("SLURM_JOB_ID") or ""` hoisted above the
    dispatch, greens it identically.
    """
    sentinel = _write_foreign_sentinel(tmp_path, "777-other-node-bbbbbbbb", monkeypatch, jobid="777", mode="tmux")
    monkeypatch.setenv("SLURM_JOB_ID", "777")
    probed: list[str] = []

    def _record(jid):
        probed.append(jid)
        return True  # squeue still reports the requeued job ALIVE

    monkeypatch.setattr(wf, "_slurm_job_is_live", _record)

    assert builder._orchestrator_liveness_gate(tmp_path) is None
    assert probed == [], "the requeue carve-out must answer without consulting squeue"
    assert not sentinel.exists()


# --------------------------------------------------------------------------- #
# absent squeue: NOT-KNOWN is a VALUE at the probe, never an exception
# --------------------------------------------------------------------------- #
#
# `_slurm_job_is_live` answers `bool | None`. `None` means the question was never
# asked — no `squeue` binary on PATH, or any other OSError from the launch — and
# the gate holds rather than reclaims on it. Before the widening the probe raised
# on that class, which left the gate's two job-id arms with OPPOSITE postures:
# the foreign-origin arm wrapped the call in `try/except OSError`, while the
# `1_job_many_srun_tasks` arm — dispatched FIRST, and with no `try` — let the
# FileNotFoundError escape a classifier whose contract is tri-state. The four
# tests below pin the probe's contract, the must-fire control that separates the
# widened exception class from the one that must stay narrow, and one gate test
# per arm.


def _disable_the_leaf_probes_process_launch(monkeypatch, exc_factory):
    """Make the REAL probe's own launch fail, and nothing else's.

    `_slurm_job_is_live` reads `subprocess` from `hhemt.slurm_liveness`'s module
    globals, so replacing THAT name confines the outage to the probe. Patching
    the shared `subprocess` module instead would also disable the gate's `ps -p`
    and `sacct` launches, and a gate test would then pass for reasons that have
    nothing to do with an absent `squeue`.

    `TimeoutExpired` is carried on the stub because the probe's
    `except subprocess.TimeoutExpired:` clause resolves that attribute from the
    same module global at the moment it handles the exception — a stub without it
    turns the handler itself into an AttributeError, and the test would then be
    measuring the stub.
    """

    def _run(*_args, **_kwargs):
        raise exc_factory()

    monkeypatch.setattr(
        sl,
        "subprocess",
        types.SimpleNamespace(run=_run, TimeoutExpired=subprocess.TimeoutExpired),
    )


def test_the_probe_answers_not_known_when_squeue_cannot_run(monkeypatch):
    """An unlaunchable `squeue` reads None, never False.

    Property: `_slurm_job_is_live` distinguishes "asked, and it is gone" (False)
    from "could not ask" (None). The distinction is the contract: every consumer
    routes not-known to a HOLD and not-live to a corroboration, so a False here
    is a death claim the probe holds no evidence for.

    Class: NEW CAPABILITY — plausible wrong implementation is the same probe
    catching OSError and returning False, under which this assertion reds.

    Reachability: {a host with no `squeue` binary} reaches the OSError; artifact:
    the pre-fix `python3 -c` run pasted in the build return, which ends in
    `FileNotFoundError: [Errno 2] No such file or directory: 'squeue'` on a box
    where `command -v squeue` prints nothing.

    Both-states anchor: the assertion turns on `None` being this module's
    NOT-KNOWN value, which is pre-existing — `_sacct_states_batched` already
    returns `{}` for the same cause and every caller already treats an absent row
    as unknown. No wording this change introduces is asserted.

    Invocation isolated: not isolatable from this tree — discrimination shown
    instead: the named change that greens it is "catch OSError from the launch
    and return None", which changes what the code DOES on this input.

    Why required: item 1e.17's done-when is literally "it returns None", and a
    raising probe makes the `1_job_many_srun_tasks` gate arm raise out of a
    tri-state classifier (see the gate test below).

    What kills it: an `except OSError` arm returning False, or no arm at all —
    the FileNotFoundError then escapes and this test errors rather than fails.

    Second correct implementation: a module-level `_squeue_state(job_id)` helper
    that returns a NOT-KNOWN marker on OSError, with the probe returning None
    when it sees that marker, greens this identically.
    """
    _disable_the_leaf_probes_process_launch(
        monkeypatch, lambda: FileNotFoundError(2, "No such file or directory", "squeue")
    )

    assert sl._slurm_job_is_live("1") is None, "an unlaunchable squeue is NOT-KNOWN, never not-live"


def test_a_squeue_timeout_still_reads_not_live(monkeypatch):
    """MUST-FIRE control for the test above: it varies the EXCEPTION CLASS.

    Property: only the unlaunchable-launch class widened. A `TimeoutExpired`
    still means the binary RAN and answered with silence, so it reads False —
    the two-valued behaviour the gate's corroboration then checks. This matters
    because a timeout on a live job is the input class the corroboration exists
    for, and moving it to "could not ask" would silently reverse that.

    This is the control the absence assertion needs rather than a second copy of
    it. A broken stub and a genuinely absent binary both surface as the probe's
    FileNotFoundError, so that assertion alone cannot tell them apart; this one
    raises the OTHER exception and must still answer False.

    Class: NEW CAPABILITY — plausible wrong implementation is the same probe with
    `except (OSError, subprocess.SubprocessError)`, sweeping both classes into
    None, under which this assertion reds.

    Reachability: {a `squeue` that starts and does not return inside `timeout_s`}
    reaches the TimeoutExpired branch.

    Both-states anchor: the assertion turns on `False` for a timeout, which the
    pre-change probe already returns; the new arm must not move it.

    Invocation isolated: not isolatable from this tree — discrimination shown
    instead: sweeping SubprocessError into None reds this and changes nothing
    else here.

    Why required: the timeout stays on the not-live path — a slow cluster is
    exactly what the callers' corroboration step is built for — and widening past
    OSError would move it there in code with no statement anywhere changing to
    say so.

    What kills it: catching `subprocess.SubprocessError` — TimeoutExpired's
    parent — instead of `OSError`.

    Second correct implementation: a `_run_squeue()` helper returning a status in
    {NOT_RUN, RAN} plus stdout, where NOT_RUN maps to None and TimeoutExpired
    maps to RAN-with-silence, greens this identically.
    """
    _disable_the_leaf_probes_process_launch(
        monkeypatch, lambda: subprocess.TimeoutExpired(cmd=["squeue"], timeout=10.0)
    )

    assert sl._slurm_job_is_live("1") is False, "a slow-but-present squeue answered; a timeout is not-known-free"


def test_absent_squeue_on_the_many_srun_tasks_arm_refuses_instead_of_raising(builder, tmp_path, monkeypatch):
    """The arm with NO try/except answers, instead of raising out of the gate.

    Property: `1_job_many_srun_tasks` is dispatched on FIRST and carries no
    try/except of its own. Before the probe widened, an absent `squeue` raised
    FileNotFoundError straight out of `_orchestrator_liveness_gate`, so a
    classifier whose contract is tri-state returned no verdict at all while the
    sibling job-id arm returned UNKNOWN for the same cause. Both arms now answer,
    and the answer is the same, because the probe returns None.

    Class: REGRESSION — red on the pre-change tree, where this same input ends in
    `FileNotFoundError: [Errno 2] No such file or directory: 'squeue'` escaping
    the gate; output pasted in the build return.

    Reachability: {mode `1_job_many_srun_tasks` sentinel, recorded job id, a host
    whose squeue is absent} reaches the unguarded arm's probe call.

    Both-states anchor: the assertion turns on the gate RAISING WorkflowError and
    on the sentinel surviving — properties of the pre-change code — and on
    "UNKNOWN/held", which is pre-existing wording.

    Invocation isolated: not isolatable from this tree — discrimination shown
    instead: returning None from the probe (rather than raising) is what greens
    it, a change to what the code DOES on this input.

    Why required: this is the asymmetry the item exists to close. The arm's own
    comment records that the corroboration was parked outside the dispatch chain
    BECAUSE the arms differ in exception posture; once the probe answers, the
    corroboration's placement no longer has to carry that difference.

    What kills it: a probe that raises on OSError, or one that returns False —
    the latter would claim death with no corroboration and reclaim the sentinel.

    Second correct implementation: an `except OSError: alive = None` added to this
    arm instead of widening the probe greens the gate half of this test but reds
    `test_the_probe_answers_not_known_when_squeue_cannot_run`, which is why the
    remedy belongs at the probe.

    Not established: the held listing's cause label on this arm still reads
    "unenriched". The cause is assigned per BRANCH inside
    `_orchestrator_liveness_gate`, and that file is outside this chunk's write
    set; the hold itself is correct and bounded, and the label is filed as an
    erratum in the build return.
    """
    sentinel = osent.write_orchestrator_sentinel(
        tmp_path,
        driver_id="777-many-srun-eeeeeeee",
        workflow_submission_mode="1_job_many_srun_tasks",
        slurm_jobid="777",
    )
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    _disable_the_leaf_probes_process_launch(
        monkeypatch, lambda: FileNotFoundError(2, "No such file or directory", "squeue")
    )

    err = builder._orchestrator_liveness_gate(tmp_path)
    assert isinstance(err, WorkflowError), "the no-try arm must return a verdict, not raise out of the gate"
    assert "UNKNOWN/held" in err.stderr
    assert sentinel.exists(), "UNKNOWN is held, never reclaimed"


def test_absent_squeue_on_the_jobid_arm_is_unknown_and_refuses(builder, tmp_path, monkeypatch):
    """No squeue on this host reads UNKNOWN, never a raise and never a reclaim.

    Property: the foreign-origin job-id arm answers UNKNOWN when the liveness
    probe could not ask, which keeps the gate's tri-state contract intact; the
    mtime-age fail-safe still bounds the hold. This is the arm that already had a
    `try/except OSError`; the test now drives the REAL probe through it, so what
    it pins is the probe's contract rather than the arm's guard.

    Class: NEW CAPABILITY — plausible wrong implementation is the widened probe
    absorbing OSError as False, under which the gate reads an unanswerable squeue
    as a not-live answer, spends a corroboration query, and — with the
    corroborator returning "dead" below, as it does for any job whose accounting
    record has aged out — RECLAIMS the sentinel and admits a resume against a
    driver that may still be running. The corroborator is stubbed to "dead"
    precisely so the wrong implementation cannot hide behind it.

    Reachability: {detached sentinel, foreign origin, a host whose squeue is
    absent} reaches the probe's OSError.

    Both-states anchor: the assertion turns on the gate RAISING WorkflowError and
    on the sentinel surviving — properties of the pre-change code — and on
    "UNKNOWN/held", pre-existing wording.

    Invocation isolated: not isolatable from this tree — discrimination shown
    instead: returning None rather than False from the probe is what greens it,
    a change to what the code DOES on this input.

    Why required: a `False` here is the false-proceed the whole corroboration
    chain exists to remove, arriving through the exception handler's replacement
    rather than through a bare squeue answer.

    What kills it: `except OSError: return False`, or catching the OSError at the
    arm and answering False there.

    Second correct implementation: returning a three-state enum member
    (UNKNOWN/NOT_LIVE/LIVE) from the probe, with this arm mapping UNKNOWN to
    None, greens this identically.
    """
    sentinel = _write_foreign_sentinel(tmp_path, "555-other-node-cccccccc", monkeypatch, jobid="555")
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    # A corroborator that confirms death is what the wrong implementation would
    # act on; it must not be asked at all when the probe could not answer.
    monkeypatch.setattr(wf, "_corroborate_not_live", lambda _jid: "dead")
    _disable_the_leaf_probes_process_launch(
        monkeypatch, lambda: FileNotFoundError(2, "No such file or directory", "squeue")
    )

    err = builder._orchestrator_liveness_gate(tmp_path)
    assert isinstance(err, WorkflowError), "an unanswerable probe is UNKNOWN, never a raise or a proceed"
    assert "UNKNOWN/held" in err.stderr
    assert sentinel.exists(), "UNKNOWN is held, never reclaimed"


def test_a_same_host_live_pid_still_wins_over_a_dead_recorded_jobid(builder, tmp_path, monkeypatch):
    """A direct pid probe must win wherever it can answer.

    Property: for a SAME-HOST sentinel the job-id arm does not fire at all, so
    the `mode == "local"` `ps -p` arm classifies it — even when a recorded job
    id would read dead. A same-host pid probe is a DIRECT measurement of that
    driver; an enclosing-allocation job id is only a proxy.

    Class: NEW CAPABILITY — plausible wrong implementation is the job-id arm
    with its `origin_host is None or not same_host` guard REMOVED, under which
    this test reds (the gate proceeds instead of refusing); artifact: the
    mutation pasted in the build return.

    Reachability: {local sentinel, same host, live pid, recorded job id} reaches
    the collision; artifact: the assertion below reds under the named wrong
    implementation.

    Both-states anchor: the assertion turns on the gate REFUSING and on the
    "liveness=ALIVE" verdict the `ps -p` arm already produces — both properties
    of the pre-change code.

    Invocation isolated: not isolatable from this tree — discrimination shown
    instead: the named change that reds it is removing the guard, which changes
    what the code DOES on this input.

    Why required: the `local` modes are the documented reclaim path for the
    detached reprocess/static-plots self-sentinels, and preempting them would
    silently discard a live driver's sentinel.

    What kills it: dropping `origin_host is None or not same_host` from the arm's
    guard.

    Second correct implementation: gating the arm on `not same_host or
    origin_host is None` computed once into a local and consulted there greens
    it identically.
    """
    sentinel = osent.write_orchestrator_sentinel(
        tmp_path,
        driver_id=f"{os.getpid()}-{socket.gethostname()}-dddddddd",
        workflow_submission_mode="local",
        pid=os.getpid(),
        slurm_jobid="999",
    )
    monkeypatch.setattr(wf, "_slurm_job_is_live", lambda _jid: False)

    err = builder._orchestrator_liveness_gate(tmp_path)
    assert isinstance(err, WorkflowError), "the same-host pid probe must answer before any job-id proxy"
    assert "liveness=ALIVE" in err.stderr
    assert sentinel.exists()


def test_requeue_carve_out_does_not_apply_to_a_local_mode_sentinel(builder, tmp_path, monkeypatch):
    """A `local` sentinel is never excused by the requeue carve-out.

    Property: the "this job id is my own allocation, so the prior invocation has
    already returned" premise is restricted to the DETACHED modes. For a
    `local` sentinel the premise is FALSE — it may be a live detached Popen
    reprocess started earlier in this same allocation — so the recorded job id
    must still be resolved by squeue rather than short-circuited to DEAD.

    Class: REGRESSION — red on the pre-change tree, where a foreign-origin
    sentinel answered UNKNOWN/held instead of resolving its job id at all, and
    red under the named wrong implementation below.

    Reachability: {local sentinel, foreign origin, recorded job id == this
    process's SLURM_JOB_ID, job still live in squeue} reaches the collision.
    Artifact: `write_orchestrator_sentinel`'s explicit-argument branch stores a
    caller-supplied `slurm_jobid` regardless of mode, so such a record is
    constructible through the public writer; and the gate's own contract is
    stated over SENTINEL CONTENT (its docstring's UNKNOWN bucket (b) covers a
    legacy sentinel with no recorded hostname), not over what the current
    writer happens to emit.

    Both-states anchor: the assertion turns on the gate REFUSING, on the
    `liveness=ALIVE` verdict the squeue arm produces, and on the liveness probe
    being CALLED — properties of the pre-change and proposed code alike, never
    on wording this change introduces.

    Invocation isolated: not isolatable from this tree — discrimination shown
    instead: two named changes to what the code DOES on this input red it —
    widening the carve-out's mode tuple to every mode, or dropping its mode
    conjunct. Both were run; outputs are in the reviewer's return.

    Why required: a carve-out that keys only on "the id equals mine" reclaims a
    sentinel whose job is genuinely alive, which is the false-proceed this gate
    exists to prevent. The mode conjunct is the whole guard; keying on the id
    alone is the natural simplification a later reader reaches.

    What kills it: `if my_jobid and jobid == my_jobid:` in place of
    `if mode in ("batch_job", "tmux") and my_jobid and jobid == my_jobid:`.

    Second correct implementation: hoisting the detachment test into a local
    (`detached = mode in ("batch_job", "tmux")`) and gating the carve-out on
    it greens this identically.
    """
    sentinel = _write_foreign_sentinel(tmp_path, "888-other-node-eeeeeeee", monkeypatch, jobid="888", mode="local")
    monkeypatch.setenv("SLURM_JOB_ID", "888")
    probed: list[str] = []

    def _record(jid):
        probed.append(jid)
        return True  # the job is genuinely alive

    monkeypatch.setattr(wf, "_slurm_job_is_live", _record)

    err = builder._orchestrator_liveness_gate(tmp_path)
    assert isinstance(err, WorkflowError), "a live job id must block, whatever the sentinel's mode"
    assert "liveness=ALIVE" in err.stderr
    assert probed == ["888"], "a `local` sentinel is not excused from the squeue probe by its job id"
    assert sentinel.exists()


# --------------------------------------------------------------------------- #
# claim: the override is an ASSERTION, not a force flag
# --------------------------------------------------------------------------- #


def test_named_override_reclaims_that_sentinel_and_proceeds(builder, tmp_path):
    victim = f"{os.getpid()}-{socket.gethostname()}-deadbeef"
    _write_live_local_sentinel(tmp_path, victim)

    did = builder._acquire_submit_driver_claim(
        tmp_path,
        workflow_submission_mode="local",
        dry_run=False,
        override_live_driver=victim,
    )
    assert did and did != victim
    ids = {p["driver_id"] for p in osent.read_orchestrator_sentinels(tmp_path)}
    # The asserted-dead sentinel is GONE from disk: the assertion is recorded as
    # an act, not merely skipped for this invocation.
    assert victim not in ids
    assert ids == {did}


def test_unmatched_override_raises_and_unlinks_nothing(builder, tmp_path):
    """DISCRIMINATING. A blanket-skip implementation passes every happy path and
    fails here: a typo'd id that silently no-opped would read to the operator as
    an override that was honoured."""
    victim = f"{os.getpid()}-{socket.gethostname()}-deadbeef"
    _write_live_local_sentinel(tmp_path, victim)

    with pytest.raises(WorkflowError) as excinfo:
        builder._acquire_submit_driver_claim(
            tmp_path,
            workflow_submission_mode="local",
            dry_run=False,
            override_live_driver="no-such-driver-id",
        )
    assert "names no sentinel" in str(excinfo.value)
    ids = {p["driver_id"] for p in osent.read_orchestrator_sentinels(tmp_path)}
    assert ids == {victim}, "an unmatched assertion must not reclaim anything"


def test_override_is_scoped_to_the_named_driver(builder, tmp_path):
    """DISCRIMINATING. A boolean force flag would proceed here; a named
    assertion must not outrun its own evidence."""
    host = socket.gethostname()
    asserted = f"{os.getpid()}-{host}-aaaaaaaa"
    other = f"{os.getpid()}-{host}-bbbbbbbb"
    _write_live_local_sentinel(tmp_path, asserted)
    _write_live_local_sentinel(tmp_path, other)

    with pytest.raises(WorkflowError) as excinfo:
        builder._acquire_submit_driver_claim(
            tmp_path,
            workflow_submission_mode="local",
            dry_run=False,
            override_live_driver=asserted,
        )
    assert other in str(excinfo.value)
    ids = {p["driver_id"] for p in osent.read_orchestrator_sentinels(tmp_path)}
    assert ids == {other}, "only the named sentinel is reclaimed"


# --------------------------------------------------------------------------- #
# generated-Snakefile write safety
# --------------------------------------------------------------------------- #


def test_write_archives_one_deep_and_keeps_only_the_previous_generation(tmp_path):
    sf = tmp_path / "Snakefile"
    _write_snakefile_atomic(sf, "gen1")
    assert sf.read_text() == "gen1"
    assert not (tmp_path / "Snakefile.prev").exists()

    _write_snakefile_atomic(sf, "gen2")
    assert sf.read_text() == "gen2"
    assert (tmp_path / "Snakefile.prev").read_text() == "gen1"

    _write_snakefile_atomic(sf, "gen3")
    assert sf.read_text() == "gen3"
    # One deep: .prev holds the SECOND-oldest, never the oldest.
    assert (tmp_path / "Snakefile.prev").read_text() == "gen2"


def test_write_leaves_no_temp_files_behind(tmp_path):
    sf = tmp_path / "Snakefile"
    _write_snakefile_atomic(sf, "x")
    _write_snakefile_atomic(sf, "y")
    assert not list(tmp_path.glob("*.tmp"))


def test_dry_run_never_displaces_an_existing_snakefile(tmp_path):
    """The measured harm: the production Snakefile's content is NOT invariant
    (it carries `alive_by_token`), so a dry run against a live arm generates the
    wait-rule form and would install it over the good one."""
    good = tmp_path / "Snakefile"
    good.write_text("the live driver's DAG")

    target = _resolve_snakefile_path(tmp_path, dry_run=True)
    assert target.name == "Snakefile.dryrun"
    _write_snakefile_atomic(target, "11394 wait rules")
    assert good.read_text() == "the live driver's DAG"


def test_dry_run_on_a_fresh_tree_still_creates_the_production_snakefile(tmp_path):
    """A dry run never DISPLACES; it may CREATE. Where there is nothing to
    displace the legacy behaviour is strictly better -- df_status and
    df_snakemake_allocations parse `analysis_dir/"Snakefile"` by literal name,
    and tests/test_workflow_1job_dry_run.py asserts it exists after a dry run."""
    assert _resolve_snakefile_path(tmp_path, dry_run=True).name == "Snakefile"


def test_real_submit_always_targets_the_production_snakefile(tmp_path):
    (tmp_path / "Snakefile").write_text("old")
    assert _resolve_snakefile_path(tmp_path, dry_run=False).name == "Snakefile"
