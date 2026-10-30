"""Write-time ``slurm_jobid`` stamping on the orchestrator sentinel.

A driver killed mid-run never reaches ``enrich_orchestrator_sentinel``, which runs
only after ``submit_workflow()`` returns and sits outside the ``try/finally`` that
removes a blocking-local sentinel. Anything a later resume needs in order to ask
"is that driver still alive?" from a DIFFERENT node must therefore be on disk at
write time, and ``slurm_jobid`` is the only identifier with that property:
``tmux has-session`` and ``ps -p`` are host-local, while SLURM job ids are
cluster-wide.

Stamping it unconditionally would be worse than not stamping it. Four of the five
write sites are reprocess / static-plots SELF-sentinels that hardcode
``workflow_submission_mode="local"`` and ``pid=os.getpid()``, and their reclaim
path is the gate's ``ps -p {pid}`` arm, which measures that driver DIRECTLY. An
enclosing-allocation job id is only a proxy for it, so a stamped id would let a
jobid arm preempt the direct probe and reclaim a demonstrably live driver. The
``!= "local"`` condition is what confines the new field to the submit path.

Property under test, across every arm below: the payload's ``slurm_jobid`` is the
explicit argument when one is supplied, else ``$SLURM_JOB_ID`` when the mode is
not ``local``, else ``None``.
"""

import json
from pathlib import Path

from hhemt import orchestrator_sentinels as osent


def _payload(sentinel_path) -> dict:
    """Parse a written sentinel file back into its JSON payload."""
    return json.loads(Path(sentinel_path).read_text())


def test_submit_path_stamps_the_enclosing_allocation_job_id(tmp_path, monkeypatch):
    """Property: a non-local write records ``$SLURM_JOB_ID`` from the environment.

    Class: REGRESSION — red on the pre-change tree, where the field was populated
    only from the optional ``slurm_jobid`` argument and so recorded ``None`` for a
    driver running inside an allocation.

    Why required: ``_acquire_submit_driver_claim`` is the one call site that passes
    a real submission mode, and it passes no job id. A field populated only by the
    argument is therefore unreachable for exactly the detached driver this change
    protects — the one whose sentinel a later resume on another node must resolve.

    What kills it: reverting the payload entry to ``"slurm_jobid": slurm_jobid``,
    or dropping the environment read, leaves the value ``None`` and the assertion
    red.

    Second correct implementation, materially different: emit the dict literal
    without the key and assign ``payload["slurm_jobid"]`` afterwards as
    ``slurm_jobid if slurm_jobid is not None else os.getenv("SLURM_JOB_ID") if
    mode != "local" else None``. It reads the same two inputs through a different
    statement shape — ``os.getenv`` rather than ``os.environ.get``, assignment
    rather than an inline conditional — and passes this test unchanged.
    """
    monkeypatch.setenv("SLURM_JOB_ID", "48987")
    sentinel = osent.write_orchestrator_sentinel(
        tmp_path,
        driver_id="d-submit",
        workflow_submission_mode="batch_job",
    )
    assert _payload(sentinel)["slurm_jobid"] == "48987"


def test_local_self_sentinel_never_stamps_the_enclosing_job_id(tmp_path, monkeypatch):
    """Property: a ``local`` mode write records ``None`` even inside an allocation.

    Class: NEW CAPABILITY — the plausible wrong implementation is the
    unconditional stamp ``os.environ.get("SLURM_JOB_ID")``; it records ``48987``
    here and the assertion is red under it. The shipped conditional is green.

    Why required: the four reprocess / static-plots self-sentinels hardcode
    ``workflow_submission_mode="local"`` and are reclaimed by the gate's direct
    ``ps -p {pid}`` probe. A job id stamped on one of them adds a proxy arm that
    can preempt that direct measurement and reclaim a live driver — the failure is
    a false-dead verdict, not a cosmetic field.

    What kills it: dropping ``and workflow_submission_mode != "local"`` from the
    condition, or applying the stamp before the ``local`` branch is considered.

    Second correct implementation, materially different: an early
    ``stamp = slurm_jobid if workflow_submission_mode != "local" else None``
    computed above the payload literal and merged in as ``stamp or
    slurm_jobid``-style composition — different placement of the guard, same
    behaviour on this input.
    """
    monkeypatch.setenv("SLURM_JOB_ID", "48987")
    sentinel = osent.write_orchestrator_sentinel(
        tmp_path,
        driver_id="d-local",
        workflow_submission_mode="local",
        pid=4321,
    )
    assert _payload(sentinel)["slurm_jobid"] is None


def test_explicit_jobid_argument_wins_over_the_environment(tmp_path, monkeypatch):
    """Property: an explicit ``slurm_jobid`` argument outranks ``$SLURM_JOB_ID``.

    Class: NEW CAPABILITY — the plausible wrong implementation is a bare
    environment read that ignores the argument (``"slurm_jobid":
    os.environ.get("SLURM_JOB_ID")``); it records ``48987`` where the caller said
    ``55`` and the assertion is red under it.

    Why required: a single-job driver learns its own job id only after submission
    and reports it through the argument; a write-time stamp that outranks the
    caller would make the recorded id the PARENT allocation's rather than the
    driver's own, which is a different and wrong liveness subject.

    What kills it: dropping the ``if slurm_jobid is not None`` precedence arm.

    Second correct implementation, materially different: two sequential ``if``
    statements assigning a local ``resolved`` variable, argument arm first —
    same precedence expressed as control flow rather than as a nested conditional.
    """
    monkeypatch.setenv("SLURM_JOB_ID", "48987")
    sentinel = osent.write_orchestrator_sentinel(
        tmp_path,
        driver_id="d-explicit",
        workflow_submission_mode="1_job_many_srun_tasks",
        slurm_jobid="55",
    )
    assert _payload(sentinel)["slurm_jobid"] == "55"


def test_no_allocation_records_none_rather_than_raising(tmp_path, monkeypatch):
    """Property: with ``$SLURM_JOB_ID`` absent a non-local write records ``None``.

    Class: NEW CAPABILITY — the plausible wrong implementation is the direct
    subscript ``os.environ["SLURM_JOB_ID"]``, which raises ``KeyError`` on a
    login-node ``batch_job`` driver and loses the sentinel entirely; the assertion
    never runs under it. The shipped ``.get`` returns ``None``, which the gate
    already classifies as its UNKNOWN bucket.

    Why required: the login-node driver is the common case, so an unguarded
    subscript would break sentinel writing for the majority path — a write-site
    crash in front of the concurrency gate.

    What kills it: replacing ``os.environ.get`` with ``os.environ[...]``, or
    defaulting to ``""`` (an empty string is not the documented absence signal the
    gate reads).

    Second correct implementation, materially different: extract the value once
    above the payload with ``os.environ.get("SLURM_JOB_ID") or None`` — a single
    hoisted read with an falsy-normalisation rather than an inline conditional —
    and pass it through; same behaviour on this input.
    """
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    sentinel = osent.write_orchestrator_sentinel(
        tmp_path,
        driver_id="d-noalloc",
        workflow_submission_mode="batch_job",
    )
    assert _payload(sentinel)["slurm_jobid"] is None
