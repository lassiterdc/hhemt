"""End-to-end tests for the ``analysis.delete()`` distributed Snakemake workflow.

Per cleanup-rerun-delete-redesign Phase 2.

Fixture strategy: per-test ``start_from_scratch=True`` synth-multisim runs (the
``synth_multi_sim_analysis`` conftest fixture, function-scoped). Each test gets
an isolated completed analysis tree because delete is destructive — a shared
cached fixture would be consumed by the first test. The synth compile is
symlinked to a shared artifact cache so the per-test cost is the sim runs
only, not compilation.

Live-SLURM-sentinel tests monkeypatch ``workflow._slurm_job_is_live`` at the
module level, matching the established pattern in
``tests/test_synth_06_submission_guard.py`` (per cleanup-rerun-delete-redesign
Phase 2 design recommendation C.2).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hhemt import workflow
from hhemt.exceptions import ConfigurationError


def _write_live_sentinel(analysis_dir: Path, name: str, jobid: str) -> Path:
    d = analysis_dir / "_status" / "_submitted"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}.json"
    path.write_text(json.dumps({"slurm_jobid": jobid, "run_uuid": "u", "submitted_at": "t"}))
    return path


def test_delete_dry_run_summary_does_not_delete(synth_multi_sim_analysis, capsys):
    """``_print_delete_dry_run_summary`` prints the per-leaf breakdown and
    leaves the analysis_dir intact."""
    from hhemt.cli import _print_delete_dry_run_summary

    analysis = synth_multi_sim_analysis
    analysis_dir = analysis.analysis_paths.analysis_dir
    assert analysis_dir.exists()

    _print_delete_dry_run_summary(analysis)

    out = capsys.readouterr().out
    assert "Delete preview" in out
    assert "scenario" in out.lower() or "member" in out.lower()
    assert analysis_dir.exists()  # not deleted


def test_delete_removes_analysis_dir(synth_multi_sim_analysis, monkeypatch):
    """``Analysis.delete()`` with no live sentinels removes the analysis_dir
    after the distributed workflow writes all expected per-rule sentinels.

    Monkeypatches ``_recover_inflight_via_comment`` to a no-op so the sacct
    sweep does not shell out during the test.
    """
    analysis = synth_multi_sim_analysis
    analysis_dir = analysis.analysis_paths.analysis_dir
    assert analysis_dir.exists()

    builder = analysis._workflow_builder
    monkeypatch.setattr(builder, "_recover_inflight_via_comment", lambda known_jobids: [])

    analysis.delete(override_in_flight=False)

    assert not analysis_dir.exists()


def test_delete_refuses_when_live_sentinel(synth_multi_sim_analysis, monkeypatch):
    """A live ``_submitted/*.json`` sentinel must block delete with
    ``ConfigurationError`` and leave the analysis_dir intact."""
    analysis = synth_multi_sim_analysis
    analysis_dir = analysis.analysis_paths.analysis_dir
    builder = analysis._workflow_builder

    _write_live_sentinel(analysis_dir, "run_tritonswmm_evt-0", "999001")
    monkeypatch.setattr(workflow, "_slurm_job_is_live", lambda jid: True)
    monkeypatch.setattr(builder, "_recover_inflight_via_comment", lambda known_jobids: [])

    with pytest.raises(ConfigurationError, match="Refusing to delete"):
        analysis.delete(override_in_flight=False)

    assert analysis_dir.exists()  # preserved


def test_delete_override_in_flight_bypasses_guard(synth_multi_sim_analysis, monkeypatch, capsys):
    """``override_in_flight=True`` bypasses the live-sentinel refusal and
    proceeds with deletion; the bypass logs a stderr-visible line naming the
    live job-ids it proceeded against."""
    analysis = synth_multi_sim_analysis
    analysis_dir = analysis.analysis_paths.analysis_dir
    builder = analysis._workflow_builder

    _write_live_sentinel(analysis_dir, "run_tritonswmm_evt-0", "999001")
    monkeypatch.setattr(workflow, "_slurm_job_is_live", lambda jid: True)
    monkeypatch.setattr(builder, "_recover_inflight_via_comment", lambda known_jobids: [])

    analysis.delete(override_in_flight=True)

    captured = capsys.readouterr()
    assert "override_in_flight=True" in captured.out
    assert "999001" in captured.out
    assert not analysis_dir.exists()


def test_delete_preserves_dir_on_missing_sentinel(synth_multi_sim_analysis, monkeypatch):
    """If a per-rule sentinel is missing after the Snakemake delete workflow
    exits (simulating a partial-failure scenario), the orchestrator must
    PRESERVE the analysis_dir for debugging rather than destroy a
    partially-deleted tree."""
    analysis = synth_multi_sim_analysis
    analysis_dir = analysis.analysis_paths.analysis_dir
    builder = analysis._workflow_builder

    monkeypatch.setattr(builder, "_recover_inflight_via_comment", lambda known_jobids: [])

    # Stub the Snakemake submission so no _status/_deleting/*.flag files are
    # ever produced — the post-check should then find the expected set
    # missing and refuse the final fast_rmtree.
    def _stub_submit(self_builder, *, override_in_flight=False, override_multi_sim_run_method=None):
        self_builder._pre_delete_guards(override_in_flight=override_in_flight)
        return {"success": True, "stub": True}

    monkeypatch.setattr(
        type(builder),
        "submit_delete_workflow",
        _stub_submit,
    )

    analysis.delete(override_in_flight=False)

    # analysis_dir is preserved because the post-check expected sentinel set
    # is non-empty but the actual set is empty.
    assert analysis_dir.exists()


# ---------------------------------------------------------------------------
# D1 corroboration at the delete-recovery site (approved shape item 1e.15,
# site 1; work chunk 2b.2).
#
# ``_recover_inflight_via_comment``'s return value IS the delete preflight's
# refusal set (:meth:`_pre_delete_guards` merges it into ``alive``), so a job
# dropped on an unconfirmed "not live" verdict is a delete ADMITTED against a
# live analysis. That is the opposite direction from the sentinel classifier,
# where a dropped verdict unlinks a file. These tests therefore assert WHAT the
# corroboration DECIDES, never only that it ran.
# ---------------------------------------------------------------------------

_SCAN_ROWS = (
    "9001|RUNNING|rule_run_tritonswmm_wildcards_event_id=evt0\n"
    "9002|COMPLETED|rule_run_tritonswmm_wildcards_event_id=evt1\n"
)


class _FakeUserSacctRun:
    """Stand-in for ``subprocess.run`` on the recovery scan's own ``sacct -u``.

    Deliberately NOT a coherent stand-in for the corroborator's ``sacct -j``
    call as well: the two queries place the comment and the exit code in the
    same column index, so one fake buffer cannot serve both, and serving both
    would silently make the corroborator's answer a function of the scan's own
    fixture text. The corroborator is pinned separately through
    ``_sacct_states_batched``.
    """

    returncode = 0
    stdout = _SCAN_ROWS


def _stub_builder(analysis_dir: Path):
    """A builder-shaped stand-in carrying only the collaborators these tests reach.

    Deliberately NOT the ``synthetic_multisim_builder`` fixture. Constructing a
    real ``Analysis`` calls ``validation.running_identity``, which REFUSES a
    checkout with modified tracked files — and a Build turn's tree is dirty by
    construction, because A-22 leaves every write uncommitted. A fixture-based
    test therefore cannot be run by the seat that has to prove it, and a test
    the author cannot execute is not evidence. The four collaborators below are
    exactly the ones ``_pre_delete_guards`` calls; the method under test, the
    merge of the recovery set into ``alive``, is the real one.

    Coverage this does not buy: the object graph beyond those four calls. The
    refusal path itself is already covered end-to-end by
    ``test_delete_refuses_when_live_sentinel``; these tests add the RECOVERY
    source of that refusal, not the refusal mechanism.
    """
    import types

    b = types.SimpleNamespace()
    b.analysis_paths = types.SimpleNamespace(analysis_dir=analysis_dir)
    b._classify_live_sentinels = lambda sentinel_paths, reclaim_dead=True: []
    b._recover_pending_from_queued = lambda pending, analysis_dir: []
    b._recover_inflight_via_comment = lambda known_jobids: (
        workflow.SnakemakeWorkflowBuilder._recover_inflight_via_comment(b, known_jobids)
    )
    return b


def _pin_recovery_scan(monkeypatch, *, live_jids):
    """Pin the scan's sacct call and its squeue answer; 9001 is live, 9002 is not."""
    import subprocess as _sp

    monkeypatch.setattr(_sp, "run", lambda *a, **kw: _FakeUserSacctRun())
    monkeypatch.setattr("hhemt.workflow._slurm_job_is_live", lambda jid: jid in live_jids)


def _pin_corroborator_states(monkeypatch, states: dict):
    """Pin the second query's answer map.

    An EMPTY map is what an aged-out accounting record, an absent ``sacct`` and
    a timeout all collapse to — the single NOT-KNOWN state — so ``{}`` is the
    unconfirmable arm rather than a special case invented for the test.
    """
    monkeypatch.setattr(
        "hhemt.workflow._sacct_states_batched",
        lambda jids, **kw: {j: states[j] for j in jids if j in states},
    )


def test_recover_inflight_holds_job_when_second_query_cannot_answer(monkeypatch, tmp_path):
    """An unconfirmable second query INCLUDES the job; it does not admit the delete.

    Property: with a comment-matched job whose squeue answer is "not live" and
    whose corroboration returns no row, ``_recover_inflight_via_comment``
    returns it, so ``_pre_delete_guards`` refuses.

    Class: REGRESSION — red on the pre-change tree, where the scan returned
    ``[]`` for 9002 and the delete proceeded.

    Why required: the returned set is the delete's refusal set, so omission is
    the unsafe direction; inclusion costs one operator valve press.

    What kills it: dropping the corroboration, or resolving "unknown" to dead.

    A second correct implementation: batching the second query once for all
    not-live candidates instead of once per candidate, then mapping an absent
    id to "unknown" — same returned set, different call structure.
    """
    _pin_recovery_scan(monkeypatch, live_jids={"9001"})
    _pin_corroborator_states(monkeypatch, {})

    found = workflow.SnakemakeWorkflowBuilder._recover_inflight_via_comment(_stub_builder(tmp_path), known_jobids=set())

    assert {jid for _, jid in found} == {"9001", "9002"}


def test_recover_inflight_omits_job_on_corroborated_death(monkeypatch, tmp_path):
    """A corroborated death still omits the job — the corroboration is a gate, not a block.

    Property: a terminal sacct State for the not-live job leaves it out of the
    returned set, exactly as the pre-change behaviour did.

    Class: NEW CAPABILITY — plausible wrong implementation is "include every
    comment-matched job once the corroborator is consulted", under which this
    test reds.

    Why required: without this arm, "corroborate" could mean "never omit" and
    every aged-out sim would block every delete forever.

    What kills it: treating ``"dead"`` as a hold.

    A second correct implementation: consulting ``job_is_dead_confirmed``
    instead of the three-state helper — it returns True (dead) here and False
    on the unknown arm, which this test does not exercise.
    """
    _pin_recovery_scan(monkeypatch, live_jids={"9001"})
    _pin_corroborator_states(monkeypatch, {"9002": ("COMPLETED", "0:0", "None")})

    found = workflow.SnakemakeWorkflowBuilder._recover_inflight_via_comment(_stub_builder(tmp_path), known_jobids=set())

    assert {jid for _, jid in found} == {"9001"}


def test_recover_inflight_includes_job_on_non_terminal_sacct_row(monkeypatch, tmp_path):
    """A non-terminal sacct row includes the job: it is still in the scheduler.

    Property: the ``"alive"`` answer resolves to inclusion, not to a hold that
    a later age tier could expire.

    Class: REGRESSION — red on the pre-change tree, where 9002 was omitted.

    Why required: squeue can be blind to a hidden-partition job that sacct
    still reports as running; dropping it would admit the delete.

    What kills it: collapsing "alive" into "unknown" and applying an age tier
    this site does not have.

    A second correct implementation: an explicit ``"alive"`` branch before the
    unknown branch, or a single ``!= "dead"`` test — both yield the same set.
    """
    _pin_recovery_scan(monkeypatch, live_jids={"9001"})
    _pin_corroborator_states(monkeypatch, {"9002": ("RUNNING", "0:0", "None")})

    found = workflow.SnakemakeWorkflowBuilder._recover_inflight_via_comment(_stub_builder(tmp_path), known_jobids=set())

    assert {jid for _, jid in found} == {"9001", "9002"}


def test_pre_delete_guards_refuses_when_recovery_holds_unconfirmed_job(monkeypatch, tmp_path):
    """The refusal set actually refuses: an unconfirmable recovery hit blocks the delete.

    Property: with NO live sentinel and NO queued sentinel, the recovery scan is
    the only source of live jobs, and ``_pre_delete_guards`` raises naming 9002.

    Class: REGRESSION — red on the pre-change tree, where the recovered set was
    empty and the guard admitted the delete.

    Why required: this is the criterion's "and ``_pre_delete_guards`` admits the
    delete" arm; a corroboration inside the scanner that the consumer ignored
    would satisfy the scanner tests and still lose the safeguard.

    What kills it: dropping the ``alive += self._recover_inflight_via_comment``
    merge, or resolving unknown to dead.

    A second correct implementation: merging the recovery result before the
    ``_queued`` sweep rather than after — the union is the same set.
    """
    _pin_recovery_scan(monkeypatch, live_jids={"9001"})
    _pin_corroborator_states(monkeypatch, {})

    with pytest.raises(ConfigurationError, match="Refusing to delete") as excinfo:
        workflow.SnakemakeWorkflowBuilder._pre_delete_guards(_stub_builder(tmp_path), override_in_flight=False)

    assert "9002" in str(excinfo.value)


def test_pre_delete_guards_override_proceeds_when_recovery_holds_unconfirmed_job(monkeypatch, tmp_path, capsys):
    """``override_in_flight=True`` remains the operator valve on the held-job path.

    Property: the guard does not raise, and it logs the job id it proceeded
    against, so the D1 over-block is visible rather than silent.

    Class: NEW CAPABILITY — plausible wrong implementation is "refuse whenever
    the recovery scan returns anything, ignoring the override", under which
    this test reds.

    Why required: the docstring names the operator valve as the bound that
    replaces the age tier this site lacks; an unpassable valve would wedge
    every tree whose accounting record aged out.

    What kills it: dropping the ``and not override_in_flight`` conjunct, or the
    ``if alive and override_in_flight`` log branch.

    A second correct implementation: logging before the raise/print fork with
    the same job-id list — same observable line, different position.
    """
    _pin_recovery_scan(monkeypatch, live_jids={"9001"})
    _pin_corroborator_states(monkeypatch, {})

    workflow.SnakemakeWorkflowBuilder._pre_delete_guards(_stub_builder(tmp_path), override_in_flight=True)

    out = capsys.readouterr().out
    assert "override_in_flight=True" in out
    assert "9002" in out


def test_recover_inflight_stderr_reports_the_held_count(monkeypatch, tmp_path, capsys):
    """The held count an operator is told to read is asserted, not just emitted.

    Property: the diagnostic splits ``len(found)`` into "confirmed live" and
    "held on an unconfirmable second query", and the held component tracks the
    corroborator's ``"unknown"`` answer — 1 held when the second query cannot
    answer, 0 held when it returns a terminal State.

    Class: NEW CAPABILITY — plausible wrong implementation is the one this test
    was written against, a counter that never increments (mutant D): the
    diagnostic then reports the held population as confirmed-live while every
    other test in the selection stays green.

    Why required: the docstring names a persistent non-zero held count as the
    observable that a tree will refuse every delete until an operator passes
    ``override_in_flight``. An operator decision procedure resting on an
    unasserted number is a documented procedure with no coverage, and the held
    population is the half of the bound that is not the valve itself.

    What kills it: deleting the ``held_unconfirmed += 1`` increment, or folding
    the two components back into a single count.

    A second correct implementation: computing the held count as
    ``sum(1 for verdict ... == "unknown")`` in a post-pass rather than
    incrementing in the loop — same emitted numbers, different accumulation.
    """
    _pin_recovery_scan(monkeypatch, live_jids={"9001"})
    _pin_corroborator_states(monkeypatch, {})

    workflow.SnakemakeWorkflowBuilder._recover_inflight_via_comment(_stub_builder(tmp_path), known_jobids=set())
    held_err = capsys.readouterr().err
    assert "2 in-flight (1 confirmed live, 1 held on an unconfirmable second query)" in held_err

    # Differently-positioned input: the same population corroborated DEAD moves
    # the held component to 0 while the confirmed-live component is unchanged,
    # so the assertion turns on the verdict and not on a constant.
    _pin_corroborator_states(monkeypatch, {"9002": ("COMPLETED", "0:0", "None")})
    workflow.SnakemakeWorkflowBuilder._recover_inflight_via_comment(_stub_builder(tmp_path), known_jobids=set())
    dead_err = capsys.readouterr().err
    assert "1 in-flight (1 confirmed live, 0 held on an unconfirmable second query)" in dead_err
