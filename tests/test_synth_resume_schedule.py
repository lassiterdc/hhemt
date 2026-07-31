"""Fast, compile-free unit tests for the multi-resume interruption schedule
watcher arithmetic (Phase 2).

These pin the per-attempt indexing that replaced the single-kill
``deterministic_kill_after_n_checkpoints`` harness. The watcher
(``TRITONSWMM_run.wait_with_deterministic_checkpoint_kill``) is unchanged — it
still fires once ``len(cfgs) >= n_checkpoints + 1``. What Phase 2 changed is how
the CALLER (``run_simulation_runner``) computes ``n_checkpoints``: it arms each
attempt with ``resume_interruption_schedule[n_resumes]`` — the ABSOLUTE schedule
entry, with NO per-attempt baseline subtraction — because the hotstart cfg dir
accumulates monotonically across attempts.

The two mutations this phase forecloses:
  1. same entry on every attempt (the retired single-kill bug), and
  2. subtracting a per-attempt baseline from an absolute entry,
both LOWER the effective threshold so the kill fires at ``schedule[0] + 1`` cfg
files — the premature self-kill at t~=0 that burns every retry. The tests below
drive the real watcher and assert the correct absolute entry does NOT fire there
while both mutations WOULD.

No cluster, no compilation, no real subprocess or process group.
"""

import os
import signal
import types

import pytest
from pydantic import ValidationError

from hhemt.config.analysis import analysis_config
from hhemt.exceptions import ConfigurationError
from hhemt.run_simulation import TRITONSWMM_run
from hhemt.validation import preflight_validate

# A representative strictly-increasing schedule (small so the tests are trivial).
# The production synth value is (36, 72, 108); the arithmetic is identical.
_SCHEDULE = (2, 4, 6)


class _FakeProc:
    """Minimal Popen stand-in: poll() returns None until 'killed' (via the
    monkeypatched os.killpg, which sets _killed) or until the sim 'completes'
    after ``complete_after_polls`` polls. ``pid`` lets the watcher compute a pgid
    via os.getpgid(proc.pid) and signal the group with os.killpg."""

    def __init__(self, complete_after_polls=None, pid=4242):
        self._killed = False
        self._polls = 0
        self._complete_after = complete_after_polls
        self.pid = pid

    def poll(self):
        if self._killed:
            return -15  # bash killed by the process-group SIGTERM
        self._polls += 1
        if self._complete_after is not None and self._polls > self._complete_after:
            return 0
        return None

    @property
    def returncode(self):
        return -15 if self._killed else 0


def _install_fake_killpg(monkeypatch, proc, recorder):
    """Patch os.getpgid/os.killpg so the watcher's process-group SIGTERM is
    captured (and marks the fake proc 'killed') without touching a real group."""
    monkeypatch.setattr(os, "getpgid", lambda pid: pid)

    def _fake_killpg(pgid, sig):
        recorder["pgid"] = pgid
        recorder["sig"] = sig
        proc._killed = True

    monkeypatch.setattr(os, "killpg", _fake_killpg)


def _cfg_dir_with(tmp_path, n_files):
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    for i in range(n_files):
        (cfg_dir / f"config_{i:04d}.cfg").write_text("x\n")
    return cfg_dir


def _entry_for_attempt(schedule, n_resumes):
    """The correct absolute checkpoint index the runner arms on the attempt
    following ``n_resumes`` resumes: ``schedule[n_resumes]``, NO baseline
    subtraction. Mirrors ``run_simulation_runner``'s ``_schedule[_n_done]``."""
    return schedule[n_resumes]


def test_correct_absolute_entry_does_not_fire_at_prior_baseline(tmp_path, monkeypatch):
    """After resume 1 the cfg dir holds ``schedule[0] + 1`` files. The runner arms
    the next attempt with the ABSOLUTE ``schedule[1]``; the watcher must NOT fire
    at that accumulated baseline (``schedule[0]+1 < schedule[1]+1``), so the sim
    runs on and completes untouched."""
    # State after resume 1: schedule[0] + 1 = 3 accumulated cfg files.
    cfg_dir = _cfg_dir_with(tmp_path, _SCHEDULE[0] + 1)
    fake_self = types.SimpleNamespace(_hotstart_cfg_dir=lambda mt: cfg_dir)

    n_checkpoints = _entry_for_attempt(_SCHEDULE, 1)  # schedule[1] = 4 (ABSOLUTE)
    assert n_checkpoints == 4

    proc = _FakeProc(complete_after_polls=2)  # sim finishes before threshold
    rec = {}
    _install_fake_killpg(monkeypatch, proc, rec)
    rc = TRITONSWMM_run.wait_with_deterministic_checkpoint_kill(
        fake_self, proc, model_type="tritonswmm", n_checkpoints=n_checkpoints, poll_interval_s=0
    )
    assert proc._killed is False
    assert rec == {}  # os.killpg never called at the prior-resume baseline
    assert rc == 0


def test_correct_absolute_entry_fires_at_schedule_k_plus_one(tmp_path, monkeypatch):
    """Once ``schedule[1] + 1`` cfg files exist, the same absolute
    ``n_checkpoints = schedule[1]`` fires exactly one process-group SIGTERM."""
    cfg_dir = _cfg_dir_with(tmp_path, _SCHEDULE[1] + 1)  # 5 files
    fake_self = types.SimpleNamespace(_hotstart_cfg_dir=lambda mt: cfg_dir)

    n_checkpoints = _entry_for_attempt(_SCHEDULE, 1)  # 4
    proc = _FakeProc(pid=4242)
    rec = {}
    _install_fake_killpg(monkeypatch, proc, rec)
    rc = TRITONSWMM_run.wait_with_deterministic_checkpoint_kill(
        fake_self, proc, model_type="tritonswmm", n_checkpoints=n_checkpoints, poll_interval_s=0
    )
    assert rec["pgid"] == 4242
    assert rec["sig"] == signal.SIGTERM
    assert proc._killed is True
    assert rc == -15


def test_same_entry_every_attempt_mutation_fires_prematurely(tmp_path, monkeypatch):
    """Mutation 1 (the retired single-kill bug): passing ``schedule[0]`` on the
    attempt AFTER resume 1 lowers the threshold to ``schedule[0]+1`` — already met
    by the accumulated dir — so the kill fires at t~=0. This test fails (the kill
    fires) if the runner ever regresses to a per-attempt-constant entry."""
    cfg_dir = _cfg_dir_with(tmp_path, _SCHEDULE[0] + 1)  # 3 files (post-resume-1)
    fake_self = types.SimpleNamespace(_hotstart_cfg_dir=lambda mt: cfg_dir)

    mutation_n = _SCHEDULE[0]  # 2 -> threshold 3, met by 3 files -> PREMATURE fire
    proc = _FakeProc(pid=4242)
    rec = {}
    _install_fake_killpg(monkeypatch, proc, rec)
    TRITONSWMM_run.wait_with_deterministic_checkpoint_kill(
        fake_self, proc, model_type="tritonswmm", n_checkpoints=mutation_n, poll_interval_s=0
    )
    assert proc._killed is True  # the mutation self-kills at the accumulated baseline
    assert rec["sig"] == signal.SIGTERM


def test_baseline_subtracting_mutation_fires_prematurely(tmp_path, monkeypatch):
    """Mutation 2 (baseline subtraction): passing ``schedule[1] - (schedule[0]+1)``
    lowers the threshold below the accumulated file count, so the kill fires
    prematurely. The correct absolute entry (tested above) does not. Pins that the
    runner must pass the ABSOLUTE entry, never an offset one."""
    cfg_dir = _cfg_dir_with(tmp_path, _SCHEDULE[0] + 1)  # 3 files
    fake_self = types.SimpleNamespace(_hotstart_cfg_dir=lambda mt: cfg_dir)

    baseline = _SCHEDULE[0] + 1  # 3
    mutation_n = _SCHEDULE[1] - baseline  # 4 - 3 = 1 -> threshold 2, met -> PREMATURE fire
    proc = _FakeProc(pid=4242)
    rec = {}
    _install_fake_killpg(monkeypatch, proc, rec)
    TRITONSWMM_run.wait_with_deterministic_checkpoint_kill(
        fake_self, proc, model_type="tritonswmm", n_checkpoints=mutation_n, poll_interval_s=0
    )
    assert proc._killed is True  # baseline-subtracting mutation self-kills prematurely
    assert rec["sig"] == signal.SIGTERM


def test_graceful_degradation_entry_beyond_checkpoint_count(tmp_path, monkeypatch):
    """An entry beyond the sim's realizable checkpoint count degrades gracefully:
    the threshold is never met, no kill fires, and the sim completes normally
    (proc.wait() returns its clean return code)."""
    cfg_dir = _cfg_dir_with(tmp_path, 3)  # sim only ever produces 3 checkpoints
    fake_self = types.SimpleNamespace(_hotstart_cfg_dir=lambda mt: cfg_dir)

    n_checkpoints = 100  # far beyond the 3 checkpoints this sim can produce
    proc = _FakeProc(complete_after_polls=2)
    rec = {}
    _install_fake_killpg(monkeypatch, proc, rec)
    rc = TRITONSWMM_run.wait_with_deterministic_checkpoint_kill(
        fake_self, proc, model_type="tritonswmm", n_checkpoints=n_checkpoints, poll_interval_s=0
    )
    assert proc._killed is False
    assert rec == {}
    assert rc == 0


# --- resume_interruption_schedule field validator (four rejection classes) -----
#
# cfgBaseModel is extra="forbid" with no validate_assignment, so the @field_validator
# fires at model_validate time, not on mutation. We build a VALID base from a real
# synth analysis config's dump and override only the schedule, so the schedule field
# is the ONLY thing that can raise -- asserted by the message containing the field
# name (a bare ValidationError would otherwise pass for an unrelated reason).


@pytest.mark.parametrize(
    "bad_schedule",
    [
        (),           # empty tuple -> use None to disable
        (2, 2, 4),    # duplicates
        (0, 2, 4),    # non-positive (zero)
        (-1, 3),      # non-positive (negative)
        (4, 2, 6),    # non-increasing (decrease)
        (2, 2),       # non-strictly-increasing (equal adjacent)
    ],
)
def test_field_validator_rejects_invalid_schedule(synth_multi_sim_analysis, bad_schedule):
    valid_dump = synth_multi_sim_analysis.cfg_analysis.model_dump()
    with pytest.raises(ValidationError) as exc:
        analysis_config.model_validate({**valid_dump, "resume_interruption_schedule": bad_schedule})
    assert "resume_interruption_schedule" in str(exc.value)


def test_field_validator_accepts_valid_schedule_and_none(synth_multi_sim_analysis):
    valid_dump = synth_multi_sim_analysis.cfg_analysis.model_dump()
    cfg = analysis_config.model_validate(
        {**valid_dump, "resume_interruption_schedule": (36, 72, 108)}
    )
    assert cfg.resume_interruption_schedule == (36, 72, 108)
    cfg_none = analysis_config.model_validate(
        {**valid_dump, "resume_interruption_schedule": None}
    )
    assert cfg_none.resume_interruption_schedule is None


# --- R6 preflight rejection: schedule + 1_job_many_srun_tasks (DoD item) --------


def test_preflight_rejects_schedule_with_1_job_many_srun_tasks(synth_multi_sim_analysis):
    """R6: a set resume_interruption_schedule under
    multi_sim_run_method='1_job_many_srun_tasks' must surface a preflight error
    that raises ConfigurationError -- that mode lacks the job-end cgroup reap that
    makes repeated per-attempt step teardown safe."""
    a = synth_multi_sim_analysis
    cfg_sys = a._system.cfg_system
    cfg_analysis = a.cfg_analysis
    cfg_analysis.multi_sim_run_method = "1_job_many_srun_tasks"
    cfg_analysis.resume_interruption_schedule = (36, 72, 108)

    result = preflight_validate(cfg_sys, cfg_analysis)
    schedule_errors = [e for e in result.errors if "resume_interruption_schedule" in e.field]
    assert schedule_errors, "preflight did not flag the schedule under 1_job_many_srun_tasks"
    assert not result.is_valid
    with pytest.raises(ConfigurationError):
        result.raise_if_invalid()


def test_preflight_accepts_schedule_with_batch_job(synth_multi_sim_analysis):
    """The same schedule under batch_job (the mode that DOES get the job-end cgroup
    reap) raises no schedule-specific preflight error."""
    a = synth_multi_sim_analysis
    cfg_sys = a._system.cfg_system
    cfg_analysis = a.cfg_analysis
    cfg_analysis.multi_sim_run_method = "batch_job"
    cfg_analysis.resume_interruption_schedule = (36, 72, 108)

    result = preflight_validate(cfg_sys, cfg_analysis)
    assert not [e for e in result.errors if "resume_interruption_schedule" in e.field]


# --- KR-a: deterministic same-timestep interruption prune ----------------------
#
# These drive the REAL picker and the REAL prune helper against a synthesized cfg
# dir carrying an OVERSHOOT (the poll-granularity artifact that made the realized
# resume boundary vary per config). The first test FAILS against pre-KR-a code:
# without the prune the picker returns the highest complete cfg, which is
# schedule[k] + M, not schedule[k].
#
# Naming note: these fixtures use the REAL on-disk format — 1-based and UNPADDED
# (config_1.cfg ... config_1080.cfg, measured on a live synth run). The older
# _cfg_dir_with above writes 0-based zero-padded names; both parse, but only this
# one describes what TRITON actually writes.


def _real_cfg_dir(tmp_path, max_step):
    """config_1.cfg .. config_{max_step}.cfg — 1-based, contiguous, unpadded."""
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    for i in range(1, max_step + 1):
        (cfg_dir / f"config_{i}.cfg").write_text("line1\nline2\nline3\n")
    return cfg_dir


def _steps_on_disk(cfg_dir):
    return sorted(int(p.name.split("_")[-1].split(".")[0]) for p in cfg_dir.glob("*.cfg"))


def test_prune_forces_picker_to_scheduled_step_despite_overshoot(tmp_path):
    """The kill lands late (poll granularity), so the dir holds schedule[k] + M cfgs.
    Pre-KR-a the picker returns schedule[k] + M; after the prune it returns exactly
    schedule[k]. This is the whole hard requirement in one assertion."""
    from hhemt.run_simulation import (
        TRITONSWMM_run,
        return_the_reporting_step_from_a_cfg,
    )

    target_step = _SCHEDULE[0]  # 2
    overshoot = 3
    cfg_dir = _real_cfg_dir(tmp_path, target_step + overshoot)  # steps 1..5

    out_dir = cfg_dir.parent
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    fake_self = types.SimpleNamespace(
        _hotstart_cfg_dir=lambda mt: cfg_dir,
        _scenario=types.SimpleNamespace(
            scen_paths=types.SimpleNamespace(
                out_triton=out_dir,
                out_tritonswmm=out_dir,
                triton_cfg=out_dir / "base.cfg",
                triton_swmm_cfg=out_dir / "base.cfg",
            )
        ),
        _analysis=types.SimpleNamespace(
            analysis_paths=types.SimpleNamespace(analysis_dir=analysis_dir)
        ),
    )

    # Pre-fix behaviour: the picker takes the HIGHEST complete cfg -> the overshoot.
    picked_before = TRITONSWMM_run._retrieve_hotstart_file_for_incomplete_triton_or_tritonswmm_simulation(
        fake_self, model_type="triton"
    )
    assert return_the_reporting_step_from_a_cfg(picked_before) == target_step + overshoot

    n_removed = TRITONSWMM_run.prune_hotstart_cfgs_above_step(
        fake_self, "triton", target_step=target_step
    )
    assert n_removed == overshoot

    picked_after = TRITONSWMM_run._retrieve_hotstart_file_for_incomplete_triton_or_tritonswmm_simulation(
        fake_self, model_type="triton"
    )
    assert return_the_reporting_step_from_a_cfg(picked_after) == target_step


def test_prune_deletes_only_the_top_preserving_contiguity_from_one(tmp_path):
    """The count-based kill arming (wait_with_deterministic_checkpoint_kill) relies on
    len(cfgs) == max(step), which holds only while numbering stays contiguous from 1.
    A prune that removed interior files would silently push every later kill late, so
    pin that the prune removes a strict SUFFIX."""
    from hhemt.run_simulation import TRITONSWMM_run

    cfg_dir = _real_cfg_dir(tmp_path, 9)
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    fake_self = types.SimpleNamespace(
        _hotstart_cfg_dir=lambda mt: cfg_dir,
        _analysis=types.SimpleNamespace(
            analysis_paths=types.SimpleNamespace(analysis_dir=analysis_dir)
        ),
    )

    TRITONSWMM_run.prune_hotstart_cfgs_above_step(fake_self, "triton", target_step=4)

    steps = _steps_on_disk(cfg_dir)
    assert steps == [1, 2, 3, 4]
    assert len(steps) == max(steps)  # the identity the watcher's count predicate needs


def test_prune_is_a_noop_on_a_fresh_attempt_with_no_cfg_dir(tmp_path):
    """Attempt 0 has no cfg dir. The helper must return 0 rather than raise — this
    no-op is what disambiguates attempt 0 from attempt 1 (both read n_resumes == 0)."""
    from hhemt.run_simulation import TRITONSWMM_run

    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    fake_self = types.SimpleNamespace(
        _hotstart_cfg_dir=lambda mt: tmp_path / "does_not_exist",
        _analysis=types.SimpleNamespace(
            analysis_paths=types.SimpleNamespace(analysis_dir=analysis_dir)
        ),
    )
    assert TRITONSWMM_run.prune_hotstart_cfgs_above_step(fake_self, "triton", target_step=2) == 0
