"""The coupled resume picker steps back past a cut-off SWMM snapshot, and the ledger records it.

THE PROBLEM. A coupled TRITON-SWMM resume from ``config_N.cfg`` needs checkpoint N's SWMM
state snapshot. When the solver is killed while writing that snapshot, N carries a ``.part``
or ``.killed`` marker and no published snapshot; the toolkit used to hand the solver
``config_N.cfg`` anyway, the solver aborted, and every retry repeated the abort. The picker
in ``hhemt.run_simulation.TRITONSWMM_run`` now steps back to the newest checkpoint with an
intact snapshot, refuses to when the cfg's ``output_folder`` is not the one it can see, and
stops (selects the checkpoint whose write FAILED) when a ``.failed`` marker is involved,
because a failed write is a stop condition rather than a fall-back. The ledger in
``hhemt.resume_events`` records
the solver's refusal lines as rows and the picker's step-back on the exec's snapshot row.

No test here compiles or runs a solver. The prepare_simulation_command arms stop at a
sentinel raised just past the hotstart branch, so nothing after the picker executes.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

import hhemt
import hhemt.scenario as scenario_mod
from hhemt import resume_events as re_mod
from hhemt.run_simulation import TRITONSWMM_run
from hhemt.synthetic_model import triton_cfg

# --------------------------------------------------------------------------------------
# Fixture builders
# --------------------------------------------------------------------------------------


class _Field:
    def __init__(self, value=None):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class _Stop(Exception):
    """Raised by the fake system's ``cfg_system``, which prepare_simulation_command reads first
    after the hotstart branch -- so a test can inspect the branch's effects and run nothing
    past it."""


class _System:
    @property
    def cfg_system(self):
        raise _Stop


def _generated_cfg_text(tmp_path: Path, monkeypatch) -> str:
    """The base cfg exactly as ``TRITONSWMM_scenario._generate_TRITON_SWMM_cfg`` writes it.

    Produced by the generator's own code rather than hand-written, because the generator
    writes the QUOTED form ``output_folder="out_tritonswmm"`` and a hand-written unquoted
    fixture would let a guard that compares raw text pass.
    """
    work = tmp_path / "gen"
    work.mkdir()
    template = work / "tritonswmm.cfg"
    template.write_text(triton_cfg._TEMPLATE, encoding="utf-8")
    extbc = work / "extbc.csv"
    extbc.write_text("x,y\n1,2\n")
    out_cfg = work / "TRITONSWMM.cfg"
    options = SimpleNamespace(
        Value=SimpleNamespace(
            START_DATE="01/01/2020", START_TIME="00:00:00", END_DATE="01/01/2020", END_TIME="01:00:00"
        )
    )
    monkeypatch.setattr(
        scenario_mod.swmmio, "Model", lambda path: SimpleNamespace(inp=SimpleNamespace(options=options))
    )
    fake = SimpleNamespace(
        _system=SimpleNamespace(
            cfg_system=SimpleNamespace(
                toggle_use_constant_mannings=True,
                constant_mannings=0.03,
                triton_swmm_configuration_template=template,
                TRITONSWMM_software_directory=None,
                TRITONSWMM_branch_key=None,
            ),
            sys_paths=SimpleNamespace(dem_processed=work / "dem.asc", mannings_processed=work / "n.asc"),
            log=SimpleNamespace(triton_head_sha=SimpleNamespace(get=lambda: None)),
        ),
        _analysis=SimpleNamespace(
            cfg_analysis=SimpleNamespace(
                manhole_diameter=1.0,
                manhole_loss_coefficient=0.5,
                TRITON_raw_output_type="bin",
                hydraulic_timestep_s=1,
                TRITON_reporting_timestep_s=60,
                open_boundaries=0,
                swmm_snapshot_disable=False,
                swmm_snapshot_keep_all=False,
                swmm_node_rim_from_dem=False,
                one_coupled_inflow_node_per_dem_cell=False,
                execution_environment="native",
            )
        ),
        _swmm_inp_for_sim_duration=lambda: work / "hydraulics.inp",
        scen_paths=SimpleNamespace(
            extbc_loc=extbc,
            swmm_hydraulics_inp=work / "hydraulics.inp",
            hyg_timeseries=work / "hyg.csv",
            hyg_locs=work / "hyg_locs.csv",
            triton_swmm_cfg=out_cfg,
        ),
        _n_hydrograph_sources=lambda: 1,
        sim_id_str="evt0",
        log=SimpleNamespace(triton_swmm_cfg_created=_Field()),
    )
    scenario_mod.TRITONSWMM_scenario._generate_TRITON_SWMM_cfg(fake)
    return out_cfg.read_text()


def _sim(tmp_path: Path, cfg_text: str, steps=(4, 5), *, failed_replay: bool = False):
    """A coupled sim directory with one cfg per step, all carrying ``cfg_text``, and a run.

    ``failed_replay`` is the value of the analysis's ``swmm_snapshot_failed_replay`` field.
    """
    sim = tmp_path / "sim"
    out = sim / "out_tritonswmm"
    (out / "cfg").mkdir(parents=True)
    (out / "swmm").mkdir()
    for k in steps:
        (out / "cfg" / f"config_{k}.cfg").write_text(cfg_text)
    log = SimpleNamespace(force_rerun_pending=_Field(False), n_resumes=_Field(0), resume_reporting_tsteps=_Field(None))
    analysis = SimpleNamespace(
        cfg_analysis=SimpleNamespace(
            multi_sim_run_method="local",
            run_mode="serial",
            n_mpi_procs=1,
            n_omp_threads=1,
            n_gpus=0,
            n_nodes=1,
            execution_environment="native",
            is_experiment_member=False,
            swmm_snapshot_failed_replay=failed_replay,
        ),
        cfg_hpc_system=None,
        analysis_paths=SimpleNamespace(simlog_directory=tmp_path / "logs" / "sims"),
        _system=_System(),
    )
    scen = SimpleNamespace(
        event_iloc=0,
        weather_event_indexers={},
        _analysis=analysis,
        scen_paths=SimpleNamespace(
            out_tritonswmm=out,
            out_triton=sim / "out_triton",
            swmm_hydraulics_inp=sim / "swmm" / "hydraulics.inp",
            triton_swmm_cfg=sim / "TRITONSWMM.cfg",
            sim_tritonswmm_executable=sim / "build" / "triton.exe",
            triton_cfg=sim / "TRITON.cfg",
            sim_triton_executable=sim / "build" / "triton_only.exe",
        ),
        get_log=lambda model_type: log,
        model_run_completed=lambda model_type: False,
    )
    (sim / "TRITONSWMM.cfg").write_text(cfg_text)
    return TRITONSWMM_run(scen), out, log


def _snap(out: Path, k: int, suffix: str = "") -> Path:
    p = out / "swmm" / f"hydraulics_state_cp{k}.snapshot{suffix}"
    p.write_bytes(b"x")
    return p


def _listing(out: Path) -> list[str]:
    return sorted(str(p.relative_to(out)) for p in out.rglob("*"))


def _realised_resume_step(run, log) -> int:
    with pytest.raises(_Stop):
        run.prepare_simulation_command(pickup_where_leftoff=True, verbose=False, model_type="tritonswmm")
    return log.resume_reporting_tsteps.get()[-1]


_QUOTED = 'output_folder="out_tritonswmm"\ndem_filename="d"\n'


# --------------------------------------------------------------------------------------
# The picker
# --------------------------------------------------------------------------------------


def test_cut_off_newest_snapshot_steps_back_to_the_published_one(tmp_path, monkeypatch):
    """Property: a cut-off snapshot at N with a published one below makes the resume start below N.

    Class: the cut-off input class -- N's snapshot unpublished with a ``.part`` or
    ``.killed`` marker, a published snapshot at a lower complete checkpoint -- driven through
    ``prepare_simulation_command`` with a cfg the generator itself wrote.

    Why required, checkable against the code: launching at N makes the solver abort on every
    retry; resuming at the published checkpoint loses only the work since it. The realised
    resume step and the single ``n_resumes`` increment are read from the log fields the
    hotstart branch writes, so a step-back counts as ONE resume attempt.

    What kills it: skipping the picker, placing it after the realised step is recorded, or
    reading the snapshot from a directory the cfg's output_folder does not name.

    A second correct implementation under which it still passes: a picker that scans the
    snapshot directory once and indexes it, instead of statting each candidate name.
    """
    cfg = _generated_cfg_text(tmp_path, monkeypatch)
    assert 'output_folder="out_tritonswmm"' in cfg
    run, out, log = _sim(tmp_path, cfg)
    _snap(out, 4)
    _snap(out, 5, ".part")
    assert _realised_resume_step(run, log) == 4
    assert log.n_resumes.get() == 1
    assert run.stepback_record == {"unusable_step": 5, "marker": ".part", "reason": "killed-write", "used_step": 4}


def test_the_picker_deletes_nothing(tmp_path):
    """Property: choosing a step-back leaves the snapshot and cfg directories byte-for-name unchanged.

    Class: the cut-off input class above. Satisfying positions differ only in whether a
    step-back is taken, and the listing must match either way.

    What kills it: a picker that renames the marker, removes the unusable snapshot or prunes
    the cfgs above M.

    A second correct implementation under which it still passes: one that copies M's cfg to
    a scratch location before returning it -- the listing under the sim's output dir is the
    assertion.
    """
    run, out, _ = _sim(tmp_path, _QUOTED)
    _snap(out, 4)
    _snap(out, 5, ".killed")
    before = _listing(out)
    chosen = run._pick_resume_cfg_past_unusable_snapshot(out / "cfg" / "config_5.cfg", verbose=False)
    assert chosen.name == "config_4.cfg"
    assert _listing(out) == before


@pytest.mark.parametrize(
    ("files_at_5", "published_4"),
    [
        (("",), True),  # N itself published
        (("", ".killed"), True),  # published N beside a stale marker
        ((".killed",), False),  # cut-off N, nothing published below
        ((), False),  # absent N, nothing published below
    ],
)
def test_selects_n_when_no_step_back_is_due(tmp_path, files_at_5, published_4):
    """Property: the picker selects N whenever N is usable or nothing below it is.

    Class: the satisfying positions for the step-back arm -- a published N (with and without
    a stale ``.killed``), and an unusable N with nothing published below.

    What kills it: stepping back when N is published, or returning a cfg below N with no
    published snapshot.

    A second correct implementation under which it still passes: one that evaluates STEP 2
    before STEP 1 but only selects M when N is unpublished.
    """
    run, out, _ = _sim(tmp_path, _QUOTED)
    if published_4:
        _snap(out, 4)
    for suffix in files_at_5:
        _snap(out, 5, suffix)
    chosen = run._pick_resume_cfg_past_unusable_snapshot(out / "cfg" / "config_5.cfg", verbose=False)
    assert chosen.name == "config_5.cfg"
    assert run.stepback_record is None


def test_an_absent_snapshot_with_no_marker_steps_back_with_marker_absent(tmp_path):
    """Property: an absent N snapshot with no marker, above an intact one, steps back and says why.

    Class: N's snapshot missing with no ``.part``, ``.killed`` or ``.failed`` marker, above
    an intact one.

    Why required, checkable against the code: an intact earlier snapshot is a sound restart
    point whatever made N's go missing, and the record names the cause ``absent`` so it is
    not confused with a cut-off write.

    What kills it: keying STEP 2 on the presence of a marker rather than on N being unpublished.

    A second correct implementation under which it still passes: one that records the marker
    as ``None`` internally and maps it to ``absent`` when writing the record.
    """
    run, out, _ = _sim(tmp_path, _QUOTED)
    _snap(out, 4)
    chosen = run._pick_resume_cfg_past_unusable_snapshot(out / "cfg" / "config_5.cfg", verbose=False)
    assert chosen.name == "config_4.cfg"
    assert run.stepback_record["marker"] == "absent"
    assert run.stepback_record["reason"] == "absent"


def test_a_failed_marker_at_n_selects_n_and_records_no_step_back(tmp_path):
    """Property: a ``.failed`` at N selects N -- the solver then stops with its write-failed error.

    Class: a snapshot whose write failed, at the newest checkpoint, with a published one
    below.

    Why required, checkable against the code: a failed write is a stop condition, never a
    fall-back to an earlier snapshot -- the writer that failed may fail again.

    What kills it: putting ``.failed`` in the step-back marker set.

    A second correct implementation under which it still passes: one that checks ``.failed``
    before ``PUBLISHED(N)``.
    """
    run, out, _ = _sim(tmp_path, _QUOTED)
    _snap(out, 4)
    _snap(out, 5, ".failed")
    chosen = run._pick_resume_cfg_past_unusable_snapshot(out / "cfg" / "config_5.cfg", verbose=False)
    assert chosen.name == "config_5.cfg"
    assert run.stepback_record is None


def test_a_failed_marker_between_the_target_and_n_selects_it(tmp_path):
    """Property: a ``.failed`` above the step-back target stops the restart there.

    Class: ``.killed`` at N, ``.failed`` at N-1, published N-2.

    Why required, checkable against the code: the stop applies to every ``.failed`` the
    picker reads, so the picker selects N-1 and the solver stops there; stepping over it to
    N-2 would hide the failed write.

    What kills it: computing F only from checkpoint N, or selecting M before testing F.

    A second correct implementation under which it still passes: one that walks down from
    N-1 and returns at the first ``.failed`` or published checkpoint it meets.
    """
    run, out, _ = _sim(tmp_path, _QUOTED, steps=(3, 4, 5))
    _snap(out, 3)
    _snap(out, 4, ".failed")
    _snap(out, 5, ".killed")
    chosen = run._pick_resume_cfg_past_unusable_snapshot(out / "cfg" / "config_5.cfg", verbose=False)
    assert chosen.name == "config_4.cfg"
    assert run.stepback_record is None


def test_a_failed_marker_with_nothing_published_below_selects_it(tmp_path):
    """Property: with no published snapshot, a ``.failed`` below N is still selected.

    Class: ``.failed`` at checkpoint 1, ``.killed`` at 2, nothing published.

    Why required, checkable against the code: without this arm the picker launches at 2 and
    the solver reports a killed write, so the failed write at 1 raises no alarm. Launched at
    1 the solver refuses with ``reason=write-failed``, which the audit reports.

    What kills it: computing F only when M exists.

    A second correct implementation under which it still passes: one that treats "no M" as
    M = 0 and keeps the ``K > M`` comparison.
    """
    run, out, _ = _sim(tmp_path, _QUOTED, steps=(1, 2))
    _snap(out, 1, ".failed")
    _snap(out, 2, ".killed")
    chosen = run._pick_resume_cfg_past_unusable_snapshot(out / "cfg" / "config_2.cfg", verbose=False)
    assert chosen.name == "config_1.cfg"
    assert run.stepback_record is None


@pytest.mark.parametrize(
    ("cfg_text", "steps_back"),
    [
        ('output_folder="out_tritonswmm"\n', True),
        ("output_folder=out_tritonswmm\n", True),
        ('output_folder="/abs/out"\n', False),
        ('output_folder="out_tritonswmm"  # c\n', False),
        ("output_folder=other\n", False),
        ('output_folder="out_tritonswmm"\noutput_folder="other"\n', True),
        ('output_folder="other"\noutput_folder="out_tritonswmm"\n', False),
        ('dem_filename="d"\n', False),
        ('output_folder = "out_tritonswmm"\n', False),
    ],
)
def test_the_output_folder_guard_is_a_two_form_whitelist_on_the_first_line(tmp_path, capsys, cfg_text, steps_back):
    """Property: the step-back happens only when the solver's own output_folder is exactly the default.

    Class: every shape of the first ``output_folder`` line -- the two accepted forms, an
    absolute path, a trailing comment (the solver erases only the ``#`` and keeps the quotes
    unstripped), another name, a second line after an accepted first and the reverse, no line
    at all, and spaces around ``=`` (a different key to the solver).

    Why required, checkable against the code: the picker reads snapshots under
    ``out_tritonswmm/swmm``; stepping back while the solver writes elsewhere would pair the
    solver with state from a directory it does not use. A refusal prints a warning.

    What kills it: stripping trailing comments, trimming the key, reading the last matching
    line, or accepting any value ending in ``out_tritonswmm``.

    A second correct implementation under which it still passes: a regex anchored on
    ``^output_folder=(out_tritonswmm|"out_tritonswmm")$`` applied to the first matching line.
    """
    run, out, _ = _sim(tmp_path, cfg_text)
    _snap(out, 4)
    _snap(out, 5, ".part")
    chosen = run._pick_resume_cfg_past_unusable_snapshot(out / "cfg" / "config_5.cfg", verbose=False)
    assert (chosen.name == "config_4.cfg") is steps_back
    if not steps_back:
        assert "REFUSED" in capsys.readouterr().out


@pytest.mark.parametrize("prune_removes", [True, False])
def test_force_rerun_pending_skips_the_picker(tmp_path, prune_removes):
    """Property: a pending force-rerun never steps back, whatever cfgs survive its prune.

    Class: ``force_rerun_pending`` True on a sim whose cfgs would otherwise step back, with a
    prune that removes every cfg (the real behaviour at step 0) and, as the
    differently-positioned input, a prune that removes none -- which isolates the explicit
    force test from the ordering.

    Why required, checkable against the code: a forced run exists to restart; a step-back
    would resume it instead.

    What kills it: siting the step-back before the force branch (red on the first arm), or
    dropping the ``not _force_pending`` conjunct and relying on the ordering (red on the
    second).

    A second correct implementation under which it still passes: a picker that reads the
    force marker itself and returns its input unchanged.
    """
    run, out, log = _sim(tmp_path, _QUOTED)
    _snap(out, 4)
    _snap(out, 5, ".part")
    log.force_rerun_pending.set(True)

    def _prune(model_type, *, target_step):
        # Stands in for the real prune, which also re-stamps disk-usage accounting this fake
        # tree does not carry.
        removed = list((out / "cfg").glob("*.cfg")) if prune_removes else []
        for p in removed:
            p.unlink()
        return len(removed)

    run.prune_hotstart_cfgs_above_step = _prune
    with pytest.raises(_Stop):
        run.prepare_simulation_command(pickup_where_leftoff=True, verbose=False, model_type="tritonswmm")
    assert run.stepback_record is None
    assert (log.resume_reporting_tsteps.get() or [None])[-1] in (None, 5)


def test_the_triton_only_model_is_unaffected(tmp_path):
    """Property: a TRITON-only resume never consults the coupled snapshot.

    Class: ``model_type="triton"`` with a cut-off coupled snapshot on disk.

    What kills it: applying the picker to every model type.

    A second correct implementation under which it still passes: one that checks the model
    type inside the picker rather than at its call site.
    """
    run, out, log = _sim(tmp_path, _QUOTED)
    triton_cfg_dir = run._scenario.scen_paths.out_triton / "cfg"
    triton_cfg_dir.mkdir(parents=True)
    for k in (4, 5):
        (triton_cfg_dir / f"config_{k}.cfg").write_text(_QUOTED)
    _snap(out, 4)
    _snap(out, 5, ".part")
    with pytest.raises(_Stop):
        run.prepare_simulation_command(pickup_where_leftoff=True, verbose=False, model_type="triton")
    assert log.resume_reporting_tsteps.get() == [5]
    assert run.stepback_record is None


# --------------------------------------------------------------------------------------
# The ledger
# --------------------------------------------------------------------------------------

_ROW_4A = (
    "SWMM snapshot step-back required: checkpoint 5 unusable "
    "[resume-event checkpoint=5 path=step-back reason=killed-write target=4]"
)
_ROW_4B = "SWMM snapshot stop: checkpoint 5 unusable [resume-event checkpoint=5 path=stop reason=write-failed target=4]"
_ROW_4C = (
    "SWMM snapshot step-back impossible: checkpoint 2 unusable, no published snapshot below it "
    "[resume-event checkpoint=2 path=clean-start reason=write-failed target=0]"
)
_SNAP = "SWMM state restored from snapshot to t=240 s; resuming live segment [resume-event checkpoint=4 path=snapshot]"
_ROW_5 = (
    "SWMM snapshot step-back taken: resumed from checkpoint 4 [resume-event checkpoint=4 path=step-back-taken from={n}]"
)


@pytest.mark.parametrize(
    ("line", "path", "checkpoint", "reason", "target"),
    [
        (_ROW_4A, "step-back-required", "5", "killed-write", "4"),
        (_ROW_4B, "stop", "5", "write-failed", "4"),
        (_ROW_4C, "clean-start", "2", "write-failed", "0"),
    ],
)
def test_each_row_4_line_becomes_one_ledger_row(line, path, checkpoint, reason, target):
    """Property: each Row-4 line the solver pins becomes exactly one ledger row of its path.

    Class: the three Row-4 lines (4a, 4b, 4c), byte-for-byte as pinned, integers excepted.

    Why required, checkable against the code: each line is its launch's only event, so with
    no row the attempt would read as evidence lost rather than a declared refusal.

    What kills it: matching the prefixes against a different spelling, or reading ``reason``
    only on replay lines.

    A second correct implementation under which it still passes: one that keys on the
    bracket's ``path=`` token instead of the prefix and maps it the same way.
    """
    (row,) = re_mod.harvest_resume_events(line)
    assert row["path"] == path
    assert row["checkpoint_id"] == checkpoint
    assert row["reason"] == reason
    assert row["target"] == target
    assert row["replay_reason_status"] == "not-applicable"


def test_a_row_5_line_adds_no_row():
    """Property: Row 5 is a corroboration only -- it adds no ledger row beside the snapshot row.

    What kills it: harvesting Row 5 as a second row, which would make one resume read as two.

    A second correct implementation under which it still passes: one that harvests Row 5 into
    a separate list returned beside the rows.
    """
    rows = re_mod.harvest_resume_events(_SNAP + "\n" + _ROW_5.format(n=5))
    assert [r["path"] for r in rows] == ["snapshot"]


_RECORD = {"unusable_step": 5, "marker": ".part", "reason": "killed-write", "used_step": 4}


@pytest.mark.parametrize(
    ("log_text", "corroboration"),
    [
        (_SNAP + "\n" + _ROW_5.format(n=5), None),  # Row 5 agrees
        (_SNAP + "\n" + _ROW_5.format(n=7), "mismatch"),  # Row 5 disagrees
        (_SNAP, None),  # no Row 5 (decision-2 step-back): NOT a mismatch
    ],
)
def test_the_step_back_record_lands_on_the_snapshot_row_and_row_5_corroborates(log_text, corroboration):
    """Property: the picker's record annotates the snapshot row; only a disagreeing Row 5 is a mismatch.

    Class: an exec that stepped back, with a Row 5 that agrees, one that disagrees, and none.

    Why required, checkable against the code: the picker made the decision and is the source;
    Row 5 is re-derived by the solver later and is a check. An absent Row 5 is normal when no
    marker survives above the restored checkpoint.

    What kills it: copying Row 5's ``from=`` into ``stepback_from``, or treating a missing
    Row 5 as a mismatch.

    A second correct implementation under which it still passes: one that compares the two
    checkpoint ids as integers rather than as strings -- the inputs here are integral.
    """
    rows = re_mod.annotate_stepback(
        re_mod.harvest_resume_events(log_text), _RECORD, re_mod.stepback_confirmations(log_text)
    )
    (row,) = rows
    assert (row["stepback_from"], row["stepback_marker"], row["stepback_reason"]) == (5, ".part", "killed-write")
    assert row.get("stepback_corroboration") == corroboration


def test_a_launch_that_did_not_step_back_is_not_annotated():
    """Property: with no picker record the rows are returned unchanged.

    What kills it: annotating every snapshot row whenever Row 5 is present.

    A second correct implementation under which it still passes: one that returns a deep
    copy of the input rather than the same list.
    """
    rows = re_mod.harvest_resume_events(_SNAP + "\n" + _ROW_5.format(n=5))
    assert re_mod.annotate_stepback(rows, None, ["5"]) == rows


def test_the_runner_annotates_the_harvest_with_the_pickers_record():
    """Property: the runner wires the picker's record into the ledger write.

    Class: a FITNESS FUNCTION over the runner's call graph, on the model of
    ``tests/test_resume_event_ledger.py::test_the_runner_reads_the_ledger_in_production``.

    Why required, checkable against the code: every other ledger test here drives
    ``resume_events`` directly, so all stay green against a runner that never annotates.

    What kills it: deleting the ``annotate_stepback`` or ``stepback_confirmations_from_logfile``
    call from the runner's ledger block.

    A second correct implementation under which it still passes: one that routes both
    through a helper in ``resume_events`` called by the runner.
    """
    src = (Path(hhemt.__file__).parent / "run_simulation_runner.py").read_text()
    called = {
        node.func.attr
        for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "annotate_stepback" in called
    assert "stepback_confirmations_from_logfile" in called


# --------------------------------------------------------------------------------------
# The failed-write replay opt-in reaches the cfg a resume launches
# --------------------------------------------------------------------------------------

_KEY_LINE = "swmm_snapshot_failed_replay=1"


def _key_lines(path: Path) -> int:
    return sum(1 for line in path.read_text().splitlines() if line.strip().startswith("swmm_snapshot_failed_replay="))


def _launch(run, capsys=None) -> str:
    """Run ``prepare_simulation_command`` up to the sentinel and return what it printed."""
    with pytest.raises(_Stop):
        run.prepare_simulation_command(pickup_where_leftoff=True, verbose=True, model_type="tritonswmm")
    return capsys.readouterr().out if capsys is not None else ""


@pytest.mark.parametrize(
    ("field", "cfg_has_key", "expected_lines"),
    [
        (True, False, 1),
        (True, True, 1),
        (False, True, 0),
        (False, False, 0),
    ],
)
def test_a_resume_launches_a_cfg_whose_key_line_matches_the_field(tmp_path, field, cfg_has_key, expected_lines):
    """Property: the hotstart cfg a coupled resume launches carries exactly one key line iff the field is True.

    Class: a coupled resume whose chosen hotstart cfg lacks the key with the field on (the
    run was prepared without it), already carries it with the field on, carries it with the
    field off (a hand-added or earlier-opted line), and lacks it with the field off.

    Why required, checkable against the code: TRITON writes each ``config_K.cfg`` as a copy of
    the cfg it was launched with, so a line written only when the base cfg is prepared never
    reaches a resume, and a line left from an earlier launch would keep replaying after the
    field is switched off. The launched cfg is read back off disk after the launch method
    reaches its sentinel.

    What kills it: writing the key only into the base cfg at a clean start, or writing it
    when the field is True and never removing it.

    A second correct implementation under which it still passes: one that launches a fresh
    copy of the hotstart cfg carrying the right line, if that copy is the cfg the launch uses.
    """
    text = _QUOTED + (_KEY_LINE + "\n" if cfg_has_key else "")
    run, out, _ = _sim(tmp_path, text, failed_replay=field)
    _snap(out, 4)
    _snap(out, 5)
    _launch(run)
    launched = out / "cfg" / "config_5.cfg"
    assert _key_lines(launched) == expected_lines
    assert launched.read_text().startswith(_QUOTED)


@pytest.mark.parametrize(
    ("field", "wording", "key_lines"),
    [
        (False, "stops there with its write-failed error", 0),
        (True, "replays the SWMM exchange history there", 1),
    ],
)
def test_the_message_and_the_launched_cfg_follow_the_field_not_the_newest_cfg(
    tmp_path, capsys, field, wording, key_lines
):
    """Property: with a failed write at F below an unusable N, the message and the launched cfg both follow the field.

    Class: ``config_N`` carries the key, ``config_F`` does not, a ``.failed`` marker sits at F
    and N's snapshot is cut off -- driven with the field off and with it on.

    Why required, checkable against the code: the picker returns F whatever the field says,
    so the cfg the solver is handed is F's, and the launch writes the field's value into it.
    A message decided by reading N's cfg would say "replays" for a launch whose F cfg carries
    no key, and "stops" for one that does.

    What kills it: deciding the wording from ``config_N.cfg``'s text, or from anything other
    than the value the launch writes into F's cfg.

    A second correct implementation under which it still passes: one that reads the key back
    from F's cfg after the launch write and words the message from it.
    """
    run, out, _ = _sim(tmp_path, _QUOTED, steps=(3, 4, 5), failed_replay=field)
    (out / "cfg" / "config_5.cfg").write_text(_QUOTED + _KEY_LINE + "\n")
    _snap(out, 4, ".failed")
    _snap(out, 5, ".killed")
    printed = _launch(run, capsys)
    assert wording in printed
    assert "resuming at checkpoint 4" in printed
    assert _key_lines(out / "cfg" / "config_4.cfg") == key_lines


def test_the_completeness_rule_ignores_the_failed_replay_line(tmp_path):
    """Property: cfgs from before and after an opt-in or opt-out resume form one complete set.

    Class: ten cfgs at one length and three one line longer by the key line, the same with
    the three one line SHORTER (the key removed from a run whose earlier cfgs carried it),
    and a final cfg cut short by more than the one key line.

    Why required, checkable against the code: the completeness rule keeps the cfgs whose line
    count is the modal count, and the launch writes or removes one key line in the cfg it
    hands the solver, which TRITON then copies into every later cfg. Counting that line would
    split the set and the modal rule would drop one group as incomplete, so the next resume
    would restart from an earlier checkpoint than the newest.

    What kills it: counting the key line, or ignoring any short cfg (the last case).

    A second correct implementation under which it still passes: one that compares cfgs by
    their line counts after dropping every line the toolkit itself writes at launch.
    """
    base = _QUOTED + "extra_option=1\n"
    run, out, _ = _sim(tmp_path, base, steps=tuple(range(1, 11)))
    for k in (11, 12, 13):
        (out / "cfg" / f"config_{k}.cfg").write_text(base + _KEY_LINE + "\n")
    complete = run._complete_hotstart_cfgs("tritonswmm")
    assert [step for step, _ in complete] == list(range(1, 14))
    assert (
        run._retrieve_hotstart_file_for_incomplete_triton_or_tritonswmm_simulation("tritonswmm").name == "config_13.cfg"
    )

    for k in range(1, 11):
        (out / "cfg" / f"config_{k}.cfg").write_text(base + _KEY_LINE + "\n")
    for k in (11, 12, 13):
        (out / "cfg" / f"config_{k}.cfg").write_text(base)
    assert [step for step, _ in run._complete_hotstart_cfgs("tritonswmm")] == list(range(1, 14))

    (out / "cfg" / "config_14.cfg").write_text(_QUOTED.splitlines()[0] + "\n")
    assert [step for step, _ in run._complete_hotstart_cfgs("tritonswmm")] == list(range(1, 14))
