"""Unit tests for the Phase-2 TRITON pin-enforcement + coupled-resume-validity surface.

Covers (per the plan's Validation Plan):
  (a) ``_verify_tritonswmm_pin`` raises ``ConfigurationError`` when HEAD != pin and is a
      no-op when ``branch_key`` is None;
  (b) ``check_coupled_resume_validity`` — all four arms: pre-fix WARN, unstamped INFO,
      no-coupled-model / no-resume pass, and post-fix-but-replay-marker-absent WARN;
  (c) the interim ``check_coupled_hotstart_resume`` is removed.

Exercised with lightweight ``SimpleNamespace`` stubs + a monkeypatched
``_read_triton_provenance`` (the zarr read), mirroring ``test_resume_status_reporting.py``.
"""

from __future__ import annotations

import logging
import subprocess
from types import SimpleNamespace

import pandas as pd
import pytest

import hhemt.analysis_validation as av
from hhemt.analysis_validation import check_coupled_resume_validity
from hhemt.exceptions import ConfigurationError
from hhemt.model_defects import SHA_EXTBC_GHOST_RING_FIX
from hhemt.system import TRITONSWMM_system


# ---------------------------------------------------------------------------
# (a) _verify_tritonswmm_pin
# ---------------------------------------------------------------------------
def _git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _init_repo(tmp_path):
    repo = tmp_path / "triton"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True, capture_output=True)
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "--allow-empty", "-m", "c1")
    return repo


def _pin_stub(repo, branch_key, sys_yaml):
    return SimpleNamespace(
        cfg_system=SimpleNamespace(
            TRITONSWMM_branch_key=branch_key,
            TRITONSWMM_software_directory=repo,
        ),
        system_config_yaml=sys_yaml,
    )


def test_verify_pin_noop_when_branch_key_none(tmp_path):
    repo = _init_repo(tmp_path)
    stub = _pin_stub(repo, None, tmp_path / "sys.yaml")
    # No exception, no requirement that the clone even exist.
    TRITONSWMM_system._verify_tritonswmm_pin(stub, verbose=False)


def test_verify_pin_passes_when_head_matches(tmp_path):
    repo = _init_repo(tmp_path)
    head = _git(repo, "rev-parse", "HEAD")
    stub = _pin_stub(repo, head, tmp_path / "sys.yaml")
    TRITONSWMM_system._verify_tritonswmm_pin(stub, verbose=False)  # no raise


def test_verify_pin_raises_when_head_differs(tmp_path):
    repo = _init_repo(tmp_path)
    first = _git(repo, "rev-parse", "HEAD")
    # Advance HEAD so the pin (first commit) no longer equals HEAD, but still resolves.
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "--allow-empty", "-m", "c2")
    stub = _pin_stub(repo, first, tmp_path / "sys.yaml")
    with pytest.raises(ConfigurationError):
        TRITONSWMM_system._verify_tritonswmm_pin(stub, verbose=False)


def test_verify_pin_raises_when_pin_unresolvable(tmp_path):
    repo = _init_repo(tmp_path)
    stub = _pin_stub(repo, "0" * 40, tmp_path / "sys.yaml")  # well-formed but absent
    with pytest.raises(ConfigurationError):
        TRITONSWMM_system._verify_tritonswmm_pin(stub, verbose=False)


# ---------------------------------------------------------------------------
# (b) check_coupled_resume_validity — four arms
# ---------------------------------------------------------------------------
#: --------------------------------------------------------------------------
#: Arm-B fixtures. READ THIS BEFORE EDITING.
#:
#: These tests originally CREATED `{scenario_directory}/logs/run_tritonswmm.log` and then
#: asserted on it — the exact path production never writes (the vestigial
#: ScenarioPaths.log_run_* convention). They were green because the fixture manufactured
#: the file the check looked for, so the suite structurally could not detect that the
#: check read NOTHING in production and passed vacuously on all 28 rows of
#: synth_cc_resume. Two properties fix that, and BOTH must be preserved:
#:
#:   1. PRODUCER-BOUND: the fixture places its log by calling the REAL
#:      `run_simulation.model_logfile_for` — the same function the check resolves through
#:      and the runner writes through. If the convention changes, fixture and check move
#:      together; if anyone hand-builds a path again, these tests fail. Stubbing the
#:      resolver instead would re-commit the original defect one level up.
#:   2. DECOY: every Arm-B test also writes a file at the OLD dead path whose marker
#:      content is the INVERSE of the real log's. A check that reads the decoy returns the
#:      inverted verdict, so each test FAILS against pre-fix code. Do NOT delete the decoys
#:      as "unused fixtures" — they are the anti-regression, and without them three of
#:      these tests pass vacuously against a check that reads nothing.
#: --------------------------------------------------------------------------
#:
#: THIRD PREDICATE (per-exec resume discriminator). Every "this exec resumed" log must
#: carry _CKPT, because the check's SCOPE gate keys on it: n_resumes is cumulative and
#: never reset, so only the in-log [OK] marker proves the LAST exec resumed. A fixture
#: that omits _CKPT is asserting a FRESH exec — which is a real, tested case
#: (test_postfix_fresh_last_exec_is_out_of_scope), not an oversight. Literal shapes are
#: taken verbatim from live logs (synth_cc_resume at the pin, 2026-07-16):
#:     [..] Reading checkpoint files
#:     [OK] Checkpoint files read
#:     [..] SWMM exchange history replayed to t=3000 s (11435 steps); resuming live segment

_CKPT_ATTEMPT = "[..] Reading checkpoint files\n"
_CKPT = _CKPT_ATTEMPT + "[OK] Checkpoint files read\n"
_REPLAY = "[..] SWMM exchange history replayed to t=3600 s (12 steps); resuming live segment\n"
_ENDS = "Simulation ends\n"

# Producing shas that stand in for the retired per-defect booleans. Each is a REAL commit whose
# ancestry the registry resolved at authoring time, so these exercise the production read path
# (no clone, cached sets) rather than a path production never takes.
_SHA_PRE_REPLAY = "15eb18a5d25afe5da295cb4b559a62669dbe5bc3"  # replay PRESENT  (Arm A)
_SHA_PRE_SCATTER = "b3820a448f304b3f732f4b6fac5564adf86ac333"  # replay absent, scatter PRESENT (Arm C)
_SHA_POST_ALL = "9db367ddc79f86c7f708686d1dd805dc992fb0a4"  # replay + scatter both absent (Arm B)


def _analysis_stub(*, coupled=True, sensitivity=False, df=None, simlog_dir=None):
    return SimpleNamespace(
        _system=SimpleNamespace(
            cfg_system=SimpleNamespace(toggle_tritonswmm_model=coupled),
        ),
        cfg_analysis=SimpleNamespace(toggle_sensitivity_analysis=sensitivity, is_experiment_member=False),
        analysis_paths=SimpleNamespace(
            analysis_datatree_zarr=None,
            sensitivity_datatree_zarr=None,
            simlog_directory=simlog_dir,
        ),
        df_status=df,
    )


def _resumed_df(scenario_directory="", event_iloc=0, member_id=None, model_type="tritonswmm"):
    row = {
        "model_type": model_type,
        "n_resumes": 2,
        "scenario_directory": scenario_directory,
        "event_iloc": event_iloc,
    }
    if member_id is not None:
        row["sa_id"] = member_id
    return pd.DataFrame([row])


def _write_real_log(analysis, event_iloc, text, model_type="tritonswmm"):
    """Place the log at the path the PRODUCER writes — resolved by the real convention.

    `model_type` mirrors `_resumed_df`'s parameter of the same name and exists so the two
    fixtures cannot disagree: the row declares which model it is, and the log must be
    written where THAT model's log lives. A mismatched pair produces a test that passes
    for the wrong reason -- a pure-TRITON row whose fixture sits at the coupled path is
    EXAMINED only because the check's log read was itself hardcoded (see VMS-9D), so the
    test goes green while the defect it pins is still present. Default preserves all
    existing callers, every one of which passes three positional arguments.
    """
    from hhemt.run_simulation import model_logfile_for

    p = model_logfile_for(analysis, event_iloc, model_type)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


def _write_dead_path_decoy(scen_dir, text):
    """Write the OLD hand-built path — the check MUST NOT read this.

    `{scenario_directory}/logs/run_tritonswmm.log` IS `ScenarioPaths.log_run_tritonswmm`
    (scenario_directory == sim_folder; logs_dir == sim_folder/"logs"), the field nothing
    writes. Content here is always the INVERSE of the real log's, so a regression back to
    the hand-built path flips the verdict and fails the test.
    """
    p = scen_dir / "logs" / "run_tritonswmm.log"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


def test_coupled_off_without_resumes_is_na(monkeypatch):
    """Coupled-off is N/A for a POPULATION reason, not a toggle reason.

    Retired property: `test_coupled_off_is_na` asserted "not enabled" in the summary,
    pinning the old toggle gate that returned N/A for EVERY pure-TRITON analysis. The
    widening deliberately falsifies that universal -- a coupled-off arm WITH resumes is
    now evaluated (see the companion test below). What survives is the narrower true
    property: with nothing resumed there is nothing to verify.
    """
    monkeypatch.setattr(av, "_read_triton_provenance", lambda a: SHA_EXTBC_GHOST_RING_FIX)
    res = check_coupled_resume_validity(_analysis_stub(coupled=False))
    assert res.passed is True
    assert res.applicable is False


def test_coupled_off_with_resumes_is_evaluated(monkeypatch, tmp_path):
    """THE arm that pins the widening: a pure-TRITON arm that resumed is EVALUATED.

    Red before VMS-9/9B/9D (the toggle gate returned N/A on `coupled=False` before any
    registry read), and red after VMS-9B alone (the row is selected but its log read is
    keyed on a hardcoded "tritonswmm", so it counts INDETERMINATE and examined stays 0).
    Green only once the log path follows the row's model_type.

    `model_type="triton"` is passed to BOTH fixtures, once each. That pairing is the
    coherence property VMS-9F exists to buy: a row declaring one model with its log
    written at another's path is EXAMINED only via the very hardcoding VMS-9D removes,
    i.e. a test that passes for the wrong reason.
    """
    monkeypatch.setattr(av, "_read_triton_provenance", lambda a: SHA_EXTBC_GHOST_RING_FIX)
    a = _analysis_stub(
        coupled=False,
        df=_resumed_df(str(tmp_path), model_type="triton"),
        simlog_dir=tmp_path / "logs" / "sims",
    )
    _write_real_log(a, 0, "start\n" + _CKPT + _REPLAY + _ENDS, model_type="triton")
    res = check_coupled_resume_validity(a)
    assert res.applicable is True, (
        "a pure-TRITON arm that resumed must be EVALUATED, not N/A; "
        f"got applicable={res.applicable!r} summary={res.summary!r}"
    )


def test_unstamped_is_indeterminate(monkeypatch):
    monkeypatch.setattr(av, "_read_triton_provenance", lambda a: None)
    res = check_coupled_resume_validity(_analysis_stub(df=_resumed_df()))
    assert res.passed is True
    assert "unknown" in res.summary
    assert res.details == []


def test_prefix_with_resume_warns(monkeypatch):
    monkeypatch.setattr(av, "_read_triton_provenance", lambda a: _SHA_PRE_REPLAY)
    res = check_coupled_resume_validity(_analysis_stub(df=_resumed_df()))
    assert res.passed is False
    assert len(res.details) == 1
    assert "PRE-FIX" in res.details[0]["detail"]


def test_prefix_without_resume_is_na(monkeypatch):
    """Pre-fix TRITON but NOTHING resumed: examined == 0, so the cell is N/A, not a PASS.

    Renamed from `test_prefix_without_resume_passes` under EW-2b. `passed is True` alone no
    longer describes the outcome — it is true of the N/A return as well — so the applicable
    assertion is what keeps this test discriminating.
    """
    monkeypatch.setattr(av, "_read_triton_provenance", lambda a: _SHA_PRE_REPLAY)
    empty = pd.DataFrame([{"model_type": "tritonswmm", "n_resumes": 0, "scenario_directory": ""}])
    res = check_coupled_resume_validity(_analysis_stub(df=empty))
    assert res.passed is True
    assert res.applicable is False, res.summary


def test_postfix_missing_replay_marker_warns(monkeypatch, tmp_path):
    """Resumed + complete last exec, no replay marker -> WARN. The only WARN case.

    FAILS PRE-FIX: today the check reads the decoy (marker PRESENT) -> 0 details ->
    passed=True, so `assert res.passed is False` fails.
    """
    monkeypatch.setattr(av, "_read_triton_provenance", lambda a: _SHA_POST_ALL)
    scen = tmp_path / "sim_0"
    a = _analysis_stub(df=_resumed_df(str(scen)), simlog_dir=tmp_path / "logs" / "sims")
    _write_real_log(a, 0, _CKPT + _ENDS)
    _write_dead_path_decoy(scen, _CKPT + _REPLAY + _ENDS)  # inverse verdict
    res = check_coupled_resume_validity(a)
    assert res.passed is False
    assert len(res.details) == 1
    assert "exchange-replay marker is ABSENT" in res.details[0]["detail"]
    assert "1 resumed coupled sim(s) examined" in res.summary


def test_postfix_with_replay_marker_passes(monkeypatch, tmp_path):
    """Resumed + complete + replayed -> PASS, and the denominator proves the check actually
    examined the sim rather than skipping it.

    FAILS PRE-FIX: today the check reads the decoy (marker ABSENT) -> 1 detail ->
    passed=False, so `assert res.passed is True` fails. The denominator assertion is the
    second lock: a vacuous pass reports "0 ... examined".
    """
    monkeypatch.setattr(av, "_read_triton_provenance", lambda a: _SHA_POST_ALL)
    scen = tmp_path / "sim_0"
    a = _analysis_stub(df=_resumed_df(str(scen)), simlog_dir=tmp_path / "logs" / "sims")
    _write_real_log(a, 0, "start\n" + _CKPT + _REPLAY + _ENDS)
    _write_dead_path_decoy(scen, _CKPT + _ENDS)  # inverse verdict
    res = check_coupled_resume_validity(a)
    assert res.passed is True
    assert res.details == []
    assert "1 resumed coupled sim(s) examined" in res.summary


def test_postfix_unreadable_log_is_indeterminate(monkeypatch, tmp_path):
    """No log at the producer path -> INDETERMINATE, counted, never a warn.

    FAILS PRE-FIX: today the check reads the decoy (marker ABSENT) -> 1 detail ->
    passed=False, so `assert res.passed is True` fails.
    """
    monkeypatch.setattr(av, "_read_triton_provenance", lambda a: _SHA_POST_ALL)
    scen = tmp_path / "sim_0"
    a = _analysis_stub(df=_resumed_df(str(scen)), simlog_dir=tmp_path / "logs" / "sims")
    # No _write_real_log — the producer path is absent.
    _write_dead_path_decoy(scen, _CKPT + _ENDS)  # inverse verdict
    res = check_coupled_resume_validity(a)
    assert res.passed is True
    assert res.details == []
    assert "0 resumed coupled sim(s) examined" in res.summary
    assert "1 INDETERMINATE" in res.summary


def test_postfix_incomplete_last_exec_is_indeterminate(monkeypatch, tmp_path):
    """The COMPLETION GATE. A resumed last exec walltime-killed BEFORE its replay carries
    the checkpoint marker but neither the replay nor the completion marker; warning on it
    would conflate a benign kill with the rank-0 silent-skip.

    FAILS PRE-FIX: today the check reads the decoy (complete, no replay marker) -> 1 detail
    -> passed=False, so `assert res.passed is True` fails. Pre-fix there is no gate at all.
    """
    monkeypatch.setattr(av, "_read_triton_provenance", lambda a: _SHA_POST_ALL)
    scen = tmp_path / "sim_0"
    a = _analysis_stub(df=_resumed_df(str(scen)), simlog_dir=tmp_path / "logs" / "sims")
    _write_real_log(a, 0, _CKPT + "running\n")  # resumed, then killed: no replay, no ends
    _write_dead_path_decoy(scen, _CKPT + _ENDS)  # inverse verdict: would warn
    res = check_coupled_resume_validity(a)
    assert res.passed is True
    assert res.details == []
    assert "1 INDETERMINATE" in res.summary


def test_postfix_fresh_last_exec_is_out_of_scope(monkeypatch, tmp_path):
    """The SCOPE GATE, and the reason it exists. n_resumes is CUMULATIVE and never reset,
    so a sim that resumed, lost its checkpoints (clear_raw / delete / force-rerun), and then
    ran FRESH to completion still has n_resumes>=1 and legitimately carries NO replay
    marker. Its data is VALID. Warning on it would be a false positive on good data.

    A fresh exec is OUT OF SCOPE — not indeterminate (the replay question never applied) and
    not examined (we tested nothing). The denominator must say so in its own words.

    FAILS PRE-FIX: today there is no scope gate at all, so this row is complete-with-no-
    replay-marker -> 1 detail -> passed=False, and `assert res.passed is True` fails. It
    fails against the PRE-ADDENDUM spec too, which had no scope gate either — this test is
    what the live checkpoint-marker evidence bought.
    """
    monkeypatch.setattr(av, "_read_triton_provenance", lambda a: _SHA_POST_ALL)
    scen = tmp_path / "sim_0"
    a = _analysis_stub(df=_resumed_df(str(scen)), simlog_dir=tmp_path / "logs" / "sims")
    _write_real_log(a, 0, "start\n" + _ENDS)  # NO _CKPT: this exec ran fresh
    _write_dead_path_decoy(scen, _CKPT + _REPLAY + _ENDS)
    res = check_coupled_resume_validity(a)
    assert res.passed is True
    assert res.details == []
    assert "0 resumed coupled sim(s) examined" in res.summary
    assert "1 out of scope" in res.summary
    assert "INDETERMINATE" not in res.summary  # out-of-scope is NOT indeterminate


def test_postfix_partial_checkpoint_read_is_indeterminate(monkeypatch, tmp_path):
    """The ANCHOR CHOICE. A checkpoint read that STARTS and does not complete, after which
    the run reaches t=end, must NOT warn: the replay reads the exchange history FROM the
    checkpoint set, so "the replay should have engaged" is unwarranted when the read never
    took. Anchoring scope on the [..] attempt form instead of the [OK] completion form would
    make this a FALSE WARN on fresh-and-complete data.

    It is INDETERMINATE rather than out-of-scope: a half-read checkpoint set is a real
    anomaly, just not the rank-0 silent-skip this arm names.

    FAILS PRE-FIX: no scope gate -> complete + no replay marker -> 1 detail -> passed=False.
    """
    monkeypatch.setattr(av, "_read_triton_provenance", lambda a: _SHA_POST_ALL)
    scen = tmp_path / "sim_0"
    a = _analysis_stub(df=_resumed_df(str(scen)), simlog_dir=tmp_path / "logs" / "sims")
    _write_real_log(a, 0, _CKPT_ATTEMPT + _ENDS)  # attempt WITHOUT the [OK] completion
    _write_dead_path_decoy(scen, _CKPT + _REPLAY + _ENDS)
    res = check_coupled_resume_validity(a)
    assert res.passed is True
    assert res.details == []
    assert "1 INDETERMINATE" in res.summary
    assert "out of scope" not in res.summary


def test_postfix_sensitivity_master_resolves_per_sub(monkeypatch, tmp_path):
    """The SENSITIVITY BRANCH. A master's df_status carries member_id and its members'
    logs live under {master}/logs/sims via the is_experiment_member branch of the convention.

    FAILS PRE-FIX: today the check reads the decoy (marker PRESENT) -> passed=True, so
    `assert res.passed is False` fails.
    """
    monkeypatch.setattr(av, "_read_triton_provenance", lambda a: _SHA_POST_ALL)
    scen = tmp_path / "sim_0"
    master_dir = tmp_path / "master"
    sub = SimpleNamespace(
        cfg_analysis=SimpleNamespace(
            is_experiment_member=True,
            analysis_id="member_0",
            experiment_cfg_yaml=master_dir / "cfg_analysis.yaml",
        ),
        # model_logfile_for now derives the master log dir from the sub's OWN analysis_dir
        # (`.parent.parent`), so the stub must carry the real two-level layout.
        analysis_paths=SimpleNamespace(
            simlog_directory=tmp_path / "unused",
            analysis_dir=master_dir / "members" / "member_0",
        ),
    )
    master = _analysis_stub(
        sensitivity=True,
        df=_resumed_df(str(scen), member_id="member_0"),
        simlog_dir=tmp_path / "unused",
    )
    master.sensitivity = SimpleNamespace(analyses={"member_0": sub})
    master.cfg_analysis.toggle_sensitivity_analysis = True
    _write_real_log(sub, 0, _CKPT + _ENDS)  # resumed, complete, no replay marker -> WARN
    assert (master_dir / "logs" / "sims" / "model_tritonswmm_member_0_evt0.log").exists()
    _write_dead_path_decoy(scen, _CKPT + _REPLAY + _ENDS)  # inverse verdict
    res = check_coupled_resume_validity(master)
    assert res.passed is False
    assert len(res.details) == 1


def test_model_logfile_method_delegates_to_free_function():
    """THE CONVENTION LOCK. The producer-side method and the detector must resolve ONE
    function. Two independent expressions of this convention is exactly what made the
    replay arm inert.

    FAILS PRE-FIX: `model_logfile_for` does not exist -> ImportError.
    """
    from pathlib import Path as _P

    from hhemt.run_simulation import TRITONSWMM_run, model_logfile_for

    a = SimpleNamespace(
        analysis_paths=SimpleNamespace(simlog_directory=_P("/x/logs/sims")),
        cfg_analysis=SimpleNamespace(is_experiment_member=False),
    )
    run = SimpleNamespace(_analysis=a, _scenario=SimpleNamespace(event_iloc=7))
    assert TRITONSWMM_run._analysis_level_model_logfile(run, "tritonswmm") == model_logfile_for(a, 7, "tritonswmm")
    assert model_logfile_for(a, 7, "tritonswmm").name == "model_tritonswmm_evt7.log"


def test_sub_model_log_lives_under_experiment_dir_not_config_dir(tmp_path):
    """THE WIPE-COVERAGE INVARIANT. A member's model runtime log MUST land inside the
    MASTER's analysis_dir, so `run(from_scratch=True)`'s fast_rmtree(analysis_dir) removes it
    along with the outputs it describes.

    FAILS PRE-FIX: the old form derived the dir from experiment_cfg_yaml.parent, so with
    the config placed outside analysis_dir (the synth case-builder's platformdirs layout, and
    the ordinary production layout where the user's config is not at the analysis root) the
    log landed outside the wipe. Empirically, that stranded 28/28 week-stale "Simulation ends"
    logs and made every sim of the pure-TRITON resume arm skip execution.
    """
    from hhemt.run_simulation import model_logfile_for

    master_dir = tmp_path / "scratch" / "exp"
    config_dir = tmp_path / "cache" / "exp"  # deliberately NOT under master_dir
    sub = SimpleNamespace(
        cfg_analysis=SimpleNamespace(
            is_experiment_member=True,
            analysis_id="member_gpu_2_r1",
            experiment_cfg_yaml=config_dir / "analysis_config.yaml",
        ),
        analysis_paths=SimpleNamespace(
            simlog_directory=master_dir / "members" / "member_gpu_2_r1" / "logs" / "sims",
            analysis_dir=master_dir / "members" / "member_gpu_2_r1",
        ),
    )
    p = model_logfile_for(sub, 0, "triton")
    assert p == master_dir / "logs" / "sims" / "model_triton_member_gpu_2_r1_evt0.log"
    assert config_dir not in p.parents, "model log must not be anchored to the config dir"


# ---------------------------------------------------------------------------
# (c) interim check removed
# ---------------------------------------------------------------------------
def test_interim_check_removed():
    assert not hasattr(av, "check_coupled_hotstart_resume")


# ---------------------------------------------------------------------------
# Cross-process carriage seam (R5): system-log -> stamp -> zarr root -> reader.
# ---------------------------------------------------------------------------
def test_provenance_stamp_read_roundtrip(tmp_path):
    import xarray as xr

    from hhemt.log import TRITONSWMM_system_log
    from hhemt.processing_analysis import _stamp_triton_provenance
    from hhemt.utils import write_datatree_zarr

    # A real system log carrying compile-time provenance (the cross-process carrier).
    sys_log = TRITONSWMM_system_log(logfile=tmp_path / "system_log.json")
    sys_log.triton_head_sha.set("cafebabecafebabecafebabecafebabecafebabe")
    sys_log.write()

    zarr_path = tmp_path / "analysis_datatree.zarr"
    analysis = SimpleNamespace(
        _system=SimpleNamespace(log=sys_log),
        cfg_analysis=SimpleNamespace(toggle_sensitivity_analysis=False),
        analysis_paths=SimpleNamespace(analysis_datatree_zarr=zarr_path, sensitivity_datatree_zarr=None),
    )

    tree = xr.DataTree.from_dict({"/": xr.Dataset(attrs={"analysis_id": "demo"})})
    _stamp_triton_provenance(tree, analysis)
    assert tree.attrs["triton_producing_sha"].startswith("cafebabe")
    # The two per-defect boolean attrs are RETIRED; the sha is the whole stamp now.
    assert "triton_has_coupled_resume_fix" not in tree.attrs
    assert "triton_has_swmm_depth_scatter_fix" not in tree.attrs

    write_datatree_zarr(tree, zarr_path)

    sha = av._read_triton_provenance(analysis)
    assert sha == "cafebabecafebabecafebabecafebabecafebabe"


def test_provenance_stamp_graceful_absent_when_unstamped(tmp_path):
    import xarray as xr

    from hhemt.log import TRITONSWMM_system_log
    from hhemt.processing_analysis import _stamp_triton_provenance

    sys_log = TRITONSWMM_system_log(logfile=tmp_path / "system_log.json")  # fields unset
    analysis = SimpleNamespace(_system=SimpleNamespace(log=sys_log))
    tree = xr.DataTree.from_dict({"/": xr.Dataset()})
    _stamp_triton_provenance(tree, analysis)
    # Unset provenance -> attrs omitted (graceful-absent -> INDETERMINATE downstream).
    assert "triton_producing_sha" not in tree.attrs
    assert "triton_has_coupled_resume_fix" not in tree.attrs


# --- _read_triton_provenance: the reader must not be inert -----------------------
#
# REGRESSION (Rivanna synth_cc_resume, 2026-07-15): the reader opened the tree with
# chunks="auto", which raises NotImplementedError ("Can not use auto rechunking with
# object dtype") on any tree carrying an object-dtype variable. The bare
# `except Exception: return None, None` turned that into a silent INDETERMINATE, so
# check_coupled_resume_validity's pre-fix warning was PERMANENTLY DISABLED on every
# experiment and passed vacuously. The stamped tree was correct on disk
# (triton_producing_sha=3a832f7d..., triton_has_coupled_resume_fix=True).


def _fake_analysis_with_zarr(zarr_path):
    return SimpleNamespace(
        cfg_analysis=SimpleNamespace(toggle_sensitivity_analysis=True),
        analysis_paths=SimpleNamespace(sensitivity_datatree_zarr=zarr_path),
    )


def test_provenance_reader_does_not_use_auto_rechunking(tmp_path, monkeypatch):
    """The reader consumes ONLY root attrs, so it must open WITHOUT auto-rechunking.
    Re-introducing chunks="auto" makes an object-dtype tree unreadable and silently
    renders the whole coupled-resume check inert -> this test fails."""
    import xarray as xr

    zarr_path = tmp_path / "sensitivity_datatree.zarr"
    zarr_path.mkdir()

    class _FakeTree:
        attrs = {
            "triton_producing_sha": "3a832f7d5eedd96aaee0dfe9181da5774adfb9f4",
        }

    def _fake_open_datatree(path, **kwargs):
        if kwargs.get("chunks") == "auto":
            raise NotImplementedError("Can not use auto rechunking with object dtype")
        return _FakeTree()

    monkeypatch.setattr(xr, "open_datatree", _fake_open_datatree)

    sha = av._read_triton_provenance(_fake_analysis_with_zarr(zarr_path))
    assert sha == "3a832f7d5eedd96aaee0dfe9181da5774adfb9f4"


def test_provenance_reader_warns_but_never_raises_on_unexpected_failure(tmp_path, monkeypatch, caplog):
    """An UNEXPECTED reader exception must not abort validation (the never-raises
    contract) but must NOT be swallowed silently either — a silently-inert check is
    exactly how the chunks="auto" defect stayed hidden."""
    import xarray as xr

    zarr_path = tmp_path / "sensitivity_datatree.zarr"
    zarr_path.mkdir()

    def _boom(path, **kwargs):
        raise RuntimeError("unexpected zarr failure")

    monkeypatch.setattr(xr, "open_datatree", _boom)

    with caplog.at_level(logging.WARNING, logger="hhemt.analysis_validation"):
        sha = av._read_triton_provenance(_fake_analysis_with_zarr(zarr_path))

    assert sha is None  # graceful-absent, never raises
    assert "INDETERMINATE" in caplog.text
    assert "RuntimeError" in caplog.text


def test_provenance_reader_is_quiet_on_genuinely_absent_tree(tmp_path, monkeypatch, caplog):
    """A genuinely absent/unreadable tree (pre-provenance or off-checkout) stays on
    the documented quiet graceful-absent path — no warning noise."""
    import xarray as xr

    zarr_path = tmp_path / "sensitivity_datatree.zarr"
    zarr_path.mkdir()

    def _absent(path, **kwargs):
        raise FileNotFoundError("no such tree")

    monkeypatch.setattr(xr, "open_datatree", _absent)

    with caplog.at_level(logging.WARNING, logger="hhemt.analysis_validation"):
        sha = av._read_triton_provenance(_fake_analysis_with_zarr(zarr_path))

    assert sha is None
    assert caplog.text == ""


# --- Q4: durable per-sub replay-evidence stamp fallback (Arm B, log purged) -------
#
# When the "w"-mode last-exec log is gone (R7 purge / cache clear), Arm B falls back to the
# `coupled_resume_replay_evidence` root attr stamped at consolidation time
# (_stamp_coupled_resume_evidence). A stamped resumed+completed sub is EXAMINED (positive if
# replayed, a real WARN if not) instead of the vacuous INDETERMINATE the purged log would force.


def test_postfix_unreadable_log_durable_stamp_replayed_passes(monkeypatch, tmp_path):
    """Q4: log purged but the durable stamp shows resumed+completed+replayed -> EXAMINED + PASS
    (not a vacuous INDETERMINATE). This is exactly the synth_cc_resume situation the R7 purge
    created, made durable so a future run never depends on a cache-resident last-exec log."""
    import json

    import xarray as xr

    from hhemt.utils import write_datatree_zarr

    monkeypatch.setattr(av, "_read_triton_provenance", lambda a: _SHA_POST_ALL)
    scen = tmp_path / "sim_0"
    a = _analysis_stub(df=_resumed_df(str(scen)), simlog_dir=tmp_path / "logs" / "sims")
    # No _write_real_log — the producer log path is ABSENT (purged).
    zarr_path = tmp_path / "analysis_datatree.zarr"
    a.analysis_paths.analysis_datatree_zarr = zarr_path
    evidence = {str(scen): {"resumed": True, "completed": True, "replayed": True}}
    tree = xr.DataTree.from_dict({"/": xr.Dataset(attrs={"coupled_resume_replay_evidence": json.dumps(evidence)})})
    write_datatree_zarr(tree, zarr_path)
    res = check_coupled_resume_validity(a)
    assert res.passed is True
    assert res.details == []
    assert "1 resumed coupled sim(s) examined" in res.summary


def test_postfix_unreadable_log_durable_stamp_not_replayed_warns(monkeypatch, tmp_path):
    """Q4: log purged but the durable stamp shows resumed+completed but NOT replayed -> WARN.
    The durable evidence catches a silently-skipped replay the purged log could no longer prove."""
    import json

    import xarray as xr

    from hhemt.utils import write_datatree_zarr

    monkeypatch.setattr(av, "_read_triton_provenance", lambda a: _SHA_POST_ALL)
    scen = tmp_path / "sim_0"
    a = _analysis_stub(df=_resumed_df(str(scen)), simlog_dir=tmp_path / "logs" / "sims")
    zarr_path = tmp_path / "analysis_datatree.zarr"
    a.analysis_paths.analysis_datatree_zarr = zarr_path
    evidence = {str(scen): {"resumed": True, "completed": True, "replayed": False}}
    tree = xr.DataTree.from_dict({"/": xr.Dataset(attrs={"coupled_resume_replay_evidence": json.dumps(evidence)})})
    write_datatree_zarr(tree, zarr_path)
    res = check_coupled_resume_validity(a)
    assert res.passed is False
    assert len(res.details) == 1
    assert "durable replay-evidence stamp shows the replay did NOT engage" in res.details[0]["detail"]
    assert "1 resumed coupled sim(s) examined" in res.summary


def test_resume_schedule_honored_warns_on_short_coupled_replay(tmp_path):
    """Arm A: a coupled resume whose replay_t landed BELOW schedule[-1]*interval
    (replay_matches_schedule=False in the durable stamp) is a warn. Disjoint from
    check_coupled_resume_validity (which fires on marker ABSENCE): here the marker is
    present but mis-positioned.
    """
    import json

    import xarray as xr

    from hhemt.analysis_validation import check_resume_schedule_honored

    zpath = tmp_path / "analysis_datatree.zarr"
    ev = {
        "member_0": {
            "resumed": True,
            "completed": True,
            "replayed": True,
            "replay_t": 300.0,
            "expected_replay_t": 600.0,
            "replay_matches_schedule": False,
        }
    }
    ds = xr.Dataset({"placeholder": (("a",), [1])})
    ds.attrs["coupled_resume_replay_evidence"] = json.dumps(ev, sort_keys=True)
    ds.to_zarr(zpath, mode="w")

    analysis = SimpleNamespace(
        _system=SimpleNamespace(cfg_system=SimpleNamespace(toggle_tritonswmm_model=True, toggle_triton_model=False)),
        cfg_analysis=SimpleNamespace(toggle_sensitivity_analysis=False),
        analysis_paths=SimpleNamespace(analysis_datatree_zarr=zpath, sensitivity_datatree_zarr=None),
    )
    res = check_resume_schedule_honored(analysis)
    assert res.passed is False
    assert res.name == "Resume schedule honored"
    assert any("did not reach the last scheduled interruption" in d["detail"] for d in res.details)
    assert "1 resumed sim(s) schedule-verified" in res.summary  # disclosed denominator


def _triton_arm_b_stub(df):
    """Pure-TRITON (Arm B) analysis stub: no sensitivity, so _iter_members_or_self
    yields (None, analysis) and the schedule is read off analysis.cfg_analysis."""
    return SimpleNamespace(
        _system=SimpleNamespace(cfg_system=SimpleNamespace(toggle_tritonswmm_model=False, toggle_triton_model=True)),
        cfg_analysis=SimpleNamespace(
            toggle_sensitivity_analysis=False,
            resume_interruption_schedule=(36, 72, 108),
        ),
        analysis_paths=SimpleNamespace(analysis_datatree_zarr=None, sensitivity_datatree_zarr=None),
        df_status=df,
    )


def test_resume_schedule_honored_surfaces_unverifiable_when_no_realized_boundaries():
    """Arm B: a resumed pure-TRITON sim whose COUNT matches the schedule but which
    recorded NO realized boundaries must be surfaced as UNVERIFIABLE, not passed.

    This is the re-sim silent-invalidation shape: re-running the resume arm over an
    existing analysis dir without start_from_scratch leaves the prior campaign's
    n_resumes in place, so no interruption arms (the gate is _n_done < len(schedule)),
    the count check passes, and a tree with no realized-boundary list has nothing to
    compare — every detector green over data the deterministic prune never touched.
    Absence of evidence must not read as evidence of correctness.
    """
    from hhemt.analysis_validation import check_resume_schedule_honored

    # n_resumes == len(schedule) (count check passes) and NO resume_reporting_tsteps.
    df = pd.DataFrame([{"model_type": "triton", "n_resumes": 3, "scenario_directory": "sim_0"}])
    res = check_resume_schedule_honored(_triton_arm_b_stub(df))

    assert any(
        "CANNOT BE VERIFIED" in d["detail"] for d in res.details
    ), "a resumed sim with no realized boundaries must be surfaced, not silently passed"
    assert any(
        "start_from_scratch" in d["detail"] for d in res.details
    ), "the detail must name the operational cause a reader can act on"


def test_resume_schedule_honored_is_quiet_when_realized_boundaries_match():
    """Regression guard for the fix above: a sim that DID record realized boundaries
    matching the configured schedule is the verified-good case and must emit no detail."""
    from hhemt.analysis_validation import check_resume_schedule_honored

    df = pd.DataFrame(
        [
            {
                "model_type": "triton",
                "n_resumes": 3,
                "scenario_directory": "sim_0",
                "resume_reporting_tsteps": [36, 72, 108],
            }
        ]
    )
    res = check_resume_schedule_honored(_triton_arm_b_stub(df))
    assert res.details == [], f"verified-good sim should emit no detail, got {res.details}"


# --- Arm C (S8): replayed SWMM node depths are never scattered to the ranks. ----------
# The three-way fixture below is the whole contract: Arm C must fire ONLY on the post-fix
# branch, must be suppressed where Arm A already fires (so a doubly-affected sim receives
# one remedy rather than two contradictory ones), and must go quiet once the upstream
# scatter fix lands. All three share a replay-marker-PRESENT log, which is what isolates
# Arm C from Arm B — the replay ran; its result was then discarded.


def test_armc_suppressed_when_arm_a_fires(monkeypatch, tmp_path):
    """has_fix=False -> Arm A fires and Arm C does NOT, so exactly one detail per scenario.

    Anchored on the DETAIL COUNT rather than on message wording: a count of 1 is true in
    both the pre-fix and post-fix worlds, whereas asserting on Arm A's wording alone would
    pass even if Arm C also appended a second, contradictory row.
    """
    monkeypatch.setattr(av, "_read_triton_provenance", lambda a: _SHA_PRE_REPLAY)
    scen = tmp_path / "sim_0"
    a = _analysis_stub(df=_resumed_df(str(scen)), simlog_dir=tmp_path / "logs" / "sims")
    _write_real_log(a, 0, "start\n" + _CKPT + _REPLAY + _ENDS)
    res = check_coupled_resume_validity(a)
    assert res.passed is False
    assert len(res.details) == 1, "Arm C must not append a second detail on the pre-fix branch"
    assert "PRE-FIX" in res.details[0]["detail"]


def test_armc_fires_on_postfix_replayed_without_scatter(monkeypatch, tmp_path):
    """has_fix=True, replay marker PRESENT, scatter absent -> Arm C fires.

    This is the case Arms A and B are both silent on by construction, which is why the
    fixture writes a complete, replayed log: Arm B passes on it, so a detail here can only
    have come from Arm C.
    """
    monkeypatch.setattr(av, "_read_triton_provenance", lambda a: _SHA_PRE_SCATTER)
    scen = tmp_path / "sim_0"
    a = _analysis_stub(df=_resumed_df(str(scen)), simlog_dir=tmp_path / "logs" / "sims")
    _write_real_log(a, 0, "start\n" + _CKPT + _REPLAY + _ENDS)
    res = check_coupled_resume_validity(a)
    assert res.passed is False
    assert len(res.details) == 1
    assert "lacks the SWMM node-depth SCATTER" in res.details[0]["detail"]
    assert "max_wlevel_m / H / MH from this sim are INVALID" in res.details[0]["detail"]
    assert "lacking the SWMM node-depth scatter" in res.summary


def test_armc_quiet_when_scatter_fix_present(monkeypatch, tmp_path):
    """has_fix=True, scatter=True -> the arm goes quiet with no other code change.

    This is the forward-compatibility assertion: when the upstream fix lands and the pin
    constant is set, the ancestry stamp starts reporting True and this arm stops firing.
    """
    monkeypatch.setattr(av, "_read_triton_provenance", lambda a: _SHA_POST_ALL)
    scen = tmp_path / "sim_0"
    a = _analysis_stub(df=_resumed_df(str(scen)), simlog_dir=tmp_path / "logs" / "sims")
    _write_real_log(a, 0, "start\n" + _CKPT + _REPLAY + _ENDS)
    res = check_coupled_resume_validity(a)
    assert res.passed is True
    assert res.details == []


# --- EW-2b: a check that EXAMINED NOTHING has not applied (P7 generalized). ------------
# `passed = len(details) == 0` cannot distinguish "examined 28, found nothing" from
# "examined 0", and the second is a green cell asserting a verification that never ran.
# Measured on the Iteration-5 combined report, three cells across BOTH models:
# clean_tritonswmm/Coupled resume validity, clean_tritonswmm/Resume schedule honored, and
# clean_triton/Resume schedule honored all rendered PASS carrying "(0 ... examined)".
# The disclosed denominator (Gotcha-71(d)) is NOT a substitute — it lives behind a hover
# title while the grid shows green.
#
# The gate is `examined == 0 AND not details`, and the second conjunct is load-bearing
# rather than defensive: see test_armc_zero_examined_finding_is_not_silenced below.


def test_coupled_zero_examined_is_na_not_a_green_pass(monkeypatch):
    """No resumed coupled sim -> N/A, not PASS.

    FAILS PRE-FIX: today this returns applicable=True with
    "No coupled-resume invalidity detected (0 resumed coupled sim(s) examined)", which the
    roll-up renderer draws as a green PASS.
    """
    monkeypatch.setattr(av, "_read_triton_provenance", lambda a: _SHA_POST_ALL)
    df = pd.DataFrame([{"model_type": "tritonswmm", "n_resumes": 0, "scenario_directory": ""}])
    res = check_coupled_resume_validity(_analysis_stub(df=df))
    assert res.applicable is False, res.summary
    assert res.passed is True
    assert res.details == []
    # The denominator survives into the N/A summary — the reason stays legible on hover.
    assert "0 resumed coupled sim(s) examined" in res.summary


def test_coupled_examined_population_stays_a_real_pass(monkeypatch, tmp_path):
    """A resumed sim that WAS examined and was clean stays a genuine PASS (applicable=True).

    Pairs with the N/A test above: without this, the same three cells would go grey under a
    change that made the check unconditionally inapplicable, which would destroy the real
    finding rather than fix the vacuous one.
    """
    monkeypatch.setattr(av, "_read_triton_provenance", lambda a: _SHA_POST_ALL)
    scen = tmp_path / "sim_0"
    a = _analysis_stub(df=_resumed_df(str(scen)), simlog_dir=tmp_path / "logs" / "sims")
    _write_real_log(a, 0, _CKPT + _REPLAY + _ENDS)
    res = check_coupled_resume_validity(a)
    assert res.applicable is True, res.summary
    assert res.passed is True
    assert "1 resumed coupled sim(s) examined" in res.summary


def test_armc_zero_examined_finding_is_not_silenced(monkeypatch, tmp_path):
    """THE SILENCING GUARD — why the gate is `examined == 0 AND not details`.

    Arm C (the SWMM node-depth scatter arm) appends a detail row for every resume candidate
    WITHOUT incrementing `examined`: the counter is touched only on the Arm A and Arm B
    paths. So a scatter-pin analysis whose per-sim logs were purged produces
    examined == 0 WITH findings, and a bare `examined == 0` gate would convert that real
    FAIL into a grey N/A — silencing exactly the class of finding this campaign's
    resume_tritonswmm cell carries.

    Green both before and after: it is a preservation test against a WRONG fix, not a
    discriminator between pre and post.
    """
    monkeypatch.setattr(av, "_read_triton_provenance", lambda a: _SHA_PRE_SCATTER)
    scen = tmp_path / "sim_0"
    # No _write_real_log: the producer path is absent, so Arm B concedes INDETERMINATE and
    # `examined` stays 0 while Arm C still fires on the pin.
    a = _analysis_stub(df=_resumed_df(str(scen)), simlog_dir=tmp_path / "logs" / "sims")
    res = check_coupled_resume_validity(a)
    assert res.details, "Arm C must have fired on the scatter-less pin"
    assert "0 resumed coupled sim(s) examined" in res.summary, "the zero-examined shape is the premise"
    assert res.applicable is True, f"Arm-C finding was silenced into N/A: {res.summary}"
    assert res.passed is False


def test_schedule_zero_examined_is_na_not_a_green_pass():
    """No resumed sim -> N/A, not PASS. Covers BOTH clean arms' schedule cell.

    FAILS PRE-FIX: today this returns applicable=True with "All resumed sims honored their
    configured resume schedule (0 resumed sim(s) schedule-verified)".
    """
    from hhemt.analysis_validation import check_resume_schedule_honored

    df = pd.DataFrame([{"model_type": "triton", "n_resumes": 0, "scenario_directory": "sim_0"}])
    res = check_resume_schedule_honored(_triton_arm_b_stub(df))
    assert res.applicable is False, res.summary
    assert res.passed is True
    assert res.details == []
    assert "0 resumed sim(s) schedule-verified" in res.summary


def test_schedule_examined_population_stays_a_real_pass():
    """A schedule-verified population stays a genuine PASS — the resume arms' cells."""
    from hhemt.analysis_validation import check_resume_schedule_honored

    df = pd.DataFrame(
        [
            {
                "model_type": "triton",
                "n_resumes": 3,
                "scenario_directory": "sim_0",
                "resume_reporting_tsteps": [36, 72, 108],
            }
        ]
    )
    res = check_resume_schedule_honored(_triton_arm_b_stub(df))
    assert res.applicable is True, res.summary
    assert res.passed is True
    assert "1 resumed sim(s) schedule-verified" in res.summary


def test_schedule_examined_population_still_fails():
    """A real schedule violation still FAILS with applicable=True."""
    from hhemt.analysis_validation import check_resume_schedule_honored

    df = pd.DataFrame(
        [
            {
                "model_type": "triton",
                "n_resumes": 5,
                "scenario_directory": "sim_0",
                "resume_reporting_tsteps": [36, 72, 108],
            }
        ]
    )
    res = check_resume_schedule_honored(_triton_arm_b_stub(df))
    assert res.applicable is True
    assert res.passed is False
    assert res.details


def test_clean_pin_with_resumed_sims_passes_positively(monkeypatch):
    """VMS-12 + VMS-13: at a pin where every registry defect resolves ABSENT, a resumed
    arm must PASS with a real denominator -- not FAIL, and not N/A.

    PRE-FIX (before VMS-12) this FAILS with passed=False: TRITON-RESUME-EXTBC-GHOST-RING
    carries trigger="resumed_any" and status="absent" at 5d2ad1e8, and the old
    `_applicable` filtered on TRIGGER only, so it admitted "triton" into
    _candidate_models on an arm the registry had just cleared. The 28 admitted rows then
    failed the coupling-only replay-marker test. This is the shipped defect in generation
    e389264af7b9, where synth_cc_resume_triton reported
    "28 resumed coupled sim(s) ... lack the exchange-replay marker".
    """
    monkeypatch.setattr(av, "_read_triton_provenance", lambda a: SHA_EXTBC_GHOST_RING_FIX)
    res = check_coupled_resume_validity(_analysis_stub(coupled=False, df=_resumed_df(model_type="triton")))
    assert res.passed is True, f"got passed={res.passed!r} summary={res.summary!r}"
    assert res.applicable is True, "a clean pin with resumed sims is a POSITIVE pass, not N/A"
    assert "no known resume defect" in res.summary


def test_affected_pin_with_resumed_sims_still_selects(monkeypatch):
    """OVER-FIRE arm: VMS-12 must not clear an arm whose pin genuinely carries a defect.

    At 9db367dd, TRITON-RESUME-EXTBC-GHOST-RING resolves PRESENT (also_present_set), so
    _affected is non-empty and the positive-PASS branch must NOT fire. Pins that the
    status filter narrows selection without disabling it.
    """
    monkeypatch.setattr(av, "_read_triton_provenance", lambda a: _SHA_POST_ALL)
    res = check_coupled_resume_validity(_analysis_stub(coupled=False, df=_resumed_df(model_type="triton")))
    assert (
        "no known resume defect" not in res.summary
    ), "the positive-PASS branch fired on a pin carrying a PRESENT defect"
