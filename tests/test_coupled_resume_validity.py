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


def _analysis_stub(*, coupled=True, sensitivity=False, df=None, simlog_dir=None):
    return SimpleNamespace(
        _system=SimpleNamespace(
            cfg_system=SimpleNamespace(toggle_tritonswmm_model=coupled),
        ),
        cfg_analysis=SimpleNamespace(toggle_sensitivity_analysis=sensitivity, is_subanalysis=False),
        analysis_paths=SimpleNamespace(
            analysis_datatree_zarr=None,
            sensitivity_datatree_zarr=None,
            simlog_directory=simlog_dir,
        ),
        df_status=df,
    )


def _resumed_df(scenario_directory="", event_iloc=0, sa_id=None):
    row = {
        "model_type": "tritonswmm",
        "n_resumes": 2,
        "scenario_directory": scenario_directory,
        "event_iloc": event_iloc,
    }
    if sa_id is not None:
        row["sa_id"] = sa_id
    return pd.DataFrame([row])


def _write_real_log(analysis, event_iloc, text):
    """Place the log at the path the PRODUCER writes — resolved by the real convention."""
    from hhemt.run_simulation import model_logfile_for

    p = model_logfile_for(analysis, event_iloc, "tritonswmm")
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


def test_coupled_off_is_na(monkeypatch):
    monkeypatch.setattr(av, "_read_triton_provenance", lambda a: (None, None))
    res = check_coupled_resume_validity(_analysis_stub(coupled=False))
    assert res.passed is True
    assert "not enabled" in res.summary


def test_unstamped_is_indeterminate(monkeypatch):
    monkeypatch.setattr(av, "_read_triton_provenance", lambda a: (None, None))
    res = check_coupled_resume_validity(_analysis_stub(df=_resumed_df()))
    assert res.passed is True
    assert "unknown" in res.summary
    assert res.details == []


def test_prefix_with_resume_warns(monkeypatch):
    monkeypatch.setattr(av, "_read_triton_provenance", lambda a: ("deadbeef", False))
    res = check_coupled_resume_validity(_analysis_stub(df=_resumed_df()))
    assert res.passed is False
    assert len(res.details) == 1
    assert "PRE-FIX" in res.details[0]["detail"]


def test_prefix_without_resume_passes(monkeypatch):
    monkeypatch.setattr(av, "_read_triton_provenance", lambda a: ("deadbeef", False))
    empty = pd.DataFrame([{"model_type": "tritonswmm", "n_resumes": 0, "scenario_directory": ""}])
    res = check_coupled_resume_validity(_analysis_stub(df=empty))
    assert res.passed is True


def test_postfix_missing_replay_marker_warns(monkeypatch, tmp_path):
    """Resumed + complete last exec, no replay marker -> WARN. The only WARN case.

    FAILS PRE-FIX: today the check reads the decoy (marker PRESENT) -> 0 details ->
    passed=True, so `assert res.passed is False` fails.
    """
    monkeypatch.setattr(av, "_read_triton_provenance", lambda a: ("cafebabe", True))
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
    monkeypatch.setattr(av, "_read_triton_provenance", lambda a: ("cafebabe", True))
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
    monkeypatch.setattr(av, "_read_triton_provenance", lambda a: ("cafebabe", True))
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
    monkeypatch.setattr(av, "_read_triton_provenance", lambda a: ("cafebabe", True))
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
    monkeypatch.setattr(av, "_read_triton_provenance", lambda a: ("cafebabe", True))
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
    monkeypatch.setattr(av, "_read_triton_provenance", lambda a: ("cafebabe", True))
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
    """The SENSITIVITY BRANCH. A master's df_status carries sa_id and its sub-analyses'
    logs live under {master}/logs/sims via the is_subanalysis branch of the convention.

    FAILS PRE-FIX: today the check reads the decoy (marker PRESENT) -> passed=True, so
    `assert res.passed is False` fails.
    """
    monkeypatch.setattr(av, "_read_triton_provenance", lambda a: ("cafebabe", True))
    scen = tmp_path / "sim_0"
    master_dir = tmp_path / "master"
    sub = SimpleNamespace(
        cfg_analysis=SimpleNamespace(
            is_subanalysis=True,
            analysis_id="sa_0",
            master_analysis_cfg_yaml=master_dir / "cfg_analysis.yaml",
        ),
        analysis_paths=SimpleNamespace(simlog_directory=tmp_path / "unused"),
    )
    master = _analysis_stub(
        sensitivity=True,
        df=_resumed_df(str(scen), sa_id="sa_0"),
        simlog_dir=tmp_path / "unused",
    )
    master.sensitivity = SimpleNamespace(sub_analyses={"sa_0": sub})
    master.cfg_analysis.toggle_sensitivity_analysis = True
    _write_real_log(sub, 0, _CKPT + _ENDS)  # resumed, complete, no replay marker -> WARN
    assert (master_dir / "logs" / "sims" / "model_tritonswmm_sa_0_evt0.log").exists()
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
        cfg_analysis=SimpleNamespace(is_subanalysis=False),
    )
    run = SimpleNamespace(_analysis=a, _scenario=SimpleNamespace(event_iloc=7))
    assert TRITONSWMM_run._analysis_level_model_logfile(run, "tritonswmm") == model_logfile_for(a, 7, "tritonswmm")
    assert model_logfile_for(a, 7, "tritonswmm").name == "model_tritonswmm_evt7.log"


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
    sys_log.triton_has_coupled_resume_fix.set(True)
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
    assert tree.attrs["triton_has_coupled_resume_fix"] is True

    write_datatree_zarr(tree, zarr_path)

    sha, has_fix = av._read_triton_provenance(analysis)
    assert sha == "cafebabecafebabecafebabecafebabecafebabe"
    assert has_fix is True


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
            "triton_has_coupled_resume_fix": True,
        }

    def _fake_open_datatree(path, **kwargs):
        if kwargs.get("chunks") == "auto":
            raise NotImplementedError("Can not use auto rechunking with object dtype")
        return _FakeTree()

    monkeypatch.setattr(xr, "open_datatree", _fake_open_datatree)

    sha, has_fix = av._read_triton_provenance(_fake_analysis_with_zarr(zarr_path))
    assert sha == "3a832f7d5eedd96aaee0dfe9181da5774adfb9f4"
    assert has_fix is True


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
        sha, has_fix = av._read_triton_provenance(_fake_analysis_with_zarr(zarr_path))

    assert (sha, has_fix) == (None, None)  # graceful-absent, never raises
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
        sha, has_fix = av._read_triton_provenance(_fake_analysis_with_zarr(zarr_path))

    assert (sha, has_fix) == (None, None)
    assert caplog.text == ""
