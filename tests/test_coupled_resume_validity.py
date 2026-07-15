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
def _analysis_stub(*, coupled=True, sensitivity=False, df=None):
    return SimpleNamespace(
        _system=SimpleNamespace(
            cfg_system=SimpleNamespace(toggle_tritonswmm_model=coupled),
        ),
        cfg_analysis=SimpleNamespace(toggle_sensitivity_analysis=sensitivity),
        analysis_paths=SimpleNamespace(
            analysis_datatree_zarr=None, sensitivity_datatree_zarr=None
        ),
        df_status=df,
    )


def _resumed_df(scenario_directory=""):
    return pd.DataFrame(
        [{"model_type": "tritonswmm", "n_resumes": 2, "scenario_directory": scenario_directory}]
    )


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
    empty = pd.DataFrame(
        [{"model_type": "tritonswmm", "n_resumes": 0, "scenario_directory": ""}]
    )
    res = check_coupled_resume_validity(_analysis_stub(df=empty))
    assert res.passed is True


def test_postfix_missing_replay_marker_warns(monkeypatch, tmp_path):
    monkeypatch.setattr(av, "_read_triton_provenance", lambda a: ("cafebabe", True))
    scen = tmp_path / "sim_0"
    (scen / "logs").mkdir(parents=True)
    (scen / "logs" / "run_tritonswmm.log").write_text("some log without the marker\n")
    res = check_coupled_resume_validity(_analysis_stub(df=_resumed_df(str(scen))))
    assert res.passed is False
    assert len(res.details) == 1
    assert "replay never engaged" in res.details[0]["detail"]


def test_postfix_with_replay_marker_passes(monkeypatch, tmp_path):
    monkeypatch.setattr(av, "_read_triton_provenance", lambda a: ("cafebabe", True))
    scen = tmp_path / "sim_0"
    (scen / "logs").mkdir(parents=True)
    (scen / "logs" / "run_tritonswmm.log").write_text(
        "start\nSWMM exchange history replayed to t=3600 s (12 steps); resuming live segment\ndone\n"
    )
    res = check_coupled_resume_validity(_analysis_stub(df=_resumed_df(str(scen))))
    assert res.passed is True


def test_postfix_unreadable_log_is_indeterminate(monkeypatch, tmp_path):
    monkeypatch.setattr(av, "_read_triton_provenance", lambda a: ("cafebabe", True))
    # scenario_directory points nowhere -> log unreadable -> INDETERMINATE, no warn.
    res = check_coupled_resume_validity(_analysis_stub(df=_resumed_df(str(tmp_path / "missing"))))
    assert res.passed is True
    assert res.details == []


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
        analysis_paths=SimpleNamespace(
            analysis_datatree_zarr=zarr_path, sensitivity_datatree_zarr=None
        ),
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
