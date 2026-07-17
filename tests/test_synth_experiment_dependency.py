"""Unit tests for the cross-experiment dependency subsystem (Phase-5 reproducibility, P2+V3).

Exercises ``bundle/_dependency`` over MINIMAL synthetic bundle dirs (each a ``cfg_system.yaml``
carrying ``TRITONSWMM_branch_key`` + a ``scenario_status.csv`` carrying ``n_resumes`` -- no real zarr,
since ``resolve_dependency`` reads only identity + role). Covers: reuse-on-match, halt-loud-on-mismatch,
absent-halt-with-emitted-command (FQ2/AR2), and the auto_satisfy opt-in seam. Fast unit tier: no HPC,
no compile, no real analysis.
"""

from __future__ import annotations

import pytest

from hhemt.bundle._dependency import (
    ExperimentDependency,
    ExperimentIdentity,
    classify_bundle_role,
    read_bundle_identity,
    resolve_dependency,
)
from hhemt.exceptions import ConfigurationError


def _write_bundle(root, *, sha: str, n_resumes: int) -> None:
    """A minimal bundle dir: cfg_system.yaml (TRITONSWMM_branch_key) + scenario_status.csv (n_resumes)."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "cfg_system.yaml").write_text(f"TRITONSWMM_branch_key: {sha}\n")
    (root / "scenario_status.csv").write_text(f"sa_id,n_resumes\nsa_0,{n_resumes}\nsa_1,{n_resumes}\n")


def _clean_dep(sha: str = "3a832f7d") -> ExperimentDependency:
    return ExperimentDependency(
        dependency_experiment_id="synth_cc_clean",
        role="clean",
        expected_identity=ExperimentIdentity(tritonswmm_sha=sha),
    )


def test_classify_bundle_role(tmp_path):
    clean, resume = tmp_path / "clean", tmp_path / "resume"
    _write_bundle(clean, sha="3a832f7d", n_resumes=0)
    _write_bundle(resume, sha="3a832f7d", n_resumes=1)
    assert classify_bundle_role(clean) == "clean"
    assert classify_bundle_role(resume) == "resume"
    assert classify_bundle_role(tmp_path / "missing") == "clean"  # absence -> clean


def test_read_bundle_identity(tmp_path):
    b = tmp_path / "b"
    _write_bundle(b, sha="3a832f7d", n_resumes=0)
    ident = read_bundle_identity(b)
    assert ident.tritonswmm_sha == "3a832f7d"
    assert ident.case_name is None  # no case.yaml bundled -> None (skipped in matches)
    assert ident.compute_config_identity is None  # v1-optional


def test_identity_matches_skips_none_declared_fields():
    # A declaration pinning only the sha matches any bundle with that sha, regardless of case_name.
    expected = ExperimentIdentity(tritonswmm_sha="3a832f7d")
    ok, bad = expected.matches(ExperimentIdentity(tritonswmm_sha="3a832f7d", case_name="norfolk"))
    assert ok and bad == []
    ok, bad = expected.matches(ExperimentIdentity(tritonswmm_sha="15eb18a5"))
    assert not ok and bad == ["tritonswmm_sha"]


def test_resolve_reuse_on_match(tmp_path):
    clean = tmp_path / "clean"
    _write_bundle(clean, sha="3a832f7d", n_resumes=0)
    resolved = resolve_dependency(_clean_dep("3a832f7d"), search_roots=[clean], emitted_command="X")
    assert resolved == clean


def test_resolve_halts_loud_on_mismatch(tmp_path):
    clean = tmp_path / "clean"
    _write_bundle(clean, sha="15eb18a5", n_resumes=0)  # WRONG (pre-fix) sha
    with pytest.raises(ConfigurationError) as ei:
        resolve_dependency(_clean_dep("3a832f7d"), search_roots=[clean], emitted_command="X")
    msg = str(ei.value)
    assert "MISMATCH" in msg
    assert "3a832f7d" in msg and "15eb18a5" in msg  # expected-vs-found diff surfaced


def test_resolve_halts_with_emitted_command_when_absent(tmp_path):
    # No clean-role bundle present (only a resume bundle) -> absent branch (AR2: emit command, do not auto-run).
    resume = tmp_path / "resume"
    _write_bundle(resume, sha="3a832f7d", n_resumes=1)
    cmd = "python -m scripts.experiments.synth_compute_config clean --system-directory X --eda --bundle"
    with pytest.raises(ConfigurationError) as ei:
        resolve_dependency(_clean_dep("3a832f7d"), search_roots=[resume], emitted_command=cmd)
    msg = str(ei.value)
    assert "ABSENT" in msg
    assert cmd in msg  # the exact committed command is emitted in the halt message


def test_resolve_auto_satisfy_seam(tmp_path):
    # The opt-in auto_satisfy seam: produce a matching clean bundle, then resolve returns it.
    produced = tmp_path / "produced_clean"

    def _produce():
        _write_bundle(produced, sha="3a832f7d", n_resumes=0)
        return produced

    resolved = resolve_dependency(_clean_dep("3a832f7d"), search_roots=[], auto_satisfy=_produce)
    assert resolved == produced
