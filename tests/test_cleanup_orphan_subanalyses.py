"""Tests for TRITONSWMM_sensitivity_analysis.cleanup_orphan_subanalysis_dirs()."""

from pathlib import Path

import pytest

import tests.utils_for_testing as tst_ut  # noqa: F401 — reused fixtures


@pytest.fixture
def sa_with_scratch_subanalyses(norfolk_sensitivity_analysis_cached, tmp_path, monkeypatch):
    """Rebind ``sensitivity.subanalysis_dir`` to a tmp_path-rooted copy of the
    cached fixture's ``subanalyses/`` so orphan-cleanup tests never mutate the
    shared cached fixture directory."""
    sa = norfolk_sensitivity_analysis_cached.sensitivity
    scratch = tmp_path / "subanalyses"
    scratch.mkdir()
    for sa_id in sa.df_setup.index.astype(str):
        (scratch / f"{sa.sub_analyses_prefix}{sa_id}").mkdir()
    monkeypatch.setattr(sa, "subanalysis_dir", scratch)
    return sa


def _make_orphan_dir(sensitivity_analysis, sa_id: str) -> Path:
    """Create a fake orphaned sub-analysis directory on disk."""
    orphan = sensitivity_analysis.subanalysis_dir / f"sa_{sa_id}"
    orphan.mkdir(parents=True, exist_ok=True)
    (orphan / "dummy.txt").write_text("stale content")
    return orphan


def test_find_orphans_empty_when_csv_matches_disk(sa_with_scratch_subanalyses):
    sa = sa_with_scratch_subanalyses
    orphans = sa.find_orphan_subanalysis_dirs()
    assert orphans == []


def test_find_orphans_detects_extra_dir(sa_with_scratch_subanalyses):
    sa = sa_with_scratch_subanalyses
    orphan = _make_orphan_dir(sa, "99_orphan")
    orphans = sa.find_orphan_subanalysis_dirs()
    assert orphan in orphans
    assert len(orphans) == 1


def test_cleanup_dry_run_does_not_delete(sa_with_scratch_subanalyses):
    sa = sa_with_scratch_subanalyses
    orphan = _make_orphan_dir(sa, "99_dry")
    result = sa.cleanup_orphan_subanalysis_dirs(dry_run=True, force=False, verbose=False)
    assert orphan in result
    assert orphan.exists(), "Dry-run must not delete"


def test_cleanup_apply_without_force_raises(sa_with_scratch_subanalyses):
    sa = sa_with_scratch_subanalyses
    orphan = _make_orphan_dir(sa, "99_noforce")
    with pytest.raises(ValueError, match="force=True"):
        sa.cleanup_orphan_subanalysis_dirs(dry_run=False, force=False, verbose=False)
    assert orphan.exists(), "Must not delete without force"


def test_cleanup_apply_with_force_deletes(sa_with_scratch_subanalyses):
    sa = sa_with_scratch_subanalyses
    orphan = _make_orphan_dir(sa, "99_apply")
    result = sa.cleanup_orphan_subanalysis_dirs(dry_run=False, force=True, verbose=False)
    assert orphan in result
    assert not orphan.exists(), "Apply+force must delete"
    for sa_id in sa.df_setup.index.astype(str):
        assert (sa.subanalysis_dir / f"{sa.sub_analyses_prefix}{sa_id}").exists()


def test_non_sa_prefix_dirs_ignored(sa_with_scratch_subanalyses):
    sa = sa_with_scratch_subanalyses
    non_sa = sa.subanalysis_dir / "other_dir"
    non_sa.mkdir(parents=True, exist_ok=True)
    orphans = sa.find_orphan_subanalysis_dirs()
    assert non_sa not in orphans


def test_sa_prefix_with_invalid_charset_not_orphaned(sa_with_scratch_subanalyses):
    """sa_* dirs whose suffix violates ^[A-Za-z0-9_.]+$ must not be returned
    as orphans — they were not created by this toolkit and must not be deleted
    by --apply --force."""
    sa = sa_with_scratch_subanalyses
    hostile = sa.subanalysis_dir / "sa_has spaces"
    hostile.mkdir(parents=True, exist_ok=True)
    orphans = sa.find_orphan_subanalysis_dirs()
    assert hostile not in orphans


def test_missing_subanalyses_dir_is_noop(norfolk_sensitivity_analysis_cached, monkeypatch, tmp_path):
    sa = norfolk_sensitivity_analysis_cached.sensitivity
    monkeypatch.setattr(sa, "subanalysis_dir", tmp_path / "nonexistent")
    orphans = sa.find_orphan_subanalysis_dirs()
    assert orphans == []
