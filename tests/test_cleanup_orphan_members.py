"""Tests for TRITONSWMM_sensitivity_analysis.cleanup_orphan_member_dirs()."""

from pathlib import Path

import pytest

import tests.utils_for_testing as tst_ut  # noqa: F401 — reused fixtures


@pytest.fixture
def member_with_scratch_analyses(norfolk_sensitivity_analysis_cached, tmp_path, monkeypatch):
    """Rebind ``sensitivity.member_dir`` to a tmp_path-rooted copy of the
    cached fixture's ``members/`` so orphan-cleanup tests never mutate the
    shared cached fixture directory."""
    member = norfolk_sensitivity_analysis_cached.sensitivity
    scratch = tmp_path / "members"
    scratch.mkdir()
    for member_id in member.df_setup.index.astype(str):
        (scratch / f"{member.member_prefix}{member_id}").mkdir()
    monkeypatch.setattr(member, "member_dir", scratch)
    return member


def _make_orphan_dir(sensitivity_analysis, member_id: str) -> Path:
    """Create a fake orphaned member directory on disk."""
    orphan = sensitivity_analysis.members_dir / f"member_{member_id}"
    orphan.mkdir(parents=True, exist_ok=True)
    (orphan / "dummy.txt").write_text("stale content")
    return orphan


def test_find_orphans_empty_when_csv_matches_disk(member_with_scratch_analyses):
    member = member_with_scratch_analyses
    orphans = member.find_orphan_member_dirs()
    assert orphans == []


def test_find_orphans_detects_extra_dir(member_with_scratch_analyses):
    member = member_with_scratch_analyses
    orphan = _make_orphan_dir(member, "99_orphan")
    orphans = member.find_orphan_member_dirs()
    assert orphan in orphans
    assert len(orphans) == 1


def test_cleanup_dry_run_does_not_delete(member_with_scratch_analyses):
    member = member_with_scratch_analyses
    orphan = _make_orphan_dir(member, "99_dry")
    result = member.cleanup_orphan_analysis_dirs(dry_run=True, force=False, verbose=False)
    assert orphan in result
    assert orphan.exists(), "Dry-run must not delete"


def test_cleanup_apply_without_force_raises(member_with_scratch_analyses):
    member = member_with_scratch_analyses
    orphan = _make_orphan_dir(member, "99_noforce")
    with pytest.raises(ValueError, match="force=True"):
        member.cleanup_orphan_analysis_dirs(dry_run=False, force=False, verbose=False)
    assert orphan.exists(), "Must not delete without force"


def test_cleanup_apply_with_force_deletes(member_with_scratch_analyses):
    member = member_with_scratch_analyses
    orphan = _make_orphan_dir(member, "99_apply")
    result = member.cleanup_orphan_analysis_dirs(dry_run=False, force=True, verbose=False)
    assert orphan in result
    assert not orphan.exists(), "Apply+force must delete"
    for member_id in member.df_setup.index.astype(str):
        assert (member.members_dir / f"{member.member_prefix}{member_id}").exists()


def test_non_member_prefix_dirs_ignored(member_with_scratch_analyses):
    member = member_with_scratch_analyses
    non_member = member.members_dir / "other_dir"
    non_member.mkdir(parents=True, exist_ok=True)
    orphans = member.find_orphan_member_dirs()
    assert non_member not in orphans


def test_member_prefix_with_invalid_charset_not_orphaned(member_with_scratch_analyses):
    """member_* dirs whose suffix violates ^[A-Za-z0-9_.]+$ must not be returned
    as orphans — they were not created by this toolkit and must not be deleted
    by --apply --force."""
    member = member_with_scratch_analyses
    hostile = member.members_dir / "member_has spaces"
    hostile.mkdir(parents=True, exist_ok=True)
    orphans = member.find_orphan_member_dirs()
    assert hostile not in orphans


def test_missing_members_dir_is_noop(norfolk_sensitivity_analysis_cached, monkeypatch, tmp_path):
    member = norfolk_sensitivity_analysis_cached.sensitivity
    monkeypatch.setattr(member, "member_dir", tmp_path / "nonexistent")
    orphans = member.find_orphan_member_dirs()
    assert orphans == []
