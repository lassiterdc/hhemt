"""Tests for scripts/check_release_metadata.py, the pre-tag release-date gate (approved shape F2, M11).

The gate reads the version from pyproject.toml and CITATION.cff and the release date from
CITATION.cff's `date-released` AND the `## vX.Y.Z (date)` heading of HISTORY.md, and `--stamp`
rewrites both dates. Each test drives `main([...])`/`check`/`stamp` over a throwaway tree.

Each test carries its CLASS. REGRESSION: red on the tree before this chunk (the gate read no
HISTORY.md, so a stale heading passed). NEW CAPABILITY: `stamp` and `--stamp` did not exist, so the
plausible wrong implementation is named instead.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import scripts.check_release_metadata as gate  # repo root is on sys.path under pytest

_REPO = Path(__file__).resolve().parent.parent
TODAY = "2026-11-13"
YESTERDAY = "2026-11-12"


def _tree(
    tmp_path: Path,
    *,
    py_version: str = "0.2.0",
    cff_version: str = "0.2.0",
    cff_date: str = TODAY,
    history: str | None = None,
) -> Path:
    (tmp_path / "pyproject.toml").write_text(f'[project]\nname = "x"\nversion = "{py_version}"\n', encoding="utf-8")
    (tmp_path / "CITATION.cff").write_text(
        f'cff-version: 1.2.0\nversion: "{cff_version}"\ndate-released: "{cff_date}"\n', encoding="utf-8"
    )
    if history is None:
        history = f"# Release history\n\n## v0.2.0 ({TODAY})\n\nbody\n\n## v0.1.0 (2026-07-14)\n\nfirst\n"
    (tmp_path / "HISTORY.md").write_text(history, encoding="utf-8")
    return tmp_path


def test_consistent_tree_has_no_findings(tmp_path: Path) -> None:
    """NEW CAPABILITY: wrong implementation = a gate that flags every tree; second correct
    implementation = one that reads HISTORY.md by line scan instead of regex."""
    assert gate.check(_tree(tmp_path), TODAY) == []


def test_stale_history_heading_is_a_finding(tmp_path: Path) -> None:
    """REGRESSION: red before this chunk, when no HISTORY.md was read and CITATION.cff alone drove
    the verdict. Satisfying input: the same tree with the heading corrected has no finding."""
    root = _tree(tmp_path, history=f"## v0.2.0 ({YESTERDAY})\n\nbody\n")
    findings = gate.check(root, TODAY)
    assert any("HISTORY.md heading" in f and YESTERDAY in f for f in findings)
    (root / "HISTORY.md").write_text(f"## v0.2.0 ({TODAY})\n", encoding="utf-8")
    assert gate.check(root, TODAY) == []


def test_two_copies_disagreeing_with_each_other_is_a_finding(tmp_path: Path) -> None:
    """REGRESSION: with the date copies disagreeing the old gate named only the CITATION.cff copy
    against today. Here CITATION.cff is right and HISTORY.md is the odd copy out."""
    root = _tree(tmp_path, cff_date=TODAY, history=f"## v0.2.0 ({YESTERDAY})\n")
    findings = gate.check(root, TODAY)
    assert any("disagrees between files" in f for f in findings)
    assert any("HISTORY.md heading" in f for f in findings)


def test_older_history_entries_are_not_read(tmp_path: Path) -> None:
    """REGRESSION: a wrong implementation reading the FIRST heading of the file, or the last, would
    pass or fail on the wrong entry. The v0.1.0 date is deliberately old and must not matter."""
    root = _tree(tmp_path, history=f"## v0.2.0 ({TODAY})\n\n## v0.1.0 (2020-01-01)\n")
    assert gate.check(root, TODAY) == []


def test_heading_for_the_release_version_may_not_be_first(tmp_path: Path) -> None:
    """REGRESSION: a first-heading reader passes this tree wrongly in the other direction. The entry
    for pyproject's version is selected by VERSION, not by position."""
    root = _tree(
        tmp_path, py_version="0.1.0", cff_version="0.1.0", history=f"## v0.2.0 ({TODAY})\n\n## v0.1.0 ({YESTERDAY})\n"
    )
    findings = gate.check(root, TODAY)
    assert any("HISTORY.md heading '## v0.1.0 (2026-11-12)'" in f for f in findings)


def test_missing_history_heading_for_version_is_an_environment_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """NEW CAPABILITY: wrong implementation = treat an absent heading as a pass (the docstring's
    'an absent value is an environment error and never a silent pass')."""
    root = _tree(tmp_path, history=f"## v0.1.0 ({TODAY})\n")
    assert gate.main(["--repo-root", str(root), "--release-date", TODAY]) == 2
    assert "no '## v0.2.0" in capsys.readouterr().err


def test_missing_history_file_is_an_environment_error(tmp_path: Path) -> None:
    """NEW CAPABILITY: wrong implementation = skip the HISTORY.md read when the file is absent."""
    root = _tree(tmp_path)
    (root / "HISTORY.md").unlink()
    assert gate.main(["--repo-root", str(root), "--release-date", TODAY]) == 2


def test_duplicate_heading_for_version_is_an_environment_error(tmp_path: Path) -> None:
    """NEW CAPABILITY: wrong implementation = take the first of two headings for one version."""
    root = _tree(tmp_path, history=f"## v0.2.0 ({TODAY})\n\n## v0.2.0 ({YESTERDAY})\n")
    assert gate.main(["--repo-root", str(root), "--release-date", TODAY]) == 2


def test_main_exit_codes_for_stale_and_current(tmp_path: Path) -> None:
    """REGRESSION: the exit is 1 on a stale HISTORY heading alone (CITATION.cff correct) and 0 once
    corrected, so a script wrapper cannot read a pass off the CITATION.cff copy."""
    root = _tree(tmp_path, history=f"## v0.2.0 ({YESTERDAY})\n")
    assert gate.main(["--repo-root", str(root), "--release-date", TODAY]) == 1
    (root / "HISTORY.md").write_text(f"## v0.2.0 ({TODAY})\n", encoding="utf-8")
    assert gate.main(["--repo-root", str(root), "--release-date", TODAY]) == 0


def test_stamp_rewrites_both_copies_to_the_release_date(tmp_path: Path) -> None:
    """NEW CAPABILITY: wrong implementation = stamp only CITATION.cff, or stamp every heading of
    HISTORY.md. Both stale; afterwards both carry TODAY and the v0.1.0 heading is untouched."""
    root = _tree(tmp_path, cff_date=YESTERDAY, history=f"## v0.2.0 ({YESTERDAY})\n\nbody\n\n## v0.1.0 (2026-07-14)\n")
    changed = gate.stamp(root, TODAY)
    assert sorted(p.name for p in changed) == ["CITATION.cff", "HISTORY.md"]
    assert f'date-released: "{TODAY}"' in (root / "CITATION.cff").read_text(encoding="utf-8")
    hist = (root / "HISTORY.md").read_text(encoding="utf-8")
    assert f"## v0.2.0 ({TODAY})" in hist and "## v0.1.0 (2026-07-14)" in hist
    assert gate.check(root, TODAY) == []


def test_stamp_is_idempotent_and_reports_no_change(tmp_path: Path) -> None:
    """NEW CAPABILITY: wrong implementation = rewrite (and so report changed) on an already-stamped
    tree, which would make `just stamp-release` try to commit nothing."""
    root = _tree(tmp_path)
    assert gate.stamp(root, TODAY) == []


def test_stamp_refuses_when_the_versions_disagree(tmp_path: Path) -> None:
    """NEW CAPABILITY: wrong implementation = stamp anyway, which makes the check pass on a wrong
    release. The files are left byte-identical."""
    root = _tree(
        tmp_path, py_version="0.3.0", cff_version="0.2.0", cff_date=YESTERDAY, history=f"## v0.3.0 ({YESTERDAY})\n"
    )
    before = ((root / "CITATION.cff").read_bytes(), (root / "HISTORY.md").read_bytes())
    with pytest.raises(ValueError, match="version disagrees"):
        gate.stamp(root, TODAY)
    assert ((root / "CITATION.cff").read_bytes(), (root / "HISTORY.md").read_bytes()) == before


def test_stamp_flag_through_main_stamps_then_passes(tmp_path: Path) -> None:
    """NEW CAPABILITY: wrong implementation = `--stamp` that stamps and exits 0 without the check,
    or one that checks before stamping and returns 1."""
    root = _tree(tmp_path, cff_date=YESTERDAY, history=f"## v0.2.0 ({YESTERDAY})\n")
    assert gate.main(["--repo-root", str(root), "--release-date", TODAY, "--stamp"]) == 0
    assert gate.main(["--repo-root", str(root), "--release-date", TODAY]) == 0


def test_the_real_repository_history_has_a_heading_for_the_current_version() -> None:
    """REGRESSION: on the real tree the gate resolves a HISTORY.md heading for pyproject's version
    (read as data, not run against today's date, which would make this test a function of the day)."""
    py_version, _cff_version, _cff_date, history_date = gate._read(_REPO)
    assert history_date  # a YYYY-MM-DD string, or _read raised
    assert gate.HISTORY_HEADING.search(f"## v{py_version} ({history_date})")


def test_justfile_has_a_stamp_release_recipe_that_commits_only_the_two_files() -> None:
    """REGRESSION: the recipe did not exist. Pins the shape the shape-item states: one commit of
    exactly the two copies, no tag and no push."""
    text = (_REPO / "justfile").read_text(encoding="utf-8")
    body = text.split("stamp-release:", 1)[1].split("\n# Tag the current version", 1)[0]
    assert "scripts/check_release_metadata.py --stamp" in body
    assert 'git commit -m "chore(release): stamp v{{VERSION}} release date" -- CITATION.cff HISTORY.md' in body
    assert "git tag" not in body and "git push" not in body
