"""Tests for scripts/check_layout_version.py."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_layout_version.py"


def _make_repo(
    tmp_path: Path,
    layout_version_at_head: int,
    layout_version_at_main: int,
    extra_files_at_head: dict[str, str] | None = None,
    extra_files_at_main: dict[str, str] | None = None,
    sentinel_yaml: str | None = None,
) -> Path:
    """Build a tiny git repo with a matching constants.py and tagged main commit."""
    extra_head = extra_files_at_head or {}
    extra_main = extra_files_at_main or {}
    (tmp_path / "src" / "hhemt" / "version_migration").mkdir(parents=True)
    (tmp_path / "tests" / "fixtures" / "legacy_layouts").mkdir(parents=True)
    constants = tmp_path / "src" / "hhemt" / "version_migration" / "constants.py"
    sentinel = tmp_path / "_layout_relevant_files.yaml"
    sentinel.write_text(
        sentinel_yaml
        or "layout_relevant:\n  paths:\n    - src/foo.py\n  globs: []\nnon_breaking_allowlist: []\n"
    )
    constants.write_text(f"LAYOUT_VERSION: int = {layout_version_at_main}\n")
    for rel, content in extra_main.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "add", "."],
        cwd=tmp_path, check=True,
    )
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "main"],
        cwd=tmp_path, check=True,
    )
    subprocess.run(["git", "checkout", "-q", "-b", "feature"], cwd=tmp_path, check=True)
    constants.write_text(f"LAYOUT_VERSION: int = {layout_version_at_head}\n")
    for rel, content in extra_head.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "add", "."],
        cwd=tmp_path, check=True,
    )
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t",
         "commit", "--allow-empty", "-qm", "head"],
        cwd=tmp_path, check=True,
    )
    return tmp_path


def _run(repo: Path, *args: str) -> subprocess.CompletedProcess:
    target = repo / "scripts" / "check_layout_version.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(SCRIPT.read_text())
    return subprocess.run(
        [sys.executable, str(target), *args], cwd=repo, capture_output=True, text=True
    )


def test_check_a_passes_when_no_version_change(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, layout_version_at_head=4, layout_version_at_main=4)
    out = _run(repo, "check-a", "main")
    assert out.returncode == 0, out.stderr


def test_check_a_fails_when_bump_without_migration(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, layout_version_at_head=5, layout_version_at_main=4)
    out = _run(repo, "check-a", "main")
    assert out.returncode == 1
    assert "no migration module" in out.stderr


def test_check_a_fails_when_bump_without_fixtures(tmp_path: Path) -> None:
    repo = _make_repo(
        tmp_path, layout_version_at_head=5, layout_version_at_main=4,
        extra_files_at_head={
            "src/hhemt/version_migration/versions/V0005__test.py": "x = 1\n",
        },
    )
    out = _run(repo, "check-a", "main")
    assert out.returncode == 1
    assert "fixture" in out.stderr


def test_check_b_passes_when_no_layout_relevant_changed(tmp_path: Path) -> None:
    repo = _make_repo(
        tmp_path, layout_version_at_head=4, layout_version_at_main=4,
        extra_files_at_head={"tests/test_unrelated.py": "def test_x(): pass\n"},
    )
    out = _run(repo, "check-b", "main")
    assert out.returncode == 0, out.stderr


def test_check_b_fails_when_layout_relevant_changed_without_bump(tmp_path: Path) -> None:
    repo = _make_repo(
        tmp_path, layout_version_at_head=4, layout_version_at_main=4,
        extra_files_at_main={"src/foo.py": "# initial\n"},
        extra_files_at_head={"src/foo.py": "# changed\n"},
    )
    out = _run(repo, "check-b", "main")
    assert out.returncode == 1
    assert "src/foo.py" in out.stderr


def test_check_b_passes_when_path_in_non_breaking_allowlist(tmp_path: Path) -> None:
    repo = _make_repo(
        tmp_path, layout_version_at_head=4, layout_version_at_main=4,
        extra_files_at_main={"src/foo.py": "# initial\n"},
        extra_files_at_head={"src/foo.py": "# changed\n"},
        sentinel_yaml=(
            "layout_relevant:\n  paths:\n    - src/foo.py\n  globs: []\n"
            "non_breaking_allowlist:\n  - src/foo.py\n"
        ),
    )
    out = _run(repo, "check-b", "main")
    assert out.returncode == 0, out.stderr


def test_check_b_fails_when_compute_event_id_slug_ast_drifts(tmp_path: Path) -> None:
    main_scenario = (
        "def compute_event_id_slug(year, event_type, event_id):\n"
        "    return f'year.{year}_event_type.{event_type}_event_id.{event_id}'\n"
    )
    head_scenario = (
        "def compute_event_id_slug(year, event_type, event_id):\n"
        "    return f'y{year}_t{event_type}_e{event_id}'\n"
    )
    repo = _make_repo(
        tmp_path, layout_version_at_head=4, layout_version_at_main=4,
        extra_files_at_main={"src/hhemt/scenario.py": main_scenario},
        extra_files_at_head={"src/hhemt/scenario.py": head_scenario},
        sentinel_yaml=(
            "layout_relevant:\n  paths:\n    - src/hhemt/scenario.py\n"
            "  globs: []\nnon_breaking_allowlist: []\n"
        ),
    )
    out = _run(repo, "check-b", "main")
    assert out.returncode == 1
    assert "compute_event_id_slug" in out.stderr
    assert "drift" in out.stderr


def test_check_c_warns_on_new_scenario_file_not_in_sentinel(tmp_path: Path) -> None:
    repo = _make_repo(
        tmp_path, layout_version_at_head=4, layout_version_at_main=4,
        extra_files_at_head={"src/hhemt/scenario_v2.py": "x = 1\n"},
    )
    out = _run(repo, "check-c", "main")
    assert out.returncode == 0
    assert "layout-suspicious" in out.stderr
    assert "scenario_v2.py" in out.stderr


# --- Package-dir rename transition (hhemt-rename Phase 1) -------------------
# These lock in the rename-aware guard behavior: a base ref that predates the
# src/TRITON_SWMM_toolkit -> src/hhemt package rename must not read as a spurious
# version bump (check-a), and a source move must not register as an on-disk
# layout change (check-b) — while in-place edits to layout files STILL flag.

def _git_t(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
        cwd=repo, check=True,
    )


def _make_rename_repo(
    tmp_path: Path,
    *,
    layout_version: int,
    sentinel_yaml: str,
    inplace_layout_file: tuple[str, str, str] | None = None,
) -> Path:
    """Repo whose `main` holds the package at src/TRITON_SWMM_toolkit/ and whose
    `feature` renames it to src/hhemt/ — the hhemt-rename Phase-1 transition.

    `inplace_layout_file` = optional (relpath, main_content, head_content) for a
    NON-renamed file edited in place at HEAD (to prove M-status still flags).
    """
    old_vm = tmp_path / "src" / "TRITON_SWMM_toolkit" / "version_migration"
    old_vm.mkdir(parents=True)
    (tmp_path / "tests" / "fixtures" / "legacy_layouts").mkdir(parents=True)
    (old_vm / "constants.py").write_text(f"LAYOUT_VERSION: int = {layout_version}\n")
    (tmp_path / "src" / "TRITON_SWMM_toolkit" / "paths.py").write_text("# paths v1\n")
    (tmp_path / "_layout_relevant_files.yaml").write_text(sentinel_yaml)
    if inplace_layout_file:
        rel, main_content, _ = inplace_layout_file
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(main_content)
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    _git_t(tmp_path, "add", ".")
    _git_t(tmp_path, "commit", "-qm", "main (pre-rename)")
    _git_t(tmp_path, "checkout", "-q", "-b", "feature")
    _git_t(tmp_path, "mv", "src/TRITON_SWMM_toolkit", "src/hhemt")
    if inplace_layout_file:
        rel, _, head_content = inplace_layout_file
        (tmp_path / rel).write_text(head_content)
    _git_t(tmp_path, "add", ".")
    _git_t(tmp_path, "commit", "-qm", "head (renamed to hhemt)")
    return tmp_path


def test_check_a_passes_across_package_rename(tmp_path: Path) -> None:
    """A package-dir rename with LAYOUT_VERSION unchanged must NOT read as a
    spurious 0->N bump. The rename-aware old-path fallback resolves the
    pre-rename base correctly. Without the fallback this fails (base reads 0)."""
    repo = _make_rename_repo(
        tmp_path, layout_version=12,
        sentinel_yaml=(
            "layout_relevant:\n  paths:\n    - src/hhemt/paths.py\n"
            "  globs: []\nnon_breaking_allowlist: []\n"
        ),
    )
    out = _run(repo, "check-a", "main")
    assert out.returncode == 0, out.stdout + out.stderr
    assert "unchanged (12)" in out.stdout


def test_check_b_skips_renamed_file_but_flags_inplace_edit(tmp_path: Path) -> None:
    """check-b must SKIP the renamed package file (a source move is not an
    on-disk-layout change) while STILL flagging an in-place edit to a
    layout-relevant file with no version bump — proving the rename-skip did not
    gut enforcement (the Option-1 over-exemption failure mode this guards)."""
    repo = _make_rename_repo(
        tmp_path, layout_version=4,
        sentinel_yaml=(
            "layout_relevant:\n  paths:\n    - src/hhemt/paths.py\n    - toplevel_layout.py\n"
            "  globs: []\nnon_breaking_allowlist: []\n"
        ),
        inplace_layout_file=("toplevel_layout.py", "# layout v1\n", "# layout v2\n"),
    )
    out = _run(repo, "check-b", "main")
    assert out.returncode == 1, out.stdout + out.stderr
    assert "toplevel_layout.py" in out.stderr  # in-place M-status edit STILL flags
    assert "src/hhemt/paths.py" not in out.stderr  # renamed file is skipped


def test_check_b_passes_when_only_renames(tmp_path: Path) -> None:
    """A pure package rename (no in-place layout edits, version unchanged) yields
    zero layout-relevant changes — the whole move set is R-status and skipped."""
    repo = _make_rename_repo(
        tmp_path, layout_version=4,
        sentinel_yaml=(
            "layout_relevant:\n  paths:\n    - src/hhemt/paths.py\n"
            "  globs: []\nnon_breaking_allowlist: []\n"
        ),
    )
    out = _run(repo, "check-b", "main")
    assert out.returncode == 0, out.stdout + out.stderr
