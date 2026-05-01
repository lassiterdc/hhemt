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
    (tmp_path / "src" / "TRITON_SWMM_toolkit" / "version_migration").mkdir(parents=True)
    (tmp_path / "tests" / "fixtures" / "legacy_layouts").mkdir(parents=True)
    constants = tmp_path / "src" / "TRITON_SWMM_toolkit" / "version_migration" / "constants.py"
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
            "src/TRITON_SWMM_toolkit/version_migration/versions/V0005__test.py": "x = 1\n",
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
        extra_files_at_main={"src/TRITON_SWMM_toolkit/scenario.py": main_scenario},
        extra_files_at_head={"src/TRITON_SWMM_toolkit/scenario.py": head_scenario},
        sentinel_yaml=(
            "layout_relevant:\n  paths:\n    - src/TRITON_SWMM_toolkit/scenario.py\n"
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
        extra_files_at_head={"src/TRITON_SWMM_toolkit/scenario_v2.py": "x = 1\n"},
    )
    out = _run(repo, "check-c", "main")
    assert out.returncode == 0
    assert "layout-suspicious" in out.stderr
    assert "scenario_v2.py" in out.stderr
