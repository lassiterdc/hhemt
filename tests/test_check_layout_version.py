"""Tests for scripts/check_layout_version.py."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import yaml

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
        sentinel_yaml or "layout_relevant:\n  paths:\n    - src/foo.py\n  globs: []\nnon_breaking_allowlist: []\n"
    )
    constants.write_text(f"LAYOUT_VERSION: int = {layout_version_at_main}\n")
    for rel, content in extra_main.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "add", "."],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "main"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(["git", "checkout", "-q", "-b", "feature"], cwd=tmp_path, check=True)
    constants.write_text(f"LAYOUT_VERSION: int = {layout_version_at_head}\n")
    for rel, content in extra_head.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "add", "."],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "--allow-empty", "-qm", "head"],
        cwd=tmp_path,
        check=True,
    )
    return tmp_path


def _run(repo: Path, *args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    target = repo / "scripts" / "check_layout_version.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(SCRIPT.read_text())
    return subprocess.run(
        [sys.executable, str(target), *args],
        cwd=repo,
        capture_output=True,
        text=True,
        # Merged rather than replaced: the child needs the ambient environment (PATH, and the
        # interpreter's own vars) as well as the ref pair under test. A test that passed
        # `env=` alone would exercise a stripped environment instead of the venue.
        env={**os.environ, **(env or {})},
    )


def _tracked_diff_vs_head(repo: Path) -> str:
    """Tracked worktree+index changes against HEAD. Untracked files are deliberately invisible:
    `_run` writes an untracked copy of the checker into every fixture repo, so a
    `git status --porcelain` probe would report every fixture dirty."""
    return subprocess.run(
        ["git", "diff", "--name-only", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()


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
        tmp_path,
        layout_version_at_head=5,
        layout_version_at_main=4,
        extra_files_at_head={
            "src/hhemt/version_migration/versions/V0005__test.py": "x = 1\n",
        },
    )
    out = _run(repo, "check-a", "main")
    assert out.returncode == 1
    assert "fixture" in out.stderr


def test_check_b_passes_when_no_layout_relevant_changed(tmp_path: Path) -> None:
    repo = _make_repo(
        tmp_path,
        layout_version_at_head=4,
        layout_version_at_main=4,
        extra_files_at_head={"tests/test_unrelated.py": "def test_x(): pass\n"},
    )
    out = _run(repo, "check-b", "main")
    assert out.returncode == 0, out.stderr


def test_check_b_fails_when_layout_relevant_changed_without_bump(tmp_path: Path) -> None:
    repo = _make_repo(
        tmp_path,
        layout_version_at_head=4,
        layout_version_at_main=4,
        extra_files_at_main={"src/foo.py": "# initial\n"},
        extra_files_at_head={"src/foo.py": "# changed\n"},
    )
    out = _run(repo, "check-b", "main")
    assert out.returncode == 1
    assert "src/foo.py" in out.stderr


def test_check_b_passes_when_allowlist_signature_matches_current_content(tmp_path: Path) -> None:
    repo = _make_repo(
        tmp_path,
        layout_version_at_head=4,
        layout_version_at_main=4,
        extra_files_at_main={"src/foo.py": "# initial\n"},
        extra_files_at_head={"src/foo.py": "# changed\n"},
        sentinel_yaml=(
            "layout_relevant:\n  paths:\n    - src/foo.py\n  globs: []\nnon_breaking_allowlist:\n"
            "  - path: src/foo.py\n    justification: baseline\n"
            '    layout_signature: "8ff88919a6004572b4249e0f72b28ed9945e61bdd0e2e7d79ddb9c5b1248674f"\n'
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
        "def compute_event_id_slug(year, event_type, event_id):\n    return f'y{year}_t{event_type}_e{event_id}'\n"
    )
    repo = _make_repo(
        tmp_path,
        layout_version_at_head=4,
        layout_version_at_main=4,
        extra_files_at_main={"src/hhemt/scenario.py": main_scenario},
        extra_files_at_head={"src/hhemt/scenario.py": head_scenario},
        sentinel_yaml=(
            "layout_relevant:\n  paths:\n    - src/hhemt/scenario.py\n  globs: []\nnon_breaking_allowlist: []\n"
        ),
    )
    out = _run(repo, "check-b", "main")
    assert out.returncode == 1
    assert "compute_event_id_slug" in out.stderr
    assert "drift" in out.stderr


def test_check_c_warns_on_new_scenario_file_not_in_sentinel(tmp_path: Path) -> None:
    repo = _make_repo(
        tmp_path,
        layout_version_at_head=4,
        layout_version_at_main=4,
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
        cwd=repo,
        check=True,
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
        tmp_path,
        layout_version=12,
        sentinel_yaml=(
            "layout_relevant:\n  paths:\n    - src/hhemt/paths.py\n  globs: []\nnon_breaking_allowlist: []\n"
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
        tmp_path,
        layout_version=4,
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
        tmp_path,
        layout_version=4,
        sentinel_yaml=(
            "layout_relevant:\n  paths:\n    - src/hhemt/paths.py\n  globs: []\nnon_breaking_allowlist: []\n"
        ),
    )
    out = _run(repo, "check-b", "main")
    assert out.returncode == 0, out.stdout + out.stderr


def _load_check_module():
    spec = importlib.util.spec_from_file_location("check_layout_version", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    # Register in sys.modules before exec: the module uses `from __future__ import
    # annotations`, so @dataclass(AllowlistEntry)'s InitVar/ClassVar resolution
    # looks up sys.modules[cls.__module__]; without registration it is None and
    # dataclasses raises AttributeError at class-definition time.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _git_tracked_relpaths(repo_root) -> list[str]:
    out = subprocess.check_output(["git", "ls-files"], cwd=repo_root, text=True)
    return [line for line in out.splitlines() if line.strip()]


def test_every_sentinel_entry_resolves_on_disk() -> None:
    """Check-B cannot silently degrade to a vacuous pass on a stale/typo'd entry."""
    mod = _load_check_module()
    sentinel = mod._load_sentinel()
    repo_root = mod.REPO_ROOT
    tracked = _git_tracked_relpaths(repo_root)
    for rel in sentinel["layout_relevant"]["paths"]:
        assert (repo_root / rel).exists(), f"sentinel path {rel} does not exist on disk"
    for g in sentinel["layout_relevant"]["globs"]:
        assert any(mod._layout_glob_match(rel, g) for rel in tracked), (
            f"sentinel glob {g!r} matches no tracked file (stale glob or matcher regression)"
        )
    # allowlist parses without SystemExit (non-empty justification enforced) and every path exists
    allow = mod._load_allowlist(sentinel)
    for rel in allow:
        assert (repo_root / rel).exists(), f"allowlisted path {rel} does not exist on disk"


def test_layout_glob_match_semantics() -> None:
    mod = _load_check_module()
    cfg = "src/hhemt/config/**/*.py"
    assert mod._layout_glob_match("src/hhemt/config/analysis.py", cfg)  # zero-dir direct child
    assert mod._layout_glob_match("src/hhemt/config/loaders/x.py", cfg)  # one-dir nested
    assert not mod._layout_glob_match("src/hhemt/config/analysis.txt", cfg)  # wrong ext
    assert not mod._layout_glob_match("src/hhemt/configXanalysis.py", cfg)  # no false-positive on prefix
    versions = "src/hhemt/version_migration/versions/*.py"
    assert mod._layout_glob_match("src/hhemt/version_migration/versions/V0001__x.py", versions)
    assert not mod._layout_glob_match("src/hhemt/version_migration/versions/sub/x.py", versions)  # single-*


def test_check_c_does_not_warn_on_new_direct_child_config_file(tmp_path: Path) -> None:
    """Post-fix, a new direct-child config/*_config.py is glob-covered, so check-c must not warn."""
    repo = _make_repo(
        tmp_path,
        layout_version_at_head=4,
        layout_version_at_main=4,
        extra_files_at_head={"src/hhemt/config/new_thing_config.py": "x = 1\n"},
        sentinel_yaml=(
            'layout_relevant:\n  paths: []\n  globs:\n    - "src/hhemt/config/**/*.py"\nnon_breaking_allowlist: []\n'
        ),
    )
    out = _run(repo, "check-c", "main")
    assert out.returncode == 0
    assert "new_thing_config.py" not in out.stderr


def _make_linear_repo(
    tmp_path: Path,
    versions: list[int],
    *,
    worktree_version: int,
    touch_layout_file: bool = True,
    migrations: tuple[int, ...] = (),
    commits: int | None = None,
    commit_pending: bool = False,
) -> Path:
    """Build a LINEAR repo of `len(versions)` commits, then leave a pending worktree change.

    `_make_repo` above builds a two-commit main/feature pair and its tests pass
    `base_ref="main"`. The check-b DISARM tests below need three commits on one branch so
    that HEAD~1, HEAD and the worktree can each carry a different LAYOUT_VERSION -- the
    disarm reads HEAD while `base_ref` reads HEAD~1, and no two-commit fixture can separate
    them. Sibling builder rather than a new parameter on `_make_repo`, so no existing test
    changes shape.
    """
    (tmp_path / "src" / "hhemt" / "version_migration" / "versions").mkdir(parents=True)
    (tmp_path / "tests" / "fixtures" / "legacy_layouts").mkdir(parents=True)
    (tmp_path / "_layout_relevant_files.yaml").write_text(
        "layout_relevant:\n  paths:\n    - src/foo.py\n  globs: []\nnon_breaking_allowlist: []\n"
    )
    (tmp_path / "src" / "foo.py").write_text("# initial\n")
    constants = tmp_path / "src" / "hhemt" / "version_migration" / "constants.py"
    for v in migrations:
        (tmp_path / "src" / "hhemt" / "version_migration" / "versions" / f"V{v:04d}__m.py").write_text('"""m"""\n')
        for n in (v - 1, v):
            d = tmp_path / "tests" / "fixtures" / "legacy_layouts" / f"v{n}"
            d.mkdir(exist_ok=True)
            (d / ".keep").touch()
    git = ["git", "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    for i, v in enumerate(versions[: commits if commits is not None else len(versions)]):
        constants.write_text(f"LAYOUT_VERSION: int = {v}\n")
        subprocess.run([*git, "add", "."], cwd=tmp_path, check=True)
        subprocess.run([*git, "commit", "--allow-empty", "-qm", f"c{i}"], cwd=tmp_path, check=True)
    constants.write_text(f"LAYOUT_VERSION: int = {worktree_version}\n")
    if touch_layout_file:
        (tmp_path / "src" / "foo.py").write_text("# changed\n")
    if commit_pending:
        subprocess.run([*git, "add", "."], cwd=tmp_path, check=True)
        subprocess.run([*git, "commit", "-qm", "pending landed"], cwd=tmp_path, check=True)
    return tmp_path


def test_check_b_scans_when_the_bump_landed_one_commit_back(tmp_path: Path) -> None:
    """THE TIDY-UP TRIPWIRE. A bump landed in the PREVIOUS commit; the pending change does
    not bump and touches a layout-relevant file. The disarm's `_layout_version_at("HEAD")`
    literal is what keeps the scan alive here -- replacing it with `base_ref` for
    consistency makes `head_v != base_v` true, the disarm fires, and check-b's file scan
    does not run at all for the whole commit."""
    repo = _make_linear_repo(tmp_path, [18, 19], worktree_version=19, migrations=(19,))
    out = _run(repo, "check-b", "HEAD~1")
    assert out.returncode == 1
    assert "src/foo.py" in out.stderr


def test_check_b_scans_when_the_pending_change_backs_a_bump_out(tmp_path: Path) -> None:
    """The previous commit bumped and the pending change REVERSES it. Without the
    `and head_v != base_v` conjunct the disarm fires on a DOWNGRADE, reporting
    `bumped (19->18)`, while check-a at the same base reports `unchanged (18); pass` -- so
    the pending layout-relevant change is neither scanned nor validated."""
    repo = _make_linear_repo(tmp_path, [18, 19], worktree_version=18, migrations=(19,))
    out = _run(repo, "check-b", "HEAD~1")
    assert out.returncode == 1
    assert "src/foo.py" in out.stderr


def test_check_b_scans_when_a_reverted_bump_is_restored(tmp_path: Path) -> None:
    """Mirror of the above: the previous commit LOWERED the version and the pending change
    restores it. Same blind handoff, same conjunct catches it."""
    repo = _make_linear_repo(tmp_path, [19, 18], worktree_version=19, migrations=(19,))
    out = _run(repo, "check-b", "HEAD~1")
    assert out.returncode == 1
    assert "src/foo.py" in out.stderr


def test_check_b_disarms_when_this_commit_carries_the_bump(tmp_path: Path) -> None:
    """The disarm MUST still fire when the PENDING change is the bump, or the conjunct has
    over-tightened into a gate that blocks every legitimate migration commit."""
    repo = _make_linear_repo(tmp_path, [18, 18], worktree_version=19, migrations=(19,))
    out = _run(repo, "check-b", "HEAD~1")
    assert out.returncode == 0
    assert "check-a covers this" in out.stdout


def test_check_b_range_mode_disarms_on_a_clean_tree_when_the_pushed_commit_carries_the_bump(
    tmp_path: Path,
) -> None:
    """THE PRE-PUSH REGRESSION ARM. The bump and its layout-relevant files are in HEAD and the
    tree is CLEAN, which is the state every pre-push invocation runs in. The pending-change
    disarm cannot fire here -- `_layout_version_at_head()` reads the worktree and
    `_layout_version_at("HEAD")` reads the commit, and on a clean tree those are equal by
    construction -- so before `--range` this state failed the push it was built to permit."""
    repo = _make_linear_repo(tmp_path, [18, 18], worktree_version=19, migrations=(19,), commit_pending=True)
    assert not _tracked_diff_vs_head(repo), "fixture must leave a CLEAN tree or it tests nothing"
    out = _run(repo, "check-b", "--range", "HEAD~1")
    assert out.returncode == 0, out.stdout + out.stderr
    assert "check-a covers this" in out.stdout


def test_check_b_pending_mode_is_inert_on_that_same_clean_tree(tmp_path: Path) -> None:
    """The defect `--range` exists for, pinned as a contrast. Same clean-tree fixture, default
    (pending-change) mode: the disarm is structurally unreachable, the file scan runs over the
    pushed commit's own files, and a legitimate migration commit is refused."""
    repo = _make_linear_repo(tmp_path, [18, 18], worktree_version=19, migrations=(19,), commit_pending=True)
    out = _run(repo, "check-b", "HEAD~1")
    assert out.returncode == 1
    assert "src/foo.py" in out.stderr


def test_check_b_range_mode_still_scans_when_the_pushed_commit_has_no_bump(tmp_path: Path) -> None:
    """The fail-open guard. `--range` must not become a blanket disarm: a clean tree whose
    pushed commit touches a layout-relevant file with NO bump is exactly the D78 shape, and
    pre-push is the last venue that can prevent it reaching the remote."""
    repo = _make_linear_repo(tmp_path, [18, 18], worktree_version=18, commit_pending=True)
    out = _run(repo, "check-b", "--range", "HEAD~1")
    assert out.returncode == 1
    assert "src/foo.py" in out.stderr


def test_check_b_range_mode_ignores_an_unrelated_pending_edit(tmp_path: Path) -> None:
    """The tree at pre-push is routinely dirty in files the push does not carry. It is NOT
    reverted to HEAD either: `stash = not all_files and not files` is not keyed on stage, so a
    branch update reverts UNSTAGED edits while STAGED-uncommitted ones survive, and a root-commit
    push stashes nothing -- the worktree there is neither HEAD nor the developer's tree.
    `--range` grades base..HEAD, so none of that can re-arm the pending-change conjunct. This is
    the property a tree-dirtiness INFERENCE cannot deliver: inferring the venue from
    `git diff HEAD` reads this state as pre-commit and fails again."""
    repo = _make_linear_repo(tmp_path, [18, 18], worktree_version=19, migrations=(19,), commit_pending=True)
    (repo / "tests" / "fixtures" / "legacy_layouts" / "v18" / ".keep").write_text("unrelated pending edit\n")
    assert _tracked_diff_vs_head(repo), "fixture must leave a DIRTY tree or it tests nothing"
    out = _run(repo, "check-b", "--range", "HEAD~1")
    assert out.returncode == 0, out.stdout + out.stderr


def test_check_b_range_mode_ignores_an_uncommitted_allowlist_entry(tmp_path: Path) -> None:
    """THE FALSE-PASS ARM, and it is the one whose direction matters most. The sentinel file
    supplies BOTH the governed-path set and the allowlist, and reading it from the working tree
    lets an UNCOMMITTED exemption disarm a check about a pushed commit. Reproduced at 32be36e2
    before the repair: adding the two V002x paths to the allowlist without committing turned
    `check-b HEAD~1` from FAIL into `no layout-relevant changes; pass`. The guard's own failure
    text instructs the developer to edit this exact file, so the disarming edit is the one it
    asks for."""
    repo = _make_linear_repo(tmp_path, [18, 18], worktree_version=18, commit_pending=True)
    sentinel = repo / "_layout_relevant_files.yaml"
    sentinel.write_text(
        sentinel.read_text().replace("non_breaking_allowlist: []", "non_breaking_allowlist:\n  - src/foo.py")
    )
    assert _tracked_diff_vs_head(repo), "the allowlist edit must be UNCOMMITTED or this tests nothing"
    out = _run(repo, "check-b", "--range", "HEAD~1")
    assert out.returncode == 1, out.stdout + out.stderr
    assert "src/foo.py" in out.stderr


def test_check_b_pending_mode_still_honours_an_uncommitted_allowlist_entry(tmp_path: Path) -> None:
    """The other side of the same boundary, and why the pre-commit branch must stay on the
    working tree: at commit time the pending allowlist edit IS the remedy being committed, so
    reading it from HEAD there would refuse every non-breaking exemption on the commit that
    introduces it."""
    repo = _make_linear_repo(tmp_path, [18, 18], worktree_version=18)
    sentinel = repo / "_layout_relevant_files.yaml"
    sentinel.write_text(
        sentinel.read_text().replace(
            "non_breaking_allowlist: []",
            "non_breaking_allowlist:\n  - path: src/foo.py\n    justification: baseline\n"
            '    layout_signature: "8ff88919a6004572b4249e0f72b28ed9945e61bdd0e2e7d79ddb9c5b1248674f"\n',
        )
    )
    out = _run(repo, "check-b", "HEAD~1")
    assert out.returncode == 0, out.stdout + out.stderr


def test_check_a_range_mode_ignores_an_uncommitted_bump(tmp_path: Path) -> None:
    """GAP 2's false BLOCK. Reproduced at 32be36e2 before the repair: an uncommitted bump to 23
    made `check-a HEAD~1` report `LAYOUT_VERSION jumped from 21 to 23` on a push carrying
    21->22. `constants.py` is the single most likely pending edit in a migration workflow, so
    leaving check-a on the working tree relocates the false block rather than removing it."""
    repo = _make_linear_repo(tmp_path, [18, 19], worktree_version=25, migrations=(19,), touch_layout_file=False)
    assert _tracked_diff_vs_head(repo), "the bump must be UNCOMMITTED or this tests nothing"
    out = _run(repo, "check-a", "--range", "HEAD~1")
    assert out.returncode == 0, out.stdout + out.stderr
    assert "V0019 migration + fixtures present" in out.stdout


def test_check_a_range_mode_ignores_an_uncommitted_migration_module(tmp_path: Path) -> None:
    """The false-PASS half of GAP 2, which the version read alone does not close. check-a's
    migration-module and fixture probes are FILESYSTEM reads, so at a range venue an
    UNCOMMITTED module would satisfy a check about a pushed commit. Here HEAD bumps 18->19 with
    no committed V0019; writing it into the working tree must NOT rescue the push."""
    repo = _make_linear_repo(tmp_path, [18, 19], worktree_version=19, touch_layout_file=False)
    (repo / "src" / "hhemt" / "version_migration" / "versions" / "V0019__uncommitted.py").write_text('"""m"""\n')
    out = _run(repo, "check-a", "--range", "HEAD~1")
    assert out.returncode == 1, out.stdout + out.stderr
    assert "no migration module" in out.stderr


def test_check_b_range_mode_refuses_when_the_sentinel_is_absent_at_head(tmp_path: Path) -> None:
    """The fail-closed arm for `_load_sentinel`'s ref branch. Range mode grades a COMMITTED rule
    set, so a HEAD carrying no sentinel has no rule set to grade and the only sound outcome is a
    legible refusal -- falling back to the working tree there IS the false-PASS this repair closes.

    The reachable input is a HEAD without the file, and nothing about the BASE commit reaches or
    avoids this branch, because `check_b` passes the literal "HEAD" and never the resolved base.
    An earlier reading of this branch proposed a two-commit repo whose FIRST commit omits the
    yaml; that construction loads cleanly, because the second commit restores it at HEAD. Reading
    the wrong interval is the exact defect this amendment repairs, and it surfaced inside the
    reachability analysis of the fix for it -- which is why the fixture below removes the file
    from HEAD rather than from the base.

    The pending-mode assertion is the boundary: at pre-commit the on-disk sentinel is the rule
    set being committed, so an absent-at-HEAD file must NOT refuse there."""
    repo = _make_linear_repo(tmp_path, [18, 18], worktree_version=18, touch_layout_file=False)
    sentinel = repo / "_layout_relevant_files.yaml"
    kept = sentinel.read_text()
    git = ["git", "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run([*git, "rm", "-q", "--cached", "_layout_relevant_files.yaml"], cwd=repo, check=True)
    subprocess.run([*git, "commit", "-qm", "drop the sentinel from HEAD"], cwd=repo, check=True)
    sentinel.write_text(kept)  # still on disk, now untracked and absent at HEAD

    # The fixture carries NO layout-relevant change, so a checker that reads the sentinel from
    # the working tree reaches `no layout-relevant changes; pass` and EXIT 0. The differential is
    # therefore on the exit code, which is expressible in both the pre-fix and post-fix worlds --
    # not on the refusal message, which cannot exist before the fix and would red for the wrong
    # reason.
    out = _run(repo, "check-b", "--range", "HEAD~1")
    assert out.returncode != 0, out.stdout + out.stderr
    assert "_layout_relevant_files.yaml is absent at HEAD" in out.stderr

    pending = _run(repo, "check-b", "HEAD~1")
    assert pending.returncode == 0, pending.stdout + pending.stderr
    assert "is absent at" not in pending.stderr


def test_check_b_at_the_pre_commit_base_refuses_a_version_decrease(tmp_path: Path) -> None:
    """THE DIRECTION ARM. At the pre-commit base the two disarm conjuncts compare the same pair,
    so `!=` disarms on a DECREASE as readily as on an increase -- and then prints
    `LAYOUT_VERSION bumped (19->18)`, a decrease described as a bump, on the path the
    contributor reads. `>` is what makes the disarm mean what its message says.

    Recorded so the next reader does not mis-price it: dropping this edit opens NO HOLE.
    check-a proceeds whenever `head_v != base_v` and requires exactly +1, so every decrease
    fails there, and its only blind branch (`head_v == base_v`) is precisely where check-b's
    scan runs. There is no version state in which both checks pass on a decrease. The edit is
    adopted because the guard would otherwise lie on a path that stays blocked."""
    repo = _make_linear_repo(tmp_path, [18, 19], worktree_version=18, migrations=(19,))
    out = _run(repo, "check-b", "HEAD")
    assert out.returncode == 1, out.stdout + out.stderr
    assert "src/foo.py" in out.stderr
    assert "bumped (19->18)" not in out.stdout


def test_check_b_at_the_pre_commit_base_ignores_the_previous_commits_files(tmp_path: Path) -> None:
    """THE NORMAL-COMMIT ARM. HEAD carries a layout-relevant file; the change being committed
    does not. At the pre-commit base (`HEAD`) the guard grades only what is being committed, so
    it passes.

    This arm REPLACES `test_check_b_scans_the_amended_commits_own_content`, which built a
    NORMAL commit despite its name, asserted EXIT 1 on exactly this fixture, and thereby pinned
    the defect as desired behaviour. Run the same fixture at `HEAD~1` and it returns EXIT 1,
    flagging a file whose only change is already committed -- that is the defect removed, not a
    regression, and it was refused deterministically on every commit following a bump because
    every bump ships a `versions/V*.py` under a live glob.

    Commit placement is the whole content of this fixture, so it is asserted rather than
    assumed: the layout-relevant change must be INSIDE HEAD and absent from the pending set."""
    repo = _make_linear_repo(tmp_path, [4, 4], worktree_version=4, touch_layout_file=True)
    git = ["git", "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run([*git, "add", "."], cwd=repo, check=True)
    subprocess.run([*git, "commit", "-qm", "c2 touches foo"], cwd=repo, check=True)
    (repo / "readme.md").write_text("# r\n")
    subprocess.run([*git, "add", "readme.md"], cwd=repo, check=True)

    in_head = subprocess.run(
        [*git, "show", "--name-only", "--format=", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.split()
    assert "src/foo.py" in in_head, "the layout-relevant change must sit INSIDE HEAD or this tests nothing"
    pending = subprocess.run(
        [*git, "diff", "--name-only", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.split()
    assert "src/foo.py" not in pending, "the layout-relevant change must be absent from the pending set"

    out = _run(repo, "check-b", "HEAD")
    assert out.returncode == 0, out.stdout + out.stderr

    wider = _run(repo, "check-b", "HEAD~1")
    assert wider.returncode == 1, "the wider base is what produced the defect; pinned here as a contrast"


def test_amend_dropping_a_bump_is_caught_by_check_a_and_at_pre_push(tmp_path: Path) -> None:
    """THE REAL AMEND ARM -- the one the replaced test's name claimed and its body never built.

    The amend touches a DIFFERENT file than the original commit. That is the discriminator: if
    it re-touched the same file, `HEAD..index` and `HEAD~1..index` would contain the same paths
    and the arm would prove nothing about which interval is graded.

    Three assertions, because any one alone is misleading. check-b at the pre-commit base is
    BLIND to the amended commit's own content -- asserted as documented behaviour, not as a
    defect. check-a catches it at the same base, because dropping the bump is a version
    DECREASE. And check-b at the pre-push venue catches it again once the amend has landed,
    which is what makes the pre-commit blindness safe rather than a hole."""
    repo = _make_linear_repo(tmp_path, [18, 18], worktree_version=18, touch_layout_file=False, migrations=(19,))
    git = ["git", "-c", "user.email=t@t", "-c", "user.name=t"]
    (repo / "src" / "other.py").write_text("# unrelated tracked file\n")
    subprocess.run([*git, "add", "."], cwd=repo, check=True)
    subprocess.run([*git, "commit", "-qm", "c2 baseline"], cwd=repo, check=True)

    (repo / "src" / "hhemt" / "version_migration" / "constants.py").write_text("LAYOUT_VERSION: int = 19\n")
    (repo / "src" / "foo.py").write_text("# layout change, bumped\n")
    subprocess.run([*git, "add", "."], cwd=repo, check=True)
    subprocess.run([*git, "commit", "-qm", "c3 BUMP 18->19 plus foo.py"], cwd=repo, check=True)

    # the amend: drop the bump, and touch a DIFFERENT file than c3 did
    (repo / "src" / "hhemt" / "version_migration" / "constants.py").write_text("LAYOUT_VERSION: int = 18\n")
    (repo / "src" / "other.py").write_text("# amended, a different file than the original commit\n")
    subprocess.run([*git, "add", "."], cwd=repo, check=True)

    pending = subprocess.run(
        [*git, "diff", "--name-only", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.split()
    assert "src/foo.py" not in pending, "the amend must touch a DIFFERENT file or the arm proves nothing"

    blind = _run(repo, "check-b", "HEAD")
    assert blind.returncode == 0, blind.stdout + blind.stderr

    caught = _run(repo, "check-a", "HEAD")
    assert caught.returncode == 1, caught.stdout + caught.stderr
    assert "jumped from 19 to 18" in caught.stderr

    subprocess.run([*git, "commit", "-q", "--amend", "-m", "c3 amended, bump dropped"], cwd=repo, check=True)
    pushed = _run(repo, "check-b", "--range", "HEAD~1")
    assert pushed.returncode == 1, pushed.stdout + pushed.stderr
    assert "src/foo.py" in pushed.stderr


def test_check_b_falls_back_when_the_base_ref_does_not_resolve(tmp_path: Path) -> None:
    """A repo's SECOND commit resolves HEAD but not HEAD~1. Before the base guard, the
    repaired disarm no longer short-circuited there and `git diff HEAD~1` raised an
    uncaught CalledProcessError. The guard must fall back to HEAD and still reach a
    verdict rather than traceback."""
    repo = _make_linear_repo(tmp_path, [4], worktree_version=4)
    out = _run(repo, "check-b", "HEAD~1")
    assert "Traceback" not in out.stderr
    assert "falling back to 'HEAD'" in out.stdout
    assert out.returncode == 1
    assert "src/foo.py" in out.stderr


def test_check_b_skips_on_an_unborn_head(tmp_path: Path) -> None:
    """Nothing is committed, so neither HEAD~1 nor HEAD resolves. There is no IN-REPO
    predecessor for the check to reason about, so the question has no referent here and the
    documented outcome is a skip -- not a traceback, and not a refusal, which would block
    the first commit of every fresh clone. Asserting the MESSAGE, not just the exit code,
    is deliberate: a silent 0 and a documented 0 are the same integer, and the message is
    the only thing that tells a future reader this was reasoned rather than swallowed."""
    repo = _make_linear_repo(tmp_path, [], worktree_version=4)
    out = _run(repo, "check-b", "HEAD~1")
    assert "Traceback" not in out.stderr
    assert out.returncode == 0
    assert "no in-repo predecessor" in out.stdout


def test_check_a_skips_on_an_unborn_head_too(tmp_path: Path) -> None:
    """BOTH entry points are `always_run: true`, so a root-commit skip in check-b is
    defeated by a hard failure in check-a. Before the shared guard, check-a at a root
    commit reported `FAIL - LAYOUT_VERSION jumped from 0 to 19` under EVERY base including
    the empty tree, because `base_v` swallows to 0 and the `+1` rule assumes a real
    predecessor."""
    repo = _make_linear_repo(tmp_path, [], worktree_version=19)
    out = _run(repo, "check-a", "HEAD~1")
    assert out.returncode == 0
    assert "no in-repo predecessor" in out.stdout
    assert "jumped from 0" not in out.stderr


def test_check_b_refuses_a_base_that_exists_but_is_not_a_commit(tmp_path: Path) -> None:
    """MISCONFIGURED INVOCATION, not absence of history. A tree-ish base would otherwise
    fall back silently and the check would run forever against a base nobody chose."""
    repo = _make_linear_repo(tmp_path, [4, 4], worktree_version=4)
    empty_tree = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
    out = _run(repo, "check-b", empty_tree)
    assert out.returncode == 1
    assert "does not name a commit" in out.stderr
    assert "misconfigured invocation" in out.stderr


def test_check_b_refuses_a_blob_base(tmp_path: Path) -> None:
    """Same category as the empty tree, reached by a different object kind."""
    repo = _make_linear_repo(tmp_path, [4, 4], worktree_version=4)
    blob = subprocess.run(
        ["git", "hash-object", "-w", "src/foo.py"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()
    out = _run(repo, "check-b", blob)
    assert out.returncode == 1
    assert "does not name a commit" in out.stderr


def test_check_b_accepts_both_tag_forms(tmp_path: Path) -> None:
    """THE PROOF THAT THE REFUSAL IS SAFE. Both tag forms peel to a commit, so a legitimate
    tag base is category OK and passes through unchanged -- the refusal cannot fire on it.
    If this ever goes red the categorical guard has become a liability rather than a check."""
    repo = _make_linear_repo(tmp_path, [4, 4], worktree_version=4)
    git = ["git", "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run([*git, "tag", "-a", "atag", "-m", "annotated"], cwd=repo, check=True)
    subprocess.run(["git", "tag", "ltag"], cwd=repo, check=True)
    for ref in ("atag", "ltag"):
        out = _run(repo, "check-b", ref)
        assert "does not name a commit" not in out.stderr, ref
        assert "falling back" not in out.stdout, ref


def test_layout_version_registrations_supply_the_range_predicate() -> None:
    """THE VENUE-WIRING ARM, RE-ANCHORED by the check-b venue move (chunk 2b.2).

    WHAT CHANGED, and why the previous anchor is not merely reworded. This arm used to require
    EVERY check-layout-version check at BOTH venues, pre-commit carrying no `--range` and
    pre-push carrying it. `check-b`'s obligation has MOVED to pre-push alone: a pre-commit
    obligation is graded at the moment a commit is CREATED, and a commit later replayed by
    rebase, cherry-pick or a plain merge is never created again on this machine, so its content
    was never graded -- measured 2026-10-28, an instance landed ungated by exactly that route.
    Keeping the old both-venues requirement would now assert the defect, so the invariant
    SPLITS: `check-a` still needs both venues, and `check-b` must appear at exactly ONE, at
    pre-push, with `--range` and with NO literal base ref.

    WHY THE NEW ANCHOR DISCRIMINATES ON BEHAVIOUR RATHER THAN ON WORDING. Every assertion below
    is over the parsed YAML as DATA -- stage sets and entry tokens -- and each names a state
    that changes what runs and where: re-adding `check-b` at pre-commit flips one, dropping
    `--range` flips another, and reinstating a literal base ref flips a third. None of them
    reads prose.

    The filter is `check_layout_version.py check-` and not `check-b`: scoping this arm to the
    hook that was just repaired is how the same defect survives one function over."""
    cfg = yaml.safe_load((Path(__file__).resolve().parent.parent / ".pre-commit-config.yaml").read_text())
    hooks = [
        h for repo in cfg["repos"] for h in repo["hooks"] if "check_layout_version.py check-" in h.get("entry", "")
    ]
    assert hooks, "no check_layout_version registration found at all"

    def _check_name(hook: dict) -> str:
        return hook["entry"].split("check_layout_version.py ")[1].split()[0]

    for hook in hooks:
        stages = hook.get("stages")
        assert stages is not None, f"{hook['id']} carries no explicit stages: it runs at every stage"
        assert len(stages) == 1, f"{hook['id']} declares {stages}; one registration serves exactly one venue"

    by_check: dict[str, dict[str, dict]] = {}
    for hook in hooks:
        check, venue = _check_name(hook), hook["stages"][0]
        assert venue not in by_check.setdefault(check, {}), f"{check} has two registrations at {venue}"
        by_check[check][venue] = hook
    assert set(by_check) == {"check-a", "check-b"}, f"unexpected checks wired: {sorted(by_check)}"

    # CHECK A keeps BOTH venues: this chunk moves check-b's obligation and nothing else.
    a = by_check["check-a"]
    assert set(a) == {"pre-commit", "pre-push"}, f"check-a is registered at {sorted(a)}, not both venues"
    assert "--range" not in a["pre-commit"]["entry"], (
        f"{a['pre-commit']['id']} runs at pre-commit with the range predicate"
    )
    assert "--range" in a["pre-push"]["entry"], f"{a['pre-push']['id']} runs at pre-push without the range predicate"
    assert a["pre-commit"]["entry"].split()[-1] == "HEAD", (
        f"{a['pre-commit']['id']} bases on {a['pre-commit']['entry'].split()[-1]!r}; the pre-commit venue grades "
        "the change being committed, and a wider base flags the previous commit's files"
    )
    assert a["pre-push"]["entry"].split()[-1] == "HEAD~1", (
        f"{a['pre-push']['id']} bases on {a['pre-push']['entry'].split()[-1]!r}; the pre-push check-a arm grades "
        "the pushed commit range"
    )

    # CHECK B is PRE-PUSH ONLY. Done-when 1, stated as data.
    b = by_check["check-b"]
    assert set(b) == {"pre-push"}, (
        f"check-b is registered at {sorted(b)}; its obligation moved to pre-push because a pre-commit "
        "obligation is graded when a commit is CREATED and a replayed commit never is"
    )
    b_hook = b["pre-push"]
    assert _check_name(b_hook) == "check-b"
    assert "--range" in b_hook["entry"], f"{b_hook['id']} runs the range venue without --range"
    # NO LITERAL BASE -- done-when 7. The ref pair arrives through the environment, and a
    # literal `HEAD~1` in `entry` would silently override it and restore "one commit of N".
    assert "HEAD~1" not in b_hook["entry"], (
        f"{b_hook['id']} pins a literal base in its entry ({b_hook['entry']!r}); the population must come "
        "from PRE_COMMIT_FROM_REF/PRE_COMMIT_TO_REF"
    )
    # And no pre-commit registration names check-b at all -- the move's whole point.
    pre_commit_b = [
        h
        for repo in cfg["repos"]
        for h in repo["hooks"]
        if "pre-commit" in (h.get("stages") or []) and "check_layout_version.py check-b" in h.get("entry", "")
    ]
    assert not pre_commit_b, f"check-b still runs at pre-commit via {[h['id'] for h in pre_commit_b]}"

    # DECLARING a stage does not INSTALL it. `default_install_hook_types` defaults to
    # ['pre-commit'], so without it `pre-commit install` -- the command docs/contributing.md
    # prescribes -- writes only .git/hooks/pre-commit and every pre-push declaration above is
    # inert. This assertion is a PROXY and is named as one: .git/hooks/ is untracked, so no
    # tracked-file check can observe the activation itself; the config key that DRIVES
    # installation is the closest observable. It does not catch a clone where `pre-commit
    # install` was never run.
    # Range over the WHOLE config, not over `hooks`. `hooks` is the layout-version filter, so
    # scoping `declared` to it would protect only the four registrations this repair touched and
    # leave `anonymization-guard-commit-msg` -- one of the two guards the key exists for --
    # unasserted. `.get("stages") or []` is deliberate: a hook with no key inherits
    # `default_stages`, which is all eleven stages, and no install list can cover eleven. The
    # question is whether every stage someone EXPLICITLY asked for is installable.
    #
    # `manual` is excluded because it is declarable but NOT installable, by upstream design:
    # `stages:` validates against STAGES (11 values) and `default_install_hook_types` against
    # HOOK_TYPES (10), and clientlib.py's own comment on the difference reads "`manual` is not
    # invoked by any installed git hook." Without this exclusion a legitimate `stages: [manual]`
    # hook makes this assertion permanently red with a remedy `pre-commit validate-config`
    # refuses -- the over-firing twin of the permanently-green assertion.
    declared = {stage for repo in cfg["repos"] for hook in repo["hooks"] for stage in (hook.get("stages") or [])}
    declared -= {"manual"}
    installed = set(cfg.get("default_install_hook_types") or ["pre-commit"])
    missing = declared - installed
    assert not missing, (
        f"stages {sorted(missing)} are declared but `pre-commit install` does not install them; "
        "add them to default_install_hook_types or the guard is silently inert"
    )


def test_check_b_refuses_a_path_that_resolves_to_no_layout_signature(tmp_path: Path) -> None:
    """The fail-open this grammar closed. A row without a layout_signature exempted its path
    from Check B forever, content-independently, and nothing in the yaml said so. The loader
    now refuses at load and names every offending path."""
    repo = _make_repo(
        tmp_path,
        layout_version_at_head=4,
        layout_version_at_main=4,
        extra_files_at_main={"src/foo.py": "# initial\n"},
        extra_files_at_head={"src/foo.py": "# changed\n"},
        sentinel_yaml=(
            "layout_relevant:\n  paths:\n    - src/foo.py\n  globs: []\nnon_breaking_allowlist:\n"
            "  - path: src/foo.py\n    justification: baseline\n"
        ),
    )
    out = _run(repo, "check-b", "main")
    assert out.returncode != 0, out.stdout + out.stderr
    assert "src/foo.py" in out.stderr
    assert "exempt from Check B forever" in out.stderr, out.stdout + out.stderr


def test_check_b_refires_when_the_allowlist_signature_no_longer_matches(tmp_path: Path) -> None:
    """The drift arm: a stamped exemption covers the content it was stamped over and nothing
    else, so a later change to the same file re-fires Check B."""
    repo = _make_repo(
        tmp_path,
        layout_version_at_head=4,
        layout_version_at_main=4,
        extra_files_at_main={"src/foo.py": "# initial\n"},
        extra_files_at_head={"src/foo.py": "# changed\n"},
        sentinel_yaml=(
            "layout_relevant:\n  paths:\n    - src/foo.py\n  globs: []\nnon_breaking_allowlist:\n"
            "  - path: src/foo.py\n    justification: baseline\n"
            '    layout_signature: "' + "deadbeef" * 8 + '"\n'
        ),
    )
    out = _run(repo, "check-b", "main")
    assert out.returncode == 1, out.stdout + out.stderr
    assert "src/foo.py" in out.stderr


def test_check_b_refuses_a_bare_string_allowlist_row(tmp_path: Path) -> None:
    """The bare-string form cannot carry a signature at all, so it IS a permanence-expressing
    form and the gate must refuse it AT LOAD. This is the arm that dies first if the gate's
    predicate is ever narrowed to the mapping form -- measured: adding `and e.justification is
    not None` makes the loader ACCEPT this row. It asserts on the loader's own refusal marker
    rather than on the exit code, because spec 9's fail-closed fallback ALSO produces a
    non-zero exit for this input -- so an exit-code assertion cannot tell a refused load from
    a flagged file, and measured, it cannot even tell the gate's TOTAL ABSENCE (gate deleted:
    the whole module still passed). The marker is emitted only by the loader raise."""
    repo = _make_repo(
        tmp_path,
        layout_version_at_head=4,
        layout_version_at_main=4,
        extra_files_at_main={"src/foo.py": "# initial\n"},
        extra_files_at_head={"src/foo.py": "# changed\n"},
        sentinel_yaml=(
            "layout_relevant:\n  paths:\n    - src/foo.py\n  globs: []\nnon_breaking_allowlist:\n  - src/foo.py\n"
        ),
    )
    out = _run(repo, "check-b", "main")
    assert out.returncode != 0, out.stdout + out.stderr
    assert "src/foo.py" in out.stderr
    assert "exempt from Check B forever" in out.stderr, out.stdout + out.stderr


def test_check_b_honours_an_appended_row_that_inherits_the_prior_signature(tmp_path: Path) -> None:
    """The append-log property the baseline depends on, in the direction phase B actually uses:
    TWO rows for one path, the first OMITTING the signature and the second SUPPLYING it, must
    RESOLVE to the SECOND row's signature and stay exempt. The fixture uses phase B's
    ACTUAL row order -- the historical signature-less row
    first, the appended baseline row second -- because that order is what makes the gate's
    placement observable. Measured: moving the gate into the per-row loop AFTER the
    carry-forward makes this legal pair raise and refuses the real 68-row baselined sentinel
    outright, and tests 3 and 4 are the only tests that notice. The PRE-merge placement is
    caught more widely -- tests 3, 4 and 5 all fail on it -- so this fixture is not the sole
    witness there."""
    repo = _make_repo(
        tmp_path,
        layout_version_at_head=4,
        layout_version_at_main=4,
        extra_files_at_main={"src/foo.py": "# initial\n"},
        extra_files_at_head={"src/foo.py": "# changed\n"},
        sentinel_yaml=(
            "layout_relevant:\n  paths:\n    - src/foo.py\n  globs: []\nnon_breaking_allowlist:\n"
            "  - path: src/foo.py\n    justification: historical row, no signature\n"
            "  - path: src/foo.py\n    justification: baseline appended by phase B\n"
            '    layout_signature: "8ff88919a6004572b4249e0f72b28ed9945e61bdd0e2e7d79ddb9c5b1248674f"\n'
        ),
    )
    out = _run(repo, "check-b", "main")
    assert out.returncode == 0, out.stdout + out.stderr


def test_check_b_honours_an_omitting_append_after_a_signature_row(tmp_path: Path) -> None:
    """The OTHER carry-forward direction, and it is a live operator move rather than a
    symmetry exercise: spec 5's printed remedy tells an operator that omitting the field on an
    appended row inherits the path's previous signature, so `sig`-then-`omit` must resolve to
    that signature. Measured: deleting the three-line carry-forward merge leaves the
    `omit`-then-`sig` fixture above passing (plain last-wins still supplies the digest) and
    breaks ONLY this direction, so one two-row fixture cannot cover both and both are live."""
    repo = _make_repo(
        tmp_path,
        layout_version_at_head=4,
        layout_version_at_main=4,
        extra_files_at_main={"src/foo.py": "# initial\n"},
        extra_files_at_head={"src/foo.py": "# changed\n"},
        sentinel_yaml=(
            "layout_relevant:\n  paths:\n    - src/foo.py\n  globs: []\nnon_breaking_allowlist:\n"
            "  - path: src/foo.py\n    justification: signature row\n"
            '    layout_signature: "8ff88919a6004572b4249e0f72b28ed9945e61bdd0e2e7d79ddb9c5b1248674f"\n'
            "  - path: src/foo.py\n    justification: later append, omits and inherits\n"
        ),
    )
    out = _run(repo, "check-b", "main")
    assert out.returncode == 0, out.stdout + out.stderr


# --- `--stamp`: the non-breaking re-declaration as a command -----------------
# The layout gate forces every edit to a layout-relevant module to be re-declared
# non-breaking with a hand-copied 64-character digest. `--stamp` makes that copy a
# command. Four properties are each silent if wrong, so each gets its own arm:
#   1. the appended row must PARSE, RESOLVE through the loader, and carry `--why`;
#   2. a missing/empty `--why` must refuse rather than write, because a mapping row with
#      no non-empty justification is refused by `_load_allowlist` at load -- which takes
#      check-b AND check-c offline install-wide until a human edits the YAML by hand;
#   3. the edit must be TEXTUAL, because the manifest is 4,400+ lines carrying 232
#      `justification:` blocks plus a header comment and a `yaml` round-trip destroys them;
#   4. it must APPEND and never mutate a pre-existing row in place, because an in-place
#      `layout_signature` rewrite leaves that row's justification describing an earlier
#      change -- the gate passes while the audit trail silently lies.

# Deliberately NOT the digest of the content `--stamp` will hash. When the old and new
# digests coincide, "the old row's signature is unchanged" is indistinguishable from "the
# old row was mutated to the same value", and every arm below asserts the INEQUALITY before
# it asserts the equality for exactly that reason.
_STAMP_EARLIER_SIGNATURE = "60a51f748ab5fa1150fdfc5639de2ad00d96e367574820fea00d1219f1d8a0e4"

# A manifest with the three things a `yaml.safe_dump` implementation would destroy: a
# header comment, a FOUR-LINE `justification:` block scalar, and a pre-existing row for the
# same path `--stamp` is asked to stamp (which is what makes append-vs-mutate observable).
_STAMP_MANIFEST = (
    "# Sentinel list for scripts/check_layout_version.py check-b.\n"
    "# A change to any path below WITHOUT a LAYOUT_VERSION bump fails CI.\n"
    "layout_relevant:\n"
    "  paths:\n"
    "    - src/foo.py\n"
    "  globs: []\n"
    "non_breaking_allowlist:\n"
    "  - path: src/foo.py\n"
    f"    layout_signature: {_STAMP_EARLIER_SIGNATURE}\n"
    "    justification: >-\n"
    "      EARLIER CHANGE. This row names a change that already landed, and its\n"
    "      justification spans four folded lines on purpose: a yaml.safe_dump\n"
    "      round-trip reflows it onto one, and the gate must leave it exactly as\n"
    "      written.\n"
)

_STAMP_CHANGED_CONTENT = "# changed\n"


def _make_stamp_repo(
    tmp_path: Path,
    *,
    manifest: str = _STAMP_MANIFEST,
    files: dict[str, str] | None = None,
) -> Path:
    """A repo-shaped fixture for `--stamp`, with NO git.

    `--stamp` is pure file-in/file-out -- no solver, no network, no git -- so the fixture
    needs none either. `_run` still copies the checker into `<repo>/scripts/`, and
    `REPO_ROOT = Path(__file__).resolve().parent.parent` then resolves to this repo, which
    is what makes the fixture manifest THE manifest the command reads.
    """
    for rel, content in (files if files is not None else {"src/foo.py": _STAMP_CHANGED_CONTENT}).items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    (tmp_path / "_layout_relevant_files.yaml").write_text(manifest)
    return tmp_path


def _stamp_manifest_text(repo: Path) -> str:
    return (repo / "_layout_relevant_files.yaml").read_text()


def _rows_for(parsed: dict, rel: str) -> list[dict]:
    return [row for row in (parsed.get("non_breaking_allowlist") or []) if row.get("path") == rel]


def test_stamp_appends_a_row_that_parses_resolves_and_carries_the_why(tmp_path: Path) -> None:
    """Done-when 1. RED pre-fix, measured: `--stamp` does not exist, so argparse rejects
    `src/foo.py` as an invalid `cmd` choice and the process exits 2 having written nothing.

    Three assertions, and the third is the one that makes the other two mean something: a
    row can parse and still be UNUSABLE by the gate, so the row is fed through the real
    `_load_allowlist` and must RESOLVE to the digest that was written. `--why` carries a
    colon and a `#` to prove the folded block scalar needs no escaping -- a quoted-scalar
    implementation would have to escape them and a broken one would misparse."""
    repo = _make_stamp_repo(tmp_path)
    why = "NON-BREAKING. Comment-only edit: no on-disk change; see #123 for context."
    out = _run(repo, "--stamp", "src/foo.py", "--why", why)
    assert out.returncode == 0, out.stdout + out.stderr

    parsed = yaml.safe_load(_stamp_manifest_text(repo))
    expected_sig = hashlib.sha256((repo / "src" / "foo.py").read_bytes()).hexdigest()
    last = parsed["non_breaking_allowlist"][-1]
    assert last["path"] == "src/foo.py"
    assert last["layout_signature"] == expected_sig
    assert last["justification"] == why

    # RESOLVES through the loader the gate actually uses -- this is what "useful row" means.
    module = _load_check_module()
    resolved = module._load_allowlist(parsed)
    assert resolved["src/foo.py"].layout_signature == expected_sig


def test_stamp_refuses_without_a_why_and_writes_nothing(tmp_path: Path) -> None:
    """Done-when 2. The measured consequence, stated because it is why this is a refusal
    rather than a defaulted justification: `_load_allowlist` refuses a mapping row with no
    non-empty `justification` BEFORE any signature logic runs, so a signature-only append
    takes check-b and check-c offline AT LOAD, install-wide, until a human edits the YAML.

    Asserting the refusal's own marker as well as the exit code is deliberate: pre-fix,
    an unknown `--stamp` also exits non-zero (argparse, code 2), so an exit-code assertion
    alone would pass on the defect this arm exists to catch. The empty and
    whitespace-only forms are the same refusal for the same reason and are asserted
    through the same marker."""
    repo = _make_stamp_repo(tmp_path)
    before = _stamp_manifest_text(repo)

    absent = _run(repo, "--stamp", "src/foo.py")
    assert absent.returncode != 0
    assert "non-empty --why justification" in absent.stderr, absent.stderr
    assert _stamp_manifest_text(repo) == before, "a refused stamp must write nothing"

    for empty in ("", "   \n\t "):
        blank = _run(repo, "--stamp", "src/foo.py", "--why", empty)
        assert blank.returncode != 0, repr(empty)
        assert "non-empty --why justification" in blank.stderr, blank.stderr
        assert _stamp_manifest_text(repo) == before, f"a refused stamp must write nothing ({empty!r})"


def test_stamp_edits_textually_preserving_comments_and_block_scalars(tmp_path: Path) -> None:
    """Done-when 3. THE ARM THAT CATCHES A `yaml.safe_dump` IMPLEMENTATION.

    `startswith(before)` is the whole claim in one expression and it is exactly right given
    append-at-EOF: every byte the command did not add is byte-identical, so the header
    comment cannot be dropped and the four-line block scalar cannot be reflowed. The
    explicit re-assertions of the comment and the block scalar follow it so a failure names
    WHICH of the two a round-trip destroyed instead of only that the prefix moved."""
    repo = _make_stamp_repo(tmp_path)
    before = _stamp_manifest_text(repo)

    out = _run(repo, "--stamp", "src/foo.py", "--why", "NON-BREAKING. Second change.")
    assert out.returncode == 0, out.stdout + out.stderr
    after = _stamp_manifest_text(repo)

    assert after.startswith(before), "everything before the appended row must be byte-identical"
    assert "# Sentinel list for scripts/check_layout_version.py check-b.\n" in after
    assert "# A change to any path below WITHOUT a LAYOUT_VERSION bump fails CI.\n" in after
    assert (
        "      EARLIER CHANGE. This row names a change that already landed, and its\n"
        "      justification spans four folded lines on purpose: a yaml.safe_dump\n"
        "      round-trip reflows it onto one, and the gate must leave it exactly as\n"
        "      written.\n"
    ) in after, "a yaml round-trip would have reflowed this block scalar onto one line"


def test_stamp_appends_rather_than_mutating_the_existing_row(tmp_path: Path) -> None:
    """Done-when 4. THE ARM THAT DISTINGUISHES APPEND FROM MUTATE, which is a distinct
    failure from "does not append at all": a builder can append correctly and still rewrite
    the older row's digest, and that failure is SILENT -- the gate passes, because the
    newest row carries a matching digest, while the older row's justification now describes
    a change its signature no longer covers.

    Two structures do the discriminating, and neither is sufficient alone. The ROW COUNT
    (2, not 1) fails a pure in-place mutation. The DIGEST INEQUALITY assertion before the
    equality assertions fails a mutation that also appends. The text-level assertions then
    pin the pre-existing row's two fields BYTE-identically rather than by parsed value, so a
    reformatted-but-equal rewrite is caught too."""
    repo = _make_stamp_repo(tmp_path)
    before = _stamp_manifest_text(repo)
    earlier_justification = (
        "EARLIER CHANGE. This row names a change that already landed, and its "
        "justification spans four folded lines on purpose: a yaml.safe_dump "
        "round-trip reflows it onto one, and the gate must leave it exactly as written."
    )

    out = _run(repo, "--stamp", "src/foo.py", "--why", "NON-BREAKING. The second change to this path.")
    assert out.returncode == 0, out.stdout + out.stderr
    after = _stamp_manifest_text(repo)
    parsed = yaml.safe_load(after)

    rows = _rows_for(parsed, "src/foo.py")
    assert len(rows) == 2, f"expected an APPENDED row beside the pre-existing one, got {len(rows)} row(s)"

    new_sig = hashlib.sha256((repo / "src" / "foo.py").read_bytes()).hexdigest()
    # NON-VACUITY: if these were equal, "the old digest is unchanged" could not be told
    # apart from "the old row was mutated to the new value".
    assert new_sig != _STAMP_EARLIER_SIGNATURE, "fixture degenerate: earlier and current digests coincide"

    assert rows[0]["layout_signature"] == _STAMP_EARLIER_SIGNATURE
    assert rows[0]["justification"] == earlier_justification
    assert rows[1]["layout_signature"] == new_sig

    # BYTE-identical, not merely value-identical: the pre-existing row's two fields keep the
    # exact text they had, which `before` still contains verbatim.
    assert after.startswith(before)
    assert f"    layout_signature: {_STAMP_EARLIER_SIGNATURE}\n" in after


def test_stamp_clears_the_check_b_failure_it_is_the_remedy_for(tmp_path: Path) -> None:
    """The integration arm, and the one the developer feels: check-b FAILS, `--stamp` runs,
    check-b PASSES. A unit arm that only parsed the row would not prove the row is the
    remedy; this one proves the whole chain through the gate itself.

    The manifest must be a BLOCK sequence (the key with an empty value, as every real
    manifest has it) rather than the `[]` flow form `_make_repo` defaults to, because
    `--stamp` appends a block item and refuses to write text that would not parse."""
    repo = _make_repo(
        tmp_path,
        layout_version_at_head=4,
        layout_version_at_main=4,
        extra_files_at_main={"src/foo.py": "# initial\n"},
        extra_files_at_head={"src/foo.py": _STAMP_CHANGED_CONTENT},
        sentinel_yaml=("layout_relevant:\n  paths:\n    - src/foo.py\n  globs: []\nnon_breaking_allowlist:\n"),
    )

    before = _run(repo, "check-b", "main")
    assert before.returncode == 1, before.stdout + before.stderr
    assert "src/foo.py" in before.stderr

    stamped = _run(repo, "--stamp", "src/foo.py", "--why", "NON-BREAKING. Comment-only; no on-disk change.")
    assert stamped.returncode == 0, stamped.stdout + stamped.stderr

    after = _run(repo, "check-b", "main")
    assert after.returncode == 0, after.stdout + after.stderr
    assert "no layout-relevant changes" in after.stdout


def test_stamp_records_a_multiline_why_verbatim(tmp_path: Path) -> None:
    """Reviewer finding F1, arm of the coder's own.

    A FOLDED block scalar (`>-`) turns every newline into a space and every blank line into
    one newline, so `--why $'A\\nB'` was recorded as `A B`: a DIFFERENT string from the one
    supplied, silently, and still non-empty, so `_load_allowlist` accepts it and nothing
    downstream reports the loss. The justification is the whole audit trail a human reads
    when adjudicating a change-scoped exemption, so a bullet list run together into one line
    destroys exactly what the row exists to carry.

    Four inputs, because they fail under a folded scalar in three DIFFERENT ways and a
    single-line arm cannot see any of them: a plain two-liner folds to `A B`; a BLANK line
    folds to a newline rather than a paragraph break (`A\\n\\nB` would come back `A\\nB`);
    a bullet list loses its structure entirely; and a trailing-space line exercises the
    renderer's deliberate lack of `.rstrip()`. The last one is a control that would pass
    vacuously under a renderer that rstripped, so it is asserted rather than omitted.
    """
    cases = [
        "NON-BREAKING. Line one.\nLine two of the same paragraph.",
        "NON-BREAKING. Paragraph one.\n\nParagraph two after a blank line.",
        "NON-BREAKING.\n- no on-disk name, path or group moves\n- the signature re-fires on the next change",
        "NON-BREAKING. Trailing spaces are content   \nand this line follows it",
    ]
    for i, why in enumerate(cases):
        repo = _make_stamp_repo(tmp_path / f"case{i}")
        out = _run(repo, "--stamp", "src/foo.py", "--why", why)
        assert out.returncode == 0, out.stdout + out.stderr
        recorded = yaml.safe_load(_stamp_manifest_text(repo))["non_breaking_allowlist"][-1]["justification"]
        assert recorded == why, f"case {i}: the recorded justification is not the text supplied"


def test_stamp_refuses_to_stamp_the_manifest_itself(tmp_path: Path) -> None:
    """Reviewer finding F2, arm of the coder's own.

    `--stamp` records `sha256(target)` and then APPENDS to the manifest. When the target IS
    the manifest, the append mutates the very file whose digest was just taken, so the row
    is stale the instant it is written: the command prints `stamped ...`, exits 0, and
    `check-b` stays red on exactly the path it was asked to fix.

    The repair is a REFUSAL rather than an append-then-re-hash, and the ground is arithmetic
    rather than convenience: the row carries the digest of a document that CONTAINS that
    digest, so a correct row would need `S == sha256(document containing S)` -- a fixed point
    of sha256 over a structure holding its own output, which cannot be found. That is why a
    second pass cannot pay the staleness down; measured before this fix, a first run recorded
    `6a487c62...` against an actual `72787b6e...` and a second recorded `72787b6e...` against
    an actual `4863555c...`, exit 0 both times.

    The CONTROL half is the must-pass arm: an ordinary target still stamps, so the refusal is
    scoped to the self-target rather than being a blanket refusal that would make this test
    green for the wrong reason. The repeat arm pins the non-convergence claim itself.
    """
    repo = _make_stamp_repo(tmp_path)
    before = _stamp_manifest_text(repo)

    for attempt, why in enumerate(("NON-BREAKING. The manifest itself.", "NON-BREAKING. Trying again."), start=1):
        out = _run(repo, "--stamp", "_layout_relevant_files.yaml", "--why", why)
        assert out.returncode != 0, f"attempt {attempt}: a self-stamp must be refused"
        assert "refuses to stamp" in out.stderr, out.stderr
        assert _stamp_manifest_text(repo) == before, f"attempt {attempt}: a refused stamp must write nothing"

    # CONTROL — the same command against an ordinary target still succeeds, so the arm above
    # is evidence about the self-target and not about a broken command.
    control = _run(repo, "--stamp", "src/foo.py", "--why", "NON-BREAKING. Control arm.")
    assert control.returncode == 0, control.stdout + control.stderr
    assert _stamp_manifest_text(repo) != before


# --- Chunk 2b.2: the venue move, the D151 gate, and the range closure --------
# `check-b`'s obligation moved from pre-commit to pre-push. Three things had to land
# together, and each has an arm below:
#   * the venue itself (done-when 1/4/7) -- asserted as DATA from the YAML, because a test
#     that observed a real push could not exist and "we observed a push" is not a test;
#   * the D151 gate (done-when 2) -- at the range venue the exemption arm must read the blob
#     at the GRADED COMMIT, never the worktree;
#   * the range closure (done-when 5/6) -- a PER-COMMIT WALK, so one bump cannot disarm the
#     scan of any other commit in the pushed range.


def _make_history_repo(
    tmp_path: Path,
    specs: list[dict],
    *,
    sentinel_yaml: str | None = None,
    branch_after: int | None = None,
) -> tuple[Path, list[str]]:
    """A LINEAR fixture repo with ONE COMMIT PER SPEC, oldest first; returns (root, shas).

    Each spec is `{"version": int, "files": {relpath: content}}`. Sibling builder rather than a
    parameter on `_make_linear_repo`, which drives the single-commit disarm arms and would have
    to grow a second meaning. The shas are returned because the multi-commit arms assert
    WHICH commit was flagged, and `HEAD~2` arithmetic in an assertion is how a fixture silently
    stops testing what it says.

    `branch_after=N` commits the first N specs on `main` and the rest on `feature`, which is
    what lets an arm tell "grade one commit (HEAD~1..HEAD)" from "grade the whole branch
    (main..HEAD)". On a fixture where every commit sits on `main` those two populations are the
    SAME and the arm cannot discriminate -- measured, that is exactly how the first version of
    the population-default arm passed against the defect it was written for.
    """
    (tmp_path / "src" / "hhemt" / "version_migration" / "versions").mkdir(parents=True, exist_ok=True)
    (tmp_path / "tests" / "fixtures" / "legacy_layouts").mkdir(parents=True, exist_ok=True)
    (tmp_path / "_layout_relevant_files.yaml").write_text(
        sentinel_yaml or "layout_relevant:\n  paths:\n    - src/foo.py\n  globs: []\nnon_breaking_allowlist:\n"
    )
    constants = tmp_path / "src" / "hhemt" / "version_migration" / "constants.py"
    git = ["git", "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    shas: list[str] = []
    for i, spec in enumerate(specs):
        if branch_after is not None and i == branch_after:
            subprocess.run(["git", "checkout", "-q", "-b", "feature"], cwd=tmp_path, check=True)
        constants.write_text(f"LAYOUT_VERSION: int = {spec['version']}\n")
        for rel, content in spec.get("files", {}).items():
            p = tmp_path / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
        subprocess.run([*git, "add", "."], cwd=tmp_path, check=True)
        # `--allow-empty` so a spec that intentionally changes nothing (a version hold, or a
        # commit whose only content is already in place) still produces the commit the range
        # arithmetic in the caller depends on. The sibling `_make_linear_repo` does the same.
        subprocess.run([*git, "commit", "--allow-empty", "-qm", f"c{i}"], cwd=tmp_path, check=True)
        shas.append(
            subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=tmp_path, capture_output=True, text=True, check=True
            ).stdout.strip()
        )
    return tmp_path, shas


def _load_check_module_at(repo: Path):
    """Load the checker THE FIXTURE CARRIES, so `REPO_ROOT` resolves to the fixture.

    `_load_check_module` above loads from the real repo, which is right for arms about the real
    sentinel and wrong for an arm that must monkeypatch a function and then drive `check_b`
    against a `tmp_path` repo.
    """
    target = repo / "scripts" / "check_layout_version.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(SCRIPT.read_text())
    name = f"check_layout_version_fixture_{abs(hash(str(repo)))}"
    spec = importlib.util.spec_from_file_location(name, target)
    module = importlib.util.module_from_spec(spec)
    # Register before exec: `from __future__ import annotations` makes the dataclass resolve
    # `sys.modules[cls.__module__]`, and None there raises AttributeError at class definition.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_check_b_range_satisfies_the_allowlist_from_the_graded_commit_not_the_worktree(tmp_path: Path) -> None:
    """Done-when 2 — THE D151 GATE.

    At the range venue the change-scoped exemption arm must resolve the file's content from the
    GRADED COMMIT, never from the worktree. The worktree at pre-push is neither HEAD nor the
    developer's tree (`stash = not all_files and not files` is not keyed on stage, so unstaged
    edits are reverted for the hook's duration while staged-uncommitted ones survive, and a
    root-commit push stashes nothing), so a worktree read there compares a signature against
    content nobody is pushing.

    THE INSTRUMENT IS A MONKEYPATCH THAT RAISES, not an assertion about which function was
    called: a call-count would pass on a call that was made and ignored. The CONTROL arm runs
    the SAME patch through the pending-change venue and REQUIRES the raise, which is what proves
    the patch is live -- without it, the range arm would "hold" on a function that was never
    reachable in either venue. The property varied between the two arms is the VENUE; the
    fixture, the patch and the allowlist row are identical.
    """
    content_at_head = "# v1\n"
    sentinel = (
        "layout_relevant:\n  paths:\n    - src/foo.py\n  globs: []\nnon_breaking_allowlist:\n"
        "  - path: src/foo.py\n    justification: baseline\n"
        f'    layout_signature: "{hashlib.sha256(content_at_head.encode()).hexdigest()}"\n'
    )
    repo, _ = _make_history_repo(
        tmp_path,
        [
            {"version": 4, "files": {"src/foo.py": "# v0\n"}},
            {"version": 4, "files": {"src/foo.py": content_at_head}},
        ],
        sentinel_yaml=sentinel,
    )
    module = _load_check_module_at(repo)

    def _explode(_path):
        raise AssertionError("the WORKTREE hash was read at the range venue (D151)")

    original = module._file_content_hash
    module._file_content_hash = _explode
    try:
        # The exemption signature matches the content AT HEAD, so the range venue must exempt
        # the change by reading the committed blob.
        assert module.check_b("HEAD~1", range_mode=True, to_ref="HEAD") == 0

        # CONTROL: the same patch, the pending-change venue. It MUST explode, which is what
        # makes the arm above evidence rather than a vacuous pass.
        raised = False
        try:
            module.check_b("HEAD~1", range_mode=False)
        except AssertionError as exc:
            raised = "D151" in str(exc)
        assert raised, "the control arm did not exercise the worktree read; the patch is not live"
    finally:
        module._file_content_hash = original


def test_check_b_range_grades_every_commit_in_the_pushed_range(tmp_path: Path) -> None:
    """Done-when 5 — THE POPULATION.

    A multi-commit range in which only the MIDDLE commit bumps. Every commit must be graded.

    RED pre-fix, measured against the `HEAD` blob of the checker: the single range-wide arm
    compared the range's two ENDPOINTS, saw a difference, printed `bumped; check-a covers this`
    and returned 0 BEFORE the file scan -- so both failing commits rode through. That is why a
    population half alone is not a closure.

    The middle commit is ALSO asserted to be reported as covered, so a walk that simply skipped
    bumped commits and graded nothing else could not pass this arm.
    """
    repo, shas = _make_history_repo(
        tmp_path,
        [
            {"version": 4, "files": {"src/foo.py": "# v0\n"}},  # base
            {"version": 4, "files": {"src/foo.py": "# v1\n"}},  # touches, no bump -> FAIL
            {"version": 5, "files": {}},  # BUMP -> covered
            {"version": 5, "files": {"src/foo.py": "# v3\n"}},  # touches, no bump -> FAIL
        ],
    )
    out = _run(repo, "check-b", "--range", "HEAD~3", "--to", "HEAD")
    assert out.returncode == 1, out.stdout + out.stderr
    assert "src/foo.py" in out.stderr
    # BOTH failing commits, by short sha -- one failure would mean the walk stopped early.
    assert shas[1][:9] in out.stderr, out.stderr
    assert shas[3][:9] in out.stderr, out.stderr
    assert out.stderr.count("src/foo.py") == 2, out.stderr
    # And the bump was SEEN, and did not disarm either scan.
    assert "check-a covers this" in out.stdout, out.stdout


def test_check_b_range_per_commit_bump_does_not_disarm_an_earlier_commit(tmp_path: Path) -> None:
    """Done-when 6 — THE PER-COMMIT WALK'S DISCRIMINATION.

    A three-commit graded range whose FIRST commit touches a layout-relevant path with no
    allowlist row, and whose THIRD commits a bump. The FIRST must be reported, and the bump must
    not be permitted to disarm it.

    RED pre-fix: the range arm returned PASS at the first bump it saw, because the bump was read
    from the range ENDPOINT rather than from the commit being graded. This arm is the one that
    distinguishes a walk from an endpoint comparison; the population arm above would still pass
    under a walk that graded only the first and last commits.
    """
    repo, shas = _make_history_repo(
        tmp_path,
        [
            {"version": 4, "files": {"src/foo.py": "# v0\n"}},  # base (not graded)
            {"version": 4, "files": {"src/foo.py": "# v1\n"}},  # FIRST graded: touches, no bump
            {"version": 4, "files": {"src/unrelated.py": "# noise\n"}},  # middle: ungoverned change
            {"version": 5, "files": {}},  # THIRD graded: the bump
        ],
    )
    out = _run(repo, "check-b", "--range", "HEAD~3", "--to", "HEAD")
    assert out.returncode == 1, out.stdout + out.stderr
    assert shas[1][:9] in out.stderr, f"the FIRST graded commit was not reported: {out.stderr}"
    assert "check-a covers this" in out.stdout, "the bump was not reached, so it was not tested against"


def test_check_b_range_population_defaults_to_one_commit_and_reads_the_env_pair(tmp_path: Path) -> None:
    """Done-when 7 — THE SAFE DEFAULT, and the population coming from the environment.

    THE FIXTURE BRANCHES ON PURPOSE, and the first version of this arm did not. It must be able
    to tell "grade one commit (HEAD~1..HEAD)" from "grade the whole branch (main..HEAD)": when
    every commit sits on `main` those are the SAME population and both the correct default and
    the refuted wide-base default return the same verdict. Measured -- on a single-branch
    fixture this arm passed against the defect, so `branch_after` exists.

    RED pre-fix, measured against the `HEAD` blob: with the ref pair unset and no positional
    base, the old code fell back to `main`, graded `main..HEAD`, found the governed change one
    commit back, and returned 1. The safe default returns 0 because it grades only HEAD.

    Three arms: the DEFAULT (one commit, no crash), the ENV PAIR (honoured, and reaching the
    failing commit), and a MUST-FIRE CONTROL naming the wide base explicitly, which must
    reproduce the failure -- so a default that "passed" on a fixture that could not fail is
    excluded.
    """
    repo, shas = _make_history_repo(
        tmp_path,
        [
            {"version": 4, "files": {"src/foo.py": "# v0\n"}},  # c0 on main
            {"version": 4, "files": {"src/foo.py": "# v1\n"}},  # c1 on feature: governed, no bump
            {"version": 4, "files": {"src/unrelated.py": "# noise\n"}},  # c2 = HEAD: ungoverned
        ],
        branch_after=1,
    )
    on_branch = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()
    assert on_branch == "feature", (
        f"the fixture is on {on_branch!r}, not 'feature'; with every commit on main the wide-base "
        "control proves nothing"
    )
    # Explicitly BLANK rather than merely absent: an ambient ref pair in the outer environment
    # would otherwise silently become the population under test.
    unset = {"PRE_COMMIT_FROM_REF": "", "PRE_COMMIT_TO_REF": ""}
    default_run = _run(repo, "check-b", "--range", env=unset)
    assert "Traceback" not in default_run.stderr, default_run.stderr
    assert default_run.returncode == 0, (
        "the unset default must grade ONE commit (HEAD~1..HEAD, the ungoverned one) and pass; "
        + default_run.stdout
        + default_run.stderr
    )

    # MUST-FIRE CONTROL: the wide base the default avoids must actually FAIL on this fixture.
    wide = _run(repo, "check-b", "--range", "main", env=unset)
    assert wide.returncode == 1, "the wide base returned PASS, so this fixture cannot discriminate"
    assert shas[1][:9] in wide.stderr, wide.stderr

    env_run = _run(repo, "check-b", "--range", env={"PRE_COMMIT_FROM_REF": "HEAD~2", "PRE_COMMIT_TO_REF": "HEAD"})
    assert env_run.returncode == 1, "the env-supplied population did not reach the failing commit"
    assert shas[1][:9] in env_run.stderr, env_run.stderr


def test_check_b_range_treats_a_null_from_ref_as_absent_history_not_a_misconfiguration(tmp_path: Path) -> None:
    """The NEW-BRANCH PUSH, which the population half made reachable for the first time.

    Before this chunk the pre-push entry carried a literal `HEAD~1`, which resolves on any
    branch. Now the population comes from `PRE_COMMIT_FROM_REF`, and pre-commit's sentinel for
    "the remote has no such ref yet" is forty zeroes. `git rev-parse --verify 0000.../40`
    SUCCEEDS (git accepts the null object id) and the `^{commit}` probe then fails, so the
    ordinary two-probe sequence classifies a null ref as EXISTS-NOT-COMMIT and REFUSES the push
    with `misconfigured invocation` -- blocking a legitimate first push of a branch, on the one
    check whose obligation is to grade what leaves the machine.

    Two arms: the null ref must reach a VERDICT rather than the refusal (here: nothing graded,
    because the range collapses), and the CONTROL asserts the refusal is still live for a
    genuinely non-commit object -- so the fix cannot have been "delete the refusal".
    """
    repo, _ = _make_history_repo(
        tmp_path,
        [
            {"version": 4, "files": {"src/foo.py": "# v0\n"}},
            {"version": 4, "files": {"src/foo.py": "# v1\n"}},
        ],
    )
    null = "0" * 40
    out = _run(repo, "check-b", "--range", null)
    assert "misconfigured invocation" not in out.stderr, (
        "a null FROM ref was treated as a misconfiguration, which refuses a new-branch push"
    )
    assert out.returncode == 0, out.stdout + out.stderr
    assert "no in-repo predecessor" in out.stdout or "no layout-relevant changes" in out.stdout, out.stdout

    # CONTROL: the non-commit refusal must still fire for an object that really is not a commit.
    empty_tree = subprocess.run(
        ["git", "hash-object", "-t", "tree", "-w", "/dev/null"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    control = _run(repo, "check-b", "--range", empty_tree)
    assert control.returncode == 1, control.stdout + control.stderr
    assert "does not name a commit" in control.stderr, control.stderr


def test_check_b_range_still_reports_the_slug_arm(tmp_path: Path) -> None:
    """Done-when 3, second arm. The move changes the VENUE and not the predicate, so both arms
    `check-b` reported before the move must still be reachable at the range venue. The file arm
    is covered by the existing range tests; this is the slug arm, which a per-commit refactor
    could easily have dropped while every file-scan arm stayed green."""
    main_scenario = "def compute_event_id_slug(year, event_type, event_id):\n    return f'year.{year}'\n"
    head_scenario = "def compute_event_id_slug(year, event_type, event_id):\n    return f'y{year}'\n"
    repo, _ = _make_history_repo(
        tmp_path,
        [
            {"version": 4, "files": {"src/hhemt/scenario.py": main_scenario}},
            {"version": 4, "files": {"src/hhemt/scenario.py": head_scenario}},
        ],
        sentinel_yaml="layout_relevant:\n  paths: []\n  globs: []\nnon_breaking_allowlist:\n",
    )
    out = _run(repo, "check-b", "--range", "HEAD~1")
    assert out.returncode == 1, out.stdout + out.stderr
    assert "compute_event_id_slug" in out.stderr
    assert "drift" in out.stderr
