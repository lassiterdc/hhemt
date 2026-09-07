"""Tests for scripts/check_layout_version.py."""

from __future__ import annotations

import importlib.util
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


def _run(repo: Path, *args: str) -> subprocess.CompletedProcess:
    target = repo / "scripts" / "check_layout_version.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(SCRIPT.read_text())
    return subprocess.run([sys.executable, str(target), *args], cwd=repo, capture_output=True, text=True)


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


def test_check_b_passes_when_path_in_non_breaking_allowlist(tmp_path: Path) -> None:
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
        sentinel.read_text().replace("non_breaking_allowlist: []", "non_breaking_allowlist:\n  - src/foo.py")
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
    """The wiring half of the venue split, over BOTH entry points. A pre-push registration
    WITHOUT `--range` reads the working tree -- for check-b an unreachable disarm that refuses
    every legitimate migration commit, for check-a a version/module/fixture read that both
    false-blocks on a pending bump and false-passes on an uncommitted module. A pre-commit
    registration WITH it would stop grading the pending change. Every check-layout-version hook
    previously carried no `stages:` key at all and ran everywhere by pre-commit's all-stages
    default, which is how the venue was reached without anyone choosing it.

    The filter is `check_layout_version.py check-` and not `check-b`: scoping this arm to the
    hook that was just repaired is how the same defect survives one function over."""
    cfg = yaml.safe_load((Path(__file__).resolve().parent.parent / ".pre-commit-config.yaml").read_text())
    hooks = [
        h for repo in cfg["repos"] for h in repo["hooks"] if "check_layout_version.py check-" in h.get("entry", "")
    ]
    assert hooks, "no check_layout_version registration found at all"
    by_check: dict[str, dict[str, dict]] = {}
    for hook in hooks:
        stages = hook.get("stages")
        assert stages is not None, f"{hook['id']} carries no explicit stages: it runs at every stage"
        assert len(stages) == 1, f"{hook['id']} declares {stages}; one registration serves exactly one venue"
        check = hook["entry"].split("check_layout_version.py ")[1].split()[0]
        assert stages[0] not in by_check.get(check, {}), f"{check} has two registrations at {stages[0]}"
        by_check.setdefault(check, {})[stages[0]] = hook
    for check, venues in sorted(by_check.items()):
        assert set(venues) == {"pre-commit", "pre-push"}, f"{check} is registered at {sorted(venues)}, not both venues"
        pre_commit, pre_push = venues["pre-commit"], venues["pre-push"]
        assert "--range" not in pre_commit["entry"], f"{pre_commit['id']} runs at pre-commit with the range predicate"
        assert "--range" in pre_push["entry"], f"{pre_push['id']} runs at pre-push without the range predicate"
        assert pre_commit["entry"].split()[-1] == "HEAD", (
            f"{pre_commit['id']} bases on {pre_commit['entry'].split()[-1]!r}; the pre-commit venue grades the "
            "change being committed, and a wider base flags the previous commit's files"
        )
        assert pre_push["entry"].split()[-1] == "HEAD~1", (
            f"{pre_push['id']} bases on {pre_push['entry'].split()[-1]!r}; the pre-push venue grades the commit range"
        )

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
