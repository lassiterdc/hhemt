"""Tests for scripts/check_layout_version.py."""

from __future__ import annotations

import importlib.util
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


def test_check_b_scans_the_amended_commits_own_content(tmp_path: Path) -> None:
    """Under `git commit --amend` HEAD is the commit being REPLACED, so that commit's own
    layout-relevant change is invisible to a HEAD-based file set and visible to a
    HEAD~1-based one. This pins that `base_ref` -- not the disarm -- governs the FILE SET,
    and that moving the file set to HEAD would reopen it."""
    repo = _make_linear_repo(tmp_path, [4, 4], worktree_version=4, touch_layout_file=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "add", "."], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "c2 touches foo"],
        cwd=repo,
        check=True,
    )
    (repo / "readme.md").write_text("# r\n")
    out = _run(repo, "check-b", "HEAD~1")
    assert out.returncode == 1
    assert "src/foo.py" in out.stderr


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
