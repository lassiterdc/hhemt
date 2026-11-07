"""Reviewer tests for `scripts/hook_env.sh` (work package 1b, chunk 1b.1).

The coder's `tests/test_hook_env.py` holds the shape's tests (a) to (h) on the paths the wrapper's
design names: the key, the age tiers, the repair, the interpreter and extras. These tests hold the
properties that design implies and those tests do not reach: what the wrapper does with a stale lock,
a missing lock, a root given in a form other than an absolute path, arguments and exit statuses that
pass through it, the precedence of the root's sources, and the guards around removal. Every test
builds a dependency-free toy project in `tmp_path`, points the root at `tmp_path`, and writes nothing
under the real cache. `HHEMT_REVIEW_WRAPPER` names an alternative script for mutation runs; unset, the
shipped wrapper is graded.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import pytest

WRAPPER = Path(
    os.environ.get("HHEMT_REVIEW_WRAPPER") or Path(__file__).resolve().parents[1] / "scripts" / "hook_env.sh"
)

pytestmark = pytest.mark.skipif(shutil.which("uv") is None, reason="uv is not on PATH")

_DAY = 86400
_HEX16 = re.compile(r"^[0-9a-f]{16}$")
_SCRUBBED = {"VIRTUAL_ENV", "HHEMT_TY_ENV", "PYTHONPATH", "UV_PROJECT_ENVIRONMENT", "HHEMT_HOOK_ENV_ROOT"}


def _make_toy(tree: Path) -> Path:
    """A dependency-free project with empty `test` and `docs` extras, locked and git-initialised."""
    (tree / "src" / "toy").mkdir(parents=True)
    (tree / "src" / "toy" / "__init__.py").write_text("", encoding="utf-8")
    (tree / "pyproject.toml").write_text(
        '[project]\nname = "toy"\nversion = "0.1.0"\nrequires-python = ">=3.11,<3.13"\n'
        "dependencies = []\n[project.optional-dependencies]\ntest = []\ndocs = []\n",
        encoding="utf-8",
    )
    subprocess.run(["uv", "lock", "--quiet"], cwd=tree, check=True)
    subprocess.run(["git", "init", "--quiet", "."], cwd=tree, check=True)
    return tree


def _run(cwd: Path, *args: str, root: str | Path | None = None, drop: tuple[str, ...] = (), **extra: str):
    env = {k: v for k, v in os.environ.items() if k not in _SCRUBBED and k not in drop}
    if root is not None:
        env["HHEMT_HOOK_ENV_ROOT"] = str(root)
    env.update(extra)
    return subprocess.run([str(WRAPPER), *args], cwd=cwd, env=env, capture_output=True, text=True)


def _dirs(root: Path) -> list[str]:
    return sorted(p.name for p in root.iterdir() if p.is_dir()) if root.exists() else []


def _backdate(path: Path, days: int) -> None:
    stamp = time.time() - days * _DAY
    os.utime(path, (stamp, stamp))


def _toy_and_root(tmp_path: Path) -> tuple[Path, Path]:
    return _make_toy(tmp_path / "toy_tree"), tmp_path / "root"


def test_a_stale_lock_is_refused_the_lock_is_left_alone_and_the_command_does_not_run(tmp_path):
    """Property: when `pyproject.toml` disagrees with `uv.lock`, the wrapper exits non-zero, rewrites
    nothing in the tree, and never runs the command.

    Class: NEW CAPABILITY -- the plausible wrong implementation syncs without `--locked`, which
    silently re-locks and runs the command against a lock nobody reviewed; red under it, pasted in
    the round file.
    Why required: the shape's D2 criterion 3 is that `--locked` keeps refusing a stale lock legibly;
    a wrapper that rewrites `uv.lock` in the committing tree dirties it during a commit.
    What kills it: removing `--locked` from the sync.
    Second correct implementation: a wrapper that compares the lock to the project with a different
    uv subcommand and refuses before syncing; it exits non-zero, leaves the lock, and runs nothing.
    """
    tree, root = _toy_and_root(tmp_path)
    before = (tree / "uv.lock").read_bytes()
    pyproject = tree / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8").replace('version = "0.1.0"', 'version = "0.2.0"'), encoding="utf-8"
    )
    marker = tmp_path / "ran"
    result = _run(tree, "python", "-c", f"open({str(marker)!r}, 'w').close()", root=root)
    assert result.returncode != 0, result.stdout
    assert (tree / "uv.lock").read_bytes() == before
    assert not marker.exists()
    assert not (tree / ".venv").exists()


def test_a_tree_with_no_lock_is_refused_with_exit_two_naming_the_lock(tmp_path):
    """Property: with no `uv.lock` at the tree root the wrapper exits 2, names `uv.lock` on stderr,
    and creates no key directory.

    Class: NEW CAPABILITY -- a tree that has no lock, reached by a clone of a project that does not
    commit one or by a hook run in the wrong directory. The plausible wrong implementation hashes an
    empty stream, mints a key from nothing, and fails later with an unrelated message.
    Why required: the key is a function of the lock's bytes (shape step 2); without a lock there is
    no key to mint, and a key minted from nothing is shared by every lock-less tree.
    What kills it: removing the lock-existence check.
    Second correct implementation: one that exits 2 from a differently worded message that still
    names `uv.lock`.
    """
    tree, root = _toy_and_root(tmp_path)
    (tree / "uv.lock").unlink()
    result = _run(tree, "python", "-c", "pass", root=root)
    assert result.returncode == 2, (result.returncode, result.stderr)
    assert "uv.lock" in result.stderr
    assert _dirs(root) == []


def test_a_relative_root_never_leaves_an_environment_somewhere_the_run_does_not_use(tmp_path):
    """Property: with `HHEMT_HOOK_ENV_ROOT` given as a RELATIVE path and the wrapper started from a
    subdirectory of the tree, either the run succeeds and the key directory is where the caller's own
    working directory puts it, or the run fails and no key directory exists anywhere inside the tree.

    Class: NEW CAPABILITY -- root forms other than an absolute path. The plausible wrong
    implementation syncs from inside the tree (a `cd` to the tree around `uv sync`) while the rest of
    the script resolves the root against the caller's directory, so the sync builds an environment
    inside the tree and the later steps look for it elsewhere; red under it, pasted in the round file.
    Why required: the wrapper's whole purpose is that no run creates an environment inside a tree (shape
    D2, criterion 1, and test (a)); a root spelled relatively must not defeat that by changing the
    directory only one step resolves against.
    What kills it: `cd "$tree"` around the sync with no absolutising of the root.
    Second correct implementation: one that refuses a relative root, or one that turns it into an
    absolute path against the caller's directory before step 3; both satisfy either arm.
    """
    tree, _ = _toy_and_root(tmp_path)
    sub = tree / "src" / "toy"
    result = _run(sub, "python", "-c", "pass", root="rel_root")
    inside = [p for p in tree.rglob("pyvenv.cfg")]
    if result.returncode == 0:
        assert any(_HEX16.match(p.parent.name) for p in (sub / "rel_root").glob("*/pyvenv.cfg")), result.stderr
    else:
        assert inside == [], (result.stderr, inside)


def test_arguments_and_the_exit_status_pass_through_unchanged(tmp_path):
    """Property: every argument after the wrapper's own options reaches the command byte-for-byte,
    including empty, spaced, quoted, dash-leading, glob and variable-shaped arguments, and the
    command's exit status is the wrapper's.

    Class: NEW CAPABILITY -- argument shapes a hook passes (`check_anonymization.py` receives a
    commit-message file path; ruff and ty receive file lists). The plausible wrong implementation runs
    `uv run --no-sync $@` unquoted, which splits a spaced argument, drops an empty one and expands a glob.
    Why required: the commit-msg hook is the one hook handed an argument, and a path with a space in it
    is a path the wrapper must not re-split.
    What kills it: an unquoted `$@`, or a `uv run` whose status is discarded.
    Second correct implementation: a wrapper that execs `uv run --no-sync -- "$@"`; the same arguments
    arrive and the same status returns.
    """
    tree, root = _toy_and_root(tmp_path)
    args = ["", "a b", "--dash", "it's", "*", "$HOME", "x\ty"]
    code = "import sys, json; print(json.dumps(sys.argv[1:])); sys.exit(7)"
    result = _run(tree, "python", "-c", code, *args, root=root)
    assert result.returncode == 7, (result.returncode, result.stderr)
    assert json.loads(result.stdout) == args


def test_the_double_dash_ends_the_wrappers_options(tmp_path):
    """Property: `--` after the wrapper's options ends them, so a command whose own first argument
    looks like a wrapper option is passed to the command rather than parsed.

    Class: NEW CAPABILITY -- a command invoked as `hook_env.sh --python 3.12 -- python --version`
    (the usage line names the `--` form). The plausible wrong implementation treats `--` as a command.
    Why required: the usage line in the script's own header promises the form; a recipe relies on it.
    What kills it: dropping the `--)` branch of the option loop.
    Second correct implementation: one that stops at the first non-option argument and also accepts `--`.
    """
    tree, root = _toy_and_root(tmp_path)
    result = _run(tree, "--", "python", "-c", "import sys; print(sys.argv[1:])", "--python", root=root)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "['--python']"


def test_a_tree_and_a_root_with_spaces_in_their_paths_still_work(tmp_path):
    """Property: a tree directory and a root directory whose names contain a space produce a working
    environment, a printed command result and no `.venv` in the tree.

    Class: NEW CAPABILITY -- paths with spaces, which a laptop's home or a scratch mount can carry. The
    plausible wrong implementation leaves one of `$tree`, `$root` or `$env` unquoted at a use site.
    Why required: a word-split path names a different directory, and a split `rm -rf` in step 6 would
    name two.
    What kills it: an unquoted expansion of the tree, the root or the environment path.
    Second correct implementation: a wrapper that rejects a space in either path with exit 2; the
    assertion below would then need a refusal arm, so this test is written for a wrapper that works.
    """
    tree = _make_toy(tmp_path / "toy tree")
    root = tmp_path / "hook envs"
    result = _run(tree, "python", "-c", "import toy; print(toy.__file__)", root=root)
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()).resolve() == (tree / "src" / "toy" / "__init__.py").resolve()
    assert not (tree / ".venv").exists()
    assert len(_dirs(root)) == 1


def test_the_command_runs_in_the_directory_it_was_started_from(tmp_path):
    """Property: the command's working directory is the directory the wrapper was started in, and the
    command still imports the tree's own package from there.

    Class: NEW CAPABILITY -- a hook started from a subdirectory of the tree. The plausible wrong
    implementation changes directory to the tree for the whole script rather than for the sync only.
    Why required: hooks and recipes pass relative file paths; a changed directory makes them name other
    files.
    What kills it: a `cd "$tree"` outside the subshell that wraps the sync.
    Second correct implementation: a wrapper that passes `--directory` to uv for the sync and leaves the
    process directory alone.
    """
    tree, root = _toy_and_root(tmp_path)
    sub = tree / "src" / "toy"
    result = _run(sub, "python", "-c", "import os, toy; print(os.getcwd()); print(toy.__file__)", root=root)
    assert result.returncode == 0, result.stderr
    cwd_line, file_line = result.stdout.split("\n")[:2]
    assert Path(cwd_line).resolve() == sub.resolve()
    assert Path(file_line).resolve() == (tree / "src" / "toy" / "__init__.py").resolve()


def test_the_set_of_extras_not_their_order_makes_the_key(tmp_path):
    """Property: the default extras given explicitly in either order yield the same single key
    directory as the default.

    Class: NEW CAPABILITY -- permutations of one extras set. The plausible wrong implementation joins the
    extras in the order given, so `--extra docs --extra test` and `--extra test --extra docs` make two
    environments of identical content.
    Why required: the shape keys on "the sorted extras" so a recipe and a hook that name the same set
    differently still share one environment.
    What kills it: dropping the `sort` from the key computation.
    Second correct implementation: a key built from a set of the extras rendered in a fixed order by a
    different tool; the property holds for any canonical order.
    """
    tree, root = _toy_and_root(tmp_path)
    for args in ((), ("--extra", "docs", "--extra", "test"), ("--extra", "test", "--extra", "docs")):
        result = _run(tree, *args, "python", "-c", "pass", root=root)
        assert result.returncode == 0, result.stderr
    assert len(_dirs(root)) == 1, _dirs(root)


def test_the_root_comes_from_the_override_then_the_xdg_cache_then_the_home_cache(tmp_path):
    """Property: with no `HHEMT_HOOK_ENV_ROOT`, the root is `$XDG_CACHE_HOME/hhemt/hook_envs`; with
    that unset too, it is `$HOME/.cache/hhemt/hook_envs`.

    Class: NEW CAPABILITY -- the two default sources of the root, set to directories under `tmp_path`
    so the real cache is never reached. The plausible wrong implementation ignores `XDG_CACHE_HOME` and
    uses `$HOME/.cache` only.
    Why required: the shape's step 1 names the three-way precedence, and a laptop that sets
    `XDG_CACHE_HOME` to a short path (the previous landing's remedy) would otherwise be ignored.
    What kills it: dropping the `${XDG_CACHE_HOME:-...}` expansion.
    Second correct implementation: one that reads the platform cache directory through a helper with the
    same precedence; the two paths asserted are the same.
    """
    tree = _make_toy(tmp_path / "toy_tree")
    xdg, home = tmp_path / "xdg", tmp_path / "home"
    home.mkdir()
    cache = {"UV_CACHE_DIR": os.environ.get("UV_CACHE_DIR", str(Path.home() / ".cache" / "uv"))}
    first = _run(tree, "python", "-c", "pass", XDG_CACHE_HOME=str(xdg), HOME=str(home), **cache)
    assert first.returncode == 0, first.stderr
    assert len(_dirs(xdg / "hhemt" / "hook_envs")) == 1
    second = _run(tree, "python", "-c", "pass", drop=("XDG_CACHE_HOME",), HOME=str(home), **cache)
    assert second.returncode == 0, second.stderr
    assert len(_dirs(home / ".cache" / "hhemt" / "hook_envs")) == 1


def test_the_current_key_is_never_removed_and_its_use_stamp_is_refreshed(tmp_path):
    """Property: a run whose own key directory carries a `.last_used` 20 days old keeps that directory
    and refreshes the stamp.

    Class: NEW CAPABILITY -- the key the run is using, aged past the bound. The plausible wrong
    implementation lets step 6 skip the current-key exclusion, or touches the stamp after the removal
    pass, so the environment just built is removed while the command is about to run in it.
    Why required: removing the environment the command runs in fails the command; the shape's step 5
    marks use before step 6 and step 6 names "not `$key`".
    What kills it: deleting the `touch` of step 5. (Deleting only the `[ "$name" = "$key" ]` exclusion
    is an EQUIVALENT mutant while the touch precedes the loop, since the stamp is fresh by then; it
    is not killed and is not claimed to be.)
    Second correct implementation: a removal pass that skips any directory modified in the current run.
    """
    tree, root = _toy_and_root(tmp_path)
    assert _run(tree, "python", "-c", "pass", root=root).returncode == 0
    key = _dirs(root)[0]
    stamp = root / key / ".last_used"
    _backdate(stamp, 20)
    old = stamp.stat().st_mtime
    result = _run(tree, "python", "-c", "pass", root=root)
    assert result.returncode == 0, result.stderr
    assert (root / key).is_dir()
    assert stamp.stat().st_mtime > old + 19 * _DAY


def test_removal_never_reaches_through_a_symlink_or_a_plain_file_named_like_a_key(tmp_path):
    """Property: under the root, a 16-hex-named symlink to a directory elsewhere, and a 16-hex-named
    plain file, both back-dated 30 days, are left alone and the symlink's target is untouched.

    Class: NEW CAPABILITY -- entries of the root that are not key directories. The plausible wrong
    implementation tests only the name, so `rm -rf` follows a link's name and removes the link, or
    removes a file; red under it for the link target's content check.
    Why required: shape step 6 says "nothing outside `$root` is reachable"; a link in the root is the
    one way a listing-derived name can point outside it.
    What kills it: a removal that resolves the entry to its target before deleting (a `pwd -P` or
    `realpath` in the `rm` argument) with no `-L` guard. A bare `rm -rf` of the link path removes the
    link and not the target, so dropping the `-L` guard alone is not killed by this test.
    Second correct implementation: one that resolves each entry with `realpath` and skips any that
    resolves outside the root.
    """
    tree, root = _toy_and_root(tmp_path)
    assert _run(tree, "python", "-c", "pass", root=root).returncode == 0
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep", encoding="utf-8")
    link = root / "0123456789abcdef"
    link.symlink_to(outside, target_is_directory=True)
    plain = root / "fedcba9876543210"
    plain.write_text("x", encoding="utf-8")
    for p in (link, plain):
        os.utime(p, (time.time() - 30 * _DAY,) * 2, follow_symlinks=False)
    _backdate(outside, 30)
    assert _run(tree, "python", "-c", "pass", root=root).returncode == 0
    assert (outside / "keep.txt").read_text(encoding="utf-8") == "keep"
    assert plain.exists()


def test_a_failed_sync_removes_no_other_key(tmp_path):
    """Property: when the sync fails (a stale lock), an unrelated key directory aged past the bound is
    still there afterwards.

    Class: NEW CAPABILITY -- a run that exits before step 5. The plausible wrong implementation runs the
    removal pass before the sync, or on a trap, so a refused commit still deletes environments.
    Why required: the shape orders removal after sync and touch; a commit refused for a stale lock must
    have no side effect on other trees' environments.
    What kills it: moving the removal loop above the sync.
    Second correct implementation: a removal that runs from an exit trap only when the exit status is 0.
    """
    tree, root = _toy_and_root(tmp_path)
    old_key = root / "0000000000000000"
    old_key.mkdir(parents=True)
    (old_key / "pyvenv.cfg").write_text("home = x\n", encoding="utf-8")
    (old_key / ".last_used").write_text("", encoding="utf-8")
    _backdate(old_key / ".last_used", 30)
    pyproject = tree / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8").replace('version = "0.1.0"', 'version = "0.3.0"'), encoding="utf-8"
    )
    result = _run(tree, "python", "-c", "pass", root=root)
    assert result.returncode != 0
    assert old_key.is_dir()


def test_an_inherited_virtual_env_is_not_passed_to_uv(tmp_path):
    """Property: a `VIRTUAL_ENV` in the caller's environment does not reach the command as the
    inherited value (uv itself sets the variable to the shared environment) and does not make uv warn
    about a mismatch.

    Class: NEW CAPABILITY -- a hook started from an activated shell. The plausible wrong implementation
    leaves `VIRTUAL_ENV` exported, so uv prints a mismatch warning on every hook.
    Why required: shape step 3 says to unset it because an inherited one makes uv warn on every hook.
    What kills it: dropping the `unset VIRTUAL_ENV`.
    Second correct implementation: a wrapper that runs uv under `env -u VIRTUAL_ENV`.
    """
    tree, root = _toy_and_root(tmp_path)
    result = _run(
        tree,
        "python",
        "-c",
        "import os; print(os.environ.get('VIRTUAL_ENV', 'unset'))",
        root=root,
        VIRTUAL_ENV=str(tmp_path / "elsewhere"),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() != str(tmp_path / "elsewhere")
    assert "VIRTUAL_ENV" not in result.stderr
