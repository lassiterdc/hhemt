"""Tests for `scripts/hook_env.sh`, the wrapper that runs a command in a shared uv environment.

The wrapper exists so that no commit and no `just` recipe creates a `.venv/` inside a tree: the
dependencies live in one environment per (uv.lock, Python version, extras) under a root held
outside every tree, the committing tree is bound by `PYTHONPATH`, and environments unused for 14
days are removed. Each test builds a dependency-free toy project in `tmp_path` and points
`HHEMT_HOOK_ENV_ROOT` at a `tmp_path` directory, so nothing is written under the real cache.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

WRAPPER = Path(__file__).resolve().parents[1] / "scripts" / "hook_env.sh"

pytestmark = pytest.mark.skipif(shutil.which("uv") is None, reason="uv is not on PATH")

_DAY = 86400
_HEX16 = re.compile(r"^[0-9a-f]{16}$")


_TOY_HOMEPAGE = "https://example.invalid/toy"


def _write_toy_pyproject(tree: Path, version: str, dependencies: str = '["setuptools>=77.0.0"]') -> None:
    (tree / "pyproject.toml").write_text(
        "[build-system]\n"
        'requires = ["setuptools>=77.0.0"]\n'
        'build-backend = "setuptools.build_meta"\n'
        "[project]\n"
        'name = "toy"\n'
        f'version = "{version}"\n'
        'requires-python = ">=3.11,<3.13"\n'
        f"dependencies = {dependencies}\n"
        "[project.urls]\n"
        f'homepage = "{_TOY_HOMEPAGE}"\n'
        "[project.scripts]\n"
        'toy = "toy.cli:main"\n'
        "[project.optional-dependencies]\n"
        "test = []\n"
        "docs = []\n",
        encoding="utf-8",
    )


def _make_toy(tree: Path, version: str = "0.1.0", dependencies: str = '["setuptools>=77.0.0"]') -> Path:
    """Build a setuptools project with a console script, empty `test` and `docs` extras, locked and git-initialised.

    The toy depends on setuptools so its shared environment can run the build backend, as `hsclient` makes
    hhemt's environment do.
    """
    (tree / "src" / "toy").mkdir(parents=True)
    (tree / "src" / "toy" / "__init__.py").write_text("", encoding="utf-8")
    (tree / "src" / "toy" / "cli.py").write_text('def main():\n    print("toy-cli-output")\n', encoding="utf-8")
    (tree / "src" / "toy" / "__main__.py").write_text("from toy.cli import main\n\nmain()\n", encoding="utf-8")
    _write_toy_pyproject(tree, version, dependencies)
    subprocess.run(["uv", "lock", "--quiet"], cwd=tree, check=True)
    subprocess.run(["git", "init", "--quiet", "."], cwd=tree, check=True)
    return tree


def _env(root: Path, **extra: str) -> dict[str, str]:
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in {"VIRTUAL_ENV", "HHEMT_TY_ENV", "PYTHONPATH", "UV_PROJECT_ENVIRONMENT"}
    }
    env["HHEMT_HOOK_ENV_ROOT"] = str(root)
    env.update(extra)
    return env


def _run(tree: Path, root: Path, *args: str, wrapper: Path = WRAPPER, **extra_env: str):
    return subprocess.run(
        [str(wrapper), *args],
        cwd=tree,
        env=_env(root, **extra_env),
        capture_output=True,
        text=True,
    )


def _keys(root: Path) -> list[str]:
    return sorted(p.name for p in root.iterdir() if p.is_dir()) if root.exists() else []


def _backdate(path: Path, days: int) -> None:
    stamp = time.time() - days * _DAY
    os.utime(path, (stamp, stamp))


def _toy_and_root(tmp_path: Path, name: str = "toy_tree") -> tuple[Path, Path]:
    return _make_toy(tmp_path / name), tmp_path / "root"


def test_a_run_creates_no_venv_in_the_tree_and_binds_the_tree_by_import_path(tmp_path):
    """Property: the command imports the committing tree's own package, the tree gains no
    `.venv`, and exactly one key directory appears under the root.

    Class: NEW CAPABILITY -- the plausible wrong implementation runs a plain `uv run`, which
    creates `.venv` in the tree; it also fails the import-path half if it binds no `PYTHONPATH`.
    Why required: a per-tree environment is 1.6 GiB on a nearly full disk and is never removed.
    What kills it: dropping `UV_PROJECT_ENVIRONMENT`, or dropping the `PYTHONPATH` export.
    Second correct implementation: one that installs nothing and sets `sys.path` through a
    `.pth` file in the shared environment per run; it prints the same path and still passes.
    """
    tree, root = _toy_and_root(tmp_path)
    result = _run(tree, root, "python", "-c", "import toy; print(toy.__file__)")
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()).resolve() == (tree / "src" / "toy" / "__init__.py").resolve()
    assert not (tree / ".venv").exists()
    assert len(_keys(root)) == 1 and _HEX16.match(_keys(root)[0])


def test_two_trees_with_identical_locks_share_one_key_and_each_prints_its_own_file(tmp_path):
    """Property: byte-identical `uv.lock` files map to one key directory, and each tree still
    imports its own code.

    Class: NEW CAPABILITY -- the plausible wrong implementation keys on the tree path, which
    gives two environments, or installs the project into the shared environment, which makes
    the second run print the first tree's file.
    Why required: sharing is what keeps a second tree's cost near zero, and the per-process
    import binding is what keeps two trees from grading each other's code.
    What kills it: keying on the tree path; an editable install of the project.
    Second correct implementation: keying on a different hash of the lock bytes; the property
    holds for any function of the bytes.
    """
    tree_one, root = _toy_and_root(tmp_path, "one")
    tree_two = _make_toy(tmp_path / "two")
    assert (tree_one / "uv.lock").read_bytes() == (tree_two / "uv.lock").read_bytes()
    for tree in (tree_one, tree_two):
        result = _run(tree, root, "python", "-c", "import toy; print(toy.__file__)")
        assert result.returncode == 0, result.stderr
        assert Path(result.stdout.strip()).resolve() == (tree / "src" / "toy" / "__init__.py").resolve()
    assert len(_keys(root)) == 1


def test_changing_the_lock_yields_a_second_key(tmp_path):
    """Property: a different `uv.lock` yields a different key directory.

    Class: NEW CAPABILITY -- the plausible wrong implementation keys on `pyproject.toml` or on a
    constant, so a changed lock reuses an environment built from the old one.
    Why required: the environment must match the lock the hooks are graded against.
    What kills it: hashing anything other than the lock's bytes.
    Second correct implementation: hashing the lock's bytes with another digest; still differs.
    """
    tree, root = _toy_and_root(tmp_path)
    assert _run(tree, root, "python", "-c", "pass").returncode == 0
    (tree / "pyproject.toml").write_text(
        (tree / "pyproject.toml").read_text(encoding="utf-8").replace('"0.1.0"', '"0.2.0"'), encoding="utf-8"
    )
    subprocess.run(["uv", "lock", "--quiet"], cwd=tree, check=True)
    assert _run(tree, root, "python", "-c", "pass").returncode == 0
    assert len(_keys(root)) == 2


def _fake_key(root: Path, name: str, kind: str, age_days: int) -> Path:
    """A key directory as the wrapper's removal step sees it, aged by the file the step reads."""
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "payload").write_text("x", encoding="utf-8")
    if kind == "last_used":
        marker = directory / ".last_used"
        marker.write_text("", encoding="utf-8")
        _backdate(marker, age_days)
    elif kind == "cfg":
        marker = directory / "pyvenv.cfg"
        marker.write_text("home = none\n", encoding="utf-8")
        _backdate(marker, age_days)
    else:
        _backdate(directory, age_days)
    return directory


@pytest.mark.parametrize("kind", ["last_used", "cfg", "bare"])
@pytest.mark.parametrize(("age_days", "removed"), [(15, True), (13, False)])
def test_unused_key_directories_are_removed_by_the_age_of_the_file_the_step_reads(tmp_path, kind, age_days, removed):
    """Property: a key directory unused for more than 14 days is removed on the next run, and one
    used 13 days ago is kept, for each of the three ways the age is read: `.last_used`, else
    `pyvenv.cfg`, else the directory itself.

    Class: NEW CAPABILITY -- the plausible wrong implementation ages every directory by its own
    modification time, which a part-written or recently synced directory defeats.
    Why required: the 14-day removal is the guarantee that environments do not accumulate.
    What kills it: a threshold other than 14 days, or ageing every class by one path.
    Second correct implementation: computing the age with `stat` instead of `find -mtime`;
    the same directories are removed.
    """
    tree, root = _toy_and_root(tmp_path)
    stale = _fake_key(root, "0123456789abcdef", kind, age_days)
    assert _run(tree, root, "python", "-c", "pass").returncode == 0
    assert stale.exists() is (not removed)


def test_a_fresh_pyvenv_cfg_keeps_a_directory_the_age_of_whose_own_mtime_is_old(tmp_path):
    """Property: with no `.last_used`, a fresh `pyvenv.cfg` keeps a directory whose own
    modification time is 20 days old.

    Class: NEW CAPABILITY -- a sync in progress in another process has a fresh `pyvenv.cfg`.
    Why required: removing it would delete an environment another process is filling.
    What kills it: ageing by the directory's own mtime before `pyvenv.cfg`.
    Second correct implementation: ageing by the newest mtime of the three paths; still kept.
    """
    tree, root = _toy_and_root(tmp_path)
    directory = _fake_key(root, "fedcba9876543210", "cfg", 0)
    _backdate(directory, 20)
    assert _run(tree, root, "python", "-c", "pass").returncode == 0
    assert directory.exists()


def test_a_directory_whose_name_is_not_16_hex_digits_is_kept_whatever_its_age(tmp_path):
    """Property: only names of exactly 16 hex digits are candidates for removal.

    Class: NEW CAPABILITY -- names of the wrong length, upper case, or non-hex characters.
    Why required: the removal runs `rm -rf`; a matcher that accepts any name could remove a
    directory a person put under the root.
    What kills it: dropping the name test, or loosening it to any 16 characters.
    Second correct implementation: a regular expression over the name instead of a length and
    a character-class test; the same names are kept.
    """
    tree, root = _toy_and_root(tmp_path)
    kept = []
    for name in ("short", "ZZZZZZZZZZZZZZZZ", "0123456789abcdeF", "0123456789abcdef0"):
        kept.append(_fake_key(root, name, "bare", 40))
    assert _run(tree, root, "python", "-c", "pass").returncode == 0
    assert all(path.exists() for path in kept)


def _current_env(tree: Path, root: Path) -> Path:
    result = _run(tree, root, "--print-env")
    assert result.returncode == 0, result.stderr
    return Path(result.stdout.strip())


def _without_repair(tmp_path: Path) -> Path:
    """A copy of the wrapper with step 4's repair lines deleted, for the must-fail arm."""
    lines = WRAPPER.read_text(encoding="utf-8").splitlines(keepends=True)
    kept, skipping = [], False
    for line in lines:
        if "# REPAIR-BEGIN" in line:
            skipping = True
        if not skipping:
            kept.append(line)
        if "# REPAIR-END" in line:
            skipping = False
    assert len(kept) < len(lines), "the repair markers were not found"
    copy = tmp_path / "hook_env_no_repair.sh"
    copy.write_text("".join(kept), encoding="utf-8")
    copy.chmod(0o755)
    return copy


def test_a_stray_directory_with_no_pyvenv_cfg_is_repaired_and_the_repair_is_what_does_it(tmp_path):
    """Property: when the current key's directory exists holding only a stray file, the run
    succeeds and leaves `pyvenv.cfg` and a working interpreter; the same run against a copy of
    the wrapper without the repair exits 2.

    Class: NEW CAPABILITY, constructed rather than raced -- a dependency-free sync finishes too
    fast for a kill to be aimed into it, so a kill-based test would pass whether or not the
    kill landed.
    Why required: uv refuses such a directory, so without the repair one stray directory blocks
    every commit on that lock until someone removes it by hand.
    What kills it: deleting step 4's repair.
    Second correct implementation: removing the directory before every sync; it also recovers,
    at the cost the shape rules out, and the property still holds.
    """
    tree, root = _toy_and_root(tmp_path)
    env = _current_env(tree, root)
    shutil.rmtree(env)
    env.mkdir()
    (env / "stray").write_text("x", encoding="utf-8")

    broken = _run(tree, root, "python", "-c", "pass", wrapper=_without_repair(tmp_path))
    assert broken.returncode == 2, (broken.returncode, broken.stderr)
    assert not (env / "pyvenv.cfg").exists()

    repaired = _run(tree, root, "python", "-c", "import sys; print(sys.prefix)")
    assert repaired.returncode == 0, repaired.stderr
    assert (env / "pyvenv.cfg").is_file()
    assert Path(repaired.stdout.strip()).resolve() == env.resolve()
    assert not (tree / ".venv").exists()


def test_an_environment_whose_bin_directory_is_missing_is_restored(tmp_path):
    """Property: a key directory that keeps `pyvenv.cfg` but has lost `bin/` is repaired by the
    next run, which restores `bin/`.

    Class: NEW CAPABILITY, constructed -- the state a killed sync can leave.
    Why required: a commit must not fail on a half-written shared environment.
    What kills it: running the command with `--no-sync` and no sync step first.
    Second correct implementation: removing and rebuilding the whole directory; `bin/` returns.
    """
    tree, root = _toy_and_root(tmp_path)
    env = _current_env(tree, root)
    shutil.move(str(env / "bin"), str(tmp_path / "bin_moved_away"))
    result = _run(tree, root, "python", "-c", "pass")
    assert result.returncode == 0, result.stderr
    assert (env / "bin").is_dir()
    assert not (tree / ".venv").exists()


def test_hhemt_ty_env_is_kept_when_preset_and_defaults_to_the_key_directory(tmp_path):
    """Property: a caller-set `HHEMT_TY_ENV` is left unchanged, and an unset one becomes the key
    directory.

    Class: NEW CAPABILITY -- both branches of `: "${HHEMT_TY_ENV:=$env}"`.
    Why required: the `ty` hooks gate on this variable, and a conda prefix a caller set must win.
    What kills it: assigning unconditionally, or never exporting.
    Second correct implementation: `[ -n "${HHEMT_TY_ENV:-}" ] || HHEMT_TY_ENV=$env`.
    """
    tree, root = _toy_and_root(tmp_path)
    show = ("python", "-c", "import os; print(os.environ.get('HHEMT_TY_ENV'))")
    preset = _run(tree, root, *show, HHEMT_TY_ENV="preset-value")
    assert preset.returncode == 0, preset.stderr
    assert preset.stdout.strip() == "preset-value"
    unset = _run(tree, root, *show)
    assert unset.returncode == 0, unset.stderr
    assert Path(unset.stdout.strip()).resolve() == _current_env(tree, root).resolve()


def _uv_has(version: str) -> bool:
    return subprocess.run(["uv", "python", "find", version], capture_output=True).returncode == 0


@pytest.mark.skipif(not (_uv_has("3.11") and _uv_has("3.12")), reason="needs Python 3.11 and 3.12")
def test_python_version_and_extras_are_part_of_the_key(tmp_path):
    """Property: one lock under `--python 3.11` and `--python 3.12` yields two key directories
    on those interpreters; `--extra test` alone yields a third; an identical request reuses
    its key.

    Class: NEW CAPABILITY -- the plausible wrong implementation keys on the lock alone, so
    `testall`'s 3.11 half silently runs on 3.12.
    Why required: a different interpreter or extra set is a different environment.
    What kills it: leaving the version or the extras out of the hashed suffix, or not passing
    `--python` to the sync.
    Second correct implementation: separating the fields with a different byte than NUL.
    """
    tree, root = _toy_and_root(tmp_path)
    show = ("python", "-c", "import sys; print(sys.version_info[:2])")
    seen = {}
    for version in ("3.11", "3.12"):
        result = _run(tree, root, "--python", version, *show)
        assert result.returncode == 0, result.stderr
        seen[version] = result.stdout.strip()
        env = _current_env_for(tree, root, "--python", version)
        cfg = (env / "pyvenv.cfg").read_text(encoding="utf-8")
        assert re.search(rf"^version_info = {re.escape(version)}\.", cfg, re.M), cfg
    assert seen == {"3.11": "(3, 11)", "3.12": "(3, 12)"}
    assert len(_keys(root)) == 2
    assert _run(tree, root, "--extra", "test", "python", "-c", "pass").returncode == 0
    assert len(_keys(root)) == 3
    assert _run(tree, root, "--python", "3.12", "python", "-c", "pass").returncode == 0
    assert len(_keys(root)) == 3


def _current_env_for(tree: Path, root: Path, *options: str) -> Path:
    result = _run(tree, root, *options, "--print-env")
    assert result.returncode == 0, result.stderr
    return Path(result.stdout.strip())


def test_print_env_prints_the_key_directory_without_running_a_command(tmp_path):
    """Property: `--print-env` prints the absolute key directory, which holds `pyvenv.cfg`, runs
    no command, and leaves no `.venv` in the tree.

    Class: NEW CAPABILITY -- the plausible wrong implementation forwards `--print-env` to
    `uv run`, which fails, or prints before syncing, which prints a path that holds no environment.
    Why required: it is how an editor is pointed at the shared environment.
    What kills it: printing before step 4, or exec-ing `uv run` regardless.
    Second correct implementation: printing `$UV_PROJECT_ENVIRONMENT` instead of `$env`.
    """
    tree, root = _toy_and_root(tmp_path)
    printed = _run(tree, root, "--print-env")
    assert printed.returncode == 0, printed.stderr
    env = Path(printed.stdout.strip())
    assert env.is_absolute() and env.parent == root and _HEX16.match(env.name)
    assert (env / "pyvenv.cfg").is_file()
    assert not (tree / ".venv").exists()
    assert _keys(root) == [env.name]


def test_a_relative_root_is_resolved_against_the_callers_directory(tmp_path):
    """Property: a relative `HHEMT_HOOK_ENV_ROOT` is taken relative to the directory the wrapper
    was started from, so `--print-env` prints that absolute location and nothing is built at the
    same relative path under the tree's top directory.

    Class: NEW CAPABILITY -- the form the root fix chose (absolutise rather than refuse); the
    wrapper is started from a subdirectory so the caller's directory and the tree's top differ.
    Why required: the sync runs from inside the tree, so a root left relative builds the
    environment where the later steps do not look.
    What kills it: refusing a relative root, or leaving it relative around the `cd` to the tree.
    Second correct implementation: resolving the root with `realpath -m` against the caller's
    directory; the printed path is the same.
    """
    tree = _make_toy(tmp_path / "toy_tree")
    sub = tree / "src" / "toy"
    result = subprocess.run(
        [str(WRAPPER), "--print-env"],
        cwd=sub,
        env=_env(Path("rel_root")),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    env = Path(result.stdout.strip())
    assert env.is_absolute() and env.parent == (sub / "rel_root").resolve()
    assert (env / "pyvenv.cfg").is_file()
    assert not (tree / "rel_root").exists()


_SHOW_DIST = (
    "python",
    "-c",
    "import importlib.metadata as m; d = m.distribution('toy'); print(d.version); "
    "print(d.metadata.get_all('Project-URL')); print(d._path)",
)


def _records(root: Path) -> list[Path]:
    return sorted(p for p in root.glob("*/hhemt_meta/*") if p.is_dir() and not p.name.startswith("."))


def _tree_files(tree: Path) -> list[str]:
    return sorted(str(p.relative_to(tree)) for p in tree.rglob("*") if ".git" not in p.relative_to(tree).parts)


def test_the_command_sees_the_trees_own_distribution_record_and_the_tree_is_untouched(tmp_path):
    """Property: under the wrapper `importlib.metadata` resolves the toy distribution to the tree's own version
    and homepage, from a record under the shared key directory, and the tree gains no file.

    Class: NEW CAPABILITY -- the plausible wrong implementation leaves the project uninstalled and unrecorded, so
    the distribution is absent; another wrong one runs the build in the tree and leaves `*.egg-info` or `build/`.
    Why required: a reader that raises on an absent distribution (provenance's code-repository lookup) fails on a
    fresh clone otherwise.
    What kills it: dropping step 4b, or running the backend with its output directory inside the tree.
    Second correct implementation: a record written by `pip install --no-deps --target`; the same fields resolve.
    """
    tree, root = _toy_and_root(tmp_path)
    before = _tree_files(tree)
    result = _run(tree, root, *_SHOW_DIST)
    assert result.returncode == 0, result.stderr
    version, urls, path = result.stdout.strip().splitlines()
    assert version == "0.1.0"
    assert _TOY_HOMEPAGE in urls
    assert Path(path).resolve().is_relative_to(root.resolve())
    assert "hhemt_meta" in Path(path).parts
    assert _tree_files(tree) == before
    assert not list(tree.rglob("*.egg-info")) and not (tree / "build").exists()


def test_a_changed_pyproject_makes_a_second_record_and_identical_trees_share_one(tmp_path):
    """Property: changing a field the record is built from (the homepage) yields a second record directory in the
    same key and the new value; two trees with byte-identical `pyproject.toml` share one record.

    Class: NEW CAPABILITY -- the plausible wrong implementation keys the record on the lock alone, so an edited
    `pyproject.toml` keeps serving the old record, or on the tree path, so identical trees do not share.
    Why required: the record must match the tree's own files and stay immutable.
    What kills it: a meta key that omits `pyproject.toml`, or one that includes the tree path.
    Second correct implementation: hashing the parsed project table instead of the raw bytes; the same changes
    produce a new key.
    """
    tree, root = _toy_and_root(tmp_path, "one")
    other = _make_toy(tmp_path / "two")
    assert (tree / "pyproject.toml").read_bytes() == (other / "pyproject.toml").read_bytes()
    for t in (tree, other):
        assert _run(t, root, "python", "-c", "pass").returncode == 0
    assert len(_records(root)) == 1
    changed = (tree / "pyproject.toml").read_text(encoding="utf-8").replace(_TOY_HOMEPAGE, _TOY_HOMEPAGE + "/new")
    (tree / "pyproject.toml").write_text(changed, encoding="utf-8")
    result = _run(tree, root, *_SHOW_DIST)
    assert result.returncode == 0, result.stderr
    assert _TOY_HOMEPAGE + "/new" in result.stdout
    assert len(_records(root)) == 2


def test_a_new_version_is_what_the_command_sees(tmp_path):
    """Property: after the version in `pyproject.toml` changes and the lock is refreshed, the command sees the new
    version.

    Class: NEW CAPABILITY. Why required: a record served from a previous version would make provenance report the
    wrong release. What kills it: a record keyed on something the version does not change.
    Second correct implementation: any keying that includes `pyproject.toml`'s bytes.
    """
    tree, root = _toy_and_root(tmp_path)
    assert _run(tree, root, *_SHOW_DIST).stdout.splitlines()[0] == "0.1.0"
    _write_toy_pyproject(tree, "0.2.0")
    subprocess.run(["uv", "lock", "--quiet"], cwd=tree, check=True)
    result = _run(tree, root, *_SHOW_DIST)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[0] == "0.2.0"


def test_the_record_wins_over_a_stale_egg_info_in_the_tree(tmp_path):
    """Property: a stale `src/toy.egg-info/PKG-INFO` claiming version 9.9.9 does not win under the wrapper, and does
    win under a bare `PYTHONPATH={tree}/src` (the control).

    Class: NEW CAPABILITY -- trees that carry a gitignored `src/*.egg-info` from an old editable install.
    Why required: such a tree otherwise masks a missing or wrong record, so laptop and fresh clone differ.
    What kills it: putting `{tree}/src` before the record on `PYTHONPATH`.
    Second correct implementation: deleting the stale record from the tree first would also pass, at the cost of
    writing into the tree, which the wrapper must not do.
    """
    tree, root = _toy_and_root(tmp_path)
    egg = tree / "src" / "toy.egg-info"
    egg.mkdir()
    (egg / "PKG-INFO").write_text("Metadata-Version: 2.1\nName: toy\nVersion: 9.9.9\n", encoding="utf-8")
    code = "import importlib.metadata as m; print(m.version('toy'))"
    wrapped = _run(tree, root, "python", "-c", code)
    assert wrapped.returncode == 0, wrapped.stderr
    assert wrapped.stdout.strip() == "0.1.0"
    bare = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "PYTHONPATH": str(tree / "src")},
        capture_output=True,
        text=True,
    )
    assert bare.stdout.strip() == "9.9.9", bare.stderr


def test_the_console_script_resolves_to_a_shim_in_the_record(tmp_path):
    """Property: `toy` is found on `PATH` under the record's `bin/` and runs the toy's command-line entry.

    Class: NEW CAPABILITY -- the plausible wrong implementation records metadata but no executable, so `toy` is
    not found, or finds another tree's install on `PATH`.
    Why required: tests call the `hhemt` console script, which does not exist in an environment without the
    project installed.
    What kills it: not writing the shim, or not prepending `{record}/bin` to `PATH`.
    Second correct implementation: a shim that imports the entry-point callable instead of using `python -m`.
    """
    tree, root = _toy_and_root(tmp_path)
    result = _run(tree, root, "sh", "-c", "command -v toy; toy")
    assert result.returncode == 0, result.stderr
    where, output = result.stdout.strip().splitlines()
    assert Path(where).parent.name == "bin" and "hhemt_meta" in Path(where).parts
    assert output == "toy-cli-output"


@pytest.mark.parametrize(("age_days", "removed"), [(15, True), (13, False)])
@pytest.mark.parametrize("kind", ["record", "tmp"])
def test_unused_records_and_killed_builds_inside_the_current_key_age_out(tmp_path, kind, age_days, removed):
    """Property: inside the current key, a record other than the current one, and a `.tmp.*` entry, are removed
    when unused for more than 14 days and kept at 13.

    Class: NEW CAPABILITY. Why required: records live inside the key directory, so without this rule a long-lived
    key would accumulate one record per `pyproject.toml` edit and keep any killed build forever.
    What kills it: leaving step 6's record loop out, or aging by the wrong path.
    Second correct implementation: aging every entry by its own directory mtime.
    """
    tree, root = _toy_and_root(tmp_path)
    assert _run(tree, root, "python", "-c", "pass").returncode == 0
    meta = next(root.glob("*/hhemt_meta"))
    name = "0123456789abcdef" if kind == "record" else ".tmp.424242"
    stale = meta / name
    stale.mkdir()
    marker = stale / ".last_used"
    marker.write_text("", encoding="utf-8")
    _backdate(marker, age_days)
    assert _run(tree, root, "python", "-c", "pass").returncode == 0
    assert stale.exists() is (not removed)


def test_a_toy_without_setuptools_in_its_environment_is_refused_naming_the_backend(tmp_path):
    """Property: when setuptools is absent from the shared environment the wrapper exits 2 and its message names
    `setuptools.build_meta`.

    Class: NEW CAPABILITY -- the transitive dependency that carries setuptools is dropped. Why required: the
    alternative is a silent run with no record, which is the failure this step exists to remove.
    What kills it: skipping the import check, or a message that does not name the backend.
    Second correct implementation: a refusal that quotes the `[build-system]` table instead.
    """
    tree = _make_toy(tmp_path / "toy_tree", dependencies="[]")
    result = _run(tree, tmp_path / "root", "python", "-c", "pass")
    assert result.returncode == 2, (result.returncode, result.stderr)
    assert "setuptools.build_meta" in result.stderr


def test_the_wrapper_uses_no_gnu_only_mv_and_handles_an_existing_record(tmp_path):
    """Property: the script carries no GNU-only `mv` option; an existing valid record is used and leaves exactly one
    `.dist-info` at its top; an existing record directory holding none makes the wrapper exit 2 naming it.

    Class: NEW CAPABILITY -- the wrapper is POSIX `sh` and step 2 already provides for macOS. Why required: `mv -T`
    does not exist there, and a lost publish race must never leave the loser's copy in front of the winner's.
    What kills it: reintroducing `mv -T`, `--no-target-directory` or `-t`, or building over an existing record.
    Second correct implementation: publishing with `ln -s` and `mv` of a symlink; the same three outcomes hold.
    """
    text = WRAPPER.read_text(encoding="utf-8")
    assert not re.search(r"\bmv\s+(-\w*[Tt]\b|--no-target-directory|--target-directory)", text)
    tree, root = _toy_and_root(tmp_path)
    assert _run(tree, root, "python", "-c", "pass").returncode == 0
    (record,) = _records(root)
    assert _run(tree, root, "python", "-c", "pass").returncode == 0
    assert len(list(record.glob("*.dist-info"))) == 1
    shutil.rmtree(next(record.glob("*.dist-info")))
    broken = _run(tree, root, "python", "-c", "pass")
    assert broken.returncode == 2 and str(record) in broken.stderr
