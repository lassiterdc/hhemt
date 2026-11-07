"""Reviewer tests for the distribution record and console script that `scripts/hook_env.sh` step 4b provides
(the wrapper's record-build step).

The coder's tests (i) to (o) in `tests/test_hook_env.py` hold the record's contents, its keying on `pyproject.toml`,
the stale-`egg-info` precedence, one console script, the age-out of records and the no-GNU-`mv` rule. These tests
hold what those do not reach: a project with more than one console script, a record keyed on the readme and
license files, the digest fallback the wrapper provides for macOS, a build run from a subdirectory, a failed build,
and the cleanliness of the first run's output and of the wrapper's own temporary directory. Every test builds a
small setuptools project in `tmp_path`, points the root at `tmp_path`, and writes nothing under the real cache.
`HHEMT_REVIEW_WRAPPER` names an alternative script for mutation runs.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

WRAPPER = Path(
    os.environ.get("HHEMT_REVIEW_WRAPPER") or Path(__file__).resolve().parents[1] / "scripts" / "hook_env.sh"
)

pytestmark = pytest.mark.skipif(shutil.which("uv") is None, reason="uv is not on PATH")

_SCRUBBED = {"VIRTUAL_ENV", "HHEMT_TY_ENV", "PYTHONPATH", "UV_PROJECT_ENVIRONMENT", "HHEMT_HOOK_ENV_ROOT"}


def _make_toy(
    tree: Path,
    *,
    scripts: dict[str, str] | None = None,
    readme: str | None = None,
    extra_files: dict[str, str] | None = None,
) -> Path:
    """A locked, git-initialised setuptools project, with optional console scripts, readme and license file."""
    (tree / "src" / "toy").mkdir(parents=True)
    (tree / "src" / "toy" / "__init__.py").write_text("", encoding="utf-8")
    (tree / "src" / "toy" / "cli.py").write_text('def main():\n    print("toy-cli-output")\n', encoding="utf-8")
    (tree / "src" / "toy" / "admin.py").write_text('def main():\n    print("toy-admin-output")\n', encoding="utf-8")
    (tree / "src" / "toy" / "__main__.py").write_text("from toy.cli import main\n\nmain()\n", encoding="utf-8")
    scripts = {"toy": "toy.cli:main"} if scripts is None else scripts
    lines = [
        '[build-system]\nrequires = ["setuptools>=77.0.0"]\nbuild-backend = "setuptools.build_meta"\n',
        '[project]\nname = "toy"\nversion = "0.1.0"\nrequires-python = ">=3.11,<3.13"\n',
        'dependencies = ["setuptools>=77.0.0"]\n',
    ]
    if readme is not None:
        (tree / "README.md").write_text(readme, encoding="utf-8")
        (tree / "LICENSE").write_text("Permission is granted.\n", encoding="utf-8")
        lines.append('readme = "README.md"\nlicense = "MIT"\nlicense-files = ["LICENSE"]\n')
    if scripts:
        lines.append("[project.scripts]\n" + "".join(f'{name} = "{target}"\n' for name, target in scripts.items()))
    lines.append("[project.optional-dependencies]\ntest = []\ndocs = []\n")
    (tree / "pyproject.toml").write_text("".join(lines), encoding="utf-8")
    for name, text in (extra_files or {}).items():
        (tree / name).write_text(text, encoding="utf-8")
    subprocess.run(["uv", "lock", "--quiet"], cwd=tree, check=True)
    subprocess.run(["git", "init", "--quiet", "."], cwd=tree, check=True)
    return tree


def _run(cwd: Path, root: Path, *args: str, path: str | None = None, **extra: str):
    env = {k: v for k, v in os.environ.items() if k not in _SCRUBBED}
    env["HHEMT_HOOK_ENV_ROOT"] = str(root)
    if path is not None:
        env["PATH"] = path
    env.update(extra)
    return subprocess.run([str(WRAPPER), *args], cwd=cwd, env=env, capture_output=True, text=True)


def _records(root: Path) -> list[Path]:
    return sorted(p for p in root.glob("*/hhemt_meta/*") if p.is_dir() and not p.name.startswith("."))


def _tree_files(tree: Path) -> list[str]:
    return sorted(str(p.relative_to(tree)) for p in tree.rglob("*") if ".git" not in p.relative_to(tree).parts)


def test_every_console_script_a_project_declares_runs_its_own_entry_point(tmp_path):
    """Property: with two console scripts that point at different callables, each name on `PATH` under the record
    either does not exist or runs its OWN callable; a name never runs another script's output.

    Class: NEW CAPABILITY -- projects with more than one `console_scripts` entry. The plausible wrong
    implementation writes one shim per entry that runs `python -m {package}` for each, so every shim runs the
    package's single `__main__` and the second script silently prints the first one's output; red under the
    shipped wrapper, pasted in the round file.
    Why required: the shape's D2 criterion 1 is that every command "behaves as it does under an install of the same
    tree -- the same console script"; a shim that runs the wrong entry point reports success with another
    command's behaviour, and the next `console_scripts` entry hhemt adds would be wrong from its first run.
    What kills it: a shim that ignores the entry's module and callable.
    Second correct implementation: shims that import the entry's module and call its callable; or a wrapper that
    provides only the one `hhemt` shim the shape names. Both satisfy "absent or its own output".
    """
    tree = _make_toy(tmp_path / "toy_tree", scripts={"toy": "toy.cli:main", "toy-admin": "toy.admin:main"})
    root = tmp_path / "root"
    result = _run(
        tree, root, "sh", "-c", "toy; if command -v toy-admin >/dev/null; then toy-admin; else echo absent; fi"
    )
    assert result.returncode == 0, result.stderr
    first, second = result.stdout.strip().splitlines()
    assert first == "toy-cli-output"
    assert second in {"toy-admin-output", "absent"}, second


def test_the_record_comes_before_other_executables_of_the_same_name_on_path(tmp_path):
    """Property: when the caller's `PATH` already holds an executable named like the project's console script, the
    record's shim wins.

    Class: NEW CAPABILITY -- a `toy` on the caller's `PATH` ahead of everything else, which is what another
    tree's install or a stale conda environment looks like. The plausible wrong implementation appends the
    record's `bin/` to `PATH` instead of prepending it.
    Why required: residual 10 of the shape names a `hhemt` taken "from `PATH`, which may be another tree's
    install"; the record's shim exists to remove that.
    What kills it: `PATH="$PATH:$record/bin"`.
    Second correct implementation: a wrapper that calls the shim by absolute path through a `uv run` argument
    rewrite; the printed output is the same.
    """
    tree = _make_toy(tmp_path / "toy_tree")
    decoy = tmp_path / "decoy"
    decoy.mkdir()
    (decoy / "toy").write_text("#!/bin/sh\necho decoy-output\n", encoding="utf-8")
    (decoy / "toy").chmod(0o755)
    result = _run(tree, tmp_path / "root", "sh", "-c", "toy", path=f"{decoy}:{os.environ['PATH']}")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "toy-cli-output"


def test_a_changed_readme_makes_a_new_record_with_the_new_description(tmp_path):
    """Property: editing the readme file `[project] readme` names yields a second record in the same key, and the
    command sees the new long description.

    Class: NEW CAPABILITY -- the files the record's key covers besides `pyproject.toml`. The plausible wrong
    implementation keys on `pyproject.toml` alone, so an edited readme keeps serving the old record, whose
    `METADATA` embeds the old text; red under it for the second assertion.
    Why required: the wrapper keys the record on `pyproject.toml` "followed by each file named by
    `[project] readme` and `license-files`"; a record built from a readme the tree no longer has is a stale
    distribution record.
    What kills it: a meta key that omits the readme, or the license files.
    Second correct implementation: a key over the backend's own input list read from a dry run; the same edits
    yield a new key.
    """
    tree = _make_toy(tmp_path / "toy_tree", readme="# Toy\n\nfirst description\n")
    root = tmp_path / "root"
    code = "import importlib.metadata as m; print(m.metadata('toy').get_payload() or m.metadata('toy')['Description'])"
    first = _run(tree, root, "python", "-c", code)
    assert first.returncode == 0, first.stderr
    assert "first description" in first.stdout
    before = _tree_files(tree)
    (tree / "README.md").write_text("# Toy\n\nsecond description\n", encoding="utf-8")
    second = _run(tree, root, "python", "-c", code)
    assert second.returncode == 0, second.stderr
    assert "second description" in second.stdout
    assert len(_records(root)) == 2
    assert _tree_files(tree) == before


def test_the_first_run_writes_only_the_commands_output_and_leaves_no_temporary_build_directory(tmp_path):
    """Property: on a fresh root, the wrapper's stdout is exactly the command's output, and afterwards the
    key's `hhemt_meta/` holds no `.tmp.*` entry.

    Class: NEW CAPABILITY -- the first run, which is the only run that builds a record. The plausible wrong
    implementation lets the build backend's log lines reach stdout, which a hook's consumer then parses as the
    hook's output, or skips the `rm -rf` of its temporary directory.
    Why required: the commit-msg hook and `--print-env` are consumed programmatically, and the wrapper's record build
    ends by removing its temporary directory with `rm -rf -- "$tmp"`.
    What kills it: dropping the `>/dev/null` around the backend, or the final `rm -rf`.
    Second correct implementation: a wrapper that captures the backend's output into a file; stdout is the same.
    """
    tree = _make_toy(tmp_path / "toy_tree")
    root = tmp_path / "root"
    result = _run(tree, root, "python", "-c", "print('hello')")
    assert result.returncode == 0, result.stderr
    assert result.stdout == "hello\n"
    assert not list(root.glob("*/hhemt_meta/.tmp.*"))
    printed = _run(tree, root, "--print-env")
    assert printed.returncode == 0 and printed.stdout.count("\n") == 1, printed.stdout


def test_the_record_is_built_when_the_wrapper_starts_in_a_subdirectory_of_the_tree(tmp_path):
    """Property: the first run started from a subdirectory builds the record from the tree's own project, and the
    command sees it.

    Class: NEW CAPABILITY -- a working directory other than the tree root, which a hook started from a
    subdirectory is. The plausible wrong implementation runs the backend in the caller's directory, where there is
    no `pyproject.toml`, and exits 2.
    Why required: the wrapper runs the backend "with `cwd=$tree`".
    What kills it: dropping the `cd "$tree"` around the backend call.
    Second correct implementation: a backend call that is given the tree as an argument instead of a working
    directory.
    """
    tree = _make_toy(tmp_path / "toy_tree")
    sub = tree / "src" / "toy"
    result = _run(sub, tmp_path / "root", "python", "-c", "import importlib.metadata as m; print(m.version('toy'))")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "0.1.0"


def test_a_failed_build_leaves_the_tree_untouched_and_no_record_that_a_later_run_could_trust(tmp_path):
    """Property: when the backend cannot build a record (an invalid license expression, which the backend rejects
    after the lock was made), the wrapper exits 2 naming the failed build, the tree is unchanged, and no
    `*.dist-info` exists in any record; a run after the project is repaired then succeeds.

    Class: NEW CAPABILITY -- a failing backend. The plausible wrong implementation publishes the half-built
    directory, so a later run's `$record/*.dist-info/METADATA` test passes on a record built from nothing.
    Why required: the record build requires a valid record at `$record` or exit 2; a published empty record is trusted
    forever and its `.last_used` keeps it from ageing out.
    What kills it: publishing `$tmp/record` before the backend has succeeded.
    Second correct implementation: a wrapper that builds in the record's final location and removes it on
    failure; the same observations hold.
    """
    tree = _make_toy(tmp_path / "toy_tree", readme="# Toy\n")
    root = tmp_path / "root"
    pyproject = tree / "pyproject.toml"
    good = pyproject.read_text(encoding="utf-8")
    pyproject.write_text(good.replace('license = "MIT"', 'license = "Not A Real License"'), encoding="utf-8")
    before = _tree_files(tree)
    broken = _run(tree, root, "python", "-c", "pass")
    assert broken.returncode == 2, (broken.returncode, broken.stderr)
    assert _tree_files(tree) == before
    assert not list(root.glob("*/hhemt_meta/*/*.dist-info/METADATA"))
    pyproject.write_text(good, encoding="utf-8")
    assert _run(tree, root, "python", "-c", "pass").returncode == 0


def _tool_dir(tmp_path: Path, names: list[str]) -> str | None:
    """A directory of symlinks to the named tools only; None when a tool the wrapper needs is not installed."""
    bin_dir = tmp_path / "restricted_bin"
    bin_dir.mkdir()
    for name in names:
        found = shutil.which(name)
        if found is None:
            return None
        (bin_dir / name).symlink_to(found)
    return str(bin_dir)


def test_the_wrapper_finds_the_same_key_and_record_when_only_shasum_is_available(tmp_path):
    """Property: with `sha256sum` absent from `PATH` and `shasum` present, the wrapper succeeds and reuses the key
    directory and the record that a `sha256sum` run made.

    Class: NEW CAPABILITY -- a host with only `shasum`, which is every macOS machine. The plausible wrong
    implementation hard-codes `sha256sum` in one of the two digests (the key's or the record's), so the host
    fails or computes a different key and builds a second environment.
    Why required: the wrapper's key computation states the `shasum -a 256` fallback "for macOS", and the record key
    adds a second digest that must use it too.
    What kills it: calling `sha256sum` by name anywhere the fallback variable should be used.
    Second correct implementation: a digest through `python -c "import hashlib"`; the same 16 hex digits result.
    """
    tools = [
        "uv", "git", "sh", "cat", "cut", "sort", "paste", "find", "ls", "head", "touch", "rm", "mv", "mkdir",
        "dirname", "perl", "env", "python3.12",
    ]  # fmt: skip
    with_both = _tool_dir(tmp_path, tools + ["sha256sum", "shasum"])
    if with_both is None or shutil.which("shasum") is None:
        pytest.skip("a tool the restricted PATH needs is not installed")
    shutil.rmtree(tmp_path / "restricted_bin")
    only_shasum = _tool_dir(tmp_path, tools + ["shasum"])
    tree = _make_toy(tmp_path / "toy_tree")
    root = tmp_path / "root"
    first = _run(tree, root, "python", "-c", "pass", path=only_shasum)
    assert first.returncode == 0, first.stderr
    keys_shasum = sorted(p.name for p in root.iterdir())
    records_shasum = [p.name for p in _records(root)]
    shutil.rmtree(root)
    second = _run(tree, root, "python", "-c", "pass")
    assert second.returncode == 0, second.stderr
    assert sorted(p.name for p in root.iterdir()) == keys_shasum
    assert [p.name for p in _records(root)] == records_shasum
