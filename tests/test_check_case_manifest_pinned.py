"""Unit tests for the case-manifest pin guard (T-INTEGRITY-GUARD).

Mirrors tests/test_check_anonymization.py: a tmp_path git repo, `git add -A` with no
commit (`git ls-files` reads the INDEX, so a commit buys nothing), and the guard driven
through `main(["--root", ...])`. No Norfolk data and no network.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import scripts.check_case_manifest_pinned as guard  # repo root is on sys.path under pytest

_GOOD_SHA = "a" * 64


def _init_repo(tmp_path: Path, files: dict[str, str]) -> Path:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    for rel, content in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    return tmp_path


def _case_yaml(manifest_block: str) -> str:
    return 'case_name: "c"\nres_identifier: "x"\nhost: "hydroshare"\n' + manifest_block


def test_populated_manifest_passes(tmp_path: Path) -> None:
    root = _init_repo(tmp_path, {"test_data/c/case.yaml": _case_yaml(f'manifest:\n  data/a.tif: "{_GOOD_SHA}"\n')})
    assert guard.main(["--root", str(root)]) == 0


def test_empty_manifest_fails(tmp_path: Path, capsys) -> None:
    root = _init_repo(tmp_path, {"test_data/c/case.yaml": _case_yaml("manifest: {}\n")})
    rc = guard.main(["--root", str(root)])
    err = capsys.readouterr().err
    assert rc == 1
    assert "manifest-empty" in err
    assert "generate_case_manifest" in err  # the message names the remedy


def test_absent_manifest_key_fails(tmp_path: Path, capsys) -> None:
    """The branch CaseManifest structurally cannot catch: default_factory=dict."""
    root = _init_repo(tmp_path, {"test_data/c/case.yaml": _case_yaml("")})
    rc = guard.main(["--root", str(root)])
    assert rc == 1
    assert "manifest-key-absent" in capsys.readouterr().err


def test_md5_length_digest_fails(tmp_path: Path, capsys) -> None:
    """bagit ships manifest-md5; a 32-char value pasted from it is not a sha256."""
    root = _init_repo(tmp_path, {"test_data/c/case.yaml": _case_yaml(f'manifest:\n  data/a.tif: "{"b" * 32}"\n')})
    rc = guard.main(["--root", str(root)])
    assert rc == 1
    assert "bad-digest" in capsys.readouterr().err


def test_uppercase_digest_fails(tmp_path: Path, capsys) -> None:
    root = _init_repo(tmp_path, {"test_data/c/case.yaml": _case_yaml(f'manifest:\n  data/a.tif: "{"A" * 64}"\n')})
    assert guard.main(["--root", str(root)]) == 1
    # Exit 1 alone stays green under an always-fail guard and under one that reports the
    # right exit for the wrong reason. `bad-digest` is the enumerated Finding.code, not the
    # prose detail beside it, so pinning it does not red on a message-wording change.
    assert "bad-digest" in capsys.readouterr().err


def test_absolute_key_fails(tmp_path: Path, capsys) -> None:
    root = _init_repo(tmp_path, {"test_data/c/case.yaml": _case_yaml(f'manifest:\n  /etc/a: "{_GOOD_SHA}"\n')})
    rc = guard.main(["--root", str(root)])
    assert rc == 1
    assert "bad-key" in capsys.readouterr().err


def test_parent_escape_key_fails(tmp_path: Path, capsys) -> None:
    root = _init_repo(tmp_path, {"test_data/c/case.yaml": _case_yaml(f'manifest:\n  ../a: "{_GOOD_SHA}"\n')})
    assert guard.main(["--root", str(root)]) == 1
    # Same reason as test_uppercase_digest_fails: pin the code, not the exit alone.
    assert "bad-key" in capsys.readouterr().err


def test_manifest_not_a_mapping_fails(tmp_path: Path, capsys) -> None:
    root = _init_repo(tmp_path, {"test_data/c/case.yaml": _case_yaml("manifest: []\n")})
    rc = guard.main(["--root", str(root)])
    assert rc == 1
    assert "manifest-not-a-mapping" in capsys.readouterr().err


def test_zero_population_fails_rather_than_passing_vacuously(tmp_path: Path, capsys) -> None:
    """An empty scan and a clean scan are indistinguishable at exit 0. This is the
    branch that keeps them distinguishable, and it is on the live T-NORFOLK-CRUFT
    trajectory: deleting the last case study must red, not go green."""
    root = _init_repo(tmp_path, {"src/ok.py": "x = 1\n"})
    rc = guard.main(["--root", str(root)])
    assert rc == 1
    assert "zero-population" in capsys.readouterr().err


def test_nested_worktree_case_yaml_is_not_scanned(tmp_path: Path) -> None:
    """A nested checkout's case.yaml is out of population two ways, and this test
    exercises the HARDER one. In the live repo a `.claude/worktrees/{slug}` checkout is a
    separate git worktree with its own index, so `git ls-files` never reports it
    (measured: 3 case.yaml on disk, 1 in `git ls-files`). Here it IS tracked, so the
    anchored `_CASE_YAML_RE` is the only barrier left, which is what this asserts."""
    root = _init_repo(
        tmp_path,
        {
            "test_data/c/case.yaml": _case_yaml(f'manifest:\n  data/a.tif: "{_GOOD_SHA}"\n'),
            ".claude/worktrees/w/test_data/c/case.yaml": _case_yaml("manifest: {}\n"),
        },
    )
    assert guard.tracked_case_yamls(root) == ["test_data/c/case.yaml"]
    assert guard.main(["--root", str(root)]) == 0


def test_guard_imports_nothing_from_src() -> None:
    """Independence invariant: the auditor must not be defined by the audited schema."""
    src = Path(guard.__file__).read_text(encoding="utf-8")
    assert "import hhemt" not in src
    assert "from hhemt" not in src


def test_list_mode_exits_zero_on_a_failing_tree(tmp_path: Path) -> None:
    root = _init_repo(tmp_path, {"test_data/c/case.yaml": _case_yaml("manifest: {}\n")})
    assert guard.main(["--root", str(root), "--list"]) == 0
