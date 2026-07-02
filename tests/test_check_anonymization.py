"""Unit tests for the ADR-14 anonymization guard."""

from __future__ import annotations

import subprocess
from pathlib import Path

import scripts.check_anonymization as guard  # repo root is on sys.path under pytest

def _init_repo(tmp_path: Path, files: dict[str, str]) -> Path:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "scripts").mkdir(parents=True, exist_ok=True)
    (tmp_path / "scripts" / "anonymization_blocklist.txt").write_text(
        "# test blocklist\nquinnlab\nhhemt_projects\nTRITON-SWMM_toolkit_projects\n",
        encoding="utf-8",
    )
    for rel, content in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    return tmp_path

def test_planted_token_fails(tmp_path: Path, capsys) -> None:
    root = _init_repo(tmp_path, {"src/leak.py": "account = 'quinnlab'\n"})
    rc = guard.main(["--root", str(root)])
    err = capsys.readouterr().err
    assert rc == 1
    assert "quinnlab" in err
    assert "src/leak.py" in err

def test_clean_tree_passes(tmp_path: Path) -> None:
    root = _init_repo(tmp_path, {"src/ok.py": "import hhemt\nx = 1\n"})
    assert guard.main(["--root", str(root)]) == 0

def test_public_prefix_not_false_positive(tmp_path: Path) -> None:
    # Public hhemt / TRITON-SWMM_toolkit appear; private *_projects do NOT.
    content = "import hhemt\n# the TRITON-SWMM_toolkit repo\nhhemt.run()\n" * 50
    root = _init_repo(tmp_path, {"src/public.py": content})
    assert guard.main(["--root", str(root)]) == 0

def test_guard_imports_nothing_from_src() -> None:
    # Independence invariant (Q6): the guard reads the blocklist, not constants.
    src = Path(guard.__file__).read_text(encoding="utf-8")
    assert "import hhemt" not in src
    assert "from hhemt" not in src
