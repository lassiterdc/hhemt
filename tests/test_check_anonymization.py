"""Unit tests for the ADR-14 anonymization guard."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import scripts.check_anonymization as guard  # repo root is on sys.path under pytest

def _init_repo(tmp_path: Path, files: dict[str, str]) -> Path:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "scripts").mkdir(parents=True, exist_ok=True)
    (tmp_path / "scripts" / "anonymization_blocklist.txt").write_text(
        "# test blocklist\n"
        "ZZTESTTOKENALPHA\nZZTESTTOKENBETA\nZZTESTTOKENGAMMA\nZZTESTTOKENDELTA\n"
        "ZZTESTTOKENEPSILON\nZZTESTTOKENZETA\nZZTESTTOKENETA\nZZTESTTOKENTHETA\n",
        encoding="utf-8",
    )
    for rel, content in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    return tmp_path

def test_planted_token_fails(tmp_path: Path, capsys) -> None:
    root = _init_repo(tmp_path, {"src/leak.py": "account = 'ZZTESTTOKENALPHA'\n"})
    rc = guard.main(["--root", str(root)])
    err = capsys.readouterr().err
    assert rc == 1
    assert "ZZTESTTOKENALPHA" in err
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


def test_header_declared_floor_is_honored(tmp_path: Path) -> None:
    """A `# min-tokens: N` header in the carrier overrides the module constant.

    The floor exists so a truncated or empty carrier is a FAILURE rather than a weaker
    pass. Declaring it in the carrier's own header is what keeps floor and list consistent
    when the carrier is edited or relocated -- a module constant lives in a different file
    (and, after relocation, a different repository) from the list it counts.
    """
    carrier = tmp_path / "carrier.txt"
    carrier.write_text("# min-tokens: 9\nZZA\nZZB\nZZC\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        guard.load_blocklist(carrier)
    assert "below the floor of 9" in str(exc.value)


def test_header_declared_floor_may_sit_below_the_module_constant(tmp_path: Path) -> None:
    """The header is authoritative in BOTH directions, not merely a tightening.

    A carrier that legitimately shrinks -- retiring an obsolete identifier -- must be able
    to lower its own floor in the same edit, or the guard reds on a correct change.
    """
    carrier = tmp_path / "carrier.txt"
    carrier.write_text("# min-tokens: 2\nZZA\nZZB\n", encoding="utf-8")
    assert guard.load_blocklist(carrier) == ["ZZA", "ZZB"]


def test_headerless_carrier_falls_back_to_the_module_constant(tmp_path: Path) -> None:
    """No header means the pre-existing constant applies, so the change is non-breaking."""
    carrier = tmp_path / "carrier.txt"
    carrier.write_text("# no declared count\nZZA\nZZB\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        guard.load_blocklist(carrier)
    assert f"below the floor of {guard._MIN_EXPECTED_TOKENS}" in str(exc.value)
