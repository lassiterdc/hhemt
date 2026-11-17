"""Reviewer tests for package 1i.1: the per-node manhole guide against the code, and must-fire mutations.

Part 1 compares statements of docs/how-to/per-node-manhole-values.md with the functions they describe.
Part 2 runs the coder's tests/test_manhole_table_docs.py inside a scratch copy of the tree, once per mutation,
and requires the expected test to go red. A mutation that leaves the file green is a hole in the coder's tests.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
_GUIDE = _REPO / "docs" / "how-to" / "per-node-manhole-values.md"


def _text() -> str:
    return _GUIDE.read_text(encoding="utf-8")


def test_the_guide_header_and_column_order_match_the_reader():
    from hhemt.scenario import MANHOLE_TABLE_COLUMNS

    assert ",".join(MANHOLE_TABLE_COLUMNS) in _text()
    cols = re.findall(r"^\| `(\w+)` \|", _text(), flags=re.MULTILINE)
    assert tuple(cols) == MANHOLE_TABLE_COLUMNS


def test_the_guide_error_sample_has_the_shape_the_code_emits():
    from hhemt.scenario import ManholeTableContext, manhole_table_problems

    nodes = [f"J{i}" for i in range(1, 21)]
    rows = []
    from hhemt.scenario import ManholeTableRow

    for i, n in enumerate(nodes[:18], start=2):
        rows.append(ManholeTableRow(line=i, node_id=n, diameter="1", loss="0.1", extra_cells=0))
    ctx = ManholeTableContext(table=Path("t.csv"), model="model.inp", diameter_constant=1.2, loss_constant=0.1)
    (msg,) = manhole_table_problems(rows, nodes, False, ctx)
    text = _text()
    assert "it gives a diameter and a loss coefficient for 18 of the 20 coupling nodes of model.inp" in msg
    found = re.search(r"- (it gives a diameter.*)", text)
    assert found is not None
    sample = found.group(1)
    # same sentence skeleton once the two node lists are masked
    mask = lambda s: re.sub(r"no row: [^.]*\.", "no row: X.", s)  # noqa: E731
    assert mask(sample.replace("J7, J12", "X")) == mask(msg.replace("J19, J20", "X"))
    assert "(1.2 m)" in sample and "(0.1)" in sample


def test_the_guide_names_ten_nodes_and_the_code_names_ten():
    from hhemt.scenario import _MANHOLE_TABLE_MAX_NAMED_NODES

    assert _MANHOLE_TABLE_MAX_NAMED_NODES == 10
    assert "up to 10 nodes" in _text()


def test_the_fill_toggle_default_and_the_unset_table_refusal():
    from hhemt.config.analysis import analysis_config

    assert analysis_config.model_fields["manhole_table_fill_with_constants"].default is False
    assert "`false`" in _text()
    with pytest.raises(ValueError, match="manhole_table is required when"):
        analysis_config.model_validate({"manhole_table_fill_with_constants": True})


def test_the_capable_build_refusal_is_not_a_preflight_refusal():
    """The guide says Preflight refuses a build that ignores the key. Count the call sites outside scenario.py."""
    src = _REPO / "src" / "hhemt"
    hits = [
        p.name
        for p in src.rglob("*.py")
        if p.name != "scenario.py" and "require_manhole_table_capable_triton" in p.read_text(encoding="utf-8")
    ]
    assert hits == [], hits
    assert "Preflight refuses a build that ignores the key" not in _text()


# ---------------------------------------------------------------------------------------------------------------
# Part 2: mutation controls against the coder's tests
# ---------------------------------------------------------------------------------------------------------------


def _scratch_copy(tmp_path: Path, mutate) -> subprocess.CompletedProcess:
    root = tmp_path / "tree"
    (root / "tests").mkdir(parents=True)
    shutil.copy(_REPO / "tests" / "test_manhole_table_docs.py", root / "tests")
    shutil.copytree(_REPO / "hooks", root / "hooks", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(_REPO / "src", root / "src", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(_REPO / "mkdocs.yml", root / "mkdocs.yml")
    shutil.copytree(_REPO / "scripts", root / "scripts", ignore=shutil.ignore_patterns("__pycache__"))
    (root / "docs" / "how-to").mkdir(parents=True)
    for f in (_REPO / "docs" / "how-to").glob("*.md"):
        shutil.copy(f, root / "docs" / "how-to")
    mutate(root)
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_manhole_table_docs.py",
            "-q",
            "-p",
            "no:cacheprovider",
            "--no-header",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "PYTHONPATH": str(root / "src")},
    )


def _failed(res: subprocess.CompletedProcess) -> set[str]:
    return set(re.findall(r"^FAILED [^:]+::(\w+)", res.stdout, flags=re.MULTILINE))


def test_control_unmutated_scratch_copy_is_green(tmp_path):
    res = _scratch_copy(tmp_path, lambda r: None)
    assert res.returncode == 0, res.stdout[-1500:]


def test_mutation_remove_the_nav_entry(tmp_path):
    def m(root):
        p = root / "mkdocs.yml"
        p.write_text(re.sub(r"^.*per-node-manhole-values\.md\n", "", p.read_text(), flags=re.MULTILINE))

    assert "test_the_manhole_guide_is_in_the_navigation" in _failed(_scratch_copy(tmp_path, m))


def test_mutation_rename_the_page(tmp_path):
    def m(root):
        (root / "docs" / "how-to" / "per-node-manhole-values.md").rename(root / "docs" / "how-to" / "renamed.md")

    failed = _failed(_scratch_copy(tmp_path, m))
    assert "test_the_manhole_guide_exists_and_names_both_manhole_fields" in failed


def test_mutation_remove_the_index_row(tmp_path):
    def m(root):
        p = root / "docs" / "how-to" / "index.md"
        p.write_text(re.sub(r"^.*per-node-manhole-values\.md.*\n", "", p.read_text(), flags=re.MULTILINE))

    assert "test_the_manhole_guide_is_in_the_how_to_index" in _failed(_scratch_copy(tmp_path, m))


@pytest.mark.parametrize("field", ["manhole_table_fill_with_constants", "swmm_coupling_lag"])
def test_mutation_remove_a_field_from_its_model(tmp_path, field):
    def m(root):
        p = root / "src" / "hhemt" / "config" / "analysis.py"
        s = p.read_text()
        s2 = s.replace(f"    {field}:", f"    _gone_{field}:", 1)
        assert s2 != s
        p.write_text(s2)

    failed = _failed(_scratch_copy(tmp_path, m))
    assert "test_added_fields_are_declared_on_the_models" in failed


def test_mutation_the_guide_loses_the_header_text(tmp_path):
    def m(root):
        p = root / "docs" / "how-to" / "per-node-manhole-values.md"
        p.write_text(p.read_text().replace("node_id,manhole_diameter_m,manhole_loss_coefficient", "node_id,d,l"))

    assert "test_the_manhole_guide_exists_and_names_both_manhole_fields" in _failed(_scratch_copy(tmp_path, m))
