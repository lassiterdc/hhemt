"""Tests of the exploratory-analysis tutorial, each with a must-fire control.

The tutorial `docs/tutorials/exploratory-analysis.md` states code facts: the command option, the files the step
writes, the notebook cell it quotes, and the groups and variables of the committed example result that its plots read.
Each test locates a statement by a stable anchor (a code span, a code block, a table row), derives the fact from the
code or the committed data, and compares the two as sets. No test runs a solver or the step, and none compares a count
or a sentence of prose. Each page-side check is one callable that its test and its control both call.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import xarray as xr

_REPO = Path(__file__).resolve().parents[2]
_PAGE = "docs/tutorials/exploratory-analysis.md"
_STORE = _REPO / "docs" / "tutorials" / "assets" / "irene-example-result.zarr"
_NOTEBOOK = _REPO / "docs" / "tutorials" / "assets" / "eda.ipynb"
_REMOVED = "docs/explanation/exploratory-analysis.md"
_HEADING = "# Exploratory analysis of an event ensemble"

_CODE_BLOCK = re.compile(r"^```python\n(.*?)^```$", re.S | re.M)
_GROUP_VARIABLE = re.compile(r'tree\["([^"]+)"\]\.to_dataset\(\)\["([^"]+)"\]')
_IMAGE = re.compile(r"^!\[[^\]]*\]\((assets/[^)\s]+\.png)\)\s*$", re.M)
_SAVEFIG = re.compile(r'fig\.savefig\("(assets/[^"]+\.png)"')


def _text(path: str = _PAGE) -> str:
    return (_REPO / path).read_text(encoding="utf-8")


def code_blocks(text: str) -> list[str]:
    return _CODE_BLOCK.findall(text)


def used_groups_and_variables(text: str) -> set[tuple[str, str]]:
    """The (group, variable) pairs that the code blocks of a page read from the tree."""
    return {pair for block in code_blocks(text) for pair in _GROUP_VARIABLE.findall(block)}


def store_groups_and_variables() -> set[tuple[str, str]]:
    tree = xr.open_datatree(_STORE, engine="zarr", consolidated=False)
    return {(p.strip("/"), v) for p in tree.groups if p != "/" for v in tree[p].to_dataset().data_vars}


def missing_from_store(text: str) -> set[tuple[str, str]]:
    return used_groups_and_variables(text) - store_groups_and_variables()


def test_the_plots_read_groups_and_variables_the_committed_result_holds():
    """Property: every group and variable that the code blocks read exists in the committed example result.

    Class: REGRESSION on the data or the page. Wrong implementation: a page that reads a dropped group.
    Reachability: the pairs are found by the tree access pattern and compared with the store as sets.
    """
    assert used_groups_and_variables(_text()) != set()
    assert missing_from_store(_text()) == set()


def test_the_group_check_refuses_a_group_the_store_lacks():
    """Control: a code block that reads a group or a variable absent from the store is refused, a real one is not."""
    wrong_group = 'x = tree["tritonswmm/no_such_group"].to_dataset()["max_flow"]'
    wrong_variable = 'x = tree["tritonswmm/swmm_link"].to_dataset()["no_such_variable"]'
    right = 'x = tree["tritonswmm/swmm_link"].to_dataset()["max_flow"]'
    assert missing_from_store(f"```python\n{wrong_group}\n```\n") != set()
    assert missing_from_store(f"```python\n{wrong_variable}\n```\n") != set()
    assert missing_from_store(f"```python\n{right}\n```\n") == set()


def image_problems(text: str, page_dir: Path) -> list[str]:
    """Images that do not exist, and images and saved figures that are not the same set."""
    images, saved = set(_IMAGE.findall(text)), set(_SAVEFIG.findall(text))
    problems = [f"missing file {i}" for i in sorted(images) if not (page_dir / i).is_file()]
    if images != saved:
        problems.append(f"images {sorted(images)} differ from saved figures {sorted(saved)}")
    return problems or ([] if images else ["the page carries no image"])


def test_each_plot_is_a_file_the_page_code_saves():
    """Property: the plot images of the page exist and are the files that its code blocks save."""
    assert image_problems(_text(), (_REPO / _PAGE).parent) == []


def test_the_image_check_refuses_a_renamed_or_missing_plot():
    """Control: an image whose name differs from the saved figure, and an image that has no file, are each refused."""
    text, here = _text(), (_REPO / _PAGE).parent
    pointing_elsewhere = text.replace("](assets/eda-peak-flow-difference.png)", "](assets/other.png)")
    assert image_problems(pointing_elsewhere, here) != []
    assert image_problems(text.replace("assets/eda-peak-flow-difference.png", "assets/absent.png"), here) != []


def notebook_first_code(notebook: Path) -> str:
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    return "".join(next(c for c in cells if c["cell_type"] == "code")["source"]).strip()


def quoted_cell_problems(text: str, notebook: Path) -> list[str]:
    """The first python block of the page must equal the first code cell of the seeded notebook."""
    first = code_blocks(text)[0].strip()
    return [] if first == notebook_first_code(notebook) else ["the quoted cell differs from the notebook cell"]


def test_the_quoted_cell_is_the_first_code_cell_of_the_notebook_asset():
    """Property: the cell that the page quotes equals the first code cell of the committed notebook."""
    assert quoted_cell_problems(_text(), _NOTEBOOK) == []


def test_the_quoted_cell_check_refuses_an_edited_cell():
    """Control: a page whose quoted cell changed by one token is refused."""
    edited = _text().replace("load_eda_context(root)", "load_eda_context(root, strict=True)", 1)
    assert quoted_cell_problems(edited, _NOTEBOOK) != []


def written_paths_in_the_table(text: str) -> set[str]:
    rows = re.findall(r"^\| `([^`|]+)`(?:, `([^`|]+)`)? \|", text, re.M)
    return {p.rstrip("/") for row in rows for p in row if p}


def paths_the_code_names() -> set[str]:
    """Names of the files and directories that the step writes, read as literals from its source."""
    eda = (_REPO / "src/hhemt/eda").glob("*.py")
    source = "\n".join(p.read_text(encoding="utf-8") for p in eda)
    return {name for name in ("eda_report", "eda_report.html", "eda_local", "raw_byte_identity") if name in source}


def test_the_table_of_written_paths_names_files_the_code_writes():
    """Property: the output directories and files of the table appear as literals in the source of the step."""
    paths = written_paths_in_the_table(_text())
    assert {"eda.ipynb", "eda_local", "eda_report/eda_report.html"} <= paths
    assert {"eda_report", "eda_report.html", "eda_local", "raw_byte_identity"} == paths_the_code_names()
    cli = (_REPO / "src/hhemt/cli.py").read_text(encoding="utf-8")
    assert '"--analysis-dir"' in cli[cli.index('@app.command(name="eda")') :]
    assert "hhemt eda --analysis-dir {analysis_dir}" in _text()


def test_the_path_table_check_refuses_a_table_without_the_notebook_row():
    """Control: the table reader finds the rows of a table and finds none in prose."""
    table = "| `eda.ipynb` | notebook |\n| `eda_local/` | package |\n"
    assert written_paths_in_the_table(table) == {"eda.ipynb", "eda_local"}
    assert written_paths_in_the_table("The notebook is eda.ipynb.") == set()


def links_to_the_removed_page(pages: dict[str, str]) -> list[str]:
    """Pages whose text links the removed explanation page."""
    return [name for name, text in sorted(pages.items()) if "explanation/exploratory-analysis" in text]


def test_the_tutorial_is_navigable_and_the_explanation_page_is_gone():
    """Property: the page has its ruled heading and navigation entry, and no tracked page links the removed page."""
    mkdocs = (_REPO / "mkdocs.yml").read_text(encoding="utf-8")
    assert _text().splitlines()[0] == _HEADING
    assert "tutorials/exploratory-analysis.md" in mkdocs and "explanation/exploratory-analysis.md" not in mkdocs
    assert not (_REPO / _REMOVED).exists()
    listed = subprocess.run(
        ["git", "-C", str(_REPO), "ls-files", "-z", "--", "docs", "README.md"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split("\0")
    pages = {p: _text(p) for p in listed if p.endswith(".md") and (_REPO / p).exists()}
    assert pages
    assert links_to_the_removed_page(pages) == []


def test_the_link_search_reports_a_planted_link():
    """Control: a page that links the removed page is reported, and a page that links the tutorial is not."""
    planted = {
        "a.md": "[x](../explanation/exploratory-analysis.md#a)",
        "b.md": "[x](../tutorials/exploratory-analysis.md)",
    }
    assert links_to_the_removed_page(planted) == ["a.md"]
