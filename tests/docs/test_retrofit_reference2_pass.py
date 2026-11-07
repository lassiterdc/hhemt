"""Tests of the Reference pages: nomenclature, moved pages, coupling-mode table and standalone configuration tables.

Each test pins a statement of a Reference page, or a structure the pass created, to the fact that decides it. None
builds an analysis, compiles a solver or starts the documentation build. Each finder has a control that feeds it the
mutated input it exists to refuse.

- The glossary defines the matrix, the experiment, the analysis and the simulation, in the relation the developer set.
- `docs/reference/` holds the Rerun triggers page and the coupled-inflow-nodes page, the FAQ and the explanation copy
  are gone, and the navigation lists the two pages.
- The coupling-mode table of the coupled-inflow-nodes page lists the modes that `system_config` accepts.
- The example-report page carries no Python code block.
- The configuration-schema hook writes a standalone full-width page for the system and the analysis tables, and the
  schema page links both.
"""

from __future__ import annotations

import importlib.util
import re
import sys
import typing
from pathlib import Path

import pytest
import yaml

_REPO = Path(__file__).resolve().parents[2]
_REF = _REPO / "docs" / "reference"


def _page(name: str) -> str:
    return (_REF / name).read_text(encoding="utf-8")


def entries(text: str) -> dict[str, str]:
    """`{heading: body}` of the `###` entries of a page, the body up to the next heading of any level."""
    out: dict[str, str] = {}
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = re.fullmatch(r"###\s+(.+)", line.strip())
        if m is None:
            continue
        end = next((j for j in range(i + 1, len(lines)) if lines[j].startswith("#")), len(lines))
        out[m.group(1)] = " ".join(" ".join(lines[i + 1 : end]).split())
    return out


def check_nomenclature(text: str) -> list[str]:
    """Problems: a missing term heading, or a definition that does not state the relation of the term."""
    found = entries(text)
    problems = [f"no `### {t}` entry" for t in ("matrix", "experiment", "analysis", "simulation") if t not in found]
    want = {
        "matrix": r"^A set of \[experiments\]\(#experiment\)",
        "experiment": r"^One or more \[analyses\]\(#analysis\)",
        "analysis": r"^One or more \[simulations\]\(#simulation\)",
    }
    for term, pattern in want.items():
        if term in found and not re.search(pattern, found[term]):
            problems.append(f"the `{term}` entry does not open with its relation: {found[term][:60]!r}")
    return problems


def test_the_glossary_defines_the_four_terms_in_the_relation_the_developer_set():
    """Property: matrix = set of experiments, experiment = one or more analyses, analysis = one or more simulations.

    Class: the three relation entries and the heading of the fourth term. Why required: every page uses the four terms
    with these meanings, and a reader looks the meaning up here. What kills it: an entry that reverts to the earlier
    wording (an analysis as a set of weather events) or a missing heading. A second correct implementation under which
    it still passes: any wording that opens with the relation and links the lower term.
    """
    assert check_nomenclature(_page("glossary.md")) == []


def test_control_a_glossary_without_a_term_or_with_the_earlier_wording_is_refused():
    """Control: a glossary that drops a term heading or reverts a relation to the earlier wording is refused."""
    text = _page("glossary.md")
    assert check_nomenclature(text.replace("### simulation", "### execution", 1))
    assert check_nomenclature(text.replace("One or more [simulations](#simulation)", "A set of weather events", 1))
    assert check_nomenclature(text.replace("A set of [experiments](#experiment)", "A table of experiments", 1))


def test_the_faq_is_gone_and_the_pages_moved_into_the_reference():
    """Property: `docs/faq/` and the explanation copy are gone, and both pages live in the Reference."""
    assert not (_REPO / "docs" / "faq").exists()
    assert not (_REPO / "docs" / "explanation" / "coupled-inflow-nodes.md").exists()
    assert _page("rerun-triggers.md").startswith("# Rerun triggers")
    assert _page("coupled-inflow-nodes.md").startswith("# Coupled inflow nodes")


def nav_pages(nav) -> list[str]:
    """Every page path in a mkdocs `nav` value, in order."""
    out: list[str] = []
    for item in nav:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict):
            for value in item.values():
                out += nav_pages(value) if isinstance(value, list) else [value]
    return out


def _nav() -> list:
    text = (_REPO / "mkdocs.yml").read_text(encoding="utf-8")
    text = re.sub(r"!!python/[\w/.:-]+", "", text)  # tags in mkdocs.yml are not needed to read the nav
    return yaml.safe_load(text)["nav"]


def test_the_navigation_lists_the_moved_pages_and_no_faq():
    """Property: the navigation lists both moved pages and no page under `faq/` or the old explanation path."""
    pages = nav_pages(_nav())
    assert "reference/rerun-triggers.md" in pages
    assert "reference/coupled-inflow-nodes.md" in pages
    assert not [p for p in pages if p.startswith("faq/") or p == "explanation/coupled-inflow-nodes.md"]


def test_control_a_navigation_with_the_faq_is_refused():
    """Control: a navigation that still lists a `faq/` page is found by the same filter."""
    pages = nav_pages([{"FAQ": ["faq/index.md"]}, {"Reference": ["reference/index.md"]}])
    assert [p for p in pages if p.startswith("faq/")]


def test_the_renamed_and_dropped_strings_are_gone_from_the_pages():
    """Property: the two dropped strings occur on no page, and the API page is Error handling."""
    for path in sorted((_REPO / "docs").rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        assert "Example experiments" not in text, path
        assert "The toolkit and its errors" not in text, path
    assert _page("api-toolkit.md").startswith("# API reference: Error handling")
    assert "[Error handling](api-toolkit.md)" in _page("api.md")


def python_fences(text: str) -> list[str]:
    """The opening lines of the Python code fences of a page."""
    return re.findall(r"^\s*```python.*$", text, re.M)


def test_the_example_report_page_carries_no_python_code():
    """Property: no Python fence on the page that describes the report. Control: a page with a fence is refused."""
    assert python_fences(_page("example-report.md")) == []


def test_control_a_python_fence_is_found():
    """Control: a page with a Python fence is found by `python_fences`."""
    assert python_fences("text\n\n```python\nanalysis.render_report()\n```\n")


def coupling_modes_on_page(text: str) -> set[str]:
    """The `swmm_coupling_mode` values in the first column of the mode table of the coupled-inflow-nodes page."""
    return set(re.findall(r"^\| `(\w+)`(?:,[^|]*)? \|", text.split("## Declaring the coupling nodes", 1)[1], re.M)) - {
        "swmm_coupling_mode"
    }


def test_the_coupling_mode_table_lists_the_modes_the_system_config_accepts():
    """Property: the table lists exactly the `Literal` values of `swmm_coupling_mode`, and the default is named.

    Class: the three values. Why required: a mode missing from the table is a mode a reader never learns, and a mode
    that the code dropped misleads. What kills it: adding or removing a value in `system_config` alone.
    """
    from hhemt.config.system import system_config

    field = system_config.model_fields["swmm_coupling_mode"]
    modes = set(typing.get_args(field.annotation))
    text = _page("coupled-inflow-nodes.md")
    assert coupling_modes_on_page(text) == modes
    assert f"| `{field.default}`, the default |" in text


def test_control_a_mode_missing_from_the_table_is_found():
    """Control: a table without the `list` row reads as the page set minus `list`, so a dropped row is visible."""
    text = _page("coupled-inflow-nodes.md")
    dropped = "\n".join(line for line in text.splitlines() if not line.startswith("| `list` |"))
    assert coupling_modes_on_page(dropped) == coupling_modes_on_page(text) - {"list"}


def _hook():
    path = _REPO / "hooks" / "config_reference.py"
    spec = importlib.util.spec_from_file_location("_hhemt_config_reference_hook", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def network_references(html: str) -> list[str]:
    """The `http` URLs that an HTML page loads as assets (`src` or stylesheet `href`)."""
    return re.findall(r"""(?:src|href)\s*=\s*["'](https?://[^"']+)""", html)


@pytest.mark.parametrize("key", ["system", "analysis"])
def test_the_standalone_table_page_holds_every_field_and_loads_nothing_from_the_network(key):
    """Property: the standalone page has a data row per field of the model, and its assets are inline.

    Class: the two models of `_STANDALONE_PAGES`. Why required: the page exists to show the long table at full width
    and the build must not depend on a network at view time. What kills it: a hook that drops a field or emits the
    CDN form of the Tabulator assets. A second correct implementation under which it still passes: any renderer of
    the same table, since the test reads the field names out of the page text.
    """
    from hhemt.config.analysis import analysis_config
    from hhemt.config.system import system_config

    hook = _hook()
    hook._closure_models()
    model = {"system": system_config, "analysis": analysis_config}[key]
    title = {k: t for k, t, _ in hook._STANDALONE_PAGES}[key]
    html = hook._standalone_table_page(model, title)
    names = [r["name"] for r in hook.field_records(model)]
    assert names
    for name in names:
        assert f'"Field": "{name}"' in html, name
    assert network_references(html) == []


def test_control_a_cdn_page_is_found_to_load_from_the_network():
    """Control: a page that loads Tabulator from the CDN is found by `network_references`."""
    assert network_references('<script src="https://cdn.jsdelivr.net/npm/tabulator-tables@6.4.0/x.js"></script>')


def test_the_schema_page_links_both_standalone_pages_and_they_are_ignored_build_output():
    """Property: the schema page links both standalone pages and `.gitignore` lists both files as build output."""
    hook = _hook()
    hook._closure_models()
    markdown = hook._render()
    gitignore = (_REPO / ".gitignore").read_text(encoding="utf-8").splitlines()
    for _key, _title, name in hook._STANDALONE_PAGES:
        assert f"]({name})" in markdown, name
        assert f"docs/reference/{name}" in gitignore, name
