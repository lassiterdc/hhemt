"""The published surfaces that are not tracked `.md` pages follow the same style rules as the pages.

`tests/docs/test_docs_style.py` reads tracked `.md` files. Five more surfaces reach the built site
or a generated report and escape it. This test renders each in the test process, with no site
build and no network, and grades its text with the same rules and the same allowlist file.

Surfaces, and what the allowlist `line` field holds for each (the entry's `surface` field names it):

- `config-schema`: the page `docs/reference/config-schema.md`, rendered by `hooks/config_reference.py::_render()`
  (the hook's own function). `page` is that path; `line` is a line of the rendered markdown.
- `api-docstring`: the docstrings that the `:::` directives of the `docs/reference/api-*.md` pages render, taken
  from `scripts/check_docs_content.py::rendered_docstrings()`, the population the content gate already derives
  with `griffe` under the options of `mkdocs.yml`. `page` is the API page that renders the object; `line` is a
  line of the docstring text (trailing whitespace removed). Docstrings of one page are joined with a blank line,
  so no phrase wraps from one docstring into the next.
- `caption`: a report caption file `src/hhemt/report_templates/captions/*.rst`. `page` is that file's path:
  each caption belongs to one report renderer and is keyed by the file that a repair edits. `line` is a raw
  line of the file.
- `workflow-description`: `src/hhemt/report_templates/workflow_description.rst.j2`, the text that every
  generated report prints as its Overview. `page` is that path; `line` is a raw template line, as for captions.
- `report-css`: the comments that `src/hhemt/report_templates/report.css.j2` ships in every report's `report.css`.
  `workflow.py` renders the template with `string.Template.safe_substitute` and writes it, so every `/* */` comment
  is present in the shipped file. The text is extracted by matching
  each non-greedy slash-star to star-slash span, newlines included, over the raw
  template and joining the comment bodies, each with its delimiters removed and the leading `*` of continuation
  lines removed, separated by a blank line (so no phrase wraps from one comment into the next). The selectors,
  properties and placeholders are not read. `page` is the template path; `line` is a line of that extracted text.
  The Jinja comments of `eda_report.html.j2` are dropped at render and are not a surface.
  The other files of `report_templates/` carry no prose that a reader of a report sees (HTML comments,
  an HTML shell with no sentence, and the vendored third-party Tabulator files).
- `eda-kind-table`: the table `hooks/eda_kind_table.py::render_table()` puts into `docs/how-to/running-eda.md`.
  `page` is that page; `line` is a table row.
- `marker-governance-table`: the table `hooks/marker_governance_table.py::render_table()` puts into
  `docs/contributing.md`. `page` is that page; `line` is a table row.

The entries of the tracked pages have no `surface` field and keep their meaning. A surface entry is graded
only by this test, so a stale surface entry fails here. Repairs happen at the source (a `Field` description,
a docstring, a hook string, a caption), which the pass of the reference subset performs.
"""

from __future__ import annotations

import functools
import importlib.util
import re
import sys
from pathlib import Path

import pytest

from tests.docs.test_docs_style import RULES, check_style, line_digest, load_allowlist, style_hits

_REPO = Path(__file__).resolve().parents[2]
_CAPTIONS = "src/hhemt/report_templates/captions"
SURFACES = (
    "config-schema",
    "api-docstring",
    "caption",
    "workflow-description",
    "report-css",
    "eda-kind-table",
    "marker-governance-table",
)
_WORKFLOW_DESCRIPTION = "src/hhemt/report_templates/workflow_description.rst.j2"
_REPORT_CSS = "src/hhemt/report_templates/report.css.j2"


def _load(name: str, relative: str):
    """Load a repository file that is not a package, by path, as mkdocs loads its hooks."""
    spec = importlib.util.spec_from_file_location(name, _REPO / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules.pop(name, None)
    spec.loader.exec_module(module)
    return module


def _config_schema() -> dict[str, str]:
    hook = _load("_hhemt_config_reference_hook", "hooks/config_reference.py")
    return {"docs/reference/config-schema.md": hook._render()}


def _api_pages() -> dict[str, list[str]]:
    """API page path -> the modules its `:::` directives name."""
    pages: dict[str, list[str]] = {}
    for path in sorted((_REPO / "docs/reference").glob("api-*.md")):
        modules = [
            line.split(":::", 1)[1].strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.startswith(":::")
        ]
        pages[path.relative_to(_REPO).as_posix()] = modules
    return pages


def _api_docstrings() -> dict[str, str]:
    gate = _load("_hhemt_docs_gate", "scripts/check_docs_content.py")
    owner = {module: page for page, modules in _api_pages().items() for module in modules}
    by_page: dict[str, list[str]] = {page: [] for page in _api_pages()}
    for qualname, _file, _lineno, doc in gate.rendered_docstrings():
        module = max((m for m in owner if qualname == m or qualname.startswith(m + ".")), key=len)
        by_page[owner[module]].append(doc)
    return {page: "\n\n".join(docs) + "\n" for page, docs in by_page.items() if docs}


def _captions() -> dict[str, str]:
    return {
        path.relative_to(_REPO).as_posix(): path.read_text(encoding="utf-8")
        for path in sorted((_REPO / _CAPTIONS).glob("*.rst"))
    }


def _workflow_description() -> dict[str, str]:
    return {_WORKFLOW_DESCRIPTION: (_REPO / _WORKFLOW_DESCRIPTION).read_text(encoding="utf-8")}


_CSS_COMMENT = re.compile(r"/\*(.*?)\*/", re.S)


def _report_css_comments() -> dict[str, str]:
    text = (_REPO / _REPORT_CSS).read_text(encoding="utf-8")
    bodies = []
    for body in _CSS_COMMENT.findall(text):
        lines = [re.sub(r"^\s*\*\s?", "", line).rstrip() for line in body.strip().splitlines()]
        bodies.append("\n".join(lines).strip())
    return {_REPORT_CSS: "\n\n".join(b for b in bodies if b) + "\n"}


def _eda_kind_table() -> dict[str, str]:
    hook = _load("_hhemt_eda_kind_table_hook", "hooks/eda_kind_table.py")
    return {"docs/" + hook.TARGET_URI: hook.render_table()}


def _marker_governance_table() -> dict[str, str]:
    hook = _load("_hhemt_marker_governance_hook", "hooks/marker_governance_table.py")
    return {"docs/" + hook.TARGET_URI: hook.render_table()}


_RENDERERS = {
    "config-schema": _config_schema,
    "api-docstring": _api_docstrings,
    "caption": _captions,
    "workflow-description": _workflow_description,
    "report-css": _report_css_comments,
    "eda-kind-table": _eda_kind_table,
    "marker-governance-table": _marker_governance_table,
}


@functools.cache
def _rendered(surface: str) -> dict[str, str]:
    """One render per surface per session: loading the API population with `griffe` dominates the run time."""
    return _RENDERERS[surface]()


def surface_pages(surface: str) -> dict[str, str]:
    """The text of one published surface, keyed by the page or file the allowlist names."""
    pages = dict(_rendered(surface))
    assert pages and all(text.strip() for text in pages.values()), f"surface {surface} rendered no text"
    return pages


def test_the_surface_list_is_the_renderer_list():
    assert sorted(SURFACES) == sorted(_RENDERERS)


@pytest.mark.parametrize("surface", SURFACES)
def test_published_surface_follows_the_style_brief(surface):
    problems = check_style(surface_pages(surface), load_allowlist(), surface=surface)
    assert problems == [], "\n".join(problems[:50]) + f"\n({len(problems)} problems)"


def test_every_surface_entry_names_a_known_surface():
    unknown = [e for e in load_allowlist() if "surface" in e and e["surface"] not in SURFACES]
    assert unknown == [], unknown


def test_the_api_pages_name_modules_and_each_renders_docstrings():
    pages = _api_pages()
    assert pages and all(modules for modules in pages.values()), pages
    assert set(surface_pages("api-docstring")) == set(pages)


def test_the_caption_glob_reads_every_caption_file():
    """The glob and the directory listing agree, and no caption is empty."""
    listing = {p.name for p in (_REPO / _CAPTIONS).iterdir() if p.suffix == ".rst"}
    assert listing and {Path(k).name for k in surface_pages("caption")} == listing


def test_a_flagged_phrase_in_a_field_description_fails_naming_the_config_schema_page():
    """Must-fire control: a phrase planted in the rendered schema text is an unlisted hit on the schema page.

    The plant is a `Field` description line as the table renders it. Reverting the schema surface to
    untracked (not rendering it) leaves no hit and turns this red.
    """
    pages = surface_pages("config-schema")
    page = "docs/reference/config-schema.md"
    planted = {page: pages[page] + "\n| `x` | str | Optional | A value which is not yet supported. |\n"}
    problems = check_style(planted, load_allowlist(), surface="config-schema")
    assert any(page in p and "unlisted hit" in p and "rule which" in p for p in problems), problems
    assert any(page in p and "rule cruft" in p for p in problems), problems


def test_a_flagged_phrase_in_a_rendered_docstring_fails_naming_the_api_page():
    """Must-fire control: a phrase planted in a docstring text is an unlisted hit on the API page that renders it."""
    pages = surface_pages("api-docstring")
    page = sorted(pages)[0]
    planted = {**pages, page: pages[page] + "\nThe run which fails.\n"}
    problems = check_style(planted, load_allowlist(), surface="api-docstring")
    assert any(page in p and "unlisted hit" in p and "The run which fails." in p for p in problems), problems


def test_a_planted_cannot_in_a_field_description_fails_naming_the_config_schema_page():
    """Must-fire control for the banned-term rules: `cannot` planted in a `Field` description line of the schema text
    is an unlisted hit under `ban_cannot` on the schema page, and the same text without the word is silent."""
    pages = surface_pages("config-schema")
    page = "docs/reference/config-schema.md"
    planted = {page: pages[page] + "\n| `x` | str | Optional | A value that cannot change. |\n"}
    problems = check_style(planted, load_allowlist(), surface="config-schema")
    assert any(page in p and "unlisted hit" in p and "rule ban_cannot" in p for p in problems), problems
    clean = {page: pages[page] + "\n| `x` | str | Optional | A value that stays fixed. |\n"}
    assert not any("rule ban_cannot" in p for p in check_style(clean, load_allowlist(), surface="config-schema"))


def test_a_planted_cannot_in_an_api_docstring_fails_naming_the_api_page():
    """Must-fire control for the banned-term rules: `cannot` planted in a docstring text is an unlisted hit under
    `ban_cannot` on the API page that renders it."""
    pages = surface_pages("api-docstring")
    page = sorted(pages)[0]
    planted = {**pages, page: pages[page] + "\nThe value cannot change.\n"}
    problems = check_style(planted, load_allowlist(), surface="api-docstring")
    assert any(page in p and "unlisted hit" in p and "rule ban_cannot" in p for p in problems), problems


def test_surface_entries_do_not_affect_the_tracked_page_check():
    """A surface entry is graded only under its surface; the tracked check ignores it."""
    entry = {
        "page": "docs/reference/config-schema.md",
        "rule": "which",
        "line_sha1": line_digest("never rendered"),
        "surface": "config-schema",
        "status": "baseline",
        "reason": "control",
    }
    assert check_style({"p.md": ""}, [entry]) == []
    assert any(p.startswith("stale entry") for p in check_style({"p.md": ""}, [entry], surface="config-schema"))


def test_rules_are_shared_with_the_page_test():
    assert "cruft" in RULES and style_hits({"p.md": "Coming soon.\n"})


def test_a_flagged_phrase_in_the_workflow_description_template_fails_naming_its_path():
    """Must-fire control: a phrase planted in the Overview template text is an unlisted hit on its path."""
    pages = surface_pages("workflow-description")
    planted = {_WORKFLOW_DESCRIPTION: pages[_WORKFLOW_DESCRIPTION] + "\nThe run which fails.\n"}
    problems = check_style(planted, load_allowlist(), surface="workflow-description")
    assert any(_WORKFLOW_DESCRIPTION in p and "unlisted hit" in p and "The run which fails." in p for p in problems), (
        problems
    )


def test_a_surface_that_renders_clean_text_fails_on_its_stale_entries():
    """Must-fire control: a clean page for the config-schema surface leaves an entry matching nothing.

    This pins the stale-entry check that a surface that is no longer rendered relies on. The test that goes red
    when the render is replaced is `test_published_surface_follows_the_style_brief[config-schema]`; the planted-text
    control passes either way, because it plants into whatever the render returns.
    """
    planted = {
        "page": "docs/reference/config-schema.md",
        "rule": "it",
        "line_sha1": line_digest("A planted line that the clean page does not hold."),
        "status": "baseline",
        "reason": "planted by the control",
        "surface": "config-schema",
    }
    problems = check_style({"docs/reference/config-schema.md": "One clean line.\n"}, [planted], surface="config-schema")
    assert any(p.startswith("stale entry") for p in problems), problems


def test_a_flagged_phrase_in_a_css_comment_fails_naming_the_template_path():
    """Must-fire control: a phrase planted in the extracted CSS comment text is an unlisted hit on the template path."""
    pages = surface_pages("report-css")
    planted = {_REPORT_CSS: pages[_REPORT_CSS] + "\nThe run which fails.\n"}
    problems = check_style(planted, load_allowlist(), surface="report-css")
    assert any(_REPORT_CSS in p and "unlisted hit" in p and "The run which fails." in p for p in problems), problems


def test_the_css_comment_extraction_reads_comments_and_not_declarations():
    """The extracted text holds comment prose and no selector, property or placeholder."""
    text = surface_pages("report-css")[_REPORT_CSS]
    assert "UVA brand palette" in text
    assert "background-color:" not in text and "$" not in text
    assert len(_CSS_COMMENT.findall((_REPO / _REPORT_CSS).read_text(encoding="utf-8"))) >= 15
