"""The published surface carries the coupling-diagnostics and manhole-table config fields and the manhole guide.

The configuration schema page is generated from the config models at documentation build time by
``hooks/config_reference.py``, so a field reaches the page by existing on its model. These tests render the page
the way the build does and pin the added fields BY NAME, so a renamed or dropped field fails here instead of
leaving a guide that names a field the schema no longer lists. The guide must exist, sit in the navigation, and
be listed in the how-to index.

Each presence assertion carries its own must-fire control: a name that no model declares finds no row.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
_HOOK = _REPO / "hooks" / "config_reference.py"
_GUIDE = "how-to/per-node-manhole-values.md"

#: The analysis-config fields the coupling-diagnostics and manhole-table work added.
ANALYSIS_FIELDS = (
    "toggle_coupling_diagnostics",
    "swmm_diagnostic_links",
    "swmm_diagnostic_node_path3_per_step",
    "swmm_coupling_interval",
    "swmm_coupling_lag",
    "swmm_report_step_s",
    "manhole_table",
    "manhole_table_fill_with_constants",
)
#: The system-config field the same work added.
SYSTEM_FIELDS = ("TRITONSWMM_extra_c_flags",)


def _load_hook():
    """Load the schema-page generator from its file and bind this checkout's `src` ahead of any `hhemt` import."""
    spec = importlib.util.spec_from_file_location("config_reference_manhole_docs", _HOOK)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["config_reference_manhole_docs"] = mod
    spec.loader.exec_module(mod)
    mod._load_local_src().bind_local_src(mod._SRC, {"hhemt"})
    return mod


@pytest.fixture(scope="module")
def schema_page() -> str:
    """The configuration schema page as the documentation build renders it."""
    return _load_hook()._render()


def _has_row(page: str, field: str) -> bool:
    """True when the page holds a table row for `field`: a line that starts with the field name in code spans."""
    return re.search(rf"^\| `{re.escape(field)}` \|", page, flags=re.MULTILINE) is not None


@pytest.mark.parametrize("field", ANALYSIS_FIELDS + SYSTEM_FIELDS)
def test_every_added_config_field_has_a_reference_row(schema_page, field):
    """Property: the rendered schema page holds a table row for each added field.

    Class: the 8 analysis fields and 1 system field of `ANALYSIS_FIELDS` and `SYSTEM_FIELDS`. Why required: the
    guide and the coupling-diagnostics settings are found through the schema page, so a field that drops off the
    models drops off the page. What kills it: renaming or deleting a field on its model. A second correct
    implementation: a hand-maintained page that lists the same rows still passes.
    """
    assert _has_row(schema_page, field), f"the schema page has no row for {field}"


def test_a_name_no_model_declares_has_no_reference_row(schema_page):
    """Must-fire control: the matcher finds no row for a name that is on no model."""
    assert not _has_row(schema_page, "manhole_table_not_a_field")


def test_added_fields_are_declared_on_the_models():
    _load_hook()  # binds this checkout's src before the models are imported
    from hhemt.config.analysis import analysis_config
    from hhemt.config.system import system_config

    analysis = set(analysis_config.model_fields)
    system = set(system_config.model_fields)
    assert [f for f in ANALYSIS_FIELDS if f not in analysis] == []
    assert [f for f in SYSTEM_FIELDS if f not in system] == []
    assert "manhole_table_not_a_field" not in analysis | system


def test_the_manhole_guide_exists_and_names_both_manhole_fields():
    """Property: the guide file exists and carries both field names and the required CSV header.

    Class: the one guide page. Why required: a reader who sets the table needs the exact header or the run is
    refused. What kills it: deleting the page, or editing the header text away from the one the reader requires.
    A second correct implementation: the same facts under other headings still pass.
    """
    text = (_REPO / "docs" / _GUIDE).read_text(encoding="utf-8")
    assert "manhole_table" in text and "manhole_table_fill_with_constants" in text
    assert "node_id,manhole_diameter_m,manhole_loss_coefficient" in text


def test_the_manhole_guide_is_in_the_navigation():
    """Property: `mkdocs.yml` carries a navigation entry for the guide, and a misspelt path has none.

    Class: the navigation lines of the how-to section. Why required: a page outside the navigation is a page no
    reader finds. What kills it: removing the entry. A second correct implementation: the entry under another title.
    """
    nav = (_REPO / "mkdocs.yml").read_text(encoding="utf-8")
    assert re.search(rf"^\s*- [^\n]+: {re.escape(_GUIDE)}\s*$", nav, flags=re.MULTILINE)
    assert not re.search(r"^\s*- [^\n]+: how-to/per-node-manhole-values-missing\.md\s*$", nav, flags=re.MULTILINE)


def test_the_manhole_guide_is_in_the_how_to_index():
    """Property: the how-to index links the guide, and links no misspelt path.

    Class: the rows of the how-to index. Why required: the index is the list a reader scans for a guide. What
    kills it: removing the row. A second correct implementation: the row in another position.
    """
    index = (_REPO / "docs" / "how-to" / "index.md").read_text(encoding="utf-8")
    assert "(per-node-manhole-values.md)" in index
    assert "(per-node-manhole-values-missing.md)" not in index
