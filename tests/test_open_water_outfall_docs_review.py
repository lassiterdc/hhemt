"""Reviewer tests for the open-water outfall guide (package 1l, chunk 1l.1).

The problem these tests pin. The guide states its facts in three places that a substring search of the whole page
cannot tell apart: the cells of the Norfolk sizing table, the config names in its prose, and the YAML example. The
chunk's own tests search the whole page for a figure, a file name or a field name, so a figure moved to the wrong
cell, a renamed config field named in prose, or a YAML value that the field no longer accepts all pass them. Each
test below reads one of those three places by position and compares it with what the code gives.

Every test reads a file or calls a pure function. None compiles or runs a solver.

Tests promoted from the chunk's own test file (their original description follows).

The published surface carries the open-water outfall fields and the how-to page that describes them.

The configuration schema page is generated from the config models at documentation build time by
``hooks/config_reference.py`` (the page is gitignored and has no file to edit), so a field reaches the page by
existing on its model. These tests render the page the way the build does and pin the two fields BY NAME, so a
renamed or dropped field fails here instead of leaving a guide that names a field the schema no longer lists.
The guide must exist, name both fields and both representations, sit in the navigation, be listed in the how-to
index, and state only facts the code holds: the file names and cfg keys it names are the ones the toolkit writes,
and the Norfolk sizing figures are the ones the sizing rule gives.

Each presence assertion carries its own must-fire control: a name that no model, page or navigation holds finds no
row, no mention and no entry.
"""

from __future__ import annotations

import importlib.util
import inspect
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from pydantic import TypeAdapter, ValidationError

from hhemt import scenario
from hhemt.config.analysis import analysis_config
from hhemt.config.system import system_config
from hhemt.swmm_runnable import _parse
from hhemt.swmm_utils import _outlet_conduit_sizing
from hhemt.water_level_boundary import EXTBC_TYPE_CODE

_GUIDE = Path(__file__).resolve().parent.parent / "docs" / "how-to" / "open-water-outfalls.md"


#: The Norfolk outlet conduits as the shape's measurement gives them, with the loss that the sizing carries through.
_MODEL = (
    "[CONDUITS]\n"
    "1444  E143254  E143250  14.789  0.013  0  0\n"
    "1445  E147008  E147007  50.596  0.013  0  0\n"
    "[XSECTIONS]\n"
    "1444  RECT_CLOSED  1.524  1.676  0  0  2\n"
    "1445  RECT_CLOSED  1.676  1.676  0  0  2\n"
)


#: Names of the page that are neither a field nor a file: the values the representation and the coupling mode take
#: and the TRITON cfg keys of the guide's table. The cfg keys are read from the toolkit, not typed here.
_VALUES = {"driven_stage", "crown_rim", "all_junctions", "legacy", "list", "processed", "full"}


_CFG_KEYS = {scenario.SWMM_DRIVEN_OUTFALLS_KEY, scenario.SWMM_DEM_RIM_EXEMPT_KEY, scenario.SWMM_MANHOLE_TABLE_KEY}


def _guide() -> str:
    return _GUIDE.read_text(encoding="utf-8")


def _norfolk_row_problems(page: str) -> list[str]:
    """The cells of the Norfolk table of `page` that differ from the sizing, one string per differing cell.

    A row is the table line that opens with the outfall name in code spans. Its cells are read by position (conduit,
    cross-section, area, diameter, crown), so a figure that is right but sits in another row or another column is
    reported, which a search of the whole page does not do.
    """
    sections = _parse(_MODEL)
    problems: list[str] = []
    for name in ("E143250", "E147007"):
        sized = _outlet_conduit_sizing(sections, name, 1.0, 0.1)
        assert not isinstance(sized, str), sized
        found = re.search(rf"^\| `{name}` \|(.*)\|\s*$", page, flags=re.MULTILINE)
        if found is None:
            problems.append(f"{name}: no table row")
            continue
        cells = [c.strip() for c in found.group(1).split("|")]
        if len(cells) != 5:
            problems.append(f"{name}: {len(cells)} cells after the name, expected 5")
            continue
        conduit, _cross_section, area, diameter, crown = cells
        for label, cell, expected in (
            ("conduit", conduit, f"`{sized.conduit}`"),
            ("area", area, f"{sized.area_m2:.3f} m²"),
            ("diameter", diameter, f"{sized.diameter_m:.3f} m"),
            ("crown", crown, f"{sized.crown_m:.3f} m"),
        ):
            if cell != expected:
                problems.append(f"{name} {label}: page {cell!r}, sizing {expected!r}")
    return problems


def test_each_norfolk_row_carries_the_sizing_of_its_own_outfall():
    """Property: in the Norfolk table, each outfall's conduit, area, diameter and crown cell equals its sizing.

    Class: the 2 rows of the table times its 4 computed cells. Why required: the table is the worked example of the
    rule, and a reader checks the diameter of one outfall against its own conduit; the chunk's test finds each
    figure anywhere on the page, so a diameter under the wrong outfall, or a crown that differs from 1.524 while
    1.524 stays in the cross-section cell, passes it. What kills it: swapping two cells of a row or editing the
    crown cell. A second correct implementation: the same cells under other column headings still pass, because
    the cells are read by position.
    """
    assert _norfolk_row_problems(_guide()) == []


def test_the_row_matcher_reports_a_swapped_diameter_and_a_wrong_crown():
    """Must-fire control: the same reader reports a page with the diameters swapped and the crown edited.

    The page text is the guide with two edits that a whole-page substring search of the figures does not see:
    the two diameters exchanged, and the crown of the first row set to 1.600 m. The second edit leaves `1.524`
    on the page, in the cross-section cell.
    """
    page = _guide()
    swapped = page.replace("2.550 m", "@@").replace("2.675 m", "2.550 m").replace("@@", "2.675 m")
    wrong_crown = swapped.replace("| 1.524 m |", "| 1.600 m |")
    problems = _norfolk_row_problems(wrong_crown)
    assert any("E143250 diameter" in p for p in problems) and any("E147007 diameter" in p for p in problems)
    assert any("E143250 crown" in p for p in problems)
    assert "1.524" in wrong_crown, "the edit left the figure the whole-page search looks for"


def _bare_names(page: str) -> set[str]:
    """Each code span of `page` that is an identifier with an underscore, or the value of a `key: value` span."""
    names: set[str] = set()
    for span in re.findall(r"`([^`\n]+)`", page):
        key = span.split(":", 1)[0].strip()
        value = span.split(":", 1)[1].strip() if ":" in span else None
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9]*(_[A-Za-z0-9]+)+", key) and not key.isupper():
            names.add(key)  # an all-capital token is a SWMM keyword such as RECT_CLOSED, not a config name
        if value is not None and re.fullmatch(r"[a-z][a-z_]*", value):
            names.add(value)
    return names


def test_every_config_name_of_the_guide_is_a_field_a_known_value_or_a_cfg_key():
    """Property: each underscore name the guide puts in a code span resolves in the toolkit.

    Class: the 21 names of the page (13 config fields of the two models, 3 TRITON cfg keys, 4 values of the
    representation, coupling-mode and DEM fields, and the record type `water_level`). Why required: the chunk's
    test pins 2 of the 13 field names; the other 11 (such as `dem_rim_resolution` and `swmm_coupling_nodes_file`)
    go stale on a rename and nothing reds. What kills it: renaming `dem_rim_resolution` or any named field. A
    second correct implementation: a page that names other existing fields still passes.
    """
    fields = set(analysis_config.model_fields) | set(system_config.model_fields)
    known = fields | _VALUES | _CFG_KEYS | set(EXTBC_TYPE_CODE)
    unresolved = sorted(n for n in _bare_names(_guide()) if n not in known and not re.search(r"\.(txt|json|inp|h)$", n))
    assert unresolved == [], unresolved
    assert {"dem_rim_resolution", "swmm_coupling_nodes_file"} <= _bare_names(_guide())


def test_a_name_that_no_model_declares_is_reported_by_the_name_matcher():
    """Must-fire control: a code span naming a field that no model has is an unresolved name."""
    fields = set(analysis_config.model_fields) | set(system_config.model_fields)
    spans = _bare_names("A listed outfall needs `dem_rim_resolutions` here, and `swmm_open_water_outfalls` there.")
    assert [n for n in sorted(spans) if n not in fields] == ["dem_rim_resolutions"]


def _yaml_blocks(page: str) -> list[dict]:
    return [yaml.safe_load(b) for b in re.findall(r"```yaml\n(.*?)```", page, flags=re.DOTALL)]


def _rejected_yaml_keys(block: dict) -> list[str]:
    """The keys of one YAML mapping whose value the analysis config's own field annotation refuses."""
    rejected = []
    for key, value in block.items():
        field = analysis_config.model_fields.get(key)
        if field is None:
            rejected.append(f"{key}: not an analysis field")
            continue
        try:
            TypeAdapter(field.annotation).validate_python(value)
        except ValidationError:
            rejected.append(f"{key}: value {value!r} refused")
    return rejected


def test_the_yaml_example_of_the_guide_is_accepted_by_the_field_annotations():
    """Property: every key and value of each fenced YAML block is an analysis field and a value it accepts.

    Class: the YAML blocks of the guide (1 block of 2 keys). Why required: the chunk's test reads the default of
    the representation and nothing of the example, so a rename of a value in the `Literal` leaves a copy-and-paste
    example that the config refuses. What kills it: dropping `driven_stage` from the representation's `Literal`
    or renaming the list field. A second correct implementation: another example with other valid values passes.
    """
    blocks = _yaml_blocks(_guide())
    assert len(blocks) >= 1
    assert [_rejected_yaml_keys(b) for b in blocks] == [[] for _ in blocks]


def test_the_yaml_matcher_refuses_a_value_the_literal_does_not_hold():
    """Must-fire control: the same check refuses an unknown representation and an unknown key."""
    refused = _rejected_yaml_keys({"open_water_outfall_representation": "crown_circle", "no_such_field": 1})
    assert [r.split(":")[0] for r in refused] == ["open_water_outfall_representation", "no_such_field"]


@pytest.mark.parametrize("name", sorted(_CFG_KEYS))
def test_each_cfg_key_of_the_build_table_is_a_row_of_that_table(name):
    """Property: a TRITON cfg key of the guide's step 3 table appears in that table's rows.

    Class: the 3 cfg keys of the toolkit. Why required: step 3 tells the reader which key each representation needs
    a build to read, and the key must be in a row of that table, not only in prose. What kills it: removing a key
    from the table. A second correct implementation: the key in another row of the same table.
    """
    page = _guide()
    table = re.search(r"\| Representation \| Key the build must read \|\n\|[-|]+\|\n((?:\|.*\n)+)", page)
    assert table is not None
    assert f"`{name}`" in table.group(1)


_REPO = Path(__file__).resolve().parent.parent


_HOOK = _REPO / "hooks" / "config_reference.py"


_GUIDE_promoted = "how-to/open-water-outfalls.md"


_GUIDE_PATH = _REPO / "docs" / _GUIDE_promoted


FIELDS = ("swmm_open_water_outfalls", "open_water_outfall_representation")


REPRESENTATIONS = ("driven_stage", "crown_rim")


def _load_hook():
    """Load the schema-page generator from its file and bind this checkout's `src` ahead of any `hhemt` import."""
    spec = importlib.util.spec_from_file_location("config_reference_open_water_docs", _HOOK)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["config_reference_open_water_docs"] = mod
    spec.loader.exec_module(mod)
    mod._load_local_src().bind_local_src(mod._SRC, {"hhemt"})
    return mod


@pytest.fixture(scope="module")
def schema_page() -> str:
    """The configuration schema page as the documentation build renders it."""
    return _load_hook()._render()


@pytest.fixture(scope="module")
def guide() -> str:
    return _GUIDE_PATH.read_text(encoding="utf-8")


def _row(page: str, field: str) -> str | None:
    """The table row of `field` on the schema page: the line that starts with the field name in code spans."""
    found = re.search(rf"^\| `{re.escape(field)}` \|.*$", page, flags=re.MULTILINE)
    return found.group(0) if found else None


@pytest.mark.parametrize("field", FIELDS)
def test_each_field_has_a_schema_reference_row(schema_page, field):
    """Property: the rendered schema page holds a table row for each of the two fields.

    Class: the 2 analysis fields of `FIELDS`. Why required: the guide sends the reader to the schema page for the
    field, so a field that drops off the model drops off the page. What kills it: renaming or deleting a field on
    its model. A second correct implementation: a hand-maintained page that lists the same rows still passes.
    """
    assert _row(schema_page, field) is not None, f"the schema page has no row for {field}"


def test_a_name_no_model_declares_has_no_schema_row(schema_page):
    """Must-fire control: the matcher finds no row for a name that is on no model."""
    assert _row(schema_page, "swmm_open_water_outfalls_not_a_field") is None


def test_the_representation_row_lists_both_options(schema_page):
    """Property: the row of the representation field carries both option names in its glossary.

    Class: the one row. Why required: the reader chooses between the two options from this row. What kills it:
    dropping an option from the field's `Literal` or from its glossary. A second correct implementation: the
    options in another column still pass, because the whole row is searched.
    """
    row = _row(schema_page, "open_water_outfall_representation")
    assert row is not None
    assert all(f"`{name}`" in row for name in REPRESENTATIONS)
    assert "`not_a_representation`" not in row


def test_fields_are_declared_on_the_analysis_model():
    _load_hook()  # binds this checkout's src before the models are imported
    from hhemt.config.analysis import analysis_config

    declared = set(analysis_config.model_fields)
    assert [f for f in FIELDS if f not in declared] == []
    assert "swmm_open_water_outfalls_not_a_field" not in declared
    default = analysis_config.model_fields["open_water_outfall_representation"].default
    assert default == "driven_stage"


def test_the_guide_exists_and_names_both_fields_and_both_representations(guide):
    """Property: the guide carries both field names and both representation names.

    Class: the one guide page. Why required: a reader who lists an outfall needs the field names and the two
    values the second field takes. What kills it: deleting the page or renaming a field or a value in it only.
    A second correct implementation: the same names under other headings still pass.
    """
    for name in (*FIELDS, *REPRESENTATIONS):
        assert f"`{name}`" in guide, f"the guide does not name {name}"
    assert "`not_a_representation`" not in guide


def test_the_guide_is_in_the_navigation():
    """Property: `mkdocs.yml` carries a navigation entry for the guide, and a misspelt path has none.

    Class: the navigation lines of the how-to section. Why required: a page outside the navigation is a page no
    reader finds. What kills it: removing the entry. A second correct implementation: the entry under another title.
    """
    nav = (_REPO / "mkdocs.yml").read_text(encoding="utf-8")
    assert re.search(rf"^\s*- [^\n]+: {re.escape(_GUIDE_promoted)}\s*$", nav, flags=re.MULTILINE)
    assert not re.search(r"^\s*- [^\n]+: how-to/open-water-outfalls-missing\.md\s*$", nav, flags=re.MULTILINE)


def test_the_guide_is_in_the_how_to_index():
    """Property: the how-to index links the guide, and links no misspelt path.

    Class: the rows of the how-to index. Why required: the index is the list a reader scans for a guide. What
    kills it: removing the row. A second correct implementation: the row in another position.
    """
    index = (_REPO / "docs" / "how-to" / "index.md").read_text(encoding="utf-8")
    assert "(open-water-outfalls.md)" in index
    assert "(open-water-outfalls-missing.md)" not in index


def test_the_file_names_and_cfg_keys_the_guide_names_are_the_ones_the_toolkit_writes(guide):
    """Property: every file name and cfg key in the guide's table of written files is one the code writes.

    Class: the 4 file names and the 3 cfg keys that the guide's table names. Why required: a reader looks for these
    files in a scenario folder and in the TRITON cfg, so a renamed file or key makes the page point at nothing.
    What kills it: renaming a file helper, a key constant or the record file in the toolkit without the page.
    A second correct implementation: another helper that returns the same names still passes. The control asks
    the same helpers for a name that the page does not carry.
    """
    from hhemt import scenario, scenario_inputs

    hydraulics = Path("scenario") / "hydraulics.inp"
    written = {
        scenario.driven_outfalls_file_for(hydraulics).name,
        scenario.dem_rim_exempt_nodes_file_for(hydraulics).name,
        scenario.manhole_table_file_for(hydraulics).name,
    }
    assert written == {"driven_outfalls.txt", "dem_rim_exempt_nodes.txt", "manhole_table.txt"}
    # the record file is written inline by the scenario-input method, so its name is read from that source
    assert '"crown_rim_outfalls.json"' in inspect.getsource(scenario_inputs)
    keys = {scenario.SWMM_DRIVEN_OUTFALLS_KEY, scenario.SWMM_DEM_RIM_EXEMPT_KEY, scenario.SWMM_MANHOLE_TABLE_KEY}
    assert keys == {"swmm_driven_outfalls", "swmm_dem_rim_exempt_nodes", "swmm_manhole_table"}
    for name in (*written, "crown_rim_outfalls.json", *keys):
        assert f"`{name}`" in guide, f"the guide does not name {name}"
    assert "`driven_outfalls_missing.txt`" not in guide


def test_the_norfolk_figures_of_the_guide_are_the_ones_the_sizing_rule_gives(guide):
    """Property: the area, diameter and crown figures in the guide's Norfolk table are what `crown_rim` sizing returns.

    Class: the 2 Norfolk outlet conduits (twin `RECT_CLOSED` barrels, zero offsets). Why required: the table is the
    worked example of the diameter rule, and a figure that the rule does not give misleads a reader who checks it.
    What kills it: changing the rule (area-equivalent diameter, area over every barrel, crown from the conduit
    height) or a figure on the page. A second correct implementation: another sizing function with the same rule
    still passes. The control sizes a conduit with 1 barrel and finds a figure the page does not carry.
    """
    from hhemt.swmm_runnable import _parse
    from hhemt.swmm_utils import _outlet_conduit_sizing

    model = (
        "[OUTFALLS]\n"
        "E143250  -1.079  FREE  NO\n"
        "E147007  -1.079  FREE  NO\n"
        "E143251  -1.079  FREE  NO\n"
        "[CONDUITS]\n"
        "1444  E143254  E143250  14.789  0.013  0  0\n"
        "1445  E147008  E147007  50.596  0.013  0  0\n"
        "1446  E143252  E143251  10.0  0.013  0  0\n"
        "[XSECTIONS]\n"
        "1444  RECT_CLOSED  1.524  1.676  0  0  2\n"
        "1445  RECT_CLOSED  1.676  1.676  0  0  2\n"
        "1446  RECT_CLOSED  1.524  1.676  0  0  1\n"
    )
    sections = _parse(model)
    sized = {name: _outlet_conduit_sizing(sections, name, 1.0, 0.1) for name in ("E143250", "E147007", "E143251")}
    assert all(not isinstance(s, str) for s in sized.values()), sized
    first, second, one_barrel = sized["E143250"], sized["E147007"], sized["E143251"]
    for s, area, diameter, crown in ((first, "5.108", "2.550", "1.524"), (second, "5.618", "2.675", "1.676")):
        assert f"{s.area_m2:.3f}" == area and f"{s.diameter_m:.3f}" == diameter and f"{s.crown_m:.3f}" == crown
        assert math.isclose(s.diameter_m, math.sqrt(4 * s.area_m2 / math.pi))
        assert area in guide and diameter in guide and crown in guide
    assert first.conduit == "1444" and second.conduit == "1445" and first.barrels == second.barrels == 2
    assert f"{one_barrel.area_m2:.3f}" not in guide  # control: a figure of a conduit the page does not describe


def test_the_guide_passes_the_docs_content_lint(guide):
    """Property: the guide holds no placeholder, bare line citation, banned punctuation or banned word.

    Class: the lint classes of `scripts/check_docs_content.py` that the schema-page generator also applies to its
    own prose. What kills it: a self-declared placeholder, a `file.py:123` citation, or a banned word in the page.
    The control feeds the same gate a line with a placeholder declaration and expects it to raise.
    """
    hook = _load_hook()
    hook._assert_lint_clean(guide)
    with pytest.raises(RuntimeError):
        hook._assert_lint_clean("This section is a placeholder and is authored in a later task.\n")


def _vale() -> str | None:
    return shutil.which("vale")


@pytest.mark.skipif(_vale() is None, reason="vale is not on PATH; the repository's Vale gate is scripts/check_vale.py")
def test_the_guide_passes_the_vale_prose_gate(tmp_path):
    """Property: Vale reports no alert on the guide under the repository's `.vale.ini`.

    Class: the one new page. Why required: `scripts/check_vale.py` fails the docs gate on any alert. What kills it:
    a banned phrase, an unfinished-sentence pattern or a heading in the wrong case. The control runs Vale on a page
    that the style bans and expects an alert, so a Vale that finds nothing anywhere is not read as a pass.
    """
    vale = _vale()
    assert vale is not None
    clean = subprocess.run(
        [vale, "--config=.vale.ini", "--output=line", f"docs/{_GUIDE_promoted}"],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert clean.stdout.strip() == "", clean.stdout
    bad = tmp_path / "bad.md"
    bad.write_text("# Bad page\n\nIt cannot be used here.\n", encoding="utf-8")
    banned = subprocess.run(
        [vale, "--config=.vale.ini", "--output=line", str(bad)], cwd=_REPO, capture_output=True, text=True, check=False
    )
    assert "BanCannot" in banned.stdout, banned.stdout
