"""Tests of the renderers' manifest data and of the text changes made around it.

The expectations come from the source as it stood before the change (a fixed commit read with `git show`), so each
test compares a property of the working tree with that earlier text and not with a table written beside the code:
which keys a renderer writes into the plot manifest and with what values, which keys the committed manifest fixtures
hold, that no executable statement moved, and that the public names of the changed modules are the same. No render,
no store and no solver is involved.

Names of the form that carries a unit token are assembled from parts, so that none appears whole in this file.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
#: The commit whose source still writes the unit-token keys and holds the earlier wording.
BEFORE = "9b644fc1"
#: The last commit of the plot-data and history-text sweep: the end of the range the executable-statement test reads.
AFTER = "f0ec9a81"

#: unit-token key -> (unit-free key, unit value of its companion).
KEY_MAP = {
    "_".join(("wse", "m", "range")): ("wse_range", "m"),
    "_".join(("rainfall", "max", "mm", "per", "hr")): ("rainfall_max", "mm h-1"),
    "_".join(("bc", "water", "level", "range", "m")): ("bc_water_level_range", "m"),
    "_".join(("depth", "boundaries", "m")): ("depth_boundaries", "m"),
}
FIELD = "_".join(("depth", "boundaries", "m"))  # the configuration field, which keeps its name
FLOW_KEY = ("_".join(("peak", "flow", "max", "cms")), "peak_flow_max", "m3 s-1")
DEPTH_RENDERER = "src/hhemt/report_renderers/per_sim_peak_flood_depth.py"
FLOW_RENDERER = "src/hhemt/report_renderers/per_sim_conduit_flow.py"
FIXTURES = REPO / "tests" / "fixtures" / "bundles"

#: The whole-word unit tokens of the stored-name convention.
UNIT_TOKEN = re.compile(r"(^|_)(m|mps|min|cms|m3|ltr|hr|percent|perc|pct|m2ps|s|mm|10e6)(_|$)")


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True).stdout


def _have(rev: str) -> bool:
    return (
        subprocess.run(["git", "-C", str(REPO), "cat-file", "-e", f"{rev}^{{commit}}"], capture_output=True).returncode
        == 0
    )


needs_before = pytest.mark.skipif(
    not (_have(BEFORE) and _have(AFTER)), reason="the sweep commits are not in this clone"
)


def _manifest_dicts(source: str) -> list[ast.Dict]:
    """The dict literals passed as `manifest_data=` in `source`, in source order."""
    tree = ast.parse(source)
    found = [n.value for n in ast.walk(tree) if isinstance(n, ast.keyword) and n.arg == "manifest_data"]
    return sorted((d for d in found if isinstance(d, ast.Dict)), key=lambda d: d.lineno)


def _entries(d: ast.Dict) -> dict[str, str]:
    """Key -> dump of the value expression, for the string keys of a dict literal."""
    return {k.value: ast.dump(v) for k, v in zip(d.keys, d.values, strict=True) if isinstance(k, ast.Constant)}


def _expected_entries(before: dict[str, str], mapping: dict[str, tuple[str, str]]) -> dict[str, str]:
    """`before` with each mapped key moved to its unit-free name and a companion constant added."""
    out = {}
    for key, dump in before.items():
        if key in mapping:
            new, unit = mapping[key]
            out[new] = dump
            out[new + "_units"] = ast.dump(ast.Constant(value=unit))
        else:
            out[key] = dump
    return out


@needs_before
@pytest.mark.parametrize(
    ("rel", "mapping"),
    [
        (DEPTH_RENDERER, KEY_MAP),
        (FLOW_RENDERER, {FLOW_KEY[0]: (FLOW_KEY[1], FLOW_KEY[2])}),
    ],
    ids=["peak_flood_depth", "conduit_flow"],
)
def test_each_manifest_dict_holds_the_same_values_under_unit_free_keys_with_a_units_companion(rel, mapping):
    """Property: in every `manifest_data` dict of the two renderers, each unit-token key is replaced by its
    unit-free key carrying the SAME value expression, directly followed by a `{key}_units` companion constant
    with the stated unit; every other entry (key and value expression) is unchanged; no entry is added or lost.

    Class: the one dict of the conduit-flow renderer and the two of the peak-flood-depth renderer (the static and
    the interactive figure), compared entry by entry with the same dicts at the earlier commit, by the AST dump
    of each value.
    Why required: the plot manifests are published and bundled, so a key that changed its value (not only its
    name) or lost its companion misstates the figure's facts; the key `depth_boundaries` must still read the
    configuration field of the same name with its unit suffix.
    Kills it: a value expression edited together with its key; a companion with the wrong unit; a dict of the two
    that was left out; a dropped entry.
    Passes under: any key order, because only the entry sets are compared.
    Must-fire control: the same comparison on source with one value expression altered reports the difference.
    """
    before = [_entries(d) for d in _manifest_dicts(_git("show", f"{BEFORE}:{rel}"))]
    after = [_entries(d) for d in _manifest_dicts((REPO / rel).read_text())]
    assert before and len(before) == len(after), (len(before), len(after))
    for b, a in zip(before, after, strict=True):
        assert any(k in b for k in mapping), "control: the earlier dict holds a unit-token key"
        assert a == _expected_entries(b, mapping)
    text = (REPO / rel).read_text()
    start = text.index("manifest_data={")
    planted = text[:start] + text[start:].replace('"event_iloc": int(event_iloc)', '"event_iloc": 0', 1)
    assert planted != text, "control: the planted edit applies"
    altered = _entries(_manifest_dicts(planted)[0])
    assert altered != after[0]


@needs_before
def test_the_boundaries_still_come_from_the_configuration_field_and_the_field_is_untouched():
    """Property: both peak-flood-depth dicts take `depth_boundaries` from the map configuration's field of that
    name with a unit suffix, and the configuration module that defines that field is byte-identical to the earlier
    commit.

    Class: the two dict literals and `src/hhemt/config/report.py`.
    Why required: the configuration models forbid unknown keys, so a field that changed its name would reject every
    analysis file that sets it; only the manifest key is unit-free.
    Kills it: a renderer reading a field that does not exist; an edit to the configuration module.
    Passes under: any expression that evaluates to the same list.
    """
    for d in _manifest_dicts((REPO / DEPTH_RENDERER).read_text()):
        value = _entries(d)["depth_boundaries"]
        assert FIELD in value and "map_cfg" in value, value
    assert _git("diff", "--stat", BEFORE, "--", "src/hhemt/config/report.py") == ""
    assert FIELD in (REPO / "src/hhemt/config/report.py").read_text()


@needs_before
def test_each_committed_peak_flood_depth_manifest_differs_from_the_earlier_one_only_in_those_four_keys():
    """Property: each of the seven committed `peak_flood_depth.manifest.json` fixtures equals its earlier text with
    the four renderer-data keys moved to their unit-free names and a companion added after each, and with
    nothing else changed (the schema version, the artists, the provenance and the other renderer-data entries are
    equal).

    Class: the seven fixtures under `tests/fixtures/bundles` (three of one experiment, four members of a
    sensitivity master), whole-document equality after applying the key mapping to the earlier document.
    Why required: the bundle tests copy and harvest these files, so a stray edit elsewhere in a manifest changes
    what they test.
    Kills it: a fixture edited by hand in one key only; a changed `bundle_schema_version`; a value that moved.
    Passes under: any byte layout of the same JSON document.
    """
    paths = sorted(FIXTURES.rglob("peak_flood_depth.manifest.json"))
    assert len(paths) == 7, [p.relative_to(REPO).as_posix() for p in paths]
    for path in paths:
        rel = path.relative_to(REPO).as_posix()
        earlier = json.loads(_git("show", f"{BEFORE}:{rel}"))
        expected = json.loads(json.dumps(earlier))
        data = {}
        for key, value in earlier["renderer_data"].items():
            if key in KEY_MAP:
                new, unit = KEY_MAP[key]
                data[new] = value
                data[new + "_units"] = unit
            else:
                data[key] = value
        expected["renderer_data"] = data
        assert json.loads(path.read_text()) == expected, rel
        assert "bundle_schema_version" not in set(earlier["renderer_data"]), "control: it is a document-level key"


# --------------------------------------------------------------------------------------------------------------
# the text changes moved no executable statement and no public name
# --------------------------------------------------------------------------------------------------------------


def _without_docstrings(tree: ast.AST) -> ast.AST:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                node.body = node.body[1:] or [ast.Pass()]
    return tree


def _executable_dump(source: str, *, drop_manifest_dicts: bool) -> str:
    """AST dump of `source` without docstrings; optionally with the `manifest_data=` dicts emptied."""
    tree = _without_docstrings(ast.parse(source))
    if drop_manifest_dicts:
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg == "manifest_data" and isinstance(node.value, ast.Dict):
                node.value = ast.Dict(keys=[], values=[])
    return ast.dump(tree)


def _public_names(source: str) -> list[str]:
    """Every function, class and module-level assigned name of `source`, sorted."""
    tree = ast.parse(source)
    names = [n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names += [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.append(node.target.id)
    return sorted(names)


def _changed_python_sources() -> list[str]:
    changed = _git("diff", "--name-only", "--diff-filter=AM", BEFORE, AFTER, "--", "src").split()
    return [p for p in changed if p.endswith(".py")]


@needs_before
def test_no_changed_source_module_moved_an_executable_statement_or_a_public_name():
    """Property: for every Python module under `src` that differs between the two sweep commits, the syntax tree
    without docstrings is identical (comments never enter a syntax tree), except for the entries of the
    `manifest_data` dicts of the two renderers, whose entries the first test compares; and the sorted list of
    function, class and module-level names is identical.

    Class: every changed `.py` file under `src` (5 when written).
    Why required: a text change must not alter behaviour; comparing syntax trees proves it for all the code at once
    and for every future edit of the same kind.
    Kills it: a statement edited while a comment was; a function given another name with its docstring; an
    identifier changed.
    Passes under: any rewording of a comment or a docstring.
    Must-fire control: a planted executable edit changes the dump.
    """
    changed = _changed_python_sources()
    assert len(changed) >= 3, changed
    for rel in changed:
        drop = rel in {DEPTH_RENDERER, FLOW_RENDERER}
        before_src, after_src = _git("show", f"{BEFORE}:{rel}"), _git("show", f"{AFTER}:{rel}")
        assert _executable_dump(before_src, drop_manifest_dicts=drop) == _executable_dump(
            after_src, drop_manifest_dicts=drop
        ), rel
        if not drop:
            assert _public_names(before_src) == _public_names(after_src), rel
    probe = _git("show", f"{AFTER}:{changed[0]}")
    planted = probe.replace("return ", "return not ", 1) if "return " in probe else probe + "\nx = 1\n"
    assert _executable_dump(planted, drop_manifest_dicts=False) != _executable_dump(probe, drop_manifest_dicts=False)


# --------------------------------------------------------------------------------------------------------------
# a units companion always has its key
# --------------------------------------------------------------------------------------------------------------


def _companion_problems(source: str) -> list[str]:
    problems = []
    for d in _manifest_dicts(source):
        keys = [k.value for k in d.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)]
        problems += [f"{k} has no key {k[:-6]}" for k in keys if k.endswith("_units") and k[:-6] not in keys]
        problems += [f"{k} carries a unit token" for k in keys if UNIT_TOKEN.search(k) and not k.endswith("_units")]
    return problems


def test_every_manifest_dict_in_src_pairs_each_companion_with_its_key_and_carries_no_unit_token_key():
    """Property: in every `manifest_data=` dict literal under `src/hhemt` (not the versioning system), each key
    ending `_units` has the key it describes in the same dict, and no other key carries a unit token as a whole
    token of its name.

    Class: every such dict in every module under `src/hhemt` (10 dicts in 10 renderer modules when written).
    Why required: a companion without its key describes nothing, and a unit token left in a key is the form the
    convention removes; scanning every module and not two names catches a renderer added later.
    Kills it: a companion left behind when its key is removed; a new renderer that names its unit in a key.
    Passes under: any dict whose keys follow the convention.
    Must-fire control: planted source with an orphan companion and a unit-token key is reported both ways.
    """
    scanned = 0
    problems = {}
    for path in sorted((REPO / "src" / "hhemt").rglob("*.py")):
        if "version_migration" in path.parts:
            continue
        found = _companion_problems(path.read_text())
        scanned += len(_manifest_dicts(path.read_text()))
        if found:
            problems[path.relative_to(REPO).as_posix()] = found
    assert scanned >= 8, scanned
    assert problems == {}
    token_key = "_".join(("flow", "cms"))
    planted = f"emit(manifest_data={{'depth_max_units': 'm', '{token_key}': 1.0, 'peak': 2.0, 'peak_units': 'm'}})\n"
    assert sorted(_companion_problems(planted)) == [
        "depth_max_units has no key depth_max",
        f"{token_key} carries a unit token",
    ]


def _grep_files(rev: str | None, pattern: str, *paths: str) -> list[str]:
    """Files holding a whole-word match of `pattern`: in the working tree and untracked files, or at `rev`."""
    args = (
        ["grep", "-l", "-w", "-E"]
        + ([] if rev else ["--untracked"])
        + [pattern]
        + ([rev] if rev else [])
        + ["--", *paths]
    )
    out = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True).stdout.split()
    return sorted(p.split(":", 1)[-1] if rev else p for p in out)


@needs_before
def test_no_unit_token_key_remains_anywhere_in_src_tests_or_docs_and_the_field_name_stays_in_two_files():
    """Property: none of the five unit-token renderer-data keys occurs as a whole word in any file under `src`,
    `tests` or `docs` (the versioning system and the bundle reference pages excepted), and the configuration
    field name with the unit suffix occurs only in the configuration module and the peak-flood-depth renderer.

    Class: a whole-word search of tracked and untracked files over three roots, for the five keys and for the one
    field name.
    Why required: a key left in a test, a document or a second renderer is a name that the manifest no longer
    writes; the census of the renderer directory alone does not look outside it.
    Kills it: a stale key in a docstring, a fixture reader or a page of the documentation; the field name copied
    into a third module.
    Passes under: any text that avoids the five spellings.
    Must-fire control: the same search at the earlier commit returns files, so the pattern and the roots reach
    the places where the keys were.
    """
    keys = "|".join([k for k in (*KEY_MAP, FLOW_KEY[0]) if k != FIELD])
    exclusions = (":!docs/reference/bundles", ":!src/hhemt/version_migration")
    assert _grep_files(None, keys, "src", "tests", "docs", *exclusions) == []
    assert _grep_files(BEFORE, keys, "src", "tests", "docs", *exclusions), "control: the earlier commit holds the keys"
    assert _grep_files(None, "depth_boundaries", "src", "tests", "docs"), "control: the working-tree search finds files"
    assert _grep_files(None, FIELD, "src", "tests", "docs", *exclusions) == [
        "src/hhemt/config/report.py",
        DEPTH_RENDERER,
    ]


# --------------------------------------------------------------------------------------------------------------
# no bare citation of a decision, a design section or a check item in added text
# --------------------------------------------------------------------------------------------------------------

#: Each fragment is joined here so that no line of this file holds a citation the pattern reports.
_CITATION = re.compile(
    r"\b(Definition"
    r"-of-"
    r"Done|Do"
    r"D)\b"
    r"|\bD[0-9]{1,2}\b"
    r"|\bsha"
    r"pe's\b"
    r"|\bsha"
    r"pe (unit|units|inventory|table)s?\b"
    r"|\bsection [0-9]+(\.[0-9]+)?\b"
    r"|\bact [0-9]+\b"
    r"|\biteration [0-9]+\b",
    re.IGNORECASE,
)
#: Stable references to documents that outlive any change: an external standard and the golden specification.
_STABLE = ("CF-1.13 section", "SPEC.md section")
_ROOTS = ("src", "docs", "scripts", "tests", "README*")
_EXCLUDED = (
    ":!docs/reference/bundles",
    ":!src/hhemt/version_migration",
    ":!scripts/vocabulary_freeze.yaml",
    ":!scripts/vocabulary_residue_exceptions.yaml",
    ":!tests/fixtures",
    ":!tests/goldens/SPEC.md",
    ":!tests/test_version_migration_v0028.py",
    ":!tests/test_review_v0028_migration.py",
)


def _citations(diff_text: str) -> list[tuple[str, str]]:
    """(path, text) of each added line that cites a decision, a design section or a check item."""
    path = ""
    out = []
    for line in diff_text.splitlines():
        if line.startswith("+++ "):
            path = line[6:] if line.startswith("+++ b/") else ""
        elif line.startswith("+") and not line.startswith("+++"):
            text = line[1:]
            if _CITATION.search(text) and not any(s in text for s in _STABLE):
                out.append((path, text.strip()))
    return out


_HAS_BASE = (
    subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "--verify", "-q", "origin/develop"], capture_output=True
    ).returncode
    == 0
)


@pytest.mark.skipif(not _HAS_BASE, reason="origin/develop is not a ref of this clone")
def test_no_added_line_cites_a_decision_a_design_section_or_a_check_item_except_stable_references():
    """Property: over the lines the working tree adds relative to `origin/develop`, no comment, docstring or
    message cites a decision by number, a design section, an act, an iteration, a check item, or "the shape" of a
    design document; the only citations left are of an external standard (CF-1.13) and of the golden
    specification (SPEC.md), which are documents a reader can open.

    Class: the added lines of `git diff -U0 origin/develop` under src, docs, scripts, tests and README*, with the
    versioning system, the vocabulary manifests and the tracked fixtures excluded, minus the two stable references.
    Why required: a citation of a document a reader of the toolkit does not have says nothing and ages; the same
    rule that bans a chunk number bans a decision number and a design-section number.
    Kills it: a comment that cites a check item by its number; a docstring that attributes a unit to a design
    document.
    Passes under: any rewording that states the property in words.
    Must-fire control: planted added lines for each citation form are reported and the two stable references are
    not.
    """
    forms = (
        "# the " + "Definition" + "-of-" + "Done item 3 census",
        "# see " + "D" + "14 for the choice",
        "    " + "sha" + "pe's unit and nothing else",
        "# the " + "sha" + "pe unit table",
        "# per " + "sec" + "tion 3.4",
        "# " + "ac" + "t 5 of the plan",
    )
    stable = ("# CF-1.13 section 7.3 admits no name", "# SPEC.md section 8 states the unit")
    present = ("# the depth above the bed", "# a unit-free name", "# the shape of the array is (2, 3)")
    diff = "+++ b/src/hhemt/x.py\n" + "".join(f"+{line}\n" for line in (*forms, *stable, *present))
    assert [t for _, t in _citations(diff)] == [f.strip() for f in forms]
    real = subprocess.run(
        ["git", "-C", str(REPO), "diff", "-U0", "origin/develop", "--", *_ROOTS, *_EXCLUDED],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    found = _citations(real)
    assert found == [], "\n".join(f"{p}: {t[:110]}" for p, t in found)
