"""Style rules for reader-facing report text, checked on the sources that hold that text.

A rendered page needs analysis data, and the sources hold every string a page can
print, so this test reads sources and never renders a page. The rules follow the
report style brief (rules 2, 3, 4, 7 and 8); the retired words are the ones the term
table lists that a matcher can tell apart. Retired words whose meaning depends on use
(for example "simulation" as a count noun) are checked by the reviewer, not here.

Population (computed by glob at run time):
  * every ``*.rst`` under ``report_templates/captions/`` and every ``*.j2`` under
    ``report_templates/`` (code spans, fenced blocks and Jinja expressions masked; for
    ``*.j2`` also markup, script and style bodies are masked, and ``*.css.j2`` is
    scanned only for its quoted strings);
  * every string constant in ``src/hhemt/report_renderers/*.py``, ``figure_caption.py``,
    ``figure_panels.py``, ``analysis_validation.py`` and ``eda/*.py``, and the
    ``_COMBINED_DESCRIPTION_PARTS`` constant of ``bundle/combined_snakefile_generator.py``.
    Not read: docstrings, arguments of logging calls, the expression of a ``raise``, and
    constants bound to module-level names ending ``_JS``, ``_CSS``, ``_OLD``, ``_NEW``,
    ``_DEFS``, ``_RE`` or ``_CDN``. An f-string is read as its literal parts joined, each
    placeholder replaced by ``{}``.

Prose rules apply to strings of three or more words; the retired-word rule applies to
every string. A hit is a rule plus a snippet (the whole string when it is short, else a
window around the match). Each source has one allowlist file,
``tests/report_style/allowlist/{name}.yaml``, where ``name`` is the source basename,
prefixed with the parent directory when two sources share a basename. An entry holds
``rule``, ``string``, ``status`` (``baseline`` or ``permitted``), ``reason`` and an
optional ``count``. The test fails on a hit with no entry and on an entry that matches
no hit. A source with no allowlist file is checked against an empty allowlist.
"""

from __future__ import annotations

import ast
import re
from collections import Counter
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
PKG = REPO_ROOT / "src" / "hhemt"
ALLOWLIST_DIR = Path(__file__).resolve().parent / "report_style" / "allowlist"

# Sources a package is still rewriting are neither scanned nor required to have an allowlist
# file. The set is empty: every population source is scanned against its allowlist file.
TRANSITIONAL: frozenset[str] = frozenset()

_PY_EXTRA = (
    "figure_caption.py",
    "figure_panels.py",
    "analysis_validation.py",
)
_COMBINED_SRC = "src/hhemt/bundle/combined_snakefile_generator.py"
_COMBINED_CONST = "_COMBINED_DESCRIPTION_PARTS"
_SKIP_NAME_SUFFIXES = ("_JS", "_CSS", "_OLD", "_NEW", "_DEFS", "_RE", "_CDN")
_LOG_METHODS = frozenset({"debug", "info", "warning", "warn", "error", "exception", "critical", "log"})
_LOG_RECEIVERS = frozenset({"logger", "log", "logging", "_logger", "_log", "LOGGER", "LOG"})

# ---- Rules -------------------------------------------------------------------------

# Words a capitalised-emphasis check must not flag: initialisms and names the reader sees.
ACRONYMS = frozenset(
    {
        "TRITON", "SWMM", "SLURM", "HTML", "JSON", "YAML", "CSV", "GPU", "GPUS", "MPI", "OPENMP",
        "NAVD", "EPSG", "DEM", "CRS", "CPU", "CPUS", "HPC", "SIF", "UTC", "URL", "HTTP", "HTTPS",
        "NVIDIA", "ZARR", "GIS", "ASCII", "UTF", "PNG", "SVG", "PDF", "RAM", "NOAA", "NASA", "ECDF", "DOCTYPE",
    }
)  # fmt: skip

_HEDGES = r"may|might|typically|usually|generally|likely|where available|best[- ]effort|approximately|honest(?:ly)?"
_NUMBER_WORDS = r"two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|dozen|hundred"
_FILLERS = (
    r"exactly|specifically|importantly|essentially|fundamentally|ultimately|particularly|inherently"
    r"|increasingly|deliberately|simply|clearly|truly"
)
# Planning and tooling words (brief rule 7). Assembled from fragments so that a scan of the
# repository for those words does not match this rule's own vocabulary.
_PLANNING = "|".join(
    [
        "".join(["sha", "pes?"]),
        "".join(["chu", "nks?"]),
        "".join(["work", r"\s+", "packages?"]),
        "".join(["item", r"\s+", r"[A-Z]?\d+"]),
        "specialists?",
        "skills?",
    ]
)
_RETIRED = (
    r"sa_ids?|sub[-_ ]?analys(?:is|es)|invocations?|read[-_ ]models?|sentinels?|sidecars?|crates?"
    r"|child bundles?|compute[-_ ]configs?|model arms?|masters?|sims?"
)

PROSE_RULES: dict[str, re.Pattern[str]] = {
    "em_dash": re.compile("—"),
    "which": re.compile(r"\bwhich\b", re.I),
    "contraction": re.compile(r"\b\w+n['’]t\b", re.I),
    "hedge": re.compile(rf"\b(?:{_HEDGES})\b", re.I),
    "number_word": re.compile(rf"\b(?:{_NUMBER_WORDS})\b", re.I),
    "filler_adverb": re.compile(rf"\b(?:{_FILLERS})\b", re.I),
    "planning_word": re.compile(rf"\b(?:{_PLANNING})\b", re.I),
}
CAPITALS = re.compile(r"\b[A-Z]{4,}\b")
RETIRED = re.compile(rf"(?<![A-Za-z0-9_])(?:{_RETIRED})(?![A-Za-z0-9_])", re.I)
ALL_RULES = (*PROSE_RULES, "capitals", "retired_word")

_WINDOW = 70
_WHOLE_STRING_MAX = 160


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _snippet(text: str, start: int, end: int) -> str:
    if len(text) <= _WHOLE_STRING_MAX:
        return _squash(text)
    return _squash(text[max(0, start - _WINDOW) : end + _WINDOW])


def scan_string(text: str) -> list[tuple[str, str]]:
    """Return ``(rule, snippet)`` for every rule hit in one reader-facing string."""
    hits: list[tuple[str, str]] = []
    for m in RETIRED.finditer(text):
        hits.append(("retired_word", _snippet(text, m.start(), m.end())))
    if len(text.split()) < 3:
        return hits
    for rule, pattern in PROSE_RULES.items():
        for m in pattern.finditer(text):
            hits.append((rule, _snippet(text, m.start(), m.end())))
    for m in CAPITALS.finditer(text):
        if m.group(0) not in ACRONYMS:
            hits.append(("capitals", _snippet(text, m.start(), m.end())))
    return hits


# ---- Python sources ----------------------------------------------------------------


def _is_log_call(node: ast.Call) -> bool:
    func = node.func
    if not (isinstance(func, ast.Attribute) and func.attr in _LOG_METHODS):
        return False
    recv = func.value
    if isinstance(recv, ast.Name):
        return recv.id in _LOG_RECEIVERS
    return isinstance(recv, ast.Call) and isinstance(recv.func, ast.Attribute) and recv.func.attr == "getLogger"


def _fstring_text(node: ast.JoinedStr) -> str:
    parts = []
    for v in node.values:
        parts.append(v.value if isinstance(v, ast.Constant) and isinstance(v.value, str) else "{}")
    return "".join(parts)


def python_strings(tree: ast.AST) -> list[str]:
    """Reader-facing string constants of a parsed module, in source order."""
    out: list[str] = []

    def visit(node: ast.AST, *, module_level_assign: bool = False) -> None:
        if isinstance(node, ast.Raise):
            return
        if isinstance(node, ast.Call) and _is_log_call(node):
            return
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
                body = body[1:]
            for child in body:
                visit(child, module_level_assign=isinstance(node, ast.Module))
            for field in ("decorator_list", "args", "returns", "bases", "keywords"):
                val = getattr(node, field, None)
                for child in val if isinstance(val, list) else ([val] if val is not None else []):
                    visit(child)
            return
        if module_level_assign and isinstance(node, ast.Assign | ast.AnnAssign):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id.endswith(_SKIP_NAME_SUFFIXES) for t in targets):
                return
        if isinstance(node, ast.JoinedStr):
            out.append(_fstring_text(node))
            return
        if isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                out.append(node.value)
            return
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(tree)
    return out


def scan_python(text: str, *, only_constant: str | None = None) -> list[tuple[str, str]]:
    tree = ast.parse(text)
    if only_constant is not None:
        strings: list[str] = []
        for node in tree.body:
            targets = node.targets if isinstance(node, ast.Assign) else [getattr(node, "target", None)]
            if any(isinstance(t, ast.Name) and t.id == only_constant for t in targets):
                strings.extend(python_strings(node.value))  # type: ignore[attr-defined]
        found = strings
    else:
        found = python_strings(tree)
    hits: list[tuple[str, str]] = []
    for s in found:
        hits.extend(scan_string(s))
    return hits


# ---- Caption and template sources --------------------------------------------------

_FENCE = re.compile(r"(?ms)^[ \t]*(`{3,}|~{3,}).*?^[ \t]*\1[ \t]*$")
_CODE_SPAN = re.compile(r"``.*?``|`[^`\n]*`", re.S)
_JINJA = re.compile(r"\{\{.*?\}\}|\{%.*?%\}|\{#.*?#\}", re.S)
_SCRIPT_STYLE = re.compile(r"(?is)<(script|style)\b.*?</\1\s*>")
_TAG = re.compile(r"<[^>]*>")
_QUOTED = re.compile(r"\"([^\"\n]*)\"|'([^'\n]*)'")


def scan_text(text: str, *, suffixes: tuple[str, ...]) -> list[tuple[str, str]]:
    if suffixes[-2:] == (".css", ".j2"):
        strings = [a or b for a, b in _QUOTED.findall(_JINJA.sub(" ", text))]
    else:
        masked = _FENCE.sub(" ", text)
        masked = _CODE_SPAN.sub(" ", masked)
        masked = _JINJA.sub(" ", masked)
        if suffixes[-1:] == (".j2",):
            masked = _TAG.sub(" ", _SCRIPT_STYLE.sub(" ", masked))
        strings = [masked]
    hits: list[tuple[str, str]] = []
    for s in strings:
        hits.extend(scan_string(s))
    return hits


# ---- Population and allowlist files --------------------------------------------------


def population() -> list[str]:
    """Repo-relative POSIX paths of every source the rules read."""
    found: set[Path] = set()
    found.update((PKG / "report_templates" / "captions").glob("*.rst"))
    found.update((PKG / "report_templates").rglob("*.j2"))
    found.update((PKG / "report_renderers").glob("*.py"))
    found.update(PKG / name for name in _PY_EXTRA)
    found.update((PKG / "eda").glob("*.py"))
    found.add(REPO_ROOT / _COMBINED_SRC)
    return sorted(p.relative_to(REPO_ROOT).as_posix() for p in found)


def allowlist_name(source: str, all_sources: list[str]) -> str:
    path = Path(source)
    shared = sum(1 for s in all_sources if Path(s).name == path.name) > 1
    return f"{path.parent.name}.{path.name}.yaml" if shared else f"{path.name}.yaml"


def scan_source(source: str) -> list[tuple[str, str]]:
    path = REPO_ROOT / source
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".py":
        return scan_python(text, only_constant=_COMBINED_CONST if source == _COMBINED_SRC else None)
    return scan_text(text, suffixes=tuple(path.suffixes[-2:]) if path.suffix == ".j2" else (path.suffix,))


def load_allowlist(name: str) -> list[dict]:
    path = ALLOWLIST_DIR / name
    if not path.exists():
        return []
    return (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("entries") or []


def compare(hits: list[tuple[str, str]], entries: list[dict]) -> tuple[list[str], list[str]]:
    """Return ``(unlisted_hits, stale_entries)`` as readable lines."""
    have = Counter(hits)
    allowed: Counter[tuple[str, str]] = Counter()
    for e in entries:
        allowed[(e["rule"], e["string"])] += int(e.get("count", 1))
    unlisted = [f"{rule}: {s!r} (x{n - allowed[(rule, s)]})" for (rule, s), n in have.items() if n > allowed[(rule, s)]]
    stale = [f"{rule}: {s!r} (x{n - have[(rule, s)]})" for (rule, s), n in allowed.items() if n > have[(rule, s)]]
    return unlisted, stale


ALL_SOURCES = population()
SCANNED = [s for s in ALL_SOURCES if s not in TRANSITIONAL]


# ---- The checks ----------------------------------------------------------------------


def test_transitional_sources_are_in_the_population() -> None:
    assert TRANSITIONAL <= set(ALL_SOURCES)


def test_no_two_sources_share_an_allowlist_file_name() -> None:
    names = Counter(allowlist_name(s, ALL_SOURCES) for s in ALL_SOURCES)
    assert [n for n, c in names.items() if c > 1] == []


def test_every_scanned_source_has_an_allowlist_file() -> None:
    missing = [s for s in SCANNED if not (ALLOWLIST_DIR / allowlist_name(s, ALL_SOURCES)).exists()]
    assert missing == []


def test_every_allowlist_file_belongs_to_a_source() -> None:
    expected = {allowlist_name(s, ALL_SOURCES) for s in ALL_SOURCES}
    present = {p.name for p in ALLOWLIST_DIR.glob("*.yaml")} if ALLOWLIST_DIR.is_dir() else set()
    assert sorted(present - expected) == []


def test_allowlist_entries_are_well_formed() -> None:
    bad = []
    for path in sorted(ALLOWLIST_DIR.glob("*.yaml")):
        for e in (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("entries") or []:
            ok = (
                e.get("rule") in ALL_RULES
                and isinstance(e.get("string"), str)
                and e.get("status") in ("baseline", "permitted")
                and isinstance(e.get("reason"), str)
                and e["reason"].strip()
            )
            if not ok:
                bad.append(f"{path.name}: {e}")
    assert bad == []


@pytest.mark.parametrize("source", SCANNED)
def test_source_matches_its_allowlist(source: str) -> None:
    unlisted, stale = compare(scan_source(source), load_allowlist(allowlist_name(source, ALL_SOURCES)))
    assert unlisted == [], f"{source}: hits with no allowlist entry"
    assert stale == [], f"{source}: allowlist entries that match no hit"


# ---- Must-fire controls ----------------------------------------------------------------

_PLANTED = f"""
A = "the output of this step \u2014 saved to disk"
B = "the table which holds the rows"
C = "the file doesn't exist here"
D = "the result is likely correct here"
E = "show two rows per page"
F = "this step simply writes rows"
G = "the NOTICE appears on top"
H = "the {"".join(["sha", "pe"])} of the page"
I = "Sub-analysis label"
"""


def test_planted_source_returns_one_hit_per_rule() -> None:
    hits = scan_python(_PLANTED)
    assert sorted(Counter(rule for rule, _ in hits).items()) == sorted((rule, 1) for rule in ALL_RULES)


def test_retired_word_in_a_two_word_label_is_caught() -> None:
    assert scan_python('LABEL = "Sub-analysis label"') == [("retired_word", "Sub-analysis label")]


def test_logging_argument_and_raise_expression_are_not_read() -> None:
    src = (
        "import logging\n"
        "logger = logging.getLogger(__name__)\n"
        "logger.warning('the table which holds rows')\n"
        "logging.getLogger(__name__).info('the table which holds rows')\n"
        "def f():\n"
        "    '''the docstring which holds rows'''\n"
        "    raise ValueError('the table which holds rows')\n"
        "STYLE_CSS = 'the table which holds rows'\n"
    )
    assert scan_python(src) == []


def test_fstring_is_read_as_its_literal_parts() -> None:
    assert scan_python('X = f"the table which holds {n} rows"') == [("which", "the table which holds {} rows")]


def test_stale_entry_fails_the_comparison() -> None:
    entries = [{"rule": "which", "string": "the table which holds rows", "status": "baseline", "reason": "x"}]
    unlisted, stale = compare([], entries)
    assert unlisted == [] and len(stale) == 1


def test_unlisted_hit_fails_the_comparison() -> None:
    unlisted, stale = compare([("which", "the table which holds rows")], [])
    assert len(unlisted) == 1 and stale == []


def test_a_hit_count_above_the_entry_count_fails() -> None:
    entries = [{"rule": "which", "string": "s", "status": "baseline", "reason": "x", "count": 1}]
    unlisted, _ = compare([("which", "s"), ("which", "s")], entries)
    assert len(unlisted) == 1


# ---- Reviewer tests (build iteration 2c.1) ------------------------------------------------
# Each test quantifies over a class of inputs the rule must catch. The assertion turns on
# whether a hit of the named rule exists, which is true in the current and any corrected
# matcher, never on the snippet wording.

_PLANNING_PLURALS = ["shapes", "skills", "specialists", "work packages", "chunks"]
_RETIRED_PLURALS = ["masters", "read-models", "compute-configs", "sentinels", "sidecars", "crates", "invocations"]


@pytest.mark.parametrize("word", _PLANNING_PLURALS)
def test_plural_of_a_planning_word_is_caught(word: str) -> None:
    """Class: the plural of each planning word. A reader sees "work packages" as readily as "work package"."""
    assert "planning_word" in [rule for rule, _ in scan_string(f"see the {word} here today")]


@pytest.mark.parametrize("word", _RETIRED_PLURALS)
def test_plural_of_a_retired_word_is_caught(word: str) -> None:
    """Class: the plural of each retired word the term table lists."""
    assert "retired_word" in [rule for rule, _ in scan_string(f"all {word}")]


@pytest.mark.parametrize(
    ("rule", "sentence"),
    [
        ("which", "Which rows does this page show"),
        ("hedge", "Likely the first row is wrong"),
        ("number_word", "Two rows are shown here"),
        ("filler_adverb", "Simply read the first row"),
        ("planning_word", "Shape of the first row"),
    ],
)
def test_a_rule_fires_on_a_sentence_initial_capital(rule: str, sentence: str) -> None:
    """Class: a capitalised first word. Reader-facing sentences start with a capital."""
    assert rule in [r for r, _ in scan_string(sentence)]


def test_a_constant_bound_to_a_skipped_name_suffix_is_not_read() -> None:
    """Class: each module-level suffix the shape excludes. Its positive pair is the planted-source test."""
    for suffix in ("_JS", "_CSS", "_OLD", "_NEW", "_DEFS", "_RE", "_CDN"):
        assert scan_python(f"X{suffix} = 'the table which holds rows'") == [], suffix
    assert scan_python("X_TEXT = 'the table which holds rows'") != []


# ---- Reviewer tests (build iteration 2c.2): the page-kind harness -------------------------
# These exercise the harness in tests/test_caption_page_kinds.py, which pages 9 and 10 reuse.

import pandas as pd  # noqa: E402

from hhemt.report_renderers._tabulator_defaults import (  # noqa: E402
    build_columns_spec,
    build_options_dict,
    build_table_fragment,
)
from tests.test_caption_page_kinds import (  # noqa: E402
    _MARKERS,
    PAGE_KINDS,
    missing_names,
    page_text,
)

# Section 8.2 lists these kinds and no others for the pages 2c.2 renders.
_EXPECTED_KIND_IDS = frozenset(
    {
        "P05-plain", "P05-sweep",
        "P06-plain", "P06-sweep", "P06-csv_missing",
        "P07-failures", "P07-all_not_applicable", "P07-record_missing",
        "P08-live", "P08-record_missing",
    }
)  # fmt: skip


def _table_fragment(container_id: str, column: str, cell: str):
    df = pd.DataFrame({column: [cell]})
    columns = build_columns_spec(df, visible_columns_default=None, header_filter=False)
    options = build_options_dict(df, columns_spec=columns, table_height="320px", pagination_size=0, persistence_id=None)
    return build_table_fragment(container_id=container_id, options=options)


def test_page_text_keeps_the_cells_of_every_table_on_a_multi_table_page() -> None:
    """Class: a page that carries several data grids, as the metadata and workflow performance pages do.

    A caption bolds a row or cell name of any grid on its page, so the page text must hold
    the cells of all of them, not the first only.
    """
    first = _table_fragment("t1", "AlphaCol", "alpha-cell")
    second = _table_fragment("t2", "BetaCol", "beta-cell")
    html = (
        "<html><body>"
        + first.markup
        + second.markup
        + "<script>"
        + first.script
        + second.script
        + "</script></body></html>"
    )
    text = page_text(html)
    assert [w for w in ("AlphaCol", "alpha-cell", "BetaCol", "beta-cell") if w not in text] == []


def test_the_listed_kinds_are_the_ones_section_8_2_names() -> None:
    assert {pk.id for pk in PAGE_KINDS} == _EXPECTED_KIND_IDS


def test_every_marker_is_a_non_empty_string() -> None:
    assert [k for k, v in _MARKERS.items() if not v] == []


def test_a_bold_name_is_compared_exactly_not_ignoring_case() -> None:
    assert missing_names("**Model runs**", "the page prints model runs") == ["Model runs"]
    assert missing_names("**Model runs**", "the page prints Model runs") == []


def test_entities_are_unescaped_before_names_are_compared() -> None:
    text = page_text("<td>Peak &amp; mean</td>")
    assert missing_names("**Peak & mean**", text) == []


def test_text_inside_a_script_body_is_not_page_text() -> None:
    text = page_text("<p>shown</p><script>var hidden = 'Hidden name';</script>")
    assert missing_names("**shown**", text) == []
    assert missing_names("**Hidden name**", text) == ["Hidden name"]


# ---- Reviewer tests (build iteration 2c.3): the shared helpers' reader-facing claims ------


def _reset_all_titles() -> list[str]:
    frames = []
    for column_panel in (True, False):
        df = pd.DataFrame({"A": [1]})
        columns = build_columns_spec(df, visible_columns_default=None, header_filter=False)
        options = build_options_dict(
            df, columns_spec=columns, table_height="320px", pagination_size=0, persistence_id="t"
        )
        frames.append((options, build_table_fragment(container_id="t", options=options, column_panel=column_panel)))
    out = []
    for _options, frag in frames:
        out.extend(re.findall(r'class="trf-reset-all"\s+title="([^"]*)"', frag.markup))
    return out


def test_the_reset_all_tooltip_names_only_state_the_table_saves() -> None:
    """Class: every toolbar variant of the Reset all button.

    The button removes the keys Tabulator saved under the table's id, so a state it names
    must be one the options save. A caption of column state needs ``persistence.columns``.
    """
    df = pd.DataFrame({"A": [1]})
    columns = build_columns_spec(df, visible_columns_default=None, header_filter=False)
    saved = build_options_dict(df, columns_spec=columns, table_height="320px", pagination_size=0, persistence_id="t")[
        "persistence"
    ]
    titles = _reset_all_titles()
    assert len(titles) == 2
    assert [t for t in titles if "column" in t.lower() and not saved["columns"]] == []


def test_the_sources_list_names_no_retired_column_of_the_sweep_definition_file() -> None:
    """Class: every column name the Sources list gives for the sweep definition file.

    The loader refuses the retired id column and requires ``member_id``, so a reader who
    looks for a listed column must find one the file can carry.
    """
    from hhemt.report_renderers._figure_emission import collect_sensitivity_source_paths
    from hhemt.sensitivity_analysis import _RETIRED_ID_COLUMNS

    listed = [
        name.strip().lower()
        for src in collect_sensitivity_source_paths("x")
        if src["path"].endswith("sensitivity_analysis_definition.csv")
        for variable in src["variables"]
        for name in variable.split(",")
    ]
    assert listed != []
    assert sorted(set(listed) & _RETIRED_ID_COLUMNS) == []


def test_the_error_panel_sentence_has_no_escaped_em_dash() -> None:
    """The style test skips constants named *_DEFS, so an escaped em dash in the injected error panel
    is checked here by name. Control: the first assertion proves the constant read holds the sentence."""
    from hhemt.report_renderers import _react_surgery

    assert "Please copy the line above" in _react_surgery._GUARD_DEFS
    assert "\\u2014" not in _react_surgery._GUARD_DEFS


_EM_DASH_SPELLINGS = ("\u2014", "\\u2014", "&mdash;", "&#8212;", "&#x2014;")
_ERROR_PANEL_ANCHOR = "Please copy the line above"


def _em_dash_spellings_on_the_sentence_lines(guard_defs: str) -> list[str]:
    """Return each em dash spelling found on the physical lines that carry the error-panel sentence.

    The window is the line holding the sentence's last words and the two lines above it, because the
    sentence is joined across source lines by ``+``. A JavaScript comment elsewhere in the constant is
    outside the window, so a comment that quotes a dash does not fail this check.
    """
    lines = guard_defs.splitlines()
    at = next(i for i, line in enumerate(lines) if _ERROR_PANEL_ANCHOR in line)
    window = "\n".join(lines[max(0, at - 2) : at + 1])
    return [spelling for spelling in _EM_DASH_SPELLINGS if spelling in window]


def test_the_error_panel_sentence_carries_no_em_dash_in_any_spelling() -> None:
    """The injected error-panel sentence prints no em dash, in any of its five spellings.

    Property: no spelling of an em dash appears on the source lines that carry the sentence.
    Class: the five spellings in ``_EM_DASH_SPELLINGS``. The sibling test above reads only the
    six-character JS escape, and the style test skips constants named ``*_DEFS``, so the character
    itself and the HTML entities are read by nothing else.
    Why required: the constant is a non-raw string, so writing a single backslash before ``u2014``
    in its source stores the character, and the sibling test stays green on it. The reader then
    sees a dash that style brief rule 8 forbids.
    What kills it: re-spelling the dash in the sentence as the character, ``&mdash;`` or a
    numeric entity.
    Second correct implementation: a comma, a colon, or a sentence split over three source lines
    still passes, because only the dash spellings are asserted.
    Control: the first loop proves the check fires on each spelling planted into a copy of the
    sentence lines, so an empty result on the real constant is not an artifact of the matcher.
    """
    from hhemt.report_renderers import _react_surgery

    guard_defs = _react_surgery._GUARD_DEFS
    assert _ERROR_PANEL_ANCHOR in guard_defs
    for spelling in _EM_DASH_SPELLINGS:
        planted = guard_defs.replace(_ERROR_PANEL_ANCHOR, f"{spelling} {_ERROR_PANEL_ANCHOR}", 1)
        assert planted != guard_defs, "the planted copy must differ from the real constant"
        assert _em_dash_spellings_on_the_sentence_lines(planted) == [spelling]
    assert _em_dash_spellings_on_the_sentence_lines(guard_defs) == []
