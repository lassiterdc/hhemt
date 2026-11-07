"""Tests of the how-to pass: four page-versus-code corrections, each with a must-fire control.

Each correction restates a sentence of a how-to page as the fact the code decides:

- RPX-03, `docs/how-to/reprex-roundtrip.md`: the `report` section of a bundle's `cfg_analysis.yaml` holds the
  settings of the render record when one exists (`emit_relative_config_pair`), not the analysis config's section.
- RPX-12, same page: the emit builds a zero-user-info scanner and raises its hits; the one caller of
  `assert_bundle_zero_user_info` is the consume-side check.
- RXD-01 and RXD-04, `docs/how-to/running-an-experiment-definition.md`: no module reads the `destinations` field,
  so the page neither promises that a provisioning step stages the input nor says `--cluster` selects `destinations`.

Every test reads source text and page text; none builds an analysis, compiles or runs a solver. The behaviour of the
emitter itself is exercised by `tests/test_bundle_run_entry_report_config.py`; the tests here pin the page to the code
facts that make its sentence true, through the syntax tree and not through a wording of the page. Each finder has a
control that feeds it the mutated code the finder exists to refuse.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src" / "hhemt"
_REPRO = _REPO / "docs" / "how-to" / "reprex-roundtrip.md"
_EXPERIMENT = _REPO / "docs" / "how-to" / "running-an-experiment-definition.md"


def _is_str(node: ast.AST, value: str) -> bool:
    return isinstance(node, ast.Constant) and node.value == value


def report_section_follows_the_record(source: str, function: str = "emit_relative_config_pair") -> bool:
    """True when `function` assigns `cfg_dict["report"]` from the `report_settings` key of a record it reads.

    The check reads the function body: an assignment whose target is the subscript `["report"]` and whose value is
    the subscript `["report_settings"]`, in a function that also calls `read_report_settings_record`.
    """
    tree = ast.parse(source)
    for fn in ast.walk(tree):
        if not (isinstance(fn, ast.FunctionDef) and fn.name == function):
            continue
        reads = any(
            isinstance(n, ast.Call)
            and getattr(n.func, "id", getattr(n.func, "attr", "")) == "read_report_settings_record"
            for n in ast.walk(fn)
        )
        assigns = any(
            isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Subscript) and _is_str(t.slice, "report") for t in n.targets)
            and isinstance(n.value, ast.Subscript)
            and _is_str(n.value.slice, "report_settings")
            for n in ast.walk(fn)
        )
        return reads and assigns
    return False


def callers_of(source: str, name: str) -> list[str]:
    """Names of the functions in `source` that call `name`, by function name or attribute."""
    tree = ast.parse(source)
    found: list[str] = []
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef):
            for node in ast.walk(fn):
                if isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", "")) == name:
                    found.append(fn.name)
    return sorted(set(found))


def readers_of_field(source: str, field: str) -> list[int]:
    """Line numbers where `source` reads `field` as an attribute, a string subscript or a keyword argument."""
    tree = ast.parse(source)
    lines: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == field and isinstance(node.ctx, ast.Load):
            lines.append(node.lineno)
        elif isinstance(node, ast.Subscript) and _is_str(node.slice, field):
            lines.append(node.lineno)
        elif isinstance(node, ast.keyword) and node.arg == field:
            lines.append(node.value.lineno)
    return sorted(lines)


def section_after(text: str, heading: str) -> str:
    """The text after `heading` up to the next level-two heading, or to the end of `text`."""
    return text.split(heading, 1)[1].split("\n## ", 1)[0]


def bullet(text: str, lead: str) -> str:
    """The list item of `text` whose first line starts with `- ` and `lead`, joined into one line."""
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith(f"- {lead}"))
    end = start + 1
    while end < len(lines) and lines[end].startswith("  "):
        end += 1
    return re.sub(r"\s+", " ", " ".join(lines[start:end]))


# --- RPX-03 -------------------------------------------------------------------------------------------


def test_the_bundle_config_bullet_names_the_report_exception_the_emitter_makes():
    """Property: the page says the `report` section follows the render record exactly when the emitter makes it so.

    Class: the bundle-root bullet for `cfg_analysis.yaml` against `emit_relative_config_pair`. The plausible wrong
    implementation is an emitter that copies the analysis config's `report` section unchanged while the page keeps
    the exception, or the reverse (the page says `kept as written` with no exception while the emitter rewrites it).
    """
    code = (_SRC / "bundle" / "_emit.py").read_text(encoding="utf-8")
    item = bullet(_REPRO.read_text(encoding="utf-8"), "`cfg_system.yaml`")
    page_names_exception = "`report` section" in item and "render record" in item and "first line" in item
    assert report_section_follows_the_record(code) == page_names_exception is True


def test_control_the_record_finder_fires_on_an_emitter_that_copies_the_section_unchanged():
    """Must-fire control: the emitter that never replaces `report` makes the finder return False."""
    copying = (
        "def emit_relative_config_pair(analysis, staging):\n"
        "    cfg_dict = analysis.cfg_analysis.model_dump()\n"
        "    return cfg_dict\n"
    )
    replacing = (
        "def emit_relative_config_pair(analysis, staging):\n"
        "    record = read_report_settings_record(analysis)\n"
        "    cfg_dict = {}\n"
        "    cfg_dict['report'] = record['report_settings']\n"
    )
    assert report_section_follows_the_record(replacing) is True
    assert report_section_follows_the_record(copying) is False


# --- RPX-12 -------------------------------------------------------------------------------------------


def test_the_emit_scans_and_refuses_and_the_page_says_so():
    """Property: the emit builds a zero-user-info scanner and raises its hits; the page says the emit refuses.

    Class: every call site, in `src/hhemt`, of `for_emit`, `raise_if_hits` and `assert_bundle_zero_user_info`.
    Why it is required: the page tells a producer that an emit which finds producer values writes no archive, and that
    `hhemt reprex check` reports without failing. Both hold only while an emit function raises the scanner's hits and
    the raising gate has no caller but the reporting wrapper.
    What kills it: an `emit_bundle` that builds the scanner and never raises its hits, a consume-side check that calls
    the raising gate directly, or a page section without the `ProcessingError` code span.
    A second correct implementation: the raise moved into another helper of `bundle/_emit.py` still passes, because
    the emit-side assertions name modules and not functions.
    """

    def sites_of(name: str) -> dict[str, list[str]]:
        sites: dict[str, list[str]] = {}
        for path in sorted(_SRC.rglob("*.py")):
            found = callers_of(path.read_text(encoding="utf-8"), name)
            if found:
                sites[path.relative_to(_SRC).as_posix()] = found
        return sites

    assert set(sites_of("for_emit")) == {"bundle/_emit.py"}
    assert set(sites_of("raise_if_hits")) == {"bundle/_emit.py", "bundle/_reprex_gate.py"}
    assert sites_of("assert_bundle_zero_user_info") == {"bundle/_reprex.py": ["_scan_zero_user_info"]}
    section = section_after(_REPRO.read_text(encoding="utf-8"), "## Producer information in the bundle")
    assert "`ProcessingError`" in section
    assert "`hhemt reprex check`" in section


def test_control_the_caller_finder_fires_on_an_emit_function_that_calls_the_gate():
    """Must-fire control: a call inside an emit function is reported, and a bare mention is not."""
    emitting = "def emit_bundle(root):\n    assert_bundle_zero_user_info(root)\n"
    mentioning = "def emit_bundle(root):\n    name = 'assert_bundle_zero_user_info'\n"
    assert callers_of(emitting, "assert_bundle_zero_user_info") == ["emit_bundle"]
    assert callers_of(mentioning, "assert_bundle_zero_user_info") == []
    attribute_call = "def emit_bundle(root):\n    ZeroUserInfoScanner.for_emit(root)\n"
    assert callers_of(attribute_call, "for_emit") == ["emit_bundle"]
    page = "## A\nraises `ProcessingError`\n## B\nreads `hhemt reprex check`\n"
    assert section_after(page, "## A") == "\nraises `ProcessingError`"
    assert "`hhemt reprex check`" not in section_after(page, "## A")


# --- RXD-01 and RXD-04 --------------------------------------------------------------------------------


def test_no_module_reads_destinations_and_the_page_promises_no_selection_or_staging():
    """Property: nothing in `src/hhemt` reads the `destinations` field, and the page claims neither a staging step
    nor a selection of `destinations` by `--cluster`.

    Class: every read of `destinations` as an attribute, a string subscript or a keyword argument in `src/hhemt`.
    The plausible wrong implementation is a page that keeps the comment `where the provisioning stages it` or the
    clause `and the per-cluster destinations that apply` after the code reads the field nowhere.
    """
    reads: dict[str, list[int]] = {}
    for path in sorted(_SRC.rglob("*.py")):
        found = readers_of_field(path.read_text(encoding="utf-8"), "destinations")
        if found:
            reads[path.relative_to(_SRC).as_posix()] = found
    assert reads == {}
    page = re.sub(r"\s+", " ", _EXPERIMENT.read_text(encoding="utf-8"))
    assert not re.search(r"provisioning|stages it", page, re.I)
    assert not re.search(r"per-cluster\s+`?destinations`?\s+that apply", page)
    assert "`--cluster` selects the `hpc_system_config[<cluster>]` entry." in page


def test_control_the_field_finder_fires_on_each_form_of_a_read():
    """Must-fire control: the attribute, subscript and keyword forms are each found; a definition is not."""
    forms = "x = cfg.destinations\ny = cfg['destinations']\nf(destinations=z)\n"
    definition = "class C:\n    destinations: dict = {}\n"
    assert readers_of_field(forms, "destinations") == [1, 2, 3]
    assert readers_of_field(definition, "destinations") == []
