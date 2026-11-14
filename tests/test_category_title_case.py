"""Every sidebar category name is in title case, and the sidebar ranks a category without regard to case.

The report's sidebar names its categories in `category_order` (one tuple per named
reporting set) and in the `category` of each figure's `RuleSpecTemplate`. The
developer's ruling is that those names are consistent: each word starts with a capital
letter, except a closed list of minor words and symbols. "Workflow performance" was the
one category that broke the rule, and it was renamed "Workflow Performance".

A Snakefile written before that rename still carries `category="Workflow performance"`
while the order list the post-render surgery receives carries "Workflow Performance".
`render_report` reads the Snakefile as the last run wrote it, so the comparator that
ranks the sidebar must rank the two spellings alike. That comparator is JavaScript that
the surgery writes into the page, so the ranking test runs it under `node`.

No test here compiles a solver or runs a simulation.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess

import pytest

from hhemt.report_renderers import _reporting_sets
from hhemt.report_renderers._react_surgery import apply_post_process_surgery
from hhemt.report_renderers._reporting_sets import REPORTING_SETS, RuleSpecTemplate

# Words and symbols a category name may leave in lower case. Closed: a new entry is a
# decision, so it is added here and not inferred.
MINOR_WORDS = frozenset({"and", "&"})


def uncapitalised_words(name: str) -> list[str]:
    """The words of `name` that do not start with a capital letter and are not minor words.

    A hyphenated word counts as two, so "Cross-experiment" yields "experiment".
    """
    words = [part for chunk in name.split() for part in chunk.split("-") if part]
    return [w for w in words if w not in MINOR_WORDS and not w[:1].isupper()]


def registered_category_names() -> dict[str, set[str]]:
    """Every category name the registry declares, keyed by the name and valued by where it is declared."""
    found: dict[str, set[str]] = {}
    for set_name, rset in REPORTING_SETS.items():
        for name in rset.category_order:
            found.setdefault(name, set()).add(f"{set_name}.category_order")
        for selection in rset.renderer_selection:
            for tmpl in selection.rule_spec_template:
                cat = tmpl.report_kwargs.get("category")
                if cat is not None:
                    found.setdefault(cat, set()).add(f"{set_name}.{tmpl.rule_name}")
    # Templates not yet attached to any set are still part of the registry's vocabulary.
    for attr, value in vars(_reporting_sets).items():
        if isinstance(value, RuleSpecTemplate):
            cat = value.report_kwargs.get("category")
            if cat is not None:
                found.setdefault(cat, set()).add(attr)
    return found


def test_every_registered_category_name_is_in_title_case() -> None:
    names = registered_category_names()
    assert names, "the registry declared no category, so this test checked nothing"
    offenders = {name: uncapitalised_words(name) for name in names if uncapitalised_words(name)}
    assert offenders == {}, (
        "category names whose words do not all start with a capital letter "
        f"(word list per name; declared at {[sorted(names[n]) for n in offenders]})"
    )


def test_the_registry_covers_the_categories_a_reader_sees() -> None:
    """The population is not empty or one-sided: it holds an order-only name, a template-only name,
    a hyphenated name and a name carrying a minor word and a symbol."""
    names = registered_category_names()
    assert "Workflow Performance" in names
    assert "Appendix" in names  # declared by a template only, ranked by no order list
    assert "Cross-Experiment Results" in names  # hyphenated
    assert "Errors and Warnings" in names  # minor word
    assert "Combine Provenance & Compatibility" in names  # symbol


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Workflow performance", ["performance"]),  # the retired spelling
        ("Cross-experiment Results", ["experiment"]),  # a hyphenated word counts as two
        ("errors and Warnings", ["errors"]),  # a minor word is exempt, its neighbour is not
        ("Errors and Warnings", []),
        ("Combine Provenance & Compatibility", []),
        ("Key Results", []),
    ],
)
def test_the_title_case_check_fires_on_the_property_not_the_token(name: str, expected: list[str]) -> None:
    """Must-fire control. Each case varies the case of one word, not the token under test, so a
    checker that matched only the string "performance" would fail the hyphen and minor-word rows."""
    assert uncapitalised_words(name) == expected


def test_a_registry_with_one_lowercase_word_fails_the_registry_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """Must-fire control on the registry check: respelling one registered name reports exactly that name."""
    default = REPORTING_SETS["default"]
    respelled = tuple("Workflow performance" if n == "Workflow Performance" else n for n in default.category_order)
    assert respelled != default.category_order
    monkeypatch.setitem(REPORTING_SETS, "default", default.__class__(**{**vars(default), "category_order": respelled}))
    offenders = [n for n in registered_category_names() if uncapitalised_words(n)]
    assert offenders == ["Workflow performance"]


# --- the sidebar comparator ranks a category without regard to case ---------------------------

_COMPARATOR_RE = re.compile(r"\(a, b\) => \{const ORDER = .*?a\.localeCompare\(b\);\}", re.S)
_NODE = shutil.which("node")


def _sidebar_order(category_order: list[str], categories: list[str]) -> list[str]:
    """Sort `categories` with the comparator the surgery writes into the page, run under node."""
    out = apply_post_process_surgery("x.sort((a, b) => a.localeCompare(b));", category_order=category_order)
    match = _COMPARATOR_RE.search(out)
    assert match is not None, "surgery wrote no ORDER comparator into the page"
    script = f"const cmp = {match.group(0)};\nconsole.log(JSON.stringify({categories!r}.sort(cmp)));\n"
    done = subprocess.run([_NODE, "-"], input=script, capture_output=True, text=True, timeout=60, check=False)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


_ORDER = ["Workflow Status", "Key Results", "Workflow Performance", "Metadata"]


@pytest.mark.skipif(_NODE is None, reason="node is not installed, and the comparator is JavaScript")
def test_a_snakefile_written_before_the_rename_still_ranks_the_page_in_place() -> None:
    """Class: a category the Snakefile spells in the old case while the order list spells it in title case.

    The category name arrives in the old spelling because `render_report` reads the last
    Snakefile as written. The comparator must rank it where the order list puts "Workflow
    Performance", between "Key Results" and "Metadata". With an exact-name lookup the old
    spelling ranks 99 and sorts after "Metadata".
    """
    got = _sidebar_order(_ORDER, ["Metadata", "Workflow performance", "Key Results", "Workflow Status"])
    assert got == ["Workflow Status", "Key Results", "Workflow performance", "Metadata"]


@pytest.mark.skipif(_NODE is None, reason="node is not installed, and the comparator is JavaScript")
def test_the_comparator_keeps_exact_names_and_the_alphabetical_tail() -> None:
    """The exact spelling ranks as before; a category the order list lacks sorts last, alphabetically."""
    got = _sidebar_order(_ORDER, ["Zeta", "Metadata", "Appendix", "Workflow Performance", "Key Results"])
    assert got == ["Key Results", "Workflow Performance", "Metadata", "Appendix", "Zeta"]


# --- reviewer additions (round 1, chunk 4a.1) -------------------------------------------------


def test_the_comparator_text_looks_up_the_exact_name_before_the_name_without_case() -> None:
    """NEW CAPABILITY (reviewer). The node-driven tests above skip when `node` is absent, so this
    source-level test keeps the guard alive without it. Class: the comparator the surgery writes.
    Property: the rank expression tries `ORDER[k]` first and the lower-cased match second, then 99.
    Must-fire: the control string is the exact-only comparator the surgery wrote before."""
    out = apply_post_process_surgery("x.sort((a, b) => a.localeCompare(b));", category_order=["Metadata"])
    match = _COMPARATOR_RE.search(out)
    assert match is not None
    text = match.group(0)
    exact_first = re.search(
        r"ORDER\[k\] \?\? ORDER\[Object\.keys\(ORDER\)\.find\(.*toLowerCase\(\).*\)\] \?\? 99", text
    )

    assert exact_first is not None
    assert re.search(r"ORDER\[k\] \?\? ORDER\[Object", "return (ORDER[a] ?? 99) - (ORDER[b] ?? 99)") is None


@pytest.mark.skipif(_NODE is None, reason="node is not installed, and the comparator is JavaScript")
def test_an_exact_name_wins_over_a_name_that_differs_only_in_case() -> None:
    """NEW CAPABILITY (reviewer). Class: an order list holding two names that differ only in case.
    The exact spelling ranks at its own position; a case-insensitive lookup that ignored the exact
    name would give both the first match's rank. Must-fire: swapping the lookup order moves a name."""
    order = ["Alpha", "alpha"]
    got = _sidebar_order(order, ["alpha", "Alpha"])
    assert got == ["Alpha", "alpha"]
