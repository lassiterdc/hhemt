"""The suite summary's advisories preamble explains every advisory tag the aggregator can emit.

The problem this file guards: `aggregate()` puts some findings into `advisories` rather than
`problems`, each led by a `[name]` tag, and `render_summary_md` prints a standing preamble that
explains how to read each kind and says how many kinds there are. The preamble is hand-written
while the tags live at their emitting sites, so a new advisory site can ship with no reading
rule and a stale count, and a reader then has an unexplained finding under a heading that says
it did not affect the verdict. This test collects the tags from the module's own source and
requires the rendered preamble to explain each and to count them correctly.

Nothing here runs a suite: the tags are read from source with `ast`, and the summary is
rendered from a minimal hand-built result.
"""

import ast
import inspect
import re

import hhemt.suite.aggregate as aggregate_module

_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8}
_ADVISORY_LISTS = {"advisories", "entry_advisories"}


def _emitted_advisory_tags() -> set[str]:
    """Return every leading `[name]` of a string appended to an advisory list in the module."""
    tags: set[str] = set()
    for node in ast.walk(ast.parse(inspect.getsource(aggregate_module))):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "append"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in _ADVISORY_LISTS
            and node.args
        ):
            continue
        first = node.args[0]
        head = first.values[0] if isinstance(first, ast.JoinedStr) else first
        if isinstance(head, ast.Constant) and isinstance(head.value, str):
            match = re.match(r"\[([a-z-]+)\]", head.value)
            if match:
                tags.add(match.group(1))
    return tags


class _LenientResult(dict):
    """A result dict whose unset keys render as a placeholder, so only advisories matter."""

    def __missing__(self, key):
        return "-"


def _rendered_preamble() -> str:
    result = _LenientResult(
        scope="array",
        run_id="r",
        verdict_line="v",
        problems=[],
        advisories=["[placeholder] an advisory, so the section renders"],
        unevaluated_by_cause={},
        chunks=[],
        slug_entry_classification={},
        ignored_transient_entries=[],
    )
    text = aggregate_module.render_summary_md(result)
    start = text.index("## Advisories")
    return text[start : text.index("- [placeholder]", start)]


def test_preamble_explains_and_counts_every_emitted_advisory_tag():
    """Property: every advisory tag `aggregate()` can emit has a reading-rule line in the
    rendered preamble, and the preamble's stated count of kinds equals the lines it lists.

    Class: every `[name]` led string appended to `advisories` or `entry_advisories` anywhere in
    the module, against the preamble `render_summary_md` prints whenever advisories exist. Why it
    is required: the advisories section says its findings did not affect the verdict, so a kind
    with no reading rule is the one a reader cannot tell is safe to ignore, and a count that
    disagrees with the list tells the reader the list is incomplete. What kills it: an advisory
    site whose tag has no preamble line (as `[complement-derivation]` had), or a stated count that
    is not the number of tag lines. A second correct implementation under which it still passes:
    a preamble generated from a tag registry rather than written by hand, since only the
    rendered lines and the stated count are read.
    """
    emitted = _emitted_advisory_tags()
    assert {"classifier-self-check", "shared-by-design-write", "derivation-under-connected"} <= emitted, (
        f"tag collection is broken: found only {sorted(emitted)}"
    )

    preamble = _rendered_preamble()
    listed = set(re.findall(r"^- `\[([a-z-]+)\]`", preamble, flags=re.MULTILINE))
    assert emitted - listed == set(), f"emitted but unexplained in the preamble: {sorted(emitted - listed)}"

    counts = re.findall(r"\b(" + "|".join(_NUMBER_WORDS) + r")\s+kinds\b", preamble.lower())
    assert counts, f"the preamble states no count of kinds:\n{preamble}"
    assert {_NUMBER_WORDS[c] for c in counts} == {len(listed)}, (
        f"stated count(s) {counts} disagree with the {len(listed)} kinds listed:\n{preamble}"
    )
