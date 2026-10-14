"""Every advisory tag gets a preamble line, whichever list method the emitting site uses.

The problem this file guards: the advisories preamble in `render_summary_md` is hand-written,
while the `[name]` tags it explains are written at their emitting sites. A collector that only
recognises `advisories.append(...)` lets a new kind arrive through `advisories.extend([...])`,
`advisories.insert(...)`, `advisories += [...]`, or a separate list merged in, and ship with no
reading rule while the existing preamble test stays green -- measured: adding
`advisories.extend(["[new-kind] ..."])` to the module leaves that test passing. This test
collects every `[name]`-led string literal that is handed to ANY list mutation in the module,
routes it by the receiving list's name, and requires each non-problem tag to be explained.

Nothing here runs a suite: the tags are read from source with `ast`, and the summary is
rendered from a minimal hand-built result.
"""

import ast
import inspect
import re

import hhemt.suite.aggregate as aggregate_module

_TAG = re.compile(r"\[([a-z][a-z-]*)\]")
_MUTATORS = {"append", "extend", "insert"}


def _leading_tag(node: ast.AST) -> str | None:
    """The `[name]` a string literal (or f-string) starts with, else None."""
    if isinstance(node, ast.JoinedStr) and node.values:
        node = node.values[0]
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        match = _TAG.match(node.value)
        return match.group(1) if match else None
    return None


def _string_elements(node: ast.AST) -> list[ast.AST]:
    """The node itself, or each element when it is a list/tuple/set display."""
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return list(node.elts)
    return [node]


def _tags_by_receiver() -> dict[str, set[str]]:
    """Map each list name to the tags written into it by any mutation form."""
    found: dict[str, set[str]] = {}
    for node in ast.walk(ast.parse(inspect.getsource(aggregate_module))):
        receiver, payloads = None, []
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _MUTATORS
            and isinstance(node.func.value, ast.Name)
        ):
            receiver = node.func.value.id
            args = node.args[1:] if node.func.attr == "insert" else node.args
            payloads = [e for a in args for e in _string_elements(a)]
        elif isinstance(node, ast.AugAssign) and isinstance(node.op, ast.Add) and isinstance(node.target, ast.Name):
            receiver = node.target.id
            payloads = _string_elements(node.value)
        for payload in payloads:
            tag = _leading_tag(payload)
            if tag:
                found.setdefault(receiver, set()).add(tag)
    return found


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


def test_every_advisory_tag_is_explained_whatever_mutation_form_emits_it():
    """Property: every `[name]` tag written into a non-problem list, by any list mutation, has a
    reading-rule line in the rendered advisories preamble.

    Class: every `[name]`-led string literal passed to `.append`, `.extend` or `.insert` on a
    named list, or added with `+=`, anywhere in the module, routed by the list's name -- a list
    whose name contains `problem` holds blocking findings and is excluded; every other list is
    an advisory list. Why it is required: the advisories heading tells the reader its findings
    did not affect the verdict, so a kind with no reading rule is one the reader cannot judge,
    and the emitting form is an implementation detail a new site is free to choose. What kills
    it: a new advisory tag emitted through any of those forms without a preamble line. A second
    correct implementation under which it still passes: a preamble generated from a tag
    registry, or advisories emitted through one helper function, since only the tags reaching a
    non-problem list and the rendered lines are compared.

    Not established: a tag built at runtime (`f"[{kind}] ..."`) or assigned to a variable before
    it is appended is invisible to a source scan; no current site does either.
    """
    by_receiver = _tags_by_receiver()
    blocking = {t for name, tags in by_receiver.items() if "problem" in name for t in tags}
    advisory = {t for name, tags in by_receiver.items() if "problem" not in name for t in tags}

    # Controls on the instrument, measured on the module as it stands: the four advisory kinds
    # are found, and the one blocking kind is routed away from them.
    assert {
        "classifier-self-check",
        "shared-by-design-write",
        "derivation-under-connected",
        "complement-derivation",
    } <= advisory, f"collector missed a known advisory site: {sorted(advisory)}"
    assert "concurrent-build" in blocking and "concurrent-build" not in advisory, by_receiver

    listed = set(re.findall(r"^- `\[([a-z-]+)\]`", _rendered_preamble(), flags=re.MULTILINE))
    assert advisory - listed == set(), f"advisory tag(s) with no preamble line: {sorted(advisory - listed)}"
