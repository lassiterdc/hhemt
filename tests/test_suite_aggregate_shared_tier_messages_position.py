"""Where the warm-build fact sits in the `[concurrent-build]` problem, and that no printed text
anywhere in the aggregator cites a private README.

The problem this file guards: the `[concurrent-build]` problem is built as a shared header
(`_sbd`, which already states that the compiled tier is written once by the warm build step)
followed by its own `BLOCKING:` clause. A check that searches the whole message for the fact is
satisfied by the header alone, so the BLOCKING clause can lose its justification -- for example
`under a tier that is written once.` -- with that check still green; measured, exactly that edit
passes `tests/test_suite_aggregate_shared_tier_messages.py`. This file reads the clause itself.
It also widens the README check from the two shared-tier messages to every string the module can
print, because the rule is that no printed message names the harness README or its numbered
properties, wherever the message is built.

Nothing here runs a suite: the first test builds a synthetic run directory and calls
`aggregate()` with the fixture builder `tests/test_verdict_partition.py` uses; the second reads
the module's source with `ast`.
"""

import ast
import inspect
import re

import hhemt.suite.aggregate as aggregate_module
from tests.test_verdict_partition import SOFTWARE, _build, _entry

_PRIVATE_CITATION = re.compile(r"readme|propert(?:y|ies)\s+\d", re.IGNORECASE)


def test_concurrent_build_blocking_clause_itself_states_the_warm_build_fact(tmp_path):
    """Property: the text after `BLOCKING:` in `[concurrent-build]` states the warm-build fact.

    Class: every `[concurrent-build]` problem `aggregate()` emits when two chunks write the same
    compiled-tier path. Why it is required: the clause is the verdict-bearing part of the line,
    and the reason it is a problem is that the tier is written once, by the warm build step
    before the chunk array; with that fact only in the shared header, the clause reads as an
    unexplained assertion. What kills it: replacing the clause's justification with a bare "a
    tier that is written once", or with any citation in place of the fact. A second correct
    implementation under which it still passes: wording the clause differently around the same
    two phrases, or restating the fact in a separate sentence after `BLOCKING:`, since only the
    clause's text is read.
    """
    _build(
        tmp_path,
        chunk_entries=[
            {SOFTWARE: _entry("_software", written=True, samples=["triton/build_tritonswmm_cpu/compilation.log"])}
            for _ in range(2)
        ],
    )
    result = aggregate_module.aggregate(tmp_path, scope="array")

    messages = [m for m in result["problems"] if m.startswith("[concurrent-build]")]
    assert messages, f"no [concurrent-build] problem emitted: {result['problems']}"
    for message in messages:
        assert "BLOCKING:" in message, message
        clause = message.split("BLOCKING:", 1)[1]
        assert "warm build step" in clause, f"the BLOCKING clause does not state the fact: {clause!r}"
        assert "before the chunk array" in clause, f"the BLOCKING clause does not state the fact: {clause!r}"


def _non_docstring_string_literals(tree: ast.Module) -> list[ast.Constant]:
    """Every string constant in the module that is not a module, class or function docstring."""
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                docstrings.add(id(first.value))
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings
    ]


def test_no_printable_string_in_the_aggregator_cites_a_readme_or_its_numbered_properties():
    """Property: no string the aggregator can print names a README or a numbered property.

    Class: every string literal in `hhemt.suite.aggregate` other than a docstring -- the
    population every printed message, summary line and problem text is built from. Why it is
    required: the summary is published, and a warrant naming a README or "properties 1 and 2"
    points the reader at a document they cannot open; the rule is about every printed message,
    not only the two shared-tier ones. What kills it: a message, at any site, that names a
    README or a numbered property. A second correct implementation under which it still passes:
    moving the messages into a separate constants module would need this scan pointed there, but
    any wording that states the fact rather than citing a document passes, since only the
    citation pattern is searched. Comments and docstrings are excluded because they are not
    printed.
    """
    # Control on the matcher, which must be able to return non-zero: the exact phrase the rule
    # removed, and the README form, are both caught.
    assert _PRIVATE_CITATION.search("under a tier properties 1 and 2 say is written once")
    assert _PRIVATE_CITATION.search("this README states the compiled tier is shared")

    tree = ast.parse(inspect.getsource(aggregate_module))
    literals = _non_docstring_string_literals(tree)
    assert len(literals) > 50, f"the literal scan found only {len(literals)} strings; it is not reading the module"
    hits = [(n.lineno, n.value[:100]) for n in literals if _PRIVATE_CITATION.search(n.value)]
    assert hits == [], f"printable strings citing a README or numbered property: {hits}"
