"""Reviewer tests for ``hhemt.swmm_runnable``.

Pure text, laptop tier, each test far under 0.15 s. Every test here is GREEN on the reviewed tree and must STAY
green: each is a held test that pins a property the coder's own tests pass without ever stating, because they
read the answer from the module under test (the dummy-chain prefix) or fix one input position (canonical
upper-case headers, single-space rows). The requirement each one enforces, and the mutation that reds
it, are in the reviewer's round file.
"""

from __future__ import annotations

import inspect
import re

import pytest

from hhemt import swmm_runnable
from hhemt.swmm_runnable import SECTION_ROLE_TABLE, derive_role_model, rewrite_forcing
from tests.fixtures.swmm_runnable_inputs import runnable_source_text

_HEADER = re.compile(r"^\[(\w+)\]", re.M)


def test_the_three_names_the_shape_specifies_take_the_shapes_parameters():
    """Property: ``derive_role_model(source_text, role, outfalls_to_couple)`` and ``rewrite_forcing(role_text, fill)``.

    Class: the two callables and the one table that a caller imports by name. A caller written
    from the documented signature (positional or keyword) must bind, so the first parameters carry the documented
    names in the documented order. Why required: the caller builds ``derive_role_model(read(cfg.SWMM_model), role,
    cfg.swmm_outfalls_to_couple)`` and ``rewrite_forcing(swmm_role_text(...), fill)`` from the signature alone.
    What kills it: renaming ``source_text`` to ``text`` (or reordering the first three parameters). A second
    correct implementation: one that adds trailing keyword-only parameters, or defaults ``outfalls_to_couple``.
    """
    derive = list(inspect.signature(derive_role_model).parameters)
    assert derive[:3] == ["source_text", "role", "outfalls_to_couple"]
    assert list(inspect.signature(rewrite_forcing).parameters)[:2] == ["role_text", "fill"]
    assert set(SECTION_ROLE_TABLE["JUNCTIONS"]) == {"full", "hydraulics", "hydrology"}


@pytest.mark.parametrize("role", ["hydraulics", "hydrology"])
def test_the_dummy_chain_is_named_with_the_shapes_literals(role):
    """Property: the chain is ``hhemt_dummy_j``, ``hhemt_dummy_c`` and ``hhemt_dummy_of``, as consumers read them.

    Class: both coupled roles, every outfall listed so the chain is added. Why required: the consumers
    name the chain by these literals, not by the module's constant: the hydrograph writer's skip, the Norfolk
    preflight test and the how-to page. The coder's tests build every expected name from the module's own
    ``DUMMY_NODE_PREFIX``, so a changed prefix passes them while breaking each of those consumers. What kills it:
    changing ``DUMMY_NODE_PREFIX`` to any other string. A second correct implementation: one that moves the
    chain's numeric inverts, length or roughness, which this test does not read.
    """
    out = derive_role_model(runnable_source_text(), role, ["dummy_outfall"])
    names = {
        "JUNCTIONS": "hhemt_dummy_j",
        "OUTFALLS": "hhemt_dummy_of",
        "CONDUITS": "hhemt_dummy_c",
        "XSECTIONS": "hhemt_dummy_c",
    }
    for section, name in names.items():
        block = re.search(rf"^\[{section}\]\n(.*?)(?=^\[|\Z)", out, re.M | re.S)
        assert block is not None, section
        first_tokens = [ln.split()[0] for ln in block.group(1).splitlines() if ln.strip() and not ln.startswith(";")]
        assert name in first_tokens, (section, first_tokens)
    assert swmm_runnable.DUMMY_NODE_PREFIX == "hhemt_dummy_"


def test_every_section_follows_its_table_row_under_lower_case_headers_with_trailing_comments():
    """Property: a header is matched by SWMM's case-folded prefix rule, so the table row applies however it is spelled.

    Class: the 57 sections of the closed table, each written as ``[name] ; note`` in lower case, a position the
    coder's canonical upper-case sink model does not reach. Why required: a runnable model saved by another tool
    spells headers its own way, and SWMM accepts all of them, so a spelling the module misses would keep a
    section the hydraulics role must drop (SWMM would then refuse it) or drop one it must keep. What kills it:
    matching a header by exact equality or by case-sensitive prefix. A second correct implementation: one that
    parses headers with a regex instead of a prefix table, provided it folds case and takes a prefix.
    """
    sink = "".join(f"[{name.lower()}] ; note\n;; marker\n\n" for name in SECTION_ROLE_TABLE)
    for role, extra in (("hydraulics", set()), ("hydrology", {"CONDUITS", "XSECTIONS"})):
        out = derive_role_model(sink, role, [])
        kept = {name.upper() for name in _HEADER.findall(out)}
        expected = {name for name, cells in SECTION_ROLE_TABLE.items() if cells[role][0] != "D"} | extra
        assert kept == expected, (role, sorted(kept ^ expected))


def test_a_tab_separated_outfall_row_with_a_trailing_comment_converts_with_its_invert_and_case_kept():
    """Property: the converted junction carries the source row's name spelling and invert text exactly.

    Class: an ``[OUTFALLS]`` row whose columns are tab-separated and which ends in a comment, a position the
    coder's single-space rows do not reach. Why required: SWMM splits on any run of blanks and drops the
    comment, so a model saved with tabs is valid, and a converter that splits on a single space would read the
    wrong invert and write a junction at the wrong elevation. What kills it: tokenising the row with
    ``line.split(" ")``. A second correct implementation: one that formats the row with other column widths.
    """
    src = runnable_source_text().replace("[OUTFALLS]\n", "[OUTFALLS]\nTab_Out\t-1.5\tFREE\tNO\t;note\n", 1)
    for role in ("hydraulics", "hydrology"):
        out = derive_role_model(src, role, ["tab_out"])
        junctions = re.search(r"^\[JUNCTIONS\]\n(.*?)(?=^\[|\Z)", out, re.M | re.S)
        assert junctions is not None
        rows = [ln.split() for ln in junctions.group(1).splitlines() if ln.strip() and not ln.startswith(";")]
        assert ["Tab_Out", "-1.5", "0", "0", "0", "0"] in rows, (role, rows)
        assert "Tab_Out\t" not in out
