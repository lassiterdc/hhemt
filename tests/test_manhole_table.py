"""The per-node manhole table: its fields, the one rule that judges it, the file TRITON reads and the cfg line.

The problem this file pins. A manhole table gives each coupling node its own diameter and loss coefficient. A
table that does not cover the coupling nodes, or that names a node the model does not couple, would run some
nodes at the constants while the configuration says otherwise, and TRITON would not say so. The rule that judges
a table is one function, `manhole_table_problems`, called at two moments: run entry, before any job is queued,
and scenario preparation, against the exact set TRITON couples. A build that does not read the cfg key would
ignore the table without a word, so preparation refuses it.

Every test here reads files, builds stand-ins or calls a pure function. None compiles or runs a solver, so the
file runs on a laptop. Every fixture table gives a row two values that differ from each other and from the two
constants (`J1 0.6 0.35` against the constants 1.2 and 0.1), and the assertions read them by position in the
written file or by name, so a reader that swaps the two columns fails.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from hhemt.exceptions import ConfigurationError

_HEADER = "node_id,manhole_diameter_m,manhole_loss_coefficient"
_DIAMETER_CONSTANT = 1.2
_LOSS_CONSTANT = 0.1
_NODES = ["J1", "J2", "J3"]

_INP = """
[OPTIONS]
FLOW_UNITS CMS

[JUNCTIONS]
{junctions}

[OUTFALLS]
O1 0 FREE NO

[CONDUITS]
C1 J1 O1 10 0.013 0 0 0 0

[XSECTIONS]
C1 CIRCULAR 1 0 0 0 1

[INFLOWS]
{inflows}

[COORDINATES]
{coordinates}
"""
_JUNCTIONS = "\n".join(f"{n} 1 2 0 0 0" for n in _NODES)
_COORDINATES = "J1 5 5\nJ2 15 5\nJ3 25 5\nO1 35 5"


def _table(tmp_path: Path, body: str, *, header: str | None = _HEADER, name: str = "table.csv") -> Path:
    """A manhole table with ``header`` (omitted when None) above the rows of ``body``."""
    path = tmp_path / name
    path.write_text(("" if header is None else header + "\n") + body)
    return path


def _context(tmp_path: Path) -> object:
    from hhemt.scenario import ManholeTableContext

    return ManholeTableContext(
        table=tmp_path / "table.csv",
        model="model.inp",
        diameter_constant=_DIAMETER_CONSTANT,
        loss_constant=_LOSS_CONSTANT,
    )


def _problems(tmp_path: Path, body: str, *, fill: bool = False, nodes=None, header: str | None = _HEADER) -> list[str]:
    from hhemt.scenario import manhole_table_problems, read_manhole_table

    rows = read_manhole_table(_table(tmp_path, body, header=header))
    return manhole_table_problems(rows, _NODES if nodes is None else nodes, fill, _context(tmp_path))


_COVERING = "J1,0.6,0.35\nJ2,0.8,0.25\nJ3,0.9,0.15\n"


def test_a_table_covering_every_coupling_node_has_no_problem(tmp_path):
    """Property (control): a complete, well-formed table is accepted with the toggle off.

    Class: one row per coupling node, each row's two values differing from each other and from the constants.
    Why: every refusal below is only informative against an input the same rule accepts. What kills it:
    refusing a table whatever its content. Second correct implementation: judging the rows in a different
    order; the assertion reads only the problem list.
    """
    assert _problems(tmp_path, _COVERING) == []


@pytest.mark.parametrize(
    ("violating", "satisfying", "column"),
    [
        ("J1,0.6,0.35\nJ2,,0.25\nJ3,0.9,0.15\n", "J3,0.9,0.15\nJ2,0.8,0.25\nJ1,0.6,0.35\n", "manhole_diameter_m"),
        ("J1,0.6,0.35\nJ2,0.8,0.25\nJ3,0.9,\n", "J2,0.8,0.25\nJ3,0.9,0.15\nJ1,0.6,0.35\n", "manhole_loss_coefficient"),
        ("J1,0.6\nJ2,0.8,0.25\nJ3,0.9,0.15\n", "J2,0.8,0.25\nJ1,0.6,0.35\nJ3,0.9,0.15\n", "manhole_loss_coefficient"),
    ],
    ids=["empty diameter, middle row", "empty loss, last row", "one value only, first row"],
)
def test_a_row_without_both_values_is_refused_naming_the_row_and_the_column(tmp_path, violating, satisfying, column):
    """Property: a row carrying only one of the two values is refused, whichever value is absent.

    Class: an empty diameter cell, an empty loss cell, and a row with a single value, each at a different row
    position, against the same table with its rows in another order and every cell filled. Why: the solver
    refuses such a row too, and a table whose nodes silently take a default for the missing value would hide
    the gap. What kills it: reading a missing cell as zero or as the constant, or checking only the first
    row. Second correct implementation: refusing at parse time; the assertions read the problem text.
    """
    problems = _problems(tmp_path, violating)
    assert any(column in p and "empty" in p for p in problems), problems
    assert _problems(tmp_path, satisfying) == []


@pytest.mark.parametrize("value", ["0", "-0.5", "nan", "inf", "-inf", "wide"])
def test_a_diameter_that_is_not_finite_and_above_zero_is_refused(tmp_path, value):
    """Property: a diameter must be a finite number above zero.

    Class: zero, negative, not-a-number, both infinities and a word, on the last row, against a tiny positive
    diameter on the first row. Why: the manhole area and perimeter are computed from it, so zero or a negative
    value would divide by or square a meaningless number. What kills it: testing ``>= 0``, or ``float()`` alone
    (which accepts ``nan`` and ``inf``). Second correct implementation: ``math.isfinite`` first and the sign
    after; the assertions read the problem text.
    """
    problems = _problems(tmp_path, f"J1,0.001,0.35\nJ2,0.8,0.25\nJ3,{value},0.15\n")
    assert any("manhole_diameter_m" in p and "line 4" in p for p in problems), problems
    assert _problems(tmp_path, "J3,0.001,0.35\nJ2,0.8,0.25\nJ1,0.9,0.15\n") == []


@pytest.mark.parametrize("value", ["-0.01", "nan", "inf", "loss"])
def test_a_loss_coefficient_that_is_not_finite_and_not_negative_is_refused(tmp_path, value):
    """Property: a loss coefficient must be a finite number, zero allowed, negative refused.

    Class: a small negative, not-a-number, infinity and a word, on the first row, against a loss of exactly
    zero on the last row. Why: the loss multiplies the exchange flow, and zero is a legal frictionless
    coefficient. What kills it: testing ``> 0``, which refuses zero, or accepting a negative value. Second
    correct implementation: the same bounds in another order; the assertions read the problem text.
    """
    problems = _problems(tmp_path, f"J1,0.6,{value}\nJ2,0.8,0.25\nJ3,0.9,0.15\n")
    assert any("manhole_loss_coefficient" in p and "line 2" in p for p in problems), problems
    assert _problems(tmp_path, "J3,0.6,0\nJ2,0.8,0.25\nJ1,0.9,0.15\n") == []


def test_a_row_with_a_cell_past_the_third_is_refused(tmp_path):
    """Property: a row has exactly the three cells the header names.

    Class: a fourth cell on the middle row, against the same rows without it. Why: an extra cell means a
    shifted or mis-edited row, and reading its first three cells would assign wrong values. What kills it:
    ignoring surplus cells. Second correct implementation: refusing a row whose length differs from the
    header's; the assertions read the problem text.
    """
    problems = _problems(tmp_path, "J1,0.6,0.35\nJ2,0.8,0.25,9\nJ3,0.9,0.15\n")
    assert any("line 3" in p and "past" in p for p in problems), problems
    assert _problems(tmp_path, "J1,0.6,0.35\nJ2,0.8,0.25\nJ3,0.9,0.15\n") == []


@pytest.mark.parametrize(
    ("violating", "satisfying"),
    [
        ("J1,0.6,0.35\nJ2,0.8,0.25\nj1,0.9,0.15\nJ3,0.7,0.2\n", "J1,0.6,0.35\nJ2,0.8,0.25\nJ3,0.9,0.15\n"),
        ("J1,0.6,0.35\nJ2,0.8,0.25\nJ3,0.9,0.15\nJ3,0.7,0.2\n", "j1,0.6,0.35\nj2,0.8,0.25\nj3,0.9,0.15\n"),
    ],
    ids=["case-folded repeat, mid table", "exact repeat, last row"],
)
def test_a_node_named_twice_after_the_case_fold_is_refused(tmp_path, violating, satisfying):
    """Property: no node name repeats once SWMM's case fold is applied.

    Class: ``J1`` and ``j1`` on different rows, and an exact repeat on the last row, against a table that
    names each node once in lower case. Why: SWMM compares names after the fold, so the two spellings are one
    node and the table would assign it two pairs of values. What kills it: comparing the spellings exactly.
    Second correct implementation: folding with ``str.upper``; the assertions read the problem text.
    """
    problems = _problems(tmp_path, violating, fill=True)
    assert any("more than once" in p for p in problems), problems
    assert _problems(tmp_path, satisfying) == []


@pytest.mark.parametrize("fill", [False, True])
def test_a_row_naming_no_coupling_node_is_refused_whatever_the_toggle(tmp_path, fill):
    """Property: a row that names no coupling node is refused with the toggle on and with it off.

    Class: ``J9``, a misspelling of ``J3``, in a table that otherwise covers the nodes, under both toggle
    values, against the same table with the name spelled in another case. Why: with the toggle on, a
    misspelt name would leave its intended node at the constants without a word. What kills it: skipping the
    rule when the toggle is on. Second correct implementation: judging unknown names before the missing rows;
    the assertions read the problem text and that J9 is named.
    """
    problems = _problems(tmp_path, "J1,0.6,0.35\nJ2,0.8,0.25\nJ9,0.9,0.15\n", fill=fill)
    assert any("J9" in p and "naming no coupling node" in p for p in problems), problems
    assert _problems(tmp_path, "j1,0.6,0.35\nj2,0.8,0.25\nj3,0.9,0.15\n", fill=fill) == []


def test_a_coupling_node_without_a_row_is_refused_naming_the_toggle(tmp_path):
    """Property: with the toggle off, a coupling node with no row refuses and the error names the toggle.

    Class: a table missing the middle node, then the first, with the toggle off, against the same tables with
    the toggle on. Why: a table that does not cover the nodes must not be taken for one that does. What kills
    it: filling the constants without the toggle, or an error that does not tell the user how to opt in.
    Second correct implementation: raising on the first missing node; the assertions read the problem text.
    """
    problems = _problems(tmp_path, "J1,0.6,0.35\nJ3,0.9,0.15\n")
    assert len(problems) == 1
    text = problems[0]
    assert "2 of the 3 coupling nodes" in text and "1 coupling node(s) have no row: J2" in text
    assert "manhole_table_fill_with_constants: true" in text
    assert "1.2 m" in text and "(0.1)" in text
    assert "no row: J1." in _problems(tmp_path, "J2,0.8,0.25\nJ3,0.9,0.15\n")[0]
    assert _problems(tmp_path, "J1,0.6,0.35\nJ3,0.9,0.15\n", fill=True) == []
    assert _problems(tmp_path, "J2,0.8,0.25\nJ3,0.9,0.15\n", fill=True) == []


def test_the_missing_row_error_names_ten_nodes_and_counts_the_rest(tmp_path):
    """Property: the error names at most ten coupling nodes without a row and counts the others.

    Class: twelve coupling nodes and a table covering none of them but one, against a table covering all
    twelve. Why: a table for a network of thousands of nodes must not print thousands of names. What kills
    it: listing every name, or truncating without the count. Second correct implementation: ten names
    chosen from the end; the assertions read the number of names and the count.
    """
    nodes = [f"J{i}" for i in range(12)]
    problems = _problems(tmp_path, "J0,0.6,0.35\n", nodes=nodes)
    assert len(problems) == 1
    named = problems[0].split("no row: ")[1].split(". Every")[0]
    assert named == "J1, J2, J3, J4, J5, J6, J7, J8, J9, J10, and 1 more"
    assert _problems(tmp_path, "".join(f"{n},0.6,0.35\n" for n in nodes), nodes=nodes) == []


@pytest.mark.parametrize(
    ("header", "body", "message"),
    [
        ("node,manhole_diameter_m,manhole_loss_coefficient", _COVERING, "must start with the header"),
        ("node_id,manhole_loss_coefficient,manhole_diameter_m", _COVERING, "must start with the header"),
    ],
    ids=["first column under another name", "swapped value columns"],
)
def test_a_header_other_than_the_documented_one_is_refused(tmp_path, header, body, message):
    """Property: the header is exactly ``node_id,manhole_diameter_m,manhole_loss_coefficient``.

    Class: a first column under another name and the two value columns in the other order, against the documented
    header after a blank line. Why: a swapped header would put every diameter in the loss column and every loss
    in the diameter column, and both are positive numbers, so nothing else would notice. What kills it:
    reading the rows by position without checking the header. Second correct implementation: a case-insensitive
    header match, if the assertion were widened; the assertions read the error.
    """
    from hhemt.scenario import read_manhole_table

    with pytest.raises(ConfigurationError, match=message) as excinfo:
        read_manhole_table(_table(tmp_path, body, header=header))
    assert excinfo.value.field == "manhole_table"
    assert len(read_manhole_table(_table(tmp_path, _COVERING, header="\n" + _HEADER, name="blank_first.csv"))) == 3


def test_an_unreadable_or_empty_table_is_refused_naming_the_file(tmp_path):
    """Property: a missing file and an empty file each refuse with the field and the file named.

    Class: a path that does not exist and a zero-byte file, against a file holding the header alone, which is
    a valid table with no rows. Why: an unreadable table must stop the run before a job is queued rather than
    surface as a solver error. What kills it: letting ``OSError`` escape, or reading an empty file as a table
    with no rows. Second correct implementation: checking existence first; the assertions read the field and
    the message.
    """
    from hhemt.scenario import read_manhole_table

    missing = tmp_path / "absent.csv"
    with pytest.raises(ConfigurationError, match="cannot be read") as excinfo:
        read_manhole_table(missing)
    assert excinfo.value.field == "manhole_table" and str(missing) in str(excinfo.value)
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    with pytest.raises(ConfigurationError, match="is empty"):
        read_manhole_table(empty)
    assert read_manhole_table(_table(tmp_path, "", name="header_only.csv")) == []


def test_the_written_file_assigns_each_value_by_position_and_fills_the_rest(tmp_path):
    """Property: ``manhole_table.txt`` lists every coupling node in node order with its own two values.

    Class: a table naming two of three nodes in the reverse of node order, the toggle on. Why: the solver
    reads the file by name and column, so the node order and the column order of each line are the whole
    assignment. What kills it: writing the rows in table order, swapping the two columns, or leaving the
    unlisted node out. Second correct implementation: writing the constants first and overwriting; the
    assertion reads the whole text.
    """
    from hhemt.scenario import manhole_table_text, read_manhole_table

    rows = read_manhole_table(_table(tmp_path, "j3,0.6,0.35\nJ1,0.8,0.25\n"))
    text = manhole_table_text(rows, _NODES, _DIAMETER_CONSTANT, _LOSS_CONSTANT)
    assert text == "J1 0.8 0.25\nJ2 1.2 0.1\nJ3 0.6 0.35\n"


def test_a_table_equal_to_the_constants_writes_the_constants_as_the_solver_reads_them(tmp_path):
    """Property: each value is written from one float, so a constant-equal table reproduces the constants.

    Class: a table whose every row equals the two constants, against the constants written for rows the table
    lacks. Why: the solver parses the text with ``strtod``, so equal text is equal doubles, and a
    constant-equal table must run exactly as no table does. What kills it: formatting with a fixed number of
    digits. Second correct implementation: ``float.hex``, if the solver read it; the assertions read the text.
    """
    from hhemt.scenario import manhole_table_text, read_manhole_table

    rows = read_manhole_table(_table(tmp_path, "J1,1.2,0.1\nJ2,1.2,0.1\nJ3,1.2,0.1\n"))
    from_rows = manhole_table_text(rows, _NODES, _DIAMETER_CONSTANT, _LOSS_CONSTANT)
    from_fill = manhole_table_text([], _NODES, _DIAMETER_CONSTANT, _LOSS_CONSTANT)
    assert from_rows == from_fill == "J1 1.2 0.1\nJ2 1.2 0.1\nJ3 1.2 0.1\n"


def test_the_refusal_names_the_field_the_file_and_every_defect(tmp_path):
    """Property: the one raising function refuses with ``field="manhole_table"`` naming the file and each problem.

    Class: a table with an empty cell and an unknown node together, against a covering table. Why: both
    callers raise through it, so their errors cannot differ. What kills it: stopping at the first defect.
    Second correct implementation: one exception per defect, if the assertions were widened; they read the
    field, the file and both defects.
    """
    from hhemt.scenario import require_valid_manhole_table

    path = _table(tmp_path, "J1,0.6,\nJ9,0.8,0.25\nJ2,0.9,0.15\nJ3,0.5,0.05\n")
    with pytest.raises(ConfigurationError) as excinfo:
        require_valid_manhole_table(path, _NODES, False, _context(tmp_path))
    assert excinfo.value.field == "manhole_table"
    assert excinfo.value.config_path == path
    message = str(excinfo.value)
    assert "empty manhole_loss_coefficient" in message and "J9" in message
    rows = require_valid_manhole_table(_table(tmp_path, _COVERING, name="ok.csv"), _NODES, False, _context(tmp_path))
    assert [r.node_id for r in rows] == _NODES


# ---------------------------------------------------------------------------
# The two moments: run entry and scenario preparation
# ---------------------------------------------------------------------------


def _inp(tmp_path: Path, *, junctions: str = _JUNCTIONS, coordinates: str = _COORDINATES, name="model") -> Path:
    path = tmp_path / f"{name}.inp"
    path.write_text(
        _INP.format(junctions=junctions, inflows='J1 FLOW "" FLOW 1 1 0', coordinates=coordinates).strip() + "\n"
    )
    return path


def _system(inp: Path, *, mode="all_junctions", coupled=True):
    return SimpleNamespace(
        cfg_system=SimpleNamespace(
            toggle_tritonswmm_model=coupled,
            toggle_swmm_model=not coupled,
            SWMM_model=inp,
            swmm_outfalls_to_couple=[],
            swmm_coupling_mode=mode,
            swmm_coupling_nodes_file=None,
        ),
        sys_paths=SimpleNamespace(dem_processed=inp.parent / "processed.dem"),
    )


def _cfg_analysis(table=None, *, fill=False, one_per_cell=False, sensitivity=False):
    return SimpleNamespace(
        toggle_sensitivity_analysis=sensitivity,
        one_coupled_inflow_node_per_dem_cell=one_per_cell,
        manhole_table=table,
        manhole_table_fill_with_constants=fill,
        manhole_diameter=_DIAMETER_CONSTANT,
        manhole_loss_coefficient=_LOSS_CONSTANT,
    )


def _analysis(system, cfg_analysis, *, members=None):
    analysis = SimpleNamespace(cfg_analysis=cfg_analysis, _system=system, analysis_config_yaml=None)
    if members is not None:
        analysis.sensitivity = SimpleNamespace(members=members)
    return analysis


def _refuse(analysis) -> None:
    from hhemt.analysis import TRITONSWMM_analysis

    TRITONSWMM_analysis._refuse_manhole_table_mismatch(analysis)


def test_run_entry_refuses_a_table_with_a_coupling_node_missing(tmp_path):
    """Property: run entry judges the table against the candidate junctions of the coupled system.

    Class: a table naming two of the three junctions of an ``all_junctions`` model, toggle off, against the
    same table with the toggle on and a table naming all three in another order. Why: the refusal must come
    before any job is submitted. What kills it: not judging at run entry, or judging a different node set. Second
    correct implementation: reading the candidates through swmmio; the assertions read the exception and
    its text.
    """
    system = _system(_inp(tmp_path))
    partial = _table(tmp_path, "J3,0.9,0.15\nJ1,0.6,0.35\n", name="partial.csv")
    with pytest.raises(ConfigurationError, match="manhole_table_fill_with_constants: true") as excinfo:
        _refuse(_analysis(system, _cfg_analysis(partial)))
    assert excinfo.value.field == "manhole_table" and "J2" in str(excinfo.value)
    _refuse(_analysis(system, _cfg_analysis(partial, fill=True)))
    _refuse(
        _analysis(system, _cfg_analysis(_table(tmp_path, "J2,0.8,0.25\nJ3,0.9,0.15\nJ1,0.6,0.35\n", name="all.csv")))
    )


def test_run_entry_refuses_a_row_naming_no_candidate_with_the_toggle_on(tmp_path):
    """Property: at run entry a row that names no candidate junction is refused with the toggle on.

    Class: a ``J9`` row beside covering rows, the toggle on, against the same rows with ``J3`` spelled in
    lower case. Why: the misspelt name would leave its node at the constants. What kills it: applying the
    rule only when the toggle is off. Second correct implementation: judging unknown names first; the
    assertions read the exception.
    """
    system = _system(_inp(tmp_path))
    bad = _table(tmp_path, "J1,0.6,0.35\nJ2,0.8,0.25\nJ9,0.9,0.15\n", name="bad.csv")
    with pytest.raises(ConfigurationError, match="J9"):
        _refuse(_analysis(system, _cfg_analysis(bad, fill=True)))
    good = _table(tmp_path, "j3,0.9,0.15\nJ2,0.8,0.25\n", name="good.csv")
    _refuse(_analysis(system, _cfg_analysis(good, fill=True)))


def test_run_entry_skips_a_template_that_names_a_placeholder_node(tmp_path):
    """Property: a template whose candidate names carry a ``$`` is left to preparation.

    Class: a template whose junction ``J$3`` is a placeholder, with a table missing a node, against the same
    table and a template with no placeholder. Why: a placeholder is a candidate whose filled name is unknown,
    so judging it would refuse a run whose filled model the table fits. What kills it: judging the unfilled
    name. Second correct implementation: filtering placeholder names out before judging, which would instead
    refuse the table's row for the real node; the assertions read whether the check raises.
    """
    table = _table(tmp_path, "J1,0.6,0.35\nJ2,0.8,0.25\n")
    placeholder = _inp(
        tmp_path,
        junctions="J1 1 2 0 0 0\nJ2 1 2 0 0 0\nJ$3 1 2 0 0 0",
        coordinates="J1 5 5\nJ2 15 5\nJ$3 25 5\nO1 35 5",
        name="placeholder",
    )
    _refuse(_analysis(_system(placeholder), _cfg_analysis(table)))
    with pytest.raises(ConfigurationError, match="J3"):
        _refuse(_analysis(_system(_inp(tmp_path, name="plain")), _cfg_analysis(table)))


@pytest.mark.parametrize(
    ("body", "refused"),
    [
        ("J1,0.6,0.35\nJ9,0.8,0.25\n", True),
        ("J1,0.6,\n", True),
        ("J1,0.6,0.35\n", False),
    ],
    ids=["names no candidate", "form defect", "a missing row is left to preparation"],
)
def test_under_legacy_with_one_node_per_cell_run_entry_judges_only_what_the_superset_decides(tmp_path, body, refused):
    """Property: with the all-inflow-nodes coupling mode and one node per cell the candidates are a superset, so no
    missing row is judged.

    Class: a table that names no candidate, one whose form is wrong, and one that merely lacks a row, with the
    toggle off. Why: the exact written set needs the per-cell grouping that runs at preparation, so a
    candidate without a row may be one preparation drops. What kills it: judging the missing-row rule against
    the superset, which refuses a run preparation would accept, or skipping the rules the superset does
    decide. Second correct implementation: running the grouping at run entry; the assertions read whether
    the check raises.
    """
    system = _system(_inp(tmp_path), mode="legacy")
    table = _table(tmp_path, body)
    analysis = _analysis(system, _cfg_analysis(table, one_per_cell=True))
    if refused:
        with pytest.raises(ConfigurationError, match="manhole_table"):
            _refuse(analysis)
    else:
        _refuse(analysis)


def test_run_entry_judges_every_sensitivity_member_and_not_the_master(tmp_path):
    """Property: under a sensitivity analysis each member's own table is judged, though the master sets none.

    Class: a master carrying no table and two members, one overlaying a table with a node missing and one
    with a covering table, against a master with the sensitivity toggle off. Why: a matrix overlay sets
    ``manhole_table`` on a member only. What kills it: reading the master's ``cfg_analysis``. Second correct
    implementation: judging every unique (table, system) pair; the assertions read whether it raises and
    what it names.
    """
    system = _system(_inp(tmp_path))
    missing = _table(tmp_path, "J1,0.6,0.35\nJ3,0.9,0.15\n", name="missing.csv")
    covering = _table(tmp_path, _COVERING, name="covering.csv")
    members = {
        "0": SimpleNamespace(cfg_analysis=_cfg_analysis(covering), _system=system),
        "1": SimpleNamespace(cfg_analysis=_cfg_analysis(missing), _system=system),
    }
    master = _cfg_analysis(None, sensitivity=True)
    with pytest.raises(ConfigurationError, match="J2"):
        _refuse(_analysis(system, master, members=members))
    _refuse(_analysis(system, master, members={"0": members["0"]}))
    _refuse(_analysis(system, _cfg_analysis(None, sensitivity=False)))


def test_run_entry_ignores_an_uncoupled_system_and_an_absent_table(tmp_path):
    """Property: only a coupled system with a table is judged.

    Class: an uncoupled system whose table would refuse, and a coupled system with no table, against the
    coupled system with the same table. Why: the table is read only for a coupled system, so refusing it
    elsewhere would stop a run the table does not touch. What kills it: judging without the coupling
    toggle. Second correct implementation: checking the toggle before reading the table; the assertions read
    whether the check raises.
    """
    partial = _table(tmp_path, "J1,0.6,0.35\n")
    inp = _inp(tmp_path)
    _refuse(_analysis(_system(inp, coupled=False), _cfg_analysis(partial)))
    _refuse(_analysis(_system(inp), _cfg_analysis(None)))
    with pytest.raises(ConfigurationError, match="manhole_table"):
        _refuse(_analysis(_system(inp), _cfg_analysis(partial)))


class _AnalysisStandIn(SimpleNamespace):
    """A stand-in analysis whose every refusal but the manhole table's is a no-op."""

    def __getattr__(self, name):
        from hhemt.analysis import TRITONSWMM_analysis

        if name.startswith("_refuse_") and hasattr(TRITONSWMM_analysis, name):
            return lambda *_a, **_k: None
        raise AttributeError(name)


@pytest.fixture
def _stand_in(tmp_path):
    """An analysis stand-in whose manhole-table refusal is the real one against a table missing a node."""
    from types import MethodType

    from hhemt.analysis import TRITONSWMM_analysis

    system = _system(_inp(tmp_path))
    analysis = _AnalysisStandIn(
        cfg_analysis=_cfg_analysis(_table(tmp_path, "J1,0.6,0.35\nJ3,0.9,0.15\n")),
        _system=system,
        analysis_config_yaml=None,
    )
    analysis._refuse_manhole_table_mismatch = MethodType(TRITONSWMM_analysis._refuse_manhole_table_mismatch, analysis)
    return analysis


@pytest.mark.parametrize("entry", ["run", "submit_workflow"])
def test_both_run_entries_call_the_manhole_table_check(_stand_in, entry):
    """Property: ``run`` and ``submit_workflow`` each raise the manhole-table refusal before anything else.

    Class: the two public entries over a stand-in whose other refusals are no-ops and whose manhole table
    misses a node. Why: an implementation that checks at one entry only would admit the same run through the
    other. What kills it: omitting the call from either entry. Second correct implementation: both entries
    calling one shared preflight that includes the check; the assertions read the exception and its field.
    """
    from hhemt.analysis import TRITONSWMM_analysis

    with pytest.raises(ConfigurationError) as excinfo:
        getattr(TRITONSWMM_analysis, entry)(_stand_in)
    assert excinfo.value.field == "manhole_table"


def _generator(tmp_path: Path, cfg_analysis, system):
    from hhemt.scenario_inputs import ScenarioInputGenerator

    gen = ScenarioInputGenerator.__new__(ScenarioInputGenerator)
    gen.scenario = SimpleNamespace(scen_paths=SimpleNamespace(swmm_hydraulics_inp=tmp_path / "hydraulics.inp"))
    gen.cfg_analysis = cfg_analysis
    gen.system = system
    return gen


def test_preparation_writes_the_complete_file_beside_the_coupling_list(tmp_path):
    """Property: with a table and the toggle on, preparation writes ``manhole_table.txt`` beside the list.

    Class: a table naming two of three coupling nodes in reverse node order, the toggle on, against no table.
    Why: the file TRITON reads must be complete, and with no table nothing may be written. What kills it:
    writing the file without a table, writing only the table's rows, or writing it elsewhere. Second correct
    implementation: writing before the coupling list; the assertions read the file's path and bytes.
    """
    from hhemt.scenario import manhole_table_file_for

    inp = tmp_path / "hydraulics.inp"
    system = _system(inp)
    nodes = tuple(_NODES)
    table = _table(tmp_path, "J3,0.6,0.35\nJ1,0.8,0.25\n")
    gen = _generator(tmp_path, _cfg_analysis(table, fill=True), system)
    gen._write_manhole_table(nodes, inp)
    target = manhole_table_file_for(inp)
    assert target == tmp_path / "manhole_table.txt"
    assert target.read_text() == "J1 0.8 0.25\nJ2 1.2 0.1\nJ3 0.6 0.35\n"
    target.unlink()
    _generator(tmp_path, _cfg_analysis(None), system)._write_manhole_table(nodes, inp)
    assert not target.exists()


@pytest.mark.parametrize("fill", [False, True])
def test_preparation_judges_the_exact_written_set(tmp_path, fill):
    """Property: preparation refuses a table against the nodes actually written, with the same rule.

    Class: a written set of ``J1`` and ``J2`` (``J3`` dropped by the per-cell grouping) and a table that
    names ``J3``, under both toggle values, against a table naming only the written nodes. Why: run entry
    judged a superset and could not see the drop, and a row for a dropped node would otherwise be ignored
    without a word. What kills it: judging the candidates again, or skipping the unknown-name rule when the
    toggle is on. Second correct implementation: a distinct preparation message, if the assertions were
    widened; they read the exception and the file.
    """
    inp = tmp_path / "hydraulics.inp"
    gen = _generator(tmp_path, _cfg_analysis(_table(tmp_path, _COVERING), fill=fill), _system(inp))
    with pytest.raises(ConfigurationError, match="J3"):
        gen._write_manhole_table(("J1", "J2"), inp)
    ok = _generator(
        tmp_path, _cfg_analysis(_table(tmp_path, "J2,0.8,0.25\nJ1,0.6,0.35\n", name="ok.csv")), _system(inp)
    )
    ok._write_manhole_table(("J1", "J2"), inp)
    assert (tmp_path / "manhole_table.txt").read_text() == "J1 0.6 0.35\nJ2 0.8 0.25\n"


# ---------------------------------------------------------------------------
# The cfg line and the build guard
# ---------------------------------------------------------------------------

_READS_BOTH = 'argsd("swmm_coupling_nodes", argmap, "");\nargsd("swmm_manhole_table", argmap, "");\n'
_READS_NODES_ONLY = 'argsd("swmm_coupling_nodes", argmap, "");\n'


def _clone(tmp_path: Path, text: str, name: str = "triton") -> Path:
    src = tmp_path / name / "src"
    src.mkdir(parents=True)
    (src / "config_utils.h").write_text(text)
    return tmp_path / name


class _AnyField:
    """A stand-in log whose every field reads False and accepts any value."""

    def __getattr__(self, name):
        return SimpleNamespace(get=lambda: False, set=lambda value: None)


def _scenario(tmp_path, monkeypatch, *, table, clone, environment="native", branch_key=None, recorded_sha=None):
    """A scenario whose coupled cfg generator runs on fakes; returns (scenario, cfg path)."""
    from hhemt import scenario as scenario_module
    from hhemt.scenario import TRITONSWMM_scenario

    (tmp_path / "TRITONSWMM.cfg").write_text("sim_start_time=0\ndem_filename=dem.dem\n")
    extbc = tmp_path / "extbc_loc.csv"
    extbc.write_text("x,y\n")
    options = SimpleNamespace(
        Value=SimpleNamespace(
            START_DATE="01/01/2020", START_TIME="00:00:00", END_DATE="01/01/2020", END_TIME="01:00:00"
        )
    )
    monkeypatch.setattr(
        scenario_module.swmmio, "Model", lambda path: SimpleNamespace(inp=SimpleNamespace(options=options))
    )
    cfg_analysis = SimpleNamespace(
        manhole_diameter=_DIAMETER_CONSTANT,
        manhole_loss_coefficient=_LOSS_CONSTANT,
        manhole_table=table,
        manhole_table_fill_with_constants=False,
        TRITON_raw_output_type="bin",
        hydraulic_timestep_s=1.0,
        TRITON_courant_number=0.5,
        TRITON_hextra_m=0.001,
        TRITON_fixed_timestep=False,
        TRITON_domain_decomposition="static",
        TRITON_domain_decomposition_check_factor=None,
        TRITON_progress_print_iterations=10000,
        TRITON_reporting_timestep_s=60,
        open_boundaries=1,
        swmm_snapshot_disable=False,
        swmm_snapshot_keep_all=False,
        swmm_node_rim_from_dem=False,
        execution_environment=environment,
        one_coupled_inflow_node_per_dem_cell=False,
    )
    cfg_system = SimpleNamespace(
        toggle_use_constant_mannings=True,
        constant_mannings=0.05,
        TRITONSWMM_software_directory=clone,
        TRITONSWMM_branch_key=branch_key,
    )
    scen = TRITONSWMM_scenario.__new__(TRITONSWMM_scenario)
    scen._analysis = SimpleNamespace(cfg_analysis=cfg_analysis, cfg_hpc_system=SimpleNamespace(gpu_aware_mpi=False))
    scen._system = SimpleNamespace(
        cfg_system=cfg_system,
        sys_paths=SimpleNamespace(dem_processed=tmp_path / "dem.dem", mannings_processed=tmp_path / "man.dem"),
        log=SimpleNamespace(triton_head_sha=SimpleNamespace(get=lambda: recorded_sha)),
    )
    cfg = tmp_path / "sim" / "TRITONSWMM.cfg"
    scen.scen_paths = SimpleNamespace(
        swmm_hydraulics_inp=tmp_path / "hydraulics.inp",
        hyg_timeseries=tmp_path / "tseries.hyg",
        hyg_locs=tmp_path / "loc.txt",
        extbc_loc=extbc,
        triton_swmm_cfg=cfg,
    )
    scen.sim_id_str = "evt"
    scen.log = _AnyField()
    scen._swmm_inp_for_sim_duration = lambda: tmp_path / "hydraulics.inp"
    scen._n_hydrograph_sources = lambda: 0
    return scen, cfg


def test_a_table_adds_exactly_one_cfg_line_and_nothing_else(tmp_path, monkeypatch):
    """Property: a table appends the one line ``swmm_manhole_table="{path}"`` and changes no other byte.

    Class: the coupled cfg generated for a build that reads the key, with a table set and with none. Why:
    with no table the cfg bytes are the bytes of a run that never heard of the key, so an arm without a table
    keeps its results. What kills it: writing the key without a table, writing it with spaces or more than once,
    or rewriting other lines. Second correct implementation: placing the line before the other appended keys,
    if the assertion were widened; it reads the whole text.
    """
    clone = _clone(tmp_path, _READS_BOTH)
    table = _table(tmp_path, _COVERING)
    without, cfg_without = _scenario(tmp_path, monkeypatch, table=None, clone=clone)
    without._generate_TRITON_SWMM_cfg()
    baseline = cfg_without.read_text()
    assert "swmm_manhole_table" not in baseline
    with_table, cfg_with = _scenario(tmp_path, monkeypatch, table=table, clone=clone)
    with_table._generate_TRITON_SWMM_cfg()
    expected_line = f'swmm_manhole_table="{tmp_path / "manhole_table.txt"}"\n'
    assert cfg_with.read_text() == baseline + expected_line


def test_a_template_line_for_the_key_is_replaced_by_the_toolkit_line(tmp_path):
    """Property: an active template line for the key is removed and the toolkit's line is the only one.

    Class: a cfg text carrying the key as an active line, as a comment, and indented, against a text without
    it. Why: a template value must never replace the table the toolkit wrote, and a comment is not an active
    line. What kills it: appending without removing, or removing a comment. Second correct implementation:
    replacing the line in place; the assertions read the set of lines.
    """
    from hhemt.scenario import set_swmm_manhole_table

    text = 'a=1\nswmm_manhole_table="old"\n  swmm_manhole_table = "older"\n# swmm_manhole_table="note"\nb=2\n'
    lines = set_swmm_manhole_table(text, "new.txt").splitlines()
    assert lines == ["a=1", '# swmm_manhole_table="note"', "b=2", 'swmm_manhole_table="new.txt"']
    assert set_swmm_manhole_table("a=1\n", "new.txt") == 'a=1\nswmm_manhole_table="new.txt"\n'
    assert set_swmm_manhole_table("", "new.txt") == 'swmm_manhole_table="new.txt"\n'


@pytest.mark.parametrize(
    ("reads", "environment", "passes"),
    [
        (_READS_BOTH, "native", True),
        (_READS_NODES_ONLY, "native", False),
        (_READS_BOTH, "container", False),
    ],
    ids=["native clone that reads the key", "native clone that does not", "container, even with a reading source"],
)
def test_preparation_refuses_a_build_that_would_ignore_the_table(tmp_path, monkeypatch, reads, environment, passes):
    """Property: the cfg is written only when the TRITON build reads ``swmm_manhole_table``.

    Class: a native clone whose ``config_utils.h`` carries the read, one that reads other keys only, and a
    container run whose recorded and configured commits are equal and listed for the other keys. Why: a build
    that ignores the key runs every node at the constants while the configuration names a table. What kills
    it: trusting the configured commit, treating any clone as capable, or admitting a container. Second
    correct implementation: testing ancestry once a first reading commit exists, if the guard were widened;
    the assertions read whether preparation raises and what it names.
    """
    listed = "a146fb6ed7c22790fb9a15ff854c446db4519ad2"
    scen, cfg = _scenario(
        tmp_path,
        monkeypatch,
        table=_table(tmp_path, _COVERING),
        clone=_clone(tmp_path, reads),
        environment=environment,
        branch_key=listed,
        recorded_sha=listed,
    )
    if passes:
        scen._generate_TRITON_SWMM_cfg()
        assert sum(line.startswith("swmm_manhole_table=") for line in cfg.read_text().splitlines()) == 1
    else:
        with pytest.raises(ConfigurationError, match="manhole_table") as excinfo:
            scen._generate_TRITON_SWMM_cfg()
        assert excinfo.value.field == "manhole_table"
        assert not cfg.exists() or "swmm_manhole_table" not in cfg.read_text()


def test_the_rim_guard_keeps_its_behaviour_through_the_shared_body(tmp_path):
    """Property: the DEM-rim guard, now a binding of the shared body, still refuses and passes as before.

    Class: a native clone without the rim read and a native clone with it, against the manhole guard on the
    same clones. Why: the two keys share one function body, and a regression in the shared body must show in
    the older guard as well. What kills it: a shared body that hard-codes either key. Second correct
    implementation: two bodies; the assertions read each guard's field in the error.
    """
    from hhemt.scenario import require_dem_rim_capable_triton, require_manhole_table_capable_triton

    rim = 'argsd("swmm_node_rim_from_dem", argmap, "0");\n'
    kwargs = {"execution_environment": "native", "branch_key": None, "recorded_sha": None}
    with pytest.raises(ConfigurationError) as excinfo:
        require_dem_rim_capable_triton(software_directory=_clone(tmp_path, _READS_NODES_ONLY, "a"), **kwargs)
    assert excinfo.value.field == "swmm_node_rim_from_dem"
    require_dem_rim_capable_triton(software_directory=_clone(tmp_path, rim, "b"), **kwargs)
    with pytest.raises(ConfigurationError) as excinfo:
        require_manhole_table_capable_triton(software_directory=_clone(tmp_path, rim, "c"), **kwargs)
    assert excinfo.value.field == "manhole_table"
    require_manhole_table_capable_triton(software_directory=_clone(tmp_path, _READS_BOTH, "d"), **kwargs)


# ---------------------------------------------------------------------------
# The fields
# ---------------------------------------------------------------------------


def test_the_toggle_without_a_table_is_refused_by_the_required_when_declaration():
    """Property: ``manhole_table_fill_with_constants`` without ``manhole_table`` is refused by the one mechanism.

    Class: the declaration read off the field, and the config model's before-validator over a mapping with the
    toggle on and the table absent, against the same mapping with a table and the toggle off. Why: the
    requirement is declared once, on the field, so the Reproduction Guide renders what is enforced and no
    hand-written validator can drift from it. What kills it: a second validator, or a declaration naming a
    different trigger. Second correct implementation: the same declaration with the roles reversed; the
    assertions read the declaration and the refusal text.
    """
    from hhemt.config.analysis import analysis_config
    from hhemt.config.base import declared, when

    def _enforce(values):
        """The model's before-validator, called on a bare mapping."""
        return analysis_config._enforce_required_when(values)  # ty: ignore[call-non-callable]

    field = analysis_config.model_fields["manhole_table"]
    assert declared(field, "required_when") == [when("manhole_table_fill_with_constants", True)]
    assert analysis_config.model_fields["manhole_table_fill_with_constants"].default is False
    with pytest.raises(ValueError, match="manhole_table is required when manhole_table_fill_with_constants is True"):
        _enforce({"manhole_table_fill_with_constants": True})
    _enforce({"manhole_table": "t.csv", "manhole_table_fill_with_constants": False})
    _enforce({"manhole_table_fill_with_constants": False})


def test_both_fields_are_classified_experiment_and_the_table_is_a_bundle_input():
    """Property: the fields are ``experiment`` in the taxonomy and the table has a bundle policy and a catalog row.

    Class: the two new fields, read from the three registries that must name a new field. Why: a table is an
    input the results depend on, so a bundle carries it, and an unclassified field fails the totality tests.
    What kills it: leaving any registry without the field. Second correct implementation: a distinct
    taxonomy class, if the assertions were widened; they read the three entries.
    """
    from hhemt.bundle._path_policy import _EXCLUDABLE_CATALOG, _PATH_FIELD_POLICY, PathPolicy
    from hhemt.config.reprex_taxonomy import all_field_bucket

    assert _PATH_FIELD_POLICY["manhole_table"] is PathPolicy.BUNDLE_RELATIVE_OR_NONE
    assert _EXCLUDABLE_CATALOG["manhole_table"].excludable is True
    assert "per-node manhole values" in _EXCLUDABLE_CATALOG["manhole_table"].reproducibility_cost
    assert all_field_bucket("manhole_table") == "experiment"
    assert all_field_bucket("manhole_table_fill_with_constants") == "experiment"
