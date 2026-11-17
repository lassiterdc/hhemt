"""Reviewer probes of the per-node manhole table: what the builder's own tests leave a wrong build able to pass."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from hhemt.exceptions import ConfigurationError
from tests.test_manhole_table import (
    _COVERING,
    _DIAMETER_CONSTANT,
    _HEADER,
    _LOSS_CONSTANT,
    _analysis,
    _cfg_analysis,
    _generator,
    _refuse,
    _system,
    _table,
)

_LEGACY_INP = """[OPTIONS]
FLOW_UNITS CMS

[JUNCTIONS]
J1 1 2 0 0 0
J2 1 2 0 0 0
J3 1 2 0 0 0

[OUTFALLS]
O1 0 FREE NO

[CONDUITS]
C1 J1 O1 10 0.013 0 0 0 0

[XSECTIONS]
C1 CIRCULAR 1 0 0 0 1

[INFLOWS]
;;Node Constituent Time Series Type Mfactor Sfactor Baseline Pattern
;;---- ----------- ----------- ---- ------- ------- -------- -------
J1 FLOW "" FLOW 1 1 0
J2 FLOW "" FLOW 1 1 0
J3 FLOW "" FLOW 1 1 0

[COORDINATES]
J1 5 5
J2 15 5
J3 25 5
O1 35 5
"""


def _legacy_system(tmp_path: Path):
    inp = tmp_path / "all_inflow.inp"
    inp.write_text(_LEGACY_INP)
    return _system(inp, mode="legacy")


def test_legacy_candidates_are_the_three_inflow_nodes(tmp_path):
    """Control: the all-inflow-nodes fixture really has three candidates, so the arms below judge a non-trivial set."""
    from hhemt.scenario import coupling_candidate_junction_coordinates

    system = _legacy_system(tmp_path)
    names = [n for n, _x, _y in coupling_candidate_junction_coordinates(system.cfg_system, _cfg_analysis())]
    assert names == ["J1", "J2", "J3"]


def test_legacy_one_per_cell_run_entry_leaves_a_missing_row_to_preparation(tmp_path):
    """Superset relaxed: three candidates, a one-row table, one node per cell -> run entry does not refuse."""
    system = _legacy_system(tmp_path)
    _refuse(_analysis(system, _cfg_analysis(_table(tmp_path, "J1,0.6,0.35\n"), one_per_cell=True)))


def test_legacy_without_one_per_cell_run_entry_refuses_a_missing_row(tmp_path):
    """Same table, same system, one node per cell off -> the written set is the candidate set, so refuse."""
    system = _legacy_system(tmp_path)
    with pytest.raises(ConfigurationError, match="J2"):
        _refuse(_analysis(system, _cfg_analysis(_table(tmp_path, "J1,0.6,0.35\n"), one_per_cell=False)))


@pytest.mark.parametrize("relaxed_first", [True, False])
def test_two_members_sharing_a_system_and_a_table_are_each_judged(tmp_path, relaxed_first):
    """Two sensitivity members, one system object, one table, the toggle off: only the strict one refuses.

    The relaxed member (one node per cell under the all-inflow-nodes mode) must not shield the strict one, in either
    order.
    """
    system = _legacy_system(tmp_path)
    table = _table(tmp_path, "J1,0.6,0.35\n")
    relaxed = SimpleNamespace(cfg_analysis=_cfg_analysis(table, one_per_cell=True), _system=system)
    strict = SimpleNamespace(cfg_analysis=_cfg_analysis(table, one_per_cell=False), _system=system)
    members = {"0": relaxed, "1": strict} if relaxed_first else {"0": strict, "1": relaxed}
    with pytest.raises(ConfigurationError, match="J2"):
        _refuse(_analysis(system, _cfg_analysis(None, sensitivity=True), members=members))


def test_placeholder_template_with_the_filled_name_in_the_table_is_not_refused(tmp_path):
    """A table naming the FILLED node (J3) against a template whose junction is the placeholder J$3.

    Judged at run entry the row J3 names no candidate, so only the placeholder skip keeps the run.
    """
    from tests.test_manhole_table import _inp

    placeholder = _inp(
        tmp_path,
        junctions="J1 1 2 0 0 0\nJ2 1 2 0 0 0\nJ$3 1 2 0 0 0",
        coordinates="J1 5 5\nJ2 15 5\nJ$3 25 5\nO1 35 5",
        name="placeholder",
    )
    _refuse(_analysis(_system(placeholder), _cfg_analysis(_table(tmp_path, _COVERING))))


def test_preparation_refuses_a_written_node_with_no_row_when_the_toggle_is_off(tmp_path):
    """Preparation applies the missing-row rule: a node written to the coupling list needs a row."""
    inp = tmp_path / "hydraulics.inp"
    gen = _generator(tmp_path, _cfg_analysis(_table(tmp_path, "J1,0.6,0.35\n"), fill=False), _system(inp))
    with pytest.raises(ConfigurationError, match="J2") as excinfo:
        gen._write_manhole_table(("J1", "J2"), inp)
    assert "manhole_table_fill_with_constants: true" in str(excinfo.value)
    assert not (tmp_path / "manhole_table.txt").exists()


def test_the_written_file_spells_a_node_as_the_coupling_list_does(tmp_path):
    """A row spelled ``j2`` names the node ``J2``; the file TRITON reads carries the coupling list's spelling."""
    from hhemt.scenario import manhole_table_text, read_manhole_table

    rows = read_manhole_table(_table(tmp_path, "j2,0.6,0.35\n"))
    text = manhole_table_text(rows, ("J1", "J2"), _DIAMETER_CONSTANT, _LOSS_CONSTANT)
    assert text == "J1 1.2 0.1\nJ2 0.6 0.35\n"


def test_a_header_only_table_fills_every_node_only_with_the_toggle(tmp_path):
    from hhemt.scenario import ManholeTableContext, require_valid_manhole_table

    path = tmp_path / "empty.csv"
    path.write_text(_HEADER + "\n")
    ctx = ManholeTableContext(table=path, model="m", diameter_constant=1.2, loss_constant=0.1)
    assert require_valid_manhole_table(path, ["J1"], True, ctx) == []
    with pytest.raises(ConfigurationError, match="J1"):
        require_valid_manhole_table(path, ["J1"], False, ctx)
