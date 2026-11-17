"""The per-node manhole values on the coupled node store.

This file guards the two node variables ``manhole_diameter`` and ``manhole_loss_coefficient`` that
``TRITONSWMM_sim_post_processing._attach_coupling_outputs`` adds from a scenario's ``manhole_table.txt``, and the
CF attributes and provenance rows of those two names in ``cf_conventions``. The module whose behaviour it guards is
the coupled node store ingestion in ``process_simulation``; it is a one-time test, promoted into that module's tests
or deleted at the landing.

Every fixture row gives a diameter and a loss coefficient that differ from each other and from the two constants
a fill would write, and the expected numbers are written out here and never read back from the code under test,
so a swapped column or a constant that leaks into a table row fails.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hhemt.cf_conventions import apply_cf_attributes, quantity_provenance
from hhemt.exceptions import ProcessingError
from hhemt.process_simulation import (
    MANHOLE_TABLE_NODE_VARIABLES,
    TRITONSWMM_sim_post_processing,
    read_written_manhole_table,
    summarize_swmm_simulation_results,
)

# Two constants a fill would write, distinct from each other and from every table value below.
FILL_DIAMETER = 1.2
FILL_LOSS = 0.1

# SWMM order is n1 n2 n3 n4; the coupling list is n3, n1, n4 (n2 is not coupled). The table rows differ
# from each other: 0.6 and 0.35 at n3, 2.5 and 0.9 at n1; n4 is a filled node holding the two constants.
COUPLING_ORDER = ("n3", "n1", "n4")
TABLE_TEXT = f"n3 0.6 0.35\nn1 2.5 0.9\nn4 {FILL_DIAMETER!r} {FILL_LOSS!r}\n"


def _node_dataset() -> xr.Dataset:
    """A node time-series dataset shaped like the coupled SWMM export: four nodes, three 120 s report times."""
    times = pd.to_datetime("2020-01-01") + pd.to_timedelta([120, 240, 360], unit="s")
    return xr.Dataset(
        {"depth": (("date_time", "node_id"), np.zeros((3, 4)))},
        coords={"date_time": times, "node_id": ["n1", "n2", "n3", "n4"]},
    )


def _scenario(tmp_path: Path, *, table_set: bool, table_text: str | None = TABLE_TEXT, names=COUPLING_ORDER):
    """A post-processing object holding only what ``_attach_coupling_outputs`` reads, and its input file."""
    swmm_dir = tmp_path / "out_tritonswmm" / "swmm"
    swmm_dir.mkdir(parents=True)
    rpt = swmm_dir / "hydraulics.rpt"
    rpt.write_text("  Starting Date ............ 01/01/2020 00:00:00\n")
    (swmm_dir / "coupling_nodes.txt").write_text("".join(f"{n}\n" for n in names))
    if table_text is not None:
        (swmm_dir / "manhole_table.txt").write_text(table_text)
    proc = TRITONSWMM_sim_post_processing.__new__(TRITONSWMM_sim_post_processing)
    proc.scen_paths = SimpleNamespace(out_tritonswmm=tmp_path / "out_tritonswmm", swmm_hydraulics_rpt=rpt)
    cfg = SimpleNamespace(toggle_coupling_exchange_flow=False)
    if table_set:
        cfg.manhole_table = tmp_path / "table.csv"
    proc._analysis = SimpleNamespace(cfg_analysis=cfg)
    return proc, swmm_dir / "hydraulics.inp"


def test_a_table_run_yields_the_values_at_the_listed_nodes_and_nan_elsewhere(tmp_path):
    """Property: with a table set, each coupling node holds the diameter and loss coefficient of its own line of
    ``manhole_table.txt`` in the right variable, the filled node holds the two constants, and the uncoupled node is NaN.

    Input class: a coupling list whose order (n3, n1, n4) differs from the dataset's node order (n1, n2, n3, n4)
    and whose rows carry unequal values. Plausible wrong implementations: assigning values by dataset order or
    sorted order (n3 would read n1's row); storing the loss coefficient in the diameter variable (swapped
    columns); filling the uncoupled node with 0 (a node with no manhole reads as a measured zero).
    Why required: the report reads these by node name; a swapped pair has the right shape and units.
    Killed by: sorting the names before computing slots, swapping the two `zip` columns, or `np.zeros` for the
    fill. Passes under a second correct implementation that builds each variable by `reindex` on `node_id`.
    """
    proc, inp = _scenario(tmp_path, table_set=True)
    out = TRITONSWMM_sim_post_processing._attach_coupling_outputs(proc, _node_dataset(), inp)
    diameter = out["manhole_diameter"]
    loss = out["manhole_loss_coefficient"]
    assert diameter.dims == ("node_id",) == loss.dims
    assert diameter.sel(node_id="n3").item() == 0.6
    assert diameter.sel(node_id="n1").item() == 2.5
    assert diameter.sel(node_id="n4").item() == FILL_DIAMETER
    assert loss.sel(node_id="n3").item() == 0.35
    assert loss.sel(node_id="n1").item() == 0.9
    assert loss.sel(node_id="n4").item() == FILL_LOSS
    assert np.isnan(diameter.sel(node_id="n2").item())
    assert np.isnan(loss.sel(node_id="n2").item())


def test_a_run_with_no_table_adds_no_variable_even_when_a_stray_file_sits_beside_the_list(tmp_path):
    """Property: with ``manhole_table`` unset the node dataset gains the coupling flag and nothing about manholes,
    so a no-table store equals the one written before the table existed.

    Input class: a coupled scenario with no table, with and without a leftover ``manhole_table.txt`` beside
    ``coupling_nodes.txt``. Plausible wrong implementation: deciding by the file's presence, which would
    add a variable to a run whose configuration asked for none. Must-fire control: the identical scenario with
    the table set carries both variables, so the absence is the setting's doing.
    Killed by: replacing the config test with `table_file.is_file()`. Passes under a second correct implementation
    that attaches the variables in a separate method called only when the table is set.
    """
    for label, text in (("bare", None), ("stray", TABLE_TEXT)):
        proc, inp = _scenario(tmp_path / label, table_set=False, table_text=text)
        out = TRITONSWMM_sim_post_processing._attach_coupling_outputs(proc, _node_dataset(), inp)
        assert sorted(out.data_vars) == ["depth", "is_coupling_node"], label
    proc_on, inp_on = _scenario(tmp_path / "on", table_set=True)
    on = TRITONSWMM_sim_post_processing._attach_coupling_outputs(proc_on, _node_dataset(), inp_on)
    assert set(MANHOLE_TABLE_NODE_VARIABLES) <= set(on.data_vars)


def test_a_stand_in_config_without_the_field_is_read_as_no_table(tmp_path):
    """Property: an ``_analysis`` whose ``cfg_analysis`` has no ``manhole_table`` attribute behaves as no table.

    Input class: the minimal stand-in configuration the other coupled-output tests build. Plausible wrong
    implementation: a strict attribute read, which raises ``AttributeError`` on every such caller. Killed by
    replacing `getattr(..., None)` with direct access. Passes under a second correct implementation that
    reads the field through ``getattr(cfg, 'manhole_table', None)``.
    """
    proc, inp = _scenario(tmp_path, table_set=False)
    assert not hasattr(proc._analysis.cfg_analysis, "manhole_table")
    out = TRITONSWMM_sim_post_processing._attach_coupling_outputs(proc, _node_dataset(), inp)
    assert "manhole_diameter" not in out.data_vars


def test_a_table_set_with_its_file_absent_is_refused_and_the_same_run_with_the_file_passes(tmp_path):
    """Property: a set table with no ``manhole_table.txt`` raises ``ProcessingError`` naming the file.

    Input class: a scenario prepared before the table was written, or whose file was removed. Plausible wrong
    implementation: skipping the attach when the file is missing, which stores a node dataset that lacks the
    variables the configuration promises. Must-fire control: the same scenario with the file present returns
    both variables. Killed by: a `return ds_nodes` on a missing file. Passes under a second correct implementation
    that raises a subclass of `ProcessingError`.
    """
    missing, inp = _scenario(tmp_path / "missing", table_set=True, table_text=None)
    with pytest.raises(ProcessingError, match="manhole_table.txt"):
        TRITONSWMM_sim_post_processing._attach_coupling_outputs(missing, _node_dataset(), inp)
    present, inp2 = _scenario(tmp_path / "present", table_set=True)
    assert (
        "manhole_diameter"
        in TRITONSWMM_sim_post_processing._attach_coupling_outputs(present, _node_dataset(), inp2).data_vars
    )


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ("n3 0.6\nn1 2.5 0.9\nn4 1.2 0.1\n", "line 1"),  # one value only
        ("n3 0.6 0.35 7\nn1 2.5 0.9\nn4 1.2 0.1\n", "line 1"),  # a third number
        ("n3 0.6 abc\nn1 2.5 0.9\nn4 1.2 0.1\n", "line 1"),  # not a number
        ("n3 0.6 0.35\nn1 inf 0.9\nn4 1.2 0.1\n", "line 2"),  # not finite
        ("n3 0.6 nan\nn1 2.5 0.9\nn4 1.2 0.1\n", "line 1"),  # not finite
    ],
)
def test_a_line_that_is_not_a_name_and_two_finite_numbers_is_refused_at_its_line(tmp_path, text, fragment):
    """Property: every line must be a name and two finite numbers, and the refusal names the line.

    Input class: a line with one value, three values, a non-numeric value, an infinity and a NaN, at the first and
    the second line. Plausible wrong implementation: `split()` unpacked leniently (a third number ignored, a
    missing loss defaulted), which stores a value the solver did not read. Must-fire control: the complete file of
    the first test parses. Killed by: dropping the token-count test or the `isfinite` test. Passes under a second
    correct implementation that parses with a regular expression.
    """
    table = tmp_path / "manhole_table.txt"
    table.write_text(text)
    with pytest.raises(ProcessingError, match=fragment):
        read_written_manhole_table(table, list(COUPLING_ORDER))
    table.write_text(TABLE_TEXT)
    diameters, losses = read_written_manhole_table(table, list(COUPLING_ORDER))
    assert diameters.tolist() == [0.6, 2.5, FILL_DIAMETER]
    assert losses.tolist() == [0.35, 0.9, FILL_LOSS]


def test_a_repeated_a_foreign_and_a_missing_node_are_each_refused_and_blank_lines_are_not(tmp_path):
    """Property: a name repeated, a name that is no coupling node and a coupling node with no line are each refused,
    while blank lines and a different line order are accepted.

    Input class: the three membership violations, and a file that is the same set of lines in another order with
    blank lines between them (the satisfying input at a different position). Plausible wrong implementations:
    a dict that keeps the last repeat silently; ignoring a foreign name; reading the file by position against
    ``coupling_nodes.txt`` (the reordered file would then swap two nodes' values). Killed by: removing any of
    the three raises, or indexing the file's lines against the list. Passes under a second correct implementation
    that builds a pandas Series and compares index sets.
    """
    table = tmp_path / "manhole_table.txt"
    nodes = list(COUPLING_ORDER)
    table.write_text("n3 0.6 0.35\nn1 2.5 0.9\nn1 2.6 0.8\nn4 1.2 0.1\n")
    with pytest.raises(ProcessingError, match="repeats the node 'n1'"):
        read_written_manhole_table(table, nodes)
    table.write_text("n3 0.6 0.35\nn1 2.5 0.9\nn9 1.2 0.1\n")
    with pytest.raises(ProcessingError, match="'n9'"):
        read_written_manhole_table(table, nodes)
    table.write_text("n3 0.6 0.35\nn1 2.5 0.9\n")
    with pytest.raises(ProcessingError, match="no line"):
        read_written_manhole_table(table, nodes)
    table.write_text(f"\nn4 {FILL_DIAMETER!r} {FILL_LOSS!r}\n\nn1 2.5 0.9\nn3 0.6 0.35\n\n")
    diameters, losses = read_written_manhole_table(table, nodes)
    assert diameters.tolist() == [0.6, 2.5, FILL_DIAMETER]
    assert losses.tolist() == [0.35, 0.9, FILL_LOSS]


def test_the_two_variables_carry_units_provenance_and_survive_the_scenario_summary(tmp_path):
    """Property: the two names are stamped with meters and 1 by the CF map, each has a provenance descriptor,
    an unlisted name beside them gets no unit (control), and the per-scenario node summary passes both through
    unchanged because they carry no time dimension.

    Input class: a coupled node dataset holding the two variables with distinct values per node. Plausible wrong
    implementations: a missing map entry (no unit stored); a summary that reduces a static variable to a
    `_max`/`_last` pair. Killed by deleting a map entry or a provenance row. Passes under a second correct
    implementation that stores the units in the attach step.
    """
    proc, inp = _scenario(tmp_path, table_set=True)
    ds = TRITONSWMM_sim_post_processing._attach_coupling_outputs(proc, _node_dataset(), inp)
    ds["manhole_unlisted"] = ("node_id", np.zeros(4))
    apply_cf_attributes(ds, "tritonswmm_swmm_node")
    assert ds["manhole_diameter"].attrs["units"] == "m"
    assert ds["manhole_loss_coefficient"].attrs["units"] == "1"
    assert "units" not in ds["manhole_unlisted"].attrs
    for name in MANHOLE_TABLE_NODE_VARIABLES:
        assert quantity_provenance(name) is not None, name
    summary = summarize_swmm_simulation_results(ds, 0)
    for name in MANHOLE_TABLE_NODE_VARIABLES:
        assert summary[name].dims == ("event_iloc", "node_id"), name
        np.testing.assert_array_equal(summary[name].values[0], ds[name].values)
        assert f"{name}_max" not in summary.data_vars
