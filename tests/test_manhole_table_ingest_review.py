"""Reviewer tests for the per-node manhole values on the coupled node store.

Module whose behaviour this file guards: the coupled node store ingestion in ``process_simulation``
(``TRITONSWMM_sim_post_processing._attach_coupling_outputs`` and ``read_written_manhole_table``). It is a one-time
test: promoted into that module's tests or deleted at the landing. Each fixture row gives a diameter and a loss
coefficient that differ from each other and from the two constants a fill would write.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hhemt.cf_conventions import _CF_VARIABLE_MAP, apply_cf_attributes, quantity_provenance
from hhemt.config.analysis import analysis_config
from hhemt.exceptions import ProcessingError
from hhemt.process_simulation import (
    MANHOLE_TABLE_NODE_VARIABLES,
    TRITONSWMM_sim_post_processing,
    read_written_manhole_table,
)

COUPLING_ORDER = ("n3", "n1", "n4")
TABLE_TEXT = "n3 0.6 0.35\nn1 2.5 0.9\nn4 1.2 0.1\n"


def _node_dataset() -> xr.Dataset:
    times = pd.to_datetime("2020-01-01") + pd.to_timedelta([120, 240, 360], unit="s")
    return xr.Dataset(
        {"depth": (("date_time", "node_id"), np.zeros((3, 4)))},
        coords={"date_time": times, "node_id": ["n1", "n2", "n3", "n4"]},
    )


def _run(tmp_path: Path, cfg) -> xr.Dataset:
    """The node store ``_attach_coupling_outputs`` returns for a scenario holding a complete manhole_table.txt."""
    swmm_dir = tmp_path / "out_tritonswmm" / "swmm"
    swmm_dir.mkdir(parents=True)
    rpt = swmm_dir / "hydraulics.rpt"
    rpt.write_text("  Starting Date ............ 01/01/2020 00:00:00\n")
    (swmm_dir / "coupling_nodes.txt").write_text("".join(f"{n}\n" for n in COUPLING_ORDER))
    (swmm_dir / "manhole_table.txt").write_text(TABLE_TEXT)
    proc = TRITONSWMM_sim_post_processing.__new__(TRITONSWMM_sim_post_processing)
    proc.scen_paths = SimpleNamespace(out_tritonswmm=tmp_path / "out_tritonswmm", swmm_hydraulics_rpt=rpt)
    proc._analysis = SimpleNamespace(cfg_analysis=cfg)
    return TRITONSWMM_sim_post_processing._attach_coupling_outputs(proc, _node_dataset(), swmm_dir / "hydraulics.inp")


def test_the_variables_follow_the_field_value_for_every_shape_of_a_config_that_holds_no_table(tmp_path):
    """Property: the two manhole variables are added iff the config's ``manhole_table`` is set to a path, and a
    config that holds the field with its default None adds neither, byte-for-byte the store of a config without the
    field.

    Input class: the shapes a config reaching the ingestion takes: a stand-in with the field absent, a stand-in
    with the field None, and the real ``analysis_config`` (declared field, default None). Reachability: the real
    model always carries the attribute (``analysis_config.model_fields``), and the file sits beside the list in
    all three, so only the field's value separates the shapes. Both-states anchor: the assertion turns on the
    field's value, which the code before and after the change both read.
    Must-fire control: a config whose field is a Path carries both variables.
    Killed by: replacing ``getattr(cfg, "manhole_table", None) is not None`` with ``hasattr(cfg, "manhole_table")``
    (a real no-table config would then demand a file that was never written, or add the variables for a run
    that asked for none). Passes under a second correct implementation that reads
    ``cfg.manhole_table`` through a helper returning ``None`` for a config without the field.
    """
    absent = _run(tmp_path / "absent", SimpleNamespace(toggle_coupling_exchange_flow=False))
    explicit_none = _run(tmp_path / "none", SimpleNamespace(toggle_coupling_exchange_flow=False, manhole_table=None))
    real_default = _run(tmp_path / "real", analysis_config.model_construct())
    for label, ds in (("absent", absent), ("none", explicit_none), ("real", real_default)):
        assert sorted(ds.data_vars) == ["depth", "is_coupling_node"], label
        assert ds.identical(absent), label
    on = _run(
        tmp_path / "on",
        analysis_config.model_construct(manhole_table=tmp_path / "table.csv"),
    )
    assert set(MANHOLE_TABLE_NODE_VARIABLES) <= set(on.data_vars)


def test_the_set_of_variables_a_table_adds_is_exactly_the_two_declared_and_each_is_stamped_with_its_unit(tmp_path):
    """Property: over every variable the table-set store holds that the no-table store does not, the set equals
    ``MANHOLE_TABLE_NODE_VARIABLES``, each name has a CF map row and a provenance row, and the unit stamped is
    ``m`` for the diameter and ``1`` for the loss coefficient (not swapped, not shared).

    Input class: the added-variable set over a coupled store; the two units are distinguishable because the two
    values of every row differ. Reachability: ``_attach_coupling_outputs`` adds the names and the CF map is the
    one stamp source. Both-states anchor: ``is_coupling_node`` is in both stores and is excluded from the set.
    Must-fire control: the same store without a table adds nothing, so an empty difference is not a broken matcher.
    Killed by: a third variable added beside the pair (a phantom the test allowlist would then hide), a name
    missing from the CF map, or the two units exchanged. Passes under a second correct implementation that builds
    the two variables in a loop over a different tuple order.
    """
    on = _run(tmp_path / "on", analysis_config.model_construct(manhole_table=tmp_path / "t.csv"))
    off = _run(tmp_path / "off", analysis_config.model_construct())
    assert set(on.data_vars) - set(off.data_vars) == set(MANHOLE_TABLE_NODE_VARIABLES)
    assert set(MANHOLE_TABLE_NODE_VARIABLES) == {"manhole_diameter", "manhole_loss_coefficient"}
    apply_cf_attributes(on, "tritonswmm_swmm_node")
    expected_unit = {"manhole_diameter": "m", "manhole_loss_coefficient": "1"}
    for name, unit in expected_unit.items():
        assert name in _CF_VARIABLE_MAP, name
        assert on[name].attrs["units"] == unit, name
        provenance = quantity_provenance(name)
        assert provenance is not None, name
    assert on["manhole_diameter"].attrs["units"] != on["manhole_loss_coefficient"].attrs["units"]


@pytest.mark.parametrize(
    ("text", "bad_line", "message"),
    [
        ("\n\nn3 0.6 0.35\n\nn1 2.5\nn4 1.2 0.1\n", 5, "line 5 "),  # one value, after three blank lines
        ("n3 0.6 0.35\nn1 2.5 0.9\n\n\n\nn4 1.2 abc", 6, "line 6 "),  # non-numeric, last line, no final newline
        ("n3 0.6 0.35\n\nn1 2.5 0.9\n\nn1 2.6 0.8\nn4 1.2 0.1\n", 5, "line 5 repeats"),  # a repeat behind blanks
        ("\nn3 0.6 0.35\nn1 2.5 0.9\n\nn9 1.2 0.1\nn4 1.2 0.1\n", 5, "line 5 names"),  # a foreign name behind blanks
        ("n3 0.6 0.35\r\n\r\nn1 -inf 0.9\r\nn4 1.2 0.1\r\n", 3, "line 3 "),  # CRLF endings, a non-finite value
    ],
)
def test_a_refusal_names_the_physical_line_of_the_offender_wherever_blank_lines_sit(tmp_path, text, bad_line, message):
    """Property: the refusal names the 1-based physical line of the offending line, counting blank lines, for
    every violation kind.

    Input class: a file with blank lines BEFORE the offender (the reader accepts them, so they are reachable), the
    offender on the last line with no newline, and CRLF endings; the coder's own refusal cases put the offender
    on line 1 or 2 with no blanks above it, a position at which a count over non-blank lines agrees with the
    physical count. Reachability: ``read_written_manhole_table`` skips blank lines and numbers by ``enumerate``.
    Both-states anchor: the same file with the offender removed parses.
    Killed by: numbering after the blank lines are filtered out, or numbering from zero. Passes under a second
    correct implementation that collects ``(number, line)`` pairs first and then validates each.
    """
    table = tmp_path / "manhole_table.txt"
    table.write_text(text)
    with pytest.raises(ProcessingError) as excinfo:
        read_written_manhole_table(table, list(COUPLING_ORDER))
    assert message in str(excinfo.value), (bad_line, str(excinfo.value))
    table.write_text(TABLE_TEXT)
    diameters, _losses = read_written_manhole_table(table, list(COUPLING_ORDER))
    assert diameters.tolist() == [0.6, 2.5, 1.2]
