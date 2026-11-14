"""No-solver test that the SWMM binary-output reader returns EVERY report period of a committed reference output.

``return_node_time_series_results_from_outfile`` once read each series through pyswmm's ``NodeSeries`` /
``LinkSeries``. pyswmm 2.1.0 (the version ``uv.lock`` pins) resolves their default end to the inclusive index
``num_periods - 1`` and passes it to the swmm-toolkit's END-EXCLUSIVE ``get_node_series``, so the final report
period was silently dropped (pyswmm 2.0.1 returns it whole). The reader now passes the window ``(0, NUM_PERIODS)``
to the toolkit itself. The first Rivanna run of the run branch found the shortfall (erratum E43) through a test that
compares the stored series with the report's printed rows; this file pins it with no solver and no report.

The reference is ``test_data/swmm_binary_reference/hydraulics.out``, one stock SWMM run (see ``make_reference.py``).
Test docstrings give Property, Class (REGRESSION: red under pyswmm 2.1.0 with the old reader), Why required, What
kills it, and a second correct implementation. Run it under both pyswmm versions with
``PYTHONPATH={a directory holding the other pyswmm}`` ahead of the environment's own.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from hhemt.swmm_output_parser import return_node_time_series_results_from_outfile

REFERENCE = Path(__file__).resolve().parent.parent / "test_data" / "swmm_binary_reference"
OUT = REFERENCE / "hydraulics.out"
RPT = REFERENCE / "hydraulics.rpt"

#: stored name -> toolkit attribute INDEX (`swmm.toolkit.shared_enum.NodeAttribute` / `LinkAttribute`), the
#: positions of the list `get_node_result` / `get_link_result` return for one time index.
NODE_POSITIONS = {"depth": 0, "head": 1, "volume": 2, "lateral_inflow": 3, "inflow_flow": 4, "flooding": 5}
LINK_POSITIONS = {"flow": 0, "link_depth": 1, "velocity": 2, "link_volume": 3, "capacity_setting": 4}


def _violations(ds_nodes, ds_links) -> list[str]:
    """What is wrong with a reader's output for the reference: period count, last time, final-period values.

    The final-period values are compared with ``get_node_result`` / ``get_link_result`` at time index ``n - 1``, a
    different toolkit call from the series read the code under test uses, and the last time with the last row of
    the report. Element names and positions come from pyswmm's ``Output.nodes`` / ``Output.links`` (a name-to-index
    map that does not touch the series accessors).
    """
    from pyswmm import Output
    from swmm.toolkit import output, shared_enum

    problems = []
    with Output(str(OUT)) as out:
        n = int(output.get_times(out.handle, shared_enum.Time.NUM_PERIODS))
        for kind, ds, positions, getter, elements, idx in (
            ("node", ds_nodes, NODE_POSITIONS, output.get_node_result, out.nodes, "node_id"),
            ("link", ds_links, LINK_POSITIONS, output.get_link_result, out.links, "link_id"),
        ):
            if ds.sizes["date_time"] != n:
                problems.append(f"{kind}: {ds.sizes['date_time']} periods, NUM_PERIODS is {n}")
            for name, element in elements.items():
                final = getter(out.handle, n - 1, element)
                for stored_name, position in positions.items():
                    stored = float(ds[stored_name].sel({idx: str(name)}).isel(date_time=-1))
                    if not np.isclose(stored, final[position], rtol=1e-6, atol=1e-9):
                        problems.append(f"{kind} {stored_name}[{name}] final period {stored} != {final[position]}")
    last_in_report = max(
        np.datetime64(f"{m.group(3)}-{m.group(1)}-{m.group(2)}T{m.group(4)}")
        for m in re.finditer(r"^\s*(\d\d)/(\d\d)/(\d{4}) (\d\d:\d\d:\d\d)\s", RPT.read_text(encoding="latin-1"), re.M)
    )
    if ds_nodes["date_time"].values[-1] != last_in_report:
        problems.append(f"last time {ds_nodes['date_time'].values[-1]} != the report's last row {last_in_report}")
    return problems


def _read():
    return return_node_time_series_results_from_outfile(OUT, capacity_name="capacity_setting")


def test_the_reader_returns_every_report_period_of_the_reference_output():
    """Property: the stores hold NUM_PERIODS periods, end at the report's last time and match the toolkit's last one.

    Class: REGRESSION. Under pyswmm 2.1.0 the accessor-based reader returned NUM_PERIODS - 1 periods and ended one
    report step early. Why required: a store missing its final period misstates every end-of-run statistic and fails
    any comparison with the report. What kills it: reading a series over `(0, num_periods - 1)`, or any accessor that
    does. Second correct implementation: `swmm.toolkit.output.get_node_result` per time index, assembled by hand.
    """
    ds_nodes, ds_links = _read()
    assert _violations(ds_nodes, ds_links) == []


def test_a_reader_that_stops_one_period_early_is_caught(monkeypatch):
    """Property: the checker reports a reader that reads one period too few (must-fire control).

    Class: NEW CAPABILITY. Plausible wrong implementation: a checker that compares the reader with itself.
    Why required: a zero from `_violations` is meaningful only if it can be non-zero on the property it checks.
    What kills it: weakening `_violations` to ignore the period count and the final values.
    Second correct implementation: the 2.1.0 accessor itself, which is exactly this shortfall.
    The variant truncates the last period of the series and of the time axis, which is what the old read produced.
    """
    import hhemt.swmm_output_parser as sop

    real_values, real_axis = sop._out_series_values, sop._out_time_axis
    monkeypatch.setattr(sop, "_out_series_values", lambda *a, **k: real_values(*a, **k)[:-1])
    monkeypatch.setattr(sop, "_out_time_axis", lambda *a, **k: real_axis(*a, **k)[:-1])
    ds_nodes, ds_links = _read()
    problems = _violations(ds_nodes, ds_links)
    assert any("periods, NUM_PERIODS is" in p for p in problems), problems
    assert any("last time" in p for p in problems), problems


def test_the_reader_refuses_a_series_that_does_not_cover_the_output(monkeypatch):
    """Property: a toolkit series shorter than NUM_PERIODS raises ProcessingError instead of being stored.

    Class: NEW CAPABILITY. Plausible wrong implementation: trusting the accessor's length. Why required: a future
    accessor change must not truncate the stores quietly again. What kills it: removing the length check in
    `_out_series_values`. Second correct implementation: comparing the stored axis with `Output.times` after the read.
    """
    from swmm.toolkit import output

    from hhemt.exceptions import ProcessingError

    real = output.get_node_series
    monkeypatch.setattr(output, "get_node_series", lambda h, i, a, s, e: real(h, i, a, s, e - 1))
    with pytest.raises(ProcessingError, match="report periods"):
        _read()


def test_the_refusal_names_the_output_file(monkeypatch):
    """Property: the `ProcessingError` for a short series carries the `.out` path as its filepath, not `.`.

    Class: REGRESSION. The refusal read `getattr(out, "_binfile", "")`, but pyswmm's `Output` stores the path as
    `binfile` (2.0.1 and 2.1.0 both), so the error named `.` whatever the file. Why required: the message is the
    only pointer to the broken output. What kills it: reading the wrong attribute name (the path falls back to `.`).
    Second correct implementation: passing the output path into `_out_series_values` explicitly.
    Control: the path is asserted equal to the reference's, so the `.` fallback fails it.
    """
    from swmm.toolkit import output

    from hhemt.exceptions import ProcessingError

    real = output.get_node_series
    monkeypatch.setattr(output, "get_node_series", lambda h, i, a, s, e: real(h, i, a, s, e - 1))
    with pytest.raises(ProcessingError) as raised:
        _read()
    assert Path(raised.value.filepath) == OUT
