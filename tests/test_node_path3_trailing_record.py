"""The per-node path-3 attach and the solver's final print-interval record.

Chunk test (change 1118-node, item 1): it guards one behaviour of
``TRITONSWMM_sim_post_processing._attach_coupling_node_diagnostics``, that a record ending one report step after
the last report time is left unattached with a warning and a recorded count (a scalar data variable), while a
record further out is still refused, and that a capture whose first record ends after the first report time is
refused. It guards the ``process_simulation`` module's node attach, and it is a one-time test to be deleted at the
landing unless it is promoted into the module's own test file.

Why the behaviour exists: the coupled SWMM output can hold one report period fewer than the event length, so the
solver's last print-interval record ends one report step after the last report time and no report row contains
it. Refusing it failed the processing of a completed simulation on every attempt.

The sibling file ``tests/test_coupling_diagnostics_ingest.py`` supplies the helpers that write the solver's raw
byte layout and ingest it into the per-node store; expected numbers here are written down by hand and never read
back from the code under test. No solver runs.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from hhemt.coupling_exchange import FT3_TO_M3_FACTOR
from hhemt.exceptions import ProcessingError
from hhemt.process_simulation import COUPLING_NODE_DIAGNOSTIC_VARIABLES
from tests.test_coupling_diagnostics_ingest import _ingest_node_store, _node_dataset

ATTR = "path3_records_unattached_after_last_report"

# Two report rows, at 120 s and 240 s, so the last report time is 240 s and the report step is 120 s. The pos
# values differ per record so that a dropped or misplaced record shows in the row sums.
_ROW_RECORDS = [
    (120.0, [1.0], [-1.0], [1]),
    (240.0, [2.0], [-2.0], [2]),
]


def _attach(tmp_path, records):
    """Ingest ``records`` for one node n1 and attach them to a two-row node dataset (rows at 120 s and 240 s)."""
    proc = _ingest_node_store(tmp_path, records, names=("n1",))
    return proc._attach_coupling_node_diagnostics(_node_dataset(names=("n1",), n_times=2, step_s=120))


def test_a_record_one_report_step_past_the_last_report_time_is_dropped_with_a_warning_and_a_count(tmp_path):
    """Property: with a record ending 120 s (one report step) after the last report time 240 s, the attach
    returns the same row sums as without that record, the scalar data variable counts 1, and one warning names the
    count and the first end time.

    Class: REGRESSION. On the pre-change tree the attach raises ``ProcessingError`` here (red, output pasted in
    the return). The input class is every trailing record ending within one report step after the last report
    time; the second input below, 360.0000005 s, is a differently positioned member of it (a hair under one step
    past, inside the 1e-6 s tolerance the row search uses).
    Reachability: the measured Rivanna case, a 116040 s record against a last report time of 115920 s at a 120 s
    step, is this input scaled.
    Both-states anchor: the assertion turns on the row sums and the count, which exist in the pre- and the
    post-change code, and not on the warning's wording (the match is on the count and the end time only).
    Plausible wrong implementations: clipping the late record into the last row (the last row would read 2 + 1000,
    not 2); dropping it silently (no warning, the count stays 0); recording the count without dropping. Each
    fails one assertion below.
    A second correct implementation: raising nothing and dropping the record by masking the input arrays before
    the row search instead of after it; the assertions hold for it unchanged.
    """
    late = (360.0, [1000.0], [-1000.0], [9])
    with pytest.warns(UserWarning, match=r"1 print-interval record.*360\.0 s") as seen:
        out = _attach(tmp_path / "late", [*_ROW_RECORDS, late])
    assert len(seen) == 1
    f = FT3_TO_M3_FACTOR
    np.testing.assert_allclose(out["path3_volume_pos"].sel(node_id="n1").values, np.array([1.0, 2.0]) * f)
    np.testing.assert_allclose(out["path3_volume_neg"].sel(node_id="n1").values, np.array([-1.0, -2.0]) * f)
    assert out["nonconverged_steps"].sel(node_id="n1").values.tolist() == [1.0, 2.0]
    assert int(out[ATTR]) == 1

    near = (360.0000005, [1000.0], [-1000.0], [9])
    with pytest.warns(UserWarning, match="1 print-interval record"):
        out_near = _attach(tmp_path / "near", [*_ROW_RECORDS, near])
    assert int(out_near[ATTR]) == 1
    np.testing.assert_allclose(out_near["path3_volume_pos"].sel(node_id="n1").values, np.array([1.0, 2.0]) * f)


def test_a_record_more_than_one_report_step_past_the_last_report_time_still_raises(tmp_path):
    """Property: a record ending two report steps (480 s) after the last report time 240 s raises
    ``ProcessingError`` naming the count and the end time, and so does one a millisecond beyond the one-step
    allowance (360.001 s); a mix of one allowed and one refused record raises and counts only the refused one.

    Class: NEW CAPABILITY (a refusal kept). The plausible wrong implementation is dropping every record past the
    last report time whatever its distance; this test is red under it (no raise at 480 s, output pasted in the
    return).
    Reachability: the three inputs sit on both sides of the allowance (one step plus 1e-6 s): 360.001 s is the
    nearest refused position, 360 s the farthest allowed one (first test), 480 s a far one.
    Both-states anchor: the assertion turns on the raise and the refused count, present in the pre- and the
    post-change code (the pre-change message carries the same words).
    A second correct implementation: measuring the allowance as ``last - previous`` report time or as the median
    report step; for the evenly spaced rows here both give 120 s and the test holds.
    """
    with pytest.raises(ProcessingError, match=r"1 print-interval record\(s\) end after the last report time 240\.0 s"):
        _attach(tmp_path / "far", [*_ROW_RECORDS, (480.0, [1.0], [-1.0], [0])])
    with pytest.raises(ProcessingError, match=r"first at 360\.001 s"):
        _attach(tmp_path / "edge", [*_ROW_RECORDS, (360.001, [1.0], [-1.0], [0])])
    with pytest.raises(ProcessingError, match=r"1 print-interval record\(s\) end after the last report time.*480\.0 s"):
        _attach(
            tmp_path / "mixed",
            [*_ROW_RECORDS, (360.0, [1.0], [-1.0], [0]), (480.0, [1.0], [-1.0], [0])],
        )


def test_no_trailing_record_leaves_the_count_at_zero_and_raises_no_warning(tmp_path):
    """Property: when every record lands in a report row (the last one ends exactly at the last report time),
    the count variable is present, scalar, integer and 0, no warning is raised, and the three variables carry each
    record's own value.

    Class: NEW CAPABILITY (the count's zero arm). The plausible wrong implementation is setting the count
    only when a record is dropped (absent variable) or warning on every attach; red under either (output pasted
    in the return).
    Must-fire control: the first test, where the same call with one more record sets the count to 1, so a
    zero here is the data's and not a matcher that never fires.
    Both-states anchor: the assertion turns on the count and the row values.
    A second correct implementation: computing the count as the number of records minus those that landed.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = _attach(tmp_path / "none", _ROW_RECORDS)
    assert ATTR in out.data_vars and int(out[ATTR]) == 0 and out[ATTR].dims == ()
    assert out[ATTR].dtype.kind == "i"
    f = FT3_TO_M3_FACTOR
    np.testing.assert_allclose(out["path3_volume_pos"].sel(node_id="n1").values, np.array([1.0, 2.0]) * f)
    assert set(COUPLING_NODE_DIAGNOSTIC_VARIABLES) <= set(out.data_vars)


def test_a_capture_must_start_at_the_first_report_row(tmp_path):
    """Property: a capture whose first record ends exactly at the first report time (120 s) or earlier (60 s) is
    accepted, and one whose first record ends one print step later (240 s, the first report time being 120 s) is
    refused with the first record's end time and the first report time in the message.

    Class: NEW CAPABILITY. The plausible wrong implementation is accepting a late-shifted capture (the build before
    this round): red under it, because the shifted capture attaches one row late with the count at 1 and no raise.
    Reachability: a capture shifted by one print step with a report step equal to the print interval, the case the
    reviewer measured (the first row NaN, every record one row late).
    Both-states anchor: the assertion turns on the raise and the two numbers in its message.
    A second correct implementation: comparing the earliest record end time instead of the first in file order; the
    solver writes records in time order, so the three inputs here behave the same.
    Not established: a shift under a report step of two or more print intervals is not caught (the shifted first
    record still ends inside the first row); no assertion here claims it is.
    """
    ok = _attach(tmp_path / "exact", _ROW_RECORDS)
    assert int(ok[ATTR]) == 0
    early = _attach(tmp_path / "early", [(60.0, [5.0], [-5.0], [0]), *_ROW_RECORDS])
    assert int(early[ATTR]) == 0
    f = FT3_TO_M3_FACTOR
    np.testing.assert_allclose(early["path3_volume_pos"].sel(node_id="n1").values, np.array([6.0, 2.0]) * f)
    shifted = [(240.0, [1.0], [-1.0], [1]), (360.0, [2.0], [-2.0], [2])]
    with pytest.raises(
        ProcessingError, match=r"first print-interval record ends at 240\.0 s, after the first report time 120\.0 s"
    ):
        _attach(tmp_path / "shifted", shifted)
