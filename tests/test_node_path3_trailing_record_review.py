"""The reviewer's probes of the per-node path-3 attach past the last report time (change 1118-node).

Chunk test: it guards ``TRITONSWMM_sim_post_processing._attach_coupling_node_diagnostics`` and the consolidation that
reads its output, and it is a one-time test to be deleted at the landing unless the coordinator promotes it into the
``process_simulation`` module's own test file.

Why these inputs exist: the coupled SWMM output can hold one report period fewer than the event length, so the solver's
final print-interval record ends one report step after the last report time and no report row contains it. The coder's
file ``tests/test_node_path3_trailing_record.py`` scales that case down to two rows. This file holds the measured
Rivanna case at full size, a differently positioned input class (several trailing records, a report step coarser than
the print interval), and the one property the coder's tests do not reach:
what happens to the recorded count when per-scenario stores are consolidated.

Helpers come from ``tests/test_coupling_diagnostics_ingest.py``. Expected numbers are written down by hand and never
read back from the code under test. No solver runs.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import xarray as xr

from hhemt.coupling_exchange import FT3_TO_M3_FACTOR
from hhemt.utils import write_zarr
from tests.test_coupling_diagnostics_ingest import _ingest_node_store, _node_dataset

ATTR = "path3_records_unattached_after_last_report"


def _attach(tmp_path, records, *, n_times, step_s):
    """Ingest ``records`` for one node n1 and attach them to an ``n_times``-row node dataset at ``step_s`` seconds."""
    proc = _ingest_node_store(tmp_path, records, names=("n1",))
    return proc._attach_coupling_node_diagnostics(_node_dataset(names=("n1",), n_times=n_times, step_s=step_s))


def test_the_measured_rivanna_shape_attaches_966_rows_and_leaves_the_967th_record_unattached(tmp_path):
    """Property: with 966 report rows at 120 s (the last report time 115920 s) and 967 print-interval records
    ending at 120 s ... 116040 s, row k holds record k's own value, the 967th record (ending 116040 s) is not
    attached, the attribute counts 1, and exactly one warning names the count and ``116040.0 s``.

    Class: REGRESSION. Red on the pre-change tree: the attach raises ``ProcessingError`` with ``1 print-interval
    record(s) end after the last report time 115920.0 s (first at 116040.0 s)`` (output pasted in the return).
    Input class: the whole measured job-23098888 shape at its real size, with a distinct value per record so a row
    that received a neighbour's record, or the 967th, shows in the sums.
    Reachability: the measured case itself, the log of ``process_member_R2a`` on jobs 23104526, 23107471 and 23107640.
    Both-states anchor: the assertion turns on the row values and the attribute, which exist in the pre- and the
    post-change code, and on the end time in the warning, not on its wording.
    A second correct implementation: masking the records before the row search rather than after it; the
    assertions hold for it unchanged.
    """
    n_rows = 966
    records = [(120.0 * (j + 1), [float(j + 1)], [-float(j + 1)], [j % 7]) for j in range(n_rows + 1)]
    with pytest.warns(UserWarning, match=r"1 print-interval record.*116040\.0 s") as seen:
        out = _attach(tmp_path, records, n_times=n_rows, step_s=120)
    assert len(seen) == 1
    f = FT3_TO_M3_FACTOR
    np.testing.assert_allclose(out["path3_volume_pos"].sel(node_id="n1").values, np.arange(1, n_rows + 1) * f)
    np.testing.assert_allclose(out["path3_volume_neg"].sel(node_id="n1").values, -np.arange(1, n_rows + 1) * f)
    assert out["nonconverged_steps"].sel(node_id="n1").values.tolist() == [float(j % 7) for j in range(n_rows)]
    assert int(out[ATTR]) == 1
    assert out["date_time"].size == n_rows


def test_several_trailing_records_in_a_coarser_report_step_are_dropped_and_every_other_record_keeps_its_row(tmp_path):
    """Property: with a 240 s report step over 120 s print intervals (4 rows, last report time 960 s) and ten
    records, the two records ending at 1080 s and 1200 s (one print interval and one full report step past the last
    report time) are dropped, the attribute counts 2, and each row is the sum of its own two records: 1 + 2, 4 + 8,
    16 + 32, 64 + 128 (value 2**k for the record ending at 120 s * (k + 1)).

    Class: REGRESSION. Red on the pre-change tree: the attach raises with ``2 print-interval record(s) end after the
    last report time 960.0 s (first at 1200.0 s)`` (output pasted in the return).
    Input class: several trailing records, the allowance measured against a report step wider than the print
    interval. This differs in position from the coder's input (one trailing record, report step equal to the print
    interval). Records are in end-time order, the order the time-marching solver writes them.
    Reachability: the record count is whatever the solver wrote and the report step is a user setting
    (``coupling_exchange`` documents the report step independent of the print interval, the H7 case), so a coarser
    report step with a short period count is an input the attach accepts.
    Both-states anchor: the assertion turns on the four row sums and the attribute, present in both code states.
    Plausible wrong implementations: dropping only the first beyond-record, or masking the values but not the row
    indices (the sums shift); each fails the row-sum assertion.
    A second correct implementation: sorting the records by end time before the row search; the assertions hold.
    """
    values = [2.0**k for k in range(10)]
    records = [(120.0 * (k + 1), [values[k]], [-values[k]], [k]) for k in range(10)]
    with pytest.warns(UserWarning, match=r"2 print-interval record.*1080\.0 s") as seen:
        out = _attach(tmp_path, records, n_times=4, step_s=240)
    assert len(seen) == 1
    f = FT3_TO_M3_FACTOR
    np.testing.assert_allclose(out["path3_volume_pos"].sel(node_id="n1").values, np.array([3.0, 12.0, 48.0, 192.0]) * f)
    np.testing.assert_allclose(
        out["path3_volume_neg"].sel(node_id="n1").values, -np.array([3.0, 12.0, 48.0, 192.0]) * f
    )
    assert out["nonconverged_steps"].sel(node_id="n1").values.tolist() == [1.0, 5.0, 9.0, 13.0]
    assert int(out[ATTR]) == 2


def _per_scenario_store(tmp_path, name, records):
    """Attach ``records`` (two rows at 120 s) for one scenario, write the node dataset as zarr, and reopen it."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = _attach(tmp_path / name, records, n_times=2, step_s=120)
    store = tmp_path / f"{name}.zarr"
    write_zarr(ds, store)
    return xr.open_dataset(store, engine="zarr", chunks=None)


def _consolidate_like_the_toolkit(stores):
    """The concat ``_retrieve_combined_timeseries`` applies to per-scenario node stores (processing_analysis.py)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return xr.concat(
            [s.assign_coords(event_iloc=[i]) for i, s in enumerate(stores)],
            dim="event_iloc",
            combine_attrs="drop_conflicts",
            join="outer",
        )


def _unattached_counts_per_event(ds):
    """The per-event counts a reader can recover from a consolidated dataset, or None when none can be recovered.

    A dimensioned variable or coordinate carries one value per event; a surviving attribute carries one value for
    all events and is accepted only as that single value.
    """
    if ATTR in ds.data_vars or ATTR in ds.coords:
        return np.asarray(ds[ATTR].values).ravel().tolist()
    return ds.attrs.get(ATTR)


def test_the_count_of_unattached_records_survives_consolidation_when_events_disagree(tmp_path):
    """Property: after the per-scenario node stores of two events are concatenated the way the toolkit consolidates
    them, a reader can still recover that event 0 left one record unattached and event 1 left none.

    Class: NEW CAPABILITY. The plausible wrong implementation is the shipped one, a dataset ATTRIBUTE: the
    consolidation concat uses ``combine_attrs="drop_conflicts"``, which deletes an attribute whose value differs
    between events (output pasted in the return: the consolidated attributes are ``{}`` and no variable carries the
    count), so the test is red under it.
    Reachability: any analysis whose events differ in the count.
    ``_retrieve_combined_timeseries('tritonswmm_swmm_node_ts')`` (``processing_analysis.py``,
    ``xr.concat(..., combine_attrs="drop_conflicts", join="outer")``) concatenates the
    per-scenario ``TRITONSWMM_SWMM_node_tseries`` stores this attach writes into; the repository documents the same
    deletion for ``V0030__swmm_run_attributes_to_variables.py`` ("deletes an attribute whose value differs between
    events"). Not measured: whether a real analysis has events that disagree; the attach code admits both counts
    from its data, and a store written before this change has no attribute at all.
    Both-states anchor: the assertion turns on whether the per-event counts can be read back from the consolidated
    dataset by any carrier (variable, coordinate or attribute), a property that holds or fails in both code states
    and does not name the carrier.
    A second correct implementation: a scalar data variable of the same name on each per-scenario store (the
    V0030 pattern), or an ``event_iloc``-dimensioned coordinate stamped at consolidation; the assertion holds for it.
    Must-fire control: two events that AGREE (both leave one record unattached) keep the attribute through the same
    concat, so a miss in the second arm is the disagreement's doing and not a matcher that cannot find a carrier.
    """
    late = [(120.0, [1.0], [-1.0], [1]), (240.0, [2.0], [-2.0], [2]), (360.0, [9.0], [-9.0], [0])]
    whole = late[:2]
    agree = _consolidate_like_the_toolkit(
        [_per_scenario_store(tmp_path, "agree0", late), _per_scenario_store(tmp_path, "agree1", late)]
    )
    assert _unattached_counts_per_event(agree) in (1, [1, 1])

    mixed = _consolidate_like_the_toolkit(
        [_per_scenario_store(tmp_path, "late", late), _per_scenario_store(tmp_path, "whole", whole)]
    )
    assert _unattached_counts_per_event(mixed) == [1, 0]
