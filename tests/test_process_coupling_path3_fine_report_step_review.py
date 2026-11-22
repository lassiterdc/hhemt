"""Reviewer probes of the path-3 skip in ``_attach_coupling_node_diagnostics`` (change 1118-r31attach).

Reviewer-authored, uncommitted. No solver runs; helpers come from ``tests/test_coupling_diagnostics_ingest.py``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import xarray as xr

from hhemt.exceptions import ProcessingError
from hhemt.process_simulation import COUPLING_NODE_DIAGNOSTIC_VARIABLES, PATH3_UNATTACHED_VARIABLE
from tests.test_coupling_diagnostics_ingest import _ingest_node_store, _node_dataset


def _attach(tmp_path, records, *, n_times, step_s):
    proc = _ingest_node_store(tmp_path, records, names=("n1",))
    return proc._attach_coupling_node_diagnostics(_node_dataset(names=("n1",), n_times=n_times, step_s=step_s))


def test_a_coarser_report_step_than_the_print_interval_attaches_without_a_warning(tmp_path):
    """Property: print interval 120 s under a 240 s report step is the ordinary coarser-report case, so the account
    is attached, the unattached count is 0 and no warning is raised (the skip must not widen to it)."""
    records = [
        (120.0, [1.0], [-1.0], [1]),
        (240.0, [2.0], [-2.0], [2]),
        (360.0, [3.0], [-3.0], [3]),
        (480.0, [4.0], [-4.0], [4]),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = _attach(tmp_path, records, n_times=2, step_s=240)
    for name in (*COUPLING_NODE_DIAGNOSTIC_VARIABLES, PATH3_UNATTACHED_VARIABLE):
        assert name in out.data_vars


def test_a_finer_print_interval_with_a_late_first_record_still_raises(tmp_path):
    """Property: print 10 s under a 120 s report step with the first record at 240 s is a shifted capture and the
    print interval is not coarser than the report step, so the refusal stays."""
    records = [(240.0, [1.0], [-1.0], [1]), (250.0, [2.0], [-2.0], [2]), (260.0, [3.0], [-3.0], [3])]
    with pytest.raises(ProcessingError, match=r"first print-interval record ends at 240\.0 s"):
        _attach(tmp_path, records, n_times=4, step_s=120)


def test_the_warning_names_member_store_both_intervals_and_the_absent_variables(tmp_path):
    records = [(120.0, [1.0], [-1.0], [1]), (240.0, [2.0], [-2.0], [2])]
    with pytest.warns(UserWarning) as seen:
        _attach(tmp_path, records, n_times=36, step_s=10)
    text = str(seen[0].message)
    for token in (
        "swmm_node_path3",
        "120.0 s",
        "10.0 s",
        "member",
        *COUPLING_NODE_DIAGNOSTIC_VARIABLES,
        PATH3_UNATTACHED_VARIABLE,
        "every other output is written",
    ):
        assert token in text, token


def test_the_unattached_count_scalar_of_a_skipped_member_concatenates_to_nan_beside_an_attached_one(tmp_path):
    attached = _attach(tmp_path / "a", [(10.0, [1.0], [-1.0], [1]), (20.0, [2.0], [-2.0], [2])], n_times=2, step_s=10)
    with pytest.warns(UserWarning):
        skipped = _attach(
            tmp_path / "s", [(120.0, [1.0], [-1.0], [1]), (240.0, [2.0], [-2.0], [2])], n_times=2, step_s=10
        )
    members = [m.assign_coords(event_iloc=i).expand_dims("event_iloc") for i, m in enumerate((attached, skipped))]
    merged = xr.concat(members, dim="event_iloc", combine_attrs="drop_conflicts")
    col = merged[PATH3_UNATTACHED_VARIABLE]
    assert float(col.sel(event_iloc=0)) == 0.0
    assert np.isnan(float(col.sel(event_iloc=1)))


@pytest.mark.xfail(
    strict=True,
    reason=(
        "FINDING: with print interval 120 s and report step 10 s the skip fires for ANY late first record, including "
        "a capture shifted by whole print intervals (first record at 240 s); only a first record near one print "
        "interval is the legitimate case. The skip is warned, not silent, but it names no first-record time."
    ),
)
def test_a_coarse_print_capture_shifted_by_a_whole_interval_still_raises(tmp_path):
    shifted = [(240.0, [1.0], [-1.0], [1]), (360.0, [2.0], [-2.0], [2]), (480.0, [3.0], [-3.0], [3])]
    with pytest.raises(ProcessingError):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _attach(tmp_path, shifted, n_times=36, step_s=10)
