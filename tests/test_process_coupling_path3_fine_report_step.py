"""The per-node path-3 attach when the SWMM report step is finer than the capture's print interval.

Chunk test (change 1118-r31attach): it guards ``TRITONSWMM_sim_post_processing._attach_coupling_node_diagnostics``,
which failed the whole member after every output zarr was written when the capture's print interval (120 s) was
coarser than the report step (10 s), the measured Rivanna case of member R31. The attach now skips the per-node
account with a warning and the member keeps every other output. No solver runs; helpers come from
``tests/test_coupling_diagnostics_ingest.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from hhemt.exceptions import ProcessingError
from hhemt.process_simulation import COUPLING_NODE_DIAGNOSTIC_VARIABLES, PATH3_UNATTACHED_VARIABLE
from tests.test_coupling_diagnostics_ingest import _ingest_node_store, _node_dataset

# Print interval 120 s: records end at 120 s, 240 s, 360 s.
_RECORDS = [
    (120.0, [1.0], [-1.0], [1]),
    (240.0, [2.0], [-2.0], [2]),
    (360.0, [3.0], [-3.0], [3]),
]


def _attach(tmp_path, records, *, n_times, step_s):
    proc = _ingest_node_store(tmp_path, records, names=("n1",))
    return proc._attach_coupling_node_diagnostics(_node_dataset(names=("n1",), n_times=n_times, step_s=step_s))


def test_a_report_step_finer_than_the_print_interval_skips_the_account_with_a_warning(tmp_path):
    """Property: with the real shape (print interval 120 s, report step 10 s, 36 report rows) the attach returns the
    node dataset with its own variable intact, adds none of the path-3 variables or the unattached count, and warns
    once naming both intervals.

    Class: REGRESSION. Red on the pre-change tree: ``ProcessingError`` "the first print-interval record ends at
    120.0 s, after the first report time 10.0 s".
    Must-fire control: the shifted-capture test below, where the same message still raises for equal intervals.
    """
    with pytest.warns(UserWarning, match=r"print interval is 120\.0 s and the SWMM report step is 10\.0 s") as seen:
        out = _attach(tmp_path, _RECORDS, n_times=36, step_s=10)
    assert len(seen) == 1
    assert "depth" in out.data_vars and out["depth"].shape == (36, 1)
    for name in (*COUPLING_NODE_DIAGNOSTIC_VARIABLES, PATH3_UNATTACHED_VARIABLE):
        assert name not in out.data_vars


def test_a_capture_shifted_late_at_equal_intervals_and_a_lone_late_record_still_raise(tmp_path):
    """Property: the refusal is kept where the print interval is not coarser than the report step (first record at
    240 s, report step 120 s, print interval 120 s), and where the print interval cannot be measured (one record).

    Class: NEW CAPABILITY (a refusal kept). A skip that fired on any late first record is red here.
    """
    shifted = [(240.0, [1.0], [-1.0], [1]), (360.0, [2.0], [-2.0], [2])]
    with pytest.raises(ProcessingError, match=r"first print-interval record ends at 240\.0 s"):
        _attach(tmp_path / "shifted", shifted, n_times=2, step_s=120)
    with pytest.raises(ProcessingError, match=r"first print-interval record ends at 120\.0 s"):
        _attach(tmp_path / "lone", [(120.0, [1.0], [-1.0], [1])], n_times=36, step_s=10)


def test_a_skipped_member_concatenates_with_an_attached_one_along_event_iloc(tmp_path):
    """Property: the per-member timeseries concat of the consolidation (``xr.concat`` along ``event_iloc`` with
    ``join="outer"``, the call in ``processing_analysis.py``) accepts one member without the path-3 variables beside
    one with them, filling the absent member with NaN.
    """
    attached = _attach(tmp_path / "a", [(10.0, [1.0], [-1.0], [1]), (20.0, [2.0], [-2.0], [2])], n_times=2, step_s=10)
    with pytest.warns(UserWarning):
        skipped = _attach(tmp_path / "s", _RECORDS, n_times=2, step_s=10)
    members = [m.assign_coords(event_iloc=i).expand_dims("event_iloc") for i, m in enumerate((attached, skipped))]
    merged = xr.concat(members, dim="event_iloc", combine_attrs="drop_conflicts", join="outer")
    pos = merged["path3_volume_pos"].sel(node_id="n1")
    assert not np.isnan(pos.sel(event_iloc=0).values).any()
    assert np.isnan(pos.sel(event_iloc=1).values).all()
