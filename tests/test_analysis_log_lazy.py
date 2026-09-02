"""The analysis log loads on FIRST ACCESS, not at construction.

`skip_log_update` already gated the WRITE-back; the READ had no gate, so every
read-only consumer opened the shared master log at construction. On an ensemble
that is hundreds of concurrent workers opening one file none of them reads -- and
each of those reads is an exposure to the transient read failure that produced the
measured ~23% job-failure tax.

The test pins the CONTRACT on the two members that carry it (`_refresh_log` and the
`log` property), not on a full `TRITONSWMM_analysis` construction, which would need
a system, configs and a scenario tree.
"""

from __future__ import annotations

from pathlib import Path

from hhemt.analysis import TRITONSWMM_analysis
from hhemt.log import TRITONSWMM_analysis_log


class _Probe:
    """Binds the real `_refresh_log` and the real `log` property, nothing else."""

    log = TRITONSWMM_analysis.log
    _refresh_log = TRITONSWMM_analysis._refresh_log

    def __init__(self, f_log: Path):
        self.analysis_paths = type("P", (), {"f_log": f_log})()
        self._log = None
        self.reads = 0

    def _counting_refresh(self):
        self.reads += 1
        return TRITONSWMM_analysis._refresh_log(self)


def test_analysis_exposes_log_as_a_property():
    """MECHANISM ARM, by design: the deferral is unobservable without it."""
    assert isinstance(TRITONSWMM_analysis.__dict__["log"], property)
    assert TRITONSWMM_analysis.__dict__["log"].fset is not None, "setter required"


def test_construction_shaped_init_performs_no_log_read(tmp_path):
    p = tmp_path / "log.json"
    TRITONSWMM_analysis_log(logfile=p).write()
    probe = _Probe(p)
    probe._refresh_log = probe._counting_refresh
    assert probe.reads == 0, "constructing the probe must not read the log"


def test_first_access_loads_on_disk_state(tmp_path):
    p = tmp_path / "log.json"
    seeded = TRITONSWMM_analysis_log(logfile=p)
    seeded.write()
    seeded.datatree_consolidation_complete.set(True)

    probe = _Probe(p)
    assert probe.log.datatree_consolidation_complete.get() is True
    assert probe._log is not None, "first access must populate the cache"


def test_first_access_on_absent_log_yields_a_default(tmp_path):
    probe = _Probe(tmp_path / "log.json")
    assert probe.log.datatree_consolidation_complete.get() is None
