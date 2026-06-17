"""Concurrency + compute-on-read regression tests for the log-write-race fix."""
from __future__ import annotations

import json
from pathlib import Path

from TRITON_SWMM_toolkit.analysis import TRITONSWMM_analysis
from TRITON_SWMM_toolkit.log import TRITONSWMM_analysis_log


# TEST-NEW-1 (main agent authored) — the canonical lost-update regression (VMS-C).
def test_locked_write_preserves_concurrent_field(tmp_path):
    logfile = tmp_path / "log.json"
    TRITONSWMM_analysis_log(logfile=logfile).write()  # materialize the file
    owner = TRITONSWMM_analysis_log.from_json(logfile)
    stale = TRITONSWMM_analysis_log.from_json(logfile)  # loaded BEFORE owner's write
    owner.datatree_consolidation_complete.set(True)  # LogField.set() auto-writes
    stale.cpu_backend_available.set(True)  # writes from a stale baseline
    reloaded = TRITONSWMM_analysis_log.from_json(logfile)
    assert reloaded.datatree_consolidation_complete.get() is True  # NOT clobbered (pre-fix: None)
    assert reloaded.cpu_backend_available.get() is True


# TEST-NEW-2 (main agent authored) — rollups compute on read; persisted fields gone (VMS-D).
def test_rollups_compute_on_read_and_are_not_persisted(synthetic_multisim_completed):
    a = getattr(synthetic_multisim_completed, "analysis", synthetic_multisim_completed)
    for f in (
        "all_scenarios_created",
        "all_sims_run",
        "all_TRITONSWMM_performance_timeseries_processed",
        "all_raw_TRITON_outputs_cleared",
        "all_raw_SWMM_outputs_cleared",
    ):
        assert f not in TRITONSWMM_analysis_log.model_fields  # persisted field removed
    assert isinstance(a.all_sims_run, bool)  # property computes on read
    assert isinstance(a.all_scenarios_created, bool)
    assert isinstance(a.all_raw_TRITON_outputs_cleared, bool)


# TEST-NEW-2b (main agent authored) — legacy log.json with stale all_* keys still loads (extra="ignore").
def test_legacy_log_with_stale_rollup_keys_loads(tmp_path):
    logfile = tmp_path / "log.json"
    logfile.write_text(
        json.dumps({"logfile": str(logfile), "all_sims_run": True, "all_scenarios_created": True})
    )
    assert TRITONSWMM_analysis_log.from_json(logfile) is not None  # no validation error


# TEST-NEW-3 (main agent authored) — read-only construction authors no log write (VMS-SW2-renderer).
# Uses the *cached* (start_from_scratch=False) multisim fixture so it does NOT wipe the
# shared `synth_multi_sim` cache that the session-scoped `synthetic_multisim_completed`
# fixture (used by test_synth_flag_writes / test_synth_07) depends on when these files
# run in the same pytest session. The start_from_scratch=True variant would destroy that
# shared state and cascade unrelated fixture-precondition failures.
def test_readonly_construction_writes_no_log(synth_multi_sim_analysis_cached):
    a = synth_multi_sim_analysis_cached
    a._update_log()  # ensure the log file exists
    logfile = a.log.logfile
    before = (logfile.stat().st_mtime_ns, logfile.read_bytes())
    TRITONSWMM_analysis(
        a.analysis_config_yaml,
        a._system,
        skip_log_update=True,
        is_main_orchestrator=False,
    )  # renderer path
    after = (logfile.stat().st_mtime_ns, logfile.read_bytes())
    assert before == after  # observer authored no write


# TEST-NEW-4 (routed from SE VMS-TEST-1) — static single-writer-invariant guard.
def test_log_single_writer_invariant():
    import TRITON_SWMM_toolkit

    src = Path(TRITON_SWMM_toolkit.__file__).parent
    assert "sub_analysis._update_log()" not in (src / "sensitivity_analysis.py").read_text()
    assert "skip_log_update=True" in (src / "report_renderers" / "_cli.py").read_text()
