"""Coverage for the opt-in per-scenario timeseries consolidation (FQ1) + the durable
``replay_t`` resume-boundary capture (FQ3) — the consolidation half of the Round-2
over-time clean-vs-resume difference figure.

Two tiers:
  * HPC-free unit — ``_parse_replay_t`` extracts the LAST replay marker's ``t=`` value
    from a model-log string (proves FQ3 without any pipeline run).
  * Synth-tier (compile) — flip ``toggle_consolidate_timeseries`` ON, run the full synth
    pipeline (which lands the per-scenario timeseries zarrs, then consolidates), and assert
    the two SWMM node/link timeseries tree nodes appear; a default-OFF re-consolidation has
    no timeseries node (byte-compatible with pre-change trees). Mirrors the
    ``test_metadata_consolidation.py`` end-to-end setup (compile-tier; slow).
"""

from __future__ import annotations

import pytest
import xarray as xr

from hhemt.analysis_validation import _TRITON_REPLAY_MARKER
from hhemt.processing_analysis import _parse_replay_t


# ---------------------------------------------------------------------------
# HPC-free unit — FQ3 replay-marker parse
# ---------------------------------------------------------------------------
def test_parse_replay_t_extracts_float():
    line = f"[00:12:34] {_TRITON_REPLAY_MARKER}123.5\n"
    assert _parse_replay_t(line, _TRITON_REPLAY_MARKER) == 123.5


def test_parse_replay_t_returns_none_when_marker_absent():
    assert _parse_replay_t("nothing to see here\n", _TRITON_REPLAY_MARKER) is None


def test_parse_replay_t_keeps_last_marker():
    # The model log is "w"-truncated per exec, but within one exec multiple markers can
    # appear; the durable value is the LAST one (the final resume boundary).
    text = (
        f"{_TRITON_REPLAY_MARKER}100.0\n"
        f"... more log ...\n"
        f"{_TRITON_REPLAY_MARKER}3000.0\n"
    )
    assert _parse_replay_t(text, _TRITON_REPLAY_MARKER) == 3000.0


def test_parse_replay_t_none_on_unparseable_value():
    assert _parse_replay_t(f"{_TRITON_REPLAY_MARKER}not_a_number\n", _TRITON_REPLAY_MARKER) is None


# ---------------------------------------------------------------------------
# Synth-tier (compile) — FQ1 opt-in timeseries consolidation
# ---------------------------------------------------------------------------
# Proven-good full-pipeline invocation (mirrors test_metadata_consolidation's kwargs);
# process_timeseries=True is required so the per-scenario node/link timeseries zarrs land.
_WORKFLOW_KWARGS = dict(
    mode="local",
    process_system_level_inputs=True,
    overwrite_system_inputs=True,
    compile_TRITON_SWMM=True,
    recompile_if_already_done_successfully=False,
    prepare_scenarios=True,
    overwrite_scenario_if_already_set_up=True,
    rerun_swmm_hydro_if_outputs_exist=True,
    process_timeseries=True,
    which="both",
    override_clear_raw="all",
    compression_level=5,
    pickup_where_leftoff=False,
    verbose=True,
)


def test_default_is_off(synth_multi_sim_builder):
    """Default-OFF regression (HPC-free). The toggle must default False so a
    consolidation that does not opt in stays byte-compatible with pre-change
    trees.

    This is asserted on the CONFIG rather than by a second full-pipeline run:
    consolidation executes in a SUBPROCESS reading the persisted
    analysis_config.yaml (see the toggle-ON test below), so an in-memory flip to
    False would not reach it either -- a "default-OFF" pipeline test written that
    way would pass for the wrong reason. The gating itself is covered because
    every OTHER synth-tier test consolidates without the toggle and none produce
    timeseries nodes.
    """
    assert synth_multi_sim_builder.cfg_analysis.toggle_consolidate_timeseries is False


@pytest.mark.usefixtures("tritonswmm_cpu_compiled")
def test_toggle_on_consolidates_node_and_link_timeseries(tmp_path, monkeypatch):
    # FQ1: with the opt-in toggle ON, the consolidated DataTree gains the two SWMM
    # node/link timeseries nodes (concatenated along event_iloc), each keeping a time dim.
    #
    # The toggle MUST be injected through the case's config-construction path, not
    # set in memory on an already-built analysis. `consolidate_to_datatree` runs in a
    # Snakemake rule-shell SUBPROCESS invoked as
    #   python -m hhemt.consolidate_workflow --analysis-config {...}/analysis_config.yaml
    # so it reads the PERSISTED config; an in-memory attribute set on the parent's
    # analysis object never reaches it and the timeseries loop silently no-ops.
    # `retrieve_synth_TRITON_SWMM_test_case` composes analysis_config.yaml from
    # `additional_analysis_configs`, which is the same seam the hhemt_projects estate
    # runner uses -- so this exercises the real operator path.
    from tests.fixtures.test_case_builder import retrieve_synth_TRITON_SWMM_test_case

    # Isolate the start_from_scratch wipe under tmp_path (mirrors the
    # synth_multi_sim_analysis fixture) so it cannot clobber the shared session cache.
    monkeypatch.setenv("HHEMT_TEST_RUNS_ROOT_OVERRIDE", str(tmp_path))
    case = retrieve_synth_TRITON_SWMM_test_case(
        analysis_name="synth_multi_sim",
        n_events=3,
        start_from_scratch=True,
        additional_analysis_configs={"toggle_consolidate_timeseries": True},
    )
    a = case.analysis
    assert a.cfg_analysis.toggle_consolidate_timeseries is True
    result = a.submit_workflow(**_WORKFLOW_KWARGS)
    assert result["success"], f"Workflow failed: {result.get('message', '')}"

    tree_path = a.analysis_paths.analysis_datatree_zarr
    assert tree_path is not None and tree_path.exists()
    tree = xr.open_datatree(tree_path, engine="zarr", consolidated=False)

    tsg = tree["tritonswmm"]
    assert "swmm_node_timeseries" in tsg
    assert "swmm_link_timeseries" in tsg
    # The per-scenario SWMM timeseries name their time axis `date_time` (NOT `time`).
    # The VMS-4 assertion contract said "time"; the live tree is
    # {event_iloc, node_id|link_id, date_time}, so the contract's name was wrong and
    # is corrected here against observed data rather than propagated.
    for node_name in ("swmm_node_timeseries", "swmm_link_timeseries"):
        node = tsg[node_name]
        assert "date_time" in node.dims, f"{node_name} missing date_time dim: {dict(node.dims)}"
        assert "event_iloc" in node.dims, f"{node_name} missing event_iloc dim: {dict(node.dims)}"

    # NOTE: the default-OFF regression is asserted in `test_default_is_off` above,
    # on the config rather than by flipping the toggle here. An in-memory flip to
    # False followed by a re-consolidate would NOT reach the consolidating
    # subprocess (same persisted-config mechanism documented at the top of this
    # test), so it would have passed while proving nothing about the gate.


@pytest.mark.usefixtures("tritonswmm_cpu_compiled")
def test_toggle_change_invalidates_an_already_consolidated_tree(tmp_path, monkeypatch):
    """The consolidation-inputs fingerprint rebuilds a STALE tree.

    This is the state transition the fingerprint exists to produce, and nothing
    else in this file exercises it: both tests above construct the case with the
    toggle already at its final value at materialization time, so neither ever
    meets a tree that is present, complete, AND built from different inputs.

    The 2026-07-20 production failure was exactly that state — the toggle was
    flipped ON in all 28 persisted per-sub configs, consolidation re-ran, and the
    output was silently unchanged, because the guard keyed only on
    ``fname_out.exists() and _log_complete``. `regenerate_existing` does not help:
    it is a driver-side DELETION flag that never reaches this guard (verified: it
    is passed to `consolidate_to_datatree` nowhere in `src/hhemt/`), and on an HPC
    method its deletion is routed to a SLURM-offloaded workflow whose failure the
    call site discards.

    Shape: consolidate with the toggle OFF, rebuild the analysis object from a
    config with the toggle ON WITHOUT wiping, consolidate again, assert the
    timeseries nodes appear. The second consolidation is invoked directly rather
    than through a second full workflow — the guard under test lives in
    ``consolidate_to_datatree``, so driving it directly is both sufficient and
    ~5 minutes cheaper.
    """
    from tests.fixtures.test_case_builder import retrieve_synth_TRITON_SWMM_test_case

    monkeypatch.setenv("HHEMT_TEST_RUNS_ROOT_OVERRIDE", str(tmp_path))

    # --- Pass 1: toggle OFF (the default). Full pipeline, so the per-scenario
    # timeseries zarrs land on disk and are AVAILABLE to consolidate -- the point
    # is that they are available and still not consolidated, not that they are absent.
    case_off = retrieve_synth_TRITON_SWMM_test_case(
        analysis_name="synth_multi_sim", n_events=3, start_from_scratch=True
    )
    a_off = case_off.analysis
    assert a_off.cfg_analysis.toggle_consolidate_timeseries is False
    result = a_off.submit_workflow(**_WORKFLOW_KWARGS)
    assert result["success"], f"Workflow failed: {result.get('message', '')}"

    tree_path = a_off.analysis_paths.analysis_datatree_zarr
    assert tree_path is not None and tree_path.exists()
    tsg_off = xr.open_datatree(tree_path, engine="zarr", consolidated=False)["tritonswmm"]
    assert "swmm_node_timeseries" not in tsg_off, "toggle OFF should not consolidate timeseries"
    mtime_after_pass1 = tree_path.stat().st_mtime

    # --- Pass 2: same on-disk analysis, toggle ON in the PERSISTED config.
    # start_from_scratch=False so the pass-1 tree SURVIVES -- that surviving,
    # complete-but-stale tree is the whole fixture.
    case_on = retrieve_synth_TRITON_SWMM_test_case(
        analysis_name="synth_multi_sim",
        n_events=3,
        start_from_scratch=False,
        additional_analysis_configs={"toggle_consolidate_timeseries": True},
    )
    a_on = case_on.analysis
    assert a_on.cfg_analysis.toggle_consolidate_timeseries is True
    assert tree_path.exists(), "pass-1 tree must survive; otherwise this tests a fresh build, not staleness"

    a_on.process.consolidate_to_datatree(compression_level=5, verbose=True)

    # The guard must have detected the fingerprint mismatch and rebuilt.
    tsg_on = xr.open_datatree(tree_path, engine="zarr", consolidated=False)["tritonswmm"]
    for node_name in ("swmm_node_timeseries", "swmm_link_timeseries"):
        assert node_name in tsg_on, (
            f"{node_name} absent after a toggle change on an already-consolidated tree — "
            "the consolidation-inputs fingerprint did not invalidate it"
        )
    assert tree_path.stat().st_mtime > mtime_after_pass1, "tree was not rewritten"
