"""Synth-tier integration: the embedded provenance core + RO-Crate sidecar appear at
consolidation (regular + sensitivity master/sub), idempotently.

These run the full synth pipeline via ``submit_workflow`` (which compiles TRITON-SWMM,
runs sims, processes summaries, and consolidates) so the Phase-3 embed/sidecar wiring is
exercised on real consolidated outputs — a bare analysis fixture has no summaries to
consolidate. Compile-tier (slow); mirrors the ``test_synth_04/05`` end-to-end setup.
"""

from __future__ import annotations

import json

import pytest
import xarray as xr

# Proven-good full-pipeline invocation (mirrors test_synth_05's submit_workflow call).
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


@pytest.mark.usefixtures("tritonswmm_cpu_compiled")
def test_consolidation_emits_core_and_sidecar(synth_multi_sim_analysis):
    # R1/R4: regular analysis. Run the pipeline (consolidation runs inside it), then assert
    # the embedded core + co-located sidecar, and that a re-consolidation is mtime-stable.
    a = synth_multi_sim_analysis
    result = a.submit_workflow(**_WORKFLOW_KWARGS)
    assert result["success"], f"Workflow failed: {result.get('message', '')}"

    tree_path = a.analysis_paths.analysis_datatree_zarr
    assert tree_path is not None and tree_path.exists()
    tree = xr.open_datatree(tree_path, engine="zarr", consolidated=False)
    assert "ro_crate_metadata" in tree.attrs
    json.loads(tree.attrs["ro_crate_metadata"])  # valid JSON-LD

    sidecar = a.analysis_paths.analysis_dir / "ro-crate-metadata.json"
    assert sidecar.exists()

    # R4: re-consolidation does not bump the sidecar mtime (idempotent: the log-complete
    # early-return + write_rocrate_sidecar compare-and-write both preserve mtime).
    m1 = sidecar.stat().st_mtime_ns
    a.process.consolidate_to_datatree()
    assert sidecar.stat().st_mtime_ns == m1


@pytest.mark.usefixtures("tritonswmm_cpu_compiled")
def test_sensitivity_consolidation_emits_master_and_sub_provenance(synth_sensitivity_analysis):
    # R2: sensitivity master carries the embedded core + co-located sidecar, and each
    # consolidated sub-analysis independently carries its own sidecar (produced by the
    # per-sub consolidate_to_datatree wiring — no sub-loop in the master path).
    a = synth_sensitivity_analysis
    result = a.submit_workflow(**_WORKFLOW_KWARGS)
    assert result["success"], f"Workflow failed: {result.get('message', '')}"

    sensitivity = a.sensitivity
    master_zarr = sensitivity.analysis_paths.sensitivity_datatree_zarr
    assert master_zarr is not None and master_zarr.exists()
    master_tree = xr.open_datatree(master_zarr, engine="zarr", consolidated=False)
    assert "ro_crate_metadata" in master_tree.attrs
    json.loads(master_tree.attrs["ro_crate_metadata"])  # valid JSON-LD
    master_sidecar = sensitivity.master_analysis.analysis_paths.analysis_dir / "ro-crate-metadata.json"
    assert master_sidecar.exists()

    consolidated_subs = 0
    for _sa_id, sub in sensitivity.sub_analyses.items():
        sub_zarr = sub.analysis_paths.analysis_datatree_zarr
        if sub_zarr is not None and sub_zarr.exists():
            consolidated_subs += 1
            assert (sub.analysis_paths.analysis_dir / "ro-crate-metadata.json").exists()
    assert consolidated_subs > 0  # at least one sub consolidated + carried a sidecar
