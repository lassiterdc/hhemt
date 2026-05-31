"""FIX 1 (reprocess rebuild correctness) — driver-side flag + log invalidation.

Compile-INDEPENDENT coverage (no GPU / no real simulation pipeline; uses the
compile-free object fixtures + ``tmp_path``-rooted on-disk trees). The applied
implementation edits (Option A) ensure the cheap per-model processing-LOG clear
runs on EVERY reprocess route (SLURM-offload AND in-process), so the rebuilt
``process_*`` rule's runner actually re-writes its outputs (the
``process_simulation.py::_already_written`` gate keys on the per-model
``processing_log.outputs`` dict, Gotcha #28 — NOT on flag presence). Without
this clear, a SLURM-routed reprocess would emit fresh flags + a rebuild rule but
the runner would skip every ``_export_*`` write (stale-zarr silent-failure mode).

The three tests here drive:

- ``(a)`` SENSITIVITY reprocess SLURM-route: ``d_process_*`` flags deleted +
  per-scenario per-model ``processing_log.outputs`` cleared for every sub.
- ``(b)`` the reprocess MASTER generator emits a ``process_*`` rebuild rule per
  (sa, event) whose ``d_process`` flag is absent, routes the per-sa consolidate
  input through that ``d_process`` flag, and the generated Snakefile PARSES via a
  Snakemake dry-run.
- ``(b2)`` NON-sensitivity reprocess SLURM-route: per-scenario per-model
  ``processing_log.outputs`` cleared (the CHANGE A1 fix).

The compile-gated end-to-end rebuild assertion lives in
``test_synth_08_sensitivity_reprocess.py::test_reprocess_rebuild_rewrites_summary``.
"""

from __future__ import annotations

import math
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from TRITON_SWMM_toolkit.constants import (
    process_timeseries_flag_per_sa,
    sa_inputs_fingerprint_flag,
    sim_run_flag_per_sa,
)
from TRITON_SWMM_toolkit.exceptions import ProcessingError
from TRITON_SWMM_toolkit.log import ProcessingEntry
from TRITON_SWMM_toolkit.process_simulation import TRITONSWMM_sim_post_processing
from TRITON_SWMM_toolkit.scenario import TRITONSWMM_scenario, compute_event_id_slug


def _seed_scenario_processing_log(scen: TRITONSWMM_scenario) -> list[str]:
    """Seed each enabled model's ``processing_log.outputs`` with a non-empty
    entry and persist. Returns the enabled model types so callers can re-read
    and assert the post-reprocess clear."""
    model_types = list(scen.run.model_types_enabled)
    for model_type in model_types:
        ml = scen.get_log(model_type)
        ml.processing_log.outputs["seed.zarr"] = ProcessingEntry(
            filepath=Path("seed.zarr"),
            size_MiB=1.0,
            time_elapsed_s=1.0,
            success=True,
        )
        ml.write()
    return model_types


# ---------------------------------------------------------------------------
# (a) sensitivity reprocess SLURM-route — flags + logs cleared for every sub
# ---------------------------------------------------------------------------


def test_reprocess_regenerate_slurm_route_clears_flags_and_logs(
    norfolk_sensitivity_analysis, monkeypatch
):
    """Sensitivity ``reprocess(start_with='process', regenerate_existing=True)``
    on the SLURM-offload route deletes every sub's ``d_process_*`` flags AND
    clears every scenario's per-model ``processing_log.outputs`` (FIX 1, hunk 2a)."""
    analysis = norfolk_sensitivity_analysis
    sensitivity = analysis.sensitivity

    # Force the SLURM-offload route: batch_job → _hpc=True → route_delete_via_slurm.
    sensitivity.master_analysis.cfg_analysis.multi_sim_run_method = "batch_job"

    # Patch BOTH submission methods to no-op stubs so no real workflow fires.
    monkeypatch.setattr(
        sensitivity._workflow_builder._base_builder,
        "submit_reprocess_delete_workflow",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        sensitivity._workflow_builder,
        "submit_reprocess_workflow",
        lambda *a, **k: {"success": True},
    )

    master_analysis_dir = sensitivity.master_analysis.analysis_paths.analysis_dir
    status_dir = master_analysis_dir / "_status"
    status_dir.mkdir(parents=True, exist_ok=True)

    # Seed, per sub: (i) dummy d_process_*_sa-<sa>_evt-<eid>_complete.flag files
    # via the real flag-name builder, and (ii) a non-empty per-model processing
    # log for each scenario.
    seeded: dict[str, list[tuple[int, list[str]]]] = {}
    for sa_id, sub_analysis in sensitivity.sub_analyses.items():
        sa_str = str(sa_id)
        per_sub: list[tuple[int, list[str]]] = []
        for event_iloc in sub_analysis.df_sims.index:
            scen = TRITONSWMM_scenario(event_iloc, sub_analysis)
            event_id = scen.event_id
            model_types = _seed_scenario_processing_log(scen)
            for model_type in model_types:
                flag_rel = process_timeseries_flag_per_sa(model_type, sa_str, event_id)
                flag_path = master_analysis_dir / flag_rel
                flag_path.parent.mkdir(parents=True, exist_ok=True)
                flag_path.write_text("")
            per_sub.append((event_iloc, model_types))
        seeded[sa_str] = per_sub

    # Precondition sanity: flags + log entries exist before reprocess.
    for sa_str, per_sub in seeded.items():
        assert list(status_dir.glob(f"d_process_*_sa-{sa_str}_*")), (
            f"precondition: d_process flags must exist for sa {sa_str}"
        )

    sensitivity.reprocess(
        start_with="process",
        regenerate_existing=True,
        delete_via_slurm=True,
        execution_mode="local",
        dry_run=False,
        verbose=False,
    )

    for sa_id, sub_analysis in sensitivity.sub_analyses.items():
        sa_str = str(sa_id)
        # (a-i) every d_process flag for this sub was deleted.
        remaining = list(status_dir.glob(f"d_process_*_sa-{sa_str}_*"))
        assert remaining == [], (
            f"sa {sa_str}: d_process_* flags must all be deleted; found {remaining!r}"
        )
        # (a-ii) every scenario's per-model processing_log.outputs is empty.
        for event_iloc, model_types in seeded[sa_str]:
            scen = TRITONSWMM_scenario(event_iloc, sub_analysis)
            for model_type in model_types:
                reloaded = scen.get_log(model_type)
                assert reloaded.processing_log.outputs == {}, (
                    f"sa {sa_str} event_iloc {event_iloc} model {model_type}: "
                    f"processing_log.outputs must be cleared; got "
                    f"{reloaded.processing_log.outputs!r}"
                )


# ---------------------------------------------------------------------------
# (b) reprocess master generator emits rebuild + parses
# ---------------------------------------------------------------------------


def test_reprocess_generator_emits_rebuild_after_invalidation(norfolk_sensitivity_analysis):
    """Pure generator test (no reprocess() call). With ``c_run`` flags present
    and ``d_process`` flags absent, the reprocess master generator emits a
    ``process_*`` rebuild rule per (sa, event) and routes each per-sa
    consolidate's input through the ``d_process`` flag. The generated Snakefile
    PARSES (Snakemake dry-run, no compiler needed)."""
    analysis = norfolk_sensitivity_analysis
    builder = analysis.sensitivity._workflow_builder

    analysis_dir = builder.master_analysis.analysis_paths.analysis_dir
    status_dir = analysis_dir / "_status"
    status_dir.mkdir(parents=True, exist_ok=True)

    # The sensitivity master generator derives a single model_type from the
    # system toggles (it forbids multi-model sensitivity). Mirror that.
    enabled_models: list[str] = []
    cfg_system = builder.system.cfg_system
    if cfg_system.toggle_triton_model:
        enabled_models.append("triton")
    if cfg_system.toggle_tritonswmm_model:
        enabled_models.append("tritonswmm")
    if cfg_system.toggle_swmm_model:
        enabled_models.append("swmm")
    assert len(enabled_models) == 1, (
        f"sensitivity fixture must enable exactly one model; got {enabled_models!r}"
    )
    model_type = enabled_models[0]

    # Seed: every (sa, event) gets its c_run flag present, d_process ABSENT
    # (mirror workflow.py:6675-6678 for the event_id slug derivation).
    sa_event_pairs: list[tuple[str, str]] = []
    for sa_id, sub_analysis in analysis.sensitivity.sub_analyses.items():
        sa_str = str(sa_id)
        # The per-sa process + consolidate rules declare the per-sa input
        # fingerprint as `input:`; seed it so the dry-run DAG resolves.
        fingerprint_rel = sa_inputs_fingerprint_flag(sa_str)
        (analysis_dir / fingerprint_rel).parent.mkdir(parents=True, exist_ok=True)
        (analysis_dir / fingerprint_rel).write_text("{}")
        for event_iloc in sub_analysis.df_sims.index:
            event_id = compute_event_id_slug(
                sub_analysis._retrieve_weather_indexer_using_integer_index(event_iloc)
            )
            c_run_rel = sim_run_flag_per_sa(model_type, sa_str, event_id)
            d_process_rel = process_timeseries_flag_per_sa(model_type, sa_str, event_id)
            (analysis_dir / c_run_rel).parent.mkdir(parents=True, exist_ok=True)
            (analysis_dir / c_run_rel).write_text("")
            (analysis_dir / d_process_rel).unlink(missing_ok=True)
            sa_event_pairs.append((sa_str, event_id))

    assert sa_event_pairs, "fixture must construct at least one (sa, event) pair"

    content = builder.generate_reprocess_master_snakefile_content(start_with="process")

    # A process rebuild rule per (sa, event) — the generator emits
    # `rule process_sa_{sa}_evt_{event}` (sa_id_rule/event_id_rule normalize
    # '.'/'-' to '_').
    for sa_str, event_id in sa_event_pairs:
        sa_rule = sa_str.replace(".", "_").replace("-", "_")
        event_rule = event_id.replace(".", "_").replace("-", "_")
        rule_name = f"process_sa_{sa_rule}_evt_{event_rule}"
        assert f"rule {rule_name}:" in content, (
            f"expected reprocess rebuild rule '{rule_name}' in generated content"
        )

    # Each per-sa consolidate routes its input through the d_process flag the
    # rebuild rule produces (conditional-process-emit routing), not c_run.
    for sa_id, sub_analysis in analysis.sensitivity.sub_analyses.items():
        sa_str = str(sa_id)
        for event_iloc in sub_analysis.df_sims.index:
            event_id = compute_event_id_slug(
                sub_analysis._retrieve_weather_indexer_using_integer_index(event_iloc)
            )
            d_process_rel = process_timeseries_flag_per_sa(model_type, sa_str, event_id)
            assert f'"{d_process_rel}"' in content, (
                f"per-sa consolidate input must include d_process flag {d_process_rel!r}"
            )

    # The generated Snakefile must PARSE — write it and run a Snakemake dry-run.
    snakefile = analysis_dir / "Snakefile.reprocess"
    snakefile.write_text(content)
    proc = subprocess.run(
        [
            "snakemake",
            "-n",
            "--snakefile",
            str(snakefile),
            "--rerun-triggers",
            "mtime",
            "--directory",
            str(analysis_dir),
        ],
        capture_output=True,
        text=True,
        cwd=str(analysis_dir),
    )
    assert proc.returncode == 0, (
        "generated reprocess Snakefile must parse + plan via `snakemake -n`; "
        f"returncode={proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )


# ---------------------------------------------------------------------------
# (b2) non-sensitivity reprocess SLURM-route — per-model log cleared (CHANGE A1)
# ---------------------------------------------------------------------------


def test_reprocess_regenerate_slurm_route_clears_log_nonsensitivity(
    norfolk_multi_sim_analysis, monkeypatch
):
    """Non-sensitivity ``reprocess(start_with='process', regenerate_existing=True)``
    on the SLURM-offload route clears each scenario's per-model
    ``processing_log.outputs`` (FIX 1, CHANGE A1)."""
    analysis = norfolk_multi_sim_analysis

    # Force the SLURM-offload route.
    analysis.cfg_analysis.multi_sim_run_method = "batch_job"

    # Patch BOTH submission methods so the route fires no real workflow.
    monkeypatch.setattr(
        analysis._workflow_builder,
        "submit_reprocess_delete_workflow",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        analysis._workflow_builder,
        "submit_reprocess_workflow",
        lambda *a, **k: {"success": True},
    )

    # Seed each scenario's per-model processing log non-empty.
    seeded: list[tuple[int, list[str]]] = []
    for event_iloc in analysis.df_sims.index:
        scen = TRITONSWMM_scenario(event_iloc, analysis)
        model_types = _seed_scenario_processing_log(scen)
        seeded.append((event_iloc, model_types))

    assert seeded, "fixture must construct at least one scenario"

    analysis.reprocess(
        start_with="process",
        regenerate_existing=True,
        delete_via_slurm=True,
        execution_mode="local",
        dry_run=False,
        verbose=False,
    )

    for event_iloc, model_types in seeded:
        scen = TRITONSWMM_scenario(event_iloc, analysis)
        for model_type in model_types:
            reloaded = scen.get_log(model_type)
            assert reloaded.processing_log.outputs == {}, (
                f"event_iloc {event_iloc} model {model_type}: processing_log.outputs "
                f"must be cleared; got {reloaded.processing_log.outputs!r}"
            )


# ---------------------------------------------------------------------------
# (c) consolidate-stage in-process reprocess PRESERVES processed/ (regression
#     guard — the FIX-1 Phase-1 defect where an unguarded inline processed/
#     rmtree deleted the rebuild source on the consolidate path, 2026-05-31)
# ---------------------------------------------------------------------------


def test_reprocess_consolidate_inprocess_preserves_processed(
    norfolk_sensitivity_analysis, monkeypatch
):
    """A CONSOLIDATE-stage in-process ``reprocess(regenerate_existing=True)`` must
    PRESERVE each sub's per-scenario ``processed/`` — that directory is the
    rebuild source the consolidate stage reads from; only a PROCESS-stage
    reprocess may delete it (the ``_delete_processed_outputs_for_reprocess``
    helper's ``start_with == "process"`` guard, analysis.py ~L2979).

    Regression guard for the FIX-1 Phase-1 escape: an inline processed/ rmtree
    in the in-process ``elif`` branch WITHOUT that guard deleted processed/ on a
    consolidate-stage reprocess, so the downstream consolidate Snakemake step
    had no summaries to rebuild from and failed (cascading the synth_08 e2e
    suite). This compile-free test exercises that exact branch in seconds — the
    coverage gap that previously let the defect reach only the ~4-min
    compile-gated Tier-2 gate.
    """
    analysis = norfolk_sensitivity_analysis
    sensitivity = analysis.sensitivity

    # In-process (local) route: pass delete_via_slurm=False so
    # route_delete_via_slurm is False and the in-process `elif` branch fires
    # (the branch that owns the processed/ + zarr deletion decision).
    monkeypatch.setattr(
        sensitivity._workflow_builder,
        "submit_reprocess_workflow",
        lambda *a, **k: {"success": True},
    )

    # Seed each sub's per-scenario processed/ dir on disk (the rebuild source).
    seeded_procs: list[Path] = []
    for sa_id, sub_analysis in sensitivity.sub_analyses.items():
        for event_iloc in range(len(sub_analysis.df_sims)):
            scen = TRITONSWMM_scenario(event_iloc, sub_analysis)
            proc = scen.scen_paths.sim_folder / "processed"
            proc.mkdir(parents=True, exist_ok=True)
            (proc / "summary.zarr").mkdir(exist_ok=True)
            seeded_procs.append(proc)
    assert seeded_procs, "fixture must construct at least one scenario"

    sensitivity.reprocess(
        start_with="consolidate",
        regenerate_existing=True,
        delete_via_slurm=False,
        execution_mode="local",
        dry_run=False,
        verbose=False,
    )

    for proc in seeded_procs:
        assert proc.exists(), (
            "consolidate-stage reprocess must PRESERVE per-scenario processed/ "
            f"(the rebuild source); {proc} was deleted"
        )


# ---------------------------------------------------------------------------
# (c) FIX 2 — append-batch decoupled from load chunk (compile-free, synthetic)
# ---------------------------------------------------------------------------


def _build_synthetic_post_processing(*, fname_out, raw_dir, batch_timesteps, ny, nx):
    """Construct a TRITONSWMM_sim_post_processing via __new__ with only the
    attributes _export_TRITONSWMM_TRITON_outputs reads, wired to synthetic
    stand-ins. Instance attributes shadow the class helper methods the export
    method calls on ``self``."""
    inst = object.__new__(TRITONSWMM_sim_post_processing)
    rds_dem = xr.Dataset(coords={"y": np.arange(ny), "x": np.arange(nx)})
    inst._resolve_clear_raw = lambda override: None
    inst._validate_path = lambda path, name: fname_out
    inst._already_written = lambda f: False
    inst._should_clear_raw_for_model = lambda resolved, model: False
    inst._analysis = SimpleNamespace(
        cfg_analysis=SimpleNamespace(
            TRITON_raw_output_type="bin",
            TRITON_reporting_timestep_s=60,
            process_output_target_chunksize_mb=200,
            process_append_batch_timesteps=batch_timesteps,
        )
    )
    inst._run = SimpleNamespace(raw_triton_output_dir=lambda model_type: raw_dir)
    inst._system = SimpleNamespace(processed_dem_rds=rds_dem)
    inst.scen_paths = SimpleNamespace(output_tritonswmm_triton_timeseries=fname_out)
    inst._scenario = SimpleNamespace(
        latest_sim_date=lambda model_type, astype: "2020-01-01"
    )
    # MagicMock tolerates the post-write log-field accesses (add_sim_processing_entry,
    # TRITON_timeseries_written.set(...)) without modelling each one.
    inst.log = MagicMock()
    return inst


def _patch_loaders(monkeypatch, *, df_outputs, ny, nx):
    """Force the load chunk to floor at 1 timestep and replace the on-disk
    loaders with synthetic in-memory datasets, so the batched-append loop is
    exercised without any real TRITON binaries or compiled engine."""
    import TRITON_SWMM_toolkit.process_simulation as ps
    import TRITON_SWMM_toolkit.utils as utils

    # estimate_timesteps_per_chunk is imported method-locally from utils, so the
    # patch target is utils (not the process_simulation namespace).
    monkeypatch.setattr(utils, "estimate_timesteps_per_chunk", lambda **k: 1)
    monkeypatch.setattr(ps, "return_fpath_wlevels", lambda fldr, interval: df_outputs)
    monkeypatch.setattr(
        ps,
        "load_triton_output_w_xarray",
        lambda rds_dem, f, varname, raw_out_type: xr.Dataset(
            {varname: (("y", "x"), np.zeros((ny, nx)))},
            coords={"y": np.arange(ny), "x": np.arange(nx)},
        ),
    )
    monkeypatch.setattr(ps, "return_dic_zarr_encodings", lambda ds, comp_level: {})
    monkeypatch.setattr(ps, "current_datetime_string", lambda: "2020-01-01")
    monkeypatch.setattr(ps, "convert_datetime_to_str", lambda attrs: attrs)
    monkeypatch.setattr(ps, "get_file_size_MiB", lambda f: 1.0)


def test_append_batch_decoupled_from_load_chunk(tmp_path, monkeypatch):
    """With the load chunk floored to 1 timestep (the fine-grid degeneration this
    fix targets), the batched-append path must emit ceil(N / append_batch_timesteps)
    zarr writes — NOT one append per timestep — and the resulting store must hold
    every timestep (no loss across batched appends). R4 + R5 (separate knob)."""
    n_timesteps = 10
    batch = 4
    ny = nx = 4

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    src_file = tmp_path / "src.bin"
    src_file.write_bytes(b"\x00")  # existence is all the loop checks; load is mocked
    fname_out = tmp_path / "out.zarr"

    columns = ["H", "QX", "QY", "MH"]
    df_outputs = pd.DataFrame(
        {col: [src_file] * n_timesteps for col in columns},
        index=list(range(n_timesteps)),
    )

    inst = _build_synthetic_post_processing(
        fname_out=fname_out, raw_dir=raw_dir, batch_timesteps=batch, ny=ny, nx=nx
    )
    _patch_loaders(monkeypatch, df_outputs=df_outputs, ny=ny, nx=nx)

    # Count actual zarr write/append operations (one per flushed batch).
    to_zarr_modes = []
    orig_to_zarr = xr.Dataset.to_zarr

    def _counting_to_zarr(self, *args, **kwargs):
        to_zarr_modes.append(kwargs.get("mode"))
        return orig_to_zarr(self, *args, **kwargs)

    monkeypatch.setattr(xr.Dataset, "to_zarr", _counting_to_zarr)

    inst._export_TRITONSWMM_TRITON_outputs(verbose=False)

    expected_appends = math.ceil(n_timesteps / batch)
    assert len(to_zarr_modes) == expected_appends, (
        f"expected {expected_appends} batched zarr writes (ceil({n_timesteps}/{batch})), "
        f"got {len(to_zarr_modes)}: {to_zarr_modes}"
    )
    # First write creates the store, the rest append.
    assert to_zarr_modes[0] == "w"
    assert all(m == "a" for m in to_zarr_modes[1:])

    # No data loss: every timestep present in the store.
    ds = xr.open_zarr(fname_out)
    try:
        assert ds.sizes["timestep_min"] == n_timesteps
    finally:
        ds.close()


def test_write_timeseries_raises_when_no_valid_timesteps(tmp_path, monkeypatch):
    """SE F-I-2 guard: if every chunk is skipped (all source files missing) the
    zarr store is never created; the tail-flush guard must raise a diagnosable
    ProcessingError instead of letting zarr.consolidate_metadata fail cryptically
    on a nonexistent store."""
    n_timesteps = 5
    ny = nx = 4

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    missing_file = tmp_path / "does_not_exist.bin"  # never created on disk
    fname_out = tmp_path / "out.zarr"

    columns = ["H", "QX", "QY", "MH"]
    df_outputs = pd.DataFrame(
        {col: [missing_file] * n_timesteps for col in columns},
        index=list(range(n_timesteps)),
    )

    inst = _build_synthetic_post_processing(
        fname_out=fname_out, raw_dir=raw_dir, batch_timesteps=4, ny=ny, nx=nx
    )
    _patch_loaders(monkeypatch, df_outputs=df_outputs, ny=ny, nx=nx)

    with pytest.raises(ProcessingError, match="no valid timesteps to write"):
        inst._export_TRITONSWMM_TRITON_outputs(verbose=False)
    assert not fname_out.exists()
