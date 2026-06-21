"""Helpers for building deliberately-failing analyses (Iter 9 Phase 7).

Each helper takes a known-good analysis (typically from one of the cached
synth fixtures) and produces a corrupted clone that exercises a specific
failure mode of the new ``analysis_validation.validate_analysis()`` checks.

The clone strategy: shutil.copytree the entire ``system_directory`` (the root
dir holding the per-analysis sub-dirs + compilations + system inputs) to a
tmp_path location, write modified config YAMLs pointing at the copy, then
construct a fresh ``TRITONSWMM_system + TRITONSWMM_analysis`` from those.
Subsequent file mutations on the copy do not touch the cached fixture.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import yaml

if TYPE_CHECKING:
    from hhemt.analysis import TRITONSWMM_analysis


def prepare_clone_dir(cached_analysis: "TRITONSWMM_analysis", tmp_path: Path) -> dict:
    """Copy the cached analysis's system_directory to tmp_path; write modified
    YAML configs that point at the copy. Does NOT construct the analysis.

    Returns a dict with ``system_yaml``, ``analysis_yaml``, ``system_dir``,
    ``analysis_dir`` (all under tmp_path) so callers can mutate the on-disk
    state BEFORE constructing the analysis (which loads logs/state into
    memory at construction-time and would otherwise cache the unmutated state).
    """
    cached_sys = cached_analysis._system
    src_system_dir = Path(cached_sys.cfg_system.system_directory)
    src_system_yaml = src_system_dir / "system_config.yaml"
    src_analysis_yaml = src_system_dir / "analysis_config.yaml"
    if not src_system_yaml.exists() or not src_analysis_yaml.exists():
        raise FileNotFoundError(
            f"Expected system_config.yaml + analysis_config.yaml in {src_system_dir} — "
            f"got system={src_system_yaml.exists()}, analysis={src_analysis_yaml.exists()}"
        )

    dst_system_dir = tmp_path / "system"
    shutil.copytree(src_system_dir, dst_system_dir)

    # Write the modified master configs INSIDE dst_system_dir (NOT at tmp_path
    # root). The source layout has system_config.yaml + analysis_config.yaml
    # living inside system_directory; a sensitivity sub derives its analysis-level
    # model-log dir as master_analysis_cfg_yaml.parent / "logs" / "sims"
    # (run_simulation.py:_analysis_level_model_logfile). Hoisting the master config
    # one level out of system_directory makes model_run_completed() look in
    # tmp_path/logs/sims/ while the cloned logs live in tmp_path/system/logs/sims/,
    # so reprocess(start_with="consolidate") spuriously fail-fasts on _scenarios_not_run.
    sys_cfg_dict = yaml.safe_load(src_system_yaml.read_text())
    sys_cfg_dict["system_directory"] = str(dst_system_dir)
    dst_system_yaml = dst_system_dir / "system_config.yaml"
    dst_system_yaml.write_text(yaml.safe_dump(sys_cfg_dict, sort_keys=False))

    ana_cfg_dict = yaml.safe_load(src_analysis_yaml.read_text())
    dst_analysis_yaml = dst_system_dir / "analysis_config.yaml"
    dst_analysis_yaml.write_text(yaml.safe_dump(ana_cfg_dict, sort_keys=False))

    analysis_id = ana_cfg_dict["analysis_id"]
    return {
        "system_yaml": dst_system_yaml,
        "analysis_yaml": dst_analysis_yaml,
        "system_dir": dst_system_dir,
        "analysis_dir": dst_system_dir / analysis_id,
        "analysis_id": analysis_id,
    }


def construct_analysis_from_paths(paths: dict) -> "TRITONSWMM_analysis":
    """Construct fresh TRITONSWMM_system + TRITONSWMM_analysis from cloned configs."""
    from hhemt.analysis import TRITONSWMM_analysis
    from hhemt.system import TRITONSWMM_system

    system = TRITONSWMM_system(paths["system_yaml"])
    analysis = TRITONSWMM_analysis(paths["analysis_yaml"], system)
    return analysis


def clone_analysis_to_tmp(cached_analysis: "TRITONSWMM_analysis", tmp_path: Path) -> "TRITONSWMM_analysis":
    """Convenience: clone + construct in one shot (no on-disk mutation between).

    For tests that need pre-construction mutation, use ``prepare_clone_dir``
    + ``construct_analysis_from_paths`` directly.
    """
    paths = prepare_clone_dir(cached_analysis, tmp_path)
    return construct_analysis_from_paths(paths)


# ---------------------------------------------------------------------------
# Failure-injection helpers (operate on the cloned analysis's on-disk state)
# ---------------------------------------------------------------------------


def _mutate_log_field(log_path: Path, field_name: str, new_value) -> None:
    """Mutate a single field in a JSON log file (TRITONSWMM_*_log schema).

    The schema wraps each field as ``{"value": ..., "set_at": ...}``. This
    helper preserves the wrap; it only replaces the inner ``value``.
    """
    log = json.loads(log_path.read_text())
    if field_name in log and isinstance(log[field_name], dict) and "value" in log[field_name]:
        log[field_name]["value"] = new_value
    else:
        log[field_name] = new_value
    log_path.write_text(json.dumps(log, indent=2, default=str))


def inject_scenario_setup_failure(scenario_dir: Path) -> None:
    """Mutate scenario_prep_log.json to set scenario_creation_complete = False
    AND delete the entire scenario directory (the most reliable way to make
    `analysis._scenarios_not_created` flag this scenario, since the analysis
    re-detects scenario state from on-disk presence at construction)."""
    if scenario_dir.exists():
        shutil.rmtree(scenario_dir)


def inject_simulation_run_failure(scenario_dir: Path, model_type: str) -> None:
    """Mutate log_{model_type}.json to set simulation_completed = False AND
    delete the model output directory so disk-state detection sees the run
    as incomplete."""
    log_path = scenario_dir / f"log_{model_type}.json"
    if log_path.exists():
        _mutate_log_field(log_path, "simulation_completed", False)
        # Also clear timeseries-written flags so the timeseries check fails
        for fld in [
            "performance_timeseries_written",
            "TRITON_timeseries_written",
            "SWMM_node_timeseries_written",
            "SWMM_link_timeseries_written",
        ]:
            try:
                _mutate_log_field(log_path, fld, False)
            except Exception:
                pass
    # Delete the model's output dir
    out_dir = scenario_dir / f"out_{model_type}"
    if out_dir.exists():
        shutil.rmtree(out_dir)


def inject_timeseries_failure(scenario_dir: Path) -> None:
    """Delete the processed/ directory so timeseries-processed checks fail."""
    processed_dir = scenario_dir / "processed"
    if processed_dir.exists():
        shutil.rmtree(processed_dir)


def inject_resource_mismatch_omp(csv_path: Path, scenario_substr: str, expected_omp: int, actual_omp: int) -> None:
    """Edit scenario_status.csv row to set actual_omp_threads != n_omp_threads."""
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    mask = df["scenario_directory"].astype(str).str.contains(scenario_substr)
    if not mask.any():
        return
    df.loc[mask, "n_omp_threads"] = expected_omp
    df.loc[mask, "actual_omp_threads"] = actual_omp
    # Ensure the row also appears as run_completed=True so the validator
    # actually checks it (the validator skips rows where run_completed=False).
    df.loc[mask, "run_completed"] = True
    df.to_csv(csv_path, index=False)


def inject_compilation_failure(system, model_type: str) -> None:
    """Mutate system_log.json so compilation_*_successful = False for model_type."""
    sys_log = Path(system.cfg_system.system_directory) / "system_log.json"
    if not sys_log.exists():
        return
    key = {
        "tritonswmm": "compilation_successful",
        "triton": "compilation_triton_only_successful",
        "swmm": "compilation_swmm_successful",
    }[model_type]
    _mutate_log_field(sys_log, key, False)


def inject_analysis_log_flag_false(analysis_dir: Path, flag_name: str) -> None:
    """Mutate analysis_dir/log.json top-level flag (e.g., all_scenarios_created)."""
    log_path = analysis_dir / "log.json"
    if not log_path.exists():
        return
    _mutate_log_field(log_path, flag_name, False)


def inject_summary_file_missing(file_or_dir) -> None:
    """Delete a summary file/dir so the file-existence check fails."""
    if file_or_dir is None:
        return
    p = Path(file_or_dir)
    if p.is_dir():
        shutil.rmtree(p)
    elif p.exists():
        p.unlink()


# ---------------------------------------------------------------------------
# Composite failure-injection scenarios (one per fixture)
# ---------------------------------------------------------------------------


def inject_multi_sim_failures_at_paths(paths: dict) -> None:
    """Inject every failure mode into a synth_multi_sim clone (3 scenarios).

    Operates on the cloned dir BEFORE constructing the analysis (so that
    analysis-construction loads the mutated state into memory rather than
    the cached pre-mutation state).

    Coverage: 1 successful + 2 stage failures + 2 system failures:
    - event_index.0: scenario setup failure (A1) — mutate scen.log
    - event_index.1: simulation run failure (A2) — mutate log_tritonswmm.json
    - event_index.2: timeseries processing failure (A3) — delete processed/
    - System: TRITON-SWMM compilation marked failed (S1) — mutate system_log.json
    - System: an analysis-summary file deleted (S2)
    """
    analysis_dir = paths["analysis_dir"]
    system_dir = paths["system_dir"]
    sims_dir = analysis_dir / "sims"
    scenario_dirs = sorted([d for d in sims_dir.iterdir() if d.is_dir()])
    assert len(scenario_dirs) >= 3, f"expected ≥3 scenario dirs, found {len(scenario_dirs)}"

    inject_scenario_setup_failure(scenario_dirs[0])
    inject_simulation_run_failure(scenario_dirs[1], model_type="tritonswmm")
    inject_timeseries_failure(scenario_dirs[2])
    # Flip analysis-level aggregate flags (the validators read these as the
    # boolean indicator; per-scenario details still derive from the per-scen
    # logs / processed dirs we mutated above).
    inject_analysis_log_flag_false(analysis_dir, "all_scenarios_created")
    inject_analysis_log_flag_false(analysis_dir, "all_sims_run")
    inject_analysis_log_flag_false(analysis_dir, "all_TRITONSWMM_performance_timeseries_processed")
    inject_analysis_log_flag_false(analysis_dir, "all_TRITON_timeseries_processed")
    inject_analysis_log_flag_false(analysis_dir, "all_SWMM_timeseries_processed")
    # System log mutation (compilation flag)
    sys_log = system_dir / "system_log.json"
    if sys_log.exists():
        _mutate_log_field(sys_log, "compilation_successful", False)
    # Delete the master DataTree (Option B's canonical artifact). Under
    # the legacy two-tier consolidation this helper deleted per-mode
    # flat zarrs; those no longer exist post-Option-B.
    target = analysis_dir / "analysis_datatree.zarr"
    if target.exists():
        inject_summary_file_missing(target)


def inject_sensitivity_failures_at_paths(paths: dict) -> None:
    """Inject failures across sub-analyses of a synth_sensitivity clone.

    Coverage:
    - sa_<first> / event_*: scenario setup failure (A1)
    - sa_<second> / event_*: simulation run failure (A2)
    - sa_<third> / event_*: timeseries processing failure (A3)
    - System: SWMM compilation marked failed (S1)
    - System: sensitivity_datatree.zarr deleted (S2)
    """
    analysis_dir = paths["analysis_dir"]
    system_dir = paths["system_dir"]
    sub_root = analysis_dir / "subanalyses"
    if not sub_root.exists():
        # Some sensitivity layouts use a different root; try common alternatives.
        for alt in ["sub_analyses", "sa", "ensemble"]:
            if (analysis_dir / alt).exists():
                sub_root = analysis_dir / alt
                break
    sa_dirs = sorted([d for d in sub_root.iterdir() if d.is_dir()])
    assert len(sa_dirs) >= 3, f"expected ≥3 sub-analysis dirs in {sub_root}, found {len(sa_dirs)}"

    def _first_scenario_in_sa(sa_dir):
        sims = sa_dir / "sims"
        return next(iter(sorted(d for d in sims.iterdir() if d.is_dir())))

    inject_scenario_setup_failure(_first_scenario_in_sa(sa_dirs[0]))
    inject_simulation_run_failure(_first_scenario_in_sa(sa_dirs[1]), model_type="triton")
    if len(sa_dirs) >= 3:
        inject_timeseries_failure(_first_scenario_in_sa(sa_dirs[2]))
    # Flip per-sub-analysis log flags so the validators see them as failed
    for i, sa_dir in enumerate(sa_dirs[:3]):
        if i == 0:
            inject_analysis_log_flag_false(sa_dir, "all_scenarios_created")
        elif i == 1:
            inject_analysis_log_flag_false(sa_dir, "all_sims_run")
        elif i == 2:
            inject_analysis_log_flag_false(sa_dir, "all_TRITONSWMM_performance_timeseries_processed")
            inject_analysis_log_flag_false(sa_dir, "all_TRITON_timeseries_processed")
            inject_analysis_log_flag_false(sa_dir, "all_SWMM_timeseries_processed")
    sys_log = system_dir / "system_log.json"
    if sys_log.exists():
        _mutate_log_field(sys_log, "compilation_swmm_successful", False)
    # Delete sensitivity_datatree.zarr if present
    sens_zarr = analysis_dir / "sensitivity_datatree.zarr"
    if sens_zarr.exists():
        inject_summary_file_missing(sens_zarr)
