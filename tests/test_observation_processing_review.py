"""Reviewer tests for observation-point series. Every test builds in memory or in a tiny ``tmp_path`` zarr.

Three of these pin gaps the review found (they are red on the reviewed tip ``844d1ba9``), and two pin readings the
review verified (green). Each carries its class and what would kill it.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import xarray as xr


def _triton_observation_rows(
    start: Fraction, last_obs: Fraction, end: Fraction, dt: Fraction, interval: Fraction, rows: list, *, stop=None
):
    """The ``simtime`` column one TRITON run appends, on the schedule of ``triton.h`` at ``faba2232``.

    Initial write at the run's start (``triton.h:2223``); a periodic write at the first step with
    ``simtime - last_print_obs_time >= print_observation`` and then ``last_print_obs_time += print_observation``
    (``:2286-2288``); the final write at the end (``:2368``). ``last_print_obs_time`` starts at the run's own start
    time (``:2255``), which is the checkpoint time on a resume. The gridded print (the checkpoint) is taken at the
    first step at or after each multiple of the interval, which is what the cfg writer records as ``sim_start_time``.
    Exact fractions, so no float edge decides a step.
    """
    t, last, checkpoints = start, last_obs, []
    rows.append(float(t))
    while t < end:
        t = min(t + dt, end)
        if stop is not None and t > stop:
            return checkpoints
        if t - last >= interval:
            last += interval
            rows.append(float(t))
        if t >= interval * (len(checkpoints) + 1):
            checkpoints.append(t)
    rows.append(float(t))
    return checkpoints


def test_a_resume_whose_rebased_schedule_lands_on_other_steps_is_read():
    """Property: the file of an interrupted-and-resumed run is read, taking the resumed run's rows.

    TRITON sets ``last_print_obs_time = simtime`` at the start of every run (``triton.h:2255``), so a resume from a
    checkpoint taken at 100.2 s fires its periodic writes at the first step at or after 150.2 s, not 150 s. The
    uninterrupted schedule wrote a row at 150.0 s before the kill and the resumed one writes 150.3 s: two distinct
    ``simtime`` values for the same period, which a keep-last-by-equal-``simtime`` rule cannot merge.
    Class: NEW CAPABILITY. Plausible wrong implementation: the reviewed one (dedupe by exact ``simtime``), which
    refuses a valid resumed file with "7 distinct row(s)" where 6 are expected. Passes under a second correct
    implementation: a rule that treats a ``simtime`` that goes backwards as a restart and keeps the rows after it.
    Killed by: any reader that needs the restarted rows to repeat the earlier ``simtime`` values exactly.
    """
    from hhemt.process_simulation import nominal_observation_times

    dt, interval, end = Fraction(3, 10), Fraction(50), Fraction(210)
    before: list[float] = []
    checkpoints = _triton_observation_rows(Fraction(0), Fraction(0), end, dt, interval, before, stop=Fraction(160))
    checkpoint = checkpoints[1]  # the checkpoint near 100 s; the run was killed at 160 s, after a write at 150.0 s
    after: list[float] = []
    _triton_observation_rows(checkpoint, checkpoint, end, dt, interval, after)
    rows = np.array(before + after)
    assert before[-1] == 150.0 and after[:2] == [100.2, 150.3], "the schedule model is the one the source sets"
    kept, nominal = nominal_observation_times(rows, 50.0, 210.0, "H")
    assert nominal.tolist() == [0.0, 50.0, 100.0, 150.0, 200.0, 210.0]
    assert rows[kept].tolist() == [0.0, 50.1, 100.2, 150.3, 200.4, 210.0], "the resumed run's rows supersede"


def test_a_store_holding_only_one_observation_variable_is_not_counted_as_written(tmp_path):
    """Property: the idempotence guard of the series writer needs all three observation variables, not the first.

    ``_store_holds_observation_series`` tests ``point_flood_depth`` alone, so a store whose write was killed after
    that variable and before ``point_unit_discharge_x`` and ``_y`` reads as written; the next pass returns without
    writing and consolidation then refuses by name with no way back short of deleting the store.
    Class: NEW CAPABILITY. Plausible wrong implementation: the reviewed one. Passes under a second correct
    implementation: a guard that tests every name of ``OBSERVATION_SERIES_FILES``.
    Killed by: a guard keyed on one variable.
    """
    from hhemt.process_simulation import OBSERVATION_SERIES_FILES, TRITONSWMM_sim_post_processing

    names = list(OBSERVATION_SERIES_FILES)
    partial = tmp_path / "partial.zarr"
    xr.Dataset({names[0]: (("point_elapsed_s", "obs_point"), np.zeros((2, 1)))}).to_zarr(partial)
    full = tmp_path / "full.zarr"
    xr.Dataset({n: (("point_elapsed_s", "obs_point"), np.zeros((2, 1))) for n in names}).to_zarr(full)
    assert TRITONSWMM_sim_post_processing._store_holds_observation_series(full) is True, "control: all three present"
    assert TRITONSWMM_sim_post_processing._store_holds_observation_series(partial) is False


def test_the_reprocess_delete_keeps_the_observation_summary_when_its_series_store_was_cleared(tmp_path):
    """Property: a group whose only source is a deleted per-scenario store is not removed by the reprocess delete.

    ``{family}/triton_observation/summary`` is derived from the per-scenario gridded store
    (``_observation_summary_slice``), which
    ``toggle_clear_scenario_timeseries_after_consolidation`` deletes and records in
    ``consolidated_observation_summary_cleared``. The delete's guard (``records_summary_clearing``) reads only the
    per-scenario SUMMARY record, and its glob ``*/*/summary`` reaches this group, so a kill between the two clearings
    followed by a reprocess removes a group no rebuild can reproduce. Class: NEW CAPABILITY.
    Plausible wrong implementation: the reviewed one. Passes under a second correct implementation: a guard that also
    reads the observation clearing fields, or a summary rebuilt from the kept ``timeseries`` group.
    Killed by: a glob that treats every ``summary`` branch as rebuildable.
    """
    from hhemt.delete_reprocess_zarr_runner import delete_summary_branches

    tree = tmp_path / "analysis_datatree.zarr"
    xr.Dataset().to_zarr(tree)
    for branch in ("summary", "timeseries"):
        xr.Dataset({"v": ("x", [1.0])}).to_zarr(tree, group=f"tritonswmm/triton_observation/{branch}", mode="a")
    xr.Dataset({"v": ("x", [1.0])}).to_zarr(tree, group="tritonswmm/triton/summary", mode="a")
    sim = tmp_path / "sims" / "event_index.0"
    sim.mkdir(parents=True)
    (sim / "log_tritonswmm.json").write_text(json.dumps({"consolidated_observation_summary_cleared": True}))
    delete_summary_branches(tree, tmp_path)
    assert not (tree / "tritonswmm" / "triton" / "summary").exists(), "control: a rebuildable summary is removed"
    assert (tree / "tritonswmm" / "triton_observation" / "timeseries").is_dir(), "control: the series is kept"
    assert (tree / "tritonswmm" / "triton_observation" / "summary").is_dir()


def test_the_observation_interval_defaults_to_the_reporting_step_which_is_triton_default():
    """Property: an unset interval reads the reporting step, the value TRITON's ``print_observation`` defaults to.

    The cfg template sets ``print_interval=${REPORTING_TSTEP_S}`` and TRITON defaults ``print_observation`` to that
    string, so the group axis must step at the reporting step when the field is None and at the field otherwise.
    Class: NEW CAPABILITY. Plausible wrong implementation: ``or`` replaced by a default of 0 or of the duration.
    Passes under a second correct implementation: reading the interval from the rendered cfg.
    Killed by: a different default or a swapped operand order.
    """
    from hhemt.processing_analysis import TRITONSWMM_analysis_post_processing as post

    def _interval(observation, reporting):
        cfg = SimpleNamespace(TRITON_observation_interval_s=observation, TRITON_reporting_timestep_s=reporting)
        proc = post.__new__(post)
        proc._analysis = SimpleNamespace(cfg_analysis=cfg)
        return proc._observation_interval_s()

    assert _interval(None, 60) == 60.0
    assert _interval(120, 60) == 120.0


def test_the_observation_groups_are_gated_on_the_point_file_not_on_the_timeseries_toggle():
    """Property: the unconditional switch reads ``observation_points`` of the SYSTEM config and nothing else.

    Class: NEW CAPABILITY. Plausible wrong implementation: reading the toggle, which the ruling says must not gate it.
    Killed by: gating on ``toggle_consolidate_timeseries``.
    """
    from hhemt.processing_analysis import TRITONSWMM_analysis_post_processing as post

    def _set(points, toggle):
        proc = post.__new__(post)
        system = SimpleNamespace(cfg_system=SimpleNamespace(observation_points=points))
        proc._analysis = SimpleNamespace(
            _system=system, cfg_analysis=SimpleNamespace(toggle_consolidate_timeseries=toggle)
        )
        return proc._observation_points_set()

    assert _set(Path("points.csv"), False) is True
    assert _set(None, True) is False
