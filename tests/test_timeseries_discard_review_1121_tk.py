"""Reviewer tests for change 1121-tk: the discard of per-scenario series stores with consolidation off.

Non-solver and fixture-free: a ``TRITONSWMM_sim_post_processing`` is built without its constructor, over stub
config / scenario / log objects and real stores made of empty directories under ``tmp_path``.

Each test states the sentence of the criterion it grades. Two of them are expected RED on commit 17de6786 (see
the Findings of the build-reviewer scratch): they assert what the developer's ruling G3 and the change's own
documentation say, not what the code does.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from hhemt.process_simulation import TRITONSWMM_sim_post_processing as Proc

_ALL_MODELS = ["tritonswmm", "triton", "swmm"]


class _Field:
    def __init__(self):
        self.value = None

    def set(self, v):
        self.value = v

    def get(self):
        return self.value


class _Log:
    def __init__(self):
        self.full_TRITON_timeseries_cleared = _Field()
        self.full_SWMM_timeseries_cleared = _Field()
        self.raw_SWMM_binaries_reclaimed = _Field()
        self.coupled_rpt_truncated = _Field()

    @contextmanager
    def deferred_writes(self):
        yield


def _proc(tmp_path: Path, model: str, *, observation_points=None, clear_raw="all", remove=("timeseries",)):
    """A post-processing object over real (empty) series stores and their ``.zarr.done`` flags for ``model``."""
    names = Proc._TIMESERIES_ATTRS_BY_MODEL[model]
    scen_paths = SimpleNamespace(sim_folder=tmp_path, out_tritonswmm=None, swmm_hydraulics_rpt=None)
    for attr in names:
        store = tmp_path / "processed" / f"{attr}_tseries.zarr"
        store.mkdir(parents=True)
        (store / "zarr.json").write_text("{}")
        flag = store.with_name(store.name + ".done")
        flag.write_text("ok")
        setattr(scen_paths, attr, store)
    cfg = SimpleNamespace(
        toggle_consolidate_timeseries=False,
        toggle_clear_scenario_timeseries_after_consolidation=False,
        clear_raw=clear_raw,
        remove_after_processing=list(remove),
    )
    proc = object.__new__(Proc)
    proc._analysis = SimpleNamespace(
        cfg_analysis=cfg,
        _get_enabled_model_types=lambda: [model],
        _system=SimpleNamespace(cfg_system=SimpleNamespace(observation_points=observation_points)),
    )
    proc._scenario = SimpleNamespace(event_iloc=0)
    proc.scen_paths = scen_paths
    proc.log = _Log()
    return proc, scen_paths, names


@pytest.mark.parametrize("model", _ALL_MODELS)
def test_every_series_store_of_the_model_is_discarded(tmp_path, model):
    """Criterion (4): every store the consolidation reads (nine over the three models) is in the drop set."""
    proc, scen_paths, names = _proc(tmp_path, model)
    proc.remove_after_processing(model_type=model, which="both")
    for attr in names:
        assert not getattr(scen_paths, attr).exists(), attr


def test_the_nine_series_attrs_equal_the_consolidation_modes():
    """Criterion (4): the drop table and the consolidation's mode table name the same nine path attributes."""
    from hhemt.processing_analysis import TRITONSWMM_analysis_post_processing as _P  # noqa: F401  # name check below

    drop = {a for names in Proc._TIMESERIES_ATTRS_BY_MODEL.values() for a in names}
    consolidated = {a for k, a in _P._TIMESERIES_MODE_CONFIG.items() if "observation" not in k}
    assert drop == consolidated


def test_system_attr_detection_is_exact_over_the_drop_table():
    """Criterion (4): only the SWMM-side series read as SWMM side; no TRITON-side name contains ``_system_``."""
    swmm_side = {a for names in Proc._TIMESERIES_ATTRS_BY_MODEL.values() for a in names if Proc._attr_is_swmm_side(a)}
    assert swmm_side == {
        "output_tritonswmm_node_time_series",
        "output_tritonswmm_link_time_series",
        "output_swmm_only_node_time_series",
        "output_swmm_only_link_time_series",
        "output_swmm_only_system_time_series",
    }


@pytest.mark.parametrize("model", _ALL_MODELS)
def test_no_completion_flag_is_left_beside_a_discarded_store(tmp_path, model):
    """Ruling G3 (no time series artifact remains) and the consolidation's own rule in
    ``_delete_cleared_stores`` (a flag left beside a deleted store asserts a completion the directory no longer
    shows). EXPECTED RED on 17de6786: ``_remove_reclaimed`` deletes the store path only, so ``*_tseries.zarr.done``
    stays, and the new test's predicate (``suffix in (.zarr, .nc)``) cannot see it."""
    proc, scen_paths, names = _proc(tmp_path, model)
    proc.remove_after_processing(model_type=model, which="both")
    left = sorted(p.name for p in (tmp_path / "processed").glob("*tseries*"))
    assert left == [], left


def test_runner_keeps_every_series_store_under_observation_points(tmp_path):
    """Ruling G18 (defence in depth): the runner reads the persisted config without preflight, so when the refused
    pair reaches it anyway, it deletes NO series store (the observation groups read the gridded one and no step
    clears it afterwards), and does not touch the completion flags either."""
    proc, scen_paths, names = _proc(tmp_path, "tritonswmm", observation_points=object())
    proc.remove_after_processing(model_type="tritonswmm", which="both")
    assert all(getattr(scen_paths, a).exists() for a in names)
    assert len(list((tmp_path / "processed").glob("*.done"))) == len(names)


_SYS = {
    "observation_points": "points.csv",
    "toggle_triton_model": False,
    "toggle_tritonswmm_model": True,
    "toggle_swmm_model": False,
}
_ANA = {"toggle_consolidate_timeseries": False, "clear_raw": "all", "remove_after_processing": ["timeseries"]}


def _refusals(system, analysis):
    from hhemt.validation import ValidationResult, _validate_timeseries_discard_observation_points

    result = ValidationResult(context="review")
    _validate_timeseries_discard_observation_points(SimpleNamespace(**system), SimpleNamespace(**analysis), result)
    return list(result.errors)


def test_preflight_refuses_the_full_conjunction_and_only_it():
    """Ruling G18: refused for the full conjunction; each single missing leg (five legs: observation_points,
    toggle off, clear_raw covering, list form, naming timeseries) lifts the refusal."""
    assert len(_refusals(_SYS, _ANA)) == 1
    legs = [
        ({"observation_points": None}, {}),
        ({}, {"toggle_consolidate_timeseries": True}),
        ({}, {"clear_raw": "none"}),
        ({}, {"clear_raw": ["triton"]}),
        ({}, {"remove_after_processing": "all"}),
        ({}, {"remove_after_processing": ["coupled_rpt"]}),
    ]
    for sys_over, ana_over in legs:
        assert _refusals({**_SYS, **sys_over}, {**_ANA, **ana_over}) == [], (sys_over, ana_over)


def test_the_refusal_follows_the_runners_predicate(monkeypatch):
    """Drift guard: the refusal calls ``process_simulation.timeseries_discard_arm`` at call time, so a change of the
    runner's predicate changes the refusal with it, in both directions."""
    import hhemt.process_simulation as ps

    monkeypatch.setattr(ps, "timeseries_discard_arm", lambda cfg, models: False)
    assert _refusals(_SYS, _ANA) == []
    monkeypatch.setattr(ps, "timeseries_discard_arm", lambda cfg, models: True)
    assert len(_refusals(_SYS, {**_ANA, "toggle_consolidate_timeseries": True})) == 1


def test_the_refusal_builds_the_enabled_list_as_the_analysis_does(monkeypatch):
    """The model list handed to the predicate equals ``TRITONSWMM_analysis._get_enabled_model_types`` on the same
    toggles, in the same order (validation's own list is built from the same three system toggles)."""
    import hhemt.process_simulation as ps
    from hhemt.analysis import TRITONSWMM_analysis

    seen = []
    monkeypatch.setattr(ps, "timeseries_discard_arm", lambda cfg, models: seen.append(list(models)) or False)
    for toggles in ((True, True, True), (False, True, False), (True, False, True), (False, False, False)):
        sys_cfg = dict(
            _SYS, toggle_triton_model=toggles[0], toggle_tritonswmm_model=toggles[1], toggle_swmm_model=toggles[2]
        )
        _refusals(sys_cfg, _ANA)
        analysis = object.__new__(TRITONSWMM_analysis)
        analysis._system = SimpleNamespace(cfg_system=SimpleNamespace(**sys_cfg))
        assert seen[-1] == analysis._get_enabled_model_types(), toggles


def test_preflight_validate_runs_the_refusal():
    """The refusal is wired into ``preflight_validate`` (one call, on the master configs)."""
    import inspect

    from hhemt.validation import preflight_validate

    assert "_validate_timeseries_discard_observation_points(cfg_system, cfg_analysis, result)" in inspect.getsource(
        preflight_validate
    )


@pytest.mark.parametrize(
    ("clear_raw", "remove"),
    [("none", ["timeseries"]), (["triton"], ["timeseries"]), ("all", "all"), ("all", ["coupled_rpt"])],
)
def test_the_series_stay_unless_the_conjunction_holds(tmp_path, clear_raw, remove):
    """Criterion (1): clear_raw not covering every enabled model, the blanket ``all`` and a list without
    ``timeseries`` each keep every series store (the config in the stub enables ``tritonswmm`` only)."""
    proc, scen_paths, names = _proc(tmp_path, "tritonswmm", clear_raw=clear_raw, remove=remove)
    proc._analysis.cfg_analysis.remove_after_processing = remove
    proc.remove_after_processing(model_type="tritonswmm", which="both")
    assert all(getattr(scen_paths, a).exists() for a in names)


def test_consolidation_on_never_discards_in_the_runner(tmp_path):
    """Criterion (1): toggle True keeps the series in the runner (the consolidation clears them after read-back)."""
    proc, scen_paths, names = _proc(tmp_path, "tritonswmm")
    proc._analysis.cfg_analysis.toggle_consolidate_timeseries = True
    proc.remove_after_processing(model_type="tritonswmm", which="both")
    assert all(getattr(scen_paths, a).exists() for a in names)


# ---------------------------------------------------------------------------------------------------------------
# Sensitivity members (round 4): minimal real ``analysis_config`` objects, stub system configs, no builder.
# ---------------------------------------------------------------------------------------------------------------
_MEMBER_SYSTEM = {
    "observation_points": "points.csv",
    "toggle_triton_model": False,
    "toggle_tritonswmm_model": True,
    "toggle_swmm_model": False,
}
# The master states the election with consolidation on and clear_raw all: not refused itself.
_MEMBER_MASTER = {"clear_raw": "all", "remove_after_processing": ["timeseries", "coupled_rpt"]}


def _member_errors(tmp_path, master_over, rows, systems=None):
    import pandas as pd

    from hhemt.config.analysis import analysis_config
    from hhemt.validation import ValidationResult, _validate_members_timeseries_discard_observation_points
    from tests.test_defaults_clear_everything import _cfg_dict

    master = analysis_config.model_validate(_cfg_dict(tmp_path, **master_over))
    df = pd.DataFrame.from_dict(rows, orient="index")
    systems = systems or {str(m): SimpleNamespace(**_MEMBER_SYSTEM) for m in rows}
    result = ValidationResult(context="review")
    _validate_members_timeseries_discard_observation_points(
        df, SimpleNamespace(**_MEMBER_SYSTEM), master, systems, result
    )
    return list(result.errors)


def test_a_row_sweeping_consolidation_off_over_an_unstated_clearing_value_is_still_judged(tmp_path):
    """Round-4 point (2): the master's dump carries the RESOLVED clearing value (True, from consolidation on). The
    helper merges the row over that dump WITHOUT the ``None`` reset ``_create_members`` applies; the check is only
    sound if that merge does not raise (the after-validator only resolves ``None``, it never refuses). A raise would
    be swallowed and the refused member would pass. The row below must be refused, naming the member."""
    errors = _member_errors(
        tmp_path,
        _MEMBER_MASTER,
        {"0": {"analysis.toggle_consolidate_timeseries": False}, "1": {"analysis.toggle_consolidate_timeseries": True}},
    )
    assert len(errors) == 1 and "member_id=0" in errors[0].message and "row[0]" in errors[0].field


def test_a_row_sweeping_clear_raw_into_the_combination_is_refused(tmp_path):
    """The clear_raw leg swept by a row: master keeps raws (``none``) with consolidation off, row 0 sets ``all``."""
    master = {"toggle_consolidate_timeseries": False, "clear_raw": "none", "remove_after_processing": ["timeseries"]}
    errors = _member_errors(tmp_path, master, {"0": {"analysis.clear_raw": "all"}, "1": {"analysis.clear_raw": "none"}})
    assert [e.field for e in errors] == ["sensitivity_analysis.row[0]"]


def test_the_system_leg_is_judged_per_member(tmp_path):
    """Observation points on member 0's own system only: member 0 is refused, member 1 (no points) is not."""
    rows = {
        "0": {"analysis.toggle_consolidate_timeseries": False},
        "1": {"analysis.toggle_consolidate_timeseries": False},
    }
    systems = {
        "0": SimpleNamespace(**_MEMBER_SYSTEM),
        "1": SimpleNamespace(**{**_MEMBER_SYSTEM, "observation_points": None}),
    }
    errors = _member_errors(tmp_path, _MEMBER_MASTER, rows, systems)
    assert [e.field for e in errors] == ["sensitivity_analysis.row[0]"]


def test_a_row_whose_cell_does_not_form_a_config_is_skipped_not_raised(tmp_path):
    """Round-4 point (3), pinned: an invalid cell (here ``clear_raw`` = bogus) makes the merge fail and the member is
    skipped silently. Acceptable because ``_create_members`` runs the same ``model_validate`` on the same values and
    cannot build that member, so it never runs; the cell is reported by the member's own validators."""
    errors = _member_errors(
        tmp_path,
        _MEMBER_MASTER,
        {"0": {"analysis.toggle_consolidate_timeseries": False, "analysis.clear_raw": "bogus"}},
    )
    assert errors == []


def test_the_extracted_overlay_helper_keeps_the_inline_loops_behaviour():
    """Round-4 point (1): empty cells skipped, hpc alias resolved, analysis prefix stripped, in row order, and an
    unknown column raises the same ConfigurationError the inline loop raised."""
    import pandas as pd

    from hhemt.exceptions import ConfigurationError
    from hhemt.sensitivity_analysis import _member_analysis_overlay_cells

    row = pd.Series(
        {
            "analysis.clear_raw": "all",
            "analysis.toggle_consolidate_timeseries": float("nan"),
            "hpc.partition": "standard",
            "hpc.setup_partition": "standard2",
        }
    )
    out = _member_analysis_overlay_cells(row)
    assert out == {
        "clear_raw": "all",
        "hpc_ensemble_partition": "standard",
        "hpc_setup_and_analysis_processing_partition": "standard2",
    }
    assert list(out) == ["clear_raw", "hpc_ensemble_partition", "hpc_setup_and_analysis_processing_partition"]
    with pytest.raises(ConfigurationError) as exc:
        _member_analysis_overlay_cells(pd.Series({"system.crs": "EPSG:1"}))
    assert exc.value.field == "sensitivity_analysis.unknown_column"
