"""The default configuration keeps the consolidated series and clears everything else.

Problem this file solves: with nothing configured, a run must keep only the consolidated store and remove every raw and
intermediate output, and the machinery that does so (the removal keyed on three switches, the ``timeseries`` class
discharged by consolidation, the hydrograph rebuild from the capture) must hold at that default. Each test below pins
one of those properties against the real code, with a must-fire control that varies the property under test.

No solver, no Snakemake and no synthetic analysis is built, so every test is laptop-admissible. The
completed-default-run rows (a real run leaves no ``sims/`` child) are in the compile tier on Rivanna and are NOT
covered here.
"""

from __future__ import annotations

import contextlib
import types
from pathlib import Path

import pytest
import xarray as xr

from hhemt.config.analysis import analysis_config
from hhemt.exceptions import ProcessingError
from hhemt.process_simulation import TRITONSWMM_sim_post_processing as _P
from hhemt.process_simulation import reclaim_scenario_scoped_classes
from hhemt.scenario_records import removal_switches_on
from hhemt.swmm_runoff_modeling import hydrograph_outputs_gate

_FOUR_KEYS = (
    "toggle_consolidate_timeseries",
    "toggle_clear_scenario_timeseries_after_consolidation",
    "clear_raw",
    "remove_after_processing",
)

# The five-class list that was the default before D24 (a); a config naming it explicitly must still load as named.
_OLD_REMOVE_DEFAULT = ["raw_swmm_binaries", "coupled_rpt", "prep_inputs", "hydrographs", "standalone_rpt"]


def _touch(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("x\n")
    return p


def _cfg_dict(tmp_path: Path, **over) -> dict:
    """A minimal analysis config carrying NONE of the four default-bearing keys unless ``over`` names one."""
    base = {
        "analysis_id": "analysis_01",
        "hhemt_sha": "0" * 40,  # schema-only; never compared to a running identity
        "weather_event_indices": ["event_id"],
        "weather_timeseries": str(_touch(tmp_path / "inputs" / "weather.nc")),
        "weather_time_series_timestep_dimension_name": "timestep",
        "weather_time_series_spatial_mean_rainfall_datavar": "RG_synth",
        "rainfall_units": "mm/hr",
        "run_mode": "serial",
        "multi_sim_run_method": "local",
        "toggle_sensitivity_analysis": False,
        "toggle_storm_tide_boundary": False,
        "weather_events_to_simulate": str(_touch(tmp_path / "inputs" / "events.csv")),
        "manhole_diameter": 1.0,
        "manhole_loss_coefficient": 0.1,
        "hydraulic_timestep_s": 1.0,
        "TRITON_reporting_timestep_s": 60,
        "open_boundaries": 1,
        "report": {},
        "force_rerun": "none",
    }
    base.update(over)
    return base


# ---------------------------------------------------------------------------
# D24 (a): the four defaults.
# ---------------------------------------------------------------------------
def test_an_analysis_config_with_none_of_the_four_keys_resolves_the_clearing_state(tmp_path):
    """Property: a config naming none of the four keys loads with consolidation on, clearing on, ``clear_raw``
    ``"all"`` and ``remove_after_processing`` ``"all"``, and that state is the one in which the directories go.

    Class: NEW CAPABILITY (the defaults changed). Plausible wrong implementation: changing the field default but
    leaving the clearing switch resolved False (it resolves from the consolidation toggle at load), which this reads
    as clearing False. Why required: the removal is keyed on the conjunction of three switches, so one of them
    resolving the old way keeps every directory with no error. Reachability: the real model validator. Must-fire
    control: each key set to its old default resolves to that value, and the removal key goes False for each.
    Killed by: any of the four defaults reverting, or the resolver reading the field default instead of the value.
    Passes under a second correct implementation that resolves the clearing switch in a different validator.
    """
    for key in _FOUR_KEYS:
        assert key not in _cfg_dict(tmp_path)
    cfg = analysis_config.model_validate(_cfg_dict(tmp_path))
    assert cfg.toggle_consolidate_timeseries is True
    assert cfg.toggle_clear_scenario_timeseries_after_consolidation is True
    assert cfg.clear_raw == "all"
    assert cfg.remove_after_processing == "all"
    assert removal_switches_on(cfg, ["tritonswmm"]) is True


@pytest.mark.parametrize(
    "key,old",
    [
        ("toggle_consolidate_timeseries", False),
        ("clear_raw", "none"),
        ("remove_after_processing", _OLD_REMOVE_DEFAULT),
        ("remove_after_processing", "none"),
    ],
)
def test_each_key_set_to_its_old_default_resolves_to_that_value_and_keeps_the_directories(tmp_path, key, old):
    """Must-fire control of the test above: the all-defaults state is not a constant of the loader.

    Varies the PROPERTY (the value a key resolves to), not the token: a loader that ignored the key would read the
    new default here and fail. With consolidation off the clearing switch resolves False as well, which is the
    resolver's rule and is why the directories stay.
    """
    cfg = analysis_config.model_validate(_cfg_dict(tmp_path, **{key: old}))
    assert getattr(cfg, key) == old
    assert removal_switches_on(cfg, ["tritonswmm"]) is False
    if key == "toggle_consolidate_timeseries":
        assert cfg.toggle_clear_scenario_timeseries_after_consolidation is False


def test_the_module_constant_for_the_retired_default_is_gone():
    """D24 (a): ``"all"`` is the literal default, so the five-class constant must not linger as a second source."""
    import hhemt.config.analysis as config_analysis

    assert not hasattr(config_analysis, "_DEFAULT_REMOVE_AFTER_PROCESSING")
    assert analysis_config.model_fields["remove_after_processing"].default == "all"
    # Control: the module still carries its other defaults, so the attribute check is not failing on a bad import.
    assert hasattr(config_analysis, "DEFAULT_FLOOD_VELOCITY_MIN_DEPTH_M")


def test_the_four_field_descriptions_state_the_defaults():
    """The descriptions carry the defaults and the three-switch conjunction."""
    fields = analysis_config.model_fields
    assert "Defaults to True" in fields["toggle_consolidate_timeseries"].description
    assert "Defaults to all" in fields["clear_raw"].description and "Set none" in fields["clear_raw"].description
    assert "Defaults to all" in fields["remove_after_processing"].description
    assert "directories" in fields["remove_after_processing"].description
    assert "directories" in fields["toggle_clear_scenario_timeseries_after_consolidation"].description


# ---------------------------------------------------------------------------
# D24 (b): the `timeseries` class is discharged by consolidation, or a logged no-op without it.
# ---------------------------------------------------------------------------
_SERIES_ATTRS = ("output_tritonswmm_node_time_series", "output_tritonswmm_link_time_series")


def _runner_probe(tmp_path: Path, *, consolidation_on: bool, reclaim_attrs=None, clearing=None):
    """Drive the real ``remove_after_processing`` with policy ``["timeseries"]`` over two real series files."""
    files = {}
    scen_paths = types.SimpleNamespace(sim_folder=tmp_path)
    for attr in _SERIES_ATTRS:
        files[attr] = _touch(tmp_path / "processed" / f"{attr}.nc")
        setattr(scen_paths, attr, files[attr])
    fake = types.SimpleNamespace(
        _analysis=types.SimpleNamespace(
            cfg_analysis=types.SimpleNamespace(
                remove_after_processing=["timeseries"],
                toggle_consolidate_timeseries=consolidation_on,
                toggle_clear_scenario_timeseries_after_consolidation=clearing,
            ),
        ),
        _scenario=types.SimpleNamespace(event_iloc=0),
        scen_paths=scen_paths,
        log=types.SimpleNamespace(
            full_TRITON_timeseries_cleared=None,
            full_SWMM_timeseries_cleared=None,
            deferred_writes=contextlib.nullcontext,
        ),
    )
    fake._reclaim_classes = _P._reclaim_classes
    fake._attr_is_swmm_side = _P._attr_is_swmm_side
    fake._reclaim_attrs = reclaim_attrs or _P._reclaim_attrs
    fake._reclaim_paths = lambda *a, **k: _P._reclaim_paths(fake, *a, **k)
    fake._remove_reclaimed = lambda p, sd, v: Path(p).unlink()
    _P.remove_after_processing(fake, model_type="tritonswmm", which="SWMM", verbose=False)
    return files


def test_with_consolidation_on_the_runner_deletes_no_series_and_records_the_discharge(tmp_path, capsys):
    """Property: with consolidation on and the ``timeseries`` class elected, the process runner deletes no
    per-scenario series, so the consolidation still has them to copy, and it records the discharge.

    Class: NEW CAPABILITY. Plausible wrong implementation: the landed order, in which the runner applies the class
    before the analysis consolidation runs and so leaves the consolidation nothing to copy. Must-fire control: the
    same probe with a ``_reclaim_attrs`` that returns the class's names (the landed behaviour) deletes both files,
    so the survival asserted first is a property of the code and not of the probe.
    Killed by: ``_reclaim_attrs`` returning names for the class. Passes under a second correct implementation that
    drops the class from the policy before calling it, if the discharge is still recorded.
    """
    files = _runner_probe(tmp_path / "on", consolidation_on=True)
    out = capsys.readouterr().out
    assert all(p.exists() for p in files.values()), "the runner deleted a per-scenario series the consolidation copies"
    assert "discharged by consolidation" in out

    def landed(model_type, *, policy, consolidation_on, which="both"):
        return _P._timeseries_drop_names(model_type, which=which)

    control = _runner_probe(tmp_path / "control", consolidation_on=True, reclaim_attrs=landed)
    assert not any(p.exists() for p in control.values()), "the control did not delete: the probe cannot see a deletion"


def test_with_consolidation_off_the_class_is_a_logged_no_op(tmp_path, capsys):
    """Property: with consolidation off and the class elected, the per-scenario series survive and a reason is
    logged, because they are then the processed output and exist nowhere else.

    Must-fire control: with consolidation on the runner records the discharge instead, so the reason asserted here
    is the off arm's and not a message printed on every path.
    """
    files = _runner_probe(tmp_path / "off", consolidation_on=False)
    out = capsys.readouterr().out
    assert all(p.exists() for p in files.values())
    assert "toggle_consolidate_timeseries is off" in out and "declining" in out
    assert "discharged by consolidation" not in out

    _runner_probe(tmp_path / "on", consolidation_on=True)
    on_out = capsys.readouterr().out
    assert "discharged by consolidation" in on_out and "toggle_consolidate_timeseries is off" not in on_out


def test_with_clearing_off_the_log_says_the_series_stay_because_clearing_is_off(tmp_path, capsys):
    """Property: with consolidation on, the clearing switch resolved False and the class elected, the runner deletes
    nothing and its log line says the series STAY in the per-scenario store because clearing is off, not that the
    consolidation will delete them.

    Class: NEW CAPABILITY (reviewer finding R2, ruling Form B). Plausible wrong implementation: the single
    consolidation-on message, which tells an operator the series will be gone when nothing deletes them. Must-fire
    control: the clearing-on arm (and the unset arm, which resolves to the consolidation toggle) prints the
    discharge message instead, so the branch is on the clearing switch and not on a constant.
    """
    files = _runner_probe(tmp_path / "off", consolidation_on=True, clearing=False)
    out = capsys.readouterr().out
    assert all(p.exists() for p in files.values())
    assert "because clearing is off" in out and "stay in the per-scenario store" in out
    assert "discharged by consolidation" not in out

    for sub, clearing in (("on", True), ("unset", None)):
        _runner_probe(tmp_path / sub, consolidation_on=True, clearing=clearing)
        on_out = capsys.readouterr().out
        assert "discharged by consolidation" in on_out and "because clearing is off" not in on_out


@pytest.mark.parametrize("consolidation_on", [True, False])
def test_reclaim_attrs_takes_the_toggle_as_an_argument_and_never_reads_it_from_a_scenario(consolidation_on):
    """``_reclaim_attrs`` is static and fixture-free: it runs on a bare class with the toggle passed in.

    Must-fire control: omitting the argument is a TypeError, so the toggle really is part of the signature.
    """
    assert _P._reclaim_attrs("tritonswmm", policy="all", consolidation_on=consolidation_on) == ()
    with pytest.raises(TypeError):
        _P._reclaim_attrs("tritonswmm", policy="all")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# D24 (c): `hydro_out` and `hydrographs` both reclaimed, then a re-prep.
# ---------------------------------------------------------------------------
def _hydrology_scenario(tmp_path: Path, *, with_capture: bool):
    """A fixture-free stand-in with hydro.out, the hydrograph files and (optionally) a real openable capture."""
    strm = tmp_path / "strmflow"
    strm.mkdir(parents=True)
    swmm = tmp_path / "swmm"
    swmm.mkdir()
    (swmm / "hydro.inp").write_text("inp")
    (swmm / "hydro.out").write_text("out")
    (strm / "tseries.hyg").write_text("hyg")
    (strm / "loc.txt").write_text("loc")
    if with_capture:
        cap = tmp_path / "processed" / "hydrology_inflow_summary.zarr"
        cap.parent.mkdir()
        xr.Dataset({"inflow": (("node_id", "timestep"), [[1.0, 2.0]])}).to_zarr(cap)
    flag = types.SimpleNamespace(get=lambda: True)
    return types.SimpleNamespace(
        event_iloc=0,
        log=types.SimpleNamespace(
            hyg_timeseries_created=flag, hyg_locs_created=flag, consolidated_hydrology_cleared=None
        ),
        scen_paths=types.SimpleNamespace(
            sim_folder=tmp_path,
            hyg_timeseries=strm / "tseries.hyg",
            hyg_locs=strm / "loc.txt",
            swmm_hydro_inp=swmm / "hydro.inp",
        ),
    )


def test_a_scenario_with_both_classes_reclaimed_re_prepares_from_the_capture(tmp_path):
    """Property: with ``hydro_out`` and ``hydrographs`` both elected, both go, and the re-prep gate then falls
    through to a rebuild from the per-node capture instead of raising.

    Class: NEW CAPABILITY (the default now elects both). Plausible wrong implementation: the landed gate, which
    raises when ``hydro.out`` is absent and the hydrographs are absent. Must-fire control: the same state with the
    capture removed raises, naming the output and the capture, so the rebuild source is what separates the two.
    Killed by: the third rebuild source being dropped from the gate. Passes under a second correct implementation
    that rebuilds from the consolidated group before the capture, if the gate still returns ``fell_through``.
    """
    scen = _hydrology_scenario(tmp_path / "with", with_capture=True)
    outcome = reclaim_scenario_scoped_classes(scen, ("hydro_out", "hydrographs"), verbose=False)
    assert outcome["hydro_out"] is True and outcome["hydrographs"] is True
    assert not scen.scen_paths.hyg_timeseries.exists() and not scen.scen_paths.hyg_locs.exists()
    assert not scen.scen_paths.swmm_hydro_inp.with_suffix(".out").exists()
    assert hydrograph_outputs_gate(scen) == "fell_through"

    bare = _hydrology_scenario(tmp_path / "bare", with_capture=False)
    for p in (
        bare.scen_paths.hyg_timeseries,
        bare.scen_paths.hyg_locs,
        bare.scen_paths.swmm_hydro_inp.with_suffix(".out"),
    ):
        p.unlink()
    with pytest.raises(ProcessingError) as err:
        hydrograph_outputs_gate(bare)
    assert "hydrology_inflow_summary.zarr" in str(err.value)


def test_the_hydrographs_class_declines_without_a_capture_so_the_pair_cannot_strand_a_scenario(tmp_path):
    """Control for the reclaim side: with no capture and no recorded consolidation, ``hydrographs`` is declined.

    Varies the property under suspicion (the capture's presence), so a reclaim that ignored its gate would delete the
    hydrograph files here and leave a scenario with no rebuild source.
    """
    scen = _hydrology_scenario(tmp_path, with_capture=False)
    outcome = reclaim_scenario_scoped_classes(scen, ("hydrographs",), verbose=False)
    assert outcome["hydrographs"] is False
    assert scen.scen_paths.hyg_timeseries.exists() and scen.scen_paths.hyg_locs.exists()


# ---------------------------------------------------------------------------
# A kept directory whose summaries were cleared: df_status reads through the clearing record.
# ---------------------------------------------------------------------------
@pytest.fixture
def _pin_clean_build(monkeypatch):
    """Force the build-stamp term of the consolidation reuse conjunction TRUE (a dirty checkout would rebuild)."""
    import hhemt.provenance as prov

    monkeypatch.setattr(prov, "store_build_mismatch", lambda store, stamp: None)


def test_df_status_reads_a_cleared_performance_summary_of_a_kept_directory(golden_extractor, _pin_clean_build):
    """Property: with consolidation on and `clear_raw` none, the consolidation clears the per-scenario summaries and
    keeps the scenario directories, and `df_status` still returns a row per event with its performance cells read
    from the consolidated tree.

    Class: REGRESSION that the new defaults expose (the state is every user who sets `clear_raw: none` and takes the
    defaults). Plausible wrong implementation: `_get_performance_summary_row` opening the per-scenario store the
    clearing deleted, which raises `ValueError: could not open file because it does not exist`. Reachability: the real
    consolidation over the committed multi-sim golden, then the real `df_status`. Must-fire control: the the same
    performance cells are read BEFORE the clearing, from the store, and must be non-null and equal afterwards.
    Killed by: the fix being reverted. Passes under a second correct implementation that reads the tree for every
    event, if the cells equal the per-scenario store's.
    """
    from tests.fixtures.golden_trees import dry_run_plan
    from tests.test_summary_store_clearing import _own, _ready, _record

    on = _ready(golden_extractor, clear=True)
    perf_stem = "TRITONSWMM_perf_summary"
    dry_run_plan(on)
    assert _own(on, 0, perf_stem).exists(), "control: before the consolidation the per-scenario store is present"
    rows_before = on.df_status
    perf = [c for c in rows_before.columns if c.startswith("perf_") and c != "perf_column_set_across_allocations"]
    assert perf and rows_before[perf].notna().any().any(), "control: the cells compared below are not all None"
    on.process.consolidate_to_datatree()
    assert _record(on, 0) is not None, "the clearing record must exist, or this test did not clear anything"
    assert not _own(on, 0, perf_stem).exists(), "the performance summary store must be cleared"
    assert (on.analysis_paths.simulation_directory / on.process._event_slug(0)).exists(), "the directory is kept"
    rows_after = on.df_status
    assert len(rows_after) == len(rows_before)
    for col in perf:
        assert rows_after[col].tolist() == rows_before[col].tolist(), col


# ---------------------------------------------------------------------------
# The golden extractor writes the per-node inflow capture the default policy leaves the hydrology consolidation.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["multisim", "sensitivity"])
def test_an_extracted_golden_carries_a_complete_inflow_capture_for_every_scenario(golden_extractor, kind):
    """Property: every scenario of an extracted golden holds `processed/hydrology_inflow_summary.zarr` with one
    `inflow` series per hydrology node, on the report steps of the golden's hydrology window, differing by node and
    by event.

    Class: fixture contract (the default policy removes `hydro.out`, so the capture is the only hydrology source left
    to the consolidation). Plausible wrong implementation: a capture for some scenarios only, or identical for every
    event (an off-by-one in the event position would then pass). Must-fire control: the node set equals the
    `[JUNCTIONS]` of the scenario's own `hydro.inp`, and two events differ.
    """
    import xarray as xr

    from tests.fixtures.golden_trees import HYDROLOGY_CAPTURE_STEPS, _hydro_inp_nodes

    root = golden_extractor(kind)
    scenario_dirs = [d for sims in sorted(root.rglob("sims")) for d in sorted(sims.iterdir()) if d.is_dir()]
    assert scenario_dirs
    by_event: dict[int, list[float]] = {}
    for scenario in scenario_dirs:
        ds = xr.open_zarr(scenario / "processed" / "hydrology_inflow_summary.zarr", consolidated=False).load()
        nodes = _hydro_inp_nodes(scenario / "swmm" / "hydro.inp")
        assert set(ds["node_id"].values.tolist()) == set(nodes)
        assert ds.sizes["timestep"] == HYDROLOGY_CAPTURE_STEPS
        assert bool(ds["inflow"].notnull().all()) and float(ds["inflow"].max()) > 0.0
        by_event[int(ds["event_iloc"].values[0])] = ds["inflow"].values.ravel().tolist()
    if len(by_event) > 1:
        assert len({tuple(v) for v in by_event.values()}) == len(by_event), "events must differ"


def test_the_hydrology_consolidation_appends_from_the_golden_capture_and_refuses_without_any_source(golden_extractor):
    """Property: on the multi-sim golden with every `hydro.out` removed (what the default policy leaves), the real
    `consolidate_hydrology` appends every event to `swmm_node_hydro`; with the capture removed as well it refuses,
    naming the scenario.

    Class: regression the new defaults expose. Plausible wrong implementation: a refusal that is also raised when the
    capture is present, or a consolidation that silently skips a source-less scenario. Must-fire control: the second
    arm, same tree, one more deletion.
    """
    import shutil

    from hhemt.exceptions import ProcessingError
    from tests.fixtures.golden_trees import construct_golden_analysis

    system_dir = golden_extractor("multisim")
    cfg = system_dir / "analysis_config.yaml"
    cfg.write_text(
        cfg.read_text()
        + "toggle_consolidate_timeseries: true\ntoggle_clear_scenario_timeseries_after_consolidation: false\n"
    )
    sims = next(system_dir.rglob("sims"))
    for scenario in sorted(p for p in sims.iterdir() if p.is_dir()):
        (scenario / "swmm" / "hydro.out").unlink()
    analysis = construct_golden_analysis(system_dir)
    store = analysis.process.consolidate_hydrology()
    events = analysis.df_sims.index
    import xarray as xr

    ts = xr.open_zarr(store, group="tritonswmm/swmm_node_hydro/timeseries", consolidated=False)
    assert [int(e) for e in ts["event_iloc"].values] == [int(e) for e in events]

    shutil.rmtree(store / "tritonswmm" / "swmm_node_hydro")
    first = sorted(p for p in sims.iterdir() if p.is_dir())[0]
    shutil.rmtree(first / "processed" / "hydrology_inflow_summary.zarr")
    with pytest.raises(ProcessingError) as err:
        analysis.process.consolidate_hydrology()
    assert "no per-node inflow capture" in str(err.value)


# ---------------------------------------------------------------------------
# A process-stage rebuild keeps the per-node inflow capture.
# ---------------------------------------------------------------------------
def test_a_regenerating_process_reprocess_keeps_the_inflow_capture_and_the_hydrology_consolidation_succeeds(
    golden_extractor,
):
    """Property: `reprocess(start_with="process", regenerate_existing=True)` deletes every output under a scenario's
    `processed/` except `hydrology_inflow_summary.zarr`, and the hydrology consolidation that follows (with
    `hydro.out` already removed by the default reclaim) appends every event.

    Class: regression the default policy exposes. Plausible wrong implementation: deleting the whole `processed/`
    directory, which removes the capture, the only hydrology source once `hydro.out` is gone. Reachability: the real
    `_delete_processed_outputs_for_reprocess` and `consolidate_hydrology` on the multi-sim golden. Must-fire control:
    with the capture removed as well, the same consolidation refuses, naming the capture and a reprocess as a cause.
    """
    import shutil

    import xarray as xr

    from hhemt.exceptions import ProcessingError
    from tests.fixtures.golden_trees import construct_golden_analysis

    system_dir = golden_extractor("multisim")
    cfg = system_dir / "analysis_config.yaml"
    cfg.write_text(
        cfg.read_text()
        + "toggle_consolidate_timeseries: true\ntoggle_clear_scenario_timeseries_after_consolidation: false\n"
    )
    sims = next(system_dir.rglob("sims"))
    scenarios = sorted(p for p in sims.iterdir() if p.is_dir())
    for scenario in scenarios:
        (scenario / "swmm" / "hydro.out").unlink()
    analysis = construct_golden_analysis(system_dir)
    analysis._delete_processed_outputs_for_reprocess("process", regenerate_existing=True)
    for scenario in scenarios:
        processed = scenario / "processed"
        assert [p.name for p in processed.iterdir()] == ["hydrology_inflow_summary.zarr"], processed
    store = analysis.process.consolidate_hydrology()
    ts = xr.open_zarr(store, group="tritonswmm/swmm_node_hydro/timeseries", consolidated=False)
    assert [int(e) for e in ts["event_iloc"].values] == [int(e) for e in analysis.df_sims.index]

    shutil.rmtree(store / "tritonswmm" / "swmm_node_hydro")
    shutil.rmtree(scenarios[0] / "processed" / "hydrology_inflow_summary.zarr")
    with pytest.raises(ProcessingError) as err:
        analysis.process.consolidate_hydrology()
    assert "no per-node inflow capture" in str(err.value) and "regenerate_existing" in str(err.value)


def test_the_processed_deletion_list_keeps_only_the_capture_and_takes_the_whole_directory_without_one(tmp_path):
    """Property: `processed_outputs_to_delete` returns every child but the capture when the capture exists, and the
    whole `processed/` directory when it does not (so the delete runner's old outcome is unchanged there).

    Must-fire control: the same directory flips between the two answers when only the capture is added.
    """
    from hhemt.swmm_runoff_modeling import processed_outputs_to_delete

    sim = tmp_path / "evt"
    processed = sim / "processed"
    (processed / "a_summary.zarr").mkdir(parents=True)
    (processed / "a_tseries.zarr").mkdir()
    assert processed_outputs_to_delete(sim) == [processed]
    (processed / "hydrology_inflow_summary.zarr").mkdir()
    assert [p.name for p in processed_outputs_to_delete(sim)] == ["a_summary.zarr", "a_tseries.zarr"]
    assert processed_outputs_to_delete(tmp_path / "absent") == []


def test_the_inflow_capture_stores_its_timestep_coordinate_as_float64_and_reads_back_float64(tmp_path):
    """Property: `node_inflow_capture_dataset` builds `timestep` as float64 hours, and a store written from it reads
    back float64, so a reader that converts hours to seconds loses no precision.

    Class: dtype contract of the capture coordinate. Plausible wrong implementation: casting the coordinate to
    float32, which a stored-variable dtype test rejects. No golden is used. Must-fire control: the inflow values
    stay float32 (their storage dtype), so the assertion is about the coordinate and not every variable.
    """
    import numpy as np
    import xarray as xr

    from hhemt.swmm_runoff_modeling import HydrologyNodeInflow, node_inflow_capture_dataset
    from hhemt.utils import write_zarr

    flows = {"J1": np.linspace(0.1, 1.0, 12), "J2": np.linspace(0.2, 2.0, 12)}
    node_inflow = HydrologyNodeInflow(flows, {"J1": (5.0, 5.0), "J2": (15.0, 5.0)}, np.arange(12) * 10 / 3600.0, "CMS")
    ds = node_inflow_capture_dataset(node_inflow, 0)
    assert ds["timestep"].dtype == np.dtype("float64")
    assert ds["inflow"].dtype == np.dtype("float32")
    store = tmp_path / "capture.zarr"
    write_zarr(ds, store, chunks=None, encoding={})
    back = xr.open_zarr(store, consolidated=False)
    assert back["timestep"].dtype == np.dtype("float64")
    np.testing.assert_array_equal(back["timestep"].values, np.arange(12) * 10 / 3600.0)
