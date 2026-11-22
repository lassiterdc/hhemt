"""With consolidation off and ``timeseries`` named in a ``remove_after_processing`` list, no series remains.

The developer's sentence (gate G3, 2026-11-21): "As long as the consolidated time series is toggled off and we are
clearing all raw and intermediate outputs, there should be no time series artifact remaining by the end of the
workflow". The coordinator's rule for it: ``toggle_consolidate_timeseries`` False, ``clear_raw`` covering every enabled
model, and ``remove_after_processing`` GIVEN AS A LIST that names ``timeseries`` (an explicit election, read from the
value because a persisted config cannot tell an omitted field from a stated one). The blanket ``all`` keeps the series
with consolidation off, as before (D24 (b)), and consolidation on is unchanged.

POPULATION: every scenario of the analysis (the three events of the multi-sim golden), never a partition of it.
DEFECT PREDICATE, applied to each scenario by ``_leftovers``: a ``*_tseries`` store (zarr or nc) exists anywhere under
its directory. The test that applies it is ``test_no_time_series_store_remains_for_any_scenario``; it asserts the
predicate is False for every scenario, after the real process runner and the real consolidation.

No solver, no compile, no Snakemake: the committed golden is a finished analysis tree. The processing step is the
REAL ``process_timeseries_runner.main`` (the subprocess entry the workflow runs, called in process with the persisted
config), followed by the real ``consolidate_to_datatree``.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import xarray as xr
import yaml

from tests.fixtures.golden_trees import construct_golden_analysis

_ALL_CLASSES = [
    "timeseries",
    "raw_swmm_binaries",
    "coupled_rpt",
    "hydro_out",
    "prep_inputs",
    "hydrographs",
    "standalone_rpt",
]
_OVERRIDDEN_KEYS = (
    "toggle_consolidate_timeseries",
    "toggle_clear_scenario_timeseries_after_consolidation",
    "clear_raw",
    "remove_after_processing",
)
# The stochastic ensemble's election: five of the seven classes, so no directory removal is involved.
_FIVE_CLASSES = ["timeseries", "coupled_rpt", "hydro_out", "prep_inputs", "standalone_rpt"]
_SUMMARY_GROUPS = ("tritonswmm/triton/summary", "tritonswmm/swmm_node/summary", "tritonswmm/swmm_link/summary")


@pytest.fixture(autouse=True)
def _pin_clean_build(monkeypatch):
    """Force the build-stamp term of the consolidation reuse conjunction TRUE (a dirty checkout would rebuild)."""
    import hhemt.provenance as prov

    monkeypatch.setattr(prov, "store_build_mismatch", lambda store, stamp: None)


def _extract(golden_extractor, *, consolidate: bool, clear_raw="all", remove=None) -> Path:
    """The multi-sim golden with the four keys written into its PERSISTED analysis config (the runner reads that)."""
    system_dir = golden_extractor("multisim")
    remove = list(_FIVE_CLASSES) if remove is None else remove
    cfg = system_dir / "analysis_config.yaml"
    text = re.sub(rf"^({'|'.join(_OVERRIDDEN_KEYS)}):.*\n", "", cfg.read_text(), flags=re.M)
    text += yaml.safe_dump(
        {"toggle_consolidate_timeseries": consolidate, "clear_raw": clear_raw, "remove_after_processing": remove}
    )
    cfg.write_text(text)
    return system_dir


def _run_runner(system_dir: Path, event_iloc: int, model: str, monkeypatch) -> int:
    """The process runner's real ``main`` for one scenario and model, with the persisted configs."""
    from hhemt import process_timeseries_runner as runner

    argv = [
        "process_timeseries_runner",
        "--event-iloc",
        str(event_iloc),
        "--analysis-config",
        str(system_dir / "analysis_config.yaml"),
        "--system-config",
        str(system_dir / "system_config.yaml"),
        "--model-type",
        model,
        "--which",
        "both",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    return runner.main()


def _population(analysis) -> list[tuple[int, str]]:
    """Every scenario of the analysis as ``(event_iloc, event_id)``."""
    return [(int(i), analysis.process._event_slug(int(i))) for i in analysis.df_sims.index]


def _tseries_stores(root: Path) -> list[Path]:
    """Every artifact that carries ``_tseries`` in its name: stores, ``.done`` flags, chapter dirs, intermediates."""
    return sorted(p for p in root.rglob("*") if "_tseries" in p.name)


def _leftovers(analysis, event_id: str) -> list[str]:
    """The defect predicate for one scenario: an artifact named ``*_tseries*`` exists anywhere under its directory."""
    sim = Path(analysis.analysis_paths.simulation_directory) / event_id
    return [str(p) for p in _tseries_stores(sim)] if sim.exists() else []


def _processed_names(analysis, event_id: str) -> list[str]:
    """What the scenario's ``processed`` directory still holds, for the failure message only."""
    processed = Path(analysis.analysis_paths.simulation_directory) / event_id / "processed"
    return sorted(p.name for p in processed.iterdir()) if processed.is_dir() else []


def _process_all(system_dir: Path, monkeypatch) -> object:
    """Run the runner for every scenario and enabled model, then the consolidation; return a fresh analysis."""
    analysis = construct_golden_analysis(system_dir)
    models = list(analysis._get_enabled_model_types())
    assert models, "control: the golden enables at least one model"
    for event_iloc, _event_id in _population(analysis):
        for model in models:
            assert _run_runner(system_dir, event_iloc, model, monkeypatch) == 0, (event_iloc, model)
    after = construct_golden_analysis(system_dir)
    after.process.consolidate_to_datatree()
    return after


def _summary_stores_open(analysis, event_iloc: int) -> bool:
    """True iff every enabled model's per-scenario summary store of the scenario exists and opens."""
    from hhemt.summary_paths import _SUMMARY_ATTRS_BY_MODEL

    scen = analysis._retrieve_sim_run_processing_object(event_iloc)._scenario
    checked = 0
    for model in analysis._get_enabled_model_types():
        for attr in _SUMMARY_ATTRS_BY_MODEL.get(model, ()):
            path = getattr(scen.scen_paths, attr, None)
            if path is None:
                continue
            if not Path(path).exists():
                return False
            xr.open_zarr(path, consolidated=False, chunks=None).load()
            checked += 1
    return checked > 0


def test_no_time_series_store_remains_for_any_scenario(golden_extractor, monkeypatch):
    """Property: consolidation off, ``clear_raw`` all and a ``remove_after_processing`` list naming ``timeseries``
    (the stochastic ensemble's five classes) leave, for EVERY scenario of the analysis, no ``*_tseries`` store, with
    every per-scenario summary store intact and openable and the consolidated tree holding every summary.

    Input class: the whole analysis (three scenarios), through the real process runner and the real consolidation.
    Must-fire controls: before the run every scenario holds a ``*_tseries`` store (the predicate is True on the
    input, so a False after is a result of the run), and the summary groups read back with every event.
    Killed by: restoring the D24 (b) decline for the explicit-list arm, or a drop table that omits a series store
    (the SWMM-only system series was one). Passes under a second correct implementation that deletes the series
    inside the consolidation instead of the runner.
    """
    system_dir = _extract(golden_extractor, consolidate=False)
    before = construct_golden_analysis(system_dir)
    population = _population(before)
    assert len(population) == 3
    for _event_iloc, event_id in population:
        assert _leftovers(before, event_id), f"control: {event_id} holds a *_tseries store before the run"

    after = _process_all(system_dir, monkeypatch)

    remaining = {event_id: _leftovers(after, event_id) for _i, event_id in population}
    assert all(not left for left in remaining.values()), "\n".join(
        f"{k}: {[p.rsplit('/sims/', 1)[-1] for p in v]} processed={_processed_names(after, k)}"
        for k, v in remaining.items()
    )
    for event_iloc, event_id in population:
        assert _summary_stores_open(after, event_iloc), f"{event_id}: a summary store is missing or unopenable"

    store = Path(after.analysis_paths.analysis_datatree_zarr)
    assert (store / "tritonswmm" / "triton" / "summary").is_dir(), "control: the summary tier is in the tree"
    for group in _SUMMARY_GROUPS:
        if (store / group).is_dir():
            ds = xr.open_zarr(store, group=group, consolidated=False, chunks=None).load()
            assert ds.sizes["event_iloc"] == len(population), (group, dict(ds.sizes))
    assert not any(store.glob("**/timeseries")), "consolidation is off, so no timeseries branch exists in the tree"


@pytest.mark.parametrize(
    ("clear_raw", "remove"),
    [
        ("all", "all"),
        ("none", list(_FIVE_CLASSES)),
        ("all", [c for c in _FIVE_CLASSES if c != "timeseries"]),
    ],
    ids=["blanket-all-keeps", "clear_raw-none-keeps", "list-without-timeseries-keeps"],
)
def test_the_series_stay_unless_all_three_keys_elect_their_removal(golden_extractor, monkeypatch, clear_raw, remove):
    """Property: with consolidation off, the blanket ``all`` (the default policy), a keeping ``clear_raw``, or a list
    that omits ``timeseries`` keeps every scenario's series stores, which is today's behaviour (D24 (b)).

    Must-fire control: the explicit-list arm above removes them from the same golden with only the keys changed, so
    a stay here is the setting's effect and not a runner that never deletes. This is the control that ``all`` with
    consolidation off still keeps the series, and it is also the check that no config outside the explicit election
    (the synthetic builder, the goldens and the estate arms all keep ``clear_raw: none`` or ``remove: none``/``all``)
    changes outcome.
    """
    system_dir = _extract(golden_extractor, consolidate=False, clear_raw=clear_raw, remove=remove)
    before = construct_golden_analysis(system_dir)
    expected = {event_id: _leftovers(before, event_id) for _i, event_id in _population(before)}
    assert all(expected.values()), "control: every scenario holds series stores before the run"
    after = _process_all(system_dir, monkeypatch)
    for _event_iloc, event_id in _population(after):
        assert _leftovers(after, event_id) == expected[event_id], f"{event_id}: the keeping setting lost a series"


@pytest.mark.parametrize(
    ("consolidate", "clear_raw", "remove", "expected"),
    [
        (False, "all", ["timeseries"], True),
        (False, ["tritonswmm", "triton", "swmm"], ["coupled_rpt", "timeseries"], True),
        (False, ["tritonswmm"], ["timeseries"], False),
        (False, "none", ["timeseries"], False),
        (False, "all", "all", False),
        (False, "all", "none", False),
        (False, "all", ["coupled_rpt"], False),
        (True, "all", ["timeseries"], False),
        (True, "all", "all", False),
    ],
)
def test_the_discard_arm_is_the_conjunction_of_the_three_keys(consolidate, clear_raw, remove, expected):
    """Name-level and fixture-free: the arm that deletes the series is exactly consolidation off, ``clear_raw`` covering
    every enabled model, and a LIST naming ``timeseries``. Each False row differs from a True row in one key."""
    from types import SimpleNamespace

    from hhemt.process_simulation import timeseries_discard_arm

    cfg = SimpleNamespace(
        toggle_consolidate_timeseries=consolidate, clear_raw=clear_raw, remove_after_processing=remove
    )
    assert timeseries_discard_arm(cfg, ["tritonswmm", "triton", "swmm"]) is expected


def test_nothing_is_deleted_when_the_summary_check_fails(golden_extractor, monkeypatch):
    """Property: with the explicit-list arm on and one scenario's summary store present but not openable, the runner
    returns non-zero for that scenario and deletes none of its series, and the scenarios whose summaries open are
    discarded as before.

    Must-fire control: the same runner call on an untouched scenario returns 0 and removes the series, so the stay
    is caused by the failed check and not by an idle runner.
    """
    system_dir = _extract(golden_extractor, consolidate=False)
    analysis = construct_golden_analysis(system_dir)
    population = _population(analysis)
    models = list(analysis._get_enabled_model_types())
    from hhemt.process_simulation import TRITONSWMM_sim_post_processing as proc_cls

    broken_iloc = population[0][0]
    broken_sim = Path(analysis._retrieve_sim_run_processing_object(broken_iloc)._scenario.scen_paths.sim_folder)
    real_open = proc_cls._open

    def open_with_one_unopenable_summary(self, path, *args, **kwargs):
        """The scenario's summary stores are present but cannot be opened: what the runner's third gate catches."""
        p = Path(path)
        if p.stem.endswith("_summary") and broken_sim in p.parents:
            raise OSError(f"unopenable summary store {p.name}")
        return real_open(self, path, *args, **kwargs)

    monkeypatch.setattr(proc_cls, "_open", open_with_one_unopenable_summary)
    series_before = _tseries_stores(broken_sim)
    assert series_before, "control: the scenario holds series stores"

    for model in models:
        assert _run_runner(system_dir, broken_iloc, model, monkeypatch) != 0
    assert _tseries_stores(broken_sim) == series_before, "series deleted after a failed check"

    ok_iloc = population[1][0]
    ok_sim = Path(analysis.analysis_paths.simulation_directory) / population[1][1]
    assert _tseries_stores(ok_sim), "control: the intact scenario holds series stores"
    for model in models:
        assert _run_runner(system_dir, ok_iloc, model, monkeypatch) == 0
    assert not _tseries_stores(ok_sim), "control: the intact scenario's series are discarded"


def test_with_consolidation_on_the_series_are_consolidated_then_cleared_as_before(golden_extractor, monkeypatch):
    """Property (unchanged behaviour): with consolidation on and the default policy, the series are copied to the
    ``timeseries`` branches first and cleared after the read-back, so every scenario ends with no series store, and
    the tree holds every event's series.

    Must-fire control: the ``timeseries`` branch exists in the tree, so the series were consolidated and not
    discarded.
    """
    system_dir = _extract(golden_extractor, consolidate=True, remove="all")
    after = _process_all(system_dir, monkeypatch)
    for _event_iloc, event_id in _population(after):
        assert _leftovers(after, event_id) == [], event_id
    store = Path(after.analysis_paths.analysis_datatree_zarr)
    branches = [p for p in store.glob("**/timeseries") if p.is_dir()]
    assert branches, "control: consolidation on writes timeseries branches"
    for branch in branches:
        ds = xr.open_zarr(store, group=str(branch.relative_to(store)), consolidated=False, chunks=None)
        assert ds.sizes["event_iloc"] == 3, (str(branch), dict(ds.sizes))
