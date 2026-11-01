"""Battery over ``analysis.test()``'s configuration logic, with no solver compile or run and no workflow run. The
first and third parts read stand-in objects; the second reads a tree extracted from the committed sensitivity golden.

``analysis.test()`` has three parts that need no solver compile or run and no workflow run to be checked. The first
two are decided by configuration; the second also reads the tree and the weather file that the extracted golden
supplies. The third is decided by what is on disk and by which of ``ldd`` and ``readelf`` are on the path, so its
checks build a temporary directory and replace ``shutil.which`` and ``subprocess.run``:

* which representatives are selected: ``TRITONSWMM_analysis._select_test_representatives`` groups candidates by
  (model toggles, compilation backend, partition, compute config) and keeps the least-demanding member of each;
* how the ``_test/`` subset is built: ``TRITONSWMM_analysis._build_test_analyses`` slices the weather, overlays the
  configuration and constructs one sub-analysis per representative;
* what ``_capture_binary_provenance`` returns for an analysis directory with and without the solver executable, which
  executable it takes first in sorted order, and what it records when a capture fails or a tool is missing.

The five model-toggle combinations below are the ones for which the synthetic test-case catalog
(``tests/fixtures/test_case_catalog.py``) builds a case (all three models; coupled with TRITON-only; coupled only, in
the sensitivity case; TRITON-only; SWMM-only). Each check calls the production function unbound on a stand-in object,
so only the attributes that function reads are supplied.

Each assertion helper has a must-fail control that feeds it the defect it exists to catch: a representative list
with a group dropped, and a provenance pair with the two arms exchanged.
"""

from __future__ import annotations

import inspect
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import xarray as xr
import yaml

from hhemt import analysis as analysis_module
from hhemt.analysis import (  # noqa: F401  (TestRepresentative: documents the type)
    TestRepresentative,
    TRITONSWMM_analysis,
)

# (toggle_tritonswmm_model, toggle_triton_model, toggle_swmm_model), the order of the selector's group key.
TOGGLE_COMBINATIONS = {
    "all_models": (True, True, True),
    "coupled_and_triton": (True, True, False),
    "coupled_only": (True, False, False),
    "triton_only": (False, True, False),
    "swmm_only": (False, False, True),
}


def _compute(run_mode, n_mpi, n_omp, n_nodes=1, n_gpus=0):
    return {"run_mode": run_mode, "n_mpi_procs": n_mpi, "n_omp_threads": n_omp, "n_gpus": n_gpus, "n_nodes": n_nodes}


# Three candidates, two distinct group keys. The two serial candidates share a key (every keyed axis is equal) and
# differ only in the non-keyed DEM resolution, which breaks the tie: the lower value sorts first and is chosen.
# The lowest-DEM serial candidate is deliberately NOT first, so merit and position differ.
CANDIDATE_ROWS = [
    {**_compute("serial", 1, 1), "dem": 20.0},
    {**_compute("serial", 1, 1), "dem": 10.0},
    {**_compute("mpi", 2, 1), "dem": 10.0},
]


def _member(toggles, row, tag, partition=None):
    row = dict(row)
    dem = row.pop("dem", 10.0)
    tritonswmm, triton, swmm = toggles
    cfg_system = SimpleNamespace(
        toggle_tritonswmm_model=tritonswmm,
        toggle_triton_model=triton,
        toggle_swmm_model=swmm,
        target_dem_resolution=dem,
    )
    cfg = SimpleNamespace(hpc_ensemble_partition=partition, toggle_sensitivity_analysis=False, **row)
    return SimpleNamespace(
        cfg_analysis=cfg, cfg_hpc_system=None, _system=SimpleNamespace(cfg_system=cfg_system), tag=tag
    )


def _sensitivity_master(toggles):
    members = {i: _member(toggles, row, f"member_{i}") for i, row in enumerate(CANDIDATE_ROWS)}
    master = _member(toggles, CANDIDATE_ROWS[0], "master")
    master.cfg_analysis.toggle_sensitivity_analysis = True
    master.sensitivity = SimpleNamespace(members=members)
    return master, list(members.values())


def _select(master):
    return TRITONSWMM_analysis._select_test_representatives(master)


def _config_of(analysis):
    cfg = analysis.cfg_analysis
    return (cfg.run_mode, cfg.n_mpi_procs, cfg.n_omp_threads, cfg.n_gpus, cfg.n_nodes)


def _assert_selection_is_total(reps, candidates) -> None:
    """The representatives cover exactly the compute configs among the candidates, no full key repeats, and each
    representative is drawn from the candidates. The toggles axis is not checked here; that is what
    ``test_the_toggle_combinations_make_distinct_groups_when_mixed`` checks."""
    keys = [rep.key for rep in reps]
    assert len(set(keys)) == len(keys), f"duplicate representative keys: {keys}"
    expected = {_config_of(c) for c in candidates}
    selected = {rep.key[3] for rep in reps}
    assert selected == expected, f"missing={expected - selected}, extra={selected - expected}"
    assert {id(rep.source_analysis) for rep in reps} <= {id(c) for c in candidates}


# ---- representative selection ---------------------------------------------------------------------------------


@pytest.mark.parametrize("combination", sorted(TOGGLE_COMBINATIONS))
def test_a_sensitivity_master_selects_one_least_demanding_representative_per_group(combination) -> None:
    toggles = TOGGLE_COMBINATIONS[combination]
    master, candidates = _sensitivity_master(toggles)
    reps = _select(master)
    _assert_selection_is_total(reps, candidates)
    assert len(reps) == 2  # three candidates, two distinct compute configs
    assert {rep.key[0] for rep in reps} == {toggles}
    by_config = {rep.key[3]: rep for rep in reps}
    # The two serial candidates share a key; the one with the lower DEM-resolution value wins the tie.
    assert by_config[("serial", 1, 1, 0, 1)].source_analysis.tag == "member_1"
    assert by_config[("mpi", 2, 1, 0, 1)].source_analysis.tag == "member_2"
    assert all(rep.partition is None for rep in reps)


@pytest.mark.parametrize("combination", sorted(TOGGLE_COMBINATIONS))
def test_a_plain_analysis_is_its_own_single_representative(combination) -> None:
    toggles = TOGGLE_COMBINATIONS[combination]
    plain = _member(toggles, _compute("serial", 1, 1), "plain")
    reps = _select(plain)
    assert len(reps) == 1
    assert reps[0].source_analysis is plain
    assert reps[0].key[0] == toggles


def test_the_toggle_combinations_make_distinct_groups_when_mixed() -> None:
    """Candidates that differ only in model toggles are separate groups (the toggles are part of the key)."""
    members = {
        i: _member(toggles, _compute("serial", 1, 1), name)
        for i, (name, toggles) in enumerate(TOGGLE_COMBINATIONS.items())
    }
    master = _member((True, True, True), _compute("serial", 1, 1), "master")
    master.cfg_analysis.toggle_sensitivity_analysis = True
    master.sensitivity = SimpleNamespace(members=members)
    reps = _select(master)
    assert {rep.key[0] for rep in reps} == set(TOGGLE_COMBINATIONS.values())
    assert len(reps) == len(TOGGLE_COMBINATIONS)


def _master_of(members):
    master = _member((True, True, True), _compute("serial", 1, 1), "master")
    master.cfg_analysis.toggle_sensitivity_analysis = True
    master.sensitivity = SimpleNamespace(members=dict(enumerate(members)))
    return master


def test_candidates_differing_only_in_partition_are_separate_groups() -> None:
    row = _compute("serial", 1, 1)
    members = [_member((True, True, True), row, "on_a", "part_a"), _member((True, True, True), row, "on_b", "part_b")]
    reps = _select(_master_of(members))
    assert sorted(rep.key[2] for rep in reps) == ["part_a", "part_b"]
    assert {rep.partition for rep in reps} == {"part_a", "part_b"}
    assert {rep.source_analysis.tag for rep in reps} == {"on_a", "on_b"}


def test_candidates_differing_only_in_gpu_backend_are_separate_groups(monkeypatch) -> None:
    """The backend key axis: resolved per partition by ``resolve_gpu_target`` (replaced here by a lookup)."""
    backends = {"nvidia_part": "CUDA", "amd_part": "HIP"}
    monkeypatch.setattr(analysis_module, "resolve_gpu_target", lambda hpc, partition: ("hardware", backends[partition]))
    # The selector resolves a backend only for a member with a non-zero n_gpus and a non-None cfg_hpc_system (set
    # below), and resolves it per partition. The members must therefore differ in partition as well: the backend is a
    # function of the partition (for one hpc config), so two members cannot differ in backend while sharing one.
    row = _compute("gpu", 1, 1, n_gpus=1)
    members = [_member((True, True, True), row, name, name) for name in backends]
    for member in members:
        member.cfg_hpc_system = object()
    reps = _select(_master_of(members))
    assert {rep.key[1] for rep in reps} == {"CUDA", "HIP"}
    assert len(reps) == 2


def test_a_selection_that_drops_a_group_is_reported() -> None:
    """Must-fail control: the totality helper reds on a selection missing one group."""
    master, candidates = _sensitivity_master(TOGGLE_COMBINATIONS["all_models"])
    reps = _select(master)
    _assert_selection_is_total(reps, candidates)
    with pytest.raises(AssertionError, match="missing="):
        _assert_selection_is_total(reps[:-1], candidates)
    with pytest.raises(AssertionError, match="duplicate"):
        _assert_selection_is_total([*reps, reps[0]], candidates)


# ---- _test/ subset construction ---------------------------------------------------------------------------------


def _member_like_config(golden_extractor):
    """A sensitivity MEMBER's configuration as ``_build_test_analyses`` receives it: the sensitivity golden's analysis
    configuration (``toggle_sensitivity_analysis`` true) with its report block replaced by ``REPORT_FIXTURE``, which
    pins a non-default ``reporting_set`` and carries a populated ``sensitivity`` sub-block, the state a member inherits
    from its master.

    ``tests/goldens/SPEC.md`` sections 1 and 5.1 describe the sensitivity golden: one sensitivity experiment with 4
    members and 1 event whose analysis config sets ``toggle_sensitivity_analysis: true``. ``golden_extractor`` (in
    ``tests/fixtures/golden_trees.py``) extracts a golden into the test's ``tmp_path`` without constructing an
    analysis."""
    from hhemt.config.analysis import analysis_config
    from hhemt.config.loaders import load_analysis_config

    root = golden_extractor("sensitivity")
    cfg = load_analysis_config(root / "analysis_config.yaml")
    report = yaml.safe_load(REPORT_FIXTURE.read_text())
    member = analysis_config.model_validate({**cfg.model_dump(), "report": report})
    return root, member


REPORT_FIXTURE = Path(__file__).parent / "fixtures" / "synthetic_model" / "report_config_synth_sensitivity.yaml"
HPC_SENTINEL = Path("/stand-in/hpc_system_config.yaml")


@pytest.mark.parametrize("combination", sorted(TOGGLE_COMBINATIONS))
def test_the_test_subset_is_built_from_each_representative(combination, golden_extractor, monkeypatch) -> None:
    root, cfg = _member_like_config(golden_extractor)
    # The inputs the overlay exists to scrub: without them the asserts below hold whether or not it runs.
    assert cfg.toggle_sensitivity_analysis is True
    assert cfg.report.sensitivity is not None
    # reporting_set is typed ``str | list[str] | None``, but a validated config holds a one-name list (a bare string is
    # normalized to one, and so is the default); the assertion accepts either spelling.
    assert cfg.report.reporting_set not in ("default", ["default"])
    toggles = TOGGLE_COMBINATIONS[combination]
    master, _ = _sensitivity_master(toggles)
    reps = _select(master)
    for rep in reps:
        rep.source_analysis.cfg_analysis = cfg  # the representative's own configuration: a member's, validated
    built: list[dict] = []
    signature = inspect.signature(TRITONSWMM_analysis)

    def recorder(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)  # the real constructor signature, so a rename or drop is caught
        built.append(
            {
                "yaml": bound.arguments["analysis_config_yaml"],
                "system": bound.arguments["system"],
                "hpc": bound.arguments["hpc_system_config_yaml"],
                "main": bound.arguments["is_main_orchestrator"],
            }
        )
        return SimpleNamespace(yaml=bound.arguments["analysis_config_yaml"])

    monkeypatch.setattr(analysis_module, "TRITONSWMM_analysis", recorder)
    analysis_dir = root / cfg.analysis_id
    holder = SimpleNamespace(
        cfg_analysis=cfg,
        analysis_paths=SimpleNamespace(analysis_dir=analysis_dir),
        analysis_config_yaml=root / "analysis_config.yaml",
        hpc_system_config_yaml=HPC_SENTINEL,
    )
    holder._atomic_write_analysis_yaml = lambda cfg_a, sub_dir: TRITONSWMM_analysis._atomic_write_analysis_yaml(
        holder, cfg_a, sub_dir
    )
    subs = TRITONSWMM_analysis._build_test_analyses(holder, reps, n_reporting_timesteps=3, reporting_timestep_s=7)

    test_root = analysis_dir / "_test"
    assert len(subs) == len(reps) == len(built)
    for i, (rep, record) in enumerate(zip(reps, built, strict=True)):
        group = f"group_{i}"
        assert (test_root / group).is_dir()
        assert record["system"] is rep.source_analysis._system, "the sub must run on the representative's own system"
        assert record["hpc"] is HPC_SENTINEL
        assert record["main"] is False
        written = yaml.safe_load(record["yaml"].read_text())
        assert record["yaml"].parent == test_root
        assert written["analysis_id"] == f"{cfg.analysis_id}_test_{group}"
        assert written["toggle_sensitivity_analysis"] is False
        assert written["is_experiment_member"] is True
        assert written["analysis_dir"] == str(test_root / group)
        assert written["experiment_cfg_yaml"] == str(root / "analysis_config.yaml")
        assert written["TRITON_reporting_timestep_s"] == 7
        assert written["report"]["sensitivity"] is None
        assert written["report"]["reporting_set"] in ("default", ["default"])
        short = test_root / f"_test_weather_{group}.nc"
        assert written["weather_timeseries"] == str(short)
        # The slice is ``isel(slice(0, n_reporting_timesteps + 1))``: the first n frames plus one more, so n of 3 gives
        # four.
        with xr.open_dataset(short, engine="h5netcdf") as wx:
            assert wx.sizes[cfg.weather_time_series_timestep_dimension_name] == 4  # 3 frames plus the endpoint


# ---- _capture_binary_provenance --------------------------------------------------------------------------------


def _sub_with_executable(tmp_path):
    exe = tmp_path / "with_exe" / "sims" / "event_index.0" / "build" / "triton.exe"
    exe.parent.mkdir(parents=True)
    exe.write_text("stand-in executable")
    return SimpleNamespace(analysis_paths=SimpleNamespace(analysis_dir=tmp_path / "with_exe")), exe


def _sub_without_executable(tmp_path):
    (tmp_path / "without_exe" / "sims").mkdir(parents=True)
    return SimpleNamespace(analysis_paths=SimpleNamespace(analysis_dir=tmp_path / "without_exe"))


@pytest.fixture
def fake_tools(monkeypatch):
    """``ldd`` and ``readelf`` present, and a ``subprocess.run`` that records its argv instead of spawning one."""
    calls: list[list[str]] = []
    monkeypatch.setattr(shutil, "which", lambda name: f"/stand-in/{name}")

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        return SimpleNamespace(stdout=f"{argv[0]} output")

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


def _assert_arms(sub_with, sub_without, exe, calls) -> None:
    """The with-executable arm captures both tools' text; the without-executable arm returns None, spawning nothing."""
    calls.clear()
    assert analysis_module._capture_binary_provenance(sub_without) is None
    assert calls == []
    prov = analysis_module._capture_binary_provenance(sub_with)
    assert prov == {"ldd_libs": "ldd output", "dynamic_tags": "readelf output"}
    assert calls == [["ldd", str(exe)], ["readelf", "-d", str(exe)]]


def test_provenance_arms_with_and_without_the_executable(tmp_path, fake_tools) -> None:
    sub_with, exe = _sub_with_executable(tmp_path)
    _assert_arms(sub_with, _sub_without_executable(tmp_path), exe, fake_tools)


def test_provenance_arms_exchanged_are_reported(tmp_path, fake_tools) -> None:
    """Must-fail control: handing the helper the two directories the wrong way round reds it."""
    sub_with, exe = _sub_with_executable(tmp_path)
    sub_without = _sub_without_executable(tmp_path)
    _assert_arms(sub_with, sub_without, exe, fake_tools)
    with pytest.raises(AssertionError):
        _assert_arms(sub_without, sub_with, exe, fake_tools)


def test_provenance_takes_the_first_executable_in_sorted_order(tmp_path, fake_tools) -> None:
    sub, exe = _sub_with_executable(tmp_path)
    later = exe.parents[2] / "event_index.9" / "build" / "triton.exe"
    later.parent.mkdir(parents=True)
    later.write_text("later")
    analysis_module._capture_binary_provenance(sub)
    assert fake_tools[0] == ["ldd", str(exe)]


def test_provenance_records_a_failed_capture_and_omits_a_missing_tool(tmp_path, monkeypatch) -> None:
    sub, exe = _sub_with_executable(tmp_path)
    monkeypatch.setattr(shutil, "which", lambda name: "/stand-in/ldd" if name == "ldd" else None)

    def failing_run(argv, **kwargs):
        raise OSError("no such device")

    monkeypatch.setattr(subprocess, "run", failing_run)
    assert analysis_module._capture_binary_provenance(sub) == {"ldd_libs": "<capture failed: no such device>"}
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert analysis_module._capture_binary_provenance(sub) is None


def test_provenance_continues_past_a_missing_ldd_to_readelf(tmp_path, monkeypatch) -> None:
    sub, exe = _sub_with_executable(tmp_path)
    monkeypatch.setattr(shutil, "which", lambda name: "/stand-in/readelf" if name == "readelf" else None)
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        return SimpleNamespace(stdout="readelf output")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert analysis_module._capture_binary_provenance(sub) == {"dynamic_tags": "readelf output"}
    assert calls == [["readelf", "-d", str(exe)]]
