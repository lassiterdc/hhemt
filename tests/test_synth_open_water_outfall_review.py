"""Reviewer test for chunk 4c.1: the node store's driven-outfall series are the capture's, not just finite.

SIMULATION-BEARING: it reuses the coder's `outfall_arm` fixture, so it compiles and runs TRITON-SWMM. Run it on
Rivanna scratch only; it is never run on a laptop.

Tests promoted from the chunk's own test file (their original description follows).

Compile-tier test of the open-water outfall representations on the synthetic coupled model, end to end.

SIMULATION-BEARING: the fixtures compile and run TRITON-SWMM. Run it on Rivanna scratch only, serialized against
the worktree's build cache; it is never run on a laptop (a laptop collects it without compiling or running a
solver).

The solver is NOT the synthetic tier's pin. `driven_stage` needs a solver that reads the cfg key
`swmm_driven_outfalls` and `crown_rim` under `swmm_node_rim_from_dem` needs one that reads
`swmm_dem_rim_exempt_nodes`; the commit pinned in `OUTFALL_SOLVER_SHA` (`tests/fixtures/outfall_tier.py`) is the tip
of a pushed work branch of the solver that reads both. Its arms point their system config at that commit and at their
own software directory, so the shared build of the synthetic pin (`TRITON_PIN`) is neither replaced nor rebuilt. The
one arm that runs on the synthetic pin is the reference of the byte-identity test and builds in the shared tier
through `tritonswmm_cpu_compiled`. The work-branch tier is provisioned by the harness's warm step
(`tests/test_synth_00_compile_models.py::test_provision_the_open_water_outfall_solver_tier`), because a suite chunk
never provisions a shared tier; a standalone session provisions it itself when the first arm builds its case.

Every arm runs ONE coupled scenario of the synthetic model with the matched outfall `SYNTH_OUTFALL` as the single
listed outfall (the synthetic model's other outfall, `dummy_outfall`, has no conduit, so it has no outlet to size
or to drive). Arms, each built once per module and shared by the tests:

  none_pin     no list, on the synthetic pin: what the synthetic pin's solver writes with the keys absent.
  none         no list, on the work-branch solver.
  driven       `driven_stage`, every capture on. The synthetic boundary water level never reaches the outfall's
               cell, so the cell stays dry and the stage the solver sets is the outfall's invert.
  driven_surge `driven_stage` with the boundary water level raised by `_SURGE_OFFSET_M` through the record's datum
               offset, so the outfall's cell is wet and the sewer takes backflow from it.
  driven_clear `driven_stage` with `clear_raw: all`.
  crown        `crown_rim` without the DEM-rim route.
  crown_rim    `crown_rim` with `swmm_node_rim_from_dem`, which exempts the listed node from the DEM rim.

Every arm sets the exchange flux and the coupling diagnostics on, so the exchange side-file and the water balance
are written, and (except `driven_clear`) processes the scenario the way a run processes it. Every chunk-created
name is looked up inside the fixtures and the test bodies, and the first statement of each fixture and test checks
the chunk's configuration fields.
"""

from __future__ import annotations

import filecmp
import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from tests.fixtures import outfall_tier

pytestmark = [pytest.mark.compile_tier, pytest.mark.shared_runs_root, pytest.mark.slow]


def _integral_over(values: np.ndarray, ends: np.ndarray, dt: np.ndarray, t0: float, t1: float) -> float:
    """The integral of a step-wise constant series over [t0, t1], each step holding its value over (end-dt, end]."""
    overlap = np.clip(np.minimum(ends, t1) - np.maximum(ends - dt, t0), 0.0, None)
    return float(np.sum(values * overlap))


def test_the_node_store_series_integrate_to_the_capture_they_were_read_from(outfall_arm) -> None:  # noqa: F811
    """Property: over the node store's reporting window, the time integral of `outfall_discharge_m3s` and of
    `outfall_stage_m` at the driven outfall equals the integral of `q_out_m3s` and of `stage_m` in the solver's
    `outfall_stage.bin` over the same window, to round-off.

    Class: NEW CAPABILITY. Plausible wrong implementation: the ingestion stores the discharge with its sign reversed
    (backflow positive), or in another unit; red under it (run on Rivanna with the ingestion's discharge negated).
    Reachability: the sign is the one fact `test_a_wet_cell_sets_the_stage_to_...` names as killed and does not kill,
    because it asserts only that the minimum is below zero and the surge run's series has both signs.
    Both-states anchor: the assertion turns on the capture's own values, present in the current and any proposed
    ingestion. Must-be-nonzero control: the capture's net discharge integral is asserted away from zero, so a negation
    cannot hide in a zero.
    Killed by: that negation. A second correct implementation averages over the interval with another quadrature of
    the same step-wise series; the integrals agree, so it still passes.
    """
    import pandas as pd

    from hhemt.coupling_diagnostics import read_outfall_stage
    from hhemt.swmm_output_parser import RPT_DATETIME_FORMAT, read_rpt_starting_date

    arm = outfall_arm("driven_surge")
    stage_file = sorted(arm.out_dir.glob("**/outfall_stage.bin"))
    assert len(stage_file) == 1, stage_file
    ds = read_outfall_stage(stage_file[0], stage_file[0].parent / "outfall_stage.nodes")
    ends = np.asarray(ds["simtime"].values, dtype="float64")
    dt = np.asarray(ds["dt"].values, dtype="float64")
    start = pd.to_datetime(read_rpt_starting_date(arm.proc.scen_paths.swmm_hydraulics_rpt), format=RPT_DATETIME_FORMAT)
    times = (pd.to_datetime(arm.store["date_time"].values) - start).total_seconds().to_numpy()
    interval = float(times[1] - times[0])
    t0, t1 = float(times[0] - interval), float(times[-1])

    capture_q = _integral_over(ds["outfall_discharge_m3s"].sel(node_id=_OUTFALL).values, ends, dt, t0, t1)
    capture_h = _integral_over(ds["outfall_stage_m"].sel(node_id=_OUTFALL).values, ends, dt, t0, t1)
    stored_q = float(arm.store["outfall_discharge_m3s"].sel(node_id=_OUTFALL).sum()) * interval
    stored_h = float(arm.store["outfall_stage_m"].sel(node_id=_OUTFALL).sum()) * interval

    assert abs(capture_q) > 1.0, f"the capture's net discharge integral is {capture_q} m3; the sign is not exercised"
    assert stored_q == pytest.approx(capture_q, rel=1e-9, abs=1e-9)
    assert stored_h == pytest.approx(capture_h, rel=1e-9, abs=1e-9)


_OUTFALL = "SYNTH_OUTFALL"


_RASTER_PREFIXES = ("H_", "QX_", "QY_", "MH_")


_EDGES = ("west", "east", "south", "north")


_RIM_LINE = "coupled-node rims set from the DEM:"


_SURGE_OFFSET_M = 1.0


_HEXTRA_M = 0.001


#: The toolkit sizes a circular conduit through `scenario.calc_area`, which rounds the full area to 0.01 m2 and takes
#: pi as 3.1415, so a 1.0 m circle (0.785 m2) is sized as 0.79 m2: a diameter 0.29 percent over the exact circle's.
#: The bound is the largest the rounding can give on this conduit (half of 0.01 m2 on 0.785 m2 is 0.64 percent of the
#: area, 0.32 percent of the diameter); a rectangular conduit is not rounded, so the Norfolk values are exact.
_CIRCULAR_AREA_ROUNDING_REL = 4e-3


_BASE_CONFIG = {
    "toggle_coupling_exchange_flow": True,
    "toggle_coupling_diagnostics": True,
    "toggle_consolidate_timeseries": True,
    "toggle_clear_scenario_timeseries_after_consolidation": False,
}


_ARM_CONFIGS: dict[str, dict] = {
    "none_pin": {},
    "none": {},
    "driven": {"swmm_open_water_outfalls": [_OUTFALL], "open_water_outfall_representation": "driven_stage"},
    "driven_surge": {
        "swmm_open_water_outfalls": [_OUTFALL],
        "open_water_outfall_representation": "driven_stage",
        "water_level_boundary_variable_datum_offsets": {"water_level": _SURGE_OFFSET_M},
    },
    "driven_clear": {
        "swmm_open_water_outfalls": [_OUTFALL],
        "open_water_outfall_representation": "driven_stage",
        "clear_raw": "all",
    },
    "crown": {"swmm_open_water_outfalls": [_OUTFALL], "open_water_outfall_representation": "crown_rim"},
    "crown_rim": {
        "swmm_open_water_outfalls": [_OUTFALL],
        "open_water_outfall_representation": "crown_rim",
        "swmm_node_rim_from_dem": True,
    },
}


#: The arms that run on the work-branch solver; `none_pin` runs on the synthetic pin.
_ON_OUTFALL_SOLVER = ("none", "driven", "driven_surge", "driven_clear", "crown", "crown_rim")


def _require_chunk() -> None:
    """Fail with an AssertionError, after the compile fixture and before any arm is built, when a field is missing."""
    from hhemt.config.analysis import analysis_config

    missing = [
        f
        for f in ("swmm_open_water_outfalls", "open_water_outfall_representation", "swmm_node_rim_from_dem")
        if f not in analysis_config.model_fields
    ]
    assert not missing, f"analysis_config has no field {missing}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class _Arm:
    """What one run left, read once so the tests share it and no test touches the solver."""

    name: str
    analysis: object
    proc: object
    root: Path
    nodes: list[str]  # the coupling nodes, in the toolkit's list order
    out_dir: Path
    rasters: dict[str, str]  # path relative to the coupled output directory -> sha256 of the bytes
    sidefile_sha: str
    domain_balance_sha: str | None
    cfg_text: str
    inp_text: str
    store: xr.Dataset  # the coupled SWMM node store, loaded: it outlives the zarr handle


def _build_case(root: Path, name: str):
    """Build one coupled synthetic case; the work-branch arms point at the solver that reads the new keys."""
    from tests.fixtures.test_case_builder import retrieve_synth_TRITON_SWMM_test_case

    system_extra: dict = {}
    if name in _ON_OUTFALL_SOLVER:
        system_extra = {
            "TRITONSWMM_branch_key": outfall_tier.OUTFALL_SOLVER_SHA,
            "TRITONSWMM_software_directory": str(outfall_tier.outfall_software_directory()),
        }
    return retrieve_synth_TRITON_SWMM_test_case(
        analysis_name="open_water_outfall",
        n_events=1,
        toggle_tritonswmm_model=True,
        toggle_triton_model=False,
        toggle_swmm_model=False,
        start_from_scratch=True,
        runs_root_override=root,
        additional_analysis_configs={**_BASE_CONFIG, **_ARM_CONFIGS[name]},
        additional_system_configs=system_extra,
    )


def _scenario_store(proc, name: str) -> xr.Dataset:
    """The per-scenario store of one capture, loaded."""
    from hhemt.process_simulation import coupling_capture_store_path

    path = coupling_capture_store_path(proc.scen_paths, name, "zarr")
    assert path.exists(), f"no per-scenario store for {name} at {path}"
    return proc._open(path).load()


def _run_arm(root: Path, name: str) -> _Arm:
    """Compile (the work-branch solver), prepare, run, process and consolidate one scenario; return what it left."""
    case = _build_case(root, name)
    analysis = case.analysis
    if name in _ON_OUTFALL_SOLVER:
        analysis._system.compile_TRITON_SWMM(backends=["cpu"], recompile_if_already_done_successfully=False)
    analysis.run_prepare_scenarios_serially(
        overwrite_scenario_if_already_set_up=True, rerun_swmm_hydro_if_outputs_exist=True
    )
    analysis.run_sims_in_sequence(pickup_where_leftoff=False)
    proc = analysis._retrieve_sim_run_processing_object(analysis.df_sims.index[0])
    cleared = _ARM_CONFIGS[name].get("clear_raw") == "all"
    proc.write_timeseries_outputs(which="both", model_type="tritonswmm", override_clear_raw=None if cleared else "none")
    if not cleared:
        proc.write_summary_outputs(which="both", model_type="tritonswmm")
        analysis.process.consolidate_to_datatree()

    from hhemt.scenario import coupling_nodes_file_for

    out = Path(proc.scen_paths.out_tritonswmm)
    raster_files = sorted(f for f in (out / "bin").glob("*") if f.is_file() and f.name.startswith(_RASTER_PREFIXES))
    sidefiles = sorted(out.glob("**/swmm/*_exchange_replay.bin"))
    balance_files = sorted(out.glob("**/domain_balance.csv"))
    if not cleared:
        assert raster_files, f"{name}: no raster file in {out / 'bin'}"
        assert len(sidefiles) == 1, f"{name}: expected the one raw exchange side-file, found {sidefiles}"
        assert len(balance_files) == 1, f"{name}: expected the one raw domain_balance.csv, found {balance_files}"
    store_path = Path(proc.scen_paths.output_tritonswmm_node_time_series)
    with xr.open_zarr(store_path) if store_path.suffix == ".zarr" else xr.open_dataset(store_path) as opened:
        store = opened.load()
    return _Arm(
        name=name,
        analysis=analysis,
        proc=proc,
        root=root,
        nodes=[
            line.strip()
            for line in coupling_nodes_file_for(proc.scen_paths.swmm_hydraulics_inp).read_text().splitlines()
            if line.strip()
        ],
        out_dir=out,
        rasters={str(f.relative_to(out)): _sha256(f) for f in raster_files},
        sidefile_sha=_sha256(sidefiles[0]) if sidefiles else "",
        domain_balance_sha=_sha256(balance_files[0]) if balance_files else None,
        cfg_text=Path(proc.scen_paths.triton_swmm_cfg).read_text(),
        inp_text=Path(proc.scen_paths.swmm_hydraulics_inp).read_text(),
        store=store,
    )


@pytest.fixture(scope="module")
def outfall_arm(tritonswmm_cpu_compiled, tmp_path_factory):
    """A getter for the module's arms: `outfall_arm("driven")` runs the arm on first use and returns it after.

    Requests `tritonswmm_cpu_compiled` as a parameter so the collection-time fixture closure sees the compile of the
    synthetic pin (the reference arm builds there; the work-branch arms compile their own software directory).
    """
    built: dict[str, _Arm | BaseException] = {}

    def get(name: str) -> _Arm:
        """Return the named arm, building it on first request; a failed build is remembered and re-raised.

        Remembering the failure keeps one red arm to one coupled run per module instead of repeating the build for
        every test that asks for the arm.
        """
        _require_chunk()
        if name not in built:
            try:
                built[name] = _run_arm(tmp_path_factory.mktemp(f"owo_{name}"), name)
            except Exception as exc:  # noqa: BLE001 -- stored and re-raised for every later request of this arm
                built[name] = exc
        outcome = built[name]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    return get


def _outfall_invert(arm: _Arm) -> float:
    """The listed outfall's invert elevation in meters, read from the scenario's coupled SWMM model."""
    import swmmio

    outfalls = swmmio.Model(str(arm.proc.scen_paths.swmm_hydraulics_inp)).inp.outfalls
    names = [str(n).upper() for n in outfalls.index]
    return float(outfalls["InvertElev"].iloc[names.index(_OUTFALL)])


def _normalized(text: str, arm: _Arm) -> str:
    """The text with this arm's run directory replaced, so two arms that differ only by where they ran compare equal."""
    return text.replace(str(arm.root), "<ROOT>")


def _balance_terms(store: xr.Dataset, extra: tuple[str, ...] = ()) -> np.ndarray:
    names = ["runoff_m3", "inflow_m3", "extbc_m3", "clamp_m3", "swmm_exchange_m3", *extra]
    names += [f"edge_{e}_m3" for e in _EDGES]
    return np.stack([store[t].values for t in names])


def test_with_no_list_the_work_branch_solver_writes_what_the_pinned_solver_wrote(outfall_arm) -> None:
    """Property: with `swmm_open_water_outfalls` empty, the coupled scenario on the work-branch solver is the
    scenario on the synthetic pin byte for byte: the same TRITON cfg and SWMM model (after the run directory is
    substituted), the same rasters, the same exchange side-file and the same `domain_balance.csv`; no driven-outfall
    file, capture or variable exists; and the node store carries none of the outfall variables.

    Class: SOLVER NEUTRALITY plus KEY ABSENCE. Both arms are built by one checkout of the toolkit and differ only in
    the solver commit and its software directory, so a toolkit change that alters the cfg or the model identically
    for both arms (a rewrite under an empty list) is NOT caught here: only the two named cfg keys and the capture
    files are asserted absent, and toolkit neutrality belongs to `tests/test_open_water_outfalls.py`. Plausible wrong
    implementations: a solver that writes `outfall_stage.bin`, or the two balance
    columns, when the key is absent; a build whose neutral path differs in the last bit from the pin's.
    Reachability and control: both arms ran and each left a non-empty raster set (a vacuous equality of two empty
    sets is refused), and `test_a_driven_outfall_changes_the_rasters...` varies the property under suspicion, the
    list, and must see the same comparison differ.
    Killed by: any difference above. A second correct implementation writes the unchanged cfg in another order; it
    fails the cfg comparison, so the comparison is on the arms' own bytes, not on a regenerated reference.
    """
    _require_chunk()
    pin, new = outfall_arm("none_pin"), outfall_arm("none")
    assert pin.rasters and new.rasters.keys() == pin.rasters.keys()
    assert new.rasters == pin.rasters
    assert new.sidefile_sha == pin.sidefile_sha
    assert new.domain_balance_sha is not None and new.domain_balance_sha == pin.domain_balance_sha
    assert _normalized(new.cfg_text, new) == _normalized(pin.cfg_text, pin)
    assert _normalized(new.inp_text, new) == _normalized(pin.inp_text, pin)
    for key in ("swmm_driven_outfalls", "swmm_dem_rim_exempt_nodes"):
        assert key not in new.cfg_text
    assert not list(new.out_dir.glob("**/outfall_stage.*"))
    for variable in ("outfall_stage_m", "outfall_discharge_m3s", "is_open_water_outfall"):
        assert variable not in new.store, f"the no-list node store carries {variable}"
    balance = _scenario_store(new.proc, "domain_balance")
    assert "outfall_backflow_m3" not in balance and "outfall_backflow_deficit_m3" not in balance


def test_a_driven_outfall_writes_its_stage_capture_and_the_ingested_series(outfall_arm) -> None:
    """Property: under `driven_stage` the toolkit writes `driven_outfalls.txt` naming the listed outfall and one cfg
    line for it; the solver writes `outfall_stage.bin` and its sidecar `outfall_stage.nodes` naming it; the node store
    carries `outfall_stage_m` and `outfall_discharge_m3s` at that node and nowhere else, and flags it in
    `is_open_water_outfall`; and the consolidated tree carries the same variables.

    Class: NEW CAPABILITY. Plausible wrong implementations: a cfg line the solver ignores (no capture file); a
    store whose series is on another node (the series is read by node name); the flag on every node; NaN at the
    outfall because the ingestion was skipped.
    Reachability and control: the run is the same synthetic scenario as `none`, whose store carries none of the
    variables (`test_with_no_list...`). The listed outfall lies ON the water-level boundary line (distance 0.0 m),
    and the base boundary level never wets its cell, so the stage is the invert at every interval (the solver's
    dry rule); `test_a_wet_cell_sets_the_stage_to_the_cells_level...` varies that property, the cell's wetness,
    and must see the stage leave the invert.
    Killed by: those. A second correct implementation lists the sidecar's names in another case; names are folded.
    """
    _require_chunk()
    from hhemt.scenario import driven_outfalls_file_for

    arm = outfall_arm("driven")
    inp = arm.proc.scen_paths.swmm_hydraulics_inp
    assert driven_outfalls_file_for(inp).read_text().split() == [_OUTFALL]
    assert arm.cfg_text.count("swmm_driven_outfalls") == 1
    assert _OUTFALL not in arm.nodes, "a driven outfall must not be a coupling node"
    stage_files = sorted(arm.out_dir.glob("**/outfall_stage.bin"))
    sidecars = sorted(arm.out_dir.glob("**/outfall_stage.nodes"))
    assert len(stage_files) == 1 and len(sidecars) == 1, (stage_files, sidecars)
    assert stage_files[0].stat().st_size > 0
    assert sidecars[0].read_text().split() == [_OUTFALL]

    store = arm.store
    assert store["is_open_water_outfall"].sel(node_id=_OUTFALL).item() == 1
    assert int(store["is_open_water_outfall"].sum()) == 1
    for variable in ("outfall_stage_m", "outfall_discharge_m3s"):
        series = store[variable]
        at = series.sel(node_id=_OUTFALL).values
        assert np.isfinite(at).all(), f"{variable} at {_OUTFALL} has non-finite values"
        elsewhere = series.sel(node_id=[n for n in store["node_id"].values if n != _OUTFALL]).values
        assert np.isnan(elsewhere).all(), f"{variable} is not NaN at every other node"
    stage = store["outfall_stage_m"].sel(node_id=_OUTFALL).values
    invert = _outfall_invert(arm)
    np.testing.assert_allclose(
        stage, invert, rtol=0.0, atol=1e-9, err_msg="a dry cell must give the outfall's invert as its stage"
    )

    tree = xr.open_datatree(arm.analysis.analysis_paths.analysis_datatree_zarr, engine="zarr", consolidated=False)
    try:
        flat = {name for node in tree.subtree for name in node.dataset.data_vars}
    finally:
        tree.close()
    assert {"outfall_stage_m", "outfall_discharge_m3s", "is_open_water_outfall"} <= flat


def test_a_driven_outfall_closes_the_domain_balance_with_the_two_new_terms(outfall_arm) -> None:
    """Property: with a driven outfall the water balance gains `outfall_backflow_m3` (never positive: it is the
    negative of the backflow volume taken from the cells) and `outfall_backflow_deficit_m3` (never negative), and each
    interval's residual equals the storage change minus the signed sum of every other term, the new one included.

    Class: NEW CAPABILITY. Plausible wrong implementations: a balance that omits the backflow term (the residual
    then carries it); the two columns swapped or read as the wrong sign; the columns absent.
    Reachability and control: the run is `driven_surge`, whose outfall cell is wet, because on the base boundary
    level the backflow is zero throughout and the closure holds for any value of the new column. The closure is also
    computed WITHOUT the new term and must then fail, and the backflow is asserted nonzero somewhere (the
    must-be-nonzero control).
    Not distinguished: the deficit column is identically zero in the surge arm (1080 rows, max 0.0), so where the
    deficit is booked (inside or outside the sum) is not discriminated, and `deficit >= 0` passes for an all-zero
    column. The nonzero-deficit case belongs to the solver's own suite.
    Killed by: a balance omitting the backflow term, and a wrong-signed backflow column.
    """
    _require_chunk()
    arm = outfall_arm("driven_surge")
    balance = _scenario_store(arm.proc, "domain_balance")
    backflow, deficit = balance["outfall_backflow_m3"].values, balance["outfall_backflow_deficit_m3"].values
    assert (backflow <= 0.0).all() and (deficit >= 0.0).all()
    with_term = _balance_terms(balance, ("outfall_backflow_m3",))
    without = _balance_terms(balance)
    storage, residual = balance["d_storage_m3"].values, balance["residual_m3"].values
    scale = max(1.0, float(np.abs(with_term).max()), float(np.abs(storage).max()))
    np.testing.assert_allclose(residual, storage - with_term.sum(axis=0), rtol=1e-9, atol=1e-9 * scale)
    assert np.any(backflow < 0.0), "no backflow at the driven outfall in the run, so the new term was never exercised"
    assert not np.allclose(residual, storage - without.sum(axis=0), rtol=1e-9, atol=1e-9 * scale)


def test_a_wet_cell_sets_the_stage_to_the_cells_level_and_the_sewer_takes_backflow(outfall_arm) -> None:
    """Property: when the boundary level wets the outfall's cell, the stage in the node store is above the invert by
    more than `TRITON_hextra_m` (the solver's wet rule), the outfall's discharge goes negative (backflow into the
    sewer), and the stage is not the invert it was on the dry run.

    Class: NEW CAPABILITY. Plausible wrong implementations: the invert always set (the dry rule applied to every
    cell); the stage ignored.
    NOT distinguished: a stage read from another wet cell, and a reversed discharge sign. The outfall lies on the
    water-level boundary line, so its stage is the pinned boundary level and any wet cell clears the threshold; the
    reversed sign (the surge arm's discharge ranges from about -3.3 to +0.36 m3/s, so a negated series still has a
    negative minimum) is killed by `tests/test_synth_open_water_outfall_review.py`, which was shown red with the
    ingestion negated while this file stayed green.
    Reachability and control: the same scenario on the base boundary level (`driven`) has the stage at the invert and
    no negative discharge (`test_a_driven_outfall_writes_...`), so the wetness of the cell is the one property
    varied (the datum offset `_SURGE_OFFSET_M` raises the whole record).
    Killed by: the invert-always and ignored-stage implementations. A second correct implementation sets the stage
    in another unit and converts back; the node store is in meters either way.
    """
    _require_chunk()
    arm = outfall_arm("driven_surge")
    invert = _outfall_invert(arm)
    stage = arm.store["outfall_stage_m"].sel(node_id=_OUTFALL).values
    discharge = arm.store["outfall_discharge_m3s"].sel(node_id=_OUTFALL).values
    assert float(stage.max()) > invert + _HEXTRA_M, (
        f"stage never left the invert {invert}: {stage.min()}..{stage.max()}"
    )
    assert float(discharge.min()) < 0.0, "no backflow at the outfall"


def test_a_driven_outfall_changes_the_rasters(outfall_arm) -> None:
    """Property: listing the outfall under `driven_stage` changes what the solver writes: at least one raster differs
    from the no-list run on the same solver.

    Class: HYDRAULICS-CHANGING BY DESIGN (the representation replaces the coupled-junction exchange at the node).
    Plausible wrong implementation: a build that ignores the key (every raster equal to the no-list run).
    Reachability and control: the no-list arm has a non-empty raster set; the same comparison on `none` against
    `none_pin` is the equality `test_with_no_list...` asserts, so the comparison discriminates.
    Killed by: a solver that ignores the key. A second correct implementation moves different cells; the
    assertion is only that some raster differs.
    """
    _require_chunk()
    none, driven = outfall_arm("none"), outfall_arm("driven")
    assert none.rasters and driven.rasters.keys() == none.rasters.keys()
    changed = [k for k in none.rasters if none.rasters[k] != driven.rasters[k]]
    assert changed, "every raster of the driven run equals the no-list run; the solver did not apply the stage"


def test_clear_raw_all_leaves_the_stage_capture_and_the_ingestion_read_it(outfall_arm) -> None:
    """Property: after a run with `clear_raw: all`, `outfall_stage.bin` and its sidecar are still present under the
    TRITON output directory (the solver writes them at the top level of `out_tritonswmm/`, beside the other
    captures, and the clearing removes only the raster directories), and the node-store ingestion read them: the
    node store carries `outfall_stage_m` and `outfall_discharge_m3s` finite at the outfall.

    Class: REGRESSION GUARD of the clear-raw allowlist. Plausible wrong implementations: a clearing that deletes the
    capture before the node store is written (the ingestion then raises, or the series is absent); a clearing that
    deletes it after, leaving no raw record.
    Reachability and control: the clearing ran, shown by the raster directory and the exchange side-file being gone
    from the same run (a vacuous presence in a tree nothing was cleared from is refused); `none` and `driven` keep
    their rasters, so the property under suspicion, the clearing, is the one varied.
    Killed by: those. A second correct implementation moves the capture under `swmm/`; the assertion finds it at any
    depth, as the toolkit does.
    """
    _require_chunk()
    arm = outfall_arm("driven_clear")
    assert not (arm.out_dir / "bin").exists(), "the clearing did not run: the raster directory is still there"
    assert not list(arm.out_dir.glob("**/swmm/*_exchange_replay.bin"))
    stage_files = sorted(arm.out_dir.glob("**/outfall_stage.bin"))
    sidecars = sorted(arm.out_dir.glob("**/outfall_stage.nodes"))
    assert len(stage_files) == 1 and len(sidecars) == 1, (stage_files, sidecars)
    for variable in ("outfall_stage_m", "outfall_discharge_m3s"):
        assert variable in arm.store, f"the node store of the cleared run has no {variable}"
        assert np.isfinite(arm.store[variable].sel(node_id=_OUTFALL).values).all()


def _expected_crown_diameter(arm: _Arm) -> float:
    """The diameter of the circle of the outlet conduit's total flow area, read from the model's own tables."""
    import swmmio

    model = swmmio.Model(str(arm.proc.scen_paths.swmm_hydraulics_inp))
    conduits = model.inp.conduits
    xsections = model.inp.xsections
    outlet = conduits[conduits["OutletNode"].astype(str).str.upper() == _OUTFALL]
    assert len(outlet) == 1, f"{_OUTFALL} has {len(outlet)} inlet conduits; the rule reads exactly one"
    conduit = str(outlet.index[0])
    row = xsections.loc[conduit]
    assert str(row["Shape"]).upper() == "CIRCULAR", f"the synthetic outlet is {row['Shape']}, not circular"
    barrels = float(row["Barrels"]) if "Barrels" in row and not np.isnan(float(row["Barrels"])) else 1.0
    area = math.pi / 4.0 * float(row["Geom1"]) ** 2 * barrels
    return math.sqrt(4.0 * area / math.pi)


@pytest.mark.parametrize("arm_name", ["crown", "crown_rim"])
def test_crown_rim_runs_with_the_outlet_conduits_area_equivalent_diameter(outfall_arm, arm_name) -> None:
    """Property: under `crown_rim` the listed outfall is a coupled node whose manhole diameter in the node store is
    the diameter of the circle of the outlet conduit's total flow area, whose loss coefficient is the analysis
    constant, and whose value differs from the constant diameter; every other coupling node keeps both constants;
    and the run completes with and without `swmm_node_rim_from_dem`. The values are read from the
    `manhole_table.txt` the toolkit wrote and the solver read: the node store carries `manhole_diameter` only
    when the analysis sets `manhole_table`.

    Class: NEW CAPABILITY. Plausible wrong implementations: the constant diameter at the node (the table row not
    written or not read); the barrels ignored; the diameter written into the loss column; the value applied to
    another node.
    Reachability and control: the expected diameter is computed from the model's own conduit table with the exact
    circle formula (within the rounding bound of `_CIRCULAR_AREA_ROUNDING_REL`; the synthetic outlet is one circular
    barrel, so the barrel multiplication is not exercised here and the Norfolk twin barrels are the laptop
    test's), and asserted different from the constant so a node left at the constant fails; the table's other rows
    are the constants, so a row written for the wrong node fails on that node.
    Not distinguished: the area-equivalent rule against the rejected perimeter-equivalent rule or `Geom1`: for a
    one-barrel circular 1.0 m outlet all three give 1.0 m. The rule choice is carried by the Norfolk rectangles in
    `tests/test_open_water_outfalls.py`.
    Killed by: the constant diameter at the node, a row on the wrong node, and a diameter in the loss column.
    """
    _require_chunk()
    from hhemt.process_simulation import read_written_manhole_table
    from hhemt.scenario import manhole_table_file_for

    arm = outfall_arm(arm_name)
    cfg = arm.analysis.cfg_analysis
    constant_diameter, constant_loss = float(cfg.manhole_diameter), float(cfg.manhole_loss_coefficient)
    expected = _expected_crown_diameter(arm)
    assert abs(expected - constant_diameter) > 1e-6, "the fixture's outlet diameter equals the constant"
    assert _OUTFALL in arm.nodes, "the crown_rim outfall is not a coupling node"
    table = manhole_table_file_for(arm.proc.scen_paths.swmm_hydraulics_inp)
    diameter_values, loss_values = read_written_manhole_table(table, arm.nodes)
    diameters, losses = (
        dict(zip(arm.nodes, diameter_values, strict=True)),
        dict(zip(arm.nodes, loss_values, strict=True)),
    )
    assert diameters[_OUTFALL] == pytest.approx(expected, rel=_CIRCULAR_AREA_ROUNDING_REL)
    assert losses[_OUTFALL] == pytest.approx(constant_loss, rel=1e-12)
    for node in set(arm.nodes) - {_OUTFALL}:
        assert diameters[node] == pytest.approx(constant_diameter, rel=1e-12), node
        assert losses[node] == pytest.approx(constant_loss, rel=1e-12), node
    assert arm.cfg_text.count("swmm_manhole_table") == 1
    assert arm.store["is_open_water_outfall"].sel(node_id=_OUTFALL).item() == 1
    assert "outfall_stage_m" not in arm.store, "a crown_rim run carries a driven-outfall series"
    assert not list(arm.out_dir.glob("**/outfall_stage.bin"))


def test_crown_rim_under_the_dem_rim_route_exempts_the_listed_node_only(outfall_arm) -> None:
    """Property: with `swmm_node_rim_from_dem` and `crown_rim`, preparation writes `dem_rim_exempt_nodes.txt` naming
    the listed outfall and one cfg line for it, and the solver's rim line counts one node fewer than the coupling
    nodes; without the DEM-rim route no exemption file or line is written.

    Class: NEW CAPABILITY. Plausible wrong implementations: the exemption written without the route (an inert key
    the solver refuses); the route on with no exemption (every coupling node, the listed one included, takes the DEM
    rim); the exemption naming every node.
    Reachability and control: the `crown` arm is the same list without the route and shows the file and line absent;
    the rim line is read from the solver's own log, and the count is compared with the coupling-node list of the
    same scenario, so a build that ignores the exemption counts every node and fails.
    Killed by: those. A second correct implementation exempts through a differently named list; the assertion is
    on the solver's count, which does not depend on the file's name.
    """
    _require_chunk()
    from hhemt.scenario import dem_rim_exempt_nodes_file_for

    plain, rim = outfall_arm("crown"), outfall_arm("crown_rim")
    assert not dem_rim_exempt_nodes_file_for(plain.proc.scen_paths.swmm_hydraulics_inp).exists()
    assert "swmm_dem_rim_exempt_nodes" not in plain.cfg_text
    assert dem_rim_exempt_nodes_file_for(rim.proc.scen_paths.swmm_hydraulics_inp).read_text().split() == [_OUTFALL]
    assert rim.cfg_text.count("swmm_dem_rim_exempt_nodes") == 1
    logs = sorted(rim.root.rglob("model_tritonswmm_evt-*.log"))
    assert len(logs) == 1, f"expected the one solver log of the scenario, found {logs}"
    text = re.sub(r"\x1b\[[0-9;]*m", "", logs[0].read_text(errors="replace"))
    counts = {int(m.group(1)) for m in re.finditer(re.escape(_RIM_LINE) + r"\s*(\d+) nodes", text)}
    assert counts == {len(rim.nodes) - 1}, (counts, len(rim.nodes))
    assert f"SWMM coupled node {_OUTFALL}: DEM rim exempt" in text
    assert text.count("DEM rim exempt") == 1, "the exemption names more than the listed node"
    plain_text = re.sub(r"\x1b\[[0-9;]*m", "", sorted(plain.root.rglob("model_tritonswmm_evt-*.log"))[0].read_text())
    assert "DEM rim exempt" not in plain_text and _RIM_LINE not in plain_text
    assert filecmp.cmp(  # the exemption moves only the rim: the coupling list itself is the one `crown` wrote
        Path(rim.proc.scen_paths.swmm_hydraulics_inp).parent / "coupling_nodes.txt",
        Path(plain.proc.scen_paths.swmm_hydraulics_inp).parent / "coupling_nodes.txt",
        shallow=False,
    )
