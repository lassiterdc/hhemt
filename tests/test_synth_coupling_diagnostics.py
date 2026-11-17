"""Compile-tier test of the coupling-diagnostics captures end to end on the synthetic coupled model: what the
solver at the pinned commit writes is ingested into the per-scenario stores, the coupled node store and the
consolidated tree with the names, units and attributes the decoders define, and the independent SWMM report step
(row H7) leaves the TRITON rasters and the exchange side-file byte for byte unchanged.

SIMULATION-BEARING: each arm compiles (through `tritonswmm_cpu_compiled`) and runs TRITON-SWMM. Run it on Rivanna
scratch only, serialized against the worktree's build cache; it is never run on a laptop, where it only collects.
The solver is the one `TRITON_PIN` names in `tests/fixtures/_triton_source_cache.py`; no pin override is applied.

Four arms, one scenario each, built once per module and shared by the tests below:

  off          every capture toggle off, the report step at its default.
  on           every capture on: `toggle_coupling_diagnostics` (T1a, T1b, T2, T5, T6), `swmm_diagnostic_links: ALL`
               (T1d) and `swmm_diagnostic_node_path3_per_step` (T1c).
  report_step  captures off, `swmm_report_step_s: 5` against the 10 s TRITON reporting step (H7).
  t3           captures on, `swmm_coupling_interval: 2` and `swmm_coupling_lag: 1` (T3).

Every arm sets the exchange flux on (so the exchange side-file is read), `toggle_consolidate_timeseries` on and the
clearing of the per-scenario stores explicitly off, so the per-scenario stores and the raw files are still there
when the tree has been written. The scenario is processed the way a run processes it (timeseries outputs, summary
outputs, consolidation to the tree).

The per-test properties name the solver-side fact each identity rests on (the file and the statement), because the
decoders and the solver were written by different parties and an identity that holds on both is the only evidence
the ingested names sit on the right columns. The T4 timers are asserted for the one property the toolkit owns:
`DIAG` is nonzero when a capture is on and exactly zero when none is, and the two `SWMM_*` timers nest in
`SWMM_STEP`. Whether capture work leaks into another category is the solver suite's differential (cases N2, N3 and
N10 of the neutrality suite), not this file's.
"""

from __future__ import annotations

import filecmp
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

# `shared_runs_root` is declared HERE as `tests/test_synth_coupling_flux.py` declares it: the compiled software lives
# under the shared slug root and is built once, serialized by its own per-build lock, while the analysis trees go
# under `tmp_path_factory`. Solver-bearing: runs on the cluster; the laptop capture selects `not slow`.
pytestmark = [pytest.mark.compile_tier, pytest.mark.shared_runs_root, pytest.mark.slow]

_CAPTURE_NAMES = (
    "swmm_step",
    "swmm_node_path3",
    "edge_flux",
    "domain_balance",
    "wet_cells",
    "dt_limiter",
    "swmm_link_flow",
)
_EDGES = ("west", "east", "south", "north")
_RASTER_PREFIXES = ("H_", "QX_", "QY_", "MH_")

_UNIFORM_CONFIG = {
    "toggle_coupling_exchange_flow": True,
    "toggle_consolidate_timeseries": True,
    "toggle_clear_scenario_timeseries_after_consolidation": False,
}
_CAPTURES_ON = {
    "toggle_coupling_diagnostics": True,
    "swmm_diagnostic_links": "ALL",
    "swmm_diagnostic_node_path3_per_step": True,
}
_ARM_CONFIGS: dict[str, dict] = {
    "off": {},
    "on": _CAPTURES_ON,
    "report_step": {"swmm_report_step_s": 5},
    "t3": {**_CAPTURES_ON, "swmm_coupling_interval": 2, "swmm_coupling_lag": 1},
}


def _require_chunk() -> None:
    """Fail with an AssertionError, before any arm is built, when a field this file drives does not exist."""
    from hhemt.config.analysis import analysis_config

    missing = [
        f
        for f in (
            "toggle_coupling_diagnostics",
            "swmm_diagnostic_links",
            "swmm_diagnostic_node_path3_per_step",
            "swmm_coupling_interval",
            "swmm_coupling_lag",
            "swmm_report_step_s",
        )
        if f not in analysis_config.model_fields
    ]
    assert not missing, f"analysis_config has no field {missing}"


def _run_arm(runs_root: Path, name: str) -> SimpleNamespace:
    """Build, run, process and consolidate one coupled synthetic scenario; return its handles."""
    from tests.fixtures.test_case_builder import retrieve_synth_TRITON_SWMM_test_case

    case = retrieve_synth_TRITON_SWMM_test_case(
        analysis_name=f"coupling_diag_{name}",
        n_events=1,
        toggle_tritonswmm_model=True,
        toggle_triton_model=False,
        toggle_swmm_model=False,
        start_from_scratch=True,
        runs_root_override=runs_root,
        additional_analysis_configs={**_UNIFORM_CONFIG, **_ARM_CONFIGS[name]},
    )
    analysis = case.analysis
    analysis.run_prepare_scenarios_serially(
        overwrite_scenario_if_already_set_up=True, rerun_swmm_hydro_if_outputs_exist=True
    )
    analysis.run_sims_in_sequence(pickup_where_leftoff=False)
    event_iloc = int(analysis.df_sims.index[0])
    proc = analysis._retrieve_sim_run_processing_object(event_iloc)
    proc.write_timeseries_outputs(which="both", model_type="tritonswmm", override_clear_raw="none")
    proc.write_summary_outputs(which="both", model_type="tritonswmm")
    analysis.process.consolidate_to_datatree()
    return SimpleNamespace(
        name=name,
        analysis=analysis,
        proc=proc,
        event_iloc=event_iloc,
        tree=Path(analysis.analysis_paths.analysis_datatree_zarr),
        out_dir=Path(proc.scen_paths.out_tritonswmm),
    )


@pytest.fixture(scope="module")
def coupled_arm(tritonswmm_cpu_compiled, tmp_path_factory):
    """A getter for the module's arms: `coupled_arm("on")` runs the arm on first use and returns the same handles after.

    Requests `tritonswmm_cpu_compiled` as a parameter so the collection-time fixture closure sees the compile and the
    compile guard and the runs-root derivation treat the file as the compile-bearing one it is.
    """
    built: dict[str, SimpleNamespace | BaseException] = {}

    def get(name: str) -> SimpleNamespace:
        """Return the named arm, building it on the first request; a failed build is remembered and re-raised.

        Remembering the failure keeps one red arm build to one coupled run per module: every later test that asks for
        the arm fails at once with the first failure instead of repeating a full build of a 180 minute model.
        """
        _require_chunk()
        if name not in built:
            try:
                built[name] = _run_arm(tmp_path_factory.mktemp(f"coupling_diag_{name}"), name)
            except Exception as exc:  # noqa: BLE001 -- stored and re-raised for every later request of this arm
                built[name] = exc
        outcome = built[name]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    return get


# ----------------------------------------------------------------------------------------------------------------
# Readers of what an arm wrote
# ----------------------------------------------------------------------------------------------------------------


def _scenario_store(arm, name: str) -> xr.Dataset:
    """The per-scenario store of one capture, loaded."""
    from hhemt.process_simulation import coupling_capture_store_path

    path = coupling_capture_store_path(arm.proc.scen_paths, name, "zarr")
    assert path.exists(), f"arm {arm.name}: no per-scenario store for {name} at {path}"
    return arm.proc._open(path).load()


def _group(arm, group: str) -> xr.Dataset:
    """One group of the arm's consolidated tree, loaded."""
    return xr.open_zarr(arm.tree, group=group, consolidated=False, chunks=None, decode_timedelta=False).load()


def _root_attrs(arm) -> dict:
    """The root attributes of the arm's consolidated tree."""
    tree = xr.open_datatree(arm.tree, engine="zarr", consolidated=False)
    try:
        return dict(tree.attrs)
    finally:
        tree.close()


def _node_store(arm) -> xr.Dataset:
    """The arm's per-scenario coupled SWMM node store, loaded."""
    return arm.proc._open(arm.proc.scen_paths.output_tritonswmm_node_time_series).load()


def _raw_files(arm, file_name: str) -> list[Path]:
    """Every file called ``file_name`` under the arm's coupled output directory, in path order."""
    return sorted(arm.out_dir.glob(f"**/{file_name}"))


def _side_file(arm) -> SimpleNamespace:
    """The exchange side-file of the arm: its path, record count, per-record `dt` and per-record end time (s)."""
    from hhemt.coupling_exchange import read_exchange_sidefile
    from hhemt.scenario import coupling_nodes_file_for

    files = sorted(arm.out_dir.glob("**/swmm/*_exchange_replay.bin"))
    assert len(files) == 1, f"arm {arm.name}: expected one exchange side-file, found {files}"
    series = read_exchange_sidefile(files[0], coupling_nodes_file_for(arm.proc.scen_paths.swmm_hydraulics_inp))
    blocks = list(series.iter_blocks())
    return SimpleNamespace(
        path=files[0],
        n=series.n_steps,
        dt=np.concatenate([b.dt_s for b in blocks]),
        end=np.concatenate([b.step_end_s for b in blocks]),
    )


def _raster_files(arm) -> list[Path]:
    """The raw TRITON rasters (H, QX, QY, MH) of the arm, in name order."""
    raw_dir = Path(arm.proc._run.raw_triton_output_dir(model_type="tritonswmm"))
    return sorted(p for p in raw_dir.iterdir() if p.is_file() and p.name.startswith(_RASTER_PREFIXES))


def _differing_files(left: list[Path], right: list[Path]) -> list[str]:
    """Names (or name lists) at which two ordered file lists differ in name or in bytes."""
    if [p.name for p in left] != [p.name for p in right]:
        return [f"file names differ: {sorted({p.name for p in left} ^ {p.name for p in right})[:6]}"]
    return [a.name for a, b in zip(left, right, strict=True) if not filecmp.cmp(a, b, shallow=False)]


def _warn_vacuous(request, what: str) -> None:
    """Record, rather than hide, an identity whose two sides are both zero on this run."""
    warnings.warn(f"{what}: both sides are zero on the synthetic model, so the identity is unverified", stacklevel=2)
    request.node.user_properties.append(("vacuous_identity", what))


def _scale(*arrays: np.ndarray) -> float:
    """The largest magnitude among the arrays, floored at 1, for an absolute tolerance relative to the data."""
    return max([1.0, *(float(np.abs(a).max()) for a in arrays if a.size)])


# ----------------------------------------------------------------------------------------------------------------
# Presence, absence and names
# ----------------------------------------------------------------------------------------------------------------


def test_every_requested_capture_is_written_ingested_and_non_empty(coupled_arm) -> None:
    """Property: with every capture on, each solver file of section 2 exists once under the coupled output
    directory with its sidecar, each of the seven ingested captures has its per-scenario store with a non-empty
    record dimension, the raw-only per-step per-node file is checked and never ingested, and the toolkit's own
    `coupling diagnostics captured` check passes.

    Class: NEW CAPABILITY. Plausible wrong implementations: a key the solver at the pin does not read (the run
    writes no file and the toolkit raises at ingestion), a capture whose store is written empty, the raw-only file
    ingested into a store and a tree group. Reachability: a real coupled run at the pinned solver with every toggle
    on. Both-states anchor: the control is the off arm of the next test, where the same assertions find nothing.
    Killed by: any of those. A second correct implementation ingests the raw-only file lazily on demand; the
    property is that no store or group of it exists.
    """
    from hhemt.analysis_validation import check_coupling_diagnostics
    from hhemt.process_simulation import (
        COUPLING_CAPTURES,
        COUPLING_RAW_ONLY_CAPTURES,
        coupling_capture_store_path,
        expected_coupling_captures,
    )

    arm = coupled_arm("on")
    assert tuple(c.name for c in expected_coupling_captures(arm.analysis.cfg_analysis)) == _CAPTURE_NAMES
    for capture in (*COUPLING_CAPTURES, *COUPLING_RAW_ONLY_CAPTURES):
        found = _raw_files(arm, capture.raw_file)
        assert len(found) == 1, f"{capture.raw_file}: expected one file under {arm.out_dir}, found {found}"
        if capture.sidecar is not None:
            assert (found[0].parent / capture.sidecar).is_file(), f"{capture.sidecar} is absent beside {found[0]}"
    for capture in COUPLING_CAPTURES:
        store = _scenario_store(arm, capture.name)
        assert int(store.sizes[capture.record_dim]) > 0, f"{capture.name}: no record on {capture.record_dim}"
    for capture in COUPLING_RAW_ONLY_CAPTURES:
        assert not coupling_capture_store_path(arm.proc.scen_paths, capture.name, "zarr").exists()
        assert not (arm.tree / "tritonswmm" / "coupling_diagnostics" / capture.name).exists()
    result = check_coupling_diagnostics(arm.analysis)
    assert result.applicable and result.passed, (result.summary, result.details)


def test_with_every_toggle_off_no_capture_file_store_or_group_exists(coupled_arm) -> None:
    """Property: with no capture toggle set, no capture file or sidecar exists under the coupled output directory,
    no `processed/coupling_diagnostics` directory exists, the tree has no `tritonswmm/coupling_diagnostics` group,
    and the toolkit's check reports not applicable.

    Class: NEW CAPABILITY (rule 1 of section 2: off by default, and with every capture off no capture file
    exists). Plausible wrong implementation: a capture file or a store written regardless of its toggle.
    Reachability: the same coupled run with the toggles at their defaults. Both-states anchor: the preceding test
    finds each of these names present on the on arm, so an absence here is the toggles' doing and not a matcher
    that never matches.

    Second correct implementation: the solver writes its capture files under another subdirectory of the output
    directory; the recursive glob still finds every file of the on arm and none of the off arm, so the test passes.
    """
    from hhemt.analysis_validation import check_coupling_diagnostics
    from hhemt.process_simulation import COUPLING_CAPTURES, COUPLING_RAW_ONLY_CAPTURES

    arm = coupled_arm("off")
    names = {c.raw_file for c in (*COUPLING_CAPTURES, *COUPLING_RAW_ONLY_CAPTURES)}
    names |= {c.sidecar for c in (*COUPLING_CAPTURES, *COUPLING_RAW_ONLY_CAPTURES) if c.sidecar}
    present = {n: [str(p) for p in _raw_files(arm, n)] for n in sorted(names) if _raw_files(arm, n)}
    assert present == {}, present
    assert not (Path(arm.proc.scen_paths.sim_folder) / "processed" / "coupling_diagnostics").exists()
    assert not (arm.tree / "tritonswmm" / "coupling_diagnostics").exists()
    assert check_coupling_diagnostics(arm.analysis).applicable is False


def test_ingested_variables_carry_the_units_and_flag_values_the_decoders_define(coupled_arm) -> None:
    """Property: after the write to the per-scenario store, every variable of each capture carries `units` and a
    non-empty `long_name`, with the units section 2 and the decoders name (m3 for volumes, m3 s-1 for fluxes and
    flows, s for times and steps, m and m s-1 for the limiting cell's depth and celerity, 1 for counts), the
    domain-balance terms carry `cell_methods: time: sum`, and the limiter's reason variable carries the three flag
    values.

    Class: NEW CAPABILITY. Plausible wrong implementations: an attribute dropped by the write or overwritten by the
    CF pass for a name the CF table also holds (the node-account names `path3_volume_pos` and `path3_volume_neg`
    sit in both), and a unit left in the engine's ft3. Reachability: the write goes through `_write_output`, which
    applies the CF pass. Both-states anchor: asserts presence of the units and their values.

    Second correct implementation: the units come from a CF table entry instead of the reader's own attribute; the
    stored attribute is the same string, so the test passes.
    """
    arm = coupled_arm("on")
    expected_units = {
        "swmm_step": {
            "swmm_step_dt": "s",
            "swmm_step_trials": "1",
            "swmm_step_nonconverged_nodes": "1",
            "swmm_step_path3_volume_pos": "m3",
            "swmm_step_path3_volume_neg": "m3",
            "swmm_step_link_volume": "m3",
            "nonconverged_node_index": "1",
        },
        "swmm_node_path3": {"path3_volume_pos": "m3", "path3_volume_neg": "m3", "nonconverged_steps": "1"},
        "edge_flux": {"edge_flux": "m3 s-1"},
        "domain_balance": {
            f"{term}_m3": "m3"
            for term in (
                "d_storage",
                "runoff",
                "inflow",
                "extbc",
                "clamp",
                "swmm_exchange",
                "edge_west",
                "edge_east",
                "edge_south",
                "edge_north",
                "residual",
            )
        },
        "wet_cells": {"wet_cells_mean": "1", "wet_cells_at_print": "1"},
        "dt_limiter": {
            "dt_limiter_dt": "s",
            "dt_limiter_depth": "m",
            "dt_limiter_celerity": "m s-1",
            "dt_limiter_rank": "1",
            "dt_limiter_row": "1",
            "dt_limiter_col": "1",
        },
        "swmm_link_flow": {"link_flow": "m3 s-1"},
    }
    for capture, variables in expected_units.items():
        store = _scenario_store(arm, capture)
        for name, units in variables.items():
            assert name in store.data_vars, f"{capture}: no variable {name}"
            attrs = store[name].attrs
            assert attrs.get("units") == units, f"{capture}.{name}: units {attrs.get('units')!r}, expected {units!r}"
            assert str(attrs.get("long_name", "")).strip(), f"{capture}.{name}: no long_name"
    balance = _scenario_store(arm, "domain_balance")
    assert {balance[v].attrs.get("cell_methods") for v in expected_units["domain_balance"]} == {"time: sum"}
    reason = _scenario_store(arm, "dt_limiter")["dt_limiter_reason"]
    assert [int(v) for v in reason.attrs["flag_values"]] == [0, 1, 2]
    assert set(np.unique(reason.values)) <= {0, 1, 2}


# ----------------------------------------------------------------------------------------------------------------
# Identities between captures, and between a capture and the exchange side-file
# ----------------------------------------------------------------------------------------------------------------


def test_the_step_capture_agrees_with_the_side_file_the_per_node_account_and_the_per_step_files(
    coupled_arm, request
) -> None:
    """Property: at the default coupling interval the per-step SWMM record (T1a) has one record per exchange
    side-file record with the same step length and the same end time; its time stamps are the running sum of its
    step lengths; its positive path-3 total is nonnegative and its negative total nonpositive; the per-node
    interval account (T1b) sums to the same positive total, negative total and non-converged count; the per-node
    non-converged counts equal the bin count of the per-step node indices; the per-step per-node file (T1c)
    reproduces the per-step positive total; and the link-flow file (T1d) has one record per step and names the
    sidecar's links.

    Class: NEW CAPABILITY. Plausible wrong implementations: a unit conversion applied to one capture and not
    another; node ordinals read against the wrong name order; a record stamped with the step start in place of the
    step end (the solver's resume positioning reads the end); a decoder that takes `n_items` for a different
    count. Reachability: the solver writes all five files in one coupled step (`capture_step_diag`, `swmm_triton.h`
    at the pin), the side-file record is written in the same step from the same `global_dt`
    (`triton.h`, `log_exchange_step(global_dt, ...)` and `capture_step_diag(simtime + global_dt, global_dt)`), and
    the interval account adds the same per-node values by sign (`node_path3_interval_t::add_step`).
    Both-states anchor: the totals identity is a statement about two files of one run; where both totals are zero on
    this model the test records the identity as unverified instead of passing it silently.

    Second correct implementation: the step file is decoded through one numpy structured dtype in place of a
    record-walking loop; every identity here holds unchanged, so the test passes.
    """
    from hhemt.coupling_diagnostics import (
        MAGIC_SWMM_NODE_PATH3_STEP,
        nonconverged_steps_per_node,
        read_diag_header,
        read_node_names,
        read_swmm_node_path3_step,
    )
    from hhemt.coupling_exchange import FT3_TO_M3_FACTOR

    arm = coupled_arm("on")
    step = _scenario_store(arm, "swmm_step")
    node = _scenario_store(arm, "swmm_node_path3")
    link = _scenario_store(arm, "swmm_link_flow")
    n = int(step.sizes["swmm_step"])
    sim = step["simtime"].values
    dt = step["swmm_step_dt"].values
    assert np.all(dt > 0) and np.all(np.diff(sim) > 0)
    np.testing.assert_allclose(np.cumsum(dt), sim, rtol=1e-9)
    assert np.all(step["swmm_step_trials"].values >= 1)
    pos = step["swmm_step_path3_volume_pos"].values
    neg = step["swmm_step_path3_volume_neg"].values
    assert np.all(pos >= 0) and np.all(neg <= 0)
    nonconverged = step["swmm_step_nonconverged_nodes"].values
    assert np.all(nonconverged >= 0) and int(nonconverged.sum()) == int(step.sizes["nonconverged_entry"])

    side = _side_file(arm)
    assert side.n == n, f"the side-file holds {side.n} records and the step capture {n}"
    np.testing.assert_allclose(dt, side.dt, rtol=1e-6)
    np.testing.assert_allclose(sim, side.end, rtol=1e-6, atol=1e-6)

    for label, per_step, per_node in (
        ("positive path-3 volume", pos.sum(), node["path3_volume_pos"].values.sum()),
        ("negative path-3 volume", neg.sum(), node["path3_volume_neg"].values.sum()),
        ("non-converged node-steps", float(nonconverged.sum()), float(node["nonconverged_steps"].values.sum())),
    ):
        np.testing.assert_allclose(per_node, per_step, rtol=1e-8, atol=1e-12, err_msg=label)
        if per_step == 0.0:
            _warn_vacuous(request, f"{label} total, step capture against node account")
    node_ids = [str(v) for v in node["node_id"].values]
    per_node_counts = nonconverged_steps_per_node(step, len(node_ids))
    np.testing.assert_array_equal(node["nonconverged_steps"].values.sum(axis=0), per_node_counts)

    raw_step = _raw_files(arm, "swmm_node_path3_step.bin")[0]
    sidecar = raw_step.parent / "swmm_node_diag.nodes"
    assert tuple(node_ids) == read_node_names(sidecar)
    read_diag_header(raw_step, MAGIC_SWMM_NODE_PATH3_STEP, len(node_ids))
    per_step_node = read_swmm_node_path3_step(raw_step, sidecar)
    volume = per_step_node["path3_volume_step"].values
    assert volume.shape == (n, len(node_ids))
    np.testing.assert_allclose(
        np.clip(volume, 0.0, None).sum(axis=1) * FT3_TO_M3_FACTOR, pos, rtol=1e-8, atol=1e-12, err_msg="T1c against T1a"
    )

    assert int(link.sizes["swmm_step"]) == n
    assert tuple(str(v) for v in link["link_id"].values) == read_node_names(
        _raw_files(arm, "swmm_link_flow_diag.links")[0]
    )


def test_the_balance_closes_and_its_edge_volumes_are_the_step_fluxes_times_the_step_length(
    coupled_arm, request
) -> None:
    """Property: each interval's residual equals the storage change minus the signed sum of the nine other terms
    (the solver's own definition, `diagnostics.h` `print_instant`), and each edge volume of the balance equals the
    sum, over the steps that end in the interval, of that edge's flux times the step length read from the step
    end times.

    Class: NEW CAPABILITY. Plausible wrong implementations: balance columns read in the wrong order (a swap of the
    residual with the storage change fails the closure), edges in a different order from the flux file's, a flux
    read as a volume. Reachability: both files are written in the same print-instant row. Both-states anchor:
    the second identity ties a binary file to a text file, so a column swap in either breaks it.

    Second correct implementation: the edge volume is summed with each step's start taken from the previous
    record's end; the identity is the same sum, so the test passes.
    """
    arm = coupled_arm("on")
    edge = _scenario_store(arm, "edge_flux")
    balance = _scenario_store(arm, "domain_balance")
    terms = ["runoff_m3", "inflow_m3", "extbc_m3", "clamp_m3", "swmm_exchange_m3"] + [f"edge_{e}_m3" for e in _EDGES]
    others = np.stack([balance[t].values for t in terms])
    scale = _scale(others, balance["d_storage_m3"].values, balance["residual_m3"].values)
    closed = balance["d_storage_m3"].values - others.sum(axis=0)
    np.testing.assert_allclose(balance["residual_m3"].values, closed, rtol=1e-9, atol=1e-9 * scale)
    if scale == 1.0 and not np.any(others):
        _warn_vacuous(request, "balance closure")

    if not np.any(edge["edge_flux"].values):
        # Both sides of the edge-volume identity below are zero, so it holds for any decoder, a column swap included.
        _warn_vacuous(request, "edge volume identity")

    t_end = balance["time_s"].values
    step_t = edge["simtime"].values
    step_dt = np.diff(step_t, prepend=0.0)
    interval = np.searchsorted(t_end, step_t - 1e-6, side="left")
    assert int(interval.max()) < len(t_end), "a flux record ends after the last balance interval"
    for k, name in enumerate(_EDGES):
        volume = np.zeros(len(t_end))
        np.add.at(volume, interval, edge["edge_flux"].values[:, k] * step_dt)
        expected = balance[f"edge_{name}_m3"].values
        np.testing.assert_allclose(
            volume, expected, rtol=1e-7, atol=1e-9 * _scale(volume, expected), err_msg=f"{name} edge volume"
        )


def test_the_surface_captures_share_the_step_clock_and_the_interval_clock(coupled_arm) -> None:
    """Property: the per-step edge-flux and limiter records carry the same end times as the step capture, the
    limiter's step length equals the step capture's, the limiter has no record the step capture lacks, the
    interval captures (balance, wet cells, per-node account) share one interval end time axis, and the wet-cell
    counts lie between zero and the cell count of the synthetic grid on the one rank of the run.

    Class: NEW CAPABILITY. Plausible wrong implementations: a record stamped with another instant than the step
    end (`diag_log.h` rule 3: the resume positioner truncates on that stamp), an interval file written at a
    different print instant from its siblings, a rank grid read against a wrong dimension. Reachability: at the
    defaults every TRITON step is also a SWMM step, so all per-step captures hold the same steps, and the interval
    captures are written in the one print-instant row. Both-states anchor: equality of three independently decoded
    time axes.

    Second correct implementation: the limiter record is written before the edge record within one step; both stay
    stamped with the step end, so every equality here holds.
    """
    from tests.fixtures.synthetic_model.cache import DEFAULT_PARAMS

    arm = coupled_arm("on")
    step = _scenario_store(arm, "swmm_step")
    edge = _scenario_store(arm, "edge_flux")
    limiter = _scenario_store(arm, "dt_limiter")
    balance = _scenario_store(arm, "domain_balance")
    wet = _scenario_store(arm, "wet_cells")
    node = _scenario_store(arm, "swmm_node_path3")
    sim = step["simtime"].values
    assert edge.sizes["step"] == limiter.sizes["step"] == step.sizes["swmm_step"]
    np.testing.assert_allclose(edge["simtime"].values, sim, rtol=1e-12)
    np.testing.assert_allclose(limiter["simtime"].values, sim, rtol=1e-12)
    np.testing.assert_allclose(limiter["dt_limiter_dt"].values, step["swmm_step_dt"].values, rtol=1e-12)
    assert np.all(limiter["dt_limiter_dt"].values > 0)
    assert np.all(limiter["dt_limiter_rank"].values == 0)

    interval_end = balance["time_s"].values
    assert np.all(np.diff(interval_end) > 0)
    np.testing.assert_allclose(wet["time_s"].values, interval_end, rtol=1e-12)
    np.testing.assert_allclose(node["time_s"].values, interval_end, rtol=1e-12)
    assert wet.sizes["Rank"] == 1
    cells = DEFAULT_PARAMS.n_cols * DEFAULT_PARAMS.n_rows
    for name in ("wet_cells_mean", "wet_cells_at_print"):
        values = wet[name].values
        assert np.all(values >= 0) and np.all(values <= cells), (name, float(values.min()), float(values.max()))


# ----------------------------------------------------------------------------------------------------------------
# The coupled node store, the consolidated tree and the timers
# ----------------------------------------------------------------------------------------------------------------


def test_the_node_store_carries_the_per_node_account_with_the_totals_of_the_capture(coupled_arm) -> None:
    """Property: the coupled node store carries `path3_volume_pos`, `path3_volume_neg` (m3) and
    `nonconverged_steps` (1) on `(date_time, node_id)`, with values only at the nodes the solver records (every
    other node is NaN throughout), and each variable's total over the store equals the total of the ingested
    per-node capture.

    Class: NEW CAPABILITY. Plausible wrong implementations: a record added into the wrong report row, a record
    dropped when the report step is coarser than the print interval, a node name matched to the wrong column.
    Reachability: `write_timeseries_outputs` attaches the capture to the node dataset before the node store is
    written. Both-states anchor: the totals are conserved by the attach (every record lands in one row), so an
    equality of totals fails on any dropped or doubled record.

    Second correct implementation: the account is attached in a separate pass after the series are written; the
    stored values and their totals are the same.
    """
    arm = coupled_arm("on")
    store = _node_store(arm)
    node = _scenario_store(arm, "swmm_node_path3")
    recorded = {str(v) for v in node["node_id"].values}
    all_nodes = {str(v) for v in store["node_id"].values}
    for name, units in (("path3_volume_pos", "m3"), ("path3_volume_neg", "m3"), ("nonconverged_steps", "1")):
        assert name in store.data_vars, f"the node store has no {name}"
        assert store[name].dims == ("date_time", "node_id"), store[name].dims
        assert store[name].attrs.get("units") == units
        silent = {str(v) for v in store["node_id"].values[np.isnan(store[name].values).all(axis=0)]}
        assert silent == all_nodes - recorded, f"{name}: NaN nodes differ from the unrecorded nodes"
        np.testing.assert_allclose(
            np.nansum(store[name].values), float(node[name].values.sum()), rtol=1e-8, atol=1e-12, err_msg=name
        )


def test_the_tree_carries_every_capture_group_the_timers_and_the_root_stamp(coupled_arm) -> None:
    """Property: the consolidated tree holds one `tritonswmm/coupling_diagnostics/{name}` group per ingested
    capture with the one event and every variable equal to the per-scenario store, holds the per-node account in
    the coupled node `timeseries` group, carries `DIAG`, `SWMM_SOLVE` and `SWMM_LOG` (units s) in the performance
    `summary` group, and stamps the root with `Conventions: CF-1.13` and the two coupling-schedule values at their
    defaults.

    Class: NEW CAPABILITY. Plausible wrong implementations: a group written for a subset, a group padded or
    reordered by the join, the timers dropped from the performance summary by a name list that predates them, a
    root stamp present only for a non-default schedule. Reachability: `consolidate_to_datatree` with the series
    toggle on and the clearing off. Both-states anchor: equality of values with the store the group was built from.

    Second correct implementation: each capture is consolidated through a region write in place of a whole-group
    rewrite; the groups hold the same values.
    """
    arm = coupled_arm("on")
    for name in _CAPTURE_NAMES:
        group = _group(arm, f"tritonswmm/coupling_diagnostics/{name}")
        assert group["event_iloc"].values.tolist() == [arm.event_iloc], name
        store = _scenario_store(arm, name)
        for variable in store.data_vars:
            np.testing.assert_array_equal(
                group[variable].isel(event_iloc=0).values, store[variable].values, err_msg=f"{name}.{variable}"
            )
    series = _group(arm, "tritonswmm/swmm_node/timeseries")
    for name in ("path3_volume_pos", "path3_volume_neg", "nonconverged_steps"):
        assert name in series.data_vars, f"the consolidated node timeseries group has no {name}"
    performance = _group(arm, "tritonswmm/performance/summary")
    for name in ("DIAG", "SWMM_SOLVE", "SWMM_LOG"):
        assert name in performance.data_vars, f"the performance summary group has no {name}"
        assert performance[name].attrs.get("units") == "s"
    attrs = _root_attrs(arm)
    assert attrs["Conventions"] == "CF-1.13"
    assert (int(attrs["swmm_coupling_interval"]), int(attrs["swmm_coupling_lag"])) == (1, 0)


def test_diag_is_nonzero_with_a_capture_on_zero_with_none_and_the_swmm_timers_nest(coupled_arm) -> None:
    """Property: with the captures on the `DIAG` column of the performance summary is positive and no larger than
    `Simulation`; with none on it is exactly zero; in both arms `SWMM_SOLVE` is positive and
    `SWMM_SOLVE + SWMM_LOG` does not exceed `SWMM_STEP` beyond the 4-significant-digit rounding of the file.

    Class: NEW CAPABILITY (the timers of T4). Plausible wrong implementations: `DIAG` never opened (the on arm
    reads zero), `DIAG` opened by a toggle that is off (the off arm reads positive), the two new `SWMM_*` columns
    read from the wrong header positions. Reachability: the run writes `performance.txt` with the header
    `SWMM_SOLVE, SWMM_LOG, DIAG` (`output.h` at the pin). Both-states anchor: the off arm is the control that
    varies the property under test (capture on or off); the same assertions on the two arms read opposite
    results for `DIAG`.

    Second correct implementation: the timers are read from the per-rank series store in place of the summary
    store; the same orderings hold.
    """
    on = coupled_arm("on")
    off = coupled_arm("off")
    rows = {arm.name: arm.analysis._get_performance_summary_row(arm.event_iloc, "tritonswmm") for arm in (on, off)}
    assert rows["on"]["perf_DIAG"] > 0.0
    assert rows["on"]["perf_DIAG"] <= rows["on"]["perf_Simulation"] * (1 + 5e-3)
    assert rows["off"]["perf_DIAG"] == 0.0
    for name, row in rows.items():
        solve, log, step = row["perf_SWMM_SOLVE"], row["perf_SWMM_LOG"], row["perf_SWMM_STEP"]
        assert solve > 0.0 and log >= 0.0 and step > 0.0, (name, solve, log, step)
        assert solve + log <= step + 5e-3 * max(step, 1.0), (name, solve, log, step)


# ----------------------------------------------------------------------------------------------------------------
# T3: the coupling interval and lag
# ----------------------------------------------------------------------------------------------------------------


def test_the_coupling_interval_arm_steps_swmm_less_often_and_the_tree_says_so(coupled_arm) -> None:
    """Property: with `swmm_coupling_interval` 2 and `swmm_coupling_lag` 1 the exchange side-file holds fewer
    records than the default arm's, the step capture holds one record per side-file record with the same step
    lengths, the captures are ingested and the toolkit's check passes, and the tree's root carries interval 2 and
    lag 1 where the default arm's carries 1 and 0.

    Class: NEW CAPABILITY (T3). Plausible wrong implementations: the keys not emitted (the run steps SWMM every
    step and the counts are equal), the step writer recording every TRITON step in place of every SWMM step
    (`stepped_this_step`, which `capture_step_diag` reads through `step_diag_due()`), the root stamp read from a
    default. Reachability: the cfg lines are written only for a departure from the default
    (`render_triton_cfg`), and the solver steps SWMM once per interval and at print instants. Both-states anchor:
    the default arm of the same module is the control that varies the property (the interval). The side-file
    shrinks by about the interval; only a strict decrease is asserted, because the hydraulics of this arm differ by
    design and the TRITON step count with them.

    Second correct implementation: the lag is a one-step queue and the interval a running mean of the exchange;
    both give the same file-level facts asserted here.
    """
    from hhemt.analysis_validation import check_coupling_diagnostics

    base = coupled_arm("on")
    arm = coupled_arm("t3")
    side = _side_file(arm)
    assert side.n < _side_file(base).n
    step = _scenario_store(arm, "swmm_step")
    assert int(step.sizes["swmm_step"]) == side.n
    np.testing.assert_allclose(step["swmm_step_dt"].values, side.dt, rtol=1e-6)
    np.testing.assert_allclose(step["simtime"].values, side.end, rtol=1e-6, atol=1e-6)
    result = check_coupling_diagnostics(arm.analysis)
    assert result.applicable and result.passed, (result.summary, result.details)
    attrs = _root_attrs(arm)
    assert (int(attrs["swmm_coupling_interval"]), int(attrs["swmm_coupling_lag"])) == (2, 1)
    base_attrs = _root_attrs(base)
    assert (int(base_attrs["swmm_coupling_interval"]), int(base_attrs["swmm_coupling_lag"])) == (1, 0)


# ----------------------------------------------------------------------------------------------------------------
# H7: the independent SWMM report step
# ----------------------------------------------------------------------------------------------------------------


def test_a_different_swmm_report_step_leaves_the_rasters_and_the_side_file_byte_identical(coupled_arm) -> None:
    """Property: the synthetic coupled model run with `swmm_report_step_s` 5 against the default (the 10 s TRITON
    reporting step) writes the same raster files (H, QX, QY, MH) with identical bytes and an identical exchange
    side-file, while the SWMM binary output of the 5 s run is larger.

    Class: NEW CAPABILITY (row H7 of section 3: neutral by argument, test pending). Plausible wrong
    implementations: a report step that reaches the TRITON coupling (`ReportStep` and `ReportTime` are read in
    SWMM's routing only inside a branch the coupled run overwrites, `swmm5.c:546`), a reporting interval that moves
    the coupling's step. Reachability: a real coupled run per arm. Both-states anchor: the SWMM binary output is
    the control that varies the property under test: it is larger at the finer report step, so the two arms
    differ in exactly the quantity the field sets and are identical in the quantities the row claims are
    unaffected. Venue: Rivanna scratch only. A raster that differs names the first differing file.

    Second correct implementation: the SWMM report step is set through the model template in place of the config
    field; the rasters and the side-file are the same bytes, so the test passes.
    """
    off = coupled_arm("off")
    report = coupled_arm("report_step")
    rasters_off, rasters_report = _raster_files(off), _raster_files(report)
    assert {p.name.split("_")[0] + "_" for p in rasters_off} == set(_RASTER_PREFIXES)
    assert _differing_files(rasters_off, rasters_report) == []
    assert filecmp.cmp(_side_file(off).path, _side_file(report).path, shallow=False)

    def swmm_out_bytes(arm) -> int:
        """Total size of the arm's SWMM binary outputs."""
        return sum(p.stat().st_size for p in arm.out_dir.glob("**/swmm/*.out"))

    assert swmm_out_bytes(report) > swmm_out_bytes(off) > 0


def test_the_swmm_series_land_on_the_report_step_axis_while_triton_keeps_its_own(coupled_arm) -> None:
    """Property: with `swmm_report_step_s` 5 the consolidated coupled SWMM node and link series sit on an
    `elapsed_s` axis of 5 s steps and the gridded TRITON series on 10 s steps; with the field unset all three sit
    on 10 s steps.

    Class: NEW CAPABILITY (H7 ingestion). Plausible wrong implementations: the SWMM series reindexed onto the
    TRITON axis (the finer values dropped), the TRITON axis moved with the field. Reachability: the consolidation of
    a real run at each report step. Both-states anchor: the unset arm is the control for the same groups.

    Second correct implementation: the SWMM series are reindexed onto their own axis at consolidation in place of
    at write; the group axis is the same.
    """
    off = coupled_arm("off")
    report = coupled_arm("report_step")

    def single_step(arm, group: str) -> float:
        """The one spacing of the ``elapsed_s`` axis of a tree group; fails when the axis has more than one."""
        axis = np.asarray(_group(arm, group)["elapsed_s"].values, dtype="float64")
        steps = np.unique(np.round(np.diff(axis), 9))
        assert len(steps) == 1, (arm.name, group, steps[:5])
        return float(steps[0])

    for group in ("tritonswmm/swmm_node/timeseries", "tritonswmm/swmm_link/timeseries"):
        assert single_step(report, group) == 5.0, group
        assert single_step(off, group) == 10.0, group
    assert single_step(report, "tritonswmm/triton/timeseries") == 10.0
    assert single_step(off, "tritonswmm/triton/timeseries") == 10.0


# ----------------------------------------------------------------------------------------------------------------
# Beyond the criterion: the captures leave the hydraulics alone at the toolkit level
# ----------------------------------------------------------------------------------------------------------------


def test_the_capture_toggles_leave_the_rasters_and_the_side_file_byte_identical(coupled_arm) -> None:
    """Property: the coupled run with every capture on writes the same rasters and the same exchange side-file,
    byte for byte, as the run with every capture off.

    Class: NEW CAPABILITY. This EXCEEDS the criterion of chunk 4a.1 (whose byte identity is row H7) and is
    recorded as such: the solver's neutrality suite (cases N1 to N3) is the instrument that discharges row-level
    neutrality, and this is the same claim read at the toolkit's own build flags, at no extra run because both arms
    exist for the tests above. A raster that differs reclassifies the failing capture as possibly
    hydraulics-affecting (section 3). Both-states anchor: the same file lists are non-empty and the on arm holds
    capture files the off arm lacks, so the comparison is between runs that differ in the toggles.

    Reachability: both arms run the same scenario and differ only in the capture toggles of the config, so any byte
    difference is the toggles' doing. Killed by: a capture that writes into solver state, which changes a raster
    or the side-file. Second correct implementation: the captures record through a read-only copy of the arrays;
    the raster and side-file bytes are unchanged, so the test passes.
    """
    off = coupled_arm("off")
    on = coupled_arm("on")
    rasters = _raster_files(off)
    assert rasters, "the off arm wrote no raster"
    assert _differing_files(rasters, _raster_files(on)) == []
    assert filecmp.cmp(_side_file(off).path, _side_file(on).path, shallow=False)
    assert _raw_files(on, "swmm_step_diag.bin") and not _raw_files(off, "swmm_step_diag.bin")
