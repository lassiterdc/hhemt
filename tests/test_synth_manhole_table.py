"""Compile-tier test of the per-node manhole table on the synthetic coupled model, end to end.

SIMULATION-BEARING: the fixtures compile and run TRITON-SWMM. Run it on Rivanna scratch only, serialized against
the worktree's build cache; it is never run on a laptop (a laptop collects it without compiling or running a
solver). The synthetic pin (`TRITON_PIN`, `deefa6e6`) reads the cfg key `swmm_manhole_table`, so no pin override
is applied and preparation's build guard passes.

The laptop tests of the same feature (`tests/test_manhole_table.py`, `tests/test_manhole_table_ingest.py`) judge
the rule, the written file and the readers on stand-ins. This file answers what they cannot: that the solver
reads the table, that a table of the constants changes nothing it writes, and that a table of other values
does. Four coupled scenarios of the synthetic model are run once each and shared by the tests:

  none       no table. The reference: its rasters and exchange side-file are what the solver writes today.
  constants  a table whose every row equals the two constants. Byte-identical to `none` in the rasters and the
             side-file (the solver parses the same doubles the constant loop assigns).
  one_node   a table whose every row equals the constants except one node, which carries a diameter AND a loss
             coefficient that differ from each other and from the constants. The coupled node store carries
             both values by variable, and the rasters differ from `none`.
  fill       a table that omits one coupling node, with `manhole_table_fill_with_constants` on. The run goes
             through and the omitted node carries the constants.

A fifth, run-entry-only case (no solver) shows that the same omission with the toggle off is refused by
`analysis.run()` before any job, naming the toggle.

Every fixture table gives its changed row two values that differ from each other and from the two constants
(the diameter halved, the loss coefficient 3.5 times its constant: 0.6 and 0.35 against 1.2 and 0.1 when the
constants are the defaults), and the node store is read by variable, so a solver or reader that swaps the two
columns fails the assignment checks.

Every chunk-created name is looked up inside the fixtures and test bodies, and the first statement of each
fixture and test checks the chunk's configuration fields. `manhole_root` takes `tritonswmm_cpu_compiled` as a
parameter, so pytest sets the compile fixture up BEFORE that check runs: on a tree without the feature each test
still collects, but it fails with an AssertionError from this file after the compile, not before it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

# `shared_runs_root` is declared HERE as well as being derivable: `manhole_root` requests `tritonswmm_cpu_compiled`
# as a fixture parameter, so the collection-time closure sees the compile, and the marker states outright that this
# file builds under the shared slug root. The analysis trees themselves go under the module's own temporary
# directory through `runs_root_override`; the marker only lets the compiled fixture build where it always does, the
# shared slug root, serialised by its own per-build lock. Solver-bearing: runs on the cluster; the laptop capture
# selects `not slow`.
pytestmark = [pytest.mark.compile_tier, pytest.mark.shared_runs_root, pytest.mark.slow]

_HEADER = "node_id,manhole_diameter_m,manhole_loss_coefficient"
#: The changed row's values, as multiples of the two constants. They give 0.6 and 0.35 against the defaults 1.2 and
#: 0.1; whatever the constants are, the row's two values differ from each other and from both constants (checked in
#: `_changed_row`).
_DIAMETER_FACTOR = 0.5
_LOSS_FACTOR = 3.5
#: The name prefixes of the raster files the solver writes flat into `{out_tritonswmm}/bin/` as `{PREFIX}_{n}_00.out`:
#: depth, the two fluxes, and the running maximum depth. Measured on Rivanna (the no-table run listed 1080 files per
#: prefix). `GR_*.out` files sit beside them in that directory; they are left out because they are the solver's
#: ghost-ring halo state, not a raster of the domain: `output<T>::write_output_ghost_ring` writes them through
#: `get_mat_path("GR", root_dir, BIN_DIR, print_id, ".out")` (`git show deefa6e6:src/output.h`, near line 803).
_RASTER_PREFIXES = ("H_", "QX_", "QY_", "MH_")
_MIN_COUPLING_NODES = 3


def _require_chunk() -> None:
    """Fail with an AssertionError when the feature's fields do not exist.

    Runs inside `manhole_root`, after pytest has set up `tritonswmm_cpu_compiled`, so a tree without the fields fails
    here after the compile fixture, not before it.
    """
    from hhemt.config.analysis import analysis_config

    for field in ("manhole_table", "manhole_table_fill_with_constants"):
        assert field in analysis_config.model_fields, f"analysis_config has no {field} field"


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
    nodes: list[str]
    rasters: dict[str, str]  # path relative to the coupled output directory -> sha256 of the bytes
    sidefile_sha: str
    cfg_text: str
    store: xr.Dataset  # loaded: it outlives the zarr handle


def _write_table(path: Path, rows: dict[str, tuple[float, float]]) -> Path:
    """A manhole table CSV with the documented header and one row per entry, values written with ``repr``."""
    path.write_text(_HEADER + "\n" + "".join(f"{n},{diameter!r},{loss!r}\n" for n, (diameter, loss) in rows.items()))
    return path


def _build_case(request, root: Path, name: str, *, table: Path | None, fill: bool = False):
    """Build one coupled synthetic case with the flux on, optionally with a manhole table."""
    from tests.fixtures.test_case_builder import retrieve_synth_TRITON_SWMM_test_case

    extra: dict = {"toggle_coupling_exchange_flow": True}
    if table is not None:
        extra["manhole_table"] = str(table)  # yaml.safe_dump cannot write a Path
        extra["manhole_table_fill_with_constants"] = fill
    return retrieve_synth_TRITON_SWMM_test_case(
        analysis_name=name,
        n_events=1,
        toggle_tritonswmm_model=True,
        toggle_triton_model=False,
        toggle_swmm_model=False,
        start_from_scratch=True,
        runs_root_override=root,
        additional_analysis_configs=extra,
    )


def _run_arm(request, root: Path, name: str, *, table: Path | None, fill: bool = False) -> _Arm:
    """Prepare, run and process one scenario with its raw outputs kept; return what it left."""
    case = _build_case(request, root, name, table=table, fill=fill)
    analysis = case.analysis
    analysis.run_prepare_scenarios_serially(
        overwrite_scenario_if_already_set_up=True, rerun_swmm_hydro_if_outputs_exist=True
    )
    analysis.run_sims_in_sequence(pickup_where_leftoff=False)
    proc = analysis._retrieve_sim_run_processing_object(analysis.df_sims.index[0])
    proc.write_timeseries_outputs(which="both", model_type="tritonswmm", override_clear_raw="none")

    from hhemt.scenario import coupling_nodes_file_for

    nodes = [
        line.strip()
        for line in coupling_nodes_file_for(proc.scen_paths.swmm_hydraulics_inp).read_text().splitlines()
        if line.strip()
    ]
    out = Path(proc.scen_paths.out_tritonswmm)
    raster_files = sorted(f for f in (out / "bin").glob("*") if f.is_file() and f.name.startswith(_RASTER_PREFIXES))
    rasters = {str(f.relative_to(out)): _sha256(f) for f in raster_files}
    per_prefix = {p: sum(f.name.startswith(p) for f in raster_files) for p in _RASTER_PREFIXES}
    assert len(set(per_prefix.values())) == 1 and min(per_prefix.values()) > 0, (
        f"{name}: the raster prefixes do not each hold the same non-zero number of files in {out / 'bin'}: {per_prefix}"
    )
    sidefiles = sorted(out.glob("**/swmm/*_exchange_replay.bin"))
    assert len(sidefiles) == 1, f"{name}: expected the one raw exchange side-file to be kept, found {sidefiles}"
    store_path = Path(proc.scen_paths.output_tritonswmm_node_time_series)
    with xr.open_zarr(store_path) if store_path.suffix == ".zarr" else xr.open_dataset(store_path) as opened:
        store = opened.load()
    return _Arm(
        name=name,
        analysis=analysis,
        nodes=nodes,
        rasters=rasters,
        sidefile_sha=_sha256(sidefiles[0]),
        cfg_text=Path(proc.scen_paths.triton_swmm_cfg).read_text(),
        store=store,
    )


def _constants(arm: _Arm) -> tuple[float, float]:
    cfg = arm.analysis.cfg_analysis
    return float(cfg.manhole_diameter), float(cfg.manhole_loss_coefficient)


def _changed_row(arm: _Arm) -> tuple[float, float]:
    """The one changed row's (diameter, loss): each differs from the other and from both constants."""
    diameter_constant, loss_constant = _constants(arm)
    diameter, loss = diameter_constant * _DIAMETER_FACTOR, loss_constant * _LOSS_FACTOR
    assert len({diameter, loss, diameter_constant, loss_constant}) == 4, (
        f"the fixture row ({diameter}, {loss}) is not four distinct numbers with the constants "
        f"({diameter_constant}, {loss_constant}); a swapped column would pass"
    )
    return diameter, loss


def _target_node(arm: _Arm) -> str:
    """The coupling node with the largest absolute exchange in the no-table run, so a change there moves the water.

    Must-be-nonzero control: a node whose exchange is zero throughout would let a table that the solver ignores
    pass the rasters-differ check, so the exchange of the chosen node is asserted above zero.
    """
    flow = np.abs(arm.store["coupling_exchange_flow"].sel(node_id=arm.nodes)).max(dim="date_time")
    assert float(flow.max()) > 0.0, "the no-table run exchanged no water at any coupling node"
    return str(flow.idxmax(dim="node_id").item())


def _by_node(arm: _Arm, variable: str) -> dict[str, float]:
    series = arm.store[variable]
    return {n: float(series.sel(node_id=n)) for n in arm.nodes}


@pytest.fixture(scope="module")
def manhole_root(tritonswmm_cpu_compiled, tmp_path_factory) -> Path:
    """One temporary directory holding every arm's analysis tree and table.

    Requests `tritonswmm_cpu_compiled` as a parameter so the collection-time fixture closure sees the compile: every
    arm and the run-entry test depend on this fixture, and a compile reached only through `request.getfixturevalue`
    is invisible to the closure and trips the compile guard on the first arm.
    """
    _require_chunk()
    return tmp_path_factory.mktemp("manhole_table")


@pytest.fixture(scope="module")
def arm_none(request, manhole_root) -> _Arm:
    """The reference run: no table."""
    _require_chunk()
    arm = _run_arm(request, manhole_root, "manhole_none", table=None)
    assert len(arm.nodes) >= _MIN_COUPLING_NODES, f"the synthetic model coupled only {arm.nodes}"
    return arm


@pytest.fixture(scope="module")
def arm_constants(request, manhole_root, arm_none) -> _Arm:
    """A table whose every row equals the two constants."""
    _require_chunk()
    diameter, loss = _constants(arm_none)
    table = _write_table(manhole_root / "constants.csv", {n: (diameter, loss) for n in arm_none.nodes})
    return _run_arm(request, manhole_root, "manhole_constants", table=table)


@pytest.fixture(scope="module")
def arm_one_node(request, manhole_root, arm_none) -> tuple[_Arm, str, tuple[float, float]]:
    """A table whose rows equal the constants except the target node's, which carries another pair of values."""
    _require_chunk()
    diameter, loss = _constants(arm_none)
    target, changed = _target_node(arm_none), _changed_row(arm_none)
    rows = {n: (diameter, loss) for n in arm_none.nodes}
    rows[target] = changed
    table = _write_table(manhole_root / "one_node.csv", rows)
    return _run_arm(request, manhole_root, "manhole_one_node", table=table), target, changed


def _omitting_table(manhole_root: Path, arm_none: _Arm) -> tuple[Path, str, dict[str, tuple[float, float]]]:
    """A table that gives every coupling node but the last a row, each row a different pair of values.

    Each row's two values differ from each other and from both constants (asserted below), and the rows differ
    from one another, so a reader that swaps the columns or shifts the rows fails.
    """
    diameter, loss = _constants(arm_none)
    omitted = arm_none.nodes[-1]
    rows = {
        n: (diameter * (0.5 + 0.07 * i), loss * (2.0 + 0.5 * i)) for i, n in enumerate(arm_none.nodes) if n != omitted
    }
    for node, (row_diameter, row_loss) in rows.items():
        assert len({row_diameter, row_loss, diameter, loss}) == 4, (
            f"the row of {node} ({row_diameter}, {row_loss}) is not four distinct numbers with the constants "
            f"({diameter}, {loss}); a swapped column or a fallback to a constant would pass"
        )
    return _write_table(manhole_root / "omitting.csv", rows), omitted, rows


@pytest.fixture(scope="module")
def arm_fill(request, manhole_root, arm_none) -> tuple[_Arm, str, dict[str, tuple[float, float]]]:
    """A table that omits a coupling node, with the fill toggle on."""
    _require_chunk()
    table, omitted, rows = _omitting_table(manhole_root, arm_none)
    # The run-entry check accepts the omission with the toggle on, before the scenario is prepared.
    case = _build_case(request, manhole_root, "manhole_fill_entry", table=table, fill=True)
    case.analysis._refuse_manhole_table_mismatch()
    return _run_arm(request, manhole_root, "manhole_fill", table=table, fill=True), omitted, rows


def test_a_table_of_the_constants_changes_no_raster_and_no_exchange_byte(arm_none, arm_constants) -> None:
    """Property: with a table whose every row equals the two constants, the rasters and the exchange side-file
    are byte-identical to the no-table run, and the cfg gains exactly the one table line.

    Class: NEUTRALITY at the constants (the table of constants must not move the solver; this is the end-to-end
    instrument for that claim on the synthetic model). Plausible wrong implementations: a reader that applies the
    diameter column to the loss (the constants differ from each other, so a swap changes the exchange), an unconverted
    unit, a table applied after the rank split by global index, or a build that parses the table to a different
    double than the constant loop assigns.
    Reachability and control: the raster set is non-empty and the side-file exists in both runs (a vacuous
    equality of two empty sets is refused below), and `test_a_table_with_one_node_changed_...` varies the
    property under suspicion, the table's values, and must see the same comparison differ.
    Killed by: any change to a raster or to the exchange side-file at the constants. A second correct
    implementation writes the constants in another text form that parses to the same doubles; it passes.
    """
    _require_chunk()
    assert arm_none.rasters, (
        "the no-table run left no raster file in bin/ named H_, QX_, QY_ or MH_; the comparison would be vacuous"
    )
    assert arm_constants.rasters.keys() == arm_none.rasters.keys()
    assert arm_constants.rasters == arm_none.rasters
    assert arm_constants.sidefile_sha == arm_none.sidefile_sha
    key = "swmm_manhole_table"
    assert key not in arm_none.cfg_text
    assert arm_constants.cfg_text.count(key) == 1, "the table's cfg line is missing or repeated"


def test_a_table_with_one_node_changed_runs_carries_both_values_and_changes_the_rasters(arm_none, arm_one_node) -> None:
    """Property: a table that gives one node a diameter and a loss coefficient different from each other and from
    the constants runs; the coupled node store carries both values at that node by variable and the constants at
    every other coupling node; and the rasters differ from the no-table run.

    Class: HYDRAULICS-CHANGING BY DESIGN: the manhole area and the loss multiply the exchange
    flow, so the surface-to-sewer exchange at the node, and the depths around it, change.
    Plausible wrong implementations: a solver build that ignores the key (every node at the constants, rasters
    equal to the no-table run); the diameter written into the loss column or the reverse (the node store's
    variable at the target would hold the other value); the value applied to the wrong node (the store's
    non-target nodes would not read the constants); a reader that fills NaN at coupling nodes.
    Reachability and control: the target is the node with the largest exchange in the no-table run (asserted
    above zero there), so a table the solver honours moves water; the no-table run has no such variables, which
    shows the variables come from the table.
    Killed by: those. A second correct implementation chooses another node; the assertions are read off the
    chosen node's name.
    """
    _require_chunk()
    arm, target, (diameter, loss) = arm_one_node
    diameter_constant, loss_constant = _constants(arm_none)
    assert "manhole_diameter" not in arm_none.store and "manhole_loss_coefficient" not in arm_none.store
    diameters, losses = _by_node(arm, "manhole_diameter"), _by_node(arm, "manhole_loss_coefficient")
    assert diameters[target] == pytest.approx(diameter, rel=1e-12), "diameter at the target is not the table's"
    assert losses[target] == pytest.approx(loss, rel=1e-12), "loss coefficient at the target is not the table's"
    for node in set(arm.nodes) - {target}:
        assert diameters[node] == pytest.approx(diameter_constant, rel=1e-12), node
        assert losses[node] == pytest.approx(loss_constant, rel=1e-12), node
    uncoupled = arm.store["is_coupling_node"].values == 0
    assert uncoupled.any(), "the synthetic node store has no uncoupled node, so the NaN check is vacuous"
    assert np.isnan(arm.store["manhole_diameter"].values[uncoupled]).all()
    assert np.isnan(arm.store["manhole_loss_coefficient"].values[uncoupled]).all()
    assert arm.rasters.keys() == arm_none.rasters.keys()
    changed = sorted(k for k in arm.rasters if arm.rasters[k] != arm_none.rasters[k])
    assert changed, (
        f"every raster of the run with {target} at diameter {diameter} and loss {loss} equals the no-table run; "
        "the solver did not apply the table, or the node moves no water"
    )


def test_a_coupling_node_without_a_row_refuses_at_run_entry_naming_the_toggle(request, manhole_root, arm_none) -> None:
    """Property: with `manhole_table_fill_with_constants` off, a table that omits a coupling node is refused by
    `analysis.run()` at its entry, before any job, as a `ConfigurationError` on the field `manhole_table` whose
    message names the toggle and the omitted node.

    Class: REFUSAL AT RUN ENTRY. Plausible wrong implementations: a run that proceeds and
    runs the omitted node at the constants without a word; a refusal that does not name the toggle the user must
    flip; a check only at preparation (the refusal would then arrive after the queue).
    Reachability and control: the same table with the toggle on is accepted by the run-entry check in the
    fill fixture, which then runs (`test_a_node_without_a_row_takes_the_constants_when_the_toggle_is_on`); the
    omitted node is a real coupling node of the model (read from the no-table run's list).
    Killed by: those. A second correct implementation refuses from `submit_workflow` as well, or words the message
    differently; it passes as long as the field, the toggle's name and the omitted node are in the error.
    `dry_run=True` reaches the same refusal and ends at planning, so this test starts no solver and, if the refusal
    is ever lost, fails fast instead of launching a workflow.
    """
    _require_chunk()
    from hhemt.exceptions import ConfigurationError

    table, omitted, _rows = _omitting_table(manhole_root, arm_none)
    case = _build_case(request, manhole_root, "manhole_missing_row", table=table, fill=False)
    with pytest.raises(ConfigurationError) as raised:
        case.analysis.run(dry_run=True)
    assert raised.value.field == "manhole_table"
    message = str(raised.value)
    assert "manhole_table_fill_with_constants" in message
    assert omitted in message
    assert omitted in arm_none.nodes


def test_a_node_without_a_row_takes_the_constants_when_the_toggle_is_on(arm_none, arm_fill) -> None:
    """Property: with the toggle on, the omitted coupling node runs and carries both constants in the node store,
    while every node with a row carries its own two values.

    Class: THE FILL. Plausible wrong implementations: the omitted node left at zero or NaN in
    the store; the toggle accepted at run entry and ignored at preparation; rows shifted by one position when a
    node is omitted.
    Reachability and control: each row's two values differ from each other and from the constants (see
    `_omitting_table`), and the omitted node is the last of the list, so a shift is visible on every earlier row.
    Killed by: those. A second correct implementation omits another node; the assertions read the name.
    """
    _require_chunk()
    arm, omitted, rows = arm_fill
    diameter_constant, loss_constant = _constants(arm_none)
    diameters, losses = _by_node(arm, "manhole_diameter"), _by_node(arm, "manhole_loss_coefficient")
    assert diameters[omitted] == pytest.approx(diameter_constant, rel=1e-12)
    assert losses[omitted] == pytest.approx(loss_constant, rel=1e-12)
    for node, (diameter, loss) in rows.items():
        assert diameters[node] == pytest.approx(diameter, rel=1e-12), node
        assert losses[node] == pytest.approx(loss, rel=1e-12), node
    assert arm.cfg_text.count("swmm_manhole_table") == 1
