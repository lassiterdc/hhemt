"""Laptop-safe review of `tests/test_synth_coupling_diagnostics.py`: can its identity tests fail, and why.

The reviewed file runs a compiled solver and is run on Rivanna only, so a red there costs a queue round trip and a
green there says nothing about whether its assertions would have caught a wrong decoder. This file answers the second
question without a compile or a solver run, in two arms:

  Static. Every name the reviewed file imports lazily resolves, every arm it requests exists, and every config key
  the arms set is a field of `analysis_config` whose value the field accepts. These are the faults that otherwise
  surface one arm-build into a Rivanna job.

  Mutation. The reviewed file's three cross-file identity tests (the step capture against the side-file and the
  per-node account, the domain balance against the edge-flux file, the shared step and interval clocks) are called
  on a small, hand-built capture set written in the solver's byte formats and decoded by the real decoders, with the
  module's own readers of an arm's stores replaced by that set. The unmodified set must pass each test and each named
  wrong implementation, applied to the decoded stores or to a raw file, must fail the test that claims to kill it.
  A final test runs the balance identity on a closed-domain set whose edge fluxes are all zero and requires that the
  test does not pass in silence there.

The hand-built set is the instrument, so it carries its own controls: the passing run on the unmodified set is the
control for every mutant (a mutant that failed because the set was broken would fail the control too), and the
mutant list varies the property each test claims to enforce, not a token of it.

Not established: that the real synthetic run is non-degenerate in the quantities the mutants exploit. The last test
shows what the reviewed file does when it is not.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import struct
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import pytest
from pydantic import TypeAdapter

import tests.test_synth_coupling_diagnostics as cand

_CAND_SOURCE = Path(cand.__file__).read_text()
_CAND_TREE = ast.parse(_CAND_SOURCE)
_FT3_TO_M3 = importlib.import_module("hhemt.coupling_exchange").FT3_TO_M3_FACTOR

# ----------------------------------------------------------------------------------------------------------------
# The hand-built capture set
# ----------------------------------------------------------------------------------------------------------------

_NODES = ("J1", "J2", "J3")
_LINKS = ("C1", "C2")
_DT = 0.5
_N_STEPS = 6
_PER_INTERVAL = 3
_T_END = np.array([1.5, 3.0])
# Signed path-3 volume per step and node, in ft3 (the engine's unit); every sign and a zero appear.
_V = np.array(
    [
        [4.0, -1.0, 0.0],
        [0.0, 2.0, -3.0],
        [1.5, 0.0, 5.0],
        [-2.0, 6.0, 0.0],
        [0.0, -1.0, 2.5],
        [3.0, 0.0, -4.0],
    ]
)
# Non-converged node ordinals per step, chosen so the per-node counts [2, 1, 3] are not palindromic.
_NONCONV = ([0], [], [0, 2], [2], [2], [1])
_FLUX = np.arange(1, _N_STEPS + 1)[:, None] * np.array([0.1, 0.2, 0.3, 0.4])  # west, east, south, north per step
_OTHER_TERMS = {
    "runoff_m3": [5.0, 6.0],
    "inflow_m3": [1.0, 0.0],
    "extbc_m3": [0.5, 0.25],
    "clamp_m3": [0.1, 0.2],
    "swmm_exchange_m3": [-2.0, -3.0],
}
_D_STORAGE = [9.0, 7.0]


def _header(magic: int, n_items: int) -> bytes:
    """The 16-byte header every capture binary starts with: magic, version 1, value width 8, item count."""
    return np.array([magic, 1, 8, n_items], dtype="<i4").tobytes()


def _write_run(root: Path, *, t1c_row_shift: int = 0, zero_edges: bool = False) -> SimpleNamespace:
    """Write every capture file of one tiny run into ``root`` and decode each with the real reader.

    Returns the decoded stores by capture name, the raw files by file name and the side-file stand-in.
    """
    import hhemt.coupling_diagnostics as cd
    from hhemt.coupling_exchange import FT3_TO_M3_FACTOR

    root.mkdir(parents=True, exist_ok=True)
    sim = _DT * np.arange(1, _N_STEPS + 1)
    pos_ft = np.clip(_V, 0.0, None).sum(axis=1)
    neg_ft = np.clip(_V, None, 0.0).sum(axis=1)
    (root / "swmm_node_diag.nodes").write_text("\n".join(_NODES) + "\n")
    (root / "swmm_link_flow_diag.links").write_text("\n".join(_LINKS) + "\n")

    step = bytearray(_header(cd.MAGIC_SWMM_STEP_DIAG, 0))
    for k in range(_N_STEPS):
        step += struct.pack("<ddiiddd", sim[k], _DT, 2, len(_NONCONV[k]), pos_ft[k], neg_ft[k], 10.0 + k)
        step += np.asarray(_NONCONV[k], dtype="<i4").tobytes()
    (root / "swmm_step_diag.bin").write_bytes(bytes(step))

    n = len(_NODES)
    acct = np.zeros(2, dtype=[("t", "<f8"), ("pos", "<f8", (n,)), ("neg", "<f8", (n,)), ("nn", "<i4", (n,))])
    for i in range(2):
        rows = slice(_PER_INTERVAL * i, _PER_INTERVAL * (i + 1))
        acct["t"][i] = _T_END[i]
        acct["pos"][i] = np.clip(_V[rows], 0.0, None).sum(axis=0)
        acct["neg"][i] = np.clip(_V[rows], None, 0.0).sum(axis=0)
        acct["nn"][i] = np.bincount([o for k in range(rows.start, rows.stop) for o in _NONCONV[k]], minlength=n)
    (root / "swmm_node_path3.bin").write_bytes(_header(cd.MAGIC_SWMM_NODE_PATH3, n) + acct.tobytes())

    per_step = np.zeros(_N_STEPS, dtype=[("t", "<f8"), ("dt", "<f8"), ("v", "<f8", (n,))])
    per_step["t"], per_step["dt"] = sim, _DT
    per_step["v"] = np.roll(_V, t1c_row_shift, axis=0)
    (root / "swmm_node_path3_step.bin").write_bytes(_header(cd.MAGIC_SWMM_NODE_PATH3_STEP, n) + per_step.tobytes())

    link = np.zeros(_N_STEPS, dtype=[("t", "<f8"), ("flow", "<f8", (len(_LINKS),))])
    link["t"], link["flow"] = sim, np.arange(_N_STEPS * len(_LINKS), dtype=float).reshape(_N_STEPS, -1)
    (root / "swmm_link_flow_diag.bin").write_bytes(_header(cd.MAGIC_SWMM_LINK_FLOW_DIAG, len(_LINKS)) + link.tobytes())

    flux = np.zeros_like(_FLUX) if zero_edges else _FLUX
    edge = np.zeros(_N_STEPS, dtype=cd._EDGE_FLUX_DTYPE)
    edge["simtime"] = sim
    for column, name in enumerate(("west", "east", "south", "north")):
        edge[name] = flux[:, column]
    (root / "edge_flux.bin").write_bytes(_header(cd.MAGIC_EDGE_FLUX, 4) + edge.tobytes())

    edge_m3 = [(flux[_PER_INTERVAL * i : _PER_INTERVAL * (i + 1)] * _DT).sum(axis=0) for i in range(2)]
    lines = ["%" + ", ".join(cd._DOMAIN_BALANCE_COLUMNS)]
    for i in range(2):
        others = [_OTHER_TERMS[c][i] for c in _OTHER_TERMS]
        residual = _D_STORAGE[i] - (sum(others) + edge_m3[i].sum())
        row = [_T_END[i], _D_STORAGE[i], *others, *edge_m3[i], residual]
        lines.append(", ".join(f"{v:.12e}" for v in row))
    (root / "domain_balance.csv").write_text("\n".join(lines) + "\n")

    lim = np.zeros(_N_STEPS, dtype=cd._DT_LIMITER_DTYPE)
    lim["simtime"], lim["dt"], lim["grow"], lim["gcol"], lim["h"], lim["celerity"] = sim, _DT, 3, 4, 0.1, 3.0
    (root / "dt_limiter.bin").write_bytes(_header(cd.MAGIC_DT_LIMITER, 0) + lim.tobytes())
    (root / "wet_cells.csv").write_text(
        "%print_id, simtime, rank, wet_mean, wet_at_print\n1, 1.5, 0, 10.5, 11\n2, 3.0, 0, 12, 12\n"
    )

    nodes_file, links_file = root / "swmm_node_diag.nodes", root / "swmm_link_flow_diag.links"
    stores = {
        "swmm_step": cd.read_swmm_step_diag(root / "swmm_step_diag.bin"),
        "swmm_node_path3": cd.read_swmm_node_path3(root / "swmm_node_path3.bin", nodes_file),
        "edge_flux": cd.read_edge_flux(root / "edge_flux.bin"),
        "domain_balance": cd.read_domain_balance(root / "domain_balance.csv"),
        "wet_cells": cd.read_wet_cells(root / "wet_cells.csv"),
        "dt_limiter": cd.read_dt_limiter(root / "dt_limiter.bin"),
        "swmm_link_flow": cd.read_swmm_link_flow_diag(root / "swmm_link_flow_diag.bin", links_file),
    }
    assert FT3_TO_M3_FACTOR > 0
    side = SimpleNamespace(path=root / "side.bin", n=_N_STEPS, dt=np.full(_N_STEPS, _DT), end=sim)
    return SimpleNamespace(stores={k: v.load() for k, v in stores.items()}, root=root, side=side)


def _run_test(monkeypatch, run, test_name: str) -> tuple[str, list]:
    """Call one reviewed test on ``run`` and classify the outcome as passed, flagged, failed or errored.

    The reviewed module's readers of an arm's stores are replaced by ``run``; its identity logic is untouched.
    ``flagged`` is a pass that recorded the identity as unverified (a warning, a user property) or a skip; the
    classification is deliberately neutral about WHICH of those channels a correct implementation uses.
    """
    arm = SimpleNamespace(name="on")
    monkeypatch.setattr(cand, "_scenario_store", lambda _arm, name: run.stores[name])
    monkeypatch.setattr(cand, "_side_file", lambda _arm: run.side)
    monkeypatch.setattr(cand, "_raw_files", lambda _arm, file_name: [run.root / file_name])
    request = SimpleNamespace(node=SimpleNamespace(user_properties=[]))
    func = getattr(cand, test_name)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            kwargs = {"coupled_arm": lambda name: arm}
            if "request" in inspect.signature(func).parameters:
                kwargs["request"] = request
            func(**kwargs)
        except AssertionError:
            return "failed", []
        except pytest.skip.Exception:
            return "flagged", []
        except Exception as exc:  # noqa: BLE001 -- a decoder refusal under a mutant is a failure of the identity
            return ("failed" if type(exc).__name__ == "ProcessingError" else "errored"), [repr(exc)]
    notes = [str(w.message) for w in caught] + [str(p) for p in request.node.user_properties]
    return ("flagged" if notes else "passed"), notes


_STEP_TEST = "test_the_step_capture_agrees_with_the_side_file_the_per_node_account_and_the_per_step_files"
_BALANCE_TEST = "test_the_balance_closes_and_its_edge_volumes_are_the_step_fluxes_times_the_step_length"
_CLOCK_TEST = "test_the_surface_captures_share_the_step_clock_and_the_interval_clock"


def _mutate(run, capture: str, edit):
    """Replace the decoded store ``capture`` of ``run`` by ``edit(store)``; return ``run``."""
    run.stores[capture] = edit(run.stores[capture])
    return run


def _swap_variables(ds, left: str, right: str):
    """The dataset with the values of two variables exchanged."""
    return ds.assign({left: (ds[left].dims, ds[right].values.copy()), right: (ds[right].dims, ds[left].values.copy())})


def _shift_stamp(ds, dim: str, delta: float):
    """The dataset with its ``simtime`` coordinate moved by ``delta`` seconds."""
    return ds.assign_coords(simtime=(dim, ds["simtime"].values + delta))


def _rescale(ds, name: str, factor: float):
    """The dataset with one variable multiplied by ``factor``, attributes kept."""
    out = ds.copy()
    out[name] = out[name] * factor
    out[name].attrs = ds[name].attrs
    return out


def _permute_ordinals(ds):
    """The step dataset with every non-converged ordinal moved to the next node, wrapping at the node count."""
    out = ds.copy()
    out["nonconverged_node_index"] = (
        out["nonconverged_node_index"].dims,
        (out["nonconverged_node_index"].values + 1) % len(_NODES),
    )
    return out


def _reverse_nodes(ds, name: str):
    """The node-account dataset with one variable's columns reversed against its node coordinate."""
    return ds.assign({name: (ds[name].dims, ds[name].values[:, ::-1].copy())})


# ----------------------------------------------------------------------------------------------------------------
# Mutation arm
# ----------------------------------------------------------------------------------------------------------------


def test_every_identity_test_passes_the_unmodified_hand_built_run(tmp_path, monkeypatch) -> None:
    """Property: the step-capture, balance and clock identity tests of the reviewed file pass on a run in which every
    cross-file identity holds, so a red on a mutant below is the mutant's doing.

    Class: the control of every mutant test in this file. Plausible wrong implementation: a hand-built run whose own
    files disagree, or a reviewed test that fails on a correct run (the reviewed test over-constrains).
    Reachability: the run is written in the solver's byte formats and decoded by the real readers.
    Both-states anchor: asserts on the unmodified set what the mutants contradict.
    Second correct implementation: the run written by the solver on a real model; the same identities hold.
    """
    outcomes = {
        name: _run_test(monkeypatch, _write_run(tmp_path / name), name)
        for name in (_STEP_TEST, _BALANCE_TEST, _CLOCK_TEST)
    }
    assert {k: v[0] for k, v in outcomes.items()} == {
        _STEP_TEST: "passed",
        _BALANCE_TEST: "passed",
        _CLOCK_TEST: "passed",
    }, outcomes


_STEP_MUTANTS = {
    "records stamped with the step start": lambda run: _mutate(
        run, "swmm_step", lambda ds: _shift_stamp(ds, "swmm_step", -_DT)
    ),
    "non-converged ordinals read against another node": lambda run: _mutate(run, "swmm_step", _permute_ordinals),
    "step volumes left in ft3": lambda run: _mutate(
        run,
        "swmm_step",
        lambda ds: _rescale(
            _rescale(ds, "swmm_step_path3_volume_pos", 1 / _FT3_TO_M3), "swmm_step_path3_volume_neg", 1 / _FT3_TO_M3
        ),
    ),
    "node account positive and negative volumes exchanged": lambda run: _mutate(
        run, "swmm_node_path3", lambda ds: _swap_variables(ds, "path3_volume_pos", "path3_volume_neg")
    ),
    "node account non-converged counts against reversed nodes": lambda run: _mutate(
        run, "swmm_node_path3", lambda ds: _reverse_nodes(ds, "nonconverged_steps")
    ),
}


@pytest.mark.parametrize("mutant", sorted(_STEP_MUTANTS))
def test_the_step_capture_test_fails_under_each_wrong_decoder_it_names(tmp_path, monkeypatch, mutant) -> None:
    """Property: the step-capture identity test fails when the T1a records, the T1b account or their join are decoded
    wrongly in any of the ways its docstring names, on a run where the unmodified decode passes it.

    Class: NEW CAPABILITY. Plausible wrong implementation is each parametrized mutant, applied to the decoded store the
    test reads. Reachability: each mutant is one edit to a store the real decoder produced; the test reads that store.
    Both-states anchor: `test_every_identity_test_passes_the_unmodified_hand_built_run` is the control. Killed by:
    deleting the assertion that names the mutated quantity. Second correct implementation: the same identities
    asserted from one structured array per file; the mutants still fail them.
    """
    run = _STEP_MUTANTS[mutant](_write_run(tmp_path))
    assert _run_test(monkeypatch, run, _STEP_TEST)[0] == "failed", mutant


def test_the_step_capture_test_fails_when_the_per_step_node_file_is_decoded_a_step_late(tmp_path, monkeypatch) -> None:
    """Property: the per-step per-node file (T1c) is compared with the step capture's positive total step by step, so a
    file whose rows sit one step off fails the step-capture identity test.

    Class: NEW CAPABILITY. Plausible wrong implementation: the T1c file read with its rows shifted by one record.
    Reachability: the raw file itself is written shifted, so the real reader produces the shifted decode.
    Both-states anchor: the control run writes the file unshifted and passes. Second correct implementation: the
    comparison made on the cumulative sum; a one-row shift still differs at the first and last step.
    """
    run = _write_run(tmp_path, t1c_row_shift=1)
    assert _run_test(monkeypatch, run, _STEP_TEST)[0] == "failed"


_BALANCE_MUTANTS = {
    "residual and storage change exchanged": lambda run: _mutate(
        run, "domain_balance", lambda ds: _swap_variables(ds, "residual_m3", "d_storage_m3")
    ),
    "west and east flux columns exchanged": lambda run: _mutate(
        run,
        "edge_flux",
        lambda ds: ds.assign(edge_flux=(ds["edge_flux"].dims, ds["edge_flux"].values[:, [1, 0, 2, 3]].copy())),
    ),
    "flux decoded with the opposite sign": lambda run: _mutate(
        run, "edge_flux", lambda ds: _rescale(ds, "edge_flux", -1.0)
    ),
    "edge volume read as a flux": lambda run: _mutate(
        run, "domain_balance", lambda ds: _rescale(ds, "edge_north_m3", 1 / _DT)
    ),
}


@pytest.mark.parametrize("mutant", sorted(_BALANCE_MUTANTS))
def test_the_balance_test_fails_under_each_wrong_decoder_it_names(tmp_path, monkeypatch, mutant) -> None:
    """Property: the balance identity test fails when the balance columns or the edge-flux columns are decoded wrongly
    in any way its docstring names, on a run whose edge fluxes are nonzero and distinct per edge.

    Class: NEW CAPABILITY. Plausible wrong implementation is each parametrized mutant. Reachability: each is one edit
    to a decoded store the test reads. Both-states anchor: the unmodified run passes (control test above).
    Second correct implementation: the edge volumes summed per interval with `np.bincount`; the mutants still differ.
    """
    run = _BALANCE_MUTANTS[mutant](_write_run(tmp_path))
    assert _run_test(monkeypatch, run, _BALANCE_TEST)[0] == "failed", mutant


_CLOCK_MUTANTS = {
    "limiter stamped with the step start": lambda run: _mutate(
        run, "dt_limiter", lambda ds: _shift_stamp(ds, "step", -_DT)
    ),
    "edge flux stamped with the step start": lambda run: _mutate(
        run, "edge_flux", lambda ds: _shift_stamp(ds, "step", -_DT)
    ),
    "limiter step length scaled": lambda run: _mutate(run, "dt_limiter", lambda ds: _rescale(ds, "dt_limiter_dt", 2.0)),
    "wet cells on another interval axis": lambda run: _mutate(
        run, "wet_cells", lambda ds: ds.assign_coords(time_s=("timestep", ds["time_s"].values + 0.5))
    ),
}


@pytest.mark.parametrize("mutant", sorted(_CLOCK_MUTANTS))
def test_the_clock_test_fails_under_each_wrong_stamp_it_names(tmp_path, monkeypatch, mutant) -> None:
    """Property: the shared-clock test fails when a per-step capture carries another instant than the step end, a
    limiter step length differs from the step capture's, or an interval capture sits on another interval axis.

    Class: NEW CAPABILITY. Plausible wrong implementation is each parametrized mutant. Reachability: one edit to the
    decoded store the test reads. Both-states anchor: the unmodified run passes (control test above).
    Second correct implementation: the clocks compared through a joined table; the mutants still disagree.
    """
    run = _CLOCK_MUTANTS[mutant](_write_run(tmp_path))
    assert _run_test(monkeypatch, run, _CLOCK_TEST)[0] == "failed", mutant


def test_the_balance_test_does_not_pass_in_silence_on_a_closed_domain_run(tmp_path, monkeypatch) -> None:
    """Property: on a run whose edge fluxes and edge volumes are all zero, the balance test either fails or records
    that its edge-volume identity is unverified; it does not pass in silence, because both sides of that identity are
    zero and a decoder that exchanges two edge columns satisfies it.

    Class: NEW CAPABILITY. Plausible wrong implementation (the reviewed file today): a vacuity record keyed on the
    whole `others` array being zero, so a run with nonzero runoff and zero edges records nothing about the edges.
    Reachability: a closed-domain model, where TRITON's domain edges carry no flux, is the usual case for a synthetic
    sloped plane with a water-level boundary inside the grid; whether the real synthetic run is one is measured on
    Rivanna and not known here. Red-today output: see the round file. Both-states anchor: the swapped-column mutant
    test above fails on a nonzero-edge run, so the instrument reads edge columns when they carry flux.
    Second correct implementation: the test skips the edge half with `pytest.skip`, or fails when every edge volume is
    zero; the classifier counts a skip as flagged and a failure as not silent.
    """
    run = _write_run(tmp_path, zero_edges=True)
    _mutate(
        run,
        "edge_flux",
        lambda ds: ds.assign(edge_flux=(ds["edge_flux"].dims, ds["edge_flux"].values[:, [1, 0, 2, 3]].copy())),
    )
    outcome, notes = _run_test(monkeypatch, run, _BALANCE_TEST)
    assert outcome in ("flagged", "failed"), f"passed in silence on an all-zero edge run: {outcome} {notes}"
    if outcome == "flagged" and notes:
        assert any("edge" in n.lower() for n in notes), f"flagged, but not about the edges: {notes}"


# ----------------------------------------------------------------------------------------------------------------
# Static arm
# ----------------------------------------------------------------------------------------------------------------


def _local_imports() -> list[tuple[str, str]]:
    """Every (module, name) pair of every `from X import Y` statement in the reviewed file, at any depth."""
    return [
        (node.module, alias.name)
        for node in ast.walk(_CAND_TREE)
        if isinstance(node, ast.ImportFrom) and node.module and node.module != "__future__"
        for alias in node.names
    ]


def test_every_name_the_reviewed_file_imports_resolves() -> None:
    """Property: every `from module import name` in the reviewed file, including the ones inside test bodies that
    collection never executes, names a module that imports and an attribute it holds.

    Class: NEW CAPABILITY. Plausible wrong implementation: a lazy import of a name another chunk renamed; the file
    collects and fails one arm-build into a Rivanna job. Reachability: the names are read from the file's syntax tree,
    so a lazy import is counted. Both-states anchor: the control asserts the population is nonempty, so a matcher
    that read no import would fail it. Second correct implementation: the imports hoisted to module level; the same
    names resolve.
    """
    pairs = _local_imports()
    assert len(pairs) >= 15, f"the matcher read only {len(pairs)} import pair(s)"
    unresolved = []
    for module, name in pairs:
        try:
            if not hasattr(importlib.import_module(module), name):
                unresolved.append(f"{module}.{name}")
        except ImportError as exc:
            unresolved.append(f"{module} ({exc})")
    assert unresolved == []


def _requested_arms() -> list[str]:
    """Every string literal passed to `coupled_arm(...)` in the reviewed file's test bodies."""
    return [
        node.args[0].value
        for node in ast.walk(_CAND_TREE)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "coupled_arm"
        and node.args
        and isinstance(node.args[0], ast.Constant)
    ]


def test_every_requested_arm_exists_and_every_declared_arm_is_requested() -> None:
    """Property: every arm name a test requests is a key of the arm table, and every key of the arm table is requested
    by at least one test, so no test builds a missing arm and no arm is built for nothing.

    Class: NEW CAPABILITY. Plausible wrong implementations: a typo in an arm name (a KeyError after a directory is
    made), and an orphan arm left in the table (a full coupled run that no assertion reads). Reachability: the names
    are read from the call sites. Both-states anchor: the control asserts at least eight requests were read.
    Second correct implementation: the arms requested through a parametrized fixture; the sets still match.
    """
    requested = _requested_arms()
    assert len(requested) >= 8, f"the matcher read only {len(requested)} request(s)"
    assert sorted(set(requested) - set(cand._ARM_CONFIGS)) == []
    assert sorted(set(cand._ARM_CONFIGS) - set(requested)) == []


def test_every_config_key_the_arms_set_is_an_analysis_field_that_accepts_the_value() -> None:
    """Property: every key of the uniform and per-arm config tables is a field of `analysis_config` and the field's
    annotation and constraints accept the value the arm sets.

    Class: NEW CAPABILITY. Plausible wrong implementation: a key the config does not have (`extra="forbid"` refuses
    it inside the arm build) or a value outside its field's type. Reachability: the tables are the module's own
    constants. Both-states anchor: the control builds the population (at least nine pairs) and one out-of-range value
    that must be refused, so a validator that accepts everything fails it. Second correct implementation: the
    tables moved into a fixture; the same pairs are validated.
    """
    from hhemt.config.analysis import analysis_config

    pairs = list(cand._UNIFORM_CONFIG.items()) + [kv for arm in cand._ARM_CONFIGS.values() for kv in arm.items()]
    assert len(pairs) >= 9, f"the matcher read only {len(pairs)} (key, value) pair(s)"

    def accepts(key: str, value) -> bool:
        field = analysis_config.model_fields[key]
        annotation = Annotated[(field.annotation, *field.metadata)] if field.metadata else field.annotation
        try:
            TypeAdapter(annotation).validate_python(value)
        except Exception:  # noqa: BLE001 -- any refusal is "does not accept"
            return False
        return True

    assert sorted({k for k, _ in pairs} - set(analysis_config.model_fields)) == []
    assert [(k, v) for k, v in pairs if not accepts(k, v)] == []
    assert not accepts("swmm_coupling_interval", 0), "the validator accepted a value below the field's minimum"
