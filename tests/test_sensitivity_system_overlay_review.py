"""Reviewer tests for change 1118-sysoverlay: a per-row ``system.*`` overlay reaches its member or is refused.

Independence. ``tests/test_sensitivity_system_overlay.py`` classifies every ``system_config`` field with the three
constants the change adds (``_SYSTEM_COMPILE_KEY_FIELDS``, ``_SYSTEM_SETUP_SHARED_FIELDS``, the rest) and then asserts
each field behaves as its constant says, so a wrongly classified field passes it. This file derives the classification
from what the code reads, and asserts the OUTCOME over every field of ``system_config.model_fields``.

* The instrument for "a field the target's one setup reads from its first row" is the matcher
  ``cfg_system\\.([A-Za-z_]+)`` over ``src/hhemt/system.py`` and ``src/hhemt/setup_workflow.py``, intersected with
  ``system_config.model_fields`` (the two files are the readers of the per-target ``setup_target_N`` rule); the unit is
  distinct field names, and the control (``crs`` is read, a name that is not a field is dropped) fires below.
* The instrument for "a field that names a different build" is the real ``TRITONSWMM_system`` constructor: two configs
  that differ in a field and yield different ``sys_paths`` name different build or input paths.

Venue: laptop-safe. No solver is compiled or run. Real ``TRITONSWMM_system`` objects are constructed under
``tmp_path`` (the constructor writes ``system_log.json`` and ``_version.json`` there); no test calls a compile method.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

import hhemt.sensitivity_analysis as sa_mod
from hhemt.analysis import _systems_run_by
from hhemt.config.loaders import load_system_config
from hhemt.config.system import system_config
from hhemt.sensitivity_analysis import (
    _SYSTEM_COMPILE_KEY_FIELDS,
    _SYSTEM_SETUP_SHARED_FIELDS,
    TRITONSWMM_sensitivity_analysis,
    _UniqueSystemTarget,
)
from hhemt.system import TRITONSWMM_system
from tests.test_per_member_system_configs import (
    _make_member_instance_for_unit_test,
    _make_stub_system,
    _write_row_system_yaml,
)
from tests.test_sensitivity_system_overlay import (
    _alternate,
    _build,
    _frame,
    _master,
    _overlay_cells,
    _preflight,
    _system_of,
)

_SRC = Path(sa_mod.__file__).resolve().parent
_ALL_FIELDS = sorted(system_config.model_fields)
_READ_RE = re.compile(r"cfg_system\.([A-Za-z_]+)")


def _fields_read_by(*filenames: str) -> set[str]:
    """Distinct ``system_config`` field names read as ``cfg_system.NAME`` in the named files of ``src/hhemt``."""
    names: set[str] = set()
    for filename in filenames:
        names |= set(_READ_RE.findall((_SRC / filename).read_text()))
    return names & set(system_config.model_fields)


#: Fields the per-target setup reads from the target's first row, derived from the readers, never from the constants.
_SETUP_READS = _fields_read_by("system.py", "setup_workflow.py")
#: The compile-identity fields: everything the checkout, the build flags or the processed grid are named by.
_BUILD_IDENTITY = sorted(
    {n for n in _ALL_FIELDS if n.startswith("TRITONSWMM_") and n != "TRITONSWMM_software_directory"}
)
_BUILD_IDENTITY += ["TRITONSWMM_software_directory", "target_dem_resolution"]


def _paths(system) -> dict[str, str]:
    """Every ``sys_paths`` entry of a real system, as strings, so two systems compare by the paths they read."""
    return {k: str(v) for k, v in dataclasses.asdict(system.sys_paths).items()}


def _real_cells(tmp_path: Path, master, name: str) -> dict:
    """``_overlay_cells``, with ``system_directory`` a real directory because the real constructor writes into it."""
    cells = _overlay_cells(tmp_path, master, name)
    if name == "system_directory":
        cells[name] = str(tmp_path / "alt_system_dir")
    return cells


def _real_system(tmp_path: Path, cfg, stem: str) -> TRITONSWMM_system:
    """A real (never compiled) system for ``cfg``, written to ``{stem}.yaml`` under ``tmp_path``."""
    path = tmp_path / f"{stem}.yaml"
    path.write_text(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False))
    return TRITONSWMM_system(path)


def _outcome(
    tmp_path: Path, monkeypatch, name: str, *, overlay_first: bool = False, own_directory: bool = False
) -> dict:
    """Run the builder and preflight over a two-row matrix whose rows differ only in ``system.{name}``.

    Returns ``build`` (``split``: two targets, each on its own value; ``honored``: one target and the overlay row's
    member system carries the overlay value; ``discarded``: one target and the overlay row runs the other row's value),
    ``refused`` (preflight is invalid and names the field) and the targets. ``own_directory`` also gives the overlay row
    its own ``TRITONSWMM_software_directory``, the form in which a checkout field names a build of its own.
    """
    master = _master(tmp_path)
    cells = _overlay_cells(tmp_path, master, name)
    if own_directory:
        cells["TRITONSWMM_software_directory"] = str(tmp_path / "own_checkout")
    expected = getattr(system_config.model_validate({**master.model_dump(), **cells}), name)
    rows = {"m1": cells, "m0": {}} if overlay_first else {"m0": {}, "m1": cells}
    df = _frame(rows)
    targets, _stub, _calls = _build(tmp_path, monkeypatch, df)
    if len(targets) == 2:
        own = {m: getattr(_system_of(targets, m).cfg_system, name) for m in ("m0", "m1")}
        build = "split" if own["m1"] == expected and own["m0"] == getattr(master, name) else "discarded"
    else:
        build = "honored" if getattr(_system_of(targets, "m1").cfg_system, name) == expected else "discarded"
    verdict = _preflight(tmp_path, monkeypatch, df)
    refused = (not verdict.is_valid) and any(name in str(e) for e in verdict.errors)
    return {"build": build, "refused": refused, "targets": targets}


# ----------------------------------------------------------------------------- the derivation, with its control


def test_the_setup_read_set_derived_from_the_readers_equals_the_builders_shared_set():
    """Property: the fields the per-target setup reads, less the compile key, are exactly ``_SETUP_SHARED_FIELDS``.

    Input class: every ``cfg_system.NAME`` read of ``system.py`` and ``setup_workflow.py`` (matcher in the module
    docstring; unit: distinct field names). Class: REGRESSION for the entry's J1 judgement (the classification is
    derived from the reads, so a field the setup reads and the set omits would be discarded in silence).
    Both-states anchor (control): the instrument fires, ``crs`` is read and a non-field name is dropped, so an empty
    intersection cannot pass for a true absence. Why it is required: a shared field outside the set is accepted by
    preflight and discarded at setup. Killed by: removing any field from ``_SYSTEM_SETUP_SHARED_FIELDS``, or adding a
    field to it that no reader reads (which would refuse a legitimate divergence). A second correct implementation
    that derives the set from the readers at import time passes unchanged.
    """
    assert "crs" in _SETUP_READS and "yaml" not in _SETUP_READS
    assert len(_SETUP_READS) > 10
    derived = _SETUP_READS - set(_SYSTEM_COMPILE_KEY_FIELDS)
    assert derived == set(_SYSTEM_SETUP_SHARED_FIELDS), (
        f"read by setup, not shared: {sorted(derived - set(_SYSTEM_SETUP_SHARED_FIELDS))}; "
        f"shared, not read by setup: {sorted(set(_SYSTEM_SETUP_SHARED_FIELDS) - derived)}"
    )


# ----------------------------------------------------------------------- the outcome over every field of the model


@pytest.mark.parametrize("overlay_first", [False, True], ids=["overlay-on-later-row", "overlay-on-first-row"])
@pytest.mark.parametrize("name", _ALL_FIELDS)
def test_no_system_config_field_is_discarded_in_silence_and_each_is_honored_split_or_refused_by_what_reads_it(
    tmp_path, monkeypatch, name, overlay_first
):
    """Property: for EVERY ``system_config`` field, a second value on one row is never discarded without a message, and
    the outcome is the one the field's reader demands: a compile-identity field splits the build, a field the target's
    setup reads is refused by preflight, any other field reaches its own member.

    Input class: every field of ``system_config.model_fields`` (derived from the model, not listed), with the overlay on
    the later row and on the first row (the group's first row is the target's config, so the position is a separate
    input). Class: REGRESSION (red before on every field outside the old key: both members ran the first row's value).
    Reachability: ``_build_unique_system_targets`` and ``_validate_per_member_system_configs`` are the real functions.
    Both-states anchor: the three outcomes each occur over the population (asserted by the sibling census test), so the
    expectation is not satisfied by one arm swallowing the rest. Why it is required: a discarded overlay corrupts an
    experiment's result with no message (members R4, R8, R27, R40, R2e, R29, R35 shared one ``hydraulics.inp``).
    Killed by: reverting the per-member system (``honored`` fields become ``discarded``), dropping a field from the key
    (a ``split`` field becomes ``honored``), or removing the validator call (a refused field becomes ``honored`` with
    no refusal, which this test treats as the discard it is). A second correct implementation that builds a system for
    every member passes, and so does one that refuses every non-key divergence only where the field is unreadable.
    """
    got = _outcome(tmp_path, monkeypatch, name, overlay_first=overlay_first)

    assert got["build"] != "discarded" or got["refused"], f"system.{name} is discarded in silence"
    if name in _SYSTEM_COMPILE_KEY_FIELDS:
        assert got["build"] == "split" and len(got["targets"]) == 2, f"system.{name} names a build: {got['build']}"
    elif name in _SETUP_READS:
        assert got["refused"], f"system.{name} is read once per target from the first row and preflight did not refuse"
    else:
        assert got["build"] == "honored" and not got["refused"], f"system.{name}: {got['build']}, {got['refused']}"


def test_the_population_has_all_three_outcomes_and_names_the_fields_the_defect_measured(tmp_path, monkeypatch):
    """Census: over every field, ``split``, ``refused`` and ``honored`` each occur, and the two measured fields honor.

    The two checkout fields (``TRITONSWMM_git_URL``, ``TRITONSWMM_branch_key``) are refused while the row keeps the
    master's software directory and split when the row also names its own, both asserted, so each field of the model is
    counted once and the arm sizes sum to the model's field count.

    Class: NEW CAPABILITY control for the parametrised test above. Unit: field names; the population is the model's
    fields (all of them are run, so the population the claim ranges over is the one read). Killed by: an
    expectation that is vacuous (every field in one arm). A second correct implementation changes the arm sizes, not
    the presence of each arm.
    """
    seen: dict[str, set[str]] = {"split": set(), "refused": set(), "honored": set()}
    checkout_fields = ("TRITONSWMM_git_URL", "TRITONSWMM_branch_key")
    for name in _ALL_FIELDS:
        (tmp_path / name).mkdir()
        got = _outcome(tmp_path / name, monkeypatch, name)
        if got["refused"]:
            seen["refused"].add(name)
        elif got["build"] in seen:
            seen[got["build"]].add(name)
        if name in checkout_fields:
            (tmp_path / f"{name}_own").mkdir()
            own = _outcome(tmp_path / f"{name}_own", monkeypatch, name, own_directory=True)
            assert own["build"] == "split" and not own["refused"], f"{name} with its own directory: {own['build']}"
    assert all(seen.values()), {k: sorted(v) for k, v in seen.items()}
    assert {"SWMM_model", "observation_points"} <= seen["honored"]
    assert {"TRITONSWMM_software_directory"} <= seen["split"]
    assert set(checkout_fields) <= seen["refused"], "a checkout field on the master's directory shares one build"
    assert sum(len(v) for v in seen.values()) == len(_ALL_FIELDS), "every field lands in exactly one arm"


@pytest.mark.parametrize("name", _BUILD_IDENTITY)
def test_two_rows_differing_only_in_a_build_identity_field_are_two_targets_with_one_member_each(
    tmp_path, monkeypatch, name
):
    """Property: rows that differ only in a ``TRITONSWMM_*`` build field or in ``target_dem_resolution`` build apart.

    Input class: every ``system_config`` field whose name starts ``TRITONSWMM_`` (derived from the model; the entry's
    item (2) names ``TRITONSWMM_git_URL``, ``TRITONSWMM_branch_key`` and ``TRITONSWMM_software_directory``) plus the
    processed-grid resolution. Class: NEW CAPABILITY for the three named fields (red before: one target, and the second
    row would build from the first row's checkout); REGRESSION for the rest. Both-states anchor: the control below, rows
    that agree stay one target. Killed by: dropping any of these from the compile key. A second correct implementation
    that keys on the whole config minus the member-reaching fields passes.
    """
    got = _outcome(tmp_path, monkeypatch, name)

    assert got["build"] == "split"
    assert sorted(t.analysis_ids for t in got["targets"]) == [["m0"], ["m1"]]
    assert all(t.member_systems == {} for t in got["targets"])


# -------------------------------------------------------------------- build artefacts shared by a member's system


@pytest.mark.parametrize("name", [n for n in _ALL_FIELDS if n not in _SETUP_READS and n not in _BUILD_IDENTITY])
def test_a_members_own_system_shares_every_build_and_input_path_of_its_target(tmp_path, monkeypatch, name):
    """Property: for every member-reaching field, the member's REAL system has the same ``sys_paths`` as its target's.

    The entry's claim is that a member on its own system "shares the build artefacts of its group". The suppressed
    constructor of the sibling file cannot test that, so this uses the real ``TRITONSWMM_system`` for the target and for
    the member (constructed by the builder from the member YAML it wrote). Input class: every field outside the compile
    key and the setup reads. Class: NEW CAPABILITY. Both-states anchor: the next test shows fields that DO move a path.
    Why it is required: a member system that pointed at another build directory would run a build nobody compiled.
    Killed by: building the member's system from a config that differs in a path-forming field. A second correct
    implementation that gives the member the target's ``sys_paths`` object passes.
    """
    master = _master(tmp_path)
    master_system = TRITONSWMM_system(tmp_path / "master.yaml")
    cells = _overlay_cells(tmp_path, master, name)
    from tests.test_sensitivity_system_target_reads_config import _orchestrator_stub

    stub = _orchestrator_stub(tmp_path, master)
    stub._system = master_system
    targets = TRITONSWMM_sensitivity_analysis._build_unique_system_targets(
        stub, _frame({"m0": {}, "m1": cells}), is_main_orchestrator=True
    )

    assert len(targets) == 1 and "m1" in targets[0].member_systems
    assert _paths(targets[0].member_systems["m1"]) == _paths(targets[0].system)


@pytest.mark.parametrize("name", _ALL_FIELDS)
def test_a_field_that_moves_a_real_build_or_input_path_is_never_honored_by_a_member_on_a_shared_target(
    tmp_path, monkeypatch, name
):
    """Property: if a second value of a field changes the REAL ``sys_paths``, the rows are split or preflight refuses.

    Independent of every constant in the change: the instrument is the real constructor (two configs, two systems, the
    ``sys_paths`` compared as strings). Input class: every ``system_config`` field. Class: NEW CAPABILITY (red before:
    one target, first row's paths). Both-states anchor: ``TRITONSWMM_software_directory``, ``target_dem_resolution``
    and ``system_directory`` move paths and the census asserts at least three fields do, so the guard is not vacuous.
    Killed by: removing ``TRITONSWMM_software_directory`` from the key and from the shared set at once. A second correct
    implementation that splits on every path-moving field passes.
    """
    master = _master(tmp_path)
    cells = _real_cells(tmp_path, master, name)
    alt = system_config.model_validate({**master.model_dump(), **cells})
    base = _real_system(tmp_path, master, "real_base")
    other = _real_system(tmp_path, alt, "real_alt")
    moved = _paths(base) != _paths(other)
    (tmp_path / "run").mkdir()
    got = _outcome(tmp_path / "run", monkeypatch, name)

    if moved:
        assert got["build"] == "split" or got["refused"], f"system.{name} moves {sorted(_paths(base))} and is honored"


def test_at_least_three_fields_move_a_real_path(tmp_path):
    """Census control for the path-moving guard: the instrument fires on the three fields known to name a path."""
    master = _master(tmp_path)
    base = _real_system(tmp_path, master, "real_base")
    moved = []
    for name in ("TRITONSWMM_software_directory", "target_dem_resolution", "system_directory", "crs"):
        sub = tmp_path / name
        sub.mkdir()
        cells = _real_cells(sub, master, name)
        alt = system_config.model_validate({**master.model_dump(), **cells})
        if _paths(_real_system(sub, alt, "alt")) != _paths(base):
            moved.append(name)
    assert {"TRITONSWMM_software_directory", "target_dem_resolution", "system_directory"} <= set(moved)
    assert "crs" not in moved, "control: a field that names no path must not register as one"


# ----------------------------------------------------------------------- targets that must not change


@pytest.mark.parametrize(
    "rows",
    [
        {"m0": {}, "m1": {}, "m2": {}},
        {"m0": {}, "m1": {"target_dem_resolution": 7.5}, "m2": {"target_dem_resolution": 7.5}, "m3": {}},
        {"m0": {}, "m1": {"TRITONSWMM_extra_c_flags": "-O3"}, "m2": {"TRITONSWMM_extra_c_flags": "-O2"}},
        {
            "m0": {"target_dem_resolution": 7.5, "TRITONSWMM_extra_c_flags": "-O3"},
            "m1": {"TRITONSWMM_extra_c_flags": "-O3"},
            "m2": {"target_dem_resolution": 7.5, "TRITONSWMM_extra_c_flags": "-O3"},
            "m3": {},
        },
    ],
    ids=["no-overlay", "dem-resolution-only", "flags-only", "both-old-key-fields"],
)
def test_arms_with_no_non_key_overlay_keep_the_targets_they_had_before_byte_for_byte(tmp_path, monkeypatch, rows):
    """Property: a matrix overlaying only the OLD compile-key fields yields the old targets, their YAMLs and no more.

    Oracle: the pre-change grouping, written here independently as ``(target_dem_resolution, flags)`` in first-seen
    order. Input class: matrices over the two old key fields, with rows in different orders and a row on the master.
    Class: REGRESSION guard (entry constraint: "arms with no non-key overlay must keep byte-identical behaviour and the
    same build targets"). Compared: the target count and order, each target's member list, each target YAML's bytes
    against ``safe_dump`` of the group's first row config, no member YAML written, ``member_systems`` empty.
    Both-states anchor: the sibling tests show the per-member path is reachable. Killed by: a member system built for a
    member equal to its target, or a changed target order. A second correct implementation with the same grouping
    passes.
    """
    master = _master(tmp_path)
    cfgs = {
        m: system_config.model_validate({**master.model_dump(), **{k: v for k, v in cells.items()}})
        for m, cells in rows.items()
    }
    expected_groups: dict[tuple, list[str]] = {}
    for m, cfg in cfgs.items():
        expected_groups.setdefault((cfg.target_dem_resolution, cfg.TRITONSWMM_extra_c_flags), []).append(m)

    targets, _stub, _calls = _build(tmp_path, monkeypatch, _frame(rows))

    assert [t.analysis_ids for t in targets] == list(expected_groups.values())
    assert all(t.member_systems == {} for t in targets)
    generated = tmp_path / "analysis_dir" / "_generated"
    assert sorted(p.name for p in generated.glob("member_*")) == []
    for target, members in zip(targets, expected_groups.values(), strict=True):
        first = cfgs[members[0]]
        assert (
            target.system_config_yaml.read_bytes()
            == yaml.safe_dump(first.model_dump(mode="json"), sort_keys=False).encode()
        )


# --------------------------------------------------------------------------------- the run-entry enumeration


def test_the_run_entry_enumeration_lists_each_member_system_once_after_its_target_system():
    """Property: ``_systems_run_by`` returns the target system and each DISTINCT member system, each once, in order.

    Input class: two targets, the first holding two members that share one system object and a third on its own, the
    second holding none. Class: REGRESSION (red before: the target systems only). Both-states anchor: the system shared
    by two members appears once and the sensitivity-off arm returns the primary alone. Killed by: iterating
    ``target.system`` only, or listing a shared member system twice. A second correct implementation that walks
    ``sensitivity.members`` and de-duplicates by identity passes.
    """
    t0, t1, a, b = (SimpleNamespace(name=n) for n in ("t0", "t1", "a", "b"))
    targets = [
        SimpleNamespace(system=t0, member_systems={"m1": a, "m2": a, "m3": b}),
        SimpleNamespace(system=t1, member_systems={}),
    ]
    analysis = SimpleNamespace(
        cfg_analysis=SimpleNamespace(toggle_sensitivity_analysis=True),
        sensitivity=SimpleNamespace(unique_system_targets=targets),
        _system="primary",
    )

    assert _systems_run_by(analysis) == [t0, a, b, t1]

    analysis.cfg_analysis.toggle_sensitivity_analysis = False
    assert _systems_run_by(analysis) == ["primary"]


# --------------------------------------------------------------- the two existing tests the stub commit edited


def test_the_dedup_test_whose_stub_was_completed_still_splits_when_a_new_key_field_differs(monkeypatch, tmp_path):
    """Property: the edited ``..._falls_back_to_master_on_null`` flow still yields TWO targets when the stub master's
    ``TRITONSWMM_software_directory`` (a field the stub completion supplies) differs from the row's.

    Class: REGRESSION guard for commit ``c5985f86`` ("completing the stub weakens no assertion"). Reproduces the edited
    test's setup and perturbs the one value the edit copies from the row; the original assertion (one target) holds when
    they agree, which the edited test itself asserts. Killed by: a stub completion that makes the three new key fields
    uncomparable. A second correct implementation completing the stub with a real ``system_config`` passes.
    """
    master_yaml = (tmp_path / "master_cfg_system.yaml").resolve()
    yaml_d = (tmp_path / "system_D.yaml").resolve()
    _write_row_system_yaml(tmp_path, master_yaml, target_dem_resolution=10.0)
    _write_row_system_yaml(tmp_path, yaml_d, target_dem_resolution=10.0)
    instance = _make_member_instance_for_unit_test(
        monkeypatch, {master_yaml: (10.0, None, None), yaml_d: (10.0, None, None)}, tmp_path
    )
    instance._system = _make_stub_system(10.0, None, None, master_yaml)
    row_cfg = load_system_config(yaml_d)
    base_dump = instance._system.cfg_system.model_dump
    perturbed = {"software_directory": None}

    def stub_dump(**kwargs):
        row = row_cfg.model_dump(**kwargs)
        row_values = {
            n: row[n] for n in ("TRITONSWMM_git_URL", "TRITONSWMM_branch_key", "TRITONSWMM_software_directory")
        }
        if perturbed["software_directory"] is not None:
            as_json = kwargs.get("mode") == "json"
            other = perturbed["software_directory"]
            row_values["TRITONSWMM_software_directory"] = str(other) if as_json else other
        return {**base_dump(**kwargs), **row_values}

    instance._system.cfg_system.model_dump = stub_dump
    df = pd.DataFrame({"system_config_yaml": [None, str(yaml_d)], "sa_id": ["0", "1"]}).set_index("sa_id")

    assert len(instance._build_unique_system_targets(df)) == 1  # agreeing values: one target (the edited test's claim)

    perturbed["software_directory"] = tmp_path / "another_checkout"
    assert len(instance._build_unique_system_targets(df)) == 2  # perturbation: the master stub differs, two targets


def test_the_store_test_target_stub_with_member_systems_is_what_the_member_loop_reads():
    """Property: ``_create_members`` reads ``target.member_systems`` as an attribute, so a target stub without it fails.

    Class: REGRESSION guard for the second stub edit of ``c5985f86`` (``member_systems={}`` added to the namespace of
    ``test_sensitivity_store_no_redundancy``): the edit supplies an attribute the production code now requires; it is
    not a loosened assertion. Killed by: removing the attribute from the stub (``AttributeError``). A second correct
    implementation reading it with ``getattr`` would pass both, which is why this asserts the read the code makes.
    """
    source = (_SRC / "sensitivity_analysis.py").read_text()
    assert "target.member_systems.get(member_id, target.system)" in source
    member_field = next(f for f in dataclasses.fields(_UniqueSystemTarget) if f.name == "member_systems")
    assert member_field.default_factory is dict


# ----------------------------------------------------------- a gap between the builder's grouping and the validator's


def test_preflight_refuses_a_setup_shared_divergence_between_two_partitions_the_builder_puts_on_one_target(
    tmp_path, monkeypatch
):
    """Property: if the builder groups two rows into ONE compile target, preflight refuses a setup-shared divergence
    between them, whatever the rows' partition NAMES are.

    The builder groups by the GPU hardware a row's partition resolves to; the validator groups by the partition NAME
    ("which preflight does not resolve", its docstring says), so two CPU partitions that resolve to the same
    (None, None) are one target to the builder and two groups to the validator, and a ``system.DEM_fullres``
    second value is
    discarded in silence. Input class: ``hpc.partition`` values ``cpu_a`` and ``cpu_b`` (no hpc system, so both resolve
    to no GPU) with the second row overlaying ``system.DEM_fullres``. Class: NEW CAPABILITY (the entry's item (3):
    refuse "any case where a row's overlay would otherwise be discarded"). Reachability: ``preflight_validate`` passes
    ``cfg_hpc_system`` to ``validate_analysis_config`` three statements above the call, so the resolution is available
    there; the builder half of this test shows the two rows ARE one target. Both-states anchor: the same matrix with ONE
    partition name is refused today (the premise assertion). Plausible wrong implementation: the current one.
    Killed by: grouping the validator by resolved hardware. A second correct implementation passes that resolves the
    hardware with the analysis' hpc system config when one is given and treats an absent one as no hardware.
    Invocation isolated: not isolatable from this tree (one tree); discrimination shown instead: grouping the
    validator's key by the pair ``resolve_gpu_target`` returns greens this test.
    """
    master = _master(tmp_path)
    other_dem = tmp_path / "other_dem.tif"
    other_dem.write_bytes(b"")
    base = _frame({"m0": {}, "m1": {"DEM_fullres": str(other_dem)}})

    same_name = base.assign(**{"hpc.partition": ["cpu_a", "cpu_a"]})
    assert not _preflight(tmp_path, monkeypatch, same_name).is_valid, "premise: one partition name is refused"

    two_names = base.assign(**{"hpc.partition": ["cpu_a", "cpu_b"]})
    targets, _stub, _calls = _build(tmp_path, monkeypatch, two_names)
    assert [t.analysis_ids for t in targets] == [["m0", "m1"]], "premise: the builder puts both rows on one target"
    assert master.DEM_fullres != other_dem
    verdict = _preflight(tmp_path, monkeypatch, two_names)
    assert not verdict.is_valid and any("DEM_fullres" in str(e) for e in verdict.errors)


def test_the_alternate_value_rule_covers_every_field_so_no_field_is_skipped(tmp_path):
    """Control for the parametrised tests: ``_alternate`` yields a schema-valid second value for every model field."""
    master = _master(tmp_path)
    for name in _ALL_FIELDS:
        cells = _overlay_cells(tmp_path, master, name)
        alt = system_config.model_validate({**master.model_dump(), **cells})
        assert getattr(alt, name) != getattr(master, name), name
    assert _alternate  # the helper is the one the sibling file uses


@pytest.mark.parametrize("name", ["TRITONSWMM_git_URL", "TRITONSWMM_branch_key"])
def test_two_checkouts_on_one_software_directory_are_refused_and_on_two_directories_are_accepted(
    tmp_path, monkeypatch, name
):
    """Property: rows naming different checkouts refuse when they build in one directory and pass when they do not.

    Input class: each checkout field of ``system_config`` (``TRITONSWMM_git_URL``, ``TRITONSWMM_branch_key``), the
    second row on the master's directory, and the same rows with the second row on its own directory. Class: NEW
    CAPABILITY (red before the refusal: the matrix was valid and the two targets built in one directory). Measured
    premise: the real ``sys_paths`` build directory is equal for the two configs when only the checkout differs.
    Both-states anchor: the own-directory matrix is accepted, so the refusal is on the shared directory and not on the
    field. Killed by: removing the refusal, or refusing every checkout divergence. A second correct implementation that
    compares the real ``sys_paths`` build directories passes.
    """
    master = _master(tmp_path)
    cells = _overlay_cells(tmp_path, master, name)
    shared = _preflight(tmp_path, monkeypatch, _frame({"m0": {}, "m1": cells}))
    own = _preflight(
        tmp_path,
        monkeypatch,
        _frame({"m0": {}, "m1": {**cells, "TRITONSWMM_software_directory": str(tmp_path / "own_checkout")}}),
    )

    assert not shared.is_valid
    assert any("name different checkouts but one software directory" in str(e) for e in shared.errors)
    assert own.is_valid, [str(e) for e in own.errors]
