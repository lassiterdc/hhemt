"""A per-row ``system.{field}`` overlay reaches its own member, builds separately, or is refused by preflight.

The measured defect (Rivanna 2026-11-19, bundle ``phase_b_gpu``): ``_build_unique_system_targets`` grouped rows by a
compile key and kept the FIRST row's system config for the whole group, so a row whose ``system.SWMM_model`` or
``system.observation_points`` differed from an earlier row's with an equal key ran the earlier row's value, with no
message. Members R4, R8, R27, R40, R2e, R29 and R35 all produced one ``hydraulics.inp`` for that reason.

What this file decides, for EVERY field of ``system_config`` and not for the three that were measured:

* a field in the compile key (``_SYSTEM_COMPILE_KEY_FIELDS``) puts two rows that differ in it into two build targets;
* a field read once per compile target from its first row (``_SYSTEM_SETUP_SHARED_FIELDS``) is refused by preflight when
  two rows that share a target differ in it, so it is never discarded in silence;
* every other field reaches the member: the member's system carries the member's own row value.

The three sets are a closed partition of ``system_config.model_fields``, asserted here, so a field added later fails the
partition test until it is classified, and a field in the third set is exercised by the parametrised test below without
the test being edited.

VENUE: laptop-safe. No solver is compiled or run and no ``TRITONSWMM_system`` is constructed: the constructor is
replaced by one that reads the YAML the builder wrote for it and returns a stand-in carrying the loaded config, so a
value that reaches a member only through the member's written YAML is observed as the runner subprocess would read
it. Every root is under ``tmp_path``.
"""

from __future__ import annotations

import typing
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

import hhemt.analysis as analysis_mod
import hhemt.validation as validation_mod
from hhemt.config.analysis import analysis_config
from hhemt.config.loaders import load_system_config
from hhemt.config.system import system_config
from hhemt.sensitivity_analysis import (
    _SYSTEM_COMPILE_KEY_FIELDS,
    _SYSTEM_SETUP_SHARED_FIELDS,
    TRITONSWMM_sensitivity_analysis,
    _UniqueSystemTarget,
)
from hhemt.validation import ValidationResult, _validate_per_member_system_configs
from tests.test_config_validation import _minimal_analysis_config_dict
from tests.test_sensitivity_system_target_reads_config import (
    _orchestrator_stub,
    _suppressing_constructor,
    _system_cfg_dict,
)

_ALL_FIELDS = sorted(system_config.model_fields)
_KEY_FIELDS = sorted(_SYSTEM_COMPILE_KEY_FIELDS)
_SHARED_FIELDS = sorted(_SYSTEM_SETUP_SHARED_FIELDS)
_MEMBER_FIELDS = sorted(set(_ALL_FIELDS) - set(_KEY_FIELDS) - set(_SHARED_FIELDS))

#: Fields a second value of which only validates beside other fields (the schema's own co-requirements), by field.
_COMPANIONS: dict[str, tuple[str, ...]] = {
    "swmm_coupling_mode": ("swmm_coupling_nodes_file",),
    "swmm_coupling_nodes_file": ("swmm_coupling_mode",),
    "toggle_use_swmm_for_hydrology": ("subcatchment_raingage_mapping", "subcatchment_raingage_mapping_gage_id_colname"),
    "toggle_use_constant_mannings": (
        "landuse_lookup_file",
        "landuse_raster",
        "landuse_lookup_class_id_colname",
        "landuse_lookup_mannings_colname",
    ),
    "processed_xllcorner": ("processed_yllcorner", "ncols", "nrows"),
    "processed_yllcorner": ("processed_xllcorner", "ncols", "nrows"),
    "ncols": ("processed_xllcorner", "processed_yllcorner", "nrows"),
    "nrows": ("processed_xllcorner", "processed_yllcorner", "ncols"),
}


def _alternate(tmp_path: Path, name: str, current):
    """A second, schema-valid value for ``system_config.{name}``, different from ``current``.

    A field whose type has no rule here raises, so a field added to ``system_config`` fails loudly until a rule is
    written for it, instead of passing without having been exercised.
    """
    annotation = system_config.model_fields[name].annotation
    text = str(annotation)
    if name == "crs":
        return {"horizontal_epsg": 6440, "vertical_epsg": 5703}
    if isinstance(current, bool):
        return not current
    if name in ("ncols", "nrows"):
        return 11
    if typing.get_args(annotation) and "Literal" in text:
        return next(choice for choice in typing.get_args(annotation) if choice != current)
    if isinstance(current, float) or "float" in text:
        return (current if current is not None else 0.0) + 1.5
    if "Path" in text:
        alt = tmp_path / f"alt_{name}.dat"
        alt.write_bytes(b"")
        return str(alt)
    if text.startswith("list[") or "list[" in text:
        return ["alt_a"]
    if "str" in text:
        return f"{current}_alt" if current else "alt_value"
    raise AssertionError(f"no alternate-value rule for system_config.{name} of type {annotation}")


def _master(tmp_path: Path):
    """The master system config: a real ``system_config`` loaded from a YAML whose every root is under ``tmp_path``."""
    master_yaml = tmp_path / "master.yaml"
    master_yaml.write_text(
        yaml.safe_dump(_system_cfg_dict(tmp_path, tmp_path / "system", tmp_path / "software"), sort_keys=False)
    )
    return load_system_config(master_yaml)


def _overlay_cells(tmp_path: Path, master, name: str) -> dict:
    """The ``system.*`` cells of one overlay row: ``name`` set to a second value, with its schema co-requirements."""
    dump = master.model_dump()
    cells = {name: _alternate(tmp_path, name, dump[name])}
    for companion in _COMPANIONS.get(name, ()):
        cells[companion] = _alternate(tmp_path, companion, dump[companion])
    return cells


def _frame(rows: dict[str, dict]) -> pd.DataFrame:
    """A sensitivity matrix: one row per member id, one ``system.{field}`` column per field any row overlays."""
    columns = sorted({name for cells in rows.values() for name in cells})
    data = {f"system.{c}": [rows[m].get(c, np.nan) for m in rows] for c in columns}
    return pd.DataFrame(data, index=pd.Index(list(rows)))


def _build(tmp_path: Path, monkeypatch, df: pd.DataFrame):
    """Run the real builder over ``df``; return ``(targets, stub, constructed_yaml_paths)``."""
    calls: list[Path] = []
    monkeypatch.setattr("hhemt.system.TRITONSWMM_system", _suppressing_constructor(calls))
    stub = _orchestrator_stub(tmp_path, _master(tmp_path))
    targets = TRITONSWMM_sensitivity_analysis._build_unique_system_targets(stub, df, is_main_orchestrator=True)
    return targets, stub, calls


def _system_of(targets, member_id: str):
    """The system a member runs on, resolved the way ``_create_members`` resolves it."""
    for target in targets:
        if member_id in target.analysis_ids:
            return target.member_systems.get(member_id, target.system)
    raise AssertionError(f"member {member_id} is on no target")


def _preflight(tmp_path: Path, monkeypatch, df: pd.DataFrame) -> ValidationResult:
    """Run the per-member preflight over ``df`` as the matrix, with the master as the primary system config."""
    csv_path = tmp_path / "matrix.csv"
    csv_path.write_text("member_id\n")
    monkeypatch.setattr(validation_mod, "_read_sensitivity_file", lambda _path: df)
    cfg_analysis = SimpleNamespace(
        toggle_sensitivity_analysis=True, sensitivity_analysis=csv_path, hpc_ensemble_partition=None
    )
    result = ValidationResult(context="test")
    _validate_per_member_system_configs(_master(tmp_path), cfg_analysis, result)
    return result


# ------------------------------------------------------------------------------------------ the classification


def test_the_three_field_sets_partition_system_config_exactly():
    """Property: every ``system_config`` field is in exactly one of key, setup-shared and member-reaching.

    Input class: the live ``system_config.model_fields``. Class: NEW CAPABILITY. The plausible wrong implementation is
    a constant that names a retired or misspelled field (the shared set silently covering nothing), or a field added to
    ``system_config`` without a classification (so the parametrised tests below never reach it).
    Both-states anchor: each set is non-empty, the member set holds the two fields the measured defect named, and the
    sets are pairwise disjoint, so the partition is not satisfied by one set swallowing the rest.
    Why it is required: a field outside the classification is an overlay column the builder accepts and nothing checks.
    Killed by: deleting a field from system_config without removing it from a set, or adding one to no set.
    A second correct implementation that derives the sets from field metadata still passes.
    """
    assert set(_KEY_FIELDS) <= set(_ALL_FIELDS)
    assert set(_SHARED_FIELDS) <= set(_ALL_FIELDS)
    assert not set(_KEY_FIELDS) & set(_SHARED_FIELDS)
    assert len(_KEY_FIELDS) + len(_SHARED_FIELDS) + len(_MEMBER_FIELDS) == len(_ALL_FIELDS)
    assert {"SWMM_model", "observation_points"} <= set(_MEMBER_FIELDS)
    assert {"TRITONSWMM_git_URL", "TRITONSWMM_branch_key", "TRITONSWMM_software_directory"} <= set(_KEY_FIELDS)
    assert _KEY_FIELDS and _SHARED_FIELDS and _MEMBER_FIELDS


# ------------------------------------------------------------------------------- member-reaching fields (the defect)


@pytest.mark.parametrize("name", _MEMBER_FIELDS)
def test_a_member_reaching_overlay_field_reaches_its_own_member_on_a_shared_target(tmp_path, monkeypatch, name):
    """Property: for every field outside the compile key and the setup-shared set, each member's system carries the
    member's OWN row value, while the two rows still share one build target.

    Input class: two rows with an equal compile key, one with no overlay (the master) and one overlaying ``name``.
    Class: REGRESSION for the measured defect (red on the pre-change tree, where both members ran the first row's
    config); the pre-change run is pasted in the build return. Reachability: the matrix is built through the real
    ``_build_unique_system_targets``; the member's value is read from the YAML the builder wrote for it, which is what
    the member's runner subprocess loads.
    Both-states anchor: the master row's own member keeps the master value (so the assertion is not satisfied by a
    builder that gives every member the overlay), and the target count is 1 (so it is not satisfied by splitting the
    rows into two builds, which would hide the defect rather than fix it).
    Why it is required: a row's overlay that runs another row's value corrupts the experiment's result with no message.
    Killed by: keeping the first row's config for the group, or giving every member the last row's.
    A second correct implementation that builds a system for every member, not only the differing ones, still passes.
    """
    master = _master(tmp_path)
    cells = _overlay_cells(tmp_path, master, name)
    expected = system_config.model_validate({**master.model_dump(), **cells})
    df = _frame({"m0": {}, "m1": cells})

    targets, _stub, _calls = _build(tmp_path, monkeypatch, df)

    assert len(targets) == 1, f"system.{name} split the rows into {len(targets)} builds; it is not a compile-key field"
    assert getattr(_system_of(targets, "m1").cfg_system, name) == getattr(expected, name)
    assert getattr(_system_of(targets, "m0").cfg_system, name) == getattr(master, name)
    assert getattr(expected, name) != getattr(master, name), "the second value must differ, or the test proves nothing"


def test_two_rows_of_equal_compile_key_with_different_swmm_models_each_run_their_own_model(tmp_path, monkeypatch):
    """Property: rows R4 and R5 share a compile key and name different ``SWMM_model`` files; each member's system
    carries its own file, and the member's YAML on disk names it.

    Input class: the measured Phase B case, three rows over two distinct models. Class: REGRESSION (red before: the
    three members' ``cfg_system.SWMM_model`` was the first row's). Both-states anchor: the third row names the first
    row's model and must still carry it. Killed by: group-first-row reuse. A second correct implementation that
    resolves each member's system from its row at member construction still passes.
    """
    master = _master(tmp_path)
    r4 = tmp_path / "swmm_variants" / "R4.inp"
    r5 = tmp_path / "swmm_variants" / "R5.inp"
    r4.parent.mkdir()
    r4.write_bytes(b"")
    r5.write_bytes(b"")
    df = _frame({"R0": {}, "R4": {"SWMM_model": str(r4)}, "R5": {"SWMM_model": str(r5)}, "R6": {"SWMM_model": str(r4)}})

    targets, _stub, _calls = _build(tmp_path, monkeypatch, df)

    assert [t.analysis_ids for t in targets] == [["R0", "R4", "R5", "R6"]]
    assert {m: str(_system_of(targets, m).cfg_system.SWMM_model) for m in ("R0", "R4", "R5", "R6")} == {
        "R0": str(master.SWMM_model),
        "R4": str(r4),
        "R5": str(r5),
        "R6": str(r4),
    }
    written = yaml.safe_load((tmp_path / "analysis_dir" / "_generated" / "member_R5.yaml").read_text())
    assert written["SWMM_model"] == str(r5)


def test_two_rows_of_equal_compile_key_with_different_observation_points_each_run_their_own(tmp_path, monkeypatch):
    """Property: the ``system.observation_points`` of each row reaches its member, with the master's file as the base.

    Input class: the measured R2e/R3 case: a master that names one observation file, a row naming another, a row naming
    none. Class: REGRESSION. Both-states anchor: the row naming none keeps the master's file (it is not dropped).
    Killed by: group-first-row reuse. A second correct implementation that stores the overlay on the analysis config
    instead is not the one under test and is out of this file's reach.
    """
    base_obs = tmp_path / "obs_base.csv"
    row_obs = tmp_path / "obs_R2e.csv"
    base_obs.write_bytes(b"")
    row_obs.write_bytes(b"")
    cfg = _system_cfg_dict(tmp_path, tmp_path / "system", tmp_path / "software")
    cfg["observation_points"] = str(base_obs)
    (tmp_path / "master.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    calls: list[Path] = []
    monkeypatch.setattr("hhemt.system.TRITONSWMM_system", _suppressing_constructor(calls))
    stub = _orchestrator_stub(tmp_path, load_system_config(tmp_path / "master.yaml"))
    df = _frame({"R0": {}, "R2e": {"observation_points": str(row_obs)}})

    targets = TRITONSWMM_sensitivity_analysis._build_unique_system_targets(stub, df, is_main_orchestrator=True)

    assert len(targets) == 1
    assert str(_system_of(targets, "R2e").cfg_system.observation_points) == str(row_obs)
    assert str(_system_of(targets, "R0").cfg_system.observation_points) == str(base_obs)


# ----------------------------------------------------------------------------------------- compile-key fields


@pytest.mark.parametrize("name", _KEY_FIELDS)
def test_rows_that_differ_in_a_compile_key_field_are_two_build_targets_one_member_each(tmp_path, monkeypatch, name):
    """Property: two rows that differ ONLY in a compile-key field resolve to two targets, one member each, and no member
    needs a per-member system of its own.

    Input class: every field of ``_SYSTEM_COMPILE_KEY_FIELDS``, which includes ``TRITONSWMM_git_URL``,
    ``TRITONSWMM_branch_key`` and ``TRITONSWMM_software_directory`` (each names a different checkout, so a different
    build). Class: NEW CAPABILITY for those three (red before: rows differing only in them were one target, and the
    second row would build from the first row's checkout); REGRESSION for the other two. Both-states anchor: the control
    below, where equal values stay one target. Killed by: dropping a field from the key.
    A second correct implementation that keys on the whole system config minus the member-reaching fields still passes.
    """
    master = _master(tmp_path)
    cells = _overlay_cells(tmp_path, master, name)
    df = _frame({"m0": {}, "m1": cells})

    targets, _stub, _calls = _build(tmp_path, monkeypatch, df)

    assert sorted(t.analysis_ids for t in targets) == [["m0"], ["m1"]]
    assert all(t.member_systems == {} for t in targets)


def test_two_rows_that_differ_only_in_the_software_directory_get_two_build_targets_and_agreeing_rows_get_one(
    tmp_path, monkeypatch
):
    """Control and named case for the key fields: the software directory splits the rows; an equal one does not.

    Class: NEW CAPABILITY, with its control in the same test. The control (two rows naming the SAME second directory
    plus one on the master) shows the split is by value and not by presence of the column. Killed by: a key that is
    unique per row, or one that ignores the directory.
    """
    other = str(tmp_path / "software_R32")
    df = _frame(
        {
            "m0": {},
            "m1": {"TRITONSWMM_software_directory": other},
            "m2": {"TRITONSWMM_software_directory": other},
        }
    )

    targets, _stub, _calls = _build(tmp_path, monkeypatch, df)

    assert sorted(t.analysis_ids for t in targets) == [["m0"], ["m1", "m2"]]
    assert {str(t.system.cfg_system.TRITONSWMM_software_directory) for t in targets} >= {other}


# --------------------------------------------------------------------------- setup-shared fields and preflight


@pytest.mark.parametrize("name", _SHARED_FIELDS)
def test_preflight_refuses_two_rows_of_one_target_that_differ_in_a_setup_shared_field(tmp_path, monkeypatch, name):
    """Property: rows that share a compile target and differ in a field the target's one setup reads from its first row
    are refused by preflight, with the field and both members named; an overlay-only matrix is checked.

    Input class: every field of ``_SYSTEM_SETUP_SHARED_FIELDS``, as an overlay-only matrix (no ``system_config_yaml``
    column). Class: NEW CAPABILITY (red before: ``_validate_per_member_system_configs`` returned at the first line after
    its overlay loop when there was no ``system_config_yaml`` column, so this matrix was never looked at).
    Both-states anchor: the same field overlaid on BOTH rows with one value is accepted, so the refusal is on the
    divergence and not on the field. Killed by: restoring the early return, or checking only the yaml arm.
    A second correct implementation that refuses at builder time instead is out of this test's reach (preflight).
    """
    master = _master(tmp_path)
    cells = _overlay_cells(tmp_path, master, name)

    refused = _preflight(tmp_path, monkeypatch, _frame({"m0": {}, "m1": cells}))
    assert not refused.is_valid
    messages = " | ".join(str(e) for e in refused.errors)
    assert f"system.{name}" in messages
    assert "member_id=m1" in messages and "member_id=m0" in messages

    agreed = _preflight(tmp_path, monkeypatch, _frame({"m0": cells, "m1": cells}))
    assert agreed.is_valid, [str(e) for e in agreed.errors]


@pytest.mark.parametrize("name", _MEMBER_FIELDS + _KEY_FIELDS)
def test_preflight_accepts_rows_that_differ_only_in_a_member_reaching_or_compile_key_field(tmp_path, monkeypatch, name):
    """Property: a divergence the builder honours (a member-reaching field) or builds separately (a compile-key
    field) is NOT refused by preflight.

    Input class: every field outside the setup-shared set. Class: REGRESSION guard against a validator that refuses
    every non-key divergence on an overlay-only matrix, which would refuse the very arms (R4's ``SWMM_model``) this
    change is for. Both-states anchor: the sibling test above refuses a setup-shared divergence on the same matrix
    shape, so this is not a validator that never refuses. Killed by: checking every field.
    """
    master = _master(tmp_path)
    cells = _overlay_cells(tmp_path, master, name)
    if name in ("TRITONSWMM_git_URL", "TRITONSWMM_branch_key"):
        # A second checkout needs its own software directory (see the one-build-directory test below).
        cells["TRITONSWMM_software_directory"] = str(tmp_path / "software_other")
    result = _preflight(tmp_path, monkeypatch, _frame({"m0": {}, "m1": cells}))

    assert result.is_valid, [str(e) for e in result.errors]


def test_the_early_return_on_a_matrix_with_no_yaml_column_no_longer_hides_an_overlay_only_divergence(
    tmp_path, monkeypatch
):
    """Named case for the validator: a matrix whose only per-row system column is ``system.DEM_fullres`` is refused.

    ``DEM_fullres`` is read once per target (the processed DEM is cut from it at setup), so a second value on a row that
    shares the first row's target is discarded. Class: NEW CAPABILITY (red before: valid, no error). Control: the same
    matrix with ``system.SWMM_model`` is valid, so the refusal is not on any overlay.
    """
    other_dem = tmp_path / "other_dem.tif"
    other_dem.write_bytes(b"")
    other_inp = tmp_path / "other.inp"
    other_inp.write_bytes(b"")

    refused = _preflight(tmp_path, monkeypatch, _frame({"m0": {}, "m1": {"DEM_fullres": str(other_dem)}}))
    accepted = _preflight(tmp_path, monkeypatch, _frame({"m0": {}, "m1": {"SWMM_model": str(other_inp)}}))

    assert not refused.is_valid
    assert any("DEM_fullres" in str(e) for e in refused.errors)
    assert accepted.is_valid, [str(e) for e in accepted.errors]


# ------------------------------------------------------------------------------ what must not change


def test_a_matrix_with_no_non_key_overlay_builds_exactly_the_targets_it_built_before(tmp_path, monkeypatch):
    """Property: rows that overlay only a compile-key field build the same targets as before this change: one per
    distinct value, no per-member system, no member YAML written, the master system reused for the master's row.

    Class: REGRESSION guard (byte-identical behaviour for arms with no non-key overlay). Both-states anchor: the
    non-key-overlay tests above show the per-member path is reachable, so its absence here is not a dead branch.
    Killed by: building a member system for every member, or writing a member YAML for a member that equals its target.
    """
    df = _frame(
        {
            "m0": {},
            "m1": {"target_dem_resolution": 7.5},
            "m2": {"target_dem_resolution": 7.5},
        }
    )

    targets, stub, _calls = _build(tmp_path, monkeypatch, df)

    assert sorted(t.analysis_ids for t in targets) == [["m0"], ["m1", "m2"]]
    assert all(t.member_systems == {} for t in targets)
    assert next(t for t in targets if t.analysis_ids == ["m0"]).system is stub._system
    generated = tmp_path / "analysis_dir" / "_generated"
    assert sorted(p.name for p in generated.glob("member_*")) == []
    assert sorted(p.name for p in generated.glob("target_*.yaml")) == ["target_0.yaml", "target_1.yaml"]


def test_a_member_yaml_is_rewritten_with_the_target_yamls_after_the_analysis_directory_is_wiped(tmp_path, monkeypatch):
    """Property: ``_materialize_target_yamls`` re-writes a per-member system YAML, which the member's runner reads.

    Class: NEW CAPABILITY. ``analysis.run(from_scratch=True)`` removes ``_generated/`` after the builder wrote it; the
    target YAMLs were re-written for that reason, and the member YAML is read by the same runners. Both-states anchor:
    the file is asserted absent after the wipe and present after the call. Killed by: re-writing only the target YAMLs.
    """
    master = _master(tmp_path)
    df = _frame({"m0": {}, "m1": _overlay_cells(tmp_path, master, "SWMM_model")})
    targets, _stub, _calls = _build(tmp_path, monkeypatch, df)
    generated = tmp_path / "analysis_dir" / "_generated"
    member_yaml = generated / "member_m1.yaml"
    assert member_yaml.is_file()
    for path in generated.iterdir():
        path.unlink()
    assert not member_yaml.exists()
    instance = SimpleNamespace(
        unique_system_targets=targets,
        experiment=SimpleNamespace(analysis_paths=SimpleNamespace(analysis_dir=tmp_path / "analysis_dir")),
    )

    TRITONSWMM_sensitivity_analysis._materialize_target_yamls(instance)

    assert member_yaml.is_file()
    member_model = targets[0].member_systems["m1"].cfg_system.SWMM_model
    assert yaml.safe_load(member_yaml.read_text())["SWMM_model"] == str(member_model)


def test_create_members_builds_each_member_on_its_own_system_when_the_target_holds_one(tmp_path, monkeypatch):
    """Property: ``_create_members`` constructs a member named in ``member_systems`` on that system and every other
    member on the target's system.

    Class: NEW CAPABILITY (red before: every member of a target was constructed on ``target.system``). Both-states
    anchor: members 0 and 2 stay on the target's system. Killed by: ignoring ``member_systems``.
    """
    master_cfg = analysis_config.model_validate(_minimal_analysis_config_dict(tmp_path))
    instance = TRITONSWMM_sensitivity_analysis.__new__(TRITONSWMM_sensitivity_analysis)
    target_system = SimpleNamespace(name="target")
    own_system = SimpleNamespace(name="own")
    instance.unique_system_targets = [
        _UniqueSystemTarget(0, tmp_path / "t.yaml", target_system, ["0", "1", "2"], member_systems={"1": own_system})
    ]
    instance.df_setup = pd.DataFrame(index=pd.Index(["0", "1", "2"], name="sa_id"))
    instance.member_prefix = "member_"
    instance.members_dir = tmp_path / "members"
    instance._is_main_orchestrator = True
    instance._skip_log_update = True
    instance.recorded_members = {}
    instance.experiment = SimpleNamespace(
        cfg_analysis=master_cfg, analysis_config_yaml=tmp_path / "master.yaml", hpc_system_config_yaml=None
    )
    system_of_member = {}

    def record_member(*, analysis_config_yaml, system, **_kwargs):
        system_of_member[Path(analysis_config_yaml).stem] = system
        return SimpleNamespace(system=system)

    monkeypatch.setattr(analysis_mod, "TRITONSWMM_analysis", record_member)

    instance._create_members()

    assert system_of_member["member_0"] is target_system
    assert system_of_member["member_1"] is own_system
    assert system_of_member["member_2"] is target_system


# ------------------------------------------------------------------------------------ the run-entry inflow check


def test_the_inflow_check_examines_the_template_of_a_member_whose_model_differs_from_its_targets_first_row(
    tmp_path, monkeypatch
):
    """Property: a member that overlays its own ``SWMM_model`` has that template read by the run-entry inflow check.

    Input class: two rows of one compile target, the first row's template valid, the second row's naming an undefined
    inflow node. Class: REGRESSION (red before: ``_systems_run_by`` returned the target's system only, so the second
    row's template was never read and the run reached the queue). The builder, ``_systems_run_by``,
    ``_inflow_check_models`` and ``_refuse_undefined_inflow_nodes`` are the real functions; only the analysis object
    around them is a stand-in. Both-states anchor: the valid first row's template is also listed (the check still
    reads the target) and a matrix whose second row names a valid template is not refused.
    Killed by: enumerating ``target.system`` only. A second correct implementation that checks the member analyses'
    systems instead still passes.
    """
    from hhemt.analysis import TRITONSWMM_analysis
    from hhemt.exceptions import ConfigurationError
    from tests.test_scenario_boundary_guards import _NODES_INP

    good_text = _NODES_INP.format(inflows='J1 FLOW "" FLOW 1 1 0').strip() + "\n"
    bad_text = _NODES_INP.format(inflows='NOSUCH FLOW "" FLOW 1 1 0').strip() + "\n"
    master = _master(tmp_path)
    Path(master.SWMM_model).write_text(good_text)
    bad_inp = tmp_path / "bad.inp"
    bad_inp.write_text(bad_text)
    good_inp = tmp_path / "good_variant.inp"
    good_inp.write_text(good_text.replace("J1 5 5", "J1 6 6"))

    def analysis_over(second_model: Path):
        targets, _stub, _calls = _build(
            tmp_path, monkeypatch, _frame({"m0": {}, "m1": {"SWMM_model": str(second_model)}})
        )
        fake = SimpleNamespace(
            cfg_analysis=SimpleNamespace(toggle_sensitivity_analysis=True),
            _system=master,
            sensitivity=SimpleNamespace(unique_system_targets=targets),
        )
        fake._inflow_check_models = lambda: TRITONSWMM_analysis._inflow_check_models(fake)
        return fake, targets

    refused, targets = analysis_over(bad_inp)
    assert len(targets) == 1 and "m1" in targets[0].member_systems
    labels = [label for label, _text in TRITONSWMM_analysis._inflow_check_models(refused)]
    assert bad_inp in labels and Path(master.SWMM_model) in labels
    with pytest.raises(ConfigurationError, match="NOSUCH"):
        TRITONSWMM_analysis._refuse_undefined_inflow_nodes(refused)

    accepted, _ = analysis_over(good_inp)
    TRITONSWMM_analysis._refuse_undefined_inflow_nodes(accepted)


# ------------------------------------------------------------------------- one build directory per checkout


def test_preflight_refuses_two_checkouts_that_would_build_in_one_software_directory(tmp_path, monkeypatch):
    """Property: rows that differ only in ``TRITONSWMM_git_URL`` are refused, naming both members and the key fields.

    Input class: two rows with an equal ``TRITONSWMM_software_directory`` and different checkouts. Class: NEW CAPABILITY
    (red before: the rows were two build targets that compiled into one directory, and nothing said so). Both-states
    anchor: the control in the next test differs in the software directory too and is accepted, and a row that
    differs in neither is accepted, so the refusal is on the pair (checkout differs, directory does not).
    Killed by: not comparing build directories, or comparing the URL alone. A second correct implementation that
    refuses at builder time is out of this test's reach (preflight).
    """
    result = _preflight(
        tmp_path, monkeypatch, _frame({"m0": {}, "m1": {"TRITONSWMM_git_URL": "https://example.invalid/other.git"}})
    )

    assert not result.is_valid
    message = " | ".join(str(e) for e in result.errors)
    assert "rows m0 and m1 name different checkouts but one software directory" in message
    assert "TRITONSWMM_git_URL" in message and "TRITONSWMM_branch_key" in message
    assert "TRITONSWMM_software_directory" in " | ".join(e.fix_hint or "" for e in result.errors)


def test_preflight_accepts_two_checkouts_with_their_own_software_directories(tmp_path, monkeypatch):
    """Control: rows that differ in the checkout AND in the software directory are accepted; so are equal rows."""
    other = {
        "TRITONSWMM_git_URL": "https://example.invalid/other.git",
        "TRITONSWMM_software_directory": str(tmp_path / "software_other"),
    }
    assert _preflight(tmp_path, monkeypatch, _frame({"m0": {}, "m1": other})).is_valid
    assert _preflight(tmp_path, monkeypatch, _frame({"m0": {}, "m1": {}})).is_valid
