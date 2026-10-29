"""REVIEWING-SEAT tests for chunk ``1a.1`` (item ``1b.1``), round 2, against shape iteration 5.

The criterion is the approved copy at
``approved/_shape.md`` (sha256 11031641...f53ae5, 969 lines). Section 7.1 items 1-6.

What changed since round 1, and what happened to the two findings I raised
----------------------------------------------------------------------------
Shape iteration 5 re-opened on F1 and F2 and resolved both, differently.

**F1 (the directory-level absence assertion) was resolved AGAINST the directory and FOR the
arm.** Decision 7 determines that the per-target construction's write into the member's
``system_directory`` is CORRECT BEHAVIOUR -- it is the object that compiles, the two
artifacts are that directory's own records, and the work is bounded by DISTINCT TARGETS
rather than by rows -- and section 7.1 item 1 is re-worded to the ARM, with the
directory-level assertion explicitly WITHDRAWN as unsatisfiable by design. My round-1 test
``test_per_member_config_differing_from_master_leaves_the_member_directory_clean`` pinned
that withdrawn criterion, so it is RETIRED from this file. Leaving it would have left a red
assertion indistinguishable, to the next reader, from a red pinning a live requirement.

The retirement is not a deletion of the property, it is an inversion of it:
``test_per_target_construction_still_stamps_the_member_directory`` now asserts the same
observable on the same input with the expectation Decision 7 establishes. That test reds
under Decision 7's rejected Option B -- suppressing the stamp at the per-target construction
-- which is the implementation this round most needs to be stopped from shipping, because it
would green the arm criterion by writing nothing anywhere and would leave the tree the chunk
compiles into as the one tree the migration ladder cannot version.

**F2 (the pre-existing builder suite's empty-placeholder fixtures) was ADOPTED.** Decision 8
adds ``tests/test_per_member_system_configs.py`` to ``1a.1``'s edit surface and the coder
repaired the two fixtures with parseable YAML. My round-1 test
``test_existing_suite_fixture_shape_is_a_regression`` asserted the OLD fixture shape was
workable -- which is Decision 8's rejected Option 3, "make the arm tolerate a ``None``
parse" -- so it is RETIRED too, and
``test_empty_row_yaml_is_refused_not_tolerated`` pins the refusal instead.

What this file decides now, and where each arm sits relative to the coder's
--------------------------------------------------------------------------------
The coder's ``tests/test_sensitivity_system_target_reads_config.py`` decides the ARM
criterion by suppressing ``TRITONSWMM_system`` and asserting the CALL LIST is exactly the
per-target construction. That is the right instrument and it is genuinely red pre-fix.
This file does not restate it. It attacks two positions that file does not reach, and holds
the controls that keep both files honest:

- ``test_yaml_arm_authors_no_construction_when_only_the_system_directory_differs`` -- the
  reuse branch defeated by a config difference OUTSIDE the dedup key. The coder's arm
  differs the row in ``target_dem_resolution``, which changes the dedup key as well as the
  config, so two independent things defeat reuse at once. Here only ``system_directory``
  differs: the dedup key is IDENTICAL to the master's, and the reuse branch is still
  refused, because reuse is byte-equality of the whole config. A fixture author who reasoned
  "same key means reuse" would get a false green; this arm denies them one.
- ``test_empty_row_yaml_is_refused_not_tolerated`` -- the empty-document input position,
  which is distinct from the coder's schema-invalid one and is the exact shape Decision 8's
  Option 3 would have had to tolerate. The two fail by DIFFERENT mechanisms: an empty
  document parses to ``None`` and the loader raises ``ValueError``; a schema-invalid one
  raises ``ConfigurationError``. A test for one does not cover the other.

VENUE: laptop-safe. No solver is compiled or executed. Every root is under ``tmp_path``,
following ``tests/test_system_observer_construction.py``'s stated pattern.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

import hhemt.system as system_mod
from hhemt.config.loaders import load_system_config
from hhemt.exceptions import ConfigurationError
from hhemt.sensitivity_analysis import TRITONSWMM_sensitivity_analysis
from hhemt.system import TRITONSWMM_system

# The two artifacts ``TRITONSWMM_system.__init__`` authors into ``cfg_system
# .system_directory``: the shared log (``system.py:270-284``, suppressible with
# ``skip_log_update``) and the layout stamp (``system.py:300``, UNCONDITIONAL).
_CONSTRUCTOR_ARTIFACTS = ("_version.json", "system_log.json")


# --------------------------------------------------------------------------- fixtures


def _cfg_dict(base: Path, system_directory: Path, dem_resolution: float, software: Path) -> dict:
    """A minimal schema-valid ``system_config`` payload whose every root is under ``base``.

    The four existence-validated inputs are empty placeholders: nothing under test reads
    them, and a real DEM/INP/template would make the test simulation-bearing for no gain.
    ``target_dem_resolution`` is a parameter because it is the dedup key's first element.
    """
    for rel in ("wshed.shp", "dem.tif", "template.cfg", "swmm/hydraulics.inp"):
        fp = base / rel
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_bytes(b"")
    return {
        "crs": {"horizontal_epsg": 3857, "vertical_epsg": 5703},
        "system_directory": str(system_directory),
        "watershed_gis_polygon": str(base / "wshed.shp"),
        "DEM_fullres": str(base / "dem.tif"),
        "TRITONSWMM_software_directory": str(software),
        "TRITONSWMM_git_URL": "https://example.invalid/triton.git",
        "triton_swmm_configuration_template": str(base / "template.cfg"),
        "SWMM_hydraulics": str(base / "swmm" / "hydraulics.inp"),
        "toggle_use_swmm_for_hydrology": False,
        "toggle_use_constant_mannings": True,
        "constant_mannings": 0.03,
        "toggle_triton_model": False,
        "toggle_tritonswmm_model": True,
        "toggle_swmm_model": False,
        "target_dem_resolution": dem_resolution,
    }


def _write_yaml(base: Path, name: str, payload: dict) -> Path:
    dst = base / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(yaml.safe_dump(payload, sort_keys=False))
    return dst


def _stub(master_cfg, analysis_dir: Path) -> SimpleNamespace:
    """Duck-typed stand-in exposing exactly the attributes ``_build_unique_system_targets`` reads.

    Enumerated from the builder body: ``self.experiment.cfg_analysis``
    (``sensitivity_analysis``, ``hpc_ensemble_partition``, ``execution_environment``),
    ``self.experiment.analysis_paths.analysis_dir``, ``self.experiment.cfg_hpc_system``,
    and ``self._system`` with ``cfg_system`` plus the two injected GPU fields the reuse
    branch compares against. ``cfg_hpc_system`` is None so ``resolve_gpu_target`` returns
    ``(None, None)``, matching the stub's ``None`` GPU fields -- so the config alone decides
    the reuse branch here.
    """
    return SimpleNamespace(
        _system=SimpleNamespace(
            cfg_system=master_cfg,
            gpu_hardware=None,
            gpu_compilation_backend=None,
        ),
        experiment=SimpleNamespace(
            cfg_analysis=SimpleNamespace(
                sensitivity_analysis=analysis_dir / "sensitivity.csv",
                hpc_ensemble_partition=None,
                execution_environment="native",
            ),
            analysis_paths=SimpleNamespace(analysis_dir=analysis_dir),
            cfg_hpc_system=None,
        ),
    )


def _artifacts_in(directory: Path) -> list:
    if not directory.exists():
        return []
    return sorted(p.name for p in directory.iterdir() if p.name in _CONSTRUCTOR_ARTIFACTS)


def _suppressing_constructor(calls: list):
    """Stand-in for ``TRITONSWMM_system`` that RECORDS the path it was handed and writes nothing.

    Suppression rather than delegation, because delegation cannot attribute: with the real
    constructor both the arm and the per-target construction write the same two files into
    the same directory, so a file on disk cannot say which of them wrote it. The recorded
    CALL is what carries attribution.
    """

    def _fake(yaml_path, **_kwargs):
        resolved = Path(yaml_path).resolve()
        calls.append(resolved)
        return SimpleNamespace(cfg_system=load_system_config(resolved), system_config_yaml=resolved)

    return _fake


# ------------------------------------------- the arm criterion, at a DIFFERENT position


def test_yaml_arm_authors_no_construction_when_only_the_system_directory_differs(tmp_path, monkeypatch):
    """THE ARM CRITERION at a position the coder's arm does not reach: key-equal, config-different.

    Property: the ``yaml_specified`` arm authors no construction of its own, on a row whose
    config differs from the master's ONLY in ``system_directory`` -- so the reuse branch is
    still refused (reuse is byte-equality of the whole config) while the dedup KEY
    ``(target_dem_resolution, gpu_hardware, gpu_backend)`` is IDENTICAL to the master's.

    Class: REGRESSION -- red on the pre-change tree, where the arm's own construction happens
    first and the call list holds the row's YAML as well as the generated one.

    Why this is required: the coder's arm differs the row in ``target_dem_resolution``, which
    changes the dedup key AND the config, so a fixture author who believed "same key means the
    reuse branch is taken" would be wrong here and right there. This arm removes that
    conflation: the only thing defeating reuse is the config's own byte difference, which is
    what ``model_dump_json()`` equality actually tests.

    What kills it: restoring any construction at the yaml arm, gated or not -- the assertion
    is on the CALL, not on what the construction writes.

    Second correct implementation under which it still passes: an implementation that resolves
    the row's config through ``yaml_to_model(yaml_path, system_config)`` instead of
    ``load_system_config`` by name. Same single construction, same path, same assertion.
    """
    base = tmp_path / "base"
    software = tmp_path / "software"

    master_dir = tmp_path / "master_system"
    master_dir.mkdir(parents=True, exist_ok=True)
    master_yaml = _write_yaml(base, "master.yaml", _cfg_dict(base, master_dir, 3.5, software))

    member_dir = tmp_path / "member_system"
    member_dir.mkdir(parents=True, exist_ok=True)
    # SAME target_dem_resolution as the master; ONLY the system_directory differs.
    member_yaml = _write_yaml(base, "member.yaml", _cfg_dict(base, member_dir, 3.5, software))

    calls: list = []
    monkeypatch.setattr("hhemt.system.TRITONSWMM_system", _suppressing_constructor(calls))

    stub = _stub(load_system_config(master_yaml), tmp_path / "analysis_dir")
    df = pd.DataFrame({"system_config_yaml": [str(member_yaml)]}, index=pd.Index(["m0"]))

    targets = TRITONSWMM_sensitivity_analysis._build_unique_system_targets(stub, df, is_main_orchestrator=True)

    generated_yaml = (tmp_path / "analysis_dir" / "_generated" / "target_0.yaml").resolve()
    assert calls == [generated_yaml], (
        "the yaml arm must not construct a system: expected exactly ONE construction, the "
        f"per-target build's, at {generated_yaml}; observed {calls}"
    )
    assert [t.analysis_ids for t in targets] == [["m0"]], "the row still resolved to one target"

    # The read supplied the TARGET's config: the generated YAML carries the ROW's directory,
    # which is only possible if the arm read the row's file and not the master's.
    dumped = yaml.safe_load(generated_yaml.read_text())
    assert Path(dumped["system_directory"]) == member_dir, (
        f"the generated target YAML must carry the row's own system_directory; it carried {dumped['system_directory']}"
    )


def test_per_target_construction_still_stamps_the_member_directory(tmp_path):
    """Decision 7's determination, pinned: the per-target write into the member directory SURVIVES.

    Property: with the real constructor in place and a row whose config differs from the
    master's, the builder produces BOTH ``_version.json`` and ``system_log.json`` in the
    member's ``system_directory``. That is correct behaviour -- the construction is the object
    that compiles, and the two files are that directory's own records.

    Class: NEW CAPABILITY. Plausible wrong implementation: the one Decision 7 rejected as
    Option B -- passing ``skip_layout_state=True`` or ``skip_log_update=True`` at the
    per-target construction. Under it this test is red with an empty artifact list, and every
    ARM-level assertion in both reviewer files stays green, which is precisely why this arm
    has to exist: the arm criterion can be satisfied by writing nothing anywhere, and this is
    the only assertion that says that answer is wrong.

    Why this is required: it is the counterweight to the arm criterion. Without it, a coder
    could green item 1 by suppressing the target's own layout record, leaving the tree this
    chunk compiles into as the one tree the migration ladder cannot version.

    What kills it: suppressing either write at the per-target construction, or taking the
    reuse branch when the configs differ.

    Second correct implementation: one that reaches the per-target construction through a
    different grouping key, as long as a system is constructed for this row and stamps into
    the row's own directory.

    Not established by this arm: WHICH construction wrote the files. Attribution is the arm
    criterion's job, above and in the coder's file; this arm asserts only that a correct
    implementation still performs the write Decision 7 blesses.
    """
    base = tmp_path / "base"
    software = tmp_path / "software"

    master_dir = tmp_path / "master_system"
    master_dir.mkdir(parents=True, exist_ok=True)
    master_yaml = _write_yaml(base, "master.yaml", _cfg_dict(base, master_dir, 3.5, software))

    member_dir = tmp_path / "member_system"
    member_dir.mkdir(parents=True, exist_ok=True)
    member_yaml = _write_yaml(base, "member.yaml", _cfg_dict(base, member_dir, 5.0, software))

    stub = _stub(load_system_config(master_yaml), tmp_path / "analysis_dir")
    df = pd.DataFrame({"system_config_yaml": [str(member_yaml)]}, index=pd.Index(["m0"]))

    assert _artifacts_in(member_dir) == [], "precondition: the member directory starts clean"

    targets = TRITONSWMM_sensitivity_analysis._build_unique_system_targets(stub, df, is_main_orchestrator=True)

    assert [t.analysis_ids for t in targets] == [["m0"]]
    assert _artifacts_in(member_dir) == sorted(_CONSTRUCTOR_ARTIFACTS), (
        "Decision 7: the per-target construction is a compiling writer and MUST materialise "
        f"the target's own directory; found {_artifacts_in(member_dir)}"
    )


# --------------------------------------------------- controls on the mechanism itself


def test_constructor_authors_both_artifacts_into_the_declared_system_directory(tmp_path):
    """MUST-FIRE CONTROL: construction writes both artifacts into ``system_directory``.

    This is the arm that makes every absence assertion in this file and in the coder's
    non-vacuous. If construction did NOT author these two names at this exact path, an empty
    directory would mean nothing.

    It also settles the path question: production writes to ``self.cfg_system
    .system_directory`` verbatim (``system.py:207`` binds it, ``:270`` builds ``log_path``
    from it, ``:300`` stamps it), so the config-declared path IS the path production resolves
    -- no resolve, no normalisation, no fixture guess.

    Class: NEW CAPABILITY. Killed by: writing the stamp to any other directory, or dropping
    either artifact. Second correct implementation: one that writes the same two names into
    the same resolved directory by a different call path.
    """
    base = tmp_path / "base"
    system_dir = tmp_path / "control_system"
    system_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = _write_yaml(base, "control.yaml", _cfg_dict(base, system_dir, 3.5, tmp_path / "software"))

    declared = load_system_config(yaml_path).system_directory
    assert declared == system_dir, "precondition: the declared directory is the one asserted on"
    assert _artifacts_in(system_dir) == [], "precondition: the directory starts clean"

    TRITONSWMM_system(yaml_path)

    assert _artifacts_in(system_dir) == sorted(_CONSTRUCTOR_ARTIFACTS), (
        f"construction must author BOTH artifacts into cfg_system.system_directory; found {_artifacts_in(system_dir)}"
    )


def test_deleted_arm_reproduced_authors_both_artifacts(tmp_path):
    """RED-pre-fix evidence: the DELETED arm authors both artifacts into the member directory.

    The pre-change source line was ``cfg = TRITONSWMM_system(yaml_path).cfg_system``. It is
    re-implemented here rather than reverted, so the proof that the arm criterion
    discriminates does not depend on a claim about a tree nobody can see.

    Class: NEW CAPABILITY. Killed by: making ``__init__`` suppress the layout stamp by
    default. Second correct implementation: unchanged -- this asserts the behaviour the
    constructor had and still has.
    """
    base = tmp_path / "base"
    member_dir = tmp_path / "member_system"
    member_dir.mkdir(parents=True, exist_ok=True)
    member_yaml = _write_yaml(base, "member.yaml", _cfg_dict(base, member_dir, 3.5, tmp_path / "software"))

    assert _artifacts_in(member_dir) == [], "precondition: the directory starts clean"

    # The deleted expression, verbatim.
    cfg = TRITONSWMM_system(member_yaml).cfg_system

    assert cfg.target_dem_resolution == 3.5, "the deleted arm still returned the parsed config"
    assert _artifacts_in(member_dir) == sorted(_CONSTRUCTOR_ARTIFACTS), (
        f"the deleted arm must author both artifacts, which is what made it red; found {_artifacts_in(member_dir)}"
    )


def test_row_config_drives_the_compile_dedup_key(tmp_path):
    """The row's own config must supply the dedup key, not the master's.

    Class: NEW CAPABILITY. Killed by: replacing the yaml arm's body with
    ``cfg = self._system.cfg_system``, which collapses these two rows into one target.
    Second correct implementation: one reaching two targets by any key derived from each
    row's own resolved config.
    """
    base = tmp_path / "base"
    software = tmp_path / "software"

    master_dir = tmp_path / "master_system"
    master_dir.mkdir(parents=True, exist_ok=True)
    master_yaml = _write_yaml(base, "master.yaml", _cfg_dict(base, master_dir, 3.5, software))

    member_dir = tmp_path / "member_system"
    member_dir.mkdir(parents=True, exist_ok=True)
    member_yaml = _write_yaml(base, "member.yaml", _cfg_dict(base, member_dir, 5.0, software))

    stub = _stub(load_system_config(master_yaml), tmp_path / "analysis_dir")
    df = pd.DataFrame(
        {"system_config_yaml": [str(member_yaml), None]},
        index=pd.Index(["m0", "m1"]),
    )

    targets = TRITONSWMM_sensitivity_analysis._build_unique_system_targets(stub, df, is_main_orchestrator=True)

    assert sorted(tuple(t.analysis_ids) for t in targets) == [("m0",), ("m1",)], (
        "the row's own config must supply the compile-dedup key; a builder that ignores "
        "the row config and falls through to the master collapses these to one target"
    )


def _stub_constructor(monkeypatch, known: dict):
    """Stand-in for ``TRITONSWMM_system`` keyed on the resolved YAML path.

    Mirrors the seam ``tests/test_per_member_system_configs.py`` establishes: a per-row YAML
    yields its stubbed dedup tuple, and a synthesized ``_generated/`` YAML is read back from
    disk.
    """

    def fake_constructor(yaml_path, gpu_hardware=None, gpu_compilation_backend=None, **_kwargs):
        resolved = Path(yaml_path).resolve()
        if resolved in known:
            return _stub_system(known[resolved], resolved)
        if resolved.is_file() and resolved.parent.name == "_generated":
            dumped = yaml.safe_load(resolved.read_text())
            return _stub_system(dumped["target_dem_resolution"], resolved)
        raise AssertionError(f"unexpected yaml_path: {resolved}")

    monkeypatch.setattr(system_mod, "TRITONSWMM_system", fake_constructor)


def _stub_system(dem_resolution: float, yaml_path: Path) -> SimpleNamespace:
    fields = {
        "target_dem_resolution": dem_resolution,
        "gpu_hardware": None,
        "gpu_compilation_backend": None,
    }
    cfg_system = SimpleNamespace(**fields)
    cfg_system.model_dump = lambda **_kwargs: dict(fields)
    cfg_system.model_dump_json = lambda **_kwargs: json.dumps(fields, sort_keys=True)
    return SimpleNamespace(cfg_system=cfg_system, system_config_yaml=yaml_path)


def test_dedup_by_compile_tuple_still_holds_with_real_row_yamls(monkeypatch, tmp_path):
    """The dedup pin, with row YAMLs the loader can actually read.

    This is the fixture shape Decision 8 adopted into ``tests/test_per_member_system_configs.py``;
    it is held here as an independent statement of the same pin, so a future regression in
    either copy is visible from the other.

    Class: NEW CAPABILITY. Killed by: keying the groups on anything but
    ``(cfg.target_dem_resolution, gpu_hardware, gpu_backend)``. Second correct implementation:
    any grouping that collapses equal keys and separates unequal ones.
    """
    base = tmp_path / "base"
    software = tmp_path / "software"

    master_yaml = _write_yaml(base, "master.yaml", _cfg_dict(base, tmp_path / "master_system", 10.0, software))
    yaml_a = _write_yaml(base, "system_A.yaml", _cfg_dict(base, tmp_path / "A", 10.0, software))
    yaml_b = _write_yaml(base, "system_B.yaml", _cfg_dict(base, tmp_path / "B", 10.0, software))
    yaml_c = _write_yaml(base, "system_C.yaml", _cfg_dict(base, tmp_path / "C", 5.0, software))

    known = {
        yaml_a.resolve(): 10.0,
        yaml_b.resolve(): 10.0,
        yaml_c.resolve(): 5.0,
    }
    _stub_constructor(monkeypatch, known)

    stub = _stub(_stub_system(10.0, master_yaml.resolve()).cfg_system, tmp_path / "analysis_dir")
    df = pd.DataFrame(
        {"system_config_yaml": [str(yaml_a), str(yaml_b), str(yaml_c)]},
        index=pd.Index(["0", "1", "2"]),
    )

    targets = TRITONSWMM_sensitivity_analysis._build_unique_system_targets(stub, df, is_main_orchestrator=True)

    assert len(targets) == 2, f"rows 0 and 1 collapse; row 2 does not. got {len(targets)}"
    assert sorted(tuple(t.analysis_ids) for t in targets) == [("0", "1"), ("2",)]


# ------------------------------------------------------------ item 4, and its two guardians


def test_null_software_directory_refusal_names_the_generated_yaml(tmp_path):
    """Section 7.1 item 4: the ``None`` refusal MOVED to the per-target construction.

    Anchored on the property that exists in BOTH states -- WHICH FILE the refusal names.
    Pre-change the arm's own construction named the member YAML; post-change the read accepts
    the nulled field and the refusal arrives at the per-target construction, which is handed
    the generated target YAML. Wording is not asserted.

    Class: REGRESSION (red on the pre-change tree, measured). Killed by: restoring any
    construction at the read site. Second correct implementation: raising from any point in
    the per-target construction that still carries the generated YAML as its ``config_path``.
    """
    base = tmp_path / "base"
    software = tmp_path / "software"

    master_dir = tmp_path / "master_system"
    master_dir.mkdir(parents=True, exist_ok=True)
    master_yaml = _write_yaml(base, "master.yaml", _cfg_dict(base, master_dir, 3.5, software))

    member_dir = tmp_path / "member_system"
    member_dir.mkdir(parents=True, exist_ok=True)
    member_payload = _cfg_dict(base, member_dir, 5.0, software)
    member_payload["TRITONSWMM_software_directory"] = None
    member_yaml = _write_yaml(base, "member.yaml", member_payload)

    # Half one: the READ accepts the nulled field. Without this, "at the construction rather
    # than at the read" is unfalsifiable.
    assert load_system_config(member_yaml).TRITONSWMM_software_directory is None

    stub = _stub(load_system_config(master_yaml), tmp_path / "analysis_dir")
    df = pd.DataFrame({"system_config_yaml": [str(member_yaml)]}, index=pd.Index(["m0"]))

    with pytest.raises(ConfigurationError) as excinfo:
        TRITONSWMM_sensitivity_analysis._build_unique_system_targets(stub, df, is_main_orchestrator=True)

    assert excinfo.value.field == "TRITONSWMM_software_directory"
    assert Path(excinfo.value.config_path) == (tmp_path / "analysis_dir" / "_generated" / "target_0.yaml"), (
        "the refusal must arrive from the per-target construction, which is handed the "
        f"generated target YAML; it named {excinfo.value.config_path}"
    )


def test_schema_invalid_row_yaml_is_still_refused_at_the_read(tmp_path):
    """The arm still PARSES for real: a schema-invalid row config is refused, naming its file.

    Class: NEW CAPABILITY. Plausible wrong implementation: catching the loader's
    ``ConfigurationError`` inside the arm and substituting a default -- which deletes the
    schema validation that is the whole behavioural difference this change introduces.
    Killed by: any arm that stops surfacing the loader's refusal. Second correct
    implementation: validating through ``yaml_to_model`` and letting the same
    ``ConfigurationError`` propagate.
    """
    base = tmp_path / "base"
    software = tmp_path / "software"

    master_dir = tmp_path / "master_system"
    master_dir.mkdir(parents=True, exist_ok=True)
    master_yaml = _write_yaml(base, "master.yaml", _cfg_dict(base, master_dir, 3.5, software))

    bad_payload = _cfg_dict(base, tmp_path / "member_system", 5.0, software)
    bad_payload["target_dem_resolution"] = "not-a-number"
    bad_yaml = _write_yaml(base, "bad_member.yaml", bad_payload)

    stub = _stub(load_system_config(master_yaml), tmp_path / "analysis_dir")
    df = pd.DataFrame({"system_config_yaml": [str(bad_yaml)]}, index=pd.Index(["m0"]))

    with pytest.raises(ConfigurationError) as excinfo:
        TRITONSWMM_sensitivity_analysis._build_unique_system_targets(stub, df, is_main_orchestrator=True)

    assert excinfo.value.field == bad_yaml.name, (
        f"a schema violation in a row's own YAML must be attributed to that file; field was {excinfo.value.field}"
    )


def test_empty_row_yaml_is_refused_not_tolerated(tmp_path):
    """Decision 8 Option 3's refusal, pinned: an EMPTY row config is refused, not defaulted.

    Property: a row whose ``system_config_yaml`` is an empty document -- the shape
    ``tests/test_per_member_system_configs.py``'s old fixtures used, and the shape Decision 8
    refused to keep -- raises out of the builder instead of being tolerated.

    Class: NEW CAPABILITY. Plausible wrong implementation: the refused Option 3, where the arm
    catches the loader's failure and substitutes a default or ``None``; under it the builder
    proceeds and this test sees no exception.

    Why this is required: it is the input position the coder's schema-invalid arm does NOT
    reach, and the two fail by different mechanisms -- an empty document parses to ``None``
    and ``_load_config`` raises ``ValueError``, while a schema-invalid one raises
    ``ConfigurationError``. A guard added for one does not cover the other, and the empty
    document is the exact shape a future "make the fixtures simple again" edit would
    reintroduce.

    What kills it: an arm that tolerates an unparseable row.

    Second correct implementation: one that raises ``ConfigurationError`` for the empty
    document instead of ``ValueError`` -- both refusals are accepted, because the property is
    REFUSAL rather than exception class. That is deliberate: asserting the class would pin a
    loader internal this chunk does not own.
    """
    base = tmp_path / "base"
    software = tmp_path / "software"

    master_dir = tmp_path / "master_system"
    master_dir.mkdir(parents=True, exist_ok=True)
    master_yaml = _write_yaml(base, "master.yaml", _cfg_dict(base, master_dir, 3.5, software))

    empty_yaml = base / "empty_row.yaml"
    empty_yaml.write_text("")

    stub = _stub(load_system_config(master_yaml), tmp_path / "analysis_dir")
    df = pd.DataFrame({"system_config_yaml": [str(empty_yaml)]}, index=pd.Index(["m0"]))

    with pytest.raises((ValueError, ConfigurationError)):
        TRITONSWMM_sensitivity_analysis._build_unique_system_targets(stub, df, is_main_orchestrator=True)
