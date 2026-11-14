"""The yaml arm of the unique-system-target builder READS a config; it must not BUILD a system.

The measured defect: when a sensitivity CSV row names a per-member ``system_config_yaml``,
``_build_unique_system_targets`` built a whole ``TRITONSWMM_system`` from that path and then
kept only ``.cfg_system`` off the object, discarding it. Construction is not a read.
``TRITONSWMM_system.__init__`` writes ``system_log.json`` into the config's
``system_directory`` (unless the caller passes ``skip_log_update``) and stamps
``_version.json`` there UNCONDITIONALLY (``src/hhemt/system.py``). So the defect was ONE
MUTATION PER ROW at a site whose entire purpose is to read a config -- and the constructor's
one write-suppression flag closes only the FIRST of the two writes, which is why gating the
construction was rejected in favour of removing it.

**What this file tests is the ARM, not a directory, and the distinction is load-bearing.**
``_build_unique_system_targets`` ALSO constructs one ``TRITONSWMM_system`` per DISTINCT TARGET,
from the ``_generated/target_{N}.yaml`` it dumps, and that construction is correct behaviour:
it is the object that drives the sensitivity compile, and the generated YAML carries the
row's own ``system_directory``, so that construction legitimately writes ``_version.json`` and
``system_log.json`` into the same directory a per-member row names. A directory-level
assertion is therefore unsatisfiable for an ordinary per-row config, and asserting one would
be asserting against the sweep doing its job. The criterion this file decides is that the
``yaml_specified`` ARM authors neither artifact -- measured on a row whose config DIFFERS FROM
THE MASTER'S, so the reuse branch cannot hide the answer, with the per-target construction
suppressed so the only thing that could write is the arm.

``test_reuse_branch_row_authors_no_constructor_artifact`` is the SUPPLEMENTARY artifact-level
arm and is labelled as such: it is the only test here that observes real files on disk, and it
observes them on the reuse branch. It is not the criterion and does not claim to be.

The removed construction also carried the ``TRITONSWMM_software_directory is None`` refusal,
which is re-performed at the per-target construction; that test pins the refusal as MOVED
rather than deleted, and ``test_schema_invalid_row_yaml_is_still_refused_at_the_read`` is the
must-fire control for the alternative repair of making the arm tolerate an unparseable file.

VENUE: laptop-safe. No solver is compiled or executed, and no synthetic-analysis fixture is
used: every root is under ``tmp_path`` and the builder is driven through a duck-typed
``SimpleNamespace`` standing in for the sensitivity orchestrator (the pattern
``tests/test_system_observer_construction.py`` establishes for construction-level tests).
Constructing a real ``TRITONSWMM_sensitivity_analysis`` is avoided deliberately -- its
``__init__`` reads a sensitivity CSV, refuses a dirty checkout, and constructs member
analyses, none of which this site depends on, and a builder's uncommitted edits are exactly
that dirty checkout.
"""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

from hhemt.config.loaders import load_system_config
from hhemt.exceptions import ConfigurationError
from hhemt.sensitivity_analysis import TRITONSWMM_sensitivity_analysis

# Both on-disk artifacts the constructor authors into a system directory. Enumerated here
# rather than imported so the assertions name the two files this module's docstring calls
# out, in the same terms a reader will look for them.
_CONSTRUCTOR_ARTIFACTS = ("_version.json", "system_log.json")


def _system_cfg_dict(tmp_path, system_directory, software_directory, target_dem_resolution=3.5):
    """A minimal schema-valid ``system_config`` payload whose every root is under ``tmp_path``.

    The four existence-validated inputs are empty placeholders: nothing in this test reads
    them, and a real DEM/INP/template would make the test simulation-bearing for no gain.
    ``toggle_use_swmm_for_hydrology`` is off so its conditional fields stay unrequired, and
    ``toggle_swmm_model`` is off so ``SWMM_software_directory`` stays unrequired too.
    ``target_dem_resolution`` is a parameter because it is the dedup key's first element: a
    row that differs from the master in it cannot take the reuse branch.
    """
    for rel in ("wshed.shp", "dem.tif", "template.cfg", "swmm/hydraulics.inp"):
        fp = tmp_path / rel
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_bytes(b"")
    return {
        "crs": {"horizontal_epsg": 3857, "vertical_epsg": 5703},
        "system_directory": str(system_directory),
        "watershed_gis_polygon": str(tmp_path / "wshed.shp"),
        "DEM_fullres": str(tmp_path / "dem.tif"),
        "TRITONSWMM_software_directory": None if software_directory is None else str(software_directory),
        "TRITONSWMM_git_URL": "https://example.invalid/triton.git",
        "triton_swmm_configuration_template": str(tmp_path / "template.cfg"),
        "SWMM_hydraulics": str(tmp_path / "swmm" / "hydraulics.inp"),
        "toggle_use_swmm_for_hydrology": False,
        "toggle_use_constant_mannings": True,
        "constant_mannings": 0.03,
        "toggle_triton_model": False,
        "toggle_tritonswmm_model": True,
        "toggle_swmm_model": False,
        "target_dem_resolution": target_dem_resolution,
    }


def _write_system_yaml(tmp_path, name, system_directory, software_directory, target_dem_resolution=3.5):
    dst = tmp_path / name
    dst.write_text(
        yaml.safe_dump(
            _system_cfg_dict(tmp_path, system_directory, software_directory, target_dem_resolution),
            sort_keys=False,
        )
    )
    return dst


def _orchestrator_stub(tmp_path, master_cfg):
    """A duck-typed stand-in exposing exactly the attributes the builder reads.

    Enumerated from the builder body rather than guessed: ``self.experiment.cfg_analysis``
    (``sensitivity_analysis``, ``hpc_ensemble_partition``, ``execution_environment``),
    ``self.experiment.analysis_paths.analysis_dir``, ``self.experiment.cfg_hpc_system``, and
    ``self._system.cfg_system`` plus the two injected GPU fields the reuse branch compares
    against. ``cfg_hpc_system`` is None on purpose: ``resolve_gpu_target`` then returns
    ``(None, None)``, which matches the stub's ``None`` GPU fields, so the injected hardware
    pair never decides a comparison here and the config alone decides the reuse branch.
    """
    return SimpleNamespace(
        _system=SimpleNamespace(
            cfg_system=master_cfg,
            gpu_hardware=None,
            gpu_compilation_backend=None,
        ),
        experiment=SimpleNamespace(
            cfg_analysis=SimpleNamespace(
                sensitivity_analysis=tmp_path / "sensitivity.csv",
                hpc_ensemble_partition=None,
                execution_environment="native",
            ),
            analysis_paths=SimpleNamespace(analysis_dir=tmp_path / "analysis_dir"),
            cfg_hpc_system=None,
        ),
    )


def _suppressing_constructor(calls):
    """A stand-in for ``TRITONSWMM_system`` that RECORDS the path it was handed and writes nothing.

    Suppression rather than delegation, because delegation cannot attribute: with the real
    constructor both the arm and the per-target construction write the same two files into the
    same directory, so a file on disk cannot say which of them wrote it. The recorded CALL is
    what carries attribution, and a call that never happens is the property under test.
    """

    def _fake(yaml_path, **_kwargs):
        resolved = Path(yaml_path).resolve()
        calls.append(resolved)
        return SimpleNamespace(cfg_system=load_system_config(resolved), system_config_yaml=resolved)

    return _fake


def test_yaml_arm_constructs_no_system_when_the_reuse_branch_cannot_be_taken(tmp_path, monkeypatch):
    """THE ARM CRITERION: the ``yaml_specified`` arm never constructs, on a row that cannot reuse.

    Property: for a per-row config that DIFFERS from the master's, the builder constructs
    ``TRITONSWMM_system`` exactly ONCE -- for the generated target YAML the per-target build
    hands it -- and never for the row's own config path.

    Class: REGRESSION -- red on the pre-change tree, where the arm's own construction happens
    FIRST and the call list is ``[member_yaml, generated_yaml]``. The pre-fix run is pasted in
    the build return.

    Why this is required: this is the only assertion here that attributes a write to the ARM
    rather than to the sweep. The member's config differs from the master's in
    ``target_dem_resolution``, so the reuse branch cannot be taken and both call sites are
    live; suppressing the constructor makes each call an observation, and a construction that
    does not happen cannot be observed.

    What kills it: restoring any construction at the yaml arm, including a gated one
    (``TRITONSWMM_system(yaml_path, skip_log_update=True).cfg_system``) -- the layout stamp is
    not gated by that flag, and more to the point the CALL is what this asserts, so a gated
    restoration reds on the call list as well.

    Second correct implementation under which it still passes: an implementation that resolves
    the row's config through ``yaml_to_model(yaml_path, system_config)`` -- the generic loader
    ``load_system_config`` itself delegates to -- rather than through ``load_system_config`` by
    name. Same single construction, same path, same assertion.

    Not established by this arm, stated so the green is not read for more than it is: with the
    constructor suppressed the "into any directory" clause below is a guard against an arm that
    writes a file DIRECTLY (``stamp_new_target``, ``TRITONSWMM_system_log.write``) rather than
    through a construction, and it is not an observation of the real constructor's behaviour.
    ``test_reuse_branch_row_authors_no_constructor_artifact`` is the arm that observes real
    files.
    """
    member_dir = tmp_path / "member_system"
    member_dir.mkdir(parents=True, exist_ok=True)
    member_yaml = _write_system_yaml(
        tmp_path, "member_system.yaml", member_dir, tmp_path / "software", target_dem_resolution=5.0
    )

    master_dir = tmp_path / "master_system"
    master_dir.mkdir(parents=True, exist_ok=True)
    master_yaml = _write_system_yaml(
        tmp_path, "master_system.yaml", master_dir, tmp_path / "software", target_dem_resolution=10.0
    )

    calls: list[Path] = []
    monkeypatch.setattr("hhemt.system.TRITONSWMM_system", _suppressing_constructor(calls))

    stub = _orchestrator_stub(tmp_path, load_system_config(master_yaml))
    df = pd.DataFrame({"system_config_yaml": [str(member_yaml)]}, index=pd.Index(["m0"]))

    targets = TRITONSWMM_sensitivity_analysis._build_unique_system_targets(stub, df, is_main_orchestrator=True)

    generated_yaml = (tmp_path / "analysis_dir" / "_generated" / "target_0.yaml").resolve()
    assert calls == [generated_yaml], (
        "the yaml arm must not construct a system: expected exactly ONE construction, the "
        f"per-target build's, at {generated_yaml}; observed {calls}"
    )
    # The "into any directory" clause: nothing this arm could write directly is on disk.
    authored = sorted(p.name for p in tmp_path.rglob("*") if p.name in _CONSTRUCTOR_ARTIFACTS)
    assert authored == [], f"the yaml arm authored constructor artifacts: {authored}"
    # The builder still RESOLVED the row: one target, carrying the member id.
    assert [t.analysis_ids for t in targets] == [["m0"]]


def test_reuse_branch_row_authors_no_constructor_artifact(tmp_path):
    """SUPPLEMENTARY (not the criterion): a config-equal row takes the reuse branch and writes nothing.

    Property: when a row's config is byte-identical to the master's, the builder reuses the
    master system object and no ``_version.json`` or ``system_log.json`` appears.

    Class: REGRESSION -- red on the pre-change tree, where the yaml arm's construction writes
    both files into this directory regardless of the reuse branch. This is the file's only
    observation of REAL on-disk artifacts, and its scope is stated in its name so it is not
    read as the arm criterion: it passes because the reuse branch was taken, which is why the
    criterion above suppresses the constructor instead of relying on an absence.

    Why this is required: the reuse branch is the path a same-config sweep takes, and a fix
    that made the arm parse but left a construction behind somewhere on that path would show
    up here and nowhere else.

    What kills it: restoring the construction at the arm. Gating it with
    ``skip_log_update=True`` is NOT killed by the ``system_log.json`` half (that write is
    gated) but IS killed by the ``_version.json`` half, whose stamp no flag suppresses.

    Second correct implementation: one that reaches the reuse branch through a different
    equality (a content hash rather than ``model_dump_json``) and writes nothing.
    """
    member_dir = tmp_path / "member_system"
    member_dir.mkdir(parents=True, exist_ok=True)
    member_yaml = _write_system_yaml(tmp_path, "member_system.yaml", member_dir, tmp_path / "software")

    stub = _orchestrator_stub(tmp_path, load_system_config(member_yaml))
    df = pd.DataFrame({"system_config_yaml": [str(member_yaml)]}, index=pd.Index(["m0"]))

    # Precondition: the directory is pre-created and empty, so an absent artifact is the
    # builder's doing and not a directory that could not be written to.
    assert sorted(p.name for p in member_dir.iterdir()) == [], "precondition: fresh member system dir"

    targets = TRITONSWMM_sensitivity_analysis._build_unique_system_targets(stub, df, is_main_orchestrator=True)

    authored = sorted(p.name for p in member_dir.iterdir() if p.name in _CONSTRUCTOR_ARTIFACTS)
    assert authored == [], (
        "a config-equal row reuses the master system object and must author no constructor "
        f"artifact into that config's system_directory; found {authored}"
    )
    assert targets[0].system is stub._system, "a config-equal row reuses the master system object"


def test_master_path_row_still_yields_a_usable_target(tmp_path):
    """CONTROL -- stops an absence from passing vacuously.

    Class: NEW CAPABILITY. Plausible wrong implementation: a builder whose yaml arm (or whose
    fall-through arm) yields nothing usable -- e.g. one that returns an empty target list, or
    one that returns the abandoned construction's ``None``. Under that implementation this
    test is red on ``targets[0]`` (IndexError) or on the identity assertion below.

    Why this is required: two of the arms above assert an ABSENCE (no call, no file), and an
    absence is also what a builder that returns nothing at all produces. Without this arm,
    deleting the builder's body would satisfy them.

    What kills it: any implementation that drops the master fall-through arm, or that returns
    a target whose ``system`` is not the master system for a row carrying no per-member
    config.

    Second correct implementation: one that reaches the same target through a different
    grouping key, as long as the single target carries the row's member id and the master
    system object.
    """
    master_dir = tmp_path / "master_system"
    master_dir.mkdir(parents=True, exist_ok=True)
    master_yaml = _write_system_yaml(tmp_path, "master_system.yaml", master_dir, tmp_path / "software")

    stub = _orchestrator_stub(tmp_path, load_system_config(master_yaml))
    # No ``system_config_yaml`` column at all: the row takes the master fall-through arm.
    df = pd.DataFrame({"some_sensitivity_axis": [1.0]}, index=pd.Index(["m0"]))

    targets = TRITONSWMM_sensitivity_analysis._build_unique_system_targets(stub, df, is_main_orchestrator=True)

    assert len(targets) == 1, "the master fall-through arm must still yield exactly one target"
    assert targets[0].analysis_ids == ["m0"]
    assert targets[0].system is stub._system, "a config-equal row reuses the master system object"
    # The master arm constructs nothing either, so the master directory stays untouchable.
    authored = sorted(p.name for p in master_dir.iterdir() if p.name in _CONSTRUCTOR_ARTIFACTS)
    assert authored == []


def test_null_software_directory_is_refused_at_the_per_target_construction(tmp_path):
    """Property: removing the construction MOVES the ``None`` refusal; it does not delete it.

    Class: REGRESSION -- red on the pre-change tree, where the yaml arm's own construction
    raises FIRST and the error names the MEMBER's YAML. Post-change the read succeeds and the
    refusal arrives from the per-target construction, naming the GENERATED target YAML. The
    pre-fix run is pasted in the build return.

    Why this is required: ``TRITONSWMM_software_directory`` is ``Path | None`` on the model so
    a portability-scrubbed render bundle loads, and the constructor is the chokepoint that
    refuses a ``None``. Deleting the construction at the read site removes that check from the
    read site; the item is only safe if the same refusal still fires where the system is
    really constructed.

    What kills it: dropping the per-target construction's refusal (the check would be lost
    outright), or keeping a construction at the read site (the error would name the member
    YAML again -- the pre-fix behaviour).

    Second correct implementation: raising from any point in the per-target construction that
    still carries the generated YAML as its ``config_path`` -- the assertion turns on which
    file the refusal names, which is a property of both states, not on wording this change
    introduces.
    """
    member_dir = tmp_path / "member_system"
    member_dir.mkdir(parents=True, exist_ok=True)
    member_yaml = _write_system_yaml(tmp_path, "member_system.yaml", member_dir, None)

    master_dir = tmp_path / "master_system"
    master_dir.mkdir(parents=True, exist_ok=True)
    master_yaml = _write_system_yaml(tmp_path, "master_system.yaml", master_dir, tmp_path / "software")

    # Half one: the READ the yaml arm now performs accepts the nulled field. This is what
    # makes "at the per-target construction rather than at the read" checkable.
    assert load_system_config(member_yaml).TRITONSWMM_software_directory is None

    stub = _orchestrator_stub(tmp_path, load_system_config(master_yaml))
    df = pd.DataFrame({"system_config_yaml": [str(member_yaml)]}, index=pd.Index(["m0"]))

    with pytest.raises(ConfigurationError) as excinfo:
        TRITONSWMM_sensitivity_analysis._build_unique_system_targets(stub, df, is_main_orchestrator=True)

    assert excinfo.value.field == "TRITONSWMM_software_directory"
    # The discriminating half: the refusal names the GENERATED target YAML the per-target
    # construction was handed, not the member YAML the read was handed.
    generated_yaml = tmp_path / "analysis_dir" / "_generated" / "target_0.yaml"
    assert excinfo.value.config_path == generated_yaml, (
        "the None refusal must arrive from the per-target construction, which is handed the "
        f"generated target YAML; it named {excinfo.value.config_path}"
    )


def test_schema_invalid_row_yaml_is_still_refused_at_the_read(tmp_path):
    """MUST-FIRE CONTROL: the arm still PARSES for real, so a malformed row config is refused.

    Class: NEW CAPABILITY. Plausible wrong implementation: making the read TOLERATE an
    unparseable file -- catching the loader's ``ConfigurationError`` and substituting a default
    or ``None`` -- which is the alternative repair this item refuses, because it deletes the
    schema validation that is the whole behavioural difference the change introduces. Under
    that implementation the builder proceeds past the read (the ``with pytest.raises`` block
    sees no raise, or sees an ``AttributeError`` from reading a field off ``None``) and this
    test is red.

    Why this is required: the change's own claim is that the arm now performs the parse the
    constructor's config step performed. A test that only asserts the construction is gone
    cannot distinguish "parses for real" from "parses nothing", and those are the two states
    this item exists to separate.

    What kills it: any arm that stops surfacing the loader's refusal for a schema-invalid row.

    Second correct implementation: an arm that validates through ``yaml_to_model`` and lets the
    same ``ConfigurationError`` propagate, or one that wraps the loader and re-raises it with
    the row's identity attached -- as long as ``ConfigurationError`` naming the row file still
    reaches the caller.
    """
    bad_yaml = tmp_path / "bad_system.yaml"
    bad_yaml.write_text(
        yaml.safe_dump({"crs": "not-a-mapping", "target_dem_resolution": "not-a-number"}, sort_keys=False)
    )

    master_dir = tmp_path / "master_system"
    master_dir.mkdir(parents=True, exist_ok=True)
    master_yaml = _write_system_yaml(tmp_path, "master_system.yaml", master_dir, tmp_path / "software")

    stub = _orchestrator_stub(tmp_path, load_system_config(master_yaml))
    df = pd.DataFrame({"system_config_yaml": [str(bad_yaml)]}, index=pd.Index(["m0"]))

    with pytest.raises(ConfigurationError) as excinfo:
        TRITONSWMM_sensitivity_analysis._build_unique_system_targets(stub, df, is_main_orchestrator=True)

    assert excinfo.value.field == "bad_system.yaml", (
        "a schema-invalid row config must be refused by the load, naming the row's own file; "
        f"the error named {excinfo.value.field!r}"
    )
