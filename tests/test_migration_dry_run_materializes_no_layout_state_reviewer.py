"""REVIEWING-SEAT tests for chunk ``1a.2`` (item ``1b.2``), against shape iteration 17.

The criterion is the approved copy at
``approved/_shape.md`` (sha256 e65b773f...43a60, 1353 lines), section 7.2, items 1-6.

What this file adds over the coder's ``tests/test_migration_dry_run_materializes_no_layout_state.py``
-------------------------------------------------------------------------------------------------
The coder's file is green and its central arm is genuinely RED pre-change: measured against a
pristine copy of HEAD's three modules, it fails with

    AssertionError: a dry run must not create the analysis simulation directory;
    production resolved and CREATED {tmp}/analysis_root/sims

so the asserted path is the path production resolved AND the mkdir line is provably reached.
Three things it does not decide, all of them silent if wrong, and this file supplies them.

1. **The APPLY path.** Every arm in the coder's file builds a ``MigrationContext`` with
   ``dry_run=True`` (or constructs a system directly). None asserts that a REAL migration still
   materialises layout state, and that is measurable rather than arguable: with
   ``skip_layout_state=True`` passed unconditionally at both constructions in ``context.py`` --
   i.e. every migration, dry run or apply, suppressing both sites -- the coder's three tests all
   pass (``3 passed``) while this file's ``test_apply_path_still_materializes_layout_state``
   reds. That is the failure the chunk exists to prevent, reached from the other side.

   A note on the polarity, because this file was written on a hypothesis that a must-fire
   control refuted. ``context.py`` passes ``skip_layout_state=self.dry_run``; the approved
   copy's Change paragraph, the coder's module docstring and one of its test docstrings all
   state ``skip_layout_state=not self.dry_run``, the INVERSE. Running that inversion as a
   mutation reds the CODER's dry-run arm as well as this file's apply-path arm, so the
   behaviour is decided by both files and the inverted expression is a PROSE defect rather than
   an undecided property. This file's apply-path arm is kept for the unconditional-True case
   above, which is a different wrong implementation and is not decided anywhere else.

2. **The analysis half's DEFAULT.** § 7.2 item 3 asks for "one correct implementation per failure
   mode, so an implementation that defaults the flag ``True`` fails". The coder's default arm
   constructs ``TRITONSWMM_system`` only; no arm constructs ``TRITONSWMM_analysis`` with the flag
   omitted. An implementation defaulting it True on the ANALYSIS alone would pass all three of the
   coder's tests while every real run stopped creating its ``sims/`` directory.
   ``test_default_analysis_construction_creates_the_simulation_directory`` closes it.

3. **That the two axes stay separate.** § 7.2's Change refuses to fold the stamp into
   ``skip_log_update`` and gives ``skip_layout_state`` its own name. In round 4 the analysis's
   docstring claimed the flag was forwarded to ``TRITONSWMM_system`` so that no ``_version.json``
   is stamped, and no such forwarding existed -- the analysis never constructs a system, and
   ``grep -n skip_layout_state src/hhemt/analysis.py`` returns the parameter, the docstring, the
   assignment and the mkdir gate, and nothing else. That was filed as F2 and has since been
   repaired: the docstring now states that it does NOT reach the system tree and names the caller
   error it used to invite. ``test_the_two_axes_are_independent`` pins the repaired contract, so a
   later convenience-forwarding change reds here rather than silently re-creating the claim.

VENUE: laptop-safe. No solver is compiled or executed; every root is under ``tmp_path``, the
configs are built from the schemas' own required fields with placeholder inputs, the running-identity
gate (which sits ABOVE the ``sims/`` mkdir) is satisfied by stubbing both identity seams, and the
analysis construction is EXPECTED to raise after the sites under test so its exception is suppressed
and counted. Following ``tests/test_system_observer_construction.py``'s stated pattern.
"""

import contextlib
from pathlib import Path

import yaml

import hhemt.analysis
from hhemt import validation
from hhemt.paths import AnalysisPaths
from hhemt.system import TRITONSWMM_system
from hhemt.validation import RunningIdentity
from hhemt.version_migration.context import MigrationContext
from tests.test_config_validation import _minimal_analysis_config_dict


def _inputs(root: Path) -> None:
    for rel in ("wshed.shp", "dem.tif", "template.cfg", "swmm/hydraulics.inp"):
        fp = root / rel
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_bytes(b"")


def _system_yaml(tmp_path: Path, system_directory: Path) -> Path:
    """A schema-valid system config whose roots are all under ``tmp_path``.

    ``system_directory`` is pre-created by the caller so that an absent ``_version.json`` is the
    code's doing rather than a directory that could not be written to, and a PRESENT one is a
    stamp rather than a fixture leftover.
    """
    _inputs(tmp_path)
    cfg = {
        "crs": {"horizontal_epsg": 3857, "vertical_epsg": 5703},
        "system_directory": str(system_directory),
        "watershed_gis_polygon": str(tmp_path / "wshed.shp"),
        "DEM_fullres": str(tmp_path / "dem.tif"),
        "TRITONSWMM_software_directory": str(tmp_path / "software"),
        "TRITONSWMM_git_URL": "https://example.invalid/triton.git",
        "triton_swmm_configuration_template": str(tmp_path / "template.cfg"),
        "SWMM_hydraulics": str(tmp_path / "swmm" / "hydraulics.inp"),
        "toggle_use_swmm_for_hydrology": False,
        "toggle_use_constant_mannings": True,
        "constant_mannings": 0.03,
        "toggle_triton_model": False,
        "toggle_tritonswmm_model": True,
        "toggle_swmm_model": False,
        "target_dem_resolution": 3.5,
    }
    dst = tmp_path / "system_config.yaml"
    dst.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return dst


def _analysis_yaml(tmp_path: Path, analysis_dir: Path) -> Path:
    """A schema-valid analysis config with an ABSOLUTE ``analysis_dir`` under ``tmp_path``.

    ABSOLUTE is the point: the constructor resolves ``simulation_directory`` from
    ``analysis_dir``, and a relative value would resolve against the process cwd, putting the
    asserted path outside the fixture and making an absence assertion prove nothing.
    """
    cfg = _minimal_analysis_config_dict(tmp_path)
    cfg["analysis_dir"] = str(analysis_dir)
    dst = tmp_path / "analysis_config.yaml"
    dst.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return dst


def _satisfy_identity_gate(monkeypatch, analysis_cfg: Path) -> None:
    """Make the running toolkit's identity equal the sha the analysis config declares.

    The analysis constructor refuses a ``hhemt_sha`` that is not the running toolkit's, and that
    gate sits ABOVE the ``sims/`` mkdir (it runs before the constructor writes anything), so an
    unstubbed run is refused before the site under test and every assertion about that site is
    about nothing. Both seams are stubbed, as ``tests/test_identity_gate_before_mkdir.py`` does:
    ``hhemt.validation.running_identity`` and the name ``hhemt.analysis`` imported from it.
    """
    declared = yaml.safe_load(analysis_cfg.read_text())["hhemt_sha"]
    stub = lambda: RunningIdentity(sha=declared, shape="checkout", dirty=False)  # noqa: E731
    monkeypatch.setattr(validation, "running_identity", stub)
    monkeypatch.setattr(hhemt.analysis, "running_identity", stub)


def _capture_paths(monkeypatch) -> list:
    """Patch ``hhemt.analysis.AnalysisPaths`` to record the instance production builds."""
    real = hhemt.analysis.AnalysisPaths
    captured: list[AnalysisPaths] = []

    def spy(**kwargs):
        paths = real(**kwargs)
        captured.append(paths)
        return paths

    monkeypatch.setattr(hhemt.analysis, "AnalysisPaths", spy)
    return captured


def test_apply_path_still_materializes_layout_state(tmp_path, monkeypatch):
    """THE APPLY PATH still creates its ``sims/`` directory and still stamps ``_version.json``.

    Property: for a ``MigrationContext`` built with ``dry_run=False`` -- the apply path, which is
    what ``runner._construct_context`` passes when ``--apply`` is set -- the analysis's
    PRODUCTION-resolved ``simulation_directory`` EXISTS afterwards, and the system tree carries
    ``_version.json``.

    Class: NEW CAPABILITY. Plausible wrong implementation, and it is the one the approved copy's
    Change paragraph literally prescribes: ``skip_layout_state=not self.dry_run``. On the apply
    path ``self.dry_run`` is False, so that expression yields True and both sites are suppressed;
    under it this test is red on the directory assertion while every arm in the coder's file
    stays green, because none of them ever builds a context with ``dry_run=False``.

    Why this is required: it is the only test in either file that discriminates
    ``skip_layout_state=self.dry_run`` from ``skip_layout_state=not self.dry_run``, and the two
    differ by inverting a safety flag on the path that actually migrates a user's tree.

    What kills it: inverting the polarity, passing the flag unconditionally, or making the flag
    default True.

    Second correct implementation: one that derives the value from ``apply`` instead of
    ``dry_run`` (``skip_layout_state=not apply``) and threads it through the same two
    constructions -- behaviourally identical, and this test asserts behaviour rather than which
    name carries it.
    """
    system_dir = tmp_path / "system_tree"
    system_dir.mkdir(parents=True, exist_ok=True)
    system_cfg = _system_yaml(tmp_path, system_dir)
    analysis_root = tmp_path / "analysis_root"
    analysis_cfg = _analysis_yaml(tmp_path, analysis_root)
    _satisfy_identity_gate(monkeypatch, analysis_cfg)

    assert sorted(p.name for p in system_dir.iterdir()) == [], "precondition: fresh system dir"

    captured = _capture_paths(monkeypatch)

    ctx = MigrationContext(
        target_dir=tmp_path / "target",
        dry_run=False,
        migration_id="V0001__rename_scenario_dirs",
        cfg_paths={"system": system_cfg, "analysis": analysis_cfg},
    )
    with contextlib.suppress(Exception):
        ctx.build_expected_slugs_for_current_version()

    assert captured, (
        "the apply-path analysis construction never reached AnalysisPaths, so the directory "
        "assertion below would be about a path production never resolved"
    )
    sim_dir = captured[-1].simulation_directory
    assert sim_dir.is_relative_to(tmp_path), (
        "the asserted simulation_directory must be the path production resolved INSIDE this "
        f"fixture's root; it resolved to {sim_dir}"
    )
    assert sim_dir.exists(), (
        "the APPLY path must still materialise layout state: production resolved and did NOT "
        f"create {sim_dir}. An implementation passing skip_layout_state=not self.dry_run "
        "suppresses the very write a real migration depends on"
    )
    assert (system_dir / "_version.json").exists(), (
        "the APPLY path must still stamp _version.json; the migration ladder and the compile "
        f"rung both read it, and {system_dir} carries none"
    )


def test_default_analysis_construction_creates_the_simulation_directory(tmp_path, monkeypatch):
    """CONTROL for the ANALYSIS half: the no-argument construction still creates ``sims/``.

    Property: ``TRITONSWMM_analysis(analysis_config_yaml, system)`` with no ``skip_layout_state``
    argument creates its resolved ``simulation_directory``.

    Class: NEW CAPABILITY. Plausible wrong implementation: defaulting the new flag to True on
    ``TRITONSWMM_analysis`` only. Under it every arm in the coder's file still passes -- the dry
    run passes the flag explicitly, and the only default-construction arm in that file builds a
    ``TRITONSWMM_system`` -- while every real run stops creating its ``sims/`` directory. This is
    the analysis-side instance of the failure mode § 7.2 item 3 says must be caught.

    Why this is required: § 7.2 asks for "one correct implementation per failure mode". There are
    TWO constructors carrying this flag and the coder's file pins the default on one of them.

    What kills it: a default of True on the analysis, or a gate that fires on the default path.

    Second correct implementation: gating the mkdir on any explicit opt-out, as long as the
    no-argument construction still creates the directory.
    """
    system_dir = tmp_path / "system_tree"
    system_dir.mkdir(parents=True, exist_ok=True)
    system_cfg = _system_yaml(tmp_path, system_dir)
    system = TRITONSWMM_system(system_cfg)

    analysis_root = tmp_path / "analysis_root"
    analysis_cfg = _analysis_yaml(tmp_path, analysis_root)
    _satisfy_identity_gate(monkeypatch, analysis_cfg)
    expected_sim_dir = analysis_root / "sims"
    assert not expected_sim_dir.exists(), "precondition: the sims directory starts absent"

    with contextlib.suppress(Exception):
        # The identity gate is satisfied above, and it runs BEFORE the mkdir; the placeholder
        # weather file is not a CSV, so the construction is EXPECTED to raise after the mkdir
        # under test. The assertion below is about a directory production
        # created, so the suppression cannot make it vacuous -- an early raise leaves it absent
        # and reds the test rather than greening it.
        hhemt.analysis.TRITONSWMM_analysis(analysis_cfg, system)

    assert expected_sim_dir.exists(), (
        "the DEFAULT analysis construction must still create its simulation_directory; a "
        f"default of True would stop every real run from materialising {expected_sim_dir}"
    )


def test_the_two_axes_are_independent(tmp_path, monkeypatch):
    """``skip_layout_state`` on the ANALYSIS does not suppress the SYSTEM's stamp.

    Property: a system constructed with the default (which stamps) keeps its ``_version.json``
    after an analysis is constructed over it with ``skip_layout_state=True``.

    Class: NEW CAPABILITY. Plausible wrong implementation: "convenience" forwarding, where the
    analysis pushes its flag onto the system it was handed. Nothing in the current code does
    this -- ``grep -n skip_layout_state src/hhemt/analysis.py`` returns the parameter, the
    docstring, the assignment and the mkdir gate, and the analysis never constructs a system --
    and the class's docstring now says so in as many words, after the round-4 form of it claimed
    the opposite and was filed as F2. Under forwarding, this test is red.

    Why this is required: § 7.2 refuses to fold two artifacts into one flag because the flag would
    then be "named for one artifact and justified by one failure". Two independent axes that one
    of them can silently drive is the same collapse reached from the other direction. It is also
    the invariant that makes F2's repair stay repaired: prose cannot enforce it, this test can.

    What kills it: any cross-wiring between the two constructors' flags.

    Second correct implementation: one that derives the analysis's behaviour from its own
    argument by any means, as long as the system it was handed keeps the stamp that system's own
    construction gave it.
    """
    system_dir = tmp_path / "system_tree"
    system_dir.mkdir(parents=True, exist_ok=True)
    system_cfg = _system_yaml(tmp_path, system_dir)
    system = TRITONSWMM_system(system_cfg)
    assert (system_dir / "_version.json").exists(), "precondition: the default construction stamps"

    analysis_root = tmp_path / "analysis_root"
    analysis_cfg = _analysis_yaml(tmp_path, analysis_root)
    _satisfy_identity_gate(monkeypatch, analysis_cfg)
    with contextlib.suppress(Exception):
        hhemt.analysis.TRITONSWMM_analysis(analysis_cfg, system, skip_layout_state=True)

    assert (system_dir / "_version.json").exists(), (
        "an analysis's skip_layout_state must not reach back into the system it was handed; the "
        "two flags govern different artifacts and the analysis does not construct a system"
    )
