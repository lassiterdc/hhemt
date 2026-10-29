"""A migration DRY RUN materialises no layout state in the tree it is inspecting.

The measured defect: ``run_migration(..., apply=False)`` advertises that it does not touch the
disk, and the runner enforces that promise against exactly ONE directory -- its own
``if apply and ...: stamp_new_target(target_dir, ...)`` guard in
``src/hhemt/version_migration/runner.py``. The dry-run path then calls each planned
migration's ``upgrade(ctx)``, and a migration that computes expected scenario slugs
(``ctx.build_expected_slugs_for_current_version()``, which V0001 calls) constructs a
``TRITONSWMM_system`` and a ``TRITONSWMM_analysis`` against the paths its ``cfg_paths`` name --
a THIRD and FOURTH directory the guard never sees. ``TRITONSWMM_system.__init__`` stamps
``_version.json`` afterwards, and ``TRITONSWMM_analysis.__init__`` creates its resolved
``simulation_directory``. Both pass ``skip_log_update=True``, which governs the shared
``system_log.json`` / ``log.json`` writes and neither of the two artifacts above.

The change adds a second, separately named axis -- ``skip_layout_state`` -- gating exactly
those two sites, and the planning construction passes ``skip_layout_state=self.dry_run``:
``runner._construct_context`` builds the context with ``dry_run=not apply``, so ``self.dry_run``
is True exactly on the planning path. The apply path is untouched: it keeps the default, so a
real migration materialises layout state by design.

VENUE: laptop-safe. No solver is compiled or executed; every root is under ``tmp_path``, the
system config is built from the schema's own required fields with placeholder inputs, and the
analysis construction is EXPECTED to raise after the site under test (see the first test's
docstring) so its exception is suppressed and counted rather than relied on.
"""

import contextlib
from pathlib import Path

import yaml

import hhemt.analysis
import hhemt.system
from hhemt.paths import AnalysisPaths
from hhemt.system import TRITONSWMM_system
from hhemt.version_migration.context import MigrationContext
from tests.test_config_validation import _minimal_analysis_config_dict


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("test")
    return path


def _system_yaml(tmp_path: Path, *, system_directory: Path | None = None) -> Path:
    """A minimal schema-valid system config whose every root is under ``tmp_path``.

    The four existence-validated inputs are empty placeholders: nothing here reads them, and
    a real DEM/INP/template would make the test simulation-bearing for no gain.
    ``TRITONSWMM_software_directory`` is present because a null one is refused at construction
    (a different check, pinned by another test file). The caller pre-creates
    ``system_directory`` when the stamp is expected to land, so an absent ``_version.json`` is
    the code's doing rather than a directory that could not be written to.
    """
    for rel in ("wshed.shp", "dem.tif", "template.cfg", "swmm/hydraulics.inp"):
        fp = tmp_path / rel
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_bytes(b"")
    cfg = {
        "crs": {"horizontal_epsg": 3857, "vertical_epsg": 5703},
        "system_directory": str(system_directory if system_directory is not None else tmp_path / "system"),
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
    """A schema-valid analysis config whose ``analysis_dir`` is ABSOLUTE and under ``tmp_path``.

    The absolute declaration is what keeps the asserted path honest: the constructor resolves
    ``simulation_directory`` as ``analysis_dir / "sims"`` when ``analysis_dir`` is set, and a
    RELATIVE value would resolve against the process cwd instead -- the vacuity trap this
    test's second done-when warns about. ``weather_events_to_simulate`` points at a placeholder
    whose content is not a CSV, so the construction raises AFTER the ``mkdir`` under test; that
    is why the exception is suppressed and the construction counted.
    """
    cfg = _minimal_analysis_config_dict(tmp_path)
    cfg["analysis_dir"] = str(analysis_dir)
    dst = tmp_path / "analysis_config.yaml"
    dst.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return dst


def test_dry_run_materializes_no_layout_state(tmp_path, monkeypatch):
    """Property: a ``dry_run=True`` planning construction leaves no ``_version.json`` and no ``sims/``.

    Class: REGRESSION -- red on the pre-change tree, where the system construction stamps
    ``_version.json`` into its config's ``system_directory`` and the analysis construction
    creates its resolved ``simulation_directory``. The pre-fix run is pasted in the build
    return, with both directories shown created.

    Why this is required: a command documented as planning-only must not create artifacts in
    the tree it is inspecting, and the runner's existing guard covers only its own
    ``target_dir``. A dry run that stamps a version record can also make the NEXT dry run
    report a different starting version than the first.

    What kills it: omitting ``skip_layout_state=self.dry_run`` at either construction in
    ``context.py`` -- the stamp lands, or the mkdir runs, or both. The polarity matters and is
    not incidental: ``runner._construct_context`` passes ``dry_run=not apply``, so
    ``self.dry_run`` is True exactly on the planning path, and the inverted expression would
    suppress layout state on the APPLY path instead while leaving this test exactly as red as
    the pre-change tree.

    Second correct implementation: a helper that builds observer systems for migration
    planning, or a constructor-side default keyed on a caller-declared read-only mode; either
    still routes through the patched constructions, so both the counted attempt and the two
    absences hold.

    The analysis construction is EXPECTED to raise here -- the placeholder weather file is not
    a CSV, and the ``hhemt_sha`` gate follows the site under test. WHICH exception is
    irrelevant and is suppressed, because the site under test (``AnalysisPaths`` construction
    then ``mkdir``) precedes every raise; the captured ``AnalysisPaths`` instance is what keeps
    the suppression from making the assertions vacuous, and it is also what makes the asserted
    path the one PRODUCTION resolved rather than one this test guessed.
    """
    system_dir = tmp_path / "system_tree"
    system_dir.mkdir(parents=True, exist_ok=True)
    cfg_yaml = _system_yaml(tmp_path, system_directory=system_dir)
    analysis_dir = tmp_path / "analysis_root"
    analysis_cfg = _analysis_yaml(tmp_path, analysis_dir)

    assert sorted(p.name for p in system_dir.iterdir()) == [], "precondition: fresh system dir"

    constructed: list[dict[str, object]] = []
    resolved_paths: list[AnalysisPaths] = []

    real_system = hhemt.system.TRITONSWMM_system

    def spy_system(*args: object, **kwargs: object) -> object:
        """Record the constructor kwargs, then delegate so the real side effects still happen.

        Delegation rather than suppression: the artifacts under test are the ones the REAL
        constructor writes, so a stub would make both absence assertions vacuous. The recorded
        kwargs are the non-vacuity control -- a count of 0 means the planning path never
        constructed and the assertions below are about nothing.
        """
        constructed.append(kwargs)
        return real_system(*args, **kwargs)

    real_paths = hhemt.analysis.AnalysisPaths

    def spy_paths(**kwargs: object) -> AnalysisPaths:
        """Capture the ``AnalysisPaths`` instance production builds, then return it unchanged.

        The captured instance is how the asserted path becomes the path PRODUCTION resolved
        rather than one this test guessed: ``simulation_directory`` is read off the object the
        constructor itself created, so a cwd-relative resolution would surface here instead of
        being silently mirrored by a hand-built expectation.
        """
        paths = real_paths(**kwargs)
        resolved_paths.append(paths)
        return paths

    monkeypatch.setattr(hhemt.system, "TRITONSWMM_system", spy_system)
    monkeypatch.setattr(hhemt.analysis, "AnalysisPaths", spy_paths)

    ctx = MigrationContext(
        target_dir=tmp_path / "target",
        dry_run=True,
        migration_id="V0001__rename_scenario_dirs",
        cfg_paths={"system": cfg_yaml, "analysis": analysis_cfg},
    )
    with contextlib.suppress(Exception):
        ctx.build_expected_slugs_for_current_version()

    assert len(constructed) == 1, (
        "migration planning must reach its system construction exactly once; a count of 0 makes "
        "this test vacuous (the planning path never constructed), and a count above 1 means the "
        "seam moved"
    )
    assert resolved_paths, (
        "the analysis construction never reached AnalysisPaths, so the simulation_directory "
        "assertion below would be about a path production never resolved"
    )

    simulation_directory = resolved_paths[-1].simulation_directory
    # Guard against the vacuity trap named in this test's docstring: a cwd-relative resolution
    # would put the asserted path outside the fixture root, where asserting absence proves
    # nothing about the tree the run was inspecting.
    assert simulation_directory.is_relative_to(tmp_path), (
        "the asserted simulation_directory must be the path production resolved INSIDE this "
        f"fixture's root; it resolved to {simulation_directory}"
    )

    assert not simulation_directory.exists(), (
        "a dry run must not create the analysis simulation directory; production resolved "
        f"and CREATED {simulation_directory}"
    )
    assert not (system_dir / "_version.json").exists(), (
        "a dry run must not stamp _version.json into the system tree it is inspecting"
    )


def test_skip_layout_state_suppresses_the_stamp_on_a_direct_construction(tmp_path):
    """Property: ``skip_layout_state=True`` on a direct ``TRITONSWMM_system`` writes no ``_version.json``.

    Class: REGRESSION -- red on the pre-change tree (the flag does not exist, so the call is a
    ``TypeError``); post-change, the stamp is skipped.

    Why this is required: it pins the axis at the SITE that performs the write, independently
    of the migration runner, so the mechanism is correct for any future read-only caller rather
    than only for the one that motivated it.

    What kills it: dropping the gate around ``stamp_new_target``, or making the flag default
    ``True`` (killed by the control below instead).

    Second correct implementation: gating the ``stamp_new_target`` call through a helper that
    takes the flag, as long as a default construction still stamps.
    """
    system_dir = tmp_path / "system_tree"
    system_dir.mkdir(parents=True, exist_ok=True)
    cfg_yaml = _system_yaml(tmp_path, system_directory=system_dir)

    TRITONSWMM_system(cfg_yaml, skip_layout_state=True)

    assert not (system_dir / "_version.json").exists(), (
        "skip_layout_state=True must leave no layout record in the system directory"
    )


def test_default_construction_still_stamps_the_layout_version(tmp_path):
    """CONTROL -- fails an implementation that defaults ``skip_layout_state`` to True.

    Class: NEW CAPABILITY. Plausible wrong implementation: defaulting the new flag to ``True``
    (every construction read-only). Under it, this test is red while the two above are green --
    which is exactly the state that would ship a toolkit whose system trees carry no layout
    record at all.

    Why this is required: the migration ladder reads ``_version.json`` to decide a tree's layout
    version, and the compile rung depends on it. A change that suppressed the stamp everywhere
    would satisfy the two absence assertions while silently removing the record.

    What kills it: a default of ``True``, or a gate that also fires on the default path.

    Second correct implementation: gating the stamp on any explicit opt-out as long as the
    no-argument call stamps.
    """
    system_dir = tmp_path / "system_tree"
    system_dir.mkdir(parents=True, exist_ok=True)
    cfg_yaml = _system_yaml(tmp_path, system_directory=system_dir)

    TRITONSWMM_system(cfg_yaml)

    assert (system_dir / "_version.json").exists(), (
        "the default construction must still stamp the layout version; the migration ladder "
        "and the compile rung both read it"
    )
