"""Observer-mode system construction authors no write to the SHARED ``system_log.json``.

The measured defect (change 1028-1828): every simulation runner constructed
``TRITONSWMM_system`` at start-up, whose ``_sync_compilation_status_on_init`` read five
compile-status properties, each of which persisted its derived verdict through
``_sync_compilation_log_field`` -> ``LogField.set``/``clear`` -> ``log.write()``. With ~30
concurrent runners on Rivanna WekaFS the create-capable ``os.open`` in ``write()``
intermittently raised ``FileNotFoundError`` (``log write (open failed)``; 14 of 822 SLURM
logs across the four 4a3r2 synth_cc arms, 1 of 196 on 4a3r3).

The fix is the shape ``TRITONSWMM_analysis`` already carries: a ``skip_log_update``
constructor flag under which the process OBSERVES compile state and persists nothing.

VENUE: laptop-safe. No solver is compiled or executed and no synthetic-analysis fixture is
used -- the system config is built here from the schema's own required fields, with
placeholder input files and every root under ``tmp_path``. The analysis fixtures are
deliberately avoided because ``TRITONSWMM_analysis`` refuses a dirty checkout, and a
builder's uncommitted edits are exactly that.

Second correct implementation, one per failure mode: ``test_default_construction_still_writes``
fails an implementation that makes the flag default TRUE (everyone read-only);
``test_observer_property_read_derives_fresh_and_persists_nothing`` fails an implementation
that only silences the constructor's ``_sync_compilation_status_on_init`` call while leaving
the property reads free to persist -- the insufficient shape measured in round 1 of this
change, where the same write stays reachable from ``scenario.py``'s property reads.
"""

import json
from pathlib import Path

import yaml

from hhemt.system import TRITONSWMM_system

_TRITON_OK = "[100%] Built target triton.exe\nBuild finished\n"


def _system_yaml(tmp_path: Path) -> Path:
    """A minimal schema-valid system config whose every root is under ``tmp_path``.

    The four existence-validated inputs are created as empty placeholders: construction
    reads none of them, and a real DEM/INP/template would make this test simulation-bearing
    for no gain. ``toggle_use_swmm_for_hydrology`` and the landuse path are off, so their
    conditional fields stay unrequired.
    """
    for rel in ("wshed.shp", "dem.tif", "template.cfg", "swmm/hydraulics.inp"):
        fp = tmp_path / rel
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_bytes(b"")
    cfg = {
        "crs": {"horizontal_epsg": 3857, "vertical_epsg": 5703},
        "system_directory": str(tmp_path / "system"),
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


def _system_log_path(cfg_yaml: Path) -> Path:
    cfg = yaml.safe_load(cfg_yaml.read_text())
    return Path(cfg["system_directory"]) / "system_log.json"


def test_observer_construction_does_not_touch_an_existing_system_log(tmp_path):
    """RED pre-fix: the flag does not exist (TypeError). Post-fix: bytes AND mtime stable."""
    cfg_yaml = _system_yaml(tmp_path)
    probe = TRITONSWMM_system(cfg_yaml)  # default mode: creates and writes the document
    log_path = Path(probe.log.logfile)
    assert log_path.exists(), "precondition: the default construction writes the system log"

    before = (log_path.stat().st_mtime_ns, log_path.read_bytes())
    observer = TRITONSWMM_system(cfg_yaml, skip_log_update=True)
    after = (log_path.stat().st_mtime_ns, log_path.read_bytes())

    assert observer.log.logfile == log_path, "the observer still reads the SAME document"
    assert after == before, "observer construction must not rewrite the shared system log"


def test_observer_construction_creates_no_system_log(tmp_path):
    """RED pre-fix (TypeError). Post-fix: an absent document stays absent.

    Pre-fix this is the branch that CREATES the file: ``from_json`` raises
    FileNotFoundError, the constructor builds a default log and calls ``write()``.
    """
    cfg_yaml = _system_yaml(tmp_path)
    log_path = _system_log_path(cfg_yaml)
    assert not log_path.exists(), "precondition: fresh tmp roots carry no system log"

    observer = TRITONSWMM_system(cfg_yaml, skip_log_update=True)

    assert observer.log.logfile == log_path
    assert not log_path.exists(), "observer construction must not CREATE the shared system log"


def test_default_construction_still_writes_the_system_log(tmp_path):
    """CONTROL — passes pre- and post-fix; fails an implementation that defaults the flag TRUE.

    Without this arm a change that made every construction read-only would green the two
    tests above while silently removing the orchestrator's provenance write.
    """
    cfg_yaml = _system_yaml(tmp_path)
    log_path = _system_log_path(cfg_yaml)
    assert not log_path.exists()

    TRITONSWMM_system(cfg_yaml)

    assert log_path.exists(), "default construction must still create/write the system log"


def test_observer_property_read_derives_fresh_and_persists_nothing(tmp_path):
    """RED pre-fix (TypeError). Post-fix: fresh in memory, nothing persisted.

    This is the arm that discriminates the INSUFFICIENT shape: suppressing only the
    constructor's ``_sync_compilation_status_on_init`` call leaves every property read free
    to call ``_sync_compilation_log_field`` -> ``set()`` -> ``write()``, and the on-disk
    document then changes, failing ``after == before``.
    """
    cfg_yaml = _system_yaml(tmp_path)
    probe = TRITONSWMM_system(cfg_yaml)
    log_path = Path(probe.log.logfile)

    build_log = probe.sys_paths.TRITON_build_dir_cpu / "compilation.log"
    build_log.parent.mkdir(parents=True, exist_ok=True)
    build_log.write_text(_TRITON_OK)

    # Force the persisted verdict to DISAGREE with what the build tree derives.
    payload = json.loads(log_path.read_text())
    payload["compilation_triton_cpu_successful"] = False
    log_path.write_text(json.dumps(payload, indent=2))
    before = (log_path.stat().st_mtime_ns, log_path.read_bytes())

    observer = TRITONSWMM_system(cfg_yaml, skip_log_update=True)
    derived = observer.compilation_triton_only_cpu_successful
    after = (log_path.stat().st_mtime_ns, log_path.read_bytes())

    assert derived is True, "the observer still derives the verdict freshly from the build tree"
    assert after == before, "an observer's property READ must persist nothing"
