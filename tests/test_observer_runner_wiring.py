"""Observer-runner wiring guards for ``skip_log_update`` (change 1028-2152).

PROBLEM. ``TRITONSWMM_system`` rewrites the SHARED ``{system_dir}/system_log.json``
every time it is constructed: ``_sync_compilation_status_on_init`` reads five compile
verdicts and each persists through ``_sync_compilation_log_field`` -> ``LogField.set``
-> ``log.write()``. With ~30 runners starting together on Rivanna WekaFS that
create-capable ``os.open`` intermittently raised ``FileNotFoundError`` and the runner
logged ``log write (open failed)`` (14 of 822 SLURM logs across the four 4a3r2
``synth_cc`` arms). ``TRITONSWMM_system(skip_log_update=True)`` is the shape change
1028-1828 landed so a process that only READS compile state abstains from that write;
its properties still derive their verdicts freshly from the build trees, so nothing a
caller reads changes.

WHY A SUBSTRING GUARD WOULD NOT DO (the alternative considered and rejected). The suite
already carries a file-TEXT guard -- ``test_log_concurrency.py::
test_log_single_writer_invariant`` asserts ``"skip_log_update=True" in
...read_text()``. That form cannot see this defect: ``process_timeseries_runner.py``
carries the string on its ``TRITONSWMM_analysis`` call while its ``TRITONSWMM_system``
call lacks the argument, so the substring guard returns TRUE on the defective tree.
These tests parse the AST and read the keyword's VALUE, which is the only form that
separates "the string appears somewhere in the file" from "THIS construction is an
observer".

VENUE: laptop-safe. No solver is compiled or executed, no synthetic-analysis fixture is
constructed, and every filesystem root is under ``tmp_path``. The system config is built
here from the schema's own required fields rather than from a ``TRITONSWMM_analysis``
fixture, because ``TRITONSWMM_analysis`` refuses a dirty checkout and a builder's
uncommitted edits are exactly that.

The system-config helper below duplicates ``_system_yaml`` from
``test_system_observer_construction.py`` rather than importing it. Deliberately NOT a
shared ``tests/_observer_helpers.py``: that would add a fourth path to a change whose
declared write set is two modules plus this file, and a test-side refactor is not this
change's job to land.
"""

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

import pytest
import yaml

import hhemt  # noqa: F401  (imported for the package location used by the runner modules)
import hhemt.system

_GUARDED_CONSTRUCTORS = ("TRITONSWMM_system", "TRITONSWMM_analysis")
_RUNNER_MODULES = (
    "process_timeseries_runner.py",
    "render_report_runner.py",
)

_CONSTRUCTION_CASES = (
    pytest.param(
        "hhemt.process_timeseries_runner",
        ["--event-iloc", "0", "--model-type", "tritonswmm"],
        id="process_timeseries_runner",
    ),
    pytest.param(
        "hhemt.render_report_runner",
        [],
        id="render_report_runner",
    ),
)


def _src_hhemt_dir() -> Path:
    """The ``src/hhemt`` tree THIS TEST FILE ships in.

    Resolved from ``__file__`` and never from ``hhemt.__file__``, because an editable
    install routinely points at a DIFFERENT checkout: a worktree grading its own sources
    resolves the ``hhemt`` package to the main tree, and a wiring guard that read the
    installed tree would grade the wrong files and report a confident pass.
    """
    return Path(__file__).resolve().parent.parent / "src" / "hhemt"


def _system_yaml(tmp_path: Path) -> Path:
    """A minimal schema-valid system config whose every root is under ``tmp_path``.

    The four existence-validated inputs are created as empty placeholders: construction
    reads none of them, and a real DEM/INP/template would make this test
    simulation-bearing for no gain. ``toggle_use_swmm_for_hydrology`` and the landuse
    path are off, so their conditional fields stay unrequired.
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


def _construction_calls(tree: ast.Module) -> list[tuple[str, ast.Call]]:
    """Every ``TRITONSWMM_system`` / ``TRITONSWMM_analysis`` call node in ``tree``.

    Both call shapes are matched by the callee's final name, so an aliased import
    (``from hhemt.system import TRITONSWMM_system as S``) is the one shape this misses;
    no runner module uses one, and the count control below would catch a silent
    population change.
    """
    found: list[tuple[str, ast.Call]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            name = getattr(func, "attr", None)
        else:
            name = None
        if name in _GUARDED_CONSTRUCTORS:
            found.append((name, node))
    return found


@pytest.mark.parametrize(("module_name", "extra_argv"), _CONSTRUCTION_CASES)
def test_observer_runner_construction_authors_no_shared_system_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, module_name: str, extra_argv: list[str]
) -> None:
    """Property: a runner's ``main()`` leaves the SHARED ``system_log.json`` uncreated.

    Input class: a runner module whose ``main()`` reaches its ``TRITONSWMM_system``
    construction and then fails on a present-but-unloadable analysis config. Both
    runners are covered because both reach the system construction before any analysis
    work, which is what makes the file's presence observable at all.

    Why it is required: the write this guards is a cross-process ``rename`` race on a
    shared filesystem. No assertion on one in-memory log object can see it; it has to
    be observed as the ABSENCE of the file after a real ``main()`` call.

    What kills it: omitting ``skip_log_update=True`` at the system construction, or
    passing ``False`` -- the constructor then creates the document before the analysis
    config is read.

    Second correct implementation: a ``hhemt.system`` factory that builds observer
    systems for runner call sites still calls the patched ``TRITONSWMM_system``, so
    both the single-construction count and the absence assertion hold.

    Deliberately NOT gated on ``/.singularity.d``'s existence. That sentinel is not the
    container guard's verdict: ``process_timeseries_runner``'s guard returns only when
    ``labels.json`` declares ``org.hhemt.toolkit_root`` AND the imported module sits
    outside that declared root, so an UNGOVERNED image only warns and a GOVERNED image
    whose module resolves inside its root neither warns nor returns -- both fall through
    to the construction, and a sentinel skip would suppress this test on exactly the
    environments that can run it. It also suppressed the ``render_report_runner``
    parameter, a module with no container guard at all (``grep -c singularity`` returns
    ``0`` there against ``2`` in ``process_timeseries_runner``, so the zero is a real
    absence). The count assertion below is what buys the protection the skip was aimed
    at: a guard that really did return leaves ``constructed`` empty and the assertion
    reds naming that cause, which is a loud, self-explaining failure rather than an
    unchallenged green.
    """
    cfg_yaml = _system_yaml(tmp_path)
    system_dir = Path(yaml.safe_load(cfg_yaml.read_text())["system_directory"])
    log_path = system_dir / "system_log.json"
    assert not log_path.exists(), "precondition: fresh tmp roots carry no system log"

    # The analysis config must EXIST: ``process_timeseries_runner`` validates the path
    # and returns 2 before constructing anything when it is absent, which would make
    # this test vacuous. ``{}`` parses as YAML and fails schema validation, so the
    # analysis construction raises AFTER the system construction has already happened.
    analysis_cfg = tmp_path / "analysis_config.yaml"
    analysis_cfg.write_text("{}\n")

    constructed: list[dict[str, object]] = []
    real_system = hhemt.system.TRITONSWMM_system

    def spy(*args: object, **kwargs: object) -> object:
        constructed.append(kwargs)
        return real_system(*args, **kwargs)

    monkeypatch.setattr(hhemt.system, "TRITONSWMM_system", spy)
    monkeypatch.setattr(
        sys,
        "argv",
        ["runner", "--system-config", str(cfg_yaml), "--analysis-config", str(analysis_cfg), *extra_argv],
    )

    runner = importlib.import_module(module_name)
    exit_code = runner.main()

    assert len(constructed) == 1, (
        "the runner must reach its system construction exactly once; a count above 1 means the seam "
        "moved. A count of 0 means an early return made this test vacuous -- under Apptainer, "
        "process_timeseries_runner's container-toolkit guard returns before the construction when "
        "labels.json declares org.hhemt.toolkit_root and the imported module sits outside it -- and "
        "it is reported as this red rather than skipped, so no environment is quietly exempt"
    )
    assert exit_code == 1, f"the unloadable analysis config must be the failure reported, got {exit_code}"
    assert not log_path.exists(), "an observer runner must not CREATE the shared system_log.json"


@pytest.mark.parametrize("module_name", _RUNNER_MODULES)
def test_observer_runner_passes_skip_log_update(module_name: str) -> None:
    """Property: EVERY system/analysis construction in the runner module is an observer.

    Input class: a runner module under ``src/hhemt`` that constructs both
    ``TRITONSWMM_system`` and ``TRITONSWMM_analysis``.

    Why it is required: the flag is a per-call keyword whose default is ``False``, so a
    construction site that simply forgets it compiles, runs, and writes the shared
    document. There is no type error, no runtime error and no log line -- the omission
    is legal at every layer.

    What kills it: deleting ``skip_log_update=True`` from a construction, or flipping it
    to ``False``.

    Second correct implementation: hoisting a construction into a module-local helper
    that carries the keyword -- the AST walk still finds the flagged call. Residual, and
    it is the over-constraint this arm accepts: a helper moved to a DIFFERENT module
    would not be found here.
    """
    src = _src_hhemt_dir() / module_name
    tree = ast.parse(src.read_text(encoding="utf-8"))

    sites = _construction_calls(tree)
    assert sites, f"{module_name} must construct at least one of {_GUARDED_CONSTRUCTORS}"

    offenders = []
    for ctor, node in sites:
        keywords = {kw.arg: kw.value for kw in node.keywords if kw.arg}
        value = keywords.get("skip_log_update")
        if not (isinstance(value, ast.Constant) and value.value is True):
            offenders.append(f"{ctor} at {module_name}:{node.lineno}")

    assert not offenders, (
        "observer constructions must pass skip_log_update=True, a literal True so the keyword's "
        f"VALUE is checked rather than its presence; offending sites: {offenders}"
    )


@pytest.mark.parametrize("module_name", _RUNNER_MODULES)
def test_observer_runner_construction_site_count_does_not_move(module_name: str) -> None:
    """CONTROL: the observer fix adds KEYWORD ARGUMENTS, never construction calls.

    Green on both the pre-change and the post-change tree by design -- that is what
    makes it a control rather than a regression arm. Its failure mode is an edit that
    "fixes" the shared write by adding a second construction (or a second analysis)
    beside the existing one instead of flagging the existing call, which would leave two
    writers where the file is read from one site.

    Why it is required: without it, ``test_observer_runner_passes_skip_log_update`` is
    satisfied by a duplicate flagged call sitting next to the unflagged original.
    """
    src = _src_hhemt_dir() / module_name
    tree = ast.parse(src.read_text(encoding="utf-8"))

    for ctor in _GUARDED_CONSTRUCTORS:
        count = sum(1 for name, _node in _construction_calls(tree) if name == ctor)
        assert count == 1, (
            f"{module_name} must construct {ctor} exactly once; found {count}. This is a CONTROL: "
            "the observer fix changes keyword arguments, so a moved count means the seam moved rather "
            "than that a caller was made read-only."
        )
