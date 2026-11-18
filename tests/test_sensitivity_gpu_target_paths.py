"""A sensitivity compile target never carries a GPU backend its build paths were not frozen for.

The measured defect (a Rivanna A100 driver, change 1118-0015): the sensitivity driver
assigns ``gpu_hardware``, ``gpu_compilation_backend`` and ``additional_modules`` onto the
primary ``TRITONSWMM_system`` AFTER construction, but ``TRITONSWMM_system.__init__`` froze
``sys_paths`` (``TRITONSWMM_build_dir_gpu``, ``compilation_script_gpu``,
``compilation_logfile_gpu``) from the constructor values, which were ``None``. A target
built from that system then reached ``compile_TRITON_SWMM`` with a truthy backend and a
``None`` build directory and died at ``build_dir.parent`` with ``AttributeError``. The
driver now assigns the pair through ``TRITONSWMM_system.assign_gpu_target``, which
recomputes the GPU ``sys_paths`` with the pair. This file tests that method directly and
drives the reuse branch of ``_build_unique_system_targets`` through a stub driver.

The defect predicate, applied to EVERY target a builder returns: a target is defective iff
``target.system.sys_paths.TRITONSWMM_build_dir_gpu is None`` while
``target.system.gpu_compilation_backend is not None``.

VENUE: laptop-safe. No solver is compiled or executed and no synthetic-analysis fixture is
used. The orchestrator is a duck-typed ``SimpleNamespace`` (the pattern
``tests/test_sensitivity_system_target_reads_config.py`` establishes), the primary system is
a REAL ``TRITONSWMM_system`` because the frozen ``sys_paths`` are the thing under test, and
the HPC-system config is a stub exposing only ``partitions`` and ``additional_modules``,
the two attributes ``resolve_gpu_target`` and ``resolve_additional_modules`` read.

Module guarded by this file: ``hhemt.sensitivity_analysis`` target construction
(``_build_unique_system_targets``) and ``TRITONSWMM_system.assign_gpu_target``. The tests are
permanent regression tests and carry no stored blob or absolute count.
"""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import yaml

from hhemt.sensitivity_analysis import TRITONSWMM_sensitivity_analysis
from hhemt.system import TRITONSWMM_system

_GPU_PARTITION = "gpu"
_CPU_PARTITION = "cpu"


def _write_system_yaml(tmp_path: Path) -> Path:
    """Write a minimal schema-valid ``system_config`` whose every root is under ``tmp_path``.

    The existence-validated inputs are empty placeholders because nothing here reads them.
    """
    for rel in ("wshed.shp", "dem.tif", "swmm/model.inp"):
        fp = tmp_path / rel
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_bytes(b"")
    system_dir = tmp_path / "system"
    system_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "crs": {"horizontal_epsg": 3857, "vertical_epsg": 5703},
        "system_directory": str(system_dir),
        "watershed_gis_polygon": str(tmp_path / "wshed.shp"),
        "DEM_fullres": str(tmp_path / "dem.tif"),
        "TRITONSWMM_software_directory": str(tmp_path / "software"),
        "TRITONSWMM_git_URL": "https://example.invalid/triton.git",
        "SWMM_model": str(tmp_path / "swmm" / "model.inp"),
        "toggle_use_swmm_for_hydrology": False,
        "toggle_use_constant_mannings": True,
        "constant_mannings": 0.03,
        "toggle_triton_model": False,
        "toggle_tritonswmm_model": True,
        "toggle_swmm_model": False,
        "target_dem_resolution": 3.5,
    }
    dst = tmp_path / "system.yaml"
    dst.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return dst


def _stub_hpc_system():
    """An HPC-system stand-in with one GPU partition and one CPU-only partition."""
    return SimpleNamespace(
        partitions={
            _GPU_PARTITION: SimpleNamespace(gpu_hardware="a100", gpu_compilation_backend="CUDA"),
            _CPU_PARTITION: SimpleNamespace(gpu_hardware=None, gpu_compilation_backend=None),
        },
        additional_modules=["cuda"],
    )


def _driver_stub(tmp_path: Path, ensemble_partition: str):
    """A duck-typed sensitivity driver whose primary system is real and CPU-shaped at construction.

    The primary system is constructed with no GPU pair, exactly as the analysis does before
    the sensitivity driver injects one, so its ``sys_paths`` are frozen CPU-only.
    """
    system_yaml = _write_system_yaml(tmp_path)
    primary = TRITONSWMM_system(system_yaml, skip_log_update=True, skip_layout_state=True)
    return SimpleNamespace(
        _system=primary,
        experiment=SimpleNamespace(
            cfg_analysis=SimpleNamespace(
                sensitivity_analysis=tmp_path / "sensitivity.csv",
                hpc_ensemble_partition=ensemble_partition,
                execution_environment="native",
            ),
            analysis_paths=SimpleNamespace(analysis_dir=tmp_path / "analysis_dir"),
            cfg_hpc_system=_stub_hpc_system(),
        ),
    ), system_yaml


def _is_defective(system) -> bool:
    """The defect predicate: a backend is set but the frozen GPU build directory is None."""
    return system.sys_paths.TRITONSWMM_build_dir_gpu is None and system.gpu_compilation_backend is not None


_GPU_PATH_FIELDS = (
    "TRITONSWMM_build_dir_gpu",
    "TRITON_build_dir_gpu",
    "compilation_script_gpu",
    "compilation_logfile_gpu",
)


def _gpu_paths(system) -> dict:
    return {name: getattr(system.sys_paths, name) for name in _GPU_PATH_FIELDS}


def test_assign_gpu_target_recomputes_gpu_paths_like_a_constructed_system(tmp_path):
    """After ``assign_gpu_target`` the GPU ``sys_paths`` equal those of a system constructed with the pair.

    Property: for an A100 CUDA pair, every GPU entry is not None and equals the same entry of
    a ``TRITONSWMM_system`` constructed with that pair.

    Class: NEW CAPABILITY. The plausible wrong implementation is plain attribute assignment
    (the pre-change driver), under which the four entries stay None; red under it.

    Why this is required: ``compile_TRITON_SWMM`` branches on the backend and then reads
    these paths, so an entry left None raises ``AttributeError`` at ``build_dir.parent``.

    What kills it: assigning the pair without recomputing ``sys_paths``, or recomputing only
    ``TRITONSWMM_build_dir_gpu``.

    Second correct implementation under which it still passes: replacing ``self.sys_paths``
    with a freshly built ``SysPaths`` instead of setting the four attributes.
    """
    system_yaml = _write_system_yaml(tmp_path)
    system = TRITONSWMM_system(system_yaml, skip_log_update=True, skip_layout_state=True)
    reference = TRITONSWMM_system(
        system_yaml,
        gpu_hardware="a100",
        gpu_compilation_backend="CUDA",
        additional_modules="cuda",
        skip_log_update=True,
        skip_layout_state=True,
    )
    assert all(v is None for v in _gpu_paths(system).values()), "control: constructed without a pair"

    system.assign_gpu_target("a100", "CUDA", "cuda")

    assert all(v is not None for v in _gpu_paths(system).values())
    assert _gpu_paths(system) == _gpu_paths(reference)
    assert (system.gpu_hardware, system.gpu_compilation_backend, system.additional_modules) == ("a100", "CUDA", "cuda")


def test_assign_gpu_target_with_no_pair_is_a_noop_on_a_cpu_system(tmp_path):
    """Assigning ``(None, None, None)`` to a CPU system leaves every GPU entry None and changes no other path.

    Property: the CPU-only call is a no-op on ``sys_paths``.

    Class: REGRESSION on behaviour preservation; green before and after (held).

    Why this is required: CPU analyses pass ``(None, None)`` from ``resolve_gpu_target`` and
    must stay byte-identical.

    What kills it: a method that derives a GPU path unconditionally.

    Second correct implementation under which it still passes: one that returns early when
    the backend is None.
    """
    system = TRITONSWMM_system(_write_system_yaml(tmp_path), skip_log_update=True, skip_layout_state=True)
    before = system.sys_paths.as_dict()

    system.assign_gpu_target(None, None, None)

    assert system.sys_paths.as_dict() == before
    assert all(v is None for v in _gpu_paths(system).values())


def test_unique_target_builder_reuse_branch_gpu_target_carries_gpu_build_paths(tmp_path):
    """Every target ``_build_unique_system_targets`` builds for a GPU partition has a GPU build directory.

    Property: for a row whose config equals the master's and whose partition resolves to the
    primary GPU pair, the returned target satisfies the negation of the defect predicate.

    Class: REGRESSION. Red on the ``origin/develop`` ``sensitivity_analysis.py`` (the reuse
    branch hands out the primary system whose paths were frozen with None GPU entries).

    Why this is required: this is the driver site the scope names; the target feeds
    ``compile_TRITON_SWMM``.

    What kills it: assigning the pair onto the primary by plain attribute assignment.

    Second correct implementation under which it still passes: a builder that never reuses
    the primary system and constructs one system per target.
    """
    stub, system_yaml = _driver_stub(tmp_path, _GPU_PARTITION)
    df = pd.DataFrame({"system_config_yaml": [str(system_yaml)]}, index=pd.Index(["m0"]))

    targets = TRITONSWMM_sensitivity_analysis._build_unique_system_targets(stub, df, is_main_orchestrator=True)

    assert targets, "the builder returns at least one target"
    assert all(t.system.gpu_compilation_backend == "CUDA" for t in targets), "the target carries the GPU backend"
    defective = [t.target_id for t in targets if _is_defective(t.system)]
    assert defective == [], f"GPU targets with a None GPU build dir: {defective}"


def test_cpu_only_unique_target_builder_still_reuses_the_primary_system(tmp_path):
    """A CPU-only row on ``_build_unique_system_targets`` keeps reusing the primary system.

    Property: for a config-equal row whose partition resolves to no GPU backend, the target's
    system IS ``self._system``.

    Class: REGRESSION on behaviour preservation (green before and after, held).

    Why this is required: the CPU sweep reuses the primary system.

    What kills it: a builder that always constructs a fresh system.

    Second correct implementation under which it still passes: a reuse test keyed on a
    config hash rather than ``model_dump_json``.
    """
    stub, system_yaml = _driver_stub(tmp_path, _CPU_PARTITION)
    df = pd.DataFrame({"system_config_yaml": [str(system_yaml)]}, index=pd.Index(["m0"]))

    targets = TRITONSWMM_sensitivity_analysis._build_unique_system_targets(stub, df, is_main_orchestrator=True)

    assert len(targets) == 1
    assert targets[0].system is stub._system, "a backend-None row reuses the primary system"


def test_runner_keeps_its_constructed_pair(tmp_path):
    """With ``is_main_orchestrator=False`` the builder never assigns a pair onto ``self._system``.

    Property: a setup_target runner's primary keeps its constructed (None) pair and paths.

    Class: REGRESSION on behaviour preservation (green before and after, held).

    Why this is required: assigning the primary ensemble pair onto a runner's CPU target
    flips it into the GPU branch with None paths (the comment at the injection site).

    What kills it: dropping the ``is_main_orchestrator`` gate at the injection site.

    Second correct implementation under which it still passes: a gate placed inside
    ``assign_gpu_target`` callers via a runner-specific builder.
    """
    stub, system_yaml = _driver_stub(tmp_path, _GPU_PARTITION)
    df = pd.DataFrame({"system_config_yaml": [str(system_yaml)]}, index=pd.Index(["m0"]))
    # A runner does not write ``_generated/``; the driver already did, so stage the file it reads.
    generated = tmp_path / "analysis_dir" / "_generated"
    generated.mkdir(parents=True)
    (generated / "target_0.yaml").write_text(system_yaml.read_text())

    TRITONSWMM_sensitivity_analysis._build_unique_system_targets(stub, df, is_main_orchestrator=False)

    assert stub._system.gpu_compilation_backend is None
    assert all(v is None for v in _gpu_paths(stub._system).values())
