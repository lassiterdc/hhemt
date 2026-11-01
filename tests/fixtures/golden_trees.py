"""Fixtures that give a test a finished analysis tree without compiling or running a solver.

A golden is a zip of a finished analysis tree written by the independent generator (``tests/goldens/SPEC.md``).
Each fixture here is function-scoped: it extracts the zip into the test's own ``tmp_path``, so a test that
mutates the tree cannot affect another test. ``golden_extractor`` returns a callable that only extracts and
constructs nothing; the other two fixtures also construct an analysis object from the tree and return it.

Two things happen at extraction, because the zip cannot carry them:

* every ``@GOLDEN_ROOT@`` token in a text member is replaced by the extraction directory (SPEC section 5.2),
  which re-roots ``system_directory`` and every other absolute path in one rule, as the clone helper
  ``tests/_failing_fixture_helpers.py::prepare_clone_dir`` does for ``system_directory`` alone;
* every file, and every directory that holds a file, gets a modification time, because extraction discards the
  stored times (the zip stores a fixed 2000-01-01 time). Times are set in dependency order, one fixed second per
  tier, so a producer is always older than what is built from it.

No fixture of the ``*_compiled`` or ``*_completed`` family is used.
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

import pytest

__all__ = ["golden_extractor", "golden_multisim_analysis", "golden_sensitivity_analysis"]

GOLDEN_DIR = Path(__file__).resolve().parent / "goldens"
GOLDEN_ROOT_TOKEN = "@GOLDEN_ROOT@"
# Members whose text carries the root token (SPEC 5.2); every other member is copied byte for byte.
_TEXT_SUFFIXES = (
    ".yaml",
    ".yml",
    ".json",
    ".cfg",
    ".csv",
    ".txt",
    ".hyg",
    ".extbc",
    ".inp",
    ".rpt",
    ".dat",
    ".log",
    ".zattrs",
)
_TEXT_NAMES = ("zarr.json",)
BASE_TIME = 1_700_000_000  # an arbitrary fixed epoch second; only the ORDER of tiers matters
TIER_SECONDS = 1


def _is_text_member(name: str) -> bool:
    base = name.rsplit("/", 1)[-1]
    return name.endswith(_TEXT_SUFFIXES) or base in _TEXT_NAMES or name.endswith(".flag.json")


def dependency_tier(rel: str) -> int:
    """Rank of a golden member in dependency order: a lower tier is produced first and is older.

    Order: system inputs and configs, the software root, the setup flag, prepared scenario inputs, prepare
    flags, raw solver outputs, run flags, the summary layer, process flags, the consolidated tiers, then the
    consolidation flags. ``rel`` is the member's path relative to the extraction root, with ``/`` separators.
    The summary layer is the ``processed`` directory (tier 7) and an analysis ``log.json`` is tier 10, between the
    consolidated tiers (tier 9) and the consolidation flags (tier 11). Within the experiment root of a
    sensitivity golden (SPEC section 3) the member tiers follow the same rule.
    """
    parts = rel.split("/")
    base = parts[-1]
    if base.startswith("a_setup"):
        return 2
    if base.startswith("b_prepare"):
        return 4
    if base.startswith("c_run"):
        return 6
    if base.startswith("d_process"):
        return 8
    if base.startswith(("e_consolidate", "f_consolidate")):
        return 11
    if parts[0] == "_software":
        return 1
    if "processed" in parts:
        return 7
    if any(p.startswith("out_") for p in parts):
        return 5
    if "sims" in parts:
        return 3
    if any(p in ("analysis_datatree.zarr", "experiment_datatree.zarr") for p in parts):
        return 9
    if base == "log.json":
        return 10
    return 0


def extract_golden(kind: str, dest: Path) -> Path:
    """Extract ``{kind}.zip`` into ``dest``, re-root the token, set times in dependency order; return ``dest``.

    ``kind`` is ``"multisim"`` or ``"sensitivity"``. ``dest`` must not already exist (``FileExistsError``), and a
    member name that is absolute or contains ``..`` raises ``ValueError``."""
    zip_path = GOLDEN_DIR / f"{kind}.zip"
    dest.mkdir(parents=True, exist_ok=False)
    root = str(dest)
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = info.filename
            if name.startswith("/") or ".." in Path(name).parts:
                raise ValueError(f"unsafe member name {name!r} in {zip_path}")
            target = dest / name
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            data = zf.read(name)
            if _is_text_member(name):
                data = data.decode("utf-8").replace(GOLDEN_ROOT_TOKEN, root).encode("utf-8")
            target.write_bytes(data)
    set_times_in_dependency_order(dest)
    return dest


def set_times_in_dependency_order(root: Path) -> None:
    """Give every file under ``root`` a time of ``BASE_TIME + tier * TIER_SECONDS``.

    A directory that holds a file, at any depth, gets the time of the newest file below it, so no such directory
    looks newer than what it holds only because of its own creation. A directory that holds no file is left
    untouched. Files in one tier share a time."""
    newest: dict[Path, int] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            stamp = BASE_TIME + dependency_tier(path.relative_to(root).as_posix()) * TIER_SECONDS
            os.utime(path, (stamp, stamp))
            for parent in path.parents:
                if parent == root.parent:
                    break
                newest[parent] = max(newest.get(parent, 0), stamp)
    for directory, stamp in newest.items():
        os.utime(directory, (stamp, stamp))


GOLDEN_PLACEHOLDER_SHA = "0" * 40  # the sha the generator writes into every analysis config (SPEC 5.1)


def reroot_toolkit_sha(system_dir: Path) -> None:
    """Replace the placeholder ``hhemt_sha`` in every extracted YAML by the sha of the toolkit tree under test.

    The analysis constructor (``hhemt.analysis.TRITONSWMM_analysis.__init__``) refuses a config whose ``hhemt_sha``
    differs from the running toolkit (``hhemt.validation.running_identity``); the generator cannot know that sha, so
    it writes a placeholder and the fixture re-roots it, as ``tests/fixtures/test_case_builder.py`` does for the
    synthetic configs."""
    from hhemt.validation import running_identity

    sha = running_identity().sha
    for yaml_path in system_dir.rglob("*.yaml"):
        text = yaml_path.read_text()
        if GOLDEN_PLACEHOLDER_SHA in text:
            yaml_path.write_text(text.replace(GOLDEN_PLACEHOLDER_SHA, sha))


def construct_golden_analysis(system_dir: Path):
    """Construct system and analysis objects from the extracted configs; return the analysis.

    It first rewrites the placeholder sha in every extracted ``*.yaml`` under ``system_dir`` in place, and it
    expects ``system_config.yaml`` and ``analysis_config.yaml`` at the top of ``system_dir``."""
    from hhemt.analysis import TRITONSWMM_analysis
    from hhemt.system import TRITONSWMM_system

    reroot_toolkit_sha(system_dir)
    system = TRITONSWMM_system(system_dir / "system_config.yaml")
    return TRITONSWMM_analysis(system_dir / "analysis_config.yaml", system)


def dry_run_plan(analysis) -> dict[str, int]:
    """Plan the workflow with ``snakemake --dry-run`` (no job runs, no solver) and return ``{rule: job count}``.

    The ``total`` row of Snakemake's job-statistics table is dropped. A failed dry run raises ``AssertionError``
    carrying the run's message."""
    from hhemt.snakemake_dry_run_report import _parse_job_stats

    result = analysis.run(dry_run=True, verbose=False)
    assert result.success, result.message
    log = Path(analysis.analysis_paths.analysis_dir) / "logs" / "snakemake_master_dry_run.log"
    return {rule: n for rule, n in _parse_job_stats(log.read_text().splitlines()) if rule != "total"}


def export_status_csv(analysis) -> Path:
    """Write ``scenario_status.csv``, which a golden does not carry (SPEC 9), by hhemt's own exporter.

    Returns the path of the written file."""
    from hhemt.export_scenario_status import export_scenario_status_to_csv

    return export_scenario_status_to_csv(analysis)


@pytest.fixture
def golden_extractor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Callable ``extractor(kind) -> system_dir``: extract a golden into this test's ``tmp_path`` WITHOUT
    constructing an analysis, so a test can mutate the tree first (construction loads the logs into memory:
    ``hhemt.system.TRITONSWMM_system.__init__`` loads ``system_log.json`` and, unless ``skip_log_update`` is set,
    ``hhemt.analysis.TRITONSWMM_analysis.__init__`` loads the analysis ``log.json``).

    The analysis constructor (``hhemt.analysis.TRITONSWMM_analysis.__init__``) refuses a dirty toolkit checkout;
    that gate is about which toolkit runs, not about the golden, and a working tree under edit is dirty, so for the
    duration of the test the constructor sees the running identity with ``dirty`` cleared. The sha is unchanged.
    Calling ``extractor`` twice for one kind in one test raises ``FileExistsError``, because the second call finds
    the first call's directory."""
    import dataclasses

    import hhemt.analysis as analysis_module

    real = analysis_module.running_identity
    monkeypatch.setattr(analysis_module, "running_identity", lambda: dataclasses.replace(real(), dirty=False))

    def extractor(kind: str) -> Path:
        return extract_golden(kind, tmp_path / f"golden_{kind}")

    return extractor


@pytest.fixture
def golden_multisim_analysis(golden_extractor):
    """The multi-sim golden (3 events, the 3 model arms ``triton``, ``tritonswmm`` and ``swmm``; SPEC section 1)
    extracted into ``tmp_path``; the analysis object."""
    return construct_golden_analysis(golden_extractor("multisim"))


@pytest.fixture
def golden_sensitivity_analysis(golden_extractor):
    """The sensitivity golden (4 members, each a ``members/member_{member_id}/`` directory per SPEC section 3; 1 event;
    one model arm, the coupled ``tritonswmm``, SPEC section 1) extracted into ``tmp_path``; the analysis."""
    return construct_golden_analysis(golden_extractor("sensitivity"))
