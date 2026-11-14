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

__all__ = [
    "golden_extractor",
    "golden_multisim_analysis",
    "golden_sensitivity_analysis",
    "golden_sensitivity_analysis_keeping_scenarios",
]

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
    add_swmm_only_system_stores(dest)
    add_hydrology_captures(dest)
    set_times_in_dependency_order(dest)
    return dest


def add_swmm_only_system_stores(root: Path) -> None:
    """Give every SWMM-only scenario of an extracted golden the system series store and summary reductions of D19.

    The committed golden predates the system series: its SWMM-only scenarios hold node and link stores and a node
    summary store with no ``system_*`` variable. Since chunk 1d.6 a processed SWMM-only scenario also holds
    ``SWMM_only_system_tseries.zarr`` (six series on the node store's own ``date_time`` axis) and the twelve ``_max``
    and ``_last`` reductions of those series in ``SWMM_only_node_summary.zarr``, and the consolidation refuses a
    scenario without them. This adds exactly those two things, with values that differ by event and by series, and
    changes nothing else; it stands in for the generator emitting them. A scenario that holds no SWMM-only node store
    (the sensitivity golden, which runs the coupled model alone) is left alone.
    """
    import numpy as np
    import xarray as xr

    from hhemt.constants import LST_SWMM_SYSTEM_SERIES
    from hhemt.utils import write_zarr

    # The units are written here as literals, not read from the toolkit's CF table, so the units test that stamps the
    # table onto this golden still compares the table with an independent statement.
    units = {
        "system_flooding_rate": "m3 s-1",
        "system_outfall_flow": "m3 s-1",
        "system_stored_volume": "m3",
        "system_lateral_inflow": "m3 s-1",
        "system_runoff": "m3 s-1",
        "system_rainfall": "mm h-1",
    }
    assert set(units) == set(LST_SWMM_SYSTEM_SERIES)

    for node_store in sorted(root.rglob("SWMM_only_node_tseries.zarr")):
        processed = node_store.parent
        summary_store = processed / "SWMM_only_node_summary.zarr"
        system_store = processed / "SWMM_only_system_tseries.zarr"
        if system_store.exists() or not summary_store.exists():
            continue
        times = xr.open_zarr(node_store, consolidated=False)["date_time"].values
        event_iloc = xr.open_zarr(summary_store, consolidated=False)["event_iloc"].values
        base = float(event_iloc[0]) + 1.0
        series = {
            name: (("date_time",), base * (k + 1) * np.linspace(0.0, 1.0, len(times)), {"units": units[name]})
            for k, name in enumerate(LST_SWMM_SYSTEM_SERIES)
        }
        write_zarr(xr.Dataset(series, coords={"date_time": times}), system_store, 5)
        reductions = {}
        for name, (_dims, values, attrs) in series.items():
            reductions[f"{name}_max"] = (("event_iloc",), np.array([values.max()]), attrs)
            reductions[f"{name}_last"] = (("event_iloc",), np.array([values[-1]]), attrs)
        xr.Dataset(reductions, coords={"event_iloc": event_iloc}).to_zarr(summary_store, mode="a")


def _hydro_inp_nodes(hydro_inp: Path) -> dict[str, tuple[float, float]]:
    """The non-outfall nodes of a ``hydro.inp`` with their coordinates, in ``[JUNCTIONS]`` order."""
    sections: dict[str, list[list[str]]] = {}
    current = None
    for line in hydro_inp.read_text().splitlines():
        if line.startswith("["):
            current = line.strip()
            sections[current] = []
        elif current is not None and line.strip() and not line.lstrip().startswith(";"):
            sections[current].append(line.split())
    coords = {row[0]: (float(row[1]), float(row[2])) for row in sections.get("[COORDINATES]", []) if len(row) >= 3}
    return {row[0]: coords[row[0]] for row in sections.get("[JUNCTIONS]", []) if row and row[0] in coords}


HYDROLOGY_CAPTURE_STEPS = 12  # the golden hydro.inp's 120 s window at its 10 s report step
HYDROLOGY_CAPTURE_STEP_S = 10


def add_hydrology_captures(root: Path) -> None:
    """Give every scenario of an extracted golden the per-node inflow capture its preparation step writes.

    The committed golden carries each scenario's ``swmm/hydro.inp`` and ``hydro.out`` but not
    ``processed/hydrology_inflow_summary.zarr``, and the per-scenario reclaim of the default policy removes
    ``hydro.out``; with neither store left the hydrology consolidation refuses the scenario, as it must. This writes
    the capture (``inflow`` on ``node_id`` and ``timestep``, built by the toolkit's own
    ``node_inflow_capture_dataset``) for each scenario whose hydrology model declares nodes: one value per node and
    report step, deterministic, differing by node and by event (the event is the scenario's position among the
    sorted scenario directories of its analysis). A scenario that already holds a capture is left alone, and a golden
    with no ``hydro.inp`` is untouched.
    """
    import numpy as np

    from hhemt.cf_conventions import apply_cf_attributes
    from hhemt.swmm_runoff_modeling import HydrologyNodeInflow, node_inflow_capture_dataset
    from hhemt.utils import write_zarr

    for sims in sorted(p for p in root.rglob("sims") if p.is_dir()):
        for event_iloc, scenario_dir in enumerate(sorted(p for p in sims.iterdir() if p.is_dir())):
            hydro_inp = scenario_dir / "swmm" / "hydro.inp"
            capture = scenario_dir / "processed" / "hydrology_inflow_summary.zarr"
            if not hydro_inp.exists() or capture.exists():
                continue
            nodes = _hydro_inp_nodes(hydro_inp)
            if not nodes:
                continue
            ramp = np.arange(1, HYDROLOGY_CAPTURE_STEPS + 1, dtype="float64") / HYDROLOGY_CAPTURE_STEPS
            flows = {n: 0.01 * (k + 1) * (event_iloc + 1) * ramp for k, n in enumerate(nodes)}
            node_inflow = HydrologyNodeInflow(
                flows,
                dict(nodes),
                np.arange(HYDROLOGY_CAPTURE_STEPS) * HYDROLOGY_CAPTURE_STEP_S / 3600.0,
                "CMS",
            )
            ds = node_inflow_capture_dataset(node_inflow, event_iloc)
            apply_cf_attributes(ds, "hydrology_inflow_summary")
            capture.parent.mkdir(parents=True, exist_ok=True)
            write_zarr(ds, capture, chunks=None, encoding={})


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


@pytest.fixture
def golden_sensitivity_analysis_keeping_scenarios(golden_extractor):
    """The sensitivity golden with the clearing switch off, so every scenario keeps its summaries and its directory.

    With the package defaults a consolidation clears each scenario's summary stores and removes its directory, and a
    process-stage reprocess of such a scenario is refused because no raw output remains. A test that deletes a summary
    and expects the process stage to rebuild it, or that re-processes a member, is about the keep-for-reprocessing
    state, so it names that state here instead of relying on the golden's silence."""
    system_dir = golden_extractor("sensitivity")
    cfg = system_dir / "analysis_config.yaml"
    cfg.write_text(cfg.read_text() + "toggle_clear_scenario_timeseries_after_consolidation: false\n")
    return construct_golden_analysis(system_dir)
