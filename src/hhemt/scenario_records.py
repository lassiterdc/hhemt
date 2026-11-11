"""Analysis-level records of scenarios and sensitivity members whose directories were removed.

Problem this file solves. Under the default configuration the consolidated ``analysis_datatree.zarr`` is the one
durable product, so after every store a scenario wrote has been read back from the tree the whole
``sims/{event_id}/`` directory is deleted (approved shape D24 (e)). Everything the directory carried that a later
reader still needs (the model logs, the completion verdict, the last disk-use sentinel, the parsed ``log.out``
fields) moves first into one JSON record at ``{analysis_dir}/_status/scenario_records/{event_id}.json``, and every
reader of those facts goes through ``record_for`` so it answers from the record once the directory is gone.

Write order, applied to every deletion: the record is written in state ``pending`` BEFORE the directory is
deleted and set to ``done`` AFTER. A resume that finds ``pending`` never reads the directory as a source; it
finishes the delete and sets ``done``. A run-stage force of a removed event sets the record to ``superseded``,
which makes the event plan as never run (D24 (g), (h)).

PATH-ONLY and a leaf: only stdlib is imported at module level. ``TRITONSWMM_scenario``'s constructor creates
directories, so nothing here may build one. Shape rows implemented: D24 (e), (f), (g), (h), D22, and the
Q-w Form 3 member view (``RecordedMemberView``).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Record states. ``pending`` and ``done`` both mean the directory is (being) removed and its facts live in the
#: record; ``superseded`` means a run-stage force re-opened the event, so the event plans as never run.
RECORD_STATES = ("pending", "done", "superseded")
REMOVED_STATES = ("pending", "done")

#: Directory (under the analysis directory) holding one record per removed scenario.
SCENARIO_RECORDS_SUBDIR = Path("_status") / "scenario_records"
#: Directory (under a sensitivity master) holding, per removed member, its moved scenario records, analysis log
#: and config YAML.
MEMBER_RECORDS_SUBDIR = Path("_status") / "member_records"
#: The root attribute-free marker file of a member whose whole directory was removed after the master copied it.
MEMBER_RECORD_FILENAME = "member_record.json"

#: ``remove_after_processing`` classes that "all" elects (config.analysis.RemoveValue). A list electing every one
#: of them is equivalent to "all" for the removal conjunction.
ALL_REMOVE_CLASSES = frozenset(
    {"timeseries", "raw_swmm_binaries", "coupled_rpt", "hydro_out", "prep_inputs", "hydrographs", "standalone_rpt"}
)
ALL_MODELS = ("tritonswmm", "triton", "swmm")


class ScenarioRemovedError(RuntimeError):
    """A directory-reading or directory-writing method was called on the handle of a removed scenario.

    The handle of a recorded event keeps ``scen_paths`` (path arithmetic) and nothing else, so a loop over
    ``df_sims`` re-creates no directory. The message names the event so the caller can tell which one.
    """

    def __init__(self, event_id: str, what: str = "") -> None:
        self.event_id = event_id
        detail = f" ({what})" if what else ""
        super().__init__(
            f"scenario {event_id!r} was removed after its results were verified into the analysis tree{detail}; "
            "its directory no longer exists. Re-run the event from scratch (override_force_rerun at the run "
            "stage), or use a new analysis with one of the clearing, clear_raw or remove_after_processing "
            "switches off."
        )


class MemberRemovedError(RuntimeError):
    """An attribute a recorded (removed) sensitivity member's read-only view does not provide was read.

    Raised loudly, naming the member and the attribute, so a gap in the view is a build defect at generation
    rather than a silent zero inside a function that swallows ``Exception``.
    """

    def __init__(self, member_id: Any, attribute: str = "") -> None:
        self.member_id = member_id
        self.attribute = attribute
        what = f" attribute {attribute!r}" if attribute else ""
        super().__init__(
            f"sensitivity member {member_id!r} was removed after the master copied its tree; its read-only "
            f"view does not provide{what}. A removed member is never constructed again as an analysis."
        )


class _RemovedScenarioHandle:
    """Placeholder bound to every directory-backed component of a removed scenario's handle.

    Any attribute read raises ``ScenarioRemovedError`` naming the event, so a method that would read or write the
    removed directory fails loudly instead of re-creating it.
    """

    def __init__(self, event_id: str) -> None:
        object.__setattr__(self, "_event_id", event_id)

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        raise ScenarioRemovedError(self._event_id, name)


# ---------------------------------------------------------------------------
# Paths and raw record I/O (stdlib json only).
# ---------------------------------------------------------------------------
def scenario_records_dir(analysis_dir: Path) -> Path:
    """The directory holding one ``{event_id}.json`` record per removed scenario of ``analysis_dir``."""
    return Path(analysis_dir) / SCENARIO_RECORDS_SUBDIR


def scenario_record_path(analysis_dir: Path, event_id: str) -> Path:
    """The record file of ``event_id`` under ``analysis_dir`` (it need not exist)."""
    return scenario_records_dir(analysis_dir) / f"{event_id}.json"


def member_records_dir(master_dir: Path, member_id: Any) -> Path:
    """Where a sensitivity master keeps the moved records of removed member ``member_id``."""
    return Path(master_dir) / MEMBER_RECORDS_SUBDIR / f"member_{member_id}"


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write ``payload`` to ``path`` through a temp file and rename, so a kill never leaves a half-written record.

    A truncated record would read as malformed, which ``read_scenario_record`` maps to "no record", and "no
    record" next to a missing directory is the one state this module must never invent.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True, default=str)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def read_scenario_record(analysis_dir: Path, event_id: str) -> dict | None:
    """The record of ``event_id`` when it exists and is well formed (a known ``state``), else None.

    A malformed record reads as absent so a hand-edited or truncated file never makes a scenario look removed.
    """
    try:
        payload = json.loads(scenario_record_path(analysis_dir, event_id).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("state") not in RECORD_STATES:
        return None
    return payload


def write_scenario_record(analysis_dir: Path, event_id: str, record: dict) -> None:
    """Persist ``record`` (atomically) as the record of ``event_id``; ``record['state']`` must be a known state."""
    if record.get("state") not in RECORD_STATES:
        raise ValueError(f"scenario record state must be one of {RECORD_STATES}, got {record.get('state')!r}")
    _atomic_write_json(scenario_record_path(analysis_dir, event_id), record)


def record_state(analysis_dir: Path, event_id: str) -> str | None:
    """``pending``, ``done``, ``superseded`` or None (no record)."""
    rec = read_scenario_record(analysis_dir, event_id)
    return None if rec is None else rec["state"]


def event_is_removed(analysis_dir: Path, event_id: str) -> bool:
    """True iff the event's record is ``pending`` or ``done``: its facts are in the record, not a directory.

    ``pending`` counts as removed on purpose. A generator that read it as present would have to finish the
    removal itself, and generation must perform no filesystem act (D24 (g)); the removal is finished by the
    first act of rule ``consolidate``.
    """
    return record_state(analysis_dir, event_id) in REMOVED_STATES


def recorded_event_ids(analysis_dir: Path) -> list[str]:
    """Event ids with a well-formed record of any state, sorted. Empty when there are none."""
    d = scenario_records_dir(analysis_dir)
    if not d.is_dir():
        return []
    return sorted(p.stem for p in d.glob("*.json") if read_scenario_record(analysis_dir, p.stem) is not None)


# ---------------------------------------------------------------------------
# The resolver.
# ---------------------------------------------------------------------------
def _sim_folder(analysis, event_id: str) -> Path:
    return Path(analysis.analysis_paths.simulation_directory) / event_id


def _read_json_file(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def record_for(analysis, event_id: str, model: str) -> dict | None:
    """The payload of ``log_{model}.json`` for ``event_id``: the live log, else the record's copy.


    Returns the parsed JSON mapping (what ``json.loads`` gives for the log file), or None when neither source
    has one. Path-only: no scenario is constructed and nothing is created. A record in state ``pending`` or
    ``done`` is the source even when the directory still exists, because a ``pending`` resume must never read
    the directory again; a ``superseded`` or absent record leaves the live log as the source.
    """
    analysis_dir = Path(analysis.analysis_paths.analysis_dir)
    rec = read_scenario_record(analysis_dir, event_id)
    if rec is not None and rec["state"] in REMOVED_STATES:
        logs = rec.get("model_logs") or {}
        payload = logs.get(model)
        return payload if isinstance(payload, dict) else None
    return _read_json_file(_sim_folder(analysis, event_id) / f"log_{model}.json")


def record_field(analysis, event_id: str, name: str, default: Any = None) -> Any:
    """A top-level key of a removed event's record (for example ``run_completed``, ``du_bytes_before_removal``)."""
    rec = read_scenario_record(Path(analysis.analysis_paths.analysis_dir), event_id)
    if rec is None or rec["state"] not in REMOVED_STATES:
        return default
    return rec.get(name, default)


def recorded_run_completed(analysis, event_id: str, model: str) -> bool | None:
    """The run-completion verdict captured at removal for ``model``, or None when the event has no removed record."""
    rec = read_scenario_record(Path(analysis.analysis_paths.analysis_dir), event_id)
    if rec is None or rec["state"] not in REMOVED_STATES:
        return None
    verdicts = rec.get("run_completed") or {}
    return bool(verdicts.get(model, False))


# ---------------------------------------------------------------------------
# Building the record and removing the directory (D24 (e), D22's order).
# ---------------------------------------------------------------------------
def _listing(sim_folder: Path) -> list[str]:
    """Every file under ``sim_folder`` as paths relative to it (unit: files), sorted."""
    out: list[str] = []
    for root, _dirs, files in os.walk(sim_folder):
        for name in files:
            out.append(str((Path(root) / name).relative_to(sim_folder)))
    return sorted(out)


def build_scenario_record(
    sim_folder: Path,
    event_id: str,
    *,
    state: str = "pending",
    run_completed: dict[str, bool] | None = None,
    captured: dict | None = None,
) -> dict:
    """Assemble the record of one scenario from its directory, before the directory is deleted.

    Carries: the full content of every ``log_{model}.json`` (``model_logs``), the scenario-preparation log
    (``scenario_log``), the last ``_status/_du.json`` payload (``du_json``) with its byte figure
    (``du_bytes_before_removal``), the list of what the directory held (``listing``, unit: files),
    ``run_completed`` per model as captured by the caller (the run-completion gate reads raw solver files that
    no longer exist after removal), and ``captured``, a free mapping for the facts only the caller can parse
    (the ``log.out`` fields and GPU-binding artifact of ``df_status``, the SWMM-only ``Total elapsed time``).
    """
    sim_folder = Path(sim_folder)
    model_logs: dict[str, dict] = {}
    for p in sorted(sim_folder.glob("log_*.json")):
        payload = _read_json_file(p)
        if payload is not None:
            model_logs[p.stem[len("log_") :]] = payload
    du = _read_json_file(sim_folder / "_status" / "_du.json")
    return {
        "event_id": event_id,
        "state": state,
        "model_logs": model_logs,
        "scenario_log": _read_json_file(sim_folder / "scenario_prep_log.json"),
        "du_json": du,
        "du_bytes_before_removal": int((du or {}).get("disk_utilization_bytes", 0)),
        "listing": _listing(sim_folder) if sim_folder.is_dir() else [],
        "run_completed": dict(run_completed or {}),
        "captured": dict(captured or {}),
    }


def df_status_rows_from_record(
    record: dict, weather_row: dict, event_iloc: Any, enabled_models: list[str], scenario_directory: str
) -> list[dict]:
    """The ``df_status`` rows of one removed event, one per enabled model, built from its record only.

    Rows captured before removal (``captured["df_status_rows"]``) win; otherwise a minimal row carries the recorded
    completion verdict and the recorded disk-use bytes, so ``scenario_status.csv`` never has a blank row for a
    removed scenario.
    """
    stored = (record.get("captured") or {}).get("df_status_rows")
    if stored:
        return [{**weather_row, **row, "event_iloc": event_iloc} for row in stored]
    verdicts = record.get("run_completed") or {}
    return [
        {
            **weather_row,
            "event_iloc": event_iloc,
            "model_type": model,
            "scenario_setup": True,
            "run_completed": bool(verdicts.get(model, False)),
            "disk_utilization_bytes": record.get("du_bytes_before_removal"),
            "scenario_directory": scenario_directory,
        }
        for model in enabled_models
    ]


#: Subdirectory of ``_status/scenario_records`` holding the kept copies of the prepared SWMM inputs the system-overview
#: panels read: one file per kind, captured from the first scenario removed. A few megabytes in total, so the
#: copy is a file beside the records and not text inside a record.
KEPT_INP_SUBDIR = "inp"
_INP_KINDS = {"full": "swmm_full_inp", "hydro": "swmm_hydro_inp", "hydraulics": "swmm_hydraulics_inp"}


def kept_inp_paths(base_dir: Path) -> dict[str, Path]:
    """The kept INP copies under ``base_dir`` (an analysis directory, or a master's ``member_{id}`` records directory).

    Looks in ``{base}/_status/scenario_records/inp`` and ``{base}/scenario_records/inp``; kinds absent are omitted.
    """
    out: dict[str, Path] = {}
    for d in (
        Path(base_dir) / SCENARIO_RECORDS_SUBDIR / KEPT_INP_SUBDIR,
        Path(base_dir) / "scenario_records" / KEPT_INP_SUBDIR,
    ):
        for kind in _INP_KINDS:
            f = d / f"{kind}.inp"
            if kind not in out and f.is_file():
                out[kind] = f
    return out


def capture_system_overview_inps(sim_folder: Path, analysis_dir: Path) -> dict[str, str]:
    """Copy the scenario's prepared INPs into the kept directory once (first removed wins); return the kinds kept.

    Read by the system-overview renderer after every scenario directory is gone. Existing copies are never
    overwritten, so the panel reads one consistent scenario. Returns ``{kind: file name}`` for the record's
    ``captured["system_overview_inp"]``.
    """
    import shutil

    dest = scenario_records_dir(analysis_dir) / KEPT_INP_SUBDIR
    kept: dict[str, str] = {}
    for kind in _INP_KINDS:
        target = dest / f"{kind}.inp"
        src = (
            Path(sim_folder) / "swmm" / {"full": "full.inp", "hydro": "hydro.inp", "hydraulics": "hydraulics.inp"}[kind]
        )
        if target.is_file():
            kept[kind] = target.name
        elif src.is_file():
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
            kept[kind] = target.name
    return kept


def finish_scenario_removal(analysis_dir: Path, event_id: str, sim_folder: Path) -> int:
    """Delete what is left of ``sim_folder`` and set the record ``done``; returns bytes removed.

    Used by the resume of a ``pending`` record and as the second half of ``remove_scenario_directory``. The
    directory is NEVER read as a source here: a partly deleted directory is only deleted further. A missing
    path is not an error (``delete_and_account`` skips absent paths).
    """
    from hhemt.du_sentinels import delete_and_account

    analysis_dir = Path(analysis_dir)
    rec = read_scenario_record(analysis_dir, event_id)
    if rec is None or rec["state"] != "pending":
        raise ValueError(
            f"finish_scenario_removal needs a pending record for {event_id!r}, found {rec and rec['state']!r}"
        )
    freed = delete_and_account([sim_folder], scope_dir=analysis_dir, scope="analysis")
    rec["state"] = "done"
    write_scenario_record(analysis_dir, event_id, rec)
    return freed


def remove_scenario_directory(
    analysis_dir: Path,
    event_id: str,
    sim_folder: Path,
    *,
    run_completed: dict[str, bool] | None = None,
    captured: dict | None = None,
) -> str:
    """Remove ``sim_folder`` recorded-before-delete: record ``pending``, delete, record ``done``.

    Returns the final record state (``done``). A record already ``done`` is left alone. A record ``pending`` is
    a resume: the directory is not read, only finished. A ``superseded`` record is replaced by a fresh
    ``pending`` one built from the new directory. The CALLER owns the precondition (every model completed, every
    store cleared under a ``done`` record, no failure) because those facts live in the processing step; this
    function owns the order.
    """
    analysis_dir = Path(analysis_dir)
    existing = read_scenario_record(analysis_dir, event_id)
    if existing is not None and existing["state"] == "done":
        return "done"
    if existing is not None and existing["state"] == "pending":
        finish_scenario_removal(analysis_dir, event_id, sim_folder)
        return "done"
    rec = build_scenario_record(sim_folder, event_id, state="pending", run_completed=run_completed, captured=captured)
    write_scenario_record(analysis_dir, event_id, rec)
    finish_scenario_removal(analysis_dir, event_id, sim_folder)
    return "done"


def supersede_scenario_record(analysis_dir: Path, event_id: str) -> bool:
    """Set a ``done`` or ``pending`` record to ``superseded`` (a run-stage force re-opens the event).

    Returns True when a record was changed. The caller deletes the event's ``b_prepare_*``, ``c_run_*``,
    ``d_process_*`` and ``f_consolidate_scenario_*`` flags and the analysis ``e_consolidate_*`` flag FIRST: a
    kill between the two leaves the record ``done`` and a force to re-issue, whereas the opposite order leaves
    ``superseded`` with every flag present and no directory, which plans nothing (D24 (g) table).
    """
    rec = read_scenario_record(analysis_dir, event_id)
    if rec is None or rec["state"] not in REMOVED_STATES:
        return False
    rec["state"] = "superseded"
    write_scenario_record(analysis_dir, event_id, rec)
    return True


# ---------------------------------------------------------------------------
# The three-switch conjunction and the per-scenario removal precondition.
# ---------------------------------------------------------------------------
def removal_switches_on(cfg_analysis, enabled_models: list[str] | tuple[str, ...]) -> bool:
    """True iff the clearing switch is on, ``clear_raw`` covers every enabled model and every class is elected.

    This conjunction is the only key of the directory removal (D24 (a)); no new field exists. Any one switch off
    keeps the directories and every class not elected, as before.
    """
    if not bool(getattr(cfg_analysis, "toggle_clear_scenario_timeseries_after_consolidation", False)):
        return False
    clear_raw = getattr(cfg_analysis, "clear_raw", "none")
    if clear_raw == "none":
        return False
    if clear_raw != "all" and not set(enabled_models) <= set(clear_raw):
        return False
    remove = getattr(cfg_analysis, "remove_after_processing", "none")
    if remove == "all":
        return True
    if remove == "none":
        return False
    return ALL_REMOVE_CLASSES <= set(remove)


def scenario_stores_cleared(sim_folder: Path, enabled_models: list[str] | tuple[str, ...]) -> bool:
    """True iff every enabled model's log records every per-scenario store it holds as cleared and done.

    Path-only over ``log_{model}.json``: the summary-clearing record must be state ``done`` and every
    ``consolidated_*_cleared`` boolean the log carries must be True (a None field is a store the model never
    wrote, which is cleared by absence). A model with no readable log is NOT cleared, so a scenario whose
    processing has not recorded anything keeps its directory.
    """
    for model in enabled_models:
        log = _read_json_file(Path(sim_folder) / f"log_{model}.json")
        if log is None:
            return False
        summaries = log.get("consolidated_summaries_cleared")
        if not isinstance(summaries, dict) or summaries.get("state") != "done":
            return False
        for key, value in log.items():
            if key.startswith("consolidated_") and key.endswith("_timeseries_cleared") and value is not None:
                if value is not True:
                    return False
    return True


# ---------------------------------------------------------------------------
# Sensitivity members (Q-w Form 3): a removed member is never constructed; a read-only view answers instead.
# ---------------------------------------------------------------------------
def member_record_path(master_dir: Path, member_id: Any) -> Path:
    """The marker record of a removed member: ``{master}/_status/member_records/member_{id}/member_record.json``."""
    return member_records_dir(master_dir, member_id) / MEMBER_RECORD_FILENAME


def read_member_record(master_dir: Path, member_id: Any) -> dict | None:
    """The member's marker record when well formed (a known state), else None."""
    payload = _read_json_file(member_record_path(master_dir, member_id))
    if payload is None or payload.get("state") not in RECORD_STATES:
        return None
    return payload


def write_member_record(master_dir: Path, member_id: Any, record: dict) -> None:
    """Persist a member marker record atomically."""
    if record.get("state") not in RECORD_STATES:
        raise ValueError(f"member record state must be one of {RECORD_STATES}, got {record.get('state')!r}")
    _atomic_write_json(member_record_path(master_dir, member_id), record)


def recorded_member_event_done(master_dir: Path, member_id: Any, event_id: str) -> bool:
    """True iff the removed member's moved record of ``event_id`` is well formed and ``done``. Path-only."""
    payload = _read_json_file(member_records_dir(master_dir, member_id) / "scenario_records" / f"{event_id}.json")
    return payload is not None and payload.get("state") == "done"


def member_is_recorded(master_dir: Path, member_id: Any) -> bool:
    """True iff the member's record is ``done``: only a ``done`` member is never constructed again (D24 (e))."""
    rec = read_member_record(master_dir, member_id)
    return rec is not None and rec["state"] == "done"


def move_member_records(
    member_dir: Path,
    master_dir: Path,
    member_id: Any,
    *,
    analysis_id: str,
    yaml_name: str,
    analysis_log: Path | None = None,
) -> None:
    """Copy the member's scenario records, analysis log and config YAML into the master BEFORE its directory goes.

    Copy-then-leave: the source stays until the whole-directory delete, so a kill here re-runs the copy. The
    marker record is written last, in state ``pending``; the caller sets it ``done`` after the directory delete.
    """
    import shutil

    member_dir = Path(member_dir)
    dest = member_records_dir(master_dir, member_id)
    dest.mkdir(parents=True, exist_ok=True)
    src_records = member_dir / SCENARIO_RECORDS_SUBDIR
    if src_records.is_dir():
        shutil.copytree(src_records, dest / "scenario_records", dirs_exist_ok=True)
    yaml_file = member_dir / yaml_name
    if yaml_file.is_file():
        shutil.copy2(yaml_file, dest / yaml_name)
    if analysis_log is not None and Path(analysis_log).is_file():
        shutil.copy2(analysis_log, dest / "analysis_log.json")
    write_member_record(
        master_dir,
        member_id,
        {"member_id": member_id, "analysis_id": analysis_id, "state": "pending", "yaml_name": yaml_name},
    )


class RecordedMemberView:
    """A read-only stand-in for a removed sensitivity member, built from its records and the master's tree.

    A removed member is never constructed as an analysis, so the view provides exactly the attributes the
    whole-experiment sites read (derived by the census test in ``tests/test_scenario_directory_removal.py``)
    and nothing else: any other attribute raises ``MemberRemovedError`` naming the member. The attributes
    the master supplies at construction (``df_sims``, ``cfg_analysis``, ``cfg_hpc_system``, the weather indexer
    function and the per-event completion reader) are passed in as plain values or callables, so this class
    imports neither the analysis nor the config packages.
    """

    _PROVIDED = frozenset(
        {
            "member_id",
            "analysis_id",
            "master_dir",
            "df_sims",
            "cfg_analysis",
            "cfg_hpc_system",
            "df_status",
            "_get_enabled_model_types",
            "_retrieve_weather_indexer_using_integer_index",
            "_retrieve_sim_runs",
            "_retrieve_sim_run_processing_object",
            "is_recorded_member",
            "n_scenarios",
            "_all_scenarios_created",
            "_scenarios_not_created",
            "_all_sims_run",
            "_scenarios_not_run",
        }
    )

    @property
    def n_scenarios(self) -> int:
        """Number of events of the member (its events table length)."""
        return len(self.df_sims)

    @property
    def _all_scenarios_created(self) -> bool:
        """True: a removed member's scenarios were all created and processed before removal (record ``done``)."""
        return True

    @property
    def _scenarios_not_created(self) -> list:
        """Empty: see ``_all_scenarios_created``."""
        return []

    @property
    def _all_sims_run(self) -> bool:
        """True: every scenario of a removed member ran to completion before its directory was removed."""
        return True

    @property
    def _scenarios_not_run(self) -> list:
        """Empty: see ``_all_sims_run``."""
        return []

    def __init__(
        self,
        member_id: Any,
        analysis_id: str,
        master_dir: Path,
        *,
        df_sims,
        cfg_analysis,
        cfg_hpc_system=None,
        enabled_models: list[str] | None = None,
        weather_indexer=None,
        df_status_builder=None,
        sim_run_reader=None,
        sim_run_processing_reader=None,
    ) -> None:
        object.__setattr__(self, "member_id", member_id)
        object.__setattr__(self, "analysis_id", analysis_id)
        object.__setattr__(self, "master_dir", Path(master_dir))
        object.__setattr__(self, "df_sims", df_sims)
        object.__setattr__(self, "cfg_analysis", cfg_analysis)
        object.__setattr__(self, "cfg_hpc_system", cfg_hpc_system)
        object.__setattr__(self, "is_recorded_member", True)
        object.__setattr__(self, "_enabled_models", list(enabled_models or []))
        object.__setattr__(self, "_weather_indexer", weather_indexer)
        object.__setattr__(self, "_df_status_builder", df_status_builder)
        object.__setattr__(self, "_sim_run_reader", sim_run_reader)
        object.__setattr__(self, "_sim_run_processing_reader", sim_run_processing_reader)

    def _get_enabled_model_types(self) -> list[str]:
        """The model types the member enabled, from its moved config (what the panels and rules read)."""
        return list(self._enabled_models)

    def _retrieve_weather_indexer_using_integer_index(self, event_iloc: int):
        """The weather indexer of one event: a pure function of ``df_sims`` and the configured event indices."""
        if self._weather_indexer is None:
            raise MemberRemovedError(self.member_id, "_retrieve_weather_indexer_using_integer_index")
        return self._weather_indexer(event_iloc)

    @property
    def df_status(self):
        """The member's status rows as built from its records, one row per event and model."""
        if self._df_status_builder is None:
            raise MemberRemovedError(self.member_id, "df_status")
        return self._df_status_builder()

    def _retrieve_sim_runs(self, event_iloc: int):
        """A read-only handle for one event whose completion and captured values come through the record resolver."""
        if self._sim_run_reader is None:
            raise MemberRemovedError(self.member_id, "_retrieve_sim_runs")
        return self._sim_run_reader(event_iloc)

    def _retrieve_sim_run_processing_object(self, event_iloc: int):
        """A read-only processing handle for one event: path arithmetic plus record-backed values."""
        if self._sim_run_processing_reader is None:
            raise MemberRemovedError(self.member_id, "_retrieve_sim_run_processing_object")
        return self._sim_run_processing_reader(event_iloc)

    def __getattr__(self, name: str):
        # Reached only for names not set in __init__ or defined on the class. Underscore-private names that
        # Python and pickling probe for must keep raising AttributeError, or copy and pickle misbehave.
        if name.startswith("__"):
            raise AttributeError(name)
        raise MemberRemovedError(self.member_id, name)

    def __setattr__(self, name: str, value) -> None:
        raise MemberRemovedError(self.member_id, f"{name} (assignment: the view is read-only)")


class RecordedSimRun:
    """A read-only handle on one event of a removed member: path arithmetic plus the values captured at removal.

    Stands in for both ``_retrieve_sim_runs(i)`` (which exposes ``._scenario.scen_paths`` and ``model_run_completed``)
    and ``_retrieve_sim_run_processing_object(i)`` (which exposes ``.scen_paths``). ``scen_paths`` is path
    arithmetic under the removed member's directory name, so a reader that tests ``.exists()`` finds the file
    absent; ``captured`` and ``record_path`` carry what replaced it.
    """

    is_recorded = True

    def __init__(self, event_id: str, sim_folder: Path, record: dict, record_path: Path) -> None:
        from types import SimpleNamespace

        self.event_id = event_id
        self._record = record
        self.record_path = Path(record_path)
        self.captured: dict = dict(record.get("captured") or {})
        self.scen_paths = SimpleNamespace(
            sim_folder=Path(sim_folder),
            swmm_full_rpt_file=Path(sim_folder) / "out_swmm" / "full.rpt",
            swmm_full_inp=Path(sim_folder) / "swmm" / "full.inp",
            swmm_hydro_inp=Path(sim_folder) / "swmm" / "hydro.inp",
            swmm_hydraulics_inp=Path(sim_folder) / "swmm" / "hydraulics.inp",
        )
        self._scenario = self

    def model_run_completed(self, model_type: str) -> bool:
        """The run-completion verdict captured at removal for ``model_type``."""
        return bool((self._record.get("run_completed") or {}).get(model_type, False))


def recorded_member_sim_run(master_dir: Path, member_id: Any, analysis_id: str, event_id: str) -> RecordedSimRun:
    """Build the ``RecordedSimRun`` of one event of a removed member from the master's moved record of it."""
    rec_path = member_records_dir(master_dir, member_id) / "scenario_records" / f"{event_id}.json"
    record = _read_json_file(rec_path)
    if record is None:
        raise MemberRemovedError(member_id, f"scenario record of event {event_id!r} (missing or unreadable)")
    sim_folder = Path(master_dir) / "members" / analysis_id / "sims" / event_id
    return RecordedSimRun(event_id, sim_folder, record, rec_path)


def all_members_for_reading(active: dict, recorded: dict) -> dict:
    """The active members followed by the recorded views, as one mapping keyed by member id.

    A reader that must enumerate EVERY member (the per-analysis summary, ``_iter_members_or_self``, the
    benchmarking renderer, the system overview, the report CLI, the whole-experiment targets of the master)
    iterates this. A key present in both is a defect: an id is either active or recorded.
    """
    overlap = set(active) & set(recorded)
    if overlap:
        raise ValueError(f"member ids both active and recorded: {sorted(overlap)!r}")
    merged = dict(active)
    merged.update(recorded)
    return merged


def all_members_for_reading_of(sensitivity) -> dict:
    """``sensitivity._all_members_for_reading()`` when the object offers it, else its plain ``members`` mapping.

    Every real sensitivity analysis offers the method; the fallback serves duck-typed stand-ins that carry only
    ``members`` (nothing is removed there, so the active mapping IS every member).
    """
    method = getattr(sensitivity, "_all_members_for_reading", None)
    return method() if method is not None else sensitivity.members


# ---------------------------------------------------------------------------
# Event-state classification for the Snakefile generators (D24 (g) table).
# ---------------------------------------------------------------------------
def event_ids_for_plan(analysis_dir: Path, all_event_ids: list[str]) -> list[str]:
    """``SIM_IDS`` membership: every event except the removed ones (record ``pending`` or ``done``).

    ``superseded``, failed, incomplete, never-run and switch-off events are all included. READ-ONLY: a
    ``pending`` record is read as removed and no filesystem act is performed, so a dry run deletes nothing.
    """
    return [e for e in all_event_ids if not event_is_removed(analysis_dir, e)]


def report_event_ids(all_event_ids: list[str]) -> list[str]:
    """``REPORT_EVENT_IDS`` membership for the production generator: every event of ``df_sims``, removed ones included.

    The per-event report pages read the tree and the records, so a removed event keeps its page target; a
    render-floor force that deletes every page must plan them all again.
    """
    return list(all_event_ids)
