# %%
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pandas as pd

from hhemt.config.hpc_system import resolve_container_spec, system_directory_bind
from hhemt.resource_management import _parse_slurm_allocated_gpus
from hhemt.scenario import TRITONSWMM_scenario
from hhemt.utils import read_text_file_as_list_of_strings

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ADR-1: in-SIF fallback paths for the model binary when a ContainerSpec does not
# pin an explicit `exe_in_sif` entry for the model. The `.def` build installs the
# binaries under /opt/hhemt/bin (Phase 3). Keyed by model_type.
_DEFAULT_EXE_NAME = {
    "triton": "triton.exe",
    "tritonswmm": "triton.exe",
    "swmm": "runswmm",
}


def model_logfile_for(analysis, event_iloc: int, model_type: Literal["triton", "tritonswmm", "swmm"]) -> Path:
    """The analysis-level model runtime-log path for one (analysis, event, model).

    SINGLE SOURCE OF TRUTH for this naming convention. The PRODUCER
    (``run_simulation_runner.py``, which opens this exact path in ``"w"`` mode on every
    exec) and EVERY consumer (``TRITONSWMM_run.model_run_completed``,
    ``_classify_model_log_failure``, ``report_renderers/per_analysis_summary.py``,
    ``analysis_validation.check_coupled_resume_validity``) MUST resolve through this
    function. NEVER hand-build the path.

    Why this is a free function and not just a method: a hand-built duplicate is exactly
    how ``check_coupled_resume_validity``'s replay-marker arm came to read
    ``{sim_folder}/logs/run_tritonswmm.log`` -- the vestigial ``ScenarioPaths.log_run_*``
    convention that NOTHING writes -- so its ``except Exception: continue`` fired on every
    row and the check passed VACUOUSLY on every experiment (28/28 rows skipped on
    synth_cc_resume, 2026-07-15). One expression of the convention, shared by producer and
    detector, is what makes that class of drift impossible rather than merely unlikely.

    PATH-ONLY, mirroring ``summary_paths.py``: derives from ``analysis`` + ``event_iloc``
    only and MUST NOT instantiate ``TRITONSWMM_scenario`` (whose constructor mkdir's
    ``processed/`` / ``swmm/`` / ``out_swmm/``). This is what lets a read-only validator
    resolve the path without mutating the tree.

    CAUTION -- the file this resolves to is opened ``"w"`` on EVERY runner exec
    (``run_simulation_runner.py``), so it retains ONLY the LAST exec of up to
    ``hpc_restart_times_simulate`` + 1 attempts. It is NOT a per-exec history. Callers
    needing per-attempt history must read ``.snakemake/slurm_logs/{rule}/{jobid}.log``.
    No toolkit-written line reaches this file either -- the runner's ``Command:`` line and
    the ``Resuming ... from hotstart`` notice go to the RUNNER's stderr, not here; the
    file carries only the model subprocess's own stdout/stderr.

    Naming convention (empirically ``model_tritonswmm_sa_gpu_0_r1_evt0.log`` on
    synth_cc_resume -- note the segment is the full ``analysis_id``, NOT ``sa{N}``):
    - Sensitivity sub-analysis:
      ``{master_analysis_dir}/logs/sims/model_{model_type}_{analysis_id}_evt{event_iloc}.log``
    - Regular analysis:
      ``{simlog_directory}/model_{model_type}_evt{event_iloc}.log``
    """
    log_dir = analysis.analysis_paths.simlog_directory
    analysis_id = ""
    if getattr(analysis.cfg_analysis, "is_experiment_member", False):
        analysis_id = str(analysis.cfg_analysis.analysis_id) + "_"
        # Derive the MASTER analysis dir STRUCTURALLY, from this sub's own analysis_dir.
        # A sub's dir is always `{master_analysis_dir}/subanalyses/sa_{sa_id}` (single
        # writer: sensitivity_analysis.py:273 + _create_sub_analyses; the same two-level
        # convention du_sentinels.py:406 detects via parent.name == "subanalyses"), so
        # `.parent.parent` IS the master analysis_dir and this expression equals the
        # master's `analysis_paths.simlog_directory` (analysis.py:273-274) by construction.
        #
        # DO NOT restore the previous `experiment_cfg_yaml.parent / "logs" / "sims"`
        # form. `experiment_cfg_yaml` is the USER'S config-file path
        # (sensitivity_analysis.py:2417 assigns experiment.analysis_config_yaml), so
        # that form anchored the model logs to an arbitrary directory that
        # `run(from_scratch=True)`'s fast_rmtree(analysis_dir) does not cover. Empirically
        # (Rivanna, 2026-08-01, synth_cc_resume_triton): the wipe ran, out_triton/ was
        # emptied, and 28/28 week-stale `Simulation ends` logs survived in the platformdirs
        # cache -- so model_run_completed's raw-marker fallback reported every sim complete,
        # all 28 sims skipped execution, and 168 process_* rules died on a missing
        # performance/ dir. The docstring above ALREADY specified {master_analysis_dir};
        # this restores agreement between the spec and the code.
        #
        # Deliberately NO fallback to the old location here. A fallback would silently
        # re-admit the stale-evidence skip this fixes. The one place a fallback IS correct
        # is eda/raw_resume_identity.py, which reads historical completed arms and gates
        # nothing.
        log_dir = analysis.analysis_paths.analysis_dir.parent.parent / "logs" / "sims"
    return log_dir / f"model_{model_type}_{analysis_id}evt{event_iloc}.log"


def read_walltime_ledger_total_s(model_logfile: Path) -> float | None:
    """Sum the append-only per-attempt wall-time ledger written by run_simulation_runner at
    each sim-finalize (F11). Returns the total wall seconds across ALL attempts of a (possibly
    resumed) sim, or None when the ledger is absent (caller falls back to the perf-summary
    total). The ledger is the ONLY kill-survivable per-attempt wall source — the perf files
    and sim_run_time_minutes are overwrite-prone (see the runner's ledger comment)."""
    import json as _json

    p = model_logfile.parent / "_walltime" / f"{model_logfile.stem}.jsonl"
    if not p.exists():
        return None
    total = 0.0
    try:
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            total += float(_json.loads(line).get("wall_s", 0.0) or 0.0)
    except (OSError, ValueError):
        return None
    return total


def probe_slurm_planned_seconds(jobid: str) -> float | None:
    """This allocation's SLURM queue ('Planned') time in seconds, or None if unobtainable.

    Called ONCE per sim allocation, at worker start, when the job is RUNNING -- which is
    what makes the value final (a job that has STARTED necessarily has a Start, so Planned
    cannot still move) and what makes the scontrol fallback viable (slurmctld has not yet
    aged the record out per MinJobAge).

    Two paths, first success wins:
      1. `sacct --format=Planned` -- the field directly, no arithmetic. Queries slurmdbd,
         whose reachability FROM A COMPUTE NODE is site policy, not a SLURM guarantee.
      2. `scontrol show job` SubmitTime/StartTime -- queries slurmctld, which every compute
         node must reach for the job to be running at all. Both timestamps are local-time
         `%Y-%m-%dT%H:%M:%S` with no offset, so subtracting them is offset-free PROVIDED
         both are taken from the same command output, which they are.

    Returns None -- never 0.0 -- on any failure. The distinction is load-bearing all the way
    to the rendered table: 0.0 asserts the sim did not wait, None says it was not measured,
    and the report renders them differently (metadata.py's em-dash vs a numeral).
    """
    import subprocess as _sp

    def _hms_to_s(text: str) -> float | None:
        # Planned renders as [DD-]HH:MM:SS.
        text = text.strip()
        if not text:
            return None
        days = 0
        if "-" in text:
            d, _, text = text.partition("-")
            days = int(d)
        parts = text.split(":")
        if len(parts) != 3:
            return None
        h, m, s = (float(p) for p in parts)
        return days * 86400.0 + h * 3600.0 + m * 60.0 + s

    try:
        out = _sp.run(
            ["sacct", "-X", "-j", str(jobid), "--format=Planned", "-P", "-n"],
            capture_output=True, text=True, timeout=30,
        )
        if out.returncode == 0:
            val = _hms_to_s(out.stdout.splitlines()[0] if out.stdout.splitlines() else "")
            if val is not None:
                return val
    except (OSError, ValueError, _sp.SubprocessError):
        pass

    try:
        out = _sp.run(
            ["scontrol", "show", "job", str(jobid)],
            capture_output=True, text=True, timeout=30,
        )
        if out.returncode == 0:
            import re as _re
            from datetime import datetime as _dt

            found = dict(_re.findall(r"(SubmitTime|StartTime)=(\S+)", out.stdout))
            if "SubmitTime" in found and "StartTime" in found:
                fmt = "%Y-%m-%dT%H:%M:%S"
                delta = _dt.strptime(found["StartTime"], fmt) - _dt.strptime(found["SubmitTime"], fmt)
                return max(0.0, delta.total_seconds())
    except (OSError, ValueError, _sp.SubprocessError):
        pass

    return None


def read_queue_ledger_seconds(model_logfile: Path) -> tuple[float | None, str, dict[str, float]]:
    """Queue-time totals from the same append-only ledger `read_walltime_ledger_total_s` reads.

    Returns `(total_seconds_or_None, coverage, by_jobid)`:
      * total  -- the sum across every allocation that recorded a queue row, or None when
                  NO row carries `queue_s` (a pre-O21 tree, or 1_job_many_srun_tasks, where
                  the capture is deliberately absent). None is NOT zero; see the probe.
      * coverage -- "k/n", k = allocations with a queue row, n = allocations seen (counted
                  from `wall_s` rows plus queue rows, deduplicated on slurm_jobid). A tree
                  that resumed across the O21 landing boundary reports e.g. "2/5", so a
                  PARTIAL total is legible as partial rather than passing as complete. This
                  is the disclosed-denominator discipline Gotcha 71(d) requires -- a bare
                  total cannot distinguish "summed 5 of 5" from "summed 2 of 5".
      * by_jobid -- per-allocation queue seconds, so the report can show each SLURM row its
                  OWN wait rather than repeating the sim total across every row of a
                  resumed sim (which would read as each allocation having waited the total).
    """
    import json as _json

    p = model_logfile.parent / "_walltime" / f"{model_logfile.stem}.jsonl"
    if not p.exists():
        return (None, "0/0", {})
    by_jobid: dict[str, float] = {}
    seen: set[str] = set()
    try:
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            rec = _json.loads(line)
            jid = str(rec.get("slurm_jobid") or "")
            if jid:
                seen.add(jid)
            if rec.get("queue_s") is not None and jid:
                by_jobid[jid] = float(rec["queue_s"])
    except (OSError, ValueError):
        return (None, "0/0", {})
    if not by_jobid:
        return (None, f"0/{len(seen)}", {})
    return (sum(by_jobid.values()), f"{len(by_jobid)}/{len(seen)}", by_jobid)


def read_attempt_index_by_jobstep(model_logfile: Path) -> dict[str, int]:
    """`{f"{slurm_jobid}.{slurm_step_id}": attempt}` from the same append-only ledger.

    The user's ask is a row per resume, numbered. That row ALREADY EXISTS in the SLURM
    efficiency table and is already joined to its simulation -- measured on the delivered
    generation, a resumed sim is ONE allocation carrying several srun STEPS (28 sims,
    `n_resumes: 3`, 18 allocations with 4 steps and 10 with 5, totalling the 122 sim rows).
    What is missing is only the label. So this returns a LABEL index, not new capture:
    `run_simulation_runner` already appends `slurm_jobid` and `slurm_step_id` alongside
    `attempt` on every attempt, and the efficiency CSV's `JobID` column is exactly
    `{jobid}.{step}` -- so the two join directly, with no heuristic and no ordering
    assumption.

    Keyed on the FULL job-step id rather than on the allocation, because the allocation is
    shared by every attempt of the sim; keying on it would label all four rows identically
    and reintroduce the "a per-sim figure no attempt experienced" error the queue-time note
    already warns about.

    Coverage is strictly better than the queue ledger's: the `wall_s` row is appended on
    every attempt regardless of `multi_sim_run_method`, whereas the `queue_s` row is
    `batch_job`-only. A row whose `slurm_step_id` is null (a pre-capture tree, or a local
    run) is SKIPPED rather than keyed on a bare allocation id -- an unlabelled row is
    correct, a wrongly-labelled one is not.

    Returns `{}` on any read or parse failure, so a damaged ledger degrades to today's
    unlabelled rendering instead of to a partial index that would mislabel some rows.
    """
    import json as _json_a

    p = model_logfile.parent / "_walltime" / f"{model_logfile.stem}.jsonl"
    if not p.exists():
        return {}
    out: dict[str, int] = {}
    try:
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            rec = _json_a.loads(line)
            if rec.get("attempt") is None:
                continue
            jid = str(rec.get("slurm_jobid") or "")
            step = rec.get("slurm_step_id")
            if not jid or step is None or str(step) == "":
                continue
            out[f"{jid}.{step}"] = int(rec["attempt"])
    except (OSError, ValueError, TypeError):
        return {}
    return out


class TRITONSWMM_run:
    def __init__(self, scenario: "TRITONSWMM_scenario") -> None:
        self._scenario = scenario
        self._analysis = scenario._analysis
        self.weather_event_indexers = scenario.weather_event_indexers

    @property
    def _triton_swmm_raw_output_directory(self):
        """Directory containing raw TRITON outputs from the TRITON-SWMM coupled model."""
        raw_type = self._analysis.cfg_analysis.TRITON_raw_output_type
        out_dir = self._scenario.scen_paths.out_tritonswmm
        if out_dir is not None and out_dir.exists():
            raw_dir = out_dir / raw_type
            if raw_dir.exists() and any(raw_dir.iterdir()):
                return out_dir
        # Fallback for legacy directory structure
        fallback = self._scenario.scen_paths.sim_folder / "output"
        if fallback.exists():
            raw_dir = fallback / raw_type
            if raw_dir.exists() and any(raw_dir.iterdir()):
                return fallback
        return out_dir if out_dir is not None else fallback

    def _analysis_level_model_logfile(self, model_type: Literal["triton", "tritonswmm", "swmm"]) -> Path:
        """Return the analysis-level model runtime-log path for this scenario.

        Thin delegate to the module-level ``model_logfile_for`` free function, which is
        the SINGLE SOURCE OF TRUTH for this convention (read its docstring before
        touching either). Retained as a method because ``TRITONSWMM_run`` is the
        producer-side call surface (``prepare_simulation_command`` writes through it,
        ``model_run_completed`` / ``_classify_model_log_failure`` read through it) and
        several tests patch it by name.
        """
        return model_logfile_for(self._analysis, self._scenario.event_iloc, model_type)

    def raw_triton_output_dir(self, model_type: Literal["triton", "tritonswmm"] = "tritonswmm"):
        """Directory containing raw TRITON binary output files (H, QX, QY, MH).

        Parameters
        ----------
        model_type : Literal["triton", "tritonswmm"]
            Which model's raw output directory to retrieve (default: "tritonswmm")

        Returns
        -------
        Path
            Directory containing raw TRITON outputs
        """
        raw_type = self._analysis.cfg_analysis.TRITON_raw_output_type

        if model_type == "triton":
            base = self._scenario.scen_paths.out_triton
        else:
            base = self._scenario.scen_paths.out_tritonswmm

        if base is None:
            # Fallback for legacy directory structure
            base = self._scenario.scen_paths.sim_folder / "output"

        raw_dir = base / raw_type
        if raw_dir.exists() and any(raw_dir.iterdir()):
            return raw_dir
        return base

    @property
    def sim_run_completed(self):
        """Legacy completion check for the coupled TRITON-SWMM model."""
        return self.model_run_completed("tritonswmm")

    def _coupled_swmm_report_finalized(self, model_type: str) -> bool:
        """Return False only for a coupled TRITON-SWMM run whose SWMM report is unfinalized.

        Coupled completion is detected from the TRITON ``"Simulation ends"`` marker
        (Gotcha 6), which does NOT prove the coupled SWMM report was written. A
        resumed (or crashed / early-exited) coupled run can leave a 0-byte or
        truncated ``out_tritonswmm/swmm/hydraulics.rpt``. This gate additionally
        requires that rpt to be finalized (carry the terminal ``"Analysis ended on"``
        marker, via ``swmm_output_parser.rpt_is_complete``). Non-``tritonswmm`` model
        types are unaffected (always True). Returning False here converts a silent
        bad coupled output into a RETRIABLE failure, so the existing ``retries:`` +
        ``--pickup-where-leftoff`` machinery re-runs (and resumes) the sim.
        """
        if model_type != "tritonswmm":
            return True
        rpt = self._scenario.scen_paths.swmm_hydraulics_rpt
        if rpt is None or not rpt.exists():
            return False
        from hhemt.swmm_output_parser import rpt_is_complete

        return rpt_is_complete(rpt)

    def model_run_completed(self, model_type: Literal["triton", "tritonswmm", "swmm"]) -> bool:
        """Check if a simulation completed for a specific model type.

        Authoritative source of truth: the per-model `TRITONSWMM_model_log` JSON
        `simulation_completed` field, accessed via `scenario.get_log(model_type)`.
        Falls back to the raw log-marker scan (`"Simulation ends"` for TRITON /
        TRITON-SWMM, `"EPA SWMM completed"` for SWMM) when the model log JSON
        does not yet exist or has not yet recorded a value — this preserves
        first-run correctness for paths where the simulation has just emitted
        its log but the log-field has not yet been written.
        """
        # Primary: read post-processing-aware log field.
        #
        # ONLY a POSITIVE (True) record is authoritative. A False/None record means
        # "not known to be complete" and MUST fall through to the raw-marker scan.
        #
        # Why (the sticky-False latch): run_simulation_runner.py WRITES this field
        # with the value this method RETURNS. The prior `if completed is not None`
        # form therefore latched: once any attempt recorded False, `bool(False) and
        # ...` short-circuited (never even evaluating the rpt gate) and this method
        # returned False FOREVER — so a later hotstart-resume that genuinely ran to
        # t=end and finalized its coupled rpt was still reported incomplete, its
        # retry re-resumed from the t=end checkpoint, replayed to `dt: -nan`, and the
        # rule burned every retry. Empirically confirmed on Rivanna (synth_cc_resume,
        # job 17021807 step .1: COMPLETED 0:0, 144/144 cfgs written, hydraulics.rpt
        # carrying "Analysis ended on" — recorded simulation_completed=False).
        #
        # The latch only bites when the runner SURVIVES a failed sim (a SLURM
        # walltime kill reaps the runner before it can write False, which is why
        # production hotstart-resume masked this). It equally broke `retries:` for
        # any genuinely crashed (e.g. segfaulting) sim.
        #
        # Behavior delta is confined to the buggy branch:
        #   True -> `bool(True) and rpt` === `rpt`   (unchanged)
        #   None -> falls through to the raw scan     (unchanged)
        #   False -> was: permanently False; now: re-derived from this run's markers
        try:
            model_log = self._scenario.get_log(model_type)
            completed = model_log.simulation_completed.get()
            if completed:
                return self._coupled_swmm_report_finalized(model_type)
        except (AttributeError, FileNotFoundError):
            pass  # log not yet written — fall through to raw-file check

        # Fallback: raw log-marker scan (first-run path)
        log_file = self._analysis_level_model_logfile(model_type)
        if not log_file.exists():
            return False
        log_content = log_file.read_text()
        if model_type in ("triton", "tritonswmm"):
            success = "Simulation ends" in log_content
        else:  # swmm
            success = "EPA SWMM completed" in log_content

        # Divergence check — WARN, but deliberately NOT load-bearing.
        #
        # The prior comment hypothesized a "raw-output-clearing race". That hypothesis is
        # FALSIFIED: _clear_raw_outputs deletes only children that are DIRECTORIES named in
        # _CLEAR_RAW_DELETE_SUBDIRS (process_simulation.py:1215), and its docstring states
        # top-level files such as performance.txt are preserved -- so the only mechanism the
        # comment named cannot produce the condition it detects. Every observed instance of
        # this condition has been a DURABLE state (a completion marker that outlived the
        # artifacts it describes), never a transient one: the log is truncated by
        # `open(model_logfile, "w")` BEFORE Popen (run_simulation_runner.py:459), so during a
        # sim `success` is False and this branch is unreachable; and every in-runner call
        # sits after proc.wait().
        #
        # Promoted DEBUG -> WARNING so the next instance is visible without a post-mortem.
        # NOT promoted to actuation (i.e. not folded into the return value) for two reasons:
        # (1) the check sits only in the raw-marker FALLBACK -- a positive
        # `simulation_completed` returns at the `if completed:` branch above and never
        # reaches here, so coverage would be partial by construction; and (2) gating
        # completion on an output artifact's presence inverts the toolkit's log-based-truth
        # principle. The durable fix is that model logs now live INSIDE analysis_dir
        # (model_logfile_for), so the wipe that removes the outputs removes the marker too.
        if model_type in ("triton", "tritonswmm") and success:
            perf_file = self.performance_file(model_type=model_type)
            if not perf_file.exists():
                logger.warning(
                    "model_run_completed: %s log reports completion but performance.txt "
                    "is absent at %s. This usually means a completion marker outlived the "
                    "outputs it describes (stale evidence from a prior campaign). "
                    "Completion is being reported from the log marker anyway; downstream "
                    "process_* rules will fail on the missing performance/ directory.",
                    model_type,
                    perf_file,
                )
        return success and self._coupled_swmm_report_finalized(model_type)

    def _classify_model_log_failure(self, model_type: Literal["triton", "tritonswmm", "swmm"]) -> str:
        """Classify the failure mode of an incomplete simulation from its model log.

        Reads the analysis-level model log and searches for known SLURM failure
        markers. Intended to be called only when ``model_run_completed()`` returns
        False for the same model_type.

        Parameters
        ----------
        model_type : Literal["triton", "tritonswmm", "swmm"]
            Which model's log to inspect.

        Returns
        -------
        str
            One of:
            - ``"timeout"`` — log contains ``DUE TO TIME LIMIT`` (SLURM wall-time kill)
            - ``"unclassified"`` — log exists but no known failure marker found
            - ``"no_log"`` — model log file does not exist
        """
        log_file = self._analysis_level_model_logfile(model_type)

        if not log_file.exists():
            return "no_log"

        log_content = log_file.read_text()

        if "DUE TO TIME LIMIT" in log_content:
            return "timeout"

        return "unclassified"

    @property
    def performance_timeseries_dir(self):
        return self._triton_swmm_raw_output_directory / "performance"

    def performance_file(self, model_type: Literal["triton", "tritonswmm", "swmm"]) -> Path:
        """Get performance.txt file for a specific model type.

        Parameters
        ----------
        model_type : Literal["triton", "tritonswmm", "swmm"]
            Which model's performance file to retrieve

        Returns
        -------
        Path
            Path to performance.txt (may not exist)
        """
        if model_type == "triton":
            output_dir = self._scenario.scen_paths.out_triton
        elif model_type == "tritonswmm":
            output_dir = self._scenario.scen_paths.out_tritonswmm
        elif model_type == "swmm":
            # SWMM doesn't write performance.txt files
            output_dir = self._scenario.scen_paths.out_swmm
        else:
            raise ValueError(f"model_type must be 'triton', 'tritonswmm', or 'swmm', got {model_type}")

        if output_dir is None:
            # Fallback for legacy structure
            output_dir = self._scenario.scen_paths.sim_folder / "output"

        return output_dir / "performance.txt"

    @property
    def model_types_enabled(self):
        """Return list of enabled model types for this scenario.

        Returns:
            List of strings: ['triton', 'tritonswmm', 'swmm']
        """
        sys_cfg = self._scenario._system.cfg_system
        enabled = []
        if sys_cfg.toggle_triton_model:
            enabled.append("triton")
        if sys_cfg.toggle_tritonswmm_model:
            enabled.append("tritonswmm")
        if sys_cfg.toggle_swmm_model:
            enabled.append("swmm")
        return enabled

    def _retrieve_hotstart_file_for_incomplete_triton_or_tritonswmm_simulation(
        self, model_type: Literal["triton", "tritonswmm"]
    ) -> Path | None:
        """Find latest hotstart CFG file for resuming incomplete TRITON/TRITON-SWMM simulation.

        Returns None if no hotstart files found (simulation never started or CFGs cleared).

        Parameters
        ----------
        model_type : Literal["triton", "tritonswmm"]
            Which model's hotstart file to retrieve

        Returns
        -------
        Path | None
            Path to latest complete CFG checkpoint, or None if not available
        """
        if model_type == "triton":
            output_dir = self._scenario.scen_paths.out_triton
            default_cfg = self._scenario.scen_paths.triton_cfg
        else:
            output_dir = self._scenario.scen_paths.out_tritonswmm
            default_cfg = self._scenario.scen_paths.triton_swmm_cfg

        if output_dir is None:
            return None

        cfg_dir = output_dir / "cfg"
        if not cfg_dir.exists():
            return None

        cfgs = list(cfg_dir.glob("*.cfg"))
        if len(cfgs) == 0:
            return None

        # Find latest complete CFG checkpoint
        dic_cfgs = {"step": [], "f_cfg": []}
        for f_cfg in cfgs:
            step = return_the_reporting_step_from_a_cfg(f_cfg)
            dic_cfgs["step"].append(step)
            dic_cfgs["f_cfg"].append(f_cfg)

        df_cfgs = pd.DataFrame(dic_cfgs).set_index("step").sort_index()
        df_cfgs["file_line_length"] = -1
        for step, cfg in df_cfgs.iloc[::-1].iterrows():
            file_as_list = read_text_file_as_list_of_strings(cfg["f_cfg"])
            df_cfgs.loc[step, "file_line_length"] = len(file_as_list)  # type: ignore

        typical_length = df_cfgs["file_line_length"][df_cfgs["file_line_length"] > 0].mode().iloc[0]
        latest_complete = df_cfgs[df_cfgs["file_line_length"] == typical_length]
        if latest_complete.empty:
            return None

        return Path(latest_complete.iloc[-1]["f_cfg"])

    def _hotstart_cfg_dir(self, model_type: Literal["triton", "tritonswmm"]) -> Path | None:
        """Directory of the ``config_NNNN.cfg`` hotstart checkpoints TRITON writes
        at the ``print_interval`` (reporting) cadence, or None when the model has
        no output dir. Mirrors the dir logic of
        ``_retrieve_hotstart_file_for_incomplete_triton_or_tritonswmm_simulation``.
        """
        if model_type == "triton":
            output_dir = self._scenario.scen_paths.out_triton
        else:
            output_dir = self._scenario.scen_paths.out_tritonswmm
        if output_dir is None:
            return None
        return output_dir / "cfg"

    def prune_hotstart_cfgs_above_step(
        self,
        model_type: Literal["triton", "tritonswmm"],
        *,
        target_step: int,
    ) -> int:
        """Delete every ``config_NNNN.cfg`` whose reporting step EXCEEDS ``target_step``,
        so the resume picker is FORCED to select exactly ``target_step``.

        KR-a (deterministic same-timestep interruption). The multi-resume harness kills
        on a POLL (``wait_with_deterministic_checkpoint_kill``, 2 s), so a fast config
        (rank-8 OpenMP/Hybrid, GPU: sub-2 s reporting periods) writes several more
        checkpoints between the threshold being met and the SIGTERM landing. The picker
        (``_retrieve_hotstart_file_for_incomplete_triton_or_tritonswmm_simulation``)
        returns ``latest_complete.iloc[-1]`` — the HIGHEST complete step — so that
        overshoot became the realized resume boundary and it VARIED PER CONFIG. The b4b
        hotstart-resume experiment is only valid if every config resumes at the SAME
        reporting step, so this prune runs BEFORE the picker and removes the overshoot.

        MUST be called before ``prepare_simulation_command`` (which invokes the picker
        internally). Calling it after is a silent no-fix: the cfg is already chosen and
        the launch command already built, so the sim still resumes from the overshoot
        while the log reads correct.

        Numbering contract (measured, not assumed): TRITON writes ``config_N.cfg``
        1-based, contiguous, unpadded (``config_9.cfg`` / ``config_99.cfg`` /
        ``config_1000.cfg``; a real 1080-step dir holds steps 1..1080 with zero gaps).
        Because numbering is contiguous from 1, ``len(cfgs) == max(step)`` identically —
        which is why deleting only the TOP preserves the count-based arming predicate in
        ``wait_with_deterministic_checkpoint_kill`` EXACTLY. Do NOT extend this to delete
        interior files: that would break the count/step identity the watcher relies on and
        the kill would then fire one reporting step late per missing file.

        Size-mutating, so it re-stamps the DU sentinels per the ``du sentinels written at
        every mutation site`` stipulation (PATTERN B: unlink + ``restamp_parent_sentinels``)
        — NOT ``# EXEMPT-DU``: these cfgs live inside the scenario scope and ARE DU-counted.

        Returns the number of cfg files removed (0 when nothing was above ``target_step``).
        """
        from hhemt.du_sentinels import restamp_parent_sentinels

        cfg_dir = self._hotstart_cfg_dir(model_type)
        if cfg_dir is None or not cfg_dir.exists():
            return 0
        analysis_dir = self._analysis.analysis_paths.analysis_dir
        n_removed = 0
        for f_cfg in sorted(cfg_dir.glob("*.cfg")):
            try:
                step = return_the_reporting_step_from_a_cfg(f_cfg)
            except (ValueError, IndexError):
                continue  # unparseable name: leave it alone rather than guess
            if step <= target_step:
                continue
            try:
                f_cfg.unlink()
                # PATTERN B — must be the IMMEDIATELY-following sibling of the unlink, in
                # the same block: check_du_sentinel_sites only accepts the next statement
                # in the same statement list, or the trailing statement of a `finally`. A
                # `finally` is wrong here because it would restamp on the failure path too.
                # Consequence of living inside the try: an OSError from the restamp is
                # caught by the same handler, so such a file is deleted but not counted —
                # an under-report that correctly signals its DU obligation did not complete.
                restamp_parent_sentinels(f_cfg, analysis_dir=analysis_dir)
            except OSError:
                continue
            n_removed += 1
        return n_removed

    def wait_with_deterministic_checkpoint_kill(
        self,
        proc,
        *,
        model_type: Literal["triton", "tritonswmm"],
        n_checkpoints: int,
        poll_interval_s: float = 2.0,
    ) -> int:
        """Wait for the sim subprocess, forcing AT MOST ONE mid-sim hard-kill per
        ATTEMPT of the multi-resume interruption schedule.

        ``n_checkpoints`` is the ABSOLUTE schedule entry for THIS attempt —
        ``resume_interruption_schedule[k]`` where ``k`` is the persisted
        ``n_resumes`` count read by the caller. It is NOT a per-attempt delta and
        MUST NOT be offset by a baseline: the hotstart cfg dir accumulates
        monotonically across attempts (the resume picker globs all cfgs and prunes
        nothing), so at the start of the attempt following ``k`` resumes the dir
        holds about ``schedule[k-1] + 1`` files, and because the schedule is
        STRICTLY INCREASING (enforced by the field validator) the predicate
        ``len(cfgs) >= schedule[k] + 1`` is by construction not yet satisfied.
        Subtracting a baseline while passing an absolute entry would push kill ``k``
        to ``baseline + schedule[k] + 1``: with ``schedule = (25, 50, 75)`` the
        second kill lands near index 76 and the third never fires at all.

        Polls the hotstart cfg dir; once ``>= n_checkpoints + 1`` ``config_NNNN.cfg``
        files exist (the newest may be mid-write, so N+1 present guarantees N are
        complete), issues a process-group SIGTERM (``os.killpg`` on the
        ``bash -lc "... srun ... triton.exe"`` wrapper's session) so the sim exits
        INCOMPLETE and the Snakemake ``retries:`` re-dispatch resumes from the
        latest complete checkpoint. If the sim COMPLETES before the threshold, no
        kill fires (graceful degradation — the sim just finishes). Returns the
        subprocess return code.

        The kill is a process-group SIGTERM, NOT ``proc.kill()``/SIGKILL: ``proc``
        is the bash WRAPPER, and the sim itself is a remote ``srun`` STEP under a
        step ``slurmstepd``. A group SIGKILL reaps bash + the srun client too fast
        for srun to tear the step down, so ``triton.exe`` ORPHANS and runs to t=end
        (empirically confirmed on Rivanna, ``proctrack/cgroup``, job 17018902). A
        group SIGTERM is CAUGHT by the srun client, which force-terminates the step
        and delivers an UNCATCHABLE SIGKILL to the task (``srun: ... task 0:
        Killed``) — so TRITON still cannot finalize-and-exit-0, but the kill is
        routed through srun's signal handler (which SIGKILL bypasses). Requires the
        launcher's ``Popen(..., start_new_session=True)`` so the wrapper leads its
        own group.
        """
        import signal
        import time

        cfg_dir = self._hotstart_cfg_dir(model_type)
        killed = False
        while proc.poll() is None:
            if not killed and cfg_dir is not None and cfg_dir.exists():
                if len(list(cfg_dir.glob("*.cfg"))) >= n_checkpoints + 1:
                    # Rivanna srun-step teardown (SchedMD SLURM, proctrack/cgroup):
                    # process-group SIGTERM, NOT proc.kill()/SIGKILL. `proc` is the
                    # `bash -lc "... srun ... triton.exe"` wrapper (launched with
                    # start_new_session=True, so it leads its own group). A group
                    # SIGKILL reaps bash + the srun client too fast for srun to tear
                    # the step down -> triton.exe ORPHANS and runs to t=end
                    # (empirically confirmed on Rivanna, job 17018902). A group
                    # SIGTERM is CAUGHT by the srun client, which force-terminates the
                    # step and delivers an UNCATCHABLE SIGKILL to the task ("srun:
                    # ... task 0: Killed", job 17018943) -- so TRITON still cannot
                    # finalize-and-exit-0, but the kill is routed through srun's
                    # handler, which a group SIGKILL fatally bypasses. New session
                    # isolates the group: no sibling step / batch job / orchestrator
                    # is signalled.
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    except (ProcessLookupError, PermissionError):
                        # Group already gone (sim raced to completion) or perms -
                        # fall back to a direct terminate so the watcher progresses.
                        proc.terminate()
                    killed = True
            time.sleep(poll_interval_s)
        return proc.returncode

    def _write_repro_script(
        self,
        script_path,
        module_load_cmd,
        env,
        launch_cmd_str,
    ):
        lines = []

        lines.append("#!/usr/bin/env bash")
        lines.append("# --- Modules ---")
        if module_load_cmd:
            # strip trailing semicolon for readability
            lines.append(module_load_cmd.rstrip("; "))
        else:
            lines.append("# (no modules)")
        lines.append("")
        lines.append("# --- Environment ---")
        for k, v in sorted(env.items()):
            lines.append(f'export {k}="{v}"')
        lines.append("")
        lines.append("# --- Launch ---")
        lines.append("")
        lines.append(launch_cmd_str)
        lines.append("")

        script_path.write_text("\n".join(lines))
        script_path.chmod(0o755)

    def prepare_simulation_command(
        self,
        pickup_where_leftoff: bool,
        verbose: bool = True,
        model_type: str = "tritonswmm",
        *,
        execution_locus: str | None = None,
    ):
        """
        Prepare simulation command for specified model type.

        Parameters
        ----------
        pickup_where_leftoff : bool
            Resume from last checkpoint if available
        verbose : bool
            Print progress messages
        model_type : str
            One of: "triton", "tritonswmm", "swmm"

        Returns
        -------
        tuple or None
            (cmd, env, logfile, sim_start_reporting_tstep) or None if already completed
        """
        valid_types = ("triton", "tritonswmm", "swmm")
        if model_type not in valid_types:
            raise ValueError(f"model_type must be one of {valid_types}, got {model_type}")

        multi_sim_run_method = self._analysis.cfg_analysis.multi_sim_run_method
        # Execution locus is a TWO-AXIS property, exactly as workflow.py:2115 resolves
        # it: the dispatch-family label OR the driver-resolved locus. A config-only
        # predicate here CANNOT express it, because `multi_sim_run_method="local"` +
        # `execution_mode="slurm"` (the doi_emitter/[Q8] path) routes every sim to a
        # per-rule sbatch while keeping the `local` label. Keying on the label alone
        # dropped the srun entirely for that path -- a 2-GPU sim ran as ONE rank with
        # HIP_VISIBLE_DEVICES set to the full list (the 0%-util failure the GPU block
        # below documents). Keying on `in_slurm` alone is equally wrong in the other
        # direction: a synth fixture inside a pytest chunk would srun-wrap against a
        # 1-task allocation. The driver knows the locus and now threads it; when it is
        # absent (legacy CLI, _create_subprocess_sim_run_launcher) the label alone
        # decides, which is correct for that path because it is always `local`.
        # DISJUNCTION, not a flag-wins conditional: the one cell where they differ
        # is an allocation-resident dispatch family carrying an explicit `local`
        # locus, and there the disjunction keeps the srun. A silent srun loss is
        # the failure this whole change exists to remove, so the fail-safe arm
        # wins over the marginally-more-precise one. Same shape as workflow.py's
        # own two-axis resolver, which is the authority cited above.
        using_srun = (
            multi_sim_run_method in {"1_job_many_srun_tasks", "batch_job"}
            or execution_locus == "slurm"
        )

        # ----------------------------
        # Model-specific paths
        # ----------------------------
        # Select executable and CFG based on model type
        if model_type == "triton":
            exe = self._scenario.scen_paths.sim_triton_executable
            cfg = self._scenario.scen_paths.triton_cfg
            model_logfile = self._analysis_level_model_logfile("triton")
        elif model_type == "tritonswmm":
            exe = self._scenario.scen_paths.sim_tritonswmm_executable
            cfg = self._scenario.scen_paths.triton_swmm_cfg
            model_logfile = self._analysis_level_model_logfile("tritonswmm")
        elif model_type == "swmm":
            exe = self._scenario.scen_paths.sim_swmm_executable
            cfg = None  # SWMM uses .inp file, not CFG
            model_logfile = self._analysis_level_model_logfile("swmm")
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # compute config
        run_mode = self._analysis.cfg_analysis.run_mode
        n_mpi_procs = self._analysis.cfg_analysis.n_mpi_procs
        n_omp_threads = self._analysis.cfg_analysis.n_omp_threads
        n_gpus = self._analysis.cfg_analysis.n_gpus
        n_nodes_per_sim = self._analysis.cfg_analysis.n_nodes

        # Container mode: the SIF carries the pre-compiled binary (the on-cluster
        # compile is skipped, Phase 2 setup_workflow.py). Rewrite `exe` to
        # `apptainer exec [gpu_flag] {sif} {in-SIF exe}`. The gpu_flag is applied
        # ONLY on GPU sim rungs (derived here from run_mode) — NEVER on CPU/processing
        # rungs (this is why a per-class exec_args map is unnecessary, OE-2). Placed
        # AFTER the compute-config resolution (SE Spec 1) so `run_mode`/`n_gpus` are
        # defined; inserting at the `exe` assignment site would NameError run_mode.
        cspec = resolve_container_spec(self._analysis.cfg_hpc_system)
        if self._analysis.cfg_analysis.execution_environment == "container" and cspec is not None:
            exe_in_sif = cspec.exe_in_sif.get(model_type) or f"/opt/hhemt/bin/{_DEFAULT_EXE_NAME[model_type]}"
            # Per-arch SIF resolution (multi-SIF cross-hardware, Option A): resolve THIS
            # row's arch (gpu_hardware) from its (sub-analysis) partition and pick the
            # matching SIF; fall back to sif_path when no map entry (single-SIF/CPU,
            # byte-identical to before). Key is gpu_hardware ("a100"/"a6000") — the same
            # namespace resolve_gpu_target[0] returns and sif_paths_by_arch is keyed on.
            from hhemt.config.hpc_system import resolve_gpu_target

            _row_hw, _ = resolve_gpu_target(
                self._analysis.cfg_hpc_system,
                self._analysis.cfg_analysis.hpc_ensemble_partition,
            )
            _sim_sif = cspec.sif_paths_by_arch.get(_row_hw) if _row_hw else None
            _sif = _sim_sif or cspec.sif_path
            _gpu = f"{cspec.gpu_flag} " if (run_mode == "gpu" and cspec.gpu_flag) else ""
            _extra = (" ".join(cspec.extra_exec_args) + " ") if cspec.extra_exec_args else ""
            # Container Change 2 (ROOT CAUSE of zero-output): TRITON derives its output
            # base dir from argv[0] two directory levels up (config_utils get_root_dir).
            # In-SIF the exe is /opt/hhemt/bin/{exe} -> two-up = /opt/hhemt INSIDE the
            # read-only SIF, so the relative output_folder write silently fails (native
            # runs .../sims/<evt>/build/{exe} -> two-up = the writable sim dir). Bind the
            # host scenario out_tritonswmm dir onto the in-SIF {project_dir}/out_tritonswmm
            # so output lands on the writable host fs. triton/tritonswmm only (SWMM writes
            # via its own inp/rpt/out args). The host out_tritonswmm is created by prepare.
            #
            # The dir is MODEL-KEYED and must stay so: _generate_TRITON_cfg writes
            # output_folder="out_triton" while _generate_TRITON_SWMM_cfg writes
            # output_folder="out_tritonswmm" (scenario.py). Hardcoding out_tritonswmm
            # here left the standalone rung's /opt/hhemt/out_triton unbound inside the
            # read-only SIF -> `[ERROR] Error reading file: ` + Kokkos::Cuda finalize
            # failure + exit 1 (Rivanna jobs 17090721/22/28/30/31/70). output_folder is
            # the ONLY relative path in either generated cfg — DEM/MANNINGS/HYDROGRAPH/
            # HYDO_SRC_LOC/EXTBC are absolute and already inside the APPTAINER_BIND
            # closure (cspec.binds + system_directory_bind + {analysis_dir}:{analysis_dir}),
            # so this one keyed entry is the complete bind map.
            _out_bind = ""
            if model_type != "swmm":
                _host_out = (
                    self._scenario.scen_paths.out_triton
                    if model_type == "triton"
                    else self._scenario.scen_paths.out_tritonswmm
                )
                _proj_dir = "/".join(exe_in_sif.split("/")[:-2])  # two-up from in-SIF exe = TRITON project_dir
                _out_bind = f"-B {_host_out}:{_proj_dir}/{_host_out.name} "
            exe = f"apptainer exec {_gpu}{_extra}{_out_bind}{_sif} {exe_in_sif}"
            # `exe` now expands inside launch_cmd_str's `{exe} {cfg}` as
            #   srun … apptainer exec --rocm -B {host_out}:/opt/hhemt/out_tritonswmm {sif} {exe} {cfg}

        # Check if already completed
        if self._scenario.model_run_completed(model_type):
            if verbose:
                print(f"{model_type} simulation already completed", flush=True)
            return None

        # Try to resume from hotstart if requested
        sim_start_reporting_tstep = 0
        if pickup_where_leftoff and model_type != "swmm":
            hotstart_cfg = self._retrieve_hotstart_file_for_incomplete_triton_or_tritonswmm_simulation(
                model_type=model_type
            )
            if hotstart_cfg is not None:
                cfg = hotstart_cfg
                sim_start_reporting_tstep = return_the_reporting_step_from_a_cfg(hotstart_cfg)
                # Track hotstart resumes as a first-class log field (P2).
                _ml = self._scenario.get_log(model_type)
                _ml.n_resumes.set((_ml.n_resumes.get() or 0) + 1)
                # KR-b: persist the REALIZED resume reporting-step (the cfg the picker chose)
                # on the SAME _ml object, so the b4b same-timestep-interruption experiment is
                # answerable from the data forever. Mirrors n_resumes' write path exactly.
                _prior_tsteps = list(_ml.resume_reporting_tsteps.get() or [])
                _ml.resume_reporting_tsteps.set(_prior_tsteps + [int(sim_start_reporting_tstep)])
                if verbose:
                    print(
                        f"Resuming {model_type} from hotstart: {hotstart_cfg}",
                        flush=True,
                    )
                # Coupled TRITON-SWMM hotstart-resume is now supported by pinned TRITON
                # 3a832f7d (persist-and-replay of the SWMM exchange history), so the interim
                # loud-failure guard that raised SimulationError here — plus its sibling
                # check_coupled_hotstart_resume in analysis_validation.py — is removed.
                # Post-fix, a resumed coupled sim whose replay silently failed to engage is
                # caught retroactively by check_coupled_resume_validity's replay-marker arm.
                # n_resumes (incremented above) remains the first-class resume record that
                # arm reads.

        og_env = os.environ.copy()
        env = dict()
        # Always prepend ${CONDA_PREFIX}/lib so triton.exe finds the conda env's
        # libstdc++.so.6 at runtime (libstdc++ ABI fix — matches the link-time
        # libstdc++ injected by system.py's compile script). Required because
        # gcc/12.4.0 module's libstdc++ maxes at GLIBCXX_3.4.30 but the conda
        # env's libgdal/libmuparser need GLIBCXX_3.4.31+.
        ld_segments = ["${CONDA_PREFIX}/lib"]
        swmm_dir = self._analysis._system.cfg_system.SWMM_software_directory
        if swmm_dir:
            swmm_path = swmm_dir / "swmm_build" / "bin"
            ld_segments.append(str(swmm_path))
        # Append the parent LD_LIBRARY_PATH so HPC-provided library paths (rocm,
        # libfabric, etc.) inherited from the SBATCH environment are preserved.
        # The SBATCH script loads all required modules before launching Snakemake,
        # so og_env["LD_LIBRARY_PATH"] already contains the full set of needed
        # paths on both login and compute nodes. Passing this via env= dict (not
        # via shell command string) avoids any ARG_MAX concerns since env vars go
        # through a separate execve() vector.
        ld_segments.append(og_env.get("LD_LIBRARY_PATH", "$LD_LIBRARY_PATH"))
        env["LD_LIBRARY_PATH"] = ":".join(ld_segments)

        # PATH is intentionally omitted from the env dict. The bash -lc (login shell)
        # rebuilds PATH from /etc/profile and the module load in the command string
        # adds the correct HPC paths. Copying os.environ["PATH"] here would propagate
        # the full accumulated module environment into the subprocess argument list,
        # which can exceed Linux's ARG_MAX limit and cause OSError: [Errno 7].

        # ----------------------------
        # OpenMP configuration
        # ----------------------------
        if run_mode in ("openmp", "hybrid"):
            env["OMP_NUM_THREADS"] = str(n_omp_threads)
            env["OMP_PROC_BIND"] = "true"
            env["OMP_PLACES"] = "cores"
        else:
            # OMP_NUM_THREADS=1 for serial/mpi/gpu modes.
            # OMP_PROC_BIND=true and OMP_PLACES=cores are REQUIRED even at 1 thread:
            # Kokkos initializes an OpenMP worker thread for every parallel_for,
            # and without binding the Linux scheduler will migrate that worker
            # across cores/sockets on a NUMA host. Post-migration, cache lines
            # first-touched on the original NUMA node become cross-socket fetches,
            # which on Cascade Lake-SP (Rivanna 'standard' partition) adds ~3-5x
            # latency to every DRAM access on TRITON's memory-bandwidth-bound
            # flux kernels. Empirically: missing the binding inflated sa_32's
            # serial wallclock relative to a properly-bound baseline (see
            # `library/docs/decisions/hhemt/LAYOUT_VERSION 8 fix per rank diff in performance aggregation.md`
            # for the empirically-verified sa_32 cumulative).
            env["OMP_NUM_THREADS"] = "1"
            env["OMP_PROC_BIND"] = "true"
            env["OMP_PLACES"] = "cores"

        # ----------------------------
        # MPI NIC policy (Frontier / Cray MPICH)
        # ----------------------------
        # Cray MPICH's default MPICH_OFI_NIC_POLICY=NUMA aborts MPI_Init when a rank's
        # allocated CPU set spans more than one NUMA domain. This happens whenever
        # cpus-per-task exceeds the allocatable cores per NUMA domain (14 on Frontier
        # with -S 8 core specialization), or when uneven task distribution across nodes
        # causes a rank to land on a non-NUMA-aligned core boundary.
        # BLOCK policy assigns NICs by rank block order instead of NUMA topology,
        # bypassing the confinement requirement entirely. It is safe for all MPI configs.
        # Empirically validated on Frontier 2026-02-27 (see
        # docs/planning/bugs/completed/empirical_frontier_srun_nic_policy_testing.md).
        if run_mode in ("mpi", "hybrid"):
            env["MPICH_OFI_NIC_POLICY"] = "BLOCK"

        # ----------------------------
        # GPU configuration
        # ----------------------------
        if run_mode == "gpu":
            # When running under srun with GPU binding flags (--gpus-per-task=1 in
            # "gpus" mode or --ntasks-per-gpu=1 in "gres" mode), SLURM automatically
            # sets CUDA_VISIBLE_DEVICES per task, remapping each task's assigned GPU
            # to local index 0. Setting CUDA_VISIBLE_DEVICES in the parent environment
            # would override this per-task remapping, causing all MPI ranks to see the
            # full GPU list and compete for GPU 0 (0% utilization on the others).
            # Only set device visibility explicitly when NOT using srun (local GPU execution).
            if not using_srun:
                gpu_list = ",".join(str(i) for i in range(n_gpus))  # type: ignore
                env["HIP_VISIBLE_DEVICES"] = gpu_list
                env["CUDA_VISIBLE_DEVICES"] = gpu_list
            env["OMP_NUM_THREADS"] = str(n_omp_threads)  # optional: threads per GPU
            env["OMP_PROC_BIND"] = "true"
            env["OMP_PLACES"] = "cores"
        # ----------------------------
        # Build command
        # ----------------------------
        # Phase-4 (4c): additional_modules is DI'd onto the system (retired off system_config).
        modules = self._scenario._system.additional_modules
        module_load_cmd = ""

        if modules:
            if verbose:
                print(f"loading modules {modules}")
            module_load_cmd = f"module load {modules}; "

        # Container Change 1: apptainer is module-only on some clusters (UVA Rivanna);
        # the seam must `module load` it in container mode or `srun … apptainer exec`
        # dies with `execve(): apptainer: No such file or directory` (the ambient
        # /opt/apptainer/current/bin PATH entry is a dead stub). Prepend the apptainer
        # module ahead of the build/runtime modules. Container-mode ONLY — native rows
        # never load it (native byte-identical).
        if (
            self._analysis.cfg_analysis.execution_environment == "container"
            and cspec is not None
            and cspec.apptainer_module
        ):
            module_load_cmd = f"module load {cspec.apptainer_module}; {module_load_cmd}"

        # ----------------------------
        # SWMM-specific command (no CFG, different structure)
        # ----------------------------
        if model_type == "swmm":
            # SWMM command: swmm5 input.inp report.rpt output.out
            inp_file = self._scenario.scen_paths.swmm_full_inp
            rpt_file = self._scenario.scen_paths.swmm_full_rpt_file
            out_file = self._scenario.scen_paths.swmm_full_out_file

            # SWMM is always CPU-only, no srun/mpirun needed
            launch_cmd_str = f"{exe} {inp_file} {rpt_file} {out_file}"

            # Build the full command
            env_exports = []
            for key, value in env.items():
                escaped_value = value.replace('"', '\\"')
                env_exports.append(f'export {key}="{escaped_value}"')
            env_export_str = "; ".join(env_exports)

            # Mirror the TRITON path's capability-guarded MPI-lib-first ordering
            # (see the triton/triton-swmm full_cmd assembly below). SWMM is serial
            # (no srun, no MPI) so the guard's mpicc test typically fails and the
            # else-branch keeps the prior conda-first behavior; the uniform code
            # path avoids a second ordering convention.
            # FI6: container host-env for the SWMM serial rung — bind-only (SWMM has
            # no MPI/GPU, so NO cray-mpich-abi / APPTAINERENV_LD_LIBRARY_PATH / srun
            # --mpi here). The in-container runswmm still needs analysis_dir bound to
            # read/write under it (interacts with FB2). "" in native mode (byte-identical).
            container_host_env_str = ""
            if self._analysis.cfg_analysis.execution_environment == "container" and cspec is not None:
                _adir = self._analysis.analysis_paths.analysis_dir
                _binds = [
                    *cspec.binds,
                    *system_directory_bind(self._analysis._system.cfg_system.system_directory, cspec.binds),
                    f"{_adir}:{_adir}",
                ]
                container_host_env_str = (
                    f'export APPTAINER_BIND="{",".join(_binds)}${{APPTAINER_BIND:+,$APPTAINER_BIND}}"; '
                )

            _container_mode = self._analysis.cfg_analysis.execution_environment == "container"
            # Derive the system/module MPI lib dir that actually holds the NEEDED
            # libmpi.so.40 / libmpi_cxx.so.40 sonames. The prior prefix+"/lib"
            # heuristic ($(dirname $(dirname mpicc))/lib) is WRONG on Debian/Ubuntu
            # multiarch: /usr/bin/mpicc -> /usr/lib, which EXISTS but holds no libmpi
            # (real libs live in /usr/lib/x86_64-linux-gnu/). Ask OpenMPI for its
            # libdir (-showme:libdirs), resolve the dev symlink libmpi.so to its real
            # .so.40.x file, and take that file's directory. Falls back to the prefix
            # heuristic when -showme is unsupported; the final guard only prepends when
            # libmpi.so.40 is actually present there (the falsifiable miss-detector).
            _mpi_derive = (
                "__MPI_LD=\"$(mpicc -showme:libdirs 2>/dev/null | awk '{print $1}')\"; "
                '__MPI_LIB="$(cd "$__MPI_LD" 2>/dev/null && '
                'dirname "$(readlink -f libmpi.so 2>/dev/null)" 2>/dev/null)"; '
                '[ -e "$__MPI_LIB/libmpi.so.40" ] || '
                '__MPI_LIB="$(dirname "$(dirname "$(command -v mpicc 2>/dev/null)")" 2>/dev/null)/lib"; '
            )
            if module_load_cmd:
                # mode-guard (M-7): the in-container loader ignores host LD_LIBRARY_PATH
                post_module_ld = (
                    ""
                    if _container_mode
                    else (
                        f"{_mpi_derive}"
                        'if [ -n "$(command -v mpicc 2>/dev/null)" ] && [ -e "$__MPI_LIB/libmpi.so.40" ]; then '
                        'export LD_LIBRARY_PATH="$__MPI_LIB:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"; '
                        'else export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"; fi; '
                    )
                )
            else:
                # Local / no-module path: a triton.exe built with the system mpic++
                # links the SYSTEM OpenMPI. The static ld_segments above prepend
                # ${CONDA_PREFIX}/lib (needed for libstdc++ on HPC); on a local dev box
                # that shadows the system libmpi while libmpi_cxx stays on system ->
                # ABI split (ompi_mpi_errors_throw_exceptions undefined). Mirror the
                # module branch's MPI-lib-first ordering so the system MPI dir precedes
                # ${CONDA_PREFIX}/lib whenever a system mpicc resolves AND its real
                # libmpi.so.40 dir is found; conda lib stays second so libstdc++ still
                # wins. No-op when no mpicc / no real MPI dir (falls back to conda-first).
                post_module_ld = (
                    ""
                    if _container_mode
                    else (
                        f"{_mpi_derive}"
                        'if [ -n "$(command -v mpicc 2>/dev/null)" ] && [ -e "$__MPI_LIB/libmpi.so.40" ]; then '
                        'export LD_LIBRARY_PATH="$__MPI_LIB:${LD_LIBRARY_PATH}"; fi; '
                    )
                )
            full_cmd = f"{env_export_str}; {module_load_cmd}{container_host_env_str}{post_module_ld}{launch_cmd_str}"
            cmd = [
                "bash",
                "-lc",
                full_cmd,
            ]

            # SWMM doesn't have checkpoint support, so pickup_where_leftoff doesn't apply
            # Return immediately with the command
            return cmd, env, model_logfile, 0

        # ----------------------------
        # TRITON/TRITON-SWMM command building
        # ----------------------------

        # CRITICAL VALIDATION: Verify SLURM allocation matches configuration requirements
        # This prevents infinite hangs when SLURM allocates fewer CPUs than configured.
        # For multi-node jobs SLURM_CPUS_ON_NODE reflects only one node's CPUs; the
        # correct total is SLURM_NTASKS × SLURM_CPUS_PER_TASK across all allocated nodes.
        #
        # NOTE: This check is only valid for batch_job mode where each SLURM job is
        # purpose-sized to one simulation. In 1_job_many_srun_tasks mode, SLURM_NTASKS
        # reflects the parent job's task count (e.g. 8 for --gres=gpu:8), NOT the
        # per-srun-step budget. Individual srun steps can use any portion of the full
        # exclusive node allocation (SLURM_CPUS_ON_NODE × SLURM_JOB_NUM_NODES).
        if using_srun and "SLURM_JOB_ID" in os.environ and multi_sim_run_method != "1_job_many_srun_tasks":
            slurm_cpus_on_node = int(os.environ.get("SLURM_CPUS_ON_NODE", 0))
            slurm_ntasks = int(os.environ.get("SLURM_NTASKS", 0))
            slurm_cpus_per_task = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))

            # Calculate what we expect vs what SLURM allocated.
            # Use NTASKS × CPUS_PER_TASK as the total — this is correct for both
            # single-node (NTASKS × CPT == CPUS_ON_NODE) and multi-node jobs
            # (NTASKS × CPT > CPUS_ON_NODE, spread across multiple nodes).
            expected_cpus = n_mpi_procs * n_omp_threads if run_mode != "gpu" else n_gpus * n_omp_threads
            slurm_allocated = slurm_ntasks * slurm_cpus_per_task

            # Raise ONLY on an AUTHORITATIVE positive read. Under the Snakemake slurm
            # executor the 2-node MPI rows run in the slurm-jobstep MPI branch (no srun
            # wrap), so the runner inherits the raw batch env where the submit-side
            # --ntasks-per-gpu/--ntasks-per-node (derived forms) never export a resolved
            # SLURM_NTASKS -> os.environ.get("SLURM_NTASKS", 0) reads 0 even though the
            # per-rule sbatch was correctly sized. A slurm_allocated of 0 means the env
            # expresses no task budget here, NOT that 0 CPUs were allocated; raising on it
            # false-fires and blocks a correctly-allocated 2-node sim. This mirrors the GPU
            # guard below, which already raises only on `allocated_gpus > 0`. A genuine
            # under-allocation still presents SLURM_NTASKS >= 1 (a 0-task job never runs a
            # shell), so `0 < slurm_allocated < expected_cpus` still catches it.
            if 0 < slurm_allocated < expected_cpus:
                error_msg = (
                    f"\n{'=' * 80}\n"
                    f"SLURM RESOURCE ALLOCATION MISMATCH DETECTED\n"
                    f"{'=' * 80}\n"
                    f"Configuration requests: {expected_cpus} CPUs\n"
                    f"  - MPI ranks: {n_mpi_procs}\n"
                    f"  - OMP threads per rank: {n_omp_threads}\n"
                    f"  - Total: {n_mpi_procs} × {n_omp_threads} = {expected_cpus} CPUs\n"
                    f"\n"
                    f"SLURM actually allocated: {slurm_allocated} CPUs\n"
                    f"  - SLURM_NTASKS × SLURM_CPUS_PER_TASK: {slurm_ntasks} × {slurm_cpus_per_task} = {slurm_allocated}\n"
                    f"  - SLURM_CPUS_ON_NODE (single-node view): {slurm_cpus_on_node}\n"
                    f"  - SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID')}\n"
                    f"\n"
                    f"CONSEQUENCE:\n"
                    f"  If we proceed, srun will request {expected_cpus} CPUs but only\n"
                    f"  {slurm_allocated} are allocated, causing an infinite hang while waiting\n"
                    f"  for resources that will never become available.\n"
                    f"\n"
                    f"{'=' * 80}\n"
                )
                raise RuntimeError(
                    f"SLURM allocated {slurm_allocated} CPUs but configuration requires "
                    f"{expected_cpus} CPUs. Cannot proceed to avoid infinite hang. "
                    f"{error_msg}"
                )

        # NOTE: Same reasoning as CPU check above — in 1_job_many_srun_tasks mode,
        # SLURM_GPUS_ON_NODE/SLURM_JOB_GPUS reflect the parent job's per-node GPU count
        # (e.g. 8), not the total pool available to srun steps. A 2-node GPU sim requesting
        # 16 GPUs is valid against a 64-GPU pool even though the parent env reports 8.
        if (
            using_srun
            and "SLURM_JOB_ID" in os.environ
            and run_mode == "gpu"
            and multi_sim_run_method != "1_job_many_srun_tasks"
        ):
            allocated_gpus = _parse_slurm_allocated_gpus(os.environ)
            expected_gpus = int(n_gpus or 0)
            if allocated_gpus > 0 and allocated_gpus < expected_gpus:
                raise RuntimeError(
                    f"\n{'=' * 80}\n"
                    f"SLURM GPU ALLOCATION MISMATCH DETECTED\n"
                    f"{'=' * 80}\n"
                    f"Configuration requires {expected_gpus} GPUs but SLURM allocation "
                    f"appears to provide {allocated_gpus}.\n"
                    f"Refusing launch to avoid hanging/oversubscription.\n"
                    f"Inspect SLURM_GPUS/SLURM_GPUS_ON_NODE/SLURM_JOB_GPUS and sbatch request.\n"
                    f"{'=' * 80}\n"
                )

        # Container mode (UVA host-OpenMPI): the OUTER srun carries --mpi={srun_mpi}
        # (e.g. pmix) so the in-container OpenMPI wires up via SLURM PMIx. Frontier
        # (srun_mpi=None) emits nothing (Cray-PALS path; ADR-3's "never --mpi=pmix"
        # holds). Precomputed as a token (NOT an inline conditional in the implicit-
        # concat srun tuple, which would be a syntax error); "" in native mode so the
        # srun string stays byte-identical.
        _mpi_flag = f"--mpi={cspec.srun_mpi} " if (cspec is not None and cspec.srun_mpi) else ""

        if run_mode != "gpu":
            if using_srun:
                launch_cmd_str = (
                    f"srun "
                    f"-N {n_nodes_per_sim} "
                    f"--ntasks={n_mpi_procs} "
                    f"--cpus-per-task={n_omp_threads} "
                    # "--exclusive "
                    "--cpu-bind=cores "
                    "--overlap "  # Required in batch_job mode: allows srun step to share
                    # the parent job's allocation rather than requesting exclusive sub-step
                    # resources. Without this, srun blocks waiting for resources that are
                    # already consumed by the batch script process, causing hangs/timeouts.
                    "--kill-on-bad-exit=1 "  # If any task exits non-zero (e.g. partial PMI
                    # launch failure where only remote-node tasks fail), srun sends SIGKILL to
                    # all surviving tasks immediately rather than waiting for them to exit
                    # naturally. Prevents indefinite hangs when tasks are blocked at MPI_Init
                    # PMI_Barrier waiting for failed peers (observed as 118-min hang in Run 7).
                    f"{_mpi_flag}"
                    f"{exe} {cfg}"
                )
            elif run_mode in ("serial", "openmp"):
                launch_cmd_str = f"{exe} {cfg}"
            elif run_mode in ("mpi", "hybrid"):
                # PRRTE sizes its slot pool from SLURM_TASKS_PER_NODE, never from the
                # CPUs the step holds. Under the pytest harness's
                # `--ntasks=1 --cpus-per-task=8` that reads 1, so `mpirun -np 2` is
                # refused ("There are not enough slots available in the system") while
                # the identical command succeeds off-scheduler, where PRRTE falls back
                # to core count. Neither a hostfile `slots=` clause nor `--host N` can
                # override it: under an RM the allocation wins, and `--host` then fails
                # one stage later in the mapper (rc=213). So correct the value PRRTE
                # reads. Do NOT reach for `--map-by :OVERSUBSCRIBE` -- it also clears
                # the refusal, but it leaves every rank UNBOUND, and because
                # OMP_PROC_BIND/OMP_PLACES are set above for every mode, two unbound
                # ranks each map their OpenMP master to place 0 and land on ONE shared
                # core. Measured, OMPI 5.0.10, 2 ranks x 1 thread: OVERSUBSCRIBE gives
                # both ranks affinity {0,1}; this form gives {0,1} and {2,3}, identical
                # to off-scheduler. Adding `--bind-to none` changes NOTHING -- PRRTE
                # already unbinds an oversubscribed job.
                # The guard tests BOTH spellings deliberately: PRRTE's SLURM RAS
                # activates on SLURM_JOBID and NOT on SLURM_JOB_ID (measured: the
                # former alone reproduces the refusal, the latter alone does not), so
                # keying on SLURM_JOB_ID alone would skip the fix in an environment
                # where the problem is present.
                if "SLURM_JOBID" in os.environ or "SLURM_JOB_ID" in os.environ:
                    _cpu_budget = int(os.environ.get("SLURM_NTASKS", 0)) * int(
                        os.environ.get("SLURM_CPUS_PER_TASK", 1)
                    )
                    _needed = n_mpi_procs * n_omp_threads
                    # `0 <` mirrors the srun-arm guard above: under the slurm-jobstep
                    # MPI branch SLURM_NTASKS can read 0 in a correctly-sized job, and
                    # raising on that would block a sim whose allocation is fine.
                    if 0 < _cpu_budget < _needed:
                        raise RuntimeError(
                            f"MPI launch refused: this SLURM step holds {_cpu_budget} CPU(s) "
                            f"(SLURM_NTASKS x SLURM_CPUS_PER_TASK) but run_mode "
                            f"'{run_mode}' requires {n_mpi_procs} rank(s) x "
                            f"{n_omp_threads} thread(s) = {_needed}. Raise the allocation "
                            "or lower the compute config; proceeding would oversubscribe "
                            "and make any timing this run produces meaningless."
                        )
                    # Declare the real slot pool. `or n_mpi_procs` covers the
                    # SLURM_NTASKS=0 case the guard above deliberately tolerates.
                    # PRRTE keeps its own over-request refusal against this value.
                    env["SLURM_TASKS_PER_NODE"] = str(_cpu_budget or n_mpi_procs)
                launch_cmd_str = f"mpirun -np {str(n_mpi_procs)} {exe} {cfg}"
        elif run_mode == "gpu":
            if using_srun:
                # GPU-to-task binding depends on the batch allocation mode.
                # The two SLURM GPU flag families are mutually exclusive:
                #
                # - "gpus" mode (Frontier): --gpus-per-task=1
                #   Assigns 1 GPU per task. Required because --ntasks-per-gpu=1
                #   expands task count to match full-node GPU count on exclusive
                #   allocations (gres.c:_handle_ntasks_per_tres_step).
                #
                # - "gres" mode (UVA): --ntasks-per-gpu=1
                #   Same flag family as the Snakemake executor's sbatch
                #   --ntasks-per-gpu=1 (submit_string.py:79-91). This is the
                #   SOLE task-count driver for the gres branch: the gres srun
                #   below carries NO explicit --ntasks, so --ntasks-per-gpu=1
                #   is load-bearing — it expands to one task per inherited GPU
                #   (triggers tres_bind=gres/gpu:single:1). --gpus-per-task
                #   MUST NOT be used here — it conflicts with the inherited
                #   SLURM_NTASKS_PER_GPU (fatal in SLURM).
                #
                # See: completed/2026-02-28_gpu-mpi-scaling-machine-file-override.md
                #      bugs/2026-03-01_fix_gpu_srun_flag_conflict.md
                # Phase-4 (4c): alloc flavor is hpc_system_config.gpu_allocation_flavor
                # (system-level), reachable via the analysis; retired off system_config.
                _cfg_hpc = self._analysis.cfg_hpc_system
                gpu_alloc_mode = (
                    _cfg_hpc.gpu_allocation_flavor
                    if (_cfg_hpc is not None and _cfg_hpc.gpu_allocation_flavor is not None)
                    else "gpus"
                )
                if gpu_alloc_mode == "gpus":
                    gpu_bind_flag = "--gpus-per-task=1 "
                    # Frontier: --gpus-per-task=1 honors --ntasks=N exactly; the
                    # whole-node parent would otherwise over-expand --ntasks-per-gpu
                    # to the full node GPU count, so clamp with explicit --ntasks.
                    ntasks_flag = f"--ntasks={n_gpus} "
                elif n_gpus >= 2 and multi_sim_run_method != "1_job_many_srun_tasks":
                    # UVA gres mode, MULTI-GPU (n_gpus>=2, single- OR multi-node) under
                    # the Snakemake slurm executor (per-rule jobstep). The per-rule sbatch
                    # is Gotcha-32-routed (gres_multi_gpu = gpus_total>=2 -> mpi=True,
                    # tasks=N, tasks_per_gpu=0), so it carries an explicit --ntasks=N and
                    # SUPPRESSES --ntasks-per-gpu. Keyed on n_gpus (NOT node count) because
                    # that routing is node-count-agnostic: a single-node strict-subset
                    # multi-GPU sim (g2: n_gpus=2, n_nodes=1) hits the identical
                    # unset-SLURM_GPUS* + no-inherited-ntasks-per-gpu state, so an inner
                    # --ntasks-per-gpu=1 fails "_handle_ntasks_per_tres_step: ntasks_per_tres
                    # was specified, but there was ... no GPU specification". Because the
                    # parent carries NO --ntasks-per-gpu here, --gpus-per-task=1 does NOT
                    # conflict (unlike the single-node else below) — mirror the Frontier
                    # gpus form: explicit --ntasks + explicit per-rank GPU binding.
                    # Surfaced on UVA gpu-a100/a6000 g2 by the [Q8] cross-hardware run
                    # (2026-07-18); the multi-node case (2026-07-01, job 16708076) is the
                    # same root cause. Single-node 1-GPU gres, 1_job_many_srun_tasks, and
                    # the local in-allocation path keep the --ntasks-per-gpu=1 form below
                    # (the 2026-05-23 collision constraint is still binding there — their
                    # parent carries --ntasks-per-gpu).
                    gpu_bind_flag = "--gpus-per-task=1 "
                    ntasks_flag = f"--ntasks={n_gpus} "
                else:
                    gpu_bind_flag = "--ntasks-per-gpu=1 "
                    # UVA gres mode, SINGLE-NODE (and 1_job_many_srun_tasks): the parent
                    # batch step holds exactly N requested GPUs and carries
                    # --ntasks-per-gpu=1. Dropping the explicit --ntasks lets the step
                    # inherit ntasks_per_tres=1 and expand to N (one task per inherited
                    # GPU). An explicit --ntasks=N collides with the 1-task batch step
                    # ("More processors requested than permitted"). Empirically confirmed
                    # on UVA gpu-a6000 (2026-05-23).
                    ntasks_flag = ""
                launch_cmd_str = (
                    f"srun "
                    f"-N {n_nodes_per_sim} "
                    f"{ntasks_flag}"
                    f"--cpus-per-task={n_omp_threads} "
                    f"{gpu_bind_flag}"
                    "--cpu-bind=cores "
                    "--overlap "  # See note above on --overlap in batch_job mode.
                    "--kill-on-bad-exit=1 "  # See note above on --kill-on-bad-exit=1.
                    f"{_mpi_flag}"
                    f"{exe} {cfg}"
                )
            else:
                launch_cmd_str = f"{exe} {cfg}"
        else:
            raise ValueError(f"Unknown run_mode: {run_mode}")

        # Build the full command with explicit environment variable exports.
        env_exports = []
        for key, value in env.items():
            escaped_value = value.replace('"', '\\"')
            env_exports.append(f'export {key}="{escaped_value}"')
        env_export_str = "; ".join(env_exports)
        # Order LD_LIBRARY_PATH so the ACTIVE MPI module's lib dir precedes
        # ${CONDA_PREFIX}/lib, which precedes the prior LD_LIBRARY_PATH. Rationale:
        # triton.exe is compiled against the cluster MODULE OpenMPI (SLURM-PMI
        # integrated). Its baked RUNPATH lists ${CONDA_PREFIX}/lib FIRST, and the
        # prior conda-only re-prepend re-asserted conda first — so under bare srun
        # triton.exe loaded conda's OpenMPI (libmpi.so.40/libopen-pal.so.80), which
        # cannot do SLURM PMI rank wireup → multi-GPU rank 1 dies ~49s. Putting the
        # module MPI lib dir first makes libmpi/libopen-pal resolve to the module
        # (PMI works); ${CONDA_PREFIX}/lib stays present so libstdc++ still resolves
        # to conda (GLIBCXX_3.4.31 for libgdal/libmuparser; module MPI dirs carry no
        # libstdc++). The MPI lib dir is derived generically from the resolved mpicc
        # wrapper — never hardcoded. The guard (mpicc resolves AND <prefix>/lib
        # exists) makes this a byte-identical no-op on hosts with no MPI compiler
        # wrapper (Frontier cray-mpich without mpicc, SWMM serial, local), so those
        # paths keep the prior conda-first behavior. Empirically confirmed by a live
        # 2-GPU salloc test on UVA Rivanna (2026-05-24): ldd flipped libmpi->module,
        # libstdc++->conda; the 2-rank srun ran 9.5+ min vs the prior 49s crash.
        # --- container-mode host-side injector env (ADR-3; OUTSIDE the wrap) ---
        # Apptainer does NOT import the host LD_LIBRARY_PATH; the host Cray-MPICH ABI
        # bind is realized by exporting APPTAINERENV_LD_LIBRARY_PATH (Apptainer PREPENDS
        # it inside the container ahead of the SIF-baked /opt/mpich/lib). Emitted as a
        # distinct segment so native stays byte-identical ("" in native mode).
        container_host_env_str = ""
        if self._analysis.cfg_analysis.execution_environment == "container" and cspec is not None:
            parts = []
            # Pre-exec modules FIRST (Frontier production: OLCF apptainer-enable-mpi/-gpu
            # + olcf-container-tools — they bind the open-ended host MPI+ROCm+compiler-
            # runtime closure so it need not be hand-enumerated into containlibs). They
            # must precede cray-mpich-abi + the APPTAINERENV exports (validated probe
            # job 4898044). Empty list => byte-identical to the prior emission.
            for _m in cspec.pre_exec_modules:
                parts.append(f"module load {_m} 2>/dev/null")
            if cspec.cray_mpich_abi_module:
                parts.append("module load cray-mpich-abi 2>/dev/null")
            if cspec.apptainerenv_ld_library_path:
                # SHELL-TEMPLATE in double quotes so ${CRAY_MPICH_DIR} etc. expand at runtime.
                parts.append(f'export APPTAINERENV_LD_LIBRARY_PATH="{cspec.apptainerenv_ld_library_path}"')
            if cspec.containlibs:
                parts.append(f'export APPTAINER_CONTAINLIBS="{",".join(cspec.containlibs)}"')
            _adir = self._analysis.analysis_paths.analysis_dir
            _binds = [
                *cspec.binds,
                *system_directory_bind(self._analysis._system.cfg_system.system_directory, cspec.binds),
                f"{_adir}:{_adir}",
            ]
            parts.append(f'export APPTAINER_BIND="{",".join(_binds)}${{APPTAINER_BIND:+,$APPTAINER_BIND}}"')
            container_host_env_str = "; ".join(parts) + "; "

        _container_mode = self._analysis.cfg_analysis.execution_environment == "container"
        # See the SWMM-path comment above: NEEDED-soname-accurate MPI lib-dir
        # derivation (multiarch-correct), reused for the module and local branches.
        _mpi_derive = (
            "__MPI_LD=\"$(mpicc -showme:libdirs 2>/dev/null | awk '{print $1}')\"; "
            '__MPI_LIB="$(cd "$__MPI_LD" 2>/dev/null && dirname "$(readlink -f libmpi.so 2>/dev/null)" 2>/dev/null)"; '
            '[ -e "$__MPI_LIB/libmpi.so.40" ] || '
            '__MPI_LIB="$(dirname "$(dirname "$(command -v mpicc 2>/dev/null)")" 2>/dev/null)/lib"; '
        )
        if module_load_cmd:
            # mode-guard (M-7): the in-container loader ignores host LD_LIBRARY_PATH
            post_module_ld = (
                ""
                if _container_mode
                else (
                    f"{_mpi_derive}"
                    'if [ -n "$(command -v mpicc 2>/dev/null)" ] && [ -e "$__MPI_LIB/libmpi.so.40" ]; then '
                    'export LD_LIBRARY_PATH="$__MPI_LIB:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"; '
                    'else export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"; fi; '
                )
            )
        else:
            # Local / no-module path — mirror the module branch's MPI-lib-first
            # ordering for a system-mpic++-built triton.exe. Conda lib stays second
            # (already first in the static ld_segments) so libstdc++ GLIBCXX_3.4.31
            # still wins; system MPI wins for libmpi/libmpi_cxx. No-op when no mpicc.
            post_module_ld = (
                ""
                if _container_mode
                else (
                    f"{_mpi_derive}"
                    'if [ -n "$(command -v mpicc 2>/dev/null)" ] && [ -e "$__MPI_LIB/libmpi.so.40" ]; then '
                    'export LD_LIBRARY_PATH="$__MPI_LIB:${LD_LIBRARY_PATH}"; fi; '
                )
            )
        full_cmd = f"{env_export_str}; {module_load_cmd}{container_host_env_str}{post_module_ld}{launch_cmd_str}"
        cmd = [
            "bash",
            "-lc",
            full_cmd,
        ]

        # ----------------------------
        # Safety checks
        # ----------------------------
        if run_mode in ("mpi", "hybrid") and n_mpi_procs < 1:  # type: ignore
            raise ValueError("n_mpi_procs must be >= 1")

        if run_mode in ("openmp", "hybrid") and n_omp_threads < 1:  # type: ignore
            raise ValueError("n_omp_threads must be >= 1")

        if run_mode == "gpu" and n_gpus < 1:  # type: ignore
            raise ValueError("n_gpus must be >= 1")
        return cmd, env, model_logfile, sim_start_reporting_tstep

    def _create_subprocess_sim_run_launcher(
        self,
        pickup_where_leftoff: bool = False,
        verbose: bool = False,
        model_type: str = "tritonswmm",
    ):
        """
        Create a launcher function that runs simulation in a subprocess (non-blocking).

        This isolates the simulation to a separate process, avoiding potential
        state conflicts when running multiple simulations concurrently.

        The launcher function:
        1. Records initial simulation metadata in simlog
        2. Executes the simulation subprocess
        3. Returns the Popen object (does NOT wait for completion)
        4. Caller is responsible for waiting and updating simlog

        This non-blocking pattern allows multiple simulations to run concurrently
        when used with process polling in the concurrent execution methods.

        Parameters
        ----------
        pickup_where_leftoff : bool
            If True, resume simulation from last checkpoint if available
        verbose : bool
            If True, print progress messages
        model_type : str
            One of: "triton", "tritonswmm", "swmm" (default: "tritonswmm")

        Returns
        -------
        tuple
            (launcher_func, metadata_dict) where:
            - launcher_func: callable that returns (proc, start_time, sim_logfile)
            - metadata_dict: dict with simulation metadata for logging
        """
        import os

        event_iloc = self._scenario.event_iloc
        sim_logfile = self._scenario.log.logfile.parent / f"sim_run_{event_iloc}.log"

        # Build command - always use direct Python execution (no srun)
        cmd = [
            f"{self._analysis._python_executable}",
            "-m",
            "hhemt.run_simulation_runner",
            "--event-iloc",
            str(event_iloc),
            "--analysis-config",
            str(self._analysis.analysis_config_yaml),
            "--system-config",
            str(self._scenario._system.system_config_yaml),
            "--model-type",
            model_type,
        ]

        # Add optional flags
        if pickup_where_leftoff:
            cmd.append("--pickup-where-leftoff")

        def launcher():
            """
            Execute simulation in a subprocess (non-blocking).

            Returns
            -------
            tuple
                (proc, start_time, sim_logfile) where proc is the Popen object
            """
            if verbose:
                print(
                    f"[Scenario {event_iloc}] Launching subprocess: {' '.join(cmd)}",
                    flush=True,
                )

            start_time = time.time()
            lf = open(sim_logfile, "w")
            proc = subprocess.Popen(
                cmd,
                stdout=lf,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
            )

            # Return process handle and metadata (do NOT wait)
            return proc, start_time, sim_logfile, lf

        def finalize_sim(proc, start_time, sim_logfile, lf):
            """
            Wait for simulation to complete and update simlog.

            Parameters
            ----------
            proc : subprocess.Popen
                The process object
            start_time : float
                Time when process was started
            sim_logfile : Path
                Path to simulation log file
            lf : file object
                Open log file handle
            """
            # Wait for subprocess to complete
            rc = proc.wait()
            lf.close()

            if verbose:
                if rc == 0:
                    print(
                        f"[Scenario {event_iloc}] Subprocess completed successfully",
                        flush=True,
                    )
                else:
                    print(
                        f"[Scenario {event_iloc}] Subprocess failed with return code {rc}",
                        flush=True,
                    )

            if rc != 0:
                # Log the error
                if sim_logfile.exists():
                    with open(sim_logfile) as f:
                        error_output = f.read()
                    if verbose:
                        print(
                            f"[Scenario {event_iloc}] Subprocess output:\n{error_output}",
                            flush=True,
                        )

            end_time = time.time()
            elapsed = end_time - start_time

            if verbose:
                completed = self.model_run_completed(model_type)
                status = "completed" if completed else "did not finish"
                print(
                    f"[Scenario {event_iloc}] Simulation {status}, elapsed={elapsed:.1f}s",
                    flush=True,
                )

        return launcher, finalize_sim


def return_the_reporting_step_from_a_cfg(f_cfg: Path):
    step = int(f_cfg.name.split("_")[-1].split(".")[0])
    return step
