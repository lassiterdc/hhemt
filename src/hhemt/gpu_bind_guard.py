"""GPU-binding guard for srun-launched GPU simulations (L1 fail-fast, L2 detection, L3 watchdog).

Context (Diagnosis, 2026-09-21): SLURM 25.05.x Ticket 24862 lets an ``--overlap`` step receive
fewer GPUs than its job on a mixed-locality partial-node grant; ``per_task:1`` binding then
doubles ranks onto the step's lowest GPU ("Not enough gres to bind"). Prevention is the
``--overlap`` drop in run_simulation.py; this module is the enforcement + detection layer that
must hold whatever the launch form does.

Leaf module: stdlib only. Imported by run_simulation.py (emit), run_simulation_runner.py
(watchdog + consolidate) and analysis.py (df_status readers).
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Literal

#: Verdicts the per-scenario artifact carries. ``pass`` requires BOTH the L1 step count and a
#: positively-evaluated L2 distinct-device count; ``not_evaluated`` is L1-pass with L2 unable
#: to evaluate (bounded wait expired) and is NEVER a pass in validate_resource_usage.
GpuBindVerdict = Literal["pass", "short_step", "shared_device", "not_evaluated", "watchdog_bind_error"]
FAILING_VERDICTS: frozenset[str] = frozenset({"short_step", "shared_device", "watchdog_bind_error"})

#: The two SLURM stderr signatures of a short ``per_task`` step (src/interfaces/gres.c, 25.05.6:
#: ``_get_gres_per_task`` and the ``bit_ffs`` fallback). Unconditional on a short step.
BIND_ERROR_SIGNATURES: tuple[str, ...] = (
    "Not enough gres to bind",
    "Binding to the first device in the allocation instead",
)

GUARD_EXIT_SHORT_STEP = 99
GUARD_EXIT_SHARED_DEVICE = 98

#: The wrapper text. Runs once per srun task on the compute-node HOST, before the solver is
#: exec'd. Config arrives via HHEMT_GB_* env vars (exported by the runner's env dict, propagated
#: to tasks by srun). Identical in every executable line to the script UVA gpu-a6000 job 20350116 ran (V-B/V-C,
#: 2026-09-21); one comment line differs (the emitter's name).
GUARD_SCRIPT_TEXT = r"""#!/usr/bin/env bash
# hhemt GPU-binding guard (L1 fail-fast + L2 detection). EMITTED BY THE TOOLKIT
# (hhemt.gpu_bind_guard) per attempt; do not edit by hand. Runs once per srun task,
# on the compute-node HOST, before the solver is exec'd.
#   argv: the solver launch (e.g. "apptainer exec ... triton.exe cfg", or "triton.exe cfg")
#   env : HHEMT_GB_EXPECTED_PER_NODE, HHEMT_GB_NTASKS, HHEMT_GB_RECORD_DIR, HHEMT_GB_BACKEND
set -u
expected="${HHEMT_GB_EXPECTED_PER_NODE:?}"
ntasks="${HHEMT_GB_NTASKS:?}"
rdir="${HHEMT_GB_RECORD_DIR:?}"
backend="${HHEMT_GB_BACKEND:-cuda}"
procid="${SLURM_PROCID:-0}"
localid="${SLURM_LOCALID:-0}"
step_gpus="${SLURM_GPUS_ON_NODE:-}"
mkdir -p "$rdir"
case "$backend" in
  cuda) dev="$(nvidia-smi --query-gpu=uuid,pci.bus_id --format=csv,noheader 2>/dev/null | head -n 1 | tr -d ' ')" ;;
  rocm) dev="$(rocm-smi --showuniqueid --csv 2>/dev/null | tail -n +2 | head -n 1 | tr -d ' ')" ;;
  *)    dev="" ;;
esac
tmp="$rdir/task_${procid}.json.tmp"
printf '{"procid": %s, "localid": %s, "host": "%s", "step_gpus_on_node": "%s", "device": "%s", "cvd": "%s"}\n' \
  "$procid" "$localid" "$(hostname)" "$step_gpus" "$dev" "${CUDA_VISIBLE_DEVICES:-${ROCR_VISIBLE_DEVICES:-}}" > "$tmp"
mv -f "$tmp" "$rdir/task_${procid}.json"
# L1: the step must expose the per-node GPU count this rank's node was configured for.
if [ -z "$step_gpus" ] || [ "$step_gpus" -lt "$expected" ]; then
  echo "[hhemt gpu-bind guard] task ${procid}: step exposes ${step_gpus:-unset} GPU(s) on $(hostname), expected ${expected}; refusing to launch (short_step)" >&2
  exit 99
fi
# L2 (detection): task 0 waits, bounded, for every task's record and counts distinct devices.
if [ "$procid" -eq 0 ]; then
  n=0
  for _ in $(seq 1 120); do
    n=$(ls "$rdir"/task_*.json 2>/dev/null | wc -l)
    [ "$n" -ge "$ntasks" ] && break
    sleep 0.5
  done
  if [ "$n" -ge "$ntasks" ]; then
    distinct=$(sed -n 's/.*"device": "\([^"]*\)".*/\1/p' "$rdir"/task_*.json | grep -v '^$' | sort -u | wc -l)
    printf '{"ntasks": %s, "records": %s, "distinct_devices": %s, "l2": "evaluated"}\n' "$ntasks" "$n" "$distinct" > "$rdir/l2.json"
    if [ "$distinct" -lt "$ntasks" ]; then
      echo "[hhemt gpu-bind guard] task 0: ${distinct} distinct device(s) across ${ntasks} tasks; refusing to launch (shared_device)" >&2
      exit 98
    fi
  else
    printf '{"ntasks": %s, "records": %s, "distinct_devices": null, "l2": "not_evaluated"}\n' "$ntasks" "$n" > "$rdir/l2.json"
  fi
fi
exec "$@"
"""  # noqa: E501 -- wrapper script bytes are load-bearing (measured on probe 20350116); never wrap


def write_guard_script(path: Path) -> Path:
    """Write GUARD_SCRIPT_TEXT to ``path`` (compare-and-write; mode 0o755). Returns ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_text() != GUARD_SCRIPT_TEXT:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(GUARD_SCRIPT_TEXT)
        os.replace(tmp, path)
    path.chmod(0o755)
    return path


def record_dir_for(sim_folder: Path, model_type: str) -> Path:
    """Per-attempt task records: ``{sim_folder}/_status/_gpu_bind/{model_type}/``."""
    return Path(sim_folder) / "_status" / "_gpu_bind" / str(model_type)


def artifact_path_for(sim_folder: Path, model_type: str) -> Path:
    """The consolidated per-scenario artifact df_status reads (path-only, mirrors model_logfile_for)."""
    return Path(sim_folder) / "_status" / f"gpu_bind_{model_type}.json"


def reset_record_dir(record_dir: Path) -> None:
    """Remove stale per-task records from a previous attempt (the wrapper writes fresh ones)."""
    if record_dir.is_dir():
        for p in record_dir.glob("*.json"):
            p.unlink()  # EXEMPT-DU: status-flag
        for p in record_dir.glob("*.json.tmp"):
            p.unlink()  # EXEMPT-DU: status-flag


def wait_with_bind_watchdog(
    proc: subprocess.Popen,
    model_logfile: Path,
    *,
    window_s: float = 120.0,
    poll_s: float = 1.0,
) -> tuple[int, bool]:
    """L3: wait for ``proc``; during the first ``window_s`` tail ``model_logfile`` for the SLURM
    short-step signatures and, on a hit, SIGTERM the process GROUP (mirrors the deterministic-kill
    harness) and return ``(rc, True)``. Otherwise ``(proc.wait() rc, False)``. Outside the window
    it is a plain wait. The runner opens the model log "w" before Popen, so a hit can only be this
    attempt's."""
    t0 = time.monotonic()
    fired = False
    while True:
        rc = proc.poll()
        if rc is not None:
            return rc, fired
        if not fired and (time.monotonic() - t0) < window_s:
            try:
                text = Path(model_logfile).read_text(errors="replace")
            except OSError:
                text = ""
            if any(sig in text for sig in BIND_ERROR_SIGNATURES):
                fired = True
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except ProcessLookupError:
                    pass
                rc = proc.wait()
                return rc, True
        time.sleep(poll_s)


def consolidate_records(
    record_dir: Path,
    *,
    expected_per_node: int,
    ntasks: int,
    artifact_path: Path,
    attempt_rc: int,
    watchdog_fired: bool,
    launch_form: str,
) -> dict:
    """Fold the wrapper's per-task records into one artifact and derive the verdict.

    Verdict precedence (A2): short_step (per-task RECORD) > shared_device > watchdog_bind_error
    (INFERRED from stderr; on a real short step both fire, so the record wins for label determinism)
    > not_evaluated > pass. All three named first are FAILING.
    Compare-and-write: byte-identical payloads do not bump mtime.
    """
    tasks: list[dict] = []
    for p in sorted(Path(record_dir).glob("task_*.json")):
        try:
            tasks.append(json.loads(p.read_text()))
        except (OSError, ValueError):
            continue
    l2: dict = {}
    try:
        l2 = json.loads((Path(record_dir) / "l2.json").read_text())
    except (OSError, ValueError):
        l2 = {}
    step_counts = {str(t.get("step_gpus_on_node", "")) for t in tasks}
    short = any((not c.isdigit()) or int(c) < expected_per_node for c in step_counts) if tasks else False
    distinct = l2.get("distinct_devices")
    if short or attempt_rc == GUARD_EXIT_SHORT_STEP:
        verdict = "short_step"
    elif attempt_rc == GUARD_EXIT_SHARED_DEVICE or (isinstance(distinct, int) and distinct < ntasks):
        verdict = "shared_device"
    elif watchdog_fired:
        verdict = "watchdog_bind_error"
    elif not tasks or not isinstance(distinct, int):
        verdict = "not_evaluated"
    else:
        verdict = "pass"
    payload = {
        "schema": 1,
        "verdict": verdict,
        "expected_gpus_per_node": expected_per_node,
        "ntasks": ntasks,
        "records": len(tasks),
        "step_gpus_on_node": sorted(step_counts),
        "distinct_devices": distinct,
        "l2": l2.get("l2"),
        "launch_form": launch_form,
        "attempt_rc": attempt_rc,
        "tasks": tasks,
    }
    text = json.dumps(payload, indent=1, sort_keys=True)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    if not artifact_path.exists() or artifact_path.read_text() != text:
        tmp = artifact_path.with_suffix(".json.tmp")
        tmp.write_text(text)
        os.replace(tmp, artifact_path)
    return payload


def read_gpu_bind_artifact(sim_folder: Path, model_type: str) -> dict | None:
    """Path-only reader for df_status. None when absent or unreadable (never a synthesized pass)."""
    p = artifact_path_for(sim_folder, model_type)
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def classify_from_signature(model_logfile: Path, *, launch_form_is_per_task: bool) -> str | None:
    """Retroactive classifier for rows that ran BEFORE the guard existed (Diagnosis claim 17).

    Applies ONLY when the launch form was the ``--ntasks=N --gpus-per-task=1`` (``per_task:1``)
    form, because the short-step signature is unconditional for that form. Labels describe the
    LAST EXECUTION (the model log is opened "w" per exec; Gotcha 71):
      ``short_step_by_signature``            the signature is present
      ``pass_by_negative_signature``          absent, log non-empty, per_task form
      None                                    not applicable / not measurable
    Never returns ``pass``.
    """
    if not launch_form_is_per_task:
        return None
    try:
        text = Path(model_logfile).read_text(errors="replace")
    except OSError:
        return None
    if not text.strip():
        return None
    if any(sig in text for sig in BIND_ERROR_SIGNATURES):
        return "short_step_by_signature"
    return "pass_by_negative_signature"


def find_runner_command_line(simlog_dir: Path, *, model_type: str, analysis_id: str, event_id: str) -> str | None:
    """The RECORDED launch form for a row: the ``Command:`` line of the runner log that the
    Snakemake rule wrote beside the model log (multisim: ``{model_type}_evt-{event_id}.log``;
    sensitivity: ``simulation_{analysis_id}_evt_{event_id with . and - -> _}.log``). None when
    no candidate exists or none carries a ``Command:`` line (=> the classifier is not applicable).
    Path-only: derives candidates from names, opens at most the matches, never instantiates a
    scenario."""
    d = Path(simlog_dir)
    sanitized = str(event_id).replace(".", "_").replace("-", "_")
    candidates = [
        d / f"{model_type}_evt-{event_id}.log",
        d / f"simulation_{analysis_id}_evt_{sanitized}.log",
    ]
    for c in candidates:
        try:
            text = c.read_text(errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            if "Command:" in line and "srun" in line:
                return line
    return None
