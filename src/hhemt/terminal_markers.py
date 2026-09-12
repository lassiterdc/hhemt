"""v2 sentinel terminal markers — the ONE implementation of the write/unlink half of the
``_status/_submitted -> _completed | _failed`` contract, importable by any runner.

Leaf module (no toolkit imports) so ``process_timeseries_runner``'s thin terminal-marker
wrapper can import it without pulling ``run_simulation_runner``'s module-level imports.
The sim runner's private ``_MarkerCtx``/``_write_failed_marker`` predate this module and
are byte-equivalent; unifying them onto these functions is a named follow-up.

Marker payloads are written atomically (temp + ``os.replace``) and carry
``status`` + ``finished_at`` on top of the caller's ``payload_base``. All writes are
no-ops when ``jobid`` is falsy (non-SLURM execution), matching the sim runner's guard.
"""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path


def rule_token_for(rule_name: str, event_id: str) -> str:
    """``{rule_name}_evt-{event_id}`` — byte-identical to ``run_simulation_runner``'s
    ``_rule_token`` for the multisim ``run_{model}`` rules (literal ``evt-``)."""
    return f"{rule_name}_evt-{event_id}"


def write_terminal_marker(
    status_dir: Path,
    rule_token: str,
    *,
    status: str,
    jobid: str | None,
    payload_base: dict | None = None,
) -> Path | None:
    """Write ``{status_dir}/_{status}/{rule_token}.json`` (``status`` is ``completed`` or
    ``failed``) atomically; return the path, or ``None`` when ``jobid`` is falsy."""
    if not jobid:
        return None
    if status not in ("completed", "failed"):
        raise ValueError(f"terminal marker status must be 'completed' or 'failed', got {status!r}")
    marker_dir = Path(status_dir) / f"_{status}"
    marker_dir.mkdir(parents=True, exist_ok=True)
    marker = marker_dir / f"{rule_token}.json"
    payload = {
        **(payload_base or {}),
        "slurm_jobid": jobid,
        "status": status,
        "finished_at": datetime.datetime.now().isoformat(),
    }
    tmp = marker.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, marker)
    return marker


def unlink_submitted_sentinel(status_dir: Path, rule_token: str) -> None:
    """Remove ``{status_dir}/_submitted/{rule_token}.json`` if present (the queued ->
    submitted -> terminal handoff's last step)."""
    # EXEMPT-DU: status-flag
    (Path(status_dir) / "_submitted" / f"{rule_token}.json").unlink(missing_ok=True)
