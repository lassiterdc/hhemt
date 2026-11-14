"""WHEN a cached tree may be handed out: the fixture completion-marker state machine.

A tree's builder owns the claim that the tree is DONE. The toolkit's own workflow flags
(``e_consolidate_complete.flag``, ``f_consolidate_experiment_complete.flag``) attest that a
workflow STAGE ran; they do not attest that the artifacts a fixture's consumers require are
present, and gating on them let an incomplete tree be re-read indefinitely. Measured
2026-08-23: ``synth_sensitivity`` carried no ``sensitivity_datatree.zarr`` for 13 hours
across three cluster runs, and ten tests reported as FAILED for a precondition their own
fixture had not met.

THE ARMS, named by :func:`marker_state`'s result:

  * ``absent``      -- no readable marker. Rebuild.
  * ``stale``       -- a marker written under a different tier-1 identity (see
                       :mod:`hhemt.cache.keys`). Rebuild.
  * ``unsatisfied`` -- a current marker whose payload is missing OR no longer matches the
                       (size, sha256) the marker recorded. Warn, then rebuild
                       (:func:`warn_unsatisfied`).
  * ``complete``    -- hand the tree out.

ARM 0, :func:`finish_build`, ends every rebuild and is what makes the others terminate: it
raises if the payload is still missing, so a rebuild ends in a marker that holds or in an
error, never in another rebuild.

WHY ``payload_digest`` IS COMPARED SEPARATELY FROM THE REST OF THE IDENTITY. It is a term of
the tier-1 key, because a payload's bytes are part of what the tree is. It is NOT part of the
IDENTITY comparison here, and the split is what makes two different faults report as two
different arms: a truncated payload leaves a file that still EXISTS, so an existence-only
payload check reads it as intact; folded into the identity comparison it would report
``stale`` and send a reader looking for a builder change that never happened. Compared on
its own it reports ``unsatisfied``, whose warning already says the tree may have been damaged
after the marker was written.
"""

from __future__ import annotations

import functools
import json
import warnings
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from hhemt.cache.keys import payload_digest as _measure_payload_digest

#: Marker path relative to the analysis directory it attests. Relative so a tree moved
#: between roots keeps its own claim with it.
FIXTURE_MARKER_REL = "_status/_fixture_complete.json"

#: The handover path for a pytest session's run-time cache facts, which the suite
#: aggregator's disclosure block is built from. Spelled HERE because both ends import it:
#: the repo-root conftest's session-finish hook writes the file, and the chunk runner sets
#: the variable and reads the file back into the chunk status. A second spelling would let
#: the two ends disagree silently -- the writer would produce a file nobody reads, and an
#: absent disclosure and a session that reported nothing are the same bytes downstream.
CACHE_DISCLOSURE_ENV = "HHEMT_SUITE_CACHE_DISCLOSURE_OUT"

#: The identity member held BACK from the identity comparison and checked as payload
#: integrity instead. One name, referenced by both the comparison and the merge below, so a
#: second spelling of it cannot drift.
_PAYLOAD_TERM = "payload_digest"


def unmet_payload(analysis_dir: Path, payload: Iterable[str]) -> list[str]:
    """The payload entries that are absent under ``analysis_dir``, in payload order."""
    return [rel for rel in payload if not (analysis_dir / rel).exists()]


@functools.lru_cache(maxsize=8)
def _read_record(analysis_dir_str: str, mtime_ns: int, size: int) -> dict[str, Any] | None:
    """A parsed marker record, or ``None`` when it is unreadable or malformed.

    Cached on the file's ``(mtime_ns, size)`` because :func:`marker_state` may be asked the
    same question several times in one construction, and the cache key makes a rewritten
    marker a MISS rather than a stale hit. Unreadable and malformed both return ``None``:
    a marker written by a killed process is indistinguishable from a corrupt one, and both
    mean the same thing to a fixture.
    """
    try:
        return json.loads(Path(analysis_dir_str).joinpath(FIXTURE_MARKER_REL).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def read_marker(analysis_dir: Path) -> dict[str, Any] | None:
    """The parsed marker record under ``analysis_dir``, or ``None`` when there is none."""
    marker = analysis_dir / FIXTURE_MARKER_REL
    try:
        stat = marker.stat()
    except OSError:
        return None
    return _read_record(str(analysis_dir), stat.st_mtime_ns, stat.st_size)


def recorded_tier0_key(analysis_dir: Path) -> str | None:
    """The ``tier0_key`` a marker recorded, or ``None`` when no marker records one.

    ``None`` rather than ``""`` so the caller can tell "no marker, so nothing to compare"
    from "a marker whose tier-0 term is empty". The tier-0 regeneration decision is the only
    consumer, and it must not treat an absent marker as a mismatch -- that would pay a full
    DEM regeneration on the first construction of every fresh tree.
    """
    record = read_marker(analysis_dir)
    if record is None:
        return None
    provenance = record.get("provenance")
    if not isinstance(provenance, Mapping):
        return None
    value = provenance.get("tier0_key")
    return value if isinstance(value, str) else None


def marker_facts(analysis_dir: Path) -> dict[str, Any] | None:
    """What a marker says about its tree, or ``None`` when there is no readable marker.

    ``written_at`` is the marker file's own mtime, and it is the ONE datum that separates a
    tree a session ADOPTED from one it BUILT: :func:`finish_build` writes the marker LAST, so
    a rebuild refreshes it, while an adopted tree keeps the marker its original builder wrote.
    A disclosure that reported only "a marker exists" would report the same value for both.

    ``payload_present`` RE-MEASURES the payload the record asserts rather than repeating the
    record's claim. That distinction is the whole point of reporting an integrity verdict: an
    adopted tree was handed out because the marker read ``complete`` at hand-out, and a record
    copied forward would restate that verdict whatever has happened to the tree since.
    """
    record = read_marker(analysis_dir)
    if record is None:
        return None
    try:
        written_at = (analysis_dir / FIXTURE_MARKER_REL).stat().st_mtime
    except OSError:
        return None
    raw_payload = record.get("payload")
    payload = [str(p) for p in raw_payload] if isinstance(raw_payload, list) else []
    provenance = record.get("provenance")
    tier0 = provenance.get("tier0_key") if isinstance(provenance, Mapping) else None
    return {
        "written_at": written_at,
        "tier0_key": tier0 if isinstance(tier0, str) else None,
        "payload": payload,
        "payload_present": not unmet_payload(analysis_dir, payload),
    }


def _identity_only(provenance: Mapping[str, Any]) -> dict[str, Any]:
    """``provenance`` without the payload term, for the identity comparison."""
    return {key: value for key, value in provenance.items() if key != _PAYLOAD_TERM}


def _payload_matches(recorded: Any, live: Mapping[str, Any] | None) -> bool:
    """True when the marker's recorded payload digests match the live measurement.

    A caller that passes no live measurement opts out of this arm; the existence check still
    runs, so declining to measure cannot turn a missing payload into a pass.
    """
    if live is None:
        return True
    recorded_map = recorded if isinstance(recorded, Mapping) else {}
    return json.loads(json.dumps(recorded_map, sort_keys=True)) == json.loads(json.dumps(dict(live), sort_keys=True))


def marker_state(
    analysis_dir: Path,
    payload: Iterable[str],
    *,
    provenance: Mapping[str, Any],
    payload_digest: Mapping[str, Any] | None = None,
) -> str:
    """One of ``absent`` / ``stale`` / ``unsatisfied`` / ``complete``.

    ``provenance`` is passed in rather than computed here because what stamps a tree is a
    property of the caller's tree (see :func:`hhemt.cache.keys.builder_provenance`).
    ``payload_digest`` is the live ``{rel_path: (size, sha256)}`` measurement, and it is
    supplied separately because it is the ONE term that reports as damaged-tree rather than
    as different-builder -- see this module's docstring.
    """
    record = read_marker(analysis_dir)
    if record is None:
        return "absent"
    if _identity_only(record.get("provenance") or {}) != _identity_only(provenance):
        return "stale"
    if unmet_payload(analysis_dir, payload):
        return "unsatisfied"
    if not _payload_matches((record.get("provenance") or {}).get(_PAYLOAD_TERM), payload_digest):
        return "unsatisfied"
    return "complete"


def finish_build(
    analysis_dir: Path,
    payload: Iterable[str],
    fixture_name: str,
    *,
    provenance: Mapping[str, Any],
) -> None:
    """ARM 0 -- verify the builder's own postcondition, RAISE if it failed, else mark.

    Raising is deliberate and is what routes the failure correctly. A tree that cannot be
    built is a LOST PRECONDITION, not a failing test: raising here leaves this fixture out
    of the chunk's ``session_fixtures_ok``, which the suite aggregator classifies VOID
    ("expected session fixture(s) did not complete setup") instead of reporting the
    downstream consumers as FAILED. Returning quietly is what produced ten tests blamed for
    a fixture's failure.

    The marker is written LAST and only here, so it can never attest a build that did not
    reach this point.

    The payload term is MEASURED HERE rather than taken from ``provenance``, and the
    measurement happens after the postcondition above has passed. A caller comparing a
    pre-build identity and stamping a post-build one would otherwise record the digest of a
    payload that did not exist yet, and every later session would read the tree as
    ``unsatisfied`` and pay a rebuild it did not need.
    """
    payload = tuple(payload)
    missing = unmet_payload(analysis_dir, payload)
    if missing:
        raise RuntimeError(
            f"{fixture_name}: build finished but its postcondition is UNMET -- missing "
            f"{missing} under {analysis_dir}. The tree is INCOMPLETE and must not be "
            "marked complete; consumers of this fixture would otherwise read it as done. "
            "This is a builder failure, not a test failure."
        )
    record_provenance = dict(provenance)
    record_provenance[_PAYLOAD_TERM] = _measure_payload_digest(analysis_dir, payload)
    marker = analysis_dir / FIXTURE_MARKER_REL
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps(
            {"provenance": record_provenance, "payload": list(payload)},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def warn_unsatisfied(analysis_dir: Path, payload: Iterable[str], fixture_name: str) -> None:
    """ARM 3 -- a marker that was earned once and no longer holds. Never silent."""
    warnings.warn(
        f"{fixture_name}: completion marker present but its payload is UNSATISFIED "
        f"(missing {unmet_payload(analysis_dir, payload)} under {analysis_dir}; a present "
        "entry whose recorded size/digest no longer matches is unsatisfied too). A prior "
        "build wrote a marker it no longer earns, or the tree was damaged after it was "
        "written. Rebuilding. This arm is deliberately UNBOUNDED -- it terminates through "
        "the postcondition raise, at the cost of the failing stage, because the workflow "
        "resumes from its own per-stage flags rather than rebuilding the tree.",
        stacklevel=2,
    )
