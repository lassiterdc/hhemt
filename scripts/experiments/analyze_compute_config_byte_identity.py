"""Read-only byte-identity analysis of completed synth compute-config sims (ssh rivanna).

Never writes to the cluster; keys on ``_status/c_run_*_complete.flag``. Final MH = max-index
``out_tritonswmm/bin/MH_<idx>_00.out``. Groups by md5; for resumed sims (grep the sim log for
'Resuming tritonswmm from hotstart') finds the divergence-onset dump vs the clean group.

Design (SE Flag 1 / Decision 1): ``_ssh`` is the SOLE cluster seam. Every function that touches
the cluster routes through it; the analysis logic lives in PURE functions that accept
already-fetched bytes/hashes so it is unit-testable without ``ssh`` (see
``tests/test_synth_compute_config_analysis.py``).
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import subprocess
from collections.abc import Mapping

# Default case root mirrors the Rivanna `system_directory` passed to
# synth_compute_config.clean_case()/resume_case() on Decision-4 project space.
# Override at runtime with `--case-root` when the case was materialized elsewhere
# (e.g., the factory's natural platformdirs cache root when no system_directory was
# passed) — read the actual path via `clean_case(...).system_directory`.
_DEFAULT_CASE_ROOT = "/project/quinnlab/dcl3nd/norfolk/synth_compute_config"


def _case_dirs(case_root: str) -> dict[str, str]:
    return {"clean": f"{case_root}/synth_cc_clean", "resume": f"{case_root}/synth_cc_resume"}


# ---------------------------------------------------------------------------
# Sole cluster seam
# ---------------------------------------------------------------------------
def _ssh(cmd: str) -> str:
    """Run one read-only command on Rivanna and return stdout. The ONLY cluster seam."""
    return subprocess.run(["ssh", "rivanna", cmd], capture_output=True, text=True, check=True).stdout


# ---------------------------------------------------------------------------
# Pure analysis functions (no ssh — unit-tested with synthetic byte fixtures)
# ---------------------------------------------------------------------------
def group_by_hash(blobs: Mapping[str, bytes]) -> dict[str, list[str]]:
    """Group sa_ids by the md5 of their already-fetched final-MH bytes.

    Returns ``{md5_hexdigest: [sa_id, ...]}`` with each sa_id list sorted. This is the
    byte-identity grouping that answers H1-H5 (which configs produce identical raw output).
    """
    groups: dict[str, list[str]] = {}
    for sa_id, blob in blobs.items():
        digest = hashlib.md5(blob).hexdigest()
        groups.setdefault(digest, []).append(sa_id)
    return {digest: sorted(ids) for digest, ids in groups.items()}


def final_mh_md5(mh_blobs: Mapping[int, bytes]) -> tuple[int, str]:
    """Select the MAX-index final-MH dump from already-fetched ``{idx: bytes}`` and md5 it.

    The final MH output is the highest-index ``MH_<idx>_00.out``; comparing finals across
    sims is the byte-identity test. Raises ValueError on an empty mapping.
    """
    if not mh_blobs:
        raise ValueError("no MH dumps provided")
    idx = max(mh_blobs)
    return idx, hashlib.md5(mh_blobs[idx]).hexdigest()


def compare_clean_vs_resume(clean_hashes: Mapping[str, str], resume_hashes: Mapping[str, str]) -> dict[str, list[str]]:
    """Group-membership comparison of two ``{sa_id: md5}`` maps.

    For every sa_id present in both, classify whether the resume run reproduced the clean
    run's final-MH bytes. ``diverged`` is the headline result: resume sims whose output
    differs from the clean (single-allocation) run for the same config.
    """
    common = sorted(set(clean_hashes) & set(resume_hashes))
    return {
        "matched": [sid for sid in common if clean_hashes[sid] == resume_hashes[sid]],
        "diverged": [sid for sid in common if clean_hashes[sid] != resume_hashes[sid]],
        "clean_only": sorted(set(clean_hashes) - set(resume_hashes)),
        "resume_only": sorted(set(resume_hashes) - set(clean_hashes)),
    }


def divergence_onset_index(clean_dump_hashes: list[str], resume_dump_hashes: list[str]) -> int | None:
    """First dump index where a resumed sim's per-dump md5 diverges from the clean sim.

    Pure: both inputs are ordered per-dump md5 lists (index 0 = first dump). Returns the
    index of the first mismatch, or None if every shared-index dump matches.
    """
    for i, (c, r) in enumerate(zip(clean_dump_hashes, resume_dump_hashes, strict=False)):
        if c != r:
            return i
    return None


# ---------------------------------------------------------------------------
# Cluster-fetch wrappers (route through _ssh; exercised on UVA in Phase 3)
# ---------------------------------------------------------------------------
def completed_sims(case_dir: str) -> list[tuple[str, str | None]]:
    """``[(sa_id, event_id)]`` for every completed sim (``c_run`` flag in ``{case_dir}/_status``).

    Sensitivity flag form is ``c_run_{model}_sa-{sa_id}_evt-{event_id}_complete.flag``
    (e.g. ``c_run_tritonswmm_sa-gpu_0_r1_evt-event_index.0_complete.flag``), so the bare
    middle must be split on ``_sa-`` and ``_evt-`` to recover the real ``sa_id`` + ``event_id``.
    A legacy multisim flag (``c_run_{sa_id}_complete.flag``) yields ``(sa_id, None)``.
    """
    out = _ssh(f"ls {case_dir}/_status 2>/dev/null || true")
    sims: list[tuple[str, str | None]] = []
    for line in out.splitlines():
        name = line.strip()
        if name.startswith("c_run_") and name.endswith("_complete.flag"):
            core = name[len("c_run_") : -len("_complete.flag")]
            if "_sa-" in core and "_evt-" in core:
                sa_part = core.split("_sa-", 1)[1]
                sa_id, event_id = sa_part.split("_evt-", 1)
                sims.append((sa_id, event_id))
            else:
                sims.append((core, None))
    return sorted(sims)


def _bin_dir(case_dir: str, sa_id: str, event_id: str | None) -> str:
    """Remote ``out_tritonswmm/bin`` dir for one sim — sensitivity layout when an event
    id is known (``subanalyses/sa_{sa_id}/sims/{event_id}/``), else legacy multisim."""
    if event_id is not None:
        return f"{case_dir}/subanalyses/sa_{sa_id}/sims/{event_id}/out_tritonswmm/bin"
    return f"{case_dir}/sims/{sa_id}/out_tritonswmm/bin"


def _fetch_mh_blobs(case_dir: str, sa_id: str, event_id: str | None = None) -> dict[int, bytes]:
    """Fetch ``{idx: bytes}`` for the MAX-index ``MH_<idx>_00.out`` (the final depth grid)
    of one sim via base64 over ssh. Only the FINAL dump is needed for the byte-identity
    comparison, so we list the dir, pick the max index, and fetch that single dump (1 ssh
    round-trip per sim rather than one per checkpoint). base64 keeps the binary intact across
    the text ssh channel; ``_ssh`` stays the sole seam.
    """
    bin_dir = _bin_dir(case_dir, sa_id, event_id)
    listing = _ssh(f"ls {bin_dir} 2>/dev/null || true")
    idxs: list[int] = []
    for name in listing.split():
        if name.startswith("MH_") and name.endswith("_00.out"):
            try:
                idxs.append(int(name[len("MH_") : -len("_00.out")]))
            except ValueError:
                continue
    if not idxs:
        return {}
    mx = max(idxs)
    b64 = _ssh(f"base64 {bin_dir}/MH_{mx}_00.out")
    return {mx: base64.b64decode(b64)}


def case_final_hashes(case_dir: str) -> dict[str, str]:
    """Map every completed sim's sa_id to the md5 of its max-index final-MH dump."""
    hashes: dict[str, str] = {}
    for sa_id, event_id in completed_sims(case_dir):
        blobs = _fetch_mh_blobs(case_dir, sa_id, event_id)
        if blobs:
            _, digest = final_mh_md5(blobs)
            hashes[sa_id] = digest
    return hashes


def resume_count_by_sa(case_dir: str) -> dict[str, int]:
    """Per-sa_id count of genuine hotstart resumes for a case.

    Counts ``"Resuming tritonswmm from hotstart"`` occurrences across each sim's rule log
    (the simulation_sa rule redirects run_simulation stdout there, run_simulation.py:382).
    0 = the sim ran in a single allocation (never killed); >=1 = it was walltime-killed
    and auto-resumed that many times — the DoD-#3 evidence. Uses ``find`` (no globstar
    dependency) so it is robust to where the analysis log dir sits under the case dir.
    """
    out: dict[str, int] = {}
    for sa_id, _event_id in completed_sims(case_dir):
        raw = _ssh(
            f"find {case_dir} -type f -name 'simulation_sa_{sa_id}_evt*.log' "
            f"-exec grep -c 'Resuming tritonswmm from hotstart' {{}} + 2>/dev/null "
            f"| awk -F: '{{s+=$NF}} END {{print s+0}}'"
        ).strip()
        out[sa_id] = int(raw or 0)
    return out


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------
def render_report(
    clean_hashes: Mapping[str, str],
    resume_hashes: Mapping[str, str] | None,
    out_path: str,
    resume_case_dir: str | None = None,
) -> None:
    """Fill ``_report_template.md`` with the byte-group + clean-vs-resume findings.

    Phase 1 wires the byte-group summary; the per-hypothesis verdicts (H1-H5) and the
    resume divergence-onset rows are completed in Phase 3 against real completed sims.
    """
    from pathlib import Path

    template = (Path(__file__).parent / "_report_template.md").read_text(encoding="utf-8")
    # Re-group the clean finals so the report shows the byte-identical groups directly.
    blob_view = {sid: digest.encode() for sid, digest in clean_hashes.items()}
    clean_groups = group_by_hash(blob_view)
    lines = [f"- group `{h[:12]}`: {ids}" for h, ids in sorted(clean_groups.items())]
    summary = "\n".join(lines) if lines else "(no completed clean sims)"
    body = template + f"\n\n## Computed byte-groups (clean)\n\n{summary}\n"
    if resume_hashes is not None:
        cmp = compare_clean_vs_resume(clean_hashes, resume_hashes)
        body += f"\n## Clean-vs-resume\n\n- matched: {cmp['matched']}\n- diverged: {cmp['diverged']}\n"
    if resume_case_dir is not None:
        rc = resume_count_by_sa(resume_case_dir)
        body += "\n## Resume counts (genuine hotstart resumes per sim)\n\n"
        body += "".join(f"- {sid}: {rc.get(sid, 0)}\n" for sid in sorted(rc))
    Path(out_path).write_text(body, encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", action="append", choices=["clean", "resume"], required=True)
    p.add_argument("--case-root", default=_DEFAULT_CASE_ROOT, help="root holding synth_cc_{clean,resume} case dirs")
    p.add_argument("--report-out", default=None)
    args = p.parse_args()

    case_dirs = _case_dirs(args.case_root)
    clean_hashes = case_final_hashes(case_dirs["clean"]) if "clean" in args.case else {}
    resume_hashes = case_final_hashes(case_dirs["resume"]) if "resume" in args.case else None

    for digest, ids in sorted(group_by_hash({s: h.encode() for s, h in clean_hashes.items()}).items()):
        print(f"clean group {digest[:12]}: {ids}")
    if resume_hashes is not None:
        cmp = compare_clean_vs_resume(clean_hashes, resume_hashes)
        print(f"clean-vs-resume diverged: {cmp['diverged']}")

    if args.report_out:
        render_report(
            clean_hashes,
            resume_hashes,
            args.report_out,
            resume_case_dir=case_dirs["resume"] if "resume" in args.case else None,
        )
        print(f"report written: {args.report_out}")


if __name__ == "__main__":
    main()
