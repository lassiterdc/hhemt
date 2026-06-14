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
_DEFAULT_CASE_ROOT = "/project/***REMOVED***/***REMOVED***/norfolk/synth_compute_config"


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
def completed_sa_ids(case_dir: str) -> list[str]:
    """sa_ids with a ``c_run_<sa_id>_complete.flag`` in ``{case_dir}/_status`` (completed sims only)."""
    out = _ssh(f"ls {case_dir}/_status 2>/dev/null || true")
    ids = []
    for line in out.splitlines():
        name = line.strip()
        if name.startswith("c_run_") and name.endswith("_complete.flag"):
            ids.append(name[len("c_run_") : -len("_complete.flag")])
    return sorted(ids)


def _fetch_mh_blobs(case_dir: str, sa_id: str) -> dict[int, bytes]:
    """Fetch ``{idx: bytes}`` for every ``MH_<idx>_00.out`` of one sim via base64 over ssh.

    base64 keeps the binary intact across the text ssh channel; ``_ssh`` stays the sole seam.
    """
    bin_dir = f"{case_dir}/sims/{sa_id}/out_tritonswmm/bin"
    listing = _ssh(f"ls {bin_dir} 2>/dev/null || true")
    blobs: dict[int, bytes] = {}
    for name in listing.split():
        if name.startswith("MH_") and name.endswith("_00.out"):
            try:
                idx = int(name[len("MH_") : -len("_00.out")])
            except ValueError:
                continue
            b64 = _ssh(f"base64 {bin_dir}/{name}")
            blobs[idx] = base64.b64decode(b64)
    return blobs


def case_final_hashes(case_dir: str) -> dict[str, str]:
    """Map every completed sim's sa_id to the md5 of its max-index final-MH dump."""
    hashes: dict[str, str] = {}
    for sa_id in completed_sa_ids(case_dir):
        blobs = _fetch_mh_blobs(case_dir, sa_id)
        if blobs:
            _, digest = final_mh_md5(blobs)
            hashes[sa_id] = digest
    return hashes


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------
def render_report(
    clean_hashes: Mapping[str, str],
    resume_hashes: Mapping[str, str] | None,
    out_path: str,
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
        render_report(clean_hashes, resume_hashes, args.report_out)
        print(f"report written: {args.report_out}")


if __name__ == "__main__":
    main()
