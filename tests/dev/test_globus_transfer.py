"""E2E Globus transfer tests.

Run interactively:
    conda run -n triton_swmm_toolkit ipython -i tests/dev/test_globus_transfer.py

Or run a specific test:
    conda run -n triton_swmm_toolkit python tests/dev/test_globus_transfer.py frontier
    conda run -n triton_swmm_toolkit python tests/dev/test_globus_transfer.py uva
    conda run -n triton_swmm_toolkit python tests/dev/test_globus_transfer.py verify
"""

import os
import sys

# Use worktree source if available, otherwise fall back to installed package
WORKTREE_SRC = "/home/***REMOVED***/dev/TRITON-SWMM_toolkit/.claude/worktrees/globus-auto-transfer-and-debug-restructuring/src"
if os.path.isdir(WORKTREE_SRC):
    sys.path.insert(0, WORKTREE_SRC)

from pathlib import Path  # noqa: E402

from TRITON_SWMM_toolkit.config.globus import PostRunTransferConfig, _get_endpoint_uuids  # noqa: E402
from TRITON_SWMM_toolkit.globus_transfer import GlobusTransferManager  # noqa: E402

# ── Test 1: Frontier → Local ──────────────────────────────────────────


def test_frontier():
    """Transfer Frontier sensitivity suite results to local machine."""
    # Clear stale tokens to force fresh auth
    token_file = Path.home() / ".globus_tokens.json"
    if token_file.exists():
        token_file.unlink()
        print("[Setup] Cleared stale tokens")

    config = PostRunTransferConfig(
        destination_root=r"D:\Dropbox\_GradSchool\repos\TRITON-SWMM_toolkit\frontier",
        system="frontier",
    )

    spec = config.to_transfer_spec(
        analysis_dir=Path(
            "/lustre/orion/***REMOVED***/proj-shared/***REMOVED***/TRITON-SWMM_toolkit/"
            "test_data/norfolk_coastal_flooding/cases/frontier_sensitivity_suite"
        ),
        analysis_id="frontier_sensitivity_suite",
    )

    print(f"Label: {spec.label}")
    print(f"Source: {spec.endpoints.source_uuid}")
    print(f"Dest:   {spec.endpoints.destination_uuid}")
    for item in spec.items:
        print(f"  {item.source_path} → {item.destination_path}")
    print(f"Exclude dirs: {config.exclude_patterns}")

    _uuid, _base, needs_data_access, session_domain = _get_endpoint_uuids(config.system)
    consent_uuids = [spec.endpoints.source_uuid] if needs_data_access else []
    session_domains = [session_domain] if session_domain else None

    manager = GlobusTransferManager(
        collection_uuids=consent_uuids,
        session_required_domains=session_domains,
    )
    task_id = manager.transfer(spec, exclude_dirs=config.exclude_patterns)
    print(f"Monitor: https://app.globus.org/activity/{task_id}")
    manager.wait(task_id, timeout_minutes=60)
    print("\n✓ Frontier transfer complete. Run 'verify' to check results.")


# ── Test 2: UVA → Local ──────────────────────────────────────────────


def test_uva():
    """Transfer UVA results to local machine. Edit paths before running."""
    config = PostRunTransferConfig(
        destination_root=r"D:\Dropbox\_GradSchool\repos\TRITON-SWMM_toolkit\uva",
        system="uva",
    )

    spec = config.to_transfer_spec(
        # TODO: replace with actual UVA analysis path
        analysis_dir=Path("/scratch/***REMOVED***/TRITON-SWMM_toolkit/EDIT_ME"),
        analysis_id="EDIT_ME",
    )

    print(f"Label: {spec.label}")
    for item in spec.items:
        print(f"  {item.source_path} → {item.destination_path}")

    _uuid, _base, needs_data_access, session_domain = _get_endpoint_uuids(config.system)
    consent_uuids = [spec.endpoints.source_uuid] if needs_data_access else []
    session_domains = [session_domain] if session_domain else None

    manager = GlobusTransferManager(
        collection_uuids=consent_uuids,
        session_required_domains=session_domains,
    )
    task_id = manager.transfer(spec, exclude_dirs=config.exclude_patterns)
    print(f"Monitor: https://app.globus.org/activity/{task_id}")
    manager.wait(task_id, timeout_minutes=60)
    print("\n✓ UVA transfer complete.")


# ── Verify ────────────────────────────────────────────────────────────


def verify():
    """Check Frontier transfer destination for correctness."""
    dest = Path("/mnt/d/Dropbox/_GradSchool/repos/TRITON-SWMM_toolkit/" "frontier/frontier_sensitivity_suite")
    print(f"Exists: {dest.exists()}")
    if not dest.exists():
        print("  Nothing to verify — destination does not exist.")
        return

    contents = [p.name for p in sorted(dest.iterdir())]
    print(f"Contents: {contents}")

    sims = dest / "sims"
    if sims.exists():
        for d in ["out_triton", "out_tritonswmm", "out_swmm"]:
            found = any((s / d).exists() for s in sims.iterdir() if s.is_dir())
            status = "✗ PRESENT (should be excluded)" if found else "✓ excluded"
            print(f"  sims/*/{d}/: {status}")

    subanalyses = dest / "subanalyses"
    status = "✗ PRESENT (should be excluded)" if subanalyses.exists() else "✓ excluded"
    print(f"  subanalyses/: {status}")


# ── CLI dispatch ──────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_globus_transfer.py [frontier|uva|verify]")
        print("  Or: ipython -i test_globus_transfer.py  (then call functions interactively)")
        sys.exit(0)

    cmd = sys.argv[1].lower()
    if cmd == "frontier":
        test_frontier()
    elif cmd == "uva":
        test_uva()
    elif cmd == "verify":
        verify()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
