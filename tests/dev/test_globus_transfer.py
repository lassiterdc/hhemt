"""E2E Globus transfer tests.

These tests require live, user-activated Globus endpoints and a valid OAuth
session. They are SKIPPED by default under pytest. To opt in:

    HHEMT_GLOBUS_E2E=1 pytest tests/dev/test_globus_transfer.py -v

They are also marked ``slow`` and ``globus_e2e`` so they are deselected by
``pytest -m "not slow"``.

Run interactively (outside pytest):
    conda run -n hhemt ipython -i tests/dev/test_globus_transfer.py

Or run a specific test via the CLI dispatch at the bottom:
    conda run -n hhemt python tests/dev/test_globus_transfer.py frontier
    conda run -n hhemt python tests/dev/test_globus_transfer.py uva
    conda run -n hhemt python tests/dev/test_globus_transfer.py verify
"""

import os
import sys

import pytest

_GLOBUS_E2E_ENABLED = os.environ.get("HHEMT_GLOBUS_E2E") == "1"
_globus_e2e = pytest.mark.skipif(
    not _GLOBUS_E2E_ENABLED,
    reason="Globus E2E tests require activated endpoints; set HHEMT_GLOBUS_E2E=1 to opt in",
)

# Use worktree source if available, otherwise fall back to installed package
WORKTREE_SRC = "/home/***REMOVED***/dev/hhemt/.claude/worktrees/globus-auto-transfer-and-debug-restructuring/src"
if os.path.isdir(WORKTREE_SRC):
    sys.path.insert(0, WORKTREE_SRC)

from pathlib import Path  # noqa: E402

from hhemt.config.globus import PostRunTransferConfig, _get_endpoint_uuids  # noqa: E402
from hhemt.globus_transfer import GlobusTransferManager  # noqa: E402

# ── Test 1: Frontier → Local ──────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.globus_e2e
@_globus_e2e
def test_frontier():
    """Transfer Frontier sensitivity suite results to local machine."""
    # Clear stale tokens to force fresh auth
    token_file = Path.home() / ".globus_tokens.json"
    if token_file.exists():
        token_file.unlink()
        print("[Setup] Cleared stale tokens")

    config = PostRunTransferConfig(
        destination_root=r"D:\Dropbox\_GradSchool\repos\hhemt\frontier",
        system="frontier",
    )

    spec = config.to_transfer_spec(
        analysis_dir=Path(
            "/lustre/orion/***REMOVED***/proj-shared/***REMOVED***/hhemt/"
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


@pytest.mark.slow
@pytest.mark.globus_e2e
@_globus_e2e
def test_uva():
    """Transfer UVA results to local machine. Edit paths before running."""
    config = PostRunTransferConfig(
        destination_root="/D/Dropbox/_GradSchool/repos/hhemt/uva",
        system="uva",
    )

    spec = config.to_transfer_spec(
        analysis_dir=Path(
            "/dtn/landings/users/d/dc/***REMOVED***/project/***REMOVED***/***REMOVED***/norfolk/"
            "hhemt/test_data/norfolk_coastal_flooding/cases/uva_sensitivity_suite"
        ),
        analysis_id="uva_sensitivity_suite",
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
    dest = Path("/mnt/d/Dropbox/_GradSchool/repos/hhemt/" "frontier/frontier_sensitivity_suite")
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
