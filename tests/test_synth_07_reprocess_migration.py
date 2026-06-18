"""Phase 2: layout-version stamping behavior on reprocess.

DoD requirement: reprocess()'s entry-time ``stamp_new_target`` call must
bring ``_version.json`` to the current ``LAYOUT_VERSION``, mirroring
``run()``'s PI-1 lazy-stamp pattern. These tests verify the stamp side of
the contract — they do not exercise the full migration runner.

* ``test_reprocess_runs_migration_when_layout_outdated``: stamp the analysis
  dir at ``LAYOUT_VERSION - 1``, run reprocess, assert ``_version.json`` is
  at current ``LAYOUT_VERSION`` afterward.
* ``test_reprocess_idempotent_when_current``: stamp at current
  ``LAYOUT_VERSION``, run reprocess, assert ``_version.json`` byte content
  is byte-identical (idempotent — no re-write).
"""
from __future__ import annotations

import pytest


@pytest.mark.usefixtures("tritonswmm_cpu_compiled")
def test_reprocess_runs_migration_when_layout_outdated(synthetic_multisim_completed):
    """When ``_version.json`` is stamped older than the current LAYOUT_VERSION,
    reprocess() must re-stamp it to the current version (the lazy-stamp
    PI-1 contract carried over from run() / submit_workflow)."""
    from hhemt.version_migration import LAYOUT_VERSION
    from hhemt.version_migration.state import (
        read_version_file,
        stamp_new_target,
    )

    a = synthetic_multisim_completed
    analysis_dir = a.analysis_paths.analysis_dir

    # Force-stamp at LAYOUT_VERSION - 1 to simulate a pre-migration target.
    # stamp_new_target overwrites when the existing version differs, so this
    # call brings _version.json to LAYOUT_VERSION - 1 regardless of prior state.
    older = LAYOUT_VERSION - 1
    if older < 0:
        pytest.skip("LAYOUT_VERSION must be >= 1 for this test")
    stamp_new_target(analysis_dir, older)

    pre = read_version_file(analysis_dir)
    assert pre is not None and pre.layout_version == older, (
        f"Failed to set up: expected _version.json at v{older}, got "
        f"{pre.layout_version if pre else None!r}"
    )

    result = a.reprocess(start_with="render", execution_mode="local", verbose=False)
    assert result.get("success"), (
        f"reprocess(render) failed: {result.get('message','(no message)')}"
    )

    post = read_version_file(analysis_dir)
    assert post is not None and post.layout_version == LAYOUT_VERSION, (
        f"Expected post-reprocess _version.json at v{LAYOUT_VERSION}, got "
        f"{post.layout_version if post else None!r}"
    )


@pytest.mark.usefixtures("tritonswmm_cpu_compiled")
def test_reprocess_idempotent_when_current(synthetic_multisim_completed):
    """When ``_version.json`` is already at current ``LAYOUT_VERSION``,
    reprocess() must not re-write the file (stamp_new_target is idempotent
    at the same version)."""
    from hhemt.version_migration import LAYOUT_VERSION
    from hhemt.version_migration.state import (
        VERSION_FILE_NAME,
        stamp_new_target,
    )

    a = synthetic_multisim_completed
    analysis_dir = a.analysis_paths.analysis_dir

    # Ensure the analysis is at current LAYOUT_VERSION before the test body.
    stamp_new_target(analysis_dir, LAYOUT_VERSION)

    vf = analysis_dir / VERSION_FILE_NAME
    assert vf.exists(), f"Expected _version.json at {vf}"
    pre_bytes = vf.read_bytes()
    pre_mtime = vf.stat().st_mtime

    result = a.reprocess(start_with="render", execution_mode="local", verbose=False)
    assert result.get("success"), (
        f"reprocess(render) failed: {result.get('message','(no message)')}"
    )

    # Idempotent: stamp_new_target returns early when version matches, so
    # _version.json should be untouched (same bytes, same mtime).
    post_bytes = vf.read_bytes()
    post_mtime = vf.stat().st_mtime
    assert post_bytes == pre_bytes, (
        "Expected _version.json to be byte-identical when already at "
        f"LAYOUT_VERSION; diff detected."
    )
    assert post_mtime == pre_mtime, (
        "Expected _version.json mtime to be unchanged when already at "
        f"LAYOUT_VERSION; pre={pre_mtime!r}, post={post_mtime!r}."
    )
