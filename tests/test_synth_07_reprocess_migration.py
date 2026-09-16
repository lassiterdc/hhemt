"""Phase 2: layout-version stamping behavior on reprocess.

Contract: reprocess()'s entry-time ``stamp_new_target`` call stamps a target
that has NO layout record, and REFUSES a target whose record states a
different version. Migrating a tree is an explicit operator act — a verb the
operator types — so no execution facade may perform one as a side effect.
These tests verify the stamp side of that contract; they do not exercise the
migration runner.

* ``test_reprocess_refuses_when_layout_outdated``: baseline the analysis dir
  down to ``LAYOUT_VERSION - 1``, run reprocess, assert it raises
  ``LayoutVersionError`` and leaves ``_version.json`` untouched.
* ``test_reprocess_idempotent_when_current``: stamp at current
  ``LAYOUT_VERSION``, run reprocess, assert ``_version.json`` byte content
  is byte-identical (idempotent — no re-write).
"""

from __future__ import annotations

import pytest


@pytest.mark.usefixtures("tritonswmm_cpu_compiled")
def test_reprocess_refuses_when_layout_outdated(synthetic_multisim_completed_isolated):
    """When ``_version.json`` states a version older than the current
    LAYOUT_VERSION, reprocess() must REFUSE and leave the record untouched.

    Relabelling the record would assert a layout the tree does not have, and
    migrating it silently would perform an operator's decision for them. The
    refusal fires at reprocess()'s entry stamp, above every destructive step,
    so nothing has been deleted when it raises."""
    from hhemt.version_migration import LAYOUT_VERSION, runner
    from hhemt.version_migration.exceptions import LayoutVersionError
    from hhemt.version_migration.state import VERSION_FILE_NAME, read_version_file

    a = synthetic_multisim_completed_isolated
    analysis_dir = a.analysis_paths.analysis_dir

    # Force-stamp at LAYOUT_VERSION - 1 to simulate a pre-migration target.
    # stamp_new_target no longer relabels an existing record, so the sanctioned
    # way to bring a stamped tree DOWN one version is baseline's --force path,
    # which resets migration_history deliberately (see runner.baseline).
    older = LAYOUT_VERSION - 1
    if older < 0:
        pytest.skip("LAYOUT_VERSION must be >= 1 for this test")
    runner.baseline(analysis_dir, older, force=True)

    pre = read_version_file(analysis_dir)
    assert pre is not None and pre.layout_version == older, (
        f"Failed to set up: expected _version.json at v{older}, got {pre.layout_version if pre else None!r}"
    )

    vf = analysis_dir / VERSION_FILE_NAME
    pre_bytes = vf.read_bytes()

    with pytest.raises(LayoutVersionError) as excinfo:
        a.reprocess(start_with="render", execution_mode="local", verbose=False)

    # Assert on the structured attributes, never on the message text: a
    # wording-keyed assertion would pass for the wrong reason the moment the
    # remedy prose is reworded, and could not have been written at all before
    # the refusal existed.
    assert excinfo.value.current == older, (
        f"Expected LayoutVersionError.current == {older}, got {excinfo.value.current!r}"
    )
    assert excinfo.value.target == LAYOUT_VERSION, (
        f"Expected LayoutVersionError.target == {LAYOUT_VERSION}, got {excinfo.value.target!r}"
    )

    # Non-destructive: a refusal that rewrote the record it declined to relabel
    # would destroy created_at / toolkit_version / migration_history.
    assert vf.read_bytes() == pre_bytes, (
        "reprocess() refused but _version.json changed; the refusal must leave "
        "the tree's own layout claim byte-identical."
    )
    post = read_version_file(analysis_dir)
    assert post is not None and post.layout_version == older, (
        f"Expected _version.json still at v{older} after the refusal, got {post.layout_version if post else None!r}"
    )


@pytest.mark.usefixtures("tritonswmm_cpu_compiled")
def test_reprocess_idempotent_when_current(synthetic_multisim_completed_isolated):
    """When ``_version.json`` is already at current ``LAYOUT_VERSION``,
    reprocess() must not re-write the file (stamp_new_target is idempotent
    at the same version)."""
    from hhemt.version_migration import LAYOUT_VERSION
    from hhemt.version_migration.state import (
        VERSION_FILE_NAME,
        stamp_new_target,
    )

    a = synthetic_multisim_completed_isolated
    analysis_dir = a.analysis_paths.analysis_dir

    # Ensure the analysis is at current LAYOUT_VERSION before the test body.
    stamp_new_target(analysis_dir, LAYOUT_VERSION)

    vf = analysis_dir / VERSION_FILE_NAME
    assert vf.exists(), f"Expected _version.json at {vf}"
    pre_bytes = vf.read_bytes()
    pre_mtime = vf.stat().st_mtime

    result = a.reprocess(start_with="render", execution_mode="local", verbose=False)
    assert result.get("success"), f"reprocess(render) failed: {result.get('message', '(no message)')}"

    # Idempotent: stamp_new_target returns early when version matches, so
    # _version.json should be untouched (same bytes, same mtime).
    post_bytes = vf.read_bytes()
    post_mtime = vf.stat().st_mtime
    assert post_bytes == pre_bytes, (
        "Expected _version.json to be byte-identical when already at LAYOUT_VERSION; diff detected."
    )
    assert post_mtime == pre_mtime, (
        "Expected _version.json mtime to be unchanged when already at "
        f"LAYOUT_VERSION; pre={pre_mtime!r}, post={post_mtime!r}."
    )
