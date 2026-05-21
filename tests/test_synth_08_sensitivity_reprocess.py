"""Phase 3: sensitivity master reprocess refreshes per-sa + master datatree.

The Phase 3 reprocess contract (R12 in the master plan): a sensitivity
master analysis at the post-master-consolidate state can be reprocessed
with ``sensitivity.reprocess(start_with="consolidate", execution_mode="local")``
to regenerate ``sensitivity_datatree.zarr`` (and the master consolidate
flag) without re-running any simulation rule.

The session-scoped ``synthetic_sensitivity_completed`` fixture
(``tests/conftest.py``) runs the synth sensitivity master once per pytest
session to the ``f_consolidate_master_complete.flag`` state; this test
body then re-invokes reprocess and asserts the master datatree zarr's
mtime advances.
"""


def test_sensitivity_reprocess_consolidate(synthetic_sensitivity_completed):
    """sensitivity.reprocess(start_with='consolidate') regenerates the master datatree."""
    sa = synthetic_sensitivity_completed
    mdt = sa.master_analysis.analysis_paths.sensitivity_datatree_zarr
    assert mdt.exists(), "fixture should have materialized sensitivity_datatree.zarr"
    mtime0 = mdt.stat().st_mtime
    result = sa.reprocess(start_with="consolidate", execution_mode="local")
    assert result["success"], f"reprocess failed: {result.get('message')!r}"
    assert mdt.stat().st_mtime > mtime0, "master datatree mtime should advance after reprocess"


def test_sensitivity_reprocess_consolidate_subset_sa_ids(synthetic_sensitivity_completed):
    """sensitivity.reprocess(sa_ids=[...]) restricts per-sa invalidation to the subset.

    The invalidation step only deletes the named per-sa consolidate flags; the
    master consolidate flag is invalidated regardless. After reprocess, both the
    subset's per-sa flags and the master flag must be re-created (success exit
    proves Snakemake completed the consolidate + master_consolidation chain).
    """
    sa = synthetic_sensitivity_completed
    status_dir = sa.master_analysis.analysis_paths.analysis_dir / "_status"
    all_sa_ids = [str(sid) for sid in sa.sub_analyses.keys()]
    # Subset: first sub-analysis only.
    subset = all_sa_ids[:1]
    result = sa.reprocess(
        start_with="consolidate",
        sa_ids=subset,
        execution_mode="local",
    )
    assert result["success"], f"reprocess(subset) failed: {result.get('message')!r}"
    # Master flag re-created after reprocess.
    assert (status_dir / "f_consolidate_master_complete.flag").exists()
    # Every per-sa flag exists post-reprocess (the subset ones re-created, the
    # others were never invalidated).
    for sid in all_sa_ids:
        assert (status_dir / f"e_consolidate_sa-{sid}_complete.flag").exists()
