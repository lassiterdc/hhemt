"""Synth-tier DU-sentinel integrity regression test (Phase 2, Deliverable A).

Asserts, on a completed SENSITIVITY tree: (1) every `_du.json` reports
`walk_errors == 0` (no silent partial DU totals); (2) the master root's own
sentinel is present and analysis-scoped, and every sentinel on the LIVE surface
(master root + `members/`) carries a scope inside the declared `Scope` vocabulary;
and (3) the
`delete` CLI dry-run emits zero "DU sentinel absent — walking tree" stderr lines
(the fallback-walk-fired regression).

Scope note — what this test may and may NOT assert. It does NOT assert that a
member root carries scope="member", and that omission is deliberate: the member
label is written by the D6 fold in `consolidate_workflow.py` as a side-effect of
the per-member consolidate RULE, while this fixture's readiness payload is
`("experiment_datatree.zarr", "_status/f_consolidate_experiment_complete.flag")`.
The gate is `_marker_state` (conftest.py:517-533), which requires a PROVENANCE
match on the builder digest + toolkit sha BEFORE it ever consults `_unmet_payload`
-- so the payload is not the gate, and every commit makes the marker stale. What
makes a mislabelled member survive is what happens NEXT: a stale marker only
re-drives `submit_workflow(mode="local")`, whose per-member rules are satisfied by
their own `e_consolidate_member-*` flags and no-op. So a tree the fixture re-drives
can still carry member roots labelled "analysis" by
`processing_analysis.consolidate_to_datatree`, and asserting `{member, analysis}`
here asserted something the fixture never promised. The durable note on that
residual lives at `du_sentinels._infer_scope`. What follows is the original
rationale for the sensitivity tree: no single
workflow tree carries all three DU scopes. The sensitivity tree carries
`member` (per-sub `--member-id` fold, D6) and `analysis` (the master-consolidate
fold in `consolidate_sensitivity_datatree`, this Phase). It carries NO
scenario-scope sentinels — the sensitivity master Snakefile emits no per-event
`consolidate_scenario` rule, so scenario-scope on sensitivity sub-events is a
deferred follow-on, not asserted here. Scenario-scope + analysis-scope on the
MULTISIM path are produced by already-shipped code (the `consolidate_scenario`
rule + `consolidate_to_datatree`), outside Phase 2's change set, so they are not
re-asserted here.

Per FINDING 2, the delete fallback warning is a bare `print(..., file=sys.stderr)`
in the delete CLI closure, so capture is via `capsys`, not `caplog`.

Warm-cache caveat: `synthetic_sensitivity_completed` is session-scoped and cached
(`start_from_scratch=False`). A tree cached before the Phase 2 master-write fold
landed will lack the master analysis-scope sentinel until its
`f_consolidate_experiment_complete.flag` is deleted to force re-materialization (the
same one-time transition cost the D6 sub-fold documented; the master write is
idempotent on the early-return path, so a flag-only delete suffices). See the
captured follow-up to add a flag-deleting fixture variant for durable robustness.
"""

from typing import get_args

from hhemt.du_sentinels import Scope, read_du_sentinel


def test_all_du_sentinels_walk_errors_zero(synthetic_sensitivity_completed):
    # Assert what the FIXTURE guarantees, not what a fully-re-materialized tree
    # happens to carry. Note the limit on that guarantee, because this file states it
    # twenty lines up: the warm-cache caveat records that a tree cached before the
    # master-write fold carries the flag with NO master sentinel, so "flag implies
    # sentinel" is FALSE IN GENERAL and holds only on trees materialized after that
    # fold -- which is every tree in the current cache, measured. The assertion below
    # is therefore a real check on today's population, not a theorem. A member root's
    # scope="member" label is NOT implied by anything in the payload (see the Scope
    # note above), so it is not asserted here.
    sensitivity = synthetic_sensitivity_completed
    analysis_dir = sensitivity.experiment.analysis_paths.analysis_dir
    sentinels = list(analysis_dir.rglob("_du.json"))
    assert sentinels, "no _du.json sentinels found on a completed sensitivity run"
    scopes_seen = set()
    for s in sentinels:
        payload = read_du_sentinel(s)
        assert payload is not None, f"corrupt/absent sentinel: {s}"
        assert payload["walk_errors"] == 0, (
            f"{s} reports walk_errors={payload['walk_errors']} -> disk_utilization_bytes is a PARTIAL total"
        )
        scopes_seen.add(payload["scope"])
    master = read_du_sentinel(analysis_dir / "_status" / "_du.json")
    assert master is not None, (
        f"the master root has no _du.json at {analysis_dir}/_status/_du.json, but the "
        "fixture's readiness payload includes f_consolidate_experiment_complete.flag, "
        "which implies consolidate_sensitivity_datatree ran and wrote it"
    )
    assert master["scope"] == "analysis", (
        f"the master root's own sentinel must be analysis-scoped; got {master['scope']!r}"
    )
    # Vocabulary conformance over the LIVE surface only: the master root and members/.
    # `subanalyses/sa_*/` is NOT part of the current layout -- it is orphan directory
    # residue from an earlier materialization in the same cache path, co-resident with
    # the renamed members/ tree, and it carries the pre-rename `sub_analysis` token.
    # Measured on the payload-satisfying tree: rglob sees {'analysis': 5,
    # 'sub_analysis': 4}, so an unscoped sweep fails here on residue this test does not
    # own. It is NOT a failed migration -- that tree was created at layout_version 22
    # with an empty migration_history, so V0019 correctly never ran -- and it has no
    # owner today, because find_orphan_member_dirs iterates members_dir only.
    live_scopes = {
        payload["scope"]
        for s2 in sentinels
        if (payload := read_du_sentinel(s2)) is not None
        and s2.parent.parent in (analysis_dir, *(analysis_dir / "members").glob("*"))
    }
    unknown = live_scopes - set(get_args(Scope))
    assert not unknown, (
        f"sentinel(s) on the LIVE surface (master root + members/) carry a scope outside "
        f"the declared vocabulary {get_args(Scope)}: {sorted(unknown)}. Orphan "
        f"subanalyses/ residue is deliberately excluded; a bad token HERE is a writer defect."
    )


def test_delete_dry_run_no_fallback_walk_warning(synthetic_sensitivity_completed, capsys):
    # analysis.delete() has NO dry_run parameter; the dry-run is a CLI-level
    # concept reached through cli._print_delete_dry_run_summary (module-importable).
    # On a sensitivity master it reads each per-sub sentinel AND the master
    # analysis-scope sentinel (written by the Phase 2 master-consolidate fold).
    # Without that fold the master read falls back to a full rglob walk and prints
    # the warning (the regression this guards). capsys (not caplog) — it is a
    # print, not a logger.
    from hhemt.cli import _print_delete_dry_run_summary

    sensitivity = synthetic_sensitivity_completed
    _print_delete_dry_run_summary(sensitivity.experiment)
    captured = capsys.readouterr()
    assert "DU sentinel absent" not in captured.err, (
        "delete dry-run fell back to a tree walk -> a parent _du.json is missing/stale"
    )
