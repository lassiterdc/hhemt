"""Synth-tier DU-sentinel integrity regression test (Phase 2, Deliverable A).

Asserts, on a completed SENSITIVITY tree: (1) every `_du.json` reports
`walk_errors == 0` (no silent partial DU totals); (2) the scopes Phase 2 makes
the sensitivity path produce are present — `{sub_analysis, analysis}`; and (3) the
`delete` CLI dry-run emits zero "DU sentinel absent — walking tree" stderr lines
(the fallback-walk-fired regression).

Scope note — why the sensitivity tree, and why two scopes (not three): no single
workflow tree carries all three DU scopes. The sensitivity tree carries
`sub_analysis` (per-sub `--sa-id` fold, D6) and `analysis` (the master-consolidate
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
`f_consolidate_master_complete.flag` is deleted to force re-materialization (the
same one-time transition cost the D6 sub-fold documented; the master write is
idempotent on the early-return path, so a flag-only delete suffices). See the
captured follow-up to add a flag-deleting fixture variant for durable robustness.
"""

from TRITON_SWMM_toolkit.du_sentinels import read_du_sentinel


def test_all_du_sentinels_walk_errors_zero(synthetic_sensitivity_completed):
    # The materialized sensitivity tree carries sub_analysis (per-sub --sa-id
    # fold, D6) and analysis (master-consolidate fold). It does NOT carry
    # scenario-scope sentinels (no per-event consolidate rule on the sensitivity
    # path), so the expected set is {sub_analysis, analysis}, not all three.
    sensitivity = synthetic_sensitivity_completed
    analysis_dir = sensitivity.master_analysis.analysis_paths.analysis_dir
    sentinels = list(analysis_dir.rglob("_du.json"))
    assert sentinels, "no _du.json sentinels found on a completed sensitivity run"
    scopes_seen = set()
    for s in sentinels:
        payload = read_du_sentinel(s)
        assert payload is not None, f"corrupt/absent sentinel: {s}"
        assert payload["walk_errors"] == 0, (
            f"{s} reports walk_errors={payload['walk_errors']} -> "
            f"disk_utilization_bytes is a PARTIAL total"
        )
        scopes_seen.add(payload["scope"])
    assert {"sub_analysis", "analysis"} <= scopes_seen, (
        f"expected sub_analysis + analysis scopes present, saw {scopes_seen}"
    )


def test_delete_dry_run_no_fallback_walk_warning(synthetic_sensitivity_completed, capsys):
    # analysis.delete() has NO dry_run parameter; the dry-run is a CLI-level
    # concept reached through cli._print_delete_dry_run_summary (module-importable).
    # On a sensitivity master it reads each per-sub sentinel AND the master
    # analysis-scope sentinel (written by the Phase 2 master-consolidate fold).
    # Without that fold the master read falls back to a full rglob walk and prints
    # the warning (the regression this guards). capsys (not caplog) — it is a
    # print, not a logger.
    from TRITON_SWMM_toolkit.cli import _print_delete_dry_run_summary

    sensitivity = synthetic_sensitivity_completed
    _print_delete_dry_run_summary(sensitivity.master_analysis)
    captured = capsys.readouterr()
    assert "DU sentinel absent" not in captured.err, (
        "delete dry-run fell back to a tree walk -> a parent _du.json is missing/stale"
    )
