#!/usr/bin/env python
"""Re-derive the hhemt-specialist VMS roll-in status from DISK.

One marker grep per spec, one test-symbol grep per cluster. Run from the hhemt
worktree root; emits a markdown table. This exists because a hand-maintained
tracker decays silently across /handle-friction detours, and a rewind or a partial
apply makes the transcript a stale copy of the tree.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# (spec, cluster, target, marker) — marker is a fixed string unless it starts "re:"
SPECS = [
    ("S8a",  "S8", "src/hhemt/system.py",                                            "_PINNED_TRITON_SWMM_DEPTH_SCATTER_FIX_SHA"),
    ("S8b",  "S8", "src/hhemt/log.py",                                               "triton_has_swmm_depth_scatter_fix"),
    ("S8c",  "S8", "src/hhemt/processing_analysis.py",                               "triton_has_swmm_depth_scatter_fix"),
    ("S8d",  "S8", "src/hhemt/analysis_validation.py",                               "ARM C (S8)"),
    ("S8e",  "S8", "$AW/library/docs/stipulations/hhemt/coupled resume validity warning fires on pre-fix triton with a resumed coupled sim.md", "The three invalidity arms"),
    ("N1a",  "N1", "src/hhemt/eda/cross_sim_identity.py",                             "def _ref_rank"),
    ("N1b",  "N1", "src/hhemt/eda/_config_diff.py",                                   "groups = _group_by_identity(subs, root)"),
    ("N2",   "N2", "src/hhemt/report_plot_ids.py",                                    "Cross-hardware raw byte identity over time"),
    ("N3a",  "N3", "src/hhemt/eda/raw_resume_identity.py",                            "N3 (user ruling): ONE GPU family"),
    ("N3c",  "N3", "src/hhemt/eda/_plotting.py",                                      'if fam == "gpu":'),
    ("N3d",  "N3", "src/hhemt/report_templates/captions/b4b_clean_identity.rst",      "over time"),
    ("N4a",  "N4", "src/hhemt/eda/raw_resume_identity.py",                            "_collapse_replicates"),
    ("N4b",  "N4", "src/hhemt/eda/_plotting.py",                                      "Each row aggregates"),
    ("N5",   "N5", "src/hhemt/report_renderers/cross_experiment_intercomparison.py",  "_asymmetry_note"),
    ("N5b",  "N5", "src/hhemt/report_templates/captions/cross_experiment_intercomparison.rst", "Row counts differ per model by construction"),
    ("F4",   "F4", "src/hhemt/analysis_validation.py",                                "check_eda_calc_ran"),
    ("F4b",  "F4", "src/hhemt/bundle/_emit.py",                                       "declared_sources_absent"),
    ("F2",   "F2", "src/hhemt/_worktree_guard.py",                                    "__EXISTS__"),
    ("F2b",  "F2", "src/hhemt/cli.py",                                                "assert_worktree_source"),
    ("F2c",  "F2", "conftest.py",                                                     "worktree_mismatch_message"),
    ("F2d",  "F2", "$AW/library/prompts/instructions/protocols/worktree aware project testing.md", "RESOLVED. The objection was that a bare interpreter"),
    ("S4",   "S4", "src/hhemt/report_renderers/cross_experiment_errors_and_warnings.py", "_swmm_invisible_divergence_finding"),
    ("CF1",  "CF", "src/hhemt/system.py",                          "derived_from_resolvable_input"),
    ("CF2",  "CF", "src/hhemt/system.py",                          "TRITON_build_dir_gpu is not None"),
    ("CF3",  "CF", "src/hhemt/system.py",                          "compilation_logfile_gpu is not None"),
    ("CF4",  "CF", "src/hhemt/analysis_validation.py",             "GPU compile term NOT evaluated"),
    ("CF5",  "CF", "tests/test_system_compile_flag_abstention.py", "__EXISTS__"),
]

# cluster -> test symbols that must exist for the cluster to count as TESTED
# Symbols READ from tests/, never guessed: a grep-derived count is admissible only
# after a positive read that the token exists (an invented symbol returns a truthful
# zero and reports a tested cluster as untested).
CLUSTER_TESTS = {
    "S8": ["test_armc_fires_on_postfix_replayed_without_scatter", "test_armc_suppressed_when_arm_a_fires"],
    "N1": ["test_reference_rank_selects_serial", "test_grouping_rule_is_identical_across_model_arms"],
    "N2": ["b4b_clean_identity\") =="],  # N2's prescribed humanize_plot_id assertion
    "N3": ["test_b4b_family_key_collapses_every_gpu_hardware", "test_b4b_ref_key_is_deterministic"],
    "N4": ["test_collapse_replicates_folds_worst_case", "test_b4b_caption_discloses"],
    "N5": ["test_intercomparison_derives_the_per_model_row_denominator"],
    "F4": ["test_eda_calc_ran_fails_when_targets_are_enumerated_but_no_verdicts_exist", "test_harvest_records_the_absent_source"],
    # F2's controls PRE-EXIST in tests/test_worktree_guard.py; F2c must not regress them.
    "F2": ["HHEMT_FORCE_WRONG_SRC", "worktree_mismatch_message"],
    "S4": ["_swmm_invisible_divergence_finding"],
    # CF: the master-level compile-flag abstention cluster (8 specs, 2026-08-02).
    "CF": [
        "test_resolvable_gpu_path_with_missing_log_still_reports_failure",
        "test_bare_then_mutated_system_abstains_and_does_not_clobber",
        "test_summary_discloses_abstention_when_gpu_path_unresolvable",
        "test_summary_is_silent_when_gpu_path_resolves",
    ],
}

AW = "/home/dcl3nd/dev/agentic-workspace/.claude/worktrees/07-23_1017_pure-triton-arms-multi-resume-b4b-pwi"


def count(path: str, marker: str) -> int | None:
    p = Path(path.replace("$AW", AW))
    if marker == "__EXISTS__":
        return 1 if p.exists() else 0
    if not p.exists():
        return None
    try:
        return sum(1 for line in p.read_text(errors="replace").splitlines() if marker in line)
    except OSError:
        return None


def tests_present(symbols: list[str]) -> int:
    hits = 0
    for sym in symbols:
        r = subprocess.run(["grep", "-rl", sym, "tests/"], capture_output=True, text=True)
        if r.stdout.strip():
            hits += 1
    return hits


def main() -> int:
    print("| Spec | Cluster | Applied | Marker count | Target |")
    print("|---|---|---|---|---|")
    applied = 0
    for spec, cluster, target, marker in SPECS:
        n = count(target, marker)
        ok = n is not None and n > 0
        applied += ok
        state = "APPLIED" if ok else ("MISSING FILE" if n is None else "not applied")
        print(f"| {spec} | {cluster} | {'YES' if ok else 'no'} | {n if n is not None else '-'} | `{target}` |")
    print()
    print(f"**Applied: {applied} of {len(SPECS)}**")
    print()
    print("| Cluster | Tests present |")
    print("|---|---|")
    for cluster, syms in CLUSTER_TESTS.items():
        h = tests_present(syms)
        print(f"| {cluster} | {h}/{len(syms)} symbol(s) found in tests/ |")
    print()
    print("_Presence, not outcome: a symbol found here may sit in a FAILING assertion._")
    print("_Deciding command for outcome: `pytest tests/ -m \"not slow\"`._")
    return 0


if __name__ == "__main__":
    sys.exit(main())
