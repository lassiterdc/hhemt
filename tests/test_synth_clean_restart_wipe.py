"""Targeted clean-restart wipe must preserve the setup-generated per-sub config.

Regression test for the `_clean_restart_wipe` over-wipe bug: a whole-dir
`fast_rmtree(sub_dir)` deleted `sa_{sa}.yaml` (written by
`_create_sub_analyses` at master `__init__`, read by the prepare runner via
`--analysis-config`), and because the surviving `a_setup_target_N_complete.flag`
makes snakemake skip setup, nothing regenerated it -> prepare aborted
"[ERROR] Analysis config not found". The fix wipes only the RUNTIME children and
preserves the config (mirrors the `.test()` config-outside-wipe precedent,
analysis.py). Construction-only: no compile, no snakemake subprocess.
"""


def test_clean_restart_wipe_preserves_sub_config(synth_sensitivity_analysis):
    analysis = synth_sensitivity_analysis
    sa = next(iter(analysis.sensitivity.analyses.keys()))  # bare sa_id, e.g. "1"
    sub_dir = analysis.analysis_paths.analysis_dir / "subanalyses" / f"sa_{sa}"
    cfg = sub_dir / f"sa_{sa}.yaml"
    assert cfg.exists()  # precondition: _create_sub_analyses wrote it at __init__

    # Seed RUNTIME state the wipe must remove.
    (sub_dir / "sims").mkdir(parents=True, exist_ok=True)
    (sub_dir / "sims" / "marker.txt").write_text("runtime")
    (sub_dir / "log.json").write_text("{}")

    analysis._clean_restart_wipe([sa])

    # The FIX: config survives (pre-fix: fast_rmtree(sub_dir) deleted it -> FAILS here).
    assert cfg.exists(), "clean-restart wipe must preserve the setup-generated sub config"
    # Invariant true in BOTH pre- and post-fix states (well-formedness anchor): runtime gone.
    assert not (sub_dir / "sims").exists()
    assert not (sub_dir / "log.json").exists()
