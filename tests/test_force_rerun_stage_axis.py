"""The force-rerun `stage` axis must survive the config-to-actuator boundary.

Every test here enters at ``_apply_force_rerun``, NOT at
``_delete_flags_for_force_rerun``. That choice is load-bearing: the actuator was
always correct, so a test that hand-builds a ``ResolvedForceRerunSpec`` with
``stage="render"`` and calls the actuator directly PASSES on pre-fix code and
proves nothing. The axis was dropped between the two, and only entry at
``_apply_force_rerun`` crosses that boundary.

Invariant under test: the ``stage`` a caller supplies is the ``stage`` the
actuator floors on.
"""

from __future__ import annotations

from hhemt.config.analysis import ForceRerunSpec


def _seed_status_and_plots(analysis):
    """Seed three completion flags and two figures — one Snakemake, one EDA."""
    root = analysis.analysis_paths.analysis_dir
    status_dir = root / "_status"
    status_dir.mkdir(parents=True, exist_ok=True)
    flags = [
        "c_run_tritonswmm_sa-0_evt-x_complete.flag",
        "d_process_tritonswmm_sa-0_evt-x_complete.flag",
        "e_consolidate_sa-0_complete.flag",
    ]
    for name in flags:
        (status_dir / name).touch()

    snakemake_fig = root / "plots" / "sensitivity" / "benchmarking" / "fig.html"
    snakemake_fig.parent.mkdir(parents=True, exist_ok=True)
    snakemake_fig.touch()
    snakemake_fig.with_suffix(snakemake_fig.suffix + ".manifest.json").touch()

    eda_fig = root / "plots" / "eda" / "config_diff_maps.html"
    eda_fig.parent.mkdir(parents=True, exist_ok=True)
    eda_fig.touch()
    eda_fig.with_suffix(eda_fig.suffix + ".manifest.json").touch()

    return status_dir, flags, snakemake_fig, eda_fig


def test_render_floor_preserves_flags_and_deletes_snakemake_figures(synth_sensitivity_analysis):
    """RQ3 primary. A render floor deletes ZERO `_status/*.flag` and deletes figures.

    RED pre-fix: the stage is dropped at the resolution boundary, the floor
    resolves to `simulate`, and all three flags are deleted -- so the
    "flags survive" assertions fail. Both asserted properties exist in the
    pre-fix and post-fix worlds, so this discriminates on BEHAVIOUR rather than
    on any message wording (pre-fix the path raises nothing and prints nothing,
    which is exactly why a wording-anchored assertion would be permanently green).
    """
    analysis = synth_sensitivity_analysis
    status_dir, flags, snakemake_fig, _eda_fig = _seed_status_and_plots(analysis)

    analysis._apply_force_rerun(ForceRerunSpec(subject="all", stage="render"))

    for name in flags:
        assert (status_dir / name).exists(), (
            f"{name} was deleted under a render floor -- the stage was dropped and the "
            f"floor fell back to 'simulate'"
        )
    assert not snakemake_fig.exists()
    assert not snakemake_fig.with_suffix(snakemake_fig.suffix + ".manifest.json").exists()


def test_render_floor_exempts_plots_eda(synth_sensitivity_analysis):
    """RQ4. `plots/eda/` survives a render floor; no Snakemake rule regenerates it.

    Those figures come from `analysis.eda()`, a non-Snakemake in-process facade,
    so deleting them here removes the family permanently -- a re-render restores
    only the Snakemake-driven figures. `bundle/_emit.py::_prune_undeclared_figures`
    already carries this exemption; the render branch did not.
    """
    analysis = synth_sensitivity_analysis
    _status_dir, _flags, snakemake_fig, eda_fig = _seed_status_and_plots(analysis)

    analysis._apply_force_rerun(ForceRerunSpec(subject="all", stage="render"))

    assert eda_fig.exists(), "plots/eda/ figure deleted under a render floor"
    assert eda_fig.with_suffix(eda_fig.suffix + ".manifest.json").exists()
    # The exemption must be narrow: a non-EDA figure is still deleted. This is the
    # differently-positioned satisfying arm -- it catches an exemption written too broadly.
    assert not snakemake_fig.exists()


def test_render_floor_does_not_invalidate_processing_log(synth_sensitivity_analysis, monkeypatch):
    """G4. The log invalidator is stage-gated, not merely the flag deleter.

    `_invalidate_processing_log_for_force_rerun` clears `processing_log.outputs`
    AND resets both `raw_*_outputs_cleared` markers, so calling it under a render
    floor re-arms the clear-raw step and re-runs processing -- the exact harm
    `_FLOOR_FLAG_PREFIXES`' own comment says a render floor must never cause.
    RED both pre-fix AND after a G2-only fix, which is what makes this the test
    that distinguishes the partial fix from the complete one.
    """
    analysis = synth_sensitivity_analysis
    _seed_status_and_plots(analysis)
    calls: list[str] = []
    monkeypatch.setattr(
        analysis,
        "_invalidate_processing_log_for_force_rerun",
        lambda spec: calls.append(spec.stage),
    )

    analysis._apply_force_rerun(ForceRerunSpec(subject="all", stage="render"))
    assert calls == [], "log invalidator ran under a render floor -- clear-raw was re-armed"

    analysis._apply_force_rerun(ForceRerunSpec(subject="all", stage="simulate"))
    assert calls == ["simulate"], "log invalidator must still run under a simulate floor"


def test_simulate_floor_deletes_flags_as_before(synth_sensitivity_analysis):
    """The historical default path is unperturbed — the second differential arm.

    Every existing caller relies on `force_rerun` meaning a simulate floor, so
    this asserts the change did not move that behaviour. A satisfying input in a
    DIFFERENT correct state than the render case above.
    """
    analysis = synth_sensitivity_analysis
    status_dir, flags, _snakemake_fig, _eda_fig = _seed_status_and_plots(analysis)

    analysis._apply_force_rerun(ForceRerunSpec(subject="all", stage="simulate"))

    for name in flags:
        assert not (status_dir / name).exists(), (
            f"{name} survived a simulate floor -- the historical default path changed"
        )


def test_legacy_string_form_still_coerces(synth_sensitivity_analysis):
    """The bare legacy `"all"` must still reach the actuator as a simulate floor.

    `_coerce_legacy` wraps any non-spec value as `{"subject": value}`, so the
    coercion boundary added at `_apply_force_rerun` must not break the historical
    string form that every existing config and CLI invocation uses.
    """
    analysis = synth_sensitivity_analysis
    status_dir, flags, _snakemake_fig, _eda_fig = _seed_status_and_plots(analysis)

    analysis._apply_force_rerun("all")

    for name in flags:
        assert not (status_dir / name).exists(), f"{name} survived a legacy 'all' force"
