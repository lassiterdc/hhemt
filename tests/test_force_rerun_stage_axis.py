"""The force-rerun `stage` axis must survive the config-to-actuator boundary.

Entry point is chosen per boundary, and each choice is load-bearing.

The STAGE-AXIS tests enter at ``_apply_force_rerun``, NOT at
``_delete_flags_for_force_rerun``. That choice is load-bearing: the actuator was
always correct, so a test that hand-builds a ``ResolvedForceRerunSpec`` with
``stage="render"`` and calls the actuator directly PASSES on pre-fix code and
proves nothing. The axis was dropped between the two, and only entry at
``_apply_force_rerun`` crosses that boundary.

Invariant under test: the ``stage`` a caller supplies is the ``stage`` the
actuator floors on.
"""

from __future__ import annotations

import pytest

from hhemt.config.analysis import ForceRerunSpec
from hhemt.exceptions import ConfigurationError


def _seed_status_and_plots(analysis):
    """Seed three completion flags and two figures — one Snakemake, one EDA."""
    root = analysis.analysis_paths.analysis_dir
    status_dir = root / "_status"
    status_dir.mkdir(parents=True, exist_ok=True)
    flags = [
        "c_run_tritonswmm_member-0_evt-x_complete.flag",
        "d_process_tritonswmm_member-0_evt-x_complete.flag",
        "e_consolidate_member-0_complete.flag",
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
        # `**_k` is deliberate and is not defensive clutter. This lambda REPLACES the
        # callee, so the arity that matters is the STUB's, not the call's -- a
        # keyword-only parameter with a default is transparent to every caller and
        # opaque to every fixed-arity replacement. Absorbing unknown keywords keeps this
        # arm testing what it says it tests (stage-gating) across future signature
        # widening instead of going red on a signature it does not care about.
        lambda spec, **_k: calls.append(spec.stage),
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


def test_raw_two_axis_dict_from_the_cli_is_coerced(synth_sensitivity_analysis):
    """THE PRODUCTION INPUT CLASS: a raw dict, exactly what `json.loads` hands the CLI.

    `--override-force-rerun '{"subject":"all","stage":"render"}'` reaches
    `_apply_force_rerun` as a plain dict, NOT as a `ForceRerunSpec`. Every other
    test in this module enters with an already-coerced spec or the legacy string,
    so none of them crosses the raw-dict path — which is the path the defect lives
    on and the only one the CLI ever takes.
    """
    analysis = synth_sensitivity_analysis
    status_dir, flags, snakemake_fig, _eda_fig = _seed_status_and_plots(analysis)

    analysis._apply_force_rerun({"subject": "all", "stage": "render"})

    for name in flags:
        assert (status_dir / name).exists(), (
            f"{name} was deleted — the raw dict was not coerced before the floor resolved"
        )
    assert not snakemake_fig.exists()


def test_malformed_subject_raises_configuration_error_not_validation_error(synth_sensitivity_analysis):
    """The coercion wrap pins the CLI exit-code contract.

    With coercion now running FIRST, `ForceRerunSpec`'s own `_validate_subject_shape`
    fires before the analysis-side validator — so a malformed subject would raise a
    pydantic `ValidationError`. `cli.py` maps `ConfigurationError` to `typer.Exit(2)`
    at five sites and has no pydantic-`ValidationError` handler (its 15 `ValidationError`
    matches are all `CLIValidationError`, a different class), so an unwrapped error
    would escape every `except` clause and crash at exit 1 — silently converting a
    documented exit-2 config error into an unhandled traceback.
    """
    analysis = synth_sensitivity_analysis
    with pytest.raises(ConfigurationError):
        analysis._apply_force_rerun({"subject": {"sa_id": []}, "stage": "simulate"})


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


class _StopBeforeSnakemake(Exception):
    """Sentinel: the sensitivity BUILDER was reached, so every pre-delete already ran."""


def _seed_and_stop_at_builder(analysis, monkeypatch):
    """Seed the tree, then make the sensitivity BUILDER raise.

    The seam is ``sensitivity._workflow_builder.submit_workflow``
    (sensitivity_analysis.py:491) and deliberately NOT
    ``analysis.sensitivity.submit_workflow``. The dispatch chain applies the
    force-rerun pre-delete TWICE -- once at analysis.py:3578 and again at
    sensitivity_analysis.py:464 -- and patching the outer of the two would halt
    between them, leaving the second invocation uncovered. A test written that way
    passes on code that still deletes every figure on a dry run.
    """
    seeded = _seed_status_and_plots(analysis)

    def _raise(*_a, **_k):
        raise _StopBeforeSnakemake

    monkeypatch.setattr(
        analysis.sensitivity._workflow_builder, "submit_workflow", _raise
    )
    return seeded


def test_render_floor_dry_run_preserves_figures(synth_sensitivity_analysis, monkeypatch):
    """THE DISCRIMINATING ARM -- RED pre-fix, green post-fix.

    Entry is ``Analysis.submit_workflow``, one layer outside this module's stage-axis
    entry point. Both ``dry_run`` and ``override_force_rerun`` already exist in that
    signature pre-fix and post-fix, so this assertion discriminates on BEHAVIOUR.
    Entering at ``_apply_force_rerun(..., dry_run=True)`` would raise TypeError pre-fix
    -- red on a missing signature rather than on the deletion, which proves nothing.

    PRE-FIX RUN at 5acbe136: analysis.py:3578 calls the actuator with no dry_run term;
    the render floor matches no flag and its figure branch unlinks every non-EDA file
    under plots/; the sentinel then raises and the first assertion below FAILS on the
    missing Snakemake figure. The EDA figure survives pre-fix too, so it is a companion
    assertion and never the discriminator.
    """
    analysis = synth_sensitivity_analysis
    _status_dir, _flags, snakemake_fig, eda_fig = _seed_and_stop_at_builder(
        analysis, monkeypatch
    )

    with pytest.raises(_StopBeforeSnakemake):
        analysis.submit_workflow(
            override_force_rerun={"subject": "all", "stage": "render"},
            dry_run=True,
        )

    assert snakemake_fig.exists(), (
        "a dry run deleted a Snakemake-rendered figure -- the force-rerun pre-delete "
        "ran unguarded on the submit path"
    )
    assert snakemake_fig.with_suffix(snakemake_fig.suffix + ".manifest.json").exists()
    assert eda_fig.exists()


def test_simulate_floor_dry_run_still_deletes_flags(synth_sensitivity_analysis, monkeypatch):
    """GREEN BOTH SIDES -- the flag-carve-out arm, not a discriminator.

    This does not fail pre-fix and is not claimed to. It exists so that a later
    "simplification" of the dry-run gate into a bare ``if not dry_run:`` around the
    whole pre-delete goes RED. The governing stipulation keeps completion-flag deletion
    OUTSIDE the guard because it is the sole signal that makes the DAG preview
    non-empty; suppressing it yields the "nothing to do" preview that stipulation
    rejects by name.

    Scope, stated because the arm deliberately stops short: a simulate floor ALSO
    clears per-scenario processing-log records. That clear IS now gated -- the guard
    lives at the destructive site in `_invalidate_processing_log_for_force_rerun` --
    and its own discriminating arm is `test_simulate_floor_dry_run_preserves_processing_log`
    below. This arm asserts the FLAG property only; the two are deliberately separate
    because a dry run must delete the flags AND preserve the log, and one assertion
    covering both would not say which half regressed.
    """
    analysis = synth_sensitivity_analysis
    status_dir, flags, _snakemake_fig, _eda_fig = _seed_and_stop_at_builder(
        analysis, monkeypatch
    )

    with pytest.raises(_StopBeforeSnakemake):
        analysis.submit_workflow(
            override_force_rerun={"subject": "all", "stage": "simulate"},
            dry_run=True,
        )

    for name in flags:
        assert not (status_dir / name).exists(), (
            f"{name} survived a simulate-floor dry run -- the dry-run gate was widened "
            f"past the figure branch and the DAG preview is now empty"
        )


def _seed_processing_log_record(analysis):
    """Write one ProcessingEntry into every enabled per-model log, and return the
    (model_log, key) pairs so a caller can assert on survival or clearance."""
    from hhemt.log import ProcessingEntry
    from hhemt.scenario import TRITONSWMM_scenario

    seeded = []
    for event_iloc in range(len(analysis.df_sims)):
        scen = TRITONSWMM_scenario(event_iloc, analysis)
        for model_type in scen.run.model_types_enabled:
            scen.get_log(model_type).processing_log.update(
                ProcessingEntry(
                    filepath=scen.scen_paths.sim_folder / "processed" / "seeded.zarr",
                    size_MiB=1.0,
                    time_elapsed_s=1.0,
                    success=True,
                )
            )
            seeded.append((event_iloc, model_type))
    assert seeded, "fixture produced no enabled model logs -- the arms below would be vacuous"
    return seeded


def _reread_has_key(analysis, event_iloc, model_type, key="seeded.zarr"):
    """Re-READ the per-model log from disk. Asserting on the handle returned by the
    seeder would be a FALSE GREEN: the actuator constructs its OWN scenario objects and
    mutates THOSE log instances before writing, so a stale in-memory handle never
    observes the clear no matter what the code under test does."""
    from hhemt.scenario import TRITONSWMM_scenario

    return key in TRITONSWMM_scenario(event_iloc, analysis).get_log(model_type).processing_log.outputs


def test_simulate_floor_dry_run_preserves_processing_log(synth_sensitivity_analysis, monkeypatch):
    """THE DISCRIMINATING ARM for the log clear -- RED pre-fix, green post-fix.

    PRE-FIX RUN at f8c47ec3: `_apply_force_rerun` reaches its stage gate, calls
    `_invalidate_processing_log_for_force_rerun(spec)` with no dry_run term, and the
    helper clears `processing_log.outputs` on every enabled per-model log -- so the
    assertion below fails on the first seeded record being gone. It discriminates on
    BEHAVIOUR: both parameters used at the entry point exist pre-fix and post-fix.
    """
    analysis = synth_sensitivity_analysis
    _seed_and_stop_at_builder(analysis, monkeypatch)
    seeded = _seed_processing_log_record(analysis)

    with pytest.raises(_StopBeforeSnakemake):
        analysis.submit_workflow(
            override_force_rerun={"subject": "all", "stage": "simulate"},
            dry_run=True,
        )

    for event_iloc, model_type in seeded:
        assert _reread_has_key(analysis, event_iloc, model_type), (
            "a dry run cleared a per-scenario processing-log record -- the force-rerun "
            "pre-delete ran its log-invalidation half unguarded"
        )


def test_simulate_floor_real_run_still_clears_processing_log(synth_sensitivity_analysis):
    """GREEN BOTH SIDES -- the anti-over-fire arm, not a discriminator.

    A REAL simulate-floor force-rerun MUST still clear the log, or `_already_written`
    keeps returning True and the re-fired rule writes nothing (Gotcha 28). This arm
    exists so the dry-run guard cannot be widened into "never clear".
    """
    analysis = synth_sensitivity_analysis
    seeded = _seed_processing_log_record(analysis)

    analysis._apply_force_rerun(ForceRerunSpec(subject="all", stage="simulate"))

    for event_iloc, model_type in seeded:
        assert not _reread_has_key(analysis, event_iloc, model_type), (
            f"seeded.zarr survived a REAL simulate-floor force-rerun ({model_type}) -- the "
            f"dry-run guard was widened past its input class and _already_written will skip"
        )
