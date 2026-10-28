"""Tests for the EDA plotting family (eda/_plotting.py)."""

from __future__ import annotations

import json
from pathlib import Path

from hhemt.config.eda import eda_config
from hhemt.eda import check_cross_sim_identity
from hhemt.eda._plotting import render_eda_plots
from hhemt.utils import resolve_experiment_tree


def test_render_cross_sim_identity_emits_master_rooted_with_source(
    synthetic_sensitivity_completed,
):
    """The config-diff EDA plot emits under MASTER-ROOTED plots/eda/ and declares a
    BUNDLE-CARRIED data source (R3/D1 carriage guard). The redesigned plot reads the
    consolidated ROOT store directly -- whichever store resolve_experiment_tree selects,
    the unified name on a tree carrying more than one
    (per-cell max_wlevel_m + per-conduit
    max_flow_cms + per-sub compute-config attrs -- no smaller per-plot summary zarr
    suffices), so it declares that tree (the primary consolidated output carried into
    every render bundle) rather than an eda/<plot_id>.zarr artifact."""
    analysis = synthetic_sensitivity_completed.experiment  # NOT the wrapper (D4)
    # Run the calc first, mirroring analysis.eda()'s calc->plots order. The config-diff
    # plot itself reads the consolidated tree (not the calc's eda zarr), but the real
    # facade always runs calc before plots, so exercise that order here.
    check_cross_sim_identity(analysis)
    root = Path(analysis.analysis_paths.analysis_dir)

    paths = render_eda_plots(root, cfg_analysis=analysis.cfg_analysis, eda_cfg=eda_config())
    assert len(paths) == 1
    out = paths[0]
    # MASTER-ROOTED: under {root}/plots/eda/, NOT under plots/sensitivity/per_sim/.
    assert out.parent == root / "plots" / "eda"
    assert out.exists()
    # Declares a BUNDLE-CARRIED data source: whichever root consolidated store
    # resolve_experiment_tree selects -- the unified name on a tree carrying more than
    # one -- which the config-diff plot reads directly and which the harvest chain carries into
    # the bundle (verified green by test_synth_eda_facade.py::test_bundle_eda_from_bundle,
    # the fresh-emit bundle_report_data() -> Bundle.eda() round-trip).
    manifest = out.parent / f"{out.stem}.manifest.json"
    payload = json.loads(manifest.read_text())
    # RESOLVED, never a literal. V0021 unified the root store to experiment_datatree.zarr
    # and left the retired names accepted, and the renderer declares whatever
    # config_diff_source_paths -> resolve_experiment_tree actually resolves. Pinning a
    # literal here made this assertion correct only on a tree carrying the retired name,
    # which is why it was green until the fixture tree gained the unified one.
    expected_store = resolve_experiment_tree(root).name
    assert expected_store in payload["source_paths_relative"], (
        f"the figure must declare the store the resolver resolves ({expected_store}); "
        f"declared: {payload['source_paths_relative']}"
    )
    assert payload["output_format"] == "html"


def test_unknown_eda_plot_kind_raises(synthetic_sensitivity_completed):
    """An unknown renderer-kind key fails fast at render_eda_plots."""
    import pytest

    analysis = synthetic_sensitivity_completed.experiment
    root = Path(analysis.analysis_paths.analysis_dir)
    with pytest.raises(ValueError, match="unknown EDA plot kind"):
        render_eda_plots(root, cfg_analysis=analysis.cfg_analysis, eda_cfg=eda_config(enabled_plots=["nope"]))


def test_b4b_absent_backing_zarr_emits_degradation_panel_not_skip(tmp_path):
    """A b4b figure whose backing eda/<stem>.zarr is absent MUST still emit an honest-
    degradation html (Gate-6), so an ENUMERATED b4b report target is never 'marked for report
    but does not exist'. Pre-fix, render_eda_plots' absence gate skipped it (returned []), which
    at report-assembly time raised WorkflowError. b4b renderers ignore cfg_analysis, so a bare
    tmp_path + a b4b enabled_plots entry exercises the path with no completed-run fixture."""
    (tmp_path / "eda").mkdir()  # exists but carries NO raw_byte_identity.zarr
    paths = render_eda_plots(
        tmp_path,
        cfg_analysis=None,
        eda_cfg=eda_config(enabled_plots=["raw_byte_identity"]),
    )
    assert len(paths) == 1, "the b4b figure must be emitted (degradation panel), not absence-skipped"
    out = paths[0]
    assert out == tmp_path / "plots" / "eda" / "raw_byte_identity.html"
    assert out.exists()
    assert "unavailable" in out.read_text().lower()  # the honest-degradation annotation text


def test_eda_config_drops_retired_b4b_clean_vs_resume_kind():
    """A frozen pre-F8 cfg_analysis.yaml carrying enabled_plots: [..., b4b_clean_vs_resume, ...]
    must DROP the retired kind at config-load (mode='before') with a DeprecationWarning, so the
    no-wipe report re-render's render_eda_plots no longer raises 'unknown EDA plot kind' at that
    position (F8: the within-master figure was removed; the result lives on hhemt combine)."""
    import pytest

    from hhemt.config.eda import eda_config

    with pytest.warns(DeprecationWarning, match="retired EDA figure kind"):
        cfg = eda_config(
            enabled_plots=[
                "config_diff_maps",
                "raw_byte_identity",
                "b4b_clean_vs_resume",
                "eda_rank_sensitivity",
            ]
        )
    assert "b4b_clean_vs_resume" not in cfg.enabled_plots
    assert cfg.enabled_plots == [
        "config_diff_maps",
        "raw_byte_identity",
        "eda_rank_sensitivity",
    ]


def test_eda_config_renames_legacy_eda_cross_sim_identity_kind():
    """Precedent coverage (previously untested): the eda_cross_sim_identity -> config_diff_maps
    rename still fires at config-load with a DeprecationWarning."""
    import pytest

    from hhemt.config.eda import eda_config

    with pytest.warns(DeprecationWarning):
        cfg = eda_config(enabled_plots=["eda_cross_sim_identity"])
    assert cfg.enabled_plots == ["config_diff_maps"]


def test_enumerated_config_diff_maps_emits_when_backing_artifact_absent(tmp_path):
    """An ENUMERATED report() target must produce its file even when the backing calc member
    legitimately skipped -- the enumerate-implies-emit invariant.

    Measured pre-fix failure (Rivanna smoke job 17609295): the b4b ReportingSet enumerates
    plots/eda/config_diff_maps.html unconditionally, check_cross_sim_identity skips at one
    member, render_eda_plots' absence gate skipped the kind, nothing was written, and
    Snakemake raised MissingOutputException -> workflow FAILED.

    Entry is the ADAPTER (not render_eda_plots directly), because the adapter's `output_path`
    IS the enumerated target -- so this asserts the real seam and fails PRE-FIX on BEHAVIOUR
    (no file written) rather than on a parameter name.
    """
    from types import SimpleNamespace

    from hhemt.report_renderers import eda_compute_sensitivity

    (tmp_path / "eda").mkdir()  # exists but carries NO eda_cross_sim_identity.zarr
    analysis = SimpleNamespace(
        analysis_paths=SimpleNamespace(analysis_dir=tmp_path),
        cfg_analysis=SimpleNamespace(eda=eda_config(enabled_plots=["config_diff_maps"])),
    )
    output_path = tmp_path / "plots" / "eda" / "config_diff_maps.html"

    eda_compute_sensitivity.render(analysis, None, output_path)

    assert output_path.exists(), (
        "an enumerated report() target must be produced; a skip yields Snakemake's "
        "MissingOutputException ('marked for report but does not exist')"
    )
    assert "not applicable" in output_path.read_text().lower()


def test_eda_config_renames_the_raw_byte_identity_predecessor_kind():
    """A config naming the raw byte-identity figure's pre-rename kind loads with it renamed.

    Property: an ``eda.enabled_plots`` entry naming the kind ``raw_byte_identity`` carried
    before its rename is rewritten to ``raw_byte_identity`` at config load, with a
    DeprecationWarning, and the other entries keep their order.

    Class: every saved analysis config and every bundle's baked ``cfg_analysis.yaml`` written
    before the rename. Without the rewrite, ``analysis.eda()`` skips the unknown kind
    silently and ``Bundle.eda()`` raises on it, so the figure is lost either way.

    What kills it: removing the ``raw_byte_identity`` entry from ``config/eda.py``'s
    ``_EDA_RENAMES``, misspelling its key, or mapping it to another kind; the predecessor
    list below then differs from the anchor.

    A second correct implementation under which it still passes: performing the rewrite in a
    field validator on ``enabled_plots`` instead of the model's before-validator; the
    assertions read only the map's predecessor, the loaded list and the warning.

    The old name is taken from V0026's ``OLD_STEM``, the stem the layout migration moves on
    disk, rather than spelled here or derived from the rename map. A name derived from the
    map would only show that the map agrees with itself, and a misspelled key would pass.
    """
    import importlib

    import pytest

    from hhemt.config.eda import _EDA_RENAMES, eda_config

    v0026 = importlib.import_module("hhemt.version_migration.versions.V0026__raw_byte_identity_stem")
    predecessors = [old for old, new in _EDA_RENAMES.items() if new == "raw_byte_identity"]
    assert predecessors == [v0026.OLD_STEM]
    with pytest.warns(DeprecationWarning, match="retired EDA figure kind"):
        cfg = eda_config(enabled_plots=["config_diff_maps", v0026.OLD_STEM])
    assert cfg.enabled_plots == ["config_diff_maps", "raw_byte_identity"]


def test_eda_config_rename_warning_names_the_rename_it_applied():
    """The rename DeprecationWarning names the rename actually applied, from the rename map.

    Property: loading a config whose ``enabled_plots`` carries the raw byte-identity figure's
    predecessor kind warns with ``{old} -> raw_byte_identity`` and does not name the
    ``eda_cross_sim_identity -> config_diff_maps`` rename, which was not applied.

    Class: every load that rewrites a subset of ``_EDA_RENAMES``; the message must describe
    that subset and nothing else.

    What kills it: a warning text that hard-codes one rename's arrow, or that lists the whole
    map instead of the renames applied.

    A second correct implementation under which it still passes: naming each applied rename
    on its own line, or in a different order; the assertions test only containment.
    """
    import warnings

    from hhemt.config.eda import _EDA_RENAMES, eda_config

    predecessors = [old for old, new in _EDA_RENAMES.items() if new == "raw_byte_identity"]
    assert len(predecessors) == 1, predecessors
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        eda_config(enabled_plots=[predecessors[0]])
    messages = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(messages) == 1, messages
    assert f"{predecessors[0]} -> raw_byte_identity" in messages[0]
    assert "eda_cross_sim_identity -> config_diff_maps" not in messages[0]
