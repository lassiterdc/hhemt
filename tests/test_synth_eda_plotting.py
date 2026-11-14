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
    (per-cell max_flood_depth + per-conduit
    max_flow + per-sub compute-config attrs -- no smaller per-plot summary zarr
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


def test_eda_config_refuses_retired_b4b_clean_vs_resume_kind_at_render(tmp_path):
    """A frozen pre-F8 cfg_analysis.yaml carrying enabled_plots: [..., b4b_clean_vs_resume, ...]
    loads untouched and is REFUSED at render by the kind validation, as an unknown kind is
    (F8: the within-master figure was removed; the result lives on hhemt combine). Config load
    neither drops the refused kind nor warns."""
    import warnings

    import pytest

    from hhemt.config.eda import eda_config

    plots = ["config_diff_maps", "raw_byte_identity", "b4b_clean_vs_resume", "eda_rank_sensitivity"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cfg = eda_config(enabled_plots=plots)
    assert cfg.enabled_plots == plots
    (tmp_path / "eda").mkdir()
    with pytest.raises(ValueError, match="unknown EDA plot kind 'b4b_clean_vs_resume'"):
        render_eda_plots(tmp_path, cfg_analysis=None, eda_cfg=cfg)


def test_eda_config_refuses_legacy_eda_cross_sim_identity_kind_at_render(tmp_path):
    """The eda_cross_sim_identity -> config_diff_maps rename is no longer applied at config
    load: the old kind loads as written and is refused at render. The edit that repairs a
    saved config is the new kind."""
    import pytest

    from hhemt.config.eda import eda_config

    cfg = eda_config(enabled_plots=["eda_cross_sim_identity"])
    assert cfg.enabled_plots == ["eda_cross_sim_identity"]
    with pytest.raises(ValueError, match="unknown EDA plot kind 'eda_cross_sim_identity'"):
        render_eda_plots(tmp_path, cfg_analysis=None, eda_cfg=cfg)


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


def test_eda_config_refuses_the_raw_byte_identity_predecessor_kind_at_render(tmp_path):
    """A config naming the raw byte-identity figure's pre-rename kind loads as written and is
    refused at render.

    Property: an ``eda.enabled_plots`` entry naming the kind ``raw_byte_identity`` carried
    before its rename is kept verbatim at config load, with no warning, and
    ``render_eda_plots`` refuses it as an unknown kind. The rename map still records the
    predecessor, because the render-time degradation panel and the combine guard read it.

    Class: every saved analysis config and every bundle's baked ``cfg_analysis.yaml`` written
    before the rename. The refusal is the documented behavior; the repair is editing the
    config to name the current kind.

    What kills it: removing the ``raw_byte_identity`` entry from ``config/eda.py``'s
    ``_EDA_RENAMES`` or misspelling its key (the predecessor list below then differs from the
    anchor), restoring a config-load rewrite (the list would change), or registering the
    predecessor as a renderer (the render would not raise).

    A second correct implementation under which it still passes: moving the refused-kind
    check into the renderer registry lookup with another message; the assertions read only
    the loaded list, the absence of a warning and the exception type and key.

    The old name is taken from V0026's ``OLD_STEM``, the stem the layout migration moves on
    disk, rather than spelled here or derived from the rename map. A name derived from the
    map would only show that the map agrees with itself, and a misspelled key would pass.
    """
    import importlib
    import warnings

    import pytest

    from hhemt.config.eda import _EDA_RENAMES, eda_config

    v0026 = importlib.import_module("hhemt.version_migration.versions.V0026__raw_byte_identity_stem")
    predecessors = [old for old, new in _EDA_RENAMES.items() if new == "raw_byte_identity"]
    assert predecessors == [v0026.OLD_STEM]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cfg = eda_config(enabled_plots=["config_diff_maps", v0026.OLD_STEM])
    assert cfg.enabled_plots == ["config_diff_maps", v0026.OLD_STEM]
    (tmp_path / "eda").mkdir()
    with pytest.raises(ValueError, match="unknown EDA plot kind"):
        render_eda_plots(tmp_path, cfg_analysis=None, eda_cfg=cfg)


def test_eda_refusal_of_a_retired_kind_names_the_key_and_the_known_kinds(tmp_path):
    """The refusal of a kind the render refuses names the key it refused and lists the kinds that render.

    Property: the ``ValueError`` message carries the refused key and every registered kind,
    so a reader of the failure can edit the config to a kind that exists.

    Class: every refused key; the arms below take one predecessor of a figure now under another name and
    one dropped kind, which are the two forms a refused key takes.

    What kills it: a refusal that omits the key, or one that lists no known kinds.

    A second correct implementation under which it still passes: any message format that
    contains the key and the registered kind names.
    """
    import pytest

    from hhemt.config.eda import _EDA_DROPS, _EDA_RENAMES, eda_config
    from hhemt.eda._plotting import _EDA_RENDERERS

    (tmp_path / "eda").mkdir()
    for stem in (sorted(_EDA_RENAMES)[0], sorted(_EDA_DROPS)[0]):
        with pytest.raises(ValueError) as caught:
            render_eda_plots(tmp_path, cfg_analysis=None, eda_cfg=eda_config(enabled_plots=[stem]))
        message = str(caught.value)
        assert repr(stem) in message
        for kind in _EDA_RENDERERS:
            assert kind in message
