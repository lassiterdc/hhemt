"""Phase-1 unit tests for hhemt.eda.load_eda_context (ADR-13, FQ1 absence contract)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

# Committed render-bundle fixture carrying a real, transported cfg_analysis.yaml
# (valid as-is) and a redacted cfg_system.yaml (software dirs nulled for
# transport). The positive-path test re-roots it into a tmp analysis root rather
# than hand-constructing the heavyweight system_config required-field surface.
_BUNDLE_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "bundles" / "multi_sim"


def test_load_eda_context_raises_on_missing_cfg_analysis(tmp_path):
    from hhemt.eda import load_eda_context

    with pytest.raises(FileNotFoundError, match="not a valid EDA root"):
        load_eda_context(tmp_path)


def test_load_eda_context_raises_on_missing_cfg_system(tmp_path):
    from hhemt.eda import load_eda_context

    (tmp_path / "cfg_analysis.yaml").write_text("placeholder: true\n")
    # cfg_analysis.yaml present but cfg_system.yaml absent -> raises on cfg_system
    with pytest.raises(FileNotFoundError, match="cfg_system.yaml absent"):
        load_eda_context(tmp_path)


def test_is_bundle_discriminator(tmp_path):
    """is_bundle is purely the bundle_manifest.json presence check on the root."""
    from hhemt.eda._context import load_eda_context

    # A valid (non-bundle) root: no bundle_manifest.json -> is_bundle False.
    _write_minimal_valid_root(tmp_path)
    assert load_eda_context(tmp_path).is_bundle is False

    # Drop the bundle marker -> the same root now discriminates as a bundle.
    (tmp_path / "bundle_manifest.json").write_text("{}")
    assert load_eda_context(tmp_path).is_bundle is True


def _write_minimal_valid_root(root: Path) -> None:
    """Materialize a MINIMAL-but-VALID, non-bundle EDA root at ``root``.

    Re-roots the committed ``multi_sim`` bundle fixture configs into ``root``:
    ``cfg_analysis.yaml`` is valid as-is; ``cfg_system.yaml`` only needs its one
    transport-redacted required path (``TRITONSWMM_software_directory``) restored
    and ``system_directory`` re-pointed at ``root`` so the loader's DEM/GIS
    resolution keys off the constructed root. Round-tripped through ``model_dump``
    (DoD: write real cfg_*.yaml via model_dump) so the on-disk YAML re-validates.
    """
    from hhemt.config.analysis import analysis_config
    from hhemt.config.system import system_config

    sys_raw = yaml.safe_load((_BUNDLE_FIXTURE / "cfg_system.yaml").read_text())
    ana_raw = yaml.safe_load((_BUNDLE_FIXTURE / "cfg_analysis.yaml").read_text())

    sys_raw["system_directory"] = str(root)
    if sys_raw.get("TRITONSWMM_software_directory") is None:
        sys_raw["TRITONSWMM_software_directory"] = str(root / "triton_software")
    # Drop the four Phase-4-retired HPC keys the legacy bundle still carries so
    # this fixture survives the eventual removal of the pop-and-warn shim in
    # system_config.validate_toggle_dependencies (keeps the test forward-clean).
    for _retired in (
        "gpu_hardware",
        "gpu_compilation_backend",
        "preferred_slurm_option_for_allocating_gpus",
        "additional_modules_needed_to_run_TRITON_SWMM_on_hpc",
    ):
        sys_raw.pop(_retired, None)

    cfg_system = system_config.model_validate(sys_raw)
    cfg_analysis = analysis_config.model_validate(ana_raw)

    (root / "cfg_system.yaml").write_text(
        yaml.safe_dump(cfg_system.model_dump(mode="json"), sort_keys=False)
    )
    (root / "cfg_analysis.yaml").write_text(
        yaml.safe_dump(cfg_analysis.model_dump(mode="json"), sort_keys=False)
    )


def test_load_eda_context_positive_path(tmp_path):
    """Populated-EdaContext path (hhemt-specialist review): exercises the config
    load surface plus the system_directory-relative ``swmm_features``/``triton_dem``
    resolution the two RAISE-on-missing-config tests cannot reach — the surface the
    Flag-1/2 (non-field ``dem_processed`` / wrong GIS root) defects would have
    tripped at Phase 1."""
    import geopandas as gpd
    from shapely.geometry import Point

    from hhemt.config.analysis import analysis_config
    from hhemt.config.system import system_config
    from hhemt.eda import EdaContext, load_eda_context

    _write_minimal_valid_root(tmp_path)

    # SWMM GIS layers resolve under {system_directory}/gis (== tmp_path/gis here).
    # Drop one real layer to positively confirm the resolution keys off
    # system_directory rather than trivially returning None.
    gis_dir = tmp_path / "gis"
    gis_dir.mkdir()
    gpd.GeoDataFrame({"id": [1]}, geometry=[Point(0, 0)], crs="EPSG:4326").to_file(
        gis_dir / "conduits.gpkg", driver="GPKG"
    )

    ctx = load_eda_context(tmp_path)

    assert isinstance(ctx, EdaContext)
    # Config-load surface: both configs parsed to their models; the re-rooted
    # system_directory survived the model_dump round-trip.
    assert isinstance(ctx.cfg_analysis, analysis_config)
    assert isinstance(ctx.cfg_system, system_config)
    assert Path(ctx.cfg_system.system_directory) == tmp_path
    # Not a bundle root (no bundle_manifest.json written).
    assert ctx.is_bundle is False
    # swmm_features resolved via system_directory/gis -> the one dropped layer.
    assert ctx.swmm_features is not None
    assert set(ctx.swmm_features) == {"conduits"}
    assert isinstance(ctx.swmm_features["conduits"], gpd.GeoDataFrame)
    # triton_dem resolution exercises cfg_system.system_directory +
    # target_dem_resolution (the real fields, not the Flag-1 non-field
    # dem_processed); the DEM artifact is absent -> None per the absence contract.
    assert ctx.triton_dem is None
    # Remaining data artifacts are legitimately absent at this minimal root.
    assert ctx.datatree is None
    assert ctx.sensitivity_datatree is None  # toggle_sensitivity_analysis is False
    assert ctx.scenario_status is None
    assert ctx.performance is None  # None exactly when datatree is None
