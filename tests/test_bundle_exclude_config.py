"""ADR-20 (as amended 2026-07-14) — the governed input-exclusion opt-out.

Covers the four surfaces the amended ADR defines:

1. the excludable CATALOG and its bidirectional guard (a new self-contained Path field
   cannot be added without a catalog entry);
2. the ``BundleExcludeConfig`` operator YAML and its fail-loud validation;
3. the EMIT side — an excluded input is not carried, emits an ``input_deposit`` block, and
   gets a URL-bearing crate ``File`` part (the Patch-2 reversal);
4. the CONSUME side — the THREE-outcome materialize gate (carried / fetchable /
   referenced-but-unfetchable-by-design).

The regression that guards the whole feature: with NO exclude-config the bundle is
self-contained and the manifest carries no ``input_deposit`` key at all.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hhemt.bundle._path_policy import (
    _EXCLUDABLE_CATALOG,
    _PATH_FIELD_POLICY,
    _SELF_CONTAINED_POLICIES,
)
from hhemt.config.bundle_exclude import BundleExcludeConfig
from hhemt.exceptions import ProcessingError
from hhemt.experiments import TRITON_SWMM_experiment

# ---------------------------------------------------------------------------
# 1. Catalog
# ---------------------------------------------------------------------------


def test_all_self_contained_fields_have_catalog_entry() -> None:
    """Bidirectional guard, mirroring ``test_all_path_fields_have_policy``.

    Every cfg field the self-contained harvest carries MUST be catalogued, and the catalog
    MUST NOT name a field the harvest does not carry. Without this, a new input field would
    silently become un-excludable (invisible to ``--list-excludable``), and the operator
    would have no way to opt out of the very input that pushed them over quota.
    """
    self_contained = {
        f for f, p in _PATH_FIELD_POLICY.items() if p in _SELF_CONTAINED_POLICIES
    }
    missing = sorted(self_contained - set(_EXCLUDABLE_CATALOG))
    extra = sorted(set(_EXCLUDABLE_CATALOG) - self_contained)
    assert not missing, f"self-contained fields with no catalog entry: {missing}"
    assert not extra, f"catalog entries that are not self-contained inputs: {extra}"


def test_brand_theme_is_catalogued_non_excludable() -> None:
    """``_emit_resolved_brand_theme`` rewrites the bundled cfg to point at an emitted
    sidecar, so excluding brand_theme would emit an input_deposit block no consumer ever
    fetches — a dangling by-reference record."""
    assert _EXCLUDABLE_CATALOG["brand_theme"].excludable is False


# ---------------------------------------------------------------------------
# 2. The operator config
# ---------------------------------------------------------------------------


def test_exclude_config_accepts_fetchable_and_unfetchable() -> None:
    cfg = BundleExcludeConfig.model_validate(
        {
            "exclusions": {
                "DEM_fullres": {
                    "citation": "City LiDAR DEM",
                    "contentUrl": "https://example.org/dem.tif",
                    "identifier": "10.5281/zenodo.1",
                },
                "SWMM_hydraulics": {
                    "citation": "Licensed network — request from County GIS",
                    "url": "https://gis.example.gov/request",
                },
            }
        }
    )
    # contentUrl present/absent IS the fetchable bit — there is no separate flag.
    assert cfg.refs_for("DEM_fullres", "dem.tif").contentUrl == "https://example.org/dem.tif"
    assert cfg.refs_for("SWMM_hydraulics", "net.inp").contentUrl is None


def test_exclude_config_list_field_maps_by_basename() -> None:
    cfg = BundleExcludeConfig.model_validate(
        {"exclusions": {"static_plot_configs": {"a.yaml": {"citation": "plot a"}}}}
    )
    assert cfg.refs_for("static_plot_configs", "a.yaml").citation == "plot a"
    assert cfg.refs_for("static_plot_configs", "b.yaml") is None


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"brand_theme": {"citation": "x"}}, "NOT excludable"),
        ({"DEM_fulres": {"citation": "x"}}, "not an excludable input"),
        ({"static_plot_configs": {"citation": "x"}}, "LIST field"),
        ({"DEM_fullres": {"contentUrl": "https://x"}}, "citation"),
    ],
)
def test_exclude_config_rejects_bad_input(payload: dict, expected: str) -> None:
    """Every rejection must be LOUD. A silent no-op here is the worst failure available:
    the operator believes an input was excluded by reference, and the defect surfaces only
    after a DOI has been minted."""
    with pytest.raises(Exception, match=expected):
        BundleExcludeConfig.model_validate({"exclusions": payload})


# ---------------------------------------------------------------------------
# 3/4. Emit + consume, exercised through the real manifest/crate machinery
# ---------------------------------------------------------------------------


def _write_bundle(root: Path, *, deposits: list[dict], cfg_value: str) -> Path:
    """Minimal on-disk bundle: a manifest + a reconstituted system cfg pointing at an
    input that is NOT on disk (i.e. an excluded one)."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "bundle_manifest.json").write_text(
        json.dumps({"analysis_id": "x", "input_deposit": deposits})
    )
    (root / "cfg_system.yaml").write_text(f"DEM_fullres: {cfg_value}\n")
    (root / "cfg_analysis.yaml").write_text("weather_timeseries: null\n")
    return root


def test_unfetchable_input_fails_closed_with_its_citation(tmp_path: Path) -> None:
    """Outcome 3 — referenced-but-unfetchable-by-design.

    This is the case the whole opt-out exists for (licensed / IP data the toolkit MUST NOT
    redistribute). It must fail CLOSED, and the error must carry the citation — a bare
    'missing input' message is useless to someone who needs to know WHERE to get the file.
    """
    root = tmp_path / "bundle"
    missing_input = root / "external" / "sewer.inp"
    _write_bundle(
        root,
        deposits=[
            {
                "relpath": "external/sewer.inp",
                "sha256": "abc123",
                "accessed": "2026-07-14",
                "citation": "Licensed municipal sewer network. Request from County GIS.",
                "url": "https://gis.example.gov/request",
            }
        ],
        cfg_value=str(missing_input),
    )

    with pytest.raises(ProcessingError) as exc:
        TRITON_SWMM_experiment._assert_declared_inputs_exist(
            root / "cfg_system.yaml", root / "cfg_analysis.yaml"
        )

    msg = str(exc.value)
    assert "REFERENCED, not carried" in msg
    assert "Request from County GIS" in msg  # the citation reached the operator
    assert "https://gis.example.gov/request" in msg
    assert "fail-closed stop, not a corrupt bundle" in msg


def test_fetchable_input_is_fetched_and_sha_verified(tmp_path: Path, monkeypatch) -> None:
    """Outcome 2 — an absent input WITH a contentUrl is fetched through the per-file seam
    and sha256-verified BEFORE the fail-closed enumeration runs."""
    root = tmp_path / "bundle"
    dem = root / "external" / "dem.tif"
    _write_bundle(
        root,
        deposits=[
            {
                "relpath": "external/dem.tif",
                "sha256": "deadbeef",
                "accessed": "2026-07-14",
                "citation": "City DEM",
                "contentUrl": "https://example.org/dem.tif",
            }
        ],
        cfg_value=str(dem),
    )

    calls: list[tuple[str, str | None]] = []

    @classmethod
    def _fake_fetch(cls, url, dest, *, expected_sha256=None):  # noqa: ANN001
        calls.append((url, expected_sha256))
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(b"raster")
        return Path(dest)

    monkeypatch.setattr(TRITON_SWMM_experiment, "_fetch_file_by_url", _fake_fetch)

    # Must NOT raise: the input is materialized by the fetch.
    TRITON_SWMM_experiment._assert_declared_inputs_exist(
        root / "cfg_system.yaml", root / "cfg_analysis.yaml"
    )

    assert calls == [("https://example.org/dem.tif", "deadbeef")]
    assert dem.exists()


def test_fetch_failure_is_terminal_not_a_silent_skip(tmp_path: Path, monkeypatch) -> None:
    """A rotted reference or a digest mismatch must be a HARD failure. A silent skip would
    hand the consumer a plausible-looking run built on a missing or wrong input."""
    root = tmp_path / "bundle"
    dem = root / "external" / "dem.tif"
    _write_bundle(
        root,
        deposits=[
            {
                "relpath": "external/dem.tif",
                "sha256": "deadbeef",
                "accessed": "2026-07-14",
                "citation": "City DEM — see landing page",
                "contentUrl": "https://example.org/gone.tif",
            }
        ],
        cfg_value=str(dem),
    )

    @classmethod
    def _boom(cls, url, dest, *, expected_sha256=None):  # noqa: ANN001
        raise RuntimeError("404")

    monkeypatch.setattr(TRITON_SWMM_experiment, "_fetch_file_by_url", _boom)

    with pytest.raises(ProcessingError, match="BY REFERENCE"):
        TRITON_SWMM_experiment._assert_declared_inputs_exist(
            root / "cfg_system.yaml", root / "cfg_analysis.yaml"
        )
