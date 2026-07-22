"""Unit tests for the Metadata report renderer (ADR-14 / C10).

HPC-free / compile-free: the renderer's only inputs are the persisted RO-Crate
sidecar and (optionally) the SLURM efficiency CSV, so every case here is driven
by a CRAFTED sidecar and a minimal analysis stand-in. Crafting the crate is not
a convenience — it is the only way to prove the R3 volatile-field exclusion (we
must plant a KNOWN hostname/wall-clock sentinel and assert it never surfaces)
and to exercise the R7 crate-shape branches (native run, sensitivity master,
empty inputs) that a single real fixture cannot produce.

End-to-end integration on the production render path is covered separately by
tests/test_synth_04_multisim_with_snakemake.py and
tests/test_synth_05_sensitivity_analysis_with_snakemake.py.
"""

from __future__ import annotations

import inspect
import json
import types
from pathlib import Path

import pytest

from hhemt.config.report import report_config
from hhemt.report_renderers import metadata

# Sentinels planted in the crafted crate's VOLATILE fields. If either ever
# reaches the rendered HTML, a bundle-shipped Metadata page would disclose the
# producing machine and wall-clock (R3 / C-ZERO-USER-INFO).
_SENTINEL_HOST = "SECRET-PRODUCER-HOST"
_SENTINEL_TIME = "2026-01-01T03:04:05"

_ANCHORS = ("provenance", "reproduction-guide", "slurm-efficiency")

_ROOT = {
    "@id": "./",
    "@type": "Dataset",
    "name": "norfolk_coastal_flooding",
    "description": "case description",
    "analysis_id": "A1",
    "system_id": "S1",
    "schemaVersion": "16",
    "license": {"@id": "https://spdx.org/licenses/CC0-1.0"},
}
_DESCRIPTOR = {"@id": "ro-crate-metadata.json", "@type": "CreativeWork"}
_LICENSE = {"@id": "https://spdx.org/licenses/CC0-1.0", "@type": "CreativeWork", "name": "CC0-1.0"}
_TOOLKIT_SRC = {
    "@id": "#hhemt-toolkit-src",
    "@type": "SoftwareSourceCode",
    # Deliberately carries HTML metacharacters: the renderer must escape them.
    "name": "H&H <Ensemble> Toolkit",
    "codeRepository": "https://github.com/lassiterdc/hhemt",
    "version": "abc123def",
}
_APP = {"@id": "#hhemt-app", "@type": "SoftwareApplication", "name": "hhemt", "softwareVersion": "abc123def"}
_SIF = {
    "@id": "#sif",
    "@type": "SoftwareApplication",
    "name": "TRITON-SWMM Apptainer container",
    "softwareVersion": "1.0",
    "sha256": "deadbeefcafe",
    "downloadUrl": "https://example.org/tritonswmm.sif",
}
_INPUT_FILE = {
    "@id": "inputs/dem.tif",
    "@type": "File",
    "sha256": "f00dfeed",
    "contentSize": "1024",
    "encodingFormat": "image/tiff",
}
_VAR = {
    "@id": "#var-max_wlevel_m",
    "@type": "PropertyValue",
    "name": "max_wlevel_m",
    "description": "maximum water surface elevation",
    "unitText": "m",
    "propertyID": "water_surface_height_above_reference_datum",
    "measurementTechnique": "time: maximum",
}
_ZARR = {
    "@id": "analysis_datatree.zarr",
    "@type": "Dataset",
    "name": "Consolidated analysis DataTree (zarr)",
    "encodingFormat": "application/x-zarr",
    "conformsTo": {"@id": "https://example.org/cf-profile"},
    "variableMeasured": [{"@id": "#var-max_wlevel_m"}],
}
_RUN = {
    "@id": "#run-none-evt0-triton",
    "@type": "CreateAction",
    "name": "TRITON-SWMM run evt0 (triton)",
    "instrument": [{"@id": "#hhemt-app"}, {"@id": "#sif"}],
    "object": [{"@id": "inputs/dem.tif"}],
    "result": [{"@id": "analysis_datatree.zarr"}],
    # VOLATILE — present in the real sidecar; must never be projected.
    "startTime": _SENTINEL_TIME,
    "agent": {"@id": f"#agent-{_SENTINEL_HOST}"},
}


def _crate(*entities: dict) -> dict:
    return {"@context": "https://w3id.org/ro/crate/1.1/context", "@graph": list(entities)}


def _full_crate() -> dict:
    """A container-run multisim crate: every sub-section has content."""
    return _crate(_ROOT, _DESCRIPTOR, _LICENSE, _TOOLKIT_SRC, _APP, _SIF, _INPUT_FILE, _VAR, _ZARR, _RUN)


def _fake_analysis(analysis_dir: Path):
    """render() reads exactly two attributes off the analysis object."""
    return types.SimpleNamespace(
        analysis_paths=types.SimpleNamespace(analysis_dir=analysis_dir),
        cfg_analysis=types.SimpleNamespace(analysis_id="A1"),
    )


def _render(tmp_path: Path, *, doc: dict | None = None, slurm_csv: str | None = None) -> tuple[str, dict, Path]:
    """Render into a fresh analysis_dir; return (html, manifest, analysis_dir)."""
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    if doc is not None:
        (analysis_dir / "ro-crate-metadata.json").write_text(json.dumps(doc))
    if slurm_csv is not None:
        # Faithful to snakemake-executor-plugin-slurm: --slurm-efficiency-report-path
        # is treated as a DIRECTORY, so the driver's `.csv`-suffixed path materializes
        # on disk as a directory that CONTAINS the real efficiency_report_{uuid}.csv.
        eff_dir = analysis_dir / "logs" / "slurm_efficiency_report"
        nested = eff_dir / "slurm_efficiency_report_20260101T000000.csv"
        nested.mkdir(parents=True, exist_ok=True)
        (nested / "efficiency_report_deadbeef.csv").write_text(slurm_csv)

    output_path = analysis_dir / "plots" / "metadata.html"
    metadata.render(_fake_analysis(analysis_dir), report_config(), output_path)
    manifest = json.loads((analysis_dir / "plots" / "metadata.manifest.json").read_text())
    return output_path.read_text(), manifest, analysis_dir


# --- R2: page structure ------------------------------------------------------


def test_renderer_emits_three_sections(tmp_path):
    """R2: the page carries the three <h3 id> sub-section headings."""
    html, _, _ = _render(tmp_path, doc=_full_crate())
    for anchor in _ANCHORS:
        assert f'id="{anchor}"' in html


def test_page_title_and_jump_nav(tmp_path):
    """R2: <h2> page title plus a jump-nav anchoring each sub-section."""
    html, _, _ = _render(tmp_path, doc=_full_crate())
    assert "<h2>Metadata — A1</h2>" in html
    for anchor in _ANCHORS:
        assert f'href="#{anchor}"' in html


# --- R3: provenance source + volatile-field exclusion -------------------------


def test_provenance_declares_only_sidecar(tmp_path):
    """R3: the sole declared source is ro-crate-metadata.json when no SLURM CSV exists."""
    _, manifest, _ = _render(tmp_path, doc=_full_crate())
    assert manifest["source_paths_relative"] == ["ro-crate-metadata.json"]
    assert manifest["plot_id"] == "metadata"


def test_volatile_fields_never_reach_the_rendered_page(tmp_path):
    """R3: the crafted sidecar's startTime + agent(hostname) must not surface.

    Asserting against a planted sentinel — rather than the real fixture's own
    hostname — is what makes this check meaningful: it fails loudly if a future
    maintainer widens the projection to dump the @graph.
    """
    doc = _full_crate()
    raw = json.dumps(doc)
    assert _SENTINEL_HOST in raw and _SENTINEL_TIME in raw, "sentinels must be present in the source crate"

    html, _, _ = _render(tmp_path, doc=doc)
    assert _SENTINEL_HOST not in html
    assert _SENTINEL_TIME not in html
    # The run unit itself IS surfaced — count + instrument->result edges only.
    assert "run unit(s) recorded" in html
    assert "#hhemt-app" in html


def test_prop_refuses_volatile_keys():
    """R3 fail-closed backstop: reaching for a volatile key is a hard error."""
    for key in ("startTime", "endTime", "agent"):
        with pytest.raises(ValueError, match="volatile"):
            metadata._prop({key: "x"}, key)


def test_producer_absolute_path_never_leaks(tmp_path):
    """C-ZERO-USER-INFO: the producing analysis_dir must not appear in the page."""
    html, _, analysis_dir = _render(tmp_path, doc=_full_crate())
    assert str(analysis_dir) not in html


def test_dynamic_values_are_html_escaped(tmp_path):
    """Spec 1: RO-Crate values carrying HTML metacharacters are escaped, not injected."""
    html, _, _ = _render(tmp_path, doc=_full_crate())
    assert "H&amp;H &lt;Ensemble&gt; Toolkit" in html
    assert "<Ensemble>" not in html


def test_verifiability_anchors_and_chain_are_rendered(tmp_path):
    """R3: the BLUF anchors + the recreation chain are projected from the crate."""
    html, _, _ = _render(tmp_path, doc=_full_crate())
    assert "Verifiability anchors" in html
    assert "abc123def" in html  # toolkit git SHA
    assert "deadbeefcafe" in html  # SIF sha256
    assert "f00dfeed" in html  # input digest
    assert "CC0-1.0" in html  # dataset license
    assert "max_wlevel_m" in html  # CF data-dictionary row
    assert "water_surface_height_above_reference_datum" in html


# --- R4: reproduction guide, zero-user-info ----------------------------------


def test_reprex_guide_takes_no_analysis_argument():
    """R4 (structural): the guide cannot leak producer values because it never sees them.

    Zero-user-info is enforced by the signature, not by discipline. If a future
    maintainer re-introduces an `analysis` parameter, the capability to leak
    returns and this test fails.
    """
    assert list(inspect.signature(metadata._build_reprex_guide_html).parameters) == []
    assert list(inspect.signature(metadata._config_field_rows).parameters) == []


def test_reprex_guide_groups_every_field_into_three_buckets(tmp_path):
    """R4: Supply / Amend / Keep blocks, each with its instruction verb."""
    html, _, _ = _render(tmp_path, doc=_full_crate())
    for verb in ("Supply", "Amend", "Keep"):
        assert verb in html
    assert "USER" in html and "HPC" in html and "EXPERIMENT" in html


def test_reprex_guide_renders_placeholders_not_values(tmp_path):
    """R4: value cells are placeholders / schema descriptions only."""
    html, _, _ = _render(tmp_path, doc=_full_crate())
    assert "{amend for your target system}" in html
    assert "{inherit — carried by the bundle}" in html
    assert "{your-default_account}" in html


def test_reprex_guide_covers_a_field_from_each_bucket():
    """R4: the taxonomy actually classifies representative fields into all three buckets."""
    rows_by_bucket, unclassified = metadata._config_field_rows()
    assert not unclassified, f"unbucketed config fields: {unclassified}"
    assert rows_by_bucket["user"], "expected at least one USER-bucket field"
    assert rows_by_bucket["hpc"], "expected at least one HPC-bucket field"
    assert rows_by_bucket["experiment"], "expected at least one EXPERIMENT-bucket field"


def _labels(rows: list[list[str]]) -> set[str]:
    """Strip the <code> wrapper off each row's Field cell."""
    return {r[0].removeprefix("<code>").removesuffix("</code>") for r in rows}


def test_supply_block_names_the_fields_a_reproducer_actually_supplies():
    """R4 + research Q2: the Supply block is the reprex_config supply set, not just config paths.

    `all_field_bucket` is total over system_config | analysis_config, and over
    that domain only the two software-directory paths are USER. The fields a
    target user literally types — account, login node, SIF path, scratch dir —
    live on `reprex_config`. Omitting them would render a Supply block that
    omits everything you must supply.
    """
    rows_by_bucket, _ = metadata._config_field_rows()
    user_labels = _labels(rows_by_bucket["user"])
    for field in ("default_account", "login_node", "sif_path", "scratch_dir"):
        assert f"reprex_config.{field}" in user_labels
    # The two config Path fields that bucket USER remain present.
    assert "system_config.TRITONSWMM_software_directory" in user_labels
    assert "system_config.SWMM_software_directory" in user_labels


def test_partition_selectors_are_amend_not_supply():
    """reprex_config's two target_* selectors are HPC-revisable, not host-local."""
    rows_by_bucket, _ = metadata._config_field_rows()
    hpc_labels = _labels(rows_by_bucket["hpc"])
    user_labels = _labels(rows_by_bucket["user"])
    for field in metadata._REPREX_SELECTOR_FIELDS:
        assert f"reprex_config.{field}" in hpc_labels
        assert f"reprex_config.{field}" not in user_labels


def test_every_config_field_appears_exactly_once():
    """R4: the guide is exhaustive over both configs — nothing silently dropped."""
    from hhemt.config.analysis import analysis_config
    from hhemt.config.system import system_config

    rows_by_bucket, unclassified = metadata._config_field_rows()
    assert not unclassified
    all_labels: list[str] = []
    for bucket in metadata._BUCKET_ORDER:
        all_labels.extend(_labels(rows_by_bucket[bucket]))

    expected = {f"system_config.{f}" for f in system_config.model_fields}
    expected |= {f"analysis_config.{f}" for f in analysis_config.model_fields}
    missing = expected - set(all_labels)
    assert not missing, f"config fields missing from the reproduction guide: {sorted(missing)}"
    assert len(all_labels) == len(set(all_labels)), "a field was rendered twice"


# --- R5: SLURM efficiency ----------------------------------------------------


def test_slurm_section_renders_table_and_declares_the_csv_file(tmp_path):
    """R5: the globbed CSV is rendered AND declared as a source — the FILE, not the dir."""
    csv_text = "rule,job_id,cpu_efficiency\nplot_metadata,12345,91.2%\n"
    html, manifest, _ = _render(tmp_path, doc=_full_crate(), slurm_csv=csv_text)
    assert "cpu_efficiency" in html
    assert "91.2%" in html
    declared = manifest["source_paths_relative"]
    assert "ro-crate-metadata.json" in declared
    assert any(p.endswith(".csv") for p in declared), declared
    # Declaring the DIRECTORY would raise in _validate_source_path once it exists.
    assert "logs/slurm_efficiency_report" not in declared


def test_slurm_csv_with_header_only_degrades(tmp_path):
    """R5: a header-only CSV yields the heading + an info banner, not an empty table."""
    html, _, _ = _render(tmp_path, doc=_full_crate(), slurm_csv="rule,cpu_efficiency\n")
    assert 'id="slurm-efficiency"' in html
    assert "no job rows" in html


# --- R7: graceful degradation across every absent-source state ---------------


def test_absent_sidecar_degrades_gracefully(tmp_path):
    """R7: no sidecar -> heading present, .banner.info body, source still declared."""
    html, manifest, _ = _render(tmp_path, doc=None)
    assert 'id="provenance"' in html
    assert "banner info" in html
    assert "ro-crate-metadata.json" in html
    # ADR-6 D3: the expected source is declared even when absent.
    assert manifest["source_paths_relative"] == ["ro-crate-metadata.json"]


def test_native_run_crate_has_no_sif_entity(tmp_path):
    """R7: native run (sif_spec=None) -> explicit reduced-verifiability note, not a blank."""
    html, _, _ = _render(tmp_path, doc=_crate(_ROOT, _DESCRIPTOR, _APP))
    assert "Native run" in html
    assert "reduced verifiability" in html


def test_sensitivity_master_crate_has_no_run_units(tmp_path):
    """R7: a master crate is emitted with with_run_units=False -> no CreateAction nodes."""
    html, _, _ = _render(tmp_path, doc=_crate(_ROOT, _DESCRIPTOR, _APP, _SIF))
    assert "consolidation-level crate" in html
    assert _SENTINEL_HOST not in html


def test_empty_input_parts_degrades_gracefully(tmp_path):
    """R7: no by-reference File parts -> 'digests not captured', not 'there were no inputs'."""
    html, _, _ = _render(tmp_path, doc=_crate(_ROOT, _DESCRIPTOR, _APP))
    assert "Input digests not captured" in html


def test_absent_slurm_csv_degrades_gracefully(tmp_path):
    """R7: no efficiency CSV -> heading present, teardown-timing explained."""
    html, manifest, _ = _render(tmp_path, doc=_full_crate())
    assert 'id="slurm-efficiency"' in html
    assert "teardown" in html
    assert not any(p.endswith(".csv") for p in manifest["source_paths_relative"])


def test_minimal_crate_renders_without_exception(tmp_path):
    """R7: the worst case (bare root + descriptor) still renders all three sections."""
    html, _, _ = _render(tmp_path, doc=_crate(_ROOT, _DESCRIPTOR))
    for anchor in _ANCHORS:
        assert f'id="{anchor}"' in html


def test_type_may_be_a_list(tmp_path):
    """RO-Crate permits @type to be a list; the bundle upgrade emits such nodes."""
    workflow_entity = {"@id": "Snakefile.source", "@type": ["File", "ComputationalWorkflow"]}
    html, _, _ = _render(tmp_path, doc=_crate(_ROOT, _DESCRIPTOR, _APP, workflow_entity))
    # The workflow entity carries no sha256 and must not be listed as an input digest.
    assert "Input digests not captured" in html
    assert 'id="provenance"' in html


# --- R5 regression (Q8): efficiency report is a `.csv`-named DIRECTORY ---------


def test_slurm_report_path_is_a_directory_not_a_file(tmp_path):
    """Regression (Q8): the plugin writes efficiency_report_{uuid}.csv INSIDE a
    `.csv`-named directory; the renderer must descend to the inner file and must
    NOT raise IsADirectoryError on read_text()."""
    csv_text = "rule,job_id,cpu_efficiency\nplot_metadata,12345,91.2%\n"
    html, manifest, _ = _render(tmp_path, doc=_full_crate(), slurm_csv=csv_text)
    assert "cpu_efficiency" in html and "91.2%" in html
    declared = manifest["source_paths_relative"]
    assert any(p.endswith("efficiency_report_deadbeef.csv") for p in declared), declared


def test_resolve_latest_efficiency_csv_flat_nested_and_absent(tmp_path):
    """Unit: resolver returns None when empty, the inner file for the nested
    plugin layout, and a flat file for the hypothetical future layout."""
    from hhemt.report_renderers.metadata import _resolve_latest_efficiency_csv

    eff = tmp_path / "logs" / "slurm_efficiency_report"
    eff.mkdir(parents=True)
    assert _resolve_latest_efficiency_csv(eff) is None
    nested = eff / "slurm_efficiency_report_20260101T000000.csv"
    nested.mkdir()
    (nested / "efficiency_report_uuid.csv").write_text("a,b\n1,2\n")
    got = _resolve_latest_efficiency_csv(eff)
    assert got is not None and got.is_file() and got.name == "efficiency_report_uuid.csv"
