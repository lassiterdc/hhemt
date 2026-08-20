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

import copy
import inspect
import json
import re
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

# Derived from the producer, not restated. The literal form went stale the moment the
# Data Availability section landed: nothing failed, the section simply lost its
# heading/jump-nav coverage in every test that iterates this tuple.
_ANCHORS = tuple(metadata._anchor(t) for t in metadata._SECTION_TITLES)

# The REAL header snakemake-executor-plugin-slurm writes (efficiency_report.py:
# parse_sacct_data -> df.to_csv, which emits the unnamed index column first). Tests
# used to invent an ad-hoc `rule,job_id,cpu_efficiency` header, which no longer
# survives the renderer's curated column projection -- and should not, since a
# fabricated schema cannot catch a real schema drift.
_EFF_HEADER = (
    ",JobID,JobName,Elapsed,NNodes,NCPUS,MaxRSS,ReqMem,Elapsed_sec,TotalCPU_sec,"
    "MaxRSS_MB,RequestedMem_MB,MainJobID,CPU Efficiency (%),Memory Usage (%)\n"
)
#: One realistic data row: a job step, as the plugin emits it (parent + .batch rows
#: are dropped upstream, which is why JobName reads `python` rather than a rule name).
_EFF_ROW = "2,18573918.0,python,00:00:23,1,1,710788K,,23.0,0.0,694.12,1000.0,18573918,0.0,69.41\n"

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


def test_provenance_declares_the_sidecar_and_nothing_undeclared(tmp_path):
    """R3: the PROVENANCE sub-section's declared source is the RO-Crate sidecar.

    R3 is scoped to that sub-section, not to the page -- both in the plan
    (reproducibility-system_metadata-report-section.md:65, "The provenance sub-section
    reads ONLY ...") and in the durable stipulation ("its single PROVENANCE
    source_paths entry"). The page legitimately grew a fourth sub-section that reads
    validation_report.json, which Gotcha 53's IO audit REQUIRES it to declare. So the
    checkable claim is membership plus a closed set, never a snapshot of the list:
    the closed set is derived from the renderer's own filename constants, so adding a
    reader means adding a constant at the producer, in view of a reviewer.
    """
    _, manifest, _ = _render(tmp_path, doc=_full_crate())
    declared = manifest["source_paths_relative"]
    assert metadata._SIDECAR_FILENAME in declared
    allowed = {metadata._SIDECAR_FILENAME, metadata._VALIDATION_REPORT_FILENAME}
    assert set(declared) <= allowed, f"undeclared-source drift: {sorted(set(declared) - allowed)}"
    assert len(declared) == len(set(declared)), f"duplicate declarations: {declared}"
    assert manifest["plot_id"] == "metadata"


def test_status_sidecars_and_tree_are_declared_when_they_exist(tmp_path):
    """Every `_status/*.flag.json` the page OPENS is declared (declared ⊆ actual).

    Globbing is audit-invisible (os.scandir), but `read_text()` is not — so a sidecar
    that is read and not declared would break the ADR-6 invariant, and one that is
    declared and not read would put a false claim in the bundle manifest.
    """
    analysis_dir = tmp_path / "analysis"
    (analysis_dir / "_status").mkdir(parents=True)
    (analysis_dir / "ro-crate-metadata.json").write_text(json.dumps(_full_crate()))
    for name, payload in (
        ("a_setup_target_0_complete.flag.json", {"rule_name": "setup_target_0", "slurm_job_id": "1"}),
        ("c_run_complete.flag.json", {"rule_name": "simulation_sa_x_evt_0", "slurm_job_id": "2"}),
    ):
        (analysis_dir / "_status" / name).write_text(json.dumps(payload))

    output_path = analysis_dir / "plots" / "metadata.html"
    metadata.render(_fake_analysis(analysis_dir), report_config(), output_path)
    manifest = json.loads((analysis_dir / "plots" / "metadata.manifest.json").read_text())
    declared = manifest["source_paths_relative"]

    assert "_status/a_setup_target_0_complete.flag.json" in declared
    assert "_status/c_run_complete.flag.json" in declared
    assert manifest["renderer_data"]["status_flag_count"] == 2

    # And the timeline they back is actually rendered, with derived purposes.
    html = output_path.read_text()
    assert "Run timeline" in html
    assert "compile / setup" in html and "simulate" in html


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


def _user_bucket_field_labels() -> list[str]:
    """Every config field the reprex taxonomy buckets as USER, derived, not hardcoded.

    Enumerated from `reprex_taxonomy.all_field_bucket` — the SAME classifier
    `bundle/_emit.py::_scrub_user_bucket_fields` trusts to decide what the bundle
    nulls out — plus `reprex_config`'s own host-local fields, which are not keys of
    that classifier (it is total over system_config | analysis_config only) and whose
    USER/HPC split is declared by `metadata._REPREX_SELECTOR_FIELDS`.

    Deriving the set is what makes the leak test exhaustive: a NEW user-bucket field
    is automatically poisoned and automatically checked, with no test edit.
    """
    from hhemt.config import reprex_taxonomy
    from hhemt.config.analysis import analysis_config
    from hhemt.config.reprex_config import reprex_config
    from hhemt.config.system import system_config

    labels = [
        f"reprex_config.{name}" for name in reprex_config.model_fields if name not in metadata._REPREX_SELECTOR_FIELDS
    ]
    for config_label, model in (("system_config", system_config), ("analysis_config", analysis_config)):
        labels += [
            f"{config_label}.{name}" for name in model.model_fields if reprex_taxonomy.all_field_bucket(name) == "user"
        ]
    return labels


def test_config_field_rows_still_takes_no_analysis_argument():
    """R4 (structural): the SCHEMA reader cannot leak a value because it never sees one.

    This is the half of the original zero-user-info guard that survives unchanged.
    `_config_field_rows` reads `model_fields` and nothing else; with no values in
    hand it cannot disclose any field, enumerated or not. If a maintainer ever gives
    it an `analysis` parameter, the capability returns and this fails.
    """
    assert list(inspect.signature(metadata._config_field_rows).parameters) == []


def _guide_text(guide) -> str:
    """Every byte the guide contributes to the rendered page.

    [Q148] moved the three tables to Tabulator, so cell CONTENT now lives in each
    fragment's `script` (a JSON options blob) rather than in the markup string. A leak
    scan over `guide.html` alone would be blind exactly where the values moved to --
    and, because `guide` is a NamedTuple, `"needle" in guide` silently evaluates
    element-identity and returns False, so such a test would PASS while checking
    nothing. Concatenate all four surfaces.
    """
    return "\n".join(
        [guide.html]
        + [f.styles for _, f in guide.fragments]
        + [f.markup for _, f in guide.fragments]
        + [f.script for _, f in guide.fragments]
    )


def test_reprex_guide_never_renders_a_user_bucket_value():
    """R4 (behavioural): no USER-bucket value reaches the page, for EVERY such field.

    `_build_reprex_guide_html` now accepts a pre-computed {field -> value} map so the
    HPC/EXPERIMENT values — which the bundle already ships in cfg_*.yaml, after
    `_scrub_user_bucket_fields` nulls the USER ones — can be displayed. The signature
    check alone can no longer express the guarantee, so it is asserted behaviourally
    here, and asserted over the DERIVED user-bucket set rather than a hand-picked
    sample: a unique sentinel is planted in every USER field and none may surface.

    Together with the signature check above this dominates the original guard on both
    axes — it closes the module-global bypass a signature check cannot see, and it has
    no unenumerated-field hole a hardcoded sample would have.
    """
    user_labels = _user_bucket_field_labels()
    assert user_labels, "derived USER bucket is empty — the enumeration is broken, not clean"

    poisoned = {label: f"SENTINEL-LEAK-{i}" for i, label in enumerate(user_labels)}
    html = _guide_text(metadata._build_reprex_guide_html(poisoned))
    leaked = [label for i, label in enumerate(user_labels) if f"SENTINEL-LEAK-{i}" in html]
    assert not leaked, f"USER-bucket values reached the rendered page: {leaked}"


def test_derived_user_bucket_matches_the_renderer_s_own_supply_block():
    """The leak test's population is the same one the renderer calls USER.

    Derived independently (from `all_field_bucket`) so it is not circular: if the
    renderer's bucketing and the taxonomy ever disagree, the leak test would be
    poisoning a different set than the page renders, and this catches that directly
    rather than letting the leak test quietly go vacuous.
    """
    rows_by_bucket, unclassified = metadata._config_field_rows()
    assert not unclassified, f"unbucketed config fields: {unclassified}"
    rendered_user = {metadata._strip_code(row[0]) for row in rows_by_bucket["user"]}
    assert rendered_user == set(_user_bucket_field_labels())


def test_reprex_guide_groups_every_field_into_three_buckets(tmp_path):
    """R4: Supply / Amend / Keep blocks, each with its instruction verb."""
    html, _, _ = _render(tmp_path, doc=_full_crate())
    for verb in ("Supply", "Amend", "Keep"):
        assert verb in html
    assert "USER" in html and "HPC" in html and "EXPERIMENT" in html


def test_reprex_guide_renders_placeholders_not_values(tmp_path):
    """R4: value cells are placeholders / schema descriptions only.

    Post-[Q148] the three tables are Tabulator fragments, so these strings are carried
    in the options JSON inside the page's <script>, not as literal table markup. The
    page-level assertion is unchanged in INTENT -- the placeholder text must reach the
    reader -- but `{` and `}` survive `json.dumps` unescaped, so the same substrings
    still assert correctly against the whole document.
    """
    html, _, _ = _render(tmp_path, doc=_full_crate())
    assert "{amend for your target system}" in html
    assert "{your-default_account}" in html
    # The EXPERIMENT placeholder contains an em-dash, and post-[Q148] these cells ride
    # in the fragment's options JSON. `json.dumps` defaults to ensure_ascii=True, so the
    # em-dash is emitted as the escape `\u2014` -- which renders as an em-dash in the
    # browser but is NOT the literal character in the file. Asserting the literal form
    # here would fail on a page that displays correctly, so the two ASCII halves are
    # asserted instead and the escape is asserted explicitly.
    assert "{inherit " in html and " carried by the bundle}" in html
    assert "\\u2014" in html, "em-dash should be JSON-escaped, not dropped"


def test_reprex_guide_rows_default_to_required_first_then_alphabetical(tmp_path, monkeypatch):
    """{16}: the guide's rows reach the table already ordered.

    Asserts on the ROW LIST handed to `_df_for`, never on rendered order. The fix is
    DATA-side by design -- Tabulator's `initialSort` sorts the RENDERED `Required` cell
    as TEXT, so ascending puts `<strong>Required</strong>` LAST and descending works only
    by the accident that `R` > `O` > `C`. Asserting rendered order would therefore pass
    for the wrong reason and would couple this test to the table engine `[Q148]` re-opened.

    BOTH clauses are required and neither is sufficient: (a) alone passes on a
    required-first list that is internally unsorted, and (b) alone passes on a fully
    alphabetical list that ignores requiredness entirely.
    """
    captured: list[list[list[str]]] = []
    real = metadata._df_for

    def _spy(headers, rows, tips):
        captured.append([list(r) for r in rows])
        return real(headers, rows, tips)

    monkeypatch.setattr(metadata, "_df_for", _spy)
    _render(tmp_path, doc=_full_crate())

    # Denominators asserted before any verdict: a zero/one-row population would make
    # the ordering claims vacuously true and indistinguishable from a passing sort.
    assert captured, "denominator: _df_for was never called -- this probe measured nothing"
    rows = captured[0]
    assert len(rows) > 1, f"denominator: {len(rows)} row(s); ordering is unfalsifiable"

    REQUIRED = "<strong>Required</strong>"
    tiers = [0 if r[2].startswith(REQUIRED) else 1 for r in rows]
    assert tiers == sorted(tiers), f"clause (a): required rows must come first; tiers={tiers}"

    for tier in (0, 1):
        names = [
            metadata._strip_code(r[0]).lower()
            for r, t in zip(rows, tiers, strict=True)
            if t == tier
        ]
        assert names == sorted(names), f"clause (b): tier {tier} not alphabetical: {names}"


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


def test_toolkit_owned_fields_are_reported_as_supplied_and_required():
    """`toolkit_owned_output` exempts an existence check; it is NOT a supply signal.

    The two software-directory fields sit in the Supply bucket, and the renderer used to
    tell the reader they were "Not supplied by you" — a contradiction inside one table,
    and false: the user names the directory the clone/build gate builds into, and
    `system.py` raises ConfigurationError when it is None. Asserted on BOTH halves so a
    future edit can restore neither the non-supply claim nor a plain "Optional", which
    would quietly invite omission and a run-time failure.
    """
    from hhemt.config.system import system_config

    for name in ("TRITONSWMM_software_directory", "SWMM_software_directory"):
        cell = metadata._requiredness_cell(system_config.model_fields[name])
        assert "Required" in cell, cell
        assert "Not supplied by you" not in cell, cell
        assert not cell.startswith("Optional"), cell


def test_partition_selectors_are_amend_not_supply():
    """reprex_config's two target_* selectors are HPC-revisable, not host-local."""
    rows_by_bucket, _ = metadata._config_field_rows()
    hpc_labels = _labels(rows_by_bucket["hpc"])
    user_labels = _labels(rows_by_bucket["user"])
    for field in metadata._REPREX_SELECTOR_FIELDS:
        assert f"reprex_config.{field}" in hpc_labels
        assert f"reprex_config.{field}" not in user_labels


def test_system_config_values_resolve_through_the_system_object(tmp_path):
    """The system config hangs off `analysis._system`, never off the analysis.

    PRE-FIX THIS FAILS with `assert None == '1.5'`. `_config_field_values` read
    `getattr(analysis, "cfg_system", None)`, an attribute `TRITONSWMM_analysis` does
    not define, so its graceful-absent `continue` dropped EVERY `system_config.*` key
    and the EXPERIMENT table rendered 33 consecutive blank `Value used` cells.

    Asserted against a two-attribute stand-in rather than a real `system_config`
    BECAUSE the graceful-absent contract is what hid the bug: the function accepts any
    object exposing `model_fields`, so a test that builds a real config would pass for
    reasons unrelated to the traversal being tested.
    """
    from pydantic import BaseModel

    class _StubSystemCfg(BaseModel):
        target_dem_resolution: float = 1.5

    analysis = types.SimpleNamespace(
        analysis_paths=types.SimpleNamespace(analysis_dir=tmp_path),
        cfg_analysis=None,
        _system=types.SimpleNamespace(cfg_system=_StubSystemCfg()),
    )
    values = metadata._config_field_values(analysis)
    assert values.get("system_config.target_dem_resolution") == "1.5"


def test_a_swept_parameter_renders_as_varied_not_as_one_arms_value():
    """A sensitivity-varied field must not render one arm's setting as the value used.

    PRE-FIX THIS FAILS with `AttributeError: module ... has no attribute
    '_sensitivity_varied_values'`. The derivation is keyed off the sensitivity frame's
    COLUMN NAMES, which are the toolkit's own declaration of what varies, so adding a
    sweep axis changes the rendered set with no renderer edit -- the reason this is
    asserted over a crafted frame rather than a literal field list.
    """
    import pandas as pd

    frame = pd.DataFrame({"analysis.run_mode": ["serial", "mpi"]})
    analysis = types.SimpleNamespace(sensitivity=types.SimpleNamespace(_df_setup_full=frame))
    varied = metadata._sensitivity_varied_values(analysis)
    # [Q148]/item 17: a swept parameter returns (marker, value-list) so the marker can
    # be the cell and the values can be the hover. Both halves are asserted -- moving
    # the values off the cell must not lose them.
    marker, tooltip = varied["analysis_config.run_mode"]
    assert "Varied by the sensitivity analysis" in marker
    assert "serial" not in marker and "mpi" not in marker, "values must leave the cell"
    assert "serial" in tooltip and "mpi" in tooltip, "values must survive in the hover"


def test_a_constant_sweep_column_returns_the_same_shape_as_a_varied_one():
    """A sweep column with ONE distinct value must not crash the reproduction guide.

    PRE-FIX THIS FAILS with `ValueError: too many values to unpack (expected 2)` from
    `_build_reprex_guide_html`'s `_cell, _tip = _v`. `_sensitivity_varied_values` wrote
    a bare string for a constant column and a 2-tuple for a varied one into the SAME
    map, and its declared `dict[str, str]` described the crashing branch while
    contradicting the working one.

    The constant column is the whole point of the fixture: every sweep column in the
    existing tests VARIES, so `len(distinct) == 1` was unreachable and 47 green tests
    in this file ran over a function that failed on every real cluster render.

    Both halves are asserted deliberately. The shape assertion pins the invariant that
    was never expressed anywhere; the end-to-end call is the exact production crash
    path. `analysis_config.run_mode` resolves to the `hpc` bucket, which is one of the
    two buckets that receive a `Value used` column, so the crashing row IS built and
    the consumer assertion cannot pass vacuously.
    """
    import pandas as pd

    def _varied_for(values: list[str]) -> dict[str, tuple[str, str]]:
        frame = pd.DataFrame({"analysis.run_mode": values})
        analysis = types.SimpleNamespace(sensitivity=types.SimpleNamespace(_df_setup_full=frame))
        return metadata._sensitivity_varied_values(analysis)

    constant = _varied_for(["serial", "serial"])
    varied = _varied_for(["serial", "mpi"])

    # 1. The return shape is UNIFORM across both branches -- this is the invariant.
    for label, out in (("constant", constant), ("varied", varied)):
        cell_tip = out["analysis_config.run_mode"]
        assert isinstance(cell_tip, tuple) and len(cell_tip) == 2, (
            f"{label} column returned {type(cell_tip).__name__}, not a (cell, tooltip) 2-tuple"
        )

    # A constant column is NOT a varied parameter: it shows its value, with no hover.
    cell, tip = constant["analysis_config.run_mode"]
    assert "serial" in cell, "a constant column must disclose its one value"
    assert "Varied by the sensitivity analysis" not in cell
    assert tip == "", "a constant column has no value list to hover"

    # 2. The consumer survives the constant column end-to-end (the production path).
    guide = metadata._build_reprex_guide_html({}, constant)
    assert guide.html, "the reproduction guide must render with a constant sweep column"


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


# --- Declarative field metadata: one declaration, two consumers --------------


def test_conditional_requirement_is_enforced_and_rendered_from_one_declaration():
    """The rendered cell and the validator read the SAME declaration.

    This is the regression test for the defect that motivated the mechanism:
    `hpc_total_job_duration_min`'s description named `1_job_many_srun_tasks`
    while its validator required it under `batch_job`.
    """
    from hhemt.config.analysis import analysis_config
    from hhemt.config.base import declared

    field_info = analysis_config.model_fields["hpc_total_job_duration_min"]
    clauses = declared(field_info, "required_when")
    assert clauses == [{"field": "multi_sim_run_method", "in": ["batch_job"]}]

    cell = metadata._requiredness_cell(field_info)
    assert "Conditional" in cell
    assert "multi_sim_run_method is batch_job" in cell

    # The declaration is the ONLY enforcement site -- no imperative twin.
    src = inspect.getsource(analysis_config)
    assert "hpc_total_job_duration_min is None" not in src


def test_undeclared_field_renders_byte_identically_to_the_prose_only_form():
    """Graceful degradation is structural, not best-effort."""
    from hhemt.config.analysis import analysis_config

    field_info = analysis_config.model_fields["analysis_id"]
    assert metadata._description_cell(field_info) == metadata._esc(field_info.description or "—")
    assert "<dl" not in metadata._description_cell(field_info)


def test_option_glossary_renders_every_literal_member():
    """Options come from the declaration, and the declaration matches the type."""
    from typing import get_args

    from hhemt.config.analysis import analysis_config
    from hhemt.config.base import declared

    for name in ("run_mode", "multi_sim_run_method"):
        field_info = analysis_config.model_fields[name]
        options = declared(field_info, "options")
        assert set(options) == set(get_args(field_info.annotation))
        # Item 18: the glossary moved from the cell to the column tooltip. The cell
        # keeps the affordance count; the tooltip carries every member AND its
        # definition. Both are asserted from the ONE `options` declaration.
        cell = metadata._description_cell(field_info)
        assert f"({len(options)} options)" in cell
        tooltip = metadata._options_tooltip(field_info)
        for member in get_args(field_info.annotation):
            assert member in tooltip, f"{name}: {member} lost from the hover"
            assert options[member] in tooltip, f"{name}: {member}'s definition lost"


def test_option_glossary_drift_is_refused_at_class_definition_time():
    """Adding a Literal member without updating the glossary must not be possible."""
    from typing import Literal

    import pytest
    from pydantic import Field

    from hhemt.config.base import cfgBaseModel, field_meta

    with pytest.raises(TypeError, match="options keys"):

        class _Drifted(cfgBaseModel):
            m: Literal["a", "b", "c"] = Field(
                "a", json_schema_extra=field_meta(options={"a": "A", "b": "B"})
            )


def test_required_when_falls_back_to_the_triggers_declared_default():
    """A YAML that omits the trigger must still be judged against its default."""
    from typing import Literal, Optional

    import pytest
    from pydantic import Field

    from hhemt.config.base import cfgBaseModel, field_meta, when

    class _M(cfgBaseModel):
        mode: Literal["local", "batch_job"] = "local"
        dur: Optional[int] = Field(
            None, json_schema_extra=field_meta(required_when=[when("mode", "batch_job")])
        )

    assert _M().dur is None  # trigger absent -> default "local" -> not required
    assert _M(mode="batch_job", dur=90).dur == 90
    with pytest.raises(ValueError, match="dur is required when mode is batch_job"):
        _M(mode="batch_job")


# --- R5: SLURM efficiency ----------------------------------------------------


def test_slurm_section_renders_table_and_declares_the_csv_file(tmp_path):
    """R5: the globbed CSV is rendered AND declared as a source — the FILE, not the dir."""
    html, manifest, _ = _render(tmp_path, doc=_full_crate(), slurm_csv=_EFF_HEADER + _EFF_ROW)
    # The table's GRAIN moved from step to job ([Q145]/[Q153]), so the identity it renders
    # is the allocation `18573918`, not the step `18573918.0`. Asserted as a whole cell
    # rather than a substring: a bare `18573918` also matches the old step id, so it could
    # not tell the two grains apart and would pass under either.
    assert "<td>18573918</td>" in html
    # Memory-used % is now a reduction over steps rather than the plugin's own column, and
    # renders at one decimal (694.12 MB of 1000.0 MB requested).
    assert "69.4" in html
    declared = manifest["source_paths_relative"]
    assert "ro-crate-metadata.json" in declared
    assert any(p.endswith(".csv") for p in declared), declared
    # Declaring the DIRECTORY would raise in _validate_source_path once it exists.
    assert "logs/slurm_efficiency_report" not in declared


def test_slurm_report_nested_inside_a_csv_named_directory(tmp_path):
    """The REAL production shape: the `.csv` path is a DIRECTORY, report nested inside.

    snakemake-executor-plugin-slurm treats `--slurm-efficiency-report-path` as a
    DIRECTORY and writes `efficiency_report_{run_uuid}.csv` into it, while the toolkit
    passes a file-shaped `slurm_efficiency_report_{ts}.csv` path. So on every SLURM run
    the plugin mkdir's a directory with that `.csv` name and the real report lands one
    level deeper. The pre-2026-07-20 flat glob matched the DIRECTORY and `read_text()`
    raised `IsADirectoryError`, killing the whole `plot_metadata` rule; on the paths
    that did not crash it silently matched no real report at all, which is why this
    panel had never rendered data on any SLURM run.

    Nothing else in this file constructs this shape — every other SLURM test writes a
    plain file — so without this test the fix is unexercised against the case that
    motivated it.
    """
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    (analysis_dir / "ro-crate-metadata.json").write_text(json.dumps(_full_crate()))

    # The plugin's actual layout: a DIRECTORY whose name ends in .csv, report inside.
    eff_dir = analysis_dir / "logs" / "slurm_efficiency_report"
    csv_named_dir = eff_dir / "slurm_efficiency_report_20260101T000000.csv"
    csv_named_dir.mkdir(parents=True, exist_ok=True)
    (csv_named_dir / "efficiency_report_abc123.csv").write_text(_EFF_HEADER + _EFF_ROW)

    output_path = analysis_dir / "plots" / "metadata.html"
    # Must not raise IsADirectoryError.
    metadata.render(_fake_analysis(analysis_dir), report_config(), output_path)

    html = output_path.read_text()
    # Job grain per [Q145]/[Q153]; whole-cell form so it cannot pass on the old step id.
    assert "<td>18573918</td>" in html, "nested report was not found by the recursive glob"
    assert "69.4" in html

    manifest = json.loads((analysis_dir / "plots" / "metadata.manifest.json").read_text())
    declared = manifest["source_paths_relative"]
    # The NESTED file must be declared -- never the .csv-named directory, which
    # _validate_source_path rejects as a directory-as-source (Gotcha 66d).
    assert any(p.endswith("efficiency_report_abc123.csv") for p in declared), declared


def test_slurm_csv_with_header_only_degrades(tmp_path):
    """R5: a header-only CSV yields the heading + an info banner, not an empty table."""
    html, _, _ = _render(tmp_path, doc=_full_crate(), slurm_csv=_EFF_HEADER)
    assert 'id="slurm-efficiency"' in html
    assert "no job rows" in html


# --- R7: graceful degradation across every absent-source state ---------------


def test_absent_sidecar_degrades_gracefully(tmp_path):
    """R7: no sidecar -> heading present, .banner.info body, source still declared."""
    html, manifest, _ = _render(tmp_path, doc=None)
    assert 'id="provenance"' in html
    assert "banner info" in html
    assert "ro-crate-metadata.json" in html
    # ADR-6 D3: the expected source is declared even when ABSENT. Membership, not the
    # whole list -- the list also carries validation_report.json, which is likewise
    # declared-when-absent and is out of R7's scope here.
    assert metadata._SIDECAR_FILENAME in manifest["source_paths_relative"]


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
    html, manifest, _ = _render(tmp_path, doc=_full_crate(), slurm_csv=_EFF_HEADER + _EFF_ROW)
    # Job grain per [Q145]/[Q153]; whole-cell form so it cannot pass on the old step id.
    assert "<td>18573918</td>" in html and "69.4" in html
    declared = manifest["source_paths_relative"]
    assert any(p.endswith("efficiency_report_deadbeef.csv") for p in declared), declared


def test_resolve_all_efficiency_csvs_flat_nested_and_absent(tmp_path):
    """Unit: resolver returns [] when empty, the inner file for the nested plugin
    layout, and a flat file for the hypothetical future layout.

    Returns EVERY report, not the newest: one report is written per Snakemake
    invocation (the plugin builds it from `sacct --name={run_uuid}`), so the union
    is the whole experiment and the newest file alone is one submission's jobs.
    """
    from hhemt.report_renderers.metadata import _resolve_all_efficiency_csvs

    eff = tmp_path / "logs" / "slurm_efficiency_report"
    eff.mkdir(parents=True)
    assert _resolve_all_efficiency_csvs(eff) == []

    nested = eff / "slurm_efficiency_report_20260101T000000.csv"
    nested.mkdir()
    (nested / "efficiency_report_uuid.csv").write_text("a,b\n1,2\n")
    got = _resolve_all_efficiency_csvs(eff)
    assert [p.name for p in got] == ["efficiency_report_uuid.csv"]
    assert got[0].is_file()

    # A second submission's report must be ADDED, not replace the first.
    older = eff / "slurm_efficiency_report_20260102T000000.csv"
    older.mkdir()
    (older / "efficiency_report_uuid2.csv").write_text("a,b\n3,4\n")
    got = _resolve_all_efficiency_csvs(eff)
    assert {p.name for p in got} == {"efficiency_report_uuid.csv", "efficiency_report_uuid2.csv"}

    # Flat layout (hypothetical future plugin / driver cleanup) is still picked up.
    (eff / "efficiency_report_flat.csv").write_text("a,b\n5,6\n")
    assert "efficiency_report_flat.csv" in {p.name for p in _resolve_all_efficiency_csvs(eff)}


def test_efficiency_reports_union_is_keyed_on_jobid_and_is_idempotent():
    """Re-running part of an experiment ADDS rows; it never rewrites untouched ones.

    That is the user-visible requirement ("only updated rows of things that were
    ACTUALLY refreshed on a re run"), and it holds by construction rather than by
    comparison: SLURM does not reissue a job id within a cluster's id epoch, so a
    later report's rows are disjoint from an earlier one's.
    """
    first = _EFF_HEADER + "0,111.0,python,00:00:23,1,1,700K,,23.0,0.0,690.0,1000.0,111,0.0,69.0\n"
    second = _EFF_HEADER + "0,222.0,python,00:00:26,1,1,780K,,26.0,0.0,760.0,2000.0,222,0.0,38.0\n"

    _h, merged = metadata._parse_efficiency_csvs([("older", first), ("newer", second)])
    assert sorted(merged) == ["111.0", "222.0"]

    # Idempotent: re-parsing the same inputs yields identical rows.
    _h2, again = metadata._parse_efficiency_csvs([("older", first), ("newer", second)])
    assert again == merged

    # Duplicate JobID across two reports: the LATER file wins (oldest-first ordering).
    dup_old = _EFF_HEADER + "0,111.0,python,00:00:23,1,1,700K,,23.0,0.0,690.0,1000.0,111,0.0,69.0\n"
    dup_new = _EFF_HEADER + "0,111.0,python,00:00:99,1,1,700K,,99.0,0.0,690.0,1000.0,111,0.0,69.0\n"
    _h3, resolved = metadata._parse_efficiency_csvs([("older", dup_old), ("newer", dup_new)])
    assert resolved["111.0"]["Elapsed"] == "00:00:99"


def test_unrecognised_csv_column_is_disclosed_not_silently_dropped():
    """A column the curated table does not display must be NAMED on the page.

    The curated column set is what keeps the table readable, so dropping an
    unexpected column is deliberate. Dropping it SILENTLY is not: a future plugin
    version that starts reporting a new measurement would otherwise lose it with
    nothing on the page to say so.
    """
    csv_text = _EFF_HEADER.rstrip("\n") + ",GPUUtilPct\n"
    csv_text += "0,111.0,python,00:00:23,1,1,700K,,23.0,0.0,690.0,1000.0,111,0.0,69.0,87.5\n"
    html = metadata._build_slurm_efficiency_html([("r", csv_text)], {}, {}, lambda p: "")
    assert "Unrecognised column" in html
    assert "GPUUtilPct" in html

    # A report carrying only known columns emits no such note.
    plain = _EFF_HEADER + "0,111.0,python,00:00:23,1,1,700K,,23.0,0.0,690.0,1000.0,111,0.0,69.0\n"
    assert "Unrecognised column" not in metadata._build_slurm_efficiency_html([("r", plain)], {}, {}, lambda p: "")


def test_zero_cpu_efficiency_renders_as_not_measured_not_as_a_measured_zero():
    """A `0.0` in a percentage column asserts a measurement; an absent one must not.

    Measured on the delivered report: `CPU eff (%)` read exactly `0.0` on 664 of 664
    rows, across jobs whose Elapsed ranged from 18s to 9m. A population that uniform
    over inputs that heterogeneous is a computation that is not running, not a fleet
    of genuinely idle CPUs. Upstream, `parse_sacct_data` maps `RequestedMem_MB` and
    `RuleName` from the main job row down onto its steps and then discards the main
    rows -- but it builds no such map for `TotalCPU`, so the efficiency ratio is
    computed from a zero numerator wherever the cluster accounts CPU time on the
    allocation row rather than the step.

    Two arms in ONE render call, because the risk in suppressing a zero is
    over-reach: a row that genuinely measured low CPU use must keep its number.
    """
    zero_row = "0,111.0,python,00:00:23,1,1,700K,,23.0,0.0,690.0,1000.0,111,0.0,69.0\n"
    # The measured arm's CPU is recorded on the STEP, not on `.batch`. That is the shape the
    # cluster actually produces: a job either runs its payload in `.batch` and has NO numeric
    # steps at all (measured on 18552587 / 18554168 / 18554677, each returning only a job row
    # and `.batch`), or runs it in `srun` steps whose `TRESUsageInTot` carries the `cpu=` key.
    # The combination this row used to assert -- a numeric step present, the CPU on `.batch`,
    # and no step-level usage -- is the STALE-CAPTURE signature, and the reducer now declines
    # to report it rather than summing the wrapper alone. Keeping the old shape here would
    # have pinned the over-reach case to a state no job produces.
    measured_row = "0,222.0,python,00:01:40,1,1,512K,,100.0,91.0,500.0,8000.0,222,91.0,6.4\n"
    # The SUBJECT moved: `CPU eff (%)` is now COMPUTED here rather than read from the
    # plugin's own column, and its numerator `TotalCPU` lives only on the job and `.batch`
    # rows the plugin discards -- which is the very mechanism this test's docstring
    # describes. Stage A recovers them, so a measured arm requires that recovery.
    recovery = {
        "111": {
            "job": {"JobID": "111", "Elapsed": "00:00:23", "NNodes": "1", "NCPUS": "1"},
            # No recorded CPU usage at all -> not measured -> em-dash.
            "batch": {"JobID": "111.batch", "TotalCPU": "00:00:00", "MaxRSS": "700K",
                      "TRESUsageInTot": ""},
        },
        "222": {
            "job": {"JobID": "222", "Elapsed": "00:01:40", "NNodes": "1", "NCPUS": "1"},
            # A genuine measurement -> the number must survive. The numerator moved off
            # `TotalCPU`, which reads zero for srun-step work, onto the recorded usage.
            "batch": {"JobID": "222.batch", "TotalCPU": "00:00:00", "MaxRSS": "512K",
                      "TRESUsageInTot": "cpu=00:00:00,energy=0,mem=512K"},
            "0": {"JobID": "222.0", "TRESUsageInTot": "cpu=00:01:31,energy=0,mem=512K"},
        },
    }
    html = metadata._build_slurm_efficiency_html(
        [("r", _EFF_HEADER + zero_row + measured_row)], {}, {}, lambda p: "", recovery
    )

    def _eff_cell(job_id: str) -> str:
        """That job's `CPU eff (%)` cell, located BY HEADER rather than by position.

        A positional index would silently follow the wrong column the next time the column
        order changes, and pass while measuring something else.
        """
        headers = re.findall(r'<span class="th-label">(.*?)</span>', html)
        idx = next(i for i, h in enumerate(headers) if h.startswith("CPU eff"))
        row = next(r for r in re.findall(r"<tr>(.*?)</tr>", html, re.S) if f"<td>{job_id}</td>" in r)
        return re.findall(r"<td>(.*?)</td>", row, re.S)[idx]

    # Arm 1 -- no step reported CPU time, so the ratio has no numerator. The cell must read
    # as not-measured; a `0.00` here would assert the job used no processor.
    assert _eff_cell("111") == "—", _eff_cell("111")

    # Arm 2 -- a real measurement survives untouched: 91 CPU-seconds over 1 CPU x 100s.
    # Guards against suppressing any cell that merely looks small.
    assert "91.00" in _eff_cell("222"), _eff_cell("222")

    # The numerator stays legible even though the CPU-seconds COLUMN was dropped, so a
    # reader can still check the ratio against `seff` without it.
    assert "91.000 CPU-seconds" in _eff_cell("222")

    # The suppression must be legible as not-measured, and the note must say so.
    assert "—" in html
    assert "not-measured" in html or "no CPU time" in html


def test_resume_attempts_are_numbered_in_the_description_column():
    """A resumed sim occupies several rows; each must say WHICH attempt it is.

    Measured on the delivered generation: a resumed simulation is ONE allocation carrying
    several srun steps (28 sims at n_resumes 3 -> 18 allocations x 4 steps + 10 x 5 = the
    122 sim rows). Those rows were already joined to their simulation and already carried
    identical purpose text, so filtering to one simulation returned an unordered set of
    indistinguishable rows -- the exact complaint this closes.

    Two arms in ONE render call: a step WITH a recorded attempt must be numbered, and a
    step WITHOUT one must keep its bare purpose rather than be guessed at.
    """
    labelled = "0,777.0,python,00:00:23,1,1,700K,,23.0,0.0,690.0,1000.0,777,0.0,69.0\n"
    resumed = "0,777.2,python,00:00:26,1,1,700K,,26.0,0.0,690.0,1000.0,777,0.0,69.0\n"
    unlabelled = "0,777.9,python,00:00:26,1,1,700K,,26.0,0.0,690.0,1000.0,777,0.0,69.0\n"
    purpose_map = metadata._job_purpose_map(
        [
            {
                "rule_name": "simulation_sa_mpi_11_r1_evt_event_index_0",
                "slurm_job_id": "777",
                "sa_id": "mpi_11_r1",
                "event_id": "event_index.0",
                "model_type": "tritonswmm",
            }
        ]
    )
    scenario_map = {
        ("mpi_11_r1", "event_index.0", "tritonswmm"): {
            "hpc.partition": "standard",
            "attempt_by_jobstep": json.dumps({"777.0": 0, "777.2": 2}),
        }
    }
    html_out = metadata._build_slurm_efficiency_html(
        [("r", _EFF_HEADER + labelled + resumed + unlabelled)],
        purpose_map,
        scenario_map,
        lambda p: "",
    )

    # The SUBJECT of both arms moved when the table went to job grain ([Q145]/[Q153]): the
    # per-attempt breakdown is no longer one row per step carrying its own label, it is the
    # `<details>` roster inside the single job row. The invariants are unchanged and are
    # re-attached to that roster; neither was rewritten to match the code.

    # Arm 1 -- attempts ARE named, from the ledger. The job's own purpose still carries the
    # attempt its `.0` step recorded, and the roster names the resume.
    assert "simulate (initial run)" in html_out
    assert "resume 2" in html_out

    # Arm 2 -- a step with no recorded attempt is not guessed at. This matters MORE after
    # aggregation than before it: folding N steps into one row puts an ordered list in front
    # of the renderer, so numbering from the step suffix becomes newly available and newly
    # tempting. `777.9` is absent from `attempt_by_jobstep`, so it must be listed WITHOUT a
    # number rather than labelled "resume 9" from its suffix.
    assert "resume 9" not in html_out
    assert "attempt not recorded" in html_out


def test_slurm_job_index_harvest_reads_the_executor_log_tree(tmp_path):
    """The jobid -> rule index comes from the plugin's own per-job log tree.

    511 of the delivered report's 570 allocations have no `_status/*.flag.json` record,
    because that sidecar is one slot per flag path and nine submissions overwrote it. The
    executor's log tree is the only retroactive source, and reading it must open nothing:
    the rule is the DIRECTORY and the job id is the FILENAME.
    """
    from hhemt.status_flags import harvest_slurm_job_index

    logs = tmp_path / ".snakemake" / "slurm_logs"
    (logs / "rule_simulation_sa_mpi_11_r1" / "mpi_11_r1_0").mkdir(parents=True)
    (logs / "rule_simulation_sa_mpi_11_r1" / "mpi_11_r1_0" / "18396671.log").write_text("")
    (logs / "rule_setup_target_0").mkdir(parents=True)
    (logs / "rule_setup_target_0" / "18396501.log").write_text("")
    # A non-jobid filename must not enter the index.
    (logs / "rule_setup_target_0" / "notes.log").write_text("")

    index = harvest_slurm_job_index(tmp_path)
    assert index == {
        "18396671": "simulation_sa_mpi_11_r1",
        "18396501": "setup_target_0",
    }

    # Absent tree -- a local/native run or an off-cluster bundle -- degrades to empty,
    # never to a partial index that would mislabel rows.
    assert harvest_slurm_job_index(tmp_path / "nope") == {}


def test_job_purpose_and_hardware_are_joined_in_from_toolkit_records():
    """The human-readable purpose is NOT read off the job — it is joined in.

    SLURM reports every job step's command name as `python`, and on a cluster that
    does not store job comments the plugin drops its RuleName column entirely. The
    toolkit's own `_status/*.flag.json` sidecars carry `rule_name` keyed by the
    parent `SLURM_JOB_ID`, which is exactly the CSV's `MainJobID`.
    """
    csv_text = _EFF_HEADER + "0,999.0,python,00:01:40,1,1,512K,,100.0,91.0,500.0,8000.0,999,91.0,6.4\n"
    purpose_map = metadata._job_purpose_map(
        [
            {
                "rule_name": "simulation_sa_gpu_0_r1_evt_event_index_0",
                "slurm_job_id": "999",
                "sa_id": "gpu_0_r1",
                "event_id": "event_index.0",
                "model_type": "tritonswmm",
            }
        ]
    )
    scenario_map = {
        ("gpu_0_r1", "event_index.0", "tritonswmm"): {
            "hpc.partition": "gpu-a6000",
            "n_gpus": "1",
            "n_mpi_procs": "1",
            "n_omp_threads": "1",
            "n_nodes": "1",
            "run_mode": "gpu",
            "backend_used": "gpu",
        }
    }
    html = metadata._build_slurm_efficiency_html(
        [("r", csv_text)], purpose_map, scenario_map, lambda p: "a6000" if p else ""
    )
    for expected in ("simulate", "gpu_0_r1", "tritonswmm", "gpu-a6000", "a6000"):
        assert expected in html, expected
    # JobName is `python` for every row, which is precisely why it is not a column; the
    # joined purpose is. The HEADER was renamed to `Job desc` by user ruling {25} -- the
    # invariant here is that the joined column exists and carries the joined value, not what
    # it happens to be called, so the assertion follows the rename rather than pinning the
    # old string.
    assert "Job desc" in html
    assert "What the job did" not in html, "the renamed header must not survive anywhere"


def test_sub_datasets_render_as_a_folder_tree_not_a_flat_run():
    """Iter-10 H: 'the presentation of sub datasets is unreadable; it's just a massive list
    of filepaths ... i think a branch structure like people use to display folder structure
    could be good for this.'

    Shape measured on the delivered bundle: 29 paths in ONE cell, space-separated, over two
    top-level prefixes. This asserts the tree markup AND that no path is lost -- a renderer
    that dropped entries would otherwise look tidier and score better.

    TWO sub-analyses, not three, and that is deliberate. Iter-11 item 14 collapses a
    contiguous run of `_TREE_SENTINEL_MIN_SIBLINGS` (3) or more structurally identical
    siblings into one `{stem…} × N` sentinel, which is a DIFFERENT rendering from the
    single-child-chain collapse this test guards. A three-sibling fixture straddles that
    threshold and would assert the Iter-11 shape while claiming to test the Iter-10 one.
    Two keeps this test in the uncollapsed regime where its subject actually exists; the
    sentinel regime is guarded independently by tests/test_path_tree_collapse.py. Do NOT
    "restore" a third sibling here -- it silently converts this into a different test.
    """
    paths = [
        "sensitivity_datatree.zarr/",
        "subanalyses/sa_gpu_0_r1/analysis_datatree.zarr",
        "subanalyses/sa_gpu_0_r2/analysis_datatree.zarr",
    ]
    html = metadata._path_tree_html(paths)

    assert html.startswith("<pre"), "the tree must be preformatted or the glyphs collapse"
    assert "├── " in html or "└── " in html, "no branch glyphs -- this is still a flat list"

    body = html.split(">", 1)[1]
    lines = [ln for ln in body.replace("</pre>", "").split("\n") if ln.strip()]

    # Every leaf survives the transformation.
    for leaf in ("sa_gpu_0_r1", "sa_gpu_0_r2", "sensitivity_datatree.zarr"):
        assert any(leaf in ln for ln in lines), leaf

    # Single-child chains COLLAPSE: each sub-analysis is one line carrying both its own
    # directory and its single child, never a directory line plus an indented leaf. Without
    # this the real 28-sub-analysis population renders 56 lines, half of them the same
    # filename, which is the unreadability the item is about reached from the other side.
    collapsed = [ln for ln in lines if "sa_gpu_0_r1/analysis_datatree.zarr" in ln]
    assert len(collapsed) == 1, f"single-child chain not collapsed: {lines}"

    # `subanalyses` is a real branch (2 children), so it must NOT be collapsed into its
    # children -- proving the collapse stops at a branch rather than flattening everything.
    assert any(ln.strip().endswith("subanalyses") for ln in lines), lines


def test_no_hand_written_conditional_requirement_survives():
    """`validate_from_toggle` is the second enforcement site `field_meta` replaced.

    The Reproduction Guide's Required cell is derived from `is_required()` +
    `required_when`; a requirement enforced anywhere else renders as `Optional`
    and lies to a reproducer. Ban the legacy helper outright rather than
    enumerate its call sites, which is how 14 of them accumulated.
    """
    import inspect

    from hhemt.config import analysis as _a
    from hhemt.config import system as _s

    for mod in (_a, _s):
        src = inspect.getsource(mod)
        assert "cls.validate_from_toggle(" not in src, (
            f"{mod.__name__} still enforces a conditional requirement outside "
            "`required_when`; the rendered Required cell cannot see it."
        )


def test_every_closed_set_field_declares_its_option_glossary():
    """A `Literal[str]` field with no `options=` restates its own vocabulary in prose.

    `__pydantic_init_subclass__` already forbids a glossary that DRIFTS from its
    Literal. This is the other half: it forbids a closed-set field with no
    glossary at all, which is where a prose re-enumeration goes.
    """
    from typing import get_args

    from hhemt.config.analysis import analysis_config
    from hhemt.config.base import declared
    from hhemt.config.reprex_config import reprex_config
    from hhemt.config.system import system_config

    missing = []
    for label, model in (
        ("system_config", system_config),
        ("analysis_config", analysis_config),
        ("reprex_config", reprex_config),
    ):
        for name, fi in model.model_fields.items():
            members = set(get_args(fi.annotation))
            if members and all(isinstance(v, str) for v in members):
                if declared(fi, "options") is None:
                    missing.append(f"{label}.{name}")
    assert not missing, (
        "closed-set fields rendered in the Reproduction Guide with no option "
        f"glossary: {sorted(missing)}"
    )


def test_varied_values_cell_carries_the_tooltip_affordance_as_one_rule():
    """T-2 / Iter-12 item 17 — the tooltip-bearing cell advertises itself as hoverable.

    Asserts the three declarations as a SET ON ONE SELECTOR rather than as three
    independent substring hits: a rule carrying only `cursor: help` would satisfy
    three separate `in` checks against the whole stylesheet while rendering no
    affordance, which is the satisfying-position failure the spec names.

    The colour is asserted EQUAL TO the resolved style's own value, never a literal.
    A hardcoded hex here would re-introduce the second colour source the brand_theme
    stipulation exists to prevent, and would red on any legitimate brand change.
    """
    from hhemt.config.report import report_config
    from hhemt.report_renderers.metadata import _resolve_inline_css

    cfg = report_config()
    css = _resolve_inline_css(cfg)
    expected_color = cfg.errors_and_warnings.primary_color

    # Isolate the ONE rule, so the declarations are checked inside a single block.
    match = re.search(r"strong\.tip-affordance\s*\{([^}]*)\}", css)
    assert match is not None, f"no strong.tip-affordance rule in the emitted CSS:\n{css}"
    block = match.group(1)

    assert "cursor: help" in block, f"affordance rule lacks the help cursor: {block!r}"
    assert re.search(r"border-bottom:\s*1px\s+dotted", block), f"affordance rule lacks the dotted underline: {block!r}"
    assert re.search(rf"(?<!-)color:\s*{re.escape(expected_color)}", block), (
        f"affordance colour is not the resolved brand colour {expected_color!r}: {block!r}"
    )


def test_varied_values_affordance_is_absent_from_the_single_value_branch():
    """The single-distinct-value cell returns an EMPTY tooltip, so it carries no affordance.

    Styling it would advertise a hover that shows nothing. This pins the asymmetry
    rather than the presence, so a future refactor that marks every cell uniformly
    fails here instead of shipping a dead hover.
    """
    from hhemt.report_renderers.metadata import _sensitivity_varied_values

    src = inspect.getsource(_sensitivity_varied_values)
    # The empty-tooltip branch and the affordance class must not co-occur on one line.
    for line in src.splitlines():
        if '""' in line and "tip-affordance" in line:
            raise AssertionError(f"single-value branch carries the affordance: {line!r}")


def test_attempt_roster_reports_the_step_state_recovered_from_sacct():
    """An attempt's CANCELLED/COMPLETED outcome must reach the per-attempt roster.

    `_build_efficiency_rows` states the stake in its own comment: "the solver steps ARE
    the resume attempts, and their CANCELLED/COMPLETED breakdown is what [Q143] calls this
    campaign's subject". Job grain folds N attempts into one row, so the roster is the only
    place that breakdown survives -- and a roster rendering "state not recorded" for every
    attempt is a disclosure that discloses nothing, which is the failure [Q153]'s
    trackability condition exists to prevent.

    The field cannot come from the plugin. Measured on this campaign's own efficiency CSVs
    (14 files, 928 rows, all numeric steps): the emitted columns are JobID/JobName/Elapsed/
    TotalCPU/NNodes/NCPUS/MaxRSS/ReqMem plus the parsed restatements, MainJobID and two
    precomputed percentages. There is no `State` column at any step kind. It exists only in
    `slurm_job_recovery._SACCT_FIELDS`, so it reaches the roster only by the join in
    `_aggregate_jobs`.

    TWO ARMS over the SAME code path, because an assertion that only checks the happy case
    cannot tell a working join from a roster that happens to print something. Arm B narrows
    `_RECOVERED_ONLY_STEP_FIELDS` to the pre-fix set and must reproduce the delivered
    artifact's exact string; if both arms rendered alike the test would prove nothing.
    """
    real_state = "CANCELLED by 554635"
    # A plugin step row in its measured shape: Elapsed and MaxRSS present, neither
    # TRESUsageInTot nor State.
    merged = {"999.1": {"JobID": "999.1", "MainJobID": "999", "Elapsed": "00:00:52", "MaxRSS": "30096K"}}
    recovery = {
        "999": {
            "job": {"JobID": "999", "Elapsed": "00:01:00", "NCPUS": "1", "State": "COMPLETED"},
            "batch": {"JobID": "999.batch", "TRESUsageInTot": "cpu=00:00:04", "State": "COMPLETED"},
            "1": {"JobID": "999.1", "TRESUsageInTot": "cpu=00:08:07", "State": real_state},
        }
    }

    def roster(fields):
        # DEEP copy per arm, and the reason is a real property of the function rather than
        # test hygiene: `_aggregate_jobs` writes the joined fields ONTO the caller's step
        # dicts. A shallow `dict(merged)` shares those inner dicts, so arm A's join leaks
        # into arm B and arm B renders the state it is supposed to lack -- which is exactly
        # how this test failed on first run. Benign in production, where `merged` is built
        # fresh per render, but fatal to any differential that reuses one input.
        saved = metadata._RECOVERED_ONLY_STEP_FIELDS
        metadata._RECOVERED_ONLY_STEP_FIELDS = fields
        try:
            rows = metadata._aggregate_jobs(copy.deepcopy(merged), copy.deepcopy(recovery))
            row = next(r for r in rows if r["JobID"] == "999")
            return metadata._attempt_details_html(row["_attempts"], {})
        finally:
            metadata._RECOVERED_ONLY_STEP_FIELDS = saved

    joined = roster(("TRESUsageInTot", "State"))
    assert real_state in joined
    assert "state not recorded" not in joined

    # Arm B: the pre-fix field set reproduces the defect exactly as the delivered report
    # rendered it, which is what makes arm A's pass meaningful rather than incidental.
    unjoined = roster(("TRESUsageInTot",))
    assert "state not recorded" in unjoined
    assert real_state not in unjoined

    # The ATTEMPT NUMBER is deliberately absent from both arms and must stay that way: it is
    # read only from the ledger, and deriving it from the step index would fabricate a value
    # for every step the ledger does not record.
    assert "attempt not recorded" in joined
