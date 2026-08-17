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
    html = metadata._build_reprex_guide_html(poisoned)
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
    cell = varied["analysis_config.run_mode"]
    assert "Varied by the sensitivity analysis" in cell
    assert "serial" in cell and "mpi" in cell


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
        cell = metadata._description_cell(field_info)
        for member in get_args(field_info.annotation):
            assert member in cell


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
    assert "18573918.0" in html
    assert "69.41" in html
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
    assert "18573918.0" in html, "nested report was not found by the recursive glob"
    assert "69.41" in html

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
    assert "18573918.0" in html and "69.41" in html
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
    # JobName is `python` for every row, which is precisely why it is not a column.
    assert "What the job did" in html


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
