"""The dataset license is optional until publish: the whole chain from config to deposit.

A dataset license is needed only to PUBLISH. Every earlier step -- loading a config,
saving and reloading it, building the provenance crate at consolidation, bundling --
must accept an analysis that has none, and publish must refuse one with the steps that
fix it. These tests walk that chain with no solver and no compile: the config model,
the shared crate builder both consolidation paths call, the bundle's crate check, and
publish against a sidecar the real emitter wrote.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Literal

import pytest
from pydantic import Field, ValidationError

from hhemt import provenance, publishing
from hhemt.bundle._emit import _crate_license_finding
from hhemt.config.analysis import analysis_config
from hhemt.config.base import cfgBaseModel, field_meta
from hhemt.exceptions import PublishError
from hhemt.metadata import write_rocrate_sidecar
from tests.test_config_validation import _minimal_analysis_config_dict


@pytest.mark.parametrize("license_value", [None, "CC-BY-SA-4.0"])
@pytest.mark.parametrize("dump_mode", ["python", "json"])
def test_config_with_or_without_a_license_survives_a_save_and_reload(tmp_path, license_value, dump_mode):
    """Property: a config round-trips through model_dump and model_validate with or without a license.

    Class: every config, unlicensed or naming one of the seven, dumped in either mode. Why
    required: the toolkit dumps and re-validates analysis configs at the sensitivity member
    rebuild, the runner reloads and the bundle writes, so a field that loads blank but
    refuses its own dumped blank breaks every unlicensed analysis at its first reload.
    Killed by: declaring the field as a flat Literal with a None default (the dump writes an
    explicit None, which the flat Literal refuses). Second correct implementation: a flat
    Literal plus a before-validator that drops an explicit None.
    """
    cfg = _minimal_analysis_config_dict(tmp_path)
    if license_value is not None:
        cfg["dataset_license"] = license_value
    loaded = analysis_config.model_validate(cfg)
    assert loaded.dataset_license == license_value
    reloaded = analysis_config.model_validate(loaded.model_dump(mode=dump_mode))
    assert reloaded.dataset_license == license_value


@pytest.mark.parametrize("value", ["MIT", "CC-BY-3.0", "cc-by-4.0", ""])
def test_a_license_outside_the_seven_is_refused_at_load(tmp_path, value):
    """Property: only the seven SPDX ids (exact case) load; anything else is refused at load.

    Class: other SPDX ids, older CC versions, a lower-cased id and an empty string. Why
    required: every later consumer looks the id up in one table, so an unknown id accepted at
    load would fail only at consolidation or deposit. Killed by: a case-insensitive or
    free-text field. Second correct implementation: a validator mapping the value through
    the license table instead of a Literal.
    """
    cfg = _minimal_analysis_config_dict(tmp_path)
    cfg["dataset_license"] = value
    with pytest.raises(ValidationError):
        analysis_config.model_validate(cfg)


def test_import_check_reads_an_optional_closed_set_and_still_refuses_drift_and_wider_unions():
    """Property: the class-definition glossary check looks through `X | None` and nothing wider.

    Class: an optional Literal whose glossary matches, one whose glossary drifts, and a
    two-member union. Why required: the optional license field must keep its glossary and its
    drift check, while the existing refusal of a Literal ORed with a list must survive.
    Killed by: refusing every union (the optional field cannot carry a glossary) or unwrapping
    every union (the wider union's glossary is checked against one member only). Second
    correct implementation: unwrapping via `typing.Optional` detection rather than NoneType
    membership.
    """

    class _Matches(cfgBaseModel):
        x: Literal["a", "b"] | None = Field(None, json_schema_extra=field_meta(options={"a": "A", "b": "B"}))

    assert _Matches.model_validate({}).x is None

    with pytest.raises(TypeError, match="options keys"):

        class _Drifts(cfgBaseModel):
            x: Literal["a", "b"] | None = Field(None, json_schema_extra=field_meta(options={"a": "A"}))

    with pytest.raises(TypeError, match="not a flat Literal"):

        class _Wider(cfgBaseModel):
            x: Literal["a", "b"] | list[Literal["a", "b"]] = Field(
                "a", json_schema_extra=field_meta(options={"a": "A", "b": "B"})
            )


class _Field:
    def __init__(self, v):
        self._v = v

    def get(self):
        """Return the held value, as a log field's `.get()` does for the crate emitter."""
        return self._v


def _fake_analysis(analysis_dir, dataset_license):
    log = SimpleNamespace(
        workflow_submission_time=_Field("2026-01-01T00:00:00"),
        workflow_submission_node=_Field("login1"),
    )
    return SimpleNamespace(
        cfg_analysis=SimpleNamespace(analysis_id="a1", dataset_license=dataset_license),
        _get_enabled_model_types=lambda: [],
        df_sims=SimpleNamespace(index=[]),
        log=log,
        _case_manifest=SimpleNamespace(case_name="norfolk", description="", manifest={"inputs/dem.tif": "ab" * 32}),
        analysis_paths=SimpleNamespace(analysis_dir=analysis_dir),
    )


# The two production callers of the shared crate builder, by the arguments each passes:
# the analysis consolidation (processing_analysis.py, emit_provenance(self._analysis, ...))
# and the sensitivity MASTER consolidation (sensitivity_analysis.py,
# emit_provenance(self.experiment, consolidated_zarr_relpath="experiment_datatree.zarr",
# sub_dataset_relpaths=..., with_run_units=False, ...)).
_CALLER_SHAPES = {
    "analysis": {},
    "sensitivity_master": {
        "consolidated_zarr_relpath": "experiment_datatree.zarr",
        "sub_dataset_relpaths": ["members/member_1/"],
        "with_run_units": False,
    },
}


@pytest.mark.parametrize("caller", sorted(_CALLER_SHAPES))
def test_both_consolidation_paths_build_an_unlicensed_crate(tmp_path, monkeypatch, caller):
    """Property: the shared crate builder succeeds with no license and writes none, for both callers.

    Class: an unlicensed analysis reaching the builder through either production call shape.
    Why required: consolidation must not need a license, and a blank passed through as the
    text "None" raised KeyError in the license table before this change, crashing both the
    analysis and the sensitivity-master consolidation. Killed by: re-introducing str() on the
    blank, or writing a license when blank. Second correct implementation: the caller
    omitting the keyword when blank instead of passing None.
    """
    monkeypatch.setattr(provenance, "_toolkit_git_sha", lambda: "deadbeef")
    core, sidecar = provenance.emit_provenance(_fake_analysis(tmp_path, None), **_CALLER_SHAPES[caller])
    for doc in (json.loads(core), json.loads(sidecar)):
        root = next(e for e in doc["@graph"] if e.get("@id") == "./")
        assert "license" not in root


def _staged_crate(staging, license_uri):
    root = {"@id": "./", "@type": "Dataset"}
    if license_uri is not None:
        root["license"] = {"@id": license_uri}
    doc = {"@graph": [{"@id": "ro-crate-metadata.json", "about": {"@id": "./"}}, root]}
    (staging / "ro-crate-metadata.json").write_text(json.dumps(doc))


@pytest.mark.parametrize(
    "cfg_license, crate_uri, expect_finding",
    [
        (None, None, False),
        ("CC-BY-SA-4.0", None, True),
        (None, "https://spdx.org/licenses/CC0-1.0", True),
        ("CC-BY-SA-4.0", "https://spdx.org/licenses/CC-BY-SA-4.0", False),
    ],
)
def test_bundle_crate_check_is_silent_only_when_config_and_crate_agree(
    tmp_path, cfg_license, crate_uri, expect_finding
):
    """Property: the bundle's crate check reports a finding exactly when config and crate disagree.

    Class: the four combinations of a blank or set config license with a blank or set crate
    license. Why required: an unpublished analysis with no license is normal and must not put
    a finding in every bundle manifest, while a crate that predates its config's license must
    still be reported. Killed by: reporting blank-and-blank, dropping the licensed-config
    blank-crate finding, or looking a blank config up in the table (KeyError). Second correct
    implementation: comparing SPDX ids recovered from the crate instead of URIs.
    """
    _staged_crate(tmp_path, crate_uri)
    analysis = SimpleNamespace(cfg_analysis=SimpleNamespace(dataset_license=cfg_license))
    finding = _crate_license_finding(analysis, tmp_path)
    assert (finding is not None) is expect_finding


def test_bundle_finding_for_a_cleared_license_names_setting_it_back_in_plain_words(tmp_path):
    """Property: the bundle's finding for a blank config against a licensed crate names the fix plainly.

    Class: a config with no license whose staged crate carries one. Why required: the user
    cleared the field after consolidating; re-consolidating would drop the license, so the
    finding must name the results' license as the value to set back, and must say "no
    license set" rather than render the blank as Python's None. Killed by: a finding that
    offers only re-consolidation, or that prints `None (None)`. Second correct
    implementation: leading with "set analysis_config.dataset_license to 'CC0-1.0' to keep it" and writing `(blank)`.
    """
    _staged_crate(tmp_path, "https://spdx.org/licenses/CC0-1.0")
    analysis = SimpleNamespace(cfg_analysis=SimpleNamespace(dataset_license=None))
    finding = _crate_license_finding(analysis, tmp_path)
    assert finding is not None
    assert "set analysis_config.dataset_license to 'CC0-1.0'" in finding
    assert "None (None)" not in finding


def test_bundle_finding_for_a_licensed_config_against_an_unlicensed_crate_says_stale_and_names_re_emit(tmp_path):
    """Property: a licensed config against an unlicensed crate is reported as a stale crate with its fix.

    Class: a config naming any license whose staged crate carries none. Why required: the user
    chose the license after consolidating, so the crate predates the config, and only a
    re-emit with `regenerate_existing=True` writes the license into it; a finding that reads
    as a malformed crate gives no path. Killed by: a finding that does not call the crate
    stale, or omits the `regenerate_existing=True` re-emit route, or does not name the
    config's license. Second correct implementation: stating the re-emit route before the
    staleness clause, with the config's id in a trailing parenthesis.
    """
    _staged_crate(tmp_path, None)
    analysis = SimpleNamespace(cfg_analysis=SimpleNamespace(dataset_license="CC-BY-SA-4.0"))
    finding = _crate_license_finding(analysis, tmp_path)
    assert finding is not None
    assert "predates the config" in finding
    assert "reprocess(start_with='consolidate', regenerate_existing=True)" in finding
    assert "CC-BY-SA-4.0" in finding


def test_publish_refuses_an_analysis_whose_emitted_crate_has_no_license(tmp_path, monkeypatch):
    """Property: publish refuses, before any deposit, results whose real emitted crate has no license.

    Class: an unlicensed analysis taken through the real crate emitter and sidecar writer and
    then published. Why required: publish is the one place a license is required, and the
    refusal must name the field and the re-emit step the user has to take. Killed by: the
    emitter writing a placeholder license (publish would then deposit under it), or a refusal
    that omits the fix. Second correct implementation: the same refusal raised by the target
    adapter before its first HTTP call.
    """
    monkeypatch.setattr(provenance, "_toolkit_git_sha", lambda: "deadbeef")
    analysis = _fake_analysis(tmp_path, None)
    _core, graph = provenance.emit_provenance(analysis)
    write_rocrate_sidecar(tmp_path, graph_json=graph)
    called = []
    monkeypatch.setitem(publishing._TARGETS, "zenodo", lambda: SimpleNamespace(publish=lambda **kw: called.append(kw)))
    with pytest.raises(PublishError) as ei:
        publishing.publish_analysis(analysis, target="zenodo")
    assert "dataset_license" in ei.value.status
    assert "regenerate_existing=True" in ei.value.status
    assert called == []
