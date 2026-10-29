"""A dataset license supplied from a details file, end to end, with no solver and no compile.

A dataset license is a CLOSED set of seven SPDX ids everywhere the toolkit reads it, so a
laboratory's own terms, or an institutional license, could not be deposited at all. This
module covers the second source: ``dataset_license_file`` names a YAML file carrying the
license's ``name`` and, optionally, its ``link``, and every consumer that used to look a
name up in ``metadata._SPDX_LICENSE_TABLE`` now takes a
``metadata.LicenseRecord`` instead.

The module is organized by consumer. One test per license-table lookup site, so a site
that keeps indexing the table on a details-file license fails here rather than at a
deposit, plus the config surface and the two publish targets.

PRE-CHANGE EVIDENCE, pasted from the pre-change tree with the equivalent custom input (a
free-form license name, since the field did not exist). Every arm below that names a
consumer cites the pre-change arm it replaces:

    site1 build_analysis_crate(custom name): RAISED KeyError: 'Our Lab Data License v1.0'
    site2 build_datacite_rightslist(custom name): RAISED KeyError: 'Our Lab Data License v1.0'
    site3 _hydroshare_rights(custom name): RAISED KeyError: 'Our Lab Data License v1.0'
    site5 _read_license_from_sidecar(crate naming its own licence): RAISED PublishError:
          publish to ? failed (doi=None): no recognizable dataset license in ro-crate sidecar
    site4 _crate_license_finding(details-file config, agreeing crate): RETURNED a FALSE
          staleness finding, because the config side was read from dataset_license alone
    site6 keep-the-crate's-licence remedy offered for a custom crate licence: False
    site4 _crate_license_finding(unnameable crate licence URI, config declaring none): RETURNED
          None, because the unrecognized-licence arm was collapsed into "no record" and both
          sides then normalized to None
    control _read_license_from_sidecar(table member): RETURNED 'CC0-1.0'
    control _crate_license_finding(table member, agreeing): RETURNED None

The two control lines are what make the raises evidence about the CUSTOM license rather
than about the probe: the same reader and the same compare, against a table member,
return cleanly.

INVOCATION ISOLATION. Every test here runs in-process against the checked-out source with
no subprocess, no environment variable and no path outside ``tmp_path``, so there is no
invocation-shaped arm to isolate: the discriminator applied per test is the second arm of
the standard's condition 4 — the named change to what the code DOES greens it, and the
pre-change output pasted above is the same input failing. A test that a mere change of
import path could green would be a red about the tree, not about the code; none of these
is, because each asserts on a returned value the pre-change code cannot produce.

REACHABILITY AND BOTH-STATES ANCHOR, per test, so the two per-test conditions of the
standard are checkable without re-deriving them. Each row is
``{test} -> reaches {the callable the failure lives in}; anchors on {the property asserted}.``
Every ANCHOR below is a property of BOTH states (it names a value, a pair or an absence that
the current code and the proposed code each define), never wording this change introduces:

- ``loader_reads_name_and_link`` -> reaches ``metadata.load_license_details``;
  anchors on the ``(name, link)`` pair the record carries.
- ``loader_refuses_a_malformed_details_file`` -> reaches the same loader;
  anchors on whether a ``ConfigurationError`` is raised for the file reached.
- ``config_with_a_details_file_round_trips_in_both_dump_modes`` -> reaches
  ``analysis_config.model_validate`` on a dumped payload; anchors on the field VALUE
  surviving a dump/load cycle, in each mode.
- ``config_accepts_either_license_source_alone_and_refuses_both`` -> reaches the model
  validator; anchors on which of the two fields is populated.
- ``bundle_rewrite_carries_the_details_file_and_preserves_its_absence`` -> reaches
  ``bundle._emit._rewrite_paths_to_relative``; anchors on the rewritten string and on
  ``None`` staying ``None``.
- ``details_file_license_is_written_as_the_crates_own_creativework_entity`` -> reaches
  ``metadata.build_analysis_crate``; anchors on the emitted ``license`` @id and entity keys.
- ``a_table_member_crate_is_unchanged_by_the_details_file_form`` -> reaches the same
  builder; anchors on the entity's key SET, which no wording change moves.
- ``datacite_rightslist_for_a_details_file_license_names_it_without_a_scheme`` -> reaches
  ``publishing.build_datacite_rightslist``; anchors on the returned dict's keys.
- ``a_table_member_still_yields_the_five_field_spdx_entry`` -> reaches the same builder;
  anchors on the five-field key set.
- ``hydroshare_rights_come_from_the_record`` -> reaches ``publishing._hydroshare_rights``;
  anchors on the returned dict.
- ``read_back_recognizes_a_crate_that_names_its_own_license`` -> reaches
  ``publishing._read_license_from_sidecar``; anchors on the returned VALUE TYPE and its
  fields, for both a record and an SPDX id.
- ``read_back_still_refuses_a_crate_that_declares_nothing_it_can_name`` -> reaches the same
  reader; anchors on whether ``PublishError`` is raised.
- ``bundle_finding_is_silent_when_the_details_file_matches_the_crate`` -> reaches
  ``bundle._emit._crate_license_finding``; anchors on None-versus-finding.
- ``bundle_finding_offers_a_details_file_route_for_a_custom_crate_license`` -> reaches the
  same check; anchors on the presence of a keep-the-license remedy (a structural property
  of the message, present in the code's vocabulary in both states).
- ``bundle_finding_reports_an_unnameable_crate_license_against_an_unlicensed_config`` ->
  reaches the same check; anchors on None-versus-finding.
- ``zenodo_payload_sends_the_custom_form_and_no_lowercased_id`` -> reaches
  ``_ZenodoTarget.publish`` through a recorded session; anchors on the ``rights`` payload.
- ``zenodo_payload_omits_the_link_when_the_details_file_gives_none`` -> reaches the same
  target; anchors on the ``rights`` payload's key set.
- ``hydroshare_rights_come_from_the_record_on_a_stubbed_target`` -> reaches
  ``_HydroShareTarget.publish``; anchors on the rights dict the resource recorded.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hhemt import metadata, publishing
from hhemt.bundle._emit import _crate_license_finding, _rewrite_paths_to_relative
from hhemt.bundle._path_policy import PathPolicy
from hhemt.config.analysis import analysis_config
from hhemt.exceptions import ConfigurationError, PublishError
from hhemt.metadata import LicenseRecord
from tests.test_config_validation import _minimal_analysis_config_dict

CUSTOM_NAME = "Our Lab Data License v1.0"
CUSTOM_LINK = "https://example.org/licenses/our-lab-license-1.0"


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _details_file(tmp_path: Path, *, name: str = CUSTOM_NAME, link: str | None = CUSTOM_LINK) -> Path:
    """Write a license-details YAML file and return its path."""
    body = f"name: {name}\n" if link is None else f"name: {name}\nlink: {link}\n"
    path = tmp_path / "license_details.yaml"
    path.write_text(body)
    return path


def _crate_graph(*, license_uri, entity_name=None, entity_url=None):
    """A crate ``@graph`` whose root Dataset declares ``license_uri``.

    ``entity_name`` adds the CreativeWork entity a details-file license is written as; with
    no entity the URI is a bare declaration, which is the malformed-crate arm.
    """
    graph = [
        {"@id": "ro-crate-metadata.json", "@type": "CreativeWork", "about": {"@id": "./"}},
        {"@id": "./", "@type": "Dataset", "name": "T", "license": {"@id": license_uri}},
    ]
    if entity_name is not None:
        entity = {"@id": license_uri, "@type": "CreativeWork", "name": entity_name}
        if entity_url is not None:
            entity["url"] = entity_url
        graph.append(entity)
    return graph


def _write_sidecar(analysis_dir: Path, graph) -> None:
    (analysis_dir / "ro-crate-metadata.json").write_text(
        json.dumps({"@context": "https://w3id.org/ro/crate/1.1/context", "@graph": graph})
    )


def _fake_case():
    return SimpleNamespace(case_name="norfolk", description="test", manifest={})


def _build_crate(**over):
    kw = dict(
        analysis_id="a1",
        system_id="s1",
        layout_version=16,
        toolkit_git_sha="deadbeef",
        code_repository="https://example/repo",
        cfg_case=_fake_case(),
        sif_spec=None,
        consolidated_zarr_relpath="analysis_datatree.zarr",
        input_parts=[],
    )
    kw.update(over)
    return metadata.build_analysis_crate(**kw)


class _DumpableModel:
    """Stand-in config exposing the ``model_dump(mode=...)`` the deposit set writes."""

    def __init__(self, payload):
        self._payload = payload

    def model_dump(self, mode="json"):
        return dict(self._payload)


def _fake_analysis(analysis_dir: Path, *, dataset_license=None, dataset_license_file=None):
    cfg = _DumpableModel({"analysis_id": "a1"})
    cfg.dataset_license = dataset_license
    cfg.dataset_license_file = dataset_license_file
    return SimpleNamespace(
        analysis_paths=SimpleNamespace(analysis_dir=analysis_dir),
        cfg_analysis=cfg,
        _system=SimpleNamespace(cfg_system=_DumpableModel({"system_id": "s1"})),
    )


class _Resp:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = ""

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _RecordingZenodoSession:
    """Records every (method, url, json) and answers with InvenioRDM-shaped responses."""

    MINTED_DOI = "10.5281/zenodo.12345"

    def __init__(self):
        self.headers = {}
        self.calls = []

    def _respond(self, url):
        if url.endswith("/api/records"):
            return _Resp(201, {"id": "12345", "links": {"self_html": "https://zenodo.org/records/12345"}})
        if url.endswith("/api/records/12345"):
            return _Resp(
                200,
                {
                    "id": "12345",
                    "doi": self.MINTED_DOI,
                    "metadata": {"doi": self.MINTED_DOI},
                    "pids": None,
                    "links": {"self_html": "https://zenodo.org/records/12345"},
                },
            )
        if url.endswith("/actions/publish"):
            return _Resp(202, {"id": "12345", "pids": None})
        return _Resp(200, {})

    def post(self, url, json=None, data=None, headers=None, timeout=None):
        self.calls.append(("POST", url, json))
        return self._respond(url)

    def put(self, url, json=None, data=None, headers=None, timeout=None):
        self.calls.append(("PUT", url, json))
        return self._respond(url)

    def get(self, url, timeout=None):
        self.calls.append(("GET", url, None))
        return self._respond(url)

    def embedded_metadata(self) -> dict:
        """The ``metadata`` block the Phase-1b PUT carried to the draft record."""
        for method, url, body in self.calls:
            if method == "PUT" and url.endswith("/draft"):
                return body["metadata"]
        raise AssertionError("no embed-metadata PUT was recorded")


def _patch_zenodo(monkeypatch):
    sess = _RecordingZenodoSession()
    monkeypatch.setattr(publishing.requests, "Session", lambda: sess)
    monkeypatch.setenv("HHEMT_ZENODO_TOKEN", "tok")
    monkeypatch.delenv("HHEMT_ZENODO_BASE_URL", raising=False)
    return sess


# --------------------------------------------------------------------------------------
# The details-file loader
# --------------------------------------------------------------------------------------


def test_loader_reads_name_and_link(tmp_path):
    """Property: a details file's ``name`` and ``link`` reach a LicenseRecord unchanged.

    Class: every well-formed details file, with and without a link. Why required: the record
    is the ONLY carrier of a custom license's terms between the file, the crate entity and
    the three rights payloads, so a loader that dropped either field would ship a deposit
    whose every surface agrees on the wrong terms. Killed by: a loader returning a bare
    string, or one that resolves ``link`` to an absolute filesystem path.
    Second correct implementation: parsing with a two-field Pydantic model rather than the
    explicit mapping checks.

    Reachability: a details file reaches this failure through
    ``metadata.load_license_details``; artifact: the two asserts below.
    Both-states anchor: the assertion turns on the (name, link) pair, a property of the file
    in both the current and the proposed code, not on wording this change introduces.
    Invocation not isolatable from this tree — discrimination shown instead: implementing
    ``load_license_details`` greens this test and the pre-change tree has no such function
    (pasted in the module docstring).
    """
    record = metadata.load_license_details(_details_file(tmp_path))
    assert (record.name, record.link) == (CUSTOM_NAME, CUSTOM_LINK)
    linkless = metadata.load_license_details(_details_file(tmp_path, link=None))
    assert (linkless.name, linkless.link) == (CUSTOM_NAME, None)


@pytest.mark.parametrize(
    "body",
    [
        "link: https://example.org/x\n",  # no name
        "name: ''\n",  # empty name
        "name: A License\nlicence: https://example.org/x\n",  # key outside {name, link}
        "- name: A License\n",  # not a mapping
        "name: A License\nlink: 3\n",  # non-string link
    ],
)
def test_loader_refuses_a_malformed_details_file(tmp_path, body):
    """Property: every malformed details file is refused at load, naming the file.

    Class: a missing name, an empty name, an unrecognized key, a non-mapping document and a
    non-string link. Why required: an ignored ``link`` would publish the license with no
    address its reader could follow, and a silently-defaulted name would publish a deposit
    whose terms say nothing -- both are claims no later reader can correct from the deposit.
    Killed by: a loader that ignores unknown keys, or that accepts an empty name.
    Second correct implementation: a Pydantic model with ``extra='forbid'`` and
    ``min_length=1`` on the name.
    """
    path = tmp_path / "license_details.yaml"
    path.write_text(body)
    with pytest.raises(ConfigurationError) as exc:
        metadata.load_license_details(path)
    assert "dataset_license_file" in str(exc.value)
    assert str(path) in str(exc.value)


# --------------------------------------------------------------------------------------
# The config surface
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("dump_mode", ["python", "json"])
def test_config_with_a_details_file_round_trips_in_both_dump_modes(tmp_path, dump_mode):
    """Property: a config naming a details file round-trips through model_dump and model_validate.

    Class: every config naming a details file, dumped in either mode. Why required: the
    toolkit dumps and re-validates analysis configs at the sensitivity member rebuild, and
    writes ``cfg_analysis.yaml`` from ``model_dump(mode='json')`` at every deposit, so a
    field that loads but refuses its own dump breaks a re-run and the archived record.
    Killed by: declaring the field as a ``str`` whose python-mode dump re-validates into a
    differently-typed value, or as a required field.
    Second correct implementation: a plain ``Path | None`` field with no validator at all
    (which is what this is) versus a before-validator coercing ``str`` to ``Path``.
    """
    path = _details_file(tmp_path)
    cfg = _minimal_analysis_config_dict(tmp_path)
    cfg["dataset_license_file"] = str(path)
    loaded = analysis_config.model_validate(cfg)
    assert loaded.dataset_license_file == path
    reloaded = analysis_config.model_validate(loaded.model_dump(mode=dump_mode))
    assert reloaded.dataset_license_file == path
    assert reloaded.dataset_license is None


def test_config_accepts_either_license_source_alone_and_refuses_both(tmp_path):
    """Property: the two license sources are mutually exclusive, and neither is required.

    Class: neither set, each set alone, and both set. Why required: two sources would let
    the crate, the sidecar and the rights payloads disagree about one dataset's terms with
    no rule saying which wins, and the loser of that contest would be invisible in the
    deposit. Equally, D-106 makes a license required only by publish, so refusing a config
    that sets neither would break every analysis that never publishes.
    Killed by: a ``required_when`` between the two fields, in either direction.
    Second correct implementation: a ``model_validator(mode='before')`` raising on the raw
    mapping, or a ``model_validator(mode='after')`` as here.
    """
    path = _details_file(tmp_path)

    base = _minimal_analysis_config_dict(tmp_path)
    bare = analysis_config.model_validate(base)
    assert (bare.dataset_license, bare.dataset_license_file) == (None, None)

    only_id = analysis_config.model_validate({**base, "dataset_license": "CC0-1.0"})
    assert only_id.dataset_license_file is None

    only_file = analysis_config.model_validate({**base, "dataset_license_file": str(path)})
    assert only_file.dataset_license is None

    with pytest.raises(ValueError) as exc:
        analysis_config.model_validate({**base, "dataset_license": "CC0-1.0", "dataset_license_file": str(path)})
    assert "dataset_license_file" in str(exc.value)


def test_bundle_rewrite_carries_the_details_file_and_preserves_its_absence(tmp_path):
    """Property: the bundle path rule makes the details file bundle-relative and keeps None.

    Class: a details file outside both carried roots (the usual case — a laboratory keeps its
    license text beside its configs), one inside the system root, and a config naming none.
    Why required: the bundled ``cfg_analysis.yaml`` is an archive record whose absolute paths
    must not leak, and the license TERMS are the one thing a deposit cannot reconstruct if the
    bundle drops the file — an absolute host path resolves on the emitter's machine and nowhere
    else, so a consumer's publish would fail on a file that was never carried. Killed by:
    leaving the policy table without a row (``test_all_path_fields_have_policy`` also reds), a
    policy that blanks the field to None, or one that hands the absolute path through.
    Second correct implementation: ``BUNDLE_RELATIVE_OR_NONE`` as here, versus a new policy
    whose ``external/`` fallback is implemented separately.
    """
    base = _minimal_analysis_config_dict(tmp_path)
    system_root = tmp_path / "system"
    analysis_dir = system_root / "analysis"
    analysis_dir.mkdir(parents=True)

    # Out of tree: neither root contains it, so the policy's external/ fallback applies.
    outside = tmp_path / "license_details.yaml"
    outside.write_text(f"name: {CUSTOM_NAME}\n")
    cfg = analysis_config.model_validate({**base, "dataset_license_file": str(outside)})
    rewritten = _rewrite_paths_to_relative(cfg.model_dump(mode="json"), analysis_config, analysis_dir, system_root)
    assert rewritten.cfg_dict["dataset_license_file"] == "external/license_details.yaml"
    assert "dataset_license_file" in rewritten.invariants[PathPolicy.BUNDLE_RELATIVE_OR_NONE.value]

    # Inside the system root: rewritten relative to it, which is where the archive puts it.
    inside = system_root / "license_text.yaml"
    inside.write_text(f"name: {CUSTOM_NAME}\n")
    cfg = analysis_config.model_validate({**base, "dataset_license_file": str(inside)})
    assert (
        _rewrite_paths_to_relative(cfg.model_dump(mode="json"), analysis_config, analysis_dir, system_root).cfg_dict[
            "dataset_license_file"
        ]
        == "license_text.yaml"
    )

    absent = analysis_config.model_validate(base).model_dump(mode="json")
    assert (
        _rewrite_paths_to_relative(absent, analysis_config, analysis_dir, system_root).cfg_dict["dataset_license_file"]
        is None
    )


# --------------------------------------------------------------------------------------
# Site 1 — metadata.build_analysis_crate: the custom-license crate entity
# --------------------------------------------------------------------------------------


def test_details_file_license_is_written_as_the_crates_own_creativework_entity():
    """Property: a details-file license becomes a CreativeWork entity the crate itself carries.

    Class: every details-file license, with and without a link. Why required: the entity is
    how the license is recognizable on read-back at all — the table lookup cannot name a
    license it does not contain, so without an entity the crate's own terms would be
    unreadable and publish would refuse a correctly-licensed analysis. Killed by: writing
    only ``root['license']`` with no entity, or writing an entity whose ``name`` is the URI.
    Second correct implementation: an entity at a fixed ``#dataset-license`` id carrying the
    link as ``url``, instead of the link as the ``@id``.

    PRE-CHANGE: ``build_analysis_crate(dataset_license=<custom name>)`` raised
    ``KeyError: 'Our Lab Data License v1.0'`` (module docstring, site1).
    """
    record = LicenseRecord(name=CUSTOM_NAME, link=CUSTOM_LINK)
    doc = json.loads(metadata.canonical_jsonld(_build_crate(dataset_license=record)))
    root = next(e for e in doc["@graph"] if e["@id"] == "./")
    assert root["license"] == {"@id": CUSTOM_LINK}
    entity = next(e for e in doc["@graph"] if e["@id"] == CUSTOM_LINK)
    assert entity["@type"] == "CreativeWork"
    assert entity["name"] == CUSTOM_NAME
    assert entity["url"] == CUSTOM_LINK

    linkless = LicenseRecord(name=CUSTOM_NAME)
    doc = json.loads(metadata.canonical_jsonld(_build_crate(dataset_license=linkless)))
    root = next(e for e in doc["@graph"] if e["@id"] == "./")
    assert root["license"] == {"@id": metadata._LOCAL_LICENSE_ENTITY_ID}
    entity = next(e for e in doc["@graph"] if e["@id"] == metadata._LOCAL_LICENSE_ENTITY_ID)
    assert "url" not in entity


def test_a_table_member_crate_is_unchanged_by_the_details_file_form():
    """Property: a table member's crate is byte-identical to what it was before the new form.

    Class: every SPDX table member. Why required: this field is additive, and every
    consolidating analysis in existence carries a table member; a crate that gained a ``url``
    property or moved its license @id would rewrite every published sidecar on the next
    re-consolidation, which is a migration this feature does not ship. Killed by: routing
    table members through the record branch, which writes ``url``.
    Second correct implementation: routing them through the record branch but omitting
    ``url`` for a table URI (a form that would still pass only if the URI and the name are
    both right).
    """
    doc = json.loads(metadata.canonical_jsonld(_build_crate(dataset_license="CC0-1.0")))
    root = next(e for e in doc["@graph"] if e["@id"] == "./")
    assert root["license"] == {"@id": "https://spdx.org/licenses/CC0-1.0"}
    entity = next(e for e in doc["@graph"] if e["@id"] == "https://spdx.org/licenses/CC0-1.0")
    assert set(entity) == {"@id", "@type", "name"}


# --------------------------------------------------------------------------------------
# Site 2 — publishing.build_datacite_rightslist
# --------------------------------------------------------------------------------------


def test_datacite_rightslist_for_a_details_file_license_names_it_without_a_scheme():
    """Property: a custom license yields name and (when given) URI, and claims no SPDX id.

    Class: every details-file license. Why required: a ``rightsIdentifier`` of a name that is
    not an SPDX id asserts an identifier scheme the license does not belong to, and a
    registry that resolves it fails or records a false claim. Killed by: emitting the five
    SPDX fields with the license name in ``rightsIdentifier``.
    Second correct implementation: ``{"rights": name, "rightsUri": link}`` built from the
    record's fields in the opposite order, or a five-field entry with the SPDX fields omitted
    rather than absent.

    PRE-CHANGE: ``build_datacite_rightslist(<custom name>)`` raised
    ``KeyError: 'Our Lab Data License v1.0'`` (module docstring, site2).
    """
    with_link = publishing.build_datacite_rightslist(LicenseRecord(name=CUSTOM_NAME, link=CUSTOM_LINK))
    assert with_link == [{"rights": CUSTOM_NAME, "rightsUri": CUSTOM_LINK}]
    linkless = publishing.build_datacite_rightslist(LicenseRecord(name=CUSTOM_NAME))
    assert linkless == [{"rights": CUSTOM_NAME}]


def test_a_table_member_still_yields_the_five_field_spdx_entry():
    """Property: a table member's DataCite entry is the five-field SPDX one, unchanged.

    Class: every SPDX table member. Why required: the same additive guarantee as the crate —
    a published dataset's rightsList must not change shape because a second license form
    arrived. Killed by: routing table members through the custom branch.
    Second correct implementation: the same dict built field by field from the record.
    """
    entry = publishing.build_datacite_rightslist("CC0-1.0")[0]
    assert set(entry) == {"rights", "rightsUri", "rightsIdentifier", "rightsIdentifierScheme", "schemeUri"}
    assert entry["rightsIdentifier"] == "CC0-1.0"
    assert entry["rightsIdentifierScheme"] == "SPDX"


# --------------------------------------------------------------------------------------
# Site 3 — publishing._hydroshare_rights
# --------------------------------------------------------------------------------------


def test_hydroshare_rights_come_from_the_record(tmp_path):
    """Property: HydroShare's rights statement is the details file's name, its url its link.

    Class: every details-file license, with and without a link. Why required: HydroShare
    takes free text, so the record needs no translation — but it must come from the RECORD,
    because the table has no entry to translate and a lookup would raise at deposit time,
    after the upload. Killed by: looking the record up in the table, or blanking ``url`` to
    an empty string when there is no link.
    Second correct implementation: ``{"statement": name, "url": link or ""}`` with the empty
    string tolerated by hsclient (a form this test rejects, because an empty url reads as an
    address the owner never published).

    PRE-CHANGE: ``_hydroshare_rights(<custom name>)`` raised
    ``KeyError: 'Our Lab Data License v1.0'`` (module docstring, site3).
    """
    assert publishing._hydroshare_rights(LicenseRecord(name=CUSTOM_NAME, link=CUSTOM_LINK)) == {
        "statement": CUSTOM_NAME,
        "url": CUSTOM_LINK,
    }
    assert publishing._hydroshare_rights(LicenseRecord(name=CUSTOM_NAME)) == {"statement": CUSTOM_NAME}


# --------------------------------------------------------------------------------------
# Site 5 — publishing._read_license_from_sidecar
# --------------------------------------------------------------------------------------


def test_read_back_recognizes_a_crate_that_names_its_own_license(tmp_path):
    """Property: a crate carrying its own CreativeWork entity reads back as that record.

    Class: every crate written with a details-file license. Why required: this is the read
    side of the same feature, and it is what makes the docstring's claim true — a custom
    license is recognizable because the CRATE names it, not because a vocabulary this
    toolkit ships contains it. Killed by: recognizing only table URIs, which refuses a
    correctly-licensed crate at deposit.
    Second correct implementation: matching on the entity's ``@type`` and reading the name
    from there, rather than matching on the license ``@id``.

    PRE-CHANGE: this exact crate raised
    ``PublishError: ... no recognizable dataset license in ro-crate sidecar``
    (module docstring, site5).
    """
    _write_sidecar(
        tmp_path,
        _crate_graph(license_uri=CUSTOM_LINK, entity_name=CUSTOM_NAME, entity_url=CUSTOM_LINK),
    )
    assert publishing._read_license_from_sidecar(tmp_path) == LicenseRecord(name=CUSTOM_NAME, link=CUSTOM_LINK)
    # A table member still reads back as its SPDX ID string, not as a record, so every
    # caller that compares it against a config field keeps working (module docstring,
    # control line).
    _write_sidecar(tmp_path, _crate_graph(license_uri="https://spdx.org/licenses/CC0-1.0"))
    assert publishing._read_license_from_sidecar(tmp_path) == "CC0-1.0"


def test_read_back_still_refuses_a_crate_that_declares_nothing_it_can_name(tmp_path):
    """Property: a bare unknown license URI with no entity behind it is still refused.

    Class: every crate whose root license @id is neither a table URI nor an entity of its
    own. Why required: the details-file form must widen recognition by exactly the crates
    that describe their own terms, not by every string — otherwise a malformed or truncated
    crate would deposit under terms nothing can name. This is the arm that keeps the
    existing contract the same; the pre-change reader refused it too.
    Killed by: falling back to a synthetic record built from the URI.
    Second correct implementation: refusing with a distinct error type for
    "declared but unrecognizable" versus "declared nothing".
    """
    _write_sidecar(tmp_path, _crate_graph(license_uri="https://example/nope"))
    with pytest.raises(PublishError):
        publishing._read_license_from_sidecar(tmp_path)


# --------------------------------------------------------------------------------------
# Site 4 — bundle/_emit._crate_license_finding, compared by record
# --------------------------------------------------------------------------------------


def test_bundle_finding_is_silent_when_the_details_file_matches_the_crate(tmp_path):
    """Property: a details-file config that agrees with the crate produces no finding.

    Class: every config whose license comes from a details file, against a crate carrying
    the same terms. Why required: the check reports STALENESS, and a run whose config and
    crate agree must not put a false "crate predates the config" line in the manifest of
    every bundle it emits — which trains the user to ignore the line on the run where it is
    real. Killed by: reading only ``dataset_license`` on the config side, which sees None.
    Second correct implementation: comparing license URIs after resolving both sides from a
    shared record builder.

    PRE-CHANGE: this exact input RETURNED a false staleness finding
    (module docstring, site4).
    """
    path = _details_file(tmp_path)
    _write_sidecar(
        tmp_path,
        _crate_graph(license_uri=CUSTOM_LINK, entity_name=CUSTOM_NAME, entity_url=CUSTOM_LINK),
    )
    analysis = _fake_analysis(tmp_path, dataset_license_file=path)
    assert _crate_license_finding(analysis, tmp_path) is None


def test_bundle_finding_offers_a_details_file_route_for_a_custom_crate_license(tmp_path):
    """Property: a custom crate license gets the no-rebuild remedy, naming the details-file field.

    Class: every config that declares no license while the crate carries a custom one. Why
    required: the finding's two remedies are ordered so the no-rebuild one comes first, and
    for a custom license that remedy is expressible ONLY through the details-file field. A
    suppressed remedy leaves the user with a full re-consolidation as the only option.
    Killed by: suppressing the keep remedy whenever the crate's license has no SPDX id.
    Second correct implementation: naming the field and the crate's license name in either
    order, or reproducing the crate's name and link as a fenced YAML snippet.

    PRE-CHANGE: the keep remedy was absent — ``offers a keep-the-crate's-licence remedy:
    False`` (module docstring, site6).
    """
    _write_sidecar(
        tmp_path,
        _crate_graph(license_uri=CUSTOM_LINK, entity_name=CUSTOM_NAME, entity_url=CUSTOM_LINK),
    )
    analysis = _fake_analysis(tmp_path, dataset_license=None)
    finding = _crate_license_finding(analysis, tmp_path)
    assert finding is not None
    assert "To keep the crate's license" in finding
    assert "dataset_license_file" in finding
    assert CUSTOM_NAME in finding
    assert "regenerate_existing=True" in finding


def test_bundle_finding_reports_an_unnameable_crate_license_against_an_unlicensed_config(tmp_path):
    """Property: a crate license this toolkit cannot name is reported even on an unlicensed config.

    Class: every crate whose root ``Dataset`` license ``@id`` is neither a table URI nor
    carried by a ``CreativeWork`` entity of the crate, against a config declaring no license
    at all. Why required: the two sides are compared BY RECORD, and
    ``metadata.read_crate_license`` raises ``UnrecognizedCrateLicense`` precisely to keep
    "declares nothing" and "declares something unnameable" apart. Collapsing that exception
    into ``crate_declared = None`` makes ``None != None`` false, so the staleness finding
    disappears and a bundle ships a crate carrying terms nothing in the deposit can name --
    the one case the check exists to surface on a hand-built crate.
    Killed by: assigning ``crate_declared = None`` in the except arm and comparing the two
    records alone, with no memory of the exception (the pre-change code).
    Second correct implementation: carrying the unrecognized URI as a sentinel object whose
    inequality against ``None`` is total, instead of a separate boolean beside the record.
    """
    _write_sidecar(tmp_path, _crate_graph(license_uri="https://creativecommons.org/licenses/by/4.0/"))
    analysis = _fake_analysis(tmp_path, dataset_license=None, dataset_license_file=None)
    finding = _crate_license_finding(analysis, tmp_path)
    assert finding is not None
    assert "https://creativecommons.org/licenses/by/4.0/" in finding


# --------------------------------------------------------------------------------------
# Publish, per target, on a stubbed target
# --------------------------------------------------------------------------------------


def test_zenodo_payload_sends_the_custom_form_and_no_lowercased_id(tmp_path, monkeypatch):
    """Property: the Zenodo ``rights`` carries title+link and never an id built from the name.

    Class: every details-file license deposited to Zenodo, with and without a link. Why
    required: InvenioRDM validates ``rights`` against its own vocabulary, and the previous
    body lower-cased whatever it was handed — so a free-form name shipped as
    ``{"id": "our lab data license v1.0"}``, an id the vocabulary does not contain and a
    silent misstatement of the license. Killed by: lower-casing the record's name into an
    ``id``, or omitting the ``link`` the file gave.
    Second correct implementation: emitting ``{"title": {"en": name}}`` with the link under
    ``props.url`` instead of ``link`` — a form this test rejects because the docstring's
    contract names ``link``.

    PRE-CHANGE: ``build_inveniordm_rights(<custom name>)`` returned
    ``[{"id": '<the name, lower-cased>'}]`` — the defect this row's chunk names.
    """
    path = _details_file(tmp_path)
    _write_sidecar(
        tmp_path,
        _crate_graph(license_uri=CUSTOM_LINK, entity_name=CUSTOM_NAME, entity_url=CUSTOM_LINK),
    )
    (tmp_path / "analysis_datatree.zarr").mkdir()
    sess = _patch_zenodo(monkeypatch)

    publishing.publish_analysis(_fake_analysis(tmp_path, dataset_license_file=path), target="zenodo")

    rights = sess.embedded_metadata()["rights"]
    assert rights == [{"title": {"en": CUSTOM_NAME}, "link": CUSTOM_LINK}]
    assert all("id" not in entry for entry in rights)
    assert CUSTOM_NAME.lower() not in json.dumps(rights)


def test_zenodo_payload_omits_the_link_when_the_details_file_gives_none(tmp_path, monkeypatch):
    """Property: a linkless details file produces a title-only custom entry.

    Class: every details-file license with no link. Why required: a record with no published
    address must not send an empty ``link``, which a reader resolves as a broken address
    rather than as "not published at a URL". Killed by: sending ``"link": ""``.
    Second correct implementation: omitting the whole ``link`` key versus sending ``null``
    (rejected here, because a null link is a claim the terms carry an address).
    """
    path = _details_file(tmp_path, link=None)
    entity_graph = _crate_graph(license_uri=metadata._LOCAL_LICENSE_ENTITY_ID, entity_name=CUSTOM_NAME, entity_url=None)
    _write_sidecar(tmp_path, entity_graph)
    (tmp_path / "analysis_datatree.zarr").mkdir()
    sess = _RecordingZenodoSession()
    monkeypatch.setattr(publishing.requests, "Session", lambda: sess)
    monkeypatch.setenv("HHEMT_ZENODO_TOKEN", "tok")
    monkeypatch.delenv("HHEMT_ZENODO_BASE_URL", raising=False)

    publishing.publish_analysis(_fake_analysis(tmp_path, dataset_license_file=path), target="zenodo")

    assert sess.embedded_metadata()["rights"] == [{"title": {"en": CUSTOM_NAME}}]


def test_hydroshare_rights_come_from_the_record_on_a_stubbed_target(tmp_path, monkeypatch):
    """Property: a HydroShare deposit sets its rights statement from the crate's record.

    Class: every details-file license deposited to HydroShare. Why required: HydroShare is
    the target whose metadata is written by hsclient, so a table lookup here fails after the
    upload has already happened — the one place a late failure costs a published resource.
    Killed by: passing the license NAME into ``_hydroshare_rights``, which indexes the table.
    Second correct implementation: setting ``rights`` from the record inside a target object
    that takes the record rather than a string.

    PRE-CHANGE: ``_hydroshare_rights(<custom name>)`` raised
    ``KeyError: 'Our Lab Data License v1.0'`` (module docstring, site3).
    """
    path = _details_file(tmp_path)
    _write_sidecar(
        tmp_path,
        _crate_graph(license_uri=CUSTOM_LINK, entity_name=CUSTOM_NAME, entity_url=CUSTOM_LINK),
    )
    (tmp_path / "analysis_datatree.zarr").mkdir()

    recorded = SimpleNamespace(rights=None, relations=None)

    class _Resource:
        resource_id = "res-1"
        metadata = recorded

        def file_upload(self, _path):
            return None

        def save(self):
            return None

        def set_sharing_status(self, **_kw):
            return None

    class _HydroShare:
        def __init__(self, **_kw):
            pass

        def create(self):
            return _Resource()

    fake_module = SimpleNamespace(HydroShare=_HydroShare)
    monkeypatch.setitem(sys.modules, "hsclient", fake_module)
    monkeypatch.setenv("HHEMT_HYDROSHARE_USERNAME", "u")
    monkeypatch.setenv("HHEMT_HYDROSHARE_PASSWORD", "p")

    result = publishing.publish_analysis(_fake_analysis(tmp_path, dataset_license_file=path), target="hydroshare")

    assert recorded.rights == {"statement": CUSTOM_NAME, "url": CUSTOM_LINK}
    assert result["target"] == "hydroshare"
