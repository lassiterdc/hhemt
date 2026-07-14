"""Publishing subsystem (C6) — deposit an analysis to a DOI-minting repo (ADR-11).

Reads the already-emitted ro-crate-metadata.json sidecar (the license + parts), layers
DataCite on top, and deposits to Zenodo (programmatic reserve-DOI) or HydroShare
(two-invocation flow; hsclient v1.1.6 has no programmatic DOI mint). NEVER re-runs
emit_provenance (crosswalk consumer, not a crate re-emitter).

Substrate note (plan Assumption A4 / Risk X4): the concrete Zenodo/InvenioRDM deposit
REST endpoints + payload schemas and the hsclient write-side calls are NOT in specialist
substrate. The HTTP call bodies below are authored against the InvenioRDM REST reference
(``https://inveniordm.docs.cern.ch``, which backs zenodo.org) and the hsclient API; the
endpoint paths are named module constants (``_ZENODO_*``) so a durable-substrate
correction is a one-line change. Live deposits run ONLY under the ``publish_e2e`` marker
(``HHEMT_PUBLISH_E2E=1``); the mock-HTTP tests pin the two-phase control flow and payload
shape, so live-API field drift cannot silently break CI.
"""

from __future__ import annotations

import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Literal

import requests
import yaml

from hhemt.exceptions import PublishError
from hhemt.metadata import _SPDX_LICENSE_TABLE

# D3 (resolved): the data->software relatedIdentifier relationType. Single tunable —
# one-token fallback to "IsDerivedFrom" if a target registry UI hides IsCompiledBy.
_DATA_TO_SOFTWARE_RELATION = "IsCompiledBy"
# The reciprocal edge written back onto the software record in the Phase-2 backfill.
_SOFTWARE_TO_DATA_RELATION = "IsSourceOf"

# InvenioRDM REST endpoint templates (A4/X4: named so a substrate correction is one line).
_ZENODO_DEFAULT_BASE = "https://zenodo.org"
_ZENODO_CREATE = "{base}/api/records"
_ZENODO_RESERVE_DOI = "{base}/api/records/{recid}/draft/pids/doi"
_ZENODO_DRAFT = "{base}/api/records/{recid}/draft"
_ZENODO_FILES_INIT = "{base}/api/records/{recid}/draft/files"
_ZENODO_FILE_CONTENT = "{base}/api/records/{recid}/draft/files/{key}/content"
_ZENODO_FILE_COMMIT = "{base}/api/records/{recid}/draft/files/{key}/commit"
_ZENODO_PUBLISH = "{base}/api/records/{recid}/draft/actions/publish"

_HTTP_TIMEOUT = 60


def _read_license_from_sidecar(analysis_dir: Path) -> str:
    """Return the SPDX id from the root Dataset.license @id in the emitted sidecar."""
    doc = json.loads((Path(analysis_dir) / "ro-crate-metadata.json").read_text())
    for e in doc.get("@graph", []):
        if e.get("@id") in ("./", ""):
            lic = e.get("license")
            uri = lic.get("@id") if isinstance(lic, dict) else lic
            for spdx, meta in _SPDX_LICENSE_TABLE.items():
                if meta["uri"] == uri:
                    return spdx
    raise PublishError(target="?", doi=None, status="no recognizable dataset license in ro-crate sidecar")


def build_datacite_rightslist(dataset_license_spdx: str) -> list[dict]:
    """Return the DataCite ``rightsList`` (5-field SPDX entry) for the dataset license."""
    e = _SPDX_LICENSE_TABLE[dataset_license_spdx]
    return [
        {
            "rights": e["name"],
            "rightsUri": e["uri"],
            "rightsIdentifier": dataset_license_spdx,
            "rightsIdentifierScheme": "SPDX",
            "schemeUri": e["scheme_uri"],
        }
    ]


def build_datacite_related(*, software_doi: str | None, paper_doi: str | None = None) -> list[dict]:
    """Return the DataCite ``relatedIdentifiers`` edges (data->software, data->paper)."""
    edges: list[dict] = []
    if software_doi:
        edges.append(
            {
                "relatedIdentifier": software_doi,
                "relatedIdentifierType": "DOI",
                "relationType": _DATA_TO_SOFTWARE_RELATION,
            }
        )
    if paper_doi:
        edges.append(
            {
                "relatedIdentifier": paper_doi,
                "relatedIdentifierType": "DOI",
                "relationType": "IsSupplementTo",
            }
        )
    return edges


def _deposit_set(analysis, consolidated_zarr_relpath: str) -> list[Path]:
    """The v1 analysis_dir deposit set: consolidated zarr + sidecar + configs.

    Configs are NOT in the live analysis_dir by construction (analysis.py:771-776);
    materialize the two fixed-name configs into the deposit exactly as analysis.eda()
    does (analysis.py:779-780). Takes the analysis (not just analysis_dir) so it can
    read the config models — see publish_analysis below.
    """
    root = Path(analysis.analysis_paths.analysis_dir)
    (root / "cfg_analysis.yaml").write_text(yaml.safe_dump(analysis.cfg_analysis.model_dump(mode="json")))
    (root / "cfg_system.yaml").write_text(yaml.safe_dump(analysis._system.cfg_system.model_dump(mode="json")))
    parts = [
        root / "ro-crate-metadata.json",
        root / consolidated_zarr_relpath,
        root / "cfg_analysis.yaml",
        root / "cfg_system.yaml",
    ]
    return [p for p in parts if p.exists()]


def _require_env(name: str, target: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise PublishError(target=target, doi=None, status=f"missing required credential env var {name}")
    return val


def _check(resp, *, target: str, doi: str | None, step: str):
    """Raise PublishError on a non-2xx response; otherwise return the parsed JSON (or {})."""
    if resp.status_code >= 400:
        raise PublishError(target=target, doi=doi, status=f"{step} failed: HTTP {resp.status_code} {resp.text[:300]}")
    try:
        return resp.json()
    except ValueError:
        return {}


def _extract_reserved_doi(payload: dict) -> str | None:
    """Read the reserved DOI from an InvenioRDM draft (fallback to classic prereserve_doi)."""
    pids = payload.get("pids") or {}
    doi = (pids.get("doi") or {}).get("identifier")
    if doi:
        return doi
    prereserve = (payload.get("metadata") or {}).get("prereserve_doi") or {}
    return prereserve.get("doi")


def _record_url(payload: dict) -> str | None:
    links = payload.get("links") or {}
    return links.get("self_html") or links.get("self")


def _resolve_upload(path: Path, staging: Path) -> tuple[Path, str]:
    """Return (upload_path, deposit_key). A directory (zarr store) is zipped into staging."""
    if path.is_dir():
        archive = staging / f"{path.name}.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
            for sub in sorted(path.rglob("*")):
                if sub.is_file():
                    zf.write(sub, sub.relative_to(path.parent))
        return archive, archive.name
    return path, path.name


class _ZenodoTarget:
    """Programmatic reserve-DOI two-phase flow against the Zenodo/InvenioRDM REST API."""

    def publish(self, *, deposit, license_spdx, software_doi, analysis_dir) -> dict:
        # create draft -> POST reserve-DOI -> embed {reserved data-DOI, software-DOI,
        #   IsCompiledBy relatedIdentifier, rightsList} -> upload files -> publish
        #   -> Phase-2 backfill reciprocal edge on the software record.
        base = os.environ.get("HHEMT_ZENODO_BASE_URL", _ZENODO_DEFAULT_BASE).rstrip("/")
        token = _require_env("HHEMT_ZENODO_TOKEN", "zenodo")
        session = requests.Session()
        session.headers.update({"Authorization": f"Bearer {token}", "Content-Type": "application/json"})

        # Phase 1a — create the draft record.
        create_body = {
            "access": {"record": "public", "files": "public"},
            "files": {"enabled": True},
            "metadata": {"resource_type": {"id": "dataset"}},
        }
        created = _check(
            session.post(_ZENODO_CREATE.format(base=base), json=create_body, timeout=_HTTP_TIMEOUT),
            target="zenodo",
            doi=None,
            step="create draft",
        )
        recid = created.get("id")

        # Phase 1b — reserve the DOI (distinct InvenioRDM action).
        reserved = _check(
            session.post(_ZENODO_RESERVE_DOI.format(base=base, recid=recid), timeout=_HTTP_TIMEOUT),
            target="zenodo",
            doi=None,
            step="reserve DOI",
        )
        data_doi = _extract_reserved_doi(reserved) or _extract_reserved_doi(created)

        # Phase 1c — embed the reserved data-DOI + DataCite rightsList + IsCompiledBy edge.
        embed_metadata = {
            "resource_type": {"id": "dataset"},
            "rightsList": build_datacite_rightslist(license_spdx),
            "relatedIdentifiers": build_datacite_related(software_doi=software_doi),
        }
        embed_body = {"metadata": embed_metadata, "pids": {"doi": {"identifier": data_doi, "provider": "datacite"}}}
        _check(
            session.put(_ZENODO_DRAFT.format(base=base, recid=recid), json=embed_body, timeout=_HTTP_TIMEOUT),
            target="zenodo",
            doi=data_doi,
            step="embed metadata",
        )

        # Phase 1d — upload the deposit files (init -> PUT content -> commit, per file).
        # Zarr stores are directories; deposit each as a single .zip (Zenodo stores loose
        # files, not directory trees). Temp archives are removed after upload.
        with tempfile.TemporaryDirectory() as staging:
            for path in deposit:
                upload_path, key = _resolve_upload(Path(path), Path(staging))
                _check(
                    session.post(
                        _ZENODO_FILES_INIT.format(base=base, recid=recid),
                        json=[{"key": key}],
                        timeout=_HTTP_TIMEOUT,
                    ),
                    target="zenodo",
                    doi=data_doi,
                    step=f"init file {key}",
                )
                with open(upload_path, "rb") as fh:
                    _check(
                        session.put(
                            _ZENODO_FILE_CONTENT.format(base=base, recid=recid, key=key),
                            data=fh,
                            headers={"Content-Type": "application/octet-stream"},
                            timeout=_HTTP_TIMEOUT,
                        ),
                        target="zenodo",
                        doi=data_doi,
                        step=f"upload file {key}",
                    )
                _check(
                    session.post(_ZENODO_FILE_COMMIT.format(base=base, recid=recid, key=key), timeout=_HTTP_TIMEOUT),
                    target="zenodo",
                    doi=data_doi,
                    step=f"commit file {key}",
                )

        # Phase 1e — publish.
        published = _check(
            session.post(_ZENODO_PUBLISH.format(base=base, recid=recid), timeout=_HTTP_TIMEOUT),
            target="zenodo",
            doi=data_doi,
            step="publish",
        )

        # Phase 2 — backfill the reciprocal IsSourceOf edge onto the software record.
        if software_doi:
            self._backfill_software_edge(session, base=base, software_doi=software_doi, data_doi=data_doi)

        return {
            "target": "zenodo",
            "data_doi": data_doi,
            "software_doi": software_doi,
            "record_url": _record_url(published) or _record_url(created),
        }

    def _backfill_software_edge(self, session, *, base: str, software_doi: str, data_doi: str | None) -> None:
        """Add the reciprocal IsSourceOf edge onto the software record (best-effort).

        The data record is already published; a backfill failure must not raise past this
        point. Resolving a DOI to an editable InvenioRDM record requires the software
        record's recid, obtained here only for a Zenodo-hosted software DOI.
        """
        if "zenodo." not in software_doi:
            return
        soft_recid = software_doi.rsplit("zenodo.", 1)[-1]
        edge = {
            "relatedIdentifier": data_doi,
            "relatedIdentifierType": "DOI",
            "relationType": _SOFTWARE_TO_DATA_RELATION,
        }
        try:
            session.put(
                _ZENODO_DRAFT.format(base=base, recid=soft_recid),
                json={"metadata": {"relatedIdentifiers": [edge]}},
                timeout=_HTTP_TIMEOUT,
            )
        except requests.RequestException:
            # Reciprocal edge is advisory; the primary deposit already succeeded.
            return


class _HydroShareTarget:
    """Two-invocation flow: hsclient publish-to-public -> manual web-UI DOI -> backfill."""

    def publish(self, *, deposit, license_spdx, software_doi, analysis_dir) -> dict:
        # Invocation 1: hs.create -> upload -> set metadata -> set_sharing_status(public=True),
        #   then STOP and surface the manual web-UI DOI-mint instruction (hsclient v1.1.6 has
        #   no programmatic .publish()/DOI mint).
        from hsclient import HydroShare

        username = _require_env("HHEMT_HYDROSHARE_USERNAME", "hydroshare")
        password = _require_env("HHEMT_HYDROSHARE_PASSWORD", "hydroshare")
        hs = HydroShare(username=username, password=password)

        resource = hs.create()
        resource_id = resource.resource_id
        try:
            for path in deposit:
                resource.file_upload(str(path))
            resource.metadata.rights = _hydroshare_rights(license_spdx)
            if software_doi:
                resource.metadata.relations = [
                    _hydroshare_relation(software_doi, _DATA_TO_SOFTWARE_RELATION),
                ]
            resource.save()
            resource.set_sharing_status(public=True)
        except Exception as exc:  # noqa: BLE001 — surface any hsclient failure as PublishError
            raise PublishError(target="hydroshare", doi=None, status=f"deposit failed: {exc}") from exc

        record_url = f"https://www.hydroshare.org/resource/{resource_id}/"
        return {
            "target": "hydroshare",
            "data_doi": None,
            "software_doi": software_doi,
            "record_url": record_url,
            "manual_step": (
                "HydroShare resource is public. hsclient v1.1.6 cannot mint a DOI programmatically — "
                f"open {record_url} and use 'Publish' in the web UI to mint the DOI, then re-run publish "
                "with the minted DOI to backfill the reciprocal software edge."
            ),
        }


def _hydroshare_rights(license_spdx: str) -> dict:
    e = _SPDX_LICENSE_TABLE[license_spdx]
    return {"statement": e["name"], "url": e["uri"]}


def _hydroshare_relation(doi: str, relation_type: str) -> dict:
    return {"type": relation_type, "value": f"https://doi.org/{doi.lstrip('/')}"}


_TARGETS = {"zenodo": _ZenodoTarget, "hydroshare": _HydroShareTarget}


def publish_analysis(
    analysis,
    *,
    target: Literal["zenodo", "hydroshare"],
    override_dataset_license: str | None = None,
    software_doi: str | None = None,
    consolidated_zarr_relpath: str = "analysis_datatree.zarr",
) -> dict:
    """Deposit an analysis to a DOI-minting repository (C6, ADR-11).

    Reads the license baked into the emitted ro-crate sidecar; override_dataset_license
    ASSERTS the caller's expected value against the sidecar (raising PublishError on
    mismatch) rather than re-stamping the archived crate.
    """
    analysis_dir = analysis.analysis_paths.analysis_dir
    license_spdx = _read_license_from_sidecar(analysis_dir)
    if override_dataset_license is not None and override_dataset_license != license_spdx:
        raise PublishError(
            target=target,
            doi=None,
            status=(
                f"override_dataset_license={override_dataset_license!r} differs from the crate "
                f"license {license_spdx!r}; set analysis_config.dataset_license and "
                f"reprocess(start_with='consolidate') first — publish does not re-stamp the archived zarr."
            ),
        )
    deposit = _deposit_set(analysis, consolidated_zarr_relpath)
    return _TARGETS[target]().publish(
        deposit=deposit,
        license_spdx=license_spdx,
        software_doi=software_doi,
        analysis_dir=analysis_dir,
    )
