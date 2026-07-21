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
import warnings
import zipfile
from datetime import date
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

# Default dataset creator when the operator supplies none. An organizational creator is a
# valid DataCite/InvenioRDM creator and unblocks publish; the operator SHOULD pass real
# personal creators via publish_analysis(creators=...) for proper authorship credit. The
# crate sidecar carries no author/creator entity, so this is the only creators source
# unless the operator overrides it.
_DEFAULT_CREATORS: list[dict] = [
    {"person_or_org": {"type": "organizational", "name": "H&H Ensemble Modeling Toolkit"}}
]

# DataCite-mandatory publisher (the entity issuing the resource). Required for DOI
# registration; Zenodo's own deposit UI defaults it to "Zenodo" for records it issues.
# This is the 6th and last DataCite-mandatory field (Identifier/Creators/Title/Publisher/
# PublicationYear/ResourceType) — the others land via the minted DOI + the metadata below.
_PUBLISHER = "Zenodo"

# InvenioRDM REST endpoint templates (A4/X4: named so a substrate correction is one line).
_ZENODO_DEFAULT_BASE = "https://zenodo.org"
_ZENODO_CREATE = "{base}/api/records"
_ZENODO_RESERVE_DOI = "{base}/api/records/{recid}/draft/pids/doi"
_ZENODO_DRAFT = "{base}/api/records/{recid}/draft"
_ZENODO_FILES_INIT = "{base}/api/records/{recid}/draft/files"
_ZENODO_FILE_CONTENT = "{base}/api/records/{recid}/draft/files/{key}/content"
_ZENODO_FILE_COMMIT = "{base}/api/records/{recid}/draft/files/{key}/commit"
_ZENODO_PUBLISH = "{base}/api/records/{recid}/draft/actions/publish"
_ZENODO_RECORD = "{base}/api/records/{recid}"

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


def _read_title_from_sidecar(analysis_dir: Path) -> str:
    """Return the root Dataset ``name`` from the emitted sidecar (generic fallback).

    ``name`` is in the emitted crate's embedded core, so it survives partitioning. A
    missing/empty/unreadable sidecar falls back to a generic title rather than raising —
    a missing title must never block an otherwise-valid deposit.
    """
    try:
        doc = json.loads((Path(analysis_dir) / "ro-crate-metadata.json").read_text())
    except (OSError, ValueError):
        return "HHEMT analysis dataset"
    for e in doc.get("@graph", []):
        if e.get("@id") in ("./", ""):
            name = e.get("name")
            if name:
                return name
    return "HHEMT analysis dataset"


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


def build_inveniordm_rights(dataset_license_spdx: str) -> list[dict]:
    """InvenioRDM-native ``rights`` (license) entry — NOT the DataCite rightsList.

    InvenioRDM keys its license vocabulary on the LOWERCASED SPDX id; the DataCite
    rightsList names (rights/rightsUri/rightsIdentifier/rightsIdentifierScheme/schemeUri)
    are unknown to the InvenioRDM metadata loader and are silently dropped, so the license
    never lands unless mapped to this shape. Both frozen-vocab licenses (cc0-1.0,
    cc-by-nc-4.0) are in Zenodo's SPDX-derived rights vocabulary. If a future SPDX id is
    ever NOT in the vocabulary, the custom form ``{"title": {"en": name}, "link": uri}``
    validates without a vocabulary lookup (InvenioRDM accepts either ``id`` OR ``title``,
    never both).
    """
    return [{"id": dataset_license_spdx.lower()}]


def build_inveniordm_related(*, software_doi: str | None, paper_doi: str | None = None) -> list[dict]:
    """InvenioRDM-native ``related_identifiers`` — NOT the DataCite relatedIdentifiers.

    ``scheme`` is ``"doi"``, ``identifier`` is the bare DOI, and ``relation_type`` is the
    LOWERCASED controlled-vocabulary id (``{"id": "iscompiledby"}``). The DataCite
    relatedIdentifiers names (relatedIdentifier/relatedIdentifierType/relationType) are
    silently dropped by InvenioRDM. Reuses the existing single-tunable relation constants.
    """
    edges: list[dict] = []
    if software_doi:
        edges.append(
            {
                "identifier": software_doi,
                "scheme": "doi",
                "relation_type": {"id": _DATA_TO_SOFTWARE_RELATION.lower()},
            }
        )
    if paper_doi:
        edges.append(
            {
                "identifier": paper_doi,
                "scheme": "doi",
                "relation_type": {"id": "issupplementto"},
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
        raise PublishError(target=target, doi=doi, status=f"{step} failed: HTTP {resp.status_code} {resp.text[:1000]}")
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
    # Verified live (sandbox record 565599, 2026-07-15): the minted DOI lands at
    # top-level ``doi`` / ``metadata.doi`` while ``pids`` was null. Newer InvenioRDM
    # exposes pids.doi.identifier; try both so the extractor is version-robust.
    if payload.get("doi"):
        return payload["doi"]
    md = payload.get("metadata") or {}
    if md.get("doi"):
        return md["doi"]
    prereserve = md.get("prereserve_doi") or {}
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


# ---------------------------------------------------------------------------
# ADR-20 — the Option-B pre-upload size validator.
#
# There is NO queryable quota on either platform through hhemt's dependency surface
# (Zenodo's InvenioRDM records/drafts API exposes no quota field; hsclient v1.1.6 wraps no
# quota endpoint — verified 2026-07-13). So the limits are DOCUMENTED constants carrying an
# as_of date, and the live server rejection is caught and reframed as a backstop
# (_classify_storage_error). Report-and-warn, not a hard block: the operator decides.
# ---------------------------------------------------------------------------

#: Documented per-target deposit limits. Refresh the `as_of` date when re-verifying.
_TARGET_LIMITS: dict[str, dict] = {
    "zenodo": {
        "max_total_bytes": 50 * 1000**3,  # 50 GB per record
        "max_files": 100,
        "max_file_bytes": 50 * 1000**3,  # 50 GB per file
        "as_of": "2026-07-13",
        "doc_url": "https://help.zenodo.org/docs/deposit/manage-files/",
        "note": "50 GB/record, 100 files/record, 50 GB/file.",
    },
    "hydroshare": {
        "max_total_bytes": 20 * 1000**3,  # 20 GB DEFAULT account quota
        "max_files": None,
        "max_file_bytes": None,
        "as_of": "2026-07-13",
        "doc_url": "https://help.hydroshare.org/about-hydroshare/policies/quota/",
        "note": "20 GB DEFAULT account quota (per-user, not per-resource); may be raised on request.",
    },
}


def _path_size_bytes(path: Path) -> int:
    """Total size of a file, or the SUM of a directory tree.

    ``Path.stat().st_size`` on a DIRECTORY returns the inode size (~4 KB), so stat-ing the
    consolidated zarr store would report a 20 GB deposit as fitting comfortably inside a
    20 GB quota — a green check that certifies nothing. The tree sum is a conservative
    upper bound on the resulting ZIP_DEFLATED archive, which errs in the safe direction.
    """
    if path.is_dir():
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return path.stat().st_size


def validate_deposit_size(deposit: list[Path], target: str) -> dict:
    """Measure the deposit against ``target``'s DOCUMENTED limits (ADR-20, Option B).

    Always runs before a deposit. Returns a report; never raises and never blocks — the
    operator chose self-contained-by-default knowing the HydroShare tension, so this
    surfaces the tension and lets them decide.

    Returns:
        ``{fits, total_bytes, limit_bytes, overflow_bytes, n_files, as_of, doc_url,
        remediation}`` — ``remediation`` is the menu the operator acts on when it overflows.
    """
    limits = _TARGET_LIMITS[target]
    sizes = [_path_size_bytes(p) for p in deposit]
    total = sum(sizes)
    limit = limits["max_total_bytes"]
    overflow = max(0, total - limit)

    reasons: list[str] = []
    if overflow:
        reasons.append(
            f"total {total / 1000**3:.2f} GB exceeds the documented {limit / 1000**3:.0f} GB "
            f"limit by {overflow / 1000**3:.2f} GB"
        )
    if limits["max_files"] is not None and len(deposit) > limits["max_files"]:
        reasons.append(f"{len(deposit)} files exceeds the {limits['max_files']}-file limit")
    if limits["max_file_bytes"] is not None:
        for p, size in zip(deposit, sizes, strict=True):
            if size > limits["max_file_bytes"]:
                reasons.append(
                    f"'{p.name}' ({size / 1000**3:.2f} GB) exceeds the per-file "
                    f"{limits['max_file_bytes'] / 1000**3:.0f} GB limit"
                )

    remediation = (
        [
            f"exclude ~{overflow / 1000**3:.2f} GB of inputs via an exclude-config "
            f"(`hhemt bundle --list-excludable` shows what may be opted out)",
            f"request a quota increase from {target}",
            "switch to a target with more headroom",
        ]
        if reasons
        else []
    )

    return {
        "target": target,
        "fits": not reasons,
        "reasons": reasons,
        "total_bytes": total,
        "limit_bytes": limit,
        "overflow_bytes": overflow,
        "n_files": len(deposit),
        "as_of": limits["as_of"],
        "doc_url": limits["doc_url"],
        "remediation": remediation,
    }


def _classify_storage_error(body: str, size_bytes: int, limit_bytes: int) -> str | None:
    """Reframe a server rejection as a storage/quota failure when the body says so (G2).

    The documented-limit validator above is only as good as its constants. This is the
    attempt-and-surface backstop: if a constant is stale, or the account carries a
    non-default quota, the LIVE rejection is caught here and reframed with the emit-side
    size delta + the same remediation menu — instead of surfacing as an opaque HTTP error.

    Returns the reframed message, or None when the body carries no storage signal (in which
    case the caller re-raises verbatim — a non-storage failure must NOT be mislabelled).
    """
    signals = (
        "quota",
        "storage",
        "exceeds",
        "exceeded",
        "too large",
        "file size",
        "size limit",
        "insufficient space",
        "no space",
        "payload too large",
        "request entity too large",
    )
    low = (body or "").lower()
    if not any(s in low for s in signals):
        return None
    over = max(0, size_bytes - limit_bytes)
    return (
        f"deposit rejected for STORAGE/QUOTA reasons. The deposit measures "
        f"{size_bytes / 1000**3:.2f} GB against a documented limit of "
        f"{limit_bytes / 1000**3:.0f} GB"
        + (f" (over by {over / 1000**3:.2f} GB)" if over else "")
        + ". Remediation: exclude inputs via an exclude-config "
        "(`hhemt bundle --list-excludable`), request a quota increase, or switch target. "
        f"Server said: {body[:500]}"
    )


class _ZenodoTarget:
    """Programmatic reserve-DOI two-phase flow against the Zenodo/InvenioRDM REST API."""

    def publish(self, *, deposit, license_spdx, software_doi, analysis_dir, creators=None) -> dict:
        # create draft -> embed {InvenioRDM-native mandatory metadata + rights +
        #   related_identifiers} -> upload files -> publish (Zenodo MINTS the DataCite DOI
        #   here; NO reserve, NO self-supplied pids) -> read minted DOI from the publish
        #   response -> Phase-2 backfill reciprocal edge on the software record.
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

        # Phase 1b — embed the InvenioRDM-NATIVE record metadata. The four InvenioRDM-
        #   mandatory fields (title, creators, publication_date, resource_type) MUST all be
        #   present or publish 400s. rights + related_identifiers use InvenioRDM-native
        #   names (the DataCite rightsList/relatedIdentifiers names are silently dropped by
        #   the loader). NO pids block: Zenodo mints + registers the DataCite DOI on publish
        #   (self-supplying pids.doi.identifier=None is exactly what produced the 400).
        embed_metadata = {
            "title": _read_title_from_sidecar(analysis_dir),
            "creators": creators or _DEFAULT_CREATORS,
            "publication_date": date.today().isoformat(),
            "resource_type": {"id": "dataset"},
            "publisher": _PUBLISHER,
            "rights": build_inveniordm_rights(license_spdx),
            "related_identifiers": build_inveniordm_related(software_doi=software_doi),
        }
        _check(
            session.put(
                _ZENODO_DRAFT.format(base=base, recid=recid),
                json={"metadata": embed_metadata},
                timeout=_HTTP_TIMEOUT,
            ),
            target="zenodo",
            doi=None,
            step="embed metadata",
        )

        # Phase 1c — upload the deposit files (init -> PUT content -> commit, per file).
        # Zarr stores are directories; deposit each as a single .zip (Zenodo stores loose
        # files, not directory trees). Temp archives are removed after upload.
        # OE-1: NOT the default /tmp. _resolve_upload zips a multi-GB zarr store into
        # this staging dir; on an HPC login node /tmp is typically a small tmpfs, and a
        # 'No space left on device' here strands an in-flight draft.
        data_doi = None
        staging_parent = os.environ.get("TMPDIR") or str(analysis_dir)
        with tempfile.TemporaryDirectory(dir=staging_parent) as staging:
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

        # Phase 1d — publish. Zenodo mints + registers the DataCite DOI here.
        published = _check(
            session.post(_ZENODO_PUBLISH.format(base=base, recid=recid), timeout=_HTTP_TIMEOUT),
            target="zenodo",
            doi=None,
            step="publish",
        )
        data_doi = _extract_reserved_doi(published)
        if not data_doi:
            # DataCite registration can lag the publish action; the minted DOI is
            # authoritative on the published record. Re-read it (verified shape:
            # top-level ``doi`` / ``metadata.doi`` on the record GET).
            record = _check(
                session.get(_ZENODO_RECORD.format(base=base, recid=recid), timeout=_HTTP_TIMEOUT),
                target="zenodo",
                doi=None,
                step="read minted DOI",
            )
            data_doi = _extract_reserved_doi(record)

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
            "identifier": data_doi,
            "scheme": "doi",
            "relation_type": {"id": _SOFTWARE_TO_DATA_RELATION.lower()},
        }
        try:
            session.put(
                _ZENODO_DRAFT.format(base=base, recid=soft_recid),
                json={"metadata": {"related_identifiers": [edge]}},
                timeout=_HTTP_TIMEOUT,
            )
        except requests.RequestException:
            # Reciprocal edge is advisory; the primary deposit already succeeded.
            return


class _HydroShareTarget:
    """Two-invocation flow: hsclient publish-to-public -> manual web-UI DOI -> backfill."""

    def publish(self, *, deposit, license_spdx, software_doi, analysis_dir, creators=None) -> dict:
        # creators: accepted for call-site parity; an hsclient creator-metadata mapping is a
        # future enhancement (the HydroShare leg does not require it). Intentionally unused.
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
    creators: list[dict] | None = None,
    consolidated_zarr_relpath: str = "analysis_datatree.zarr",
    deposit_source: Literal["analysis_dir", "reprex_bundle"] | Path = "analysis_dir",
    container_defs: list[Path] | None = None,
) -> dict:
    """Deposit an analysis to a DOI-minting repository (C6, ADR-11).

    Reads the license baked into the emitted ro-crate sidecar; override_dataset_license
    ASSERTS the caller's expected value against the sidecar (raising PublishError on
    mismatch) rather than re-stamping the archived crate.

    Args:
        deposit_source: WHICH artifact to deposit (D6/R5 — the seam anticipated by the
            2026-07-08 decision `publish deposits the analysis_dir set not the render
            bundle`). This is the ONLY new parameter that decision permits, and it is
            default-preserving:

            - ``"analysis_dir"`` (DEFAULT) — the ADR-11 data-DOI deposit set (consolidated
              zarr + crate sidecar + configs). Byte-identical to the pre-seam behavior (R9).
            - ``"reprex_bundle"`` — the runnable reprex-bundle ZIP, emitted fresh and
              self-contained (no exclusions).
            - a ``Path`` — deposit exactly this artifact. This is how the
              ``publish_reprex_bundle`` facade passes a bundle it emitted with an
              exclude-config, without needing a second parameter here.

            NOTE the bundle is deposited as the ZIP that ``emit_bundle`` wrote, NOT via
            ``analysis.reprex_bundle()`` — that facade returns the EXTRACTED DIRECTORY, and
            ``_resolve_upload`` would re-zip a directory with ZIP_DEFLATED and
            ``relative_to(path.parent)`` arcnames, producing a non-deterministic archive
            that extracts to a nested root, so the ingest side's ``Bundle.from_directory``
            would raise ``FileNotFoundError`` and the round-trip (R8) could not pass.
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

    if isinstance(deposit_source, Path):
        deposit = [deposit_source]
    elif deposit_source == "reprex_bundle":
        from hhemt.bundle import emit_bundle

        deposit = [emit_bundle(analysis, container_defs=container_defs)]
    else:
        deposit = _deposit_set(analysis, consolidated_zarr_relpath)

    # ADR-20: the size validator ALWAYS runs before an irrevocable deposit. It reports and
    # warns; it never blocks (the operator elected self-contained-by-default knowing the
    # HydroShare tension). The measured total is threaded to _classify_storage_error so a
    # live rejection can be reframed with the real delta.
    size_report = validate_deposit_size(deposit, target)
    if not size_report["fits"]:
        warnings.warn(
            f"[{target}] deposit may exceed documented limits (as_of {size_report['as_of']}, "
            f"{size_report['doc_url']}): "
            + "; ".join(size_report["reasons"])
            + ". Remediation: "
            + " | ".join(size_report["remediation"])
            + ". Attempting the deposit anyway — a live rejection will be surfaced verbatim.",
            stacklevel=2,
        )

    try:
        return _TARGETS[target]().publish(
            deposit=deposit,
            license_spdx=license_spdx,
            software_doi=software_doi,
            analysis_dir=analysis_dir,
            creators=creators,
        )
    except PublishError as exc:
        # G2 attempt-and-surface backstop: if the documented constant was stale or the
        # account carries a non-default quota, the LIVE rejection lands here. Reframe it
        # with the emit-side delta; otherwise re-raise verbatim (a non-storage failure must
        # never be mislabelled as a quota problem).
        reframed = _classify_storage_error(
            str(exc.status if hasattr(exc, "status") else exc),
            size_report["total_bytes"],
            size_report["limit_bytes"],
        )
        if reframed is None:
            raise
        raise PublishError(target=target, doi=None, status=reframed) from exc
