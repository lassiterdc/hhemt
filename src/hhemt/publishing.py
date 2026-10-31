"""Deposit a consolidated analysis to a DOI-minting repository.

An analysis's results become citable only once they sit in a public repository under a
DOI. This module performs that deposit. It reads the already-emitted
ro-crate-metadata.json sidecar (the license, and the root `name` used as the title),
layers the target repository's own record metadata on top, and deposits to Zenodo
(Zenodo mints the DOI when the record is published; nothing is reserved beforehand) or
HydroShare (two-step flow; hsclient v1.1.6 has no programmatic DOI mint). NEVER re-runs
emit_provenance (crosswalk consumer, not a crate re-emitter).

Not established: the concrete Zenodo/InvenioRDM deposit REST endpoints + payload schemas
and the hsclient write-side calls were not checked against a pinned copy of either API.
The HTTP call bodies below are authored against the InvenioRDM REST reference
(``https://inveniordm.docs.cern.ch``, which backs zenodo.org) and the hsclient API; the
endpoint paths are named module constants (``_ZENODO_*``) so a durable-substrate
correction is a one-line change. Live deposits run ONLY under the ``publish_e2e`` marker
(``HHEMT_PUBLISH_E2E=1``); the mock-HTTP tests pin the two-phase control flow and payload
shape, so live-API field drift cannot silently break CI.
"""

from __future__ import annotations

import getpass
import json
import os
import sys
import tempfile
import warnings
import zipfile
from datetime import date
from pathlib import Path
from typing import Literal

import requests
import yaml

from hhemt.exceptions import PublishError
from hhemt.metadata import (
    _SPDX_LICENSE_TABLE,
    LicenseRecord,
    UnrecognizedCrateLicense,
    as_license_record,
    read_crate_license,
    resolve_dataset_license,
)

# The data->software relatedIdentifier relationType, held in one constant so that falling
# back to "IsDerivedFrom", if a target registry's UI hides IsCompiledBy, is a one-token change.
_DATA_TO_SOFTWARE_RELATION = "IsCompiledBy"
# The reciprocal edge written back onto the software record in the Phase-2 backfill.
_SOFTWARE_TO_DATA_RELATION = "IsSourceOf"

# Default dataset creator when the operator supplies none. An organizational creator is a
# valid DataCite/InvenioRDM creator and unblocks publish; the operator SHOULD pass real
# personal creators via publish_analysis(creators=...) for proper authorship credit. The
# crate sidecar carries no author/creator entity, so this is the only creators source
# unless the operator overrides it.
_DEFAULT_CREATORS: list[dict] = [{"person_or_org": {"type": "organizational", "name": "H&H Ensemble Modeling Toolkit"}}]

# DataCite-mandatory publisher (the entity issuing the resource). Required for DOI
# registration; Zenodo's own deposit UI defaults it to "Zenodo" for records it issues.
# This is the 6th and last DataCite-mandatory field (Identifier/Creators/Title/Publisher/
# PublicationYear/ResourceType) — the others land via the minted DOI + the metadata below.
_PUBLISHER = "Zenodo"

# InvenioRDM REST endpoint templates, named so that correcting an unverified path is a
# one-line change.
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


# The one message both refusal arms of `_read_license_from_sidecar` carry: a crate that
# declares no license and one that declares terms this toolkit cannot name are the same
# answer to "what license do these results carry", and a caller distinguishes them through
# `_sidecar_declares_license` instead.
_NO_SIDECAR_LICENSE = "no recognizable dataset license in ro-crate sidecar"


def _read_license_from_sidecar(analysis_dir: Path) -> str | LicenseRecord:
    """Return the license the emitted sidecar declares: an SPDX id, or a LicenseRecord.

    An SPDX id comes back for a table member, so a table member's read-back is unchanged
    by the details-file form. A license outside the table comes back as a
    :class:`~hhemt.metadata.LicenseRecord` built from the crate's OWN CreativeWork
    entity — the crate names its own terms, which is what makes such a license
    recognizable without a vocabulary the crate does not ship.

    Raises:
        PublishError: the root Dataset carries no license, or carries one whose `@id` is
            neither a table URI nor an entity the crate itself describes.
    """
    doc = json.loads((Path(analysis_dir) / "ro-crate-metadata.json").read_text())
    try:
        declared = read_crate_license(doc.get("@graph", []))
    except UnrecognizedCrateLicense as exc:
        raise PublishError(target="?", doi=None, status=_NO_SIDECAR_LICENSE) from exc
    if declared is None:
        raise PublishError(target="?", doi=None, status=_NO_SIDECAR_LICENSE)
    return declared


def _sidecar_declares_license(analysis_dir: Path) -> bool:
    """Return whether the emitted sidecar's root Dataset carries any `license` at all.

    Distinct from `_read_license_from_sidecar`, which raises on BOTH a missing and an
    unrecognized license. publish needs to tell the two apart: a missing license is the
    ordinary state of an analysis consolidated before a license was chosen, and is refused
    with the steps that fix it, while an unrecognized one is a malformed crate.
    """
    doc = json.loads((Path(analysis_dir) / "ro-crate-metadata.json").read_text())
    for e in doc.get("@graph", []):
        if e.get("@id") in ("./", ""):
            return e.get("license") is not None
    return False


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


def build_datacite_rightslist(dataset_license_spdx: str | LicenseRecord) -> list[dict]:
    """Return the DataCite ``rightsList`` entry for the dataset license.

    A table member yields the five-field SPDX entry. A details-file license cannot: it has
    no SPDX identifier and belongs to no identifier scheme, so it yields the license name
    and, when its file gives one, the address it is published at. Emitting a
    ``rightsIdentifier`` for it would assert an SPDX id the license does not have.
    """
    if isinstance(dataset_license_spdx, LicenseRecord):
        entry = {"rights": dataset_license_spdx.name}
        if dataset_license_spdx.link is not None:
            entry["rightsUri"] = dataset_license_spdx.link
        return [entry]
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


def build_inveniordm_rights(dataset_license_spdx: str | LicenseRecord) -> list[dict]:
    """InvenioRDM-native ``rights`` (license) entry — NOT the DataCite rightsList.

    InvenioRDM keys its license vocabulary on the LOWERCASED SPDX id; the DataCite
    rightsList names (rights/rightsUri/rightsIdentifier/rightsIdentifierScheme/schemeUri)
    are unknown to the InvenioRDM metadata loader and are silently dropped, so the license
    never lands unless mapped to this shape.

    Not established: that every SPDX id in ``_SPDX_LICENSE_TABLE`` is in Zenodo's
    SPDX-derived rights vocabulary. The table carries seven ids; an earlier revision of
    this docstring asserted that cc0-1.0 and cc-by-nc-4.0 are in it, and that assertion has
    not been re-verified (the tests here check only the lower-casing).

    A license OUTSIDE the table — a details-file record — is sent in the custom form
    ``{"title": {"en": name}, "link": uri}``, which validates without a vocabulary lookup
    (InvenioRDM accepts either ``id`` OR ``title``, never both). The ``link`` is included
    only when the record carries one. A bare string that is not a table key is treated as
    a NAME and takes the same custom form, so no input whatsoever is lower-cased into an
    id the vocabulary does not contain — the silent lower-casing of a free-form name is
    the behaviour this replaces.
    """
    if isinstance(dataset_license_spdx, str) and dataset_license_spdx in _SPDX_LICENSE_TABLE:
        return [{"id": dataset_license_spdx.lower()}]
    record = (
        dataset_license_spdx
        if isinstance(dataset_license_spdx, LicenseRecord)
        else LicenseRecord(name=dataset_license_spdx)
    )
    entry: dict = {"title": {"en": record.name}}
    if record.link is not None:
        entry["link"] = record.link
    return [entry]


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
    """The analysis_dir deposit set: consolidated zarr + sidecar + configs.

    Configs are NOT in the live analysis_dir by construction (only a bundle staging dir
    carries them); materialize the two fixed-name configs into the deposit exactly as
    analysis.eda() does. Takes the analysis (not just analysis_dir) so it can
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


def _stdin_is_interactive() -> bool:
    """True when a person can answer a prompt on this process's stdin.

    A prompt is only a question when someone is there to answer it. Asking unconditionally
    turns an unattended deposit into a hang: a SLURM step or a CI job blocks on a read that
    never returns, and the refusal an operator could act on becomes a stuck allocation with
    no message. So the interactive path is gated on the terminal, and everything else keeps
    the refusal.
    """
    try:
        return bool(sys.stdin) and sys.stdin.isatty()
    except (AttributeError, ValueError):  # a closed stream, or one with no isatty()
        return False


def _credential_is_secret(name: str) -> bool:
    """True when ``name``'s value must not be echoed to the terminal."""
    return any(word in name.upper() for word in ("TOKEN", "PASSWORD", "SECRET", "KEY"))


def _credential_label(name: str) -> str:
    """A prompt label for the credential variable ``name`` (``HHEMT_ZENODO_TOKEN`` -> ``Zenodo token``)."""
    words = name.removeprefix("HHEMT_").split("_")
    return " ".join([words[0].capitalize(), *(word.lower() for word in words[1:])]) if words else name


def _prompt_credential(name: str) -> str:
    """Ask the operator for one credential and return it. NOTHING is stored or cached.

    Deliberately NOT persisted, and the omission is the requirement: a prompt-obtained
    credential written to disk outlives the process that needed it, is swept into backups and
    shared-home snapshots, and turns one interactive answer into a standing secret that
    nobody owns and nobody rotates. The value exists as this call's return value and nowhere
    else, so a caller that needs it twice is asked twice. A secret-shaped name (a token, a
    password, a secret) is read with the echo off; a username is echoed so a typo is visible.
    """
    label = _credential_label(name)
    if _credential_is_secret(name):
        return getpass.getpass(f"{label} (input hidden): ")
    return input(f"{label}: ")


def _require_env(name: str, target: str) -> str:
    """The credential ``name``: its environment variable when set, else an interactive prompt.

    The environment variable stays the first source, so an unattended run keeps working by
    exporting it. When it is absent and a person is present, ask rather than refuse; when it
    is absent and stdin is not a terminal, refuse as before and name both routes, because a
    prompt nobody can answer is a hang rather than a question.
    """
    val = os.environ.get(name)
    if val:
        return val
    if _stdin_is_interactive():
        prompted = _prompt_credential(name)
        if prompted:
            return prompted
    raise PublishError(
        target=target,
        doi=None,
        status=(
            f"missing required credential {name}: export it, or run interactively and answer the prompt "
            "(nothing is stored, so it is asked for again next time)"
        ),
    )


def _check(resp, *, target: str, doi: str | None, step: str):
    """Raise PublishError on a non-2xx response; otherwise return the parsed JSON (or {})."""
    if resp.status_code >= 400:
        raise PublishError(target=target, doi=doi, status=f"{step} failed: HTTP {resp.status_code} {resp.text[:1000]}")
    try:
        return resp.json()
    except ValueError:
        return {}


def _extract_reserved_doi(payload: dict) -> str | None:
    """Read the minted DOI from an InvenioRDM publish response or record (fallback to prereserve_doi)."""
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
# Pre-upload size validator: measure the deposit against documented limits and warn.
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
    20 GB quota — a green check that certifies nothing. The tree sum only APPROXIMATES the
    resulting ZIP_DEFLATED archive and can UNDER-state it: ZIP adds per-file local and
    central-directory headers, and DEFLATE cannot shrink already-compressed zarr chunks.
    Measured on 200 files of 4096 random bytes, the archive came out 2.4-2.7% larger than
    the tree sum, so a deposit just under a limit can report ``fits`` and still be rejected
    live (``_classify_storage_error`` reframes that rejection).
    """
    if path.is_dir():
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return path.stat().st_size


def validate_deposit_size(deposit: list[Path], target: str) -> dict:
    """Measure the deposit against ``target``'s DOCUMENTED limits.

    Always runs before a deposit. Returns a report; never raises and never blocks — a
    deposit that carries its inputs can exceed HydroShare's 20 GB default account quota,
    and whether to proceed anyway is the operator's decision, so this surfaces the
    overflow and lets them decide.

    Returns:
        ``{target, fits, reasons, total_bytes, limit_bytes, overflow_bytes, n_files, as_of,
        doc_url, remediation}`` — ``reasons`` lists each limit exceeded (``fits`` is True
        iff it is empty, and ``publish_analysis`` joins it into its warning);
        ``remediation`` is the menu the operator acts on when it overflows.
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
    """Reframe a server rejection as a storage/quota failure when the body says so.

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
    """Two-phase flow against the Zenodo/InvenioRDM REST API; Zenodo mints the DOI on publish."""

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
            # license_spdx is an SPDX id for a table license and a LicenseRecord for a
            # details-file one; build_inveniordm_rights maps each to its own payload form.
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
        # NOT the default /tmp. _resolve_upload zips a multi-GB zarr store into
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
    """Two-step flow: one hsclient publish call (create, upload, publish-to-public), then a manual web-UI DOI mint.

    No backfill leg exists for this target: hsclient v1.1.6 cannot mint the DOI, and a second
    ``publish`` call creates a second resource, so the data-to-software relation must ride on
    the creating call's ``software_doi``.
    """

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
                "HydroShare resource is public. hsclient v1.1.6 cannot mint a DOI programmatically. "
                f"Open {record_url} and use 'Publish' in the web UI to mint the DOI. Do not re-run "
                "publish afterwards, because every HydroShare publish call creates a new resource and "
                "the data-to-software relation is recorded only from the software_doi passed on this call."
            ),
        }


def _hydroshare_rights(license_spdx: str | LicenseRecord) -> dict:
    """HydroShare ``metadata.rights`` for the dataset license: statement, and url when known.

    HydroShare takes free text, so a details-file license needs no translation beyond its
    own name. ``url`` is omitted rather than blanked when the record carries no link: an
    empty string would read to a depositor as an address the owner never published.
    """
    if isinstance(license_spdx, LicenseRecord):
        rights = {"statement": license_spdx.name}
        if license_spdx.link is not None:
            rights["url"] = license_spdx.link
        return rights
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
    """Deposit an analysis to a DOI-minting repository.

    Reads the license baked into the emitted ro-crate sidecar — as an SPDX id when the
    crate carries a table license, or as a details-file record when the crate names its own
    terms — and compares it against what the config declares, by record, so either form on
    either side is one comparison. override_dataset_license ASSERTS the caller's expected
    SPDX id against the sidecar (raising PublishError on mismatch) rather than re-stamping
    the archived crate.

    Args:
        deposit_source: WHICH artifact to deposit. The data DOI's deposit is the
            analysis_dir set, not the render bundle; this parameter is the single seam
            for depositing something else, and it is default-preserving, so a caller
            that passes nothing deposits exactly what publish deposited before it existed:

            - ``"analysis_dir"`` (DEFAULT) — the data-DOI deposit set (consolidated
              zarr + crate sidecar + configs). Byte-identical to the pre-seam behavior.
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
            would raise ``FileNotFoundError`` and the deposit-then-ingest round trip could
            not pass.
    """
    analysis_dir = analysis.analysis_paths.analysis_dir
    # The publish-time license requirement. A dataset license is optional everywhere else,
    # so an analysis consolidated without one reaches this point with no license in its
    # archived results. It is refused here, before any deposit, with the three steps that
    # fix it. publish does not write the license itself: the crate is baked at consolidation
    # and this function never re-stamps archived results.
    if not _sidecar_declares_license(analysis_dir):
        _shadowed = (
            f" The override_dataset_license={override_dataset_license!r} you passed was NOT"
            f" evaluated: an override asserts the license the results already carry, and they carry none."
            if override_dataset_license is not None
            else ""
        )
        raise PublishError(
            target=target,
            doi=None,
            status=(
                "this analysis has no dataset license in its archived results, and a deposit needs one. "
                "Set analysis_config.dataset_license to one of "
                f"{', '.join(_SPDX_LICENSE_TABLE)} (if it is not already set), then re-run "
                "reprocess(start_with='consolidate', regenerate_existing=True) so the license is written "
                "into the results, then publish again. regenerate_existing=True is REQUIRED: it defaults "
                "to False, which deletes no consolidated zarr, so consolidation early-returns and the "
                f"license is never written.{_shadowed}"
            ),
        )
    license_declared = _read_license_from_sidecar(analysis_dir)
    cfg_license = analysis.cfg_analysis.dataset_license
    # getattr rather than a bare read: this is handed config-like objects that may predate
    # the details-file field (test doubles built from a SimpleNamespace), and a license
    # check must not turn into an AttributeError on such an object.
    cfg_license_file = getattr(analysis.cfg_analysis, "dataset_license_file", None)
    # Compared BY RECORD, not by an id string: the config may declare either form and the
    # crate may carry either, and normalizing both sides is what lets one comparison cover
    # the four combinations instead of a branch per form.
    crate_record = as_license_record(license_declared)
    cfg_record = as_license_record(
        resolve_dataset_license(dataset_license=cfg_license, dataset_license_file=cfg_license_file)
    )
    # The crate's SPDX id, when its license is a table member. None for a details-file
    # license, which has no id any config field could be set to.
    crate_spdx = license_declared if isinstance(license_declared, str) else None
    if cfg_record != crate_record:
        _shadowed = (
            f" The override_dataset_license={override_dataset_license!r} you passed was NOT"
            f" evaluated: this check runs first, because an override is compared against the"
            f" crate and the crate is stale."
            if override_dataset_license is not None
            else ""
        )
        # Two remedies, the no-rebuild one first. Re-consolidating is the right fix only for a
        # deliberate license CHANGE: for a field that was cleared or edited by mistake it costs
        # a full re-consolidation, and with a cleared field it writes unlicensed results that
        # the refusal above then stops, so the message would otherwise send the user in a circle.
        if cfg_record is None:
            _cfg_state = "analysis_config.dataset_license is not set"
        elif cfg_license_file is not None:
            _cfg_state = f"analysis_config.dataset_license_file is {str(cfg_license_file)!r}"
        else:
            _cfg_state = f"analysis_config.dataset_license is {cfg_license!r}"
        _crate_state = (
            f"the license {crate_spdx!r}"
            if crate_spdx is not None
            else "the license they declare as "
            f"{crate_record.name!r}" + (f" ({crate_record.link})" if crate_record.link is not None else " (no link)")
        )
        # The no-rebuild remedy differs by the CRATE's license form, because the config side
        # has two fields and only one of them can express each form: an SPDX id is restored
        # by naming it in dataset_license, while a details-file license can only be restored
        # by pointing dataset_license_file at a file declaring the same terms.
        _keep = (
            f"To publish these results under {crate_spdx!r}, set analysis_config.dataset_license "
            f"to {crate_spdx!r} and publish again; no re-consolidation is needed. "
            if crate_spdx is not None
            else (
                "To publish these results under the license they carry, set "
                "analysis_config.dataset_license_file to a details file declaring name "
                f"{crate_record.name!r}"
                + (f" and link {crate_record.link!r}" if crate_record.link is not None else "")
                + " and publish again; no re-consolidation is needed. "
            )
        )
        raise PublishError(
            target=target,
            doi=None,
            status=(
                f"{_cfg_state}, but the archived results carry {_crate_state}, and "
                f"publish does not re-stamp them. {_keep}"
                f"To publish under a different license, set it and "
                f"re-emit the crate with reprocess(start_with='consolidate', regenerate_existing=True), "
                f"then publish again. regenerate_existing=True is REQUIRED: it defaults to False, which "
                f"deletes no consolidated zarr, so consolidation early-returns and the crate "
                f"is never re-emitted.{_shadowed}"
            ),
        )
    if override_dataset_license is not None and override_dataset_license != crate_spdx:
        raise PublishError(
            target=target,
            doi=None,
            status=(
                f"override_dataset_license={override_dataset_license!r} differs from the crate "
                f"license "
                + (f"{crate_spdx!r}" if crate_spdx is not None else f"{crate_record.name!r}, which has no SPDX id")
                + "; set analysis_config.dataset_license and re-emit the "
                "crate with reprocess(start_with='consolidate', regenerate_existing=True) "
                "first — publish does not re-stamp the archived zarr."
            ),
        )

    if isinstance(deposit_source, Path):
        deposit = [deposit_source]
    elif deposit_source == "reprex_bundle":
        from hhemt.bundle import emit_bundle

        deposit = [emit_bundle(analysis, container_defs=container_defs)]
    else:
        deposit = _deposit_set(analysis, consolidated_zarr_relpath)

    # The size validator ALWAYS runs before an irrevocable deposit. It reports and warns;
    # it never blocks, because a deposit that carries its inputs can exceed HydroShare's
    # 20 GB default quota and whether to proceed is the operator's call. The measured total
    # is threaded to _classify_storage_error so a live rejection can be reframed with the
    # real delta.
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
            license_spdx=license_declared,
            software_doi=software_doi,
            analysis_dir=analysis_dir,
            creators=creators,
        )
    except PublishError as exc:
        # Attempt-and-surface backstop: if the documented constant was stale or the
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
