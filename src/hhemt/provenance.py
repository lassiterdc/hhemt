"""Dual-write provenance emitter (C2) for the reproducibility-system (ADR-7).

Reads log.py READ-ONLY at consolidation and renders a per-run PROV CreateAction +
per-output wasGeneratedBy graph. NEVER mutates log.py (the _already_written
completion-gate is load-bearing across the DAG — Gotchas 28/34/40).
"""

from __future__ import annotations

import importlib.metadata
import socket
from types import SimpleNamespace

from rocrate.model.contextentity import ContextEntity

from hhemt.constants import LAYOUT_VERSION
from hhemt.metadata import (
    build_analysis_crate,
    canonical_jsonld,
    canonical_jsonld_from_doc,
    partition_core_vs_sidecar,
)


def _default_code_repository() -> str:
    """Canonical public repo URL for the RO-Crate codeRepository, sourced from the
    INSTALLED package metadata (pyproject.toml [project.urls].homepage = "https://github.com/lassiterdc/hhemt")
    — single source of truth, no duplicated literal. RAISES (never silently falls back to
    a stale/guessed literal) when the metadata homepage is absent: a missing homepage is a
    genuinely broken install that MUST surface loudly, not be papered into durable archival
    provenance. CI pins the expected value (test_provenance.py::test_default_code_repository_pins_homepage)
    so a pyproject regression is caught at CI, before deployment, not at hour-3 of an HPC consolidation."""
    from hhemt.exceptions import ProcessingError  # lazy: matches this module's import style; avoids any cycle

    for entry in importlib.metadata.metadata("hhemt").get_all("Project-URL") or []:
        label, _sep, url = entry.partition(",")
        if label.strip().lower() == "homepage":
            return url.strip()
    raise ProcessingError(
        operation="provenance_code_repository",
        filepath=None,
        reason="hhemt package metadata exposes no 'homepage' Project-URL; cannot resolve the RO-Crate "
        "codeRepository. Fix pyproject.toml [project.urls].homepage. (No silent fallback by design — "
        "a guessed/stale URL must never be baked into durable provenance.)",
    )


def _toolkit_git_sha() -> str:
    from hhemt.bundle._emit import _get_toolkit_git_sha

    return _get_toolkit_git_sha(strict=False)


def _resolve_case_manifest(analysis):
    """CaseManifest for input File parts, or a minimal stand-in (empty manifest).

    The analysis object does not currently carry a CaseManifest; a thin accessor
    (`analysis._case_manifest`) is the narrow wiring point. Until it lands, degrade
    to a minimal descriptor with zero input parts — the crate stays valid.
    """
    cm = getattr(analysis, "_case_manifest", None)
    if cm is not None:
        return cm
    return SimpleNamespace(
        case_name=str(analysis.cfg_analysis.analysis_id),
        description="",
        manifest={},
    )


def _input_parts_from_case(cfg_case) -> list[dict]:
    return [
        {"@id": fname, "sha256": hexsha, "contentSize": None, "encodingFormat": None}
        for fname, hexsha in getattr(cfg_case, "manifest", {}).items()
    ]


def _iter_run_units(analysis):
    """Yield (sa_id, event_id, model_type) per real invocation unit.

    Regular analysis: sa_id is "" ; one unit per (event_iloc, enabled model_type).
    """
    sa_id = str(getattr(analysis, "sa_id", "") or "")
    enabled = analysis._get_enabled_model_types()  # encapsulates self._system.cfg_system.toggle_* (analysis.py:1431)
    for event_iloc in analysis.df_sims.index:
        for model_type in enabled:
            yield (sa_id, str(event_iloc), model_type)


def _output_ids(analysis, sa_id, event_id, model_type) -> list[str]:
    """Per-output @ids for one run unit, derived from the per-model processing_log."""
    from hhemt.scenario import TRITONSWMM_scenario

    scen = TRITONSWMM_scenario(int(event_id), analysis)
    mlog = scen.get_log(model_type)
    outs = getattr(mlog.processing_log, "outputs", {}) or {}
    # NOTE: the canonical event-id slug lives on the scenario itself (`scen.event_id`,
    # scenario.py:60 — `self.event_id = self.sim_id_str`), NOT on `scen.scen_paths`
    # (ScenarioPaths, paths.py:83, carries no event_id). Verified at Phase-2 preflight.
    return [f"sims/{scen.event_id}/processed/{name}" for name in sorted(outs)]


def _agent_id(node: str | None) -> str:
    return f"#agent-{node or socket.gethostname()}"


def emit_provenance(
    analysis,
    *,
    sif_spec=None,
    code_repository: str | None = None,
    consolidated_zarr_relpath: str = "analysis_datatree.zarr",
    sub_dataset_relpaths=None,
    with_run_units: bool = True,
) -> tuple[str, str]:
    """Build the analysis crate + render the per-run CreateAction graph from log.py.

    Returns (embedded_core_jsonld, sidecar_jsonld). The caller writes the first to
    tree.attrs["ro_crate_metadata"] (via cf_conventions.apply_provenance_core) and the
    second to analysis_dir/ro-crate-metadata.json (via metadata.write_rocrate_sidecar).
    log.py is the SOURCE-OF-RECORD; this reads it and NEVER writes it.
    """
    code_repository = (
        code_repository or _default_code_repository()
    )  # resolve at call time (throw-on-absence; no import-time read)
    cfg_case = _resolve_case_manifest(analysis)
    input_parts = _input_parts_from_case(cfg_case)
    alog = analysis.log  # TRITONSWMM_analysis_log (read-only)
    crate = build_analysis_crate(
        analysis_id=str(analysis.cfg_analysis.analysis_id),
        system_id=None,
        layout_version=LAYOUT_VERSION,
        toolkit_git_sha=_toolkit_git_sha(),
        code_repository=code_repository,
        cfg_case=cfg_case,
        dataset_license=str(analysis.cfg_analysis.dataset_license),
        sif_spec=sif_spec,
        consolidated_zarr_relpath=consolidated_zarr_relpath,
        input_parts=input_parts,
        sub_dataset_relpaths=sub_dataset_relpaths,
    )

    for sa_id, event_id, model_type in _iter_run_units(analysis) if with_run_units else ():
        out_ids = _output_ids(analysis, sa_id, event_id, model_type)
        action = crate.add(
            ContextEntity(
                crate,
                f"#run-{sa_id}-{event_id}-{model_type}",
                properties={
                    "@type": "CreateAction",
                    "name": f"TRITON-SWMM run {event_id} ({model_type})",
                    "instrument": [{"@id": "#hhemt-app"}] + ([{"@id": sif_spec["@id"]}] if sif_spec else []),
                    "object": [{"@id": p["@id"]} for p in input_parts],
                    "result": [{"@id": oid} for oid in out_ids],
                    # VOLATILE — present only in the sidecar full graph; stripped from the core by partition:
                    "startTime": alog.workflow_submission_time.get(),
                    "agent": {"@id": _agent_id(alog.workflow_submission_node.get())},
                },
            )
        )
        for oid in out_ids:
            node = crate.dereference(oid)  # ro-crate-py returns None (NOT KeyError) for an absent @id
            if node is not None:
                node["wasGeneratedBy"] = {"@id": action.id}
            # else: output not yet a graph node (no summary) — skip the inverse edge

    full_doc = crate.metadata.generate()
    sidecar = canonical_jsonld(crate)  # full graph (incl. volatile)
    core = canonical_jsonld_from_doc(partition_core_vs_sidecar(full_doc))  # deterministic subset
    return core, sidecar
