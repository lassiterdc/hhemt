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
from hhemt.exceptions import StalePlotsError
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
def _describe_version() -> str:
    """PEP-440 local version derived from `git describe`, not from the static pin.

    `pyproject.toml` carries a STATIC `version = "0.1.0"` that has not moved in 241
    commits, so `importlib.metadata.version("hhemt")` is a constant that distinguishes
    nothing — it was identical on all four arms of the delivered generation. This
    derives `{tag}+{N}.g{sha}` from `git describe --tags --long`, which DOES
    distinguish generations and is deterministically recomputable from an archived
    sha alone (verified: `git describe --tags --long 01655abb60c2` -> `v0.1.0-241-g01655ab`).

    Falls back to the installed metadata version when git is unavailable — a wheel
    install is the intended fallback case, not a failure. NEVER raises: a provenance
    minter that can abort a 3-hour consolidation is worse than one that degrades.
    """
    import subprocess

    from hhemt.bundle._emit import _toolkit_source_dir

    try:
        out = subprocess.run(
            ["git", "describe", "--tags", "--long", "--abbrev=12"],
            cwd=_toolkit_source_dir(),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        tag, n, gsha = out.rsplit("-", 2)
        return f"{tag.lstrip('v')}+{n}.{gsha}"
    except Exception:
        try:
            return importlib.metadata.version("hhemt")
        except Exception:
            return "0+unknown"


def _is_dirty() -> bool:
    """True when the toolkit checkout has uncommitted changes at mint time.

    This closes a hole the codebase currently states but does not enforce:
    `process_simulation._resolve_producing_stamp`'s docstring claims the sha is
    `"unknown"` when the working tree is dirty, but `_get_toolkit_git_sha` only
    degrades on CalledProcessError/FileNotFoundError — `git rev-parse HEAD` succeeds
    on a dirty tree, so a run with uncommitted edits stamps a CLEAN sha and reads
    authoritative. The only dirty check in the toolkit today is `_carry_source_tree`'s
    emit-time warning, which fires long after capture. False on any error: an
    undeterminable tree must not be asserted dirty.
    """
    import subprocess

    from hhemt.bundle._emit import _toolkit_source_dir

    try:
        return bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=_toolkit_source_dir(),
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except Exception:
        return False


def producing_stamp() -> dict[str, str]:
    """THE single minter for every stage's hhemt version-provenance stamp (ADR-15 widening).

    Every stage that writes an artifact calls THIS, at write time, from the running
    process. Three fields, one shape, six carriers. The minting rule this enforces:
    a stamp records the code that ACTUALLY ran the stage, so it may never be resolved
    from a module-level constant, a config field, or a value carried over from an
    earlier stage. EW-3 is the counter-example — a field stamped from a constant that
    had no value yet at run time, which read authoritative and was wrong.

    Corollary the CALL SITES must honor and this function cannot: if the stage did not
    execute, do not call this. An unconditional stamp cannot distinguish
    "re-ran at this sha" from "skipped, carrying an older artifact".
    """
    return {
        "hhemt_sha": _toolkit_git_sha(),
        "hhemt_version": _describe_version(),
        "hhemt_dirty": "true" if _is_dirty() else "false",
    }


def collect_plot_stamps(analysis_dir: "Path") -> tuple[set[tuple[str, str]], int]:
    """Distinct (sha, dirty) keys across every figure sidecar, and the sidecar count.

    Reads ALL sidecars rather than one, because uniform-within-a-render is a measured
    property and not a guaranteed one: a targeted re-render refreshes some sidecars and
    leaves others, which yields two distinct keys and is its own staleness class
    (figures inconsistent with EACH OTHER, remedied by a full stage:render force rather
    than by re-rendering the report). An unreadable or unstamped sidecar contributes no
    key and is counted, so `keys == set()` with `n > 0` means "figures present, none
    stamped" -- absent, which the caller treats as never-equal.
    """
    import json
    from pathlib import Path as _Path

    keys: set[tuple[str, str]] = set()
    n = 0
    plots = _Path(analysis_dir) / "plots"
    if not plots.exists():
        return keys, 0
    for sidecar in sorted(plots.rglob("*.manifest.json")):
        n += 1
        try:
            payload = json.loads(sidecar.read_text())
        except Exception:
            continue
        sha = str(payload.get("hhemt_sha") or "").strip()
        if sha:
            keys.add((sha, str(payload.get("hhemt_dirty") or "unknown")))
    return keys, n


def assert_plots_match_running_build(analysis_dir: "Path", *, declare_stale_plots: bool = False):
    """Refuse a render whose figures were not produced by the build now running.

    The report-side operand is `producing_stamp()` computed IN-PROCESS, never a
    persisted report stamp: reading `report_manifest.json` would compare a render
    against itself (the writer runs at the tail of the same call), would refuse every
    first render for want of a prior stamp, and would depend on a writer that fails
    non-fatally. Equality on the (sha, dirty) tuple, never an ordering.
    """
    keys, n = collect_plot_stamps(analysis_dir)
    running = producing_stamp()
    mine = (str(running.get("hhemt_sha") or "").strip(), str(running.get("hhemt_dirty") or "unknown"))
    if len(keys) > 1:
        why = (
            f"the {n} figure sidecar(s) carry {len(keys)} DIFFERENT build stamps "
            f"({sorted(s[:12] for s, _ in keys)}) -- the figures are inconsistent with each "
            "other, which a partial re-render produces. Force a full re-render with "
            'force_rerun {"subject": "all", "stage": "render"}'
        )
    elif not keys:
        why = (
            f"no figure carries a build stamp ({n} sidecar(s) found) -- the figures predate "
            "the provenance capture site and cannot be shown to match this build"
        )
    elif not mine[0]:
        why = "the running build could not be identified (no toolkit sha resolvable)"
    elif keys == {mine}:
        return None
    else:
        (psha, pdirty) = next(iter(keys))
        why = (
            f"figures built at {psha[:12]} (dirty={pdirty}), report rendering at "
            f"{mine[0][:12]} (dirty={mine[1]})"
        )
    msg = (
        f"Report/figure build mismatch: {why}. The figures may not reflect the renderer "
        "code now producing this report. Re-render with force_rerun "
        '{"subject": "all", "stage": "render"}, or pass --declare-stale-plots to proceed '
        "and record the mismatch."
    )
    if declare_stale_plots:
        print(f"[render_report] DECLARED STALE PLOTS: {msg}", flush=True)
        return msg
    raise StalePlotsError(msg)


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
    emitted_vars: set[str] | None = None,
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
        emitted_vars=emitted_vars,
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
