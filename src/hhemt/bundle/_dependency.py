"""First-class cross-experiment dependency: declare + verify + reuse/auto-satisfy/halt (P2+V3).

Net-new for the resume-vs-clean intercomparison. A resume experiment declares a first-class
``depends_on`` on the clean experiment (an ``ExperimentDependency`` carrying an expected
provenance-identity tuple); ``resolve_dependency`` verifies a candidate clean bundle's OWN recorded
identity against that tuple and reuses / auto-satisfies / halts-loud. Reuses the existing per-bundle
identity reader (``_compatibility._read_jsonld_core``) and the ``n_resumes`` clean/resume role
classifier convention (``eda.compute_sensitivity``). Lives in ``bundle/`` (NOT ``config/``) so it is
not layout-relevant (the ``config/**/*.py`` glob).

Verification is by recorded provenance-IDENTITY (case_name + pinned TRITON sha + compute-config),
NOT mtime (timestamp-not-content) and NOT a full-byte hash (brittle). This is the reproducibility
contract: a single committed entry reproduces the intercomparison with no hand-run glue.
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from hhemt.exceptions import ConfigurationError


class ExperimentIdentity(BaseModel):
    """V3 provenance-identity tuple. None fields are NOT compared (forward-compatible: a declaration
    may pin only the fields it knows; ``compute_config_identity`` is optional in v1)."""

    case_name: str | None = None
    tritonswmm_sha: str | None = None
    compute_config_identity: str | None = None  # v1: optional; refinement axis (see module note)

    def matches(self, other: ExperimentIdentity) -> tuple[bool, list[str]]:
        """Field-by-field match over the fields THIS (the declared expected) identity pins.
        Returns ``(ok, mismatched_field_names)``. Unset (None) declared fields are not compared."""
        bad: list[str] = []
        for f in ("case_name", "tritonswmm_sha", "compute_config_identity"):
            exp = getattr(self, f)
            if exp is not None and exp != getattr(other, f):
                bad.append(f)
        return (not bad, bad)


class ExperimentDependency(BaseModel):
    """The unmistakeable, version-controlled declaration the reproducible driver reads."""

    dependency_experiment_id: str  # e.g. "synth_cc_clean"
    role: Literal["clean"] = "clean"  # the arm the dependency must BE
    expected_identity: ExperimentIdentity


def classify_bundle_role(bundle_root: Path) -> Literal["clean", "resume"]:
    """clean iff every ``scenario_status.csv`` ``n_resumes`` == 0; resume iff any > 0.

    Reuses the ``eda.compute_sensitivity`` ``n_resumes`` clean/resume classifier convention. Reads
    the BUNDLED ``scenario_status.csv`` (copied by ``_emit._copy_supporting_files``). Absence -> clean
    (no resume evidence)."""
    p = bundle_root / "scenario_status.csv"
    if not p.exists():
        return "clean"
    max_r = 0
    with p.open() as fh:
        for row in csv.DictReader(fh):
            try:
                max_r = max(max_r, int(float(row.get("n_resumes") or 0)))
            except (TypeError, ValueError):
                continue
    return "resume" if max_r > 0 else "clean"


def read_bundle_identity(bundle_root: Path) -> ExperimentIdentity:
    """Assemble the V3 identity tuple from an unpacked bundle.

    Reuses ``_compatibility._read_jsonld_core`` for ``case_name`` (bundle-readable only after the
    Phase-5 ``case.yaml`` copy) and reads ``TRITONSWMM_branch_key`` directly from the bundled
    ``cfg_system.yaml``. ``compute_config_identity`` is a v1-optional digest (None until refined)."""
    from hhemt.bundle._compatibility import _CFG_SYSTEM_FILENAME, _load_yaml, _read_jsonld_core

    core = _read_jsonld_core(bundle_root)
    sysd = _load_yaml(bundle_root / _CFG_SYSTEM_FILENAME)
    return ExperimentIdentity(
        case_name=core.get("case_name"),
        tritonswmm_sha=sysd.get("TRITONSWMM_branch_key"),
        compute_config_identity=None,  # v1: refinement axis (see module note)
    )


def resolve_dependency(
    dep: ExperimentDependency,
    *,
    search_roots: list[Path],
    auto_satisfy: Callable[[], Path] | None = None,
    emitted_command: str | None = None,
) -> Path:
    """Verify + reuse / auto-satisfy / halt-loud (the P2 guard). Returns the resolved dependency
    bundle ROOT path.

    Behavior:
      - a bundle in ``search_roots`` with ``role == dep.role`` AND matching ``expected_identity`` -> REUSE.
      - a role-matching bundle whose identity MISMATCHES -> HALT-LOUD (``ConfigurationError``, diff).
      - NO role-matching bundle present (absent):
          * if ``auto_satisfy`` given -> call it, then re-resolve once.
          * else -> HALT with the ``emitted_command`` (FQ2/AR2: emit-command, no improvised code)."""
    role_matches = [r for r in search_roots if classify_bundle_role(r) == dep.role]
    for r in sorted(role_matches):
        ok, _bad = dep.expected_identity.matches(read_bundle_identity(r))
        if ok:
            return r
    # no matching bundle -> distinguish "present-but-mismatched" from "absent"
    if role_matches:
        found = read_bundle_identity(sorted(role_matches)[0])
        raise ConfigurationError(
            field="depends_on",
            message=(
                f"Dependency '{dep.dependency_experiment_id}' present but identity MISMATCH: "
                f"expected {dep.expected_identity.model_dump(exclude_none=True)}, "
                f"found {found.model_dump(exclude_none=True)}."
            ),
            config_path=None,
        )
    if auto_satisfy is not None:
        produced = auto_satisfy()
        return resolve_dependency(
            dep,
            search_roots=[*search_roots, produced],
            auto_satisfy=None,
            emitted_command=emitted_command,
        )
    raise ConfigurationError(
        field="depends_on",
        message=(
            f"Dependency '{dep.dependency_experiment_id}' ({dep.role}) bundle is ABSENT under "
            f"{[str(p) for p in search_roots]}. Produce it with the committed command:\n"
            f"    {emitted_command or '<no emitted command supplied>'}"
        ),
        config_path=None,
    )
